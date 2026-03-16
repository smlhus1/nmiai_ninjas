"""
Coordinator: the central orchestrator that ties everything together.

This is the single entry point for the game loop. Each round:
1. Parse game state -> immutable GameState
2. Build WorldModel (enriched view)
3. TaskPlanner assigns/updates tasks
4. ActionResolver converts tasks to actions
5. Return JSON response

The Coordinator owns all persistent state between rounds:
- Bot assignments (task + cached path)
- PathEngine (grid cache + BFS distance cache)
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from bot.config import CoordinatorConfig
from bot.models import Action, Bot, GameState, BotCommand, Pos, apply_move
from bot.engine.pathfinding import PathEngine
from bot.engine.world_model import WorldModel
from bot.strategy.planner import TaskPlanner
from bot.strategy.action_resolver import ActionResolver
from bot.strategy.task import BotAssignment, TaskType
from bot.recon.logger import GameLogger

logger = logging.getLogger(__name__)

_MOVE_ACTIONS = frozenset({
    Action.MOVE_UP, Action.MOVE_DOWN,
    Action.MOVE_LEFT, Action.MOVE_RIGHT,
})


class Coordinator:
    """
    Main bot coordinator. Create once, call on_game_state() each round.
    """

    def __init__(self, config: CoordinatorConfig | None = None) -> None:
        self._config = config
        self._path_engine = PathEngine()
        self._planner = TaskPlanner()
        self._resolver = ActionResolver(self._path_engine)
        self._assignments: dict[int, BotAssignment] = {}
        self._round = 0
        self._last_commands: dict[int, BotCommand] = {}
        self._last_bot_positions: dict[int, Pos] = {}
        self._round_offset = False
        self._offset_checked = False
        # Collision tracking: count moves that server rejected
        self._collision_log: list[dict] = []  # [{round, bot_id, action, expected, actual}]
        self._game_logger = GameLogger()
        self._shelf_positions: frozenset[Pos] = frozenset()
        self._replay_attempted = False
        self._stuck_rounds: dict[int, int] = {}  # bot_id -> consecutive rounds without moving
        self._position_history: dict[int, list[Pos]] = {}  # bot_id -> last N positions
        # Time-space planner (lazy init for multi-bot)
        self._time_space_planner: Any = None
        # Guidance graph (lazy init when guidance_enabled)
        self._guidance_graph: Any = None
        # Visualization broadcaster (optional, started via start_viz())
        self._viz: Any = None

    def start_viz(self, port: int = 8765) -> None:
        """Start visualization WebSocket broadcaster."""
        from bot.viz_broadcaster import VizBroadcaster
        self._viz = VizBroadcaster(port=port)
        self._viz.start()

    def on_game_state(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Main entry point. Takes raw game state dict, returns action response dict.
        This is the only method the WebSocket client needs to call.
        """
        t_start = time.perf_counter()

        # 1. Parse (real server state)
        real_state = GameState.from_dict(raw)
        self._round = real_state.round

        # 1.5. Track collision mismatches (expected vs actual position)
        if self._last_commands and real_state.round >= 1:
            for bot in real_state.bots:
                cmd = self._last_commands.get(bot.id)
                old_pos = self._last_bot_positions.get(bot.id)
                if cmd is None or old_pos is None:
                    continue
                if cmd.action in _MOVE_ACTIONS:
                    expected = apply_move(old_pos, cmd.action)
                    if bot.position != expected and bot.position == old_pos:
                        # Bot tried to move but stayed — server rejected the move
                        self._collision_log.append({
                            "round": real_state.round,
                            "bot_id": bot.id,
                            "action": cmd.action.value,
                            "from": old_pos,
                            "expected": expected,
                            "actual": bot.position,
                        })
                        logger.warning(
                            "COLLISION R%d: B%d sent %s from %s, expected %s, got %s (stayed)",
                            real_state.round, bot.id, cmd.action.value,
                            old_pos, expected, bot.position,
                        )

        # 2. Detect round offset (our actions applied 1 round late)
        if real_state.round >= 1 and not self._offset_checked:
            self._detect_offset(real_state)

        # 3. Compensate: predict where bots WILL BE when our action is applied
        state = self._compensate_offset(real_state) if self._round_offset else real_state

        # 4. Initialize assignments for new bots
        for bot in state.bots:
            if bot.id not in self._assignments:
                self._assignments[bot.id] = BotAssignment(bot_id=bot.id)

        active_ids = {b.id for b in state.bots}
        self._assignments = {
            k: v for k, v in self._assignments.items() if k in active_ids
        }

        # 5. Set up pathfinding — merge shelf positions into grid walls
        if not self._shelf_positions and state.items:
            self._shelf_positions = frozenset(item.position for item in state.items)
            # Enable one-way aisles for multi-bot (eliminates head-on collisions)
            if len(state.bots) >= 2:
                self._path_engine.enable_one_way(True)
        if self._shelf_positions:
            merged_walls = state.grid.walls | self._shelf_positions
            from bot.models import Grid
            merged_grid = Grid(state.grid.width, state.grid.height, merged_walls)
            self._path_engine.set_grid(merged_grid, drop_off=state.drop_off)
        else:
            self._path_engine.set_grid(state.grid, drop_off=state.drop_off)
        self._path_engine.new_round()

        # 5.4. Guidance graph — congestion-aware routing for PIBT
        if self._config and getattr(self._config, 'guidance_enabled', False):
            if self._guidance_graph is None:
                from bot.engine.guidance import GuidanceGraph
                self._guidance_graph = GuidanceGraph(
                    self._path_engine._grid,
                    one_way=self._path_engine._one_way,
                    alpha=getattr(self._config, 'guidance_alpha', 2.0),
                    beta=getattr(self._config, 'guidance_beta', 3.0),
                    decay=getattr(self._config, 'guidance_decay', 0.7),
                    update_interval=getattr(self._config, 'guidance_update_interval', 5),
                )
                self._resolver._guidance = self._guidance_graph
            # Update traffic data
            bot_positions = {b.id: b.position for b in state.bots}
            self._guidance_graph.on_round(bot_positions, state.round)

        # 5.5. Viz metadata (first round)
        if self._viz and state.round == 0:
            self._viz.set_metadata(
                shelf_positions=self._shelf_positions,
                one_way=self._path_engine._one_way,
            )

        # 5.5b. Recon logging
        self._game_logger.on_round(state, self._shelf_positions)

        # 5.6. Auto-detect difficulty + replay plan (first round only)
        if state.round == 0 and not self._replay_attempted:
            self._replay_attempted = True
            if self._config is None:
                self._config = CoordinatorConfig.for_difficulty(len(state.bots))
                n = len(state.bots)
                diff = "easy" if n == 1 else ("medium" if n <= 3 else ("nightmare" if n >= 20 else ("expert" if n >= 10 else "hard")))
                logger.info("Auto-detected %d bot(s) -> %s config (replay=%s)",
                            len(state.bots), diff, self._config.replay_enabled)
            if self._config.replay_enabled:
                fp = self._game_logger.fingerprint
                logs_dir = getattr(self, "_logs_dir", Path("logs"))
                plan_path = logs_dir / f"{fp}_{date.today()}_plan.json"
                if plan_path.exists():
                    try:
                        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
                        from bot.recon.replay import ReplayPlanner
                        self._planner = ReplayPlanner(plan_data, TaskPlanner())
                        logger.info("REPLAY MODE: loaded plan from %s", plan_path)
                    except Exception:
                        logger.exception("Failed to load replay plan, staying reactive")
            else:
                logger.info("Replay disabled for this difficulty, staying reactive")

        # 5.7. Swap planner to v2 if configured (first round only)
        if (state.round == 0 and self._config
                and getattr(self._config, 'planner_version', 1) == 2
                and not isinstance(self._planner, TaskPlanner.__class__)):
            from bot.strategy.v2.planner import V2TaskPlanner
            if not isinstance(self._planner, V2TaskPlanner):
                self._planner = V2TaskPlanner()
                logger.info("V2 planner activated")

        # 6. Build world model
        max_ri = self._config.max_route_items if self._config else 3
        world = WorldModel(state, self._path_engine, max_route_items=max_ri)

        # 6.5. Stuck-bot detection: clear assignments for bots that haven't moved 5+ rounds
        self._detect_stuck_bots(state, world)

        # 6.6. Clear navigation overrides — they are re-set each round by _schedule_dropoff
        for a in self._assignments.values():
            a.navigation_override = None

        # 6.7. Pass config to planner (for deliver_detour_threshold etc.)
        planner = getattr(self._planner, "_reactive", self._planner)
        planner._config = self._config
        # Pass shelf_map for genome shelf preference (evolutionary search)
        if not hasattr(self._config, '_shelf_map') and hasattr(self, '_game_logger'):
            self._config._shelf_map = getattr(self._game_logger, '_shelf_map', {})
        # Inject shelf preference from config into planner
        if self._config and getattr(self._config, 'shelf_preference', None):
            planner._shelf_preference = self._config.shelf_preference
        # Inject order sequence for future-order pipeline staging
        if (self._config and getattr(self._config, 'order_sequence', None)
                and hasattr(planner, 'set_future_orders') and not planner._future_orders):
            planner.set_future_orders(self._config.order_sequence)

        # 7. Plan tasks
        self._assignments = self._planner.plan(world, self._assignments)

        # 7.5 Drop-off scheduling
        self._schedule_dropoff(world)

        # 7.6 Hivemind: clear path for active deliverer
        if len(state.bots) >= 10:
            self._hivemind_clear_path(world)

        # 7.7 Time-space planning (multi-bot only)
        planned_paths = None
        if len(state.bots) >= 3:
            if self._time_space_planner is None:
                from bot.engine.time_space_planner import TimeSpacePlanner
                horizon = self._config.tsp_horizon if self._config else 30
                budget = self._config.tsp_max_planning_ms if self._config else 500
                self._time_space_planner = TimeSpacePlanner(
                    self._path_engine, horizon=horizon, max_planning_ms=budget,
                )
            planned_paths = self._time_space_planner.plan(
                world, self._assignments, state.round,
            )

        # 8. Resolve to actions
        commands = self._resolver.resolve(state, self._assignments, planned_paths)

        # 8.5. Broadcast to viz
        if self._viz:
            self._viz.send_state(state, self._assignments)

        # 9. Build response
        response = {"actions": [cmd.to_dict() for cmd in commands]}

        # Track positions for stuck detection next round
        self._last_bot_positions = {b.id: b.position for b in real_state.bots}

        t_elapsed = time.perf_counter() - t_start

        # --- Logging ---
        if real_state.round == 0:
            for item in real_state.items:
                logger.info("MAP: item %s (%s) @ %s", item.id, item.type, item.position)
            logger.info("MAP: drop_off @ %s | zones=%s | grid %dx%d",
                        real_state.drop_off, real_state.drop_off_zones,
                        real_state.grid.width, real_state.grid.height)

        if real_state.round % 20 == 0:
            for order in real_state.orders:
                logger.info(
                    "Round %d | Order %s [%s]: required=%s delivered=%s remaining=%s",
                    real_state.round, order.id, order.status.value,
                    list(order.items_required), list(order.items_delivered),
                    list(order.items_remaining),
                )
            type_counts = Counter(item.type for item in real_state.items)
            logger.info("Round %d | Items on map: %s (total=%d)",
                        real_state.round, dict(type_counts), len(real_state.items))

        for cmd in commands:
            real_bot = real_state.get_bot(cmd.bot_id)
            plan_bot = state.get_bot(cmd.bot_id) if self._round_offset else real_bot
            a = self._assignments.get(cmd.bot_id)
            inv_str = list(real_bot.inventory) if real_bot else "?"
            real_pos = real_bot.position if real_bot else "?"
            plan_pos = plan_bot.position if plan_bot else "?"
            target = a.effective_target if a else None
            offset_tag = f" plan@{plan_pos}" if self._round_offset and real_pos != plan_pos else ""
            logger.info(
                "R%d B%d@%s%s inv=%s -> %s tgt=%s",
                real_state.round, cmd.bot_id, real_pos, offset_tag,
                inv_str, cmd.action.value, target,
            )

        if t_elapsed > 1.0:
            logger.warning("Round %d took %.1fms!", real_state.round, t_elapsed * 1000)

        # Store for next round's offset detection
        self._last_commands = {cmd.bot_id: cmd for cmd in commands}
        self._last_bot_positions = {b.id: b.position for b in real_state.bots}

        return response

    def _detect_offset(self, state: GameState) -> None:
        """Check if our previous action was applied or delayed by 1 round."""
        for bot in state.bots:
            if bot.id not in self._last_commands or bot.id not in self._last_bot_positions:
                continue
            old_pos = self._last_bot_positions[bot.id]
            old_cmd = self._last_commands[bot.id]
            if old_cmd.action not in _MOVE_ACTIONS:
                continue
            expected = apply_move(old_pos, old_cmd.action)
            if not state.grid.is_walkable(expected):
                continue
            if bot.position == old_pos and expected != old_pos:
                # Check if another bot occupied the expected position last round
                # (collision, not offset)
                was_blocked = any(
                    pos == expected for bid, pos in self._last_bot_positions.items()
                    if bid != bot.id
                )
                if was_blocked:
                    continue
                self._round_offset = True
                self._offset_checked = True
                logger.warning(
                    "OFFSET DETECTED R%d: sent %s from %s, expected %s, actual %s",
                    state.round, old_cmd.action.value, old_pos, expected, bot.position,
                )
                return
            if bot.position == expected:
                self._offset_checked = True
                logger.info("No offset: action applied normally at R%d", state.round)
                return

    def _compensate_offset(self, state: GameState) -> GameState:
        """
        Build adjusted state with predicted bot positions.
        With 1-round offset, our action A(N) is applied at round N+1.
        At that point, bot is at apply(A(N-1), P(N)).
        """
        adjusted_bots = []
        for bot in state.bots:
            cmd = self._last_commands.get(bot.id)
            if cmd:
                predicted = apply_move(bot.position, cmd.action)
                if state.grid.is_walkable(predicted):
                    adjusted_bots.append(Bot(
                        id=bot.id, position=predicted, inventory=bot.inventory,
                    ))
                else:
                    adjusted_bots.append(bot)
            else:
                adjusted_bots.append(bot)
        return GameState(
            round=state.round,
            max_rounds=state.max_rounds,
            grid=state.grid,
            bots=tuple(adjusted_bots),
            items=state.items,
            orders=state.orders,
            drop_off=state.drop_off,
            drop_off_zones=state.drop_off_zones,
            score=state.score,
        )

    def _schedule_dropoff(self, world: WorldModel) -> None:
        """Token-based: one bot at a time may approach drop-off. Others stage for auto-delivery."""
        self._evict_dropoff_blockers(world)

        active = world.state.active_orders
        remaining_types = set(active[0].items_remaining) if active else set()
        need = Counter(active[0].items_remaining) if active else Counter()

        deliverers: list[tuple[int, int, int, bool]] = []  # (distance, bot_id, matching_count, completes_order)
        bots_by_id: dict[int, Bot] = {}
        for bot_id, assignment in self._assignments.items():
            if assignment.task and assignment.task.task_type == TaskType.DELIVER:
                bot = world.state.get_bot(bot_id)
                if bot:
                    has_match = any(inv in remaining_types for inv in bot.inventory)
                    if has_match:
                        d = world.distance(bot.position, world.nearest_drop_off(bot.position, bot.id))
                        matching_count = sum(1 for inv in bot.inventory if inv in remaining_types)
                        bots_by_id[bot_id] = bot
                        deliverers.append((d, bot_id, matching_count, False))  # completes set below

        if not deliverers:
            return

        # Compute total matching inventory across deliverers; then mark completer-bots
        have: Counter[str] = Counter()
        for bot_id, _, _, _ in deliverers:
            bot = bots_by_id.get(bot_id)
            if bot:
                for inv in bot.inventory:
                    if inv in need:
                        have[inv] += 1
        updated: list[tuple[int, int, int, bool]] = []
        for (d, bot_id, matching_count, _) in deliverers:
            bot = bots_by_id.get(bot_id)
            completes = False
            if bot and need:
                for inv in bot.inventory:
                    if inv in need and have[inv] - sum(1 for i in bot.inventory if i == inv) < need[inv]:
                        completes = True
                        break
            updated.append((d, bot_id, matching_count, completes))
        deliverers = updated

        # Sort: completer first, then more matching items, then shorter distance, then bot ID
        deliverers.sort(key=lambda x: (not x[3], -x[2], x[0], x[1]))

        # First N bots (token-style) proceed to drop-off; rest stage or queue.
        n_bots = len(world.state.bots)
        if n_bots >= 20:
            # 20+ bots with 3 zones: all deliverers proceed, PIBT handles queuing.
            for _, bot_id, _, _ in deliverers:
                self._assignments[bot_id].navigation_override = None
        elif n_bots >= 10:
            # 10-19 bots: all deliverers proceed to drop-off, PIBT handles queuing.
            for _, bot_id, _, _ in deliverers:
                self._assignments[bot_id].navigation_override = None
        else:
            max_slots = min(2, max(len(world.dropoff_adjacent_positions()), 1))
            staging = world.staging_positions()
            for i, (_, bot_id, _, _) in enumerate(deliverers):
                if i < max_slots:
                    self._assignments[bot_id].navigation_override = None
                elif staging:
                    bot = world.state.get_bot(bot_id)
                    if bot:
                        best_staging = min(
                            staging,
                            key=lambda p: world.distance(bot.position, p),
                        )
                        self._assignments[bot_id].navigation_override = best_staging
                        self._assignments[bot_id].path = None

    def _evict_dropoff_blockers(self, world: WorldModel) -> None:
        """Move bots away from drop_off zones if they aren't actively delivering matching items."""
        state = world.state

        drop_off_set = set(state.drop_off_zones)
        active = state.active_orders
        remaining_types = set(active[0].items_remaining) if active else set()

        # Check if any deliverers need a drop_off zone
        has_waiting_deliverers = False
        for bot_id, assignment in self._assignments.items():
            if not (assignment.task and assignment.task.task_type == TaskType.DELIVER):
                continue
            bot = state.get_bot(bot_id)
            if bot and bot.position not in drop_off_set and any(inv in remaining_types for inv in bot.inventory):
                has_waiting_deliverers = True
                break

        if not has_waiting_deliverers:
            return

        # Collect already-occupied eviction targets to avoid sending multiple bots to same spot
        occupied_evict: set[Pos] = set()
        for bot in state.bots:
            occupied_evict.add(bot.position)

        # Evict bots at or near any drop_off zone that aren't actively delivering
        staging = world.staging_positions()
        if not staging:
            return
        for bot in state.bots:
            nearest_drop = world.nearest_drop_off(bot.position, bot.id)
            evict_radius = 4 if len(state.bots) >= 10 else 2
            d_to_drop = world.distance(bot.position, nearest_drop)
            if d_to_drop > evict_radius:
                continue  # Far enough away
            assignment = self._assignments.get(bot.id)
            if not assignment:
                continue
            # Bot is delivering with matching items — let it stay
            if (assignment.task and assignment.task.task_type == TaskType.DELIVER
                    and any(inv in remaining_types for inv in bot.inventory)):
                continue
            # Bot already has a non-DELIVER task (e.g. PICK_UP) heading away — it will leave
            if (assignment.task and assignment.task.task_type in (TaskType.PICK_UP, TaskType.PRE_PICK)
                    and assignment.task.target_pos not in drop_off_set):
                continue
            # Find eviction target: shelf-adjacent position away from drop-off approach
            evict_pos = self._find_spread_eviction(world, bot, remaining_types, occupied_evict)
            if evict_pos:
                occupied_evict.add(evict_pos)
                assignment.navigation_override = evict_pos
                assignment.path = None
                logger.info("EVICT: B%d near drop_off (d=%d), routing to %s",
                            bot.id, d_to_drop, evict_pos)

    def _find_spread_eviction(
        self,
        world: WorldModel,
        bot: Bot,
        active_types: set[str],
        occupied: set[Pos],
    ) -> Pos | None:
        """Find a shelf-adjacent parking spot spread across the map.

        Rules:
        - Must be walkable, not occupied by another bot/eviction target
        - Must be adjacent to a shelf (so bot is "waiting at a shelf")
        - NOT adjacent to shelves holding items needed by active order
        - Prefer positions far from drop-off to keep delivery lanes clear
        - Max 1 bot per shelf-adjacent position
        """
        grid = world._grid
        state = world.state
        drop_off = world.nearest_drop_off(bot.position, bot.id)

        # Build set of shelf positions holding active-order items
        active_shelves: set[Pos] = set()
        for item in state.items:
            if item.type in active_types:
                active_shelves.add(item.position)

        # Build shelf set
        shelf_set: set[Pos] = set()
        for item in state.items:
            shelf_set.add(item.position)

        candidates: list[tuple[int, int, Pos]] = []  # (travel_dist, -drop_dist, pos)
        for x in range(grid.width):
            for y in range(grid.height):
                pos = (x, y)
                if not grid.is_walkable(pos):
                    continue
                if pos in occupied:
                    continue
                # Must be adjacent to at least one shelf
                adjacent_to_shelf = False
                adjacent_to_active = False
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    n = (x + dx, y + dy)
                    if n in shelf_set:
                        adjacent_to_shelf = True
                    if n in active_shelves:
                        adjacent_to_active = True
                if not adjacent_to_shelf:
                    continue
                if adjacent_to_active:
                    continue  # Don't park near shelves with needed items
                # Avoid drop-off approach corridor (x <= 2 on left-drop maps)
                drop_x = drop_off[0]
                if abs(x - drop_x) <= 2:
                    continue
                d_travel = world.distance(bot.position, pos)
                if d_travel >= 9999:
                    continue
                d_from_drop = world.distance(pos, drop_off)
                # Prefer: close to bot (minimize travel) but far from drop-off
                candidates.append((d_travel, -d_from_drop, pos))

        if not candidates:
            # Fallback: any position far from drop-off
            staging = world.staging_positions()
            if staging:
                return min(staging, key=lambda p: world.distance(bot.position, p))
            return None

        candidates.sort()
        return candidates[0][2]

    def _hivemind_clear_path(self, world: WorldModel) -> None:
        """Hivemind: find the priority deliverer's path and clear ALL blockers.

        Instead of PIBT pushing one bot at a time (5+ rounds to clear a corridor),
        preemptively move ALL bots on the delivery path to adjacent side positions.
        Clears the whole corridor in 1-2 rounds.
        """
        state = world.state
        active = state.active_orders
        if not active:
            return
        remaining_types = set(active[0].items_remaining)

        # Find the priority deliverer: closest bot with matching inventory to any zone
        best_deliverer = None
        best_dist = 9999
        best_drop_off = state.drop_off
        for bot in state.bots:
            a = self._assignments.get(bot.id)
            if not (a and a.task and a.task.task_type == TaskType.DELIVER):
                continue
            if not any(inv in remaining_types for inv in bot.inventory):
                continue
            nearest = world.nearest_drop_off(bot.position, bot.id)
            d = world.distance(bot.position, nearest)
            if d < best_dist:
                best_dist = d
                best_deliverer = bot
                best_drop_off = nearest

        if best_deliverer is None or best_dist <= 1:
            return  # No deliverer or already at drop-off

        # Find deliverer's path to nearest drop-off
        path = self._path_engine.find_path(best_deliverer.position, best_drop_off)
        if not path:
            return

        path_set = set(path)
        bot_positions = {b.id: b.position for b in state.bots}

        # Find all bots ON the path (except the deliverer)
        for bot in state.bots:
            if bot.id == best_deliverer.id:
                continue
            if bot.position not in path_set:
                continue

            a = self._assignments.get(bot.id)
            if a is None:
                continue
            # Don't redirect bots that are also delivering with matching items
            if (a.task and a.task.task_type == TaskType.DELIVER
                    and any(inv in remaining_types for inv in bot.inventory)):
                continue

            # Find adjacent position OFF the path to dodge to
            bx, by = bot.position
            dodge_pos = None
            best_dodge_d = 9999
            grid = world._grid
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = bx + dx, by + dy
                candidate = (nx, ny)
                if not grid.is_walkable(candidate):
                    continue
                if candidate in path_set:
                    continue  # Don't dodge onto the path
                # Don't dodge into another bot
                if any(pos == candidate for bid, pos in bot_positions.items() if bid != bot.id):
                    continue
                # Prefer positions that are further from drop-off (less likely to block later)
                d = world.distance(candidate, best_drop_off)
                if d < best_dodge_d:
                    best_dodge_d = d
                    dodge_pos = candidate

            if dodge_pos:
                a.navigation_override = dodge_pos
                a.path = None
                logger.debug("HIVEMIND: B%d dodging from %s to %s (clearing for B%d)",
                            bot.id, bot.position, dodge_pos, best_deliverer.id)

    def _detect_stuck_bots(self, state: GameState, world: WorldModel) -> None:
        """Clear assignments for bots stuck or oscillating without progress."""
        STUCK_THRESHOLD = 5
        OSCILLATION_WINDOW = 40
        OSCILLATION_MAX_UNIQUE = 3

        for bot in state.bots:
            # Update position history
            hist = self._position_history.setdefault(bot.id, [])
            hist.append(bot.position)
            if len(hist) > OSCILLATION_WINDOW:
                hist.pop(0)

            prev_pos = self._last_bot_positions.get(bot.id)
            if prev_pos is None:
                self._stuck_rounds[bot.id] = 0
                continue
            if bot.position == prev_pos:
                self._stuck_rounds[bot.id] = self._stuck_rounds.get(bot.id, 0) + 1
            else:
                self._stuck_rounds[bot.id] = 0

            assignment = self._assignments.get(bot.id)
            if not assignment or not assignment.task:
                continue
            task = assignment.task

            # --- Stationary stuck detection ---
            rounds_stuck = self._stuck_rounds[bot.id]
            if rounds_stuck >= STUCK_THRESHOLD:
                if task.task_type in (TaskType.PICK_UP, TaskType.DELIVER):
                    if bot.position == assignment.effective_target:
                        continue
                if task.task_type == TaskType.IDLE:
                    from bot.strategy.task import Task
                    assignment.task = Task(task_type=TaskType.IDLE, target_pos=bot.position)
                    assignment.path = None
                    self._stuck_rounds[bot.id] = 0
                    continue
                logger.warning(
                    "STUCK: B%d at %s for %d rounds (task=%s tgt=%s) — clearing",
                    bot.id, bot.position, rounds_stuck,
                    task.task_type.name, assignment.effective_target,
                )
                if task.item_id and task.task_type in (TaskType.PICK_UP, TaskType.PRE_PICK):
                    planner = getattr(self._planner, "_reactive", self._planner)
                    expiry = getattr(self._config, "blacklist_expiry_rounds", 8)
                    planner.blacklist_item(task.item_id, state.round + expiry)
                assignment.clear()
                self._stuck_rounds[bot.id] = 0
                continue

            # --- Oscillation detection (10+ bots, DELIVER only) ---
            # Detect bots oscillating near a blocked drop-off and reroute to another zone.
            # Only triggers if another bot is parked ON the target drop-off.
            if (len(state.bots) >= 10
                    and len(hist) >= OSCILLATION_WINDOW
                    and task.task_type == TaskType.DELIVER
                    and len(state.drop_off_zones) > 1):
                unique = set(hist)
                target = assignment.effective_target
                if (len(unique) <= OSCILLATION_MAX_UNIQUE
                        and target not in unique):
                    # Check if target is blocked by a non-delivering bot
                    blocker = None
                    for other in state.bots:
                        if other.id == bot.id:
                            continue
                        if other.position == target:
                            other_a = self._assignments.get(other.id)
                            if other_a and other_a.task:
                                if other_a.task.task_type != TaskType.DELIVER:
                                    blocker = other
                            break
                    if blocker is not None:
                        other_zones = [z for z in state.drop_off_zones if z != target]
                        new_target = min(other_zones,
                            key=lambda z: world.distance(bot.position, z))
                        from bot.strategy.task import Task
                        assignment.task = Task(task_type=TaskType.DELIVER,
                                               target_pos=new_target)
                        assignment.path = None
                        hist.clear()
                        logger.warning(
                            "OSCILLATE: B%d rerouted DELIVER %s -> %s (blocked by B%d)",
                            bot.id, target, new_target, blocker.id,
                        )

    def finalize_game(self, total_rounds: int, final_score: int) -> None:
        """Called at game_over. Saves recon data and generates plan."""
        recon_data = self._game_logger.finalize(total_rounds, final_score)
        logger.info(
            "Game finalized: score=%d, rounds=%d, orders=%d",
            final_score, total_rounds, len(recon_data.get("order_sequence", [])),
        )
        # Collision summary
        if self._collision_log:
            logger.warning("=== COLLISION SUMMARY: %d blocked moves ===", len(self._collision_log))
            from collections import Counter
            by_bot = Counter(c["bot_id"] for c in self._collision_log)
            for bot_id, count in sorted(by_bot.items()):
                logger.warning("  B%d: %d blocked moves", bot_id, count)
            # Show first 20 collisions
            for c in self._collision_log[:20]:
                logger.warning("  R%d B%d %s: %s -> expected %s, stayed at %s",
                              c["round"], c["bot_id"], c["action"],
                              c["from"], c["expected"], c["actual"])
            if len(self._collision_log) > 20:
                logger.warning("  ... and %d more", len(self._collision_log) - 20)

    def reset(self) -> None:
        """Reset all state for a new game. Config is preserved."""
        self._assignments.clear()
        self._round = 0
        self._last_commands.clear()
        self._last_bot_positions.clear()
        self._stuck_rounds.clear()
        self._position_history.clear()
        self._round_offset = False
        self._offset_checked = False
        self._collision_log = []
        self._game_logger = GameLogger()
        self._shelf_positions = frozenset()
        self._replay_attempted = False
        self._time_space_planner = None
        if self._config and getattr(self._config, 'planner_version', 1) == 2:
            from bot.strategy.v2.planner import V2TaskPlanner
            self._planner = V2TaskPlanner()
        else:
            self._planner = TaskPlanner()
