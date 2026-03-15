"""V3Coordinator — thin orchestration layer.

Replaces V1 Coordinator with job-based architecture.
Same interface: on_game_state(raw) -> response dict.

Pipeline per round:
1. Parse GameState
2. Setup grid/pathfinding (cached)
3. Update pipeline state
4. Dispatch jobs (Hungarian + multi-item)
5. Generate intents from jobs
6. EPIBT resolve (guidance-weighted)
7. LNS refine
8. Convert to BotCommands
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Optional
import json

from bot.config import CoordinatorConfig
from bot.models import (
    Action, Bot, BotCommand, GameState, Grid, Pos, apply_move,
)
from bot.engine.pathfinding import PathEngine
from bot.engine.world_model import WorldModel
from bot.recon.logger import GameLogger

from bot.v3.job import BotState, Job
from bot.v3.guidance import SoftGuidance
from bot.v3.epibt import EPIBTResolver
from bot.engine.pibt import PIBTResolver
from bot.v3.dispatcher import JobDispatcher
from bot.v3.pipeline import OrderPipeline, PipelinePhase
from bot.v3.lns import CooperativeLNS

logger = logging.getLogger(__name__)

_MOVE_ACTIONS = frozenset({
    Action.MOVE_UP, Action.MOVE_DOWN,
    Action.MOVE_LEFT, Action.MOVE_RIGHT,
})


class V3Coordinator:
    """V3 bot coordinator. Drop-in replacement for V1 Coordinator."""

    def __init__(self, config: CoordinatorConfig | None = None) -> None:
        self._config = config
        self._path_engine = PathEngine()
        self._guidance: SoftGuidance | None = None
        self._epibt: EPIBTResolver | None = None
        self._pibt: PIBTResolver | None = None
        self._lns: CooperativeLNS | None = None
        self._dispatcher = JobDispatcher()
        self._pipeline = OrderPipeline()
        self._game_logger = GameLogger()

        self._shelf_positions: frozenset[Pos] = frozenset()
        self._merged_grid: Grid | None = None
        self._round = 0
        self._last_commands: dict[int, BotCommand] = {}
        self._last_bot_positions: dict[int, Pos] = {}
        self._collision_log: list[dict] = []
        self._round_offset = False
        self._offset_checked = False

        # Stuck detection
        self._stuck_rounds: dict[int, int] = {}
        self._position_history: dict[int, list[Pos]] = {}
        # Bot cooldown: bot_id -> round when cooldown expires (don't assign new jobs)
        self._bot_cooldown: dict[int, int] = {}
        self._stuck_count: dict[int, int] = {}  # bot_id -> total stuck events

        # Viz (optional)
        self._viz: Any = None

    def start_viz(self, port: int = 8765) -> None:
        from bot.viz_broadcaster import VizBroadcaster
        self._viz = VizBroadcaster(port=port)
        self._viz.start()

    def on_game_state(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Main entry point. Same interface as V1 Coordinator."""
        t_start = time.perf_counter()

        # 1. Parse
        real_state = GameState.from_dict(raw)
        self._round = real_state.round

        # 1.5. Collision tracking
        if self._last_commands and real_state.round >= 1:
            self._track_collisions(real_state)

        # 2. Offset detection
        if real_state.round >= 1 and not self._offset_checked:
            self._detect_offset(real_state)
        state = self._compensate_offset(real_state) if self._round_offset else real_state

        # 3. Setup grid (merge shelf positions, cached)
        self._setup_grid(state)
        self._path_engine.new_round()

        # 4. Auto-detect config (first round)
        if state.round == 0:
            if self._config is None:
                self._config = CoordinatorConfig.for_difficulty(len(state.bots))
                n = len(state.bots)
                diff = ("easy" if n == 1 else "medium" if n <= 3
                        else "nightmare" if n >= 20 else "expert" if n >= 10 else "hard")
                logger.info("V3: auto-detected %d bots -> %s config", n, diff)

        # 5. Setup guidance (once, after grid)
        if self._guidance is None and self._merged_grid is not None:
            self._guidance = SoftGuidance(
                self._merged_grid,
                state.drop_off_zones,
            )
            logger.info("V3: SoftGuidance initialized")

        # 5.5. Update guidance congestion
        if self._guidance:
            self._guidance.on_round(
                {b.id: b.position for b in state.bots},
                state.round,
            )

        # 5.6. Setup EPIBT (once, after guidance)
        if self._epibt is None and self._merged_grid is not None:
            self._epibt = EPIBTResolver(
                self._merged_grid,
                self._path_engine.distance,
                guidance=self._guidance,
                path_engine=self._path_engine,
            )

        # 5.7. Setup LNS (for 10+ bots)
        if self._lns is None and len(state.bots) >= 10:
            self._lns = CooperativeLNS(budget_ms=200)

        # 5.8. Recon logging
        self._game_logger.on_round(state, self._shelf_positions)

        # 5.9. Viz metadata
        if self._viz and state.round == 0:
            self._viz.set_metadata(
                shelf_positions=self._shelf_positions,
                one_way={},  # V3 uses soft guidance, no hard one-way
            )

        # 6. Build world model
        max_ri = self._config.max_route_items if self._config else 3
        world = WorldModel(state, self._path_engine, max_route_items=max_ri)

        # 7. Update pipeline
        self._pipeline.update(state)

        # 8. Dispatch jobs
        bot_states = self._dispatcher.dispatch(
            state, self._path_engine, world, self._pipeline,
        )

        # 9. Stuck detection
        self._detect_stuck_bots(state, bot_states)

        # 9.5. Drop-off eviction: move non-delivering bots away from drop-off
        self._evict_dropoff_blockers(state, bot_states, world)

        # 10. Generate intents and resolve actions
        commands = self._resolve_actions(state, bot_states, world)

        # 10.5. Viz
        if self._viz:
            # Fake assignments for viz compatibility
            self._viz.send_state(state, {})

        # 11. Build response
        response = {"actions": [cmd.to_dict() for cmd in commands]}

        t_elapsed = time.perf_counter() - t_start

        # --- Logging ---
        if real_state.round == 0:
            logger.info("V3: grid %dx%d, %d bots, zones=%s",
                        real_state.grid.width, real_state.grid.height,
                        len(real_state.bots), real_state.drop_off_zones)

        if real_state.round % 50 == 0:
            active_jobs = sum(1 for bs in bot_states.values() if bs.job is not None)
            delivering = sum(
                1 for bs in bot_states.values()
                if bs.job and bs.job.is_delivering
            )
            multi = sum(
                1 for bs in bot_states.values()
                if bs.job and len(bs.job.pickups) > 1
            )
            logger.info(
                "V3 R%d: score=%d, jobs=%d (delivering=%d, multi=%d), pipeline=%s",
                real_state.round, real_state.score,
                active_jobs, delivering, multi,
                self._pipeline.phase.name,
            )

        if t_elapsed > 1.0:
            logger.warning("V3 R%d took %.1fms!", real_state.round, t_elapsed * 1000)

        self._last_commands = {cmd.bot_id: cmd for cmd in commands}
        self._last_bot_positions = {b.id: b.position for b in real_state.bots}

        return response

    def _setup_grid(self, state: GameState) -> None:
        """Merge shelf positions into grid (cached after first round)."""
        if not self._shelf_positions and state.items:
            self._shelf_positions = frozenset(item.position for item in state.items)
            # Enable one-way aisles for multi-bot (essential for corridor deadlocks)
            if len(state.bots) >= 4:
                self._path_engine.enable_one_way(True)
        if self._shelf_positions:
            merged_walls = state.grid.walls | self._shelf_positions
            self._merged_grid = Grid(state.grid.width, state.grid.height, merged_walls)
            self._path_engine.set_grid(self._merged_grid, drop_off=state.drop_off)
        else:
            self._path_engine.set_grid(state.grid, drop_off=state.drop_off)

    def _resolve_actions(
        self,
        state: GameState,
        bot_states: dict[int, BotState],
        world: WorldModel,
    ) -> list[BotCommand]:
        """Convert job states to EPIBT intents, resolve collisions, emit commands."""
        commands: dict[int, BotCommand] = {}
        movement_bots: dict[int, Bot] = {}
        movement_targets: dict[int, Pos] = {}
        urgency: dict[int, int] = {}
        idle_bot_ids: set[int] = set()

        sorted_bots = sorted(state.bots, key=lambda b: b.id)

        for bot in sorted_bots:
            bs = bot_states.get(bot.id)
            job = bs.job if bs else None

            if job is None:
                # IDLE: check if evicted from drop-off area
                evict_pos = getattr(self, '_evict_targets', {}).get(bot.id)
                movement_bots[bot.id] = bot
                if evict_pos:
                    movement_targets[bot.id] = evict_pos
                    urgency[bot.id] = -1
                elif len(state.bots) >= 10:
                    # Park idle bots in safe spots (not in one-way aisles)
                    safe = self._find_safe_idle_pos(bot.position)
                    movement_targets[bot.id] = safe if safe else bot.position
                    urgency[bot.id] = 3
                    idle_bot_ids.add(bot.id)
                else:
                    movement_targets[bot.id] = bot.position
                    urgency[bot.id] = 3
                    idle_bot_ids.add(bot.id)
                continue

            if job.is_delivering:
                # DELIVER phase
                if bot.position in state.drop_off_zones:
                    # At drop-off: check if we have matching items
                    active = state.active_orders
                    if active:
                        remaining = list(active[0].items_remaining)
                        has_match = any(inv in remaining for inv in bot.inventory)
                    else:
                        has_match = bool(bot.inventory)

                    if has_match:
                        commands[bot.id] = BotCommand(bot.id, Action.DROP_OFF)
                        continue
                    else:
                        # No matching items — cancel delivery job, escape
                        bs = bot_states.get(bot.id)
                        if bs:
                            bs.job = None
                        movement_bots[bot.id] = bot
                        urgency[bot.id] = -1  # ESCAPE priority
                        escape = self._find_escape_pos(bot.position, state)
                        movement_targets[bot.id] = escape
                        continue

                # Navigate to delivery zone
                movement_bots[bot.id] = bot
                movement_targets[bot.id] = job.delivery_zone
                urgency[bot.id] = 0  # DELIVER priority
                continue

            # PICKUP phase
            step = job.current_pickup
            if step is None:
                # Shouldn't happen, but safety
                movement_bots[bot.id] = bot
                movement_targets[bot.id] = bot.position
                urgency[bot.id] = 3
                continue

            # Check if adjacent to shelf -> pick up
            if self._path_engine.manhattan(bot.position, step.shelf_pos) == 1:
                if len(bot.inventory) < 3:
                    # Find actual item at this shelf
                    item_id = self._find_item_at_shelf(
                        state, step.shelf_pos, step.item_type,
                    )
                    if item_id:
                        commands[bot.id] = BotCommand(
                            bot.id, Action.PICK_UP, item_id=item_id,
                        )
                        continue

            # Navigate to pickup position (but evict if near drop-off)
            evict_pos = getattr(self, '_evict_targets', {}).get(bot.id)
            movement_bots[bot.id] = bot
            if evict_pos:
                movement_targets[bot.id] = evict_pos
                urgency[bot.id] = -1  # Escape priority
            else:
                movement_targets[bot.id] = step.pickup_pos
                urgency[bot.id] = 1 if job.priority <= 1 else 2

        # Add stationary bots to PIBT as obstacles
        for bot in sorted_bots:
            if bot.id in commands and bot.id not in movement_bots:
                movement_bots[bot.id] = bot
                movement_targets[bot.id] = bot.position
                urgency[bot.id] = 3
                idle_bot_ids.add(bot.id)

        # EPIBT resolve
        if movement_bots and self._epibt:
            bot_positions = {bid: bot.position for bid, bot in movement_bots.items()}
            next_positions = self._epibt.resolve(
                bot_positions, movement_targets,
                urgency=urgency,
                tiebreak_offset=state.round,
            )

            # Convert to actions
            for bot_id, next_pos in next_positions.items():
                if bot_id in commands:
                    continue
                bot = movement_bots[bot_id]
                action = self._pos_to_action(bot.position, next_pos)
                commands[bot_id] = BotCommand(bot_id, action)

        return [commands[bot.id] for bot in sorted_bots if bot.id in commands]

    def _find_escape_pos(self, pos: Pos, state: GameState) -> Pos:
        """Find a walkable cell away from drop-off for escape."""
        if self._merged_grid is None:
            return pos
        drop_zones = set(state.drop_off_zones)
        x, y = pos
        best: Pos | None = None
        best_d = -1
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            candidate = (nx, ny)
            if not self._merged_grid.is_walkable(candidate):
                continue
            if candidate in drop_zones:
                continue
            # Prefer moving away from all drop-off zones
            min_drop_d = min(
                abs(nx - dz[0]) + abs(ny - dz[1]) for dz in state.drop_off_zones
            )
            if min_drop_d > best_d:
                best_d = min_drop_d
                best = candidate
        return best if best else pos

    @staticmethod
    def _find_item_at_shelf(
        state: GameState, shelf_pos: Pos, item_type: str,
    ) -> Optional[str]:
        """Find an actual item ID at a shelf position."""
        for item in state.items:
            if item.position == shelf_pos and item.type == item_type:
                return item.id
        return None

    @staticmethod
    def _pos_to_action(from_pos: Pos, to_pos: Pos) -> Action:
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        if dx == 1:
            return Action.MOVE_RIGHT
        elif dx == -1:
            return Action.MOVE_LEFT
        elif dy == 1:
            return Action.MOVE_DOWN
        elif dy == -1:
            return Action.MOVE_UP
        return Action.WAIT

    def _find_safe_idle_pos(self, pos: Pos) -> Pos | None:
        """Find nearest non-one-way walkable cell for idle parking."""
        if not self._path_engine._one_way:
            return None  # No one-way rules, anywhere is fine

        # If already in a safe spot, stay
        if pos not in self._path_engine._one_way:
            return pos

        # BFS to find nearest non-one-way walkable cell
        from collections import deque
        grid = self._merged_grid
        if grid is None:
            return None
        visited = {pos}
        queue = deque([(pos, 0)])
        while queue:
            curr, d = queue.popleft()
            if d > 0 and curr not in self._path_engine._one_way:
                return curr
            if d >= 8:  # Don't search too far
                continue
            cx, cy = curr
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                npos = (nx, ny)
                if npos not in visited and grid.is_walkable(npos):
                    visited.add(npos)
                    queue.append((npos, d + 1))
        return None

    def _evict_dropoff_blockers(
        self,
        state: GameState,
        bot_states: dict[int, BotState],
        world: WorldModel,
    ) -> None:
        """Mark non-delivering bots near drop-off for eviction.

        Stores evict target in _evict_targets dict, consumed by _resolve_actions.
        """
        self._evict_targets: dict[int, Pos] = {}  # bot_id -> evict_pos

        drop_zones = set(state.drop_off_zones)
        active = state.active_orders
        remaining_types = set(active[0].items_remaining) if active else set()

        # Check if any deliverer needs to reach a zone
        has_deliverer = False
        for bs in bot_states.values():
            if bs.job and bs.job.is_delivering:
                bot = state.get_bot(bs.bot_id)
                if bot and bot.position not in drop_zones:
                    if any(inv in remaining_types for inv in bot.inventory):
                        has_deliverer = True
                        break

        if not has_deliverer:
            return

        evict_radius = 4 if len(state.bots) >= 10 else 2
        for bot in state.bots:
            bs = bot_states.get(bot.id)
            if not bs:
                continue

            nearest_drop = world.nearest_drop_off(bot.position, bot.id)
            d = world.distance(bot.position, nearest_drop)
            if d > evict_radius:
                continue

            # Don't evict delivering bots with matching items
            if (bs.job and bs.job.is_delivering
                    and any(inv in remaining_types for inv in bot.inventory)):
                continue

            # Don't evict bots picking up nearby
            if bs.job and not bs.job.is_delivering:
                pickup = bs.job.current_pickup
                if pickup and world.distance(bot.position, pickup.pickup_pos) <= 3:
                    continue

            self._evict_targets[bot.id] = self._find_escape_pos(bot.position, state)

    def _detect_stuck_bots(
        self, state: GameState, bot_states: dict[int, BotState],
    ) -> None:
        """Clear jobs for bots stuck or looping too long."""
        threshold = 10 if len(state.bots) >= 20 else 6

        for bot in state.bots:
            prev_pos = self._last_bot_positions.get(bot.id)
            if prev_pos is None:
                self._stuck_rounds[bot.id] = 0
                self._position_history.setdefault(bot.id, [])
                continue

            if bot.position == prev_pos:
                self._stuck_rounds[bot.id] = self._stuck_rounds.get(bot.id, 0) + 1
            else:
                self._stuck_rounds[bot.id] = 0

            # Track position history (for future use, not actively clearing)
            hist = self._position_history.setdefault(bot.id, [])
            hist.append(bot.position)
            if len(hist) > 20:
                hist.pop(0)

            bs = bot_states.get(bot.id)
            if not bs or not bs.job:
                continue

            rounds_stuck = self._stuck_rounds[bot.id]
            if rounds_stuck < threshold:
                continue

            job = bs.job
            # Don't clear if at target
            if bot.position == job.current_target:
                continue

            # Don't clear delivering bots near drop-off (they're queued, not stuck)
            if job.is_delivering:
                near_dropoff = False
                for zone in state.drop_off_zones:
                    if abs(bot.position[0] - zone[0]) + abs(bot.position[1] - zone[1]) <= 6:
                        near_dropoff = True
                        break
                if near_dropoff:
                    continue

            logger.warning(
                "V3 STUCK: B%d at %s for %d rounds (job=%s) — clearing",
                bot.id, bot.position, rounds_stuck, job.job_id,
            )
            bs.job = None
            self._stuck_rounds[bot.id] = 0

    def _track_collisions(self, state: GameState) -> None:
        """Track server-rejected moves for diagnostics."""
        for bot in state.bots:
            cmd = self._last_commands.get(bot.id)
            old_pos = self._last_bot_positions.get(bot.id)
            if cmd is None or old_pos is None:
                continue
            if cmd.action in _MOVE_ACTIONS:
                expected = apply_move(old_pos, cmd.action)
                if bot.position != expected and bot.position == old_pos:
                    self._collision_log.append({
                        "round": state.round, "bot_id": bot.id,
                        "action": cmd.action.value,
                        "from": old_pos, "expected": expected, "actual": bot.position,
                    })

    def _detect_offset(self, state: GameState) -> None:
        """Check if actions are delayed by 1 round."""
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
                was_blocked = any(
                    pos == expected for bid, pos in self._last_bot_positions.items()
                    if bid != bot.id
                )
                if was_blocked:
                    continue
                self._round_offset = True
                self._offset_checked = True
                logger.warning("V3 OFFSET DETECTED R%d", state.round)
                return
            if bot.position == expected:
                self._offset_checked = True
                return

    def _compensate_offset(self, state: GameState) -> GameState:
        """Predict bot positions if actions are delayed."""
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
            round=state.round, max_rounds=state.max_rounds,
            grid=state.grid, bots=tuple(adjusted_bots),
            items=state.items, orders=state.orders,
            drop_off=state.drop_off, drop_off_zones=state.drop_off_zones,
            score=state.score,
        )

    def finalize_game(self, total_rounds: int, final_score: int) -> None:
        """Called at game_over."""
        recon_data = self._game_logger.finalize(total_rounds, final_score)
        logger.info(
            "V3 Game finalized: score=%d, rounds=%d, orders=%d, collisions=%d",
            final_score, total_rounds,
            len(recon_data.get("order_sequence", [])),
            len(self._collision_log),
        )

    def reset(self) -> None:
        """Reset for new game. Config preserved."""
        self._dispatcher.reset()
        self._pipeline = OrderPipeline()
        self._game_logger = GameLogger()
        self._shelf_positions = frozenset()
        self._merged_grid = None
        self._guidance = None
        self._epibt = None
        self._pibt = None
        self._lns = None
        self._round = 0
        self._last_commands.clear()
        self._last_bot_positions.clear()
        self._stuck_rounds.clear()
        self._position_history.clear()
        self._collision_log = []
        self._round_offset = False
        self._offset_checked = False
