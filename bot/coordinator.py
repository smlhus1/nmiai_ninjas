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
import os
import time
from collections import Counter
from datetime import date
from typing import Any

from bot.models import Action, Bot, GameState, BotCommand, Grid, Pos, apply_move
from bot.engine.pathfinding import PathEngine
from bot.engine.world_model import WorldModel
from bot.strategy.planner import TaskPlanner
from bot.strategy.action_resolver import ActionResolver
from bot.strategy.task import BotAssignment, TaskType
from bot.recon.logger import GameLogger
from bot.recon.replay import ReplayPlanner

logger = logging.getLogger(__name__)

_LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

_MOVE_ACTIONS = frozenset({
    Action.MOVE_UP, Action.MOVE_DOWN,
    Action.MOVE_LEFT, Action.MOVE_RIGHT,
})


class Coordinator:
    """
    Main bot coordinator. Create once, call on_game_state() each round.
    """

    def __init__(self) -> None:
        self._path_engine = PathEngine()
        self._planner = TaskPlanner()
        self._active_planner = self._planner  # May switch to ReplayPlanner
        self._resolver = ActionResolver(self._path_engine)
        self._assignments: dict[int, BotAssignment] = {}
        self._round = 0
        self._last_commands: dict[int, BotCommand] = {}
        self._last_bot_positions: dict[int, Pos] = {}
        self._round_offset = False
        self._offset_checked = False
        self._game_logger = GameLogger()
        self._shelf_positions: frozenset[Pos] = frozenset()
        self._replay_checked = False
        self._logs_dir = _LOGS_DIR

    def on_game_state(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Main entry point. Takes raw game state dict, returns action response dict.
        This is the only method the WebSocket client needs to call.
        """
        t_start = time.perf_counter()

        # 1. Parse (real server state)
        real_state = GameState.from_dict(raw)
        self._round = real_state.round

        # 2. Detect round offset (diagnostic logging only)
        if real_state.round >= 1 and not self._offset_checked:
            self._detect_offset(real_state)

        # 3. Use real state directly — offset compensation is disabled because:
        #    - Prediction errors cascade (collisions make moves fail unpredictably)
        #    - pick_up/drop_off silently become WAIT on wrong positions
        #    - Shelf positions aren't walls in raw grid, causing invalid predictions
        state = real_state

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
        if self._shelf_positions:
            merged_walls = state.grid.walls | self._shelf_positions
            merged_grid = Grid(state.grid.width, state.grid.height, merged_walls)
            self._path_engine.set_grid(merged_grid)
        else:
            self._path_engine.set_grid(state.grid)
        self._path_engine.new_round()

        # 5.5. Recon logging
        self._game_logger.on_round(state, self._shelf_positions)

        # 5.6. Try loading replay plan (once, after fingerprint is available)
        if not self._replay_checked and self._shelf_positions:
            self._replay_checked = True
            self._try_load_plan()

        # 6. Build world model
        world = WorldModel(state, self._path_engine)

        # 7. Plan tasks
        self._assignments = self._active_planner.plan(world, self._assignments)

        # 7.5 Drop-off scheduling
        self._schedule_dropoff(world)

        # 8. Resolve to actions
        commands = self._resolver.resolve(state, self._assignments)

        # 9. Build response
        response = {"actions": [cmd.to_dict() for cmd in commands]}

        t_elapsed = time.perf_counter() - t_start

        # --- Logging ---
        if real_state.round == 0:
            for item in real_state.items:
                logger.info("MAP: item %s (%s) @ %s", item.id, item.type, item.position)
            logger.info("MAP: drop_off @ %s | grid %dx%d",
                        real_state.drop_off, real_state.grid.width, real_state.grid.height)

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
        # Positions where bots were last round (when move was attempted)
        prev_positions = set(self._last_bot_positions.values())
        curr_positions = {b.position for b in state.bots}

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
                # Move failed. Could be offset OR collision.
                # Check if another bot was at the expected position last round
                # (collision happens at action-resolution time, not current state).
                was_blocked = any(
                    pos == expected for bid, pos in self._last_bot_positions.items()
                    if bid != bot.id
                )
                if was_blocked:
                    continue  # Collision, not offset
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
            score=state.score,
        )

    def _schedule_dropoff(self, world: WorldModel) -> None:
        """Limit concurrent drop-off approaches to avoid gridlock."""
        # Count walkable cells adjacent to drop-off = max concurrent deliverers
        max_slots = max(len(world.dropoff_adjacent_positions()), 1)

        # Clear all navigation overrides for deliverers first
        for assignment in self._assignments.values():
            if assignment.task and assignment.task.task_type == TaskType.DELIVER:
                assignment.navigation_override = None

        # Find all bots with DELIVER tasks, sorted by distance to drop-off
        deliverers: list[tuple[int, int]] = []  # (distance, bot_id)
        for bot_id, assignment in self._assignments.items():
            if assignment.task and assignment.task.task_type == TaskType.DELIVER:
                bot = world.state.get_bot(bot_id)
                if bot:
                    d = world.distance(bot.position, world.state.drop_off)
                    deliverers.append((d, bot_id))

        if len(deliverers) <= max_slots:
            return  # No scheduling needed

        # Sort by distance (closest first) — let closest bots deliver
        deliverers.sort()

        # Bots beyond the limit get redirected to staging via override
        staging = world.staging_positions()
        for i, (_, bot_id) in enumerate(deliverers):
            if i >= max_slots and staging:
                bot = world.state.get_bot(bot_id)
                if bot:
                    best_staging = min(
                        staging,
                        key=lambda p: world.distance(bot.position, p),
                    )
                    self._assignments[bot_id].navigation_override = best_staging
                    self._assignments[bot_id].path = None  # Force recompute

    def _try_load_plan(self) -> None:
        """Check for an existing plan file matching today's fingerprint."""
        fp = self._game_logger.fingerprint
        if not fp:
            return

        today = date.today().isoformat()
        # Check canonical logs dir first, then instance logs dir
        plan_path = os.path.join(_LOGS_DIR, f"{fp}_{today}_plan.json")
        if not os.path.exists(plan_path):
            plan_path = os.path.join(self._logs_dir, f"{fp}_{today}_plan.json")
        if not os.path.exists(plan_path):
            logger.info("No plan file found at %s — running in recon mode", plan_path)
            return

        try:
            with open(plan_path) as f:
                game_plan = json.load(f)
            self._active_planner = ReplayPlanner(game_plan, self._planner)
            logger.info("REPLAY MODE: loaded plan from %s (%d orders)",
                        plan_path, len(game_plan.get("order_plans", [])))
        except Exception:
            logger.exception("Failed to load plan from %s — using reactive", plan_path)

    def finalize_game(self, total_rounds: int, final_score: int) -> None:
        """Called at game_over. Saves recon data and generates plan for next run."""
        recon_data = self._game_logger.finalize(total_rounds, final_score)
        logger.info(
            "Game finalized: score=%d, rounds=%d, orders=%d",
            final_score, total_rounds, len(recon_data.get("order_sequence", [])),
        )

        os.makedirs(self._logs_dir, exist_ok=True)
        fp = recon_data.get("fingerprint", "unknown")
        today = date.today().isoformat()

        # Save recon data
        recon_path = os.path.join(self._logs_dir, f"{fp}_{today}_recon.json")
        with open(recon_path, "w") as f:
            json.dump(recon_data, f, indent=2)
        logger.info("Recon data saved to %s", recon_path)

        # Generate plan from recon data
        try:
            from bot.recon.analyzer import OfflinePlanner as ReconAnalyzer
            analyzer = ReconAnalyzer(recon_data, self._path_engine)
            game_plan = analyzer.plan()
            plan_path = os.path.join(self._logs_dir, f"{fp}_{today}_plan.json")
            with open(plan_path, "w") as f:
                json.dump(game_plan, f, indent=2)
            logger.info("Plan generated and saved to %s (%d orders)",
                        plan_path, len(game_plan.get("order_plans", [])))
        except Exception:
            logger.exception("Failed to generate plan from recon data")

    def reset(self) -> None:
        """Reset all state for a new game."""
        self._assignments.clear()
        self._round = 0
        self._last_commands.clear()
        self._last_bot_positions.clear()
        self._round_offset = False
        self._offset_checked = False
        self._game_logger = GameLogger()
        self._shelf_positions = frozenset()
        self._replay_checked = False
        self._planner = TaskPlanner()
        self._active_planner = self._planner
