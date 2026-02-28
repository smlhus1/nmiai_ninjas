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
from typing import Any

from bot.models import GameState, BotCommand
from bot.engine.pathfinding import PathEngine
from bot.engine.world_model import WorldModel
from bot.strategy.planner import TaskPlanner
from bot.strategy.action_resolver import ActionResolver
from bot.strategy.task import BotAssignment, TaskType

logger = logging.getLogger(__name__)


class Coordinator:
    """
    Main bot coordinator. Create once, call on_game_state() each round.
    """

    def __init__(self) -> None:
        self._path_engine = PathEngine()
        self._planner = TaskPlanner()
        self._resolver = ActionResolver(self._path_engine)
        self._assignments: dict[int, BotAssignment] = {}
        self._round = 0

    def on_game_state(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Main entry point. Takes raw game state dict, returns action response dict.
        This is the only method the WebSocket client needs to call.
        """
        t_start = time.perf_counter()

        # 1. Parse
        state = GameState.from_dict(raw)
        self._round = state.round

        # 2. Initialize assignments for new bots
        for bot in state.bots:
            if bot.id not in self._assignments:
                self._assignments[bot.id] = BotAssignment(bot_id=bot.id)

        # Remove assignments for bots that no longer exist
        active_ids = {b.id for b in state.bots}
        self._assignments = {
            k: v for k, v in self._assignments.items() if k in active_ids
        }

        # 3. Set up pathfinding
        self._path_engine.set_grid(state.grid)
        self._path_engine.new_round()

        # 4. Build world model
        world = WorldModel(state, self._path_engine)

        # 5. Plan tasks
        self._assignments = self._planner.plan(world, self._assignments)

        # 5.5 Drop-off scheduling: limit concurrent deliverers
        self._schedule_dropoff(world)

        # 6. Resolve to actions
        commands = self._resolver.resolve(state, self._assignments)

        # 7. Build response
        response = {"actions": [cmd.to_dict() for cmd in commands]}

        t_elapsed = time.perf_counter() - t_start
        logger.info(
            "Round %d: %d bots, %.1fms",
            state.round,
            len(state.bots),
            t_elapsed * 1000,
        )
        if t_elapsed > 1.0:
            logger.warning("Round %d took %.1fms — dangerously close to 2s limit!", state.round, t_elapsed * 1000)

        return response

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

    def reset(self) -> None:
        """Reset all state for a new game."""
        self._assignments.clear()
        self._round = 0
        # Keep path engine — grid cache might still be valid
