"""
TimeSpacePlanner: finds alternative routes for stuck bots using time-space A*.

PIBT handles normal collision resolution well (push/yield). TSP only activates
for bots that PIBT can't resolve — bots stuck at the same position for 3+ rounds.
TSP provides an alternative waypoint that routes around the blockage.

This is a targeted intervention, not a replacement for PIBT.
"""

from __future__ import annotations

import logging
import time

from bot.models import Pos
from bot.engine.pathfinding import PathEngine
from bot.engine.reservation import ReservationTable
from bot.engine.time_space_astar import find_path_tsa
from bot.engine.world_model import WorldModel
from bot.strategy.task import BotAssignment, TaskType

logger = logging.getLogger(__name__)

# Only plan for bots stuck this many rounds
_STUCK_THRESHOLD = 3


class TimeSpacePlanner:
    """
    Plans alternative routes for stuck bots using time-space A*.

    Normal movement uses PIBT. TSP only activates when a bot is stuck
    (same position for 3+ rounds) and provides a waypoint to break the deadlock.
    """

    def __init__(self, path_engine: PathEngine, horizon: int = 30,
                 max_planning_ms: int = 500) -> None:
        self._path = path_engine
        self._horizon = horizon
        self._max_planning_ms = max_planning_ms
        self._last_positions: dict[int, Pos] = {}
        self._stuck_count: dict[int, int] = {}  # bot_id -> rounds at same pos

    def plan(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        current_round: int,
    ) -> dict[int, list[Pos]]:
        """
        Plan alternative routes for stuck bots only.

        Returns:
            dict[bot_id, path] for stuck bots with valid alternative paths.
            Non-stuck bots are NOT included — they use PIBT as normal.
        """
        state = world.state
        result: dict[int, list[Pos]] = {}

        # Update stuck tracking
        for bot in state.bots:
            last_pos = self._last_positions.get(bot.id)
            if last_pos == bot.position:
                self._stuck_count[bot.id] = self._stuck_count.get(bot.id, 0) + 1
            else:
                self._stuck_count[bot.id] = 0
            self._last_positions[bot.id] = bot.position

        # Find stuck bots that have a real target (not IDLE at target)
        stuck_bots: list[int] = []
        for bot in state.bots:
            if self._stuck_count.get(bot.id, 0) < _STUCK_THRESHOLD:
                continue
            assignment = assignments.get(bot.id)
            if not assignment or not assignment.task:
                continue
            target = self._get_target(bot.position, assignment, state.drop_off)
            if target == bot.position:
                continue  # Already at target, not really stuck
            stuck_bots.append(bot.id)

        if not stuck_bots:
            return result

        t_start = time.perf_counter()

        # Build reservation table with ALL bot current positions as obstacles
        reservations = ReservationTable(horizon=self._horizon)
        for bot in state.bots:
            if bot.id not in stuck_bots:
                # Non-stuck bots: reserve current position at t=0 only
                # (they'll move via PIBT, we don't know where)
                reservations.reserve_position(bot.id, bot.position, current_round)

        # Plan each stuck bot
        time_per_bot = self._max_planning_ms / max(len(stuck_bots), 1)
        for bot_id in stuck_bots:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            if elapsed_ms > self._max_planning_ms:
                break

            bot = state.get_bot(bot_id)
            if not bot:
                continue
            assignment = assignments.get(bot_id)
            if not assignment:
                continue

            target = self._get_target(bot.position, assignment, state.drop_off)

            path = find_path_tsa(
                start=bot.position,
                goal=target,
                start_t=current_round,
                grid=self._path._grid,
                reservations=reservations,
                bot_id=bot_id,
                directed_neighbors_fn=self._path._directed_neighbors,
                distance_fn=self._path.distance,
                max_t=self._horizon,
                deadline_ms=min(time_per_bot, self._max_planning_ms - elapsed_ms),
            )

            if path and len(path) > 1:
                result[bot_id] = path[1:]  # Exclude current position
                # Reserve this path so other stuck bots avoid it
                reservations.reserve_path(bot_id, path, current_round)
                logger.info(
                    "TSP: B%d stuck %d rounds at %s, planned %d-step path to %s",
                    bot_id, self._stuck_count[bot_id], bot.position,
                    len(path) - 1, target,
                )
                # Reset stuck counter so we don't spam re-plans
                self._stuck_count[bot_id] = 0

        return result

    def _get_target(self, bot_pos: Pos, assignment: BotAssignment,
                    drop_off: Pos) -> Pos:
        """Determine the navigation target for a bot."""
        if assignment.navigation_override is not None:
            return assignment.navigation_override
        if assignment.task is None:
            return bot_pos
        task = assignment.task
        if task.task_type == TaskType.DELIVER:
            # Use task target (set to nearest zone by planner) if available
            return task.target_pos if task.target_pos else drop_off
        elif task.task_type in (TaskType.PICK_UP, TaskType.PRE_PICK):
            return task.target_pos if task.target_pos else bot_pos
        elif task.task_type == TaskType.IDLE:
            return task.target_pos if task.target_pos else bot_pos
        return bot_pos

    def reset(self) -> None:
        """Reset all planning state for a new game."""
        self._last_positions.clear()
        self._stuck_count.clear()
