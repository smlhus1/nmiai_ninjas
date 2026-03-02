"""
ActionResolver: converts Tasks + Paths into concrete per-round Actions.

This is the tactical layer — it takes the strategic decisions from TaskPlanner
and turns them into the actual move/pick_up/drop_off actions for this round.

Uses PIBT (Priority Inheritance with Backtracking) for collision-free movement.
"""

from __future__ import annotations

import logging

from bot.models import (
    Action,
    Bot,
    BotCommand,
    GameState,
    Pos,
)
from typing import Optional
from bot.engine.pathfinding import PathEngine
from bot.engine.pibt import PIBTResolver
from bot.strategy.task import BotAssignment, TaskType

logger = logging.getLogger(__name__)


def _resolve_camp_item(
    state: GameState,
    shelf_pos: Pos,
    item_type: str | None,
) -> str | None:
    """Find an actual item at a shelf position for camp-style picks."""
    for item in state.items:
        if item.position == shelf_pos and (item_type is None or item.type == item_type):
            return item.id
    return None


class ActionResolver:
    """
    Resolves assignments into concrete actions for one round.
    Uses PIBT for collision-free cooperative movement.
    """

    def __init__(self, path_engine: PathEngine) -> None:
        self._path = path_engine

    def resolve(
        self,
        state: GameState,
        assignments: dict[int, BotAssignment],
    ) -> list[BotCommand]:
        """
        Resolve all bot assignments into commands for this round.
        """
        commands: dict[int, BotCommand] = {}

        # Step 1: Resolve immediate actions (pick_up / drop_off) — no movement needed
        movement_bots: dict[int, Bot] = {}
        movement_targets: dict[int, Pos] = {}

        sorted_bots = sorted(state.bots, key=lambda b: b.id)

        for bot in sorted_bots:
            assignment = assignments.get(bot.id)
            if assignment is None:
                commands[bot.id] = BotCommand(bot.id, Action.WAIT)
                continue

            task = assignment.task
            if task is None or task.task_type == TaskType.IDLE:
                # IDLE bot: use navigation_override if set, else current position
                idle_target = bot.position
                if assignment.navigation_override:
                    idle_target = assignment.navigation_override
                movement_bots[bot.id] = bot
                movement_targets[bot.id] = idle_target
                continue

            if task.task_type in (TaskType.PICK_UP, TaskType.PRE_PICK):
                if task.item_pos and task.item_id:
                    if (self._path.manhattan(bot.position, task.item_pos) == 1
                            and len(bot.inventory) < 3):
                        item_id = task.item_id
                        if item_id.startswith("camp_"):
                            item_id = _resolve_camp_item(
                                state, task.item_pos, task.item_type
                            )
                        if item_id:
                            commands[bot.id] = BotCommand(
                                bot.id, Action.PICK_UP, item_id=item_id
                            )
                            continue

            # DELIVER: at drop-off? (use state.drop_off, not task.target_pos)
            if task.task_type == TaskType.DELIVER:
                if bot.position == state.drop_off:
                    # Only issue drop_off if bot has items matching active order
                    active = state.active_orders
                    if active:
                        remaining = list(active[0].items_remaining)
                        has_match = any(inv in remaining for inv in bot.inventory)
                    else:
                        has_match = bool(bot.inventory)
                    if has_match:
                        commands[bot.id] = BotCommand(bot.id, Action.DROP_OFF)
                        continue
                    # No matching items — move away, let PIBT handle
                    movement_bots[bot.id] = bot
                    movement_targets[bot.id] = assignment.effective_target or task.target_pos
                    continue

            # This bot needs to move — collect for PIBT
            target = assignment.effective_target or task.target_pos
            movement_bots[bot.id] = bot
            movement_targets[bot.id] = target

        # Step 1.5: Add stationary bots (pick_up/drop_off) to PIBT as obstacles.
        # Without this, PIBT doesn't know these positions are occupied and sends
        # moving bots into them, causing collisions in the game engine.
        for bot in sorted_bots:
            if bot.id in commands and bot.id not in movement_bots:
                movement_bots[bot.id] = bot
                movement_targets[bot.id] = bot.position  # Stay in place

        # Step 2: Run PIBT for all moving bots
        if movement_bots:
            pibt = PIBTResolver(state.grid, self._path.distance, self._path.corridors)
            bot_positions = {bid: bot.position for bid, bot in movement_bots.items()}

            # Identify IDLE bots (lowest PIBT priority) and DELIVER bots (highest)
            idle_bot_ids: set[int] = set()
            high_priority_ids: set[int] = set()
            critical_bot_ids: set[int] = set()  # Order-completing deliverers
            for bot_id in movement_bots:
                assignment = assignments.get(bot_id)
                if assignment:
                    task = assignment.task
                    if task is None or task.task_type == TaskType.IDLE:
                        idle_bot_ids.add(bot_id)
                    elif task.task_type == TaskType.DELIVER and len(state.bots) >= 4:
                        high_priority_ids.add(bot_id)
                # Stationary bots (pick_up/drop_off added as obstacles) are also idle-priority
                if bot_id in commands:
                    idle_bot_ids.add(bot_id)

            # Detect order-completing deliverers: bots whose delivery would
            # complete the active order (huge +5 bonus). Give them absolute
            # highest PIBT priority so they punch through congestion.
            active = state.active_orders
            if active and len(state.bots) >= 4:
                remaining = list(active[0].items_remaining)
                for bot_id in high_priority_ids:
                    bot = movement_bots[bot_id]
                    # Check if this bot's matching items cover ALL remaining
                    check_remaining = list(remaining)
                    for inv_item in bot.inventory:
                        if inv_item in check_remaining:
                            check_remaining.remove(inv_item)
                    if not check_remaining:
                        critical_bot_ids.add(bot_id)

            next_positions = pibt.resolve(
                bot_positions, movement_targets,
                tiebreak_offset=state.round,
                idle_bots=idle_bot_ids,
                high_priority_bots=high_priority_ids,
                critical_bots=critical_bot_ids,
            )

            # Step 3: Convert positions to actions (skip bots that already have commands)
            for bot_id, next_pos in next_positions.items():
                if bot_id in commands:
                    continue  # Stationary bot — keep pick_up/drop_off command
                bot = movement_bots[bot_id]
                action = self._pos_to_action(bot.position, next_pos)
                commands[bot_id] = BotCommand(bot_id, action)

                # Update cached path
                assignment = assignments.get(bot_id)
                if assignment:
                    assignment.path = None  # PIBT computes fresh each round

        # Build ordered result
        return [commands[bot.id] for bot in sorted_bots if bot.id in commands]

    @staticmethod
    def _pos_to_action(from_pos: Pos, to_pos: Pos) -> Action:
        """Convert a position change to a movement action."""
        """Convert a position change to a movement action."""
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
