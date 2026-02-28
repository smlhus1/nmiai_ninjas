"""
Task definitions — the unit of work assigned to a bot.

A Task is a high-level goal: "go pick up item X" or "go deliver items".
Tasks are created by the TaskPlanner and consumed by the ActionResolver.

Tasks persist between rounds (a bot keeps its task until completed or
reassigned). This prevents flip-flopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from bot.models import Pos


class TaskType(Enum):
    PICK_UP = auto()  # Go to item, pick it up
    DELIVER = auto()  # Go to drop-off, deliver inventory
    PRE_PICK = auto()  # Pre-pick preview order item (auto-delivery on order transition)
    IDLE = auto()  # No useful work — stay put or wander


@dataclass
class Task:
    """
    A unit of work for a bot.

    task_type: what to do
    target_pos: where to go
    item_id: which item to pick up (PICK_UP only)
    item_type: what type of item (for logging/debugging)
    order_id: which order this contributes to (for coordination)
    """

    task_type: TaskType
    target_pos: Pos          # Where the bot should navigate TO (walkable cell)
    item_id: Optional[str] = None
    item_type: Optional[str] = None
    item_pos: Optional[Pos] = None  # Where the item actually IS (shelf, may not be walkable)
    order_id: Optional[str] = None

    def __repr__(self) -> str:
        if self.task_type == TaskType.PICK_UP:
            return f"Task(PICK_UP {self.item_type} @ {self.target_pos})"
        elif self.task_type == TaskType.DELIVER:
            return f"Task(DELIVER @ {self.target_pos})"
        return "Task(IDLE)"


@dataclass
class BotAssignment:
    """
    Persistent state for a single bot across rounds.
    Owned by the Coordinator, updated each round.
    """

    bot_id: int
    task: Optional[Task] = None
    path: list[Pos] | None = None  # Pre-computed path to target
    navigation_override: Optional[Pos] = None  # Temporary nav target (e.g. staging)

    def clear(self) -> None:
        self.task = None
        self.path = None
        self.navigation_override = None

    @property
    def effective_target(self) -> Optional[Pos]:
        """Navigation target: override if set, otherwise task target."""
        if self.navigation_override is not None:
            return self.navigation_override
        return self.task.target_pos if self.task else None

    @property
    def has_task(self) -> bool:
        return self.task is not None and self.task.task_type != TaskType.IDLE
