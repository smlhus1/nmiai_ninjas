"""Job-based data models for V3 architecture.

A Job is a complete work unit: pick up 1-3 items, then deliver.
This replaces V1's single-item Task system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from bot.models import Pos


@dataclass
class PickupStep:
    """One item to pick up in a Job."""

    item_type: str
    shelf_pos: Pos      # Shelf position (wall cell where item lives)
    pickup_pos: Pos     # Walkable cell adjacent to shelf (where bot stands)


@dataclass
class Job:
    """Multi-item pickup-deliver sequence.

    A bot with a Job follows this flow:
    1. Navigate to pickups[current_step].pickup_pos
    2. Pick up item (current_step advances)
    3. Repeat until all pickups done
    4. Navigate to delivery_zone
    5. Drop off all items
    6. Job complete
    """

    job_id: str
    order_id: str
    pickups: list[PickupStep]
    delivery_zone: Pos
    priority: int               # 0=completes-order, 1=active, 2=preview
    current_step: int = 0
    assigned_bot: int | None = None

    @property
    def is_delivering(self) -> bool:
        """True if all pickups done, bot should deliver."""
        return self.current_step >= len(self.pickups)

    @property
    def current_target(self) -> Pos:
        """Current navigation target."""
        if self.is_delivering:
            return self.delivery_zone
        return self.pickups[self.current_step].pickup_pos

    @property
    def current_pickup(self) -> Optional[PickupStep]:
        """Current pickup step, or None if delivering."""
        if self.is_delivering:
            return None
        return self.pickups[self.current_step]

    @property
    def remaining_pickups(self) -> int:
        return max(0, len(self.pickups) - self.current_step)


@dataclass
class BotState:
    """Persistent per-bot state for V3."""

    bot_id: int
    position: Pos = (0, 0)
    inventory: tuple[str, ...] = ()
    prev_inventory: tuple[str, ...] = ()
    job: Job | None = None
    idle_since: int = 0
