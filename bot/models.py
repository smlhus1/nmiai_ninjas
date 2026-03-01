"""
Immutable data models for game state.

All game state received from the server is parsed into these frozen dataclasses.
They are NEVER mutated — each round creates fresh instances.
This ensures no accidental state leakage between rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --- Primitives ---

Pos = tuple[int, int]  # (x, y) grid coordinate


class Action(str, Enum):
    """All possible bot actions."""

    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    PICK_UP = "pick_up"
    DROP_OFF = "drop_off"
    WAIT = "wait"


class OrderStatus(str, Enum):
    ACTIVE = "active"
    PREVIEW = "preview"


# Direction vectors keyed by action
DIRECTION: dict[Action, Pos] = {
    Action.MOVE_UP: (0, -1),
    Action.MOVE_DOWN: (0, 1),
    Action.MOVE_LEFT: (-1, 0),
    Action.MOVE_RIGHT: (1, 0),
}


def apply_move(pos: Pos, action: Action) -> Pos:
    """Return new position after a move action. Non-move actions return same pos."""
    dx, dy = DIRECTION.get(action, (0, 0))
    return (pos[0] + dx, pos[1] + dy)


# --- Game State (immutable, per-round) ---


@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    walls: frozenset[Pos]

    @classmethod
    def from_dict(cls, data: dict) -> Grid:
        return cls(
            width=data["width"],
            height=data["height"],
            walls=frozenset(tuple(w) for w in data["walls"]),
        )

    def in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, pos: Pos) -> bool:
        return self.in_bounds(pos) and pos not in self.walls


@dataclass(frozen=True)
class Bot:
    id: int
    position: Pos
    inventory: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict) -> Bot:
        return cls(
            id=data["id"],
            position=tuple(data["position"]),
            inventory=tuple(data.get("inventory", [])),
        )


@dataclass(frozen=True)
class Item:
    id: str
    type: str
    position: Pos

    @classmethod
    def from_dict(cls, data: dict) -> Item:
        return cls(
            id=data["id"],
            type=data["type"],
            position=tuple(data["position"]),
        )


@dataclass(frozen=True)
class Order:
    id: str
    items_required: tuple[str, ...]
    items_delivered: tuple[str, ...]
    complete: bool
    status: OrderStatus

    @classmethod
    def from_dict(cls, data: dict) -> Order:
        return cls(
            id=data["id"],
            items_required=tuple(data["items_required"]),
            items_delivered=tuple(data.get("items_delivered", [])),
            complete=data.get("complete", False),
            status=OrderStatus(data.get("status", "active")),
        )

    @property
    def items_remaining(self) -> tuple[str, ...]:
        """Items still needed for this order."""
        remaining = list(self.items_required)
        for item in self.items_delivered:
            if item in remaining:
                remaining.remove(item)
        return tuple(remaining)


@dataclass(frozen=True)
class GameState:
    """Complete game state for a single round. Immutable."""

    round: int
    max_rounds: int
    grid: Grid
    bots: tuple[Bot, ...]
    items: tuple[Item, ...]
    orders: tuple[Order, ...]
    drop_off: Pos
    score: int

    @classmethod
    def from_dict(cls, data: dict) -> GameState:
        return cls(
            round=data["round"],
            max_rounds=data["max_rounds"],
            grid=Grid.from_dict(data["grid"]),
            bots=tuple(Bot.from_dict(b) for b in data["bots"]),
            items=tuple(Item.from_dict(i) for i in data["items"]),
            orders=tuple(Order.from_dict(o) for o in data["orders"]),
            drop_off=tuple(data["drop_off"]),
            score=data.get("score", 0),
        )

    def get_bot(self, bot_id: int) -> Optional[Bot]:
        for bot in self.bots:
            if bot.id == bot_id:
                return bot
        return None

    @property
    def active_orders(self) -> tuple[Order, ...]:
        return tuple(o for o in self.orders if o.status == OrderStatus.ACTIVE and not o.complete)

    @property
    def preview_orders(self) -> tuple[Order, ...]:
        return tuple(o for o in self.orders if o.status == OrderStatus.PREVIEW and not o.complete)

    @property
    def rounds_remaining(self) -> int:
        return self.max_rounds - self.round


# --- Bot Command (output) ---


@dataclass
class BotCommand:
    """A single action for a single bot."""

    bot_id: int
    action: Action
    item_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"bot": self.bot_id, "action": self.action.value}
        if self.item_id is not None:
            d["item_id"] = self.item_id
        return d
