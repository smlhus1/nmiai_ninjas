"""
GameLogger: observer that records game events for offline analysis.

Captures order transitions, shelf map, and map fingerprint with minimal
coupling to the rest of the bot. Called each round by the Coordinator.

Only logs transitions (order activations, completions), NOT full game state
per round. This keeps log files small and focused.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from bot.models import GameState, OrderStatus, Pos

logger = logging.getLogger(__name__)


class GameLogger:
    """Observer that records game events for recon analysis."""

    def __init__(self) -> None:
        self._fingerprint: str = ""
        self._grid_size: tuple[int, int] = (0, 0)
        self._walls: list[list[int]] = []
        self._drop_off: list[int] = []
        self._shelf_map: dict[str, list[list[int]]] = {}  # type -> [[x, y], ...]
        self._bot_count: int = 0
        self._bot_start_positions: list[list[int]] = []

        # Order tracking
        self._order_sequence: list[dict[str, Any]] = []
        self._known_order_ids: set[str] = set()
        self._active_order_id: str | None = None
        self._order_activation_rounds: dict[str, int] = {}
        self._order_completion_rounds: dict[str, int] = {}

        self._initialized = False

    def on_round(self, state: GameState, shelf_positions: frozenset[Pos]) -> None:
        """Called each round. Records order transitions + shelf map (first round)."""
        if not self._initialized:
            self._initialize(state, shelf_positions)
            self._initialized = True

        self._track_orders(state)

    def _initialize(self, state: GameState, shelf_positions: frozenset[Pos]) -> None:
        """Capture static map data on first round."""
        self._grid_size = (state.grid.width, state.grid.height)
        # Use original walls (without shelves merged in)
        # Shelves are in shelf_positions, walls in state.grid.walls
        # But state.grid.walls already includes shelves at this point,
        # so subtract shelf_positions to get original walls
        original_walls = state.grid.walls - shelf_positions
        self._walls = sorted([list(w) for w in original_walls])
        self._drop_off = list(state.drop_off)
        self._bot_count = len(state.bots)
        self._bot_start_positions = [list(bot.position) for bot in state.bots]

        # Build shelf map: item_type -> shelf positions
        shelf_map: dict[str, list[list[int]]] = {}
        for item in state.items:
            shelf_map.setdefault(item.type, []).append(list(item.position))
        self._shelf_map = shelf_map

        # Compute fingerprint
        self._fingerprint = self._compute_fingerprint(state, original_walls)

        logger.info("RECON: map fingerprint=%s, grid=%dx%d, shelves=%d, bots=%d",
                     self._fingerprint, state.grid.width, state.grid.height,
                     len(shelf_positions), self._bot_count)

    def _compute_fingerprint(
        self, state: GameState, original_walls: frozenset[Pos]
    ) -> str:
        """Compute deterministic map fingerprint from grid geometry."""
        sorted_walls = sorted(original_walls)
        data = f"{state.grid.width}x{state.grid.height}|{sorted_walls}|{state.drop_off}"
        return hashlib.sha256(data.encode()).hexdigest()[:8]

    def _track_orders(self, state: GameState) -> None:
        """Track order appearances, activations, and completions."""
        for order in state.orders:
            # New order seen for the first time
            if order.id not in self._known_order_ids:
                self._known_order_ids.add(order.id)
                self._order_sequence.append({
                    "id": order.id,
                    "items_required": list(order.items_required),
                    "first_seen_round": state.round,
                    "status_when_seen": order.status.value,
                })

            # Track activation (preview -> active)
            if (
                order.status == OrderStatus.ACTIVE
                and order.id not in self._order_activation_rounds
            ):
                self._order_activation_rounds[order.id] = state.round

            # Track completion
            if order.complete and order.id not in self._order_completion_rounds:
                self._order_completion_rounds[order.id] = state.round

        # Track current active order for transition detection
        active = [o for o in state.orders if o.status == OrderStatus.ACTIVE and not o.complete]
        self._active_order_id = active[0].id if active else None

    def finalize(self, total_rounds: int, final_score: int) -> dict:
        """Called at game_over. Returns serializable recon data dict."""
        # Enrich order sequence with activation/completion rounds
        for entry in self._order_sequence:
            oid = entry["id"]
            entry["activated_round"] = self._order_activation_rounds.get(oid)
            entry["completed_round"] = self._order_completion_rounds.get(oid)

        return {
            "fingerprint": self._fingerprint,
            "grid_size": list(self._grid_size),
            "walls": self._walls,
            "drop_off": self._drop_off,
            "shelf_map": self._shelf_map,
            "bot_count": self._bot_count,
            "bot_start_positions": self._bot_start_positions,
            "order_sequence": self._order_sequence,
            "total_rounds": total_rounds,
            "final_score": final_score,
        }

    @property
    def fingerprint(self) -> str:
        return self._fingerprint
