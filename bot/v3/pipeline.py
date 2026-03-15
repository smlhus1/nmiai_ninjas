"""OrderPipeline — state machine for order overlap.

Manages the transition between active and preview orders,
enabling preview pre-picking during the DRAINING phase.

States:
- FILLING: bots working on active order items
- DRAINING: active order almost done (<=2 items left), idle bots start preview
- TRANSITIONING: active order just completed, preview -> active
"""

from __future__ import annotations

from enum import Enum, auto

from bot.models import GameState


class PipelinePhase(Enum):
    FILLING = auto()
    DRAINING = auto()
    TRANSITIONING = auto()


class OrderPipeline:
    """Order overlap state machine."""

    def __init__(self) -> None:
        self._phase = PipelinePhase.FILLING
        self._active_order_id: str | None = None

    def update(self, state: GameState) -> PipelinePhase:
        """Update pipeline state based on current game state."""
        active = state.active_orders

        if not active:
            self._phase = PipelinePhase.TRANSITIONING
            return self._phase

        order = active[0]

        # Detect order transition (new active order)
        if self._active_order_id is not None and self._active_order_id != order.id:
            self._phase = PipelinePhase.TRANSITIONING
            self._active_order_id = order.id
            return self._phase

        self._active_order_id = order.id
        remaining = len(order.items_remaining)

        if remaining <= 2:
            self._phase = PipelinePhase.DRAINING
        else:
            self._phase = PipelinePhase.FILLING

        return self._phase

    @property
    def phase(self) -> PipelinePhase:
        return self._phase

    @property
    def allow_preview(self) -> bool:
        """True if bots can start preview order work."""
        return self._phase == PipelinePhase.DRAINING

    @property
    def is_transitioning(self) -> bool:
        return self._phase == PipelinePhase.TRANSITIONING
