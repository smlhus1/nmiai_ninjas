"""
DropoffScheduler: pre-schedules delivery time slots for multiple bots.

Ensures bots arrive at drop-off sequentially, preventing congestion.
Each bot gets a delivery round — it should arrive AT or BEFORE that round
and perform DROP_OFF on that round.
"""

from __future__ import annotations

from dataclasses import dataclass

from bot.models import Pos
from bot.engine.reservation import ReservationTable


@dataclass
class DeliverySlot:
    """A scheduled delivery for one bot."""
    bot_id: int
    arrival_round: int   # Earliest round bot can reach drop-off
    delivery_round: int  # Round bot should deliver (may wait if earlier bots ahead)


class DropoffScheduler:
    """Schedules sequential drop-off deliveries to prevent congestion."""

    def schedule(
        self,
        deliverers: list[tuple[int, int]],  # (bot_id, estimated_arrival_round)
        reservations: ReservationTable,
        drop_off: Pos,
        current_round: int,
    ) -> list[DeliverySlot]:
        """
        Assign delivery time slots.

        Args:
            deliverers: List of (bot_id, estimated_arrival) sorted by priority
            reservations: Current reservation table
            drop_off: Drop-off position
            current_round: Current game round

        Returns:
            List of DeliverySlot with assigned delivery rounds.
        """
        if not deliverers:
            return []

        # Sort by estimated arrival (earliest first)
        sorted_deliverers = sorted(deliverers, key=lambda x: x[1])

        slots: list[DeliverySlot] = []
        next_available = current_round  # Earliest round drop-off is free

        for bot_id, est_arrival in sorted_deliverers:
            # Bot can't deliver before it arrives
            delivery_round = max(est_arrival, next_available)

            slots.append(DeliverySlot(
                bot_id=bot_id,
                arrival_round=est_arrival,
                delivery_round=delivery_round,
            ))

            # Next bot must wait at least 1 round after this delivery
            # (only one DROP_OFF action succeeds per round at same cell)
            next_available = delivery_round + 1

        return slots
