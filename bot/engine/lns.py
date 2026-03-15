"""
Large Neighborhood Search (LNS) refinement for PIBT solutions.

After PIBT generates initial collision-free moves, LNS iteratively
improves the solution by:
1. Destroying: selecting a subset of "worst" bots (most WAITs, furthest from target)
2. Repairing: re-running PIBT for only those bots (others frozen)
3. Accepting: keeping the new solution if it's better

Runs within a strict time budget to stay within the 2s response limit.
"""

from __future__ import annotations

import logging
import time

from bot.models import Pos
from bot.engine.pibt import PIBTResolver

logger = logging.getLogger(__name__)


class LNSRefiner:
    """Post-PIBT refinement via Large Neighborhood Search."""

    def __init__(self, budget_ms: int = 500, neighborhood_size: int = 5) -> None:
        self._budget_ms = budget_ms
        self._neighborhood_size = neighborhood_size

    def refine(
        self,
        pibt: PIBTResolver,
        bots: dict[int, Pos],
        targets: dict[int, Pos],
        initial_moves: dict[int, Pos],
        urgency: dict[int, int],
        tiebreak_offset: int,
        idle_bots: set[int],
    ) -> dict[int, Pos]:
        """Iteratively improve PIBT solution within time budget."""
        best = dict(initial_moves)
        best_score = self._score(bots, targets, best)
        iterations = 0

        start = time.monotonic()
        deadline_s = self._budget_ms / 1000.0

        while (time.monotonic() - start) < deadline_s:
            iterations += 1

            # Destroy: select worst bots (waited or moved away from target)
            worst = self._select_worst(bots, targets, best)
            if not worst:
                break

            # Repair: re-run PIBT for subset with others frozen
            candidate = self._repair(
                pibt, bots, targets, best, worst,
                urgency, tiebreak_offset + iterations, idle_bots,
            )

            # Accept if strictly better
            score = self._score(bots, targets, candidate)
            if score < best_score:
                best = candidate
                best_score = score

        if iterations > 0:
            logger.debug(
                "LNS: %d iterations in %.1fms, score %d -> %d",
                iterations, (time.monotonic() - start) * 1000,
                self._score(bots, targets, initial_moves), best_score,
            )

        return best

    def _score(
        self,
        bots: dict[int, Pos],
        targets: dict[int, Pos],
        moves: dict[int, Pos],
    ) -> int:
        """Lower = better. Count of bots that didn't make progress toward target."""
        score = 0
        for bot_id, current in bots.items():
            target = targets.get(bot_id, current)
            if target == current:
                continue  # Already at target, no penalty
            next_pos = moves.get(bot_id, current)
            # Manhattan distance change
            d_before = abs(current[0] - target[0]) + abs(current[1] - target[1])
            d_after = abs(next_pos[0] - target[0]) + abs(next_pos[1] - target[1])
            if d_after >= d_before:
                score += 1  # Didn't make progress
        return score

    def _select_worst(
        self,
        bots: dict[int, Pos],
        targets: dict[int, Pos],
        moves: dict[int, Pos],
    ) -> list[int]:
        """Select bots that didn't make progress (waited or moved away)."""
        stalled: list[tuple[int, int]] = []  # (distance_to_target, bot_id)
        for bot_id, current in bots.items():
            target = targets.get(bot_id, current)
            if target == current:
                continue
            next_pos = moves.get(bot_id, current)
            d_before = abs(current[0] - target[0]) + abs(current[1] - target[1])
            d_after = abs(next_pos[0] - target[0]) + abs(next_pos[1] - target[1])
            if d_after >= d_before:
                stalled.append((d_before, bot_id))

        if not stalled:
            return []

        # Pick the N worst (furthest from target among stalled)
        stalled.sort(reverse=True)
        return [bot_id for _, bot_id in stalled[:self._neighborhood_size]]

    def _repair(
        self,
        pibt: PIBTResolver,
        bots: dict[int, Pos],
        targets: dict[int, Pos],
        current_moves: dict[int, Pos],
        subset: list[int],
        urgency: dict[int, int],
        tiebreak_offset: int,
        idle_bots: set[int],
    ) -> dict[int, Pos]:
        """Re-run PIBT for subset bots, treating others as fixed obstacles.

        Frozen bots are placed at their current_moves positions and marked
        as already-decided (idle + at target) so PIBT treats them as walls.
        """
        subset_set = set(subset)

        # Build combined bot positions: frozen bots at their move destinations,
        # subset bots at their current positions (to be re-planned)
        combined_bots: dict[int, Pos] = {}
        combined_targets: dict[int, Pos] = {}
        combined_urgency: dict[int, int] = {}
        combined_idle: set[int] = set()

        for bot_id in bots:
            if bot_id in subset_set:
                # Re-plan this bot from its current position
                combined_bots[bot_id] = bots[bot_id]
                combined_targets[bot_id] = targets.get(bot_id, bots[bot_id])
                combined_urgency[bot_id] = urgency.get(bot_id, 1)
                if bot_id in idle_bots:
                    combined_idle.add(bot_id)
            else:
                # Frozen: place at decided position, mark as immovable
                frozen_pos = current_moves.get(bot_id, bots[bot_id])
                combined_bots[bot_id] = frozen_pos
                combined_targets[bot_id] = frozen_pos  # At target = won't move
                combined_urgency[bot_id] = 3  # Lowest priority
                combined_idle.add(bot_id)

        # Re-run PIBT with modified tiebreak for variation
        new_positions = pibt.resolve(
            combined_bots, combined_targets,
            tiebreak_offset=tiebreak_offset,
            idle_bots=combined_idle,
            urgency=combined_urgency,
        )

        # Merge: keep frozen bots' original moves, use new moves for subset
        result = dict(current_moves)
        for bot_id in subset:
            if bot_id in new_positions:
                result[bot_id] = new_positions[bot_id]

        return result
