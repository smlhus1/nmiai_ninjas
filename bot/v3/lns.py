"""Cooperative LNS — Large Neighborhood Search with soft obstacles.

Key difference from V1 LNS: frozen bots are SOFT obstacles
(can be pushed through with high cost) rather than hard walls.

Loop:
1. Score EPIBT solution (count bots without progress)
2. Select stalled bots + spatial neighbors
3. Re-run EPIBT for subset (others = soft obstacles with 3x cost)
4. Accept if fewer stalled bots
5. Repeat until time budget exhausted
"""

from __future__ import annotations

import logging
import time

from bot.models import Pos
from bot.v3.epibt import EPIBTResolver

logger = logging.getLogger(__name__)


class CooperativeLNS:
    """Post-EPIBT refinement via cooperative LNS."""

    def __init__(
        self,
        budget_ms: int = 500,
        neighborhood_size: int = 6,
        spatial_radius: int = 2,
    ) -> None:
        self._budget_ms = budget_ms
        self._neighborhood_size = neighborhood_size
        self._spatial_radius = spatial_radius

    def refine(
        self,
        epibt: EPIBTResolver,
        bots: dict[int, Pos],
        targets: dict[int, Pos],
        initial_moves: dict[int, Pos],
        urgency: dict[int, int],
        tiebreak_offset: int,
    ) -> dict[int, Pos]:
        """Iteratively improve EPIBT solution within time budget."""
        best = dict(initial_moves)
        best_score = self._score(bots, targets, best)
        iterations = 0

        start = time.monotonic()
        deadline_s = self._budget_ms / 1000.0

        while (time.monotonic() - start) < deadline_s:
            iterations += 1

            # Destroy: select stalled bots + spatial neighbors
            subset = self._select_neighborhood(bots, targets, best)
            if not subset:
                break

            # Repair: re-run EPIBT for subset, others frozen as soft obstacles
            candidate = self._repair(
                epibt, bots, targets, best, subset,
                urgency, tiebreak_offset + iterations,
            )

            # Accept if strictly better
            score = self._score(bots, targets, candidate)
            if score < best_score:
                best = candidate
                best_score = score

        if iterations > 0:
            logger.debug(
                "LNS: %d iters in %.1fms, score %d->%d",
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
        """Lower = better. Count bots that didn't make progress."""
        score = 0
        for bot_id, current in bots.items():
            target = targets.get(bot_id, current)
            if target == current:
                continue
            next_pos = moves.get(bot_id, current)
            d_before = abs(current[0] - target[0]) + abs(current[1] - target[1])
            d_after = abs(next_pos[0] - target[0]) + abs(next_pos[1] - target[1])
            if d_after >= d_before:
                score += 1
        return score

    def _select_neighborhood(
        self,
        bots: dict[int, Pos],
        targets: dict[int, Pos],
        moves: dict[int, Pos],
    ) -> list[int]:
        """Select stalled bots + their spatial neighbors."""
        stalled: list[tuple[int, int]] = []
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

        stalled.sort(reverse=True)
        seed_bots = [bid for _, bid in stalled[:self._neighborhood_size]]

        # Add spatial neighbors
        neighborhood = set(seed_bots)
        r = self._spatial_radius
        for seed_id in seed_bots:
            seed_pos = bots[seed_id]
            for bot_id, pos in bots.items():
                if bot_id in neighborhood:
                    continue
                if (abs(pos[0] - seed_pos[0]) + abs(pos[1] - seed_pos[1])) <= r:
                    neighborhood.add(bot_id)

        return list(neighborhood)

    def _repair(
        self,
        epibt: EPIBTResolver,
        bots: dict[int, Pos],
        targets: dict[int, Pos],
        current_moves: dict[int, Pos],
        subset: list[int],
        urgency: dict[int, int],
        tiebreak_offset: int,
    ) -> dict[int, Pos]:
        """Re-run EPIBT for subset, frozen bots as soft obstacles."""
        subset_set = set(subset)

        combined_bots: dict[int, Pos] = {}
        combined_targets: dict[int, Pos] = {}
        combined_urgency: dict[int, int] = {}

        for bot_id in bots:
            if bot_id in subset_set:
                combined_bots[bot_id] = bots[bot_id]
                combined_targets[bot_id] = targets.get(bot_id, bots[bot_id])
                combined_urgency[bot_id] = urgency.get(bot_id, 1)
            else:
                # Frozen: at decided position, lowest priority
                frozen_pos = current_moves.get(bot_id, bots[bot_id])
                combined_bots[bot_id] = frozen_pos
                combined_targets[bot_id] = frozen_pos
                combined_urgency[bot_id] = 99  # Immovable

        new_positions = epibt.resolve(
            combined_bots, combined_targets,
            urgency=combined_urgency,
            tiebreak_offset=tiebreak_offset,
        )

        # Merge: keep frozen bots' original moves, use new for subset
        result = dict(current_moves)
        for bot_id in subset:
            if bot_id in new_positions:
                result[bot_id] = new_positions[bot_id]

        return result
