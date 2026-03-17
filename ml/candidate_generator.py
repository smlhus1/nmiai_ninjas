"""
CandidateGenerator: produces top-K item candidates per bot.

Filters by claimed items, BFS reachability, and distance.
Always includes DELIVER (if inventory > 0) and IDLE.
"""
from __future__ import annotations

from bot.engine.pathfinding import PathEngine
from bot.models import GameState, Pos


# Sentinel actions (not item IDs)
DELIVER = "DELIVER"
IDLE = "IDLE"


class CandidateGenerator:
    """Generate top-K candidate actions per bot."""

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def generate(
        self,
        state: GameState,
        path_engine: PathEngine,
        claimed_items: set[str],
    ) -> dict[int, list[str]]:
        """Return candidate actions per bot.

        Each bot gets up to K item IDs + DELIVER (if has inventory) + IDLE.
        Items already in claimed_items are excluded.
        """
        result: dict[int, list[str]] = {}

        for bot in state.bots:
            candidates: list[str] = []

            # Score all unclaimed items by BFS distance
            scored: list[tuple[int, str]] = []
            for item in state.items:
                if item.id in claimed_items:
                    continue

                d = path_engine.distance(bot.position, item.position)
                if d >= 9999:
                    # Item on shelf — try adjacent walkable cells
                    d = min(
                        (path_engine.distance(bot.position, (item.position[0] + dx, item.position[1] + dy))
                         for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))),
                        default=9999,
                    )
                if d < 9999:
                    scored.append((d, item.id))

            # Sort by distance, take top-K
            scored.sort()
            candidates = [item_id for _, item_id in scored[:self.k]]

            # DELIVER if bot has inventory
            if len(bot.inventory) > 0:
                candidates.append(DELIVER)

            # Always include IDLE
            candidates.append(IDLE)

            result[bot.id] = candidates

        return result
