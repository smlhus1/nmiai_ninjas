"""
Congestion-Aware Guidance Graph.

Replaces raw BFS distance with congestion-weighted Dijkstra distances.
Traffic patterns are learned from bot movement history and used to route
bots away from congested corridors.

Key concepts:
- vertex_congestion: how many bots have occupied a cell recently
- contraflow: bidirectional traffic on an edge (A->B and B->A both busy)
- guided_distance: Dijkstra shortest path using congestion-weighted edges
"""

from __future__ import annotations

import heapq
from collections import defaultdict

from bot.models import Grid, Pos


class GuidanceGraph:
    """
    Congestion-weighted guidance for PIBT candidate sorting.

    Maintains traffic history and provides guided_distance() that
    accounts for congestion when choosing movement directions.
    """

    def __init__(
        self,
        grid: Grid,
        one_way: dict[Pos, tuple[int, int]] | None = None,
        alpha: float = 2.0,      # vertex congestion weight
        beta: float = 3.0,       # contraflow penalty weight
        decay: float = 0.7,      # exponential decay per update cycle
        update_interval: int = 5,  # rounds between weight updates
    ) -> None:
        self._grid = grid
        self._one_way = one_way or {}
        self._alpha = alpha
        self._beta = beta
        self._decay = decay
        self._update_interval = update_interval

        # Traffic counters (accumulated between updates)
        self._vertex_visits: dict[Pos, float] = defaultdict(float)
        self._edge_flow: dict[tuple[Pos, Pos], float] = defaultdict(float)

        # Computed edge weights (updated periodically)
        self._edge_weight: dict[tuple[Pos, Pos], float] = {}

        # Dijkstra cache: target -> {pos: guided_distance}
        self._distance_cache: dict[Pos, dict[Pos, float]] = {}

        self._round = 0
        self._last_positions: dict[int, Pos] = {}  # bot_id -> last pos

    def on_round(
        self,
        bot_positions: dict[int, Pos],
        round_num: int,
    ) -> None:
        """Record bot positions for this round. Call every round."""
        self._round = round_num

        # Record vertex visits
        for pos in bot_positions.values():
            self._vertex_visits[pos] += 1.0

        # Record edge flow (movement from last position)
        for bot_id, pos in bot_positions.items():
            last = self._last_positions.get(bot_id)
            if last is not None and last != pos:
                self._edge_flow[(last, pos)] += 1.0

        self._last_positions = dict(bot_positions)

        # Periodically update weights and clear cache
        if round_num > 0 and round_num % self._update_interval == 0:
            self._update_weights()

    def _update_weights(self) -> None:
        """Recompute edge weights from traffic data, then decay counters."""
        self._edge_weight.clear()
        self._distance_cache.clear()

        # Build edge weights for all walkable edges
        grid = self._grid
        for x in range(grid.width):
            for y in range(grid.height):
                pos = (x, y)
                if not grid.is_walkable(pos):
                    continue
                for neighbor in self._directed_neighbors(pos):
                    edge = (pos, neighbor)
                    # Base cost
                    w = 1.0

                    # Vertex congestion at destination
                    vc = self._vertex_visits.get(neighbor, 0.0)
                    w += vc * self._alpha

                    # Contraflow: traffic in both directions on this edge
                    forward = self._edge_flow.get((pos, neighbor), 0.0)
                    backward = self._edge_flow.get((neighbor, pos), 0.0)
                    if forward > 0 and backward > 0:
                        w += (forward * backward) * self._beta

                    self._edge_weight[edge] = w

        # Decay counters for next cycle
        decay = self._decay
        for pos in list(self._vertex_visits):
            self._vertex_visits[pos] *= decay
            if self._vertex_visits[pos] < 0.01:
                del self._vertex_visits[pos]

        for edge in list(self._edge_flow):
            self._edge_flow[edge] *= decay
            if self._edge_flow[edge] < 0.01:
                del self._edge_flow[edge]

    def guided_distance(self, start: Pos, target: Pos) -> float:
        """
        Congestion-weighted distance from start to target.

        Uses Dijkstra with edge weights that penalize congested areas.
        Returns float (not int) since weights are fractional.
        Falls back to large value if unreachable.
        """
        if start == target:
            return 0.0

        # No weights computed yet — return 0 so guidance is neutral as tiebreaker
        if not self._edge_weight:
            return 0.0

        if target not in self._distance_cache:
            self._distance_cache[target] = self._dijkstra_from_target(target)

        return self._distance_cache[target].get(start, 9999.0)

    def _dijkstra_from_target(self, target: Pos) -> dict[Pos, float]:
        """
        Reverse Dijkstra from target: compute guided_distance for all reachable cells.

        Uses reverse neighbors (cells that CAN REACH pos) so distances[pos]
        gives the cost FROM pos TO target following one-way rules.
        """
        dist: dict[Pos, float] = {target: 0.0}
        heap: list[tuple[float, Pos]] = [(0.0, target)]

        while heap:
            d, pos = heapq.heappop(heap)
            if d > dist.get(pos, 9999.0):
                continue

            # Reverse: which cells can reach `pos`?
            for neighbor in self._reverse_neighbors(pos):
                edge = (neighbor, pos)
                w = self._edge_weight.get(edge, 1.0)
                nd = d + w
                if nd < dist.get(neighbor, 9999.0):
                    dist[neighbor] = nd
                    heapq.heappush(heap, (nd, neighbor))

        return dist

    def _directed_neighbors(self, pos: Pos) -> list[Pos]:
        """Walkable neighbors respecting one-way rules."""
        x, y = pos
        rule = self._one_way.get(pos)
        neighbors = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if not self._grid.is_walkable((nx, ny)):
                continue
            if rule:
                if rule[1] != 0 and dx == 0 and dy != 0 and dy != rule[1]:
                    continue
                if rule[0] != 0 and dy == 0 and dx != 0 and dx != rule[0]:
                    continue
            neighbors.append((nx, ny))
        return neighbors

    def _reverse_neighbors(self, pos: Pos) -> list[Pos]:
        """Cells that can reach pos following one-way rules."""
        x, y = pos
        neighbors = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if not self._grid.is_walkable((nx, ny)):
                continue
            neighbor = (nx, ny)
            rule = self._one_way.get(neighbor)
            if rule:
                if rule[1] != 0 and (-dx) == 0 and (-dy) != 0 and (-dy) != rule[1]:
                    continue
                if rule[0] != 0 and (-dy) == 0 and (-dx) != 0 and (-dx) != rule[0]:
                    continue
            neighbors.append(neighbor)
        return neighbors

    def congestion_at(self, pos: Pos) -> float:
        """Current congestion value at a cell (for diagnostics)."""
        return self._vertex_visits.get(pos, 0.0)
