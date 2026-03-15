"""SoftGuidance — asymmetric edge weights for traffic flow.

Replaces hard one-way rules with soft cost-based guidance.
Bots CAN go against flow — it just costs more.

Flow directions computed from BFS toward drop-off zones.
Congestion learned from bot movement patterns.
"""

from __future__ import annotations

from collections import defaultdict, deque

from bot.models import Grid, Pos


class SoftGuidance:
    """Soft traffic guidance using asymmetric edge weights.

    Key difference from hard one-way: bots are never BLOCKED,
    just steered via cost. This gives flexibility while maintaining
    traffic flow.
    """

    def __init__(
        self,
        grid: Grid,
        drop_off_zones: tuple[Pos, ...],
        *,
        with_flow_weight: float = 0.8,
        against_flow_weight: float = 2.5,
        congestion_alpha: float = 0.3,
        update_interval: int = 3,
        decay: float = 0.7,
    ) -> None:
        self._grid = grid
        self._zones = drop_off_zones
        self._with_flow = with_flow_weight
        self._against_flow = against_flow_weight
        self._congestion_alpha = congestion_alpha
        self._update_interval = update_interval
        self._decay = decay

        # Flow direction per cell: (dx, dy) toward nearest drop-off
        self._flow: dict[Pos, tuple[int, int]] = {}

        # Base edge weights (from flow analysis, static)
        self._base_weight: dict[tuple[Pos, Pos], float] = {}

        # Live edge weights (base + congestion, updated periodically)
        self._edge_weight: dict[tuple[Pos, Pos], float] = {}

        # Congestion tracking
        self._vertex_visits: dict[Pos, float] = defaultdict(float)
        self._last_positions: dict[int, Pos] = {}

        # Compute flow and base weights
        self._compute_flow()
        self._compute_base_weights()
        self._edge_weight = dict(self._base_weight)

    def _compute_flow(self) -> None:
        """Multi-source BFS from all drop-off zones to compute flow directions."""
        grid = self._grid
        dist: dict[Pos, int] = {}
        parent: dict[Pos, Pos] = {}
        queue: deque[Pos] = deque()

        for zone in self._zones:
            if grid.is_walkable(zone):
                dist[zone] = 0
                queue.append(zone)

        while queue:
            pos = queue.popleft()
            d = dist[pos]
            x, y = pos
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                n = (x + dx, y + dy)
                if n not in dist and grid.is_walkable(n):
                    dist[n] = d + 1
                    parent[n] = pos
                    queue.append(n)

        # Flow direction: toward BFS parent (toward nearest drop-off)
        for pos, par in parent.items():
            dx = par[0] - pos[0]
            dy = par[1] - pos[1]
            self._flow[pos] = (dx, dy)

    def _compute_base_weights(self) -> None:
        """Compute base asymmetric edge weights from flow directions."""
        grid = self._grid
        for x in range(grid.width):
            for y in range(grid.height):
                pos = (x, y)
                if not grid.is_walkable(pos):
                    continue
                flow = self._flow.get(pos)
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    n = (x + dx, y + dy)
                    if not grid.is_walkable(n):
                        continue

                    if flow:
                        if (dx, dy) == flow:
                            w = self._with_flow      # With flow: cheap
                        elif (dx, dy) == (-flow[0], -flow[1]):
                            w = self._against_flow    # Against flow: expensive
                        else:
                            w = 1.0                   # Perpendicular: neutral
                    else:
                        w = 1.0

                    self._base_weight[(pos, n)] = w

    def edge_weight(self, from_pos: Pos, to_pos: Pos) -> float:
        """Cost of moving from from_pos to to_pos."""
        return self._edge_weight.get((from_pos, to_pos), 1.0)

    def on_round(self, bot_positions: dict[int, Pos], round_num: int) -> None:
        """Update congestion tracking. Call every round."""
        for pos in bot_positions.values():
            self._vertex_visits[pos] += 1.0

        self._last_positions = dict(bot_positions)

        if round_num > 0 and round_num % self._update_interval == 0:
            self._update_weights()

    def _update_weights(self) -> None:
        """Recompute edge weights: base + congestion."""
        self._edge_weight.clear()

        for edge, base_w in self._base_weight.items():
            _, to_pos = edge
            congestion = self._vertex_visits.get(to_pos, 0.0)
            w = base_w + congestion * self._congestion_alpha
            w = max(0.5, min(5.0, w))  # Clamp
            self._edge_weight[edge] = w

        # Decay congestion
        for pos in list(self._vertex_visits):
            self._vertex_visits[pos] *= self._decay
            if self._vertex_visits[pos] < 0.01:
                del self._vertex_visits[pos]
