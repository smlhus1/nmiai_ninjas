"""Grid and map data model parsed from recon JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

Pos = tuple[int, int]


@dataclass
class Grid:
    """Grid with walls, shelves, and optional one-way rules."""

    width: int
    height: int
    walls: frozenset[Pos]
    shelves: frozenset[Pos]
    one_way: dict[Pos, tuple[int, int]] | None = None

    def __post_init__(self):
        if self.one_way is None:
            self.one_way = {}

    @property
    def obstacles(self) -> frozenset[Pos]:
        return self.walls | self.shelves

    def walkable(self, pos: Pos) -> bool:
        x, y = pos
        return (
            0 <= x < self.width
            and 0 <= y < self.height
            and pos not in self.obstacles
        )

    def neighbors(self, pos: Pos, respect_one_way: bool = True) -> list[Pos]:
        """Return walkable cardinal neighbors of pos, respecting one-way rules."""
        x, y = pos
        rule = self.one_way.get(pos) if respect_one_way else None
        result = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nb = (x + dx, y + dy)
            if not self.walkable(nb):
                continue
            if rule:
                # one_way rule: (dx_allowed, dy_allowed). 0 means no restriction on that axis.
                if rule[1] != 0 and dx == 0 and dy != 0 and dy != rule[1]:
                    continue
                if rule[0] != 0 and dy == 0 and dx != 0 and dx != rule[0]:
                    continue
            result.append(nb)
        return result

    def walkable_cells(self) -> list[Pos]:
        """Return all walkable positions on the grid."""
        return [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.walkable((x, y))
        ]


def pickup_positions(grid: Grid, shelf_pos: Pos) -> list[Pos]:
    """Return walkable neighbors adjacent to a shelf (valid pickup spots)."""
    x, y = shelf_pos
    result = []
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nb = (x + dx, y + dy)
        if grid.walkable(nb):
            result.append(nb)
    return result


@dataclass(frozen=True)
class GameMap:
    """Complete map data: grid, drop-off zones, shelf locations, spawn."""

    grid: Grid
    drop_off_zones: tuple[Pos, ...]
    shelf_map: dict[str, list[Pos]]  # item_type -> list of shelf positions
    spawn: Pos
    bot_count: int

    @staticmethod
    def from_recon(path: str) -> GameMap:
        """Load a GameMap from a recon JSON file."""
        with open(path) as f:
            data = json.load(f)

        width, height = data["grid_size"]

        walls = frozenset(tuple(w) for w in data["walls"])

        # Collect all shelf positions from shelf_map
        shelf_map: dict[str, list[Pos]] = {}
        all_shelf_positions: set[Pos] = set()
        for item_type, positions in data["shelf_map"].items():
            typed_positions = [tuple(p) for p in positions]
            shelf_map[item_type] = typed_positions
            all_shelf_positions.update(typed_positions)

        shelves = frozenset(all_shelf_positions)

        grid = Grid(
            width=width,
            height=height,
            walls=walls,
            shelves=shelves,
        )

        drop_off_zones = tuple(tuple(z) for z in data["drop_off_zones"])

        # Spawn = first bot start position
        spawn = tuple(data["bot_start_positions"][0])

        bot_count = data["bot_count"]

        # One-way disabled for solver (PIBT handles congestion, one-way adds distance)
        # one_way = GameMap._nightmare_one_way(grid) if width >= 30 else {}

        return GameMap(
            grid=grid,
            drop_off_zones=drop_off_zones,
            shelf_map=shelf_map,
            spawn=spawn,
            bot_count=bot_count,
        )

    @staticmethod
    def _nightmare_one_way(grid: Grid) -> dict[Pos, tuple[int, int]]:
        """One-way rules for nightmare map.

        y=16: LEFT only (delivery lane toward drop-offs)
        Vertical aisles: alternate DOWN/UP
        Cross-corridors (y=1, y=9, y=15): free (no restriction)
        """
        one_way: dict[Pos, tuple[int, int]] = {}

        # Detect vertical aisles (columns where most cells are walkable)
        aisle_cols = []
        for x in range(grid.width):
            walkable_count = sum(1 for y in range(2, 15) if grid.walkable((x, y)))
            if walkable_count >= 5:
                aisle_cols.append(x)

        cross_rows = {1, 9, 15, 16}

        # y=16: LEFT only (delivery lane)
        for x in range(1, grid.width - 1):
            if grid.walkable((x, 16)):
                one_way[(x, 16)] = (-1, 0)

        # Vertical aisles: alternate DOWN/UP
        for i, ax in enumerate(sorted(aisle_cols)):
            dy = 1 if (i % 2 == 0) else -1  # even=DOWN, odd=UP
            for y in range(1, 16):
                if y in cross_rows:
                    continue
                if grid.walkable((ax, y)):
                    one_way[(ax, y)] = (0, dy)

        return one_way
