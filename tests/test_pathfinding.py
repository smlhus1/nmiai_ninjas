"""Tests for pathfinding engine."""

from bot.models import Grid
from bot.engine.pathfinding import PathEngine


def _simple_grid(width: int = 10, height: int = 10, walls: list | None = None) -> Grid:
    return Grid(width=width, height=height, walls=frozenset(walls or []))


def test_distance_same_position():
    engine = PathEngine()
    engine.set_grid(_simple_grid())
    assert engine.distance((0, 0), (0, 0)) == 0


def test_distance_straight_line():
    engine = PathEngine()
    engine.set_grid(_simple_grid())
    assert engine.distance((0, 0), (3, 0)) == 3
    assert engine.distance((0, 0), (0, 5)) == 5


def test_distance_around_wall():
    # Wall blocking direct horizontal path
    walls = [(2, 0), (2, 1), (2, 2)]
    engine = PathEngine()
    engine.set_grid(_simple_grid(walls=walls))
    # Must go around the wall
    d = engine.distance((0, 0), (3, 0))
    assert d > 3  # Can't go straight


def test_find_path_basic():
    engine = PathEngine()
    engine.set_grid(_simple_grid())
    path = engine.find_path((0, 0), (3, 0))
    assert path[0] == (0, 0)
    assert path[-1] == (3, 0)
    assert len(path) == 4  # 3 steps + start


def test_find_path_with_obstacles():
    engine = PathEngine()
    engine.set_grid(_simple_grid())
    engine.new_round(obstacles={(1, 0)})
    path = engine.find_path((0, 0), (2, 0))
    # Should avoid (1,0) and go around
    assert (1, 0) not in path[1:-1]  # obstacle not in middle of path


def test_find_path_unreachable():
    # Completely walled off
    walls = [(1, 0), (0, 1), (1, 1)]
    engine = PathEngine()
    engine.set_grid(_simple_grid(width=3, height=2, walls=walls))
    path = engine.find_path((0, 0), (2, 1))
    assert path == []  # No path


def test_distance_cache_reuse():
    engine = PathEngine()
    engine.set_grid(_simple_grid())
    # Same destination should use cache
    d1 = engine.distance((0, 0), (5, 5))
    d2 = engine.distance((1, 1), (5, 5))
    # Both should work (cache is per-destination)
    assert d1 == 10
    assert d2 == 8


def test_manhattan():
    engine = PathEngine()
    assert engine.manhattan((0, 0), (3, 4)) == 7
    assert engine.manhattan((5, 5), (5, 5)) == 0
