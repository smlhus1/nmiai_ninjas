"""Tests for PIBT collision resolution."""

from bot.models import Grid
from bot.engine.pathfinding import PathEngine
from bot.engine.pibt import PIBTResolver


def _grid(width=10, height=10, walls=None):
    return Grid(width=width, height=height, walls=frozenset(walls or []))


def _distance_fn(grid):
    engine = PathEngine()
    engine.set_grid(grid)
    return engine.distance


def test_head_on_deadlock_resolved():
    """Two bots facing each other in a corridor should not deadlock."""
    # Narrow corridor: row y=0 is open, walls above and below
    walls = [(x, 1) for x in range(10)]
    grid = _grid(walls=walls)
    dist = _distance_fn(grid)

    pibt = PIBTResolver(grid, dist)
    # Bot 0 at (2,0) wants to go to (4,0), Bot 1 at (3,0) wants to go to (1,0)
    bots = {0: (2, 0), 1: (3, 0)}
    targets = {0: (4, 0), 1: (1, 0)}

    result = pibt.resolve(bots, targets)

    # Both bots should get positions (no crash)
    assert 0 in result
    assert 1 in result
    # No collision: different positions
    assert result[0] != result[1]


def test_all_bots_get_positions():
    """Every bot should get a position, even in crowded scenarios."""
    grid = _grid(5, 5)
    dist = _distance_fn(grid)

    pibt = PIBTResolver(grid, dist)
    bots = {i: (i, 0) for i in range(5)}
    targets = {i: (4 - i, 0) for i in range(5)}

    result = pibt.resolve(bots, targets)

    assert len(result) == 5
    # All positions should be unique (no collisions)
    positions = list(result.values())
    assert len(set(positions)) == len(positions)


def test_no_collisions():
    """PIBT should never produce collisions."""
    grid = _grid(6, 6)
    dist = _distance_fn(grid)

    pibt = PIBTResolver(grid, dist)
    # 4 bots all wanting to go to the same target
    bots = {0: (0, 0), 1: (5, 0), 2: (0, 5), 3: (5, 5)}
    targets = {0: (3, 3), 1: (3, 3), 2: (3, 3), 3: (3, 3)}

    result = pibt.resolve(bots, targets)

    positions = list(result.values())
    assert len(set(positions)) == len(positions), "PIBT produced collisions!"


def test_stationary_bot_stays():
    """Bot at its target should stay put."""
    grid = _grid(5, 5)
    dist = _distance_fn(grid)

    pibt = PIBTResolver(grid, dist)
    bots = {0: (2, 2)}
    targets = {0: (2, 2)}

    result = pibt.resolve(bots, targets)
    assert result[0] == (2, 2)


def test_single_bot_moves_toward_target():
    """Single bot should move toward its target."""
    grid = _grid(10, 10)
    dist = _distance_fn(grid)

    pibt = PIBTResolver(grid, dist)
    bots = {0: (0, 0)}
    targets = {0: (5, 0)}

    result = pibt.resolve(bots, targets)
    # Should move right (closer to target)
    assert result[0] == (1, 0)


def test_narrow_corridor_three_bots():
    """Three bots in a 1-wide corridor should resolve without deadlock."""
    # Only y=0 is walkable
    walls = [(x, 1) for x in range(8)]
    grid = _grid(8, 2, walls=walls)
    dist = _distance_fn(grid)

    pibt = PIBTResolver(grid, dist)
    bots = {0: (1, 0), 1: (3, 0), 2: (5, 0)}
    targets = {0: (6, 0), 1: (0, 0), 2: (2, 0)}

    result = pibt.resolve(bots, targets)

    # All unique
    positions = list(result.values())
    assert len(set(positions)) == len(positions)
    # All within bounds
    for pos in positions:
        assert grid.is_walkable(pos)
