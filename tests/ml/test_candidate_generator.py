"""Tests for CandidateGenerator."""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.engine.pathfinding import PathEngine
from bot.models import Bot, GameState, Grid, Item, Order, OrderStatus
from ml.candidate_generator import DELIVER, IDLE, CandidateGenerator


def _make_state(bots, items, width=16, height=12, drop_off=(1, 10)):
    grid = Grid(width=width, height=height, walls=frozenset())
    return GameState(
        round=10, max_rounds=300, grid=grid,
        bots=tuple(bots), items=tuple(items),
        orders=(), drop_off=drop_off,
        drop_off_zones=(drop_off,), score=0,
    )


def _make_pe(state):
    pe = PathEngine()
    pe.set_grid(state.grid, state.drop_off)
    return pe


class TestCandidateGenerator:
    def test_all_bots_present(self):
        bots = [Bot(id=i, position=(i + 2, 5), inventory=()) for i in range(3)]
        items = [Item(id=f"item_{i}", type="milk", position=(10, i + 3)) for i in range(5)]
        state = _make_state(bots, items)
        pe = _make_pe(state)

        gen = CandidateGenerator(k=5)
        result = gen.generate(state, pe, set())
        assert set(result.keys()) == {0, 1, 2}

    def test_no_claimed_items(self):
        bots = [Bot(id=0, position=(5, 5), inventory=())]
        items = [Item(id=f"item_{i}", type="milk", position=(6 + i, 5)) for i in range(3)]
        state = _make_state(bots, items)
        pe = _make_pe(state)

        gen = CandidateGenerator(k=5)
        claimed = {"item_0", "item_1"}
        result = gen.generate(state, pe, claimed)

        # Only item_2 should be a candidate
        item_candidates = [c for c in result[0] if c not in (DELIVER, IDLE)]
        assert "item_0" not in item_candidates
        assert "item_1" not in item_candidates
        assert "item_2" in item_candidates

    def test_deliver_only_with_inventory(self):
        bots = [
            Bot(id=0, position=(5, 5), inventory=()),
            Bot(id=1, position=(7, 5), inventory=("milk",)),
        ]
        items = [Item(id="item_0", type="milk", position=(8, 5))]
        state = _make_state(bots, items)
        pe = _make_pe(state)

        gen = CandidateGenerator(k=5)
        result = gen.generate(state, pe, set())

        assert DELIVER not in result[0]  # empty inventory
        assert DELIVER in result[1]      # has inventory

    def test_idle_always_present(self):
        bots = [Bot(id=0, position=(5, 5), inventory=())]
        items = [Item(id="item_0", type="milk", position=(8, 5))]
        state = _make_state(bots, items)
        pe = _make_pe(state)

        gen = CandidateGenerator(k=5)
        result = gen.generate(state, pe, set())
        assert IDLE in result[0]

    def test_k_limit(self):
        bots = [Bot(id=0, position=(5, 5), inventory=())]
        items = [Item(id=f"item_{i}", type="milk", position=(6 + i, 5)) for i in range(20)]
        state = _make_state(bots, items)
        pe = _make_pe(state)

        gen = CandidateGenerator(k=3)
        result = gen.generate(state, pe, set())

        item_candidates = [c for c in result[0] if c not in (DELIVER, IDLE)]
        assert len(item_candidates) == 3

    def test_sorted_by_distance(self):
        bots = [Bot(id=0, position=(5, 5), inventory=())]
        items = [
            Item(id="far", type="milk", position=(14, 5)),
            Item(id="near", type="milk", position=(6, 5)),
            Item(id="mid", type="milk", position=(9, 5)),
        ]
        state = _make_state(bots, items)
        pe = _make_pe(state)

        gen = CandidateGenerator(k=3)
        result = gen.generate(state, pe, set())

        item_candidates = [c for c in result[0] if c not in (DELIVER, IDLE)]
        assert item_candidates[0] == "near"
        assert item_candidates[1] == "mid"
        assert item_candidates[2] == "far"
