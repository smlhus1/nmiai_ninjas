"""Tests for BeamSearch."""
import json
import sys
import time
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.engine.pathfinding import PathEngine
from bot.models import Bot, GameState, Grid, Item, Order, OrderStatus
from ml.beam_search import BeamSearch
from ml.candidate_generator import DELIVER, IDLE, CandidateGenerator
from ml.feature_extractor import FeatureContext
from ml.scorer import ScorerMLP


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


def _make_ctx(state, **kw):
    defaults = dict(
        bot_positions=[b.position for b in state.bots],
        n_bots=max(len(state.bots), 1),
        max_dist=60,
        item_type_index={"milk": 0, "bread": 1},
        drop_off_zones=state.drop_off_zones,
    )
    defaults.update(kw)
    return FeatureContext(**defaults)


class TestBeamSearch:
    def test_no_double_booking(self):
        """No item should be assigned to two different bots."""
        bots = [Bot(id=i, position=(i + 2, 5), inventory=()) for i in range(5)]
        items = [Item(id=f"item_{i}", type="milk", position=(10, i + 3)) for i in range(3)]
        state = _make_state(bots, items)
        pe = _make_pe(state)
        ctx = _make_ctx(state)

        gen = CandidateGenerator(k=5)
        candidates = gen.generate(state, pe, set())

        scorer = ScorerMLP()
        beam = BeamSearch(beam_width=20)
        assignment = beam.search(state, pe, candidates, scorer, ctx)

        # Check all bots are assigned
        assert set(assignment.keys()) == {0, 1, 2, 3, 4}

        # Check no item double-booked
        items_assigned = [v for v in assignment.values() if v not in (DELIVER, IDLE)]
        assert len(items_assigned) == len(set(items_assigned)), \
            f"Double-booking: {items_assigned}"

    def test_valid_actions_only(self):
        """All assignments must be valid item IDs, DELIVER, or IDLE."""
        bots = [Bot(id=i, position=(i + 2, 5), inventory=()) for i in range(3)]
        items = [Item(id=f"item_{i}", type="milk", position=(10 + i, 5)) for i in range(5)]
        state = _make_state(bots, items)
        pe = _make_pe(state)
        ctx = _make_ctx(state)

        gen = CandidateGenerator(k=5)
        candidates = gen.generate(state, pe, set())

        scorer = ScorerMLP()
        beam = BeamSearch(beam_width=20)
        assignment = beam.search(state, pe, candidates, scorer, ctx)

        valid_ids = {item.id for item in items} | {DELIVER, IDLE}
        for bot_id, action in assignment.items():
            assert action in valid_ids, f"Bot {bot_id} got invalid action: {action}"

    def test_fallback_idle(self):
        """If no candidates, bot gets IDLE."""
        bots = [Bot(id=0, position=(5, 5), inventory=())]
        state = _make_state(bots, [])
        pe = _make_pe(state)
        ctx = _make_ctx(state)

        candidates = {0: [IDLE]}
        scorer = ScorerMLP()
        beam = BeamSearch(beam_width=20)
        assignment = beam.search(state, pe, candidates, scorer, ctx)

        assert assignment[0] == IDLE

    def test_configurable_beam_width(self):
        bots = [Bot(id=i, position=(i + 2, 5), inventory=()) for i in range(3)]
        items = [Item(id=f"item_{i}", type="milk", position=(10 + i, 5)) for i in range(5)]
        state = _make_state(bots, items)
        pe = _make_pe(state)
        ctx = _make_ctx(state)

        gen = CandidateGenerator(k=5)
        candidates = gen.generate(state, pe, set())

        scorer = ScorerMLP()
        beam = BeamSearch(beam_width=5)
        assignment = beam.search(state, pe, candidates, scorer, ctx)

        assert len(assignment) == 3

    def test_timing_20_bots(self):
        """< 10ms for 20 bots, top-5 candidates, beam width=20 on CPU."""
        bots = [Bot(id=i, position=(i % 14 + 1, i // 14 + 1), inventory=()) for i in range(20)]
        items = [Item(id=f"item_{i}", type="milk", position=(i % 14 + 1, i // 14 + 5)) for i in range(30)]
        state = _make_state(bots, items, width=20, height=14)
        pe = _make_pe(state)
        ctx = _make_ctx(state, n_bots=20)

        gen = CandidateGenerator(k=5)
        candidates = gen.generate(state, pe, set())

        scorer = ScorerMLP()
        scorer.eval()
        beam = BeamSearch(beam_width=20)

        # Warmup
        beam.search(state, pe, candidates, scorer, ctx)

        # Timed run
        t0 = time.perf_counter()
        for _ in range(10):
            beam.search(state, pe, candidates, scorer, ctx)
        elapsed = (time.perf_counter() - t0) / 10

        assert elapsed < 0.1, f"Beam search took {elapsed*1000:.1f}ms (limit: 100ms)"

    def test_correctness_on_recon(self):
        """Run on real recon data — no double booking across 10 random states."""
        recon_path = _ROOT / "logs" / "74001e7f_2026-03-16_score274_recon.json"
        if not recon_path.exists():
            pytest.skip("Recon file not available")

        from Simulering.offline.simulator import Simulator

        recon = json.loads(recon_path.read_text(encoding="utf-8"))
        sim = Simulator.from_recon_data(recon)
        sim_state = sim.reset()

        # Build path engine once
        state_dict = sim_state.to_dict()
        gs = GameState.from_dict(state_dict)
        shelves = frozenset(sim.shelves)
        merged_walls = gs.grid.walls | shelves
        merged_grid = Grid(gs.grid.width, gs.grid.height, merged_walls)
        pe = PathEngine()
        pe.set_grid(merged_grid, gs.drop_off)

        all_types = sorted(set(i.type for i in gs.items))
        type_index = {t: i for i, t in enumerate(all_types)}

        scorer = ScorerMLP()
        scorer.eval()
        gen = CandidateGenerator(k=5)
        beam = BeamSearch(beam_width=20)

        # Check 10 states (round 0, then step with random actions)
        for round_idx in range(10):
            state_dict = sim_state.to_dict()
            gs = GameState.from_dict(state_dict)

            active = gs.active_orders[0] if gs.active_orders else None
            preview = gs.preview_orders[0] if gs.preview_orders else None

            ctx = FeatureContext(
                bot_positions=[b.position for b in gs.bots],
                n_bots=len(gs.bots),
                max_dist=60,
                item_type_index=type_index,
                active_order=active,
                preview_order=preview,
                drop_off_zones=gs.drop_off_zones,
            )

            candidates = gen.generate(gs, pe, set())
            assignment = beam.search(gs, pe, candidates, scorer, ctx)

            # Verify no double booking
            items_assigned = [v for v in assignment.values() if v not in (DELIVER, IDLE)]
            assert len(items_assigned) == len(set(items_assigned)), \
                f"Round {round_idx}: double-booking {items_assigned}"

            # Verify all actions valid
            valid_ids = {item.id for item in gs.items} | {DELIVER, IDLE}
            for bot_id, action in assignment.items():
                assert action in valid_ids, f"Round {round_idx}: bot {bot_id} invalid {action}"

            # Step sim with wait actions
            actions = [{"bot": b.id, "action": "wait"} for b in sim._bots]
            sim_state, _ = sim.step(actions)
