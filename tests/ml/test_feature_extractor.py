"""Tests for FeatureExtractor — bot, item, global features and encode_pair."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.engine.pathfinding import PathEngine
from bot.models import Bot, GameState, Grid, Item, Order, OrderStatus, Pos
from bot.strategy.task import BotAssignment, Task, TaskType
from ml.feature_extractor import FeatureContext, FeatureExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_grid(width=16, height=12, walls=None):
    return Grid(width=width, height=height, walls=frozenset(walls or []))


def _make_state(
    *,
    round_=10,
    max_rounds=300,
    width=16,
    height=12,
    bots=None,
    items=None,
    orders=None,
    drop_off=(1, 10),
    walls=None,
    score=50,
):
    grid = _make_grid(width, height, walls)
    return GameState(
        round=round_,
        max_rounds=max_rounds,
        grid=grid,
        bots=tuple(bots or []),
        items=tuple(items or []),
        orders=tuple(orders or []),
        drop_off=drop_off,
        drop_off_zones=(drop_off,),
        score=score,
    )


def _make_path_engine(state: GameState) -> PathEngine:
    pe = PathEngine()
    pe.set_grid(state.grid, state.drop_off)
    return pe


def _make_ctx(**overrides) -> FeatureContext:
    defaults = dict(
        n_bots=20,
        max_dist=60,
        drop_off_zones=((1, 10),),
    )
    defaults.update(overrides)
    return FeatureContext(**defaults)


# ---------------------------------------------------------------------------
# TASK-3-1: Bot features (14 floats)
# ---------------------------------------------------------------------------


class TestBotFeatures:
    def test_shape_and_range(self):
        bot = Bot(id=3, position=(5, 8), inventory=("milk",))
        state = _make_state(bots=[bot])
        pe = _make_path_engine(state)
        ctx = _make_ctx(bot_positions=[(5, 8)])

        f = FeatureExtractor.extract_bot_features(bot, state, pe, ctx)
        assert f.shape == (14,)
        assert (f >= 0.0).all(), f"Values below 0: {f}"
        assert (f <= 1.0).all(), f"Values above 1: {f}"

    def test_bfs_distance_not_manhattan(self):
        """BFS dist to drop-off should differ from Manhattan when wall blocks path."""
        # Wall blocks direct path — BFS must go around
        walls = frozenset([(3, 8), (3, 9), (3, 10)])
        bot = Bot(id=0, position=(5, 8), inventory=())
        state = _make_state(bots=[bot], drop_off=(1, 10), walls=walls)
        pe = _make_path_engine(state)

        bfs_dist = pe.distance((5, 8), (1, 10))
        manhattan = abs(5 - 1) + abs(8 - 10)
        assert bfs_dist > manhattan, f"BFS {bfs_dist} should be > Manhattan {manhattan} with wall"

        ctx = _make_ctx(bot_positions=[(5, 8)], max_dist=bfs_dist, drop_off_zones=((1, 10),))
        f = FeatureExtractor.extract_bot_features(bot, state, pe, ctx)
        # dist_to_dropoff = bfs_dist / max_dist = 1.0
        assert f[2] == pytest.approx(1.0, abs=0.01)

    def test_empty_inventory(self):
        bot = Bot(id=0, position=(5, 5), inventory=())
        state = _make_state(bots=[bot])
        pe = _make_path_engine(state)
        ctx = _make_ctx(bot_positions=[(5, 5)])

        f = FeatureExtractor.extract_bot_features(bot, state, pe, ctx)
        assert f[3] == 0.0  # inv_size
        assert f[4] == 0.0  # inv_active_match
        assert f[5] == 0.0  # inv_preview_match

    def test_task_onehot_sums_to_one(self):
        """Task one-hot should sum to exactly 1.0 for all valid types."""
        for tt in TaskType:
            bot = Bot(id=0, position=(5, 5), inventory=())
            state = _make_state(bots=[bot])
            pe = _make_path_engine(state)
            assignment = BotAssignment(bot_id=0, task=Task(tt, (5, 5)))
            ctx = _make_ctx(
                assignments={0: assignment},
                bot_positions=[(5, 5)],
            )
            f = FeatureExtractor.extract_bot_features(bot, state, pe, ctx)
            assert f[6:10].sum() == pytest.approx(1.0), f"Task {tt}: one-hot sum != 1.0"

    def test_idle_when_no_assignment(self):
        bot = Bot(id=5, position=(5, 5), inventory=())
        state = _make_state(bots=[bot])
        pe = _make_path_engine(state)
        ctx = _make_ctx(bot_positions=[(5, 5)])

        f = FeatureExtractor.extract_bot_features(bot, state, pe, ctx)
        # No assignment -> IDLE -> task_idle=1.0
        assert f[9] == 1.0  # task_idle

    def test_inventory_active_match(self):
        """Bot with milk inventory + active order needing milk -> match=1/3."""
        active = Order(
            id="o1",
            items_required=("milk", "bread"),
            items_delivered=(),
            complete=False,
            status=OrderStatus.ACTIVE,
        )
        bot = Bot(id=0, position=(5, 5), inventory=("milk",))
        state = _make_state(bots=[bot])
        pe = _make_path_engine(state)
        ctx = _make_ctx(
            active_order=active,
            bot_positions=[(5, 5)],
        )
        f = FeatureExtractor.extract_bot_features(bot, state, pe, ctx)
        assert f[4] == pytest.approx(1 / 3.0)  # inv_active_match


# ---------------------------------------------------------------------------
# TASK-3-2: Item features (12 floats)
# ---------------------------------------------------------------------------


class TestItemFeatures:
    def test_shape_and_range(self):
        bot = Bot(id=0, position=(5, 5), inventory=())
        item = Item(id="item_1", type="milk", position=(7, 5))
        state = _make_state(bots=[bot], items=[item])
        pe = _make_path_engine(state)
        ctx = _make_ctx(
            bot_positions=[(5, 5)],
            item_type_index={"milk": 0, "bread": 1},
        )

        f = FeatureExtractor.extract_item_features(bot, item, state, pe, ctx)
        assert f.shape == (12,)
        assert (f >= 0.0).all(), f"Values below 0: {f}"
        assert (f <= 1.0).all(), f"Values above 1: {f}"

    def test_active_needed(self):
        active = Order(
            id="o1",
            items_required=("milk", "bread"),
            items_delivered=(),
            complete=False,
            status=OrderStatus.ACTIVE,
        )
        bot = Bot(id=0, position=(5, 5), inventory=())
        # milk is needed
        item_milk = Item(id="item_1", type="milk", position=(7, 5))
        # eggs is NOT needed
        item_eggs = Item(id="item_2", type="eggs", position=(7, 6))

        state = _make_state(bots=[bot], items=[item_milk, item_eggs])
        pe = _make_path_engine(state)
        ctx = _make_ctx(
            active_order=active,
            bot_positions=[(5, 5)],
            item_type_index={"milk": 0, "bread": 1, "eggs": 2},
        )

        f_milk = FeatureExtractor.extract_item_features(bot, item_milk, state, pe, ctx)
        f_eggs = FeatureExtractor.extract_item_features(bot, item_eggs, state, pe, ctx)
        assert f_milk[4] == 1.0  # is_active_needed
        assert f_eggs[4] == 0.0  # NOT active needed

    def test_claimed(self):
        bot = Bot(id=0, position=(5, 5), inventory=())
        item = Item(id="item_1", type="milk", position=(7, 5))
        state = _make_state(bots=[bot], items=[item])
        pe = _make_path_engine(state)

        # Not claimed
        ctx = _make_ctx(bot_positions=[(5, 5)], claimed_items=set())
        f = FeatureExtractor.extract_item_features(bot, item, state, pe, ctx)
        assert f[8] == 0.0

        # Claimed
        ctx2 = _make_ctx(bot_positions=[(5, 5)], claimed_items={"item_1"})
        f2 = FeatureExtractor.extract_item_features(bot, item, state, pe, ctx2)
        assert f2[8] == 1.0

    def test_no_demand_no_crash(self):
        """No future orders -> demand_score = 0.0, not crash."""
        bot = Bot(id=0, position=(5, 5), inventory=())
        item = Item(id="item_1", type="milk", position=(7, 5))
        state = _make_state(bots=[bot], items=[item])
        pe = _make_path_engine(state)
        ctx = _make_ctx(bot_positions=[(5, 5)], demand={})

        f = FeatureExtractor.extract_item_features(bot, item, state, pe, ctx)
        assert f[7] == 0.0  # demand_score

    def test_bfs_distance_with_wall(self):
        """dist_bot_to_item uses BFS, not Manhattan."""
        walls = frozenset([(6, 5)])  # blocks direct path
        bot = Bot(id=0, position=(5, 5), inventory=())
        item = Item(id="item_1", type="milk", position=(7, 5))
        state = _make_state(bots=[bot], items=[item], walls=walls)
        pe = _make_path_engine(state)
        ctx = _make_ctx(bot_positions=[(5, 5)], max_dist=60)

        bfs_dist = pe.distance((5, 5), (7, 5))
        manhattan = abs(5 - 7) + abs(5 - 5)
        assert bfs_dist > manhattan

        f = FeatureExtractor.extract_item_features(bot, item, state, pe, ctx)
        assert f[2] == pytest.approx(bfs_dist / 60.0, abs=0.01)


# ---------------------------------------------------------------------------
# TASK-3-3: Global features (22 floats)
# ---------------------------------------------------------------------------


class TestGlobalFeatures:
    def test_shape_and_range(self):
        bot = Bot(id=0, position=(5, 5), inventory=())
        state = _make_state(bots=[bot])
        pe = _make_path_engine(state)
        ctx = _make_ctx(
            assignments={0: BotAssignment(bot_id=0, task=Task(TaskType.IDLE, (5, 5)))},
            bot_positions=[(5, 5)],
        )

        f = FeatureExtractor.extract_global_features(state, pe, ctx)
        assert f.shape == (22,)
        assert (f >= 0.0).all(), f"Values below 0: {f}"
        assert (f <= 1.0).all(), f"Values above 1: {f}"

    def test_score_velocity_early_rounds(self):
        """score_velocity returns 0.0 when < 2 history entries."""
        state = _make_state(round_=3)
        pe = _make_path_engine(state)
        ctx = _make_ctx(score_history=[0])

        f = FeatureExtractor.extract_global_features(state, pe, ctx)
        assert f[19] == 0.0  # score_velocity

    def test_score_velocity_with_history(self):
        """score_velocity computed from last 10 rounds."""
        history = list(range(0, 25, 2))  # [0, 2, 4, ..., 24]
        state = _make_state(round_=12)
        pe = _make_path_engine(state)
        ctx = _make_ctx(score_history=history)

        f = FeatureExtractor.extract_global_features(state, pe, ctx)
        # Last 10 entries: gain = 24 - 4 = 20, window = 10, velocity = 20/10/2.0 = 1.0
        assert f[19] == pytest.approx(1.0, abs=0.05)

    def test_next_order_types_sum(self):
        """next_order_types (7 floats) sum <= 1.0 per slot."""
        preview = Order(
            id="p1",
            items_required=("milk", "milk", "bread"),
            items_delivered=(),
            complete=False,
            status=OrderStatus.PREVIEW,
        )
        state = _make_state()
        pe = _make_path_engine(state)
        ctx = _make_ctx(
            preview_order=preview,
            item_type_index={"milk": 0, "bread": 1, "eggs": 2},
        )

        f = FeatureExtractor.extract_global_features(state, pe, ctx)
        # next_order_types is f[7:14]
        types_slice = f[7:14]
        assert (types_slice <= 1.0).all()

    def test_pipeline_status(self):
        """Verify bot counts by task type."""
        assignments = {
            0: BotAssignment(bot_id=0, task=Task(TaskType.DELIVER, (1, 10))),
            1: BotAssignment(bot_id=1, task=Task(TaskType.PICK_UP, (5, 5))),
            2: BotAssignment(bot_id=2, task=Task(TaskType.PRE_PICK, (7, 7))),
            3: BotAssignment(bot_id=3, task=Task(TaskType.IDLE, (3, 3))),
        }
        bots = [Bot(id=i, position=(i, 5), inventory=()) for i in range(4)]
        state = _make_state(bots=bots)
        pe = _make_path_engine(state)
        ctx = _make_ctx(
            assignments=assignments,
            bot_positions=[(i, 5) for i in range(4)],
            n_bots=4,
        )

        f = FeatureExtractor.extract_global_features(state, pe, ctx)
        assert f[3] == pytest.approx(1 / 4)  # bots_delivering
        assert f[4] == pytest.approx(1 / 4)  # bots_picking
        assert f[5] == pytest.approx(1 / 4)  # bots_prepicking
        assert f[6] == pytest.approx(1 / 4)  # bots_idle


# ---------------------------------------------------------------------------
# TASK-3-3: encode_pair and encode_all_pairs
# ---------------------------------------------------------------------------


class TestEncodePair:
    def test_shape_48(self):
        bot = Bot(id=0, position=(5, 5), inventory=())
        item = Item(id="item_1", type="milk", position=(7, 5))
        state = _make_state(bots=[bot], items=[item])
        pe = _make_path_engine(state)
        ctx = _make_ctx(bot_positions=[(5, 5)], item_type_index={"milk": 0})

        f = FeatureExtractor.encode_pair(bot, item, state, pe, ctx)
        assert f.shape == (48,)
        assert (f >= 0.0).all()
        assert (f <= 1.0).all()

    def test_concat_order(self):
        """Verify encode_pair = [bot(14) | item(12) | global(22)]."""
        bot = Bot(id=0, position=(5, 5), inventory=())
        item = Item(id="item_1", type="milk", position=(7, 5))
        state = _make_state(bots=[bot], items=[item])
        pe = _make_path_engine(state)
        ctx = _make_ctx(bot_positions=[(5, 5)], item_type_index={"milk": 0})

        full = FeatureExtractor.encode_pair(bot, item, state, pe, ctx)
        bot_f = FeatureExtractor.extract_bot_features(bot, state, pe, ctx)
        item_f = FeatureExtractor.extract_item_features(bot, item, state, pe, ctx)
        glob_f = FeatureExtractor.extract_global_features(state, pe, ctx)

        assert torch.allclose(full[:14], bot_f)
        assert torch.allclose(full[14:26], item_f)
        assert torch.allclose(full[26:48], glob_f)

    def test_encode_all_pairs_shape(self):
        bots = [Bot(id=i, position=(i + 2, 5), inventory=()) for i in range(3)]
        items = [
            Item(id=f"item_{i}", type="milk", position=(10 + i, 5))
            for i in range(4)
        ]
        state = _make_state(bots=bots, items=items)
        pe = _make_path_engine(state)
        ctx = _make_ctx(
            bot_positions=[(i + 2, 5) for i in range(3)],
            item_type_index={"milk": 0},
            n_bots=3,
        )

        result = FeatureExtractor.encode_all_pairs(state, pe, ctx)
        assert result.shape == (3 * 4, 48)  # 3 bots * 4 items
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_no_nan_no_inf_on_recon(self):
        """Integration: load recon, parse round 1, encode all pairs."""
        recon_path = _ROOT / "logs" / "74001e7f_2026-03-16_score274_recon.json"
        if not recon_path.exists():
            pytest.skip("Recon file not available")

        recon = json.loads(recon_path.read_text(encoding="utf-8"))

        # Build a minimal state from recon data
        from Simulering.offline.simulator import Simulator
        sim = Simulator.from_recon_data(recon)
        sim_state = sim.reset()
        state_dict = sim_state.to_dict()
        state = GameState.from_dict(state_dict)

        # Build path engine with shelf-merged grid
        shelves = frozenset(sim.shelves)
        merged_walls = state.grid.walls | shelves
        merged_grid = Grid(state.grid.width, state.grid.height, merged_walls)
        pe = PathEngine()
        pe.set_grid(merged_grid, state.drop_off)

        # Build item type index
        all_types = sorted(set(i.type for i in state.items))
        type_index = {t: i for i, t in enumerate(all_types)}

        # Active/preview orders
        active = state.active_orders[0] if state.active_orders else None
        preview = state.preview_orders[0] if state.preview_orders else None

        ctx = FeatureContext(
            bot_positions=[b.position for b in state.bots],
            n_bots=len(state.bots),
            max_dist=60,
            item_type_index=type_index,
            active_order=active,
            preview_order=preview,
            drop_off_zones=state.drop_off_zones,
        )

        result = FeatureExtractor.encode_all_pairs(state, pe, ctx)
        n_bots = len(state.bots)
        n_items = len(state.items)

        assert result.shape == (n_bots * n_items, 48)
        assert not torch.isnan(result).any(), "NaN found in features"
        assert not torch.isinf(result).any(), "Inf found in features"
        assert (result >= 0.0).all(), "Values below 0"
        assert (result <= 1.0).all(), f"Values above 1: max={result.max()}"
