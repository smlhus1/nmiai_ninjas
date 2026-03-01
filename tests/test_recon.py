"""Tests for the recon/replay system: GameLogger, OfflinePlanner, ReplayPlanner."""

import json
import tempfile
from pathlib import Path

from bot.models import GameState, Grid, Pos
from bot.engine.pathfinding import PathEngine
from bot.engine.world_model import WorldModel
from bot.recon.logger import GameLogger
from bot.recon.analyzer import OfflinePlanner
from bot.recon.replay import ReplayPlanner
from bot.strategy.planner import TaskPlanner
from bot.strategy.task import BotAssignment, TaskType


def _make_state(
    round_num: int = 0,
    max_rounds: int = 300,
    bots: list | None = None,
    items: list | None = None,
    orders: list | None = None,
    walls: list | None = None,
    drop_off: list | None = None,
    score: int = 0,
) -> dict:
    """Build a raw game state dict for testing."""
    default_walls = [[5, y] for y in range(0, 8)]
    return {
        "type": "game_state",
        "round": round_num,
        "max_rounds": max_rounds,
        "grid": {"width": 14, "height": 10, "walls": walls or default_walls},
        "bots": bots or [{"id": 0, "position": [3, 3], "inventory": []}],
        "items": items or [
            {"id": "item_0", "type": "milk", "position": [5, 3]},
            {"id": "item_1", "type": "bread", "position": [5, 1]},
        ],
        "orders": orders or [
            {
                "id": "order_0",
                "items_required": ["milk", "bread"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            }
        ],
        "drop_off": drop_off or [7, 5],
        "score": score,
    }


def _parse_state(raw: dict) -> GameState:
    return GameState.from_dict(raw)


def _make_world(raw: dict, shelf_positions: frozenset[Pos] | None = None) -> WorldModel:
    state = _parse_state(raw)
    if shelf_positions:
        enhanced_grid = Grid(
            width=state.grid.width,
            height=state.grid.height,
            walls=state.grid.walls | shelf_positions,
        )
        state = GameState(
            round=state.round,
            max_rounds=state.max_rounds,
            grid=enhanced_grid,
            bots=state.bots,
            items=state.items,
            orders=state.orders,
            drop_off=state.drop_off,
            score=state.score,
        )
    engine = PathEngine()
    engine.set_grid(state.grid)
    return WorldModel(state, engine)


# ============================================================
# GameLogger tests
# ============================================================

class TestGameLogger:
    def test_initialize_captures_map_data(self):
        """Logger should capture grid, walls, shelf map, and fingerprint on first round."""
        logger = GameLogger()
        raw = _make_state()
        state = _parse_state(raw)
        shelf_positions = frozenset(item.position for item in state.items)

        logger.on_round(state, shelf_positions)

        assert logger.fingerprint != ""
        assert len(logger.fingerprint) == 8  # sha256[:8]

    def test_finalize_returns_complete_data(self):
        """finalize() should return a dict with all required keys."""
        logger = GameLogger()
        raw = _make_state()
        state = _parse_state(raw)
        shelf_positions = frozenset(item.position for item in state.items)

        logger.on_round(state, shelf_positions)
        data = logger.finalize(total_rounds=100, final_score=42)

        assert data["fingerprint"] == logger.fingerprint
        assert data["grid_size"] == [14, 10]
        assert data["bot_count"] == 1
        assert data["total_rounds"] == 100
        assert data["final_score"] == 42
        assert "shelf_map" in data
        assert "order_sequence" in data
        assert "milk" in data["shelf_map"]
        assert "bread" in data["shelf_map"]

    def test_order_tracking(self):
        """Logger should track new orders and their transitions."""
        logger = GameLogger()
        raw = _make_state()
        state = _parse_state(raw)
        shelf_positions = frozenset(item.position for item in state.items)

        # Round 0: initial order
        logger.on_round(state, shelf_positions)

        # Round 5: order completes, new preview appears
        raw2 = _make_state(
            round_num=5,
            orders=[
                {
                    "id": "order_0",
                    "items_required": ["milk", "bread"],
                    "items_delivered": ["milk", "bread"],
                    "complete": True,
                    "status": "active",
                },
                {
                    "id": "order_1",
                    "items_required": ["cheese"],
                    "items_delivered": [],
                    "complete": False,
                    "status": "preview",
                },
            ],
        )
        state2 = _parse_state(raw2)
        logger.on_round(state2, shelf_positions)

        data = logger.finalize(100, 10)
        assert len(data["order_sequence"]) == 2
        assert data["order_sequence"][0]["id"] == "order_0"
        assert data["order_sequence"][1]["id"] == "order_1"
        assert data["order_sequence"][0]["completed_round"] == 5

    def test_fingerprint_deterministic(self):
        """Same map should produce same fingerprint."""
        logger1 = GameLogger()
        logger2 = GameLogger()
        raw = _make_state()
        state = _parse_state(raw)
        shelves = frozenset(item.position for item in state.items)

        logger1.on_round(state, shelves)
        logger2.on_round(state, shelves)

        assert logger1.fingerprint == logger2.fingerprint

    def test_fingerprint_changes_with_map(self):
        """Different map layout should produce different fingerprint."""
        logger1 = GameLogger()
        logger2 = GameLogger()

        raw1 = _make_state(walls=[[5, y] for y in range(0, 8)])
        raw2 = _make_state(walls=[[3, y] for y in range(0, 8)])

        state1 = _parse_state(raw1)
        state2 = _parse_state(raw2)
        shelves1 = frozenset(item.position for item in state1.items)
        shelves2 = frozenset(item.position for item in state2.items)

        logger1.on_round(state1, shelves1)
        logger2.on_round(state2, shelves2)

        assert logger1.fingerprint != logger2.fingerprint

    def test_serializable(self):
        """finalize() output should be JSON-serializable."""
        logger = GameLogger()
        raw = _make_state()
        state = _parse_state(raw)
        shelves = frozenset(item.position for item in state.items)
        logger.on_round(state, shelves)

        data = logger.finalize(100, 42)
        json_str = json.dumps(data)
        assert json_str  # Should not throw


# ============================================================
# OfflinePlanner tests
# ============================================================

class TestOfflinePlanner:
    def _make_recon_data(self, orders=None, shelf_map=None, walls=None) -> dict:
        """Build recon data dict for OfflinePlanner."""
        return {
            "fingerprint": "test1234",
            "grid_size": [14, 10],
            "walls": walls or [[5, y] for y in range(0, 8)],
            "drop_off": [7, 5],
            "shelf_map": shelf_map or {
                "milk": [[5, 3]],
                "bread": [[5, 1]],
                "cheese": [[5, 5]],
            },
            "bot_count": 1,
            "order_sequence": orders or [
                {
                    "id": "order_0",
                    "items_required": ["milk", "bread"],
                    "first_seen_round": 0,
                    "status_when_seen": "active",
                },
            ],
            "total_rounds": 100,
            "final_score": 20,
        }

    def _make_engine(self, walls=None) -> PathEngine:
        """Create a PathEngine with grid set up."""
        default_walls = [(5, y) for y in range(0, 8)]
        wall_set = frozenset(walls) if walls else frozenset(default_walls)
        # Add shelf positions as walls
        shelf_walls = frozenset([(5, 1), (5, 3), (5, 5)])
        grid = Grid(width=14, height=10, walls=wall_set | shelf_walls)
        engine = PathEngine()
        engine.set_grid(grid)
        return engine

    def test_plan_basic(self):
        """Should generate a plan with pickup sequences for each order."""
        recon = self._make_recon_data()
        engine = self._make_engine()
        planner = OfflinePlanner(recon, engine)

        plan = planner.plan()

        assert plan["fingerprint"] == "test1234"
        assert len(plan["order_plans"]) == 1
        order_plan = plan["order_plans"][0]
        assert len(order_plan["pickup_sequence"]) == 2
        assert order_plan["estimated_rounds"] > 0

    def test_plan_batching(self):
        """Order with more items than inventory cap should be split into batches."""
        recon = self._make_recon_data(
            orders=[{
                "id": "order_0",
                "items_required": ["milk", "bread", "cheese", "milk"],
                "first_seen_round": 0,
                "status_when_seen": "active",
            }],
            shelf_map={
                "milk": [[5, 3], [5, 7]],
                "bread": [[5, 1]],
                "cheese": [[5, 5]],
            },
        )
        engine = self._make_engine()
        planner = OfflinePlanner(recon, engine)

        plan = planner.plan()
        order_plan = plan["order_plans"][0]

        assert len(order_plan["batches"]) >= 2  # 4 items / cap 3 = at least 2 batches

    def test_plan_serializable(self):
        """Plan output should be JSON-serializable."""
        recon = self._make_recon_data()
        engine = self._make_engine()
        planner = OfflinePlanner(recon, engine)

        plan = planner.plan()
        json_str = json.dumps(plan)
        assert json_str

    def test_plan_duplicate_shelf_split_across_batches(self):
        """When order needs 2x same type from same shelf, they should be in separate batches."""
        recon = self._make_recon_data(
            orders=[{
                "id": "order_0",
                "items_required": ["milk", "bread", "milk"],
                "first_seen_round": 0,
                "status_when_seen": "active",
            }],
            shelf_map={
                "milk": [[5, 3]],  # Only ONE shelf for milk
                "bread": [[5, 1]],
            },
        )
        engine = self._make_engine()
        planner = OfflinePlanner(recon, engine)

        plan = planner.plan()
        order_plan = plan["order_plans"][0]

        # Should have 2+ batches because 2x milk from same shelf can't be in same batch
        assert len(order_plan["batches"]) >= 2

        # Verify no batch has duplicate shelf positions
        for batch in order_plan["batches"]:
            shelf_positions = [tuple(entry["shelf_pos"]) for entry in batch]
            assert len(shelf_positions) == len(set(shelf_positions)), \
                f"Batch has duplicate shelves: {shelf_positions}"

    def test_plan_cross_order_prepick(self):
        """Plan should include pre-picks for next order if capacity allows."""
        recon = self._make_recon_data(
            orders=[
                {
                    "id": "order_0",
                    "items_required": ["milk"],
                    "first_seen_round": 0,
                    "status_when_seen": "active",
                },
                {
                    "id": "order_1",
                    "items_required": ["bread"],
                    "first_seen_round": 10,
                    "status_when_seen": "preview",
                },
            ],
        )
        engine = self._make_engine()
        planner = OfflinePlanner(recon, engine)

        plan = planner.plan()
        # First order plan may have pre-picks if bread shelf is on the way
        order_plan = plan["order_plans"][0]
        # Pre-picks may or may not be found depending on geometry,
        # but the field should exist
        assert "pre_picks" in order_plan


# ============================================================
# ReplayPlanner tests
# ============================================================

class TestReplayPlanner:
    def _make_plan(self) -> dict:
        """Build a simple game plan for testing."""
        return {
            "fingerprint": "test1234",
            "bot_count": 1,
            "drop_off": [7, 5],
            "order_plans": [
                {
                    "order_index": 0,
                    "order_id": "order_0",
                    "items_required": ["milk", "bread"],
                    "pickup_sequence": [
                        {"shelf_pos": [5, 3], "pickup_pos": [4, 3], "item_type": "milk"},
                        {"shelf_pos": [5, 1], "pickup_pos": [4, 1], "item_type": "bread"},
                    ],
                    "batches": [
                        [
                            {"shelf_pos": [5, 3], "pickup_pos": [4, 3], "item_type": "milk"},
                            {"shelf_pos": [5, 1], "pickup_pos": [4, 1], "item_type": "bread"},
                        ],
                    ],
                    "pre_picks": [],
                    "estimated_rounds": 15,
                },
            ],
        }

    def _make_world_for_replay(self, round_num=1) -> tuple[WorldModel, dict[int, BotAssignment]]:
        """Build WorldModel and assignments for replay testing."""
        raw = _make_state(
            round_num=round_num,
            bots=[{"id": 0, "position": [3, 3], "inventory": []}],
            items=[
                {"id": "item_A", "type": "milk", "position": [5, 3]},
                {"id": "item_B", "type": "bread", "position": [5, 1]},
            ],
            orders=[{
                "id": "order_0",
                "items_required": ["milk", "bread"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            }],
        )
        shelf_positions = frozenset([(5, 3), (5, 1)])
        world = _make_world(raw, shelf_positions)
        assignments = {0: BotAssignment(bot_id=0)}
        return world, assignments

    def test_replay_assigns_from_plan(self):
        """ReplayPlanner should assign tasks based on the pre-computed plan."""
        plan = self._make_plan()
        reactive = TaskPlanner()
        replay = ReplayPlanner(plan, reactive)

        world, assignments = self._make_world_for_replay()
        result = replay.plan(world, assignments)

        # Bot should have a task assigned
        assert result[0].task is not None
        assert result[0].task.task_type in (TaskType.PICK_UP, TaskType.DELIVER, TaskType.PRE_PICK)

    def test_replay_fallback_on_exhausted_plan(self):
        """When plan is exhausted, should fall back to reactive planner."""
        plan = self._make_plan()
        plan["order_plans"] = []  # Empty plan
        reactive = TaskPlanner()
        replay = ReplayPlanner(plan, reactive)

        world, assignments = self._make_world_for_replay()
        result = replay.plan(world, assignments)

        # Should still work — reactive planner takes over
        assert result[0].task is not None

    def test_replay_divergence_switch(self):
        """After MAX_DIVERGENCE consecutive divergences, should switch to reactive."""
        plan = self._make_plan()
        # Set an order_id that won't match the game state
        plan["order_plans"][0]["order_id"] = "nonexistent_order"
        reactive = TaskPlanner()
        replay = ReplayPlanner(plan, reactive)

        for i in range(6):
            raw = _make_state(
                round_num=i,
                bots=[{"id": 0, "position": [3, 3], "inventory": []}],
                items=[
                    {"id": "item_A", "type": "milk", "position": [5, 3]},
                    {"id": "item_B", "type": "bread", "position": [5, 1]},
                ],
                orders=[{
                    "id": "order_0",
                    "items_required": ["milk", "bread"],
                    "items_delivered": [],
                    "complete": False,
                    "status": "active",
                }],
            )
            shelf_positions = frozenset([(5, 3), (5, 1)])
            world = _make_world(raw, shelf_positions)
            assignments = {0: BotAssignment(bot_id=0)}
            replay.plan(world, assignments)

        # After enough divergences, mode should be reactive
        assert replay._mode == "reactive"

    def test_replay_shelf_based_item_resolution(self):
        """Replay should find items by shelf position + type, not by ID."""
        plan = self._make_plan()
        reactive = TaskPlanner()
        replay = ReplayPlanner(plan, reactive)

        # Items have different IDs than what the plan might reference
        raw = _make_state(
            round_num=1,
            bots=[{"id": 0, "position": [3, 3], "inventory": []}],
            items=[
                {"id": "DIFFERENT_ID_1", "type": "milk", "position": [5, 3]},
                {"id": "DIFFERENT_ID_2", "type": "bread", "position": [5, 1]},
            ],
            orders=[{
                "id": "order_0",
                "items_required": ["milk", "bread"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            }],
        )
        shelf_positions = frozenset([(5, 3), (5, 1)])
        world = _make_world(raw, shelf_positions)
        assignments = {0: BotAssignment(bot_id=0)}

        result = replay.plan(world, assignments)

        # Should successfully assign despite different item IDs
        assert result[0].task is not None
        if result[0].task.item_id:
            assert result[0].task.item_id in ("DIFFERENT_ID_1", "DIFFERENT_ID_2")


# ============================================================
# Integration: Logger -> Analyzer -> Plan
# ============================================================

class TestReconIntegration:
    def test_logger_to_analyzer_pipeline(self):
        """Full pipeline: log a game, analyze, produce plan."""
        # Step 1: Log game data
        game_logger = GameLogger()
        raw = _make_state(
            items=[
                {"id": "item_0", "type": "milk", "position": [5, 3]},
                {"id": "item_1", "type": "bread", "position": [5, 1]},
                {"id": "item_2", "type": "cheese", "position": [5, 5]},
            ],
            orders=[
                {
                    "id": "order_0",
                    "items_required": ["milk", "bread"],
                    "items_delivered": [],
                    "complete": False,
                    "status": "active",
                },
                {
                    "id": "order_1",
                    "items_required": ["cheese"],
                    "items_delivered": [],
                    "complete": False,
                    "status": "preview",
                },
            ],
        )
        state = _parse_state(raw)
        shelf_positions = frozenset(item.position for item in state.items)
        game_logger.on_round(state, shelf_positions)

        recon_data = game_logger.finalize(100, 20)

        # Step 2: Analyze
        grid = Grid(
            width=14, height=10,
            walls=frozenset(tuple(w) for w in recon_data["walls"]) | shelf_positions,
        )
        engine = PathEngine()
        engine.set_grid(grid)

        analyzer = OfflinePlanner(recon_data, engine)
        plan = analyzer.plan()

        # Step 3: Verify plan structure
        assert plan["fingerprint"] == recon_data["fingerprint"]
        assert len(plan["order_plans"]) == 2
        assert plan["order_plans"][0]["items_required"] == ["milk", "bread"]
        assert plan["order_plans"][1]["items_required"] == ["cheese"]

        # Plan should be JSON-serializable (round-trip test)
        json_str = json.dumps(plan)
        reloaded = json.loads(json_str)
        assert reloaded["order_plans"][0]["order_id"] == "order_0"
