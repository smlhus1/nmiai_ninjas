"""Tests for route builder and planner route tracking."""

from bot.models import GameState, Grid, Bot, Item, Order
from bot.engine.pathfinding import PathEngine
from bot.engine.world_model import WorldModel
from bot.strategy.task import (
    BotAssignment, Task, TaskType, Route, RouteStop,
)
from bot.strategy.route_builder import build_routes


def _make_world(bots, items, orders=None, walls=None, round_num=1, max_rounds=300):
    """Build a WorldModel for testing."""
    default_walls = [[5, y] for y in range(0, 8)]
    raw = {
        "type": "game_state",
        "round": round_num,
        "max_rounds": max_rounds,
        "grid": {"width": 14, "height": 10, "walls": walls or default_walls},
        "bots": bots,
        "items": items,
        "orders": orders or [
            {
                "id": "order_0",
                "items_required": ["milk", "bread", "cheese"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            }
        ],
        "drop_off": [7, 5],
        "score": 0,
    }
    state = GameState.from_dict(raw)
    engine = PathEngine()
    engine.set_grid(state.grid)
    return WorldModel(state, engine)


class TestBuildRoutes:
    """Tests for the route builder."""

    def test_build_routes_basic(self):
        """Should generate routes with up to 3 stops for 3-item order."""
        world = _make_world(
            bots=[{"id": 0, "position": [1, 1], "inventory": []}],
            items=[
                {"id": "item_0", "type": "milk", "position": [5, 1]},
                {"id": "item_1", "type": "bread", "position": [5, 3]},
                {"id": "item_2", "type": "cheese", "position": [5, 5]},
            ],
        )

        bot = world.state.bots[0]
        order = world.state.active_orders[0]
        routes = build_routes(bot, world, order, set())

        assert len(routes) > 0
        # Should have at least one multi-stop route
        multi_stop = [r for r in routes if len(r.stops) > 1]
        assert len(multi_stop) > 0, "Should generate multi-item routes"
        # Best route should have 3 stops (all items)
        assert any(len(r.stops) == 3 for r in routes)

    def test_build_routes_respects_claimed(self):
        """Claimed items should be excluded from routes."""
        world = _make_world(
            bots=[{"id": 0, "position": [1, 1], "inventory": []}],
            items=[
                {"id": "item_0", "type": "milk", "position": [5, 1]},
                {"id": "item_1", "type": "bread", "position": [5, 3]},
                {"id": "item_2", "type": "cheese", "position": [5, 5]},
            ],
        )

        bot = world.state.bots[0]
        order = world.state.active_orders[0]
        claimed = {"item_0", "item_1"}  # milk and bread claimed
        routes = build_routes(bot, world, order, claimed)

        # Only cheese should be available
        for route in routes:
            for stop in route.stops:
                assert stop.item_id not in claimed

    def test_build_routes_respects_capacity(self):
        """Bot with 1 item in inventory should get routes with max 2 stops."""
        world = _make_world(
            bots=[{"id": 0, "position": [1, 1], "inventory": ["juice"]}],
            items=[
                {"id": "item_0", "type": "milk", "position": [5, 1]},
                {"id": "item_1", "type": "bread", "position": [5, 3]},
                {"id": "item_2", "type": "cheese", "position": [5, 5]},
            ],
        )

        bot = world.state.bots[0]
        order = world.state.active_orders[0]
        routes = build_routes(bot, world, order, set())

        for route in routes:
            assert len(route.stops) <= 2, "Should respect remaining capacity"

    def test_build_routes_includes_single_item_fallback(self):
        """Should always include single-item routes as fallback."""
        world = _make_world(
            bots=[{"id": 0, "position": [1, 1], "inventory": []}],
            items=[
                {"id": "item_0", "type": "milk", "position": [5, 1]},
                {"id": "item_1", "type": "bread", "position": [5, 3]},
            ],
            orders=[
                {
                    "id": "order_0",
                    "items_required": ["milk", "bread"],
                    "items_delivered": [],
                    "complete": False,
                    "status": "active",
                }
            ],
        )

        bot = world.state.bots[0]
        order = world.state.active_orders[0]
        routes = build_routes(bot, world, order, set())

        single_item = [r for r in routes if len(r.stops) == 1]
        assert len(single_item) >= 1, "Should include single-item fallback routes"

    def test_build_routes_sorted_by_cost(self):
        """Routes should be sorted by total cost (cheapest first)."""
        world = _make_world(
            bots=[{"id": 0, "position": [1, 1], "inventory": []}],
            items=[
                {"id": "item_0", "type": "milk", "position": [5, 1]},
                {"id": "item_1", "type": "bread", "position": [5, 3]},
                {"id": "item_2", "type": "cheese", "position": [5, 5]},
            ],
        )

        bot = world.state.bots[0]
        order = world.state.active_orders[0]
        routes = build_routes(bot, world, order, set())

        costs = [r.total_cost for r in routes]
        assert costs == sorted(costs), "Routes should be sorted by cost"

    def test_build_routes_no_order(self):
        """No order should return empty routes."""
        world = _make_world(
            bots=[{"id": 0, "position": [1, 1], "inventory": []}],
            items=[{"id": "item_0", "type": "milk", "position": [5, 1]}],
        )

        bot = world.state.bots[0]
        routes = build_routes(bot, world, None, set())
        assert routes == []

    def test_build_routes_full_inventory(self):
        """Bot with full inventory should get no routes."""
        world = _make_world(
            bots=[{"id": 0, "position": [1, 1], "inventory": ["a", "b", "c"]}],
            items=[{"id": "item_0", "type": "milk", "position": [5, 1]}],
        )

        bot = world.state.bots[0]
        order = world.state.active_orders[0]
        routes = build_routes(bot, world, order, set())
        assert routes == []


class TestRouteAdvancement:
    """Tests for planner route advancement logic."""

    def test_route_advance_on_pickup(self):
        """When current route item is picked (gone from map), route step should advance."""
        from bot.strategy.planner import TaskPlanner

        # Bot picked up item_0 (not on map anymore), should advance to item_1
        world = _make_world(
            bots=[{"id": 0, "position": [4, 3], "inventory": ["milk"]}],
            items=[
                # item_0 (milk) is GONE — picked up
                {"id": "item_1", "type": "bread", "position": [5, 3]},
                {"id": "item_2", "type": "cheese", "position": [5, 5]},
            ],
        )

        route = Route(
            stops=[
                RouteStop("item_0", "milk", (5, 1), (4, 1)),
                RouteStop("item_1", "bread", (5, 3), (4, 3)),
                RouteStop("item_2", "cheese", (5, 5), (4, 5)),
            ],
            order_id="order_0",
            total_cost=20.0,
        )

        assignments = {
            0: BotAssignment(
                bot_id=0,
                task=Task(
                    task_type=TaskType.PICK_UP,
                    target_pos=(4, 1),
                    item_id="item_0",
                    item_type="milk",
                    item_pos=(5, 1),
                    order_id="order_0",
                ),
                route=route,
                route_step=0,
            ),
        }

        planner = TaskPlanner()
        planner._advance_routes(world, assignments)

        a = assignments[0]
        assert a.route_step == 1
        assert a.task.item_id == "item_1"
        assert a.task.item_type == "bread"

    def test_route_complete_triggers_deliver(self):
        """When all route stops are done, bot should switch to DELIVER."""
        from bot.strategy.planner import TaskPlanner

        # Bot has picked all items (none left on map for route)
        world = _make_world(
            bots=[{"id": 0, "position": [6, 5], "inventory": ["milk", "bread"]}],
            items=[
                {"id": "item_2", "type": "cheese", "position": [5, 5]},  # Not in route
            ],
        )

        route = Route(
            stops=[
                RouteStop("item_0", "milk", (5, 1), (4, 1)),
                RouteStop("item_1", "bread", (5, 3), (4, 3)),
            ],
            order_id="order_0",
            total_cost=15.0,
        )

        assignments = {
            0: BotAssignment(
                bot_id=0,
                task=Task(
                    task_type=TaskType.PICK_UP,
                    target_pos=(4, 3),
                    item_id="item_1",
                    item_type="bread",
                    item_pos=(5, 3),
                    order_id="order_0",
                ),
                route=route,
                route_step=1,  # On last stop
            ),
        }

        planner = TaskPlanner()
        planner._advance_routes(world, assignments)

        a = assignments[0]
        assert a.task.task_type == TaskType.DELIVER
        assert a.route is None

    def test_route_invalidation_removes_gone_stops(self):
        """Future route stops that disappear should be removed from route."""
        from bot.strategy.planner import TaskPlanner

        # item_1 (bread) is gone from map, but item_0 (milk) still there
        world = _make_world(
            bots=[{"id": 0, "position": [3, 1], "inventory": []}],
            items=[
                {"id": "item_0", "type": "milk", "position": [5, 1]},
                # item_1 (bread) is GONE
                {"id": "item_2", "type": "cheese", "position": [5, 5]},
            ],
        )

        route = Route(
            stops=[
                RouteStop("item_0", "milk", (5, 1), (4, 1)),
                RouteStop("item_1", "bread", (5, 3), (4, 3)),
                RouteStop("item_2", "cheese", (5, 5), (4, 5)),
            ],
            order_id="order_0",
            total_cost=20.0,
        )

        assignments = {
            0: BotAssignment(
                bot_id=0,
                task=Task(
                    task_type=TaskType.PICK_UP,
                    target_pos=(4, 1),
                    item_id="item_0",
                    item_type="milk",
                    item_pos=(5, 1),
                    order_id="order_0",
                ),
                route=route,
                route_step=0,
            ),
        }

        planner = TaskPlanner()
        planner._invalidate_stale(world, assignments)

        a = assignments[0]
        # Route should still exist (item_0 and item_2 are still on map)
        assert a.route is not None
        # bread stop should be removed from remaining stops
        remaining_ids = {s.item_id for s in a.route.stops[a.route_step:]}
        assert "item_1" not in remaining_ids


class TestPreviewInRoutes:
    """Tests for preview items included in routes."""

    def test_build_routes_includes_preview_on_the_way(self):
        """Routes completing active order should include nearby preview items."""
        world = _make_world(
            bots=[{"id": 0, "position": [4, 1], "inventory": []}],
            items=[
                {"id": "item_0", "type": "milk", "position": [5, 1]},
                {"id": "item_1", "type": "cheese", "position": [5, 3]},  # preview
            ],
            orders=[
                {"id": "order_0", "items_required": ["milk"],
                 "items_delivered": [], "complete": False, "status": "active"},
                {"id": "order_1", "items_required": ["cheese"],
                 "items_delivered": [], "complete": False, "status": "preview"},
            ],
        )
        bot = world.state.bots[0]
        active = world.state.active_orders[0]
        preview = world.state.preview_orders[0]
        routes = build_routes(bot, world, active, set(), preview)
        multi = [r for r in routes if len(r.stops) >= 2]
        assert len(multi) >= 1, "Should include preview item on the way"


class TestEndgameRoutes:
    """Tests for endgame multi-item route planning."""

    def test_endgame_uses_multi_item_routes(self):
        """In endgame with >15 rounds, should batch items."""
        world = _make_world(
            bots=[{"id": 0, "position": [4, 1], "inventory": []}],
            items=[
                {"id": "item_0", "type": "milk", "position": [5, 1]},
                {"id": "item_1", "type": "bread", "position": [5, 3]},
            ],
            orders=[{"id": "order_0", "items_required": ["milk", "bread"],
                     "items_delivered": [], "complete": False, "status": "active"}],
            round_num=275, max_rounds=300,  # 25 rounds left
        )
        from bot.strategy.planner import TaskPlanner
        planner = TaskPlanner()
        assignments = {0: BotAssignment(bot_id=0)}
        planner._plan_endgame(world, assignments, set())
        a = assignments[0]
        assert a.task.task_type == TaskType.PICK_UP
        assert a.route is not None
        assert len(a.route.stops) >= 2
