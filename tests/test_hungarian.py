"""Tests for Hungarian (optimal) assignment with route support."""

from bot.models import GameState, Grid, Bot, Item
from bot.engine.pathfinding import PathEngine
from bot.engine.world_model import WorldModel
from bot.strategy.task import BotAssignment, TaskType


def _make_world(bots, items, orders=None, walls=None, round_num=1):
    """Build a WorldModel for testing."""
    default_walls = [[5, y] for y in range(0, 8)]
    raw = {
        "type": "game_state",
        "round": round_num,
        "max_rounds": 300,
        "grid": {"width": 14, "height": 10, "walls": walls or default_walls},
        "bots": bots,
        "items": items,
        "orders": orders or [
            {
                "id": "order_0",
                "items_required": ["milk", "bread"],
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


def test_hungarian_basic_assignment():
    """Hungarian should assign bots to items with no double-booking."""
    world = _make_world(
        bots=[
            {"id": 0, "position": [1, 1], "inventory": []},
            {"id": 1, "position": [3, 3], "inventory": []},
        ],
        items=[
            {"id": "item_0", "type": "milk", "position": [5, 1]},
            {"id": "item_1", "type": "bread", "position": [5, 3]},
        ],
    )

    assignments = {
        0: BotAssignment(bot_id=0),
        1: BotAssignment(bot_id=1),
    }

    from bot.strategy.hungarian import solve_assignment
    result = solve_assignment(
        [world.state.bots[0], world.state.bots[1]],
        world, assignments, set(), set(),
    )

    # At least one bot should get a task
    assert len(result) >= 1
    # All tasks should be PICK_UP
    assert all(t.task_type == TaskType.PICK_UP for t, _ in result.values())
    # No double-booking: collect all item IDs across tasks and routes
    all_item_ids: list[str] = []
    for task, route in result.values():
        if route:
            all_item_ids.extend(route.item_ids)
        elif task.item_id:
            all_item_ids.append(task.item_id)
    assert len(all_item_ids) == len(set(all_item_ids)), "No double-booking"


def test_hungarian_routes_cover_all_items():
    """
    With routes: one bot can take a multi-item route covering all order items.
    This is more efficient than splitting across bots when items are along a path.
    """
    world = _make_world(
        bots=[
            {"id": 0, "position": [4, 1], "inventory": []},  # Near milk at (5,1)
            {"id": 1, "position": [4, 7], "inventory": []},  # Near eggs at (5,7)
        ],
        items=[
            {"id": "item_0", "type": "milk", "position": [5, 1]},
            {"id": "item_1", "type": "bread", "position": [5, 3]},
            {"id": "item_2", "type": "eggs", "position": [5, 7]},
        ],
        orders=[
            {
                "id": "order_0",
                "items_required": ["milk", "bread", "eggs"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            }
        ],
    )

    assignments = {
        0: BotAssignment(bot_id=0),
        1: BotAssignment(bot_id=1),
    }

    from bot.strategy.hungarian import solve_assignment
    result = solve_assignment(
        [world.state.bots[0], world.state.bots[1]],
        world, assignments, set(), set(),
    )

    # At least one bot should get an assignment
    assert len(result) >= 1
    # Collect all assigned items
    all_items: set[str] = set()
    for task, route in result.values():
        if route:
            all_items.update(route.item_ids)
        elif task.item_id:
            all_items.add(task.item_id)
    # All order items should be covered
    assert all_items == {"item_0", "item_1", "item_2"}


def test_hungarian_respects_claimed():
    """Already claimed items should not be assigned."""
    world = _make_world(
        bots=[{"id": 0, "position": [1, 1], "inventory": []}],
        items=[
            {"id": "item_0", "type": "milk", "position": [5, 1]},
            {"id": "item_1", "type": "bread", "position": [5, 3]},
        ],
    )

    assignments = {0: BotAssignment(bot_id=0)}
    claimed = {"item_0"}  # milk already claimed

    from bot.strategy.hungarian import solve_assignment
    result = solve_assignment(
        [world.state.bots[0]], world, assignments, claimed, set(),
    )

    assert len(result) == 1
    task, _ = result[0]
    assert task.item_id == "item_1"  # Should get bread, not milk


def test_hungarian_delivers_with_inventory():
    """Bot with partial matching inventory should pick more, not deliver immediately."""
    world = _make_world(
        bots=[{"id": 0, "position": [6, 5], "inventory": ["milk"]}],
        items=[{"id": "item_1", "type": "bread", "position": [5, 3]}],
    )

    assignments = {0: BotAssignment(bot_id=0)}

    from bot.strategy.hungarian import solve_assignment
    result = solve_assignment(
        [world.state.bots[0]], world, assignments, set(), set(),
    )

    task, _ = result[0]
    assert task.task_type == TaskType.PICK_UP  # Should pick bread, not deliver milk alone


def test_hungarian_delivers_when_completing_order():
    """Bot with ALL remaining order items should deliver immediately."""
    world = _make_world(
        bots=[{"id": 0, "position": [6, 5], "inventory": ["milk", "bread"]}],
        items=[{"id": "item_2", "type": "cheese", "position": [5, 5]}],
        orders=[{
            "id": "order_0",
            "items_required": ["milk", "bread"],
            "items_delivered": [],
            "complete": False, "status": "active",
        }],
    )
    assignments = {0: BotAssignment(bot_id=0)}

    from bot.strategy.hungarian import solve_assignment
    result = solve_assignment(
        [world.state.bots[0]], world, assignments, set(), set(),
    )

    task, _ = result[0]
    assert task.task_type == TaskType.DELIVER


def test_hungarian_batches_instead_of_early_deliver():
    """Bot with 1 matching item should pick more if available, not deliver early."""
    world = _make_world(
        bots=[{"id": 0, "position": [4, 1], "inventory": ["milk"]}],
        items=[{"id": "item_1", "type": "bread", "position": [5, 1]}],
        orders=[{
            "id": "order_0",
            "items_required": ["milk", "bread"],
            "items_delivered": [],
            "complete": False, "status": "active",
        }],
    )
    assignments = {0: BotAssignment(bot_id=0)}

    from bot.strategy.hungarian import solve_assignment
    result = solve_assignment(
        [world.state.bots[0]], world, assignments, set(), set(),
    )

    task, _ = result[0]
    assert task.task_type == TaskType.PICK_UP


def test_hungarian_active_item_not_pre_pick():
    """Items matching active order should be PICK_UP, never PRE_PICK even if also in preview."""
    world = _make_world(
        bots=[{"id": 0, "position": [4, 1], "inventory": []}],
        items=[{"id": "item_0", "type": "milk", "position": [5, 1]}],
        orders=[
            {
                "id": "order_0",
                "items_required": ["milk"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            },
            {
                "id": "order_1",
                "items_required": ["milk"],
                "items_delivered": [],
                "complete": False,
                "status": "preview",
            },
        ],
    )

    assignments = {0: BotAssignment(bot_id=0)}
    preview_ids = {"item_0"}

    from bot.strategy.hungarian import solve_assignment
    result = solve_assignment(
        [world.state.bots[0]], world, assignments, set(), preview_ids,
    )

    task, _ = result[0]
    assert task.task_type == TaskType.PICK_UP, \
        "Active order items must be PICK_UP, not PRE_PICK"


def test_hungarian_assigns_routes():
    """With 3 items for same order, a bot should get a multi-item route."""
    world = _make_world(
        bots=[
            {"id": 0, "position": [4, 1], "inventory": []},
        ],
        items=[
            {"id": "item_0", "type": "milk", "position": [5, 1]},
            {"id": "item_1", "type": "bread", "position": [5, 3]},
            {"id": "item_2", "type": "cheese", "position": [5, 5]},
        ],
        orders=[
            {
                "id": "order_0",
                "items_required": ["milk", "bread", "cheese"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            }
        ],
    )

    assignments = {0: BotAssignment(bot_id=0)}

    from bot.strategy.hungarian import solve_assignment
    result = solve_assignment(
        [world.state.bots[0]], world, assignments, set(), set(),
    )

    assert 0 in result
    task, route = result[0]
    assert task.task_type == TaskType.PICK_UP
    # Must get a multi-item route (items are close, capacity allows 3)
    assert route is not None, "Should assign multi-item route, not single item"
    assert len(route.stops) >= 2
    # All items should be from the order
    route_item_ids = route.item_ids
    assert route_item_ids.issubset({"item_0", "item_1", "item_2"})


def test_hungarian_routes_not_treated_as_preview():
    """Active order routes must NOT be treated as preview even when item types overlap."""
    world = _make_world(
        bots=[
            {"id": 0, "position": [4, 1], "inventory": []},
        ],
        items=[
            {"id": "item_0", "type": "milk", "position": [5, 1]},
            {"id": "item_1", "type": "bread", "position": [5, 3]},
            {"id": "item_2", "type": "cheese", "position": [5, 5]},
        ],
        orders=[
            {
                "id": "order_0",
                "items_required": ["milk", "bread", "cheese"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            },
            {
                "id": "order_1",
                "items_required": ["milk", "bread"],
                "items_delivered": [],
                "complete": False,
                "status": "preview",
            },
        ],
    )

    assignments = {0: BotAssignment(bot_id=0)}
    # Mark ALL items as preview (simulating overlap between active and preview orders)
    preview_ids = {"item_0", "item_1", "item_2"}

    from bot.strategy.hungarian import solve_assignment
    result = solve_assignment(
        [world.state.bots[0]], world, assignments, set(), preview_ids,
    )

    task, route = result[0]
    assert task.task_type == TaskType.PICK_UP
    # Must get multi-item route even though all items are in preview_ids
    assert route is not None, \
        "Active order routes should use normal cost formula, not preview formula"
    assert len(route.stops) >= 2


def test_hungarian_two_bots_split_items():
    """Two bots should split items from the same order into separate routes."""
    world = _make_world(
        bots=[
            {"id": 0, "position": [4, 1], "inventory": []},
            {"id": 1, "position": [4, 5], "inventory": []},
        ],
        items=[
            {"id": "item_0", "type": "milk", "position": [5, 1]},
            {"id": "item_1", "type": "bread", "position": [5, 3]},
            {"id": "item_2", "type": "cheese", "position": [5, 5]},
            {"id": "item_3", "type": "butter", "position": [5, 7]},
        ],
        orders=[
            {
                "id": "order_0",
                "items_required": ["milk", "bread", "cheese", "butter"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            }
        ],
    )

    assignments = {
        0: BotAssignment(bot_id=0),
        1: BotAssignment(bot_id=1),
    }

    from bot.strategy.hungarian import solve_assignment
    result = solve_assignment(
        [world.state.bots[0], world.state.bots[1]],
        world, assignments, set(), set(),
    )

    assert len(result) == 2
    task_0, route_0 = result[0]
    task_1, route_1 = result[1]

    # No item should be assigned to both bots
    items_0 = route_0.item_ids if route_0 else {task_0.item_id}
    items_1 = route_1.item_ids if route_1 else {task_1.item_id}
    assert not items_0.intersection(items_1), "Bots should not share items"
