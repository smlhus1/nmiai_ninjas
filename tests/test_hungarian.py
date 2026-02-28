"""Tests for Hungarian (optimal) assignment."""

from bot.models import GameState, Grid, Bot, Item
from bot.engine.pathfinding import PathEngine
from bot.engine.world_model import WorldModel
from bot.strategy.task import BotAssignment, TaskType
from bot.strategy.hungarian import solve_assignment


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
    """Hungarian should assign bots to items."""
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

    result = solve_assignment(
        [world.state.bots[0], world.state.bots[1]],
        world, assignments, set(), set(),
    )

    # Both bots should get tasks
    assert len(result) == 2
    # Both should be PICK_UP
    assert all(t.task_type == TaskType.PICK_UP for t in result.values())
    # No double-booking
    item_ids = {t.item_id for t in result.values()}
    assert len(item_ids) == 2


def test_hungarian_beats_greedy():
    """
    Classic example: Hungarian should find lower total cost than greedy.
    Bot 0 is close to item A, Bot 1 is close to item B.
    Greedy might suboptimally assign Bot 0 to B (if B has higher order value).
    """
    world = _make_world(
        bots=[
            {"id": 0, "position": [4, 1], "inventory": []},  # Near milk at (5,1)
            {"id": 1, "position": [4, 3], "inventory": []},  # Near bread at (5,3)
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

    result = solve_assignment(
        [world.state.bots[0], world.state.bots[1]],
        world, assignments, set(), set(),
    )

    # Optimal: Bot 0 -> milk (1 step), Bot 1 -> bread (1 step)
    assert result[0].item_id == "item_0"  # milk
    assert result[1].item_id == "item_1"  # bread


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

    result = solve_assignment(
        [world.state.bots[0]], world, assignments, claimed, set(),
    )

    assert len(result) == 1
    assert result[0].item_id == "item_1"  # Should get bread, not milk


def test_hungarian_delivers_with_inventory():
    """Bot with matching inventory should deliver, not pick."""
    world = _make_world(
        bots=[{"id": 0, "position": [6, 5], "inventory": ["milk"]}],
        items=[{"id": "item_1", "type": "bread", "position": [5, 3]}],
    )

    assignments = {0: BotAssignment(bot_id=0)}

    result = solve_assignment(
        [world.state.bots[0]], world, assignments, set(), set(),
    )

    assert result[0].task_type == TaskType.DELIVER
