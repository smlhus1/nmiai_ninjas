"""Tests for data models — parsing, immutability, helper methods."""

from bot.models import GameState, Grid, Bot, Item, Order, OrderStatus, Action, apply_move


SAMPLE_STATE = {
    "type": "game_state",
    "round": 42,
    "max_rounds": 300,
    "grid": {"width": 14, "height": 10, "walls": [[1, 1], [1, 2]]},
    "bots": [{"id": 0, "position": [3, 7], "inventory": ["milk"]}],
    "items": [{"id": "item_0", "type": "milk", "position": [2, 1]}],
    "orders": [
        {
            "id": "order_0",
            "items_required": ["milk", "bread"],
            "items_delivered": ["milk"],
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
    "drop_off": [6, 9],
    "score": 12,
}


def test_parse_game_state():
    state = GameState.from_dict(SAMPLE_STATE)
    assert state.round == 42
    assert state.max_rounds == 300
    assert state.score == 12
    assert state.drop_off == (6, 9)
    assert len(state.bots) == 1
    assert len(state.items) == 1
    assert len(state.orders) == 2


def test_grid_walls():
    state = GameState.from_dict(SAMPLE_STATE)
    assert (1, 1) in state.grid.walls
    assert (1, 2) in state.grid.walls
    assert state.grid.is_walkable((0, 0))
    assert not state.grid.is_walkable((1, 1))
    assert not state.grid.is_walkable((-1, 0))  # out of bounds


def test_bot_parsing():
    state = GameState.from_dict(SAMPLE_STATE)
    bot = state.bots[0]
    assert bot.id == 0
    assert bot.position == (3, 7)
    assert bot.inventory == ("milk",)


def test_order_remaining():
    state = GameState.from_dict(SAMPLE_STATE)
    order = state.active_orders[0]
    assert order.items_remaining == ("bread",)  # milk already delivered


def test_order_filters():
    state = GameState.from_dict(SAMPLE_STATE)
    assert len(state.active_orders) == 1
    assert len(state.preview_orders) == 1
    assert state.active_orders[0].id == "order_0"


def test_rounds_remaining():
    state = GameState.from_dict(SAMPLE_STATE)
    assert state.rounds_remaining == 258


def test_apply_move():
    assert apply_move((3, 3), Action.MOVE_UP) == (3, 2)
    assert apply_move((3, 3), Action.MOVE_DOWN) == (3, 4)
    assert apply_move((3, 3), Action.MOVE_LEFT) == (2, 3)
    assert apply_move((3, 3), Action.MOVE_RIGHT) == (4, 3)
    assert apply_move((3, 3), Action.PICK_UP) == (3, 3)
    assert apply_move((3, 3), Action.WAIT) == (3, 3)


def test_action_wait_value():
    """Action.WAIT should serialize to 'wait' for the API."""
    assert Action.WAIT.value == "wait"


def test_bot_command_serialization():
    from bot.models import BotCommand

    cmd = BotCommand(bot_id=0, action=Action.MOVE_UP)
    assert cmd.to_dict() == {"bot": 0, "action": "move_up"}

    cmd_pickup = BotCommand(bot_id=1, action=Action.PICK_UP, item_id="item_3")
    assert cmd_pickup.to_dict() == {"bot": 1, "action": "pick_up", "item_id": "item_3"}

    cmd_wait = BotCommand(bot_id=0, action=Action.WAIT)
    assert cmd_wait.to_dict() == {"bot": 0, "action": "wait"}
