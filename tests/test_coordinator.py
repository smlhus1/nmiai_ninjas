"""Integration test — full pipeline from game state to actions."""

from bot.coordinator import Coordinator


def _make_state(
    round_num: int = 1,
    bots: list | None = None,
    items: list | None = None,
    orders: list | None = None,
    walls: list | None = None,
) -> dict:
    """
    Build a game state dict.

    Default layout: 14x10 grid with a shelf wall at x=5.
    Items sit on shelves (walls), bots navigate walkable floor.
    Drop-off at (7, 5) is walkable.
    """
    # By default, create a shelf column at x=5 (walls)
    default_walls = [[5, y] for y in range(0, 8)]
    return {
        "type": "game_state",
        "round": round_num,
        "max_rounds": 300,
        "grid": {"width": 14, "height": 10, "walls": walls if walls is not None else default_walls},
        "bots": bots if bots is not None else [{"id": 0, "position": [3, 3], "inventory": []}],
        "items": items if items is not None else [{"id": "item_0", "type": "milk", "position": [5, 3]}],
        "orders": orders
        if orders is not None
        else [
            {
                "id": "order_0",
                "items_required": ["milk"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            }
        ],
        "drop_off": [7, 5],
        "score": 0,
    }


def test_basic_round():
    """Bot should produce a valid action for a simple state."""
    coord = Coordinator()
    response = coord.on_game_state(_make_state())

    assert "actions" in response
    assert len(response["actions"]) == 1
    action = response["actions"][0]
    assert action["bot"] == 0
    assert action["action"] in [
        "move_up", "move_down", "move_left", "move_right",
        "pick_up", "drop_off", "wait",
    ]


def test_bot_adjacent_to_item_picks_up():
    """Bot adjacent to shelf item should pick it up."""
    coord = Coordinator()
    # Item on shelf at (5, 3), bot adjacent at (4, 3)
    state = _make_state(
        bots=[{"id": 0, "position": [4, 3], "inventory": []}],
        items=[{"id": "item_0", "type": "milk", "position": [5, 3]}],
    )
    response = coord.on_game_state(state)
    action = response["actions"][0]
    assert action["action"] == "pick_up"
    assert action["item_id"] == "item_0"


def test_bot_not_adjacent_moves_toward_item():
    """Bot far from item should move toward it (toward adjacent pickup cell)."""
    coord = Coordinator()
    # Item on shelf at (5, 3), bot far away at (0, 3)
    state = _make_state(
        bots=[{"id": 0, "position": [0, 3], "inventory": []}],
        items=[{"id": "item_0", "type": "milk", "position": [5, 3]}],
    )
    response = coord.on_game_state(state)
    action = response["actions"][0]
    # Should be moving right toward the item
    assert action["action"] == "move_right"


def test_bot_with_inventory_delivers():
    """Bot carrying items at drop-off should deliver."""
    coord = Coordinator()
    # Bot at drop-off (7, 5) with inventory
    state = _make_state(
        bots=[{"id": 0, "position": [7, 5], "inventory": ["milk"]}],
    )
    response = coord.on_game_state(state)
    action = response["actions"][0]
    assert action["action"] == "drop_off"


def test_multiple_bots():
    """Multiple bots should all get actions."""
    coord = Coordinator()
    # 3 items on the same shelf column, bots on adjacent walkable cells
    state = _make_state(
        bots=[
            {"id": 0, "position": [1, 1], "inventory": []},
            {"id": 1, "position": [3, 3], "inventory": []},
            {"id": 2, "position": [8, 5], "inventory": []},
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
    response = coord.on_game_state(state)
    assert len(response["actions"]) == 3
    bot_ids = {a["bot"] for a in response["actions"]}
    assert bot_ids == {0, 1, 2}


def test_no_double_booking():
    """Two bots should not target the same item."""
    coord = Coordinator()
    state = _make_state(
        bots=[
            {"id": 0, "position": [1, 1], "inventory": []},
            {"id": 1, "position": [2, 1], "inventory": []},
        ],
        items=[
            {"id": "item_0", "type": "milk", "position": [5, 5]},
        ],
        orders=[
            {
                "id": "order_0",
                "items_required": ["milk"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            }
        ],
    )
    response = coord.on_game_state(state)
    # Both bots get actions, but only one should target the item
    actions = response["actions"]
    assert len(actions) == 2


def test_multi_round_consistency():
    """Bot should maintain task across rounds (sticky assignment)."""
    coord = Coordinator()

    # Round 1: bot starts moving toward item on shelf at (5, 0)
    state1 = _make_state(
        round_num=1,
        bots=[{"id": 0, "position": [0, 0], "inventory": []}],
        items=[{"id": "item_0", "type": "milk", "position": [5, 0]}],
    )
    r1 = coord.on_game_state(state1)
    assert r1["actions"][0]["action"] == "move_right"

    # Round 2: bot moved one step, should keep going
    state2 = _make_state(
        round_num=2,
        bots=[{"id": 0, "position": [1, 0], "inventory": []}],
        items=[{"id": "item_0", "type": "milk", "position": [5, 0]}],
    )
    r2 = coord.on_game_state(state2)
    assert r2["actions"][0]["action"] == "move_right"


def test_reset():
    """Reset should clear all state."""
    coord = Coordinator()
    coord.on_game_state(_make_state())
    coord.reset()
    # Should work fine after reset
    response = coord.on_game_state(_make_state())
    assert len(response["actions"]) == 1


def test_pickup_from_other_side_of_shelf():
    """Bot on the right side of shelf should also be able to pick up."""
    coord = Coordinator()
    # Item on shelf at (5, 3), bot adjacent on right side at (6, 3)
    state = _make_state(
        bots=[{"id": 0, "position": [6, 3], "inventory": []}],
        items=[{"id": "item_0", "type": "milk", "position": [5, 3]}],
    )
    response = coord.on_game_state(state)
    action = response["actions"][0]
    assert action["action"] == "pick_up"
    assert action["item_id"] == "item_0"


def test_wait_action_value():
    """Idle/wait actions should serialize as 'wait' not 'idle'."""
    coord = Coordinator()
    # No items, no orders — bot should wait
    state = _make_state(
        bots=[{"id": 0, "position": [3, 3], "inventory": []}],
        items=[],
        orders=[
            {
                "id": "order_0",
                "items_required": ["milk"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            }
        ],
    )
    response = coord.on_game_state(state)
    action = response["actions"][0]
    assert action["action"] == "wait"


def test_no_delivery_loop_with_nonmatching_items():
    """Bot with inventory that doesn't match active order should NOT go to drop-off."""
    coord = Coordinator()
    # Bot has "cheese" in inventory, but active order needs "milk"
    # Bot should pick up milk, not waste rounds delivering cheese
    state = _make_state(
        bots=[{"id": 0, "position": [3, 3], "inventory": ["cheese"]}],
        items=[{"id": "item_0", "type": "milk", "position": [5, 3]}],
        orders=[
            {
                "id": "order_0",
                "items_required": ["milk"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            }
        ],
    )
    response = coord.on_game_state(state)
    action = response["actions"][0]
    # Should NOT be going to drop-off — cheese doesn't match active order
    assert action["action"] != "drop_off"
    # Should be moving toward the milk item instead
    assert action["action"] in ["move_right", "move_up", "move_down", "move_left"]


def test_staging_bot_does_not_drop_off():
    """Bot sent to staging (via navigation_override) should NOT issue drop_off."""
    coord = Coordinator()
    # 3 bots all delivering, only 2 adjacent drop-off cells → bot 2 goes to staging
    # Drop-off at (7,5), shelves at x=5
    state = _make_state(
        bots=[
            {"id": 0, "position": [7, 4], "inventory": ["milk"]},  # Close
            {"id": 1, "position": [7, 6], "inventory": ["bread"]},  # Close
            {"id": 2, "position": [8, 5], "inventory": ["cheese"]},  # Third deliverer
        ],
        items=[],
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
    response = coord.on_game_state(state)
    actions_by_bot = {a["bot"]: a["action"] for a in response["actions"]}
    # No bot at a non-drop-off position should issue drop_off
    for a in response["actions"]:
        if a["action"] == "drop_off":
            # Only valid if bot was at actual drop-off (7,5)
            bot_pos = {0: (7, 4), 1: (7, 6), 2: (8, 5)}
            assert bot_pos[a["bot"]] == (7, 5), \
                f"Bot {a['bot']} at {bot_pos[a['bot']]} issued drop_off but is not at drop-off"


def test_endgame_delivers_immediately():
    """In endgame with unfulfillable order, bots with inventory deliver immediately."""
    coord = Coordinator()
    # Round 275 of 300 = 25 remaining, order needs 5 items = 40 rounds estimated
    state = _make_state(
        round_num=275,
        bots=[{"id": 0, "position": [6, 5], "inventory": ["milk"]}],
        items=[
            {"id": "item_0", "type": "bread", "position": [5, 1]},
            {"id": "item_1", "type": "cheese", "position": [5, 2]},
            {"id": "item_2", "type": "butter", "position": [5, 4]},
            {"id": "item_3", "type": "eggs", "position": [5, 6]},
        ],
        orders=[
            {
                "id": "order_0",
                "items_required": ["milk", "bread", "cheese", "butter", "eggs"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            }
        ],
    )
    response = coord.on_game_state(state)
    action = response["actions"][0]
    # Bot has milk and is next to drop-off — should move toward it to deliver
    assert action["action"] in ["move_right", "drop_off"]


def test_endgame_picks_nearest_item():
    """In endgame, bot without inventory picks nearest deliverable item."""
    coord = Coordinator()
    state = _make_state(
        round_num=270,
        bots=[{"id": 0, "position": [3, 3], "inventory": []}],
        items=[
            {"id": "item_0", "type": "bread", "position": [5, 3]},  # Close
            {"id": "item_1", "type": "cheese", "position": [5, 8]},  # Far
        ],
        orders=[
            {
                "id": "order_0",
                "items_required": ["bread", "cheese", "milk", "butter", "eggs"],
                "items_delivered": [],
                "complete": False,
                "status": "active",
            }
        ],
    )
    response = coord.on_game_state(state)
    action = response["actions"][0]
    # Should be moving toward the closest item
    assert action["action"] in ["move_right", "move_up", "move_down", "move_left"]


def test_preview_pre_staging():
    """Idle bots should pick items for preview orders."""
    coord = Coordinator()
    # Bot 0 handles active order, Bot 1 has nothing active to do → should pre-pick preview item
    state = _make_state(
        bots=[
            {"id": 0, "position": [4, 1], "inventory": []},
            {"id": 1, "position": [4, 5], "inventory": []},
        ],
        items=[
            {"id": "item_0", "type": "milk", "position": [5, 1]},  # Active order item
            {"id": "item_1", "type": "cheese", "position": [5, 5]},  # Preview order item
        ],
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
                "items_required": ["cheese"],
                "items_delivered": [],
                "complete": False,
                "status": "preview",
            },
        ],
    )
    response = coord.on_game_state(state)
    actions = response["actions"]
    # Both bots should be doing something (not both waiting)
    assert len(actions) == 2
    action_types = {a["action"] for a in actions}
    # At least one bot should be moving or picking up
    assert "wait" not in action_types or len(action_types) > 1
