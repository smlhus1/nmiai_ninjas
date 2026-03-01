"""
Collision model verification: confirm Simulator matches game server rules.

Tests every game mechanic that could diverge between simulator and server:
- Sequential bot-ID-order action resolution
- Movement collision rules
- Pickup adjacency / inventory / item validity
- Drop-off: only matching active-order items delivered
- Auto-delivery on order transition (for ALL bots)
- Item respawn with new IDs
- Order lifecycle: active → complete, preview → active, hidden → preview

Run: py -m pytest Simulering/offline/test_collision_model.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path for imports
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Simulering.offline.simulator import Simulator, SimBot, SimItem, SimOrder

Pos = tuple[int, int]


def _make_sim(
    width: int = 8,
    height: int = 6,
    walls: set[Pos] | None = None,
    shelves: set[Pos] | None = None,
    shelf_types: dict[Pos, str] | None = None,
    drop_off: Pos = (1, 4),
    spawns: list[Pos] | None = None,
    orders: list[dict] | None = None,
    max_rounds: int = 50,
) -> Simulator:
    """Build a minimal Simulator for testing specific mechanics."""
    if walls is None:
        walls = set()
    if shelves is None:
        shelves = {(3, 1), (3, 3)}
    if shelf_types is None:
        shelf_types = {(3, 1): "milk", (3, 3): "bread"}
    if spawns is None:
        spawns = [(5, 4)]
    if orders is None:
        orders = [
            {"id": "order_0", "items_required": ["milk", "bread"]},
            {"id": "order_1", "items_required": ["milk"]},
        ]

    return Simulator(
        width=width, height=height,
        walls=walls, shelves=shelves,
        drop_off=drop_off,
        spawn_positions=spawns,
        order_sequence=orders,
        item_types_at_shelves=shelf_types,
        max_rounds=max_rounds,
    )


# =====================================================================
# Movement rules
# =====================================================================

class TestMovement:
    def test_valid_move(self):
        """Bot moves to an empty walkable cell."""
        sim = _make_sim(spawns=[(4, 2)])
        state = sim.reset()
        bot = state.bots[0]
        assert bot.position == (4, 2)

        state, _ = sim.step([{"bot": 0, "action": "move_left"}])
        assert state.bots[0].position == (3, 2), "Bot should move left"

    def test_move_into_wall_fails_silently(self):
        """Moving into a wall = wait. No error, no position change."""
        sim = _make_sim(walls={(4, 2)}, spawns=[(5, 2)])
        state = sim.reset()

        state, _ = sim.step([{"bot": 0, "action": "move_left"}])
        assert state.bots[0].position == (5, 2), "Wall should block movement"

    def test_move_into_shelf_fails_silently(self):
        """Shelves are non-walkable. Moving into one fails."""
        sim = _make_sim(spawns=[(4, 1)])
        state = sim.reset()

        state, _ = sim.step([{"bot": 0, "action": "move_left"}])
        assert state.bots[0].position == (4, 1), "Shelf at (3,1) should block"

    def test_move_out_of_bounds_fails(self):
        """Moving off the grid edge fails silently."""
        sim = _make_sim(spawns=[(0, 0)])
        state = sim.reset()

        state, _ = sim.step([{"bot": 0, "action": "move_left"}])
        assert state.bots[0].position == (0, 0)

        state, _ = sim.step([{"bot": 0, "action": "move_up"}])
        assert state.bots[0].position == (0, 0)

    def test_invalid_action_is_wait(self):
        """Unknown action string treated as wait."""
        sim = _make_sim(spawns=[(4, 2)])
        state = sim.reset()

        state, _ = sim.step([{"bot": 0, "action": "teleport"}])
        assert state.bots[0].position == (4, 2)

    def test_missing_action_is_wait(self):
        """No action for a bot = wait."""
        sim = _make_sim(spawns=[(4, 2)])
        sim.reset()

        state, _ = sim.step([])
        assert state.bots[0].position == (4, 2)


# =====================================================================
# Multi-bot collision: sequential ID-order resolution
# =====================================================================

class TestMultiBotCollision:
    def test_bot_blocks_other_bot(self):
        """Bot can't move into a cell occupied by another bot."""
        sim = _make_sim(spawns=[(4, 2), (5, 2)])
        state = sim.reset()

        state, _ = sim.step([
            {"bot": 0, "action": "move_right"},
            {"bot": 1, "action": "wait"},
        ])
        assert state.bots[0].position == (4, 2), "Bot 1 at (5,2) blocks bot 0"

    def test_sequential_id_order(self):
        """Lower-ID bots move first. Bot 0's move resolves before bot 1's."""
        sim = _make_sim(spawns=[(4, 2), (5, 2)])
        state = sim.reset()

        # Both bots move right. Bot 0 moves first: (4,2)→(5,2).
        # But (5,2) has bot 1. So bot 0 is blocked.
        # Then bot 1 moves: (5,2)→(6,2). Succeeds.
        state, _ = sim.step([
            {"bot": 0, "action": "move_right"},
            {"bot": 1, "action": "move_right"},
        ])
        # Bot 0 blocked by bot 1 (bot 1 hasn't moved yet when bot 0 tries)
        assert state.bots[0].position == (4, 2), "Bot 0 blocked by bot 1's original pos"
        assert state.bots[1].position == (6, 2), "Bot 1 moves right successfully"

    def test_follow_chain(self):
        """Bot 0 follows bot 1 who follows bot 2 — sequential resolution."""
        sim = _make_sim(spawns=[(4, 2), (5, 2), (6, 2)])
        state = sim.reset()

        # All move right. Bot 0 blocked by 1 (still at 5,2).
        # Bot 1 blocked by 2 (still at 6,2). Bot 2 moves to (7,2).
        state, _ = sim.step([
            {"bot": 0, "action": "move_right"},
            {"bot": 1, "action": "move_right"},
            {"bot": 2, "action": "move_right"},
        ])
        assert state.bots[0].position == (4, 2), "Bot 0 blocked"
        assert state.bots[1].position == (5, 2), "Bot 1 blocked"
        assert state.bots[2].position == (7, 2), "Bot 2 moves"

    def test_swap_prevented(self):
        """Two bots trying to swap positions: lower ID blocked."""
        sim = _make_sim(spawns=[(4, 2), (5, 2)])
        sim.reset()

        # Bot 0 wants (5,2), bot 1 wants (4,2).
        # Bot 0 tries first: (5,2) has bot 1 → blocked.
        # Bot 1 tries: (4,2) still has bot 0 → blocked.
        state, _ = sim.step([
            {"bot": 0, "action": "move_right"},
            {"bot": 1, "action": "move_left"},
        ])
        assert state.bots[0].position == (4, 2), "Swap blocked for bot 0"
        assert state.bots[1].position == (5, 2), "Swap blocked for bot 1"

    def test_lower_id_vacates_for_higher(self):
        """When lower-ID bot moves away, higher-ID can take the vacated spot."""
        sim = _make_sim(spawns=[(4, 2), (5, 2)])
        sim.reset()

        # Bot 0 moves up (vacates 4,2), bot 1 moves left into (4,2).
        # Bot 0 resolves first: (4,2)→(4,1). Position (4,2) is now free.
        # Bot 1 resolves: (5,2)→(4,2). Position (4,2) is free → success.
        state, _ = sim.step([
            {"bot": 0, "action": "move_up"},
            {"bot": 1, "action": "move_left"},
        ])
        assert state.bots[0].position == (4, 1), "Bot 0 moved up"
        assert state.bots[1].position == (4, 2), "Bot 1 took vacated spot"

    def test_two_bots_same_target(self):
        """Two bots targeting same cell: lower ID wins."""
        sim = _make_sim(spawns=[(4, 2), (6, 2)])
        sim.reset()

        # Both move toward (5,2). Bot 0 arrives first (lower ID).
        state, _ = sim.step([
            {"bot": 0, "action": "move_right"},
            {"bot": 1, "action": "move_left"},
        ])
        assert state.bots[0].position == (5, 2), "Bot 0 gets the cell"
        assert state.bots[1].position == (6, 2), "Bot 1 blocked"


# =====================================================================
# Pickup rules
# =====================================================================

class TestPickup:
    def test_pickup_adjacent(self):
        """Bot adjacent (Manhattan 1) to shelf can pick up."""
        sim = _make_sim(spawns=[(4, 1)])
        state = sim.reset()

        milk_item = next(i for i in state.items if i.item_type == "milk")
        state, _ = sim.step([
            {"bot": 0, "action": "pick_up", "item_id": milk_item.id},
        ])
        assert state.bots[0].inventory == ["milk"]

    def test_pickup_not_adjacent_fails(self):
        """Bot NOT adjacent (Manhattan > 1) can't pick up."""
        sim = _make_sim(spawns=[(5, 1)])
        state = sim.reset()

        milk_item = next(i for i in state.items if i.item_type == "milk")
        state, _ = sim.step([
            {"bot": 0, "action": "pick_up", "item_id": milk_item.id},
        ])
        assert state.bots[0].inventory == [], "Manhattan 2 = not adjacent"

    def test_pickup_diagonal_fails(self):
        """Diagonal adjacency (Manhattan 2) doesn't count."""
        sim = _make_sim(spawns=[(4, 2)])
        state = sim.reset()

        milk_item = next(i for i in state.items if i.item_type == "milk")
        state, _ = sim.step([
            {"bot": 0, "action": "pick_up", "item_id": milk_item.id},
        ])
        assert state.bots[0].inventory == [], "Diagonal = Manhattan 2"

    def test_pickup_full_inventory_fails(self):
        """Can't pick up with 3 items already in inventory."""
        sim = _make_sim(
            shelves={(3, 1), (3, 2), (3, 3), (3, 4)},
            shelf_types={(3, 1): "a", (3, 2): "b", (3, 3): "c", (3, 4): "d"},
            orders=[{"id": "o0", "items_required": ["a", "b", "c", "d"]}],
            spawns=[(4, 1)],
        )
        state = sim.reset()

        # Pick up 3 items (moving between shelves)
        items = {i.item_type: i.id for i in state.items}
        state, _ = sim.step([{"bot": 0, "action": "pick_up", "item_id": items["a"]}])
        assert len(state.bots[0].inventory) == 1

        state, _ = sim.step([{"bot": 0, "action": "move_down"}])
        items_r2 = {i.item_type: i.id for i in state.items}
        state, _ = sim.step([{"bot": 0, "action": "pick_up", "item_id": items_r2["b"]}])
        assert len(state.bots[0].inventory) == 2

        state, _ = sim.step([{"bot": 0, "action": "move_down"}])
        items_r4 = {i.item_type: i.id for i in state.items}
        state, _ = sim.step([{"bot": 0, "action": "pick_up", "item_id": items_r4["c"]}])
        assert len(state.bots[0].inventory) == 3

        # Fourth pickup should fail
        state, _ = sim.step([{"bot": 0, "action": "move_down"}])
        items_r6 = {i.item_type: i.id for i in state.items}
        state, _ = sim.step([{"bot": 0, "action": "pick_up", "item_id": items_r6["d"]}])
        assert len(state.bots[0].inventory) == 3, "Inventory capped at 3"

    def test_pickup_invalid_item_id(self):
        """Pick up with non-existent item_id fails silently."""
        sim = _make_sim(spawns=[(4, 1)])
        sim.reset()

        state, _ = sim.step([{"bot": 0, "action": "pick_up", "item_id": "fake_id"}])
        assert state.bots[0].inventory == []

    def test_pickup_no_item_id(self):
        """Pick up without item_id field fails silently."""
        sim = _make_sim(spawns=[(4, 1)])
        sim.reset()

        state, _ = sim.step([{"bot": 0, "action": "pick_up"}])
        assert state.bots[0].inventory == []


# =====================================================================
# Drop-off rules
# =====================================================================

class TestDropOff:
    def test_dropoff_matching_items(self):
        """Items matching active order are delivered. Score +1 per item."""
        sim = _make_sim(drop_off=(1, 4), spawns=[(2, 1)])
        state = sim.reset()

        # Pick up milk (adjacent to shelf at (3,1))
        milk = next(i for i in state.items if i.item_type == "milk")
        state, _ = sim.step([{"bot": 0, "action": "pick_up", "item_id": milk.id}])
        assert state.bots[0].inventory == ["milk"]

        # Navigate to drop-off (simplified: just teleport by setting position)
        # Instead, use the step method to navigate
        sim._bots[0].position = (1, 4)
        state, _ = sim.step([{"bot": 0, "action": "drop_off"}])

        assert state.bots[0].inventory == [], "Milk delivered"
        assert state.score == 1, "+1 for delivering milk"

    def test_dropoff_non_matching_stays(self):
        """Items NOT matching active order stay in inventory."""
        sim = _make_sim(
            orders=[{"id": "o0", "items_required": ["bread"]}],
            spawns=[(2, 1)],
        )
        state = sim.reset()

        # Pick up milk (active order wants bread, not milk)
        milk = next(i for i in state.items if i.item_type == "milk")
        state, _ = sim.step([{"bot": 0, "action": "pick_up", "item_id": milk.id}])

        sim._bots[0].position = sim.drop_off
        state, _ = sim.step([{"bot": 0, "action": "drop_off"}])

        assert state.bots[0].inventory == ["milk"], "Milk doesn't match, stays"
        assert state.score == 0

    def test_dropoff_not_on_cell_fails(self):
        """Drop off when not on drop_off cell does nothing."""
        sim = _make_sim(spawns=[(2, 1)])
        state = sim.reset()

        milk = next(i for i in state.items if i.item_type == "milk")
        state, _ = sim.step([{"bot": 0, "action": "pick_up", "item_id": milk.id}])
        state, _ = sim.step([{"bot": 0, "action": "drop_off"}])

        assert state.bots[0].inventory == ["milk"], "Not at drop-off"

    def test_dropoff_partial_delivery(self):
        """Mix of matching and non-matching: only matching items delivered.
        When order completes, auto-delivery fires for next order's matches."""
        sim = _make_sim(
            shelves={(3, 1), (3, 2)},
            shelf_types={(3, 1): "milk", (3, 2): "juice"},
            orders=[
                {"id": "o0", "items_required": ["milk"]},
                {"id": "o1", "items_required": ["juice"]},
                {"id": "o2", "items_required": ["milk"]},
            ],
            spawns=[(4, 1)],
        )
        state = sim.reset()

        # Pick milk
        milk = next(i for i in state.items if i.item_type == "milk")
        state, _ = sim.step([{"bot": 0, "action": "pick_up", "item_id": milk.id}])

        # Pick juice
        sim._bots[0].position = (4, 2)
        state = sim._get_state()
        juice = next(i for i in state.items if i.item_type == "juice")
        state, _ = sim.step([{"bot": 0, "action": "pick_up", "item_id": juice.id}])
        assert sorted(state.bots[0].inventory) == ["juice", "milk"]

        # Drop off: milk matches o0 → delivered → o0 complete → o1 active
        # juice matches o1 → auto-delivered → o1 complete → o2 active
        sim._bots[0].position = sim.drop_off
        state, _ = sim.step([{"bot": 0, "action": "drop_off"}])

        assert state.bots[0].inventory == [], "Both delivered via chain auto-delivery"
        # milk(+1) + o0(+5) + juice_auto(+1) + o1(+5) = 12
        assert state.score == 12

    def test_dropoff_non_matching_stays_in_inventory(self):
        """Items NOT matching any transitioning order stay in inventory."""
        sim = _make_sim(
            shelves={(3, 1), (3, 2)},
            shelf_types={(3, 1): "milk", (3, 2): "juice"},
            orders=[
                {"id": "o0", "items_required": ["milk"]},
                {"id": "o1", "items_required": ["milk"]},  # o1 also wants milk, NOT juice
            ],
            spawns=[(4, 1)],
        )
        state = sim.reset()

        # Pick milk + juice
        milk = next(i for i in state.items if i.item_type == "milk")
        state, _ = sim.step([{"bot": 0, "action": "pick_up", "item_id": milk.id}])
        sim._bots[0].position = (4, 2)
        state = sim._get_state()
        juice = next(i for i in state.items if i.item_type == "juice")
        state, _ = sim.step([{"bot": 0, "action": "pick_up", "item_id": juice.id}])

        # Deliver: milk matches o0, juice doesn't match o0.
        # o0 completes → o1 active. Juice doesn't match o1 either → stays.
        sim._bots[0].position = sim.drop_off
        state, _ = sim.step([{"bot": 0, "action": "drop_off"}])

        assert state.bots[0].inventory == ["juice"], "Juice doesn't match any order"


# =====================================================================
# Order lifecycle + completion bonus
# =====================================================================

class TestOrderLifecycle:
    def test_order_completion_bonus(self):
        """Completing an order gives +5 bonus on top of per-item +1."""
        sim = _make_sim(
            shelves={(3, 1)},
            shelf_types={(3, 1): "milk"},
            orders=[
                {"id": "o0", "items_required": ["milk"]},
                {"id": "o1", "items_required": ["milk"]},
            ],
            spawns=[(4, 1)],
        )
        state = sim.reset()

        milk = next(i for i in state.items if i.item_type == "milk")
        sim.step([{"bot": 0, "action": "pick_up", "item_id": milk.id}])
        sim._bots[0].position = sim.drop_off
        state, _ = sim.step([{"bot": 0, "action": "drop_off"}])

        assert state.score == 6, "1 item + 5 completion bonus = 6"

    def test_preview_becomes_active(self):
        """After completing active order, preview becomes active."""
        sim = _make_sim(
            shelves={(3, 1)},
            shelf_types={(3, 1): "milk"},
            orders=[
                {"id": "o0", "items_required": ["milk"]},
                {"id": "o1", "items_required": ["milk"]},
                {"id": "o2", "items_required": ["milk"]},
            ],
            spawns=[(4, 1)],
        )
        state = sim.reset()

        # Verify initial order states
        active = [o for o in state.orders if o.status == "active"]
        preview = [o for o in state.orders if o.status == "preview"]
        assert len(active) == 1 and active[0].id == "o0"
        assert len(preview) == 1 and preview[0].id == "o1"

        # Complete o0
        milk = next(i for i in state.items if i.item_type == "milk")
        sim.step([{"bot": 0, "action": "pick_up", "item_id": milk.id}])
        sim._bots[0].position = sim.drop_off
        state, _ = sim.step([{"bot": 0, "action": "drop_off"}])

        # o1 should now be active, o2 should be preview
        active = [o for o in state.orders if o.status == "active"]
        preview = [o for o in state.orders if o.status == "preview"]
        assert len(active) == 1 and active[0].id == "o1"
        assert len(preview) == 1 and preview[0].id == "o2"


# =====================================================================
# Auto-delivery on order transition
# =====================================================================

class TestAutoDelivery:
    def test_auto_delivery_delivering_bot(self):
        """Bot that completes an order: matching next-order items auto-deliver."""
        sim = _make_sim(
            shelves={(3, 1), (3, 3)},
            shelf_types={(3, 1): "milk", (3, 3): "bread"},
            orders=[
                {"id": "o0", "items_required": ["milk"]},
                {"id": "o1", "items_required": ["bread"]},
                {"id": "o2", "items_required": ["milk"]},
            ],
            spawns=[(4, 1)],
        )
        state = sim.reset()

        # Pick milk
        milk = next(i for i in state.items if i.item_type == "milk")
        sim.step([{"bot": 0, "action": "pick_up", "item_id": milk.id}])

        # Pick bread
        sim._bots[0].position = (4, 3)
        state = sim._get_state()
        bread = next(i for i in state.items if i.item_type == "bread")
        sim.step([{"bot": 0, "action": "pick_up", "item_id": bread.id}])
        assert sorted(sim._bots[0].inventory) == ["bread", "milk"]

        # Deliver: milk matches o0. Complete → o1 activates.
        # Bread matches o1 → auto-delivered!
        sim._bots[0].position = sim.drop_off
        state, _ = sim.step([{"bot": 0, "action": "drop_off"}])

        assert state.bots[0].inventory == [], "Both items delivered via auto-delivery"
        # milk: +1 (delivered) + 5 (o0 complete) = 6
        # bread: +1 (auto-delivered) + 5 (o1 complete) = 6
        # Total: 12
        assert state.score == 12, f"Expected 12, got {state.score}"

    def test_auto_delivery_other_bot(self):
        """Auto-delivery fires for ALL bots, not just the one at drop-off."""
        sim = _make_sim(
            shelves={(3, 1), (3, 3)},
            shelf_types={(3, 1): "milk", (3, 3): "bread"},
            orders=[
                {"id": "o0", "items_required": ["milk"]},
                {"id": "o1", "items_required": ["bread"]},
                {"id": "o2", "items_required": ["milk"]},
            ],
            spawns=[(4, 1), (4, 3)],
        )
        state = sim.reset()

        # Bot 0 picks milk
        milk = next(i for i in state.items if i.item_type == "milk")
        # Bot 1 picks bread
        bread = next(i for i in state.items if i.item_type == "bread")
        sim.step([
            {"bot": 0, "action": "pick_up", "item_id": milk.id},
            {"bot": 1, "action": "pick_up", "item_id": bread.id},
        ])
        assert sim._bots[0].inventory == ["milk"]
        assert sim._bots[1].inventory == ["bread"]

        # Bot 0 delivers milk → o0 complete → o1 active → bot 1's bread auto-delivers!
        sim._bots[0].position = sim.drop_off
        state, _ = sim.step([
            {"bot": 0, "action": "drop_off"},
            {"bot": 1, "action": "wait"},
        ])

        assert state.bots[0].inventory == []
        assert state.bots[1].inventory == [], "Bot 1's bread auto-delivered"
        # milk(+1) + o0_complete(+5) + bread_auto(+1) + o1_complete(+5) = 12
        assert state.score == 12


# =====================================================================
# Item respawn
# =====================================================================

class TestItemRespawn:
    def test_picked_item_respawns(self):
        """Picked item respawns at same shelf with new ID."""
        sim = _make_sim(spawns=[(4, 1)])
        state = sim.reset()

        milk = next(i for i in state.items if i.item_type == "milk")
        old_id = milk.id
        old_pos = milk.position

        sim.step([{"bot": 0, "action": "pick_up", "item_id": milk.id}])
        state = sim._get_state()

        # Find the respawned milk
        new_milk = next(
            (i for i in state.items if i.item_type == "milk" and i.position == old_pos),
            None,
        )
        assert new_milk is not None, "Milk should respawn at same shelf"
        assert new_milk.id != old_id, "Respawned item gets a new ID"

    def test_shelf_count_preserved(self):
        """Number of items on map stays constant (1 per shelf)."""
        sim = _make_sim(spawns=[(4, 1)])
        state = sim.reset()
        initial_count = len(state.items)

        milk = next(i for i in state.items if i.item_type == "milk")
        state, _ = sim.step([{"bot": 0, "action": "pick_up", "item_id": milk.id}])

        assert len(state.items) == initial_count, "Respawn keeps item count stable"


# =====================================================================
# Game over
# =====================================================================

class TestGameOver:
    def test_game_ends_at_max_rounds(self):
        """Game ends after max_rounds steps."""
        sim = _make_sim(max_rounds=3, spawns=[(5, 4)])
        sim.reset()

        _, done = sim.step([{"bot": 0, "action": "wait"}])
        assert not done
        _, done = sim.step([{"bot": 0, "action": "wait"}])
        assert not done
        _, done = sim.step([{"bot": 0, "action": "wait"}])
        assert done, "Game should end after 3 rounds"


# =====================================================================
# to_dict compatibility with live server format
# =====================================================================

class TestToDictFormat:
    def test_format_matches_server(self):
        """SimState.to_dict() should produce same structure as live server."""
        sim = _make_sim(spawns=[(5, 4)])
        state = sim.reset()
        d = state.to_dict()

        assert d["type"] == "game_state"
        assert "round" in d
        assert "max_rounds" in d
        assert "grid" in d
        assert "width" in d["grid"]
        assert "height" in d["grid"]
        assert "walls" in d["grid"]
        assert "bots" in d
        assert "items" in d
        assert "orders" in d
        assert "drop_off" in d
        assert "score" in d

        # Walls should NOT include shelves (matching server behavior)
        wall_set = {tuple(w) for w in d["grid"]["walls"]}
        for item in d["items"]:
            item_pos = tuple(item["position"])
            assert item_pos not in wall_set, \
                f"Shelf {item_pos} should not be in grid.walls (server doesn't include them)"

    def test_hidden_orders_not_visible(self):
        """Only active and preview orders appear in to_dict()."""
        sim = _make_sim(orders=[
            {"id": "o0", "items_required": ["milk"]},
            {"id": "o1", "items_required": ["bread"]},
            {"id": "o2", "items_required": ["milk"]},
            {"id": "o3", "items_required": ["bread"]},
        ])
        state = sim.reset()
        d = state.to_dict()

        order_ids = {o["id"] for o in d["orders"]}
        assert "o0" in order_ids, "Active order visible"
        assert "o1" in order_ids, "Preview order visible"
        assert "o2" not in order_ids, "Hidden order not visible"
        assert "o3" not in order_ids, "Hidden order not visible"


# =====================================================================
# from_recon_data round-trip
# =====================================================================

class TestFromReconData:
    def test_round_trip(self):
        """Recon data → Simulator → run → valid game."""
        recon = {
            "fingerprint": "test1234",
            "grid_size": [8, 6],
            "walls": [],
            "drop_off": [1, 4],
            "shelf_map": {"milk": [[3, 1]], "bread": [[3, 3]]},
            "bot_count": 1,
            "bot_start_positions": [[5, 4]],
            "order_sequence": [
                {"id": "o0", "items_required": ["milk", "bread"]},
                {"id": "o1", "items_required": ["milk"]},
            ],
            "total_rounds": 50,
            "final_score": 0,
        }

        sim = Simulator.from_recon_data(recon)
        assert sim.width == 8
        assert sim.height == 6
        assert len(sim.shelves) == 2
        assert sim.drop_off == (1, 4)
        assert len(sim.order_sequence) == 2

        # Should be able to run a game
        def wait_all(state_dict):
            bots = state_dict["bots"]
            return {"actions": [{"bot": b["id"], "action": "wait"} for b in bots]}

        result = sim.run(wait_all)
        assert result["score"] == 0
        assert result["rounds_used"] == 50

    def test_multi_bot_spawns(self):
        """Multi-bot recon data creates correct number of bots."""
        recon = {
            "fingerprint": "med12345",
            "grid_size": [16, 12],
            "walls": [],
            "drop_off": [1, 10],
            "shelf_map": {"milk": [[3, 1]]},
            "bot_count": 3,
            "bot_start_positions": [[14, 10], [14, 8], [14, 6]],
            "order_sequence": [
                {"id": "o0", "items_required": ["milk"]},
            ],
            "total_rounds": 50,
            "final_score": 0,
        }

        sim = Simulator.from_recon_data(recon)
        state = sim.reset()
        assert len(state.bots) == 3
        assert state.bots[0].position == (14, 10)
        assert state.bots[1].position == (14, 8)
        assert state.bots[2].position == (14, 6)
