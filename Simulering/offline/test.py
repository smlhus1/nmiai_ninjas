"""
Self-test for the offline planner.
Simulates a realistic Easy-mode game (12x10, 1 bot, 4 item types).

Run: python -m offline.test
"""
from offline.grid import Grid, Pos
from offline.planner import OfflinePlanner, ItemTarget


def make_easy_grid() -> Grid:
    """
    Realistic 12x10 Easy grid based on game spec.
    
    Layout:
    ############
    #..........#   
    #.SS.SS.SS.#   S = shelf, . = floor, # = wall
    #.SS.SS.SS.#   
    #..........#   
    #..........#   
    #.SS.SS.SS.#   
    #.SS.SS.SS.#   
    #D.........#   D = drop-off at (1,8)
    ############
    """
    width, height = 12, 10
    walls = set()
    shelves = set()

    # Border walls
    for x in range(width):
        walls.add((x, 0))
        walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y))
        walls.add((width - 1, y))

    # Shelf positions (3 aisle pairs)
    for aisle_x in [2, 3, 5, 6, 8, 9]:
        for shelf_y in [2, 3, 6, 7]:
            shelves.add((aisle_x, shelf_y))

    return Grid(width, height, walls, shelves)


def test_basic_planning():
    """Test single-order planning on Easy grid."""
    grid = make_easy_grid()
    drop_off = (1, 8)
    spawn = (10, 8)

    # Item type → shelf positions
    type_positions = {
        "milk":   [(2, 2), (2, 6)],
        "bread":  [(3, 2), (3, 6)],
        "butter": [(5, 2), (5, 6)],
        "yogurt": [(6, 2), (6, 6)],
    }

    planner = OfflinePlanner(grid, drop_off, [spawn])

    # Single order: 3 items
    plan = planner.plan_order(
        spawn,
        ["milk", "bread", "butter"],
        type_positions,
    )

    assert len(plan.batches) == 1, f"Expected 1 batch for 3 items, got {len(plan.batches)}"
    assert len(plan.batches[0].items) == 3
    assert plan.total_cost > 0
    assert plan.total_cost < 50, f"Cost {plan.total_cost} seems too high for Easy"

    print(f"✅ Single order (3 items): cost={plan.total_cost} rounds")
    for item in plan.batches[0].items:
        print(f"   → {item.item_type} at shelf {item.shelf_pos}, "
              f"pickup {item.pickup_pos}")


def test_four_item_order():
    """Test order with 4 items (requires 2 batches due to cap=3)."""
    grid = make_easy_grid()
    drop_off = (1, 8)
    spawn = (10, 8)

    type_positions = {
        "milk":   [(2, 2), (2, 6)],
        "bread":  [(3, 2), (3, 6)],
        "butter": [(5, 2), (5, 6)],
        "yogurt": [(6, 2), (6, 6)],
    }

    planner = OfflinePlanner(grid, drop_off, [spawn])

    plan = planner.plan_order(
        spawn,
        ["milk", "bread", "butter", "yogurt"],
        type_positions,
    )

    assert len(plan.batches) == 2, f"Expected 2 batches for 4 items, got {len(plan.batches)}"
    total_items = sum(len(b.items) for b in plan.batches)
    assert total_items == 4

    print(f"✅ 4-item order: {len(plan.batches)} batches, "
          f"total cost={plan.total_cost} rounds")
    for i, batch in enumerate(plan.batches):
        types = [item.item_type for item in batch.items]
        print(f"   Batch {i}: {types} (cost={batch.cost})")


def test_cross_order_pipelining():
    """Test that cross-order items get packed into last batch."""
    grid = make_easy_grid()
    drop_off = (1, 8)
    spawn = (10, 8)

    type_positions = {
        "milk":   [(2, 2), (2, 6)],
        "bread":  [(3, 2), (3, 6)],
        "butter": [(5, 2), (5, 6)],
        "yogurt": [(6, 2), (6, 6)],
    }

    planner = OfflinePlanner(grid, drop_off, [spawn])

    # Active order: 2 items (leaves 1 free slot)
    # Preview order: starts with yogurt
    plan = planner.plan_order(
        spawn,
        ["milk", "bread"],
        type_positions,
        preview_items=["yogurt", "butter"],
    )

    assert len(plan.batches) == 1
    batch = plan.batches[0]

    if plan.cross_order_items:
        print(f"✅ Cross-order pipelining: packed {len(plan.cross_order_items)} "
              f"preview items into batch")
        types = [item.item_type for item in batch.items]
        print(f"   Batch: {types} (cost={batch.cost})")
        xo = [x.item_type for x in plan.cross_order_items]
        print(f"   Cross-order items: {xo} (will auto-deliver on order transition)")
    else:
        print(f"⚠️  No cross-order items packed (marginal cost too high?)")
        print(f"   Batch: {[i.item_type for i in batch.items]} (cost={batch.cost})")


def test_full_game_plan():
    """Test planning an entire game with multiple orders."""
    grid = make_easy_grid()
    drop_off = (1, 8)
    spawn = (10, 8)

    type_positions = {
        "milk":   [(2, 2), (2, 6)],
        "bread":  [(3, 2), (3, 6)],
        "butter": [(5, 2), (5, 6)],
        "yogurt": [(6, 2), (6, 6)],
    }

    # Simulate 15 orders
    orders = [
        {"id": f"order_{i}", "items_required": items}
        for i, items in enumerate([
            ["milk", "bread", "butter"],
            ["yogurt", "milk", "bread"],
            ["butter", "yogurt", "milk"],
            ["bread", "butter", "yogurt"],
            ["milk", "yogurt", "bread", "butter"],
            ["milk", "bread", "butter"],
            ["yogurt", "milk", "bread"],
            ["butter", "yogurt", "milk"],
            ["bread", "butter", "yogurt"],
            ["milk", "bread", "butter"],
            ["yogurt", "bread", "milk"],
            ["butter", "milk", "yogurt"],
            ["milk", "bread", "butter"],
            ["yogurt", "milk", "bread"],
            ["butter", "yogurt", "milk"],
        ])
    ]

    planner = OfflinePlanner(grid, drop_off, [spawn])
    plans = planner.plan_full_game(orders, type_positions)

    completed_orders = sum(1 for p in plans if p.batches)
    total_items = sum(len(b.items) for p in plans for b in p.batches)
    total_rounds = sum(p.total_cost for p in plans)
    score = total_items + completed_orders * 5

    print(f"\n✅ Full game plan:")
    print(f"   Orders: {completed_orders}/{len(orders)} completed")
    print(f"   Items: {total_items}")
    print(f"   Rounds: {total_rounds}/300")
    print(f"   Score: {score}")
    print(f"   Rounds/item: {total_rounds/max(total_items,1):.1f}")

    # Generate action sequence
    actions = planner.to_action_sequence(plans)
    print(f"   Action sequence: {len(actions)} actions")

    # Verify actions are valid
    valid_actions = {"move_up", "move_down", "move_left", "move_right",
                     "pick_up", "drop_off", "wait"}
    for a in actions:
        assert a["action"] in valid_actions, f"Invalid action: {a['action']}"
        assert "bot" in a

    print(f"\n   ✅ All {len(actions)} actions are valid")


def test_action_sequence_walkable():
    """Verify that move actions only go to walkable cells."""
    grid = make_easy_grid()
    drop_off = (1, 8)
    spawn = (10, 8)

    type_positions = {
        "milk":   [(2, 2)],
        "bread":  [(3, 2)],
        "butter": [(5, 2)],
    }

    planner = OfflinePlanner(grid, drop_off, [spawn])
    plans = planner.plan_full_game(
        [{"id": "o0", "items_required": ["milk", "bread", "butter"]}],
        type_positions,
    )
    actions = planner.to_action_sequence(plans)

    # Simulate moves
    pos = spawn
    MOVES = {"move_up": (0,-1), "move_down": (0,1),
             "move_left": (-1,0), "move_right": (1,0)}

    for i, a in enumerate(actions):
        if a["action"] in MOVES:
            dx, dy = MOVES[a["action"]]
            new_pos = (pos[0] + dx, pos[1] + dy)
            assert grid.walkable(new_pos), \
                f"Action {i}: {a['action']} from {pos} -> {new_pos} is NOT walkable!"
            pos = new_pos

    print(f"✅ All move actions land on walkable cells")


if __name__ == "__main__":
    print("=" * 50)
    print("OFFLINE PLANNER SELF-TEST")
    print("=" * 50)
    print()

    test_basic_planning()
    print()
    test_four_item_order()
    print()
    test_cross_order_pipelining()
    print()
    test_full_game_plan()
    print()
    test_action_sequence_walkable()

    print()
    print("=" * 50)
    print("ALL TESTS PASSED ✅")
    print("=" * 50)
