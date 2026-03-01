"""
Analyze a game log from recon run.
Answers: order sequence, item positions, throughput, theoretical max.

Usage:
    python -m offline.analyzer logs/game_log_2026-03-01_score95.json
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from .grid import Grid, Pos


def load_log(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def analyze(log: dict, verbose: bool = True) -> dict:
    """Full analysis of a game log. Returns structured results."""
    rounds = log["rounds"]
    orders = log.get("orders_sequence", [])
    r0 = rounds[0]

    # --- Grid info ---
    grid_w = r0["grid"]["width"] if "grid" in r0 else "?"
    grid_h = r0["grid"]["height"] if "grid" in r0 else "?"
    drop_off = tuple(r0["drop_off"]) if "drop_off" in r0 else None
    num_bots = len(r0["bots"])

    # --- Item analysis ---
    all_item_ids: set[str] = set()
    new_items_by_round: dict[int, list[str]] = {}
    item_positions: dict[str, Pos] = {}  # item_id -> shelf position
    item_types: dict[str, str] = {}  # item_id -> type
    type_positions: dict[str, set[Pos]] = {}  # type -> set of shelf positions

    for r in rounds:
        rnd = r["round"]
        for item in r["items"]:
            iid = item["id"]
            itype = item["type"]
            ipos = tuple(item["position"])
            item_positions[iid] = ipos
            item_types[iid] = itype
            if itype not in type_positions:
                type_positions[itype] = set()
            type_positions[itype].add(ipos)
            if iid not in all_item_ids:
                all_item_ids.add(iid)
                if rnd not in new_items_by_round:
                    new_items_by_round[rnd] = []
                new_items_by_round[rnd].append(iid)

    items_round_1 = len(r0["items"])
    items_total = len(all_item_ids)
    items_spawn_dynamically = len(new_items_by_round) > 1

    # --- Determinism check data ---
    # Fingerprint: grid + item types + positions (for diffing between runs)
    shelf_positions = sorted(type_positions.keys())

    # --- Order analysis ---
    order_sizes = [len(o["items_required"]) for o in orders]
    order_types_needed = [Counter(o["items_required"]) for o in orders]

    # --- Throughput analysis ---
    score_at_round = {}
    for r in rounds:
        if r["round"] % 25 == 0 or r["round"] == len(rounds) - 1:
            score_at_round[r["round"]] = r["score"]

    final_score = rounds[-1]["score"] if rounds else 0
    final_round = rounds[-1]["round"] if rounds else 0

    # --- Delivery analysis (track score jumps) ---
    deliveries = []
    prev_score = 0
    for r in rounds:
        s = r["score"]
        if s > prev_score:
            delta = s - prev_score
            deliveries.append({
                "round": r["round"],
                "score_delta": delta,
                "items_delta": delta if delta < 5 else delta % 5 or 5,
            })
        prev_score = s

    # --- Theoretical max estimate ---
    # For each order: min rounds = pickup + travel + deliver
    # Rough: 3 items * 5 avg_dist + 6 deliver = 21 rounds/order
    avg_order_size = sum(order_sizes) / len(order_sizes) if order_sizes else 3.5
    theoretical_orders = int(300 / (avg_order_size * 4 + 8))  # rough estimate
    theoretical_items = int(theoretical_orders * avg_order_size)
    theoretical_score = theoretical_items + theoretical_orders * 5

    results = {
        "grid": f"{grid_w}x{grid_h}",
        "num_bots": num_bots,
        "drop_off": drop_off,
        "items_round_1": items_round_1,
        "items_total_seen": items_total,
        "items_spawn_dynamically": items_spawn_dynamically,
        "new_items_by_round": {k: len(v) for k, v in new_items_by_round.items()},
        "unique_item_types": len(type_positions),
        "type_shelf_positions": {t: [list(p) for p in sorted(ps)]
                                  for t, ps in sorted(type_positions.items())},
        "num_orders_seen": len(orders),
        "order_sizes": order_sizes,
        "avg_order_size": round(avg_order_size, 1),
        "order_sequence": [
            {"id": o["id"], "items": o["items_required"],
             "first_seen": o["first_seen_round"]}
            for o in orders
        ],
        "score_progression": score_at_round,
        "final_score": final_score,
        "final_round": final_round,
        "deliveries": deliveries,
        "theoretical_max_orders": theoretical_orders,
        "theoretical_max_score": theoretical_score,
    }

    if verbose:
        _print_analysis(results)

    return results


def _print_analysis(r: dict):
    print("=" * 60)
    print(f"GROCERY BOT GAME ANALYSIS")
    print("=" * 60)

    print(f"\n--- Grid & Setup ---")
    print(f"  Grid: {r['grid']}, Bots: {r['num_bots']}, Drop-off: {r['drop_off']}")
    print(f"  Item types: {r['unique_item_types']}")

    print(f"\n--- Item Spawning ---")
    print(f"  Items visible round 1: {r['items_round_1']}")
    print(f"  Total unique items seen: {r['items_total_seen']}")
    if r["items_spawn_dynamically"]:
        print(f"  ⚠️  Items spawn dynamically! New items appear in rounds:")
        for rnd, count in sorted(r["new_items_by_round"].items()):
            if int(rnd) > 1:
                print(f"       Round {rnd}: +{count} new items")
    else:
        print(f"  ✅ All items present from round 1 — full pre-compute possible!")

    print(f"\n--- Item Shelf Positions (per type) ---")
    for itype, positions in r["type_shelf_positions"].items():
        print(f"  {itype}: {positions}")

    print(f"\n--- Order Sequence ({r['num_orders_seen']} orders seen) ---")
    for i, o in enumerate(r["order_sequence"]):
        print(f"  {i:2d}. {o['id']}: {o['items']} (visible from round {o['first_seen']})")

    print(f"\n--- Order Stats ---")
    print(f"  Sizes: {r['order_sizes']}")
    print(f"  Average: {r['avg_order_size']} items/order")

    print(f"\n--- Score Progression ---")
    for rnd, score in sorted(r["score_progression"].items()):
        bar = "█" * (score // 3)
        print(f"  Round {rnd:3d}: {score:3d} {bar}")
    print(f"  Final: {r['final_score']} (round {r['final_round']})")

    print(f"\n--- Theoretical Estimates ---")
    print(f"  Max orders in 300 rounds: ~{r['theoretical_max_orders']}")
    print(f"  Theoretical max score: ~{r['theoretical_max_score']}")
    print()


def check_determinism(log1_path: str, log2_path: str):
    """Compare two logs from same day to verify determinism."""
    l1 = load_log(log1_path)
    l2 = load_log(log2_path)

    a1 = analyze(l1, verbose=False)
    a2 = analyze(l2, verbose=False)

    print("=" * 60)
    print("DETERMINISM CHECK")
    print("=" * 60)

    # Check grid
    same_grid = a1["grid"] == a2["grid"]
    print(f"\n  Grid:        {'✅ MATCH' if same_grid else '❌ DIFFER'}")

    # Check order sequence
    seq1 = [(o["id"], tuple(o["items"])) for o in a1["order_sequence"]]
    seq2 = [(o["id"], tuple(o["items"])) for o in a2["order_sequence"]]
    # Compare just item types (IDs might differ)
    items1 = [o["items"] for o in a1["order_sequence"]]
    items2 = [o["items"] for o in a2["order_sequence"]]
    same_orders = items1 == items2
    print(f"  Orders:      {'✅ MATCH' if same_orders else '❌ DIFFER'}")

    # Check item positions per type
    same_shelves = a1["type_shelf_positions"] == a2["type_shelf_positions"]
    print(f"  Shelf spots: {'✅ MATCH' if same_shelves else '❌ DIFFER'}")

    # Check drop-off
    same_drop = a1["drop_off"] == a2["drop_off"]
    print(f"  Drop-off:    {'✅ MATCH' if same_drop else '❌ DIFFER'}")

    all_match = same_grid and same_orders and same_shelves and same_drop
    print(f"\n  {'✅ DETERMINISTIC — two-pass is valid!' if all_match else '❌ NOT FULLY DETERMINISTIC — check diffs above'}")

    if not same_orders and len(items1) > 0 and len(items2) > 0:
        print(f"\n  First diverging order:")
        for i, (o1, o2) in enumerate(zip(items1, items2)):
            if o1 != o2:
                print(f"    Order {i}: Run1={o1}  Run2={o2}")
                break

    return all_match


if __name__ == "__main__":
    if len(sys.argv) == 2:
        log = load_log(sys.argv[1])
        analyze(log)
    elif len(sys.argv) == 3:
        check_determinism(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  python -m offline.analyzer <log.json>           # Analyze single log")
        print("  python -m offline.analyzer <log1> <log2>        # Check determinism")
