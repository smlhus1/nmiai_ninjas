"""
Theoretical maximum score calculator for NM i AI Grocery Bot.

Given perfect information (all orders, shelf positions, grid layout),
computes the best possible score assuming NO collisions between bots.
Respects: BFS distances with one-way corridors, inventory cap (3),
sequential order activation, pickup/dropoff action costs.

Usage:
    py theoretical_max.py logs/74001e7f_2026-03-12_recon.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from bot.models import Grid, Pos
from bot.engine.pathfinding import PathEngine


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ShelfOption:
    """One shelf that has a specific item type."""
    item_type: str
    shelf_pos: Pos
    pickup_pos: Pos  # best walkable cell adjacent to shelf


@dataclass
class IdealBot:
    id: int
    position: Pos
    inventory: list[str] = field(default_factory=list)
    free_at: int = 0  # round when bot becomes available


@dataclass
class BotTrip:
    """A planned trip: bot picks items then delivers."""
    bot_id: int
    items: list[str]
    pickup_positions: list[Pos]   # one per item
    drop_off: Pos
    start_round: int
    pickup_done_round: int
    delivery_round: int  # round when drop_off happens


@dataclass
class OrderResult:
    order_id: str
    items_required: list[str]
    items_delivered: int = 0
    completed: bool = False
    activated_round: int = 0
    completed_round: int | None = None
    trips: list[BotTrip] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Grid & distance setup
# ---------------------------------------------------------------------------

def load_recon(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_grid_and_engine(recon: dict) -> tuple[Grid, PathEngine]:
    """Build grid with shelves as walls, and PathEngine with one-way corridors."""
    width, height = recon["grid_size"]
    walls: set[Pos] = {tuple(w) for w in recon["walls"]}

    # Add shelves as walls (they're non-walkable)
    for positions in recon["shelf_map"].values():
        for p in positions:
            walls.add(tuple(p))

    grid = Grid(width=width, height=height, walls=frozenset(walls))

    drop_off: Pos = tuple(recon["drop_off"])
    engine = PathEngine()
    engine.enable_one_way(True)
    engine.set_grid(grid, drop_off=drop_off)

    return grid, engine


def build_shelf_lookup(
    recon: dict, grid: Grid, engine: PathEngine
) -> dict[str, list[ShelfOption]]:
    """For each item type, list all shelf options with best pickup positions."""
    lookup: dict[str, list[ShelfOption]] = defaultdict(list)

    for item_type, positions in recon["shelf_map"].items():
        for raw_pos in positions:
            shelf_pos = tuple(raw_pos)
            # Find walkable adjacent cells
            pickup_positions = []
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                pp = (shelf_pos[0] + dx, shelf_pos[1] + dy)
                if grid.is_walkable(pp):
                    pickup_positions.append(pp)

            if not pickup_positions:
                continue

            # Store all pickup positions — we'll pick best per bot later
            for pp in pickup_positions:
                lookup[item_type].append(ShelfOption(
                    item_type=item_type,
                    shelf_pos=shelf_pos,
                    pickup_pos=pp,
                ))

    return dict(lookup)


# ---------------------------------------------------------------------------
# Core: trip time calculation
# ---------------------------------------------------------------------------

def trip_time(
    engine: PathEngine,
    bot_pos: Pos,
    pickup_positions: list[Pos],
    drop_off: Pos,
) -> int:
    """
    Calculate total rounds for a bot trip:
    travel to each pickup (sequential) + 1 pick action each + travel to drop-off + 1 drop action.
    """
    time = 0
    pos = bot_pos
    for pp in pickup_positions:
        dist = engine.distance(pos, pp)
        if dist >= 9999:
            return 9999
        time += dist + 1  # travel + pick_up action
        pos = pp
    # Travel to drop-off + drop action
    dist = engine.distance(pos, drop_off)
    if dist >= 9999:
        return 9999
    time += dist + 1  # travel + drop_off action
    return time


def best_drop_off(engine: PathEngine, pos: Pos, zones: list[Pos]) -> Pos:
    """Nearest drop-off zone from a position."""
    return min(zones, key=lambda z: engine.distance(pos, z))


# ---------------------------------------------------------------------------
# Greedy assignment: assign items to bots
# ---------------------------------------------------------------------------

def assign_order_items(
    engine: PathEngine,
    shelf_lookup: dict[str, list[ShelfOption]],
    drop_off_zones: list[Pos],
    bots: list[IdealBot],
    items_needed: list[str],
    current_round: int,
) -> list[BotTrip]:
    """
    Assign items to bots greedily to minimize makespan.

    Strategy: assign one item at a time to the bot that can pick it up
    fastest. When a bot has 3 items, schedule its delivery trip and
    make it unavailable until delivery completes.
    """
    # Track which bots are building trips
    bot_items: dict[int, list[tuple[str, Pos]]] = defaultdict(list)  # bot_id -> [(type, pickup_pos)]
    bot_pos: dict[int, Pos] = {b.id: b.position for b in bots}
    bot_available: dict[int, int] = {b.id: max(b.free_at, current_round) for b in bots}

    trips: list[BotTrip] = []
    remaining = list(items_needed)

    while remaining:
        item_type = remaining[0]
        options = shelf_lookup.get(item_type, [])
        if not options:
            remaining.pop(0)
            continue

        # Find best (bot, shelf_option) pair
        best_bot_id = -1
        best_option = None
        best_arrival = 9999

        for bot in bots:
            avail = bot_available[bot.id]
            if len(bot_items[bot.id]) >= 3:
                continue  # bot batch full, needs to deliver first

            cur_pos = bot_pos[bot.id]
            for opt in options:
                dist = engine.distance(cur_pos, opt.pickup_pos)
                arrival = avail + dist + 1  # arrive + pick
                if arrival < best_arrival:
                    best_arrival = arrival
                    best_bot_id = bot.id
                    best_option = opt

        if best_bot_id < 0 or best_option is None:
            # All bots full — flush the one that finishes earliest
            earliest_bot = _flush_earliest(
                engine, bots, bot_items, bot_pos, bot_available,
                drop_off_zones, trips, current_round
            )
            if earliest_bot is None:
                break  # stuck
            continue  # retry this item

        # Assign item to bot
        remaining.pop(0)
        bot_items[best_bot_id].append((item_type, best_option.pickup_pos))
        bot_pos[best_bot_id] = best_option.pickup_pos
        bot_available[best_bot_id] = best_arrival

    # Flush remaining partial batches
    for bot in bots:
        if bot_items[bot.id]:
            _create_trip(
                engine, bot, bot_items[bot.id], bot_pos[bot.id],
                bot_available[bot.id], drop_off_zones, trips
            )

    return trips


def _flush_earliest(
    engine: PathEngine,
    bots: list[IdealBot],
    bot_items: dict[int, list],
    bot_pos: dict[int, Pos],
    bot_available: dict[int, int],
    drop_off_zones: list[Pos],
    trips: list[BotTrip],
    current_round: int,
) -> IdealBot | None:
    """Flush the bot with most items (or earliest available) to free a slot."""
    candidates = [(b, bot_items[b.id]) for b in bots if bot_items[b.id]]
    if not candidates:
        return None

    # Prefer bots with 3 items, then most items, then earliest available
    candidates.sort(key=lambda x: (-len(x[1]), bot_available[x[0].id]))
    bot, items = candidates[0]

    trip = _create_trip(
        engine, bot, items, bot_pos[bot.id],
        bot_available[bot.id], drop_off_zones, trips
    )
    # Update bot state after delivery
    bot_pos[bot.id] = trip.drop_off
    bot_available[bot.id] = trip.delivery_round
    bot_items[bot.id] = []
    return bot


def _create_trip(
    engine: PathEngine,
    bot: IdealBot,
    items: list[tuple[str, Pos]],
    last_pos: Pos,
    available_at: int,
    drop_off_zones: list[Pos],
    trips: list[BotTrip],
) -> BotTrip:
    """Create a BotTrip from accumulated items."""
    zone = best_drop_off(engine, last_pos, drop_off_zones)
    dist_to_zone = engine.distance(last_pos, zone)

    trip = BotTrip(
        bot_id=bot.id,
        items=[it[0] for it in items],
        pickup_positions=[it[1] for it in items],
        drop_off=zone,
        start_round=bot.free_at,
        pickup_done_round=available_at,
        delivery_round=available_at + dist_to_zone + 1,
    )
    trips.append(trip)
    return trip


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def simulate(recon: dict) -> tuple[int, list[OrderResult]]:
    """
    Run ideal simulation with pipelining.

    Key insight: while a few bots deliver the active order, the remaining
    17-18 bots can pre-pick items for the NEXT order. When the active order
    completes, pre-picked items are instantly ready for delivery.

    Approach: event-driven simulation. Each bot has scheduled tasks (pickup
    sequences + delivery). We advance time to the next event.
    """
    grid, engine = build_grid_and_engine(recon)
    shelf_lookup = build_shelf_lookup(recon, grid, engine)

    raw_zones = recon.get("drop_off_zones")
    drop_off_zones: list[Pos] = (
        [tuple(z) for z in raw_zones] if raw_zones
        else [tuple(recon["drop_off"])]
    )

    spawns = [tuple(p) for p in recon.get("bot_start_positions", [])]
    n_bots = recon.get("bot_count", len(spawns))
    max_rounds = recon.get("total_rounds", 500)

    orders = recon["order_sequence"]

    # Initialize bots
    bots = []
    for i in range(n_bots):
        pos = spawns[i % len(spawns)] if spawns else (28, 16)
        bots.append(IdealBot(id=i, position=pos))

    score = 0
    results: list[OrderResult] = []

    # --- Phase 1: Schedule all pickup trips with pipelining ---
    # For each order, assign bots to pick items ASAP (even before order activates).
    # Bots can start picking order N+1 items while order N is being delivered.
    # Delivery can only happen when the order is active.

    # Pre-plan: for each order, schedule pickup trips
    # Track bot availability
    all_order_trips: list[list[BotTrip]] = []

    for order_idx, order in enumerate(orders):
        items_needed = list(order["items_required"])

        # Bots start picking as soon as they're free
        # (no need to wait for previous order to complete for PICKUP)
        trips = assign_order_items(
            engine, shelf_lookup, drop_off_zones,
            bots, items_needed, current_round=0,  # bots use their own free_at
        )

        # Update bot states based on trips
        for trip in trips:
            bot = bots[trip.bot_id]
            bot.position = trip.drop_off
            bot.free_at = trip.delivery_round

        all_order_trips.append(trips)

    # --- Phase 2: Simulate delivery with order activation constraints ---
    # Orders activate sequentially: order N+1 activates when order N completes.
    # Items can only be DELIVERED when the order is active.
    # But pickup can happen before activation (pipelining).

    # Reset bot states for delivery simulation
    for bot in bots:
        bot.free_at = 0

    active_round = 0  # round when current active order started

    for order_idx, order in enumerate(orders):
        items_needed = list(order["items_required"])
        trips = all_order_trips[order_idx]

        order_result = OrderResult(
            order_id=order["id"],
            items_required=list(items_needed),
            activated_round=active_round,
        )

        # Delivery can only happen at max(trip.delivery_round, active_round)
        # because items can't be delivered until the order is active
        items_remaining = list(items_needed)
        effective_deliveries = []

        for trip in trips:
            # Delivery happens at the later of: trip completion or order activation
            effective_round = max(trip.delivery_round, active_round)
            effective_deliveries.append((effective_round, trip))

        effective_deliveries.sort(key=lambda x: x[0])

        last_delivery = active_round
        for eff_round, trip in effective_deliveries:
            if eff_round > max_rounds:
                break

            delivered_count = 0
            for item_type in trip.items:
                if item_type in items_remaining:
                    items_remaining.remove(item_type)
                    delivered_count += 1
                    score += 1

            order_result.items_delivered += delivered_count
            last_delivery = max(last_delivery, eff_round)

        if not items_remaining:
            order_result.completed = True
            order_result.completed_round = last_delivery
            if last_delivery <= max_rounds:
                score += 5
            active_round = last_delivery  # next order activates immediately
        else:
            order_result.completed = False
            order_result.completed_round = last_delivery
            # If order not completed within max_rounds, remaining orders also fail
            results.append(order_result)
            # Continue to see if partial credit possible for remaining orders
            active_round = last_delivery
            continue

        results.append(order_result)

    return score, results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(recon: dict, score: int, results: list[OrderResult]) -> None:
    max_rounds = recon.get("total_rounds", 500)
    n_bots = recon.get("bot_count", 20)
    orders = recon["order_sequence"]
    total_items = sum(len(o["items_required"]) for o in orders)
    fp = recon.get("fingerprint", "?")

    print(f"\n{'='*60}")
    print(f"  THEORETICAL MAXIMUM SCORE CALCULATOR")
    print(f"  Map: {fp} ({recon['grid_size'][0]}x{recon['grid_size'][1]}, "
          f"{n_bots} bots, {len(recon.get('drop_off_zones', [recon['drop_off']]))} zones)")
    print(f"  Orders: {len(orders)} ({total_items} items), {max_rounds} rounds")
    print(f"{'='*60}\n")

    items_delivered = 0
    orders_completed = 0

    print(f"{'Nr':>3}  {'Ordre':<10} {'Items':>5} {'Akt.R':>6} {'Ferdig R':>8} "
          f"{'Levert':>6} {'Bonus':>5} {'Score':>5}  {'Status'}")
    print("-" * 75)

    for i, r in enumerate(results):
        n_items = len(r.items_required)
        bonus = 5 if r.completed and (r.completed_round or 0) <= max_rounds else 0
        order_score = r.items_delivered + bonus
        items_delivered += r.items_delivered
        if r.completed and (r.completed_round or 0) <= max_rounds:
            orders_completed += 1
        status = "OK" if r.completed and (r.completed_round or 0) <= max_rounds else "TIMEOUT"

        print(f"{i+1:>3}  {r.order_id:<10} {n_items:>5} {r.activated_round:>6} "
              f"{r.completed_round or '-':>8} {r.items_delivered:>6} {bonus:>5} "
              f"{order_score:>5}  {status}")

    print("-" * 75)
    item_score = items_delivered
    bonus_score = orders_completed * 5

    print(f"\n  Items delivered:   {items_delivered}/{total_items}")
    print(f"  Orders completed:  {orders_completed}/{len(orders)}")
    print(f"  Item score:        {item_score}")
    print(f"  Order bonus:       {bonus_score} ({orders_completed} x 5)")
    print(f"  " + "-" * 29)
    print(f"  THEORETICAL MAX:   {score}")
    print()

    # Bot utilization
    if results:
        last_round = max(
            (r.completed_round or 0) for r in results
        )
        print(f"  Last delivery at round: {last_round} / {max_rounds}")
        if last_round < max_rounds:
            print(f"  Unused rounds: {max_rounds - last_round}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def export_trips(recon: dict) -> dict:
    """
    Export all trips with order activation rounds for MAPF planning.

    Returns dict with:
        - trips: list of BotTrip (all orders)
        - order_activations: dict[order_idx -> activation_round]
        - order_items: dict[order_idx -> list[str]]
        - grid_info: grid/engine setup data
        - score: theoretical max score
    """
    grid, engine = build_grid_and_engine(recon)
    shelf_lookup = build_shelf_lookup(recon, grid, engine)

    raw_zones = recon.get("drop_off_zones")
    drop_off_zones: list[Pos] = (
        [tuple(z) for z in raw_zones] if raw_zones
        else [tuple(recon["drop_off"])]
    )

    spawns = [tuple(p) for p in recon.get("bot_start_positions", [])]
    n_bots = recon.get("bot_count", len(spawns))
    max_rounds = recon.get("total_rounds", 500)
    orders = recon["order_sequence"]

    # Initialize bots
    bots = []
    for i in range(n_bots):
        pos = spawns[i % len(spawns)] if spawns else (28, 16)
        bots.append(IdealBot(id=i, position=pos))

    # Phase 1: assign all trips (same as simulate)
    all_trips: list[BotTrip] = []
    order_trips_map: dict[int, list[BotTrip]] = {}

    for order_idx, order in enumerate(orders):
        items_needed = list(order["items_required"])
        trips = assign_order_items(
            engine, shelf_lookup, drop_off_zones,
            bots, items_needed, current_round=0,
        )
        for trip in trips:
            bot = bots[trip.bot_id]
            bot.position = trip.drop_off
            bot.free_at = trip.delivery_round
        all_trips.extend(trips)
        order_trips_map[order_idx] = trips

    # Phase 2: compute order activation rounds (same logic as simulate)
    for bot in bots:
        bot.free_at = 0

    order_activations: dict[int, int] = {}
    order_items: dict[int, list[str]] = {}
    active_round = 0

    for order_idx, order in enumerate(orders):
        items_needed = list(order["items_required"])
        order_activations[order_idx] = active_round
        order_items[order_idx] = items_needed
        trips = order_trips_map[order_idx]

        items_remaining = list(items_needed)
        effective_deliveries = []
        for trip in trips:
            effective_round = max(trip.delivery_round, active_round)
            effective_deliveries.append((effective_round, trip))
        effective_deliveries.sort(key=lambda x: x[0])

        last_delivery = active_round
        for eff_round, trip in effective_deliveries:
            if eff_round > max_rounds:
                break
            for item_type in trip.items:
                if item_type in items_remaining:
                    items_remaining.remove(item_type)
            last_delivery = max(last_delivery, eff_round)

        if not items_remaining:
            active_round = last_delivery
        else:
            active_round = last_delivery

    # Tag each trip with its order index
    trip_order_map: list[tuple[int, BotTrip]] = []
    for order_idx, trips in order_trips_map.items():
        for trip in trips:
            trip_order_map.append((order_idx, trip))

    return {
        "trips": trip_order_map,  # list of (order_idx, BotTrip)
        "order_activations": order_activations,
        "order_items": order_items,
        "drop_off_zones": drop_off_zones,
        "n_bots": n_bots,
        "max_rounds": max_rounds,
        "spawns": spawns,
        "grid": grid,
        "engine": engine,
        "shelf_lookup": shelf_lookup,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: py theoretical_max.py <recon_file>")
        print("Example: py theoretical_max.py logs/74001e7f_2026-03-12_recon.json")
        sys.exit(1)

    recon = load_recon(sys.argv[1])
    score, results = simulate(recon)
    print_results(recon, score, results)


if __name__ == "__main__":
    main()
