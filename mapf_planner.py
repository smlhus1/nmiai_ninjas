"""
Offline MAPF (Multi-Agent Path Finding) planner for NM i AI Grocery Bot.

Given perfect information from recon, computes collision-free paths for all
20 bots by simulating sequential execution round by round.

Key design:
- Simulates the game server's sequential processing (bot 0 first, bot 19 last)
- Each round: for each bot in ID order, try to take one step toward goal
- Collision check is EXACT: already-processed bots at NEW pos, unprocessed at OLD pos
- Zero conflicts by construction — the plan IS the execution
- Uses BFS distance to pick the best next step toward each waypoint

Usage:
    py mapf_planner.py logs/74001e7f_2026-03-13_recon.json
    py mapf_planner.py logs/74001e7f_2026-03-13_recon.json --save mapf_plan.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from bot.models import Pos, Grid
from bot.engine.pathfinding import PathEngine
from theoretical_max import load_recon, export_trips, BotTrip

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MAPFAction:
    action: str
    position: Pos
    item_type: str = ""


@dataclass
class MAPFPlan:
    actions: dict[int, list[MAPFAction]]
    total_rounds: int
    expected_score: int
    order_activations: dict[int, int]
    pickup_schedule: list[dict]
    dropoff_schedule: list[dict]


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

DELTAS = {
    "move_right": (1, 0), "move_left": (-1, 0),
    "move_down": (0, 1), "move_up": (0, -1),
}
DELTA_TO_ACTION = {v: k for k, v in DELTAS.items()}


def direction_action(from_pos: Pos, to_pos: Pos) -> str:
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    return DELTA_TO_ACTION.get((dx, dy), "wait")


# ---------------------------------------------------------------------------
# Waypoint queue for each bot
# ---------------------------------------------------------------------------

@dataclass
class BotWaypoint:
    """A single waypoint a bot needs to reach, with an action to perform there."""
    position: Pos
    action: str        # "pick_up", "drop_off", "move" (just pass through)
    item_type: str = ""
    order_idx: int = -1
    min_round: int = 0  # earliest round this action can be performed (for delivery timing)


# ---------------------------------------------------------------------------
# Sequential MAPF planner
# ---------------------------------------------------------------------------

class SequentialMAPFPlanner:
    """Plans by simulating sequential execution round by round.

    For each round, processes bots in ID order (matching the game server).
    Each bot tries to take one step toward its current waypoint target.
    Collision check uses the exact server model:
    - Already-processed bots (lower ID): check their NEW position
    - Not-yet-processed bots (higher ID): check their OLD position
    """

    def __init__(self, recon: dict) -> None:
        self._export = export_trips(recon)
        self._grid: Grid = self._export["grid"]
        self._engine: PathEngine = self._export["engine"]
        self._n_bots: int = self._export["n_bots"]
        self._max_rounds: int = self._export["max_rounds"]
        self._spawns: list[Pos] = self._export["spawns"]
        self._drop_off_zones: list[Pos] = self._export["drop_off_zones"]
        self._spawn_pos: Pos = self._spawns[0] if self._spawns else (28, 16)

    def plan(self, method: str = "tsa") -> MAPFPlan:
        t_start = time.perf_counter()
        order_activations = self._export["order_activations"]
        trips = self._export["trips"]

        if method == "tsa":
            # Round-by-round BFS simulation (collision-free by construction)
            bot_actions, pickup_schedule, dropoff_schedule = self._plan_with_tsa(
                {}  # rebuilds queues internally from trips
            )
        elif method == "pibt":
            # PIBT + theoretical max waypoints + throttling
            waypoint_queues = self._build_waypoint_queues(trips, order_activations)
            bot_actions, pickup_schedule, dropoff_schedule = self._simulate(
                waypoint_queues, order_activations,
            )
        elif method == "hybrid":
            # Hybrid: theoretical max trips + reactive PIBT execution
            bot_actions, pickup_schedule, dropoff_schedule = self._simulate_hybrid(trips)
        else:
            # Reactive strategy (no theoretical max)
            bot_actions, pickup_schedule, dropoff_schedule = self._simulate_reactive()

        total_rounds = max(len(acts) for acts in bot_actions.values()) if bot_actions else 0

        # Compute expected score: items delivered + 5 per completed order
        total_delivered = sum(d.get("delivered", 1) for d in dropoff_schedule)
        completed_orders = len(set(d["order_idx"] for d in dropoff_schedule))
        expected_score = total_delivered + completed_orders * 5

        plan = MAPFPlan(
            actions=bot_actions,
            total_rounds=total_rounds,
            expected_score=expected_score,
            order_activations=order_activations,
            pickup_schedule=pickup_schedule,
            dropoff_schedule=dropoff_schedule,
        )

        elapsed = time.perf_counter() - t_start
        print(f"\nMAPF ({method}): {len(pickup_schedule)} pickups, "
              f"{len(dropoff_schedule)} dropoffs, {total_rounds} rounds, "
              f"score {expected_score}, {elapsed:.1f}s")
        return plan

    def _simulate_hybrid(
        self,
        trips: list[tuple[int, 'BotTrip']],
    ) -> tuple[dict[int, list[MAPFAction]], list[dict], list[dict]]:
        """Hybrid: use theoretical_max trip assignments + reactive PIBT execution.

        Takes pre-planned trips (optimal item-to-bot assignment with multi-item routes),
        and executes them with PIBT collision avoidance. Falls back to reactive assignment
        for any items not covered by trips.
        """
        from bot.engine.pibt import PIBTResolver

        orders = self._export["order_items"]
        shelf_lookup = self._export["shelf_lookup"]

        pibt = PIBTResolver(
            grid=self._grid,
            distance_fn=self._engine.distance,
            one_way=getattr(self._engine, '_one_way', None),
        )

        bot_pos: dict[int, Pos] = {i: self._spawn_pos for i in range(self._n_bots)}
        bot_inventory: dict[int, list[str]] = {i: [] for i in range(self._n_bots)}
        bot_actions: dict[int, list[MAPFAction]] = {i: [] for i in range(self._n_bots)}
        pickup_schedule: list[dict] = []
        dropoff_schedule: list[dict] = []

        n_orders = len(orders)
        active_order_idx = 0
        order_items_remaining: dict[int, list[str]] = {
            idx: list(items) for idx, items in orders.items()
        }
        score = 0

        # Build trip queues per bot: list of (order_idx, pickups, items, drop_off)
        bot_trips: dict[int, list[tuple[int, list[Pos], list[str], Pos]]] = {
            i: [] for i in range(self._n_bots)
        }
        for order_idx, trip in trips:
            bot_trips[trip.bot_id].append((
                order_idx,
                list(trip.pickup_positions),
                list(trip.items),
                trip.drop_off,
            ))
        # Sort by order_idx, then start_round
        for bid in range(self._n_bots):
            bot_trips[bid].sort(key=lambda x: x[0])

        # Current trip state per bot
        bot_trip_cursor: dict[int, int] = {i: 0 for i in range(self._n_bots)}
        bot_task: dict[int, str] = {i: "idle" for i in range(self._n_bots)}
        bot_target: dict[int, Pos | None] = {i: None for i in range(self._n_bots)}
        bot_target_type: dict[int, str] = {i: "" for i in range(self._n_bots)}
        bot_target_order: dict[int, int] = {i: -1 for i in range(self._n_bots)}
        bot_pickup_idx: dict[int, int] = {i: 0 for i in range(self._n_bots)}  # within current trip

        # Stuck detection
        bot_pos_history: dict[int, list[Pos]] = {i: [] for i in range(self._n_bots)}
        STUCK_THRESHOLD = 5

        # Grid dimensions and idle positions
        grid_w = self._grid.width
        grid_h = self._grid.height
        _idle_positions: list[Pos] = []
        for y in [grid_h // 4, grid_h // 2, 3 * grid_h // 4]:
            for x in [grid_w // 4, grid_w // 2, 3 * grid_w // 4]:
                if self._grid.is_walkable((x, y)):
                    _idle_positions.append((x, y))
        if not _idle_positions:
            _idle_positions = [self._spawn_pos]

        _dropoff_set = set(self._drop_off_zones)
        _dropoff_adjacent: set[Pos] = set()
        for dz in self._drop_off_zones:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                adj = (dz[0] + dx, dz[1] + dy)
                if 0 <= adj[0] < grid_w and 0 <= adj[1] < grid_h:
                    _dropoff_adjacent.add(adj)

        def _nearest_dropoff(pos: Pos) -> Pos:
            best = self._drop_off_zones[0]
            best_d = self._engine.distance(pos, best)
            for z in self._drop_off_zones[1:]:
                d = self._engine.distance(pos, z)
                if d < best_d:
                    best, best_d = z, d
            return best

        def _assign_from_trips():
            """Assign tasks from pre-planned trips."""
            for bid in range(self._n_bots):
                if bot_task[bid] != "idle":
                    continue
                cursor = bot_trip_cursor[bid]
                if cursor >= len(bot_trips[bid]):
                    continue
                order_idx, pickups, items, drop_off = bot_trips[bid][cursor]

                # Only start trips for active order (pre-pick handled separately)
                if order_idx > active_order_idx:
                    continue

                # Start this trip
                if bot_pickup_idx[bid] < len(pickups):
                    # Still have pickups to do
                    pi = bot_pickup_idx[bid]
                    if len(bot_inventory[bid]) >= 3:
                        # Full inventory — need to deliver first
                        remaining = order_items_remaining.get(active_order_idx, [])
                        has_match = any(item in remaining for item in bot_inventory[bid])
                        if has_match:
                            bot_task[bid] = "deliver"
                            bot_target[bid] = _nearest_dropoff(bot_pos[bid])
                        continue

                    bot_task[bid] = "pick" if order_idx <= active_order_idx else "pre_pick"
                    bot_target[bid] = pickups[pi]
                    bot_target_type[bid] = items[pi] if pi < len(items) else ""
                    bot_target_order[bid] = order_idx
                else:
                    # All pickups done — deliver
                    remaining = order_items_remaining.get(active_order_idx, [])
                    has_match = any(item in remaining for item in bot_inventory[bid])
                    if has_match:
                        bot_task[bid] = "deliver"
                        bot_target[bid] = drop_off
                        bot_target_order[bid] = order_idx
                    else:
                        # Items don't match active order — hold and wait
                        pass

            # Also: idle bots with matching inventory → deliver
            for bid in range(self._n_bots):
                if bot_task[bid] != "idle" or not bot_inventory[bid]:
                    continue
                remaining = order_items_remaining.get(active_order_idx, [])
                if any(item in remaining for item in bot_inventory[bid]):
                    bot_task[bid] = "deliver"
                    bot_target[bid] = _nearest_dropoff(bot_pos[bid])

        for round_t in range(self._max_rounds):
            if active_order_idx >= n_orders:
                break

            if round_t % 50 == 0 or round_t < 5:
                n_idle = sum(1 for bid in range(self._n_bots) if bot_task[bid] == "idle")
                n_pick = sum(1 for bid in range(self._n_bots) if bot_task[bid] == "pick")
                n_deliv = sum(1 for bid in range(self._n_bots) if bot_task[bid] == "deliver")
                n_pre = sum(1 for bid in range(self._n_bots) if bot_task[bid] == "pre_pick")
                remaining_count = len(order_items_remaining.get(active_order_idx, []))
                print(f"  Round {round_t}: order={active_order_idx}, "
                      f"score={score}, pick/deliv/pre/idle={n_pick}/{n_deliv}/{n_pre}/{n_idle}, "
                      f"remaining={remaining_count}")

            # Stuck detection
            for bid in range(self._n_bots):
                bot_pos_history[bid].append(bot_pos[bid])
                if len(bot_pos_history[bid]) > STUCK_THRESHOLD:
                    bot_pos_history[bid] = bot_pos_history[bid][-STUCK_THRESHOLD:]
                if (len(bot_pos_history[bid]) >= STUCK_THRESHOLD
                        and bot_task[bid] in ("pick", "deliver", "pre_pick")
                        and len(set(bot_pos_history[bid])) == 1):
                    bot_task[bid] = "idle"
                    bot_target[bid] = None
                    bot_target_type[bid] = ""
                    bot_pos_history[bid] = []

            _assign_from_trips()

            # Phase 1: Actions at target
            action_bots: set[int] = set()
            for bot_id in range(self._n_bots):
                cur = bot_pos[bot_id]
                task = bot_task[bot_id]
                target = bot_target[bot_id]

                if task in ("pick", "pre_pick") and cur == target and len(bot_inventory[bot_id]) < 3:
                    item_type = bot_target_type[bot_id]
                    action_bots.add(bot_id)
                    bot_actions[bot_id].append(MAPFAction("pick_up", cur, item_type))
                    bot_inventory[bot_id].append(item_type)
                    pickup_schedule.append({
                        "round": round_t, "bot_id": bot_id,
                        "position": cur, "item_type": item_type,
                        "order_idx": bot_target_order[bot_id],
                    })
                    # Advance to next pickup in trip
                    bot_pickup_idx[bot_id] += 1
                    cursor = bot_trip_cursor[bot_id]
                    if cursor < len(bot_trips[bot_id]):
                        order_idx, pickups, items, drop_off = bot_trips[bot_id][cursor]
                        pi = bot_pickup_idx[bot_id]
                        if pi < len(pickups):
                            bot_target[bot_id] = pickups[pi]
                            bot_target_type[bot_id] = items[pi] if pi < len(items) else ""
                        else:
                            # All pickups done — deliver
                            bot_task[bot_id] = "deliver"
                            bot_target[bot_id] = drop_off
                            bot_target_type[bot_id] = ""
                    else:
                        bot_task[bot_id] = "idle"
                        bot_target[bot_id] = None

                elif task == "deliver" and cur == target:
                    remaining = order_items_remaining.get(active_order_idx, [])
                    has_match = any(item in remaining for item in bot_inventory[bot_id])

                    if has_match:
                        action_bots.add(bot_id)
                        bot_actions[bot_id].append(MAPFAction("drop_off", cur))

                        new_inv = []
                        delivered = 0
                        for item in bot_inventory[bot_id]:
                            if item in remaining:
                                remaining.remove(item)
                                delivered += 1
                            else:
                                new_inv.append(item)
                        bot_inventory[bot_id] = new_inv
                        score += delivered

                        dropoff_schedule.append({
                            "round": round_t, "bot_id": bot_id,
                            "position": cur, "order_idx": active_order_idx,
                            "delivered": delivered,
                        })

                        # Order completion chain
                        while not remaining and active_order_idx < n_orders:
                            score += 5
                            print(f"  ORDER {active_order_idx} COMPLETE at round {round_t} "
                                  f"(score={score})")
                            active_order_idx += 1
                            if active_order_idx >= n_orders:
                                break
                            remaining = order_items_remaining.get(active_order_idx, [])
                            auto_inv = []
                            for item in bot_inventory[bot_id]:
                                if item in remaining:
                                    remaining.remove(item)
                                    score += 1
                                else:
                                    auto_inv.append(item)
                            bot_inventory[bot_id] = auto_inv

                        # Advance trip cursor
                        bot_trip_cursor[bot_id] += 1
                        bot_pickup_idx[bot_id] = 0
                        bot_task[bot_id] = "idle"
                        bot_target[bot_id] = None

                        # Reset pre_pick bots for completed orders
                        for bid2 in range(self._n_bots):
                            if bid2 == bot_id:
                                continue
                            if bot_task[bid2] == "pre_pick" and bot_target_order[bid2] <= active_order_idx:
                                has_match2 = any(item in order_items_remaining.get(active_order_idx, [])
                                                 for item in bot_inventory[bid2])
                                if has_match2:
                                    bot_task[bid2] = "deliver"
                                    bot_target[bid2] = _nearest_dropoff(bot_pos[bid2])
                                else:
                                    bot_task[bid2] = "idle"
                                    bot_target[bid2] = None
                            elif bot_task[bid2] == "pick" and bot_target_order[bid2] < active_order_idx:
                                bot_task[bid2] = "idle"
                                bot_target[bid2] = None
                            elif bot_task[bid2] == "deliver":
                                new_rem = order_items_remaining.get(active_order_idx, [])
                                if not any(item in new_rem for item in bot_inventory[bid2]):
                                    bot_task[bid2] = "idle"
                                    bot_target[bid2] = None
                    else:
                        bot_task[bot_id] = "idle"
                        bot_target[bot_id] = None

            # Re-assign after Phase 1
            _assign_from_trips()

            # Phase 2: PIBT
            targets: dict[int, Pos] = {}
            urgency_map: dict[int, int] = {}
            idle_bots: set[int] = set()

            for bot_id in range(self._n_bots):
                if bot_id in action_bots:
                    targets[bot_id] = bot_pos[bot_id]
                    urgency_map[bot_id] = -1
                    continue

                cur = bot_pos[bot_id]
                task = bot_task[bot_id]
                target = bot_target[bot_id]

                # ESCAPE from drop-off area
                if task != "deliver" and (cur in _dropoff_set or cur in _dropoff_adjacent):
                    if task in ("pick", "pre_pick") and target:
                        targets[bot_id] = target
                    else:
                        targets[bot_id] = _idle_positions[bot_id % len(_idle_positions)]
                    urgency_map[bot_id] = -1
                    continue

                if task == "deliver" and target:
                    targets[bot_id] = target
                    urgency_map[bot_id] = 0
                elif task == "pick" and target:
                    targets[bot_id] = target
                    urgency_map[bot_id] = 1
                elif task == "pre_pick" and target:
                    targets[bot_id] = target
                    urgency_map[bot_id] = 2
                else:
                    targets[bot_id] = _idle_positions[bot_id % len(_idle_positions)]
                    urgency_map[bot_id] = 3
                    idle_bots.add(bot_id)

            next_positions = pibt.resolve(
                bots=bot_pos,
                targets=targets,
                tiebreak_offset=round_t,
                idle_bots=idle_bots,
                urgency=urgency_map,
            )

            # Phase 3: Record movements
            for bot_id in range(self._n_bots):
                if bot_id in action_bots:
                    continue
                cur = bot_pos[bot_id]
                next_p = next_positions[bot_id]
                if next_p != cur:
                    bot_actions[bot_id].append(MAPFAction(direction_action(cur, next_p), cur))
                else:
                    bot_actions[bot_id].append(MAPFAction("wait", cur))

            new_pos: dict[int, Pos] = {}
            for bot_id in range(self._n_bots):
                if bot_id in action_bots:
                    new_pos[bot_id] = bot_pos[bot_id]
                else:
                    new_pos[bot_id] = next_positions[bot_id]
            bot_pos = new_pos

        print(f"\nHybrid: score={score}, orders={active_order_idx}/{n_orders}")
        delivering = sum(1 for bid in range(self._n_bots) if bot_task[bid] == "deliver")
        picking = sum(1 for bid in range(self._n_bots) if bot_task[bid] == "pick")
        idle = sum(1 for bid in range(self._n_bots) if bot_task[bid] == "idle")
        print(f"Final state: {delivering} delivering, {picking} picking, {idle} idle")

        return bot_actions, pickup_schedule, dropoff_schedule

    def _simulate_reactive(
        self,
    ) -> tuple[dict[int, list[MAPFAction]], list[dict], list[dict]]:
        """Simulate with reactive order-by-order strategy + PIBT collision avoidance.

        Key features matching reactive bot:
        - ESCAPE urgency for bots at/near drop-off that aren't delivering
        - Unlimited deliverers for 10+ bots (PIBT handles queuing)
        - Stuck detection: reset bots stuck for 5+ rounds
        - Mid-map idle parking instead of far-away spawn
        - Item batching: pick up nearby items before delivering
        - Closest-pair matching for bot-to-item assignment
        - PIBT handles all collision resolution
        """
        from bot.engine.pibt import PIBTResolver

        orders = self._export["order_items"]
        shelf_lookup = self._export["shelf_lookup"]

        pibt = PIBTResolver(
            grid=self._grid,
            distance_fn=self._engine.distance,
            one_way=getattr(self._engine, '_one_way', None),
        )

        bot_pos: dict[int, Pos] = {i: self._spawn_pos for i in range(self._n_bots)}
        bot_inventory: dict[int, list[str]] = {i: [] for i in range(self._n_bots)}
        bot_actions: dict[int, list[MAPFAction]] = {i: [] for i in range(self._n_bots)}
        # Route: list of (pos, item_type) to pick, then deliver
        bot_route: dict[int, list[tuple[Pos, str]]] = {i: [] for i in range(self._n_bots)}
        bot_task: dict[int, str] = {i: "idle" for i in range(self._n_bots)}
        bot_target: dict[int, Pos | None] = {i: None for i in range(self._n_bots)}
        bot_target_type: dict[int, str] = {i: "" for i in range(self._n_bots)}
        bot_target_order: dict[int, int] = {i: -1 for i in range(self._n_bots)}
        pickup_schedule: list[dict] = []
        dropoff_schedule: list[dict] = []

        # Stuck detection: track position history per bot
        bot_pos_history: dict[int, list[Pos]] = {i: [] for i in range(self._n_bots)}
        STUCK_THRESHOLD = 5

        n_orders = len(orders)
        active_order_idx = 0
        order_items_remaining: dict[int, list[str]] = {
            idx: list(items) for idx, items in orders.items()
        }
        claimed_items: set[tuple[Pos, str]] = set()
        score = 0

        # Dropoff scheduling: unlimited for 10+ bots, throttled for fewer
        MAX_DELIVERERS = 999 if self._n_bots >= 10 else 2 * len(self._drop_off_zones)

        # Compute mid-map idle position (center of grid, walkable)
        grid_w = self._grid.width
        grid_h = self._grid.height
        mid_x, mid_y = grid_w // 2, grid_h // 2
        # Find nearest walkable cell to center
        _idle_target = (mid_x, mid_y)
        if not self._grid.is_walkable((mid_x, mid_y)):
            # Search outward for walkable
            for r in range(1, max(grid_w, grid_h)):
                found = False
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        nx, ny = mid_x + dx, mid_y + dy
                        if self._grid.is_walkable((nx, ny)):
                            _idle_target = (nx, ny)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        # Distribute idle bots across multiple positions to avoid clustering
        _idle_positions: list[Pos] = []
        # Generate spread-out idle positions along cross-corridors
        for y in [grid_h // 4, grid_h // 2, 3 * grid_h // 4]:
            for x in [grid_w // 4, grid_w // 2, 3 * grid_w // 4]:
                if self._grid.is_walkable((x, y)):
                    _idle_positions.append((x, y))
        if not _idle_positions:
            _idle_positions = [_idle_target]

        # Drop-off zone set for fast lookup
        _dropoff_set = set(self._drop_off_zones)
        # Adjacent cells to drop-off zones
        _dropoff_adjacent: set[Pos] = set()
        for dz in self._drop_off_zones:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                adj = (dz[0] + dx, dz[1] + dy)
                if 0 <= adj[0] < grid_w and 0 <= adj[1] < grid_h:
                    _dropoff_adjacent.add(adj)

        def _nearest_dropoff(pos: Pos) -> Pos:
            best = self._drop_off_zones[0]
            best_d = self._engine.distance(pos, best)
            for z in self._drop_off_zones[1:]:
                d = self._engine.distance(pos, z)
                if d < best_d:
                    best, best_d = z, d
            return best

        def _find_item_near(ref_pos: Pos, item_type: str) -> tuple[Pos, int] | None:
            """Find nearest unclaimed shelf with this item type from ref position."""
            options = shelf_lookup.get(item_type, [])
            if not options:
                return None
            best_pos = None
            best_dist = float("inf")
            for opt in options:
                key = (opt.pickup_pos, item_type)
                if key in claimed_items:
                    continue
                d = self._engine.distance(ref_pos, opt.pickup_pos)
                if d < best_dist:
                    best_dist = d
                    best_pos = opt.pickup_pos
            if best_pos is None:
                return None
            return best_pos, best_dist

        def _try_batch_route(bid: int, first_type: str, first_pos: Pos) -> list[tuple[Pos, str]]:
            """Try to batch nearby active-order items into a multi-pickup route.

            Returns list of additional (pos, type) stops.
            If first_type is non-empty, it represents an item being picked (not yet in inventory),
            so we subtract 1 extra slot for it.
            Max inventory is 3.

            Strategy: pick up items that are ON THE WAY between first_pos and drop-off.
            E.g., if heading to butter at (4,2), pick up eggs at (4,3) and flour at (4,5)
            on the return trip. This adds only 0-2 steps per extra item.
            """
            remaining = order_items_remaining.get(active_order_idx, [])
            if not remaining:
                return []

            inv_size = len(bot_inventory[bid])
            # If first_type is set, bot hasn't picked it yet but will
            slots_left = 3 - inv_size - (1 if first_type else 0)
            if slots_left <= 0:
                return []

            # Items still needed (excluding what pickers are targeting + first_type)
            needed_types: list[str] = list(remaining)
            # Remove first_type (already assigned)
            if first_type and first_type in needed_types:
                needed_types.remove(first_type)
            for b2 in range(self._n_bots):
                if b2 == bid:
                    continue
                if bot_task[b2] == "pick":
                    if bot_target_type[b2] in needed_types:
                        needed_types.remove(bot_target_type[b2])
                    for rpos, rtype in bot_route[b2]:
                        if rtype in needed_types:
                            needed_types.remove(rtype)
                if bot_task[b2] == "deliver":
                    for item in bot_inventory[b2]:
                        if item in needed_types:
                            needed_types.remove(item)

            route: list[tuple[Pos, str]] = []
            cur_pos = first_pos

            # Use a larger radius for batching — items along x=4 corridor are
            # on the return path to drop-off, so extra distance is minimal
            BATCH_RADIUS = 25

            for _ in range(slots_left):
                if not needed_types:
                    break
                best_item = None
                best_pos = None
                best_score = float("inf")
                seen = set()
                for nt in needed_types:
                    if nt in seen:
                        continue
                    seen.add(nt)
                    result = _find_item_near(cur_pos, nt)
                    if result is None:
                        continue
                    item_pos, dist_from_cur = result
                    if dist_from_cur > BATCH_RADIUS:
                        continue
                    # Score: how much EXTRA distance does this detour add?
                    # Without detour: cur_pos -> drop-off
                    # With detour: cur_pos -> item_pos -> drop-off
                    # Extra = (dist_cur_to_item + dist_item_to_dropoff) - dist_cur_to_dropoff
                    dist_item_to_dropoff = self._engine.distance(
                        item_pos, _nearest_dropoff(item_pos))
                    dist_cur_to_dropoff = self._engine.distance(
                        cur_pos, _nearest_dropoff(cur_pos))
                    extra_distance = (dist_from_cur + dist_item_to_dropoff) - dist_cur_to_dropoff
                    # Only batch if detour is small (saves more than it costs)
                    if extra_distance < 15:
                        if extra_distance < best_score:
                            best_score = extra_distance
                            best_pos = item_pos
                            best_item = nt
                if best_item and best_pos:
                    route.append((best_pos, best_item))
                    claimed_items.add((best_pos, best_item))
                    needed_types.remove(best_item)
                    cur_pos = best_pos
                else:
                    break

            return route

        def _assign_tasks():
            """Assign tasks to idle bots with dropoff throttling + batching + preview pre-pick."""
            nonlocal active_order_idx
            if active_order_idx >= n_orders:
                return

            remaining = order_items_remaining.get(active_order_idx, [])

            # Count current deliverers
            current_deliverers = sum(
                1 for bid in range(self._n_bots) if bot_task[bid] == "deliver"
            )

            # Bots with items matching active order -> deliver (if slots available)
            idle_with_items: list[tuple[int, int]] = []
            for bid in range(self._n_bots):
                if bot_task[bid] != "idle":
                    continue
                if not bot_inventory[bid]:
                    continue
                has_match = any(item in remaining for item in bot_inventory[bid])
                if has_match:
                    d = self._engine.distance(bot_pos[bid], _nearest_dropoff(bot_pos[bid]))
                    idle_with_items.append((d, bid))

            idle_with_items.sort()
            for d, bid in idle_with_items:
                if current_deliverers < MAX_DELIVERERS:
                    bot_task[bid] = "deliver"
                    bot_target[bid] = _nearest_dropoff(bot_pos[bid])
                    current_deliverers += 1

            # Build list of still-needed items (subtract picking bots' target items + route items)
            needed = list(remaining)
            for bid in range(self._n_bots):
                if bot_task[bid] == "pick":
                    if bot_target_type[bid] and bot_target_type[bid] in needed:
                        needed.remove(bot_target_type[bid])
                    for rpos, rtype in bot_route[bid]:
                        if rtype in needed:
                            needed.remove(rtype)
                if bot_task[bid] == "deliver":
                    for item in bot_inventory[bid]:
                        if item in needed:
                            needed.remove(item)

            # Assign items using Hungarian matching (optimal assignment)
            idle_with_room = [
                bid for bid in range(self._n_bots)
                if bot_task[bid] == "idle" and len(bot_inventory[bid]) < 3
            ]

            if idle_with_room and needed:
                # De-duplicate needed items by type (group indices)
                unique_needed: list[tuple[str, list[int]]] = []
                seen_types: dict[str, int] = {}
                for ni, item_type in enumerate(needed):
                    if item_type not in seen_types:
                        seen_types[item_type] = len(unique_needed)
                        unique_needed.append((item_type, [ni]))
                    else:
                        unique_needed[seen_types[item_type]][1].append(ni)

                # Build item options: for each unique type, find nearest shelf per bot
                item_options: list[tuple[str, int, Pos | None, dict[int, tuple[Pos, int]]]] = []
                for item_type, indices in unique_needed:
                    bot_dists: dict[int, tuple[Pos, int]] = {}
                    for bid in idle_with_room:
                        result = _find_item_near(bot_pos[bid], item_type)
                        if result is not None:
                            bot_dists[bid] = result
                    item_options.append((item_type, len(indices), None, bot_dists))

                # Expand to individual item slots for assignment
                expanded_items: list[tuple[str, int, dict[int, tuple[Pos, int]]]] = []
                for item_type, count, _, bot_dists in item_options:
                    for _ in range(count):
                        expanded_items.append((item_type, _, bot_dists))

                n_bots_avail = len(idle_with_room)
                n_items = len(expanded_items)

                if n_bots_avail > 0 and n_items > 0:
                    try:
                        import numpy as np
                        from scipy.optimize import linear_sum_assignment

                        # Build cost matrix: bots x items
                        INF = 9999
                        cost = np.full((n_bots_avail, n_items), INF, dtype=np.float64)
                        pos_lookup: dict[tuple[int, int], Pos] = {}  # (bot_idx, item_idx) -> pickup_pos

                        for bi, bid in enumerate(idle_with_room):
                            for ii, (item_type, _, bot_dists) in enumerate(expanded_items):
                                if bid in bot_dists:
                                    pos, dist = bot_dists[bid]
                                    cost[bi, ii] = dist
                                    pos_lookup[(bi, ii)] = pos

                        row_ind, col_ind = linear_sum_assignment(cost)

                        for bi, ii in zip(row_ind, col_ind):
                            if cost[bi, ii] >= INF:
                                continue
                            bid = idle_with_room[bi]
                            item_type = expanded_items[ii][0]
                            pos = pos_lookup[(bi, ii)]
                            claimed_items.add((pos, item_type))
                            bot_task[bid] = "pick"
                            bot_target[bid] = pos
                            bot_target_type[bid] = item_type
                            bot_target_order[bid] = active_order_idx

                            # Try to batch more items into a route
                            route = _try_batch_route(bid, item_type, pos)
                            bot_route[bid] = route

                    except ImportError:
                        # Fallback: greedy matching if scipy unavailable
                        pairs: list[tuple[int, int, int, str, Pos]] = []
                        for bid in idle_with_room:
                            st: set[str] = set()
                            for ni, item_type in enumerate(needed):
                                if item_type in st:
                                    continue
                                st.add(item_type)
                                result = _find_item_near(bot_pos[bid], item_type)
                                if result is not None:
                                    pos, dist = result
                                    pairs.append((dist, bid, ni, item_type, pos))
                        pairs.sort()
                        assigned_bots_fb: set[int] = set()
                        assigned_items_fb: set[int] = set()
                        for dist, bid, ni, item_type, pos in pairs:
                            if bid in assigned_bots_fb:
                                continue
                            target_ni = None
                            for i, it in enumerate(needed):
                                if it == item_type and i not in assigned_items_fb:
                                    target_ni = i
                                    break
                            if target_ni is None:
                                continue
                            claimed_items.add((pos, item_type))
                            assigned_items_fb.add(target_ni)
                            bot_task[bid] = "pick"
                            bot_target[bid] = pos
                            bot_target_type[bid] = item_type
                            bot_target_order[bid] = active_order_idx
                            assigned_bots_fb.add(bid)
                            route = _try_batch_route(bid, item_type, pos)
                            bot_route[bid] = route
                            if len(assigned_items_fb) >= len(needed):
                                break

            # --- PREVIEW PRE-PICK: assign remaining idle bots to next order ---
            # But ONLY if we know the item types (CRITICAL: orders beyond preview are unknown)
            preview_idx = active_order_idx + 1
            if preview_idx < n_orders and preview_idx in order_items_remaining:
                preview_remaining = list(order_items_remaining.get(preview_idx, []))
                # Subtract items being actively pre-picked for this preview order
                for bid in range(self._n_bots):
                    if bot_task[bid] == "pre_pick" and bot_target_order[bid] == preview_idx:
                        if bot_target_type[bid] in preview_remaining:
                            preview_remaining.remove(bot_target_type[bid])
                        for rpos, rtype in bot_route[bid]:
                            if rtype in preview_remaining:
                                preview_remaining.remove(rtype)
                # Subtract items held by idle bots ONLY if not needed for active order
                active_remaining = list(order_items_remaining.get(active_order_idx, []))
                for bid in range(self._n_bots):
                    if bot_task[bid] == "idle" and bot_inventory[bid]:
                        for item in bot_inventory[bid]:
                            if item in active_remaining:
                                # This item is for the active order, don't subtract from preview
                                active_remaining.remove(item)
                            elif item in preview_remaining:
                                # Item not needed for active, must be pre-picked for preview
                                preview_remaining.remove(item)

                preview_idle = [
                    bid for bid in range(self._n_bots)
                    if bot_task[bid] == "idle" and len(bot_inventory[bid]) < 3
                ]

                if preview_idle and preview_remaining:
                    pairs2: list[tuple[int, int, int, str, Pos]] = []
                    for bid in preview_idle:
                        seen_types2: set[str] = set()
                        for ni, item_type in enumerate(preview_remaining):
                            if item_type in seen_types2:
                                continue
                            seen_types2.add(item_type)
                            result = _find_item_near(bot_pos[bid], item_type)
                            if result is not None:
                                pos, dist = result
                                pairs2.append((dist, bid, ni, item_type, pos))

                    pairs2.sort()
                    assigned_bots2: set[int] = set()
                    assigned_items2: set[int] = set()
                    for dist, bid, ni, item_type, pos in pairs2:
                        if bid in assigned_bots2:
                            continue
                        target_ni = None
                        for i, it in enumerate(preview_remaining):
                            if it == item_type and i not in assigned_items2:
                                target_ni = i
                                break
                        if target_ni is None:
                            continue
                        claimed_items.add((pos, item_type))
                        assigned_items2.add(target_ni)
                        bot_task[bid] = "pre_pick"
                        bot_target[bid] = pos
                        bot_target_type[bid] = item_type
                        bot_target_order[bid] = preview_idx
                        assigned_bots2.add(bid)
                        if len(assigned_items2) >= len(preview_remaining):
                            break

            pass  # (no second-preview look-ahead)


        for round_t in range(self._max_rounds):
            if active_order_idx >= n_orders:
                break

            if round_t % 50 == 0 or round_t < 5:
                n_idle = sum(1 for bid in range(self._n_bots) if bot_task[bid] == "idle")
                n_pick = sum(1 for bid in range(self._n_bots) if bot_task[bid] == "pick")
                n_deliv = sum(1 for bid in range(self._n_bots) if bot_task[bid] == "deliver")
                n_pre = sum(1 for bid in range(self._n_bots) if bot_task[bid] == "pre_pick")
                remaining_count = len(order_items_remaining.get(active_order_idx, []))
                print(f"  Round {round_t}: order={active_order_idx}, "
                      f"score={score}, pick/deliv/pre/idle={n_pick}/{n_deliv}/{n_pre}/{n_idle}, "
                      f"remaining={remaining_count}")

            # Stuck detection: reset bots that haven't moved for STUCK_THRESHOLD rounds
            for bid in range(self._n_bots):
                bot_pos_history[bid].append(bot_pos[bid])
                if len(bot_pos_history[bid]) > STUCK_THRESHOLD:
                    bot_pos_history[bid] = bot_pos_history[bid][-STUCK_THRESHOLD:]
                if (len(bot_pos_history[bid]) >= STUCK_THRESHOLD
                        and bot_task[bid] in ("pick", "deliver", "pre_pick")
                        and len(set(bot_pos_history[bid])) == 1):
                    # Bot stuck — release its claims and go idle
                    if bot_task[bid] in ("pick", "pre_pick"):
                        key = (bot_target[bid], bot_target_type[bid])
                        claimed_items.discard(key)
                        for rpos, rtype in bot_route[bid]:
                            claimed_items.discard((rpos, rtype))
                    bot_task[bid] = "idle"
                    bot_target[bid] = None
                    bot_target_type[bid] = ""
                    bot_route[bid] = []
                    bot_pos_history[bid] = []

            # Assign tasks to idle bots
            _assign_tasks()

            # Phase 1: Determine action bots (at target, perform action)
            action_bots: set[int] = set()

            for bot_id in range(self._n_bots):
                cur = bot_pos[bot_id]
                task = bot_task[bot_id]
                target = bot_target[bot_id]

                if task in ("pick", "pre_pick") and cur == target and len(bot_inventory[bot_id]) < 3:
                    # At pickup position — pick up item
                    item_type = bot_target_type[bot_id]
                    action_bots.add(bot_id)
                    bot_actions[bot_id].append(MAPFAction("pick_up", cur, item_type))
                    bot_inventory[bot_id].append(item_type)
                    pickup_schedule.append({
                        "round": round_t, "bot_id": bot_id,
                        "position": cur, "item_type": item_type,
                        "order_idx": bot_target_order[bot_id],
                    })
                    if task == "pre_pick":
                        # Pre-pick done — go idle (hold item for next order)
                        bot_task[bot_id] = "idle"
                        bot_target[bot_id] = None
                        bot_target_type[bot_id] = ""
                    elif bot_route[bot_id]:
                        # More pickups in the route
                        next_stop = bot_route[bot_id].pop(0)
                        bot_target[bot_id] = next_stop[0]
                        bot_target_type[bot_id] = next_stop[1]
                    elif len(bot_inventory[bot_id]) < 3:
                        # No route remaining, but room for more items — try to batch
                        # Always try batching — the detour cost check in
                        # _try_batch_route ensures we only batch if it's worth it
                        route = _try_batch_route(bot_id, "", cur)
                        if route:
                            next_stop = route.pop(0)
                            bot_target[bot_id] = next_stop[0]
                            bot_target_type[bot_id] = next_stop[1]
                            bot_route[bot_id] = route
                        else:
                            bot_task[bot_id] = "deliver"
                            bot_target[bot_id] = _nearest_dropoff(bot_pos[bot_id])
                            bot_target_type[bot_id] = ""
                    else:
                        # Go deliver
                        bot_task[bot_id] = "deliver"
                        bot_target[bot_id] = _nearest_dropoff(bot_pos[bot_id])
                        bot_target_type[bot_id] = ""

                elif task == "deliver" and cur == target:
                    # At dropoff — deliver matching items
                    remaining = order_items_remaining.get(active_order_idx, [])
                    has_match = any(item in remaining for item in bot_inventory[bot_id])

                    if has_match:
                        action_bots.add(bot_id)
                        bot_actions[bot_id].append(MAPFAction("drop_off", cur))

                        new_inv = []
                        delivered = 0
                        for item in bot_inventory[bot_id]:
                            if item in remaining:
                                remaining.remove(item)
                                delivered += 1
                            else:
                                new_inv.append(item)
                        bot_inventory[bot_id] = new_inv

                        score += delivered

                        dropoff_schedule.append({
                            "round": round_t, "bot_id": bot_id,
                            "position": cur, "order_idx": active_order_idx,
                            "delivered": delivered,
                        })

                        # Order completion + auto-delivery chain
                        while not remaining and active_order_idx < n_orders:
                            order_bonus = 5
                            score += order_bonus
                            print(f"  ORDER {active_order_idx} COMPLETE at round {round_t} "
                                  f"(score={score})")
                            active_order_idx += 1
                            if active_order_idx >= n_orders:
                                break
                            remaining = order_items_remaining.get(active_order_idx, [])
                            # Auto-deliver from delivering bot ONLY (matches game rules)
                            auto_inv = []
                            for item in bot_inventory[bot_id]:
                                if item in remaining:
                                    remaining.remove(item)
                                    score += 1
                                else:
                                    auto_inv.append(item)
                            bot_inventory[bot_id] = auto_inv

                        # Reset this bot
                        bot_task[bot_id] = "idle"
                        bot_target[bot_id] = None
                        # Clear claimed items for new order
                        claimed_items.clear()
                        for bid2 in range(self._n_bots):
                            if bid2 == bot_id:
                                continue
                            task2 = bot_task[bid2]
                            if task2 in ("pick", "deliver", "pre_pick"):
                                new_remaining = order_items_remaining.get(active_order_idx, [])
                                if task2 == "deliver":
                                    has_match2 = any(item in new_remaining for item in bot_inventory[bid2])
                                    if not has_match2:
                                        bot_task[bid2] = "idle"
                                        bot_target[bid2] = None
                                elif task2 == "pre_pick":
                                    if bot_target_order[bid2] <= active_order_idx:
                                        # Was pre-picking for now-active order
                                        # If bot has items matching active order, deliver
                                        has_match2 = any(item in new_remaining for item in bot_inventory[bid2])
                                        if has_match2:
                                            bot_task[bid2] = "deliver"
                                            bot_target[bid2] = _nearest_dropoff(bot_pos[bid2])
                                            bot_target_type[bid2] = ""
                                        else:
                                            bot_task[bid2] = "idle"
                                            bot_target[bid2] = None
                                    else:
                                        # Still pre-picking for future order — keep
                                        key = (bot_target[bid2], bot_target_type[bid2])
                                        claimed_items.add(key)
                                elif task2 == "pick":
                                    if bot_target_order[bid2] < active_order_idx:
                                        bot_task[bid2] = "idle"
                                        bot_target[bid2] = None
                                        bot_route[bid2] = []
                                    else:
                                        key = (bot_target[bid2], bot_target_type[bid2])
                                        claimed_items.add(key)
                                        for rpos, rtype in bot_route[bid2]:
                                            claimed_items.add((rpos, rtype))
                    else:
                        # No matching items — go idle, reassign
                        bot_task[bot_id] = "idle"
                        bot_target[bot_id] = None

            # Re-assign after deliveries (bots freed in Phase 1 get new tasks immediately)
            _assign_tasks()

            # Phase 2: PIBT for moving bots
            targets: dict[int, Pos] = {}
            urgency_map: dict[int, int] = {}
            idle_bots: set[int] = set()

            for bot_id in range(self._n_bots):
                if bot_id in action_bots:
                    targets[bot_id] = bot_pos[bot_id]
                    urgency_map[bot_id] = -1  # Stay in place (action)
                    continue

                cur = bot_pos[bot_id]
                task = bot_task[bot_id]
                target = bot_target[bot_id]

                # ESCAPE: bot at/adjacent to drop-off but NOT delivering → must move away
                if (task != "deliver"
                        and (cur in _dropoff_set or cur in _dropoff_adjacent)):
                    if task in ("pick", "pre_pick") and target:
                        targets[bot_id] = target
                    else:
                        targets[bot_id] = _idle_positions[bot_id % len(_idle_positions)]
                    urgency_map[bot_id] = -1  # ESCAPE (highest priority)
                    continue

                if task == "deliver" and target:
                    targets[bot_id] = target
                    urgency_map[bot_id] = 0  # DELIVER
                elif task == "pick" and target:
                    targets[bot_id] = target
                    urgency_map[bot_id] = 1  # PICK_UP
                elif task == "pre_pick" and target:
                    targets[bot_id] = target
                    urgency_map[bot_id] = 2  # PRE_PICK
                else:
                    # Idle — distribute across mid-map positions
                    idle_pos = _idle_positions[bot_id % len(_idle_positions)]
                    targets[bot_id] = idle_pos
                    urgency_map[bot_id] = 3
                    idle_bots.add(bot_id)

            next_positions = pibt.resolve(
                bots=bot_pos,
                targets=targets,
                tiebreak_offset=round_t,
                idle_bots=idle_bots,
                urgency=urgency_map,
            )

            # Phase 3: Record movements
            for bot_id in range(self._n_bots):
                if bot_id in action_bots:
                    continue
                cur = bot_pos[bot_id]
                next_p = next_positions[bot_id]
                if next_p != cur:
                    bot_actions[bot_id].append(
                        MAPFAction(direction_action(cur, next_p), cur)
                    )
                else:
                    bot_actions[bot_id].append(MAPFAction("wait", cur))

            # Update positions
            new_pos: dict[int, Pos] = {}
            for bot_id in range(self._n_bots):
                if bot_id in action_bots:
                    new_pos[bot_id] = bot_pos[bot_id]
                else:
                    new_pos[bot_id] = next_positions[bot_id]
            bot_pos = new_pos

        print(f"\nFinal: score={score}, orders={active_order_idx}/{n_orders}")

        # Per-order timing analysis
        delivering = sum(1 for bid in range(self._n_bots) if bot_task[bid] == "deliver")
        picking = sum(1 for bid in range(self._n_bots) if bot_task[bid] == "pick")
        idle = sum(1 for bid in range(self._n_bots) if bot_task[bid] == "idle")
        print(f"Final state: {delivering} delivering, {picking} picking, {idle} idle")

        return bot_actions, pickup_schedule, dropoff_schedule

    def _build_waypoint_queues(
        self,
        trips: list[tuple[int, BotTrip]],
        order_activations: dict[int, int],
        max_preview_orders: int = 1,
    ) -> dict[int, list[BotWaypoint]]:
        """Build ordered waypoint queue for each bot from trip assignments.

        Each trip becomes a sequence: [pickup_1, pickup_2, ..., drop_off].
        Limits pre-queuing to active order + max_preview_orders to prevent dead weight.
        """
        queues: dict[int, list[BotWaypoint]] = {i: [] for i in range(self._n_bots)}

        # Sort trips per bot by start_round (temporal order)
        bot_trips: dict[int, list[tuple[int, BotTrip]]] = {}
        for order_idx, trip in trips:
            bot_trips.setdefault(trip.bot_id, []).append((order_idx, trip))

        # Determine max order index to pre-queue
        # For 10+ bots, limit to first 2 orders to reduce dead weight
        max_order_idx = max_preview_orders  # active(0) + preview(1)
        if self._n_bots >= 10:
            max_order_idx = max_preview_orders

        for bot_id in range(self._n_bots):
            if bot_id not in bot_trips:
                continue
            sorted_trips = sorted(bot_trips[bot_id], key=lambda x: x[1].start_round)
            for order_idx, trip in sorted_trips:
                # Limit pre-queuing: only queue trips for first N orders initially
                # The _simulate method will dynamically add more as orders complete
                if order_idx > max_order_idx:
                    continue
                for i, pos in enumerate(trip.pickup_positions):
                    item_type = trip.items[i] if i < len(trip.items) else ""
                    queues[bot_id].append(BotWaypoint(
                        position=pos,
                        action="pick_up",
                        item_type=item_type,
                        order_idx=order_idx,
                    ))
                queues[bot_id].append(BotWaypoint(
                    position=trip.drop_off,
                    action="drop_off",
                    order_idx=order_idx,
                ))

        for bot_id in range(self._n_bots):
            print(f"  Bot {bot_id}: {len(queues[bot_id])} waypoints")

        return queues

    def _simulate(
        self,
        waypoint_queues: dict[int, list[BotWaypoint]],
        order_activations: dict[int, int],
    ) -> tuple[dict[int, list[MAPFAction]], list[dict], list[dict]]:
        """Simulate game round by round using PIBT for collision-free movement.

        Uses the same PIBTResolver as the reactive bot, which already handles
        the game's sequential ID-order execution model correctly (cancels
        following conflicts, prevents swaps).

        Each round:
        1. Compute targets and urgency from waypoint queues
        2. PIBT resolves collision-free next positions for all 20 bots
        3. Process actions at waypoints (pickup, dropoff, order completion)
        """
        from bot.engine.pibt import PIBTResolver

        orders = self._export["order_items"]

        pibt = PIBTResolver(
            grid=self._grid,
            distance_fn=self._engine.distance,
            one_way=getattr(self._engine, '_one_way', None),
        )

        bot_pos: dict[int, Pos] = {i: self._spawn_pos for i in range(self._n_bots)}
        bot_inventory: dict[int, list[str]] = {i: [] for i in range(self._n_bots)}
        bot_actions: dict[int, list[MAPFAction]] = {i: [] for i in range(self._n_bots)}
        bot_cursor: dict[int, int] = {i: 0 for i in range(self._n_bots)}
        pickup_schedule: list[dict] = []
        dropoff_schedule: list[dict] = []

        n_orders = len(orders)
        active_order_idx = 0
        order_items_remaining: dict[int, list[str]] = {
            idx: list(items) for idx, items in orders.items()
        }

        # Track bots that perform an action this round (stay in place)
        _dbg_moves = 0
        _dbg_waits = 0
        MAX_DELIVERERS = 999 if self._n_bots >= 10 else 2 * len(self._drop_off_zones)
        score = 0
        # Dead-weight tracking: rounds since bot picked up items that don't match
        bot_dead_weight_rounds: dict[int, int] = {i: 0 for i in range(self._n_bots)}

        for round_t in range(self._max_rounds):
            if active_order_idx >= n_orders:
                break
            if round_t % 50 == 0 or round_t < 5:
                active_bots = sum(1 for b in range(self._n_bots)
                                  if bot_cursor[b] < len(waypoint_queues[b]))
                print(f"  Round {round_t}: {active_bots} active, order={active_order_idx}, "
                      f"pickups={len(pickup_schedule)}, dropoffs={len(dropoff_schedule)}")
                if round_t < 5 or round_t % 100 == 0:
                    for bid in range(min(5, self._n_bots)):
                        cur = bot_cursor[bid]
                        q = waypoint_queues[bid]
                        wp_info = f"wp[{cur}]={q[cur].action}@{q[cur].position} o{q[cur].order_idx}" if cur < len(q) else "DONE"
                        print(f"    B{bid} at {bot_pos[bid]} inv={bot_inventory[bid]} {wp_info}")

            # Phase 1: Determine which bots perform actions (stay in place)
            # and which bots need to move toward waypoints
            action_bots: set[int] = set()  # bots performing pickup/dropoff/wait this round

            # Count deliverers this round for throttling
            deliverers_this_round = 0

            for bot_id in range(self._n_bots):
                cur = bot_pos[bot_id]
                cursor = bot_cursor[bot_id]
                queue = waypoint_queues[bot_id]

                # DYNAMIC DELIVERY: if bot is at a dropoff zone AND has items
                # matching active order, deliver NOW (regardless of cursor).
                # This handles dead-weight items that coincidentally match.
                if (cur in self._drop_off_zones
                        and bot_inventory[bot_id]
                        and deliverers_this_round < MAX_DELIVERERS):
                    remaining = order_items_remaining.get(active_order_idx, [])
                    has_match = any(
                        item in remaining for item in bot_inventory[bot_id])
                    if has_match:
                        action_bots.add(bot_id)
                        bot_actions[bot_id].append(MAPFAction("drop_off", cur))
                        deliverers_this_round += 1

                        new_inv = []
                        delivered = 0
                        for item in bot_inventory[bot_id]:
                            if item in remaining:
                                remaining.remove(item)
                                delivered += 1
                                score += 1
                            else:
                                new_inv.append(item)
                        bot_inventory[bot_id] = new_inv

                        dropoff_schedule.append({
                            "round": round_t, "bot_id": bot_id,
                            "position": cur,
                            "order_idx": active_order_idx,
                            "delivered": delivered,
                        })

                        # If cursor was at dropoff for THIS order, advance it
                        if (cursor < len(queue)
                                and queue[cursor].action == "drop_off"
                                and queue[cursor].order_idx <= active_order_idx):
                            bot_cursor[bot_id] = cursor + 1

                        # Order completion chain
                        while not remaining and active_order_idx < n_orders:
                            score += 5
                            print(f"  ORDER {active_order_idx} COMPLETE "
                                  f"at round {round_t} (score={score})")
                            active_order_idx += 1
                            if active_order_idx >= n_orders:
                                break
                            remaining = order_items_remaining.get(
                                active_order_idx, [])
                            auto_inv = []
                            for item in bot_inventory[bot_id]:
                                if item in remaining:
                                    remaining.remove(item)
                                    score += 1
                                else:
                                    auto_inv.append(item)
                            bot_inventory[bot_id] = auto_inv
                        continue  # Done with this bot for Phase 1

                if cursor >= len(queue):
                    continue

                # Skip waypoints for completed orders (dead weight avoidance)
                wp = queue[cursor]
                while cursor < len(queue) and wp.order_idx < active_order_idx:
                    bot_cursor[bot_id] = cursor + 1
                    cursor += 1
                    if cursor < len(queue):
                        wp = queue[cursor]
                if cursor >= len(queue):
                    continue

                wp = queue[cursor]
                if cur != wp.position:
                    continue  # Not at waypoint — needs to move

                # At waypoint — perform action
                if wp.action == "pick_up" and len(bot_inventory[bot_id]) < 3:
                    action_bots.add(bot_id)
                    bot_actions[bot_id].append(
                        MAPFAction("pick_up", cur, wp.item_type)
                    )
                    bot_inventory[bot_id].append(wp.item_type)
                    pickup_schedule.append({
                        "round": round_t, "bot_id": bot_id,
                        "position": cur, "item_type": wp.item_type,
                        "order_idx": wp.order_idx,
                    })
                    bot_cursor[bot_id] = cursor + 1

                elif wp.action == "drop_off":
                    # Check if bot has items matching the active order
                    remaining = order_items_remaining.get(active_order_idx, [])
                    has_match = any(item in remaining for item in bot_inventory[bot_id])

                    # Dropoff throttling: limit concurrent deliveries
                    deliverers_this_round = sum(
                        1 for bid in action_bots
                        if bot_actions[bid] and bot_actions[bid][-1].action == "drop_off"
                    )

                    if has_match and deliverers_this_round < MAX_DELIVERERS:
                        action_bots.add(bot_id)
                        bot_actions[bot_id].append(MAPFAction("drop_off", cur))

                        new_inv = []
                        delivered = 0
                        for item in bot_inventory[bot_id]:
                            if item in remaining:
                                remaining.remove(item)
                                delivered += 1
                                score += 1
                            else:
                                new_inv.append(item)
                        bot_inventory[bot_id] = new_inv

                        dropoff_schedule.append({
                            "round": round_t, "bot_id": bot_id,
                            "position": cur, "order_idx": wp.order_idx,
                            "delivered": delivered,
                        })
                        bot_cursor[bot_id] = cursor + 1

                        # Order completion + auto-delivery chain
                        while not remaining and active_order_idx < n_orders:
                            score += 5  # completion bonus
                            print(f"  ORDER {active_order_idx} COMPLETE "
                                  f"at round {round_t} (score={score})")
                            active_order_idx += 1
                            if active_order_idx >= n_orders:
                                break
                            remaining = order_items_remaining.get(active_order_idx, [])
                            # Auto-deliver matching items from THIS bot
                            auto_inv = []
                            for item in bot_inventory[bot_id]:
                                if item in remaining:
                                    remaining.remove(item)
                                    score += 1
                                else:
                                    auto_inv.append(item)
                            bot_inventory[bot_id] = auto_inv
                            # Also check ALL bots at dropoff zones for auto-delivery
                            for other_id in range(self._n_bots):
                                if other_id == bot_id:
                                    continue
                                if bot_pos[other_id] not in self._drop_off_zones:
                                    continue
                                auto_inv2 = []
                                for item in bot_inventory[other_id]:
                                    if item in remaining:
                                        remaining.remove(item)
                                        score += 1
                                    else:
                                        auto_inv2.append(item)
                                bot_inventory[other_id] = auto_inv2
                    else:
                        # No matching items OR throttled — PIBT will route
                        # bot to spawn (not dropoff) via Phase 2 target logic
                        pass

                elif wp.action == "pick_up":
                    # Inventory full — wait
                    action_bots.add(bot_id)
                    bot_actions[bot_id].append(MAPFAction("wait", cur))

            # Phase 2: PIBT resolution for moving bots
            # Build targets and urgency for PIBT
            targets: dict[int, Pos] = {}
            urgency: dict[int, int] = {}
            idle_bots: set[int] = set()

            for bot_id in range(self._n_bots):
                if bot_id in action_bots:
                    # Action bots stay in place — give them current pos as target
                    # with high urgency so they don't get pushed
                    targets[bot_id] = bot_pos[bot_id]
                    urgency[bot_id] = -1  # ESCAPE — highest priority
                    continue

                cursor = bot_cursor[bot_id]
                queue = waypoint_queues[bot_id]

                # Dynamic delivery override: if bot has items matching
                # active order, route to nearest dropoff (high priority)
                remaining = order_items_remaining.get(active_order_idx, [])
                has_deliverable = (
                    bot_inventory[bot_id]
                    and any(item in remaining for item in bot_inventory[bot_id])
                )

                # Dead-weight detection: track rounds with non-matching inventory
                if bot_inventory[bot_id] and not has_deliverable:
                    bot_dead_weight_rounds[bot_id] += 1
                else:
                    bot_dead_weight_rounds[bot_id] = 0

                if has_deliverable:
                    nearest = min(
                        self._drop_off_zones,
                        key=lambda z: self._engine.distance(bot_pos[bot_id], z),
                    )
                    targets[bot_id] = nearest
                    urgency[bot_id] = 0  # DELIVER
                elif (bot_inventory[bot_id] and bot_dead_weight_rounds[bot_id] >= 20):
                    # Dead weight: bot has items for 20+ rounds that don't match
                    # Route to drop-off anyway — items might match after order transition
                    nearest = min(
                        self._drop_off_zones,
                        key=lambda z: self._engine.distance(bot_pos[bot_id], z),
                    )
                    targets[bot_id] = nearest
                    urgency[bot_id] = 2  # PRE_PICK priority (low)
                elif cursor >= len(queue):
                    # Idle — park at spawn
                    targets[bot_id] = self._spawn_pos
                    urgency[bot_id] = 3
                    idle_bots.add(bot_id)
                else:
                    wp = queue[cursor]
                    if wp.action == "drop_off":
                        can_deliver = any(
                            item in remaining
                            for item in bot_inventory[bot_id])
                        if can_deliver:
                            targets[bot_id] = wp.position
                            urgency[bot_id] = 0  # DELIVER
                        else:
                            # Can't deliver yet — park at spawn
                            targets[bot_id] = self._spawn_pos
                            urgency[bot_id] = 3
                            idle_bots.add(bot_id)
                    else:
                        targets[bot_id] = wp.position
                        urgency[bot_id] = 1  # PICK_UP

            # PIBT resolves collision-free positions for ALL bots
            next_positions = pibt.resolve(
                bots=bot_pos,
                targets=targets,
                tiebreak_offset=round_t,
                idle_bots=idle_bots,
                urgency=urgency,
            )

            # Phase 3: Record movements for non-action bots
            for bot_id in range(self._n_bots):
                if bot_id in action_bots:
                    # Action already recorded — stay in place
                    continue

                cur = bot_pos[bot_id]
                next_p = next_positions[bot_id]

                if next_p != cur:
                    bot_actions[bot_id].append(
                        MAPFAction(direction_action(cur, next_p), cur)
                    )
                    _dbg_moves += 1
                else:
                    bot_actions[bot_id].append(MAPFAction("wait", cur))
                    _dbg_waits += 1

            # Update positions — action bots stay, moving bots use PIBT result
            new_pos: dict[int, Pos] = {}
            for bot_id in range(self._n_bots):
                if bot_id in action_bots:
                    new_pos[bot_id] = bot_pos[bot_id]
                else:
                    new_pos[bot_id] = next_positions[bot_id]
            bot_pos = new_pos

        print(f"\nPIBT sim: score={score}, orders={active_order_idx}/{n_orders}, "
              f"moves={_dbg_moves}, waits={_dbg_waits}")

        return bot_actions, pickup_schedule, dropoff_schedule

    def _plan_with_tsa(
        self,
        waypoint_queues: dict[int, list[BotWaypoint]],
    ) -> tuple[dict[int, list[MAPFAction]], list[dict], list[dict]]:
        """Simplest possible TSA*: vertex-only conflict detection.

        No edge/swap detection (sequential_mode=True). Just prevent
        two bots at the same position at the same time. Let the
        game server handle any remaining edge conflicts as waits.

        Key simplifications:
        - reserve_position() only (no append_path, no _bot_paths)
        - Spawn exempt from reservation (stacking allowed)
        - No idle extension (bots park at dropoff, next trip starts there)
        - sequential_mode=True for maximum path-finding success rate
        """
        from bot.engine.reservation import ReservationTable
        from bot.engine.time_space_astar import find_path_tsa

        spawn = self._spawn_pos
        trips = self._export["trips"]
        horizon = self._max_rounds + 10

        # No spawn exemption — reserve everything normally.
        # Stagger bot starts instead (1 bot per round exits spawn).
        res = ReservationTable(horizon=horizon)
        bot_actions: dict[int, list[MAPFAction]] = {
            i: [] for i in range(self._n_bots)
        }
        pickup_schedule: list[dict] = []
        dropoff_schedule: list[dict] = []
        bot_pos: dict[int, Pos] = {i: spawn for i in range(self._n_bots)}
        # Stagger starts: bot N starts at round N (only 1 can exit per round)
        bot_time: dict[int, int] = {i: i for i in range(self._n_bots)}
        total_failed = 0

        # Reserve spawn for each bot during their wait period
        for bid in range(self._n_bots):
            for t in range(bid):
                res.reserve_position(bid, spawn, t)

        # Group trips per bot
        bot_trip_list: dict[int, list[tuple[int, BotTrip]]] = {
            i: [] for i in range(self._n_bots)
        }
        for order_idx, trip in trips:
            bot_trip_list[trip.bot_id].append((order_idx, trip))
        for bid in range(self._n_bots):
            bot_trip_list[bid].sort(key=lambda x: x[1].start_round)

        max_trips = max(
            (len(tl) for tl in bot_trip_list.values()), default=0
        )
        print(f"\nTSA* (vertex-only): {len(trips)} trips, "
              f"max {max_trips}/bot, horizon {horizon}")

        def _tsa(bot_id: int, start: Pos, goal: Pos,
                 start_t: int) -> list[Pos] | None:
            if start_t >= self._max_rounds:
                return None
            return find_path_tsa(
                start=start, goal=goal, start_t=start_t,
                grid=self._grid, reservations=res, bot_id=bot_id,
                directed_neighbors_fn=self._engine._directed_neighbors,
                distance_fn=self._engine.distance,
                max_t=min(300, self._max_rounds - start_t),
                deadline_ms=5000, goal_hold=1,
                sequential_mode=True,  # vertex-only, max path success
            )

        def _reserve_path(bot_id: int, path: list[Pos],
                          start_t: int) -> None:
            """Reserve vertex positions along path."""
            for i, pos in enumerate(path):
                t = start_t + i
                if t <= horizon:
                    res.reserve_position(bot_id, pos, t)

        # Round-robin: trip 0 for all bots, then trip 1...
        for trip_round in range(max_trips):
            round_failed = 0
            for bot_id in range(self._n_bots):
                trip_list = bot_trip_list[bot_id]
                if trip_round >= len(trip_list):
                    continue

                order_idx, trip = trip_list[trip_round]
                pos = bot_pos[bot_id]
                t = bot_time[bot_id]
                if t >= self._max_rounds:
                    continue

                # Pad waits
                while len(bot_actions[bot_id]) < t:
                    bot_actions[bot_id].append(MAPFAction("wait", pos))

                trip_ok = True

                # Pickups
                for i, pickup_pos in enumerate(trip.pickup_positions):
                    if t >= self._max_rounds:
                        trip_ok = False
                        break
                    item_type = trip.items[i] if i < len(trip.items) else ""
                    path = _tsa(bot_id, pos, pickup_pos, t)
                    if path is None:
                        total_failed += 1
                        round_failed += 1
                        trip_ok = False
                        break

                    # Reserve and record travel
                    _reserve_path(bot_id, path, t)
                    for j in range(1, len(path)):
                        p0, p1 = path[j-1], path[j]
                        if p1 == p0:
                            bot_actions[bot_id].append(
                                MAPFAction("wait", p0))
                        else:
                            bot_actions[bot_id].append(
                                MAPFAction(direction_action(p0, p1), p0))
                    t += len(path) - 1
                    pos = path[-1]

                    # Pickup action
                    bot_actions[bot_id].append(
                        MAPFAction("pick_up", pos, item_type))
                    if t + 1 <= horizon:
                        res.reserve_position(bot_id, pos, t + 1)
                    t += 1

                    pickup_schedule.append({
                        "round": t - 1, "bot_id": bot_id,
                        "position": pos, "item_type": item_type,
                        "order_idx": order_idx,
                    })

                # Dropoff
                if trip_ok and t < self._max_rounds:
                    path = _tsa(bot_id, pos, trip.drop_off, t)
                    if path is None:
                        total_failed += 1
                        round_failed += 1
                        trip_ok = False

                    if trip_ok:
                        _reserve_path(bot_id, path, t)
                        for j in range(1, len(path)):
                            p0, p1 = path[j-1], path[j]
                            if p1 == p0:
                                bot_actions[bot_id].append(
                                    MAPFAction("wait", p0))
                            else:
                                bot_actions[bot_id].append(
                                    MAPFAction(direction_action(p0, p1), p0))
                        t += len(path) - 1
                        pos = path[-1]

                        bot_actions[bot_id].append(
                            MAPFAction("drop_off", pos))
                        if t + 1 <= horizon:
                            res.reserve_position(bot_id, pos, t + 1)
                        t += 1

                        dropoff_schedule.append({
                            "round": t - 1, "bot_id": bot_id,
                            "position": pos, "order_idx": order_idx,
                        })

                # Reserve idle position between trips (prevent others
                # from planning through where this bot is parked)
                for idle_t in range(t, min(t + 30, horizon + 1)):
                    existing = res._table.get((pos[0], pos[1], idle_t))
                    if existing is None or existing == bot_id:
                        res.reserve_position(bot_id, pos, idle_t)
                    else:
                        break  # another bot already there

                bot_pos[bot_id] = pos
                bot_time[bot_id] = t

            print(f"  Trip round {trip_round}: "
                  f"failed {round_failed}, total {total_failed}")

        print(f"\nTSA* done: {len(pickup_schedule)} pickups, "
              f"{len(dropoff_schedule)} dropoffs, {total_failed} failed")
        return bot_actions, pickup_schedule, dropoff_schedule

    def _bfs_step(
        self,
        bot_id: int,
        cur: Pos,
        goal: Pos,
        new_pos: dict[int, Pos],
        old_pos: dict[int, Pos],
        max_depth: int = 12,
    ) -> Pos | None:
        """Find first step of shortest path to goal, avoiding other bots.

        Uses BFS with bot positions as temporary walls. Can find multi-step
        detours around blockages that greedy 1-step would miss.
        Falls back to any free neighbor if no path to goal exists.
        """
        from collections import deque

        # Build blocked set: other bots' positions this round
        blocked = set()
        for oid in range(self._n_bots):
            if oid == bot_id:
                continue
            if oid < bot_id:
                pos = new_pos.get(oid)
            else:
                pos = old_pos.get(oid)
            if pos and pos != self._spawn_pos:
                blocked.add(pos)

        if goal == cur:
            return cur

        # BFS to find shortest path to goal avoiding blocked cells
        queue = deque([(cur, None)])  # (pos, first_step)
        visited = {cur}

        while queue:
            pos, first = queue.popleft()

            for nb in self._engine._directed_neighbors(pos):
                if nb in visited or nb in blocked:
                    continue
                step1 = first if first is not None else nb
                if nb == goal:
                    return step1
                visited.add(nb)
                if len(visited) < max_depth * 30:  # limit search
                    queue.append((nb, step1))

        # No path to goal — try ANY free neighbor (escape deadlock)
        neighbors = self._engine._directed_neighbors(cur)
        cur_dist = self._engine.distance(cur, goal)
        candidates = sorted(neighbors, key=lambda n: self._engine.distance(n, goal))
        for nb in candidates:
            if not self._is_blocked(bot_id, nb, new_pos, old_pos):
                return nb

        return None

    def _is_blocked(
        self,
        bot_id: int,
        target: Pos,
        new_pos: dict[int, Pos],
        old_pos: dict[int, Pos],
    ) -> bool:
        """Check if target cell is blocked for bot_id in sequential execution.

        Spawn position allows stacking (multiple bots can be there).
        """
        if target == self._spawn_pos:
            return False  # Spawn allows stacking
        for other_id in range(self._n_bots):
            if other_id == bot_id:
                continue
            if other_id < bot_id:
                # Already processed — check NEW position
                if new_pos.get(other_id) == target:
                    return True
            else:
                # Not yet processed — check OLD position
                if old_pos.get(other_id) == target:
                    return True
        return False


# ---------------------------------------------------------------------------
# Verification (sequential simulation)
# ---------------------------------------------------------------------------

def verify_sequential(plan: MAPFPlan, grid: Grid, spawn: Pos | None = None,
                      spawn_stack: bool = True) -> list[str]:
    """Verify plan by simulating sequential execution.

    Returns list of blocked moves (should be empty for plans from SequentialMAPFPlanner).
    Each action[t] is the action sent at round t. action[t].position is the bot's
    position at the START of round t (before the action).
    """
    bot_ids = sorted(plan.actions.keys())
    errors: list[str] = []

    # All bots start at spawn (or their first action's position)
    actual_pos: dict[int, Pos] = {}
    for bid in bot_ids:
        actual_pos[bid] = plan.actions[bid][0].position if plan.actions[bid] else spawn

    # Process every round starting from round 0
    for t in range(plan.total_rounds):
        new_pos: dict[int, Pos] = {}

        for bid in bot_ids:
            if t >= len(plan.actions[bid]):
                new_pos[bid] = actual_pos[bid]
                continue

            planned = plan.actions[bid][t]
            cur = actual_pos[bid]

            if planned.action in DELTAS:
                dx, dy = DELTAS[planned.action]
                target = (cur[0] + dx, cur[1] + dy)

                blocked = False
                if not (spawn_stack and spawn and target == spawn):
                    for other in bot_ids:
                        if other == bid:
                            continue
                        check_pos = new_pos[other] if other < bid else actual_pos[other]
                        if check_pos == target:
                            blocked = True
                            break

                if blocked:
                    new_pos[bid] = cur
                    errors.append(f"t={t} bot {bid} blocked at {target} from {cur}")
                else:
                    new_pos[bid] = target
            else:
                new_pos[bid] = cur

        actual_pos = new_pos

    return errors


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def plan_to_dict(plan: MAPFPlan) -> dict:
    actions_dict = {}
    for bot_id, actions in plan.actions.items():
        actions_dict[str(bot_id)] = [
            {"action": a.action, "position": list(a.position), "item_type": a.item_type}
            for a in actions
        ]
    return {
        "total_rounds": plan.total_rounds,
        "expected_score": plan.expected_score,
        "order_activations": {str(k): v for k, v in plan.order_activations.items()},
        "pickup_schedule": [
            {**s, "position": list(s["position"])} for s in plan.pickup_schedule
        ],
        "dropoff_schedule": [
            {**s, "position": list(s["position"])} for s in plan.dropoff_schedule
        ],
        "actions": actions_dict,
    }


def plan_from_dict(data: dict) -> MAPFPlan:
    actions = {}
    for bid, alist in data["actions"].items():
        actions[int(bid)] = [
            MAPFAction(action=a["action"], position=tuple(a["position"]),
                       item_type=a.get("item_type", ""))
            for a in alist
        ]
    return MAPFPlan(
        actions=actions, total_rounds=data["total_rounds"],
        expected_score=data["expected_score"],
        order_activations={int(k): v for k, v in data["order_activations"].items()},
        pickup_schedule=data.get("pickup_schedule", []),
        dropoff_schedule=data.get("dropoff_schedule", []),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: py mapf_planner.py <recon_file> [--save <plan_file>] [--method tsa|pibt|reactive]")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    recon_path = sys.argv[1]
    save_path = None
    method = "tsa"
    if "--save" in sys.argv:
        idx = sys.argv.index("--save")
        if idx + 1 < len(sys.argv):
            save_path = sys.argv[idx + 1]
    if "--method" in sys.argv:
        idx = sys.argv.index("--method")
        if idx + 1 < len(sys.argv):
            method = sys.argv[idx + 1]

    recon = load_recon(recon_path)
    planner = SequentialMAPFPlanner(recon)
    plan = planner.plan(method=method)

    # Verify (should be 0 conflicts by construction)
    errors = verify_sequential(plan, planner._grid, spawn=planner._spawn_pos)
    if errors:
        print(f"\nSequential verification: {len(errors)} blocked moves")
        for e in errors[:20]:
            print(f"  {e}")
    else:
        print("Sequential verification PASSED! (0 blocked moves)")

    if save_path:
        data = plan_to_dict(plan)
        Path(save_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Plan saved to {save_path}")


if __name__ == "__main__":
    main()
