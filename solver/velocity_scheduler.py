"""
Velocity scheduler for nightmare map (20 bots, 3 drop-off zones).

Assigns items to bots with zone partitioning and order pipelining.
Uses BFS distances from PathEngine for accurate travel time estimates.

Usage:
    py -m solver.velocity_scheduler --recon logs/74001e7f_2026-03-16_recon.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Optional

from bot.engine.pathfinding import PathEngine
from bot.models import Grid, Pos


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BotTask:
    bot_id: int
    item_type: str
    shelf_pos: tuple[int, int]       # the shelf cell (non-walkable)
    pickup_pos: tuple[int, int]      # walkable cell adjacent to shelf
    drop_off: tuple[int, int]
    order_index: int
    is_prepick: bool                 # True = picking for next order
    est_start: int = 0               # round bot starts moving
    est_pickup: int = 0              # round bot arrives at pickup
    est_deliver: int = 0             # round bot finishes dropoff

    def __repr__(self) -> str:
        tag = "PRE" if self.is_prepick else "ACT"
        return (f"BotTask(bot={self.bot_id}, {tag}, ord={self.order_index}, "
                f"{self.item_type}, pickup={self.pickup_pos}, drop={self.drop_off}, "
                f"r{self.est_start}->{self.est_deliver})")


@dataclass
class OrderPlan:
    order_index: int
    tasks: list[BotTask]
    estimated_start_round: int
    estimated_end_round: int
    items_count: int
    score: int = 0  # items + order bonus

    def __repr__(self) -> str:
        return (f"OrderPlan(ord={self.order_index}, items={self.items_count}, "
                f"rounds={self.estimated_start_round}-{self.estimated_end_round}, "
                f"score={self.score})")


# ---------------------------------------------------------------------------
# Zone definitions
# ---------------------------------------------------------------------------

ZONE_DROPOFFS: list[Pos] = [
    (1, 16),   # left
    (15, 16),  # center
    (27, 16),  # right
]

ZONE_X_RANGES = [
    (0, 9),    # left zone
    (10, 19),  # center zone
    (20, 29),  # right zone
]


def shelf_zone(shelf_pos: Pos) -> int:
    """Which zone a shelf belongs to by x coordinate."""
    x = shelf_pos[0]
    for i, (lo, hi) in enumerate(ZONE_X_RANGES):
        if lo <= x <= hi:
            return i
    return 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_pickup_positions(shelf_pos: Pos, grid: Grid) -> list[Pos]:
    """Walkable cells adjacent to a shelf."""
    x, y = shelf_pos
    out = []
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nx, ny = x + dx, y + dy
        if grid.is_walkable((nx, ny)):
            out.append((nx, ny))
    return out


@dataclass
class BotState:
    bot_id: int
    pos: Pos
    available_round: int = 0  # round when bot is free
    has_item: str | None = None  # item type if holding prepicked item


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class VelocityScheduler:
    """Schedule item assignments for 20 bots across 3 zones with order pipelining."""

    def __init__(self, recon: dict, drop_offs: list[Pos] | None = None):
        self.recon = recon
        self.grid_size = tuple(recon["grid_size"])

        # Drop-offs
        raw_dz = recon.get("drop_off_zones", [])
        self.drop_offs: list[Pos] = drop_offs or [tuple(z) for z in raw_dz] if raw_dz else ZONE_DROPOFFS[:]
        if not self.drop_offs or not isinstance(self.drop_offs[0], tuple):
            self.drop_offs = ZONE_DROPOFFS[:]

        # Grid with shelves as walls
        raw_walls = [tuple(w) for w in recon["walls"]]
        shelf_positions: set[Pos] = set()
        for positions in recon["shelf_map"].values():
            for p in positions:
                shelf_positions.add(tuple(p))
        all_walls = frozenset(raw_walls) | shelf_positions

        self.grid = Grid(width=self.grid_size[0], height=self.grid_size[1], walls=all_walls)

        # PathEngine
        self.engine = PathEngine()
        self.engine.enable_one_way(True)
        self.engine.set_grid(self.grid, drop_off=self.drop_offs[0])

        # Shelf map: item_type -> [(shelf_pos, [pickup_pos, ...])]
        self.shelf_map: dict[str, list[tuple[Pos, list[Pos]]]] = {}
        for item_type, positions in recon["shelf_map"].items():
            entries = []
            for p in positions:
                sp = tuple(p)
                pps = find_pickup_positions(sp, self.grid)
                if pps:
                    entries.append((sp, pps))
            self.shelf_map[item_type] = entries

        # Precompute: for each (item_type, dropoff) -> sorted list of
        # (pickup_pos, shelf_pos, dist_pickup_to_dropoff)
        self._item_dropoff_options: dict[tuple[str, Pos], list[tuple[Pos, Pos, int]]] = {}
        for item_type, entries in self.shelf_map.items():
            for dropoff in self.drop_offs:
                options = []
                for shelf_pos, pickups in entries:
                    for pp in pickups:
                        d = self.engine.distance(pp, dropoff)
                        options.append((pp, shelf_pos, d))
                options.sort(key=lambda x: x[2])
                self._item_dropoff_options[(item_type, dropoff)] = options

        # Orders
        self.orders = recon["order_sequence"]
        self.n_bots = recon.get("bot_count", 20)

        # Bot states
        spawn = tuple(recon.get("bot_start_positions", [[28, 16]])[0])
        self.bots = [BotState(bot_id=i, pos=spawn) for i in range(self.n_bots)]

    def _find_best_assignment(
        self,
        bot: BotState,
        item_type: str,
        current_round: int,
    ) -> tuple[Pos, Pos, Pos, int, int, int] | None:
        """Find the best (pickup, shelf, dropoff, start, pickup_round, deliver_round) for a bot+item.

        Tries all dropoffs and all shelf positions, returns the one with earliest delivery.
        """
        start_round = max(current_round, bot.available_round)
        best = None
        best_deliver = 999999

        for dropoff in self.drop_offs:
            options = self._item_dropoff_options.get((item_type, dropoff), [])
            for pickup_pos, shelf_pos, dist_to_drop in options:
                dist_to_pickup = self.engine.distance(bot.pos, pickup_pos)
                pickup_round = start_round + dist_to_pickup
                # +1 for pickup action, then travel to dropoff, +1 for dropoff action
                deliver_round = pickup_round + 1 + dist_to_drop + 1
                if deliver_round < best_deliver:
                    best_deliver = deliver_round
                    best = (pickup_pos, shelf_pos, dropoff, start_round, pickup_round, deliver_round)
                # Early exit: first option per dropoff is closest shelf, unlikely to beat
                # a much closer option, so check only top-3 per dropoff
                if len(options) > 3:
                    break

        return best

    def schedule(self) -> list[OrderPlan]:
        """Build the full schedule for all known orders."""
        plans: list[OrderPlan] = []
        # prepicked[order_idx] = [(bot_id, item_type, deliver_round_estimate)]
        prepicked: dict[int, list[tuple[int, str, int]]] = {}

        for order_idx, order in enumerate(self.orders):
            items_needed = list(order["items_required"])
            order_tasks: list[BotTask] = []

            # Determine earliest possible start: when the previous order finishes
            if plans:
                est_start = plans[-1].estimated_end_round
            else:
                est_start = 0

            # ----- Phase 1: Deliver prepicked items -----
            pre = prepicked.pop(order_idx, [])
            pre_by_type: dict[str, list[tuple[int, int]]] = {}
            for bot_id, itype, deliver_est in pre:
                pre_by_type.setdefault(itype, []).append((bot_id, deliver_est))

            remaining_items: list[str] = []
            for item_type in items_needed:
                if item_type in pre_by_type and pre_by_type[item_type]:
                    bot_id, old_deliver_est = pre_by_type[item_type].pop(0)
                    bot = self.bots[bot_id]
                    # Bot has item, just needs to deliver to nearest dropoff
                    best_drop = None
                    best_d = 999999
                    for dropoff in self.drop_offs:
                        d = self.engine.distance(bot.pos, dropoff)
                        if d < best_d:
                            best_d = d
                            best_drop = dropoff
                    deliver_round = max(est_start, bot.available_round) + best_d + 1
                    task = BotTask(
                        bot_id=bot_id,
                        item_type=item_type,
                        shelf_pos=bot.pos,
                        pickup_pos=bot.pos,
                        drop_off=best_drop,
                        order_index=order_idx,
                        is_prepick=False,
                        est_start=max(est_start, bot.available_round),
                        est_pickup=max(est_start, bot.available_round),
                        est_deliver=deliver_round,
                    )
                    order_tasks.append(task)
                    bot.available_round = deliver_round
                    bot.pos = best_drop
                    bot.has_item = None
                else:
                    remaining_items.append(item_type)

            # ----- Phase 2: Assign remaining items to bots -----
            # Greedy: for each item, find the bot that can deliver it soonest
            for item_type in remaining_items:
                best_bot_idx = -1
                best_assignment = None
                best_deliver = 999999

                for bi, bot in enumerate(self.bots):
                    result = self._find_best_assignment(bot, item_type, est_start)
                    if result and result[5] < best_deliver:
                        best_deliver = result[5]
                        best_bot_idx = bi
                        best_assignment = result

                if best_assignment:
                    pickup_pos, shelf_pos, dropoff, start, pickup_r, deliver_r = best_assignment
                    bot = self.bots[best_bot_idx]
                    task = BotTask(
                        bot_id=bot.bot_id,
                        item_type=item_type,
                        shelf_pos=shelf_pos,
                        pickup_pos=pickup_pos,
                        drop_off=dropoff,
                        order_index=order_idx,
                        is_prepick=False,
                        est_start=start,
                        est_pickup=pickup_r,
                        est_deliver=deliver_r,
                    )
                    order_tasks.append(task)
                    bot.available_round = deliver_r
                    bot.pos = dropoff

            # ----- Compute order completion -----
            active_tasks = [t for t in order_tasks if not t.is_prepick]
            if active_tasks:
                est_end = max(t.est_deliver for t in active_tasks)
            else:
                est_end = est_start + 15  # fallback

            items_count = len(items_needed)
            score = items_count + 5

            plan = OrderPlan(
                order_index=order_idx,
                tasks=order_tasks,
                estimated_start_round=est_start,
                estimated_end_round=est_end,
                items_count=items_count,
                score=score,
            )
            plans.append(plan)

            # ----- Phase 3: Pre-pick NEXT order's items with idle bots -----
            if order_idx + 1 < len(self.orders):
                next_order = self.orders[order_idx + 1]
                next_items = list(next_order["items_required"])

                # Bots that finish before this order completes can prepick
                prepicked_for_next: list[tuple[int, str, int]] = []

                for item_type in next_items:
                    best_bot_idx = -1
                    best_pickup_pos = None
                    best_shelf_pos = None
                    best_finish_pickup = 999999  # when bot finishes picking (before delivery)

                    for bi, bot in enumerate(self.bots):
                        # Bot must be free before order ends to have time to prepick
                        if bot.available_round > est_end:
                            continue
                        # Find closest shelf for pickup only (no delivery needed)
                        for item_entries in [self.shelf_map.get(item_type, [])]:
                            for shelf_pos, pickups in item_entries:
                                for pp in pickups:
                                    start = max(est_start, bot.available_round)
                                    dist = self.engine.distance(bot.pos, pp)
                                    finish = start + dist + 1  # +1 pickup action
                                    if finish <= est_end and finish < best_finish_pickup:
                                        best_finish_pickup = finish
                                        best_bot_idx = bi
                                        best_pickup_pos = pp
                                        best_shelf_pos = shelf_pos

                    if best_bot_idx >= 0:
                        bot = self.bots[best_bot_idx]
                        # Estimate delivery time for info
                        best_drop_dist = min(
                            self.engine.distance(best_pickup_pos, d) for d in self.drop_offs
                        )
                        est_deliver = best_finish_pickup + best_drop_dist + 1

                        task = BotTask(
                            bot_id=bot.bot_id,
                            item_type=item_type,
                            shelf_pos=best_shelf_pos,
                            pickup_pos=best_pickup_pos,
                            drop_off=min(self.drop_offs,
                                         key=lambda d: self.engine.distance(best_pickup_pos, d)),
                            order_index=order_idx + 1,
                            is_prepick=True,
                            est_start=max(est_start, bot.available_round),
                            est_pickup=best_finish_pickup,
                            est_deliver=est_deliver,
                        )
                        order_tasks.append(task)

                        bot.available_round = best_finish_pickup
                        bot.pos = best_pickup_pos
                        bot.has_item = item_type
                        prepicked_for_next.append((bot.bot_id, item_type, est_deliver))

                if prepicked_for_next:
                    prepicked[order_idx + 1] = prepicked_for_next

        return plans

    def print_summary(self, plans: list[OrderPlan], max_round: int = 180) -> None:
        """Print schedule summary and score estimate."""
        print("=" * 80)
        print(f"VELOCITY SCHEDULER -- {self.n_bots} bots, {len(self.drop_offs)} drop-offs")
        print(f"Orders: {len(self.orders)}, Total items: "
              f"{sum(len(o['items_required']) for o in self.orders)}")
        print("=" * 80)

        total_score = 0
        total_items = 0
        orders_in_time = 0
        prepick_count = 0

        for plan in plans:
            in_time = plan.estimated_end_round <= max_round
            marker = "OK" if in_time else "LATE"

            prepicks = sum(1 for t in plan.tasks if t.is_prepick)
            active_tasks = [t for t in plan.tasks if not t.is_prepick]

            # Zone distribution by dropoff
            zone_counts: dict[Pos, int] = {}
            for t in active_tasks:
                zone_counts[t.drop_off] = zone_counts.get(t.drop_off, 0) + 1
            zone_str = "/".join(
                str(zone_counts.get(d, 0)) for d in self.drop_offs
            )

            # Unique bots used
            bot_ids = set(t.bot_id for t in active_tasks)

            print(f"Order {plan.order_index:2d}: {plan.items_count} items, "
                  f"rounds {plan.estimated_start_round:3d}-{plan.estimated_end_round:3d} "
                  f"({plan.estimated_end_round - plan.estimated_start_round:2d}r), "
                  f"zones=[{zone_str}], bots={len(bot_ids)}, "
                  f"prepicks={prepicks}, [{marker}]")

            if in_time:
                total_score += plan.score
                total_items += plan.items_count
                orders_in_time += 1
            prepick_count += prepicks

        print("=" * 80)
        print(f"Orders completed by round {max_round}: {orders_in_time}/{len(plans)}")
        print(f"Items delivered: {total_items}")
        print(f"Estimated score at round {max_round}: {total_score}")
        print(f"Velocity: {total_score / max(max_round, 1):.2f} score/round")
        print(f"Total prepicks: {prepick_count}")
        print()

        # Full game estimate
        full_score = sum(p.score for p in plans)
        last_round = max((p.estimated_end_round for p in plans), default=0)
        print(f"Full game ({len(plans)} orders): {full_score} score in {last_round} rounds")
        if last_round > 0:
            print(f"Full game velocity: {full_score / last_round:.2f} score/round")

        # Show at various time horizons
        print()
        for horizon in [100, 180, 300, 500]:
            s = sum(p.score for p in plans if p.estimated_end_round <= horizon)
            n = sum(1 for p in plans if p.estimated_end_round <= horizon)
            print(f"  Round {horizon:3d}: {n:2d} orders, {s:3d} score, "
                  f"{s / max(horizon, 1):.2f}/round")

    def print_detailed_tasks(self, plans: list[OrderPlan], max_orders: int = 5) -> None:
        """Print detailed task breakdown for first N orders."""
        print()
        print(f"DETAILED TASK BREAKDOWN (first {max_orders} orders):")
        print("-" * 80)
        for plan in plans[:max_orders]:
            items = self.orders[plan.order_index]["items_required"]
            print(f"\nOrder {plan.order_index}: {items}")
            print(f"  Rounds: {plan.estimated_start_round} -> {plan.estimated_end_round}")
            for task in sorted(plan.tasks, key=lambda t: (t.is_prepick, t.est_deliver)):
                tag = "PRE-PICK" if task.is_prepick else "ACTIVE  "
                print(f"  [{tag}] Bot {task.bot_id:2d}: {task.item_type:10s} "
                      f"r{task.est_start:3d}->pickup@{task.pickup_pos} r{task.est_pickup:3d}"
                      f"->drop@{task.drop_off} r{task.est_deliver:3d}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Velocity scheduler for nightmare map")
    parser.add_argument("--recon", required=True, help="Path to recon JSON file")
    parser.add_argument("--max-round", type=int, default=180,
                        help="Target round for score estimation (default: 180)")
    parser.add_argument("--detailed", action="store_true",
                        help="Print detailed task breakdown")
    args = parser.parse_args()

    with open(args.recon) as f:
        recon = json.load(f)

    scheduler = VelocityScheduler(recon)
    plans = scheduler.schedule()

    scheduler.print_summary(plans, max_round=args.max_round)
    if args.detailed:
        scheduler.print_detailed_tasks(plans)


if __name__ == "__main__":
    main()
