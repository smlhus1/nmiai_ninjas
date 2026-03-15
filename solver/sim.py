"""Reactive simulator with PIBT and pre-picking.

Bots that finish active order items can pre-pick next order items.
When active order completes, pre-picked bots go straight to delivery.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum

from .grid import GameMap, Pos
from .pathfinding import DistanceCache
from .orders import Order, ShelfIndex
from .trips import PickupStep, _best_permutation


class Action(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    PICK_UP = "pick_up"
    DROP_OFF = "drop_off"
    WAIT = "wait"


PRIORITY_DELIVER = 0
PRIORITY_PICKUP = 1
PRIORITY_PREPICK = 2
PRIORITY_DISPERSE = 3
PRIORITY_IDLE = 4


@dataclass
class SimBot:
    bot_id: int
    pos: Pos
    inventory: list[str] = field(default_factory=list)
    goal: str = "idle"
    path: list[Pos] = field(default_factory=list)
    pending_pickup_item: str | None = None
    remaining_steps: list[PickupStep] = field(default_factory=list)
    trip_dropoff: Pos | None = None
    stuck_rounds: int = 0
    order_idx: int = -1  # which order this bot is working on (-1 = none)

    @property
    def priority(self) -> int:
        if self.goal in ("move_to_dropoff", "do_dropoff"):
            return PRIORITY_DELIVER
        elif self.goal in ("move_to_pickup", "do_pickup"):
            if self.order_idx >= 0:
                return PRIORITY_PICKUP
            return PRIORITY_PREPICK
        elif self.goal == "disperse":
            return PRIORITY_DISPERSE
        return PRIORITY_IDLE


@dataclass
class SimResult:
    score: int
    items_delivered: int
    orders_completed: int
    rounds_used: int
    collisions: int


class ReactiveSim:
    def __init__(
        self,
        game_map: GameMap,
        dist_cache: DistanceCache,
        shelf_index: ShelfIndex,
        orders: list[Order],
        max_rounds: int = 500,
    ) -> None:
        self.game_map = game_map
        self.grid = game_map.grid
        self.dist_cache = dist_cache
        self.shelf_index = shelf_index
        self.orders = orders
        self.max_rounds = max_rounds

    def run(self, max_items_per_trip: int = 1, verbose: bool = False) -> SimResult:
        bots = [
            SimBot(bot_id=i, pos=self.game_map.spawn)
            for i in range(self.game_map.bot_count)
        ]

        active_idx = 0
        items_remaining = Counter(self.orders[0].items_as_counter()) if self.orders else Counter()
        # Track claims per order
        active_claimed: Counter[str] = Counter()
        preview_claimed: Counter[str] = Counter()

        score = 0
        items_delivered = 0
        orders_completed = 0
        collisions = 0

        for round_num in range(self.max_rounds):
            if active_idx >= len(self.orders):
                break

            # Assign trips to idle bots
            for bot in bots:
                if bot.goal in ("idle", "disperse"):
                    # Try active order first
                    assigned = self._assign_from_order(
                        bot, active_idx, items_remaining, active_claimed,
                        max_items_per_trip, is_active=True,
                    )
                    # If no work on active, try preview orders (pre-pick)
                    if not assigned:
                        for lookahead in range(1, 3):  # preview + preview+1
                            pidx = active_idx + lookahead
                            if pidx < len(self.orders):
                                preview_remaining = self.orders[pidx].items_as_counter()
                                assigned = self._assign_from_order(
                                    bot, pidx, preview_remaining, preview_claimed,
                                    max_items_per_trip, is_active=False,
                                )
                                if assigned:
                                    break
                    # If still idle and at spawn, disperse
                    if not assigned and bot.goal == "idle":
                        self._disperse_bot(bot, bots)

            # PIBT movement
            new_positions = self._resolve_pibt(bots)

            # Apply actions
            for bot in bots:
                new_pos = new_positions[bot.bot_id]

                if bot.goal == "do_pickup" and new_pos == bot.pos:
                    if bot.pending_pickup_item:
                        bot.inventory.append(bot.pending_pickup_item)
                        if verbose:
                            print(f"R{round_num:3d}: Bot {bot.bot_id} PICKUP {bot.pending_pickup_item} "
                                  f"(order={bot.order_idx})")
                        bot.pending_pickup_item = None

                        if bot.remaining_steps:
                            step = bot.remaining_steps.pop(0)
                            bot.pending_pickup_item = step.item_type
                            bot.path = self._path(bot.pos, step.pickup_pos)
                            bot.goal = "move_to_pickup" if bot.path else "do_pickup"
                        else:
                            # Pickup done — go to dropoff only if this is active order
                            if bot.order_idx == active_idx:
                                bot.path = self._path(bot.pos, bot.trip_dropoff)
                                bot.goal = "move_to_dropoff" if bot.path else "do_dropoff"
                            else:
                                # Pre-pick done — wait with items until order becomes active
                                bot.goal = "idle"

                elif bot.goal == "do_dropoff" and new_pos == bot.pos:
                    if bot.inventory and active_idx < len(self.orders):
                        to_keep = []
                        delivered = 0
                        for item in bot.inventory:
                            if items_remaining[item] > 0:
                                items_remaining[item] -= 1
                                delivered += 1
                            else:
                                to_keep.append(item)

                        items_delivered += delivered
                        score += delivered
                        bot.inventory = to_keep

                        if verbose and delivered:
                            print(f"R{round_num:3d}: Bot {bot.bot_id} DELIVER {delivered}, score={score}")

                        if items_remaining.total() == 0:
                            score += 5
                            orders_completed += 1
                            active_idx += 1
                            if verbose:
                                print(f"R{round_num:3d}: ORDER {active_idx-1} COMPLETE! score={score}")

                            if active_idx < len(self.orders):
                                items_remaining = Counter(self.orders[active_idx].items_as_counter())

                                # Auto-delivery for this bot
                                ak = []
                                ad = 0
                                for item in bot.inventory:
                                    if items_remaining[item] > 0:
                                        items_remaining[item] -= 1
                                        ad += 1
                                    else:
                                        ak.append(item)
                                if ad:
                                    items_delivered += ad
                                    score += ad
                                    bot.inventory = ak
                                    if verbose:
                                        print(f"R{round_num:3d}: Bot {bot.bot_id} AUTO {ad}, score={score}")
                                    if items_remaining.total() == 0:
                                        score += 5
                                        orders_completed += 1
                                        active_idx += 1
                                        if active_idx < len(self.orders):
                                            items_remaining = Counter(self.orders[active_idx].items_as_counter())

                                # Reset claimed for new active order
                                active_claimed = Counter(preview_claimed)
                                preview_claimed.clear()

                                # Redirect pre-pick bots: those with preview items can now deliver
                                for b in bots:
                                    if b.order_idx == active_idx and b.goal == "idle" and b.inventory:
                                        # Has items for now-active order — deliver
                                        has_match = any(items_remaining[item] > 0 for item in b.inventory)
                                        if has_match:
                                            dz = self._nearest_dz(b.pos)
                                            b.path = self._path(b.pos, dz)
                                            b.trip_dropoff = dz
                                            b.goal = "move_to_dropoff" if b.path else "do_dropoff"

                    bot.goal = "idle"
                    bot.order_idx = -1

                else:
                    old_pos = bot.pos
                    bot.pos = new_pos
                    if bot.path and bot.path[0] == new_pos:
                        bot.path.pop(0)

                    if new_pos == old_pos and bot.path:
                        bot.stuck_rounds += 1
                        if bot.stuck_rounds > 8:
                            target = bot.path[-1]
                            occupied = {b.pos for b in bots if b.bot_id != bot.bot_id}
                            new_path = self._path_avoiding(bot.pos, target, occupied)
                            if new_path:
                                bot.path = new_path
                            bot.stuck_rounds = 0
                    else:
                        bot.stuck_rounds = 0

                    if not bot.path:
                        if bot.goal == "move_to_pickup":
                            bot.goal = "do_pickup"
                        elif bot.goal == "move_to_dropoff":
                            bot.goal = "do_dropoff"
                        elif bot.goal == "disperse":
                            bot.goal = "idle"

        return SimResult(
            score=score,
            items_delivered=items_delivered,
            orders_completed=orders_completed,
            rounds_used=min(round_num + 1, self.max_rounds) if 'round_num' in dir() else 0,
            collisions=collisions,
        )

    def _assign_from_order(
        self,
        bot: SimBot,
        order_idx: int,
        items_remaining: Counter[str],
        claimed: Counter[str],
        max_items: int,
        is_active: bool,
    ) -> bool:
        """Try to assign a trip from the specified order. Returns True if assigned."""
        if order_idx >= len(self.orders):
            return False

        unclaimed: Counter[str] = Counter()
        for it, needed in items_remaining.items():
            already = claimed.get(it, 0)
            if needed > already:
                unclaimed[it] = needed - already

        if unclaimed.total() == 0:
            # All claimed — check if bot has deliverable items for this order
            if is_active and bot.inventory:
                has_match = any(items_remaining[item] > 0 for item in bot.inventory)
                if has_match:
                    dz = self._nearest_dz(bot.pos)
                    bot.path = self._path(bot.pos, dz)
                    bot.trip_dropoff = dz
                    bot.order_idx = order_idx
                    bot.goal = "move_to_dropoff" if bot.path else "do_dropoff"
                    return True
            return False

        # Find nearest items
        options: list[tuple[str, PickupStep, int]] = []
        for it in unclaimed:
            best_entry = None
            best_d = float("inf")
            for dz in self.game_map.drop_off_zones:
                entries = self.shelf_index.get(it, dz)
                if entries:
                    for entry in entries[:3]:
                        d = self.dist_cache.distance(bot.pos, entry.pickup_pos)
                        if d is not None and d < best_d:
                            best_d = d
                            best_entry = entry
            if best_entry:
                options.append((it, PickupStep(it, best_entry.shelf_pos, best_entry.pickup_pos), int(best_d)))

        options.sort(key=lambda x: x[2])
        steps: list[PickupStep] = []
        to_claim: list[str] = []
        seen: Counter[str] = Counter()

        for it, step, _ in options:
            if len(steps) >= max_items:
                break
            if seen[it] >= unclaimed[it]:
                continue
            steps.append(step)
            to_claim.append(it)
            seen[it] += 1

        if not steps:
            return False

        last_pickup = steps[-1].pickup_pos
        dz = self._nearest_dz(last_pickup)

        result = _best_permutation(bot.pos, steps, dz, self.dist_cache)
        if not result:
            return False

        ordered, _ = result

        for it in to_claim:
            claimed[it] += 1

        first = ordered[0]
        bot.pending_pickup_item = first.item_type
        bot.remaining_steps = list(ordered[1:])
        bot.trip_dropoff = dz
        bot.order_idx = order_idx
        bot.path = self._path(bot.pos, first.pickup_pos)
        bot.goal = "move_to_pickup" if bot.path else "do_pickup"
        return True

    def _resolve_pibt(self, bots: list[SimBot]) -> dict[int, Pos]:
        sorted_bots = sorted(bots, key=lambda b: (b.priority, b.bot_id))

        occupied: dict[Pos, int] = {}
        final: dict[int, Pos] = {}
        original: dict[int, Pos] = {b.bot_id: b.pos for b in bots}

        desired: dict[int, Pos] = {}
        for bot in bots:
            if bot.goal in ("do_pickup", "do_dropoff"):
                desired[bot.bot_id] = bot.pos
            elif bot.path:
                next_pos = bot.path[0]
                desired[bot.bot_id] = next_pos if self.grid.walkable(next_pos) else bot.pos
            else:
                desired[bot.bot_id] = bot.pos

        for bot in sorted_bots:
            pos = desired[bot.bot_id]

            if pos == bot.pos:
                final[bot.bot_id] = pos
                if pos not in occupied:
                    occupied[pos] = bot.bot_id
                continue

            if pos in occupied:
                blocker_id = occupied[pos]
                blocker = next(b for b in bots if b.bot_id == blocker_id)
                if (bot.priority, bot.bot_id) < (blocker.priority, blocker.bot_id):
                    alt = self._find_yield(blocker, occupied, final)
                    if alt and alt != pos:
                        final[blocker_id] = alt
                        occupied[alt] = blocker_id
                        if occupied.get(pos) == blocker_id:
                            del occupied[pos]
                        final[bot.bot_id] = pos
                        occupied[pos] = bot.bot_id
                        continue
                pos = bot.pos

            if pos != bot.pos:
                for oid, opos in final.items():
                    if pos == original[oid] and opos == bot.pos:
                        pos = bot.pos
                        break

            final[bot.bot_id] = pos
            if pos not in occupied:
                occupied[pos] = bot.bot_id

        return final

    def _find_yield(self, bot: SimBot, occupied: dict[Pos, int], final: dict[int, Pos]) -> Pos | None:
        for nb in self.grid.neighbors(bot.pos):
            if nb not in occupied and nb not in final.values():
                return nb
        return None

    def _disperse_bot(self, bot: SimBot, all_bots: list[SimBot]) -> None:
        """Move idle bot away from spawn to reduce congestion."""
        # Only disperse if at a crowded position
        bots_at_pos = sum(1 for b in all_bots if b.pos == bot.pos)
        if bots_at_pos <= 1:
            return

        # Pick a dispersal target — spread along corridors
        # Use bot_id to pick different targets
        walkable = self.grid.walkable_cells()
        # Filter to corridor positions (y=1, y=9, y=15)
        corridors = [(x, y) for x, y in walkable if y in (1, 9, 15)]
        if not corridors:
            corridors = walkable

        # Pick target based on bot_id for spread
        target = corridors[bot.bot_id % len(corridors)]
        bot.path = self._path(bot.pos, target)
        if bot.path:
            bot.goal = "disperse"

    def _nearest_dz(self, pos: Pos) -> Pos:
        best = None
        best_d = float("inf")
        for dz in self.game_map.drop_off_zones:
            d = self.dist_cache.distance(pos, dz)
            if d is not None and d < best_d:
                best_d = d
                best = dz
        return best or self.game_map.drop_off_zones[0]

    def _path(self, start: Pos, end: Pos) -> list[Pos]:
        if start == end:
            return []
        parent: dict[Pos, Pos] = {start: start}
        q = deque([start])
        while q:
            pos = q.popleft()
            if pos == end:
                path = []
                c = end
                while c != start:
                    path.append(c)
                    c = parent[c]
                path.reverse()
                return path
            for nb in self.grid.neighbors(pos):
                if nb not in parent:
                    parent[nb] = pos
                    q.append(nb)
        return []

    def _path_avoiding(self, start: Pos, end: Pos, avoid: set[Pos]) -> list[Pos]:
        if start == end:
            return []
        parent: dict[Pos, Pos] = {start: start}
        q = deque([start])
        while q:
            pos = q.popleft()
            if pos == end:
                path = []
                c = end
                while c != start:
                    path.append(c)
                    c = parent[c]
                path.reverse()
                return path
            for nb in self.grid.neighbors(pos):
                if nb not in parent and nb not in avoid:
                    parent[nb] = pos
                    q.append(nb)
        return self._path(start, end)
