"""
Parameterized bot strategy for optimizer.

Every decision is controlled by numeric parameters that can be tuned.
The optimizer mutates these parameters and measures score impact.

Usage:
    params = StrategyParams()  # defaults
    strategy = ParameterizedStrategy(params, grid)
    score = simulator.run(strategy)
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

Pos = tuple[int, int]


@dataclass
class StrategyParams:
    """
    All tunable parameters. Each affects a specific decision.
    Optimizer mutates these to find optimal combination.
    """
    # --- Delivery timing ---
    # Minimum items in inventory before delivering (1-3)
    min_items_to_deliver: int = 1
    # Always deliver if this many items match active order
    force_deliver_matching: int = 2
    # Deliver with fewer items if distance to drop-off < this
    deliver_if_close: int = 4

    # --- Cross-order pipelining ---
    # Start pre-picking preview items when active has ≤ this many remaining
    cross_order_threshold: int = 0
    # Max preview items to pre-pick (0 = disabled)
    cross_order_max: int = 1

    # --- Item selection ---
    # Weight: distance to item (higher = prefer closer items)
    w_distance: float = 1.0
    # Weight: bonus for items that complete an order (higher = prefer completing)
    w_completion: float = 5.0
    # Weight: prefer items near other needed items (cluster bonus)
    w_cluster: float = 0.5
    # Weight: prefer items closer to drop-off (saves return trip)
    w_dropoff_proximity: float = 0.3

    # --- Endgame ---
    # Switch to greedy delivery when ≤ this many rounds remain
    endgame_threshold: int = 30
    # In endgame, deliver with ANY matching items (ignore min_items_to_deliver)
    endgame_greedy: bool = True

    # --- Multi-bot (Medium+) ---
    # Zone stickiness: penalty for picking items outside own zone
    zone_penalty: float = 3.0
    # Dedicated deliverer: bot 0 prioritizes delivery
    use_dedicated_deliverer: bool = False
    # Drop-off staging distance
    staging_distance: int = 2

    def mutate(self, temperature: float = 1.0) -> "StrategyParams":
        """Create a mutated copy. Higher temperature = larger changes."""
        import random
        p = StrategyParams(**self.__dict__)

        # Pick 1-3 parameters to mutate
        mutations = random.randint(1, 3)
        fields = [
            ("min_items_to_deliver", 1, 3, "int"),
            ("force_deliver_matching", 1, 3, "int"),
            ("deliver_if_close", 1, 8, "int"),
            ("cross_order_threshold", 0, 3, "int"),
            ("cross_order_max", 0, 3, "int"),
            ("w_distance", 0.1, 3.0, "float"),
            ("w_completion", 0.0, 15.0, "float"),
            ("w_cluster", 0.0, 3.0, "float"),
            ("w_dropoff_proximity", 0.0, 2.0, "float"),
            ("endgame_threshold", 10, 60, "int"),
            ("endgame_greedy", 0, 1, "bool"),
            ("zone_penalty", 0.0, 10.0, "float"),
            ("staging_distance", 1, 4, "int"),
        ]

        for _ in range(mutations):
            name, lo, hi, dtype = random.choice(fields)
            if dtype == "int":
                delta = random.choice([-1, 0, 1])
                val = max(lo, min(hi, getattr(p, name) + delta))
                setattr(p, name, val)
            elif dtype == "float":
                delta = random.gauss(0, 0.3 * temperature)
                val = max(lo, min(hi, getattr(p, name) + delta))
                setattr(p, name, round(val, 2))
            elif dtype == "bool":
                if random.random() < 0.3:
                    setattr(p, name, not getattr(p, name))

        return p


class ParameterizedStrategy:
    """
    Bot strategy where every decision is parameter-driven.
    Implements BFS pathfinding internally (no external deps).
    
    Pass shared_bfs_cache to reuse BFS across optimizer iterations.
    """

    def __init__(self, params: StrategyParams, width: int, height: int,
                 walls: set[Pos], shelves: set[Pos],
                 shared_bfs_cache: dict[Pos, dict[Pos, int]] = None):
        self.p = params
        self.width = width
        self.height = height
        self.blocked = walls | shelves
        self.shelves = shelves
        self._bfs_cache: dict[Pos, dict[Pos, int]] = shared_bfs_cache if shared_bfs_cache is not None else {}
        self._adj_cache: dict[Pos, list[Pos]] = {}

    def precompute_bfs(self):
        """Precompute BFS from every walkable cell. Call once, reuse across iterations."""
        for x in range(self.width):
            for y in range(self.height):
                p = (x, y)
                if p not in self.blocked:
                    if p not in self._bfs_cache:
                        self._bfs_cache[p] = self._bfs_from(p)

    def __call__(self, state: dict) -> dict:
        """Strategy function compatible with Simulator.run() via dict."""
        bots = state["bots"]
        items = state["items"]
        orders = state["orders"]
        drop_off = tuple(state["drop_off"])
        current_round = state["round"]
        max_rounds = state["max_rounds"]
        rounds_left = max_rounds - current_round

        active = next((o for o in orders if o["status"] == "active"), None)
        preview = next((o for o in orders if o["status"] == "preview"), None)

        actions = []
        claimed_items: set[str] = set()

        for bot_data in sorted(bots, key=lambda b: b["id"]):
            action = self._decide_bot(
                bot_data, items, active, preview, drop_off,
                rounds_left, claimed_items, bots,
            )
            actions.append(action)

        return {"actions": actions}

    def decide_fast(self, sim) -> list[dict]:
        """
        Fast path: read directly from simulator internals.
        Avoids dict serialization entirely.
        """
        bots = sim._bots
        items = sim._items
        orders = sim._orders
        drop_off = sim.drop_off
        rounds_left = sim.max_rounds - sim._round

        active = None
        preview = None
        for o in orders:
            if o.status == "active":
                active = o
            elif o.status == "preview":
                preview = o

        actions = []
        claimed: set[str] = set()

        for bot in sorted(bots, key=lambda b: b.id):
            action = self._decide_bot_fast(
                bot, items, active, preview, drop_off,
                rounds_left, claimed,
            )
            actions.append(action)

        return actions

    def _decide_bot_fast(self, bot, items, active, preview,
                         drop_off, rounds_left, claimed) -> dict:
        """Fast decision using SimBot/SimItem/SimOrder objects directly."""
        bid = bot.id
        bpos = bot.position
        inv = bot.inventory

        if not active:
            return {"bot": bid, "action": "wait"}

        # Active order remaining items
        remaining = list(active.items_remaining)
        matching = [i for i in inv if i in remaining]

        # Endgame override
        if rounds_left <= self.p.endgame_threshold and self.p.endgame_greedy:
            if matching and bpos == drop_off:
                return {"bot": bid, "action": "drop_off"}
            if matching:
                return self._navigate(bid, bpos, drop_off)

        # Drop off if on cell with matching
        if bpos == drop_off and matching:
            return {"bot": bid, "action": "drop_off"}

        # Should deliver?
        if self._should_deliver_fast(bpos, inv, matching, remaining,
                                     items, drop_off, rounds_left, claimed):
            return self._navigate(bid, bpos, drop_off)

        # Pickup if adjacent
        pickup = self._try_pickup_fast(bid, bpos, items, remaining,
                                       preview, inv, claimed)
        if pickup:
            return pickup

        # Navigate to best item
        target = self._select_item_fast(bpos, items, remaining,
                                        preview, inv, drop_off, claimed)
        if target:
            claimed.add(target.id)
            adj = self._best_adjacent(bpos, target.position)
            if adj:
                return self._navigate(bid, bpos, adj)

        # Fallback: deliver if anything matches
        if matching:
            return self._navigate(bid, bpos, drop_off)

        return {"bot": bid, "action": "wait"}

    def _should_deliver_fast(self, bpos, inv, matching, remaining,
                             items, drop_off, rounds_left, claimed) -> bool:
        if not matching:
            return False
        if len(matching) >= self.p.force_deliver_matching:
            return True
        # Completes order?
        rem = list(remaining)
        for m in matching:
            if m in rem: rem.remove(m)
        if not rem:
            return True
        if len(inv) >= 3:
            return True
        if self._dist(bpos, drop_off) <= self.p.deliver_if_close and matching:
            return True
        if len(matching) >= self.p.min_items_to_deliver:
            for item in items:
                if item.picked or item.id in claimed:
                    continue
                if item.item_type in remaining:
                    # Check nearby using Manhattan (avoid _dist for speed)
                    d = abs(bpos[0] - item.position[0]) + abs(bpos[1] - item.position[1])
                    if d <= 5:
                        return False
            return True
        if rounds_left <= self.p.endgame_threshold:
            return True
        return False

    def _try_pickup_fast(self, bid, bpos, items, remaining,
                         preview, inv, claimed) -> Optional[dict]:
        if len(inv) >= 3:
            return None
        want = set(remaining)
        if (preview and len(remaining) <= self.p.cross_order_threshold
                and self.p.cross_order_max > 0):
            prev_rem = list(preview.items_remaining)
            already = sum(1 for i in inv if i in prev_rem)
            if already < self.p.cross_order_max:
                want.update(prev_rem)

        for item in items:
            if item.picked or item.id in claimed:
                continue
            ipos = item.position
            if abs(bpos[0] - ipos[0]) + abs(bpos[1] - ipos[1]) == 1:
                if item.item_type in want:
                    claimed.add(item.id)
                    return {"bot": bid, "action": "pick_up", "item_id": item.id}
        return None

    def _select_item_fast(self, bpos, items, remaining, preview,
                          inv, drop_off, claimed) -> Optional[object]:
        """Fast item selection — only scores items of wanted types."""
        want = set(remaining)
        preview_types = set()
        if (preview and len(remaining) <= self.p.cross_order_threshold
                and self.p.cross_order_max > 0):
            prev_rem = list(preview.items_remaining)
            already = sum(1 for i in inv if i in prev_rem)
            if already < self.p.cross_order_max:
                preview_types = set(prev_rem)

        all_wanted = want | preview_types
        if not all_wanted or len(inv) >= 3:
            return None

        best_score = -9999.0
        best_item = None

        for item in items:
            if item.picked or item.id in claimed:
                continue
            if item.item_type not in all_wanted:
                continue

            ipos = item.position
            adj = self._best_adjacent(bpos, ipos)
            if not adj:
                continue

            d = self._dist(bpos, adj)
            if d >= 9999:
                continue

            score = -self.p.w_distance * d
            if item.item_type in want:
                rem_after = len(remaining) - 1
                score += self.p.w_completion * (2.0 if rem_after == 0 else 1.0)
            else:
                score += self.p.w_completion * 0.3

            # Skip cluster calc for speed (minor impact)
            d_drop = self._dist(adj, drop_off)
            if d_drop < 9999:
                score -= self.p.w_dropoff_proximity * d_drop

            if score > best_score:
                best_score = score
                best_item = item

        return best_item

    def _decide_bot(self, bot: dict, items: list, active: Optional[dict],
                    preview: Optional[dict], drop_off: Pos,
                    rounds_left: int, claimed: set[str],
                    all_bots: list) -> dict:
        bid = bot["id"]
        bpos = tuple(bot["position"])
        inv = bot["inventory"]

        if not active:
            return {"bot": bid, "action": "wait"}

        # What does active order still need?
        remaining = list(active["items_required"])
        for d in active["items_delivered"]:
            if d in remaining:
                remaining.remove(d)

        # What in inventory matches active?
        matching = [i for i in inv if i in remaining]
        non_matching = [i for i in inv if i not in remaining]

        # --- Endgame override ---
        if rounds_left <= self.p.endgame_threshold and self.p.endgame_greedy:
            if matching and bpos == drop_off:
                return {"bot": bid, "action": "drop_off"}
            if matching:
                return self._navigate(bid, bpos, drop_off)

        # --- Drop off if on drop-off and have matching items ---
        if bpos == drop_off and matching:
            return {"bot": bid, "action": "drop_off"}

        # --- Should we deliver now? ---
        if self._should_deliver(bpos, inv, matching, remaining,
                                items, drop_off, rounds_left, claimed):
            return self._navigate(bid, bpos, drop_off)

        # --- Pick up if adjacent to a needed item ---
        pickup = self._try_pickup(bid, bpos, items, remaining,
                                  preview, inv, claimed)
        if pickup:
            return pickup

        # --- Navigate to best item ---
        target_item = self._select_item(bpos, items, remaining,
                                        preview, inv, drop_off,
                                        claimed, active)
        if target_item:
            claimed.add(target_item["id"])
            ipos = tuple(target_item["position"])
            # Navigate to adjacent cell
            adj = self._best_adjacent(bpos, ipos)
            if adj:
                return self._navigate(bid, bpos, adj)

        # --- If have ANY matching items, deliver ---
        if matching:
            return self._navigate(bid, bpos, drop_off)

        return {"bot": bid, "action": "wait"}

    def _should_deliver(self, bpos, inv, matching, remaining,
                        items, drop_off, rounds_left, claimed) -> bool:
        """Decide if bot should go deliver now."""
        if not matching:
            return False

        # Always deliver if enough matching items
        if len(matching) >= self.p.force_deliver_matching:
            return True

        # Delivery completes the order → always deliver
        remaining_after = list(remaining)
        for m in matching:
            if m in remaining_after:
                remaining_after.remove(m)
        if not remaining_after:
            return True

        # Full inventory
        if len(inv) >= 3:
            return True

        # Close to drop-off and have items
        d_drop = self._dist(bpos, drop_off)
        if d_drop <= self.p.deliver_if_close and matching:
            return True

        # Enough items based on threshold
        if len(matching) >= self.p.min_items_to_deliver:
            # But check if more items are nearby
            for item in items:
                if item["id"] in claimed:
                    continue
                if item["type"] in remaining:
                    d = self._dist(bpos, tuple(item["position"]))
                    if d <= 4:  # nearby
                        return False  # keep picking
            return True

        # Endgame
        if rounds_left <= self.p.endgame_threshold:
            return True

        return False

    def _try_pickup(self, bid, bpos, items, remaining,
                    preview, inv, claimed) -> Optional[dict]:
        """If adjacent to a needed item, pick it up."""
        if len(inv) >= 3:
            return None

        # What types do we want?
        want_types = set(remaining)

        # Cross-order: also want preview items if threshold met
        if (preview and len(remaining) <= self.p.cross_order_threshold
                and self.p.cross_order_max > 0):
            preview_remaining = list(preview["items_required"])
            for d in preview.get("items_delivered", []):
                if d in preview_remaining:
                    preview_remaining.remove(d)
            already_in_inv = sum(1 for i in inv if i in preview_remaining)
            if already_in_inv < self.p.cross_order_max:
                want_types.update(preview_remaining)

        for item in items:
            if item["id"] in claimed:
                continue
            ipos = tuple(item["position"])
            manhattan = abs(bpos[0] - ipos[0]) + abs(bpos[1] - ipos[1])
            if manhattan == 1 and item["type"] in want_types:
                claimed.add(item["id"])
                return {"bot": bid, "action": "pick_up", "item_id": item["id"]}

        return None

    def _select_item(self, bpos, items, remaining, preview,
                     inv, drop_off, claimed, active) -> Optional[dict]:
        """Select best item to go pick up based on weighted scoring."""
        want_types = set(remaining)
        preview_types = set()

        # Cross-order items
        if (preview and len(remaining) <= self.p.cross_order_threshold
                and self.p.cross_order_max > 0):
            preview_remaining = list(preview["items_required"])
            for d in preview.get("items_delivered", []):
                if d in preview_remaining:
                    preview_remaining.remove(d)
            already = sum(1 for i in inv if i in preview_remaining)
            if already < self.p.cross_order_max:
                preview_types = set(preview_remaining)

        all_wanted = want_types | preview_types
        if not all_wanted:
            return None

        best_score = -999
        best_item = None

        for item in items:
            if item["id"] in claimed:
                continue
            if item["type"] not in all_wanted:
                continue
            if len(inv) >= 3:
                continue

            ipos = tuple(item["position"])
            adj = self._best_adjacent(bpos, ipos)
            if not adj:
                continue

            d = self._dist(bpos, adj)
            if d >= 9999:
                continue

            # Score components
            score = 0.0

            # Distance (negative = closer is better)
            score -= self.p.w_distance * d

            # Completion bonus: does this item help complete the order?
            if item["type"] in want_types:
                # How many items remain including this one?
                rem_after = len(remaining) - 1  # -1 for this item
                if rem_after == 0:
                    score += self.p.w_completion * 2  # completes order!
                elif rem_after <= 2:
                    score += self.p.w_completion
            else:
                # Preview item — worth less
                score += self.p.w_completion * 0.3

            # Cluster bonus: other needed items nearby?
            for other in items:
                if other["id"] == item["id"] or other["id"] in claimed:
                    continue
                if other["type"] in want_types:
                    od = abs(ipos[0] - other["position"][0]) + abs(ipos[1] - other["position"][1])
                    if od <= 4:
                        score += self.p.w_cluster

            # Drop-off proximity
            d_drop = self._dist(adj, drop_off)
            if d_drop < 9999:
                score -= self.p.w_dropoff_proximity * d_drop

            if score > best_score:
                best_score = score
                best_item = item

        return best_item

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _navigate(self, bid: int, fr: Pos, to: Pos) -> dict:
        """Generate move action toward target using BFS distance gradient. O(1)."""
        if fr == to:
            return {"bot": bid, "action": "wait"}

        # Use BFS distance map: step to neighbor with lowest dist to target
        best_dist = self._dist(fr, to)
        best_action = "wait"

        for action, (dx, dy) in [("move_up", (0, -1)), ("move_down", (0, 1)),
                                   ("move_left", (-1, 0)), ("move_right", (1, 0))]:
            n = (fr[0] + dx, fr[1] + dy)
            if not self._walkable(n):
                continue
            d = self._dist(n, to)
            if d < best_dist:
                best_dist = d
                best_action = action

        return {"bot": bid, "action": best_action}

    def _best_adjacent(self, bot_pos: Pos, shelf_pos: Pos) -> Optional[Pos]:
        """Best walkable cell adjacent to shelf for pickup. Candidates cached."""
        if shelf_pos not in self._adj_cache:
            candidates = []
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                n = (shelf_pos[0] + dx, shelf_pos[1] + dy)
                if self._walkable(n):
                    candidates.append(n)
            self._adj_cache[shelf_pos] = candidates
        
        candidates = self._adj_cache[shelf_pos]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        return min(candidates, key=lambda c: self._dist(bot_pos, c))

    def _walkable(self, p: Pos) -> bool:
        return (0 <= p[0] < self.width and 0 <= p[1] < self.height
                and p not in self.blocked)

    def _dist(self, a: Pos, b: Pos) -> int:
        """BFS distance, cached."""
        if b not in self._bfs_cache:
            self._bfs_cache[b] = self._bfs_from(b)
        return self._bfs_cache[b].get(a, 9999)

    def _bfs_from(self, start: Pos) -> dict[Pos, int]:
        dist = {start: 0}
        q = deque([start])
        while q:
            pos = q.popleft()
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                n = (pos[0] + dx, pos[1] + dy)
                if n not in dist and self._walkable(n):
                    dist[n] = dist[pos] + 1
                    q.append(n)
        return dist

    def _bfs_path(self, start: Pos, end: Pos) -> list[Pos]:
        if start == end:
            return []
        prev = {start: None}
        q = deque([start])
        while q:
            pos = q.popleft()
            if pos == end:
                path = []
                cur = end
                while cur != start:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                n = (pos[0] + dx, pos[1] + dy)
                if n not in prev and self._walkable(n):
                    prev[n] = pos
                    q.append(n)
        return []
