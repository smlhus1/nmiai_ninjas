"""
Nightmare Pipeline Strategy: purpose-built 20-bot architecture.

Core systems:
1. Fast spawn dispersal (all bots spread immediately, no staggering)
2. Zone-based item assignment (6 zones, more bots near drop-off)
3. Drop-off queue management (max concurrent deliverers)
4. Claim system preventing over-assignment

Bot state machine:
    DISPERSING -> PICKING -> DELIVERING -> PICKING (loop)

Design: every bot is a picker-deliverer. Focus on preventing over-assignment
and ensuring smooth delivery flow.
"""
from __future__ import annotations

import copy
import random as rng
from collections import Counter, deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

Pos = tuple[int, int]

MOVES = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}
MOVE_LIST = list(MOVES.items())


# ---------------------------------------------------------------------------
# BFS utilities
# ---------------------------------------------------------------------------

def multi_source_bfs(w: int, h: int, blocked: frozenset, sources: list[Pos]) -> dict[Pos, int]:
    """BFS from multiple sources simultaneously. Returns {pos: min_distance_to_any_source}."""
    dist: dict[Pos, int] = {}
    q: deque[Pos] = deque()
    for s in sources:
        if s not in blocked and 0 <= s[0] < w and 0 <= s[1] < h:
            dist[s] = 0
            q.append(s)
    while q:
        pos = q.popleft()
        d = dist[pos]
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            n = (pos[0] + dx, pos[1] + dy)
            if 0 <= n[0] < w and 0 <= n[1] < h and n not in blocked and n not in dist:
                dist[n] = d + 1
                q.append(n)
    return dist


# ---------------------------------------------------------------------------
# Bot states
# ---------------------------------------------------------------------------

class BotState(IntEnum):
    DISPERSING = 0
    PICKING = 1
    DELIVERING = 2


# ---------------------------------------------------------------------------
# Parameters (tunable by optimizer)
# ---------------------------------------------------------------------------

@dataclass
class NightmareParams:
    # Zone allocation: number of bots per zone [zone0..zone5]
    zone_bots: tuple[int, ...] = (5, 4, 3, 3, 3, 2)

    # Delivery
    max_carry: int = 2          # deliver after picking this many matching items
    carry_if_close: int = 8     # deliver with 1+ item if within this dist
    max_deliverers: int = 1     # max concurrent bots heading to/at drop-off

    # Endgame
    endgame_rounds: int = 60

    # Item selection
    w_distance: float = 1.0
    w_dropoff: float = 0.3
    w_completion: float = 15.0
    zone_penalty: float = 0.0
    batch_bonus: float = 5.0

    # Exploration
    seed: int = 0

    def mutate(self, temperature: float = 1.0) -> NightmareParams:
        p = copy.copy(self)
        r = rng.random
        if r() < 0.3 * temperature:
            z = list(p.zone_bots)
            i = rng.randint(0, 5)
            j = rng.randint(0, 5)
            if i != j and z[i] > 1:
                z[i] -= 1
                z[j] += 1
            p.zone_bots = tuple(z)
        if r() < 0.3 * temperature:
            p.max_carry = rng.choice([1, 2, 3])
        if r() < 0.25 * temperature:
            p.carry_if_close = max(1, min(20, p.carry_if_close + rng.choice([-3, -1, 1, 3])))
        if r() < 0.25 * temperature:
            p.max_deliverers = max(1, min(8, p.max_deliverers + rng.choice([-1, 1])))
        if r() < 0.2 * temperature:
            p.endgame_rounds = max(20, min(100, p.endgame_rounds + rng.choice([-10, -5, 5, 10])))
        if r() < 0.25 * temperature:
            p.w_distance = max(0.1, p.w_distance + rng.uniform(-0.3, 0.3))
        if r() < 0.2 * temperature:
            p.w_dropoff = max(0, p.w_dropoff + rng.uniform(-0.2, 0.2))
        if r() < 0.2 * temperature:
            p.w_completion = max(0, p.w_completion + rng.uniform(-5, 5))
        if r() < 0.2 * temperature:
            p.zone_penalty = max(0, p.zone_penalty + rng.uniform(-5, 5))
        if r() < 0.2 * temperature:
            p.batch_bonus = max(0, p.batch_bonus + rng.uniform(-3, 3))
        return p

    @classmethod
    def random(cls) -> NightmareParams:
        z = [rng.randint(1, 6) for _ in range(6)]
        total = sum(z)
        z = [max(1, round(b * 20 / total)) for b in z]
        diff = sum(z) - 20
        for i in range(abs(diff)):
            idx = rng.randint(0, 5)
            z[idx] += 1 if diff < 0 else (-1 if z[idx] > 1 else 0)
        return cls(
            zone_bots=tuple(z),
            max_carry=rng.choice([1, 2, 3]),
            carry_if_close=rng.randint(3, 15),
            max_deliverers=rng.randint(1, 6),
            endgame_rounds=rng.randint(20, 80),
            w_distance=rng.uniform(0.3, 2.0),
            w_dropoff=rng.uniform(0, 1.0),
            w_completion=rng.uniform(5, 25),
            zone_penalty=rng.uniform(5, 30),
            batch_bonus=rng.uniform(0, 10),
            seed=rng.randint(0, 99999),
        )

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                d[k] = list(v) if isinstance(v, tuple) else v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> NightmareParams:
        valid = {f for f in cls.__dataclass_fields__}
        kw = {}
        for k, v in d.items():
            if k in valid:
                if k == 'zone_bots' and isinstance(v, list):
                    v = tuple(v)
                kw[k] = v
        return cls(**kw)


# ---------------------------------------------------------------------------
# Nightmare map geometry
# ---------------------------------------------------------------------------

# Shelf X values: 3,5,7,9,11,13,15,17,19,21,23,25
# Walkable vertical aisles: 1,4,8,12,16,20,24,27,28
# Corridors (fully open rows): y=1, y=9, y=15, y=16

ZONES = {
    0: (1, 5),    # shelves x=3,5
    1: (6, 9),    # shelves x=7,9
    2: (10, 13),  # shelves x=11,13
    3: (14, 17),  # shelves x=15,17
    4: (18, 21),  # shelves x=19,21
    5: (22, 28),  # shelves x=23,25
}


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class NightmareStrategy:
    """
    Pipeline strategy for nightmare (20-bot) optimization.

    Class-level BFS caches are shared across instances for the same grid.
    Per-instance state (bot states, claims) is reset each game.
    """

    # Class-level cache
    _cached_blocked: Optional[frozenset] = None
    _dist_to_dropoff: Optional[dict[Pos, int]] = None
    _dist_to_type: Optional[dict[str, dict[Pos, int]]] = None
    _shelf_adj: Optional[dict[Pos, list[Pos]]] = None
    _type_shelves: Optional[dict[str, list[Pos]]] = None
    _type_dropoff_dist: Optional[dict[str, int]] = None
    _zone_shelves: Optional[dict[int, set[Pos]]] = None
    _disperse_targets: Optional[list[Pos]] = None
    _disperse_dists: Optional[list[dict[Pos, int]]] = None
    _w: int = 0
    _h: int = 0

    def __init__(self, params: NightmareParams):
        self.p = params
        self._rng = rng.Random(params.seed)

    def decide_fast(self, sim) -> list[dict]:
        if sim._round == 0:
            self._init_game(sim)
        return self._decide(sim)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_game(self, sim):
        w, h = sim.width, sim.height
        blocked = sim.blocked

        if NightmareStrategy._cached_blocked != blocked:
            NightmareStrategy._cached_blocked = blocked
            NightmareStrategy._w = w
            NightmareStrategy._h = h

            # Shelf adjacency
            NightmareStrategy._shelf_adj = {}
            for shelf in sim.shelves:
                adj = []
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    n = (shelf[0] + dx, shelf[1] + dy)
                    if 0 <= n[0] < w and 0 <= n[1] < h and n not in blocked:
                        adj.append(n)
                NightmareStrategy._shelf_adj[shelf] = adj

            # Type -> shelves
            NightmareStrategy._type_shelves = {}
            for pos, itype in sim.shelf_types.items():
                NightmareStrategy._type_shelves.setdefault(itype, []).append(pos)

            # Zone -> shelves
            NightmareStrategy._zone_shelves = {}
            for zone_id, (x_min, x_max) in ZONES.items():
                zone_set: set[Pos] = set()
                for pos in sim.shelves:
                    if x_min <= pos[0] <= x_max:
                        zone_set.add(pos)
                NightmareStrategy._zone_shelves[zone_id] = zone_set

            # Distance from drop-off
            NightmareStrategy._dist_to_dropoff = multi_source_bfs(
                w, h, blocked, [sim.drop_off]
            )
            dd = NightmareStrategy._dist_to_dropoff

            # Distance to each item type
            NightmareStrategy._dist_to_type = {}
            for itype, shelves in NightmareStrategy._type_shelves.items():
                sources = []
                for s in shelves:
                    sources.extend(NightmareStrategy._shelf_adj.get(s, []))
                NightmareStrategy._dist_to_type[itype] = multi_source_bfs(
                    w, h, blocked, sources
                )

            # Min distance from each type to drop-off
            NightmareStrategy._type_dropoff_dist = {}
            for itype, shelves in NightmareStrategy._type_shelves.items():
                min_d = 999
                for s in shelves:
                    for adj in NightmareStrategy._shelf_adj.get(s, []):
                        d = dd.get(adj, 999)
                        if d < min_d:
                            min_d = d
                NightmareStrategy._type_dropoff_dist[itype] = min_d

            # Pre-compute 20 dispersal targets: spread bots across map
            # Each zone gets bots spread along its vertical aisles at various y
            NightmareStrategy._disperse_targets = _compute_disperse_targets(
                w, h, blocked, sim.drop_off
            )
            NightmareStrategy._disperse_dists = [
                multi_source_bfs(w, h, blocked, [t])
                for t in NightmareStrategy._disperse_targets
            ]

        # --- Per-game state ---
        n_bots = len(sim._bots)

        self._state: dict[int, BotState] = {i: BotState.DISPERSING for i in range(n_bots)}

        # Zone assignment
        self._bot_zone: dict[int, int] = {}
        bot_idx = 0
        for zone_id in range(6):
            count = self.p.zone_bots[zone_id] if zone_id < len(self.p.zone_bots) else 0
            for _ in range(count):
                if bot_idx < n_bots:
                    self._bot_zone[bot_idx] = zone_id
                    bot_idx += 1
        while bot_idx < n_bots:
            self._bot_zone[bot_idx] = 0
            bot_idx += 1

        # Claims: tracking what each bot is pursuing
        self._bot_target_type: dict[int, Optional[str]] = {}
        self._claims: Counter = Counter()  # type -> n bots assigned
        self._delivering: set[int] = set()  # bots currently heading to/at drop-off

        # Order tracking
        self._prev_active_order_id: Optional[str] = None

        # Stuck detection: track positions for oscillation detection
        self._prev_pos: dict[int, Pos] = {}
        self._stuck_count: dict[int, int] = {i: 0 for i in range(n_bots)}

        # Per-bot RNG for varied tie-breaking (seeded by base seed + bot ID)
        self._bot_rngs: dict[int, rng.Random] = {
            i: rng.Random(self.p.seed * 1000 + i) for i in range(n_bots)
        }

    # ------------------------------------------------------------------
    # Per-round decision
    # ------------------------------------------------------------------

    def _decide(self, sim) -> list[dict]:
        w, h = NightmareStrategy._w, NightmareStrategy._h
        blocked = sim.blocked
        dd = NightmareStrategy._dist_to_dropoff

        # Get orders
        active_order = None
        for o in sim._orders:
            if o.status == "active":
                active_order = o
                break

        if not active_order:
            return [{"bot": b.id, "action": "wait"} for b in sim._bots]

        # Detect order transition
        if self._prev_active_order_id and self._prev_active_order_id != active_order.id:
            self._handle_order_transition(sim, active_order)
        self._prev_active_order_id = active_order.id

        # What's still needed (raw = from order, adjusted = minus in-transit inventory)
        raw_remaining = Counter(active_order.items_remaining)
        adjusted = Counter(raw_remaining)
        for b in sim._bots:
            for t in b.inventory:
                if adjusted[t] > 0:
                    adjusted[t] -= 1

        endgame = (sim.max_rounds - sim._round) <= self.p.endgame_rounds
        # Clean up delivering set (bots that no longer have matching items)
        for bid in list(self._delivering):
            bot = sim._bots[bid]
            matching = sum(1 for t in bot.inventory if raw_remaining[t] > 0)
            if matching == 0 and bot.position != sim.drop_off:
                self._delivering.discard(bid)
                self._state[bid] = BotState.PICKING

        # Track occupied
        occupied: set[Pos] = set()
        bot_at: dict[Pos, int] = {}
        for b in sim._bots:
            occupied.add(b.position)
            bot_at[b.position] = bot_at.get(b.position, 0) + 1

        # Rebuild claims to be accurate
        self._rebuild_claims(sim, adjusted)

        actions = []
        for bot in sorted(sim._bots, key=lambda b: b.id):
            bot_at[bot.position] -= 1
            if bot_at[bot.position] <= 0:
                occupied.discard(bot.position)

            action = self._decide_bot(
                bot, sim, active_order, raw_remaining, adjusted,
                occupied, endgame
            )
            actions.append(action)

            if action["action"] in MOVES:
                dx, dy = MOVES[action["action"]]
                new_pos = (bot.position[0] + dx, bot.position[1] + dy)
                occupied.add(new_pos)
                bot_at[new_pos] = bot_at.get(new_pos, 0) + 1
            else:
                occupied.add(bot.position)
                bot_at[bot.position] = bot_at.get(bot.position, 0) + 1

        return actions

    def _rebuild_claims(self, sim, adjusted):
        """Rebuild claims from scratch each round for accuracy."""
        self._claims.clear()
        for bid, t in self._bot_target_type.items():
            if t and adjusted.get(t, 0) > 0:
                self._claims[t] += 1
            elif t:
                # Stale claim — type no longer needed
                self._bot_target_type[bid] = None

    # ------------------------------------------------------------------
    # Per-bot decision
    # ------------------------------------------------------------------

    def _decide_bot(self, bot, sim, active_order, raw_remaining, adjusted,
                    occupied, endgame) -> dict:
        dd = NightmareStrategy._dist_to_dropoff
        w, h = NightmareStrategy._w, NightmareStrategy._h
        blocked = sim.blocked

        # === DISPERSING: escape spawn stack, then pick immediately ===
        if self._state.get(bot.id) == BotState.DISPERSING:
            stacked = sum(1 for b in sim._bots if b.position == bot.position and b.id != bot.id)
            if stacked == 0:
                self._state[bot.id] = BotState.PICKING
            else:
                # Try to move to any free adjacent cell
                bot_rng = self._bot_rngs[bot.id]
                candidates = list(MOVE_LIST)
                bot_rng.shuffle(candidates)
                for action, (dx, dy) in candidates:
                    n = (bot.position[0] + dx, bot.position[1] + dy)
                    if (0 <= n[0] < w and 0 <= n[1] < h
                            and n not in blocked and n not in occupied):
                        return {"bot": bot.id, "action": action}
                return {"bot": bot.id, "action": "wait"}

        # === AT DROP-OFF ===
        if bot.position == sim.drop_off:
            matching = sum(1 for t in bot.inventory if raw_remaining[t] > 0)
            if matching > 0:
                # Deliver
                self._delivering.discard(bot.id)
                self._clear_claim(bot.id)
                self._state[bot.id] = BotState.PICKING
                return {"bot": bot.id, "action": "drop_off"}
            else:
                # At drop-off with no matching items — move away
                self._delivering.discard(bot.id)
                self._state[bot.id] = BotState.PICKING
                # Move right to clear drop-off
                right = (bot.position[0] + 1, bot.position[1])
                if right not in blocked and right not in occupied:
                    return {"bot": bot.id, "action": "move_right"}
                up = (bot.position[0], bot.position[1] - 1)
                if up not in blocked and up not in occupied:
                    return {"bot": bot.id, "action": "move_up"}
                return {"bot": bot.id, "action": "wait"}

        # === DELIVERING ===
        if bot.id in self._delivering:
            # Head toward drop-off
            return self._navigate(bot, dd, blocked, occupied, w, h)

        # === PICKING ===
        remaining_types = Counter(raw_remaining)
        matching = sum(1 for t in bot.inventory if remaining_types[t] > 0)
        inv_count = len(bot.inventory)

        # Should this bot deliver?
        should_deliver = False
        if matching > 0:
            dist = dd.get(bot.position, 999)
            if endgame and matching >= 1:
                should_deliver = True
            elif inv_count >= 3:
                should_deliver = True
            elif matching >= self.p.max_carry:
                should_deliver = True
            elif dist <= self.p.carry_if_close and matching >= 1:
                should_deliver = True

        if should_deliver:
            if len(self._delivering) < self.p.max_deliverers or inv_count >= 3:
                self._delivering.add(bot.id)
                self._clear_claim(bot.id)
                self._state[bot.id] = BotState.DELIVERING
                return self._navigate(bot, dd, blocked, occupied, w, h)

        # Try pickup (adjacent to needed item)
        if inv_count < 3:
            pickup = self._try_pickup(bot, sim, raw_remaining, adjusted)
            if pickup:
                return pickup

        # Find item to pursue
        if inv_count < 3 and any(v > 0 for v in adjusted.values()):
            target_type = self._find_best_type(bot, sim, adjusted, endgame)
            if target_type:
                self._set_claim(bot.id, target_type)
                dist_map = NightmareStrategy._dist_to_type.get(target_type)
                if dist_map:
                    return self._navigate(bot, dist_map, blocked, occupied, w, h)

        # Has matching items but queue full? Deliver anyway
        if matching > 0 and bot.id not in self._delivering:
            self._delivering.add(bot.id)
            self._clear_claim(bot.id)
            self._state[bot.id] = BotState.DELIVERING
            return self._navigate(bot, dd, blocked, occupied, w, h)

        # Dead weight: inventory has items that don't match -> deliver to clear
        if inv_count > 0 and matching == 0:
            self._delivering.add(bot.id)
            self._state[bot.id] = BotState.DELIVERING
            return self._navigate(bot, dd, blocked, occupied, w, h)

        # Bot is idle — move away from critical paths to avoid blocking
        if self._is_blocking_dropoff(bot.position, sim.drop_off):
            return self._move_away_from_dropoff(bot, sim, occupied, w, h, blocked)

        # Idle: stay put to avoid creating congestion
        return {"bot": bot.id, "action": "wait"}

    # ------------------------------------------------------------------
    # Pickup logic
    # ------------------------------------------------------------------

    def _try_pickup(self, bot, sim, raw_remaining, adjusted) -> Optional[dict]:
        """Pick up adjacent item that's still needed (respecting claims)."""
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            adj = (bot.position[0] + dx, bot.position[1] + dy)
            if adj not in sim.shelves:
                continue
            itype = sim.shelf_types.get(adj)
            if not itype or raw_remaining.get(itype, 0) <= 0:
                continue
            # Don't pick if adjusted shows 0 remaining and we don't already carry it
            if adjusted.get(itype, 0) <= 0 and itype not in bot.inventory:
                # Over-supplied — skip unless this is our claimed type
                if self._bot_target_type.get(bot.id) != itype:
                    continue
            for item in sim._items:
                if not item.picked and item.position == adj:
                    self._clear_claim(bot.id)
                    return {"bot": bot.id, "action": "pick_up", "item_id": item.id}
        return None

    def _find_best_type(self, bot, sim, adjusted, endgame) -> Optional[str]:
        """Find best item type for this bot, considering zone and claims."""
        dt = NightmareStrategy._dist_to_type
        zone = self._bot_zone.get(bot.id, 0)
        zone_shelf_set = NightmareStrategy._zone_shelves.get(zone, set())

        best_type = None
        best_score = float('inf')

        for itype, needed in adjusted.items():
            if needed <= 0:
                continue
            # Don't over-assign: skip if already enough bots chasing this type
            if self._claims.get(itype, 0) >= needed:
                # Allow if this bot is already claiming this type
                if self._bot_target_type.get(bot.id) != itype:
                    continue

            dist_map = dt.get(itype)
            if not dist_map:
                continue
            dist = dist_map.get(bot.position, 999)

            # Drop-off proximity
            dropoff_cost = NightmareStrategy._type_dropoff_dist.get(itype, 0) * self.p.w_dropoff

            # Zone penalty
            zone_pen = 0
            if not endgame:
                type_shelves = NightmareStrategy._type_shelves.get(itype, [])
                in_zone = any(s in zone_shelf_set for s in type_shelves)
                if not in_zone:
                    zone_pen = self.p.zone_penalty

            # Completion bonus
            completion = -self.p.w_completion if needed == 1 else 0

            # Batch bonus: prefer items already in inventory
            batch = -self.p.batch_bonus if itype in bot.inventory else 0

            score = dist * self.p.w_distance + dropoff_cost + zone_pen + completion + batch
            if score < best_score:
                best_score = score
                best_type = itype

        return best_type

    # ------------------------------------------------------------------
    # Claim management
    # ------------------------------------------------------------------

    def _set_claim(self, bot_id: int, item_type: str):
        old = self._bot_target_type.get(bot_id)
        if old == item_type:
            return
        if old and self._claims[old] > 0:
            self._claims[old] -= 1
        self._bot_target_type[bot_id] = item_type
        self._claims[item_type] += 1

    def _clear_claim(self, bot_id: int):
        old = self._bot_target_type.get(bot_id)
        if old and self._claims[old] > 0:
            self._claims[old] -= 1
        self._bot_target_type[bot_id] = None

    # ------------------------------------------------------------------
    # Order transition
    # ------------------------------------------------------------------

    def _handle_order_transition(self, sim, new_active_order):
        """Reset claims when order changes."""
        self._claims.clear()
        for bid in list(self._bot_target_type.keys()):
            self._bot_target_type[bid] = None

        new_remaining = Counter(new_active_order.items_remaining)

        # Delivering bots: check if items still match
        for bid in list(self._delivering):
            bot = sim._bots[bid]
            matching = sum(1 for t in bot.inventory if new_remaining[t] > 0)
            if matching == 0 and not bot.inventory:
                # Empty inventory, no point delivering
                self._delivering.discard(bid)
                self._state[bid] = BotState.PICKING

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate(self, bot, dist_map: dict[Pos, int], blocked: frozenset,
                  occupied: set[Pos], w: int, h: int) -> dict:
        """Greedy move toward lower distance with stuck detection.

        Each bot has its own RNG (seeded by bot ID) for different tie-breaking.
        When stuck: pick the most "open" neighbor to escape congestion.
        """
        prev = self._prev_pos.get(bot.id)
        if prev == bot.position:
            self._stuck_count[bot.id] = self._stuck_count.get(bot.id, 0) + 1
        else:
            self._stuck_count[bot.id] = 0
        self._prev_pos[bot.id] = bot.position

        stuck = self._stuck_count.get(bot.id, 0) >= 2

        best_dir = None
        best_d = dist_map.get(bot.position, 9999)
        all_legal: list[tuple[str, int]] = []

        # Per-bot shuffle using bot's own RNG
        bot_rng = self._bot_rngs[bot.id]
        candidates = list(MOVE_LIST)
        bot_rng.shuffle(candidates)

        for action, (dx, dy) in candidates:
            n = (bot.position[0] + dx, bot.position[1] + dy)
            if not (0 <= n[0] < w and 0 <= n[1] < h):
                continue
            if n in blocked or n in occupied:
                continue

            d = dist_map.get(n, 9999)
            all_legal.append((action, d))
            if d < best_d:
                best_d = d
                best_dir = action

        if stuck and all_legal:
            self._stuck_count[bot.id] = 0
            # Pick least-congested neighbor
            best_action = all_legal[0][0]
            best_openness = -1
            for action, d in all_legal:
                dx, dy = MOVES[action]
                n = (bot.position[0] + dx, bot.position[1] + dy)
                openness = 0
                for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    nn = (n[0] + ddx, n[1] + ddy)
                    if (0 <= nn[0] < w and 0 <= nn[1] < h
                            and nn not in blocked and nn not in occupied):
                        openness += 1
                if openness > best_openness:
                    best_openness = openness
                    best_action = action
            return {"bot": bot.id, "action": best_action}

        if best_dir:
            return {"bot": bot.id, "action": best_dir}
        if all_legal:
            return {"bot": bot.id, "action": all_legal[0][0]}
        return {"bot": bot.id, "action": "wait"}

    def _is_blocking_dropoff(self, pos: Pos, drop_off: Pos) -> bool:
        """Check if position is on the drop-off approach lane and could block deliverers."""
        # Drop-off approach: x=1 y=10-16, y=15 x=1-10, drop-off itself
        x, y = pos
        if pos == drop_off:
            return True
        if x == 1 and 10 <= y <= 16:
            return True
        if y == 15 and x <= 10:
            return True
        return False

    def _move_away_from_dropoff(self, bot, sim, occupied, w, h, blocked) -> dict:
        """Move idle bot away from drop-off approach lane."""
        # Try to move to a non-blocking position
        # Priority: right (away from x=1), then up (away from y=15/16)
        moves_priority = [
            ("move_right", (1, 0)),
            ("move_up", (0, -1)),
            ("move_down", (0, 1)),
            ("move_left", (-1, 0)),
        ]
        for action, (dx, dy) in moves_priority:
            n = (bot.position[0] + dx, bot.position[1] + dy)
            if (0 <= n[0] < w and 0 <= n[1] < h
                    and n not in blocked and n not in occupied):
                return {"bot": bot.id, "action": action}
        return {"bot": bot.id, "action": "wait"}


# ---------------------------------------------------------------------------
# Dispersal target computation
# ---------------------------------------------------------------------------

def _compute_disperse_targets(w: int, h: int, blocked: frozenset,
                              drop_off: Pos) -> list[Pos]:
    """Compute 20 spread-out dispersal targets.

    Place them on zone entry positions (aisles at various y coords).
    Prioritize positions far from each other and from spawn.
    """
    dd = multi_source_bfs(w, h, blocked, [drop_off])

    # Candidate positions: walkable cells on aisles
    aisles = [4, 8, 12, 16, 20, 24]
    shelf_ys_upper = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    shelf_ys_lower = [9, 10, 11, 12, 13, 14, 15]
    candidates = []
    for x in aisles:
        for y in shelf_ys_upper + shelf_ys_lower:
            pos = (x, y)
            if pos not in blocked:
                candidates.append(pos)
    # Add edge positions
    for x in [1, 27, 28]:
        for y in [1, 5, 9, 13, 15]:
            pos = (x, y)
            if pos not in blocked:
                candidates.append(pos)

    # Remove duplicates
    candidates = list(set(candidates))

    # Greedy selection: maximize minimum distance between selected targets
    targets: list[Pos] = []
    for _ in range(20):
        best_pos = None
        best_score = -1
        for pos in candidates:
            if pos in targets:
                continue
            min_to_others = min(
                (abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in targets),
                default=999
            )
            # Score: spread from others + moderate distance from drop-off
            score = min_to_others * 3 + min(dd.get(pos, 0), 15)
            if score > best_score:
                best_score = score
                best_pos = pos
        if best_pos:
            targets.append(best_pos)
        else:
            # Fallback: just pick any walkable
            for pos in candidates:
                if pos not in targets:
                    targets.append(pos)
                    break
            else:
                targets.append((1, 1))  # emergency fallback

    return targets
