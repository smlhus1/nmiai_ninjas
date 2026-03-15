"""Live coordinator using solver logic.

Fully reactive — recalculates paths from server positions every round.
No internal position tracking (server is always truth).
"""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field

from .grid import GameMap, Grid, Pos, pickup_positions
from .pathfinding import DistanceCache
from .orders import ShelfIndex
from .trips import PickupStep, _best_permutation

logger = logging.getLogger(__name__)


@dataclass
class LiveBot:
    bot_id: int
    pos: Pos = (0, 0)
    inventory: list[str] = field(default_factory=list)
    goal: str = "idle"  # idle, pickup, deliver, disperse
    target_pos: Pos | None = None
    target_item_type: str | None = None
    remaining_steps: list[PickupStep] = field(default_factory=list)
    trip_dropoff: Pos | None = None
    target_order: str | None = None


class LiveCoordinator:
    """Reactive coordinator — computes one action per bot per round."""

    def __init__(self) -> None:
        self._initialized = False
        self._game_map: GameMap | None = None
        self._dist_cache: DistanceCache | None = None
        self._shelf_index: ShelfIndex | None = None
        self._bots: list[LiveBot] = []
        self._round = 0
        self._claimed_active: Counter[str] = Counter()
        self._claimed_preview: Counter[str] = Counter()
        self._last_active_id: str | None = None
        self._max_items = 2

    def handle_round(self, state: dict) -> dict:
        t0 = time.time()

        if not self._initialized:
            self._init_from_state(state)

        self._round = state.get("round", self._round)

        # Build item lookup
        items_at: dict[Pos, list[dict]] = {}
        for item in state.get("items", []):
            pos = tuple(item["position"])
            items_at.setdefault(pos, []).append(item)

        # Update bot state from server
        for bot_data in state.get("bots", []):
            bot = self._bots[bot_data["id"]]
            bot.pos = tuple(bot_data["position"])
            raw_inv = bot_data.get("inventory", [])
            bot.inventory = [i["type"] if isinstance(i, dict) else i for i in raw_inv]

        # Parse orders
        active_order = None
        preview_order = None
        for order in state.get("orders", []):
            status = order.get("status", "")
            if status == "active" and not order.get("complete"):
                active_order = order
            elif status == "preview":
                preview_order = order

        active_id = active_order["id"] if active_order else None
        preview_id = preview_order["id"] if preview_order else None

        # Detect order transition
        if self._last_active_id and active_id != self._last_active_id:
            self._claimed_active = Counter(self._claimed_preview)
            self._claimed_preview.clear()
            # Bots with pre-picked items for now-active order → deliver
            for bot in self._bots:
                if bot.target_order == active_id and bot.goal == "idle" and bot.inventory:
                    bot.goal = "deliver"
                    bot.target_pos = self._nearest_dz(bot.pos)
                    bot.trip_dropoff = bot.target_pos

        self._last_active_id = active_id

        # Items remaining for active order
        items_remaining = Counter(active_order.get("items_required", [])) if active_order else Counter()
        if active_order:
            for it in active_order.get("items_delivered", []):
                if items_remaining[it] > 0:
                    items_remaining[it] -= 1
        preview_remaining = Counter(preview_order.get("items_required", [])) if preview_order else Counter()

        # Assign trips to idle bots
        for bot in self._bots:
            if bot.goal == "idle":
                assigned = self._assign_trip(bot, items_remaining, self._claimed_active,
                                             active_id, is_active=True)
                if not assigned and preview_id:
                    assigned = self._assign_trip(bot, preview_remaining, self._claimed_preview,
                                                 preview_id, is_active=False)
                if not assigned:
                    self._disperse_bot(bot)

        # Generate all actions (includes pickup/dropoff for bots at target)
        actions = self._build_actions(state, items_at)
        elapsed = time.time() - t0
        self._log_round(state, items_remaining, elapsed)

        # Log first few rounds in detail
        if self._round < 3:
            import json
            logger.info(f"  RESPONSE: {json.dumps(actions)[:300]}")

        return actions

    def _build_actions(self, state: dict, items_at: dict[Pos, list[dict]]) -> dict:
        """Generate actions with collision avoidance."""
        # First pass: determine desired next positions
        desired: dict[int, Pos] = {}
        specials: dict[int, dict] = {}  # bot_id -> action dict for pickup/dropoff

        for bot in self._bots:
            if bot.goal == "pickup" and bot.pos == bot.target_pos:
                item_id = self._find_item_id(bot.pos, bot.target_item_type, items_at)
                if item_id:
                    specials[bot.bot_id] = {"bot": bot.bot_id, "action": "pick_up", "item_id": item_id}
                    logger.info(f"R{self._round}: Bot {bot.bot_id} PICKUP {bot.target_item_type} "
                                f"item_id={item_id} at {bot.pos}")
                    self._advance_after_pickup(bot, self._last_active_id)
                else:
                    specials[bot.bot_id] = {"bot": bot.bot_id, "action": "wait"}
                    logger.warning(f"R{self._round}: Bot {bot.bot_id} at {bot.pos} wants {bot.target_item_type} "
                                   f"but NO ITEM FOUND! items_at_adjacent={self._debug_items_at(bot.pos, items_at)}")
                desired[bot.bot_id] = bot.pos

            elif bot.goal == "deliver" and bot.pos == bot.target_pos:
                specials[bot.bot_id] = {"bot": bot.bot_id, "action": "drop_off"}
                logger.info(f"R{self._round}: Bot {bot.bot_id} DROP_OFF at {bot.pos} inv={bot.inventory}")
                bot.goal = "idle"
                bot.target_order = None
                desired[bot.bot_id] = bot.pos

            elif bot.target_pos and bot.pos != bot.target_pos:
                path = self._path(bot.pos, bot.target_pos)
                if path:
                    desired[bot.bot_id] = path[0]
                else:
                    desired[bot.bot_id] = bot.pos
            else:
                desired[bot.bot_id] = bot.pos

        # Second pass: resolve collisions (lower ID wins)
        claimed: set[Pos] = set()
        final: dict[int, Pos] = {}

        for bot in sorted(self._bots, key=lambda b: b.bot_id):
            pos = desired[bot.bot_id]

            if pos == bot.pos:
                final[bot.bot_id] = pos
                # Don't claim our own position (multiple bots can stay put)
                continue

            if pos in claimed:
                # Collision — try alternative directions
                alt = self._find_alt_move(bot, claimed)
                final[bot.bot_id] = alt
                if alt != bot.pos:
                    claimed.add(alt)
            else:
                # Check swap collision
                swapped = False
                for oid, opos in final.items():
                    if pos == self._bots[oid].pos and opos == bot.pos:
                        swapped = True
                        break
                if swapped:
                    final[bot.bot_id] = bot.pos
                else:
                    final[bot.bot_id] = pos
                    claimed.add(pos)

        # Convert to actions
        result = []
        for bot in self._bots:
            if bot.bot_id in specials:
                result.append(specials[bot.bot_id])
            else:
                new_pos = final[bot.bot_id]
                if new_pos == bot.pos:
                    result.append({"bot": bot.bot_id, "action": "wait"})
                else:
                    dx = new_pos[0] - bot.pos[0]
                    dy = new_pos[1] - bot.pos[1]
                    if dx == 1:
                        result.append({"bot": bot.bot_id, "action": "move_right"})
                    elif dx == -1:
                        result.append({"bot": bot.bot_id, "action": "move_left"})
                    elif dy == 1:
                        result.append({"bot": bot.bot_id, "action": "move_down"})
                    elif dy == -1:
                        result.append({"bot": bot.bot_id, "action": "move_up"})
                    else:
                        result.append({"bot": bot.bot_id, "action": "wait"})

        return {"actions": result}

    def _find_alt_move(self, bot: LiveBot, claimed: set[Pos]) -> Pos:
        """Find an alternative move when desired position is claimed."""
        grid = self._game_map.grid
        for nb in grid.neighbors(bot.pos):
            if nb not in claimed:
                return nb
        return bot.pos  # stay put

    def _advance_after_pickup(self, bot: LiveBot, active_id: str | None) -> None:
        """After successful pickup, move to next step or delivery."""
        if bot.remaining_steps:
            step = bot.remaining_steps.pop(0)
            bot.target_item_type = step.item_type
            bot.target_pos = step.pickup_pos
            bot.goal = "pickup"
        else:
            if bot.target_order == active_id:
                bot.target_pos = bot.trip_dropoff
                bot.goal = "deliver"
            else:
                # Pre-pick done — wait for order to become active
                bot.goal = "idle"

    def _debug_items_at(self, pos: Pos, items_at: dict[Pos, list[dict]]) -> str:
        """Debug: what items are at/adjacent to pos."""
        result = []
        for dx, dy in ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)):
            adj = (pos[0] + dx, pos[1] + dy)
            items = items_at.get(adj, [])
            if items:
                result.append(f"{adj}:[{','.join(i['type'] for i in items)}]")
        return " ".join(result) if result else "NONE"

    def _find_item_id(self, pos: Pos, item_type: str | None, items_at: dict[Pos, list[dict]]) -> str | None:
        """Find item_id matching type at pos or adjacent shelf."""
        if not item_type:
            return None
        for dx, dy in ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)):
            adj = (pos[0] + dx, pos[1] + dy)
            for item in items_at.get(adj, []):
                if item["type"] == item_type:
                    return item["id"]
        return None

    def _log_round(self, state: dict, items_remaining: Counter, elapsed: float) -> None:
        if self._round % 50 == 0 or self._round < 3:
            active = sum(1 for b in self._bots if b.goal not in ("idle", "disperse"))
            score = state.get("score", 0)
            logger.info(f"R{self._round}: {elapsed*1000:.0f}ms score={score} active={active} "
                        f"needed={items_remaining.total()}")

    def _init_from_state(self, state: dict) -> None:
        grid_data = state.get("grid", {})
        width = grid_data.get("width", 30)
        height = grid_data.get("height", 18)
        walls = [tuple(w) for w in grid_data.get("walls", [])]

        shelf_map: dict[str, list[Pos]] = {}
        all_shelves: set[Pos] = set()
        for item in state.get("items", []):
            item_type = item["type"]
            pos = tuple(item["position"])
            shelf_map.setdefault(item_type, []).append(pos)
            all_shelves.add(pos)

        grid = Grid(width=width, height=height,
                     walls=frozenset(walls), shelves=frozenset(all_shelves))

        raw_zones = state.get("drop_off_zones", [])
        drop_off_zones = tuple(tuple(z) for z in raw_zones) if raw_zones else (tuple(state.get("drop_off", [1, 16])),)

        bots_data = state.get("bots", [])
        spawn = tuple(bots_data[0]["position"]) if bots_data else (28, 16)

        self._game_map = GameMap(grid=grid, drop_off_zones=drop_off_zones,
                                  shelf_map=shelf_map, spawn=spawn, bot_count=len(bots_data))
        self._dist_cache = DistanceCache(grid)
        self._shelf_index = ShelfIndex(self._game_map, self._dist_cache)
        self._bots = [LiveBot(bot_id=b["id"], pos=tuple(b["position"])) for b in bots_data]

        self._initialized = True
        logger.info(f"Solver: {width}x{height}, {len(bots_data)} bots, {len(drop_off_zones)} zones, "
                     f"{len(shelf_map)} types")

    def _assign_trip(self, bot: LiveBot, items_remaining: Counter[str],
                     claimed: Counter[str], order_id: str | None, is_active: bool) -> bool:
        unclaimed: Counter[str] = Counter()
        for it, needed in items_remaining.items():
            if needed > claimed.get(it, 0):
                unclaimed[it] = needed - claimed.get(it, 0)

        if unclaimed.total() == 0:
            if is_active and bot.inventory:
                if any(items_remaining[i] > 0 for i in bot.inventory):
                    bot.target_pos = self._nearest_dz(bot.pos)
                    bot.trip_dropoff = bot.target_pos
                    bot.target_order = order_id
                    bot.goal = "deliver"
                    return True
            return False

        # Find nearest items
        options = []
        for it in unclaimed:
            best_entry = None
            best_d = float("inf")
            for dz in self._game_map.drop_off_zones:
                entries = self._shelf_index.get(it, dz)
                for entry in (entries or [])[:3]:
                    d = self._dist_cache.distance(bot.pos, entry.pickup_pos)
                    if d is not None and d < best_d:
                        best_d = d
                        best_entry = entry
            if best_entry:
                options.append((it, PickupStep(it, best_entry.shelf_pos, best_entry.pickup_pos), int(best_d)))

        options.sort(key=lambda x: x[2])
        steps = []
        to_claim = []
        seen: Counter[str] = Counter()
        for it, step, _ in options:
            if len(steps) >= self._max_items:
                break
            if seen[it] >= unclaimed[it]:
                continue
            steps.append(step)
            to_claim.append(it)
            seen[it] += 1

        if not steps:
            return False

        dz = self._nearest_dz(steps[-1].pickup_pos)
        result = _best_permutation(bot.pos, steps, dz, self._dist_cache)
        if not result:
            return False

        ordered, _ = result
        for it in to_claim:
            claimed[it] += 1

        first = ordered[0]
        bot.target_item_type = first.item_type
        bot.target_pos = first.pickup_pos
        bot.remaining_steps = list(ordered[1:])
        bot.trip_dropoff = dz
        bot.target_order = order_id
        bot.goal = "pickup"
        return True

    def _disperse_bot(self, bot: LiveBot) -> None:
        bots_here = sum(1 for b in self._bots if b.pos == bot.pos)
        if bots_here <= 1:
            return
        grid = self._game_map.grid
        corridors = [(x, y) for x in range(grid.width) for y in (1, 9, 15) if grid.walkable((x, y))]
        if corridors:
            target = corridors[bot.bot_id % len(corridors)]
            bot.target_pos = target
            bot.goal = "disperse"

    def _nearest_dz(self, pos: Pos) -> Pos:
        best = self._game_map.drop_off_zones[0]
        best_d = float("inf")
        for dz in self._game_map.drop_off_zones:
            d = self._dist_cache.distance(pos, dz)
            if d is not None and d < best_d:
                best_d = d
                best = dz
        return best

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
            for nb in self._game_map.grid.neighbors(pos):
                if nb not in parent:
                    parent[nb] = pos
                    q.append(nb)
        return []
