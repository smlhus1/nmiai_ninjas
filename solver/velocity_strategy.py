"""Velocity-optimized strategy: maximize score/round through aggressive pre-picking.

Key insight: with 20 bots and ~6 items per order, 14 bots are idle.
Use ALL idle bots to pre-pick next order's items. When current order
completes → pre-picked items already in inventory → deliver instantly.

Target: ~5 rounds per order after warmup → 2.5 score/round.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

Pos = tuple[int, int]


class VelocityStrategy:
    """Pure velocity strategy: pre-pick everything, deliver instantly."""

    def __init__(self, recon: dict, shelf_pref: dict[str, int] | None = None):
        grid_w, grid_h = recon["grid_size"]
        self.w = grid_w
        self.h = grid_h

        walls = set(tuple(w) for w in recon["walls"])
        shelves: set[Pos] = set()
        self._shelf_map: dict[str, list[Pos]] = {}
        for item_type, positions in recon["shelf_map"].items():
            shelf_list = [tuple(p) for p in positions]
            self._shelf_map[item_type] = shelf_list
            for p in shelf_list:
                shelves.add(p)
        obstacles = walls | shelves

        from bot.models import Grid as BotGrid
        from bot.engine.pathfinding import PathEngine
        from bot.engine.pibt import PIBTResolver
        from bot.engine.guidance import GuidanceGraph

        self._grid = BotGrid(grid_w, grid_h, frozenset(obstacles))
        self._pe = PathEngine()
        self.drop_offs = [tuple(z) for z in recon["drop_off_zones"]]
        self._pe.set_grid(self._grid, drop_off=self.drop_offs[0])
        self._pe._one_way = self._pe._detect_one_way_aisles(self._grid, self.drop_offs[0])
        self._pe._one_way_enabled = True

        self._guidance = GuidanceGraph(
            self._grid, one_way=self._pe._one_way,
            alpha=2.0, beta=2.0, decay=0.5, update_interval=3,
        )
        self._pibt = PIBTResolver(
            self._grid, self._pe.distance, self._pe.corridors,
            one_way=self._pe._one_way,
            guidance_fn=self._guidance.guided_distance,
        )

        self.obstacles = obstacles
        self.order_sequence = recon.get("order_sequence", [])
        self.shelf_pref = shelf_pref or {}

        # State
        self._goal: dict[int, str] = {}       # pickup, deliver, prepick, idle
        self._target: dict[int, Pos] = {}
        self._item_type: dict[int, str | None] = {}
        self._prev_inv: dict[int, list[str]] = {}
        self._prev_pos: dict[int, Pos] = {}
        self._stuck: dict[int, int] = {}
        self._order_id: str | None = None
        self._order_idx: int = 0
        self._claimed: set[str] = set()  # claimed item IDs

        # Safe idle spots
        one_way = self._pe._one_way
        drop_set = set(self.drop_offs)
        self._idle_spots = [
            (x, y) for y in range(grid_h) for x in range(grid_w)
            if (x, y) not in obstacles and (x, y) not in drop_set and (x, y) not in one_way
        ]
        self._idle_spots.sort(key=lambda p: (p[1], p[0]))

    def _safe_idle(self, pos: Pos) -> Pos:
        if pos in self._idle_spots:
            return pos
        best, best_d = pos, 9999
        for p in self._idle_spots[:50]:
            d = self._pe.distance(pos, p) or 9999
            if d < best_d:
                best_d = d
                best = p
                if d <= 3:
                    break
        return best

    def _pickup_pos(self, bot_pos: Pos, shelf_pos: Pos) -> Pos | None:
        best, best_d = None, 9999
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            adj = (shelf_pos[0] + dx, shelf_pos[1] + dy)
            if 0 <= adj[0] < self.w and 0 <= adj[1] < self.h and adj not in self.obstacles:
                d = self._pe.distance(bot_pos, adj) or 9999
                if d < best_d:
                    best_d = d
                    best = adj
        return best

    def _find_item_id(self, pos: Pos, item_type: str | None, items_by_pos: dict) -> str | None:
        if not item_type:
            return None
        for dx, dy in [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            adj = (pos[0] + dx, pos[1] + dy)
            for item in items_by_pos.get(adj, []):
                if item["type"] == item_type:
                    return item["id"]
        return None

    def _nearest_dropoff(self, pos: Pos, load: Counter | None = None) -> Pos:
        """Nearest drop-off with load balancing."""
        best, best_score = self.drop_offs[0], 9999
        for dz in self.drop_offs:
            d = self._pe.distance(pos, dz) or 9999
            penalty = (load.get(dz, 0) * 5) if load else 0
            score = d + penalty
            if score < best_score:
                best_score = score
                best = dz
        return best

    def _preferred_shelf(self, item_type: str) -> Pos | None:
        """Get preferred shelf position for item type."""
        shelves = self._shelf_map.get(item_type, [])
        if not shelves:
            return None
        idx = self.shelf_pref.get(item_type, 0) % len(shelves)
        return shelves[idx]

    def _assign_pickup(self, bid: int, pos: Pos, item_type: str,
                       items: list[dict], items_by_pos: dict,
                       target_types: set[str]) -> bool:
        """Assign bot to pick up an item. Returns True if assigned."""
        # Try preferred shelf first
        pref_shelf = self._preferred_shelf(item_type)
        if pref_shelf:
            pp = self._pickup_pos(pos, pref_shelf)
            if pp:
                self._goal[bid] = "pickup"
                self._target[bid] = pp
                self._item_type[bid] = item_type
                return True

        # Fallback: nearest item of this type
        best_d, best_pp, best_type = 9999, None, None
        for item in items:
            if item["id"] in self._claimed:
                continue
            if item["type"] not in target_types:
                continue
            ipos = tuple(item["position"])
            pp = self._pickup_pos(pos, ipos)
            if not pp:
                continue
            d = self._pe.distance(pos, pp) or 9999
            if d < best_d:
                best_d = d
                best_pp = pp
                best_type = item["type"]
                best_id = item["id"]

        if best_pp:
            self._goal[bid] = "pickup"
            self._target[bid] = best_pp
            self._item_type[bid] = best_type
            self._claimed.add(best_id)
            return True
        return False

    def __call__(self, state: dict) -> dict:
        bots = state.get("bots", [])
        items = state.get("items", [])
        orders = state.get("orders", [])
        round_num = state.get("round", 0)

        # Parse orders
        active, preview = None, None
        for o in orders:
            if o.get("status") == "active" and not o.get("complete"):
                active = o
            elif o.get("status") == "preview":
                preview = o

        active_id = active["id"] if active else None

        # Remaining for active order
        remaining = Counter()
        remaining_types: set[str] = set()
        if active:
            remaining = Counter(active["items_required"])
            for d in active.get("items_delivered", []):
                if remaining[d] > 0:
                    remaining[d] -= 1
            remaining_types = set(t for t, c in remaining.items() if c > 0)

        preview_types: set[str] = set()
        preview_remaining = Counter()
        if preview:
            preview_remaining = Counter(preview.get("items_required", []))
            preview_types = set(preview_remaining.keys())

        # Order changed → promote prepick bots to deliver
        if active_id != self._order_id:
            if self._order_id is not None:
                self._order_idx += 1
            self._order_id = active_id
            self._claimed.clear()

            for bid in list(self._goal.keys()):
                g = self._goal[bid]
                if g in ("prepick", "prepick_done"):
                    # Check if bot has matching items
                    inv = []
                    for b in bots:
                        if b["id"] == bid:
                            inv = b.get("inventory", [])
                            break
                    if any(remaining.get(t, 0) > 0 for t in inv):
                        pos = self._prev_pos.get(bid, (0, 0))
                        self._goal[bid] = "deliver"
                        self._target[bid] = self._nearest_dropoff(pos)
                    else:
                        self._goal[bid] = "idle"
                elif g not in ("deliver",):
                    self._goal[bid] = "idle"
                    self._item_type[bid] = None

        # Build lookups
        items_by_pos: dict[Pos, list] = {}
        for item in items:
            items_by_pos.setdefault(tuple(item["position"]), []).append(item)
        bot_positions = {b["id"]: tuple(b["position"]) for b in bots}

        self._guidance.on_round(bot_positions, round_num)

        # Stuck detection
        for bot in bots:
            bid = bot["id"]
            pos = tuple(bot["position"])
            self._stuck[bid] = (self._stuck.get(bid, 0) + 1) if pos == self._prev_pos.get(bid) else 0
            self._prev_pos[bid] = pos

        # Detect pickup completion → deliver or prepick_done
        for bot in bots:
            bid = bot["id"]
            inv = bot.get("inventory", [])
            prev_inv = self._prev_inv.get(bid, [])
            if len(inv) > len(prev_inv):
                g = self._goal.get(bid, "idle")
                pos = tuple(bot["position"])
                if g == "pickup":
                    self._goal[bid] = "deliver"
                    self._target[bid] = self._nearest_dropoff(pos)
                    self._item_type[bid] = None
                elif g == "prepick":
                    self._goal[bid] = "prepick_done"
                    self._target[bid] = self._nearest_dropoff(pos)
                    self._item_type[bid] = None

        # Init untracked
        for bot in bots:
            bid = bot["id"]
            if bid not in self._goal:
                self._goal[bid] = "idle"
                self._target[bid] = self._safe_idle(tuple(bot["position"]))
                self._item_type[bid] = None

        # === DELIVER: bots with matching inventory ===
        dropoff_load = Counter()
        for bid, g in self._goal.items():
            if g == "deliver":
                dropoff_load[self._target.get(bid)] += 1

        for bot in bots:
            bid = bot["id"]
            inv = bot.get("inventory", [])
            mc = sum(1 for t in inv if remaining.get(t, 0) > 0)
            if mc > 0 and self._goal.get(bid) not in ("deliver",):
                pos = tuple(bot["position"])
                self._goal[bid] = "deliver"
                self._target[bid] = self._nearest_dropoff(pos, dropoff_load)
                dropoff_load[self._target[bid]] += 1

        # === VALIDATE ===
        for bot in bots:
            bid = bot["id"]
            g = self._goal.get(bid, "idle")
            inv = bot.get("inventory", [])
            pos = tuple(bot["position"])

            if g == "deliver" and not any(remaining.get(t, 0) > 0 for t in inv):
                if any(t in preview_types for t in inv):
                    self._goal[bid] = "prepick_done"
                    self._target[bid] = self._nearest_dropoff(pos)
                else:
                    self._goal[bid] = "idle"

            if g in ("pickup", "prepick") and self._stuck.get(bid, 0) > 12:
                self._goal[bid] = "idle"
                self._item_type[bid] = None
                self._stuck[bid] = 0

        # === ASSIGN IDLE BOTS ===
        # Count what's being picked/filled
        filling = Counter()
        for bid, g in self._goal.items():
            if g in ("pickup", "prepick") and self._item_type.get(bid):
                filling[self._item_type[bid]] += 1

        for bot in bots:
            bid = bot["id"]
            if self._goal.get(bid) != "idle":
                continue
            inv = bot.get("inventory", [])
            pos = tuple(bot["position"])

            # Full inventory → deliver if matching, else park
            if len(inv) >= 3:
                mc = sum(1 for t in inv if remaining.get(t, 0) > 0)
                if mc > 0:
                    self._goal[bid] = "deliver"
                    self._target[bid] = self._nearest_dropoff(pos, dropoff_load)
                else:
                    self._target[bid] = self._safe_idle(pos)
                continue

            # Priority 1: Active order items (not yet covered)
            assigned = False
            if remaining_types:
                for t in remaining_types:
                    need = remaining[t] - filling.get(t, 0)
                    if need > 0:
                        if self._assign_pickup(bid, pos, t, items, items_by_pos, remaining_types):
                            filling[t] += 1
                            assigned = True
                            break

            # Priority 2: Preview/prepick items
            if not assigned and preview_types:
                # Count what's already pre-picked in inventories
                prepicked = Counter()
                for b in bots:
                    for t in b.get("inventory", []):
                        if t in preview_types:
                            prepicked[t] += 1
                for bid2, g2 in self._goal.items():
                    if g2 == "prepick" and self._item_type.get(bid2):
                        prepicked[self._item_type[bid2]] += 1

                for t in preview_types:
                    need = preview_remaining[t] - prepicked.get(t, 0)
                    if need > 0:
                        # Use preferred shelf
                        pref = self._preferred_shelf(t)
                        if pref:
                            pp = self._pickup_pos(pos, pref)
                            if pp:
                                self._goal[bid] = "prepick"
                                self._target[bid] = pp
                                self._item_type[bid] = t
                                assigned = True
                                break

            if not assigned:
                self._target[bid] = self._safe_idle(pos)

        # === ACTIONS ===
        immediate: dict[int, dict] = {}
        move_bots: dict[int, Pos] = {}
        move_targets: dict[int, Pos] = {}

        drop_set = set(self.drop_offs)

        for bot in bots:
            bid = bot["id"]
            pos = tuple(bot["position"])
            inv = bot.get("inventory", [])
            g = self._goal.get(bid, "idle")
            target = self._target.get(bid, pos)

            # Pickup at target
            if g in ("pickup", "prepick") and pos == target:
                item_id = self._find_item_id(pos, self._item_type.get(bid), items_by_pos)
                if item_id:
                    immediate[bid] = {"bot": bid, "action": "pick_up", "item_id": item_id}
                    continue
                self._goal[bid] = "idle"
                self._target[bid] = self._safe_idle(pos)

            # Drop-off
            if g == "deliver" and pos in drop_set:
                if inv and any(remaining.get(t, 0) > 0 for t in inv):
                    immediate[bid] = {"bot": bid, "action": "drop_off"}
                    self._goal[bid] = "idle"
                    self._target[bid] = self._safe_idle(pos)
                    continue
                # Escape
                move_bots[bid] = pos
                move_targets[bid] = self._safe_idle(pos)
                continue

            move_bots[bid] = pos
            move_targets[bid] = target

        # Stationary for PIBT
        for bid in immediate:
            if bid not in move_bots:
                move_bots[bid] = bot_positions[bid]
                move_targets[bid] = bot_positions[bid]

        # Urgency
        urgency: dict[int, int] = {}
        idle_set: set[int] = set()
        for bid in move_bots:
            pos = move_bots[bid]
            g = self._goal.get(bid, "idle")
            if pos in drop_set and bid not in immediate:
                urgency[bid] = -1
            elif g == "deliver":
                urgency[bid] = 0
            elif g == "pickup":
                urgency[bid] = 1
            elif g == "prepick":
                urgency[bid] = 2
            else:
                urgency[bid] = 3
                idle_set.add(bid)

        resolved = self._pibt.resolve(
            move_bots, move_targets,
            tiebreak_offset=round_num,
            urgency=urgency,
            idle_bots=idle_set,
        ) if move_bots else {}

        actions = []
        for bot in bots:
            bid = bot["id"]
            pos = tuple(bot["position"])
            if bid in immediate:
                actions.append(immediate[bid])
            elif bid in resolved:
                new_pos = resolved[bid]
                if new_pos == pos:
                    actions.append({"bot": bid, "action": "wait"})
                else:
                    dx, dy = new_pos[0] - pos[0], new_pos[1] - pos[1]
                    act = {(1, 0): "move_right", (-1, 0): "move_left",
                           (0, 1): "move_down", (0, -1): "move_up"}.get((dx, dy), "wait")
                    actions.append({"bot": bid, "action": act})
            else:
                actions.append({"bot": bid, "action": "wait"})

        for bot in bots:
            self._prev_inv[bot["id"]] = list(bot.get("inventory", []))

        return {"actions": actions}


def run_velocity(recon_path: str, shelf_pref: dict | None = None) -> tuple[int, int, int, int]:
    """Run velocity strategy. Returns (score, orders, rounds, score_at_180)."""
    with open(recon_path) as f:
        recon = json.load(f)

    strategy = VelocityStrategy(recon, shelf_pref=shelf_pref)

    from Simulering.offline.simulator import Simulator
    sim = Simulator.from_recon_file(recon_path)

    state = sim.reset()
    score_180 = 0
    for r in range(sim.max_rounds):
        response = strategy(state.to_dict())
        state, game_over = sim.step(response.get("actions", []))
        if r == 179:
            score_180 = sim._score
        if game_over:
            break

    return sim._score, sim._orders_completed, sim._round, score_180


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--shelf-pref", help="JSON file with shelf preferences")
    args = parser.parse_args()

    shelf_pref = None
    if args.shelf_pref:
        with open(args.shelf_pref) as f:
            shelf_pref = json.load(f)

    t0 = time.time()
    score, orders, rounds, s180 = run_velocity(args.recon, shelf_pref)
    elapsed = time.time() - t0
    print(f"Score: {score}, Orders: {orders}, Rounds: {rounds}")
    print(f"Score@180: {s180} ({s180 / 180:.2f}/round)")
    print(f"Velocity: {score / max(rounds, 1):.2f}/round")
    print(f"Time: {elapsed:.1f}s")
