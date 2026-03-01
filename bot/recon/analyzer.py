"""
OfflinePlanner: brute-force optimal game plan from recon data.

Takes recon data (order sequence, shelf map) and a PathEngine to
compute the optimal pickup sequence for each order.

Key improvements over reactive planner:
- Multi-shelf search: considers ALL shelves per item type, not just closest
- Integrated shelf+permutation brute-force: finds globally optimal combo
- Position-aware: tracks bot position through entire plan (spawn → pickups → drop-off)
- Cross-order pipelining: pre-picks from next order if "on the way"
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import Any

from bot.models import Pos
from bot.engine.pathfinding import PathEngine

logger = logging.getLogger(__name__)

MAX_BRUTE_FORCE = 50000  # Max shelf combos × permutations before fallback to greedy
INVENTORY_CAP = 3
ON_THE_WAY_MARGIN = 4


class OfflinePlanner:
    """Generate optimal game plan from recon data + pathfinding."""

    def __init__(self, recon_data: dict, path_engine: PathEngine) -> None:
        self._recon = recon_data
        self._path = path_engine
        # shelf_lookup: type -> [(shelf_pos, pickup_pos)]
        self._shelf_lookup: dict[str, list[tuple[Pos, Pos]]] = {}
        self._drop_off: Pos = tuple(recon_data["drop_off"])
        # Bot starting position (for first order)
        starts = recon_data.get("bot_start_positions", [])
        self._bot_start: Pos = tuple(starts[0]) if starts else self._drop_off
        self._build_shelf_lookup()

    def _build_shelf_lookup(self) -> None:
        """Build lookup: item_type -> [(shelf_pos, best_pickup_pos)]."""
        shelf_map = self._recon["shelf_map"]
        for item_type, positions in shelf_map.items():
            entries: list[tuple[Pos, Pos]] = []
            for pos_list in positions:
                shelf_pos: Pos = tuple(pos_list)
                pickup = self._find_pickup_pos(shelf_pos)
                if pickup is not None:
                    entries.append((shelf_pos, pickup))
            self._shelf_lookup[item_type] = entries

    def _find_pickup_pos(self, shelf_pos: Pos) -> Pos | None:
        """Find best walkable cell adjacent to a shelf position."""
        candidates: list[tuple[Pos, int]] = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            pos = (shelf_pos[0] + dx, shelf_pos[1] + dy)
            d = self._path.distance(pos, self._drop_off)
            if d < 9999:
                candidates.append((pos, d))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def plan(self) -> dict:
        """Generate optimal game plan. Returns serializable dict."""
        orders = self._recon["order_sequence"]
        order_plans: list[dict[str, Any]] = []
        # Track where the bot will be at the start of each order
        current_start_pos = self._bot_start

        for i, order_info in enumerate(orders):
            items_required = order_info["items_required"]
            next_order = orders[i + 1] if i + 1 < len(orders) else None

            plan, end_pos = self._plan_order(items_required, next_order, current_start_pos)
            plan["order_index"] = i
            plan["order_id"] = order_info["id"]
            plan["items_required"] = items_required
            order_plans.append(plan)

            # After delivering, bot is at drop-off
            current_start_pos = self._drop_off

        result = {
            "fingerprint": self._recon["fingerprint"],
            "bot_count": self._recon["bot_count"],
            "drop_off": list(self._drop_off),
            "bot_start": list(self._bot_start),
            "order_plans": order_plans,
        }

        total_est = sum(p.get("estimated_rounds", 0) for p in order_plans)
        logger.info("ANALYZER: planned %d orders, estimated %d total rounds",
                     len(order_plans), total_est)

        return result

    def _plan_order(
        self,
        items_required: list[str],
        next_order: dict | None,
        start_pos: Pos,
    ) -> tuple[dict[str, Any], Pos]:
        """Plan optimal pickup sequence for a single order.
        Returns (plan_dict, end_position)."""
        # For each item, list ALL candidate shelves
        # item_candidates[i] = [(type, shelf_pos, pickup_pos), ...]
        item_candidates: list[list[tuple[str, Pos, Pos]]] = []
        for item_type in items_required:
            shelves = self._shelf_lookup.get(item_type, [])
            candidates = [(item_type, s, p) for s, p in shelves]
            if candidates:
                item_candidates.append(candidates)

        if not item_candidates:
            return {"pickup_sequence": [], "batches": [], "pre_picks": [], "estimated_rounds": 0}, start_pos

        n_items = len(item_candidates)

        # Count search space
        n_combos = 1
        for cands in item_candidates:
            n_combos *= len(cands)
        n_perms = math.factorial(n_items)
        total_search = n_combos * n_perms

        if total_search <= MAX_BRUTE_FORCE:
            best_sequence = self._brute_force_multi_shelf(item_candidates, start_pos)
        else:
            best_sequence = self._greedy_multi_shelf(item_candidates, start_pos)

        # Split into batches
        batches = self._split_batches(best_sequence)

        # Estimate total rounds from actual start position
        estimated_rounds = self._estimate_rounds(batches, start_pos)

        # Cross-order pipelining
        pre_picks: list[dict[str, Any]] = []
        if next_order and batches:
            pre_picks = self._find_pre_picks(batches[-1], next_order, start_pos)

        # Build output
        pickup_sequence = [
            {"shelf_pos": list(shelf), "pickup_pos": list(pickup), "item_type": itype}
            for itype, shelf, pickup in best_sequence
        ]

        # End position: drop-off after last delivery
        end_pos = self._drop_off

        return {
            "pickup_sequence": pickup_sequence,
            "batches": [
                [
                    {"shelf_pos": list(shelf), "pickup_pos": list(pickup), "item_type": itype}
                    for itype, shelf, pickup in batch
                ]
                for batch in batches
            ],
            "pre_picks": pre_picks,
            "estimated_rounds": estimated_rounds,
        }, end_pos

    def _brute_force_multi_shelf(
        self,
        item_candidates: list[list[tuple[str, Pos, Pos]]],
        start_pos: Pos,
    ) -> list[tuple[str, Pos, Pos]]:
        """Try all shelf combinations × all permutations. Return cheapest."""
        best_cost = float("inf")
        best_perm: list[tuple[str, Pos, Pos]] = []

        for combo in itertools.product(*item_candidates):
            for perm in itertools.permutations(combo):
                perm_list = list(perm)
                batches = self._split_batches(perm_list)
                cost = self._estimate_rounds(batches, start_pos)
                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm_list

        return best_perm

    def _greedy_multi_shelf(
        self,
        item_candidates: list[list[tuple[str, Pos, Pos]]],
        start_pos: Pos,
    ) -> list[tuple[str, Pos, Pos]]:
        """Greedy nearest-neighbor with best shelf per step."""
        remaining_indices = set(range(len(item_candidates)))
        result: list[tuple[str, Pos, Pos]] = []
        current_pos = start_pos

        while remaining_indices:
            best_candidate = None
            best_idx = -1
            best_dist = float("inf")

            for i in remaining_indices:
                for candidate in item_candidates[i]:
                    _, _, pickup = candidate
                    d = self._path.distance(current_pos, pickup)
                    if d < best_dist:
                        best_dist = d
                        best_candidate = candidate
                        best_idx = i

            if best_candidate is None:
                break
            remaining_indices.discard(best_idx)
            result.append(best_candidate)
            current_pos = best_candidate[2]

        return result

    def _split_batches(
        self,
        sequence: list[tuple[str, Pos, Pos]],
    ) -> list[list[tuple[str, Pos, Pos]]]:
        """Split sequence into batches of INVENTORY_CAP items.
        Ensures no batch has duplicate shelf positions — only 1 item exists
        per shelf at a time, so duplicates must go in separate batches."""
        batches: list[list[tuple[str, Pos, Pos]]] = []
        current_batch: list[tuple[str, Pos, Pos]] = []
        current_shelves: set[Pos] = set()

        for item in sequence:
            _, shelf_pos, _ = item
            if len(current_batch) >= INVENTORY_CAP or shelf_pos in current_shelves:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [item]
                current_shelves = {shelf_pos}
            else:
                current_batch.append(item)
                current_shelves.add(shelf_pos)

        if current_batch:
            batches.append(current_batch)

        return batches

    def _estimate_rounds(
        self,
        batches: list[list[tuple[str, Pos, Pos]]],
        start_pos: Pos,
    ) -> int:
        """Estimate total rounds for all batches from a given start position."""
        total = 0
        current_pos = start_pos

        for batch in batches:
            for _, _, pickup in batch:
                total += self._path.distance(current_pos, pickup)
                total += 1  # pick_up action
                current_pos = pickup

            total += self._path.distance(current_pos, self._drop_off)
            total += 1  # drop_off action
            current_pos = self._drop_off

        return total

    def _find_pre_picks(
        self,
        last_batch: list[tuple[str, Pos, Pos]],
        next_order: dict,
        start_pos: Pos,
    ) -> list[dict[str, Any]]:
        """Find items from next order that are 'on the way' during last batch delivery."""
        if len(last_batch) >= INVENTORY_CAP:
            return []

        spare = INVENTORY_CAP - len(last_batch)
        last_pickup = last_batch[-1][2] if last_batch else self._drop_off
        d_direct = self._path.distance(last_pickup, self._drop_off)

        pre_picks: list[dict[str, Any]] = []
        used_types: set[str] = set()

        for item_type in next_order.get("items_required", []):
            if item_type in used_types:
                continue
            if len(pre_picks) >= spare:
                break

            shelves = self._shelf_lookup.get(item_type, [])
            best_detour: tuple[Pos, Pos, int] | None = None
            for shelf_pos, pickup_pos in shelves:
                d_via = (
                    self._path.distance(last_pickup, pickup_pos)
                    + self._path.distance(pickup_pos, self._drop_off)
                )
                if d_via <= d_direct + ON_THE_WAY_MARGIN:
                    if best_detour is None or d_via < best_detour[2]:
                        best_detour = (shelf_pos, pickup_pos, d_via)

            if best_detour is not None:
                pre_picks.append({
                    "shelf_pos": list(best_detour[0]),
                    "pickup_pos": list(best_detour[1]),
                    "item_type": item_type,
                })
                used_types.add(item_type)

        return pre_picks
