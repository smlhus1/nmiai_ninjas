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
        # Bot starting positions
        starts = recon_data.get("bot_start_positions", [])
        self._bot_start: Pos = tuple(starts[0]) if starts else self._drop_off
        self._n_bots: int = recon_data.get("bot_count", 1)
        self._bot_starts: list[Pos] = [tuple(s) for s in starts] if starts else [self._drop_off]
        # Pad to n_bots if recon data has fewer starts
        while len(self._bot_starts) < self._n_bots:
            self._bot_starts.append(self._drop_off)
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
        """Generate optimal game plan. Returns serializable dict.

        Always uses single-bot optimal sequence (v1 format), even for multi-bot.
        VRP partitioning across bots causes drop-off congestion that outweighs
        parallelism benefits. Instead, idle bots claim batches from the shared
        sequence, and remaining bots use reactive fallback for active items.
        """
        orders = self._recon["order_sequence"]
        order_plans: list[dict[str, Any]] = []

        # Use drop-off as start for all orders: in multi-bot, any bot may
        # claim a batch, and bots end at drop-off after delivery.
        current_start_pos = self._bot_start

        for i, order_info in enumerate(orders):
            items_required = order_info["items_required"]
            next_order = orders[i + 1] if i + 1 < len(orders) else None

            plan, end_pos = self._plan_order(items_required, next_order, current_start_pos)
            plan["order_index"] = i
            plan["order_id"] = order_info["id"]
            plan["items_required"] = items_required
            order_plans.append(plan)

            current_start_pos = self._drop_off

        result = {
            "version": 1,
            "fingerprint": self._recon["fingerprint"],
            "bot_count": self._recon["bot_count"],
            "drop_off": list(self._drop_off),
            "bot_start": list(self._bot_start),
            "order_plans": order_plans,
        }

        total_est = sum(p.get("estimated_rounds", 0) for p in order_plans)
        logger.info("ANALYZER: planned %d orders (%d bots, v%d), estimated %d total rounds",
                     len(order_plans), self._n_bots,
                     result["version"], total_est)

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

    # ---- Multi-bot VRP planning ----

    def _plan_order_multi(
        self,
        items_required: list[str],
        next_order: dict | None,
        bot_starts: list[Pos],
    ) -> tuple[dict[str, Any], list[Pos]]:
        """Plan optimal pickup+delivery for a multi-bot order.
        Partitions items across bots to minimize makespan.
        Returns (plan_dict, new_bot_starts)."""
        n_bots = len(bot_starts)

        # Build item candidates (same as single-bot)
        item_candidates: list[list[tuple[str, Pos, Pos]]] = []
        for item_type in items_required:
            shelves = self._shelf_lookup.get(item_type, [])
            candidates = [(item_type, s, p) for s, p in shelves]
            if candidates:
                item_candidates.append(candidates)

        if not item_candidates:
            empty_assignments = {
                str(b): {"batches": []} for b in range(n_bots)
            }
            return {
                "bot_assignments": empty_assignments,
                "estimated_rounds": 0,
            }, bot_starts

        n_items = len(item_candidates)

        # Search space: partitions × shelf combos × permutations per bot
        n_combos = 1
        for cands in item_candidates:
            n_combos *= len(cands)
        # Rough upper bound: n_bots^n_items partitions × combos × avg perms
        partition_count = n_bots ** n_items
        total_search = partition_count * n_combos

        if total_search <= MAX_BRUTE_FORCE:
            bot_sequences = self._brute_force_multi_bot(
                item_candidates, bot_starts
            )
        else:
            bot_sequences = self._greedy_multi_bot(
                item_candidates, bot_starts
            )

        # Build output with per-bot batches
        bot_assignments: dict[str, dict[str, Any]] = {}
        for bot_idx in range(n_bots):
            seq = bot_sequences.get(bot_idx, [])
            batches = self._split_batches(seq) if seq else []
            bot_assignments[str(bot_idx)] = {
                "batches": [
                    [
                        {"shelf_pos": list(shelf), "pickup_pos": list(pickup), "item_type": itype}
                        for itype, shelf, pickup in batch
                    ]
                    for batch in batches
                ]
            }

        # Estimate rounds as makespan
        estimated_rounds = self._calculate_makespan(bot_sequences, bot_starts)

        # After delivery, all bots end at drop-off
        new_starts = [self._drop_off] * n_bots

        logger.info(
            "ANALYZER: multi-bot order: %d items across %d bots, "
            "makespan=%d, items/bot=%s",
            n_items, n_bots, estimated_rounds,
            {b: len(s) for b, s in bot_sequences.items()},
        )

        return {
            "bot_assignments": bot_assignments,
            "estimated_rounds": estimated_rounds,
        }, new_starts

    def _brute_force_multi_bot(
        self,
        item_candidates: list[list[tuple[str, Pos, Pos]]],
        bot_starts: list[Pos],
    ) -> dict[int, list[tuple[str, Pos, Pos]]]:
        """Try all bot partitions × shelf combos × per-bot permutations."""
        n_items = len(item_candidates)
        n_bots = len(bot_starts)
        best_makespan = float("inf")
        best_assignment: dict[int, list[tuple[str, Pos, Pos]]] = {}

        # Enumerate all shelf combos first
        for combo in itertools.product(*item_candidates):
            # For each combo, try all bot assignments (which bot gets which item)
            for partition in itertools.product(range(n_bots), repeat=n_items):
                # Group items by bot
                bot_items: dict[int, list[tuple[str, Pos, Pos]]] = {
                    b: [] for b in range(n_bots)
                }
                for item_idx, bot_idx in enumerate(partition):
                    bot_items[bot_idx].append(combo[item_idx])

                # For each bot's items, find best permutation
                bot_sequences: dict[int, list[tuple[str, Pos, Pos]]] = {}
                total_makespan = 0

                for bot_idx in range(n_bots):
                    items = bot_items[bot_idx]
                    if not items:
                        bot_sequences[bot_idx] = []
                        continue

                    start = bot_starts[bot_idx]
                    # Try all permutations for this bot's items
                    best_bot_cost = float("inf")
                    best_bot_perm: list[tuple[str, Pos, Pos]] = items

                    if len(items) <= 6:  # factorial(6)=720, reasonable
                        for perm in itertools.permutations(items):
                            perm_list = list(perm)
                            batches = self._split_batches(perm_list)
                            cost = self._estimate_rounds(batches, start)
                            if cost < best_bot_cost:
                                best_bot_cost = cost
                                best_bot_perm = perm_list
                    else:
                        # Greedy for large item sets
                        best_bot_perm = self._greedy_sequence(items, start)
                        batches = self._split_batches(best_bot_perm)
                        best_bot_cost = self._estimate_rounds(batches, start)

                    bot_sequences[bot_idx] = best_bot_perm
                    total_makespan = max(total_makespan, best_bot_cost)

                if total_makespan < best_makespan:
                    best_makespan = total_makespan
                    best_assignment = dict(bot_sequences)

        return best_assignment

    def _greedy_multi_bot(
        self,
        item_candidates: list[list[tuple[str, Pos, Pos]]],
        bot_starts: list[Pos],
    ) -> dict[int, list[tuple[str, Pos, Pos]]]:
        """Greedy: assign each item to the bot that increases makespan least."""
        n_bots = len(bot_starts)
        bot_sequences: dict[int, list[tuple[str, Pos, Pos]]] = {
            b: [] for b in range(n_bots)
        }
        bot_costs: dict[int, int] = {b: 0 for b in range(n_bots)}

        # Sort items by distance from drop-off (furthest first = hardest to reach)
        scored_items: list[tuple[int, int, tuple[str, Pos, Pos]]] = []
        for idx, cands in enumerate(item_candidates):
            # Pick best shelf (closest to drop-off) for scoring
            best = min(cands, key=lambda c: self._path.distance(c[2], self._drop_off))
            score = self._path.distance(best[2], self._drop_off)
            scored_items.append((score, idx, best))
        scored_items.sort(reverse=True)  # Furthest first

        for _, item_idx, _ in scored_items:
            cands = item_candidates[item_idx]
            best_bot = -1
            best_new_makespan = float("inf")
            best_candidate: tuple[str, Pos, Pos] | None = None

            for bot_idx in range(n_bots):
                for candidate in cands:
                    # Simulate adding this item to this bot
                    trial = bot_sequences[bot_idx] + [candidate]
                    batches = self._split_batches(trial)
                    cost = self._estimate_rounds(batches, bot_starts[bot_idx])
                    new_makespan = max(
                        cost,
                        max((c for b, c in bot_costs.items() if b != bot_idx), default=0),
                    )
                    if new_makespan < best_new_makespan:
                        best_new_makespan = new_makespan
                        best_bot = bot_idx
                        best_candidate = candidate

            if best_candidate is not None and best_bot >= 0:
                bot_sequences[best_bot].append(best_candidate)
                batches = self._split_batches(bot_sequences[best_bot])
                bot_costs[best_bot] = self._estimate_rounds(
                    batches, bot_starts[best_bot]
                )

        return bot_sequences

    def _greedy_sequence(
        self,
        items: list[tuple[str, Pos, Pos]],
        start_pos: Pos,
    ) -> list[tuple[str, Pos, Pos]]:
        """Order items by greedy nearest-neighbor from start_pos."""
        remaining = list(items)
        result: list[tuple[str, Pos, Pos]] = []
        current_pos = start_pos

        while remaining:
            best_idx = 0
            best_dist = float("inf")
            for i, (_, _, pickup) in enumerate(remaining):
                d = self._path.distance(current_pos, pickup)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            chosen = remaining.pop(best_idx)
            result.append(chosen)
            current_pos = chosen[2]

        return result

    def _calculate_makespan(
        self,
        bot_sequences: dict[int, list[tuple[str, Pos, Pos]]],
        bot_starts: list[Pos],
    ) -> int:
        """Makespan = MAX over bots of completion time."""
        makespan = 0
        for bot_idx, seq in bot_sequences.items():
            if not seq:
                continue
            start = bot_starts[bot_idx] if bot_idx < len(bot_starts) else self._drop_off
            batches = self._split_batches(seq)
            cost = self._estimate_rounds(batches, start)
            makespan = max(makespan, cost)
        return makespan

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
