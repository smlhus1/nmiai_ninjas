"""JobDispatcher — assigns multi-item jobs to bots.

Replaces TaskPlanner + Hungarian + RouteBuilder from V1.

Algorithm:
1. Advance existing jobs (detect pickups/deliveries via inventory changes)
2. Collect idle bots
3. Hungarian assignment of single items to idle bots
4. Extend single-item assignments to multi-item routes (greedy detour)
5. TSP-optimize pickup order within each job
"""

from __future__ import annotations

import logging
from collections import Counter
from itertools import permutations
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from bot.models import GameState, Bot, Item, Order, Pos
from bot.engine.pathfinding import PathEngine
from bot.engine.world_model import WorldModel
from bot.v3.job import Job, PickupStep, BotState
from bot.v3.pipeline import OrderPipeline, PipelinePhase

logger = logging.getLogger(__name__)


class JobDispatcher:
    """Assigns multi-item pickup-deliver jobs to bots."""

    def __init__(
        self,
        max_items_per_job: int = 3,
        order_completion_bonus: float = 5.0,
        detour_threshold: int = 6,
    ) -> None:
        self._max_items = max_items_per_job
        self._completion_bonus = order_completion_bonus
        self._detour_threshold = detour_threshold
        self._bot_states: dict[int, BotState] = {}
        self._job_counter = 0
        self._last_active_order_id: str | None = None

    def dispatch(
        self,
        state: GameState,
        path_engine: PathEngine,
        world: WorldModel,
        pipeline: OrderPipeline,
    ) -> dict[int, BotState]:
        """Main dispatch loop. Returns updated bot states."""
        self._sync_bots(state)
        self._handle_order_transition(state)
        self._advance_jobs(state, world)
        self._assign_jobs(state, path_engine, world, pipeline)
        return self._bot_states

    @property
    def bot_states(self) -> dict[int, BotState]:
        return self._bot_states

    def reset(self) -> None:
        """Reset all state for new game."""
        self._bot_states.clear()
        self._job_counter = 0
        self._last_active_order_id = None

    # --- Internal ---

    def _sync_bots(self, state: GameState) -> None:
        """Sync bot states with game state."""
        active_ids = {b.id for b in state.bots}
        self._bot_states = {
            k: v for k, v in self._bot_states.items() if k in active_ids
        }
        for bot in state.bots:
            if bot.id not in self._bot_states:
                self._bot_states[bot.id] = BotState(bot_id=bot.id)
            bs = self._bot_states[bot.id]
            bs.prev_inventory = bs.inventory
            bs.position = bot.position
            bs.inventory = bot.inventory

    def _handle_order_transition(self, state: GameState) -> None:
        """On order change: bots with inventory get delivery jobs, others cleared."""
        active = state.active_orders
        if not active:
            return
        current_id = active[0].id
        if self._last_active_order_id is not None and current_id != self._last_active_order_id:
            logger.info("ORDER TRANSITION: %s -> %s, reassigning jobs",
                        self._last_active_order_id, current_id)
            new_order = active[0]
            remaining_types = set(new_order.items_remaining)
            for bs in self._bot_states.values():
                matching = [inv for inv in bs.inventory if inv in remaining_types]
                if matching:
                    # Bot has matching items — send to drop-off
                    delivery = state.drop_off_zones[0] if state.drop_off_zones else state.drop_off
                    self._job_counter += 1
                    bs.job = Job(
                        job_id=f"j{self._job_counter}",
                        order_id=current_id,
                        pickups=[],
                        delivery_zone=delivery,
                        priority=0,
                        current_step=0,
                        assigned_bot=bs.bot_id,
                    )
                    logger.info("ORDER TRANSITION B%d: %d matching items, delivery job",
                                bs.bot_id, len(matching))
                else:
                    bs.job = None
        self._last_active_order_id = current_id

    def _advance_jobs(self, state: GameState, world: WorldModel) -> None:
        """Detect pickups and deliveries, advance job steps."""
        for bot in state.bots:
            bs = self._bot_states.get(bot.id)
            if not bs or not bs.job:
                continue

            job = bs.job
            prev = Counter(bs.prev_inventory)
            curr = Counter(bot.inventory)

            # Detect pickup: gained expected item type
            if not job.is_delivering:
                step = job.pickups[job.current_step]
                gained = curr - prev
                if gained.get(step.item_type, 0) > 0:
                    logger.info("JOB B%d: picked up %s (step %d/%d)",
                                bot.id, step.item_type,
                                job.current_step + 1, len(job.pickups))
                    job.current_step += 1

            # Detect delivery: inventory decreased at drop-off
            if job.is_delivering and bot.position in state.drop_off_zones:
                lost = prev - curr
                if sum(lost.values()) > 0:
                    logger.info("JOB B%d: delivered, job %s complete",
                                bot.id, job.job_id)
                    bs.job = None
                elif not bot.inventory:
                    # Inventory empty (all delivered or nothing to deliver)
                    logger.info("JOB B%d: inventory empty, job %s done",
                                bot.id, job.job_id)
                    bs.job = None

    def _assign_jobs(
        self,
        state: GameState,
        path_engine: PathEngine,
        world: WorldModel,
        pipeline: OrderPipeline,
    ) -> None:
        """Assign jobs to idle bots via Hungarian + greedy extension."""
        idle = [b for b in state.bots if self._bot_states[b.id].job is None]
        if not idle:
            return

        # Bots with inventory: send to drop-off ONLY if items match active order
        still_idle = []
        active = state.active_orders
        remaining_types = set(active[0].items_remaining) if active else set()
        for bot in idle:
            matching = [inv for inv in bot.inventory if inv in remaining_types]
            if matching:
                delivery = world.nearest_drop_off(bot.position, bot.id)
                self._job_counter += 1
                self._bot_states[bot.id].job = Job(
                    job_id=f"j{self._job_counter}",
                    order_id=active[0].id if active else "cleanup",
                    pickups=[],
                    delivery_zone=delivery,
                    priority=0,
                    current_step=0,
                    assigned_bot=bot.id,
                )
                logger.info("ASSIGN B%d: has %d matching items, delivery job",
                            bot.id, len(matching))
            else:
                still_idle.append(bot)
        idle = still_idle
        if not idle:
            return

        # Pick target order
        order = self._get_target_order(state, pipeline)
        if not order:
            return

        remaining = list(order.items_remaining)
        if not remaining:
            return

        # Subtract items already being worked on by assigned bots
        in_progress = Counter()
        for bs in self._bot_states.values():
            if bs.job and bs.job.order_id == order.id:
                for step in bs.job.pickups[bs.job.current_step:]:
                    in_progress[step.item_type] += 1
                # Also count items already in inventory heading to delivery
                if bs.job.is_delivering:
                    for inv_type in bs.inventory:
                        if inv_type in Counter(remaining):
                            in_progress[inv_type] += 1

        needed = Counter(remaining) - in_progress
        needed = +needed  # Remove zero/negative entries
        if not needed:
            # All items covered — try preview if allowed
            if pipeline.allow_preview:
                preview = state.preview_orders
                if preview:
                    order = preview[0]
                    remaining = list(order.items_remaining)
                    needed = Counter(remaining)
                else:
                    return
            else:
                return

        if not needed:
            return

        # Build item index
        items_by_type: dict[str, list[Item]] = {}
        for item in state.items:
            items_by_type.setdefault(item.type, []).append(item)

        # Expand needed into individual slots
        slots: list[str] = []  # Each slot = one item type instance
        for item_type, count in needed.items():
            if item_type in items_by_type:
                slots.extend([item_type] * count)

        if not slots or not idle:
            return

        # Step 1: Hungarian for single-item assignment
        n_bots = len(idle)
        n_slots = len(slots)
        cost = np.full((n_bots, n_slots), 9999.0)

        for i, bot in enumerate(idle):
            for j, item_type in enumerate(slots):
                items = items_by_type.get(item_type, [])
                best_d = 9999.0
                for item in items:
                    pp = world.best_pickup_position(bot.position, item.position)
                    if pp is None:
                        continue
                    d_pick = path_engine.distance(bot.position, pp)
                    d_del = path_engine.distance(pp, world.nearest_drop_off(pp, bot.id))
                    total = d_pick + 1 + d_del + 1
                    best_d = min(best_d, total)
                cost[i, j] = best_d

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_bots: list[int] = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 9998:
                continue
            bot = idle[r]
            item_type = slots[c]

            # Find nearest item of this type to this bot
            items = items_by_type.get(item_type, [])
            best_item: Optional[Item] = None
            best_pp: Optional[Pos] = None
            best_d = 9999
            for item in items:
                pp = world.best_pickup_position(bot.position, item.position)
                if pp is None:
                    continue
                d = path_engine.distance(bot.position, pp)
                if d < best_d:
                    best_d = d
                    best_item = item
                    best_pp = pp

            if best_item is None or best_pp is None:
                continue

            delivery = world.nearest_drop_off(best_pp, bot.id)
            step = PickupStep(item_type, best_item.position, best_pp)

            # Check order completion
            remaining_after = Counter(remaining)
            remaining_after[item_type] -= 1
            priority = 0 if sum(remaining_after.values()) == 0 else 1
            if order.status.value == "preview":
                priority = 2

            self._job_counter += 1
            job = Job(
                job_id=f"j{self._job_counter}",
                order_id=order.id,
                pickups=[step],
                delivery_zone=delivery,
                priority=priority,
                assigned_bot=bot.id,
            )
            self._bot_states[bot.id].job = job
            assigned_bots.append(bot.id)

        # Step 2: Extend to multi-item routes
        if self._max_items > 1:
            self._extend_routes(
                assigned_bots, state, path_engine, world, order, needed,
            )

        # Step 3: TSP-optimize route order
        self._optimize_route_order(assigned_bots, path_engine)

    def _get_target_order(
        self, state: GameState, pipeline: OrderPipeline,
    ) -> Optional[Order]:
        """Get the order to work on."""
        active = state.active_orders
        if active:
            return active[0]
        preview = state.preview_orders
        if preview:
            return preview[0]
        return None

    def _extend_routes(
        self,
        assigned_bots: list[int],
        state: GameState,
        path_engine: PathEngine,
        world: WorldModel,
        order: Order,
        needed: Counter,
    ) -> None:
        """Try to add extra items to single-item jobs."""
        # Track what's already assigned
        assigned_types: Counter = Counter()
        for bs in self._bot_states.values():
            if bs.job:
                for step in bs.job.pickups:
                    assigned_types[step.item_type] += 1

        unassigned = needed - assigned_types
        unassigned = +unassigned

        items_by_type: dict[str, list[Item]] = {}
        for item in state.items:
            items_by_type.setdefault(item.type, []).append(item)

        for bot_id in assigned_bots:
            bs = self._bot_states[bot_id]
            job = bs.job
            if not job or len(job.pickups) >= self._max_items:
                continue

            for item_type in list(unassigned.elements()):
                if len(job.pickups) >= self._max_items:
                    break

                items = items_by_type.get(item_type, [])
                if not items:
                    continue

                last_pp = job.pickups[-1].pickup_pos
                delivery = job.delivery_zone
                base_d = path_engine.distance(last_pp, delivery)

                best_item: Optional[Item] = None
                best_pp: Optional[Pos] = None
                best_detour = 9999
                for item in items:
                    pp = world.best_pickup_position(last_pp, item.position)
                    if pp is None:
                        continue
                    d_to = path_engine.distance(last_pp, pp)
                    d_from = path_engine.distance(pp, delivery)
                    detour = d_to + 1 + d_from - base_d
                    if detour < best_detour:
                        best_detour = detour
                        best_item = item
                        best_pp = pp

                if (best_item is not None and best_pp is not None
                        and best_detour <= self._detour_threshold):
                    job.pickups.append(PickupStep(
                        item_type, best_item.position, best_pp,
                    ))
                    unassigned[item_type] -= 1
                    logger.info("JOB B%d: extended with %s (detour=%d)",
                                bot_id, item_type, best_detour)

    def _optimize_route_order(
        self, assigned_bots: list[int], path_engine: PathEngine,
    ) -> None:
        """TSP-optimize pickup order for multi-item jobs."""
        for bot_id in assigned_bots:
            bs = self._bot_states[bot_id]
            job = bs.job
            if not job or len(job.pickups) <= 1:
                continue

            best_cost = 9999
            best_perm = None

            for perm in permutations(range(len(job.pickups))):
                cost = 0
                pos = bs.position
                for idx in perm:
                    pp = job.pickups[idx].pickup_pos
                    cost += path_engine.distance(pos, pp) + 1
                    pos = pp
                cost += path_engine.distance(pos, job.delivery_zone) + 1

                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm

            if best_perm and best_perm != tuple(range(len(job.pickups))):
                job.pickups = [job.pickups[i] for i in best_perm]
