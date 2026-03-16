"""
V2TaskPlanner: order-speed architecture.

Drop-in replacement for TaskPlanner. Same plan() and maintain() signatures.

Key differences from v1:
- OrderSolver: min-makespan assignment (brute-force partitioning)
- Event-driven re-planning (only re-plan on state changes)
- Drop-off sequencing modeled in solver
- PipelineManager for preview pre-picking
"""

from __future__ import annotations

import logging
from collections import Counter

from bot.models import GameState, Bot, Item, Order, Pos
from bot.engine.world_model import WorldModel
from bot.strategy.task import Task, TaskType, BotAssignment, Route, RouteStop
from bot.strategy.v2.order_solver import OrderSolver, OrderPlan
from bot.strategy.v2.pipeline import PipelineManager

logger = logging.getLogger(__name__)


class V2TaskPlanner:
    """
    V2 planner: order-speed optimized. Drop-in replacement for TaskPlanner.
    """

    def __init__(self) -> None:
        self._prev_inventory: dict[int, tuple[str, ...]] = {}
        self._blacklisted_items: dict[str, int] = {}
        self._stuck_pick_rounds: dict[int, int] = {}
        self._stuck_deliver_rounds: dict[int, int] = {}
        self._last_active_order_id: str | None = None
        self._last_active_remaining: int = -1
        self._solver = OrderSolver()
        self._pipeline = PipelineManager()
        self._config = None  # Set by coordinator
        self._last_state: GameState | None = None  # Set each round for demand scoring
        # Track bots that just delivered and have leftover non-matching inventory
        # to prevent EVICT loops (bot→drop_off→evict→back→evict...)
        self._post_deliver_cooldown: dict[int, int] = {}  # bot_id -> round cooldown expires
        # Recon-based future order knowledge
        self._future_orders: list[dict] = []  # order_sequence from recon
        self._current_order_index: int = 0  # tracks which order we're on
        # Spawn dispersal: pre-computed scatter targets for spawn-stacked bots
        self._scatter_targets: dict[int, Pos] = {}  # bot_id -> scatter position
        self._scatter_spawn: Pos | None = None  # detected spawn position
        # Genome shelf preference: item_type -> preferred shelf position index
        # Overrides nearest-item selection when set. Used by evolutionary search.
        self._shelf_preference: dict[str, int] = {}  # item_type -> shelf_index

    def set_future_orders(self, order_sequence: list[dict]) -> None:
        """Set the full order sequence from recon data for future knowledge."""
        self._future_orders = order_sequence
        self._current_order_index = 0
        logger.info("FUTURE: loaded %d orders for look-ahead", len(order_sequence))

    def _get_future_order_types(self, n_ahead: int = 3) -> list[tuple[set[str], list[str]]]:
        """Get item types needed by future orders (beyond current active + preview).

        Returns list of (type_set, items_required) for orders N+2, N+3, ..., N+2+n_ahead.
        N+0 = active, N+1 = preview (handled by game state), N+2+ = recon knowledge.
        """
        if not self._future_orders:
            return []
        # Start from current_order_index + 2 (skip active + preview)
        start = self._current_order_index + 2
        result = []
        for i in range(start, min(start + n_ahead, len(self._future_orders))):
            order = self._future_orders[i]
            items = order.get("items_required", [])
            result.append((set(items), items))
        return result

    def _build_demand_score(self, n_ahead: int = 8) -> Counter:
        """Count how many of the next N orders need each item type.

        Includes active + preview + future orders from recon.
        Higher count = more valuable to pick up now.
        """
        demand: Counter = Counter()
        state_ref = getattr(self, '_last_state', None)
        # Active order
        if state_ref and state_ref.active_orders:
            for t in set(state_ref.active_orders[0].items_remaining):
                demand[t] += 3  # Active order is 3x weight (immediate need)
        # Preview order
        if state_ref and state_ref.preview_orders:
            for t in set(state_ref.preview_orders[0].items_remaining):
                demand[t] += 2  # Preview is 2x weight (next up)
        # Future orders from recon
        if self._future_orders:
            start = self._current_order_index + 2
            for i in range(start, min(start + n_ahead, len(self._future_orders))):
                order = self._future_orders[i]
                for t in set(order.get("items_required", [])):
                    demand[t] += 1
        return demand

    def _sync_order_index(self, state) -> None:
        """Sync current_order_index with the active order from game state."""
        if not self._future_orders:
            return
        active = state.active_orders
        if not active:
            return
        active_id = active[0].id
        # Search for matching order in sequence
        for i, order in enumerate(self._future_orders):
            if order["id"] == active_id:
                if i != self._current_order_index:
                    logger.info("FUTURE: order index %d -> %d (active=%s)",
                               self._current_order_index, i, active_id)
                    self._current_order_index = i
                return
        # Fallback: match by items_required (in case IDs differ between runs)
        active_items = sorted(active[0].items_required)
        for i, order in enumerate(self._future_orders):
            if sorted(order.get("items_required", [])) == active_items:
                if i != self._current_order_index:
                    logger.info("FUTURE: order index %d -> %d (items match)",
                               self._current_order_index, i)
                    self._current_order_index = i
                return

    def blacklist_item(self, item_id: str, expiry_round: int) -> None:
        """Register an item as blacklisted until expiry_round."""
        self._blacklisted_items[item_id] = expiry_round

    def maintain(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        *,
        skip_route_abort: bool = False,
        skip_time_check: bool = False,
    ) -> dict[int, BotAssignment]:
        """Task lifecycle management (used by ReplayPlanner)."""
        state = world.state
        self._advance_routes(world, assignments)
        self._invalidate_stale(world, assignments, skip_time_check=skip_time_check)
        self._expire_blacklist(state.round)
        for bot in state.bots:
            self._prev_inventory[bot.id] = bot.inventory
        return assignments

    def plan(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
    ) -> dict[int, BotAssignment]:
        """
        Main planning entry point. Called once per round by Coordinator.

        Event-driven: only re-solve when state changed meaningfully.
        """
        state = world.state

        # Sync order index for future order lookahead
        self._sync_order_index(state)

        # Step 0: Lifecycle — advance routes, invalidate stale tasks
        self._advance_routes(world, assignments)
        self._invalidate_stale(world, assignments)
        self._expire_blacklist(state.round)

        # Step 1: Detect events
        events = self._detect_events(world, assignments)

        # Step 1.5: On order change with 5+ bots, clear PRE_PICK tasks for the old
        # preview order.  With many bots, stale PRE_PICK creates dead-weight that
        # blocks aisles.  For 3-bot maps the conveyor belt needs sticky PRE_PICK.
        if "order_changed" in events and len(state.bots) >= 5:
            for bot_id, a in assignments.items():
                if a.task and a.task.task_type == TaskType.PRE_PICK:
                    a.clear()

        # Step 2: Build claimed items set
        claimed_items = self._build_claimed_set(assignments)

        # NIGHTMARE STRATEGY: Queue-based for 20+ bots
        # Fill bots with diverse items → queue on bottom highway → deliver on match
        if len(state.bots) >= 20:
            self._nightmare_queue_strategy(world, assignments, claimed_items, events)
            # Spawn dispersal: only during initial stacking.
            # After bots disperse, they go directly to task targets.
            self._apply_spawn_dispersal(world, assignments, state)
            for bot in state.bots:
                self._prev_inventory[bot.id] = bot.inventory
            return assignments

        # Step 3: Endgame check
        if world.is_endgame() and not world.can_complete_active_order():
            self._plan_endgame(world, assignments, claimed_items)
            for bot in state.bots:
                self._prev_inventory[bot.id] = bot.inventory
            return assignments

        # Step 3.5: Clear drop-off blockers proactively
        self._clear_dropoff_blockers(world, assignments)

        # Step 4: If meaningful event or unassigned bots exist -> re-solve
        unassigned_exist = any(not a.has_task for a in assignments.values())
        if events or unassigned_exist:
            self._solve_and_assign(world, assignments, claimed_items)

        # Step 4.5: Stage bots with preview-matching items for auto-delivery
        self._stage_for_auto_delivery(world, assignments)

        # Step 4.6: Spread idle bots to future order shelves (10+ bots)
        if len(state.bots) >= 5:
            self._spread_idle_bots(world, assignments)

        # Step 5: Save inventory snapshots
        for bot in state.bots:
            self._prev_inventory[bot.id] = bot.inventory

        return assignments

    def _detect_events(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
    ) -> list[str]:
        """Detect state changes that warrant re-planning."""
        events = []
        state = world.state
        active = state.active_orders

        # Order changed?
        current_order_id = active[0].id if active else None
        if current_order_id != self._last_active_order_id:
            events.append("order_changed")
            self._last_active_order_id = current_order_id
            self._last_active_remaining = len(active[0].items_remaining) if active else 0

        # Items picked or delivered? (remaining count changed)
        if active:
            current_remaining = len(active[0].items_remaining)
            if current_remaining != self._last_active_remaining:
                events.append("items_changed")
                self._last_active_remaining = current_remaining

        # Any bot inventory changed?
        for bot in state.bots:
            prev = self._prev_inventory.get(bot.id, ())
            if bot.inventory != prev:
                events.append("inventory_changed")
                break

        if events:
            logger.debug("V2 events: %s", events)
        return events

    def _solve_and_assign(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        claimed_items: set[str],
    ) -> None:
        """Run OrderSolver for active order, assign results, then pipeline for idle."""
        state = world.state
        active = state.active_orders

        if not active:
            # No active order — all bots to preview pipeline
            all_ids = [b.id for b in state.bots]
            self._pipeline.assign_idle_bots(world, assignments, all_ids, claimed_items)
            return

        order = active[0]
        remaining_types = set(order.items_remaining)
        remaining_count = len(order.items_remaining)

        # Completion gate: when active order has <=2 items left and preview exists,
        # hold back deliverers for a few rounds to let pre-pick bots fill up.
        # This ensures order transition delivers pre-picked items via auto-delivery.
        gate_delay = getattr(self._config, 'gate_max_delay', 0) if self._config else 0
        gating = False
        if (gate_delay > 0 and remaining_count <= 2 and remaining_count > 0
                and state.preview_orders and len(state.bots) >= 5
                and state.rounds_remaining > 40):
            # Count how many preview items are already in inventories
            preview_items = state.preview_orders[0].items_remaining
            preview_in_inv = 0
            for bot in state.bots:
                for inv in bot.inventory:
                    if inv in set(preview_items):
                        preview_in_inv += 1
            # Gate if preview isn't fully pre-picked yet
            if preview_in_inv < len(preview_items) * 0.6:
                held = getattr(self, '_gate_rounds_held', 0)
                if held < gate_delay:
                    gating = True
                    self._gate_rounds_held = held + 1

        if not gating:
            self._gate_rounds_held = 0

        # Step 1: Bots with matching inventory should deliver
        # Skip bots on post-deliver cooldown (prevents EVICT pendling)
        for bot in state.bots:
            a = assignments.get(bot.id)
            if a and a.has_task:
                continue  # Already has a valid task
            if not bot.inventory:
                continue
            # Cooldown: bot recently delivered and got evicted — don't send back
            # Only applies if bot is NEAR drop_off (prevents pendling)
            cooldown = self._post_deliver_cooldown.get(bot.id, 0)
            if cooldown > state.round:
                d_to_drop = world.distance(bot.position, world.nearest_drop_off(bot.position, bot.id))
                if d_to_drop <= 5:
                    # Near drop_off with non-matching — IDLE to prevent OrderSolver re-assign
                    a = assignments[bot.id]
                    a.task = Task(task_type=TaskType.IDLE, target_pos=self._safe_idle_pos(bot.position, world))
                    a.path = None
                    continue
                else:
                    # Far enough away — clear cooldown
                    del self._post_deliver_cooldown[bot.id]
            matching = [inv for inv in bot.inventory if inv in remaining_types]
            if matching:
                # Completion gate: hold back if gating (let pre-pick fill up)
                if gating and remaining_count <= 1:
                    a = assignments[bot.id]
                    a.task = Task(task_type=TaskType.IDLE, target_pos=self._safe_idle_pos(bot.position, world))
                    a.path = None
                    continue
                a = assignments[bot.id]
                a.task = Task(task_type=TaskType.DELIVER, target_pos=world.nearest_drop_off(bot.position, bot.id))
                a.path = None
                # Clear cooldown on successful delivery assignment
                self._post_deliver_cooldown.pop(bot.id, None)

        # Step 2: Collect truly unassigned bots
        bots_to_assign: list[Bot] = []
        for bot in state.bots:
            a = assignments.get(bot.id)
            if not a or not a.has_task:
                bots_to_assign.append(bot)

        if not bots_to_assign:
            return

        # Step 2.5: Account for items already in-transit (other bots' inventory + routes)
        # Without this, solver assigns duplicates of items already being carried
        in_transit_types: list[str] = []
        for bot in state.bots:
            if bot.id in {b.id for b in bots_to_assign}:
                continue  # Unassigned bot — solver handles its inventory
            a = assignments.get(bot.id)
            if a and a.has_task:
                # Items in inventory heading to drop-off
                for inv_type in bot.inventory:
                    in_transit_types.append(inv_type)
                # Items in remaining route stops (being picked up)
                if a.route:
                    for stop in a.route.stops[a.route_step:]:
                        in_transit_types.append(stop.item_type)

        # Step 3: Solve assignment for unassigned bots
        # Conveyor belt (merge active+preview): only for ≤3 bots.
        # For 10+ bots: conveyor belt fills bots with preview items they can't auto-deliver
        # (auto-delivery only works for the delivering bot, not all bots).
        # For 5 bots: also harmful — dead weight blocks delivery lanes.
        if len(state.bots) <= 3:
            preview_order = state.preview_orders[0] if state.preview_orders else None
        else:
            preview_order = None
        logger.debug("R%d solver: %d bots (%s), order=%s remaining=%d",
                    state.round, len(bots_to_assign),
                    [b.id for b in bots_to_assign],
                    order.id[:8], len(order.items_remaining))
        plan = self._solver.solve(
            world, bots_to_assign, order, claimed_items, state.round,
            in_transit_types=in_transit_types,
            preview_order=preview_order,
        )

        # Apply OrderPlan -> set Task + Route per bot
        self._apply_plan(world, assignments, plan, claimed_items)

        # Step 4: Pipeline — assign idle bots to preview pre-picking
        idle_ids = list(plan.idle_bots)
        for bot in state.bots:
            if bot.id not in plan.bot_plans:
                if not assignments[bot.id].has_task:
                    idle_ids.append(bot.id)

        estimated_completion = plan.makespan
        self._pipeline.assign_idle_bots(
            world, assignments, idle_ids, claimed_items, estimated_completion,
        )

        # Step 5: Bots still unassigned with inventory — stay in place
        # (parking is too far on large maps, staging near drop_off causes gridlock)
        for bot in state.bots:
            a = assignments.get(bot.id)
            if a and not a.has_task and bot.inventory:
                a.task = Task(task_type=TaskType.IDLE, target_pos=self._safe_idle_pos(bot.position, world))
                a.path = None

    def _apply_plan(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        plan: OrderPlan,
        claimed_items: set[str],
    ) -> None:
        """Convert OrderPlan into Task/Route assignments.

        Conveyor belt: pickups marked as preview get TaskType.PRE_PICK,
        active pickups get TaskType.PICK_UP. Bot picks all items in sequence,
        then delivers. Active items auto-deliver on drop_off, preview items
        auto-deliver when preview becomes active (order transition).
        """
        state = world.state

        for bot_id, bot_plan in plan.bot_plans.items():
            a = assignments.get(bot_id)
            if a is None:
                continue

            bot = state.get_bot(bot_id)
            if not bot_plan.pickups:
                # Delivery only (items already in inventory)
                a.task = Task(
                    task_type=TaskType.DELIVER,
                    target_pos=world.nearest_drop_off(bot.position, bot_id) if bot else state.drop_off,
                )
                a.route = None
                a.route_step = 0
                a.path = None
                continue

            preview_set = set(bot_plan.preview_indices)

            # Build route from pickups
            stops = []
            for item, pickup_pos in bot_plan.pickups:
                stops.append(RouteStop(
                    item_id=item.id,
                    item_type=item.type,
                    item_pos=item.position,
                    pickup_pos=pickup_pos,
                ))
                claimed_items.add(item.id)

            # First stop task type depends on whether it's a preview item
            first_is_preview = 0 in preview_set
            first_stop = stops[0]
            a.task = Task(
                task_type=TaskType.PRE_PICK if first_is_preview else TaskType.PICK_UP,
                target_pos=first_stop.pickup_pos,
                item_id=first_stop.item_id,
                item_type=first_stop.item_type,
                item_pos=first_stop.item_pos,
                order_id=plan.preview_order_id if first_is_preview else plan.order_id,
            )
            a.path = None

            if len(stops) > 1:
                # Compute total route cost
                pos = state.get_bot(bot_id).position
                cost = 0
                for stop in stops:
                    cost += world.distance(pos, stop.pickup_pos)
                    cost += 1  # pick action
                    pos = stop.pickup_pos
                cost += world.distance(pos, world.nearest_drop_off(pos, bot_id)) + 1

                a.route = Route(
                    stops=stops,
                    order_id=plan.order_id,
                    total_cost=cost,
                    preview_stop_indices=preview_set,
                )
                a.route_step = 0
            else:
                a.route = None
                a.route_step = 0

            n_preview = len(preview_set)
            n_active = len(bot_plan.pickups) - n_preview
            logger.debug("V2 Bot %d: %d active + %d preview pickups, arrival=%d, delivery=%d",
                         bot_id, n_active, n_preview,
                         bot_plan.estimated_arrival, bot_plan.delivery_round)

    # --- Lifecycle methods (ported from v1, simplified) ---

    def _advance_routes(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
    ) -> None:
        """Advance route progress for bots that have picked up their current stop."""
        state = world.state
        current_item_ids = {item.id for item in state.items}

        for bot_id, assignment in assignments.items():
            if not assignment.route:
                continue

            bot = state.get_bot(bot_id)
            if bot is None:
                continue

            current_stop = assignment.current_route_stop
            if current_stop is None:
                # All stops completed -> DELIVER
                assignment.task = Task(
                    task_type=TaskType.DELIVER,
                    target_pos=world.nearest_drop_off(bot.position, bot_id),
                )
                assignment.route = None
                assignment.route_step = 0
                assignment.path = None
                continue

            # Detect pickup via inventory change
            if current_stop.item_id.startswith("camp_"):
                picked_up = self._item_picked_up(
                    bot_id, current_stop.item_type, bot.inventory
                )
            else:
                picked_up = (
                    current_stop.item_id not in current_item_ids
                    or self._item_picked_up(bot_id, current_stop.item_type, bot.inventory)
                )

            if picked_up:
                assignment.route_step += 1
                next_stop = assignment.current_route_stop
                if next_stop is None:
                    assignment.task = Task(
                        task_type=TaskType.DELIVER,
                        target_pos=world.nearest_drop_off(bot.position, bot_id),
                    )
                    assignment.route = None
                    assignment.route_step = 0
                    assignment.path = None
                else:
                    is_preview_stop = assignment.route_step in assignment.route.preview_stop_indices
                    # For preview stops, use the preview order ID stored on the route
                    # (route.order_id is the active order ID)
                    stop_order_id = assignment.route.order_id
                    if is_preview_stop:
                        # Find preview order ID from state
                        preview_orders = world.state.preview_orders
                        if preview_orders:
                            stop_order_id = preview_orders[0].id
                    assignment.task = Task(
                        task_type=TaskType.PRE_PICK if is_preview_stop else TaskType.PICK_UP,
                        target_pos=next_stop.pickup_pos,
                        item_id=next_stop.item_id,
                        item_type=next_stop.item_type,
                        item_pos=next_stop.item_pos,
                        order_id=stop_order_id,
                    )
                    assignment.path = None
                    logger.debug("V2 Bot %d route advanced to step %d: %s (%s)",
                                 bot_id, assignment.route_step, next_stop.item_type,
                                 "preview" if is_preview_stop else "active")

    def _invalidate_stale(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        skip_time_check: bool = False,
    ) -> None:
        """Remove tasks that are no longer valid."""
        state = world.state
        current_item_ids = {item.id for item in state.items}
        active_order_ids = {o.id for o in state.active_orders}
        # Track occupied dead-weight parking spots to spread bots
        deadweight_occupied: set[Pos] = {b.position for b in state.bots}

        for bot_id, assignment in assignments.items():
            task = assignment.task
            if task is None:
                continue

            bot = state.get_bot(bot_id)
            if bot is None:
                assignment.clear()
                continue

            # Route item re-resolution (items respawn with new IDs)
            if assignment.route:
                remaining_stops = assignment.route.stops[assignment.route_step:]
                for stop in remaining_stops:
                    if stop.item_id not in current_item_ids and not stop.item_id.startswith("camp_"):
                        for item in state.items:
                            if item.position == stop.item_pos and item.type == stop.item_type:
                                stop.item_id = item.id
                                break

                remaining_stops = [
                    s for s in remaining_stops
                    if s.item_id in current_item_ids or s.item_id.startswith("camp_")
                ]
                if not remaining_stops:
                    if bot.inventory:
                        assignment.task = Task(task_type=TaskType.DELIVER, target_pos=world.nearest_drop_off(bot.position, bot_id))
                    else:
                        assignment.clear()
                    assignment.route = None
                    assignment.route_step = 0
                    assignment.path = None
                    continue
                else:
                    original_count = len(assignment.route.stops) - assignment.route_step
                    if len(remaining_stops) < original_count:
                        assignment.route.stops = (
                            assignment.route.stops[:assignment.route_step] + remaining_stops
                        )
                    current_stop = assignment.current_route_stop
                    if current_stop and task.item_id != current_stop.item_id:
                        assignment.task = Task(
                            task_type=TaskType.PICK_UP,
                            target_pos=current_stop.pickup_pos,
                            item_id=current_stop.item_id,
                            item_type=current_stop.item_type,
                            item_pos=current_stop.item_pos,
                            order_id=assignment.route.order_id,
                        )
                        assignment.path = None

            # PICK_UP / PRE_PICK: detect successful pick
            if task.task_type in (TaskType.PICK_UP, TaskType.PRE_PICK):
                if not assignment.route and task.item_type:
                    if self._item_picked_up(bot_id, task.item_type, bot.inventory):
                        self._stuck_pick_rounds.pop(bot_id, None)
                        assignment.clear()
                        continue

                # Full inventory
                if len(bot.inventory) >= 3:
                    self._stuck_pick_rounds.pop(bot_id, None)
                    if self._has_matching_items(bot, world):
                        assignment.task = Task(task_type=TaskType.DELIVER, target_pos=world.nearest_drop_off(bot.position, bot_id))
                    else:
                        # Stay in place — don't waste time traveling to parking (30+ rounds)
                        # or staging near drop_off (gridlock risk).
                        # When order changes, planner reassigns to DELIVER.
                        assignment.task = Task(task_type=TaskType.IDLE, target_pos=self._safe_idle_pos(bot.position, world))
                    assignment.route = None
                    assignment.route_step = 0
                    assignment.path = None
                    continue

                # Stuck picking detection
                prev_inv = self._prev_inventory.get(bot_id, ())
                if bot.inventory == prev_inv and task.target_pos:
                    d_to_target = world.distance(bot.position, task.target_pos)
                    if d_to_target <= 1:
                        stuck = self._stuck_pick_rounds.get(bot_id, 0) + 1
                        self._stuck_pick_rounds[bot_id] = stuck
                        if stuck >= 5:
                            logger.info("V2 Bot %d: stuck picking %s for %d rounds, blacklisting",
                                        bot_id, task.item_id, stuck)
                            if task.item_id:
                                expiry = getattr(self._config, 'blacklist_expiry_rounds', 8) if self._config else 8
                                self._blacklisted_items[task.item_id] = state.round + expiry
                            assignment.clear()
                            self._stuck_pick_rounds.pop(bot_id, None)
                            continue
                    else:
                        self._stuck_pick_rounds.pop(bot_id, None)
                else:
                    self._stuck_pick_rounds.pop(bot_id, None)

            if task.task_type == TaskType.PICK_UP:
                # Item gone
                if (task.item_id and task.item_id not in current_item_ids
                        and not task.item_id.startswith("camp_")):
                    assignment.clear()
                # Time feasibility
                elif not skip_time_check and task.item_pos and not world.can_complete_trip(bot, task.item_pos):
                    assignment.clear()
                # Type no longer needed by active order
                elif (task.order_id and task.item_type and state.active_orders
                      and task.order_id == state.active_orders[0].id):
                    remaining_types = Counter(state.active_orders[0].items_remaining)
                    if remaining_types.get(task.item_type, 0) <= 0:
                        assignment.clear()

            elif task.task_type == TaskType.PRE_PICK:
                if task.item_id and task.item_id not in current_item_ids:
                    assignment.clear()
                elif task.order_id and task.order_id in active_order_ids:
                    assignment.clear()

            elif task.task_type == TaskType.DELIVER:
                if not bot.inventory:
                    assignment.clear()
                    self._stuck_deliver_rounds.pop(bot_id, None)
                elif not self._has_matching_items(bot, world) and not world.is_endgame():
                    # Non-matching inventory: park out of the way
                    # Set cooldown to prevent EVICT loop
                    self._post_deliver_cooldown[bot_id] = state.round + 3
                    # For 10+ bots: spread to shelf-adjacent spots away from delivery lanes
                    if len(state.bots) >= 10:
                        park_pos = self._find_deadweight_parking(bot, world, deadweight_occupied)
                        deadweight_occupied.add(park_pos)
                    else:
                        park_pos = bot.position
                    assignment.task = Task(task_type=TaskType.IDLE, target_pos=park_pos)
                    assignment.navigation_override = None
                    assignment.path = None
                    self._stuck_deliver_rounds.pop(bot_id, None)
                elif bot.position in state.drop_off_zones:
                    prev_inv = self._prev_inventory.get(bot_id)
                    if prev_inv == bot.inventory:
                        if not self._has_matching_items(bot, world):
                            # Just delivered, but leftover non-matching
                            self._post_deliver_cooldown[bot_id] = state.round + 3
                            assignment.clear()
                            self._stuck_deliver_rounds.pop(bot_id, None)
                        else:
                            rounds_stuck = self._stuck_deliver_rounds.get(bot_id, 0) + 1
                            self._stuck_deliver_rounds[bot_id] = rounds_stuck
                            if rounds_stuck >= 2:
                                assignment.clear()
                                self._stuck_deliver_rounds.pop(bot_id, None)
                    else:
                        self._stuck_deliver_rounds.pop(bot_id, None)

    def _plan_endgame(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        claimed_items: set[str],
    ) -> None:
        """Endgame: maximize items delivered before time runs out.

        Two phases:
        1. If order completable: still use normal solver (order bonus > item count)
        2. Otherwise: deliver any matching inventory, pick nearest deliverable item
        """
        state = world.state
        active_types = set(state.active_orders[0].items_remaining) if state.active_orders else set()

        # Phase 1: Force ALL bots with matching inventory to deliver immediately
        # (don't wait for solver — every round counts in endgame)
        for bot_id, assignment in assignments.items():
            bot = state.get_bot(bot_id)
            if bot is None:
                continue
            if bot.inventory and any(inv in active_types for inv in bot.inventory):
                nearest_drop = world.nearest_drop_off(bot.position, bot_id)
                d_to_drop = world.distance(bot.position, nearest_drop)
                if d_to_drop + 1 <= world.rounds_remaining:
                    assignment.task = Task(task_type=TaskType.DELIVER, target_pos=nearest_drop)
                    assignment.path = None
                    assignment.navigation_override = None

        # Phase 2: Unassigned bots pick nearest deliverable item
        for bot_id, assignment in assignments.items():
            if assignment.has_task:
                continue

            bot = state.get_bot(bot_id)
            if bot is None:
                continue

            if bot.inventory:
                if any(inv in active_types for inv in bot.inventory):
                    assignment.task = Task(task_type=TaskType.DELIVER, target_pos=world.nearest_drop_off(bot.position, bot.id))
                    assignment.path = None
                    continue
                elif len(bot.inventory) >= 3:
                    # Full with non-matching — park (never in one-way aisle)
                    assignment.task = Task(task_type=TaskType.IDLE, target_pos=self._safe_idle_pos(bot.position, world))
                    continue

            best_task = None
            best_cost = 9999

            for item in state.items:
                if item.id in claimed_items:
                    continue
                if active_types and item.type not in active_types:
                    continue
                pickup_pos = world.best_pickup_position(bot.position, item.position)
                if pickup_pos is None:
                    continue
                d_pick = world.distance(bot.position, pickup_pos)
                d_drop = world.distance(pickup_pos, world.nearest_drop_off(pickup_pos, bot.id))
                total = d_pick + d_drop + 2
                if total > world.rounds_remaining:
                    continue
                # Prefer items with lowest total trip cost (not just nearest)
                if total < best_cost:
                    best_cost = total
                    best_task = Task(
                        task_type=TaskType.PICK_UP,
                        target_pos=pickup_pos,
                        item_id=item.id,
                        item_type=item.type,
                        item_pos=item.position,
                    )

            if best_task:
                assignment.task = best_task
                assignment.path = None
                if best_task.item_id:
                    claimed_items.add(best_task.item_id)
            else:
                assignment.task = Task(task_type=TaskType.IDLE, target_pos=self._safe_idle_pos(bot.position, world))

    @staticmethod
    def _find_cross_corridors(grid) -> list[int]:
        """Find cross-corridor y-values (rows where >=60% of cells are walkable).

        Returns sorted list of y-values, bottommost last.
        """
        cross_ys = []
        for y in range(grid.height):
            walkable = sum(1 for x in range(grid.width) if grid.is_walkable((x, y)))
            if walkable >= grid.width * 0.6:
                cross_ys.append(y)
        return sorted(cross_ys)

    @staticmethod
    def _safe_idle_pos(bot_pos: Pos, world: WorldModel) -> Pos:
        """Return a safe position for IDLE parking — never in a one-way aisle.

        Bots parked in one-way aisles are immovable (AT target = won't yield in PIBT)
        and block all traffic behind them permanently.
        Only applies for 10+ bots — fewer bots rarely have this deadlock.

        Uses one-way-aware distance to find reachable non-one-way positions,
        not raw BFS (which ignores one-way and finds unreachable targets).
        """
        if len(world.state.bots) < 10:
            return bot_pos  # Few bots: not worth the repositioning cost
        one_way = getattr(world.path, '_one_way', {})
        if bot_pos not in one_way:
            return bot_pos  # Current position is safe
        # Find nearest reachable non-one-way position using one-way-aware distance
        grid = world._grid
        best_pos = bot_pos
        best_d = 9999
        # Check cross-corridor positions (these are always non-one-way)
        # and any other non-one-way walkable positions within reasonable range
        for x in range(grid.width):
            for y in range(grid.height):
                pos = (x, y)
                if not grid.is_walkable(pos) or pos in one_way:
                    continue
                d = world.distance(bot_pos, pos)
                if d < best_d:
                    best_d = d
                    best_pos = pos
        return best_pos


    def _clear_dropoff_blockers(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
    ) -> None:
        """Move bots away from drop_off if they have no matching inventory.

        This prevents the EVICT pendling loop where coordinator evicts a bot
        to staging, then next round planner sends it back to drop_off.
        By handling this IN the planner, we give the bot a real task (pickup)
        instead of just navigation_overriding it to staging.
        """
        state = world.state
        active = state.active_orders
        if not active:
            return
        remaining_types = set(active[0].items_remaining)

        for bot in state.bots:
            if bot.position not in state.drop_off_zones:
                continue
            if not bot.inventory:
                continue
            if any(inv in remaining_types for inv in bot.inventory):
                continue  # Has matching items — let it deliver

            # Bot is at drop_off with non-matching inventory — clear its task
            # so solver can give it a pickup assignment
            a = assignments.get(bot.id)
            if a and a.has_task:
                if a.task.task_type != TaskType.IDLE:
                    a.clear()
                    # Set cooldown to prevent immediate re-DELIVER
                    self._post_deliver_cooldown[bot.id] = state.round + 3
                    logger.debug("Cleared drop_off blocker B%d (non-matching inv)", bot.id)
                else:
                    # IDLE bot ON drop-off with inventory — must evacuate
                    # Stay in place = permanent blockage for all deliverers
                    a.clear()
                    self._post_deliver_cooldown[bot.id] = state.round + 5
                    logger.debug("Cleared IDLE drop_off blocker B%d (non-matching inv)", bot.id)

    def _stage_for_auto_delivery(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
    ) -> None:
        """Send bots with preview-matching inventory to drop_off for auto-delivery.

        Only triggers when active order is VERY close to done (≤2 items left)
        and ONLY for bots that are already idle/parked (not actively picking).
        Bots with active tasks keep working — we only redirect bots that have
        nothing better to do anyway.
        """
        state = world.state
        active = state.active_orders
        preview = state.preview_orders
        if not active or not preview:
            return

        remaining = active[0].items_remaining
        if len(remaining) > 2:
            return  # Only stage when order is nearly done

        preview_types = set(preview[0].items_required)

        for bot in state.bots:
            if not bot.inventory:
                continue
            matching_preview = [inv for inv in bot.inventory if inv in preview_types]
            if not matching_preview:
                continue
            # Don't redirect bots delivering active items
            active_remaining = set(remaining)
            if any(inv in active_remaining for inv in bot.inventory):
                continue

            a = assignments.get(bot.id)
            if a is None:
                continue

            drop_off = world.nearest_drop_off(bot.position, bot.id)
            d_to_drop = world.distance(bot.position, drop_off)
            if d_to_drop == 0:
                continue

            # Only redirect IDLE/parked bots or bots with PRE_PICK that are full
            if a.task and a.task.task_type == TaskType.IDLE:
                a.task = Task(task_type=TaskType.DELIVER, target_pos=drop_off)
                a.route = None
                a.route_step = 0
                a.path = None
                logger.debug("Auto-delivery staging: B%d idle -> drop_off (d=%d, preview=%s)",
                             bot.id, d_to_drop, matching_preview)
            elif (a.task and a.task.task_type == TaskType.PRE_PICK
                  and len(bot.inventory) >= 3):
                # Full with preview items, no more to pick — redirect to drop_off
                a.task = Task(task_type=TaskType.DELIVER, target_pos=drop_off)
                a.route = None
                a.route_step = 0
                a.path = None
                logger.debug("Auto-delivery staging: B%d full PRE_PICK -> drop_off (d=%d, preview=%s)",
                             bot.id, d_to_drop, matching_preview)

    def _spread_idle_bots(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
    ) -> None:
        """Pre-position IDLE bots near shelves needed by upcoming orders.

        Uses recon future order knowledge to send idle bots to pickup positions
        adjacent to shelves that will be needed soon. When the order activates,
        the bot is already 1 step from the item — saving 10-20 rounds of travel.

        Falls back to aisle intersection spread if no future order data available.
        """
        state = world.state

        # Collect empty IDLE bots
        idle_bots: list[int] = []
        for bot in state.bots:
            a = assignments.get(bot.id)
            if a and a.task and a.task.task_type == TaskType.IDLE and not bot.inventory:
                idle_bots.append(bot.id)

        if not idle_bots:
            return

        # Collect positions already targeted by active bots
        occupied: set[Pos] = set()
        targeted: set[Pos] = set()
        for bot in state.bots:
            a = assignments.get(bot.id)
            if a and a.has_task and a.task.task_type != TaskType.IDLE:
                occupied.add(bot.position)
                if a.task.target_pos:
                    targeted.add(a.task.target_pos)

        # Strategy: use future order knowledge to send idle bots to PRE_PICK
        # items from N+2, N+3 orders. When those orders activate, items are
        # already in inventory → instant delivery instead of 20-round cycle.
        future_orders = self._get_future_order_types(n_ahead=5)

        # Build list of (item_type, item_obj, pickup_pos) for future orders
        future_pickups: list[tuple[str, object, Pos]] = []
        if future_orders:
            claimed_types: Counter = Counter()
            # Count what's already being picked/pre-picked
            for bot in state.bots:
                a = assignments.get(bot.id)
                if a and a.has_task and a.task.task_type in (TaskType.PICK_UP, TaskType.PRE_PICK):
                    if a.task.item_type:
                        claimed_types[a.task.item_type] += 1
            # Count inventory across all bots
            for bot in state.bots:
                for t in bot.inventory:
                    claimed_types[t] += 1

            for type_set, items_list in future_orders:
                need = Counter(items_list)
                for item_type, count in need.items():
                    # How many more do we need beyond what's claimed?
                    extra_needed = count - claimed_types.get(item_type, 0)
                    if extra_needed <= 0:
                        continue
                    # Find items on map to pick
                    for item in state.items:
                        if item.type != item_type:
                            continue
                        if item.id in self._claimed_items:
                            continue
                        pickup = world.best_pickup_position(item.position, item.position)
                        if pickup and pickup not in targeted and pickup not in occupied:
                            future_pickups.append((item_type, item, pickup))
                            targeted.add(pickup)
                            claimed_types[item_type] += 1
                            extra_needed -= 1
                            if extra_needed <= 0:
                                break

        # Assign idle bots to PRE_PICK future items
        assigned_targets: set[Pos] = set()
        for bot_id in idle_bots:
            bot = state.get_bot(bot_id)
            if bot is None or len(bot.inventory) >= 3:
                continue

            best_pickup = None
            best_dist = float('inf')
            best_idx = -1
            for idx, (item_type, item_obj, pickup_pos) in enumerate(future_pickups):
                if pickup_pos in assigned_targets:
                    continue
                d = world.distance(bot.position, pickup_pos)
                if d < best_dist and d < 9999:
                    best_dist = d
                    best_pickup = (item_type, item_obj, pickup_pos)
                    best_idx = idx

            if best_pickup:
                item_type, item_obj, pickup_pos = best_pickup
                a = assignments[bot_id]
                a.task = Task(
                    task_type=TaskType.PRE_PICK,
                    target_pos=pickup_pos,
                    item_id=item_obj.id,
                    item_type=item_type,
                    item_pos=item_obj.position,
                )
                a.path = None
                assigned_targets.add(pickup_pos)
                occupied.add(pickup_pos)
                self._claimed_items.add(item_obj.id)

    @staticmethod
    def _find_deadweight_parking(
        bot: "Bot",
        world: WorldModel,
        occupied: set[Pos] | None = None,
    ) -> Pos:
        """Find parking for dead-weight bots (non-matching inventory).

        For 20+ bots: ONLY park on bottom highways (no single-width corridors).
        For fewer bots: shelf-adjacent positions away from delivery lanes.

        CRITICAL: Never park in one-way aisles — parked bots there are
        immovable by PIBT (AT target = won't yield) and block all traffic.
        """
        grid = world._grid
        state = world.state

        bot_positions: set[Pos] = occupied or {b.position for b in state.bots}

        # For 20+ bots: highway-only parking
        if len(state.bots) >= 20:
            sorted_cross = V2TaskPlanner._find_cross_corridors(grid)
            highway_ys = [sorted_cross[-1]]  # Only bottommost row = parking lane

            best_pos = bot.position
            best_d = 9999
            for y in highway_ys:
                for x in range(3, grid.width - 1, 2):
                    pos = (x, y)
                    if not grid.is_walkable(pos) or pos in bot_positions:
                        continue
                    if world.distance(pos, world.nearest_drop_off(pos, bot.id)) <= 3:
                        continue
                    d = world.distance(bot.position, pos)
                    if d < best_d:
                        best_d = d
                        best_pos = pos
            return best_pos

        # For fewer bots: shelf-adjacent parking
        active_types: set[str] = set()
        if state.active_orders:
            active_types = set(state.active_orders[0].items_remaining)

        active_shelves: set[Pos] = set()
        shelf_set: set[Pos] = set()
        for item in state.items:
            shelf_set.add(item.position)
            if item.type in active_types:
                active_shelves.add(item.position)

        # One-way positions are off-limits — parked bots there block traffic permanently
        one_way = getattr(world.path, '_one_way', {})

        best_pos = bot.position
        best_score = (9999, 0)

        for x in range(grid.width):
            for y in range(grid.height):
                pos = (x, y)
                if not grid.is_walkable(pos) or pos in bot_positions:
                    continue
                if pos in one_way:
                    continue  # Never park in one-way aisles
                adj_shelf = any(
                    (x + dx, y + dy) in shelf_set
                    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
                )
                adj_active = any(
                    (x + dx, y + dy) in active_shelves
                    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
                )
                if not adj_shelf or adj_active:
                    continue
                nearest_drop = world.nearest_drop_off(pos, bot.id)
                if abs(x - nearest_drop[0]) <= 2:
                    continue
                d_travel = world.distance(bot.position, pos)
                if d_travel >= 9999:
                    continue
                d_from_drop = world.distance(pos, nearest_drop)
                score = (d_travel, -d_from_drop)
                if score < best_score:
                    best_score = score
                    best_pos = pos

        return best_pos

    # ================================================================
    # NIGHTMARE QUEUE STRATEGY (10+ bots)
    # ================================================================

    # Zone definitions for nightmare map (3 zones matching 3 drop-off positions)
    NIGHTMARE_ZONES = [
        (1, 9),    # Left zone: shelves x=3,5,7,9, drop-off (1,16)
        (10, 18),  # Center zone: shelves x=11,13,15,17, drop-off (15,16)
        (19, 28),  # Right zone: shelves x=19,21,23,25, drop-off (27,16)
    ]

    @staticmethod
    def _bot_zone(bot_id: int, n_bots: int) -> int:
        """Assign bot to zone by ID. Spread evenly across 3 zones."""
        zone_size = n_bots // 3
        remainder = n_bots % 3
        # Distribute remainder: zone 0 gets +1 if remainder>=1, zone 1 if remainder>=2
        if bot_id < zone_size + (1 if remainder >= 1 else 0):
            return 0
        elif bot_id < 2 * zone_size + (1 if remainder >= 1 else 0) + (1 if remainder >= 2 else 0):
            return 1
        else:
            return 2

    def _compute_scatter_targets(
        self, spawn: Pos, bot_ids: list[int], world: WorldModel,
    ) -> dict[int, Pos]:
        """Compute BFS scatter targets from spawn to disperse stacked bots.

        Lower bot IDs get closer targets (game engine processes low IDs first,
        so they can move to adjacent cells in round 1 while higher IDs wait).
        BFS respects one-way aisles via PathEngine._directed_neighbors.
        """
        from collections import deque

        grid = world._grid
        path_engine = world.path

        visited = set()
        queue = deque([(spawn, 0)])
        visited.add(spawn)
        positions: list[Pos] = []

        while queue and len(positions) < len(bot_ids):
            pos, dist = queue.popleft()
            if dist > 0:
                positions.append(pos)
            for neighbor in path_engine._directed_neighbors(pos):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # Assign: sorted bot IDs get BFS-ordered positions (closest first)
        sorted_ids = sorted(bot_ids)
        targets: dict[int, Pos] = {}
        for i, bot_id in enumerate(sorted_ids):
            if i < len(positions):
                targets[bot_id] = positions[i]
            else:
                targets[bot_id] = spawn  # Fallback if not enough positions
        return targets

    def _apply_spawn_dispersal(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        state,
    ) -> bool:
        """Apply spawn dispersal overrides for early rounds.

        Detects spawn stacking (many bots at same position) and assigns
        navigation_override to scatter targets. Returns True if dispersal
        is still active (some bots haven't reached their targets).
        """
        n_bots = len(state.bots)

        # Compute scatter targets on first round if bots are clustered
        if not self._scatter_targets and state.round <= 2:
            from collections import Counter as Ctr
            pos_counts = Ctr(b.position for b in state.bots)
            most_common_pos, count = pos_counts.most_common(1)[0]
            if count >= n_bots * 0.5:
                self._scatter_spawn = most_common_pos
                stacked_ids = [b.id for b in state.bots
                               if b.position == most_common_pos]
                self._scatter_targets = self._compute_scatter_targets(
                    most_common_pos, stacked_ids, world,
                )
                logger.info(
                    "SCATTER: %d bots stacked at %s, computed %d scatter targets",
                    count, most_common_pos, len(self._scatter_targets),
                )

        if not self._scatter_targets:
            return False

        # Check if spawn is clear (main goal: unstick the spawn bottleneck)
        spawn = self._scatter_spawn
        bots_at_spawn = sum(1 for b in state.bots if b.position == spawn)

        # Apply scatter override only to bots still at or near spawn
        overrides_set = 0
        for bot in state.bots:
            target = self._scatter_targets.get(bot.id)
            if target is None:
                continue
            if bot.position == target:
                continue  # Already at scatter target
            # Only override bots that are still near spawn (within 3 cells)
            dist_to_spawn = abs(bot.position[0] - spawn[0]) + abs(bot.position[1] - spawn[1])
            if dist_to_spawn > 3:
                continue  # Already dispersed enough
            a = assignments.get(bot.id)
            if a is None:
                continue
            a.navigation_override = target
            a.path = None
            overrides_set += 1

        # Spawn is clear enough (≤2 bots) or enough time passed → stop scatter
        if bots_at_spawn <= 2 or overrides_set == 0:
            logger.info("SCATTER: complete at R%d (spawn=%d, overrides=%d)",
                       state.round, bots_at_spawn, overrides_set)
            self._scatter_targets.clear()
            return False

        if state.round > 20:
            logger.info("SCATTER: timeout at R%d, %d at spawn", state.round, bots_at_spawn)
            self._scatter_targets.clear()
            return False

        return True

    def _nightmare_queue_strategy(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        claimed_items: set[str],
        events: list[str],
    ) -> None:
        """Zone-based strategy for 20-bot nightmare map.

        Each bot is assigned to one of 3 zones (left/center/right).
        Bots pick items ONLY from their zone's shelves, prioritizing
        active-order types. Delivers to nearest drop-off (= zone drop-off).
        This eliminates cross-map travel and reduces corridor congestion.
        """
        state = world.state
        self._last_state = state  # For demand scoring
        grid = world._grid
        active = state.active_orders
        remaining_types = set(active[0].items_remaining) if active else set()
        remaining_list = list(active[0].items_remaining) if active else []
        n_bots = len(state.bots)

        # Find queue row (bottommost cross-corridor)
        sorted_cross = self._find_cross_corridors(grid)
        queue_y = sorted_cross[-2] if len(sorted_cross) >= 2 else sorted_cross[0]
        drive_y = sorted_cross[-1] if sorted_cross else grid.height - 1

        # Count matching items per bot for active order
        def match_count(bot):
            if not remaining_list:
                return 0
            temp = list(remaining_list)
            count = 0
            for inv in bot.inventory:
                if inv in temp:
                    count += 1
                    temp.remove(inv)
            return count

        # Track which item types are already claimed by bots filling up
        filling_types: list[str] = []
        for bot_id, a in assignments.items():
            if a.task and a.task.task_type == TaskType.PICK_UP and a.task.item_type:
                filling_types.append(a.task.item_type)

        # Inventory coverage: what types do all bots collectively have?
        all_inventory_types: list[str] = []
        for bot in state.bots:
            all_inventory_types.extend(bot.inventory)

        # Queue positions: spread along queue_y
        queue_positions: list[Pos] = []
        for x in range(3, grid.width - 1):
            pos = (x, queue_y)
            if grid.is_walkable(pos):
                queue_positions.append(pos)

        occupied_queue: set[Pos] = set()
        for bot in state.bots:
            if bot.position[1] == queue_y:
                occupied_queue.add(bot.position)

        # === PHASE 1: Assign DELIVER to best matching queued bot ===
        if remaining_types:
            # Find bots with matching items, prefer those already on/near queue
            candidates = []
            for bot in state.bots:
                mc = match_count(bot)
                if mc <= 0:
                    continue
                d_to_drop = world.distance(bot.position, world.nearest_drop_off(bot.position, bot.id))
                # Prefer: more matches, then closer to drop-off
                candidates.append((-mc, d_to_drop, bot.id, bot))

            candidates.sort()

            # Dynamic max_deliverers: increase when many bots have matches
            # (= right after order switch with pre-picked preview items)
            n_zones = len(state.drop_off_zones)
            high_match_bots = sum(1 for _, _, _, b in candidates if match_count(b) >= 2)
            max_deliverers = n_zones  # Base: 1 per zone (3)
            if high_match_bots >= 4:
                max_deliverers = n_zones * 3  # Flush mode: 3 per zone (9)

            delivering = 0
            for _, _, _, bot in candidates:
                if delivering >= max_deliverers:
                    break
                a = assignments.get(bot.id)
                if a is None:
                    continue
                # Already delivering? Keep it
                if a.task and a.task.task_type == TaskType.DELIVER:
                    delivering += 1
                    continue
                # Assign deliver
                a.task = Task(task_type=TaskType.DELIVER, target_pos=world.nearest_drop_off(bot.position, bot.id))
                a.path = None
                a.navigation_override = None
                delivering += 1
                logger.debug("QUEUE: B%d DELIVER (match=%d, inv=%s)",
                            bot.id, match_count(bot), bot.inventory)

        # === PHASE 2: Sprint + Pipeline ===
        # Sprint team (5-6 bots): smash active order fast with single-item pickup
        # Pipeline team (14-15 bots): pre-fill inventory for preview/future orders
        # Seamless fallback: when recon orders exhausted → existing reactive architecture
        #
        preview_orders = state.preview_orders
        preview_order = preview_orders[0] if preview_orders else None
        preview_types = set(preview_order.items_remaining) if preview_order else set()

        # Build demand score: how many upcoming orders need each item type
        self._sync_order_index(state)
        demand = self._build_demand_score(n_ahead=8)

        # --- Validate existing tasks (all bots) ---
        item_ids = {i.id for i in state.items}
        for bot in state.bots:
            a = assignments.get(bot.id)
            if a is None:
                continue
            # Deliverers with no active matches → clear or idle
            if a.task and a.task.task_type == TaskType.DELIVER:
                if not any(inv in remaining_types for inv in bot.inventory):
                    if not bot.inventory:
                        a.clear()
                    else:
                        a.task = Task(task_type=TaskType.IDLE, target_pos=self._queue_pos(
                            bot, queue_positions, occupied_queue, world))
                        a.path = None
            # Pickers with vanished items → clear
            elif a.task and a.task.task_type == TaskType.PICK_UP:
                if a.task.item_id and a.task.item_id not in item_ids:
                    a.clear()

        # --- PHASE 2: Identify active team, then assign all bots in ID order ---
        import math
        active_team_max = math.ceil(len(remaining_list) / 3) if remaining_list else 2
        active_team: set[int] = set()

        # Pre-identify active team (deliver + picking active + holders capped)
        for bot in state.bots:
            a = assignments.get(bot.id)
            if a is None:
                continue
            if a.task and a.task.task_type == TaskType.DELIVER:
                active_team.add(bot.id)
            elif a.task and a.task.task_type == TaskType.PICK_UP:
                if a.task.item_type and a.task.item_type in remaining_types:
                    active_team.add(bot.id)

        # Also count bots holding active items (capped at active_team_max)
        for bot in state.bots:
            if bot.id in active_team:
                continue
            mc = match_count(bot)
            if mc > 0 and len(active_team) < active_team_max:
                active_team.add(bot.id)

        # Single interleaved loop — all bots in ID order
        for bot in state.bots:
            a = assignments.get(bot.id)
            if a is None:
                continue

            # Skip bots already delivering
            if a.task and a.task.task_type == TaskType.DELIVER:
                if not any(inv in remaining_types for inv in bot.inventory):
                    if not bot.inventory:
                        a.clear()
                    else:
                        a.task = Task(task_type=TaskType.IDLE, target_pos=self._queue_pos(
                            bot, queue_positions, occupied_queue, world))
                        a.path = None
                continue

            # Skip bots actively picking (validate item still exists)
            if a.task and a.task.task_type == TaskType.PICK_UP:
                if a.task.item_id and a.task.item_id not in {i.id for i in state.items}:
                    a.clear()
                else:
                    continue

            # Bot needs a task
            if len(bot.inventory) >= 3:
                # Full inventory — check matches
                preview_mc = sum(1 for inv in bot.inventory if inv in preview_types) if preview_types else 0
                active_mc = match_count(bot)

                if active_mc > 0:
                    nearest_do = world.nearest_drop_off(bot.position, bot.id)
                    a.task = Task(task_type=TaskType.PRE_PICK, target_pos=nearest_do)
                    a.path = None
                elif preview_mc > 0:
                    nearest_do = world.nearest_drop_off(bot.position, bot.id)
                    d = world.distance(bot.position, nearest_do)
                    if d > 3:
                        a.task = Task(task_type=TaskType.PRE_PICK, target_pos=nearest_do)
                        a.path = None
                    else:
                        a.task = Task(task_type=TaskType.IDLE, target_pos=self._safe_idle_pos(bot.position, world))
                        a.path = None
                else:
                    qp = self._queue_pos(bot, queue_positions, occupied_queue, world)
                    a.task = Task(task_type=TaskType.IDLE, target_pos=qp)
                    a.path = None
                continue

            # Bot has inventory space — assign pickup
            if state.round <= 20:
                combined_early = remaining_types | preview_types
                combined_early_list = remaining_list + (list(preview_order.items_remaining) if preview_order else [])
                if combined_early:
                    self._assign_targeted_pickup(bot, a, world, claimed_items,
                                                 filling_types, combined_early, combined_early_list,
                                                 demand=demand)
                else:
                    self._assign_fill_pickup(bot, a, world, claimed_items,
                                            all_inventory_types, filling_types,
                                            active_need=remaining_types, demand=demand)
            elif bot.id in active_team and remaining_types:
                self._assign_targeted_pickup(bot, a, world, claimed_items,
                                             filling_types, remaining_types, remaining_list,
                                             demand=demand)
            elif preview_types:
                self._assign_targeted_pickup(bot, a, world, claimed_items,
                                             filling_types, preview_types,
                                             list(preview_order.items_remaining),
                                             demand=demand)
            else:
                # Extended pre-picking: use known future orders (N+2, N+3...)
                # to pick items BEFORE orders activate. This eliminates slow
                # orders caused by zero overlap with previous order.
                future = self._get_future_order_types(n_ahead=5)
                if future:
                    # Combine all future types into one target set
                    future_types: set[str] = set()
                    future_list: list[str] = []
                    for ts, il in future:
                        future_types.update(ts)
                        future_list.extend(il)
                    if future_types:
                        self._assign_targeted_pickup(bot, a, world, claimed_items,
                                                     filling_types, future_types,
                                                     future_list, demand=demand)
                    else:
                        self._assign_fill_pickup(bot, a, world, claimed_items,
                                                all_inventory_types, filling_types,
                                                active_need=remaining_types, demand=demand)
                else:
                    self._assign_fill_pickup(bot, a, world, claimed_items,
                                            all_inventory_types, filling_types,
                                            active_need=remaining_types, demand=demand)

        # === PHASE 3: Stranded deliverers on drive lane ===
        for bot in state.bots:
            if bot.position[1] != drive_y:
                continue
            a = assignments.get(bot.id)
            if a is None or not a.task:
                continue
            if a.task.task_type == TaskType.IDLE and a.task.target_pos == bot.position:
                qp = self._queue_pos(bot, queue_positions, occupied_queue, world)
                a.task = Task(task_type=TaskType.IDLE, target_pos=qp)
                a.path = None

    def _assign_targeted_pickup(
        self,
        bot,
        assignment: BotAssignment,
        world: WorldModel,
        claimed_items: set[str],
        filling_types: list[str],
        target_types: set[str],
        target_list: list[str],
        demand: Counter | None = None,
    ) -> None:
        """Pick ONLY items matching target_types. No diverse fill fallback.

        Uses claimed_items for dedup (no two bots target same item).
        Uses type budget (Counter) to limit bots per type.
        Soft zone preference: items in bot's assigned zone get priority.
        Falls back to idle if nothing available nearby.
        """
        state = world.state

        # Type budget: allow 3x overbooking so many bots can pre-pick
        # With 20 bots and 4-6 types, we want 3-4 bots per type
        type_budget = Counter(target_list)
        for t in type_budget:
            type_budget[t] *= 3  # Allow 3x copies
        # Subtract what bots already have in inventory
        for b in state.bots:
            for inv in b.inventory:
                if inv in type_budget and type_budget[inv] > 0:
                    type_budget[inv] -= 1
        # Subtract what's being picked by other bots
        for t in filling_types:
            if t in type_budget and type_budget[t] > 0:
                type_budget[t] -= 1

        # Zone preference for 20+ bots: soft penalty for out-of-zone items
        zone_x_range = None
        if len(state.bots) >= 20 and len(self.NIGHTMARE_ZONES) == 3:
            zone_id = self._bot_zone(bot.id, len(state.bots))
            zone_x_range = self.NIGHTMARE_ZONES[zone_id]

        best_task = None
        best_score = (9999, 9999)

        for item in state.items:
            if item.id in claimed_items:
                continue
            if item.type not in target_types:
                continue
            if type_budget.get(item.type, 0) <= 0:
                continue

            pp = world.best_pickup_position(bot.position, item.position)
            if pp is None:
                continue
            d = world.distance(bot.position, pp)
            if d >= 9999:
                continue

            zone_penalty = 0
            if zone_x_range:
                ix = item.position[0]
                if ix < zone_x_range[0] or ix > zone_x_range[1]:
                    zone_penalty = 10

            # Genome shelf preference: bonus for preferred shelf position
            shelf_bonus = 0
            if self._shelf_preference and item.type in self._shelf_preference:
                pref_idx = self._shelf_preference[item.type]
                # Match shelf by position — items at preferred shelf get -5 distance bonus
                shelf_map = getattr(self._config, '_shelf_map', None)
                if shelf_map and item.type in shelf_map:
                    shelves_list = shelf_map[item.type]
                    if pref_idx < len(shelves_list):
                        pref_pos = tuple(shelves_list[pref_idx])
                        if item.position == pref_pos:
                            shelf_bonus = 5

            demand_bonus = min(demand.get(item.type, 0), 3) if demand else 0
            score = (d + zone_penalty - demand_bonus - shelf_bonus, d)
            if score < best_score:
                best_score = score
                best_task = Task(
                    task_type=TaskType.PICK_UP,
                    target_pos=pp,
                    item_id=item.id,
                    item_type=item.type,
                    item_pos=item.position,
                )

        if best_task:
            assignment.task = best_task
            assignment.path = None
            claimed_items.add(best_task.item_id)
            filling_types.append(best_task.item_type)
        else:
            all_inv = []
            for b in state.bots:
                all_inv.extend(b.inventory)
            self._assign_fill_pickup(bot, assignment, world, claimed_items,
                                    all_inv, filling_types,
                                    active_need=target_types, demand=demand)

    def _assign_fill_pickup(
        self,
        bot,
        assignment: BotAssignment,
        world: WorldModel,
        claimed_items: set[str],
        all_inventory_types: list[str],
        filling_types: list[str],
        active_need: set[str] | None = None,
        zone_x_range: tuple[int, int] | None = None,
        active_only: bool = False,
        demand: Counter | None = None,
    ) -> None:
        """Assign a PICK_UP for an item to fill bot's inventory.

        zone_x_range: if set, restrict to shelves in this x-range and use
        per-bot active targeting (each bot tries to collect 3 active types).
        Otherwise: diverse fill with global supply budget.
        """
        state = world.state
        covered = Counter(all_inventory_types + filling_types)

        # Active order types needed
        active_counter: Counter = Counter()
        active_supply: Counter = Counter()
        if state.active_orders:
            active_counter = Counter(state.active_orders[0].items_remaining)
            for b in state.bots:
                for inv in b.inventory:
                    if inv in active_counter:
                        active_supply[inv] += 1
            for t in filling_types:
                if t in active_counter:
                    active_supply[t] += 1

        # Per-bot: what active types does THIS bot still need?
        bot_active_needs: set[str] = set()
        if zone_x_range and active_counter:
            bot_inv_counter = Counter(bot.inventory)
            for t, need in active_counter.items():
                if bot_inv_counter.get(t, 0) < need:
                    bot_active_needs.add(t)

        best_task = None
        best_score = (9999, 9999)  # (priority, effective_distance)

        for item in state.items:
            if item.id in claimed_items:
                continue

            # Zone restriction: only pick from zone shelves
            if zone_x_range:
                ix = item.position[0]
                if ix < zone_x_range[0] or ix > zone_x_range[1]:
                    continue

            pp = world.best_pickup_position(bot.position, item.position)
            if pp is None:
                continue
            d = world.distance(bot.position, pp)
            if d >= 9999:
                continue

            is_active_type = (active_counter.get(item.type, 0) > 0
                              and active_supply.get(item.type, 0) < active_counter[item.type])
            if is_active_type:
                priority = 0
            else:
                priority = 1 + covered.get(item.type, 0)

            # Demand bonus: high-demand items get up to 2 cells virtual distance reduction
            demand_bonus = min(demand.get(item.type, 0), 2) if demand else 0

            # Genome shelf preference bonus
            shelf_bonus = 0
            if self._shelf_preference and item.type in self._shelf_preference:
                pref_idx = self._shelf_preference[item.type]
                shelf_map = getattr(self._config, '_shelf_map', None)
                if shelf_map and item.type in shelf_map:
                    shelves_list = shelf_map[item.type]
                    if pref_idx < len(shelves_list):
                        pref_pos = tuple(shelves_list[pref_idx])
                        if item.position == pref_pos:
                            shelf_bonus = 5

            score = (priority, d - demand_bonus - shelf_bonus)
            if score < best_score:
                best_score = score
                best_task = Task(
                    task_type=TaskType.PICK_UP,
                    target_pos=pp,
                    item_id=item.id,
                    item_type=item.type,
                    item_pos=item.position,
                )

        if best_task:
            assignment.task = best_task
            assignment.path = None
            claimed_items.add(best_task.item_id)
            filling_types.append(best_task.item_type)
        else:
            assignment.task = Task(task_type=TaskType.IDLE, target_pos=self._safe_idle_pos(bot.position, world))
            assignment.path = None

    @staticmethod
    def _queue_pos(
        bot,
        queue_positions: list[Pos],
        occupied: set[Pos],
        world: WorldModel,
    ) -> Pos:
        """Find nearest unoccupied queue position."""
        best = bot.position
        best_d = 9999
        for pos in queue_positions:
            if pos in occupied:
                continue
            d = world.distance(bot.position, pos)
            if d < best_d:
                best_d = d
                best = pos
        if best != bot.position:
            occupied.add(best)
        return best

    # --- Helpers ---

    def _build_claimed_set(self, assignments: dict[int, BotAssignment]) -> set[str]:
        """Build set of all claimed item IDs."""
        claimed: set[str] = set(self._blacklisted_items.keys())
        for a in assignments.values():
            if a.route:
                for stop in a.route.stops[a.route_step:]:
                    claimed.add(stop.item_id)
            elif a.task and a.task.task_type in (TaskType.PICK_UP, TaskType.PRE_PICK) and a.task.item_id:
                claimed.add(a.task.item_id)
        return claimed

    def _expire_blacklist(self, current_round: int) -> None:
        expired = [iid for iid, exp in self._blacklisted_items.items() if current_round >= exp]
        for iid in expired:
            del self._blacklisted_items[iid]
        # Expire delivery cooldowns
        expired_cd = [bid for bid, exp in self._post_deliver_cooldown.items() if current_round >= exp]
        for bid in expired_cd:
            del self._post_deliver_cooldown[bid]

    def _item_picked_up(self, bot_id: int, item_type: str, bot_inventory: tuple) -> bool:
        prev_inv = self._prev_inventory.get(bot_id, ())
        prev_count = Counter(prev_inv).get(item_type, 0)
        curr_count = Counter(bot_inventory).get(item_type, 0)
        return curr_count > prev_count

    @staticmethod
    def _has_matching_items(bot: Bot, world: WorldModel) -> bool:
        active = world.state.active_orders
        if not active:
            return False
        remaining = list(active[0].items_remaining)
        for inv_item in bot.inventory:
            if inv_item in remaining:
                return True
        return False

