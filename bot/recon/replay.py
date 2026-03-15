"""
ReplayPlanner: executes a pre-computed game plan.

Same plan() interface as TaskPlanner. Uses the offline plan to generate
tasks, with automatic fallback to the reactive planner on divergence.

Key design: shelf-based, NOT item_id-based. Items respawn with new IDs
each game. The plan says "go to shelf (5,1) and pick 'milk'" — at runtime
we find the actual item at that position with matching type.

Critical fix: calls reactive.maintain() for task lifecycle (route advancement,
invalidation, stuck detection) but does NOT call reactive.plan() which would
shadow the replay assignments entirely.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from bot.models import GameState, Pos, OrderStatus
from bot.engine.world_model import WorldModel
from bot.strategy.task import (
    Task, TaskType, BotAssignment, Route, RouteStop,
)
from bot.strategy.planner import TaskPlanner

logger = logging.getLogger(__name__)

MAX_DIVERGENCE = 5  # Consecutive divergent rounds before permanent reactive switch
INVENTORY_CAP = 3


class ReplayPlanner:
    """
    Executes a pre-computed game plan with fallback to reactive planning.

    The plan contains ordered pickup sequences per order. This planner
    translates those into Tasks/Routes using the same interface as TaskPlanner.

    Uses reactive.maintain() for task lifecycle management (route advancement,
    invalidation, stuck detection) but handles assignment directly from the plan.
    """

    def __init__(self, game_plan: dict, reactive_planner: TaskPlanner) -> None:
        self._plan = game_plan
        self._reactive = reactive_planner
        self._order_plans: list[dict[str, Any]] = game_plan.get("order_plans", [])
        self._current_order_idx = 0
        self._current_batch_idx = 0  # v1 single-bot batch index
        self._bot_batch_idx: dict[int, int] = {}  # v2 per-bot batch index
        self._is_v2 = game_plan.get("version", 1) >= 2
        self._mode = "replay"  # or "reactive"
        self._divergence_count = 0
        self._last_active_order_id: str | None = None

    def plan(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
    ) -> dict[int, BotAssignment]:
        """Same interface as TaskPlanner.plan()."""
        if self._mode == "reactive":
            return self._reactive.plan(world, assignments)

        state = world.state

        # Maintenance: advance routes, invalidate stale tasks, track inventory.
        # This does NOT assign new tasks — that's our job.
        # For multi-bot with distributed 1-item batches, reactive heuristics
        # are safe: route-abort only triggers at 2+ items, time-check is useful.
        # For single-bot, skip route-abort to preserve 3-item replay batches.
        is_multi = len(world.state.bots) >= 2
        assignments = self._reactive.maintain(
            world, assignments,
            skip_route_abort=not is_multi,
            skip_time_check=False,
        )

        # Divergence check
        if not self._check_divergence(state):
            self._divergence_count += 1
            if self._divergence_count >= MAX_DIVERGENCE:
                logger.warning("REPLAY: %d consecutive divergences — switching to REACTIVE",
                               MAX_DIVERGENCE)
                self._mode = "reactive"
                return self._reactive.plan(world, assignments)
        else:
            self._divergence_count = 0

        # Detect order transitions
        self._detect_order_transition(state)

        # Get current order plan
        order_plan = self._current_order_plan()
        if order_plan is None:
            # Plan exhausted — use reactive for endgame/remaining
            logger.info("REPLAY: plan exhausted at order_idx=%d, falling back to reactive",
                        self._current_order_idx)
            self._mode = "reactive"
            return self._reactive.plan(world, assignments)

        # Assign from plan
        return self._execute_plan(world, assignments, order_plan)

    def _check_divergence(self, state: GameState) -> bool:
        """Check if game state matches expected plan state. Returns True if OK.

        Matches by items_required (same seed → same items) rather than order_id
        (which changes between game sessions).
        """
        if self._current_order_idx >= len(self._order_plans):
            return True

        expected_order = self._order_plans[self._current_order_idx]
        expected_items = sorted(expected_order.get("items_required", []))

        active = [o for o in state.orders if o.status == OrderStatus.ACTIVE and not o.complete]
        if not active:
            return True

        current_items = sorted(active[0].items_required)

        if expected_items != current_items:
            # Try to find matching order plan by items_required
            for i, op in enumerate(self._order_plans):
                if sorted(op.get("items_required", [])) == current_items:
                    logger.info("REPLAY: order sync by items — jumping from plan idx %d to %d",
                                self._current_order_idx, i)
                    self._current_order_idx = i
                    self._current_batch_idx = 0
                    self._bot_batch_idx = {}
                    return True
            logger.warning("REPLAY: active order items %s not found in plan", current_items)
            return False

        return True

    def _detect_order_transition(self, state: GameState) -> None:
        """Detect when active order changes and advance plan accordingly.

        Matches by items_required rather than order_id (IDs change between sessions).
        """
        active = [o for o in state.orders if o.status == OrderStatus.ACTIVE and not o.complete]
        current_id = active[0].id if active else None

        if current_id and current_id != self._last_active_order_id:
            if self._last_active_order_id is not None and active:
                current_items = sorted(active[0].items_required)
                for i, op in enumerate(self._order_plans):
                    if sorted(op.get("items_required", [])) == current_items:
                        self._current_order_idx = i
                        self._current_batch_idx = 0
                        self._bot_batch_idx = {}
                        logger.info("REPLAY: order transition → plan idx %d (items %s)",
                                    i, current_items)
                        break

        self._last_active_order_id = current_id

    def _current_order_plan(self) -> dict | None:
        """Get the current order plan entry, or None if exhausted."""
        if self._current_order_idx < len(self._order_plans):
            return self._order_plans[self._current_order_idx]
        return None

    def _execute_plan(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        order_plan: dict,
    ) -> dict[int, BotAssignment]:
        """Assign tasks from plan. Dispatches to v1 or v2 based on plan format."""
        if "bot_assignments" in order_plan:
            return self._execute_multi_bot_plan(world, assignments, order_plan)
        return self._execute_single_bot_plan(world, assignments, order_plan)

    def _execute_multi_bot_plan(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        order_plan: dict,
    ) -> dict[int, BotAssignment]:
        """Assign tasks from v2 multi-bot plan. Each bot has its own batch list."""
        state = world.state
        bot_assignments_plan = order_plan["bot_assignments"]

        # Find bots that need new work
        idle_bots = sorted(
            bot_id for bot_id, a in assignments.items()
            if not a.has_task or a.task.task_type == TaskType.IDLE
        )

        if not idle_bots:
            return assignments

        assigned_any = False
        still_idle: list[int] = []

        for bot_id in idle_bots:
            bot = state.get_bot(bot_id)
            if bot is None:
                continue

            bot_key = str(bot_id)
            bot_plan = bot_assignments_plan.get(bot_key)
            if bot_plan is None:
                still_idle.append(bot_id)
                continue

            batches = bot_plan.get("batches", [])
            batch_idx = self._bot_batch_idx.get(bot_id, 0)

            if batch_idx >= len(batches):
                # This bot's plan is exhausted
                still_idle.append(bot_id)
                continue

            batch = batches[batch_idx]

            # Validate: at least one batch item is still needed by active order.
            # Not ALL must match — other bots may have delivered some items already,
            # and multi-bot batches can have partial overlap.
            active = state.active_orders
            if active:
                remaining_types = set(active[0].items_remaining)
                batch_types = [entry.get("item_type") for entry in batch]
                matching = [t for t in batch_types if t in remaining_types]
                if not matching:
                    logger.warning(
                        "REPLAY: bot %d batch %d items %s — none match remaining %s, skipping",
                        bot_id, batch_idx, batch_types, remaining_types,
                    )
                    self._bot_batch_idx[bot_id] = batch_idx + 1
                    still_idle.append(bot_id)
                    continue
                # Filter batch to only matching items
                if len(matching) < len(batch_types):
                    batch = [e for e in batch if e.get("item_type") in remaining_types]

            # Account for items already in inventory
            capacity = INVENTORY_CAP - len(bot.inventory)
            if capacity <= 0:
                # Bot needs to deliver first — let maintain() handle it
                continue
            effective_batch = batch[:capacity] if capacity < len(batch) else batch

            route_stops = self._resolve_batch_items(world, effective_batch)
            if not route_stops:
                logger.warning("REPLAY: bot %d couldn't resolve batch %d, skipping",
                               bot_id, batch_idx)
                self._bot_batch_idx[bot_id] = batch_idx + 1
                still_idle.append(bot_id)
                continue

            # Assign route
            first_stop = route_stops[0]
            route = Route(stops=route_stops)
            assignments[bot_id].task = Task(
                task_type=TaskType.PICK_UP,
                target_pos=first_stop.pickup_pos,
                item_id=first_stop.item_id,
                item_type=first_stop.item_type,
                item_pos=first_stop.item_pos,
            )
            assignments[bot_id].route = route
            assignments[bot_id].route_step = 0
            assignments[bot_id].path = None

            self._bot_batch_idx[bot_id] = batch_idx + 1
            assigned_any = True

            logger.info("REPLAY: bot %d assigned batch %d/%d (%d items)",
                        bot_id, batch_idx + 1, len(batches), len(route_stops))

        # Bots without planned work get reactive fallback for active order items.
        # Only active items — preview items create dead weight in multi-bot.
        if still_idle:
            self._reactive_assign_active_only(world, assignments, still_idle)

        return assignments

    def _execute_single_bot_plan(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        order_plan: dict,
    ) -> dict[int, BotAssignment]:
        """Assign tasks from v1 single-bot plan. Only assigns to idle bots — active
        tasks are managed by maintain()."""
        state = world.state
        batches = order_plan.get("batches", [])

        # Find bots that need new work
        idle_bots = sorted(
            bot_id for bot_id, a in assignments.items()
            if not a.has_task or a.task.task_type == TaskType.IDLE
        )

        if not idle_bots:
            return assignments

        # All batches done for this order?
        if self._current_batch_idx >= len(batches):
            # Try pre-picks from plan, then fall back to reactive for idle bots
            pre_picks = order_plan.get("pre_picks", [])
            if pre_picks:
                self._assign_pre_picks(world, assignments, idle_bots, pre_picks)
                # Remaining idle bots get reactive assignment
                still_idle = [
                    bid for bid in idle_bots
                    if not assignments[bid].has_task or assignments[bid].task.task_type == TaskType.IDLE
                ]
                if still_idle:
                    self._reactive_assign_active_only(world, assignments, still_idle)
            else:
                self._reactive_assign_active_only(world, assignments, idle_bots)
            return assignments

        batch = batches[self._current_batch_idx]

        # Filter batch to only include items still needed by active order.
        # Pre-picked items from other bots may have auto-delivered, reducing
        # items_remaining. Skip entire batch if nothing matches.
        active = state.active_orders
        if active:
            remaining_types = set(active[0].items_remaining)
            batch = [e for e in batch if e.get("item_type") in remaining_types]
            if not batch:
                logger.info("REPLAY: batch %d fully delivered by pre-picks, advancing",
                            self._current_batch_idx)
                self._current_batch_idx += 1
                return assignments

        # Multi-bot: distribute batch items across all idle bots (1 item each).
        # Single-bot: give entire batch to the one bot as a multi-item route.
        if len(state.bots) >= 2:
            self._distribute_batch_items(world, assignments, idle_bots, batch)
        else:
            bot_id = idle_bots[0]
            bot = state.get_bot(bot_id)
            if bot is None:
                self._current_batch_idx += 1
                return assignments

            capacity = INVENTORY_CAP - len(bot.inventory)
            effective_batch = batch[:capacity] if capacity < len(batch) else batch

            route_stops = self._resolve_batch_items(world, effective_batch)
            if not route_stops:
                logger.warning("REPLAY: couldn't resolve batch %d items, skipping",
                               self._current_batch_idx)
                self._current_batch_idx += 1
                return assignments

            first_stop = route_stops[0]
            route = Route(stops=route_stops)
            assignments[bot_id].task = Task(
                task_type=TaskType.PICK_UP,
                target_pos=first_stop.pickup_pos,
                item_id=first_stop.item_id,
                item_type=first_stop.item_type,
                item_pos=first_stop.item_pos,
            )
            assignments[bot_id].route = route
            assignments[bot_id].route_step = 0
            assignments[bot_id].path = None

            logger.info("REPLAY: assigned batch %d/%d (%d items) to bot %d",
                         self._current_batch_idx + 1, len(batches),
                         len(route_stops), bot_id)

            # Remaining idle bots: active-only reactive (never preview pre-pick)
            remaining_idle = [bid for bid in idle_bots[1:]]
            if remaining_idle:
                self._reactive_assign_active_only(world, assignments, remaining_idle)

        self._current_batch_idx += 1

        return assignments

    def _distribute_batch_items(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        idle_bots: list[int],
        batch: list[dict],
    ) -> None:
        """Distribute batch items across idle bots: 1 item per bot.

        Uses replay plan's optimal shelf selection, but parallelizes
        across all available bots for faster order completion.
        Remaining idle bots get active-only reactive fallback.
        """
        state = world.state
        used_ids: set[str] = set()
        assigned_bots: list[int] = []

        for i, entry in enumerate(batch):
            if i >= len(idle_bots):
                break

            bot_id = idle_bots[i]
            bot = state.get_bot(bot_id)
            if bot is None:
                continue

            # Skip bots with full inventory (need to deliver first)
            if len(bot.inventory) >= INVENTORY_CAP:
                continue

            shelf_pos: Pos = tuple(entry["shelf_pos"])
            pickup_pos: Pos = tuple(entry["pickup_pos"])
            item_type: str = entry["item_type"]

            item = self._find_item_at_shelf(world, shelf_pos, item_type, used_ids)
            if item is None:
                continue

            used_ids.add(item.id)
            stop = RouteStop(
                item_id=item.id,
                item_type=item_type,
                item_pos=shelf_pos,
                pickup_pos=pickup_pos,
            )

            # Single-item route so route advancement creates DELIVER after pickup
            assignments[bot_id].task = Task(
                task_type=TaskType.PICK_UP,
                target_pos=pickup_pos,
                item_id=item.id,
                item_type=item_type,
                item_pos=shelf_pos,
            )
            assignments[bot_id].route = Route(stops=[stop])
            assignments[bot_id].route_step = 0
            assignments[bot_id].path = None
            assigned_bots.append(bot_id)

        if assigned_bots:
            logger.info("REPLAY: distributed batch %d/%d (%d items) to bots %s",
                        self._current_batch_idx + 1,
                        len(self._order_plans[self._current_order_idx].get("batches", [])),
                        len(assigned_bots), assigned_bots)

        # Remaining idle bots: active-only reactive
        remaining = [bid for bid in idle_bots if bid not in assigned_bots]
        if remaining:
            self._reactive_assign_active_only(world, assignments, remaining)

    def _reactive_assign_active_only(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        idle_bots: list[int],
    ) -> None:
        """Assign idle bots with type-budget awareness.

        Priority:
        1. Deliver if carrying items matching active order
        2. Pick active-order items if type budget > 0
        3. Pre-pick preview-order items (auto-deliver on order transition)
        """
        from collections import Counter

        claimed: set[str] = set()
        for a in assignments.values():
            if a.route:
                for stop in a.route.stops[a.route_step:]:
                    claimed.add(stop.item_id)
            elif a.task and a.task.item_id:
                claimed.add(a.task.item_id)

        state = world.state
        active = state.active_orders
        if not active:
            return

        # Type budget: how many of each type still needed for active order
        type_budget = Counter(active[0].items_remaining)

        # Subtract items in routes/tasks
        for a in assignments.values():
            if a.route:
                for stop in a.route.stops[a.route_step:]:
                    if stop.item_type in type_budget:
                        type_budget[stop.item_type] -= 1
            elif a.task and a.task.item_type and a.task.task_type == TaskType.PICK_UP:
                if a.task.item_type in type_budget:
                    type_budget[a.task.item_type] -= 1

        # Subtract items already in ALL bot inventories (being carried)
        for bot in state.bots:
            for inv_type in bot.inventory:
                if inv_type in type_budget:
                    type_budget[inv_type] -= 1

        needed_types = {t for t, c in type_budget.items() if c > 0}

        # Preview order types for pre-picking
        preview = [o for o in state.orders if o.status == OrderStatus.PREVIEW]
        preview_types = Counter(preview[0].items_required) if preview else Counter()
        # Subtract preview items already in inventory or being picked
        for bot in state.bots:
            for inv_type in bot.inventory:
                if inv_type in preview_types:
                    preview_types[inv_type] -= 1
        for a in assignments.values():
            if a.task and a.task.item_type and a.task.task_type == TaskType.PRE_PICK:
                if a.task.item_type in preview_types:
                    preview_types[a.task.item_type] -= 1
        preview_needed = {t for t, c in preview_types.items() if c > 0}

        still_idle: list[int] = []

        for bot_id in idle_bots:
            bot = state.get_bot(bot_id)
            if bot is None:
                continue

            # Deliver if bot has items matching ACTIVE order
            if bot.inventory:
                active_types = set(active[0].items_remaining)
                has_match = any(inv in active_types for inv in bot.inventory)
                if has_match:
                    assignments[bot_id].task = Task(
                        task_type=TaskType.DELIVER,
                        target_pos=world.nearest_drop_off(bot.position, bot_id),
                    )
                    assignments[bot_id].path = None
                    continue

            # Pick nearest item of a type that's still needed for active order
            if needed_types:
                item, pickup = self._find_nearest_of_types(
                    world, bot, needed_types, claimed
                )
                if item and pickup:
                    assignments[bot_id].task = Task(
                        task_type=TaskType.PICK_UP,
                        target_pos=pickup,
                        item_id=item.id,
                        item_type=item.type,
                        item_pos=item.position,
                    )
                    assignments[bot_id].path = None
                    claimed.add(item.id)
                    needed_types.discard(item.type)
                    type_budget[item.type] -= 1
                    continue

            still_idle.append(bot_id)

        # Remaining idle bots: pre-pick preview items
        for bot_id in still_idle:
            bot = state.get_bot(bot_id)
            if bot is None or len(bot.inventory) >= INVENTORY_CAP:
                continue
            if not preview_needed:
                continue

            item, pickup = self._find_nearest_of_types(
                world, bot, preview_needed, claimed
            )
            if item and pickup:
                assignments[bot_id].task = Task(
                    task_type=TaskType.PRE_PICK,
                    target_pos=pickup,
                    item_id=item.id,
                    item_type=item.type,
                    item_pos=item.position,
                )
                assignments[bot_id].path = None
                claimed.add(item.id)
                preview_needed.discard(item.type)
                preview_types[item.type] -= 1

    def _find_nearest_of_types(
        self,
        world: WorldModel,
        bot: Any,
        types: set[str],
        claimed: set[str],
    ) -> tuple[Any, Pos | None]:
        """Find nearest unclaimed item of given types."""
        best_item = None
        best_dist = float("inf")
        best_pickup = None

        for item in world.state.items:
            if item.id in claimed:
                continue
            if item.type not in types:
                continue
            pickup = world.best_pickup_position(bot.position, item.position)
            if pickup is None:
                continue
            d = world.distance(bot.position, pickup)
            if d < best_dist:
                best_dist = d
                best_item = item
                best_pickup = pickup

        return best_item, best_pickup

    def _reactive_assign_idle(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        idle_bots: list[int],
    ) -> None:
        """Let reactive planner assign tasks to remaining idle bots."""
        # Build claimed items set
        claimed: set[str] = set()
        for a in assignments.values():
            if a.route:
                for stop in a.route.stops[a.route_step:]:
                    claimed.add(stop.item_id)
            elif a.task and a.task.item_id:
                claimed.add(a.task.item_id)

        state = world.state
        for bot_id in idle_bots:
            bot = state.get_bot(bot_id)
            if bot is None:
                continue
            # Prefer active-order items (_find_best_task), then any item (_find_fallback_task)
            task = self._reactive._find_best_task(world, bot, claimed, assignments)
            if not task:
                task = self._reactive._find_fallback_task(world, bot, claimed, assignments)
            if task:
                assignments[bot_id].task = task
                assignments[bot_id].path = None
                if task.item_id:
                    claimed.add(task.item_id)

    def _resolve_batch_items(
        self,
        world: WorldModel,
        batch: list[dict],
    ) -> list[RouteStop]:
        """Resolve plan batch entries to actual RouteStops with current item IDs."""
        stops: list[RouteStop] = []
        used_ids: set[str] = set()

        for entry in batch:
            shelf_pos: Pos = tuple(entry["shelf_pos"])
            pickup_pos: Pos = tuple(entry["pickup_pos"])
            item_type: str = entry["item_type"]

            item = self._find_item_at_shelf(world, shelf_pos, item_type, used_ids)
            if item is None:
                logger.warning("REPLAY: no %s found at shelf %s", item_type, shelf_pos)
                continue

            used_ids.add(item.id)
            stops.append(RouteStop(
                item_id=item.id,
                item_type=item_type,
                item_pos=shelf_pos,
                pickup_pos=pickup_pos,
            ))

        return stops

    def _find_item_at_shelf(
        self, world: WorldModel, shelf_pos: Pos, item_type: str,
        exclude_ids: set[str] | None = None,
    ) -> Optional[Any]:
        """Find an item at a specific shelf position with matching type."""
        exclude = exclude_ids or set()
        for item in world.state.items:
            if item.id in exclude:
                continue
            if item.position == shelf_pos and item.type == item_type:
                return item
        return None

    def _assign_pre_picks(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        idle_bots: list[int],
        pre_picks: list[dict],
    ) -> None:
        """Assign pre-pick tasks from the plan."""
        for i, entry in enumerate(pre_picks):
            if i >= len(idle_bots):
                break

            bot_id = idle_bots[i]
            bot = world.state.get_bot(bot_id)
            if bot is None or len(bot.inventory) >= 2:
                continue

            shelf_pos: Pos = tuple(entry["shelf_pos"])
            pickup_pos: Pos = tuple(entry["pickup_pos"])
            item_type: str = entry["item_type"]

            item = self._find_item_at_shelf(world, shelf_pos, item_type)
            if item is None:
                continue

            assignments[bot_id].task = Task(
                task_type=TaskType.PRE_PICK,
                target_pos=pickup_pos,
                item_id=item.id,
                item_type=item_type,
                item_pos=shelf_pos,
            )
            assignments[bot_id].path = None
            logger.debug("REPLAY: pre-pick %s assigned to bot %d", item_type, bot_id)
