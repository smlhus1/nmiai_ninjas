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
        self._current_batch_idx = 0
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

        # Auto-switch to reactive for multi-bot scenarios (replay plan
        # is optimized for single-bot sequential execution)
        if len(world.state.bots) > 2:
            logger.info("REPLAY: %d bots detected — switching to reactive",
                        len(world.state.bots))
            self._mode = "reactive"
            return self._reactive.plan(world, assignments)

        state = world.state

        # Maintenance: advance routes, invalidate stale tasks, track inventory.
        # This does NOT assign new tasks — that's our job.
        assignments = self._reactive.maintain(world, assignments)

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
        """Check if game state matches expected plan state. Returns True if OK."""
        if self._current_order_idx >= len(self._order_plans):
            return True

        expected_order = self._order_plans[self._current_order_idx]
        expected_id = expected_order.get("order_id")

        active = [o for o in state.orders if o.status == OrderStatus.ACTIVE and not o.complete]
        if not active:
            return True

        current_id = active[0].id
        if expected_id and current_id != expected_id:
            for i, op in enumerate(self._order_plans):
                if op.get("order_id") == current_id:
                    logger.info("REPLAY: order sync — jumping from plan idx %d to %d",
                                self._current_order_idx, i)
                    self._current_order_idx = i
                    self._current_batch_idx = 0
                    return True
            logger.warning("REPLAY: active order %s not found in plan", current_id)
            return False

        return True

    def _detect_order_transition(self, state: GameState) -> None:
        """Detect when active order changes and advance plan accordingly."""
        active = [o for o in state.orders if o.status == OrderStatus.ACTIVE and not o.complete]
        current_id = active[0].id if active else None

        if current_id and current_id != self._last_active_order_id:
            if self._last_active_order_id is not None:
                for i, op in enumerate(self._order_plans):
                    if op.get("order_id") == current_id:
                        self._current_order_idx = i
                        self._current_batch_idx = 0
                        logger.info("REPLAY: order transition → plan idx %d (order %s)",
                                    i, current_id)
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
        """Assign tasks from plan. Only assigns to idle bots — active tasks are
        managed by maintain()."""
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
                    self._reactive_assign_idle(world, assignments, still_idle)
            else:
                self._reactive_assign_idle(world, assignments, idle_bots)
            return assignments

        # Assign next batch to first idle bot
        bot_id = idle_bots[0]
        bot = state.get_bot(bot_id)
        if bot is None:
            return assignments

        batch = batches[self._current_batch_idx]

        # Account for items already in inventory (reduce batch to fit capacity)
        capacity = 3 - len(bot.inventory)
        effective_batch = batch[:capacity] if capacity < len(batch) else batch

        route_stops = self._resolve_batch_items(world, effective_batch)
        if not route_stops:
            logger.warning("REPLAY: couldn't resolve batch %d items, skipping",
                           self._current_batch_idx)
            self._current_batch_idx += 1
            return assignments

        # Always use Route (even for 1 item) so route advancement creates DELIVER
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

        self._current_batch_idx += 1

        logger.info("REPLAY: assigned batch %d/%d (%d items) to bot %d",
                     self._current_batch_idx, len(batches),
                     len(route_stops), bot_id)

        # Assign remaining idle bots via reactive (for multi-bot scenarios)
        remaining_idle = [bid for bid in idle_bots[1:]]
        if remaining_idle:
            self._reactive_assign_idle(world, assignments, remaining_idle)

        return assignments

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
