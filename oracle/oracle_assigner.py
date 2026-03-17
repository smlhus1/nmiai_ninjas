"""
OracleAssigner: standalone oracle with perfect future knowledge.

Built from scratch with:
- Sticky assignments with proper lifecycle
- Multi-item route support
- Aggressive pre-pick with dead-weight management
- Parking for idle bots (critical for PIBT yield-on-push)
- Order transition handling: clear stale PRE_PICK on order change
"""

from __future__ import annotations

import logging
from collections import Counter

from bot.models import Pos
from bot.engine.world_model import WorldModel
from bot.strategy.task import Task, TaskType, BotAssignment, Route, RouteStop

logger = logging.getLogger(__name__)

INF = 9999


class OracleAssigner:
    """Standalone oracle planner — drop-in replacement for V2TaskPlanner."""

    def __init__(self, recon_data: dict) -> None:
        self._order_sequence = recon_data.get("order_sequence", [])
        self._prev_inventory: dict[int, tuple[str, ...]] = {}
        self._blacklisted_items: dict[str, int] = {}
        self._config = None
        self._future_orders: list = []
        self._shelf_preference: dict | None = None
        self._last_active_order_id: str | None = None

    # --- Compat stubs for Coordinator ---
    def set_future_orders(self, orders: list) -> None:
        pass

    def blacklist_item(self, item_id: str, expiry_round: int) -> None:
        self._blacklisted_items[item_id] = expiry_round

    def maintain(
        self, world: WorldModel, assignments: dict[int, BotAssignment],
        *, skip_route_abort: bool = False, skip_time_check: bool = False,
    ) -> dict[int, BotAssignment]:
        for bot in world.state.bots:
            self._prev_inventory[bot.id] = bot.inventory
        return assignments

    def plan(
        self, world: WorldModel, assignments: dict[int, BotAssignment],
    ) -> dict[int, BotAssignment]:
        state = world.state
        active_orders = [o for o in state.orders if o.status.value == "active"]
        preview_orders = [o for o in state.orders if o.status.value == "preview"]
        active = active_orders[0] if active_orders else None
        preview = preview_orders[0] if preview_orders else None

        remaining_list = list(active.items_remaining) if active else []
        remaining_types = set(remaining_list)

        # Expire blacklist
        self._blacklisted_items = {
            k: v for k, v in self._blacklisted_items.items() if v > state.round
        }
        map_item_ids = {item.id for item in state.items}

        # === Order transition: clear PRE_PICK tasks (dead weight prevention) ===
        current_order_id = active.id if active else None
        if current_order_id != self._last_active_order_id:
            if self._last_active_order_id is not None:
                # Order changed — clear ALL PRE_PICK tasks
                for bot in state.bots:
                    a = assignments[bot.id]
                    if a.task and a.task.task_type == TaskType.PRE_PICK:
                        a.clear()
            self._last_active_order_id = current_order_id

        # === Step 1: Lifecycle ===
        for bot in state.bots:
            a = assignments[bot.id]
            if a.task is None:
                continue
            prev_inv = self._prev_inventory.get(bot.id, ())

            # Pickup completed
            if (a.task.task_type in (TaskType.PICK_UP, TaskType.PRE_PICK)
                    and len(bot.inventory) > len(prev_inv)):
                a.clear()
                continue

            # Item gone or blacklisted
            if a.task.task_type in (TaskType.PICK_UP, TaskType.PRE_PICK):
                if a.task.item_id and (
                    a.task.item_id not in map_item_ids
                    or a.task.item_id in self._blacklisted_items
                ):
                    a.clear()
                    continue

            # PICK_UP type no longer needed in active order
            if a.task.task_type == TaskType.PICK_UP:
                if a.task.item_type not in remaining_types:
                    a.clear()
                    continue

            # DELIVER with no matching inventory
            if a.task.task_type == TaskType.DELIVER:
                if not bot.inventory or not any(inv in remaining_types for inv in bot.inventory):
                    a.clear()
                    continue

        # === Step 2: Build claimed set ===
        claimed: set[str] = set()
        assigned_bots: set[int] = set()

        for bot in state.bots:
            a = assignments[bot.id]
            if a.task and a.task.task_type != TaskType.IDLE:
                assigned_bots.add(bot.id)
                if a.task.item_id:
                    claimed.add(a.task.item_id)

        # === Step 3: Promote bots with matching active inventory to DELIVER ===
        for bot in state.bots:
            matching = [inv for inv in bot.inventory if inv in remaining_types]
            if not matching:
                continue

            a = assignments[bot.id]
            if a.task and a.task.task_type == TaskType.DELIVER:
                continue

            drop = world.nearest_drop_off(bot.position, bot.id)
            if a.task and a.task.item_id:
                claimed.discard(a.task.item_id)
            a.task = Task(task_type=TaskType.DELIVER, target_pos=drop)
            a.path = None
            a.route = None
            a.route_step = 0
            assigned_bots.add(bot.id)

        # === Step 4: Assign idle bots to active items ===
        type_budget = Counter(remaining_list)

        # Subtract items in delivering bots' inventories
        for bot in state.bots:
            a = assignments[bot.id]
            if a.task and a.task.task_type == TaskType.DELIVER:
                for inv in bot.inventory:
                    if type_budget[inv] > 0:
                        type_budget[inv] -= 1

        # Subtract items being picked up
        for bot in state.bots:
            a = assignments[bot.id]
            if (a.task and a.task.task_type == TaskType.PICK_UP
                    and a.task.item_type and type_budget[a.task.item_type] > 0):
                type_budget[a.task.item_type] -= 1

        item_pool = []
        for item in state.items:
            if item.type not in type_budget or type_budget[item.type] <= 0:
                continue
            if item.id in claimed or item.id in self._blacklisted_items:
                continue
            drop = world.nearest_drop_off(item.position)
            pp = world.best_pickup_position(drop, item.position)
            if pp is None:
                continue
            d_drop = world.distance(pp, drop)
            if d_drop >= INF:
                continue
            item_pool.append((d_drop, item, pp))

        item_pool.sort(key=lambda x: -x[0])  # Far first

        available_bots = [
            b for b in state.bots
            if b.id not in assigned_bots and len(b.inventory) < 3
        ]

        for d_drop, item, pp in item_pool:
            if type_budget[item.type] <= 0:
                continue
            if not available_bots:
                break

            best_bot = min(available_bots, key=lambda b: world.distance(b.position, pp))
            d_bot = world.distance(best_bot.position, pp)
            if d_bot >= INF:
                continue
            if d_bot + d_drop + 2 > world.rounds_remaining:
                continue

            a = assignments[best_bot.id]
            a.task = Task(
                task_type=TaskType.PICK_UP, target_pos=pp,
                item_id=item.id, item_type=item.type, item_pos=item.position,
            )
            a.path = None
            a.route = None
            claimed.add(item.id)
            type_budget[item.type] -= 1
            assigned_bots.add(best_bot.id)
            available_bots.remove(best_bot)

        # === Step 5: Pre-pick preview with remaining empty-inventory bots ===
        remaining_idle = [
            b for b in state.bots
            if b.id not in assigned_bots and len(b.inventory) == 0
        ]
        if preview and remaining_idle:
            preview_budget = Counter(preview.items_remaining)

            # Subtract items already being pre-picked
            for bot in state.bots:
                a = assignments[bot.id]
                if (a.task and a.task.task_type == TaskType.PRE_PICK
                        and a.task.item_type and preview_budget[a.task.item_type] > 0):
                    preview_budget[a.task.item_type] -= 1

            preview_pool = []
            for item in state.items:
                if item.type not in preview_budget or preview_budget[item.type] <= 0:
                    continue
                if item.id in claimed or item.id in self._blacklisted_items:
                    continue
                drop = world.nearest_drop_off(item.position)
                pp = world.best_pickup_position(drop, item.position)
                if pp is None:
                    continue
                d_drop = world.distance(pp, drop)
                preview_pool.append((d_drop, item, pp))

            preview_pool.sort(key=lambda x: -x[0])

            for d_drop, item, pp in preview_pool:
                if preview_budget[item.type] <= 0:
                    continue
                if not remaining_idle:
                    break

                best_bot = min(remaining_idle, key=lambda b: world.distance(b.position, pp))
                d_bot = world.distance(best_bot.position, pp)
                if d_bot >= INF:
                    continue

                a = assignments[best_bot.id]
                a.task = Task(
                    task_type=TaskType.PRE_PICK, target_pos=pp,
                    item_id=item.id, item_type=item.type, item_pos=item.position,
                )
                a.path = None
                a.route = None
                claimed.add(item.id)
                preview_budget[item.type] -= 1
                assigned_bots.add(best_bot.id)
                remaining_idle.remove(best_bot)

        # === Step 6: Park idle bots ===
        parking = world.parking_positions()
        park_idx = 0
        for bot in state.bots:
            if bot.id not in assigned_bots:
                a = assignments[bot.id]
                if parking:
                    park_pos = parking[park_idx % len(parking)]
                    park_idx += 1
                else:
                    park_pos = bot.position
                a.task = Task(task_type=TaskType.IDLE, target_pos=park_pos)
                a.path = None
                a.route = None

        # Save inventory
        for bot in state.bots:
            self._prev_inventory[bot.id] = bot.inventory

        return assignments
