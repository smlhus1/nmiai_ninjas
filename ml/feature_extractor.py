"""
FeatureExtractor: GameState -> float tensors for ScorerMLP.

Encodes (bot, item) pairs into 48-float feature vectors:
  [bot_features(14) | item_features(12) | global_features(22)]

All values normalized to [0.0, 1.0]. Uses BFS distances from PathEngine
(NOT Manhattan — nightmare map has narrow passages where Manhattan is wrong).
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import torch

from bot.engine.pathfinding import PathEngine
from bot.models import Bot, GameState, Item, Order, OrderStatus, Pos
from bot.strategy.task import BotAssignment, TaskType


class FeatureContext:
    """Pre-computed context shared across all (bot, item) pairs in a round."""

    __slots__ = (
        "assignments", "claimed_items", "active_order", "preview_order",
        "active_remaining", "preview_remaining", "demand", "score_history",
        "orders_completed", "max_orders", "item_type_index", "bot_positions",
        "n_bots", "max_dist", "drop_off_zones",
    )

    def __init__(
        self,
        *,
        assignments: dict[int, BotAssignment] | None = None,
        claimed_items: set[str] | None = None,
        active_order: Order | None = None,
        preview_order: Order | None = None,
        demand: dict[str, int] | None = None,
        score_history: list[int] | None = None,
        orders_completed: int = 0,
        max_orders: int = 50,
        item_type_index: dict[str, int] | None = None,
        bot_positions: list[Pos] | None = None,
        n_bots: int = 20,
        max_dist: int = 60,
        drop_off_zones: tuple[Pos, ...] = (),
    ) -> None:
        self.assignments = assignments or {}
        self.claimed_items = claimed_items or set()
        self.active_order = active_order
        self.preview_order = preview_order
        self.active_remaining = list(active_order.items_remaining) if active_order else []
        self.preview_remaining = list(preview_order.items_remaining) if preview_order else []
        self.demand = demand or {}
        self.score_history = score_history or []
        self.orders_completed = orders_completed
        self.max_orders = max(max_orders, 1)
        self.item_type_index = item_type_index or {}
        self.bot_positions = bot_positions or []
        self.n_bots = max(n_bots, 1)
        self.max_dist = max(max_dist, 1)
        self.drop_off_zones = drop_off_zones


class FeatureExtractor:
    """Extracts normalized feature vectors from game state."""

    # ----------------------------------------------------------------
    # Bot features (14 floats)
    # ----------------------------------------------------------------

    @staticmethod
    def extract_bot_features(
        bot: Bot,
        state: GameState,
        path_engine: PathEngine,
        ctx: FeatureContext,
    ) -> torch.Tensor:
        """14-float bot feature vector, all values in [0.0, 1.0]."""
        w = max(state.grid.width - 1, 1)
        h = max(state.grid.height - 1, 1)

        # Position normalized
        bot_x_norm = bot.position[0] / w
        bot_y_norm = bot.position[1] / h

        # BFS distance to nearest drop-off
        if ctx.drop_off_zones:
            dist_do = min(
                path_engine.distance(bot.position, z)
                for z in ctx.drop_off_zones
            )
        else:
            dist_do = path_engine.distance(bot.position, state.drop_off)
        dist_to_dropoff = min(dist_do / ctx.max_dist, 1.0)

        # Inventory status
        inv = list(bot.inventory)
        inv_size = len(inv) / 3.0

        active_rem = list(ctx.active_remaining)
        active_match = 0
        for t in inv:
            if t in active_rem:
                active_rem.remove(t)
                active_match += 1
        inv_active_match = active_match / 3.0

        preview_rem = list(ctx.preview_remaining)
        preview_match = 0
        for t in inv:
            if t in preview_rem:
                preview_rem.remove(t)
                preview_match += 1
        inv_preview_match = preview_match / 3.0

        # Task one-hot (4 dims)
        assignment = ctx.assignments.get(bot.id)
        task_type = assignment.task.task_type if (assignment and assignment.task) else TaskType.IDLE
        task_pickup = 1.0 if task_type == TaskType.PICK_UP else 0.0
        task_deliver = 1.0 if task_type == TaskType.DELIVER else 0.0
        task_prepick = 1.0 if task_type == TaskType.PRE_PICK else 0.0
        task_idle = 1.0 if task_type == TaskType.IDLE else 0.0

        # Congestion: bots within Manhattan radius 3
        nearby = sum(
            1 for p in ctx.bot_positions
            if p != bot.position and abs(p[0] - bot.position[0]) + abs(p[1] - bot.position[1]) <= 3
        )
        nearby_bots = min(nearby / ctx.n_bots, 1.0)

        # Round info
        round_norm = state.round / max(state.max_rounds, 1)
        rounds_left_norm = state.rounds_remaining / max(state.max_rounds, 1)

        # Bot ID
        bot_id_norm = bot.id / ctx.n_bots

        features = [
            bot_x_norm, bot_y_norm, dist_to_dropoff,
            inv_size, inv_active_match, inv_preview_match,
            task_pickup, task_deliver, task_prepick, task_idle,
            nearby_bots, round_norm, rounds_left_norm, bot_id_norm,
        ]
        return torch.tensor(features, dtype=torch.float32)

    # ----------------------------------------------------------------
    # Item features (12 floats)
    # ----------------------------------------------------------------

    @staticmethod
    def extract_item_features(
        bot: Bot,
        item: Item,
        state: GameState,
        path_engine: PathEngine,
        ctx: FeatureContext,
    ) -> torch.Tensor:
        """12-float item feature vector, all values in [0.0, 1.0]."""
        w = max(state.grid.width - 1, 1)
        h = max(state.grid.height - 1, 1)

        # Item position
        item_x_norm = item.position[0] / w
        item_y_norm = item.position[1] / h

        # BFS distance bot -> item (use adjacent walkable cell for shelf items)
        dist_bi = path_engine.distance(bot.position, item.position)
        if dist_bi >= 9999:
            # Item is on shelf — find nearest adjacent walkable cell
            best = 9999
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                adj = (item.position[0] + dx, item.position[1] + dy)
                d = path_engine.distance(bot.position, adj)
                if d < best:
                    best = d
            dist_bi = best
        dist_bot_to_item = min(dist_bi / ctx.max_dist, 1.0)

        # BFS distance item -> nearest drop-off
        if ctx.drop_off_zones:
            dist_id = min(
                path_engine.distance(item.position, z)
                for z in ctx.drop_off_zones
            )
        else:
            dist_id = path_engine.distance(item.position, state.drop_off)
        if dist_id >= 9999:
            # Shelf item — use adjacent walkable cell
            best = 9999
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                adj = (item.position[0] + dx, item.position[1] + dy)
                if ctx.drop_off_zones:
                    d = min(path_engine.distance(adj, z) for z in ctx.drop_off_zones)
                else:
                    d = path_engine.distance(adj, state.drop_off)
                if d < best:
                    best = d
            dist_id = best
        dist_item_to_do = min(dist_id / ctx.max_dist, 1.0)

        # Active/preview order matching
        is_active_needed = 1.0 if item.type in ctx.active_remaining else 0.0
        active_count = ctx.active_remaining.count(item.type)
        active_count_needed = min(active_count / 7.0, 1.0)

        is_preview_needed = 1.0 if item.type in ctx.preview_remaining else 0.0

        # Demand score
        demand_val = ctx.demand.get(item.type, 0)
        demand_score = min(demand_val / 8.0, 1.0)

        # Claimed
        is_claimed = 1.0 if item.id in ctx.claimed_items else 0.0

        # Bots closer to this item than current bot
        closer = 0
        for p in ctx.bot_positions:
            if p == bot.position:
                continue
            other_dist = path_engine.distance(p, item.position)
            if other_dist >= 9999:
                # Try adjacent cells for shelf items
                other_dist = min(
                    (path_engine.distance(p, (item.position[0] + dx, item.position[1] + dy))
                     for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))),
                    default=9999,
                )
            if other_dist < dist_bi:
                closer += 1
        bots_closer = min(closer / ctx.n_bots, 1.0)

        # Item type index
        type_idx = ctx.item_type_index.get(item.type, 0)
        n_types = max(len(ctx.item_type_index), 1)
        item_type_norm = type_idx / max(n_types, 21)

        # Item congestion (bots within Manhattan radius 2 of item)
        item_nearby = sum(
            1 for p in ctx.bot_positions
            if abs(p[0] - item.position[0]) + abs(p[1] - item.position[1]) <= 2
        )
        item_congestion = min(item_nearby / ctx.n_bots, 1.0)

        features = [
            item_x_norm, item_y_norm, dist_bot_to_item, dist_item_to_do,
            is_active_needed, active_count_needed, is_preview_needed, demand_score,
            is_claimed, bots_closer, item_type_norm, item_congestion,
        ]
        return torch.tensor(features, dtype=torch.float32)

    # ----------------------------------------------------------------
    # Global features (22 floats)
    # ----------------------------------------------------------------

    @staticmethod
    def extract_global_features(
        state: GameState,
        path_engine: PathEngine,
        ctx: FeatureContext,
    ) -> torch.Tensor:
        """22-float global feature vector, all values in [0.0, 1.0]."""
        # Order progress
        active_remaining_count = min(len(ctx.active_remaining) / 7.0, 1.0)
        preview_remaining_count = min(len(ctx.preview_remaining) / 7.0, 1.0)
        orders_completed = min(ctx.orders_completed / ctx.max_orders, 1.0)

        # Pipeline status — count bots by task type
        n_delivering = 0
        n_picking = 0
        n_prepicking = 0
        n_idle = 0
        for a in ctx.assignments.values():
            if a.task is None or a.task.task_type == TaskType.IDLE:
                n_idle += 1
            elif a.task.task_type == TaskType.DELIVER:
                n_delivering += 1
            elif a.task.task_type == TaskType.PICK_UP:
                n_picking += 1
            elif a.task.task_type == TaskType.PRE_PICK:
                n_prepicking += 1
        bots_delivering = min(n_delivering / ctx.n_bots, 1.0)
        bots_picking = min(n_picking / ctx.n_bots, 1.0)
        bots_prepicking = min(n_prepicking / ctx.n_bots, 1.0)
        bots_idle = min(n_idle / ctx.n_bots, 1.0)

        # Next order type distribution (7 floats)
        # Use preview_order if available, else zeros
        next_order_types = [0.0] * 7
        if ctx.preview_order:
            type_counts = Counter(ctx.preview_order.items_required)
            for item_type, count in type_counts.items():
                idx = ctx.item_type_index.get(item_type, 0)
                if idx < 7:
                    next_order_types[idx] = min(count / 7.0, 1.0)

        # Collective match: how many items in all bot inventories match preview order
        collective_match = 0
        if ctx.preview_order:
            remaining = list(ctx.preview_order.items_remaining)
            for a in ctx.assignments.values():
                bot = state.get_bot(a.bot_id)
                if bot:
                    for t in bot.inventory:
                        if t in remaining:
                            remaining.remove(t)
                            collective_match += 1
        collective_next_match = min(collective_match / 7.0, 1.0)

        # Dropoff congestion
        do_pos = state.drop_off
        do_nearby = sum(
            1 for p in ctx.bot_positions
            if abs(p[0] - do_pos[0]) + abs(p[1] - do_pos[1]) <= 4
        )
        dropoff_congestion = min(do_nearby / ctx.n_bots, 1.0)

        # Zone occupancy (nightmare-specific: 3 zones)
        zone_occupied = [0.0, 0.0, 0.0]
        occupied_set = set(ctx.bot_positions)
        zones = list(ctx.drop_off_zones)[:3] if ctx.drop_off_zones else []
        for i, z in enumerate(zones):
            if z in occupied_set:
                zone_occupied[i] = 1.0

        # Score velocity: average score per round over last 10 rounds
        score_velocity = 0.0
        hist = ctx.score_history
        if len(hist) >= 2:
            window = min(10, len(hist) - 1)
            recent_gain = hist[-1] - hist[-(window + 1)]
            score_velocity = min(max(recent_gain / window / 2.0, 0.0), 1.0)

        # Total inventory fullness across all bots
        total_inv = sum(len(state.get_bot(a.bot_id).inventory)
                        for a in ctx.assignments.values()
                        if state.get_bot(a.bot_id)) if ctx.assignments else 0
        if not ctx.assignments:
            total_inv = sum(len(b.inventory) for b in state.bots)
        inventory_fullness = min(total_inv / (3.0 * ctx.n_bots), 1.0)

        # Active order size (how big is the current order)
        active_order_size = 0.0
        if ctx.active_order:
            active_order_size = min(len(ctx.active_order.items_required) / 7.0, 1.0)

        features = [
            active_remaining_count, preview_remaining_count, orders_completed,
            bots_delivering, bots_picking, bots_prepicking, bots_idle,
            *next_order_types,  # 7 floats
            collective_next_match,
            dropoff_congestion,
            *zone_occupied,  # 3 floats
            score_velocity,
            inventory_fullness, active_order_size,
        ]
        return torch.tensor(features, dtype=torch.float32)

    # ----------------------------------------------------------------
    # Full pair encoding (48 floats)
    # ----------------------------------------------------------------

    @staticmethod
    def encode_pair(
        bot: Bot,
        item: Item,
        state: GameState,
        path_engine: PathEngine,
        ctx: FeatureContext,
    ) -> torch.Tensor:
        """48-float feature vector for one (bot, item) pair."""
        bot_f = FeatureExtractor.extract_bot_features(bot, state, path_engine, ctx)
        item_f = FeatureExtractor.extract_item_features(bot, item, state, path_engine, ctx)
        global_f = FeatureExtractor.extract_global_features(state, path_engine, ctx)
        return torch.cat([bot_f, item_f, global_f])

    @staticmethod
    def encode_all_pairs(
        state: GameState,
        path_engine: PathEngine,
        ctx: FeatureContext,
    ) -> torch.Tensor:
        """Encode all (bot, item) pairs. Returns (N_bots * N_items, 48) tensor."""
        pairs = []
        for bot in state.bots:
            for item in state.items:
                pairs.append(
                    FeatureExtractor.encode_pair(bot, item, state, path_engine, ctx)
                )
        if not pairs:
            return torch.zeros(0, 48)
        return torch.stack(pairs)
