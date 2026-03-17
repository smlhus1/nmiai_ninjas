"""
MLPlanner: hybrid ML + V2TaskPlanner.

V2 runs the full pipeline (routes, delivery scheduling, stuck detection).
ML scorer post-processes assignments: re-ranks IDLE/unassigned bots' item
choices using learned ScorerMLP + BeamSearch.

When no checkpoint: pure V2 fallback (identical behavior).
When checkpoint loaded: V2 first, then ML re-assignment for idle bots.

Same interface: plan(world, assignments) and maintain(world, assignments).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch

from bot.engine.pathfinding import PathEngine
from bot.engine.world_model import WorldModel
from bot.models import GameState, Item
from bot.strategy.task import BotAssignment, Task, TaskType
from bot.strategy.v2.planner import V2TaskPlanner

logger = logging.getLogger(__name__)


class MLPlanner:
    """Hybrid planner: V2TaskPlanner + ML-guided item selection for idle bots."""

    def __init__(self, model_dir: Path = Path("models/")) -> None:
        self._model_dir = model_dir
        self._v2 = V2TaskPlanner()
        self._use_ml = False
        self._scorer = None
        self._device = "cpu"
        self._config = None

        # Proxy attributes V2 exposes
        self._prev_inventory = self._v2._prev_inventory
        self._blacklisted_items = self._v2._blacklisted_items
        self._future_orders: list[dict] = []

        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        from ml.scorer import ScorerMLP

        ckpt = self._find_latest_checkpoint()
        if ckpt:
            try:
                self._scorer = ScorerMLP()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._scorer.load_state_dict(
                    torch.load(ckpt, map_location=device, weights_only=True)
                )
                self._scorer.to(device).eval()
                self._device = device
                self._use_ml = True
                logger.info("MLPlanner: loaded %s (device=%s)", ckpt.name, device)
            except Exception:
                logger.exception("MLPlanner: checkpoint load failed, V2-only mode")
                self._use_ml = False
        else:
            logger.info("MLPlanner: no checkpoint, V2-only mode")

    def _find_latest_checkpoint(self) -> Optional[Path]:
        if not self._model_dir.exists():
            return None
        pts = sorted(self._model_dir.glob("scorer_*.pt"), reverse=True)
        return pts[0] if pts else None

    # --- Forward attributes to V2 ---

    def __getattr__(self, name):
        """Proxy unknown attributes to V2 (config, shelf_preference, etc.)."""
        if name.startswith("_") and name != "_config":
            raise AttributeError(name)
        return getattr(self._v2, name)

    def __setattr__(self, name, value):
        if name in ("_model_dir", "_v2", "_use_ml", "_scorer", "_device",
                     "_config", "_future_orders"):
            super().__setattr__(name, value)
        elif name == "_config":
            super().__setattr__(name, value)
            if hasattr(self, "_v2"):
                self._v2._config = value
        else:
            setattr(self._v2, name, value)

    def set_future_orders(self, order_sequence: list[dict]) -> None:
        self._future_orders = order_sequence
        self._v2.set_future_orders(order_sequence)

    def blacklist_item(self, item_id: str, expiry_round: int) -> None:
        self._v2.blacklist_item(item_id, expiry_round)

    # --- Core interface ---

    def maintain(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        *,
        skip_route_abort: bool = False,
        skip_time_check: bool = False,
    ) -> dict[int, BotAssignment]:
        return self._v2.maintain(
            world, assignments,
            skip_route_abort=skip_route_abort,
            skip_time_check=skip_time_check,
        )

    def plan(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
    ) -> dict[int, BotAssignment]:
        """V2 plans first, then ML re-ranks idle bots' assignments."""
        # Step 1: V2 does full planning
        assignments = self._v2.plan(world, assignments)

        if not self._use_ml:
            return assignments

        # Step 2: ML re-assignment for IDLE bots only
        t0 = time.perf_counter()
        state = world.state

        # Only reassign bots that are truly IDLE with empty inventory
        # (bots V2 deliberately parked should stay parked)
        idle_bots = [
            b for b in state.bots
            if (assignments[b.id].task is None
                or assignments[b.id].task.task_type == TaskType.IDLE)
            and len(b.inventory) == 0  # Don't touch bots with inventory
        ]

        if not idle_bots:
            return assignments

        # Collect claimed items from non-idle bots
        claimed: set[str] = set()
        for a in assignments.values():
            if a.task and a.task.item_id:
                claimed.add(a.task.item_id)
            if a.route:
                claimed.update(a.route.item_ids)

        self._ml_reassign(state, world.path, assignments, idle_bots, claimed)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 20:
            logger.debug("MLPlanner ML phase: %.1fms for %d idle bots",
                        elapsed_ms, len(idle_bots))

        return assignments

    def _ml_reassign(
        self,
        state: GameState,
        pe: PathEngine,
        assignments: dict[int, BotAssignment],
        idle_bots: list,
        claimed: set[str],
    ) -> None:
        """Use ML scorer to find better assignments for idle bots."""
        from ml.beam_search import BeamSearch
        from ml.candidate_generator import DELIVER, IDLE, CandidateGenerator
        from ml.feature_extractor import FeatureContext

        active = state.active_orders[0] if state.active_orders else None
        preview = state.preview_orders[0] if state.preview_orders else None
        active_remaining_set = set(active.items_remaining) if active else set()

        all_types = sorted(set(i.type for i in state.items))
        type_index = {t: i for i, t in enumerate(all_types)}

        ctx = FeatureContext(
            assignments=assignments,
            claimed_items=claimed,
            active_order=active,
            preview_order=preview,
            bot_positions=[b.position for b in state.bots],
            n_bots=len(state.bots),
            max_dist=60,
            item_type_index=type_index,
            drop_off_zones=state.drop_off_zones,
        )

        # Build filtered state with only idle bots
        filtered_state = GameState(
            round=state.round, max_rounds=state.max_rounds, grid=state.grid,
            bots=tuple(idle_bots), items=state.items, orders=state.orders,
            drop_off=state.drop_off, drop_off_zones=state.drop_off_zones,
            score=state.score,
        )

        k = max(5, len(idle_bots))
        gen = CandidateGenerator(k=k)
        candidates = gen.generate(filtered_state, pe, claimed)

        beam = BeamSearch(beam_width=min(20, max(5, len(idle_bots))))
        assignment_map = beam.search(
            filtered_state, pe, candidates, self._scorer, ctx, device=self._device
        )

        # Convert to tasks
        item_by_id = {item.id: item for item in state.items}
        for bot in idle_bots:
            action = assignment_map.get(bot.id, IDLE)
            a = assignments[bot.id]

            if action == DELIVER and len(bot.inventory) > 0:
                nearest_do = state.drop_off
                if state.drop_off_zones:
                    nearest_do = min(state.drop_off_zones,
                                    key=lambda z: pe.distance(bot.position, z))
                a.task = Task(TaskType.DELIVER, nearest_do)

            elif action == IDLE:
                pass  # Keep V2's IDLE

            else:
                item = item_by_id.get(action)
                if item is None:
                    continue

                task_type = TaskType.PICK_UP if item.type in active_remaining_set else TaskType.PRE_PICK

                pickup_pos = item.position
                if not state.grid.is_walkable(item.position):
                    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                        adj = (item.position[0] + dx, item.position[1] + dy)
                        if state.grid.is_walkable(adj):
                            pickup_pos = adj
                            break

                a.task = Task(
                    task_type, pickup_pos,
                    item_id=item.id, item_type=item.type, item_pos=item.position,
                )
