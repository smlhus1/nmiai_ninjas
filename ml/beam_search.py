"""
BeamSearch: global assignment via beam search over bot-item pairs.

Pre-computes ALL (bot, candidate) features and scores in one batch,
then does pure combinatorial search using pre-computed score lookups.
"""
from __future__ import annotations

import torch

from bot.engine.pathfinding import PathEngine
from bot.models import Bot, GameState, Item
from ml.candidate_generator import DELIVER, IDLE
from ml.feature_extractor import FeatureContext, FeatureExtractor
from ml.scorer import ScorerMLP


class BeamSearch:
    """Beam search over global bot-item assignment."""

    def __init__(self, beam_width: int = 20) -> None:
        self.beam_width = beam_width

    def search(
        self,
        state: GameState,
        path_engine: PathEngine,
        candidates: dict[int, list[str]],
        scorer: ScorerMLP,
        ctx: FeatureContext,
        device: str = "cpu",
    ) -> dict[int, str]:
        """Find best global assignment via beam search.

        Pre-computes all features and scores in a single batch, then
        does pure Python combinatorial search with score lookups.
        """
        bots_sorted = sorted(state.bots, key=lambda b: b.id)
        item_by_id: dict[str, Item] = {item.id: item for item in state.items}
        bot_by_id: dict[int, Bot] = {b.id: b for b in state.bots}

        # --- Phase 1: Pre-compute ALL (bot, candidate) scores ---
        all_features: list[torch.Tensor] = []
        score_keys: list[tuple[int, str]] = []  # (bot_id, action)

        for bot in bots_sorted:
            bot_candidates = candidates.get(bot.id, [IDLE])
            for action in bot_candidates:
                if action in (DELIVER, IDLE):
                    feat = self._encode_action(bot, action, state, path_engine, ctx)
                else:
                    item = item_by_id.get(action)
                    if item is None:
                        continue
                    feat = FeatureExtractor.encode_pair(bot, item, state, path_engine, ctx)
                all_features.append(feat)
                score_keys.append((bot.id, action))

        if not all_features:
            return {b.id: IDLE for b in bots_sorted}

        # Batch score all at once
        batch = torch.stack(all_features).to(device)
        scorer.eval()
        with torch.no_grad():
            all_scores = scorer(batch).squeeze(-1)  # (N,)

        # Build lookup: (bot_id, action) -> score
        score_lookup: dict[tuple[int, str], float] = {}
        for i, (bot_id, action) in enumerate(score_keys):
            score_lookup[(bot_id, action)] = all_scores[i].item()

        # --- Phase 2: Beam search using pre-computed scores ---
        # beam: list of (total_score, assignment_dict, claimed_set)
        beam: list[tuple[float, dict[int, str], frozenset[str]]] = [
            (0.0, {}, frozenset())
        ]

        for bot in bots_sorted:
            bot_candidates = candidates.get(bot.id, [IDLE])
            if not bot_candidates:
                bot_candidates = [IDLE]

            new_beam: list[tuple[float, dict[int, str], frozenset[str]]] = []

            for total_score, assignment, claimed in beam:
                for action in bot_candidates:
                    # Skip double-booked items
                    if action not in (DELIVER, IDLE) and action in claimed:
                        continue

                    action_score = score_lookup.get((bot.id, action), 0.0)
                    new_assignment = dict(assignment)
                    new_assignment[bot.id] = action

                    new_claimed = claimed
                    if action not in (DELIVER, IDLE):
                        new_claimed = claimed | frozenset([action])

                    new_beam.append((
                        total_score + action_score,
                        new_assignment,
                        new_claimed,
                    ))

            # Prune: keep top beam_width by score (highest first)
            if len(new_beam) > self.beam_width:
                new_beam.sort(key=lambda x: -x[0])
                beam = new_beam[:self.beam_width]
            else:
                beam = new_beam

            if not beam:
                beam = [(0.0, {}, frozenset())]

        # Return best
        beam.sort(key=lambda x: -x[0])
        return beam[0][1]

    @staticmethod
    def _encode_action(
        bot: Bot,
        action: str,
        state: GameState,
        path_engine: PathEngine,
        ctx: FeatureContext,
    ) -> torch.Tensor:
        """Encode DELIVER or IDLE as a 48-float feature vector."""
        from bot.models import Item

        if action == DELIVER:
            nearest_do = state.drop_off
            if ctx.drop_off_zones:
                nearest_do = min(
                    ctx.drop_off_zones,
                    key=lambda z: path_engine.distance(bot.position, z),
                )
            dummy = Item(id="__deliver__", type="__deliver__", position=nearest_do)
        else:
            dummy = Item(id="__idle__", type="__idle__", position=bot.position)

        return FeatureExtractor.encode_pair(bot, dummy, state, path_engine, ctx)
