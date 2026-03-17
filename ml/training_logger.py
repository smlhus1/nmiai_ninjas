"""
TrainingLogger: captures (round, score, bot_decisions) during sim runs.

Reward_5 is computed in post-processing (not online) because it depends
on future rounds. Last 5 rounds use available rounds and normalize accordingly.

Usage:
    logger = TrainingLogger()
    # During game loop:
    logger.on_round(round_num, score)
    # After game:
    points = logger.finalize(game_id="game_0")
    logger.save(Path("data/training.pkl"))
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingPoint:
    """One training data point: features + reward for a (bot, round) pair."""
    game_id: str
    round_num: int
    bot_id: int
    reward_5: float
    # Features are attached separately during collection (not stored here)


@dataclass
class RoundRecord:
    """Score at a specific round."""
    round_num: int
    score: int


class TrainingLogger:
    """Logs per-round scores and computes reward_5 in post-processing."""

    def __init__(self) -> None:
        self._rounds: list[RoundRecord] = []
        self._all_points: list[TrainingPoint] = []
        self._n_bots: int = 0

    def on_round(self, round_num: int, score: int, n_bots: int = 20) -> None:
        """Log score after a round. Call once per round."""
        self._rounds.append(RoundRecord(round_num=round_num, score=score))
        self._n_bots = n_bots

    def finalize(self, game_id: str = "game_0") -> list[TrainingPoint]:
        """Compute reward_5 for all rounds and return training points.

        reward_5 = (score[T+5] - score[T]) / 10.0, clamped to [0, 1].
        For the last rounds where T+5 exceeds game length, use the last
        available round instead.
        """
        scores = {r.round_num: r.score for r in self._rounds}
        sorted_rounds = sorted(scores.keys())

        if not sorted_rounds:
            return []

        points: list[TrainingPoint] = []
        max_round = sorted_rounds[-1]

        for r in sorted_rounds:
            # Find the score 5 rounds ahead (or last available)
            future_r = min(r + 5, max_round)
            score_now = scores[r]
            score_future = scores.get(future_r, score_now)
            delta = score_future - score_now
            reward = max(0.0, min(delta / 10.0, 1.0))

            for bot_id in range(self._n_bots):
                points.append(TrainingPoint(
                    game_id=game_id,
                    round_num=r,
                    bot_id=bot_id,
                    reward_5=reward,
                ))

        self._all_points.extend(points)
        return points

    def save(self, path: Path) -> None:
        """Save all accumulated training points to pickle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._all_points, f, protocol=pickle.HIGHEST_PROTOCOL)

    def reset(self) -> None:
        """Reset for next game (keeps accumulated points)."""
        self._rounds.clear()

    @property
    def total_points(self) -> int:
        return len(self._all_points)
