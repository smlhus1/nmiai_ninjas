"""Tests for TrainingLogger."""
import pickle
import tempfile
from pathlib import Path

import pytest

from ml.training_logger import TrainingLogger, TrainingPoint


class TestTrainingLogger:
    def test_reward_5_basic(self):
        """reward_5 for round T = (score[T+5] - score[T]) / 10.0."""
        logger = TrainingLogger()
        # Scores: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
        for r in range(10):
            logger.on_round(r, r * 2, n_bots=1)

        points = logger.finalize("test")
        # Round 0: reward_5 = (10 - 0) / 10 = 1.0
        r0 = [p for p in points if p.round_num == 0]
        assert len(r0) == 1
        assert r0[0].reward_5 == pytest.approx(1.0)

        # Round 3: reward_5 = (16 - 6) / 10 = 1.0
        r3 = [p for p in points if p.round_num == 3]
        assert r3[0].reward_5 == pytest.approx(1.0)

    def test_reward_5_exact_index(self):
        """Round 0 reward uses score[5], not score[4] or score[6]."""
        logger = TrainingLogger()
        scores = [0, 1, 2, 3, 4, 50, 51, 52, 53, 54]
        for r, s in enumerate(scores):
            logger.on_round(r, s, n_bots=1)

        points = logger.finalize("test")
        r0 = [p for p in points if p.round_num == 0][0]
        # score[5] - score[0] = 50 - 0 = 50 -> 50/10 = 5.0 -> clamped to 1.0
        assert r0.reward_5 == pytest.approx(1.0)

        r1 = [p for p in points if p.round_num == 1][0]
        # score[6] - score[1] = 51 - 1 = 50 -> 1.0
        assert r1.reward_5 == pytest.approx(1.0)

    def test_last_rounds_use_available(self):
        """Last 5 rounds use available rounds, not 0.0."""
        logger = TrainingLogger()
        # 10 rounds: scores 0..9
        for r in range(10):
            logger.on_round(r, r, n_bots=1)

        points = logger.finalize("test")

        # Round 8: future = min(13, 9) = 9, reward = (9-8)/10 = 0.1
        r8 = [p for p in points if p.round_num == 8][0]
        assert r8.reward_5 == pytest.approx(0.1)

        # Round 9: future = min(14, 9) = 9, reward = (9-9)/10 = 0.0
        r9 = [p for p in points if p.round_num == 9][0]
        assert r9.reward_5 == pytest.approx(0.0)

    def test_multiple_bots(self):
        """n_bots=3 should produce 3 points per round."""
        logger = TrainingLogger()
        for r in range(5):
            logger.on_round(r, r * 10, n_bots=3)

        points = logger.finalize("test")
        assert len(points) == 5 * 3  # 5 rounds * 3 bots

    def test_save_load(self):
        """Saved pickle can be loaded and iterated."""
        logger = TrainingLogger()
        for r in range(5):
            logger.on_round(r, r, n_bots=2)
        logger.finalize("test")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        logger.save(path)

        with open(path, "rb") as f:
            loaded = pickle.load(f)

        assert len(loaded) == 10  # 5 rounds * 2 bots
        assert all(isinstance(p, TrainingPoint) for p in loaded)

        path.unlink()

    def test_reward_clamped(self):
        """Negative score delta should clamp to 0.0."""
        logger = TrainingLogger()
        # Score goes down
        scores = [100, 90, 80, 70, 60, 50]
        for r, s in enumerate(scores):
            logger.on_round(r, s, n_bots=1)

        points = logger.finalize("test")
        for p in points:
            assert p.reward_5 >= 0.0
