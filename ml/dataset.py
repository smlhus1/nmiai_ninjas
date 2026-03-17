"""
TrainingDataset: PyTorch Dataset wrapper for training data.

Loads pickled list of (features_48, reward) tuples from collect_training_data.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    """Dataset of (features, reward) pairs for ScorerMLP training."""

    def __init__(self, path: Path) -> None:
        with open(path, "rb") as f:
            self._data: list[tuple[torch.Tensor, float]] = pickle.load(f)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, reward = self._data[idx]
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        return features, torch.tensor([reward], dtype=torch.float32)
