"""
ScorerMLP: scores (bot, item) pairs based on 48-float feature vectors.

Architecture (from spec section 4.1):
  Linear(48, 256) -> LayerNorm(256) -> ReLU
  Linear(256, 256) -> LayerNorm(256) -> ReLU
  Linear(256, 128) -> ReLU
  Linear(128, 1) -> sigmoid

Output in [0, 1]. ~144K parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ScorerMLP(nn.Module):
    """Scores quality of a (bot, item) assignment given state features."""

    def __init__(self, input_dim: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score batch of feature vectors. Input: (N, 48), Output: (N, 1)."""
        return self.net(x)
