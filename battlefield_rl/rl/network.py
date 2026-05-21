from __future__ import annotations

import random

import torch
import torch.nn as nn


class TacticalD3QN(nn.Module):
    def __init__(self, in_channels: int = 5, num_actions: int = 8, window_size: int = 25):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, window_size, window_size)
            flat_dim = self.feature(dummy).view(1, -1).shape[1]

        self.value_head = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x).flatten(1)
        value = self.value_head(feat)
        adv = self.adv_head(feat)
        return value + (adv - adv.mean(dim=1, keepdim=True))


def masked_q_values(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return q + (1.0 - mask) * (-1e9)


def epsilon_greedy(q_masked: torch.Tensor, mask: torch.Tensor, epsilon: float) -> int:
    valid = (mask > 0.5).nonzero(as_tuple=False).flatten().tolist()
    if not valid:
        return 0
    if random.random() < epsilon:
        return random.choice(valid)
    return int(torch.argmax(q_masked).item())
