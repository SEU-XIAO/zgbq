from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    mask: np.ndarray
    next_mask: np.ndarray
    guided: bool = False
    expert_action: int = -1


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Transition] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        size = len(self.buffer)
        prios = self.priorities[:size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(size, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        weights = (size * probs[indices]) ** (-beta)
        weights /= weights.max()
        return indices, samples, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(max(prio, 1e-6))
