from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Transition:
    state: np.ndarray # 当前环境状态 s 5*25*25
    action: int # AI 在这一步做出的具体动作 a（0~7 之间的整数）
    reward: float # 环境给的真实奖励值 r
    next_state: np.ndarray # 做出动作后，环境变成的下一个新状态 s'
    done: bool # 游戏是否结束了
    mask: np.ndarray # 当前状态下的动作掩码（哪些路能走）
    next_mask: np.ndarray # 下一个新状态下的动作掩码
    guided: bool = False # 这步动作是否由引导辅助完成的
    expert_action: int = -1 # 如果是引导的，专家的正确动作是什么（默认-1表示无）


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity # 记忆池最大容量
        self.alpha = alpha # 遵循优先级的强度
        self.buffer: List[Transition] = [] # 记忆池
        self.priorities = np.zeros((capacity,), dtype=np.float32) # 存对应位置记忆的优先级
        self.pos = 0 # 记录当前新记忆的存放位置

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        # 刚进来的记忆没有被训练过，所以我们默认它优先级最高
        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity # 自增，满了自动指向最旧的

    # 抽取记忆做训练
    def sample(self, batch_size: int, beta: float):
        size = len(self.buffer)
        prios = self.priorities[:size]
        # 把优先级prios变成概率分布probs
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(size, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # 重要性采样权重，随训练慢慢从 0.4 变成 1.0，用来乘以loss
        weights = (size * probs[indices]) ** (-beta)
        # 归一化
        weights /= weights.max()
        return indices, samples, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(max(prio, 1e-6))
