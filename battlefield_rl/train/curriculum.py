from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class RollingMetrics:
    success_rate: float
    timeout_rate: float
    path_efficiency: float


class StageGate:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.success_hist: Deque[float] = deque(maxlen=window_size)
        self.timeout_hist: Deque[float] = deque(maxlen=window_size)
        self.eff_hist: Deque[float] = deque(maxlen=window_size)

    def update(self, reached_goal: bool, timeout: bool, path_efficiency: float) -> None:
        self.success_hist.append(1.0 if reached_goal else 0.0)
        self.timeout_hist.append(1.0 if timeout else 0.0)
        self.eff_hist.append(float(path_efficiency))

    def ready(self) -> bool:
        return len(self.success_hist) == self.window_size

    def summary(self) -> RollingMetrics:
        if not self.success_hist:
            return RollingMetrics(0.0, 1.0, 999.0)
        return RollingMetrics(
            success_rate=sum(self.success_hist) / len(self.success_hist),
            timeout_rate=sum(self.timeout_hist) / len(self.timeout_hist),
            path_efficiency=sum(self.eff_hist) / len(self.eff_hist),
        )
