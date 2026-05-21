from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import List, Optional, Tuple

import numpy as np

from battlefield_rl.config import TrainingSceneConfig
from battlefield_rl.map.loader import BattlefieldMap

Coord = Tuple[int, int]


@dataclass(frozen=True)
class TrainingScene:
    grid: BattlefieldMap
    start: Coord
    goal: Coord
    enemies: List[object]
    source: str
    enemy_case: str


class LocalSceneSampler:
    def __init__(self, base_grid: BattlefieldMap, cfg: TrainingSceneConfig, window_size: int):
        self.base_grid = base_grid
        self.cfg = cfg
        self.window_size = window_size
        self.scene_half = cfg.scene_size // 2
        self.view_half = window_size // 2

    def sample(self, stage: int) -> TrainingScene:
        for _ in range(self.cfg.max_sample_attempts):
            grid, source = self._sample_grid()
            start = (self.scene_half, self.scene_half)
            goal = self._sample_goal(grid, start)
            if goal is None:
                continue
            enemies, enemy_case = self._sample_enemies(stage, start)
            return TrainingScene(grid=grid, start=start, goal=goal, enemies=enemies, source=source, enemy_case=enemy_case)

        grid = self._empty_grid()
        start = (self.scene_half, self.scene_half)
        goal = (self.scene_half, min(self.cfg.scene_size - 1, self.scene_half + self.cfg.goal_min_distance))
        return TrainingScene(grid=grid, start=start, goal=goal, enemies=[], source="fallback", enemy_case="none")

    def _sample_grid(self) -> Tuple[BattlefieldMap, str]:
        if random.random() < self.cfg.procedural_ratio:
            return self._procedural_grid(), "procedural"
        cropped = self._crop_real_grid()
        if cropped is not None:
            return cropped, "real_crop"
        return self._procedural_grid(), "procedural"

    def _empty_grid(self) -> BattlefieldMap:
        size = self.cfg.scene_size
        return BattlefieldMap(
            heights=np.zeros((size, size), dtype=np.int16),
            types=np.zeros((size, size), dtype=np.int8),
        )

    def _procedural_grid(self) -> BattlefieldMap:
        size = self.cfg.scene_size
        grid = np.zeros((size, size), dtype=np.int8)

        # 轻量程序化结构：块状建筑、短墙、门洞，比纯随机噪声更接近LoS训练需要。
        for _ in range(random.randint(3, 7)):
            h = random.randint(2, 6)
            w = random.randint(2, 8)
            r = random.randint(1, size - h - 1)
            c = random.randint(1, size - w - 1)
            grid[r : r + h, c : c + w] = random.choice((1, 2))

        for _ in range(random.randint(2, 5)):
            horizontal = random.random() < 0.5
            length = random.randint(8, 20)
            if horizontal:
                r = random.randint(2, size - 3)
                c = random.randint(1, size - length - 1)
                grid[r, c : c + length] = random.choice((1, 2))
                gap = random.randint(c + 1, c + length - 2)
                grid[r, gap] = 0
            else:
                r = random.randint(1, size - length - 1)
                c = random.randint(2, size - 3)
                grid[r : r + length, c] = random.choice((1, 2))
                gap = random.randint(r + 1, r + length - 2)
                grid[gap, c] = 0

        center = self.scene_half
        grid[center - 1 : center + 2, center - 1 : center + 2] = 0
        heights = np.zeros((size, size), dtype=np.int16)
        return BattlefieldMap(heights=heights, types=grid)

    def _crop_real_grid(self) -> Optional[BattlefieldMap]:
        rows, cols = self.base_grid.shape
        size = self.cfg.scene_size
        if rows < size or cols < size:
            return None

        for _ in range(20):
            r = random.randint(0, rows - size)
            c = random.randint(0, cols - size)
            types = self.base_grid.types[r : r + size, c : c + size].copy()
            center = self.scene_half
            if int(types[center, center]) in (1, 2):
                continue
            heights = np.zeros((size, size), dtype=np.int16)
            return BattlefieldMap(heights=heights, types=types.astype(np.int8))
        return None

    def _sample_goal(self, grid: BattlefieldMap, start: Coord) -> Optional[Coord]:
        candidates: List[Coord] = []
        r_min = max(0, start[0] - self.view_half)
        r_max = min(self.cfg.scene_size - 1, start[0] + self.view_half)
        c_min = max(0, start[1] - self.view_half)
        c_max = min(self.cfg.scene_size - 1, start[1] + self.view_half)
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if grid.is_static_blocked(r, c):
                    continue
                d = abs(r - start[0]) + abs(c - start[1])
                if self.cfg.goal_min_distance <= d <= self.cfg.goal_max_distance:
                    candidates.append((r, c))
        random.shuffle(candidates)
        for goal in candidates:
            if self._reachable(grid, start, goal):
                return goal
        return None

    def _reachable(self, grid: BattlefieldMap, start: Coord, goal: Coord) -> bool:
        q = deque([start])
        seen = {start}
        while q:
            cur = q.popleft()
            if cur == goal:
                return True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nxt = (cur[0] + dr, cur[1] + dc)
                    if nxt in seen or not grid.in_bounds(nxt[0], nxt[1]):
                        continue
                    if grid.is_static_blocked(nxt[0], nxt[1]):
                        continue
                    seen.add(nxt)
                    q.append(nxt)
        return False

    def _sample_enemies(self, stage: int, start: Coord) -> Tuple[List[object], str]:
        case = self._sample_enemy_case(stage)
        if case == "none":
            return [], case
        if case == "inside25":
            return [self._sample_inside_enemy(start)], case
        return [self._sample_outer35_enemy(start)], case

    def _sample_enemy_case(self, stage: int) -> str:
        if stage <= 1:
            return "none"
        if stage == 2:
            return "inside25" if random.random() < self.cfg.stage2_inside_ratio else "none"

        p = random.random()
        if p < self.cfg.stage3_inside_ratio:
            return "inside25"
        if p < self.cfg.stage3_inside_ratio + self.cfg.stage3_outer_ratio:
            return "outer35"
        return "none"

    def _sample_inside_enemy(self, start: Coord) -> object:
        r = random.randint(start[0] - self.view_half, start[0] + self.view_half)
        c = random.randint(start[1] - self.view_half, start[1] + self.view_half)
        if (r, c) == start:
            r = min(self.cfg.scene_size - 1, r + 1)
        return self._make_enemy((r, c), start)

    def _sample_outer35_enemy(self, start: Coord) -> object:
        candidates: List[Coord] = []
        for r in range(self.cfg.scene_size):
            for c in range(self.cfg.scene_size):
                inside25 = abs(r - start[0]) <= self.view_half and abs(c - start[1]) <= self.view_half
                if not inside25:
                    candidates.append((r, c))
        return self._make_enemy(random.choice(candidates), start)

    def _make_enemy(self, rc: Coord, look_at: Coord) -> object:
        from battlefield_rl.env import EnemySpec

        facing = self._angle_towards(rc, look_at)
        jitter = random.uniform(-self.cfg.enemy_facing_jitter_deg, self.cfg.enemy_facing_jitter_deg)
        return EnemySpec(
            row=rc[0],
            col=rc[1],
            facing_deg=(facing + jitter) % 360.0,
            fov_deg=self.cfg.enemy_fov_deg,
            max_range=self.cfg.enemy_range,
        )

    @staticmethod
    def _angle_towards(src: Coord, dst: Coord) -> float:
        dr = dst[0] - src[0]
        dc = dst[1] - src[1]
        return (np.degrees(np.arctan2(-dr, dc)) + 360.0) % 360.0
