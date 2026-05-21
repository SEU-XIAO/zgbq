from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from battlefield_rl.config import EnvConfig
from battlefield_rl.map import BattlefieldMap

Coord = Tuple[int, int]

ACTIONS_8: List[Coord] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]


@dataclass
class EnemySpec:
    row: int
    col: int
    facing_deg: float
    fov_deg: float
    max_range: int


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, object]


def bresenham_line(r0: int, c0: int, r1: int, c1: int) -> List[Coord]:
    points: List[Coord] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return points


def angle_deg(from_rc: Coord, to_rc: Coord) -> float:
    dr = to_rc[0] - from_rc[0]
    dc = to_rc[1] - from_rc[1]
    rad = math.atan2(-dr, dc)
    return (math.degrees(rad) + 360.0) % 360.0


def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


class TacticalBattlefieldEnv:
    def __init__(self, grid: BattlefieldMap, config: EnvConfig):
        self.grid = grid
        self.cfg = config
        self.window_radius = config.window_size // 2

        self.start: Coord = (0, 0)
        self.goal: Coord = (0, 0)
        self.pos: Coord = (0, 0)
        self.enemies: List[EnemySpec] = []
        self.waypoints: List[Coord] = []
        self.waypoint_idx: int = 0

        self.steps = 0
        self.no_progress_steps = 0
        self.prev_goal_dist = float("inf")

        self.visited = np.zeros(self.grid.shape, dtype=np.float32)
        ws = self.cfg.window_size
        self.current_threat_local = np.zeros((ws, ws), dtype=np.float32)
        self.current_visible_local = np.zeros((ws, ws), dtype=np.float32)
        self.threat_scale = 1.0

    def reset(
        self,
        start: Coord,
        goal: Coord,
        enemies: Sequence[EnemySpec],
        waypoints: Sequence[Coord],
        threat_scale: float = 1.0,
    ) -> np.ndarray:
        self.start = start
        self.goal = goal
        self.pos = start
        self.enemies = list(enemies)
        self.waypoints = list(waypoints) if waypoints else [goal]
        self.waypoint_idx = 0
        self.steps = 0
        self.no_progress_steps = 0
        self.prev_goal_dist = self._euclidean(self.pos, self.goal)
        self.visited.fill(0.0)
        self.visited[self.pos[0], self.pos[1]] = 1.0
        self.threat_scale = threat_scale
        self._update_threat_cache()
        return self.build_observation()

    def current_waypoint(self) -> Coord:
        return self.waypoints[min(self.waypoint_idx, len(self.waypoints) - 1)]

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(len(ACTIONS_8), dtype=np.float32)
        for i, (dr, dc) in enumerate(ACTIONS_8):
            nxt = (self.pos[0] + dr, self.pos[1] + dc)
            if self.grid.is_passable(self.pos, nxt, 0):
                mask[i] = 1.0
        return mask

    def step(self, action: int) -> StepResult:
        self.steps += 1
        mask = self.action_mask()
        valid = mask[action] > 0.5

        old_pos = self.pos
        if valid:
            dr, dc = ACTIONS_8[action]
            self.pos = (self.pos[0] + dr, self.pos[1] + dc)

        self.visited *= 0.97
        self.visited[self.pos[0], self.pos[1]] = 1.0

        self._update_threat_cache()

        reward = self.cfg.step_penalty
        reward += self._progress_reward(old_pos, self.pos)

        half = self.window_radius
        threat = float(self.current_threat_local[half, half])
        reward += self.cfg.exposure_penalty_scale * self.threat_scale * threat

        visible_event = float(self.current_visible_local[half, half])
        reward += self.cfg.visible_event_penalty * self.threat_scale * visible_event

        revisit = float(self.visited[self.pos[0], self.pos[1]])
        reward += self.cfg.revisit_penalty_scale * revisit

        if not valid:
            reward -= 0.1

        wp = self.current_waypoint()
        if self.pos == wp and self.waypoint_idx < len(self.waypoints) - 1:
            self.waypoint_idx += 1
            reward += self.cfg.waypoint_reward

        done = False
        timeout = self.steps >= self.cfg.timeout_steps

        cur_dist = self._euclidean(self.pos, self.goal)
        if cur_dist + 1e-6 < self.prev_goal_dist:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1
        self.prev_goal_dist = cur_dist
        stuck = self.no_progress_steps >= self.cfg.no_progress_limit

        if self.pos == self.goal:
            reward += self.cfg.goal_reward
            done = True
        elif timeout or stuck:
            reward += self.cfg.timeout_penalty
            done = True

        obs = self.build_observation()
        info: Dict[str, object] = {
            "mask": mask,
            "threat": threat,
            "visible": bool(visible_event > 0.5),
            "timeout": timeout,
            "stuck": stuck,
            "waypoint_index": self.waypoint_idx,
            "distance_to_goal": cur_dist,
        }
        return StepResult(observation=obs, reward=float(reward), done=done, info=info)

    def build_observation(self) -> np.ndarray:
        size = self.cfg.window_size
        half = self.window_radius
        ch_free = np.zeros((size, size), dtype=np.float32)
        ch_block = np.ones((size, size), dtype=np.float32)
        ch_threat = self.current_threat_local.copy()
        ch_wp = np.zeros((size, size), dtype=np.float32)
        ch_visit = np.zeros((size, size), dtype=np.float32)

        for wr in range(size):
            for wc in range(size):
                gr = self.pos[0] + (wr - half)
                gc = self.pos[1] + (wc - half)
                if not self.grid.in_bounds(gr, gc):
                    continue
                blocked = 1.0 if self.grid.is_static_blocked(gr, gc) else 0.0
                ch_block[wr, wc] = blocked
                ch_free[wr, wc] = 1.0 - blocked
                ch_visit[wr, wc] = self.visited[gr, gc]

        wp = self.current_waypoint()
        wr = wp[0] - self.pos[0] + half
        wc = wp[1] - self.pos[1] + half
        if 0 <= wr < size and 0 <= wc < size:
            ch_wp[wr, wc] = 1.0

        return np.stack([ch_free, ch_block, ch_threat, ch_wp, ch_visit], axis=0)

    def _update_threat_cache(self) -> None:
        self.current_threat_local, self.current_visible_local = self.render_local_threat_map()

    def render_local_threat_map(self) -> Tuple[np.ndarray, np.ndarray]:
        size = self.cfg.window_size
        half = self.window_radius

        threat = np.zeros((size, size), dtype=np.float32)
        visible = np.zeros((size, size), dtype=np.float32)

        win_r_min = self.pos[0] - half
        win_r_max = self.pos[0] + half
        win_c_min = self.pos[1] - half
        win_c_max = self.pos[1] + half

        rows, cols = self.grid.shape

        for enemy in self.enemies:
            src = (enemy.row, enemy.col)
            fov_r_min = enemy.row - enemy.max_range
            fov_r_max = enemy.row + enemy.max_range
            fov_c_min = enemy.col - enemy.max_range
            fov_c_max = enemy.col + enemy.max_range

            r_min = max(win_r_min, fov_r_min, 0)
            r_max = min(win_r_max, fov_r_max, rows - 1)
            c_min = max(win_c_min, fov_c_min, 0)
            c_max = min(win_c_max, fov_c_max, cols - 1)
            if r_min > r_max or c_min > c_max:
                continue

            for gr in range(r_min, r_max + 1):
                for gc in range(c_min, c_max + 1):
                    dst = (gr, gc)
                    d = self._euclidean(src, dst)
                    if d > enemy.max_range:
                        continue
                    ang = angle_deg(src, dst)
                    if angle_diff(ang, enemy.facing_deg) > enemy.fov_deg * 0.5:
                        continue
                    if not self._visible_by_obstacle(src, dst):
                        continue

                    wr = gr - self.pos[0] + half
                    wc = gc - self.pos[1] + half
                    if not (0 <= wr < size and 0 <= wc < size):
                        continue

                    val = max(0.0, 1.0 - d / max(1.0, float(enemy.max_range)))
                    threat[wr, wc] = 1.0 - (1.0 - threat[wr, wc]) * (1.0 - val)
                    visible[wr, wc] = 1.0

        return threat, visible

    def _visible_by_obstacle(self, src: Coord, dst: Coord) -> bool:
        pts = bresenham_line(src[0], src[1], dst[0], dst[1])
        if len(pts) <= 2:
            return True
        for r, c in pts[1:-1]:
            if self.grid.is_static_blocked(r, c):
                return False
        return True

    def _progress_reward(self, old_pos: Coord, new_pos: Coord) -> float:
        d0 = self._euclidean(old_pos, self.goal)
        d1 = self._euclidean(new_pos, self.goal)
        return self.cfg.progress_reward_scale * (d0 - d1)

    @staticmethod
    def _euclidean(a: Coord, b: Coord) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def should_replan(self) -> bool:
        return self.no_progress_steps >= self.cfg.no_progress_limit
