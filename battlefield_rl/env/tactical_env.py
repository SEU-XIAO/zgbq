from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from battlefield_rl.config import EnvConfig
from battlefield_rl.map import BattlefieldMap

Coord = Tuple[int, int]

# 8个方向动作
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
    # 行列
    row: int
    col: int
    # 面朝
    facing_deg: float
    # 张角
    fov_deg: float
    # 视长
    max_range: int


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    # 调试信息
    info: Dict[str, object]


def bresenham_line(r0: int, c0: int, r1: int, c1: int) -> List[Coord]:
    # 画线算法，找到最接近一条两点间直线的格子序列
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
    # 计算从一个点 from_rc 指向另一个点 to_rc 的绝对角度
    dr = to_rc[0] - from_rc[0]
    dc = to_rc[1] - from_rc[1]
    rad = math.atan2(-dr, dc)
    return (math.degrees(rad) + 360.0) % 360.0


def angle_diff(a: float, b: float) -> float:
    # 计算两个角度 a 和 b 之间的最小夹角
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


class TacticalBattlefieldEnv:
    def __init__(self, grid: BattlefieldMap, config: EnvConfig):
        self.grid = grid # 传入战场对象
        self.cfg = config # 传进环境配置参数
        self.window_radius = config.window_size // 2 # 计算局部视野的半径，这里是正方形边长的一半

        self.start: Coord = (0, 0) # 智能体视野中看到的起点
        self.goal: Coord = (0, 0) # 智能体视野中看到的终点
        self.pos: Coord = (0, 0) # 智能体所处的实时位置
        self.enemies: List[EnemySpec] = [] # 传入的敌人数组
        self.waypoints: List[Coord] = [] # 路标列表
        self.waypoint_idx: int = 0 # 当前阶段路标索引

        self.steps = 0 # 总步数计数器
        self.no_progress_steps = 0 # 无进展（原地打转/卡住）的步数计数器。
        self.prev_goal_dist = float("inf") # 智能体在上一步距离目标的距离，初始值设为正无穷

        self.visited = np.zeros(self.grid.shape, dtype=np.float32) # 足迹地图
        ws = self.cfg.window_size # 局部视野正方形边长
        self.current_threat_local = np.zeros((ws, ws), dtype=np.float32) # 当前局部视野内的敌人威胁程度（比如离敌人越近，格子里的数值越高）。
        self.current_visible_local = np.zeros((ws, ws), dtype=np.float32) # 标记可见性
        self.threat_scale = 1.0 # 威胁程度的缩放系数

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
        self.waypoints = list(waypoints) if waypoints else [goal] # 若为空列表，goal作为唯一路标
        self.waypoint_idx = 0 # 从第一个路标开始走。
        self.steps = 0
        self.no_progress_steps = 0
        self.prev_goal_dist = self._euclidean(self.pos, self.goal) # 计算并记录初始状态下，当前位置到终点的欧几里得距离（直线距离）。
        self.visited.fill(0.0)
        self.visited[self.pos[0], self.pos[1]] = 1.0
        self.threat_scale = threat_scale
        self._update_threat_cache() # 只有第一次需要计算敌人威胁的热力图，后续读缓存复用就可以了
        return self.build_observation() # 生成并返回第一帧视野。

    def current_waypoint(self) -> Coord:
        return self.waypoints[min(self.waypoint_idx, len(self.waypoints) - 1)]

    # 掩码为1是可通行
    def action_mask(self) -> np.ndarray:
        mask = np.zeros(len(ACTIONS_8), dtype=np.float32)
        for i, (dr, dc) in enumerate(ACTIONS_8):
            nxt = (self.pos[0] + dr, self.pos[1] + dc)
            if self.grid.is_passable(self.pos, nxt, 0):
                mask[i] = 1.0
        return mask

    def heuristic_action(
        self,
        distance_weight: float = 1.0, # 渴望前进的程度。权重越高，越倾向于抄近路直奔终点。
        threat_weight: float = 1.5, # 怕死的程度。由于设为了 1.5（最高），说明该算法宁可绕远路，也绝对要避开敌人。
        revisit_weight: float = 0.2, # 讨厌重复的程度。防止智能体在两个格子之间来回鬼畜打转。
    ) -> int: # 返回得分最高的动作索引 best_action（0 ~ 7 的整数）
        mask = self.action_mask()
        best_action = 0 # 初始化最优动作为 0
        best_score = -float("inf") # 后面计算出的任何有效得分都能覆盖它
        current_dist = self._euclidean(self.pos, self.goal)
        half = self.window_radius

        for i, (dr, dc) in enumerate(ACTIONS_8):
            if mask[i] <= 0.5: # 碰壁过滤
                continue
            nxt = (self.pos[0] + dr, self.pos[1] + dc)
            progress = current_dist - self._euclidean(nxt, self.goal)
            # 坐标转换。将全局地图上的坐标 nxt 转换映射到以自身为中心的局部视野矩阵（current_threat_local）中的行列索引 (wr, wc)。
            wr = nxt[0] - self.pos[0] + half
            wc = nxt[1] - self.pos[1] + half
            threat = 0.0
            if 0 <= wr < self.cfg.window_size and 0 <= wc < self.cfg.window_size:
                threat = float(self.current_threat_local[wr, wc])

            revisit = float(self.visited[nxt[0], nxt[1]])
            # 进行打分
            score = distance_weight * progress - threat_weight * threat - revisit_weight * revisit
            if score > best_score:
                best_score = score
                best_action = i
        # 返回最高分的启发式动作索引
        return best_action

    def step(self, action: int) -> StepResult: # 接受agent的action并返回步进结果
        self.steps += 1
        mask = self.action_mask()
        valid = mask[action] > 0.5

        old_pos = self.pos
        if valid: # 如果动作不合法（撞墙），则保持原坐标不动（原地踏步）。
            dr, dc = ACTIONS_8[action]
            self.pos = (self.pos[0] + dr, self.pos[1] + dc)

        self.visited *= 0.97 # 足迹害怕衰减机制
        self.visited[self.pos[0], self.pos[1]] = 1.0

        self._update_threat_cache()

        # 时间惩罚
        reward = self.cfg.step_penalty
        # 推进奖励
        reward += self._progress_reward(old_pos, self.pos)

        # 处于敌人威胁下的惩罚
        half = self.window_radius
        threat = float(self.current_threat_local[half, half])
        reward += self.cfg.exposure_penalty_scale * self.threat_scale * threat

        # 处于敌人可见性格子下的惩罚
        visible_event = float(self.current_visible_local[half, half])
        reward += self.cfg.visible_event_penalty * self.threat_scale * visible_event

        # 惩罚重复访问
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
