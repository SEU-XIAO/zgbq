from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Dict, List, Optional, Tuple

from battlefield_rl.map import BattlefieldMap

Coord = Tuple[int, int]

DIRS_8 = [
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
class PlanResult:
    path: List[Coord]
    jump_points: List[Coord]
    waypoints: List[Coord]


def heuristic(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _neighbors(grid: BattlefieldMap, node: Coord, max_height_diff: int, allow_diagonal: bool) -> List[Coord]:
    dirs = DIRS_8 if allow_diagonal else DIRS_8[:4]
    out: List[Coord] = []
    for dr, dc in dirs:
        nxt = (node[0] + dr, node[1] + dc)
        if grid.is_passable(node, nxt, max_height_diff):
            out.append(nxt)
    return out


def astar(grid: BattlefieldMap, start: Coord, goal: Coord, max_height_diff: int, allow_diagonal: bool = True) -> List[Coord]:
    if start == goal:
        return [start]

    open_heap: List[Tuple[float, Coord]] = []
    heapq.heappush(open_heap, (0.0, start))
    came_from: Dict[Coord, Optional[Coord]] = {start: None}
    g_score: Dict[Coord, float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            break

        for nxt in _neighbors(grid, current, max_height_diff, allow_diagonal):
            step_cost = heuristic(current, nxt)
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, nxt))

    if goal not in came_from:
        return []

    path: List[Coord] = []
    node: Optional[Coord] = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


def compress_jump_points(path: List[Coord]) -> List[Coord]:
    if len(path) <= 2:
        return path[:]

    jumps = [path[0]]
    prev_dr = path[1][0] - path[0][0]
    prev_dc = path[1][1] - path[0][1]
    for i in range(2, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        if (dr, dc) != (prev_dr, prev_dc):
            jumps.append(path[i - 1])
        prev_dr, prev_dc = dr, dc
    jumps.append(path[-1])
    return jumps


def interpolate_waypoints(path: List[Coord], interval: int = 22) -> List[Coord]:
    if not path:
        return []
    if len(path) == 1:
        return path[:]

    waypoints = [path[0]]
    acc = 0.0
    prev = path[0]
    for cur in path[1:]:
        seg = heuristic(prev, cur)
        acc += seg
        if acc >= interval:
            waypoints.append(cur)
            acc = 0.0
        prev = cur
    if waypoints[-1] != path[-1]:
        waypoints.append(path[-1])
    return waypoints


def plan_global_path(
    grid: BattlefieldMap,
    start: Coord,
    goal: Coord,
    max_height_diff: int,
    waypoint_interval: int,
    allow_diagonal: bool = True,
) -> PlanResult:
    path = astar(grid, start, goal, max_height_diff=max_height_diff, allow_diagonal=allow_diagonal)
    if not path:
        return PlanResult(path=[], jump_points=[], waypoints=[])
    jumps = compress_jump_points(path)
    waypoints = interpolate_waypoints(path, interval=waypoint_interval)
    return PlanResult(path=path, jump_points=jumps, waypoints=waypoints)
