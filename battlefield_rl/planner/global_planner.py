from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

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

DIRS_4 = DIRS_8[:4]


@dataclass
class PlanResult:
    path: List[Coord]
    jump_points: List[Coord]
    waypoints: List[Coord]
    planner: str = ""


def heuristic(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _neighbors(grid: BattlefieldMap, node: Coord, max_height_diff: int, allow_diagonal: bool) -> List[Coord]:
    dirs = DIRS_8 if allow_diagonal else DIRS_4
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


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _is_open(grid: BattlefieldMap, node: Coord, max_height_diff: int) -> bool:
    return grid.in_bounds(node[0], node[1]) and grid.is_passable(node, node, max_height_diff)


def _can_step(grid: BattlefieldMap, node: Coord, direction: Coord, max_height_diff: int) -> bool:
    nxt = (node[0] + direction[0], node[1] + direction[1])
    return grid.is_passable(node, nxt, max_height_diff)


def _blocked(grid: BattlefieldMap, node: Coord, max_height_diff: int) -> bool:
    return not _is_open(grid, node, max_height_diff)


def _has_forced_neighbor(grid: BattlefieldMap, node: Coord, direction: Coord, max_height_diff: int) -> bool:
    dr, dc = direction
    row, col = node

    if dr != 0 and dc != 0:
        return (
            _blocked(grid, (row - dr, col), max_height_diff)
            and _is_open(grid, (row - dr, col + dc), max_height_diff)
        ) or (
            _blocked(grid, (row, col - dc), max_height_diff)
            and _is_open(grid, (row + dr, col - dc), max_height_diff)
        )

    if dr != 0:
        return (
            _blocked(grid, (row, col + 1), max_height_diff)
            and _is_open(grid, (row + dr, col + 1), max_height_diff)
        ) or (
            _blocked(grid, (row, col - 1), max_height_diff)
            and _is_open(grid, (row + dr, col - 1), max_height_diff)
        )

    if dc != 0:
        return (
            _blocked(grid, (row + 1, col), max_height_diff)
            and _is_open(grid, (row + 1, col + dc), max_height_diff)
        ) or (
            _blocked(grid, (row - 1, col), max_height_diff)
            and _is_open(grid, (row - 1, col + dc), max_height_diff)
        )

    return False


def _jump(
    grid: BattlefieldMap,
    node: Coord,
    direction: Coord,
    goal: Coord,
    max_height_diff: int,
) -> Optional[Coord]:
    dr, dc = direction
    current = node

    while True:
        nxt = (current[0] + dr, current[1] + dc)
        if not grid.is_passable(current, nxt, max_height_diff):
            return None
        if nxt == goal:
            return nxt
        if _has_forced_neighbor(grid, nxt, direction, max_height_diff):
            return nxt
        if dr != 0 and dc != 0:
            if _jump(grid, nxt, (dr, 0), goal, max_height_diff) is not None:
                return nxt
            if _jump(grid, nxt, (0, dc), goal, max_height_diff) is not None:
                return nxt
        current = nxt


def _pruned_directions(
    grid: BattlefieldMap,
    node: Coord,
    parent: Optional[Coord],
    max_height_diff: int,
    allow_diagonal: bool,
) -> List[Coord]:
    if parent is None:
        return DIRS_8 if allow_diagonal else DIRS_4

    dr = _sign(node[0] - parent[0])
    dc = _sign(node[1] - parent[1])
    directions: List[Coord] = []

    def add(direction: Coord) -> None:
        if direction == (0, 0):
            return
        if not allow_diagonal and direction[0] != 0 and direction[1] != 0:
            return
        if direction not in directions and _can_step(grid, node, direction, max_height_diff):
            directions.append(direction)

    row, col = node
    if dr != 0 and dc != 0:
        add((dr, dc))
        add((dr, 0))
        add((0, dc))
        if _blocked(grid, (row - dr, col), max_height_diff):
            add((-dr, dc))
        if _blocked(grid, (row, col - dc), max_height_diff):
            add((dr, -dc))
    elif dr != 0:
        add((dr, 0))
        if _blocked(grid, (row, col + 1), max_height_diff):
            add((dr, 1))
        if _blocked(grid, (row, col - 1), max_height_diff):
            add((dr, -1))
    elif dc != 0:
        add((0, dc))
        if _blocked(grid, (row + 1, col), max_height_diff):
            add((1, dc))
        if _blocked(grid, (row - 1, col), max_height_diff):
            add((-1, dc))

    return directions


def _reconstruct(came_from: Dict[Coord, Optional[Coord]], goal: Coord) -> List[Coord]:
    path: List[Coord] = []
    node: Optional[Coord] = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


def _expand_jump_path(grid: BattlefieldMap, jump_path: List[Coord], max_height_diff: int) -> List[Coord]:
    if len(jump_path) <= 1:
        return jump_path[:]

    full_path = [jump_path[0]]
    for start, end in zip(jump_path, jump_path[1:]):
        dr = _sign(end[0] - start[0])
        dc = _sign(end[1] - start[1])
        row_delta = abs(end[0] - start[0])
        col_delta = abs(end[1] - start[1])

        if not (row_delta == col_delta or row_delta == 0 or col_delta == 0):
            return []

        current = start
        for _ in range(max(row_delta, col_delta)):
            nxt = (current[0] + dr, current[1] + dc)
            if not grid.is_passable(current, nxt, max_height_diff):
                return []
            full_path.append(nxt)
            current = nxt

    return full_path


def _valid_path(grid: BattlefieldMap, path: List[Coord], max_height_diff: int) -> bool:
    if not path:
        return False
    return all(grid.is_passable(cur, nxt, max_height_diff) for cur, nxt in zip(path, path[1:]))


def jps(
    grid: BattlefieldMap,
    start: Coord,
    goal: Coord,
    max_height_diff: int,
    allow_diagonal: bool = True,
) -> List[Coord]:
    """Jump Point Search with A* kept as the caller-side safety fallback."""
    if start == goal:
        return [start]
    if not _is_open(grid, start, max_height_diff) or not _is_open(grid, goal, max_height_diff):
        return []

    open_heap: List[Tuple[float, int, Coord]] = []
    counter = 0
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start))
    came_from: Dict[Coord, Optional[Coord]] = {start: None}
    g_score: Dict[Coord, float] = {start: 0.0}
    closed: Set[Coord] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            jump_path = _reconstruct(came_from, goal)
            path = _expand_jump_path(grid, jump_path, max_height_diff)
            return path if _valid_path(grid, path, max_height_diff) else []

        closed.add(current)
        parent = came_from[current]
        for direction in _pruned_directions(grid, current, parent, max_height_diff, allow_diagonal):
            jump_point = _jump(grid, current, direction, goal, max_height_diff)
            if jump_point is None or jump_point in closed:
                continue

            tentative_g = g_score[current] + heuristic(current, jump_point)
            if tentative_g < g_score.get(jump_point, float("inf")):
                came_from[jump_point] = current
                g_score[jump_point] = tentative_g
                counter += 1
                heapq.heappush(open_heap, (tentative_g + heuristic(jump_point, goal), counter, jump_point))

    return []


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


def interpolate_waypoints(
    path: List[Coord],
    interval: int = 22,
    max_chebyshev: Optional[int] = None,
) -> List[Coord]:
    if not path:
        return []
    if len(path) == 1:
        return path[:]

    if max_chebyshev is not None:
        step_limit = max(1, min(interval, max_chebyshev))
        waypoints = [path[0]]
        steps_since_anchor = 0
        for cur in path[1:]:
            steps_since_anchor += 1
            if steps_since_anchor >= step_limit:
                waypoints.append(cur)
                steps_since_anchor = 0
        if waypoints[-1] != path[-1]:
            waypoints.append(path[-1])
        return waypoints

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
    use_jps: bool = True,
    waypoint_max_chebyshev: Optional[int] = None,
) -> PlanResult:
    planner = "jps"
    path = jps(grid, start, goal, max_height_diff=max_height_diff, allow_diagonal=allow_diagonal) if use_jps else []
    if not path:
        planner = "astar"
        path = astar(grid, start, goal, max_height_diff=max_height_diff, allow_diagonal=allow_diagonal)
    if not path:
        return PlanResult(path=[], jump_points=[], waypoints=[], planner="failed")
    jumps = compress_jump_points(path)
    waypoints = interpolate_waypoints(path, interval=waypoint_interval, max_chebyshev=waypoint_max_chebyshev)
    return PlanResult(path=path, jump_points=jumps, waypoints=waypoints, planner=planner)
