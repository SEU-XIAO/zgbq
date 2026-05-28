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
    planner: str = "" #标志是哪种规划器


def heuristic(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

# 切比雪夫距离，如果上下左右代价一样
def chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _neighbors(grid: BattlefieldMap, node: Coord, max_height_diff: int, allow_diagonal: bool) -> List[Coord]:
    # 是否允许斜角对穿
    dirs = DIRS_8 if allow_diagonal else DIRS_4
    out: List[Coord] = []
    for dr, dc in dirs:
        nxt = (node[0] + dr, node[1] + dc)
        if grid.is_passable(node, nxt, max_height_diff):
            out.append(nxt)
    return out

# 只考虑距离影响的a*
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

# 跳点搜索的寻找强迫邻居
def _has_forced_neighbor(grid: BattlefieldMap, node: Coord, direction: Coord, max_height_diff: int) -> bool:
    dr, dc = direction
    row, col = node
    # 斜角移动寻找
    if dr != 0 and dc != 0:
        return (
            _blocked(grid, (row - dr, col), max_height_diff)
            and _is_open(grid, (row - dr, col + dc), max_height_diff)
        ) or (
            _blocked(grid, (row, col - dc), max_height_diff)
            and _is_open(grid, (row + dr, col - dc), max_height_diff)
        )
    # 竖直移动的检测
    if dr != 0:
        return (
            _blocked(grid, (row, col + 1), max_height_diff)
            and _is_open(grid, (row + dr, col + 1), max_height_diff)
        ) or (
            _blocked(grid, (row, col - 1), max_height_diff)
            and _is_open(grid, (row + dr, col - 1), max_height_diff)
        )
    # 水平移动的检测
    if dc != 0:
        return (
            _blocked(grid, (row + 1, col), max_height_diff)
            and _is_open(grid, (row + 1, col + dc), max_height_diff)
        ) or (
            _blocked(grid, (row - 1, col), max_height_diff)
            and _is_open(grid, (row - 1, col + dc), max_height_diff)
        )

    return False

# 跳过中间节点，直到遇到障碍物，终点或者跳点
def _jump(
    grid: BattlefieldMap,
    node: Coord, # 跳跃的起始节点
    direction: Coord, # 跳跃的方向
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
        if _has_forced_neighbor(grid, nxt, direction, max_height_diff): # 找到强迫邻居了
            return nxt
        if dr != 0 and dc != 0: # 正在斜向探测，需要对垂直方向和水平方向都进行探测
            if _jump(grid, nxt, (dr, 0), goal, max_height_diff) is not None: # 垂直的递归调用
                return nxt
            if _jump(grid, nxt, (0, dc), goal, max_height_diff) is not None: # 水平的递归调用
                return nxt
        current = nxt

# 对8个方向进行剪枝，减少不必要的考察
def _pruned_directions(
    grid: BattlefieldMap,
    node: Coord,
    parent: Optional[Coord],
    max_height_diff: int,
    allow_diagonal: bool,
) -> List[Coord]:
    if parent is None: # 起点没有前驱方向，必须全部探测
        return DIRS_8 if allow_diagonal else DIRS_4

    dr = _sign(node[0] - parent[0])
    dc = _sign(node[1] - parent[1])
    directions: List[Coord] = []

    def add(direction: Coord) -> None:
        if direction == (0, 0):
            return
        if not allow_diagonal and direction[0] != 0 and direction[1] != 0: # 选择过滤斜角方向
            return
        if direction not in directions and _can_step(grid, node, direction, max_height_diff): # 该方向还没有添加过，通过_can_step验证这个方向迈一步是可行，不撞墙的
            directions.append(direction)

    row, col = node
    if dr != 0 and dc != 0:
        add((dr, dc))
        add((dr, 0))
        add((0, dc))
        if _blocked(grid, (row - dr, col), max_height_diff):
            add((-dr, dc)) # 如果当前节点垂直后方 (row - dr, col) 被阻挡了，说明这里有个坎。为了能绕过这个坎，必须把“倒转垂直向，继续水平向” (-dr, dc) 的分叉方向加进来。
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

# jump_path是一系列跳点，这个函数会把跳过的中间格子给补齐
def _expand_jump_path(grid: BattlefieldMap, jump_path: List[Coord], max_height_diff: int) -> List[Coord]:
    if len(jump_path) <= 1:
        return jump_path[:]

    full_path = [jump_path[0]]
    for start, end in zip(jump_path, jump_path[1:]): # 交错对齐，相邻的点之间做补全
        dr = _sign(end[0] - start[0])
        dc = _sign(end[1] - start[1])
        row_delta = abs(end[0] - start[0])
        col_delta = abs(end[1] - start[1])

        # 合规性检查，必须是竖直水平和斜角
        if not (row_delta == col_delta or row_delta == 0 or col_delta == 0):
            return []

        current = start
        for _ in range(max(row_delta, col_delta)):
            nxt = (current[0] + dr, current[1] + dc)
            # 理论上规划出的路径应该是通畅的
            if not grid.is_passable(current, nxt, max_height_diff):
                return []
            full_path.append(nxt)
            current = nxt

    return full_path

# 重新对选出来的路径逐格判断可通行性
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
    """借用了 A* 算法的框架，在扩展邻居时不是一格一格移动，而是通过裁剪方向后进行“跳跃”探测（_jump），直接寻找下一个关键跳点。"""
    if start == goal:
        return [start]
    # 对起点终点进行合法检查
    if not _is_open(grid, start, max_height_diff) or not _is_open(grid, goal, max_height_diff):
        return []

    open_heap: List[Tuple[float, int, Coord]] = []
    counter = 0
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start))
    came_from: Dict[Coord, Optional[Coord]] = {start: None} # 父节点字典
    g_score: Dict[Coord, float] = {start: 0.0} # 从起点走到各个节点的实际消耗代价，初始化起点的消耗为 0.0
    closed: Set[Coord] = set() # 不要再访问的点

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

            # 当前点的潜在实际代价
            tentative_g = g_score[current] + heuristic(current, jump_point)
            if tentative_g < g_score.get(jump_point, float("inf")):
                came_from[jump_point] = current
                g_score[jump_point] = tentative_g
                # counter维护的是堆是第几次push操作把这个点push进来
                counter += 1
                heapq.heappush(open_heap, (tentative_g + heuristic(jump_point, goal), counter, jump_point))

    return []


# 输入完整path，保留起点，终点和跳点
def compress_jump_points(path: List[Coord]) -> List[Coord]:
    if len(path) <= 2:
        return path[:]

    jumps = [path[0]]
    prev_dr = path[1][0] - path[0][0]
    prev_dc = path[1][1] - path[0][1]
    for i in range(2, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        # 检查这一次迈步方向和上一次迈步方向
        if (dr, dc) != (prev_dr, prev_dc):
            jumps.append(path[i - 1])
        prev_dr, prev_dc = dr, dc
    jumps.append(path[-1])
    return jumps


# 将一条过长、格子过密的完整路径（path），按照指定的距离间隔（interval）抽取出一系列“宏观航路点”。
def interpolate_waypoints(
    path: List[Coord],
    interval: int = 22,
    max_chebyshev: Optional[int] = None, # 有两种插值策略，根据是否选择传入此参数-网格步数
) -> List[Coord]:
    if not path:
        return []
    if len(path) == 1:
        return path[:]

    if max_chebyshev is not None:
        step_limit = max(1, min(interval, max_chebyshev))
        waypoints = [path[0]]
        # 距离上一个录入的锚点走了多少步
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
    # 浮点数累加器
    acc = 0.0
    prev = path[0]
    for cur in path[1:]:
        # 根据距离来计算
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
