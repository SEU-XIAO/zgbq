"""
实战场景可见性回避测试

在真实大地图上，敌人视野覆盖在 A* 最优路径上，
模拟"敌人较远但能看到必经之路"的实战情况。

对比 episode_6000 / episode_8000 / BFS / A* 的表现，
重点看模型是否会选择绕路避开敌人视野。

用法:
  python tests/compare_visibility_realmap.py
  python tests/compare_visibility_realmap.py --model episode_6000.pt --cases 50
  python tests/compare_visibility_realmap.py --n-enemies 3 --enemy-range 80
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battlefield_rl.env import ACTIONS_8, EnemySpec
from battlefield_rl.map import BattlefieldMap, load_txt_map
from battlefield_rl.env.tactical_env import angle_deg, angle_diff, bresenham_line
from battlefield_rl.config import EnvConfig
from battlefield_rl.planner import plan_global_path
from interfaces.config import DEFAULT_MAP_PATH
from scripts.eval_hierarchical_executor import plan_and_execute

Coord = tuple[int, int]


# ═══════════════════════════════════════════════════════════════
#  可见性计算
# ═══════════════════════════════════════════════════════════════

def _point_visible(grid: BattlefieldMap, point: Coord, enemy: EnemySpec) -> bool:
    src = (enemy.row, enemy.col)
    d = math.hypot(point[0] - src[0], point[1] - src[1])
    if d > enemy.max_range:
        return False
    if angle_diff(angle_deg(src, point), enemy.facing_deg) > enemy.fov_deg * 0.5:
        return False
    line = bresenham_line(src[0], src[1], point[0], point[1])
    for r, c in line[1:-1]:
        if grid.is_static_blocked(r, c):
            return False
    return True


def path_visible_stats(
    grid: BattlefieldMap,
    path: Sequence[Coord],
    enemies: Sequence[EnemySpec],
) -> dict[str, Any]:
    if not enemies or len(path) == 0:
        return {"visible_steps": 0, "total_steps": len(path), "visible_ratio": 0.0}

    visible_count = sum(
        1 for p in path
        if any(_point_visible(grid, p, e) for e in enemies)
    )
    return {
        "visible_steps": visible_count,
        "total_steps": len(path),
        "visible_ratio": visible_count / max(1, len(path)),
    }


# ═══════════════════════════════════════════════════════════════
#  路径规划
# ═══════════════════════════════════════════════════════════════

def passable_cells(grid: BattlefieldMap) -> list[Coord]:
    rows, cols = grid.shape
    return [(r, c) for r in range(rows) for c in range(cols)
            if not grid.is_static_blocked(r, c)]


def bfs_path(grid: BattlefieldMap, start: Coord, goal: Coord) -> list[Coord]:
    if start == goal:
        return [start]
    queue: deque[Coord] = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}
    while queue:
        cur = queue.popleft()
        for dr, dc in ACTIONS_8:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in parent or not grid.is_passable(cur, nxt, 0):
                continue
            parent[nxt] = cur
            if nxt == goal:
                queue.clear()
                break
            queue.append(nxt)
    if goal not in parent:
        return []
    path: list[Coord] = []
    node: Coord | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def astar_path(grid: BattlefieldMap, start: Coord, goal: Coord) -> list[Coord]:
    def h(pos: Coord) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    open_set: list[tuple[int, Coord]] = [(h(start), start)]
    parent: dict[Coord, Coord | None] = {start: None}
    g_score: dict[Coord, int] = {start: 0}

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            break
        for dr, dc in ACTIONS_8:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not grid.is_passable(cur, nxt, 0):
                continue
            new_g = g_score[cur] + 1
            if nxt not in g_score or new_g < g_score[nxt]:
                g_score[nxt] = new_g
                parent[nxt] = cur
                heapq.heappush(open_set, (new_g + h(nxt), nxt))

    if goal not in parent:
        return []
    path: list[Coord] = []
    node: Coord | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def get_astar_path(grid: BattlefieldMap, start: Coord, goal: Coord) -> list[Coord]:
    result = plan_global_path(grid, start, goal, max_height_diff=1, waypoint_interval=18)
    return result.path if result.path else []


# ═══════════════════════════════════════════════════════════════
#  敌人放置策略
# ═══════════════════════════════════════════════════════════════

def angle_from_to(src: Coord, dst: Coord) -> float:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    return (math.degrees(math.atan2(-dr, dc)) + 360.0) % 360.0


def place_enemy_covering_path(
    grid: BattlefieldMap,
    path: list[Coord],
    enemy_range: int,
    enemy_fov: int,
    rng: random.Random,
) -> EnemySpec | None:
    """
    在路径附近放置敌人，使其视野覆盖路径上的多个点。
    模拟"敌人较远但能看到必经之路"的情况。
    """
    if len(path) < 10:
        return None

    # 取路径 30-70% 位置
    lo = int(len(path) * 0.3)
    hi = int(len(path) * 0.7)
    ref_idx = rng.randint(lo, hi)
    ref = path[ref_idx]

    # 敌人朝向路径方向
    look_ahead = min(ref_idx + 5, len(path) - 1)
    facing = angle_from_to(ref, path[look_ahead])

    # 在路径附近找合法位置（不在路径上，但在视野范围内）
    path_set = set(path)
    candidates = []
    for dr in range(-8, 9):
        for dc in range(-8, 9):
            if dr == 0 and dc == 0:
                continue
            r, c = ref[0] + dr, ref[1] + dc
            if not grid.in_bounds(r, c) or grid.is_static_blocked(r, c):
                continue
            if (r, c) in path_set:
                continue
            # 检查到路径的距离
            dist = math.hypot(dr, dc)
            if dist <= enemy_range * 0.7:  # 确保能覆盖路径
                candidates.append((r, c, dist))

    if not candidates:
        # 扩大搜索范围
        for dr in range(-15, 16):
            for dc in range(-15, 16):
                if dr == 0 and dc == 0:
                    continue
                r, c = ref[0] + dr, ref[1] + dc
                if not grid.in_bounds(r, c) or grid.is_static_blocked(r, c):
                    continue
                dist = math.hypot(dr, dc)
                if dist <= enemy_range * 0.8:
                    candidates.append((r, c, dist))

    if not candidates:
        return None

    # 优先选择距离适中的位置（不要太近也不要太远）
    candidates.sort(key=lambda x: abs(x[2] - enemy_range * 0.5))
    pos = rng.choice(candidates[:min(10, len(candidates))])
    jitter = rng.uniform(-30, 30)

    return EnemySpec(
        row=pos[0], col=pos[1],
        facing_deg=(facing + jitter) % 360.0,
        fov_deg=enemy_fov,
        max_range=enemy_range,
    )


def place_enemies_covering_path(
    grid: BattlefieldMap,
    path: list[Coord],
    n_enemies: int,
    enemy_range: int,
    enemy_fov: int,
    rng: random.Random,
) -> list[EnemySpec]:
    """沿路径放置多个敌人，覆盖路径的不同区段"""
    enemies = []
    if len(path) < 10:
        return enemies

    # 均匀分布在路径的不同区段
    seg_len = len(path) // (n_enemies + 1)
    for i in range(n_enemies):
        center_idx = seg_len * (i + 1)
        lo = max(5, center_idx - seg_len // 2)
        hi = min(len(path) - 5, center_idx + seg_len // 2)
        if lo >= hi:
            continue

        # 临时修改 rng 位置
        ref_idx = rng.randint(lo, hi)
        ref = path[ref_idx]
        look_ahead = min(ref_idx + 5, len(path) - 1)
        facing = angle_from_to(ref, path[look_ahead])

        path_set = set(path)
        candidates = []
        for dr in range(-10, 11):
            for dc in range(-10, 11):
                if dr == 0 and dc == 0:
                    continue
                r, c = ref[0] + dr, ref[1] + dc
                if not grid.in_bounds(r, c) or grid.is_static_blocked(r, c):
                    continue
                if (r, c) in path_set:
                    continue
                dist = math.hypot(dr, dc)
                if dist <= enemy_range * 0.7:
                    candidates.append((r, c, dist))

        if not candidates:
            continue

        # 选择距离适中的位置
        candidates.sort(key=lambda x: abs(x[2] - enemy_range * 0.5))
        pos = rng.choice(candidates[:min(5, len(candidates))])
        jitter = rng.uniform(-30, 30)
        enemies.append(EnemySpec(
            row=pos[0], col=pos[1],
            facing_deg=(facing + jitter) % 360.0,
            fov_deg=enemy_fov,
            max_range=enemy_range,
        ))

    return enemies


# ═══════════════════════════════════════════════════════════════
#  采样与执行
# ═══════════════════════════════════════════════════════════════

def sample_cases(
    rng: random.Random,
    grid: BattlefieldMap,
    cases: int,
    min_steps: int = 40,
    max_steps: int = 120,
) -> list[tuple[Coord, Coord]]:
    cells = passable_cells(grid)
    sampled = []
    attempts = 0
    while len(sampled) < cases and attempts < cases * 3000:
        attempts += 1
        start = rng.choice(cells)
        goal = rng.choice(cells)
        if start == goal:
            continue
        path = bfs_path(grid, start, goal)
        steps = len(path) - 1
        if path and min_steps <= steps <= max_steps:
            sampled.append((start, goal))
    if len(sampled) < cases:
        raise RuntimeError(f"只采样到 {len(sampled)} 个, 需要 {cases} 个")
    return sampled


def run_comparison(
    map_path: str,
    model_paths: list[str],
    pairs: list[tuple[Coord, Coord]],
    n_enemies: int,
    enemy_range: int,
    enemy_fov: int,
    use_fallback: bool,
    seed: int,
) -> dict[str, Any]:
    grid = load_txt_map(map_path)
    rng = random.Random(seed)
    results: dict[str, list[dict[str, Any]]] = {m: [] for m in model_paths}
    results["bfs"] = []
    results["astar"] = []

    for i, (start, goal) in enumerate(pairs):
        # 获取 A* 路径用于放置敌人
        astar_p = get_astar_path(grid, start, goal)
        if not astar_p:
            astar_p = bfs_path(grid, start, goal)
        astar_steps = max(1, len(astar_p) - 1)

        # 在 A* 路径上放置敌人
        enemies = place_enemies_covering_path(
            grid, astar_p, n_enemies, enemy_range, enemy_fov, rng
        )

        # BFS baseline
        bfs_p = bfs_path(grid, start, goal)
        bfs_vis = path_visible_stats(grid, bfs_p, enemies)
        results["bfs"].append({
            "case": i,
            "success": len(bfs_p) > 0 and bfs_p[-1] == goal,
            "steps": max(0, len(bfs_p) - 1),
            **bfs_vis,
        })

        # A* baseline
        astar_vis = path_visible_stats(grid, astar_p, enemies)
        results["astar"].append({
            "case": i,
            "success": len(astar_p) > 0 and astar_p[-1] == goal,
            "steps": astar_steps,
            **astar_vis,
        })

        # 模型执行
        for model_path in model_paths:
            t0 = time.perf_counter()
            r = plan_and_execute(
                map_path=map_path, start=start, goal=goal,
                enemies=enemies, model_path=model_path,
                use_fallback=use_fallback,
            )
            elapsed = time.perf_counter() - t0

            # 计算完整路径的可见性
            full_path = []
            if "path" in r:
                full_path = [tuple(p) for p in r["path"]]
            model_vis = path_visible_stats(grid, full_path, enemies) if full_path else {"visible_steps": 0, "total_steps": r["total_steps"], "visible_ratio": 0}

            results[model_path].append({
                "case": i,
                "success": r["success"],
                "local_success": r.get("local_executor_success", False),
                "total_steps": r["total_steps"],
                "path_efficiency": r.get("path_efficiency", 0),
                "geometric_fallback_count": r.get("geometric_fallback_count", 0),
                "failed_reason": r.get("failed_reason", ""),
                "runtime_sec": elapsed,
                **model_vis,
            })

        # 逐 case 打印
        parts = []
        for key in ["bfs", "astar"] + model_paths:
            rr = results[key][-1]
            name = key.split("/")[-1].split(".")[0] if "/" in key else key
            icon = "OK" if rr["success"] else "XX"
            vis = rr["visible_ratio"]
            steps = rr["steps"] if "steps" in rr else rr.get("total_steps", 0)
            parts.append(f"{name}={icon}({steps}步 vis={vis:.2f})")

        print(f"  case {i:3d}: {start}->{goal} astar={astar_steps} enemies={len(enemies)}")
        print(f"           {'  '.join(parts)}")

    return {
        "map": map_path,
        "models": model_paths,
        "cases": len(pairs),
        "n_enemies": n_enemies,
        "enemy_range": enemy_range,
        "enemy_fov": enemy_fov,
        "use_fallback": use_fallback,
        "results": results,
    }


def print_summary(data: dict[str, Any]) -> None:
    results = data["results"]
    n = data["cases"]

    print(f"\n{'='*80}")
    print(f"  REAL MAP VISIBILITY COMPARISON  ({n} cases, {data['n_enemies']} enemies, "
          f"range={data['enemy_range']}, fov={data['enemy_fov']}, "
          f"fallback={'ON' if data['use_fallback'] else 'OFF'})")
    print(f"{'='*80}")

    keys = ["bfs", "astar"] + data["models"]
    short_names = []
    for k in keys:
        if "/" in k:
            short_names.append(k.split("/")[-1].split(".")[0])
        else:
            short_names.append(k)

    header = f"  {'Metric':<30s}" + "".join(f"{sn:>12s}" for sn in short_names)
    print(header)
    print(f"  {'-'*80}")

    for metric, key, fmt in [
        ("Success rate", "success", lambda v: f"{sum(v)/len(v):>11.1%}"),
        ("Avg steps", "total_steps", lambda v: f"{sum(v)/len(v):>11.1f}"),
        ("Avg visible ratio", "visible_ratio", lambda v: f"{sum(v)/len(v):>11.3f}"),
        ("Avg visible steps", "visible_steps", lambda v: f"{sum(v)/len(v):>11.1f}"),
    ]:
        vals = []
        for k in keys:
            # 兼容 steps 和 total_steps
            if key == "total_steps" and "steps" in results[k][0] and "total_steps" not in results[k][0]:
                vals.append(fmt([r.get("steps", 0) for r in results[k]]))
            else:
                vals.append(fmt([r.get(key, 0) for r in results[k]]))
        print(f"  {metric:<30s}" + "".join(f"{v:>12s}" for v in vals))

    # 只看有敌人的情况
    print(f"\n  Cases with visible enemies (BFS vis > 0.01):")
    enemy_cases = [i for i in range(n) if results["bfs"][i]["visible_ratio"] > 0.01]
    if enemy_cases:
        print(f"    Found {len(enemy_cases)} cases")
        for metric, key, fmt in [
            ("Success rate", "success", lambda v: f"{sum(v)/len(v):>11.1%}"),
            ("Avg steps", "total_steps", lambda v: f"{sum(v)/len(v):>11.1f}"),
            ("Avg visible ratio", "visible_ratio", lambda v: f"{sum(v)/len(v):>11.3f}"),
        ]:
            vals = []
            for k in keys:
                if key == "total_steps" and "steps" in results[k][0] and "total_steps" not in results[k][0]:
                    vals.append(fmt([results[k][i].get("steps", 0) for i in enemy_cases]))
                else:
                    vals.append(fmt([results[k][i].get(key, 0) for i in enemy_cases]))
            print(f"    {metric:<30s}" + "".join(f"{v:>12s}" for v in vals))
    else:
        print(f"    No cases with visible enemies")

    # 逐 case 可见性对比
    print(f"\n  Top cases where model differs from BFS/A* (visibility):")
    for mp in data["models"]:
        mp_short = mp.split("/")[-1].split(".")[0]
        diffs = []
        for i in range(n):
            bfs_vis = results["bfs"][i]["visible_ratio"]
            mp_vis = results[mp][i].get("visible_ratio", 0)
            if bfs_vis > 0.05:  # 只看有可见敌人的情况
                diff = bfs_vis - mp_vis
                diffs.append((i, diff, bfs_vis, mp_vis, results[mp][i].get("total_steps", 0)))
        diffs.sort(key=lambda x: x[1], reverse=True)
        print(f"\n    {mp_short} vs BFS (top 5 avoidance cases):")
        for i, diff, bfs_v, mp_v, mp_s in diffs[:5]:
            icon = "OK" if results[mp][i]["success"] else "XX"
            print(f"      case {i}: BFS={bfs_v:.3f} {mp_short}={mp_v:.3f} "
                  f"delta={diff:+.3f} {icon}({mp_s}步)")

    print(f"\n{'='*80}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare visibility avoidance on real map")
    p.add_argument("--map", default=DEFAULT_MAP_PATH)
    p.add_argument("--model", action="append", default=[], help="model path (repeatable)")
    p.add_argument("--cases", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--n-enemies", type=int, default=3, help="enemies per case")
    p.add_argument("--enemy-range", type=int, default=80, help="enemy vision range")
    p.add_argument("--enemy-fov", type=int, default=120, help="enemy field of view degrees")
    p.add_argument("--no-fallback", action="store_true")
    p.add_argument("--output-dir", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    model_paths = args.model or ["episode_6000.pt", "episode_8000.pt"]
    rng = random.Random(args.seed)
    grid = load_txt_map(args.map)
    pairs = sample_cases(rng, grid, args.cases)

    print(f"Map: {args.map}  Cases: {args.cases}")
    print(f"Models: {model_paths}")
    print(f"Enemies: {args.n_enemies} per case, range={args.enemy_range}, fov={args.enemy_fov}")
    print(f"Fallback: {'OFF' if args.no_fallback else 'ON'}")

    data = run_comparison(
        map_path=args.map, model_paths=model_paths,
        pairs=pairs, n_enemies=args.n_enemies,
        enemy_range=args.enemy_range, enemy_fov=args.enemy_fov,
        use_fallback=not args.no_fallback, seed=args.seed,
    )
    print_summary(data)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = out_dir / f"visibility_realmap_{ts}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
