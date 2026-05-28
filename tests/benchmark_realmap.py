"""
实战场景 Benchmark

对比 episode_6000 + 全局算法 vs BFS vs A*
曼哈顿距离 ~70 格，敌人覆盖在路径上

指标：速度、可见性、步数、路径效率

用法:
  python tests/benchmark_realmap.py
  python tests/benchmark_realmap.py --cases 50 --enemy-range 60
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battlefield_rl.env import ACTIONS_8, EnemySpec
from battlefield_rl.map import BattlefieldMap, load_txt_map
from battlefield_rl.env.tactical_env import angle_deg, angle_diff, bresenham_line
from battlefield_rl.config import EnvConfig
from battlefield_rl.planner import plan_global_path
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
#  基线寻路算法
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


# ═══════════════════════════════════════════════════════════════
#  敌人放置
# ═══════════════════════════════════════════════════════════════

def angle_from_to(src: Coord, dst: Coord) -> float:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    return (math.degrees(math.atan2(-dr, dc)) + 360.0) % 360.0


def place_enemy_on_path(
    grid: BattlefieldMap,
    path: list[Coord],
    enemy_range: int,
    enemy_fov: int,
    rng: random.Random,
) -> EnemySpec | None:
    """在路径中段放置敌人，使其视野覆盖路径"""
    if len(path) < 10:
        return None

    lo = int(len(path) * 0.3)
    hi = int(len(path) * 0.7)
    ref_idx = rng.randint(lo, hi)
    ref = path[ref_idx]

    look_ahead = min(ref_idx + 5, len(path) - 1)
    facing = angle_from_to(ref, path[look_ahead])

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
            dist = math.hypot(dr, dc)
            if dist <= enemy_range * 0.7:
                candidates.append((r, c))

    if not candidates:
        return None

    pos = rng.choice(candidates)
    jitter = rng.uniform(-25, 25)

    return EnemySpec(
        row=pos[0], col=pos[1],
        facing_deg=(facing + jitter) % 360.0,
        fov_deg=enemy_fov,
        max_range=enemy_range,
    )


# ═══════════════════════════════════════════════════════════════
#  采样
# ═══════════════════════════════════════════════════════════════

def sample_cases(
    rng: random.Random,
    grid: BattlefieldMap,
    cases: int,
    target_dist: int = 70,
    tolerance: int = 15,
) -> list[tuple[Coord, Coord]]:
    """采样曼哈顿距离约 target_dist 的 start-goal 对"""
    cells = passable_cells(grid)
    sampled = []
    attempts = 0
    while len(sampled) < cases and attempts < cases * 5000:
        attempts += 1
        start = rng.choice(cells)
        goal = rng.choice(cells)
        if start == goal:
            continue
        dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if abs(dist - target_dist) <= tolerance:
            # 验证可达性
            path = bfs_path(grid, start, goal)
            if path and len(path) > 1:
                sampled.append((start, goal))
    if len(sampled) < cases:
        raise RuntimeError(f"只采样到 {len(sampled)} 个, 需要 {cases} 个")
    return sampled


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def run_benchmark(
    map_path: str,
    model_path: str,
    pairs: list[tuple[Coord, Coord]],
    n_enemies: int,
    enemy_range: int,
    enemy_fov: int,
    seed: int,
) -> dict[str, Any]:
    grid = load_txt_map(map_path)
    rng = random.Random(seed)

    results: dict[str, list[dict[str, Any]]] = {
        "bfs": [], "astar": [], "model": []
    }

    for i, (start, goal) in enumerate(pairs):
        manhattan_dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])

        # 获取 A* 路径用于放置敌人
        astar_p = astar_path(grid, start, goal)
        if not astar_p:
            continue

        # 在路径上放置敌人
        enemies = []
        for _ in range(n_enemies):
            e = place_enemy_on_path(grid, astar_p, enemy_range, enemy_fov, rng)
            if e:
                enemies.append(e)

        # BFS
        t0 = time.perf_counter()
        bfs_p = bfs_path(grid, start, goal)
        bfs_time = time.perf_counter() - t0
        bfs_vis = path_visible_stats(grid, bfs_p, enemies)
        results["bfs"].append({
            "case": i,
            "start": list(start),
            "goal": list(goal),
            "manhattan_dist": manhattan_dist,
            "success": len(bfs_p) > 0 and bfs_p[-1] == goal,
            "steps": max(0, len(bfs_p) - 1),
            "runtime_ms": bfs_time * 1000,
            "enemies": len(enemies),
            **bfs_vis,
        })

        # A*
        t0 = time.perf_counter()
        astar_p = astar_path(grid, start, goal)
        astar_time = time.perf_counter() - t0
        astar_vis = path_visible_stats(grid, astar_p, enemies)
        results["astar"].append({
            "case": i,
            "success": len(astar_p) > 0 and astar_p[-1] == goal,
            "steps": max(0, len(astar_p) - 1),
            "runtime_ms": astar_time * 1000,
            **astar_vis,
        })

        # Model (plan_and_execute)
        t0 = time.perf_counter()
        r = plan_and_execute(
            map_path=map_path, start=start, goal=goal,
            enemies=enemies, model_path=model_path,
            use_fallback=True,
        )
        model_time = time.perf_counter() - t0

        # 计算模型路径的可见性
        full_path = []
        if "path" in r:
            full_path = [tuple(p) for p in r["path"]]
        model_vis = path_visible_stats(grid, full_path, enemies) if full_path else {
            "visible_steps": 0, "total_steps": r["total_steps"], "visible_ratio": 0
        }

        results["model"].append({
            "case": i,
            "success": r["success"],
            "local_success": r.get("local_executor_success", False),
            "total_steps": r["total_steps"],
            "path_efficiency": r.get("path_efficiency", 0),
            "geometric_fallback_count": r.get("geometric_fallback_count", 0),
            "geometric_fallback_steps": r.get("geometric_fallback_steps", 0),
            "failed_reason": r.get("failed_reason", ""),
            "runtime_ms": model_time * 1000,
            **model_vis,
        })

        # 打印
        bfs_icon = "OK" if results["bfs"][-1]["success"] else "XX"
        astar_icon = "OK" if results["astar"][-1]["success"] else "XX"
        model_icon = "OK" if results["model"][-1]["success"] else "XX"

        print(f"  case {i:2d}: {start}->{goal} dist={manhattan_dist} enemies={len(enemies)}")
        print(f"           BFS={bfs_icon}({results['bfs'][-1]['steps']}步 {bfs_vis['visible_ratio']:.2f}vis {bfs_time*1000:.1f}ms)")
        print(f"           A*={astar_icon}({results['astar'][-1]['steps']}步 {astar_vis['visible_ratio']:.2f}vis {astar_time*1000:.1f}ms)")
        print(f"           RL={model_icon}({results['model'][-1]['total_steps']}步 {model_vis['visible_ratio']:.2f}vis {model_time*1000:.0f}ms)")

    return {
        "map": map_path,
        "model": model_path,
        "cases": len(results["bfs"]),
        "n_enemies": n_enemies,
        "enemy_range": enemy_range,
        "enemy_fov": enemy_fov,
        "results": results,
    }


def print_summary(data: dict[str, Any]) -> None:
    results = data["results"]
    n = data["cases"]

    print(f"\n{'='*80}")
    print(f"  BENCHMARK RESULTS  ({n} cases, {data['n_enemies']} enemies, "
          f"range={data['enemy_range']}, fov={data['enemy_fov']})")
    print(f"  Model: {data['model']}")
    print(f"{'='*80}")

    keys = ["bfs", "astar", "model"]
    names = ["BFS", "A*", "RL+Global"]

    header = f"  {'Metric':<30s}" + "".join(f"{sn:>14s}" for sn in names)
    print(header)
    print(f"  {'-'*80}")

    for metric, key, fmt in [
        ("Success rate", "success", lambda v: f"{sum(v)/len(v):>13.1%}"),
        ("Avg steps", "steps", lambda v: f"{sum(v)/len(v):>13.1f}"),
        ("Avg visible ratio", "visible_ratio", lambda v: f"{sum(v)/len(v):>13.3f}"),
        ("Avg visible steps", "visible_steps", lambda v: f"{sum(v)/len(v):>13.1f}"),
        ("Avg runtime (ms)", "runtime_ms", lambda v: f"{sum(v)/len(v):>13.1f}"),
    ]:
        vals = []
        for k in keys:
            # 兼容 steps 和 total_steps
            if key == "steps" and k == "model":
                vals.append(fmt([r.get("total_steps", 0) for r in results[k]]))
            else:
                vals.append(fmt([r.get(key, 0) for r in results[k]]))
        print(f"  {metric:<30s}" + "".join(f"{v:>14s}" for v in vals))

    # 路径效率
    model_pe = [r.get("path_efficiency", 0) for r in results["model"] if r["success"]]
    if model_pe:
        print(f"  {'Model path efficiency':<30s} {sum(model_pe)/len(model_pe):>13.3f}")

    # 回退统计
    fallback_cases = [r for r in results["model"] if r.get("geometric_fallback_count", 0) > 0]
    print(f"  {'Model used fallback':<30s} {len(fallback_cases):>13d} / {n}")

    # 可见性分布
    print(f"\n  Visibility distribution:")
    for k, sn in zip(keys, names):
        vis_vals = [r["visible_ratio"] for r in results[k]]
        zero = sum(1 for v in vis_vals if v < 0.01)
        low = sum(1 for v in vis_vals if 0.01 <= v < 0.20)
        mid = sum(1 for v in vis_vals if 0.20 <= v < 0.50)
        high = sum(1 for v in vis_vals if v >= 0.50)
        print(f"    {sn:>10s}: zero={zero:3d}  low={low:3d}  mid={mid:3d}  high={high:3d}")

    # 有敌人 case 的详细对比
    enemy_cases = [i for i in range(n) if results["bfs"][i]["visible_ratio"] > 0.01]
    if enemy_cases:
        print(f"\n  Cases with visible enemies: {len(enemy_cases)}")
        for metric, key, fmt in [
            ("Success rate", "success", lambda v: f"{sum(v)/len(v):>13.1%}"),
            ("Avg steps", "steps", lambda v: f"{sum(v)/len(v):>13.1f}"),
            ("Avg visible ratio", "visible_ratio", lambda v: f"{sum(v)/len(v):>13.3f}"),
        ]:
            vals = []
            for k in keys:
                if key == "steps" and k == "model":
                    vals.append(fmt([results[k][i].get("total_steps", 0) for i in enemy_cases]))
                else:
                    vals.append(fmt([results[k][i].get(key, 0) for i in enemy_cases]))
            print(f"    {metric:<30s}" + "".join(f"{v:>14s}" for v in vals))

    # 逐 case 可见性对比
    print(f"\n  Per-case visibility comparison (model vs BFS, top differences):")
    diffs = []
    for i in range(n):
        bfs_vis = results["bfs"][i]["visible_ratio"]
        model_vis = results["model"][i].get("visible_ratio", 0)
        if bfs_vis > 0.05:
            diff = bfs_vis - model_vis
            diffs.append((i, diff, bfs_vis, model_vis, results["model"][i].get("total_steps", 0)))
    diffs.sort(key=lambda x: x[1], reverse=True)
    for i, diff, bfs_v, model_v, model_s in diffs[:8]:
        icon = "OK" if results["model"][i]["success"] else "XX"
        print(f"    case {i:2d}: BFS={bfs_v:.3f} Model={model_v:.3f} "
              f"delta={diff:+.3f} {icon}({model_s}步)")

    print(f"\n{'='*80}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark on real map")
    p.add_argument("--map", default="MyPath_Data417.txt")
    p.add_argument("--model", default="episode_6000.pt")
    p.add_argument("--cases", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--n-enemies", type=int, default=2, help="enemies per case")
    p.add_argument("--enemy-range", type=int, default=60, help="enemy vision range")
    p.add_argument("--enemy-fov", type=int, default=120, help="enemy field of view degrees")
    p.add_argument("--target-dist", type=int, default=70, help="target manhattan distance")
    p.add_argument("--output-dir", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    grid = load_txt_map(args.map)
    pairs = sample_cases(rng, grid, args.cases, args.target_dist)

    print(f"Map: {args.map}  Cases: {args.cases}  Target dist: {args.target_dist}")
    print(f"Model: {args.model}")
    print(f"Enemies: {args.n_enemies} per case, range={args.enemy_range}, fov={args.enemy_fov}")

    data = run_benchmark(
        map_path=args.map, model_path=args.model,
        pairs=pairs, n_enemies=args.n_enemies,
        enemy_range=args.enemy_range, enemy_fov=args.enemy_fov,
        seed=args.seed,
    )
    print_summary(data)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = out_dir / f"benchmark_realmap_{ts}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
