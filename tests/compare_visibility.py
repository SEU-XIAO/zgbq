"""
可见性回避对比测试

敌人视野直接覆盖在 A* 最优路径上，迫使模型选择绕路。
对比不同模型的可见比例、路径效率、绕路程度。

用法:
  python tests/compare_visibility.py
  python tests/compare_visibility.py --model episode_6000.pt --model episode_8000.pt
  python tests/compare_visibility.py --cases 50 --enemy-range 60 --enemy-fov 120
"""

from __future__ import annotations

import argparse
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
from battlefield_rl.planner import plan_global_path
from battlefield_rl.config import EnvConfig, PlannerConfig
from interfaces.config import DEFAULT_MAP_PATH
from scripts.eval_hierarchical_executor import plan_and_execute

Coord = tuple[int, int]


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


def get_astar_path(grid: BattlefieldMap, start: Coord, goal: Coord) -> list[Coord]:
    """获取 A* 全局路径"""
    result = plan_global_path(grid, start, goal, max_height_diff=1, waypoint_interval=18)
    return result.path if result.path else []


def angle_from_to(src: Coord, dst: Coord) -> float:
    """从 src 到 dst 的角度（度）"""
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
    """
    在 A* 路径中段放置敌人，使其视野覆盖路径。
    敌人不直接站在路径上，而是偏移几步。
    """
    if len(path) < 10:
        return None

    # 取路径 30-70% 位置作为参考点
    lo = int(len(path) * 0.3)
    hi = int(len(path) * 0.7)
    ref_idx = rng.randint(lo, hi)
    ref = path[ref_idx]

    # 敌人朝向路径方向
    look_ahead = min(ref_idx + 5, len(path) - 1)
    facing = angle_from_to(ref, path[look_ahead])

    # 在参考点附近找一个不在路径上的合法位置
    path_set = set(path)
    candidates = []
    for dr in range(-4, 5):
        for dc in range(-4, 5):
            if dr == 0 and dc == 0:
                continue
            r, c = ref[0] + dr, ref[1] + dc
            if not grid.in_bounds(r, c):
                continue
            if grid.is_static_blocked(r, c):
                continue
            if (r, c) in path_set:
                continue
            # 检查到路径的距离足够近，能覆盖
            min_dist = min(abs(r - p[0]) + abs(c - p[1]) for p in path[lo:hi+1])
            if min_dist <= enemy_range // 2:
                candidates.append((r, c))

    if not candidates:
        # 退而求其次，直接放在路径附近
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                r, c = ref[0] + dr, ref[1] + dc
                if grid.in_bounds(r, c) and not grid.is_static_blocked(r, c):
                    candidates.append((r, c))

    if not candidates:
        return None

    pos = rng.choice(candidates)
    # 朝向稍微随机化
    jitter = rng.uniform(-20, 20)

    return EnemySpec(
        row=pos[0], col=pos[1],
        facing_deg=(facing + jitter) % 360.0,
        fov_deg=enemy_fov,
        max_range=enemy_range,
    )


def place_enemies_along_path(
    grid: BattlefieldMap,
    path: list[Coord],
    n_enemies: int,
    enemy_range: int,
    enemy_fov: int,
    rng: random.Random,
) -> list[EnemySpec]:
    """沿 A* 路径放置多个敌人"""
    enemies = []
    if len(path) < 10:
        return enemies

    # 均匀分布在路径的不同区段
    seg_len = len(path) // (n_enemies + 1)
    for i in range(n_enemies):
        center_idx = seg_len * (i + 1)
        # 在该区段附近采样
        lo = max(5, center_idx - seg_len // 2)
        hi = min(len(path) - 5, center_idx + seg_len // 2)
        if lo >= hi:
            continue

        # 临时修改 rng 位置以产生不同敌人
        ref_idx = rng.randint(lo, hi)
        ref = path[ref_idx]
        look_ahead = min(ref_idx + 5, len(path) - 1)
        facing = angle_from_to(ref, path[look_ahead])

        path_set = set(path)
        candidates = []
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                if dr == 0 and dc == 0:
                    continue
                r, c = ref[0] + dr, ref[1] + dc
                if not grid.in_bounds(r, c) or grid.is_static_blocked(r, c):
                    continue
                if (r, c) in path_set:
                    continue
                candidates.append((r, c))

        if not candidates:
            continue

        pos = rng.choice(candidates)
        jitter = rng.uniform(-30, 30)
        enemies.append(EnemySpec(
            row=pos[0], col=pos[1],
            facing_deg=(facing + jitter) % 360.0,
            fov_deg=enemy_fov,
            max_range=enemy_range,
        ))

    return enemies


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

    for i, (start, goal) in enumerate(pairs):
        # 获取 A* 路径用于放置敌人和计算基准
        astar = get_astar_path(grid, start, goal)
        astar_steps = max(1, len(astar) - 1)

        # 在 A* 路径上放置敌人
        enemies = place_enemies_along_path(
            grid, astar, n_enemies, enemy_range, enemy_fov, rng
        )

        for model_path in model_paths:
            t0 = time.perf_counter()
            r = plan_and_execute(
                map_path=map_path, start=start, goal=goal,
                enemies=enemies, model_path=model_path,
                use_fallback=use_fallback,
            )
            elapsed = time.perf_counter() - t0

            results[model_path].append({
                "case": i,
                "start": list(start),
                "goal": list(goal),
                "astar_steps": astar_steps,
                "success": r["success"],
                "local_success": r.get("local_executor_success", False),
                "total_steps": r["total_steps"],
                "path_efficiency": r.get("path_efficiency", 0),
                "visible_ratio": r.get("visible_ratio", 0),
                "visible_steps": r.get("visible_steps", 0),
                "repeat_ratio": r.get("repeat_ratio", 0),
                "geometric_fallback_count": r.get("geometric_fallback_count", 0),
                "geometric_fallback_steps": r.get("geometric_fallback_steps", 0),
                "failed_reason": r.get("failed_reason", ""),
                "n_enemies": len(enemies),
                "runtime_sec": elapsed,
            })

        # 逐 case 打印
        status_parts = []
        for mp in model_paths:
            rr = results[mp][-1]
            icon = "OK" if rr["success"] else "XX"
            vis = f"vis={rr['visible_ratio']:.2f}"
            eff = f"eff={rr['path_efficiency']:.2f}"
            status_parts.append(f"{mp.split('/')[-1].split('.')[0]}={icon}({rr['total_steps']}步 {vis} {eff})")

        print(f"  case {i:3d}: {start}->{goal} astar={astar_steps} enemies={len(enemies)}")
        print(f"           {'  '.join(status_parts)}")

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

    print(f"\n{'='*70}")
    print(f"  VISIBILITY COMPARISON  ({n} cases, {data['n_enemies']} enemies, "
          f"range={data['enemy_range']}, fov={data['enemy_fov']}, "
          f"fallback={'ON' if data['use_fallback'] else 'OFF'})")
    print(f"{'='*70}")

    models = list(results.keys())
    short_names = [m.split("/")[-1].split(".")[0] for m in models]
    header = f"  {'Metric':<28s}" + "".join(f"{sn:>14s}" for sn in short_names)
    print(header)
    print(f"  {'-'*70}")

    for metric, key, fmt in [
        ("Success rate", "success", lambda v: f"{sum(v)/len(v):>13.1%}"),
        ("Local-only success", "local_success", lambda v: f"{sum(v)/len(v):>13.1%}"),
        ("Avg total steps", "total_steps", lambda v: f"{sum(v)/len(v):>13.1f}"),
        ("Avg path efficiency", "path_efficiency", lambda v: f"{sum(v)/len(v):>13.3f}"),
        ("Avg visible ratio", "visible_ratio", lambda v: f"{sum(v)/len(v):>13.3f}"),
        ("Avg visible steps", "visible_steps", lambda v: f"{sum(v)/len(v):>13.1f}"),
        ("Avg repeat ratio", "repeat_ratio", lambda v: f"{sum(v)/len(v):>13.3f}"),
        ("Avg fallback count", "geometric_fallback_count", lambda v: f"{sum(v)/len(v):>13.2f}"),
        ("Avg fallback steps", "geometric_fallback_steps", lambda v: f"{sum(v)/len(v):>13.1f}"),
        ("Avg runtime (s)", "runtime_sec", lambda v: f"{sum(v)/len(v):>13.2f}"),
    ]:
        vals = [fmt([r[key] for r in results[m]]) for m in models]
        print(f"  {metric:<28s}" + "".join(f"{v:>14s}" for v in vals))

    # 可见性分布分析
    print(f"\n  Visibility distribution:")
    for mp, sn in zip(models, short_names):
        vis_vals = [r["visible_ratio"] for r in results[mp]]
        zero_vis = sum(1 for v in vis_vals if v < 0.01)
        low_vis = sum(1 for v in vis_vals if 0.01 <= v < 0.15)
        mid_vis = sum(1 for v in vis_vals if 0.15 <= v < 0.35)
        high_vis = sum(1 for v in vis_vals if v >= 0.35)
        print(f"    {sn}: zero={zero_vis} low={low_vis} mid={mid_vis} high={high_vis}")

    # 失败原因
    print(f"\n  Failure reasons:")
    for mp, sn in zip(models, short_names):
        from collections import Counter
        reasons = Counter(r["failed_reason"] for r in results[mp] if not r["success"])
        if reasons:
            print(f"    {sn}: {dict(reasons)}")
        else:
            print(f"    {sn}: (none)")

    # 可见性 vs 路径效率散点分析
    print(f"\n  Visibility vs efficiency (cases with visible_ratio > 0.05):")
    for mp, sn in zip(models, short_names):
        vis_cases = [(r["visible_ratio"], r["path_efficiency"], r["total_steps"])
                     for r in results[mp] if r["visible_ratio"] > 0.05]
        if vis_cases:
            avg_vis = sum(v for v, _, _ in vis_cases) / len(vis_cases)
            avg_eff = sum(e for _, e, _ in vis_cases) / len(vis_cases)
            avg_steps = sum(s for _, _, s in vis_cases) / len(vis_cases)
            print(f"    {sn}: {len(vis_cases)} cases, avg_vis={avg_vis:.3f}, "
                  f"avg_eff={avg_eff:.3f}, avg_steps={avg_steps:.1f}")
        else:
            print(f"    {sn}: no cases with visible_ratio > 0.05")

    # 逐 case 可见性对比（只显示有差异的）
    if len(models) == 2:
        print(f"\n  Per-case visibility comparison (top differences):")
        diffs = []
        for i in range(n):
            v0 = results[models[0]][i]["visible_ratio"]
            v1 = results[models[1]][i]["visible_ratio"]
            diff = v0 - v1
            if abs(diff) > 0.02:
                diffs.append((i, diff, v0, v1))
        diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        for i, diff, v0, v1 in diffs[:10]:
            r0, r1 = results[models[0]][i], results[models[1]][i]
            print(f"    case {i}: {short_names[0]}={v0:.3f}({r0['total_steps']}步) "
                  f"vs {short_names[1]}={v1:.3f}({r1['total_steps']}步) "
                  f"delta={diff:+.3f}")

    print(f"{'='*70}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare models on visibility avoidance")
    p.add_argument("--map", default=DEFAULT_MAP_PATH)
    p.add_argument("--model", action="append", default=[], help="model path (repeatable)")
    p.add_argument("--cases", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--n-enemies", type=int, default=3, help="enemies per case")
    p.add_argument("--enemy-range", type=int, default=60, help="enemy vision range")
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
        out_path = out_dir / f"visibility_comparison_{ts}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
