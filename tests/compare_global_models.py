"""
全局模型对比测试

在真实大地图上用 plan_and_execute (全局规划 + 局部执行 + 几何回退)
对比多个模型的完整表现。

用法:
  python tests/compare_global_models.py
  python tests/compare_global_models.py --model episode_6000.pt --model episode_8000.pt
  python tests/compare_global_models.py --cases 30 --bucket medium --enemy 50,50,225,90,25
  python tests/compare_global_models.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battlefield_rl.env import ACTIONS_8, EnemySpec
from battlefield_rl.map import BattlefieldMap, load_txt_map
from interfaces.config import DEFAULT_MAP_PATH
from scripts.eval_hierarchical_executor import plan_and_execute

Coord = tuple[int, int]

BUCKETS = {
    "short": (20, 80),
    "medium": (90, 180),
    "long": (190, 330),
}


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


def sample_cases(
    rng: random.Random,
    grid: BattlefieldMap,
    cases: int,
    bucket: str,
) -> list[tuple[Coord, Coord]]:
    low, high = BUCKETS[bucket]
    cells = passable_cells(grid)
    sampled: list[tuple[Coord, Coord]] = []
    attempts = 0
    while len(sampled) < cases and attempts < cases * 2000:
        attempts += 1
        start = rng.choice(cells)
        goal = rng.choice(cells)
        if start == goal:
            continue
        path = bfs_path(grid, start, goal)
        steps = len(path) - 1
        if path and low <= steps <= high:
            sampled.append((start, goal))
    if len(sampled) < cases:
        raise RuntimeError(f"只采样到 {len(sampled)} 个, 需要 {cases} 个 (bucket={bucket})")
    return sampled


def parse_enemy(text: str) -> EnemySpec:
    parts = text.split(",")
    return EnemySpec(
        row=int(parts[0]), col=int(parts[1]),
        facing_deg=float(parts[2]), fov_deg=float(parts[3]),
        max_range=int(parts[4]),
    )


def run_comparison(
    map_path: str,
    model_paths: list[str],
    pairs: list[tuple[Coord, Coord]],
    enemies: list[EnemySpec],
    use_fallback: bool,
) -> dict[str, Any]:
    grid = load_txt_map(map_path)
    results: dict[str, list[dict[str, Any]]] = {m: [] for m in model_paths}

    for i, (start, goal) in enumerate(pairs):
        bfs = bfs_path(grid, start, goal)
        shortest = max(1, len(bfs) - 1)

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
                "bfs_steps": shortest,
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
                "segments": r.get("segments", []),
                "runtime_sec": elapsed,
            })

        # 逐 case 打印
        status_parts = []
        for mp in model_paths:
            rr = results[mp][-1]
            icon = "OK" if rr["success"] else "XX"
            loc = "L" if rr["local_success"] else "F" if rr["geometric_fallback_count"] > 0 else "X"
            status_parts.append(f"{mp.split('/')[-1].split('.')[0]}={icon}/{loc}({rr['total_steps']}步)")

        print(f"  case {i:3d}: {start}->{goal} bfs={shortest}  {'  '.join(status_parts)}")

    return {
        "map": map_path,
        "models": model_paths,
        "cases": len(pairs),
        "enemies": [e.__dict__ for e in enemies],
        "use_fallback": use_fallback,
        "results": results,
    }


def print_summary(data: dict[str, Any]) -> None:
    results = data["results"]
    n = data["cases"]

    print(f"\n{'═'*70}")
    print(f"  GLOBAL COMPARISON  ({n} cases, fallback={'ON' if data['use_fallback'] else 'OFF'})")
    print(f"{'═'*70}")

    # 表头
    models = list(results.keys())
    short_names = [m.split("/")[-1].split(".")[0] for m in models]
    header = f"  {'Metric':<28s}" + "".join(f"{sn:>14s}" for sn in short_names)
    print(header)
    print(f"  {'─'*70}")

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

    # 失败原因分布
    print(f"\n  Failure reasons:")
    for mp, sn in zip(models, short_names):
        reasons = Counter(r["failed_reason"] for r in results[mp] if not r["success"])
        if reasons:
            print(f"    {sn}: {dict(reasons)}")
        else:
            print(f"    {sn}: (none)")

    # 逐 case 成功率对比
    print(f"\n  Per-case agreement:")
    agree_both = 0
    agree_fail = 0
    disagree = 0
    for i in range(n):
        succ = [results[mp][i]["success"] for mp in models]
        if all(succ):
            agree_both += 1
        elif not any(succ):
            agree_fail += 1
        else:
            disagree += 1
    print(f"    Both succeed: {agree_both}/{n}")
    print(f"    Both fail   : {agree_fail}/{n}")
    print(f"    Disagree    : {disagree}/{n}")

    if disagree > 0 and len(models) == 2:
        print(f"\n    Disagreement details:")
        for i in range(n):
            s0 = results[models[0]][i]["success"]
            s1 = results[models[1]][i]["success"]
            if s0 != s1:
                r0, r1 = results[models[0]][i], results[models[1]][i]
                print(f"      case {i}: {short_names[0]}={'OK' if s0 else 'XX'}({r0['total_steps']}步 "
                      f"vis={r0['visible_ratio']:.2f}) "
                      f"vs {short_names[1]}={'OK' if s1 else 'XX'}({r1['total_steps']}步 "
                      f"vis={r1['visible_ratio']:.2f})")

    print(f"{'═'*70}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare models with full global+local pipeline")
    p.add_argument("--map", default=DEFAULT_MAP_PATH)
    p.add_argument("--model", action="append", default=[], help="model path (repeatable)")
    p.add_argument("--cases", type=int, default=30)
    p.add_argument("--bucket", choices=sorted(BUCKETS), default="short")
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--enemy", action="append", default=[], help="row,col,facing,fov,range")
    p.add_argument("--no-fallback", action="store_true")
    p.add_argument("--output-dir", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    model_paths = args.model or ["episode_6000.pt", "episode_8000.pt"]
    enemies = [parse_enemy(e) for e in args.enemy]
    rng = random.Random(args.seed)
    grid = load_txt_map(args.map)
    pairs = sample_cases(rng, grid, args.cases, args.bucket)

    print(f"Map: {args.map}  Cases: {args.cases}  Bucket: {args.bucket}")
    print(f"Models: {model_paths}")
    print(f"Enemies: {len(enemies)}  Fallback: {'OFF' if args.no_fallback else 'ON'}")

    data = run_comparison(
        map_path=args.map, model_paths=model_paths,
        pairs=pairs, enemies=enemies,
        use_fallback=not args.no_fallback,
    )
    print_summary(data)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = out_dir / f"compare_global_models_{ts}.json"
        # 不保存完整 segments 避免文件过大
        for mp in data["results"]:
            for r in data["results"][mp]:
                r.pop("segments", None)
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
