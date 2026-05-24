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

from battlefield_rl.config import EnvConfig
from battlefield_rl.env import ACTIONS_8, EnemySpec, TacticalBattlefieldEnv
from battlefield_rl.map import BattlefieldMap, load_txt_map
from scripts.eval_hierarchical_executor import plan_and_execute, visibility_stats


Coord = tuple[int, int]

BUCKETS = {
    "short": (20, 80),
    "medium": (90, 180),
    "long": (190, 330),
}


def parse_coord(text: str) -> Coord:
    row, col = text.split(",")
    return int(row), int(col)


def parse_enemy(text: str) -> EnemySpec:
    row, col, facing_deg, fov_deg, max_range = text.split(",")
    return EnemySpec(
        row=int(row),
        col=int(col),
        facing_deg=float(facing_deg),
        fov_deg=float(fov_deg),
        max_range=int(max_range),
    )


def coord_to_list(coord: Coord) -> list[int]:
    return [int(coord[0]), int(coord[1])]


def coords_to_list(path: Sequence[Coord]) -> list[list[int]]:
    return [coord_to_list(point) for point in path]


def passable_cells(grid: BattlefieldMap) -> list[Coord]:
    rows, cols = grid.shape
    return [
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if not grid.is_static_blocked(row, col)
    ]


def validate_endpoint(grid: BattlefieldMap, point: Coord, label: str) -> None:
    if not grid.in_bounds(point[0], point[1]):
        raise ValueError(f"{label} is out of map bounds: {point}")
    if grid.is_static_blocked(point[0], point[1]):
        raise ValueError(f"{label} is blocked: {point}")


def validate_path(grid: BattlefieldMap, path: Sequence[Coord]) -> dict[str, int | bool]:
    out_of_bounds = 0
    blocked = 0
    non_adjacent = 0
    stay_steps = 0

    for row, col in path:
        if not grid.in_bounds(row, col):
            out_of_bounds += 1
        elif grid.is_static_blocked(row, col):
            blocked += 1

    for cur, nxt in zip(path, path[1:]):
        dr = abs(cur[0] - nxt[0])
        dc = abs(cur[1] - nxt[1])
        if dr == 0 and dc == 0:
            stay_steps += 1
        if max(dr, dc) > 1:
            non_adjacent += 1

    return {
        "valid": out_of_bounds == 0 and blocked == 0 and non_adjacent == 0 and stay_steps == 0,
        "out_of_bounds": out_of_bounds,
        "blocked": blocked,
        "non_adjacent": non_adjacent,
        "stay_steps": stay_steps,
    }


def repeat_stats(path: Sequence[Coord]) -> dict[str, Any]:
    counter = Counter(path)
    repeat_steps = len(path) - len(counter)
    return {
        "repeat_steps": repeat_steps,
        "repeat_ratio": repeat_steps / max(1, len(path)),
        "most_repeated_points": [
            {"point": coord_to_list(point), "count": int(count)}
            for point, count in counter.most_common(5)
            if count > 1
        ],
    }


def bfs_path(grid: BattlefieldMap, start: Coord, goal: Coord) -> list[Coord]:
    if start == goal:
        return [start]

    queue: deque[Coord] = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}

    while queue:
        cur = queue.popleft()
        for dr, dc in ACTIONS_8:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in parent:
                continue
            if not grid.is_passable(cur, nxt, 0):
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


def run_heuristic(
    grid: BattlefieldMap,
    start: Coord,
    goal: Coord,
    enemies: Sequence[EnemySpec],
    *,
    max_steps: int,
) -> tuple[list[Coord], bool, str]:
    env = TacticalBattlefieldEnv(grid, EnvConfig(timeout_steps=max_steps, no_progress_limit=max_steps + 1))
    env.reset(start=start, goal=goal, enemies=enemies, waypoints=[goal], threat_scale=1.0)
    path = [start]
    recent: deque[Coord] = deque(maxlen=12)

    for _ in range(max_steps):
        action = env.heuristic_action()
        step = env.step(action)
        path.append(env.pos)
        recent.append(env.pos)
        if env.pos == goal:
            return path, True, "reached"
        if len(recent) == recent.maxlen and len(set(recent)) <= 3:
            return path, False, "oscillation"
        if step.done:
            if step.info.get("timeout"):
                return path, False, "timeout"
            if step.info.get("stuck"):
                return path, False, "stuck"
            return path, False, "done"

    return path, False, "step_limit"


def summarize_path_result(
    *,
    name: str,
    grid: BattlefieldMap,
    path: Sequence[Coord],
    enemies: Sequence[EnemySpec],
    success: bool,
    reason: str,
    shortest_steps: int,
    runtime_sec: float,
) -> dict[str, Any]:
    visible_steps, visible_ratio = visibility_stats(grid, path, enemies)
    total_steps = max(0, len(path) - 1)
    return {
        "algorithm": name,
        "success": bool(success),
        "reason": reason,
        "total_steps": total_steps,
        "shortest_steps": shortest_steps,
        "path_efficiency": total_steps / max(1, shortest_steps),
        "visible_steps": visible_steps,
        "visible_ratio": visible_ratio,
        "runtime_sec": runtime_sec,
        "path_validity": validate_path(grid, path),
        **repeat_stats(path),
        "path_sample": {
            "head": coords_to_list(path[:20]),
            "tail": coords_to_list(path[-20:]),
        },
    }


def normalize_current_result(
    *,
    grid: BattlefieldMap,
    result: dict[str, Any],
    shortest_steps: int,
    runtime_sec: float,
) -> dict[str, Any]:
    path = [
        (int(point[0]), int(point[1]))
        for point in result.get("path", [])
        if isinstance(point, list) and len(point) >= 2
    ]
    out = summarize_path_result(
        name="current",
        grid=grid,
        path=path,
        enemies=[],
        success=bool(result.get("success")),
        reason=str(result.get("failed_reason") or result.get("error") or ""),
        shortest_steps=shortest_steps,
        runtime_sec=runtime_sec,
    )
    out["visible_steps"] = int(result.get("visible_steps", out["visible_steps"]))
    out["visible_ratio"] = float(result.get("visible_ratio", out["visible_ratio"]))
    out["raw_summary"] = {
        "final_pos": result.get("final_pos"),
        "replans": result.get("replans"),
        "visibility_refreshes": result.get("visibility_refreshes"),
        "geometric_fallback_count": result.get("geometric_fallback_count"),
        "local_executor_success": result.get("local_executor_success"),
        "plans_used": result.get("plans_used"),
    }
    return out


def run_case(
    *,
    grid: BattlefieldMap,
    map_path: str,
    start: Coord,
    goal: Coord,
    enemies: Sequence[EnemySpec],
    skip_current: bool,
    heuristic_max_steps: int,
) -> dict[str, Any]:
    validate_endpoint(grid, start, "start")
    validate_endpoint(grid, goal, "goal")

    case: dict[str, Any] = {
        "start": coord_to_list(start),
        "goal": coord_to_list(goal),
        "enemy_count": len(enemies),
        "algorithms": {},
    }

    started = time.perf_counter()
    bfs = bfs_path(grid, start, goal)
    bfs_runtime = time.perf_counter() - started
    shortest_steps = max(0, len(bfs) - 1)
    case["algorithms"]["bfs"] = summarize_path_result(
        name="bfs",
        grid=grid,
        path=bfs,
        enemies=enemies,
        success=bool(bfs and bfs[-1] == goal),
        reason="reached" if bfs and bfs[-1] == goal else "no_path",
        shortest_steps=max(1, shortest_steps),
        runtime_sec=bfs_runtime,
    )

    max_steps = heuristic_max_steps if heuristic_max_steps > 0 else max(80, shortest_steps * 4 + 30)
    started = time.perf_counter()
    heuristic_path, heuristic_success, heuristic_reason = run_heuristic(
        grid,
        start,
        goal,
        enemies,
        max_steps=max_steps,
    )
    heuristic_runtime = time.perf_counter() - started
    case["algorithms"]["heuristic"] = summarize_path_result(
        name="heuristic",
        grid=grid,
        path=heuristic_path,
        enemies=enemies,
        success=heuristic_success,
        reason=heuristic_reason,
        shortest_steps=max(1, shortest_steps),
        runtime_sec=heuristic_runtime,
    )

    if not skip_current:
        started = time.perf_counter()
        current_result = plan_and_execute(map_path=map_path, start=start, goal=goal, enemies=enemies)
        current_runtime = time.perf_counter() - started
        case["algorithms"]["current"] = normalize_current_result(
            grid=grid,
            result=current_result,
            shortest_steps=max(1, shortest_steps),
            runtime_sec=current_runtime,
        )

    return case


def sample_cases(
    rng: random.Random,
    grid: BattlefieldMap,
    *,
    cases: int,
    bucket: str,
) -> list[tuple[Coord, Coord]]:
    low, high = BUCKETS[bucket]
    cells = passable_cells(grid)
    sampled: list[tuple[Coord, Coord]] = []
    attempts = 0
    while len(sampled) < cases and attempts < cases * 1000:
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
        raise RuntimeError(f"only sampled {len(sampled)} cases for bucket={bucket}, requested {cases}")
    return sampled


def aggregate(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    algorithms = sorted({name for case in cases for name in case["algorithms"]})
    summary: dict[str, Any] = {}
    for name in algorithms:
        rows = [case["algorithms"][name] for case in cases if name in case["algorithms"]]
        successes = [row for row in rows if row["success"]]

        def avg(key: str, source: Sequence[dict[str, Any]] = rows) -> float:
            if not source:
                return 0.0
            return sum(float(row.get(key, 0.0)) for row in source) / len(source)

        summary[name] = {
            "cases": len(rows),
            "success_rate": len(successes) / max(1, len(rows)),
            "avg_total_steps": avg("total_steps"),
            "avg_path_efficiency": avg("path_efficiency", successes),
            "avg_visible_ratio": avg("visible_ratio"),
            "avg_runtime_sec": avg("runtime_sec"),
            "failure_reasons": dict(Counter(str(row["reason"]) for row in rows if not row["success"])),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare BFS, heuristic executor, and current path algorithm")
    parser.add_argument("--map", required=True, help="txt map path")
    parser.add_argument("--start", default="", help="single-case start as row,col")
    parser.add_argument("--goal", default="", help="single-case goal as row,col")
    parser.add_argument("--enemy", action="append", default=[], help="row,col,facing_deg,fov_deg,range")
    parser.add_argument("--cases", type=int, default=1, help="random cases when --start/--goal are omitted")
    parser.add_argument("--bucket", choices=sorted(BUCKETS), default="short")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--skip-current", action="store_true", help="do not load/run the current model executor")
    parser.add_argument("--heuristic-max-steps", type=int, default=0, help="0 means auto")
    parser.add_argument("--output-dir", default="outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    grid = load_txt_map(args.map)
    enemies = [parse_enemy(item) for item in args.enemy]

    if args.start or args.goal:
        if not args.start or not args.goal:
            raise ValueError("--start and --goal must be provided together")
        pairs = [(parse_coord(args.start), parse_coord(args.goal))]
    else:
        pairs = sample_cases(rng, grid, cases=args.cases, bucket=args.bucket)

    started = time.perf_counter()
    cases = [
        run_case(
            grid=grid,
            map_path=args.map,
            start=start,
            goal=goal,
            enemies=enemies,
            skip_current=args.skip_current,
            heuristic_max_steps=args.heuristic_max_steps,
        )
        for start, goal in pairs
    ]

    report = {
        "map": args.map,
        "seed": args.seed,
        "bucket": args.bucket,
        "case_count": len(cases),
        "enemy_count": len(enemies),
        "skip_current": bool(args.skip_current),
        "runtime_sec": time.perf_counter() - started,
        "summary": aggregate(cases),
        "cases": cases,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = out_dir / f"benchmark_path_algorithms_{timestamp}.json"
    report["output_path"] = str(output_path)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
