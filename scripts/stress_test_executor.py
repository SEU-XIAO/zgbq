from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battlefield_rl.config import EnvConfig, PlannerConfig
from battlefield_rl.env import EnemySpec
from battlefield_rl.map import BattlefieldMap, load_txt_map
from interfaces.config import DEFAULT_ENEMY_RANGE, DEFAULT_FOV_DEG, DEFAULT_MAP_PATH, DEFAULT_MODEL_PATH
from scripts.eval_hierarchical_executor import (
    FALLBACK_ESCAPE_STEPS,
    GOAL_TOLERANCE,
    MAX_FALLBACK_EVENTS,
    MAX_FALLBACK_STEPS,
    MAX_REPLANS_PER_TRAP,
    MAX_TOTAL_STEPS,
    MAX_WAYPOINTS,
    PROGRESS_RESET_MIN,
    SEGMENT_STEPS,
    WAYPOINT_INTERVAL,
    WAYPOINT_MAX_CHEBYSHEV,
    WAYPOINT_TOLERANCE,
    build_plan,
    chebyshev,
    coord_to_list,
    coords_to_list,
    enemy_to_dict,
    follow_geometric_path,
    load_policy,
    make_waypoints,
    manhattan,
    repeat_stats,
    run_segment,
    select_device,
    visibility_stats,
)

Coord = Tuple[int, int]

BUCKETS = {
    "short": (30, 90),
    "medium": (100, 180),
    "long": (200, 330),
}


def passable_cells(grid: BattlefieldMap) -> List[Coord]:
    rows, cols = grid.shape
    return [
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if not grid.is_static_blocked(row, col)
    ]


def random_enemy(rng: random.Random, cells: Sequence[Coord]) -> EnemySpec:
    row, col = rng.choice(cells)
    return EnemySpec(
        row=row,
        col=col,
        facing_deg=float(rng.randrange(0, 360, 45)),
        fov_deg=DEFAULT_FOV_DEG,
        max_range=DEFAULT_ENEMY_RANGE,
    )


def validate_path(grid: BattlefieldMap, path: Sequence[Coord]) -> Dict[str, int | bool]:
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


def execute_case(
    grid: BattlefieldMap,
    model,
    device,
    env_cfg: EnvConfig,
    planner_cfg: PlannerConfig,
    map_path: str,
    start: Coord,
    goal: Coord,
    enemies: Sequence[EnemySpec],
) -> Dict[str, object]:
    initial_plan = build_plan(
        grid=grid,
        current=start,
        goal=goal,
        env_cfg=env_cfg,
        planner_cfg=planner_cfg,
        waypoint_interval=WAYPOINT_INTERVAL,
        waypoint_max_chebyshev=WAYPOINT_MAX_CHEBYSHEV,
    )
    if not initial_plan.path:
        return {
            "success": False,
            "error": "no_global_path",
            "start": coord_to_list(start),
            "goal": coord_to_list(goal),
            "path_valid": False,
        }

    pos = start
    full_trace: List[Coord] = [start]
    waypoints = make_waypoints(initial_plan, pos, goal, WAYPOINT_TOLERANCE, MAX_WAYPOINTS)
    selected_waypoints: List[Coord] = []
    fallback_events: List[Dict[str, object]] = []

    reached_count = 0
    failed_segments = 0
    total_replans = 0
    trap_replans = 0
    visibility_refreshes = 0
    fallback_steps = 0
    best_goal_distance = manhattan(pos, goal)
    failed_reason = ""

    while (
        waypoints
        and reached_count < MAX_WAYPOINTS
        and len(full_trace) - 1 < MAX_TOTAL_STEPS
    ):
        if WAYPOINT_MAX_CHEBYSHEV > 0 and chebyshev(pos, waypoints[0]) > WAYPOINT_MAX_CHEBYSHEV:
            refresh = build_plan(
                grid=grid,
                current=pos,
                goal=goal,
                env_cfg=env_cfg,
                planner_cfg=planner_cfg,
                waypoint_interval=WAYPOINT_INTERVAL,
                waypoint_max_chebyshev=WAYPOINT_MAX_CHEBYSHEV,
            )
            visibility_refreshes += 1
            if not refresh.path:
                failed_reason = "visibility_refresh_failed"
                break
            remaining = max(1, MAX_WAYPOINTS - reached_count)
            waypoints = make_waypoints(refresh, pos, goal, WAYPOINT_TOLERANCE, remaining)

        waypoint = waypoints.pop(0)
        selected_waypoints.append(waypoint)
        tolerance = GOAL_TOLERANCE if waypoint == goal else WAYPOINT_TOLERANCE
        pos, trace, reached, reason = run_segment(
            grid=grid,
            model=model,
            device=device,
            start=pos,
            waypoint=waypoint,
            enemies=enemies,
            env_cfg=env_cfg,
            max_steps=SEGMENT_STEPS,
            tolerance=tolerance,
        )
        full_trace.extend(trace[1:])

        if reached:
            reached_count += 1
            current_goal_distance = manhattan(pos, goal)
            if current_goal_distance <= best_goal_distance - PROGRESS_RESET_MIN:
                best_goal_distance = current_goal_distance
                trap_replans = 0
            if manhattan(pos, goal) <= GOAL_TOLERANCE:
                break
            continue

        failed_segments += 1
        failed_reason = reason

        if trap_replans >= MAX_REPLANS_PER_TRAP:
            if len(fallback_events) >= MAX_FALLBACK_EVENTS:
                failed_reason = "max_fallback_events_reached"
                break
            remaining_fallback_steps = MAX_FALLBACK_STEPS - fallback_steps
            if remaining_fallback_steps <= 0:
                failed_reason = "max_fallback_steps_reached"
                break

            fallback_start = pos
            chunk_steps = min(FALLBACK_ESCAPE_STEPS, remaining_fallback_steps)
            pos, trace, fallback_success, fallback_reason = follow_geometric_path(
                grid=grid,
                start=pos,
                goal=goal,
                env_cfg=env_cfg,
                planner_cfg=planner_cfg,
                max_steps=chunk_steps,
            )
            chunk_used = len(trace) - 1
            fallback_steps += chunk_used
            full_trace.extend(trace[1:])
            fallback_events.append(
                {
                    "start": coord_to_list(fallback_start),
                    "end": coord_to_list(pos),
                    "steps": chunk_used,
                    "success": bool(fallback_success),
                    "reason": fallback_reason,
                }
            )
            if chunk_used <= 0:
                break
            if manhattan(pos, goal) <= GOAL_TOLERANCE:
                break

            refresh = build_plan(
                grid=grid,
                current=pos,
                goal=goal,
                env_cfg=env_cfg,
                planner_cfg=planner_cfg,
                waypoint_interval=WAYPOINT_INTERVAL,
                waypoint_max_chebyshev=WAYPOINT_MAX_CHEBYSHEV,
            )
            visibility_refreshes += 1
            if not refresh.path:
                failed_reason = "post_fallback_plan_failed"
                break
            remaining = max(1, MAX_WAYPOINTS - reached_count)
            waypoints = make_waypoints(refresh, pos, goal, WAYPOINT_TOLERANCE, remaining)
            best_goal_distance = min(best_goal_distance, manhattan(pos, goal))
            trap_replans = 0
            continue

        if manhattan(pos, goal) <= GOAL_TOLERANCE:
            break

        trap_replans += 1
        total_replans += 1
        replan_interval = max(4, WAYPOINT_INTERVAL // (trap_replans + 1))
        replan = build_plan(
            grid=grid,
            current=pos,
            goal=goal,
            env_cfg=env_cfg,
            planner_cfg=planner_cfg,
            waypoint_interval=replan_interval,
            waypoint_max_chebyshev=WAYPOINT_MAX_CHEBYSHEV,
        )
        if not replan.path:
            failed_reason = "replan_failed"
            break
        remaining = max(1, MAX_WAYPOINTS - reached_count)
        waypoints = make_waypoints(replan, pos, goal, WAYPOINT_TOLERANCE, remaining)

    success = manhattan(pos, goal) <= GOAL_TOLERANCE
    total_steps = len(full_trace) - 1
    visible_count, visible_ratio = visibility_stats(grid, full_trace, enemies)
    repeats = repeat_stats(full_trace, limit=5)
    validity = validate_path(grid, full_trace)
    path_efficiency = total_steps / max(1, len(initial_plan.path) - 1)

    return {
        "success": bool(success),
        "start": coord_to_list(start),
        "goal": coord_to_list(goal),
        "final_pos": coord_to_list(pos),
        "enemies": [enemy_to_dict(enemy) for enemy in enemies],
        "total_steps": total_steps,
        "visible_ratio": visible_ratio,
        "visible_steps": visible_count,
        "initial_global_path_steps": len(initial_plan.path) - 1,
        "path_efficiency": path_efficiency,
        "selected_waypoint_count": len(selected_waypoints),
        "replans": total_replans,
        "visibility_refreshes": visibility_refreshes,
        "geometric_fallback_count": len(fallback_events),
        "geometric_fallback_steps": fallback_steps,
        "local_executor_success": bool(success and not fallback_events),
        "failed_segments": failed_segments,
        "failed_reason": failed_reason,
        "path_valid": bool(validity["valid"]),
        "path_validity": validity,
        **repeats,
        "path_sample": {
            "head": coords_to_list(full_trace[:20]),
            "tail": coords_to_list(full_trace[-20:]),
        },
        "map": str(Path(map_path)),
    }


def sample_case(
    rng: random.Random,
    grid: BattlefieldMap,
    cells: Sequence[Coord],
    env_cfg: EnvConfig,
    planner_cfg: PlannerConfig,
    bucket: str,
    max_attempts: int = 500,
) -> Tuple[Coord, Coord, PlanResult]:
    low, high = BUCKETS[bucket]
    for _ in range(max_attempts):
        start = rng.choice(cells)
        goal = rng.choice(cells)
        if start == goal:
            continue
        if not (low <= manhattan(start, goal) <= high * 2):
            continue
        plan = build_plan(
            grid=grid,
            current=start,
            goal=goal,
            env_cfg=env_cfg,
            planner_cfg=planner_cfg,
            waypoint_interval=WAYPOINT_INTERVAL,
            waypoint_max_chebyshev=WAYPOINT_MAX_CHEBYSHEV,
        )
        steps = len(plan.path) - 1
        if plan.path and low <= steps <= high:
            return start, goal, plan
    raise RuntimeError(f"failed to sample {bucket} case")


def aggregate(cases: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not cases:
        return {}
    success_cases = [case for case in cases if case.get("success")]
    valid_cases = [case for case in cases if case.get("path_valid")]

    def avg(key: str, source: Sequence[Dict[str, object]] = cases) -> float:
        values = [float(case.get(key, 0.0)) for case in source]
        return sum(values) / max(1, len(values))

    failures = Counter(str(case.get("failed_reason", "")) for case in cases if not case.get("success"))
    return {
        "cases": len(cases),
        "success_rate": len(success_cases) / len(cases),
        "path_valid_rate": len(valid_cases) / len(cases),
        "local_executor_success_rate": sum(1 for case in cases if case.get("local_executor_success")) / len(cases),
        "avg_total_steps": avg("total_steps"),
        "avg_initial_global_path_steps": avg("initial_global_path_steps"),
        "avg_path_efficiency_success": avg("path_efficiency", success_cases),
        "avg_repeat_ratio": avg("repeat_ratio"),
        "avg_visible_ratio": avg("visible_ratio"),
        "avg_fallback_count": avg("geometric_fallback_count"),
        "avg_replans": avg("replans"),
        "failure_reasons": dict(failures),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test hierarchical battlefield executor")
    parser.add_argument("--map", default=DEFAULT_MAP_PATH)
    parser.add_argument("--cases-per-bucket", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    grid = load_txt_map(args.map)
    cells = passable_cells(grid)
    env_cfg = EnvConfig()
    planner_cfg = PlannerConfig()
    device = select_device()
    model = load_policy(PROJECT_ROOT / DEFAULT_MODEL_PATH, device)

    cases: List[Dict[str, object]] = []
    started = time.perf_counter()
    for bucket in BUCKETS:
        for _ in range(args.cases_per_bucket):
            start, goal, _ = sample_case(rng, grid, cells, env_cfg, planner_cfg, bucket)
            enemy_count = rng.choice([0, 1, 2])
            enemies = [random_enemy(rng, cells) for _ in range(enemy_count)]
            case_started = time.perf_counter()
            result = execute_case(
                grid=grid,
                model=model,
                device=device,
                env_cfg=env_cfg,
                planner_cfg=planner_cfg,
                map_path=args.map,
                start=start,
                goal=goal,
                enemies=enemies,
            )
            result["bucket"] = bucket
            result["runtime_sec"] = time.perf_counter() - case_started
            cases.append(result)

    by_bucket = {
        bucket: aggregate([case for case in cases if case.get("bucket") == bucket])
        for bucket in BUCKETS
    }
    report = {
        "seed": args.seed,
        "map": args.map,
        "device": str(device),
        "total_runtime_sec": time.perf_counter() - started,
        "summary": aggregate(cases),
        "by_bucket": by_bucket,
        "worst_by_efficiency": sorted(
            cases,
            key=lambda case: float(case.get("path_efficiency", 0.0)),
            reverse=True,
        )[:5],
        "cases": cases,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"stress_test_{timestamp}.json"
    report["output_path"] = str(output_path)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
