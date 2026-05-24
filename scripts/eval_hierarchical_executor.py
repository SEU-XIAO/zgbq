from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battlefield_rl.config import EnvConfig, PlannerConfig
from battlefield_rl.env import ACTIONS_8, EnemySpec, TacticalBattlefieldEnv
from battlefield_rl.env.tactical_env import angle_deg, angle_diff, bresenham_line
from battlefield_rl.map import BattlefieldMap, load_txt_map
from battlefield_rl.planner import PlanResult, plan_global_path
from battlefield_rl.rl.network import TacticalD3QN, masked_q_values

Coord = Tuple[int, int]

DEFAULT_MODEL_PATH = "episode_8000.pt"
WAYPOINT_INTERVAL = 18
WAYPOINT_MAX_CHEBYSHEV = 10
SEGMENT_STEPS = 45
WAYPOINT_TOLERANCE = 3
GOAL_TOLERANCE = 0
MAX_WAYPOINTS = 500
MAX_REPLANS_PER_TRAP = 2
FALLBACK_ESCAPE_STEPS = 20
MAX_FALLBACK_EVENTS = 20
MAX_FALLBACK_STEPS = 2000
PROGRESS_RESET_MIN = 3
MAX_TOTAL_STEPS = 12000
OSCILLATION_WINDOW = 10
OSCILLATION_UNIQUE_LIMIT = 3
OSCILLATION_MIN_STEPS = 8


def parse_coord(text: str) -> Coord:
    row, col = text.split(",")
    return int(row), int(col)


def parse_enemy(text: str) -> EnemySpec:
    parts = text.split(",")
    if len(parts) != 5:
        raise ValueError("enemy format must be row,col,facing_deg,fov_deg,range")
    return EnemySpec(
        row=int(parts[0]),
        col=int(parts[1]),
        facing_deg=float(parts[2]),
        fov_deg=float(parts[3]),
        max_range=int(parts[4]),
    )


def safe_coord_name(coord: Coord) -> str:
    return f"{coord[0]}_{coord[1]}"


def coord_to_list(coord: Coord) -> List[int]:
    return [int(coord[0]), int(coord[1])]


def coords_to_list(coords: Sequence[Coord]) -> List[List[int]]:
    return [coord_to_list(coord) for coord in coords]


def enemy_to_dict(enemy: EnemySpec) -> Dict[str, float | int]:
    return {
        "row": int(enemy.row),
        "col": int(enemy.col),
        "facing_deg": float(enemy.facing_deg),
        "fov_deg": float(enemy.fov_deg),
        "range": int(enemy.max_range),
    }


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_policy(path: str | Path, device: torch.device) -> TacticalD3QN:
    model = TacticalD3QN().to(device)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint["online"] if isinstance(checkpoint, dict) and "online" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def greedy_action(model: TacticalD3QN, obs: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
    x = torch.from_numpy(obs).unsqueeze(0).float().to(device)
    m = torch.from_numpy(mask).unsqueeze(0).float().to(device)
    q = masked_q_values(model(x), m)
    return int(torch.argmax(q, dim=1).item())


def validate_endpoint(grid: BattlefieldMap, point: Coord, label: str) -> None:
    if not grid.in_bounds(point[0], point[1]):
        raise ValueError(f"{label} is out of map bounds: {point}")
    if grid.is_static_blocked(point[0], point[1]):
        raise ValueError(f"{label} is blocked: {point}")


def build_plan(
    grid: BattlefieldMap,
    current: Coord,
    goal: Coord,
    env_cfg: EnvConfig,
    planner_cfg: PlannerConfig,
    waypoint_interval: int,
    waypoint_max_chebyshev: int | None,
) -> PlanResult:
    return plan_global_path(
        grid,
        current,
        goal,
        max_height_diff=env_cfg.max_height_diff,
        waypoint_interval=waypoint_interval,
        allow_diagonal=planner_cfg.allow_diagonal,
        use_jps=True,
        waypoint_max_chebyshev=waypoint_max_chebyshev,
    )


def make_waypoints(plan: PlanResult, current: Coord, goal: Coord, tolerance: int, limit: int) -> List[Coord]:
    waypoints: List[Coord] = []
    for waypoint in plan.waypoints:
        if waypoint == current:
            continue
        if waypoint != goal and manhattan(current, waypoint) <= tolerance:
            continue
        waypoints.append(waypoint)
    if not waypoints or waypoints[-1] != goal:
        waypoints.append(goal)
    return waypoints[:limit]


def run_segment(
    grid: BattlefieldMap,
    model: TacticalD3QN,
    device: torch.device,
    start: Coord,
    waypoint: Coord,
    enemies: Sequence[EnemySpec],
    env_cfg: EnvConfig,
    max_steps: int,
    tolerance: int,
) -> Tuple[Coord, List[Coord], bool, str]:
    env = TacticalBattlefieldEnv(grid, env_cfg)
    obs = env.reset(start=start, goal=waypoint, enemies=enemies, waypoints=[waypoint], threat_scale=1.0)
    trace = [start]

    for _ in range(max_steps):
        if manhattan(env.pos, waypoint) <= tolerance:
            return env.pos, trace, True, "reached"
        action = greedy_action(model, obs, env.action_mask(), device)
        step = env.step(action)
        obs = step.observation
        trace.append(env.pos)
        if manhattan(env.pos, waypoint) <= tolerance:
            return env.pos, trace, True, "reached"
        if len(trace) >= OSCILLATION_MIN_STEPS:
            recent = trace[-OSCILLATION_WINDOW:]
            if len(set(recent)) <= OSCILLATION_UNIQUE_LIMIT:
                return env.pos, trace, False, "oscillation"
        if step.done:
            if step.info.get("stuck"):
                return env.pos, trace, False, "stuck"
            if step.info.get("timeout"):
                return env.pos, trace, False, "timeout"
            return env.pos, trace, False, "done"

    return env.pos, trace, False, "segment_limit"


def follow_geometric_path(
    grid: BattlefieldMap,
    start: Coord,
    goal: Coord,
    env_cfg: EnvConfig,
    planner_cfg: PlannerConfig,
    max_steps: int,
) -> Tuple[Coord, List[Coord], bool, str]:
    plan = build_plan(
        grid=grid,
        current=start,
        goal=goal,
        env_cfg=env_cfg,
        planner_cfg=planner_cfg,
        waypoint_interval=WAYPOINT_INTERVAL,
        waypoint_max_chebyshev=None,
    )
    if not plan.path:
        return start, [start], False, "fallback_no_path"

    trace = [start]
    pos = start
    for nxt in plan.path[1 : max_steps + 1]:
        dr = nxt[0] - pos[0]
        dc = nxt[1] - pos[1]
        if (dr, dc) not in ACTIONS_8:
            return pos, trace, False, "fallback_non_adjacent_path"
        if not grid.is_passable(pos, nxt, env_cfg.max_height_diff):
            return pos, trace, False, "fallback_blocked_path"
        pos = nxt
        trace.append(pos)
        if manhattan(pos, goal) <= GOAL_TOLERANCE:
            return pos, trace, True, "fallback_reached"

    if manhattan(pos, goal) <= GOAL_TOLERANCE:
        return pos, trace, True, "fallback_reached"
    return pos, trace, False, "fallback_step_limit"


def visible_by_enemy(grid: BattlefieldMap, point: Coord, enemy: EnemySpec) -> bool:
    src = (enemy.row, enemy.col)
    distance = math.hypot(point[0] - src[0], point[1] - src[1])
    if distance > enemy.max_range:
        return False
    if angle_diff(angle_deg(src, point), enemy.facing_deg) > enemy.fov_deg * 0.5:
        return False

    line = bresenham_line(src[0], src[1], point[0], point[1])
    for row, col in line[1:-1]:
        if grid.is_static_blocked(row, col):
            return False
    return True


def visibility_stats(
    grid: BattlefieldMap,
    path: Sequence[Coord],
    enemies: Sequence[EnemySpec],
) -> Tuple[int, float]:
    visible_count = 0
    for point in path:
        if any(visible_by_enemy(grid, point, enemy) for enemy in enemies):
            visible_count += 1
    visible_ratio = visible_count / max(1, len(path))
    return visible_count, visible_ratio


def repeat_stats(path: Sequence[Coord], limit: int = 10) -> Dict[str, object]:
    counter = Counter(path)
    repeat_steps = len(path) - len(counter)
    repeat_ratio = repeat_steps / max(1, len(path))
    most_repeated = [
        {"point": coord_to_list(point), "count": int(count)}
        for point, count in counter.most_common(limit)
        if count > 1
    ]
    return {
        "repeat_steps": repeat_steps,
        "repeat_ratio": repeat_ratio,
        "most_repeated_points": most_repeated,
    }


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    if model_path is None or str(model_path).strip() == "":
        return PROJECT_ROOT / DEFAULT_MODEL_PATH
    path = Path(model_path)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def plan_and_execute(
    map_path: str,
    start: Coord,
    goal: Coord,
    enemies: Sequence[EnemySpec],
    model_path: str | Path | None = None,
) -> Dict[str, object]:
    grid = load_txt_map(map_path)
    validate_endpoint(grid, start, "start")
    validate_endpoint(grid, goal, "goal")

    resolved_model_path = resolve_model_path(model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"model not found: {resolved_model_path}")

    env_cfg = EnvConfig()
    planner_cfg = PlannerConfig()
    device = select_device()

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
            "path": [coord_to_list(start)],
        }

    model = load_policy(resolved_model_path, device)
    pos = start
    full_trace: List[Coord] = [start]
    plans_used = [initial_plan.planner]
    waypoints = make_waypoints(initial_plan, pos, goal, WAYPOINT_TOLERANCE, MAX_WAYPOINTS)
    initial_waypoints = waypoints[:]
    selected_waypoints: List[Coord] = []
    segments: List[Dict[str, object]] = []
    fallback_events: List[Dict[str, object]] = []

    reached_count = 0
    failed_segments = 0
    total_replans = 0
    trap_replans = 0
    visibility_refreshes = 0
    fallback_steps = 0
    best_goal_distance = manhattan(pos, goal)
    failed_waypoint: Coord | None = None
    failed_pos: Coord | None = None
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
            plans_used.append(refresh.planner)
            visibility_refreshes += 1
            if not refresh.path:
                failed_waypoint = waypoints[0]
                failed_pos = pos
                failed_reason = "visibility_refresh_failed"
                break
            remaining = max(1, MAX_WAYPOINTS - reached_count)
            waypoints = make_waypoints(refresh, pos, goal, WAYPOINT_TOLERANCE, remaining)

        waypoint = waypoints.pop(0)
        selected_waypoints.append(waypoint)
        tolerance = GOAL_TOLERANCE if waypoint == goal else WAYPOINT_TOLERANCE
        segment_start = pos
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
        segments.append(
            {
                "mode": "model",
                "start": coord_to_list(segment_start),
                "target": coord_to_list(waypoint),
                "end": coord_to_list(pos),
                "steps": len(trace) - 1,
                "reached": bool(reached),
                "reason": reason,
            }
        )

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
        failed_waypoint = waypoint
        failed_pos = pos
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
            plans_used.append(refresh.planner)
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
        plans_used.append(replan.planner)
        if not replan.path:
            failed_reason = "replan_failed"
            break
        remaining = max(1, MAX_WAYPOINTS - reached_count)
        waypoints = make_waypoints(replan, pos, goal, WAYPOINT_TOLERANCE, remaining)

    success = manhattan(pos, goal) <= GOAL_TOLERANCE
    total_steps = len(full_trace) - 1
    visible_count, visible_ratio = visibility_stats(grid, full_trace, enemies)
    repeats = repeat_stats(full_trace)
    path_efficiency = total_steps / max(1, len(initial_plan.path) - 1)

    selected_intermediate = [point for point in selected_waypoints if point != goal]
    initial_intermediate = [point for point in initial_waypoints if point != goal]

    return {
        "success": bool(success),
        "start": coord_to_list(start),
        "goal": coord_to_list(goal),
        "final_pos": coord_to_list(pos),
        "map": str(Path(map_path)),
        "model": str(resolved_model_path),
        "device": str(device),
        "enemies": [enemy_to_dict(enemy) for enemy in enemies],
        "path": coords_to_list(full_trace),
        "total_steps": total_steps,
        "visible_ratio": visible_ratio,
        "visible_steps": visible_count,
        **repeats,
        "initial_global_path_steps": len(initial_plan.path) - 1,
        "initial_global_jump_points_count": len(initial_plan.jump_points),
        "path_efficiency": path_efficiency,
        "initial_intermediate_count": len(initial_intermediate),
        "initial_intermediate_points": coords_to_list(initial_intermediate),
        "selected_intermediate_count": len(selected_intermediate),
        "selected_intermediate_points": coords_to_list(selected_intermediate),
        "selected_waypoints": coords_to_list(selected_waypoints),
        "segments": segments,
        "replans": total_replans,
        "visibility_refreshes": visibility_refreshes,
        "geometric_fallback_count": len(fallback_events),
        "geometric_fallback_steps": fallback_steps,
        "geometric_fallback_events": fallback_events,
        "local_executor_success": bool(success and not fallback_events),
        "failed_segments": failed_segments,
        "failed_waypoint": coord_to_list(failed_waypoint) if failed_waypoint is not None else None,
        "failed_pos": coord_to_list(failed_pos) if failed_pos is not None else None,
        "failed_reason": failed_reason,
        "plans_used": plans_used,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Return battlefield path planning result as JSON")
    parser.add_argument("--map", required=True, help="txt map path")
    parser.add_argument("--model", default=None, help="model checkpoint path; defaults to episode_8000.pt")
    parser.add_argument("--start", required=True, help="row,col")
    parser.add_argument("--goal", required=True, help="row,col")
    parser.add_argument("--enemy", action="append", default=[], help="row,col,facing_deg,fov_deg,range")
    parser.add_argument("--output-dir", default=None, help="optional directory to write result json")
    return parser.parse_args()


def write_result_if_requested(result: Dict[str, object], output_dir: str | None, start: Coord, goal: Coord) -> str | None:
    if not output_dir:
        return None
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"path_{safe_coord_name(start)}_to_{safe_coord_name(goal)}_{timestamp}.json"
    output_path = out_dir / filename
    result["output_path"] = str(output_path)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def main() -> None:
    args = parse_args()
    start: Coord | None = None
    goal: Coord | None = None
    try:
        start = parse_coord(args.start)
        goal = parse_coord(args.goal)
        result = plan_and_execute(
            map_path=args.map,
            start=start,
            goal=goal,
            enemies=[parse_enemy(enemy) for enemy in args.enemy],
            model_path=args.model,
        )
        write_result_if_requested(result, args.output_dir, start, goal)
    except Exception as exc:
        result = {
            "success": False,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        }
        if args.output_dir and start is not None and goal is not None:
            try:
                write_result_if_requested(result, args.output_dir, start, goal)
            except Exception:
                pass
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
