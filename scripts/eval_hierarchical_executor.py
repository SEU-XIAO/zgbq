from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battlefield_rl.config import EnvConfig, PlannerConfig
from battlefield_rl.env import ACTIONS_8, EnemySpec, TacticalBattlefieldEnv
from battlefield_rl.map import BattlefieldMap, load_txt_map
from battlefield_rl.planner import PlanResult, plan_global_path
from battlefield_rl.rl.network import TacticalD3QN, masked_q_values

Coord = Tuple[int, int]


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


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def load_policy(path: str, device: torch.device) -> TacticalD3QN:
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
    waypoint_interval: int,
    tolerance: int,
    max_steps: int,
    use_jps: bool,
) -> Tuple[Coord, List[Coord], bool, str]:
    plan = build_plan(
        grid=grid,
        current=start,
        goal=goal,
        env_cfg=env_cfg,
        planner_cfg=planner_cfg,
        waypoint_interval=waypoint_interval,
        use_jps=use_jps,
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
        if manhattan(pos, goal) <= tolerance:
            return pos, trace, True, "fallback_reached"

    if manhattan(pos, goal) <= tolerance:
        return pos, trace, True, "fallback_reached"
    return pos, trace, False, "fallback_step_limit"


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


def summarize_points(points: Sequence[Coord], max_items: int = 12) -> str:
    if len(points) <= max_items:
        return str(list(points))
    head = list(points[: max_items // 2])
    tail = list(points[-(max_items // 2) :])
    return f"{head} ... {tail}"


def build_plan(
    grid: BattlefieldMap,
    current: Coord,
    goal: Coord,
    env_cfg: EnvConfig,
    planner_cfg: PlannerConfig,
    waypoint_interval: int,
    use_jps: bool,
    waypoint_max_chebyshev: int | None,
) -> PlanResult:
    return plan_global_path(
        grid,
        current,
        goal,
        max_height_diff=env_cfg.max_height_diff,
        waypoint_interval=waypoint_interval,
        allow_diagonal=planner_cfg.allow_diagonal,
        use_jps=use_jps,
        waypoint_max_chebyshev=waypoint_max_chebyshev,
    )


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def validate_endpoint(grid: BattlefieldMap, point: Coord, label: str) -> None:
    if not grid.in_bounds(point[0], point[1]):
        raise ValueError(f"{label} is out of map bounds: {point}")
    if grid.is_static_blocked(point[0], point[1]):
        raise ValueError(f"{label} is blocked: {point}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JPS/A* global planner + trained local executor")
    parser.add_argument("--map", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--start", required=True, help="row,col")
    parser.add_argument("--goal", required=True, help="row,col")
    parser.add_argument("--enemy", action="append", default=[], help="row,col,facing_deg,fov_deg,range")
    parser.add_argument("--waypoint-interval", type=int, default=10)
    parser.add_argument("--waypoint-max-chebyshev", type=int, default=10)
    parser.add_argument("--segment-steps", type=int, default=35)
    parser.add_argument("--waypoint-tolerance", type=int, default=3)
    parser.add_argument("--goal-tolerance", type=int, default=0)
    parser.add_argument("--max-waypoints", type=int, default=200)
    parser.add_argument("--max-replans", type=int, default=2)
    parser.add_argument("--fallback-escape-steps", type=int, default=20)
    parser.add_argument("--max-fallback-events", type=int, default=20)
    parser.add_argument("--max-fallback-steps", type=int, default=2000)
    parser.add_argument("--progress-reset-min", type=int, default=3)
    parser.add_argument("--no-replan-on-fail", action="store_true")
    parser.add_argument("--no-emergency-fallback", action="store_true")
    parser.add_argument("--disable-jps", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = select_device(args.device)
    grid = load_txt_map(args.map)
    start = parse_coord(args.start)
    goal = parse_coord(args.goal)
    enemies = [parse_enemy(x) for x in args.enemy]

    validate_endpoint(grid, start, "start")
    validate_endpoint(grid, goal, "goal")

    env_cfg = EnvConfig()
    planner_cfg = PlannerConfig()
    use_jps = not args.disable_jps

    initial_plan = build_plan(
        grid=grid,
        current=start,
        goal=goal,
        env_cfg=env_cfg,
        planner_cfg=planner_cfg,
        waypoint_interval=args.waypoint_interval,
        use_jps=use_jps,
        waypoint_max_chebyshev=args.waypoint_max_chebyshev,
    )
    if not initial_plan.path:
        print("global_plan_success: false")
        print("reason: no path from start to goal")
        return

    model = load_policy(args.model, device)
    pos = start
    full_trace = [start]
    plans_used = [initial_plan.planner]
    reached_count = 0
    failed_segments = 0
    total_replans = 0
    trap_replans = 0
    visibility_refreshes = 0
    emergency_fallback_used = False
    emergency_fallback_steps = 0
    emergency_fallback_events = 0
    failed_waypoint: Coord | None = None
    failed_pos: Coord | None = None
    last_fail_reason = ""
    fallback_reason = ""
    best_goal_distance = manhattan(pos, goal)

    waypoint_limit = max(1, args.max_waypoints)
    waypoints = make_waypoints(initial_plan, pos, goal, args.waypoint_tolerance, waypoint_limit)

    if args.verbose:
        print("=== initial global plan ===")
        print(f"planner: {initial_plan.planner}")
        print(f"path_steps: {len(initial_plan.path) - 1}")
        print(f"jump_points: {summarize_points(initial_plan.jump_points)}")
        print(f"waypoints: {summarize_points(waypoints)}")
        print(f"waypoint_max_chebyshev: {args.waypoint_max_chebyshev}")
        print("=== local execution ===")

    while waypoints and reached_count < args.max_waypoints:
        if args.waypoint_max_chebyshev > 0 and chebyshev(pos, waypoints[0]) > args.waypoint_max_chebyshev:
            refresh = build_plan(
                grid=grid,
                current=pos,
                goal=goal,
                env_cfg=env_cfg,
                planner_cfg=planner_cfg,
                waypoint_interval=args.waypoint_interval,
                use_jps=use_jps,
                waypoint_max_chebyshev=args.waypoint_max_chebyshev,
            )
            plans_used.append(refresh.planner)
            visibility_refreshes += 1
            if not refresh.path:
                failed_segments += 1
                failed_waypoint = waypoints[0]
                failed_pos = pos
                last_fail_reason = "visibility_refresh_failed"
                break
            remaining = max(1, args.max_waypoints - reached_count)
            waypoints = make_waypoints(refresh, pos, goal, args.waypoint_tolerance, remaining)
            if args.verbose:
                print(
                    f"visibility_refresh {visibility_refreshes}: from={pos} to={goal} "
                    f"path_steps={len(refresh.path) - 1} next_waypoints={summarize_points(waypoints)}"
                )

        waypoint = waypoints.pop(0)
        tolerance = args.goal_tolerance if waypoint == goal else args.waypoint_tolerance
        segment_start = pos
        pos, trace, reached, reason = run_segment(
            grid=grid,
            model=model,
            device=device,
            start=pos,
            waypoint=waypoint,
            enemies=enemies,
            env_cfg=env_cfg,
            max_steps=args.segment_steps,
            tolerance=tolerance,
        )
        full_trace.extend(trace[1:])

        if args.verbose:
            print(
                f"segment {reached_count + failed_segments + 1}: "
                f"start={segment_start} target={waypoint} end={pos} "
                f"steps={len(trace) - 1} reached={reached} reason={reason}"
            )

        if reached:
            reached_count += 1
            current_goal_distance = manhattan(pos, goal)
            made_real_progress = current_goal_distance <= best_goal_distance - args.progress_reset_min
            if made_real_progress:
                best_goal_distance = current_goal_distance
                trap_replans = 0
            elif args.verbose and trap_replans > 0:
                print(
                    f"minor_waypoint_reached_without_progress: "
                    f"goal_dist={current_goal_distance}, best_goal_dist={best_goal_distance}, "
                    f"trap_replans_kept={trap_replans}"
                )
            if manhattan(pos, goal) <= args.goal_tolerance:
                break
            continue

        failed_segments += 1
        failed_waypoint = waypoint
        failed_pos = pos
        last_fail_reason = reason

        if args.no_replan_on_fail:
            break

        if trap_replans >= args.max_replans:
            if args.verbose:
                print(
                    f"max_replans_reached: trap_replans={trap_replans}, "
                    "using geometric escape, then returning to local executor"
                )
            if args.no_emergency_fallback:
                break
            if emergency_fallback_events >= args.max_fallback_events:
                last_fail_reason = "max_fallback_events_reached"
                break
            remaining_fallback_steps = args.max_fallback_steps - emergency_fallback_steps
            if remaining_fallback_steps <= 0:
                last_fail_reason = "max_fallback_steps_reached"
                break

            fallback_start = pos
            chunk_steps = min(args.fallback_escape_steps, remaining_fallback_steps)
            pos, trace, fallback_success, fallback_reason = follow_geometric_path(
                grid=grid,
                start=pos,
                goal=goal,
                env_cfg=env_cfg,
                planner_cfg=planner_cfg,
                waypoint_interval=args.waypoint_interval,
                tolerance=args.goal_tolerance,
                max_steps=chunk_steps,
                use_jps=use_jps,
            )
            chunk_used = len(trace) - 1
            emergency_fallback_used = True
            emergency_fallback_events += 1
            emergency_fallback_steps += chunk_used
            full_trace.extend(trace[1:])
            if args.verbose:
                print(
                    "geometric_escape: "
                    f"start={fallback_start} end={pos} steps={chunk_used} "
                    f"success={fallback_success} reason={fallback_reason}"
                )

            if chunk_used <= 0:
                break
            if manhattan(pos, goal) <= args.goal_tolerance:
                break

            refresh = build_plan(
                grid=grid,
                current=pos,
                goal=goal,
                env_cfg=env_cfg,
                planner_cfg=planner_cfg,
                waypoint_interval=args.waypoint_interval,
                use_jps=use_jps,
                waypoint_max_chebyshev=args.waypoint_max_chebyshev,
            )
            plans_used.append(refresh.planner)
            visibility_refreshes += 1
            if not refresh.path:
                last_fail_reason = "post_fallback_plan_failed"
                break
            remaining = max(1, args.max_waypoints - reached_count)
            waypoints = make_waypoints(refresh, pos, goal, args.waypoint_tolerance, remaining)
            current_goal_distance = manhattan(pos, goal)
            if current_goal_distance < best_goal_distance:
                best_goal_distance = current_goal_distance
            trap_replans = 0
            if args.verbose:
                print(
                    f"return_to_model: from={pos} to={goal} "
                    f"path_steps={len(refresh.path) - 1} next_waypoints={summarize_points(waypoints)}"
                )
            continue

            break
        if manhattan(pos, goal) <= args.goal_tolerance:
            break

        trap_replans += 1
        total_replans += 1
        replan_interval = max(4, args.waypoint_interval // (trap_replans + 1))
        replan = build_plan(
            grid=grid,
            current=pos,
            goal=goal,
            env_cfg=env_cfg,
            planner_cfg=planner_cfg,
            waypoint_interval=replan_interval,
            use_jps=use_jps,
            waypoint_max_chebyshev=args.waypoint_max_chebyshev,
        )
        plans_used.append(replan.planner)
        if not replan.path:
            last_fail_reason = "replan_failed"
            if args.verbose:
                print(f"replan {total_replans}: failed from {pos} to {goal}")
            break

        remaining = max(1, args.max_waypoints - reached_count)
        waypoints = make_waypoints(replan, pos, goal, args.waypoint_tolerance, remaining)
        if args.verbose:
            print(
                f"replan {total_replans}: trap_replans={trap_replans} "
                f"planner={replan.planner} from={pos} to={goal} "
                f"path_steps={len(replan.path) - 1} next_waypoints={summarize_points(waypoints)}"
            )

    success = manhattan(pos, goal) <= args.goal_tolerance

    executed_steps = len(full_trace) - 1
    reference_steps = max(1, len(initial_plan.path) - 1)
    path_efficiency = executed_steps / reference_steps

    print(f"device: {device}")
    print(f"map_size: {grid.shape[0]} x {grid.shape[1]}")
    print(f"global_planner_initial: {initial_plan.planner}")
    print(f"global_path_steps_initial: {len(initial_plan.path) - 1}")
    print(f"global_jump_points_initial: {len(initial_plan.jump_points)}")
    print(f"planned_waypoints_initial: {len(initial_plan.waypoints)}")
    print(f"plans_used: {','.join(plans_used)}")
    print(f"replans: {total_replans}")
    print(f"max_replans_per_trap: {args.max_replans}")
    print(f"best_goal_distance: {best_goal_distance}")
    print(f"visibility_refreshes: {visibility_refreshes}")
    print(f"emergency_fallback_used: {emergency_fallback_used}")
    print(f"emergency_fallback_events: {emergency_fallback_events}")
    print(f"emergency_fallback_steps: {emergency_fallback_steps}")
    if emergency_fallback_used:
        print(f"emergency_fallback_reason: {fallback_reason}")
    print(f"reached_waypoints: {reached_count}")
    print(f"failed_segments: {failed_segments}")
    if failed_waypoint is not None:
        print(f"failed_waypoint: {failed_waypoint}")
        print(f"failed_pos: {failed_pos}")
        failed_distance = manhattan(failed_pos, failed_waypoint) if failed_pos is not None else -1
        print(f"failed_distance_at_failure: {failed_distance}")
        print(f"failed_reason: {last_fail_reason}")
    print(f"local_executor_success: {success and not emergency_fallback_used}")
    print(f"final_pos: {pos}")
    print(f"goal: {goal}")
    print(f"success: {success}")
    print(f"executed_steps: {executed_steps}")
    print(f"path_efficiency: {path_efficiency:.3f}")


if __name__ == "__main__":
    main()
