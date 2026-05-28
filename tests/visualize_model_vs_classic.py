from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import sys
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battlefield_rl.config import EnvConfig, PlannerConfig
from battlefield_rl.env import ACTIONS_8, EnemySpec, TacticalBattlefieldEnv
from battlefield_rl.env.tactical_env import angle_deg, angle_diff, bresenham_line
from battlefield_rl.map import BattlefieldMap, load_txt_map
from battlefield_rl.planner import plan_global_path
from battlefield_rl.rl.network import TacticalD3QN, masked_q_values
from scripts.eval_hierarchical_executor import (
    WAYPOINT_INTERVAL,
    WAYPOINT_MAX_CHEBYSHEV,
    WAYPOINT_TOLERANCE,
    build_plan,
    coord_to_list,
    coords_to_list,
    enemy_to_dict,
    load_policy,
    make_waypoints,
    manhattan,
    select_device,
)


Coord = tuple[int, int]


class RecencyMaskedEnv(TacticalBattlefieldEnv):
    OSC_WINDOW = 10
    OSC_UNIQUE_LIMIT = 3
    OSC_MIN_STEPS = 8
    RECENT_BLOCK = 3

    def __init__(self, grid: BattlefieldMap, config: EnvConfig):
        super().__init__(grid, config)
        self.recent_positions: deque[Coord] = deque(maxlen=self.OSC_WINDOW)
        self.oscillation_active = False
        self.mask_interventions = 0

    def reset(self, start, goal, enemies, waypoints, threat_scale=1.0):
        obs = super().reset(start, goal, enemies, waypoints, threat_scale)
        self.recent_positions.clear()
        self.recent_positions.append(self.pos)
        self.oscillation_active = False
        self.mask_interventions = 0
        return obs

    def step(self, action: int):
        result = super().step(action)
        self.recent_positions.append(self.pos)
        if len(self.recent_positions) >= self.OSC_MIN_STEPS:
            recent = list(self.recent_positions)[-self.OSC_WINDOW :]
            self.oscillation_active = len(set(recent)) <= self.OSC_UNIQUE_LIMIT
        else:
            self.oscillation_active = False
        return result

    def action_mask(self) -> np.ndarray:
        mask = super().action_mask()
        if not self.oscillation_active:
            return mask
        recent_set = set(list(self.recent_positions)[-self.RECENT_BLOCK :])
        recent_set.discard(self.pos)
        if not recent_set:
            return mask

        original = mask.copy()
        blocked = False
        for i, (dr, dc) in enumerate(ACTIONS_8):
            if mask[i] <= 0.5:
                continue
            nxt = (self.pos[0] + dr, self.pos[1] + dc)
            if nxt in recent_set:
                mask[i] = 0.0
                blocked = True
        if blocked and np.any(mask > 0.5):
            self.mask_interventions += 1
            return mask
        return original


def parse_coord(text: str) -> Coord:
    row, col = text.split(",")
    return int(row), int(col)


def passable_cells(grid: BattlefieldMap) -> list[Coord]:
    rows, cols = grid.shape
    return [
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if not grid.is_static_blocked(row, col)
    ]


def bfs_path(grid: BattlefieldMap, start: Coord, goal: Coord) -> list[Coord]:
    queue: deque[Coord] = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for dr, dc in ACTIONS_8:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in parent or not grid.is_passable(cur, nxt, 0):
                continue
            parent[nxt] = cur
            queue.append(nxt)
    return reconstruct(parent, goal)


def astar_path(grid: BattlefieldMap, start: Coord, goal: Coord) -> list[Coord]:
    def h(point: Coord) -> float:
        return math.hypot(point[0] - goal[0], point[1] - goal[1])

    heap: list[tuple[float, int, Coord]] = [(h(start), 0, start)]
    counter = 0
    parent: dict[Coord, Coord | None] = {start: None}
    g_score: dict[Coord, float] = {start: 0.0}
    closed: set[Coord] = set()
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        if cur == goal:
            break
        closed.add(cur)
        for dr, dc in ACTIONS_8:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not grid.is_passable(cur, nxt, 0):
                continue
            step = math.hypot(dr, dc)
            new_g = g_score[cur] + step
            if new_g < g_score.get(nxt, float("inf")):
                parent[nxt] = cur
                g_score[nxt] = new_g
                counter += 1
                heapq.heappush(heap, (new_g + h(nxt), counter, nxt))
    return reconstruct(parent, goal)


def reconstruct(parent: dict[Coord, Coord | None], goal: Coord) -> list[Coord]:
    if goal not in parent:
        return []
    path: list[Coord] = []
    node: Coord | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def point_visible(grid: BattlefieldMap, point: Coord, enemy: EnemySpec) -> bool:
    src = (enemy.row, enemy.col)
    dist = math.hypot(point[0] - src[0], point[1] - src[1])
    if dist > enemy.max_range:
        return False
    if angle_diff(angle_deg(src, point), enemy.facing_deg) > enemy.fov_deg * 0.5:
        return False
    for row, col in bresenham_line(src[0], src[1], point[0], point[1])[1:-1]:
        if grid.is_static_blocked(row, col):
            return False
    return True


def visible_stats(grid: BattlefieldMap, path: Sequence[Coord], enemies: Sequence[EnemySpec]) -> dict[str, Any]:
    visible = [
        point
        for point in path
        if any(point_visible(grid, point, enemy) for enemy in enemies)
    ]
    return {
        "visible_steps": len(visible),
        "visible_ratio": len(visible) / max(1, len(path)),
        "visible_points": coords_to_list(visible),
    }


def angle_from_to(src: Coord, dst: Coord) -> float:
    return (math.degrees(math.atan2(-(dst[0] - src[0]), dst[1] - src[1])) + 360.0) % 360.0


def place_enemies_on_classic_path(
    grid: BattlefieldMap,
    path: Sequence[Coord],
    *,
    count: int,
    enemy_range: int,
    enemy_fov: int,
    rng: random.Random,
) -> list[EnemySpec]:
    if len(path) < 20:
        return []
    enemies: list[EnemySpec] = []
    path_set = set(path)
    anchors = [int(len(path) * frac) for frac in np.linspace(0.28, 0.72, count)]
    for anchor in anchors:
        ref = path[min(max(anchor, 1), len(path) - 2)]
        look_at = path[min(anchor + 4, len(path) - 1)]
        candidates: list[tuple[float, Coord]] = []
        for dr in range(-16, 17):
            for dc in range(-16, 17):
                if dr == 0 and dc == 0:
                    continue
                pos = (ref[0] + dr, ref[1] + dc)
                if not grid.in_bounds(pos[0], pos[1]) or grid.is_static_blocked(pos[0], pos[1]):
                    continue
                if pos in path_set:
                    continue
                dist = math.hypot(dr, dc)
                if 5 <= dist <= enemy_range * 0.75:
                    candidates.append((abs(dist - enemy_range * 0.35), pos))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        pos = rng.choice(candidates[: min(12, len(candidates))])[1]
        enemies.append(
            EnemySpec(
                row=pos[0],
                col=pos[1],
                facing_deg=(angle_from_to(pos, look_at) + rng.uniform(-12.0, 12.0)) % 360.0,
                fov_deg=float(enemy_fov),
                max_range=int(enemy_range),
            )
        )
    return enemies


@torch.no_grad()
def greedy_action(model: TacticalD3QN, obs: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
    x = torch.from_numpy(obs).unsqueeze(0).float().to(device)
    m = torch.from_numpy(mask).unsqueeze(0).float().to(device)
    q = masked_q_values(model(x), m)
    return int(torch.argmax(q, dim=1).item())


def run_recency_model(
    *,
    grid: BattlefieldMap,
    model: TacticalD3QN,
    device: torch.device,
    start: Coord,
    goal: Coord,
    enemies: Sequence[EnemySpec],
    max_segment_steps: int,
    max_total_steps: int,
) -> dict[str, Any]:
    env_cfg = EnvConfig(timeout_steps=max_segment_steps, no_progress_limit=max_segment_steps + 1)
    planner_cfg = PlannerConfig()
    plan = build_plan(
        grid=grid,
        current=start,
        goal=goal,
        env_cfg=env_cfg,
        planner_cfg=planner_cfg,
        waypoint_interval=WAYPOINT_INTERVAL,
        waypoint_max_chebyshev=WAYPOINT_MAX_CHEBYSHEV,
    )
    if not plan.path:
        return {"success": False, "reason": "no_global_path", "path": [start], "interventions": 0}

    pos = start
    full_path = [start]
    waypoints = make_waypoints(plan, start, goal, WAYPOINT_TOLERANCE, 500)
    interventions = 0
    segments: list[dict[str, Any]] = []
    reason = ""

    for waypoint in waypoints:
        env = RecencyMaskedEnv(grid, env_cfg)
        obs = env.reset(start=pos, goal=waypoint, enemies=enemies, waypoints=[waypoint], threat_scale=1.0)
        segment_start = pos
        reached = False
        for _ in range(max_segment_steps):
            if manhattan(env.pos, waypoint) <= (0 if waypoint == goal else WAYPOINT_TOLERANCE):
                reached = True
                break
            action = greedy_action(model, obs, env.action_mask(), device)
            step = env.step(action)
            obs = step.observation
            pos = env.pos
            full_path.append(pos)
            if len(full_path) - 1 >= max_total_steps:
                reason = "max_total_steps"
                break
            if manhattan(env.pos, waypoint) <= (0 if waypoint == goal else WAYPOINT_TOLERANCE):
                reached = True
                break
            if step.done:
                reason = "env_done"
                break
        interventions += env.mask_interventions
        segments.append(
            {
                "start": coord_to_list(segment_start),
                "target": coord_to_list(waypoint),
                "end": coord_to_list(pos),
                "reached": reached,
                "interventions": env.mask_interventions,
            }
        )
        if pos == goal:
            reason = "reached"
            break
        if not reached or reason == "max_total_steps":
            reason = reason or "segment_failed"
            break

    return {
        "success": pos == goal,
        "reason": "reached" if pos == goal else reason,
        "path": full_path,
        "interventions": interventions,
        "segments": segments,
        "initial_global_steps": len(plan.path) - 1,
    }


def summarize_route(
    name: str,
    grid: BattlefieldMap,
    path: Sequence[Coord],
    enemies: Sequence[EnemySpec],
    success: bool,
    reason: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counter = Counter(path)
    out = {
        "name": name,
        "success": bool(success),
        "reason": reason,
        "steps": max(0, len(path) - 1),
        "repeat_ratio": (len(path) - len(counter)) / max(1, len(path)),
        "max_repeat": max(counter.values()) if counter else 0,
        "path": coords_to_list(path),
        **visible_stats(grid, path, enemies),
    }
    if extra:
        out.update(extra)
    return out


def sample_pair(rng: random.Random, grid: BattlefieldMap, min_steps: int, max_steps: int) -> tuple[Coord, Coord] | None:
    cells = passable_cells(grid)
    for _ in range(3000):
        start = rng.choice(cells)
        goal = rng.choice(cells)
        if start == goal:
            continue
        path = astar_path(grid, start, goal)
        steps = len(path) - 1
        if path and min_steps <= steps <= max_steps:
            return start, goal
    return None


def score_case(routes: dict[str, dict[str, Any]]) -> float:
    model = routes["model"]
    if not model["success"]:
        return -999.0
    classic_visible_steps = min(routes["bfs"]["visible_steps"], routes["astar"]["visible_steps"])
    step_gain = classic_visible_steps - model["visible_steps"]
    classic_vis = min(routes["bfs"]["visible_ratio"], routes["astar"]["visible_ratio"])
    ratio_gain = classic_vis - model["visible_ratio"]
    step_penalty = max(0.0, model["steps"] - routes["astar"]["steps"]) / max(1, routes["astar"]["steps"])
    if classic_visible_steps < 4:
        return -999.0
    return step_gain * 5.0 + ratio_gain * 8.0 - step_penalty + (0.2 if model.get("interventions", 0) > 0 else 0.0)


def find_top_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    grid = load_txt_map(args.map)
    device = select_device()
    model = load_policy(args.model, device)
    top_cases: list[dict[str, Any]] = []
    seen_pairs: set[tuple[Coord, Coord]] = set()

    for _ in range(args.search_cases):
        pair = sample_pair(rng, grid, args.min_steps, args.max_steps)
        if pair is None:
            continue
        start, goal = pair
        if (start, goal) in seen_pairs:
            continue
        seen_pairs.add((start, goal))
        astar = astar_path(grid, start, goal)
        bfs = bfs_path(grid, start, goal)
        enemies = place_enemies_on_classic_path(
            grid,
            astar or bfs,
            count=args.enemies,
            enemy_range=args.enemy_range,
            enemy_fov=args.enemy_fov,
            rng=rng,
        )
        if not enemies:
            continue
        model_run = run_recency_model(
            grid=grid,
            model=model,
            device=device,
            start=start,
            goal=goal,
            enemies=enemies,
            max_segment_steps=args.segment_steps,
            max_total_steps=args.max_total_steps,
        )
        routes = {
            "bfs": summarize_route("BFS", grid, bfs, enemies, bool(bfs and bfs[-1] == goal)),
            "astar": summarize_route("A*", grid, astar, enemies, bool(astar and astar[-1] == goal)),
            "model": summarize_route(
                "Model6000+RecencyMask",
                grid,
                model_run["path"],
                enemies,
                bool(model_run["success"]),
                str(model_run.get("reason", "")),
                {
                    "interventions": model_run.get("interventions", 0),
                    "initial_global_steps": model_run.get("initial_global_steps", 0),
                },
            ),
        }
        score = score_case(routes)
        if score <= -900.0:
            continue
        top_cases.append(
            {
                "score": score,
                "start": start,
                "goal": goal,
                "enemies": enemies,
                "routes": routes,
            }
        )
        top_cases.sort(key=lambda item: item["score"], reverse=True)
        del top_cases[args.top_k :]

    if not top_cases:
        raise RuntimeError("failed to find a visualizable case")
    return top_cases


def crop_bounds(paths: Sequence[Sequence[Coord]], enemies: Sequence[EnemySpec], grid: BattlefieldMap, margin: int) -> tuple[int, int, int, int]:
    rows = [p[0] for path in paths for p in path] + [e.row for e in enemies]
    cols = [p[1] for path in paths for p in path] + [e.col for e in enemies]
    r1 = max(0, min(rows) - margin)
    r2 = min(grid.shape[0] - 1, max(rows) + margin)
    c1 = max(0, min(cols) - margin)
    c2 = min(grid.shape[1] - 1, max(cols) + margin)
    return r1, r2, c1, c2


def draw_report(case: dict[str, Any], grid: BattlefieldMap, output_png: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    route_paths = [
        [(int(p[0]), int(p[1])) for p in route["path"]]
        for route in case["routes"].values()
    ]
    enemies = case["enemies"]
    r1, r2, c1, c2 = crop_bounds(route_paths, enemies, grid, margin=18)
    visible_image = np.zeros((r2 - r1 + 1, c2 - c1 + 1), dtype=np.uint8)
    for row in range(r1, r2 + 1):
        for col in range(c1, c2 + 1):
            rr = row - r1
            cc = col - c1
            if grid.is_static_blocked(row, col):
                visible_image[rr, cc] = 2
            elif any(point_visible(grid, (row, col), enemy) for enemy in enemies):
                visible_image[rr, cc] = 1
            else:
                visible_image[rr, cc] = 0

    cmap = ListedColormap([
        "#eef8ee",
        "#ffb3a7",
        "#222222",
    ])

    rows, cols = visible_image.shape
    fig_w = max(11.0, cols * 0.18)
    fig_h = max(9.0, rows * 0.18)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    ax.imshow(visible_image, cmap=cmap, vmin=0, vmax=2, origin="upper", interpolation="none")
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#9a9a9a", linewidth=0.3, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)

    colors = {"bfs": "#2f80ed", "astar": "#27ae60", "model": "#e74c3c"}
    labels = {"bfs": "BFS", "astar": "A*", "model": "Model 6000 + recency mask"}
    for key, route in case["routes"].items():
        path = [(int(p[0]), int(p[1])) for p in route["path"]]
        if not path:
            continue
        xs = [p[1] - c1 for p in path]
        ys = [p[0] - r1 for p in path]
        ax.plot(xs, ys, color=colors[key], linewidth=2.2, label=f"{labels[key]} vis={route['visible_ratio']:.2f}")
        visible_points = [(int(p[0]), int(p[1])) for p in route.get("visible_points", [])]
        if visible_points:
            ax.scatter(
                [p[1] - c1 for p in visible_points],
                [p[0] - r1 for p in visible_points],
                s=12,
                color=colors[key],
                marker="x",
                alpha=0.8,
            )

    start = case["start"]
    goal = case["goal"]
    ax.scatter([start[1] - c1], [start[0] - r1], s=80, c="#00aa55", marker="o", label="Start")
    ax.scatter([goal[1] - c1], [goal[0] - r1], s=100, c="#111111", marker="*", label="Goal")

    for idx, enemy in enumerate(enemies, start=1):
        ex = enemy.col - c1
        ey = enemy.row - r1
        ax.scatter([ex], [ey], s=90, c="#8e44ad", marker="^")
        ax.text(ex + 1, ey + 1, f"E{idx}", color="#8e44ad", fontsize=9)

    summary_lines = []
    for key in ["bfs", "astar", "model"]:
        r = case["routes"][key]
        extra = f", mask={r.get('interventions', 0)}" if key == "model" else ""
        summary_lines.append(
            f"{labels[key]}: success={r['success']} steps={r['steps']} visible={r['visible_ratio']:.3f}{extra}"
        )
    legend_items = [
        Patch(facecolor="#eef8ee", edgecolor="#9a9a9a", label="Not visible passable cell"),
        Patch(facecolor="#ffb3a7", edgecolor="#9a9a9a", label="Enemy-visible passable cell"),
        Patch(facecolor="#222222", edgecolor="#9a9a9a", label="Obstacle"),
        Line2D([0], [0], color=colors["bfs"], lw=2.2, label="BFS path"),
        Line2D([0], [0], color=colors["astar"], lw=2.2, label="A* path"),
        Line2D([0], [0], color=colors["model"], lw=2.8, label="Model path"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#8e44ad", markersize=10, label="Enemy"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.92)
    ax.set_title("Path comparison with exact line-of-sight visibility cells")
    ax.text(
        0.01,
        0.01,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#cccccc"},
    )
    ax.set_xlim(0, c2 - c1)
    ax.set_ylim(r2 - r1, 0)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize model 6000 with recency masking vs BFS/A*")
    p.add_argument("--map", default="MyPath_Data417.txt")
    p.add_argument("--model", default="episode_6000.pt")
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--search-cases", type=int, default=25)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--min-steps", type=int, default=45)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--enemies", type=int, default=2)
    p.add_argument("--enemy-range", type=int, default=65)
    p.add_argument("--enemy-fov", type=int, default=120)
    p.add_argument("--segment-steps", type=int, default=55)
    p.add_argument("--max-total-steps", type=int, default=260)
    p.add_argument("--output-dir", default="outputs")
    return p


def main() -> None:
    args = build_parser().parse_args()
    grid = load_txt_map(args.map)
    top_cases = find_top_cases(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    outputs = []
    for rank, case in enumerate(top_cases, start=1):
        case_json = {
            "rank": rank,
            "score": case["score"],
            "start": coord_to_list(case["start"]),
            "goal": coord_to_list(case["goal"]),
            "enemies": [enemy_to_dict(enemy) for enemy in case["enemies"]],
            "routes": case["routes"],
        }
        json_path = out_dir / f"model_vs_classic_top{rank}_{timestamp}.json"
        png_path = out_dir / f"model_vs_classic_top{rank}_{timestamp}.png"
        json_path.write_text(json.dumps(case_json, ensure_ascii=False, indent=2), encoding="utf-8")
        draw_report(case, grid, png_path)
        outputs.append(
            {
                "rank": rank,
                "json": str(json_path),
                "png": str(png_path),
                "start": case_json["start"],
                "goal": case_json["goal"],
                "enemies": case_json["enemies"],
                "summary": {
                    key: {
                        "success": route["success"],
                        "steps": route["steps"],
                        "visible_ratio": route["visible_ratio"],
                        "visible_steps": route["visible_steps"],
                        "repeat_ratio": route["repeat_ratio"],
                        "interventions": route.get("interventions", 0),
                    }
                    for key, route in case_json["routes"].items()
                },
            }
        )

    print(json.dumps({"outputs": outputs}, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
