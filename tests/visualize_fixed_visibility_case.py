from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TESTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from battlefield_rl.env import EnemySpec
from battlefield_rl.map import BattlefieldMap, load_txt_map
from interfaces.config import DEFAULT_MAP_PATH
from scripts.eval_hierarchical_executor import coord_to_list, coords_to_list, enemy_to_dict, load_policy, select_device
from visualize_model_vs_classic import (
    astar_path,
    bfs_path,
    point_visible,
    run_recency_model,
    summarize_route,
)


Coord = tuple[int, int]


DEFAULT_START: Coord = (324, 90)
DEFAULT_GOAL: Coord = (384, 9)
DEFAULT_ENEMIES = [
    EnemySpec(row=350, col=52, facing_deg=42.90889928500577, fov_deg=120.0, max_range=75),
    EnemySpec(row=350, col=47, facing_deg=217.24367439719543, fov_deg=120.0, max_range=75),
]


def crop_bounds(
    *,
    paths: list[list[Coord]],
    enemies: list[EnemySpec],
    start: Coord,
    goal: Coord,
    grid: BattlefieldMap,
    margin: int,
) -> tuple[int, int, int, int]:
    rows = [start[0], goal[0]] + [enemy.row for enemy in enemies]
    cols = [start[1], goal[1]] + [enemy.col for enemy in enemies]
    for path in paths:
        rows.extend(point[0] for point in path)
        cols.extend(point[1] for point in path)
    r1 = max(0, min(rows) - margin)
    r2 = min(grid.shape[0] - 1, max(rows) + margin)
    c1 = max(0, min(cols) - margin)
    c2 = min(grid.shape[1] - 1, max(cols) + margin)
    return r1, r2, c1, c2


def visibility_grid(
    grid: BattlefieldMap,
    enemies: list[EnemySpec],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    r1, r2, c1, c2 = bounds
    image = np.zeros((r2 - r1 + 1, c2 - c1 + 1), dtype=np.uint8)
    for row in range(r1, r2 + 1):
        for col in range(c1, c2 + 1):
            rr = row - r1
            cc = col - c1
            if grid.is_static_blocked(row, col):
                image[rr, cc] = 2
            elif any(point_visible(grid, (row, col), enemy) for enemy in enemies):
                image[rr, cc] = 1
            else:
                image[rr, cc] = 0
    return image


def draw_case(
    *,
    grid: BattlefieldMap,
    start: Coord,
    goal: Coord,
    enemies: list[EnemySpec],
    routes: dict[str, dict[str, Any]],
    output_png: Path,
    margin: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    path_by_key = {
        key: [(int(point[0]), int(point[1])) for point in route["path"]]
        for key, route in routes.items()
    }
    bounds = crop_bounds(
        paths=list(path_by_key.values()),
        enemies=enemies,
        start=start,
        goal=goal,
        grid=grid,
        margin=margin,
    )
    r1, r2, c1, c2 = bounds
    image = visibility_grid(grid, enemies, bounds)

    cmap = ListedColormap([
        "#eef8ee",  # passable and not visible
        "#ffb3a7",  # visible by any enemy
        "#222222",  # obstacle
    ])

    rows, cols = image.shape
    fig_w = max(10.0, cols * 0.22)
    fig_h = max(8.0, rows * 0.22)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
    ax.imshow(image, cmap=cmap, vmin=0, vmax=2, origin="upper", interpolation="none")

    # Make every grid cell explicit.
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#9a9a9a", linewidth=0.35, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    colors = {
        "bfs": "#1f77b4",
        "astar": "#2ca02c",
        "model": "#d62728",
    }
    labels = {
        "bfs": "BFS",
        "astar": "A*",
        "model": "Model 4500 + recency/reverse",
    }
    widths = {"bfs": 2.2, "astar": 2.2, "model": 2.8}
    for key in ("bfs", "astar", "model"):
        path = path_by_key[key]
        xs = [point[1] - c1 for point in path]
        ys = [point[0] - r1 for point in path]
        ax.plot(xs, ys, color=colors[key], linewidth=widths[key], alpha=0.95, label=labels[key])
        visible_points = [(int(point[0]), int(point[1])) for point in routes[key].get("visible_points", [])]
        if visible_points:
            ax.scatter(
                [point[1] - c1 for point in visible_points],
                [point[0] - r1 for point in visible_points],
                s=28,
                marker="x",
                color=colors[key],
                linewidths=1.5,
                alpha=0.95,
            )

    ax.scatter([start[1] - c1], [start[0] - r1], s=130, marker="o", color="#00a65a", edgecolor="white", linewidth=1.2)
    ax.scatter([goal[1] - c1], [goal[0] - r1], s=180, marker="*", color="#111111", edgecolor="white", linewidth=1.0)

    for idx, enemy in enumerate(enemies, start=1):
        ex = enemy.col - c1
        ey = enemy.row - r1
        ax.scatter([ex], [ey], s=150, marker="^", color="#7d3c98", edgecolor="white", linewidth=1.0)
        ax.text(ex + 0.8, ey + 0.8, f"E{idx}", color="#7d3c98", fontsize=10, weight="bold")

    summary = []
    for key in ("bfs", "astar", "model"):
        route = routes[key]
        extra = f", debounce={route.get('interventions', 0)}" if key == "model" else ""
        summary.append(
            f"{labels[key]}: steps={route['steps']}, visible={route['visible_steps']} ({route['visible_ratio']:.1%}){extra}"
        )

    ax.text(
        0.012,
        0.012,
        "\n".join(summary),
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.88},
    )

    legend_items = [
        Patch(facecolor="#eef8ee", edgecolor="#9a9a9a", label="Not visible passable cell"),
        Patch(facecolor="#ffb3a7", edgecolor="#9a9a9a", label="Enemy-visible passable cell"),
        Patch(facecolor="#222222", edgecolor="#9a9a9a", label="Obstacle"),
        Line2D([0], [0], color=colors["bfs"], lw=2.2, label="BFS path"),
        Line2D([0], [0], color=colors["astar"], lw=2.2, label="A* path"),
        Line2D([0], [0], color=colors["model"], lw=2.8, label="Model path"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#7d3c98", markersize=10, label="Enemy"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#00a65a", markersize=10, label="Start"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#111111", markersize=13, label="Goal"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9, framealpha=0.92)
    ax.set_title("Fixed case visibility grid: obstacles, enemy-visible cells, and routes")
    ax.set_xlabel(f"col offset from {c1}")
    ax.set_ylabel(f"row offset from {r1}")
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def build_routes(args: argparse.Namespace) -> tuple[BattlefieldMap, dict[str, dict[str, Any]]]:
    grid = load_txt_map(args.map)
    device = select_device()
    model = load_policy(args.model, device)

    bfs = bfs_path(grid, DEFAULT_START, DEFAULT_GOAL)
    astar = astar_path(grid, DEFAULT_START, DEFAULT_GOAL)
    model_run = run_recency_model(
        grid=grid,
        model=model,
        device=device,
        start=DEFAULT_START,
        goal=DEFAULT_GOAL,
        enemies=DEFAULT_ENEMIES,
        max_segment_steps=args.segment_steps,
        max_total_steps=args.max_total_steps,
    )

    routes = {
        "bfs": summarize_route("BFS", grid, bfs, DEFAULT_ENEMIES, bool(bfs and bfs[-1] == DEFAULT_GOAL)),
        "astar": summarize_route("A*", grid, astar, DEFAULT_ENEMIES, bool(astar and astar[-1] == DEFAULT_GOAL)),
        "model": summarize_route(
            "Model4500+RecencyReverse",
            grid,
            model_run["path"],
            DEFAULT_ENEMIES,
            bool(model_run["success"]),
            str(model_run.get("reason", "")),
            {
                "interventions": int(model_run.get("interventions", 0)),
                "initial_global_steps": int(model_run.get("initial_global_steps", 0)),
                "executor_segments": model_run.get("executor_segments", model_run.get("segments", [])),
                "geometric_fallback_events": model_run.get("geometric_fallback_events", []),
                "geometric_fallback_count": int(model_run.get("geometric_fallback_count", 0)),
                "geometric_fallback_steps": int(model_run.get("geometric_fallback_steps", 0)),
                "local_executor_success": bool(model_run.get("local_executor_success", False)),
            },
        ),
    }
    return grid, routes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the fixed visibility comparison case cell by cell")
    parser.add_argument("--map", default=DEFAULT_MAP_PATH)
    parser.add_argument("--model", default="episode_6000.pt")
    parser.add_argument("--segment-steps", type=int, default=55)
    parser.add_argument("--max-total-steps", type=int, default=260)
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    grid, routes = build_routes(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    png_path = output_dir / f"fixed_visibility_grid_{timestamp}.png"
    json_path = output_dir / f"fixed_visibility_grid_{timestamp}.json"

    draw_case(
        grid=grid,
        start=DEFAULT_START,
        goal=DEFAULT_GOAL,
        enemies=DEFAULT_ENEMIES,
        routes=routes,
        output_png=png_path,
        margin=args.margin,
    )

    report = {
        "png": str(png_path),
        "start": coord_to_list(DEFAULT_START),
        "goal": coord_to_list(DEFAULT_GOAL),
        "enemies": [enemy_to_dict(enemy) for enemy in DEFAULT_ENEMIES],
        "routes": routes,
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "png": str(png_path),
        "json": str(json_path),
        "summary": {
            key: {
                "success": route["success"],
                "steps": route["steps"],
                "visible_steps": route["visible_steps"],
                "visible_ratio": route["visible_ratio"],
                "repeat_ratio": route["repeat_ratio"],
                "interventions": route.get("interventions", 0),
            }
            for key, route in routes.items()
        },
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
