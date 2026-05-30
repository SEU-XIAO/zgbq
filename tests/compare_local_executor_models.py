from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battlefield_rl.config import EnvConfig, TrainingSceneConfig
from battlefield_rl.env import EnemySpec, TacticalBattlefieldEnv
from battlefield_rl.map import BattlefieldMap, LocalSceneSampler, TrainingScene, load_txt_map
from interfaces.config import DEFAULT_MAP_PATH
from scripts.eval_hierarchical_executor import coord_to_list, enemy_to_dict, load_policy, select_device


Coord = tuple[int, int]


@dataclass(frozen=True)
class FrozenScene:
    grid: BattlefieldMap
    start: Coord
    goal: Coord
    enemies: list[EnemySpec]
    source: str
    enemy_case: str


def parse_model_arg(text: str) -> tuple[str, Path]:
    if "=" in text:
        name, path = text.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(text)
    return path.stem, path


def resolve_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def freeze_scene(scene: TrainingScene) -> FrozenScene:
    return FrozenScene(
        grid=BattlefieldMap(heights=scene.grid.heights.copy(), types=scene.grid.types.copy()),
        start=scene.start,
        goal=scene.goal,
        enemies=list(scene.enemies),
        source=scene.source,
        enemy_case=scene.enemy_case,
    )


def build_scenes(
    *,
    base_grid: BattlefieldMap,
    cases: int,
    stage: int,
    seed: int,
) -> list[FrozenScene]:
    random.seed(seed)
    np.random.seed(seed)
    cfg = TrainingSceneConfig(enabled=True)
    sampler = LocalSceneSampler(base_grid, cfg, EnvConfig().window_size)
    return [freeze_scene(sampler.sample(stage=stage)) for _ in range(cases)]


@torch.no_grad()
def greedy_action(model, obs: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
    from battlefield_rl.rl.network import masked_q_values

    x = torch.from_numpy(obs).unsqueeze(0).float().to(device)
    m = torch.from_numpy(mask).unsqueeze(0).float().to(device)
    q = masked_q_values(model(x), m)
    return int(torch.argmax(q, dim=1).item())


def visible_count(path: Sequence[Coord], visible_flags: Sequence[bool]) -> tuple[int, float]:
    count = sum(1 for item in visible_flags if item)
    return count, count / max(1, len(path))


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


def run_local_episode(
    *,
    model,
    device: torch.device,
    scene: FrozenScene,
    max_steps: int,
    detect_oscillation: bool,
) -> dict[str, Any]:
    env_cfg = EnvConfig(timeout_steps=max_steps, no_progress_limit=max_steps + 1)
    env = TacticalBattlefieldEnv(scene.grid, env_cfg)
    obs = env.reset(
        start=scene.start,
        goal=scene.goal,
        enemies=scene.enemies,
        waypoints=[scene.goal],
        threat_scale=1.0,
    )

    path = [scene.start]
    visible_flags: list[bool] = []
    rewards: list[float] = []
    recent: deque[Coord] = deque(maxlen=12)
    reason = "step_limit"

    for _ in range(max_steps):
        action = greedy_action(model, obs, env.action_mask(), device)
        step = env.step(action)
        obs = step.observation
        path.append(env.pos)
        rewards.append(float(step.reward))
        visible_flags.append(bool(step.info.get("visible", False)))
        recent.append(env.pos)

        if env.pos == scene.goal:
            reason = "reached"
            break
        if detect_oscillation and len(recent) == recent.maxlen and len(set(recent)) <= 3:
            reason = "oscillation"
            break
        if step.done:
            if step.info.get("timeout"):
                reason = "timeout"
            elif step.info.get("stuck"):
                reason = "stuck"
            else:
                reason = "done"
            break

    visible_steps, visible_ratio = visible_count(path, visible_flags)
    success = path[-1] == scene.goal
    manhattan_distance = abs(path[-1][0] - scene.goal[0]) + abs(path[-1][1] - scene.goal[1])

    return {
        "success": bool(success),
        "reason": reason,
        "steps": max(0, len(path) - 1),
        "final_pos": coord_to_list(path[-1]),
        "goal_distance_manhattan": int(manhattan_distance),
        "total_reward": float(sum(rewards)),
        "avg_reward": float(sum(rewards) / max(1, len(rewards))),
        "visible_steps": visible_steps,
        "visible_ratio": visible_ratio,
        **repeat_stats(path),
        "path_sample": {
            "head": [coord_to_list(point) for point in path[:16]],
            "tail": [coord_to_list(point) for point in path[-16:]],
        },
    }


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["success"]]

    def avg(key: str, source: Sequence[dict[str, Any]] = rows) -> float:
        if not source:
            return 0.0
        return sum(float(row.get(key, 0.0)) for row in source) / len(source)

    return {
        "cases": len(rows),
        "success_rate": len(successes) / max(1, len(rows)),
        "avg_steps_success": avg("steps", successes),
        "avg_goal_distance": avg("goal_distance_manhattan"),
        "avg_total_reward": avg("total_reward"),
        "avg_visible_ratio": avg("visible_ratio"),
        "avg_repeat_ratio": avg("repeat_ratio"),
        "failure_reasons": dict(Counter(str(row["reason"]) for row in rows if not row["success"])),
    }


def compare_models(
    *,
    map_path: str,
    model_specs: Sequence[tuple[str, Path]],
    cases: int,
    stage: int,
    seed: int,
    max_steps: int,
    detect_oscillation: bool,
) -> dict[str, Any]:
    base_grid = load_txt_map(map_path)
    scenes = build_scenes(base_grid=base_grid, cases=cases, stage=stage, seed=seed)
    device = select_device()

    models = {
        name: {
            "path": resolve_path(path),
            "model": load_policy(resolve_path(path), device),
        }
        for name, path in model_specs
    }

    started = time.perf_counter()
    case_rows: list[dict[str, Any]] = []
    per_model: dict[str, list[dict[str, Any]]] = {name: [] for name in models}

    for idx, scene in enumerate(scenes):
        case_result: dict[str, Any] = {
            "case_index": idx,
            "start": coord_to_list(scene.start),
            "goal": coord_to_list(scene.goal),
            "source": scene.source,
            "enemy_case": scene.enemy_case,
            "enemies": [enemy_to_dict(enemy) for enemy in scene.enemies],
            "models": {},
        }
        for name, model_info in models.items():
            result = run_local_episode(
                model=model_info["model"],
                device=device,
                scene=scene,
                max_steps=max_steps,
                detect_oscillation=detect_oscillation,
            )
            case_result["models"][name] = result
            per_model[name].append(result)
        case_rows.append(case_result)

    return {
        "map": map_path,
        "stage": stage,
        "seed": seed,
        "cases": cases,
        "max_steps": max_steps,
        "device": str(device),
        "models": {name: str(info["path"]) for name, info in models.items()},
        "runtime_sec": time.perf_counter() - started,
        "summary": {name: aggregate(rows) for name, rows in per_model.items()},
        "case_results": case_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare local executor ability of model checkpoints")
    parser.add_argument("--map", default=DEFAULT_MAP_PATH, help="txt map path")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="model spec as name=path or path; repeatable",
    )
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--no-oscillation-detect", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_args = args.model or ["5800=episode_5800.pt", "8000=episode_8000.pt"]
    model_specs = [parse_model_arg(item) for item in model_args]

    report = compare_models(
        map_path=args.map,
        model_specs=model_specs,
        cases=args.cases,
        stage=args.stage,
        seed=args.seed,
        max_steps=args.max_steps,
        detect_oscillation=not args.no_oscillation_detect,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = out_dir / f"compare_local_executor_models_{timestamp}.json"
    report["output_path"] = str(output_path)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
