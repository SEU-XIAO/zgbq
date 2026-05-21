from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import random
import sys
from typing import Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battlefield_rl.config import (
    AgentConfig,
    CurriculumConfig,
    CurriculumThreshold,
    EnvConfig,
    PlannerConfig,
    TrainingSceneConfig,
)
from battlefield_rl.map import load_txt_map
from battlefield_rl.train import HierarchicalTrainer


def parse_coord(text: str) -> Tuple[int, int]:
    a, b = text.split(",")
    return int(a), int(b)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动回退到 CPU")
        return "cpu"
    return device_arg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 2D local tactical executor on 35x35 mixed scenes")
    parser.add_argument("--map", type=str, required=True)
    parser.add_argument("--start", type=str, default="20,20", help="row,col, used only when scene sampling is disabled")
    parser.add_argument("--goal", type=str, default="200,200", help="row,col, used only when scene sampling is disabled")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--disable-scene-sampler", action="store_true")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)

    grid = load_txt_map(args.map)
    start = parse_coord(args.start)
    goal = parse_coord(args.goal)

    trainer = HierarchicalTrainer(
        grid=grid,
        env_cfg=EnvConfig(),
        planner_cfg=PlannerConfig(),
        agent_cfg=AgentConfig(),
        curriculum_cfg=CurriculumConfig(),
        gate_cfg=CurriculumThreshold(),
        scene_cfg=TrainingSceneConfig(enabled=not args.disable_scene_sampler),
        device=device,
    )

    start_episode = 1
    best_success = -1.0
    if args.resume:
        checkpoint = trainer.agent.load_checkpoint(args.resume)
        trainer.stage = int(checkpoint.get("stage", trainer.stage))
        trainer.stage_episode = int(checkpoint.get("stage_episode", trainer.stage_episode))
        start_episode = int(checkpoint.get("episode", 0)) + 1
        best_success = float(checkpoint.get("best_success", best_success))
        print(f"已加载 checkpoint: {args.resume}")

    print(f"地图尺寸: {grid.shape[0]} x {grid.shape[1]}")
    print(f"训练设备: {device}")
    save_dir = Path(args.save_dir)
    recent_success = deque(maxlen=100)

    end_episode = start_episode + args.episodes - 1
    for ep in range(start_episode, end_episode + 1):
        stats = trainer.run_episode(ep, start, goal, train=True)
        recent_success.append(1.0 if stats.reached_goal else 0.0)
        rolling_success = sum(recent_success) / len(recent_success)

        if ep % 10 == 0 or ep == 1:
            print(
                f"[EP {stats.episode}] stage={stats.stage} steps={stats.steps} "
                f"reward={stats.reward:.3f} goal={stats.reached_goal} timeout={stats.timeout} "
                f"path_eff={stats.path_efficiency:.3f} "
                f"gate_success={stats.rolling_success_rate:.3f} "
                f"gate_timeout={stats.rolling_timeout_rate:.3f} "
                f"gate_eff={stats.rolling_path_efficiency:.3f} "
                f"guide_prob={stats.guide_prob:.3f} "
                f"guided={stats.guided_actions} "
                f"success100={rolling_success:.3f}"
            )

        extra = {
            "episode": ep,
            "stage": trainer.stage,
            "stage_episode": trainer.stage_episode,
            "best_success": max(best_success, rolling_success),
        }
        if args.save_every > 0 and ep % args.save_every == 0:
            trainer.agent.save_checkpoint(save_dir / "latest.pt", extra=extra)
            trainer.agent.save_checkpoint(save_dir / f"episode_{ep}.pt", extra=extra)

        if rolling_success > best_success and len(recent_success) == recent_success.maxlen:
            best_success = rolling_success
            extra["best_success"] = best_success
            trainer.agent.save_checkpoint(save_dir / "best.pt", extra=extra)


if __name__ == "__main__":
    main()
