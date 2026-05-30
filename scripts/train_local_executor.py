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
from interfaces.config import DEFAULT_MAP_PATH
from scripts.config import (
    DEFAULT_DEVICE,
    DEFAULT_EPISODES,
    DEFAULT_SAVE_DIR,
    DEFAULT_SAVE_EVERY,
    DEFAULT_SEED,
)
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
    parser.add_argument("--map", type=str, default=DEFAULT_MAP_PATH, help="地形图文件路径")
    parser.add_argument("--start", type=str, default="20,20", help="起始坐标 row,col（仅禁用场景采样时生效）")
    parser.add_argument("--goal", type=str, default="200,200", help="目标坐标 row,col（仅禁用场景采样时生效）")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="训练总 episode 数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, choices=["auto", "cpu", "cuda"], help="训练设备")
    parser.add_argument("--disable-scene-sampler", action="store_true", help="禁用局部场景采样，使用真实地图")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR, help="模型 checkpoint 输出目录")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="每隔 N 个 episode 保存一次 checkpoint")
    parser.add_argument("--resume", type=str, default="", help="从指定 checkpoint 恢复训练")
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
        trainer.agent.global_step = 0
        trainer.stage = int(checkpoint.get("stage", trainer.stage))
        trainer.stage_episode = int(checkpoint.get("stage_episode", trainer.stage_episode))
        start_episode = int(checkpoint.get("episode", 0)) + 1
        best_success = float(checkpoint.get("best_success", best_success))
        trainer.stage_episode = 0  # 重置阶段内计数，专家引导概率从 start 重新衰减
        print(f"已加载 checkpoint: {args.resume}")

    print(f"地图尺寸: {grid.shape[0]} x {grid.shape[1]}")
    print(f"训练设备: {device}")
    print(
        "训练配置: "
        f"timeout={trainer.env_cfg.timeout_steps}, "
        f"guide_stage1={trainer.curriculum_cfg.stage1_guide_start:.2f}->{trainer.curriculum_cfg.stage1_guide_end:.2f}, "
        f"imitation_weight={trainer.agent.cfg.imitation_loss_weight:.2f}, "
        f"eval_every={trainer.gate_cfg.eval_every_episodes}"
    )
    save_dir = Path(args.save_dir)
    recent_success = deque(maxlen=100)

    end_episode = start_episode + args.episodes - 1
    for ep in range(start_episode, end_episode + 1):
        stats = trainer.run_episode(ep, start, goal, train=True)
        recent_success.append(1.0 if stats.reached_goal else 0.0)
        rolling_success = sum(recent_success) / len(recent_success)
        eval_success = stats.rolling_success_rate

        if ep % 10 == 0 or ep == 1 or stats.eval_ran:
            print(
                f"[EP {stats.episode}] stage={stats.stage} steps={stats.steps} "
                f"reward={stats.reward:.3f} goal={stats.reached_goal} timeout={stats.timeout} "
                f"path_eff={stats.path_efficiency:.3f} "
                f"eval_success={stats.rolling_success_rate:.3f} "
                f"eval_timeout={stats.rolling_timeout_rate:.3f} "
                f"eval_eff={stats.rolling_path_efficiency:.3f} "
                f"guide_prob={stats.guide_prob:.3f} "
                f"guided={stats.guided_actions} "
                f"eval={stats.eval_ran} "
                f"success100={rolling_success:.3f}"
            )

        extra = {
            "episode": ep,
            "stage": trainer.stage,
            "stage_episode": trainer.stage_episode,
            "best_success": max(best_success, eval_success),
        }
        if args.save_every > 0 and ep % args.save_every == 0:
            trainer.agent.save_checkpoint(save_dir / "latest.pt", extra=extra)
            trainer.agent.save_checkpoint(save_dir / f"episode_{ep}.pt", extra=extra)

        if stats.eval_ran and eval_success > best_success:
            best_success = eval_success
            extra["best_success"] = best_success
            trainer.agent.save_checkpoint(save_dir / "best.pt", extra=extra)


if __name__ == "__main__":
    main()
