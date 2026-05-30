from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

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
from battlefield_rl.planner import plan_global_path
from interfaces.config import DEFAULT_MAP_PATH
from battlefield_rl.train import HierarchicalTrainer


def parse_coord(text: str) -> Tuple[int, int]:
    a, b = text.split(",")
    return int(a), int(b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick demo for pure-height LoS tactical planner")
    parser.add_argument("--map", type=str, default=DEFAULT_MAP_PATH, help="地形图文件路径")
    parser.add_argument("--start", type=str, required=True, help="起始坐标 row,col")
    parser.add_argument("--goal", type=str, required=True, help="目标坐标 row,col")
    parser.add_argument("--episodes", type=int, default=3, help="运行 episode 数")
    args = parser.parse_args()

    grid = load_txt_map(args.map)
    start = parse_coord(args.start)
    goal = parse_coord(args.goal)

    env_cfg = EnvConfig()
    planner_cfg = PlannerConfig()

    planner_res = plan_global_path(
        grid,
        start,
        goal,
        max_height_diff=env_cfg.max_height_diff,
        waypoint_interval=planner_cfg.waypoint_interval,
        allow_diagonal=planner_cfg.allow_diagonal,
    )

    print(f"地图尺寸: {grid.shape[0]} x {grid.shape[1]}")
    print(f"全局路径长度: {len(planner_res.path)}")
    print(f"跳点数量: {len(planner_res.jump_points)}")
    print(f"虚拟路标数量: {len(planner_res.waypoints)}")

    trainer = HierarchicalTrainer(
        grid=grid,
        env_cfg=env_cfg,
        planner_cfg=planner_cfg,
        agent_cfg=AgentConfig(),
        curriculum_cfg=CurriculumConfig(),
        gate_cfg=CurriculumThreshold(),
        scene_cfg=TrainingSceneConfig(enabled=False),
        device="cpu",
    )

    for ep in range(1, args.episodes + 1):
        stats = trainer.run_episode(ep, start, goal, train=True)
        print(
            f"[Episode {stats.episode}] stage={stats.stage} steps={stats.steps} "
            f"reward={stats.reward:.3f} reached_goal={stats.reached_goal}"
        )


if __name__ == "__main__":
    main()
