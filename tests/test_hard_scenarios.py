"""用收集到的困难场景测试指定模型。"""

from __future__ import annotations

import json
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battlefield_rl.config import EnvConfig
from battlefield_rl.env import EnemySpec, TacticalBattlefieldEnv
from battlefield_rl.map import BattlefieldMap
from battlefield_rl.rl.network import masked_q_values
from scripts.eval_hierarchical_executor import coord_to_list, load_policy, select_device

Coord = tuple[int, int]


@torch.no_grad()
def greedy_action(model, obs: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
    x = torch.from_numpy(obs).unsqueeze(0).float().to(device)
    m = torch.from_numpy(mask).unsqueeze(0).float().to(device)
    q = masked_q_values(model(x), m)
    return int(torch.argmax(q, dim=1).item())


def run_episode(model, device, grid, start, goal, enemies, max_steps=100):
    env_cfg = EnvConfig(timeout_steps=max_steps, no_progress_limit=max_steps + 1)
    env = TacticalBattlefieldEnv(grid, env_cfg)
    obs = env.reset(start=start, goal=goal, enemies=enemies,
                    waypoints=[goal], threat_scale=1.0)

    path = [start]
    visible_flags = []
    recent: deque = deque(maxlen=12)
    reason = "step_limit"

    for _ in range(max_steps):
        action = greedy_action(model, obs, env.action_mask(), device)
        step = env.step(action)
        obs = step.observation
        path.append(env.pos)
        visible_flags.append(bool(step.info.get("visible", False)))
        recent.append(env.pos)
        if env.pos == goal:
            reason = "reached"
            break
        if len(recent) == recent.maxlen and len(set(recent)) <= 3:
            reason = "oscillation"
            break
        if step.done:
            reason = "timeout" if step.info.get("timeout") else ("stuck" if step.info.get("stuck") else "done")
            break

    p = [tuple(c) for c in path]
    net = abs(p[-1][0] - p[0][0]) + abs(p[-1][1] - p[0][1])
    disp_eff = net / max(1, len(p) - 1)
    deltas = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(p, p[1:])]
    reversals = sum(1 for i in range(len(deltas) - 1) if deltas[i] == (-deltas[i + 1][0], -deltas[i + 1][1]))
    rev_rate = reversals / max(1, len(deltas) - 1)
    unique = len(set(p))
    rep_ratio = 1.0 - unique / len(p)
    vis_count = sum(1 for v in visible_flags if v)

    return {
        "success": path[-1] == goal,
        "reason": reason,
        "steps": max(0, len(path) - 1),
        "goal_distance": abs(path[-1][0] - goal[0]) + abs(path[-1][1] - goal[1]),
        "displacement_eff": round(disp_eff, 3),
        "reversal_rate": round(rev_rate, 3),
        "repeat_ratio": round(rep_ratio, 3),
        "visible_ratio": round(vis_count / max(1, len(path)), 3),
    }


def load_scenario(sc: dict) -> tuple:
    g = sc["grid"]
    grid = BattlefieldMap(
        heights=np.array(g["height"], dtype=np.int32),
        types=np.array(g["type"], dtype=np.int32),
    )
    start = tuple(sc["start"])
    goal = tuple(sc["goal"])
    enemies = [
        EnemySpec(row=e["row"], col=e["col"], facing_deg=e["facing_deg"],
                  fov_deg=e["fov_deg"], max_range=e["max_range"])
        for e in sc["enemies"]
    ]
    return grid, start, goal, enemies


def main():
    models_to_test = sys.argv[1:] if len(sys.argv) > 1 else ["v1.pt", "v2.pt"]

    # 找最新的 hard scenarios 文件
    scenario_dir = Path("outputs/hard_scenarios")
    files = sorted(scenario_dir.glob("hard_scenarios_*.json"))
    if not files:
        print("No hard scenarios found. Run collect_hard_scenarios.py first.")
        return

    data = json.loads(open(files[-1], encoding="utf-8").read())
    scenarios = data["scenarios"]
    print(f"Loaded {len(scenarios)} hard scenarios from {files[-1].name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in models_to_test:
        model_path = PROJECT_ROOT / model_name
        if not model_path.exists():
            print(f"[SKIP] {model_name} not found")
            continue

        model = load_policy(model_path, device)
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        results = []
        for i, sc in enumerate(scenarios):
            grid, start, goal, enemies = load_scenario(sc)
            r = run_episode(model, device, grid, start, goal, enemies)
            results.append(r)
            status = "OK" if r["success"] else r["reason"].upper()
            print(f"  [{i+1:3d}/{len(scenarios)}] "
                  f"start={start} goal={goal} "
                  f"steps={r['steps']:>3} dist={r['goal_distance']:>2} "
                  f"disp={r['displacement_eff']:.3f} rev={r['reversal_rate']:.3f} "
                  f"vis={r['visible_ratio']:.0%} {status}")

        # 汇总
        total = len(results)
        ok = sum(1 for r in results if r["success"])
        osc = sum(1 for r in results if r["reason"] == "oscillation")
        avg_disp = sum(r["displacement_eff"] for r in results) / total
        avg_rev = sum(r["reversal_rate"] for r in results) / total
        avg_vis = sum(r["visible_ratio"] for r in results) / total

        print(f"\n  success={ok}/{total} ({ok/total:.0%})  "
              f"oscillation={osc}  "
              f"avg_disp={avg_disp:.3f}  avg_rev={avg_rev:.3f}  avg_vis={avg_vis:.0%}")


if __name__ == "__main__":
    main()
