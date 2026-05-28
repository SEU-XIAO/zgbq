"""
可见性回避倾向测试

对比 episode_6000 / episode_8000 / BFS / A* 四种寻路方式，
证明 episode_6000 有主动回避可见性格子的倾向。

设计思路：
- 用 LocalSceneSampler 生成有敌人的场景
- BFS/A* 不考虑可见性，走最短路径（baseline）
- 如果 episode_6000 的可见比例显著低于 BFS/A*，说明它在主动绕路躲避
- 同时对比 episode_8000，看加入可见性惩罚后是否有改善

用法:
  python tests/compare_visibility_tendency.py
  python tests/compare_visibility_tendency.py --model episode_6000.pt --cases 100
  python tests/compare_visibility_tendency.py --stage 3 --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battlefield_rl.config import EnvConfig, TrainingSceneConfig
from battlefield_rl.env import ACTIONS_8, EnemySpec, TacticalBattlefieldEnv
from battlefield_rl.env.tactical_env import angle_deg, angle_diff, bresenham_line
from battlefield_rl.map import BattlefieldMap, LocalSceneSampler, load_txt_map
from battlefield_rl.rl.network import TacticalD3QN, masked_q_values

Coord = tuple[int, int]


# ═══════════════════════════════════════════════════════════════
#  可见性计算
# ═══════════════════════════════════════════════════════════════

def _point_visible(grid: BattlefieldMap, point: Coord, enemy: EnemySpec) -> bool:
    src = (enemy.row, enemy.col)
    d = math.hypot(point[0] - src[0], point[1] - src[1])
    if d > enemy.max_range:
        return False
    if angle_diff(angle_deg(src, point), enemy.facing_deg) > enemy.fov_deg * 0.5:
        return False
    line = bresenham_line(src[0], src[1], point[0], point[1])
    for r, c in line[1:-1]:
        if grid.is_static_blocked(r, c):
            return False
    return True


def path_visible_stats(
    grid: BattlefieldMap,
    path: Sequence[Coord],
    enemies: Sequence[EnemySpec],
) -> dict[str, Any]:
    """计算路径的可见性统计"""
    if not enemies or len(path) == 0:
        return {"visible_steps": 0, "total_steps": len(path), "visible_ratio": 0.0}

    visible_count = sum(
        1 for p in path
        if any(_point_visible(grid, p, e) for e in enemies)
    )
    return {
        "visible_steps": visible_count,
        "total_steps": len(path),
        "visible_ratio": visible_count / max(1, len(path)),
    }


# ═══════════════════════════════════════════════════════════════
#  BFS / A* 基线寻路
# ═══════════════════════════════════════════════════════════════

def bfs_path(grid: BattlefieldMap, start: Coord, goal: Coord) -> list[Coord]:
    """BFS 最短路径（不考虑可见性）"""
    if start == goal:
        return [start]
    queue: deque[Coord] = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}
    while queue:
        cur = queue.popleft()
        for dr, dc in ACTIONS_8:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in parent or not grid.is_passable(cur, nxt, 0):
                continue
            parent[nxt] = cur
            if nxt == goal:
                queue.clear()
                break
            queue.append(nxt)
    if goal not in parent:
        return []
    path: list[Coord] = []
    node: Coord | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def astar_path(grid: BattlefieldMap, start: Coord, goal: Coord) -> list[Coord]:
    """A* 最短路径（不考虑可见性）"""
    import heapq

    def h(pos: Coord) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    open_set: list[tuple[int, Coord]] = [(h(start), start)]
    parent: dict[Coord, Coord | None] = {start: None}
    g_score: dict[Coord, int] = {start: 0}

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            break
        for dr, dc in ACTIONS_8:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not grid.is_passable(cur, nxt, 0):
                continue
            new_g = g_score[cur] + 1
            if nxt not in g_score or new_g < g_score[nxt]:
                g_score[nxt] = new_g
                parent[nxt] = cur
                heapq.heappush(open_set, (new_g + h(nxt), nxt))

    if goal not in parent:
        return []
    path: list[Coord] = []
    node: Coord | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


# ═══════════════════════════════════════════════════════════════
#  模型执行
# ═══════════════════════════════════════════════════════════════

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


def run_model_episode(
    model: TacticalD3QN,
    device: torch.device,
    grid: BattlefieldMap,
    start: Coord,
    goal: Coord,
    enemies: list[EnemySpec],
    max_steps: int = 50,
) -> dict[str, Any]:
    """用模型跑一个 episode，返回路径"""
    env_cfg = EnvConfig(timeout_steps=max_steps, no_progress_limit=max_steps + 1)
    env = TacticalBattlefieldEnv(grid, env_cfg)
    obs = env.reset(start=start, goal=goal, enemies=enemies, waypoints=[goal], threat_scale=1.0)

    path: list[Coord] = [start]
    reason = "step_limit"

    for _ in range(max_steps):
        action = greedy_action(model, obs, env.action_mask(), device)
        step = env.step(action)
        obs = step.observation
        path.append(env.pos)

        if env.pos == goal:
            reason = "reached"
            break
        # 检测振荡
        if len(path) >= 8:
            recent = path[-10:]
            if len(set(recent)) <= 3:
                reason = "oscillation"
                break
        if step.done:
            if step.info.get("stuck"):
                reason = "stuck"
            elif step.info.get("timeout"):
                reason = "timeout"
            else:
                reason = "done"
            break

    return {
        "path": path,
        "reason": reason,
        "success": path[-1] == goal,
        "steps": len(path) - 1,
    }


# ═══════════════════════════════════════════════════════════════
#  场景生成
# ═══════════════════════════════════════════════════════════════

def build_scenes(base_grid: BattlefieldMap, cases: int, stage: int, seed: int) -> list:
    random.seed(seed)
    np.random.seed(seed)
    cfg = TrainingSceneConfig(enabled=True)
    sampler = LocalSceneSampler(base_grid, cfg, EnvConfig().window_size)
    return [sampler.sample(stage=stage) for _ in range(cases)]


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def run_comparison(
    map_path: str,
    model_paths: list[str],
    cases: int,
    stage: int,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    base_grid = load_txt_map(map_path)
    scenes = build_scenes(base_grid, cases, stage, seed)
    device = select_device()

    models = {}
    for mp in model_paths:
        resolved = Path(mp)
        if not resolved.is_absolute() or not resolved.exists():
            resolved = PROJECT_ROOT / mp
        models[mp] = load_policy(resolved, device)
        print(f"Loaded: {resolved}")

    results: dict[str, list[dict[str, Any]]] = {m: [] for m in model_paths}
    results["bfs"] = []
    results["astar"] = []

    for i, scene in enumerate(scenes):
        grid = scene.grid
        start = scene.start
        goal = scene.goal
        enemies = scene.enemies

        # BFS baseline
        bfs_p = bfs_path(grid, start, goal)
        bfs_vis = path_visible_stats(grid, bfs_p, enemies)
        results["bfs"].append({
            "case": i,
            "success": len(bfs_p) > 0 and bfs_p[-1] == goal,
            "steps": max(0, len(bfs_p) - 1),
            "path": bfs_p,
            **bfs_vis,
        })

        # A* baseline
        astar_p = astar_path(grid, start, goal)
        astar_vis = path_visible_stats(grid, astar_p, enemies)
        results["astar"].append({
            "case": i,
            "success": len(astar_p) > 0 and astar_p[-1] == goal,
            "steps": max(0, len(astar_p) - 1),
            "path": astar_p,
            **astar_vis,
        })

        # 模型执行
        for mp in model_paths:
            r = run_model_episode(models[mp], device, grid, start, goal, enemies, max_steps)
            vis = path_visible_stats(grid, r["path"], enemies)
            results[mp].append({
                "case": i,
                "success": r["success"],
                "steps": r["steps"],
                "reason": r["reason"],
                "path": r["path"],
                **vis,
            })

        # 逐 case 打印
        parts = []
        for key in ["bfs", "astar"] + model_paths:
            rr = results[key][-1]
            name = key.split("/")[-1].split(".")[0] if "/" in key else key
            icon = "OK" if rr["success"] else "XX"
            vis = rr["visible_ratio"]
            parts.append(f"{name}={icon}({rr['steps']}步 vis={vis:.2f})")

        print(f"  case {i:3d}: {start}->{goal} enemies={len(enemies)}")
        print(f"           {'  '.join(parts)}")

    return {
        "map": map_path,
        "models": model_paths,
        "cases": cases,
        "stage": stage,
        "seed": seed,
        "max_steps": max_steps,
        "results": results,
    }


def print_summary(data: dict[str, Any]) -> None:
    results = data["results"]
    n = data["cases"]

    print(f"\n{'='*80}")
    print(f"  VISIBILITY TENDENCY COMPARISON  ({n} cases, stage={data['stage']})")
    print(f"{'='*80}")

    # 表头
    keys = ["bfs", "astar"] + data["models"]
    short_names = []
    for k in keys:
        if "/" in k:
            short_names.append(k.split("/")[-1].split(".")[0])
        else:
            short_names.append(k)

    header = f"  {'Metric':<30s}" + "".join(f"{sn:>12s}" for sn in short_names)
    print(header)
    print(f"  {'-'*80}")

    # 指标
    for metric, key, fmt in [
        ("Success rate", "success", lambda v: f"{sum(v)/len(v):>11.1%}"),
        ("Avg steps", "steps", lambda v: f"{sum(v)/len(v):>11.1f}"),
        ("Avg visible ratio", "visible_ratio", lambda v: f"{sum(v)/len(v):>11.3f}"),
        ("Avg visible steps", "visible_steps", lambda v: f"{sum(v)/len(v):>11.1f}"),
    ]:
        vals = [fmt([r[key] for r in results[k]]) for k in keys]
        print(f"  {metric:<30s}" + "".join(f"{v:>12s}" for v in vals))

    # 可见性分布
    print(f"\n  Visibility ratio distribution:")
    for k, sn in zip(keys, short_names):
        vis_vals = [r["visible_ratio"] for r in results[k]]
        zero = sum(1 for v in vis_vals if v < 0.01)
        low = sum(1 for v in vis_vals if 0.01 <= v < 0.20)
        mid = sum(1 for v in vis_vals if 0.20 <= v < 0.50)
        high = sum(1 for v in vis_vals if v >= 0.50)
        print(f"    {sn:>10s}: zero={zero:3d}  low={low:3d}  mid={mid:3d}  high={high:3d}")

    # 只看都有敌人的 case（visible_ratio > 0 的 case）
    print(f"\n  Cases with enemies present (visible_ratio > 0 for BFS):")
    enemy_cases = [i for i in range(n) if results["bfs"][i]["visible_ratio"] > 0.01]
    if enemy_cases:
        print(f"    Found {len(enemy_cases)} cases")
        for metric, key, fmt in [
            ("Success rate", "success", lambda v: f"{sum(v)/len(v):>11.1%}"),
            ("Avg steps", "steps", lambda v: f"{sum(v)/len(v):>11.1f}"),
            ("Avg visible ratio", "visible_ratio", lambda v: f"{sum(v)/len(v):>11.3f}"),
        ]:
            vals = [fmt([results[k][i][key] for i in enemy_cases]) for k in keys]
            print(f"    {metric:<30s}" + "".join(f"{v:>12s}" for v in vals))
    else:
        print(f"    No cases with visible enemies")

    # 逐 case 可见性对比（只显示模型和 BFS 差异大的）
    print(f"\n  Top cases where model differs from BFS (visibility):")
    for mp in data["models"]:
        mp_short = mp.split("/")[-1].split(".")[0]
        diffs = []
        for i in range(n):
            bfs_vis = results["bfs"][i]["visible_ratio"]
            mp_vis = results[mp][i]["visible_ratio"]
            if bfs_vis > 0.01:  # 只看有可见敌人的情况
                diff = bfs_vis - mp_vis  # 正值 = 模型更少可见
                diffs.append((i, diff, bfs_vis, mp_vis, results[mp][i]["steps"]))
        diffs.sort(key=lambda x: x[1], reverse=True)
        print(f"\n    {mp_short} vs BFS (top 5 avoidance cases):")
        for i, diff, bfs_v, mp_v, mp_s in diffs[:5]:
            icon = "OK" if results[mp][i]["success"] else "XX"
            print(f"      case {i}: BFS={bfs_v:.3f} {mp_short}={mp_v:.3f} "
                  f"delta={diff:+.3f} {icon}({mp_s}步)")

    print(f"\n{'='*80}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare visibility avoidance tendency")
    p.add_argument("--map", default="MyPath_Data417.txt")
    p.add_argument("--model", action="append", default=[], help="model path (repeatable)")
    p.add_argument("--cases", type=int, default=80)
    p.add_argument("--stage", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--output-dir", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    model_paths = args.model or ["episode_6000.pt", "episode_8000.pt"]

    print(f"Map: {args.map}  Cases: {args.cases}  Stage: {args.stage}")
    print(f"Models: {model_paths}")

    data = run_comparison(
        map_path=args.map, model_paths=model_paths,
        cases=args.cases, stage=args.stage,
        seed=args.seed, max_steps=args.max_steps,
    )
    print_summary(data)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = out_dir / f"visibility_tendency_{ts}.json"
        # 不保存完整路径避免文件过大
        for k in data["results"]:
            for r in data["results"][k]:
                r.pop("path", None)
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
