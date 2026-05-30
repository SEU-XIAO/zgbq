"""
条件式 Recency Masking 测试

只在检测到振荡时才启用禁行掩码，正常走路不受影响。
对比 vanilla env 和 masked env 在相同场景上的表现。

用法:
  python tests/test_recency_masking.py
  python tests/test_recency_masking.py --model episode_4600.pt --cases 100
  python tests/test_recency_masking.py --output-dir outputs
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
from interfaces.config import DEFAULT_MAP_PATH, DEFAULT_MODEL_PATH

Coord = tuple[int, int]


# ═══════════════════════════════════════════════════════════════
#  条件式 Recency Masking 环境
# ═══════════════════════════════════════════════════════════════

class RecencyMaskedEnv(TacticalBattlefieldEnv):
    """
    只在检测到振荡时才封锁最近走过的格子。

    振荡判定: 最近 OSC_WINDOW 步内去重坐标 <= OSC_UNIQUE_LIMIT
    禁行范围: 最近 RECENT_BLOCK 步内去过的格子不允许再进入
    """

    OSC_WINDOW = 10
    OSC_UNIQUE_LIMIT = 3
    OSC_MIN_STEPS = 8
    RECENT_BLOCK = 3  # 封锁最近 3 步去过的格子

    def __init__(self, grid: BattlefieldMap, config: EnvConfig):
        super().__init__(grid, config)
        self._recent_positions: deque[Coord] = deque(maxlen=self.OSC_WINDOW)
        self._oscillation_active = False
        self._mask_interventions = 0  # 统计干预次数

    def reset(self, start, goal, enemies, waypoints, threat_scale=1.0):
        obs = super().reset(start, goal, enemies, waypoints, threat_scale)
        self._recent_positions.clear()
        self._recent_positions.append(self.pos)
        self._oscillation_active = False
        self._mask_interventions = 0
        return obs

    def step(self, action: int):
        result = super().step(action)
        self._recent_positions.append(self.pos)

        # 实时检测振荡状态
        if len(self._recent_positions) >= self.OSC_MIN_STEPS:
            recent = list(self._recent_positions)[-self.OSC_WINDOW:]
            self._oscillation_active = len(set(recent)) <= self.OSC_UNIQUE_LIMIT
        else:
            self._oscillation_active = False

        return result

    def action_mask(self) -> np.ndarray:
        mask = super().action_mask()

        # 只在振荡激活时才封锁
        if not self._oscillation_active:
            return mask

        # 取最近 RECENT_BLOCK 个位置 (排除当前步)
        recent_set = set(list(self._recent_positions)[-self.RECENT_BLOCK:])
        # 当前步本身也在 recent_set 里，需要排除当前 pos
        recent_set.discard(self.pos)

        if not recent_set:
            return mask

        blocked_any = False
        for i, (dr, dc) in enumerate(ACTIONS_8):
            if mask[i] <= 0.5:
                continue
            nxt = (self.pos[0] + dr, self.pos[1] + dc)
            if nxt in recent_set:
                mask[i] = 0.0
                blocked_any = True

        if blocked_any:
            # 如果全部被封死了，回退到原始 mask，避免卡死
            if not np.any(mask > 0.5):
                mask = super().action_mask()
            else:
                self._mask_interventions += 1

        return mask


# ═══════════════════════════════════════════════════════════════
#  模型加载 & 执行
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


def run_episode(
    model: TacticalD3QN,
    device: torch.device,
    grid: BattlefieldMap,
    start: Coord,
    goal: Coord,
    enemies: list[EnemySpec],
    env_cls: type[TacticalBattlefieldEnv],
    max_steps: int = 50,
    detect_oscillation: bool = True,
) -> dict[str, Any]:
    """
    detect_oscillation: True = 检测到振荡立刻停止 (vanilla 模式)
                        False = 不提前停止，让 masking 自己破局 (masked 模式)
    """
    env_cfg = EnvConfig(timeout_steps=max_steps, no_progress_limit=max_steps + 1)
    env = env_cls(grid, env_cfg)
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
        # 只有 vanilla 模式才因振荡提前停止
        if detect_oscillation and len(path) >= 8:
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

    interventions = getattr(env, "_mask_interventions", 0)

    # 事后检测: 路径中是否存在振荡段
    post_osc = False
    if len(path) >= 10:
        for i in range(len(path) - 9):
            if len(set(path[i:i+10])) <= 3:
                post_osc = True
                break

    # 可见比例
    vis_count, vis_ratio = compute_visible_ratio(grid, path, enemies)

    return {
        "path": path,
        "reason": reason,
        "success": path[-1] == goal,
        "steps": len(path) - 1,
        "interventions": interventions,
        "post_oscillation": post_osc,
        "visible_steps": vis_count,
        "visible_ratio": vis_ratio,
    }


# ═══════════════════════════════════════════════════════════════
#  分析函数 (简化版)
# ═══════════════════════════════════════════════════════════════

def path_efficiency(path: Sequence[Coord]) -> float:
    if len(path) < 2:
        return 0.0
    net = abs(path[0][0]-path[-1][0]) + abs(path[0][1]-path[-1][1])
    cum = sum(abs(a[0]-b[0])+abs(a[1]-b[1]) for a, b in zip(path, path[1:]))
    return net / max(1, cum)


def reversal_rate(path: Sequence[Coord]) -> float:
    if len(path) < 3:
        return 0.0
    deltas = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    rev = sum(1 for i in range(len(deltas)-1)
              if deltas[i] == (-deltas[i+1][0], -deltas[i+1][1]))
    return rev / max(1, len(deltas)-1)


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


def compute_visible_ratio(
    grid: BattlefieldMap,
    path: Sequence[Coord],
    enemies: Sequence[EnemySpec],
) -> tuple[int, float]:
    """计算路径中被敌人看到的步数和比例。"""
    if not enemies:
        return 0, 0.0
    visible_count = sum(
        1 for p in path
        if any(_point_visible(grid, p, e) for e in enemies)
    )
    return visible_count, visible_count / max(1, len(path))


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def build_scenes(base_grid: BattlefieldMap, cases: int, stage: int, seed: int) -> list:
    random.seed(seed)
    np.random.seed(seed)
    cfg = TrainingSceneConfig(enabled=True)
    sampler = LocalSceneSampler(base_grid, cfg, EnvConfig().window_size)
    return [sampler.sample(stage=stage) for _ in range(cases)]


def run_comparison(
    map_path: str,
    model_path: str | Path,
    cases: int,
    stage: int,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    base_grid = load_txt_map(map_path)
    scenes = build_scenes(base_grid, cases, stage, seed)
    device = select_device()

    resolved = Path(model_path)
    if not resolved.is_absolute() or not resolved.exists():
        resolved = PROJECT_ROOT / model_path
    model = load_policy(resolved, device)
    print(f"Model: {resolved}  Device: {device}  Cases: {cases}")

    vanilla_results: list[dict[str, Any]] = []
    masked_results: list[dict[str, Any]] = []

    for i, scene in enumerate(scenes):
        # Vanilla: 检测到振荡立刻停止
        r_v = run_episode(model, device, scene.grid, scene.start, scene.goal,
                          scene.enemies, TacticalBattlefieldEnv, max_steps,
                          detect_oscillation=True)
        # Masked: 不提前停止，让 masking 自己破局
        r_m = run_episode(model, device, scene.grid, scene.start, scene.goal,
                          scene.enemies, RecencyMaskedEnv, max_steps,
                          detect_oscillation=False)

        vanilla_results.append(r_v)
        masked_results.append(r_m)

        # 逐 case 打印差异
        v_icon = "OK" if r_v["success"] else "XX"
        m_icon = "OK" if r_m["success"] else "XX"
        changed = ""
        if r_v["success"] != r_m["success"]:
            changed = " ★ CHANGED"
        elif r_v["reason"] != r_m["reason"]:
            changed = " ~reason"

        interventions = r_m.get("interventions", 0)
        intv_str = f" intv={interventions}" if interventions > 0 else ""

        print(f"  case {i:3d}: start={scene.start} goal={scene.goal} "
              f"vanilla={v_icon}({r_v['reason']:10s} {r_v['steps']:2d}步) "
              f"masked={m_icon}({r_m['reason']:10s} {r_m['steps']:2d}步)"
              f"{intv_str}{changed}")

    return {
        "model": str(model_path),
        "cases": cases,
        "stage": stage,
        "seed": seed,
        "max_steps": max_steps,
        "vanilla": vanilla_results,
        "masked": masked_results,
    }


def print_summary(data: dict[str, Any]) -> None:
    v = data["vanilla"]
    m = data["masked"]
    n = len(v)

    v_succ = sum(1 for r in v if r["success"])
    m_succ = sum(1 for r in m if r["success"])
    v_osc = sum(1 for r in v if r["reason"] == "oscillation")
    m_osc = sum(1 for r in m if r["reason"] == "oscillation")
    v_avg_eff = sum(path_efficiency(r["path"]) for r in v) / max(1, n)
    m_avg_eff = sum(path_efficiency(r["path"]) for r in m) / max(1, n)
    v_avg_rev = sum(reversal_rate(r["path"]) for r in v) / max(1, n)
    m_avg_rev = sum(reversal_rate(r["path"]) for r in m) / max(1, n)
    v_avg_steps = sum(r["steps"] for r in v) / max(1, n)
    m_avg_steps = sum(r["steps"] for r in m) / max(1, n)

    # 可见比例 (只统计有敌人的 case)
    v_vis_cases = [r for r in v if r.get("visible_ratio") is not None]
    m_vis_cases = [r for r in m if r.get("visible_ratio") is not None]
    v_avg_vis = sum(r["visible_ratio"] for r in v_vis_cases) / max(1, len(v_vis_cases))
    m_avg_vis = sum(r["visible_ratio"] for r in m_vis_cases) / max(1, len(m_vis_cases))

    # 统计干预次数
    total_intv = sum(r.get("interventions", 0) for r in m)
    # masked 环境中事后检测到振荡的 case 数
    m_post_osc = sum(1 for r in m if r.get("post_oscillation", False))

    # 找出翻转的 case (vanilla 失败 -> masked 成功, 或反过来)
    fixed = []
    broken = []
    for i, (rv, rm) in enumerate(zip(v, m)):
        if not rv["success"] and rm["success"]:
            fixed.append(i)
        elif rv["success"] and not rm["success"]:
            broken.append(i)

    print(f"\n{'═'*60}")
    print(f"  COMPARISON SUMMARY  ({n} cases, model={data['model']})")
    print(f"{'═'*60}")
    print(f"  {'Metric':<28s} {'Vanilla':>10s} {'Masked':>10s} {'Delta':>10s}")
    print(f"  {'─'*58}")
    print(f"  {'Success rate':<28s} {v_succ/n:>9.1%} {m_succ/n:>9.1%} {(m_succ-v_succ)/n:>+9.1%}")
    print(f"  {'Oscillation stops':<28s} {v_osc:>10d} {'N/A':>10s}")
    print(f"  {'Post-oscillation detected':<28s} {'N/A':>10s} {m_post_osc:>10d}")
    print(f"  {'Avg displacement eff':<28s} {v_avg_eff:>10.3f} {m_avg_eff:>10.3f} {m_avg_eff-v_avg_eff:>+10.3f}")
    print(f"  {'Avg reversal rate':<28s} {v_avg_rev:>10.3f} {m_avg_rev:>10.3f} {m_avg_rev-v_avg_rev:>+10.3f}")
    print(f"  {'Avg steps':<28s} {v_avg_steps:>10.1f} {m_avg_steps:>10.1f} {m_avg_steps-v_avg_steps:>+10.1f}")
    print(f"  {'Avg visible ratio':<28s} {v_avg_vis:>10.3f} {m_avg_vis:>10.3f} {m_avg_vis-v_avg_vis:>+10.3f}")
    print(f"  {'Total mask interventions':<28s} {'N/A':>10s} {total_intv:>10d}")

    if fixed:
        print(f"\n  FIXED (vanilla failed, masked succeeded): {fixed}")
        for i in fixed:
            rv, rm = v[i], m[i]
            print(f"    case {i}: {rv['steps']}步({rv['reason']}) -> {rm['steps']}步({rm['reason']}), "
                  f"interventions={rm.get('interventions',0)}")
    if broken:
        print(f"\n  BROKEN (vanilla succeeded, masked failed): {broken}")
        for i in broken:
            rv, rm = v[i], m[i]
            print(f"    case {i}: {rv['steps']}步({rv['reason']}) -> {rm['steps']}步({rm['reason']})")

    if not fixed and not broken:
        print(f"\n  No case outcomes changed.")

    print(f"{'═'*60}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test conditional recency masking")
    p.add_argument("--map", default=DEFAULT_MAP_PATH)
    p.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p.add_argument("--cases", type=int, default=50)
    p.add_argument("--stage", type=int, default=3, choices=[1,2,3])
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--output-dir", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    data = run_comparison(
        map_path=args.map, model_path=args.model,
        cases=args.cases, stage=args.stage,
        seed=args.seed, max_steps=args.max_steps,
    )
    print_summary(data)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = out_dir / f"recency_masking_test_{ts}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
