"""
收集困难场景（超时/卡死/震荡）用于 Stage 4 训练。

用纯局部执行器（不开全局规划、不开 fallback）跑大量场景，
收集所有失败 case 的完整场景信息（grid、start、goal、enemies），
保存为可直接加载的 JSON 文件。

用法:
  python tests/collect_hard_scenarios.py
  python tests/collect_hard_scenarios.py --cases 200 --stage 3 --model episode_8000.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
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
from battlefield_rl.map import BattlefieldMap, LocalSceneSampler, load_txt_map
from battlefield_rl.rl.network import TacticalD3QN, masked_q_values
from interfaces.config import DEFAULT_MAP_PATH, DEFAULT_MODEL_PATH
from scripts.config import DEFAULT_SEED
from scripts.eval_hierarchical_executor import coord_to_list, load_policy, select_device

Coord = tuple[int, int]


# ═══════════════════════════════════════════════════════════════
#  场景序列化
# ═══════════════════════════════════════════════════════════════

def scene_to_dict(scene: Any) -> dict[str, Any]:
    """将训练场景序列化为可保存的 dict。"""
    grid = scene.grid
    rows, cols = grid.shape
    height_data = grid.heights.tolist()
    type_data = grid.types.tolist()

    return {
        "grid": {
            "rows": rows,
            "cols": cols,
            "height": height_data,
            "type": type_data,
        },
        "start": coord_to_list(scene.start),
        "goal": coord_to_list(scene.goal),
        "enemies": [
            {
                "row": e.row, "col": e.col,
                "facing_deg": float(e.facing_deg),
                "fov_deg": float(e.fov_deg),
                "max_range": int(e.max_range),
            }
            for e in scene.enemies
        ],
        "source": scene.source,
        "enemy_case": scene.enemy_case,
    }


def load_scene_from_dict(data: dict[str, Any]) -> Any:
    """从 dict 恢复训练场景。"""
    from battlefield_rl.map import BattlefieldMap as BM

    g = data["grid"]
    grid = BM(
        heights=np.array(g["height"], dtype=np.int32),
        types=np.array(g["type"], dtype=np.int32),
    )
    start = tuple(data["start"])
    goal = tuple(data["goal"])
    enemies = [
        EnemySpec(row=e["row"], col=e["col"],
                  facing_deg=e["facing_deg"], fov_deg=e["fov_deg"],
                  max_range=e["max_range"])
        for e in data["enemies"]
    ]
    return grid, start, goal, enemies


# ═══════════════════════════════════════════════════════════════
#  纯局部执行
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_action(model, obs: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
    x = torch.from_numpy(obs).unsqueeze(0).float().to(device)
    m = torch.from_numpy(mask).unsqueeze(0).float().to(device)
    q = masked_q_values(model(x), m)
    return int(torch.argmax(q, dim=1).item())


def run_local_episode(
    *,
    model,
    device: torch.device,
    grid: BattlefieldMap,
    start: Coord,
    goal: Coord,
    enemies: list[EnemySpec],
    max_steps: int,
) -> dict[str, Any]:
    """纯局部执行，不开全局规划、不开 fallback。"""
    env_cfg = EnvConfig(timeout_steps=max_steps, no_progress_limit=max_steps + 1)
    env = TacticalBattlefieldEnv(grid, env_cfg)
    obs = env.reset(
        start=start, goal=goal, enemies=enemies,
        waypoints=[goal], threat_scale=1.0,
    )

    path = [start]
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

        if env.pos == goal:
            reason = "reached"
            break
        if len(recent) == recent.maxlen and len(set(recent)) <= 3:
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

    # 路径抖动指标
    p = [tuple(c) for c in path]
    net = abs(p[-1][0] - p[0][0]) + abs(p[-1][1] - p[0][1])
    cumulative = len(p) - 1
    disp_eff = net / max(1, cumulative)

    deltas = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(p, p[1:])]
    reversals = sum(1 for i in range(len(deltas) - 1)
                    if deltas[i] == (-deltas[i + 1][0], -deltas[i + 1][1]))
    reversal_rate = reversals / max(1, len(deltas) - 1)

    unique = len(set(p))
    repeat_ratio = 1.0 - unique / len(p)

    visible_count = sum(1 for v in visible_flags if v)

    return {
        "success": path[-1] == goal,
        "reason": reason,
        "steps": max(0, len(path) - 1),
        "final_pos": coord_to_list(path[-1]),
        "goal_distance": abs(path[-1][0] - goal[0]) + abs(path[-1][1] - goal[1]),
        "displacement_eff": round(disp_eff, 3),
        "reversal_rate": round(reversal_rate, 3),
        "repeat_ratio": round(repeat_ratio, 3),
        "visible_ratio": round(visible_count / max(1, len(path)), 3),
        "total_reward": round(float(sum(rewards)), 2),
        "path_head": [coord_to_list(c) for c in path[:20]],
        "path_tail": [coord_to_list(c) for c in path[-20:]],
    }


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect hard scenarios for Stage 4 training")
    p.add_argument("--map", default=DEFAULT_MAP_PATH)
    p.add_argument("--models", nargs="+", default=[DEFAULT_MODEL_PATH])
    p.add_argument("--cases", type=int, default=200, help="total scenes to test")
    p.add_argument("--stage", type=int, default=3, choices=[1, 2, 3],
                   help="curriculum stage for scene generation")
    p.add_argument("--max-steps", type=int, default=100,
                   help="max steps per episode (longer than training to catch slow cases)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", default="outputs/hard_scenarios")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p


def run_one_model(
    model_name: str,
    device: torch.device,
    scenes: list[Any],
    max_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """对单个模型跑所有场景，返回 (结果列表, 困难场景列表)。"""
    model_path = PROJECT_ROOT / model_name
    model = load_policy(model_path, device)

    results: list[dict[str, Any]] = []
    hard: list[dict[str, Any]] = []

    for i, scene in enumerate(scenes):
        result = run_local_episode(
            model=model, device=device,
            grid=scene.grid, start=scene.start, goal=scene.goal,
            enemies=list(scene.enemies), max_steps=max_steps,
        )
        result["case_index"] = i
        results.append(result)

        status = "OK" if result["success"] else result["reason"].upper()
        print(f"  [{i+1:4d}/{len(scenes)}] "
              f"start={coord_to_list(scene.start)} goal={coord_to_list(scene.goal)} "
              f"steps={result['steps']:>4} "
              f"disp={result['displacement_eff']:.3f} "
              f"rev={result['reversal_rate']:.3f} "
              f"rep={result['repeat_ratio']:.3f} "
              f"vis={result['visible_ratio']:.2%} "
              f"{status}")

        # 收集困难场景
        is_hard = (
            not result["success"]
            or result["reason"] in ("oscillation", "timeout", "stuck")
            or result["displacement_eff"] < 0.5
            or result["reversal_rate"] > 0.2
            or result["repeat_ratio"] > 0.4
        )
        if is_hard:
            scenario_data = scene_to_dict(scene)
            scenario_data["difficulty"] = {
                "reason": result["reason"],
                "displacement_eff": result["displacement_eff"],
                "reversal_rate": result["reversal_rate"],
                "repeat_ratio": result["repeat_ratio"],
            }
            hard.append(scenario_data)

    return results, hard


def summarize_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总单个模型的结果。"""
    total = len(results)
    if total == 0:
        return {}
    successes = [r for r in results if r["success"]]
    reason_counts = Counter(r["reason"] for r in results)

    def _avg(key: str, items: list[dict] = results) -> float:
        vals = [r[key] for r in items if key in r]
        return sum(vals) / max(1, len(vals))

    # 仅成功 case 的指标
    def _avg_ok(key: str) -> float:
        vals = [r[key] for r in successes if key in r]
        return sum(vals) / max(1, len(vals))

    return {
        "total": total,
        "success_count": len(successes),
        "success_rate": round(len(successes) / total, 4),
        "oscillation_count": reason_counts.get("oscillation", 0),
        "timeout_count": reason_counts.get("timeout", 0),
        "stuck_count": reason_counts.get("stuck", 0),
        # 全部 case 的平均值
        "avg_displacement_eff": round(_avg("displacement_eff"), 3),
        "avg_reversal_rate": round(_avg("reversal_rate"), 3),
        "avg_repeat_ratio": round(_avg("repeat_ratio"), 3),
        "avg_visible_ratio": round(_avg("visible_ratio"), 4),
        # 仅成功 case
        "avg_steps_on_success": round(_avg_ok("steps"), 1),
        "avg_disp_eff_on_success": round(_avg_ok("displacement_eff"), 3),
        "avg_rev_rate_on_success": round(_avg_ok("reversal_rate"), 3),
        "avg_rep_ratio_on_success": round(_avg_ok("repeat_ratio"), 3),
        "avg_vis_ratio_on_success": round(_avg_ok("visible_ratio"), 4),
        "reasons": dict(reason_counts),
    }


def print_comparison(summaries: dict[str, dict[str, Any]]) -> None:
    """打印多模型对比表格。"""
    models = list(summaries.keys())
    col_w = 14

    print(f"\n{'=' * (16 + col_w * len(models))}")
    print("  六模型局部执行器对比")
    print(f"{'=' * (16 + col_w * len(models))}")

    header = f"{'指标':<16}" + "".join(f"{m:>{col_w}}" for m in models)
    print(header)
    print("-" * len(header))

    rows = [
        ("成功率",          "success_rate",            lambda v: f"{v:>11.1%}"),
        ("震荡停止",        "oscillation_count",       lambda v: f"{v:>12d}"),
        ("超时",            "timeout_count",           lambda v: f"{v:>12d}"),
        ("卡死",            "stuck_count",             lambda v: f"{v:>12d}"),
        ("--- 全部case ---", None,                      lambda v: f"{'':>{col_w}}"),
        ("平均位移效率",    "avg_displacement_eff",    lambda v: f"{v:>12.3f}"),
        ("平均反向率",      "avg_reversal_rate",       lambda v: f"{v:>11.1%}"),
        ("平均重复率",      "avg_repeat_ratio",        lambda v: f"{v:>11.1%}"),
        ("平均可见率",      "avg_visible_ratio",       lambda v: f"{v:>11.1%}"),
        ("--- 仅成功case --", None,                     lambda v: f"{'':>{col_w}}"),
        ("成功时平均步数",  "avg_steps_on_success",    lambda v: f"{v:>12.1f}"),
        ("成功时位移效率",  "avg_disp_eff_on_success", lambda v: f"{v:>12.3f}"),
        ("成功时反向率",    "avg_rev_rate_on_success", lambda v: f"{v:>11.1%}"),
        ("成功时重复率",    "avg_rep_ratio_on_success",lambda v: f"{v:>11.1%}"),
        ("成功时可见率",    "avg_vis_ratio_on_success",lambda v: f"{v:>11.1%}"),
    ]

    for label, key, fmt in rows:
        if key is None:
            print(f"  {label}")
            continue
        row = f"{label:<16}"
        for m in models:
            v = summaries[m].get(key, 0)
            row += fmt(v)
        print(row)

    print("-" * len(header))

    # 失败原因
    print("\n失败原因分布:")
    for m in models:
        reasons = summaries[m].get("reasons", {})
        osc = reasons.get("oscillation", 0)
        timeout = reasons.get("timeout", 0)
        stuck = reasons.get("stuck", 0)
        print(f"  {m:<24} oscillation={osc}  timeout={timeout}  stuck={stuck}")


def main() -> None:
    args = build_parser().parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # 生成固定场景（所有模型用同一组场景，公平对比）
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    base_grid = load_txt_map(args.map)
    cfg = TrainingSceneConfig(enabled=True)
    sampler = LocalSceneSampler(base_grid, cfg, EnvConfig().window_size)
    scenes = [sampler.sample(stage=args.stage) for _ in range(args.cases)]
    print(f"Generated {len(scenes)} scenes (stage={args.stage}, seed={args.seed})")

    # 跑所有模型
    all_summaries: dict[str, dict[str, Any]] = {}
    all_hard: dict[str, list[dict[str, Any]]] = {}
    all_details: dict[str, list[dict[str, Any]]] = {}

    for model in args.models:
        print(f"\n{'-' * 60}")
        print(f"  Model: {model}")
        print(f"{'-' * 60}")

        t0 = time.perf_counter()
        results, hard = run_one_model(model, device, scenes, args.max_steps)
        elapsed = time.perf_counter() - t0

        summary = summarize_model(results)
        all_summaries[model] = summary
        all_hard[model] = hard
        all_details[model] = results

        print(f"\n  成功: {summary['success_count']}/{summary['total']} "
              f"({summary['success_rate']:.1%})  "
              f"震荡: {summary['oscillation_count']}  "
              f"困难场景: {len(hard)}  "
              f"耗时: {elapsed:.1f}s")

    # 打印对比表
    print_comparison(all_summaries)

    # 保存
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 合并所有模型的困难场景（去重）
    merged_hard: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for model, hard_list in all_hard.items():
        for item in hard_list:
            key = f"{item['start']}_{item['goal']}_{len(item['enemies'])}"
            if key not in seen_keys:
                seen_keys.add(key)
                item["failed_models"] = [model]
                merged_hard.append(item)
            else:
                # 找到已有的，追加模型名
                for existing in merged_hard:
                    ekey = f"{existing['start']}_{existing['goal']}_{len(existing['enemies'])}"
                    if ekey == key:
                        existing.setdefault("failed_models", []).append(model)
                        break

    # 按失败模型数排序（越多模型失败 = 越难）
    merged_hard.sort(key=lambda x: len(x.get("failed_models", [])), reverse=True)

    # 保存困难场景
    hard_path = out_dir / f"hard_scenarios_{ts}.json"
    hard_path.write_text(json.dumps({
        "meta": {
            "models": args.models,
            "cases": args.cases,
            "stage": args.stage,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "timestamp": ts,
        },
        "summaries": all_summaries,
        "total_unique_hard": len(merged_hard),
        "scenarios": merged_hard,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n困难场景已保存: {hard_path} ({len(merged_hard)} 个)")

    # 保存完整结果
    full_path = out_dir / f"full_results_{ts}.json"
    full_path.write_text(json.dumps({
        "meta": {
            "models": args.models, "cases": args.cases,
            "stage": args.stage, "seed": args.seed,
        },
        "summaries": all_summaries,
        "details": all_details,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完整结果已保存: {full_path}")

    # 困难场景统计
    if merged_hard:
        n_all_fail = sum(1 for s in merged_hard if len(s.get("failed_models", [])) >= len(args.models) * 0.8)
        n_half_fail = sum(1 for s in merged_hard if len(s.get("failed_models", [])) >= len(args.models) * 0.5)
        print(f"\n[!] {n_all_fail} 个场景 >80% 模型失败（通用困难）")
        print(f"[!] {n_half_fail} 个场景 >50% 模型失败（中等困难）")


if __name__ == "__main__":
    main()
