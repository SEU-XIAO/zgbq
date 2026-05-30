"""
六模型综合评测脚本

对比 6 个 D3QN checkpoint 的：成功率、超时率、路径效率、避敌能力、路径抖动、fallback 依赖。
超时/失败场景自动保存到 outputs/timeout_scenarios/，可用于后续 Stage 4 困难课程训练。

用法:
  python tests/eval_models_comprehensive.py
  python tests/eval_models_comprehensive.py --cases 30 --seed 123
  python tests/eval_models_comprehensive.py --models episode_8000.pt episode_6000.pt
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battlefield_rl.env import ACTIONS_8, EnemySpec
from battlefield_rl.env.tactical_env import angle_deg, angle_diff, bresenham_line
from battlefield_rl.map import BattlefieldMap, load_txt_map
from battlefield_rl.planner import plan_global_path
from interfaces.config import DEFAULT_MAP_PATH
from scripts.eval_hierarchical_executor import plan_and_execute

Coord = tuple[int, int]

ALL_MODELS = [
    "episode_4000.pt",
    "episode_4500.pt",
    "episode_4600.pt",
    "episode_6000.pt",
    "episode_7300.pt",
    "episode_8000.pt",
]

# ── 距离桶 ──────────────────────────────────────────────
DISTANCE_BUCKETS = {
    "short":  (20, 80),
    "medium": (90, 180),
    "long":   (190, 330),
}


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════

def passable_cells(grid: BattlefieldMap) -> list[Coord]:
    rows, cols = grid.shape
    return [(r, c) for r in range(rows) for c in range(cols)
            if not grid.is_static_blocked(r, c)]


def bfs_path(grid: BattlefieldMap, start: Coord, goal: Coord) -> list[Coord]:
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


def angle_from_to(a: Coord, b: Coord) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


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
    grid: BattlefieldMap, path: Sequence[Coord], enemies: Sequence[EnemySpec],
) -> dict[str, Any]:
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
#  路径抖动分析
# ═══════════════════════════════════════════════════════════════

def oscillation_metrics(path: Sequence[Coord]) -> dict[str, Any]:
    """计算路径抖动指标。"""
    p = [tuple(c) for c in path]
    if len(p) < 2:
        return {
            "displacement_eff": 0.0, "reversal_rate": 0.0,
            "repeat_ratio": 0.0, "unique_ratio": 1.0,
        }

    # 位移效率
    net = abs(p[-1][0] - p[0][0]) + abs(p[-1][1] - p[0][1])
    cumulative = len(p) - 1
    disp_eff = net / max(1, cumulative)

    # 反向率
    deltas = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(p, p[1:])]
    reversals = sum(
        1 for i in range(len(deltas) - 1)
        if deltas[i] == (-deltas[i + 1][0], -deltas[i + 1][1])
    )
    reversal_rate = reversals / max(1, len(deltas) - 1)

    # 重复访问率
    unique = len(set(p))
    repeat_ratio = 1.0 - unique / len(p)

    return {
        "displacement_eff": disp_eff,
        "reversal_rate": reversal_rate,
        "repeat_ratio": repeat_ratio,
        "unique_ratio": unique / len(p),
    }


# ═══════════════════════════════════════════════════════════════
#  场景生成
# ═══════════════════════════════════════════════════════════════

def place_enemy_on_path(
    grid: BattlefieldMap, path: list[Coord],
    enemy_range: int, enemy_fov: int, rng: random.Random,
) -> EnemySpec | None:
    if len(path) < 10:
        return None
    lo = int(len(path) * 0.3)
    hi = int(len(path) * 0.7)
    ref_idx = rng.randint(lo, hi)
    ref = path[ref_idx]
    look_ahead = min(ref_idx + 5, len(path) - 1)
    facing = angle_from_to(ref, path[look_ahead])
    path_set = set(path)
    candidates = []
    for dr in range(-8, 9):
        for dc in range(-8, 9):
            if dr == 0 and dc == 0:
                continue
            r, c = ref[0] + dr, ref[1] + dc
            if not grid.in_bounds(r, c) or grid.is_static_blocked(r, c):
                continue
            if (r, c) in path_set:
                continue
            dist = math.hypot(dr, dc)
            if dist <= enemy_range * 0.7:
                candidates.append((r, c))
    if not candidates:
        return None
    pos = rng.choice(candidates)
    jitter = rng.uniform(-25, 25)
    return EnemySpec(
        row=pos[0], col=pos[1],
        facing_deg=(facing + jitter) % 360.0,
        fov_deg=enemy_fov,
        max_range=enemy_range,
    )


def sample_cases(
    rng: random.Random, grid: BattlefieldMap,
    cases: int, dist_min: int, dist_max: int,
) -> list[tuple[Coord, Coord]]:
    cells = passable_cells(grid)
    sampled: list[tuple[Coord, Coord]] = []
    attempts = 0
    while len(sampled) < cases and attempts < cases * 5000:
        attempts += 1
        start = rng.choice(cells)
        goal = rng.choice(cells)
        if start == goal:
            continue
        dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if dist_min <= dist <= dist_max:
            path = bfs_path(grid, start, goal)
            if path and len(path) > 1:
                sampled.append((start, goal))
    return sampled


def generate_scenarios(
    grid: BattlefieldMap, rng: random.Random,
    cases_per_bucket: int, n_enemies: int,
    enemy_range: int, enemy_fov: int,
) -> list[dict[str, Any]]:
    """生成跨距离桶的评测场景。"""
    scenarios: list[dict[str, Any]] = []
    for bucket_name, (dist_min, dist_max) in DISTANCE_BUCKETS.items():
        pairs = sample_cases(rng, grid, cases_per_bucket, dist_min, dist_max)
        for start, goal in pairs:
            # A* 路径用于布敌
            plan = plan_global_path(grid, start, goal, max_height_diff=0,
                                    waypoint_interval=22, allow_diagonal=True)
            a_star_path = plan.path if plan.path else bfs_path(grid, start, goal)
            enemies: list[EnemySpec] = []
            for _ in range(n_enemies):
                e = place_enemy_on_path(grid, a_star_path, enemy_range, enemy_fov, rng)
                if e is not None:
                    enemies.append(e)
            scenarios.append({
                "bucket": bucket_name,
                "start": list(start),
                "goal": list(goal),
                "enemies": [
                    {"row": e.row, "col": e.col, "facing_deg": e.facing_deg,
                     "fov_deg": e.fov_deg, "max_range": e.max_range}
                    for e in enemies
                ],
                "a_star_path_len": len(a_star_path),
            })
    return scenarios


# ═══════════════════════════════════════════════════════════════
#  单 case 评测
# ═══════════════════════════════════════════════════════════════

def evaluate_case(
    grid: BattlefieldMap,
    scenario: dict[str, Any],
    model_path: str,
) -> dict[str, Any]:
    """对单个场景运行一次 plan_and_execute，返回完整指标。"""
    start = tuple(scenario["start"])
    goal = tuple(scenario["goal"])
    enemies = [
        EnemySpec(row=e["row"], col=e["col"], facing_deg=e["facing_deg"],
                  fov_deg=e["fov_deg"], max_range=e["max_range"])
        for e in scenario["enemies"]
    ]

    t0 = time.perf_counter()
    result = plan_and_execute(
        map_path=DEFAULT_MAP_PATH,
        start=start,
        goal=goal,
        enemies=enemies,
        model_path=model_path,
        use_fallback=True,
    )
    elapsed = time.perf_counter() - t0

    path = result.get("path", [])
    vis = path_visible_stats(grid, path, enemies)
    osc = oscillation_metrics(path)
    shortest = scenario.get("a_star_path_len", max(1, len(bfs_path(grid, start, goal))))
    total_steps = result.get("total_steps", max(0, len(path) - 1))
    path_eff = total_steps / max(1, shortest)

    return {
        "success": result.get("success", False),
        "failed_reason": result.get("failed_reason", ""),
        "total_steps": total_steps,
        "shortest_steps": shortest,
        "path_efficiency": round(path_eff, 3),
        "visible_ratio": round(vis["visible_ratio"], 4),
        "visible_steps": vis["visible_steps"],
        "displacement_eff": round(osc["displacement_eff"], 3),
        "reversal_rate": round(osc["reversal_rate"], 3),
        "repeat_ratio": round(osc["repeat_ratio"], 3),
        "fallback_count": result.get("geometric_fallback_count", 0),
        "fallback_steps": result.get("geometric_fallback_steps", 0),
        "local_executor_success": result.get("local_executor_success", False),
        "runtime_sec": round(elapsed, 2),
    }


# ═══════════════════════════════════════════════════════════════
#  汇总统计
# ═══════════════════════════════════════════════════════════════

def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    total = len(results)
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    def _avg(key: str, items: list[dict]) -> float:
        vals = [r[key] for r in items if key in r]
        return sum(vals) / max(1, len(vals))

    # 按失败原因分类
    fail_reasons: dict[str, int] = {}
    for r in failures:
        reason = r.get("failed_reason", "unknown") or "unknown"
        fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    return {
        "total": total,
        "success_count": len(successes),
        "success_rate": round(len(successes) / total, 4),
        "timeout_count": fail_reasons.get("max_fallback_steps_reached", 0)
                         + fail_reasons.get("max_fallback_events_reached", 0),
        "avg_path_efficiency": round(_avg("path_efficiency", successes), 3),
        "avg_visible_ratio": round(_avg("visible_ratio", results), 4),
        "avg_displacement_eff": round(_avg("displacement_eff", successes), 3),
        "avg_reversal_rate": round(_avg("reversal_rate", results), 3),
        "avg_repeat_ratio": round(_avg("repeat_ratio", results), 3),
        "avg_fallback_count": round(_avg("fallback_count", results), 2),
        "avg_fallback_steps": round(_avg("fallback_steps", results), 1),
        "local_only_success_rate": round(
            sum(1 for r in results if r.get("local_executor_success")) / total, 4
        ),
        "avg_runtime_sec": round(_avg("runtime_sec", results), 2),
        "fail_reasons": fail_reasons,
    }


def print_comparison(summaries: dict[str, dict[str, Any]]) -> None:
    """打印多模型对比表格。"""
    models = list(summaries.keys())
    metrics = [
        ("成功率",        "success_rate",        "{:>8.1%}"),
        ("纯本地成功率",  "local_only_success_rate", "{:>8.1%}"),
        ("路径效率",      "avg_path_efficiency",  "{:>8.3f}"),
        ("可见率",        "avg_visible_ratio",    "{:>8.2%}"),
        ("位移效率",      "avg_displacement_eff", "{:>8.3f}"),
        ("反向率",        "avg_reversal_rate",    "{:>8.3f}"),
        ("重复率",        "avg_repeat_ratio",     "{:>8.3f}"),
        ("平均fallback",  "avg_fallback_count",   "{:>8.1f}"),
        ("耗时(s)",       "avg_runtime_sec",      "{:>8.1f}"),
    ]

    # 表头
    col_w = 14
    header = f"{'指标':<14}" + "".join(f"{m:>{col_w}}" for m in models)
    print(f"\n{'═' * len(header)}")
    print("  六模型综合评测对比")
    print(f"{'═' * len(header)}")
    print(header)
    print("─" * len(header))

    for label, key, fmt in metrics:
        row = f"{label:<14}"
        for m in models:
            val = summaries[m].get(key, 0)
            row += f"{fmt.format(val):>{col_w}}"
        print(row)

    print("─" * len(header))

    # 失败原因汇总
    print("\n失败原因分布:")
    for m in models:
        reasons = summaries[m].get("fail_reasons", {})
        if reasons:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
            print(f"  {m:<24} {parts}")
        else:
            print(f"  {m:<24} (全部成功)")


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Comprehensive multi-model evaluation")
    p.add_argument("--map", default=DEFAULT_MAP_PATH, help="terrain map txt")
    p.add_argument("--models", nargs="+", default=ALL_MODELS,
                   help="model checkpoint filenames")
    p.add_argument("--cases", type=int, default=20,
                   help="cases per distance bucket (total = cases * 3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-enemies", type=int, default=2, help="enemies per case")
    p.add_argument("--enemy-range", type=int, default=25)
    p.add_argument("--enemy-fov", type=int, default=90)
    p.add_argument("--output-dir", default="outputs")
    return p


def main() -> None:
    args = build_parser().parse_args()
    grid = load_txt_map(args.map)
    rng = random.Random(args.seed)

    # 生成场景
    print(f"生成场景: {args.cases} case/桶 × 3 桶 = {args.cases * 3} 个场景")
    scenarios = generate_scenarios(
        grid, rng, args.cases, args.n_enemies, args.enemy_range, args.enemy_fov,
    )
    print(f"实际生成: {len(scenarios)} 个场景")

    # 评测所有模型
    all_summaries: dict[str, dict[str, Any]] = {}
    all_details: dict[str, list[dict[str, Any]]] = {}
    timeout_scenarios: list[dict[str, Any]] = []

    for model in args.models:
        print(f"\n{'─' * 60}")
        print(f"  评测模型: {model}")
        print(f"{'─' * 60}")

        model_results: list[dict[str, Any]] = []
        for i, sc in enumerate(scenarios):
            result = evaluate_case(grid, sc, model)
            result["case_index"] = i
            result["bucket"] = sc["bucket"]
            model_results.append(result)

            status = "OK" if result["success"] else "XX"
            print(f"  [{i+1:3d}/{len(scenarios)}] {sc['bucket']:>6} "
                  f"start={sc['start']} goal={sc['goal']} "
                  f"steps={result['total_steps']:>4} "
                  f"eff={result['path_efficiency']:.2f} "
                  f"vis={result['visible_ratio']:.2%} "
                  f"osc={result['displacement_eff']:.2f} "
                  f"fb={result['fallback_count']} "
                  f"{status}")

            # 收集失败场景
            if not result["success"]:
                timeout_scenarios.append({
                    "model": model,
                    "scenario": sc,
                    "result": {k: v for k, v in result.items()
                               if k not in ("runtime_sec",)},
                })

        all_details[model] = model_results
        all_summaries[model] = summarize(model_results)

    # 打印对比表
    print_comparison(all_summaries)

    # 保存结果
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 详细结果 JSON
    detail_path = out_dir / f"eval_comprehensive_{ts}.json"
    detail_path.write_text(json.dumps({
        "meta": {
            "map": args.map,
            "models": args.models,
            "cases_per_bucket": args.cases,
            "seed": args.seed,
            "n_enemies": args.n_enemies,
            "enemy_range": args.enemy_range,
            "enemy_fov": args.enemy_fov,
            "total_scenarios": len(scenarios),
            "timestamp": ts,
        },
        "summaries": all_summaries,
        "details": all_details,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细结果已保存: {detail_path}")

    # 保存超时/失败场景
    if timeout_scenarios:
        scenario_dir = out_dir / "timeout_scenarios"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # 汇总文件
        all_path = scenario_dir / f"all_failures_{ts}.json"
        all_path.write_text(json.dumps({
            "total_failures": len(timeout_scenarios),
            "scenarios": timeout_scenarios,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"失败场景已保存: {all_path} ({len(timeout_scenarios)} 个)")

        # 按模型分目录
        for item in timeout_scenarios:
            model_name = Path(item["model"]).stem
            model_dir = scenario_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            case_idx = item["result"]["case_index"]
            bucket = item["result"]["bucket"]
            case_path = model_dir / f"fail_{bucket}_{case_idx:03d}.json"
            case_path.write_text(json.dumps({
                "model": item["model"],
                "scenario": item["scenario"],
                "result": item["result"],
            }, ensure_ascii=False, indent=2), encoding="utf-8")

        # 统计哪些场景是所有模型都失败的
        scenario_fail_count: dict[str, int] = {}
        for item in timeout_scenarios:
            key = f"{item['scenario']['start']}_{item['scenario']['goal']}"
            scenario_fail_count[key] = scenario_fail_count.get(key, 0) + 1

        universal_fails = {k: v for k, v in scenario_fail_count.items()
                           if v >= len(args.models) * 0.8}
        if universal_fails:
            print(f"\n[!] {len(universal_fails)} 个场景 >80% 模型都失败（困难场景候选）")
    else:
        print("\n[OK] 全部场景均成功，无失败场景需要保存。")

    # Stage 4 建议
    print(f"\n{'═' * 60}")
    print("  Stage 4 课程训练建议")
    print(f"{'═' * 60}")
    if timeout_scenarios:
        unique_scenarios = set()
        for item in timeout_scenarios:
            key = (tuple(item["scenario"]["start"]), tuple(item["scenario"]["goal"]))
            unique_scenarios.add(key)
        print(f"  共收集 {len(unique_scenarios)} 个独立困难场景")
        print(f"  建议：将 outputs/timeout_scenarios/ 作为困难场景库")
        print(f"  在 LocalSceneSampler 中增加 hard_scenario_dir 参数")
        print(f"  Stage 4 训练时以一定概率从困难场景库中采样")
    else:
        print("  当前模型已全部通过评测，暂不需要 Stage 4。")


if __name__ == "__main__":
    main()
