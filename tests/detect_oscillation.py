"""
路径振荡检测工具

对 episode_8000.pt 模型在多个场景上的局部执行路径做振荡分析。

功能:
  1. 复用 LocalSceneSampler 生成可复现的测试场景
  2. 在每个场景上用 greedy_action 跑模型，记录完整路径
  3. 对路径做多维度振荡检测 (滑窗/折返/循环/位移效率/推进分析)
  4. 输出可读报告 + 可选 JSON

用法:
  python tests/detect_oscillation.py
  python tests/detect_oscillation.py --cases 200 --stage 3 --seed 42
  python tests/detect_oscillation.py --model episode_5800.pt --cases 50
  python tests/detect_oscillation.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import random
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

Coord = tuple[int, int]

# 8个方向动作的反向映射：上↔下，左↔右，左上↔右下，右上↔左下
REVERSE_MAP = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 5: 6, 6: 5, 7: 4}


# ═══════════════════════════════════════════════════════════════
#  模型加载 & 场景执行 (复用项目逻辑)
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
def greedy_action(
    model: TacticalD3QN,
    obs: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    prev_action: int | None = None,
    return_penalty: float = 2.0,
) -> int:
    x = torch.from_numpy(obs).unsqueeze(0).float().to(device)
    m = torch.from_numpy(mask).unsqueeze(0).float().to(device)
    q = model(x)
    # 对上一步的反方向施加惩罚，抑制振荡
    if prev_action is not None:
        reverse_action = REVERSE_MAP[prev_action]
        q[0, reverse_action] -= return_penalty
    q = masked_q_values(q, m)
    return int(torch.argmax(q, dim=1).item())


def run_episode_with_trace(
    model: TacticalD3QN,
    device: torch.device,
    grid: BattlefieldMap,
    start: Coord,
    goal: Coord,
    enemies: list[EnemySpec],
    max_steps: int = 50,
) -> dict[str, Any]:
    """在单个场景上跑模型，返回完整路径和终止原因。"""
    env_cfg = EnvConfig(timeout_steps=max_steps, no_progress_limit=max_steps + 1)
    env = TacticalBattlefieldEnv(grid, env_cfg)
    obs = env.reset(start=start, goal=goal, enemies=enemies, waypoints=[goal], threat_scale=1.0)

    path: list[Coord] = [start]
    reason = "step_limit"
    prev_action = None

    for _ in range(max_steps):
        action = greedy_action(model, obs, env.action_mask(), device, prev_action)
        prev_action = action
        step = env.step(action)
        obs = step.observation
        path.append(env.pos)

        if env.pos == goal:
            reason = "reached"
            break
        # 与 run_segment 相同的振荡检测逻辑
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
    }


# ═══════════════════════════════════════════════════════════════
#  振荡检测核心算法
# ═══════════════════════════════════════════════════════════════

def detect_sliding_window_oscillation(
    path: Sequence[Coord],
    window: int = 10,
    unique_limit: int = 3,
) -> list[dict[str, Any]]:
    """滑动窗口: 窗口内去重坐标 <= unique_limit 判定为振荡。"""
    if len(path) < window:
        return []
    regions: list[dict[str, Any]] = []
    in_region = False
    region_start = 0
    for i in range(len(path) - window + 1):
        seg = path[i : i + window]
        if len(set(seg)) <= unique_limit:
            if not in_region:
                in_region = True
                region_start = i
        else:
            if in_region:
                end = i + window - 1
                rpath = path[region_start : end + 1]
                regions.append({
                    "start_idx": region_start,
                    "end_idx": end,
                    "length": end - region_start + 1,
                    "unique_cells": len(set(rpath)),
                    "cells": list(dict.fromkeys(rpath)),
                })
                in_region = False
    if in_region:
        end = len(path) - 1
        rpath = path[region_start : end + 1]
        regions.append({
            "start_idx": region_start, "end_idx": end,
            "length": end - region_start + 1,
            "unique_cells": len(set(rpath)),
            "cells": list(dict.fromkeys(rpath)),
        })
    return regions


def detect_back_and_forth(path: Sequence[Coord]) -> list[dict[str, Any]]:
    """检测 A→B→A 即时折返。"""
    out = []
    for i in range(len(path) - 2):
        if path[i] == path[i + 2] and path[i] != path[i + 1]:
            out.append({"idx": i, "a": path[i], "b": path[i + 1]})
    return out


def detect_loop_cycles(path: Sequence[Coord], max_cycle: int = 6) -> list[dict[str, Any]]:
    """检测长度 2~max_cycle 的循环模式。"""
    found: dict[tuple[Coord, ...], int] = {}
    for clen in range(2, max_cycle + 1):
        for i in range(len(path) - clen * 2 + 1):
            cand = tuple(path[i : i + clen])
            nxt = tuple(path[i + clen : i + clen * 2])
            if cand == nxt:
                found[cand] = found.get(cand, 0) + 1
    return sorted(
        [{"pattern": list(k), "length": len(k), "occurrences": v} for k, v in found.items()],
        key=lambda x: -x["occurrences"],
    )


def direction_stats(path: Sequence[Coord]) -> dict[str, Any]:
    """方向变化 & 反向频率。"""
    if len(path) < 3:
        return {"changes": 0, "reversals": 0, "change_rate": 0.0, "reversal_rate": 0.0}
    deltas = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    changes = sum(1 for i in range(len(deltas)-1) if deltas[i] != deltas[i+1])
    reversals = sum(
        1 for i in range(len(deltas)-1)
        if deltas[i] == (-deltas[i+1][0], -deltas[i+1][1])
    )
    n = max(1, len(deltas) - 1)
    return {"changes": changes, "reversals": reversals,
            "change_rate": changes / n, "reversal_rate": reversals / n}


def displacement_efficiency(path: Sequence[Coord]) -> dict[str, Any]:
    """位移效率 = 净位移 / 累计距离。接近 0 = 原地打转。"""
    if len(path) < 2:
        return {"total_steps": 0, "net": 0, "cumulative": 0,
                "efficiency": 0.0, "wasted": 0, "wasted_ratio": 0.0}
    total = len(path) - 1
    net = abs(path[0][0]-path[-1][0]) + abs(path[0][1]-path[-1][1])
    cum = sum(abs(a[0]-b[0])+abs(a[1]-b[1]) for a, b in zip(path, path[1:]))
    wasted = cum - net
    return {
        "total_steps": total, "net": net, "cumulative": cum,
        "efficiency": net / max(1, cum),
        "wasted": wasted, "wasted_ratio": wasted / max(1, cum),
    }


def visit_hotspots(path: Sequence[Coord], top_n: int = 10) -> list[dict[str, Any]]:
    """高频重复访问的格子。"""
    return [
        {"cell": list(c), "visits": n}
        for c, n in Counter(path).most_common(top_n)
        if n > 1
    ]


def progress_windows(
    path: Sequence[Coord], goal: Coord, window: int = 10,
) -> list[dict[str, Any]]:
    """切分路径为窗口，分析每窗口对目标的推进。"""
    if len(path) < window:
        return []
    out = []
    for i in range(0, len(path) - window + 1, window):
        seg = path[i : i + window]
        d0 = abs(seg[0][0]-goal[0]) + abs(seg[0][1]-goal[1])
        d1 = abs(seg[-1][0]-goal[0]) + abs(seg[-1][1]-goal[1])
        uniq = len(set(seg))
        out.append({
            "start_idx": i, "end_idx": min(i+window-1, len(path)-1),
            "from": list(seg[0]), "to": list(seg[-1]),
            "dist_start": d0, "dist_end": d1,
            "delta": d0 - d1,
            "unique": uniq, "stagnant": uniq <= 3,
        })
    return out


def _severity(osc_ratio: float, rev_rate: float, eff: float) -> str:
    s = 0
    if osc_ratio > 0.5: s += 3
    elif osc_ratio > 0.2: s += 2
    elif osc_ratio > 0.05: s += 1
    if rev_rate > 0.3: s += 2
    elif rev_rate > 0.15: s += 1
    if eff < 0.15: s += 3
    elif eff < 0.3: s += 2
    elif eff < 0.5: s += 1
    if s >= 6: return "severe"
    if s >= 4: return "moderate"
    if s >= 2: return "mild"
    return "none"


def analyze_path(
    path: Sequence[Coord],
    goal: Coord | None = None,
    window: int = 10,
    unique_limit: int = 3,
) -> dict[str, Any]:
    """对单条路径做全套振荡检测。"""
    p = [tuple(c) for c in path]
    sw = detect_sliding_window_oscillation(p, window, unique_limit)
    bf = detect_back_and_forth(p)
    ds = direction_stats(p)
    de = displacement_efficiency(p)
    osc_steps = sum(r["length"] for r in sw)
    osc_ratio = osc_steps / max(1, len(p))

    report: dict[str, Any] = {
        "path_length": len(p),
        "has_oscillation": (
            len(sw) > 0 or len(bf) > len(p) * 0.1
            or ds["reversal_rate"] > 0.3 or de["efficiency"] < 0.3
        ),
        "severity": _severity(osc_ratio, ds["reversal_rate"], de["efficiency"]),
        "sliding_window": {
            "window": window, "unique_limit": unique_limit,
            "regions": sw, "region_count": len(sw),
            "osc_steps": osc_steps, "osc_ratio": osc_ratio,
        },
        "back_and_forth": {"count": len(bf), "ratio": len(bf)/max(1, len(p)-2), "samples": bf[:10]},
        "loop_cycles": detect_loop_cycles(p)[:10],
        "direction": ds,
        "displacement": de,
        "hotspots": visit_hotspots(p),
    }
    if goal is not None:
        report["progress"] = progress_windows(p, goal, window)
    return report


# ═══════════════════════════════════════════════════════════════
#  打印报告
# ═══════════════════════════════════════════════════════════════

_SEV_ICON = {"severe": "[!!!]", "moderate": "[!! ]", "mild": "[!  ]", "none": "[   ]"}


def _print_path_map(path: Sequence[Coord], osc_regions: list[dict[str, Any]], max_w: int = 60) -> None:
    """ASCII 路径地图, 振荡区域用 x 标记。"""
    if len(path) < 2:
        return
    p = [tuple(c) for c in path]
    rs = [c[0] for c in p]
    cs = [c[1] for c in p]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    sc = max(1, max((h + max_w - 1) // max_w, (w + max_w - 1) // max_w))
    gh, gw = (h + sc - 1) // sc, (w + sc - 1) // sc
    grid = [[" "] * gw for _ in range(gh)]

    osc_set: set[int] = set()
    for r in osc_regions:
        osc_set.update(range(r["start_idx"], r["end_idx"] + 1))

    for idx, (r, c) in enumerate(p):
        gr, gc = (r - r0) // sc, (c - c0) // sc
        grid[gr][gc] = "x" if idx in osc_set else "."

    sr, sc_ = (p[0][0]-r0)//sc, (p[0][1]-c0)//sc
    er, ec = (p[-1][0]-r0)//sc, (p[-1][1]-c0)//sc
    grid[sr][sc_] = "S"
    grid[er][ec] = "G" if grid[er][ec] != "S" else "="

    print(f"  Map (1:{sc}, S=start G=goal .=path x=osc):")
    print(f"  +{'-'*gw}+")
    for row in grid:
        print(f"  |{''.join(row)}|")
    print(f"  +{'-'*gw}+")


def print_case_report(idx: int, result: dict[str, Any], report: dict[str, Any]) -> None:
    """打印单个 case 的振荡报告。"""
    sev = report["severity"]
    icon = _SEV_ICON.get(sev, "[?]")
    meta = result["meta"]
    print(f"\n{'─'*60}")
    print(f"  Case {idx}: {meta['start']} -> {meta['goal']}  "
          f"enemies={meta['enemy_count']}  source={meta['source']}")
    print(f"  {icon}  severity={sev.upper()}  "
          f"steps={report['path_length']}  reason={result['reason']}  "
          f"success={result['success']}")

    sw = report["sliding_window"]
    print(f"  SlidingWindow(w={sw['window']},u<={sw['unique_limit']}): "
          f"{sw['region_count']} regions, {sw['osc_steps']} steps ({sw['osc_ratio']:.1%})")
    if sw["regions"]:
        for r in sw["regions"][:4]:
            cells = "→".join(f"({c[0]},{c[1]})" for c in r["cells"][:5])
            if len(r["cells"]) > 5:
                cells += "→..."
            print(f"    [{r['start_idx']}-{r['end_idx']}] len={r['length']} "
                  f"unique={r['unique_cells']}: {cells}")

    bf = report["back_and_forth"]
    if bf["count"] > 0:
        print(f"  BackForth(A→B→A): {bf['count']} ({bf['ratio']:.1%})")

    ds = report["direction"]
    print(f"  Direction: changes={ds['changes']}({ds['change_rate']:.2f}) "
          f"reversals={ds['reversals']}({ds['reversal_rate']:.2f})")

    de = report["displacement"]
    print(f"  Displacement: net={de['net']} cum={de['cumulative']} "
          f"eff={de['efficiency']:.3f} wasted={de['wasted']}({de['wasted_ratio']:.1%})")

    hs = report.get("hotspots", [])
    if hs:
        top = ", ".join(f"({h['cell'][0]},{h['cell'][1]})x{h['visits']}" for h in hs[:5])
        print(f"  Hotspots: {top}")

    pw = report.get("progress", [])
    if pw:
        stagnant = [w for w in pw if w["stagnant"]]
        regress = [w for w in pw if w["delta"] < 0]
        if stagnant or regress:
            print(f"  Progress: {len(pw)} windows, "
                  f"stagnant={len(stagnant)}, regressing={len(regress)}")


def print_summary(reports: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    """打印总体统计。"""
    total = len(reports)
    osc = [r for r in reports if r["has_oscillation"]]
    severe = [r for r in reports if r["severity"] == "severe"]
    moderate = [r for r in reports if r["severity"] == "moderate"]
    mild = [r for r in reports if r["severity"] == "mild"]

    reached = sum(1 for r in results if r["reason"] == "reached")
    osc_stopped = sum(1 for r in results if r["reason"] == "oscillation")

    avg_eff = sum(r["displacement"]["efficiency"] for r in reports) / max(1, total)
    avg_osc_ratio = sum(r["sliding_window"]["osc_ratio"] for r in reports) / max(1, total)
    avg_rev = sum(r["direction"]["reversal_rate"] for r in reports) / max(1, total)
    avg_wasted = sum(r["displacement"]["wasted_ratio"] for r in reports) / max(1, total)

    print(f"\n{'═'*60}")
    print(f"  SUMMARY  ({total} cases)")
    print(f"{'═'*60}")
    print(f"  Reached goal     : {reached}/{total} ({reached/max(1,total):.1%})")
    print(f"  Oscillation stop : {osc_stopped}/{total}")
    print(f"  Has oscillation  : {len(osc)}/{total} ({len(osc)/max(1,total):.1%})")
    print(f"    severe         : {len(severe)}")
    print(f"    moderate       : {len(moderate)}")
    print(f"    mild           : {len(mild)}")
    print(f"  Avg displacement eff : {avg_eff:.3f}")
    print(f"  Avg oscillation ratio: {avg_osc_ratio:.3f}")
    print(f"  Avg reversal rate    : {avg_rev:.3f}")
    print(f"  Avg wasted ratio     : {avg_wasted:.1%}")

    # 振荡原因分析
    print(f"\n  --- Oscillation Root Cause Hints ---")
    low_eff_cases = [r for r in reports if r["displacement"]["efficiency"] < 0.3]
    high_rev_cases = [r for r in reports if r["direction"]["reversal_rate"] > 0.2]
    hotspot_cases = [r for r in reports if any(h["visits"] >= 4 for h in r.get("hotspots", []))]
    print(f"  Low efficiency (<0.3)     : {len(low_eff_cases)}/{total}")
    print(f"  High reversal rate (>0.2) : {len(high_rev_cases)}/{total}")
    print(f"  Severe hotspots (>=4 visits): {len(hotspot_cases)}/{total}")

    # 常见振荡模式
    all_cycles: dict[str, int] = {}
    for r in reports:
        for c in r.get("loop_cycles", []):
            key = f"{c['length']}-cycle"
            all_cycles[key] = all_cycles.get(key, 0) + c["occurrences"]
    if all_cycles:
        print(f"  Loop patterns: {dict(sorted(all_cycles.items(), key=lambda x:-x[1]))}")

    print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def build_scenes(
    base_grid: BattlefieldMap,
    cases: int,
    stage: int,
    seed: int,
) -> list:
    """复用 LocalSceneSampler 生成可复现场景。"""
    random.seed(seed)
    np.random.seed(seed)
    cfg = TrainingSceneConfig(enabled=True)
    sampler = LocalSceneSampler(base_grid, cfg, EnvConfig().window_size)
    return [sampler.sample(stage=stage) for _ in range(cases)]


def run_all(
    map_path: str,
    model_path: str | Path,
    cases: int,
    stage: int,
    seed: int,
    max_steps: int,
    window: int,
    unique_limit: int,
    show_maps: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """跑全部场景并分析。返回 (results, reports)。"""
    base_grid = load_txt_map(map_path)
    scenes = build_scenes(base_grid, cases, stage, seed)
    device = select_device()

    resolved = Path(model_path)
    if not resolved.is_absolute() or not resolved.exists():
        resolved = PROJECT_ROOT / model_path
    model = load_policy(resolved, device)
    print(f"Model: {resolved}  Device: {device}  Cases: {cases}  Stage: {stage}")

    results: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for i, scene in enumerate(scenes):
        t0 = time.perf_counter()
        episode = run_episode_with_trace(
            model, device, scene.grid, scene.start, scene.goal,
            scene.enemies, max_steps,
        )
        elapsed = time.perf_counter() - t0

        path = [tuple(c) for c in episode["path"]]
        goal = scene.goal

        result = {
            "case": i,
            "meta": {
                "start": list(scene.start),
                "goal": list(scene.goal),
                "source": scene.source,
                "enemy_case": scene.enemy_case,
                "enemy_count": len(scene.enemies),
            },
            "reason": episode["reason"],
            "success": episode["success"],
            "steps": len(path) - 1,
            "runtime_ms": elapsed * 1000,
            "path": [list(c) for c in path],
        }
        results.append(result)

        report = analyze_path(path, goal=goal, window=window, unique_limit=unique_limit)
        reports.append(report)

        print_case_report(i, result, report)
        if show_maps:
            _print_path_map(path, report["sliding_window"]["regions"])

    return results, reports


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detect oscillation in model local execution paths")
    p.add_argument("--map", default=DEFAULT_MAP_PATH, help="terrain map txt")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, help="model checkpoint")
    p.add_argument("--cases", type=int, default=50, help="number of test scenes")
    p.add_argument("--stage", type=int, default=3, choices=[1,2,3], help="curriculum stage for scene generation")
    p.add_argument("--seed", type=int, default=20260527, help="random seed")
    p.add_argument("--max-steps", type=int, default=50, help="max RL steps per episode")
    p.add_argument("--window", type=int, default=10, help="sliding window size")
    p.add_argument("--unique-limit", type=int, default=3, help="unique cell threshold")
    p.add_argument("--show-map", action="store_true", help="print ASCII path map per case")
    p.add_argument("--output-dir", default=None, help="save JSON report")
    return p


def main() -> None:
    args = build_parser().parse_args()

    results, reports = run_all(
        map_path=args.map,
        model_path=args.model,
        cases=args.cases,
        stage=args.stage,
        seed=args.seed,
        max_steps=args.max_steps,
        window=args.window,
        unique_limit=args.unique_limit,
        show_maps=args.show_map,
    )

    print_summary(reports, results)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = out_dir / f"oscillation_report_{ts}.json"
        payload = {
            "map": args.map,
            "model": str(args.model),
            "cases": args.cases,
            "stage": args.stage,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "window": args.window,
            "unique_limit": args.unique_limit,
            "results": results,
            "analysis": reports,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
