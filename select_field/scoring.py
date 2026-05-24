from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from .geometry import euclidean
from .map_loader import MapGrid
from .models import CandidateScore, Point, Rect, TargetSpec
from .visibility import VisibilityCache


@dataclass(frozen=True)
class SamplingConfig:
    max_candidates: int = 1200
    max_target_samples: int = 1500
    patch_size: int = 5
    top_k: int = 5


@dataclass(frozen=True)
class PositionSearchConfig:
    coarse_candidates_limit: int = 300
    fine_refine_radius: int = 2
    coarse_step_target: int = 2500
    min_clearance_ratio: float = 0.8


def _adaptive_step(rect: Rect, max_samples: int) -> int:
    area = max(1, rect.w * rect.h)
    if area <= max_samples:
        return 1
    return max(1, int(math.ceil(math.sqrt(area / max_samples))))


def _sample_points(rect: Rect, max_samples: int) -> list[Point]:
    step = _adaptive_step(rect, max_samples)
    points = list(rect.iter_cells(step=step))
    if not points:
        return []

    edge_point = Point(rect.right - 1, rect.bottom - 1)
    if points[-1] != edge_point:
        points.append(edge_point)
    return points


def _coverage_score(
    visibility: VisibilityCache,
    source: Point,
    target_points: list[Point],
) -> tuple[float, dict[str, float]]:
    visible = sum(1 for point in target_points if visibility.line_of_sight(source, point))
    total = max(1, len(target_points))
    ratio = visible / total
    return ratio, {
        "visible_count": float(visible),
        "visible_ratio": ratio,
        "sample_count": float(total),
    }


def select_lookout_points(
    grid: MapGrid,
    region_a: Rect,
    region_b: Rect,
    *,
    top_k: int = 5,
) -> dict:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    region_a = region_a.clip(grid.width, grid.height)
    region_b = region_b.clip(grid.width, grid.height)
    visibility = VisibilityCache(grid, [])

    candidates_a = [p for p in _sample_points(region_a, max_samples=1200) if grid.point_passable(p)]
    candidates_b = [p for p in _sample_points(region_b, max_samples=1200) if grid.point_passable(p)]
    targets_a = [p for p in _sample_points(region_a, max_samples=1500) if grid.point_passable(p)]
    targets_b = [p for p in _sample_points(region_b, max_samples=1500) if grid.point_passable(p)]

    scored_a: list[CandidateScore] = []
    for point in candidates_a:
        score, details = _coverage_score(visibility, point, targets_b)
        scored_a.append(CandidateScore(rect=Rect(point.x, point.y, 1, 1), score=score, details=details))

    scored_b: list[CandidateScore] = []
    for point in candidates_b:
        score, details = _coverage_score(visibility, point, targets_a)
        scored_b.append(CandidateScore(rect=Rect(point.x, point.y, 1, 1), score=score, details=details))

    scored_a.sort(key=lambda item: item.score, reverse=True)
    scored_b.sort(key=lambda item: item.score, reverse=True)

    top_candidates = [
        {"side": "A", **cand.to_dict()} for cand in scored_a[:top_k]
    ] + [
        {"side": "B", **cand.to_dict()} for cand in scored_b[:top_k]
    ]
    top_candidates.sort(key=lambda item: item["score"], reverse=True)

    return {
        "best_in_a_for_b": scored_a[0].to_dict() if scored_a else None,
        "best_in_b_for_a": scored_b[0].to_dict() if scored_b else None,
        "top_k": top_candidates[:top_k],
        "meta": {
            "region_a": region_a.to_dict(),
            "region_b": region_b.to_dict(),
            "sample_config": {
                "candidates_a": len(candidates_a),
                "candidates_b": len(candidates_b),
                "targets_a": len(targets_a),
                "targets_b": len(targets_b),
            },
        },
    }


def _build_binary_integral(grid: MapGrid) -> list[list[int]]:
    integral = [[0] * (grid.width + 1) for _ in range(grid.height + 1)]
    for y in range(grid.height):
        row_sum = 0
        for x in range(grid.width):
            row_sum += 1 if grid.is_passable(x, y) else 0
            integral[y + 1][x + 1] = integral[y][x + 1] + row_sum
    return integral


def _rect_sum(integral: list[list[int]], rect: Rect) -> int:
    return (
        integral[rect.bottom][rect.right]
        - integral[rect.y][rect.right]
        - integral[rect.bottom][rect.x]
        + integral[rect.y][rect.x]
    )


def _patch_clearance_ratio(integral: list[list[int]], rect: Rect) -> float:
    total = max(1, rect.w * rect.h)
    return _rect_sum(integral, rect) / total


def _ring_openness_ratio(integral: list[list[int]], grid: MapGrid, rect: Rect) -> float:
    outer = Rect(rect.x - 1, rect.y - 1, rect.w + 2, rect.h + 2).clip(grid.width, grid.height)
    inner = rect.clip(grid.width, grid.height)
    ring_area = outer.w * outer.h - inner.w * inner.h
    if ring_area <= 0:
        return 0.0

    outer_clear = _rect_sum(integral, outer)
    inner_clear = _rect_sum(integral, inner)
    return max(0.0, (outer_clear - inner_clear) / ring_area)


def _distance_band_score(distance: float, low: float, high: float) -> float:
    if distance < low:
        return distance / max(low, 1.0)
    if distance > high:
        overflow = distance - high
        return max(0.0, 1.0 - overflow / max(high, 1.0))
    return 1.0


def _candidate_seed_score(
    patch: Rect,
    targets: list[TargetSpec],
    clearance_ratio: float,
    direct_openness: float,
    grid: MapGrid,
) -> float:
    center = patch.center
    score = 0.8 * clearance_ratio + 0.4 * direct_openness
    for target in targets:
        distance = euclidean(center, target.enemy)
        if target.type.upper() == "ZM":
            score += 1.0 / (distance + 1.0)
        else:
            score += 1.2 * _distance_band_score(distance, low=12.0, high=max(24.0, grid.width * 0.4))
    return score


def _estimate_coarse_step(search_region: Rect, config: PositionSearchConfig) -> int:
    area = max(1, search_region.w * search_region.h)
    return max(1, int(math.ceil(math.sqrt(area / max(1, config.coarse_step_target)))))


def _iter_patch_origins(search_region: Rect, patch_size: int, step: int):
    max_x = search_region.right - patch_size
    max_y = search_region.bottom - patch_size
    if max_x < search_region.x or max_y < search_region.y:
        return

    seen: set[tuple[int, int]] = set()
    y = search_region.y
    while y <= max_y:
        x = search_region.x
        while x <= max_x:
            key = (x, y)
            seen.add(key)
            yield key
            x += step
        y += step

    for x in range(search_region.x, max_x + 1):
        key = (x, max_y)
        if key not in seen:
            seen.add(key)
            yield key
    for y in range(search_region.y, max_y + 1):
        key = (max_x, y)
        if key not in seen:
            seen.add(key)
            yield key


def _coarse_candidate_rects(
    grid: MapGrid,
    search_region: Rect,
    targets: list[TargetSpec],
    patch_size: int,
    integral: list[list[int]],
    config: PositionSearchConfig,
) -> list[Rect]:
    step = _estimate_coarse_step(search_region, config)
    heap: list[tuple[float, int, int]] = []

    for x, y in _iter_patch_origins(search_region, patch_size, step):
        patch = Rect(x, y, patch_size, patch_size)
        clearance_ratio = _patch_clearance_ratio(integral, patch)
        if clearance_ratio < config.min_clearance_ratio:
            continue

        direct_openness = _ring_openness_ratio(integral, grid, patch)
        seed_score = _candidate_seed_score(patch, targets, clearance_ratio, direct_openness, grid)
        item = (seed_score, x, y)
        if len(heap) < config.coarse_candidates_limit:
            heapq.heappush(heap, item)
        else:
            heapq.heappushpop(heap, item)

    coarse = [Rect(x, y, patch_size, patch_size) for _, x, y in heap]
    coarse.sort(key=lambda rect: (rect.y, rect.x))
    return coarse


def _refine_candidate_rects(
    grid: MapGrid,
    search_region: Rect,
    patch_size: int,
    integral: list[list[int]],
    coarse_rects: list[Rect],
    config: PositionSearchConfig,
) -> list[Rect]:
    max_x = search_region.right - patch_size
    max_y = search_region.bottom - patch_size
    candidates: dict[tuple[int, int], Rect] = {}

    for coarse in coarse_rects:
        for dy in range(-config.fine_refine_radius, config.fine_refine_radius + 1):
            for dx in range(-config.fine_refine_radius, config.fine_refine_radius + 1):
                x = coarse.x + dx
                y = coarse.y + dy
                if x < search_region.x or x > max_x or y < search_region.y or y > max_y:
                    continue

                patch = Rect(x, y, patch_size, patch_size)
                if _patch_clearance_ratio(integral, patch) >= config.min_clearance_ratio:
                    candidates[(x, y)] = patch

    if not candidates:
        for x, y in _iter_patch_origins(search_region, patch_size, 1):
            patch = Rect(x, y, patch_size, patch_size)
            if _patch_clearance_ratio(integral, patch) >= config.min_clearance_ratio:
                candidates[(x, y)] = patch

    return list(candidates.values())


def _zm_score(
    patch: Rect,
    targets: list[TargetSpec],
    visibility: VisibilityCache,
    clearance_ratio: float,
    direct_openness: float,
) -> CandidateScore:
    center = patch.center
    score = 0.8 * clearance_ratio + 0.4 * direct_openness
    details: dict[str, float] = {
        "clearance_ratio": clearance_ratio,
        "direct_openness": direct_openness,
        "patch_size": float(patch.w * patch.h),
    }

    for idx, target in enumerate(targets):
        if target.type.upper() != "ZM":
            continue
        enemy_index = visibility.enemy_index(target.enemy)
        visible = visibility.visible_between(enemy_index, center)
        distance = euclidean(center, target.enemy)
        score += (2.5 if visible else -3.0) + (1.0 / (distance + 1.0)) + clearance_ratio + 0.5 * direct_openness
        details[f"target_{idx}_zm_visible"] = 1.0 if visible else 0.0
        details[f"target_{idx}_distance"] = distance

    return CandidateScore(rect=patch, score=score, details=details)


def _jm_score(
    grid: MapGrid,
    patch: Rect,
    targets: list[TargetSpec],
    visibility: VisibilityCache,
    clearance_ratio: float,
    direct_openness: float,
) -> CandidateScore:
    center = patch.center
    exposure = visibility.visible_to_any_enemy(center)
    hidden_bonus = 1.0 if not exposure else 0.0
    score = 0.8 * clearance_ratio + 0.4 * direct_openness
    details: dict[str, float] = {
        "clearance_ratio": clearance_ratio,
        "direct_openness": direct_openness,
        "patch_size": float(patch.w * patch.h),
    }

    for idx, target in enumerate(targets):
        if target.type.upper() != "JM":
            continue
        distance = euclidean(center, target.enemy)
        distance_score = _distance_band_score(distance, low=12.0, high=max(24.0, grid.width * 0.4))
        score += 1.8 * hidden_bonus + 1.2 * distance_score + 0.8 * clearance_ratio + 0.5 * direct_openness
        details[f"target_{idx}_jm_hidden"] = hidden_bonus
        details[f"target_{idx}_distance"] = distance

    return CandidateScore(rect=patch, score=score, details=details)


def _score_candidate_patch(
    grid: MapGrid,
    integral: list[list[int]],
    visibility: VisibilityCache,
    patch: Rect,
    targets: list[TargetSpec],
) -> tuple[CandidateScore, CandidateScore | None, CandidateScore | None]:
    clearance_ratio = _patch_clearance_ratio(integral, patch)
    direct_openness = _ring_openness_ratio(integral, grid, patch)
    has_zm = any(target.type.upper() == "ZM" for target in targets)
    has_jm = any(target.type.upper() == "JM" for target in targets)

    zm_candidate = _zm_score(patch, targets, visibility, clearance_ratio, direct_openness) if has_zm else None
    jm_candidate = _jm_score(grid, patch, targets, visibility, clearance_ratio, direct_openness) if has_jm else None

    score = 0.8 * clearance_ratio + 0.4 * direct_openness
    details: dict[str, float] = {
        "clearance_ratio": clearance_ratio,
        "direct_openness": direct_openness,
        "patch_size": float(patch.w * patch.h),
    }
    if zm_candidate is not None:
        score += zm_candidate.score
        details.update({f"zm_{k}": v for k, v in zm_candidate.details.items()})
    if jm_candidate is not None:
        score += jm_candidate.score
        details.update({f"jm_{k}": v for k, v in jm_candidate.details.items()})

    return CandidateScore(rect=patch, score=score, details=details), zm_candidate, jm_candidate


def select_positions(
    grid: MapGrid,
    search_region: Rect,
    targets: list[TargetSpec],
    *,
    patch_size: int = 5,
    top_k: int = 5,
    search_config: PositionSearchConfig | None = None,
) -> dict:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    search_region = search_region.clip(grid.width, grid.height)
    if search_region.w < patch_size or search_region.h < patch_size:
        raise ValueError("search_region must be at least patch_size by patch_size")

    config = search_config or PositionSearchConfig()
    visibility = VisibilityCache(grid, [target.enemy for target in targets])
    integral = _build_binary_integral(grid)

    coarse_rects = _coarse_candidate_rects(grid, search_region, targets, patch_size, integral, config)
    candidate_rects = _refine_candidate_rects(grid, search_region, patch_size, integral, coarse_rects, config)

    scored: list[CandidateScore] = []
    best_combined: CandidateScore | None = None
    best_zm: CandidateScore | None = None
    best_jm: CandidateScore | None = None

    for patch in candidate_rects:
        combined, zm_candidate, jm_candidate = _score_candidate_patch(grid, integral, visibility, patch, targets)
        scored.append(combined)

        if best_combined is None or combined.score > best_combined.score:
            best_combined = combined
        if zm_candidate is not None and (best_zm is None or zm_candidate.score > best_zm.score):
            best_zm = zm_candidate
        if jm_candidate is not None and (best_jm is None or jm_candidate.score > best_jm.score):
            best_jm = jm_candidate

    scored.sort(key=lambda item: item.score, reverse=True)

    return {
        "best_combined": best_combined.to_dict() if best_combined else None,
        "best_zm": best_zm.to_dict() if best_zm else None,
        "best_jm": best_jm.to_dict() if best_jm else None,
        "top_k": [item.to_dict() for item in scored[:top_k]],
        "meta": {
            "search_region": search_region.to_dict(),
            "patch_size": patch_size,
            "coarse_candidate_count": len(coarse_rects),
            "candidate_count": len(candidate_rects),
            "target_count": len(targets),
            "coarse_step": _estimate_coarse_step(search_region, config),
        },
    }
