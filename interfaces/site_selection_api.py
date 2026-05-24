from __future__ import annotations

from typing import Any


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"missing required field: {key}")
    return payload[key]


def _as_point(value: Any, name: str):
    from select_field import Point

    if isinstance(value, dict):
        return Point(int(value["x"]), int(value["y"]))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return Point(int(value[0]), int(value[1]))
    raise ValueError(f"{name} must be [x, y] or {{x, y}}")


def _as_rect(value: Any, name: str):
    from select_field import Rect

    if not isinstance(value, dict):
        raise ValueError(f"{name} must be {{x, y, w, h}}")
    return Rect(int(value["x"]), int(value["y"]), int(value["w"]), int(value["h"]))


def _as_target(value: Any):
    from select_field import TargetSpec

    if not isinstance(value, dict):
        raise ValueError("target must be an object")
    return TargetSpec(enemy=_as_point(value["enemy"], "target.enemy"), type=str(value["type"]))


def handle_site_selection(payload: dict[str, Any]) -> dict[str, Any]:
    from select_field import TacticalAnalyzer

    map_path = str(_required(payload, "map"))
    mode = str(_required(payload, "mode")).strip().lower()
    top_k = int(payload.get("top_k", 5))
    analyzer = TacticalAnalyzer.from_file(map_path)

    if mode == "lookout":
        result = analyzer.choose_lookout(
            _as_rect(_required(payload, "region_a"), "region_a"),
            _as_rect(_required(payload, "region_b"), "region_b"),
            top_k=top_k,
        )
    elif mode == "position":
        targets = [_as_target(item) for item in payload.get("targets", [])]
        result = analyzer.choose_positions(
            _as_rect(_required(payload, "search_region"), "search_region"),
            targets,
            patch_size=int(payload.get("patch_size", 5)),
            top_k=top_k,
        )
    else:
        raise ValueError("site_selection.mode must be 'lookout' or 'position'")

    return {
        "success": True,
        "task": "site_selection",
        "mode": mode,
        "map": map_path,
        "result": result,
    }
