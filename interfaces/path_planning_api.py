from __future__ import annotations

from typing import Any


Coord = tuple[int, int]


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"missing required field: {key}")
    return payload[key]


def _as_coord(value: Any, name: str) -> Coord:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = value
    if not isinstance(parts, (list, tuple)) or len(parts) != 2:
        raise ValueError(f"{name} must be [row, col]")
    return int(parts[0]), int(parts[1])


def _as_enemy(value: Any):
    from battlefield_rl.env import EnemySpec

    if isinstance(value, dict):
        max_range = value.get("range", value.get("max_range"))
        if max_range is None:
            raise ValueError("enemy.range is required")
        return EnemySpec(
            row=int(value["row"]),
            col=int(value["col"]),
            facing_deg=float(value["facing_deg"]),
            fov_deg=float(value["fov_deg"]),
            max_range=int(max_range),
        )

    if isinstance(value, (list, tuple)) and len(value) == 5:
        return EnemySpec(
            row=int(value[0]),
            col=int(value[1]),
            facing_deg=float(value[2]),
            fov_deg=float(value[3]),
            max_range=int(value[4]),
        )

    raise ValueError("enemy must be {row,col,facing_deg,fov_deg,range} or [row,col,facing_deg,fov_deg,range]")


def handle_path_planning(payload: dict[str, Any]) -> dict[str, Any]:
    from scripts.eval_hierarchical_executor import plan_and_execute, write_result_if_requested

    map_path = str(_required(payload, "map"))
    start = _as_coord(_required(payload, "start"), "start")
    goal = _as_coord(_required(payload, "goal"), "goal")
    enemies = [_as_enemy(item) for item in payload.get("enemies", [])]

    result = plan_and_execute(map_path=map_path, start=start, goal=goal, enemies=enemies)
    output_dir = payload.get("output_dir")
    if output_dir:
        write_result_if_requested(result, str(output_dir), start, goal)
    return result
