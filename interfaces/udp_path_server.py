from __future__ import annotations

import argparse
import json
import socket
import struct
from dataclasses import dataclass
from typing import Any

from .path_planning_api import handle_path_planning
from .site_selection_api import handle_site_selection


MSG_TYPE_PATH_PLANNING = 0x3311
MSG_TYPE_SITE_SELECTION = 0x3311
MSG_ID_SIZE = 20
HEADER_SIZE = 2 + MSG_ID_SIZE
DEFAULT_MAP_PATH = "MyPath_Data417.txt"
DEFAULT_FOV_DEG = 90.0
DEFAULT_ENEMY_RANGE = 20
DEFAULT_BUFFER_SIZE = 65535


@dataclass(frozen=True)
class UdpPacket:
    msg_type: int
    msg_id: str
    payload: dict[str, Any]


def _struct_format(endian: str) -> str:
    if endian == "big":
        return ">H20s"
    if endian == "little":
        return "<H20s"
    raise ValueError("endian must be 'little' or 'big'")


def _decode_msg_id(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()


def _encode_msg_id(msg_id: str) -> bytes:
    raw = msg_id.encode("utf-8")[:MSG_ID_SIZE]
    return raw.ljust(MSG_ID_SIZE, b"\x00")


def decode_packet(data: bytes, *, endian: str = "little") -> UdpPacket:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"udp packet too short: {len(data)} bytes")

    msg_type, msg_id_raw = struct.unpack(_struct_format(endian), data[:HEADER_SIZE])
    json_bytes = data[HEADER_SIZE:]
    if not json_bytes.strip():
        raise ValueError("json_str is empty")

    payload = json.loads(json_bytes.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("json_str must be a JSON object")
    return UdpPacket(msg_type=msg_type, msg_id=_decode_msg_id(msg_id_raw), payload=payload)


def encode_packet(msg_type: int, msg_id: str, payload: dict[str, Any], *, endian: str = "little") -> bytes:
    header = struct.pack(_struct_format(endian), msg_type, _encode_msg_id(msg_id))
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return header + body


def _coord_xy_to_row_col(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"{name} must be [x, y]")
    x = float(value[0])
    y = float(value[1])
    return [int(round(y)), int(round(x))]


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"invalid boolean value: {value}")


def _coord_xy(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"{name} must be [x, y]")
    return [int(round(float(value[0]))), int(round(float(value[1])))]


def _rect_from_points(points: Any, name: str) -> dict[str, int]:
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        raise ValueError(f"{name} must contain at least two [x, y] points")
    parsed = [_coord_xy(point, name) for point in points]
    xs = [point[0] for point in parsed]
    ys = [point[1] for point in parsed]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return {"x": x1, "y": y1, "w": x2 - x1 + 1, "h": y2 - y1 + 1}


def _normalize_rect(value: Any, name: str) -> dict[str, int]:
    if isinstance(value, dict):
        return {
            "x": int(round(float(value["x"]))),
            "y": int(round(float(value["y"]))),
            "w": int(round(float(value["w"]))),
            "h": int(round(float(value["h"]))),
        }
    if isinstance(value, (list, tuple)) and len(value) == 4 and not isinstance(value[0], (list, tuple, dict)):
        return {
            "x": int(round(float(value[0]))),
            "y": int(round(float(value[1]))),
            "w": int(round(float(value[2]))),
            "h": int(round(float(value[3]))),
        }
    return _rect_from_points(value, name)


def _full_map_rect(map_path: str) -> dict[str, int]:
    from select_field import MapGrid

    grid = MapGrid.from_file(map_path)
    return {"x": 0, "y": 0, "w": grid.width, "h": grid.height}


def _enemy_to_path_enemy(
    enemy: Any,
    *,
    default_fov_deg: float,
    default_enemy_range: int,
) -> dict[str, Any]:
    if not isinstance(enemy, dict):
        raise ValueError("enemyInfo item must be an object")
    row, col = _coord_xy_to_row_col(enemy.get("position"), "enemyInfo.position")
    return {
        "row": row,
        "col": col,
        "facing_deg": float(enemy.get("dir", enemy.get("facing_deg", 0.0))),
        "fov_deg": float(enemy.get("fov_deg", default_fov_deg)),
        "range": int(enemy.get("range", enemy.get("max_range", default_enemy_range))),
    }


def _target_from_external(value: Any, default_type: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("target/enemyInfo item must be an object")
    if "enemy" in value:
        enemy = value["enemy"]
    elif "position" in value:
        enemy = value["position"]
    else:
        raise ValueError("target item must contain enemy or position")
    return {
        "enemy": _coord_xy(enemy, "target.enemy"),
        "type": str(value.get("type", default_type)),
    }


def adapt_maneuver_path_request(
    payload: dict[str, Any],
    *,
    map_path: str,
    default_fov_deg: float = DEFAULT_FOV_DEG,
    default_enemy_range: int = DEFAULT_ENEMY_RANGE,
) -> dict[str, Any]:
    request = {
        "task": "path_planning",
        "map": str(payload.get("map", map_path)),
        "start": _coord_xy_to_row_col(payload.get("startPos"), "startPos"),
        "goal": _coord_xy_to_row_col(payload.get("endPos"), "endPos"),
        "enemies": [
            _enemy_to_path_enemy(
                item,
                default_fov_deg=default_fov_deg,
                default_enemy_range=default_enemy_range,
            )
            for item in payload.get("enemyInfo", [])
        ],
    }
    model_path = payload.get("model_path", payload.get("modelPath", payload.get("model")))
    if model_path:
        request["model_path"] = str(model_path)
    request["use_fallback"] = _as_bool(payload.get("use_fallback", payload.get("fallback")), True)
    return request


def adapt_site_selection_request(payload: dict[str, Any], *, map_path: str) -> dict[str, Any]:
    request_map = str(payload.get("map", map_path))
    mode = str(payload.get("mode", payload.get("siteMode", "position"))).strip().lower()
    if mode in {"阵地选择", "position_select"}:
        mode = "position"
    if mode in {"瞭望点", "lookout_select"}:
        mode = "lookout"

    if mode == "lookout":
        region_a = payload.get("region_a", payload.get("regionA"))
        region_b = payload.get("region_b", payload.get("regionB"))
        if region_a is None or region_b is None:
            areas = payload.get("ZCArea", payload.get("areas"))
            if isinstance(areas, (list, tuple)) and len(areas) >= 2:
                midpoint = len(areas) // 2
                region_a = areas[:midpoint]
                region_b = areas[midpoint:]
        return {
            "task": "site_selection",
            "mode": "lookout",
            "map": request_map,
            "region_a": _normalize_rect(region_a, "region_a"),
            "region_b": _normalize_rect(region_b, "region_b"),
            "top_k": int(payload.get("top_k", payload.get("topK", 5))),
        }

    search_region = payload.get("search_region", payload.get("searchRegion"))
    if search_region is None:
        search_region = _full_map_rect(request_map)

    default_type = str(payload.get("type", "ZM"))
    raw_targets = payload.get("targets")
    if raw_targets is None:
        raw_targets = payload.get("enemyInfo")
    if raw_targets is None and "enemy" in payload:
        raw_targets = [{"enemy": payload["enemy"], "type": default_type}]
    if raw_targets is None:
        raw_targets = []

    return {
        "task": "site_selection",
        "mode": "position",
        "map": request_map,
        "search_region": _normalize_rect(search_region, "search_region"),
        "targets": [_target_from_external(item, default_type) for item in raw_targets],
        "patch_size": int(payload.get("patch_size", payload.get("patchSize", 5))),
        "top_k": int(payload.get("top_k", payload.get("topK", 5))),
    }


def _path_row_col_to_xy(path: Any) -> list[list[int]]:
    if not isinstance(path, list):
        return []
    out: list[list[int]] = []
    for item in path:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            row = int(item[0])
            col = int(item[1])
            out.append([col, row])
    return out


def adapt_path_response(result: dict[str, Any]) -> dict[str, Any]:
    response = dict(result)
    response["result"] = "success" if result.get("success") else "failed"
    response["steps"] = int(result.get("total_steps", max(0, len(result.get("path", [])) - 1)))
    response["pathXY"] = _path_row_col_to_xy(result.get("path", []))
    return response


def _site_rect_alias(candidate: Any) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    rect = candidate.get("rect")
    if not isinstance(rect, dict):
        return dict(candidate)
    out = dict(candidate)
    out["top_left"] = [int(rect["x"]), int(rect["y"])]
    out["center"] = [int(rect["x"]) + int(rect["w"]) // 2, int(rect["y"]) + int(rect["h"]) // 2]
    out["patch_size"] = int(rect["w"])
    return out


def adapt_site_response(result: dict[str, Any]) -> dict[str, Any]:
    response = dict(result)
    response["result"] = "success" if result.get("success") else "failed"
    selection = result.get("result")
    if isinstance(selection, dict):
        response["top_k"] = selection.get("top_k", [])
        response["best_direct"] = _site_rect_alias(selection.get("best_zm"))
        response["best_indirect"] = _site_rect_alias(selection.get("best_jm"))
        response["best_combined"] = _site_rect_alias(selection.get("best_combined"))
        response["best_in_a_for_b"] = _site_rect_alias(selection.get("best_in_a_for_b"))
        response["best_in_b_for_a"] = _site_rect_alias(selection.get("best_in_b_for_a"))
    return response


def _is_site_selection_payload(payload: dict[str, Any]) -> bool:
    task = str(payload.get("task", "")).strip().lower()
    if task == "site_selection":
        return True
    if "startPos" in payload or "endPos" in payload:
        return False
    site_keys = {
        "enemy",
        "targets",
        "search_region",
        "searchRegion",
        "region_a",
        "regionA",
        "region_b",
        "regionB",
        "ZCArea",
    }
    return any(key in payload for key in site_keys)


def handle_udp_payload(
    packet: UdpPacket,
    *,
    map_path: str,
    default_fov_deg: float,
    default_enemy_range: int,
) -> dict[str, Any]:
    if packet.msg_type not in {MSG_TYPE_PATH_PLANNING, MSG_TYPE_SITE_SELECTION}:
        raise ValueError(f"unsupported MsgType: 0x{packet.msg_type:04X}")

    if _is_site_selection_payload(packet.payload):
        site_request = adapt_site_selection_request(packet.payload, map_path=map_path)
        return adapt_site_response(handle_site_selection(site_request))

    path_request = adapt_maneuver_path_request(
        packet.payload,
        map_path=map_path,
        default_fov_deg=default_fov_deg,
        default_enemy_range=default_enemy_range,
    )
    return adapt_path_response(handle_path_planning(path_request))


def serve(
    *,
    host: str,
    port: int,
    map_path: str,
    endian: str,
    default_fov_deg: float,
    default_enemy_range: int,
    buffer_size: int,
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"UDP path server listening on {host}:{port}, map={map_path}, endian={endian}", flush=True)

    while True:
        data, address = sock.recvfrom(buffer_size)
        msg_type = MSG_TYPE_PATH_PLANNING
        msg_id = ""
        try:
            packet = decode_packet(data, endian=endian)
            msg_type = packet.msg_type
            msg_id = packet.msg_id
            response = handle_udp_payload(
                packet,
                map_path=map_path,
                default_fov_deg=default_fov_deg,
                default_enemy_range=default_enemy_range,
            )
        except Exception as exc:
            response = {
                "success": False,
                "result": "failed",
                "error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                },
            }
        sock.sendto(encode_packet(msg_type, msg_id, response, endian=endian), address)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UDP wrapper for path planning and site selection")
    parser.add_argument("--host", default="127.0.0.1", help="local address to bind")
    parser.add_argument("--port", type=int, required=True, help="local UDP port to listen on")
    parser.add_argument("--map", default=DEFAULT_MAP_PATH, help="txt map path")
    parser.add_argument("--endian", choices=["little", "big"], default="little", help="USHORT byte order")
    parser.add_argument("--default-fov-deg", type=float, default=DEFAULT_FOV_DEG)
    parser.add_argument("--default-enemy-range", type=int, default=DEFAULT_ENEMY_RANGE)
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    serve(
        host=args.host,
        port=args.port,
        map_path=args.map,
        endian=args.endian,
        default_fov_deg=args.default_fov_deg,
        default_enemy_range=args.default_enemy_range,
        buffer_size=args.buffer_size,
    )


if __name__ == "__main__":
    main()
