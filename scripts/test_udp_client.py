"""UDP 服务器测试客户端。

用法：
    1. 先启动服务器：python -m interfaces.udp_path_server
    2. 再跑本脚本：  python scripts/test_udp_client.py
"""

import json
import socket
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interfaces.config import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_FOV_DEG,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_ENEMY_RANGE,
    ENDIAN,
    HEADER_SIZE,
    MSG_ID_SIZE,
    MSG_TYPE_PATH_PLANNING,
)

SERVER = (DEFAULT_HOST, DEFAULT_PORT)


def _struct_format_char() -> str:
    return ">H" if ENDIAN == "big" else "<H"


def build_packet(msg_id: str, payload: dict) -> bytes:
    header = struct.pack(_struct_format_char(), MSG_TYPE_PATH_PLANNING) + msg_id.encode("utf-8").ljust(MSG_ID_SIZE, b"\x00")
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return header + body


def send_and_recv(sock: socket.socket, label: str, payload: dict) -> None:
    print(f"\n{'='*60}")
    print(f"[{label}] 发送请求...")
    print(f"  payload: {json.dumps(payload, ensure_ascii=False, indent=2)[:500]}")
    packet = build_packet(label, payload)
    sock.sendto(packet, SERVER)

    sock.settimeout(120)
    try:
        resp_data, _ = sock.recvfrom(DEFAULT_BUFFER_SIZE)
    except socket.timeout:
        print(f"  ❌ 超时未收到响应")
        return

    resp = json.loads(resp_data[HEADER_SIZE:])
    print(f"  ✅ 收到响应:")
    print(json.dumps(resp, ensure_ascii=False, indent=2)[:1000])


def test_path_planning(sock: socket.socket) -> None:
    """测试寻路请求（外部协议格式：startPos/endPos/enemyInfo）"""
    send_and_recv(sock, "寻路测试", {
        "startPos": [20, 20],
        "endPos": [97, 73],
        "enemyInfo": [
            {"position": [50, 50], "dir": 225, "fov_deg": DEFAULT_FOV_DEG, "range": DEFAULT_ENEMY_RANGE},
        ],
        "use_fallback": True,
    })


def test_site_selection_position(sock: socket.socket) -> None:
    """测试阵地选择"""
    send_and_recv(sock, "阵地选择测试", {
        "mode": "position",
        "search_region": {"x": 10, "y": 10, "w": 120, "h": 120},
        "targets": [
            {"enemy": [80, 80], "type": "ZM"},
            {"enemy": [120, 110], "type": "JM"},
        ],
        "patch_size": 5,
        "top_k": 3,
    })


def test_site_selection_lookout(sock: socket.socket) -> None:
    """测试瞭望点选择"""
    send_and_recv(sock, "瞭望点测试", {
        "mode": "lookout",
        "region_a": {"x": 10, "y": 10, "w": 80, "h": 80},
        "region_b": {"x": 180, "y": 180, "w": 80, "h": 80},
        "top_k": 3,
    })


def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        test_path_planning(sock)
        test_site_selection_position(sock)
        test_site_selection_lookout(sock)
    finally:
        sock.close()
    print(f"\n{'='*60}")
    print("全部测试完成。")


if __name__ == "__main__":
    main()
