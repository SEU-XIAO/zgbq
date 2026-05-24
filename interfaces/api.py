from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _read_request(input_path: str | None) -> dict[str, Any]:
    text = Path(input_path).read_text(encoding="utf-8") if input_path else sys.stdin.read()
    if not text.strip():
        raise ValueError("request json is empty")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("request json must be an object")
    return payload


def dispatch_request(payload: dict[str, Any]) -> dict[str, Any]:
    task = str(payload.get("task", "")).strip().lower()
    if task == "path_planning":
        from .path_planning_api import handle_path_planning

        return handle_path_planning(payload)
    if task == "site_selection":
        from .site_selection_api import handle_site_selection

        return handle_site_selection(payload)
    raise ValueError("task must be 'path_planning' or 'site_selection'")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified JSON API for ZGBQ")
    parser.add_argument("--input", default=None, help="request json file; stdin is used when omitted")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        result = dispatch_request(_read_request(args.input))
    except Exception as exc:
        result = {
            "success": False,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        }
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
