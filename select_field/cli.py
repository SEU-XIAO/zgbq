from __future__ import annotations

import argparse
import json

from .models import Point, Rect, TargetSpec
from .service import TacticalAnalyzer


def _add_map_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--map", required=True, help="txt map path")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Site selection tool")
    sub = parser.add_subparsers(dest="mode", required=True)

    lookout = sub.add_parser("lookout", help="choose lookout points between two regions")
    _add_map_argument(lookout)
    lookout.add_argument("--region-a", nargs=4, required=True, metavar=("X", "Y", "W", "H"))
    lookout.add_argument("--region-b", nargs=4, required=True, metavar=("X", "Y", "W", "H"))
    lookout.add_argument("--top-k", type=int, default=5)

    position = sub.add_parser("position", help="choose tactical positions in a search region")
    _add_map_argument(position)
    position.add_argument("--search-region", nargs=4, required=True, metavar=("X", "Y", "W", "H"))
    position.add_argument("--target", action="append", nargs=3, default=[], metavar=("X", "Y", "TYPE"))
    position.add_argument("--patch-size", type=int, default=5)
    position.add_argument("--top-k", type=int, default=5)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    analyzer = TacticalAnalyzer.from_file(args.map)

    if args.mode == "lookout":
        result = analyzer.choose_lookout(
            Rect(*map(int, args.region_a)),
            Rect(*map(int, args.region_b)),
            top_k=args.top_k,
        )
    elif args.mode == "position":
        targets = [
            TargetSpec(enemy=Point(int(x), int(y)), type=str(target_type))
            for x, y, target_type in args.target
        ]
        result = analyzer.choose_positions(
            Rect(*map(int, args.search_region)),
            targets,
            patch_size=args.patch_size,
            top_k=args.top_k,
        )
    else:
        raise RuntimeError("unsupported mode")

    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
