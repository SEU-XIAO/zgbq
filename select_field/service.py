from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .map_loader import MapGrid
from .models import Point, Rect, TargetSpec
from .scoring import select_lookout_points, select_positions


@dataclass
class TacticalAnalyzer:
    grid: MapGrid

    @classmethod
    def from_file(cls, path: str | Path) -> "TacticalAnalyzer":
        return cls(MapGrid.from_file(path))

    def choose_lookout(self, region_a: dict | Rect, region_b: dict | Rect, top_k: int = 5) -> dict:
        return select_lookout_points(
            self.grid,
            self._to_rect(region_a),
            self._to_rect(region_b),
            top_k=top_k,
        )

    def choose_positions(
        self,
        search_region: dict | Rect,
        targets: list[dict | TargetSpec],
        patch_size: int = 5,
        top_k: int = 5,
    ) -> dict:
        parsed_targets = [
            t if isinstance(t, TargetSpec) else TargetSpec(enemy=self._to_point(t["enemy"]), type=str(t["type"]))
            for t in targets
        ]
        return select_positions(
            self.grid,
            self._to_rect(search_region),
            parsed_targets,
            patch_size=patch_size,
            top_k=top_k,
        )

    @staticmethod
    def _to_point(value: tuple[int, int] | list[int] | Point) -> Point:
        if isinstance(value, Point):
            return value
        return Point(int(value[0]), int(value[1]))

    @staticmethod
    def _to_rect(value: dict | Rect) -> Rect:
        if isinstance(value, Rect):
            return value
        return Rect(int(value["x"]), int(value["y"]), int(value["w"]), int(value["h"]))
