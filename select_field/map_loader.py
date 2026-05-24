from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .models import Point


_CELL_PATTERN = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


@dataclass
class MapGrid:
    """Terrain grid loaded from txt cells formatted as ``(height, kind)``.

    Site selection uses only ``kind``:
    0 is passable, while 1 and 2 are blocked and block line of sight.
    """

    width: int
    height: int
    values: list[list[int]]
    kinds: list[list[int]]
    blocked_kinds: frozenset[int] = frozenset({1, 2})

    @classmethod
    def from_file(cls, path: str | Path, blocked_kinds: frozenset[int] | None = None) -> "MapGrid":
        text = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        rows_values: list[list[int]] = []
        rows_kinds: list[list[int]] = []

        for line in text:
            matches = _CELL_PATTERN.findall(line)
            if not matches:
                continue

            value_row: list[int] = []
            kind_row: list[int] = []
            for value_str, kind_str in matches:
                value_row.append(int(value_str))
                kind_row.append(int(kind_str))
            rows_values.append(value_row)
            rows_kinds.append(kind_row)

        if not rows_values:
            raise ValueError(f"no grid cells found in txt map: {path}")

        width = len(rows_values[0])
        for idx, row in enumerate(rows_values, start=1):
            if len(row) != width:
                raise ValueError(f"row {idx} has width {len(row)}, expected {width}")
        for idx, row in enumerate(rows_kinds, start=1):
            if len(row) != width:
                raise ValueError(f"kind row {idx} has width {len(row)}, expected {width}")

        return cls(
            width=width,
            height=len(rows_values),
            values=rows_values,
            kinds=rows_kinds,
            blocked_kinds=blocked_kinds if blocked_kinds is not None else frozenset({1, 2}),
        )

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def value_at(self, x: int, y: int) -> int:
        return self.values[y][x]

    def kind_at(self, x: int, y: int) -> int:
        return self.kinds[y][x]

    def is_passable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.kind_at(x, y) not in self.blocked_kinds

    def point_passable(self, point: Point) -> bool:
        return self.is_passable(point.x, point.y)
