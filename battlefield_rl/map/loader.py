from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Tuple

import numpy as np

CELL_PATTERN = re.compile(r"\(([-]?\d+),([-]?\d+)\)")


@dataclass(frozen=True)
class BattlefieldMap:
    heights: np.ndarray
    types: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.heights.shape

    def in_bounds(self, row: int, col: int) -> bool:
        rows, cols = self.shape
        return 0 <= row < rows and 0 <= col < cols

    def is_static_blocked(self, row: int, col: int) -> bool:
        return int(self.types[row, col]) in (1, 2)

    def is_passable(self, from_rc: Tuple[int, int], to_rc: Tuple[int, int], max_height_diff: int = 0) -> bool:
        tr, tc = to_rc
        if not self.in_bounds(tr, tc):
            return False
        return not self.is_static_blocked(tr, tc)


def load_txt_map(path: str | Path) -> BattlefieldMap:
    text = Path(path).read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows_h: List[List[int]] = []
    rows_t: List[List[int]] = []

    for line_idx, line in enumerate(lines):
        matches = CELL_PATTERN.findall(line)
        if not matches:
            raise ValueError(f"第 {line_idx + 1} 行未匹配到有效格子")
        row_h: List[int] = []
        row_t: List[int] = []
        for h, t in matches:
            row_h.append(int(h))
            row_t.append(int(t))
        rows_h.append(row_h)
        rows_t.append(row_t)

    width = len(rows_h[0])
    if any(len(r) != width for r in rows_h):
        raise ValueError("地图每行列数不一致")

    heights = np.asarray(rows_h, dtype=np.int16)
    types = np.asarray(rows_t, dtype=np.int8)
    return BattlefieldMap(heights=heights, types=types)
