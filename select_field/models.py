from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def to_list(self) -> list[int]:
        return [self.x, self.y]


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle using top-left origin and width/height."""

    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> Point:
        return Point(self.x + self.w // 2, self.y + self.h // 2)

    def normalize(self) -> "Rect":
        x1 = min(self.x, self.right)
        y1 = min(self.y, self.bottom)
        x2 = max(self.x, self.right)
        y2 = max(self.y, self.bottom)
        return Rect(x1, y1, x2 - x1, y2 - y1)

    def clip(self, width: int, height: int) -> "Rect":
        rect = self.normalize()
        x1 = max(0, min(rect.x, width))
        y1 = max(0, min(rect.y, height))
        x2 = max(0, min(rect.right, width))
        y2 = max(0, min(rect.bottom, height))
        return Rect(x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom

    def iter_cells(self, step: int = 1):
        for yy in range(self.y, self.bottom, step):
            for xx in range(self.x, self.right, step):
                yield Point(xx, yy)

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass(frozen=True)
class TargetSpec:
    enemy: Point
    type: str

    def to_dict(self) -> dict[str, Any]:
        return {"enemy": self.enemy.to_list(), "type": self.type}


@dataclass
class CandidateScore:
    rect: Rect
    score: float
    details: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rect": self.rect.to_dict(),
            "score": round(self.score, 6),
            "details": {k: round(v, 6) for k, v in self.details.items()},
        }
