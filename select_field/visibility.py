from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import bresenham_line, euclidean
from .map_loader import MapGrid
from .models import Point


@dataclass
class VisibilityCache:
    """Caches binary line-of-sight checks for site selection scoring."""

    grid: MapGrid
    enemies: list[Point]
    _cache: dict[tuple[int, int, int], bool] = field(default_factory=dict)
    _enemy_lookup: dict[tuple[int, int], int] = field(init=False)

    def __post_init__(self) -> None:
        self._enemy_lookup = {}
        for idx, enemy in enumerate(self.enemies):
            self._enemy_lookup.setdefault((enemy.x, enemy.y), idx)

    def enemy_index(self, enemy: Point) -> int | None:
        return self._enemy_lookup.get((enemy.x, enemy.y))

    def visible_between(self, enemy_index: int | None, point: Point) -> bool:
        if enemy_index is None:
            return False
        return self.visible_from_enemy(enemy_index, point)

    def visible_from_enemy(self, enemy_index: int, point: Point) -> bool:
        key = (enemy_index, point.x, point.y)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        enemy = self.enemies[enemy_index]
        visible = self.line_of_sight(enemy, point)
        self._cache[key] = visible
        return visible

    def visible_to_any_enemy(self, point: Point) -> bool:
        return any(self.visible_from_enemy(idx, point) for idx in range(len(self.enemies)))

    def exposure_score(self, point: Point) -> float:
        exposure = 0.0
        for idx, enemy in enumerate(self.enemies):
            if self.visible_from_enemy(idx, point):
                exposure += 1.0 / (euclidean(enemy, point) + 1.0)
        return exposure

    def visible_enemy_count(self, point: Point) -> int:
        return sum(1 for idx in range(len(self.enemies)) if self.visible_from_enemy(idx, point))

    def line_of_sight(self, start: Point, end: Point) -> bool:
        for cell in bresenham_line(start, end)[1:-1]:
            if not self.grid.is_passable(cell.x, cell.y):
                return False
        return True

    def _line_of_sight(self, start: Point, end: Point) -> bool:
        return self.line_of_sight(start, end)
