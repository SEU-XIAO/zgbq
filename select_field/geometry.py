from __future__ import annotations

import math

from .models import Point


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def bresenham_line(start: Point, end: Point) -> list[Point]:
    """Return the grid cells crossed by a line segment, including endpoints."""

    x0, y0 = start.x, start.y
    x1, y1 = end.x, end.y
    points: list[Point] = []

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        points.append(Point(x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    return points
