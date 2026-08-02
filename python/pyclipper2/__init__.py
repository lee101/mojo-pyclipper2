"""A focused Mojo implementation of pyclipper2's integer-path API.

The accelerated subset is deliberately explicit: predicates, convex polygon
intersection, convex miter offsets, and exact boolean operations on collections
of axis-aligned rectangles.  Unsupported general overlapping non-convex inputs
raise ``NotImplementedError`` instead of returning a plausible but wrong path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import atan2, ceil, cos, pi, sin
from numbers import Integral
from typing import Iterable, Sequence

import numpy as np

from ._lib import addr, f64, lib

VERSION = "mojo-0.1.0"
_MAX_EXACT_FLOAT64_INTEGER = 2**53


def _int64_coordinate(value) -> int:
    """Accept an integral coordinate only when the FFI can represent it exactly."""
    if not isinstance(value, Integral):
        raise TypeError("Point64 coordinates must be integers")
    value = int(value)
    if not -_MAX_EXACT_FLOAT64_INTEGER <= value <= _MAX_EXACT_FLOAT64_INTEGER:
        raise OverflowError("Point64 coordinates outside +/-2**53 are not supported by the float64 kernel")
    return value


def _ffi_coordinate(value) -> float:
    if isinstance(value, Integral):
        return float(_int64_coordinate(value))
    return float(value)


class ClipType(IntEnum):
    NO_CLIP = 0
    INTERSECTION = 1
    UNION = 2
    DIFFERENCE = 3
    XOR = 4


class FillRule(IntEnum):
    EVEN_ODD = 0
    NON_ZERO = 1
    POSITIVE = 2
    NEGATIVE = 3


class JoinType(IntEnum):
    SQUARE = 0
    BEVEL = 1
    ROUND = 2
    MITER = 3


class EndType(IntEnum):
    POLYGON = 0
    JOINED = 1
    BUTT = 2
    SQUARE = 3
    ROUND = 4


class PathType(IntEnum):
    SUBJECT = 0
    CLIP = 1


class JoinWith(IntEnum):
    NO_JOIN = 0
    LEFT = 1
    RIGHT = 2


class PointInPolygonResult(IntEnum):
    IS_ON = 0
    IS_INSIDE = 1
    IS_OUTSIDE = 2


@dataclass
class Point64:
    x: int
    y: int

    def __post_init__(self):
        self.x, self.y = _int64_coordinate(self.x), _int64_coordinate(self.y)


@dataclass
class PointD:
    x: float
    y: float

    def __post_init__(self):
        self.x, self.y = float(self.x), float(self.y)


@dataclass
class Rect64:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self):
        self.left, self.top = _int64_coordinate(self.left), _int64_coordinate(self.top)
        self.right, self.bottom = _int64_coordinate(self.right), _int64_coordinate(self.bottom)


@dataclass
class RectD:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self):
        self.left, self.top = float(self.left), float(self.top)
        self.right, self.bottom = float(self.right), float(self.bottom)


def _is_integer_path(path: Sequence) -> bool:
    if isinstance(path, np.ndarray):
        return False
    return bool(path) and all(isinstance(p, Point64) for p in path)


def _xy(path: Sequence) -> np.ndarray:
    try:
        if isinstance(path, np.ndarray):
            if path.dtype.kind in "iu":
                if (path.dtype.kind == "u" and np.any(path > _MAX_EXACT_FLOAT64_INTEGER)) or (
                    path.dtype.kind == "i" and (np.any(path > _MAX_EXACT_FLOAT64_INTEGER) or np.any(path < -_MAX_EXACT_FLOAT64_INTEGER))
                ):
                    raise OverflowError("integer coordinates outside +/-2**53 are not supported by the float64 kernel")
            xy = f64(path).reshape(-1, 2)
        else:
            values = (_ffi_coordinate(coordinate) for point in path for coordinate in
                      ((point.x, point.y) if hasattr(point, "x") else point))
            xy = np.fromiter(values, dtype=np.float64, count=2 * len(path)).reshape(-1, 2)
        if not np.isfinite(xy).all():
            raise ValueError("path coordinates must be finite")
        return xy
    except TypeError as exc:
        raise TypeError("a path must contain Point64, PointD, or (x, y) pairs") from exc


def _path(xy: np.ndarray, integer: bool) -> list[Point64] | list[PointD]:
    if integer:
        return [Point64(int(round(x)), int(round(y))) for x, y in xy]
    return [PointD(float(x), float(y)) for x, y in xy]


def make_path(points) -> list[Point64]:
    return [Point64(p[0], p[1]) for p in points]


def make_path_double(points) -> list[PointD]:
    return [PointD(float(p[0]), float(p[1])) for p in points]


def area(path) -> float:
    xy = _xy(path)
    return float(lib().mpc_area(addr(xy), len(xy)))


def is_positive(path) -> bool:
    return area(path) > 0.0


def point_in_polygon(pt, polygon) -> PointInPolygonResult:
    point = (pt.x, pt.y) if hasattr(pt, "x") else pt
    xy = _xy(polygon)
    value = int(lib().mpc_point_in_polygon(float(point[0]), float(point[1]), addr(xy), len(xy)))
    return (PointInPolygonResult.IS_ON if value < 0 else
            PointInPolygonResult.IS_INSIDE if value else PointInPolygonResult.IS_OUTSIDE)


def _convex(xy: np.ndarray) -> bool:
    if len(xy) < 3:
        return False
    direction = 0.0
    for i in range(len(xy)):
        a, b, c = xy[i - 1], xy[i], xy[(i + 1) % len(xy)]
        u, v = b - a, c - b
        cross = float(u[0] * v[1] - u[1] * v[0])
        if abs(cross) > 1e-12:
            if direction and cross * direction < 0:
                return False
            direction = cross
    return bool(direction)


def _convex_intersection(subject, clip):
    a, b = _xy(subject), _xy(clip)
    if not (_convex(a) and _convex(b)):
        raise NotImplementedError("general polygon overlay is not covered; use convex paths or rectangles")
    capacity = len(a) + len(b) + 4
    first = np.empty((capacity, 2), dtype=np.float64)
    second = np.empty_like(first)
    n = int(lib().mpc_convex_intersection(addr(a), len(a), addr(b), len(b), addr(first), addr(second), capacity))
    if n < 0:
        raise RuntimeError("convex clip output exceeded its validated capacity")
    return _path(first[:n], _is_integer_path(subject) and _is_integer_path(clip))


def _rectangle(path):
    xy = _xy(path)
    if len(xy) != 4:
        return None
    xs, ys = sorted(set(xy[:, 0])), sorted(set(xy[:, 1]))
    if len(xs) != 2 or len(ys) != 2:
        return None
    expected = {(xs[0], ys[0]), (xs[0], ys[1]), (xs[1], ys[0]), (xs[1], ys[1])}
    return (xs[0], ys[0], xs[1], ys[1]) if set(map(tuple, xy)) == expected else None


def _rect_boolean(subjects, clips, operation: str):
    srects, crects = [_rectangle(p) for p in subjects], [_rectangle(p) for p in clips]
    if any(r is None for r in srects + crects):
        return None
    all_rects = srects + crects
    if not all_rects:
        return []
    xs = sorted({v for r in all_rects for v in (r[0], r[2])})
    ys = sorted({v for r in all_rects for v in (r[1], r[3])})
    def contains(rects, x, y):
        return any(r[0] < x < r[2] and r[1] < y < r[3] for r in rects)
    cells = set()
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            left, right, bottom, top = xs[i], xs[i + 1], ys[j], ys[j + 1]
            a, b = contains(srects, (left + right) / 2, (bottom + top) / 2), contains(crects, (left + right) / 2, (bottom + top) / 2)
            keep = {"union": a or b, "intersection": a and b, "difference": a and not b, "xor": a != b}[operation]
            if keep:
                cells.add((i, j))
    edges = {}
    def add_edge(start, end):
        if edges.get(end) == start:
            del edges[end]
        else:
            edges[start] = end
    for i, j in cells:
        x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
        if (i, j - 1) not in cells: add_edge((x0, y0), (x1, y0))
        if (i + 1, j) not in cells: add_edge((x1, y0), (x1, y1))
        if (i, j + 1) not in cells: add_edge((x1, y1), (x0, y1))
        if (i - 1, j) not in cells: add_edge((x0, y1), (x0, y0))
    integer = all(_is_integer_path(p) for p in subjects + clips)
    paths = []
    while edges:
        start = next(iter(edges))
        ring, cursor = [start], start
        while cursor in edges:
            cursor = edges.pop(cursor)
            if cursor == start:
                break
            ring.append(cursor)
        paths.append(_path(np.asarray(ring), integer))
    return paths


def union(subjects, fill_rule: FillRule = FillRule.NON_ZERO):
    rect = _rect_boolean(list(subjects), [], "union")
    if rect is not None:
        return rect
    paths = list(subjects)
    if len(paths) == 1:
        return [list(paths[0])]
    raise NotImplementedError("union is covered for axis-aligned rectangle paths")


def intersection(subjects, clips, fill_rule: FillRule = FillRule.NON_ZERO):
    subjects, clips = list(subjects), list(clips)
    rect = _rect_boolean(subjects, clips, "intersection")
    if rect is not None:
        return rect
    if len(subjects) == len(clips) == 1:
        result = _convex_intersection(subjects[0], clips[0])
        return [result] if result else []
    raise NotImplementedError("intersection is covered for one convex pair or rectangle collections")


def difference(subjects, clips, fill_rule: FillRule = FillRule.NON_ZERO):
    rect = _rect_boolean(list(subjects), list(clips), "difference")
    if rect is not None:
        return rect
    if not clips:
        return [list(p) for p in subjects]
    raise NotImplementedError("difference is covered for axis-aligned rectangle paths")


def xor_(subjects, clips, fill_rule: FillRule = FillRule.NON_ZERO):
    subjects, clips = list(subjects), list(clips)
    # Keeping A\B and B\A as separate contours avoids joining them through a
    # zero-area corner in the rectangular grid representation.
    if all(_rectangle(p) is not None for p in subjects + clips):
        return _rect_boolean(subjects, clips, "difference") + _rect_boolean(clips, subjects, "difference")
    rect = _rect_boolean(subjects, clips, "xor")
    if rect is not None:
        return rect
    if not subjects:
        return [list(p) for p in clips]
    if not clips:
        return [list(p) for p in subjects]
    raise NotImplementedError("xor is covered for axis-aligned rectangle paths")


def _outward_normals(xy: np.ndarray):
    sign = 1.0 if area([PointD(*p) for p in xy]) >= 0 else -1.0
    normals = []
    for a, b in zip(xy, np.roll(xy, -1, axis=0)):
        edge = b - a
        length = float(np.hypot(*edge))
        normals.append(sign * np.array([edge[1], -edge[0]]) / length)
    return normals


def _inflate_convex(path, delta, jt, arc_tolerance, miter_limit):
    xy = _xy(path)
    if not _convex(xy):
        raise NotImplementedError("offsetting is covered for convex closed paths")
    integer = _is_integer_path(path)
    if jt == JoinType.MITER:
        dst = np.empty_like(xy)
        if not lib().mpc_offset_miter(addr(xy), len(xy), float(delta), addr(dst)):
            raise ValueError("degenerate convex path")
        if delta and np.any(np.hypot(*(dst - xy).T) > abs(float(delta)) * miter_limit + 1e-12):
            raise NotImplementedError("miter joins exceeding miter_limit are not covered")
        return _path(dst, integer)
    normals = _outward_normals(xy)
    if jt == JoinType.BEVEL or jt == JoinType.SQUARE:
        out = []
        for i, point in enumerate(xy):
            out.extend((point + delta * normals[i - 1], point + delta * normals[i]))
        return _path(np.asarray(out), integer)
    if jt == JoinType.ROUND:
        out = []
        step = max(0.05, 0.25 if not arc_tolerance else min(0.5, abs(float(arc_tolerance) / max(abs(delta), 1e-12))))
        winding = area([PointD(*p) for p in xy])
        for i, point in enumerate(xy):
            start, end = atan2(normals[i - 1][1], normals[i - 1][0]), atan2(normals[i][1], normals[i][0])
            if winding >= 0:
                while end < start: end += 2 * pi
            else:
                while end > start: end -= 2 * pi
            count = max(1, int(ceil(abs(end - start) / step)))
            for k in range(count + 1):
                angle = start + (end - start) * k / count
                out.append(point + delta * np.array([cos(angle), sin(angle)]))
        return _path(np.asarray(out), integer)
    raise ValueError("unknown JoinType")


def inflate_paths(paths, delta: float, jt: JoinType, et: EndType,
                  miter_limit: float = 2.0, precision: int | float = 2,
                  arc_tolerance: float = 0.0):
    if et != EndType.POLYGON:
        raise NotImplementedError("offsetting is covered for closed polygons (EndType.POLYGON)")
    if precision != 2:
        raise NotImplementedError("offset precision values other than the default 2 are not covered")
    miter_limit = float(miter_limit)
    if miter_limit <= 0:
        raise ValueError("miter_limit must be positive")
    return [_inflate_convex(path, delta, JoinType(jt), arc_tolerance, miter_limit) for path in paths]


class ClipperOffset:
    def __init__(self, miter_limit: float = 2.0, arc_tolerance: float = 0.0, reverse_solution: bool = False):
        self.miter_limit, self.arc_tolerance, self.reverse_solution = miter_limit, arc_tolerance, reverse_solution
        self._items = []

    def add_path(self, path, join_type: JoinType, end_type: EndType):
        self._items.append((list(path), JoinType(join_type), EndType(end_type)))

    def add_paths(self, paths, join_type: JoinType, end_type: EndType):
        for path in paths:
            self.add_path(path, join_type, end_type)

    def execute(self, delta: float, solution: list):
        result = []
        for path, jt, et in self._items:
            result.extend(inflate_paths([path], delta, jt, et, self.miter_limit, arc_tolerance=self.arc_tolerance))
        if self.reverse_solution:
            result = [list(reversed(path)) for path in result]
        solution.extend(result)

    def clear(self):
        self._items.clear()
