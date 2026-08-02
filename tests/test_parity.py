"""Parity cases derived from pyclipper2's published binding and Shapely tests.

pyclipper2 0.0.8 publishes only a macOS arm64 wheel, so these Linux tests use
independent pure-Python reference geometry for the covered subset.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pyclipper2 as pc


def coords(path):
    return [(p.x, p.y) for p in path]


def reference_area(path):
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(path, path[1:] + path[:1])) / 2


def reference_convex_clip(subject, clip):
    """Independent Sutherland-Hodgman reference used for numeric parity."""
    output = [tuple(p) for p in subject]
    orientation = 1 if reference_area(clip) >= 0 else -1
    for c, d in zip(clip, clip[1:] + clip[:1]):
        source, output = output, []
        for a, b in zip(source, source[1:] + source[:1]):
            cross_a = orientation * ((d[0] - c[0]) * (a[1] - c[1]) - (d[1] - c[1]) * (a[0] - c[0]))
            cross_b = orientation * ((d[0] - c[0]) * (b[1] - c[1]) - (d[1] - c[1]) * (b[0] - c[0]))
            if (cross_a >= 0) != (cross_b >= 0):
                rx, ry, sx, sy = b[0] - a[0], b[1] - a[1], d[0] - c[0], d[1] - c[1]
                t = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / (rx * sy - ry * sx)
                output.append((a[0] + t * rx, a[1] + t * ry))
            if cross_b >= 0:
                output.append(b)
    return output


@pytest.fixture
def squares():
    return (pc.make_path([[0, 0], [10, 0], [10, 10], [0, 10]]),
            pc.make_path([[5, 5], [15, 5], [15, 15], [5, 15]]))


def test_public_types_and_enum_values():
    assert pc.Point64(2, 3).x == 2
    assert pc.PointD(2.5, 3.5).y == 3.5
    assert pc.Rect64(0, 1, 2, 3).bottom == 3
    assert pc.ClipType.XOR == 4 and pc.FillRule.NON_ZERO == 1
    assert pc.JoinType.MITER == 3 and pc.EndType.POLYGON == 0
    assert pc.PathType.SUBJECT == 0 and pc.JoinWith.RIGHT == 2
    assert pc.RectD(0.5, 1.5, 2.5, 3.5).bottom == 3.5


def test_integer_coordinates_are_not_silently_narrowed():
    with pytest.raises(TypeError):
        pc.Point64(1.5, 2)
    with pytest.raises(OverflowError):
        pc.Point64(2**53 + 1, 0)
    with pytest.raises(OverflowError):
        pc.area(np.array([[0, 0], [2**53 + 1, 0], [0, 1]], dtype=np.int64))
    with pytest.raises(OverflowError):
        pc.area([(0, 0), (2**53 + 1, 0), (0, 1)])
    with pytest.raises(ValueError):
        pc.area(np.array([[0.0, 0.0], [np.inf, 0.0], [0.0, 1.0]]))


def test_make_path_and_area_matches_independent_shoelace(squares):
    square, _ = squares
    assert coords(square) == [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert pc.area(square) == reference_area(coords(square)) == 100
    assert pc.is_positive(square)
    assert not pc.is_positive(list(reversed(square)))


@pytest.mark.parametrize(("point", "expected"), [((5, 5), pc.PointInPolygonResult.IS_INSIDE), ((15, 5), pc.PointInPolygonResult.IS_OUTSIDE), ((0, 5), pc.PointInPolygonResult.IS_ON)])
def test_point_in_polygon_matches_published_cases(squares, point, expected):
    assert pc.point_in_polygon(pc.Point64(*point), squares[0]) == expected


def test_convex_intersection_matches_independent_reference():
    subject = pc.make_path_double([[0, 0], [8, 0], [8, 7], [0, 8]])
    clip = pc.make_path_double([[2, -1], [9, 3], [4, 10]])
    got = pc.intersection([subject], [clip])[0]
    ref = reference_convex_clip(coords(subject), coords(clip))
    assert pc.area(got) == pytest.approx(reference_area(ref), abs=1e-9)
    assert len(got) == len(ref)


def test_published_square_intersection_vector(squares):
    got = pc.intersection([squares[0]], [squares[1]])
    assert len(got) == 1
    assert pc.area(got[0]) == 25


@pytest.mark.parametrize(("function", "expected"), [(pc.union, 175), (pc.intersection, 25), (pc.difference, 75), (pc.xor_, 150)])
def test_rectangle_boolean_areas_match_published_vectors(squares, function, expected):
    if function is pc.union:
        paths = function(list(squares))
    else:
        paths = function([squares[0]], [squares[1]], pc.FillRule.NON_ZERO)
    assert sum(abs(pc.area(path)) for path in paths) == pytest.approx(expected)


def test_disjoint_rectangle_union_has_two_components():
    a = pc.make_path([[0, 0], [1, 0], [1, 1], [0, 1]])
    b = pc.make_path([[3, 0], [4, 0], [4, 1], [3, 1]])
    assert len(pc.union([a, b])) == 2


def test_miter_offset_expands_square_by_delta(squares):
    result = pc.inflate_paths([squares[0]], 2, pc.JoinType.MITER, pc.EndType.POLYGON)
    assert coords(result[0]) == [(-2, -2), (12, -2), (12, 12), (-2, 12)]
    assert pc.area(result[0]) == 196


def test_bevel_and_round_offsets_are_outward(squares):
    bevel = pc.inflate_paths([squares[0]], 2, pc.JoinType.BEVEL, pc.EndType.POLYGON)[0]
    rounded = pc.inflate_paths([squares[0]], 2, pc.JoinType.ROUND, pc.EndType.POLYGON)[0]
    assert len(bevel) == 8 and len(rounded) > len(bevel)
    assert pc.area(bevel) > pc.area(squares[0])
    assert pc.area(rounded) > pc.area(squares[0])


def test_square_offset_is_covered(squares):
    square = pc.inflate_paths([squares[0]], 2, pc.JoinType.SQUARE, pc.EndType.POLYGON)[0]
    assert len(square) == 8
    assert pc.area(square) > pc.area(squares[0])


def test_unsupported_offset_options_fail_explicitly():
    acute = pc.make_path([[0, 0], [10, 0], [1, 1]])
    with pytest.raises(NotImplementedError, match="miter_limit"):
        pc.inflate_paths([acute], 1, pc.JoinType.MITER, pc.EndType.POLYGON)
    with pytest.raises(NotImplementedError, match="precision"):
        pc.inflate_paths([acute], 1, pc.JoinType.BEVEL, pc.EndType.POLYGON, precision=3)


def test_clipper_offset_mutates_solution_as_upstream_does(squares):
    offset, solution = pc.ClipperOffset(), []
    offset.add_path(squares[0], pc.JoinType.MITER, pc.EndType.POLYGON)
    assert offset.execute(1, solution) is None
    assert len(solution) == 1 and pc.area(solution[0]) == 144
    offset.clear()
    offset.execute(1, solution)
    assert len(solution) == 1


def test_nonconvex_overlap_is_refused_not_fabricated():
    concave = pc.make_path([[0, 0], [4, 0], [4, 4], [2, 2], [0, 4]])
    with pytest.raises(NotImplementedError):
        pc.intersection([concave], [concave])
    with pytest.raises(NotImplementedError):
        pc.inflate_paths([concave], 1, pc.JoinType.MITER, pc.EndType.POLYGON)


def test_double_paths_keep_float_coordinates():
    path = pc.make_path_double([[0.25, 0.5], [2.25, 0.5], [0.25, 2.5]])
    assert pc.area(path) == pytest.approx(2.0)
    assert isinstance(pc.inflate_paths([path], 0.25, pc.JoinType.MITER, pc.EndType.POLYGON, miter_limit=3)[0][0], pc.PointD)


def test_simd_area_and_scalar_tail_match_shoelace():
    path = pc.make_path_double([
        (math.cos(2 * math.pi * i / 13), math.sin(2 * math.pi * i / 13))
        for i in range(13)
    ])
    assert pc.area(path) == pytest.approx(reference_area(coords(path)), abs=1e-12)
    assert pc.point_in_polygon((0, 0), path) == pc.PointInPolygonResult.IS_INSIDE


def test_contiguous_float64_paths_stay_zero_copy():
    path = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    assert pc.area(path) == 6.0
