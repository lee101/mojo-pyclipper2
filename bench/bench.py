"""Measured Mojo kernel timings versus independent pure-Python references."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import pyclipper2 as pc  # noqa: E402


def timer(fn, reps=5):
    best, value = float("inf"), None
    for _ in range(reps):
        start = time.perf_counter()
        value = fn()
        best = min(best, time.perf_counter() - start)
    return best, value


def pure_area(path):
    return sum(a.x * b.y - b.x * a.y for a, b in zip(path, path[1:] + path[:1])) * 0.5


def pure_pip(point, path):
    inside = False
    for a, b in zip(path, path[1:] + path[:1]):
        if (a.y > point.y) != (b.y > point.y) and a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y) > point.x:
            inside = not inside
    return inside


def pure_convex_clip(subject, clip):
    output = [(p.x, p.y) for p in subject]
    for c, d in zip(clip, clip[1:] + clip[:1]):
        source, output = output, []
        for a, b in zip(source, source[1:] + source[:1]):
            ca = (d.x - c.x) * (a[1] - c.y) - (d.y - c.y) * (a[0] - c.x)
            cb = (d.x - c.x) * (b[1] - c.y) - (d.y - c.y) * (b[0] - c.x)
            if (ca >= 0) != (cb >= 0):
                rx, ry, sx, sy = b[0] - a[0], b[1] - a[1], d.x - c.x, d.y - c.y
                t = ((c.x - a[0]) * sy - (c.y - a[1]) * sx) / (rx * sy - ry * sx)
                output.append((a[0] + t * rx, a[1] + t * ry))
            if cb >= 0:
                output.append(b)
    return output


def row(name, mojo, reference, label):
    print(f"| {name} | {mojo * 1e3:.3f} ms | {reference * 1e3:.3f} ms | {reference / mojo:.2f}x | {label} |")


def main():
    n = 4096
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    subject = [pc.PointD(float(np.cos(a)), float(np.sin(a))) for a in angles]
    clip = [pc.PointD(float(0.8 * np.cos(a) + 0.2), float(0.8 * np.sin(a))) for a in angles]
    point = pc.PointD(0.1, 0.1)
    print("| kernel | mojo-pyclipper2 | pure Python | speedup | reference |")
    print("| --- | ---: | ---: | ---: | --- |")
    m, _ = timer(lambda: pc.area(subject))
    r, _ = timer(lambda: pure_area(subject))
    row("signed area, 4,096 vertices", m, r, "shoelace")
    m, _ = timer(lambda: pc.point_in_polygon(point, subject))
    r, _ = timer(lambda: pure_pip(point, subject))
    row("point in polygon, 4,096 vertices", m, r, "ray crossing")
    m, _ = timer(lambda: pc.intersection([subject], [clip]), reps=3)
    r, _ = timer(lambda: pure_convex_clip(subject, clip), reps=3)
    row("convex intersection, 4,096 + 4,096", m, r, "Sutherland-Hodgman")
    m, _ = timer(lambda: pc.inflate_paths([subject], 0.1, pc.JoinType.MITER, pc.EndType.POLYGON))
    row("convex miter offset, 4,096 vertices", m, m, "no independent fast reference")


if __name__ == "__main__":
    main()
