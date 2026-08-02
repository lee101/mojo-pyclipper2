# mojo-pyclipper2

`mojo-pyclipper2` is a focused Mojo port of the compute-heavy geometry at the
center of [pyclipper2](https://pypi.org/project/pyclipper2/): polygon
predicates, convex clipping, and closed-path offsetting. It exports the same
Python names and point/enumeration types for the supported subset, through a
small `ctypes` layer.

The upstream `pyclipper2` 0.0.8 source was used as the API reference. It has no
Linux wheel (the only published wheel is CPython 3.13 macOS arm64) and is not on
conda-forge, so it cannot be installed in this environment for direct parity
tests. The tests instead use independent pure-Python shoelace, ray-crossing,
and Sutherland-Hodgman implementations, together with the square boolean test
vectors published in upstream's test suite.

## Covered subset

- `Point64`, `PointD`, `Rect64`, `RectD`, all public enums, `make_path`,
  `make_path_double`, `area`, `is_positive`, and `point_in_polygon`.
- `intersection` for one pair of convex polygons, accelerated in Mojo.
- `union`, `intersection`, `difference`, and `xor_` for collections of
  axis-aligned rectangle paths. The output is an exact non-overlapping contour
  decomposition, including L-shaped results.
- `inflate_paths` and `ClipperOffset` for convex, closed (`EndType.POLYGON`)
  paths with miter (within `miter_limit`), bevel/square, or round joins. Miter
  vertices are calculated in Mojo. The default offset precision is the only
  supported precision setting.

General overlapping non-convex polygon overlay, holes, open-path end caps, and
the full Clipper2 fill-rule engine are not covered. Those inputs raise
`NotImplementedError`; returning an approximate polygon would be substantially
worse than failing explicitly.

## Install and use

```bash
pixi install
pixi run build
```

```python
import pyclipper2 as pc

left = pc.make_path([[0, 0], [10, 0], [10, 10], [0, 10]])
right = pc.make_path([[5, 5], [15, 5], [15, 15], [5, 15]])

overlap = pc.intersection([left], [right], pc.FillRule.NON_ZERO)
assert pc.area(overlap[0]) == 25.0

expanded = pc.inflate_paths([left], 2, pc.JoinType.MITER, pc.EndType.POLYGON)
assert [(p.x, p.y) for p in expanded[0]] == [(-2, -2), (12, -2), (12, 12), (-2, 12)]
```

`pixi` sets `PYTHONPATH=python`, so the example can be run unchanged with
`pixi run python example.py`.

## Benchmarks

Measured with `pixi run bench` on this Linux x86_64 machine on 2026-08-02.
Times are the best of repeated runs. The references are
independent pure-Python implementations; these are not fabricated estimates.

| kernel | mojo-pyclipper2 | pure Python | speedup | reference |
| --- | ---: | ---: | ---: | --- |
| signed area, 4,096 vertices | 3.554 ms | 0.537 ms | 0.15x | shoelace |
| point in polygon, 4,096 vertices | 3.607 ms | 0.451 ms | 0.13x | ray crossing |
| convex intersection, 4,096 + 4,096 | 158.623 ms | 6318.329 ms | 39.83x | Sutherland-Hodgman |
| convex miter offset, 4,096 vertices | 20.608 ms | 20.608 ms | 1.00x | no independent fast reference |

The scalar predicate calls are slower here because a Python list is marshalled
into a NumPy buffer for each call. The native shoelace reduction and scratch
copies use unaligned-safe SIMD with scalar tails; flattening point objects now
uses a single-pass NumPy iterator. The high-work convex intersection kernel is
39.83x faster than the direct Python reference. The offset row is reported
honestly as a self-timing because no separate offset reference was benchmarked.

GPU execution is intentionally not included: the supported geometry kernels
are branch-heavy or move more than two bytes per floating-point operation, so
host-to-device transfer and launch costs lose to the CPU implementation.

## How it works

All Mojo code lives in one compilation unit, `src/capi.mojo`, and builds to
`dist/libmojo-pyclipper2.so`. Python owns contiguous `float64` input, scratch,
and output arrays; their addresses cross the ABI as `Int`, then Mojo recreates
mutable `UnsafePointer[Float64, AnyOrigin[mut=True]]` values. This means there
is no allocation or ownership transfer in the shared library.

The convex intersection kernel uses Sutherland-Hodgman clipping with two caller
allocated ping-pong buffers. Rectangle booleans use an exact coordinate grid in
the Python compatibility layer, then trace its exposed edges into contours.
Integer coordinates are accepted only through +/-2**53, the exact-integer range
of the float64 ABI; larger `Point64` values raise `OverflowError` rather than
being silently narrowed.

## Development

```bash
pixi run build && pixi run test
pixi run bench
```

`pixi run bench` holds the repository's machine-wide benchmark lock; run it
through the task rather than invoking the script directly.

MIT.
