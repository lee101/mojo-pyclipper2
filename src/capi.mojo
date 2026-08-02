"""C ABI for the convex polygon kernels used by the Python compatibility API."""

from std.math import sqrt
from std.sys import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def signed_area(xy: Ptr, n: Int) -> Float64:
    if n < 3:
        return 0.0
    comptime W = simd_width_of[DType.float64]()
    comptime C = W // 2
    var accum = SIMD[DType.float64, W](0.0).deinterleave()[0]
    var i = 0
    while i + C < n:
        var current = xy.load[width=W, alignment=1](2 * i).deinterleave()
        var following = xy.load[width=W, alignment=1](2 * i + 2).deinterleave()
        accum += current[0] * following[1] - current[1] * following[0]
        i += C
    var total = accum.reduce_add()
    while i < n:
        var j = i + 1
        if j == n:
            j = 0
        total += xy[2 * i] * xy[2 * j + 1] - xy[2 * j] * xy[2 * i + 1]
        i += 1
    return total * 0.5


def copy_values(dst: Ptr, src: Ptr, count: Int):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= count:
        dst.store[alignment=1](i, src.load[width=W, alignment=1](i))
        i += W
    while i < count:
        dst[i] = src[i]
        i += 1


@export("mpc_area")
def mpc_area(xy: Int, n: Int) abi("C") -> Float64:
    if n <= 0 or xy == 0:
        return 0.0
    return signed_area(Ptr(unsafe_from_address=xy), n)


@export("mpc_point_in_polygon")
def mpc_point_in_polygon(px: Float64, py: Float64, xy: Int, n: Int) abi("C") -> Int:
    if n < 3 or xy == 0:
        return 0
    var p = Ptr(unsafe_from_address=xy)
    var inside = False
    for i in range(n):
        var j = i + 1
        if j == n:
            j = 0
        var ax = p[2 * i]
        var ay = p[2 * i + 1]
        var bx = p[2 * j]
        var by = p[2 * j + 1]
        var cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
        var minx = ax if ax < bx else bx
        var maxx = bx if ax < bx else ax
        var miny = ay if ay < by else by
        var maxy = by if ay < by else ay
        if cross > -1e-12 and cross < 1e-12 and px >= minx and px <= maxx and py >= miny and py <= maxy:
            return -1
        if (ay > py) != (by > py):
            var xcross = ax + (py - ay) * (bx - ax) / (by - ay)
            if xcross > px:
                inside = not inside
    return 1 if inside else 0


def line_intersection(ax: Float64, ay: Float64, bx: Float64, by: Float64,
                      cx: Float64, cy: Float64, dx: Float64, dy: Float64,
                      dst: Ptr, k: Int):
    var r_x = bx - ax
    var r_y = by - ay
    var s_x = dx - cx
    var s_y = dy - cy
    var den = r_x * s_y - r_y * s_x
    if den > -1e-18 and den < 1e-18:
        dst[2 * k] = bx
        dst[2 * k + 1] = by
        return
    var t = ((cx - ax) * s_y - (cy - ay) * s_x) / den
    dst[2 * k] = ax + t * r_x
    dst[2 * k + 1] = ay + t * r_y


@export("mpc_convex_intersection")
def mpc_convex_intersection(a: Int, n: Int, b: Int, m: Int,
                            scratch_a: Int, scratch_b: Int, cap: Int) abi("C") -> Int:
    if n < 3 or m < 3 or cap < n or a == 0 or b == 0 or scratch_a == 0 or scratch_b == 0:
        return 0
    var subject = Ptr(unsafe_from_address=a)
    var clip = Ptr(unsafe_from_address=b)
    var cur = Ptr(unsafe_from_address=scratch_a)
    var nxt = Ptr(unsafe_from_address=scratch_b)
    copy_values(cur, subject, n * 2)
    var count = n
    var winding = signed_area(clip, m)
    for edge in range(m):
        if count == 0:
            break
        var edge_next = edge + 1
        if edge_next == m:
            edge_next = 0
        var cx = clip[2 * edge]
        var cy = clip[2 * edge + 1]
        var dx = clip[2 * edge_next]
        var dy = clip[2 * edge_next + 1]
        var write = 0
        for ii in range(count):
            var jj = ii + 1
            if jj == count:
                jj = 0
            var ax = cur[2 * ii]
            var ay = cur[2 * ii + 1]
            var bx = cur[2 * jj]
            var by = cur[2 * jj + 1]
            var ca = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
            var cb = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
            var a_in = ca >= -1e-12 if winding >= 0.0 else ca <= 1e-12
            var b_in = cb >= -1e-12 if winding >= 0.0 else cb <= 1e-12
            if a_in != b_in:
                if write >= cap:
                    return -1
                line_intersection(ax, ay, bx, by, cx, cy, dx, dy, nxt, write)
                write += 1
            if b_in:
                if write >= cap:
                    return -1
                nxt[2 * write] = bx
                nxt[2 * write + 1] = by
                write += 1
        copy_values(cur, nxt, write * 2)
        count = write
    return count


@export("mpc_offset_miter")
def mpc_offset_miter(xy: Int, n: Int, delta: Float64, dst_addr: Int) abi("C") -> Int:
    if n < 3 or xy == 0 or dst_addr == 0:
        return 0
    var src = Ptr(unsafe_from_address=xy)
    var dst = Ptr(unsafe_from_address=dst_addr)
    var winding = signed_area(src, n)
    for i in range(n):
        var prev = i - 1
        if prev < 0:
            prev = n - 1
        var following = i + 1
        if following == n:
            following = 0
        var e1x = src[2 * i] - src[2 * prev]
        var e1y = src[2 * i + 1] - src[2 * prev + 1]
        var e2x = src[2 * following] - src[2 * i]
        var e2y = src[2 * following + 1] - src[2 * i + 1]
        var l1 = sqrt(e1x * e1x + e1y * e1y)
        var l2 = sqrt(e2x * e2x + e2y * e2y)
        if l1 <= 1e-18 or l2 <= 1e-18:
            return 0
        var sign = 1.0 if winding >= 0.0 else -1.0
        var n1x = sign * e1y / l1
        var n1y = -sign * e1x / l1
        var n2x = sign * e2y / l2
        var n2y = -sign * e2x / l2
        var sx = n1x + n2x
        var sy = n1y + n2y
        var denom = 1.0 + n1x * n2x + n1y * n2y
        if denom <= 1e-12:
            return 0
        dst[2 * i] = src[2 * i] + delta * sx / denom
        dst[2 * i + 1] = src[2 * i + 1] + delta * sy / denom
    return n
