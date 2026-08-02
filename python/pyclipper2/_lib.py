"""ctypes bridge for the single Mojo shared object."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIBRARY = os.path.join(ROOT, "dist", "libmojo-pyclipper2.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpc_area": ([I, I], F),
    "mpc_point_in_polygon": ([F, F, I, I], I),
    "mpc_convex_intersection": ([I, I, I, I, I, I, I], I),
    "mpc_offset_miter": ([I, I, F, I], I),
}
_cached: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _cached
    if _cached is None:
        if not os.path.exists(LIBRARY):
            subprocess.run(["bash", os.path.join(ROOT, "build", "build.sh")], check=True)
        _cached = ctypes.CDLL(LIBRARY)
        for name, (args, result) in _SIGNATURES.items():
            fn = getattr(_cached, name)
            fn.argtypes, fn.restype = args, result
    return _cached


def f64(points) -> np.ndarray:
    return np.ascontiguousarray(points, dtype=np.float64)


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)
