from __future__ import annotations

from typing import List, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]


def translate(t: Vec3) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = float(t[0])
    m[1, 3] = float(t[1])
    m[2, 3] = float(t[2])
    return m


def rot_x(deg: float) -> np.ndarray:
    r = np.eye(4, dtype=np.float64)
    a = np.deg2rad(float(deg))
    c, s = np.cos(a), np.sin(a)
    r[1, 1] = c
    r[1, 2] = -s
    r[2, 1] = s
    r[2, 2] = c
    return r


def rot_y(deg: float) -> np.ndarray:
    r = np.eye(4, dtype=np.float64)
    a = np.deg2rad(float(deg))
    c, s = np.cos(a), np.sin(a)
    r[0, 0] = c
    r[0, 2] = s
    r[2, 0] = -s
    r[2, 2] = c
    return r


def rot_z(deg: float) -> np.ndarray:
    r = np.eye(4, dtype=np.float64)
    a = np.deg2rad(float(deg))
    c, s = np.cos(a), np.sin(a)
    r[0, 0] = c
    r[0, 1] = -s
    r[1, 0] = s
    r[1, 1] = c
    return r


def compose_euler_from_channels(rot_channels: List[str], rot_values: List[float]) -> np.ndarray:
    """Compose rotation matrix in the exact channel order.

    BVH stores Euler angles in degrees. The CHANNELS order determines composition order.
    """
    m = np.eye(4, dtype=np.float64)
    for ch, v in zip(rot_channels, rot_values):
        if ch == "Xrotation":
            m = m @ rot_x(v)
        elif ch == "Yrotation":
            m = m @ rot_y(v)
        elif ch == "Zrotation":
            m = m @ rot_z(v)
        else:
            # ignore unknown rotation channels
            pass
    return m


def apply_transform(m: np.ndarray, p: Vec3) -> Vec3:
    v = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    out = m @ v
    return (float(out[0]), float(out[1]), float(out[2]))
