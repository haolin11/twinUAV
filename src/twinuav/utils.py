from __future__ import annotations
import math
import numpy as np

def quat_wxyz_to_R(q):
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R

def yaw_to_quat_wxyz(yaw: float):
    c = math.cos(0.5 * yaw)
    s = math.sin(0.5 * yaw)
    return (c, 0.0, 0.0, s)

def fov_v_to_K(width: int, height: int, fov_v_deg: float):
    fov_v = math.radians(fov_v_deg)
    fy = (height/2.0) / math.tan(fov_v/2.0)
    fx = fy
    cx, cy = width/2.0, height/2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)