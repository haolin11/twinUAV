from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass
class KinState:
    x: float; y: float; z: float; yaw: float

class KinematicUAV:
    def __init__(self, init: KinState, z_ref: float = 1.0):
        self.s = init; self.z_ref = z_ref
    def step(self, v_fwd: float, yaw_rate: float, dt: float):
        self.s.yaw += yaw_rate * dt
        self.s.x += v_fwd * math.cos(self.s.yaw) * dt
        self.s.y += v_fwd * math.sin(self.s.yaw) * dt
        self.s.z += 1.5 * (self.z_ref - self.s.z) * dt
        return self.s

class BodyRateController:
    def __init__(self, v_max: float = 1.0, yaw_rate_max: float = 1.0):
        self.v_max = v_max; self.yaw_rate_max = yaw_rate_max
    def track(self, state: KinState, target_xy: tuple[float, float]):
        dx = target_xy[0] - state.x; dy = target_xy[1] - state.y
        dist = (dx*dx + dy*dy) ** 0.5
        tgt_yaw = math.atan2(dy, dx)
        yaw_err = (tgt_yaw - state.yaw + math.pi) % (2*math.pi) - math.pi
        yaw_rate = max(-self.yaw_rate_max, min(self.yaw_rate_max, 1.5*yaw_err))
        v = min(self.v_max, 0.8*dist)
        return v, yaw_rate