from __future__ import annotations
import math
from typing import Tuple

class RuleEngine:
    def __init__(self, objects: dict):
        self.objects = objects; self.triggered = set()
    def near(self, name: str, pos: Tuple[float, float, float], radius: float) -> bool:
        if name not in self.objects: return False
        ox, oy, oz = self.objects[name]
        dx, dy, dz = pos[0]-ox, pos[1]-oy, pos[2]-oz
        return math.sqrt(dx*dx + dy*dy + dz*dz) <= radius
    def once(self, key: str, cond: bool) -> bool:
        if cond and key not in self.triggered:
            self.triggered.add(key); return True
        return False