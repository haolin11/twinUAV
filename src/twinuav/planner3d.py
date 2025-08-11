from __future__ import annotations
from dataclasses import dataclass
import math, yaml, argparse
from heapq import heappush, heappop
from typing import Tuple, Optional, Dict
from .voxel_map3d import VoxelGrid3D, voxelize_3dgs_ply

@dataclass(order=True)
class _Node3D:
    f: float
    g: float
    x: float
    y: float
    z: float
    yaw: float
    parent: Optional[Tuple[float,float,float,float]]

class HybridAStar3D:
    def __init__(self, grid: VoxelGrid3D, step_xyz: float = 0.15, step_yaw: float = 0.35, n_heading: int = 16):
        self.grid = grid; self.step = step_xyz; self.dyaw = step_yaw; self.nh = n_heading
    def _heuristic(self, x,y,z,yaw, gx,gy,gz):
        return math.sqrt((gx-x)**2 + (gy-y)**2 + (gz-z)**2) + 0.05*abs(yaw)
    def _free(self, x,y,z):
        ix = int((x - self.grid.xmin)/self.grid.res)
        iy = int((y - self.grid.ymin)/self.grid.res)
        iz = int((z - self.grid.zmin)/self.grid.res)
        if not self.grid.in_bounds(ix,iy,iz): return False
        return not self.grid.occ[ix,iy,iz]
    def _neighbors(self, x,y,z,yaw):
        for dyaw in (0.0, +self.dyaw, -self.dyaw):
            yaw2 = yaw + dyaw
            for vz in (-self.step, 0.0, +self.step):
                x2 = x + self.step * math.cos(yaw2)
                y2 = y + self.step * math.sin(yaw2)
                z2 = z + vz
                if self._free(x2,y2,z2):
                    yield (x2,y2,z2, ((yaw2+math.pi)%(2*math.pi))-math.pi)
    def plan(self, start: Tuple[float,float,float,float], goal_xyz: Tuple[float,float,float], max_iter=200000):
        sx,sy,sz,syaw = start; gx,gy,gz = goal_xyz
        start_node = _Node3D(self._heuristic(sx,sy,sz,syaw,gx,gy,gz), 0.0, sx,sy,sz,syaw, None)
        openq = []; heappush(openq, start_node)
        came: Dict[Tuple[float,float,float,float], Optional[Tuple[float,float,float,float]]] = {}
        cost: Dict[Tuple[float,float,float,float], float] = {}
        key0 = (round(sx,2), round(sy,2), round(sz,2), round(syaw,2))
        cost[key0] = 0.0
        it = 0; goal_key = None
        while openq and it < max_iter:
            it += 1
            cur = heappop(openq)
            key = (round(cur.x,2), round(cur.y,2), round(cur.z,2), round(cur.yaw,2))
            if math.sqrt((cur.x-gx)**2 + (cur.y-gy)**2 + (cur.z-gz)**2) < max(2*self.step, 0.2):
                goal_key = key; came[key] = cur.parent; break
            if key in came: continue
            came[key] = cur.parent
            for (nx,ny,nz,nyaw) in self._neighbors(cur.x,cur.y,cur.z,cur.yaw):
                nk = (round(nx,2), round(ny,2), round(nz,2), round(nyaw,2))
                ng = cost[key] + self.step
                if nk not in cost or ng < cost[nk]:
                    cost[nk] = ng
                    f = ng + self._heuristic(nx,ny,nz,nyaw,gx,gy,gz)
                    heappush(openq, _Node3D(f, ng, nx,ny,nz,nyaw, key))
        path = []
        if goal_key is None: return path
        k = goal_key
        while k is not None:
            x,y,z,yaw = k; path.append((x,y,z,yaw)); k = came.get(k, None)
        path.reverse(); return path

# CLI demo
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/planner3d_demo.yaml')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config,'r'))
    bounds = tuple(cfg['voxel']['bounds'])
    res = cfg['voxel']['resolution']
    inflate = cfg['voxel']['inflate_radius']
    grid = voxelize_3dgs_ply(cfg['voxel']['ply'], bounds, res, inflate_radius=inflate)
    planner = HybridAStar3D(grid, step_xyz=cfg['plan3d']['step_xyz'], step_yaw=cfg['plan3d']['step_yaw'], n_heading=cfg['plan3d']['n_heading'])
    start = tuple(cfg['plan3d']['start'])
    goal = tuple(cfg['plan3d']['goal'])
    path = planner.plan(start, goal)
    print(f"3D path length: {len(path)}")