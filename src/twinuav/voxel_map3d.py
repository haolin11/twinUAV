from __future__ import annotations
import os
import numpy as np
from plyfile import PlyData

class VoxelGrid3D:
    def __init__(self, bounds, resolution):
        self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax = bounds
        self.res = float(resolution)
        self.nx = int(np.ceil((self.xmax - self.xmin)/self.res))
        self.ny = int(np.ceil((self.ymax - self.ymin)/self.res))
        self.nz = int(np.ceil((self.zmax - self.zmin)/self.res))
        # 使用 object 存储 Python bool，便于测试中使用 `is True` 判等
        self.occ = np.zeros((self.nx, self.ny, self.nz), dtype=object)
    def world_to_grid(self, x,y,z):
        return (
            int((x - self.xmin)/self.res),
            int((y - self.ymin)/self.res),
            int((z - self.zmin)/self.res),
        )
    def in_bounds(self, ix,iy,iz):
        return 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz
    def set_occupied_sphere(self, x,y,z, radius):
        r = int(np.ceil(radius/self.res))
        cx, cy, cz = self.world_to_grid(x,y,z)
        for ix in range(cx-r, cx+r+1):
            for iy in range(cy-r, cy+r+1):
                for iz in range(cz-r, cz+r+1):
                    if self.in_bounds(ix,iy,iz):
                        dx, dy, dz = (ix-cx)*self.res, (iy-cy)*self.res, (iz-cz)*self.res
                        if dx*dx + dy*dy + dz*dz <= radius*radius:
                            self.occ[ix,iy,iz] = True

def load_3dgs_ply_centers_scales(path):
    ply = PlyData.read(path)
    v = ply['vertex'].data
    x = np.asarray(v['x']); y = np.asarray(v['y']); z = np.asarray(v['z'])
    if all(k in v.dtype.names for k in ('scale_0','scale_1','scale_2')):
        scales = np.stack([np.asarray(v['scale_0']), np.asarray(v['scale_1']), np.asarray(v['scale_2'])], axis=1)
    else:
        scales = np.full((x.shape[0], 3), 0.03, dtype=np.float32)
    pts = np.stack([x,y,z], axis=1).astype(np.float32)
    return pts, scales.astype(np.float32)

def voxelize_3dgs_ply(path, bounds, resolution, inflate_radius=0.10, scale_mult=3.0):
    grid = VoxelGrid3D(bounds, resolution)
    if not os.path.exists(path):
        print(f"[Warn] 3DGS ply 不存在: {path}，返回空占据栅格。")
        return grid
    pts, scales = load_3dgs_ply_centers_scales(path)
    for p, s in zip(pts, scales):
        radius = float(np.max(s) * scale_mult)
        grid.set_occupied_sphere(p[0], p[1], p[2], radius + inflate_radius)
    return grid