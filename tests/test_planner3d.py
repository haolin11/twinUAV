import numpy as np
from twinuav.voxel_map3d import VoxelGrid3D
from twinuav.planner3d import HybridAStar3D

def test_planner3d_simple_corridor():
    bounds = (-1,-1,0, 1,1,1)
    vg = VoxelGrid3D(bounds, 0.1)
    for ix in range(0, vg.nx):
        for iz in range(0, vg.nz):
            iy = vg.ny//2
            if iz != vg.nz//2:
                vg.occ[ix,iy,iz] = True
    planner = HybridAStar3D(vg, step_xyz=0.1, step_yaw=0.5, n_heading=8)
    path = planner.plan(start=(-0.8,-0.8,0.1,0.0), goal_xyz=(0.8,0.8,0.9))
    assert isinstance(path, list) and len(path) > 0