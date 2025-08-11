import numpy as np
from twinuav.voxel_map3d import VoxelGrid3D

def test_voxel_sphere_mark():
    vg = VoxelGrid3D(bounds=(-1,-1,-1,1,1,1), resolution=0.1)
    vg.set_occupied_sphere(0.0,0.0,0.0, radius=0.15)
    assert vg.occ.sum() > 0
    cx,cy,cz = vg.world_to_grid(0.0,0.0,0.0)
    assert vg.occ[cx,cy,cz] is True