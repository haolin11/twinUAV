from __future__ import annotations
import os, time
import numpy as np
from typing import Tuple

try:
    import genesis as gs
except ModuleNotFoundError as e:
    gs = None

from stereo_rig import StereoRig
from utils import quat_wxyz_to_R, yaw_to_quat_wxyz


class GenesisSim:
    def __init__(self, urdf: str, base_link: str, viewer: bool):
        if gs is None:
            raise RuntimeError("Genesis 未安装或不可用，请按官方文档安装后再运行。")
        gs.init(seed=0)
        self.scene = gs.Scene(
            show_viewer=viewer,
            viewer_options=gs.options.ViewerOptions(res=(1280, 960), camera_fov=50, max_FPS=60),
            vis_options=gs.options.VisOptions(show_world_frame=True),
            renderer=gs.renderers.Rasterizer(),
        )
        self.scene.add_entity(gs.morphs.Plane())
        # 选择无人机：优先使用内置 quadrotor，无内置则尝试 URDF，再退化为 Box
        self.drone = None
        self.base_link = None
        try:
            if isinstance(urdf, str) and urdf.startswith('builtin:'):
                name = urdf.split(':',1)[1].strip().lower()
                if hasattr(gs.morphs, 'Quadrotor') and name in ("genesis_quadrotor", "quadrotor", "uav"):
                    self.drone = self.scene.add_entity(gs.morphs.Quadrotor())
                elif hasattr(gs.morphs, 'Drone'):
                    self.drone = self.scene.add_entity(gs.morphs.Drone())
            elif isinstance(urdf, str) and os.path.exists(urdf):
                if hasattr(gs.morphs, 'Drone'):
                    self.drone = self.scene.add_entity(gs.morphs.Drone(file=urdf, scale=1.0))
        except Exception:
            self.drone = None
        if self.drone is None:
            # 后备：用 Box 代替机体以保证流程可运行
            try:
                self.drone = self.scene.add_entity(gs.morphs.Box(size=(0.3,0.3,0.1), pos=(0,0,0.5)))
            except Exception:
                pass
        # 尝试获取基础 link，不存在则用 drone 自身
        try:
            self.base_link = self.drone.get_link(base_link) if hasattr(self.drone, 'get_link') else self.drone
        except Exception:
            self.base_link = self.drone

    def attach_stereo(self, rig: StereoRig):
        self.rig = rig
        self.cam_L = self.scene.add_camera(res=rig.res, pos=(0,0,0), lookat=(1,0,0), fov=rig.fov_v_deg, GUI=False)
        self.cam_R = self.scene.add_camera(res=rig.res, pos=(0,0,0), lookat=(1,0,0), fov=rig.fov_v_deg, GUI=False)
        self.scene.build()

    def set_obstacle_box(self, pos: Tuple[float,float,float], size: Tuple[float,float,float]):
        try:
            self.scene.add_entity(gs.morphs.Box(size=size, pos=pos))
        except Exception:
            pass

    def step(self):
        self.scene.step()

    def write_state(self, x,y,z,yaw):
        if hasattr(self.base_link, 'set_pos'):
            try: self.base_link.set_pos((x,y,z))
            except Exception: pass
        if hasattr(self.base_link, 'set_quat'):
            try: self.base_link.set_quat(yaw_to_quat_wxyz(yaw))
            except Exception: pass

    def update_cameras(self, x,y,z,yaw):
        pos_w = np.array([x,y,z], dtype=np.float32)
        R_wb = quat_wxyz_to_R(np.array(yaw_to_quat_wxyz(yaw), dtype=np.float32))
        R_mount = self.rig.MOUNT_R
        tL = self.rig.MOUNT_t_left
        tR = self.rig.right_offset()
        R_wc = R_wb @ R_mount
        t_wc_L = R_wb @ tL + pos_w
        t_wc_R = R_wb @ tR + pos_w
        fwd = (R_wb @ np.array([1.0,0.0,0.0], dtype=np.float32))
        self.cam_L.set_pose(pos=tuple(t_wc_L.tolist()), lookat=tuple((t_wc_L + fwd).tolist()))
        self.cam_R.set_pose(pos=tuple(t_wc_R.tolist()), lookat=tuple((t_wc_R + fwd).tolist()))
        return R_wc, t_wc_L, t_wc_R