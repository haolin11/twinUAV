#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
import numpy as np
import yaml

import rospy
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from .stereo_rig import StereoRig
from .renderer_3dgs import Stereo3DGSRenderer
from .planner3d import HybridAStar3D
from .voxel_map3d import voxelize_3dgs_ply
from .control import KinState, KinematicUAV, BodyRateController
from .imu_sim import IMUSim
from .rules import RuleEngine
from .genesis_sim import GenesisSim
from .calib_io import load_camera_yaml, camera_info_from_yaml


class RosStreamSim:
    def __init__(self, sim_cfg: dict, ros_cfg: dict):
        # ROS pubs/subs
        self.rate_cam = int(ros_cfg['ros']['rate_cam'])
        self.rate_imu = int(ros_cfg['ros']['rate_imu'])
        tp = ros_cfg['ros']['topics']
        self.pub_cam0 = rospy.Publisher(tp['cam0_img'], Image, queue_size=10)
        self.pub_cam1 = rospy.Publisher(tp['cam1_img'], Image, queue_size=10)
        self.pub_info0 = rospy.Publisher(tp['cam0_info'], CameraInfo, queue_size=10)
        self.pub_info1 = rospy.Publisher(tp['cam1_info'], CameraInfo, queue_size=10)
        self.pub_imu = rospy.Publisher(tp['imu'], Imu, queue_size=200)
        self.sub_vins = rospy.Subscriber(tp['vins_odom'], PoseStamped, self.vins_cb, queue_size=5)
        self.bridge = CvBridge()
        # CameraInfo
        cam0_yaml = ros_cfg['calib']['cam0']; cam1_yaml = ros_cfg['calib']['cam1']
        K0, D0, P0, size0 = load_camera_yaml(cam0_yaml)
        K1, D1, P1, size1 = load_camera_yaml(cam1_yaml)
        self.info0 = camera_info_from_yaml(K0, D0, P0, size0)
        self.info1 = camera_info_from_yaml(K1, D1, P1, size1)

        # Genesis + UAV
        urdf = sim_cfg['urdf']; base_link = sim_cfg.get('base_link','base_link')
        viewer = bool(sim_cfg.get('viewer', False))
        try:
            self.sim = GenesisSim(urdf=urdf, base_link=base_link, viewer=viewer)
        except Exception as e:
            if viewer:
                print(f"[Warn] 可视化窗口初始化失败，自动切换为无窗口模式（headless）：{e}")
                self.sim = GenesisSim(urdf=urdf, base_link=base_link, viewer=False)
            else:
                raise

        # Objects
        obj_cfg = sim_cfg.get('objects',{}).get('table', None)
        if obj_cfg:
            self.sim.set_obstacle_box(tuple(obj_cfg['pos']), tuple(obj_cfg['size']))

        # Stereo rig
        s = sim_cfg['stereo']
        self.rig = StereoRig(baseline=s['baseline'], res=tuple(s['res']), fov_v_deg=s['fov_v_deg'])
        self.rig.cam0_yaml = s.get('cam0_yaml', None)
        self.rig.cam1_yaml = s.get('cam1_yaml', None)
        _ = self.rig.load_from_yaml()
        self.sim.attach_stereo(self.rig)

        # Renderer
        rcfg = sim_cfg['renderer']
        self.renderer = Stereo3DGSRenderer(rcfg['model'], backend_coords=rcfg.get('backend_coords','opencv'))

        # 3D voxel map + planner
        vcfg = sim_cfg['voxel']
        grid3d = voxelize_3dgs_ply(vcfg['ply'], tuple(vcfg['bounds']), float(vcfg['resolution']), inflate_radius=float(vcfg['inflate_radius']))
        pcfg = sim_cfg['plan3d']
        self.start = tuple(pcfg['start'])
        self.goal = tuple(pcfg['goal'])
        self.planner = HybridAStar3D(grid3d, step_xyz=pcfg['step_xyz'], step_yaw=pcfg['step_yaw'], n_heading=pcfg['n_heading'])
        self.path = self.planner.plan(start=self.start, goal_xyz=self.goal)

        # Control + IMU
        ccfg = sim_cfg['control']
        self.kin = KinematicUAV(KinState(x=self.start[0], y=self.start[1], z=self.start[2], yaw=self.start[3]), z_ref=self.start[2])
        self.ctrl = BodyRateController(v_max=ccfg['v_max'], yaw_rate_max=ccfg['yaw_rate_max'])
        self.imu_sim = IMUSim()

        # Rules
        self.rules = RuleEngine({'table': tuple(obj_cfg['pos'])}) if obj_cfg else RuleEngine({})
        self.cap = sim_cfg['capture']
        self.near_key = f"near_{self.cap['near_object']}"

        self.path_idx = 0
        self.latest_vins = None

    def vins_cb(self, msg: PoseStamped):
        self.latest_vins = msg

    def spin(self):
        rate = float(self.sim.scene._viewer_options.max_FPS if hasattr(self.sim, 'scene') else 240.0)
        dt = float(1.0/rate)
        next_cam = rospy.Time.now()
        next_imu = rospy.Time.now()
        while not rospy.is_shutdown():
            self.sim.step()
            # Path tracking
            if self.path and self.path_idx < len(self.path):
                tx, ty, tz, tth = self.path[self.path_idx]
                if ( (self.kin.s.x - tx)**2 + (self.kin.s.y - ty)**2 + (self.kin.s.z - tz)**2 ) ** 0.5 < 0.25:
                    self.path_idx += 1
                target_xy = (tx, ty)
                self.kin.z_ref = tz
            else:
                target_xy = (self.goal[0], self.goal[1])
                self.kin.z_ref = self.goal[2]
            v, r = self.ctrl.track(self.kin.s, target_xy)
            self.kin.step(v, r, dt)
            self.sim.write_state(self.kin.s.x, self.kin.s.y, self.kin.s.z, self.kin.s.yaw)
            R_wc, tL, tR = self.sim.update_cameras(self.kin.s.x, self.kin.s.y, self.kin.s.z, self.kin.s.yaw)

            now = rospy.Time.now()
            # IMU publish at rate_imu
            if now >= next_imu:
                gyro, accel = self.imu_sim.step(v, r, self.kin.s.yaw)
                imu = Imu(); imu.header.stamp = now
                imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = [float(x) for x in gyro]
                imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = [float(x) for x in accel]
                self.pub_imu.publish(imu)
                next_imu = now + rospy.Duration(1.0/self.rate_imu)

            # Camera publish at rate_cam
            if now >= next_cam:
                imgL = self.renderer.render(self.rig.K, R_wc, tL, self.rig.res)
                imgR = self.renderer.render(self.rig.K, R_wc, tR, self.rig.res)
                self.info0.header.stamp = now; self.info1.header.stamp = now
                self.pub_info0.publish(self.info0); self.pub_info1.publish(self.info1)
                self.pub_cam0.publish(self.bridge.cv2_to_imgmsg(imgL, encoding='bgr8'))
                self.pub_cam1.publish(self.bridge.cv2_to_imgmsg(imgR, encoding='bgr8'))
                next_cam = now + rospy.Duration(1.0/self.rate_cam)

            rospy.sleep(max(0.0, 0.0005))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim_config', type=str, default='configs/demo_default.yaml')
    ap.add_argument('--ros_config', type=str, default='configs/vins_ros.yaml')
    args = ap.parse_args()
    sim_cfg = yaml.safe_load(open(args.sim_config,'r'))
    ros_cfg = yaml.safe_load(open(args.ros_config,'r'))
    rospy.init_node('twinuav_ros_stream_sim')
    RosStreamSim(sim_cfg, ros_cfg).spin()


if __name__ == '__main__':
    main()


