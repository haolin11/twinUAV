#!/usr/bin/env python3
from __future__ import annotations
import argparse, time
import numpy as np

import rospy
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from twinuav.stereo_rig import StereoRig
from twinuav.calib_io import load_camera_yaml, camera_info_from_yaml

class VINSBridge:
    def __init__(self, cfg):
        self.rate_cam = cfg['ros']['rate_cam']
        self.rate_imu = cfg['ros']['rate_imu']
        tp = cfg['ros']['topics']
        self.pub_cam0 = rospy.Publisher(tp['cam0_img'], Image, queue_size=10)
        self.pub_cam1 = rospy.Publisher(tp['cam1_img'], Image, queue_size=10)
        self.pub_info0 = rospy.Publisher(tp['cam0_info'], CameraInfo, queue_size=10)
        self.pub_info1 = rospy.Publisher(tp['cam1_info'], CameraInfo, queue_size=10)
        self.pub_imu = rospy.Publisher(tp['imu'], Imu, queue_size=200)
        self.sub_vins = rospy.Subscriber(tp['vins_odom'], PoseStamped, self.vins_cb, queue_size=5)
        self.bridge = CvBridge()
        cam0_yaml = cfg['calib']['cam0']; cam1_yaml = cfg['calib']['cam1']
        self.K0, self.D0, self.P0, self.size0 = load_camera_yaml(cam0_yaml)
        self.K1, self.D1, self.P1, self.size1 = load_camera_yaml(cam1_yaml)
        self.info0 = camera_info_from_yaml(self.K0, self.D0, self.P0, self.size0)
        self.info1 = camera_info_from_yaml(self.K1, self.D1, self.P1, self.size1)
        self.latest_pose = None

    def vins_cb(self, msg: PoseStamped):
        self.latest_pose = msg

    def spin(self):
        r_cam = rospy.Rate(self.rate_cam)
        r_imu = rospy.Rate(self.rate_imu)
        next_cam = rospy.Time.now(); next_imu = rospy.Time.now()
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if now >= next_cam:
                frame0 = np.zeros((self.size0[1], self.size0[0], 3), dtype=np.uint8)
                frame1 = np.zeros((self.size1[1], self.size1[0], 3), dtype=np.uint8)
                self.info0.header.stamp = now; self.info1.header.stamp = now
                self.pub_info0.publish(self.info0); self.pub_info1.publish(self.info1)
                self.pub_cam0.publish(self.bridge.cv2_to_imgmsg(frame0, encoding='bgr8'))
                self.pub_cam1.publish(self.bridge.cv2_to_imgmsg(frame1, encoding='bgr8'))
                next_cam = now + rospy.Duration(1.0/self.rate_cam)
            if now >= next_imu:
                imu = Imu(); imu.header.stamp = now
                # 这里保持零 IMU；实际运行时建议从 `run_demo` 产生的 npz 回放或接入仿真串流
                self.pub_imu.publish(imu)
                next_imu = now + rospy.Duration(1.0/self.rate_imu)
            rospy.sleep(0.0005)

if __name__ == '__main__':
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/vins_ros.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    rospy.init_node('twinuav_vins_bridge')
    VINSBridge(cfg).spin()