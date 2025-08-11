#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo, Imu
from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3, Quaternion
from twinuav.calib_io import load_camera_yaml, camera_info_from_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='npz file saved by run_demo')
    parser.add_argument('--config', type=str, default='configs/vins_ros.yaml')
    args = parser.parse_args()

    cfg = __import__('yaml').safe_load(open(args.config, 'r'))
    rate_cam = cfg['ros']['rate_cam']
    rate_imu = cfg['ros']['rate_imu']
    tp = cfg['ros']['topics']

    data = np.load(args.dataset)
    imgsL = data['rgb_left']
    imgsR = data['rgb_right']
    K = data['K']
    gyro = data.get('gyro', None)
    accel = data.get('accel', None)

    K0, D0, P0, size0 = load_camera_yaml(cfg['calib']['cam0'])
    K1, D1, P1, size1 = load_camera_yaml(cfg['calib']['cam1'])
    info0 = camera_info_from_yaml(K0, D0, P0, size0)
    info1 = camera_info_from_yaml(K1, D1, P1, size1)

    rospy.init_node('twinuav_ros1_playback')
    pub_cam0 = rospy.Publisher(tp['cam0_img'], Image, queue_size=10)
    pub_cam1 = rospy.Publisher(tp['cam1_img'], Image, queue_size=10)
    pub_info0 = rospy.Publisher(tp['cam0_info'], CameraInfo, queue_size=10)
    pub_info1 = rospy.Publisher(tp['cam1_info'], CameraInfo, queue_size=10)
    pub_imu = rospy.Publisher(tp['imu'], Imu, queue_size=200)
    br = CvBridge()

    r_cam = rospy.Rate(rate_cam)
    r_imu = rospy.Rate(rate_imu)
    t0 = rospy.Time.now()
    idx = 0
    imu_idx = 0

    while not rospy.is_shutdown() and idx < len(imgsL):
        now = rospy.Time.now()
        # publish camera
        imgL = imgsL[idx]
        imgR = imgsR[idx]
        info0.header.stamp = now
        info1.header.stamp = now
        pub_info0.publish(info0)
        pub_info1.publish(info1)
        pub_cam0.publish(br.cv2_to_imgmsg(imgL, encoding='bgr8'))
        pub_cam1.publish(br.cv2_to_imgmsg(imgR, encoding='bgr8'))
        idx += 1

        # publish several IMU ticks between frames
        imu_ticks = max(1, int(float(rate_imu)/float(rate_cam)))
        for _ in range(imu_ticks):
            imu_msg = Imu()
            imu_msg.header.stamp = rospy.Time.now()
            if gyro is not None and imu_idx < len(gyro):
                gx, gy, gz = gyro[imu_idx]
                imu_msg.angular_velocity.x = float(gx)
                imu_msg.angular_velocity.y = float(gy)
                imu_msg.angular_velocity.z = float(gz)
            if accel is not None and imu_idx < len(accel):
                ax, ay, az = accel[imu_idx]
                imu_msg.linear_acceleration.x = float(ax)
                imu_msg.linear_acceleration.y = float(ay)
                imu_msg.linear_acceleration.z = float(az)
            pub_imu.publish(imu_msg)
            imu_idx += 1
            r_imu.sleep()

        r_cam.sleep()


if __name__ == '__main__':
    main()


