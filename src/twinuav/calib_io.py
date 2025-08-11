from __future__ import annotations
import yaml
import numpy as np
try:
    from sensor_msgs.msg import CameraInfo  # type: ignore
except Exception:
    CameraInfo = None  # 在非 ROS 环境下允许导入

def load_camera_yaml(path: str):
    data = yaml.safe_load(open(path, 'r'))
    K = np.array(data['K'], dtype=np.float32).reshape(3,3)
    D = np.array(data.get('D', []), dtype=np.float32)
    P = np.array(data['P'], dtype=np.float32).reshape(3,4)
    size = (int(data['width']), int(data['height']))
    return K, D, P, size

def camera_info_from_yaml(K, D, P, size):
    if CameraInfo is None:
        raise RuntimeError("camera_info_from_yaml 需要 ROS sensor_msgs；请在 ROS 环境中调用或安装相应依赖。")
    msg = CameraInfo()
    msg.width, msg.height = int(size[0]), int(size[1])
    msg.K = K.reshape(-1).tolist()
    msg.D = D.reshape(-1).tolist()
    msg.R = np.eye(3, dtype=np.float32).reshape(-1).tolist()
    msg.P = P.reshape(-1).tolist()
    msg.distortion_model = 'plumb_bob'
    return msg