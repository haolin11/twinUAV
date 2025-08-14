# UAV-3DGS-Genesis

室内无人机数字孪生与数据工厂：Genesis 仿真 + 双目相机 + 3DGS 渲染 + 3D 体素地图 + 3维混合 A* + 规则触发采集 + ROS Noetic 桥接 VINS-Fusion。

## 项目目标
- **Real→Sim**：用 3DGS 还原真实室内，获得高逼真的视觉帧；
- **可飞行**：Genesis 提供物理/时间轴/多传感器；
- **自动采集**：以“规则”为触发（如“靠近桌子”）自动批量采集 `(o_t, a_t, o_{t+1})`；
- **VIO 对接**：ROS Noetic 下，以 **VINS-Fusion (stereo+IMU)** 作为定位源（D435i 30 Hz + Pixhawk 6C Mini 150 Hz）；
- **路径规划**：基于 **3DGS 点云体素化的 3D 占据栅格**，进行 **3维混合 A*** 搜索；
- **可扩展**：替换/插拔 VIO、规划、控制模块。

## 功能概述
- 双目相机挂架（默认面向 **Intel RealSense D435i**：30 Hz，baseline≈0.05 m，可由标定文件覆盖）；
- 3DGS 渲染适配层（OpenCV/OpenGL 坐标；gsplat 或自定义后端）；
- **3D 体素地图**：从 3DGS `.ply`（高斯中心与尺度）或点云生成 3D 占据栅格（支持膨胀/边界裁剪）；
- **3D Hybrid A***：在 3D 栅格中搜索 `(x,y,z,yaw)` 可行路径（离散 yaw）；
- ROS Noetic 桥接：发布 `/cam0`,`/cam1` 图像与 CameraInfo（30 Hz）， `/imu0`（150 Hz），订阅 `/vins_estimator/odometry`；
- 规则引擎（一次性触发 + 邻近判断）；
- 数据输出（左右目 RGB、外参、K、位姿/动作/时间戳）。

## 依赖
- Python 3.9+
- 必需：numpy, pyyaml, plyfile
- 可选：torch, gsplat, opencv-python（非 ROS 环境下图像处理）
- **Genesis**：需按官方指引安装；未安装时，`run_demo.py` 会给出清晰错误。
- **ROS Noetic**（Ubuntu 20.04）：`roscore`、`rospy`、`cv_bridge`、`image_transport`、`sensor_msgs`、`geometry_msgs` 等（使用 apt 安装）。

## 安装
```bash
pip install -e .
# 或最小依赖
pip install -r requirements.txt
```
## 快速开始
```bash
# 1) 创建隔离环境并安装最小依赖
mamba create -n twinuav python=3.10 -y
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install genesis-world==0.2.1 mujoco transformers==4.51.3 gdown setuptools==59.5 libigl==2.5.1 pyglet gsplat==0.1.11 open3d



export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8
(twinuav) dell@dell-Precision-3680 [~/twinmanip] git:(master) ✗ ➜  pip install -r simulation/requirements.txt --verbose
mamba install -n twinuav -c conda-forge "libstdcxx-ng>=13" "libgcc-ng>=13"
mamba run -n twinuav python -m pip install -r requirements.txt

# 2) 运行测试（无需 Genesis）
mamba run -n twinuav bash -lc "PYTHONPATH=src pytest -q"

# 3) 启动 demo（需要 Genesis，URDF/3DGS 路径需存在）
mamba run -n twinuav python -m twinuav.run_demo --config configs/demo_default.yaml --viewer

# 4) 将数据回放到 ROS Noetic 供 VINS-Fusion （需在 ROS 环境中）
roscore &
ROS_NAMESPACE=/ mamba run -n twinuav python -m twinuav.ros1_playback \
  --dataset outputs/dataset.npz --config configs/vins_ros.yaml
```

说明：
- 若未安装 `genesis` 或缺少 `assets/3dgs/*.ply`，demo 将退化为黑屏渲染/空占据栅格，但流程可跑通。
- `sam_segment.py` 提供可选的 Segment-Anything 推理器包装，若未安装其依赖将自动回退为 no-op。

nv -u LD_LIBRARY_PATH LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia PYOPENGL_PLATFORM=glx /home/dell/.local/share/mamba/envs/twinuav/bin/python /home/dell/twinUAV/src/twinuav/run_circle.py --config /home/dell/twinUAV/configs/demo_default.yaml | cat

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python /home/dell/twinUAV/src/twinuav/run_circle.py --config /home/dell/twinUAV/configs/demo_default.yaml

python src/twinuav/run_circle.py --config /data/home/chenhaolin/twinUAV/configs/demo_default.yaml