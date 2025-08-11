from __future__ import annotations
import os
import argparse
import math
import numpy as np
import yaml

from .stereo_rig import StereoRig
from .renderer_3dgs import Stereo3DGSRenderer
from .genesis_sim import GenesisSim
from .utils import quat_wxyz_to_R, yaw_to_quat_wxyz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/demo_default.yaml')
    ap.add_argument('--viewer', action='store_true', default=False)
    ap.add_argument('--seconds', type=float, default=None, help='覆盖配置的飞行时长，单位秒')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))

    # Genesis + UAV
    urdf = cfg['urdf']; base_link = cfg.get('base_link','base_link')
    viewer = bool(cfg.get('viewer', args.viewer))
    sim = None
    try:
        sim = GenesisSim(urdf=urdf, base_link=base_link, viewer=viewer)
    except Exception as e:
        if viewer:
            print(f"[Warn] 可视化窗口初始化失败，自动切换为无窗口模式（headless）：{e}")
            try:
                sim = GenesisSim(urdf=urdf, base_link=base_link, viewer=False)
            except Exception as ee:
                print(f"[Warn] Genesis 不可用，使用无 Genesis 回退渲染：{ee}")
                sim = None
        else:
            print(f"[Warn] Genesis 不可用，使用无 Genesis 回退渲染：{e}")
            sim = None

    # Stereo rig
    s = cfg['stereo']
    rig = StereoRig(baseline=s['baseline'], res=tuple(s['res']), fov_v_deg=s['fov_v_deg'])
    rig.cam0_yaml = s.get('cam0_yaml', None)
    rig.cam1_yaml = s.get('cam1_yaml', None)
    _ = rig.load_from_yaml()
    if sim is not None:
        try:
            sim.attach_stereo(rig)
        except Exception as e:
            if viewer:
                print(f"[Warn] 构建可视化失败，自动重建为无窗口模式（headless）：{e}")
                try:
                    sim = GenesisSim(urdf=urdf, base_link=base_link, viewer=False)
                    sim.attach_stereo(rig)
                except Exception as ee:
                    print(f"[Warn] 无法启用 Genesis 场景，将继续以无 Genesis 路径渲染：{ee}")
                    sim = None
            else:
                print(f"[Warn] 无法附加相机/构建场景：{e}")

    # Renderer (支持 external 后端)
    rcfg = cfg['renderer']
    backend = rcfg.get('backend', None)
    external_path = rcfg.get('external_path', None)
    renderer = Stereo3DGSRenderer(rcfg['model'], backend_coords=rcfg.get('backend_coords','opencv'), backend=backend, external_path=external_path)

    # Circle params
    cc = cfg.get('circle', {})
    cx, cy = float(cc.get('center_xy', [0.0, 0.0])[0]), float(cc.get('center_xy', [0.0, 0.0])[1])
    radius = float(cc.get('radius', 1.5))
    z = float(cc.get('z', 0.8))
    omega = float(cc.get('omega', 0.35))  # rad/s
    duration = float(args.seconds if args.seconds is not None else cc.get('duration', 20.0))
    dt = float(cfg.get('step_dt', 1.0/240.0))
    steps = int(duration/dt)

    # outputs
    out_dir = os.path.join(cfg['output']['dir'], 'circle')
    os.makedirs(out_dir, exist_ok=True)

    t = 0.0
    for i in range(steps):
        if sim is not None:
            sim.step()
        # circular path
        x = cx + radius * math.cos(omega * t)
        y = cy + radius * math.sin(omega * t)
        yaw = (math.atan2(math.sin(omega * t), math.cos(omega * t)) + math.pi/2.0)
        if sim is not None:
            sim.write_state(x, y, z, yaw)
            R_wc, tL, tR = sim.update_cameras(x, y, z, yaw)
        else:
            # 无 Genesis: 手动计算相机位姿
            pos_w = np.array([x, y, z], dtype=np.float32)
            R_wb = quat_wxyz_to_R(np.array(yaw_to_quat_wxyz(yaw), dtype=np.float32))
            R_mount = rig.MOUNT_R
            tL = rig.MOUNT_t_left
            tR = rig.right_offset()
            R_wc = R_wb @ R_mount
            tL = R_wb @ tL + pos_w
            tR = R_wb @ tR + pos_w

        # render
        imgL = renderer.render(rig.K, R_wc, tL, rig.res)
        imgR = renderer.render(rig.K, R_wc, tR, rig.res)

        # save sample frames (可选：每 10 帧保存一对)
        if i % 10 == 0:
            saved = False
            try:
                import imageio
                imageio.imwrite(os.path.join(out_dir, f"L_{i:05d}.png"), imgL)
                imageio.imwrite(os.path.join(out_dir, f"R_{i:05d}.png"), imgR)
                saved = True
            except Exception:
                try:
                    from PIL import Image
                    Image.fromarray(imgL).save(os.path.join(out_dir, f"L_{i:05d}.png"))
                    Image.fromarray(imgR).save(os.path.join(out_dir, f"R_{i:05d}.png"))
                    saved = True
                except Exception:
                    pass
            if not saved:
                if i == 0:
                    print('[Warn] 无法保存PNG（缺少 imageio 或 pillow），将仅执行渲染不存盘。')

        t += dt

    print(f"[OK] circle render finished. frames_saved~{steps//10} at {out_dir}")


if __name__ == '__main__':
    main()


