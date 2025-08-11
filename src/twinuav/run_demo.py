from __future__ import annotations
import os, time
import argparse
import numpy as np
import yaml

from .stereo_rig import StereoRig
from .renderer_3dgs import Stereo3DGSRenderer
from .planner3d import HybridAStar3D
from .voxel_map3d import voxelize_3dgs_ply
from .control import KinState, KinematicUAV, BodyRateController
from .imu_sim import IMUSim
from .rules import RuleEngine
from .genesis_sim import GenesisSim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/demo_default.yaml')
    ap.add_argument('--viewer', action='store_true', default=False)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))

    # Genesis + UAV
    urdf = cfg['urdf']; base_link = cfg.get('base_link','base_link')
    viewer = bool(cfg.get('viewer', args.viewer))
    try:
        sim = GenesisSim(urdf=urdf, base_link=base_link, viewer=viewer)
    except Exception as e:
        if viewer:
            print(f"[Warn] 可视化窗口初始化失败，自动切换为无窗口模式（headless）：{e}")
            sim = GenesisSim(urdf=urdf, base_link=base_link, viewer=False)
        else:
            raise

    # Objects
    obj_cfg = cfg.get('objects',{}).get('table', None)
    if obj_cfg:
        sim.set_obstacle_box(tuple(obj_cfg['pos']), tuple(obj_cfg['size']))

    # Stereo rig with D435i calibration
    s = cfg['stereo']
    rig = StereoRig(baseline=s['baseline'], res=tuple(s['res']), fov_v_deg=s['fov_v_deg'])
    rig.cam0_yaml = s.get('cam0_yaml', None)
    rig.cam1_yaml = s.get('cam1_yaml', None)
    _ = rig.load_from_yaml()
    sim.attach_stereo(rig)

    # 3DGS renderer
    rcfg = cfg['renderer']
    renderer = Stereo3DGSRenderer(rcfg['model'], backend_coords=rcfg.get('backend_coords','opencv'))

    # 3D voxel map + 3D Hybrid A*
    vcfg = cfg['voxel']
    bounds = tuple(vcfg['bounds'])
    res = float(vcfg['resolution'])
    inflate = float(vcfg['inflate_radius'])
    grid3d = voxelize_3dgs_ply(vcfg['ply'], bounds, res, inflate_radius=inflate)

    pcfg = cfg['plan3d']
    start = tuple(pcfg['start'])  # x,y,z,yaw
    goal = tuple(pcfg['goal'])    # x,y,z
    planner = HybridAStar3D(grid3d, step_xyz=pcfg['step_xyz'], step_yaw=pcfg['step_yaw'], n_heading=pcfg['n_heading'])
    path = planner.plan(start=start, goal_xyz=goal)

    # Control + IMU
    ccfg = cfg['control']
    kin = KinematicUAV(KinState(x=start[0], y=start[1], z=start[2], yaw=start[3]), z_ref=start[2])
    ctrl = BodyRateController(v_max=ccfg['v_max'], yaw_rate_max=ccfg['yaw_rate_max'])
    imu_sim = IMUSim()

    # Rules + capture
    cap = cfg['capture']
    rules = RuleEngine({'table': tuple(obj_cfg['pos'])}) if obj_cfg else RuleEngine({})
    near_key = f"near_{cap['near_object']}"

    # Loop
    steps = int(cfg.get('steps', 2400)); dt = float(cfg.get('step_dt', 1.0/240.0))
    dataset = []
    imu_buf_g = []
    imu_buf_a = []
    path_idx = 0
    os.makedirs(cfg['output']['dir'], exist_ok=True)

    for i in range(steps):
        sim.step()
        # path following
        if path and path_idx < len(path):
            tx, ty, tz, tth = path[path_idx]
            if ( (kin.s.x - tx)**2 + (kin.s.y - ty)**2 + (kin.s.z - tz)**2 ) ** 0.5 < max(0.2, pcfg['step_xyz']*1.5):
                path_idx += 1
            target_xy = (tx, ty)
            kin.z_ref = tz
        else:
            target_xy = (goal[0], goal[1])
            kin.z_ref = goal[2]
        v, r = ctrl.track(kin.s, target_xy)
        kin.step(v, r, dt)
        sim.write_state(kin.s.x, kin.s.y, kin.s.z, kin.s.yaw)
        R_wc, tL, tR = sim.update_cameras(kin.s.x, kin.s.y, kin.s.z, kin.s.yaw)

        # IMU
        gyro, accel = imu_sim.step(v, r, kin.s.yaw)
        imu_buf_g.append(gyro)
        imu_buf_a.append(accel)

        # rule-triggered capture
        cond = False
        if obj_cfg:
            cond = rules.near(cap['near_object'], (kin.s.x, kin.s.y, kin.s.z), radius=cap['radius'])
        if rules.once(near_key, cond):
            for dy in np.linspace(-0.1, 0.1, int(cap['burst'])):
                Ry = np.array([[np.cos(dy), -np.sin(dy), 0.0], [np.sin(dy), np.cos(dy), 0.0], [0.0,0.0,1.0]], dtype=np.float32)
                R_wc_s = R_wc @ Ry
                imgL = renderer.render(rig.K, R_wc_s, tL, rig.res)
                imgR = renderer.render(rig.K, R_wc_s, tR, rig.res)
                dataset.append({
                    'rgb_left': imgL, 'rgb_right': imgR, 'K': rig.K,
                    'R_wc': R_wc_s, 't_wc_L': tL, 't_wc_R': tR,
                    'state': {'x': kin.s.x, 'y': kin.s.y, 'z': kin.s.z, 'yaw': kin.s.yaw},
                    'action': {'v': float(v), 'yaw_rate': float(r)},
                    'timestamp': float(i*dt)
                })

    # save
    if bool(cfg['output'].get('save_npz', True)) and len(dataset) > 0:
        out = os.path.join(cfg['output']['dir'], 'dataset.npz')
        np.savez_compressed(
            out,
            rgb_left=np.stack([it['rgb_left'] for it in dataset], axis=0),
            rgb_right=np.stack([it['rgb_right'] for it in dataset], axis=0),
            K=dataset[0]['K'],
            R_wc=np.stack([it['R_wc'] for it in dataset], axis=0),
            t_wc_L=np.stack([it['t_wc_L'] for it in dataset], axis=0),
            t_wc_R=np.stack([it['t_wc_R'] for it in dataset], axis=0),
            gyro=np.asarray(imu_buf_g, dtype=np.float32),
            accel=np.asarray(imu_buf_a, dtype=np.float32),
        )
        print(f"[OK] Saved dataset to {out}, samples={len(dataset)}")
    else:
        print(f"[OK] Finished. Captured {len(dataset)} samples.")

if __name__ == '__main__':
    main()