from __future__ import annotations
import argparse
import os
from typing import Tuple

import numpy as np
import open3d as o3d  # type: ignore
from plyfile import PlyData, PlyElement  # type: ignore


def _axis_angle_to_quaternion(axis_angle: np.ndarray, order: str = "wxyz") -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-12:
        if order == "wxyz":
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis_angle / angle
    s = np.sin(angle / 2.0)
    c = np.cos(angle / 2.0)
    if order == "wxyz":
        return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)
    elif order == "xyzw":
        return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float64)
    raise ValueError(f"Unsupported quaternion order: {order}")


def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray, order: str = "wxyz") -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    if order == "wxyz":
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z], dtype=np.float64)
    elif order == "xyzw":
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([x, y, z, w], dtype=np.float64)
    raise ValueError(f"Unsupported quaternion order: {order}")


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return q
    return q / n


def _compute_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    # 使用 Open3D 的 Rodrigues 轴角到旋转矩阵
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle.astype(np.float64))


def _rotate_points(points: np.ndarray, R: np.ndarray, center: np.ndarray) -> np.ndarray:
    shifted = points - center[None, :]
    rotated = shifted @ R.T
    return rotated + center[None, :]


def _read_ply_vertices(ply_path: str):
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"].data
    return ply, vertex


def _write_ply_vertices(ply: PlyData, vertex_new: np.ndarray, out_path: str) -> None:
    vert_elem = PlyElement.describe(vertex_new, "vertex")
    PlyData([vert_elem], text=(ply.text if hasattr(ply, "text") else False)).write(out_path)


def interactive_adjust_orientation(input_ply: str, output_ply: str, quat_order: str = "wxyz", move_origin: np.ndarray | Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    ply, v = _read_ply_vertices(input_ply)
    xyz = np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], axis=1).astype(np.float64)
    center = np.mean(xyz, axis=0)
    move_origin = np.asarray(move_origin, dtype=np.float64).reshape(3)

    # 为交互创建点云（仅作为预览，不会写回）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz - center[None, :])
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Adjust 3DGS Orientation (only rotation)")
    vis.add_geometry(pcd)
    vis.add_geometry(axis)

    r_axis_angles = np.zeros((3,), dtype=np.float64)

    def _apply_preview():
        R = _compute_rotation_matrix(r_axis_angles)
        xyz_rot = (xyz - center[None, :]) @ R.T
        pcd.points = o3d.utility.Vector3dVector(xyz_rot)
        vis.update_geometry(pcd)

    def _rot_x_pos(vis_):
        r_axis_angles[0] += 0.5 * np.pi / 180.0
        _apply_preview()

    def _rot_x_neg(vis_):
        r_axis_angles[0] -= 0.5 * np.pi / 180.0
        _apply_preview()

    def _rot_y_pos(vis_):
        r_axis_angles[1] += 0.5 * np.pi / 180.0
        _apply_preview()

    def _rot_y_neg(vis_):
        r_axis_angles[1] -= 0.5 * np.pi / 180.0
        _apply_preview()

    def _rot_z_pos(vis_):
        r_axis_angles[2] += 0.5 * np.pi / 180.0
        _apply_preview()

    def _rot_z_neg(vis_):
        r_axis_angles[2] -= 0.5 * np.pi / 180.0
        _apply_preview()

    def _view_xy(vis_):
        vc = vis.get_view_control()
        vc.set_lookat([0, 0, 0])
        vc.set_up([0, 1, 0])
        vc.set_front([0, 0, -1])

    def _view_xz(vis_):
        vc = vis.get_view_control()
        vc.set_lookat([0, 0, 0])
        vc.set_up([0, 0, 1])
        vc.set_front([0, 1, 0])

    def _view_yz(vis_):
        vc = vis.get_view_control()
        vc.set_lookat([0, 0, 0])
        vc.set_up([0, 1, 0])
        vc.set_front([1, 0, 0])

    # 初始视角
    _view_xy(vis)
    vis.register_key_callback(ord('1'), _view_xy)
    vis.register_key_callback(ord('2'), _rot_z_pos)
    vis.register_key_callback(ord('3'), _rot_z_neg)
    vis.register_key_callback(ord('4'), _view_xz)
    vis.register_key_callback(ord('5'), _rot_y_pos)
    vis.register_key_callback(ord('6'), _rot_y_neg)
    vis.register_key_callback(ord('7'), _view_yz)
    vis.register_key_callback(ord('8'), _rot_x_pos)
    vis.register_key_callback(ord('9'), _rot_x_neg)

    vis.run()
    vis.destroy_window()

    # 计算最终旋转并应用到位置与每个高斯的四元数
    R = _compute_rotation_matrix(r_axis_angles)
    qR = _axis_angle_to_quaternion(r_axis_angles, order=quat_order)
    qR = _normalize_quaternion(qR)

    # 最终位置：先绕质心旋转，再将“原点移动”move_origin（等价于对点云整体平移 -move_origin）
    xyz_out = _rotate_points(xyz, R, center) - move_origin[None, :]

    # 构造新的顶点结构化数组，保持其他属性不变
    v_dtype = v.dtype
    v_new = np.empty(v.shape, dtype=v_dtype)
    for name in v_dtype.names:
        v_new[name] = v[name]
    v_new['x'] = xyz_out[:, 0].astype(v_new['x'].dtype)
    v_new['y'] = xyz_out[:, 1].astype(v_new['y'].dtype)
    v_new['z'] = xyz_out[:, 2].astype(v_new['z'].dtype)

    # 旋转每个高斯的朝向（四元数）
    rot_field_names = [n for n in v_dtype.names if n.startswith('rot_')]
    rot_field_names_sorted = sorted(rot_field_names, key=lambda s: int(s.split('_')[-1]))
    if len(rot_field_names_sorted) == 4:
        # 读取 Nx4 四元数
        rots = np.stack([np.asarray(v[n]) for n in rot_field_names_sorted], axis=1).astype(np.float64)
        rots = rots / np.maximum(np.linalg.norm(rots, axis=1, keepdims=True), 1e-12)
        # 左乘全局旋转：q_new = qR ⊗ q_old
        qR_batched = np.repeat(qR[None, :], rots.shape[0], axis=0)
        q_new = np.empty_like(rots)
        # 向量化四元数乘法
        if quat_order == 'wxyz':
            w1, x1, y1, z1 = qR_batched[:, 0], qR_batched[:, 1], qR_batched[:, 2], qR_batched[:, 3]
            w2, x2, y2, z2 = rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]
            q_new[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            q_new[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            q_new[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            q_new[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        else:  # xyzw
            x1, y1, z1, w1 = qR_batched[:, 0], qR_batched[:, 1], qR_batched[:, 2], qR_batched[:, 3]
            x2, y2, z2, w2 = rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]
            q_new[:, 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # w
            q_new[:, 0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            q_new[:, 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            q_new[:, 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        q_new = q_new / np.maximum(np.linalg.norm(q_new, axis=1, keepdims=True), 1e-12)
        for i, name in enumerate(rot_field_names_sorted):
            v_new[name] = q_new[:, i].astype(v_new[name].dtype)
    else:
        # 未找到旋转字段，跳过（但仍然写位置更新）
        pass

    _write_ply_vertices(ply, v_new, output_ply)

    return R, qR


def main():
    parser = argparse.ArgumentParser(description="Interactively rotate a 3D Gaussian Splatting PLY and save with updated centers and quaternions. No scaling.")
    parser.add_argument('--input', type=str, default='/data/home/chenhaolin/twinUAV/assets/3dgs/scene.ply')
    parser.add_argument('--output', type=str, default=None, help="Output PLY path. Default: input basename + _rot.ply")
    parser.add_argument('--quat_order', type=str, default='wxyz', choices=['wxyz', 'xyzw'], help="Quaternion storage order in PLY rot_* fields")
    parser.add_argument('--translate', type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=('DX','DY','DZ'), help="Move world origin by (DX,DY,DZ) after rotation. Equivalent to translating the model by (-DX,-DY,-DZ).")
    args = parser.parse_args()

    input_ply = args.input
    if args.output is None:
        base, ext = os.path.splitext(input_ply)
        output_ply = base + '_rot' + ext
    else:
        output_ply = args.output

    os.makedirs(os.path.dirname(output_ply), exist_ok=True)

    interactive_adjust_orientation(input_ply, output_ply, quat_order=args.quat_order, move_origin=np.asarray(args.translate, dtype=np.float64))


if __name__ == '__main__':
    main()


