from __future__ import annotations
import os
import sys
import numpy as np
from typing import Tuple, Optional, Any


class Stereo3DGSRenderer:
    def __init__(self, gs_model_path: str, backend_coords: str = 'opencv', backend: str | None = None, external_path: Optional[str] = None):
        """
        backend 可选:
          - None/auto: 首选 gsplat 内置加载；失败回退 external；再回退 simple-PLY（CPU）；最后黑图
          - 'gsplat': 使用 gsplat.GaussianModel.load
          - 'external': 尝试导入 twinmanip_me 风格的 gs_viewer_utils/fast_gaussian_model_manager
        external_path: 若不为空，将加入 sys.path 以便导入 external 渲染器依赖
        """
        self.model_path = gs_model_path
        self.ready = False
        self.device = None
        self.model: Any = None
        self.backend: Optional[str] = None
        self.backend_coords = backend_coords
        self._last_error: Optional[BaseException] = None
        self._ext_handles: dict[str, Any] = {}
        self._ply_cache: dict[str, Any] = {}

        if external_path and os.path.isdir(external_path):
            if external_path not in sys.path:
                sys.path.insert(0, external_path)

        prefer = backend or 'auto'
        # try gsplat first
        if prefer in ('auto', 'gsplat'):
            try:
                if os.path.exists(self.model_path):
                    import torch  # type: ignore
                    try:
                        from gsplat import GaussianModel  # type: ignore
                        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        self.model = GaussianModel.load(self.model_path)
                        if hasattr(self.model, 'to'):
                            self.model = self.model.to(self.device)
                        self.backend = 'gsplat'
                        self.ready = True
                    except ModuleNotFoundError as e:
                        self._last_error = e
                        self.ready = False
            except Exception as e:
                self._last_error = e
                self.ready = False
        # try external viewer utils
        if not self.ready and prefer in ('auto', 'external'):
            try:
                from gs_viewer_utils import GSPlatRenderer, Cameras  # type: ignore
                from fast_gaussian_model_manager import construct_from_ply  # type: ignore
                import torch  # type: ignore
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                gaussian = construct_from_ply(self.model_path, device=torch.device(self.device))
                renderer = GSPlatRenderer()
                background = torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
                self._ext_handles = {
                    'renderer': renderer,
                    'gaussian': gaussian,
                    'Cameras': Cameras,
                    'torch': torch,
                    'background': background,
                }
                self.backend = 'external'
                self.ready = True
            except Exception as e:
                self._last_error = e
                self.ready = False
        # try simple ply fallback
        if not self.ready and os.path.exists(self.model_path):
            try:
                from plyfile import PlyData  # type: ignore
                ply = PlyData.read(self.model_path)
                v = ply['vertex'].data
                xyz = np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], axis=1).astype(np.float32)
                # 颜色优先使用 f_dc_*（DC of SH），否则使用 rgb
                if all(k in v.dtype.names for k in ('f_dc_0','f_dc_1','f_dc_2')):
                    col = np.stack([np.asarray(v['f_dc_0']), np.asarray(v['f_dc_1']), np.asarray(v['f_dc_2'])], axis=1).astype(np.float32)
                    col = np.clip(col + 0.5, 0.0, 1.0)
                elif all(k in v.dtype.names for k in ('red','green','blue')):
                    col = np.stack([np.asarray(v['red']), np.asarray(v['green']), np.asarray(v['blue'])], axis=1).astype(np.float32)/255.0
                else:
                    col = np.full((xyz.shape[0], 3), 0.8, dtype=np.float32)
                if all(k in v.dtype.names for k in ('scale_0','scale_1','scale_2')):
                    scales = np.stack([np.asarray(v['scale_0']), np.asarray(v['scale_1']), np.asarray(v['scale_2'])], axis=1).astype(np.float32)
                    rad = np.max(scales, axis=1)
                else:
                    rad = np.full((xyz.shape[0],), 0.03, dtype=np.float32)
                self._ply_cache = {
                    'xyz': xyz,
                    'rgb': (col*255.0).astype(np.uint8),
                    'rad': rad,
                }
                self.backend = 'simple'
                self.ready = True
            except Exception as e:
                self._last_error = e
                self.ready = False

    @staticmethod
    def _invert_cam_world(R_cw: np.ndarray, t_cw: np.ndarray) -> np.ndarray:
        Twc = np.eye(4, dtype=np.float32)
        Rwc = R_cw.T
        twc = -Rwc @ t_cw
        Twc[:3, :3] = Rwc
        Twc[:3, 3] = twc
        return Twc

    def _maybe_to_opengl(self, Twc: np.ndarray) -> np.ndarray:
        if self.backend_coords.lower() == 'opengl':
            S = np.eye(4, dtype=np.float32)
            S[1, 1] = -1.0
            S[2, 2] = -1.0
            return S @ Twc
        return Twc

    def _render_external(self, K: np.ndarray, T_wc_backend: np.ndarray, res: Tuple[int, int], need_depth: bool = False):
        try:
            Cameras = self._ext_handles['Cameras']
            torch = self._ext_handles['torch']
            renderer = self._ext_handles['renderer']
            gaussian = self._ext_handles['gaussian']
            bg = self._ext_handles['background']
            W, H = int(res[0]), int(res[1])
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            # 世界到相机
            Rwc = T_wc_backend[:3, :3]
            Twc = T_wc_backend[:3, 3]
            R = torch.as_tensor(Rwc, dtype=torch.float32).unsqueeze(0)
            T = torch.as_tensor(Twc, dtype=torch.float32).unsqueeze(0)
            cam = Cameras(
                R=R,
                T=T,
                fx=torch.tensor([fx], dtype=torch.float32),
                fy=torch.tensor([fy], dtype=torch.float32),
                cx=torch.tensor([cx], dtype=torch.float32),
                cy=torch.tensor([cy], dtype=torch.float32),
                width=torch.tensor([W], dtype=torch.int32),
                height=torch.tensor([H], dtype=torch.int32),
                appearance_id=torch.tensor([0], dtype=torch.int32),
                normalized_appearance_id=torch.tensor([0.0], dtype=torch.float32),
                distortion_params=None,
                camera_type=torch.tensor([0], dtype=torch.int32),
            )[0].to_device(torch.device(self.device))
            with torch.no_grad():
                outputs = renderer.render(cam, gaussian, bg, scaling_modifier=1.0, render_types=["rgb", "depth"] if need_depth else ["rgb"])  # type: ignore
                rgb = outputs["render"].permute(1, 2, 0).detach().cpu().numpy()
                rgb = np.clip(rgb, 0.0, 1.0)
                rgb = (rgb * 255.0).astype(np.uint8)
                depth = None
                if need_depth and outputs.get('depth', None) is not None:
                    depth_t = outputs['depth'][0]
                    depth = depth_t.detach().cpu().numpy()
                return rgb, depth
        except Exception as e:
            self._last_error = e
            H = int(res[1]); W = int(res[0])
            return np.zeros((H, W, 3), dtype=np.uint8), None

    def _render_simple(self, K: np.ndarray, T_wc_backend: np.ndarray, res: Tuple[int, int]) -> np.ndarray:
        """快速点渲染（1像素 splat + 深度）— 向量化实现，比逐像素圆盘快数十倍。
        可在高密度点云上得到稀疏但可见的结果，用于快速验证。
        """
        H = int(res[1]); W = int(res[0])
        img = np.zeros((H, W, 3), dtype=np.uint8)
        cache = self._ply_cache
        if not cache:
            return img
        pts = cache['xyz']  # world
        rgb = cache['rgb']
        # transform to camera
        R = T_wc_backend[:3, :3]
        t = T_wc_backend[:3, 3]
        Pc = (pts @ R.T) + t[None, :]
        Z = Pc[:, 2]
        valid = Z > 1e-4
        if not np.any(valid):
            return img
        Pc = Pc[valid]
        Z = Z[valid]
        rgb = rgb[valid]
        # 可选：点数过大时子采样
        N = Pc.shape[0]
        Nmax = int(8e5)
        if N > Nmax:
            idx = np.random.default_rng(123).choice(N, Nmax, replace=False)
            Pc = Pc[idx]; Z = Z[idx]; rgb = rgb[idx]
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        u = (fx * (Pc[:, 0] / Z) + cx).astype(np.int32)
        v = (fy * (Pc[:, 1] / Z) + cy).astype(np.int32)
        # 裁剪到图像范围
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(inb):
            return img
        u = u[inb]; v = v[inb]; Z = Z[inb]; rgb = rgb[inb]
        lin = (v * W + u).astype(np.int64)
        # 计算每个像素的最小深度（scatter-min）
        min_z = np.full(H * W, np.inf, dtype=np.float32)
        np.minimum.at(min_z, lin, Z)
        # 选择具有最小深度的点
        selected = Z <= (min_z[lin] + 1e-6)
        if not np.any(selected):
            return img
        lin_sel = lin[selected]
        rgb_sel = rgb[selected]
        # 避免同一像素多次赋值：取首个
        uniq_lin, first_idx = np.unique(lin_sel, return_index=True)
        img_flat = img.reshape(-1, 3)
        img_flat[uniq_lin] = rgb_sel[first_idx]
        # 轻微膨胀以减少稀疏感（3x3 max）
        try:
            import cv2 as cv
            img = cv.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
        except Exception:
            pass
        return img

    def render(self, K: np.ndarray, R_cam_world: np.ndarray, t_cam_world: np.ndarray, res: Tuple[int, int], return_depth: bool = False):
        W, H = int(res[0]), int(res[1])
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        T_wc = self._invert_cam_world(R_cam_world.astype(np.float32), t_cam_world.astype(np.float32))
        T_wc_backend = self._maybe_to_opengl(T_wc)
        # external backend
        if self.backend == 'external' and self._ext_handles:
            rgb, depth = self._render_external(K, T_wc_backend, (W, H), need_depth=return_depth)
            return (rgb, depth) if return_depth else rgb
        # gsplat model
        if self.backend == 'gsplat' and self.model is not None and hasattr(self.model, 'render'):
            try:
                import torch  # type: ignore
                T_tensor = torch.as_tensor(T_wc_backend, dtype=torch.float32)
                if 'width' in self.model.render.__code__.co_varnames:
                    rgb = self.model.render(width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy, pose=T_tensor)
                else:
                    rgb = self.model.render(W, H, fx, fy, cx, cy, T_tensor)
                if hasattr(rgb, 'detach'):
                    rgb = rgb.detach().cpu().numpy()
                rgb = np.asarray(rgb)
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
                if rgb.shape[-1] == 4:
                    rgb = rgb[..., :3]
                if rgb.shape[0] == W and rgb.shape[1] == H:
                    rgb = np.transpose(rgb, (1, 0, 2))
                if return_depth:
                    return rgb, None
                return rgb
            except Exception as e:
                self._last_error = e
                if return_depth:
                    return np.zeros((H, W, 3), dtype=np.uint8), None
                return np.zeros((H, W, 3), dtype=np.uint8)
        # simple ply fallback
        if self.backend == 'simple' and self._ply_cache:
            rgb = self._render_simple(K, T_wc_backend, (W, H))
            return (rgb, None) if return_depth else rgb
        # 用户自定义回调
        render_fn = getattr(self, 'render_fn', None)
        if callable(render_fn):
            try:
                rgb = render_fn(K, T_wc_backend, (W, H))
                rgb = np.asarray(rgb)
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
                if rgb.shape[-1] == 4:
                    rgb = rgb[..., :3]
                if return_depth:
                    return rgb, None
                return rgb
            except Exception as e:
                self._last_error = e
                if return_depth:
                    return np.zeros((H, W, 3), dtype=np.uint8), None
                return np.zeros((H, W, 3), dtype=np.uint8)
        # fallback
        if return_depth:
            return np.zeros((H, W, 3), dtype=np.uint8), None
        return np.zeros((H, W, 3), dtype=np.uint8)