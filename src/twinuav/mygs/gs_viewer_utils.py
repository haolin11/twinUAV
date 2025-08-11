"""
简化版外部渲染接口，复制自 /home/dell/twinmanip/simulation 中的接口签名，
但去掉 viser/warp 等重依赖，仅保留渲染需要的最小功能：
- Cameras 数据结构（期望 world-to-camera，Y down, Z forward）
- GSPlatRenderer.render，调用 gsplat 的 project_gaussians + rasterize_gaussians

确保与现有 `twinuav.renderer_3dgs.Stereo3DGSRenderer` 的 external 路径兼容。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any, TYPE_CHECKING

import torch
import numpy as np

from gsplat import project_gaussians  # type: ignore
from gsplat.rasterize import rasterize_gaussians  # type: ignore
from gsplat.sh import spherical_harmonics  # type: ignore

if TYPE_CHECKING:
    from .gaussian_model import GaussianModel


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def _fov2focal(fov: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    # 兼容占位，当前不在管线中使用
    return pixels / (2 * torch.tan(fov / 2))


class CameraType:
    PERSPECTIVE: int = 0
    FISHEYE: int = 1


@dataclass
class Camera:
    R: torch.Tensor  # [3,3]
    T: torch.Tensor  # [3]
    fx: torch.Tensor
    fy: torch.Tensor
    fov_x: torch.Tensor
    fov_y: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor
    width: torch.Tensor
    height: torch.Tensor
    appearance_id: torch.Tensor
    normalized_appearance_id: torch.Tensor
    time: torch.Tensor
    distortion_params: Optional[torch.Tensor]
    camera_type: torch.Tensor

    world_to_camera: torch.Tensor
    projection: torch.Tensor
    full_projection: torch.Tensor
    camera_center: torch.Tensor

    def to_device(self, device: torch.device):
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self


@dataclass
class Cameras:
    """Y down, Z forward; 输入为 batches 的相机参数，内部构造投影矩阵等缓存。"""
    R: torch.Tensor
    T: torch.Tensor
    fx: torch.Tensor
    fy: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor
    width: torch.Tensor
    height: torch.Tensor
    appearance_id: torch.Tensor
    normalized_appearance_id: Optional[torch.Tensor]
    distortion_params: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    camera_type: torch.Tensor

    fov_x: torch.Tensor = field(init=False)
    fov_y: torch.Tensor = field(init=False)
    world_to_camera: torch.Tensor = field(init=False)
    projection: torch.Tensor = field(init=False)
    full_projection: torch.Tensor = field(init=False)
    camera_center: torch.Tensor = field(init=False)
    time: Optional[torch.Tensor] = None

    def __post_init__(self):
        self._calculate_fov()
        self._calculate_w2c()
        self._calculate_ndc_projection_matrix()
        self._calculate_camera_center()
        if self.time is None:
            self.time = torch.zeros(self.R.shape[0])
        if self.distortion_params is None:
            self.distortion_params = torch.zeros(self.R.shape[0], 4)

    def _calculate_fov(self):
        self.fov_x = 2 * torch.atan((self.width / 2) / self.fx)
        self.fov_y = 2 * torch.atan((self.height / 2) / self.fy)

    def _calculate_w2c(self):
        self.world_to_camera = torch.zeros((self.R.shape[0], 4, 4), dtype=self.R.dtype, device=self.R.device)
        self.world_to_camera[:, :3, :3] = self.R
        self.world_to_camera[:, :3, 3] = self.T
        self.world_to_camera[:, 3, 3] = 1.0
        self.world_to_camera = torch.transpose(self.world_to_camera, 1, 2)

    def _calculate_ndc_projection_matrix(self):
        zfar = 100.0
        znear = 0.01
        tanHalfFovY = torch.tan((self.fov_y / 2))
        tanHalfFovX = torch.tan((self.fov_x / 2))
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right
        P = torch.zeros(self.fov_y.shape[0], 4, 4, dtype=self.R.dtype, device=self.R.device)
        z_sign = 1.0
        P[:, 0, 0] = 2.0 * znear / (right - left)
        P[:, 1, 1] = 2.0 * znear / (top - bottom)
        P[:, 0, 2] = (right + left) / (right - left)
        P[:, 1, 2] = (top + bottom) / (top - bottom)
        P[:, 3, 2] = z_sign
        P[:, 2, 2] = z_sign * zfar / (zfar - znear)
        P[:, 2, 3] = -(zfar * znear) / (zfar - znear)
        self.projection = torch.transpose(P, 1, 2)
        self.full_projection = self.world_to_camera.bmm(self.projection)

    def _calculate_camera_center(self):
        self.camera_center = torch.linalg.inv(self.world_to_camera)[:, 3, :3]

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, index: int) -> Camera:
        return Camera(
            R=self.R[index],
            T=self.T[index],
            fx=self.fx[index],
            fy=self.fy[index],
            fov_x=self.fov_x[index],
            fov_y=self.fov_y[index],
            cx=self.cx[index],
            cy=self.cy[index],
            width=self.width[index],
            height=self.height[index],
            appearance_id=self.appearance_id[index],
            normalized_appearance_id=(self.normalized_appearance_id[index] if isinstance(self.normalized_appearance_id, torch.Tensor) else torch.tensor(0.0, dtype=self.fx.dtype)),
            distortion_params=(self.distortion_params[index] if isinstance(self.distortion_params, torch.Tensor) else None),
            time=(self.time[index] if isinstance(self.time, torch.Tensor) else torch.tensor(0.0, dtype=self.fx.dtype)),
            camera_type=self.camera_type[index],
            world_to_camera=self.world_to_camera[index],
            projection=self.projection[index],
            full_projection=self.full_projection[index],
            camera_center=self.camera_center[index],
        )


DEFAULT_BLOCK_SIZE: int = 16
DEFAULT_ANTI_ALIASED_STATUS: bool = False


class GSPlatRenderer:
    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE, anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS) -> None:
        self.block_size = int(block_size)
        self.anti_aliased = bool(anti_aliased)

    def render(self, viewpoint_camera: Camera, pc: "GaussianModel", bg_color: torch.Tensor, scaling_modifier: float = 1.0, render_types: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        if render_types is None:
            render_types = ["rgb"]
        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(
            means3d=pc.get_xyz,
            scales=pc.get_scaling,
            glob_scale=float(scaling_modifier),
            quats=pc.get_rotation / pc.get_rotation.norm(dim=-1, keepdim=True),
            viewmat=viewpoint_camera.world_to_camera.T[:3, :],
            fx=float(viewpoint_camera.fx.item()),
            fy=float(viewpoint_camera.fy.item()),
            cx=float(viewpoint_camera.cx.item()),
            cy=float(viewpoint_camera.cy.item()),
            img_height=img_height,
            img_width=img_width,
            block_width=self.block_size,
        )
        viewdirs = pc.get_xyz.detach() - viewpoint_camera.camera_center
        rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)
        opacities = pc.get_opacity
        if self.anti_aliased:
            opacities = opacities * comp[:, None]
        rgba = None
        rgb3 = None
        if "rgb" in render_types:
            rgb, alpha = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                rgbs,
                opacities,
                img_height=img_height,
                img_width=img_width,
                block_width=self.block_size,
                background=bg_color,
                return_alpha=True,
            )
            rgb = rgb.permute(2, 0, 1)
            rgba = torch.cat([rgb, alpha.unsqueeze(0)], dim=0)
            rgb3 = rgb
        depth_im = None
        if "depth" in render_types:
            depth_im, depth_alpha = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                depths.unsqueeze(-1).repeat(1, 3),
                opacities,
                img_height=img_height,
                img_width=img_width,
                block_width=self.block_size,
                background=torch.zeros_like(bg_color),
                return_alpha=True,
            )
            depth_alpha = depth_alpha[..., None]
            depth_im = torch.where(depth_alpha > 0, depth_im / depth_alpha, torch.tensor(10.0, dtype=depth_im.dtype, device=depth_im.device))
            depth_im = depth_im.permute(2, 0, 1)
        out: Dict[str, Any] = {
            "render": rgba if rgba is not None else torch.zeros((3, img_height, img_width), dtype=bg_color.dtype, device=bg_color.device),
            "depth": depth_im,
            "alpha": None,
            "viewspace_points": xys,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
        if rgb3 is not None:
            out["rgb"] = rgb3
        return out


