from __future__ import annotations
from typing import Any
import numpy as np
import torch
from plyfile import PlyData  # type: ignore


def _load_array_from_plyelement(plyelement: Any, name_prefix: str) -> np.ndarray:
    names = [p.name for p in plyelement.properties if p.name.startswith(name_prefix)]
    if len(names) == 0:
        return np.empty((plyelement["x"].shape[0], 0))
    names = sorted(names, key=lambda x: int(x.split('_')[-1]))
    v_list = [np.asarray(plyelement[n]) for n in names]
    return np.stack(v_list, axis=1)


class GaussianModel(torch.nn.Module):
    def __init__(self, xyz: np.ndarray, scaling: np.ndarray, rotation: np.ndarray, opacity: np.ndarray, features: np.ndarray, sh_degrees: int, device: torch.device = torch.device("cuda:0"), dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self._xyz = torch.tensor(xyz).to(dtype).to(device)
        self._scaling = torch.exp(torch.tensor(scaling).to(dtype).to(device))
        self._rotation = torch.nn.functional.normalize(torch.tensor(rotation).to(dtype).to(device))
        self._opacity = torch.sigmoid(torch.tensor(opacity).to(dtype).to(device))
        self._features = torch.tensor(features).to(dtype).to(device)
        self.sh_degrees = int(sh_degrees)
        self.active_sh_degree = int(sh_degrees)

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features

    @property
    def get_opacity(self):
        return self._opacity


def construct_from_ply(ply_path: str, device: torch.device = torch.device("cuda:0")) -> GaussianModel:
    plydata = PlyData.read(ply_path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[:, None]
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    features_rest = _load_array_from_plyelement(plydata.elements[0], "f_rest_").reshape((xyz.shape[0], 3, -1))
    features_rest_dims = features_rest.shape[-1]
    sh_degrees = 0
    for i in range(4):
        if features_rest_dims == (i + 1) ** 2 - 1:
            sh_degrees = i
            break
    features = np.transpose(np.concatenate([features_dc, features_rest], axis=2), (0, 2, 1))
    scales = _load_array_from_plyelement(plydata.elements[0], "scale_")
    rots = _load_array_from_plyelement(plydata.elements[0], "rot_")
    return GaussianModel(xyz=xyz, opacity=opacities, features=features, scaling=scales, rotation=rots, device=device, sh_degrees=sh_degrees)


