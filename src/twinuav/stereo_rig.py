from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from utils import fov_v_to_K
from calib_io import load_camera_yaml

@dataclass
class StereoRig:
    baseline: float = 0.050   # D435i depth stereo 默认 50mm，可被标定覆盖
    res: tuple[int, int] = (848, 480)
    fov_v_deg: float = 58.0   # 仅在无标定时回退使用
    MOUNT_R: np.ndarray = np.eye(3, dtype=np.float32)
    MOUNT_t_left: np.ndarray = np.array([0.10, 0.0, 0.05], dtype=np.float32)
    cam0_yaml: str | None = None
    cam1_yaml: str | None = None

    def right_offset(self) -> np.ndarray:
        return self.MOUNT_t_left + np.array([0.0, -self.baseline, 0.0], dtype=np.float32)

    @property
    def K(self) -> np.ndarray:
        if self.cam0_yaml:
            try:
                K0, _, _, size0 = load_camera_yaml(self.cam0_yaml)
                return K0
            except Exception:
                pass
        return fov_v_to_K(self.res[0], self.res[1], self.fov_v_deg)

    def load_from_yaml(self):
        if self.cam0_yaml and self.cam1_yaml:
            try:
                K0, D0, P0, size0 = load_camera_yaml(self.cam0_yaml)
                K1, D1, P1, size1 = load_camera_yaml(self.cam1_yaml)
                self.res = (int(size0[0]), int(size0[1]))
                fx = P0[0,0]
                self.baseline = float(abs(-P1[0,3] / fx)) if P1[0,3] != 0 else self.baseline
                return True
            except Exception:
                return False
        return False