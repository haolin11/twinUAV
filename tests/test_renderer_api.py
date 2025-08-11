import numpy as np
from twinuav.renderer_3dgs import Stereo3DGSRenderer

def test_renderer_black_when_no_backend():
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    K = np.eye(3, dtype=np.float32)
    ren = Stereo3DGSRenderer('non_existing.ply')
    img = ren.render(K, R, t, (64, 32))
    assert img.shape == (32, 64, 3)
    assert img.dtype == np.uint8