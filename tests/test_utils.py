import numpy as np
from twinuav.utils import quat_wxyz_to_R, fov_v_to_K

def test_quat_to_R_identity():
    R = quat_wxyz_to_R((1,0,0,0))
    assert np.allclose(R, np.eye(3), atol=1e-6)

def test_fov_to_K_square():
    K = fov_v_to_K(640, 480, 60.0)
    assert K.shape == (3,3)
    assert K[0,0] == K[1,1]