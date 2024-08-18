import numpy as np
from scipy.spatial.transform import Rotation as R


def get_intrinsic_mat(intrinsic_params):
    fx = intrinsic_params[0]
    fy = intrinsic_params[1]
    cx = intrinsic_params[2]
    cy = intrinsic_params[3]
    intrinsic_mat = np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]],
        dtype=np.float32
    )   # [3,3]
    return intrinsic_mat


def get_extrinsic_mat(quat, trans):
    translation = np.array(trans)[:, np.newaxis]  # [3,1]
    rot_mat = R.from_quat(quat).as_matrix()  # [3,3]
    transform_mat = np.concatenate(
        [rot_mat, translation], axis=1, dtype=np.float32)  # [3,4]
    return transform_mat


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [3, 3]
    """
    x, y, z = angle[0], angle[1], angle[2]

    cosz = np.cos(z)
    sinz = np.sin(z)
    zmat = np.array([cosz, -sinz, 0,
                     sinz, cosz, 0,
                     0, 0, 1]).reshape(3, 3)

    cosy = np.cos(y)
    siny = np.sin(y)
    ymat = np.array([cosy, 0, siny,
                     0, 1, 0,
                     -siny, 0, cosy]).reshape(3, 3)

    cosx = np.cos(x)
    sinx = np.sin(x)
    xmat = np.array([1, 0, 0,
                     0, cosx, -sinx,
                     0, sinx, cosx]).reshape(3, 3)

    rot_mat = xmat @ ymat @ zmat
    return rot_mat
