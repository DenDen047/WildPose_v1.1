import numpy as np


def make_intrinsic_mat(fx, fy, cx, cy):
    intrinsic_mat = np.eye(4)
    intrinsic_mat[0, 0] = fx
    intrinsic_mat[0, 2] = cx
    intrinsic_mat[1, 1] = fy
    intrinsic_mat[1, 2] = cy

    return intrinsic_mat


def make_extrinsic_mat(rot_mat, translation):
    extrinsic_mat = np.eye(4)
    extrinsic_mat[:3, :3] = rot_mat
    extrinsic_mat[:-1, -1] = translation
    extrinsic_mat[-1, -1] = 1

    return extrinsic_mat


def lidar2cam_projection(pcd, extrinsic):
    tmp = np.insert(pcd, 3, 1, axis=1).T
    # removing points that are behind the camera
    tmp = np.delete(tmp, np.where(tmp[0, :] < 0), axis=1)

    return extrinsic @ tmp


def cam2image_projection(pcd_in_cam, intrinsic):
    pcd_in_image = intrinsic @ pcd_in_cam
    pcd_in_image[:2] /= pcd_in_image[2, :]

    return pcd_in_image
