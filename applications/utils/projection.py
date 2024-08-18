import numpy as np


def lidar2cam_projection(pcd, extrinsic):
    tmp = np.insert(pcd, 3, 1, axis=1).T
    tmp = np.delete(tmp, np.where(tmp[0, :] < 0), axis=1)
    pcd_in_cam = extrinsic.dot(tmp)

    return pcd_in_cam


def cam2image_projection(pcd_in_cam, intrinsic):
    pcd_in_image = intrinsic.dot(pcd_in_cam)
    pcd_in_image[:2] /= pcd_in_image[2, :]

    return pcd_in_image


