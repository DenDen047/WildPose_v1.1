import os
import glob
import numpy as np
import cv2
import open3d as o3d
from itertools import combinations

from utils.file_loader import load_pcd, load_camera_parameters
from utils import camera as camera_utils
from projection_functions import closest_point


ECAL_FOLDER = 'data/calibration/ecal_meas/2024-06-13_11-22-14.768_measurement'
CAMERA_PARAM_FILENAME = 'manual_calibration.json'
FRAME_START_INDEX = 0
FRAME_END_INDEX = 200
PATTERN_SIZE = (7, 10)  # for example
DEFAULT_CORNERS = [
    [816, 383], [1088, 391],
    [836, 679],
]


def find_checkerboard_corners(image_path, pattern_size):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        return corners
    else:
        print('INFO: the corner could not be detected.')
        return DEFAULT_CORNERS


def main():
    # load the data
    all_rgb_fpaths = sorted(glob.glob(os.path.join(ECAL_FOLDER, 'sync_rgb', '*.jpeg')))
    all_pcd_fpaths = sorted(glob.glob(os.path.join(ECAL_FOLDER, 'lidar', '*.pcd')))
    # assert len(all_rgb_fpaths) == len(all_pcd_fpaths)

    rgb_fpaths = all_rgb_fpaths[FRAME_START_INDEX:FRAME_END_INDEX]
    pcd_fpaths = all_pcd_fpaths[FRAME_START_INDEX:FRAME_END_INDEX]

    merged_pcd = o3d.geometry.PointCloud()
    for pcd_fpath in pcd_fpaths:
        merged_pcd += load_pcd(pcd_fpath, mode='open3d')
    pts_in_ptc = np.array(merged_pcd.points)    # [N, 3]

    # load the camera parameters
    calib_fpath = os.path.join(ECAL_FOLDER, CAMERA_PARAM_FILENAME)
    fx, fy, cx, cy, rot_mat, translation = load_camera_parameters(calib_fpath)
    intrinsic_mat = camera_utils.make_intrinsic_mat(fx, fy, cx, cy)
    extrinsic_mat = camera_utils.make_extrinsic_mat(rot_mat, translation)

    for image_path in rgb_fpaths:
        # get the checker pattern points from the image
        corners = find_checkerboard_corners(image_path, pattern_size=PATTERN_SIZE)
        if corners is None:
            print(f"Error: failed to find corners in {image_path}")
            continue

        # project the point cloud to camera and its image sensor
        pts_in_cam = camera_utils.lidar2cam_projection(pts_in_ptc, extrinsic_mat)
        pts_in_img = camera_utils.cam2image_projection(pts_in_cam, intrinsic_mat)
        pts_in_img = pts_in_img.T[:, :-1]   # [N, 3]

        # get 3D point indices corresponding with checker pattern
        distances = []
        for pt2d_a, pt2d_b in combinations(corners, 2):
            _, pt_idx_a = closest_point(pt2d_a, pts_in_img[:, :2])
            _, pt_idx_b = closest_point(pt2d_b, pts_in_img[:, :2])
            pt3d_a = pts_in_ptc[pt_idx_a, :]
            pt3d_b = pts_in_ptc[pt_idx_b, :]
            length = np.linalg.norm(pt3d_a - pt3d_b)
            distances.append(pt3d_a[0])
            distances.append(pt3d_b[0])
            print(f'{length:.5f} m between {pt3d_a} and {pt3d_b}')

        # average distance to the target
        print(f'{np.mean(distances):.5f} m')

        break


if __name__ == '__main__':
    main()
