import os
import glob
import numpy as np
import pandas as pd

from utils.file_loader import load_pcd, load_camera_parameters
from utils import camera as camera_utils
from projection_functions import closest_point


ECAL_FOLDER = './data/martial_eagle_stand'
CAMERA_PARAM_FILENAME = 'manual_calibration.json'
X_RANGE = [18, 19]
Y_RANGE = [0, 1]
Z_RANGE = [-0.2, 0.4]
_KEYPOINTS = {   # giraffe_stand
    '000_004.jpeg': {
        'nose': [839, 159],
        'r_eye': [774, 113],
        'neck': [713, 499],
        'hip': [552, 559],
    },
    '001_005.jpeg': {
        'nose': [838, 159],
        'r_eye': [776, 114],
        'neck': [710, 497],
        'hip': [555, 557],
    },
    '002_006.jpeg': {
        'nose': [836, 160],
        'r_eye': [777, 115],
        'neck': [709, 497],
        'hip': [552, 561],
    },
    '003_007.jpeg': {
        'nose': [837, 159],
        'r_eye': [778, 113],
        'neck': [711, 499],
        'hip': [551, 555],
    },
    '004_008.jpeg': {
        'nose': [839, 160],
        'r_eye': [777, 114],
        'neck': [713, 500],
        'hip': [551, 562],
    },
    '005_009.jpeg': {
        'nose': [841, 162],
        'r_eye': [776, 115],
        'neck': [713, 497],
        'hip': [552, 559],
    },
    '006_010.jpeg': {
        'nose': [841, 161],
        'r_eye': [778, 115],
        'neck': [713, 504],
        'hip': [552, 558],
    },
    '007_011.jpeg': {
        'nose': [839, 160],
        'r_eye': [780, 114],
        'neck': [711, 503],
        'hip': [552, 558],
    },
    '008_012.jpeg': {
        'nose': [841, 163],
        'r_eye': [780, 116],
        'neck': [713, 497],
        'hip': [557, 559],
    },
    '009_013.jpeg': {
        'nose': [839, 160],
        'r_eye': [777, 114],
        'neck': [711, 505],
        'hip': [548, 561],
    },
    '010_014.jpeg': {
        'nose': [839, 163],
        'r_eye': [779, 117],
        'neck': [712, 503],
        'hip': [542, 558],
    },
    '011_015.jpeg': {
        'nose': [840, 163],
        'r_eye': [779, 118],
        'neck': [712, 500],
        'hip': [548, 563],
    },
    '012_016.jpeg': {
        'nose': [840, 164],
        'r_eye': [778, 116],
        'neck': [713, 505],
        'hip': [547, 558],
    },
    '013_017.jpeg': {
        'nose': [838, 164],
        'r_eye': [777, 118],
        'neck': [710, 497],
        'hip': [547, 559],
    },
    '014_018.jpeg': {
        'nose': [840, 166],
        'r_eye': [777, 119],
        'neck': [710, 501],
        'hip': [548, 565],
    },
    '015_019.jpeg': {
        'nose': [840, 162],
        'r_eye': [777, 118],
        'neck': [709, 500],
        'hip': [550, 563],
    },
    '016_020.jpeg': {
        'nose': [838, 162],
        'r_eye': [776, 117],
        'neck': [711, 503],
        'hip': [546, 561],
    },
}
KEYPOINTS = {   # martial_eagle_stand
    '000_004.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [108, 382],
    },
    '001_005.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '002_006.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '003_007.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '004_008.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '005_009.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '006_010.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '007_011.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '008_012.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '009_013.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '010_014.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '011_015.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '012_016.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '013_017.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '014_018.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '015_019.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '016_020.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
}
COMBINATIONS = [
    ['nose', 'r_eye'],
    ['r_eye', 'neck'],
    ['neck', 'hip'],
]


def main():
    # load the data
    rgb_fpaths = sorted(glob.glob(os.path.join(ECAL_FOLDER, 'merged_rgb', '*.jpeg')))
    pcd_fpaths = sorted(glob.glob(os.path.join(ECAL_FOLDER, 'merged_pcd', '*.pcd')))

    # load the camera parameters
    calib_fpath = os.path.join(ECAL_FOLDER, CAMERA_PARAM_FILENAME)
    fx, fy, cx, cy, rot_mat, translation = load_camera_parameters(calib_fpath)
    intrinsic_mat = camera_utils.make_intrinsic_mat(fx, fy, cx, cy)
    extrinsic_mat = camera_utils.make_extrinsic_mat(rot_mat, translation)

    result = []
    for rgb_fpath, pcd_fpath in zip(rgb_fpaths, pcd_fpaths):
        pcd_o3d = load_pcd(pcd_fpath, mode='open3d')
        pts_in_ptc = np.array(pcd_o3d.points)

        # filtering with ranges
        mask = (
            (X_RANGE[0] < pts_in_ptc[:, 0]) & (pts_in_ptc[:, 0] < X_RANGE[1]) &
            (Y_RANGE[0] < pts_in_ptc[:, 1]) & (pts_in_ptc[:, 1] < Y_RANGE[1]) &
            (Z_RANGE[0] < pts_in_ptc[:, 2]) & (pts_in_ptc[:, 2] < Z_RANGE[1])
        )
        filtered_pts_in_ptc = pts_in_ptc[mask]

        # load keypoints
        key = os.path.basename(rgb_fpath)
        keypoints = KEYPOINTS[key]

        # project the point cloud to camera and its image sensor
        pts_in_cam = camera_utils.lidar2cam_projection(filtered_pts_in_ptc, extrinsic_mat)
        pts_in_img = camera_utils.cam2image_projection(pts_in_cam, intrinsic_mat)
        pts_in_img = pts_in_img.T[:, :-1]   # [N, 3]

        # get 3D point indices corresponding with checker pattern
        result_row = []
        for kp_a, kp_b in COMBINATIONS:
            _, pt_idx_a = closest_point(keypoints[kp_a], pts_in_img[:, :2])
            _, pt_idx_b = closest_point(keypoints[kp_b], pts_in_img[:, :2])
            pt3d_a = filtered_pts_in_ptc[pt_idx_a, :]
            pt3d_b = filtered_pts_in_ptc[pt_idx_b, :]
            distance = np.linalg.norm(pt3d_a - pt3d_b)
            result_row.append(distance)
        result.append(result_row)

    df = pd.DataFrame(
        result,
        columns=[f'{a}–{b}' for a, b in COMBINATIONS]
    )
    df.to_csv('results/output.csv')
    print(df)


if __name__ == '__main__':
    main()