import hjson
import numpy as np
import open3d
import json
import cv2
from scipy.spatial.transform import Rotation as R


def load_config_file(fpath: str):
    config = None
    with open(fpath, mode='r') as f:
        config = hjson.loads(f.read())
    return config


def load_pcd(path: str, mode='open3d'):
    if mode == 'wildpose':
        with open(path, "r") as pcd_file:
            lines = [line.strip().split(" ") for line in pcd_file.readlines()]
        is_data = False
        data = []
        for line in lines:
            if line[0] == "DATA":
                is_data = True
            elif is_data:
                x = float(line[0])
                y = float(line[1])
                z = float(line[2])
                intensity = float(line[3])
                data.append([x, y, z, intensity])
        return np.asarray(data, dtype=np.float32).transpose()
    elif mode == 'open3d':
        pcd_data = open3d.io.read_point_cloud(path)
        return pcd_data
    else:
        return None


def load_rgb_img(fpath: str):
    bgr_img = cv2.imread(fpath)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img


def load_camera_parameters(path: str):
    with open(path, 'r') as f:
        cam_params = json.load(f)
    rot_mat = np.array(cam_params['extrinsics_R'])
    translation = np.array(cam_params['extrinsics_t'])
    fx, fy, cx, cy = cam_params['intrinsics']

    return fx, fy, cx, cy, rot_mat, translation
