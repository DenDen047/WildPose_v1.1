import os
import re
import cv2
import glob
import json
import pandas as pd
import numpy as np
import scipy.ndimage
import open3d as o3d

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.graph_objs as go

from utils.file_loader import load_camera_parameters
from utils.camera import make_intrinsic_mat, make_extrinsic_mat
from projection_functions import extract_rgb_from_image_pure, get_3d_from_2d_point


# plt.style.use(['science', 'nature', 'no-latex'])
# figure(figsize=(10, 6))
plt.rcParams.update({
    'legend.frameon': False,
    "pdf.fonttype": 42,
})

CONFIG = {
    "scene_dir": "data/lion_walk",
    "pcd_dir": "data/lion_walk/lidar",
    "sync_rgb_dir": "data/lion_walk/sync_rgb",
    "textured_pcd_dir": "data/lion_walk/textured_pcds",
}
IMG_WIDTH, IMG_HEIGHT = 1280, 720
XLIM = (-100, 100)
YLIM = (-100, 100)
ZLIM = (42, 45)
FRAME_MARGIN = 1.5

JOINTS = [
    {
        'rgb_filename': '659835_1670481560_458554312.jpeg',
        'lidar_filename': 'livox_frame_1670481560_459611816.pcd',
        'nose': [473, 215],
        'r_eye': [436, 158],
        'l_eye': [489, 149],
        'r_shoulder': [None, None],
        'r_elbow': [None, None],
        'r_wrist': [495, 437],
        'r_hip': [None, None],
        'r_knee': [None, None],
        'r_ankle': [None, None],
        'l_shoulder': [593, 192],
        'l_elbow': [595, 337],
        'l_wrist': [596, 481],
        'l_hip': [769, 377],
        'l_knee': [663, 377],
        'l_ankle': [715, 475],
    },
    {
        'rgb_filename': '660005_1670481561_458553064.jpeg',
        'lidar_filename': 'livox_frame_1670481561_459852744.pcd',
        'nose': [577, 164],
        'r_eye': [543, 100],
        'l_eye': [595, 102],
        'r_shoulder': [457, 174],
        'r_elbow': [494, 330],
        'r_wrist': [533, 480],
        'r_hip': [None, None],
        'r_knee': [None, None],
        'r_ankle': [None, None],
        'l_shoulder': [None, None],
        'l_elbow': [596, 329],
        'l_wrist': [597, 416],
        'l_hip': [None, None],
        'l_knee': [None, None],
        'l_ankle': [None, None],
    },
    {
        'rgb_filename': '660260_1670481562_958554632.jpeg',
        'lidar_filename': 'livox_frame_1670481562_958825160.pcd',
        'nose': [844, 274],
        'r_eye': [817, 202],
        'l_eye': [None, None],
        'r_shoulder': [641, 231],
        'r_elbow': [614, 355],
        'r_wrist': [660, 457],
        'r_hip': [535, 172],
        'r_knee': [555, 296],
        'r_ankle': [544, 422],
        'l_shoulder': [None, None],
        'l_elbow': [None, None],
        'l_wrist': [None, None],
        'l_hip': [None, None],
        'l_knee': [None, None],
        'l_ankle': [None, None],
    },
    {
        'rgb_filename': '660447_1670481564_058760744.jpeg',
        'lidar_filename': 'livox_frame_1670481564_059593064.pcd',
        'nose': [829, 328],
        'r_eye': [800, 264],
        'l_eye': [852, 267],
        'r_shoulder': [679, 259],
        'r_elbow': [693, 426],
        'r_wrist': [None, None],
        'r_hip': [588, 202],
        'r_knee': [568, 333],
        'r_ankle': [553, 441],
        'l_shoulder': [None, None],
        'l_elbow': [814, 431],
        'l_wrist': [836, 516],
        'l_hip': [None, None],
        'l_knee': [None, None],
        'l_ankle': [None, None],
    }
]

BONES = [
    ['nose', 'r_eye'], ['nose', 'l_eye'], ['r_eye', 'l_eye'],
    ['r_eye', 'r_shoulder'], ['r_shoulder', 'r_elbow'], ['r_elbow', 'r_wrist'],
    ['r_shoulder', 'r_hip'], ['r_hip', 'r_knee'], ['r_knee', 'r_ankle'],
    ['l_eye', 'l_shoulder'], ['l_shoulder', 'l_elbow'], ['l_elbow', 'l_wrist'],
    ['l_shoulder', 'l_hip'], ['l_hip', 'l_knee'], ['l_knee', 'l_ankle'],
]


def lidar2cam_projection(pcd, extrinsic):
    tmp = np.insert(pcd, 3, 1, axis=1).T
    tmp = np.delete(tmp, np.where(tmp[0, :] < 0), axis=1)
    pcd_in_cam = extrinsic.dot(tmp)

    return pcd_in_cam


def cam2image_projection(pcd_in_cam, intrinsic):
    pcd_in_image = intrinsic.dot(pcd_in_cam)
    pcd_in_image[:2] /= pcd_in_image[2, :]

    return pcd_in_image


def sync_lidar_and_rgb(lidar_dir, rgb_dir):
    rgb_fpaths = sorted(glob.glob(os.path.join(rgb_dir, '*.jpeg')))
    lidar_fpaths = sorted(glob.glob(os.path.join(lidar_dir, '*.pcd')))

    rgb_list = []
    lidar_list = []

    for rgb_fpath in rgb_fpaths:
        rgb_filename = os.path.basename(rgb_fpath).split('.')[0]
        second_rgb, decimal_rgb = rgb_filename.split('_')[1:3]

        second_rgb = int(second_rgb)
        decimal_rgb = float('0.' + decimal_rgb)
        rgb_timestamp = second_rgb + decimal_rgb

        diff_list = []
        lidar_fp_list = []
        for lidar_fpath in lidar_fpaths:
            lidar_filename = os.path.basename(lidar_fpath).split('.')[0]
            _, _, second_lidar, decimal_lidar = lidar_filename.split('_')
            second_lidar = int(second_lidar)
            decimal_lidar = float('0.' + decimal_lidar)

            lidar_timestamp = second_lidar + decimal_lidar
            diff = abs(rgb_timestamp - lidar_timestamp)

            diff_list.append(diff)
            lidar_fp_list.append(lidar_fpath)

        diff_list = np.array(diff_list)
        matching_lidar_file = lidar_fp_list[np.argmin(diff_list)]

        rgb_list.append(rgb_fpath)
        assert os.path.exists(matching_lidar_file)
        lidar_list.append(matching_lidar_file)

    return lidar_list, rgb_list


def get_2D_gt(gt_json_path):

    gt_json = json.load(open(gt_json_path, 'r'))
    img_dict = {}

    annotations = gt_json['annotations']
    images = gt_json['images']

    for idx in range(len(annotations)):

        anno = annotations[idx]
        bbox = anno['bbox']
        img_id = anno['image_id']
        obj_id = anno['category_id']
        img_filename = os.path.basename(images[img_id]['file_name'])

        if img_filename not in img_dict:
            img_dict[img_filename] = []
        img_dict[img_filename].append([bbox, obj_id])
    return img_dict


def load_rgb_img(fpath: str):
    bgr_img = cv2.imread(fpath)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img


def equal_3d_aspect(ax):
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    ax.set_box_aspect((
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        zlim[1] - zlim[0]
    ))


def main():
    # arguments
    data_dir = CONFIG['scene_dir']
    lidar_dir = CONFIG['pcd_dir']
    rgb_dir = CONFIG['sync_rgb_dir']
    calib_fpath = os.path.join(data_dir, 'manual_calibration.json')
    output_dir = CONFIG['textured_pcd_dir']

    lidar_list, rgb_list = sync_lidar_and_rgb(lidar_dir, rgb_dir)

    # load the camera parameters
    fx, fy, cx, cy, rot_mat, translation = load_camera_parameters(calib_fpath)
    intrinsic = make_intrinsic_mat(fx, fy, cx, cy)
    extrinsic = make_extrinsic_mat(rot_mat, translation)

    # prepare the plot
    fig = go.Figure()

    x_pos_margin = 0
    for joint_info in JOINTS:
        # load the texture image
        img_fpath = os.path.join(rgb_dir, joint_info['rgb_filename'])
        rgb_img = load_rgb_img(img_fpath)

        # load the pcd
        pcd_fpath = os.path.join(lidar_dir, joint_info['lidar_filename'])
        pcd_in_lidar = o3d.io.read_point_cloud(pcd_fpath)
        pcd_points = np.asarray(pcd_in_lidar.points)  # [N, 3]

        # project the point cloud to camera and its image sensor
        pcd_in_cam = lidar2cam_projection(pcd_points, extrinsic)
        pcd_in_img = cam2image_projection(pcd_in_cam, intrinsic)
        pcd_in_cam = pcd_in_cam.T[:, :-1]
        pcd_in_img = pcd_in_img.T[:, :-1]

        pcd_colors, valid_mask_save = extract_rgb_from_image_pure(
            pcd_in_img, rgb_img, width=IMG_WIDTH, height=IMG_HEIGHT)
        pcd_with_rgb = np.concatenate([pcd_in_cam, pcd_colors], 1)
        pcd_with_rgb = pcd_with_rgb[valid_mask_save]  # [N, 6]
        textured_pcd = o3d.geometry.PointCloud()
        textured_pcd.points = o3d.utility.Vector3dVector(
            pcd_with_rgb[:, :3])
        textured_pcd.colors = o3d.utility.Vector3dVector(
            pcd_with_rgb[:, 3:])

        # load data
        points = np.array(textured_pcd.points)  # [N, 3]
        colors = np.array(textured_pcd.colors)  # [N, 3]

        # make the mask
        mask = (
            (XLIM[0] < points[:, 0]) & (points[:, 0] < XLIM[1]) &
            (YLIM[0] < points[:, 1]) & (points[:, 1] < YLIM[1]) &
            (ZLIM[0] < points[:, 2]) & (points[:, 2] < ZLIM[1])
        )
        masked_points = points[mask]
        masked_colors = colors[mask]

        # plot the PCD frame
        point_cloud_scatter = go.Scatter3d(
            x=masked_points[:, 0] + x_pos_margin,
            y=masked_points[:, 1],
            z=masked_points[:, 2],
            mode='markers',
            marker=dict(size=2, color=masked_colors)
        )
        fig.add_trace(point_cloud_scatter)

        # plot joints and collect 3D coordinates of joints
        joint_3d_coords = {}
        for joint_name, joint_2d in joint_info.items():
            if isinstance(joint_2d, list) and joint_2d[0] is not None and joint_2d[1] is not None:
                _, idx = get_3d_from_2d_point(pcd_in_img, joint_2d, z_range=ZLIM)
                if idx is None:
                    raise 'Error: the axis ranges were too narrow.'
                joint_3d_coords[joint_name] = pcd_in_cam[idx, :]
                keypoint_scatter = go.Scatter3d(
                    x=[joint_3d_coords[joint_name][0] + x_pos_margin],
                    y=[joint_3d_coords[joint_name][1]],
                    z=[joint_3d_coords[joint_name][2]],
                    mode='markers',
                    marker=dict(size=3, color='red')
                )
                fig.add_trace(keypoint_scatter)
                print(f'{joint_name}: {joint_3d_coords[joint_name]}')
            else:
                joint_3d_coords[joint_name] = None

        # plot bones
        for bone in BONES:
            start_joint, end_joint = bone
            if joint_3d_coords[start_joint] is not None and joint_3d_coords[end_joint] is not None:
                bone_line = go.Scatter3d(
                    x=[joint_3d_coords[start_joint][0] + x_pos_margin, joint_3d_coords[end_joint][0] + x_pos_margin],
                    y=[joint_3d_coords[start_joint][1], joint_3d_coords[end_joint][1]],
                    z=[joint_3d_coords[start_joint][2], joint_3d_coords[end_joint][2]],
                    mode='lines',
                    line=dict(width=3, color='blue')  # Customize the color as needed
                )
                fig.add_trace(bone_line)

        # increase the margin
        x_pos_margin += FRAME_MARGIN

    def _axis_dict(title):
        return dict(
            title=title,
            ticks='outside',
            tickangle=0,
            backgroundcolor='rgb(230, 230, 230)',
            tickformat='.1f',
        )

    fig.update_layout(
        font_family='Arial',
        font_size=14,
        scene=dict(
            xaxis=_axis_dict('x (m)'),
            yaxis=_axis_dict('y (m)'),
            zaxis=_axis_dict('Depth (m)'),
            aspectmode='data',
        ),
    )
    # fig.layout.scene.camera.projection.type = "orthographic"

    fig.show()
    # fig.write_html("results/test.html", include_mathjax='cdn')


if __name__ == '__main__':
    main()