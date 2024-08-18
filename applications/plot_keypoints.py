import os
import re
import cv2
import glob
import pickle
import hjson
import pandas as pd
import numpy as np
import scipy.ndimage
import open3d as o3d
from itertools import islice

from numpy import linalg as LA
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.graph_objs as go
import plotly.io as pio
pio.kaleido.scope.mathjax = None

from utils.file_loader import load_camera_parameters
from utils.camera import make_intrinsic_mat, make_extrinsic_mat
from utils.format_conversion import get_timestamp_from_img_fpath
from projection_functions import extract_rgb_from_image_pure, get_3d_from_2d_point


# plt.style.use(['science', 'nature', 'no-latex'])
# figure(figsize=(10, 6))
plt.rcParams.update({
    'legend.frameon': False,
    "pdf.fonttype": 42,
})

CONFIG = {
    "scene_dir": "data/red_hartebeest_walk2",
    "pcd_dir": "data/red_hartebeest_walk2/lidar",
    "sync_rgb_dir": "data/red_hartebeest_walk2/sync_rgb",
    "mask_dir": "data/red_hartebeest_walk2/masks",
    "textured_pcd_dir": "data/red_hartebeest_walk2/textured_pcds",
    "result_dir": "results",
}
IMG_WIDTH, IMG_HEIGHT = 1280, 720
XLIM = (-1, 1.3)
YLIM = (-100, 100)
ZLIM = (54.5, 56)
# FRAME_MARGIN = 0
FRAME_MARGIN = 2

JOINTS = hjson.load(open(os.path.join(CONFIG['scene_dir'], 'joints.hjson'), 'r'))
BONES = [
    ['nose', 'r_eye'], ['nose', 'l_eye'], ['r_eye', 'l_eye'],
    ['r_eye', 'r_scapula_base'], ['r_scapula_base', 'r_shoulder'], ['r_shoulder', 'r_elbow'], ['r_elbow', 'r_wrist'], ['r_wrist', 'r_palm'],
    ['r_scapula_base', 'r_hip'], ['r_hip', 'r_knee'], ['r_knee', 'r_ankle'], ['r_ankle', 'r_toe'],
    ['l_eye', 'l_scapula_base'], ['l_scapula_base', 'l_shoulder'], ['l_shoulder', 'l_elbow'], ['l_elbow', 'l_wrist'], ['l_wrist', 'l_palm'],
    ['l_scapula_base', 'l_hip'], ['l_hip', 'l_knee'], ['l_knee', 'l_ankle'], ['l_ankle', 'l_toe'],
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


def sliding_window(seq, window_size):
    it = iter(seq)
    result = tuple(islice(it, window_size))
    if len(result) == window_size:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitudes = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / magnitudes)
    return angle


def main(mode):
    # arguments
    data_dir = CONFIG['scene_dir']
    lidar_dir = CONFIG['pcd_dir']
    rgb_dir = CONFIG['sync_rgb_dir']
    calib_fpath = os.path.join(data_dir, 'manual_calibration.json')

    # load the camera parameters
    fx, fy, cx, cy, cam_rot_mat, cam_translation = load_camera_parameters(calib_fpath)
    intrinsic = make_intrinsic_mat(fx, fy, cx, cy)
    extrinsic = make_extrinsic_mat(cam_rot_mat, cam_translation)

    # load the masks
    mask_fpath = os.path.join(CONFIG['mask_dir'], 'metadata_result.pickle')
    mask_dict = pickle.load(open(mask_fpath, 'rb'))
    masks = np.array(mask_dict['masks'])

    # define bones
    if mode == 'full':
        bones = BONES
        target_joints = list(set(joint for bone in bones for joint in bone))
    elif mode == 'single_frame':
        bones = BONES
        target_joints = list(set(joint for bone in bones for joint in bone))
    elif mode == 'left_hind_leg':
        # plot the left hind leg bones
        bones = [
            ['l_hip', 'l_knee'], ['l_knee', 'l_ankle'], ['l_ankle', 'l_toe']
        ]
        target_joints = ['l_hip', 'l_knee', 'l_ankle', 'l_toe']
    elif mode == 'left_front_leg':
        # plot the left front leg bones
        bones = [
            ['l_scapula_base', 'l_shoulder'], ['l_shoulder', 'l_elbow'],
            ['l_elbow', 'l_wrist'], ['l_wrist', 'l_palm'],
        ]
        target_joints = ['l_scapula_base', 'l_shoulder', 'l_elbow', 'l_wrist', 'l_palm']
    elif mode == 'left_leg':
        bones = [
            ['l_scapula_base', 'l_shoulder'], ['l_shoulder', 'l_elbow'],
            ['l_elbow', 'l_wrist'], ['l_wrist', 'l_palm'],
            ['l_hip', 'l_knee'], ['l_knee', 'l_ankle'], ['l_ankle', 'l_toe']
        ]
        front_bones = [
            ['l_scapula_base', 'l_shoulder'], ['l_shoulder', 'l_elbow'],
            ['l_elbow', 'l_wrist'], ['l_wrist', 'l_palm'],
        ]
        hind_bones = [
            ['l_hip', 'l_knee'], ['l_knee', 'l_ankle'], ['l_ankle', 'l_toe']
        ]
        target_joints = ['l_scapula_base', 'l_shoulder', 'l_elbow', 'l_wrist', 'l_palm',
                         'l_hip', 'l_knee', 'l_ankle', 'l_toe']

    # prepare the plot
    fig3d = go.Figure()
    fig2d = go.Figure()

    num_frames = len(JOINTS)
    joints = []
    x_pos_margin = 0
    for idx in range(num_frames):
        # === Load Data ===
        # load the joint info
        joint_info = JOINTS[idx]
        # load the texture image
        img_fpath = os.path.join(rgb_dir, joint_info['rgb_filename'])
        rgb_img = load_rgb_img(img_fpath)
        # load the mask
        img_mask = masks[idx]   # (1, H, W)
        img_mask = np.squeeze(img_mask, axis=0)  # (H, W)
        # load the pcd
        pcd_fpath = os.path.join(lidar_dir, joint_info['lidar_filename'])
        pcd_in_lidar = o3d.io.read_point_cloud(pcd_fpath)
        pcd_points = np.asarray(pcd_in_lidar.points)  # [N, 3]

        # === Colored Point Cloud ===
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
        pcd_dist_mask = (
            (XLIM[0] < points[:, 0]) & (points[:, 0] < XLIM[1]) &
            (YLIM[0] < points[:, 1]) & (points[:, 1] < YLIM[1]) &
            (ZLIM[0] < points[:, 2]) & (points[:, 2] < ZLIM[1])
        )
        masked_points = points[pcd_dist_mask]
        masked_colors = colors[pcd_dist_mask]

        # mask the points that are out of the image segmentation
        pcd_img_mask = []
        for point in pcd_in_img:
            y = int(point[1])
            x = int(point[0])
            # if 0 <= y < IMG_HEIGHT and 0 <= x < IMG_WIDTH:
            #     img_mask_idx = (y, x)
            #     pcd_img_mask.append(img_mask[img_mask_idx])
            # else:
            #     pcd_img_mask.append(False)
            pcd_img_mask.append(True)
        pcd_img_mask = np.array(pcd_img_mask)

        # === Collect 3D coordinates of joints ===
        joint_3d_coords = {}
        for joint_name, joint_2d in joint_info.items():
            if isinstance(joint_2d, list):
                if joint_2d[0] is not None and joint_2d[1] is not None:
                    _, masked_pcd_in_img_idx = get_3d_from_2d_point(
                        pcd_in_img[pcd_img_mask, :], joint_2d, z_range=ZLIM)
                    if masked_pcd_in_img_idx is None:
                        raise 'Error: the axis ranges were too narrow.'

                    # masked_pcd_in_img_idx -> pcd_in_cam_idx
                    pcd_in_cam_idx = np.where(pcd_img_mask)[
                        0][masked_pcd_in_img_idx]

                    joint_3d_coords[joint_name] = pcd_in_cam[pcd_in_cam_idx, :]
                    print(f'{joint_name}: {joint_3d_coords[joint_name]}')
                else:
                    joint_3d_coords[joint_name] = None
            else:
                joint_3d_coords[joint_name] = joint_2d
        joints.append(joint_3d_coords)

        # === Visualize in 3D ===
        if mode == 'full' or mode == 'single_frame':
            if mode == 'single_frame' and idx != 9:
                continue

            # plot the PCD frame
            point_cloud_scatter = go.Scatter3d(
                x=masked_points[:, 0] + x_pos_margin,
                y=masked_points[:, 2],
                z=masked_points[:, 1],
                mode='markers',
                marker=dict(size=3, color=masked_colors)
            )
            fig3d.add_trace(point_cloud_scatter)

            # plot joints
            for joint_name, joint_2d in joint_info.items():
                if isinstance(joint_2d, list) and joint_2d[0] is not None and joint_2d[1] is not None:
                    keypoint_scatter = go.Scatter3d(
                        x=[joint_3d_coords[joint_name][0] + x_pos_margin],
                        y=[joint_3d_coords[joint_name][2]],
                        z=[joint_3d_coords[joint_name][1]],
                        mode='markers',
                        marker=dict(size=4, color='red')
                    )
                    fig3d.add_trace(keypoint_scatter)

            # plot bones
            for bone in bones:
                start_joint, end_joint = bone
                if joint_3d_coords[start_joint] is not None and joint_3d_coords[end_joint] is not None:
                    bone_line = go.Scatter3d(
                        x=[joint_3d_coords[start_joint][0] + x_pos_margin, joint_3d_coords[end_joint][0] + x_pos_margin],
                        y=[joint_3d_coords[start_joint][2], joint_3d_coords[end_joint][2]],
                        z=[joint_3d_coords[start_joint][1], joint_3d_coords[end_joint][1]],
                        mode='lines',
                        line=dict(width=3, color='blue')  # Customize the color as needed
                    )
                    fig3d.add_trace(bone_line)

            # increase the margin
            x_pos_margin -= FRAME_MARGIN
        elif mode == 'left_hind_leg' or mode == 'left_front_leg':
            for bone in bones:
                start_joint, end_joint = bone
                if joint_3d_coords[start_joint] is not None and joint_3d_coords[end_joint] is not None:
                    bone_line = go.Scatter3d(
                        x=[joint_3d_coords[start_joint][0],
                            joint_3d_coords[end_joint][0]],
                        y=[joint_3d_coords[start_joint][2],
                            joint_3d_coords[end_joint][2]],
                        z=[joint_3d_coords[start_joint][1],
                            joint_3d_coords[end_joint][1]],
                        mode='lines',
                        # Customize the color as needed
                        line=dict(width=3, color='blue')
                    )
                    fig3d.add_trace(bone_line)
        elif mode == 'left_leg':
            if idx == 9:
                # plot the PCD frame
                point_cloud_scatter = go.Scatter3d(
                    x=masked_points[:, 0],
                    y=masked_points[:, 2],
                    z=masked_points[:, 1],
                    mode='markers',
                    marker=dict(size=3, color=masked_colors)
                )
                fig3d.add_trace(point_cloud_scatter)

            for _bones in [front_bones, hind_bones]:
                color = 'red' if _bones == front_bones else 'blue'
                for bone in _bones:
                    start_joint, end_joint = bone
                    if joint_3d_coords[start_joint] is not None and joint_3d_coords[end_joint] is not None:
                        bone_line = go.Scatter3d(
                            x=[joint_3d_coords[start_joint][0],
                                joint_3d_coords[end_joint][0]],
                            y=[joint_3d_coords[start_joint][2],
                                joint_3d_coords[end_joint][2]],
                            z=[joint_3d_coords[start_joint][1],
                                joint_3d_coords[end_joint][1]],
                            mode='lines',
                            # Customize the color as needed
                            line=dict(width=3, color=color)
                        )
                        fig3d.add_trace(bone_line)

    # save the joints
    with open(os.path.join(CONFIG['result_dir'], "joints.pkl"), "wb") as f:
        pickle.dump(joints, f)

    # === Get the Sagittal Plane Plot ===
    if mode == 'left_hind_leg' or mode == 'left_front_leg':
        # get the keypoints
        keypoints = []
        for joint in joints:
            for joint_name in target_joints:
                keypoints.append(joint[joint_name])
        keypoints = np.array(keypoints)

        # get the sagittal plane with RANSAC algorithm
        keypoint_pcd = o3d.geometry.PointCloud()
        keypoint_pcd.points = o3d.utility.Vector3dVector(keypoints)

        # estimate the global ground plane
        global_plane_model, inliers = keypoint_pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )   # Plane model: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0
        a, b, c, d = global_plane_model

        # Calculate the centroid of inlier points
        inlier_cloud = keypoint_pcd.select_by_index(inliers)
        inlier_points = np.asarray(inlier_cloud.points)
        centroid = np.mean(inlier_points, axis=0)

        # plot the plane in Plotly
        xx, yy = np.meshgrid(np.linspace(centroid[0]-1, centroid[0]+1, 2),
                             np.linspace(centroid[1]-1, centroid[1]+1, 2))
        z = (-d - a * xx - b * yy) / c
        plane_mesh = go.Surface(x=xx, y=z, z=yy, opacity=0.5, colorscale='Greys')
        fig3d.add_trace(plane_mesh)

        # project the keypoints to the plane
        projected_keypoints = []
        for point in keypoints:
            # Calculate the vector from the point to the plane
            v = np.array(point) - centroid
            # Calculate the distance from the point to the plane
            distance = np.abs(np.dot(v, [a, b, c])) / np.sqrt(a**2 + b**2 + c**2)
            # calculate the sign of the distance
            sign = np.sign(np.dot(v, [a, b, c]))
            # Calculate the projection of the point onto the plane
            projection = np.array(point) - sign * distance * np.array([a, b, c])
            projected_keypoints.append(projection)

        # the projected keypoints in a 2D plot
        projected_keypoints = np.array(projected_keypoints) - centroid[np.newaxis, :]
        # rotation matrix from the plane vector to x-y plane
        rotation, _ = Rotation.align_vectors(np.array([[0, -1, 0]]), np.array([[a, b, c]]))
        rot_mat = rotation.as_matrix()
        # rotate the keypoints
        keypoints_rotated = np.dot(projected_keypoints, rot_mat.T)

        # rotate the 2D keypoints to be aligned with the horizontal line
        keypoints_2d = keypoints_rotated[:, [0, 2]]
        # Find the line of best fit using least squares with `scapula_base` points
        indices = list(range(0, len(keypoints_2d), len(target_joints)))
        x = keypoints_2d[indices, 0]
        y = keypoints_2d[indices, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        # Calculate the angle of rotation
        angle = np.arctan(m)
        # Create rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        keypoints_2d_aligned = np.dot(keypoints_2d, rotation_matrix)

        # get angles of each joint
        angles_deg = {}
        # [n_keypoints * n_frames, 2] -> [n_frames, n_keypoints, 2]
        keypoints = keypoints_2d_aligned.reshape(num_frames, -1, 2)
        for frame_idx in range(num_frames):
            for bone in bones:
                # calculate the angle of the bone along horizontal line
                vec = keypoints[frame_idx, target_joints.index(bone[1]), :] - keypoints[frame_idx, target_joints.index(bone[0]), :]
                angle_deg = np.rad2deg(np.arctan2(vec[0], vec[1]))
                # append the angle to the dictionary
                joint_name = f'{bone[0]}-{bone[1]}'
                if joint_name not in angles_deg:
                    angles_deg[joint_name] = []
                angles_deg[joint_name].append(angle_deg)

        # show the angles in Plotly
        fig_angle = go.Figure()
        # Plot angles over time
        time_steps = sorted(
            [get_timestamp_from_img_fpath(j['rgb_filename']) for j in JOINTS]
        )
        time_steps = np.array(time_steps) - time_steps[0]
        for joint, angles in angles_deg.items():
            fig_angle.add_trace(go.Scatter(
                x=time_steps,
                y=angles,
                mode='lines',
                name=joint
            ))

        fig_angle.update_layout(
            xaxis_title='Time (s)',
            yaxis_title='Angle (degrees)',
            font=dict(family='Arial', size=14),
            legend_title='Joints'
        )

        fig_angle.write_html(os.path.join(
            CONFIG['result_dir'], "joint_angles_plot.html"))
        fig_angle.write_image(os.path.join(
            CONFIG['result_dir'], "joint_angles_plot.pdf"))
        # fig_angle.show()

        # show bones in 2D
        for i in range(0, len(keypoints_2d_aligned) - 1):
            if (i - len(target_joints) + 1) % len(target_joints) == 0:
                continue

            keypoint_1 = keypoints_2d_aligned[i]
            keypoint_2 = keypoints_2d_aligned[i+1]
            fig2d.add_trace(go.Scatter(
                x=[keypoint_1[0], keypoint_2[0]],
                y=[keypoint_1[1], keypoint_2[1]],
                mode='lines',
                line=dict(width=3, color='blue')
            ))
        # show keypoints in 2D
        fig2d.add_trace(go.Scatter(
            x=keypoints_2d_aligned[:, 0],
            y=keypoints_2d_aligned[:, 1],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Projected Keypoints'
        ))

    # === Update Layout ===
    def _axis_dict(title, reverse=False):
        return dict(
            title=title,
            ticks='outside',
            tickangle=0,
            backgroundcolor='rgb(230, 230, 230)',
            tickformat='.1f',
            autorange='reversed' if reverse else None,
        )

    fig3d.update_layout(
        font_family='Arial',
        font_size=14,
        scene=dict(
            xaxis=_axis_dict('x (m)'),
            yaxis=_axis_dict('Depth (m)'),
            zaxis=_axis_dict('y (m)', reverse=True),
            aspectmode='data',
        ),
    )
    fig3d.update_layout(
        showlegend=False
    )
    if mode == 'single_frame':
        fig3d.layout.scene.camera.projection.type = "orthographic"
    elif mode == 'left_leg' or mode == 'left_hind_leg' or mode == 'left_front_leg':
        fig3d.layout.scene.camera.projection.type = "orthographic"
        fig2d.update_layout(
            font_family='Arial',
            font_size=14,
            xaxis_title='x (m)',
            yaxis_title='y (m)',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig2d.update_yaxes(
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1
        )
        fig2d.update_layout(
            showlegend=False
        )
        fig2d.update_xaxes(showgrid=False)
        fig2d.update_yaxes(showgrid=False)

    fig3d.write_html(os.path.join(
        CONFIG['result_dir'], "keypoints_plot_3d.html"))
    fig3d.write_image(os.path.join(
        CONFIG['result_dir'], "keypoints_plot_3d.pdf"))
    fig2d.write_html(os.path.join(
        CONFIG['result_dir'], "keypoints_plot_2d.html"))
    fig2d.write_image(os.path.join(
        CONFIG['result_dir'], "keypoints_plot_2d.pdf"))

    fig3d.show()
    # fig2d.show()


if __name__ == '__main__':
    main(mode='single_frame')
