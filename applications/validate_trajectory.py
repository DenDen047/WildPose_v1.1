import os
import re
import glob
import pandas as pd
import numpy as np
import json
import cv2
import pickle
from tqdm import tqdm
import scipy.ndimage
import open3d as o3d
import math
import random
import time
import argparse
from loguru import logger

import plotly.graph_objs as go
import plotly.express as px
import plotly.colors as pc
import plotly.subplots as sp
import plotly.io as pio
pio.kaleido.scope.mathjax = None

from utils.file_loader import load_camera_parameters, load_rgb_img, load_pcd
from utils.camera import make_intrinsic_mat, make_extrinsic_mat
from utils.projection import lidar2cam_projection, cam2image_projection
from utils.format_conversion import get_timestamp_from_img_fpath
from config import COLORS, colors_indices

from projection_functions import closest_point


IMG_WIDTH, IMG_HEIGHT = 1920, 1080
DATA_DIR = '/Users/ikuta/Documents/Projects/PhD/WildPose_v1.1/data/calibration_v1.2/2024-05-26_15-28-32.795_measurement'

# set random seeds with the current time
random.seed(42)
np.random.seed(42)


def mean_ignore_nan(arr):
    arr = arr.flatten()
    return np.sum(arr[~np.isnan(arr)]) / np.sum(~np.isnan(arr))


def erode_mask(mask, kernel_size=(5, 5), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
    return eroded_mask


def circle_fitting(x, y):
    """最小二乗法による円フィッティングをする関数
        input: x,y 円フィッティングする点群

        output  cxe 中心x座標
                cye 中心y座標
                re  半径

        参考
        一般式による最小二乗法（円の最小二乗法） 画像処理ソリューション
        http://imagingsolution.blog107.fc2.com/blog-entry-16.html
    """

    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum([ix ** 2 for ix in x])
    sumy2 = sum([iy ** 2 for iy in y])
    sumxy = sum([ix * iy for (ix, iy) in zip(x, y)])

    F = np.array([[sumx2, sumxy, sumx],
                  [sumxy, sumy2, sumy],
                  [sumx, sumy, len(x)]])

    G = np.array([[-sum([ix ** 3 + ix * iy ** 2 for (ix, iy) in zip(x, y)])],
                  [-sum([ix ** 2 * iy + iy ** 3 for (ix, iy) in zip(x, y)])],
                  [-sum([ix ** 2 + iy ** 2 for (ix, iy) in zip(x, y)])]])

    T = np.linalg.inv(F).dot(G)

    cxe = float(T[0] / -2)
    cye = float(T[1] / -2)
    r = math.sqrt(cxe**2 + cye**2 - T[2])

    return (cxe, cye, r)


def distance_to_circle(points, center, radius):
    # Calculate distances as a matrix operation
    diff = points - center
    distances = np.abs(np.linalg.norm(diff, axis=1) - radius)
    return distances


def ransac_circle_fit(points, n_sample=5, max_iterations=1000, threshold=2e-2, min_inliers=0.6):
    best_model = None
    best_inliers = []
    n_points = len(points)

    for i in range(max_iterations):
        # Randomly sample 3 points
        sample = random.sample(range(n_points), n_sample)
        x_sample = points[sample, 0]
        y_sample = points[sample, 1]

        # Fit a circle to these points
        try:
            center_x, center_y, radius = circle_fitting(x_sample, y_sample)
        except np.linalg.LinAlgError:
            continue  # Skip if the matrix is singular

        # Count inliers
        distances = distance_to_circle(points, np.array([center_x, center_y]), radius)
        inliers = points[distances < threshold]

        # Check if this model is the best so far
        if len(inliers) > len(best_inliers):
            best_model = (center_x, center_y, radius)
            best_inliers = inliers

        # If we found a model that fits enough points, we're done
        if len(inliers) > min_inliers * n_points:
            break

        if i == max_iterations - 1:
            print('Warning: Max iterations reached')

    # Refit the model using all inliers
    if best_model is not None:
        x_inliers = [p[0] for p in best_inliers]
        y_inliers = [p[1] for p in best_inliers]
        best_model = circle_fitting(x_inliers, y_inliers)

    return best_model, best_inliers


def estimate_start_end_angle(points_2d, center=None):
    if center is None:
        center = np.mean(points_2d, axis=0)

    start_angle = np.arctan2(points_2d[0, 1] - center[1], points_2d[0, 0] - center[0])
    end_angle = np.arctan2(points_2d[-1, 1] - center[1], points_2d[-1, 0] - center[0])
    # Convert to 0-2pi range
    start_angle = (start_angle + 2*np.pi) % (2*np.pi)
    end_angle = (end_angle + 2*np.pi) % (2*np.pi)
    return start_angle, end_angle


def calculate_trajectory_errors(
    points_2d,
    cx, cy, r,
    timestamps,
    n_revolutions=2,
):
    """Temporal consistency analysis.

    Args:
        points_2d: Nx2 array of reconstructed trajectory points
        cx, cy, r: Circle parameters
        timestamps: Array of timestamps for temporal analysis
        n_revolutions: Number of complete revolutions (default: 2)

    Returns:
        dict: Error metrics including spatial and temporal components
    """
    # Calculate center based on known distance and x_offset
    center = np.array([cx, cy])

    # Convert timestamps to seconds from start
    t = timestamps - timestamps[0]

    # Calculate expected positions for constant speed motion
    total_time = t[-1]
    # Starting from π (leftmost position) and completing n_revolutions
    start_angle, end_angle = estimate_start_end_angle(points_2d, center=center)
    total_angle = 2 * np.pi * n_revolutions - np.abs(end_angle - start_angle)
    angular_velocity = total_angle / total_time
    expected_angles = start_angle + angular_velocity * t

    # Generate expected positions
    expected_positions = np.zeros_like(points_2d)
    expected_positions[:, 0] = center[0] + r * np.cos(expected_angles)
    expected_positions[:, 1] = center[1] + r * np.sin(expected_angles)

    # Calculate temporal position errors
    temporal_errors = np.linalg.norm(points_2d - expected_positions, axis=1)

    return temporal_errors, expected_positions


def main(
    mode,
    data_dir,
    ref_distance=1.0,
    ref_radius=2.0,
    n_revolutions=2,
):
    lidar_dir = os.path.join(data_dir, 'lidar')
    rgb_dir = os.path.join(data_dir, 'sync_rgb')
    mask_dir = os.path.join(data_dir, 'masks')
    calib_fpath = os.path.join(data_dir, 'manual_calibration.json')
    result_dir = os.path.join(data_dir, f'{mode}_results')
    log_fpath = os.path.join(result_dir, 'loguru.log')
    logger.add(log_fpath)

    os.makedirs(result_dir, exist_ok=True)

    # load data
    img_fpaths = sorted(glob.glob(os.path.join(rgb_dir, '*.jpeg')))
    pcd_fpaths = sorted(glob.glob(os.path.join(lidar_dir, '*.pcd')))
    assert len(img_fpaths) == len(pcd_fpaths)
    n_frame = len(img_fpaths)
    mask_fpath = os.path.join(mask_dir, 'metadata_result.pickle')
    mask_info = pickle.load(open(mask_fpath, 'rb'))
    fx, fy, cx, cy, rot_mat, translation = load_camera_parameters(calib_fpath)
    intrinsic = make_intrinsic_mat(fx, fy, cx, cy)
    extrinsic = make_extrinsic_mat(rot_mat, translation)

    # prepare the masks list([n_frame, n_id, H, W])
    masks = mask_info['masks']
    timestamp0 = get_timestamp_from_img_fpath(img_fpaths[0])

    positions_3d_fpath = os.path.join(result_dir, 'positions_3d.pickle')
    if os.path.exists(positions_3d_fpath):
        logger.info(f'Loading positions_3d from {positions_3d_fpath}')
        with open(positions_3d_fpath, 'rb') as f:
            positions_3d = pickle.load(f)
    else:
        # collect the 3D positions with Segment Anything Model
        positions_3d = {}
        for i, (img_fpath, pcd_fpath, seg_mask) in tqdm(enumerate(zip(img_fpaths, pcd_fpaths, masks)), total=n_frame):
            # load the frame
            # rgb_img = load_rgb_img(img_fpath)
            pcd_open3d = load_pcd(pcd_fpath, mode='open3d')
            pts_in_lidar = np.asarray(pcd_open3d.points)
            seg_mask = np.array(seg_mask)  # [n_id, H, W]
            timestamp = get_timestamp_from_img_fpath(img_fpath)

            # reprojection
            pcd_in_cam = lidar2cam_projection(pts_in_lidar, extrinsic)
            pcd_in_img = cam2image_projection(pcd_in_cam, intrinsic)
            pcd_in_cam = pcd_in_cam.T[:, :-1]
            pcd_in_img = pcd_in_img.T[:, :-1]

            # # eroded_2d_mask -> median 3D point
            # # erode the segmentation mask to reduce the error of estimated 3d positions
            # for i in range(seg_mask.shape[0]):
            #     # seg_mask.shape should be (n, 1, H, W)
            #     seg_mask[i, 0, :, :] = erode_mask(seg_mask[i, 0, :, :], kernel_size=(5,5), iterations=4)

            # colors, valid_mask, obj_points, obj_mask_from_color = extract_rgb_from_image(
            #     pcd_in_img, pcd_in_cam, rgb_img, seg_mask, obj_ids,
            #     width=IMG_WIDTH, height=IMG_HEIGHT
            # )

            # # store the position data
            # for id, points in obj_points.items():
            #     position_3d = np.median(points, axis=0)
            #     if id not in positions_3d.keys():
            #         positions_3d[id] = []
            #     positions_3d[id].append([timestamp] + position_3d.tolist())

            # median_2d -> 3d point
            n_id = seg_mask.shape[0]
            assert n_id == len(mask_info['id2label'])
            for id, label in mask_info['id2label'].items():
                mask = seg_mask[id]
                if np.sum(mask) == 0:
                    positions_3d[label].append([timestamp] + [None] * 3)
                else:
                    mask_ys, mask_xs = np.where(mask)
                    target_2d_pt = np.array([
                        np.median(mask_xs),
                        np.median(mask_ys),
                    ]) / mask_info['scale_factor']
                    _, pt_idx = closest_point(target_2d_pt, pcd_in_img[:, :2])
                    pt3d = pcd_in_cam[pt_idx, :]
                    if label not in positions_3d.keys():
                        positions_3d[label] = []
                    positions_3d[label].append([timestamp] + pt3d.tolist())

        # save the data
        with open(positions_3d_fpath, 'wb') as f:
            pickle.dump(positions_3d, f, protocol=pickle.HIGHEST_PROTOCOL)

    # array to dataframe
    dfs = {}
    for key in positions_3d.keys():
        dfs[key] = pd.DataFrame(
            positions_3d[key],
            columns=['time', 'x', 'y', 'z']
        )

    # filter the positions
    # dfs = median_filter_3d_positions(dfs, filter_size=5)

    fig = go.Figure()
    if mode == 'position_3d':
        # plot the data
        duration = get_timestamp_from_img_fpath(img_fpaths[-1]) - timestamp0
        for label, v in dfs.items():
            # Define base color for each object
            base_color = pc.DEFAULT_PLOTLY_COLORS[int(mask_info['label2id'][label]) % len(pc.DEFAULT_PLOTLY_COLORS)]

            # Normalize timestamps to [0, 1] range
            norm_time = (v['time'] - timestamp0) / duration

            # Create color array that transitions from light to dark
            colors = [
                f'rgba({base_color[4:-1]}, {0.3 + 0.7 * t})'
                for t in norm_time
            ]
            legend_color = f'rgba({base_color[4:-1]}, 1.0)'

            plot_line = go.Scatter3d(
                x=v['z'],
                y=v['x'],
                z=v['y'],
                name=label,
                mode='lines',
                line=dict(
                    width=4,
                    color=colors,
                ),
                hoverinfo='name+text',
                text=[f'Time: {t:.2f}' for t in v['time']],
                marker=dict(
                    color=legend_color,
                    size=1
                ),
            )
            fig.add_trace(plot_line)

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
                xaxis=_axis_dict('z (m)'),
                yaxis=_axis_dict('x (m)'),
                zaxis=_axis_dict('y (m)'),
                aspectmode='data',
            ),
        )
        fig.update_scenes(xaxis_autorange="reversed")
        fig.layout.scene.camera.projection.type = "orthographic"
    elif mode == 'position_2d':
        for label, v in dfs.items():
            # remove rows having NaN
            v = v.dropna()

            # define color
            rgb = COLORS[colors_indices[int(mask_info['label2id'][label])]][
                'color']
            # plot
            x_data = v['x']
            y_data = v['z']
            plot_scatter = go.Scatter(
                x=x_data,
                y=y_data,
                name=label,
                mode='markers',
                marker=dict(
                    size=5,
                    color=f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'
                )
            )
            fig.add_trace(plot_scatter)

            # fit a circle to the points
            points_2d = np.array([x_data, y_data]).T
            (cx, cy, r), inliers = ransac_circle_fit(
                points_2d,
                n_sample=int(len(x_data) * 0.3),
                max_iterations=10000,
                threshold=2e-1,
                min_inliers=0.6
            )
            # get the fitting score
            distances = distance_to_circle(points_2d, np.array([cx, cy]), r)
            inliner_distances = distances[distances < 1]
            avg_error = np.mean(inliner_distances)
            print(f'Average error: {avg_error}')
            # save the distances
            saved_data = {
                'points_2d': points_2d,
                'cx': cx,
                'cy': cy,
                'r': r,
                'error_distances': distances,
                'average_error': avg_error,
            }
            with open(os.path.join(result_dir, 'calibration_result.pickle'), 'wb') as f:
                pickle.dump(saved_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=cx - r, y0=cy - r,
                x1=cx + r, y1=cy + r,
                line_color="LightSeaGreen",
            )

        fig.update_layout(
            font_family='Arial',
            font_size=14,
            xaxis_title='x (m)',
            yaxis_title='Depth (m)',
        )
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1
        )
    elif mode == 'motion_2d':
        for label, v in dfs.items():
            # remove rows having NaN
            v = v.dropna()

            # define color
            rgb = COLORS[colors_indices[int(mask_info['label2id'][label])]]

            # Get 2D positions
            x_data = v['x']
            y_data = v['z']
            points_2d = np.array([x_data, y_data]).T

            # set the ground truth
            ground_truth = {
                'distance': ref_distance,
                'radius': ref_radius,
            }
            expected_center = np.array([0, ground_truth['distance']])

            # ransac circle fit
            (cx, cy, r), inliers = ransac_circle_fit(
                points_2d,
                n_sample=int(len(x_data) * 0.3),
                max_iterations=10000,
                threshold=2e-1,
                min_inliers=0.6
            )
            measured_center = np.array([cx, cy])
            measured_radius = r

            # define valid_point_mask by the distance from the measured center
            threshold = 10 # m
            dists = np.linalg.norm(points_2d - measured_center, axis=1)
            valid_point_mask = dists < threshold

            # Calculate errors with optimized position
            temporal_position_errors, expected_positions = calculate_trajectory_errors(
                points_2d, cx, cy, r,
                timestamps=np.array(v['time']),
                n_revolutions=n_revolutions,
            )

            # Plot the fitting circle
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=cx - r, y0=cy - r,
                x1=cx + r, y1=cy + r,
                line_color="LightSeaGreen",
            )

            # Plot lines connecting real and expected positions
            for i, (real, expected) in enumerate(zip(points_2d, expected_positions)):
                if i % 5 == 0:
                    fig.add_trace(go.Scatter(
                        x=[real[0], expected[0]],
                        y=[real[1], expected[1]],
                        mode='lines',
                        line=dict(color='gray', width=1),
                        showlegend=False
                    ))

            # show the expected center
            fig.add_trace(go.Scatter(
                x=[0],
                y=[ground_truth['distance']],
                mode='markers',
                marker=dict(color='black', size=10, symbol='star'),
                showlegend=False
            ))
            # show the real center
            fig.add_trace(go.Scatter(
                x=[cx],
                y=[cy],
                mode='markers',
                marker=dict(color='red', size=10, symbol='star'),
                showlegend=False
            ))

            # Plot non-outlier points
            fig.add_trace(go.Scatter(
                x=x_data[valid_point_mask],
                y=y_data[valid_point_mask],
                name=f"{label} (valid)",
                mode='markers',
                marker=dict(
                    size=5,
                    color=f"rgb({rgb['color'][0]}, {rgb['color'][1]}, {rgb['color'][2]})"
                )
            ))

            # Plot outliers if any exist
            if valid_point_mask.sum() < points_2d.shape[0]:
                fig.add_trace(go.Scatter(
                    x=x_data[~valid_point_mask],
                    y=y_data[~valid_point_mask],
                    name=f"{label} (outliers)",
                    mode='markers',
                    marker=dict(
                        size=9,
                        color='blue',
                        symbol='x'
                    )
                ))

            # Plot ground truth circle with optimized position
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = 0 + ground_truth['radius'] * np.cos(theta)
            circle_y = ground_truth['distance'] + ground_truth['radius'] * np.sin(theta)

            fig.add_trace(go.Scatter(
                x=circle_x,
                y=circle_y,
                name='Ground Truth',
                mode='lines',
                line=dict(
                    color='black',
                    dash='dash'
                )
            ))

            logger.info(f'Total points: {points_2d.shape[0]}')
            logger.info(f'Valid points: {valid_point_mask.sum()}')
            fitted_center = np.array([cx, cy])
            diff_distance = np.linalg.norm(
                fitted_center - expected_center)
            logger.info(
                f'Distance error: {diff_distance:.3f} m')
            diff_radius = measured_radius - ground_truth['radius']
            logger.info(
                f'Radius error: {diff_radius:.3f} m')
            avg_position_error = np.mean(
                distance_to_circle(points_2d[valid_point_mask], np.array([cx, cy]), measured_radius)
            )
            logger.info(
                f'Average position error: {avg_position_error:.3f} m')
            avg_temporal_error = np.mean(
                temporal_position_errors[valid_point_mask])
            logger.info(
                f'Average temporal-position error: {avg_temporal_error:.3f} m')

            # Save detailed results
            saved_data = {
                'points_2d': points_2d,
                'ground_truth': ground_truth,
                'fitted_circle': [cx, cy, measured_radius],
                'valid_point_mask': valid_point_mask,
                'temporal_position_errors': temporal_position_errors,
                'expected_positions': expected_positions,
            }
            with open(os.path.join(result_dir, 'trajectory_validation_result.pickle'), 'wb') as f:
                pickle.dump(saved_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        fig.update_layout(
            font_family='Arial',
            font_size=14,
            xaxis_title='x (m)',
            yaxis_title='Depth (m)',
            showlegend=False,
            xaxis=dict(
                range=[expected_center[0]-5, expected_center[0]+5]
            ),
            yaxis=dict(
                range=[expected_center[1]-5, expected_center[1]+5]
            ),
            margin=dict(l=60, r=20, t=20, b=60)
        )
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1
        )

    fig.write_image(os.path.join(result_dir, "plot_validate_trajectory.png"))
    fig.write_html(os.path.join(result_dir, "plot_validate_trajectory.html"))
    fig.write_image(os.path.join(result_dir, "plot_validate_trajectory.pdf"))
    fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='motion_2d',
                        choices=['position_2d', 'position_3d', 'motion_2d'], help='Mode to run the script')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to data directory')
    parser.add_argument('--ref_distance', type=float, default=1.0, help='Reference distance for validation (m)')
    parser.add_argument('--ref_radius', type=float, default=1.0, help='Reference radius for validation (m)')
    parser.add_argument('--n_revolutions', type=int, default=2, help='Number of revolutions')
    args = parser.parse_args()
    main(args.mode, args.data_dir, args.ref_distance, args.ref_radius, args.n_revolutions)