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
DATA_DIR = 'data/calibration/2024-05-26_15-28-32.795_measurement'

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
    distances = np.abs(np.sqrt(np.sum(diff**2, axis=1)) - radius)
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


def main(mode):
    lidar_dir = os.path.join(DATA_DIR, 'lidar')
    rgb_dir = os.path.join(DATA_DIR, 'sync_rgb')
    mask_dir = os.path.join(DATA_DIR, 'masks')
    calib_fpath = os.path.join(DATA_DIR, 'manual_calibration.json')
    result_dir = os.path.join(DATA_DIR, 'calibration_results')

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

    # collect the 3D positions with Segment Anything Model
    timestamp0 = get_timestamp_from_img_fpath(img_fpaths[0])
    positions_3d = {}
    for i, (img_fpath, pcd_fpath, seg_mask) in tqdm(enumerate(zip(img_fpaths, pcd_fpaths, masks)), total=n_frame):
        # if i > 200:
        #     break

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

    fig.write_image(os.path.join(result_dir, "plot_validate_trajectory.png"))
    fig.write_html(os.path.join(result_dir, "plot_validate_trajectory.html"))
    fig.write_image(os.path.join(result_dir, "plot_validate_trajectory.pdf"))
    fig.show()


if __name__ == '__main__':
    main('position_2d')
