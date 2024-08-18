import os
import re
import glob
import pandas as pd
import numpy as np
import json
import cv2
from tqdm import tqdm
import scipy.ndimage
import open3d as o3d

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
from projection_functions import extract_rgb_from_image
from config import COLORS, colors_indices

from projection_functions import closest_point


IMG_WIDTH, IMG_HEIGHT = 1280, 720


def mean_ignore_nan(arr):
    arr = arr.flatten()
    return np.sum(arr[~np.isnan(arr)]) / np.sum(~np.isnan(arr))


def get_timestamp_from_pcd_fpath(fpath: str) -> float:
    fname = os.path.splitext(os.path.basename(fpath))[0]
    # get timestamp
    fname = fname.split('_')
    msg_id = '_'.join(fname[:-2])
    timestamp = float(fname[-2] + '.' + fname[-1])

    return timestamp


def median_filter_3d_positions(dfs, filter_size=3):
    filtered_dfs = {}
    for key, df in dfs.items():
        filtered_df = df.copy()
        filtered_df['x'] = scipy.ndimage.median_filter(
            df['x'], size=filter_size)
        filtered_df['y'] = scipy.ndimage.median_filter(
            df['y'], size=filter_size)
        filtered_df['z'] = scipy.ndimage.median_filter(
            df['z'], size=filter_size)
        filtered_dfs[key] = filtered_df
    return filtered_dfs


def get_2D_gt(gt_json_path):
    gt_json = json.load(open(gt_json_path, 'r'))
    img_dict = {}

    annotations = gt_json['annotations']
    images = gt_json['images']

    for anno in annotations:
        bbox = anno['bbox']
        img_id = anno['image_id']
        obj_id = anno['category_id']
        img_filename = os.path.basename(images[img_id]['file_name'])

        if img_filename not in img_dict:
            img_dict[img_filename] = []
        img_dict[img_filename].append([bbox,obj_id])
    return img_dict


def erode_mask(mask, kernel_size=(5,5), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
    return eroded_mask


def calculate_precision(positions_3d):
    precision_results = {}
    all_deviations = []

    for id, positions in positions_3d.items():
        if len(positions) > 1:
            # Extracting just the x, y, z coordinates
            coordinates = np.array([
                [pos['x'], pos['z']]
                for i, pos in positions.iterrows()
            ])

            # Calculate mean position
            mean_position = np.mean(coordinates, axis=0)

            # Calculate deviations from the mean
            deviations = np.linalg.norm(coordinates - mean_position, axis=1)
            all_deviations += deviations.tolist()

            # Calculate standard deviation (precision)
            precision = np.std(deviations)

            precision_results[id] = precision

    precision_result_total = np.std(all_deviations)

    return precision_results, precision_result_total


def main(mode):
    data_dir = 'data/springbok_herd/'
    lidar_dir = os.path.join(data_dir, 'lidar')
    rgb_dir = os.path.join(data_dir, 'sync_rgb')
    mask_dir = os.path.join(data_dir, 'masks2')
    calib_fpath = os.path.join(data_dir, 'manual_calibration.json')

    # load data
    img_fpaths = sorted(glob.glob(os.path.join(rgb_dir, '*.jpeg')))
    pcd_fpaths = sorted(glob.glob(os.path.join(lidar_dir, '*.pcd')))
    assert len(img_fpaths) == len(pcd_fpaths)
    n_frame = len(img_fpaths)
    mask_fpaths = [os.path.join(mask_dir, '{0}.npy'.format(i)) for i in range(n_frame)]
    mask_id_fpaths = [os.path.join(mask_dir, '{0}_obj_ids.npy'.format(i)) for i in range(n_frame)]
    fx, fy, cx, cy, rot_mat, translation = load_camera_parameters(calib_fpath)
    intrinsic = make_intrinsic_mat(fx, fy, cx, cy)
    extrinsic = make_extrinsic_mat(rot_mat, translation)

    # collect the 3D positions with Segment Anything Model
    timestamp0 = get_timestamp_from_img_fpath(img_fpaths[0])
    accumulated_pcd_in_cam = None
    positions_3d = {}
    for img_fpath, pcd_fpath, mask_fpath, mask_id_fpath in tqdm(zip(img_fpaths, pcd_fpaths, mask_fpaths, mask_id_fpaths)):
        # load the frame
        rgb_img = load_rgb_img(img_fpath)
        pcd_open3d = load_pcd(pcd_fpath, mode='open3d')
        pts_in_lidar = np.asarray(pcd_open3d.points)
        seg_mask = np.load(mask_fpath)  # [n_id, 1, H, W]
        obj_ids = np.load(mask_id_fpath)
        timestamp = get_timestamp_from_img_fpath(img_fpath)
        img_key = os.path.basename(img_fpath).replace('.jpeg', '_3.jpeg')

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
        assert n_id == len(obj_ids)
        for idx, obj_id in enumerate(obj_ids):
            mask = seg_mask[idx, 0]
            mask_ys, mask_xs = np.where(mask)
            target_2d_pt = np.array([
                np.median(mask_xs),
                np.median(mask_ys),
            ])
            _, pt_idx = closest_point(target_2d_pt, pcd_in_img[:, :2])
            pt3d = pcd_in_cam[pt_idx, :]
            if obj_id not in positions_3d.keys():
                positions_3d[obj_id] = []
            positions_3d[obj_id].append([timestamp] + pt3d.tolist())

        # accumulate the pcd
        if accumulated_pcd_in_cam is None:
            accumulated_pcd_in_cam = pcd_in_cam
        else:
            accumulated_pcd_in_cam = np.vstack((accumulated_pcd_in_cam, pcd_in_cam))

    # find the global plane
    accumulated_pcd = o3d.geometry.PointCloud()
    accumulated_pcd.points = o3d.utility.Vector3dVector(accumulated_pcd_in_cam[:, :3])

    # # estimate the global ground plane
    # print('generating the ground...')
    # global_plane_model, _ = accumulated_pcd.segment_plane(
    #     distance_threshold=0.01,
    #     ransac_n=3,
    #     num_iterations=1000
    # )   # Plane model: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0
    # print('Done')

    # array to dataframe
    dfs = {}
    for key in positions_3d.keys():
        dfs[key] = pd.DataFrame(
            positions_3d[key],
            columns =['time', 'x', 'y', 'z']
        )

    # filter the positions
    dfs = median_filter_3d_positions(dfs, filter_size=5)

    # calculate the precision with stationary individuals
    # ID 4&8
    # Frame: 39--99
    stationary_ids ={
        4: dfs[4][39:99+1],
        8: dfs[8][39:99+1],
    }
    precisions, total_precision = calculate_precision(stationary_ids)
    print(precisions)
    print(f'total precision: {total_precision}')
    # show the xyz precisions of the stationary individuals
    for k, v in stationary_ids.items():
        # calculate the mean
        mean_x = np.mean(v['x'])
        mean_y = np.mean(v['y'])
        mean_z = np.mean(v['z'])
        print(f'{k}: {mean_x}, {mean_y}, {mean_z}')
        # calculate the standard deviation
        std_x = np.std(v['x'])
        std_y = np.std(v['y'])
        std_z = np.std(v['z'])
        print(f'{k}: {std_x}, {std_y}, {std_z}')

    fig = go.Figure()
    if mode == 'position_3d':
        # plot the data
        duration = get_timestamp_from_img_fpath(img_fpaths[-1]) - timestamp0
        for k, v in dfs.items():
            # Define base color for each object
            base_color = pc.DEFAULT_PLOTLY_COLORS[int(k) % len(pc.DEFAULT_PLOTLY_COLORS)]

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
                name=str(k),
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
                aspectratio=dict(x=1, y=9/30 * 3, z=0.8/30 * 3),
                annotations=[
                    dict(
                        x=np.mean(dfs[k]['z']),
                        y=np.mean(dfs[k]['x']),
                        z=np.mean(dfs[k]['y']),
                        text=f'{k}',
                        showarrow=False,
                        font=dict(size=16),
                    )
                    for k in dfs.keys()
                ]
            ),
        )
        fig.update_scenes(xaxis_autorange="reversed")
        # fig.layout.scene.camera.projection.type = "orthographic"
    elif mode == 'position_without_y':
        # collect the 3D positions with Segment Anything Model
        timestamp0 = get_timestamp_from_img_fpath(img_fpaths[0])
        positions_3d = {}
        for img_fpath, pcd_fpath, mask_fpath, mask_id_fpath in tqdm(zip(img_fpaths, pcd_fpaths, mask_fpaths, mask_id_fpaths)):
            # load the frame
            pcd_open3d = load_pcd(pcd_fpath, mode='open3d')
            pts_in_lidar = np.asarray(pcd_open3d.points)
            seg_mask = np.load(mask_fpath)  # [n_id, 1, H, W]
            obj_ids = np.load(mask_id_fpath)
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
            assert n_id == len(obj_ids)
            for idx, obj_id in enumerate(obj_ids):
                mask = seg_mask[idx, 0]
                mask_ys, mask_xs = np.where(mask)
                target_2d_pt = np.array([
                    np.median(mask_xs),
                    np.median(mask_ys),
                ])
                _, pt_idx = closest_point(target_2d_pt, pcd_in_img[:, :2])
                pt3d = pcd_in_cam[pt_idx, :]
                if obj_id not in positions_3d.keys():
                    positions_3d[obj_id] = []
                positions_3d[obj_id].append([timestamp] + pt3d.tolist())

        # array to dataframe
        dfs = {}
        for key in positions_3d.keys():
            dfs[key] = pd.DataFrame(
                positions_3d[key],
                columns=['time', 'x', 'y', 'z']
            )

        # filter the positions
        dfs = median_filter_3d_positions(dfs, filter_size=5)

        # calculate the precision with stationary individuals
        # ID 4&8
        # Frame: 39--99
        stationary_ids = {
            4: dfs[4][39:99 + 1],
            8: dfs[8][39:99 + 1],
        }
        precisions, total_precision = calculate_precision(stationary_ids)
        print(precisions)
        print(f'total precision: {total_precision}')
        # show the xyz precisions of the stationary individuals
        for k, v in stationary_ids.items():
            # calculate the mean
            mean_x = np.mean(v['x'])
            mean_y = np.mean(v['y'])
            mean_z = np.mean(v['z'])
            print(f'{k}: {mean_x}, {mean_y}, {mean_z}')
            # calculate the standard deviation
            std_x = np.std(v['x'])
            std_y = np.std(v['y'])
            std_z = np.std(v['z'])
            print(f'{k}: {std_x}, {std_y}, {std_z}')

        # plot the data
        fig = go.Figure()
        for k, v in dfs.items():
            # define color
            rgb = COLORS[colors_indices[int(k)]]['color']
            # plot
            plot_line = go.Scatter3d(
                x=v['time'] - timestamp0,
                y=v['x'],
                z=v['z'],
                name=str(k),
                mode='lines',
                line=dict(
                    width=6 if k in [6, 7] else 3,
                    color=f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'
                )
            )
            fig.add_trace(plot_line)

        # # add planes
        # xmin = np.min([np.min(v.x) for k, v in dfs.items()])
        # xmax = np.max([np.max(v.x) for k, v in dfs.items()])
        # ymin = np.min([np.min(v.z) for k, v in dfs.items()])
        # ymax = np.max([np.max(v.z) for k, v in dfs.items()])
        # for h in [0, 14]:
        #     plane = go.Mesh3d(
        #         x=[xmin, xmax, xmin, xmax],
        #         y=[ymin, ymin, ymax, ymax],
        #         z=[h] * 4,
        #         color='rgb(194, 158, 249)',
        #         # colorscale=[[x, 'rgb(194, 158, 249)'] for x in [0, 1]],
        #         opacity=0.3,
        #         showscale=False
        #     )
        #     fig.add_trace(plane)

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
            font_size=12,
            scene=dict(
                xaxis=dict(
                    **_axis_dict('Time (s)'),
                    autorange='reversed'
                ),
                yaxis=_axis_dict('x (m)'),
                zaxis=_axis_dict('z (m)'),
                aspectratio=dict(x=2, y=1, z=1),
            ),
        )
        fig.layout.scene.camera.projection.type = "orthographic"
    elif mode == 'velocity':
        for k, v in dfs.items():
            # Calculate velocity
            # velocisty = np.sqrt(
            #     np.diff(v['x'])**2 +
            #     np.diff(v['y'])**2 +
            #     np.diff(v['z'])**2
            # ) / np.diff(v['time'])
            velocisty = np.diff(v['z']) / np.diff(v['time'])

            # Add a final velocity point (assuming same as the last calculated velocity)
            velocisty = np.append(velocisty, velocisty[-1])

            # Moving average
            moving_average = np.convolve(velocisty, np.ones(5)/5, mode='valid')

            # Get color from COLORS
            rgb = COLORS[colors_indices[k]]['color']

            # Create the velocity plot
            fig.add_trace(go.Scatter(
                x=v['time'] - timestamp0,  # Adjust time to start from 0
                y=moving_average,
                mode='lines',
                name=f'{k}',
                line=dict(
                    width=2,
                    color=f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'
                )
            ))

        fig.update_layout(
            xaxis_title='Time (s)',
            yaxis_title='Velocity along z-axis (m/s)',
            font=dict(family="Arial", size=14),
            legend_title='Individuals',
            hovermode='x unified'
        )
    elif mode == 'neighbor_density':
        # Calculate average distances between individuals
        def calculate_average_distances(all_positions):
            ids = sorted(list(all_positions.keys()))
            n_individuals = len(ids)
            distance_matrix = np.zeros((n_individuals, n_individuals))

            for i, id1 in enumerate(ids):
                for j, id2 in enumerate(ids):
                    if i == j:
                        continue

                    positions1 = all_positions[id1]
                    positions2 = all_positions[id2]

                    # Merge on 'time' column
                    merged = pd.merge(positions1, positions2,
                                    on='time', suffixes=('_1', '_2'))

                    # Calculate distances
                    distances = np.sqrt(
                        (merged['x_1'] - merged['x_2'])**2 +
                        (merged['y_1'] - merged['y_2'])**2 +
                        (merged['z_1'] - merged['z_2'])**2
                    )

                    distance_matrix[i, j] = np.mean(distances)

            return distance_matrix, ids

        # Calculate average distances
        distance_matrix, ids = calculate_average_distances(dfs)

        # Create the heatmap with updated style
        fig = go.Figure(data=go.Heatmap(
            z=distance_matrix,
            x=ids,
            y=ids,
            colorscale='Reds_r',
            colorbar=dict(
                title='Average distance (m)',
                titleside='right',
                titlefont=dict(size=14),
                tickfont=dict(size=12),
            ),
            hoverinfo='x+y+z',
            zmin=np.nanmin(distance_matrix),
            zmax=np.nanmax(distance_matrix),
            showscale=True
        ))

        fig.update_layout(
            xaxis=dict(
                title='Individual ID',
                tickmode='array',
                tickvals=ids,
                ticktext=ids,
                side='top',
                tickfont=dict(size=12),
                titlefont=dict(size=14)
            ),
            yaxis=dict(
                title='Individual ID',
                tickmode='array',
                tickvals=ids,
                ticktext=ids,
                tickfont=dict(size=12),
                titlefont=dict(size=14),
                autorange="reversed"
            ),
            width=700,
            height=700,
            margin=dict(l=80, r=80, t=100, b=80),
        )

        # Make the plot square
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

        # Add text annotations
        mean_distance = mean_ignore_nan(distance_matrix)
        for i in range(len(ids)):
            for j in range(len(ids)):
                fig.add_annotation(
                    x=ids[j],
                    y=ids[i],
                    text=f"{distance_matrix[i, j]:.2f}" if not np.isnan(
                        distance_matrix[i, j]) else "N/A",
                    showarrow=False,
                    font=dict(
                        color="white" if distance_matrix[i, j] < mean_distance else "black",
                        size=10
                    )
                )

        # Update overall font and remove gridlines
        fig.update_layout(
            font=dict(family="Arial", size=14),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
    elif mode == 'neighbor_density_animation':
        # Function to calculate distances at a specific time
        def calculate_distances_at_time(positions, time):
            ids = sorted(list(positions.keys()))
            n_individuals = len(ids)
            distance_matrix = np.full((n_individuals, n_individuals), np.nan)

            for i, id1 in enumerate(ids):
                for j, id2 in enumerate(ids):
                    if i == j:
                        continue

                    pos1 = positions[id1][positions[id1]['time'] == time]
                    pos2 = positions[id2][positions[id2]['time'] == time]

                    if pos1.empty or pos2.empty:
                        continue

                    distance = np.sqrt(
                        (pos1['x'].iloc[0] - pos2['x'].iloc[0])**2 +
                        (pos1['y'].iloc[0] - pos2['y'].iloc[0])**2 +
                        (pos1['z'].iloc[0] - pos2['z'].iloc[0])**2
                    )

                    distance_matrix[i, j] = distance

            return distance_matrix, ids

        # At the beginning of the 'neighbor_density_animation' mode:
        timestamps = sorted(set(t for df in dfs.values() for t in df['time']))
        start_time = timestamps[0]

        # Calculate the overall min and max distances
        all_distances = []
        for timestamp in timestamps:
            matrix, _ = calculate_distances_at_time(dfs, timestamp)
            all_distances.extend(matrix[~np.isnan(matrix)].flatten())

        overall_min = np.min(all_distances)
        overall_max = np.max(all_distances)

        # Create initial heatmap
        initial_matrix, ids = calculate_distances_at_time(dfs, timestamps[0])
        heatmap = go.Heatmap(
            z=initial_matrix,
            x=ids,
            y=ids,
            colorscale='Reds_r',
            colorbar=dict(
                title='Distance (m)',
                titleside='right',
                titlefont=dict(size=14),
                tickfont=dict(size=12),
            ),
            hoverinfo='x+y+z',
            zmin=overall_min,
            zmax=overall_max
        )
        fig.add_trace(heatmap)

        # Update layout
        fig.update_layout(
            xaxis=dict(
                title='Individual ID',
                tickmode='array',
                tickvals=ids,
                ticktext=ids,
                side='top',
                tickfont=dict(size=12),
                titlefont=dict(size=14)
            ),
            yaxis=dict(
                title='Individual ID',
                tickmode='array',
                tickvals=ids,
                ticktext=ids,
                tickfont=dict(size=12),
                titlefont=dict(size=14),
                autorange="reversed"
            ),
            width=700,
            height=700,
            margin=dict(l=80, r=80, t=100, b=80),
        )

        # Make the plot square
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # Create frames for animation
        frames = []
        for timestamp in timestamps:
            matrix, _ = calculate_distances_at_time(dfs, timestamp)
            frame = go.Frame(
                data=[go.Heatmap(
                    z=matrix,
                    zmin=overall_min,
                    zmax=overall_max
                )],
                name=f't{timestamp - start_time:.2f}'
            )
            frames.append(frame)

        fig.frames = frames

        # Update slider
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(label='Play',
                             method='animate',
                             args=[None, {'frame': {'duration': 10, 'redraw': True},
                                          'fromcurrent': True,
                                          'transition': {'duration': 0}}]),
                        dict(label='Pause',
                             method='animate',
                             args=[[None], {'frame': {'duration': 0, 'redraw': True},
                                            'mode': 'immediate',
                                            'transition': {'duration': 0}}])
                    ]
                )
            ],
            sliders=[{
                'currentvalue': {'prefix': 'Time: ', 'suffix': ' s'},
                'steps': [{'args': [[f't{t - start_time:.2f}'],
                                    {'frame': {'duration': 0, 'redraw': True},
                                    'mode': 'immediate',
                                     'transition': {'duration': 0}}],
                           'label': f'{t - start_time:.2f}',
                           'method': 'animate'} for t in timestamps]
            }]
        )

    fig.show()
    fig.write_image(os.path.join('results', "plot_3d_trajectory.png"))
    fig.write_html(os.path.join('results', "plot_3d_trajectory.html"))
    fig.write_image(os.path.join('results', "plot_3d_trajectory.pdf"))


if __name__ == '__main__':
    main('position_3d')
