import os
import re
import glob
import pandas as pd
import numpy as np
import scipy.ndimage
import open3d as o3d

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scienceplots
import plotly.graph_objs as go


# plt.style.use(['science', 'nature', 'no-latex'])
# figure(figsize=(10, 6))
plt.rcParams.update({
    'legend.frameon': False,
    "pdf.fonttype": 42,
})


def equal_3d_aspect(ax):
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))


def main(mode='3d'):
    # pcd_fpath = './data/giraffe_stand/textured_pcds/coloured_accumulation.pcd'
    # xlim = (-100, 100)
    # ylim = (-100, 100)
    # zlim = (70, 150)
    pcd_fpath = './data/martial_eagle_stand/textured_pcds/coloured_accumulation.pcd'
    xlim = (-100, -0.1)
    ylim = (-100, 100)
    zlim = (18.6, 18.9)

    # load data
    pcd = o3d.io.read_point_cloud(pcd_fpath)
    points = np.asarray(pcd.points)  # [N, 3]
    colors = np.asarray(pcd.colors)  # [N, 3]

    # make the mask
    if xlim is not None:
        mask = (
            (xlim[0] < points[:, 0]) & (points[:, 0] < xlim[1]) &
            (ylim[0] < points[:, 1]) & (points[:, 1] < ylim[1]) &
            (zlim[0] < points[:, 2]) & (points[:, 2] < zlim[1])
        )
        masked_points = points[mask]
        masked_colors = colors[mask]
    else:
        masked_points = points
        masked_colors = colors

    def _axis_dict(title, range=None):
        return dict(
            title=title,
            ticks='outside',
            tickangle=0,
            backgroundcolor='rgb(230, 230, 230)',
            tickformat='.1f',
            range=None
        )

    if mode == '3d':
        # plot the 3D coloured point cloud
        fig = go.Figure()
        point_cloud_scatter = go.Scatter3d(
            x=masked_points[:, 2],
            y=masked_points[:, 0],
            z=masked_points[:, 1],
            mode='markers',
            marker=dict(
                size=2,
                color=masked_colors if len(masked_colors) > 0 else None
            )
        )
        fig.add_trace(point_cloud_scatter)

        fig.update_layout(
            font_family='Arial',
            font_size=14,
            scene=dict(
                xaxis=_axis_dict('Depth (m)'),
                yaxis=_axis_dict('x (m)'),
                zaxis=_axis_dict('y (m)'),
                aspectmode='data',
            ),
        )
        fig.update_scenes(
            xaxis_autorange="reversed",
            zaxis_autorange="reversed",
        )
        fig.layout.scene.camera.projection.type = "orthographic"
    elif mode == 'XY':
        if len(masked_colors) > 0:
            # [N, 3] -> [N, 3] with "rgb(r, g, b)" format
            masked_colors = (masked_colors * 255).astype(int)
            color = [f"rgb({c[0]}, {c[1]}, {c[2]})" for c in masked_colors]
        else:
            color = None

        # plot the 2D image in XY plane
        fig = go.Figure()
        point_cloud_scatter = go.Scatter(
            x=masked_points[:, 0],
            y=masked_points[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=color
            ),
        )
        fig.add_trace(point_cloud_scatter)

        fig.update_xaxes(
            title='x (m)',
            scaleanchor='y',
            scaleratio=1,
            ticks='outside',
            tickformat='.1f',
        )
        fig.update_yaxes(
            title='y (m)',
            ticks='outside',
            tickformat='.1f',
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            font_family='Arial',
            font_size=20,
            scene=dict(
                xaxis=_axis_dict('x (m)'),
                yaxis=_axis_dict('y (m)')
            ),
            plot_bgcolor='rgb(230, 230, 230)',
        )
    elif mode == 'ZY':
        if len(masked_colors) > 0:
            # [N, 3] -> [N, 3] with "rgb(r, g, b)" format
            masked_colors = (masked_colors * 255).astype(int)
            color = [f"rgb({c[0]}, {c[1]}, {c[2]})" for c in masked_colors]
        else:
            color = None

        # plot the 2D image in XY plane
        fig = go.Figure()
        point_cloud_scatter = go.Scatter(
            x=masked_points[:, 2],
            y=masked_points[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=color
            ),
        )
        fig.add_trace(point_cloud_scatter)

        fig.update_xaxes(
            title='Depth (m)',
            scaleanchor='y',
            scaleratio=1,
            ticks='outside',
            tickformat='.1f',
        )
        fig.update_yaxes(
            title='y (m)',
            ticks='outside',
            tickformat='.1f',
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            font_family='Arial',
            font_size=20,
            scene=dict(
                xaxis=_axis_dict('x (m)'),
                yaxis=_axis_dict('y (m)')
            ),
            plot_bgcolor='rgb(230, 230, 230)',
        )

    fig.show()


if __name__ == '__main__':
    main(mode='3d')
