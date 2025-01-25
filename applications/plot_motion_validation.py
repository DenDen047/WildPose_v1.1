import os
from venv import logger
import numpy as np
import pickle
import glob
import argparse

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


pio.kaleido.scope.mathjax = None

CONFIG = {
    "data_dir": "data/calibration_v1.2",
    "result_dir": "results",
}
PREDEFINED_DISTANCES = [x for x in range(0, 200, 20)]


def distance_to_circle(points, center, radius):
    # Calculate distances as a matrix operation
    diff = points - center
    distances = np.abs(np.linalg.norm(diff, axis=1) - radius)
    return distances


def main(mode, simplified=False):
    # Get all validation result paths
    result_fpaths = sorted(glob.glob(os.path.join(
        CONFIG['data_dir'], '**', 'trajectory_validation_result.pickle'), recursive=True))

    metrics = {
        'distance_error': [],
        'radius_error': [],
        'position_error': [],
        'temporal_position_error': [],
    }

    # Prepare data structures for each metric
    distances = []
    radii = []  # Store radii for grouping

    for result_fpath in result_fpaths:
        with open(result_fpath, 'rb') as f:
            result = pickle.load(f)

        valid_point_mask = result['valid_point_mask']
        points_2d = result['points_2d']
        temporal_position_errors = result['temporal_position_errors']

        cx, cy, measured_radius = result['fitted_circle']
        expected_center = np.array([0, result['ground_truth']['distance']])
        diff_distance = np.linalg.norm(
            np.array([cx, cy]) - expected_center)
        diff_radius = measured_radius - result['ground_truth']['radius']

        distances.append(result['ground_truth']['distance'])
        metrics['distance_error'].append(diff_distance)
        metrics['radius_error'].append(diff_radius)
        metrics['position_error'].append(
            distance_to_circle(
                points_2d[valid_point_mask],
                np.array([cx, cy]),
                measured_radius
            )
        )
        metrics['temporal_position_error'].append(
            temporal_position_errors[valid_point_mask])

        # Store radius for each measurement
        radii.append(result['ground_truth']['radius'])

    # change error unit to mm if simplified is True
    for key in metrics:
        metrics[key] = [x * 1000 for x in metrics[key]]
    error_unit = 'mm'

    if simplified:
        # Create simplified plot similar to trajectory validation
        fig = go.Figure()
        values = metrics[mode]

        xs = []
        ys = []
        for d, v in zip(distances, values):
            if isinstance(v, (np.ndarray, list)):
                xs.extend([d] * len(v))
                ys.extend(v)
            else:
                xs.append(d)
                ys.append(v)

        fig.add_trace(go.Box(
            x=xs,
            y=ys,
            boxpoints='all',
            quartilemethod='linear',
            line=dict(color='black', width=1),
            fillcolor='lightblue',
            whiskerwidth=0.5,
            marker=dict(
                size=1,
                opacity=0.7,
            )
        ))

        # Simplified layout
        fig.update_layout(
            width=600,
            height=400,
            font=dict(family='Arial', size=18),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=20, t=20, b=60),
            showlegend=False,
            boxmode='group'
        )
    else:
        # Define colors for different radii
        unique_radii = sorted(set(radii))
        colors = ['lightblue', 'lightgreen', 'orange', 'lightyellow', 'lightgray']

        # Add box plots for each metric
        fig = go.Figure()
        values = metrics[mode]
        for i, radius in enumerate(unique_radii):
            radius_mask = np.array(radii) == radius

            if isinstance(values[0], (np.ndarray, list)):
                xs = []
                ys = []
                for d, v, mask in zip(distances, values, radius_mask):
                    if mask:
                        xs.extend([d] * len(v))
                        ys.extend(v)
            else:
                xs = np.array(distances)[radius_mask]
                ys = np.array(values)[radius_mask]

            box_plot = go.Box(
                x=xs,
                y=ys,
                name=f'r={int(radius)}m',
                boxpoints='all',
                line=dict(color='black', width=1),
                fillcolor=colors[i % len(colors)],
                # showlegend=True,  # Show legend for all plots
                marker=dict(
                    color=colors[i % len(colors)].replace('light', ''),
                    size=3,
                    opacity=0.5,
                    symbol='diamond'
                ),
            )
            fig.add_trace(box_plot)

        fig.update_layout(
            width=700,
            height=400,
            font=dict(family='Arial', size=18),
            legend_title='Circle radius',
            boxmode='group',
            boxgap=0.2,
            boxgroupgap=0.4,
            margin=dict(l=60, r=20, t=20, b=60),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
        )

    # Update axes
    fig.update_xaxes(
        title_text='Distance (m)',
        tickmode='array',
        tickvals=[20, 40, 60, 80, 100, 120, 140, 160],
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=False,
        linecolor='black',
        linewidth=1,
        ticks='outside',
        tickwidth=1,
        tickcolor='black',
        ticklen=5
    )
    fig.update_yaxes(
        title_text=f'Absolute error ({error_unit})',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=False,
        linecolor='black',
        linewidth=1,
        ticks='outside',
        tickwidth=1,
        tickcolor='black',
        ticklen=5,
        range=[-30, 1600]  # Set y-axis range from 0 to 2.0
    )

    # Save the plot
    if not os.path.exists(CONFIG['result_dir']):
        os.makedirs(CONFIG['result_dir'])

    fig.write_html(os.path.join(CONFIG['result_dir'], f"motion_validation_plot_{mode}.html"))
    fig.write_image(os.path.join(CONFIG['result_dir'], f"motion_validation_plot_{mode}.pdf"))
    fig.show()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', type=str, default='all',
    #                   choices=['all', 'distance', 'radius', 'position', 'temporal'],
    #                   help='Type of plot to generate')
    # args = parser.parse_args()
    main('temporal_position_error', simplified=False)
