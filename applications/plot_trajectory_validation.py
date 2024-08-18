import os
import numpy as np
import pickle
import glob

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


pio.kaleido.scope.mathjax = None

CONFIG = {
    "data_dir": "data/calibration",
    "result_dir": "results",
}
PREDEFINED_DISTANCES = [x for x in range(0, 200, 20)]


def main():
    # Get all error.npy paths under all subdirectories
    calib_result_fpaths = sorted(glob.glob(os.path.join(
        CONFIG['data_dir'], '**', 'calibration_result.pickle'), recursive=True))
    if not calib_result_fpaths:
        raise FileNotFoundError(
            "No calibration_result.pickle files found in the specified directory.")

    # Prepare the values for plot
    xs = []
    y_points = []
    for calib_result_fpath in calib_result_fpaths:
        # load the calibration result
        with open(calib_result_fpath, 'rb') as f:
            calib_result = pickle.load(f)

        error_distances = calib_result['error_distances']
        inliner_distances = error_distances[error_distances < 1]

        # Find the closest predefined distance
        cy = calib_result['cy']
        closest_distance = min(PREDEFINED_DISTANCES, key=lambda x: abs(x - cy))
        xs.extend([closest_distance] * len(inliner_distances))
        y_points.extend(inliner_distances)

    # Create figure
    fig = go.Figure()

    # Add box plots
    fig.add_trace(go.Box(
        x=xs,
        y=np.array(y_points) * 1000,
        boxpoints=False,
        line=dict(color='black', width=1),
        fillcolor='lightblue',
        whiskerwidth=0.5,
        marker=dict(
            size=1,
            opacity=0.7,
        )
    ))

    # Customize layout
    fig.update_layout(
        title=None,  # Remove title for scientific paper style
        xaxis_title='Distance (m)',
        yaxis_title='Absolute error (mm)',
        width=600,
        height=400,
        font=dict(family='Arial', size=18),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=20, t=20, b=60),
        showlegend=False,
        boxmode='group'
    )

    # Update axes
    fig.update_xaxes(
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

    # fig.update_yaxes({'range': (-5, 150), 'autorange': False})

    # Save the plot
    if not os.path.exists(CONFIG['result_dir']):
        os.makedirs(CONFIG['result_dir'])

    fig.write_html(os.path.join(
        CONFIG['result_dir'], "validation_plot.html"))
    fig.write_image(os.path.join(
        CONFIG['result_dir'], "validation_plot.pdf"))

    # Show the plot
    fig.show()


if __name__ == '__main__':
    main()
