import os
import numpy as np
import hjson
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


pio.kaleido.scope.mathjax = None

CONFIG = {
    "data_fpath": "data/calibration/validation_data_wildpose_v1.2.json",
    "result_dir": "results",
}


def main():
    # Load the data
    with open(CONFIG['data_fpath'], 'r') as f:
        dataset = hjson.loads(f.read())

    # Prepare the values for plot
    xs = []
    y_points = []
    angles = []  # New list to store angles
    for scene in dataset:
        distance = scene['distance (m)']
        angle = scene['board angle (deg)']  # Extract angle
        for data in scene['data']:
            gt = data['true lengths (m)']
            measurements = np.array(data['measured lengths (m)'])
            xs.extend([distance] * len(measurements))
            y_points.extend(np.abs(measurements - gt) * 1e3)
            # Store angle for each measurement
            angles.extend([angle] * len(measurements))

    # Create figure with secondary y-axis
    fig = make_subplots()

    # Create separate box plots for each unique angle
    unique_angles = sorted(set(angles))
    # Add more colors if needed
    colors = ['lightblue', 'lightgreen', 'lightpink']

    for i, angle in enumerate(unique_angles):
        mask = np.array(angles) == angle
        # Add box plots for this angle
        fig.add_trace(go.Box(
            x=np.array(xs)[mask],
            y=np.array(y_points)[mask],
            name=f'{angle}°',
            boxpoints='all',  # Show all points
            jitter=0,  # Add some random spread
            pointpos=0,  # Center points on the box
            line=dict(color='black', width=1),
            fillcolor=colors[i],
            whiskerwidth=0.7,
            marker=dict(
                color=colors[i].replace('light', ''),
                size=5,
                opacity=1,
                symbol='diamond'
            ),
        ))

    # Update layout
    fig.update_layout(
        legend_title='Board Angle',
        title=None,  # Remove title for scientific paper style
        xaxis_title='Distance (m)',
        yaxis_title='Absolute error (mm)',
        width=1200,
        height=800,
        font=dict(family='Arial', size=18),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=20, t=20, b=60),
        showlegend=True,
        boxmode='group',
        boxgap=0.5,
        boxgroupgap=0.07
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

    # Save the plot
    if not os.path.exists(CONFIG['result_dir']):
        os.makedirs(CONFIG['result_dir'])

    fig.write_html(os.path.join(
        CONFIG['result_dir'], "validation_plot_by_angles.html"))
    fig.write_image(os.path.join(
        CONFIG['result_dir'], "validation_plot_by_angles.pdf"))

    # Show the plot
    fig.show()


if __name__ == '__main__':
    main()
