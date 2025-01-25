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

    # Create scatter plots and line plots for each unique angle
    unique_angles = sorted(set(angles))
    symbols = ['circle', 'diamond', 'square']
    dark_colors = ['blue', 'green', 'orange']  # Darker colors for scatter
    light_colors = ['lightblue', 'lightgreen', 'orange']  # Lighter colors for lines

    for i, angle in enumerate(unique_angles):
        mask = np.array(angles) == angle
        x_values = np.array(xs)[mask]
        y_values = np.array(y_points)[mask]

        # Add scatter points with darker colors
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            name=f'{angle}° (points)',
            mode='markers',
            marker=dict(
                color=dark_colors[i],
                size=8,  # Reduced size to match circles plot
                symbol=symbols[i],
                opacity=0.8  # Matched opacity with circles plot
            ),
            showlegend=True
        ))

        # Calculate and add average line with lighter colors
        unique_x = np.unique(x_values)
        avg_y = [np.mean(y_values[x_values == x]) for x in unique_x]

        fig.add_trace(go.Scatter(
            x=unique_x,
            y=avg_y,
            name=f'{angle}° (mean)',
            mode='lines',
            line=dict(
                color=light_colors[i],
                width=2,
                dash='dot',
            ),
            showlegend=True
        ))

    # Update layout
    fig.update_layout(
        legend_title='Board Angle',
        title=None,
        xaxis_title='Distance (m)',
        yaxis_title='Absolute error (mm)',
        width=600,
        height=400,
        font=dict(family='Arial', size=18),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=20, t=20, b=60),
        showlegend=True
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
