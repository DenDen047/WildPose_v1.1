import os
import numpy as np
import hjson
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


pio.kaleido.scope.mathjax = None

CONFIG = {
    "data_fpath": "data/calibration/validation_data_wildpose.json",
    "result_dir": "results",
}


def main():
    # Load the data
    with open(CONFIG['data_fpath'], 'r') as f:
        dataset = hjson.loads(f.read())

    # Prepare the values for plot
    xs = []
    y_points = []
    for scene in dataset:
        distance = scene['distance (m)']
        for data in scene['data']:
            gt = data['true lengths (m)']
            measurements = np.array(data['measured lengths (m)'])
            xs.extend([distance] * len(measurements))
            y_points.extend(np.abs(measurements - gt) * 1e3)

    # Create figure
    fig = make_subplots()

    # Add box plots
    fig.add_trace(go.Box(
        x=xs,
        y=y_points,
        name='Error Distribution',
        boxpoints=False,  # hide the scatter points from the box plot
        line=dict(color='black', width=1),
        fillcolor='lightblue',
        whiskerwidth=0.5,
    ))

    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=xs,
        y=y_points,
        mode='markers',
        name='Individual Errors',
        marker=dict(
            color='green',
            size=5,
            opacity=0.7,
            symbol='diamond'
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
