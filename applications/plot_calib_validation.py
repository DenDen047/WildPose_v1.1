import os
import numpy as np
import hjson
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from loguru import logger


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
    error_ratios = []
    for scene in dataset:
        distance = scene['distance (m)']
        for data in scene['data']:
            gt = data['true lengths (m)']
            measurements = np.array(data['measured lengths (m)'])
            xs.extend([distance] * len(measurements))
            y_points.extend(np.abs(measurements - gt) * 1e3)
            error_ratios.extend(measurements / gt)

    # get the standard deviation of the error
    for distance in np.unique(xs):
        mask = np.array(xs) == distance
        std_dev = np.std(np.array(y_points)[mask])
        logger.info(f"SD at distance {distance:.1f} m: {std_dev:.2f} mm")

    # get the mean of the error ratio for each distance
    for distance in np.unique(xs):
        mask = np.array(xs) == distance
        mean_error_ratio = np.mean(np.array(error_ratios)[mask])
        logger.info(f"Mean error ratio at distance {distance:.1f} m: {mean_error_ratio:.3f}")

    # Calculate mean error ratios per distance
    unique_distances = np.unique(xs)
    mean_ratios = []
    for distance in unique_distances:
        mask = np.array(xs) == distance
        mean_ratio = np.mean(np.array(error_ratios)[mask])
        mean_ratios.append(mean_ratio)

    # === Error ratio plot ===

    # Create figure for error ratios
    fig_ratio = go.Figure()

    # Add horizontal reference line at y=1.0
    fig_ratio.add_trace(go.Scatter(
        x=[0, max(unique_distances)+10],
        y=[1.0, 1.0],
        mode='lines',
        line=dict(color='lightgray', dash='dot'),
        showlegend=False
    ))

    # Add scatter plot for mean error ratios
    fig_ratio.add_trace(go.Scatter(
        x=unique_distances,
        y=mean_ratios,
        mode='markers',
        marker=dict(
            color='black',
            size=8,
            symbol='circle-open',
            line=dict(width=1)
        ),
        showlegend=False
    ))

    # Customize layout
    fig_ratio.update_layout(
        xaxis_title='Distance (m)',
        yaxis_title='mean measured / true length',
        width=600,
        height=400,
        font=dict(family='Arial', size=18),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=20, t=20, b=60),
        yaxis=dict(range=[0.94, 1.06])  # Match the y-axis range from the image
    )

    # Update axes
    fig_ratio.update_xaxes(
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
    )
    fig_ratio.update_yaxes(
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

    # Save the ratio plot
    fig_ratio.write_html(os.path.join(
        CONFIG['result_dir'], "validation_ratio_plot.html"))
    fig_ratio.write_image(os.path.join(
        CONFIG['result_dir'], "validation_ratio_plot.pdf"))

    # Show the ratio plot
    fig_ratio.show()

    #  === Fig. 2A ===
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
