import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg

# Define the root directory where all measurement folders are located
ROOT_DIR = '/Users/ikuta/Documents/Projects/PhD/WildPose_v1.1/data/calibration_v1.2'
RESULT_DIR = 'results'

# Define the distances and radii
distances = [20, 40, 60, 80, 100, 120, 140, 160]  # cm
radii = [1, 2, 3]  # m

# Define the explicit mapping of folders to distance and radius
folder_dist_rad = [
    [f"{ROOT_DIR}/2024-05-26_14-31-30.321_measurement", 20, 1],
    [f"{ROOT_DIR}/2024-05-26_14-32-52.247_measurement", 20, 2],
    [f"{ROOT_DIR}/2024-05-26_14-36-31.564_measurement", 40, 1],
    [f"{ROOT_DIR}/2024-05-26_14-37-55.275_measurement", 40, 2],
    [f"{ROOT_DIR}/2024-05-26_14-42-08.824_measurement", 60, 1],
    [f"{ROOT_DIR}/2024-05-26_14-43-38.205_measurement", 60, 2],
    [f"{ROOT_DIR}/2024-05-26_14-45-17.671_measurement", 60, 3],
    [f"{ROOT_DIR}/2024-05-26_14-48-08.768_measurement", 80, 1],
    [f"{ROOT_DIR}/2024-05-26_14-50-22.893_measurement", 80, 2],
    [f"{ROOT_DIR}/2024-05-26_14-52-18.990_measurement", 80, 3],
    [f"{ROOT_DIR}/2024-05-26_14-58-39.732_measurement", 100, 1],
    [f"{ROOT_DIR}/2024-05-26_15-00-13.424_measurement", 100, 2],
    [f"{ROOT_DIR}/2024-05-26_15-02-59.674_measurement", 100, 3],
    [f"{ROOT_DIR}/2024-05-26_15-06-12.368_measurement", 120, 1],
    [f"{ROOT_DIR}/2024-05-26_15-07-44.596_measurement", 120, 2],
    [f"{ROOT_DIR}/2024-05-26_15-09-34.972_measurement", 120, 3],
    [f"{ROOT_DIR}/2024-05-26_15-15-30.672_measurement", 140, 1],
    [f"{ROOT_DIR}/2024-05-26_15-16-40.500_measurement", 140, 2],
    [f"{ROOT_DIR}/2024-05-26_15-18-23.431_measurement", 140, 3],
    [f"{ROOT_DIR}/2024-05-26_15-23-42.342_measurement", 160, 1],
    [f"{ROOT_DIR}/2024-05-26_15-25-53.645_measurement", 160, 2],
    [f"{ROOT_DIR}/2024-05-26_15-28-32.795_measurement", 160, 3],
]

if __name__ == '__main__':
    # Create figure with GridSpec
    fig = plt.figure(figsize=(15, 24))  # Adjust size as needed
    gs = GridSpec(len(distances), len(radii), figure=fig)
    gs.update(wspace=0.1, hspace=0.1)  # Adjust spacing between plots

    # Create a mapping of (distance, radius) to measurement folder
    measurements = {}
    for folder, dist, rad in folder_dist_rad:
        result_path = os.path.join(folder, 'motion_2d_results', 'plot_validate_trajectory.png')
        if os.path.exists(result_path):
            measurements[(dist, rad)] = result_path

    # Plot each image in the grid
    for i, dist in enumerate(distances):
        for j, rad in enumerate(radii):
            ax = fig.add_subplot(gs[i, j])

            # Add radius labels at the top of first row
            if i == 0:
                ax.set_title(f'Radius: {rad}m', pad=10, fontsize=12)

            # Try to find and load the corresponding image
            if (dist, rad) in measurements:
                img_path = measurements[(dist, rad)]
                img = mpimg.imread(img_path)
                ax.imshow(img)
            else:
                # For missing combinations, create an empty plot with a note
                ax.text(0.5, 0.5, 'No data',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)

            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])

            # Add distance labels on the left
            if j == 0:  # Leftmost column
                ax.set_ylabel(f'Distance: {dist}m')

    # Save the synthetic figure
    output_path = os.path.join(RESULT_DIR, 'trajectory_validations.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Synthetic figure saved to: {output_path}")