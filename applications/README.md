# WildPose Applications

This repository contains the code for the figures and tables in the WildPose paper.

## Setting up the environment

```bash
conda env create -f environment.yml
conda activate wildpose
```

## File Organization for a scene data

The data is organized as follows:

```
data/
├── manual_calibration.json
├── sync_rgb/
│   ├── 000001.jpeg
│   ├── 000002.jpeg
│   ├── 000003.jpeg
│   ├── ...
├── lidar/
│   ├── 000001.pcd
│   ├── 000002.pcd
│   ├── 000003.pcd
│   ├── ...
├── masks/
    ├── 000001.png
    ├── 000002.png
    ├── 000003.png
    ├── ...
```

where `sync_rgb` is the synchronized RGB images, `lidar` is the LiDAR point clouds, and `masks` is the masks of the animals.

## Making the figures and tables in the WildPose paper

### Introduction section

Fig. 1D.
```bash
python make_depth_image.py
```

### Calibration Validation

For the calibration validation figures (Fig. 2A), run:
```bash
python plot_calib_validation.py             # Fig. S1
python plot_calib_validation_by_angles.py   # Fig. 2A
```

### Object Tracking Precision

Analyze the validation results of the object tracking precision.
```bash
# First, run the validation:
python validate_trajectory.py \
    --mode position_2d \
    --data_dir /path/to/data
./batch_validate_trajectory.sh

# Then generate figures:
python plot_trajectory_validation_by_circles.py # Fig. 2D
python plot_motion_validation.py                # Fig. 2E
python create_synthetic_figure.py               # Fig. S2
```

### Animal Morphology & Locomotion

Get the morphometrics data (Table 1).
```bash
python measure_morphometrics.py
```

Fig. 3A&B.
```bash
python plot_keypoints.py
```

- `single_frame` mode (Fig. 3A)
- `left_{front|hind}_leg` mode (Fig. 3B)

Fig. S3.
```bash
python plot_coloured_pcd.py
```

### Tracking Individual Animals in 3D

```bash
python plot_3d_trajectory.py    # Supports multiple modes:
# --mode position_3d            # Fig. S4A
# --mode position_without_y     # Fig. 3C
# --mode velocity               # Fig. S4B-D
# --mode neighbor_density       # Fig. 3D
# --mode neighbor_density_animation # Movie 1
```

### Fine Scale Deformation Monitoring

Prerequisites:
1. Generate bounding boxes for target animals
2. Predict individual masks using Segment Anything (`segment_anything.py`)
3. Estimate body size transitions (`body_size_estimator.py`)

```bash
python plot_breathing.py    # Supports modes:
# --mode filtered   # Fig. 4A
# --mode fft        # Fig. 4B
# --mode imu        # Fig. 4C and 4D
```
