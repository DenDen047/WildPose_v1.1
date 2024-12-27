# Manual Calibrator for WildPose v1.1

This tool helps calibrate camera parameters for the WildPose system. It allows users to manually adjust both intrinsic and extrinsic camera parameters through an interactive interface.

## Purpose
- Assists in fine-tuning camera calibration parameters for WildPose
- Provides visual feedback for parameter adjustments
- Enables batch processing of multiple frames
- Supports manual refinement of auto-calibrated results

## Installation
```bash
$ conda env create -f environment.yml
$ conda activate wildpose
```

## Usage
1. Configure your settings in `debug_config.json`
2. Run the calibrator:
```bash
$ python manual_calibrator.py --config debug_config.json
```

## Controls and Operation
```yaml
# Intrinsic Parameter Controls
←/→: Adjust principal point (c_x) left/right
↑/↓: Adjust principal point (c_y) up/down

# Extrinsic Parameter Controls
w/s: Increase/decrease camera pitch
a/d: Increase/decrease camera yaw
e/q: Increase/decrease camera roll
W/S: Increase/decrease camera Z position (depth)
A/D: Increase/decrease camera X position

# Additional Controls
0: Reset all parameters to default
m: Merge frames in current batch
c: Toggle point color display mode
>/< : Increase/decrease adjustment step size
n: Advance to next frame
Enter: Save current parameters
```

## Output
The calibrator saves the adjusted parameters in the format specified in your configuration file. These parameters can then be used in the main WildPose system for accurate pose estimation.

## Acknowledgement
This project is based on [victoresque/pytorch-template](https://github.com/victoresque/pytorch-template).

