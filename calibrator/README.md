# Manual Calibrator for WildPose v1.1

## Manual Calibrator

After the change `debug_config.json`, run:
```bash
$ python manual_calibrator.py --config debug_config.json
```

Operation
```yaml
# change intrinsic parameters
←: decrease x-position of principal point c_x
→: increase x-position of principal point c_x
↑: decrease y-position of principal point c_y
↓: increase y-position of principal point c_y

# change extrinsic parameters
w: increase the camera pitch
s: decrease the camera pitch
a: increase the camera yaw
d: decrease the camera yaw
e: increase the camera roll
q: decrease the camera roll
W: increase the camera position z (depth direction)
S: decrease the camera position z
A: increase the camera position x
D: decrease the camera position x

0: reset parameters
m: merge frames in a batch
c: change the point color mode
>: increase the change step
<: decrease the change step
n: show the next frame
return/enter: save the current parameter
```

## Acknowledgement

This project is based on [victoresque/pytorch-template](https://github.com/victoresque/pytorch-template).

