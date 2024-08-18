import os
import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scienceplots

from utils.file_loader import load_camera_parameters, load_pcd
from utils.camera import make_intrinsic_mat, make_extrinsic_mat
from utils.projection import lidar2cam_projection, cam2image_projection


# plt.style.use(['science', 'nature', 'no-latex'])
# figure(figsize=(10, 6))
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Arial')
plt.rcParams.update({
    'legend.frameon': False,
    "pdf.fonttype": 42,
})

MEAS_DIR = './data/red_hartebeest_stand'
PCD_FPATH = os.path.join(
    MEAS_DIR, 'lidar',
    'livox_frame_1670832407_462632296.pcd'
)
CALIB_FPATH = os.path.join(MEAS_DIR, 'manual_calibration.json')

IMG_WIDTH, IMG_HEIGHT = 1280, 720


def _cstm_rgba(x):
    # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    # min_val (yellow:0) -> max_val (red:1)
    rgba = plt.cm.hot((np.clip(x * 10, 2, 10) - 2) / 8.)
    return rgba


def linear2pixel(x: np.ndarray):
    # https://gist.github.com/andrewgiessel/4589258
    # get image histogram
    imhist, bins = np.histogram(x.flatten(), bins=256, density=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(x.flatten(), bins[:-1], cdf)
    return np.clip(
        im2.reshape(x.shape),
        a_min=0.0, a_max=1.0).astype('float32')


def visualize_img2ptc_result(
    prj_pts,   # [N,3]
):
    # get valid points
    u = prj_pts[:, 0]
    v = prj_pts[:, 1]
    d = prj_pts[:, 2]
    mask = (0 <= u) * (u < IMG_WIDTH) * (0 <= v) * (v < IMG_HEIGHT)
    masked_u = u[mask]
    masked_v = v[mask]
    # the color depends on "inverse depth" or user-defined values
    masked_d = 1 / d[mask]
    masked_d_norm = linear2pixel(
        (masked_d - masked_d.min()) /
        (masked_d.max() - masked_d.min() + 1e-6)
    )

    # draw
    img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
    for u, v, d in zip(masked_u, masked_v, masked_d_norm):
        img = cv2.circle(
            img,
            center=(int(u), int(v)),
            radius=5,
            color=np.array(_cstm_rgba(d)[:3]) * 255,
            thickness=-1)

    return img


def main():
    pcd_open3d = load_pcd(PCD_FPATH, mode='open3d')
    pts_in_lidar = np.asarray(pcd_open3d.points)
    fx, fy, cx, cy, rot_mat, translation = load_camera_parameters(CALIB_FPATH)
    intrinsic = make_intrinsic_mat(fx, fy, cx, cy)
    extrinsic = make_extrinsic_mat(rot_mat, translation)

    # reprojection
    pcd_in_cam = lidar2cam_projection(pts_in_lidar, extrinsic)
    pcd_in_img = cam2image_projection(pcd_in_cam, intrinsic)
    pcd_in_cam = pcd_in_cam.T[:, :-1]
    pcd_in_img = pcd_in_img.T[:, :-1]

    result_img = visualize_img2ptc_result(pcd_in_img)

    # export the result
    cv2.imwrite(
        f'results/output.png',
        cv2.cvtColor(result_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    )


if __name__ == '__main__':
    main()