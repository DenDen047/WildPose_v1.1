import cv2
import os
import numpy as np
import open3d as o3d
import pickle
import rigid_body_motion as rbm
import quaternion
import glob
import os
import pdb
import time
from typing import Tuple


def make_intrinsic(fx, fy, cx, cy):

    intrinsic_mat = np.eye(4)
    intrinsic_mat[0, 0] = fx
    intrinsic_mat[0, 2] = cx
    intrinsic_mat[1, 1] = fy
    intrinsic_mat[1, 2] = cy

    return intrinsic_mat


def make_extrinsic(rot_mat, translation):

    extrinsic_mat = np.eye(4)
    extrinsic_mat[:3, :3] = rot_mat
    extrinsic_mat[:-1, -1] = translation
    extrinsic_mat[-1, -1] = 1

    return extrinsic_mat


def lidar2cam_projection(pcd, extrinsic):
    velo = np.insert(pcd, 3, 1, axis=1).T
    velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)
    pcd_in_cam = extrinsic.dot(velo)

    return pcd_in_cam


def cam2image_projection(pcd, intrinsic):

    pcd_in_image = intrinsic.dot(pcd)
    pcd_in_image[:2] /= pcd_in_image[2, :]

    return pcd_in_image


def extract_rgb_from_image(img, pcd_in_image, width, height):

    valid_mask = \
        (pcd_in_image[:, 0] >= 0) & \
        (pcd_in_image[:, 0] < width) & \
        (pcd_in_image[:, 1] >= 0) & \
        (pcd_in_image[:, 1] < height) & \
        (pcd_in_image[:, 2] > 0)

    pixel_locs = np.concatenate(
        [pcd_in_image[valid_mask, 1][:, None], pcd_in_image[valid_mask, 0][:, None]], 1)
    pixel_locs = pixel_locs.astype(int)

    colors = np.zeros((len(pcd_in_image), 3))
    colors[:, :3] = 0.4
    colors[valid_mask, :] = img[pixel_locs[:, 0], pixel_locs[:, 1]] / 255.0

    return colors, valid_mask


def visualize_opencv(img, pcd_in_image, width, height):

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    valid_mask = \
        (pcd_in_image[:, 0] >= 0) & \
        (pcd_in_image[:, 0] < width) & \
        (pcd_in_image[:, 1] >= 0) & \
        (pcd_in_image[:, 1] < height) & \
        (pcd_in_image[:, 2] > 0)

    pixel_locs = np.concatenate(
        (pcd_in_image[valid_mask, 1][:, None],
         pcd_in_image[valid_mask, 0][:, None]),
        axis=1)
    pixel_locs = pixel_locs.astype(int)

    img_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    img_mask[pixel_locs[:, 0], pixel_locs[:, 1]] = 1

    img_bgr *= img_mask
    img_bgr_canvas = np.copy(img_bgr)

    for valid_idx in range(len(pixel_locs)):
        color = img_bgr[pixel_locs[valid_idx, 0], pixel_locs[valid_idx, 1]]
        cv2.circle(img_bgr_canvas, (pixel_locs[valid_idx, 1], pixel_locs[valid_idx, 0]),
                   5, (int(color[0]), int(color[1]), int(color[2])), -1)

    return img_bgr_canvas


def get_manual_viewpoint(pcd_with_rgb, window_size: Tuple[int]):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_size[0], height=window_size[1])

    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = o3d.utility.Vector3dVector(pcd_with_rgb[:, :3])
    all_pcd.colors = o3d.utility.Vector3dVector(pcd_with_rgb[:, 3:])

    vis.add_geometry(all_pcd)

    print('Adjust the viewpoint as needed, then close the visualization window to continue...')

    # Run the visualizer
    vis.run()
    view_ctl = vis.get_view_control()  # Set the viewpoint
    parameters = view_ctl.convert_to_pinhole_camera_parameters()

    vis.destroy_window()

    # Return the manually set parameters
    return parameters


def visualize_open3d(pcd_with_rgb, parameters, window_size: Tuple[int]):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_size[0], height=window_size[1])

    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = o3d.utility.Vector3dVector(pcd_with_rgb[:, :3])
    all_pcd.colors = o3d.utility.Vector3dVector(pcd_with_rgb[:, 3:])

    vis.add_geometry(all_pcd)
    view_ctl = vis.get_view_control()

    view_ctl.convert_from_pinhole_camera_parameters(parameters)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("tmp.png")
    vis.destroy_window()

    img = cv2.imread('tmp.png')

    # Resize the image while keeping aspect ratio the same
    zoom_factor = 2
    new_size = (int(img.shape[1] * zoom_factor),
                int(img.shape[0] * zoom_factor))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    # If the image is larger than the original size, crop it to the original
    # size
    if zoom_factor > 1.0:
        start_x = (img.shape[1] - window_size[0]) // 2
        start_y = (img.shape[0] - window_size[1]) // 2
        img = img[start_y:start_y + window_size[1],
                  start_x:start_x + window_size[0]]

    return img


def euler2mat(yaw, pitch, roll):
    z = yaw
    y = pitch
    x = roll

    cosz = np.cos(z)
    sinz = np.sin(z)

    zeros = 0
    ones = 1
    zmat = np.array([cosz, -sinz, zeros,
                     sinz, cosz, zeros,
                     zeros, zeros, ones]).reshape(3, 3)

    cosy = np.cos(y)
    siny = np.sin(y)

    ymat = np.array([cosy, zeros, siny,
                     zeros, ones, zeros,
                     -siny, zeros, cosy]).reshape(3, 3)

    cosx = np.cos(x)
    sinx = np.sin(x)

    xmat = np.array([ones, zeros, zeros,
                     zeros, cosx, -sinx,
                     zeros, sinx, cosx]).reshape(3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def main():
    data_dir = 'data/lion_walk'
    lidar_dir = os.path.join(data_dir, 'lidar/')
    rgb_dir = os.path.join(data_dir, 'rgb/')
    sync_rgb_dir = os.path.join(data_dir, 'sync_rgb/')

    lidar_list = sorted(glob.glob(os.path.join(lidar_dir, '*.pcd')))
    rgb_list = sorted(glob.glob(os.path.join(rgb_dir, '*.jpeg')))
    sync_rgb_list = sorted(glob.glob(os.path.join(sync_rgb_dir, '*.jpeg')))
    parameters = None
    # parameters = o3d.io.read_pinhole_camera_parameters("data/params2.json")

    # lion_walk
    translation = np.array([0.14, 0.00149, -0.076])
    rot_mat = euler2mat(1.5707963267948966, 0, 1.5707963267948966)
    fx, fy, cx, cy = 16000, 16000, 700, 370
    # # pcg_take_off
    # fx, fy, cx, cy = 28000, 28000, 870, 470
    # translation = np.array([0.14, 0.00149, -0.076])
    # rot_mat = euler2mat(1.5707963267948966, 0, 1.5707963267948966)

    IMG_WIDTH, IMG_HEIGHT = 1280, 720
    window_size = (IMG_WIDTH, IMG_HEIGHT)

    OUTPUT_WIDTH, OUTPUT_HEIGHT = IMG_WIDTH * 3, IMG_HEIGHT
    FPS = 170
    OUTPUT_VIDEO_PATH = 'output.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS,
                          (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    # choose only the image will be used for the video
    rgb_fnames = [os.path.basename(x) for x in rgb_list]
    sync_rgb_fnames = [os.path.basename(x) for x in sync_rgb_list]
    start_idx = rgb_fnames.index(sync_rgb_fnames[0])
    end_idx = rgb_fnames.index(sync_rgb_fnames[-1])
    rgb_list = rgb_list[start_idx:end_idx + 1]

    # Get the first RGB and LiDAR paths
    idx = -1    # the index for sync_rgb image and pcd file
    update_idx = False
    img_valid_mask = None
    pcd_img = None
    for rgb_fpath in rgb_list:
        if os.path.basename(rgb_fpath) == os.path.basename(
                sync_rgb_fnames[idx + 1]):
            idx += 1
            update_idx = True
        sync_rgb_path = sync_rgb_list[idx]
        lidar_path = lidar_list[idx]

        # load camera image
        img_bgr = cv2.imread(rgb_fpath)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if update_idx:
            pcd_in_lidar = o3d.io.read_point_cloud(lidar_path)
            pcd_in_lidar = np.asarray(pcd_in_lidar.points)

            intrinsic = make_intrinsic(fx, fy, cx, cy)
            extrinsic = make_extrinsic(rot_mat, translation)

            pcd_in_cam = lidar2cam_projection(pcd_in_lidar, extrinsic)
            pcd_in_image = cam2image_projection(pcd_in_cam, intrinsic)

            pcd_in_cam = pcd_in_cam.T[:, :-1]
            pcd_in_image = pcd_in_image.T[:, :-1]

            colors, valid_mask = extract_rgb_from_image(
                img, pcd_in_image, width=IMG_WIDTH, height=IMG_HEIGHT)
            pcd_with_rgb = np.concatenate([pcd_in_cam, colors], 1)

            # manually adjust the viewpoint
            if parameters is None:
                parameters = get_manual_viewpoint(pcd_with_rgb, window_size)

            pcd_img = visualize_open3d(pcd_with_rgb, parameters, window_size)
            img_valid_mask = visualize_opencv(
                img, pcd_in_image, width=IMG_WIDTH, height=IMG_HEIGHT)

        # write the video frame
        merged_img = np.concatenate([pcd_img, img_valid_mask, img_bgr], 1)
        out.write(merged_img)

        # reset the flag
        update_idx = False

        # if rgb_fpath == rgb_list[10]:
        #     break

    out.release()


if __name__ == "__main__":
    main()
