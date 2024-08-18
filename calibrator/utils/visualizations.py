import cv2
import random
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from typing import List, Tuple
from utils import coord2idx, idx2coord

import torch
import torchvision


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


def visualize_img2img_result(
    pred_depth_img: torch.Tensor,   # [B,H,W]
    warped_proj_depth_img: torch.Tensor,  # [B,H,W]
    warped_proj_depth_mask: torch.Tensor = None  # [B,H,W]
):
    pred_depth_img = pred_depth_img.detach().cpu().numpy()
    warped_proj_depth_img = warped_proj_depth_img.detach().cpu().numpy()
    warped_proj_depth_mask = warped_proj_depth_mask.detach().cpu().numpy()

    batch_size, img_h, img_w = pred_depth_img.shape

    transform = torchvision.transforms.ToTensor()
    prj_imgs = []
    for b in range(batch_size):
        base_img = cv2.cvtColor(
            linear2pixel(pred_depth_img[b, :, :]),
            cv2.COLOR_GRAY2RGB)
        overlie_img = _cstm_rgba(
            linear2pixel(warped_proj_depth_img[b, :, :])
        ).astype(np.float32)[:, :, :3]
        # draw
        if warped_proj_depth_mask is None:
            img = cv2.addWeighted(
                base_img * 255, 0.7,
                overlie_img * 255, 0.3,
                0)
        else:
            base_img *= 0.5
            overlie_img *= warped_proj_depth_mask[b, :, :, np.newaxis] * 1.0
            img = np.clip((base_img + overlie_img) * 255, 0, 255)

        prj_imgs.append(transform(img))

    return prj_imgs


def visualize_img2ptc_result(
    base_imgs: torch.Tensor,
    prj_pts: torch.Tensor,   # [B,3,N]
    pt_values: torch.Tensor = None, # [B,N]
    mode: str = 'individual'    # "individual" or "merge"
):
    """make image overlapping with projected points

    Args:
        base_imgs (torch.Tensor): the background image [B,H,W,C]
        prj_pts (torch.Tensor): 2D point cloud [B,3,N]
        pt_values (torch.Tensor): the user-defined color values [B,N] (default: None)

    Returns:
        List[torch.Tensor]: the list of projected images
    """
    base_imgs = base_imgs.detach()
    prj_pts = prj_pts.detach()
    D = prj_pts[:, 2, :].cpu().numpy()
    U = prj_pts[:, 0, :].cpu().numpy() / D
    V = prj_pts[:, 1, :].cpu().numpy() / D

    batch_size = base_imgs.size(0)
    img_h = base_imgs.size(1)
    img_w = base_imgs.size(2)

    transform = torchvision.transforms.ToTensor()
    prj_imgs = []
    if mode == 'individual':
        for b in range(batch_size):
            # get valid points
            u = U[b]
            v = V[b]
            d = D[b]
            mask = (0 <= u) * (u < img_w) * (0 <= v) * (v < img_h)
            masked_u = u[mask]
            masked_v = v[mask]
            # the color depends on "inverse depth" or user-defined values
            masked_d = d[mask]
            masked_inv_d = 1 / masked_d if pt_values is None else pt_values.cpu().numpy()[b][mask]
            masked_d_norm = linear2pixel(
                (masked_inv_d - masked_inv_d.min()) /
                (masked_inv_d.max() - masked_inv_d.min() + 1e-6)
            )

            # draw
            if len(base_imgs.size()) == 3:   # depth image
                img = cv2.cvtColor(
                    linear2pixel(base_imgs[b].cpu().numpy()) * 255,
                    cv2.COLOR_GRAY2RGB)
            elif len(base_imgs.size()) == 4:   # rgb image
                img = base_imgs[b].cpu().numpy().copy()

            # Create a mask for valid points
            valid_mask = (masked_u >= 0) & (masked_u < img_w) & (masked_v >= 0) & (masked_v < img_h)
            valid_u = masked_u[valid_mask].astype(int)
            valid_v = masked_v[valid_mask].astype(int)
            valid_d = masked_d_norm[valid_mask]

            # Create an empty image for the circles
            circle_img = np.zeros_like(img, dtype=np.float32)

            # Define the radius of the circles
            radius = 3

            # Create a grid of coordinates
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = x**2 + y**2 <= radius**2

            # Draw circles on the empty image
            for u, v, d, d_meter in zip(valid_u, valid_v, valid_d, masked_d[valid_mask]):
                y_min = max(0, v-radius)
                y_max = min(img_h, v+radius+1)
                x_min = max(0, u-radius)
                x_max = min(img_w, u+radius+1)
                mask_y_min = max(0, radius-v)
                mask_y_max = min(2*radius+1, img_h-v+radius)
                mask_x_min = max(0, radius-u)
                mask_x_max = min(2*radius+1, img_w-u+radius)

                # Check if depth value is around 100
                if 97 <= d_meter <= 103:
                    color = np.array([0, 0, 1])  # Blue color for depth around 100
                else:
                    color = np.array(_cstm_rgba(d)[:3])

                circle_img[y_min:y_max, x_min:x_max][mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]] = color * 255

                # circle_img[y_min:y_max, x_min:x_max][mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]] = np.array(_cstm_rgba(d)[:3]) * 255

            # Add the circles to the original image
            mask_nonzero = circle_img.sum(axis=-1) > 0
            img[mask_nonzero] = circle_img[mask_nonzero]

            prj_imgs.append(transform(img))
    elif mode == 'merge':
        # make the base image
        alpha = 1.0 / batch_size
        base_img = np.zeros_like(base_imgs[0].cpu().numpy(), dtype=np.float32)
        for b in range(batch_size):
            image = base_imgs[b].cpu().numpy()
            # Weighted addition
            base_img += alpha * image
        # put the projected points
        img = base_img
        for b in range(batch_size):
            # get valid points
            u = U[b]
            v = V[b]
            d = D[b]
            mask = (0 <= u) * (u < img_w) * (0 <= v) * (v < img_h)
            masked_u = u[mask]
            masked_v = v[mask]
            # the color depends on "inverse depth" or user-defined values
            masked_d = 1 / d[mask] if pt_values is None else pt_values.cpu().numpy()[b][mask]
            masked_d_norm = linear2pixel(
                (masked_d - masked_d.min()) /
                (masked_d.max() - masked_d.min() + 1e-6)
            )

            # draw
            for u, v, d in zip(masked_u, masked_v, masked_d_norm):
                img = cv2.circle(
                    img,
                    center=(int(u), int(v)),
                    radius=1,
                    color=np.array(_cstm_rgba(d)[:3]) * 255,
                    thickness=-1)

        prj_imgs.append(transform(img))
    else:
        raise ValueError(f'mode={mode} is not defined.')

    return prj_imgs


def visualize_point_pairs(
    point_idx_pairs: List[torch.Tensor],    # B * [2,N]
    base_imgs: List[torch.Tensor] = None  # B * [3,H,W]
) -> List[torch.Tensor]:
    assert len(point_idx_pairs) == len(base_imgs)
    B = len(point_idx_pairs)
    transform_tensor = torchvision.transforms.ToTensor()

    result_imgs = []
    for b in range(B):
        pairs = point_idx_pairs[b].cpu().numpy()    # [2,N]
        base_img = base_imgs[b].cpu().numpy().transpose((1, 2, 0))

        # Scale the image data to [0, 255] and convert it to uint8
        img = base_img.copy()

        H, W, _ = img.shape
        n_pair = pairs.shape[1]

        for i in range(int(n_pair * 0.03)):
            indices = pairs[:, i]
            start_point = [x.item() for x in idx2coord(indices[0], W)]
            end_point = [x.item() for x in idx2coord(indices[1], W)]
            img = cv2.line(
                img,
                tuple(start_point),
                tuple(end_point),
                color=(0, 255, 0),
                thickness=1,
            )
        result_imgs.append(transform_tensor(img))

    return result_imgs


def visualize_open3d(pcd_with_rgb, parameters=None,
                     window_size: Tuple[int] = (1280, 720)):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_size[0], height=window_size[1])

    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = o3d.utility.Vector3dVector(pcd_with_rgb[:, :3])
    all_pcd.colors = o3d.utility.Vector3dVector(pcd_with_rgb[:, 3:])

    # show the axis
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())

    # show the point cloud
    vis.add_geometry(all_pcd)

    # change the background color
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.7, 0.7, 0.7])

    # Run the visualizer
    vis.run()
    view_ctl = vis.get_view_control()  # Set the viewpoint
    if parameters is not None:
        view_ctl.convert_from_pinhole_camera_parameters(parameters)

    # update the view point
    parameters = view_ctl.convert_to_pinhole_camera_parameters()

    vis.destroy_window()

    return parameters
