from typing import Tuple
import numpy as np
import cv2
import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Function
import matplotlib.pyplot as plt
import utils


def nll_loss(output, target):
    return F.nll_loss(output, target)


class MultiStepFunction(Function):
    @staticmethod
    def forward(ctx, input, tau):
        ctx.save_for_backward(input)
        ctx.tau = tau

        # Initializing output
        output = torch.zeros_like(input)

        # f(x) = -1 if x < 1/(1+tau)
        output[input < 1 / (1 + tau)] = -1

        # f(x) = 0 if 1/(1+tau) <= x < (1+tau)
        output[(input >= 1 / (1 + tau)) & (input < 1 + tau)] = 0

        # f(x) = 1 else
        output[input >= 1 + tau] = 1

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        tau = ctx.tau
        grad_input = torch.zeros_like(input)

        # f(x) = -1 if x < 1/(1+tau)
        grad_input[input < 1 / (1 + tau)] = grad_output[input < 1 / (1 + tau)]

        # f(x) = 0 if 1/(1+tau) <= x < (1+tau)
        grad_input[(input >= 1 / (1 + tau)) & (input < 1 + tau)] = 0

        # f(x) = 1 else
        grad_input[input >= 1 + tau] = grad_output[input >= 1 + tau]

        return grad_input, None


class SmoothMultiStepFunction(Function):
    @staticmethod
    def forward(ctx, input, tau):
        ctx.save_for_backward(input, tau)
        output = torch.where(
            input >= (1 + tau),
            input, torch.where(
                (input > 1 / (1 + tau)) & (input < (1 + tau)),
                0., -torch.abs(input)
            )
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, tau = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(input >= 1 / (1 + tau)) & (input <= (1 + tau))] = 0.
        grad_input[input < 1 /
                   (1 + tau)] = grad_output[input < 1 / (1 + tau)] * -1
        return grad_input, None


def preprocessing(pred_depth_img, intrinsic_init_mat, extrinsic_init_mat):
    if len(pred_depth_img.size()) == 2:
        pred_depth_img = pred_depth_img[None, :, :]

    intrinsic_init_mat = intrinsic_init_mat.mean(dim=0)  # [3,3]
    extrinsic_init_mat = extrinsic_init_mat.mean(dim=0)  # [3,4]
    return pred_depth_img, intrinsic_init_mat, extrinsic_init_mat


def warp(
    pred_depth_img,
    proj_depth_img, proj_depth_mask,
    intrinsic_mat, pose_mat,
    intrinsic_init_mat,
):
    """Photometric Loss

    Args:
        pred_depth_img (torch.Tensor): the predicted depth image -- [B,H,W]
        proj_depth_img (torch.Tensor): the depth image where the point cloud was projected -- [B,H,W]
        proj_depth_mask (torch.Tensor): the depth mask showing the valid pixels -- [B,H,W]
        intrinsic_mat (torch.Tensor): the intrinsic camera matrix -- [3,3]
        pose_mat (torch.Tensor): the extrinsic camera matrix -- [3,4]
        intrinsic_init_mat (torch.Tensor): the projection matrix -- [B,3,3]

    Returns:
        float: the loss value
    """
    B, H, W = pred_depth_img.size()
    device = pred_depth_img.device

    # define coordinate grids
    coordinate = torch.tensor(
        np.mgrid[0:H, 0:W][[1, 0], :, :],
        dtype=torch.float32,
        requires_grad=False,
        device=device)  # [uv, H, W]
    coordinate = coordinate.view(2, -1)[None, :, :].expand(
        B, -1, -1)   # [B, uv, H*W]

    pred_img_coord = torch.cat(
        [coordinate, torch.ones((B, 1, H * W), device=device)],
        dim=1
    ) * proj_depth_img.view(B, -1)[:, None, :]  # [B, uv1, H*W]

    # image plane -> camera coordinate (making undistorted image)
    # [B, 3, H*W]
    pred_cam_coord = torch.inverse(intrinsic_mat) @ pred_img_coord

    # get new image coordinate
    pred_camera_coordinate = intrinsic_init_mat @ inv_pose_operation(
        pred_cam_coord, pose_mat)  # [B, 3, H*W]

    # warping
    U = pred_camera_coordinate[:, 0]
    V = pred_camera_coordinate[:, 1]
    D = pred_camera_coordinate[:, 2].clamp(min=1e-12)
    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B,
    # H*W]
    U_norm = 2 * (U / D) / (W - 1) - 1
    V_norm = 2 * (V / D) / (H - 1) - 1

    init2pred_grid = torch.stack([U_norm, V_norm], dim=2)  # [B, H*W, 2]

    warped_proj_depth_img = 1. / F.grid_sample(  # inverse depth
        proj_depth_img[:, None, :, :],  # [B,1,H,W]
        grid=init2pred_grid.reshape(-1, H, W, 2),  # [B,H,W,2]
        padding_mode='border',
        align_corners=False
    )[:, 0, :, :]  # [B,H,W]
    warped_proj_depth_mask = F.grid_sample(
        proj_depth_mask[:, None, :, :],  # [B,1,H,W]
        grid=init2pred_grid.reshape(-1, H, W, 2),  # [B,H,W,2]
        padding_mode='zeros',
        align_corners=False
    )[:, 0, :, :]  # [B,H,W]

    return warped_proj_depth_img, warped_proj_depth_mask


def ranking_loss(
    pred_depth_img,
    proj_depth_img, proj_depth_mask, ptcld,
    intrinsic_mat, pose_mat,
    intrinsic_init_mat,
    extrinsic_init_mat,
    return_visualization=False
):
    """Ranking Loss (https://github.com/JiawangBian/sc_depth_pl/tree/master)

    Args:
        pred_depth_img (torch.Tensor): the predicted depth image -- [B,H,W]
        proj_depth_img (torch.Tensor): the depth image where the point cloud was projected -- [B,H,W]
        proj_depth_mask (torch.Tensor): the depth mask showing the valid pixels -- [B,H,W]
        ptcld (torch.Tensor): point cloud data -- [B,N]
        intrinsic_mat (torch.Tensor): the intrinsic camera matrix -- [3,3]
        pose_mat (torch.Tensor): the extrinsic camera matrix -- [3,4]
        intrinsic_init_mat (torch.Tensor): the projection matrix -- [B,3,3]
        return_visualization (bool): if True, returns projected images

    Returns:
        float: the loss value
    """
    assert pred_depth_img.size() == proj_depth_img.size() == proj_depth_mask.size()
    B, H, W = pred_depth_img.size()
    device = pred_depth_img.device
    sample_num = 100000
    loss = torch.tensor(0.0).to(device)
    multistep = MultiStepFunction.apply

    # preprocessing and warping
    pred_depth_img, intrinsic_init_mat, extrinsic_init_mat = preprocessing(
        pred_depth_img, intrinsic_init_mat, extrinsic_init_mat
    )
    warped_proj_depth_img, warped_proj_depth_mask = warp(
        pred_depth_img,
        proj_depth_img, proj_depth_mask,
        intrinsic_mat, pose_mat,
        intrinsic_init_mat,
    )
    norm_pred_depth_img = preprocess_depth_img(
        pred_depth_img, mode='normalization')
    norm_warped_proj_depth_img = preprocess_depth_img(
        warped_proj_depth_img, mode='normalization')

    pair_idx_list = []
    for b in range(B):
        img = norm_pred_depth_img[b, :, :]  # [H,W]
        lidar = norm_warped_proj_depth_img[b, :, :]  # [H,W]
        mask = warped_proj_depth_mask[b, :, :]  # [H,W]

        index_candidates = (0.5 < mask).nonzero()   # [N,yx]
        combinations = torch.combinations(
            torch.arange(index_candidates.size(0), device=device),
            r=2
        )   # [nC2,2]
        pair_candidates = index_candidates[combinations]  # [nC2, 2, yx]
        pair_distances = torch.linalg.vector_norm(
            (pair_candidates[:, 0, :] - pair_candidates[:, 1, :]).float(), dim=1)   # [N]
        valid_distance_mask = (20 <= pair_distances) & (pair_distances <= 70)
        valid_pairs = pair_candidates[valid_distance_mask]  # [N,2,yx]
        a_idxs = utils.coord2idx(valid_pairs[:, 0, 1], valid_pairs[:, 0, 0], W)
        b_idxs = utils.coord2idx(valid_pairs[:, 1, 1], valid_pairs[:, 1, 0], W)

        # ignore the same point pairs
        _mask = (a_idxs - b_idxs) != 0
        a_idxs = a_idxs[_mask]
        b_idxs = b_idxs[_mask]

        pair_idxs = torch.stack(
            (a_idxs, b_idxs), dim=0).long().detach()

        # loss of a pair of points with predicted depth values
        def _pair_ranking_loss(
            p_imgA: torch.Tensor,
            p_imgB: torch.Tensor,
            p_lidarA: torch.Tensor,
            p_lidarB: torch.Tensor,
        ):
            tau = 0.15

            x = p_imgA / (p_imgB + 1e-6)
            weight = multistep(x, torch.tensor([tau]).to(device))
            loss1 = F.softplus(-weight * (p_lidarA - p_lidarB))

            x = p_lidarA / (p_lidarB + 1e-6)
            weight = multistep(x, torch.tensor([tau]).to(device))
            loss2 = F.softplus(-weight * (p_imgA - p_imgB))

            return loss1 + loss2

        taken_losses = _pair_ranking_loss(
            torch.take(img, index=pair_idxs[0, :]),
            torch.take(img, index=pair_idxs[1, :]),
            torch.take(lidar, index=pair_idxs[0, :]),
            torch.take(lidar, index=pair_idxs[1, :]),
        )
        _mask_a = torch.take(mask, index=pair_idxs[0, :])
        _mask_b = torch.take(mask, index=pair_idxs[1, :])
        taken_mask = (_mask_a + _mask_b) / 2

        loss = loss + torch.mean(taken_mask * taken_losses)
        # save data for visualization
        sorted, indices = torch.sort(
            (taken_mask * taken_losses).detach(),
            descending=True
        )
        pair_idx_list.append(pair_idxs[:, indices])
    loss = loss / B

    # loss
    if return_visualization:
        # warped result
        warp_imgs = utils.visualize_img2img_result(
            pred_depth_img,
            warped_proj_depth_img,
            warped_proj_depth_mask)
        warp_imgs = utils.visualize_point_pairs(
            pair_idx_list, base_imgs=warp_imgs)
        # projected point cloud
        prj_mat = get_projection_matrix(
            intrinsic_mat, extrinsic_init_mat, pose_mat)    # [3,4]
        ptcld[:, 3, :] = 1
        prj_pts = prj_mat[None, :, :] @ ptcld  # [B,3,N]
        prj_imgs = utils.visualize_img2ptc_result(
            pred_depth_img, prj_pts)
        return loss, warp_imgs, warped_proj_depth_mask, prj_imgs
    else:
        return loss


def photometric_loss(
    pred_depth_img,
    proj_depth_img, proj_depth_mask, ptcld,
    intrinsic_mat, pose_mat,
    intrinsic_init_mat,
    extrinsic_init_mat,
    return_visualization=False
):
    """Photometric Loss

    Args:
        pred_depth_img (torch.Tensor): the predicted depth image -- [B,H,W]
        proj_depth_img (torch.Tensor): the depth image where the point cloud was projected -- [B,H,W]
        proj_depth_mask (torch.Tensor): the depth mask showing the valid pixels -- [B,H,W]
        ptcld (torch.Tensor): point cloud data -- [B,N]
        intrinsic_mat (torch.Tensor): the intrinsic camera matrix -- [3,3]
        pose_mat (torch.Tensor): the extrinsic camera matrix -- [3,4]
        intrinsic_init_mat (torch.Tensor): the projection matrix -- [B,3,3]
        return_visualization (bool): if True, returns projected images

    Returns:
        float: the loss value
    """
    pred_depth_img, intrinsic_init_mat, extrinsic_init_mat = preprocessing(
        pred_depth_img, intrinsic_init_mat, extrinsic_init_mat
    )

    warped_proj_depth_img, warped_proj_depth_mask = warp(
        pred_depth_img,
        proj_depth_img, proj_depth_mask,
        intrinsic_mat, pose_mat,
        intrinsic_init_mat,
    )

    # normalize the depth images [B,H,W]
    norm_pred_depth_img = preprocess_depth_img(
        pred_depth_img, mode='normalization')
    norm_warped_proj_depth_img = preprocess_depth_img(
        warped_proj_depth_img, mode='normalization')

    loss = ((
        norm_pred_depth_img - norm_warped_proj_depth_img
    ) * warped_proj_depth_mask).abs().sum() / warped_proj_depth_mask.sum()

    # loss
    if return_visualization:
        # warped result
        warp_imgs = utils.visualize_img2img_result(
            pred_depth_img,
            warped_proj_depth_img,
            warped_proj_depth_mask)
        # projected point cloud
        prj_mat = get_projection_matrix(
            intrinsic_mat, extrinsic_init_mat, pose_mat)    # [3,4]
        ptcld[:, 3, :] = 1
        prj_pts = prj_mat[None, :, :] @ ptcld  # [B,3,N]
        prj_imgs = utils.visualize_img2ptc_result(
            pred_depth_img, prj_pts)
        return loss, warp_imgs, warped_proj_depth_mask, prj_imgs
    else:
        return loss


def debug(
    rgb_img, pred_depth_img,
    proj_depth_img, proj_depth_mask, ptcld,
    intrinsic_mat,
    pose_mat,   # [3,4]
    intrinsic_init_mat,
    extrinsic_init_mat,
    point_color_mode='depth',   # 'depth' or 'reflectivity'
    image_mode='individual',    # 'individual' or 'mode'
):
    assert proj_depth_img.size() == proj_depth_mask.size()

    # initialize the tensor shapes
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img.unsqueeze(0)  # [1,H,W,3]
        pred_depth_img = pred_depth_img.unsqueeze(0)
        proj_depth_img = proj_depth_img.unsqueeze(0)
        proj_depth_mask = proj_depth_mask.unsqueeze(0)
        ptcld = ptcld.unsqueeze(0)
    elif len(rgb_img.shape) == 4:
        pass
    else:
        raise ValueError(f'rgb_img.shape should be 3 or 4, but it is {rgb_img.shape}')

    # preprocessing and warping
    pred_depth_img, intrinsic_init_mat, extrinsic_init_mat = preprocessing(
        pred_depth_img, intrinsic_init_mat, extrinsic_init_mat
    )

    # projected point cloud
    intrinsic_mat, extrinsic_mat = get_projection_matrix(
        intrinsic_mat, extrinsic_init_mat, pose_mat,
        two_matrices=True)    # [3,4]
    prj_mat = intrinsic_mat @ extrinsic_mat
    tmp_ptcld = ptcld.clone()
    tmp_ptcld[:, 3, :] = 1
    prj_pts = prj_mat[None, :, :] @ tmp_ptcld   # [B,3,N]

    # define the projected point color values
    if point_color_mode == 'depth':
        pt_values = None
    elif point_color_mode == 'reflectivity':
        pt_values = ptcld[:, 3, :]

    # make the point-projected image
    prj_imgs = utils.visualize_img2ptc_result(
        rgb_img, prj_pts,
        pt_values=pt_values,
        mode=image_mode
    )

    # textured point cloud
    colors = get_colored_ptcld(rgb_img, prj_pts)    # [B,3,N]
    colors = colors[:, [2, 1, 0], :]    # BGR -> RGB
    pcd_with_rgb = np.concatenate([ptcld[:, :3, :], colors], axis=1)

    return prj_imgs, pcd_with_rgb, intrinsic_mat, extrinsic_mat


def get_projection_matrix(intrinsic_mat: torch.Tensor,
                          extrinsic_init_mat: torch.Tensor,
                          pose_mat: torch.Tensor,
                          two_matrices=False):
    intrinsic_mat = intrinsic_mat.detach()
    extra_row = torch.tensor(
        [0, 0, 0, 1], device=pose_mat.device).reshape(1, 4)
    pose_mat = pose_mat.detach()
    extrinsic_init_mat = torch.cat(
        [extrinsic_init_mat.detach(), extra_row], dim=0)

    if two_matrices:
        return intrinsic_mat, pose_mat @ extrinsic_init_mat
    else:
        return intrinsic_mat @ pose_mat @ extrinsic_init_mat


def pose_operation(
    points: torch.Tensor,   # [3,N]
    pose_mat: torch.Tensor  # [3,4]
) -> torch.Tensor:
    rot_mat = pose_mat[:, :3]   # [3,3]
    tr_vec = pose_mat[:, 3:]  # [3,1]
    return (rot_mat @ points) + tr_vec


def inv_pose_operation(
    points: torch.Tensor,   # [3,N]
    pose_mat: torch.Tensor  # [3,4]
) -> torch.Tensor:
    rot_mat = pose_mat[:, :3]   # [3,3]
    tr_vec = pose_mat[:, 3:]  # [3,1]
    return torch.inverse(rot_mat) @ (points - tr_vec)


def preprocess_depth_img(img: torch.Tensor,
                         mode: str = 'normalization') -> torch.Tensor:
    b, h, w = img.size()

    if mode == 'normalization':
        img = img.reshape(b, h * w)
        img_min = img.min(dim=1, keepdim=True)[0]
        img_max = img.max(dim=1, keepdim=True)[0]
        img = (
            (img - img_min) / (img_max - img_min).clamp(min=1e-12)
        ).reshape(b, h, w)
    elif mode == 'standardization':
        img = img.reshape(b, h * w)
        means = img.mean(dim=1, keepdim=True)
        std = img.std(dim=1, keepdim=True)
        img = ((img - means) / std).reshape(b, h, w)
    elif mode == 'softmax':
        img = img.reshape(b, h * w)
        img = F.softmax(img, dim=1).reshape(b, h, w)
        img = preprocess_depth_img(img, mode='normalization')
    elif mode == 'log_softmax':
        img = img.reshape(b, h * w)
        img = F.log_softmax(img, dim=1).reshape(b, h, w)
        img = preprocess_depth_img(img, mode='normalization')

    return img


def get_colored_ptcld(
    img: np.ndarray,
    prj_pts: np.ndarray,
):
    img = img.detach()  # [B,H,W,3]
    prj_pts = prj_pts.detach()   # [B,3,N]
    D = prj_pts[:, 2, :].cpu().numpy()
    U = prj_pts[:, 0, :].cpu().numpy() / D  # [B,N]
    V = prj_pts[:, 1, :].cpu().numpy() / D  # [B,N]

    batch_size = img.shape[0]
    img_h = img.shape[1]
    img_w = img.shape[2]
    num_pts = prj_pts.shape[-1]

    colors = []
    for b in range(batch_size):
        valid_mask = \
            (0 <= U[b]) & (U[b] < img_w) & \
            (0 <= V[b]) & (V[b] < img_h) & \
            (0 < D[b])

        color = np.zeros((num_pts, 3)) + 0.4
        color[valid_mask, :] = img[
            b,
            V[b, valid_mask].astype(int),
            U[b, valid_mask].astype(int), :] / 255.0
        colors.append(color.T)

    return np.array(colors)  # [B,3,N]
