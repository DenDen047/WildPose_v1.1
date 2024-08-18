import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from scipy.spatial.transform import Rotation
import numpy as np

from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SimpleModel(BaseModel):
    """The simplest model with MiDaS (https://pytorch.org/hub/intelisl_midas_v2/) as a depth estimator

    Args:
        init_cam_intrinsic (List): the initial camera intrinsic parameters
        init_cam_extrinsic (List): the initial 6DoF pose parameters from lidar to camera
        depth_model_type (str): the model type (default: DPT_Large)
        depth_image_size (tuple): the output depth image size [H,W].
    """

    def __init__(
        self,
        init_cam_intrinsic: List,  # fx, fy, cx, cy
        init_cam_extrinsic_R: List,  # q0, q1, q2, q3
        init_cam_extrinsic_t: List,  # tx, ty, tz
        depth_image_size=[720, 1280],
    ):
        super().__init__()
        self.depth_image_size = depth_image_size
        self.intrinsic_params = torch.nn.Parameter(
            torch.tensor(
                self.normalize_intrinsic_params(init_cam_intrinsic),
                dtype=torch.float32),
            requires_grad=True
        )
        self.extrinsic_params = torch.nn.Parameter(
            torch.tensor(init_cam_extrinsic_t + init_cam_extrinsic_R,
                         dtype=torch.float32),
            requires_grad=True
        )

    def forward(self):
        intrinsic_mat = self.preprocess_intrinsic_mat(self.intrinsic_params)
        pose_mat = self.pose_vec2mat(
            self.extrinsic_params,
            expand=False,
            rotation_mode='quat')

        return (
            self.denormalize_intrinsic_mat(intrinsic_mat),
            pose_mat
        )

    def normalize_intrinsic_params(self, intrinsics):
        def _norm(x, min, max):
            return (x - min) / (max - min)
        intrinsics[0] = _norm(intrinsics[0], 10000, 20000)   # fx
        intrinsics[1] = _norm(intrinsics[1], 10000, 20000)   # fy
        intrinsics[2] = _norm(
            intrinsics[2], 0, self.depth_image_size[0])    # cx
        intrinsics[3] = _norm(
            intrinsics[3], 0, self.depth_image_size[1])    # cy
        return intrinsics

    def denormalize_intrinsic_mat(self, intrinsics):
        def _denorm(x, min, max):
            return x * (max - min) + min
        intrinsics[0, 0] = _denorm(intrinsics[0, 0], 10000, 20000)   # fx
        intrinsics[1, 1] = _denorm(intrinsics[1, 1], 10000, 20000)   # fy
        intrinsics[0, 2] = _denorm(
            intrinsics[0, 2], 0, self.depth_image_size[0])    # cx
        intrinsics[1, 2] = _denorm(
            intrinsics[1, 2], 0, self.depth_image_size[1])    # cy
        return intrinsics

    def get_intrinsic_params(self):
        intrinsic_mat = self.denormalize_intrinsic_mat(
            self.preprocess_intrinsic_mat(self.intrinsic_params))
        return (
            intrinsic_mat[0, 0],
            intrinsic_mat[1, 1],
            intrinsic_mat[0, 2],
            intrinsic_mat[1, 2],
        )

    def get_extrinsic_params(self, extrinsic_init_mat: torch.Tensor):
        extrinsic_init_mat = extrinsic_init_mat.detach().cpu()
        extrinsic_pred_mat = self.pose_vec2mat(
            self.extrinsic_params, expand=False, rotation_mode='quat').detach().cpu()  # [3,4]

        # expand the matrix
        extra_row = torch.tensor([0, 0, 0, 1]).reshape(1, 4)
        extrinsic_pred_mat = torch.cat([extrinsic_pred_mat, extra_row], dim=0)
        extrinsic_init_mat = torch.cat([extrinsic_init_mat, extra_row], dim=0)

        extrinsic_mat = extrinsic_pred_mat @ extrinsic_init_mat.type(
            torch.FloatTensor)

        # translation
        translation = extrinsic_mat[:3, 3]
        # rotation matrix -> euler angle (ZYX)
        extrinsic_mat = extrinsic_mat.detach().cpu().numpy()
        r = Rotation.from_matrix(extrinsic_mat[:3, :3])
        quat = r.as_quat()    # radian

        return np.concatenate((translation, quat), axis=0)

    def preprocess_intrinsic_mat(self, vec) -> torch.Tensor:
        """The preprocessing of the intrinsic matrix for training

        Args:
            vec (array_lik): the intrinsic camera parameters

        Returns:
            A intrinsic matrix: [3,4]
        """
        vec = vec[None, :]
        fx = vec[:, 0]
        fy = vec[:, 1]
        cx = vec[:, 2]
        cy = vec[:, 3]

        zero = fx.detach() * 0
        one = zero.detach() + 1
        mat = torch.stack([fx, zero, cx,
                           zero, fy, cy,
                           zero, zero, one], dim=1).reshape(1, 3, 3)
        return mat.squeeze()

    def pose_vec2mat(
        self,
        vec,
        rotation_mode='quat',
        expand=False
    ) -> torch.Tensor:
        """Convert 6DoF parameters to transformation matrix.

        Args:
            vec: 6DoF parameters in the order of tx, ty, tz, q1, q2, q3
            expand (bool): if True, expand the matrix to [4, 4]

        Returns:
            A transformation matrix -- [3,4] or [4,4]
        """
        translation = vec[None, :3, None]  # [1, 3, 1]
        rot = vec[None, 3:]  # [1, 3]
        if rotation_mode == 'euler':
            rot_mat = euler2mat(rot)  # [B, 3, 3]
        elif rotation_mode == 'quat':
            rot_mat = quat2mat(rot)  # [B, 3, 3]
        transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
        if expand:
            B = transform_mat.size(0)
            new_row = torch.zeros((B, 1, 4), device=transform_mat.device)
            new_row[:, 0, 3] = 1      # [B, 1, 4]
            transform_mat = torch.cat([transform_mat, new_row], dim=1)
        return transform_mat.squeeze()


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourth is then computed to have a norm of 1 -- size = [B, 4]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach() * 0 + 1, quat], dim=1)
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
        2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
        2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
    ], dim=1).reshape(B, 3, 3)
    return rotMat
