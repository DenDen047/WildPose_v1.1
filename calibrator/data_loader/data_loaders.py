import os
import torch.utils.data as data
import numpy as np
from scipy import ndimage
from numpy.linalg import inv
from typing import List, Dict
from PIL import Image
import glob
import cv2
import shutil
import collections
from scipy.interpolate import griddata
import torch
import torch.nn.functional as F

from utils import get_intrinsic_mat, get_extrinsic_mat


class EcalDataset(data.Dataset):
    """The simple dataset loader converted from ecalmeas.

    Args:
        TODO
    """

    def __init__(
        self,
        data_dir: str,
        init_cam_intrinsic: List,
        init_cam_extrinsic_R: List,
        init_cam_extrinsic_t: List,
        device,
        disable_depth_map: bool = False,
    ):
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, 'sync_rgb')
        self.pcd_dir = os.path.join(self.data_dir, 'lidar')
        self.device = device

        # check the image dir
        if not os.path.exists(self.img_dir):
            self.img_dir = self.make_sync_rgb(
                os.path.join(self.data_dir, 'rgb'),
                self.pcd_dir
            )

        # get the projection matrix (numpy)
        self.intrinsic_mat = get_intrinsic_mat(init_cam_intrinsic)  # [3,3]
        self.extrinsic_mat = get_extrinsic_mat(
            init_cam_extrinsic_R, init_cam_extrinsic_t)  # [3,4]
        self.proj_mat = self.intrinsic_mat @ self.extrinsic_mat  # [3,4]

        # get the data files
        self.samples = self.crawl_folders(self.img_dir, self.pcd_dir)

        # prepare the depth net
        self.disable_depth_map = disable_depth_map
        if not self.disable_depth_map:
            self.midas = torch.hub.load(
                "intel-isl/MiDaS",
                'DPT_Large',    # DPT_Hybrid
                source='github')
            self.midas.eval()
            for param in self.midas.parameters():
                param.requires_grad = False
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.cam_transform = midas_transforms.dpt_transform

    def make_sync_rgb(self, rgb_dir: str, pcd_dir: str) -> str:
        sync_rgb_dir = os.path.join(self.data_dir, 'sync_rgb')
        img_fpaths = sorted(
            glob.glob(os.path.join(rgb_dir, '*.jpeg')))
        pcd_fpaths = sorted(
            glob.glob(os.path.join(pcd_dir, '*.pcd')))

        # get images corresponding with the pcd files
        def _get_timestamp_from_fpath(fpath: str) -> float:
            fname = os.path.splitext(os.path.basename(fpath))[0]
            # get timestamp
            fname = fname.split('_')
            msg_id = '_'.join(fname[:-2])
            timestamp = float(fname[-2] + '.' + fname[-1])
            return timestamp

        # load image file paths
        img_timestamps = []
        for fpath in img_fpaths:
            timestamp = _get_timestamp_from_fpath(fpath)
            img_timestamps.append(timestamp)
        # load PCD file paths
        pcd_timestamps = []
        for fpath in pcd_fpaths:
            timestamp = _get_timestamp_from_fpath(fpath)
            pcd_timestamps.append(timestamp)

        # get the image file paths synchronized PCD files
        sync_img_fpaths = []
        for pcd_timestamp in pcd_timestamps:
            idx = np.argmin([abs(x - pcd_timestamp)
                            for x in img_timestamps])
            sync_img_fpaths.append(img_fpaths[idx])

        sync_img_counter = collections.Counter(sync_img_fpaths)
        sync_img_fpaths = sorted(list(set(sync_img_fpaths)))

        # move the files to sync_rgb dir
        os.makedirs(sync_rgb_dir, exist_ok=True)
        # fill sync_rgb directory
        for src_fpath in sync_img_fpaths:
            # make the different names for duplicate names
            if sync_img_counter[src_fpath] > 1:
                for i in range(sync_img_counter[src_fpath]):
                    basename = os.path.basename(src_fpath)
                    name, extension = os.path.splitext(basename)
                    new_file_name = f"{name}_{i}{extension}"
                    dst_fpath = os.path.join(sync_rgb_dir, new_file_name)
                    shutil.copyfile(src=src_fpath, dst=dst_fpath)
            else:
                dst_fpath = os.path.join(
                    sync_rgb_dir, os.path.basename(src_fpath))
                shutil.copyfile(src=src_fpath, dst=dst_fpath)

        return sync_rgb_dir

    def crawl_folders(self, img_dir: str, pcd_dir: str) -> List[Dict]:
        img_fpaths = sorted(
            glob.glob(os.path.join(img_dir, '*.jpeg')))
        pcd_fpaths = sorted(
            glob.glob(os.path.join(pcd_dir, '*.pcd')))

        # assert len(pcd_fpaths) == len(img_fpaths)

        pcd_fpaths = sorted(pcd_fpaths)
        img_fpaths = sorted(img_fpaths)

        samples = []
        for i in range(len(img_fpaths)):
            samples.append({
                'cam_img': img_fpaths[i],
                'lidar_pcd': pcd_fpaths[i],
            })

        return samples

    def load_img_as_float(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img  # [C, H, W]

    def load_pcd(self, path):
        with open(path, "r") as pcd_file:
            lines = [line.strip().split(" ") for line in pcd_file.readlines()]
        is_data = False
        data = []
        for line in lines:
            if line[0] == "DATA":
                is_data = True
            elif is_data:
                x = float(line[0])
                y = float(line[1])
                z = float(line[2])
                intensity = float(line[3])
                data.append([x, y, z, intensity])
        return np.asarray(data, dtype=np.float32).transpose()

    def ptcld_to_depth_image(
        self,
        ptcld: np.ndarray,  # [4,N]
        img_h: int, img_w: int,
    ):
        """make the projection image from 3d point cloud

        Args:
            ptcld (np.array): point cloud -- [4,N]
            img_h (int): the height of the output image
            img_w (int): the width of the output image

        Returns:
            np.array: a projected image -- [H,W]
            np.array: a image mask -- [H,W]
        """
        _, N = ptcld.shape
        points_3d = ptcld[:3, :]    # [3,N]

        # projecting 3d points onto the image plane
        proj_points_2d = self.proj_mat @ np.concatenate(
            [points_3d, np.ones((1, N))], axis=0)   # [2+1, N]

        x = proj_points_2d[0, :] / proj_points_2d[2, :]
        y = proj_points_2d[1, :] / proj_points_2d[2, :]

        # rendering the image and mask
        bins = [img_w, img_h]  # [img_w, img_h]
        range_xy = [[0., img_w],   # x-axis range
                    [0., img_h]]   # y-axis range

        # Calculate the sum of power returns that fall into the same 2D image
        # pixel
        power_sum, _, _ = np.histogram2d(
            x=x, y=y,
            bins=bins,
            weights=proj_points_2d[2, :], normed=False,
            range=range_xy
        )
        # Calculate the number of points in each pixel
        power_count, _, _ = np.histogram2d(
            x=x, y=y,
            bins=bins,
            range=range_xy
        )
        # Calculate the mean of power return in each pixel.
        # histogram2d does either sums or finds the number of points, no
        # average.
        prj_img = np.divide(
            power_sum, power_count,
            out=np.zeros_like(power_sum), where=power_count != 0
        )
        prj_img = np.transpose(prj_img).astype(np.float32)  # / 255.
        prj_mask = (prj_img > 0).astype(np.float32)

        # # dilate the mask
        # for _ in range(1):
        #     prj_mask = ndimage.binary_dilation(
        #         prj_mask,
        #         [[True, True, True],
        #          [True, True, True],
        #          [True, True, True]])

        # nearest image
        grid_x, grid_y = np.mgrid[0:img_w, 0:img_h]
        grid_img = griddata(
            points=np.concatenate(
                [x[:, np.newaxis], y[:, np.newaxis]], axis=1),
            values=proj_points_2d[2, :],    # depth
            xi=(grid_x, grid_y),
            method='nearest',
            fill_value=0,
        )
        prj_img = np.nan_to_num(grid_img.T)    # nan to zero

        return prj_img, prj_mask

    def __getitem__(self, index):
        img_path = self.samples[index]['cam_img']
        pcd_path = self.samples[index]['lidar_pcd']

        # load data
        rgb_img = self.load_img_as_float(img_path)  # [C,H,W]
        ptcld = self.load_pcd(pcd_path)   # [4,N]

        # generate the depth image from the point cloud
        print('point cloud to depth image...')
        h, w, _ = rgb_img.shape
        prj_img, prj_mask = self.ptcld_to_depth_image(ptcld, h, w)
        print('Done!')

        if not self.disable_depth_map:
            # transform for the image
            img_input = self.cam_transform(rgb_img)
            if len(img_input.shape) == 4:
                img_input = img_input[0, :, :, :]

            # depth net
            prediction = self.midas(
                img_input.unsqueeze(0).to(self.device))  # [B,H',W']
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=[h, w],
                mode="nearest",
            )[0, 0, :, :]
            depth_img = prediction.numpy(force=True)
        else:
            depth_img = rgb_img

        return (
            np.float32(rgb_img),
            np.float32(depth_img),
            np.float32(prj_img),
            np.float32(prj_mask),
            np.float32(ptcld),
            np.float32(self.intrinsic_mat),
            np.float32(self.extrinsic_mat)
        )

    def __len__(self):
        return len(self.samples)
