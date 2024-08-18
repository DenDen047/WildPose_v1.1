import argparse
import collections
import os
import json
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import torch
import torch.utils.data as data
import open3d as o3d
from typing import Tuple

from parse_config import ConfigParser
import data_loader.data_loaders as module_data
import model.model as module_arch
import model.loss as module_loss
import model.metric as module_metric
import utils

import cv2
from torchvision.utils import make_grid
from base import BaseTrainer


class VisualDebugger(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = utils.inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.extrinsic_init_mat = None

    def visual_debug(self):
        """perform visual debugging on the first batch of the first epoch
        """
        self.model.eval()
        with torch.no_grad():
            # get data of the first epoch
            rgb_img, depth_img, prj_img, prj_mask, ptcld, intrinsic_init_mat, extrinsic_init_mat = next(iter(self.data_loader))

            # the first values of the initial intrinsic and extrinsic matrices
            intrinsic_mat, pose_mat = self.model()
            init_intrinsic_mat = intrinsic_mat.clone()
            init_pose_mat = pose_mat.clone()

        init_diff_pixel = 50
        init_diff_focal_length = 1000
        init_diff_angle = 0.1  # degree
        init_diff_trans = 0.01  # meter
        diff_pixel = None
        diff_focal_length = None
        diff_angle = None
        diff_trans = None

        def _reset_params():
            nonlocal diff_pixel
            nonlocal diff_focal_length
            nonlocal diff_angle
            nonlocal intrinsic_mat
            nonlocal pose_mat
            nonlocal diff_trans
            diff_pixel = init_diff_pixel
            diff_focal_length = init_diff_focal_length
            diff_angle = init_diff_angle
            diff_trans = init_diff_trans
            intrinsic_mat = init_intrinsic_mat.clone()
            pose_mat = init_pose_mat.clone()
        _reset_params()

        update_intrinsic = True
        viewpoint_params = None
        b = 0
        Rt = None
        K = None
        batch_size = rgb_img.shape[0]
        idx_color_mode = 0
        point_color_modes = ['depth', 'reflectivity']
        while True:
            if update_intrinsic:
                prj_imgs, pcd_with_rgb, K, Rt = self.criterion(
                    rgb_img[b], depth_img[b],
                    prj_img[b], prj_mask[b], ptcld[b],
                    intrinsic_mat, pose_mat,
                    intrinsic_init_mat,
                    extrinsic_init_mat,
                    point_color_mode=point_color_modes[idx_color_mode % len(point_color_modes)],
                    image_mode='individual' if type(b) == int else 'merge'
                )

                result_img = np.transpose(prj_imgs[0].numpy(), (1, 2, 0))
                result_img = cv2.convertScaleAbs(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                window_name = 'debugging (updated)' if update_intrinsic else 'debugging'
                cv2.imshow(window_name, result_img)
            # Small delay to allow for continuous updates
            key = cv2.waitKey(10)

            # Adjust matrices based on key press
            update_intrinsic = True
            # change the intrinsic parameters
            if key == 2:  # Left arrow key
                intrinsic_mat[0, 2] -= diff_pixel
            elif key == 3:  # Right arrow key
                intrinsic_mat[0, 2] += diff_pixel
            elif key == 0:  # Up arrow key
                intrinsic_mat[1, 2] -= diff_pixel
            elif key == 1:  # Down arrow key
                intrinsic_mat[1, 2] += diff_pixel
            elif key == 43:  # "+"
                intrinsic_mat[0, 0] += diff_focal_length
                intrinsic_mat[1, 1] += diff_focal_length
            elif key == 45:  # "-"
                intrinsic_mat[0, 0] -= diff_focal_length
                intrinsic_mat[1, 1] -= diff_focal_length
            # change the extrinsic parameters
            elif key == 0x77:   # "w"
                rot = R.from_euler(
                    'xyz', [-diff_angle, 0, 0], degrees=True).as_matrix()
                pose_mat[:3, :3] = torch.Tensor(rot) @ pose_mat[:3, :3]
            elif key == 0x73:   # "s"
                rot = R.from_euler(
                    'xyz', [diff_angle, 0, 0], degrees=True).as_matrix()
                pose_mat[:3, :3] = torch.Tensor(rot) @ pose_mat[:3, :3]
            elif key == 0x61:   # "a"
                rot = R.from_euler(
                    'xyz', [0, diff_angle, 0], degrees=True).as_matrix()
                pose_mat[:3, :3] = torch.Tensor(rot) @ pose_mat[:3, :3]
            elif key == 0x64:   # "d"
                rot = R.from_euler(
                    'xyz', [0, -diff_angle, 0], degrees=True).as_matrix()
                pose_mat[:3, :3] = torch.Tensor(rot) @ pose_mat[:3, :3]
            elif key == 0x71:   # "q"
                rot = R.from_euler(
                    'xyz', [0, 0, diff_angle], degrees=True).as_matrix()
                pose_mat[:3, :3] = torch.Tensor(rot) @ pose_mat[:3, :3]
            elif key == 0x65:   # "e"
                rot = R.from_euler(
                    'xyz', [0, 0, -diff_angle], degrees=True).as_matrix()
                pose_mat[:3, :3] = torch.Tensor(rot) @ pose_mat[:3, :3]
            elif key == 0x57:   # "W"
                d = torch.Tensor([0, diff_trans, 0])
                pose_mat[:, 3] = pose_mat[:, 3] + d
            elif key == 0x53:   # "S"
                d = torch.Tensor([0, -diff_trans, 0])
                pose_mat[:, 3] = pose_mat[:, 3] + d
            elif key == 0x41:   # "A"
                d = torch.Tensor([diff_trans, 0, 0])
                pose_mat[:, 3] = pose_mat[:, 3] + d
            elif key == 0x44:   # "D"
                d = torch.Tensor([-diff_trans, 0, 0])
                pose_mat[:, 3] = pose_mat[:, 3] + d
            # reset
            elif key == 0x30:  # "0"
                _reset_params()
            # change the diff scale
            elif key == 62:  # ">"
                diff_pixel *= 2
                diff_focal_length *= 2
                diff_angle *= 2
            elif key == 60:  # "<"
                diff_pixel /= 2
                diff_focal_length /= 2
                diff_angle /= 2
            # merge/undo frame in the batch
            elif key == ord('m'):
                if type(b) == int:
                    merge_size = 10
                    if merge_size < batch_size - b:
                        b = [i for i in range(b, b+merge_size)]
                    else:
                        b = [i for i in range(b, batch_size)]
                else:
                    b = b[0]
            # change the coloring
            elif key == ord('c'):  # "c"
                idx_color_mode += 1
            # next frame
            elif key == 0x6E:   # "n"
                if type(b) is not list:
                    b = (b + 1) % batch_size
            # save the parameters
            elif key == 13:  # "return"
                # the intrinsic parameters as a list
                intrinsics = [float(K[0, 0]), float(K[1, 1]),
                                float(K[0, 2]), float(K[1, 2])]
                # the extrinsic parameters as a quaternion
                extrinsics_R = Rt[:3, :3].tolist()
                extrinsics_t = Rt[:, 3].tolist()
                # export as a JSON file
                json_object = json.dumps(
                    {
                        'intrinsics': intrinsics,
                        'extrinsics_R': extrinsics_R,
                        'extrinsics_t': extrinsics_t,
                    },
                    indent=4)
                json_fpath = os.path.join(
                    config.config['data_loader']['args']['data_dir'],
                    'manual_calibration.json'
                )
                with open(json_fpath, 'w') as f:
                    f.write(json_object)
                print(f'Saved to {json_fpath}')
            # 3D visualization
            elif key == 0x76:   # "v"
                viewpoint_params = utils.visualize_open3d(
                    pcd_with_rgb[0].T, parameters=viewpoint_params)
            elif key == -1:
                update_intrinsic = False    # no key pressed
            else:
                print(f'Error: "{key}" is not defined.')
                update_intrinsic = False

            if update_intrinsic:
                print(intrinsic_mat)
                print(pose_mat)
                print('')

        # closing all open windows
        cv2.destroyAllWindows()


def main(config: ConfigParser):
    device, device_ids = utils.prepare_device(config['n_gpu'])

    # define the logger
    logger = config.get_logger('train')

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # setup data_loader instances
    dataset = config.init_obj(
        'data_loader',
        module_data,
        device=device)
    cfg_data_loader = config['data_loader']
    batch_size = cfg_data_loader['batch_size']
    num_workers = cfg_data_loader['num_workers']
    shuffle = cfg_data_loader['shuffle']
    data_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, collate_fn=None)

    # prepare for (multi-device) GPU training
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if config['mode'] == 'try_best':
        best_path = config['best_model']
        model.load_state_dict(
            torch.load(
                best_path,
                map_location=device)['state_dict'])

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing
    # lr_scheduler for disabling scheduler
    trainable_params = filter(
        lambda p: p.requires_grad,
        model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj(
        'lr_scheduler',
        torch.optim.lr_scheduler,
        optimizer)

    trainer = VisualDebugger(model, criterion, metrics, optimizer,
                             config=config,
                             device=device,
                             data_loader=data_loader,
                             lr_scheduler=lr_scheduler)

    # show the projected image with initial camera parameters
    trainer.visual_debug()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in
    # json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'],
                   type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size'),
    ]
    config = ConfigParser.from_args(args, options)

    # random seeds for reproducibility
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config['seed'])

    # torch setting
    torch.autograd.set_detect_anomaly(True)

    main(config)
