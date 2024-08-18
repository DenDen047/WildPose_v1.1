import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.extrinsic_init_mat = None

        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.camera_param_metrics = MetricTracker(
        #     'fx', 'fy', 'cx', 'cy', 'tx', 'ty', 'tz', 'yaw', 'pitch', 'roll',
        #     writer=self.writer
        # )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (rgb_img, prj_img, prj_mask, ptcld, intrinsic_init_mat,
                        extrinsic_init_mat) in enumerate(self.data_loader):
            rgb_img = rgb_img.to(self.device)  # [B,C,H',W']
            prj_img = prj_img.to(self.device)  # [B,H,W]
            prj_mask = prj_mask.to(self.device)  # [B,H,W]
            intrinsic_init_mat = intrinsic_init_mat.to(
                self.device)  # [B,3,3]
            self.extrinsic_init_mat = extrinsic_init_mat.to(
                self.device)  # [B,3,4]
            ptcld = ptcld.to(self.device)

            self.optimizer.zero_grad()
            pred_depth_img, intrinsic_mat, pose_mat = self.model(rgb_img)

            loss = self.criterion(pred_depth_img, prj_img, prj_mask, ptcld,
                                  intrinsic_mat, pose_mat,
                                  intrinsic_init_mat,
                                  self.extrinsic_init_mat,)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__, met(
                        pred_depth_img, ptcld, prj_mat))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self._add_images('train', rgb_img, pred_depth_img)

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{f'val_{k}': v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(self.valid_metrics.result()['loss'])
            else:
                self.lr_scheduler.step()

        self.show_result()
        return log

    def show_result(self):
        # record the camera parameters
        intrinsic_params = self.model.get_intrinsic_params()
        for key, idx in zip(
            ['fx', 'fy', 'cx', 'cy'],
            [0, 1, 2, 3]
        ):
            self.logger.info(
                '    {:15s}: {}'.format(
                    str(key), intrinsic_params[idx]))

        extrinsic_params = self.model.get_extrinsic_params(
            self.extrinsic_init_mat.mean(dim=0))
        for key, idx in zip(
            ['tx', 'ty', 'tz', 'q0', 'q1', 'q2', 'q3'],
            [0, 1, 2, 3, 4, 5]
        ):
            self.logger.info(
                '    {:15s}: {}'.format(
                    str(key), extrinsic_params[idx]))
        return

    def _valid_epoch(self, epoch, save_output_images=False):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (rgb_img, prj_img, prj_mask, ptcld, intrinsic_init_mat,
                            extrinsic_init_mat) in enumerate(self.data_loader):
                rgb_img = rgb_img.to(self.device)  # [B,C,H',W']
                prj_img = prj_img.to(self.device)  # [B,H,W]
                prj_mask = prj_mask.to(self.device)  # [B,H,W]
                intrinsic_init_mat = intrinsic_init_mat.to(
                    self.device)  # [B,3,3]
                extrinsic_init_mat = extrinsic_init_mat.to(
                    self.device)  # [B,3,4]
                ptcld = ptcld.to(self.device)

                pred_depth_img, intrinsic_mat, pose_mat = self.model(rgb_img)
                loss, warp_imgs, warp_masks, prj_imgs = self.criterion(pred_depth_img,
                                                                       prj_img, prj_mask, ptcld,
                                                                       intrinsic_mat, pose_mat,
                                                                       intrinsic_init_mat,
                                                                       extrinsic_init_mat,
                                                                       return_visualization=True)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(pred_depth_img, ptcld, prj_mat))

                if save_output_images:
                    self.logger.info(f'Loss: {loss}')
                    for b in range(len(warp_imgs)):
                        cv2.imwrite(
                            f'{b}_input.png',
                            cv2.cvtColor(
                                np.transpose(
                                    rgb_img[b].numpy(), (1, 2, 0)) * 255,
                                cv2.COLOR_BGR2RGB))
                        cv2.imwrite(
                            f'{b}_warped_result.png',
                            cv2.cvtColor(np.transpose(warp_imgs[b].numpy(), (1, 2, 0)), cv2.COLOR_BGR2RGB))
                        cv2.imwrite(
                            f'{b}_warped_mask.png',
                            cv2.cvtColor(warp_masks[b, :, :].numpy() * 255, cv2.COLOR_GRAY2RGB))
                        cv2.imwrite(
                            f'{b}_prjed_result.png',
                            cv2.cvtColor(np.transpose(prj_imgs[b].numpy(), (1, 2, 0)), cv2.COLOR_BGR2RGB))
                    break   # DEBUG
                self._add_images(
                    'valid',
                    rgb_img,
                    pred_depth_img,
                    warp_imgs,
                    prj_imgs)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def show_best_result(self):
        best_path = self.get_model_best()
        self._resume_checkpoint(best_path)
        self.show_result()

    def _add_images(self, label: str, rgb_img, pred_depth_img,
                    warp_imgs=None, prj_imgs=None):
        self.writer.add_image(
            f'{label}_input', make_grid(
                rgb_img.cpu(), nrow=8, normalize=True))
        self.writer.add_image(
            f'{label}_depthnet_out', make_grid(
                pred_depth_img.unsqueeze(1).cpu(),
                nrow=8, normalize=True))
        if warp_imgs is not None:
            self.writer.add_image(
                f'{label}_warp_imgs', make_grid(
                    warp_imgs, nrow=8, normalize=True))
        if prj_imgs is not None:
            self.writer.add_image(
                f'{label}_prj_imgs', make_grid(
                    prj_imgs, nrow=8, normalize=True))

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
