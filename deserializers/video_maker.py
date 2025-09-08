import os
import sys
import shutil
import glob
import argparse
import cv2
import numpy as np
import subprocess
import collections
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--force', action='store_true')
parser.add_argument('--meas_dir', type=str, default='~/WildPose_v1.1/wildpose/record/ecal_meas/2023-08-16_20-27-46.008_wildpose_v1.1')
parser.add_argument('--cam_context_fpath', type=str, default='~/WildPose_v1.1/wildpose/record/cam_context_2023-08-16_19-39-12.bin')
args = parser.parse_args()

raw_dir = os.path.join('.', 'tmp_raw/')
rgb_dir = os.path.join(args.meas_dir, 'rgb/')
lidar_dir = os.path.join(args.meas_dir, 'lidar/')
sync_rgb_dir = os.path.join(args.meas_dir, 'sync_rgb/')
tmp_video_path = os.path.join('.', 'tmp_video.mp4')
video_path = os.path.join(args.meas_dir, 'video.mp4')
imu_json_path = os.path.join(args.meas_dir, 'imu.json')


def get_timestamp_from_fpath(fpath: str) -> float:
    fname = os.path.splitext(os.path.basename(fpath))[0]
    # get timestamp
    fname = fname.split('_')
    msg_id = '_'.join(fname[:-2])
    timestamp = float(fname[-2] + '.' + fname[-1])

    return timestamp


def make_sync_rgb(sync_rgb_dir: str, rgb_dir: str, pcd_dir: str) -> str:
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


def main():
    # priotized_cmd = ['sudo', 'nice', '-n', '-20']
    priotized_cmd = []

    # reconstruction
    if args.force or not os.path.exists(rgb_dir):
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(rgb_dir, exist_ok=True)
        cmd = priotized_cmd + [
            './ecal_sample_ximea',
            args.meas_dir,
            'rt/image_raw',
            args.cam_context_fpath,
            rgb_dir, raw_dir
        ]
        _ = subprocess.run(cmd, check=True)
    if args.force or not os.path.exists(lidar_dir):
        os.makedirs(lidar_dir, exist_ok=True)
        cmd = priotized_cmd + [
            './ecal_sample_lidar',
            args.meas_dir,
            'rt/livox/lidar',
            lidar_dir,
        ]
        print('Generate PCD...')
        _ = subprocess.run(cmd, check=True)
        print('Done!')
    if args.force or not os.path.exists(imu_json_path):
        cmd = priotized_cmd + [
            './ecal_sample_livox_imu',
            args.meas_dir,
            'rt/livox/imu',
            imu_json_path,
        ]
        print('Generate IMU json...')
        _ = subprocess.run(cmd, check=True)
        print('Done!')

    # load files
    img_fpaths = sorted(glob.glob(os.path.join(rgb_dir, '*.jpeg')))

    # sync images
    if args.force or not os.path.exists(sync_rgb_dir):
        print('Making sync_rgb dir...')
        make_sync_rgb(sync_rgb_dir, rgb_dir, lidar_dir)
        print('Done!')

    # make a video from color images
    if args.force or not os.path.exists(video_path):
        frame = cv2.imread(img_fpaths[0])
        height, width, layers = frame.shape

        video = cv2.VideoWriter(
            tmp_video_path,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=170,
            frameSize=(width, height)
        )

        print('Making a colour video...')
        for img_fpath in tqdm(img_fpaths):
            video.write(cv2.imread(img_fpath))
        print('Done!')

        cv2.destroyAllWindows()
        video.release()

        print('Copying the video to the destination...')
        shutil.copyfile(src=tmp_video_path, dst=video_path)
        print('Done!')


if __name__ == "__main__":
    main()
