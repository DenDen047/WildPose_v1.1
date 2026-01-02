import argparse
import collections
import glob
import os
import shutil
import subprocess

import numpy as np
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--force", action="store_true")
parser.add_argument(
    "--meas_dir", type=str, default="/mnt/vault/WildPose_v1.1/Cheetah/2022-12-03_013"
)
parser.add_argument(
    "--cam_context_fpath",
    type=str,
    default="",
)
parser.add_argument(
    "--skip_raw",
    action="store_true",
    help="Skip raw image generation (pass 'none' to deserializer)",
)
parser.add_argument(
    "--image_format",
    type=str,
    default="webp",
    choices=["webp", "jpeg"],
    help="Output image format for RGB images (default: webp)",
)
args = parser.parse_args()

raw_dir = os.path.join(args.meas_dir, "raw/")
rgb_dir = os.path.join(args.meas_dir, "rgb/")
lidar_dir = os.path.join(args.meas_dir, "lidar/")
sync_rgb_dir = os.path.join(args.meas_dir, "sync_rgb/")
tmp_video_path = os.path.join(".", "tmp_video.mp4")
video_path = os.path.join(args.meas_dir, "video.mp4")
imu_json_path = os.path.join(args.meas_dir, "imu.json")


def get_timestamp_from_fpath(fpath: str) -> float:
    fname = os.path.splitext(os.path.basename(fpath))[0]
    # get timestamp
    fname = fname.split("_")
    msg_id = "_".join(fname[:-2])
    timestamp = float(fname[-2] + "." + fname[-1])

    return timestamp


def make_sync_rgb(sync_rgb_dir: str, rgb_dir: str, pcd_dir: str) -> str:
    # Support both webp and jpeg formats
    img_fpaths = sorted(
        glob.glob(os.path.join(rgb_dir, "*.jpeg"))
        + glob.glob(os.path.join(rgb_dir, "*.webp"))
    )
    pcd_fpaths = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))

    # load timestamps from file paths
    img_timestamps = [get_timestamp_from_fpath(fpath) for fpath in img_fpaths]
    pcd_timestamps = [get_timestamp_from_fpath(fpath) for fpath in pcd_fpaths]

    # get the image file paths synchronized PCD files
    sync_img_fpaths = []
    for pcd_timestamp in pcd_timestamps:
        idx = np.argmin([abs(x - pcd_timestamp) for x in img_timestamps])
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
            dst_fpath = os.path.join(sync_rgb_dir, os.path.basename(src_fpath))
            shutil.copyfile(src=src_fpath, dst=dst_fpath)

    return sync_rgb_dir


def make_vfr_video(
    img_fpaths: list[str],
    output_path: str,
    tmp_video_path: str,
) -> None:
    """タイムスタンプに基づいたVFR動画を生成する。

    最初のフレームのタイムスタンプを0秒として、
    各フレームの相対的なタイミングを再現する。

    Parameters
    ----------
    img_fpaths : list[str]
        画像ファイルパスのリスト（ソート済み）
    output_path : str
        出力動画パス
    tmp_video_path : str
        一時ファイルパス（concat list用のディレクトリとしても使用）
    """
    # Check FFmpeg availability
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        logger.error("FFmpeg not found in PATH")
        raise RuntimeError("FFmpeg not found")

    if len(img_fpaths) < 2:
        logger.error(f"Need at least 2 images to create video, got {len(img_fpaths)}")
        raise ValueError("Insufficient images")

    # Extract timestamps and convert to relative time (first frame = 0s)
    timestamps = [get_timestamp_from_fpath(fp) for fp in img_fpaths]
    base_timestamp = timestamps[0]
    relative_timestamps = [t - base_timestamp for t in timestamps]

    # Log FPS statistics
    deltas = [
        relative_timestamps[i + 1] - relative_timestamps[i]
        for i in range(len(relative_timestamps) - 1)
    ]
    valid_deltas = [d for d in deltas if d > 0]
    if len(valid_deltas) > 0:
        fps_values = [1.0 / d for d in valid_deltas]
        logger.info(
            f"FPS stats - min: {min(fps_values):.1f}, max: {max(fps_values):.1f}, "
            f"median: {np.median(fps_values):.1f}"
        )

    logger.info(
        f"Video duration: {relative_timestamps[-1]:.3f}s, frames: {len(img_fpaths)}"
    )

    # Create concat demuxer file for FFmpeg
    tmp_dir = os.path.dirname(tmp_video_path) or "."
    concat_file = os.path.join(tmp_dir, "concat_list.txt")

    with open(concat_file, "w") as f:
        for i, fpath in enumerate(img_fpaths):
            f.write(f"file '{os.path.abspath(fpath)}'\n")

            if i < len(img_fpaths) - 1:
                duration = relative_timestamps[i + 1] - relative_timestamps[i]
                # Clamp invalid durations
                if duration <= 0:
                    logger.warning(
                        f"Frame {i}: non-positive duration {duration:.6f}s, using 0.001s"
                    )
                    duration = 0.001
                elif duration > 1.0:
                    logger.warning(
                        f"Frame {i}: large gap {duration:.3f}s, clamping to 1.0s"
                    )
                    duration = 1.0
            else:
                # Last frame: use average duration
                avg_duration = relative_timestamps[-1] / (len(relative_timestamps) - 1)
                duration = avg_duration

            f.write(f"duration {duration:.6f}\n")

    # Generate VFR video with FFmpeg
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-vsync",
        "vfr",
        tmp_video_path,
    ]

    logger.info("Making a VFR video with FFmpeg...")
    subprocess.run(cmd, check=True)

    # Cleanup and move to destination
    os.remove(concat_file)
    shutil.move(src=tmp_video_path, dst=output_path)
    logger.info("Done!")


def main():
    # priotized_cmd = ['sudo', 'nice', '-n', '-20']
    priotized_cmd = []

    # check if the cam_context_fpath is valid
    cam_context_fpath = args.cam_context_fpath
    if len(cam_context_fpath) == 0:
        bin_fpaths = sorted(glob.glob(os.path.join(args.meas_dir, "*.bin")))
        if len(bin_fpaths) == 0:
            logger.error("No bin files found")
            exit(1)
        cam_context_fpath = bin_fpaths[0]
        logger.warning(
            f"Camera context file is empty, using the first bin file: {cam_context_fpath}"
        )
    elif not os.path.exists(cam_context_fpath):
        logger.error(f"Camera context file {cam_context_fpath} does not exist")
        exit(1)
    else:
        logger.info(f"Camera context file: {cam_context_fpath}")

    # reconstruction
    if args.force or not os.path.exists(rgb_dir):
        if not args.skip_raw:
            os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(rgb_dir, exist_ok=True)

        # Determine raw output path
        raw_output_path = "none" if args.skip_raw else raw_dir

        cmd = priotized_cmd + [
            "./ecal_sample_ximea",
            args.meas_dir,
            "rt/image_raw",
            cam_context_fpath,
            rgb_dir,
            raw_output_path,
            args.image_format,
        ]
        logger.info(
            f"Running deserializer with image_format={args.image_format}, skip_raw={args.skip_raw}"
        )
        _ = subprocess.run(cmd, check=True)
    if args.force or not os.path.exists(lidar_dir):
        os.makedirs(lidar_dir, exist_ok=True)
        cmd = priotized_cmd + [
            "./ecal_sample_lidar",
            args.meas_dir,
            "rt/livox/lidar",
            lidar_dir,
        ]
        logger.info("Generate PCD...")
        _ = subprocess.run(cmd, check=True)
        logger.info("Done!")
    if args.force or not os.path.exists(imu_json_path):
        cmd = priotized_cmd + [
            "./ecal_sample_livox_imu",
            args.meas_dir,
            "rt/livox/imu",
            imu_json_path,
        ]
        logger.info("Generate IMU json...")
        _ = subprocess.run(cmd, check=True)
        logger.info("Done!")

    # load files (support both webp and jpeg formats)
    img_fpaths = sorted(
        glob.glob(os.path.join(rgb_dir, "*.jpeg"))
        + glob.glob(os.path.join(rgb_dir, "*.webp"))
    )

    # sync images
    if args.force or not os.path.exists(sync_rgb_dir):
        logger.info("Making sync_rgb dir...")
        make_sync_rgb(sync_rgb_dir, rgb_dir, lidar_dir)
        logger.info("Done!")

    # make a video from color images
    if args.force or not os.path.exists(video_path):
        make_vfr_video(img_fpaths, video_path, tmp_video_path)


if __name__ == "__main__":
    main()
