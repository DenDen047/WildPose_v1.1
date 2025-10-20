#!/usr/bin/env python3

"""
Batch deserializer for multiple eCAL measurement folders.

This script processes all subdirectories within a parent directory,
running the deserializer.py script for each measurement folder.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from loguru import logger
from tqdm import tqdm


def find_measurement_dirs(parent_dir: Path) -> List[Path]:
    """Find all measurement directories within the parent directory.

    A directory is considered a measurement directory if it contains
    at least one .ecalmeas file (eCAL measurement files).

    Parameters
    ----------
    parent_dir : Path
        Parent directory containing multiple measurement folders.

    Returns
    -------
    List[Path]
        List of measurement directory paths sorted alphabetically.

    Raises
    ------
    FileNotFoundError
        If the parent directory does not exist.
    ValueError
        If no measurement directories are found.
    """
    if not parent_dir.exists() or not parent_dir.is_dir():
        message = f"Parent directory does not exist or is not a directory: {parent_dir}"
        logger.error(message)
        raise FileNotFoundError(message)

    # Find all subdirectories that contain .ecalmeas files
    meas_dirs: List[Path] = []
    for item in sorted(parent_dir.iterdir()):
        if item.is_dir():
            # Check if this directory contains .ecalmeas files
            ecalmeas_files = list(item.glob("*.ecalmeas"))
            if len(ecalmeas_files) > 0:
                meas_dirs.append(item)
                logger.debug(f"Found measurement directory: {item}")

    if not meas_dirs:
        message = f"No measurement directories found in: {parent_dir}"
        logger.error(message)
        raise ValueError(message)

    logger.info(f"Found {len(meas_dirs)} measurement directories")
    return sorted(meas_dirs)


def run_deserializer(
    meas_dir: Path,
    skip_raw: bool,
    image_format: str,
    force: bool,
    cam_context_fpath: str,
) -> bool:
    """Run deserializer.py for a single measurement directory.

    Parameters
    ----------
    meas_dir : Path
        Measurement directory path.
    skip_raw : bool
        Whether to skip raw image generation.
    image_format : str
        Output image format ('webp' or 'jpeg').
    force : bool
        Whether to force reprocessing.
    cam_context_fpath : str
        Camera context file path (empty string for auto-detection).

    Returns
    -------
    bool
        True if deserialization succeeded, False otherwise.
    """
    # Build command
    cmd = [
        sys.executable,
        "deserializer.py",
        "--meas_dir",
        str(meas_dir),
        "--image_format",
        image_format,
    ]

    if skip_raw:
        cmd.append("--skip_raw")

    if force:
        cmd.append("--force")

    if cam_context_fpath:
        cmd.extend(["--cam_context_fpath", cam_context_fpath])

    logger.info(f"Processing: {meas_dir.name}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.debug(f"stdout: {result.stdout}")
        if result.stderr:
            logger.debug(f"stderr: {result.stderr}")
        logger.success(f"Processed: {meas_dir.name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to process {meas_dir.name}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False


def main() -> None:
    """Main entry point for batch deserialization."""
    parser = argparse.ArgumentParser(
        description="Batch deserializer for multiple eCAL measurement folders"
    )
    parser.add_argument(
        "--parent_dir",
        type=str,
        required=True,
        help="Parent directory containing multiple measurement folders",
    )
    parser.add_argument(
        "--make_raw_images",
        action="store_true",
        help="Make raw images",
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="webp",
        choices=["webp", "jpeg"],
        help="Output image format for RGB images (default: webp)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force reprocessing even if output directories exist",
    )
    parser.add_argument(
        "--cam_context_fpath",
        type=str,
        default="",
        help="Camera context file path (empty for auto-detection per measurement)",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue processing remaining directories if one fails",
    )

    args = parser.parse_args()
    args.skip_raw = not args.make_raw_images

    parent_dir = Path(args.parent_dir).resolve()

    logger.info("=" * 80)
    logger.info("Batch Deserializer")
    logger.info("=" * 80)
    logger.info(f"Parent directory: {parent_dir}")
    logger.info(f"Skip raw: {args.skip_raw}")
    logger.info(f"Image format: {args.image_format}")
    logger.info(f"Force: {args.force}")
    logger.info(f"Continue on error: {args.continue_on_error}")
    logger.info("=" * 80)

    # Find all measurement directories
    meas_dirs = find_measurement_dirs(parent_dir)

    # Process each measurement directory
    success_count = 0
    fail_count = 0

    for meas_dir in tqdm(meas_dirs, desc="Deserializing measurements"):
        success = run_deserializer(
            meas_dir=meas_dir,
            skip_raw=args.skip_raw,
            image_format=args.image_format,
            force=args.force,
            cam_context_fpath=args.cam_context_fpath,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1
            if not args.continue_on_error:
                logger.error(
                    "Stopping due to error (use --continue_on_error to continue)"
                )
                break

    # Summary
    logger.info("=" * 80)
    logger.info("Batch Deserialization Summary")
    logger.info("=" * 80)
    logger.info(f"Total directories: {len(meas_dirs)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info("=" * 80)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
