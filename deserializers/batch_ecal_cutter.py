#!/usr/bin/env python3

"""
Batch cutter for eCAL measurement folders using ecal_meas_cutter.

Edit the global variables below (OUTPUT_ROOT and JOBS) to specify
input measurement folders and time ranges. Then run this script.

No CLI arguments are used by design.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml  # type: ignore
from loguru import logger
from tqdm import tqdm

# ============ User-editable globals (no CLI) ============

# Base output directory where cut measurements will be written.
OUTPUT_ROOT: str = "/mnt/data/WildPose_v1.1"

# Path or command name for ecal_meas_cutter (must be in PATH if only name).
ECAL_MEAS_CUTTER: str = "ecal_meas_cutter"

# Default parameters used for each job's config unless overridden per job.
DEFAULT_BASENAME: str = "measurement"
DEFAULT_SPLIT_SIZE_MB: int = 1024


@dataclass(frozen=True)
class CutRange:
    """Time trimming range for a job.

    Parameters
    ----------
    start_seconds : float
        Start time offset.
    end_seconds : float
        End time offset.
    start_base : str, optional
        Base reference for start (e.g., "start" or "end"). Default is "start".
    end_base : str, optional
        Base reference for end (e.g., "start" or "end"). Default is "start".
    scale : str, optional
        Time scale for the cutter (e.g., "s" for seconds). Default is "s".
    """

    start_seconds: float
    end_seconds: float
    start_base: str = "start"
    end_base: str = "start"
    scale: str = "s"


@dataclass(frozen=True)
class CutterJob:
    """One batch-cut job definition.

    Parameters
    ----------
    input_path : str
        Path to the source eCAL measurement folder.
    range : CutRange
        Trimming range configuration.
    basename : Optional[str]
        Basename field for the cutter config; defaults to DEFAULT_BASENAME.
    split_size_mb : Optional[int]
        Split size in MB for the cutter config; defaults to DEFAULT_SPLIT_SIZE_MB.
    output_root : Optional[str]
        Optional override for the base output directory for this job only. If not
        provided, the global OUTPUT_ROOT is used.
    """

    input_path: str
    range: CutRange
    basename: Optional[str] = None
    split_size_mb: Optional[int] = None
    output_root: Optional[str] = None


# Edit this list to add jobs. Start with it empty to avoid accidental runs.
JOBS: List[CutterJob] = [
    # Example (uncomment and edit):
    # CutterJob(
    #     input_path="/media/ikuta/Expansion/2022-12-03/",
    #     range=CutRange(start_seconds=0.0, end_seconds=0.0),
    #     output_root="/mnt/data/WildPose_v1.1//snippet_000",
    # ),
    CutterJob(
        input_path="/media/ikuta/Expansion/2022-12-05/Afternoon/2022-12-05_17-00-31.024_wildpose_v1.1",
        range=CutRange(start_seconds=7, end_seconds=20),
        output_root="/mnt/vault/WildPose_v1.1/PCG/2022-12-05_003",
    ),
    CutterJob(
        input_path="/media/ikuta/Expansion/2022-12-05/Afternoon/2022-12-05_17-00-31.024_wildpose_v1.1",
        range=CutRange(start_seconds=125, end_seconds=128),
        output_root="/mnt/vault/WildPose_v1.1/PCG/2022-12-05_004",
    ),
    CutterJob(
        input_path="/media/ikuta/Expansion/2022-12-05/Afternoon/2022-12-05_17-00-31.024_wildpose_v1.1",
        range=CutRange(start_seconds=132, end_seconds=151),
        output_root="/mnt/vault/WildPose_v1.1/PCG/2022-12-05_005",
    ),
]


def _validate_environment() -> None:
    """Validate required environment and fail fast.

    Raises
    ------
    FileNotFoundError
        If the ecal_meas_cutter binary cannot be found.
    """

    cutter_path = (
        shutil.which(ECAL_MEAS_CUTTER)
        if os.path.sep not in ECAL_MEAS_CUTTER
        else ECAL_MEAS_CUTTER
    )
    if cutter_path is None or not Path(cutter_path).exists():
        message = f"ecal_meas_cutter not found: '{ECAL_MEAS_CUTTER}'. Ensure it is installed and on PATH."
        logger.error(message)
        raise FileNotFoundError(message)


def _build_config_yaml_text(job: CutterJob) -> str:
    """Build the YAML configuration text for the cutter.

    Parameters
    ----------
    job : CutterJob
        Job with required parameters.

    Returns
    -------
    str
        YAML content as text.
    """

    basename = job.basename if job.basename else DEFAULT_BASENAME
    split_size = (
        job.split_size_mb if job.split_size_mb is not None else DEFAULT_SPLIT_SIZE_MB
    )

    # Keep structure aligned with deserializers/ecal_cutter_config.yml
    cfg = {
        "basename": basename,
        "splitsize": split_size,
        "trim": {
            "start": {
                "time": job.range.start_seconds,
                "base": job.range.start_base,
                "scale": job.range.scale,
            },
            "end": {
                "time": job.range.end_seconds,
                "base": job.range.end_base,
                "scale": job.range.scale,
            },
        },
    }

    if job.range.start_seconds == 0.0:
        del cfg["trim"]["start"]
    if job.range.end_seconds == -1:
        del cfg["trim"]["end"]

    return yaml.safe_dump(cfg, sort_keys=False)


def _write_metadata_yaml(output_dir: Path, input_path: Path, job: CutterJob) -> Path:
    """Write minimal metadata YAML capturing the original input path.

    Parameters
    ----------
    output_dir : Path
        Destination directory where metadata will be saved.
    input_path : Path
        Source measurement directory path.
    job : CutterJob
        The job definition (used for optional label inclusion).

    Returns
    -------
    Path
        Path to the written metadata file.
    """

    meta = {
        "input_path": str(input_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Remove None values for cleanliness
    meta = {k: v for k, v in meta.items() if v is not None}

    meta_path = output_dir / "ecal_cutter_metadata.yml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    logger.info(f"Saved metadata: {meta_path}")
    return meta_path


def _run_cutter(
    job: CutterJob, input_path: Path, output_dir: Path, config_yaml_text: str
) -> None:
    """Execute ecal_meas_cutter for a single job.

    Parameters
    ----------
    job : CutterJob
        The job definition (used to enrich metadata).
    input_path : Path
        Source measurement directory.
    output_dir : Path
        Destination output root directory.
    config_yaml_text : str
        YAML configuration text to save into the output directory as
        'ecal_cutter_config.yml' and pass to the cutter.

    Raises
    ------
    FileNotFoundError
        If the input path does not exist.
    subprocess.CalledProcessError
        If the cutter command fails.
    """

    if not input_path.exists() or not input_path.is_dir():
        message = f"Input path does not exist or is not a directory: {input_path}"
        logger.error(message)
        raise FileNotFoundError(message)

    # Persist a copy of the YAML alongside outputs for reproducibility
    persistent_config_path = output_dir / "ecal_cutter_config.yml"
    with open(persistent_config_path, "w") as f_out:
        f_out.write(config_yaml_text)
    logger.info(f"Saved cutter config: {persistent_config_path}")

    # Save minimal metadata (original input path, optional label, timestamp)
    _write_metadata_yaml(output_dir, input_path, job=job)

    cmd = [
        ECAL_MEAS_CUTTER,
        "-i",
        str(input_path),
        "-o",
        str(output_dir),
        "--config",
        str(persistent_config_path),
    ]

    logger.info("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_batch(jobs: List[CutterJob]) -> None:
    """Run the batch cutter on all provided jobs.

    Parameters
    ----------
    jobs : list of CutterJob
        Jobs to execute sequentially.

    Raises
    ------
    ValueError
        If the job list is empty.
    """

    if not jobs:
        message = "No jobs specified. Edit JOBS in this script and try again."
        logger.error(message)
        raise ValueError(message)

    _validate_environment()

    for job in tqdm(jobs, desc="Batch cutting"):
        logger.info(f"Starting job for input: {job.input_path}")
        out_dir = Path(job.output_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        yaml_text = _build_config_yaml_text(job)
        _run_cutter(job, Path(job.input_path), out_dir, yaml_text)
        logger.info(f"Completed job -> output: {out_dir}")


if __name__ == "__main__":
    # Intentional: no CLI. Edit the globals above, then run the script.
    run_batch(JOBS)
