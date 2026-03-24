#!/usr/bin/env python3

"""
Batch Uploader for WildPose Videos to YouTube.

This script uploads video.mp4 files from the WildPose dataset to YouTube.
It checks already uploaded videos by title to avoid duplicates, supporting
parallel manual uploads. Each video's description includes a pCloud shared
link to the corresponding raw data folder (generated via rclone).

Usage:
    python batch_uploader.py --parent_dir /mnt/vault/WildPose_dataset
    python batch_uploader.py --parent_dir /mnt/vault/WildPose_dataset --dry_run
    python batch_uploader.py --parent_dir /mnt/vault/WildPose_dataset --update_descriptions
    python batch_uploader.py --parent_dir /mnt/vault/WildPose_dataset --update_descriptions --dry_run
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from loguru import logger
from tqdm import tqdm

# Configuration
SECRETS_DIR = Path(__file__).parent / ".secrets"
TOKEN_PATH = SECRETS_DIR / "youtube_token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube"]

# Target Playlists (hardcoded)
# 1. WildPose v1.1
# 2. 3D Wildlife Dataset
TARGET_PLAYLIST_IDS = [
    "PLB9Pwo4Wnh7XQg2RmqRyzIzcOIjPWBh_O",
    "PLB9Pwo4Wnh7U5ENPaePVcoEWByQWjPQeC",
]

# Daily Limit Safety
# Cost: Upload(1600) + 2*Playlist(50) = 1700 units/video
# 10,000 / 1700 = 5.88 videos
MAX_UPLOADS_PER_RUN = 50

PCLOUD_LINK_PREFIX = "Download raw data:"

# Title must be "{word}/{word}" where words contain only alphanumerics, underscores, hyphens, dots, and spaces
_TITLE_RE = re.compile(r"^[\w\-. ]+/[\w\-. ]+$")


def _make_title(parent_dir: Path, video_path: Path) -> str:
    """Derive the YouTube title from a video.mp4 path.

    Parameters
    ----------
    parent_dir : Path
        Root dataset directory.
    video_path : Path
        Path to the video.mp4 file.

    Returns
    -------
    str
        Title in "{Animal}/{Measurement}" format.
    """
    meas_dir = video_path.parent
    animal_dir = meas_dir.parent
    if parent_dir not in animal_dir.parents:
        return f"{parent_dir.name}/{meas_dir.name}"
    return f"{animal_dir.name}/{meas_dir.name}"


def get_local_titles(parent_dir: Path) -> set[str]:
    """Scan parent_dir for video.mp4 files and return the set of titles.

    Parameters
    ----------
    parent_dir : Path
        Root dataset directory.

    Returns
    -------
    set[str]
        Set of titles in "{Animal}/{Measurement}" format.
    """
    titles: set[str] = set()
    for video_path in parent_dir.rglob("video.mp4"):
        titles.add(_make_title(parent_dir, video_path))
    return titles


def get_pcloud_link(title: str, pcloud_base_path: str) -> Optional[str]:
    """Get a pCloud shared link for a measurement folder via rclone.

    Parameters
    ----------
    title : str
        Video title in the format "{Animal}/{Date_Number}".
    pcloud_base_path : str
        rclone path prefix, e.g. "PhD Project/WildPose_dataset".

    Returns
    -------
    Optional[str]
        The public link URL, or None on failure.
    """
    if not _TITLE_RE.match(title):
        logger.warning(f"Skipping invalid title: {title!r}")
        return None

    remote_path = f"pCloud:{pcloud_base_path}/{title}"
    try:
        result = subprocess.run(
            ["rclone", "link", remote_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"rclone link timed out for {remote_path}")
        return None
    except FileNotFoundError:
        logger.warning("rclone not found on PATH; pCloud links will be skipped.")
        return None

    if result.returncode != 0:
        logger.warning(f"rclone link failed for {remote_path}: {result.stderr.strip()}")
        return None

    url = result.stdout.strip()
    if not url.startswith("https://"):
        logger.warning(f"Unexpected rclone output for {remote_path!r}: {url!r}")
        return None

    return url


def build_description(
    pcloud_link: Optional[str], existing_description: str = ""
) -> str:
    """Build or update a video description with a pCloud link.

    Parameters
    ----------
    pcloud_link : Optional[str]
        The pCloud public link URL, or None.
    existing_description : str
        Current description to update.

    Returns
    -------
    str
        The updated description.
    """
    if pcloud_link is None:
        return existing_description

    link_line = f"{PCLOUD_LINK_PREFIX} {pcloud_link}"

    if not existing_description:
        return link_line

    # Replace existing pCloud link line if present
    pattern = re.compile(rf"^{re.escape(PCLOUD_LINK_PREFIX)}.*$", re.MULTILINE)
    if pattern.search(existing_description):
        return pattern.sub(link_line, existing_description)

    # Append to existing description
    return f"{existing_description}\n{link_line}"


def get_uploaded_videos(youtube) -> Dict[str, str]:
    """Fetch all uploaded videos as a title -> video_id mapping.

    Parameters
    ----------
    youtube
        YouTube API client.

    Returns
    -------
    Dict[str, str]
        Mapping of video title to video ID.
    """
    logger.info("Fetching list of already uploaded videos...")
    uploaded_videos: Dict[str, str] = {}

    channels_response = (
        youtube.channels().list(mine=True, part="contentDetails").execute()
    )

    if not channels_response.get("items"):
        logger.error("No channel found for this account.")
        sys.exit(1)

    uploads_playlist_id = channels_response["items"][0]["contentDetails"][
        "relatedPlaylists"
    ]["uploads"]

    next_page_token = None
    while True:
        playlist_response = (
            youtube.playlistItems()
            .list(
                playlistId=uploads_playlist_id,
                part="snippet",
                maxResults=50,
                pageToken=next_page_token,
            )
            .execute()
        )

        for item in playlist_response["items"]:
            title = item["snippet"]["title"]
            video_id = item["snippet"]["resourceId"]["videoId"]
            uploaded_videos[title] = video_id

        next_page_token = playlist_response.get("nextPageToken")
        if not next_page_token:
            break

    logger.success(f"Found {len(uploaded_videos)} videos on channel.")
    return uploaded_videos


def update_video_description(
    youtube, video_id: str, new_description: str
) -> bool:
    """Update the description of an existing YouTube video.

    Fetches the current snippet first to preserve title, categoryId, etc.

    Parameters
    ----------
    youtube
        YouTube API client.
    video_id : str
        YouTube video ID.
    new_description : str
        The new description to set.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        response = youtube.videos().list(id=video_id, part="snippet").execute()
    except HttpError as e:
        logger.error(f"Failed to fetch video {video_id}: {e}")
        return False

    if not response.get("items"):
        logger.warning(f"Video {video_id} not found.")
        return False

    snippet = response["items"][0]["snippet"]
    snippet["description"] = new_description

    try:
        youtube.videos().update(
            part="snippet",
            body={"id": video_id, "snippet": snippet},
        ).execute()
    except HttpError as e:
        logger.error(f"Failed to update video {video_id}: {e}")
        return False

    return True


def authenticate() -> Credentials:
    """Load and refresh OAuth2 credentials.

    Returns
    -------
    Credentials
        Valid OAuth2 credentials for YouTube API.

    Raises
    ------
    SystemExit
        If token file is not found or cannot be refreshed.
    """
    if not TOKEN_PATH.exists():
        logger.error(f"Token file not found: {TOKEN_PATH}")
        logger.error("Please run 'tools/test_youtube_auth.py' first.")
        sys.exit(1)

    credentials = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            logger.info("Token expired, refreshing...")
            credentials.refresh(Request())
            TOKEN_PATH.write_text(credentials.to_json())
        else:
            logger.error("Token invalid and cannot be refreshed.")
            sys.exit(1)

    return credentials


def add_to_playlist(youtube, video_id: str, playlist_id: str) -> None:
    """Add a video to a playlist.

    Parameters
    ----------
    youtube
        YouTube API client.
    video_id : str
        YouTube video ID.
    playlist_id : str
        Target playlist ID.
    """
    youtube.playlistItems().insert(
        part="snippet",
        body={
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {"kind": "youtube#video", "videoId": video_id},
            }
        },
    ).execute()
    logger.info(f"Added to playlist {playlist_id}")


def upload_video(
    youtube, video_path: Path, title: str, description: str = ""
) -> Optional[str]:
    """Upload a video to YouTube with resumable upload and retry logic.

    Parameters
    ----------
    youtube
        YouTube API client.
    video_path : Path
        Path to the video file.
    title : str
        Video title.
    description : str
        Video description.

    Returns
    -------
    Optional[str]
        YouTube video ID if successful, None otherwise.
    """
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": "28",  # Science & Technology
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(video_path),
        chunksize=1024 * 1024,  # 1MB chunks
        resumable=True,
        mimetype="video/mp4",
    )

    request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media,
    )

    file_size_mb = video_path.stat().st_size / 1024 / 1024
    logger.info(f"Uploading: {title} ({file_size_mb:.1f} MB)")

    response = None
    retry_count = 0
    max_retries = 10

    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                print(f"  Progress: {progress}%", end="\r")

        except HttpError as e:
            if e.resp.status in [500, 502, 503, 504]:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error("Max retries exceeded.")
                    return None

                sleep_time = 2**retry_count
                logger.warning(
                    f"Network error {e.resp.status}. Retrying in {sleep_time}s..."
                )
                time.sleep(sleep_time)
                continue

            if "quotaExceeded" in str(e):
                logger.critical("YouTube API Daily Quota Exceeded!")
                return None

            logger.error(f"Upload failed: {e}")
            return None

    print("")  # Newline after progress
    return response["id"]


def main() -> None:
    """Main entry point for batch upload."""
    parser = argparse.ArgumentParser(
        description="Batch upload WildPose videos to YouTube."
    )
    parser.add_argument(
        "--parent_dir",
        type=str,
        required=True,
        help="Root dataset directory (e.g., /mnt/vault/WildPose_dataset)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Check lists but do not upload or update",
    )
    parser.add_argument(
        "--pcloud_base_path",
        type=str,
        default="PhD Project/WildPose_dataset",
        help="rclone remote path prefix for pCloud (default: 'PhD Project/WildPose_dataset')",
    )
    parser.add_argument(
        "--update_descriptions",
        action="store_true",
        help="Update descriptions of already-uploaded videos with pCloud links",
    )
    args = parser.parse_args()

    parent_dir = Path(args.parent_dir).resolve()
    if not parent_dir.exists():
        logger.error(f"Directory not found: {parent_dir}")
        sys.exit(1)

    # 1. Authenticate
    credentials = authenticate()
    youtube = build("youtube", "v3", credentials=credentials)

    # 2. Get already uploaded videos (Source of Truth)
    uploaded_videos = get_uploaded_videos(youtube)

    # 3. Update descriptions mode
    if args.update_descriptions:
        _update_existing_descriptions(youtube, uploaded_videos, parent_dir, args)
        return

    # 4. Scan local videos for upload
    _upload_new_videos(youtube, uploaded_videos, parent_dir, args)


def _update_existing_descriptions(
    youtube,
    uploaded_videos: Dict[str, str],
    parent_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Update descriptions of already-uploaded videos with pCloud links."""
    # Filter to only videos that exist in the local dataset
    local_titles = get_local_titles(parent_dir)
    target_videos = {
        title: vid for title, vid in uploaded_videos.items() if title in local_titles
    }

    logger.info(
        f"Update descriptions mode: {len(target_videos)} dataset videos "
        f"(out of {len(uploaded_videos)} total on channel)"
    )

    update_count = 0
    titles = sorted(target_videos.keys())

    for title in tqdm(titles, desc="Updating Descriptions"):
        video_id = target_videos[title]

        pcloud_link = get_pcloud_link(title, args.pcloud_base_path)
        if pcloud_link is None:
            logger.debug(f"Skip: {title} (no pCloud folder found)")
            continue

        # Fetch current description
        try:
            response = youtube.videos().list(id=video_id, part="snippet").execute()
        except HttpError as e:
            logger.error(f"Failed to fetch video {video_id}: {e}")
            continue

        if not response.get("items"):
            continue

        current_description = response["items"][0]["snippet"].get("description", "")
        new_description = build_description(pcloud_link, current_description)

        if new_description == current_description:
            logger.debug(f"Skip: {title} (description already up to date)")
            continue

        if args.dry_run:
            logger.info(f"  [DRY RUN] Would update: {title}")
            logger.info(f"    pCloud link: {pcloud_link}")
            update_count += 1
            continue

        if update_video_description(youtube, video_id, new_description):
            logger.success(f"Updated: {title}")
            update_count += 1
            time.sleep(1)
        else:
            logger.error(f"Failed to update: {title}")

    logger.info(f"Session finished. Updated {update_count} video descriptions.")


def _upload_new_videos(
    youtube,
    uploaded_videos: Dict[str, str],
    parent_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Scan local videos and upload new ones with pCloud links in description."""
    logger.info(f"Scanning {parent_dir} for video.mp4...")
    videos_to_upload = []
    total_found = 0

    for video_path in parent_dir.rglob("video.mp4"):
        total_found += 1
        title = _make_title(parent_dir, video_path)

        if title in uploaded_videos:
            logger.debug(f"Skip: {title} (Already on YouTube)")
            continue

        videos_to_upload.append({"path": video_path, "title": title})

    logger.info(f"Total local videos found: {total_found}")
    logger.info(f"Already uploaded: {len(uploaded_videos)}")
    logger.info(f"New videos to upload: {len(videos_to_upload)}")

    if args.dry_run:
        logger.info("Dry run mode. Videos to upload:")
        for v in videos_to_upload:
            pcloud_link = get_pcloud_link(v["title"], args.pcloud_base_path)
            logger.info(f"  - {v['title']} (pCloud: {pcloud_link or 'N/A'})")
        return

    if not videos_to_upload:
        logger.info("All videos are already uploaded!")
        return

    # Upload Loop
    upload_count = 0
    logger.info("Starting uploads...")

    videos_to_upload = sorted(videos_to_upload, key=lambda x: x["title"])
    for video in tqdm(videos_to_upload, desc="Batch Upload"):
        if upload_count >= MAX_UPLOADS_PER_RUN:
            logger.warning(
                f"Daily safety limit ({MAX_UPLOADS_PER_RUN}) reached. Stopping."
            )
            break

        # Get pCloud link for description
        pcloud_link = get_pcloud_link(video["title"], args.pcloud_base_path)
        description = build_description(pcloud_link)

        video_id = upload_video(
            youtube, video["path"], video["title"], description=description
        )

        if video_id:
            logger.success(f"Uploaded: {video['title']} (ID: {video_id})")

            # Add to playlists
            for playlist_id in TARGET_PLAYLIST_IDS:
                try:
                    add_to_playlist(youtube, video_id, playlist_id)
                    time.sleep(1)  # Gentle delay between API calls
                except HttpError as e:
                    logger.error(f"Failed to add to playlist {playlist_id}: {e}")

            upload_count += 1
            time.sleep(5)  # Buffer between uploads
        else:
            logger.error("Stopping due to upload error.")
            break

    logger.info(f"Session finished. Uploaded {upload_count} videos.")


if __name__ == "__main__":
    main()
