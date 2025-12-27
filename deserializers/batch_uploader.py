#!/usr/bin/env python3

"""
Batch Uploader for WildPose Videos to YouTube.

This script uploads video.mp4 files from the WildPose dataset to YouTube.
It checks already uploaded videos by title to avoid duplicates, supporting
parallel manual uploads.

Usage:
    python batch_uploader.py --parent_dir /mnt/vault/WildPose_dataset
    python batch_uploader.py --parent_dir /mnt/vault/WildPose_dataset --dry_run
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Set

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


def get_uploaded_titles(youtube) -> Set[str]:
    """Fetch all video titles from the user's 'Uploads' playlist.

    This is used to check which videos have already been uploaded,
    supporting both automated and manual uploads.

    Parameters
    ----------
    youtube
        YouTube API client.

    Returns
    -------
    Set[str]
        Set of video titles already on the channel.
    """
    logger.info("Fetching list of already uploaded videos...")
    uploaded_titles: Set[str] = set()

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
            uploaded_titles.add(item["snippet"]["title"])

        next_page_token = playlist_response.get("nextPageToken")
        if not next_page_token:
            break

    logger.success(f"Found {len(uploaded_titles)} videos on channel.")
    return uploaded_titles


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


def upload_video(youtube, video_path: Path, title: str) -> Optional[str]:
    """Upload a video to YouTube with resumable upload and retry logic.

    Parameters
    ----------
    youtube
        YouTube API client.
    video_path : Path
        Path to the video file.
    title : str
        Video title.

    Returns
    -------
    Optional[str]
        YouTube video ID if successful, None otherwise.
    """
    body = {
        "snippet": {
            "title": title,
            "description": "",
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
        help="Check lists but do not upload",
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
    uploaded_titles = get_uploaded_titles(youtube)

    # 3. Scan local videos
    logger.info(f"Scanning {parent_dir} for video.mp4...")
    videos_to_upload = []

    for video_path in sorted(parent_dir.rglob("video.mp4")):
        meas_dir = video_path.parent
        animal_dir = meas_dir.parent

        # Generate title: {Animal}/{Date}_{Number}
        # Structure: parent_dir / Animal / Measurement / video.mp4
        if parent_dir not in animal_dir.parents:
            # User pointed directly to Animal dir
            title = f"{parent_dir.name}/{meas_dir.name}"
        else:
            # User pointed to Dataset root
            title = f"{animal_dir.name}/{meas_dir.name}"

        if title in uploaded_titles:
            logger.debug(f"Skip: {title} (Already on YouTube)")
            continue

        videos_to_upload.append({"path": video_path, "title": title})

    logger.info(f"Total local videos found: {len(list(parent_dir.rglob('video.mp4')))}")
    logger.info(f"Already uploaded: {len(uploaded_titles)}")
    logger.info(f"New videos to upload: {len(videos_to_upload)}")

    if args.dry_run:
        logger.info("Dry run mode. Videos to upload:")
        for v in videos_to_upload:
            logger.info(f"  - {v['title']}")
        return

    if not videos_to_upload:
        logger.info("All videos are already uploaded!")
        return

    # 4. Upload Loop
    upload_count = 0
    logger.info("Starting uploads...")

    videos_to_upload = sorted(videos_to_upload, key=lambda x: x["title"])
    for video in tqdm(videos_to_upload, desc="Batch Upload"):
        if upload_count >= MAX_UPLOADS_PER_RUN:
            logger.warning(
                f"Daily safety limit ({MAX_UPLOADS_PER_RUN}) reached. Stopping."
            )
            break

        video_id = upload_video(youtube, video["path"], video["title"])

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
