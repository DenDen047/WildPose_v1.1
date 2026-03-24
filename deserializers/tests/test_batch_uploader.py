"""Tests for batch_uploader pCloud link integration."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from batch_uploader import (
    build_description,
    get_local_titles,
    get_pcloud_link,
    get_uploaded_videos,
    update_video_description,
)

# --- pCloud base path used across tests ---
PCLOUD_BASE_PATH = "PhD Project/WildPose_dataset"


# =========================================================
# Tests for get_pcloud_link()
# =========================================================
class TestGetPcloudLink:
    """Tests for get_pcloud_link(title, pcloud_base_path)."""

    @patch("batch_uploader.subprocess.run")
    def test_returns_link_on_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="https://u.pcloud.link/publink/show?code=abc123\n",
            stderr="",
        )

        link = get_pcloud_link("Cheetah/2022-12-03_013", PCLOUD_BASE_PATH)

        assert link == "https://u.pcloud.link/publink/show?code=abc123"
        mock_run.assert_called_once_with(
            [
                "rclone",
                "link",
                f"pCloud:{PCLOUD_BASE_PATH}/Cheetah/2022-12-03_013",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

    @patch("batch_uploader.subprocess.run")
    def test_returns_none_on_rclone_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="directory not found",
        )

        link = get_pcloud_link("NonExistent/2022-01-01_001", PCLOUD_BASE_PATH)

        assert link is None

    @patch("batch_uploader.subprocess.run")
    def test_returns_none_on_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="rclone", timeout=30)

        link = get_pcloud_link("Cheetah/2022-12-03_013", PCLOUD_BASE_PATH)

        assert link is None

    @patch("batch_uploader.subprocess.run")
    def test_returns_none_when_rclone_not_found(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError

        link = get_pcloud_link("Cheetah/2022-12-03_013", PCLOUD_BASE_PATH)

        assert link is None

    @patch("batch_uploader.subprocess.run")
    def test_returns_none_on_non_url_output(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="NOTICE: some rclone warning\n",
            stderr="",
        )

        link = get_pcloud_link("Cheetah/2022-12-03_013", PCLOUD_BASE_PATH)

        assert link is None


# =========================================================
# Tests for get_local_titles()
# =========================================================
class TestGetLocalTitles:
    """Tests for get_local_titles(parent_dir)."""

    def test_returns_titles_from_video_mp4_paths(self, tmp_path: Path) -> None:
        # Create: parent_dir/Cheetah/2022-12-03_013/video.mp4
        (tmp_path / "Cheetah" / "2022-12-03_013").mkdir(parents=True)
        (tmp_path / "Cheetah" / "2022-12-03_013" / "video.mp4").touch()
        (tmp_path / "Lion_male" / "2022-12-05_001").mkdir(parents=True)
        (tmp_path / "Lion_male" / "2022-12-05_001" / "video.mp4").touch()

        titles = get_local_titles(tmp_path)

        assert titles == {"Cheetah/2022-12-03_013", "Lion_male/2022-12-05_001"}

    def test_ignores_dirs_without_video_mp4(self, tmp_path: Path) -> None:
        (tmp_path / "Cheetah" / "2022-12-03_013").mkdir(parents=True)
        # No video.mp4 created

        titles = get_local_titles(tmp_path)

        assert titles == set()

    def test_returns_empty_for_empty_dir(self, tmp_path: Path) -> None:
        titles = get_local_titles(tmp_path)

        assert titles == set()


# =========================================================
# Tests for build_description()
# =========================================================
class TestBuildDescription:
    """Tests for build_description(pcloud_link, existing_description)."""

    def test_creates_description_with_link(self) -> None:
        link = "https://u.pcloud.link/publink/show?code=abc123"

        desc = build_description(link)

        assert "Download raw data:" in desc
        assert link in desc

    def test_returns_empty_when_no_link(self) -> None:
        desc = build_description(None)

        assert desc == ""

    def test_appends_link_to_existing_description(self) -> None:
        link = "https://u.pcloud.link/publink/show?code=abc123"
        existing = "This is a wildlife video."

        desc = build_description(link, existing_description=existing)

        assert desc.startswith("This is a wildlife video.")
        assert link in desc

    def test_replaces_existing_link_in_description(self) -> None:
        old_link = "https://u.pcloud.link/publink/show?code=OLD"
        new_link = "https://u.pcloud.link/publink/show?code=NEW"
        existing = f"Some text.\nDownload raw data: {old_link}\nMore text."

        desc = build_description(new_link, existing_description=existing)

        assert old_link not in desc
        assert new_link in desc
        assert "Some text." in desc
        assert "More text." in desc

    def test_idempotent_update(self) -> None:
        link = "https://u.pcloud.link/publink/show?code=abc123"
        existing = f"Some text.\nDownload raw data: {link}"

        desc = build_description(link, existing_description=existing)

        assert desc.count(link) == 1
        assert desc.count("Download raw data:") == 1

    def test_no_link_preserves_existing_description(self) -> None:
        existing = "This is a wildlife video."

        desc = build_description(None, existing_description=existing)

        assert desc == existing


# =========================================================
# Tests for get_uploaded_videos()
# =========================================================
class TestGetUploadedVideos:
    """Tests for get_uploaded_videos(youtube) -> Dict[str, str]."""

    def test_returns_title_to_video_id_mapping(self) -> None:
        youtube = MagicMock()

        # Mock channels().list()
        youtube.channels().list().execute.return_value = {
            "items": [
                {
                    "contentDetails": {
                        "relatedPlaylists": {"uploads": "UU_playlist_id"}
                    }
                }
            ]
        }

        # Mock playlistItems().list() - single page
        youtube.playlistItems().list().execute.return_value = {
            "items": [
                {
                    "snippet": {
                        "title": "Cheetah/2022-12-03_013",
                        "resourceId": {"videoId": "vid_001"},
                    }
                },
                {
                    "snippet": {
                        "title": "Lion_male/2022-12-05_001",
                        "resourceId": {"videoId": "vid_002"},
                    }
                },
            ],
            "nextPageToken": None,
        }

        result = get_uploaded_videos(youtube)

        assert isinstance(result, dict)
        assert result["Cheetah/2022-12-03_013"] == "vid_001"
        assert result["Lion_male/2022-12-05_001"] == "vid_002"

    def test_handles_pagination(self) -> None:
        youtube = MagicMock()

        youtube.channels().list().execute.return_value = {
            "items": [
                {
                    "contentDetails": {
                        "relatedPlaylists": {"uploads": "UU_playlist_id"}
                    }
                }
            ]
        }

        # Page 1
        page1 = {
            "items": [
                {
                    "snippet": {
                        "title": "Cheetah/2022-12-03_013",
                        "resourceId": {"videoId": "vid_001"},
                    }
                },
            ],
            "nextPageToken": "page2_token",
        }
        # Page 2
        page2 = {
            "items": [
                {
                    "snippet": {
                        "title": "Lion_male/2022-12-05_001",
                        "resourceId": {"videoId": "vid_002"},
                    }
                },
            ],
        }

        youtube.playlistItems().list().execute.side_effect = [page1, page2]

        result = get_uploaded_videos(youtube)

        assert len(result) == 2
        assert "Cheetah/2022-12-03_013" in result
        assert "Lion_male/2022-12-05_001" in result

    def test_returns_empty_dict_when_no_videos(self) -> None:
        youtube = MagicMock()

        youtube.channels().list().execute.return_value = {
            "items": [
                {
                    "contentDetails": {
                        "relatedPlaylists": {"uploads": "UU_playlist_id"}
                    }
                }
            ]
        }

        youtube.playlistItems().list().execute.return_value = {
            "items": [],
        }

        result = get_uploaded_videos(youtube)

        assert result == {}


# =========================================================
# Tests for update_video_description()
# =========================================================
class TestUpdateVideoDescription:
    """Tests for update_video_description(youtube, video_id, new_description)."""

    def test_fetches_current_snippet_and_updates(self) -> None:
        youtube = MagicMock()

        youtube.videos().list().execute.return_value = {
            "items": [
                {
                    "snippet": {
                        "title": "Cheetah/2022-12-03_013",
                        "description": "",
                        "categoryId": "28",
                        "tags": [],
                    }
                }
            ]
        }

        update_video_description(
            youtube,
            video_id="vid_001",
            new_description="Download raw data: https://example.com",
        )

        youtube.videos().update.assert_called_once()
        call_kwargs = youtube.videos().update.call_args
        body = call_kwargs[1]["body"] if "body" in call_kwargs[1] else call_kwargs[0][0]
        assert body["snippet"]["description"] == "Download raw data: https://example.com"
        assert body["snippet"]["title"] == "Cheetah/2022-12-03_013"
        assert body["snippet"]["categoryId"] == "28"

    def test_returns_true_on_success(self) -> None:
        youtube = MagicMock()

        youtube.videos().list().execute.return_value = {
            "items": [
                {
                    "snippet": {
                        "title": "Cheetah/2022-12-03_013",
                        "description": "",
                        "categoryId": "28",
                    }
                }
            ]
        }

        result = update_video_description(
            youtube,
            video_id="vid_001",
            new_description="New description",
        )

        assert result is True

    def test_returns_false_on_api_error(self) -> None:
        youtube = MagicMock()

        youtube.videos().list().execute.return_value = {
            "items": [
                {
                    "snippet": {
                        "title": "Test",
                        "description": "",
                        "categoryId": "28",
                    }
                }
            ]
        }

        from googleapiclient.errors import HttpError
        from unittest.mock import PropertyMock

        resp = MagicMock()
        type(resp).status = PropertyMock(return_value=403)
        youtube.videos().update().execute.side_effect = HttpError(
            resp=resp, content=b"quota exceeded"
        )

        result = update_video_description(
            youtube,
            video_id="vid_001",
            new_description="New description",
        )

        assert result is False

    def test_returns_false_when_video_not_found(self) -> None:
        youtube = MagicMock()

        youtube.videos().list().execute.return_value = {"items": []}

        result = update_video_description(
            youtube,
            video_id="nonexistent",
            new_description="New description",
        )

        assert result is False
        youtube.videos().update.assert_not_called()
