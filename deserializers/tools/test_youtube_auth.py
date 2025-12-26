"""
YouTube Data API v3 認証テスト
==============================

OAuth 2.0認証が正しく動作するかテストするスクリプト。
初回実行時はブラウザが開き、Googleアカウントでの認証が必要。

Usage:
    python test_youtube_auth.py
"""

from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from loguru import logger

# YouTube Data API v3のスコープ（アップロード用）
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

# パス設定
SECRETS_DIR = Path(__file__).parent / ".secrets"
CLIENT_SECRETS_PATH = SECRETS_DIR / "youtube_api_v3_client_secret.json"
TOKEN_PATH = SECRETS_DIR / "youtube_token.json"


def authenticate() -> Credentials:
    """OAuth 2.0認証を行い、credentialsを返す。"""
    # 既存のトークンがあれば読み込む
    if TOKEN_PATH.exists():
        logger.info(f"Loading existing token from {TOKEN_PATH}")
        credentials = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

        if credentials.valid:
            logger.info("Token is valid")
            return credentials

        # トークンが期限切れの場合、リフレッシュを試みる
        if credentials.expired and credentials.refresh_token:
            logger.info("Token expired, refreshing...")
            from google.auth.transport.requests import Request

            credentials.refresh(Request())
            TOKEN_PATH.write_text(credentials.to_json())
            logger.info("Token refreshed and saved")
            return credentials

    # 新規認証フロー
    if not CLIENT_SECRETS_PATH.exists():
        logger.error(f"Client secrets file not found: {CLIENT_SECRETS_PATH}")
        raise FileNotFoundError(CLIENT_SECRETS_PATH)

    logger.info("Starting OAuth flow (browser will open)...")
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_PATH), SCOPES)
    credentials = flow.run_local_server(port=0)

    # トークンを保存
    TOKEN_PATH.write_text(credentials.to_json())
    logger.info(f"Token saved to {TOKEN_PATH}")

    return credentials


def test_youtube_connection(credentials: Credentials) -> None:
    """YouTube APIへの接続をテスト。"""
    logger.info("Building YouTube API client...")
    youtube = build("youtube", "v3", credentials=credentials)

    logger.info(f"トークンファイル: {TOKEN_PATH}")
    logger.info(f"トークン有効: {credentials.valid}")
    logger.info(f"スコープ: {credentials.scopes}")

    # チャンネル情報を取得してテスト
    logger.info("Fetching channel info...")
    request = youtube.channels().list(part="snippet", mine=True)
    response = request.execute()

    if response.get("items"):
        channel = response["items"][0]["snippet"]
        logger.info("✅ 認証成功！")
        logger.info(f"   チャンネル名: {channel['title']}")
    else:
        logger.warning("チャンネル情報が取得できませんでした（新規チャンネルの可能性）")
        logger.info("✅ 認証成功！（APIクライアント構築完了）")


def main() -> None:
    logger.info("=" * 50)
    logger.info("YouTube Data API v3 認証テスト")
    logger.info("=" * 50)

    # 認証
    credentials = authenticate()

    # 接続テスト
    test_youtube_connection(credentials)

    logger.info("=" * 50)
    logger.info("テスト完了")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
