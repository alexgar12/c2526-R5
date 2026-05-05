import io
import logging
import os
from pathlib import Path

import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/drive"]
_FOLDER_NAME = "MTA_Realtime_Windows"

_DEFAULT_TOKEN_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src" / "ETL" / "alertas_oficiales_tiempo_real" / "token_drive.json"
)


def _get_service(token_path: Path) -> object:
    if not token_path.exists():
        raise FileNotFoundError(
            f"token_drive.json not found at {token_path}. "
            "Run the OAuth flow locally once to generate it, or mount the file into the container."
        )

    creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _get_folder_id(service, folder_name: str, parent_id: str | None = None) -> str:
    parent_clause = f"and '{parent_id}' in parents " if parent_id else ""
    result = service.files().list(
        q=(
            f"name = '{folder_name}' "
            f"and mimeType = 'application/vnd.google-apps.folder' "
            f"and trashed = false "
            f"{parent_clause}"
        ),
        fields="files(id, name)",
    ).execute()
    files = result.get("files", [])
    if not files:
        raise ValueError(
            f"Drive folder '{folder_name}' not found. "
            "It is created automatically on the first ETL run."
        )
    return files[0]["id"]


def download_daily_file(
    filename: str,
    subfolder: str | None = None,
    root_folder: str = "MTA_Daily_Data",
    token_path: Path | None = None,
) -> pd.DataFrame:

    token_path = token_path or _DEFAULT_TOKEN_PATH
    service = _get_service(token_path)

    root_id = _get_folder_id(service, root_folder)
    folder_id = _get_folder_id(service, subfolder, parent_id=root_id) if subfolder else root_id

    result = service.files().list(
        q=(
            f"'{folder_id}' in parents "
            f"and name = '{filename}' "
            f"and trashed = false"
        ),
        fields="files(id, name)",
        pageSize=1,
    ).execute()

    files = result.get("files", [])
    if not files:
        path = f"{root_folder}/{subfolder}/{filename}" if subfolder else f"{root_folder}/{filename}"
        raise FileNotFoundError(
            f"'{path}' not found in Drive. Run upload_daily_data.py first."
        )

    req = service.files().get_media(fileId=files[0]["id"])
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    df = pd.read_parquet(buf)
    logger.info("Drive daily file loaded: %s (%d rows)", filename, len(df))
    return df


def download_windows(
    n_windows: int = 8,
    token_path: Path | None = None,
    folder_name: str = _FOLDER_NAME,
) -> list[pd.DataFrame]:

    token_path = token_path or _DEFAULT_TOKEN_PATH
    service = _get_service(token_path)
    folder_id = _get_folder_id(service, folder_name)

    result = service.files().list(
        q=(
            f"'{folder_id}' in parents "
            f"and name contains 'ventana_' "
            f"and name contains '.parquet' "
            f"and trashed = false"
        ),
        orderBy="name desc",
        pageSize=n_windows,
        fields="files(id, name)",
    ).execute()

    files = result.get("files", [])
    if not files:
        raise ValueError(
            f"No parquet windows found in Drive folder '{folder_name}'. "
            "The ETL pipeline (upload_realtime_window.py) must run first."
        )

    files = files[:n_windows]  # newest-first from Drive

    windows: list[pd.DataFrame] = []
    for info in reversed(files):  # oldest first in output
        req = service.files().get_media(fileId=info["id"])
        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        buf.seek(0)
        df = pd.read_parquet(buf)
        windows.append(df)
        logger.debug("Drive window loaded: %s (%d rows)", info["name"], len(df))

    logger.info("Loaded %d windows from Drive folder '%s'", len(windows), folder_name)
    return windows
