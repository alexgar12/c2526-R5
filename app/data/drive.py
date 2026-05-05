"""
Módulo de acceso a Google Drive para descargar datos del pipeline ETL de la MTA.

Proporciona funciones para autenticarse con la API de Google Drive y descargar
los ficheros generados por el pipeline ETL: ventanas de datos en tiempo real
(Parquet) y ficheros diarios de datos estáticos.

Dependencias:
- google-auth y google-api-python-client para la autenticación OAuth2 y la API de Drive.
- pandas para leer los ficheros Parquet y CSV descargados.
- token_drive.json: fichero de credenciales OAuth2 generado con el flujo local de
  autenticación (debe montarse en el contenedor en producción).

Notas:
- La ruta por defecto del token se resuelve relativamente al repositorio; puede
  sobreescribirse con la variable de entorno GDRIVE_TOKEN_PATH.
- Si el token ha expirado pero tiene refresh_token, se renueva automáticamente
  y se guarda de vuelta en disco.
"""

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
    """
    Construye y devuelve un cliente autenticado de la API de Google Drive v3.

    Lee las credenciales OAuth2 desde el fichero token_drive.json indicado.
    Si el token ha expirado y dispone de refresh_token, lo renueva y actualiza
    el fichero en disco.

    Parámetros:
        token_path: Ruta al fichero JSON de credenciales OAuth2.

    Retorna:
        Objeto de servicio de Google Drive listo para realizar peticiones.

    Lanza:
        FileNotFoundError: si el fichero token_drive.json no existe en la ruta indicada.
    """
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
    """
    Obtiene el ID de una carpeta de Google Drive buscando por nombre.

    Parámetros:
        service: Cliente autenticado de la API de Google Drive.
        folder_name: Nombre exacto de la carpeta a buscar.
        parent_id: ID de la carpeta padre en la que buscar (opcional; si se omite,
                   busca en todo Drive).

    Retorna:
        ID de la primera carpeta encontrada con el nombre indicado.

    Lanza:
        ValueError: si no se encuentra ninguna carpeta con ese nombre.
    """
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
    """
    Descarga un fichero Parquet desde una carpeta (o subcarpeta) de Drive y lo retorna como DataFrame.

    Busca el fichero por nombre exacto dentro de la estructura root_folder/subfolder/.
    Útil para obtener datos estáticos diarios como el CSV de estaciones.

    Parámetros:
        filename: Nombre exacto del fichero a descargar (p.ej. 'adjacency_matrix.parquet').
        subfolder: Nombre de la subcarpeta dentro de root_folder (opcional).
        root_folder: Nombre de la carpeta raíz en Drive. Por defecto 'MTA_Daily_Data'.
        token_path: Ruta al token OAuth2. Si es None, usa _DEFAULT_TOKEN_PATH.

    Retorna:
        DataFrame de pandas con el contenido del fichero Parquet descargado.

    Lanza:
        FileNotFoundError: si el fichero no existe en la ruta indicada de Drive.
    """
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
    """
    Descarga las N ventanas de datos en tiempo real más recientes desde Google Drive.

    Los ficheros se llaman ventana_*.parquet y son generados por el script ETL
    upload_realtime_window.py. Se descargan ordenados de más reciente a más antiguo
    (según nombre) y se devuelven en orden cronológico ascendente (más antiguo primero).

    Parámetros:
        n_windows: Número máximo de ventanas a descargar. Por defecto 8.
        token_path: Ruta al token OAuth2. Si es None, usa _DEFAULT_TOKEN_PATH.
        folder_name: Nombre de la carpeta en Drive. Por defecto 'MTA_Realtime_Windows'.

    Retorna:
        Lista de DataFrames de pandas, ordenados de más antiguo a más reciente.

    Lanza:
        ValueError: si no se encuentran ficheros Parquet en la carpeta indicada.
    """
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

    files = files[:n_windows]  # Las más recientes primero, según Drive

    windows: list[pd.DataFrame] = []
    for info in reversed(files):  # Invertir para devolver de más antiguo a más reciente
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
