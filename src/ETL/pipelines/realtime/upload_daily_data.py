"""
Sube una vez al día a Google Drive los datasets de referencia:
  - GTFS Supplemented (stop_times.txt descomprimido como parquet)
  - Clima del día actual
  - Eventos del día actual

Se ejecuta a las 00:00 NY mediante cron en la VM de Google Cloud.
Cada archivo se sobreescribe con los datos más recientes.

Uso:
    uv run python src/ETL/pipelines/realtime/upload_daily_data.py

Variables de entorno requeridas:
    GDRIVE_TOKEN_JSON, GDRIVE_SHARE_EMAILS
"""

import io
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from src.ETL.tiempo_real_metro.realtime_data import ( creacion_df_previsto)

from src.ETL.pipelines.realtime.generate_realtime_dataset import (
    load_realtime_weather,
    load_realtime_events,
)

load_dotenv()

# ── Configuración ────────────────────────────────────────────────
FOLDER_RAIZ   = "MTA_Daily_Data"
FOLDER_GTFS   = "gtfs_supplemented"
FOLDER_CLIMA  = "clima"
FOLDER_EVENTOS = "eventos"

SCOPES = ['https://www.googleapis.com/auth/drive']
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "alertas_oficiales_tiempo_real"
TOKEN_PATH = BASE_DIR / "token_drive.json"


# ── Cliente de Google Drive ──────────────────────────────────────

def get_drive_service():
    token_json_content = os.getenv("GDRIVE_TOKEN_JSON")
    if token_json_content:
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(token_json_content)

    if not TOKEN_PATH.exists():
        raise RuntimeError("token_drive.json no encontrado.")

    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())

    return build('drive', 'v3', credentials=creds)


# ── Helpers de Drive ─────────────────────────────────────────────

def get_or_create_folder(service, nombre: str, parent_id: str = None) -> str:
    """Busca una carpeta por nombre (y parent si se indica). La crea si no existe."""
    query = (
        f"name = '{nombre}' "
        f"and mimeType = 'application/vnd.google-apps.folder' "
        f"and trashed = false"
    )
    if parent_id:
        query += f" and '{parent_id}' in parents"

    result = service.files().list(
        q=query,
        fields='files(id, name)'
    ).execute()

    archivos = result.get('files', [])
    if archivos:
        return archivos[0]['id']

    metadata = {
        'name': nombre,
        'mimeType': 'application/vnd.google-apps.folder',
    }
    if parent_id:
        metadata['parents'] = [parent_id]

    folder = service.files().create(
        body=metadata,
        fields='id'
    ).execute()

    print(f"  Carpeta creada: {nombre}")
    return folder.get('id')


def subir_o_sobreescribir_parquet(service, folder_id: str, nombre: str, df: pd.DataFrame):
    """
    Sube un DataFrame como parquet a Drive.
    Si ya existe un archivo con ese nombre en la carpeta, lo sobreescribe.
    Si no existe, lo crea.
    """
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    # Buscar si ya existe
    query = (
        f"name = '{nombre}' "
        f"and '{folder_id}' in parents "
        f"and trashed = false"
    )
    result = service.files().list(
        q=query,
        fields='files(id, name)'
    ).execute()

    media = MediaIoBaseUpload(
        buf,
        mimetype='application/octet-stream',
        resumable=False
    )

    archivos = result.get('files', [])
    if archivos:
        # Sobreescribir el existente
        file_id = archivos[0]['id']
        service.files().update(
            fileId=file_id,
            media_body=media,
        ).execute()
        print(f"  [UPDATE] {nombre} sobreescrito ({len(df)} filas)")
    else:
        # Crear nuevo
        file_metadata = {
            'name': nombre,
            'parents': [folder_id],
        }
        service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print(f"  [CREATE] {nombre} creado ({len(df)} filas)")


def compartir_carpeta(service, folder_id: str):
    """Comparte la carpeta raíz con los emails de GDRIVE_SHARE_EMAILS."""
    emails_raw = os.getenv("GDRIVE_SHARE_EMAILS", "")
    emails = [e.strip() for e in emails_raw.split(",") if e.strip()]
    for email in emails:
        try:
            service.permissions().create(
                fileId=folder_id,
                body={
                    'type': 'user',
                    'role': 'writer',
                    'emailAddress': email
                },
                sendNotificationEmail=False
            ).execute()
        except Exception:
            pass


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  UPLOAD DAILY DATA")
    print(f"  {datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M NY')}")
    print("=" * 55)

    service = get_drive_service()

    # Crear estructura de carpetas
    print("\nCreando estructura de carpetas en Drive...")
    folder_raiz    = get_or_create_folder(service, FOLDER_RAIZ)
    folder_gtfs    = get_or_create_folder(service, FOLDER_GTFS,    parent_id=folder_raiz)
    folder_clima   = get_or_create_folder(service, FOLDER_CLIMA,   parent_id=folder_raiz)
    folder_eventos = get_or_create_folder(service, FOLDER_EVENTOS, parent_id=folder_raiz)

    # Compartir carpeta raíz la primera vez
    compartir_carpeta(service, folder_raiz)

    # ── 1. GTFS Supplemented ──────────────────────────────────────
    print("\n[1/3] GTFS Supplemented...")
    try:
        df_gtfs = creacion_df_previsto()
        subir_o_sobreescribir_parquet(service, folder_gtfs, "stop_times.parquet", df_gtfs)
    except Exception as e:
        print(f"  [ERROR] GTFS: {e}")

    # ── 2. Clima ─────────────────────────────────────────────────
    print("\n[2/3] Clima...")
    try:
        df_clima = load_realtime_weather()
        subir_o_sobreescribir_parquet(service, folder_clima, "clima_hoy.parquet", df_clima)
    except Exception as e:
        print(f"  [ERROR] Clima: {e}")

    # ── 3. Eventos ───────────────────────────────────────────────
    print("\n[3/3] Eventos...")
    try:
        df_eventos = load_realtime_events()
        if not df_eventos.empty:
            subir_o_sobreescribir_parquet(service, folder_eventos, "eventos_hoy.parquet", df_eventos)
        else:
            print("  Sin eventos hoy.")
    except Exception as e:
        print(f"  [ERROR] Eventos: {e}")

    print("\n[OK] Upload diario completado.")


if __name__ == "__main__":
    main()