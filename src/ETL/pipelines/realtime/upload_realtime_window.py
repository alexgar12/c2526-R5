"""
Orquesta la ejecución del pipeline RT y sube la ventana agregada más reciente
a Google Drive, manteniendo solo las N ventanas más recientes (ventana deslizante).

Flujo:
    La carpeta la crea y gestiona la cuenta de servicio en su propio Drive.
    Se comparte automáticamente con los emails configurados en GDRIVE_SHARE_EMAILS.

Uso:
    uv run python src/ETL/pipelines/realtime/upload_realtime_window.py

Variables de entorno requeridas:
    GOOGLE_CREDENTIALS_JSON  ← contenido del JSON de la cuenta de servicio
    GDRIVE_SHARE_EMAILS      ← emails separados por coma
"""

import io
import json
import os
from pathlib import Path
from time import time

import pandas as pd
from dotenv import load_dotenv

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from src.ETL.pipelines.realtime.generate_realtime_dataset import build_realtime_dataset
from src.ETL.pipelines.realtime.aggregate_realtime_dataset import aggregate_realtime


load_dotenv()

# Configuración
FOLDER_NAME = "MTA_Realtime_Windows"
NUM_VENTANAS_A_MANTENER = 8 
TIEMPO_VENTANA = "15"   
SCOPES = ['https://www.googleapis.com/auth/drive']

BASE_DIR = Path(__file__).resolve()
while BASE_DIR.name != "c2526-R5":
    BASE_DIR = BASE_DIR.parent
BASE_DIR = BASE_DIR / "src" / "ETL" / "alertas_oficiales_tiempo_real"
TOKEN_PATH = BASE_DIR / "token_drive.json"


def get_drive_service():
    """
    Crea el cliente de Google Drive usando OAuth del usuario.
    En GitHub Actions, reconstruye token_drive.json desde el secret
    GDRIVE_TOKEN_JSON antes de crear el servicio.
    """
    # Si estamos en GitHub Actions, reconstruir el token desde el secret
    token_json_content = os.getenv("GDRIVE_TOKEN_JSON")
    if token_json_content:
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(token_json_content)

    if not TOKEN_PATH.exists():
        raise RuntimeError(
            "token_drive.json no encontrado. "
            "Ejecuta generar_token_drive.py localmente para generarlo."
        )

    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    # Refrescar si ha expirado
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())

    return build('drive', 'v3', credentials=creds)


def get_or_create_folder(service, folder_name: str) -> str:
    """
    Busca una carpeta con ese nombre en el Drive de la cuenta de servicio.
    Si no existe, la crea y la comparte con los emails configurados.
    Devuelve el folder_id.
    """
    query = (
        f"name = '{folder_name}' "
        f"and mimeType = 'application/vnd.google-apps.folder' "
        f"and trashed = false"
    )
    result = service.files().list(
        q=query,
        fields='files(id, name)'
    ).execute()

    archivos = result.get('files', [])
    if archivos:
        folder_id = archivos[0]['id']
        print(f"  Carpeta encontrada: {folder_name} (id={folder_id})")
        return folder_id

    # Crear la carpeta
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = service.files().create(
        body=file_metadata,
        fields='id'
    ).execute()
    folder_id = folder.get('id')
    print(f"  Carpeta creada: {folder_name} (id={folder_id})")

    # Compartir con los emails configurados
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
            print(f"  Carpeta compartida con: {email}")
        except Exception as e:
            print(f"  [WARN] No se pudo compartir con {email}: {e}")

    return folder_id


def listar_archivos_drive(service, folder_id: str) -> list[dict]:
    """Lista archivos parquet en la carpeta de Drive, ordenados por fecha de creación."""
    query = (
        f"'{folder_id}' in parents "
        f"and name contains 'ventana_' "
        f"and name contains '.parquet' "
        f"and trashed = false"
    )
    result = service.files().list(
        q=query,
        orderBy='createdTime',
        fields='files(id, name, createdTime)'
    ).execute()
    return result.get('files', [])


def subir_parquet_drive(service, folder_id: str, nombre: str, df: pd.DataFrame) -> str:
    """Sube un DataFrame como parquet a Google Drive. Devuelve el file_id."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    file_metadata = {
        'name': nombre,
        'parents': [folder_id],
    }
    media = MediaIoBaseUpload(buf, mimetype='application/octet-stream', resumable=False)
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    return file.get('id')


def borrar_archivo_drive(service, file_id: str) -> None:
    """Borra un archivo de Google Drive por su ID."""
    service.files().delete(fileId=file_id).execute()


def main():
    credentials_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if not credentials_json:
        raise RuntimeError("GOOGLE_CREDENTIALS_JSON no está definida.")

    # 1) Generar dataset RT
    print("\n[1/3] Generando dataset RT...")
    df_rt = build_realtime_dataset()

    # 2) Agregar en ventanas
    print("\n[2/3] Agregando en ventanas temporales...")
    df_agregado = aggregate_realtime(df_rt, tiempo=TIEMPO_VENTANA)

    if df_agregado.empty:
        print("[WARN] Agregado vacío. No se sube nada a Drive.")
        return

    # 3) Obtener o crear carpeta y subir
    service = get_drive_service()
    folder_id = get_or_create_folder(service, FOLDER_NAME)

    ventana = df_agregado['merge_time'].iloc[0]
    nombre = f"ventana_{ventana.strftime('%Y-%m-%d_%H-%M')}.parquet"

    print(f"\n[3/3] Subiendo a Google Drive: {nombre}")
    file_id = subir_parquet_drive(service, folder_id, nombre, df_agregado)
    print(f"  [OK] Subido: {len(df_agregado)} filas → file_id={file_id}")

    # Esperar a que Drive indexe el archivo recién subido
    max_intentos = 5
    for intento in range(max_intentos):
        archivos = listar_archivos_drive(service, folder_id)
        nombres = [f['name'] for f in archivos]
        if nombre in nombres:
            break
        print(f"  Esperando indexación Drive... ({intento+1}/{max_intentos})")
        time.sleep(3)

    # 4) Mantener solo las N ventanas más recientes (ventana deslizante)
    print(f"\n[CLEANUP] Conservando solo las {NUM_VENTANAS_A_MANTENER} ventanas más recientes...")
    archivos = listar_archivos_drive(service, folder_id)

    if len(archivos) > NUM_VENTANAS_A_MANTENER:
        a_borrar = archivos[:-NUM_VENTANAS_A_MANTENER]
        for f in a_borrar:
            borrar_archivo_drive(service, f['id'])
            print(f"  [DEL] {f['name']}")

    # Mostrar las ventanas conservadas
    archivos_finales = listar_archivos_drive(service, folder_id)
    print(f"\n  Ventanas en Drive ({len(archivos_finales)}):")
    for f in archivos_finales:
        print(f"    - {f['name']}")


if __name__ == "__main__":
    main()