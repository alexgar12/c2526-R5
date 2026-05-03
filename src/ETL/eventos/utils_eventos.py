"""
utils.py — Funciones compartidas para todos los scripts de ingesta de eventos.

Incluye:
  - Descarga y cálculo de paradas de metro afectadas (Haversine)
  - Fusión de estaciones duplicadas
"""

import io
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _descargar_metro_csv_drive() -> pd.DataFrame:
    """Descarga MTA_Subway_Stations.csv desde Drive (MTA_Daily_Data/GTFS_estatic/)."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    token_path = (
        Path(__file__).resolve().parent.parent.parent
        / "ETL" / "alertas_oficiales_tiempo_real" / "token_drive.json"
    )
    token_json = os.getenv("GDRIVE_TOKEN_JSON")
    if token_json:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(token_json)

    creds = Credentials.from_authorized_user_file(
        str(token_path), ["https://www.googleapis.com/auth/drive"]
    )
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json())

    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    def _folder_id(name, parent=None):
        q = (
            f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder'"
            f" and trashed = false"
        )
        if parent:
            q += f" and '{parent}' in parents"
        res = service.files().list(q=q, fields="files(id)").execute()
        files = res.get("files", [])
        if not files:
            raise FileNotFoundError(f"Carpeta Drive '{name}' no encontrada")
        return files[0]["id"]

    root_id = _folder_id("MTA_Daily_Data")
    folder_id = _folder_id("GTFS_estatic", parent=root_id)

    res = service.files().list(
        q=f"'{folder_id}' in parents and name = 'MTA_Subway_Stations.csv' and trashed = false",
        fields="files(id)",
        pageSize=1,
    ).execute()
    files = res.get("files", [])
    if not files:
        raise FileNotFoundError("MTA_Subway_Stations.csv no encontrado en Drive/GTFS_estatic")

    req = service.files().get_media(fileId=files[0]["id"])
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    return pd.read_csv(buf)


#  Paradas de metro
def cargar_paradas_df():
    """
    Descarga el CSV de paradas del metro de NY desde Drive y lo devuelve como
    DataFrame con columnas: nombre, lineas, lon, lat.
    Devuelve None si la descarga falla.
    """
    try:
        df = _descargar_metro_csv_drive()
        df = df[["Stop Name", "Daytime Routes", "GTFS Longitude", "GTFS Latitude"]].copy()
        df = df.rename(columns={
            "Stop Name":      "nombre",
            "Daytime Routes": "lineas",
            "GTFS Longitude": "lon",
            "GTFS Latitude":  "lat",
        })
        return df
    except Exception as exc:
        print(f"[utils] Error descargando paradas de metro: {exc}")
        return None


def obtener_paradas_afectadas(coords, df_paradas, max_metros=500):
    """
    Devuelve las paradas de metro a menos de max_metros metros de coords.

    Parámetros
    ----------
    coords      : (longitud, latitud) del evento.
    df_paradas  : DataFrame con columnas lon, lat, nombre, lineas.
    max_metros  : radio de búsqueda en metros.

    Devuelve
    -------
    Lista de tuplas [(nombre_parada, lineas)].
    """
    if not coords or None in coords or df_paradas is None or df_paradas.empty:
        return []

    lon_ev, lat_ev = coords

    # Haversine formula
    lat1 = np.radians(lat_ev)
    lon1 = np.radians(lon_ev)
    lat2 = np.radians(df_paradas["lat"].to_numpy())
    lon2 = np.radians(df_paradas["lon"].to_numpy())

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    distancias = 2 * np.arcsin(np.sqrt(a)) * 6371000  

    cercanas = df_paradas[distancias <= max_metros]
    return [(row["nombre"], row["lineas"]) for _, row in cercanas.iterrows()]


def fusionar_lista_estaciones(lista_tuplas):
    """
    Agrupa las tuplas (nombre, lineas) por nombre de estación,
    unificando las líneas en un único string ordenado.
    """
    if not isinstance(lista_tuplas, list) or not lista_tuplas:
        return lista_tuplas

    fusionadas = defaultdict(set)
    for nombre, lineas in lista_tuplas:
        fusionadas[nombre].update(str(lineas).split())

    return [
        (nombre, " ".join(sorted(lineas_set)))
        for nombre, lineas_set in fusionadas.items()
    ]