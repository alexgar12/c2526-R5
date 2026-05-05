"""
Módulo de ingesta histórica de alertas oficiales de la MTA.

Descarga alertas históricas del dataset público data.ny.gov (recurso 7kct-peq7)
para un rango de fechas dado y las sube como JSON a la capa RAW de MinIO.

Funcionamiento:
  1. La función `fetch_data` consulta la API con paginación hasta obtener todos
     los registros del período solicitado.
  2. La función `ingest_alertas` valida credenciales, descarga los datos y los
     sube a MinIO bajo la ruta estructurada por rango de fechas.

Dependencias:
  - requests         : peticiones HTTP a la API de NY Open Data
  - src.common.minio_client.upload_json : subida del JSON a MinIO

Variables de entorno requeridas:
  - MINIO_ACCESS_KEY
  - MINIO_SECRET_KEY

Nota: La función `ingest_alertas` es el punto de entrada llamado por el
orquestador del pipeline (run_extraccion).
"""

import requests
import os
import json
from datetime import datetime
from typing import List, Dict
from src.common.minio_client import upload_json

BASE_URL = "https://data.ny.gov/resource/7kct-peq7.json"
MINIO_BASE_PATH = "grupo5/raw/official_alerts"


def fetch_data(start_date: str, end_date: str, limit: int = 50000):
    """
    Descarga todos los registros de alertas oficiales entre dos fechas usando paginación.

    Itera con offset incremental de `limit` registros por petición hasta que la API
    devuelve una respuesta vacía, momento en que se detiene.

    Parámetros
    ----------
    start_date : Fecha de inicio en formato 'YYYY-MM-DDTHH:MM:SS.sss'.
    end_date   : Fecha de fin en formato 'YYYY-MM-DDTHH:MM:SS.sss'.
    limit      : Número máximo de registros por petición (por defecto 50 000).

    Devuelve
    --------
    Lista de diccionarios con todos los registros descargados.
    """
    all_results = []
    offset = 0

    while True:
        params = {
            "$where": f"date between '{start_date}' and '{end_date}'",
            "$limit": limit,
            "$offset": offset
        }

        print(f"[alertas] Descargando offset={offset}")

        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()

        if not data:
            break

        all_results.extend(data)
        offset += limit

    return all_results


def ingest_alertas(start: str, end: str) -> None:
    """
    Punto de entrada principal llamado por el orquestador del pipeline.

    Valida las credenciales de MinIO, descarga los datos históricos en el rango
    indicado y sube el JSON resultante a la ruta estructurada en MinIO.

    Parámetros
    ----------
    start : Fecha de inicio en formato 'YYYY-MM-DD'.
    end   : Fecha de fin en formato 'YYYY-MM-DD'.

    Lanza
    -----
    ValueError si las variables de entorno de MinIO no están definidas.
    """
    print(f"[alertas] START start={start} end={end}")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    if not access_key or not secret_key:
        raise ValueError(
            "Las variables de entorno MINIO_ACCESS_KEY y MINIO_SECRET_KEY deben estar definidas."
        )
    data = fetch_data(start, end)
    object_name = (
        f"{MINIO_BASE_PATH}/"
        f"range={start}_to_{end}/"
        f"alertas_oficiales_2025.json"
    )
    upload_json(
        access_key=access_key,
        secret_key=secret_key,
        object_name=object_name,
        data=data
    )
    print(f"[alertas] Subido a Minio://pd1/{object_name}")

    print("[alertas] DONE")
