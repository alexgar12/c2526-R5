"""
Transformación de alertas oficiales de la MTA. Coordinado con el orquestador.

Pasos del pipeline:
  1. Lee el fichero RAW (JSON) desde MinIO.
  2. Aplica una transformación básica (parseo de fechas, deduplicación).
  3. Sube una partición diaria a la capa 'processed' en MinIO.
  4. Aplica limpieza y enriquecimiento final (agrupación, categorización, renombrado).
  5. Sube la partición diaria a la capa 'cleaned' en MinIO.

Rutas MinIO:
  RAW       : grupo5/raw/official_alerts/range=<start>_to_<end>/alertas_oficiales_2025.json
  PROCESSED : grupo5/processed/official_alerts/date=<YYYY-MM-DD>/alerts.parquet
  CLEANED   : grupo5/cleaned/official_alerts/date=<YYYY-MM-DD>/alerts.parquet

Dependencias:
  - pandas
  - src.common.minio_client : download_json, upload_df_parquet

Variables de entorno requeridas:
  - MINIO_ACCESS_KEY
  - MINIO_SECRET_KEY
"""

import os
import pandas as pd
from src.common.minio_client import download_json, upload_df_parquet

BUCKET = "pd1"

RAW_BASE = "grupo5/raw/official_alerts"
PROCESSED_BASE = "grupo5/processed/official_alerts"
CLEANED_BASE = "grupo5/cleaned/official_alerts"


def upload_processed_day(access_key: str, secret_key: str, df_day: pd.DataFrame, day_date):
    """
    Sube el Parquet diario a la capa PROCESSED de MinIO.

    Si el DataFrame está vacío, sube un archivo marcador '_empty.parquet' para
    dejar constancia de que el día fue procesado sin datos.

    Parámetros
    ----------
    access_key : Clave de acceso a MinIO.
    secret_key : Clave secreta de MinIO.
    df_day     : DataFrame con los datos del día.
    day_date   : Objeto date que identifica la partición.
    """
    processed_prefix = f"{PROCESSED_BASE}/date={day_date}"

    if df_day.empty:
        upload_df_parquet(
            access_key,
            secret_key,
            f"{processed_prefix}/_empty.parquet",
            pd.DataFrame()
        )
        print(f"[processed] Carpeta creada sin datos: {day_date}")
    else:
        upload_df_parquet(
            access_key,
            secret_key,
            f"{processed_prefix}/alerts.parquet",
            df_day
        )


def agrupar_alertas(df):
    """
    Consolida actualizaciones repetidas de una misma alerta en una sola fila.

    Varias filas del DataFrame pueden compartir el mismo event_id, lo que indica
    que se trata de la misma alerta actualizada en distintos momentos. Esta función
    agrupa por (event_id, status_label, affected, header) y produce una fila única
    con el timestamp inicial y final, el número de actualizaciones y la última
    descripción disponible.

    Parámetros
    ----------
    df : DataFrame con columnas event_id, status_label, affected, header, date,
         agency y description.

    Devuelve
    --------
    DataFrame agrupado y ordenado por timestamp_inicial.
    """
    df_grouped = (
        df.groupby(
            ["event_id", "status_label", "affected", "header"],
            as_index=False
        )
        .agg(
            timestamp_inicial=("date", "min"),
            timestamp_final=("date", "max"),
            agency=("agency", "first"),
            description=("description", "last"),
            num_updates=("date", "count")
        )
    )
    df_grouped["num_updates"] = df_grouped["num_updates"] - 1
    df_grouped["num_updates"] = df_grouped["num_updates"].clip(lower=0)
    df_grouped = df_grouped.sort_values("timestamp_inicial")
    return df_grouped


def map_category(status):
    """
    Clasifica el status_label en una categoría única priorizando los casos más graves.

    Algunas filas contienen más de un valor en status_label; esta función aplica
    una jerarquía de prioridad: suspensiones > retrasos severos > retrasos >
    cambios de servicio > cancelaciones > otros.

    Parámetros
    ----------
    status : Cadena con el status_label de la alerta (puede ser NaN).

    Devuelve
    --------
    Cadena con la categoría asignada.
    """
    if pd.isna(status):
        return "Other"
    status = status.lower()
    if "suspended" in status or "part-suspended" in status:
        return "Suspension"
    elif "severe-delays" in status:
        return "Severe Delay"
    elif "delay" in status:
        return "Delay"
    elif "reroute" in status or "express-to-local" in status or "stops-skipped" in status or "boarding-change" in status:
        return "Service Change"
    elif "cancellations" in status:
        return "Cancellation"
    else:
        return "Other"


def upload_cleaned_day(access_key: str, secret_key: str, df_day: pd.DataFrame, day_date):
    """
    Aplica limpieza final y sube el Parquet diario a la capa CLEANED de MinIO.

    El objetivo es que el esquema resultante sea lo más compatible posible con el
    DataFrame de alertas en tiempo real. Los pasos son:
      1. Filtrar solo alertas del metro (agency == 'NYCT Subway').
      2. Agrupar actualizaciones repetidas con `agrupar_alertas`.
      3. Categorizar el status_label con `map_category`.
      4. Renombrar y reordenar columnas para alinearse con el esquema realtime.
      5. Sustituir el separador '|' por comas en la columna de líneas.

    Parámetros
    ----------
    access_key : Clave de acceso a MinIO.
    secret_key : Clave secreta de MinIO.
    df_day     : DataFrame con los datos del día (capa processed).
    day_date   : Objeto date que identifica la partición.
    """
    cleaned_prefix = f"{CLEANED_BASE}/date={day_date}"

    # Solo alertas del metro de Nueva York
    df_day_clean = df_day[df_day["agency"] == "NYCT Subway"].copy()

    if df_day_clean.empty:
        upload_df_parquet(
            access_key,
            secret_key,
            f"{cleaned_prefix}/_empty.parquet",
            pd.DataFrame()
        )
        print(f"[cleaned] Carpeta creada sin datos: {day_date}")
    else:
        df_day_clean = agrupar_alertas(df_day_clean)
        df_day_clean["category"] = df_day_clean["status_label"].apply(map_category)
        df_day_clean.columns = df_day_clean.columns.str.strip()
        df_day_clean = df_day_clean.drop(
            columns=["agency", "status_label"],
            errors="ignore"
        )
        # Renombrado de columnas para alinearse con el esquema del DataFrame realtime
        df_day_clean = df_day_clean.rename(columns={
            "timestamp_inicial": "timestamp_start",
            "timestamp_final": "timestamp_end",
            "affected": "lines",
            "header": "text_snippet"
        })
        # Reordenar columnas
        df_day_clean = df_day_clean[
            [
                "event_id",
                "timestamp_start",
                "timestamp_end",
                "category",
                "lines",
                "text_snippet",
                "description"
            ]
        ]
        # En la columna lines se sustituye el separador '|' por comas
        df_day_clean["lines"] = (
            df_day_clean["lines"]
                .str.replace(r"\s*\|\s*", ", ", regex=True)
        )
        upload_df_parquet(
            access_key,
            secret_key,
            f"{cleaned_prefix}/alerts.parquet",
            df_day_clean
        )


def run_transform(start: str, end: str):
    """
    Punto de entrada llamado por el orquestador run_transform.

    Descarga el JSON RAW de MinIO, lo convierte en DataFrame, elimina duplicados
    y filas con fecha inválida, y luego itera día a día para subir las capas
    processed y cleaned.

    Parámetros
    ----------
    start : Fecha de inicio en formato 'YYYY-MM-DD'.
    end   : Fecha de fin en formato 'YYYY-MM-DD'.

    Lanza
    -----
    ValueError si las variables de entorno de MinIO no están definidas.
    """
    print(f"[alertas_transform] START start={start} end={end}")

    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    if not access_key or not secret_key:
        raise ValueError(
            "MINIO_ACCESS_KEY y MINIO_SECRET_KEY deben estar definidas."
        )

    raw_file = (
        f"{RAW_BASE}/"
        f"range={start}_to_{end}/"
        f"alertas_oficiales_2025.json"
    )

    print("[alertas_transform] Descargando RAW...")

    data = download_json(
        access_key=access_key,
        secret_key=secret_key,
        object_name=raw_file
    )

    df = pd.DataFrame(data)

    print(f"[alertas_transform] Registros RAW: {len(df)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates()

    print(f"[alertas_transform] Registros tras transform: {len(df)}")

    all_days = pd.date_range(start, end, freq="D")

    for day in all_days:
        day_date = day.date()

        df_day = df[df["date"].dt.date == day_date]

        upload_processed_day(access_key, secret_key, df_day, day_date)
        upload_cleaned_day(access_key, secret_key, df_day, day_date)

    print("[alertas_transform] DONE")
