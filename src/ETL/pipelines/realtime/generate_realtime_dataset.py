"""
Construye el dataset final de inferencia en tiempo real a partir de las
fuentes en vivo del metro de Nueva York.

  GTFS cleaned  →  realtime_data.py         (retrasos actuales por parada)
  Clima cleaned →  clima_realtime.py         (condiciones actuales Open-Meteo)
  Eventos clean →  ingest_actual_eventos.py  (SeatGeek + NYC Open Data + ESPN)
  Alertas clean →  extract_alertas_oficiales_tiempo_real.py  (Gmail MTA)


Uso:
  uv run python src/ETL/pipelines/realtime/generate_realtime_dataset.py

"""

import base64
import gc
import io
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import openmeteo_requests
import pandas as pd
import requests
from retry_requests import retry
from scipy import stats

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from src.ETL.tiempo_real_metro.realtime_data import (
    creacion_df_tiempo_real,
    creacion_df_previsto,
    union_dataframes,
    filter_delay_outliers,
    hora_ciclica,
)
from src.ETL.clima.clima_realtime import extraer_clima_actual
from src.ETL.alertas_oficiales_tiempo_real.extract_alertas_oficiales_tiempo_real import (
    get_gmail_service,
    parse_mta_body,
)

# ── Drive config ──────────────────
_SCOPES = ['https://www.googleapis.com/auth/drive']
_BASE_DIR = Path(__file__).resolve().parent.parent.parent / "alertas_oficiales_tiempo_real"
_TOKEN_PATH = _BASE_DIR / "token_drive.json"


def _get_drive_service():
    token_json_content = os.getenv("GDRIVE_TOKEN_JSON")
    if token_json_content:
        _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TOKEN_PATH.write_text(token_json_content)
    if not _TOKEN_PATH.exists():
        raise RuntimeError("token_drive.json no encontrado.")
    creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), _SCOPES)
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _TOKEN_PATH.write_text(creds.to_json())
    return build('drive', 'v3', credentials=creds)


# ──────────────────────────────────────────────────────────────
#  Helpers de generate_final_dataset.py
# ──────────────────────────────────────────────────────────────

def normalize_route_id(series: pd.Series) -> pd.Series:
    """Normaliza códigos de línea/ruta (trim + mayúsculas)."""
    return series.astype("string").str.strip().str.upper()


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memoria: object→category si baja cardinalidad, num→int32/float32."""
    if df.empty:
        return df
    for col in df.columns:
        col_data = df[col]
        dtype = col_data.dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            total = len(col_data)
            if total == 0:
                continue
            try:
                ratio = col_data.nunique(dropna=True) / total
            except TypeError:
                continue
            if ratio < 0.5:
                df[col] = col_data.astype("category")
            continue
        if (
            pd.api.types.is_datetime64_any_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            continue
        if pd.api.types.is_numeric_dtype(dtype):
            c_min = col_data.min(skipna=True)
            c_max = col_data.max(skipna=True)
            if pd.isna(c_min) or pd.isna(c_max):
                continue
            if pd.api.types.is_integer_dtype(dtype):
                if c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = col_data.astype(
                        "Int32" if pd.api.types.is_extension_array_dtype(dtype) else np.int32
                    )
            else:
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = col_data.astype(np.float32)
    return df


def _split_lines(value: object) -> list:
    """Convierte el campo `lines` (string/lista) en lista limpia de route_id."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip().upper() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [p.strip().upper() for p in value.replace("|", ",").split(",") if p.strip()]
    return []


def _time_str_to_seconds(t: object) -> float:
    """Convierte HH:MM[:SS] a segundos desde medianoche."""
    if pd.isna(t):
        return np.nan
    parts = str(t).split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + (int(parts[2]) if len(parts) > 2 else 0)


# ──────────────────────────────────────────────────────────────
#  1. GTFS en tiempo real
# ──────────────────────────────────────────────────────────────

def load_realtime_gtfs(df_previsto: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Extrae los retrasos actuales del metro y los adapta al esquema que
    espera generate_final_dataset (columnas: stop_id, route_id, direction,
    delay_seconds, merge_time, hour, dow, is_weekend, hour_sin, hour_cos).

    Args:
        df_previsto: stop_times pre-cargado (e.g. desde Drive). Si es None,
                     se descarga el ZIP del GTFS estático.
    """
    print("  [GTFS RT] Extrayendo datos en tiempo real...")
    df_real = creacion_df_tiempo_real()

    if df_previsto is None:
        print("  [GTFS RT] Descargando horarios previstos...")
        df_previsto = creacion_df_previsto()
    else:
        print("  [GTFS RT] Usando horarios previstos pre-cargados.")

    print("  [GTFS RT] Calculando retrasos...")
    df = union_dataframes(df_real, df_previsto)

    if df.empty:
        raise ValueError("El DataFrame GTFS en tiempo real está vacío tras la unión.")

    # ── Renombrar/crear columnas para el esquema esperado ──
    rename_map = {}
    if "linea_id" in df.columns:
        rename_map["linea_id"] = "route_id"
    if "parada_id" in df.columns:
        rename_map["parada_id"] = "stop_id"
    if "delay" in df.columns:
        rename_map["delay"] = "delay_seconds"
    if "direccion" in df.columns:
        rename_map["direccion"] = "direction"
    df = df.rename(columns=rename_map)

    # merge_time: momento real de la observación
    if "hora_llegada" in df.columns:
        df["merge_time"] = pd.to_datetime(df["hora_llegada"], errors="coerce")
    else:
        df["merge_time"] = pd.Timestamp.now(tz="America/New_York")

    df["hour"] = df["merge_time"].dt.hour

    # Columnas requeridas para  merge
    df["date"] = df["merge_time"].dt.date
    df["service_date"] = df["merge_time"].dt.strftime("%Y-%m-%d")

    # actual_seconds: segundos desde medianoche para cruce con eventos
    df["actual_seconds"] = (
        df["merge_time"].dt.hour * 3600
        + df["merge_time"].dt.minute * 60
        + df["merge_time"].dt.second
    )

    # is_unscheduled siempre False para datos RT (son viajes programados)
    if "is_unscheduled" not in df.columns:
        df["is_unscheduled"] = False

    # match_key: identificador único del viaje (trip_id + stop_id si existe)
    if "viaje_id" in df.columns:
        df["match_key"] = df["viaje_id"].astype(str)

    if "route_id" in df.columns:
        df["route_id"] = normalize_route_id(df["route_id"])
    if "stop_id" in df.columns:
        df["stop_id"] = df["stop_id"].astype("string")

    # Rellenar route_id cuando viene vacío, parseándolo del match_key.
    # Los viajes unscheduled a veces llegan sin route_id en el feed.
    mask = df["route_id"].isna() | (df["route_id"].astype(str) == "")
    if mask.any():
        df.loc[mask, "route_id"] = (
            df.loc[mask, "match_key"]
            .astype(str)
            .str.split("_").str[1]        # "033150_2..N08R" -> "2..N08R"
            .str.split(".").str[0]        # "2..N08R" -> "2"
        )
        df["route_id"] = normalize_route_id(df["route_id"])

    print(f"  [GTFS RT] {len(df)} filas, {df['route_id'].nunique()} líneas.")
    return df


# ──────────────────────────────────────────────────────────────
#  2. Clima en tiempo real
# ──────────────────────────────────────────────────────────────

def load_realtime_weather() -> pd.DataFrame:
    """
    Obtiene las condiciones meteorológicas actuales de Open-Meteo y añade
    las features de impacto que genera el transform histórico de clima:
      temp_extreme, is_freezing, is_high_wind, precip_3h_accum, apparent_temp,
      hour, is_rush_hour.
    """
    print("  [CLIMA RT] Extrayendo clima actual...")

    # extraer_clima_actual() devuelve el DataFrame horario + current al final
    # Lo reimplementamos aquí directamente para no depender del efecto secundario
    # de subida a MinIO que contiene esa función.

    session = requests.Session()
    retry_session = retry(session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 40.78,
        "longitude": -73.97,
        "hourly": [
            "temperature_2m", "rain", "precipitation",
            "wind_speed_10m", "snowfall", "cloud_cover",
        ],
        "current": [
            "wind_speed_10m", "temperature_2m", "precipitation",
            "rain", "snowfall", "cloud_cover",
        ],
        "timezone": "America/New_York",
        "forecast_days": 1,
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Hora actual
    current = response.Current()
    current_time = datetime.fromtimestamp(current.Time(), tz=timezone.utc)

    hourly = response.Hourly()
    hourly_data = {
        "Date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
        "Temperature":  hourly.Variables(0).ValuesAsNumpy(),
        "Rain":         hourly.Variables(1).ValuesAsNumpy(),
        "Precipitation":hourly.Variables(2).ValuesAsNumpy(),
        "Wind Speed":   hourly.Variables(3).ValuesAsNumpy(),
        "Snow":         hourly.Variables(4).ValuesAsNumpy(),
        "Cloud Cover":  hourly.Variables(5).ValuesAsNumpy(),
    }

    df = pd.DataFrame(hourly_data)

    # Añadir fila con datos "current"
    current_row = pd.DataFrame([{
        "Date":        current_time,
        "Temperature": current.Variables(1).Value(),
        "Rain":        current.Variables(3).Value(),
        "Precipitation": current.Variables(2).Value(),
        "Wind Speed":  current.Variables(0).Value(),
        "Snow":        current.Variables(4).Value(),
        "Cloud Cover": current.Variables(5).Value(),
    }])
    df = pd.concat([df, current_row], ignore_index=True)

    # ── Feature engineering ──
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert("America/New_York")
    df = df.dropna(subset=["Date", "Temperature"]).drop_duplicates()

    # Outliers de temperatura (z-score)
    if len(df) > 5:
        z = np.abs(stats.zscore(df["Temperature"]))
        df = df[z < 3]

    df = df.sort_values("Date")

    # Sensación térmica
    def _apparent_temp(t, ws):
        return 13.12 + 0.6215 * t - 11.37 * (ws ** 0.16) + 0.3965 * t * (ws ** 0.16)

    df["apparent_temp"] = df.apply(
        lambda x: _apparent_temp(x["Temperature"], x["Wind Speed"]), axis=1
    )

    # Precipitación acumulada 3h
    df["precip_3h_accum"] = df["Precipitation"].rolling(window=3, min_periods=1).sum()

    # Flags de riesgo
    df["is_freezing"]  = (df["Temperature"] <= 0).astype(int)
    df["is_high_wind"] = (df["Wind Speed"] > 50).astype(int)

    # temp_extreme: percentiles 10/90 si hay suficientes datos, sino umbrales fijos
    if len(df) >= 10:
        low  = df["Temperature"].quantile(0.10)
        high = df["Temperature"].quantile(0.90)
    else:
        low, high = -5.0, 35.0
    df["temp_extreme"] = ((df["Temperature"] < low) | (df["Temperature"] > high)).astype(int)

    # Variables temporales para JOIN con GTFS
    df["hour"]         = df["Date"].dt.hour
    df["date"]         = df["Date"].dt.date
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    df = df.drop_duplicates(subset=["date", "hour"], keep="last")

    print(f"  [CLIMA RT] {len(df)} registros horarios.")
    return df


# ──────────────────────────────────────────────────────────────
#  3. Eventos en tiempo real 
# ──────────────────────────────────────────────────────────────


def load_realtime_events() -> pd.DataFrame:
    """
    Lee eventos_hoy.parquet desde Google Drive (MTA_Daily_Data/eventos/).
    El archivo lo genera y sube ingest_actual_eventos.ingest_eventos() a través
    de upload_daily_data.py, y ya incluye todas las transformaciones necesarias
    para el merge con GTFS (stop_id, parada_nombre, parada_lineas, etc.).

    Si el archivo no existe en Drive o falla la lectura, devuelve DataFrame vacío.
    """
    print("  [EVENTOS RT] Extrayendo eventos del día...")

    try:
        service = _get_drive_service()

        # Buscar carpeta MTA_Daily_Data/eventos
        folder_query = (
            "name = 'eventos' "
            "and mimeType = 'application/vnd.google-apps.folder' "
            "and trashed = false"
        )
        folder_result = service.files().list(
            q=folder_query, fields='files(id, name)'
        ).execute()
        folders = folder_result.get('files', [])
        if not folders:
            print("  [EVENTOS RT] Carpeta 'eventos' no encontrada en Drive.")
            return pd.DataFrame()

        folder_id = folders[0]['id']

        # Buscar el archivo eventos_hoy.parquet
        file_query = (
            f"name = 'eventos_hoy.parquet' "
            f"and '{folder_id}' in parents "
            f"and trashed = false"
        )
        file_result = service.files().list(
            q=file_query, fields='files(id, name)'
        ).execute()
        files = file_result.get('files', [])
        if not files:
            print("  [EVENTOS RT] eventos_hoy.parquet no encontrado en Drive.")
            return pd.DataFrame()

        file_id = files[0]['id']

        # Descargar y leer como parquet
        content = service.files().get_media(fileId=file_id).execute()
        df = pd.read_parquet(io.BytesIO(content))

        print(f"  [EVENTOS RT] {len(df)} filas cargadas desde Drive.")
        return df

    except Exception as e:
        print(f"  [EVENTOS RT] Error leyendo desde Drive: {e}. Devolviendo vacío.")
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────
#  4. Alertas en tiempo real 
# ──────────────────────────────────────────────────────────────

def load_realtime_alerts() -> pd.DataFrame:
    """
    Lee los correos de alerta MTA de los últimos 30 minutos desde Gmail
    y los convierte al esquema de cleaned/official_alerts:
      timestamp_start, timestamp_end, category, lines, text_snippet,
      description, num_updates, event_id.
    """
    print("  [ALERTAS RT] Extrayendo alertas Gmail MTA...")

    try:
        service = get_gmail_service()
    except Exception as e:
        print(f"  [ALERTAS RT] No se pudo autenticar Gmail: {e}")
        return pd.DataFrame()

    cutoff_utc = datetime.now(timezone.utc) - timedelta(minutes=30)
    data_log = []
    page_token = None

    while True:
        results = service.users().messages().list(
            userId="me",
            q="label:mta_alerts newer_than:30m",
            pageToken=page_token,
        ).execute()

        messages = results.get("messages", [])
        if not messages:
            break

        for msg in messages:
            try:
                m = service.users().messages().get(
                    userId="me", id=msg["id"], format="full"
                ).execute()

                ts_utc = pd.to_datetime(int(m["internalDate"]), unit="ms", utc=True)
                if ts_utc.to_pydatetime() < cutoff_utc:
                    continue

                def get_html_part(payload):
                    if payload.get("mimeType") == "text/html":
                        data = payload.get("body", {}).get("data")
                        if not data:
                            return None
                        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                    if "parts" in payload:
                        for part in payload["parts"]:
                            html = get_html_part(part)
                            if html:
                                return html
                    return None

                html_body = get_html_part(m.get("payload", {}))
                if not html_body:
                    continue

                lines, reason, category, location, clean_text = parse_mta_body(html_body)

                data_log.append({
                    "timestamp_start": ts_utc.tz_convert("America/New_York"),
                    "timestamp_end":   ts_utc.tz_convert("America/New_York"),
                    "category":        category,
                    "lines":           lines,
                    "text_snippet":    clean_text,
                    "description":     reason,
                    "num_updates":     0,
                    "event_id":        msg["id"],
                })
            except Exception:
                continue

        page_token = results.get("nextPageToken")
        if not page_token:
            break

    if not data_log:
        print("  [ALERTAS RT] Sin alertas en los últimos 30 min.")
        return pd.DataFrame()

    df = pd.DataFrame(data_log).drop_duplicates(subset=["event_id"])

    # Agrupar correos que representan la misma alerta actualizada.
    # Clave lógica: misma categoría + mismas líneas afectadas + mismo motivo.
    # Colapsamos a una sola fila por alerta lógica con:
    #   - timestamp_start: primer correo (más antiguo)
    #   - timestamp_end:   último correo (más reciente)
    #   - num_updates:     nº de correos adicionales tras el primero
    grupo_keys = ["category", "lines", "description"]
    df_grouped = (
        df.groupby(grupo_keys, as_index=False)
        .agg(
            timestamp_start=("timestamp_start", "min"),
            timestamp_end=("timestamp_start", "max"),
            text_snippet=("text_snippet", "last"),
            event_id=("event_id", "first"),
            num_updates=("event_id", "count"),
        )
    )
    # num_updates cuenta apariciones; restamos 1 para que "0" = sin updates adicionales
    df_grouped["num_updates"] = (df_grouped["num_updates"] - 1).clip(lower=0)

    df_grouped = df_grouped.sort_values("timestamp_start").reset_index(drop=True)
    print(f"  [ALERTAS RT] {len(df_grouped)} alertas lógicas.")
    return df_grouped


# ──────────────────────────────────────────────────────────────
#  5. Cruces 
# ──────────────────────────────────────────────────────────────

def merge_gtfs_weather_rt(df_gtfs: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """LEFT JOIN GTFS + Clima por `date` y `hour`."""
    if df_weather.empty:
        return df_gtfs.copy()
    return df_gtfs.merge(df_weather, how="left", on=["date", "hour"], suffixes=("", "_weather"))


def merge_gtfs_events_rt(df_base: pd.DataFrame, df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Cruce GTFS + Eventos por stop_id + date en ventana temporal de ±1.5h.
    Genera: n_eventos_afectando, tipo_referente, afecta_previo/durante/despues.
    """
    VENTANA = 5400  # segundos

    _defaults = {
        "n_eventos_afectando": 0,
        "tipo_referente":      "Ninguno",
        "afecta_previo":       0,
        "afecta_durante":      0,
        "afecta_despues":      0,
    }

    if df_events.empty:
        return df_base.assign(**_defaults)

    base = df_base.copy()
    evt  = df_events.copy()

    base["service_date"] = pd.to_datetime(base["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    base["stop_id"]      = base["stop_id"].astype("string")
    base["_actual_secs"] = pd.to_numeric(base.get("actual_seconds", np.nan), errors="coerce")
    base["_tren_idx"]    = base.index

    evt["service_date"]   = pd.to_datetime(evt["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    evt["stop_id"]        = evt["stop_id"].astype("string")
    evt["score"]          = pd.to_numeric(evt.get("score", 0), errors="coerce").fillna(0.0)
    evt["_inicio_secs"]   = evt["hora_inicio"].apply(_time_str_to_seconds)
    evt["_fin_secs"]      = evt["hora_salida_estimada"].apply(_time_str_to_seconds)
    tipo_col              = "tipo" if "tipo" in evt.columns else "tipo_evento"
    if tipo_col not in evt.columns:
        evt[tipo_col] = "Desconocido"

    merged = base[["_tren_idx", "stop_id", "service_date", "_actual_secs"]].merge(
        evt[["service_date", "stop_id", tipo_col, "score", "_inicio_secs", "_fin_secs"]],
        on=["stop_id", "service_date"],
        how="inner",
    )

    mask = (
        merged["_actual_secs"].notna()
        & (merged["_actual_secs"] >= merged["_inicio_secs"] - VENTANA)
        & (merged["_actual_secs"] <= merged["_fin_secs"]   + VENTANA)
    )
    afectados = merged[mask].copy()

    if afectados.empty:
        return base.drop(columns=["_tren_idx", "_actual_secs", "service_date"], errors="ignore").assign(**_defaults)

    afectados["es_previo"]  = (afectados["_actual_secs"] < afectados["_inicio_secs"]).astype(int)
    afectados["es_durante"] = (
        (afectados["_actual_secs"] >= afectados["_inicio_secs"])
        & (afectados["_actual_secs"] <= afectados["_fin_secs"])
    ).astype(int)
    afectados["es_despues"] = (afectados["_actual_secs"] > afectados["_fin_secs"]).astype(int)

    agg = pd.concat([
        afectados.groupby("_tren_idx").size().rename("n_eventos_afectando"),
        afectados.sort_values("score", ascending=False).groupby("_tren_idx").first()[tipo_col].rename("tipo_referente"),
        afectados.groupby("_tren_idx")["es_previo"].sum().rename("afecta_previo"),
        afectados.groupby("_tren_idx")["es_durante"].sum().rename("afecta_durante"),
        afectados.groupby("_tren_idx")["es_despues"].sum().rename("afecta_despues"),
    ], axis=1)

    out = base.join(agg, on="_tren_idx")
    out = out.drop(columns=["_tren_idx", "_actual_secs", "service_date"], errors="ignore")
    out["n_eventos_afectando"] = out["n_eventos_afectando"].fillna(0).astype(int)
    out["afecta_previo"]       = out["afecta_previo"].fillna(0).astype(int)
    out["afecta_durante"]      = out["afecta_durante"].fillna(0).astype(int)
    out["afecta_despues"]      = out["afecta_despues"].fillna(0).astype(int)
    out["tipo_referente"]      = out["tipo_referente"].fillna("Ninguno")
    return out


def merge_gtfs_alerts_rt(df_base: pd.DataFrame, df_alerts: pd.DataFrame) -> pd.DataFrame:
    """
    LEFT JOIN temporal GTFS + Alertas con merge_asof por route_id.
    Calcula seconds_since_last_alert, alert_in_next_15m/30m.
    """
    _na_fields = {
        "category":                  pd.NA,
        "num_updates":               0,
        "timestamp_start":           pd.NaT,
        "seconds_since_last_alert":  np.nan,
        "is_alert_just_published":   0,
    }

    if df_alerts.empty:
        return df_base.assign(**_na_fields)

    left  = df_base.copy()
    right = df_alerts[["route_id", "timestamp_start", "category", "num_updates"]].copy() \
        if "route_id" in df_alerts.columns else _prepare_alert_route(df_alerts)

    left["route_id"]  = normalize_route_id(left["route_id"])
    right["route_id"] = normalize_route_id(right["route_id"])
    left["merge_time"] = (
        pd.to_datetime(left["merge_time"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .astype("datetime64[ns, America/New_York]")
    )
    right["timestamp_start"] = (
        pd.to_datetime(right["timestamp_start"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .astype("datetime64[ns, America/New_York]")
    )

    mask_valid   = left["route_id"].notna() & left["merge_time"].notna()
    left_valid   = left[mask_valid].copy()
    left_invalid = left[~mask_valid].assign(**_na_fields)

    right = right.dropna(subset=["route_id", "timestamp_start"])
    left_valid["route_id"]  = left_valid["route_id"].astype("category")
    right["route_id"]       = right["route_id"].astype("category")

    chunks = []
    for route in left_valid["route_id"].unique():
        lc = left_valid[left_valid["route_id"] == route].sort_values("merge_time").copy()
        rc = right[right["route_id"] == route].sort_values("timestamp_start").copy()

        if rc.empty:
            chunks.append(lc.assign(**_na_fields))
            continue

        rc_clean = rc.drop(columns=["route_id"], errors="ignore")
        mc = pd.merge_asof(lc, rc_clean, left_on="merge_time", right_on="timestamp_start",
                        direction="backward", suffixes=("", "_alert"))
        mc["num_updates"] = pd.to_numeric(mc["num_updates"], errors="coerce").fillna(0)
        mc["seconds_since_last_alert"] = (mc["merge_time"] - mc["timestamp_start"]).dt.total_seconds()
        mc["is_alert_just_published"] = (mc["seconds_since_last_alert"] <= 60).astype(int)
        chunks.append(mc)

    if not chunks:
        return left_invalid.sort_values("merge_time").reset_index(drop=True)

    merged = pd.concat(chunks + [left_invalid], ignore_index=True)
    return merged.sort_values("merge_time").reset_index(drop=True)


def _prepare_alert_route(df_alerts: pd.DataFrame) -> pd.DataFrame:
    """Añade route_id desde lines si las alertas no traen route_id explícito."""
    df = df_alerts.copy()
    if "lines" in df.columns:
        df["lines"] = df["lines"].apply(_split_lines)
        df = df.explode("lines", ignore_index=True).dropna(subset=["lines"])
        df["route_id"] = normalize_route_id(df["lines"])
    else:
        df["route_id"] = pd.NA
    return df[["route_id", "timestamp_start", "category", "num_updates"]]


# ──────────────────────────────────────────────────────────────
#  6. Política de columnas final (igual que generate_final_dataset)
# ──────────────────────────────────────────────────────────────

def apply_final_column_policy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona únicamente las columnas del esquema de entrenamiento.
    Idéntico a generate_final_dataset.apply_final_column_policy.

    Distingue dos grupos:
      - FEATURES: variables de entrada al modelo. Se calculan en RT a
        partir de datos del presente/pasado.
      - TARGETS: variables que los modelos predicen (retrasos futuros y
        aparición de alertas en los próximos minutos). Nunca se calculan
        en RT — se incluyen como NaN solo para mantener la compatibilidad
        de esquema con el dataset histórico de entrenamiento.
    """
    # ── Features (se calculan en RT) ────────────────────────────
    gtfs_keep = [
        "date", "match_key", "stop_id", "route_id", "direction",
        "delay_seconds",
        "lagged_delay_1", "lagged_delay_2",
        "route_rolling_delay", "actual_headway_seconds",
        "is_unscheduled",
        "hour_sin", "hour_cos", "dow", "is_weekend",
        "merge_time", "stops_to_end", "scheduled_time_to_end",
    ]
    weather_keep = ["temp_extreme"]
    events_keep = [
        "n_eventos_afectando", "tipo_referente",
        "afecta_previo", "afecta_durante", "afecta_despues",
    ]
    alerts_keep = [
        "category", "num_updates", "timestamp_start",
        "seconds_since_last_alert", "is_alert_just_published",
    ]

    # ── Targets (NO se calculan en RT, se rellenan con NaN) ─────
    targets_keep = [
        # Targets del modelo de retrasos
        "target_delay_10m", "target_delay_20m", "target_delay_30m",
        "target_delay_45m", "target_delay_60m", "target_delay_end",
        "delta_delay_10m", "delta_delay_20m", "delta_delay_30m",
        "delta_delay_45m", "delta_delay_60m", "delta_delay_end",
        "station_delay_10m", "station_delay_20m", "station_delay_30m",
        # Targets del modelo de predicción de alertas
        "seconds_to_next_alert",
        "alert_in_next_15m",
        "alert_in_next_30m",
    ]

    features = gtfs_keep + weather_keep + events_keep + alerts_keep
    wanted = features + targets_keep

    # Seleccionar solo features existentes en el DataFrame
    existing_features = [c for c in features if c in df.columns]
    out = df.loc[:, existing_features].copy()

    # Rellenar features ausentes con NaN (por si alguna fuente falló)
    for col in features:
        if col not in out.columns:
            out[col] = np.nan

    # Añadir columnas de target siempre vacías (NaN) para compatibilidad
    # de esquema con el dataset histórico de entrenamiento.
    for col in targets_keep:
        out[col] = np.nan

    return out[wanted]


# ──────────────────────────────────────────────────────────────
#  7. Orquestador principal
# ──────────────────────────────────────────────────────────────

def build_realtime_dataset() -> pd.DataFrame:
    """
    Orquesta la generación del dataset de inferencia en tiempo real.

    Devuelve
    --------
    pd.DataFrame con el mismo esquema que el dataset histórico de entrenamiento.
    """
    print("\n" + "=" * 60)
    print("  BUILD REALTIME DATASET")
    print("=" * 60)

    # ── 1. GTFS ──────────────────────────────────────────────
    print("\n[1/4] GTFS en tiempo real")
    gtfs = load_realtime_gtfs()

    # ── 2. Clima ─────────────────────────────────────────────
    print("\n[2/4] Clima en tiempo real")
    try:
        weather = load_realtime_weather()
    except Exception as e:
        print(f"  [WARN] Clima no disponible: {e}. Continuando sin clima.")
        weather = pd.DataFrame()

    # ── 3. Cruces GTFS + Clima ────────────────────────────────
    print("\n[3/4] Cruzando fuentes...")
    merged = merge_gtfs_weather_rt(gtfs, weather)
    del gtfs, weather
    gc.collect()

    # ── 4. Eventos ───────────────────────────────────────────
    try:
        events = load_realtime_events()
    except Exception as e:
        print(f"  [WARN] Eventos no disponibles: {e}.")
        events = pd.DataFrame()

    merged = merge_gtfs_events_rt(merged, events)
    del events
    gc.collect()

    # ── 5. Alertas ───────────────────────────────────────────
    try:
        alerts_raw = load_realtime_alerts()
        alerts = _prepare_alert_route(alerts_raw) if not alerts_raw.empty else pd.DataFrame()
    except Exception as e:
        print(f"  [WARN] Alertas no disponibles: {e}.")
        alerts = pd.DataFrame()

    merged = merge_gtfs_alerts_rt(merged, alerts)
    del alerts
    gc.collect()

    # ── 6. Post-proceso ───────────────────────────────────────
    merged = reduce_mem_usage(merged)
    final_df = apply_final_column_policy(merged)
    del merged
    gc.collect()

    print(
        f"\n[OK] Dataset realtime generado: "
        f"{len(final_df)} filas, {len(final_df.columns)} columnas"
    )

    return final_df



def main() -> int:
    df = build_realtime_dataset()
    print(df[["match_key","stop_id", "delay_seconds","stops_to_end", "lagged_delay_1", "lagged_delay_2",]].head(20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())