"""
Genera el dataset de inferencia en tiempo real e indexa por trip_id (match_key).

Los datos diarios estáticos (clima, eventos, stop_times) se leen desde Google Drive
(carpeta MTA_Daily_Data/), actualizada por upload_daily_data.py una vez al día.
Las únicas llamadas en tiempo real son:
  - GTFS-RT  : endpoint de la línea concreta (extraída del trip_id)
  - Alertas  : Gmail MTA

Uso desde otro módulo:
    from src.ETL.pipelines.realtime.preprocess_realtime_lgbm import get_single_trip_features

    features = get_single_trip_features("033150_2..N08R")

Uso standalone (un único trip):
    uv run python src/ETL/pipelines/realtime/preprocess_realtime_lgbm.py <trip_id>
"""

import gc
import os
import logging

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.common.minio_client import download_json, upload_json

from src.ETL.pipelines.realtime.generate_realtime_dataset import (
    load_realtime_gtfs,
    load_realtime_alerts,
    _prepare_alert_route,
    merge_gtfs_weather_rt,
    merge_gtfs_events_rt,
    merge_gtfs_alerts_rt,
    apply_final_column_policy,
    reduce_mem_usage,
    normalize_route_id,
)
from src.ETL.tiempo_real_metro.realtime_data import (
    FUENTES,
    extraccion_linea,
    conversion_hora_NYC,
    dia_segun_fecha_y_formato,
    direccion_tren,
    union_dataframes,
)
from app.data.drive import download_daily_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Columnas que no son features (igual que en eval_lgbm.py).
# stop_id se conserva: necesario para el target encoding del modelo.
DROP_COLS = {
    "date", "merge_time", "timestamp_start", "is_unscheduled", "service_date",
    "target_delay_10m", "target_delay_20m", "target_delay_30m",
    "target_delay_45m", "target_delay_60m", "target_delay_end",
    "delta_delay_10m",  "delta_delay_20m",  "delta_delay_30m",
    "delta_delay_45m",  "delta_delay_60m",  "delta_delay_end",
    "station_delay_10m", "station_delay_20m", "station_delay_30m",
    "alert_in_next_15m", "alert_in_next_30m", "seconds_to_next_alert",
    "delay_minutes", "scheduled_time", "actual_time",
}

# ── Histórico de retrasos en MinIO ──────────────────────────────────────────

CACHE_FILE = "grupo5/realtime/delays_state_cache.json"

def _get_lagged_state() -> dict:
    """Descarga el estado de lags desde MinIO (bucket pd1, endpoint minio.fdi.ucm.es)."""
    try:
        return download_json(
            access_key=os.environ["MINIO_ACCESS_KEY"],
            secret_key=os.environ["MINIO_SECRET_KEY"],
            object_name=CACHE_FILE,
        )
    except Exception as e:
        log.warning("Sin caché histórico previo en MinIO (%s)", e)
        return {}

def _save_lagged_state(new_state: dict) -> None:
    """Sube el estado de lags a MinIO (bucket pd1, endpoint minio.fdi.ucm.es)."""
    try:
        upload_json(
            access_key=os.environ["MINIO_ACCESS_KEY"],
            secret_key=os.environ["MINIO_SECRET_KEY"],
            object_name=CACHE_FILE,
            data=new_state,
        )
    except Exception as e:
        log.error("Fallo subiendo caché histórico a MinIO: %s", e)

def _apply_and_update_lags(df: pd.DataFrame, update_cache: bool = False) -> pd.DataFrame:
    """
    Aplica lagged_delay_1/2 desde el estado MinIO con lógica por parada:
    el shift solo ocurre cuando el tren ha avanzado a una nueva parada (stop_id distinto).
    Si la parada no ha cambiado, los lags se mantienen del ciclo anterior.

    Espera 1 fila por match_key (ya colapsado a la parada más inminente).
    """
    if df.empty or "match_key" not in df.columns:
        return df

    prev_state = _get_lagged_state()
    df = df.copy()

    lag1_vals, lag2_vals = [], []
    new_state: dict = {}

    for _, row in df.iterrows():
        mk            = row["match_key"]
        current_stop  = str(row.get("stop_id", ""))
        current_delay = float(row.get("delay_seconds") or 0)

        prev       = prev_state.get(mk, {})
        prev_stop  = prev.get("stop_id")
        prev_delay = prev.get("delay", current_delay)
        prev_lag1  = prev.get("lag1")
        prev_lag2  = prev.get("lag2")

        if prev_stop is None:
            # Primera vez que vemos este trip: sin historial aún
            lag1, lag2 = current_delay, current_delay
        elif current_stop != prev_stop:
            # Tren avanzó a la siguiente parada → shift
            lag1 = prev_delay
            lag2 = prev_lag1 if prev_lag1 is not None else prev_delay
        else:
            # Misma parada que el ciclo anterior → mantener lags
            lag1 = prev_lag1 if prev_lag1 is not None else current_delay
            lag2 = prev_lag2 if prev_lag2 is not None else lag1

        lag1_vals.append(lag1)
        lag2_vals.append(lag2)

        if update_cache:
            new_state[mk] = {
                "stop_id": current_stop,
                "delay":   current_delay,
                "lag1":    lag1,
                "lag2":    lag2,
            }

    df["lagged_delay_1"] = pd.array(lag1_vals, dtype=float)
    df["lagged_delay_2"] = pd.array(lag2_vals, dtype=float)

    if update_cache:
        _save_lagged_state(new_state)

    return df


# ── Features de línea ────────────────────────────────────────────────────────

def _add_line_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula route_rolling_delay y actual_headway_seconds sobre el snapshot
    colapsado (1 fila por trip), usando scheduled_time_to_end como eje de
    posición en la ruta (trenes más adelantados = menos tiempo restante):

    - route_rolling_delay   : media móvil del delay de los trenes que van
                              DELANTE (shift(1)), igual que el histórico.
    - actual_headway_seconds: diff de scheduled_time_to_end entre trenes
                              consecutivos → tiempo de separación en ruta.
    """
    if df.empty:
        return df

    needed = {"delay_seconds", "route_id", "direction", "scheduled_time_to_end"}
    if not needed.issubset(df.columns):
        return df

    df = df.copy()
    df_s = (
        df[["route_id", "direction", "scheduled_time_to_end", "delay_seconds"]]
        .sort_values(["route_id", "direction", "scheduled_time_to_end"])
        .reset_index().rename(columns={"index": "_idx"})
    )
    grp = df_s.groupby(["route_id", "direction"])
    df_s["route_rolling_delay"]    = grp["delay_seconds"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df_s["actual_headway_seconds"] = grp["scheduled_time_to_end"].transform("diff")

    idx = df_s.set_index("_idx")
    df["route_rolling_delay"]    = idx["route_rolling_delay"].reindex(df.index)
    df["actual_headway_seconds"] = idx["actual_headway_seconds"].reindex(df.index)
    return df


# ── Features derivadas ───────────────────────────────────────────────────────

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mismas features derivadas que add_derived_features() en los scripts de evaluación."""
    if "lagged_delay_1" in df.columns and "delay_seconds" in df.columns:
        df["delay_velocity"] = df["delay_seconds"] - df["lagged_delay_1"]
    if "lagged_delay_1" in df.columns and "lagged_delay_2" in df.columns:
        df["delay_acceleration"] = (
            (df["delay_seconds"] - df["lagged_delay_1"])
            - (df["lagged_delay_1"] - df["lagged_delay_2"])
        )
    if "delay_seconds" in df.columns and "stops_to_end" in df.columns:
        df["delay_x_stops_remaining"] = df["delay_seconds"] * df["stops_to_end"]
    if "delay_seconds" in df.columns and "scheduled_time_to_end" in df.columns:
        df["delay_ratio"] = df["delay_seconds"] / (df["scheduled_time_to_end"] + 1)
    return df


# ── Fuentes diarias desde Drive (caché en RAM) ───────────────────────────────

_CACHE_STOP_TIMES: pd.DataFrame | None = None
_CACHE_WEATHER:    pd.DataFrame | None = None
_CACHE_EVENTS:     pd.DataFrame | None = None


def _load_stop_times_drive() -> pd.DataFrame:
    global _CACHE_STOP_TIMES
    if _CACHE_STOP_TIMES is None:
        log.info("  [STOP TIMES] Leyendo desde Drive...")
        _CACHE_STOP_TIMES = download_daily_file("stop_times.parquet", subfolder="gtfs_supplemented")
    return _CACHE_STOP_TIMES


def _load_weather_drive() -> pd.DataFrame:
    global _CACHE_WEATHER
    if _CACHE_WEATHER is None:
        log.info("  [CLIMA] Leyendo desde Drive...")
        _CACHE_WEATHER = download_daily_file("clima_hoy.parquet", subfolder="clima")
    return _CACHE_WEATHER


def _load_events_drive() -> pd.DataFrame:
    global _CACHE_EVENTS
    if _CACHE_EVENTS is None:
        log.info("  [EVENTOS] Leyendo desde Drive...")
        try:
            _CACHE_EVENTS = download_daily_file("eventos_hoy.parquet", subfolder="eventos")
            log.info("  [EVENTOS] %d filas.", len(_CACHE_EVENTS))
        except Exception as e:
            log.warning("  [EVENTOS] No disponible en Drive: %s", e)
            _CACHE_EVENTS = pd.DataFrame()
    return _CACHE_EVENTS


# ── GTFS-RT para una línea concreta ─────────────────────────────────────────

def _route_id_from_trip(trip_id: str) -> str:
    """Extrae route_id del trip_id: '033150_2..N08R' → '2'."""
    return trip_id.split("_")[1].split(".")[0]


def _load_gtfs_rt_line(route_id: str) -> pd.DataFrame:
    """Llama solo al endpoint GTFS-RT de la línea dada."""
    url = None
    for info in FUENTES.values():
        if route_id.upper() in [l.upper() for l in info["lineas"]]:
            url = info["url"]
            break
    if url is None:
        raise ValueError(f"route_id '{route_id}' no encontrado en FUENTES")

    log.info("  [GTFS RT] Llamando endpoint para línea %s...", route_id)
    datos = extraccion_linea(url, route_id)
    df = pd.DataFrame(datos)
    if df.empty:
        raise ValueError(f"Sin datos RT para la línea {route_id}")

    df = conversion_hora_NYC(df)
    df = dia_segun_fecha_y_formato(df)
    df = direccion_tren(df)
    df = df.dropna(subset=["hora_llegada", "viaje_id", "parada_id", "linea_id"])
    df["segundos_reales"] = (
        df["hora_llegada"].dt.hour * 3600
        + df["hora_llegada"].dt.minute * 60
        + df["hora_llegada"].dt.second
    )
    log.info("  [GTFS RT] %d filas para línea %s.", len(df), route_id)
    return df


def _gtfs_rt_to_features(df_real: pd.DataFrame, df_previsto: pd.DataFrame) -> pd.DataFrame:
    """
    A partir del feed RT de una línea y los horarios previstos (ya cargados),
    calcula el delay y adapta las columnas al esquema del pipeline.
    Equivalente a load_realtime_gtfs() pero para una sola línea.
    """
    log.info("  [GTFS] Calculando retrasos...")
    df = union_dataframes(df_real, df_previsto)
    if df.empty:
        raise ValueError("DataFrame GTFS vacío tras unión.")

    rename_map = {}
    if "linea_id"  in df.columns: rename_map["linea_id"]  = "route_id"
    if "parada_id" in df.columns: rename_map["parada_id"] = "stop_id"
    if "delay"     in df.columns: rename_map["delay"]     = "delay_seconds"
    if "direccion" in df.columns: rename_map["direccion"] = "direction"
    df = df.rename(columns=rename_map)

    if "hora_llegada" in df.columns:
        df["merge_time"] = pd.to_datetime(df["hora_llegada"], errors="coerce")
    else:
        df["merge_time"] = pd.Timestamp.now(tz="America/New_York")

    df["hour"]           = df["merge_time"].dt.hour
    df["date"]           = df["merge_time"].dt.date
    df["service_date"]   = df["merge_time"].dt.strftime("%Y-%m-%d")
    df["actual_seconds"] = (
        df["merge_time"].dt.hour * 3600
        + df["merge_time"].dt.minute * 60
        + df["merge_time"].dt.second
    )

    if "viaje_id" in df.columns:
        df["match_key"] = df["viaje_id"].astype(str)
    if "route_id" in df.columns:
        df["route_id"] = normalize_route_id(df["route_id"])
    if "stop_id" in df.columns:
        df["stop_id"] = df["stop_id"].astype("string")

    return df


# ── Merges comunes ────────────────────────────────────────────────────────────

def _merge_all(df_gtfs: pd.DataFrame) -> pd.DataFrame:
    """Aplica merges de clima, eventos y alertas sobre el DataFrame GTFS."""
    try:
        weather = _load_weather_drive()
    except Exception as e:
        log.warning("Clima no disponible: %s", e)
        weather = pd.DataFrame()

    merged = merge_gtfs_weather_rt(df_gtfs, weather)
    del df_gtfs, weather
    gc.collect()

    events = _load_events_drive()
    merged = merge_gtfs_events_rt(merged, events)
    del events
    gc.collect()

    try:
        alerts_raw = load_realtime_alerts()
        alerts = _prepare_alert_route(alerts_raw) if not alerts_raw.empty else pd.DataFrame()
    except Exception as e:
        log.warning("Alertas no disponibles: %s", e)
        alerts = pd.DataFrame()

    merged = merge_gtfs_alerts_rt(merged, alerts)
    del alerts
    gc.collect()

    merged = reduce_mem_usage(merged)
    return apply_final_column_policy(merged)


# ── API pública ───────────────────────────────────────────────────────────────

def update_lag_state() -> None:
    """
    Función ligera para el worker: solo actualiza el estado de lags en MinIO.
    Descarga el feed GTFS-RT completo, colapsa a la parada más inminente por trip
    y actualiza {stop_id, delay, lag1, lag2} en MinIO.
    No descarga clima, eventos ni alertas.
    """
    log.info("=== UPDATE LAG STATE ===")

    df_previsto = _load_stop_times_drive()
    df_gtfs     = load_realtime_gtfs(df_previsto=df_previsto)

    if "stops_to_end" not in df_gtfs.columns or df_gtfs.empty:
        log.warning("Feed GTFS vacío o sin stops_to_end. Abortando update.")
        return

    df = (
        df_gtfs[df_gtfs["stops_to_end"] > 0]
        .sort_values("stops_to_end", ascending=True)
        .drop_duplicates(subset=["match_key"], keep="first")
        .copy()
    )

    _apply_and_update_lags(df, update_cache=True)
    log.info("Estado de lags actualizado: %d trips.", len(df))


def get_single_trip_features(trip_id: str) -> dict | None:
    """
    Genera las features completas para un único trip_id en el momento de la petición.

    Flujo:
      1. GTFS-RT de la línea → parada más inminente del trip (posición actual)
      2. Lags desde MinIO   → lag1/lag2 ya calculados por el worker
      3. Clima / Eventos    → Drive (con caché RAM)
      4. Alertas            → Gmail
      5. Features derivadas → delay_velocity, delay_acceleration, etc.
    """
    log.info("=== SINGLE TRIP: %s ===", trip_id)

    route_id    = _route_id_from_trip(trip_id)
    df_previsto = _load_stop_times_drive()

    df_real = _load_gtfs_rt_line(route_id)
    if df_real[df_real["viaje_id"] == trip_id].empty:
        log.warning("trip_id '%s' no encontrado en el feed RT.", trip_id)
        return None

    df_gtfs = _gtfs_rt_to_features(df_real, df_previsto)

    # Colapsar a parada más inminente por trip para que _add_line_features
    # opere con un tren por fila
    if "stops_to_end" in df_gtfs.columns:
        df_collapsed = (
            df_gtfs[df_gtfs["stops_to_end"] > 0]
            .sort_values("stops_to_end", ascending=True)
            .drop_duplicates(subset=["match_key"], keep="first")
            .copy()
        )
    else:
        df_collapsed = df_gtfs.copy()

    # Rolling delay y headway calculados sobre todos los trips de la línea
    df_collapsed = _add_line_features(df_collapsed)

    # Filtrar al trip pedido
    df = df_collapsed[df_collapsed["match_key"] == trip_id].copy()
    if df.empty:
        return None

    # Lags desde MinIO (sin actualizar el estado, solo lectura)
    df = _apply_and_update_lags(df, update_cache=False)

    # Enriquecimiento completo: clima, eventos, alertas
    df = _merge_all(df)

    df = _add_derived_features(df)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    return df.iloc[0].to_dict() if not df.empty else None


def get_trip_features(match_key: str) -> dict | None:
    """Genera las features de un trip listas para predecir (encoding de categóricas en el caller)."""
    return get_single_trip_features(match_key)

