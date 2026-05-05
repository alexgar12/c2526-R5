"""
Inferencia de los modelos LightGBM de predicción de retraso absoluto.

Proporciona dos funciones de inferencia:
- run_delay_single: predicción para un único viaje (tren concreto) a partir de
  un diccionario de features extraídas por get_trip_features.
- run_delays: predicción masiva para todas las paradas de la última ventana de Drive,
  con filtros opcionales por ruta y parada.

Los modelos predicen el retraso esperado en segundos en dos horizontes:
- target_delay_30m: retraso en 30 minutos.
- target_delay_end: retraso al final del recorrido del tren.

Dependencias:
- app.data.transforms.windows_to_delay_features: prepara el DataFrame de features.
- app.models.registry.LGBMDelayEntry: contenedor del modelo y preprocesado cargados.
- app.schemas.DelayPrediction / DelayResponse: esquemas Pydantic de respuesta.
- lightgbm (implícito a través de joblib): el modelo se carga como objeto LightGBM Booster.

Notas:
- _apply_preprocessing codifica variables categóricas (label encoding, target encoding)
  y calcula features derivadas (delay_velocity, delay_acceleration, etc.) tal como
  se hizo en el entrenamiento.
- Las predicciones se recortan a 0 por abajo (no puede haber retraso negativo en la respuesta).
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from app.data.transforms import windows_to_delay_features
from app.models.registry import LGBMDelayEntry
from app.schemas import DelayPrediction, DelayResponse

logger = logging.getLogger(__name__)


def _apply_preprocessing(df: pd.DataFrame, prep: dict) -> pd.DataFrame:
    """
    Aplica el preprocesado guardado en el artefacto de W&B al DataFrame de inferencia.

    Realiza en orden:
    1. Label encoding de columnas categóricas según los mapeos del artefacto.
    2. Target encoding de stop_id usando el encoder entrenado y la media global.
    3. Eliminación de columnas de identificación (stop_id, match_key).
    4. Cálculo de features derivadas (velocidad, aceleración, ratio de retraso).

    Parámetros:
        df: DataFrame con las features crudas de inferencia.
        prep: Dict de preprocesado cargado desde preprocessing_*.json del artefacto.

    Retorna:
        DataFrame con las features transformadas, listo para llamar a model.predict().
    """
    df = df.copy()

    for col, mapping in prep.get("label_encoders", {}).items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

    stop_enc = prep.get("target_encoder_stop_id", {})
    global_mean = prep.get("target_encoder_global_mean", 0.0)
    if stop_enc and "stop_id" in df.columns:
        df["stop_id_target_enc"] = (
            df["stop_id"].astype(str).map(stop_enc).fillna(global_mean)
        )

    df = df.drop(columns=[c for c in ("stop_id", "match_key") if c in df.columns])

    for feat in prep.get("derived_features", []):
        if feat == "delay_velocity" and "delay_seconds_mean" in df.columns and "lagged_delay_1_mean" in df.columns:
            df["delay_velocity"] = df["delay_seconds_mean"] - df["lagged_delay_1_mean"]
        elif feat == "delay_acceleration" and "delay_seconds_mean" in df.columns:
            d1 = df.get("lagged_delay_1_mean", 0)
            d2 = df.get("lagged_delay_2_mean", 0)
            df["delay_acceleration"] = (df["delay_seconds_mean"] - d1) - (d1 - d2)
        elif feat == "delay_x_stops_remaining" and "delay_seconds_mean" in df.columns and "stops_to_end_mean" in df.columns:
            df["delay_x_stops_remaining"] = df["delay_seconds_mean"] * df["stops_to_end_mean"]
        elif feat == "delay_ratio" and "delay_seconds_mean" in df.columns and "scheduled_time_to_end_mean" in df.columns:
            df["delay_ratio"] = df["delay_seconds_mean"] / (df["scheduled_time_to_end_mean"] + 1)

    return df


def run_delay_single(entry: LGBMDelayEntry, features: dict) -> float:
    """
    Ejecuta el modelo LightGBM de retraso sobre las features de un único viaje.

    Aplica el label encoding, target encoding de stop_id y alinea las columnas
    con las features del modelo. Las columnas faltantes se rellenan con 0.

    Parámetros:
        entry: Contenedor LGBMDelayEntry con el modelo y el preprocesado.
        features: Dict de features extraídas por get_trip_features para un tren concreto.

    Retorna:
        Retraso predicho en segundos como float.
    """
    df = pd.DataFrame([features])

    for col, mapping in entry.preprocessing.get("label_encoders", {}).items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

    stop_enc = entry.preprocessing.get("target_encoder_stop_id", {})
    global_mean = entry.preprocessing.get("target_encoder_global_mean", 0.0)
    if stop_enc and "stop_id" in df.columns:
        df["stop_id_target_enc"] = df["stop_id"].astype(str).map(stop_enc).fillna(global_mean)

    df = df.drop(columns=[c for c in ("stop_id", "match_key") if c in df.columns])

    feature_names = entry.model.feature_name()
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return float(entry.model.predict(df[feature_names])[0])


def run_delays(
    entry: LGBMDelayEntry,
    windows: list,
    route_id_filter: Optional[str] = None,
    stop_id_filter: Optional[str] = None,
    min_delay_seconds: float = 0.0,
) -> DelayResponse:
    """
    Ejecuta el modelo LightGBM de retraso sobre todas las paradas de la última ventana.

    Prepara las features con windows_to_delay_features, aplica los filtros opcionales,
    llama a _apply_preprocessing, alinea las columnas y ejecuta model.predict().
    Las predicciones menores que min_delay_seconds se omiten de la respuesta.

    Parámetros:
        entry: Contenedor LGBMDelayEntry con el modelo y el preprocesado.
        windows: Lista de DataFrames de ventanas; solo se usa la última (windows[-1]).
        route_id_filter: Si se especifica, solo se predicen retrasos para esa línea.
        stop_id_filter: Si se especifica, filtra por stop_id exacto o base (sin sufijo N/S).
        min_delay_seconds: Umbral mínimo de retraso para incluir en la respuesta.

    Retorna:
        DelayResponse con las predicciones, el target y el timestamp. Si no hay datos
        tras el filtrado, devuelve una respuesta con lista vacía.
    """
    df = windows_to_delay_features(windows)

    if route_id_filter and "route_id" in df.columns:
        df = df[df["route_id"].astype(str) == route_id_filter]
    if stop_id_filter and "stop_id" in df.columns:
        base = df["stop_id"].astype(str).str.rstrip("NS")
        df = df[(df["stop_id"].astype(str) == stop_id_filter) | (base == stop_id_filter)]

    if df.empty:
        return DelayResponse(
            predicted_at=datetime.now(timezone.utc).isoformat(),
            target=entry.preprocessing.get("target", "unknown"),
            n_stops=0,
            predictions=[],
        )

    stop_ids = df["stop_id"].astype(str).tolist() if "stop_id" in df.columns else []
    route_ids = df["route_id"].astype(str).tolist() if "route_id" in df.columns else []
    directions = df["direction"].astype(str).tolist() if "direction" in df.columns else []

    df = _apply_preprocessing(df, entry.preprocessing)

    model = entry.model
    try:
        feature_names = model.feature_name()
    except Exception:
        feature_names = []

    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        X = df[feature_names]
    else:
        X = df.select_dtypes(include=[np.number])

    preds = model.predict(X)

    predictions: list[DelayPrediction] = []
    for i, pred in enumerate(preds):
        if pred < min_delay_seconds:
            continue
        predictions.append(DelayPrediction(
            stop_id=stop_ids[i] if i < len(stop_ids) else "?",
            route_id=route_ids[i] if i < len(route_ids) else "?",
            direction=directions[i] if i < len(directions) else "?",
            delay_seconds=float(np.clip(pred, 0, None)),
            delay_minutes=float(np.clip(pred, 0, None)) / 60.0,
        ))

    return DelayResponse(
        predicted_at=datetime.now(timezone.utc).isoformat(),
        target=entry.preprocessing.get("target", "unknown"),
        n_stops=len(predictions),
        predictions=predictions,
    )
