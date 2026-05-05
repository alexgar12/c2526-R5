"""
Inferencia de los modelos LightGBM de tendencia de retraso (delta).

Los modelos delta predicen si el retraso de un tren mejorará (disminuirá) o
empeorará (aumentará) en un horizonte temporal concreto (10, 20 o 30 minutos).
La salida es una probabilidad de mejora y una clasificación binaria aplicando
el umbral óptimo almacenado en el preprocesado del artefacto.

Proporciona dos funciones de inferencia:
- run_delta_single: predicción para un único viaje a partir de un dict de features.
- run_delta: predicción masiva para todas las paradas de la última ventana de Drive.

Dependencias:
- app.data.transforms.windows_to_delay_features: prepara el DataFrame de features.
- app.models.registry.DeltaEntry: contenedor del modelo y preprocesado cargados.
- app.schemas.DeltaPrediction / DeltaResponse: esquemas Pydantic de respuesta.

Notas:
- _apply_preprocessing codifica las variables categóricas usando los vocabularios
  guardados en preprocessing_delta_*.json del artefacto (formato vocabs en lugar
  de label_encoders, diferente al modelo de retraso).
- El umbral de clasificación (best_threshold) se guarda en el preprocesado; puede
  sobreescribirse en cada llamada.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from app.data.transforms import windows_to_delay_features
from app.models.registry import DeltaEntry
from app.schemas import DeltaPrediction, DeltaResponse

logger = logging.getLogger(__name__)


def _apply_preprocessing(df: pd.DataFrame, prep: dict) -> pd.DataFrame:
    """
    Aplica la codificación de variables categóricas del preprocesado delta al DataFrame.

    Usa los vocabularios (vocabs) guardados en el artefacto para mapear strings a enteros.
    Las categorías desconocidas se mapean a -1.

    Parámetros:
        df: DataFrame con las features crudas de inferencia.
        prep: Dict de preprocesado con la clave 'vocabs' (dict de col → {valor: código}).

    Retorna:
        Copia del DataFrame con las columnas categóricas codificadas numéricamente.
    """
    df = df.copy()
    vocabs: dict = prep.get("vocabs", {})
    for col, vocab in vocabs.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(vocab).fillna(-1).astype(int)
    return df


def run_delta_single(
    entry: DeltaEntry,
    features: dict,
    threshold: Optional[float] = None,
) -> tuple[float, bool]:
    """
    Ejecuta el modelo delta sobre las features de un único viaje.

    Codifica las variables categóricas con los vocabularios del preprocesado,
    elimina las columnas de identificación, alinea las features y ejecuta
    model.predict() para obtener la probabilidad de mejora.

    Parámetros:
        entry: Contenedor DeltaEntry con el modelo y el preprocesado.
        features: Dict de features del tren extraídas por get_trip_features.
        threshold: Umbral de clasificación. Si es None, usa prep['best_threshold'] (0.5 por defecto).

    Retorna:
        Tupla (prob, mejora_predicted) donde prob es la probabilidad de mejora [0,1]
        y mejora_predicted es True si prob >= threshold.
    """
    prep = entry.preprocessing
    thr = threshold if threshold is not None else float(prep.get("best_threshold", 0.5))

    df = pd.DataFrame([features])

    vocabs: dict = prep.get("vocabs", {})
    for col, vocab in vocabs.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(vocab).fillna(-1).astype(int)

    df = df.drop(columns=[c for c in ("stop_id", "match_key") if c in df.columns])

    feature_list = entry.model.feature_name() or prep.get("features", [])
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0
    prob = float(entry.model.predict(df[feature_list].fillna(0))[0])
    return prob, bool(prob >= thr)


def run_delta(
    entry: DeltaEntry,
    windows: list,
    horizon: str,
    threshold: Optional[float] = None,
    route_id_filter: Optional[str] = None,
    stop_id_filter: Optional[str] = None,
) -> DeltaResponse:
    """
    Ejecuta el modelo delta sobre todas las paradas de la última ventana de datos.

    Prepara las features con windows_to_delay_features, aplica los filtros opcionales,
    codifica variables categóricas y ejecuta model.predict() para obtener probabilidades
    de mejora. Incluye todas las predicciones en la respuesta (sin filtro de probabilidad mínima).

    Parámetros:
        entry: Contenedor DeltaEntry con el modelo y el preprocesado.
        windows: Lista de DataFrames de ventanas; solo se usa la última (windows[-1]).
        horizon: Nombre del horizonte temporal ('delta_delay_10m', '20m' o '30m').
        threshold: Umbral de clasificación. Si es None, usa prep['best_threshold'].
        route_id_filter: Si se especifica, solo se predicen tendencias para esa línea.
        stop_id_filter: Si se especifica, filtra por stop_id exacto o base (sin sufijo N/S).

    Retorna:
        DeltaResponse con las predicciones, el horizonte, el umbral y el timestamp.
        Si no hay datos tras el filtrado, devuelve una respuesta con lista vacía.
    """
    prep = entry.preprocessing
    thr = threshold if threshold is not None else float(prep.get("best_threshold", 0.5))

    df = windows_to_delay_features(windows)

    if route_id_filter and "route_id" in df.columns:
        df = df[df["route_id"].astype(str) == route_id_filter]
    if stop_id_filter and "stop_id" in df.columns:
        base = df["stop_id"].astype(str).str.rstrip("NS")
        df = df[(df["stop_id"].astype(str) == stop_id_filter) | (base == stop_id_filter)]

    if df.empty:
        return DeltaResponse(
            predicted_at=datetime.now(timezone.utc).isoformat(),
            horizon=horizon,
            threshold_used=thr,
            n_stops=0,
            predictions=[],
        )

    stop_ids = df["stop_id"].astype(str).tolist() if "stop_id" in df.columns else []
    route_ids = df["route_id"].astype(str).tolist() if "route_id" in df.columns else []
    directions = df["direction"].astype(str).tolist() if "direction" in df.columns else []

    df = _apply_preprocessing(df, prep)

    model = entry.model
    feature_list: list[str] = prep.get("features", [])
    try:
        feature_list = model.feature_name() or feature_list
    except Exception:
        pass

    if feature_list:
        for col in feature_list:
            if col not in df.columns:
                df[col] = 0
        X = df[feature_list]
    else:
        X = df.select_dtypes(include=[np.number])

    X = X.fillna(0)
    probs = model.predict(X)

    predictions: list[DeltaPrediction] = []
    for i, prob in enumerate(probs):
        predictions.append(DeltaPrediction(
            stop_id=stop_ids[i] if i < len(stop_ids) else "?",
            route_id=route_ids[i] if i < len(route_ids) else "?",
            direction=directions[i] if i < len(directions) else "?",
            mejora_prob=float(prob),
            mejora_predicted=bool(prob >= thr),
        ))

    return DeltaResponse(
        predicted_at=datetime.now(timezone.utc).isoformat(),
        horizon=horizon,
        threshold_used=thr,
        n_stops=len(predictions),
        predictions=predictions,
    )
