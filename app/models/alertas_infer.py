"""
Inferencia del modelo XGBoost de alertas de incidencia por línea.

Dado un conjunto de ventanas de datos en tiempo real, agrega la información
por línea y dirección, codifica las variables categóricas, ejecuta el modelo
XGBoost para obtener probabilidades de alerta y construye la respuesta con
las predicciones filtradas por umbral y probabilidad mínima.

Dependencias:
- app.data.transforms.windows_to_alertas_features: agrega las ventanas por línea.
- app.models.registry.AlertEntry: contenedor del modelo y umbral cargados.
- app.schemas.AlertPrediction / AlertResponse: esquemas Pydantic de respuesta.

Notas:
- Los route_id y direction se codifican numéricamente antes de la inferencia
  usando los diccionarios _ROUTE_CODES y _DIR_CODES definidos en este módulo.
- Si el modelo tiene feature_names_in_ (sklearn) o get_booster().feature_names
  (XGBoost), se usa esa lista para garantizar el orden correcto de columnas.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from app.data.transforms import windows_to_alertas_features
from app.models.registry import AlertEntry
from app.schemas import AlertPrediction, AlertResponse

logger = logging.getLogger(__name__)


# Orden canónico de líneas para la codificación numérica de route_id
_ROUTE_ORDER = sorted([
    "1", "2", "3", "4", "5", "6", "7",
    "A", "B", "C", "D", "E", "F", "G",
    "J", "L", "M", "N", "Q", "R",
    "S", "SIR", "Sf", "Sr",
    "W", "Z",
])
_ROUTE_CODES: dict[str, int] = {r: i for i, r in enumerate(_ROUTE_ORDER)}

_DIR_ORDER = ["N", "S"]
_DIR_CODES: dict[str, int] = {d: i for i, d in enumerate(_DIR_ORDER)}


def run_alerts(
    entry: AlertEntry,
    windows: list,
    threshold: Optional[float] = None,
    route_id_filter: Optional[str] = None,
    min_prob: float = 0.0,
) -> AlertResponse:
    """
    Ejecuta el modelo XGBoost de alertas sobre las ventanas de datos y devuelve las predicciones.

    Pasos:
    1. Agrega las ventanas por línea y dirección con windows_to_alertas_features.
    2. Filtra por route_id si se especifica.
    3. Codifica numéricamente route_id y direction.
    4. Alinea las columnas con las features del modelo entrenado.
    5. Ejecuta predict_proba y construye la respuesta filtrando por min_prob.

    Parámetros:
        entry: Contenedor AlertEntry con el modelo XGBoost y el umbral de clasificación.
        windows: Lista de DataFrames de ventanas de datos en tiempo real.
        threshold: Umbral de probabilidad para clasificar como alerta. Si es None,
                   se usa el umbral guardado en entry.threshold.
        route_id_filter: Si se especifica, solo se predicen alertas para esa línea.
        min_prob: Probabilidad mínima para incluir una predicción en la respuesta.

    Retorna:
        AlertResponse con la lista de predicciones, el timestamp y el umbral usado.
        Si no hay datos de línea disponibles, devuelve una respuesta con lista vacía.
    """
    thr = threshold if threshold is not None else entry.threshold

    df_linea = windows_to_alertas_features(windows)

    if df_linea.empty:
        return AlertResponse(
            predicted_at=datetime.now(timezone.utc).isoformat(),
            threshold_used=thr,
            n_lines=0,
            predictions=[],
        )

    if route_id_filter:
        df_linea = df_linea[df_linea["route_id"].astype(str) == route_id_filter]

    model = entry.model
    feat_attr = getattr(model, "feature_names_in_", None)
    known_features = list(feat_attr) if feat_attr is not None else []
    if not known_features:
        try:
            known_features = model.get_booster().feature_names or []
        except Exception:
            known_features = []

    df_feat = df_linea.copy()

    if "route_id" in df_feat.columns:
        df_feat["route_id"] = (
            df_feat["route_id"].astype(str).map(_ROUTE_CODES).fillna(-1).astype(int)
        )
    if "direction" in df_feat.columns:
        df_feat["direction"] = (
            df_feat["direction"].astype(str).map(_DIR_CODES).fillna(-1).astype(int)
        )

    if known_features:
        for col in known_features:
            if col not in df_feat.columns:
                df_feat[col] = 0
        X = df_feat[known_features]
    else:
        X = df_feat.select_dtypes(include=[np.number])

    X = X.fillna(0)
    probs = model.predict_proba(X)[:, 1]

    predictions: list[AlertPrediction] = []
    for i, prob in enumerate(probs):
        if prob < min_prob:
            continue
        route_id = str(df_linea["route_id"].iloc[i]) if "route_id" in df_linea.columns else "?"
        direction = str(df_linea["direction"].iloc[i]) if "direction" in df_linea.columns else "?"
        pct = float(df_linea["pct_paradas_retrasadas"].iloc[i]) if "pct_paradas_retrasadas" in df_linea.columns else None
        delay_mean = float(df_linea["delay_mean_linea"].iloc[i]) if "delay_mean_linea" in df_linea.columns else None

        predictions.append(AlertPrediction(
            route_id=route_id,
            direction=direction,
            alert_probability=float(prob),
            alert_predicted=bool(prob >= thr),
            pct_stops_delayed=pct,
            delay_mean_seconds=delay_mean,
        ))

    return AlertResponse(
        predicted_at=datetime.now(timezone.utc).isoformat(),
        threshold_used=thr,
        n_lines=len(predictions),
        predictions=predictions,
    )
