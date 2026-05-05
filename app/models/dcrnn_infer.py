"""
Inferencia del modelo DCRNN (Diffusion Convolutional Recurrent Neural Network) de propagación de retrasos.

Dado un conjunto de ventanas de datos en tiempo real, construye el tensor de
entrada para el DCRNN, ejecuta la inferencia en modo sin gradiente y aplica una
calibración en tiempo de inferencia que combina la salida del modelo con una
estimación de persistencia del retraso observado.

Calibración en tiempo de inferencia:
El modelo DCRNN tiende a predecir valores cercanos a 0–3 segundos para casi todos
los nodos, porque en inferencia las ventanas en tiempo real cubren solo ~13% de los
nodos del grafo y la GNN suaviza el nodo activo contra ~1900 vecinos en reposo.
Para compensar esto, se combina la salida del modelo con un baseline de persistencia
del retraso observado, decaído exponencialmente según el horizonte temporal:
  - Factores de decaimiento (exp(-h/τ), τ≈22 min): 0.64 / 0.40 / 0.25 a 10/20/30 min.
  - Factor de antigüedad: cada ventana de 15 min hacia atrás reduce la contribución × 0.75.

Dependencias:
- app.data.transforms.windows_to_dcrnn_tensor: construye el tensor de entrada.
- app.models.registry.DCRNNEntry: contenedor del modelo, escaladores y grafo cargados.
- app.schemas.PropagationPrediction / PropagationResponse: esquemas de respuesta.
- torch: para la inferencia en CPU sin gradiente.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import torch

from app.data.transforms import windows_to_dcrnn_tensor
from app.models.registry import DCRNNEntry
from app.schemas import PropagationPrediction, PropagationResponse

logger = logging.getLogger(__name__)


def run_propagation(
    entry: DCRNNEntry,
    windows: list,
    stations_meta: Optional[dict] = None,
    stop_id_filter: Optional[str] = None,
    route_id_filter: Optional[str] = None,
) -> PropagationResponse:
    """
    Ejecuta el modelo DCRNN y devuelve predicciones de retraso propagado a 10, 20 y 30 minutos.

    Pasos principales:
    1. Construye el tensor de entrada (1, history_len, N, F) con windows_to_dcrnn_tensor.
    2. Ejecuta la red DCRNN sin gradiente y desnormaliza la salida con scaler_Y.
    3. Aplica la calibración de persistencia: suma al output del modelo el retraso
       observado en las ventanas, ponderado por antigüedad y decaimiento temporal.
    4. Filtra los resultados por stop_id y/o route_id si se especifican.

    Parámetros:
        entry: Contenedor DCRNNEntry con el modelo, escaladores, nodos y grafo.
        windows: Lista de DataFrames de ventanas de datos en tiempo real.
        stations_meta: Dict opcional de stop_id → {lat, lon, ...} para añadir
                       coordenadas a las predicciones.
        stop_id_filter: Si se especifica, solo se devuelve la predicción para esa parada.
                        Se busca por stop_id exacto o por base (sin sufijo N/S).
        route_id_filter: Si se especifica, filtra los nodos por línea.

    Retorna:
        PropagationResponse con las predicciones de retraso por parada,
        el timestamp de predicción y el número de estaciones.
    """
    X = windows_to_dcrnn_tensor(
        windows=windows,
        nodes=entry.nodes,
        feature_set=entry.feature_set,
        scaler_X=entry.scaler_X,
        history_len=entry.history_len,
    )  # (1, history_len, N, n_features)

    with torch.no_grad():
        y_hat = entry.model(X, entry.edge_index, entry.edge_weight)
    # y_hat tiene forma (1, 1, N, 3) → se aplana a (N, 3)
    y_scaled = y_hat.squeeze(0).squeeze(0).cpu().numpy()

    import numpy as np
    N, H = y_scaled.shape
    y_sec = entry.scaler_Y.inverse_transform(y_scaled.reshape(-1, H)).reshape(N, H)

    nodes_sorted = sorted(entry.nodes)
    predictions: list[PropagationPrediction] = []

    # Calibración en tiempo de inferencia:
    # Factores de decaimiento: exp(-h/τ) con τ≈22 min → 0.64/0.40/0.25 a 10/20/30 min.
    # Factor de antigüedad: cada ventana de 15 min atrás reduce la contribución × 0.75.
    DECAY = np.array([0.64, 0.40, 0.25], dtype=np.float32)
    AGE_DECAY = 0.75

    if windows:
        import pandas as pd
        need_cols = {"delay_seconds_mean", "stop_id", "route_id"}
        valid_windows = [w for w in windows if need_cols.issubset(w.columns)]
        if valid_windows:
            # Etiquetar cada ventana con su antigüedad (0 = más reciente) y concatenar.
            tagged = []
            for age, win in enumerate(reversed(valid_windows)):
                tmp = win[["route_id", "stop_id", "delay_seconds_mean"]].copy()
                tmp["age"] = age
                tagged.append(tmp)
            all_wins = pd.concat(tagged, ignore_index=True)
            all_wins["node_key"] = all_wins["route_id"].astype(str) + "_" + all_wins["stop_id"].astype(str)

            # Para cada nodo, conservar solo la observación más reciente (menor age).
            all_wins = all_wins.sort_values("age")
            best = (
                all_wins.dropna(subset=["delay_seconds_mean"])
                .groupby("node_key", as_index=False)
                .first()[["node_key", "delay_seconds_mean", "age"]]
            )

            node_idx = {sid: i for i, sid in enumerate(nodes_sorted)}
            for row in best.itertuples(index=False):
                idx = node_idx.get(row.node_key)
                if idx is not None and np.isfinite(row.delay_seconds_mean):
                    age_factor = AGE_DECAY ** row.age
                    y_sec[idx] = row.delay_seconds_mean * age_factor * DECAY + y_sec[idx]


    def _split(sid: str) -> tuple[str, str]:
        """Divide un node_key 'route_stop_id' en (route, stop_id)."""
        parts = sid.split("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "", sid

    def _base(sid: str) -> str:
        """Elimina el sufijo de dirección N/S de un stop_id."""
        _, stop = _split(sid)
        return stop[:-1] if stop and stop[-1] in ("N", "S") else stop

    def _route_matches(node_route: str, filt: str) -> bool:
        """Comprueba si el route_id de un nodo coincide con el filtro indicado."""
        if not filt:
            return True
        norm = filt.strip().split("-")[0].split("_")[0]
        return node_route == norm

    if stop_id_filter:
        matched = []
        for i, sid in enumerate(nodes_sorted):
            r, stop = _split(sid)
            base = stop[:-1] if stop and stop[-1] in ("N", "S") else stop
            if (stop == stop_id_filter or base == stop_id_filter) and _route_matches(r, route_id_filter or ""):
                matched.append(i)
        if not matched:
            logger.warning(
                "DCRNN: stop_id '%s' (route=%s) not found in %d nodes. Sample: %s",
                stop_id_filter, route_id_filter, len(nodes_sorted), nodes_sorted[:5],
            )
        if matched:
            # Tomar el máximo entre los nodos coincidentes (tipicamente N y S de la misma parada).
            # Se usa el máximo en lugar de la media para no diluir la señal de la dirección activa
            # cuando el feed solo tiene datos para una de las dos direcciones.
            agg = y_sec[matched].max(axis=0)
            base_meta = (stations_meta or {}).get(stop_id_filter, {})
            predictions.append(PropagationPrediction(
                stop_id=stop_id_filter,
                lat=base_meta.get("lat"),
                lon=base_meta.get("lon"),
                delay_10m=float(np.clip(agg[0], 0, None)),
                delay_20m=float(np.clip(agg[1], 0, None)),
                delay_30m=float(np.clip(agg[2], 0, None)),
            ))
    else:
        for i, stop_id in enumerate(nodes_sorted):
            r, stop = _split(stop_id)
            base = stop[:-1] if stop and stop[-1] in ("N", "S") else stop
            if not _route_matches(r, route_id_filter or ""):
                continue
            meta = (stations_meta or {}).get(base, {})
            predictions.append(PropagationPrediction(
                stop_id=stop_id,
                lat=meta.get("lat"),
                lon=meta.get("lon"),
                delay_10m=float(np.clip(y_sec[i, 0], 0, None)),
                delay_20m=float(np.clip(y_sec[i, 1], 0, None)),
                delay_30m=float(np.clip(y_sec[i, 2], 0, None)),
            ))

    return PropagationResponse(
        predicted_at=datetime.now(timezone.utc).isoformat(),
        n_stations=len(predictions),
        predictions=predictions,
    )
