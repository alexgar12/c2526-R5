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

    X = windows_to_dcrnn_tensor(
        windows=windows,
        nodes=entry.nodes,
        feature_set=entry.feature_set,
        scaler_X=entry.scaler_X,
        history_len=entry.history_len,
    )  # (1, history_len, N, n_features)

    with torch.no_grad():
        y_hat = entry.model(X, entry.edge_index, entry.edge_weight)
    # y_hat: (1, 1, N, 3) → (N, 3)
    y_scaled = y_hat.squeeze(0).squeeze(0).cpu().numpy()

    import numpy as np
    N, H = y_scaled.shape
    y_sec = entry.scaler_Y.inverse_transform(y_scaled.reshape(-1, H)).reshape(N, H)

    nodes_sorted = sorted(entry.nodes)
    predictions: list[PropagationPrediction] = []


    def _base(sid: str) -> str:
        return sid[:-1] if sid and sid[-1] in ("N", "S") else sid

    if stop_id_filter:
        matched = [i for i, sid in enumerate(nodes_sorted)
                   if sid == stop_id_filter or _base(sid) == stop_id_filter]
        if not matched:
            logger.warning(
                "DCRNN: stop_id '%s' not found in %d nodes. "
                "Sample nodes: %s",
                stop_id_filter, len(nodes_sorted),
                nodes_sorted[:5],
            )
        if matched:
            avg = y_sec[matched].mean(axis=0)
            base_meta = (stations_meta or {}).get(stop_id_filter, {})
            predictions.append(PropagationPrediction(
                stop_id=stop_id_filter,
                lat=base_meta.get("lat"),
                lon=base_meta.get("lon"),
                delay_10m=float(np.clip(avg[0], 0, None)),
                delay_20m=float(np.clip(avg[1], 0, None)),
                delay_30m=float(np.clip(avg[2], 0, None)),
            ))
    else:
        for i, stop_id in enumerate(nodes_sorted):
            base = _base(stop_id)
            if route_id_filter and stations_meta:
                meta = stations_meta.get(base, {})
                if route_id_filter not in str(meta.get("routes", "")):
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
