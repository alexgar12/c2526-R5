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

    # Inference-time calibration. The model is biased toward predicting near
    # mean_Y - 0.16·std_Y (≈ 0–3 s) for nearly every node because the realtime
    # windows are far sparser than training (≈13 % node coverage) and the GNN
    # smooths a single hot node against ~1900 calm neighbours. Blend the raw
    # model output with the observed current delay decayed exponentially over
    # the horizon (τ ≈ 22 min ⇒ factors at 10/20/30 min ≈ 0.64/0.40/0.25). The
    # model output then acts as a small additive correction to the persistence
    # baseline rather than the dominant signal.
    # Decay factors: exp(-h / τ) with τ ≈ 22 min → 0.64 / 0.40 / 0.25 at 10/20/30 min.
    # Age decay: each 15-min window back reduces the baseline by 0.75 so a delay
    # seen 30 min ago contributes less than one seen in the latest slot.
    DECAY = np.array([0.64, 0.40, 0.25], dtype=np.float32)
    AGE_DECAY = 0.75

    if windows:
        import pandas as pd
        need_cols = {"delay_seconds_mean", "stop_id", "route_id"}
        valid_windows = [w for w in windows if need_cols.issubset(w.columns)]
        if valid_windows:
            # Tag each window with its age (0 = latest) then concat once.
            tagged = []
            for age, win in enumerate(reversed(valid_windows)):
                tmp = win[["route_id", "stop_id", "delay_seconds_mean"]].copy()
                tmp["age"] = age
                tagged.append(tmp)
            all_wins = pd.concat(tagged, ignore_index=True)
            all_wins["node_key"] = all_wins["route_id"].astype(str) + "_" + all_wins["stop_id"].astype(str)

            # For each node keep the most recent (lowest age) observation.
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
        parts = sid.split("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "", sid

    def _base(sid: str) -> str:
        _, stop = _split(sid)
        return stop[:-1] if stop and stop[-1] in ("N", "S") else stop

    def _route_matches(node_route: str, filt: str) -> bool:
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
            # Take the max across matched nodes (typically the two N/S directions
            # of the requested route). Averaging dilutes the active-direction
            # signal with the zero-padded direction whenever the realtime feed
            # only has rows for one of the two.
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
