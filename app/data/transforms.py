import gc
import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


ALL_FEATURE_COLS = [
    "delay_seconds",
    "lagged_delay_1",
    "lagged_delay_2",
    "is_unscheduled",
    "temp_extreme",
    "n_eventos_afectando",
    "route_rolling_delay",
    "actual_headway_seconds",
    "hour_sin",
    "hour_cos",
    "dow",
    "afecta_previo",
    "afecta_durante",
    "afecta_despues",
]

_DCRNN_COL_MAP: dict[str, str] = {
    "delay_seconds":          "delay_seconds_mean",
    "lagged_delay_1":         "lagged_delay_1_mean",
    "lagged_delay_2":         "lagged_delay_2_mean",
    "is_unscheduled":         "is_unscheduled_max",
    "temp_extreme":           "temp_extreme_max",
    "n_eventos_afectando":    "n_eventos_afectando_max",
    "route_rolling_delay":    "route_rolling_delay_mean",
    "actual_headway_seconds": "actual_headway_seconds_mean",
    "hour_sin":               "hour_sin_first",
    "hour_cos":               "hour_cos_first",
    "dow":                    "dow_first",
    "afecta_previo":          "afecta_previo_max",
    "afecta_durante":         "afecta_durante_max",
    "afecta_despues":         "afecta_despues_max",
}


def windows_to_dcrnn_tensor(
    windows: list[pd.DataFrame],
    nodes: list[str],
    feature_set: list[int],
    scaler_X,
    history_len: int,
) -> torch.Tensor:

    N = len(nodes)
    F_all = len(ALL_FEATURE_COLS)
    node_set = set(nodes)

    df = pd.concat(windows, ignore_index=True)
    df["merge_time"] = pd.to_datetime(df["merge_time"])

    rename = {v: k for k, v in _DCRNN_COL_MAP.items() if v in df.columns}
    df = df.rename(columns=rename)

    df["time_bin"] = df["merge_time"].dt.floor("15min")
    df["hour_sin"] = np.sin(2 * np.pi * df["time_bin"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["time_bin"].dt.hour / 24)
    df["dow"] = df["time_bin"].dt.dayofweek.astype(float)

    df["_node_key"] = df["route_id"].astype(str) + "_" + df["stop_id"].astype(str)
    df = df[df["_node_key"].isin(node_set)]
    agg_rules = {f: "mean" for f in ALL_FEATURE_COLS if f in df.columns}
    df_st = (
        df.groupby(["time_bin", "_node_key"])
        .agg(agg_rules)
        .reset_index()
        .rename(columns={"_node_key": "stop_id"})
    )
    del df
    gc.collect()

    time_bins = sorted(df_st["time_bin"].unique())
    T = len(time_bins)
    nodes_sorted = sorted(nodes)
    full_idx = pd.MultiIndex.from_product(
        [time_bins, nodes_sorted], names=["time_bin", "stop_id"]
    )
    df_full = (
        df_st.set_index(["time_bin", "stop_id"])
        .reindex(full_idx)
        .reset_index()
        .fillna(0)
        .sort_values(["time_bin", "stop_id"])
    )
    del df_st
    gc.collect()

    X = np.zeros((T, N, F_all), dtype=np.float32)
    for fi, feat in enumerate(ALL_FEATURE_COLS):
        if feat in df_full.columns:
            X[:, :, fi] = df_full[feat].values.reshape(T, N)
    del df_full
    gc.collect()

    X_scaled = scaler_X.transform(X.reshape(-1, F_all)).reshape(T, N, F_all).astype(np.float32)

    X_sel = X_scaled[:, :, feature_set]  # (T, N, n_sel)

    n_sel = len(feature_set)
    if T < history_len:
        pad = np.zeros((history_len - T, N, n_sel), dtype=np.float32)
        X_sel = np.concatenate([pad, X_sel], axis=0)
    else:
        X_sel = X_sel[-history_len:]

    return torch.from_numpy(X_sel).unsqueeze(0)  # (1, history_len, N, n_sel)



_DELAY_EXCLUDE = {
    "target_delay_10m_mean", "target_delay_10m_max",
    "target_delay_20m_mean", "target_delay_20m_max",
    "target_delay_30m_mean", "target_delay_30m_max",
    "target_delay_45m_mean", "target_delay_45m_max",
    "target_delay_60m_mean", "target_delay_60m_max",
    "target_delay_end_mean", "target_delay_end_max",
    "delta_delay_10m_mean", "delta_delay_10m_max",
    "delta_delay_20m_mean", "delta_delay_20m_max",
    "delta_delay_30m_mean", "delta_delay_30m_max",
    "delta_delay_45m_mean", "delta_delay_45m_max",
    "delta_delay_60m_mean", "delta_delay_60m_max",
    "delta_delay_end_mean", "delta_delay_end_max",
    "station_delay_10m_mean", "station_delay_10m_max",
    "station_delay_20m_mean", "station_delay_20m_max",
    "station_delay_30m_mean", "station_delay_30m_max",
    "alert_in_next_15m_max", "alert_in_next_30m_max",
    "seconds_to_next_alert_mean", "afecta_despues_max",
    "match_key_nunique",
}


def windows_to_delay_features(windows: list[pd.DataFrame]) -> pd.DataFrame:

    df = windows[-1].copy()
    df["merge_time"] = pd.to_datetime(df["merge_time"])

    # Temporal features added by procesar() in the training script
    df["hora"] = df["merge_time"].dt.hour
    df["minuto"] = df["merge_time"].dt.minute
    df["dia_semana"] = df["merge_time"].dt.dayofweek
    df["hora_mean"] = (
        pd.to_datetime(df["merge_time_mean"]).dt.hour
        if "merge_time_mean" in df.columns
        else df["hora"]
    )

    for col in ("stop_id", "route_id", "direction"):
        if col in df.columns:
            df[col] = df[col].astype("category")

    drop = [c for c in ("merge_time", "merge_time_mean") if c in df.columns]
    df = df.drop(columns=drop)

    drop_exc = [c for c in _DELAY_EXCLUDE if c in df.columns]
    df = df.drop(columns=drop_exc)

    return df




def windows_to_alertas_features(windows: list[pd.DataFrame]) -> pd.DataFrame:

    from src.models.modelos_alertas.common.pipeline_linea import (
        agregar_por_linea,
        agregar_features_rolling_retraso,
    )

    df = pd.concat(windows, ignore_index=True)
    df["merge_time"] = pd.to_datetime(df["merge_time"])
    df = df.sort_values(["stop_id", "route_id", "direction", "merge_time"]).reset_index(drop=True)

    grp = df.groupby(["stop_id", "route_id", "direction"])
    df["delay_1_before"] = grp["delay_seconds_mean"].shift(2).fillna(0)
    df["delay_2_before"] = grp["delay_seconds_mean"].shift(4).fillna(0)
    df["delay_3_before"] = grp["delay_seconds_mean"].shift(6).fillna(0)

    if "seconds_since_last_alert_mean" not in df.columns:
        df["seconds_since_last_alert_mean"] = 999_999.0

    df["alert_in_next_15m_max"] = 0

    df_linea = agregar_por_linea(df)
    df_linea = agregar_features_rolling_retraso(df_linea)

    df_linea = df_linea.sort_values("merge_time")
    df_latest = df_linea.groupby(["route_id", "direction"]).last().reset_index()

    return df_latest
