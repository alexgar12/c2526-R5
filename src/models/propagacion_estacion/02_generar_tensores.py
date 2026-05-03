"""
Procesa el DataFrame histórico (dataset_final) para generar tensores
espaciotemporales X e Y, aplica split cronológico y StandardScaler, y guarda
un diccionario .pt con todo lo necesario para entrenar y evaluar el modelo.

Carga:    artefactos/grafo.pt        (nodes, n_nodes)
Guarda:   artefactos/tensores.pt
  {
    'X_train': (T_tr, N, 14),  'Y_train': (T_tr, N, 3),
    'X_val':   (T_va, N, 14),  'Y_val':   (T_va, N, 3),
    'X_test':  (T_te, N, 14),  'Y_test':  (T_te, N, 3),
    'scaler_X': StandardScaler,
    'scaler_Y': StandardScaler,
    'times':   pd.DatetimeIndex,
    'nodes':   list[str],
  }

IMPORTANT: Los meses en RUTAS_DATASET deben ser consecutivos. Si los meses
tienen un gap largo (ej. feb-2025 y ene-2026), pd.date_range crea ~34K
time-steps en lugar de ~5K y el tensor intermedio no cabe en memoria.

Uso
---
    uv run python src/models/propagacion_estacion/02_generar_tensores.py
"""
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.common.minio_client import download_df_parquet

# Params
RUTAS_DATASET = [
    "grupo5/final/year=2025/month=11/dataset_final.parquet",
    "grupo5/final/year=2025/month=12/dataset_final.parquet",
    "grupo5/final/year=2026/month=01/dataset_final.parquet",
    "grupo5/final/year=2026/month=02/dataset_final.parquet",
]
FREQ         = "15min"
SPLIT_TRAIN  = 0.70
SPLIT_VAL    = 0.10   # el resto test

RUTA_GRAFO   = Path(__file__).parent / "artefactos" / "grafo.pt"
RUTA_SALIDA  = Path(__file__).parent / "artefactos" / "tensores.pt"

FEATURE_COLS = [
    'delay_seconds', 'lagged_delay_1', 'lagged_delay_2',
    'is_unscheduled', 'temp_extreme', 'n_eventos_afectando',
    'route_rolling_delay', 'actual_headway_seconds',
    'hour_sin', 'hour_cos', 'dow',
    'afecta_previo', 'afecta_durante', 'afecta_despues',
]
TARGET_COLS = ['station_delay_10m', 'station_delay_20m', 'station_delay_30m']

# Credenciales 
access_key = os.environ["MINIO_ACCESS_KEY"]
secret_key = os.environ["MINIO_SECRET_KEY"]


_AGG_RULES = {
    'delay_seconds':          'mean',
    'lagged_delay_1':         'mean',
    'lagged_delay_2':         'mean',
    'is_unscheduled':         'sum',
    'temp_extreme':           'max',
    'n_eventos_afectando':    'max',
    'route_rolling_delay':    'mean',
    'actual_headway_seconds': 'mean',
    'afecta_previo':          'max',
    'afecta_durante':         'max',
    'afecta_despues':         'max',
    'station_delay_10m':      'mean',
    'station_delay_20m':      'mean',
    'station_delay_30m':      'mean',
}

_CTX_COLS = ['temp_extreme', 'n_eventos_afectando', 'afecta_previo', 'afecta_durante', 'afecta_despues']


def _agrupar(
    df_container: list,
    scheduled_nodes: list[str],
    freq: str,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str]]:
    """Fase 1: agrupa df por (time_bin, node_key). Libera df_final antes de retornar.

    Recibe df en una lista de un elemento para poder eliminar la referencia del
    llamador antes de ejecutar la agregación (patrón container → libera ~10 GB).
    """
    df = df_container.pop()   # única referencia; container queda vacío
    scheduled_set = set(scheduled_nodes)
    node_key = df['route_id'].astype(str) + '_' + df['stop_id'].astype(str)
    mask     = node_key.isin(scheduled_set)

    # Groupby con claves externas: evita copiar df (ahorra ~3 GB de df_work)
    time_bin_s  = pd.to_datetime(df.loc[mask, 'merge_time']).dt.floor(freq).rename('time_bin')
    node_key_s  = node_key[mask].rename('node_key')
    grouped = (
        df.loc[mask, list(_AGG_RULES.keys())]
        .groupby([time_bin_s, node_key_s])
        .agg(_AGG_RULES)
    )
    del df, node_key, mask, time_bin_s, node_key_s
    gc.collect()

    all_times = pd.date_range(
        start=grouped.index.get_level_values('time_bin').min(),
        end=grouped.index.get_level_values('time_bin').max(),
        freq=freq,
    )
    all_nodes = sorted(scheduled_nodes)
    return grouped, all_times, all_nodes


def _construir_tensores(
    grouped: pd.DataFrame,
    all_times: pd.DatetimeIndex,
    all_nodes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fase 2: construye tensores X/Y desde grouped usando indexado numpy directo.

    Evita el unstack (que generaba un DataFrame float64 de ~7 GB en memoria).
    """
    T = len(all_times)
    N = len(all_nodes)
    F = len(FEATURE_COLS)
    H = len(TARGET_COLS)

    time2idx = {t: i for i, t in enumerate(all_times)}
    node2idx = {n: i for i, n in enumerate(all_nodes)}

    # ctx_cols → NaN para ffill/bfill posterior; el resto → 0 por defecto
    X_tensor = np.zeros((T, N, F), dtype='float32')
    Y_tensor = np.zeros((T, N, H), dtype='float32')
    ctx_indices = [FEATURE_COLS.index(c) for c in _CTX_COLS]
    for fi in ctx_indices:
        X_tensor[:, :, fi] = np.nan

    grp = grouped.reset_index()
    del grouped
    gc.collect()

    t_idx = grp['time_bin'].map(time2idx).to_numpy(dtype='int32')
    n_idx = grp['node_key'].map(node2idx).to_numpy(dtype='int32')

    non_temporal = [c for c in FEATURE_COLS if c not in ('hour_sin', 'hour_cos', 'dow')]
    for col in non_temporal:
        fi = FEATURE_COLS.index(col)
        X_tensor[t_idx, n_idx, fi] = grp[col].to_numpy(dtype='float32')
    for hi, col in enumerate(TARGET_COLS):
        Y_tensor[t_idx, n_idx, hi] = grp[col].to_numpy(dtype='float32')

    del grp, t_idx, n_idx
    gc.collect()

    # ffill/bfill por slice (T, N): cada slice ≈ 44 MB, sin problema
    for fi in ctx_indices:
        tmp = pd.DataFrame(X_tensor[:, :, fi])
        X_tensor[:, :, fi] = tmp.ffill().bfill().fillna(0.0).to_numpy(dtype='float32')
        del tmp

    # Features temporales (broadcast a lo largo del eje nodo)
    hour_arr = all_times.hour.to_numpy(dtype='float32')
    X_tensor[:, :, FEATURE_COLS.index('hour_sin')] = np.sin(2 * np.pi * hour_arr / 24)[:, np.newaxis]
    X_tensor[:, :, FEATURE_COLS.index('hour_cos')] = np.cos(2 * np.pi * hour_arr / 24)[:, np.newaxis]
    X_tensor[:, :, FEATURE_COLS.index('dow')]      = all_times.dayofweek.to_numpy(dtype='float32')[:, np.newaxis]

    return X_tensor, Y_tensor


def build_spatiotemporal_tensor(
    df: pd.DataFrame,
    scheduled_nodes: list[str],
    freq: str = FREQ,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """Wrapper de compatibilidad; internamente usa _agrupar + _construir_tensores."""
    if not scheduled_nodes:
        raise ValueError("scheduled_nodes está vacío — no se puede construir el tensor.")
    container = [df]
    grouped, all_times, all_nodes = _agrupar(container, scheduled_nodes, freq)
    X, Y = _construir_tensores(grouped, all_times, all_nodes)
    del grouped
    gc.collect()
    return X, Y, all_times, all_nodes


def escalar(X_train, X_val, X_test, Y_train, Y_val, Y_test):
    """Ajusta StandardScaler sobre train y transforma los tres splits."""
    _, _, F = X_train.shape
    H = Y_train.shape[-1]

    scaler_X = StandardScaler()
    scaler_X.fit(X_train.reshape(-1, F))
    X_tr_s = scaler_X.transform(X_train.reshape(-1, F)).astype('float32').reshape(X_train.shape)
    X_va_s = scaler_X.transform(X_val.reshape(-1, F)).astype('float32').reshape(X_val.shape)
    X_te_s = scaler_X.transform(X_test.reshape(-1, F)).astype('float32').reshape(X_test.shape)

    scaler_Y = StandardScaler()
    scaler_Y.fit(Y_train.reshape(-1, H))
    Y_tr_s = scaler_Y.transform(Y_train.reshape(-1, H)).astype('float32').reshape(Y_train.shape)
    Y_va_s = scaler_Y.transform(Y_val.reshape(-1, H)).astype('float32').reshape(Y_val.shape)
    Y_te_s = scaler_Y.transform(Y_test.reshape(-1, H)).astype('float32').reshape(Y_test.shape)

    return X_tr_s, X_va_s, X_te_s, Y_tr_s, Y_va_s, Y_te_s, scaler_X, scaler_Y


def main():
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

    print("=== 02 Generar Tensores ===")

    # Carga el grafo para obtener la lista de nodos del grafo
    grafo      = torch.load(RUTA_GRAFO, weights_only=False)
    nodes_list = grafo['nodes']
    print(f"Nodos del grafo: {len(nodes_list)}"); sys.stdout.flush()

    print("Descargando dataset_final..."); sys.stdout.flush()
    partes = []
    for ruta in RUTAS_DATASET:
        print(f"  · {ruta}"); sys.stdout.flush()
        partes.append(download_df_parquet(access_key, secret_key, ruta))
    df_final = pd.concat(partes, ignore_index=True)
    del partes
    gc.collect()
    print(f"Filas cargadas: {len(df_final):,}"); sys.stdout.flush()

    # Filtrar a columnas estrictamente necesarias para liberar memoria
    COLS_NECESARIAS = ['route_id', 'stop_id', 'merge_time',
                       'delay_seconds', 'lagged_delay_1', 'lagged_delay_2',
                       'is_unscheduled', 'temp_extreme', 'n_eventos_afectando',
                       'route_rolling_delay', 'actual_headway_seconds',
                       'afecta_previo', 'afecta_durante', 'afecta_despues',
                       'station_delay_10m', 'station_delay_20m', 'station_delay_30m']
    df_final = df_final[COLS_NECESARIAS]
    gc.collect()

    print("Construyendo tensor espaciotemporal..."); sys.stdout.flush()
    # Pasar df_final en un container de un elemento para que _agrupar lo libere
    # desde dentro antes de ejecutar la agregación (ahorra ~10 GB durante el groupby)
    container = [df_final]
    del df_final
    grouped, times, nodes = _agrupar(container, nodes_list, FREQ)
    gc.collect()
    print(f"Timesteps: {len(times)} | Nodos activos: {len(nodes)}"); sys.stdout.flush()

    X, Y = _construir_tensores(grouped, times, nodes)
    del grouped
    gc.collect()
    print(f"X: {X.shape}  Y: {Y.shape}"); sys.stdout.flush()

    # Split cronológico (tensores ya son float32 sin NaN; copy=False evita duplicar ~4 GB)
    X_np = np.nan_to_num(X, nan=0.0, copy=False)
    Y_np = np.nan_to_num(Y, nan=0.0, copy=False)
    del X, Y
    gc.collect()

    T         = X_np.shape[0]
    train_end = int(T * SPLIT_TRAIN)
    val_end   = train_end + int(T * SPLIT_VAL)

    X_train, X_val, X_test = X_np[:train_end], X_np[train_end:val_end], X_np[val_end:]
    Y_train, Y_val, Y_test = Y_np[:train_end], Y_np[train_end:val_end], Y_np[val_end:]
    del X_np, Y_np
    gc.collect()

    print(f"Split → train: {X_train.shape[0]} | val: {X_val.shape[0]} | test: {X_test.shape[0]}"); sys.stdout.flush()

    print("Escalando features y targets..."); sys.stdout.flush()
    X_tr_s, X_va_s, X_te_s, Y_tr_s, Y_va_s, Y_te_s, scaler_X, scaler_Y = escalar(
        X_train, X_val, X_test, Y_train, Y_val, Y_test
    )
    del X_train, X_val, X_test, Y_train, Y_val, Y_test
    gc.collect()

    print("Guardando tensores..."); sys.stdout.flush()
    torch.save(
        {
            'X_train': X_tr_s, 'Y_train': Y_tr_s,
            'X_val':   X_va_s, 'Y_val':   Y_va_s,
            'X_test':  X_te_s, 'Y_test':  Y_te_s,
            'scaler_X': scaler_X,
            'scaler_Y': scaler_Y,
            'times':    times,
            'nodes':    nodes,
        },
        RUTA_SALIDA,
    )
    print(f"Tensores guardados en: {RUTA_SALIDA}"); sys.stdout.flush()


if __name__ == "__main__":
    main()
