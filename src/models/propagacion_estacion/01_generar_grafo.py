"""
Procesa archivos GTFS scheduled de MinIO para construir el grafo de la red de metro:
  - edge_index  (2, E)   — pares (origen, destino) en formato COO
  - edge_weight (E,)     — pesos Gaussian Kernel sobre tiempos medianos de viaje
  - nodes       list[str]— lista ordenada de stop_ids (índice → nodo)

Guarda en disco: artefactos/grafo.pt
  {
    'edge_index':  torch.Tensor (2, E),
    'edge_weight': torch.Tensor (E,),
    'nodes':       list[str],
    'n_nodes':     int,
  }

Uso
---
    uv run python src/models/propagacion_estacion/01_generar_grafo.py
"""
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import add_self_loops, coalesce

from src.common.minio_client import download_df_parquet

# Params
START_DATE = "2026-02-06"
END_DATE   = "2026-02-19"   # 2 semanas capturan toda la topología del grafo

RUTA_GTFS_TEMPLATE = "grupo5/cleaned/gtfs_clean_scheduled/date={date}/gtfs_scheduled_{date}.parquet"
RUTA_SALIDA        = Path(__file__).parent / "artefactos" / "grafo.pt"

# Credenciales
access_key = os.environ["MINIO_ACCESS_KEY"]
secret_key = os.environ["MINIO_SECRET_KEY"]


def descargar_gtfs(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end).strftime("%Y-%m-%d").tolist()
    dfs = []
    for date in dates:
        try:
            df = download_df_parquet(
                access_key, secret_key,
                RUTA_GTFS_TEMPLATE.format(date=date),
            )
            dfs.append(df)
        except Exception:
            print(f"  No disponible: {date}")

    if not dfs:
        raise RuntimeError("No se pudo descargar ningún archivo GTFS.")

    resultado = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    print(f"GTFS cargado: {len(resultado):,} filas, {resultado['trip_uid'].nunique():,} viajes únicos")
    return resultado


def construir_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae aristas (route_stop → next_route_stop) con tiempo mediano de viaje.

    El nodo es la clave compuesta '{route_id}_{stop_id}', de modo que paradas
    físicamente compartidas por varias líneas se representan como nodos distintos.
    """
    df = df.sort_values(['trip_uid', 'scheduled_seconds']).reset_index(drop=True)
    df['next_stop_id']           = df.groupby('trip_uid')['stop_id'].shift(-1)
    df['next_scheduled_seconds'] = df.groupby('trip_uid')['scheduled_seconds'].shift(-1)

    edges = df.dropna(subset=['next_stop_id']).copy()
    del df
    gc.collect()

    edges['travel_time'] = edges['next_scheduled_seconds'] - edges['scheduled_seconds']
    edges = edges[edges['travel_time'] > 0]

    # Clave compuesta: identifica de forma única una parada dentro de su línea
    edges['node_src'] = edges['route_id'].astype(str) + '_' + edges['stop_id'].astype(str)
    edges['node_dst'] = edges['route_id'].astype(str) + '_' + edges['next_stop_id'].astype(str)

    graph_df = edges.groupby(['node_src', 'node_dst']).agg(
        median_travel_time=('travel_time', 'median'),
        trip_count=('trip_uid', 'count'),
    ).reset_index()
    del edges
    gc.collect()

    print(f"Aristas únicas en el grafo: {len(graph_df):,}")
    return graph_df


def construir_tensores(graph_df: pd.DataFrame):
    """Convierte graph_df a edge_index y edge_weight con Gaussian Kernel."""

    nodes     = sorted(set(graph_df['node_src']) | set(graph_df['node_dst']))
    n_nodes   = len(nodes)
    node2idx  = {s: i for i, s in enumerate(nodes)}

    src_idx = graph_df['node_src'].map(node2idx).values
    dst_idx = graph_df['node_dst'].map(node2idx).values

    distances = graph_df['median_travel_time'].values
    sigma     = distances.std()
    weights   = np.exp(-(distances ** 2) / (sigma ** 2 + 1e-6)).astype(np.float32)

    edge_index  = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    # Auto-conexiones con peso 1.0 para evitar NaNs en la convolución
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value=1.0, num_nodes=n_nodes
    )
    # Eliminar duplicados (dense_to_sparse interno de DConv los crearía)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes=n_nodes)

    print(f"Nodos: {n_nodes} | Aristas (con self-loops, sin duplicados): {edge_weight.numel()}")
    return edge_index, edge_weight, nodes, n_nodes


def main():
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

    print("01: Generar Grafo")
    df_gtfs   = descargar_gtfs(START_DATE, END_DATE)
    graph_df  = construir_edges(df_gtfs)
    del df_gtfs

    edge_index, edge_weight, nodes, n_nodes = construir_tensores(graph_df)
    del graph_df

    torch.save(
        {
            'edge_index':  edge_index,
            'edge_weight': edge_weight,
            'nodes':       nodes,
            'n_nodes':     n_nodes,
        },
        RUTA_SALIDA,
    )
    print(f"Grafo guardado en: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
