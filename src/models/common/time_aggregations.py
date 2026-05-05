"""
Agrega el dataset histórico de transporte en ventanas de tiempo configurable.

Descripción
-----------
Este módulo descarga los datos mensuales del dataset_final desde MinIO, aplica
One-Hot Encoding a variables categóricas, limpia outliers temporales (valores
superiores a 12 horas se reemplazan por NaN) y agrupa los registros por
(stop_id, route_id, direction, ventana_temporal). Al finalizar, calcula lags
de retraso (delay_1_before, delay_2_before, delay_3_before) y sube el DataFrame
resultante de vuelta a MinIO.

Dependencias
------------
- pandas, numpy, gc, argparse
- src.common.minio_client  (download_df_parquet, upload_df_parquet)
- Variables de entorno: MINIO_ACCESS_KEY, MINIO_SECRET_KEY

Uso
---
    uv run python src/models/common/time_aggregations.py --time 60

Notas importantes
-----------------
- El parámetro --time controla el tamaño de la ventana en minutos (p.ej. 30, 60).
- Los lags temporales se invalidan (NaN) si el hueco entre registros supera el
  umbral esperado (~5 minutos extra sobre el período de la ventana).
"""
import os
import pandas as pd
import numpy as np
import gc
import argparse

from src.common.minio_client import download_df_parquet, upload_df_parquet

INPUT_PATH = "grupo5/final/year=2025/month={mes}/dataset_final.parquet"
OUTPUT_PATH = "grupo5/aggregations/DataFrameGroupedByMin={time}.parquet"

def agrupar_mes(df, tiempo):
    """
    Agrupa un DataFrame mensual por ventanas de tiempo y nodo de transporte.

    Aplica One-Hot Encoding a variables categóricas, reemplaza valores de
    retraso fuera del rango [-12h, +12h] por NaN y agrega las columnas
    según su naturaleza (booleanos → max, retrasos → mean/max, etc.).

    Parámetros
    ----------
    df     : pd.DataFrame con los datos del mes (columna merge_time requerida)
    tiempo : str — tamaño de la ventana en minutos, p.ej. '60'

    Devuelve
    --------
    pd.DataFrame agrupado con columnas renombradas (tuples aplanadas a strings).
    """
    df['merge_time'] = pd.to_datetime(df['merge_time'])
    df = pd.get_dummies(df, columns=['category', 'tipo_referente'], dummy_na=False)

    limite_segundos = 43200

    columnas_tiempo = [
        'delay_seconds', 'lagged_delay_1', 'lagged_delay_2',
        'route_rolling_delay', 'target_delay_10m', 'target_delay_20m',
        'target_delay_30m', 'target_delay_45m', 'target_delay_60m',
        'station_delay_10m', 'station_delay_20m', 'station_delay_30m',
        'target_delay_end', 'delta_delay_10m', 'delta_delay_20m',
        'delta_delay_30m', 'delta_delay_45m', 'delta_delay_60m',
        'delta_delay_end',
    ]

    for col in columnas_tiempo:
        if col in df.columns:
            df.loc[(df[col] > limite_segundos) | (df[col] < -limite_segundos), col] = np.nan

    # Nombres de columnas categóricas generadas por get_dummies
    cat_columns = [col for col in df.columns if col.startswith(('category_', 'tipo_referente_'))]

    agg_dict = {
        # Booleanos y flags → max (si ocurrió al menos una vez, se considera True)
        'is_unscheduled': 'max', 'is_weekend': 'max', 'temp_extreme': 'max',
        'afecta_previo': 'max', 'afecta_durante': 'max', 'afecta_despues': 'max',
        'is_alert_just_published': 'max', 'alert_in_next_15m': 'max', 'alert_in_next_30m': 'max',

        # Retrasos y objetivos → media y máximo
        'delay_seconds': ['mean', 'max'], 'lagged_delay_1': ['mean', 'max'], 'lagged_delay_2': ['mean', 'max'],
        'route_rolling_delay': ['mean', 'max'], 'target_delay_10m': ['mean', 'max'], 'target_delay_20m': ['mean', 'max'],
        'target_delay_30m': ['mean', 'max'], 'target_delay_45m': ['mean', 'max'], 'target_delay_60m': ['mean', 'max'],
        'station_delay_10m': ['mean', 'max'], 'station_delay_20m': ['mean', 'max'], 'station_delay_30m': ['mean', 'max'],
        'target_delay_end': ['mean', 'max'], 'delta_delay_10m': ['mean', 'max'], 'delta_delay_20m': ['mean', 'max'],
        'delta_delay_30m': ['mean', 'max'], 'delta_delay_45m': ['mean', 'max'], 'delta_delay_60m': ['mean', 'max'],
        'delta_delay_end': ['mean', 'max'],

        'match_key': 'nunique',

        # Tiempos, distancias y conteos → media o suma
        'actual_headway_seconds': 'mean', 'stops_to_end': 'mean', 'merge_time': 'mean',
        'scheduled_time_to_end': 'mean', 'seconds_since_last_alert': 'mean', 'seconds_to_next_alert': 'mean',
        'num_updates': 'sum', 'n_eventos_afectando': 'max',

        # Variables cíclicas → first (constantes dentro de la ventana)
        'hour_sin': 'first', 'hour_cos': 'first', 'dow': 'first'
    }

    # Columnas One-Hot → suma
    for col in cat_columns:
        agg_dict[col] = 'sum'

    frecuencia = tiempo + 'min'
    print("Iniciando agrupación (esto puede tardar unos minutos)...")
    df_grouped = df.groupby([
        'stop_id',
        'route_id',
        'direction',
        pd.Grouper(key='merge_time', freq=frecuencia)
    ]).agg(agg_dict).reset_index()

    nuevas_columnas = []
    for col in df_grouped.columns.values:
        if isinstance(col, tuple):
            # Tupla del tipo ('delay_seconds', 'mean') → 'delay_seconds_mean'
            if col[1]:
                nuevas_columnas.append(f"{col[0]}_{col[1]}")
            else:
                nuevas_columnas.append(col[0])
        else:
            nuevas_columnas.append(col)

    df_grouped.columns = nuevas_columnas

    return df_grouped


def aggregate_by_x_min(tiempo = '60'):
    """
    Descarga los 12 meses del dataset_final, agrega por ventana de tiempo y
    calcula lags de retraso antes de subir el resultado a MinIO.

    Parámetros
    ----------
    tiempo : str — tamaño de la ventana de agregación en minutos (default '60')

    Notas
    -----
    Los lags delay_1_before, delay_2_before y delay_3_before se anulan (NaN)
    cuando el hueco temporal entre registros consecutivos supera el umbral
    esperado (35, 65 y 95 minutos respectivamente para ventanas de 30 min).
    """
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    if not access_key or not secret_key:
        raise ValueError("Las variables de entorno MINIO_ACCESS_KEY y MINIO_SECRET_KEY no están definidas")

    meses = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    lista_df_agrupados = []
    for mes in meses:
        try:
            df_mes = download_df_parquet(access_key, secret_key, INPUT_PATH.format(mes = mes))

        except Exception as e:
           print(f"\n ERROR al leer el mes {mes}: {type(e).__name__} - {e}")
           continue

        df_mes_agrupado = agrupar_mes(df_mes, tiempo)
        lista_df_agrupados.append(df_mes_agrupado)
        del df_mes
        gc.collect()
        print(f"  -> Mes {mes} agrupado y guardado en lista. RAM liberada.\n")


    df_final = pd.concat(lista_df_agrupados, ignore_index=True)
    del lista_df_agrupados
    gc.collect()
    print("Calculando variables de retraso previo (lags)...")

    df_final = df_final.sort_values(by=['stop_id', 'route_id', 'direction', 'merge_time'])

    columna_objetivo = 'delay_seconds_mean'
    grupos = df_final.groupby(['stop_id', 'route_id', 'direction'])

    df_final['delay_1_before'] = grupos[columna_objetivo].shift(1)
    df_final['delay_2_before'] = grupos[columna_objetivo].shift(2)
    df_final['delay_3_before'] = grupos[columna_objetivo].shift(3)

    # Verificar que el lag corresponde realmente al período esperado
    tiempo_anterior = grupos['merge_time'].shift(1)
    diff_minutos = (df_final['merge_time'] - tiempo_anterior).dt.total_seconds() / 60

    # Si el hueco supera el umbral, el lag no es fiable → NaN
    df_final.loc[diff_minutos > 35, 'delay_1_before'] = np.nan
    df_final.loc[diff_minutos > 65, 'delay_2_before'] = np.nan
    df_final.loc[diff_minutos > 95, 'delay_3_before'] = np.nan
    print("Cargando datos a MinIO...")
    upload_df_parquet(access_key, secret_key, OUTPUT_PATH.format(time=tiempo), df_final)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para agregar datos del transporte en ventanas de tiempo.")

    parser.add_argument(
        '--time',
        type=str,
        default='60',
        help="Tamaño de la ventana de tiempo en minutos (ej. 30, 60, 120)"
    )

    args = parser.parse_args()

    print(f"Iniciando el proceso con una ventana de {args.time} minutos...")
    aggregate_by_x_min(tiempo=args.time)
