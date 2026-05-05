"""
Transformación de datos históricos del clima (processed -> cleaned).

Lee los Parquets diarios de la capa 'processed' en MinIO, aplica limpieza,
feature engineering y genera un informe de calidad en JSON para cada día.

Columnas de entrada esperadas:
  Date, Temperature, Rain, Precipitation, Wind Speed, Snow, Cloud Cover

Rutas MinIO:
  Entrada : grupo5/processed/Clima/Clima_Historico/<YYYY-MM-DD>/Clima_Historico_<YYYY-MM-DD>.parquet
  Salida  : grupo5/cleaned/clima_clean/date=<YYYY-MM-DD>/clima_<YYYY-MM-DD>.parquet
            grupo5/cleaned/clima_clean/date=<YYYY-MM-DD>/quality_report_<YYYY-MM-DD>.json

Variables derivadas que se crean:
  - apparent_temp    : sensación térmica (Wind Chill simplificado)
  - precip_3h_accum  : precipitación acumulada en las últimas 3 horas
  - is_freezing      : indicador de temperatura <= 0 °C (riesgo de hielo en tercer raíl)
  - is_high_wind     : indicador de viento > 50 km/h (peligro en puentes)
  - temp_extreme     : indicador de temperatura fuera del rango percentil 10-90
  - hour             : hora del día (0-23)
  - is_rush_hour     : indicador de hora punta (7-9 h y 16-19 h)

Dependencias:
  - pandas, numpy, scipy.stats
  - src.common.minio_client : download_df_parquet, upload_df_parquet, upload_json

Variables de entorno requeridas:
  - MINIO_ACCESS_KEY
  - MINIO_SECRET_KEY
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from scipy import stats

from src.common.minio_client import download_df_parquet, upload_df_parquet, upload_json


REQUIRED_COLS = [
    "Date",
    "Temperature",
    "Rain",
    "Precipitation",
    "Wind Speed",
    "Snow",
    "Cloud Cover",
]

INPUT_BASE_PATH = "grupo5/processed/Clima/Clima_Historico/{day}/Clima_Historico_{day}.parquet"
OUTPUT_DATA_PATH = "grupo5/cleaned/clima_clean/date={day}/clima_{day}.parquet"
OUTPUT_JSON_PATH = "grupo5/cleaned/clima_clean/date={day}/quality_report_{day}.json"


def calculate_apparent_temp(t, ws):
    """
    Calcula la sensación térmica usando la fórmula de Wind Chill simplificada.

    Aproximación válida para climas similares al de Nueva York (temperaturas
    bajas y viento moderado).

    Parámetros
    ----------
    t  : Temperatura en grados Celsius.
    ws : Velocidad del viento en km/h.

    Devuelve
    --------
    Sensación térmica en grados Celsius.
    """
    return 13.12 + 0.6215 * t - 11.37 * (ws ** 0.16) + 0.3965 * t * (ws ** 0.16)


def generate_quality_report(df_before, df_after):
    """
    Genera un informe de calidad en formato diccionario (serializable a JSON).

    Incluye estadísticas básicas de filas eliminadas, nulos y rangos de
    temperatura y precipitación.

    Parámetros
    ----------
    df_before : DataFrame original antes de la limpieza.
    df_after  : DataFrame resultante tras la limpieza.

    Devuelve
    --------
    Diccionario con las métricas de calidad.
    """
    return {
        "execution_at": datetime.now().isoformat(),
        "stats": {
            "rows_raw": len(df_before),
            "rows_clean": len(df_after),
            "removed_rows": len(df_before) - len(df_after),
            "nulls_in_temp": int(df_before["Temperature"].isna().sum())
        },
        "data_ranges": {
            "min_temp": float(df_after["Temperature"].min()) if not df_after.empty else 0,
            "max_temp": float(df_after["Temperature"].max()) if not df_after.empty else 0,
            "total_precip_day": float(df_after["Precipitation"].sum()) if not df_after.empty else 0
        }
    }


def transform_weather_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Aplica la lógica completa de transformación a los datos meteorológicos de un día.

    Pasos:
      1. Conversión de tipos y eliminación de filas con fecha o temperatura nula.
      2. Deduplicación.
      3. Filtrado de outliers de temperatura por Z-Score (umbral 3).
      4. Cálculo de sensación térmica (apparent_temp).
      5. Precipitación acumulada en ventana de 3 horas (precip_3h_accum).
      6. Indicadores de riesgo para el metro (is_freezing, is_high_wind, temp_extreme).
      7. Variables temporales (hour, is_rush_hour) para facilitar joins con datos GTFS.

    Parámetros
    ----------
    df : DataFrame con las columnas requeridas (ver REQUIRED_COLS).

    Devuelve
    --------
    Tupla (df_limpio, informe_de_calidad).
    """
    df_raw = df.copy()

    df['Date'] = pd.to_datetime(df['Date'])
    numeric_cols = ["Temperature", "Precipitation", "Wind Speed", "Snow"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Eliminamos duplicados y filas donde la temperatura o fecha sean nulas
    df = df.dropna(subset=['Date', 'Temperature']).drop_duplicates()

    # Filtrado de outliers por Z-Score: elimina errores de sensor (> 3 desviaciones estándar)
    if len(df) > 5:
        z = np.abs(stats.zscore(df['Temperature']))
        df = df[z < 3]

    df = df.sort_values('Date')

    # Sensación térmica: importante para predecir demanda en estaciones exteriores
    df['apparent_temp'] = df.apply(
        lambda x: calculate_apparent_temp(x['Temperature'], x['Wind Speed']), axis=1
    )

    # Acumulado de lluvia en ventana de 3 horas: el metro se inunda por acumulación
    df['precip_3h_accum'] = df['Precipitation'].rolling(window=3, min_periods=1).sum()

    # Indicadores de riesgo para el metro de NYC
    df['is_freezing'] = (df['Temperature'] <= 0).astype(int)   # riesgo de hielo en tercer raíl
    df['is_high_wind'] = (df['Wind Speed'] > 50).astype(int)   # peligro en puentes (N, Q, B, D)

    # Indicador de temperatura extrema: usa percentiles si hay suficientes datos, si no umbrales fijos
    if len(df) >= 10:
        low = df['Temperature'].quantile(0.10)
        high = df['Temperature'].quantile(0.90)
    else:
        low, high = -5.0, 35.0
    df['temp_extreme'] = ((df['Temperature'] < low) | (df['Temperature'] > high)).astype(int)

    # Variables temporales para facilitar joins con el dataset de la API de metro
    df['hour'] = df['Date'].dt.hour
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    report = generate_quality_report(df_raw, df)
    return df, report


def run_pipeline(start_str: str, end_str: str):
    """
    Descarga, transforma y sube los datos del clima para un rango de fechas.

    Itera día a día entre start_str y end_str, descarga el Parquet de la capa
    processed, aplica `transform_weather_data` y sube el resultado limpio y
    el informe de calidad a la capa cleaned en MinIO.

    Parámetros
    ----------
    start_str : Fecha de inicio en formato 'YYYY-MM-DD'.
    end_str   : Fecha de fin en formato 'YYYY-MM-DD'.
    """
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_str, "%Y-%m-%d").date()

    curr = start_dt
    while curr <= end_dt:
        day = curr.strftime("%Y-%m-%d")
        try:
            df_weather = download_df_parquet(access_key, secret_key, INPUT_BASE_PATH.format(day=day))
            df_clean, report = transform_weather_data(df_weather)
            upload_df_parquet(access_key, secret_key, OUTPUT_DATA_PATH.format(day=day), df_clean)
            upload_json(access_key, secret_key, OUTPUT_JSON_PATH.format(day=day), report)
            print(f"{day}: Procesado correctamente.")
        except Exception as e:
            print(f"{day}: Error en transformación -> {str(e)}")

        curr += timedelta(days=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    run_pipeline(args.start, args.end)
