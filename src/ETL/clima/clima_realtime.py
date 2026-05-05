"""
Extracción de datos meteorológicos en tiempo real para Nueva York.

Consulta la API pública de Open-Meteo (endpoint /v1/forecast) para obtener
los datos horarios y el registro actual de las principales variables
meteorológicas en las coordenadas de Nueva York (Central Park).

Las variables extraídas son:
  - temperature_2m    : temperatura a 2 metros (°C)
  - rain              : lluvia (mm)
  - precipitation     : precipitación total (mm)
  - wind_speed_10m    : velocidad del viento a 10 metros (km/h)
  - snowfall          : nieve (cm)
  - cloud_cover       : cobertura nublada (%)

El DataFrame resultante (datos horarios + fila extra con el instante actual)
se sube a MinIO como Parquet en:
  grupo5/processed/Clima/DataFrame_Clima_TiempoReal.parquet

Dependencias:
  - openmeteo_requests : cliente de la API de Open-Meteo
  - retry_requests     : reintentos automáticos ante fallos de red
  - pandas
  - src.common.minio_client.upload_df_parquet

Variables de entorno requeridas:
  - MINIO_ACCESS_KEY
  - MINIO_SECRET_KEY
"""

import openmeteo_requests

import pandas as pd
import requests
from retry_requests import retry
import datetime
import os

from src.common.minio_client import upload_df_parquet


def extraer_clima_actual():
    """
    Descarga los datos meteorológicos actuales y horarios desde Open-Meteo
    y los sube como Parquet a MinIO.

    Se usa una sesión sin caché para garantizar que siempre se obtienen datos
    frescos. La última fila del DataFrame corresponde al instante 'current'
    devuelto por la API.
    """
    session = requests.Session()
    retry_session = retry(session, retries=5, backoff_factor=0.2)

    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 40.47,
        "longitude": -73.58,
        "hourly": ["temperature_2m", "rain", "precipitation", "wind_speed_10m", "snowfall", "cloud_cover"],
        "current": ["wind_speed_10m", "temperature_2m", "precipitation", "rain", "snowfall", "cloud_cover"],
        "timezone": "America/New_York"
    }
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    current = response.Current()
    current_wind_speed_10m = current.Variables(0).Value()
    current_temperature_2m = current.Variables(1).Value()
    current_precipitation = current.Variables(2).Value()
    current_rain = current.Variables(3).Value()
    current_snowfall = current.Variables(4).Value()
    current_cloud_cover = current.Variables(5).Value()
    print(current_snowfall)
    current_time = current.Time()
    fecha_utc = datetime.datetime.fromtimestamp(current_time, datetime.timezone.utc)

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_rain = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(4).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(5).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["rain"] = hourly_rain
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["snowfall"] = hourly_snowfall
    hourly_data["cloud_cover"] = hourly_cloud_cover

    hourly_dataframe = pd.DataFrame(data=hourly_data)

    # La última fila del DataFrame contiene los valores del instante 'current'
    hourly_dataframe.loc[len(hourly_dataframe)] = [
        fecha_utc, current_temperature_2m, current_rain,
        current_precipitation, current_wind_speed_10m,
        current_snowfall, current_cloud_cover
    ]

    ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
    upload_df_parquet(ACCESS_KEY, SECRET_KEY, 'grupo5/processed/Clima/DataFrame_Clima_TiempoReal.parquet', hourly_dataframe)


if __name__ == "__main__":
    df = extraer_clima_actual()
