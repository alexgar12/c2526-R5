"""
Extracción de datos meteorológicos históricos para Nueva York.

Consulta la API de archivo de Open-Meteo (endpoint /v1/archive) para obtener
datos horarios de las principales variables meteorológicas en las coordenadas
de Central Park (40.47°N, -73.58°W) para el rango de fechas solicitado.

Las variables extraídas son:
  - Temperature   : temperatura a 2 metros (°C)
  - Rain          : lluvia (mm)
  - Precipitation : precipitación total (mm)
  - Wind Speed    : velocidad del viento a 10 metros (km/h)
  - Snow          : nieve (cm)
  - Cloud Cover   : cobertura nublada (%)

El resultado se particiona por día y se sube a MinIO en:
  grupo5/processed/Clima/Clima_Historico/<YYYY-MM-DD>/Clima_Historico_<YYYY-MM-DD>.parquet

Dependencias:
  - openmeteo_requests : cliente de la API de Open-Meteo
  - retry_requests     : reintentos automáticos ante fallos de red
  - pandas
  - src.common.minio_client.upload_df_parquet

Variables de entorno requeridas:
  - MINIO_ACCESS_KEY
  - MINIO_SECRET_KEY
"""

from datetime import datetime, timedelta
import openmeteo_requests
import requests
import pandas as pd
from retry_requests import retry
import os

from src.common.minio_client import upload_df_parquet


def extraccion(fechaini, fechafin):
    """
    Descarga los datos meteorológicos históricos desde Open-Meteo para el rango dado.

    Utiliza una sesión sin caché para garantizar datos frescos y aplica reintentos
    automáticos ante errores de red.

    Parámetros
    ----------
    fechaini : Fecha de inicio en formato 'YYYY-MM-DD'.
    fechafin : Fecha de fin en formato 'YYYY-MM-DD'.

    Devuelve
    --------
    DataFrame con columnas: Date, Temperature, Rain, Precipitation,
    Wind Speed, Snow, Cloud Cover. Devuelve None si ocurre un error.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    session = requests.Session()
    retry_session = retry(session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    parametros = {
        "latitude": 40.47,
        "longitude": -73.58,  # coordenadas de Central Park
        "start_date": fechaini,
        "end_date": fechafin,
        "hourly": ["temperature_2m", "rain", "precipitation", "wind_speed_10m", "snowfall", "cloud_cover"],
        "timezone": "America/New_York"
    }
    try:
        respuestas = openmeteo.weather_api(url, params=parametros)
        df = transformar_a_df(respuestas)
        return df
    except Exception as e:
        print("Error", e)


def transformar_a_df(respuestas):
    """
    Convierte la respuesta de la API de Open-Meteo en un DataFrame de pandas.

    Toma la primera respuesta del listado, extrae los arrays horarios de cada
    variable y construye un índice temporal en zona horaria de Nueva York.

    Parámetros
    ----------
    respuestas : Lista de objetos de respuesta devueltos por openmeteo_requests.

    Devuelve
    --------
    DataFrame con columnas: Date, Temperature, Rain, Precipitation,
    Wind Speed, Snow, Cloud Cover.
    """
    respuesta = respuestas[0]
    hourly = respuesta.Hourly()
    temp_array = hourly.Variables(0).ValuesAsNumpy()
    lluvia_array = hourly.Variables(1).ValuesAsNumpy()
    prec_array = hourly.Variables(2).ValuesAsNumpy()
    wind_speed_array = hourly.Variables(3).ValuesAsNumpy()
    nieve_array = hourly.Variables(4).ValuesAsNumpy()
    cloud_cover_array = hourly.Variables(5).ValuesAsNumpy()

    dates = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ).tz_convert("America/New_York").tz_localize(None)
    print(dates)
    datos = {
        "Date": dates,
        "Temperature": temp_array,
        "Rain": lluvia_array,
        "Precipitation": prec_array,
        "Wind Speed": wind_speed_array,
        "Snow": nieve_array,
        "Cloud Cover": cloud_cover_array
    }
    df = pd.DataFrame(datos)
    return df


def separar_dias(df):
    """
    Particiona el DataFrame por día y sube cada partición a MinIO.

    Itera sobre los grupos diarios del DataFrame y llama a `subir_a_MinIO`
    para cada uno.

    Parámetros
    ----------
    df : DataFrame con columna 'Date' de tipo datetime.
    """
    for dia, df_dia in df.groupby(df['Date'].dt.date):
        subir_a_MinIO(dia, df_dia)
    print("Todo subido con exito")


def subir_a_MinIO(dia, df_dia):
    """
    Sube el DataFrame de un día concreto a MinIO como Parquet.

    Parámetros
    ----------
    dia    : Objeto date que identifica la partición.
    df_dia : DataFrame con los datos de ese día.
    """
    ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
    name = 'grupo5/processed/Clima/Clima_Historico/' + str(dia) + '/Clima_Historico_' + str(dia) + '.parquet'
    upload_df_parquet(ACCESS_KEY, SECRET_KEY, name, df_dia)


def extraccion_historico(fechaini="2024-12-31", fechafin="2026-01-01"):
    """
    Descarga datos históricos con fechas por defecto que cubren 2025 completo.

    Parámetros
    ----------
    fechaini : Fecha de inicio (por defecto '2024-12-31').
    fechafin : Fecha de fin (por defecto '2026-01-01').

    Devuelve
    --------
    DataFrame con los datos históricos descargados.
    """
    return extraccion(fechaini, fechafin)


def ingest_clima_historico(fechaini="2024-12-31", fechafin="2026-01-01"):
    """
    Orquesta la descarga y subida a MinIO de los datos históricos del clima.

    Descarga los datos del rango indicado y los particiona y sube por día.

    Parámetros
    ----------
    fechaini : Fecha de inicio (por defecto '2024-12-31').
    fechafin : Fecha de fin (por defecto '2026-01-01').
    """
    df_historico = extraccion_historico(fechaini, fechafin)
    separar_dias(df_historico)


if __name__ == "__main__":
    ingest_clima_historico()
