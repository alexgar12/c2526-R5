import openmeteo_requests
import requests_cache
import requests
import pandas as pd
from retry_requests import retry

#cache_session = requests_cache.CachedSession('.cache', expire_after=-1) #Guarda en un archivo local .cache para no tener que pedirlo de nuevo
#retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

session = requests.Session()
retry_session = retry(session, retries=5, backoff_factor=0.2)

openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"

parametros = {
    "latitude": 40.47,
    "longitude" : -73.58,
    "start_date" : "2024-01-01",
    "end_date" : "2025-01-01",
    "hourly" : "temperature_2m,rain,snowfall",
    "timezone" : "auto"
}
try:
    respuestas = openmeteo.weather_api(url, params = parametros)
    respuesta = respuestas[0]
    print(respuesta)
    hourly = respuesta.Hourly()
    print(hourly)
    temp_array = hourly.Variables(0).ValuesAsNumpy()
    lluvia_array = hourly.Variables(1).ValuesAsNumpy()
    nieve_array = hourly.Variables(2).ValuesAsNumpy()

    dates = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
    )
    
    datos = {
        "fecha" : dates,
        "temperatura" : temp_array,
        "lluvia" : lluvia_array,
        "nieve": nieve_array,
    }

    df = pd.DataFrame(datos)
    df.to_parquet("clima_2024.parquet", index = False)
    print("Guardado")
except Exception as e:
    print("Error", e)
