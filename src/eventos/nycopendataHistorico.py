import os
import requests
from datetime import datetime, timedelta
import pandas as pd
import json

urlbase = "https://data.cityofnewyork.us/resource/"

def desde_fecha(fecha_str):
       return f'{fecha_str}T00:00:00.000'

def hasta_fecha(fecha_str):
   return f'{fecha_str}T23:59:59.000'

def extraccion_actual(ini, fin, token):
    url_eventos = f"{urlbase}bkfu-528j.json"
    
    param = {
        "$where": f"start_date_time >= '{ini}' AND start_date_time <= '{fin}'",
        "$limit": 7000000,
        "$offset": 0
    }

    header = {
        "X-App-Token": token
    }

    response = requests.get(url_eventos, params=param, headers=header)
    assert response.status_code == 200, "Error en la extracción"

    data = response.json()
    df = pd.DataFrame(data)
    return df

TOKEN = os.getenv("NYCOPENDATA_TOKEN")
assert TOKEN is not None, "Falta la variable de entorno NYCOPENDATA_TOKEN"

inicio_2025 = desde_fecha('2025-01-01')
final_2025 = hasta_fecha('2025-12-31')

df = extraccion_actual(inicio_2025, final_2025, TOKEN)
print(df.shape)
print(df.columns)

if df.empty:
    print("No hay eventos en ese rango de fechas")
    exit()

df["borough"] = df["event_borough"]

df = df[['event_name', 'event_type', 'start_date_time', 'end_date_time',
         'event_location', 'borough', 'community_board']]

df['start_date_time'] = pd.to_datetime(df['start_date_time'])
df['end_date_time'] = pd.to_datetime(df['end_date_time'])

df['duration_hours'] = (df['end_date_time'] - df['start_date_time']).dt.total_seconds() / 3600

df = df.dropna(subset=['event_type'])

riesgo_map = {
    'Parade': 10,
    'Athletic Race / Tour': 10,
    'Street Event': 8,
    'Special Event': 7,
    'Plaza Event': 6,
    'Plaza Partner Event': 6,
    'Theater Load in and Load Outs': 5,
    'Religious Event': 3,
    'Farmers Market': 2,
    'Sidewalk Sale': 2,
    'Production Event': 1,
    'Sport - Adult': 1,
    'Sport - Youth': 1,
    'Miscellaneous': 1,
    'Open Street Partner Event': 2
}

df['nivel_riesgo_tipo'] = df['event_type'].map(riesgo_map)

print(df)
