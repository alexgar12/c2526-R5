import os
import requests
from datetime import datetime, timedelta
import pandas as pd
import json
import time
#saca mal las cosas, hay que seguir mirando si la api es buena o no
root_url = "https://app.ticketmaster.com/discovery/v2/"

def desde_fecha(fecha_str):
        """
        Convierte 'YYYY-MM-DD' a 'YYYY-MM-DDTHH:MM:SSZ'
        """
        return f'{fecha_str}T00:00:00Z'

def hasta_fecha(fecha_str):
    """
    Convierte 'YYYY-MM-DD' a 'YYYY-MM-DDTHH:MM:SSZ' (Final del día)
    """
    return f'{fecha_str}T23:59:59Z'

def get_cp(embedded):
    venues = embedded.get("venues", [])
    if not venues:
        return None
    return venues[0].get("postalCode")


def extraccion_radio(latlong, radio_millas, fecha_inicio, API_KEY, fecha_fin, pais='US'):
    import time

    url = f'{root_url}events.json'
    page = 0
    size = 200
    all_events = []

    while True:
        params = {
            'apikey': API_KEY,
            'countryCode': pais,
            'startDateTime': fecha_inicio,
            'endDateTime': fecha_fin,
            'latlong': latlong,
            'radius': radio_millas,
            'unit': 'miles',
            'page': page,
            'size': size
        }

        for _ in range(3):
            try:
                response = requests.get(url, params=params, timeout=20)
                if response.status_code == 200:
                    break
                if "Spike arrest violation" in response.text:
                    time.sleep(1.5)
            except requests.exceptions.RequestException:
                time.sleep(2)

        assert response.status_code == 200, f"Error en la extracción: {response.text}"
        data = response.json()

        eventos = data.get('_embedded', {}).get('events', [])
        if not eventos:
            break

        all_events.extend(eventos)

        total_pages = data.get('page', {}).get('totalPages', 0)
        page += 1

        if page * size >= 1000:
            break

        if page >= total_pages:
            break

        time.sleep(0.25)

    return pd.DataFrame(all_events)

API_KEY = os.getenv('TICKETMASTER_API_KEY')
assert API_KEY is not None, "Falta la variable de entorno TICKETMASTER_API_KEY"
inicio_2025 = desde_fecha('2025-01-01')
final_2025 = hasta_fecha('2025-12-31')

def meses_2025():
    rangos = [
        ("2025-01-01", "2025-01-31"),
        ("2025-02-01", "2025-02-28"),
        ("2025-03-01", "2025-03-31"),
        ("2025-04-01", "2025-04-30"),
        ("2025-05-01", "2025-05-31"),
        ("2025-06-01", "2025-06-30"),
        ("2025-07-01", "2025-07-31"),
        ("2025-08-01", "2025-08-31"),
        ("2025-09-01", "2025-09-30"),
        ("2025-10-01", "2025-10-31"),
        ("2025-11-01", "2025-11-30"),
        ("2025-12-01", "2025-12-31"),
    ]
    return [(desde_fecha(i), hasta_fecha(f)) for i, f in rangos]

dfs = []

for ini, fin in meses_2025():
    print("Descargando rango:", ini, "→", fin)
    df_mes = extraccion_radio("40.7128,-74.0060", 25, ini, API_KEY, fin)
    dfs.append(df_mes)
    time.sleep(0.5)

df = pd.concat(dfs, ignore_index=True)

if "id" in df.columns:
    df = df.drop_duplicates(subset=["id"])



df = df[["name",  "dates", "_embedded"]]
df['hora_inicio'] = df['dates'].apply(lambda x: x['start']['dateTime'])
df['hora_final'] = df['dates'].apply(lambda x: x.get('end', {}).get('dateTime'))
df['direccion'] = df["_embedded"].apply(lambda val: val["venues"][0].get("address", {}).get("line1") if val.get("venues") else None)
df["CP"] = df["_embedded"].apply(get_cp)
df['parking'] = df["_embedded"].apply(lambda val: "Yes" if (val.get("venues") and val["venues"][0].get("parkingDetail")) else "No")
df = df.drop("dates", axis = 1)
df = df.drop("_embedded", axis=1)
df['hora_inicio'] = pd.to_datetime((df['hora_inicio']))
df['hora_final'] = pd.to_datetime((df['hora_final']))
df['hora_inicio'] = df['hora_inicio'].dt.tz_convert('America/New_York')
df['hora_final'] = df['hora_final'].dt.tz_convert('America/New_York')
df['hora_inicio'] = df['hora_inicio'].dt.strftime('%H:%M')
df['hora_final'] = df['hora_final'].dt.strftime('%H:%M')
print(df)
print(len(df))