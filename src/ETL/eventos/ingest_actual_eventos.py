"""
ingest_actual_eventos.py — Extracción de eventos en tiempo real para NYC.

Obtiene los eventos del día de hoy desde tres fuentes:
  - SeatGeek      → conciertos y eventos musicales
  - NYC Open Data → eventos públicos de alto impacto (desfiles, carreras, etc.)
  - ESPN          → partidos en casa de equipos NYC

Por cada evento calcula las paradas de metro afectadas en un radio de 500m
usando la fórmula de Haversine.

La fusión de fuentes y el transform a stop_id se realiza en
generate_realtime_dataset.py, que importa directamente api_seatgeek,
api_nycopendata y api_espn.

Variables de entorno necesarias:
  - CLIENT_ID_SEATGEEK
  - NYC_OPEN_DATA_TOKEN
  - (ESPN no requiere API key, es pública)
"""

import os
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

from src.ETL.eventos.utils_eventos import (
    cargar_paradas_df,
    fusionar_lista_estaciones,
    obtener_paradas_afectadas,
)

# ─────────────────────────────────────────────
#  Constantes ESPN
# ─────────────────────────────────────────────

BASE_URL_ESPN = "https://site.api.espn.com/apis/site/v2/sports"

NYC_TEAMS = {
    'basketball/nba': ['knicks', 'nets'],
    'baseball/mlb':   ['yankees', 'mets'],
    'hockey/nhl':     ['rangers', 'islanders', 'devils'],
    'football/nfl':   ['giants', 'jets'],
    'soccer/usa.1':   ['new-york-city-fc', 'new-york-red-bulls'],
}

DURACIONES_ESPN = {
    'nba':   2.5,
    'mlb':   3.0,
    'nhl':   2.5,
    'nfl':   3.5,
    'usa.1': 2.0,
}

VENUES_NYC = {
    'Madison Square Garden':  [-73.9934, 40.7505],
    'UBS Arena':              [-73.7229, 40.7226],
    'Prudential Center':      [-74.1713, 40.7334],
    'Yankee Stadium':         [-73.9262, 40.8296],
    'Citi Field':             [-73.8456, 40.7571],
    'MetLife Stadium':        [-74.0744, 40.8135],
    'Red Bull Arena':         [-74.1502, 40.7369],
    'Yankee Stadium II':      [-73.9262, 40.8296],
}

CIUDADES_NYC = {'New York', 'Elmont', 'Newark', 'East Rutherford', 'Harrison'}


# ─────────────────────────────────────────────
#  SeatGeek
# ─────────────────────────────────────────────

def extraccion_actual(fecha, CLIENT_ID, manana):
    """Llama a la API de SeatGeek y devuelve los eventos de NYC entre fecha y manana."""
    url = "https://api.seatgeek.com/2/events"
    params = {
        "client_id": CLIENT_ID,
        "venue.city": "New York",
        "sort": "score.desc",
        "per_page": 100,
        "datetime_local.gte": fecha,
        "datetime_local.lte": manana,
    }
    response = requests.get(url, params=params)
    assert response.status_code == 200, "Error en la extracción de eventos"
    return response.json()


def calcular_salida(fila, tiempos_salida):
    """Calcula la hora de salida estimada sumando la duración del tipo de evento a la hora de inicio."""
    horas_duracion = tiempos_salida.get(fila['tipo'], 2.5)
    hora_fin = pd.to_datetime(fila['hora_inicio']) + timedelta(hours=horas_duracion)
    return hora_fin.strftime('%H:%M')


def api_seatgeek(df_paradas):
    """
    Extrae conciertos y eventos musicales de NYC para hoy desde SeatGeek,
    calcula paradas de metro afectadas y devuelve un DataFrame limpio.
    """
    fecha_hoy_obj = datetime.now()
    manana_obj = fecha_hoy_obj + timedelta(days=1)
    fecha_hoy_str = fecha_hoy_obj.strftime('%Y-%m-%d')
    manana_str = manana_obj.strftime('%Y-%m-%d')

    API_KEY = os.getenv('CLIENT_ID_SEATGEEK')
    assert API_KEY is not None, "Falta la variable de entorno CLIENT_ID_SEATGEEK"

    TIPOS_CONCIERTO = {'concert', 'music_festival', 'classical', 'opera', 'ballet'}

    data = extraccion_actual(fecha_hoy_str, API_KEY, manana_str)
    eventos_limpios = []

    for e in data['events']:
        if e.get('type') not in TIPOS_CONCIERTO:
            continue
        eventos_limpios.append({
            'nombre_evento':     e.get('title'),
            'tipo':              'concert',
            'hora_inicio':       e.get('datetime_local'),
            'lugar':             e['venue'].get('name'),
            'direccion':         e['venue'].get('address', 'Dirección no disponible'),
            'latitud':           e['venue']['location'].get('lat'),
            'longitud':          e['venue']['location'].get('lon'),
            'capacidad':         e['venue'].get('capacity'),
            'popularidad_score': e.get('score'),
            'venue_score':       e['venue'].get('score'),
        })

    df = pd.DataFrame(eventos_limpios)
    if df.empty:
        return df

    df['capacidad'] = df['capacidad'].replace(0, np.nan)

    df['hora_inicio'] = pd.to_datetime(df['hora_inicio'])
    df['fecha_inicio'] = df['hora_inicio'].dt.strftime('%Y-%m-%d')
    df['hora_inicio_str'] = df['hora_inicio'].dt.strftime('%H:%M')

    tiempos_salida = {
        'concert': 3.0, 'music_festival': 8.0, 'classical': 2.5,
        'opera': 3.0, 'ballet': 2.5,
    }

    df['hora_salida_estimada'] = df.apply(lambda fila: calcular_salida(fila, tiempos_salida), axis=1)
    df['fecha_final'] = df['fecha_inicio']
    df['hora_inicio'] = df['hora_inicio_str']
    df = df.drop(columns=['hora_inicio_str'])

    df["coordinates"] = df.apply(lambda fila: [fila['longitud'], fila['latitud']], axis=1)
    df = df.drop(['longitud', 'latitud', 'lugar', 'direccion'], axis=1)

    coords_invalidas = df["coordinates"].apply(lambda c: c == [0, 0] or None in c)
    if coords_invalidas.sum() > 0:
        df = df[~coords_invalidas].copy()

    df["paradas_afectadas"] = df["coordinates"].apply(
        lambda cor: obtener_paradas_afectadas(cor, df_paradas)
    )
    df['paradas_afectadas'] = df['paradas_afectadas'].apply(fusionar_lista_estaciones)
    df = df.drop(columns=["coordinates"])

    return df



#  NYC Open Data


def desde_fecha(fecha_str):
    """Formatea la fecha como inicio del día para la query de NYC Open Data."""
    return f'{fecha_str}T00:00:00.000'


def hasta_fecha(fecha_str):
    """Formatea la fecha como fin del día para la query de NYC Open Data."""
    return f'{fecha_str}T23:59:59.000'


def extraer_intersecciones(localizacion, barrio):
    """
    Parsea el campo event_location de NYC Open Data y convierte en
    intersecciones geocodificables.
    """
    intersecciones = []
    for segmento in localizacion.split(","):
        segmento = segmento.strip()
        if " between " in segmento:
            partes = segmento.split(" between ")
            calle_principal = partes[0].strip()
            for cruce in partes[1].split(" and "):
                cruce = cruce.strip()
                if cruce:
                    intersecciones.append(f"{calle_principal} & {cruce}, {barrio}, New York")
    return intersecciones if intersecciones else [localizacion + f", {barrio}, New York"]


def extraer_coord(localizacion, barrio, geocode):
    """Devuelve (longitud, latitud) de un evento a partir de su campo de localización textual."""
    if pd.isna(localizacion):
        return None, None

    if ":" in localizacion:
        resultado = geocode(localizacion.split(":")[0].strip() + f", {barrio}, New York")
        if resultado:
            return resultado.longitude, resultado.latitude
        return None, None

    coords = []
    for intersection in extraer_intersecciones(localizacion, barrio):
        try:
            resultado = geocode(intersection)
            if resultado:
                coords.append((resultado.latitude, resultado.longitude))
        except Exception:
            continue

    if coords:
        return np.mean([c[1] for c in coords]), np.mean([c[0] for c in coords])
    return None, None


def api_nycopendata(df_paradas):
    """
    Extrae eventos públicos de NYC del día de hoy desde NYC Open Data,
    filtra por nivel de impacto en el tráfico y calcula paradas afectadas.
    """
    urlbase = "https://data.cityofnewyork.us/resource/"
    url_eventos = f"{urlbase}tvpp-9vvx.json"

    load_dotenv()
    token = os.getenv('NYC_OPEN_DATA_TOKEN')
    assert token is not None, "Falta la variable de entorno NYC_OPEN_DATA_TOKEN"

    fecha_hoy_str = datetime.now().strftime('%Y-%m-%d')
    param = {
        "$where": f"start_date_time >= '{desde_fecha(fecha_hoy_str)}' AND start_date_time <= '{hasta_fecha(fecha_hoy_str)}'",
    }

    header = {"X-App-Token": token}
    eventos = requests.get(url=url_eventos, params=param, headers=header)
    if eventos.status_code != 200:
        print(f"Error Parques: {eventos.text}")
    assert eventos.status_code == 200, "Error en la extracción de eventos"

    df = pd.DataFrame(eventos.json())
    if df.empty:
        return df

    df['start_date_time'] = pd.to_datetime(df['start_date_time'], format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce')
    df['end_date_time'] = pd.to_datetime(df['end_date_time'], errors='coerce', format='%Y-%m-%dT%H:%M:%S.%f')

    df = df.drop(["event_id", "event_agency", "street_closure_type", 'community_board',
                  'police_precinct', 'cemsid', 'event_street_side'], axis=1)

    riesgo_map = {
        'Parade': 10, 'Athletic Race / Tour': 10, 'Street Event': 8,
        'Stationary Demonstration': 7, 'Street Festival': 7, 'Special Event': 6,
        'Single Block Festival': 6, 'Bike the Block': 6, 'BID Multi-Block': 6,
        'Plaza Event': 6, 'Plaza Partner Event': 6, 'Block Party': 5,
        'Theater Load in and Load Outs': 5, 'Open Culture': 4, 'Religious Event': 3,
        'Press Conference': 3, 'Health Fair': 3, 'Rigging Permit': 3,
        'Farmers Market': 2, 'Sidewalk Sale': 2, 'Shooting Permit': 2,
        'Filming/Photography': 2, 'Open Street Partner Event': 2, 'Production Event': 1,
        'Sport - Adult': 1, 'Sport - Youth': 1, 'Miscellaneous': 1,
        'Stickball': 1, 'Clean-Up': 1,
    }

    df['nivel_riesgo_tipo'] = df['event_type'].map(riesgo_map).fillna(1)
    df = df.sort_values(by="nivel_riesgo_tipo", ascending=False)
    df = df[df.nivel_riesgo_tipo > 6]

    geolocator = Nominatim(user_agent="nyc_events_geocoder", timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2, swallow_exceptions=True, error_wait_seconds=2.0)

    df["coordenadas"] = df.apply(
        lambda row: list(extraer_coord(row["event_location"], row["event_borough"], geocode)), axis=1
    )

    coords_invalidas = df["coordenadas"].apply(lambda c: None in c)
    if coords_invalidas.sum() > 0:
        df = df[~coords_invalidas].copy()

    df["paradas_afectadas"] = df["coordenadas"].apply(
        lambda cor: obtener_paradas_afectadas(cor, df_paradas)
    )
    df['paradas_afectadas'] = df['paradas_afectadas'].apply(fusionar_lista_estaciones)

    df['fecha_inicio'] = df['start_date_time'].dt.strftime('%Y-%m-%d')
    df['fecha_final'] = df['end_date_time'].dt.strftime('%Y-%m-%d')

    df = df.drop(columns=["coordenadas", "event_location", "event_borough"])
    df = df.rename(columns={
        'event_name': 'nombre_evento',
        'event_type': 'tipo',
        'start_date_time': 'hora_inicio',
        'end_date_time': 'hora_salida_estimada',
    })

    df['hora_inicio'] = pd.to_datetime(df['hora_inicio'], errors='coerce').dt.strftime('%H:%M')
    df['hora_salida_estimada'] = pd.to_datetime(df['hora_salida_estimada'], errors='coerce').dt.strftime('%H:%M')

    return df


# ─────────────────────────────────────────────
#  ESPN
# ─────────────────────────────────────────────

def extraer_scoreboard_espn(session, sport, fecha_gte, fecha_lte):
    """Llama al endpoint scoreboard de ESPN para un deporte y rango de fechas."""
    url = f"{BASE_URL_ESPN}/{sport}/scoreboard"
    respuesta = session.get(url, params={"dates": f"{fecha_gte}-{fecha_lte}"}, timeout=(10, 60))
    respuesta.raise_for_status()
    return respuesta.json()


def es_partido_en_casa_nyc(evento, equipos_nyc):
    """Comprueba si el equipo local es uno de los equipos NYC que nos interesan."""
    for competidor in evento.get("competitions", [{}])[0].get("competitors", []):
        if competidor.get("homeAway") == "home":
            equipo = competidor.get("team", {})
            slug = equipo.get("slug", "").lower()
            nombre = equipo.get("displayName", "").lower()
            if any(nyc in slug or nyc in nombre for nyc in equipos_nyc):
                return True
    return False


def es_venue_nyc(competicion):
    """Verifica que el partido se juega en un estadio del área de NYC."""
    if not competicion:
        return False
    venue = competicion.get("venue", {})
    ciudad = venue.get("address", {}).get("city", "")
    nombre_venue = venue.get("fullName", "")
    return (ciudad in CIUDADES_NYC) or (nombre_venue in VENUES_NYC)


def geocodificar_venue(nombre_venue, funcion_geocode):
    """Devuelve (latitud, longitud) de un venue, usando VENUES_NYC como caché."""
    if nombre_venue in VENUES_NYC:
        longitud, latitud = VENUES_NYC[nombre_venue]
        return latitud, longitud
    try:
        resultado = funcion_geocode(f"{nombre_venue}, New York")
        if resultado:
            return resultado.latitude, resultado.longitude
    except Exception:
        pass
    return None, None


def api_espn(df_paradas):
    """
    Extrae partidos en casa de equipos NYC para el día de hoy desde ESPN
    y busca paradas de metro afectadas por cada estadio.
    """
    hoy = date.today()
    fecha_gte = hoy.strftime("%Y%m%d")
    fecha_lte = fecha_gte

    session = requests.Session()
    geolocator = Nominatim(user_agent="espn_nyc_geocoder", timeout=10)
    funcion_geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2, swallow_exceptions=True, error_wait_seconds=2.0)

    filas = []

    for sport, equipos in NYC_TEAMS.items():
        try:
            data = extraer_scoreboard_espn(session, sport, fecha_gte, fecha_lte)
            for ev in data.get("events", []):
                comp = ev.get("competitions", [{}])[0]
                if es_partido_en_casa_nyc(ev, equipos) and es_venue_nyc(comp):
                    liga = sport.split("/")[1]
                    nombre_venue = comp.get("venue", {}).get("fullName", "")
                    latitud, longitud = geocodificar_venue(nombre_venue, funcion_geocode)

                    dt_ny = pd.to_datetime(ev.get("date")).tz_convert('America/New_York')
                    duracion = DURACIONES_ESPN.get(liga, 2.5)
                    dt_fin = dt_ny + pd.to_timedelta(duracion, unit='h')
                    hora_salida = dt_fin.strftime('%H:%M')

                    coordinates = [longitud, latitud] if (longitud and latitud) else []

                    paradas = []
                    if coordinates and df_paradas is not None:
                        paradas = fusionar_lista_estaciones(
                            obtener_paradas_afectadas(coordinates, df_paradas)
                        )

                    filas.append({
                        'nombre_evento':        ev.get("name"),
                        'tipo':                 liga,
                        'fecha_inicio':         dt_ny.strftime('%Y-%m-%d'),
                        'hora_inicio':          dt_ny.strftime('%H:%M'),
                        'fecha_final':          dt_fin.strftime('%Y-%m-%d'),
                        'hora_salida_estimada': hora_salida,
                        'score':                1.0,
                        'paradas_afectadas':    paradas,
                    })
        except Exception:
            pass

    return pd.DataFrame(filas)


if __name__ == "__main__":
    load_dotenv()

    print("\nCargando paradas de metro...")
    df_paradas = cargar_paradas_df()

    try:
        print("\nExtrayendo eventos de SeatGeek...")
        df_seat_geek = api_seatgeek(df_paradas)
        print(f"  {len(df_seat_geek)} eventos")
        print(df_seat_geek.head())
    except Exception as e:
        print(f"  Error en SeatGeek: {e}")

    try:
        print("\nExtrayendo eventos de NYC Open Data...")
        df_nyc = api_nycopendata(df_paradas)
        print(f"  {len(df_nyc)} eventos")
        print(df_nyc.head())
    except Exception as e:
        print(f"  Error en NYC Open Data: {e}")

    try:
        print("\nExtrayendo partidos ESPN...")
        df_espn = api_espn(df_paradas)
        print(f"  {len(df_espn)} partidos")
        print(df_espn.head())
    except Exception as e:
        print(f"  Error en ESPN: {e}")
