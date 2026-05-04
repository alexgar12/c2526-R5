"""
Script para la extracción y procesamiento de datos en tiempo real del metro
de Nueva York (MTA) con el objetivo de calcular el retraso de los trenes
respecto a sus horarios previstos.

FUENTES DE DATOS:
    - API GTFS-Realtime MTA: tiempos de llegada/salida actuales de los trenes
      para todas las líneas del metro de Nueva York (A/C/E, B/D/F/M, G, J/Z,
      N/Q/R/W, L, 1-7/S y SIR).
    - GTFS Supplemented (S3 MTA): horarios previstos oficiales de cada tren
      en cada parada, descargado automáticamente desde un ZIP en la nube.

PROCESO:
    1. Se extraen los datos en tiempo real de la API para cada línea y se
       construye un DataFrame con el viaje, parada, hora de llegada/salida
       real y timestamp de la extracción.
    2. Se descargan los horarios previstos y se adaptan para que sean
       compatibles con los datos en tiempo real.
    3. Se cruzan ambos DataFrames y se calcula el retraso en segundos,
       filtrando predicciones futuras y ajustando viajes que cruzan la
       medianoche.

OUTPUT:
    DataFrame con el retraso real (en segundos) de cada tren en cada parada,
    junto con información del viaje, línea, dirección y tipo de día.
"""


import re
import requests
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from google.transit import gtfs_realtime_pb2
import urllib.request
import zipfile
import io
import math
import time


# ─────────────────────────────────────────────
#  Fuentes de datos MTA Real Time


FUENTES = {
    "ACES": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace",
        "lineas": ["A", "C", "E", "Sr"]
    },
    "BDFMS": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-bdfm",
        "lineas": ["B", "D", "F", "M", "Sf"]
    },
    "G": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-g",
        "lineas": ["G"]
    },
    "JZ": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-jz",
        "lineas": ["J", "Z"]
    },
    "NQRW": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-nqrw",
        "lineas": ["N", "Q", "R", "W"]
    },
    "L": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-l",
        "lineas": ["L"]
    },
    "1234567S": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs",
        "lineas": ["1", "2", "3", "4", "5", "6", "7", "S", "GS", "FS", "H"]
    },
    "SIR": {
        "url": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-si",
        "lineas": ["SIR", "SI"]
    }
}



# Datos a DataFrame


def extraccion_linea(url, linea, reintentos=3):
    """
    Extrae los datos de una línea
    """

    for intento in range(reintentos):
        try:
            response = requests.get(url, timeout = 10)
            fuentes = gtfs_realtime_pb2.FeedMessage()
            fuentes.ParseFromString(response.content)

            datos_linea = []
            for entity in fuentes.entity:
                if entity.HasField('trip_update'):
                    trayecto = entity.trip_update

                    if trayecto.trip.route_id == linea:
                        for stop in trayecto.stop_time_update:
                            campos = {
                                'viaje_id': trayecto.trip.trip_id,
                                'linea_id': trayecto.trip.route_id,
                                'parada_id': stop.stop_id,
                                'hora_llegada': (
                                    datetime.fromtimestamp(stop.arrival.time, tz=timezone.utc)
                                    if stop.HasField('arrival') and stop.arrival.time > 0
                                    else None
                                ),
                                'hora_partida': (
                                    datetime.fromtimestamp(stop.departure.time, tz=timezone.utc)
                                    if stop.HasField('departure') and stop.departure.time > 0
                                    else None
                                ),
                                'timestamp': datetime.now(tz=timezone.utc),
                            }

                            datos_linea.append(campos)
            return datos_linea
        
        except Exception as e:
            if intento == reintentos - 1:
                print(f"  [ERROR] Línea {linea} fallida tras {reintentos} intentos: {e}")
                return []
            espera = 2 ** intento
            time.sleep(espera)


def extraccion_datos():
    """
    Repite la función de extracción para cada línea y unifica la información
    de todas ellas en un DataFrame.
    """
    todos_los_datos = []

    for grupo, info in FUENTES.items():
        url = info['url']
        for linea in info['lineas']:
            todos_los_datos.extend(extraccion_linea(url, linea))

    return pd.DataFrame(todos_los_datos)



#  Funciones auxiliares

def conversion_hora_NYC(df):

    """
    Para las variables de tipo datetime, modifica el valor a la hora local de NY
    """
    
    for col in ['hora_llegada', 'hora_partida', 'timestamp']:
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert('America/New_York')
    return df

def dia_segun_fecha_y_formato(df):

    """
    Según el dia en el que se ha hecho la extracción, crea una nueva variable
    que lo clasifica en 3 grupos (Weekday, Saturday, Sunday).

    También se extrae si es fin de semana y que día de la semana es numéricamente

    Posteriormente cambia el formato de las horas y lo convierte a string.
    """

    df['dia'] = df['timestamp'].dt.strftime("%A").apply(
        lambda x: 'Weekday' if x not in ('Saturday', 'Sunday') else x
    )

    df['dow']        = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)

    return df

def direccion_tren(df):

    """
    Según el id de cada parada, se crea una nueva columna que contiene la dirección del tren (0,1)
    """

    norte = (df['parada_id'].str[-1] == 'N')
    sur = (df['parada_id'].str[-1] == 'S')

    df.loc[norte, 'direccion'] = 1 #Dirección Norte
    df.loc[sur, 'direccion'] = 0 #Dirección Sur

    df['direccion'] = df['direccion'].astype('Int64')

    return df


def normalizar_horas(columna):

    """
    Para horas mayores a 24 horas, se convierte a hora del día siguiente
    """
    def ajustar(hora):
        if pd.isna(hora):
            return hora
        partes = hora.split(':')
        h = int(partes[0]) % 24
        return f"{h:02d}:{partes[1]}:{partes[2]}"
 
    return columna.apply(ajustar)

def hora_a_segundos(hora):

    """
    Dado un string con una hora, se calculan los segundos totales
    """
    if pd.isna(hora): 
        return np.nan
    
    partes = hora.split(':')

    return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])


def hora_posterior(hora1, hora2):

    """
    Comprueba si la hora dada como primer parámetro es mayor
    segunda.
    """
    
    s1 = hora_a_segundos(hora1)
    s2 = hora_a_segundos(hora2)
    dif = s1 - s2

    # Tenemos en cuenta posibles primeras horas del día siguiente (00:15) y
    # asumimos que si la diferencia supera las 12 horas es cruce de media noche
    if dif > 43200:   
        dif -= 86400
    elif dif < -43200:
        dif += 86400

    return dif > 0

def filter_delay_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtro suave de outliers: delays fuera de +/- 2.5h suelen ser ruido (pero ajustable)
    """
    antes = len(df)
    mask = df["delay"].isna() | df["delay"].between(-9000, 9000)
    df = df[mask]
    descartadas = antes - len(df)
    if descartadas:
        print(f"  Outliers de delay eliminados: {descartadas} filas ({descartadas/antes*100:.1f}%)")
    return df


def hora_ciclica(df):
    """Codifica la hora de llegada como coordenadas cíclicas (sin/cos).""" 
    hour_float = df["hora_llegada"].dt.hour.astype(float)
    df["hour_sin"] = hour_float.apply(lambda h: math.sin(2 * math.pi * h / 24) if pd.notna(h) else None)
    df["hour_cos"] = hour_float.apply(lambda h: math.cos(2 * math.pi * h / 24) if pd.notna(h) else None)

    return df



#  DataFrame tiempo real



def creacion_df_tiempo_real():

    """
    Creación de dataframe de tiempo real
    """
    df = extraccion_datos()

    if df.empty:
        raise ValueError("No se obtuvieron datos de tiempo real de ninguna línea.")
    
    df = conversion_hora_NYC(df)
    df = dia_segun_fecha_y_formato(df)
    df = direccion_tren(df)

    # Solo descartamos filas sin hora_llegada (necesaria para calcular delay).
    # hora_partida puede ser None en primera/última parada del viaje.
    df = df.dropna(subset=['hora_llegada', 'viaje_id', 'parada_id', 'linea_id'])

    df['segundos_reales'] = (df['hora_llegada'].dt.hour * 3600 +
                             df['hora_llegada'].dt.minute * 60 +
                             df['hora_llegada'].dt.second)
    print(f"  DataFrame tiempo real: {len(df)} filas, {df['linea_id'].nunique()} líneas")

    return df



#  DataFrame horarios previstos
def creacion_df_previsto():

    """
    Creación de dataframe de horarios previstos
    """

    url = "e"

    with urllib.request.urlopen(url) as response:
        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size and total_size.isdigit() else None
        chunk_size = 1024 * 1024
        downloaded = 0
        chunks = []

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
            downloaded += len(chunk)

    zip_data = io.BytesIO(b"".join(chunks))

    with zipfile.ZipFile(zip_data, 'r') as z:
        with z.open("stop_times.txt") as f:
            df = pd.read_csv(f)

    #Día en el que se lleva a cabo el servivio, viene dado como parte del trip_id
    df['day'] = df['trip_id'].str.split('-').str[-2]

    #Modificamos trip_id para que tenga el mismo formato que el id del otro dataframe
    df['trip_id'] = df['trip_id'].str.split('_', n=1).str[-1]

    df['arrival_time'] = normalizar_horas(df['arrival_time'])
    df['departure_time'] = normalizar_horas(df['departure_time'])

    df['segundos_previstos'] = df['arrival_time'].apply(hora_a_segundos)

    return df


def calcular_features_rt(df, df_schedule=None):
    """
    Calcula features derivados de la secuencia del viaje y del histórico
    reciente por línea, necesarios para inferencia en tiempo real.

    Genera:
      - lagged_delay_1, lagged_delay_2: delay en las 1-2 paradas previas
        del mismo viaje.
      - route_rolling_delay: media móvil del delay por línea (últimos 30 min).
      - actual_headway_seconds: segundos entre este tren y el anterior en
        la misma parada y dirección.
      - stops_to_end: paradas restantes hasta el final del viaje.
      - scheduled_time_to_end: segundos programados hasta la última parada.
    
    Parámetros:
      - df: DataFrame filtrado con paradas ya pasadas
      - df_schedule: DataFrame del schedule completo (sin filtrar). Si se proporciona,
        se usa para calcular max_seq desde el schedule completo en lugar de usar
        solo las paradas pasadas en df. Esto corrige el bug donde max_seq era solo
        el máximo de las paradas pasadas observadas en el feed RT.
    """
    if df.empty:
        return df

    # 1) Lagged delays dentro del mismo viaje.
    # Se hace forward-fill del delay por viaje antes del shift para que una
    # parada no-matched (delay=NaN) no rompa la cadena de lags de las siguientes.
    df = df.sort_values(['viaje_id', 'segundos_reales']).reset_index(drop=True)
    _delay_filled = df.groupby('viaje_id')['delay'].transform('ffill')
    same_trip_1 = df['viaje_id'] == df['viaje_id'].shift(1)
    same_trip_2 = df['viaje_id'] == df['viaje_id'].shift(2)
    df['lagged_delay_1'] = _delay_filled.shift(1).where(same_trip_1, np.nan)
    df['lagged_delay_2'] = _delay_filled.shift(2).where(same_trip_2, np.nan)

    # 2) Rolling delay por línea (ventana temporal 30 min sobre hora_llegada)
    df_sorted = (
        df[['linea_id', 'direccion', 'segundos_reales', 'delay']]
        .sort_values(['linea_id', 'direccion', 'segundos_reales'])
        .reset_index()
        .rename(columns={'index': '_orig_idx'})
    )

    df_sorted['route_rolling_delay'] = (
        df_sorted
        .groupby(['linea_id', 'direccion'])['delay']
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    )

    df['route_rolling_delay'] = (
        df_sorted
        .set_index('_orig_idx')['route_rolling_delay']
        .reindex(df.index)
    )

    # 3) Headway: tiempo desde el tren anterior en (parada_id)
    df_hw = (
        df[['parada_id', 'segundos_reales']]
        .sort_values(['parada_id', 'segundos_reales'])
        .reset_index()
        .rename(columns={'index': '_orig_idx'})
    )

    df_hw['actual_headway_seconds'] = (
        df_hw.groupby('parada_id')['segundos_reales'].diff()
    )

    df['actual_headway_seconds'] = (
        df_hw
        .set_index('_orig_idx')['actual_headway_seconds']
        .reindex(df.index)
    )

    # 4) stops_to_end y scheduled_time_to_end por viaje
    # BUG FIX: Si df_schedule está disponible, usar el schedule completo para calcular
    # max_seq en lugar de usar solo el df filtrado (que solo contiene paradas pasadas).
    # De lo contrario, max_seq sería solo el máximo de las paradas pasadas observadas
    # en el feed RT, no el final real del viaje.
    
    if df_schedule is not None and not df_schedule.empty:
        # Calcular desde el schedule completo (df_schedule), usando trip_id como viaje_id
        final_por_viaje = (
            df_schedule.groupby('trip_id').agg(
                max_seq=('stop_sequence', 'max'),
                final_secs=('segundos_previstos', 'max'),
            )
            .reset_index()
            .rename(columns={'trip_id': 'viaje_id'})
            .set_index('viaje_id')
        )
    else:
        # Fallback: usar df (método anterior, pero con los datos filtrados)
        final_por_viaje = df.groupby('viaje_id').agg(
            max_seq=('stop_sequence', 'max'),
            final_secs=('segundos_previstos', 'max'),
        )
    
    df = df.merge(final_por_viaje, left_on='viaje_id', right_index=True, how='left')
    df['stops_to_end'] = df['max_seq'] - df['stop_sequence']
    df['scheduled_time_to_end'] = df['final_secs'] - df['segundos_previstos']
    df = df.drop(columns=['max_seq', 'final_secs'])

    # Cuando el join con stop_times falla (trip_id no está en el schedule),
    # stops_to_end y scheduled_time_to_end quedan NaN. Usar los valores calculados
    # directamente del feed RT como fallback.
    if 'stops_to_end_rt' in df.columns:
        mask = df['stops_to_end'].isna()
        df.loc[mask, 'stops_to_end'] = df.loc[mask, 'stops_to_end_rt']
    if 'scheduled_time_to_end_rt' in df.columns:
        mask_t = df['scheduled_time_to_end'].isna()
        df.loc[mask_t, 'scheduled_time_to_end'] = df.loc[mask_t, 'scheduled_time_to_end_rt']
    df = df.drop(columns=['stops_to_end_rt', 'scheduled_time_to_end_rt'], errors='ignore')

    return df

#  Unión DataFrames
def union_dataframes(df1, df2, inference_mode=False):

    """
    Une los DataFrames de tiempo real y horarios previstos, calcula el delay
    y aplica las transformaciones finales.

    inference_mode: si True, para trenes sin paradas pasadas (todas futuras)
    conserva la parada más inmediata como proxy de posición actual.
    """

    # Intentar primero merge que respete el día de servicio (Weekday/Saturday/Sunday)
    # para evitar elegir el calendario equivocado cuando exista la misma
    # (trip_id, stop_id) repetido para varios periodos. df1 tiene la columna
    # 'dia' (creada en creacion_df_tiempo_real) y df2 tiene 'day' extraída del
    # trip_id en creacion_df_previsto, por lo que podemos hacer un merge
    # izquierdo usando esa columna.
    df = pd.merge(
        df1,
        df2,
        left_on=['viaje_id', 'parada_id', 'dia'],
        right_on=['trip_id', 'stop_id', 'day'],
        how='left',
        suffixes=('', '_sched'),
        indicator=True
    )

    # Para las filas que no encontraron match por día, intentar un merge de
    # fallback ignorando el campo 'day' (comportamiento anterior). Solo rellenar
    # las columnas de schedule cuando estaban ausentes en el merge por día,
    # preservando así los matches por día cuando existan.
    missing_mask = df['_merge'] == 'left_only'
    if missing_mask.any():
        # Extraer las filas no-matcheadas directamente del DataFrame fusionado
        # Esto evita problemas de desalineación de índices entre `df` y `df1`.
        df1_missing = df.loc[missing_mask, df1.columns].copy()
        df1_missing['_row_id'] = df1_missing.index
        
        fallback = pd.merge(
            df1_missing,
            df2,
            left_on=['viaje_id', 'parada_id'],
            right_on=['trip_id', 'stop_id'],
            how='left',
            suffixes=('', '_fb')
        )

        # Si un mismo (viaje_id, parada_id) tiene varias filas de schedule,
        # nos quedamos con la primera coincidencia estable por fila original.
        # Así cada índice left-only recibe como mucho una fila de fallback.
        fallback = (
            fallback
            .sort_values(['_row_id'])
            .drop_duplicates(subset=['_row_id'], keep='first')
            .set_index('_row_id')
        )

        # Rellenar columnas de schedule solo donde estaban NaN
        sched_cols = [c for c in fallback.columns if c not in df1.columns]
        for col in sched_cols:
            if col in df.columns:
                # Rellenar solo las filas que fueron left_only
                df.loc[df1_missing.index, col] = fallback[col].reindex(df1_missing.index).values
            else:
                # Añadir columna nueva (solo para filas missing)
                series = pd.Series(index=df.index, dtype=fallback[col].dtype if col in fallback.columns else object)
                series.loc[df1_missing.index] = fallback[col].reindex(df1_missing.index).values
                df[col] = series

        # Limpiar columnas auxiliares del fallback
        fb_cols = [c for c in df.columns if c.endswith('_fb')]
        if fb_cols:
            df = df.drop(columns=fb_cols)

    # Limpiar columna de indicador de merge
    df = df.drop(columns=['_merge'])

    # Tercer fallback: normalizar shape suffix del trip_id.
    # La MTA a veces publica trips sin shape (ej. '025900_A..N') que en stop_times
    # existen con shape (ej. '025900_A..N09X011'). Stripeamos el sufijo numérico
    # tras N/S en ambos lados y reintentamos el merge solo para las filas sin match.
    _shape_re = re.compile(r'(?<=[NS])\d+\w*$')
    still_missing = df['trip_id'].isna()
    if still_missing.any():
        df1_missing = df.loc[still_missing, [c for c in df1.columns if c in df.columns]].copy()
        df1_missing['_viaje_base'] = df1_missing['viaje_id'].str.replace(_shape_re, '', regex=True)
        df1_missing['_row_id'] = df1_missing.index

        df2_norm = df2.copy()
        df2_norm['_trip_base'] = df2_norm['trip_id'].str.replace(_shape_re, '', regex=True)

        fallback_norm = pd.merge(
            df1_missing,
            df2_norm,
            left_on=['_viaje_base', 'parada_id'],
            right_on=['_trip_base', 'stop_id'],
            how='left',
            suffixes=('', '_fn')
        )
        fallback_norm = (
            fallback_norm
            .sort_values(['_row_id'])
            .drop_duplicates(subset=['_row_id'], keep='first')
            .set_index('_row_id')
        )

        sched_cols_norm = [c for c in df2.columns if c not in df1.columns]
        for col in sched_cols_norm:
            if col in df.columns:
                df.loc[df1_missing.index, col] = fallback_norm[col].reindex(df1_missing.index).values
            else:
                series = pd.Series(index=df.index, dtype=object)
                series.loc[df1_missing.index] = fallback_norm[col].reindex(df1_missing.index).values
                df[col] = series

        fn_cols = [c for c in df.columns if c.endswith('_fn') or c in ('_viaje_base', '_trip_base')]
        if fn_cols:
            df = df.drop(columns=fn_cols, errors='ignore')

    # Marcar trenes no programados: no encontraron match en el schedule
    df['is_unscheduled'] = df['trip_id'].isna()

    # Calcula el retraso: tiempo de llegada predicho/real menos el tiempo programado
    df['delay'] = df['segundos_reales'] - df['segundos_previstos']

    # Ajuste para viajes que cruzan medianoche
    df.loc[df['delay'] > 43200, 'delay'] -= 86400
    df.loc[df['delay'] < -43200, 'delay'] += 86400

    # El GTFS suplementado repite cada (trip_id, stop_id) una vez por período de
    # calendario (WKD, SAT, SUN, etc.), multiplicando las filas. Se eliminan
    # duplicados conservando la entrada cuyo horario programado más se acerca a
    # la llegada real (menor |delay|), que es la del período activo hoy.
    if df.duplicated(subset=['viaje_id', 'parada_id']).any():
        df['_abs_delay'] = df['delay'].abs().fillna(np.inf)
        df = (
            df.sort_values('_abs_delay')
              .drop_duplicates(subset=['viaje_id', 'parada_id'], keep='first')
              .drop(columns=['_abs_delay'])
        )

    # Si el mejor match de calendario tiene |delay| > 1h, es casi seguro un
    # falso match (trip_id colisionado de otro servicio). Se trata como no programado.
    MAX_DELAY_MATCH = 3600
    bad_match = (~df['is_unscheduled']) & (df['delay'].abs() > MAX_DELAY_MATCH)
    if bad_match.any():
        cols_to_null = [c for c in ['delay', 'segundos_previstos', 'stop_sequence'] if c in df.columns]
        df.loc[bad_match, cols_to_null] = np.nan
        df.loc[bad_match, 'is_unscheduled'] = True
        print(f"  [WARN] {bad_match.sum()} filas marcadas unscheduled (|delay|>{MAX_DELAY_MATCH}s)")

    # Descartar paradas futuras: solo nos interesan las ya pasadas o en curso.
    df_past = df[df['timestamp'] >= df['hora_llegada']].copy()

    if inference_mode:
        # Para trenes en tránsito (todas sus paradas son futuras), conservar la
        # parada más inmediata como proxy de posición. Sin esto, el tren queda
        # con 0 filas y se reporta falsamente como "finalizando recorrido".
        trips_with_past = set(df_past['viaje_id']) if not df_past.empty else set()
        trips_no_past = set(df['viaje_id']) - trips_with_past
        if trips_no_past:
            df_imminent = (
                df[df['viaje_id'].isin(trips_no_past) & (df['timestamp'] < df['hora_llegada'])]
                .sort_values('hora_llegada')
                .drop_duplicates(subset=['viaje_id'], keep='first')
                .copy()
            )
            df = pd.concat([df_past, df_imminent], ignore_index=True)
        else:
            df = df_past
    else:
        df = df_past

    #Filtro para delays con valores masivos y transformacion del día de la semana a valor numérico
    df = filter_delay_outliers(df)
    df = hora_ciclica(df)
    df = calcular_features_rt(df, df_schedule=df2)

    columnas_a_eliminar = [
        'dia', 'hora_partida', 'timestamp',
        'segundos_reales', 'trip_id', 'stop_id',
        'arrival_time', 'departure_time', 'day'
    ]
    df = df.drop(columns=columnas_a_eliminar, errors='ignore')

    return df



if __name__ == "__main__":

    df_real_time = None
    df_previsto = None
    
    try:
        print("\nExtrayendo horarios de trenes en tiempo real...")
        df_real_time = creacion_df_tiempo_real()
    except Exception as e:
        print(f"  Error en datos tiempo real: {e}")

    try:
        print("\nExtrayendo horarios de trenes previstos...")
        df_previsto = creacion_df_previsto()
    except Exception as e:
        print(f"  Error en datos previstos: {e}")

    if df_real_time is None or df_previsto is None:
        print("\n[FATAL] No se puede continuar: uno o ambos DataFrames no se pudieron obtener.")
        exit(1)
    try:
        print("\nUniendo DataFrames...")
        df_final = union_dataframes(df_real_time, df_previsto)
    except Exception as e:
        print(f"  [ERROR] Unión de DataFrames: {e}")
        exit(1)

    ruta = "/tmp/realtime_data.parquet"
    df_final.to_parquet(ruta, index=False)
    print(f"\nGuardado en {ruta} ({ruta.stat().st_size / 1024:.1f} KB, {len(df_final)} filas)")