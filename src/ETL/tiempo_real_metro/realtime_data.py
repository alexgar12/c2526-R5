"""
Extracción y procesamiento de datos en tiempo real del metro de Nueva York (MTA)
para calcular el retraso de los trenes respecto a sus horarios previstos.

Fuentes de datos:
    - API GTFS-Realtime MTA: tiempos de llegada/salida actuales de los trenes
      para todas las líneas del metro de Nueva York (A/C/E, B/D/F/M, G, J/Z,
      N/Q/R/W, L, 1-7/S y SIR).
    - GTFS Supplemented (S3 MTA): horarios previstos oficiales de cada tren
      en cada parada, descargado automáticamente desde un ZIP en la nube.

Proceso:
    1. Se extraen los datos en tiempo real de la API para cada línea y se
       construye un DataFrame con el viaje, parada, hora de llegada/salida
       real y timestamp de la extracción.
    2. Se descargan los horarios previstos y se adaptan para que sean
       compatibles con los datos en tiempo real.
    3. Se cruzan ambos DataFrames y se calcula el retraso en segundos,
       filtrando predicciones futuras y ajustando viajes que cruzan la
       medianoche.

Output:
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


# URLs de los feeds GTFS-RT de la MTA por grupo de líneas
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


def extraccion_linea(url, linea, reintentos=3):
    """
    Descarga el feed GTFS-RT de una URL y extrae los datos de una línea concreta.

    Realiza hasta `reintentos` intentos con backoff exponencial ante fallos de red.

    Parámetros:
        url       : URL del endpoint GTFS-RT de la MTA para el grupo de líneas.
        linea     : Identificador de la línea a extraer (p. ej. "A", "1", "SIR").
        reintentos: Número máximo de intentos antes de devolver lista vacía.

    Retorna:
        Lista de diccionarios, uno por actualización de parada, con los campos:
        viaje_id, linea_id, parada_id, hora_llegada, hora_partida, timestamp.
    """
    for intento in range(reintentos):
        try:
            response = requests.get(url, timeout=10)
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
    Itera sobre todos los grupos de líneas definidos en FUENTES y consolida
    sus datos en un único DataFrame.

    Retorna:
        DataFrame con todas las actualizaciones de paradas de todas las líneas,
        con columnas: viaje_id, linea_id, parada_id, hora_llegada, hora_partida, timestamp.
    """
    todos_los_datos = []

    for grupo, info in FUENTES.items():
        url = info['url']
        for linea in info['lineas']:
            todos_los_datos.extend(extraccion_linea(url, linea))

    return pd.DataFrame(todos_los_datos)


def conversion_hora_NYC(df):
    """
    Convierte las columnas de fecha/hora del DataFrame a la zona horaria de Nueva York.

    Parámetros:
        df : DataFrame con columnas hora_llegada, hora_partida y timestamp en UTC.

    Retorna:
        DataFrame con esas tres columnas convertidas a 'America/New_York'.
    """
    for col in ['hora_llegada', 'hora_partida', 'timestamp']:
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert('America/New_York')
    return df

def dia_segun_fecha_y_formato(df):
    """
    Clasifica el día de la extracción en tres categorías (Weekday, Saturday, Sunday)
    y añade columnas auxiliares de día de la semana.

    Columnas añadidas:
        dia        : 'Weekday', 'Saturday' o 'Sunday'.
        dow        : Número de día de la semana (0=lunes, 6=domingo).
        is_weekend : 1 si es sábado o domingo, 0 en caso contrario.

    Parámetros:
        df : DataFrame con columna timestamp en hora de Nueva York.

    Retorna:
        DataFrame con las nuevas columnas añadidas.
    """
    df['dia'] = df['timestamp'].dt.strftime("%A").apply(
        lambda x: 'Weekday' if x not in ('Saturday', 'Sunday') else x
    )

    df['dow']        = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)

    return df

def direccion_tren(df):
    """
    Determina la dirección de circulación del tren a partir del sufijo del stop_id.

    El GTFS de la MTA usa 'N' (norte) o 'S' (sur) como último carácter del stop_id.

    Columnas añadidas:
        direccion : 1 para norte, 0 para sur, NaN si no se puede determinar.

    Parámetros:
        df : DataFrame con columna parada_id.

    Retorna:
        DataFrame con la columna direccion añadida (tipo Int64 nullable).
    """
    norte = (df['parada_id'].str[-1] == 'N')
    sur = (df['parada_id'].str[-1] == 'S')

    df.loc[norte, 'direccion'] = 1
    df.loc[sur, 'direccion'] = 0

    df['direccion'] = df['direccion'].astype('Int64')

    return df


def normalizar_horas(columna):
    """
    Normaliza horas superiores a 23:59 convirtiéndolas al equivalente del día siguiente.

    El GTFS puede codificar horas como "25:30:00" para las 01:30 del día siguiente.
    Esta función las convierte al formato estándar HH:MM:SS de 0 a 23.

    Parámetros:
        columna : Serie de pandas con strings en formato HH:MM:SS.

    Retorna:
        Serie con las horas normalizadas en formato HH:MM:SS.
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
    Convierte una hora en formato HH:MM:SS a segundos desde medianoche.

    Parámetros:
        hora : String con formato HH:MM:SS, o valor NaN.

    Retorna:
        Entero con los segundos totales, o np.nan si la entrada es NaN.
    """
    if pd.isna(hora):
        return np.nan

    partes = hora.split(':')

    return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])


def hora_posterior(hora1, hora2):
    """
    Comprueba si hora1 es posterior a hora2, teniendo en cuenta cruces de medianoche.

    Si la diferencia supera las 12 horas, se asume que hay un cruce de medianoche
    y se ajusta la comparación en consecuencia.

    Parámetros:
        hora1 : String HH:MM:SS de la primera hora.
        hora2 : String HH:MM:SS de la segunda hora.

    Retorna:
        True si hora1 es posterior a hora2, False en caso contrario.
    """
    s1 = hora_a_segundos(hora1)
    s2 = hora_a_segundos(hora2)
    dif = s1 - s2

    if dif > 43200:
        dif -= 86400
    elif dif < -43200:
        dif += 86400

    return dif > 0

def filter_delay_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas cuyo retraso supere el umbral de ±2.5 horas (±9000 segundos).

    Los valores extremos suelen ser ruido de la API o errores de emparejamiento
    entre el feed RT y el horario estático.

    Parámetros:
        df : DataFrame con columna 'delay' en segundos.

    Retorna:
        DataFrame sin las filas consideradas outliers.
    """
    antes = len(df)
    mask = df["delay"].isna() | df["delay"].between(-9000, 9000)
    df = df[mask]
    descartadas = antes - len(df)
    if descartadas:
        print(f"  Outliers de delay eliminados: {descartadas} filas ({descartadas/antes*100:.1f}%)")
    return df


def hora_ciclica(df):
    """
    Codifica la hora de llegada como par de coordenadas cíclicas (seno y coseno).

    Esta representación evita la discontinuidad entre las 23:00 y las 00:00,
    permitiendo que los modelos de ML traten la hora como una variable continua.

    Columnas añadidas:
        hour_sin : Seno de la hora normalizada al ciclo de 24 horas.
        hour_cos : Coseno de la hora normalizada al ciclo de 24 horas.

    Parámetros:
        df : DataFrame con columna hora_llegada de tipo datetime.

    Retorna:
        DataFrame con las columnas hour_sin y hour_cos añadidas.
    """
    hour_float = df["hora_llegada"].dt.hour.astype(float)
    df["hour_sin"] = hour_float.apply(lambda h: math.sin(2 * math.pi * h / 24) if pd.notna(h) else None)
    df["hour_cos"] = hour_float.apply(lambda h: math.cos(2 * math.pi * h / 24) if pd.notna(h) else None)

    return df


def creacion_df_tiempo_real():
    """
    Construye el DataFrame de tiempo real con todos los trenes en circulación.

    Realiza la extracción, conversión de zona horaria, clasificación del día,
    asignación de dirección y calcula los segundos desde medianoche para cada
    hora de llegada.

    Retorna:
        DataFrame limpio con columnas: viaje_id, linea_id, parada_id,
        hora_llegada, hora_partida, timestamp, dia, dow, is_weekend,
        direccion, segundos_reales.

    Lanza ValueError si no se obtuvieron datos de ninguna línea.
    """
    df = extraccion_datos()

    if df.empty:
        raise ValueError("No se obtuvieron datos de tiempo real de ninguna línea.")

    df = conversion_hora_NYC(df)
    df = dia_segun_fecha_y_formato(df)
    df = direccion_tren(df)

    # Se descartan filas sin hora_llegada (necesaria para calcular el delay).
    # hora_partida puede ser None en la primera o última parada del viaje.
    df = df.dropna(subset=['hora_llegada', 'viaje_id', 'parada_id', 'linea_id'])

    df['segundos_reales'] = (df['hora_llegada'].dt.hour * 3600 +
                             df['hora_llegada'].dt.minute * 60 +
                             df['hora_llegada'].dt.second)
    print(f"  DataFrame tiempo real: {len(df)} filas, {df['linea_id'].nunique()} líneas")

    return df


def creacion_df_previsto():
    """
    Descarga el GTFS suplementado de la MTA y construye el DataFrame de horarios previstos.

    El ZIP se descarga desde S3 y se extrae el archivo stop_times.txt. Se normalizan
    las horas superiores a 23:59, se extrae el tipo de día del trip_id y se calcula
    segundos_previstos para cada parada.

    Retorna:
        DataFrame con columnas: trip_id, stop_id, arrival_time, departure_time,
        stop_sequence, day, segundos_previstos (entre otras del GTFS).
    """
    url = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_supplemented.zip"

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

    # El tipo de día de servicio viene codificado en el trip_id (Weekday, Saturday, Sunday)
    df['day'] = df['trip_id'].str.split('-').str[-2]

    # Adaptar el trip_id al mismo formato que el feed en tiempo real
    df['trip_id'] = df['trip_id'].str.split('_', n=1).str[-1]

    df['arrival_time'] = normalizar_horas(df['arrival_time'])
    df['departure_time'] = normalizar_horas(df['departure_time'])

    df['segundos_previstos'] = df['arrival_time'].apply(hora_a_segundos)

    return df


def calcular_features_rt(df, df_schedule=None):
    """
    Calcula features derivadas de la secuencia del viaje y del histórico reciente
    por línea, necesarias para la inferencia en tiempo real.

    Features generadas:
        lagged_delay_1, lagged_delay_2    : Retraso en las 1-2 paradas previas del viaje.
        route_rolling_delay               : Media móvil del delay por línea (ventana de 5 obs.).
        actual_headway_seconds            : Segundos entre este tren y el anterior en la misma parada.
        stops_to_end                      : Paradas restantes hasta el final del viaje.
        scheduled_time_to_end             : Segundos programados hasta la última parada.

    Parámetros:
        df          : DataFrame filtrado con las paradas ya pasadas.
        df_schedule : DataFrame del schedule completo sin filtrar. Si se proporciona,
                      se usa para calcular max_seq desde el schedule completo en lugar
                      de solo las paradas pasadas del feed RT, evitando subestimar
                      stops_to_end.

    Retorna:
        DataFrame enriquecido con las nuevas columnas de features.
    """
    if df.empty:
        return df

    # Lags de retraso dentro del mismo viaje con forward-fill previo para no
    # romper la cadena cuando una parada no tiene match de schedule (delay=NaN).
    df = df.sort_values(['viaje_id', 'segundos_reales']).reset_index(drop=True)
    _delay_filled = df.groupby('viaje_id')['delay'].transform('ffill')
    same_trip_1 = df['viaje_id'] == df['viaje_id'].shift(1)
    same_trip_2 = df['viaje_id'] == df['viaje_id'].shift(2)
    df['lagged_delay_1'] = _delay_filled.shift(1).where(same_trip_1, np.nan)
    df['lagged_delay_2'] = _delay_filled.shift(2).where(same_trip_2, np.nan)

    # Media móvil del delay por línea y dirección (ventana de 5 observaciones, shift(1))
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

    # Headway: diferencia de tiempo entre trenes consecutivos en la misma parada
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

    # stops_to_end y scheduled_time_to_end por viaje.
    # Si df_schedule está disponible, se usa el schedule completo para evitar
    # que max_seq sea solo el máximo de las paradas pasadas observadas en el feed RT.
    if df_schedule is not None and not df_schedule.empty:
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
        final_por_viaje = df.groupby('viaje_id').agg(
            max_seq=('stop_sequence', 'max'),
            final_secs=('segundos_previstos', 'max'),
        )

    df = df.merge(final_por_viaje, left_on='viaje_id', right_index=True, how='left')
    df['stops_to_end'] = df['max_seq'] - df['stop_sequence']
    df['scheduled_time_to_end'] = df['final_secs'] - df['segundos_previstos']
    df = df.drop(columns=['max_seq', 'final_secs'])

    # Fallback: si el join con stop_times falla, usar los valores calculados
    # directamente del feed RT (columnas stops_to_end_rt / scheduled_time_to_end_rt).
    if 'stops_to_end_rt' in df.columns:
        mask = df['stops_to_end'].isna()
        df.loc[mask, 'stops_to_end'] = df.loc[mask, 'stops_to_end_rt']
    if 'scheduled_time_to_end_rt' in df.columns:
        mask_t = df['scheduled_time_to_end'].isna()
        df.loc[mask_t, 'scheduled_time_to_end'] = df.loc[mask_t, 'scheduled_time_to_end_rt']
    df = df.drop(columns=['stops_to_end_rt', 'scheduled_time_to_end_rt'], errors='ignore')

    return df


def union_dataframes(df1, df2, inference_mode=False):
    """
    Une los DataFrames de tiempo real y horarios previstos, calcula el retraso
    y aplica las transformaciones finales.

    El proceso de merge es progresivo:
        1. Merge principal por (viaje_id, parada_id, día de servicio).
        2. Fallback ignorando el día de servicio para filas sin match.
        3. Segundo fallback normalizando el sufijo de shape del trip_id.

    Tras el merge:
        - Se calcula delay = segundos_reales - segundos_previstos.
        - Se ajusta para cruces de medianoche.
        - Se eliminan duplicados conservando el horario más próximo (menor |delay|).
        - Se descartan paradas futuras (timestamp < hora_llegada).
        - Se aplican filtros de outliers y se calculan features adicionales.

    Parámetros:
        df1            : DataFrame de tiempo real (salida de creacion_df_tiempo_real).
        df2            : DataFrame de horarios previstos (salida de creacion_df_previsto).
        inference_mode : Si True, conserva la parada más inmediata para trenes
                         cuyas paradas son todas futuras (útil en tiempo real).

    Retorna:
        DataFrame con delay calculado y columnas auxiliares eliminadas, listo
        para su uso en el pipeline de inferencia o entrenamiento.
    """
    # Merge principal respetando el día de servicio para evitar colisiones de
    # (trip_id, stop_id) entre distintos calendarios (WKD, SAT, SUN).
    df = pd.merge(
        df1,
        df2,
        left_on=['viaje_id', 'parada_id', 'dia'],
        right_on=['trip_id', 'stop_id', 'day'],
        how='left',
        suffixes=('', '_sched'),
        indicator=True
    )

    # Fallback sin día de servicio para filas sin match en el merge principal.
    missing_mask = df['_merge'] == 'left_only'
    if missing_mask.any():
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

        # Por cada fila original sin match, conservar solo la primera coincidencia estable.
        fallback = (
            fallback
            .sort_values(['_row_id'])
            .drop_duplicates(subset=['_row_id'], keep='first')
            .set_index('_row_id')
        )

        sched_cols = [c for c in fallback.columns if c not in df1.columns]
        for col in sched_cols:
            if col in df.columns:
                df.loc[df1_missing.index, col] = fallback[col].reindex(df1_missing.index).values
            else:
                series = pd.Series(index=df.index, dtype=fallback[col].dtype if col in fallback.columns else object)
                series.loc[df1_missing.index] = fallback[col].reindex(df1_missing.index).values
                df[col] = series

        fb_cols = [c for c in df.columns if c.endswith('_fb')]
        if fb_cols:
            df = df.drop(columns=fb_cols)

    df = df.drop(columns=['_merge'])

    # Tercer fallback: normalizar el sufijo de shape del trip_id.
    # La MTA a veces publica trips sin shape ('025900_A..N') que en stop_times
    # existen con shape ('025900_A..N09X011').
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

    # Trenes sin match en ningún calendario del schedule
    df['is_unscheduled'] = df['trip_id'].isna()

    # Cálculo del retraso en segundos
    df['delay'] = df['segundos_reales'] - df['segundos_previstos']

    # Ajuste para viajes que cruzan la medianoche
    df.loc[df['delay'] > 43200, 'delay'] -= 86400
    df.loc[df['delay'] < -43200, 'delay'] += 86400

    # Eliminar duplicados de (viaje_id, parada_id) conservando el match con menor |delay|
    if df.duplicated(subset=['viaje_id', 'parada_id']).any():
        df['_abs_delay'] = df['delay'].abs().fillna(np.inf)
        df = (
            df.sort_values('_abs_delay')
              .drop_duplicates(subset=['viaje_id', 'parada_id'], keep='first')
              .drop(columns=['_abs_delay'])
        )

    # Si el mejor match tiene |delay| > 1h, es probablemente un falso positivo
    MAX_DELAY_MATCH = 3600
    bad_match = (~df['is_unscheduled']) & (df['delay'].abs() > MAX_DELAY_MATCH)
    if bad_match.any():
        cols_to_null = [c for c in ['delay', 'segundos_previstos', 'stop_sequence'] if c in df.columns]
        df.loc[bad_match, cols_to_null] = np.nan
        df.loc[bad_match, 'is_unscheduled'] = True
        print(f"  [WARN] {bad_match.sum()} filas marcadas unscheduled (|delay|>{MAX_DELAY_MATCH}s)")

    # Conservar solo paradas ya pasadas o en curso
    df_past = df[df['timestamp'] >= df['hora_llegada']].copy()

    if inference_mode:
        # En modo inferencia, conservar la parada más inmediata de trenes en tránsito
        # cuyas paradas son todas futuras, para no perder el tren del cómputo.
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
    print(f"\nGuardado en {ruta} ({len(df_final)} filas)")
