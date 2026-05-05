"""
Obtención de posiciones en tiempo real de los trenes del metro de Nueva York.

Descarga los feeds GTFS-RT de la MTA para todas las líneas, parsea los mensajes
Protobuf y construye una lista de posiciones de trenes con coordenadas interpoladas.

Para los trenes en movimiento entre paradas, la posición se calcula como el punto
medio entre la parada actual (next_stop) y la parada anterior según el horario
estático GTFS. Esto produce una visualización más suave en el mapa que mostrar
el tren siempre en la parada siguiente.

Dependencias:
- requests: para descargar los feeds GTFS-RT.
- google.transit.gtfs_realtime_pb2: para parsear los mensajes Protobuf.
- gtfs_stops (dict): coordenadas de paradas del GTFS estático.
- prev_stop_for_route (dict): parada anterior por (route_id, stop_id) del GTFS estático.

Notas:
- Los feeds se consultan en dos pasadas: primero trip_updates (para obtener
  stops_to_end y si el viaje ha comenzado) y luego vehicle positions.
- El trip_id canónico incluye el shape suffix (p.ej. '073200_C..N04R'); los
  mensajes de vehículo a veces emiten solo la versión corta ('073200_C..N'),
  por lo que se busca por prefijo si no hay coincidencia exacta.
"""

import logging
import time

import requests
from google.transit import gtfs_realtime_pb2

logger = logging.getLogger(__name__)

# URLs de los feeds GTFS-RT agrupados por conjunto de líneas
_FEEDS = {
    "ACE":      "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace",
    "BDFM":     "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-bdfm",
    "G":        "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-g",
    "JZ":       "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-jz",
    "NQRW":     "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-nqrw",
    "L":        "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-l",
    "1234567S": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs",
    "SIR":      "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-si",
}

_VALID_ROUTES = {
    '1','2','3','4','5','6','7',
    'A','C','E','B','D','F','M',
    'G','J','Z','L','N','Q','R','W',
    'S','GS','FS','H','SIR',
}

# Estados GTFS-RT que indican que el tren está entre paradas (en movimiento)
_MOVING = frozenset({0, 2})


def _normalize_route(rid: str) -> str | None:
    """
    Normaliza un route_id al identificador canónico de la línea.

    Maneja los sufijos habituales en los feeds de la MTA (guiones, guiones bajos)
    y la variante 'SI' para el Staten Island Railway.

    Parámetros:
        rid: Identificador de ruta crudo del feed GTFS-RT.

    Retorna:
        El route_id normalizado si es una línea válida, None en caso contrario.
    """
    rid = rid.strip()
    if rid in _VALID_ROUTES:
        return rid
    base = rid.split('-')[0].split('_')[0]
    if base in _VALID_ROUTES:
        return base
    if base == 'SI':
        return 'SIR'
    return None


def fetch_positions(
    gtfs_stops: dict[str, tuple[float, float]],
    prev_stop_for_route: dict[tuple[str, str], str],
) -> list[dict]:
    """
    Descarga y parsea los feeds GTFS-RT de todas las líneas para obtener las posiciones de los trenes.

    Realiza dos pasadas por cada feed:
    1. Recorre los trip_updates para construir un mapa (trip_id → paradas restantes y estado).
    2. Recorre los vehicle positions para extraer ruta, parada actual y coordenadas.

    Para trenes en movimiento (status 0 o 2), interpola la posición como el punto
    medio entre la parada actual y la anterior según el GTFS estático.

    Parámetros:
        gtfs_stops: Dict de stop_id → (lat, lon) con coordenadas del GTFS estático.
        prev_stop_for_route: Dict de (route_id, stop_id) → stop_id_anterior para
                             la interpolación de posición.

    Retorna:
        Lista de dicts con los campos: route_id, trip_id, lat, lon, next_stop_id,
        schedule_relationship, direction, status, stops_to_end e is_predictable.
        Los feeds que fallen se omiten con un warning de log.
    """
    results = []

    for feed_key, url in _FEEDS.items():
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            msg = gtfs_realtime_pb2.FeedMessage()
            msg.ParseFromString(resp.content)
        except Exception as exc:
            logger.warning("Feed %s unavailable: %s", feed_key, exc)
            continue

        now_ts = int(time.time())

        # Pasada 1: información completa de cada trip_update.
        # Los trip_update usan el trip_id canónico (con sufijo de shape, ej. "073200_C..N04R"),
        # mientras que vehicle a veces emite la versión corta ("073200_C..N").
        tu_map: dict[str, dict] = {}
        for e in msg.entity:
            if not e.HasField("trip_update"):
                continue
            tu = e.trip_update
            tid = tu.trip.trip_id
            if not tid:
                continue
            stops = list(tu.stop_time_update)
            has_started = False
            for stu in stops:
                t = (stu.arrival.time if stu.HasField("arrival") else 0) or \
                    (stu.departure.time if stu.HasField("departure") else 0)
                if t and t <= now_ts:
                    has_started = True
                    break
            tu_map[tid] = {"stops": stops, "has_started": has_started}

        # Pasada 2: posiciones de vehículos.
        for entity in msg.entity:
            if not entity.HasField("vehicle"):
                continue
            v = entity.vehicle
            if not v.trip.trip_id or not v.stop_id:
                continue

            stop_id = v.stop_id
            route_norm = _normalize_route(v.trip.route_id)
            if route_norm is None:
                continue

            coords = gtfs_stops.get(stop_id)
            if coords is None:
                if v.HasField("position"):
                    coords = (v.position.latitude, v.position.longitude)
                else:
                    continue

            # Resolver trip_id canónico: si vehicle emite ID sin shape suffix,
            # buscar el trip_update cuyo ID empiece por el ID del vehículo.
            vid = v.trip.trip_id
            if vid in tu_map:
                canonical = vid
            else:
                canonical = next((t for t in tu_map if t.startswith(vid)), vid)

            # Extraer stops_to_end del trip_update correspondiente.
            # No filtramos por has_started: la MTA elimina paradas pasadas del feed,
            # por lo que has_started es False para casi todos los trenes en servicio.
            tu_info = tu_map.get(canonical)
            if tu_info:
                stops = tu_info["stops"]
                current_idx = next(
                    (i for i, s in enumerate(stops) if s.stop_id == stop_id),
                    len(stops),
                )
                stops_to_end = len(stops) - current_idx - 1
            else:
                stops_to_end = 0

            lat, lon = coords
            if v.current_status in _MOVING:
                prev_sid = prev_stop_for_route.get((route_norm, stop_id))
                if prev_sid:
                    prev = gtfs_stops.get(prev_sid)
                    if prev:
                        lat = (lat + prev[0]) / 2
                        lon = (lon + prev[1]) / 2

            direction = "N" if stop_id.endswith("N") else "S" if stop_id.endswith("S") else None
            results.append({
                "route_id":              route_norm,
                "trip_id":               canonical,
                "lat":                   lat,
                "lon":                   lon,
                "next_stop_id":          stop_id,
                "schedule_relationship": v.trip.schedule_relationship,
                "direction":             direction,
                "status":                v.current_status,
                "stops_to_end":          max(stops_to_end, 0),
                "is_predictable":        tu_info is not None and stops_to_end > 0,
            })

    logger.info("Vehicle positions: %d trains fetched", len(results))
    return results
