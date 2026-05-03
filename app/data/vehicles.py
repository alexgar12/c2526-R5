"""Real-time train positions from MTA GTFS-RT feeds."""
import logging
import time

import requests
from google.transit import gtfs_realtime_pb2

logger = logging.getLogger(__name__)

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

# 1 = STOPPED_AT, 0 = INCOMING_AT, 2 = IN_TRANSIT_TO
_MOVING = frozenset({0, 2})


def _normalize_route(rid: str) -> str | None:
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
    Returns list of train positions for all vehicles currently in the GTFS-RT feeds.

    Each entry includes:
      - trip_id: canonical ID from trip_update (resolves vehicle/trip_update format mismatch)
      - stops_to_end: remaining stops according to trip_update
      - has_started: whether any stop time is already in the past
      - is_predictable: True when has_started=True and stops_to_end>0

    Coordinates: GTFS stop position (STOPPED_AT) or midpoint with previous stop
    (IN_TRANSIT_TO/INCOMING_AT). Falls back to GPS position if stop unknown.
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

            # Coordenadas: parada GTFS o GPS del vehículo como fallback.
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

            # Extraer stops_to_end y has_started del trip_update correspondiente.
            tu_info = tu_map.get(canonical)
            if tu_info:
                stops = tu_info["stops"]
                has_started = tu_info["has_started"]
                # Pre-asignado: tiene trip_update pero ninguna parada ha pasado aún.
                if not has_started:
                    continue
                current_idx = next(
                    (i for i, s in enumerate(stops) if s.stop_id == stop_id),
                    len(stops),
                )
                stops_to_end = len(stops) - current_idx - 1
            else:
                has_started = False
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
                "route_id":             route_norm,
                "trip_id":              canonical,
                "lat":                  lat,
                "lon":                  lon,
                "next_stop_id":         stop_id,
                "schedule_relationship": v.trip.schedule_relationship,
                "direction":            direction,
                "status":               v.current_status,
                "stops_to_end":         max(stops_to_end, 0),
                "has_started":          has_started,
                "is_predictable":       has_started and stops_to_end > 0,
            })

    logger.info("Vehicle positions: %d trains fetched", len(results))
    return results
