"""Real-time train positions from MTA GTFS-RT feeds."""
import logging

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
    Returns list of {route_id, lat, lon} for all trains currently in service.

    STOPPED_AT → coordinates of that stop.
    IN_TRANSIT_TO / INCOMING_AT → midpoint between the previous stop and the
    next stop, looked up via (normalized_route_id, stop_id) in prev_stop_for_route.
    Falls back to next-stop coordinates when the previous stop cannot be resolved.
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

        for entity in msg.entity:
            if not entity.HasField("vehicle"):
                continue
            v = entity.vehicle
            if not v.trip.trip_id or not v.stop_id:
                continue

            stop_id = v.stop_id
            coords = gtfs_stops.get(stop_id)
            if coords is None:
                continue

            route_norm = _normalize_route(v.trip.route_id)
            if route_norm is None:
                continue

            lat, lon = coords

            if v.current_status in _MOVING:
                prev_sid = prev_stop_for_route.get((route_norm, stop_id))
                if prev_sid:
                    prev = gtfs_stops.get(prev_sid)
                    if prev:
                        lat = (lat + prev[0]) / 2
                        lon = (lon + prev[1]) / 2

            direction = "N" if stop_id.endswith("N") else "S" if stop_id.endswith("S") else None
            scheduled = v.trip.schedule_relationship == 0
            results.append({
                "route_id": route_norm,
                "trip_id": v.trip.trip_id,
                "lat": lat,
                "lon": lon,
                "next_stop_id": stop_id,
                "schedule_relationship": v.trip.schedule_relationship,
                "direction": direction,
                "status": v.current_status,
                "is_predictable": scheduled,
            })

    logger.info("Vehicle positions: %d trains fetched", len(results))
    return results
