import logging
import math
from datetime import datetime, timezone

import pandas as pd
import requests
from google.transit import gtfs_realtime_pb2

logger = logging.getLogger(__name__)


class FeedUnavailable(Exception):
    """Raised when the GTFS-RT feed request fails (timeout, HTTP error, etc.)."""


class TripNotFound(Exception):
    """Raised when the trip_id is not present in the feed."""

_FEED_FOR_ROUTE: dict[str, str] = {
    **{r: "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace"
       for r in ("A", "C", "E")},
    **{r: "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-bdfm"
       for r in ("B", "D", "F", "M")},
    "G": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-g",
    **{r: "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-jz"
       for r in ("J", "Z")},
    **{r: "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-nqrw"
       for r in ("N", "Q", "R", "W")},
    "L": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-l",
    **{r: "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs"
       for r in ("1", "2", "3", "4", "5", "6", "7", "S", "GS", "FS", "H")},
    "SIR": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-si",
}


def _su_delay(su) -> float | None:
    try:
        if su.HasField("arrival") and su.arrival.delay != 0:
            return float(su.arrival.delay)
    except Exception:
        pass
    try:
        if su.HasField("departure") and su.departure.delay != 0:
            return float(su.departure.delay)
    except Exception:
        pass
    return None


def _normalize_route(rid: str) -> str:
    return rid.strip().split("-")[0].split("_")[0]


def fetch_train_features(
    trip_id: str,
    route_id: str,
    stop_id: str,
    windows: list,
) -> pd.DataFrame | None:

    route_norm = _normalize_route(route_id)
    feed_url = _FEED_FOR_ROUTE.get(route_norm)
    if feed_url is None:
        raise FeedUnavailable(f"No GTFS-RT feed URL for route '{route_norm}'")

    try:
        resp = requests.get(feed_url, timeout=10)
        resp.raise_for_status()
        msg = gtfs_realtime_pb2.FeedMessage()
        msg.ParseFromString(resp.content)
    except Exception as exc:
        raise FeedUnavailable(f"Feed for route {route_norm} unavailable: {exc}") from exc

    trip_update = None
    for entity in msg.entity:
        if entity.HasField("trip_update") and entity.trip_update.trip.trip_id == trip_id:
            trip_update = entity.trip_update
            break

    if trip_update is None:
        raise TripNotFound(f"trip_id '{trip_id}' not found in feed for route {route_norm}")

    stops = list(trip_update.stop_time_update)
    if not stops:
        raise TripNotFound(f"trip_id '{trip_id}' has no stop_time_updates")

    current_idx = next(
        (i for i, s in enumerate(stops) if s.stop_id == stop_id),
        len(stops) - 1,   # fall back to last known stop
    )
    current_su = stops[current_idx]
    future_stops = stops[current_idx + 1:]

    delay_now   = _su_delay(current_su) or 0.0
    prev1_delay = _su_delay(stops[current_idx - 1]) if current_idx >= 1 else 0.0
    prev2_delay = _su_delay(stops[current_idx - 2]) if current_idx >= 2 else 0.0

    stops_to_end = len(future_stops)


    scheduled_time_to_end = 0.0
    try:
        last_su = future_stops[-1] if future_stops else current_su
        curr_t = current_su.arrival.time if current_su.HasField("arrival") else 0
        last_t = last_su.arrival.time    if last_su.HasField("arrival")    else 0
        last_d = _su_delay(last_su) or delay_now
        if curr_t > 0 and last_t > 0:
            scheduled_time_to_end = float((last_t - last_d) - (curr_t - delay_now))
    except Exception:
        pass

    direction    = "N" if stop_id.endswith("N") else "S"
    is_unscheduled = int(trip_update.trip.schedule_relationship != 0)

    now   = datetime.now(timezone.utc)
    hour  = now.hour
    dow   = float(now.weekday())

    row: dict = {
        "stop_id":    stop_id,
        "route_id":   route_norm,
        "direction":  direction,
        "merge_time": now,          

        "delay_seconds_mean":  float(max(0.0, delay_now)),
        "delay_seconds_max":   float(max(0.0, delay_now)),
        "lagged_delay_1_mean": float(max(0.0, prev1_delay or 0.0)),
        "lagged_delay_1_max":  float(max(0.0, prev1_delay or 0.0)),
        "lagged_delay_2_mean": float(max(0.0, prev2_delay or 0.0)),
        "lagged_delay_2_max":  float(max(0.0, prev2_delay or 0.0)),

        "stops_to_end_mean":          float(stops_to_end),
        "scheduled_time_to_end_mean": float(max(0.0, scheduled_time_to_end)),

        "hour_sin_first": math.sin(2 * math.pi * hour / 24),
        "hour_cos_first": math.cos(2 * math.pi * hour / 24),
        "dow_first":      dow,
        "is_weekend_max": float(1 if now.weekday() >= 5 else 0),

        "is_unscheduled_max": float(is_unscheduled),

        "route_rolling_delay_mean":     float(max(0.0, delay_now)),
        "actual_headway_seconds_mean":  0.0,
        "n_eventos_afectando_max":      0.0,
        "temp_extreme_max":             0.0,
        "afecta_previo_max":            0.0,
        "afecta_durante_max":           0.0,
        "afecta_despues_max":           0.0,
        "seconds_since_last_alert_mean": 999_999.0,
    }

    if windows:
        try:
            df_w = windows[-1]
            mask = df_w["route_id"].astype(str) == route_norm
            if "direction" in df_w.columns:
                mask &= df_w["direction"].astype(str) == direction
            sub = df_w[mask]
            if not sub.empty:
                for col in (
                    "route_rolling_delay_mean",
                    "actual_headway_seconds_mean",
                    "n_eventos_afectando_max",
                    "temp_extreme_max",
                    "afecta_previo_max",
                    "afecta_durante_max",
                    "afecta_despues_max",
                    "seconds_since_last_alert_mean",
                ):
                    if col in sub.columns:
                        val = sub[col].mean()
                        import math as _m
                        if not _m.isnan(val):
                            row[col] = float(val)
        except Exception as exc:
            logger.debug("Could not enrich from Drive windows: %s", exc)

    return pd.DataFrame([row])
