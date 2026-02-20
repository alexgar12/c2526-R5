import tarfile
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from google.transit import gtfs_realtime_pb2
from google.protobuf.message import DecodeError

fecha = "2026-02-02"

RAW_DIR = Path(f"data/raw")
TAR_PATH = next(RAW_DIR.glob(f"subwaydatanyc_{fecha}_gtfsrt_*.tar.xz"))
OUT_DIR = Path(f"data/raw/parquet_compressed/cada_60s/date={fecha}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 200_000

TS_RE = re.compile(r"nycsubway_([A-Za-z0-9_]+)_(\d{8}T\d{6}(?:\.\d+)?)Z_")

def parse_gtfsrt_bytes(raw_bytes, snapshot_ts, feed_id):
    """
    Devuelve lista de dicts con eventos (arrival)
    """
    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        feed.ParseFromString(raw_bytes)
    except DecodeError:
        return []  # lo ignoramos

    rows = []

    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue

        tu = entity.trip_update
        trip_id = tu.trip.trip_id
        route_id = tu.trip.route_id or None

        for stu in tu.stop_time_update:
            if not stu.HasField("arrival"):
                continue

            arrival = stu.arrival
            arrival_time = arrival.time if arrival.time else None
            delay_sec = arrival.delay if arrival.delay else None

            rows.append({
                "snapshot_ts_utc": snapshot_ts,
                "feed_id": feed_id,
                "trip_id": trip_id,
                "route_id": route_id,
                "stop_id": stu.stop_id,
                "arrival_time_utc": (
                    datetime.fromtimestamp(arrival_time, tz=timezone.utc)
                    if arrival_time else None
                ),
                "delay_sec": delay_sec,
            })

    return rows

ALLOWED_FEEDS = {"1234567","ACE","BDFM","G","JZ","L","NQRW","SIR"}

def parse_feed_and_ts(name: str):
    m = TS_RE.search(name)
    if not m:
        raise ValueError(f"No se pudo parsear feed/timestamp de {name}")

    feed_id = m.group(1)
    ts = m.group(2)

    if "." in ts:
        dt = datetime.strptime(ts, "%Y%m%dT%H%M%S.%f")
    else:
        dt = datetime.strptime(ts, "%Y%m%dT%H%M%S")

    return feed_id, dt.replace(tzinfo=timezone.utc)


def main():
    batch = []
    part = 0
    ok = 0
    bad = 0
    skipped = 0

    with tarfile.open(TAR_PATH, "r:*") as tar:
        last_bucket = {}  # feed_id -> datetime del bucket (30s)

        members = [m for m in tar.getmembers() if m.name.endswith(".gtfsrt")]

        for member in tqdm(members, desc="Procesando snapshots"):
            # leer bytes
            f = tar.extractfile(member)
            if f is None:
                bad += 1
                continue

            raw = f.read()
            if not raw or len(raw) < 50:
                bad += 1
                continue

            # parsear feed + timestamp del nombre
            try:
                feed_id, snapshot_ts = parse_feed_and_ts(member.name)
            except ValueError:
                bad += 1
                continue
            
            # downsampling: 1 snapshot por cada 30s y por feed 
            bucket_ts = snapshot_ts.replace(microsecond=0)
            bucket_ts = bucket_ts - timedelta(seconds=bucket_ts.second % 30)

            prev = last_bucket.get(feed_id)
            if prev == bucket_ts:
                skipped += 1
                continue
            last_bucket[feed_id] = bucket_ts

            # filtrar feeds no operativos (ej: alerts)
            if feed_id not in ALLOWED_FEEDS:
                skipped += 1
                continue

            # parsear protobuf → filas
            rows = parse_gtfsrt_bytes(raw, snapshot_ts, feed_id)
            if not rows:
                bad += 1
                continue

            ok += 1
            batch.extend(rows)

            # flush batch
            if len(batch) >= BATCH_SIZE:
                df = pd.DataFrame(batch)

                df["snapshot_ts_utc"] = pd.to_datetime(df["snapshot_ts_utc"], utc=True)
                df["arrival_time_utc"] = pd.to_datetime(df["arrival_time_utc"], utc=True)
                df["delay_sec"] = df["delay_sec"].astype("float64")
                df["feed_id"] = df["feed_id"].astype("string")
                df["trip_id"] = df["trip_id"].astype("string")
                df["route_id"] = df["route_id"].astype("string")
                df["stop_id"] = df["stop_id"].astype("string")

                out = OUT_DIR / f"events_part_{part}.parquet"
                df.to_parquet(out, index=False, compression="snappy")
                batch.clear()
                part += 1

        # último batch
        if batch:
            df = pd.DataFrame(batch)
            df["snapshot_ts_utc"] = pd.to_datetime(df["snapshot_ts_utc"], utc=True)
            df["arrival_time_utc"] = pd.to_datetime(df["arrival_time_utc"], utc=True)
            df["delay_sec"] = df["delay_sec"].astype("float64")
            df["feed_id"] = df["feed_id"].astype("string")
            df["trip_id"] = df["trip_id"].astype("string")
            df["route_id"] = df["route_id"].astype("string")
            df["stop_id"] = df["stop_id"].astype("string")
            out = OUT_DIR / f"events_part_{part}.parquet"
            df.to_parquet(out, index=False, compression="snappy")

    print(f"Snapshots OK: {ok} | Bad/invalid: {bad} | Skipped feeds: {skipped}")

if __name__ == "__main__":
    main()