import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from app.config import settings
from app.data.drive import download_windows
from app.models.alertas_infer import run_alerts
from app.models.delay_infer import run_delays, run_delay_single
from app.models.delta_infer import run_delta, run_delta_single
from src.ETL.pipelines.realtime.preprocess_realtime_lgbm import get_trip_features
from app.models.dcrnn_infer import run_propagation
from app.schemas import (
    AlertResponse,
    AllPredictionsResponse,
    DeltaResponse,
    DelayResponse,
    PropagationResponse,
)

router = APIRouter(prefix="/predict")
logger = logging.getLogger(__name__)


def _drive_window_fallback(windows: list, route_id: str, stop_id: str):
    """
    Build a minimal single-row DataFrame from the latest Drive window when the
    GTFS-RT feed is unavailable.  Provides enough features to run the models
    at the cost of having no per-train lag/trip-progress information.
    """
    import math
    import numpy as np
    from datetime import datetime, timezone
    import pandas as pd

    if not windows:
        return None
    df = windows[-1].copy()
    base = df["stop_id"].astype(str).str.rstrip("NS") if "stop_id" in df.columns else df.index.astype(str)
    mask = (df["stop_id"].astype(str) == stop_id) | (base == stop_id)
    sub = df[mask]
    if sub.empty:
        # Fall back to route-level average
        route_norm = route_id.strip().split("-")[0].split("_")[0]
        if "route_id" in df.columns:
            sub = df[df["route_id"].astype(str) == route_norm]
    if sub.empty:
        return None

    row = sub.iloc[0:1].copy()
    now = datetime.now(timezone.utc)
    row["merge_time"] = now
    # Clear trip-specific fields we can't derive from the window
    for col in ("stops_to_end_mean", "scheduled_time_to_end_mean",
                "lagged_delay_1_mean", "lagged_delay_2_mean"):
        if col not in row.columns:
            row[col] = 0.0
    return row


async def _get_windows(request: Request) -> list:
    """Return cached windows or download fresh ones from Drive."""
    cache = request.app.state.cache
    cached = cache.get("windows")
    if cached is not None:
        return cached

    windows = await asyncio.to_thread(
        download_windows,
        n_windows=settings.n_windows,
        token_path=settings.drive_token_path,
        folder_name=settings.google_drive_folder_name,
    )
    cache.set("windows", windows)
    return windows


# ── Current observed delay (latest window) ───────────────────────────────────

@router.get("/current")
async def get_current_delay(
    request: Request,
    stop_id: Optional[str] = Query(default=None),
) -> dict:
    """Return the last observed delay_seconds from the most recent Drive window."""
    windows = await _get_windows(request)
    df = windows[-1].copy()

    if stop_id and "stop_id" in df.columns:
        base = df["stop_id"].astype(str).str.rstrip("NS")
        df = df[(df["stop_id"].astype(str) == stop_id) | (base == stop_id)]

    if df.empty or "delay_seconds_mean" not in df.columns:
        window_routes = windows[-1]["route_id"].astype(str).unique().tolist() if "route_id" in windows[-1].columns else []
        logger.warning(
            "predict/current: no data for stop_id=%r. "
            "Window has %d rows, routes=%s, stop_id sample=%s",
            stop_id,
            len(windows[-1]),
            sorted(window_routes)[:10],
            windows[-1]["stop_id"].astype(str).unique()[:5].tolist() if "stop_id" in windows[-1].columns else [],
        )
        return {"stop_id": stop_id, "delay_seconds": None}

    import numpy as np
    delay = float(np.clip(df["delay_seconds_mean"].mean(), 0, None))
    logger.debug("predict/current: stop_id=%r → delay=%.1fs (%d rows matched)", stop_id, delay, len(df))
    return {"stop_id": stop_id, "delay_seconds": delay}


# ── Propagation (DCRNN) ───────────────────────────────────────────────────────

@router.get("/propagation", response_model=PropagationResponse)
async def predict_propagation(
    request: Request,
    stop_id: Optional[str] = Query(default=None),
    route_id: Optional[str] = Query(default=None),
) -> PropagationResponse:
    registry = request.app.state.registry
    if registry.dcrnn is None:
        raise HTTPException(503, detail=f"DCRNN not available: {registry.errors.get('dcrnn', 'not loaded')}")
    windows = await _get_windows(request)
    return await asyncio.to_thread(
        run_propagation,
        entry=registry.dcrnn,
        windows=windows,
        stations_meta=request.app.state.stations_meta,
        stop_id_filter=stop_id,
        route_id_filter=route_id,
    )


# ── Delay (LightGBM) ─────────────────────────────────────────────────────────

@router.get("/delay/30m", response_model=DelayResponse)
async def predict_delay_30m(
    request: Request,
    route_id: Optional[str] = Query(default=None),
    stop_id: Optional[str] = Query(default=None),
    min_delay: float = Query(default=0.0, ge=0),
) -> DelayResponse:
    registry = request.app.state.registry
    if registry.lgbm_delay_30m is None:
        raise HTTPException(503, detail=f"delay/30m not available: {registry.errors.get('lgbm_delay_30m', 'not loaded')}")
    windows = await _get_windows(request)
    return await asyncio.to_thread(
        run_delays,
        entry=registry.lgbm_delay_30m,
        windows=windows,
        route_id_filter=route_id,
        stop_id_filter=stop_id,
        min_delay_seconds=min_delay,
    )


@router.get("/delay/end", response_model=DelayResponse)
async def predict_delay_end(
    request: Request,
    route_id: Optional[str] = Query(default=None),
    stop_id: Optional[str] = Query(default=None),
    min_delay: float = Query(default=0.0, ge=0),
) -> DelayResponse:
    registry = request.app.state.registry
    if registry.lgbm_delay_end is None:
        raise HTTPException(503, detail=f"delay/end not available: {registry.errors.get('lgbm_delay_end', 'not loaded')}")
    windows = await _get_windows(request)
    return await asyncio.to_thread(
        run_delays,
        entry=registry.lgbm_delay_end,
        windows=windows,
        route_id_filter=route_id,
        stop_id_filter=stop_id,
        min_delay_seconds=min_delay,
    )


# ── Delta (LightGBM binary) ───────────────────────────────────────────────────

@router.get("/delta/10m", response_model=DeltaResponse)
async def predict_delta_10m(
    request: Request,
    route_id: Optional[str] = Query(default=None),
    stop_id: Optional[str] = Query(default=None),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
) -> DeltaResponse:
    registry = request.app.state.registry
    if registry.delta_10m is None:
        raise HTTPException(503, detail=f"delta/10m not available: {registry.errors.get('delta_10m', 'not loaded')}")
    windows = await _get_windows(request)
    return await asyncio.to_thread(
        run_delta,
        entry=registry.delta_10m,
        windows=windows,
        horizon="delta_delay_10m",
        threshold=threshold,
        route_id_filter=route_id,
        stop_id_filter=stop_id,
    )


@router.get("/delta/20m", response_model=DeltaResponse)
async def predict_delta_20m(
    request: Request,
    route_id: Optional[str] = Query(default=None),
    stop_id: Optional[str] = Query(default=None),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
) -> DeltaResponse:
    registry = request.app.state.registry
    if registry.delta_20m is None:
        raise HTTPException(503, detail=f"delta/20m not available: {registry.errors.get('delta_20m', 'not loaded')}")
    windows = await _get_windows(request)
    return await asyncio.to_thread(
        run_delta,
        entry=registry.delta_20m,
        windows=windows,
        horizon="delta_delay_20m",
        threshold=threshold,
        route_id_filter=route_id,
        stop_id_filter=stop_id,
    )


@router.get("/delta/30m", response_model=DeltaResponse)
async def predict_delta_30m(
    request: Request,
    route_id: Optional[str] = Query(default=None),
    stop_id: Optional[str] = Query(default=None),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
) -> DeltaResponse:
    registry = request.app.state.registry
    if registry.delta_30m is None:
        raise HTTPException(503, detail=f"delta/30m not available: {registry.errors.get('delta_30m', 'not loaded')}")
    windows = await _get_windows(request)
    return await asyncio.to_thread(
        run_delta,
        entry=registry.delta_30m,
        windows=windows,
        horizon="delta_delay_30m",
        threshold=threshold,
        route_id_filter=route_id,
        stop_id_filter=stop_id,
    )


# ── Alerts (XGBoost) ─────────────────────────────────────────────────────────

@router.get("/alerts", response_model=AlertResponse)
async def predict_alerts(
    request: Request,
    route_id: Optional[str] = Query(default=None),
    min_prob: float = Query(default=0.0, ge=0.0, le=1.0),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
) -> AlertResponse:
    registry = request.app.state.registry
    if registry.alertas is None:
        raise HTTPException(503, detail=f"alerts not available: {registry.errors.get('alertas', 'not loaded')}")
    windows = await _get_windows(request)
    return await asyncio.to_thread(
        run_alerts,
        entry=registry.alertas,
        windows=windows,
        threshold=threshold,
        route_id_filter=route_id,
        min_prob=min_prob,
    )


# ── Per-train prediction ─────────────────────────────────────────────────────

@router.get("/train")
async def predict_train(
    request: Request,
    match_key: str = Query(..., description="match_key del tren (trip_id GTFS-RT)"),
) -> dict:
    """
    Per-train predictions.

    Construye las features del trip con get_trip_features(match_key), aplica
    encoding y ejecuta:
      - lgbm_delay_end   si scheduled_time_to_end < 30 min
      - lgbm_delay_30m   en caso contrario
      - delta_10m / delta_20m / delta_30m
    """
    registry = request.app.state.registry

    features = await asyncio.to_thread(get_trip_features, match_key)
    if features is None:
        raise HTTPException(404, detail=f"match_key '{match_key}' no encontrado en el feed RT")

    current_delay_s       = float(features.get("delay_seconds", 0.0))
    stops_to_end          = int(features.get("stops_to_end", 0))
    scheduled_time_to_end = float(features.get("scheduled_time_to_end", 0.0))
    route_id              = str(features.get("route_id", ""))
    stop_id               = str(features.get("stop_id", ""))

    # < 30 min: solo delay_end; >= 30 min: delay_30m + delay_end
    near_end = scheduled_time_to_end < 1800

    async def _safe(name: str, fn, **kw):
        try:
            return await asyncio.to_thread(fn, **kw)
        except Exception as exc:
            logger.warning("Train model %s failed for match_key %s: %s", name, match_key, exc)
            return None

    coros = {
        "delay_end": (
            _safe("delay_end", run_delay_single, entry=registry.lgbm_delay_end, features=features)
            if registry.lgbm_delay_end is not None else asyncio.sleep(0, result=None)
        ),
        "delay_30m": (
            _safe("delay_30m", run_delay_single, entry=registry.lgbm_delay_30m, features=features)
            if (not near_end and registry.lgbm_delay_30m is not None) else asyncio.sleep(0, result=None)
        ),
        "delta_10m": (
            _safe("delta_10m", run_delta_single, entry=registry.delta_10m, features=features)
            if registry.delta_10m is not None else asyncio.sleep(0, result=None)
        ),
        "delta_20m": (
            _safe("delta_20m", run_delta_single, entry=registry.delta_20m, features=features)
            if registry.delta_20m is not None else asyncio.sleep(0, result=None)
        ),
        "delta_30m": (
            _safe("delta_30m", run_delta_single, entry=registry.delta_30m, features=features)
            if registry.delta_30m is not None else asyncio.sleep(0, result=None)
        ),
    }

    results_list = await asyncio.gather(*coros.values())
    res = dict(zip(coros.keys(), results_list))

    def _fmt_delta(result):
        if result is None:
            return None
        prob, mejora = result
        return {"mejora_prob": round(prob, 4), "mejora_predicted": mejora}

    return {
        "match_key":               match_key,
        "route_id":                route_id,
        "stop_id":                 stop_id,
        "current_delay_s":         round(current_delay_s, 1),
        "stops_to_end":            stops_to_end,
        "scheduled_time_to_end_s": round(scheduled_time_to_end, 1),
        "delay_end_s":             round(res["delay_end"], 1) if res["delay_end"] is not None else None,
        "delay_30m_s":             round(res["delay_30m"], 1) if res["delay_30m"] is not None else None,
        "delta_10m":               _fmt_delta(res["delta_10m"]),
        "delta_20m":               _fmt_delta(res["delta_20m"]),
        "delta_30m":               _fmt_delta(res["delta_30m"]),
    }


# ── All ───────────────────────────────────────────────────────────────────────

@router.get("/all", response_model=AllPredictionsResponse)
async def predict_all(request: Request) -> AllPredictionsResponse:
    """Run all available models concurrently."""
    registry = request.app.state.registry
    windows = await _get_windows(request)
    predicted_at = datetime.now(timezone.utc).isoformat()
    errors: dict[str, str] = {}

    async def _safe(name: str, coro):
        try:
            return await coro
        except Exception as exc:
            logger.error("Model %s failed: %s", name, exc, exc_info=True)
            errors[name] = str(exc)
            return None

    def _thread(fn, **kw):
        return asyncio.to_thread(fn, **kw)

    tasks = {
        "propagation": (
            _safe("propagation", _thread(run_propagation, entry=registry.dcrnn, windows=windows, stations_meta=request.app.state.stations_meta))
            if registry.dcrnn else asyncio.sleep(0, result=None)
        ),
        "delay_30m": (
            _safe("delay_30m", _thread(run_delays, entry=registry.lgbm_delay_30m, windows=windows))
            if registry.lgbm_delay_30m else asyncio.sleep(0, result=None)
        ),
        "delay_end": (
            _safe("delay_end", _thread(run_delays, entry=registry.lgbm_delay_end, windows=windows))
            if registry.lgbm_delay_end else asyncio.sleep(0, result=None)
        ),
        "delta_10m": (
            _safe("delta_10m", _thread(run_delta, entry=registry.delta_10m, windows=windows, horizon="delta_delay_10m"))
            if registry.delta_10m else asyncio.sleep(0, result=None)
        ),
        "delta_20m": (
            _safe("delta_20m", _thread(run_delta, entry=registry.delta_20m, windows=windows, horizon="delta_delay_20m"))
            if registry.delta_20m else asyncio.sleep(0, result=None)
        ),
        "delta_30m": (
            _safe("delta_30m", _thread(run_delta, entry=registry.delta_30m, windows=windows, horizon="delta_delay_30m"))
            if registry.delta_30m else asyncio.sleep(0, result=None)
        ),
        "alerts": (
            _safe("alerts", _thread(run_alerts, entry=registry.alertas, windows=windows, threshold=settings.alert_threshold))
            if registry.alertas else asyncio.sleep(0, result=None)
        ),
    }

    for name, entry_attr in [
        ("propagation", "dcrnn"), ("delay_30m", "lgbm_delay_30m"), ("delay_end", "lgbm_delay_end"),
        ("delta_10m", "delta_10m"), ("delta_20m", "delta_20m"), ("delta_30m", "delta_30m"),
        ("alerts", "alertas"),
    ]:
        if getattr(registry, entry_attr) is None:
            errors[name] = registry.errors.get(entry_attr, "model not loaded")

    results = await asyncio.gather(*tasks.values())
    result_map = dict(zip(tasks.keys(), results))

    return AllPredictionsResponse(
        predicted_at=predicted_at,
        propagation=result_map["propagation"],
        delay_30m=result_map["delay_30m"],
        delay_end=result_map["delay_end"],
        delta_10m=result_map["delta_10m"],
        delta_20m=result_map["delta_20m"],
        delta_30m=result_map["delta_30m"],
        alerts=result_map["alerts"],
        errors=errors,
    )
