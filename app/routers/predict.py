"""
Router de los endpoints de predicción de la API Express-Bound.

Expone los endpoints GET /api/predict/* para ejecutar los modelos ML sobre
los datos en tiempo real descargados de Google Drive. Los endpoints disponibles son:

- /predict/current: retraso actual observado (sin modelo, solo datos).
- /predict/propagation: propagación de retraso a 10/20/30 min (DCRNN).
- /predict/delay/30m: retraso absoluto en 30 minutos (LightGBM).
- /predict/delay/end: retraso absoluto al final del recorrido (LightGBM).
- /predict/delta/10m|20m|30m: tendencia de retraso en cada horizonte (LightGBM).
- /predict/alerts: alertas de incidencia por línea y dirección (XGBoost).
- /predict/train: predicción individual para un tren concreto (match_key).
- /predict/all: todos los modelos ejecutados de forma concurrente.

Dependencias:
- app.data.drive.download_windows: descarga ventanas de Drive si no están en caché.
- app.models.*_infer: funciones de inferencia de cada modelo.
- src.ETL.pipelines.realtime.preprocess_realtime_lgbm.get_trip_features: extrae
  features de un tren concreto desde el feed GTFS-RT.
- app.schemas: esquemas Pydantic para las respuestas.

Notas:
- Las ventanas de Drive se cachean en app.state.cache con TTL configurado en settings.
- Todos los modelos pesados se ejecutan en hilos separados (asyncio.to_thread) para
  no bloquear el event loop de FastAPI.
- Si un modelo no está cargado, el endpoint devuelve HTTP 503 con el motivo del error.
"""

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
    Construye una fila de features de fallback desde la última ventana de Drive.

    Se usa cuando get_trip_features no puede obtener datos del feed GTFS-RT para
    un tren concreto. Busca en la última ventana una fila con el stop_id indicado;
    si no la encuentra, usa la media a nivel de ruta.

    Parámetros:
        windows: Lista de DataFrames de ventanas de Drive.
        route_id: Identificador de la línea del tren.
        stop_id: Identificador GTFS de la parada actual.

    Retorna:
        DataFrame de una fila con las features del fallback, o None si no hay datos.
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
        # Si no se encuentra el stop_id exacto, usar la media a nivel de ruta
        route_norm = route_id.strip().split("-")[0].split("_")[0]
        if "route_id" in df.columns:
            sub = df[df["route_id"].astype(str) == route_norm]
    if sub.empty:
        return None

    row = sub.iloc[0:1].copy()
    now = datetime.now(timezone.utc)
    row["merge_time"] = now
    for col in ("stops_to_end_mean", "scheduled_time_to_end_mean",
                "lagged_delay_1_mean", "lagged_delay_2_mean"):
        if col not in row.columns:
            row[col] = 0.0
    return row


async def _get_windows(request: Request) -> list:
    """
    Obtiene las ventanas de datos desde la caché o las descarga de Drive si han expirado.

    Comprueba primero la caché TTL de la aplicación; si no hay datos válidos, descarga
    las N ventanas más recientes de la carpeta de Drive configurada y las cachea.

    Parámetros:
        request: Objeto Request de FastAPI (accede a app.state.cache).

    Retorna:
        Lista de DataFrames con las ventanas de datos en tiempo real, ordenadas
        de más antigua a más reciente.
    """
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


@router.get("/current")
async def get_current_delay(
    request: Request,
    stop_id: Optional[str] = Query(default=None),
    route_id: Optional[str] = Query(default=None),
) -> dict:
    """
    Devuelve el retraso medio observado actualmente para un stop_id y/o route_id.

    No usa modelos ML: simplemente agrega el campo delay_seconds_mean de la
    última ventana de datos de Drive, filtrando por los parámetros indicados.

    Parámetros:
        request: Objeto Request de FastAPI.
        stop_id: Parada a consultar (exacto o base sin sufijo N/S, opcional).
        route_id: Línea a consultar (se normaliza eliminando sufijos, opcional).

    Retorna:
        Dict con 'stop_id' y 'delay_seconds' (float o None si no hay datos).
    """
    windows = await _get_windows(request)
    df = windows[-1].copy()

    if stop_id and "stop_id" in df.columns:
        base = df["stop_id"].astype(str).str.rstrip("NS")
        df = df[(df["stop_id"].astype(str) == stop_id) | (base == stop_id)]

    if route_id and "route_id" in df.columns:
        norm = route_id.strip().split("-")[0].split("_")[0]
        df = df[df["route_id"].astype(str) == norm]

    if df.empty or "delay_seconds_mean" not in df.columns:
        window_routes = windows[-1]["route_id"].astype(str).unique().tolist() if "route_id" in windows[-1].columns else []
        logger.warning(
            "predict/current: no data for stop_id=%r route_id=%r. "
            "Window has %d rows, routes=%s",
            stop_id, route_id, len(windows[-1]), sorted(window_routes)[:10],
        )
        return {"stop_id": stop_id, "delay_seconds": None}

    import numpy as np
    delay = float(np.clip(df["delay_seconds_mean"].mean(), 0, None))
    logger.debug("predict/current: stop_id=%r route_id=%r → delay=%.1fs (%d rows)", stop_id, route_id, delay, len(df))
    return {"stop_id": stop_id, "delay_seconds": delay}


@router.get("/propagation", response_model=PropagationResponse)
async def predict_propagation(
    request: Request,
    stop_id: Optional[str] = Query(default=None),
    route_id: Optional[str] = Query(default=None),
) -> PropagationResponse:
    """
    Ejecuta el modelo DCRNN y devuelve predicciones de propagación de retraso.

    Parámetros:
        request: Objeto Request de FastAPI.
        stop_id: Filtra la respuesta a una parada concreta (opcional).
        route_id: Filtra la respuesta a una línea concreta (opcional).

    Retorna:
        PropagationResponse con predicciones a 10, 20 y 30 minutos por parada.

    Lanza:
        HTTPException 503 si el modelo DCRNN no está disponible.
    """
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


@router.get("/delay/30m", response_model=DelayResponse)
async def predict_delay_30m(
    request: Request,
    route_id: Optional[str] = Query(default=None),
    stop_id: Optional[str] = Query(default=None),
    min_delay: float = Query(default=0.0, ge=0),
) -> DelayResponse:
    """
    Ejecuta el modelo LightGBM y devuelve predicciones de retraso en 30 minutos.

    Parámetros:
        request: Objeto Request de FastAPI.
        route_id: Filtra por línea (opcional).
        stop_id: Filtra por parada (opcional).
        min_delay: Umbral mínimo de retraso en segundos para incluir en la respuesta.

    Retorna:
        DelayResponse con las predicciones de retraso en segundos y minutos.

    Lanza:
        HTTPException 503 si el modelo no está disponible.
    """
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
    """
    Ejecuta el modelo LightGBM y devuelve predicciones de retraso al final del recorrido.

    Parámetros:
        request: Objeto Request de FastAPI.
        route_id: Filtra por línea (opcional).
        stop_id: Filtra por parada (opcional).
        min_delay: Umbral mínimo de retraso en segundos para incluir en la respuesta.

    Retorna:
        DelayResponse con las predicciones de retraso en segundos y minutos.

    Lanza:
        HTTPException 503 si el modelo no está disponible.
    """
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


@router.get("/delta/10m", response_model=DeltaResponse)
async def predict_delta_10m(
    request: Request,
    route_id: Optional[str] = Query(default=None),
    stop_id: Optional[str] = Query(default=None),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
) -> DeltaResponse:
    """
    Ejecuta el modelo delta y devuelve predicciones de tendencia de retraso a 10 minutos.

    Parámetros:
        request: Objeto Request de FastAPI.
        route_id: Filtra por línea (opcional).
        stop_id: Filtra por parada (opcional).
        threshold: Umbral de clasificación [0,1]. Si es None, usa el del artefacto.

    Retorna:
        DeltaResponse con la probabilidad de mejora y la clasificación por parada.

    Lanza:
        HTTPException 503 si el modelo no está disponible.
    """
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
    """
    Ejecuta el modelo delta y devuelve predicciones de tendencia de retraso a 20 minutos.

    Parámetros:
        request: Objeto Request de FastAPI.
        route_id: Filtra por línea (opcional).
        stop_id: Filtra por parada (opcional).
        threshold: Umbral de clasificación [0,1]. Si es None, usa el del artefacto.

    Retorna:
        DeltaResponse con la probabilidad de mejora y la clasificación por parada.

    Lanza:
        HTTPException 503 si el modelo no está disponible.
    """
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
    """
    Ejecuta el modelo delta y devuelve predicciones de tendencia de retraso a 30 minutos.

    Parámetros:
        request: Objeto Request de FastAPI.
        route_id: Filtra por línea (opcional).
        stop_id: Filtra por parada (opcional).
        threshold: Umbral de clasificación [0,1]. Si es None, usa el del artefacto.

    Retorna:
        DeltaResponse con la probabilidad de mejora y la clasificación por parada.

    Lanza:
        HTTPException 503 si el modelo no está disponible.
    """
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


@router.get("/alerts", response_model=AlertResponse)
async def predict_alerts(
    request: Request,
    route_id: Optional[str] = Query(default=None),
    min_prob: float = Query(default=0.0, ge=0.0, le=1.0),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
) -> AlertResponse:
    """
    Ejecuta el modelo XGBoost de alertas y devuelve predicciones por línea y dirección.

    Parámetros:
        request: Objeto Request de FastAPI.
        route_id: Filtra la respuesta a una línea concreta (opcional).
        min_prob: Probabilidad mínima para incluir una predicción en la respuesta.
        threshold: Umbral de clasificación [0,1]. Si es None, usa el del artefacto.

    Retorna:
        AlertResponse con la probabilidad de alerta y la clasificación por línea.

    Lanza:
        HTTPException 503 si el modelo no está disponible.
    """
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


@router.get("/train")
async def predict_train(
    request: Request,
    match_key: str = Query(..., description="match_key del tren (trip_id GTFS-RT)"),
) -> dict:
    """
    Ejecuta predicciones individuales para un tren concreto identificado por su match_key.

    Obtiene las features del viaje con get_trip_features(match_key) y ejecuta en paralelo:
    - lgbm_delay_end: si el tiempo restante es menor de 30 minutos.
    - lgbm_delay_30m: si el tiempo restante es mayor o igual a 30 minutos.
    - delta_10m, delta_20m, delta_30m: en todos los casos.

    Si get_trip_features devuelve None (tren no encontrado, ya terminado o no programado),
    devuelve una respuesta con predictable=False y el motivo clasificado.

    Parámetros:
        request: Objeto Request de FastAPI.
        match_key: Identificador del viaje en el feed GTFS-RT (trip_id canónico).

    Retorna:
        Dict con route_id, stop_id, retraso actual, paradas restantes, tiempo restante
        y las predicciones de cada modelo disponible (None si el modelo no está cargado
        o falló para este viaje).
    """
    registry = request.app.state.registry

    features, reason = await asyncio.to_thread(get_trip_features, match_key)
    if features is None:
        if "no encontrado en el feed RT" in reason:
            reason_type = "temporal"
        elif "última parada" in reason:
            reason_type = "ended"
        else:
            reason_type = "unscheduled"
        return {
            "match_key":   match_key,
            "predictable": False,
            "reason":      reason,
            "reason_type": reason_type,
        }

    current_delay_s       = float(features.get("delay_seconds", 0.0))
    stops_to_end          = int(features.get("stops_to_end", 0))
    scheduled_time_to_end = float(features.get("scheduled_time_to_end", 0.0))
    route_id              = str(features.get("route_id", ""))
    stop_id               = str(features.get("stop_id", ""))

    # Si quedan menos de 30 minutos, solo se predice delay_end; en caso contrario ambos modelos
    near_end = scheduled_time_to_end < 1800

    async def _safe(name: str, fn, **kw):
        """
        Ejecuta una función de inferencia en un hilo y captura errores sin propagar.

        Parámetros:
            name: Nombre del modelo (para el log de warning).
            fn: Función de inferencia a ejecutar.
            **kw: Argumentos de palabra clave para la función.

        Retorna:
            El resultado de fn o None si ocurre alguna excepción.
        """
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
        """
        Formatea el resultado de un modelo delta para la respuesta JSON.

        Parámetros:
            result: Tupla (prob, mejora_predicted) o None si el modelo no produjo resultado.

        Retorna:
            Dict con 'mejora_prob' y 'mejora_predicted', o None si result es None.
        """
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


@router.get("/all", response_model=AllPredictionsResponse)
async def predict_all(request: Request) -> AllPredictionsResponse:
    """
    Ejecuta todos los modelos disponibles de forma concurrente y devuelve sus resultados.

    Los modelos que no estén cargados producen None en el campo correspondiente y
    su error se registra en el campo 'errors' de la respuesta. Los modelos cargados
    se ejecutan en paralelo para minimizar la latencia total.

    Parámetros:
        request: Objeto Request de FastAPI.

    Retorna:
        AllPredictionsResponse con los resultados de todos los modelos disponibles,
        el timestamp de predicción y un dict de errores para los modelos que fallaron.
    """
    registry = request.app.state.registry
    windows = await _get_windows(request)
    predicted_at = datetime.now(timezone.utc).isoformat()
    errors: dict[str, str] = {}

    async def _safe(name: str, coro):
        """
        Ejecuta una corutina de inferencia capturando errores sin propagar.

        Parámetros:
            name: Nombre del modelo (para el log de error y el dict errors).
            coro: Corutina a ejecutar.

        Retorna:
            El resultado de la corutina o None si ocurre alguna excepción.
        """
        try:
            return await coro
        except Exception as exc:
            logger.error("Model %s failed: %s", name, exc, exc_info=True)
            errors[name] = str(exc)
            return None

    def _thread(fn, **kw):
        """Envuelve una función bloqueante para ejecutarla en un hilo del executor."""
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
