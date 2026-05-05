"""
Router del endpoint de salud del servidor Express-Bound.

Expone el endpoint GET /health que devuelve el estado actual del sistema:
el estado de carga de cada modelo ML y la disponibilidad de las ventanas
de datos cacheadas desde Google Drive.

El estado global es 'ok' si todos los modelos están cargados, o 'degraded'
si alguno no pudo cargarse (el servidor sigue operativo con los modelos disponibles).

Dependencias:
- app.config.settings: para obtener los nombres de artefacto de cada modelo.
- app.schemas.HealthResponse / ModelStatus / DataStatus: esquemas de respuesta.

Notas:
- cache.timestamp() devuelve time.monotonic(), no un Unix timestamp. Para calcular
  la hora de escritura en tiempo de reloj de pared se usa la diferencia con el
  monotónico actual y se resta al timestamp UTC presente.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.config import settings
from app.schemas import DataStatus, HealthResponse, ModelStatus

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """
    Devuelve el estado del sistema: modelos cargados y datos disponibles.

    Para cada modelo del registro, comprueba si hay error, si está cargado o si
    no se ha intentado cargar, y construye un ModelStatus con la información
    correspondiente. Para los datos, extrae las ventanas cacheadas y calcula
    el timestamp de escritura en UTC.

    Parámetros:
        request: Objeto Request de FastAPI (accede a app.state.registry y app.state.cache).

    Retorna:
        HealthResponse con status 'ok' o 'degraded', estado de cada modelo y datos.
    """
    registry = request.app.state.registry
    cache = request.app.state.cache

    def _status(key: str, entry, artifact: str) -> ModelStatus:
        """
        Construye el ModelStatus para un modelo concreto.

        Parámetros:
            key: Clave del modelo en registry.errors.
            entry: Entrada del modelo (None si no está cargado).
            artifact: Nombre del artefacto W&B configurado para este modelo.

        Retorna:
            ModelStatus con loaded=True si está cargado, o con el error si falló.
        """
        if key in registry.errors:
            return ModelStatus(loaded=False, artifact=artifact, error=registry.errors[key])
        if entry is None:
            return ModelStatus(loaded=False, artifact=artifact, error="Not loaded")
        return ModelStatus(loaded=True, artifact=artifact, loaded_at=entry.loaded_at)

    models = {
        "dcrnn":          _status("dcrnn",          registry.dcrnn,          settings.dcrnn_artifact),
        "lgbm_delay_30m": _status("lgbm_delay_30m", registry.lgbm_delay_30m, settings.lgbm_delay_30m_artifact),
        "lgbm_delay_end": _status("lgbm_delay_end", registry.lgbm_delay_end, settings.lgbm_delay_end_artifact),
        "delta_10m":      _status("delta_10m",      registry.delta_10m,      settings.delta_10m_artifact),
        "delta_20m":      _status("delta_20m",      registry.delta_20m,      settings.delta_20m_artifact),
        "delta_30m":      _status("delta_30m",      registry.delta_30m,      settings.delta_30m_artifact),
        "alertas":        _status("alertas",         registry.alertas,        settings.alertas_artifact),
    }

    cached = cache.get("windows")
    if cached is not None:
        ts = cache.timestamp("windows")
        oldest = str(cached[0]["merge_time"].min()) if cached else None
        newest = str(cached[-1]["merge_time"].max()) if cached else None
        # cache.timestamp() devuelve time.monotonic(), no un Unix timestamp.
        # Se convierte calculando el tiempo de reloj de pared en el momento de escritura:
        # now_wall - (mono_now - mono_ts).
        import time
        cached_at = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() - (time.monotonic() - ts),
            tz=timezone.utc,
        ).isoformat() if ts else None
        data_status = DataStatus(
            windows_available=len(cached),
            oldest_window=oldest,
            newest_window=newest,
            cached_at=cached_at,
        )
    else:
        data_status = DataStatus(windows_available=0)

    all_loaded = all(m.loaded for m in models.values())
    return HealthResponse(
        status="ok" if all_loaded else "degraded",
        models=models,
        data=data_status,
    )
