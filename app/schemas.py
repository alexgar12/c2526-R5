"""
Modelos Pydantic para las respuestas de la API Express-Bound.

Define los esquemas de validación y serialización utilizados por los endpoints
REST de predicción y salud del servidor. Cada modelo Pydantic garantiza que
los datos enviados al cliente tienen el formato y los tipos correctos.

Estructura:
- ModelStatus / DataStatus / HealthResponse: estado del sistema (/health).
- PropagationPrediction / PropagationResponse: propagación de retrasos (DCRNN).
- DelayPrediction / DelayResponse: predicción de retraso absoluto (LightGBM).
- DeltaPrediction / DeltaResponse: predicción de mejora/empeoramiento del retraso.
- AlertPrediction / AlertResponse: alertas de incidencia por línea (XGBoost).
- AllPredictionsResponse: respuesta combinada con todos los modelos.

Dependencias:
- pydantic (BaseModel) para validación automática de tipos.
"""

from pydantic import BaseModel
from typing import Optional


class ModelStatus(BaseModel):
    """Estado de carga de un modelo ML individual."""

    loaded: bool
    artifact: Optional[str] = None
    error: Optional[str] = None
    loaded_at: Optional[str] = None


class DataStatus(BaseModel):
    """Estado de las ventanas de datos cacheadas desde Google Drive."""

    windows_available: int
    oldest_window: Optional[str] = None
    newest_window: Optional[str] = None
    cached_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Respuesta completa del endpoint /health con estado de modelos y datos."""

    status: str
    models: dict[str, ModelStatus]
    data: DataStatus


class PropagationPrediction(BaseModel):
    """Predicción de retraso propagado para una parada concreta en tres horizontes temporales."""

    stop_id: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    delay_10m: float
    delay_20m: float
    delay_30m: float


class PropagationResponse(BaseModel):
    """Respuesta del modelo DCRNN con predicciones de propagación de retraso para todas las paradas."""

    predicted_at: str
    horizon_minutes: list[int] = [10, 20, 30]
    n_stations: int
    predictions: list[PropagationPrediction]


class DelayPrediction(BaseModel):
    """Predicción de retraso absoluto (en segundos y minutos) para una parada y dirección."""

    stop_id: str
    route_id: str
    direction: str
    delay_seconds: float
    delay_minutes: float


class DelayResponse(BaseModel):
    """Respuesta del modelo LightGBM de retraso con predicciones para todas las paradas."""

    predicted_at: str
    target: str               # "target_delay_30m" | "target_delay_end"
    n_stops: int
    predictions: list[DelayPrediction]


class DeltaPrediction(BaseModel):
    """Predicción de si el retraso mejorará o empeorará en un horizonte temporal concreto."""

    stop_id: str
    route_id: str
    direction: str
    mejora_prob: float        # Probabilidad de que el retraso mejore
    mejora_predicted: bool    # True = mejora predicha, False = empeoramiento predicho


class DeltaResponse(BaseModel):
    """Respuesta del modelo Delta con predicciones de tendencia de retraso para todas las paradas."""

    predicted_at: str
    horizon: str              # "delta_delay_10m" | "delta_delay_20m" | "delta_delay_30m"
    threshold_used: float
    n_stops: int
    predictions: list[DeltaPrediction]


class AlertPrediction(BaseModel):
    """Predicción de alerta de incidencia para una línea y dirección concretas."""

    route_id: str
    direction: str
    alert_probability: float
    alert_predicted: bool
    pct_stops_delayed: Optional[float] = None
    delay_mean_seconds: Optional[float] = None


class AlertResponse(BaseModel):
    """Respuesta del modelo XGBoost de alertas con predicciones por línea y dirección."""

    predicted_at: str
    threshold_used: float
    n_lines: int
    predictions: list[AlertPrediction]


class AllPredictionsResponse(BaseModel):
    """
    Respuesta agregada del endpoint /predict/all con la salida de todos los modelos disponibles.

    Los campos de predicción son opcionales: si un modelo no está cargado o falla,
    el campo correspondiente es None y el error se registra en el dict 'errors'.
    """

    predicted_at: str
    propagation: Optional[PropagationResponse] = None
    delay_30m: Optional[DelayResponse] = None
    delay_end: Optional[DelayResponse] = None
    delta_10m: Optional[DeltaResponse] = None
    delta_20m: Optional[DeltaResponse] = None
    delta_30m: Optional[DeltaResponse] = None
    alerts: Optional[AlertResponse] = None
    errors: dict[str, str] = {}
