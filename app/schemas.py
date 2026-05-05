from pydantic import BaseModel
from typing import Optional


class ModelStatus(BaseModel):
    loaded: bool
    artifact: Optional[str] = None
    error: Optional[str] = None
    loaded_at: Optional[str] = None


class DataStatus(BaseModel):
    windows_available: int
    oldest_window: Optional[str] = None
    newest_window: Optional[str] = None
    cached_at: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    models: dict[str, ModelStatus]
    data: DataStatus



class PropagationPrediction(BaseModel):
    stop_id: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    delay_10m: float
    delay_20m: float
    delay_30m: float


class PropagationResponse(BaseModel):
    predicted_at: str
    horizon_minutes: list[int] = [10, 20, 30]
    n_stations: int
    predictions: list[PropagationPrediction]



class DelayPrediction(BaseModel):
    stop_id: str
    route_id: str
    direction: str
    delay_seconds: float
    delay_minutes: float


class DelayResponse(BaseModel):
    predicted_at: str
    target: str               # "target_delay_30m" | "target_delay_end"
    n_stops: int
    predictions: list[DelayPrediction]



class DeltaPrediction(BaseModel):
    stop_id: str
    route_id: str
    direction: str
    mejora_prob: float        # P(retraso mejora)
    mejora_predicted: bool    # True = mejora, False = empeora


class DeltaResponse(BaseModel):
    predicted_at: str
    horizon: str              # "delta_delay_10m" | "delta_delay_20m" | "delta_delay_30m"
    threshold_used: float
    n_stops: int
    predictions: list[DeltaPrediction]



class AlertPrediction(BaseModel):
    route_id: str
    direction: str
    alert_probability: float
    alert_predicted: bool
    pct_stops_delayed: Optional[float] = None
    delay_mean_seconds: Optional[float] = None


class AlertResponse(BaseModel):
    predicted_at: str
    threshold_used: float
    n_lines: int
    predictions: list[AlertPrediction]



class AllPredictionsResponse(BaseModel):
    predicted_at: str
    propagation: Optional[PropagationResponse] = None
    delay_30m: Optional[DelayResponse] = None
    delay_end: Optional[DelayResponse] = None
    delta_10m: Optional[DeltaResponse] = None
    delta_20m: Optional[DeltaResponse] = None
    delta_30m: Optional[DeltaResponse] = None
    alerts: Optional[AlertResponse] = None
    errors: dict[str, str] = {}
