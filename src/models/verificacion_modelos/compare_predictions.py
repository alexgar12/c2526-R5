"""
Comparador de predicciones en tiempo real.

Compara predicciones pasadas con valores actuales:
- Ejecuta predicciones cada 10 minutos
- Compara predicción de t-10 para 10min con valor actual
- Compara predicción de t-20 para 20min con valor actual  
- Compara predicción de t-30 para 30min con valor actual

Uso:
    # Ejecutar continuamente (cada 10 minutos)
    uv run python src/models/verificacion_modelos/compare_predictions.py

    # O una sola vez para probar
    uv run python src/models/verificacion_modelos/compare_predictions.py --once
"""
import argparse
import asyncio
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_INTERVAL = 10  # minutes


def get_endpoints(base_url: str) -> dict:
    return {
        "current": f"{base_url}/api/predict/current",
        "propagation": f"{base_url}/api/predict/propagation",
        "delay_30m": f"{base_url}/api/predict/delay/30m",
        "delay_end": f"{base_url}/api/predict/delay/end",
        "alerts": f"{base_url}/api/predict/alerts",
    }


async def call_endpoint(session: aiohttp.ClientSession, url: str, timeout: int = 60) -> dict:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                return {"error": f"http_{resp.status}"}
            return await resp.json()
    except Exception as e:
        return {"error": str(e)}


async def call_all_predictions(session: aiohttp.ClientSession, base_url: str) -> dict:
    endpoints = get_endpoints(base_url)
    tasks = {name: call_endpoint(session, url) for name, url in endpoints.items()}
    results = await asyncio.gather(*tasks.values())
    result = dict(zip(endpoints.keys(), results))
    
    # Debug: mostrar qué devuelve cada endpoint
    print("\n🔍 Debug - Respuestas de la API:")
    for name, resp in result.items():
        if isinstance(resp, dict) and "error" in resp:
            print(f"   {name}: ERROR - {resp}")
        elif isinstance(resp, dict):
            # Mostrar resumen
            keys = list(resp.keys()) if isinstance(resp, dict) else "list"
            print(f"   {name}: OK - keys: {keys}")
        else:
            print(f"   {name}: {type(resp)}")
    
    return result


class PredictionHistory:
    """Guarda historial de predicciones para comparar."""
    
    def __init__(self):
        # {model_name: [{timestamp, predictions}, ...]}
        self.history = defaultdict(list)
        # Cuántos ciclos mantener (30min = 3 ciclos de 10min)
        self.max_cycles = 4
    
    def add(self, predictions: dict):
        """Añade un ciclo de predicciones."""
        timestamp = datetime.now(timezone.utc)
        
        for name, pred in predictions.items():
            if isinstance(pred, dict) and "error" not in pred:
                self.history[name].append({
                    "timestamp": timestamp,
                    "data": pred
                })
        
        # Debug: mostrar qué se guardó
        print(f"\n💾 Guardado en historial:")
        for name in self.history:
            if self.history[name]:
                latest = self.history[name][-1]
                age = (timestamp - latest["timestamp"]).total_seconds()
                print(f"   {name}: {len(self.history[name])} entradas, última hace {age:.0f}s")
        
        # Limpiar ciclos antiguos
        for name in self.history:
            if len(self.history[name]) > self.max_cycles:
                self.history[name] = self.history[name][-self.max_cycles:]
    
    def get_prediction_at(self, model_name: str, minutes_ago: int) -> dict | None:
        """Obtiene la predicción de hace X minutos."""
        if model_name not in self.history:
            return None
        
        target_time = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        
        for entry in reversed(self.history[model_name]):
            diff = abs((entry["timestamp"] - target_time).total_seconds())
            if diff <= 120:  # Within 2 minutes tolerance
                return entry["data"]
        return None


def extract_delay_value(prediction: dict) -> float | None:
    """Extrae el valor de delay de una predicción."""
    if not prediction:
        return None
    
    predictions = prediction.get("predictions", [])
    if predictions and isinstance(predictions, list):
        delays = [p.get("delay_seconds", 0) for p in predictions if p.get("delay_seconds") is not None]
        if delays:
            return sum(delays) / len(delays)
    
    return prediction.get("delay_seconds")


def extract_delta_value(prediction: dict) -> float | None:
    """Extrae el valor de probabilidad delta de una predicción."""
    if not prediction:
        return None
    
    predictions = prediction.get("predictions", [])
    if predictions and isinstance(predictions, list):
        probs = [p.get("mejora_prob", 0) for p in predictions if p.get("mejora_prob") is not None]
        if probs:
            return sum(probs) / len(probs)
    
    return prediction.get("mejora_prob")


def extract_alert_value(prediction: dict) -> float | None:
    """Extrae el valor de probabilidad de alerta."""
    if not prediction:
        return None
    
    predictions = prediction.get("predictions", [])
    if predictions and isinstance(predictions, list):
        probs = [p.get("alerta_prob", 0) for p in predictions if p.get("alerta_prob") is not None]
        if probs:
            return sum(probs) / len(probs)
    
    return prediction.get("alerta_prob")


def station_key(pred: dict) -> tuple[str, str, str]:
    """Crea una clave única para una predicción de estación."""
    stop_id = str(pred.get("stop_id", ""))
    lat = str(pred.get("lat", ""))
    lon = str(pred.get("lon", ""))
    return (stop_id, lat, lon)


def select_top_propagation_stations(prediction: dict, count: int = 5) -> list[dict]:
    """Selecciona las estaciones con mayor delay previsto en propagación."""
    if not prediction:
        return []
    predictions = prediction.get("predictions", [])
    if not isinstance(predictions, list):
        return []

    def score(item: dict) -> float:
        return float(item.get("delay_30m") or item.get("delay_20m") or item.get("delay_10m") or 0)

    return sorted(predictions, key=score, reverse=True)[:count]


def compare_propagation_by_station(old_pred: dict, new_pred: dict, count: int = 5) -> list[dict]:
    """Compara la propagación para un conjunto de estaciones seleccionadas."""
    if not old_pred or not new_pred:
        return []

    old_map = {station_key(p): p for p in old_pred.get("predictions", []) if isinstance(p, dict)}
    selected = select_top_propagation_stations(new_pred, count)
    results = []

    for current_station in selected:
        key = station_key(current_station)
        old_station = old_map.get(key)
        if not old_station:
            continue

        station_result = {
            "stop_id": current_station.get("stop_id"),
            "lat": current_station.get("lat"),
            "lon": current_station.get("lon"),
            "horizons": []
        }

        for horizon in (10, 20, 30):
            key_name = f"delay_{horizon}m"
            old_val = old_station.get(key_name)
            new_val = current_station.get(key_name)
            if old_val is None or new_val is None:
                continue
            station_result["horizons"].append({
                "horizon": horizon,
                "predicted_10m_ago": old_val,
                "current": new_val,
                "diff": new_val - old_val,
            })

        if station_result["horizons"]:
            results.append(station_result)

    return results


def compare_predictions(history: PredictionHistory):
    """Compara predicciones pasadas con valores actuales."""
    now = datetime.now(timezone.utc)
    
    print("\n" + "="*70)
    print(f"COMPARACIÓN DE PREDICCIONES - {now.isoformat()}")
    print("="*70)
    
    results = {}
    
    # Comparar delay_30m (predicción hace 10min vs actual)
    pred_10m_ago = history.get_prediction_at("delay_30m", 10)
    current = history.get_prediction_at("delay_30m", 0)
    
    if pred_10m_ago and current:
        old_val = extract_delay_value(pred_10m_ago)
        new_val = extract_delay_value(current)
        if old_val is not None and new_val is not None:
            diff = new_val - old_val
            print(f"\nDELAY 30m (predicción hace 10min vs actual):")
            print(f"   Predicho hace 10min: {old_val:.1f} seg ({old_val/60:.1f} min)")
            print(f"   Valor actual:        {new_val:.1f} seg ({new_val/60:.1f} min)")
            print(f"   Diferencia:          {diff:+.1f} seg ({diff/60:+.1f} min)")
            results["delay_30m"] = {"predicted_10m_ago": old_val, "current": new_val, "diff": diff}
    
    # Comparar delay_end
    pred_10m_ago_end = history.get_prediction_at("delay_end", 10)
    current_end = history.get_prediction_at("delay_end", 0)
    
    if pred_10m_ago_end and current_end:
        old_val = extract_delay_value(pred_10m_ago_end)
        new_val = extract_delay_value(current_end)
        if old_val is not None and new_val is not None:
            diff = new_val - old_val
            print(f"\nDELAY END (predicción hace 10min vs actual):")
            print(f"   Predicho hace 10min: {old_val:.1f} seg ({old_val/60:.1f} min)")
            print(f"   Valor actual:        {new_val:.1f} seg ({new_val/60:.1f} min)")
            print(f"   Diferencia:          {diff:+.1f} seg ({diff/60:+.1f} min)")
            results["delay_end"] = {"predicted_10m_ago": old_val, "current": new_val, "diff": diff}
    
    # Comparar alerts
    pred_10m_ago_alerts = history.get_prediction_at("alerts", 10)
    current_alerts = history.get_prediction_at("alerts", 0)
    
    if pred_10m_ago_alerts and current_alerts:
        old_val = extract_alert_value(pred_10m_ago_alerts)
        new_val = extract_alert_value(current_alerts)
        if old_val is not None and new_val is not None:
            diff = new_val - old_val
            print(f"\nALERTS (predicción hace 10min vs actual):")
            print(f"   Predicho hace 10min: {old_val:.3f}")
            print(f"   Valor actual:        {new_val:.3f}")
            print(f"   Diferencia:          {diff:+.3f}")
            results["alerts"] = {"predicted_10m_ago": old_val, "current": new_val, "diff": diff}

    # Comparar propagación por estaciones
    pred_10m_ago_prop = history.get_prediction_at("propagation", 10)
    current_prop = history.get_prediction_at("propagation", 0)
    if pred_10m_ago_prop and current_prop:
        print(f"\nPROPAGACIÓN (selección de estaciones):")
        station_comparisons = compare_propagation_by_station(pred_10m_ago_prop, current_prop, count=5)
        if station_comparisons:
            for station in station_comparisons:
                stop_id = station.get("stop_id")
                lat = station.get("lat")
                lon = station.get("lon")
                header = f"   Estación {stop_id}" if stop_id else "   Estación"
                if lat is not None and lon is not None:
                    header += f" ({lat:.6f}, {lon:.6f})"
                print(header)
                for item in station["horizons"]:
                    diff = item["diff"]
                    print(f"      {item['horizon']}m: predicho hace 10min = {item['predicted_10m_ago']:.1f} seg, actual = {item['current']:.1f} seg, diff = {diff:+.1f} seg")
            results["propagation"] = station_comparisons
        else:
            print("   No hay coincidencias de estaciones entre predicciones antiguas y actuales.")
    
    # Mostrar historial disponible
    print(f"\nHistorial disponible:")
    for name, entries in history.history.items():
        if entries:
            ages = [(now - e["timestamp"]).total_seconds() / 60 for e in entries]
            print(f"   {name}: {len(entries)} entradas, {ages[0]:.0f}-{ages[-1]:.0f} min atrás")
    
    print("\n" + "="*70)
    
    return results


async def run_comparison(interval_minutes: int, base_url: str, once: bool):
    logger.info(f"Conectando a API: {base_url}")
    logger.info(f"Intervalo: {interval_minutes} minutos")
    
    history = PredictionHistory()
    
    async with aiohttp.ClientSession() as session:
        if once:
            # Una sola ejecución
            predictions = await call_all_predictions(session, base_url)
            history.add(predictions)
            
            print("\nPredicciones actuales:")
            for name, pred in predictions.items():
                print(f"\n--- {name.upper()} ---")
                print(json.dumps(pred, indent=2, default=str))
            
            print("\nEjecuta de nuevo después de 10 minutos para ver comparaciones")
        else:
            # Ciclo continuo
            while True:
                logger.info("Obteniendo predicciones...")
                predictions = await call_all_predictions(session, base_url)
                history.add(predictions)
                
                # Mostrar predicciones actuales
                print("\n" + "="*70)
                print(f"PREDICCIONES ACTUALES - {datetime.now(timezone.utc).isoformat()}")
                print("="*70)
                for name, pred in predictions.items():
                    if isinstance(pred, dict) and "error" not in pred:
                        print(f"\n--- {name.upper()} ---")
                        if name == "delay_30m":
                            val = extract_delay_value(pred)
                            print(f"   Delay promedio: {val:.1f} seg" if val else "   N/A")
                        elif name == "propagation":
                            selected = select_top_propagation_stations(pred, count=5)
                            if selected:
                                print("   Ejemplo estaciones con mayor propagación:")
                                for station in selected:
                                    stop_id = station.get("stop_id")
                                    lat = station.get("lat")
                                    lon = station.get("lon")
                                    description = f"      - {stop_id or 'sin_stop_id'}"
                                    if lat is not None and lon is not None:
                                        description += f" ({lat:.6f}, {lon:.6f})"
                                    print(description)
                                    for horizon in (10, 20, 30):
                                        val = station.get(f"delay_{horizon}m")
                                        if val is not None:
                                            print(f"         {horizon}m: {val:.1f} seg")
                            else:
                                print("   N/A")
                        elif name.startswith("delta_"):
                            val = extract_delta_value(pred)
                            print(f"   Probabilidad mejora: {val:.3f}" if val else "   N/A")
                        elif name == "alerts":
                            val = extract_alert_value(pred)
                            print(f"   Probabilidad alerta: {val:.3f}" if val else "   N/A")
                
                # Comparar con historial
                compare_predictions(history)
                
                logger.info(f"Esperando {interval_minutes} minutos...")
                await asyncio.sleep(interval_minutes * 60)


def main():
    parser = argparse.ArgumentParser(description="Comparador de predicciones en tiempo real")
    parser.add_argument("--once", action="store_true", help="Ejecutar solo una vez")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="Intervalo en minutos")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="URL de la API")
    args = parser.parse_args()
    
    asyncio.run(run_comparison(args.interval, args.api_url, args.once))


if __name__ == "__main__":
    main()
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",

logger = logging.getLogger(__name__)

# Configuración por defecto
DEFAULT_OUTPUT_DIR = Path("./verificacion")
DATA_TEMPLATE = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"


def get_output_dir() -> Path:
    """Obtiene el directorio de salida para los resultados."""
    output_dir = Path(os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    return output_dir


def load_predictions(output_dir: Path, prediction_type: str) -> list[dict]:
    """Carga todas las predicciones guardadas de un tipo."""
    type_dir = output_dir / prediction_type
    if not type_dir.exists():
        logger.warning(f"Directorio no encontrado: {type_dir}")
        return []

    predictions = []
    for filepath in sorted(type_dir.glob("*.json")):
        try:
            with open(filepath) as f:
                data = json.load(f)
                predictions.append(data)
        except Exception as e:
            logger.warning(f"Error cargando {filepath}: {e}")

    logger.info(f"Cargadas {len(predictions)} predicciones de {prediction_type}")
    return predictions


def load_predictions_batch(output_dir: Path) -> list[dict]:
    """Carga todas las predicciones por lotes."""
    batch_dir = output_dir / "batches"
    if not batch_dir.exists():
        return []

    batches = []
    for filepath in sorted(batch_dir.glob("*.json")):
        try:
            with open(filepath) as f:
                data = json.load(f)
                batches.append(data)
        except Exception as e:
            logger.warning(f"Error cargando {filepath}: {e}")

    logger.info(f"Cargados {len(batches)} lotes de predicciones")
    return batches


def load_actual_data(
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Carga los datos reales del dataset histórico de MinIO."""
    access_key = os.environ["MINIO_ACCESS_KEY"]
    secret_key = os.environ["MINIO_SECRET_KEY"]

    dfs = []
    current = start_date.replace(day=1)  # Empezar desde el primer día del mes

    while current <= end_date:
        year = current.year
        month = current.month
        path = DATA_TEMPLATE.format(year=year, month=month)

        try:
            df = download_df_parquet(access_key, secret_key, path)
            # Filtrar por rango de fechas
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            dfs.append(df)
            logger.info(f"Cargados datos de {year}-{month:02d}: {len(df)} filas")
        except Exception as e:
            logger.warning(f"No se pudieron cargar datos de {year}-{month:02d}: {e}")

        # Avanzar al siguiente mes
        if month == 12:
            current = datetime(year + 1, 1, 1)
        else:
            current = datetime(year, month + 1, 1)

    if not dfs:
        logger.warning("No se cargaron datos reales")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def match_prediction_with_actual(
    prediction: dict,
    actual_data: pd.DataFrame,
    prediction_type: str,
) -> Optional[dict]:
    """
    Empareja una predicción con el valor real correspondiente.

    Args:
        prediction: Predicción guardada con timestamp y datos
        actual_data: DataFrame con datos reales
        prediction_type: Tipo de predicción

    Returns:
        Dict con predicción y valor real emparejados, o None si no hay match
    """
    pred_time = datetime.fromisoformat(prediction["timestamp"])
    data = prediction.get("data", {})

    # Extraer stop_id y route_id de la predicción
    predictions_list = data.get("predictions", [])
    if not predictions_list:
        return None

    matched = []
    for pred in predictions_list:
        stop_id = pred.get("stop_id")
        route_id = pred.get("route_id")
        direction = pred.get("direction")

        # Buscar en datos reales
        mask = pd.Series([True] * len(actual_data))
        if stop_id:
            mask &= actual_data["stop_id"].astype(str) == stop_id
        if route_id:
            mask &= actual_data["route_id"].astype(str) == route_id
        if direction:
            mask &= actual_data["direction"].astype(str) == direction

        # Filtrar por tiempo cercano (dentro de 15 minutos)
        if "merge_time" in actual_data.columns:
            actual_data["merge_time"] = pd.to_datetime(actual_data["merge_time"])
            time_diff = abs((actual_data["merge_time"] - pred_time).dt.total_seconds())
            mask &= time_diff <= 900  # 15 minutos

        matches = actual_data[mask]
        if not matches.empty:
            # Obtener el target correspondiente
            target_col = None
            if prediction_type == "delay_30m":
                target_col = "target_delay_30m"
            elif prediction_type == "delay_end":
                target_col = "target_delay_end"
            elif prediction_type.startswith("delta_"):
                horizon = prediction_type.split("_")[1]
                target_col = f"delta_delay_{horizon}"

            if target_col and target_col in matches.columns:
                actual_value = matches[target_col].iloc[0]
                matched.append({
                    "predicted": pred.get("delay_seconds") or pred.get("mejora_prob"),
                    "actual": actual_value,
                    "stop_id": stop_id,
                    "route_id": route_id,
                    "direction": direction,
                    "pred_time": pred_time.isoformat(),
                })

    return matched if matched else None


def compute_delay_metrics(
    predictions: list[dict],
    actual_data: pd.DataFrame,
) -> dict[str, Any]:
    """Calcula métricas para predicciones de retraso (regresión)."""
    y_true = []
    y_pred = []

    for pred in predictions:
        matched = match_prediction_with_actual(pred, actual_data, "delay")
        if matched:
            for m in matched:
                y_true.append(m["actual"])
                y_pred.append(m["predicted"])

    if not y_true:
        return {"error": "No hay suficientes datos para calcular métricas"}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "n_samples": len(y_true),
        "mae_seconds": round(mae, 2),
        "mae_minutes": round(mae / 60, 2),
        "rmse_seconds": round(rmse, 2),
        "r2": round(r2, 4),
    }


def compute_delta_metrics(
    predictions: list[dict],
    actual_data: pd.DataFrame,
    horizon: str = "30m",
) -> dict[str, Any]:
    """Calcula métricas para predicciones delta (clasificación binaria)."""
    y_true = []
    y_pred = []
    y_prob = []

    prediction_type = f"delta_{horizon}"
    for pred in predictions:
        matched = match_prediction_with_actual(pred, actual_data, prediction_type)
        if matched:
            for m in matched:
                # Para delta: actual > 0 significa mejora
                y_true.append(1 if m["actual"] > 0 else 0)
                y_pred.append(1 if m["predicted"] > 0.5 else 0)
                y_prob.append(m["predicted"])

    if not y_true:
        return {"error": "No hay suficientes datos para calcular métricas"}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ap = average_precision_score(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    return {
        "n_samples": len(y_true),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "average_precision": round(ap, 4),
        "auc_roc": round(auc, 4),
    }


def compute_alerts_metrics(
    predictions: list[dict],
    actual_data: pd.DataFrame,
) -> dict[str, Any]:
    """Calcula métricas para predicciones de alertas."""
    # Similar a delta pero con alert_in_next_15m/30m
    y_true = []
    y_pred = []
    y_prob = []

    for pred in predictions:
        matched = match_prediction_with_actual(pred, actual_data, "alerts")
        if matched:
            for m in matched:
                # Buscar si hay alerta en los próximos 30 min
                if "alert_in_next_30m" in actual_data.columns:
                    y_true.append(1 if m["actual"] > 0 else 0)
                    y_pred.append(1 if m["predicted"] > 0.5 else 0)
                    y_prob.append(m["predicted"])

    if not y_true:
        return {"error": "No hay suficientes datos para calcular métricas"}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ap = average_precision_score(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    return {
        "n_samples": len(y_true),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "average_precision": round(ap, 4),
        "auc_roc": round(auc, 4),
    }


def generate_report(
    output_dir: Path,
    start_date: datetime,
    end_date: datetime,
) -> dict[str, Any]:
    """Genera un informe completo de comparación."""
    logger.info(f"Generando informe de {start_date.date()} a {end_date.date()}")

    # Cargar datos reales
    logger.info("Cargando datos reales...")
    actual_data = load_actual_data(start_date, end_date)

    if actual_data.empty:
        return {"error": "No se pudieron cargar datos reales"}

    logger.info(f"Datos reales cargados: {len(actual_data)} filas")

    # Cargar predicciones
    report = {
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {},
    }

    # Delay 30m
    logger.info("Calculando métricas de delay_30m...")
    delay_30m_preds = load_predictions(output_dir, "delay_30m")
    if delay_30m_preds:
        report["metrics"]["delay_30m"] = compute_delay_metrics(delay_30m_preds, actual_data)

    # Delay End
    logger.info("Calculando métricas de delay_end...")
    delay_end_preds = load_predictions(output_dir, "delay_end")
    if delay_end_preds:
        report["metrics"]["delay_end"] = compute_delay_metrics(delay_end_preds, actual_data)

    # Delta 10m, 20m, 30m
    for horizon in ["10m", "20m", "30m"]:
        logger.info(f"Calculando métricas de delta_{horizon}...")
        delta_preds = load_predictions(output_dir, f"delta_{horizon}")
        if delta_preds:
            report["metrics"][f"delta_{horizon}"] = compute_delta_metrics(
                delta_preds, actual_data, horizon
            )

    # Alerts
    logger.info("Calculando métricas de alerts...")
    alerts_preds = load_predictions(output_dir, "alerts")
    if alerts_preds:
        report["metrics"]["alerts"] = compute_alerts_metrics(alerts_preds, actual_data)

    # Guardar informe
    report_path = output_dir / "reports" / f"report_{start_date.date()}_{end_date.date()}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Informe guardado en {report_path}")

    return report


def print_report(report: dict[str, Any]) -> None:
    """Imprime el informe de manera legible."""
    print("\n" + "=" * 60)
    print("  INFORME DE VERIFICACIÓN DE MODELOS")
    print("=" * 60)

    if "error" in report:
        print(f"\nError: {report['error']}")
        return

    period = report.get("period", {})
    print(f"\nPeríodo: {period.get('start', '?')} a {period.get('end', '?')}")
    print(f"Generado: {report.get('generated_at', '?')}")

    metrics = report.get("metrics", {})
    if not metrics:
        print("\nNo hay métricas disponibles")
        return

    for model_name, model_metrics in metrics.items():
        print(f"\n{'─' * 40}")
        print(f"  {model_name.upper()}")
        print(f"{'─' * 40}")

        if "error" in model_metrics:
            print(f"  {model_metrics['error']}")
            continue

        for metric_name, value in model_metrics.items():
            print(f"  {metric_name}: {value}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Comparar predicciones con valores reales"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Número de días hacia atrás desde hoy (default: 1)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Fecha de inicio (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Fecha de fin (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directorio de predicciones (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    if args.output_dir:
        os.environ["OUTPUT_DIR"] = args.output_dir

    output_dir = get_output_dir()

    # Calcular fechas
    if args.start_date and args.end_date:
        start_date = datetime.fromisoformat(args.start_date)
        end_date = datetime.fromisoformat(args.end_date)
    else:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=args.days)

    logger.info(f"Período: {start_date.date()} a {end_date.date()}")

    # Generar informe
    report = generate_report(output_dir, start_date, end_date)

    # Imprimir informe
    print_report(report)


if __name__ == "__main__":
    main()