"""
Registro de modelos ML y lógica de descarga desde Weights & Biases.

Define los dataclasses que actúan como contenedores para cada modelo cargado
(DCRNNEntry, LGBMDelayEntry, DeltaEntry, AlertEntry) y la clase ModelRegistry,
que gestiona la descarga de artefactos desde W&B y la carga de cada modelo
en memoria al arrancar el servidor.

La descarga incluye reintentos automáticos con espera incremental para
manejar errores transitorios de red o rate limiting de la API de W&B.

Dependencias:
- wandb: para acceder a la API de artefactos de Weights & Biases.
- torch: para cargar los checkpoints del modelo DCRNN (.pth) y el grafo (.pt).
- joblib: para cargar los modelos LightGBM y XGBoost (.joblib).
- src.models.propagacion_estacion.models.dcrnn.SubwayDCRNN: arquitectura del modelo DCRNN.

Notas:
- Los artefactos se descargan a directorios temporales (tempfile.mkdtemp) y no
  se limpian al finalizar; el SO los eliminará en el siguiente reinicio.
- Si un modelo falla al cargar, el error se registra en ModelRegistry.errors y
  el servidor continúa con los modelos restantes (modo degradado).
"""

import json
import logging
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import wandb

logger = logging.getLogger(__name__)


@dataclass
class DCRNNEntry:
    """
    Contenedor con todos los elementos necesarios para la inferencia del modelo DCRNN.

    Atributos:
        model: Instancia del modelo SubwayDCRNN en modo evaluación.
        scaler_X: Scaler entrenado para normalizar las features de entrada.
        scaler_Y: Scaler entrenado para desnormalizar las predicciones de salida.
        nodes: Lista ordenada de identificadores de nodo ('route_stop_id').
        feature_set: Índices de las columnas de features seleccionadas.
        history_len: Número de pasos temporales de entrada al modelo.
        edge_index: Tensor de índices de aristas del grafo del metro.
        edge_weight: Tensor de pesos de aristas del grafo del metro.
        artifact_name: Nombre del artefacto W&B del que se cargó el modelo.
        loaded_at: Timestamp ISO 8601 en UTC del momento de carga.
    """
    model: Any
    scaler_X: Any
    scaler_Y: Any
    nodes: list[str]
    feature_set: list[int]
    history_len: int
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    artifact_name: str
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class LGBMDelayEntry:
    """
    Contenedor con el modelo LightGBM de retraso absoluto y su configuración de preprocesado.

    Atributos:
        model: Objeto Booster de LightGBM cargado con joblib.
        preprocessing: Dict con label_encoders, target_encoder, derived_features y target.
        artifact_name: Nombre del artefacto W&B del que se cargó el modelo.
        loaded_at: Timestamp ISO 8601 en UTC del momento de carga.
    """
    model: Any
    preprocessing: dict
    artifact_name: str
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class DeltaEntry:
    """
    Contenedor con el modelo LightGBM de tendencia de retraso (delta) y su preprocesado.

    Atributos:
        model: Objeto Booster de LightGBM cargado con joblib.
        preprocessing: Dict con vocabs, features y best_threshold.
        artifact_name: Nombre del artefacto W&B del que se cargó el modelo.
        loaded_at: Timestamp ISO 8601 en UTC del momento de carga.
    """
    model: Any
    preprocessing: dict
    artifact_name: str
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AlertEntry:
    """
    Contenedor con el modelo XGBoost de alertas por línea y su umbral de clasificación.

    Atributos:
        model: Clasificador XGBoost cargado con joblib.
        threshold: Umbral de probabilidad a partir del cual se predice alerta.
        artifact_name: Nombre del artefacto W&B del que se cargó el modelo.
        loaded_at: Timestamp ISO 8601 en UTC del momento de carga.
    """
    model: Any
    threshold: float
    artifact_name: str
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ModelRegistry:
    """
    Registro central que gestiona la carga y el acceso a todos los modelos ML.

    Almacena una instancia de cada modelo cargado (o None si no se pudo cargar)
    y un dict de errores para los que fallaron. Los métodos load_* descargan el
    artefacto de W&B y construyen la entrada correspondiente.
    """

    def __init__(self):
        """Inicializa el registro con todos los modelos a None y errores vacíos."""
        self.dcrnn: Optional[DCRNNEntry] = None
        self.lgbm_delay_30m: Optional[LGBMDelayEntry] = None
        self.lgbm_delay_end: Optional[LGBMDelayEntry] = None
        self.delta_10m: Optional[DeltaEntry] = None
        self.delta_20m: Optional[DeltaEntry] = None
        self.delta_30m: Optional[DeltaEntry] = None
        self.alertas: Optional[AlertEntry] = None
        self.errors: dict[str, str] = {}
        # _api guarda la primera instancia creada únicamente para inicializar el
        # singleton interno de wandb bajo lock. Cada hilo recibe su propia instancia
        # (pool HTTP independiente) — wandb.Api no es thread-safe si se comparte.
        self._api: Optional[wandb.Api] = None
        self._api_lock = threading.Lock()

    def _get_api(self) -> wandb.Api:
        """
        Inicializa el singleton de wandb bajo lock (solo la primera vez) y devuelve
        una instancia nueva por llamada para que cada hilo tenga su propio pool HTTP.
        """
        with self._api_lock:
            if self._api is None:
                self._api = wandb.Api(timeout=120)
        return wandb.Api(timeout=120)

    def _download(self, entity: str, project: str, artifact_ref: str) -> Path:
        """
        Descarga un artefacto de W&B a un directorio temporal con reintentos automáticos.

        Intenta hasta 4 veces con espera incremental (15, 30, 45 segundos) ante errores
        de red, rate limiting (HTTP 429/500) o timeouts. Otros errores se propagan
        inmediatamente sin reintentar.

        Parámetros:
            entity: Entidad (usuario u organización) de W&B.
            project: Nombre del proyecto de W&B.
            artifact_ref: Referencia del artefacto en formato 'nombre:alias'.

        Retorna:
            Path al directorio temporal donde se descargaron los ficheros del artefacto.

        Lanza:
            La última excepción capturada si todos los reintentos fallan.
        """
        full_ref = f"{entity}/{project}/{artifact_ref}"
        logger.info("Downloading artifact: %s", full_ref)
        last_exc = None
        for attempt in range(4):
            if attempt:
                wait = 15 * attempt
                logger.info("W&B error, retrying %s in %ds (attempt %d/4)…", artifact_ref, wait, attempt + 1)
                time.sleep(wait)
            try:
                api = self._get_api()
                artifact = api.artifact(full_ref)
                tmpdir = tempfile.mkdtemp(prefix="wandb_")
                artifact.download(root=tmpdir)
                return Path(tmpdir)
            except Exception as exc:
                msg = str(exc).lower()
                if "429" in str(exc) or "500" in str(exc) or "rate limit" in msg or "timed out" in msg or "timeout" in msg or "deadline" in msg or "connection" in msg or "network" in msg or "comm" in msg:
                    last_exc = exc
                    continue
                raise
        raise last_exc


    def load_dcrnn(self, entity: str, project: str, artifact_ref: str) -> None:
        """
        Descarga y carga el modelo DCRNN de propagación de retrasos desde W&B.

        Busca el fichero .pth del checkpoint y el fichero grafo.pt (o cualquier .pt)
        en el directorio del artefacto. Construye la instancia SubwayDCRNN con los
        hiperparámetros del checkpoint y la pone en modo evaluación.

        Parámetros:
            entity: Entidad de W&B.
            project: Proyecto de W&B donde está el artefacto.
            artifact_ref: Referencia del artefacto DCRNN (p.ej. 'dcrnn-final:latest').

        En caso de error registra el mensaje en self.errors['dcrnn'] y retorna sin lanzar.
        """
        try:
            from src.models.propagacion_estacion.models.dcrnn import SubwayDCRNN

            path = self._download(entity, project, artifact_ref)
            ckpt_file = next(path.glob("*.pth"), None)
            if not ckpt_file:
                raise FileNotFoundError(f"No .pth file in artifact {artifact_ref}")

            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            for key in ("scaler_X", "scaler_Y", "nodes", "feature_set", "history_len"):
                if key not in ckpt:
                    raise KeyError(f"Checkpoint missing '{key}'. Re-run 09_entrenamiento_final_dcrnn.py.")

            model = SubwayDCRNN(
                in_channels=ckpt["n_features"],
                hidden_channels=ckpt["hidden_channels"],
                out_horizons=ckpt["out_horizons"],
                K=ckpt["K"],
                dropout=0.0,
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            grafo_file = next(path.glob("grafo.pt"), None) or next(path.glob("*.pt"), None)
            if not grafo_file:
                raise FileNotFoundError(f"grafo.pt not found in artifact {artifact_ref}")
            grafo = torch.load(grafo_file, map_location="cpu", weights_only=False)

            self.dcrnn = DCRNNEntry(
                model=model,
                scaler_X=ckpt["scaler_X"],
                scaler_Y=ckpt["scaler_Y"],
                nodes=ckpt["nodes"],
                feature_set=ckpt["feature_set"],
                history_len=ckpt["history_len"],
                edge_index=grafo["edge_index"],
                edge_weight=grafo["edge_weight"],
                artifact_name=artifact_ref,
            )
            logger.info("DCRNN loaded (%d nodes)", len(ckpt["nodes"]))
        except Exception as exc:
            logger.error("Failed to load DCRNN: %s", exc, exc_info=True)
            self.errors["dcrnn"] = str(exc)


    def _load_lgbm_delay(self, entity: str, project: str, artifact_ref: str, key: str) -> None:
        """
        Lógica común de descarga y carga para los modelos LightGBM de retraso.

        Busca el fichero .joblib del modelo y el fichero preprocessing_*.json del
        preprocesado en el directorio del artefacto y construye un LGBMDelayEntry.

        Parámetros:
            entity: Entidad de W&B.
            project: Proyecto de W&B donde está el artefacto.
            artifact_ref: Referencia del artefacto.
            key: Nombre del atributo de ModelRegistry donde se almacenará la entrada
                 (p.ej. 'lgbm_delay_30m' o 'lgbm_delay_end').

        En caso de error registra el mensaje en self.errors[key] y retorna sin lanzar.
        """
        try:
            import joblib

            path = self._download(entity, project, artifact_ref)
            model_file = next(path.glob("*.joblib"), None)
            if not model_file:
                raise FileNotFoundError(f"No .joblib file in artifact {artifact_ref}")

            prep_file = next(path.glob("preprocessing_*.json"), None)
            if not prep_file:
                raise FileNotFoundError(f"No preprocessing_*.json in artifact {artifact_ref}")

            model = joblib.load(model_file)
            with open(prep_file) as f:
                preprocessing = json.load(f)

            entry = LGBMDelayEntry(model=model, preprocessing=preprocessing, artifact_name=artifact_ref)
            setattr(self, key, entry)
            logger.info("LightGBM %s loaded (target=%s)", key, preprocessing.get("target"))
        except Exception as exc:
            logger.error("Failed to load %s: %s", key, exc, exc_info=True)
            self.errors[key] = str(exc)

    def load_lgbm_delay_30m(self, entity: str, project: str, artifact_ref: str) -> None:
        """
        Descarga y carga el modelo LightGBM de predicción de retraso a 30 minutos.

        Parámetros:
            entity: Entidad de W&B.
            project: Proyecto de W&B.
            artifact_ref: Referencia del artefacto (p.ej. 'lgbm-delay-30m:latest').
        """
        self._load_lgbm_delay(entity, project, artifact_ref, "lgbm_delay_30m")

    def load_lgbm_delay_end(self, entity: str, project: str, artifact_ref: str) -> None:
        """
        Descarga y carga el modelo LightGBM de predicción de retraso al final del recorrido.

        Parámetros:
            entity: Entidad de W&B.
            project: Proyecto de W&B.
            artifact_ref: Referencia del artefacto (p.ej. 'lgbm-delay-end:latest').
        """
        self._load_lgbm_delay(entity, project, artifact_ref, "lgbm_delay_end")


    def _load_delta(self, entity: str, project: str, artifact_ref: str, key: str) -> None:
        """
        Lógica común de descarga y carga para los modelos LightGBM de tendencia (delta).

        El fichero .joblib puede contener el modelo directamente o un dict con claves
        'model', 'lgbm', 'classifier' o 'booster'. El preprocesado se lee de
        preprocessing_delta_*.json.

        Parámetros:
            entity: Entidad de W&B.
            project: Proyecto de W&B donde está el artefacto.
            artifact_ref: Referencia del artefacto.
            key: Nombre del atributo de ModelRegistry donde se almacenará la entrada
                 (p.ej. 'delta_10m', 'delta_20m' o 'delta_30m').

        En caso de error registra el mensaje en self.errors[key] y retorna sin lanzar.
        """
        try:
            import joblib

            path = self._download(entity, project, artifact_ref)
            model_file = next(path.glob("*.joblib"), None)
            if not model_file:
                raise FileNotFoundError(f"No .joblib file in artifact {artifact_ref}")

            prep_file = next(path.glob("preprocessing_delta_*.json"), None)
            if not prep_file:
                raise FileNotFoundError(
                    f"No preprocessing_delta_*.json in artifact {artifact_ref}. "
                    "Re-run binary_classification_delta.py to regenerate."
                )

            data = joblib.load(model_file)
            if isinstance(data, dict):
                model = (data.get("model") or data.get("lgbm")
                         or data.get("classifier") or data.get("booster"))
                if model is None:
                    raise ValueError(f"Could not extract model from dict in {artifact_ref}: keys={list(data.keys())}")
            else:
                model = data

            with open(prep_file) as f:
                preprocessing = json.load(f)

            entry = DeltaEntry(model=model, preprocessing=preprocessing, artifact_name=artifact_ref)
            setattr(self, key, entry)
            logger.info("Delta %s loaded (threshold=%.2f)", key, preprocessing.get("best_threshold", 0.5))
        except Exception as exc:
            logger.error("Failed to load %s: %s", key, exc, exc_info=True)
            self.errors[key] = str(exc)

    def load_delta_10m(self, entity: str, project: str, artifact_ref: str) -> None:
        """
        Descarga y carga el modelo delta de tendencia de retraso a 10 minutos.

        Parámetros:
            entity: Entidad de W&B.
            project: Proyecto de W&B.
            artifact_ref: Referencia del artefacto (p.ej. 'lgbm-delta_delay_10m:latest').
        """
        self._load_delta(entity, project, artifact_ref, "delta_10m")

    def load_delta_20m(self, entity: str, project: str, artifact_ref: str) -> None:
        """
        Descarga y carga el modelo delta de tendencia de retraso a 20 minutos.

        Parámetros:
            entity: Entidad de W&B.
            project: Proyecto de W&B.
            artifact_ref: Referencia del artefacto (p.ej. 'lgbm-delta_delay_20m:latest').
        """
        self._load_delta(entity, project, artifact_ref, "delta_20m")

    def load_delta_30m(self, entity: str, project: str, artifact_ref: str) -> None:
        """
        Descarga y carga el modelo delta de tendencia de retraso a 30 minutos.

        Parámetros:
            entity: Entidad de W&B.
            project: Proyecto de W&B.
            artifact_ref: Referencia del artefacto (p.ej. 'lgbm-delta_delay_30m:latest').
        """
        self._load_delta(entity, project, artifact_ref, "delta_30m")


    def load_alertas(self, entity: str, project: str, artifact_ref: str) -> None:
        """
        Descarga y carga el modelo XGBoost de alertas de incidencia por línea desde W&B.

        El fichero .joblib puede contener el modelo directamente, un dict con claves
        'model'/'classifier'/'xgb_classifier'/'xgb', o una tupla (modelo, umbral).
        Si no se especifica umbral en el fichero, se usa 0.35 por defecto.

        Parámetros:
            entity: Entidad de W&B.
            project: Proyecto de W&B.
            artifact_ref: Referencia del artefacto (p.ej. 'modelo_xgb_alertas:latest').

        En caso de error registra el mensaje en self.errors['alertas'] y retorna sin lanzar.
        """
        try:
            import joblib

            path = self._download(entity, project, artifact_ref)
            model_file = next(path.glob("*.joblib"), None)
            if not model_file:
                raise FileNotFoundError(f"No .joblib file in artifact {artifact_ref}")

            data = joblib.load(model_file)

            # El artefacto puede estar empaquetado en varios formatos posibles
            if isinstance(data, dict):
                model = (data.get("model") or data.get("classifier")
                         or data.get("xgb_classifier") or data.get("xgb"))
                threshold = float(data.get("threshold", 0.35))
            elif isinstance(data, (list, tuple)) and len(data) >= 2:
                model, threshold = data[0], float(data[1])
            else:
                model, threshold = data, 0.35

            if model is None:
                raise ValueError(f"Could not extract classifier from pkl in {artifact_ref}")

            self.alertas = AlertEntry(model=model, threshold=threshold, artifact_name=artifact_ref)
            logger.info("XGBoost alertas loaded (threshold=%.2f)", threshold)
        except Exception as exc:
            logger.error("Failed to load alertas: %s", exc, exc_info=True)
            self.errors["alertas"] = str(exc)
