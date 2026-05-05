import json
import logging
import tempfile
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
    model: Any                  
    preprocessing: dict        
    artifact_name: str
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class DeltaEntry:
    model: Any                  
    preprocessing: dict       
    artifact_name: str
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AlertEntry:
    model: Any                
    threshold: float
    artifact_name: str
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ModelRegistry:
    def __init__(self):
        self.dcrnn: Optional[DCRNNEntry] = None
        self.lgbm_delay_30m: Optional[LGBMDelayEntry] = None
        self.lgbm_delay_end: Optional[LGBMDelayEntry] = None
        self.delta_10m: Optional[DeltaEntry] = None
        self.delta_20m: Optional[DeltaEntry] = None
        self.delta_30m: Optional[DeltaEntry] = None
        self.alertas: Optional[AlertEntry] = None
        self.errors: dict[str, str] = {}

    def _download(self, entity: str, project: str, artifact_ref: str) -> Path:
        full_ref = f"{entity}/{project}/{artifact_ref}"
        logger.info("Downloading artifact: %s", full_ref)
        last_exc = None
        for attempt in range(4):
            if attempt:
                wait = 15 * attempt
                logger.info("W&B error, retrying %s in %ds (attempt %d/4)…", artifact_ref, wait, attempt + 1)
                time.sleep(wait)
            try:
                api = wandb.Api(timeout=120)
                artifact = api.artifact(full_ref)
                tmpdir = tempfile.mkdtemp(prefix="wandb_")
                artifact.download(root=tmpdir)
                return Path(tmpdir)
            except Exception as exc:
                msg = str(exc).lower()
                if "429" in str(exc) or "500" in str(exc) or "rate limit" in msg or "timed out" in msg or "timeout" in msg or "deadline" in msg:
                    last_exc = exc
                    continue
                raise
        raise last_exc


    def load_dcrnn(self, entity: str, project: str, artifact_ref: str) -> None:
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
        self._load_lgbm_delay(entity, project, artifact_ref, "lgbm_delay_30m")

    def load_lgbm_delay_end(self, entity: str, project: str, artifact_ref: str) -> None:
        self._load_lgbm_delay(entity, project, artifact_ref, "lgbm_delay_end")


    def _load_delta(self, entity: str, project: str, artifact_ref: str, key: str) -> None:
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
        self._load_delta(entity, project, artifact_ref, "delta_10m")

    def load_delta_20m(self, entity: str, project: str, artifact_ref: str) -> None:
        self._load_delta(entity, project, artifact_ref, "delta_20m")

    def load_delta_30m(self, entity: str, project: str, artifact_ref: str) -> None:
        self._load_delta(entity, project, artifact_ref, "delta_30m")


    def load_alertas(self, entity: str, project: str, artifact_ref: str) -> None:
        try:
            import joblib

            path = self._download(entity, project, artifact_ref)
            model_file = next(path.glob("*.joblib"), None)
            if not model_file:
                raise FileNotFoundError(f"No .joblib file in artifact {artifact_ref}")

            data = joblib.load(model_file)

            # Handle multiple possible pkl formats
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
