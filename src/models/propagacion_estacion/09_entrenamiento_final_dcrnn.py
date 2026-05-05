"""
Entrenamiento final DCRNN sobre Train + Validación concatenados.

Carga la mejor configuración HPO, combina los splits Train y Val en un único
conjunto de entrenamiento y entrena durante NUM_EPOCHS épocas fijas (sin Early
Stopping, ya que no hay conjunto de validación disponible). Guarda únicamente
los pesos del modelo; la evaluación sobre Test se realiza en 12_evaluacion_modelos.py.

Carga:  artefactos/tensores.pt  (X_train, Y_train, X_val, Y_val, X_test, Y_test,
                                  scaler_X, scaler_Y, times, nodes)
        artefactos/grafo.pt     (edge_index, edge_weight, n_nodes)
        artefactos/hpo.pt       (best_params, feature_set, subset_name)
        artefactos/ablacion.pt  (df_ablacion para W&B)

Guarda: artefactos/dcrnn_final.pth
  {
    'model_state_dict':   ...,
    'best_params':        dict,
    'feature_set':        list[int],
    'subset_name':        str,
    'best_trainval_loss': float,
    'best_epoch':         int,
    'scaler_Y':           StandardScaler,
    'n_features':         int,
    'out_horizons':       int,
    'K':                  int,
    'hidden_channels':    int,
    'history_len':        int,
  }

Uso
---
    uv run python src/models/propagacion_estacion/09_entrenamiento_final_dcrnn.py
"""
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.dcrnn import SubwayDCRNN
from utils.dataset import SubwayDataset, validar_batch_vs_grafo

RUTA_TENSORES = Path(__file__).parent / "artefactos" / "tensores.pt"
RUTA_GRAFO    = Path(__file__).parent / "artefactos" / "grafo.pt"
RUTA_HPO      = Path(__file__).parent / "artefactos" / "hpo.pt"
RUTA_ABLACION = Path(__file__).parent / "artefactos" / "ablacion.pt"
RUTA_MODELO   = Path(__file__).parent / "artefactos" / "dcrnn_final.pth"

WANDB_PROJECT  = "pd1-c2526-team5"
WANDB_RUN_NAME = "dcrnn-final-trainval"

NUM_EPOCHS   = 12
OUT_HORIZONS = 3
SEED         = 42


def fijar_semilla(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def main():
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)

    print("=== 09 Entrenamiento Final DCRNN (Train + Val) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    # cudnn.benchmark=True es lento en GPUs con throttling (300 MHz): re-benchmarkea
    # kernels en cada epoch y multiplica el tiempo de batch x1.5. Desactivado.
    fijar_semilla(SEED)

    # ── Cargar artefactos ─────────────────────────────────────────────────────
    datos    = torch.load(RUTA_TENSORES, weights_only=False)
    X_train  = datos['X_train']
    Y_train  = datos['Y_train']
    X_val    = datos['X_val']
    Y_val    = datos['Y_val']
    scaler_X = datos['scaler_X']
    scaler_Y = datos['scaler_Y']
    nodes    = datos['nodes']

    grafo       = torch.load(RUTA_GRAFO, weights_only=False)
    edge_index  = grafo['edge_index'].to(device)
    edge_weight = grafo['edge_weight'].to(device)
    n_nodes     = grafo['n_nodes']

    hpo           = torch.load(RUTA_HPO, weights_only=False)
    best_params   = hpo['best_params']
    feature_set   = hpo['feature_set']
    subset_name   = hpo['subset_name']
    best_strategy = hpo['best_strategy']

    ablacion    = torch.load(RUTA_ABLACION, weights_only=False)
    df_ablacion = ablacion['df_resultados']

    nf  = len(feature_set)
    hl  = best_params['history_len']
    bs  = best_params['batch_size']
    hc  = best_params['hidden_channels']
    K   = best_params['K']
    drp = best_params['dropout']
    lr  = best_params['lr']
    gc_ = best_params['grad_clip']
    sp  = best_params['scheduler_patience']
    sf  = best_params['scheduler_factor']

    print(f"Subset: {subset_name!r} ({nf} features) | history_len={hl} | hidden={hc} | K={K}")
    print(f"lr={lr:.2e} | dropout={drp:.3f} | batch={bs} | grad_clip={gc_}")

    # ── Concatenar Train + Val ────────────────────────────────────────────────
    # Ambos splits ya están escalados con el mismo scaler (ajustado sobre Train).
    # La concatenación es válida; el scaler_Y sigue siendo aplicable en el
    # script de evaluación (12_evaluacion_modelos.py).
    X_trainval = np.concatenate([X_train[:, :, feature_set], X_val[:, :, feature_set]], axis=0)
    Y_trainval = np.concatenate([Y_train, Y_val], axis=0)
    print(f"Muestras Train+Val: {X_trainval.shape[0]}  (train={X_train.shape[0]}, val={X_val.shape[0]})")

    tv_ld = DataLoader(
        SubwayDataset(X_trainval, Y_trainval, history_len=hl),
        batch_size=bs,
        shuffle=True,
    )

    # ── Modelo ────────────────────────────────────────────────────────────────
    modelo = SubwayDCRNN(
        in_channels=nf,
        hidden_channels=hc,
        out_horizons=OUT_HORIZONS,
        K=K,
        dropout=drp,
    ).to(device)

    opt  = torch.optim.Adam(modelo.parameters(), lr=lr)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', patience=sp, factor=sf, min_lr=1e-5
    )
    crit = torch.nn.L1Loss()

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            'subset_name':        subset_name,
            'best_strategy':      best_strategy,
            'feature_set':        feature_set,
            'n_features':         nf,
            'n_nodes':            n_nodes,
            'n_edges':            edge_index.shape[1],
            'out_horizons':       OUT_HORIZONS,
            'history_len':        hl,
            'num_epochs':         NUM_EPOCHS,
            'trainval_samples':   len(tv_ld.dataset),
            **{f'hp_{k}': v for k, v in best_params.items()},
        },
    )
    wandb.log({'ablacion': wandb.Table(dataframe=df_ablacion)})

    # ── Bucle de entrenamiento (épocas fijas, sin Early Stopping) ─────────────
    # Al unir Train+Val no existe conjunto de validación para ES. Se guarda el
    # estado con menor loss de entrenamiento como mejor aproximación.
    best_loss  = float('inf')
    best_epoch = 0
    best_state = None
    t0         = time.time()

    # Validar consistencia una sola vez antes del bucle de entrenamiento
    xb0, yb0 = next(iter(tv_ld))
    validar_batch_vs_grafo(xb0.to(device), yb0.to(device), edge_index, edge_weight, tag="trainval")
    del xb0, yb0

    print(f"batch_size: {bs} | AMP: desactivado | cudnn.benchmark: desactivado")

    for epoch in range(1, NUM_EPOCHS + 1):
        modelo.train()
        acc_loss = 0.0
        for xb, yb in tv_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(modelo(xb, edge_index, edge_weight), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), gc_)
            opt.step()
            acc_loss += loss.item()

        train_loss = acc_loss / len(tv_ld)
        curr_lr    = opt.param_groups[0]['lr']
        sch.step(train_loss)
        new_lr     = opt.param_groups[0]['lr']
        lr_tag     = f"  ↓ lr={new_lr:.2e}" if new_lr < curr_lr else ""

        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'lr': new_lr})
        print(f"Época {epoch:02d}/{NUM_EPOCHS} | trainval={train_loss:.6f}{lr_tag}"
              + ("  [best]" if train_loss < best_loss else ""))

        if train_loss < best_loss:
            best_loss  = train_loss
            best_epoch = epoch
            best_state = copy.deepcopy(modelo.state_dict())

    tiempo_total = time.time() - t0
    print(f"\nEntrenamiento completado en {tiempo_total:.0f}s | Mejor loss: {best_loss:.6f} (época {best_epoch})")

    if best_state is not None:
        modelo.load_state_dict(best_state)

    wandb.log({'best_trainval_loss': best_loss, 'best_epoch': best_epoch, 'train_time_sec': tiempo_total})

    # ── Guardar modelo (sin métricas de test) ─────────────────────────────────
    torch.save(
        {
            'model_state_dict':    modelo.state_dict(),
            'best_params':         best_params,
            'feature_set':         feature_set,
            'subset_name':         subset_name,
            'best_trainval_loss':  best_loss,
            'best_epoch':          best_epoch,
            'scaler_X':            scaler_X,
            'scaler_Y':            scaler_Y,
            'nodes':               nodes,
            'n_features':          nf,
            'out_horizons':        OUT_HORIZONS,
            'K':                   K,
            'hidden_channels':     hc,
            'history_len':         hl,
        },
        RUTA_MODELO,
    )
    print(f"Modelo final DCRNN guardado en: {RUTA_MODELO}")
    print("NOTA: la evaluación sobre Test se realiza en 12_evaluacion_modelos.py")

    # ── Subir modelo a W&B Artifacts ──────────────────────────────────────────
    artifact = wandb.Artifact(
        name="dcrnn-final",
        type="model",
        description="Entrenamiento final DCRNN sobre Train+Val",
        metadata={
            'subset_name':        subset_name,
            'best_trainval_loss': best_loss,
            'best_epoch':         best_epoch,
            'n_features':         nf,
            'hidden_channels':    hc,
            'K':                  K,
            'history_len':        hl,
            'out_horizons':       OUT_HORIZONS,
        },
    )
    artifact.add_file(str(RUTA_MODELO))
    artifact.add_file(str(RUTA_GRAFO))
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
