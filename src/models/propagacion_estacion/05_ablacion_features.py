"""
Ablación de subconjuntos de features

Entrena SubwayDCRNN con 5 subconjuntos de features durante un número reducido
de épocas y guarda el subconjunto con menor val_loss en artefactos/ablacion.pt.

Grupos semánticos (índices sobre F=14):
    base_delay  : [0,1,2,6,7]  — delay_seconds, lagged×2, rolling, headway
    contexto    : [4,5,11,12,13]— temp_extreme, n_eventos, afecta×3
    calendario  : [8,9,10]      — hour_sin, hour_cos, dow
    operativa   : [3]           — is_unscheduled

Subsets evaluados:
    all_features   → todos los 14 índices
    sin_contexto   → sin el grupo 'contexto'
    sin_calendario → sin el grupo 'calendario'
    sin_operativa  → sin el grupo 'operativa'
    solo_base_delay→ solo el grupo 'base_delay'

Carga:  artefactos/tensores.pt, artefactos/grafo.pt
Guarda: artefactos/ablacion.pt
  {
    'selected_subset_name': str,
    'selected_feature_set': list[int],
    'df_resultados':        pd.DataFrame,
  }

Uso
---
    uv run python src/models/propagacion_estacion/05_ablacion_features.py
"""
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.dcrnn import SubwayDCRNN
from utils.dataset import SubwayDataset, validar_batch_vs_grafo

RUTA_TENSORES = Path(__file__).parent / "artefactos" / "tensores.pt"
RUTA_GRAFO    = Path(__file__).parent / "artefactos" / "grafo.pt"
RUTA_SALIDA   = Path(__file__).parent / "artefactos" / "ablacion.pt"

# Hiperparámetros base para la ablación 
PARAMS_BASE = {
    'hidden_channels':    64,
    'K':                  2,
    'dropout':            0.1,
    'lr':                 1e-3,
    'batch_size':         16,
    'history_len':        8,
    'grad_clip':          3.0,
    'scheduler_patience': 3,
    'scheduler_factor':   0.5,
}
MAX_EPOCHS_ABLACION = 15
ES_PATIENCE         = 7
# Limitar muestras de entrenamiento para ablación: solo necesitamos ranking relativo,
# no convergencia completa. El entrenamiento final (09) usa el dataset completo.
MAX_T_ABLACION      = 1000
MAX_T_VAL_ABLACION  = 400   # limitar también la validación (sino domina el coste por época)
OUT_HORIZONS        = 3

ALL_IDX = list(range(14))
GRUPOS  = {
    'base_delay': [0, 1, 2, 6, 7],
    'contexto':   [4, 5, 11, 12, 13],
    'calendario': [8, 9, 10],
    'operativa':  [3],
}
SUBSETS = {
    'all_features':    ALL_IDX,
    'sin_contexto':    [i for i in ALL_IDX if i not in GRUPOS['contexto']],
    'sin_calendario':  [i for i in ALL_IDX if i not in GRUPOS['calendario']],
    'sin_operativa':   [i for i in ALL_IDX if i not in GRUPOS['operativa']],
    'solo_base_delay': GRUPOS['base_delay'],
}


def entrenar_subset(
    X_train, Y_train, X_val, Y_val,
    edge_index, edge_weight,
    feature_idx: list[int],
    params: dict,
    device: torch.device,
    seed: int = 0,
) -> dict:
    """Entrena un modelo con el subconjunto de features indicado y devuelve métricas."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr = X_train[:, :, feature_idx]
    X_vl = X_val[:, :, feature_idx]
    nf   = len(feature_idx)
    hl   = params['history_len']
    bs   = params['batch_size']

    tr_ld = DataLoader(SubwayDataset(X_tr, Y_train, history_len=hl), batch_size=bs, shuffle=True)
    vl_ld = DataLoader(SubwayDataset(X_vl, Y_val,   history_len=hl), batch_size=bs, shuffle=False)

    modelo = SubwayDCRNN(
        in_channels=nf,
        hidden_channels=params['hidden_channels'],
        out_horizons=OUT_HORIZONS,
        K=params['K'],
        dropout=params['dropout'],
    ).to(device)

    opt  = torch.optim.Adam(modelo.parameters(), lr=params['lr'])
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min',
        patience=params['scheduler_patience'],
        factor=params['scheduler_factor'],
        min_lr=1e-5,
    )
    crit = torch.nn.L1Loss()

    # Validar consistencia una sola vez antes del bucle (evita GPU-CPU sync en cada batch)
    xb0, yb0 = next(iter(tr_ld))
    validar_batch_vs_grafo(xb0.to(device), yb0.to(device), edge_index, edge_weight, tag="ablacion")
    del xb0, yb0

    best_val = float('inf')
    no_imp   = 0
    t0       = time.time()

    for ep in range(1, MAX_EPOCHS_ABLACION + 1):
        modelo.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(modelo(xb, edge_index, edge_weight), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), params['grad_clip'])
            opt.step()

        modelo.eval()
        with torch.no_grad():
            vl_acc = sum(
                crit(modelo(xb.to(device), edge_index, edge_weight), yb.to(device)).item()
                for xb, yb in vl_ld
            )
            vl = vl_acc / len(vl_ld)
        sch.step(vl)

        if vl < best_val:
            best_val, no_imp = vl, 0
        else:
            no_imp += 1
        if no_imp >= ES_PATIENCE:
            break

    del modelo, opt, sch, tr_ld, vl_ld, X_tr, X_vl
    gc.collect()

    return {
        'best_val_loss_scaled': best_val,
        'train_time_sec':       time.time() - t0,
        'n_features':           nf,
        'feature_idx':          list(feature_idx),
    }


def main():
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

    print("05: Ablación de Features")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    datos     = torch.load(RUTA_TENSORES, weights_only=False)
    X_train   = datos['X_train'][:MAX_T_ABLACION]
    Y_train   = datos['Y_train'][:MAX_T_ABLACION]
    X_val     = datos['X_val'][:MAX_T_VAL_ABLACION]
    Y_val     = datos['Y_val'][:MAX_T_VAL_ABLACION]
    print(f"T_train (ablación): {len(X_train)} | T_val: {len(X_val)}")

    grafo        = torch.load(RUTA_GRAFO, weights_only=False)
    edge_index   = grafo['edge_index'].to(device)
    edge_weight  = grafo['edge_weight'].to(device)

    resultados = []
    N = len(SUBSETS)
    for i, (nombre, idx) in enumerate(SUBSETS.items(), start=1):
        print(f"  [{i}/{N}] {nombre!r:20s} ({len(idx)} features) ... ", end='', flush=True)
        res = entrenar_subset(
            X_train, Y_train, X_val, Y_val,
            edge_index, edge_weight,
            feature_idx=idx,
            params=PARAMS_BASE,
            device=device,
            seed=i,
        )
        resultados.append({'subset_name': nombre, **res})
        print(f"val_loss={res['best_val_loss_scaled']:.6f}  ({res['train_time_sec']:.0f}s)")

    df = pd.DataFrame(resultados).sort_values('best_val_loss_scaled').reset_index(drop=True)
    print("\n=== Resultados (ordenados por val_loss) ===")
    print(df[['subset_name', 'n_features', 'best_val_loss_scaled', 'train_time_sec']].to_string(index=False))

    mejor_nombre = df.iloc[0]['subset_name']
    mejor_idx    = SUBSETS[mejor_nombre]
    print(f"\nSubconjunto seleccionado: {mejor_nombre!r} ({len(mejor_idx)} features, índices: {mejor_idx})")

    torch.save(
        {
            'selected_subset_name': mejor_nombre,
            'selected_feature_set': mejor_idx,
            'df_resultados':        df,
        },
        RUTA_SALIDA,
    )
    print(f"Ablación guardada en: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
