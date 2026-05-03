"""
Búsqueda de hiperparámetros: Random Search + Optuna

Ejecuta dos estrategias sobre el subconjunto de features seleccionado en la
ablación, las compara y guarda la configuración del ganador en artefactos/hpo.pt.

  Fase 1 — Random Search  (10 trials muestreados aleatoriamente)
  Fase 2 — Optuna TPE + MedianPruner  (8 trials)
  Fase 3 — Comparación de estrategias → se elige el menor val_loss

Espacio de búsqueda:
    hidden_channels    : {32, 64, 96}
    K                  : {2, 3}
    dropout            : [0.0, 0.4]
    lr                 : [1e-4, 3e-3] (log)
    batch_size         : {8, 16}
    history_len        : {4, 8, 12}
    grad_clip          : {1.0, 3.0, 5.0}
    scheduler_patience : {2, 3, 4}
    scheduler_factor   : {0.3, 0.5, 0.7}

Carga:  artefactos/tensores.pt, artefactos/grafo.pt, artefactos/ablacion.pt
Guarda: artefactos/hpo.pt
  {
    'best_params':        dict,
    'best_val_loss':      float,
    'best_strategy':      str,   # 'random' | 'optuna'
    'feature_set':        list[int],
    'subset_name':        str,
    'df_random':          pd.DataFrame,
    'df_optuna':          pd.DataFrame,
    'strategy_comparison':pd.DataFrame,
  }

Uso
---
    uv run python src/models/propagacion_estacion/06_tuning_hpo_dcrnn.py
"""
import gc
import random as _rnd
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from models.dcrnn import SubwayDCRNN
from utils.dataset import SubwayDataset, validar_batch_vs_grafo

optuna.logging.set_verbosity(optuna.logging.WARNING)

RUTA_TENSORES = Path(__file__).parent / "artefactos" / "tensores.pt"
RUTA_GRAFO    = Path(__file__).parent / "artefactos" / "grafo.pt"
RUTA_ABLACION = Path(__file__).parent / "artefactos" / "ablacion.pt"
RUTA_SALIDA   = Path(__file__).parent / "artefactos" / "hpo.pt"

N_TRIALS_RANDOM = 8
N_TRIALS_OPTUNA = 7
MAX_EPOCHS      = 10
ES_PATIENCE     = 5
# Limitar train Y val para HPO — la validación completa (1133 timesteps) consume
# tanto como entrenar y haría inviable el HPO (>70 h con 40 trials).
MAX_T_HPO       = 1000
MAX_T_VAL_HPO   = 400
OUT_HORIZONS    = 3

ESPACIO_HP = {
    'hidden_channels':    [32, 64, 96],
    'K':                  [2, 3],
    'dropout':            (0.0, 0.4),
    'lr':                 (1e-4, 3e-3),
    'batch_size':         [16],
    'history_len':        [4, 8, 12],
    'grad_clip':          [1.0, 3.0, 5.0],
    'scheduler_patience': [2, 3, 4],
    'scheduler_factor':   [0.3, 0.5, 0.7],
}


def muestrear_params_aleatorios(seed: int) -> dict:
    rng = _rnd.Random(seed)
    lr_min, lr_max = ESPACIO_HP['lr']
    return {
        'hidden_channels':    rng.choice(ESPACIO_HP['hidden_channels']),
        'K':                  rng.choice(ESPACIO_HP['K']),
        'dropout':            rng.uniform(*ESPACIO_HP['dropout']),
        'lr':                 10 ** rng.uniform(np.log10(lr_min), np.log10(lr_max)),
        'batch_size':         rng.choice(ESPACIO_HP['batch_size']),
        'history_len':        rng.choice(ESPACIO_HP['history_len']),
        'grad_clip':          rng.choice(ESPACIO_HP['grad_clip']),
        'scheduler_patience': rng.choice(ESPACIO_HP['scheduler_patience']),
        'scheduler_factor':   rng.choice(ESPACIO_HP['scheduler_factor']),
    }


def train_trial(
    params: dict,
    feature_idx: list[int],
    X_train, Y_train, X_val, Y_val,
    edge_index, edge_weight,
    device: torch.device,
    seed: int,
    epoch_callback=None,
) -> dict:
    """
    Entrena un trial. epoch_callback(ep, val_loss) → bool detiene si True (pruner Optuna).
    """
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
    validar_batch_vs_grafo(xb0.to(device), yb0.to(device), edge_index, edge_weight, tag="hpo")
    del xb0, yb0

    best_val = float('inf')
    no_imp   = 0
    pruned   = False
    t0       = time.time()

    for ep in range(1, MAX_EPOCHS + 1):
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

        if epoch_callback is not None and epoch_callback(ep, vl):
            pruned = True
            break

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
        'pruned':               pruned,
    }


def fase_random(feature_set, X_train, Y_train, X_val, Y_val, edge_index, edge_weight, device):
    print(f"\nFASE 1 — Random Search ({N_TRIALS_RANDOM} trials × {MAX_EPOCHS} épocas)\n")

    filas = []
    best_val    = float('inf')
    best_params = {}
    best_time   = 0.0
    t_inicio    = time.time()

    for num in range(N_TRIALS_RANDOM):
        params = muestrear_params_aleatorios(seed=num * 13 + 7)
        res    = train_trial(
            params, feature_set,
            X_train, Y_train, X_val, Y_val,
            edge_index, edge_weight,
            device=device, seed=num,
        )
        filas.append({'trial': num + 1, 'val_loss': res['best_val_loss_scaled'],
                      'tiempo_s': res['train_time_sec'], **params})
        if res['best_val_loss_scaled'] < best_val:
            best_val    = res['best_val_loss_scaled']
            best_params = params.copy()
            best_time   = res['train_time_sec']
        print(f"  Trial {num+1:02d}/{N_TRIALS_RANDOM} | val={res['best_val_loss_scaled']:.6f}"
              f"  ({res['train_time_sec']:.0f}s)")
        del res, params

    t_total  = time.time() - t_inicio
    df       = pd.DataFrame(filas).sort_values('val_loss').reset_index(drop=True)
    cols_vis = [c for c in ['trial', 'val_loss', 'tiempo_s', 'hidden_channels', 'K', 'lr', 'dropout']
                if c in df.columns]
    print(f"\n=== Top-5 Random Search ===")
    print(df[cols_vis].head(5).to_string(index=False))
    print(f"Mejor val_loss: {best_val:.6f} | Tiempo total: {t_total:.0f}s")

    return best_val, best_params, best_time, t_total, df


def fase_optuna(feature_set, X_train, Y_train, X_val, Y_val, edge_index, edge_weight, device):
    print(f"\nFASE 2 — Optuna TPE + MedianPruner ({N_TRIALS_OPTUNA} trials × {MAX_EPOCHS} épocas)\n")

    def objetivo(trial: optuna.Trial) -> float:
        params = {
            'hidden_channels':    trial.suggest_categorical('hidden_channels', [32, 64, 96]),
            'K':                  trial.suggest_categorical('K', [2, 3]),
            'dropout':            trial.suggest_float('dropout', 0.0, 0.4),
            'lr':                 trial.suggest_float('lr', 1e-4, 3e-3, log=True),
            'batch_size':         trial.suggest_categorical('batch_size', [16]),
            'history_len':        trial.suggest_categorical('history_len', [4, 8, 12]),
            'grad_clip':          trial.suggest_categorical('grad_clip', [1.0, 3.0, 5.0]),
            'scheduler_patience': trial.suggest_categorical('scheduler_patience', [2, 3, 4]),
            'scheduler_factor':   trial.suggest_categorical('scheduler_factor', [0.3, 0.5, 0.7]),
        }

        def callback_pruner(ep: int, vl: float) -> bool:
            trial.report(vl, ep)
            return trial.should_prune()

        res = train_trial(
            params, feature_set,
            X_train, Y_train, X_val, Y_val,
            edge_index, edge_weight,
            device=device, seed=trial.number,
            epoch_callback=callback_pruner,
        )
        print(f"  Trial {trial.number+1:02d}/{N_TRIALS_OPTUNA} | val={res['best_val_loss_scaled']:.6f}"
              f"  ({res['train_time_sec']:.0f}s)"
              + ("  [PRUNED]" if res['pruned'] else ""))

        if res['pruned']:
            raise optuna.exceptions.TrialPruned()
        return res['best_val_loss_scaled']

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    t_inicio = time.time()
    study.optimize(objetivo, n_trials=N_TRIALS_OPTUNA, show_progress_bar=False)
    t_total  = time.time() - t_inicio

    filas = []
    for t in study.trials:
        dur = t.duration.total_seconds() if t.duration else 0.0
        filas.append({
            'trial':    t.number + 1,
            'val_loss': t.value if t.value is not None else float('nan'),
            'tiempo_s': dur,
            'estado':   t.state.name,
            **t.params,
        })
    df       = pd.DataFrame(filas).sort_values('val_loss').reset_index(drop=True)
    df_ok    = df.dropna(subset=['val_loss'])
    n_podados = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)

    cols_vis = [c for c in ['trial', 'val_loss', 'tiempo_s', 'hidden_channels', 'K', 'lr', 'dropout']
                if c in df_ok.columns]
    print(f"\n=== Top-5 Optuna ===")
    print(df_ok[cols_vis].head(5).to_string(index=False))
    print(f"Trials podados: {n_podados}/{N_TRIALS_OPTUNA} | Tiempo total: {t_total:.0f}s")

    best_val    = study.best_value
    best_params = dict(study.best_params)
    best_time   = study.best_trial.duration.total_seconds() if study.best_trial.duration else 0.0
    print(f"Mejor val_loss: {best_val:.6f}")

    return best_val, best_params, best_time, t_total, df


def comparar_estrategias(
    best_random_val, t_random_total, n_random,
    best_optuna_val, t_optuna_total, n_optuna,
) -> pd.DataFrame:
    df = pd.DataFrame([
        {
            'strategy':              'random',
            'n_trials':              n_random,
            'best_val_loss':         best_random_val,
            'total_search_time_sec': t_random_total,
            'val_loss_per_min':      best_random_val / (t_random_total / 60),
        },
        {
            'strategy':              'optuna',
            'n_trials':              n_optuna,
            'best_val_loss':         best_optuna_val,
            'total_search_time_sec': t_optuna_total,
            'val_loss_per_min':      best_optuna_val / (t_optuna_total / 60),
        },
    ])

    print("\nFASE 3 — Comparación de estrategias\n")
    print(df.to_string(index=False))

    ganador   = df.loc[df['best_val_loss'].idxmin(), 'strategy']
    diff_pct  = abs(best_random_val - best_optuna_val) / max(best_random_val, best_optuna_val) * 100
    print(f"\nGanador (menor val_loss): {ganador.upper()}")
    print(f"Diferencia de calidad   : {diff_pct:.2f}%")
    if diff_pct < 2.0:
        print("Nota: diferencia < 2% — ambas estrategias son equivalentes para este presupuesto.")

    return df, ganador


def main():
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

    print("=== 06 HPO: Random Search + Optuna ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    datos       = torch.load(RUTA_TENSORES, weights_only=False)
    X_train     = datos['X_train'][:MAX_T_HPO]
    Y_train     = datos['Y_train'][:MAX_T_HPO]
    X_val       = datos['X_val'][:MAX_T_VAL_HPO]
    Y_val       = datos['Y_val'][:MAX_T_VAL_HPO]
    print(f"T_train (HPO): {len(X_train)} | T_val: {len(X_val)}")

    grafo       = torch.load(RUTA_GRAFO, weights_only=False)
    edge_index  = grafo['edge_index'].to(device)
    edge_weight = grafo['edge_weight'].to(device)

    ablacion    = torch.load(RUTA_ABLACION, weights_only=False)
    feature_set = ablacion['selected_feature_set']
    subset_name = ablacion['selected_subset_name']
    print(f"Subset: {subset_name!r} ({len(feature_set)} features)")

    args = (feature_set, X_train, Y_train, X_val, Y_val, edge_index, edge_weight, device)

    best_random_val, best_random_params, _, t_random, df_random = fase_random(*args)
    best_optuna_val, best_optuna_params, _, t_optuna, df_optuna = fase_optuna(*args)

    df_comparacion, ganador = comparar_estrategias(
        best_random_val, t_random, N_TRIALS_RANDOM,
        best_optuna_val, t_optuna, N_TRIALS_OPTUNA,
    )

    if ganador == 'random':
        best_params   = best_random_params
        best_val_loss = best_random_val
    else:
        best_params   = best_optuna_params
        best_val_loss = best_optuna_val

    print(f"\nConfiguración seleccionada ({ganador.upper()}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    torch.save(
        {
            'best_params':         best_params,
            'best_val_loss':       best_val_loss,
            'best_strategy':       ganador,
            'feature_set':         feature_set,
            'subset_name':         subset_name,
            'df_random':           df_random,
            'df_optuna':           df_optuna,
            'strategy_comparison': df_comparacion,
        },
        RUTA_SALIDA,
    )
    print(f"\nHPO guardado en: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
