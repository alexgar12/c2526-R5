"""
Búsqueda nocturna: ventana temporal + esquema de pesos — delay_30m
==================================================================


Combina:
  • 7 ventanas  (inicio en ene/abr/jul/sep/oct/nov/dic 2025 → ene 2026)
  • 4 esquemas de pesos (uniform / linear / exponential / step)
  Total: 28 combinaciones


Uso:
    uv run python src/models/prediccion_retrasos/delay_30m/search/window_weight_search.py

Variables de entorno:
    MINIO_ACCESS_KEY  MINIO_SECRET_KEY  WANDB_API_KEY
"""

import os
import time
import warnings
import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common.minio_client import download_df_parquet

warnings.filterwarnings("ignore")

try:
    import psutil
    def _ram_pct() -> float:
        return psutil.virtual_memory().percent
except ImportError:
    def _ram_pct() -> float:
        return -1.0

ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

TARGET        = "target_delay_30m"
DATA_TEMPLATE = "grupo5/final/year={year}/month={month:02d}/dataset_final.parquet"

TEST_YEAR   = 2026
TEST_MONTH  = 2

# Meses a pre-cargar: todos los de 2025 + ene 2026
CACHE_2025  = list(range(4, 13))
CACHE_2026  = [1]

# Ventanas: cada entrada indica qué meses de 2025 usar (+ ene2026 siempre)
WINDOW_CONFIGS = [
    ("desde_ene25", list(range(1, 13))),
    ("desde_abr25", list(range(4, 13))),
    ("desde_jul25", list(range(7, 13))),
    ("desde_sep25", list(range(9, 13))),
    ("desde_oct25", list(range(10, 13))),
    ("desde_nov25", list(range(11, 13))),
    ("desde_dic25", list(range(12, 13))),
]

WEIGHT_SCHEMES  = ["uniform", "linear", "exponential", "step"]
NUM_BOOST_ROUND = 2000      # reducido para la búsqueda (óptimo es 4260)
SAMPLE_FRAC     = 0.5      # fracción de datos por mes (1.0 = todos)

LGBM_PARAMS = {
    "objective":         "regression_l1",
    "metric":            "mae",
    "learning_rate":     0.05,
    "num_leaves":        511,
    "max_depth":         16,
    "min_child_samples": 100,
    "min_split_gain":    0.37042771510661165,
    "feature_fraction":  0.7426288737567357,
    "bagging_fraction":  0.8165370010747616,
    "bagging_freq":      5,
    "reg_alpha":         1.5346393797283635,
    "reg_lambda":        1.2926631392622208,
    "n_jobs":            -1,
    "verbose":           -1,
    "seed":              42,
}

CAT_FEATURES = ["route_id", "direction", "category", "tipo_referente"]
STOP_ID_COL  = "stop_id"
EXCLUDE_COLS = {
    "date", "match_key", "stop_id", "merge_time", "timestamp_start",
    "service_date", "trip_uid", "is_unscheduled",
    "target_delay_10m", "target_delay_20m", "target_delay_30m",
    "target_delay_45m", "target_delay_60m", "target_delay_end",
    "delta_delay_10m", "delta_delay_20m", "delta_delay_30m",
    "delta_delay_45m", "delta_delay_60m", "delta_delay_end",
    "alert_in_next_15m", "alert_in_next_30m", "seconds_to_next_alert",
    "delay_minutes", "scheduled_time", "actual_time",
    "station_delay_10m", "station_delay_20m", "station_delay_30m",
    "delay_vs_station", "station_trend",
}

WANDB_PROJECT = "pd1-c2526-team5"
OUTPUT_CSV    = "window_weight_search_30m_results.csv"


# ── Carga y caché ─────────────────────────────────────────────────────────────

def fetch_month(year: int, month: int) -> pd.DataFrame | None:
    path = DATA_TEMPLATE.format(year=year, month=month)
    try:
        df = download_df_parquet(ACCESS_KEY, SECRET_KEY, path)
        df = df[df["is_unscheduled"] == False]
        df = df.dropna(subset=[TARGET])
        df = df[df["scheduled_time_to_end"] >= 1800]
        if SAMPLE_FRAC < 1.0:
            df = df.sample(frac=SAMPLE_FRAC, random_state=42)
        for col in CAT_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype("category")
        print(f"  ✓ {year}-{month:02d}: {len(df):,} filas")
        return df
    except Exception as e:
        print(f"  ✗ {year}-{month:02d}: {e}")
        return None


def preload_all() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Descarga todos los meses necesarios una sola vez."""
    print("── Pre-cargando datos (una sola vez) ──────────────────────────────")

    cache_2025: dict[int, pd.DataFrame] = {}
    print("2025:")
    for m in CACHE_2025:
        df = fetch_month(2025, m)
        if df is not None:
            cache_2025[m] = df

    print("2026 ene:")
    df_ene26 = fetch_month(2026, 1)

    print(f"Test (feb-2026):")
    df_test  = fetch_month(TEST_YEAR, TEST_MONTH)

    print(f"\nCaché lista: {len(cache_2025)} meses 2025 | ene26: {len(df_ene26):,} | test: {len(df_test):,}\n")
    return cache_2025, df_ene26, df_test


# ── Feature engineering ───────────────────────────────────────────────────────

def encode_categoricals(df_train, df_test):
    for col in CAT_FEATURES:
        if col not in df_train.columns:
            continue
        vocab = {v: i for i, v in enumerate(df_train[col].astype(str).unique())}
        df_train[col] = df_train[col].astype(str).map(vocab).astype(int)
        df_test[col]  = df_test[col].astype(str).map(vocab).fillna(-1).astype(int)
    return df_train, df_test


def add_target_encoding(df_train, df_test):
    means = df_train.groupby(STOP_ID_COL)[TARGET].mean()
    global_mean = df_train[TARGET].mean()
    df_train[f"{STOP_ID_COL}_target_enc"] = df_train[STOP_ID_COL].map(means)
    df_test[f"{STOP_ID_COL}_target_enc"]  = df_test[STOP_ID_COL].map(means).fillna(global_mean)
    return df_train, df_test


def add_derived_features(df):
    if "lagged_delay_1" in df.columns and "delay_seconds" in df.columns:
        df["delay_velocity"] = df["delay_seconds"] - df["lagged_delay_1"]
    if "lagged_delay_1" in df.columns and "lagged_delay_2" in df.columns:
        df["delay_acceleration"] = (
            (df["delay_seconds"] - df["lagged_delay_1"])
            - (df["lagged_delay_1"] - df["lagged_delay_2"])
        )
    if "delay_seconds" in df.columns and "stops_to_end" in df.columns:
        df["delay_x_stops_remaining"] = df["delay_seconds"] * df["stops_to_end"]
    if "delay_seconds" in df.columns and "scheduled_time_to_end" in df.columns:
        df["delay_ratio"] = df["delay_seconds"] / (df["scheduled_time_to_end"] + 1)
    return df




def get_features(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS and c != TARGET]


# ── Pesos ─────────────────────────────────────────────────────────────────────

def build_weights(month_sizes: list[int], scheme: str) -> np.ndarray:
    n = len(month_sizes)
    if scheme == "uniform":
        w = np.ones(n)
    elif scheme == "linear":
        w = np.arange(1, n + 1, dtype=float)
    elif scheme == "exponential":
        lam = np.log(10) / max(n - 1, 1)
        w = np.exp(lam * np.arange(n))
    elif scheme == "step":
        w = np.ones(n)
        if n >= 1: w[-1] = 3.0
        if n >= 2: w[-2] = 1.5
    else:
        raise ValueError(scheme)
    w = w / w.mean()
    return np.concatenate([np.full(s, wi) for s, wi in zip(month_sizes, w)]).astype(np.float32)


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, prefix="") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {
        f"{prefix}mae_s":   round(mae, 2),
        f"{prefix}mae_min": round(mae / 60, 3),
        f"{prefix}rmse_s":  round(rmse, 2),
        f"{prefix}r2":      round(r2, 4),
    }


# ── Búsqueda ──────────────────────────────────────────────────────────────────

def run_search():
    print("=" * 70)
    print("Búsqueda ventana × peso — delay_30m")
    print(f"  {len(WINDOW_CONFIGS)} ventanas × {len(WEIGHT_SCHEMES)} esquemas = "
          f"{len(WINDOW_CONFIGS) * len(WEIGHT_SCHEMES)} combinaciones")
    print(f"  Iteraciones búsqueda: {NUM_BOOST_ROUND}  |  sample_frac: {SAMPLE_FRAC}")
    print("=" * 70 + "\n")

    cache_2025, df_ene26, df_test_raw = preload_all()

    results   = []
    n_total   = len(WINDOW_CONFIGS) * len(WEIGHT_SCHEMES)
    n_done    = 0

    for win_label, months_2025 in WINDOW_CONFIGS:
        dfs_2025 = [cache_2025[m] for m in months_2025 if m in cache_2025]
        if not dfs_2025:
            print(f"⚠ Ventana {win_label}: sin datos, saltando.")
            continue

        sizes = [len(d) for d in dfs_2025] + [len(df_ene26)]
        print(f"\n{'─'*60}")
        print(f"Ventana: {win_label}  ({len(dfs_2025)} meses 2025 + ene26 = {sum(sizes):,} filas)")

        # Preprocesar una sola vez por ventana para evitar repetir trabajo pesado.
        t_prep = time.time()
        df_train = pd.concat(dfs_2025 + [df_ene26], ignore_index=True)
        # shallow copy: comparte arrays de columnas no modificadas con df_test_raw
        df_test  = df_test_raw.copy(deep=False)

        df_train, df_test = encode_categoricals(df_train, df_test)
        df_train, df_test = add_target_encoding(df_train, df_test)
        df_train = add_derived_features(df_train)
        df_test  = add_derived_features(df_test)

        feats = get_features(df_train)
        X_tr = df_train[feats].copy()
        y_tr = df_train[TARGET].copy()
        X_te = df_test[feats].copy()
        y_te = df_test[TARGET].copy()
        n_train = len(X_tr)
        n_test  = len(X_te)

        del df_train, df_test, dfs_2025
        gc.collect()

        prep_elapsed = time.time() - t_prep
        print(f"  Preprocesado ventana: {prep_elapsed/60:.1f} min  |  RAM: {_ram_pct():.0f}%")

        for scheme in WEIGHT_SCHEMES:
            n_done += 1
            run_name = f"win_{win_label}__w_{scheme}"
            print(f"\n  [{n_done}/{n_total}] {run_name}  |  RAM: {_ram_pct():.0f}%")
            t0 = time.time()

            weights = build_weights(sizes, scheme)
            train_set = lgb.Dataset(
                X_tr,
                label=y_tr,
                weight=weights,
                free_raw_data=True,
            )

            run = wandb.init(
                project=WANDB_PROJECT, name=run_name,
                group="window-weight-search-30m",
                config={**LGBM_PARAMS, "target": TARGET, "window": win_label,
                        "weight_scheme": scheme, "n_months": len(sizes),
                        "n_train": n_train, "n_test": n_test,
                        "num_boost_round": NUM_BOOST_ROUND, "sample_frac": SAMPLE_FRAC},
                reinit=True,
            )

            model = lgb.train(
                LGBM_PARAMS,
                train_set,
                num_boost_round=NUM_BOOST_ROUND,
                callbacks=[lgb.log_evaluation(500)],
            )

            m_tr = compute_metrics(y_tr, model.predict(X_tr), "train_")
            m_te = compute_metrics(y_te, model.predict(X_te), "test_")
            elapsed = time.time() - t0

            print(f"    train MAE={m_tr['train_mae_s']}s  test MAE={m_te['test_mae_s']}s  "
                  f"R²={m_te['test_r2']}  ({elapsed/60:.1f} min)  RAM: {_ram_pct():.0f}%")

            wandb.log({**m_tr, **m_te, "elapsed_s": elapsed})
            run.finish()

            results.append({"window": win_label, "weight_scheme": scheme,
                            "n_months": len(sizes), "n_train": n_train,
                            **m_tr, **m_te, "elapsed_s": round(elapsed, 1)})

            del model, train_set, weights
            gc.collect()

            pd.DataFrame(results).sort_values("test_mae_s").to_csv(OUTPUT_CSV, index=False)

        del X_tr, y_tr, X_te, y_te
        gc.collect()

    df_res = pd.DataFrame(results).sort_values("test_mae_s")
    df_res.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "=" * 70)
    print(f"delay_30m completado. Resultados: {OUTPUT_CSV}")
    print("\nTop 5:")
    cols = ["window", "weight_scheme", "n_months", "test_mae_s", "test_r2", "train_mae_s"]
    print(df_res[cols].head(5).to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    run_search()
