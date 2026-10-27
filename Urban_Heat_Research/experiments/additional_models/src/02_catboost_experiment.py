"""
STEP 02 - CATBOOST SUPPLEMENTARY EXPERIMENT
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Evaluate CatBoost as a separate supplementary candidate using the SAME
2018-2023 development data, 93 predictors, strict expanding rolling-origin
validation, and spatial-anomaly + prior-mean annual-level framework used by the
finalized LightGBM development strategy.

IMPORTANT
---------
- Existing production src/models/outputs are NOT modified.
- 2024 reserved validation is NOT loaded.
- 2025 final lockbox is NOT loaded.
- Hyperparameter selection uses ONLY 2018-2023 development evidence.
- Primary selection metric: mean rolling-origin MAE.
- The saved CatBoost model is EXPERIMENTAL only.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise RuntimeError(
        "CatBoost is not installed in the active Urban_Heat_Research venv.\n"
        "Run: python -m pip install catboost\n"
        "Then rerun Step 02."
    ) from exc


# =============================================================================
# 1. SETTINGS
# =============================================================================

TARGET = "LST_C"
GRID_COL = "grid_id"
AREA_COL = "area_name"
YEAR_COL = "year"

DEVELOPMENT_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
ROLLING_VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]

EXPECTED_ROWS = 13_164
EXPECTED_GRIDS = 2_194
EXPECTED_PREDICTORS = 93
EXPECTED_FOLDS = 5
RANDOM_SEED = 42

MODEL_NAME = "CatBoost"
STRATEGY_NAME = "catboost_anomaly_plus_prior_mean_level"

NON_PREDICTOR_COLUMNS = [GRID_COL, TARGET, AREA_COL]

# Small predeclared development-only search.
CANDIDATE_CONFIGS = [
    {
        "config_id": "CB01",
        "iterations": 300,
        "depth": 4,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
    },
    {
        "config_id": "CB02",
        "iterations": 500,
        "depth": 5,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
    },
    {
        "config_id": "CB03",
        "iterations": 400,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
        "random_strength": 1.0,
    },
    {
        "config_id": "CB04",
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 10.0,
        "random_strength": 1.0,
    },
    {
        "config_id": "CB05",
        "iterations": 600,
        "depth": 4,
        "learning_rate": 0.02,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
    },
]


# =============================================================================
# 2. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = EXPERIMENT_ROOT / "data" / "01_experiment_dev_2018_2023.csv"
POLICY_PATH = EXPERIMENT_ROOT / "outputs" / "reports" / "01_experiment_policy.json"
STEP01_AUDIT_PATH = (
    EXPERIMENT_ROOT / "outputs" / "reports" / "01_prepare_experiment_data_audit.csv"
)

MODEL_DIR = EXPERIMENT_ROOT / "models"
METRIC_DIR = EXPERIMENT_ROOT / "outputs" / "metrics"
PREDICTION_DIR = EXPERIMENT_ROOT / "outputs" / "predictions"
REPORT_DIR = EXPERIMENT_ROOT / "outputs" / "reports"

for directory in [MODEL_DIR, METRIC_DIR, PREDICTION_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SEARCH_FOLD_METRICS_PATH = METRIC_DIR / "02_catboost_search_fold_metrics.csv"
SEARCH_SUMMARY_PATH = METRIC_DIR / "02_catboost_search_summary.csv"
BEST_FOLD_METRICS_PATH = METRIC_DIR / "02_catboost_best_fold_metrics.csv"
BEST_PREDICTIONS_PATH = PREDICTION_DIR / "02_catboost_best_rolling_predictions.csv"
BEST_CONFIG_PATH = REPORT_DIR / "02_catboost_best_config.json"
AUDIT_PATH = REPORT_DIR / "02_catboost_experiment_audit.csv"
REPORT_PATH = REPORT_DIR / "02_catboost_experiment_report.txt"
HASH_PATH = REPORT_DIR / "02_catboost_experiment_hashes.csv"
MODEL_PATH = MODEL_DIR / "02_catboost_anomaly_trained_2018_2023.joblib"


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 122)
    print(title)
    print("=" * 122)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        low_memory=False,
        dtype={GRID_COL: "string", AREA_COL: "string"},
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": mape_percent(y_true, y_pred),
        "mean_prediction_bias_pred_minus_actual_c": float(np.mean(y_pred - y_true)),
    }


def build_model(config: dict) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=int(config["iterations"]),
        depth=int(config["depth"]),
        learning_rate=float(config["learning_rate"]),
        l2_leaf_reg=float(config["l2_leaf_reg"]),
        random_strength=float(config["random_strength"]),
        loss_function="RMSE",
        random_seed=RANDOM_SEED,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def build_training_anomaly(train_df: pd.DataFrame) -> np.ndarray:
    annual_means = train_df.groupby(YEAR_COL)[TARGET].transform("mean").to_numpy(dtype=float)
    y = train_df[TARGET].to_numpy(dtype=float)
    return y - annual_means


def prior_mean_level(train_df: pd.DataFrame) -> float:
    annual_means = train_df.groupby(YEAR_COL)[TARGET].mean().sort_index()
    return float(annual_means.mean())


def evaluate_fold(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    predictors: list[str],
) -> tuple[dict, pd.DataFrame]:
    validation_year = int(val_df[YEAR_COL].iloc[0])
    train_years = sorted(train_df[YEAR_COL].unique().tolist())

    if not train_years or max(train_years) >= validation_year:
        raise RuntimeError("Leakage detected in rolling-origin fold.")

    X_train = train_df[predictors]
    X_val = val_df[predictors]
    y_train_anomaly = build_training_anomaly(train_df)
    y_val = val_df[TARGET].to_numpy(dtype=float)

    model = build_model(config)
    start = time.perf_counter()
    model.fit(X_train, y_train_anomaly)
    raw_pred_anomaly = np.asarray(model.predict(X_val), dtype=float)
    elapsed = float(time.perf_counter() - start)

    # Prediction-only recentering: no validation target is used here.
    centered_pred_anomaly = raw_pred_anomaly - raw_pred_anomaly.mean()

    # Validation target mean below is evaluation-only.
    actual_val_mean = float(y_val.mean())
    actual_centered_anomaly = y_val - actual_val_mean

    level_forecast = prior_mean_level(train_df)
    final_pred = level_forecast + centered_pred_anomaly

    final_m = metrics(y_val, final_pred)
    anomaly_m = metrics(actual_centered_anomaly, centered_pred_anomaly)

    fold_row = {
        "config_id": config["config_id"],
        "model": MODEL_NAME,
        "strategy": STRATEGY_NAME,
        "validation_year": validation_year,
        "train_start_year": int(min(train_years)),
        "train_end_year": int(max(train_years)),
        "train_year_count": int(len(train_years)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "predictor_count": int(len(predictors)),
        "annual_level_strategy": "prior_mean_level",
        "annual_level_forecast_c": level_forecast,
        "actual_validation_mean_lst_c_eval_only": actual_val_mean,
        "annual_level_error_c_eval_only": float(level_forecast - actual_val_mean),
        "raw_predicted_anomaly_mean_c": float(raw_pred_anomaly.mean()),
        "centered_predicted_anomaly_mean_c": float(centered_pred_anomaly.mean()),
        "anomaly_mae": anomaly_m["mae"],
        "anomaly_rmse": anomaly_m["rmse"],
        "anomaly_r2": anomaly_m["r2"],
        "centered_spatial_r2": anomaly_m["r2"],
        "mae": final_m["mae"],
        "rmse": final_m["rmse"],
        "r2": final_m["r2"],
        "mape_percent": final_m["mape_percent"],
        "mean_prediction_bias_pred_minus_actual_c": final_m[
            "mean_prediction_bias_pred_minus_actual_c"
        ],
        "fit_predict_seconds": elapsed,
    }

    pred_df = pd.DataFrame(
        {
            GRID_COL: val_df[GRID_COL].astype(str).to_numpy(),
            AREA_COL: val_df[AREA_COL].astype(str).to_numpy(),
            YEAR_COL: val_df[YEAR_COL].astype(int).to_numpy(),
            "config_id": config["config_id"],
            "model": MODEL_NAME,
            "strategy": STRATEGY_NAME,
            "actual_lst_c": y_val,
            "actual_centered_anomaly_c_eval_only": actual_centered_anomaly,
            "raw_predicted_anomaly_c": raw_pred_anomaly,
            "centered_predicted_anomaly_c": centered_pred_anomaly,
            "annual_level_forecast_c": np.full(len(val_df), level_forecast, dtype=float),
            "predicted_lst_c": final_pred,
            "prediction_error_c": final_pred - y_val,
        }
    )

    return fold_row, pred_df


# =============================================================================
# 4. LOAD + VERIFY STEP 01
# =============================================================================

section("STEP 02 - CATBOOST SUPPLEMENTARY EXPERIMENT")

print(f"Project root       : {PROJECT_ROOT}")
print(f"Experiment root    : {EXPERIMENT_ROOT}")
print(f"Input              : {DATA_PATH}")
print("Production src     : UNTOUCHED")
print("Production models  : UNTOUCHED")
print("Production outputs : UNTOUCHED")
print("2024 data          : NOT LOADED")
print("2025 data          : NOT LOADED")
print(f"Candidate configs  : {len(CANDIDATE_CONFIGS)}")
print("Primary objective  : mean rolling-origin MAE")

for p in [DATA_PATH, POLICY_PATH, STEP01_AUDIT_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Required Step 01 file not found: {p}")

step01_audit = pd.read_csv(STEP01_AUDIT_PATH)
if not (step01_audit["status"].astype(str) == "PASS").all():
    raise RuntimeError("Step 01 audit is not fully PASS.")

policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
df = load_csv(DATA_PATH)
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="raise").astype(int)
actual_years = sorted(df[YEAR_COL].unique().tolist())
predictors = [c for c in df.columns if c not in NON_PREDICTOR_COLUMNS]

if actual_years != DEVELOPMENT_YEARS:
    raise RuntimeError(f"Unexpected years: {actual_years}")
if df[YEAR_COL].isin([2024, 2025]).any():
    raise RuntimeError("2024 or 2025 rows were unexpectedly loaded.")
if len(df) != EXPECTED_ROWS:
    raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, found {len(df)}.")
if int(df[GRID_COL].nunique()) != EXPECTED_GRIDS:
    raise RuntimeError("Unexpected unique grid count.")
if len(predictors) != EXPECTED_PREDICTORS:
    raise RuntimeError(
        f"Expected {EXPECTED_PREDICTORS} predictors, found {len(predictors)}."
    )
if predictors != policy.get("predictors", []):
    raise RuntimeError("Predictor order does not match frozen Step 01 policy.")
if df[predictors].isna().any().any():
    raise RuntimeError("Predictor missingness detected.")
if not np.isfinite(df[predictors].to_numpy(dtype=float)).all():
    raise RuntimeError("Non-finite predictor value detected.")

print("\nPASS: Step 01 audit is fully PASS.")
print(f"Rows               : {len(df):,}")
print(f"Unique grids       : {df[GRID_COL].nunique():,}")
print(f"Years              : {actual_years}")
print(f"Predictors         : {len(predictors)}")


# =============================================================================
# 5. STRICT ROLLING-ORIGIN FOLDS
# =============================================================================

section("1. STRICT ROLLING-ORIGIN FOLDS")

for validation_year in ROLLING_VALIDATION_YEARS:
    print(f"2018-{validation_year - 1} -> {validation_year}")


# =============================================================================
# 6. DEVELOPMENT-ONLY SEARCH
# =============================================================================

section("2. CATBOOST DEVELOPMENT-ONLY SEARCH")

search_rows = []

for idx, config in enumerate(CANDIDATE_CONFIGS, start=1):
    print(f"\n--- {config['config_id']} ({idx}/{len(CANDIDATE_CONFIGS)}) ---")
    print(json.dumps(config, sort_keys=True))

    for validation_year in ROLLING_VALIDATION_YEARS:
        train_df = df[df[YEAR_COL] < validation_year].copy()
        val_df = df[df[YEAR_COL] == validation_year].copy()

        fold_row, _ = evaluate_fold(config, train_df, val_df, predictors)
        search_rows.append(fold_row)

        print(
            f"{fold_row['train_start_year']}-{fold_row['train_end_year']} "
            f"-> {validation_year} | "
            f"MAE={fold_row['mae']:.4f} | "
            f"RMSE={fold_row['rmse']:.4f} | "
            f"R²={fold_row['r2']:.4f} | "
            f"Spatial R²={fold_row['centered_spatial_r2']:.4f}"
        )

search_df = pd.DataFrame(search_rows)
search_df.to_csv(SEARCH_FOLD_METRICS_PATH, index=False, encoding="utf-8")
print(f"\nSaved: {SEARCH_FOLD_METRICS_PATH}")


# =============================================================================
# 7. SUMMARIZE + SELECT BEST CATBOOST CONFIG
# =============================================================================

section("3. CATBOOST SEARCH SUMMARY")

summary_df = (
    search_df.groupby("config_id", as_index=False)
    .agg(
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"),
        mean_r2=("r2", "mean"),
        mean_centered_spatial_r2=("centered_spatial_r2", "mean"),
        minimum_r2=("r2", "min"),
        negative_r2_count=("r2", lambda s: int((s < 0).sum())),
        worst_year_mae=("mae", "max"),
        mean_mape_percent=("mape_percent", "mean"),
        mean_fit_predict_seconds=("fit_predict_seconds", "mean"),
    )
)

summary_df = summary_df.merge(
    pd.DataFrame(CANDIDATE_CONFIGS),
    on="config_id",
    how="left",
    validate="one_to_one",
)

summary_df = (
    summary_df.sort_values(
        ["mean_mae", "mean_rmse", "mean_centered_spatial_r2"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    .reset_index(drop=True)
)
summary_df["development_rank"] = np.arange(len(summary_df)) + 1
summary_df.to_csv(SEARCH_SUMMARY_PATH, index=False, encoding="utf-8")

print(
    summary_df[
        [
            "development_rank",
            "config_id",
            "mean_mae",
            "mean_rmse",
            "mean_r2",
            "mean_centered_spatial_r2",
            "negative_r2_count",
            "worst_year_mae",
        ]
    ].to_string(index=False)
)

best_config_id = str(summary_df.iloc[0]["config_id"])
best_config = next(c for c in CANDIDATE_CONFIGS if c["config_id"] == best_config_id)

BEST_CONFIG_PATH.write_text(
    json.dumps(
        {
            "model": MODEL_NAME,
            "strategy": STRATEGY_NAME,
            "selection_scope": "2018-2023 development only",
            "selection_metric": "mean rolling-origin MAE",
            "tie_breakers": ["mean RMSE", "mean centered spatial R2"],
            "best_config": best_config,
            "2024_loaded_or_used": False,
            "2025_loaded_or_used": False,
            "production_model_replaced": False,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"\nSelected CatBoost config: {best_config_id}")
print(json.dumps(best_config, indent=2))
print(f"Saved: {SEARCH_SUMMARY_PATH}")
print(f"Saved: {BEST_CONFIG_PATH}")


# =============================================================================
# 8. RE-RUN BEST CONFIG FOR FROZEN PREDICTIONS
# =============================================================================

section("4. BEST CATBOOST CONFIG - FROZEN ROLLING PREDICTIONS")

best_fold_rows = []
best_pred_frames = []

for validation_year in ROLLING_VALIDATION_YEARS:
    train_df = df[df[YEAR_COL] < validation_year].copy()
    val_df = df[df[YEAR_COL] == validation_year].copy()

    fold_row, pred_df = evaluate_fold(best_config, train_df, val_df, predictors)
    best_fold_rows.append(fold_row)
    best_pred_frames.append(pred_df)

    print(
        f"{fold_row['train_start_year']}-{fold_row['train_end_year']} "
        f"-> {validation_year} | "
        f"MAE={fold_row['mae']:.4f} | "
        f"RMSE={fold_row['rmse']:.4f} | "
        f"R²={fold_row['r2']:.4f} | "
        f"Spatial R²={fold_row['centered_spatial_r2']:.4f}"
    )

best_fold_df = pd.DataFrame(best_fold_rows)
best_pred_df = pd.concat(best_pred_frames, ignore_index=True)

best_fold_df.to_csv(BEST_FOLD_METRICS_PATH, index=False, encoding="utf-8")
best_pred_df.to_csv(BEST_PREDICTIONS_PATH, index=False, encoding="utf-8")

print(f"\nSaved: {BEST_FOLD_METRICS_PATH}")
print(f"Saved: {BEST_PREDICTIONS_PATH}")


# =============================================================================
# 9. FIT FULL DEVELOPMENT EXPERIMENTAL CANDIDATE
# =============================================================================

section("5. FIT FULL 2018-2023 EXPERIMENTAL CANDIDATE")

final_model = build_model(best_config)
final_model.fit(df[predictors], build_training_anomaly(df))
joblib.dump(final_model, MODEL_PATH)

print(f"Saved: {MODEL_PATH}")
print("NOTE: this is an EXPERIMENTAL model, not a production replacement.")


# =============================================================================
# 10. RESULT SUMMARY
# =============================================================================

section("6. CATBOOST DEVELOPMENT RESULT")

mean_mae = float(best_fold_df["mae"].mean())
std_mae = float(best_fold_df["mae"].std())
mean_rmse = float(best_fold_df["rmse"].mean())
mean_r2 = float(best_fold_df["r2"].mean())
mean_spatial_r2 = float(best_fold_df["centered_spatial_r2"].mean())
negative_r2_count = int((best_fold_df["r2"] < 0).sum())
worst_year_mae = float(best_fold_df["mae"].max())
minimum_r2 = float(best_fold_df["r2"].min())

print(f"Best config              : {best_config_id}")
print(f"Mean MAE                 : {mean_mae:.6f} °C")
print(f"Std MAE                  : {std_mae:.6f} °C")
print(f"Mean RMSE                : {mean_rmse:.6f} °C")
print(f"Mean R²                  : {mean_r2:.6f}")
print(f"Mean centered spatial R² : {mean_spatial_r2:.6f}")
print(f"Negative-R² fold count   : {negative_r2_count}")
print(f"Worst-year MAE           : {worst_year_mae:.6f} °C")
print(f"Minimum R²               : {minimum_r2:.6f}")
print("2024 validation          : UNTOUCHED")
print("2025 final lockbox       : UNTOUCHED")
print("Production model         : UNCHANGED")


# =============================================================================
# 11. STRICT AUDIT
# =============================================================================

section("7. STRICT STEP 02 AUDIT")

expected_search_rows = len(CANDIDATE_CONFIGS) * EXPECTED_FOLDS
expected_prediction_rows = EXPECTED_GRIDS * EXPECTED_FOLDS

training_anomaly_means = []
for year in DEVELOPMENT_YEARS:
    one_year = df[df[YEAR_COL] == year]
    training_anomaly_means.append(float(build_training_anomaly(one_year).mean()))

predicted_anomaly_means = (
    best_pred_df.groupby(YEAR_COL)["centered_predicted_anomaly_c"]
    .mean()
    .to_numpy(dtype=float)
)

audit_rows = [
    {"check": "step01_audit", "status": "PASS", "details": "Step 01 fully PASS."},
    {
        "check": "development_years",
        "status": "PASS" if actual_years == DEVELOPMENT_YEARS else "FAIL",
        "details": str(actual_years),
    },
    {
        "check": "predictor_count",
        "status": "PASS" if len(predictors) == EXPECTED_PREDICTORS else "FAIL",
        "details": f"actual={len(predictors)}, expected={EXPECTED_PREDICTORS}",
    },
    {
        "check": "search_fold_metric_rows",
        "status": "PASS" if len(search_df) == expected_search_rows else "FAIL",
        "details": f"actual={len(search_df)}, expected={expected_search_rows}",
    },
    {
        "check": "best_fold_metric_rows",
        "status": "PASS" if len(best_fold_df) == EXPECTED_FOLDS else "FAIL",
        "details": f"actual={len(best_fold_df)}, expected={EXPECTED_FOLDS}",
    },
    {
        "check": "best_prediction_rows",
        "status": "PASS" if len(best_pred_df) == expected_prediction_rows else "FAIL",
        "details": f"actual={len(best_pred_df)}, expected={expected_prediction_rows}",
    },
    {
        "check": "training_anomaly_zero_mean_within_year",
        "status": "PASS" if np.all(np.abs(training_anomaly_means) < 1e-10) else "FAIL",
        "details": str(training_anomaly_means),
    },
    {
        "check": "predicted_anomaly_recentered_without_target",
        "status": "PASS" if np.all(np.abs(predicted_anomaly_means) < 1e-10) else "FAIL",
        "details": str(predicted_anomaly_means.tolist()),
    },
    {
        "check": "2024_loaded_or_used",
        "status": "PASS",
        "details": "No. 2024 is not loaded by Step 02.",
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": "No. 2025 is not loaded by Step 02.",
    },
    {
        "check": "random_kfold_used",
        "status": "PASS",
        "details": "No. Strict expanding rolling-origin folds only.",
    },
    {
        "check": "metrics_finite",
        "status": "PASS"
        if np.isfinite(
            best_fold_df[["mae", "rmse", "r2", "centered_spatial_r2"]].to_numpy(dtype=float)
        ).all()
        else "FAIL",
        "details": "Best-config fold metrics are finite.",
    },
    {
        "check": "production_pipeline_modified",
        "status": "PASS",
        "details": "Only experiments/additional_models is written.",
    },
    {
        "check": "production_model_replaced",
        "status": "PASS",
        "details": "No. CatBoost model is experimental only.",
    },
]

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(AUDIT_PATH, index=False, encoding="utf-8")
print(audit_df.to_string(index=False))

if not (audit_df["status"] == "PASS").all():
    raise RuntimeError("Step 02 audit contains one or more FAIL rows.")


# =============================================================================
# 12. HASHES + REPORT
# =============================================================================

section("8. REPRODUCIBILITY HASHES")

hash_paths = [
    DATA_PATH,
    SEARCH_FOLD_METRICS_PATH,
    SEARCH_SUMMARY_PATH,
    BEST_FOLD_METRICS_PATH,
    BEST_PREDICTIONS_PATH,
    BEST_CONFIG_PATH,
    MODEL_PATH,
    AUDIT_PATH,
]

hash_df = pd.DataFrame(
    [
        {
            "file": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(path),
        }
        for path in hash_paths
    ]
)
hash_df.to_csv(HASH_PATH, index=False, encoding="utf-8")
print(hash_df.to_string(index=False))
print(f"\nSaved: {HASH_PATH}")

section("9. SAVE STEP 02 REPORT")

report_lines = [
    "STEP 02 - CATBOOST SUPPLEMENTARY EXPERIMENT",
    "",
    f"Best config: {best_config_id}",
    f"Mean MAE: {mean_mae:.6f} °C",
    f"Std MAE: {std_mae:.6f} °C",
    f"Mean RMSE: {mean_rmse:.6f} °C",
    f"Mean R2: {mean_r2:.6f}",
    f"Mean centered spatial R2: {mean_spatial_r2:.6f}",
    f"Negative-R2 fold count: {negative_r2_count}",
    f"Worst-year MAE: {worst_year_mae:.6f} °C",
    f"Minimum R2: {minimum_r2:.6f}",
    "",
    "2024 reserved validation: NOT LOADED",
    "2025 final lockbox: NOT LOADED",
    "Production pipeline: UNTOUCHED",
    "Production model: UNCHANGED",
    "",
    "STEP 02 STATUS: PASS",
]
REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
print(f"Saved: {REPORT_PATH}")

section("STEP 02 COMPLETED SUCCESSFULLY")
print("CatBoost supplementary development experiment completed.")
print("2024 reserved validation remains untouched.")
print("2025 final lockbox remains untouched.")
print("Production model remains unchanged.")
print("Next: Step 03 - Extra Trees supplementary experiment.")
