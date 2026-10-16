"""
STEP 03 - EXTRA TREES SUPPLEMENTARY EXPERIMENT
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Evaluate ExtraTreesRegressor as a separate supplementary candidate using the
same development years, 93 predictors, strict rolling-origin folds, and
spatial-anomaly + prior-mean annual-level framework used by the finalized
production-development methodology.

IMPORTANT
---------
- Existing production src/models/outputs are NOT modified.
- 2024 reserved validation is NOT loaded.
- 2025 final lockbox is NOT loaded.
- Hyperparameter selection uses ONLY 2018-2023 development evidence.
- Primary selection metric: mean rolling-origin MAE.
- The saved Extra Trees model is EXPERIMENTAL only.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
MODEL_NAME = "Extra Trees"
STRATEGY_NAME = "extra_trees_anomaly_plus_prior_mean_level"

NON_PREDICTOR_COLUMNS = [GRID_COL, TARGET, AREA_COL]

# Small predeclared development-only search.
# No 2024/2025 information is used to define or select these configurations.
CANDIDATE_CONFIGS = [
    {
        "config_id": "ET01",
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    {
        "config_id": "ET02",
        "n_estimators": 700,
        "max_depth": None,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": 0.7,
    },
    {
        "config_id": "ET03",
        "n_estimators": 700,
        "max_depth": 20,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "config_id": "ET04",
        "n_estimators": 900,
        "max_depth": 30,
        "min_samples_split": 5,
        "min_samples_leaf": 3,
        "max_features": 0.7,
    },
    {
        "config_id": "ET05",
        "n_estimators": 900,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 5,
        "max_features": 0.5,
    },
]


# =============================================================================
# 2. PATHS
# =============================================================================

# File location:
# Urban_Heat_Research/
#   experiments/
#     additional_models/
#       src/
#         03_extra_trees_experiment.py
#
# parents[3] -> Urban_Heat_Research
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

SEARCH_METRICS_PATH = METRIC_DIR / "03_extra_trees_search_fold_metrics.csv"
SEARCH_SUMMARY_PATH = METRIC_DIR / "03_extra_trees_search_summary.csv"
BEST_FOLD_METRICS_PATH = METRIC_DIR / "03_extra_trees_best_fold_metrics.csv"
BEST_PREDICTIONS_PATH = (
    PREDICTION_DIR / "03_extra_trees_best_rolling_predictions.csv"
)
BEST_CONFIG_PATH = REPORT_DIR / "03_extra_trees_best_config.json"
AUDIT_PATH = REPORT_DIR / "03_extra_trees_experiment_audit.csv"
REPORT_PATH = REPORT_DIR / "03_extra_trees_experiment_report.txt"
HASH_PATH = REPORT_DIR / "03_extra_trees_experiment_hashes.csv"
FINAL_EXPERIMENT_MODEL_PATH = (
    MODEL_DIR / "03_extra_trees_anomaly_trained_2018_2023.joblib"
)


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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": mape_percent(y_true, y_pred),
        "mean_prediction_bias_pred_minus_actual_c": float(np.mean(y_pred - y_true)),
    }


def build_extra_trees_model(config: dict) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=int(config["n_estimators"]),
        max_depth=config["max_depth"],
        min_samples_split=int(config["min_samples_split"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=config["max_features"],
        bootstrap=False,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def build_training_anomaly(train_df: pd.DataFrame) -> np.ndarray:
    """Center each training year's LST using that training year's mean."""
    annual_means = (
        train_df.groupby(YEAR_COL)[TARGET]
        .transform("mean")
        .to_numpy(dtype=float)
    )
    target = train_df[TARGET].to_numpy(dtype=float)
    return target - annual_means


def prior_mean_level(train_df: pd.DataFrame) -> float:
    """Forecast annual level using only prior training-year annual means."""
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

    if max(train_years) >= validation_year:
        raise RuntimeError(
            "Leakage detected: training includes validation year or a future year."
        )

    X_train = train_df[predictors]
    X_val = val_df[predictors]
    y_train_anomaly = build_training_anomaly(train_df)
    y_val = val_df[TARGET].to_numpy(dtype=float)

    model = build_extra_trees_model(config)

    start = time.perf_counter()
    model.fit(X_train, y_train_anomaly)
    raw_predicted_anomaly = np.asarray(model.predict(X_val), dtype=float)
    fit_predict_seconds = float(time.perf_counter() - start)

    # Leakage-safe recentering: prediction mean only, no validation target used.
    centered_predicted_anomaly = (
        raw_predicted_anomaly - raw_predicted_anomaly.mean()
    )

    # Validation target mean is used only for evaluation diagnostics.
    actual_validation_mean = float(y_val.mean())
    actual_centered_anomaly = y_val - actual_validation_mean

    level_forecast = prior_mean_level(train_df)
    final_prediction = level_forecast + centered_predicted_anomaly

    final_metrics = regression_metrics(y_val, final_prediction)
    anomaly_metrics = regression_metrics(
        actual_centered_anomaly,
        centered_predicted_anomaly,
    )

    fold_result = {
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
        "actual_validation_mean_lst_c_eval_only": actual_validation_mean,
        "annual_level_error_c_eval_only": float(
            level_forecast - actual_validation_mean
        ),
        "raw_predicted_anomaly_mean_c": float(raw_predicted_anomaly.mean()),
        "centered_predicted_anomaly_mean_c": float(
            centered_predicted_anomaly.mean()
        ),
        "anomaly_mae": anomaly_metrics["mae"],
        "anomaly_rmse": anomaly_metrics["rmse"],
        "anomaly_r2": anomaly_metrics["r2"],
        "centered_spatial_r2": anomaly_metrics["r2"],
        "mae": final_metrics["mae"],
        "rmse": final_metrics["rmse"],
        "r2": final_metrics["r2"],
        "mape_percent": final_metrics["mape_percent"],
        "mean_prediction_bias_pred_minus_actual_c": final_metrics[
            "mean_prediction_bias_pred_minus_actual_c"
        ],
        "fit_predict_seconds": fit_predict_seconds,
    }

    prediction_df = pd.DataFrame(
        {
            GRID_COL: val_df[GRID_COL].astype(str).to_numpy(),
            AREA_COL: val_df[AREA_COL].astype(str).to_numpy(),
            YEAR_COL: val_df[YEAR_COL].astype(int).to_numpy(),
            "config_id": config["config_id"],
            "model": MODEL_NAME,
            "strategy": STRATEGY_NAME,
            "actual_lst_c": y_val,
            "actual_centered_anomaly_c_eval_only": actual_centered_anomaly,
            "raw_predicted_anomaly_c": raw_predicted_anomaly,
            "centered_predicted_anomaly_c": centered_predicted_anomaly,
            "annual_level_forecast_c": np.full(
                len(val_df), level_forecast, dtype=float
            ),
            "predicted_lst_c": final_prediction,
            "prediction_error_c": final_prediction - y_val,
        }
    )

    return fold_result, prediction_df


# =============================================================================
# 4. LOAD + VERIFY STEP 01
# =============================================================================

section("STEP 03 - EXTRA TREES SUPPLEMENTARY EXPERIMENT")

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

for required_path in [DATA_PATH, POLICY_PATH, STEP01_AUDIT_PATH]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required Step 01 file not found: {required_path}")

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
    raise RuntimeError("Reserved/final years were unexpectedly loaded.")
if len(df) != EXPECTED_ROWS:
    raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, found {len(df)}.")
if df[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError("Unexpected unique grid count.")
if len(predictors) != EXPECTED_PREDICTORS:
    raise RuntimeError(
        f"Expected {EXPECTED_PREDICTORS} predictors, found {len(predictors)}."
    )
if predictors != policy.get("predictors", []):
    raise RuntimeError("Predictor order does not match the frozen Step 01 policy.")
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

fold_definitions = []
for validation_year in ROLLING_VALIDATION_YEARS:
    train_years = [y for y in DEVELOPMENT_YEARS if y < validation_year]
    fold_definitions.append(
        {"validation_year": validation_year, "train_years": train_years}
    )
    print(f"{min(train_years)}-{max(train_years)} -> {validation_year}")


# =============================================================================
# 6. DEVELOPMENT-ONLY SEARCH
# =============================================================================

section("2. EXTRA TREES DEVELOPMENT-ONLY SEARCH")

search_fold_rows = []

for config_index, config in enumerate(CANDIDATE_CONFIGS, start=1):
    print(
        f"\n--- {config['config_id']} "
        f"({config_index}/{len(CANDIDATE_CONFIGS)}) ---"
    )
    print(json.dumps(config, sort_keys=True))

    for fold in fold_definitions:
        validation_year = fold["validation_year"]
        train_df = df[df[YEAR_COL] < validation_year].copy()
        val_df = df[df[YEAR_COL] == validation_year].copy()

        fold_result, _ = evaluate_fold(
            config=config,
            train_df=train_df,
            val_df=val_df,
            predictors=predictors,
        )
        search_fold_rows.append(fold_result)

        print(
            f"{fold_result['train_start_year']}-{fold_result['train_end_year']} "
            f"-> {validation_year} | "
            f"MAE={fold_result['mae']:.4f} | "
            f"RMSE={fold_result['rmse']:.4f} | "
            f"R²={fold_result['r2']:.4f} | "
            f"Spatial R²={fold_result['centered_spatial_r2']:.4f}"
        )

search_fold_df = pd.DataFrame(search_fold_rows)
search_fold_df.to_csv(SEARCH_METRICS_PATH, index=False, encoding="utf-8")
print(f"\nSaved: {SEARCH_METRICS_PATH}")


# =============================================================================
# 7. SUMMARIZE + SELECT BEST CONFIG
# =============================================================================

section("3. EXTRA TREES SEARCH SUMMARY")

summary_df = (
    search_fold_df.groupby("config_id", as_index=False)
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
        mean_abs_prediction_bias_c=(
            "mean_prediction_bias_pred_minus_actual_c",
            lambda s: float(np.mean(np.abs(np.asarray(s, dtype=float)))),
        ),
        mean_fit_predict_seconds=("fit_predict_seconds", "mean"),
    )
)

config_df = pd.DataFrame(CANDIDATE_CONFIGS)
summary_df = summary_df.merge(
    config_df,
    on="config_id",
    how="left",
    validate="one_to_one",
)

# Primary objective: mean MAE. Tie-breakers: RMSE, then centered spatial R².
summary_df = summary_df.sort_values(
    ["mean_mae", "mean_rmse", "mean_centered_spatial_r2"],
    ascending=[True, True, False],
    kind="mergesort",
).reset_index(drop=True)
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
best_config = next(
    c for c in CANDIDATE_CONFIGS if c["config_id"] == best_config_id
)

print(f"\nSelected Extra Trees config: {best_config_id}")
print(json.dumps(best_config, indent=2))

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

print(f"Saved: {SEARCH_SUMMARY_PATH}")
print(f"Saved: {BEST_CONFIG_PATH}")


# =============================================================================
# 8. RE-RUN BEST CONFIG FOR FROZEN ROLLING PREDICTIONS
# =============================================================================

section("4. BEST EXTRA TREES CONFIG - FROZEN ROLLING PREDICTIONS")

best_fold_rows = []
best_prediction_frames = []

for fold in fold_definitions:
    validation_year = fold["validation_year"]
    train_df = df[df[YEAR_COL] < validation_year].copy()
    val_df = df[df[YEAR_COL] == validation_year].copy()

    fold_result, prediction_df = evaluate_fold(
        config=best_config,
        train_df=train_df,
        val_df=val_df,
        predictors=predictors,
    )
    best_fold_rows.append(fold_result)
    best_prediction_frames.append(prediction_df)

    print(
        f"{fold_result['train_start_year']}-{fold_result['train_end_year']} "
        f"-> {validation_year} | "
        f"MAE={fold_result['mae']:.4f} | "
        f"RMSE={fold_result['rmse']:.4f} | "
        f"R²={fold_result['r2']:.4f} | "
        f"Spatial R²={fold_result['centered_spatial_r2']:.4f}"
    )

best_fold_df = pd.DataFrame(best_fold_rows)
best_predictions_df = pd.concat(best_prediction_frames, ignore_index=True)

best_fold_df.to_csv(BEST_FOLD_METRICS_PATH, index=False, encoding="utf-8")
best_predictions_df.to_csv(BEST_PREDICTIONS_PATH, index=False, encoding="utf-8")

print(f"\nSaved: {BEST_FOLD_METRICS_PATH}")
print(f"Saved: {BEST_PREDICTIONS_PATH}")


# =============================================================================
# 9. FIT FULL DEVELOPMENT EXPERIMENTAL MODEL
# =============================================================================

section("5. FIT FULL 2018-2023 EXPERIMENTAL CANDIDATE")

X_full = df[predictors]
y_full_anomaly = build_training_anomaly(df)

final_experiment_model = build_extra_trees_model(best_config)
final_experiment_model.fit(X_full, y_full_anomaly)
joblib.dump(final_experiment_model, FINAL_EXPERIMENT_MODEL_PATH)

print(f"Saved: {FINAL_EXPERIMENT_MODEL_PATH}")
print("NOTE: this is an EXPERIMENTAL model, not a production replacement.")


# =============================================================================
# 10. DEVELOPMENT RESULT
# =============================================================================

section("6. EXTRA TREES DEVELOPMENT RESULT")

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

section("7. STRICT STEP 03 AUDIT")

expected_search_rows = len(CANDIDATE_CONFIGS) * EXPECTED_FOLDS
expected_prediction_rows = EXPECTED_GRIDS * EXPECTED_FOLDS

training_anomaly_year_means = [
    float(build_training_anomaly(df[df[YEAR_COL] == year]).mean())
    for year in DEVELOPMENT_YEARS
]

predicted_anomaly_year_means = (
    best_predictions_df.groupby(YEAR_COL)["centered_predicted_anomaly_c"]
    .mean()
    .to_numpy(dtype=float)
)

audit_rows = [
    {
        "check": "step01_audit",
        "status": "PASS",
        "details": "Step 01 fully PASS.",
    },
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
        "status": "PASS" if len(search_fold_df) == expected_search_rows else "FAIL",
        "details": f"actual={len(search_fold_df)}, expected={expected_search_rows}",
    },
    {
        "check": "best_fold_metric_rows",
        "status": "PASS" if len(best_fold_df) == EXPECTED_FOLDS else "FAIL",
        "details": f"actual={len(best_fold_df)}, expected={EXPECTED_FOLDS}",
    },
    {
        "check": "best_prediction_rows",
        "status": "PASS" if len(best_predictions_df) == expected_prediction_rows else "FAIL",
        "details": f"actual={len(best_predictions_df)}, expected={expected_prediction_rows}",
    },
    {
        "check": "training_anomaly_zero_mean_within_year",
        "status": "PASS"
        if all(abs(v) < 1e-10 for v in training_anomaly_year_means)
        else "FAIL",
        "details": str(training_anomaly_year_means),
    },
    {
        "check": "predicted_anomaly_recentered_without_target",
        "status": "PASS"
        if np.all(np.abs(predicted_anomaly_year_means) < 1e-10)
        else "FAIL",
        "details": str(predicted_anomaly_year_means.tolist()),
    },
    {
        "check": "2024_loaded_or_used",
        "status": "PASS",
        "details": "No. 2024 is not loaded by Step 03.",
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": "No. 2025 is not loaded by Step 03.",
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
            best_fold_df[["mae", "rmse", "r2", "centered_spatial_r2"]].to_numpy(
                dtype=float
            )
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
        "details": "No. Extra Trees model is experimental only.",
    },
]

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(AUDIT_PATH, index=False, encoding="utf-8")
print(audit_df.to_string(index=False))

if not (audit_df["status"] == "PASS").all():
    raise RuntimeError("Step 03 audit contains one or more FAIL rows.")


# =============================================================================
# 12. HASHES
# =============================================================================

section("8. REPRODUCIBILITY HASHES")

hash_paths = [
    DATA_PATH,
    SEARCH_METRICS_PATH,
    SEARCH_SUMMARY_PATH,
    BEST_FOLD_METRICS_PATH,
    BEST_PREDICTIONS_PATH,
    BEST_CONFIG_PATH,
    FINAL_EXPERIMENT_MODEL_PATH,
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


# =============================================================================
# 13. REPORT
# =============================================================================

section("9. SAVE STEP 03 REPORT")

report_lines = [
    "STEP 03 - EXTRA TREES SUPPLEMENTARY EXPERIMENT",
    "",
    f"Project root: {PROJECT_ROOT}",
    f"Experiment root: {EXPERIMENT_ROOT}",
    "",
    "Experiment scope: 2018-2023 development only",
    "2024 reserved validation: NOT LOADED",
    "2025 final lockbox: NOT LOADED",
    "Production pipeline: UNTOUCHED",
    "Production model: UNCHANGED",
    "",
    f"Predictors: {len(predictors)}",
    f"Candidate configurations: {len(CANDIDATE_CONFIGS)}",
    f"Best configuration: {best_config_id}",
    "",
    "Evaluation framework:",
    "- Strict expanding rolling-origin folds",
    "- Training target = within-training-year spatial anomaly",
    "- Validation predicted anomaly centered using predictions only",
    "- Annual level = mean of prior training-year annual LST means",
    "- Validation-year actual mean used for evaluation only",
    "",
    f"Mean MAE: {mean_mae:.6f} °C",
    f"Std MAE: {std_mae:.6f} °C",
    f"Mean RMSE: {mean_rmse:.6f} °C",
    f"Mean R2: {mean_r2:.6f}",
    f"Mean centered spatial R2: {mean_spatial_r2:.6f}",
    f"Negative-R2 fold count: {negative_r2_count}",
    f"Worst-year MAE: {worst_year_mae:.6f} °C",
    f"Minimum R2: {minimum_r2:.6f}",
    "",
    "Interpretation:",
    (
        "This is supplementary development evidence only. The frozen production "
        "LightGBM model is NOT replaced by this step."
    ),
    "",
    "STEP 03 STATUS: PASS",
]

REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
print(f"Saved: {REPORT_PATH}")


section("STEP 03 COMPLETED SUCCESSFULLY")

print("Extra Trees supplementary development experiment completed.")
print("2024 reserved validation remains untouched.")
print("2025 final lockbox remains untouched.")
print("Production model remains unchanged.")
print("Next: Step 04 - HistGradientBoosting supplementary experiment.")
