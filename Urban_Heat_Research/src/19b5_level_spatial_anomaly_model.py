"""
STEP 19B5 - CAUSAL ANNUAL-LEVEL + SPATIAL-ANOMALY DECOMPOSITION
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Test the strongest remaining hypothesis from Step 19A:

    LST(grid, year)
        =
    annual thermal level(year)
        +
    within-year spatial anomaly(grid, year)

The earlier diagnostics showed that the models can preserve useful spatial
pattern even when the absolute annual LST level is badly biased.

STRICT LEAKAGE RULE
-------------------
For validation year t:

1. Annual-level forecast uses only annual target means from years < t.
2. Spatial-anomaly model trains only on rows from years < t.
3. Training anomaly target:
       LST(row) - mean_LST(training_year_of_row)
4. Validation predicted anomalies are re-centered using their OWN prediction
   mean only.
5. The actual validation-year mean LST is NEVER used in an eligible prediction.

Eligible prediction:
       causal_forecast_level_t + centered_predicted_anomaly_i,t

ANNUAL-LEVEL RULES
------------------
1. persistence_level
   Next level = most recent historical annual mean.

2. prior_mean_level
   Next level = mean of all prior annual means.

3. linear_trend_level
   OLS trend fitted to prior annual means only.
   Requires at least 3 prior annual years.
   If fewer than 3 exist, predefined fallback = persistence_level.

SPATIAL-ANOMALY MODELS
----------------------
- Random Forest
- LightGBM

Both reuse their frozen Step 17 tuned hyperparameters.
No hyperparameter retuning occurs.

ORACLE DIAGNOSTIC
-----------------
A separate oracle prediction uses the ACTUAL validation-year mean LST plus
the centered predicted anomaly.

This deliberately leaks the current target and is therefore:
- diagnostic only
- saved separately
- excluded from eligible strategy summaries
- NOT usable for deployment, Step 20, or future forecasting

DATA POLICY
-----------
- 2018-2023 development rows only.
- 93 predictors.
- Strict expanding rolling-origin folds.
- 2024 reserved validation is NOT loaded.
- 2025 final lockbox is NOT loaded.
- No final model selection occurs.
"""

from pathlib import Path
import gc
import hashlib
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# SETTINGS
# =============================================================================

TARGET = "LST_C"
GRID_COL = "grid_id"
AREA_COL = "area_name"
YEAR_COL = "year"

RANDOM_STATE = 42

DEVELOPMENT_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]

EXPECTED_UNIQUE_GRIDS = 2194
EXPECTED_PREDICTOR_COUNT = 93

ANOMALY_MODELS = ["Random Forest", "LightGBM"]

LEVEL_STRATEGIES = [
    "persistence_level",
    "prior_mean_level",
    "linear_trend_level",
]

ELIGIBLE_STRATEGIES = [
    "random_forest_raw_reference",
    "lightgbm_raw_reference",
    "rf_anomaly_plus_persistence_level",
    "rf_anomaly_plus_prior_mean_level",
    "rf_anomaly_plus_linear_trend_level",
    "lgbm_anomaly_plus_persistence_level",
    "lgbm_anomaly_plus_prior_mean_level",
    "lgbm_anomaly_plus_linear_trend_level",
]

ORACLE_STRATEGIES = [
    "rf_anomaly_plus_oracle_actual_level",
    "lgbm_anomaly_plus_oracle_actual_level",
]

MIN_TREND_YEARS = 3
METRIC_TOLERANCE = 1e-9
ZERO_MEAN_TOLERANCE = 1e-10


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "tree"
    / "tree_train_2018_2023_ready.csv"
)

STEP17_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "17_best_tuned_rolling_predictions.csv"
)

STEP17_FOLD_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "17_best_config_fold_metrics.csv"
)

STEP17_PARAMS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "17_best_hyperparameters.json"
)

PREREQUISITE_AUDITS = [
    (
        PROJECT_ROOT / "outputs/reports/17_hyperparameter_tuning_audit.csv",
        "Step 17",
    ),
    (
        PROJECT_ROOT / "outputs/reports/19a_temporal_diagnosis_audit.csv",
        "Step 19A",
    ),
    (
        PROJECT_ROOT / "outputs/reports/19b1_temporal_bias_correction_audit.csv",
        "Step 19B1",
    ),
    (
        PROJECT_ROOT / "outputs/reports/19b2_stable_feature_subset_audit.csv",
        "Step 19B2",
    ),
    (
        PROJECT_ROOT / "outputs/reports/19b3_recent_year_weighting_audit.csv",
        "Step 19B3",
    ),
    (
        PROJECT_ROOT / "outputs/reports/19b4_temporal_ensemble_audit.csv",
        "Step 19B4",
    ),
]

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

for folder in [METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR, FIGURES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

ANNUAL_HISTORY_PATH = METRICS_DIR / "19b5_annual_level_history.csv"
LEVEL_FORECASTS_PATH = METRICS_DIR / "19b5_annual_level_forecasts.csv"
ANOMALY_METRICS_PATH = METRICS_DIR / "19b5_spatial_anomaly_fold_metrics.csv"
FOLD_METRICS_PATH = METRICS_DIR / "19b5_level_anomaly_fold_metrics.csv"
SUMMARY_PATH = METRICS_DIR / "19b5_level_anomaly_summary.csv"
ORACLE_METRICS_PATH = METRICS_DIR / "19b5_oracle_level_diagnostic.csv"
STRUCTURAL_DIAG_PATH = METRICS_DIR / "19b5_structural_diagnostic_summary.csv"

PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "19b5_level_anomaly_predictions.csv"
)

ORACLE_PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "19b5_oracle_level_predictions.csv"
)

REFERENCE_REPRO_PATH = (
    REPORTS_DIR / "19b5_step17_reference_reproduction_check.csv"
)

AUDIT_PATH = (
    REPORTS_DIR / "19b5_level_spatial_anomaly_audit.csv"
)

REPORT_PATH = (
    REPORTS_DIR / "19b5_level_spatial_anomaly_report.txt"
)

HASH_PATH = (
    REPORTS_DIR / "19b5_level_spatial_anomaly_hashes.csv"
)

FIG_LEVEL_PATH = FIGURES_DIR / "19b5_annual_level_forecast.png"
FIG_ANOMALY_R2_PATH = FIGURES_DIR / "19b5_spatial_anomaly_r2.png"
FIG_FINAL_R2_PATH = FIGURES_DIR / "19b5_final_lst_r2_by_year.png"
FIG_ORACLE_PATH = FIGURES_DIR / "19b5_raw_vs_oracle_r2.png"


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 128)
    print(title)
    print("=" * 128)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_audit_all_pass(path, label):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} audit: {path}")

    audit = pd.read_csv(path)

    if "status" not in audit.columns:
        raise RuntimeError(f"{label} audit has no status column")

    bad = audit[audit["status"].astype(str).str.upper() != "PASS"]

    if not bad.empty:
        print(bad.to_string(index=False))
        raise RuntimeError(f"{label} audit is not fully PASS")


def mape_percent(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.abs(y_true) > 1e-12

    if not np.any(valid):
        return np.nan

    return float(
        np.mean(
            np.abs(
                (y_true[valid] - y_pred[valid])
                / y_true[valid]
            )
        )
        * 100.0
    )


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    bias = float(np.mean(y_pred - y_true))

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": mape_percent(y_true, y_pred),
        "mean_prediction_bias_pred_minus_actual_c": bias,
        "abs_mean_prediction_bias_c": abs(bias),
    }


def clean_params(params):
    cleaned = {}

    for key, value in params.items():
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, np.bool_):
            value = bool(value)

        cleaned[str(key)] = value

    return cleaned


def build_anomaly_model(model_name, params):
    p = clean_params(params)

    if model_name == "Random Forest":
        return RandomForestRegressor(
            **p,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "LightGBM":
        return LGBMRegressor(
            **p,
            objective="regression",
            subsample_freq=1,
            importance_type="gain",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

    raise ValueError(f"Unsupported anomaly model: {model_name}")


def forecast_level(annual_history, validation_year, strategy):
    prior = (
        annual_history[
            annual_history["year"] < validation_year
        ]
        .sort_values("year")
        .copy()
    )

    if prior.empty:
        raise RuntimeError(
            f"No prior annual-level history for validation year {validation_year}"
        )

    latest_year = int(prior["year"].max())
    latest_level = float(
        prior.loc[
            prior["year"] == latest_year,
            "annual_mean_lst_c",
        ].iloc[0]
    )

    if strategy == "persistence_level":
        forecast = latest_level
        method_used = "persistence_level"
        fallback_used = False

    elif strategy == "prior_mean_level":
        forecast = float(prior["annual_mean_lst_c"].mean())
        method_used = "mean_of_all_prior_annual_levels"
        fallback_used = False

    elif strategy == "linear_trend_level":
        if len(prior) < MIN_TREND_YEARS:
            forecast = latest_level
            method_used = "persistence_fallback_insufficient_trend_years"
            fallback_used = True
        else:
            model = LinearRegression()
            model.fit(
                prior[["year"]].to_numpy(dtype=float),
                prior["annual_mean_lst_c"].to_numpy(dtype=float),
            )
            forecast = float(
                model.predict(
                    np.asarray([[validation_year]], dtype=float)
                )[0]
            )
            method_used = "ols_linear_trend_on_prior_annual_levels"
            fallback_used = False

    else:
        raise ValueError(f"Unknown level strategy: {strategy}")

    return {
        "forecast_level_c": forecast,
        "history_year_count": int(len(prior)),
        "history_start_year": int(prior["year"].min()),
        "history_end_year": latest_year,
        "max_source_year": latest_year,
        "method_used": method_used,
        "fallback_used": bool(fallback_used),
    }


# =============================================================================
# PREREQUISITES
# =============================================================================

section(
    "STEP 19B5 - CAUSAL ANNUAL-LEVEL + SPATIAL-ANOMALY DECOMPOSITION"
)

for path in [
    INPUT_PATH,
    STEP17_PREDICTIONS_PATH,
    STEP17_FOLD_METRICS_PATH,
    STEP17_PARAMS_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)

for path, label in PREREQUISITE_AUDITS:
    check_audit_all_pass(path, label)

print("PASS: all prerequisite audits are fully PASS.")
print("Step 19B5 uses development years only.")
print("2024 and 2025 are NOT loaded.")
print("No hyperparameter retuning will occur.")


# =============================================================================
# LOAD DEVELOPMENT DATA
# =============================================================================

section("1. LOAD 2018-2023 DEVELOPMENT DATA")

df = pd.read_csv(
    INPUT_PATH,
    dtype={GRID_COL: str},
)

required_columns = {
    TARGET,
    GRID_COL,
    AREA_COL,
    YEAR_COL,
}

missing_required = sorted(required_columns - set(df.columns))

if missing_required:
    raise RuntimeError(f"Missing required columns: {missing_required}")

years = sorted(
    pd.to_numeric(
        df[YEAR_COL],
        errors="coerce",
    )
    .astype(int)
    .unique()
    .tolist()
)

if years != DEVELOPMENT_YEARS:
    raise RuntimeError(
        f"Development years={years}; expected={DEVELOPMENT_YEARS}"
    )

if df[GRID_COL].nunique() != EXPECTED_UNIQUE_GRIDS:
    raise RuntimeError(
        f"Unique grids={df[GRID_COL].nunique()}; "
        f"expected={EXPECTED_UNIQUE_GRIDS}"
    )

predictors = [
    column
    for column in df.columns
    if column not in {
        TARGET,
        GRID_COL,
        AREA_COL,
    }
]

if len(predictors) != EXPECTED_PREDICTOR_COUNT:
    raise RuntimeError(
        f"Predictor count={len(predictors)}; "
        f"expected={EXPECTED_PREDICTOR_COUNT}"
    )

for column in predictors + [TARGET]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

if df[predictors + [TARGET]].isna().any().any():
    raise RuntimeError("Missing numeric values detected")

if not np.isfinite(
    df[predictors + [TARGET]].to_numpy(dtype=float)
).all():
    raise RuntimeError("Non-finite numeric values detected")

print(f"Rows               : {len(df):,}")
print(f"Columns            : {len(df.columns)}")
print(f"Unique grids       : {df[GRID_COL].nunique():,}")
print(f"Years              : {years}")
print(f"Predictors         : {len(predictors)}")
print("2024 data          : NOT LOADED")
print("2025 data          : NOT LOADED")


# =============================================================================
# LOAD FROZEN CONFIGURATION
# =============================================================================

section("2. LOAD FROZEN STEP 17 CONFIGURATION")

with STEP17_PARAMS_PATH.open("r", encoding="utf-8") as f:
    params_payload = json.load(f)

if "best_params" not in params_payload:
    raise RuntimeError("Step 17 hyperparameter JSON has no best_params key")

best_params = params_payload["best_params"]

for model_name in ANOMALY_MODELS:
    if model_name not in best_params:
        raise RuntimeError(f"Missing frozen params for {model_name}")

    print(
        f"{model_name}: "
        f"{json.dumps(clean_params(best_params[model_name]), sort_keys=True)}"
    )

print("\nPASS: RF and LightGBM frozen Step 17 hyperparameters loaded.")


# =============================================================================
# LOAD + REPRODUCE RAW REFERENCES
# =============================================================================

section("3. LOAD + REPRODUCE RAW STEP 17 REFERENCES")

step17_pred = pd.read_csv(
    STEP17_PREDICTIONS_PATH,
    dtype={GRID_COL: str},
)

step17_fold = pd.read_csv(
    STEP17_FOLD_METRICS_PATH
)

step17_pred["validation_year"] = pd.to_numeric(
    step17_pred["validation_year"],
    errors="coerce",
).astype(int)

required_prediction_columns = {
    "model",
    GRID_COL,
    AREA_COL,
    "validation_year",
    "actual_lst_c",
    "predicted_lst_c",
}

missing_prediction_columns = sorted(
    required_prediction_columns - set(step17_pred.columns)
)

if missing_prediction_columns:
    raise RuntimeError(
        f"Step 17 prediction file missing: {missing_prediction_columns}"
    )

reference_repro_rows = []

for model_name in ANOMALY_MODELS:
    for validation_year in VALIDATION_YEARS:
        subset = step17_pred[
            (step17_pred["model"] == model_name)
            & (step17_pred["validation_year"] == validation_year)
        ]

        if len(subset) != EXPECTED_UNIQUE_GRIDS:
            raise RuntimeError("Step 17 reference row count mismatch")

        current = regression_metrics(
            subset["actual_lst_c"],
            subset["predicted_lst_c"],
        )

        ref = step17_fold[
            (step17_fold["model"] == model_name)
            & (step17_fold["validation_year"] == validation_year)
        ]

        if len(ref) != 1:
            raise RuntimeError("Step 17 fold metric row mismatch")

        ref = ref.iloc[0]

        mae_diff = (
            current["mae"]
            - float(ref["validation_mae"])
        )

        rmse_diff = (
            current["rmse"]
            - float(ref["validation_rmse"])
        )

        r2_diff = (
            current["r2"]
            - float(ref["validation_r2"])
        )

        ok = (
            abs(mae_diff) <= METRIC_TOLERANCE
            and abs(rmse_diff) <= METRIC_TOLERANCE
            and abs(r2_diff) <= METRIC_TOLERANCE
        )

        reference_repro_rows.append(
            {
                "model": model_name,
                "validation_year": validation_year,
                "mae_difference": mae_diff,
                "rmse_difference": rmse_diff,
                "r2_difference": r2_diff,
                "status": "PASS" if ok else "FAIL",
            }
        )

reference_repro = pd.DataFrame(reference_repro_rows)

reference_repro.to_csv(
    REFERENCE_REPRO_PATH,
    index=False,
)

print(reference_repro.to_string(index=False))

if (reference_repro["status"] != "PASS").any():
    raise RuntimeError("Step 17 reference reproduction failed")

print("\nPASS: RF and LightGBM Step 17 references reproduced.")


# =============================================================================
# ANNUAL HISTORY
# =============================================================================

section("4. BUILD DEVELOPMENT ANNUAL THERMAL LEVEL HISTORY")

annual_history = (
    df.groupby(
        YEAR_COL,
        as_index=False,
    )
    .agg(
        annual_mean_lst_c=(TARGET, "mean"),
        annual_median_lst_c=(TARGET, "median"),
        annual_std_lst_c=(TARGET, "std"),
        rows=(TARGET, "size"),
    )
    .rename(columns={YEAR_COL: "year"})
)

annual_history["year"] = annual_history["year"].astype(int)

annual_history.to_csv(
    ANNUAL_HISTORY_PATH,
    index=False,
)

print(annual_history.to_string(index=False))


# =============================================================================
# CAUSAL ANNUAL-LEVEL FORECASTS
# =============================================================================

section("5. BUILD CAUSAL ANNUAL-LEVEL FORECASTS")

level_rows = []
level_lookup = {}

for validation_year in VALIDATION_YEARS:
    actual_level = float(
        annual_history.loc[
            annual_history["year"] == validation_year,
            "annual_mean_lst_c",
        ].iloc[0]
    )

    for strategy in LEVEL_STRATEGIES:
        result = forecast_level(
            annual_history,
            validation_year,
            strategy,
        )

        forecast_value = float(
            result["forecast_level_c"]
        )

        level_lookup[
            (validation_year, strategy)
        ] = forecast_value

        level_rows.append(
            {
                "validation_year": validation_year,
                "strategy": strategy,
                "history_year_count": result["history_year_count"],
                "history_start_year": result["history_start_year"],
                "history_end_year": result["history_end_year"],
                "max_source_year": result["max_source_year"],
                "method_used": result["method_used"],
                "fallback_used": result["fallback_used"],
                "forecast_level_c": forecast_value,
                "actual_validation_level_c_evaluation_only": actual_level,
                "level_error_forecast_minus_actual_c": (
                    forecast_value - actual_level
                ),
                "abs_level_error_c": abs(
                    forecast_value - actual_level
                ),
                "uses_current_validation_target_for_forecast": False,
            }
        )

        print(
            f"{validation_year} | {strategy:<24} | "
            f"history={result['history_start_year']}-"
            f"{result['history_end_year']} | "
            f"forecast={forecast_value:.4f} °C | "
            f"actual(eval only)={actual_level:.4f} °C | "
            f"error={forecast_value - actual_level:+.4f} °C | "
            f"method={result['method_used']}"
        )

level_forecasts = pd.DataFrame(level_rows)

level_forecasts.to_csv(
    LEVEL_FORECASTS_PATH,
    index=False,
)

if (
    level_forecasts["max_source_year"]
    >= level_forecasts["validation_year"]
).any():
    raise RuntimeError(
        "Annual-level forecast used current/future target year"
    )

print(f"\nSaved: {LEVEL_FORECASTS_PATH}")


# =============================================================================
# RAW REFERENCES + SPATIAL ANOMALY MODELS
# =============================================================================

section("6. STRICT ROLLING-ORIGIN SPATIAL-ANOMALY EVALUATION")

anomaly_metric_rows = []
eligible_fold_rows = []
eligible_prediction_frames = []
oracle_metric_rows = []
oracle_prediction_frames = []


# -------------------------------------------------------------------------
# Raw Step 17 references
# -------------------------------------------------------------------------

for model_name, strategy_name in [
    (
        "Random Forest",
        "random_forest_raw_reference",
    ),
    (
        "LightGBM",
        "lightgbm_raw_reference",
    ),
]:
    for validation_year in VALIDATION_YEARS:
        raw_subset = step17_pred[
            (step17_pred["model"] == model_name)
            & (step17_pred["validation_year"] == validation_year)
        ].copy()

        y_true = raw_subset["actual_lst_c"].to_numpy(dtype=float)
        y_pred = raw_subset["predicted_lst_c"].to_numpy(dtype=float)

        current = regression_metrics(
            y_true,
            y_pred,
        )

        actual_level = float(np.mean(y_true))
        predicted_level = float(np.mean(y_pred))

        eligible_fold_rows.append(
            {
                "strategy": strategy_name,
                "anomaly_model": model_name,
                "level_strategy": "implicit_raw_model_level",
                "validation_year": validation_year,
                "rows": len(raw_subset),
                "forecast_level_c": predicted_level,
                "actual_validation_level_c_evaluation_only": actual_level,
                "level_error_forecast_minus_actual_c": (
                    predicted_level - actual_level
                ),
                "abs_level_error_c": abs(
                    predicted_level - actual_level
                ),
                "mae": current["mae"],
                "rmse": current["rmse"],
                "r2": current["r2"],
                "mape_percent": current["mape_percent"],
                "mean_prediction_bias_pred_minus_actual_c": (
                    current["mean_prediction_bias_pred_minus_actual_c"]
                ),
                "abs_mean_prediction_bias_c": (
                    current["abs_mean_prediction_bias_c"]
                ),
                "negative_r2": bool(current["r2"] < 0),
                "eligible_for_strategy_comparison": True,
                "uses_current_validation_target_for_prediction": False,
            }
        )

        eligible_prediction_frames.append(
            pd.DataFrame(
                {
                    "strategy": strategy_name,
                    "anomaly_model": model_name,
                    "level_strategy": "implicit_raw_model_level",
                    "validation_year": validation_year,
                    GRID_COL: raw_subset[GRID_COL].astype(str).to_numpy(),
                    AREA_COL: raw_subset[AREA_COL].astype(str).to_numpy(),
                    "actual_lst_c": y_true,
                    "predicted_lst_c": y_pred,
                    "predicted_centered_spatial_anomaly_c": np.nan,
                    "forecast_level_c": predicted_level,
                    "actual_validation_level_c_evaluation_only": actual_level,
                    "oracle_target_leakage": False,
                }
            )
        )


# -------------------------------------------------------------------------
# Causal spatial-anomaly models
# -------------------------------------------------------------------------

for model_name in ANOMALY_MODELS:
    print(f"\n--- Spatial anomaly model: {model_name} ---")

    model_prefix = (
        "rf"
        if model_name == "Random Forest"
        else "lgbm"
    )

    for validation_year in VALIDATION_YEARS:
        train_df = df[
            df[YEAR_COL] < validation_year
        ].copy()

        val_df = df[
            df[YEAR_COL] == validation_year
        ].copy()

        train_years = sorted(
            train_df[YEAR_COL]
            .astype(int)
            .unique()
            .tolist()
        )

        # Historical training-year means only.
        train_year_mean_map = (
            train_df.groupby(YEAR_COL)[TARGET]
            .mean()
            .to_dict()
        )

        train_levels = (
            train_df[YEAR_COL]
            .map(train_year_mean_map)
            .to_numpy(dtype=float)
        )

        y_train_anomaly = (
            train_df[TARGET].to_numpy(dtype=float)
            - train_levels
        )

        max_abs_train_year_anomaly_mean = 0.0

        for train_year in train_years:
            mask = (
                train_df[YEAR_COL]
                .to_numpy(dtype=int)
                == train_year
            )

            current_mean = float(
                np.mean(
                    y_train_anomaly[mask]
                )
            )

            max_abs_train_year_anomaly_mean = max(
                max_abs_train_year_anomaly_mean,
                abs(current_mean),
            )

        if (
            max_abs_train_year_anomaly_mean
            > ZERO_MEAN_TOLERANCE
        ):
            raise RuntimeError(
                "Training anomaly target is not zero-mean within year"
            )

        X_train = train_df[predictors]
        X_val = val_df[predictors]

        model = build_anomaly_model(
            model_name,
            best_params[model_name],
        )

        model.fit(
            X_train,
            y_train_anomaly,
        )

        pred_anomaly_raw = np.asarray(
            model.predict(X_val),
            dtype=float,
        )

        # Prediction-only centering; no validation target used here.
        pred_anomaly_centered = (
            pred_anomaly_raw
            - float(np.mean(pred_anomaly_raw))
        )

        centered_pred_mean = float(
            np.mean(pred_anomaly_centered)
        )

        if abs(centered_pred_mean) > ZERO_MEAN_TOLERANCE:
            raise RuntimeError(
                "Centered predicted anomalies are not zero mean"
            )

        actual_level = float(
            val_df[TARGET].mean()
        )

        actual_anomaly = (
            val_df[TARGET].to_numpy(dtype=float)
            - actual_level
        )

        anomaly_metrics = regression_metrics(
            actual_anomaly,
            pred_anomaly_centered,
        )

        anomaly_metric_rows.append(
            {
                "model": model_name,
                "validation_year": validation_year,
                "train_start_year": min(train_years),
                "train_end_year": max(train_years),
                "training_year_count": len(train_years),
                "train_rows": len(train_df),
                "validation_rows": len(val_df),
                "predictor_count": len(predictors),
                "max_abs_training_year_anomaly_mean_c": (
                    max_abs_train_year_anomaly_mean
                ),
                "raw_predicted_anomaly_mean_c": float(
                    np.mean(pred_anomaly_raw)
                ),
                "centered_predicted_anomaly_mean_c": (
                    centered_pred_mean
                ),
                "actual_validation_anomaly_mean_c_evaluation_only": float(
                    np.mean(actual_anomaly)
                ),
                "anomaly_mae_c": anomaly_metrics["mae"],
                "anomaly_rmse_c": anomaly_metrics["rmse"],
                "anomaly_r2": anomaly_metrics["r2"],
                "anomaly_negative_r2": bool(
                    anomaly_metrics["r2"] < 0
                ),
                "validation_target_used_for_anomaly_training": False,
                "validation_target_used_for_prediction_recentering": False,
            }
        )

        print(
            f"{train_years[0]}-{train_years[-1]} -> "
            f"{validation_year} | "
            f"anomaly MAE={anomaly_metrics['mae']:.4f} | "
            f"RMSE={anomaly_metrics['rmse']:.4f} | "
            f"R²={anomaly_metrics['r2']:.4f} | "
            f"raw mean={np.mean(pred_anomaly_raw):+.4f} | "
            f"centered mean={centered_pred_mean:+.12f}"
        )

        # Eligible causal level + anomaly strategies.
        for level_strategy in LEVEL_STRATEGIES:
            forecast_level_c = float(
                level_lookup[
                    (validation_year, level_strategy)
                ]
            )

            final_prediction = (
                forecast_level_c
                + pred_anomaly_centered
            )

            final_metrics = regression_metrics(
                val_df[TARGET],
                final_prediction,
            )

            strategy_name = (
                f"{model_prefix}_anomaly_plus_{level_strategy}"
            )

            eligible_fold_rows.append(
                {
                    "strategy": strategy_name,
                    "anomaly_model": model_name,
                    "level_strategy": level_strategy,
                    "validation_year": validation_year,
                    "rows": len(val_df),
                    "forecast_level_c": forecast_level_c,
                    "actual_validation_level_c_evaluation_only": actual_level,
                    "level_error_forecast_minus_actual_c": (
                        forecast_level_c - actual_level
                    ),
                    "abs_level_error_c": abs(
                        forecast_level_c - actual_level
                    ),
                    "mae": final_metrics["mae"],
                    "rmse": final_metrics["rmse"],
                    "r2": final_metrics["r2"],
                    "mape_percent": final_metrics["mape_percent"],
                    "mean_prediction_bias_pred_minus_actual_c": (
                        final_metrics[
                            "mean_prediction_bias_pred_minus_actual_c"
                        ]
                    ),
                    "abs_mean_prediction_bias_c": (
                        final_metrics[
                            "abs_mean_prediction_bias_c"
                        ]
                    ),
                    "negative_r2": bool(
                        final_metrics["r2"] < 0
                    ),
                    "eligible_for_strategy_comparison": True,
                    "uses_current_validation_target_for_prediction": False,
                }
            )

            eligible_prediction_frames.append(
                pd.DataFrame(
                    {
                        "strategy": strategy_name,
                        "anomaly_model": model_name,
                        "level_strategy": level_strategy,
                        "validation_year": validation_year,
                        GRID_COL: val_df[GRID_COL].astype(str).to_numpy(),
                        AREA_COL: val_df[AREA_COL].astype(str).to_numpy(),
                        "actual_lst_c": (
                            val_df[TARGET].to_numpy(dtype=float)
                        ),
                        "predicted_lst_c": final_prediction,
                        "predicted_centered_spatial_anomaly_c": (
                            pred_anomaly_centered
                        ),
                        "forecast_level_c": forecast_level_c,
                        "actual_validation_level_c_evaluation_only": (
                            actual_level
                        ),
                        "oracle_target_leakage": False,
                    }
                )
            )

        # Oracle diagnostic only.
        oracle_prediction = (
            actual_level
            + pred_anomaly_centered
        )

        oracle_metrics = regression_metrics(
            val_df[TARGET],
            oracle_prediction,
        )

        oracle_strategy = (
            f"{model_prefix}_anomaly_plus_oracle_actual_level"
        )

        oracle_metric_rows.append(
            {
                "strategy": oracle_strategy,
                "anomaly_model": model_name,
                "validation_year": validation_year,
                "actual_level_used_c": actual_level,
                "mae": oracle_metrics["mae"],
                "rmse": oracle_metrics["rmse"],
                "r2": oracle_metrics["r2"],
                "mape_percent": oracle_metrics["mape_percent"],
                "mean_prediction_bias_pred_minus_actual_c": (
                    oracle_metrics[
                        "mean_prediction_bias_pred_minus_actual_c"
                    ]
                ),
                "eligible_for_strategy_comparison": False,
                "uses_current_validation_target_for_prediction": True,
            }
        )

        oracle_prediction_frames.append(
            pd.DataFrame(
                {
                    "strategy": oracle_strategy,
                    "anomaly_model": model_name,
                    "validation_year": validation_year,
                    GRID_COL: val_df[GRID_COL].astype(str).to_numpy(),
                    AREA_COL: val_df[AREA_COL].astype(str).to_numpy(),
                    "actual_lst_c": (
                        val_df[TARGET].to_numpy(dtype=float)
                    ),
                    "predicted_lst_c": oracle_prediction,
                    "predicted_centered_spatial_anomaly_c": (
                        pred_anomaly_centered
                    ),
                    "actual_level_used_c": actual_level,
                    "oracle_target_leakage": True,
                }
            )
        )

        del model
        gc.collect()


anomaly_metrics_df = pd.DataFrame(
    anomaly_metric_rows
)

fold_metrics = pd.DataFrame(
    eligible_fold_rows
)

eligible_predictions = pd.concat(
    eligible_prediction_frames,
    ignore_index=True,
)

oracle_metrics_df = pd.DataFrame(
    oracle_metric_rows
)

oracle_predictions = pd.concat(
    oracle_prediction_frames,
    ignore_index=True,
)

anomaly_metrics_df.to_csv(
    ANOMALY_METRICS_PATH,
    index=False,
)

fold_metrics.to_csv(
    FOLD_METRICS_PATH,
    index=False,
)

eligible_predictions.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

oracle_metrics_df.to_csv(
    ORACLE_METRICS_PATH,
    index=False,
)

oracle_predictions.to_csv(
    ORACLE_PREDICTIONS_PATH,
    index=False,
)

print(f"\nSaved: {ANOMALY_METRICS_PATH}")
print(f"Saved: {FOLD_METRICS_PATH}")
print(f"Saved: {PREDICTIONS_PATH}")
print(f"Saved: {ORACLE_METRICS_PATH}")
print(f"Saved: {ORACLE_PREDICTIONS_PATH}")


# =============================================================================
# ELIGIBLE STRATEGY SUMMARY
# =============================================================================

section("7. ELIGIBLE STRATEGY SUMMARY")

summary_rows = []

for strategy in ELIGIBLE_STRATEGIES:
    subset = fold_metrics[
        fold_metrics["strategy"] == strategy
    ]

    if len(subset) != len(VALIDATION_YEARS):
        raise RuntimeError(
            f"{strategy} does not contain all five validation folds"
        )

    summary_rows.append(
        {
            "strategy": strategy,
            "anomaly_model": (
                subset["anomaly_model"].iloc[0]
            ),
            "level_strategy": (
                subset["level_strategy"].iloc[0]
            ),
            "mean_mae": float(subset["mae"].mean()),
            "std_mae": float(subset["mae"].std(ddof=0)),
            "mean_rmse": float(subset["rmse"].mean()),
            "mean_r2": float(subset["r2"].mean()),
            "negative_r2_count": int(
                subset["negative_r2"].sum()
            ),
            "worst_year_mae": float(
                subset["mae"].max()
            ),
            "minimum_r2": float(
                subset["r2"].min()
            ),
            "mean_abs_level_error_c": float(
                subset["abs_level_error_c"].mean()
            ),
            "mean_abs_prediction_bias_c": float(
                subset["abs_mean_prediction_bias_c"].mean()
            ),
        }
    )

summary = pd.DataFrame(
    summary_rows
)

summary["descriptive_mean_mae_rank"] = (
    summary["mean_mae"]
    .rank(
        method="min",
        ascending=True,
    )
    .astype(int)
)

summary = summary.sort_values(
    [
        "descriptive_mean_mae_rank",
        "worst_year_mae",
        "mean_rmse",
    ]
).reset_index(drop=True)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

print(summary.to_string(index=False))

print(
    "\nNOTE: this is development evidence only. "
    "No final strategy is selected."
)


# =============================================================================
# ORACLE DIAGNOSTIC
# =============================================================================

section("8. ORACLE CURRENT-YEAR LEVEL DIAGNOSTIC - INELIGIBLE")

print(
    "WARNING: oracle rows use the ACTUAL validation-year mean LST."
)
print(
    "They are diagnostic only and cannot be used for future forecasting."
)

print(
    oracle_metrics_df.to_string(index=False)
)

oracle_summary = (
    oracle_metrics_df.groupby(
        [
            "strategy",
            "anomaly_model",
        ],
        as_index=False,
    )
    .agg(
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_r2=("r2", "mean"),
        minimum_r2=("r2", "min"),
    )
)

print("\nOracle summary:")
print(
    oracle_summary.to_string(index=False)
)


# =============================================================================
# STRUCTURAL DIAGNOSTIC
# =============================================================================

section("9. STRUCTURAL DIAGNOSTIC SUMMARY")

diagnostic_rows = []

for model_name in ANOMALY_MODELS:
    prefix = (
        "rf"
        if model_name == "Random Forest"
        else "lgbm"
    )

    raw_strategy = (
        "random_forest_raw_reference"
        if model_name == "Random Forest"
        else "lightgbm_raw_reference"
    )

    raw = fold_metrics[
        fold_metrics["strategy"] == raw_strategy
    ]

    anomaly = anomaly_metrics_df[
        anomaly_metrics_df["model"] == model_name
    ]

    oracle = oracle_metrics_df[
        oracle_metrics_df["strategy"]
        == f"{prefix}_anomaly_plus_oracle_actual_level"
    ]

    row = {
        "model": model_name,
        "raw_mean_mae": float(raw["mae"].mean()),
        "raw_mean_r2": float(raw["r2"].mean()),
        "mean_spatial_anomaly_mae_c": float(
            anomaly["anomaly_mae_c"].mean()
        ),
        "mean_spatial_anomaly_r2": float(
            anomaly["anomaly_r2"].mean()
        ),
        "oracle_level_mean_mae_ineligible": float(
            oracle["mae"].mean()
        ),
        "oracle_level_mean_r2_ineligible": float(
            oracle["r2"].mean()
        ),
    }

    for level_strategy in LEVEL_STRATEGIES:
        s = fold_metrics[
            fold_metrics["strategy"]
            == f"{prefix}_anomaly_plus_{level_strategy}"
        ]

        row[
            f"{level_strategy}_mean_mae"
        ] = float(s["mae"].mean())

        row[
            f"{level_strategy}_mean_r2"
        ] = float(s["r2"].mean())

        row[
            f"{level_strategy}_mean_abs_level_error_c"
        ] = float(
            s["abs_level_error_c"].mean()
        )

    diagnostic_rows.append(row)

diagnostic_table = pd.DataFrame(
    diagnostic_rows
)

diagnostic_table.to_csv(
    STRUCTURAL_DIAG_PATH,
    index=False,
)

print(
    diagnostic_table.to_string(index=False)
)


# =============================================================================
# FIGURES
# =============================================================================

section("10. FIGURES")

# Annual level forecast figure
fig, ax = plt.subplots(figsize=(11, 7))

actual = annual_history[
    annual_history["year"].isin(VALIDATION_YEARS)
]

ax.plot(
    actual["year"],
    actual["annual_mean_lst_c"],
    marker="o",
    label="Actual annual mean (evaluation only)",
)

for strategy in LEVEL_STRATEGIES:
    sub = (
        level_forecasts[
            level_forecasts["strategy"] == strategy
        ]
        .sort_values("validation_year")
    )

    ax.plot(
        sub["validation_year"],
        sub["forecast_level_c"],
        marker="o",
        label=strategy,
    )

ax.set_xticks(VALIDATION_YEARS)
ax.set_xlabel("Validation year")
ax.set_ylabel("Annual thermal level (°C)")
ax.set_title("Step 19B5 - Causal Annual-Level Forecast")
ax.grid(alpha=0.2)
ax.legend()
plt.tight_layout()
plt.savefig(
    FIG_LEVEL_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Spatial anomaly R²
fig, ax = plt.subplots(figsize=(10, 6))

for model_name in ANOMALY_MODELS:
    sub = (
        anomaly_metrics_df[
            anomaly_metrics_df["model"] == model_name
        ]
        .sort_values("validation_year")
    )

    ax.plot(
        sub["validation_year"],
        sub["anomaly_r2"],
        marker="o",
        label=model_name,
    )

ax.axhline(0, linewidth=1)
ax.set_xticks(VALIDATION_YEARS)
ax.set_xlabel("Validation year")
ax.set_ylabel("Spatial anomaly R²")
ax.set_title("Step 19B5 - Within-Year Spatial-Anomaly Skill")
ax.grid(alpha=0.2)
ax.legend()
plt.tight_layout()
plt.savefig(
    FIG_ANOMALY_R2_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Eligible final R²
fig, ax = plt.subplots(figsize=(12, 7))

for strategy in ELIGIBLE_STRATEGIES:
    sub = (
        fold_metrics[
            fold_metrics["strategy"] == strategy
        ]
        .sort_values("validation_year")
    )

    ax.plot(
        sub["validation_year"],
        sub["r2"],
        marker="o",
        label=strategy,
    )

ax.axhline(0, linewidth=1)
ax.set_xticks(VALIDATION_YEARS)
ax.set_xlabel("Validation year")
ax.set_ylabel("Final LST R²")
ax.set_title("Step 19B5 - Raw vs Causal Level+Anomaly")
ax.grid(alpha=0.2)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(
    FIG_FINAL_R2_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Raw vs oracle
fig, ax = plt.subplots(figsize=(11, 7))

for model_name in ANOMALY_MODELS:
    if model_name == "Random Forest":
        raw_strategy = "random_forest_raw_reference"
        oracle_strategy = "rf_anomaly_plus_oracle_actual_level"
    else:
        raw_strategy = "lightgbm_raw_reference"
        oracle_strategy = "lgbm_anomaly_plus_oracle_actual_level"

    raw = (
        fold_metrics[
            fold_metrics["strategy"] == raw_strategy
        ]
        .sort_values("validation_year")
    )

    oracle = (
        oracle_metrics_df[
            oracle_metrics_df["strategy"] == oracle_strategy
        ]
        .sort_values("validation_year")
    )

    ax.plot(
        raw["validation_year"],
        raw["r2"],
        marker="o",
        label=f"{model_name} raw",
    )

    ax.plot(
        oracle["validation_year"],
        oracle["r2"],
        marker="o",
        label=f"{model_name} oracle level",
    )

ax.axhline(0, linewidth=1)
ax.set_xticks(VALIDATION_YEARS)
ax.set_xlabel("Validation year")
ax.set_ylabel("R²")
ax.set_title("Step 19B5 - Raw vs Oracle-Level Diagnostic")
ax.grid(alpha=0.2)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(
    FIG_ORACLE_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

for figure_path in [
    FIG_LEVEL_PATH,
    FIG_ANOMALY_R2_PATH,
    FIG_FINAL_R2_PATH,
    FIG_ORACLE_PATH,
]:
    print(f"Saved: {figure_path}")


# =============================================================================
# STRICT AUDIT
# =============================================================================

section("11. STRICT STEP 19B5 AUDIT")

expected_reference_rows = (
    len(ANOMALY_MODELS)
    * len(VALIDATION_YEARS)
)

expected_level_rows = (
    len(LEVEL_STRATEGIES)
    * len(VALIDATION_YEARS)
)

expected_anomaly_rows = (
    len(ANOMALY_MODELS)
    * len(VALIDATION_YEARS)
)

expected_fold_rows = (
    len(ELIGIBLE_STRATEGIES)
    * len(VALIDATION_YEARS)
)

expected_prediction_rows = (
    expected_fold_rows
    * EXPECTED_UNIQUE_GRIDS
)

expected_oracle_rows = (
    len(ORACLE_STRATEGIES)
    * len(VALIDATION_YEARS)
)

expected_oracle_prediction_rows = (
    expected_oracle_rows
    * EXPECTED_UNIQUE_GRIDS
)

expected_summary_rows = len(
    ELIGIBLE_STRATEGIES
)

level_prior_only = bool(
    (
        level_forecasts["max_source_year"]
        < level_forecasts["validation_year"]
    ).all()
)

level_no_current_target = bool(
    not level_forecasts[
        "uses_current_validation_target_for_forecast"
    ]
    .astype(bool)
    .any()
)

anomaly_training_safe = bool(
    not anomaly_metrics_df[
        "validation_target_used_for_anomaly_training"
    ]
    .astype(bool)
    .any()
)

anomaly_recentering_safe = bool(
    not anomaly_metrics_df[
        "validation_target_used_for_prediction_recentering"
    ]
    .astype(bool)
    .any()
)

train_anomaly_zero_mean = bool(
    (
        anomaly_metrics_df[
            "max_abs_training_year_anomaly_mean_c"
        ].abs()
        <= ZERO_MEAN_TOLERANCE
    ).all()
)

pred_anomaly_zero_mean = bool(
    (
        anomaly_metrics_df[
            "centered_predicted_anomaly_mean_c"
        ].abs()
        <= ZERO_MEAN_TOLERANCE
    ).all()
)

eligible_target_safe = bool(
    not fold_metrics[
        "uses_current_validation_target_for_prediction"
    ]
    .astype(bool)
    .any()
)

oracle_ineligible = bool(
    (
        oracle_metrics_df[
            "eligible_for_strategy_comparison"
        ]
        == False
    ).all()
)

oracle_flagged = bool(
    oracle_metrics_df[
        "uses_current_validation_target_for_prediction"
    ]
    .astype(bool)
    .all()
)

all_eligible_metrics_finite = bool(
    np.isfinite(
        fold_metrics[
            [
                "forecast_level_c",
                "actual_validation_level_c_evaluation_only",
                "level_error_forecast_minus_actual_c",
                "abs_level_error_c",
                "mae",
                "rmse",
                "r2",
                "mape_percent",
                "mean_prediction_bias_pred_minus_actual_c",
                "abs_mean_prediction_bias_c",
            ]
        ]
        .to_numpy(dtype=float)
    ).all()
)

all_anomaly_metrics_finite = bool(
    np.isfinite(
        anomaly_metrics_df[
            [
                "anomaly_mae_c",
                "anomaly_rmse_c",
                "anomaly_r2",
            ]
        ]
        .to_numpy(dtype=float)
    ).all()
)

all_oracle_metrics_finite = bool(
    np.isfinite(
        oracle_metrics_df[
            [
                "mae",
                "rmse",
                "r2",
                "mape_percent",
            ]
        ]
        .to_numpy(dtype=float)
    ).all()
)

eligible_strategy_set_ok = bool(
    sorted(
        fold_metrics["strategy"]
        .unique()
        .tolist()
    )
    == sorted(ELIGIBLE_STRATEGIES)
)

oracle_strategy_set_ok = bool(
    sorted(
        oracle_metrics_df["strategy"]
        .unique()
        .tolist()
    )
    == sorted(ORACLE_STRATEGIES)
)

audit_rows = [
    {
        "check": "all_prerequisite_audits",
        "status": "PASS",
        "details": "Step 17 and Steps 19A-19B4 were loaded and fully PASS.",
    },
    {
        "check": "development_years",
        "status": "PASS" if years == DEVELOPMENT_YEARS else "FAIL",
        "details": str(years),
    },
    {
        "check": "predictor_count",
        "status": (
            "PASS"
            if len(predictors) == EXPECTED_PREDICTOR_COUNT
            else "FAIL"
        ),
        "details": (
            f"actual={len(predictors)}, expected={EXPECTED_PREDICTOR_COUNT}"
        ),
    },
    {
        "check": "step17_reference_reproduction",
        "status": (
            "PASS"
            if (
                len(reference_repro) == expected_reference_rows
                and (reference_repro["status"] == "PASS").all()
            )
            else "FAIL"
        ),
        "details": (
            f"actual={len(reference_repro)}, expected={expected_reference_rows}"
        ),
    },
    {
        "check": "annual_level_forecast_rows",
        "status": (
            "PASS"
            if len(level_forecasts) == expected_level_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(level_forecasts)}, expected={expected_level_rows}"
        ),
    },
    {
        "check": "annual_level_prior_years_only",
        "status": "PASS" if level_prior_only else "FAIL",
        "details": (
            "Every eligible annual-level forecast uses annual target history "
            "strictly earlier than its validation year."
        ),
    },
    {
        "check": "annual_level_current_target_not_used",
        "status": "PASS" if level_no_current_target else "FAIL",
        "details": (
            "Validation-year actual level appears only as an evaluation field."
        ),
    },
    {
        "check": "spatial_anomaly_metric_rows",
        "status": (
            "PASS"
            if len(anomaly_metrics_df) == expected_anomaly_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(anomaly_metrics_df)}, expected={expected_anomaly_rows}"
        ),
    },
    {
        "check": "training_anomaly_zero_mean_within_year",
        "status": "PASS" if train_anomaly_zero_mean else "FAIL",
        "details": (
            "Historical anomaly targets are centered separately within "
            "each training year."
        ),
    },
    {
        "check": "anomaly_training_validation_target_not_used",
        "status": "PASS" if anomaly_training_safe else "FAIL",
        "details": "Anomaly model fitting uses training years only.",
    },
    {
        "check": "predicted_anomaly_recentered_without_target",
        "status": (
            "PASS"
            if pred_anomaly_zero_mean and anomaly_recentering_safe
            else "FAIL"
        ),
        "details": (
            "Validation predicted anomalies are centered using their "
            "prediction mean only."
        ),
    },
    {
        "check": "eligible_strategy_set",
        "status": "PASS" if eligible_strategy_set_ok else "FAIL",
        "details": str(
            sorted(
                fold_metrics["strategy"]
                .unique()
                .tolist()
            )
        ),
    },
    {
        "check": "eligible_fold_metric_rows",
        "status": (
            "PASS"
            if len(fold_metrics) == expected_fold_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(fold_metrics)}, expected={expected_fold_rows}"
        ),
    },
    {
        "check": "eligible_prediction_rows",
        "status": (
            "PASS"
            if len(eligible_predictions) == expected_prediction_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(eligible_predictions)}, expected={expected_prediction_rows}"
        ),
    },
    {
        "check": "eligible_predictions_current_target_not_used",
        "status": "PASS" if eligible_target_safe else "FAIL",
        "details": (
            "No eligible final prediction uses validation-year target information."
        ),
    },
    {
        "check": "oracle_strategy_set",
        "status": "PASS" if oracle_strategy_set_ok else "FAIL",
        "details": str(
            sorted(
                oracle_metrics_df["strategy"]
                .unique()
                .tolist()
            )
        ),
    },
    {
        "check": "oracle_metric_rows",
        "status": (
            "PASS"
            if len(oracle_metrics_df) == expected_oracle_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(oracle_metrics_df)}, expected={expected_oracle_rows}"
        ),
    },
    {
        "check": "oracle_prediction_rows",
        "status": (
            "PASS"
            if len(oracle_predictions) == expected_oracle_prediction_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(oracle_predictions)}, "
            f"expected={expected_oracle_prediction_rows}"
        ),
    },
    {
        "check": "oracle_ineligible_and_target_leaking_flagged",
        "status": (
            "PASS"
            if oracle_ineligible and oracle_flagged
            else "FAIL"
        ),
        "details": (
            "Oracle current-year level is explicitly target-leaking "
            "and excluded from eligible comparisons."
        ),
    },
    {
        "check": "eligible_summary_rows",
        "status": (
            "PASS"
            if len(summary) == expected_summary_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(summary)}, expected={expected_summary_rows}"
        ),
    },
    {
        "check": "eligible_metrics_finite",
        "status": "PASS" if all_eligible_metrics_finite else "FAIL",
        "details": str(all_eligible_metrics_finite),
    },
    {
        "check": "anomaly_metrics_finite",
        "status": "PASS" if all_anomaly_metrics_finite else "FAIL",
        "details": str(all_anomaly_metrics_finite),
    },
    {
        "check": "oracle_metrics_finite",
        "status": "PASS" if all_oracle_metrics_finite else "FAIL",
        "details": str(all_oracle_metrics_finite),
    },
    {
        "check": "2024_loaded_or_used",
        "status": "PASS",
        "details": "No. 2024 was not loaded by Step 19B5.",
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": "No. 2025 was not loaded by Step 19B5.",
    },
    {
        "check": "hyperparameters_retuned",
        "status": "PASS",
        "details": (
            "No. Frozen Step 17 RF/LightGBM hyperparameters were reused."
        ),
    },
    {
        "check": "broad_annual_level_search_performed",
        "status": "PASS",
        "details": (
            "No. Only three predeclared simple annual-level rules were tested."
        ),
    },
    {
        "check": "final_model_selected",
        "status": "PASS",
        "details": (
            "No. Step 19B5 is a development structural experiment only."
        ),
    },
]

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
)

print(audit_df.to_string(index=False))

if (audit_df["status"] != "PASS").any():
    raise RuntimeError(
        "One or more Step 19B5 audit checks failed"
    )


# =============================================================================
# REPORT
# =============================================================================

section("12. SAVE STEP 19B5 REPORT")

report_lines = [
    "STEP 19B5 - CAUSAL ANNUAL-LEVEL + SPATIAL-ANOMALY DECOMPOSITION",
    "=" * 116,
    "",
    "PURPOSE",
    "-" * 116,
    (
        "Test whether weak temporal generalization is driven mainly by "
        "difficulty forecasting the annual thermal level while spatial "
        "heat pattern remains learnable."
    ),
    "",
    "FORMULATION",
    "-" * 116,
    (
        "LST(grid, year) = annual thermal level(year) "
        "+ spatial anomaly(grid, year)"
    ),
    "",
    "DATA POLICY",
    "-" * 116,
    "Development rows: 2018-2023 only.",
    "2024 reserved validation: NOT loaded.",
    "2025 final lockbox: NOT loaded.",
    "",
    "ANNUAL LEVEL",
    "-" * 116,
    "Eligible level forecasts use prior annual target means only.",
    "Rules: persistence, prior-mean, and simple OLS trend.",
    (
        "Linear trend requires at least "
        f"{MIN_TREND_YEARS} prior annual years; "
        "otherwise it falls back to persistence."
    ),
    "",
    "SPATIAL ANOMALY",
    "-" * 116,
    (
        "Random Forest and LightGBM reuse frozen Step 17 hyperparameters."
    ),
    (
        "Training anomaly targets are centered separately inside "
        "each historical training year."
    ),
    (
        "Validation predicted anomalies are re-centered using "
        "prediction mean only."
    ),
    "",
    "ELIGIBLE STRATEGY SUMMARY",
    "-" * 116,
    summary.to_string(index=False),
    "",
    "SPATIAL ANOMALY METRICS",
    "-" * 116,
    anomaly_metrics_df.to_string(index=False),
    "",
    "ORACLE DIAGNOSTIC - INELIGIBLE",
    "-" * 116,
    (
        "The oracle uses the actual current validation-year mean LST "
        "and is target-leaking by design."
    ),
    oracle_summary.to_string(index=False),
    "",
    "STRUCTURAL DIAGNOSTIC SUMMARY",
    "-" * 116,
    diagnostic_table.to_string(index=False),
    "",
    "INTERPRETATION",
    "-" * 116,
    (
        "If anomaly R² stays positive while causal final-LST R² remains weak "
        "because annual-level forecast error is large, annual thermal level "
        "is supported as a major operational bottleneck."
    ),
    (
        "If oracle-level predictions are strong but causal level forecasts "
        "are weak, the decomposition is informative but does not yet solve "
        "future absolute-LST forecasting."
    ),
    (
        "If anomaly skill itself is weak, the annual-level hypothesis alone "
        "is insufficient."
    ),
    "",
    "NO FINAL MODEL SELECTION",
    "-" * 116,
    (
        "Step 19B5 remains development evidence only. "
        "Step 19C must compare legitimate Step 19 strategies before 2024."
    ),
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {REPORT_PATH}")
print(f"Saved: {AUDIT_PATH}")


# =============================================================================
# HASHES
# =============================================================================

section("13. REPRODUCIBILITY HASHES")

hash_files = [
    INPUT_PATH,
    STEP17_PREDICTIONS_PATH,
    STEP17_FOLD_METRICS_PATH,
    STEP17_PARAMS_PATH,
    ANNUAL_HISTORY_PATH,
    LEVEL_FORECASTS_PATH,
    ANOMALY_METRICS_PATH,
    FOLD_METRICS_PATH,
    SUMMARY_PATH,
    ORACLE_METRICS_PATH,
    STRUCTURAL_DIAG_PATH,
    PREDICTIONS_PATH,
    ORACLE_PREDICTIONS_PATH,
    REFERENCE_REPRO_PATH,
    AUDIT_PATH,
    REPORT_PATH,
]

hash_rows = []

for path in hash_files:
    digest = sha256_file(path)

    hash_rows.append(
        {
            "file": str(path),
            "sha256": digest,
        }
    )

    print(f"{path.name}: {digest}")

pd.DataFrame(
    hash_rows
).to_csv(
    HASH_PATH,
    index=False,
)

print(f"Saved: {HASH_PATH}")


# =============================================================================
# FINAL STATUS
# =============================================================================

section("STEP 19B5 COMPLETED SUCCESSFULLY")

print(
    "Causal annual-level + spatial-anomaly decomposition completed.\n"
    "Only 2018-2023 development rows were loaded.\n"
    "Annual-level forecasts used prior annual target means only.\n"
    "RF/LightGBM anomaly models reused frozen Step 17 hyperparameters.\n"
    "Validation predicted anomalies were centered without validation targets.\n"
    "Eligible predictions never used the current validation-year target.\n"
    "Oracle current-year-level predictions were separated and marked ineligible.\n"
    "2024 reserved validation remained untouched.\n"
    "2025 final lockbox remained untouched.\n"
    "No broad retuning or final model selection occurred.\n"
    "Review the complete output before Step 19C."
)
