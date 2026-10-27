
"""
STEP 13 - XGBOOST CANDIDATE MODEL ANALYSIS
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Evaluate XGBoost as a FIXED, UNTUNED candidate model using only the
2018-2023 development period.

STRICT PROTECTION
-----------------
2024 validation data is NOT loaded.
2025 final lockbox data is NOT loaded.

ROLLING-ORIGIN FOLDS
--------------------
2018       -> 2019
2018-2019  -> 2020
2018-2020  -> 2021
2018-2021  -> 2022
2018-2022  -> 2023

WHAT THIS STEP DOES
-------------------
- Fit the same predefined XGBoost configuration in every rolling fold.
- Calculate train and validation MAE, RMSE, R² and MAPE.
- Quantify train-vs-validation generalization gaps.
- Compare XGBoost against Step 11 baselines.
- Save fold-level predictions and residuals.
- Calculate fold-level validation permutation importance.
- Save full-development XGBoost feature importance.
- Fit and save ONE full 2018-2023 XGBoost candidate artifact for later use.

WHAT THIS STEP DOES NOT DO
--------------------------
- No hyperparameter search/tuning.
- No early-stopping parameter optimization.
- No 2024 evaluation.
- No 2025 evaluation.
- No final model selection.
"""

from pathlib import Path
import hashlib
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception as exc:
    print("ERROR: xgboost could not be imported.")
    print("Install/check xgboost inside the project virtual environment.")
    print(f"Original error: {exc}")
    sys.exit(1)


# =============================================================================
# 1. SETTINGS
# =============================================================================

TARGET = "LST_C"
SEED = 42

DEVELOPMENT_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
ROLLING_VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]

# Fixed candidate configuration.
# These values are predefined for candidate screening only.
# They are NOT the result of hyperparameter optimization.
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "tree_method": "hist",
    "eval_metric": "rmse",
}

PERMUTATION_REPEATS = 5


# =============================================================================
# 2. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "tree"
    / "tree_train_2018_2023_ready.csv"
)

BASELINE_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "11_baseline_rolling_origin_metrics.csv"
)

RF_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "12_random_forest_rolling_origin_metrics.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "candidates"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

for d in [MODEL_DIR, METRICS_DIR, PREDICTION_DIR, REPORT_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ROLLING_METRICS_PATH = (
    METRICS_DIR / "13_xgboost_rolling_origin_metrics.csv"
)

AGG_METRICS_PATH = (
    METRICS_DIR / "13_xgboost_aggregated_metrics.csv"
)

BASELINE_COMPARISON_PATH = (
    METRICS_DIR / "13_xgboost_vs_baselines.csv"
)

RF_REFERENCE_PATH = (
    METRICS_DIR / "13_xgboost_vs_random_forest_reference.csv"
)

PREDICTIONS_PATH = (
    PREDICTION_DIR / "13_xgboost_rolling_predictions.csv"
)

RESIDUAL_SUMMARY_PATH = (
    REPORT_DIR / "13_xgboost_residual_summary.csv"
)

PERM_FOLD_PATH = (
    REPORT_DIR / "13_xgboost_permutation_importance_by_fold.csv"
)

PERM_AGG_PATH = (
    REPORT_DIR / "13_xgboost_permutation_importance_aggregated.csv"
)

GAIN_IMPORTANCE_PATH = (
    REPORT_DIR / "13_xgboost_full_development_feature_importance.csv"
)

MODEL_PATH = (
    MODEL_DIR / "13_xgboost_candidate_2018_2023.joblib"
)

PARAMS_PATH = (
    REPORT_DIR / "13_xgboost_candidate_parameters.json"
)

AUDIT_PATH = (
    REPORT_DIR / "13_xgboost_audit.csv"
)

HASH_PATH = (
    REPORT_DIR / "13_xgboost_file_hashes.csv"
)

REPORT_PATH = (
    REPORT_DIR / "13_xgboost_model_report.txt"
)

MAE_FIG_PATH = (
    FIGURE_DIR / "13_xgboost_mae_by_year.png"
)

R2_FIG_PATH = (
    FIGURE_DIR / "13_xgboost_r2_by_year.png"
)

RESIDUAL_FIG_PATH = (
    FIGURE_DIR / "13_xgboost_residuals_by_year.png"
)

IMPORTANCE_FIG_PATH = (
    FIGURE_DIR / "13_xgboost_permutation_importance_top25.png"
)


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 124)
    print(title)
    print("=" * 124)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mape_percent(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.abs(y_true) > 1e-12

    if not mask.any():
        return np.nan

    return float(
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100.0
    )


def calculate_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": mape_percent(y_true, y_pred),
    }


def check_predictions(pred, expected_length, label) -> np.ndarray:
    arr = np.asarray(pred, dtype=float)

    if len(arr) != expected_length:
        raise ValueError(
            f"{label}: prediction length {len(arr)} != {expected_length}"
        )

    if np.isnan(arr).any():
        raise ValueError(f"{label}: NaN predictions detected.")

    if np.isinf(arr).any():
        raise ValueError(f"{label}: infinite predictions detected.")

    return arr


# =============================================================================
# 4. LOAD DEVELOPMENT DATA ONLY
# =============================================================================

section("STEP 13 - XGBOOST CANDIDATE MODEL ANALYSIS")

print(f"Project root       : {PROJECT_ROOT}")
print(f"Tree input         : {INPUT_PATH}")
print(f"Step 11 baselines  : {BASELINE_METRICS_PATH}")
print(f"Step 12 RF metrics : {RF_METRICS_PATH}")
print("2024 data          : NOT LOADED")
print("2025 data          : NOT LOADED")

if not INPUT_PATH.exists():
    print(f"\nERROR: required Step 10 tree file not found:\n{INPUT_PATH}")
    sys.exit(1)

if not BASELINE_METRICS_PATH.exists():
    print(
        "\nERROR: Step 11 baseline metrics not found. "
        "Step 11 must be completed before Step 13."
    )
    sys.exit(1)

if not RF_METRICS_PATH.exists():
    print(
        "\nERROR: Step 12 Random Forest metrics not found. "
        "Step 12 should be completed before Step 13."
    )
    sys.exit(1)

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False,
    dtype={
        "grid_id": "string",
        "area_name": "string",
    },
)

baseline_metrics = pd.read_csv(BASELINE_METRICS_PATH)
rf_metrics = pd.read_csv(RF_METRICS_PATH)

required = {
    "grid_id",
    "year",
    TARGET,
    "area_name",
}

missing_required = sorted(required - set(df.columns))

if missing_required:
    print(f"\nERROR: missing required columns: {missing_required}")
    sys.exit(1)

actual_years = sorted(
    df["year"].astype(int).unique().tolist()
)

if actual_years != DEVELOPMENT_YEARS:
    print(
        f"\nERROR: expected development years {DEVELOPMENT_YEARS}, "
        f"got {actual_years}"
    )
    sys.exit(1)

duplicate_keys = int(
    df.duplicated(
        ["grid_id", "year"],
        keep=False,
    ).sum()
)

if duplicate_keys != 0:
    print(f"\nERROR: duplicate grid-year rows = {duplicate_keys}")
    sys.exit(1)

print(f"Rows         : {len(df):,}")
print(f"Columns      : {len(df.columns):,}")
print(f"Unique grids : {df['grid_id'].nunique():,}")
print(f"Years        : {actual_years}")


# =============================================================================
# 5. FEATURE AUDIT
# =============================================================================

section("1. XGBOOST FEATURE AUDIT")

feature_columns = [
    c
    for c in df.columns
    if c not in [
        "grid_id",
        TARGET,
        "area_name",
    ]
]

if TARGET in feature_columns:
    print("\nERROR: target leaked into predictors.")
    sys.exit(1)

if "grid_id" in feature_columns or "area_name" in feature_columns:
    print("\nERROR: identifier/reporting field leaked into predictors.")
    sys.exit(1)

non_numeric = [
    c
    for c in feature_columns
    if not pd.api.types.is_numeric_dtype(df[c])
]

if non_numeric:
    print(f"\nERROR: non-numeric XGBoost predictors found: {non_numeric}")
    sys.exit(1)

missing_cells = int(
    df[feature_columns].isna().sum().sum()
)

infinite_cells = int(
    np.isinf(
        df[feature_columns].to_numpy(dtype=float)
    ).sum()
)

if missing_cells != 0 or infinite_cells != 0:
    print(
        f"\nERROR: predictor missing={missing_cells}, "
        f"infinite={infinite_cells}"
    )
    sys.exit(1)

print(f"Predictor count : {len(feature_columns)}")
print(f"Missing cells   : {missing_cells}")
print(f"Infinite cells  : {infinite_cells}")
print("Scaling         : NONE")
print("Imputation      : NONE")
print("PASS: finalized tree/nonlinear feature set is model-ready.")


# =============================================================================
# 6. FIXED CANDIDATE CONFIGURATION
# =============================================================================

section("2. FIXED UNTUNED XGBOOST CONFIGURATION")

print(json.dumps(XGB_PARAMS, indent=2))

PARAMS_PATH.write_text(
    json.dumps(
        {
            "model": "XGBRegressor",
            "purpose": "untuned candidate analysis",
            "parameters": XGB_PARAMS,
            "hyperparameter_search_performed": False,
            "early_stopping_optimization_performed": False,
            "development_years": DEVELOPMENT_YEARS,
            "2024_used": False,
            "2025_used": False,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(
    "\nIMPORTANT: these parameters are fixed for Step 13. "
    "No parameter search or early-stopping optimization is performed here."
)


# =============================================================================
# 7. ROLLING-ORIGIN EVALUATION
# =============================================================================

section("3. ROLLING-ORIGIN XGBOOST EVALUATION")

metric_rows = []
prediction_frames = []
permutation_rows = []

for fold, validation_year in enumerate(
    ROLLING_VALIDATION_YEARS,
    start=1,
):
    train_years = [
        y
        for y in DEVELOPMENT_YEARS
        if y < validation_year
    ]

    train = df[
        df["year"].isin(train_years)
    ].copy()

    val = df[
        df["year"] == validation_year
    ].copy()

    if train.empty or val.empty:
        raise ValueError(
            f"Fold {fold}: empty train or validation set."
        )

    X_train = train[feature_columns]
    y_train = train[TARGET]

    X_val = val[feature_columns]
    y_val = val[TARGET]

    model = XGBRegressor(**XGB_PARAMS)

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    train_pred = check_predictions(
        model.predict(X_train),
        len(train),
        f"Fold {fold} train",
    )

    val_pred = check_predictions(
        model.predict(X_val),
        len(val),
        f"Fold {fold} validation",
    )

    train_result = calculate_metrics(
        y_train.to_numpy(dtype=float),
        train_pred,
    )

    val_result = calculate_metrics(
        y_val.to_numpy(dtype=float),
        val_pred,
    )

    mae_gap = (
        val_result["mae"]
        - train_result["mae"]
    )

    rmse_gap = (
        val_result["rmse"]
        - train_result["rmse"]
    )

    r2_gap = (
        train_result["r2"]
        - val_result["r2"]
    )

    metric_rows.append(
        {
            "fold": fold,
            "train_start_year": min(train_years),
            "train_end_year": max(train_years),
            "validation_year": validation_year,
            "train_rows": len(train),
            "validation_rows": len(val),
            "train_mae": train_result["mae"],
            "validation_mae": val_result["mae"],
            "mae_generalization_gap": mae_gap,
            "train_rmse": train_result["rmse"],
            "validation_rmse": val_result["rmse"],
            "rmse_generalization_gap": rmse_gap,
            "train_r2": train_result["r2"],
            "validation_r2": val_result["r2"],
            "r2_generalization_gap": r2_gap,
            "train_mape_percent": train_result["mape_percent"],
            "validation_mape_percent": val_result["mape_percent"],
        }
    )

    residual = (
        y_val.to_numpy(dtype=float)
        - val_pred
    )

    fold_predictions = pd.DataFrame(
        {
            "grid_id": val["grid_id"].to_numpy(),
            "area_name": val["area_name"].to_numpy(),
            "validation_year": validation_year,
            "actual_LST_C": y_val.to_numpy(dtype=float),
            "predicted_LST_C": val_pred,
            "residual_actual_minus_predicted": residual,
            "absolute_error": np.abs(residual),
        }
    )

    prediction_frames.append(
        fold_predictions
    )

    print(
        f"\nFold {fold}: "
        f"{min(train_years)}-{max(train_years)} -> {validation_year}"
    )
    print(
        f"Train      : MAE={train_result['mae']:.4f} | "
        f"RMSE={train_result['rmse']:.4f} | "
        f"R2={train_result['r2']:.4f} | "
        f"MAPE={train_result['mape_percent']:.2f}%"
    )
    print(
        f"Validation : MAE={val_result['mae']:.4f} | "
        f"RMSE={val_result['rmse']:.4f} | "
        f"R2={val_result['r2']:.4f} | "
        f"MAPE={val_result['mape_percent']:.2f}%"
    )
    print(
        f"Gap        : MAE={mae_gap:.4f} | "
        f"RMSE={rmse_gap:.4f} | "
        f"R2={r2_gap:.4f}"
    )

    # Validation-only permutation importance.
    # Diagnostic only: no feature removal/tuning occurs here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        perm = permutation_importance(
            model,
            X_val,
            y_val,
            scoring="neg_mean_absolute_error",
            n_repeats=PERMUTATION_REPEATS,
            random_state=SEED + fold,
            n_jobs=-1,
        )

    for feature, mean_imp, std_imp in zip(
        feature_columns,
        perm.importances_mean,
        perm.importances_std,
    ):
        permutation_rows.append(
            {
                "fold": fold,
                "validation_year": validation_year,
                "feature": feature,
                "permutation_importance_mean": float(mean_imp),
                "permutation_importance_std": float(std_imp),
            }
        )


# =============================================================================
# 8. SAVE ROLLING RESULTS
# =============================================================================

section("4. SAVE ROLLING RESULTS")

rolling_metrics = pd.DataFrame(metric_rows)

rolling_predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

permutation_fold = pd.DataFrame(
    permutation_rows
)

rolling_metrics.to_csv(
    ROLLING_METRICS_PATH,
    index=False,
)

rolling_predictions.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

permutation_fold.to_csv(
    PERM_FOLD_PATH,
    index=False,
)

print(f"Saved: {ROLLING_METRICS_PATH}")
print(f"Saved: {PREDICTIONS_PATH}")
print(f"Saved: {PERM_FOLD_PATH}")


# =============================================================================
# 9. AGGREGATED XGBOOST METRICS
# =============================================================================

section("5. AGGREGATED XGBOOST METRICS")

aggregated_xgb = pd.DataFrame(
    [
        {
            "model": "XGBoost",
            "folds": len(rolling_metrics),
            "mean_train_mae": rolling_metrics["train_mae"].mean(),
            "mean_validation_mae": rolling_metrics["validation_mae"].mean(),
            "median_validation_mae": rolling_metrics["validation_mae"].median(),
            "std_validation_mae": rolling_metrics["validation_mae"].std(),
            "mean_validation_rmse": rolling_metrics["validation_rmse"].mean(),
            "median_validation_rmse": rolling_metrics["validation_rmse"].median(),
            "mean_validation_r2": rolling_metrics["validation_r2"].mean(),
            "median_validation_r2": rolling_metrics["validation_r2"].median(),
            "negative_r2_folds": int(
                (rolling_metrics["validation_r2"] < 0).sum()
            ),
            "mean_validation_mape_percent": (
                rolling_metrics["validation_mape_percent"].mean()
            ),
            "mean_mae_generalization_gap": (
                rolling_metrics["mae_generalization_gap"].mean()
            ),
            "mean_rmse_generalization_gap": (
                rolling_metrics["rmse_generalization_gap"].mean()
            ),
            "mean_r2_generalization_gap": (
                rolling_metrics["r2_generalization_gap"].mean()
            ),
        }
    ]
)

aggregated_xgb.to_csv(
    AGG_METRICS_PATH,
    index=False,
)

print(
    aggregated_xgb.to_string(
        index=False
    )
)


# =============================================================================
# 10. COMPARE WITH STEP 11 BASELINES
# =============================================================================

section("6. XGBOOST VS STEP 11 BASELINES")

required_baseline_cols = {
    "validation_year",
    "model",
    "mae",
    "rmse",
    "r2",
    "mape_percent",
}

if not required_baseline_cols.issubset(
    baseline_metrics.columns
):
    print(
        "\nERROR: Step 11 baseline metrics file has unexpected schema."
    )
    sys.exit(1)

comparison_rows = []

for _, xgb_row in rolling_metrics.iterrows():
    year = int(xgb_row["validation_year"])

    year_baselines = baseline_metrics[
        baseline_metrics["validation_year"] == year
    ].copy()

    if year_baselines.empty:
        print(
            f"\nERROR: no Step 11 baseline metrics found for {year}."
        )
        sys.exit(1)

    best_baseline = (
        year_baselines
        .sort_values(
            ["mae", "rmse", "model"],
            ascending=[True, True, True],
        )
        .iloc[0]
    )

    grid_baseline_rows = year_baselines[
        year_baselines["model"] == "Grid Historical Mean"
    ]

    if grid_baseline_rows.empty:
        print(
            f"\nERROR: Grid Historical Mean missing for {year}."
        )
        sys.exit(1)

    grid_baseline = grid_baseline_rows.iloc[0]

    comparison_rows.append(
        {
            "validation_year": year,
            "xgb_mae": xgb_row["validation_mae"],
            "xgb_rmse": xgb_row["validation_rmse"],
            "xgb_r2": xgb_row["validation_r2"],
            "best_baseline_model_that_year": best_baseline["model"],
            "best_baseline_mae": best_baseline["mae"],
            "xgb_mae_improvement_vs_best_baseline": (
                best_baseline["mae"]
                - xgb_row["validation_mae"]
            ),
            "xgb_beats_best_baseline_mae": (
                xgb_row["validation_mae"]
                < best_baseline["mae"]
            ),
            "grid_historical_mean_mae": grid_baseline["mae"],
            "xgb_mae_improvement_vs_grid_historical_mean": (
                grid_baseline["mae"]
                - xgb_row["validation_mae"]
            ),
            "xgb_beats_grid_historical_mean_mae": (
                xgb_row["validation_mae"]
                < grid_baseline["mae"]
            ),
        }
    )

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    BASELINE_COMPARISON_PATH,
    index=False,
)

print(
    comparison_df.to_string(
        index=False
    )
)

xgb_beats_best_count = int(
    comparison_df[
        "xgb_beats_best_baseline_mae"
    ].sum()
)

xgb_beats_grid_count = int(
    comparison_df[
        "xgb_beats_grid_historical_mean_mae"
    ].sum()
)

print(
    f"\nXGBoost beats the best Step 11 baseline by MAE in "
    f"{xgb_beats_best_count}/{len(comparison_df)} folds."
)

print(
    f"XGBoost beats Grid Historical Mean by MAE in "
    f"{xgb_beats_grid_count}/{len(comparison_df)} folds."
)


# =============================================================================
# 11. XGBOOST VS RANDOM FOREST REFERENCE
# =============================================================================

section("7. XGBOOST VS STEP 12 RANDOM FOREST REFERENCE")

required_rf_cols = {
    "validation_year",
    "validation_mae",
    "validation_rmse",
    "validation_r2",
}

if not required_rf_cols.issubset(rf_metrics.columns):
    print(
        "\nERROR: Step 12 RF metrics file has unexpected schema."
    )
    sys.exit(1)

rf_reference = (
    rolling_metrics[
        [
            "validation_year",
            "validation_mae",
            "validation_rmse",
            "validation_r2",
        ]
    ]
    .rename(
        columns={
            "validation_mae": "xgb_mae",
            "validation_rmse": "xgb_rmse",
            "validation_r2": "xgb_r2",
        }
    )
    .merge(
        rf_metrics[
            [
                "validation_year",
                "validation_mae",
                "validation_rmse",
                "validation_r2",
            ]
        ].rename(
            columns={
                "validation_mae": "rf_mae",
                "validation_rmse": "rf_rmse",
                "validation_r2": "rf_r2",
            }
        ),
        on="validation_year",
        how="inner",
        validate="one_to_one",
    )
)

rf_reference["xgb_mae_minus_rf_mae"] = (
    rf_reference["xgb_mae"]
    - rf_reference["rf_mae"]
)

rf_reference["xgb_lower_mae_than_rf"] = (
    rf_reference["xgb_mae"]
    < rf_reference["rf_mae"]
)

rf_reference.to_csv(
    RF_REFERENCE_PATH,
    index=False,
)

print(
    rf_reference.to_string(
        index=False
    )
)

xgb_lower_mae_rf_count = int(
    rf_reference[
        "xgb_lower_mae_than_rf"
    ].sum()
)

print(
    f"\nXGBoost has lower MAE than Step 12 RF in "
    f"{xgb_lower_mae_rf_count}/{len(rf_reference)} folds."
)

print(
    "NOTE: this is only a descriptive development-stage reference. "
    "Formal multi-model comparison remains Step 16."
)


# =============================================================================
# 12. RESIDUAL DIAGNOSTICS
# =============================================================================

section("8. RESIDUAL DIAGNOSTICS")

residual_summary = (
    rolling_predictions
    .groupby("validation_year")
    .agg(
        residual_mean=(
            "residual_actual_minus_predicted",
            "mean",
        ),
        residual_median=(
            "residual_actual_minus_predicted",
            "median",
        ),
        residual_std=(
            "residual_actual_minus_predicted",
            "std",
        ),
        mean_absolute_error=(
            "absolute_error",
            "mean",
        ),
        p90_absolute_error=(
            "absolute_error",
            lambda s: float(np.percentile(s, 90)),
        ),
        p95_absolute_error=(
            "absolute_error",
            lambda s: float(np.percentile(s, 95)),
        ),
        max_absolute_error=(
            "absolute_error",
            "max",
        ),
    )
    .reset_index()
)

residual_summary.to_csv(
    RESIDUAL_SUMMARY_PATH,
    index=False,
)

print(
    residual_summary.to_string(
        index=False
    )
)


# =============================================================================
# 13. AGGREGATE PERMUTATION IMPORTANCE
# =============================================================================

section("9. PERMUTATION IMPORTANCE STABILITY")

perm_agg = (
    permutation_fold
    .groupby("feature")
    .agg(
        mean_importance=(
            "permutation_importance_mean",
            "mean",
        ),
        median_importance=(
            "permutation_importance_mean",
            "median",
        ),
        std_importance=(
            "permutation_importance_mean",
            "std",
        ),
        positive_fold_fraction=(
            "permutation_importance_mean",
            lambda s: float((s > 0).mean()),
        ),
        negative_fold_fraction=(
            "permutation_importance_mean",
            lambda s: float((s < 0).mean()),
        ),
        min_importance=(
            "permutation_importance_mean",
            "min",
        ),
        max_importance=(
            "permutation_importance_mean",
            "max",
        ),
    )
    .reset_index()
)

perm_agg["robust_importance_score"] = (
    perm_agg["median_importance"].clip(lower=0)
    * perm_agg["positive_fold_fraction"]
)

perm_agg = perm_agg.sort_values(
    [
        "robust_importance_score",
        "mean_importance",
        "feature",
    ],
    ascending=[False, False, True],
).reset_index(drop=True)

perm_agg["rank"] = np.arange(
    1,
    len(perm_agg) + 1,
)

perm_agg.to_csv(
    PERM_AGG_PATH,
    index=False,
)

print("Top 25 robust permutation features:")
print(
    perm_agg.head(25).to_string(
        index=False
    )
)


# =============================================================================
# 14. FIT FULL 2018-2023 XGBOOST CANDIDATE ARTIFACT
# =============================================================================

section("10. FIT FULL 2018-2023 XGBOOST CANDIDATE")

full_model = XGBRegressor(
    **XGB_PARAMS
)

full_model.fit(
    df[feature_columns],
    df[TARGET],
    verbose=False,
)

joblib.dump(
    {
        "model": full_model,
        "feature_names": feature_columns,
        "fit_years": DEVELOPMENT_YEARS,
        "target": TARGET,
        "parameters": XGB_PARAMS,
        "hyperparameter_tuned": False,
        "early_stopping_optimized": False,
        "2024_used": False,
        "2025_used": False,
        "purpose": (
            "Untuned XGBoost candidate artifact for later "
            "reserved validation/comparison."
        ),
    },
    MODEL_PATH,
)

importance_df = pd.DataFrame(
    {
        "feature": feature_columns,
        "xgboost_feature_importance": full_model.feature_importances_,
    }
).sort_values(
    [
        "xgboost_feature_importance",
        "feature",
    ],
    ascending=[False, True],
).reset_index(drop=True)

importance_df["rank"] = np.arange(
    1,
    len(importance_df) + 1,
)

importance_df.to_csv(
    GAIN_IMPORTANCE_PATH,
    index=False,
)

print(f"Saved candidate model: {MODEL_PATH}")
print(f"Saved model importance: {GAIN_IMPORTANCE_PATH}")


# =============================================================================
# 15. FIGURES
# =============================================================================

section("11. FIGURES")

# Train vs validation MAE
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    rolling_metrics["validation_year"].astype(str),
    rolling_metrics["train_mae"],
    marker="o",
    label="Train MAE",
)

ax.plot(
    rolling_metrics["validation_year"].astype(str),
    rolling_metrics["validation_mae"],
    marker="o",
    label="Validation MAE",
)

ax.set_title("XGBoost Train vs Validation MAE")
ax.set_xlabel("Validation Year")
ax.set_ylabel("MAE (°C)")
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()
plt.savefig(
    MAE_FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Train vs validation R2
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    rolling_metrics["validation_year"].astype(str),
    rolling_metrics["train_r2"],
    marker="o",
    label="Train R²",
)

ax.plot(
    rolling_metrics["validation_year"].astype(str),
    rolling_metrics["validation_r2"],
    marker="o",
    label="Validation R²",
)

ax.axhline(0, linewidth=1)

ax.set_title("XGBoost Train vs Validation R²")
ax.set_xlabel("Validation Year")
ax.set_ylabel("R²")
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()
plt.savefig(
    R2_FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Residual boxplot by validation year
fig, ax = plt.subplots(figsize=(10, 6))

year_groups = [
    rolling_predictions.loc[
        rolling_predictions["validation_year"] == year,
        "residual_actual_minus_predicted",
    ].to_numpy()
    for year in ROLLING_VALIDATION_YEARS
]

ax.boxplot(
    year_groups,
    tick_labels=[
        str(year)
        for year in ROLLING_VALIDATION_YEARS
    ],
)

ax.axhline(0, linewidth=1)

ax.set_title("XGBoost Validation Residuals by Year")
ax.set_xlabel("Validation Year")
ax.set_ylabel("Residual (Actual - Predicted) °C")
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    RESIDUAL_FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Robust rolling permutation importance
top_perm = (
    perm_agg
    .head(25)
    .sort_values(
        "robust_importance_score",
        ascending=True,
    )
)

fig, ax = plt.subplots(
    figsize=(10, 9)
)

ax.barh(
    top_perm["feature"],
    top_perm["robust_importance_score"],
)

ax.set_title(
    "XGBoost Robust Rolling Permutation Importance - Top 25"
)
ax.set_xlabel("Robust importance score")
ax.set_ylabel("Feature")
ax.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig(
    IMPORTANCE_FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print(f"Saved: {MAE_FIG_PATH}")
print(f"Saved: {R2_FIG_PATH}")
print(f"Saved: {RESIDUAL_FIG_PATH}")
print(f"Saved: {IMPORTANCE_FIG_PATH}")


# =============================================================================
# 16. STRICT AUDIT
# =============================================================================

section("12. STRICT STEP 13 AUDIT")

validation_years_used = sorted(
    rolling_metrics[
        "validation_year"
    ].unique().tolist()
)

expected_rows = len(
    ROLLING_VALIDATION_YEARS
)

all_predictions_finite = bool(
    np.isfinite(
        rolling_predictions[
            [
                "actual_LST_C",
                "predicted_LST_C",
                "residual_actual_minus_predicted",
                "absolute_error",
            ]
        ].to_numpy(dtype=float)
    ).all()
)

rf_reference_complete = (
    len(rf_reference) == expected_rows
)

audit_df = pd.DataFrame(
    [
        {
            "check": "development_input_years",
            "status": (
                "PASS"
                if actual_years == DEVELOPMENT_YEARS
                else "FAIL"
            ),
            "details": str(actual_years),
        },
        {
            "check": "2024_loaded_or_used",
            "status": "PASS",
            "details": "No. 2024 validation data was not loaded.",
        },
        {
            "check": "2025_loaded_or_used",
            "status": "PASS",
            "details": "No. 2025 lockbox data was not loaded.",
        },
        {
            "check": "rolling_validation_years",
            "status": (
                "PASS"
                if validation_years_used == ROLLING_VALIDATION_YEARS
                else "FAIL"
            ),
            "details": str(validation_years_used),
        },
        {
            "check": "expected_fold_rows",
            "status": (
                "PASS"
                if len(rolling_metrics) == expected_rows
                else "FAIL"
            ),
            "details": (
                f"actual={len(rolling_metrics)}, "
                f"expected={expected_rows}"
            ),
        },
        {
            "check": "target_in_predictors",
            "status": "PASS",
            "details": "No. LST_C excluded from XGBoost predictors.",
        },
        {
            "check": "identifiers_in_predictors",
            "status": "PASS",
            "details": "No. grid_id and area_name excluded.",
        },
        {
            "check": "scaling_or_imputation",
            "status": "PASS",
            "details": "None applied to XGBoost branch.",
        },
        {
            "check": "hyperparameter_search",
            "status": "PASS",
            "details": "No. Fixed candidate configuration only.",
        },
        {
            "check": "early_stopping_optimization",
            "status": "PASS",
            "details": "No. Not used for candidate tuning.",
        },
        {
            "check": "finite_predictions",
            "status": (
                "PASS"
                if all_predictions_finite
                else "FAIL"
            ),
            "details": str(all_predictions_finite),
        },
        {
            "check": "baseline_comparison_available",
            "status": (
                "PASS"
                if len(comparison_df) == expected_rows
                else "FAIL"
            ),
            "details": f"rows={len(comparison_df)}",
        },
        {
            "check": "rf_reference_complete",
            "status": (
                "PASS"
                if rf_reference_complete
                else "FAIL"
            ),
            "details": f"rows={len(rf_reference)}",
        },
        {
            "check": "final_model_selected",
            "status": "PASS",
            "details": "No. XGBoost remains a candidate model.",
        },
    ]
)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
)

print(
    audit_df.to_string(
        index=False
    )
)

if (audit_df["status"] != "PASS").any():
    print(
        "\nERROR: one or more Step 13 audit checks failed."
    )
    sys.exit(1)


# =============================================================================
# 17. REPRODUCIBILITY HASHES
# =============================================================================

section("13. REPRODUCIBILITY HASHES")

hash_files = [
    INPUT_PATH,
    BASELINE_METRICS_PATH,
    RF_METRICS_PATH,
    ROLLING_METRICS_PATH,
    AGG_METRICS_PATH,
    BASELINE_COMPARISON_PATH,
    RF_REFERENCE_PATH,
    PREDICTIONS_PATH,
    RESIDUAL_SUMMARY_PATH,
    PERM_FOLD_PATH,
    PERM_AGG_PATH,
    GAIN_IMPORTANCE_PATH,
    MODEL_PATH,
    PARAMS_PATH,
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


# =============================================================================
# 18. REPORT
# =============================================================================

section("14. SAVE STEP 13 REPORT")

agg_row = aggregated_xgb.iloc[0]

report_lines = [
    "STEP 13 - XGBOOST CANDIDATE MODEL REPORT",
    "=" * 108,
    "",
    "DATA PROTECTION",
    "-" * 108,
    "Used: 2018-2023 development data only.",
    "2024: NOT loaded.",
    "2025: NOT loaded.",
    "",
    "MODEL",
    "-" * 108,
    "XGBRegressor",
    "Status: UNTUNED fixed candidate model",
    f"Parameters: {XGB_PARAMS}",
    "",
    "ROLLING-ORIGIN DEVELOPMENT VALIDATION",
    "-" * 108,
    "2018 -> 2019",
    "2018-2019 -> 2020",
    "2018-2020 -> 2021",
    "2018-2021 -> 2022",
    "2018-2022 -> 2023",
    "",
    "AGGREGATED PERFORMANCE",
    "-" * 108,
    f"Mean validation MAE: {agg_row['mean_validation_mae']:.6f}",
    f"Median validation MAE: {agg_row['median_validation_mae']:.6f}",
    f"Mean validation RMSE: {agg_row['mean_validation_rmse']:.6f}",
    f"Mean validation R²: {agg_row['mean_validation_r2']:.6f}",
    f"Median validation R²: {agg_row['median_validation_r2']:.6f}",
    f"Negative-R² folds: {int(agg_row['negative_r2_folds'])}",
    f"Mean MAE generalization gap: {agg_row['mean_mae_generalization_gap']:.6f}",
    "",
    "BASELINE COMPARISON",
    "-" * 108,
    (
        f"XGBoost beats the best Step 11 baseline by validation MAE in "
        f"{xgb_beats_best_count}/{len(comparison_df)} folds."
    ),
    (
        f"XGBoost beats Grid Historical Mean by validation MAE in "
        f"{xgb_beats_grid_count}/{len(comparison_df)} folds."
    ),
    "",
    "RANDOM FOREST REFERENCE",
    "-" * 108,
    (
        f"XGBoost has lower validation MAE than Step 12 Random Forest in "
        f"{xgb_lower_mae_rf_count}/{len(rf_reference)} folds."
    ),
    "This is descriptive only. Formal candidate comparison is Step 16.",
    "",
    "IMPORTANT",
    "-" * 108,
    "This step does NOT tune XGBoost.",
    "This step does NOT use early stopping to optimize the candidate.",
    "This step does NOT select XGBoost as the final model.",
    "Negative R² or large generalization gaps are retained as research findings.",
    "",
    "NEXT",
    "-" * 108,
    "Proceed to Step 14 LightGBM candidate analysis only after reviewing",
    "the complete Step 13 output and resolving any Step 13 issues.",
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {REPORT_PATH}")
print(f"Saved: {AUDIT_PATH}")
print(f"Saved: {HASH_PATH}")


# =============================================================================
# 19. FINAL STATUS
# =============================================================================

section("STEP 13 COMPLETED SUCCESSFULLY")

print(
    "Untuned XGBoost candidate analysis completed using 2018-2023 only.\n"
    "2024 remains untouched for reserved validation.\n"
    "2025 remains untouched as the final lockbox.\n"
    "Five rolling-origin folds were evaluated.\n"
    "Train-vs-validation gaps, residuals, baseline comparison, Random Forest "
    "reference, and permutation importance were saved.\n"
    "No hyperparameter tuning, early-stopping optimization, or final model "
    "selection was performed.\n"
    "Review the complete Step 13 output before Step 14."
)
