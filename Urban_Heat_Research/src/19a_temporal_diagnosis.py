
"""
STEP 19A - TEMPORAL DIAGNOSIS
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Diagnose WHY tuned development models show weak temporal generalization,
especially negative R² in 2021 and 2023, before trying any improvement.

THIS FILE DOES NOT:
- tune hyperparameters
- change features
- apply recent-year weighting
- correct temporal bias
- build ensembles
- select a final model
- load or evaluate 2024
- load or evaluate 2025

DATA USED
---------
1) 2018-2023 development model-ready data
2) Step 17 strict rolling-origin tuned predictions for 2019-2023
3) Frozen grid geometry for residual spatial-autocorrelation diagnosis

DIAGNOSTICS
-----------
A. Year-level prediction bias
B. Raw R² versus centered/spatial-pattern R²
C. Feature distribution shift (SMD, KS, PSI, Wasserstein)
D. Feature-LST relationship stability across years
E. Residual distributions
F. Residual-feature associations
G. Spatial autocorrelation of residuals (Moran's I)
H. Focused evidence tables for 2021 and 2023

IMPORTANT
---------
Any "instability flags" are diagnostic heuristics only.
They are NOT used here to select a feature subset.
Feature/strategy improvements belong to Step 19B.
"""

from pathlib import Path
import hashlib
import math
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

from scipy.stats import (
    ks_2samp,
    pearsonr,
    spearmanr,
    wasserstein_distance,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from libpysal.weights import Queen
from esda.moran import Moran


# =============================================================================
# 1. SETTINGS
# =============================================================================

TARGET = "LST_C"
GRID_COL = "grid_id"
AREA_COL = "area_name"
YEAR_COL = "year"

DEVELOPMENT_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
ROLLING_VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]
PROBLEM_YEARS = [2021, 2023]

EXPECTED_UNIQUE_GRIDS = 2194
EXPECTED_PREDICTOR_COUNT = 93

MODEL_ORDER = [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "NGBoost",
]

EXPECTED_PREDICTION_ROWS = (
    len(MODEL_ORDER)
    * len(ROLLING_VALIDATION_YEARS)
    * EXPECTED_UNIQUE_GRIDS
)

# Diagnostic thresholds only.
# They DO NOT select features in Step 19A.
PSI_MODERATE_SHIFT = 0.25
ABS_SMD_MODERATE_SHIFT = 0.50
RHO_RANGE_INSTABILITY = 0.40
PROBLEM_YEAR_RHO_JUMP = 0.30
MEANINGFUL_RHO_SIGN_THRESHOLD = 0.05

# Moran residual diagnostic
MORAN_PERMUTATIONS = 999

# Numerical tolerances
EPS = 1e-12
STEP17_METRIC_TOLERANCE = 1e-9


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

STEP17_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "17_hyperparameter_tuning_audit.csv"
)

STEP18_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "18_spatial_validation_audit.csv"
)

GEOMETRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "spatial"
    / "spatial_fold_grid_utm44n.geojson"
)

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

for folder in [METRICS_DIR, REPORTS_DIR, FIGURES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

YEAR_LEVEL_DIAG_PATH = (
    METRICS_DIR / "19a_year_level_prediction_diagnostics.csv"
)

TARGET_YEAR_STATS_PATH = (
    METRICS_DIR / "19a_target_distribution_by_year.csv"
)

RESIDUAL_SUMMARY_PATH = (
    METRICS_DIR / "19a_residual_summary_by_model_year.csv"
)

RESIDUAL_FEATURE_ASSOC_PATH = (
    METRICS_DIR / "19a_residual_feature_associations.csv"
)

RESIDUAL_MORAN_PATH = (
    METRICS_DIR / "19a_residual_moran_by_model_year.csv"
)

FEATURE_SHIFT_PATH = (
    METRICS_DIR / "19a_feature_distribution_shift_by_fold.csv"
)

FEATURE_SHIFT_SUMMARY_PATH = (
    METRICS_DIR / "19a_feature_distribution_shift_summary.csv"
)

YEAR_SHIFT_SUMMARY_PATH = (
    METRICS_DIR / "19a_year_distribution_shift_summary.csv"
)

FEATURE_YEAR_ASSOC_PATH = (
    METRICS_DIR / "19a_feature_lst_spearman_by_year.csv"
)

FEATURE_STABILITY_PATH = (
    METRICS_DIR / "19a_feature_temporal_stability.csv"
)

PROBLEM_YEAR_EVIDENCE_PATH = (
    METRICS_DIR / "19a_problem_year_evidence.csv"
)

STEP17_REPRO_CHECK_PATH = (
    REPORTS_DIR / "19a_step17_prediction_reproduction_check.csv"
)

AUDIT_PATH = (
    REPORTS_DIR / "19a_temporal_diagnosis_audit.csv"
)

REPORT_PATH = (
    REPORTS_DIR / "19a_temporal_diagnosis_report.txt"
)

HASH_PATH = (
    REPORTS_DIR / "19a_temporal_diagnosis_hashes.csv"
)

FIG_YEAR_MEANS = (
    FIGURES_DIR / "19a_actual_vs_predicted_year_mean.png"
)

FIG_RAW_R2 = (
    FIGURES_DIR / "19a_raw_r2_by_year.png"
)

FIG_CENTERED_R2 = (
    FIGURES_DIR / "19a_centered_spatial_r2_by_year.png"
)

FIG_BIAS = (
    FIGURES_DIR / "19a_prediction_bias_by_year.png"
)

FIG_SHIFT = (
    FIGURES_DIR / "19a_top_feature_shift_mean_psi.png"
)

FIG_MORAN = (
    FIGURES_DIR / "19a_residual_moran_by_year.png"
)


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 126)
    print(title)
    print("=" * 126)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def check_audit_all_pass(path: Path, label: str) -> None:
    if not path.exists():
        print(
            f"\nERROR: required {label} audit is missing: {path}"
        )
        sys.exit(1)

    audit = pd.read_csv(path)

    if "status" not in audit.columns:
        print(
            f"\nERROR: {label} audit has no 'status' column."
        )
        sys.exit(1)

    bad = audit[
        audit["status"].astype(str).str.upper()
        != "PASS"
    ]

    if not bad.empty:
        print(
            f"\nERROR: {label} audit is not fully PASS."
        )
        print(bad.to_string(index=False))
        sys.exit(1)


def mape_percent(
    y_true,
    y_pred,
) -> float:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    valid = np.abs(y_true) > EPS

    if not np.any(valid):
        return np.nan

    return float(
        np.mean(
            np.abs(
                (
                    y_true[valid]
                    - y_pred[valid]
                )
                / y_true[valid]
            )
        )
        * 100.0
    )


def safe_pearson(
    x,
    y,
) -> float:
    x = np.asarray(
        x,
        dtype=float,
    )
    y = np.asarray(
        y,
        dtype=float,
    )

    if (
        len(x) < 3
        or np.std(x) < EPS
        or np.std(y) < EPS
    ):
        return np.nan

    return float(
        pearsonr(
            x,
            y,
        ).statistic
    )


def safe_spearman(
    x,
    y,
) -> float:
    x = np.asarray(
        x,
        dtype=float,
    )
    y = np.asarray(
        y,
        dtype=float,
    )

    if (
        len(x) < 3
        or np.std(x) < EPS
        or np.std(y) < EPS
    ):
        return np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = spearmanr(
            x,
            y,
            nan_policy="omit",
        )

    return float(result.statistic)


def population_stdev(values) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.std(
            values,
            ddof=0,
        )
    )


def regression_diagnostics(
    y_true,
    y_pred,
) -> dict:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    actual_mean = float(
        np.mean(y_true)
    )
    predicted_mean = float(
        np.mean(y_pred)
    )

    bias_pred_minus_actual = (
        predicted_mean
        - actual_mean
    )

    residual = (
        y_true
        - y_pred
    )

    mae = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    mse = float(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    rmse = float(
        math.sqrt(mse)
    )

    r2 = float(
        r2_score(
            y_true,
            y_pred,
        )
    )

    # Why R² can become negative:
    # R² = 1 - SSE/TSS.
    sse = float(
        np.sum(
            (
                y_true
                - y_pred
            )
            ** 2
        )
    )

    tss = float(
        np.sum(
            (
                y_true
                - actual_mean
            )
            ** 2
        )
    )

    sse_over_tss = (
        sse / tss
        if tss > EPS
        else np.nan
    )

    actual_centered = (
        y_true
        - actual_mean
    )

    predicted_centered = (
        y_pred
        - predicted_mean
    )

    centered_error = (
        actual_centered
        - predicted_centered
    )

    centered_mse = float(
        np.mean(
            centered_error ** 2
        )
    )

    centered_rmse = float(
        math.sqrt(
            centered_mse
        )
    )

    centered_mae = float(
        np.mean(
            np.abs(
                centered_error
            )
        )
    )

    centered_r2 = float(
        1.0
        - (
            np.sum(
                centered_error ** 2
            )
            / tss
        )
    ) if tss > EPS else np.nan

    bias_mse_component = (
        bias_pred_minus_actual ** 2
    )

    bias_mse_share = (
        bias_mse_component / mse
        if mse > EPS
        else 0.0
    )

    centered_mse_share = (
        centered_mse / mse
        if mse > EPS
        else 0.0
    )

    actual_std = population_stdev(
        y_true
    )

    predicted_std = population_stdev(
        y_pred
    )

    std_ratio = (
        predicted_std / actual_std
        if actual_std > EPS
        else np.nan
    )

    return {
        "rows": len(y_true),
        "actual_mean_c": actual_mean,
        "predicted_mean_c": predicted_mean,
        "prediction_bias_pred_minus_actual_c": bias_pred_minus_actual,
        "actual_std_c": actual_std,
        "predicted_std_c": predicted_std,
        "predicted_to_actual_std_ratio": std_ratio,
        "mae_c": mae,
        "rmse_c": rmse,
        "mape_percent": mape_percent(
            y_true,
            y_pred,
        ),
        "r2": r2,
        "pearson_r": safe_pearson(
            y_true,
            y_pred,
        ),
        "spearman_rho": safe_spearman(
            y_true,
            y_pred,
        ),
        "sse": sse,
        "tss": tss,
        "sse_over_tss": sse_over_tss,
        "centered_spatial_mae_c": centered_mae,
        "centered_spatial_rmse_c": centered_rmse,
        "centered_spatial_r2": centered_r2,
        "bias_squared_share_of_mse": bias_mse_share,
        "centered_error_share_of_mse": centered_mse_share,
        "negative_r2": bool(
            r2 < 0
        ),
        "r2_failure_condition_sse_gt_tss": bool(
            sse > tss
        ),
    }


def standardized_mean_difference(
    train_values,
    val_values,
):
    train_values = np.asarray(
        train_values,
        dtype=float,
    )

    val_values = np.asarray(
        val_values,
        dtype=float,
    )

    train_std = float(
        np.std(
            train_values,
            ddof=0,
        )
    )

    train_mean = float(
        np.mean(
            train_values,
        )
    )

    val_mean = float(
        np.mean(
            val_values,
        )
    )

    if train_std < EPS:
        return (
            np.nan,
            True,
        )

    return (
        float(
            (
                val_mean
                - train_mean
            )
            / train_std
        ),
        False,
    )


def population_stability_index(
    train_values,
    val_values,
    quantile_bins=10,
) -> float:
    """
    PSI using quantile bins estimated from TRAIN data only.

    This is a diagnostic distribution-shift statistic.
    It is not used for feature selection in Step 19A.
    """

    train_values = np.asarray(
        train_values,
        dtype=float,
    )

    val_values = np.asarray(
        val_values,
        dtype=float,
    )

    quantiles = np.linspace(
        0.0,
        1.0,
        quantile_bins + 1,
    )

    edges = np.unique(
        np.quantile(
            train_values,
            quantiles,
        )
    )

    if len(edges) < 2:
        # Constant training feature.
        # If validation is also exactly the same, no shift.
        if (
            np.std(val_values) < EPS
            and abs(
                np.mean(val_values)
                - np.mean(train_values)
            ) < EPS
        ):
            return 0.0

        return np.nan

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    train_counts, _ = np.histogram(
        train_values,
        bins=edges,
    )

    val_counts, _ = np.histogram(
        val_values,
        bins=edges,
    )

    train_prop = (
        train_counts
        / max(
            len(train_values),
            1,
        )
    )

    val_prop = (
        val_counts
        / max(
            len(val_values),
            1,
        )
    )

    tiny = 1e-6

    train_prop = np.clip(
        train_prop,
        tiny,
        None,
    )

    val_prop = np.clip(
        val_prop,
        tiny,
        None,
    )

    return float(
        np.sum(
            (
                val_prop
                - train_prop
            )
            * np.log(
                val_prop
                / train_prop
            )
        )
    )


def meaningful_sign(
    rho,
) -> int:
    if pd.isna(rho):
        return 0

    if rho >= MEANINGFUL_RHO_SIGN_THRESHOLD:
        return 1

    if rho <= -MEANINGFUL_RHO_SIGN_THRESHOLD:
        return -1

    return 0


def chronological_sign_flips(
    rhos,
) -> int:
    signs = [
        meaningful_sign(rho)
        for rho in rhos
    ]

    signs = [
        sign
        for sign in signs
        if sign != 0
    ]

    if len(signs) < 2:
        return 0

    return int(
        sum(
            signs[i] != signs[i - 1]
            for i in range(
                1,
                len(signs),
            )
        )
    )


# =============================================================================
# 4. PREREQUISITE CHECKS
# =============================================================================

section(
    "STEP 19A - TEMPORAL DIAGNOSIS"
)

required_files = [
    INPUT_PATH,
    STEP17_PREDICTIONS_PATH,
    STEP17_FOLD_METRICS_PATH,
    STEP17_AUDIT_PATH,
    STEP18_AUDIT_PATH,
    GEOMETRY_PATH,
]

for path in required_files:
    if not path.exists():
        print(
            f"\nERROR: required file is missing: {path}"
        )
        sys.exit(1)

check_audit_all_pass(
    STEP17_AUDIT_PATH,
    "Step 17",
)

check_audit_all_pass(
    STEP18_AUDIT_PATH,
    "Step 18",
)

print(
    "PASS: Step 17 audit fully PASS."
)
print(
    "PASS: Step 18 audit fully PASS."
)
print(
    "19A is DIAGNOSIS ONLY - no model tuning or improvement."
)


# =============================================================================
# 5. LOAD 2018-2023 DEVELOPMENT DATA ONLY
# =============================================================================

section(
    "1. LOAD DEVELOPMENT DATA"
)

df = pd.read_csv(
    INPUT_PATH,
    dtype={
        GRID_COL: str,
    },
)

required_cols = {
    TARGET,
    GRID_COL,
    AREA_COL,
    YEAR_COL,
}

missing_required = sorted(
    required_cols
    - set(df.columns)
)

if missing_required:
    print(
        f"\nERROR: missing required columns: {missing_required}"
    )
    sys.exit(1)

years = sorted(
    df[YEAR_COL]
    .astype(int)
    .unique()
    .tolist()
)

if years != DEVELOPMENT_YEARS:
    print(
        f"\nERROR: development years are {years}; "
        f"expected {DEVELOPMENT_YEARS}."
    )
    sys.exit(1)

if (
    df[GRID_COL].nunique()
    != EXPECTED_UNIQUE_GRIDS
):
    print(
        f"\nERROR: unique grid count is "
        f"{df[GRID_COL].nunique()}; "
        f"expected {EXPECTED_UNIQUE_GRIDS}."
    )
    sys.exit(1)

predictors = [
    col
    for col in df.columns
    if col not in {
        TARGET,
        GRID_COL,
        AREA_COL,
    }
]

if (
    len(predictors)
    != EXPECTED_PREDICTOR_COUNT
):
    print(
        f"\nERROR: predictor count is {len(predictors)}; "
        f"expected {EXPECTED_PREDICTOR_COUNT}."
    )
    sys.exit(1)

for feature in predictors:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce",
    )

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce",
)

predictor_missing = int(
    df[predictors]
    .isna()
    .sum()
    .sum()
)

predictor_finite = bool(
    np.isfinite(
        df[predictors]
        .to_numpy(
            dtype=float,
        )
    ).all()
)

target_finite = bool(
    np.isfinite(
        df[TARGET]
        .to_numpy(
            dtype=float,
        )
    ).all()
)

if (
    predictor_missing != 0
    or not predictor_finite
    or not target_finite
):
    print(
        "\nERROR: development data contains missing/non-finite "
        "modeling values."
    )
    sys.exit(1)

print(
    f"Rows               : {len(df):,}"
)
print(
    f"Columns            : {len(df.columns)}"
)
print(
    f"Unique grids       : {df[GRID_COL].nunique():,}"
)
print(
    f"Years              : {years}"
)
print(
    f"Predictors         : {len(predictors)}"
)
print(
    f"Missing predictors : {predictor_missing}"
)
print(
    f"Finite predictors  : {predictor_finite}"
)
print(
    "2024 data          : NOT LOADED"
)
print(
    "2025 data          : NOT LOADED"
)


# =============================================================================
# 6. LOAD + VERIFY STEP 17 SAVED ROLLING PREDICTIONS
# =============================================================================

section(
    "2. LOAD + VERIFY STEP 17 TUNED ROLLING PREDICTIONS"
)

pred = pd.read_csv(
    STEP17_PREDICTIONS_PATH,
    dtype={
        GRID_COL: str,
    },
)

required_pred_cols = {
    "model",
    GRID_COL,
    "validation_year",
    "actual_lst_c",
    "predicted_lst_c",
    "residual_actual_minus_predicted",
}

missing_pred_cols = sorted(
    required_pred_cols
    - set(pred.columns)
)

if missing_pred_cols:
    print(
        f"\nERROR: Step 17 prediction file missing columns: "
        f"{missing_pred_cols}"
    )
    sys.exit(1)

pred["validation_year"] = pd.to_numeric(
    pred["validation_year"],
    errors="coerce",
).astype(int)

for col in [
    "actual_lst_c",
    "predicted_lst_c",
    "residual_actual_minus_predicted",
]:
    pred[col] = pd.to_numeric(
        pred[col],
        errors="coerce",
    )

pred_models = sorted(
    pred["model"]
    .dropna()
    .unique()
    .tolist()
)

if sorted(MODEL_ORDER) != pred_models:
    print(
        f"\nERROR: prediction models are {pred_models}; "
        f"expected {sorted(MODEL_ORDER)}."
    )
    sys.exit(1)

pred_years = sorted(
    pred["validation_year"]
    .unique()
    .tolist()
)

if pred_years != ROLLING_VALIDATION_YEARS:
    print(
        f"\nERROR: prediction years are {pred_years}; "
        f"expected {ROLLING_VALIDATION_YEARS}."
    )
    sys.exit(1)

if len(pred) != EXPECTED_PREDICTION_ROWS:
    print(
        f"\nERROR: Step 17 prediction rows={len(pred):,}; "
        f"expected {EXPECTED_PREDICTION_ROWS:,}."
    )
    sys.exit(1)

if not np.isfinite(
    pred[
        [
            "actual_lst_c",
            "predicted_lst_c",
            "residual_actual_minus_predicted",
        ]
    ].to_numpy(
        dtype=float,
    )
).all():
    print(
        "\nERROR: Step 17 prediction file contains "
        "non-finite prediction values."
    )
    sys.exit(1)

prediction_key_counts = (
    pred.groupby(
        [
            "model",
            GRID_COL,
            "validation_year",
        ]
    )
    .size()
)

if not (
    len(prediction_key_counts)
    == EXPECTED_PREDICTION_ROWS
    and prediction_key_counts.eq(1).all()
):
    print(
        "\nERROR: duplicate/missing model-grid-year prediction keys."
    )
    sys.exit(1)

# Verify actual target in predictions equals development source target.
target_reference = (
    df[
        [
            GRID_COL,
            YEAR_COL,
            TARGET,
        ]
    ]
    .rename(
        columns={
            YEAR_COL: "validation_year",
            TARGET: "source_lst_c",
        }
    )
)

pred_check = pred.merge(
    target_reference,
    on=[
        GRID_COL,
        "validation_year",
    ],
    how="left",
    validate="many_to_one",
)

if pred_check["source_lst_c"].isna().any():
    print(
        "\nERROR: some Step 17 predictions cannot be matched "
        "to the development source data."
    )
    sys.exit(1)

target_match_max_abs_diff = float(
    np.max(
        np.abs(
            pred_check["actual_lst_c"]
            - pred_check["source_lst_c"]
        )
    )
)

if (
    target_match_max_abs_diff
    > STEP17_METRIC_TOLERANCE
):
    print(
        "\nERROR: actual target in Step 17 predictions does not "
        "match the source development data."
    )
    print(
        f"Maximum absolute difference: "
        f"{target_match_max_abs_diff}"
    )
    sys.exit(1)

print(
    f"Prediction rows     : {len(pred):,}"
)
print(
    f"Models              : {pred_models}"
)
print(
    f"Validation years    : {pred_years}"
)
print(
    f"Target match max Δ  : {target_match_max_abs_diff:.12f}"
)
print(
    "PASS: Step 17 rolling predictions verified against source target."
)


# =============================================================================
# 7. REPRODUCE STEP 17 FOLD METRICS FROM SAVED PREDICTIONS
# =============================================================================

section(
    "3. STEP 17 PREDICTION REPRODUCTION CHECK"
)

step17_fold_metrics = pd.read_csv(
    STEP17_FOLD_METRICS_PATH
)

needed_fold_cols = {
    "model",
    "validation_year",
    "validation_mae",
    "validation_rmse",
    "validation_r2",
}

missing_fold_cols = sorted(
    needed_fold_cols
    - set(
        step17_fold_metrics.columns
    )
)

if missing_fold_cols:
    print(
        f"\nERROR: Step 17 fold metrics missing columns: "
        f"{missing_fold_cols}"
    )
    sys.exit(1)

repro_rows = []

for model_name in MODEL_ORDER:
    for year in ROLLING_VALIDATION_YEARS:
        sub = pred[
            (
                pred["model"]
                == model_name
            )
            & (
                pred["validation_year"]
                == year
            )
        ]

        y_true = sub[
            "actual_lst_c"
        ].to_numpy(
            dtype=float,
        )

        y_pred = sub[
            "predicted_lst_c"
        ].to_numpy(
            dtype=float,
        )

        mae = float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        )

        r2 = float(
            r2_score(
                y_true,
                y_pred,
            )
        )

        ref = step17_fold_metrics[
            (
                step17_fold_metrics["model"]
                == model_name
            )
            & (
                step17_fold_metrics["validation_year"]
                == year
            )
        ]

        if len(ref) != 1:
            print(
                f"\nERROR: expected one Step 17 fold metric row for "
                f"{model_name}, {year}; found {len(ref)}."
            )
            sys.exit(1)

        ref = ref.iloc[0]

        mae_diff = (
            mae
            - float(
                ref["validation_mae"]
            )
        )

        rmse_diff = (
            rmse
            - float(
                ref["validation_rmse"]
            )
        )

        r2_diff = (
            r2
            - float(
                ref["validation_r2"]
            )
        )

        status = (
            "PASS"
            if (
                abs(mae_diff)
                <= STEP17_METRIC_TOLERANCE
                and abs(rmse_diff)
                <= STEP17_METRIC_TOLERANCE
                and abs(r2_diff)
                <= STEP17_METRIC_TOLERANCE
            )
            else "FAIL"
        )

        repro_rows.append(
            {
                "model": model_name,
                "validation_year": year,
                "step17_mae": float(
                    ref["validation_mae"]
                ),
                "recomputed_mae": mae,
                "mae_difference": mae_diff,
                "step17_rmse": float(
                    ref["validation_rmse"]
                ),
                "recomputed_rmse": rmse,
                "rmse_difference": rmse_diff,
                "step17_r2": float(
                    ref["validation_r2"]
                ),
                "recomputed_r2": r2,
                "r2_difference": r2_diff,
                "status": status,
            }
        )

repro_df = pd.DataFrame(
    repro_rows
)

repro_df.to_csv(
    STEP17_REPRO_CHECK_PATH,
    index=False,
)

print(
    repro_df.to_string(
        index=False,
    )
)

if (
    repro_df["status"]
    != "PASS"
).any():
    print(
        "\nERROR: saved Step 17 predictions do not reproduce "
        "the finalized Step 17 fold metrics."
    )
    sys.exit(1)

print(
    "\nPASS: all 20 tuned rolling prediction sets reproduce Step 17."
)


# =============================================================================
# 8. TARGET DISTRIBUTION BY YEAR
# =============================================================================

section(
    "4. TARGET DISTRIBUTION BY YEAR"
)

target_year_rows = []

for year in DEVELOPMENT_YEARS:
    values = (
        df.loc[
            df[YEAR_COL] == year,
            TARGET,
        ]
        .to_numpy(
            dtype=float,
        )
    )

    target_year_rows.append(
        {
            "year": year,
            "rows": len(values),
            "mean_lst_c": float(
                np.mean(values)
            ),
            "median_lst_c": float(
                np.median(values)
            ),
            "std_lst_c": float(
                np.std(
                    values,
                    ddof=0,
                )
            ),
            "iqr_lst_c": float(
                np.quantile(
                    values,
                    0.75,
                )
                - np.quantile(
                    values,
                    0.25,
                )
            ),
            "min_lst_c": float(
                np.min(values)
            ),
            "max_lst_c": float(
                np.max(values)
            ),
            "tss_within_year": float(
                np.sum(
                    (
                        values
                        - np.mean(values)
                    )
                    ** 2
                )
            ),
        }
    )

target_year_stats = pd.DataFrame(
    target_year_rows
)

target_year_stats.to_csv(
    TARGET_YEAR_STATS_PATH,
    index=False,
)

print(
    target_year_stats.to_string(
        index=False,
    )
)


# =============================================================================
# 9. YEAR-LEVEL PREDICTION BIAS + R² DECOMPOSITION
# =============================================================================

section(
    "5. YEAR-LEVEL PREDICTION BIAS + R² DECOMPOSITION"
)

year_diag_rows = []

for model_name in MODEL_ORDER:
    print(
        f"\n--- {model_name} ---"
    )

    for year in ROLLING_VALIDATION_YEARS:
        sub = pred[
            (
                pred["model"]
                == model_name
            )
            & (
                pred["validation_year"]
                == year
            )
        ]

        diagnostics = regression_diagnostics(
            sub["actual_lst_c"]
            .to_numpy(
                dtype=float,
            ),
            sub["predicted_lst_c"]
            .to_numpy(
                dtype=float,
            ),
        )

        diagnostics.update(
            {
                "model": model_name,
                "validation_year": year,
            }
        )

        year_diag_rows.append(
            diagnostics
        )

        print(
            f"{year} | "
            f"Actual mean={diagnostics['actual_mean_c']:.3f} | "
            f"Pred mean={diagnostics['predicted_mean_c']:.3f} | "
            f"Bias={diagnostics['prediction_bias_pred_minus_actual_c']:+.3f} °C | "
            f"MAE={diagnostics['mae_c']:.3f} | "
            f"R²={diagnostics['r2']:.4f} | "
            f"Centered R²={diagnostics['centered_spatial_r2']:.4f} | "
            f"SSE/TSS={diagnostics['sse_over_tss']:.3f} | "
            f"Bias MSE share={diagnostics['bias_squared_share_of_mse']:.3f}"
        )

year_diag = pd.DataFrame(
    year_diag_rows
)

year_diag = year_diag[
    [
        "model",
        "validation_year",
        "rows",
        "actual_mean_c",
        "predicted_mean_c",
        "prediction_bias_pred_minus_actual_c",
        "actual_std_c",
        "predicted_std_c",
        "predicted_to_actual_std_ratio",
        "mae_c",
        "rmse_c",
        "mape_percent",
        "r2",
        "pearson_r",
        "spearman_rho",
        "sse",
        "tss",
        "sse_over_tss",
        "centered_spatial_mae_c",
        "centered_spatial_rmse_c",
        "centered_spatial_r2",
        "bias_squared_share_of_mse",
        "centered_error_share_of_mse",
        "negative_r2",
        "r2_failure_condition_sse_gt_tss",
    ]
]

year_diag.to_csv(
    YEAR_LEVEL_DIAG_PATH,
    index=False,
)


# =============================================================================
# 10. RESIDUAL DISTRIBUTION PATTERNS
# =============================================================================

section(
    "6. RESIDUAL DISTRIBUTION PATTERNS"
)

residual_rows = []

for model_name in MODEL_ORDER:
    for year in ROLLING_VALIDATION_YEARS:
        sub = pred[
            (
                pred["model"]
                == model_name
            )
            & (
                pred["validation_year"]
                == year
            )
        ].copy()

        residual = (
            sub[
                "residual_actual_minus_predicted"
            ]
            .to_numpy(
                dtype=float,
            )
        )

        abs_residual = np.abs(
            residual
        )

        residual_rows.append(
            {
                "model": model_name,
                "validation_year": year,
                "rows": len(residual),
                "mean_residual_actual_minus_predicted_c": float(
                    np.mean(
                        residual,
                    )
                ),
                "median_residual_c": float(
                    np.median(
                        residual,
                    )
                ),
                "residual_std_c": float(
                    np.std(
                        residual,
                        ddof=0,
                    )
                ),
                "residual_q05_c": float(
                    np.quantile(
                        residual,
                        0.05,
                    )
                ),
                "residual_q25_c": float(
                    np.quantile(
                        residual,
                        0.25,
                    )
                ),
                "residual_q75_c": float(
                    np.quantile(
                        residual,
                        0.75,
                    )
                ),
                "residual_q95_c": float(
                    np.quantile(
                        residual,
                        0.95,
                    )
                ),
                "median_absolute_residual_c": float(
                    np.median(
                        abs_residual,
                    )
                ),
                "q90_absolute_residual_c": float(
                    np.quantile(
                        abs_residual,
                        0.90,
                    )
                ),
                "underprediction_fraction_actual_gt_pred": float(
                    np.mean(
                        residual > 0
                    )
                ),
                "overprediction_fraction_actual_lt_pred": float(
                    np.mean(
                        residual < 0
                    )
                ),
            }
        )

residual_summary = pd.DataFrame(
    residual_rows
)

residual_summary.to_csv(
    RESIDUAL_SUMMARY_PATH,
    index=False,
)

print(
    residual_summary[
        residual_summary[
            "validation_year"
        ].isin(
            PROBLEM_YEARS
        )
    ].to_string(
        index=False,
    )
)


# =============================================================================
# 11. FEATURE DISTRIBUTION SHIFT - STRICT ROLLING ORIGIN
# =============================================================================

section(
    "7. FEATURE DISTRIBUTION SHIFT"
)

shift_rows = []

for val_year in ROLLING_VALIDATION_YEARS:
    train_mask = (
        df[YEAR_COL]
        < val_year
    )

    val_mask = (
        df[YEAR_COL]
        == val_year
    )

    train_years = sorted(
        df.loc[
            train_mask,
            YEAR_COL,
        ]
        .unique()
        .tolist()
    )

    print(
        f"\nTrain {train_years[0]}-{train_years[-1]} "
        f"-> validation {val_year}"
    )

    for feature in predictors:
        train_values = (
            df.loc[
                train_mask,
                feature,
            ]
            .to_numpy(
                dtype=float,
            )
        )

        val_values = (
            df.loc[
                val_mask,
                feature,
            ]
            .to_numpy(
                dtype=float,
            )
        )

        smd, constant_train = (
            standardized_mean_difference(
                train_values,
                val_values,
            )
        )

        ks = ks_2samp(
            train_values,
            val_values,
            alternative="two-sided",
            method="auto",
        )

        psi = population_stability_index(
            train_values,
            val_values,
            quantile_bins=10,
        )

        raw_wasserstein = float(
            wasserstein_distance(
                train_values,
                val_values,
            )
        )

        train_std = float(
            np.std(
                train_values,
                ddof=0,
            )
        )

        normalized_wasserstein = (
            raw_wasserstein
            / train_std
            if train_std > EPS
            else np.nan
        )

        shift_rows.append(
            {
                "validation_year": val_year,
                "train_start_year": int(
                    train_years[0]
                ),
                "train_end_year": int(
                    train_years[-1]
                ),
                "feature": feature,
                "is_year_feature": (
                    feature == YEAR_COL
                ),
                "train_mean": float(
                    np.mean(
                        train_values,
                    )
                ),
                "validation_mean": float(
                    np.mean(
                        val_values,
                    )
                ),
                "train_std": train_std,
                "validation_std": float(
                    np.std(
                        val_values,
                        ddof=0,
                    )
                ),
                "standardized_mean_difference": smd,
                "abs_standardized_mean_difference": (
                    abs(smd)
                    if pd.notna(smd)
                    else np.nan
                ),
                "train_constant": constant_train,
                "ks_statistic": float(
                    ks.statistic
                ),
                "ks_pvalue": float(
                    ks.pvalue
                ),
                "psi": psi,
                "wasserstein_distance_raw": raw_wasserstein,
                "wasserstein_distance_over_train_std": normalized_wasserstein,
            }
        )

feature_shift = pd.DataFrame(
    shift_rows
)

feature_shift.to_csv(
    FEATURE_SHIFT_PATH,
    index=False,
)

feature_shift_summary = (
    feature_shift.groupby(
        [
            "feature",
            "is_year_feature",
        ],
        as_index=False,
    )
    .agg(
        evaluated_folds=(
            "validation_year",
            "count",
        ),
        mean_abs_smd=(
            "abs_standardized_mean_difference",
            "mean",
        ),
        max_abs_smd=(
            "abs_standardized_mean_difference",
            "max",
        ),
        mean_ks=(
            "ks_statistic",
            "mean",
        ),
        max_ks=(
            "ks_statistic",
            "max",
        ),
        mean_psi=(
            "psi",
            "mean",
        ),
        max_psi=(
            "psi",
            "max",
        ),
        mean_normalized_wasserstein=(
            "wasserstein_distance_over_train_std",
            "mean",
        ),
        max_normalized_wasserstein=(
            "wasserstein_distance_over_train_std",
            "max",
        ),
    )
)

feature_shift_summary[
    "distribution_shift_flag"
] = (
    (
        feature_shift_summary[
            "max_psi"
        ]
        >= PSI_MODERATE_SHIFT
    )
    |
    (
        feature_shift_summary[
            "max_abs_smd"
        ]
        >= ABS_SMD_MODERATE_SHIFT
    )
)

feature_shift_summary = (
    feature_shift_summary.sort_values(
        [
            "distribution_shift_flag",
            "mean_psi",
            "mean_ks",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
    .reset_index(
        drop=True,
    )
)

feature_shift_summary.to_csv(
    FEATURE_SHIFT_SUMMARY_PATH,
    index=False,
)

year_shift_rows = []

non_year_shift = feature_shift[
    ~feature_shift[
        "is_year_feature"
    ]
].copy()

for year in ROLLING_VALIDATION_YEARS:
    ys = non_year_shift[
        non_year_shift[
            "validation_year"
        ]
        == year
    ]

    year_shift_rows.append(
        {
            "validation_year": year,
            "features_evaluated": len(ys),
            "mean_psi": float(
                ys["psi"].mean()
            ),
            "median_psi": float(
                ys["psi"].median()
            ),
            "max_psi": float(
                ys["psi"].max()
            ),
            "features_psi_ge_0_10": int(
                (
                    ys["psi"]
                    >= 0.10
                ).sum()
            ),
            "features_psi_ge_0_25": int(
                (
                    ys["psi"]
                    >= PSI_MODERATE_SHIFT
                ).sum()
            ),
            "mean_abs_smd": float(
                ys[
                    "abs_standardized_mean_difference"
                ].mean()
            ),
            "features_abs_smd_ge_0_50": int(
                (
                    ys[
                        "abs_standardized_mean_difference"
                    ]
                    >= ABS_SMD_MODERATE_SHIFT
                ).sum()
            ),
            "mean_ks": float(
                ys["ks_statistic"].mean()
            ),
            "max_ks": float(
                ys["ks_statistic"].max()
            ),
        }
    )

year_shift_summary = pd.DataFrame(
    year_shift_rows
)

year_shift_summary.to_csv(
    YEAR_SHIFT_SUMMARY_PATH,
    index=False,
)

print(
    "\nOverall top 20 shifted non-year features by mean PSI:"
)

print(
    feature_shift_summary[
        ~feature_shift_summary[
            "is_year_feature"
        ]
    ]
    .head(20)
    .to_string(
        index=False,
    )
)

print(
    "\nYear-level shift summary:"
)

print(
    year_shift_summary.to_string(
        index=False,
    )
)


# =============================================================================
# 12. FEATURE-LST RELATIONSHIP STABILITY ACROSS YEARS
# =============================================================================

section(
    "8. FEATURE-LST RELATIONSHIP STABILITY"
)

assoc_rows = []

for year in DEVELOPMENT_YEARS:
    year_df = df[
        df[YEAR_COL] == year
    ]

    y_year = (
        year_df[TARGET]
        .to_numpy(
            dtype=float,
        )
    )

    for feature in predictors:
        x = (
            year_df[feature]
            .to_numpy(
                dtype=float,
            )
        )

        rho = safe_spearman(
            x,
            y_year,
        )

        assoc_rows.append(
            {
                "year": year,
                "feature": feature,
                "is_year_feature": (
                    feature == YEAR_COL
                ),
                "feature_constant_within_year": bool(
                    np.std(
                        x,
                        ddof=0,
                    )
                    < EPS
                ),
                "spearman_rho_with_lst": rho,
                "abs_spearman_rho_with_lst": (
                    abs(rho)
                    if pd.notna(rho)
                    else np.nan
                ),
                "meaningful_sign": meaningful_sign(
                    rho
                ),
            }
        )

feature_year_assoc = pd.DataFrame(
    assoc_rows
)

feature_year_assoc.to_csv(
    FEATURE_YEAR_ASSOC_PATH,
    index=False,
)

stability_rows = []

for feature in predictors:
    sub = (
        feature_year_assoc[
            feature_year_assoc[
                "feature"
            ]
            == feature
        ]
        .sort_values(
            "year"
        )
    )

    rho_map = {
        int(row["year"]): row[
            "spearman_rho_with_lst"
        ]
        for _, row in sub.iterrows()
    }

    finite_rhos = [
        float(value)
        for value in rho_map.values()
        if pd.notna(value)
    ]

    meaningful_signs = [
        meaningful_sign(
            rho_map[year]
        )
        for year in DEVELOPMENT_YEARS
        if pd.notna(
            rho_map[year]
        )
    ]

    nonzero_signs = [
        sign
        for sign in meaningful_signs
        if sign != 0
    ]

    if nonzero_signs:
        pos = sum(
            sign > 0
            for sign in nonzero_signs
        )

        neg = sum(
            sign < 0
            for sign in nonzero_signs
        )

        sign_consistency = (
            max(
                pos,
                neg,
            )
            / len(
                nonzero_signs
            )
        )
    else:
        sign_consistency = np.nan

    prior_2021 = [
        rho_map[year]
        for year in [
            2018,
            2019,
            2020,
        ]
        if pd.notna(
            rho_map[year]
        )
    ]

    prior_2023 = [
        rho_map[year]
        for year in [
            2018,
            2019,
            2020,
            2021,
            2022,
        ]
        if pd.notna(
            rho_map[year]
        )
    ]

    prior_2021_median = (
        float(
            np.median(
                prior_2021,
            )
        )
        if prior_2021
        else np.nan
    )

    prior_2023_median = (
        float(
            np.median(
                prior_2023,
            )
        )
        if prior_2023
        else np.nan
    )

    rho_2021 = rho_map.get(
        2021,
        np.nan,
    )

    rho_2023 = rho_map.get(
        2023,
        np.nan,
    )

    delta_2021 = (
        rho_2021
        - prior_2021_median
        if (
            pd.notna(
                rho_2021
            )
            and pd.notna(
                prior_2021_median
            )
        )
        else np.nan
    )

    delta_2023 = (
        rho_2023
        - prior_2023_median
        if (
            pd.notna(
                rho_2023
            )
            and pd.notna(
                prior_2023_median
            )
        )
        else np.nan
    )

    if finite_rhos:
        rho_mean = float(
            np.mean(
                finite_rhos,
            )
        )

        mean_abs_rho = float(
            np.mean(
                np.abs(
                    finite_rhos,
                )
            )
        )

        rho_std = float(
            np.std(
                finite_rhos,
                ddof=0,
            )
        )

        rho_min = float(
            np.min(
                finite_rhos,
            )
        )

        rho_max = float(
            np.max(
                finite_rhos,
            )
        )

        rho_range = (
            rho_max
            - rho_min
        )
    else:
        rho_mean = np.nan
        mean_abs_rho = np.nan
        rho_std = np.nan
        rho_min = np.nan
        rho_max = np.nan
        rho_range = np.nan

    stability_rows.append(
        {
            "feature": feature,
            "is_year_feature": (
                feature == YEAR_COL
            ),
            "years_with_valid_rho": len(
                finite_rhos
            ),
            "mean_rho": rho_mean,
            "mean_abs_rho": mean_abs_rho,
            "rho_std": rho_std,
            "rho_min": rho_min,
            "rho_max": rho_max,
            "rho_range": rho_range,
            "meaningful_sign_consistency": sign_consistency,
            "meaningful_sign_flip_count": chronological_sign_flips(
                [
                    rho_map.get(
                        year,
                        np.nan,
                    )
                    for year in DEVELOPMENT_YEARS
                ]
            ),
            "rho_2018": rho_map.get(
                2018,
                np.nan,
            ),
            "rho_2019": rho_map.get(
                2019,
                np.nan,
            ),
            "rho_2020": rho_map.get(
                2020,
                np.nan,
            ),
            "rho_2021": rho_2021,
            "rho_2022": rho_map.get(
                2022,
                np.nan,
            ),
            "rho_2023": rho_2023,
            "prior_median_rho_for_2021": prior_2021_median,
            "rho_jump_2021_vs_prior_median": delta_2021,
            "prior_median_rho_for_2023": prior_2023_median,
            "rho_jump_2023_vs_prior_median": delta_2023,
        }
    )

feature_stability = pd.DataFrame(
    stability_rows
)

feature_stability = feature_stability.merge(
    feature_shift_summary[
        [
            "feature",
            "mean_abs_smd",
            "max_abs_smd",
            "mean_ks",
            "max_ks",
            "mean_psi",
            "max_psi",
            "distribution_shift_flag",
        ]
    ],
    on="feature",
    how="left",
    validate="one_to_one",
)

feature_stability[
    "relationship_instability_flag"
] = (
    (
        feature_stability[
            "rho_range"
        ]
        >= RHO_RANGE_INSTABILITY
    )
    |
    (
        feature_stability[
            "meaningful_sign_flip_count"
        ]
        >= 1
    )
)

problem_jump_abs = np.maximum(
    np.abs(
        feature_stability[
            "rho_jump_2021_vs_prior_median"
        ].fillna(
            0.0
        )
    ),
    np.abs(
        feature_stability[
            "rho_jump_2023_vs_prior_median"
        ].fillna(
            0.0
        )
    ),
)

feature_stability[
    "problem_year_relationship_jump_flag"
] = (
    problem_jump_abs
    >= PROBLEM_YEAR_RHO_JUMP
)

feature_stability[
    "diagnostic_instability_signal_count"
] = (
    feature_stability[
        "distribution_shift_flag"
    ].astype(int)
    + feature_stability[
        "relationship_instability_flag"
    ].astype(int)
    + feature_stability[
        "problem_year_relationship_jump_flag"
    ].astype(int)
)

feature_stability = (
    feature_stability.sort_values(
        [
            "diagnostic_instability_signal_count",
            "rho_range",
            "mean_psi",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
    .reset_index(
        drop=True,
    )
)

feature_stability.to_csv(
    FEATURE_STABILITY_PATH,
    index=False,
)

print(
    "\nTop diagnostic instability signals "
    "(NOT a feature-selection result):"
)

print(
    feature_stability[
        ~feature_stability[
            "is_year_feature"
        ]
    ]
    .head(25)
    .to_string(
        index=False,
    )
)


# =============================================================================
# 13. RESIDUAL-FEATURE ASSOCIATIONS
# =============================================================================

section(
    "9. RESIDUAL-FEATURE ASSOCIATIONS"
)

# Keep the original `year` predictor and create a separate merge key.
# `predictors` already contains YEAR_COL, so selecting YEAR_COL again and then
# renaming it would create duplicate `validation_year` columns.
feature_lookup = df[
    [
        GRID_COL,
        *predictors,
    ]
].copy()

feature_lookup["validation_year"] = (
    feature_lookup[YEAR_COL]
    .astype(int)
)

pred_features = pred.merge(
    feature_lookup,
    on=[
        GRID_COL,
        "validation_year",
    ],
    how="left",
    validate="many_to_one",
)

if pred_features[
    predictors
].isna().any().any():
    print(
        "\nERROR: residual-feature merge produced missing feature values."
    )
    sys.exit(1)

residual_assoc_rows = []

for model_name in MODEL_ORDER:
    for year in ROLLING_VALIDATION_YEARS:
        sub = pred_features[
            (
                pred_features["model"]
                == model_name
            )
            & (
                pred_features["validation_year"]
                == year
            )
        ]

        residual = (
            sub[
                "residual_actual_minus_predicted"
            ]
            .to_numpy(
                dtype=float,
            )
        )

        for feature in predictors:
            rho = safe_spearman(
                sub[feature]
                .to_numpy(
                    dtype=float,
                ),
                residual,
            )

            residual_assoc_rows.append(
                {
                    "model": model_name,
                    "validation_year": year,
                    "feature": feature,
                    "is_year_feature": (
                        feature == YEAR_COL
                    ),
                    "spearman_rho_feature_vs_residual": rho,
                    "abs_spearman_rho_feature_vs_residual": (
                        abs(rho)
                        if pd.notna(rho)
                        else np.nan
                    ),
                }
            )

residual_feature_assoc = pd.DataFrame(
    residual_assoc_rows
)

residual_feature_assoc.to_csv(
    RESIDUAL_FEATURE_ASSOC_PATH,
    index=False,
)

for year in PROBLEM_YEARS:
    print(
        f"\nTop residual associations in {year}:"
    )

    top_problem = (
        residual_feature_assoc[
            (
                residual_feature_assoc[
                    "validation_year"
                ]
                == year
            )
            & (
                ~residual_feature_assoc[
                    "is_year_feature"
                ]
            )
        ]
        .sort_values(
            "abs_spearman_rho_feature_vs_residual",
            ascending=False,
        )
        .groupby(
            "model",
            group_keys=False,
        )
        .head(5)
    )

    print(
        top_problem[
            [
                "model",
                "feature",
                "spearman_rho_feature_vs_residual",
                "abs_spearman_rho_feature_vs_residual",
            ]
        ].to_string(
            index=False,
        )
    )


# =============================================================================
# 14. SPATIAL AUTOCORRELATION OF TEMPORAL RESIDUALS
# =============================================================================

section(
    "10. RESIDUAL SPATIAL AUTOCORRELATION - MORAN'S I"
)

gdf = gpd.read_file(
    GEOMETRY_PATH
)

if GRID_COL not in gdf.columns:
    print(
        f"\nERROR: {GRID_COL} missing from geometry."
    )
    sys.exit(1)

gdf[GRID_COL] = (
    gdf[GRID_COL]
    .astype(str)
)

gdf = (
    gdf[
        [
            GRID_COL,
            "geometry",
        ]
    ]
    .drop_duplicates(
        subset=[
            GRID_COL,
        ]
    )
    .copy()
)

if (
    gdf[GRID_COL].nunique()
    != EXPECTED_UNIQUE_GRIDS
):
    print(
        f"\nERROR: geometry grid count is "
        f"{gdf[GRID_COL].nunique()}; "
        f"expected {EXPECTED_UNIQUE_GRIDS}."
    )
    sys.exit(1)

gdf = gdf.set_index(
    GRID_COL,
    drop=False,
)

with warnings.catch_warnings():
    warnings.simplefilter(
        "ignore"
    )

    spatial_weights = Queen.from_dataframe(
        gdf,
        use_index=True,
    )

spatial_weights.transform = "r"

islands = list(
    spatial_weights.islands
)

grid_order = list(
    spatial_weights.id_order
)

print(
    f"Queen weights grids : {len(grid_order):,}"
)
print(
    f"Spatial islands     : {len(islands)}"
)

moran_rows = []

for model_name in MODEL_ORDER:
    for year in ROLLING_VALIDATION_YEARS:
        sub = (
            pred[
                (
                    pred["model"]
                    == model_name
                )
                & (
                    pred[
                        "validation_year"
                    ]
                    == year
                )
            ]
            .set_index(
                GRID_COL
            )
        )

        residual_series = (
            sub[
                "residual_actual_minus_predicted"
            ]
            .reindex(
                grid_order
            )
        )

        if residual_series.isna().any():
            print(
                f"\nERROR: residual grid alignment failed for "
                f"{model_name}, {year}."
            )
            sys.exit(1)

        residual_values = (
            residual_series
            .to_numpy(
                dtype=float,
            )
        )

        moran = Moran(
            residual_values,
            spatial_weights,
            permutations=MORAN_PERMUTATIONS,
            two_tailed=True,
        )

        moran_rows.append(
            {
                "model": model_name,
                "validation_year": year,
                "moran_i": float(
                    moran.I
                ),
                "expected_i": float(
                    moran.EI
                ),
                "permutation_pvalue": float(
                    moran.p_sim
                ),
                "simulation_z_score": float(
                    moran.z_sim
                ),
                "permutations": MORAN_PERMUTATIONS,
                "positive_significant_clustering_p_lt_0_05": bool(
                    (
                        moran.I > 0
                    )
                    and (
                        moran.p_sim
                        < 0.05
                    )
                ),
            }
        )

        print(
            f"{model_name:<14} | {year} | "
            f"Moran I={moran.I:.4f} | "
            f"p={moran.p_sim:.4f}"
        )

residual_moran = pd.DataFrame(
    moran_rows
)

residual_moran.to_csv(
    RESIDUAL_MORAN_PATH,
    index=False,
)


# =============================================================================
# 15. BUILD 2021 / 2023 PROBLEM-YEAR EVIDENCE TABLE
# =============================================================================

section(
    "11. 2021 / 2023 PROBLEM-YEAR EVIDENCE"
)

problem_rows = []

for year in PROBLEM_YEARS:
    year_shift = year_shift_summary[
        year_shift_summary[
            "validation_year"
        ]
        == year
    ].iloc[0]

    for model_name in MODEL_ORDER:
        diag = year_diag[
            (
                year_diag["model"]
                == model_name
            )
            & (
                year_diag[
                    "validation_year"
                ]
                == year
            )
        ].iloc[0]

        moran = residual_moran[
            (
                residual_moran[
                    "model"
                ]
                == model_name
            )
            & (
                residual_moran[
                    "validation_year"
                ]
                == year
            )
        ].iloc[0]

        problem_rows.append(
            {
                "validation_year": year,
                "model": model_name,
                "raw_r2": diag["r2"],
                "sse_over_tss": diag["sse_over_tss"],
                "actual_std_c": diag["actual_std_c"],
                "predicted_std_c": diag["predicted_std_c"],
                "predicted_to_actual_std_ratio": diag[
                    "predicted_to_actual_std_ratio"
                ],
                "prediction_bias_pred_minus_actual_c": diag[
                    "prediction_bias_pred_minus_actual_c"
                ],
                "bias_squared_share_of_mse": diag[
                    "bias_squared_share_of_mse"
                ],
                "centered_spatial_r2": diag[
                    "centered_spatial_r2"
                ],
                "centered_spatial_rmse_c": diag[
                    "centered_spatial_rmse_c"
                ],
                "pearson_r": diag["pearson_r"],
                "spearman_rho": diag["spearman_rho"],
                "residual_moran_i": moran[
                    "moran_i"
                ],
                "residual_moran_p": moran[
                    "permutation_pvalue"
                ],
                "residual_spatial_clustering_significant": moran[
                    "positive_significant_clustering_p_lt_0_05"
                ],
                "year_mean_psi_non_year_features": year_shift[
                    "mean_psi"
                ],
                "year_max_psi_non_year_features": year_shift[
                    "max_psi"
                ],
                "features_psi_ge_0_25": int(
                    year_shift[
                        "features_psi_ge_0_25"
                    ]
                ),
                "features_abs_smd_ge_0_50": int(
                    year_shift[
                        "features_abs_smd_ge_0_50"
                    ]
                ),
                "negative_r2_confirmed": bool(
                    diag["r2"] < 0
                ),
                "raw_r2_failure_is_sse_gt_tss": bool(
                    diag[
                        "r2_failure_condition_sse_gt_tss"
                    ]
                ),
                "annual_level_bias_major_mse_component": bool(
                    diag[
                        "bias_squared_share_of_mse"
                    ]
                    >= 0.50
                ),
                "centered_spatial_pattern_still_positive_r2": bool(
                    diag[
                        "centered_spatial_r2"
                    ]
                    > 0
                ),
            }
        )

problem_year_evidence = pd.DataFrame(
    problem_rows
)

problem_year_evidence.to_csv(
    PROBLEM_YEAR_EVIDENCE_PATH,
    index=False,
)

print(
    problem_year_evidence.to_string(
        index=False,
    )
)


# =============================================================================
# 16. FIGURES
# =============================================================================

section(
    "12. FIGURES"
)

# ---- Actual vs predicted annual mean ----
actual_mean_by_year = (
    pred.groupby(
        "validation_year"
    )[
        "actual_lst_c"
    ]
    .mean()
    .sort_index()
)

fig, ax = plt.subplots(
    figsize=(
        10,
        6,
    )
)

ax.plot(
    actual_mean_by_year.index,
    actual_mean_by_year.values,
    marker="o",
    linewidth=2,
    label="Actual LST mean",
)

for model_name in MODEL_ORDER:
    model_year = (
        year_diag[
            year_diag[
                "model"
            ]
            == model_name
        ]
        .sort_values(
            "validation_year"
        )
    )

    ax.plot(
        model_year[
            "validation_year"
        ],
        model_year[
            "predicted_mean_c"
        ],
        marker="o",
        label=model_name,
    )

ax.set_title(
    "Step 19A - Actual vs Predicted Annual Mean LST"
)
ax.set_xlabel(
    "Validation year"
)
ax.set_ylabel(
    "Mean LST (°C)"
)
ax.set_xticks(
    ROLLING_VALIDATION_YEARS
)
ax.grid(
    alpha=0.2
)
ax.legend()

plt.tight_layout()
plt.savefig(
    FIG_YEAR_MEANS,
    dpi=300,
    bbox_inches="tight",
)
plt.close(
    fig
)


# ---- Raw R² ----
fig, ax = plt.subplots(
    figsize=(
        10,
        6,
    )
)

for model_name in MODEL_ORDER:
    model_year = (
        year_diag[
            year_diag[
                "model"
            ]
            == model_name
        ]
        .sort_values(
            "validation_year"
        )
    )

    ax.plot(
        model_year[
            "validation_year"
        ],
        model_year[
            "r2"
        ],
        marker="o",
        label=model_name,
    )

ax.axhline(
    0,
    linewidth=1,
)
ax.set_title(
    "Step 19A - Raw Temporal R² by Validation Year"
)
ax.set_xlabel(
    "Validation year"
)
ax.set_ylabel(
    "R²"
)
ax.set_xticks(
    ROLLING_VALIDATION_YEARS
)
ax.grid(
    alpha=0.2
)
ax.legend()

plt.tight_layout()
plt.savefig(
    FIG_RAW_R2,
    dpi=300,
    bbox_inches="tight",
)
plt.close(
    fig
)


# ---- Centered spatial R² ----
fig, ax = plt.subplots(
    figsize=(
        10,
        6,
    )
)

for model_name in MODEL_ORDER:
    model_year = (
        year_diag[
            year_diag[
                "model"
            ]
            == model_name
        ]
        .sort_values(
            "validation_year"
        )
    )

    ax.plot(
        model_year[
            "validation_year"
        ],
        model_year[
            "centered_spatial_r2"
        ],
        marker="o",
        label=model_name,
    )

ax.axhline(
    0,
    linewidth=1,
)
ax.set_title(
    "Step 19A - Centered Spatial-Pattern R² by Year"
)
ax.set_xlabel(
    "Validation year"
)
ax.set_ylabel(
    "Centered spatial-pattern R²"
)
ax.set_xticks(
    ROLLING_VALIDATION_YEARS
)
ax.grid(
    alpha=0.2
)
ax.legend()

plt.tight_layout()
plt.savefig(
    FIG_CENTERED_R2,
    dpi=300,
    bbox_inches="tight",
)
plt.close(
    fig
)


# ---- Prediction bias ----
fig, ax = plt.subplots(
    figsize=(
        10,
        6,
    )
)

for model_name in MODEL_ORDER:
    model_year = (
        year_diag[
            year_diag[
                "model"
            ]
            == model_name
        ]
        .sort_values(
            "validation_year"
        )
    )

    ax.plot(
        model_year[
            "validation_year"
        ],
        model_year[
            "prediction_bias_pred_minus_actual_c"
        ],
        marker="o",
        label=model_name,
    )

ax.axhline(
    0,
    linewidth=1,
)
ax.set_title(
    "Step 19A - Annual Prediction Bias"
)
ax.set_xlabel(
    "Validation year"
)
ax.set_ylabel(
    "Predicted mean - actual mean (°C)"
)
ax.set_xticks(
    ROLLING_VALIDATION_YEARS
)
ax.grid(
    alpha=0.2
)
ax.legend()

plt.tight_layout()
plt.savefig(
    FIG_BIAS,
    dpi=300,
    bbox_inches="tight",
)
plt.close(
    fig
)


# ---- Top distribution shift features ----
shift_plot = (
    feature_shift_summary[
        ~feature_shift_summary[
            "is_year_feature"
        ]
    ]
    .dropna(
        subset=[
            "mean_psi",
        ]
    )
    .sort_values(
        "mean_psi",
        ascending=False,
    )
    .head(20)
    .sort_values(
        "mean_psi",
        ascending=True,
    )
)

fig, ax = plt.subplots(
    figsize=(
        10,
        8,
    )
)

ax.barh(
    shift_plot[
        "feature"
    ],
    shift_plot[
        "mean_psi"
    ],
)

ax.set_title(
    "Step 19A - Top 20 Mean PSI Distribution Shifts"
)
ax.set_xlabel(
    "Mean PSI across rolling validation years"
)
ax.set_ylabel(
    "Feature"
)
ax.grid(
    axis="x",
    alpha=0.2,
)

plt.tight_layout()
plt.savefig(
    FIG_SHIFT,
    dpi=300,
    bbox_inches="tight",
)
plt.close(
    fig
)


# ---- Residual Moran's I ----
fig, ax = plt.subplots(
    figsize=(
        10,
        6,
    )
)

for model_name in MODEL_ORDER:
    model_moran = (
        residual_moran[
            residual_moran[
                "model"
            ]
            == model_name
        ]
        .sort_values(
            "validation_year"
        )
    )

    ax.plot(
        model_moran[
            "validation_year"
        ],
        model_moran[
            "moran_i"
        ],
        marker="o",
        label=model_name,
    )

ax.axhline(
    0,
    linewidth=1,
)
ax.set_title(
    "Step 19A - Residual Spatial Autocorrelation"
)
ax.set_xlabel(
    "Validation year"
)
ax.set_ylabel(
    "Moran's I of residuals"
)
ax.set_xticks(
    ROLLING_VALIDATION_YEARS
)
ax.grid(
    alpha=0.2
)
ax.legend()

plt.tight_layout()
plt.savefig(
    FIG_MORAN,
    dpi=300,
    bbox_inches="tight",
)
plt.close(
    fig
)

for figure_path in [
    FIG_YEAR_MEANS,
    FIG_RAW_R2,
    FIG_CENTERED_R2,
    FIG_BIAS,
    FIG_SHIFT,
    FIG_MORAN,
]:
    print(
        f"Saved: {figure_path}"
    )


# =============================================================================
# 17. STRICT STEP 19A AUDIT
# =============================================================================

section(
    "13. STRICT STEP 19A AUDIT"
)

expected_feature_shift_rows = (
    len(
        ROLLING_VALIDATION_YEARS
    )
    * len(
        predictors
    )
)

expected_feature_assoc_rows = (
    len(
        DEVELOPMENT_YEARS
    )
    * len(
        predictors
    )
)

expected_residual_assoc_rows = (
    len(
        MODEL_ORDER
    )
    * len(
        ROLLING_VALIDATION_YEARS
    )
    * len(
        predictors
    )
)

expected_year_diag_rows = (
    len(
        MODEL_ORDER
    )
    * len(
        ROLLING_VALIDATION_YEARS
    )
)

expected_moran_rows = (
    expected_year_diag_rows
)

core_diag_finite = bool(
    np.isfinite(
        year_diag[
            [
                "actual_mean_c",
                "predicted_mean_c",
                "prediction_bias_pred_minus_actual_c",
                "actual_std_c",
                "predicted_std_c",
                "mae_c",
                "rmse_c",
                "r2",
                "sse",
                "tss",
                "sse_over_tss",
                "centered_spatial_rmse_c",
                "centered_spatial_r2",
                "bias_squared_share_of_mse",
                "centered_error_share_of_mse",
            ]
        ]
        .to_numpy(
            dtype=float,
        )
    ).all()
)

negative_r2_rows = int(
    year_diag[
        "negative_r2"
    ].sum()
)

expected_negative_r2_rows = 8
# Step 17 finalized result:
# 2 negative-R² folds per model × 4 models = 8.

problem_negative_rows = int(
    year_diag[
        year_diag[
            "validation_year"
        ].isin(
            PROBLEM_YEARS
        )
    ][
        "negative_r2"
    ].sum()
)

audit_rows = [
    {
        "check": "step17_audit",
        "status": "PASS",
        "details": (
            "Step 17 audit loaded and fully PASS."
        ),
    },
    {
        "check": "step18_audit",
        "status": "PASS",
        "details": (
            "Step 18 audit loaded and fully PASS."
        ),
    },
    {
        "check": "development_years",
        "status": (
            "PASS"
            if years
            == DEVELOPMENT_YEARS
            else "FAIL"
        ),
        "details": str(
            years
        ),
    },
    {
        "check": "predictor_count",
        "status": (
            "PASS"
            if len(
                predictors
            )
            == EXPECTED_PREDICTOR_COUNT
            else "FAIL"
        ),
        "details": (
            f"actual={len(predictors)}, "
            f"expected={EXPECTED_PREDICTOR_COUNT}"
        ),
    },
    {
        "check": "prediction_row_count",
        "status": (
            "PASS"
            if len(
                pred
            )
            == EXPECTED_PREDICTION_ROWS
            else "FAIL"
        ),
        "details": (
            f"actual={len(pred)}, "
            f"expected={EXPECTED_PREDICTION_ROWS}"
        ),
    },
    {
        "check": "prediction_target_matches_source",
        "status": (
            "PASS"
            if target_match_max_abs_diff
            <= STEP17_METRIC_TOLERANCE
            else "FAIL"
        ),
        "details": (
            f"max_abs_difference="
            f"{target_match_max_abs_diff:.12f}"
        ),
    },
    {
        "check": "step17_fold_metric_reproduction",
        "status": (
            "PASS"
            if (
                repro_df[
                    "status"
                ]
                == "PASS"
            ).all()
            else "FAIL"
        ),
        "details": (
            f"PASS rows="
            f"{int((repro_df['status'] == 'PASS').sum())}/20"
        ),
    },
    {
        "check": "year_diagnostic_rows",
        "status": (
            "PASS"
            if len(
                year_diag
            )
            == expected_year_diag_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(year_diag)}, "
            f"expected={expected_year_diag_rows}"
        ),
    },
    {
        "check": "feature_shift_rows",
        "status": (
            "PASS"
            if len(
                feature_shift
            )
            == expected_feature_shift_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(feature_shift)}, "
            f"expected={expected_feature_shift_rows}"
        ),
    },
    {
        "check": "feature_year_association_rows",
        "status": (
            "PASS"
            if len(
                feature_year_assoc
            )
            == expected_feature_assoc_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(feature_year_assoc)}, "
            f"expected={expected_feature_assoc_rows}"
        ),
    },
    {
        "check": "feature_stability_rows",
        "status": (
            "PASS"
            if len(
                feature_stability
            )
            == EXPECTED_PREDICTOR_COUNT
            else "FAIL"
        ),
        "details": (
            f"actual={len(feature_stability)}, "
            f"expected={EXPECTED_PREDICTOR_COUNT}"
        ),
    },
    {
        "check": "residual_feature_association_rows",
        "status": (
            "PASS"
            if len(
                residual_feature_assoc
            )
            == expected_residual_assoc_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(residual_feature_assoc)}, "
            f"expected={expected_residual_assoc_rows}"
        ),
    },
    {
        "check": "residual_moran_rows",
        "status": (
            "PASS"
            if len(
                residual_moran
            )
            == expected_moran_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(residual_moran)}, "
            f"expected={expected_moran_rows}"
        ),
    },
    {
        "check": "core_diagnostics_finite",
        "status": (
            "PASS"
            if core_diag_finite
            else "FAIL"
        ),
        "details": str(
            core_diag_finite
        ),
    },
    {
        "check": "negative_r2_preserved",
        "status": (
            "PASS"
            if negative_r2_rows
            == expected_negative_r2_rows
            else "FAIL"
        ),
        "details": (
            f"actual_negative_rows={negative_r2_rows}, "
            f"expected={expected_negative_r2_rows}; "
            "negative R² was diagnosed, not altered."
        ),
    },
    {
        "check": "problem_year_negative_r2_rows",
        "status": (
            "PASS"
            if problem_negative_rows
            == 8
            else "FAIL"
        ),
        "details": (
            f"2021/2023 negative model-year rows="
            f"{problem_negative_rows}, expected=8"
        ),
    },
    {
        "check": "2024_loaded_or_used",
        "status": "PASS",
        "details": (
            "No. 2024 was not loaded by Step 19A."
        ),
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": (
            "No. 2025 was not loaded by Step 19A."
        ),
    },
    {
        "check": "model_training_performed",
        "status": "PASS",
        "details": (
            "No. Step 19A uses finalized Step 17 saved predictions."
        ),
    },
    {
        "check": "hyperparameter_tuning_performed",
        "status": "PASS",
        "details": (
            "No tuning occurs in Step 19A."
        ),
    },
    {
        "check": "temporal_improvement_applied",
        "status": "PASS",
        "details": (
            "No weighting, feature reduction, bias correction, "
            "ensemble, or level/anomaly strategy was applied."
        ),
    },
    {
        "check": "feature_selection_performed",
        "status": "PASS",
        "details": (
            "No. Instability flags are diagnostic heuristics only."
        ),
    },
    {
        "check": "final_model_selected",
        "status": "PASS",
        "details": (
            "No. Step 19A performs diagnosis only."
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

print(
    audit_df.to_string(
        index=False,
    )
)

if (
    audit_df["status"]
    != "PASS"
).any():
    print(
        "\nERROR: one or more Step 19A audit checks failed."
    )
    sys.exit(1)


# =============================================================================
# 18. REPORT
# =============================================================================

section(
    "14. SAVE STEP 19A REPORT"
)

report_lines = [
    "STEP 19A - TEMPORAL DIAGNOSIS",
    "=" * 112,
    "",
    "SCOPE",
    "-" * 112,
    "Diagnosis only. No temporal improvement strategy was applied.",
    "2018-2023 development data only.",
    "2024 reserved validation was NOT loaded.",
    "2025 final lockbox was NOT loaded.",
    "",
    "WHY NEGATIVE R2 MATTERS",
    "-" * 112,
    "R² = 1 - SSE/TSS.",
    "A negative R² occurs when model squared error (SSE) exceeds the",
    "within-year total variation around that year's observed mean (TSS).",
    "",
    "PROBLEM-YEAR MODEL EVIDENCE",
    "-" * 112,
]

for _, row in problem_year_evidence.iterrows():
    report_lines.append(
        f"{int(row['validation_year'])} | {row['model']}: "
        f"raw R²={row['raw_r2']:.6f}, "
        f"SSE/TSS={row['sse_over_tss']:.6f}, "
        f"bias={row['prediction_bias_pred_minus_actual_c']:+.6f} °C, "
        f"bias MSE share={row['bias_squared_share_of_mse']:.6f}, "
        f"centered spatial R²={row['centered_spatial_r2']:.6f}, "
        f"residual Moran I={row['residual_moran_i']:.6f}, "
        f"p={row['residual_moran_p']:.6f}."
    )

report_lines.extend(
    [
        "",
        "YEAR-LEVEL DISTRIBUTION SHIFT",
        "-" * 112,
    ]
)

for _, row in year_shift_summary.iterrows():
    report_lines.append(
        f"{int(row['validation_year'])}: "
        f"mean PSI={row['mean_psi']:.6f}, "
        f"max PSI={row['max_psi']:.6f}, "
        f"features PSI>=0.25={int(row['features_psi_ge_0_25'])}, "
        f"features |SMD|>=0.50={int(row['features_abs_smd_ge_0_50'])}."
    )

report_lines.extend(
    [
        "",
        "TOP DIAGNOSTICALLY UNSTABLE FEATURES",
        "-" * 112,
        "These are evidence flags only; they are NOT selected or removed in 19A.",
    ]
)

top_unstable = (
    feature_stability[
        ~feature_stability[
            "is_year_feature"
        ]
    ]
    .head(20)
)

for _, row in top_unstable.iterrows():
    report_lines.append(
        f"{row['feature']}: "
        f"signals={int(row['diagnostic_instability_signal_count'])}, "
        f"rho range={row['rho_range']:.6f}, "
        f"sign flips={int(row['meaningful_sign_flip_count'])}, "
        f"2021 rho jump={row['rho_jump_2021_vs_prior_median']:.6f}, "
        f"2023 rho jump={row['rho_jump_2023_vs_prior_median']:.6f}, "
        f"mean PSI={row['mean_psi']:.6f}, "
        f"max PSI={row['max_psi']:.6f}."
    )

report_lines.extend(
    [
        "",
        "INTERPRETATION POLICY",
        "-" * 112,
        "Step 19A does not automatically claim one cause.",
        "The outputs separate annual-level bias, within-year spatial-pattern error,",
        "feature distribution shift, changing feature-LST relationships,",
        "residual-feature structure, and residual spatial clustering.",
        "",
        "NEXT",
        "-" * 112,
        "Use this diagnosis to predefine justified Step 19B experiments.",
        "Do not use 2024 or 2025 to design those experiments.",
    ]
)

REPORT_PATH.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)

print(
    f"Saved: {REPORT_PATH}"
)
print(
    f"Saved: {AUDIT_PATH}"
)


# =============================================================================
# 19. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "15. REPRODUCIBILITY HASHES"
)

hash_files = [
    INPUT_PATH,
    STEP17_PREDICTIONS_PATH,
    STEP17_FOLD_METRICS_PATH,
    STEP17_AUDIT_PATH,
    STEP18_AUDIT_PATH,
    GEOMETRY_PATH,
    YEAR_LEVEL_DIAG_PATH,
    TARGET_YEAR_STATS_PATH,
    RESIDUAL_SUMMARY_PATH,
    RESIDUAL_FEATURE_ASSOC_PATH,
    RESIDUAL_MORAN_PATH,
    FEATURE_SHIFT_PATH,
    FEATURE_SHIFT_SUMMARY_PATH,
    YEAR_SHIFT_SUMMARY_PATH,
    FEATURE_YEAR_ASSOC_PATH,
    FEATURE_STABILITY_PATH,
    PROBLEM_YEAR_EVIDENCE_PATH,
    STEP17_REPRO_CHECK_PATH,
    AUDIT_PATH,
    REPORT_PATH,
]

hash_rows = []

for path in hash_files:
    digest = sha256_file(
        path
    )

    hash_rows.append(
        {
            "file": str(
                path
            ),
            "sha256": digest,
        }
    )

    print(
        f"{path.name}: {digest}"
    )

pd.DataFrame(
    hash_rows
).to_csv(
    HASH_PATH,
    index=False,
)

print(
    f"Saved: {HASH_PATH}"
)


# =============================================================================
# 20. FINAL STATUS
# =============================================================================

section(
    "STEP 19A COMPLETED SUCCESSFULLY"
)

print(
    "Temporal diagnosis completed without applying any improvement.\n"
    "Step 17 saved tuned rolling predictions were reproduced and verified.\n"
    "Year-level bias and raw-vs-centered R² were diagnosed.\n"
    "Feature distribution shift was measured under strict rolling-origin history.\n"
    "Feature-LST relationship stability was measured across 2018-2023.\n"
    "Residual distributions, residual-feature associations, and residual Moran's I were saved.\n"
    "Focused evidence for 2021 and 2023 was saved.\n"
    "2024 reserved validation remained untouched.\n"
    "2025 final lockbox remained untouched.\n"
    "No feature subset, weighting, bias correction, ensemble, or final model was selected.\n"
    "Review the complete Step 19A output before deciding which Step 19B experiments are justified."
)
