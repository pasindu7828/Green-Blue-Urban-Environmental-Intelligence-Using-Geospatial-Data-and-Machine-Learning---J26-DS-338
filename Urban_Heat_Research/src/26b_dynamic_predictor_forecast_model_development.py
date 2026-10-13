"""
STEP 26B - DYNAMIC PREDICTOR FORECAST MODEL DEVELOPMENT
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Develop and freeze candidate methods for forecasting the dynamic base
predictors needed later to build future 2026-2030 feature rows for the frozen
Step-22B LightGBM spatial-anomaly model.

Future pipeline:
    forecast dynamic base predictors
        -> reconstruct future lag/history/delta/interaction/neighbour features
        -> frozen Step-22B LightGBM
        -> future centered spatial anomaly
        + annual thermal-level forecast
        -> final future grid LST

IMPORTANT
---------
Step 26B does NOT forecast final future LST.
Step 26B does NOT retrain the Step-22B LightGBM.
Step 26B does NOT independently forecast engineered features such as lag2,
delta1, hist3_mean or neighbourhood means. Those are reconstructed later from
the forecast base-predictor history using the original feature-engineering
logic.

DEVELOPMENT / VALIDATION PROTOCOL
---------------------------------
Observed predictor data: 2015-2025
Candidate-development target years: <= 2023
Reserved Step-27 validation target years: 2024 and 2025

Forecast horizons:
    H1, H2, H3, H4, H5 years

Candidate methods for each dynamic continuous/fractional predictor:
1. Persistence
2. Rolling-3 Mean
3. Bounded OLS Trend
4. Bounded Theil-Sen Trend

The trend forecasts are clipped only to the min/max range OBSERVED IN THE
TRAINING HISTORY AVAILABLE AT THAT ORIGIN. This is conservative and does not
use future/reserved information.

If a candidate predictor is genuinely binary in the development data, forecasts
from Rolling-3 / trend candidates are thresholded at 0.5 after clipping.

METRICS
-------
Predictor variables can contain zero or negative values, therefore MAPE is NOT
used here. We save:
- MAE
- RMSE
- R²
- Bias
- normalized MAE = MAE / SD(actual)
- normalized RMSE = RMSE / SD(actual)

The normalized metrics allow scale-aware comparisons across different
predictors.

DYNAMIC-PREDICTOR AUDIT
-----------------------
The script audits likely dynamic base predictors:
    NDVI, NDBI, NDWI, EVI, Albedo,
    NDBI_ADJ, NDVI_ADJ, NightLights,
    green_mask, building_mask, road_mask

A predictor is treated as dynamic only if it shows substantive temporal change
in the 2015-2023 development period. Completely/nearly static predictors are
flagged to be frozen at their latest observed value later.

Structural context such as dist_water, dist_road, dist_main_road, LCZ and
area_name is NOT forecast here.

No final candidate method is selected in Step 26B.
All candidate definitions proceed unchanged to Step 27.
"""

from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"

EXPECTED_ROWS = 24134
EXPECTED_GRIDS = 2194
EXPECTED_YEARS = list(range(2015, 2026))

DEV_END_YEAR = 2023
RESERVED_VALIDATION_YEARS = [2024, 2025]
MIN_ORIGIN_YEAR = 2018
HORIZONS = [1, 2, 3, 4, 5]

TEMPORAL_CHANGE_TOL = 1e-8
MIN_CHANGED_TRANSITION_FRACTION = 0.001  # 0.1%

PREDICTOR_POOL = [
    "NDVI",
    "NDBI",
    "NDWI",
    "EVI",
    "Albedo",
    "NDBI_ADJ",
    "NDVI_ADJ",
    "NightLights",
    "green_mask",
    "building_mask",
    "road_mask",
]

STRUCTURAL_FROZEN = [
    "dist_road",
    "dist_main_road",
    "dist_water",
    "LCZ",
    "area_name",
]

CANDIDATES = [
    "Persistence",
    "Rolling-3 Mean",
    "Bounded OLS Trend",
    "Bounded Theil-Sen Trend",
]

EXPECTED_HASHES = {
    "kaduwela_preprocessed_base.csv":
        "0e030b2ec0b5c4080cd4226bec798189a178aa45efd9962d19f38dc9cb170f46",
    "21_final_feature_list.csv":
        "6769a195b5027850ebd458a4bef7c9da0ecc43c4969bcc81f746e46bce3be9c0",
    "25_forecasting_baseline_audit.csv":
        "aef89269cb4accd889947a5daa117bf74c4116646a95d3881187959f06084df3",
}


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_DATA_PATH = (
    ROOT / "data" / "processed" / "kaduwela_preprocessed_base.csv"
)

FEATURE_LIST_PATH = (
    ROOT / "outputs" / "reports" / "21_final_feature_list.csv"
)

STEP25_AUDIT_PATH = (
    ROOT / "outputs" / "reports" / "25_forecasting_baseline_audit.csv"
)

FORECAST_DIR = ROOT / "outputs" / "forecasting"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [FORECAST_DIR, FIGURE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TEMPORAL_AUDIT_PATH = (
    FORECAST_DIR / "26b_predictor_temporal_audit.csv"
)

FOLD_METRICS_PATH = (
    FORECAST_DIR / "26b_candidate_fold_metrics_dev.csv"
)

HORIZON_METRICS_PATH = (
    FORECAST_DIR / "26b_candidate_metrics_by_predictor_horizon_dev.csv"
)

SUMMARY_PATH = (
    FORECAST_DIR / "26b_candidate_equal_horizon_summary_dev.csv"
)

LEADERS_PATH = (
    FORECAST_DIR / "26b_development_leaders_reference_only.csv"
)

METHOD_OVERALL_PATH = (
    FORECAST_DIR / "26b_candidate_overall_normalized_summary_dev.csv"
)

FROZEN_STRUCTURAL_PATH = (
    FORECAST_DIR / "26b_structural_predictor_policy.csv"
)

MAE_FIGURE_PATH = (
    FIGURE_DIR / "26b_predictor_candidate_normalized_mae.png"
)

DYNAMIC_FIGURE_PATH = (
    FIGURE_DIR / "26b_predictor_temporal_change_audit.png"
)

FREEZE_PATH = (
    REPORT_DIR / "26b_predictor_candidate_freeze.json"
)

METHOD_PATH = (
    REPORT_DIR / "26b_predictor_forecasting_methodology.json"
)

AUDIT_PATH = (
    REPORT_DIR / "26b_predictor_forecasting_audit.csv"
)

REPORT_PATH = (
    REPORT_DIR / "26b_predictor_forecasting_report.txt"
)

HASH_PATH = (
    REPORT_DIR / "26b_predictor_forecasting_hashes.csv"
)


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


def verify_hash(path, expected, label):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual  : {actual}"
        )

    print(f"PASS hash: {label} = {actual}")
    return actual


def require_all_pass(path, label):
    audit = pd.read_csv(path)

    if "status" not in audit.columns:
        raise RuntimeError(f"{label} audit has no status column.")

    bad = audit[
        audit["status"].astype(str).str.upper() != "PASS"
    ]

    if not bad.empty:
        print(bad.to_string(index=False))
        raise RuntimeError(f"{label} audit is not fully PASS.")


def detect_feature_column(feature_df, available_columns):
    candidates = []

    for column in feature_df.columns:
        values = feature_df[column].dropna().astype(str).tolist()

        if (
            len(values) == 93
            and len(set(values)) == 93
            and all(value in available_columns for value in values)
        ):
            candidates.append(column)

    if len(candidates) == 1:
        return candidates[0]

    # The feature-list file can be read without the model-ready feature matrix
    # here. Fall back safely to a conventional column name if present.
    for conventional in ["feature", "feature_name", "predictor"]:
        if conventional in feature_df.columns:
            return conventional

    raise RuntimeError(
        "Could not identify the feature-name column in the frozen feature list."
    )


def base_predictor_referenced(base_name, features):
    base_lower = base_name.lower()

    for feature in features:
        f = str(feature).lower()

        if (
            f == base_lower
            or f.startswith(base_lower + "_")
            or ("_" + base_lower + "_") in f
            or f.endswith("_" + base_lower)
            or (base_lower + "_x_") in f
            or ("_x_" + base_lower) in f
        ):
            return True

    return False


def is_binary_like(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return False

    unique = np.unique(np.round(values, 12))

    return bool(
        len(unique) <= 2
        and np.all(
            np.isclose(unique[:, None], np.array([0.0, 1.0]), atol=1e-8)
            .any(axis=1)
        )
    )


def predictor_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError("Actual/predicted lengths differ.")

    error = predicted - actual

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))

    actual_sd = float(np.std(actual, ddof=0))

    if actual_sd > 0:
        nmae = float(mae / actual_sd)
        nrmse = float(rmse / actual_sd)
    else:
        nmae = np.nan
        nrmse = np.nan

    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))

    r2 = (
        float(1.0 - ss_res / ss_tot)
        if ss_tot > 0
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Bias": bias,
        "Actual_SD": actual_sd,
        "NMAE_SD": nmae,
        "NRMSE_SD": nrmse,
    }


def ols_forecast_matrix(history_values, history_years, target_year):
    """
    history_values: [n_grids, n_years]
    """
    x = np.asarray(history_years, dtype=float)
    y = np.asarray(history_values, dtype=float)

    x_mean = float(np.mean(x))
    x_centered = x - x_mean
    denominator = float(np.sum(x_centered ** 2))

    if denominator == 0:
        return y[:, -1].copy()

    y_mean = np.mean(y, axis=1)

    slopes = (
        np.sum(
            (y - y_mean[:, None]) * x_centered[None, :],
            axis=1,
        )
        / denominator
    )

    prediction = (
        y_mean
        + slopes * (float(target_year) - x_mean)
    )

    return prediction


def theil_sen_forecast_matrix(history_values, history_years, target_year):
    """
    Vectorized Theil-Sen using median pairwise slopes per grid.
    Intercept uses median(y) - slope*median(x), matching a robust
    separate-intercept construction.
    """
    x = np.asarray(history_years, dtype=float)
    y = np.asarray(history_values, dtype=float)

    pair_slopes = []

    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            dx = x[j] - x[i]

            pair_slopes.append(
                (y[:, j] - y[:, i]) / dx
            )

    slopes = np.median(
        np.stack(pair_slopes, axis=1),
        axis=1,
    )

    intercepts = (
        np.median(y, axis=1)
        - slopes * np.median(x)
    )

    return (
        intercepts
        + slopes * float(target_year)
    )


def constrain_prediction(
    prediction,
    history_values,
    binary_like,
):
    """
    Conservative constraint using only values observed in training history.
    No reserved/future values are used.
    """
    pred = np.asarray(prediction, dtype=float).copy()

    history_min = float(np.nanmin(history_values))
    history_max = float(np.nanmax(history_values))

    pred = np.clip(
        pred,
        history_min,
        history_max,
    )

    if binary_like:
        pred = (
            pred >= 0.5
        ).astype(float)

    return pred


def make_prediction(
    method,
    history_values,
    history_years,
    target_year,
    binary_like,
):
    if method == "Persistence":
        prediction = history_values[:, -1].copy()

    elif method == "Rolling-3 Mean":
        n_recent = min(3, history_values.shape[1])

        prediction = np.mean(
            history_values[:, -n_recent:],
            axis=1,
        )

    elif method == "Bounded OLS Trend":
        prediction = ols_forecast_matrix(
            history_values,
            history_years,
            target_year,
        )

    elif method == "Bounded Theil-Sen Trend":
        prediction = theil_sen_forecast_matrix(
            history_values,
            history_years,
            target_year,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return constrain_prediction(
        prediction,
        history_values,
        binary_like,
    )


def aggregate_by_horizon(fold_metrics):
    rows = []

    group_cols = [
        "predictor",
        "method",
        "horizon_years",
    ]

    for keys, part in fold_metrics.groupby(group_cols):
        predictor, method, horizon = keys

        # Fold-level metrics are averaged equally so each historical target year
        # has equal influence.
        rows.append(
            {
                "predictor": predictor,
                "method": method,
                "horizon_years": int(horizon),
                "fold_count": int(len(part)),
                "mean_fold_MAE": float(part["MAE"].mean()),
                "mean_fold_RMSE": float(part["RMSE"].mean()),
                "mean_fold_R2": float(part["R2"].mean()),
                "mean_abs_fold_Bias": float(part["Bias"].abs().mean()),
                "mean_fold_NMAE_SD": float(part["NMAE_SD"].mean()),
                "mean_fold_NRMSE_SD": float(part["NRMSE_SD"].mean()),
            }
        )

    return pd.DataFrame(rows)


def equal_horizon_summary(horizon_metrics):
    rows = []

    for (predictor, method), part in horizon_metrics.groupby(
        ["predictor", "method"]
    ):
        rows.append(
            {
                "predictor": predictor,
                "method": method,
                "horizon_count": int(part["horizon_years"].nunique()),
                "mean_horizon_MAE": float(part["mean_fold_MAE"].mean()),
                "mean_horizon_RMSE": float(part["mean_fold_RMSE"].mean()),
                "mean_horizon_R2": float(part["mean_fold_R2"].mean()),
                "mean_horizon_NMAE_SD": float(
                    part["mean_fold_NMAE_SD"].mean()
                ),
                "mean_horizon_NRMSE_SD": float(
                    part["mean_fold_NRMSE_SD"].mean()
                ),
                "mean_abs_horizon_Bias": float(
                    part["mean_abs_fold_Bias"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 0. VERIFY INPUTS
# =============================================================================

section(
    "STEP 26B - DYNAMIC PREDICTOR FORECAST MODEL DEVELOPMENT"
)

verify_hash(
    BASE_DATA_PATH,
    EXPECTED_HASHES["kaduwela_preprocessed_base.csv"],
    "Canonical preprocessed base data",
)

verify_hash(
    FEATURE_LIST_PATH,
    EXPECTED_HASHES["21_final_feature_list.csv"],
    "Frozen Step-21 feature list",
)

verify_hash(
    STEP25_AUDIT_PATH,
    EXPECTED_HASHES["25_forecasting_baseline_audit.csv"],
    "Step-25 strict audit",
)

require_all_pass(
    STEP25_AUDIT_PATH,
    "Step 25",
)

print("\nPASS: Step-25 audit is fully PASS.")
print("\nDevelopment protocol:")
print("- dynamic base predictors only")
print("- candidate targets stop at 2023")
print("- 2024 and 2025 reserved for Step 27")
print("- H1-H5 grid-level predictor hindcasting")
print("- no final 2026-2030 predictor forecast")
print("- no future LST forecast")
print("- Step-22B LightGBM remains frozen")


# =============================================================================
# 1. LOAD + VALIDATE BASE DATA
# =============================================================================

section(
    "1. LOAD + VALIDATE 2015-2025 BASE PREDICTOR DATA"
)

data = pd.read_csv(
    BASE_DATA_PATH,
    dtype={GRID_COL: str},
)

if len(data) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Rows={len(data)}, expected={EXPECTED_ROWS}"
    )

if data[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Unique grids={data[GRID_COL].nunique()}, expected={EXPECTED_GRIDS}"
    )

data[YEAR_COL] = pd.to_numeric(
    data[YEAR_COL],
    errors="raise",
).astype(int)

years = sorted(
    data[YEAR_COL].unique().tolist()
)

if years != EXPECTED_YEARS:
    raise RuntimeError(
        f"Years={years}, expected={EXPECTED_YEARS}"
    )

if data.duplicated([GRID_COL, YEAR_COL]).any():
    raise RuntimeError(
        "Duplicate grid-year rows found."
    )

rows_per_year = (
    data.groupby(YEAR_COL)[GRID_COL]
    .nunique()
)

if not (
    rows_per_year == EXPECTED_GRIDS
).all():
    raise RuntimeError(
        "Every observed year must contain all 2,194 grids."
    )

missing_pool = [
    p for p in PREDICTOR_POOL
    if p not in data.columns
]

if missing_pool:
    raise RuntimeError(
        "Expected base predictor columns are missing: "
        f"{missing_pool}"
    )

for predictor in PREDICTOR_POOL:
    data[predictor] = pd.to_numeric(
        data[predictor],
        errors="raise",
    )

    if not np.isfinite(
        data[predictor].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            f"{predictor} contains missing/non-finite values."
        )

print(f"Rows         : {len(data):,}")
print(f"Unique grids : {data[GRID_COL].nunique():,}")
print(f"Years        : {years}")
print(f"Candidate base predictors: {len(PREDICTOR_POOL)}")


# =============================================================================
# 2. LOAD FROZEN FEATURE LIST + CHECK RELEVANCE
# =============================================================================

section(
    "2. CHECK BASE-PREDICTOR RELEVANCE TO FROZEN LIGHTGBM FEATURES"
)

feature_df = pd.read_csv(
    FEATURE_LIST_PATH
)

feature_col = detect_feature_column(
    feature_df,
    set(feature_df.columns) | set(data.columns),
)

features = (
    feature_df[feature_col]
    .dropna()
    .astype(str)
    .tolist()
)

if len(features) != 93:
    raise RuntimeError(
        f"Frozen feature count={len(features)}, expected=93"
    )

relevance_rows = []

for predictor in PREDICTOR_POOL:
    relevance_rows.append(
        {
            "predictor": predictor,
            "referenced_by_frozen_feature_names": base_predictor_referenced(
                predictor,
                features,
            ),
        }
    )

relevance = pd.DataFrame(
    relevance_rows
)

print(
    relevance.to_string(
        index=False
    )
)

print(
    "\nNOTE: Derived features will be reconstructed later using the original "
    "feature-engineering logic; they are not forecast independently."
)


# =============================================================================
# 3. TEMPORAL VARIATION AUDIT
# =============================================================================

section(
    "3. TEMPORAL VARIATION AUDIT - IDENTIFY DYNAMIC BASE PREDICTORS"
)

dev_data = data[
    data[YEAR_COL] <= DEV_END_YEAR
].copy()

temporal_rows = []

wide_by_predictor = {}

for predictor in PREDICTOR_POOL:
    wide = (
        dev_data.pivot(
            index=GRID_COL,
            columns=YEAR_COL,
            values=predictor,
        )
        .sort_index()
        .sort_index(axis=1)
    )

    if wide.shape[0] != EXPECTED_GRIDS:
        raise RuntimeError(
            f"{predictor}: grid count mismatch in development matrix."
        )

    if wide.isna().any().any():
        raise RuntimeError(
            f"{predictor}: development matrix contains missing values."
        )

    wide_by_predictor[predictor] = wide

    diff = np.diff(
        wide.to_numpy(dtype=float),
        axis=1,
    )

    changed = (
        np.abs(diff) > TEMPORAL_CHANGE_TOL
    )

    changed_fraction = float(
        np.mean(changed)
    )

    mean_abs_change = float(
        np.mean(
            np.abs(diff)
        )
    )

    values = wide.to_numpy(dtype=float)

    binary = is_binary_like(
        values.ravel()
    )

    dynamic = bool(
        changed_fraction
        >= MIN_CHANGED_TRANSITION_FRACTION
    )

    temporal_rows.append(
        {
            "predictor": predictor,
            "development_min": float(np.min(values)),
            "development_max": float(np.max(values)),
            "development_mean": float(np.mean(values)),
            "development_sd": float(np.std(values)),
            "binary_like": binary,
            "changed_transition_fraction": changed_fraction,
            "changed_transition_pct": changed_fraction * 100.0,
            "mean_absolute_year_to_year_change": mean_abs_change,
            "forecast_policy": (
                "dynamic_forecast"
                if dynamic
                else "freeze_latest_observed"
            ),
            "referenced_by_frozen_feature_names": bool(
                relevance.loc[
                    relevance["predictor"] == predictor,
                    "referenced_by_frozen_feature_names",
                ].iloc[0]
            ),
        }
    )

temporal_audit = pd.DataFrame(
    temporal_rows
)

temporal_audit.to_csv(
    TEMPORAL_AUDIT_PATH,
    index=False,
)

dynamic_predictors = (
    temporal_audit.loc[
        temporal_audit["forecast_policy"] == "dynamic_forecast",
        "predictor",
    ]
    .tolist()
)

static_pool_predictors = (
    temporal_audit.loc[
        temporal_audit["forecast_policy"] == "freeze_latest_observed",
        "predictor",
    ]
    .tolist()
)

if len(dynamic_predictors) == 0:
    raise RuntimeError(
        "No dynamic predictors were identified."
    )

print(
    temporal_audit.to_string(
        index=False
    )
)

print(
    f"\nDynamic predictors selected for model development ({len(dynamic_predictors)}):"
)

for predictor in dynamic_predictors:
    print(f"  - {predictor}")

if static_pool_predictors:
    print(
        "\nCandidate-pool predictors treated as static/frozen:"
    )
    for predictor in static_pool_predictors:
        print(f"  - {predictor}")


# =============================================================================
# 4. STRUCTURAL / FROZEN PREDICTOR POLICY
# =============================================================================

section(
    "4. STRUCTURAL / FROZEN PREDICTOR POLICY"
)

structural_rows = []

for predictor in STRUCTURAL_FROZEN:
    if predictor in data.columns:
        structural_rows.append(
            {
                "predictor": predictor,
                "policy": "freeze_latest_observed_or_keep_structural",
                "reason": (
                    "structural/location context; not forecast as a dynamic "
                    "annual environmental variable in Step 26B"
                ),
            }
        )

for predictor in static_pool_predictors:
    structural_rows.append(
        {
            "predictor": predictor,
            "policy": "freeze_latest_observed",
            "reason": (
                "no substantive temporal variation detected in 2015-2023 "
                "development history"
            ),
        }
    )

structural_policy = pd.DataFrame(
    structural_rows
).drop_duplicates(
    subset=["predictor"],
    keep="first",
)

structural_policy.to_csv(
    FROZEN_STRUCTURAL_PATH,
    index=False,
)

print(
    structural_policy.to_string(
        index=False
    )
)


# =============================================================================
# 5. GRID-LEVEL ROLLING-ORIGIN DEVELOPMENT
# =============================================================================

section(
    "5. GRID-LEVEL ROLLING-ORIGIN DEVELOPMENT HINDCASTS"
)

fold_metric_rows = []

expected_fold_counts = {
    1: 5,
    2: 4,
    3: 3,
    4: 2,
    5: 1,
}

for predictor_number, predictor in enumerate(
    dynamic_predictors,
    start=1,
):
    print(
        f"\nPredictor {predictor_number}/{len(dynamic_predictors)}: {predictor}"
    )

    full_wide = (
        data.pivot(
            index=GRID_COL,
            columns=YEAR_COL,
            values=predictor,
        )
        .sort_index()
        .sort_index(axis=1)
    )

    binary = bool(
        temporal_audit.loc[
            temporal_audit["predictor"] == predictor,
            "binary_like",
        ].iloc[0]
    )

    for horizon in HORIZONS:
        fold_counter = 0

        for origin_year in range(
            MIN_ORIGIN_YEAR,
            DEV_END_YEAR,
        ):
            target_year = origin_year + horizon

            if target_year > DEV_END_YEAR:
                continue

            history_years = [
                year
                for year in EXPECTED_YEARS
                if year <= origin_year
            ]

            history_values = (
                full_wide[history_years]
                .to_numpy(dtype=float)
            )

            actual = (
                full_wide[target_year]
                .to_numpy(dtype=float)
            )

            if history_values.shape[1] < 4:
                raise RuntimeError(
                    f"{predictor}: fewer than four history years at origin {origin_year}."
                )

            for method in CANDIDATES:
                predicted = make_prediction(
                    method=method,
                    history_values=history_values,
                    history_years=history_years,
                    target_year=target_year,
                    binary_like=binary,
                )

                metrics = predictor_metrics(
                    actual,
                    predicted,
                )

                history_min = float(
                    np.min(history_values)
                )

                history_max = float(
                    np.max(history_values)
                )

                # Because constrain_prediction is applied, violations should be zero.
                below = int(
                    np.sum(predicted < history_min - 1e-12)
                )

                above = int(
                    np.sum(predicted > history_max + 1e-12)
                )

                fold_metric_rows.append(
                    {
                        "predictor": predictor,
                        "method": method,
                        "horizon_years": horizon,
                        "origin_year": origin_year,
                        "target_year": target_year,
                        "grid_count": EXPECTED_GRIDS,
                        "training_start_year": min(history_years),
                        "training_end_year": max(history_years),
                        "training_year_count": len(history_years),
                        "binary_like": binary,
                        "training_history_min": history_min,
                        "training_history_max": history_max,
                        "range_violation_count": below + above,
                        **metrics,
                    }
                )

            fold_counter += 1

        print(
            f"  H{horizon}: {fold_counter} development folds"
        )

        if fold_counter != expected_fold_counts[horizon]:
            raise RuntimeError(
                f"{predictor} H{horizon}: fold_count={fold_counter}, "
                f"expected={expected_fold_counts[horizon]}"
            )

fold_metrics = pd.DataFrame(
    fold_metric_rows
)

fold_metrics.to_csv(
    FOLD_METRICS_PATH,
    index=False,
)

print(
    f"\nSaved: {FOLD_METRICS_PATH}"
)


# =============================================================================
# 6. AGGREGATED DEVELOPMENT METRICS
# =============================================================================

section(
    "6. AGGREGATED DEVELOPMENT METRICS"
)

horizon_metrics = aggregate_by_horizon(
    fold_metrics
)

horizon_metrics.to_csv(
    HORIZON_METRICS_PATH,
    index=False,
)

summary = equal_horizon_summary(
    horizon_metrics
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

print(
    "Equal-horizon summary by predictor/method:"
)

print(
    summary.sort_values(
        [
            "predictor",
            "mean_horizon_NMAE_SD",
            "mean_horizon_MAE",
        ]
    ).to_string(
        index=False
    )
)


# =============================================================================
# 7. DEVELOPMENT LEADERS - REFERENCE ONLY
# =============================================================================

section(
    "7. DEVELOPMENT LEADERS - REFERENCE ONLY"
)

leader_rows = []

for predictor in dynamic_predictors:
    for horizon in HORIZONS:
        part = (
            horizon_metrics[
                (
                    horizon_metrics["predictor"] == predictor
                )
                & (
                    horizon_metrics["horizon_years"] == horizon
                )
            ]
            .sort_values(
                [
                    "mean_fold_NMAE_SD",
                    "mean_fold_MAE",
                    "mean_fold_RMSE",
                ]
            )
            .reset_index(drop=True)
        )

        if part.empty:
            continue

        row = part.iloc[0].to_dict()
        row["status"] = "development_reference_only"
        leader_rows.append(row)

leaders = pd.DataFrame(
    leader_rows
)

leaders.to_csv(
    LEADERS_PATH,
    index=False,
)

print(
    leaders[
        [
            "predictor",
            "horizon_years",
            "method",
            "mean_fold_MAE",
            "mean_fold_RMSE",
            "mean_fold_R2",
            "mean_fold_NMAE_SD",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nThese are NOT final method selections. "
    "All candidate methods remain frozen for Step 27."
)


# =============================================================================
# 8. OVERALL NORMALIZED CANDIDATE REFERENCE
# =============================================================================

section(
    "8. OVERALL NORMALIZED CANDIDATE REFERENCE"
)

method_overall = (
    summary.groupby(
        "method",
        as_index=False,
    )
    .agg(
        predictor_count=(
            "predictor",
            "nunique",
        ),
        mean_predictor_NMAE_SD=(
            "mean_horizon_NMAE_SD",
            "mean",
        ),
        mean_predictor_NRMSE_SD=(
            "mean_horizon_NRMSE_SD",
            "mean",
        ),
        mean_predictor_R2=(
            "mean_horizon_R2",
            "mean",
        ),
    )
    .sort_values(
        [
            "mean_predictor_NMAE_SD",
            "mean_predictor_NRMSE_SD",
        ]
    )
    .reset_index(drop=True)
)

method_overall.to_csv(
    METHOD_OVERALL_PATH,
    index=False,
)

print(
    method_overall.to_string(
        index=False
    )
)

print(
    "\nOverall normalized summary is descriptive only; predictor-specific "
    "performance remains the main evidence."
)


# =============================================================================
# 9. FREEZE CANDIDATE DEFINITIONS FOR STEP 27
# =============================================================================

section(
    "9. FREEZE STEP-26B CANDIDATE DEFINITIONS"
)

freeze = {
    "step": "26B",
    "purpose": "dynamic base-predictor forecast model development",
    "observed_years": EXPECTED_YEARS,
    "development_target_year_max": DEV_END_YEAR,
    "reserved_validation_target_years": RESERVED_VALIDATION_YEARS,
    "horizons_years": HORIZONS,
    "minimum_origin_year": MIN_ORIGIN_YEAR,
    "dynamic_predictors": dynamic_predictors,
    "static_or_frozen_from_candidate_pool": static_pool_predictors,
    "structural_predictors_not_forecast": [
        p for p in STRUCTURAL_FROZEN
        if p in data.columns
    ],
    "candidate_methods": {
        "Persistence": {
            "definition": "last observed grid value at forecast origin",
        },
        "Rolling-3 Mean": {
            "definition": "mean of latest up-to-3 observed grid values",
        },
        "Bounded OLS Trend": {
            "definition": (
                "per-grid OLS trend using all history available at origin; "
                "target-year extrapolation clipped to training-history global min/max"
            ),
        },
        "Bounded Theil-Sen Trend": {
            "definition": (
                "per-grid median pairwise-slope robust trend using all history "
                "available at origin; target-year extrapolation clipped to "
                "training-history global min/max"
            ),
        },
    },
    "binary_rule": (
        "If development values are genuinely binary {0,1}, candidate forecasts "
        "are clipped then thresholded at 0.5."
    ),
    "metric_policy": {
        "MAPE": "not used because predictor values may be zero or negative",
        "primary_scale_aware_metric": "NMAE_SD",
        "saved_metrics": [
            "MAE",
            "RMSE",
            "R2",
            "Bias",
            "NMAE_SD",
            "NRMSE_SD",
        ],
    },
    "engineered_feature_policy": (
        "Do not forecast engineered lag/delta/history/interaction/neighbour "
        "features independently. Reconstruct them later from forecast base "
        "predictors using the original feature-engineering logic."
    ),
    "selection_policy": (
        "No final predictor forecasting method selected in Step 26B. "
        "All frozen candidate definitions proceed to Step 27."
    ),
    "step22b_model_changed": False,
    "future_2026_2030_generated": False,
}

FREEZE_PATH.write_text(
    json.dumps(
        freeze,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        freeze,
        indent=2,
    )
)


# =============================================================================
# 10. FIGURES
# =============================================================================

section(
    "10. SAVE STEP-26B FIGURES"
)

plot_change = (
    temporal_audit
    .sort_values(
        "changed_transition_pct"
    )
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.barh(
    plot_change["predictor"],
    plot_change["changed_transition_pct"],
)

ax.set_xlabel(
    "Changed year-to-year transitions (%)"
)

ax.set_ylabel(
    "Base predictor"
)

ax.set_title(
    "Temporal Change Audit for Candidate Future Predictors (2015–2023)"
)

ax.grid(
    axis="x",
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    DYNAMIC_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


plot_method = method_overall.sort_values(
    "mean_predictor_NMAE_SD",
    ascending=True,
)

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.barh(
    plot_method["method"],
    plot_method["mean_predictor_NMAE_SD"],
)

ax.set_xlabel(
    "Mean predictor normalized MAE (MAE / actual SD)"
)

ax.set_ylabel(
    "Candidate method"
)

ax.set_title(
    "Step 26B Dynamic-Predictor Forecast Candidate Development"
)

ax.grid(
    axis="x",
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    MAE_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved: {DYNAMIC_FIGURE_PATH}")
print(f"Saved: {MAE_FIGURE_PATH}")


# =============================================================================
# 11. METHODOLOGY
# =============================================================================

section(
    "11. SAVE STEP-26B METHODOLOGY"
)

methodology = {
    "step": "26B",
    "analysis": "Dynamic base-predictor forecast model development",
    "future_pipeline": (
        "future base predictors -> reconstructed engineered features -> "
        "frozen Step-22B LightGBM spatial anomaly -> annual level + anomaly -> future LST"
    ),
    "development_target_year_max": DEV_END_YEAR,
    "reserved_validation_target_years": RESERVED_VALIDATION_YEARS,
    "dynamic_predictor_audit_source_years": list(range(2015, 2024)),
    "dynamic_change_threshold": {
        "absolute_transition_tolerance": TEMPORAL_CHANGE_TOL,
        "minimum_changed_transition_fraction": MIN_CHANGED_TRANSITION_FRACTION,
    },
    "dynamic_predictors": dynamic_predictors,
    "candidate_methods": CANDIDATES,
    "metrics": [
        "MAE",
        "RMSE",
        "R2",
        "Bias",
        "NMAE_SD",
        "NRMSE_SD",
    ],
    "why_no_mape": (
        "Several environmental indices can be zero or negative, making MAPE "
        "unstable or undefined."
    ),
    "constraint_rule": (
        "Candidate forecasts are clipped to the predictor's min/max observed "
        "in training history available at that origin; reserved/future values "
        "are never used for clipping."
    ),
    "binary_rule": (
        "Binary-like predictors are thresholded at 0.5 after clipping."
    ),
    "structural_policy": structural_policy.to_dict(
        orient="records"
    ),
    "engineered_feature_policy": (
        "Future lag/history/delta/interaction/neighbour variables are computed "
        "later from the future base-predictor sequence; they are not separately forecast."
    ),
    "selection_policy": (
        "Development leaders are reference only. All candidate methods are "
        "frozen and evaluated on reserved 2024-2025 targets in Step 27."
    ),
    "future_prediction_policy": (
        "No 2026-2030 predictor maps or future LST predictions are produced in Step 26B."
    ),
}

METHOD_PATH.write_text(
    json.dumps(
        methodology,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        methodology,
        indent=2,
    )
)


# =============================================================================
# 12. STRICT AUDIT
# =============================================================================

section(
    "12. STRICT STEP-26B AUDIT"
)

all_targets_dev_only = bool(
    fold_metrics["target_year"].max()
    <= DEV_END_YEAR
)

reserved_absent = bool(
    not fold_metrics["target_year"]
    .isin(RESERVED_VALIDATION_YEARS)
    .any()
)

training_precedes_target = bool(
    (
        fold_metrics["training_end_year"]
        < fold_metrics["target_year"]
    ).all()
)

origin_matches_cutoff = bool(
    (
        fold_metrics["origin_year"]
        == fold_metrics["training_end_year"]
    ).all()
)

all_range_violations_zero = bool(
    (
        fold_metrics["range_violation_count"] == 0
    ).all()
)

all_predictor_method_horizons = True

for predictor in dynamic_predictors:
    part = horizon_metrics[
        horizon_metrics["predictor"] == predictor
    ]

    if (
        set(part["method"]) != set(CANDIDATES)
        or set(part["horizon_years"]) != set(HORIZONS)
    ):
        all_predictor_method_horizons = False
        break

expected_fold_rows = (
    len(dynamic_predictors)
    * len(CANDIDATES)
    * sum(expected_fold_counts.values())
)

actual_fold_rows = len(
    fold_metrics
)

dynamic_audit_dev_only = bool(
    dev_data[YEAR_COL].max() == DEV_END_YEAR
)

audit_rows = [
    {
        "check": "canonical_base_hash",
        "status": "PASS",
        "details": EXPECTED_HASHES["kaduwela_preprocessed_base.csv"],
    },
    {
        "check": "frozen_feature_list_hash",
        "status": "PASS",
        "details": EXPECTED_HASHES["21_final_feature_list.csv"],
    },
    {
        "check": "step25_audit",
        "status": "PASS",
        "details": "Step-25 strict audit verified fully PASS.",
    },
    {
        "check": "base_rows",
        "status": "PASS" if len(data) == EXPECTED_ROWS else "FAIL",
        "details": f"actual={len(data)}, expected={EXPECTED_ROWS}",
    },
    {
        "check": "grid_count",
        "status": (
            "PASS"
            if data[GRID_COL].nunique() == EXPECTED_GRIDS
            else "FAIL"
        ),
        "details": (
            f"actual={data[GRID_COL].nunique()}, expected={EXPECTED_GRIDS}"
        ),
    },
    {
        "check": "observed_years",
        "status": "PASS" if years == EXPECTED_YEARS else "FAIL",
        "details": str(years),
    },
    {
        "check": "dynamic_audit_uses_development_only",
        "status": "PASS" if dynamic_audit_dev_only else "FAIL",
        "details": "Temporal dynamic/static classification uses only 2015-2023.",
    },
    {
        "check": "at_least_one_dynamic_predictor",
        "status": "PASS" if len(dynamic_predictors) > 0 else "FAIL",
        "details": f"dynamic_predictor_count={len(dynamic_predictors)}",
    },
    {
        "check": "candidate_targets_stop_at_2023",
        "status": "PASS" if all_targets_dev_only else "FAIL",
        "details": f"max_target={fold_metrics['target_year'].max()}",
    },
    {
        "check": "reserved_2024_2025_not_used_as_targets",
        "status": "PASS" if reserved_absent else "FAIL",
        "details": "2024 and 2025 reserved for Step 27.",
    },
    {
        "check": "training_precedes_target",
        "status": "PASS" if training_precedes_target else "FAIL",
        "details": "All candidate histories end before target year.",
    },
    {
        "check": "origin_matches_training_cutoff",
        "status": "PASS" if origin_matches_cutoff else "FAIL",
        "details": "Each origin equals final year in its available history.",
    },
    {
        "check": "all_methods_all_horizons_per_dynamic_predictor",
        "status": (
            "PASS"
            if all_predictor_method_horizons
            else "FAIL"
        ),
        "details": (
            f"{len(CANDIDATES)} candidates across H1-H5 for each dynamic predictor."
        ),
    },
    {
        "check": "expected_fold_metric_rows",
        "status": (
            "PASS"
            if actual_fold_rows == expected_fold_rows
            else "FAIL"
        ),
        "details": (
            f"actual={actual_fold_rows}, expected={expected_fold_rows}"
        ),
    },
    {
        "check": "training_history_range_constraints",
        "status": (
            "PASS"
            if all_range_violations_zero
            else "FAIL"
        ),
        "details": "All bounded predictions remain inside origin-known training range.",
    },
    {
        "check": "mape_not_used_for_dynamic_predictors",
        "status": "PASS",
        "details": (
            "Predictors can be zero/negative; scale-aware NMAE/NRMSE are used instead."
        ),
    },
    {
        "check": "engineered_features_not_independently_forecast",
        "status": "PASS",
        "details": (
            "Lag/delta/history/interaction/neighbour features are deferred to future reconstruction."
        ),
    },
    {
        "check": "candidate_definitions_frozen_for_step27",
        "status": "PASS" if FREEZE_PATH.exists() else "FAIL",
        "details": "Step-26B candidate definitions frozen before reserved validation.",
    },
    {
        "check": "no_2026_2030_forecast",
        "status": "PASS",
        "details": "No operational future predictor values/maps created.",
    },
    {
        "check": "step22b_model_untouched",
        "status": "PASS",
        "details": "Frozen production LightGBM is not loaded/retrained/modified.",
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
        index=False
    )
)

if (
    audit_df["status"] != "PASS"
).any():
    raise RuntimeError(
        "One or more Step-26B audit checks failed."
    )


# =============================================================================
# 13. REPORT
# =============================================================================

section(
    "13. SAVE STEP-26B REPORT"
)

leader_text = leaders[
    [
        "predictor",
        "horizon_years",
        "method",
        "mean_fold_MAE",
        "mean_fold_RMSE",
        "mean_fold_R2",
        "mean_fold_NMAE_SD",
    ]
].to_string(
    index=False
)

report_lines = [
    "STEP 26B - DYNAMIC PREDICTOR FORECAST MODEL DEVELOPMENT",
    "=" * 116,
    "",
    "DYNAMIC PREDICTOR AUDIT",
    "-" * 116,
    temporal_audit.to_string(index=False),
    "",
    "DEVELOPMENT LEADERS BY PREDICTOR / HORIZON - REFERENCE ONLY",
    "-" * 116,
    leader_text,
    "",
    "OVERALL NORMALIZED CANDIDATE SUMMARY - DESCRIPTIVE ONLY",
    "-" * 116,
    method_overall.to_string(index=False),
    "",
    "IMPORTANT",
    "-" * 116,
    (
        "No final predictor forecasting method is selected here. "
        "All candidate definitions proceed unchanged to Step 27."
    ),
    (
        "2024 and 2025 remain reserved validation targets."
    ),
    (
        "Future engineered model features are reconstructed later from the "
        "forecast base-predictor sequence; they are not forecast independently."
    ),
    (
        "No 2026-2030 predictor maps or future LST maps are created in Step 26B."
    ),
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {REPORT_PATH}")


# =============================================================================
# 14. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "14. REPRODUCIBILITY HASHES"
)

hash_files = [
    BASE_DATA_PATH,
    FEATURE_LIST_PATH,
    STEP25_AUDIT_PATH,
    TEMPORAL_AUDIT_PATH,
    FOLD_METRICS_PATH,
    HORIZON_METRICS_PATH,
    SUMMARY_PATH,
    LEADERS_PATH,
    METHOD_OVERALL_PATH,
    FROZEN_STRUCTURAL_PATH,
    DYNAMIC_FIGURE_PATH,
    MAE_FIGURE_PATH,
    FREEZE_PATH,
    METHOD_PATH,
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

    print(
        f"{path.name}: {digest}"
    )

pd.DataFrame(
    hash_rows
).to_csv(
    HASH_PATH,
    index=False,
)

print(f"Saved: {HASH_PATH}")


# =============================================================================
# FINAL
# =============================================================================

section(
    "STEP 26B COMPLETED SUCCESSFULLY"
)

print(
    "Dynamic base-predictor forecast model development completed.\n"
    f"Dynamic predictors identified: {len(dynamic_predictors)}.\n"
    "Candidate-development targets were restricted to years <= 2023.\n"
    "2024 and 2025 remain reserved for Step-27 validation.\n"
    "Persistence, Rolling-3, bounded OLS and bounded Theil-Sen definitions were frozen.\n"
    "Development leaders are reference only; no final method was selected.\n"
    "Static/structural predictors were assigned a freeze/keep policy.\n"
    "Engineered future features were not forecast independently.\n"
    "No 2026-2030 predictor maps or future LST forecast was generated.\n"
    "The Step-22B production LightGBM was not changed.\n"
    "\nNEXT: review Step-26B outputs before starting Step 27."
)
