"""
STEP 27 - RESERVED FORECAST VALIDATION AND FORECAST-STRATEGY FREEZE
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Validate the two forecasting branches required for future 2026-2030 heat
prediction using the RESERVED 2024-2025 observed years:

A. Kaduwela annual thermal-level forecasting
B. Grid-level dynamic base-predictor forecasting

Then freeze the forecasting strategies that will be used in Step 28.

IMPORTANT
---------
This is forecast-branch validation, not a new test of the Step-22B LST model.
The Step-22B production LightGBM remains frozen and untouched.

The reserved forecast-validation years are:
    2024 and 2025

The candidate definitions were frozen earlier:
    Step 25  - annual-level baselines
    Step 26A - annual-level candidate models
    Step 26B - dynamic-predictor candidate models

No candidate definition or hyperparameter is changed in this script.

ROLLING-ORIGIN RESERVED VALIDATION
----------------------------------
For each target year and horizon, only history available at that forecast
origin is used.

Examples:
    target 2024, H1 -> origin 2023
    target 2024, H5 -> origin 2019
    target 2025, H1 -> origin 2024
    target 2025, H2 -> origin 2023
    target 2025, H5 -> origin 2020

Using observed 2024 history for the 2025 H1 forecast is valid rolling-origin
forecasting because 2024 is known at that 2025 forecast origin.

FINAL SELECTION POLICY
----------------------
Annual thermal level:
    Select ONE coherent annual-level method across H1-H5 using reserved
    validation evidence. Primary criterion = mean horizon MAE.
    Tie-break = mean horizon RMSE, then mean absolute bias.

Dynamic predictors:
    Select ONE coherent method PER PREDICTOR across H1-H5.
    Primary criterion = mean horizon NMAE_SD.
    Tie-break = mean horizon NRMSE_SD, then mean absolute bias.

Using one method across all horizons for a predictor gives a coherent
2026-2030 forecast path and avoids overfitting a separate method to only two
reserved target years per horizon.

No 2026-2030 operational forecast is generated in Step 27.
That is Step 28.
"""

from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
except ImportError as exc:
    raise ImportError(
        "\nstatsmodels is required for Step 27.\n"
        "Install it with:\n"
        "    pip install statsmodels\n"
    ) from exc

try:
    from scipy.stats import theilslopes
except ImportError as exc:
    raise ImportError(
        "\nscipy is required for Step 27.\n"
        "Install it with:\n"
        "    pip install scipy\n"
    ) from exc


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
ANNUAL_VALUE_COL = "annual_mean_lst_c"

EXPECTED_ROWS = 24134
EXPECTED_GRIDS = 2194
EXPECTED_YEARS = list(range(2015, 2026))

RESERVED_TARGET_YEARS = [2024, 2025]
HORIZONS = [1, 2, 3, 4, 5]

ANNUAL_BASELINE_METHODS = [
    "Persistence",
    "Expanding Mean",
    "Rolling-3 Mean",
    "Drift",
    "OLS Linear Trend",
]

ANNUAL_CANDIDATE_METHODS = [
    "Simple Exponential Smoothing",
    "Damped Holt Trend",
    "Theil-Sen Robust Trend",
]

PREDICTOR_CANDIDATE_METHODS = [
    "Persistence",
    "Rolling-3 Mean",
    "Bounded OLS Trend",
    "Bounded Theil-Sen Trend",
]

EXPECTED_HASHES = {
    "kaduwela_preprocessed_base.csv":
        "0e030b2ec0b5c4080cd4226bec798189a178aa45efd9962d19f38dc9cb170f46",
    "25_annual_kaduwela_mean_lst_2015_2025.csv":
        "910fa5268bedd3fe50ff4abc0db82f48a345d1fe6cf79ffe1849be24cc004869",
    "25_annual_level_baseline_rolling_origin_predictions.csv":
        "319bf955d3c555e7b1c24111746c8c10de89b55e218241dcb22c1d7caa853007",
    "26a_candidate_model_freeze.json":
        "c725abd13b67dc021deb5c6bf8f78273129439392cbf76aa243bdb63289df5b9",
    "26a_annual_level_audit.csv":
        "1c0f8d4eb0b63c36034d500f0f741633c9136fa4c54959cac6a2109024ea4fdb",
    "26b_predictor_candidate_freeze.json":
        "91b23fe3b10bf6f3d66252c7cf16304b8cfa9108865df444ad471767de157363",
    "26b_predictor_forecasting_audit.csv":
        "b7acccce6430b17123db452c510b279d3e577101d4233fca08f37a463ed4bf22",
}


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_DATA_PATH = (
    ROOT / "data" / "processed" / "kaduwela_preprocessed_base.csv"
)

FORECAST_DIR = ROOT / "outputs" / "forecasting"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [FORECAST_DIR, FIGURE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

ANNUAL_SERIES_PATH = (
    FORECAST_DIR / "25_annual_kaduwela_mean_lst_2015_2025.csv"
)

STEP25_BASELINE_PREDICTIONS_PATH = (
    FORECAST_DIR / "25_annual_level_baseline_rolling_origin_predictions.csv"
)

STEP26A_FREEZE_PATH = (
    REPORT_DIR / "26a_candidate_model_freeze.json"
)

STEP26A_AUDIT_PATH = (
    REPORT_DIR / "26a_annual_level_audit.csv"
)

STEP26B_FREEZE_PATH = (
    REPORT_DIR / "26b_predictor_candidate_freeze.json"
)

STEP26B_AUDIT_PATH = (
    REPORT_DIR / "26b_predictor_forecasting_audit.csv"
)

ANNUAL_RESERVED_PREDICTIONS_PATH = (
    FORECAST_DIR / "27_annual_level_reserved_predictions_2024_2025.csv"
)

ANNUAL_RESERVED_METRICS_PATH = (
    FORECAST_DIR / "27_annual_level_reserved_metrics_by_horizon.csv"
)

ANNUAL_RESERVED_SUMMARY_PATH = (
    FORECAST_DIR / "27_annual_level_reserved_equal_horizon_summary.csv"
)

ANNUAL_SELECTION_PATH = (
    FORECAST_DIR / "27_annual_level_final_selection.csv"
)

PREDICTOR_FOLD_METRICS_PATH = (
    FORECAST_DIR / "27_predictor_reserved_fold_metrics.csv"
)

PREDICTOR_HORIZON_METRICS_PATH = (
    FORECAST_DIR / "27_predictor_reserved_metrics_by_horizon.csv"
)

PREDICTOR_SUMMARY_PATH = (
    FORECAST_DIR / "27_predictor_reserved_equal_horizon_summary.csv"
)

PREDICTOR_SELECTION_PATH = (
    FORECAST_DIR / "27_predictor_final_method_selection.csv"
)

FINAL_FREEZE_PATH = (
    REPORT_DIR / "27_forecast_strategy_freeze_for_step28.json"
)

METHOD_PATH = (
    REPORT_DIR / "27_forecast_validation_methodology.json"
)

AUDIT_PATH = (
    REPORT_DIR / "27_forecast_validation_audit.csv"
)

REPORT_PATH = (
    REPORT_DIR / "27_forecast_validation_report.txt"
)

HASH_PATH = (
    REPORT_DIR / "27_forecast_validation_hashes.csv"
)

ANNUAL_FIGURE_PATH = (
    FIGURE_DIR / "27_annual_level_reserved_validation_mae.png"
)

PREDICTOR_FIGURE_PATH = (
    FIGURE_DIR / "27_predictor_reserved_validation_nmae.png"
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


def annual_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = predicted - actual

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))

    if np.any(actual == 0):
        mape = np.nan
    else:
        mape = float(
            np.mean(np.abs(error / actual)) * 100.0
        )

    if len(actual) < 2:
        r2 = np.nan
    else:
        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        r2 = (
            float(1.0 - ss_res / ss_tot)
            if ss_tot > 0
            else np.nan
        )

    return {
        "MAE_C": mae,
        "RMSE_C": rmse,
        "MAPE_pct": mape,
        "R2": r2,
        "Bias_C": bias,
    }


def predictor_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = predicted - actual

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))

    actual_sd = float(np.std(actual, ddof=0))

    nmae = (
        float(mae / actual_sd)
        if actual_sd > 0
        else np.nan
    )

    nrmse = (
        float(rmse / actual_sd)
        if actual_sd > 0
        else np.nan
    )

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


def annual_candidate_forecast(
    train_years,
    train_values,
    target_year,
    method,
):
    years = np.asarray(train_years, dtype=float)
    values = np.asarray(train_values, dtype=float)

    origin_year = int(years[-1])
    horizon = int(target_year - origin_year)

    if horizon < 1:
        raise ValueError("Target year must be after origin.")

    if method == "Simple Exponential Smoothing":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            fitted = SimpleExpSmoothing(
                values,
                initialization_method="estimated",
            ).fit(
                optimized=True,
                remove_bias=False,
            )

        return float(fitted.forecast(horizon)[-1])

    if method == "Damped Holt Trend":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            fitted = Holt(
                values,
                damped_trend=True,
                initialization_method="estimated",
            ).fit(
                optimized=True,
                remove_bias=False,
            )

        return float(fitted.forecast(horizon)[-1])

    if method == "Theil-Sen Robust Trend":
        slope, intercept, _, _ = theilslopes(
            values,
            years,
            alpha=0.95,
            method="separate",
        )

        return float(intercept + slope * target_year)

    raise ValueError(f"Unknown annual candidate method: {method}")


def is_binary_like(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return False

    unique = np.unique(np.round(values, 12))

    return bool(
        len(unique) <= 2
        and np.all(
            np.isclose(
                unique[:, None],
                np.array([0.0, 1.0]),
                atol=1e-8,
            ).any(axis=1)
        )
    )


def ols_forecast_matrix(history_values, history_years, target_year):
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

    return (
        y_mean
        + slopes * (float(target_year) - x_mean)
    )


def theil_sen_forecast_matrix(
    history_values,
    history_years,
    target_year,
):
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

    return intercepts + slopes * float(target_year)


def constrain_prediction(
    prediction,
    history_values,
    binary_like,
):
    pred = np.asarray(prediction, dtype=float).copy()

    history_min = float(np.nanmin(history_values))
    history_max = float(np.nanmax(history_values))

    pred = np.clip(
        pred,
        history_min,
        history_max,
    )

    if binary_like:
        pred = (pred >= 0.5).astype(float)

    return pred


def predictor_forecast(
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
        raise ValueError(f"Unknown predictor method: {method}")

    return constrain_prediction(
        prediction,
        history_values,
        binary_like,
    )


# =============================================================================
# 0. VERIFY FROZEN INPUTS
# =============================================================================

section(
    "STEP 27 - RESERVED FORECAST VALIDATION AND STRATEGY FREEZE"
)

verify_hash(
    BASE_DATA_PATH,
    EXPECTED_HASHES["kaduwela_preprocessed_base.csv"],
    "Canonical preprocessed base data",
)

verify_hash(
    ANNUAL_SERIES_PATH,
    EXPECTED_HASHES["25_annual_kaduwela_mean_lst_2015_2025.csv"],
    "Step-25 annual Kaduwela mean-LST series",
)

verify_hash(
    STEP25_BASELINE_PREDICTIONS_PATH,
    EXPECTED_HASHES[
        "25_annual_level_baseline_rolling_origin_predictions.csv"
    ],
    "Step-25 annual baseline predictions",
)

verify_hash(
    STEP26A_FREEZE_PATH,
    EXPECTED_HASHES["26a_candidate_model_freeze.json"],
    "Step-26A frozen annual candidate definitions",
)

verify_hash(
    STEP26A_AUDIT_PATH,
    EXPECTED_HASHES["26a_annual_level_audit.csv"],
    "Step-26A strict audit",
)

verify_hash(
    STEP26B_FREEZE_PATH,
    EXPECTED_HASHES["26b_predictor_candidate_freeze.json"],
    "Step-26B frozen predictor candidate definitions",
)

verify_hash(
    STEP26B_AUDIT_PATH,
    EXPECTED_HASHES["26b_predictor_forecasting_audit.csv"],
    "Step-26B strict audit",
)

require_all_pass(
    STEP26A_AUDIT_PATH,
    "Step 26A",
)

require_all_pass(
    STEP26B_AUDIT_PATH,
    "Step 26B",
)

print("\nPASS: Step-26A and Step-26B audits are fully PASS.")
print("\nReserved validation protocol:")
print("- target years = 2024 and 2025 only")
print("- H1-H5 rolling-origin validation")
print("- candidate definitions remain frozen")
print("- no hyperparameter tuning")
print("- no 2026-2030 operational forecast")
print("- Step-22B LightGBM remains untouched")


# =============================================================================
# 1. LOAD DATA + FROZEN DEFINITIONS
# =============================================================================

section(
    "1. LOAD DATA + FROZEN FORECAST DEFINITIONS"
)

annual = pd.read_csv(
    ANNUAL_SERIES_PATH
)

annual[YEAR_COL] = pd.to_numeric(
    annual[YEAR_COL],
    errors="raise",
).astype(int)

annual[ANNUAL_VALUE_COL] = pd.to_numeric(
    annual[ANNUAL_VALUE_COL],
    errors="raise",
)

if sorted(annual[YEAR_COL].tolist()) != EXPECTED_YEARS:
    raise RuntimeError("Annual-series years are not 2015-2025.")

data = pd.read_csv(
    BASE_DATA_PATH,
    dtype={GRID_COL: str},
)

data[YEAR_COL] = pd.to_numeric(
    data[YEAR_COL],
    errors="raise",
).astype(int)

if len(data) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Rows={len(data)}, expected={EXPECTED_ROWS}"
    )

if data[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Grids={data[GRID_COL].nunique()}, expected={EXPECTED_GRIDS}"
    )

if sorted(data[YEAR_COL].unique().tolist()) != EXPECTED_YEARS:
    raise RuntimeError("Base-data years are not 2015-2025.")

with STEP26A_FREEZE_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    freeze26a = json.load(f)

with STEP26B_FREEZE_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    freeze26b = json.load(f)

dynamic_predictors = freeze26b[
    "dynamic_predictors"
]

static_pool_predictors = freeze26b[
    "static_or_frozen_from_candidate_pool"
]

structural_predictors = freeze26b[
    "structural_predictors_not_forecast"
]

if freeze26a["reserved_validation_target_years"] != RESERVED_TARGET_YEARS:
    raise RuntimeError(
        "Step-26A reserved validation years do not match Step 27."
    )

if freeze26b["reserved_validation_target_years"] != RESERVED_TARGET_YEARS:
    raise RuntimeError(
        "Step-26B reserved validation years do not match Step 27."
    )

if set(freeze26a["candidates"].keys()) != set(ANNUAL_CANDIDATE_METHODS):
    raise RuntimeError(
        "Step-26A frozen candidate names differ from Step-27 configuration."
    )

if set(freeze26b["candidate_methods"].keys()) != set(PREDICTOR_CANDIDATE_METHODS):
    raise RuntimeError(
        "Step-26B frozen candidate names differ from Step-27 configuration."
    )

print(f"Dynamic predictors: {len(dynamic_predictors)}")

for predictor in dynamic_predictors:
    print(f"  - {predictor}")

print(
    f"\nStatic candidate-pool predictors: {static_pool_predictors}"
)

print(
    f"Structural predictors: {structural_predictors}"
)


# =============================================================================
# 2. ANNUAL THERMAL-LEVEL RESERVED VALIDATION
# =============================================================================

section(
    "2. ANNUAL THERMAL-LEVEL RESERVED VALIDATION"
)

annual_lookup = (
    annual
    .set_index(YEAR_COL)[ANNUAL_VALUE_COL]
    .to_dict()
)

annual_rows = []

# Step-25 baselines were fixed before Step 26 and already contain predictions
# through 2025. We reuse those exact stored predictions for the reserved years.
step25_predictions = pd.read_csv(
    STEP25_BASELINE_PREDICTIONS_PATH
)

reserved_baselines = step25_predictions[
    step25_predictions["target_year"].isin(RESERVED_TARGET_YEARS)
].copy()

required_baseline_rows = (
    len(ANNUAL_BASELINE_METHODS)
    * len(HORIZONS)
    * len(RESERVED_TARGET_YEARS)
)

if len(reserved_baselines) != required_baseline_rows:
    raise RuntimeError(
        f"Reserved Step-25 baseline rows={len(reserved_baselines)}, "
        f"expected={required_baseline_rows}"
    )

if set(reserved_baselines["method"].unique()) != set(ANNUAL_BASELINE_METHODS):
    raise RuntimeError(
        "Step-25 reserved baseline methods do not match frozen baseline list."
    )

for _, row in reserved_baselines.iterrows():
    annual_rows.append(
        {
            "model_group": "Step25 baseline",
            "method": row["method"],
            "horizon_years": int(row["horizon_years"]),
            "origin_year": int(row["origin_year"]),
            "target_year": int(row["target_year"]),
            "train_start_year": int(row["train_start_year"]),
            "train_end_year": int(row["train_end_year"]),
            "actual_annual_mean_lst_c": float(
                row["actual_annual_mean_lst_c"]
            ),
            "predicted_annual_mean_lst_c": float(
                row["predicted_annual_mean_lst_c"]
            ),
            "error_c": float(row["error_c"]),
            "absolute_error_c": float(row["absolute_error_c"]),
        }
    )

# Step-26A candidates are now evaluated on reserved 2024-2025 targets.
for target_year in RESERVED_TARGET_YEARS:
    for horizon in HORIZONS:
        origin_year = target_year - horizon

        history = annual[
            annual[YEAR_COL] <= origin_year
        ].copy()

        train_years = history[YEAR_COL].to_numpy()
        train_values = history[ANNUAL_VALUE_COL].to_numpy()

        if len(train_values) < 4:
            raise RuntimeError(
                f"Annual candidate history too short for target={target_year}, "
                f"H{horizon}."
            )

        actual = float(
            annual_lookup[target_year]
        )

        for method in ANNUAL_CANDIDATE_METHODS:
            predicted = annual_candidate_forecast(
                train_years=train_years,
                train_values=train_values,
                target_year=target_year,
                method=method,
            )

            annual_rows.append(
                {
                    "model_group": "Step26A candidate",
                    "method": method,
                    "horizon_years": horizon,
                    "origin_year": origin_year,
                    "target_year": target_year,
                    "train_start_year": int(train_years.min()),
                    "train_end_year": int(train_years.max()),
                    "actual_annual_mean_lst_c": actual,
                    "predicted_annual_mean_lst_c": predicted,
                    "error_c": predicted - actual,
                    "absolute_error_c": abs(predicted - actual),
                }
            )

annual_reserved_predictions = pd.DataFrame(
    annual_rows
)

annual_reserved_predictions.to_csv(
    ANNUAL_RESERVED_PREDICTIONS_PATH,
    index=False,
)

print(
    annual_reserved_predictions.sort_values(
        [
            "target_year",
            "horizon_years",
            "model_group",
            "method",
        ]
    ).to_string(index=False)
)


# =============================================================================
# 3. ANNUAL RESERVED METRICS + FINAL SELECTION
# =============================================================================

section(
    "3. ANNUAL RESERVED METRICS + FINAL SELECTION"
)

annual_metric_rows = []

for (method, horizon), part in annual_reserved_predictions.groupby(
    ["method", "horizon_years"]
):
    metrics = annual_metrics(
        part["actual_annual_mean_lst_c"],
        part["predicted_annual_mean_lst_c"],
    )

    group = part["model_group"].iloc[0]

    annual_metric_rows.append(
        {
            "model_group": group,
            "method": method,
            "horizon_years": int(horizon),
            "reserved_target_count": int(len(part)),
            **metrics,
        }
    )

annual_metrics_by_horizon = pd.DataFrame(
    annual_metric_rows
)

annual_metrics_by_horizon.to_csv(
    ANNUAL_RESERVED_METRICS_PATH,
    index=False,
)

annual_summary_rows = []

for (group, method), part in annual_metrics_by_horizon.groupby(
    ["model_group", "method"]
):
    annual_summary_rows.append(
        {
            "model_group": group,
            "method": method,
            "horizon_count": int(part["horizon_years"].nunique()),
            "mean_horizon_MAE_C": float(part["MAE_C"].mean()),
            "mean_horizon_RMSE_C": float(part["RMSE_C"].mean()),
            "mean_horizon_MAPE_pct": float(part["MAPE_pct"].mean()),
            "mean_horizon_R2": float(part["R2"].mean(skipna=True)),
            "mean_abs_horizon_Bias_C": float(
                part["Bias_C"].abs().mean()
            ),
        }
    )

annual_summary = (
    pd.DataFrame(annual_summary_rows)
    .sort_values(
        [
            "mean_horizon_MAE_C",
            "mean_horizon_RMSE_C",
            "mean_abs_horizon_Bias_C",
        ]
    )
    .reset_index(drop=True)
)

annual_summary.to_csv(
    ANNUAL_RESERVED_SUMMARY_PATH,
    index=False,
)

annual_selected = annual_summary.iloc[0].copy()

annual_selection = pd.DataFrame(
    [
        {
            **annual_selected.to_dict(),
            "selection_status": "selected_for_step28",
            "selection_rule": (
                "lowest reserved mean-horizon MAE; tie-break RMSE then absolute bias"
            ),
        }
    ]
)

annual_selection.to_csv(
    ANNUAL_SELECTION_PATH,
    index=False,
)

print(
    "Reserved annual-level equal-horizon summary:"
)

print(
    annual_summary.to_string(index=False)
)

print(
    "\nFINAL annual-level method selected for Step 28:"
)

print(
    annual_selection.to_string(index=False)
)


# =============================================================================
# 4. DYNAMIC-PREDICTOR RESERVED VALIDATION
# =============================================================================

section(
    "4. DYNAMIC-PREDICTOR RESERVED VALIDATION"
)

predictor_fold_rows = []

for predictor_number, predictor in enumerate(
    dynamic_predictors,
    start=1,
):
    print(
        f"\nPredictor {predictor_number}/{len(dynamic_predictors)}: {predictor}"
    )

    if predictor not in data.columns:
        raise RuntimeError(
            f"Missing dynamic predictor column: {predictor}"
        )

    data[predictor] = pd.to_numeric(
        data[predictor],
        errors="raise",
    )

    wide = (
        data.pivot(
            index=GRID_COL,
            columns=YEAR_COL,
            values=predictor,
        )
        .sort_index()
        .sort_index(axis=1)
    )

    if wide.shape != (
        EXPECTED_GRIDS,
        len(EXPECTED_YEARS),
    ):
        raise RuntimeError(
            f"{predictor}: unexpected wide shape {wide.shape}"
        )

    if wide.isna().any().any():
        raise RuntimeError(
            f"{predictor}: missing values in wide matrix."
        )

    # Keep the Step-26B binary/non-binary classification based on <=2023 only.
    development_values = (
        wide[
            [year for year in EXPECTED_YEARS if year <= 2023]
        ]
        .to_numpy(dtype=float)
    )

    binary_like = is_binary_like(
        development_values.ravel()
    )

    for target_year in RESERVED_TARGET_YEARS:
        for horizon in HORIZONS:
            origin_year = target_year - horizon

            history_years = [
                year
                for year in EXPECTED_YEARS
                if year <= origin_year
            ]

            history_values = (
                wide[history_years]
                .to_numpy(dtype=float)
            )

            actual = (
                wide[target_year]
                .to_numpy(dtype=float)
            )

            if history_values.shape[1] < 4:
                raise RuntimeError(
                    f"{predictor}: fewer than four years at origin {origin_year}."
                )

            for method in PREDICTOR_CANDIDATE_METHODS:
                predicted = predictor_forecast(
                    method=method,
                    history_values=history_values,
                    history_years=history_years,
                    target_year=target_year,
                    binary_like=binary_like,
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

                range_violation_count = int(
                    np.sum(predicted < history_min - 1e-12)
                    + np.sum(predicted > history_max + 1e-12)
                )

                predictor_fold_rows.append(
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
                        "binary_like": binary_like,
                        "training_history_min": history_min,
                        "training_history_max": history_max,
                        "range_violation_count": range_violation_count,
                        **metrics,
                    }
                )

        print(
            f"  Reserved targets completed: {RESERVED_TARGET_YEARS}"
        )

predictor_fold_metrics = pd.DataFrame(
    predictor_fold_rows
)

predictor_fold_metrics.to_csv(
    PREDICTOR_FOLD_METRICS_PATH,
    index=False,
)

print(
    f"\nSaved: {PREDICTOR_FOLD_METRICS_PATH}"
)


# =============================================================================
# 5. PREDICTOR RESERVED METRICS + FINAL METHOD PER PREDICTOR
# =============================================================================

section(
    "5. PREDICTOR RESERVED METRICS + FINAL METHOD PER PREDICTOR"
)

predictor_horizon_rows = []

for (
    predictor,
    method,
    horizon,
), part in predictor_fold_metrics.groupby(
    [
        "predictor",
        "method",
        "horizon_years",
    ]
):
    predictor_horizon_rows.append(
        {
            "predictor": predictor,
            "method": method,
            "horizon_years": int(horizon),
            "reserved_target_count": int(len(part)),
            "mean_MAE": float(part["MAE"].mean()),
            "mean_RMSE": float(part["RMSE"].mean()),
            "mean_R2": float(part["R2"].mean()),
            "mean_abs_Bias": float(part["Bias"].abs().mean()),
            "mean_NMAE_SD": float(part["NMAE_SD"].mean()),
            "mean_NRMSE_SD": float(part["NRMSE_SD"].mean()),
        }
    )

predictor_horizon_metrics = pd.DataFrame(
    predictor_horizon_rows
)

predictor_horizon_metrics.to_csv(
    PREDICTOR_HORIZON_METRICS_PATH,
    index=False,
)

predictor_summary_rows = []

for (
    predictor,
    method,
), part in predictor_horizon_metrics.groupby(
    [
        "predictor",
        "method",
    ]
):
    predictor_summary_rows.append(
        {
            "predictor": predictor,
            "method": method,
            "horizon_count": int(part["horizon_years"].nunique()),
            "mean_horizon_MAE": float(part["mean_MAE"].mean()),
            "mean_horizon_RMSE": float(part["mean_RMSE"].mean()),
            "mean_horizon_R2": float(part["mean_R2"].mean()),
            "mean_horizon_NMAE_SD": float(
                part["mean_NMAE_SD"].mean()
            ),
            "mean_horizon_NRMSE_SD": float(
                part["mean_NRMSE_SD"].mean()
            ),
            "mean_abs_horizon_Bias": float(
                part["mean_abs_Bias"].mean()
            ),
        }
    )

predictor_summary = pd.DataFrame(
    predictor_summary_rows
)

predictor_summary.to_csv(
    PREDICTOR_SUMMARY_PATH,
    index=False,
)

selection_rows = []

for predictor in dynamic_predictors:
    part = (
        predictor_summary[
            predictor_summary["predictor"] == predictor
        ]
        .sort_values(
            [
                "mean_horizon_NMAE_SD",
                "mean_horizon_NRMSE_SD",
                "mean_abs_horizon_Bias",
            ]
        )
        .reset_index(drop=True)
    )

    selected = part.iloc[0].to_dict()

    selection_rows.append(
        {
            **selected,
            "selection_status": "selected_for_step28",
            "selection_rule": (
                "lowest reserved mean-horizon NMAE_SD; "
                "tie-break NRMSE_SD then absolute bias"
            ),
        }
    )

predictor_selection = pd.DataFrame(
    selection_rows
)

predictor_selection.to_csv(
    PREDICTOR_SELECTION_PATH,
    index=False,
)

print(
    "Reserved predictor summary:"
)

print(
    predictor_summary.sort_values(
        [
            "predictor",
            "mean_horizon_NMAE_SD",
        ]
    ).to_string(index=False)
)

print(
    "\nFINAL predictor methods selected for Step 28:"
)

print(
    predictor_selection[
        [
            "predictor",
            "method",
            "mean_horizon_MAE",
            "mean_horizon_RMSE",
            "mean_horizon_R2",
            "mean_horizon_NMAE_SD",
            "mean_horizon_NRMSE_SD",
        ]
    ].to_string(index=False)
)


# =============================================================================
# 6. FREEZE FINAL FORECAST STRATEGY FOR STEP 28
# =============================================================================

section(
    "6. FREEZE FINAL FORECAST STRATEGY FOR STEP 28"
)

selected_predictor_methods = {
    row["predictor"]: row["method"]
    for _, row in predictor_selection.iterrows()
}

final_freeze = {
    "step": 27,
    "purpose": "reserved forecast validation and Step-28 strategy freeze",
    "reserved_validation_target_years": RESERVED_TARGET_YEARS,
    "horizons_years": HORIZONS,
    "annual_level": {
        "selected_method": str(
            annual_selection.iloc[0]["method"]
        ),
        "selected_model_group": str(
            annual_selection.iloc[0]["model_group"]
        ),
        "selection_rule": (
            "one coherent method across H1-H5; lowest reserved mean-horizon MAE, "
            "tie-break RMSE then absolute bias"
        ),
    },
    "dynamic_predictors": {
        "selected_methods": selected_predictor_methods,
        "selection_rule": (
            "one coherent method per predictor across H1-H5; lowest reserved "
            "mean-horizon NMAE_SD, tie-break NRMSE_SD then absolute bias"
        ),
    },
    "static_or_frozen_candidate_pool": static_pool_predictors,
    "structural_predictors_not_forecast": structural_predictors,
    "future_feature_policy": (
        "Do not forecast engineered features independently. Step 28 must "
        "reconstruct lag/delta/history/interaction/neighbour features from the "
        "forecast base-predictor sequence using the original feature-engineering logic."
    ),
    "future_origin_policy": (
        "Step 28 operational forecasts use actual observed history through 2025 "
        "and generate H1-H5 for 2026-2030."
    ),
    "spatial_anomaly_model_policy": (
        "Use the existing frozen Step-22B production LightGBM trained on actual "
        "2018-2025; do not retrain it on pseudo-future forecasts."
    ),
    "future_grid_lst_equation": (
        "future grid LST = forecast annual Kaduwela thermal level + "
        "centered Step-22B spatial-anomaly prediction"
    ),
    "future_2026_2030_generated_in_step27": False,
}

FINAL_FREEZE_PATH.write_text(
    json.dumps(
        final_freeze,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        final_freeze,
        indent=2,
    )
)


# =============================================================================
# 7. FIGURES
# =============================================================================

section(
    "7. SAVE STEP-27 VALIDATION FIGURES"
)

annual_plot = annual_summary.sort_values(
    "mean_horizon_MAE_C",
    ascending=True,
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.barh(
    annual_plot["method"],
    annual_plot["mean_horizon_MAE_C"],
)

ax.set_xlabel(
    "Reserved mean-horizon MAE (°C)"
)

ax.set_ylabel(
    "Annual-level method"
)

ax.set_title(
    "Step 27 Reserved Annual Thermal-Level Validation"
)

ax.grid(
    axis="x",
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    ANNUAL_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

predictor_plot = predictor_selection.sort_values(
    "mean_horizon_NMAE_SD",
    ascending=True,
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

labels = (
    predictor_plot["predictor"]
    + " — "
    + predictor_plot["method"]
)

ax.barh(
    labels,
    predictor_plot["mean_horizon_NMAE_SD"],
)

ax.set_xlabel(
    "Reserved mean-horizon normalized MAE"
)

ax.set_ylabel(
    "Selected predictor strategy"
)

ax.set_title(
    "Step 27 Selected Dynamic-Predictor Forecast Strategies"
)

ax.grid(
    axis="x",
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    PREDICTOR_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved: {ANNUAL_FIGURE_PATH}")
print(f"Saved: {PREDICTOR_FIGURE_PATH}")


# =============================================================================
# 8. METHODOLOGY
# =============================================================================

section(
    "8. SAVE STEP-27 METHODOLOGY"
)

methodology = {
    "step": 27,
    "analysis": "Reserved forecast validation",
    "reserved_target_years": RESERVED_TARGET_YEARS,
    "forecast_horizons_years": HORIZONS,
    "rolling_origin_rule": (
        "For every target/horizon pair, training history stops at "
        "origin = target year - horizon."
    ),
    "2025_h1_note": (
        "The 2025 H1 forecast may use observed 2024 history because 2024 is "
        "known at the 2025 forecast origin. This is valid rolling-origin forecasting."
    ),
    "annual_candidates": (
        ANNUAL_BASELINE_METHODS
        + ANNUAL_CANDIDATE_METHODS
    ),
    "annual_selection_rule": (
        "Select one method across H1-H5 by reserved mean-horizon MAE; "
        "tie-break RMSE and absolute bias."
    ),
    "dynamic_predictors": dynamic_predictors,
    "predictor_candidates": PREDICTOR_CANDIDATE_METHODS,
    "predictor_selection_rule": (
        "Select one method per predictor across H1-H5 by reserved mean-horizon "
        "NMAE_SD; tie-break NRMSE_SD and absolute bias."
    ),
    "why_one_method_per_predictor": (
        "Only two reserved target years are available per horizon. A single "
        "method per predictor avoids overfitting a separate method to each horizon "
        "and produces a more coherent 2026-2030 predictor path."
    ),
    "mape_predictors": (
        "Not used because indices can be zero or negative."
    ),
    "post_step27_status": (
        "After Step 27, 2024-2025 are validation evidence and are no longer held out. "
        "Step 28 may use all actual observations through 2025 for operational forecasting."
    ),
    "step22b_policy": (
        "Production LightGBM remains frozen and is not re-tested/retrained here."
    ),
    "future_forecast_policy": (
        "No 2026-2030 operational forecast is generated until Step 28."
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
# 9. STRICT AUDIT
# =============================================================================

section(
    "9. STRICT STEP-27 AUDIT"
)

annual_reserved_only = bool(
    set(
        annual_reserved_predictions["target_year"].unique()
    )
    == set(RESERVED_TARGET_YEARS)
)

annual_training_precedes_target = bool(
    (
        annual_reserved_predictions["train_end_year"]
        < annual_reserved_predictions["target_year"]
    ).all()
)

annual_origin_matches_cutoff = bool(
    (
        annual_reserved_predictions["origin_year"]
        == annual_reserved_predictions["train_end_year"]
    ).all()
)

annual_methods_complete = bool(
    set(
        annual_reserved_predictions["method"].unique()
    )
    == set(
        ANNUAL_BASELINE_METHODS
        + ANNUAL_CANDIDATE_METHODS
    )
)

annual_horizons_complete = bool(
    set(
        annual_reserved_predictions["horizon_years"].unique()
    )
    == set(HORIZONS)
)

expected_annual_reserved_rows = (
    (
        len(ANNUAL_BASELINE_METHODS)
        + len(ANNUAL_CANDIDATE_METHODS)
    )
    * len(HORIZONS)
    * len(RESERVED_TARGET_YEARS)
)

actual_annual_reserved_rows = len(
    annual_reserved_predictions
)

predictor_reserved_only = bool(
    set(
        predictor_fold_metrics["target_year"].unique()
    )
    == set(RESERVED_TARGET_YEARS)
)

predictor_training_precedes_target = bool(
    (
        predictor_fold_metrics["training_end_year"]
        < predictor_fold_metrics["target_year"]
    ).all()
)

predictor_origin_matches_cutoff = bool(
    (
        predictor_fold_metrics["origin_year"]
        == predictor_fold_metrics["training_end_year"]
    ).all()
)

predictor_ranges_valid = bool(
    (
        predictor_fold_metrics["range_violation_count"]
        == 0
    ).all()
)

expected_predictor_fold_rows = (
    len(dynamic_predictors)
    * len(PREDICTOR_CANDIDATE_METHODS)
    * len(HORIZONS)
    * len(RESERVED_TARGET_YEARS)
)

actual_predictor_fold_rows = len(
    predictor_fold_metrics
)

predictor_selection_complete = bool(
    set(
        predictor_selection["predictor"]
    )
    == set(dynamic_predictors)
    and len(predictor_selection) == len(dynamic_predictors)
)

all_predictor_horizons_complete = True

for predictor in dynamic_predictors:
    part = predictor_horizon_metrics[
        predictor_horizon_metrics["predictor"] == predictor
    ]

    if (
        set(part["method"]) != set(PREDICTOR_CANDIDATE_METHODS)
        or set(part["horizon_years"]) != set(HORIZONS)
    ):
        all_predictor_horizons_complete = False
        break

audit_rows = [
    {
        "check": "canonical_base_hash",
        "status": "PASS",
        "details": EXPECTED_HASHES["kaduwela_preprocessed_base.csv"],
    },
    {
        "check": "step26a_audit",
        "status": "PASS",
        "details": "Step-26A audit verified fully PASS.",
    },
    {
        "check": "step26b_audit",
        "status": "PASS",
        "details": "Step-26B audit verified fully PASS.",
    },
    {
        "check": "annual_reserved_targets_only",
        "status": "PASS" if annual_reserved_only else "FAIL",
        "details": str(
            sorted(
                annual_reserved_predictions["target_year"]
                .unique()
                .tolist()
            )
        ),
    },
    {
        "check": "annual_training_precedes_target",
        "status": "PASS" if annual_training_precedes_target else "FAIL",
        "details": "All annual training histories stop before target year.",
    },
    {
        "check": "annual_origin_matches_cutoff",
        "status": "PASS" if annual_origin_matches_cutoff else "FAIL",
        "details": "Annual origin equals final observed training year.",
    },
    {
        "check": "annual_methods_complete",
        "status": "PASS" if annual_methods_complete else "FAIL",
        "details": (
            f"{len(ANNUAL_BASELINE_METHODS) + len(ANNUAL_CANDIDATE_METHODS)} "
            "fixed annual methods evaluated."
        ),
    },
    {
        "check": "annual_horizons_complete",
        "status": "PASS" if annual_horizons_complete else "FAIL",
        "details": "H1-H5 evaluated.",
    },
    {
        "check": "annual_expected_reserved_rows",
        "status": (
            "PASS"
            if actual_annual_reserved_rows == expected_annual_reserved_rows
            else "FAIL"
        ),
        "details": (
            f"actual={actual_annual_reserved_rows}, "
            f"expected={expected_annual_reserved_rows}"
        ),
    },
    {
        "check": "annual_final_selection_unique",
        "status": "PASS" if len(annual_selection) == 1 else "FAIL",
        "details": (
            f"selected={annual_selection.iloc[0]['method']}"
            if len(annual_selection) == 1
            else "selection_count_invalid"
        ),
    },
    {
        "check": "predictor_reserved_targets_only",
        "status": "PASS" if predictor_reserved_only else "FAIL",
        "details": str(
            sorted(
                predictor_fold_metrics["target_year"]
                .unique()
                .tolist()
            )
        ),
    },
    {
        "check": "predictor_training_precedes_target",
        "status": "PASS" if predictor_training_precedes_target else "FAIL",
        "details": "All predictor histories stop before target year.",
    },
    {
        "check": "predictor_origin_matches_cutoff",
        "status": "PASS" if predictor_origin_matches_cutoff else "FAIL",
        "details": "Predictor origin equals final observed training year.",
    },
    {
        "check": "predictor_expected_fold_rows",
        "status": (
            "PASS"
            if actual_predictor_fold_rows == expected_predictor_fold_rows
            else "FAIL"
        ),
        "details": (
            f"actual={actual_predictor_fold_rows}, "
            f"expected={expected_predictor_fold_rows}"
        ),
    },
    {
        "check": "predictor_methods_horizons_complete",
        "status": "PASS" if all_predictor_horizons_complete else "FAIL",
        "details": "All frozen methods evaluated across H1-H5 for every dynamic predictor.",
    },
    {
        "check": "predictor_range_constraints",
        "status": "PASS" if predictor_ranges_valid else "FAIL",
        "details": "All bounded predictor forecasts remain inside origin-known range.",
    },
    {
        "check": "one_final_method_per_dynamic_predictor",
        "status": "PASS" if predictor_selection_complete else "FAIL",
        "details": (
            f"selected_predictors={len(predictor_selection)}, "
            f"expected={len(dynamic_predictors)}"
        ),
    },
    {
        "check": "no_hyperparameter_tuning",
        "status": "PASS",
        "details": "Step 27 evaluates only previously frozen definitions.",
    },
    {
        "check": "final_strategy_freeze_written",
        "status": "PASS" if FINAL_FREEZE_PATH.exists() else "FAIL",
        "details": "Selected forecast strategies frozen for Step 28.",
    },
    {
        "check": "no_2026_2030_operational_forecast",
        "status": "PASS",
        "details": "Step 27 contains validation only.",
    },
    {
        "check": "step22b_model_untouched",
        "status": "PASS",
        "details": "Production spatial-anomaly LightGBM is not loaded or changed.",
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
    audit_df.to_string(index=False)
)

if (audit_df["status"] != "PASS").any():
    raise RuntimeError(
        "One or more Step-27 audit checks failed."
    )


# =============================================================================
# 10. REPORT
# =============================================================================

section(
    "10. SAVE STEP-27 REPORT"
)

report_lines = [
    "STEP 27 - RESERVED FORECAST VALIDATION AND STRATEGY FREEZE",
    "=" * 116,
    "",
    "RESERVED VALIDATION YEARS",
    "-" * 116,
    "2024 and 2025 only.",
    "",
    "ANNUAL THERMAL-LEVEL VALIDATION",
    "-" * 116,
    annual_summary.to_string(index=False),
    "",
    "FINAL ANNUAL-LEVEL SELECTION",
    "-" * 116,
    annual_selection.to_string(index=False),
    "",
    "FINAL DYNAMIC-PREDICTOR METHOD SELECTIONS",
    "-" * 116,
    predictor_selection[
        [
            "predictor",
            "method",
            "mean_horizon_MAE",
            "mean_horizon_RMSE",
            "mean_horizon_R2",
            "mean_horizon_NMAE_SD",
            "mean_horizon_NRMSE_SD",
        ]
    ].to_string(index=False),
    "",
    "INTERPRETATION",
    "-" * 116,
    (
        "These selections are based on reserved forecast-validation evidence, "
        "not on future 2026-2030 observations."
    ),
    (
        "Only two reserved target years exist per horizon, so small-sample "
        "limitations must be stated in the final research report."
    ),
    (
        "The selected methods are now frozen for Step 28 operational 2026-2030 generation."
    ),
    (
        "After Step 27, all actual observations through 2025 may be used as history "
        "for Step 28 operational forecasting."
    ),
    (
        "The Step-22B production LightGBM remains unchanged."
    ),
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {REPORT_PATH}")


# =============================================================================
# 11. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "11. REPRODUCIBILITY HASHES"
)

hash_files = [
    BASE_DATA_PATH,
    ANNUAL_SERIES_PATH,
    STEP25_BASELINE_PREDICTIONS_PATH,
    STEP26A_FREEZE_PATH,
    STEP26A_AUDIT_PATH,
    STEP26B_FREEZE_PATH,
    STEP26B_AUDIT_PATH,
    ANNUAL_RESERVED_PREDICTIONS_PATH,
    ANNUAL_RESERVED_METRICS_PATH,
    ANNUAL_RESERVED_SUMMARY_PATH,
    ANNUAL_SELECTION_PATH,
    PREDICTOR_FOLD_METRICS_PATH,
    PREDICTOR_HORIZON_METRICS_PATH,
    PREDICTOR_SUMMARY_PATH,
    PREDICTOR_SELECTION_PATH,
    FINAL_FREEZE_PATH,
    ANNUAL_FIGURE_PATH,
    PREDICTOR_FIGURE_PATH,
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

    print(f"{path.name}: {digest}")

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
    "STEP 27 COMPLETED SUCCESSFULLY"
)

print(
    "Reserved forecast validation completed using 2024 and 2025 only.\n"
    "All annual-level fixed methods were evaluated across H1-H5.\n"
    "One annual-level method was selected and frozen for Step 28.\n"
    f"All {len(dynamic_predictors)} dynamic predictors were validated across H1-H5.\n"
    "One coherent method per dynamic predictor was selected and frozen for Step 28.\n"
    "No hyperparameter tuning was performed in Step 27.\n"
    "No 2026-2030 operational forecasts were generated.\n"
    "The Step-22B production LightGBM was not changed.\n"
    "\nNEXT: review Step-27 validation results before starting Step 28."
)
