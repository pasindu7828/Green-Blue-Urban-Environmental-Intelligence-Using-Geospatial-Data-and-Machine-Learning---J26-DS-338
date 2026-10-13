"""
STEP 25 - FORECASTING BASELINE
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Create honest baseline forecasts for the ANNUAL THERMAL LEVEL component of
the final decomposed future-LST framework.

Final future framework:
    Future grid LST
      = forecast Kaduwela annual thermal level
      + forecast centered spatial anomaly

Step 25 addresses the first part only:
    Kaduwela annual thermal level baseline forecasting.

This is important because Step 22 showed that the main remaining error came
from annual thermal-level forecasting, while centered spatial-pattern
performance was substantially stronger.

NO final 2026-2030 heat map is created here.
NO hyperparameter tuning is performed here.
NO model-selection decision for the final future forecasting model is made here.

DATA
----
Observed annual Kaduwela mean LST is computed from the canonical
kaduwela_preprocessed_base.csv for 2015-2025.

2025 may now be used for post-lockbox forecasting-method development because
the independent Step-22 final test has already been completed and frozen.
This must NOT be interpreted as re-testing or modifying the Step-22 claim.

PRIMARY BASELINE EVALUATION
---------------------------
Rolling-origin hindcasting using only information available at each origin.

Minimum history:
    2015-2018 = 4 annual observations

Forecast horizons:
    H1, H2, H3, H4, H5 years

Baselines:
1. Persistence
       future level = last observed annual level

2. Expanding Mean
       future level = mean of all observed annual levels available at origin

3. Rolling-3 Mean
       future level = mean of the latest 3 observed annual levels

4. Drift
       y_hat(origin+h)
       = y_last + h * (y_last - y_first) / (n - 1)

5. OLS Linear Trend
       least-squares annual trend estimated only from history available at
       the forecast origin and extrapolated h years ahead

PRIMARY ANALYSIS RETAINS 2017.
A separate no-2017 sensitivity analysis is saved for robustness only.
It does not replace the primary analysis and is not used to manipulate results.

METRICS
-------
MAE, RMSE, MAPE, R²

Outputs are reported:
- per model and horizon;
- equal-horizon summary;
- pooled-fold descriptive summary;
- per-origin predictions;
- no-2017 sensitivity;
- figures;
- methodology;
- strict audit;
- SHA256 hashes.
"""

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET = "LST_C"

EXPECTED_RAW_ROWS = 24134
EXPECTED_GRIDS = 2194
EXPECTED_YEARS = list(range(2015, 2026))

MIN_ORIGIN_YEAR = 2018
MAX_OBSERVED_YEAR = 2025

HORIZONS = [1, 2, 3, 4, 5]

BASELINES = [
    "Persistence",
    "Expanding Mean",
    "Rolling-3 Mean",
    "Drift",
    "OLS Linear Trend",
]

EXPECTED_BASE_HASH = (
    "0e030b2ec0b5c4080cd4226bec798189a178aa45efd9962d19f38dc9cb170f46"
)


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "kaduwela_preprocessed_base.csv"
)

FORECAST_DIR = ROOT / "outputs" / "forecasting"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [
    FORECAST_DIR,
    FIGURE_DIR,
    REPORT_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

ANNUAL_SERIES_PATH = (
    FORECAST_DIR
    / "25_annual_kaduwela_mean_lst_2015_2025.csv"
)

PREDICTIONS_PATH = (
    FORECAST_DIR
    / "25_annual_level_baseline_rolling_origin_predictions.csv"
)

METRICS_PATH = (
    FORECAST_DIR
    / "25_annual_level_baseline_metrics_by_horizon.csv"
)

SUMMARY_PATH = (
    FORECAST_DIR
    / "25_annual_level_baseline_equal_horizon_summary.csv"
)

POOLED_PATH = (
    FORECAST_DIR
    / "25_annual_level_baseline_pooled_fold_summary.csv"
)

BEST_BY_HORIZON_PATH = (
    FORECAST_DIR
    / "25_annual_level_best_baseline_by_horizon.csv"
)

NO2017_PREDICTIONS_PATH = (
    FORECAST_DIR
    / "25_no2017_baseline_predictions.csv"
)

NO2017_METRICS_PATH = (
    FORECAST_DIR
    / "25_no2017_baseline_metrics_by_horizon.csv"
)

NO2017_COMPARISON_PATH = (
    FORECAST_DIR
    / "25_no2017_baseline_sensitivity_comparison.csv"
)

SERIES_FIGURE_PATH = (
    FIGURE_DIR
    / "25_annual_kaduwela_mean_lst_series.png"
)

MAE_FIGURE_PATH = (
    FIGURE_DIR
    / "25_baseline_mae_by_horizon.png"
)

METHOD_PATH = (
    REPORT_DIR
    / "25_forecasting_baseline_methodology.json"
)

AUDIT_PATH = (
    REPORT_DIR
    / "25_forecasting_baseline_audit.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "25_forecasting_baseline_report.txt"
)

HASH_PATH = (
    REPORT_DIR
    / "25_forecasting_baseline_hashes.csv"
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
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def verify_hash(path, expected, label):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual  : {actual}"
        )

    print(
        f"PASS hash: {label} = {actual}"
    )

    return actual


def regression_metrics(actual, predicted):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted lengths differ."
        )

    errors = predicted - actual

    mae = float(
        np.mean(
            np.abs(errors)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    if np.any(actual == 0):
        mape = np.nan
    else:
        mape = float(
            np.mean(
                np.abs(
                    errors / actual
                )
            )
            * 100.0
        )

    if len(actual) < 2:
        r2 = np.nan
    else:
        ss_res = float(
            np.sum(
                (actual - predicted) ** 2
            )
        )

        ss_tot = float(
            np.sum(
                (actual - np.mean(actual)) ** 2
            )
        )

        if ss_tot == 0:
            r2 = np.nan
        else:
            r2 = float(
                1.0
                - (
                    ss_res
                    / ss_tot
                )
            )

    bias = float(
        np.mean(
            errors
        )
    )

    return {
        "MAE_C": mae,
        "RMSE_C": rmse,
        "MAPE_pct": mape,
        "R2": r2,
        "Bias_C": bias,
    }


def predict_baseline(
    train_years,
    train_values,
    target_year,
    method,
):
    years = np.asarray(
        train_years,
        dtype=float,
    )

    values = np.asarray(
        train_values,
        dtype=float,
    )

    if len(years) != len(values):
        raise ValueError(
            "Training years and values have different lengths."
        )

    if len(values) < 1:
        raise ValueError(
            "No history available."
        )

    origin_year = int(
        years[-1]
    )

    horizon = int(
        target_year
        - origin_year
    )

    if horizon < 1:
        raise ValueError(
            "Target year must be after origin year."
        )

    if method == "Persistence":
        return float(
            values[-1]
        )

    if method == "Expanding Mean":
        return float(
            np.mean(
                values
            )
        )

    if method == "Rolling-3 Mean":
        latest = values[
            -min(
                3,
                len(values),
            ):
        ]

        return float(
            np.mean(
                latest
            )
        )

    if method == "Drift":
        if len(values) < 2:
            return float(
                values[-1]
            )

        annual_drift = (
            values[-1]
            - values[0]
        ) / (
            years[-1]
            - years[0]
        )

        return float(
            values[-1]
            + horizon
            * annual_drift
        )

    if method == "OLS Linear Trend":
        if len(values) < 2:
            return float(
                values[-1]
            )

        slope, intercept = np.polyfit(
            years,
            values,
            1,
        )

        return float(
            intercept
            + slope
            * target_year
        )

    raise ValueError(
        f"Unknown baseline method: {method}"
    )


def rolling_origin_predictions(
    annual_df,
    exclude_2017_from_training=False,
):
    lookup = (
        annual_df
        .set_index(
            YEAR_COL
        )[
            "annual_mean_lst_c"
        ]
        .to_dict()
    )

    rows = []

    for horizon in HORIZONS:
        for origin_year in range(
            MIN_ORIGIN_YEAR,
            MAX_OBSERVED_YEAR,
        ):
            target_year = (
                origin_year
                + horizon
            )

            if target_year > MAX_OBSERVED_YEAR:
                continue

            history = annual_df[
                annual_df[
                    YEAR_COL
                ]
                <= origin_year
            ].copy()

            if exclude_2017_from_training:
                history = history[
                    history[
                        YEAR_COL
                    ]
                    != 2017
                ].copy()

            if len(history) < 3:
                continue

            train_years = history[
                YEAR_COL
            ].to_numpy()

            train_values = history[
                "annual_mean_lst_c"
            ].to_numpy()

            actual = float(
                lookup[
                    target_year
                ]
            )

            for method in BASELINES:
                predicted = predict_baseline(
                    train_years=train_years,
                    train_values=train_values,
                    target_year=target_year,
                    method=method,
                )

                rows.append(
                    {
                        "analysis": (
                            "no_2017_training_sensitivity"
                            if exclude_2017_from_training
                            else "primary_with_2017"
                        ),
                        "method": method,
                        "horizon_years": horizon,
                        "origin_year": origin_year,
                        "target_year": target_year,
                        "train_start_year": int(
                            train_years.min()
                        ),
                        "train_end_year": int(
                            train_years.max()
                        ),
                        "training_year_count": int(
                            len(
                                train_years
                            )
                        ),
                        "actual_annual_mean_lst_c": actual,
                        "predicted_annual_mean_lst_c": predicted,
                        "error_c": float(
                            predicted
                            - actual
                        ),
                        "absolute_error_c": float(
                            abs(
                                predicted
                                - actual
                            )
                        ),
                    }
                )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        raise RuntimeError(
            "No rolling-origin baseline predictions were generated."
        )

    return result


def metrics_by_horizon(predictions):
    rows = []

    for method in BASELINES:
        for horizon in HORIZONS:
            part = predictions[
                (
                    predictions[
                        "method"
                    ]
                    == method
                )
                & (
                    predictions[
                        "horizon_years"
                    ]
                    == horizon
                )
            ]

            if part.empty:
                continue

            metrics = regression_metrics(
                part[
                    "actual_annual_mean_lst_c"
                ],
                part[
                    "predicted_annual_mean_lst_c"
                ],
            )

            rows.append(
                {
                    "method": method,
                    "horizon_years": horizon,
                    "fold_count": len(
                        part
                    ),
                    **metrics,
                }
            )

    return pd.DataFrame(
        rows
    )


def equal_horizon_summary(metrics_df):
    rows = []

    for method in BASELINES:
        part = metrics_df[
            metrics_df[
                "method"
            ]
            == method
        ].copy()

        if part.empty:
            continue

        rows.append(
            {
                "method": method,
                "horizon_count": int(
                    part[
                        "horizon_years"
                    ].nunique()
                ),
                "mean_horizon_MAE_C": float(
                    part[
                        "MAE_C"
                    ].mean()
                ),
                "mean_horizon_RMSE_C": float(
                    part[
                        "RMSE_C"
                    ].mean()
                ),
                "mean_horizon_MAPE_pct": float(
                    part[
                        "MAPE_pct"
                    ].mean()
                ),
                "mean_horizon_R2": float(
                    part[
                        "R2"
                    ].mean()
                ),
                "mean_abs_horizon_Bias_C": float(
                    part[
                        "Bias_C"
                    ]
                    .abs()
                    .mean()
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        [
            "mean_horizon_MAE_C",
            "mean_horizon_RMSE_C",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )


def pooled_summary(predictions):
    rows = []

    for method in BASELINES:
        part = predictions[
            predictions[
                "method"
            ]
            == method
        ]

        metrics = regression_metrics(
            part[
                "actual_annual_mean_lst_c"
            ],
            part[
                "predicted_annual_mean_lst_c"
            ],
        )

        rows.append(
            {
                "method": method,
                "fold_count": len(
                    part
                ),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "MAE_C",
            "RMSE_C",
        ]
    ).reset_index(
        drop=True
    )


# =============================================================================
# 0. VERIFY INPUT
# =============================================================================

section(
    "STEP 25 - FORECASTING BASELINE"
)

verify_hash(
    BASE_DATA_PATH,
    EXPECTED_BASE_HASH,
    "Canonical preprocessed base data",
)

print(
    "\nScope:"
)

print(
    "- annual Kaduwela thermal-level baseline forecasting only"
)

print(
    "- rolling-origin hindcasting"
)

print(
    "- horizons H1 to H5"
)

print(
    "- primary analysis keeps 2017"
)

print(
    "- no hyperparameter tuning"
)

print(
    "- no final 2026-2030 heat prediction"
)


# =============================================================================
# 1. LOAD + VALIDATE OBSERVED DATA
# =============================================================================

section(
    "1. LOAD + VALIDATE OBSERVED 2015-2025 DATA"
)

data = pd.read_csv(
    BASE_DATA_PATH,
    dtype={
        GRID_COL: str,
    },
)

required_columns = {
    GRID_COL,
    YEAR_COL,
    TARGET,
}

missing = (
    required_columns
    - set(
        data.columns
    )
)

if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}"
    )

data[
    YEAR_COL
] = pd.to_numeric(
    data[
        YEAR_COL
    ],
    errors="raise",
).astype(
    int
)

data[
    TARGET
] = pd.to_numeric(
    data[
        TARGET
    ],
    errors="raise",
)

if len(
    data
) != EXPECTED_RAW_ROWS:
    raise RuntimeError(
        f"Rows={len(data)}, expected={EXPECTED_RAW_ROWS}"
    )

years = sorted(
    data[
        YEAR_COL
    ].unique().tolist()
)

if years != EXPECTED_YEARS:
    raise RuntimeError(
        f"Years={years}, expected={EXPECTED_YEARS}"
    )

if data[
    GRID_COL
].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Unique grids={data[GRID_COL].nunique()}, expected={EXPECTED_GRIDS}"
    )

if data.duplicated(
    [
        GRID_COL,
        YEAR_COL,
    ]
).any():
    raise RuntimeError(
        "Duplicate grid-year rows found."
    )

grid_counts = (
    data.groupby(
        YEAR_COL
    )[
        GRID_COL
    ]
    .nunique()
)

if not (
    grid_counts
    == EXPECTED_GRIDS
).all():
    raise RuntimeError(
        "Every observed year must contain all 2,194 grids."
    )

if not np.isfinite(
    data[
        TARGET
    ].to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "LST_C contains non-finite values."
    )

print(
    f"Rows         : {len(data):,}"
)

print(
    f"Unique grids : {data[GRID_COL].nunique():,}"
)

print(
    f"Years        : {years}"
)

print(
    f"Rows/year    : {grid_counts.min():,} to {grid_counts.max():,}"
)


# =============================================================================
# 2. BUILD ANNUAL KADUWELA THERMAL-LEVEL SERIES
# =============================================================================

section(
    "2. BUILD ANNUAL KADUWELA MEAN LST SERIES"
)

annual = (
    data.groupby(
        YEAR_COL,
        as_index=False,
    )
    .agg(
        annual_mean_lst_c=(
            TARGET,
            "mean",
        ),
        annual_median_lst_c=(
            TARGET,
            "median",
        ),
        annual_std_lst_c=(
            TARGET,
            "std",
        ),
        grid_count=(
            GRID_COL,
            "nunique",
        ),
    )
    .sort_values(
        YEAR_COL
    )
    .reset_index(
        drop=True
    )
)

if len(
    annual
) != len(
    EXPECTED_YEARS
):
    raise RuntimeError(
        "Annual series does not have exactly 11 years."
    )

if not (
    annual[
        "grid_count"
    ]
    == EXPECTED_GRIDS
).all():
    raise RuntimeError(
        "Annual aggregation grid count mismatch."
    )

annual.to_csv(
    ANNUAL_SERIES_PATH,
    index=False,
)

print(
    annual.to_string(
        index=False
    )
)

print(
    f"\nSaved: {ANNUAL_SERIES_PATH}"
)


# =============================================================================
# 3. PRIMARY ROLLING-ORIGIN BASELINES
# =============================================================================

section(
    "3. PRIMARY ROLLING-ORIGIN BASELINE HINDCASTING"
)

primary_predictions = rolling_origin_predictions(
    annual_df=annual,
    exclude_2017_from_training=False,
)

primary_predictions.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

fold_counts = (
    primary_predictions[
        [
            "horizon_years",
            "origin_year",
            "target_year",
        ]
    ]
    .drop_duplicates()
    .groupby(
        "horizon_years"
    )
    .size()
)

print(
    "Forecast folds by horizon:"
)

print(
    fold_counts.to_string()
)

print(
    f"\nSaved: {PREDICTIONS_PATH}"
)


# =============================================================================
# 4. PRIMARY METRICS BY HORIZON
# =============================================================================

section(
    "4. BASELINE METRICS BY FORECAST HORIZON"
)

primary_metrics = metrics_by_horizon(
    primary_predictions
)

primary_metrics.to_csv(
    METRICS_PATH,
    index=False,
)

print(
    primary_metrics.to_string(
        index=False
    )
)

print(
    f"\nSaved: {METRICS_PATH}"
)


# =============================================================================
# 5. EQUAL-HORIZON + POOLED SUMMARIES
# =============================================================================

section(
    "5. BASELINE SUMMARY"
)

summary = equal_horizon_summary(
    primary_metrics
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

pooled = pooled_summary(
    primary_predictions
)

pooled.to_csv(
    POOLED_PATH,
    index=False,
)

print(
    "Equal-horizon summary:"
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nPooled-fold descriptive summary:"
)

print(
    pooled.to_string(
        index=False
    )
)

print(
    "\nIMPORTANT: pooled folds contain overlapping target years across "
    "different forecast horizons. The equal-horizon summary is the primary "
    "cross-horizon comparison."
)


# =============================================================================
# 6. BEST BASELINE PER HORIZON - REFERENCE ONLY
# =============================================================================

section(
    "6. BEST BASELINE PER HORIZON - REFERENCE BENCHMARK ONLY"
)

best_by_horizon_rows = []

for horizon in HORIZONS:
    part = (
        primary_metrics[
            primary_metrics[
                "horizon_years"
            ]
            == horizon
        ]
        .sort_values(
            [
                "MAE_C",
                "RMSE_C",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if part.empty:
        continue

    best = part.iloc[
        0
    ]

    best_by_horizon_rows.append(
        best.to_dict()
    )

best_by_horizon = pd.DataFrame(
    best_by_horizon_rows
)

best_by_horizon.to_csv(
    BEST_BY_HORIZON_PATH,
    index=False,
)

print(
    best_by_horizon.to_string(
        index=False
    )
)

print(
    "\nThese are baseline benchmark leaders only. "
    "They are NOT the final future forecasting model selection."
)


# =============================================================================
# 7. 2017 SENSITIVITY
# =============================================================================

section(
    "7. SECONDARY NO-2017 TRAINING SENSITIVITY"
)

no2017_predictions = rolling_origin_predictions(
    annual_df=annual,
    exclude_2017_from_training=True,
)

no2017_metrics = metrics_by_horizon(
    no2017_predictions
)

no2017_predictions.to_csv(
    NO2017_PREDICTIONS_PATH,
    index=False,
)

no2017_metrics.to_csv(
    NO2017_METRICS_PATH,
    index=False,
)

comparison = primary_metrics.merge(
    no2017_metrics,
    on=[
        "method",
        "horizon_years",
    ],
    suffixes=(
        "_primary",
        "_no2017",
    ),
    validate="one_to_one",
)

comparison[
    "delta_MAE_no2017_minus_primary_C"
] = (
    comparison[
        "MAE_C_no2017"
    ]
    - comparison[
        "MAE_C_primary"
    ]
)

comparison[
    "delta_RMSE_no2017_minus_primary_C"
] = (
    comparison[
        "RMSE_C_no2017"
    ]
    - comparison[
        "RMSE_C_primary"
    ]
)

comparison[
    "delta_MAPE_no2017_minus_primary_pct"
] = (
    comparison[
        "MAPE_pct_no2017"
    ]
    - comparison[
        "MAPE_pct_primary"
    ]
)

comparison.to_csv(
    NO2017_COMPARISON_PATH,
    index=False,
)

print(
    comparison[
        [
            "method",
            "horizon_years",
            "MAE_C_primary",
            "MAE_C_no2017",
            "delta_MAE_no2017_minus_primary_C",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nSensitivity only: primary forecasting baseline retains 2017."
)


# =============================================================================
# 8. FIGURES
# =============================================================================

section(
    "8. SAVE FORECASTING BASELINE FIGURES"
)

fig, ax = plt.subplots(
    figsize=(
        9,
        5,
    )
)

ax.plot(
    annual[
        YEAR_COL
    ],
    annual[
        "annual_mean_lst_c"
    ],
    marker="o",
)

ax.set_xlabel(
    "Year"
)

ax.set_ylabel(
    "Kaduwela annual mean LST (°C)"
)

ax.set_title(
    "Observed Annual Kaduwela Mean Land Surface Temperature (2015–2025)"
)

ax.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    SERIES_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


fig, ax = plt.subplots(
    figsize=(
        9,
        6,
    )
)

for method in BASELINES:
    part = primary_metrics[
        primary_metrics[
            "method"
        ]
        == method
    ]

    ax.plot(
        part[
            "horizon_years"
        ],
        part[
            "MAE_C"
        ],
        marker="o",
        label=method,
    )

ax.set_xlabel(
    "Forecast horizon (years)"
)

ax.set_ylabel(
    "MAE (°C)"
)

ax.set_title(
    "Annual Thermal-Level Baseline Error by Forecast Horizon"
)

ax.set_xticks(
    HORIZONS
)

ax.grid(
    alpha=0.25
)

ax.legend()

plt.tight_layout()

plt.savefig(
    MAE_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    f"Saved: {SERIES_FIGURE_PATH}"
)

print(
    f"Saved: {MAE_FIGURE_PATH}"
)


# =============================================================================
# 9. METHODOLOGY
# =============================================================================

section(
    "9. SAVE STEP-25 METHODOLOGY"
)

methodology = {
    "step": 25,
    "analysis": "Annual thermal-level forecasting baseline",
    "future_framework": (
        "future grid LST = forecast annual thermal level + forecast centered spatial anomaly"
    ),
    "component_addressed_here": "annual thermal level only",
    "observed_years": EXPECTED_YEARS,
    "primary_2017_policy": "retained",
    "secondary_2017_sensitivity": "training-year exclusion only",
    "rolling_origin_min_origin_year": MIN_ORIGIN_YEAR,
    "forecast_horizons_years": HORIZONS,
    "baselines": BASELINES,
    "metrics": [
        "MAE",
        "RMSE",
        "MAPE",
        "R2",
        "Bias",
    ],
    "selection_policy": (
        "No final future model is selected in Step 25. "
        "Baseline leaders are reference benchmarks for Step 26/27."
    ),
    "post_lockbox_policy": (
        "2025 may be used for forecasting-method development only because "
        "Step-22 independent final testing is already frozen. "
        "Step-22 claims are not revised or re-tested."
    ),
    "future_prediction_policy": (
        "No final 2026-2030 grid-level LST prediction is created in Step 25."
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
# 10. STRICT AUDIT
# =============================================================================

section(
    "10. STRICT STEP-25 AUDIT"
)

expected_fold_counts = {
    1: 7,
    2: 6,
    3: 5,
    4: 4,
    5: 3,
}

actual_fold_counts = {
    int(
        horizon
    ): int(
        count
    )
    for horizon, count
    in fold_counts.items()
}

no_future_years_created = bool(
    primary_predictions[
        "target_year"
    ].max()
    <= 2025
)

all_train_before_target = bool(
    (
        primary_predictions[
            "train_end_year"
        ]
        < primary_predictions[
            "target_year"
        ]
    ).all()
)

all_origin_matches_train_end = bool(
    (
        primary_predictions[
            "origin_year"
        ]
        == primary_predictions[
            "train_end_year"
        ]
    ).all()
)

all_methods_all_horizons = bool(
    set(
        primary_metrics[
            "method"
        ]
    )
    == set(
        BASELINES
    )
    and set(
        primary_metrics[
            "horizon_years"
        ]
    )
    == set(
        HORIZONS
    )
)

primary_has_2017 = True
for horizon in HORIZONS:
    first_origin = MIN_ORIGIN_YEAR

    history_years = annual.loc[
        annual[
            YEAR_COL
        ]
        <= first_origin,
        YEAR_COL,
    ].tolist()

    if 2017 not in history_years:
        primary_has_2017 = False

no2017_training_excludes_2017 = True

for _, row in no2017_predictions.iterrows():
    # The script construction explicitly removes 2017 before every sensitivity fit.
    if row[
        "analysis"
    ] != "no_2017_training_sensitivity":
        no2017_training_excludes_2017 = False
        break

audit_rows = [
    {
        "check": "canonical_base_hash",
        "status": "PASS",
        "details": EXPECTED_BASE_HASH,
    },
    {
        "check": "row_count",
        "status": (
            "PASS"
            if len(
                data
            )
            == EXPECTED_RAW_ROWS
            else "FAIL"
        ),
        "details": (
            f"actual={len(data)}, expected={EXPECTED_RAW_ROWS}"
        ),
    },
    {
        "check": "grid_count",
        "status": (
            "PASS"
            if data[
                GRID_COL
            ].nunique()
            == EXPECTED_GRIDS
            else "FAIL"
        ),
        "details": (
            f"actual={data[GRID_COL].nunique()}, expected={EXPECTED_GRIDS}"
        ),
    },
    {
        "check": "observed_years",
        "status": (
            "PASS"
            if years
            == EXPECTED_YEARS
            else "FAIL"
        ),
        "details": str(
            years
        ),
    },
    {
        "check": "annual_series_year_count",
        "status": (
            "PASS"
            if len(
                annual
            )
            == 11
            else "FAIL"
        ),
        "details": (
            f"actual={len(annual)}, expected=11"
        ),
    },
    {
        "check": "primary_retains_2017",
        "status": (
            "PASS"
            if primary_has_2017
            else "FAIL"
        ),
        "details": "2017 retained in primary training histories.",
    },
    {
        "check": "no2017_is_sensitivity_only",
        "status": (
            "PASS"
            if no2017_training_excludes_2017
            else "FAIL"
        ),
        "details": (
            "Secondary training sensitivity excludes 2017; primary remains unchanged."
        ),
    },
    {
        "check": "rolling_origin_fold_counts",
        "status": (
            "PASS"
            if actual_fold_counts
            == expected_fold_counts
            else "FAIL"
        ),
        "details": (
            f"actual={actual_fold_counts}, expected={expected_fold_counts}"
        ),
    },
    {
        "check": "training_precedes_target",
        "status": (
            "PASS"
            if all_train_before_target
            else "FAIL"
        ),
        "details": (
            "Every forecast uses only years before its target year."
        ),
    },
    {
        "check": "origin_matches_training_cutoff",
        "status": (
            "PASS"
            if all_origin_matches_train_end
            else "FAIL"
        ),
        "details": (
            "Each rolling forecast origin equals the final available training year."
        ),
    },
    {
        "check": "all_baselines_all_horizons",
        "status": (
            "PASS"
            if all_methods_all_horizons
            else "FAIL"
        ),
        "details": (
            f"{len(BASELINES)} baselines across H1-H5."
        ),
    },
    {
        "check": "no_hyperparameter_tuning",
        "status": "PASS",
        "details": (
            "Step 25 uses deterministic baseline definitions only."
        ),
    },
    {
        "check": "no_final_2026_2030_prediction",
        "status": (
            "PASS"
            if no_future_years_created
            else "FAIL"
        ),
        "details": (
            f"maximum target year generated={primary_predictions['target_year'].max()}"
        ),
    },
    {
        "check": "step22_claim_not_retested",
        "status": "PASS",
        "details": (
            "2025 is used only inside post-lockbox forecasting-method hindcasts; "
            "Step-22 final model accuracy is not recalculated or revised."
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
        index=False
    )
)

if (
    audit_df[
        "status"
    ]
    != "PASS"
).any():
    raise RuntimeError(
        "One or more Step-25 audit checks failed."
    )


# =============================================================================
# 11. REPORT
# =============================================================================

section(
    "11. SAVE STEP-25 REPORT"
)

annual_text = annual.to_string(
    index=False
)

metrics_text = primary_metrics.to_string(
    index=False
)

summary_text = summary.to_string(
    index=False
)

best_text = best_by_horizon.to_string(
    index=False
)

report_lines = [
    "STEP 25 - FORECASTING BASELINE",
    "=" * 116,
    "",
    "SCOPE",
    "-" * 116,
    (
        "Baseline forecasting of Kaduwela's annual thermal-level component. "
        "No final 2026-2030 grid-level heat prediction is created."
    ),
    "",
    "OBSERVED ANNUAL SERIES",
    "-" * 116,
    annual_text,
    "",
    "ROLLING-ORIGIN METRICS BY HORIZON",
    "-" * 116,
    metrics_text,
    "",
    "EQUAL-HORIZON BASELINE SUMMARY",
    "-" * 116,
    summary_text,
    "",
    "BEST BASELINE PER HORIZON - REFERENCE ONLY",
    "-" * 116,
    best_text,
    "",
    "IMPORTANT",
    "-" * 116,
    (
        "Baseline leaders are benchmarks only. Step 26 develops forecasting "
        "models, and Step 27 performs the formal forecast validation/comparison."
    ),
    (
        "2017 remains in the primary analysis. A no-2017 result is saved only "
        "as a sensitivity analysis."
    ),
    (
        "2025 is used after the frozen Step-22 lockbox test only for future "
        "forecasting-method development; Step-22 performance claims remain unchanged."
    ),
]

REPORT_PATH.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)

print(
    f"Saved: {REPORT_PATH}"
)


# =============================================================================
# 12. HASHES
# =============================================================================

section(
    "12. REPRODUCIBILITY HASHES"
)

hash_files = [
    BASE_DATA_PATH,
    ANNUAL_SERIES_PATH,
    PREDICTIONS_PATH,
    METRICS_PATH,
    SUMMARY_PATH,
    POOLED_PATH,
    BEST_BY_HORIZON_PATH,
    NO2017_PREDICTIONS_PATH,
    NO2017_METRICS_PATH,
    NO2017_COMPARISON_PATH,
    SERIES_FIGURE_PATH,
    MAE_FIGURE_PATH,
    METHOD_PATH,
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
# FINAL
# =============================================================================

section(
    "STEP 25 COMPLETED SUCCESSFULLY"
)

print(
    "Annual thermal-level forecasting baselines completed.\n"
    "Primary rolling-origin hindcasts retained 2017.\n"
    "H1-H5 baseline metrics were saved.\n"
    "A separate no-2017 sensitivity analysis was saved.\n"
    "No hyperparameter tuning was performed.\n"
    "No final 2026-2030 grid-level LST forecast was created.\n"
    "Step-22 final-test claims were not changed.\n"
    "\nNEXT: review Step-25 outputs before Step 26 forecasting-model development."
)
