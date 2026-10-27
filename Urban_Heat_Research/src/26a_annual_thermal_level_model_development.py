"""
STEP 26A - ANNUAL THERMAL-LEVEL FORECAST MODEL DEVELOPMENT
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Develop and freeze a small set of scientifically reasonable candidate models
for forecasting Kaduwela's ANNUAL THERMAL LEVEL.

Final future-LST framework:
    Future grid LST
      = forecast Kaduwela annual thermal level
      + forecast centered spatial anomaly

Step 26A handles ONLY the annual thermal-level branch.

IMPORTANT PROTOCOL
------------------
Step 25 already characterized simple baselines using historical rolling-origin
hindcasts. Step 26A now develops candidate annual-level models without using
2024-2025 as candidate-development targets.

Development target years:
    <= 2023

Reserved forecast-validation target years:
    2024 and 2025

All Step-26A candidate definitions are carried forward unchanged to Step 27.
Step 27 will evaluate the frozen candidates on the reserved 2024-2025 targets.

This avoids choosing/tuning a candidate using the same target years that will
later be used for forecast validation.

OBSERVED SERIES
---------------
Canonical annual Kaduwela mean LST:
    2015-2025

2017 remains in the PRIMARY analysis.
A separate no-2017 sensitivity analysis is retained for robustness only.

CANDIDATE MODELS
----------------
1. Simple Exponential Smoothing (SES)
   - level-only forecasting
   - smoothing parameter estimated from training history at each origin

2. Damped Holt Trend
   - local level + damped trend
   - parameters estimated from training history at each origin

3. Robust Theil-Sen Trend
   - non-parametric robust linear slope on available annual history
   - extrapolated to the target year

These are intentionally simple because the annual series has only 11 observed
years. Deep learning / high-parameter time-series models are not justified for
this branch.

STEP-25 BASELINES
-----------------
The exact Step-25 rolling-origin baseline predictions are loaded and filtered to
the SAME Step-26A development target years for fair reference comparison.

No final annual-level model is selected in Step 26A.
No 2026-2030 forecast is generated here.
No Step-22B spatial-anomaly model is retrained or changed.
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
        "\nstatsmodels is required for Step 26A.\n"
        "Install it once with:\n"
        "    pip install statsmodels\n"
        "Then rerun:\n"
        "    python src/26a_annual_thermal_level_model_development.py\n"
    ) from exc

try:
    from scipy.stats import theilslopes
except ImportError as exc:
    raise ImportError(
        "\nscipy is required for Step 26A.\n"
        "Install it once with:\n"
        "    pip install scipy\n"
        "Then rerun:\n"
        "    python src/26a_annual_thermal_level_model_development.py\n"
    ) from exc


# =============================================================================
# SETTINGS
# =============================================================================

YEAR_COL = "year"
VALUE_COL = "annual_mean_lst_c"

EXPECTED_YEARS = list(range(2015, 2026))

DEV_MAX_TARGET_YEAR = 2023
RESERVED_VALIDATION_YEARS = [2024, 2025]

MIN_ORIGIN_YEAR = 2018
HORIZONS = [1, 2, 3, 4, 5]

CANDIDATES = [
    "Simple Exponential Smoothing",
    "Damped Holt Trend",
    "Theil-Sen Robust Trend",
]

STEP25_BASELINES = [
    "Persistence",
    "Expanding Mean",
    "Rolling-3 Mean",
    "Drift",
    "OLS Linear Trend",
]

EXPECTED_HASHES = {
    "25_annual_kaduwela_mean_lst_2015_2025.csv":
        "910fa5268bedd3fe50ff4abc0db82f48a345d1fe6cf79ffe1849be24cc004869",
    "25_annual_level_baseline_rolling_origin_predictions.csv":
        "319bf955d3c555e7b1c24111746c8c10de89b55e218241dcb22c1d7caa853007",
    "25_forecasting_baseline_audit.csv":
        "aef89269cb4accd889947a5daa117bf74c4116646a95d3881187959f06084df3",
}


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

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

STEP25_ANNUAL_PATH = (
    FORECAST_DIR
    / "25_annual_kaduwela_mean_lst_2015_2025.csv"
)

STEP25_PREDICTIONS_PATH = (
    FORECAST_DIR
    / "25_annual_level_baseline_rolling_origin_predictions.csv"
)

STEP25_AUDIT_PATH = (
    REPORT_DIR
    / "25_forecasting_baseline_audit.csv"
)

CANDIDATE_PREDICTIONS_PATH = (
    FORECAST_DIR
    / "26a_candidate_predictions_dev_targets_to_2023.csv"
)

CANDIDATE_METRICS_PATH = (
    FORECAST_DIR
    / "26a_candidate_metrics_by_horizon_dev.csv"
)

CANDIDATE_SUMMARY_PATH = (
    FORECAST_DIR
    / "26a_candidate_equal_horizon_summary_dev.csv"
)

BASELINE_DEV_METRICS_PATH = (
    FORECAST_DIR
    / "26a_step25_baseline_metrics_same_dev_window.csv"
)

COMBINED_DEV_METRICS_PATH = (
    FORECAST_DIR
    / "26a_candidates_vs_baselines_dev_metrics.csv"
)

COMBINED_DEV_SUMMARY_PATH = (
    FORECAST_DIR
    / "26a_candidates_vs_baselines_equal_horizon_summary_dev.csv"
)

NO2017_PREDICTIONS_PATH = (
    FORECAST_DIR
    / "26a_no2017_candidate_predictions_dev.csv"
)

NO2017_METRICS_PATH = (
    FORECAST_DIR
    / "26a_no2017_candidate_metrics_by_horizon_dev.csv"
)

NO2017_COMPARISON_PATH = (
    FORECAST_DIR
    / "26a_no2017_candidate_sensitivity_comparison.csv"
)

DEVELOPMENT_FIGURE_PATH = (
    FIGURE_DIR
    / "26a_annual_level_candidate_mae_dev.png"
)

SERIES_SPLIT_FIGURE_PATH = (
    FIGURE_DIR
    / "26a_annual_series_dev_reserved_split.png"
)

FREEZE_PATH = (
    REPORT_DIR
    / "26a_candidate_model_freeze.json"
)

METHOD_PATH = (
    REPORT_DIR
    / "26a_annual_level_methodology.json"
)

AUDIT_PATH = (
    REPORT_DIR
    / "26a_annual_level_audit.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "26a_annual_level_report.txt"
)

HASH_PATH = (
    REPORT_DIR
    / "26a_annual_level_hashes.csv"
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


def require_all_pass(path, label):
    audit = pd.read_csv(
        path
    )

    if "status" not in audit.columns:
        raise RuntimeError(
            f"{label} audit has no status column."
        )

    failed = audit[
        audit[
            "status"
        ]
        .astype(str)
        .str.upper()
        != "PASS"
    ]

    if not failed.empty:
        print(
            failed.to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"{label} audit is not fully PASS."
        )


def regression_metrics(actual, predicted):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    errors = (
        predicted
        - actual
    )

    mae = float(
        np.mean(
            np.abs(
                errors
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    if np.any(
        actual
        == 0
    ):
        mape = np.nan
    else:
        mape = float(
            np.mean(
                np.abs(
                    errors
                    / actual
                )
            )
            * 100.0
        )

    bias = float(
        np.mean(
            errors
        )
    )

    if len(
        actual
    ) < 2:
        r2 = np.nan
    else:
        ss_res = float(
            np.sum(
                (
                    actual
                    - predicted
                ) ** 2
            )
        )

        ss_tot = float(
            np.sum(
                (
                    actual
                    - np.mean(
                        actual
                    )
                ) ** 2
            )
        )

        if ss_tot == 0:
            r2 = np.nan
        else:
            r2 = float(
                1.0
                - ss_res
                / ss_tot
            )

    return {
        "MAE_C": mae,
        "RMSE_C": rmse,
        "MAPE_pct": mape,
        "R2": r2,
        "Bias_C": bias,
    }


def forecast_candidate(
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

    if len(
        years
    ) != len(
        values
    ):
        raise ValueError(
            "Training years and values have different lengths."
        )

    if len(
        values
    ) < 4:
        raise ValueError(
            "Step 26A candidates require at least four annual observations."
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
            "Target year must be later than forecast origin."
        )

    if method == "Simple Exponential Smoothing":
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )

            fitted = SimpleExpSmoothing(
                values,
                initialization_method="estimated",
            ).fit(
                optimized=True,
                remove_bias=False,
            )

        forecast = fitted.forecast(
            horizon
        )

        return float(
            forecast[
                -1
            ]
        )

    if method == "Damped Holt Trend":
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )

            fitted = Holt(
                values,
                damped_trend=True,
                initialization_method="estimated",
            ).fit(
                optimized=True,
                remove_bias=False,
            )

        forecast = fitted.forecast(
            horizon
        )

        return float(
            forecast[
                -1
            ]
        )

    if method == "Theil-Sen Robust Trend":
        slope, intercept, _, _ = theilslopes(
            values,
            years,
            alpha=0.95,
            method="separate",
        )

        return float(
            intercept
            + slope
            * target_year
        )

    raise ValueError(
        f"Unknown candidate method: {method}"
    )


def build_candidate_predictions(
    annual_df,
    exclude_2017=False,
):
    annual_lookup = (
        annual_df
        .set_index(
            YEAR_COL
        )[
            VALUE_COL
        ]
        .to_dict()
    )

    rows = []

    for horizon in HORIZONS:
        for origin_year in range(
            MIN_ORIGIN_YEAR,
            DEV_MAX_TARGET_YEAR,
        ):
            target_year = (
                origin_year
                + horizon
            )

            if target_year > DEV_MAX_TARGET_YEAR:
                continue

            history = annual_df[
                annual_df[
                    YEAR_COL
                ]
                <= origin_year
            ].copy()

            if exclude_2017:
                history = history[
                    history[
                        YEAR_COL
                    ]
                    != 2017
                ].copy()

            # All Step-26A candidates require at least four annual observations.
            # In the no-2017 sensitivity analysis the earliest origin (2018)
            # has only 2015, 2016 and 2018 after removing 2017, so that fold is
            # intentionally skipped rather than weakening the minimum-history rule.
            if len(
                history
            ) < 4:
                continue

            train_years = (
                history[
                    YEAR_COL
                ]
                .to_numpy()
            )

            train_values = (
                history[
                    VALUE_COL
                ]
                .to_numpy()
            )

            actual = float(
                annual_lookup[
                    target_year
                ]
            )

            for method in CANDIDATES:
                prediction = forecast_candidate(
                    train_years=train_years,
                    train_values=train_values,
                    target_year=target_year,
                    method=method,
                )

                rows.append(
                    {
                        "analysis": (
                            "no_2017_training_sensitivity"
                            if exclude_2017
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
                        "predicted_annual_mean_lst_c": prediction,
                        "error_c": float(
                            prediction
                            - actual
                        ),
                        "absolute_error_c": float(
                            abs(
                                prediction
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
            "No Step-26A candidate forecasts were generated."
        )

    return result


def metrics_by_horizon(
    predictions,
    methods,
):
    rows = []

    for method in methods:
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
                    "fold_count": int(
                        len(
                            part
                        )
                    ),
                    **metrics,
                }
            )

    return pd.DataFrame(
        rows
    )


def equal_horizon_summary(
    metrics_df,
):
    rows = []

    for method in metrics_df[
        "method"
    ].drop_duplicates():
        part = metrics_df[
            metrics_df[
                "method"
            ]
            == method
        ].copy()

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
                    ].mean(
                        skipna=True
                    )
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


# =============================================================================
# 0. VERIFY STEP-25 INPUTS
# =============================================================================

section(
    "STEP 26A - ANNUAL THERMAL-LEVEL FORECAST MODEL DEVELOPMENT"
)

verify_hash(
    STEP25_ANNUAL_PATH,
    EXPECTED_HASHES[
        "25_annual_kaduwela_mean_lst_2015_2025.csv"
    ],
    "Step-25 annual Kaduwela mean-LST series",
)

verify_hash(
    STEP25_PREDICTIONS_PATH,
    EXPECTED_HASHES[
        "25_annual_level_baseline_rolling_origin_predictions.csv"
    ],
    "Step-25 rolling-origin baseline predictions",
)

verify_hash(
    STEP25_AUDIT_PATH,
    EXPECTED_HASHES[
        "25_forecasting_baseline_audit.csv"
    ],
    "Step-25 strict audit",
)

require_all_pass(
    STEP25_AUDIT_PATH,
    "Step 25",
)

print(
    "\nPASS: Step-25 audit is fully PASS."
)

print(
    "\nDevelopment protocol:"
)

print(
    "- candidate-development target years end at 2023"
)

print(
    "- 2024 and 2025 are reserved for Step-27 forecast validation"
)

print(
    "- all three candidate methods will be carried to Step 27"
)

print(
    "- 2017 retained in primary analysis"
)

print(
    "- no 2026-2030 forecast generated"
)

print(
    "- Step-22B LightGBM is untouched"
)


# =============================================================================
# 1. LOAD + VALIDATE ANNUAL SERIES
# =============================================================================

section(
    "1. LOAD + VALIDATE ANNUAL THERMAL-LEVEL SERIES"
)

annual = pd.read_csv(
    STEP25_ANNUAL_PATH
)

required_columns = {
    YEAR_COL,
    VALUE_COL,
    "grid_count",
}

missing = (
    required_columns
    - set(
        annual.columns
    )
)

if missing:
    raise RuntimeError(
        f"Step-25 annual series missing columns: {sorted(missing)}"
    )

annual[
    YEAR_COL
] = pd.to_numeric(
    annual[
        YEAR_COL
    ],
    errors="raise",
).astype(
    int
)

annual[
    VALUE_COL
] = pd.to_numeric(
    annual[
        VALUE_COL
    ],
    errors="raise",
)

years = sorted(
    annual[
        YEAR_COL
    ].tolist()
)

if years != EXPECTED_YEARS:
    raise RuntimeError(
        f"Annual years={years}, expected={EXPECTED_YEARS}"
    )

if not np.isfinite(
    annual[
        VALUE_COL
    ].to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "Annual mean LST contains non-finite values."
    )

if not (
    annual[
        "grid_count"
    ]
    == 2194
).all():
    raise RuntimeError(
        "Annual series does not contain 2,194 grids in every year."
    )

print(
    annual.to_string(
        index=False
    )
)

print(
    "\nDevelopment targets: <= 2023"
)

print(
    "Reserved validation targets: 2024, 2025"
)


# =============================================================================
# 2. PRIMARY CANDIDATE ROLLING-ORIGIN DEVELOPMENT
# =============================================================================

section(
    "2. PRIMARY CANDIDATE ROLLING-ORIGIN DEVELOPMENT"
)

candidate_predictions = build_candidate_predictions(
    annual_df=annual,
    exclude_2017=False,
)

candidate_predictions.to_csv(
    CANDIDATE_PREDICTIONS_PATH,
    index=False,
)

candidate_fold_counts = (
    candidate_predictions[
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
    "Development folds by horizon:"
)

print(
    candidate_fold_counts.to_string()
)

print(
    f"\nSaved: {CANDIDATE_PREDICTIONS_PATH}"
)


# =============================================================================
# 3. CANDIDATE DEVELOPMENT METRICS
# =============================================================================

section(
    "3. CANDIDATE DEVELOPMENT METRICS"
)

candidate_metrics = metrics_by_horizon(
    predictions=candidate_predictions,
    methods=CANDIDATES,
)

candidate_metrics.to_csv(
    CANDIDATE_METRICS_PATH,
    index=False,
)

candidate_summary = equal_horizon_summary(
    candidate_metrics
)

candidate_summary.to_csv(
    CANDIDATE_SUMMARY_PATH,
    index=False,
)

print(
    "Candidate metrics by horizon:"
)

print(
    candidate_metrics.to_string(
        index=False
    )
)

print(
    "\nEqual-horizon candidate development summary:"
)

print(
    candidate_summary.to_string(
        index=False
    )
)

print(
    "\nNOTE: H5 development has only one target year, so H5 R² is undefined."
)


# =============================================================================
# 4. SAME-WINDOW STEP-25 BASELINE REFERENCE
# =============================================================================

section(
    "4. STEP-25 BASELINES FILTERED TO SAME DEVELOPMENT WINDOW"
)

step25_predictions = pd.read_csv(
    STEP25_PREDICTIONS_PATH
)

baseline_dev_predictions = step25_predictions[
    step25_predictions[
        "target_year"
    ]
    <= DEV_MAX_TARGET_YEAR
].copy()

if baseline_dev_predictions.empty:
    raise RuntimeError(
        "No Step-25 baseline predictions remain after development-window filtering."
    )

baseline_dev_metrics = metrics_by_horizon(
    predictions=baseline_dev_predictions,
    methods=STEP25_BASELINES,
)

baseline_dev_metrics.to_csv(
    BASELINE_DEV_METRICS_PATH,
    index=False,
)

print(
    baseline_dev_metrics.to_string(
        index=False
    )
)

print(
    f"\nSaved: {BASELINE_DEV_METRICS_PATH}"
)


# =============================================================================
# 5. CANDIDATES VS BASELINES ON DEVELOPMENT TARGETS
# =============================================================================

section(
    "5. CANDIDATES VS BASELINES - DEVELOPMENT REFERENCE ONLY"
)

candidate_metrics_tagged = candidate_metrics.copy()
candidate_metrics_tagged.insert(
    0,
    "model_group",
    "Step26A candidate",
)

baseline_metrics_tagged = baseline_dev_metrics.copy()
baseline_metrics_tagged.insert(
    0,
    "model_group",
    "Step25 baseline",
)

combined_metrics = pd.concat(
    [
        candidate_metrics_tagged,
        baseline_metrics_tagged,
    ],
    ignore_index=True,
)

combined_metrics.to_csv(
    COMBINED_DEV_METRICS_PATH,
    index=False,
)

combined_summary = equal_horizon_summary(
    combined_metrics
)

group_lookup = (
    combined_metrics[
        [
            "method",
            "model_group",
        ]
    ]
    .drop_duplicates()
    .set_index(
        "method"
    )[
        "model_group"
    ]
    .to_dict()
)

combined_summary.insert(
    0,
    "model_group",
    combined_summary[
        "method"
    ].map(
        group_lookup
    ),
)

combined_summary.to_csv(
    COMBINED_DEV_SUMMARY_PATH,
    index=False,
)

print(
    combined_summary.to_string(
        index=False
    )
)

print(
    "\nIMPORTANT: This is development evidence only."
)

print(
    "All Step-26A candidates remain frozen and move forward to Step 27."
)


# =============================================================================
# 6. NO-2017 SENSITIVITY
# =============================================================================

section(
    "6. SECONDARY NO-2017 CANDIDATE SENSITIVITY"
)

no2017_predictions = build_candidate_predictions(
    annual_df=annual,
    exclude_2017=True,
)

no2017_predictions.to_csv(
    NO2017_PREDICTIONS_PATH,
    index=False,
)

no2017_metrics = metrics_by_horizon(
    predictions=no2017_predictions,
    methods=CANDIDATES,
)

no2017_metrics.to_csv(
    NO2017_METRICS_PATH,
    index=False,
)

# Fair sensitivity comparison must use the exact same forecast folds in the
# primary and no-2017 analyses. The no-2017 series cannot support origin 2018
# under the fixed minimum-history rule, so those primary folds are excluded
# from this secondary comparison only.
common_fold_keys = (
    no2017_predictions[
        [
            "method",
            "horizon_years",
            "origin_year",
            "target_year",
        ]
    ]
    .drop_duplicates()
)

primary_common_predictions = candidate_predictions.merge(
    common_fold_keys,
    on=[
        "method",
        "horizon_years",
        "origin_year",
        "target_year",
    ],
    how="inner",
    validate="one_to_one",
)

primary_common_metrics = metrics_by_horizon(
    predictions=primary_common_predictions,
    methods=CANDIDATES,
)

sensitivity = primary_common_metrics.merge(
    no2017_metrics,
    on=[
        "method",
        "horizon_years",
    ],
    suffixes=(
        "_primary_common_folds",
        "_no2017",
    ),
    validate="one_to_one",
)

sensitivity[
    "delta_MAE_no2017_minus_primary_C"
] = (
    sensitivity[
        "MAE_C_no2017"
    ]
    - sensitivity[
        "MAE_C_primary_common_folds"
    ]
)

sensitivity[
    "delta_RMSE_no2017_minus_primary_C"
] = (
    sensitivity[
        "RMSE_C_no2017"
    ]
    - sensitivity[
        "RMSE_C_primary_common_folds"
    ]
)

sensitivity.to_csv(
    NO2017_COMPARISON_PATH,
    index=False,
)

print(
    "No-2017 sensitivity fold counts:"
)

print(
    no2017_predictions[
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
    .to_string()
)

print(
    "\nCommon-fold sensitivity comparison:"
)

print(
    sensitivity[
        [
            "method",
            "horizon_years",
            "fold_count_primary_common_folds",
            "fold_count_no2017",
            "MAE_C_primary_common_folds",
            "MAE_C_no2017",
            "delta_MAE_no2017_minus_primary_C",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nSensitivity only. Primary candidate development retains 2017."
)

print(
    "H5 has no valid no-2017 development fold under the fixed minimum-history "
    "rule, so no H5 no-2017 sensitivity claim is made."
)


# =============================================================================
# 7. FREEZE CANDIDATE DEFINITIONS FOR STEP 27
# =============================================================================

section(
    "7. FREEZE CANDIDATE DEFINITIONS FOR STEP 27"
)

freeze = {
    "step": "26A",
    "purpose": "annual thermal-level candidate development",
    "development_target_year_max": DEV_MAX_TARGET_YEAR,
    "reserved_validation_target_years": RESERVED_VALIDATION_YEARS,
    "forecast_horizons_years": HORIZONS,
    "primary_2017_policy": "retained",
    "candidates": {
        "Simple Exponential Smoothing": {
            "library": "statsmodels",
            "class": "SimpleExpSmoothing",
            "initialization_method": "estimated",
            "fit_optimized": True,
            "remove_bias": False,
        },
        "Damped Holt Trend": {
            "library": "statsmodels",
            "class": "Holt",
            "damped_trend": True,
            "initialization_method": "estimated",
            "fit_optimized": True,
            "remove_bias": False,
        },
        "Theil-Sen Robust Trend": {
            "library": "scipy",
            "function": "scipy.stats.theilslopes",
            "alpha": 0.95,
            "method": "separate",
        },
    },
    "selection_policy": (
        "No final annual-level model selected in Step 26A. "
        "All three frozen candidates proceed to Step 27 reserved validation."
    ),
    "step27_rule": (
        "Candidate definitions/configurations must not be changed before "
        "evaluating reserved 2024-2025 targets."
    ),
    "step22b_spatial_model_changed": False,
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
# 8. FIGURES
# =============================================================================

section(
    "8. SAVE STEP-26A FIGURES"
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
        VALUE_COL
    ],
    marker="o",
)

ax.axvline(
    2023.5,
    linestyle="--",
)

ax.set_xlabel(
    "Year"
)

ax.set_ylabel(
    "Kaduwela annual mean LST (°C)"
)

ax.set_title(
    "Annual Thermal-Level Series: Development vs Reserved Validation"
)

ax.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    SERIES_SPLIT_FIGURE_PATH,
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

for method in CANDIDATES:
    part = candidate_metrics[
        candidate_metrics[
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
    "Development MAE (°C)"
)

ax.set_title(
    "Step 26A Annual Thermal-Level Candidate Error"
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
    DEVELOPMENT_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    f"Saved: {SERIES_SPLIT_FIGURE_PATH}"
)

print(
    f"Saved: {DEVELOPMENT_FIGURE_PATH}"
)


# =============================================================================
# 9. METHODOLOGY
# =============================================================================

section(
    "9. SAVE STEP-26A METHODOLOGY"
)

methodology = {
    "step": "26A",
    "analysis": "Annual thermal-level forecast model development",
    "observed_series_years": EXPECTED_YEARS,
    "development_targets": "2019-2023 depending on horizon",
    "reserved_validation_target_years": RESERVED_VALIDATION_YEARS,
    "minimum_origin_year": MIN_ORIGIN_YEAR,
    "horizons": HORIZONS,
    "candidate_models": CANDIDATES,
    "primary_2017_policy": "retained",
    "no2017_policy": (
        "secondary sensitivity only; folds with fewer than four observations "
        "after removing 2017 are skipped, and comparisons use identical common folds"
    ),
    "baseline_reference": (
        "Exact Step-25 rolling-origin predictions filtered to the same "
        "development target years."
    ),
    "parameter_estimation_note": (
        "SES and damped Holt estimate smoothing parameters using only the "
        "training history available at each rolling origin. This is ordinary "
        "within-fit parameter estimation, not a cross-origin hyperparameter search."
    ),
    "selection_policy": (
        "No final model selected in Step 26A; all candidates proceed unchanged "
        "to Step 27 reserved 2024-2025 forecast validation."
    ),
    "future_prediction_policy": (
        "No 2026-2030 annual-level forecast is produced in Step 26A."
    ),
    "spatial_model_policy": (
        "The Step-22B LightGBM spatial-anomaly production model is not loaded, "
        "retrained, or modified."
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
    "10. STRICT STEP-26A AUDIT"
)

expected_dev_fold_counts = {
    1: 5,
    2: 4,
    3: 3,
    4: 2,
    5: 1,
}

actual_dev_fold_counts = {
    int(
        horizon
    ): int(
        count
    )
    for horizon, count
    in candidate_fold_counts.items()
}

candidate_targets_reserved_clean = bool(
    candidate_predictions[
        "target_year"
    ].max()
    <= DEV_MAX_TARGET_YEAR
)

candidate_train_precedes_target = bool(
    (
        candidate_predictions[
            "train_end_year"
        ]
        < candidate_predictions[
            "target_year"
        ]
    ).all()
)

origin_matches_train_cutoff = bool(
    (
        candidate_predictions[
            "origin_year"
        ]
        == candidate_predictions[
            "train_end_year"
        ]
    ).all()
)

all_candidates_all_horizons = bool(
    set(
        candidate_metrics[
            "method"
        ]
    )
    == set(
        CANDIDATES
    )
    and set(
        candidate_metrics[
            "horizon_years"
        ]
    )
    == set(
        HORIZONS
    )
)

baseline_targets_reserved_clean = bool(
    baseline_dev_predictions[
        "target_year"
    ].max()
    <= DEV_MAX_TARGET_YEAR
)

reserved_targets_absent = bool(
    not candidate_predictions[
        "target_year"
    ].isin(
        RESERVED_VALIDATION_YEARS
    ).any()
)

primary_uses_2017 = bool(
    2017
    in annual.loc[
        annual[
            YEAR_COL
        ]
        <= MIN_ORIGIN_YEAR,
        YEAR_COL,
    ].tolist()
)

audit_rows = [
    {
        "check": "step25_audit",
        "status": "PASS",
        "details": "Step-25 strict audit verified fully PASS.",
    },
    {
        "check": "step25_annual_series_hash",
        "status": "PASS",
        "details": EXPECTED_HASHES[
            "25_annual_kaduwela_mean_lst_2015_2025.csv"
        ],
    },
    {
        "check": "step25_prediction_hash",
        "status": "PASS",
        "details": EXPECTED_HASHES[
            "25_annual_level_baseline_rolling_origin_predictions.csv"
        ],
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
        "check": "development_fold_counts",
        "status": (
            "PASS"
            if actual_dev_fold_counts
            == expected_dev_fold_counts
            else "FAIL"
        ),
        "details": (
            f"actual={actual_dev_fold_counts}, "
            f"expected={expected_dev_fold_counts}"
        ),
    },
    {
        "check": "candidate_targets_stop_at_2023",
        "status": (
            "PASS"
            if candidate_targets_reserved_clean
            else "FAIL"
        ),
        "details": (
            f"max_candidate_target={candidate_predictions['target_year'].max()}"
        ),
    },
    {
        "check": "reserved_2024_2025_not_used_as_candidate_targets",
        "status": (
            "PASS"
            if reserved_targets_absent
            else "FAIL"
        ),
        "details": (
            "2024 and 2025 are reserved for Step-27 validation."
        ),
    },
    {
        "check": "candidate_training_precedes_target",
        "status": (
            "PASS"
            if candidate_train_precedes_target
            else "FAIL"
        ),
        "details": (
            "All candidate fits use only years available at forecast origin."
        ),
    },
    {
        "check": "origin_matches_training_cutoff",
        "status": (
            "PASS"
            if origin_matches_train_cutoff
            else "FAIL"
        ),
        "details": (
            "Every candidate origin equals its training-history cutoff."
        ),
    },
    {
        "check": "all_candidates_all_horizons",
        "status": (
            "PASS"
            if all_candidates_all_horizons
            else "FAIL"
        ),
        "details": (
            f"{len(CANDIDATES)} candidates across H1-H5."
        ),
    },
    {
        "check": "step25_baselines_same_dev_window",
        "status": (
            "PASS"
            if baseline_targets_reserved_clean
            else "FAIL"
        ),
        "details": (
            "Step-25 reference baselines filtered to target years <= 2023."
        ),
    },
    {
        "check": "primary_retains_2017",
        "status": (
            "PASS"
            if primary_uses_2017
            else "FAIL"
        ),
        "details": (
            "2017 remains in primary annual-level development history."
        ),
    },
    {
        "check": "no2017_minimum_history_preserved",
        "status": (
            "PASS"
            if (
                no2017_predictions[
                    "training_year_count"
                ].min()
                >= 4
            )
            else "FAIL"
        ),
        "details": (
            "No-2017 sensitivity keeps the same minimum of four annual observations."
        ),
    },
    {
        "check": "no2017_common_fold_comparison",
        "status": (
            "PASS"
            if (
                not sensitivity.empty
                and (
                    sensitivity[
                        "fold_count_primary_common_folds"
                    ]
                    == sensitivity[
                        "fold_count_no2017"
                    ]
                ).all()
            )
            else "FAIL"
        ),
        "details": (
            "Primary vs no-2017 sensitivity metrics use identical origin-target folds."
        ),
    },
    {
        "check": "no_hyperparameter_grid_search",
        "status": "PASS",
        "details": (
            "No candidate hyperparameter grid/random/Bayesian search performed."
        ),
    },
    {
        "check": "candidate_definitions_frozen_for_step27",
        "status": (
            "PASS"
            if FREEZE_PATH.exists()
            else "FAIL"
        ),
        "details": (
            "All three candidate definitions saved before Step-27 reserved validation."
        ),
    },
    {
        "check": "no_2026_2030_forecast",
        "status": (
            "PASS"
            if candidate_predictions[
                "target_year"
            ].max()
            <= 2023
            else "FAIL"
        ),
        "details": (
            "No future operational annual-level forecast generated."
        ),
    },
    {
        "check": "step22b_spatial_model_untouched",
        "status": "PASS",
        "details": (
            "Step-22B LightGBM is not loaded/retrained/modified in Step 26A."
        ),
    },
    {
        "check": "dynamic_predictor_forecasting_not_started",
        "status": "PASS",
        "details": (
            "NDVI/NDBI/NDWI/EVI/etc. belong to Step 26B, not Step 26A."
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
        "One or more Step-26A audit checks failed."
    )


# =============================================================================
# 11. REPORT
# =============================================================================

section(
    "11. SAVE STEP-26A REPORT"
)

report_lines = [
    "STEP 26A - ANNUAL THERMAL-LEVEL FORECAST MODEL DEVELOPMENT",
    "=" * 116,
    "",
    "PROTOCOL",
    "-" * 116,
    "Candidate-development target years: <= 2023.",
    "Reserved Step-27 validation target years: 2024 and 2025.",
    "Primary analysis retains 2017.",
    (
        "No-2017 sensitivity preserves the same four-observation minimum and "
        "uses common folds only; H5 has no valid no-2017 development fold."
    ),
    "No final candidate is selected in Step 26A.",
    "",
    "CANDIDATE METRICS BY HORIZON",
    "-" * 116,
    candidate_metrics.to_string(
        index=False
    ),
    "",
    "CANDIDATE EQUAL-HORIZON DEVELOPMENT SUMMARY",
    "-" * 116,
    candidate_summary.to_string(
        index=False
    ),
    "",
    "CANDIDATES + STEP-25 BASELINES ON SAME DEVELOPMENT WINDOW",
    "-" * 116,
    combined_summary.to_string(
        index=False
    ),
    "",
    "IMPORTANT",
    "-" * 116,
    (
        "All three Step-26A candidates are frozen and carried unchanged into "
        "Step 27. Development results must not be treated as reserved-validation results."
    ),
    (
        "Step 26A does not produce 2026-2030 forecasts and does not change "
        "the Step-22B production spatial-anomaly model."
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
# 12. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "12. REPRODUCIBILITY HASHES"
)

hash_files = [
    STEP25_ANNUAL_PATH,
    STEP25_PREDICTIONS_PATH,
    STEP25_AUDIT_PATH,
    CANDIDATE_PREDICTIONS_PATH,
    CANDIDATE_METRICS_PATH,
    CANDIDATE_SUMMARY_PATH,
    BASELINE_DEV_METRICS_PATH,
    COMBINED_DEV_METRICS_PATH,
    COMBINED_DEV_SUMMARY_PATH,
    NO2017_PREDICTIONS_PATH,
    NO2017_METRICS_PATH,
    NO2017_COMPARISON_PATH,
    SERIES_SPLIT_FIGURE_PATH,
    DEVELOPMENT_FIGURE_PATH,
    FREEZE_PATH,
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
    "STEP 26A COMPLETED SUCCESSFULLY"
)

print(
    "Annual thermal-level candidate development completed.\n"
    "Candidate-development targets were restricted to years <= 2023.\n"
    "2024 and 2025 remain reserved for Step-27 forecast validation.\n"
    "Primary analysis retained 2017; no-2017 sensitivity was saved separately.\n"
    "SES, damped Holt and Theil-Sen candidate definitions were frozen.\n"
    "All three candidates proceed unchanged to Step 27.\n"
    "No 2026-2030 forecast was generated.\n"
    "The Step-22B production LightGBM was not changed.\n"
    "\nNEXT: review Step-26A outputs before starting Step 26B."
)
