
"""
STEP 19B4 - ROBUST TEMPORAL ENSEMBLE EXPERIMENT
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Test whether combining the four finalized Step 17 tuned model predictions
reduces year-to-year temporal instability.

BACKGROUND
----------
Step 19A:
    diagnosed strong annual-level temporal instability.

Step 19B1:
    simple historical residual bias correction was not robust.

Step 19B2:
    train-only stable-feature filtering did not improve generalization.

Step 19B3:
    recent-year weighting produced only small, model-specific improvements
    and did not remove the difficult negative-R² years.

Step 19B4 therefore asks:

    Can a robust ensemble reduce dependence on any one model's errors?

IMPORTANT METHODOLOGY
---------------------
This file DOES NOT retrain any model.

It uses the finalized out-of-sample rolling predictions from Step 17:

    2018       -> 2019
    2018-2019  -> 2020
    2018-2020  -> 2021
    2018-2021  -> 2022
    2018-2022  -> 2023

The ensemble therefore combines predictions that were already generated
without using the corresponding validation year's target during training.

ELIGIBLE ENSEMBLE STRATEGIES
----------------------------
1. random_forest_single_reference
   - strongest single temporal candidate from Step 17
   - used as the main single-model reference

2. equal_mean_all4
   - arithmetic mean of RF, XGBoost, LightGBM, NGBoost
   - fixed; no target-dependent weighting

3. median_all4
   - median of the four predictions
   - robust to an extreme member prediction

4. trimmed_mean_middle2
   - sort the four predictions grid-by-grid
   - remove the lowest and highest
   - average the middle two
   - robust to one low and one high extreme member

5. causal_inverse_mae_all4
   - for year t, model weights are learned ONLY from earlier rolling
     validation years y < t
   - weight_m ∝ 1 / prior_mean_MAE_m
   - 2019 has no earlier validation history -> equal weights

6. causal_inverse_rmse_all4
   - same causal rule, using prior mean RMSE
   - 2019 fallback -> equal weights

No strategy uses the current validation year's target to construct its
prediction or weights.

RESIDUAL-CORRELATION DIAGNOSTIC
-------------------------------
Ensembles help most when member errors are not perfectly correlated.
This script therefore measures pairwise residual correlations by year and
pooled across 2019-2023.

ENSEMBLE DISAGREEMENT
---------------------
The standard deviation across the four member predictions is saved as a
diagnostic disagreement signal. It is NOT treated as calibrated uncertainty.

GENERALIZATION GAP
------------------
This step combines validation predictions only. Comparable ensemble training
predictions do not exist for every strategy, so a train-vs-validation
generalization gap is intentionally NOT fabricated here.

PROTECTED YEARS
---------------
2024 reserved validation is NOT loaded.
2025 final lockbox is NOT loaded.

NO final model or strategy is selected in Step 19B4.
"""

from pathlib import Path
import hashlib
import itertools
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =============================================================================
# 1. SETTINGS
# =============================================================================

GRID_COL = "grid_id"
AREA_COL = "area_name"

MODEL_ORDER = [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "NGBoost",
]

MODEL_SHORT = {
    "Random Forest": "rf",
    "XGBoost": "xgb",
    "LightGBM": "lgbm",
    "NGBoost": "ngb",
}

VALIDATION_YEARS = [
    2019,
    2020,
    2021,
    2022,
    2023,
]

CAUSAL_WEIGHT_ACTIVE_YEARS = [
    2020,
    2021,
    2022,
    2023,
]

EXPECTED_UNIQUE_GRIDS = 2194

EXPECTED_STEP17_ROWS = (
    len(MODEL_ORDER)
    * len(VALIDATION_YEARS)
    * EXPECTED_UNIQUE_GRIDS
)

ELIGIBLE_STRATEGIES = [
    "random_forest_single_reference",
    "equal_mean_all4",
    "median_all4",
    "trimmed_mean_middle2",
    "causal_inverse_mae_all4",
    "causal_inverse_rmse_all4",
]

EPS = 1e-12
METRIC_TOLERANCE = 1e-9


# =============================================================================
# 2. PATHS
# =============================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
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

STEP19A_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "19a_temporal_diagnosis_audit.csv"
)

STEP19B1_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "19b1_temporal_bias_correction_audit.csv"
)

STEP19B2_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "19b2_stable_feature_subset_audit.csv"
)

STEP19B3_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "19b3_recent_year_weighting_audit.csv"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)

for folder in [
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

MEMBER_METRICS_PATH = (
    METRICS_DIR
    / "19b4_member_fold_metrics.csv"
)

ERROR_CORRELATION_PATH = (
    METRICS_DIR
    / "19b4_member_residual_correlations.csv"
)

CAUSAL_WEIGHTS_PATH = (
    METRICS_DIR
    / "19b4_causal_ensemble_weights.csv"
)

FOLD_METRICS_PATH = (
    METRICS_DIR
    / "19b4_ensemble_fold_metrics.csv"
)

SUMMARY_PATH = (
    METRICS_DIR
    / "19b4_ensemble_summary.csv"
)

DESCRIPTIVE_BEST_PATH = (
    METRICS_DIR
    / "19b4_descriptive_best_ensemble.csv"
)

DISAGREEMENT_PATH = (
    METRICS_DIR
    / "19b4_ensemble_disagreement_by_year.csv"
)

PREDICTIONS_PATH = (
    PREDICTIONS_DIR
    / "19b4_temporal_ensemble_predictions.csv"
)

REFERENCE_REPRO_PATH = (
    REPORTS_DIR
    / "19b4_step17_reference_reproduction_check.csv"
)

AUDIT_PATH = (
    REPORTS_DIR
    / "19b4_temporal_ensemble_audit.csv"
)

REPORT_PATH = (
    REPORTS_DIR
    / "19b4_temporal_ensemble_report.txt"
)

HASH_PATH = (
    REPORTS_DIR
    / "19b4_temporal_ensemble_hashes.csv"
)

FIG_MAE_PATH = (
    FIGURES_DIR
    / "19b4_ensemble_mean_mae.png"
)

FIG_R2_PATH = (
    FIGURES_DIR
    / "19b4_ensemble_r2_by_year.png"
)

FIG_CORR_PATH = (
    FIGURES_DIR
    / "19b4_member_residual_correlation.png"
)

FIG_DISAGREEMENT_PATH = (
    FIGURES_DIR
    / "19b4_ensemble_member_disagreement.png"
)


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(
    title: str,
) -> None:
    print(
        "\n"
        + "=" * 126
    )
    print(
        title
    )
    print(
        "=" * 126
    )


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as file_obj:
        for chunk in iter(
            lambda: file_obj.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def check_audit_all_pass(
    path: Path,
    label: str,
) -> None:
    if not path.exists():
        print(
            f"\nERROR: required {label} audit is missing: {path}"
        )
        sys.exit(1)

    audit = pd.read_csv(
        path
    )

    if "status" not in audit.columns:
        print(
            f"\nERROR: {label} audit has no status column."
        )
        sys.exit(1)

    bad = audit[
        audit[
            "status"
        ]
        .astype(str)
        .str.upper()
        != "PASS"
    ]

    if not bad.empty:
        print(
            f"\nERROR: {label} audit is not fully PASS."
        )
        print(
            bad.to_string(
                index=False
            )
        )
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

    valid = (
        np.abs(
            y_true
        )
        > EPS
    )

    if not np.any(
        valid
    ):
        return np.nan

    return float(
        np.mean(
            np.abs(
                (
                    y_true[
                        valid
                    ]
                    - y_pred[
                        valid
                    ]
                )
                / y_true[
                    valid
                ]
            )
        )
        * 100.0
    )


def regression_metrics(
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

    residual = (
        y_true
        - y_pred
    )

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "mape_percent": mape_percent(
            y_true,
            y_pred,
        ),
        "mean_prediction_bias_pred_minus_actual_c": float(
            np.mean(
                y_pred
                - y_true
            )
        ),
        "abs_mean_prediction_bias_c": float(
            abs(
                np.mean(
                    y_pred
                    - y_true
                )
            )
        ),
        "residual_std_c": float(
            np.std(
                residual,
                ddof=0,
            )
        ),
    }


def normalize_inverse_error_weights(
    errors: dict,
) -> dict:
    inverse = {}

    for model_name in MODEL_ORDER:
        error = float(
            errors[
                model_name
            ]
        )

        if (
            not np.isfinite(
                error
            )
            or error
            <= 0
        ):
            raise ValueError(
                f"Invalid prior error for {model_name}: {error}"
            )

        inverse[
            model_name
        ] = (
            1.0
            / error
        )

    total = float(
        sum(
            inverse.values()
        )
    )

    if (
        not np.isfinite(
            total
        )
        or total
        <= 0
    ):
        raise ValueError(
            "Invalid inverse-error weight total."
        )

    return {
        model_name: (
            inverse[
                model_name
            ]
            / total
        )
        for model_name in MODEL_ORDER
    }


def equal_weights() -> dict:
    return {
        model_name: (
            1.0
            / len(
                MODEL_ORDER
            )
        )
        for model_name in MODEL_ORDER
    }


def weighted_prediction(
    prediction_matrix,
    weights: dict,
) -> np.ndarray:
    weight_vector = np.asarray(
        [
            weights[
                model_name
            ]
            for model_name in MODEL_ORDER
        ],
        dtype=float,
    )

    if not np.isclose(
        np.sum(
            weight_vector
        ),
        1.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Ensemble weights do not sum to 1."
        )

    return np.asarray(
        prediction_matrix
        @ weight_vector,
        dtype=float,
    )


# =============================================================================
# 4. PREREQUISITE AUDITS
# =============================================================================

section(
    "STEP 19B4 - ROBUST TEMPORAL ENSEMBLE"
)

required_files = [
    STEP17_PREDICTIONS_PATH,
    STEP17_FOLD_METRICS_PATH,
    STEP17_AUDIT_PATH,
    STEP19A_AUDIT_PATH,
    STEP19B1_AUDIT_PATH,
    STEP19B2_AUDIT_PATH,
    STEP19B3_AUDIT_PATH,
]

for path in required_files:
    if not path.exists():
        print(
            f"\nERROR: required file missing: {path}"
        )
        sys.exit(1)

for path, label in [
    (
        STEP17_AUDIT_PATH,
        "Step 17",
    ),
    (
        STEP19A_AUDIT_PATH,
        "Step 19A",
    ),
    (
        STEP19B1_AUDIT_PATH,
        "Step 19B1",
    ),
    (
        STEP19B2_AUDIT_PATH,
        "Step 19B2",
    ),
    (
        STEP19B3_AUDIT_PATH,
        "Step 19B3",
    ),
]:
    check_audit_all_pass(
        path,
        label,
    )

print(
    "PASS: prerequisite audits are fully PASS."
)
print(
    "Step 19B4 combines saved Step 17 rolling predictions only."
)
print(
    "No model training or hyperparameter tuning occurs."
)
print(
    "2024 and 2025 are not loaded."
)


# =============================================================================
# 5. LOAD + VERIFY STEP 17 PREDICTIONS
# =============================================================================

section(
    "1. LOAD + VERIFY STEP 17 ROLLING PREDICTIONS"
)

pred = pd.read_csv(
    STEP17_PREDICTIONS_PATH,
    dtype={
        GRID_COL: str,
    },
)

step17_fold = pd.read_csv(
    STEP17_FOLD_METRICS_PATH
)

required_prediction_columns = {
    "model",
    GRID_COL,
    AREA_COL,
    "validation_year",
    "actual_lst_c",
    "predicted_lst_c",
    "residual_actual_minus_predicted",
}

missing_prediction_columns = sorted(
    required_prediction_columns
    - set(
        pred.columns
    )
)

if missing_prediction_columns:
    print(
        "\nERROR: Step 17 prediction file missing: "
        f"{missing_prediction_columns}"
    )
    sys.exit(1)

pred[
    "validation_year"
] = pd.to_numeric(
    pred[
        "validation_year"
    ],
    errors="coerce",
).astype(
    int
)

for column in [
    "actual_lst_c",
    "predicted_lst_c",
    "residual_actual_minus_predicted",
]:
    pred[
        column
    ] = pd.to_numeric(
        pred[
            column
        ],
        errors="coerce",
    )

if len(
    pred
) != EXPECTED_STEP17_ROWS:
    print(
        f"\nERROR: Step 17 rows={len(pred):,}; "
        f"expected={EXPECTED_STEP17_ROWS:,}."
    )
    sys.exit(1)

if sorted(
    pred[
        "model"
    ]
    .unique()
    .tolist()
) != sorted(
    MODEL_ORDER
):
    print(
        "\nERROR: Step 17 model names do not match."
    )
    sys.exit(1)

if sorted(
    pred[
        "validation_year"
    ]
    .unique()
    .tolist()
) != VALIDATION_YEARS:
    print(
        "\nERROR: Step 17 validation years do not match."
    )
    sys.exit(1)

if not np.isfinite(
    pred[
        [
            "actual_lst_c",
            "predicted_lst_c",
            "residual_actual_minus_predicted",
        ]
    ]
    .to_numpy(
        dtype=float
    )
).all():
    print(
        "\nERROR: non-finite Step 17 prediction values found."
    )
    sys.exit(1)

key_counts = (
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
    len(
        key_counts
    )
    == EXPECTED_STEP17_ROWS
    and key_counts.eq(
        1
    ).all()
):
    print(
        "\nERROR: duplicate/missing model-grid-year prediction keys."
    )
    sys.exit(1)

print(
    f"Rows             : {len(pred):,}"
)
print(
    f"Models           : {MODEL_ORDER}"
)
print(
    f"Validation years : {VALIDATION_YEARS}"
)
print(
    "2024             : NOT LOADED"
)
print(
    "2025             : NOT LOADED"
)


# =============================================================================
# 6. REPRODUCE STEP 17 MEMBER METRICS
# =============================================================================

section(
    "2. REPRODUCE STEP 17 MEMBER METRICS"
)

required_fold_columns = {
    "model",
    "validation_year",
    "validation_mae",
    "validation_rmse",
    "validation_r2",
    "validation_mape_percent",
}

missing_fold_columns = sorted(
    required_fold_columns
    - set(
        step17_fold.columns
    )
)

if missing_fold_columns:
    print(
        "\nERROR: Step 17 fold metrics missing: "
        f"{missing_fold_columns}"
    )
    sys.exit(1)

member_rows = []
repro_rows = []

for model_name in MODEL_ORDER:
    for validation_year in VALIDATION_YEARS:
        subset = pred[
            (
                pred[
                    "model"
                ]
                == model_name
            )
            & (
                pred[
                    "validation_year"
                ]
                == validation_year
            )
        ]

        current = regression_metrics(
            subset[
                "actual_lst_c"
            ],
            subset[
                "predicted_lst_c"
            ],
        )

        member_rows.append(
            {
                "model": model_name,
                "validation_year": (
                    validation_year
                ),
                "mae": current[
                    "mae"
                ],
                "rmse": current[
                    "rmse"
                ],
                "r2": current[
                    "r2"
                ],
                "mape_percent": current[
                    "mape_percent"
                ],
                "mean_prediction_bias_pred_minus_actual_c": current[
                    "mean_prediction_bias_pred_minus_actual_c"
                ],
                "abs_mean_prediction_bias_c": current[
                    "abs_mean_prediction_bias_c"
                ],
                "negative_r2": bool(
                    current[
                        "r2"
                    ]
                    < 0
                ),
            }
        )

        ref = step17_fold[
            (
                step17_fold[
                    "model"
                ]
                == model_name
            )
            & (
                step17_fold[
                    "validation_year"
                ]
                == validation_year
            )
        ]

        if len(
            ref
        ) != 1:
            print(
                "\nERROR: Step 17 fold metric row mismatch."
            )
            sys.exit(1)

        ref = ref.iloc[
            0
        ]

        mae_diff = (
            current[
                "mae"
            ]
            - float(
                ref[
                    "validation_mae"
                ]
            )
        )

        rmse_diff = (
            current[
                "rmse"
            ]
            - float(
                ref[
                    "validation_rmse"
                ]
            )
        )

        r2_diff = (
            current[
                "r2"
            ]
            - float(
                ref[
                    "validation_r2"
                ]
            )
        )

        repro_rows.append(
            {
                "model": model_name,
                "validation_year": (
                    validation_year
                ),
                "mae_difference": mae_diff,
                "rmse_difference": rmse_diff,
                "r2_difference": r2_diff,
                "status": (
                    "PASS"
                    if (
                        abs(
                            mae_diff
                        )
                        <= METRIC_TOLERANCE
                        and abs(
                            rmse_diff
                        )
                        <= METRIC_TOLERANCE
                        and abs(
                            r2_diff
                        )
                        <= METRIC_TOLERANCE
                    )
                    else "FAIL"
                ),
            }
        )

member_metrics = pd.DataFrame(
    member_rows
)

reference_repro = pd.DataFrame(
    repro_rows
)

member_metrics.to_csv(
    MEMBER_METRICS_PATH,
    index=False,
)

reference_repro.to_csv(
    REFERENCE_REPRO_PATH,
    index=False,
)

print(
    reference_repro.to_string(
        index=False
    )
)

if (
    reference_repro[
        "status"
    ]
    != "PASS"
).any():
    print(
        "\nERROR: Step 17 reproduction failed."
    )
    sys.exit(1)

print(
    "\nPASS: all 20 member model-year predictions reproduce Step 17."
)


# =============================================================================
# 7. BUILD ALIGNED MEMBER PREDICTION TABLE
# =============================================================================

section(
    "3. ALIGN MEMBER PREDICTIONS"
)

base_keys = (
    pred[
        [
            GRID_COL,
            AREA_COL,
            "validation_year",
            "actual_lst_c",
        ]
    ]
    .drop_duplicates(
        subset=[
            GRID_COL,
            "validation_year",
        ]
    )
    .copy()
)

expected_base_rows = (
    len(
        VALIDATION_YEARS
    )
    * EXPECTED_UNIQUE_GRIDS
)

if len(
    base_keys
) != expected_base_rows:
    print(
        f"\nERROR: base grid-year rows={len(base_keys)}; "
        f"expected={expected_base_rows}."
    )
    sys.exit(1)

aligned = base_keys.copy()

for model_name in MODEL_ORDER:
    short = MODEL_SHORT[
        model_name
    ]

    model_pred = pred[
        pred[
            "model"
        ]
        == model_name
    ][
        [
            GRID_COL,
            "validation_year",
            "actual_lst_c",
            "predicted_lst_c",
            "residual_actual_minus_predicted",
        ]
    ].copy()

    model_pred = model_pred.rename(
        columns={
            "actual_lst_c": (
                f"actual_{short}"
            ),
            "predicted_lst_c": (
                f"pred_{short}"
            ),
            "residual_actual_minus_predicted": (
                f"resid_{short}"
            ),
        }
    )

    aligned = aligned.merge(
        model_pred,
        on=[
            GRID_COL,
            "validation_year",
        ],
        how="left",
        validate="one_to_one",
    )

    actual_diff = np.max(
        np.abs(
            aligned[
                f"actual_{short}"
            ]
            - aligned[
                "actual_lst_c"
            ]
        )
    )

    if (
        not np.isfinite(
            actual_diff
        )
        or actual_diff
        > METRIC_TOLERANCE
    ):
        print(
            f"\nERROR: actual target mismatch for {model_name}."
        )
        sys.exit(1)

if aligned.isna().any().any():
    print(
        "\nERROR: missing values after member prediction alignment."
    )
    sys.exit(1)

prediction_cols = [
    f"pred_{MODEL_SHORT[model_name]}"
    for model_name in MODEL_ORDER
]

residual_cols = [
    f"resid_{MODEL_SHORT[model_name]}"
    for model_name in MODEL_ORDER
]

print(
    f"Aligned grid-year rows : {len(aligned):,}"
)
print(
    f"Member prediction cols : {prediction_cols}"
)
print(
    "PASS: all four model predictions aligned by grid-year."
)


# =============================================================================
# 8. MEMBER RESIDUAL CORRELATION
# =============================================================================

section(
    "4. MEMBER RESIDUAL CORRELATION DIAGNOSTIC"
)

correlation_rows = []

for scope_year in [
    *VALIDATION_YEARS,
    "pooled_2019_2023",
]:
    if scope_year == "pooled_2019_2023":
        subset = aligned
    else:
        subset = aligned[
            aligned[
                "validation_year"
            ]
            == int(
                scope_year
            )
        ]

    for model_a, model_b in itertools.combinations(
        MODEL_ORDER,
        2,
    ):
        col_a = (
            f"resid_{MODEL_SHORT[model_a]}"
        )

        col_b = (
            f"resid_{MODEL_SHORT[model_b]}"
        )

        corr = float(
            np.corrcoef(
                subset[
                    col_a
                ]
                .to_numpy(
                    dtype=float
                ),
                subset[
                    col_b
                ]
                .to_numpy(
                    dtype=float
                ),
            )[
                0,
                1
            ]
        )

        correlation_rows.append(
            {
                "scope": str(
                    scope_year
                ),
                "model_a": model_a,
                "model_b": model_b,
                "pearson_residual_correlation": (
                    corr
                ),
                "absolute_correlation": abs(
                    corr
                ),
            }
        )

error_correlations = pd.DataFrame(
    correlation_rows
)

error_correlations.to_csv(
    ERROR_CORRELATION_PATH,
    index=False,
)

print(
    error_correlations.to_string(
        index=False
    )
)

pooled_corr = error_correlations[
    error_correlations[
        "scope"
    ]
    == "pooled_2019_2023"
]

print(
    "\nPooled residual correlation:"
)
print(
    f"Mean absolute correlation = "
    f"{pooled_corr['absolute_correlation'].mean():.4f}"
)
print(
    f"Minimum absolute correlation = "
    f"{pooled_corr['absolute_correlation'].min():.4f}"
)
print(
    f"Maximum absolute correlation = "
    f"{pooled_corr['absolute_correlation'].max():.4f}"
)


# =============================================================================
# 9. BUILD CAUSAL ENSEMBLE WEIGHTS
# =============================================================================

section(
    "5. BUILD CAUSAL PRIOR-YEAR ENSEMBLE WEIGHTS"
)

weight_rows = []
weight_lookup = {}

for validation_year in VALIDATION_YEARS:
    prior_years = [
        year
        for year in VALIDATION_YEARS
        if year < validation_year
    ]

    for strategy in [
        "causal_inverse_mae_all4",
        "causal_inverse_rmse_all4",
    ]:
        if not prior_years:
            weights = equal_weights()
            source_metric = (
                "equal_fallback_no_prior_oof_history"
            )

        else:
            prior_metrics = member_metrics[
                member_metrics[
                    "validation_year"
                ]
                .isin(
                    prior_years
                )
            ]

            if strategy == "causal_inverse_mae_all4":
                mean_error = (
                    prior_metrics.groupby(
                        "model"
                    )[
                        "mae"
                    ]
                    .mean()
                    .to_dict()
                )

                weights = normalize_inverse_error_weights(
                    mean_error
                )

                source_metric = (
                    "prior_year_mean_oof_mae"
                )

            else:
                mean_error = (
                    prior_metrics.groupby(
                        "model"
                    )[
                        "rmse"
                    ]
                    .mean()
                    .to_dict()
                )

                weights = normalize_inverse_error_weights(
                    mean_error
                )

                source_metric = (
                    "prior_year_mean_oof_rmse"
                )

        weight_lookup[
            (
                validation_year,
                strategy,
            )
        ] = weights

        for model_name in MODEL_ORDER:
            weight_rows.append(
                {
                    "validation_year": (
                        validation_year
                    ),
                    "strategy": strategy,
                    "prior_oof_year_count": len(
                        prior_years
                    ),
                    "prior_oof_years": (
                        ",".join(
                            str(
                                year
                            )
                            for year in prior_years
                        )
                        if prior_years
                        else ""
                    ),
                    "max_source_year": (
                        max(
                            prior_years
                        )
                        if prior_years
                        else np.nan
                    ),
                    "source_metric": (
                        source_metric
                    ),
                    "model": model_name,
                    "weight": float(
                        weights[
                            model_name
                        ]
                    ),
                    "uses_current_validation_year_target": (
                        False
                    ),
                }
            )

        print(
            f"{validation_year} | {strategy:<26} | "
            + " | ".join(
                f"{MODEL_SHORT[m]}="
                f"{weights[m]:.4f}"
                for m in MODEL_ORDER
            )
        )

causal_weights = pd.DataFrame(
    weight_rows
)

causal_weights.to_csv(
    CAUSAL_WEIGHTS_PATH,
    index=False,
)

weight_sums = (
    causal_weights.groupby(
        [
            "validation_year",
            "strategy",
        ]
    )[
        "weight"
    ]
    .sum()
)

if not np.allclose(
    weight_sums.to_numpy(
        dtype=float
    ),
    1.0,
    atol=1e-12,
):
    print(
        "\nERROR: causal ensemble weights do not sum to 1."
    )
    sys.exit(1)

future_weight_leakage = bool(
    (
        causal_weights[
            "max_source_year"
        ]
        .dropna()
        >= causal_weights.loc[
            causal_weights[
                "max_source_year"
            ]
            .notna(),
            "validation_year",
        ]
        .to_numpy()
    ).any()
)

if future_weight_leakage:
    print(
        "\nERROR: causal weight construction used current/future year."
    )
    sys.exit(1)

print(
    f"\nSaved: {CAUSAL_WEIGHTS_PATH}"
)
print(
    "PASS: causal weights use prior OOF years only."
)


# =============================================================================
# 10. BUILD ENSEMBLE PREDICTIONS
# =============================================================================

section(
    "6. BUILD ELIGIBLE ENSEMBLE PREDICTIONS"
)

prediction_frames = []
fold_metric_rows = []
disagreement_rows = []

for validation_year in VALIDATION_YEARS:
    year_df = (
        aligned[
            aligned[
                "validation_year"
            ]
            == validation_year
        ]
        .sort_values(
            GRID_COL
        )
        .reset_index(
            drop=True
        )
    )

    y_true = year_df[
        "actual_lst_c"
    ].to_numpy(
        dtype=float
    )

    prediction_matrix = year_df[
        prediction_cols
    ].to_numpy(
        dtype=float
    )

    member_std = np.std(
        prediction_matrix,
        axis=1,
        ddof=0,
    )

    member_range = (
        np.max(
            prediction_matrix,
            axis=1,
        )
        - np.min(
            prediction_matrix,
            axis=1,
        )
    )

    disagreement_rows.append(
        {
            "validation_year": (
                validation_year
            ),
            "rows": len(
                year_df
            ),
            "mean_member_prediction_std_c": float(
                np.mean(
                    member_std
                )
            ),
            "median_member_prediction_std_c": float(
                np.median(
                    member_std
                )
            ),
            "p90_member_prediction_std_c": float(
                np.quantile(
                    member_std,
                    0.90,
                )
            ),
            "mean_member_prediction_range_c": float(
                np.mean(
                    member_range
                )
            ),
            "p90_member_prediction_range_c": float(
                np.quantile(
                    member_range,
                    0.90,
                )
            ),
        }
    )

    sorted_matrix = np.sort(
        prediction_matrix,
        axis=1,
    )

    strategy_predictions = {
        "random_forest_single_reference": (
            year_df[
                "pred_rf"
            ]
            .to_numpy(
                dtype=float
            )
        ),
        "equal_mean_all4": np.mean(
            prediction_matrix,
            axis=1,
        ),
        "median_all4": np.median(
            prediction_matrix,
            axis=1,
        ),
        "trimmed_mean_middle2": np.mean(
            sorted_matrix[
                :,
                1:3,
            ],
            axis=1,
        ),
        "causal_inverse_mae_all4": weighted_prediction(
            prediction_matrix,
            weight_lookup[
                (
                    validation_year,
                    "causal_inverse_mae_all4",
                )
            ],
        ),
        "causal_inverse_rmse_all4": weighted_prediction(
            prediction_matrix,
            weight_lookup[
                (
                    validation_year,
                    "causal_inverse_rmse_all4",
                )
            ],
        ),
    }

    if sorted(
        strategy_predictions.keys()
    ) != sorted(
        ELIGIBLE_STRATEGIES
    ):
        print(
            "\nERROR: ensemble strategy dictionary mismatch."
        )
        sys.exit(1)

    for strategy in ELIGIBLE_STRATEGIES:
        y_pred = np.asarray(
            strategy_predictions[
                strategy
            ],
            dtype=float,
        )

        if not np.isfinite(
            y_pred
        ).all():
            print(
                f"\nERROR: non-finite prediction for {strategy}, "
                f"{validation_year}."
            )
            sys.exit(1)

        metrics = regression_metrics(
            y_true,
            y_pred,
        )

        fold_metric_rows.append(
            {
                "strategy": strategy,
                "validation_year": (
                    validation_year
                ),
                "rows": len(
                    year_df
                ),
                "causal_weight_history_active": bool(
                    strategy.startswith(
                        "causal_"
                    )
                    and validation_year
                    in CAUSAL_WEIGHT_ACTIVE_YEARS
                ),
                "mae": metrics[
                    "mae"
                ],
                "rmse": metrics[
                    "rmse"
                ],
                "r2": metrics[
                    "r2"
                ],
                "mape_percent": metrics[
                    "mape_percent"
                ],
                "mean_prediction_bias_pred_minus_actual_c": (
                    metrics[
                        "mean_prediction_bias_pred_minus_actual_c"
                    ]
                ),
                "abs_mean_prediction_bias_c": (
                    metrics[
                        "abs_mean_prediction_bias_c"
                    ]
                ),
                "residual_std_c": (
                    metrics[
                        "residual_std_c"
                    ]
                ),
                "negative_r2": bool(
                    metrics[
                        "r2"
                    ]
                    < 0
                ),
                "generalization_gap_available": (
                    False
                ),
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "strategy": strategy,
                    "validation_year": (
                        validation_year
                    ),
                    GRID_COL: year_df[
                        GRID_COL
                    ]
                    .astype(str)
                    .to_numpy(),
                    AREA_COL: year_df[
                        AREA_COL
                    ]
                    .astype(str)
                    .to_numpy(),
                    "actual_lst_c": (
                        y_true
                    ),
                    "predicted_lst_c": (
                        y_pred
                    ),
                    "residual_actual_minus_predicted": (
                        y_true
                        - y_pred
                    ),
                    "member_prediction_std_c": (
                        member_std
                    ),
                    "member_prediction_range_c": (
                        member_range
                    ),
                }
            )
        )

        print(
            f"{validation_year} | "
            f"{strategy:<31} | "
            f"MAE={metrics['mae']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | "
            f"R²={metrics['r2']:.4f} | "
            f"Bias="
            f"{metrics['mean_prediction_bias_pred_minus_actual_c']:+.3f} °C"
        )

fold_metrics = pd.DataFrame(
    fold_metric_rows
)

ensemble_predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

disagreement_summary = pd.DataFrame(
    disagreement_rows
)

fold_metrics.to_csv(
    FOLD_METRICS_PATH,
    index=False,
)

ensemble_predictions.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

disagreement_summary.to_csv(
    DISAGREEMENT_PATH,
    index=False,
)

print(
    f"\nSaved: {FOLD_METRICS_PATH}"
)
print(
    f"Saved: {PREDICTIONS_PATH}"
)
print(
    f"Saved: {DISAGREEMENT_PATH}"
)


# =============================================================================
# 11. ENSEMBLE SUMMARY
# =============================================================================

section(
    "7. ENSEMBLE STRATEGY SUMMARY"
)

rf_reference = fold_metrics[
    fold_metrics[
        "strategy"
    ]
    == "random_forest_single_reference"
]

rf_all_mean_mae = float(
    rf_reference[
        "mae"
    ].mean()
)

rf_active_mean_mae = float(
    rf_reference[
        rf_reference[
            "validation_year"
        ]
        .isin(
            CAUSAL_WEIGHT_ACTIVE_YEARS
        )
    ][
        "mae"
    ].mean()
)

summary_rows = []

for strategy in ELIGIBLE_STRATEGIES:
    subset_all = fold_metrics[
        fold_metrics[
            "strategy"
        ]
        == strategy
    ]

    subset_active = subset_all[
        subset_all[
            "validation_year"
        ]
        .isin(
            CAUSAL_WEIGHT_ACTIVE_YEARS
        )
    ]

    if len(
        subset_all
    ) != len(
        VALIDATION_YEARS
    ):
        print(
            "\nERROR: incomplete ensemble all-fold summary."
        )
        sys.exit(1)

    if len(
        subset_active
    ) != len(
        CAUSAL_WEIGHT_ACTIVE_YEARS
    ):
        print(
            "\nERROR: incomplete active-year ensemble summary."
        )
        sys.exit(1)

    summary_rows.append(
        {
            "strategy": strategy,
            "all_folds_mean_mae": float(
                subset_all[
                    "mae"
                ].mean()
            ),
            "all_folds_std_mae": float(
                subset_all[
                    "mae"
                ].std(
                    ddof=0
                )
            ),
            "all_folds_mean_rmse": float(
                subset_all[
                    "rmse"
                ].mean()
            ),
            "all_folds_mean_r2": float(
                subset_all[
                    "r2"
                ].mean()
            ),
            "all_folds_negative_r2_count": int(
                subset_all[
                    "negative_r2"
                ].sum()
            ),
            "all_folds_worst_mae": float(
                subset_all[
                    "mae"
                ].max()
            ),
            "all_folds_min_r2": float(
                subset_all[
                    "r2"
                ].min()
            ),
            "all_folds_mean_abs_year_bias_c": float(
                subset_all[
                    "abs_mean_prediction_bias_c"
                ].mean()
            ),
            "active_2020_2023_mean_mae": float(
                subset_active[
                    "mae"
                ].mean()
            ),
            "active_2020_2023_std_mae": float(
                subset_active[
                    "mae"
                ].std(
                    ddof=0
                )
            ),
            "active_2020_2023_mean_rmse": float(
                subset_active[
                    "rmse"
                ].mean()
            ),
            "active_2020_2023_mean_r2": float(
                subset_active[
                    "r2"
                ].mean()
            ),
            "active_2020_2023_negative_r2_count": int(
                subset_active[
                    "negative_r2"
                ].sum()
            ),
            "active_2020_2023_worst_mae": float(
                subset_active[
                    "mae"
                ].max()
            ),
            "active_2020_2023_min_r2": float(
                subset_active[
                    "r2"
                ].min()
            ),
            "active_2020_2023_mean_abs_year_bias_c": float(
                subset_active[
                    "abs_mean_prediction_bias_c"
                ].mean()
            ),
            "all_folds_mae_improvement_vs_rf": (
                rf_all_mean_mae
                - float(
                    subset_all[
                        "mae"
                    ].mean()
                )
            ),
            "active_2020_2023_mae_improvement_vs_rf": (
                rf_active_mean_mae
                - float(
                    subset_active[
                        "mae"
                    ].mean()
                )
            ),
            "generalization_gap": np.nan,
            "generalization_gap_note": (
                "Not defined: Step 19B4 combines OOF validation "
                "predictions only; comparable ensemble train predictions "
                "are unavailable."
            ),
        }
    )

summary = pd.DataFrame(
    summary_rows
)

summary[
    "descriptive_all_fold_mae_rank"
] = (
    summary[
        "all_folds_mean_mae"
    ]
    .rank(
        method="min",
        ascending=True,
    )
    .astype(
        int
    )
)

summary[
    "descriptive_active_2020_2023_mae_rank"
] = (
    summary[
        "active_2020_2023_mean_mae"
    ]
    .rank(
        method="min",
        ascending=True,
    )
    .astype(
        int
    )
)

summary = summary.sort_values(
    [
        "descriptive_all_fold_mae_rank",
        "all_folds_worst_mae",
        "all_folds_mean_rmse",
    ]
).reset_index(
    drop=True
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

print(
    summary.to_string(
        index=False
    )
)

descriptive_best = (
    summary.sort_values(
        [
            "all_folds_mean_mae",
            "all_folds_worst_mae",
            "all_folds_mean_rmse",
            "all_folds_negative_r2_count",
        ]
    )
    .head(
        1
    )
    .copy()
)

descriptive_best.to_csv(
    DESCRIPTIVE_BEST_PATH,
    index=False,
)

print(
    "\nDESCRIPTIVE BEST STEP 19B4 STRATEGY "
    "(development evidence only; NOT final selection):"
)
print(
    descriptive_best[
        [
            "strategy",
            "all_folds_mean_mae",
            "all_folds_mean_rmse",
            "all_folds_mean_r2",
            "all_folds_negative_r2_count",
            "all_folds_worst_mae",
            "all_folds_min_r2",
            "all_folds_mean_abs_year_bias_c",
            "all_folds_mae_improvement_vs_rf",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# 12. FIGURES
# =============================================================================

section(
    "8. FIGURES"
)

# ---- Mean MAE by strategy ----
plot_summary = (
    summary.set_index(
        "strategy"
    )
    .reindex(
        ELIGIBLE_STRATEGIES
    )
)

fig, ax = plt.subplots(
    figsize=(
        12,
        7,
    )
)

ax.bar(
    plot_summary.index,
    plot_summary[
        "all_folds_mean_mae"
    ],
)

ax.set_ylabel(
    "Mean rolling-origin MAE 2019-2023 (°C)"
)

ax.set_title(
    "Step 19B4 - Robust Temporal Ensemble Comparison"
)

ax.tick_params(
    axis="x",
    rotation=22,
)

ax.grid(
    axis="y",
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    FIG_MAE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ---- R² by year ----
fig, ax = plt.subplots(
    figsize=(
        12,
        7,
    )
)

for strategy in ELIGIBLE_STRATEGIES:
    sub = (
        fold_metrics[
            fold_metrics[
                "strategy"
            ]
            == strategy
        ]
        .sort_values(
            "validation_year"
        )
    )

    ax.plot(
        sub[
            "validation_year"
        ],
        sub[
            "r2"
        ],
        marker="o",
        label=strategy,
    )

ax.axhline(
    0,
    linewidth=1,
)

ax.set_xticks(
    VALIDATION_YEARS
)

ax.set_xlabel(
    "Validation year"
)

ax.set_ylabel(
    "R²"
)

ax.set_title(
    "Step 19B4 - Temporal R² by Ensemble Strategy"
)

ax.grid(
    alpha=0.2,
)

ax.legend(
    fontsize=8,
)

plt.tight_layout()

plt.savefig(
    FIG_R2_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ---- pooled residual correlations ----
pooled_plot = (
    pooled_corr.copy()
    .sort_values(
        "pearson_residual_correlation"
    )
)

labels = (
    pooled_plot[
        "model_a"
    ]
    + " vs "
    + pooled_plot[
        "model_b"
    ]
)

fig, ax = plt.subplots(
    figsize=(
        10,
        6,
    )
)

ax.barh(
    labels,
    pooled_plot[
        "pearson_residual_correlation"
    ],
)

ax.set_xlabel(
    "Pooled residual Pearson correlation"
)

ax.set_title(
    "Step 19B4 - Correlation of Member Model Errors"
)

ax.grid(
    axis="x",
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    FIG_CORR_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ---- member disagreement ----
fig, ax = plt.subplots(
    figsize=(
        10,
        6,
    )
)

ax.plot(
    disagreement_summary[
        "validation_year"
    ],
    disagreement_summary[
        "mean_member_prediction_std_c"
    ],
    marker="o",
    label="Mean member prediction SD",
)

ax.plot(
    disagreement_summary[
        "validation_year"
    ],
    disagreement_summary[
        "p90_member_prediction_std_c"
    ],
    marker="o",
    label="90th percentile member prediction SD",
)

ax.set_xticks(
    VALIDATION_YEARS
)

ax.set_xlabel(
    "Validation year"
)

ax.set_ylabel(
    "Across-model prediction disagreement (°C)"
)

ax.set_title(
    "Step 19B4 - Ensemble Member Disagreement"
)

ax.grid(
    alpha=0.2,
)

ax.legend()

plt.tight_layout()

plt.savefig(
    FIG_DISAGREEMENT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

for figure_path in [
    FIG_MAE_PATH,
    FIG_R2_PATH,
    FIG_CORR_PATH,
    FIG_DISAGREEMENT_PATH,
]:
    print(
        f"Saved: {figure_path}"
    )


# =============================================================================
# 13. STRICT AUDIT
# =============================================================================

section(
    "9. STRICT STEP 19B4 AUDIT"
)

expected_member_metric_rows = (
    len(
        MODEL_ORDER
    )
    * len(
        VALIDATION_YEARS
    )
)

expected_correlation_rows = (
    (
        len(
            MODEL_ORDER
        )
        * (
            len(
                MODEL_ORDER
            )
            - 1
        )
        // 2
    )
    * (
        len(
            VALIDATION_YEARS
        )
        + 1
    )
)

expected_weight_rows = (
    len(
        VALIDATION_YEARS
    )
    * 2
    * len(
        MODEL_ORDER
    )
)

expected_fold_metric_rows = (
    len(
        ELIGIBLE_STRATEGIES
    )
    * len(
        VALIDATION_YEARS
    )
)

expected_prediction_rows = (
    expected_fold_metric_rows
    * EXPECTED_UNIQUE_GRIDS
)

expected_summary_rows = len(
    ELIGIBLE_STRATEGIES
)

all_fold_metrics_finite = bool(
    np.isfinite(
        fold_metrics[
            [
                "mae",
                "rmse",
                "r2",
                "mape_percent",
                "mean_prediction_bias_pred_minus_actual_c",
                "abs_mean_prediction_bias_c",
                "residual_std_c",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    ).all()
)

all_predictions_finite = bool(
    np.isfinite(
        ensemble_predictions[
            [
                "actual_lst_c",
                "predicted_lst_c",
                "residual_actual_minus_predicted",
                "member_prediction_std_c",
                "member_prediction_range_c",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    ).all()
)

rf_reference_matches = True

for validation_year in VALIDATION_YEARS:
    ensemble_rf = fold_metrics[
        (
            fold_metrics[
                "strategy"
            ]
            == "random_forest_single_reference"
        )
        & (
            fold_metrics[
                "validation_year"
            ]
            == validation_year
        )
    ]

    member_rf = member_metrics[
        (
            member_metrics[
                "model"
            ]
            == "Random Forest"
        )
        & (
            member_metrics[
                "validation_year"
            ]
            == validation_year
        )
    ]

    if (
        len(
            ensemble_rf
        )
        != 1
        or len(
            member_rf
        )
        != 1
    ):
        rf_reference_matches = False
        break

    ensemble_rf = ensemble_rf.iloc[
        0
    ]

    member_rf = member_rf.iloc[
        0
    ]

    if not (
        abs(
            float(
                ensemble_rf[
                    "mae"
                ]
            )
            - float(
                member_rf[
                    "mae"
                ]
            )
        )
        <= METRIC_TOLERANCE
        and abs(
            float(
                ensemble_rf[
                    "rmse"
                ]
            )
            - float(
                member_rf[
                    "rmse"
                ]
            )
        )
        <= METRIC_TOLERANCE
        and abs(
            float(
                ensemble_rf[
                    "r2"
                ]
            )
            - float(
                member_rf[
                    "r2"
                ]
            )
        )
        <= METRIC_TOLERANCE
    ):
        rf_reference_matches = False
        break

causal_weight_sums_ok = bool(
    np.allclose(
        causal_weights.groupby(
            [
                "validation_year",
                "strategy",
            ]
        )[
            "weight"
        ]
        .sum()
        .to_numpy(
            dtype=float
        ),
        1.0,
        atol=1e-12,
    )
)

causal_prior_only = bool(
    (
        causal_weights[
            "max_source_year"
        ]
        .dropna()
        < causal_weights.loc[
            causal_weights[
                "max_source_year"
            ]
            .notna(),
            "validation_year",
        ]
        .to_numpy()
    ).all()
)

causal_no_current_target_flag = bool(
    not causal_weights[
        "uses_current_validation_year_target"
    ]
    .astype(bool)
    .any()
)

year_2019_equal_weights_ok = bool(
    np.allclose(
        causal_weights[
            causal_weights[
                "validation_year"
            ]
            == 2019
        ][
            "weight"
        ]
        .to_numpy(
            dtype=float
        ),
        0.25,
        atol=1e-12,
    )
)

generalization_gap_intentionally_missing = bool(
    fold_metrics[
        "generalization_gap_available"
    ]
    .eq(
        False
    )
    .all()
    and summary[
        "generalization_gap"
    ]
    .isna()
    .all()
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
        "check": "step19a_audit",
        "status": "PASS",
        "details": (
            "Step 19A audit loaded and fully PASS."
        ),
    },
    {
        "check": "step19b1_audit",
        "status": "PASS",
        "details": (
            "Step 19B1 audit loaded and fully PASS."
        ),
    },
    {
        "check": "step19b2_audit",
        "status": "PASS",
        "details": (
            "Step 19B2 audit loaded and fully PASS."
        ),
    },
    {
        "check": "step19b3_audit",
        "status": "PASS",
        "details": (
            "Step 19B3 audit loaded and fully PASS."
        ),
    },
    {
        "check": "step17_prediction_rows",
        "status": (
            "PASS"
            if len(
                pred
            )
            == EXPECTED_STEP17_ROWS
            else "FAIL"
        ),
        "details": (
            f"actual={len(pred)}, "
            f"expected={EXPECTED_STEP17_ROWS}"
        ),
    },
    {
        "check": "step17_member_metric_reproduction",
        "status": (
            "PASS"
            if len(
                reference_repro
            )
            == expected_member_metric_rows
            and (
                reference_repro[
                    "status"
                ]
                == "PASS"
            ).all()
            else "FAIL"
        ),
        "details": (
            f"actual={len(reference_repro)}, "
            f"expected={expected_member_metric_rows}"
        ),
    },
    {
        "check": "member_residual_correlation_rows",
        "status": (
            "PASS"
            if len(
                error_correlations
            )
            == expected_correlation_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(error_correlations)}, "
            f"expected={expected_correlation_rows}"
        ),
    },
    {
        "check": "causal_weight_rows",
        "status": (
            "PASS"
            if len(
                causal_weights
            )
            == expected_weight_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(causal_weights)}, "
            f"expected={expected_weight_rows}"
        ),
    },
    {
        "check": "causal_weights_sum_to_one",
        "status": (
            "PASS"
            if causal_weight_sums_ok
            else "FAIL"
        ),
        "details": (
            "Each causal model-weight vector sums to 1."
        ),
    },
    {
        "check": "causal_weights_prior_years_only",
        "status": (
            "PASS"
            if causal_prior_only
            else "FAIL"
        ),
        "details": (
            "Every learned ensemble weight uses only earlier "
            "rolling-origin validation years."
        ),
    },
    {
        "check": "causal_weights_current_target_not_used",
        "status": (
            "PASS"
            if causal_no_current_target_flag
            else "FAIL"
        ),
        "details": (
            "No causal ensemble weight uses the current validation target."
        ),
    },
    {
        "check": "2019_causal_weight_fallback_equal",
        "status": (
            "PASS"
            if year_2019_equal_weights_ok
            else "FAIL"
        ),
        "details": (
            "2019 has no prior OOF history, so both causal "
            "strategies use equal 0.25 model weights."
        ),
    },
    {
        "check": "rf_single_reference_matches_step17",
        "status": (
            "PASS"
            if rf_reference_matches
            else "FAIL"
        ),
        "details": (
            "Random Forest single-model reference exactly matches "
            "its Step 17 validation metrics."
        ),
    },
    {
        "check": "ensemble_fold_metric_rows",
        "status": (
            "PASS"
            if len(
                fold_metrics
            )
            == expected_fold_metric_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(fold_metrics)}, "
            f"expected={expected_fold_metric_rows}"
        ),
    },
    {
        "check": "ensemble_prediction_rows",
        "status": (
            "PASS"
            if len(
                ensemble_predictions
            )
            == expected_prediction_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(ensemble_predictions)}, "
            f"expected={expected_prediction_rows}"
        ),
    },
    {
        "check": "ensemble_summary_rows",
        "status": (
            "PASS"
            if len(
                summary
            )
            == expected_summary_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(summary)}, "
            f"expected={expected_summary_rows}"
        ),
    },
    {
        "check": "all_fold_metrics_finite",
        "status": (
            "PASS"
            if all_fold_metrics_finite
            else "FAIL"
        ),
        "details": str(
            all_fold_metrics_finite
        ),
    },
    {
        "check": "all_ensemble_predictions_finite",
        "status": (
            "PASS"
            if all_predictions_finite
            else "FAIL"
        ),
        "details": str(
            all_predictions_finite
        ),
    },
    {
        "check": "generalization_gap_not_fabricated",
        "status": (
            "PASS"
            if generalization_gap_intentionally_missing
            else "FAIL"
        ),
        "details": (
            "No train-vs-validation gap is reported because "
            "comparable ensemble train predictions are unavailable."
        ),
    },
    {
        "check": "ensemble_disagreement_not_called_uncertainty",
        "status": "PASS",
        "details": (
            "Across-model prediction spread is saved only as "
            "diagnostic disagreement, not calibrated uncertainty."
        ),
    },
    {
        "check": "2024_loaded_or_used",
        "status": "PASS",
        "details": (
            "No. 2024 was not loaded by Step 19B4."
        ),
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": (
            "No. 2025 was not loaded by Step 19B4."
        ),
    },
    {
        "check": "model_training_performed",
        "status": "PASS",
        "details": (
            "No. Step 19B4 combines finalized Step 17 OOF predictions."
        ),
    },
    {
        "check": "hyperparameter_tuning_performed",
        "status": "PASS",
        "details": (
            "No hyperparameter tuning occurs."
        ),
    },
    {
        "check": "feature_selection_performed",
        "status": "PASS",
        "details": (
            "No feature selection occurs."
        ),
    },
    {
        "check": "bias_correction_applied",
        "status": "PASS",
        "details": (
            "No post-hoc annual bias correction occurs."
        ),
    },
    {
        "check": "final_model_selected",
        "status": "PASS",
        "details": (
            "No. Rankings are descriptive within Step 19B4 only."
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
    print(
        "\nERROR: one or more Step 19B4 audit checks failed."
    )
    sys.exit(1)


# =============================================================================
# 14. REPORT
# =============================================================================

section(
    "10. SAVE STEP 19B4 REPORT"
)

report_lines = [
    "STEP 19B4 - ROBUST TEMPORAL ENSEMBLE",
    "=" * 112,
    "",
    "PURPOSE",
    "-" * 112,
    "Test whether combining the four finalized Step 17 rolling predictions",
    "improves temporal robustness without current-year target leakage.",
    "",
    "DATA POLICY",
    "-" * 112,
    "Input predictions: finalized Step 17 rolling-origin OOF predictions.",
    "Development validation years: 2019-2023.",
    "2024 reserved validation: NOT loaded.",
    "2025 final lockbox: NOT loaded.",
    "",
    "ELIGIBLE STRATEGIES",
    "-" * 112,
]

for strategy in ELIGIBLE_STRATEGIES:
    report_lines.append(
        strategy
    )

report_lines.extend(
    [
        "",
        "CAUSAL WEIGHT POLICY",
        "-" * 112,
        "For validation year t, inverse-MAE/RMSE ensemble weights use only",
        "earlier rolling-origin validation years y < t.",
        "2019 has no prior OOF history and therefore uses equal model weights.",
        "",
        "RESIDUAL CORRELATION",
        "-" * 112,
        (
            "Pooled mean absolute pairwise residual correlation: "
            f"{pooled_corr['absolute_correlation'].mean():.6f}"
        ),
        (
            "Pooled minimum absolute pairwise residual correlation: "
            f"{pooled_corr['absolute_correlation'].min():.6f}"
        ),
        (
            "Pooled maximum absolute pairwise residual correlation: "
            f"{pooled_corr['absolute_correlation'].max():.6f}"
        ),
        "",
        "DESCRIPTIVE STEP 19B4 RESULT",
        "-" * 112,
    ]
)

best_row = descriptive_best.iloc[
    0
]

report_lines.append(
    f"Lowest all-fold development MAE strategy: "
    f"{best_row['strategy']} | "
    f"mean MAE={best_row['all_folds_mean_mae']:.6f} °C | "
    f"mean RMSE={best_row['all_folds_mean_rmse']:.6f} °C | "
    f"mean R²={best_row['all_folds_mean_r2']:.6f} | "
    f"negative-R² folds="
    f"{int(best_row['all_folds_negative_r2_count'])} | "
    f"worst MAE={best_row['all_folds_worst_mae']:.6f} °C | "
    f"minimum R²={best_row['all_folds_min_r2']:.6f} | "
    f"MAE improvement vs RF="
    f"{best_row['all_folds_mae_improvement_vs_rf']:.6f} °C."
)

report_lines.extend(
    [
        "",
        "GENERALIZATION GAP",
        "-" * 112,
        "Not reported for ensemble strategies.",
        "This step combines out-of-sample validation predictions only, and",
        "comparable ensemble training predictions are unavailable. A gap is",
        "therefore intentionally not fabricated.",
        "",
        "ENSEMBLE DISAGREEMENT",
        "-" * 112,
        "Across-member prediction spread is diagnostic only.",
        "It is NOT a calibrated predictive uncertainty interval.",
        "",
        "NO FINAL SELECTION",
        "-" * 112,
        "Any descriptive ranking here is development evidence only.",
        "Step 19C must compare the legitimate Step 19 strategies before",
        "reserved-2024 validation.",
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
# 15. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "11. REPRODUCIBILITY HASHES"
)

hash_files = [
    STEP17_PREDICTIONS_PATH,
    STEP17_FOLD_METRICS_PATH,
    STEP17_AUDIT_PATH,
    STEP19A_AUDIT_PATH,
    STEP19B1_AUDIT_PATH,
    STEP19B2_AUDIT_PATH,
    STEP19B3_AUDIT_PATH,
    MEMBER_METRICS_PATH,
    ERROR_CORRELATION_PATH,
    CAUSAL_WEIGHTS_PATH,
    FOLD_METRICS_PATH,
    SUMMARY_PATH,
    DESCRIPTIVE_BEST_PATH,
    DISAGREEMENT_PATH,
    PREDICTIONS_PATH,
    REFERENCE_REPRO_PATH,
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
# 16. FINAL STATUS
# =============================================================================

section(
    "STEP 19B4 COMPLETED SUCCESSFULLY"
)

print(
    "Robust temporal ensemble experiment completed.\n"
    "Finalized Step 17 rolling-origin predictions were reproduced first.\n"
    "Fixed mean, median, and trimmed ensembles were evaluated.\n"
    "Causal inverse-MAE and inverse-RMSE weights used earlier OOF years only.\n"
    "2019 causal weighting used the predeclared equal-weight fallback.\n"
    "Member residual correlations and ensemble disagreement were diagnosed.\n"
    "No ensemble generalization gap was fabricated from unavailable train predictions.\n"
    "2024 reserved validation remained untouched.\n"
    "2025 final lockbox remained untouched.\n"
    "No model training, retuning, feature selection, bias correction, or final\n"
    "model selection occurred.\n"
    "Review the complete output before deciding whether Step 19B5 is justified."
)
