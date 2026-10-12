
"""
STEP 19B3 - RECENT-YEAR WEIGHTING EXPERIMENT
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Test whether giving more importance to recent training years improves
strict rolling-origin temporal generalization.

BACKGROUND
----------
Step 19A showed strong year-level temporal instability.
Step 19B1 showed simple prior residual bias correction was not robust.
Step 19B2 showed train-only stable-feature filtering did not improve
temporal generalization.

Step 19B3 therefore tests a different hypothesis:

    Older training years may be less representative of the next year.
    Giving recent years higher training weight may help the model adapt.

IMPORTANT
---------
This is a DEVELOPMENT experiment only.

- Uses 2018-2023 development data only.
- Uses the full 93-feature configuration.
- Uses frozen Step 17 tuned hyperparameters.
- Uses strict expanding rolling-origin folds.
- Uses only training-year information to construct sample weights.
- 2024 is NOT loaded or used.
- 2025 is NOT loaded or used.
- No hyperparameter retuning.
- No stable-feature subset.
- No bias correction.
- No ensemble.
- No final model selection.

PREDECLARED WEIGHTING STRATEGIES
--------------------------------
1. uniform_reference
   Every training row receives equal weight AND the model is refit through
   the same sample_weight code path used by all recency strategies.
   This is the fair within-19B3 baseline.

   The original Step 17 unweighted prediction is still reproduced separately
   as an external reference, but it is NOT used as the weighting baseline.

2. exp_half_life_4y
   Mild recency weighting:
       raw weight = 0.5 ** (age_in_years / 4)

3. exp_half_life_2y
   Moderate recency weighting:
       raw weight = 0.5 ** (age_in_years / 2)

4. exp_half_life_1y
   Strong recency weighting:
       raw weight = 0.5 ** age_in_years

For every rolling fold:
    age_in_years = latest_training_year - row_year

Weights are normalized to mean 1 inside each fold. This keeps overall
sample-weight scale comparable while changing relative importance.

2019 has only one training year (2018), so every weighting strategy is
identical to the uniform reference in that fold. Primary recency-weighting
comparison is therefore also reported for 2020-2023.
"""

from pathlib import Path
import gc
import hashlib
import inspect
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore


# =============================================================================
# 1. SETTINGS
# =============================================================================

TARGET = "LST_C"
GRID_COL = "grid_id"
AREA_COL = "area_name"
YEAR_COL = "year"

RANDOM_STATE = 42

DEVELOPMENT_YEARS = [
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
]

VALIDATION_YEARS = [
    2019,
    2020,
    2021,
    2022,
    2023,
]

WEIGHTING_ACTIVE_YEARS = [
    2020,
    2021,
    2022,
    2023,
]

EXPECTED_UNIQUE_GRIDS = 2194
EXPECTED_PREDICTOR_COUNT = 93

MODEL_ORDER = [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "NGBoost",
]

STRATEGY_HALF_LIVES = {
    "uniform_reference": None,
    "exp_half_life_4y": 4.0,
    "exp_half_life_2y": 2.0,
    "exp_half_life_1y": 1.0,
}

STRATEGIES = list(
    STRATEGY_HALF_LIVES.keys()
)

METRIC_TOLERANCE = 1e-9
WEIGHT_TOLERANCE = 1e-12


# =============================================================================
# 2. PATHS
# =============================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

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

WEIGHT_TABLE_PATH = (
    METRICS_DIR
    / "19b3_recency_weight_table.csv"
)

FOLD_METRICS_PATH = (
    METRICS_DIR
    / "19b3_recent_year_weighting_fold_metrics.csv"
)

SUMMARY_PATH = (
    METRICS_DIR
    / "19b3_recent_year_weighting_summary.csv"
)

DESCRIPTIVE_BEST_PATH = (
    METRICS_DIR
    / "19b3_descriptive_best_weighting_by_model.csv"
)

PREDICTIONS_PATH = (
    PREDICTIONS_DIR
    / "19b3_recent_year_weighting_predictions.csv"
)

BASELINE_REPRO_PATH = (
    REPORTS_DIR
    / "19b3_step17_reference_reproduction_check.csv"
)

UNIFORM_VS_STEP17_PATH = (
    METRICS_DIR
    / "19b3_uniform_sample_weight_vs_step17.csv"
)

SAMPLE_WEIGHT_SUPPORT_PATH = (
    REPORTS_DIR
    / "19b3_sample_weight_support_audit.csv"
)

AUDIT_PATH = (
    REPORTS_DIR
    / "19b3_recent_year_weighting_audit.csv"
)

REPORT_PATH = (
    REPORTS_DIR
    / "19b3_recent_year_weighting_report.txt"
)

HASH_PATH = (
    REPORTS_DIR
    / "19b3_recent_year_weighting_hashes.csv"
)

FIG_MAE_PATH = (
    FIGURES_DIR
    / "19b3_recent_year_weighting_mean_mae.png"
)

FIG_R2_PATH = (
    FIGURES_DIR
    / "19b3_recent_year_weighting_r2_by_year.png"
)

FIG_BIAS_PATH = (
    FIGURES_DIR
    / "19b3_recent_year_weighting_abs_year_bias.png"
)

FIG_WEIGHTS_PATH = (
    FIGURES_DIR
    / "19b3_recency_weight_profiles.png"
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


def py_value(
    value,
):
    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    return value


def clean_params(
    params: dict,
) -> dict:
    return {
        str(
            key
        ): py_value(
            value
        )
        for key, value in params.items()
    }


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
        > 1e-12
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
        "mean_residual_actual_minus_predicted": float(
            np.mean(
                residual
            )
        ),
        "mean_prediction_bias_pred_minus_actual": float(
            np.mean(
                y_pred
                - y_true
            )
        ),
        "abs_mean_prediction_bias": float(
            abs(
                np.mean(
                    y_pred
                    - y_true
                )
            )
        ),
    }


def build_model(
    model_name: str,
    params: dict,
):
    p = clean_params(
        params
    )

    if model_name == "Random Forest":
        return RandomForestRegressor(
            **p,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "XGBoost":
        return XGBRegressor(
            **p,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
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

    if model_name == "NGBoost":
        base = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=int(
                p[
                    "base_max_depth"
                ]
            ),
            min_samples_split=int(
                p[
                    "base_min_samples_split"
                ]
            ),
            min_samples_leaf=int(
                p[
                    "base_min_samples_leaf"
                ]
            ),
            random_state=RANDOM_STATE,
        )

        return NGBRegressor(
            Dist=Normal,
            Score=LogScore,
            Base=base,
            natural_gradient=True,
            n_estimators=int(
                p[
                    "n_estimators"
                ]
            ),
            learning_rate=float(
                p[
                    "learning_rate"
                ]
            ),
            minibatch_frac=float(
                p[
                    "minibatch_frac"
                ]
            ),
            col_sample=float(
                p[
                    "col_sample"
                ]
            ),
            verbose=False,
            random_state=RANDOM_STATE,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def build_sample_weights(
    train_year_series,
    half_life,
):
    years = np.asarray(
        train_year_series,
        dtype=int,
    )

    latest_training_year = int(
        np.max(
            years
        )
    )

    age = (
        latest_training_year
        - years
    ).astype(
        float
    )

    if half_life is None:
        raw_weights = np.ones(
            len(
                years
            ),
            dtype=float,
        )
    else:
        raw_weights = (
            0.5
            ** (
                age
                / float(
                    half_life
                )
            )
        )

    mean_weight = float(
        np.mean(
            raw_weights
        )
    )

    if (
        not np.isfinite(
            mean_weight
        )
        or mean_weight
        <= 0
    ):
        raise ValueError(
            "Invalid sample-weight mean."
        )

    normalized_weights = (
        raw_weights
        / mean_weight
    )

    return (
        normalized_weights,
        raw_weights,
        age,
        latest_training_year,
    )


def effective_sample_size(
    weights,
) -> float:
    weights = np.asarray(
        weights,
        dtype=float,
    )

    numerator = (
        np.sum(
            weights
        )
        ** 2
    )

    denominator = np.sum(
        weights
        ** 2
    )

    if denominator <= 0:
        return np.nan

    return float(
        numerator
        / denominator
    )


# =============================================================================
# 4. PREREQUISITE AUDITS
# =============================================================================

section(
    "STEP 19B3 - RECENT-YEAR WEIGHTING"
)

required_files = [
    INPUT_PATH,
    STEP17_PREDICTIONS_PATH,
    STEP17_FOLD_METRICS_PATH,
    STEP17_PARAMS_PATH,
    STEP17_AUDIT_PATH,
    STEP19A_AUDIT_PATH,
    STEP19B1_AUDIT_PATH,
    STEP19B2_AUDIT_PATH,
]

for path in required_files:
    if not path.exists():
        print(
            f"\nERROR: required file missing: {path}"
        )
        sys.exit(1)

check_audit_all_pass(
    STEP17_AUDIT_PATH,
    "Step 17",
)

check_audit_all_pass(
    STEP19A_AUDIT_PATH,
    "Step 19A",
)

check_audit_all_pass(
    STEP19B1_AUDIT_PATH,
    "Step 19B1",
)

check_audit_all_pass(
    STEP19B2_AUDIT_PATH,
    "Step 19B2",
)

print(
    "PASS: Step 17 audit fully PASS."
)
print(
    "PASS: Step 19A audit fully PASS."
)
print(
    "PASS: Step 19B1 audit fully PASS."
)
print(
    "PASS: Step 19B2 audit fully PASS."
)
print(
    "19B3 changes sample weights only."
)
print(
    "2024 and 2025 are not loaded."
)


# =============================================================================
# 5. LOAD DEVELOPMENT DATA
# =============================================================================

section(
    "1. LOAD 2018-2023 DEVELOPMENT DATA"
)

df = pd.read_csv(
    INPUT_PATH,
    dtype={
        GRID_COL: str,
    },
)

required_columns = {
    TARGET,
    GRID_COL,
    AREA_COL,
    YEAR_COL,
}

missing_required = sorted(
    required_columns
    - set(
        df.columns
    )
)

if missing_required:
    print(
        f"\nERROR: missing required columns: {missing_required}"
    )
    sys.exit(1)

years = sorted(
    pd.to_numeric(
        df[
            YEAR_COL
        ],
        errors="coerce",
    )
    .astype(int)
    .unique()
    .tolist()
)

if years != DEVELOPMENT_YEARS:
    print(
        f"\nERROR: development years={years}; "
        f"expected={DEVELOPMENT_YEARS}."
    )
    sys.exit(1)

if (
    df[
        GRID_COL
    ].nunique()
    != EXPECTED_UNIQUE_GRIDS
):
    print(
        f"\nERROR: unique grids={df[GRID_COL].nunique()}; "
        f"expected={EXPECTED_UNIQUE_GRIDS}."
    )
    sys.exit(1)

predictors = [
    column
    for column in df.columns
    if column not in {
        TARGET,
        GRID_COL,
        AREA_COL,
    }
]

if len(
    predictors
) != EXPECTED_PREDICTOR_COUNT:
    print(
        f"\nERROR: predictor count={len(predictors)}; "
        f"expected={EXPECTED_PREDICTOR_COUNT}."
    )
    sys.exit(1)

for feature in predictors:
    df[
        feature
    ] = pd.to_numeric(
        df[
            feature
        ],
        errors="coerce",
    )

df[
    TARGET
] = pd.to_numeric(
    df[
        TARGET
    ],
    errors="coerce",
)

if (
    df[
        predictors
    ]
    .isna()
    .any()
    .any()
):
    print(
        "\nERROR: missing predictor values found."
    )
    sys.exit(1)

if not np.isfinite(
    df[
        predictors
    ]
    .to_numpy(
        dtype=float
    )
).all():
    print(
        "\nERROR: non-finite predictor values found."
    )
    sys.exit(1)

if not np.isfinite(
    df[
        TARGET
    ]
    .to_numpy(
        dtype=float
    )
).all():
    print(
        "\nERROR: non-finite target values found."
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
    "2024 data          : NOT LOADED"
)
print(
    "2025 data          : NOT LOADED"
)


# =============================================================================
# 6. LOAD FROZEN STEP 17 CONFIGURATION
# =============================================================================

section(
    "2. LOAD FROZEN STEP 17 CONFIGURATION"
)

with STEP17_PARAMS_PATH.open(
    "r",
    encoding="utf-8",
) as file_obj:
    params_payload = json.load(
        file_obj
    )

if "best_params" not in params_payload:
    print(
        "\nERROR: Step 17 best-hyperparameter JSON has no "
        "'best_params' key."
    )
    sys.exit(1)

best_params = params_payload[
    "best_params"
]

if sorted(
    best_params.keys()
) != sorted(
    MODEL_ORDER
):
    print(
        "\nERROR: Step 17 best-parameter model names do not "
        "match expected models."
    )
    sys.exit(1)

for model_name in MODEL_ORDER:
    print(
        f"{model_name}: "
        f"{json.dumps(clean_params(best_params[model_name]), sort_keys=True)}"
    )

print(
    "\nPASS: frozen Step 17 hyperparameters loaded."
)
print(
    "No hyperparameter retuning will occur."
)


# =============================================================================
# 7. SAMPLE-WEIGHT API SUPPORT CHECK
# =============================================================================

section(
    "3. SAMPLE-WEIGHT SUPPORT CHECK"
)

support_rows = []

for model_name in MODEL_ORDER:
    model = build_model(
        model_name,
        best_params[
            model_name
        ],
    )

    fit_signature = inspect.signature(
        model.fit
    )

    supports_sample_weight = bool(
        "sample_weight"
        in fit_signature.parameters
    )

    support_rows.append(
        {
            "model": model_name,
            "fit_signature": str(
                fit_signature
            ),
            "sample_weight_supported": (
                supports_sample_weight
            ),
        }
    )

    print(
        f"{model_name:<14} | "
        f"sample_weight_supported="
        f"{supports_sample_weight}"
    )

    del model
    gc.collect()

support_df = pd.DataFrame(
    support_rows
)

support_df.to_csv(
    SAMPLE_WEIGHT_SUPPORT_PATH,
    index=False,
)

if not support_df[
    "sample_weight_supported"
].all():
    print(
        "\nERROR: at least one installed model implementation "
        "does not expose sample_weight in fit()."
    )
    print(
        "Do not silently substitute row duplication or another "
        "weighting mechanism."
    )
    sys.exit(1)

print(
    "\nPASS: all four model implementations support sample_weight."
)


# =============================================================================
# 8. LOAD + VERIFY STEP 17 REFERENCE
# =============================================================================

section(
    "4. LOAD + VERIFY STEP 17 REFERENCE"
)

step17_pred = pd.read_csv(
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
}

missing_prediction_columns = sorted(
    required_prediction_columns
    - set(
        step17_pred.columns
    )
)

if missing_prediction_columns:
    print(
        "\nERROR: Step 17 prediction file missing: "
        f"{missing_prediction_columns}"
    )
    sys.exit(1)

step17_pred[
    "validation_year"
] = pd.to_numeric(
    step17_pred[
        "validation_year"
    ],
    errors="coerce",
).astype(
    int
)

required_fold_columns = {
    "model",
    "validation_year",
    "train_mae",
    "train_rmse",
    "train_r2",
    "train_mape_percent",
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

expected_reference_rows = (
    len(
        MODEL_ORDER
    )
    * len(
        VALIDATION_YEARS
    )
    * EXPECTED_UNIQUE_GRIDS
)

if len(
    step17_pred
) != expected_reference_rows:
    print(
        f"\nERROR: Step 17 prediction rows={len(step17_pred)}, "
        f"expected={expected_reference_rows}."
    )
    sys.exit(1)

reference_rows = []

for model_name in MODEL_ORDER:
    for validation_year in VALIDATION_YEARS:
        subset = step17_pred[
            (
                step17_pred[
                    "model"
                ]
                == model_name
            )
            & (
                step17_pred[
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

        status = (
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
        )

        reference_rows.append(
            {
                "model": model_name,
                "validation_year": validation_year,
                "mae_difference": mae_diff,
                "rmse_difference": rmse_diff,
                "r2_difference": r2_diff,
                "status": status,
            }
        )

reference_repro = pd.DataFrame(
    reference_rows
)

reference_repro.to_csv(
    BASELINE_REPRO_PATH,
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
        "\nERROR: Step 17 reference reproduction failed."
    )
    sys.exit(1)

print(
    "\nPASS: all 20 Step 17 validation predictions reproduced."
)


# =============================================================================
# 9. BUILD RECENCY WEIGHT TABLES
# =============================================================================

section(
    "5. BUILD PREDECLARED RECENCY WEIGHTS"
)

weight_rows = []
weight_lookup = {}

for validation_year in VALIDATION_YEARS:
    train_df = df[
        df[
            YEAR_COL
        ]
        < validation_year
    ]

    train_years = sorted(
        train_df[
            YEAR_COL
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    for strategy in STRATEGIES:
        half_life = (
            STRATEGY_HALF_LIVES[
                strategy
            ]
        )

        (
            normalized_weights,
            raw_weights,
            ages,
            latest_training_year,
        ) = build_sample_weights(
            train_df[
                YEAR_COL
            ],
            half_life,
        )

        weight_lookup[
            (
                validation_year,
                strategy,
            )
        ] = normalized_weights

        unique_year_table = pd.DataFrame(
            {
                YEAR_COL: train_df[
                    YEAR_COL
                ]
                .astype(int)
                .to_numpy(),
                "age_years": ages,
                "raw_weight": raw_weights,
                "normalized_weight": (
                    normalized_weights
                ),
            }
        )

        year_summary = (
            unique_year_table.groupby(
                YEAR_COL,
                as_index=False,
            )
            .agg(
                age_years=(
                    "age_years",
                    "first",
                ),
                raw_weight=(
                    "raw_weight",
                    "first",
                ),
                normalized_weight=(
                    "normalized_weight",
                    "first",
                ),
                row_count=(
                    YEAR_COL,
                    "size",
                ),
            )
        )

        for _, row in year_summary.iterrows():
            weight_rows.append(
                {
                    "validation_year": (
                        validation_year
                    ),
                    "strategy": strategy,
                    "half_life_years": (
                        half_life
                    ),
                    "latest_training_year": (
                        latest_training_year
                    ),
                    "training_year": int(
                        row[
                            YEAR_COL
                        ]
                    ),
                    "age_years": float(
                        row[
                            "age_years"
                        ]
                    ),
                    "raw_weight": float(
                        row[
                            "raw_weight"
                        ]
                    ),
                    "normalized_weight": float(
                        row[
                            "normalized_weight"
                        ]
                    ),
                    "row_count": int(
                        row[
                            "row_count"
                        ]
                    ),
                    "fold_mean_normalized_weight": float(
                        np.mean(
                            normalized_weights
                        )
                    ),
                    "fold_min_normalized_weight": float(
                        np.min(
                            normalized_weights
                        )
                    ),
                    "fold_max_normalized_weight": float(
                        np.max(
                            normalized_weights
                        )
                    ),
                    "fold_weight_ratio_max_over_min": float(
                        np.max(
                            normalized_weights
                        )
                        / np.min(
                            normalized_weights
                        )
                    ),
                    "effective_sample_size_rows": (
                        effective_sample_size(
                            normalized_weights
                        )
                    ),
                }
            )

        print(
            f"{train_years[0]}-{train_years[-1]} -> "
            f"{validation_year} | "
            f"{strategy:<20} | "
            f"min={np.min(normalized_weights):.4f} | "
            f"max={np.max(normalized_weights):.4f} | "
            f"ratio="
            f"{np.max(normalized_weights) / np.min(normalized_weights):.2f} | "
            f"ESS={effective_sample_size(normalized_weights):.1f}"
        )

weight_table = pd.DataFrame(
    weight_rows
)

weight_table.to_csv(
    WEIGHT_TABLE_PATH,
    index=False,
)

print(
    f"\nSaved: {WEIGHT_TABLE_PATH}"
)


# =============================================================================
# 10. STRICT ROLLING-ORIGIN EVALUATION
# =============================================================================

section(
    "6. STRICT ROLLING-ORIGIN WEIGHTING EVALUATION"
)

fold_metric_rows = []
prediction_frames = []

for model_name in MODEL_ORDER:
    print(
        f"\n--- {model_name} ---"
    )

    for validation_year in VALIDATION_YEARS:
        train_df = df[
            df[
                YEAR_COL
            ]
            < validation_year
        ]

        val_df = df[
            df[
                YEAR_COL
            ]
            == validation_year
        ]

        train_years = sorted(
            train_df[
                YEAR_COL
            ]
            .astype(int)
            .unique()
            .tolist()
        )

        X_train = train_df[
            predictors
        ]

        y_train = train_df[
            TARGET
        ].to_numpy(
            dtype=float
        )

        X_val = val_df[
            predictors
        ]

        y_val = val_df[
            TARGET
        ].to_numpy(
            dtype=float
        )

        for strategy in STRATEGIES:
            # ---------------------------------------------------------
            # FAIR WITHIN-19B3 COMPARISON
            # ---------------------------------------------------------
            # Every strategy, INCLUDING uniform_reference, is refit through
            # exactly the same model.fit(..., sample_weight=...) path.
            #
            # This matters because some estimators (notably Random Forest)
            # can produce tiny numerical/tree differences when sample_weight
            # is supplied even if every weight equals 1. Reusing the Step 17
            # unweighted prediction for only the uniform strategy would make
            # the comparison slightly asymmetric.
            # ---------------------------------------------------------

            sample_weights = (
                weight_lookup[
                    (
                        validation_year,
                        strategy,
                    )
                ]
            )

            if len(
                sample_weights
            ) != len(
                train_df
            ):
                print(
                    "\nERROR: sample-weight row count mismatch."
                )
                sys.exit(1)

            if (
                not np.isfinite(
                    sample_weights
                ).all()
                or np.any(
                    sample_weights
                    <= 0
                )
            ):
                print(
                    "\nERROR: invalid sample weights."
                )
                sys.exit(1)

            model = build_model(
                model_name,
                best_params[
                    model_name
                ],
            )

            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
            )

            pred_train = np.asarray(
                model.predict(
                    X_train
                ),
                dtype=float,
            )

            pred_val = np.asarray(
                model.predict(
                    X_val
                ),
                dtype=float,
            )

            train_metrics = regression_metrics(
                y_train,
                pred_train,
            )

            val_metrics = regression_metrics(
                y_val,
                pred_val,
            )

            train_mae = float(
                train_metrics[
                    "mae"
                ]
            )

            train_rmse = float(
                train_metrics[
                    "rmse"
                ]
            )

            train_r2 = float(
                train_metrics[
                    "r2"
                ]
            )

            train_mape = float(
                train_metrics[
                    "mape_percent"
                ]
            )

            source = (
                "refit_frozen_step17_model_with_sample_weights"
            )

            grid_ids = (
                val_df[
                    GRID_COL
                ]
                .astype(str)
                .to_numpy()
            )

            area_names = (
                val_df[
                    AREA_COL
                ]
                .astype(str)
                .to_numpy()
            )

            actual_values = y_val

            del model
            gc.collect()

            if not np.isfinite(
                pred_val
            ).all():
                print(
                    "\nERROR: non-finite validation predictions found."
                )
                sys.exit(1)

            latest_training_year = max(
                train_years
            )

            mean_weight = float(
                np.mean(
                    sample_weights
                )
            )

            min_weight = float(
                np.min(
                    sample_weights
                )
            )

            max_weight = float(
                np.max(
                    sample_weights
                )
            )

            fold_metric_rows.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "validation_year": (
                        validation_year
                    ),
                    "train_start_year": min(
                        train_years
                    ),
                    "train_end_year": max(
                        train_years
                    ),
                    "latest_training_year": (
                        latest_training_year
                    ),
                    "training_year_count": len(
                        train_years
                    ),
                    "train_rows": len(
                        train_df
                    ),
                    "validation_rows": len(
                        val_df
                    ),
                    "predictor_count": len(
                        predictors
                    ),
                    "half_life_years": (
                        STRATEGY_HALF_LIVES[
                            strategy
                        ]
                    ),
                    "weighting_active": bool(
                        len(
                            train_years
                        )
                        > 1
                        and strategy
                        != "uniform_reference"
                    ),
                    "mean_sample_weight": (
                        mean_weight
                    ),
                    "min_sample_weight": (
                        min_weight
                    ),
                    "max_sample_weight": (
                        max_weight
                    ),
                    "weight_ratio_max_over_min": (
                        max_weight
                        / min_weight
                    ),
                    "effective_sample_size_rows": (
                        effective_sample_size(
                            sample_weights
                        )
                    ),
                    "evaluation_source": source,
                    "train_mae": train_mae,
                    "validation_mae": float(
                        val_metrics[
                            "mae"
                        ]
                    ),
                    "mae_generalization_gap": float(
                        val_metrics[
                            "mae"
                        ]
                        - train_mae
                    ),
                    "train_rmse": train_rmse,
                    "validation_rmse": float(
                        val_metrics[
                            "rmse"
                        ]
                    ),
                    "rmse_generalization_gap": float(
                        val_metrics[
                            "rmse"
                        ]
                        - train_rmse
                    ),
                    "train_r2": train_r2,
                    "validation_r2": float(
                        val_metrics[
                            "r2"
                        ]
                    ),
                    "r2_generalization_gap": float(
                        train_r2
                        - val_metrics[
                            "r2"
                        ]
                    ),
                    "train_mape_percent": (
                        train_mape
                    ),
                    "validation_mape_percent": float(
                        val_metrics[
                            "mape_percent"
                        ]
                    ),
                    "validation_mean_prediction_bias_pred_minus_actual_c": float(
                        val_metrics[
                            "mean_prediction_bias_pred_minus_actual"
                        ]
                    ),
                    "validation_abs_mean_prediction_bias_c": float(
                        val_metrics[
                            "abs_mean_prediction_bias"
                        ]
                    ),
                    "negative_r2": bool(
                        val_metrics[
                            "r2"
                        ]
                        < 0
                    ),
                }
            )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "strategy": strategy,
                        "validation_year": (
                            validation_year
                        ),
                        GRID_COL: grid_ids,
                        AREA_COL: area_names,
                        "actual_lst_c": (
                            actual_values
                        ),
                        "predicted_lst_c": (
                            pred_val
                        ),
                        "residual_actual_minus_predicted": (
                            actual_values
                            - pred_val
                        ),
                    }
                )
            )

            print(
                f"{train_years[0]}-{train_years[-1]} -> "
                f"{validation_year} | "
                f"{strategy:<20} | "
                f"MAE={val_metrics['mae']:.4f} | "
                f"RMSE={val_metrics['rmse']:.4f} | "
                f"R²={val_metrics['r2']:.4f} | "
                f"Bias={val_metrics['mean_prediction_bias_pred_minus_actual']:+.3f} °C"
            )

fold_metrics = pd.DataFrame(
    fold_metric_rows
)

all_predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

fold_metrics.to_csv(
    FOLD_METRICS_PATH,
    index=False,
)

all_predictions.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

print(
    f"\nSaved: {FOLD_METRICS_PATH}"
)
print(
    f"Saved: {PREDICTIONS_PATH}"
)

# -----------------------------------------------------------------------------
# Transparent comparison between:
#   A) finalized Step 17 unweighted fit
#   B) 19B3 uniform sample_weight=1 refit
#
# This is NOT the within-19B3 weighting baseline comparison. It is saved only
# to show any implementation-path difference introduced by supplying
# sample_weight, even when every weight equals 1.
# -----------------------------------------------------------------------------

uniform_vs_step17_rows = []

for model_name in MODEL_ORDER:
    for validation_year in VALIDATION_YEARS:
        uniform_row = fold_metrics[
            (
                fold_metrics[
                    "model"
                ]
                == model_name
            )
            & (
                fold_metrics[
                    "strategy"
                ]
                == "uniform_reference"
            )
            & (
                fold_metrics[
                    "validation_year"
                ]
                == validation_year
            )
        ]

        ref_row = step17_fold[
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

        if len(uniform_row) != 1 or len(ref_row) != 1:
            print(
                "\nERROR: uniform-vs-Step17 comparison row mismatch."
            )
            sys.exit(1)

        uniform_row = uniform_row.iloc[0]
        ref_row = ref_row.iloc[0]

        uniform_vs_step17_rows.append(
            {
                "model": model_name,
                "validation_year": validation_year,
                "step17_unweighted_mae": float(
                    ref_row[
                        "validation_mae"
                    ]
                ),
                "uniform_sample_weight_mae": float(
                    uniform_row[
                        "validation_mae"
                    ]
                ),
                "mae_difference_uniform_minus_step17": float(
                    uniform_row[
                        "validation_mae"
                    ]
                    - ref_row[
                        "validation_mae"
                    ]
                ),
                "step17_unweighted_rmse": float(
                    ref_row[
                        "validation_rmse"
                    ]
                ),
                "uniform_sample_weight_rmse": float(
                    uniform_row[
                        "validation_rmse"
                    ]
                ),
                "rmse_difference_uniform_minus_step17": float(
                    uniform_row[
                        "validation_rmse"
                    ]
                    - ref_row[
                        "validation_rmse"
                    ]
                ),
                "step17_unweighted_r2": float(
                    ref_row[
                        "validation_r2"
                    ]
                ),
                "uniform_sample_weight_r2": float(
                    uniform_row[
                        "validation_r2"
                    ]
                ),
                "r2_difference_uniform_minus_step17": float(
                    uniform_row[
                        "validation_r2"
                    ]
                    - ref_row[
                        "validation_r2"
                    ]
                ),
            }
        )

uniform_vs_step17 = pd.DataFrame(
    uniform_vs_step17_rows
)

uniform_vs_step17.to_csv(
    UNIFORM_VS_STEP17_PATH,
    index=False,
)

print(
    f"Saved: {UNIFORM_VS_STEP17_PATH}"
)


# =============================================================================
# 11. STRATEGY SUMMARY
# =============================================================================

section(
    "7. RECENT-YEAR WEIGHTING SUMMARY"
)

summary_rows = []

for model_name in MODEL_ORDER:
    reference_active = fold_metrics[
        (
            fold_metrics[
                "model"
            ]
            == model_name
        )
        & (
            fold_metrics[
                "strategy"
            ]
            == "uniform_reference"
        )
        & (
            fold_metrics[
                "validation_year"
            ]
            .isin(
                WEIGHTING_ACTIVE_YEARS
            )
        )
    ]

    reference_mean_mae = float(
        reference_active[
            "validation_mae"
        ].mean()
    )

    reference_mean_rmse = float(
        reference_active[
            "validation_rmse"
        ].mean()
    )

    reference_mean_r2 = float(
        reference_active[
            "validation_r2"
        ].mean()
    )

    reference_mean_abs_bias = float(
        reference_active[
            "validation_abs_mean_prediction_bias_c"
        ].mean()
    )

    reference_negative_count = int(
        reference_active[
            "negative_r2"
        ].sum()
    )

    for strategy in STRATEGIES:
        subset_all = fold_metrics[
            (
                fold_metrics[
                    "model"
                ]
                == model_name
            )
            & (
                fold_metrics[
                    "strategy"
                ]
                == strategy
            )
        ]

        subset_active = subset_all[
            subset_all[
                "validation_year"
            ]
            .isin(
                WEIGHTING_ACTIVE_YEARS
            )
        ]

        if len(
            subset_all
        ) != len(
            VALIDATION_YEARS
        ):
            print(
                "\nERROR: incomplete all-fold summary input."
            )
            sys.exit(1)

        if len(
            subset_active
        ) != len(
            WEIGHTING_ACTIVE_YEARS
        ):
            print(
                "\nERROR: incomplete weighting-active summary input."
            )
            sys.exit(1)

        mean_mae = float(
            subset_active[
                "validation_mae"
            ].mean()
        )

        mean_rmse = float(
            subset_active[
                "validation_rmse"
            ].mean()
        )

        mean_r2 = float(
            subset_active[
                "validation_r2"
            ].mean()
        )

        mean_abs_bias = float(
            subset_active[
                "validation_abs_mean_prediction_bias_c"
            ].mean()
        )

        negative_count = int(
            subset_active[
                "negative_r2"
            ].sum()
        )

        summary_rows.append(
            {
                "model": model_name,
                "strategy": strategy,
                "half_life_years": (
                    STRATEGY_HALF_LIVES[
                        strategy
                    ]
                ),
                "all_folds_mean_mae": float(
                    subset_all[
                        "validation_mae"
                    ].mean()
                ),
                "all_folds_std_mae": float(
                    subset_all[
                        "validation_mae"
                    ].std(
                        ddof=0
                    )
                ),
                "all_folds_mean_rmse": float(
                    subset_all[
                        "validation_rmse"
                    ].mean()
                ),
                "all_folds_mean_r2": float(
                    subset_all[
                        "validation_r2"
                    ].mean()
                ),
                "all_folds_negative_r2_count": int(
                    subset_all[
                        "negative_r2"
                    ].sum()
                ),
                "all_folds_worst_mae": float(
                    subset_all[
                        "validation_mae"
                    ].max()
                ),
                "all_folds_mean_abs_year_bias_c": float(
                    subset_all[
                        "validation_abs_mean_prediction_bias_c"
                    ].mean()
                ),
                "active_2020_2023_mean_mae": (
                    mean_mae
                ),
                "active_2020_2023_std_mae": float(
                    subset_active[
                        "validation_mae"
                    ].std(
                        ddof=0
                    )
                ),
                "active_2020_2023_mean_rmse": (
                    mean_rmse
                ),
                "active_2020_2023_mean_r2": (
                    mean_r2
                ),
                "active_2020_2023_negative_r2_count": (
                    negative_count
                ),
                "active_2020_2023_worst_mae": float(
                    subset_active[
                        "validation_mae"
                    ].max()
                ),
                "active_2020_2023_mean_generalization_gap": float(
                    subset_active[
                        "mae_generalization_gap"
                    ].mean()
                ),
                "active_2020_2023_mean_abs_year_bias_c": (
                    mean_abs_bias
                ),
                "mean_mae_improvement_vs_uniform_2020_2023": (
                    reference_mean_mae
                    - mean_mae
                ),
                "mean_rmse_improvement_vs_uniform_2020_2023": (
                    reference_mean_rmse
                    - mean_rmse
                ),
                "mean_r2_improvement_vs_uniform_2020_2023": (
                    mean_r2
                    - reference_mean_r2
                ),
                "mean_abs_bias_improvement_vs_uniform_2020_2023": (
                    reference_mean_abs_bias
                    - mean_abs_bias
                ),
                "negative_r2_count_change_vs_uniform_2020_2023": (
                    negative_count
                    - reference_negative_count
                ),
            }
        )

summary = pd.DataFrame(
    summary_rows
)

summary[
    "within_model_19b3_mae_rank"
] = (
    summary.groupby(
        "model"
    )[
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
        "model",
        "within_model_19b3_mae_rank",
        "active_2020_2023_worst_mae",
        "active_2020_2023_mean_rmse",
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
            "model",
            "active_2020_2023_mean_mae",
            "active_2020_2023_worst_mae",
            "active_2020_2023_mean_rmse",
            "active_2020_2023_negative_r2_count",
        ]
    )
    .groupby(
        "model",
        as_index=False,
    )
    .first()
)

descriptive_best.to_csv(
    DESCRIPTIVE_BEST_PATH,
    index=False,
)

print(
    "\nDESCRIPTIVE BEST WEIGHTING STRATEGY WITHIN 19B3 "
    "(NOT final model selection):"
)

print(
    descriptive_best[
        [
            "model",
            "strategy",
            "active_2020_2023_mean_mae",
            "active_2020_2023_mean_rmse",
            "active_2020_2023_mean_r2",
            "active_2020_2023_negative_r2_count",
            "active_2020_2023_worst_mae",
            "active_2020_2023_mean_abs_year_bias_c",
            "mean_mae_improvement_vs_uniform_2020_2023",
            "mean_abs_bias_improvement_vs_uniform_2020_2023",
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

# ---- mean MAE over active years ----
fig, ax = plt.subplots(
    figsize=(
        12,
        7,
    )
)

x = np.arange(
    len(
        STRATEGIES
    )
)

width = (
    0.8
    / len(
        MODEL_ORDER
    )
)

for model_index, model_name in enumerate(
    MODEL_ORDER
):
    model_summary = (
        summary[
            summary[
                "model"
            ]
            == model_name
        ]
        .set_index(
            "strategy"
        )
        .reindex(
            STRATEGIES
        )
    )

    offset = (
        model_index
        - (
            len(
                MODEL_ORDER
            )
            - 1
        )
        / 2
    ) * width

    ax.bar(
        x
        + offset,
        model_summary[
            "active_2020_2023_mean_mae"
        ],
        width,
        label=model_name,
    )

ax.set_xticks(
    x,
    STRATEGIES,
    rotation=18,
    ha="right",
)

ax.set_ylabel(
    "Mean MAE for weighting-active years 2020-2023 (°C)"
)

ax.set_title(
    "Step 19B3 - Recent-Year Weighting Comparison"
)

ax.grid(
    axis="y",
    alpha=0.2,
)

ax.legend()

plt.tight_layout()

plt.savefig(
    FIG_MAE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ---- mean R² by year across models ----
fig, ax = plt.subplots(
    figsize=(
        11,
        7,
    )
)

for strategy in STRATEGIES:
    strategy_mean = (
        fold_metrics[
            fold_metrics[
                "strategy"
            ]
            == strategy
        ]
        .groupby(
            "validation_year",
            as_index=False,
        )[
            "validation_r2"
        ]
        .mean()
    )

    ax.plot(
        strategy_mean[
            "validation_year"
        ],
        strategy_mean[
            "validation_r2"
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
    "Mean R² across models"
)

ax.set_title(
    "Step 19B3 - Mean Temporal R² by Weighting Strategy"
)

ax.grid(
    alpha=0.2,
)

ax.legend()

plt.tight_layout()

plt.savefig(
    FIG_R2_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ---- mean absolute annual bias ----
bias_plot = (
    summary.groupby(
        "strategy",
        as_index=False,
    )[
        "active_2020_2023_mean_abs_year_bias_c"
    ]
    .mean()
    .set_index(
        "strategy"
    )
    .reindex(
        STRATEGIES
    )
)

fig, ax = plt.subplots(
    figsize=(
        10,
        6,
    )
)

ax.bar(
    bias_plot.index,
    bias_plot[
        "active_2020_2023_mean_abs_year_bias_c"
    ],
)

ax.set_ylabel(
    "Mean absolute annual prediction bias, 2020-2023 (°C)"
)

ax.set_title(
    "Step 19B3 - Annual-Level Bias by Weighting Strategy"
)

ax.tick_params(
    axis="x",
    rotation=18,
)

ax.grid(
    axis="y",
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    FIG_BIAS_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ---- 2023 fold weight profiles ----
plot_weight_table = weight_table[
    weight_table[
        "validation_year"
    ]
    == 2023
].copy()

fig, ax = plt.subplots(
    figsize=(
        10,
        6,
    )
)

for strategy in STRATEGIES:
    sub = (
        plot_weight_table[
            plot_weight_table[
                "strategy"
            ]
            == strategy
        ]
        .sort_values(
            "training_year"
        )
    )

    ax.plot(
        sub[
            "training_year"
        ],
        sub[
            "normalized_weight"
        ],
        marker="o",
        label=strategy,
    )

ax.set_xlabel(
    "Training year"
)

ax.set_ylabel(
    "Normalized sample weight"
)

ax.set_title(
    "Step 19B3 - Recency Weight Profiles for 2023 Forecast Origin"
)

ax.grid(
    alpha=0.2,
)

ax.legend()

plt.tight_layout()

plt.savefig(
    FIG_WEIGHTS_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

for figure_path in [
    FIG_MAE_PATH,
    FIG_R2_PATH,
    FIG_BIAS_PATH,
    FIG_WEIGHTS_PATH,
]:
    print(
        f"Saved: {figure_path}"
    )


# =============================================================================
# 13. STRICT AUDIT
# =============================================================================

section(
    "9. STRICT STEP 19B3 AUDIT"
)

expected_reference_repro_rows = (
    len(
        MODEL_ORDER
    )
    * len(
        VALIDATION_YEARS
    )
)

expected_fold_metric_rows = (
    len(
        MODEL_ORDER
    )
    * len(
        STRATEGIES
    )
    * len(
        VALIDATION_YEARS
    )
)

expected_prediction_rows = (
    EXPECTED_UNIQUE_GRIDS
    * expected_fold_metric_rows
)

expected_summary_rows = (
    len(
        MODEL_ORDER
    )
    * len(
        STRATEGIES
    )
)

expected_weight_table_rows = sum(
    len(
        [
            year
            for year in DEVELOPMENT_YEARS
            if year < validation_year
        ]
    )
    * len(
        STRATEGIES
    )
    for validation_year in VALIDATION_YEARS
)

all_weight_means_one = bool(
    (
        weight_table[
            "fold_mean_normalized_weight"
        ]
        .sub(
            1.0
        )
        .abs()
        <= WEIGHT_TOLERANCE
    ).all()
)

all_weights_positive = bool(
    (
        weight_table[
            "normalized_weight"
        ]
        > 0
    ).all()
)

no_future_weight_years = bool(
    (
        weight_table[
            "training_year"
        ]
        < weight_table[
            "validation_year"
        ]
    ).all()
)

uniform_weights_identity = bool(
    (
        weight_table[
            weight_table[
                "strategy"
            ]
            == "uniform_reference"
        ][
            "normalized_weight"
        ]
        .sub(
            1.0
        )
        .abs()
        <= WEIGHT_TOLERANCE
    ).all()
)

year_2019_all_identity = bool(
    (
        weight_table[
            weight_table[
                "validation_year"
            ]
            == 2019
        ][
            "normalized_weight"
        ]
        .sub(
            1.0
        )
        .abs()
        <= WEIGHT_TOLERANCE
    ).all()
)

all_metrics_finite = bool(
    np.isfinite(
        fold_metrics[
            [
                "train_mae",
                "validation_mae",
                "mae_generalization_gap",
                "train_rmse",
                "validation_rmse",
                "train_r2",
                "validation_r2",
                "validation_mape_percent",
                "validation_mean_prediction_bias_pred_minus_actual_c",
                "validation_abs_mean_prediction_bias_c",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    ).all()
)

# The original Step 17 fit did not explicitly pass sample_weight.
# Step 19B3 refits ALL strategies through sample_weight so the weighting
# comparison is symmetric. Therefore exact equality with Step 17 is NOT
# required here; the difference is recorded transparently instead.

uniform_vs_step17_finite = bool(
    np.isfinite(
        uniform_vs_step17[
            [
                "mae_difference_uniform_minus_step17",
                "rmse_difference_uniform_minus_step17",
                "r2_difference_uniform_minus_step17",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    ).all()
)

# 2019 has only one training year. All four strategies therefore have
# exactly the same weight vector and, because they are now refit through
# the same API path, should also produce the same predictions.
pred_2019 = all_predictions[
    all_predictions[
        "validation_year"
    ]
    == 2019
]

year_2019_prediction_identity = True
max_2019_prediction_difference = 0.0

for model_name in MODEL_ORDER:
    model_2019 = pred_2019[
        pred_2019[
            "model"
        ]
        == model_name
    ]

    base = (
        model_2019[
            model_2019[
                "strategy"
            ]
            == "uniform_reference"
        ]
        .sort_values(
            GRID_COL
        )[
            "predicted_lst_c"
        ]
        .to_numpy(
            dtype=float
        )
    )

    for strategy in STRATEGIES:
        current = (
            model_2019[
                model_2019[
                    "strategy"
                ]
                == strategy
            ]
            .sort_values(
                GRID_COL
            )[
                "predicted_lst_c"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if len(current) != len(base):
            year_2019_prediction_identity = False
            continue

        current_max_diff = float(
            np.max(
                np.abs(
                    current
                    - base
                )
            )
        )

        max_2019_prediction_difference = max(
            max_2019_prediction_difference,
            current_max_diff,
        )

        if current_max_diff > METRIC_TOLERANCE:
            year_2019_prediction_identity = False

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
        "check": "sample_weight_support_all_models",
        "status": (
            "PASS"
            if support_df[
                "sample_weight_supported"
            ].all()
            else "FAIL"
        ),
        "details": (
            f"supported="
            f"{int(support_df['sample_weight_supported'].sum())}/"
            f"{len(MODEL_ORDER)}"
        ),
    },
    {
        "check": "step17_reference_reproduction",
        "status": (
            "PASS"
            if len(
                reference_repro
            )
            == expected_reference_repro_rows
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
            f"expected={expected_reference_repro_rows}"
        ),
    },
    {
        "check": "weight_table_rows",
        "status": (
            "PASS"
            if len(
                weight_table
            )
            == expected_weight_table_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(weight_table)}, "
            f"expected={expected_weight_table_rows}"
        ),
    },
    {
        "check": "weights_use_training_years_only",
        "status": (
            "PASS"
            if no_future_weight_years
            else "FAIL"
        ),
        "details": (
            "Every weight source year is earlier than "
            "its validation year."
        ),
    },
    {
        "check": "normalized_weight_mean_is_one",
        "status": (
            "PASS"
            if all_weight_means_one
            else "FAIL"
        ),
        "details": (
            "Every fold-strategy normalized weight vector "
            "has mean 1."
        ),
    },
    {
        "check": "all_weights_positive",
        "status": (
            "PASS"
            if all_weights_positive
            else "FAIL"
        ),
        "details": str(
            all_weights_positive
        ),
    },
    {
        "check": "uniform_strategy_weights_are_one",
        "status": (
            "PASS"
            if uniform_weights_identity
            else "FAIL"
        ),
        "details": (
            "uniform_reference uses weight=1 for all training rows."
        ),
    },
    {
        "check": "2019_all_weight_vectors_identity",
        "status": (
            "PASS"
            if year_2019_all_identity
            else "FAIL"
        ),
        "details": (
            "2019 has only 2018 training rows, so every "
            "recency strategy uses the same all-ones weights."
        ),
    },
    {
        "check": "2019_all_strategy_predictions_identity",
        "status": (
            "PASS"
            if year_2019_prediction_identity
            else "FAIL"
        ),
        "details": (
            f"max_abs_prediction_difference="
            f"{max_2019_prediction_difference:.12f}"
        ),
    },
    {
        "check": "uniform_sample_weight_vs_step17_recorded",
        "status": (
            "PASS"
            if uniform_vs_step17_finite
            else "FAIL"
        ),
        "details": (
            "Step 17 remains the external unweighted reference; "
            "the 19B3 uniform baseline is refit through the same "
            "sample_weight path as weighted strategies."
        ),
    },
    {
        "check": "fold_metric_rows",
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
        "check": "prediction_rows",
        "status": (
            "PASS"
            if len(
                all_predictions
            )
            == expected_prediction_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(all_predictions)}, "
            f"expected={expected_prediction_rows}"
        ),
    },
    {
        "check": "summary_rows",
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
        "check": "all_metrics_finite",
        "status": (
            "PASS"
            if all_metrics_finite
            else "FAIL"
        ),
        "details": str(
            all_metrics_finite
        ),
    },
    {
        "check": "2024_loaded_or_used",
        "status": "PASS",
        "details": (
            "No. 2024 was not loaded by Step 19B3."
        ),
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": (
            "No. 2025 was not loaded by Step 19B3."
        ),
    },
    {
        "check": "hyperparameters_retuned",
        "status": "PASS",
        "details": (
            "No. Frozen Step 17 tuned hyperparameters were reused."
        ),
    },
    {
        "check": "feature_subset_applied",
        "status": "PASS",
        "details": (
            "No. All 93 predictors are retained."
        ),
    },
    {
        "check": "bias_correction_applied",
        "status": "PASS",
        "details": (
            "No bias correction occurs."
        ),
    },
    {
        "check": "ensemble_applied",
        "status": "PASS",
        "details": (
            "No ensemble occurs."
        ),
    },
    {
        "check": "final_model_selected",
        "status": "PASS",
        "details": (
            "No. Rankings are descriptive within Step 19B3 only."
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
        "\nERROR: one or more Step 19B3 audit checks failed."
    )
    sys.exit(1)


# =============================================================================
# 14. REPORT
# =============================================================================

section(
    "10. SAVE STEP 19B3 REPORT"
)

report_lines = [
    "STEP 19B3 - RECENT-YEAR WEIGHTING",
    "=" * 112,
    "",
    "PURPOSE",
    "-" * 112,
    "Test whether giving more training importance to recent years improves",
    "strict rolling-origin development generalization.",
    "",
    "DATA POLICY",
    "-" * 112,
    "Development data: 2018-2023 only.",
    "2024 reserved validation: NOT loaded.",
    "2025 final lockbox: NOT loaded.",
    "",
    "FEATURE / MODEL POLICY",
    "-" * 112,
    "All 93 predictors were retained.",
    "Frozen Step 17 tuned hyperparameters were reused.",
    "No retuning, feature filtering, bias correction, or ensemble occurred.",
    "",
    "PREDECLARED WEIGHTING STRATEGIES",
    "-" * 112,
    "uniform_reference",
    "exp_half_life_4y",
    "exp_half_life_2y",
    "exp_half_life_1y",
    "",
    "WEIGHT FORMULA",
    "-" * 112,
    "age = latest_training_year - row_year",
    "raw weight = 0.5 ** (age / half_life)",
    "weights are normalized to mean 1 inside each fold.",
    "",
    "PRIMARY COMPARISON WINDOW",
    "-" * 112,
    "2019 has only one training year, so all weight strategies are identical.",
    "The active recency-weighting comparison is therefore 2020-2023.",
    "",
    "DESCRIPTIVE BEST 19B3 STRATEGY BY MODEL",
    "-" * 112,
]

for _, row in descriptive_best.iterrows():
    report_lines.append(
        f"{row['model']}: {row['strategy']} | "
        f"2020-2023 mean MAE="
        f"{row['active_2020_2023_mean_mae']:.6f} °C | "
        f"mean RMSE="
        f"{row['active_2020_2023_mean_rmse']:.6f} °C | "
        f"mean R²="
        f"{row['active_2020_2023_mean_r2']:.6f} | "
        f"negative-R² folds="
        f"{int(row['active_2020_2023_negative_r2_count'])} | "
        f"worst MAE="
        f"{row['active_2020_2023_worst_mae']:.6f} °C | "
        f"mean absolute annual bias="
        f"{row['active_2020_2023_mean_abs_year_bias_c']:.6f} °C | "
        f"MAE improvement vs uniform="
        f"{row['mean_mae_improvement_vs_uniform_2020_2023']:.6f} °C."
    )

report_lines.extend(
    [
        "",
        "INTERPRETATION POLICY",
        "-" * 112,
        "A recency-weighted strategy is not accepted merely because recent",
        "years receive larger weights. It must improve rolling-origin metrics",
        "without creating worse instability.",
        "If uniform weighting remains better, that is valid evidence that",
        "simple recency emphasis does not solve the temporal regime problem.",
        "",
        "NO FINAL MODEL SELECTION",
        "-" * 112,
        "Step 19B3 is a development experiment only.",
        "Results must later be compared with the other Step 19B strategies",
        "inside Step 19C before reserved-2024 validation.",
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
    INPUT_PATH,
    STEP17_PREDICTIONS_PATH,
    STEP17_FOLD_METRICS_PATH,
    STEP17_PARAMS_PATH,
    STEP17_AUDIT_PATH,
    STEP19A_AUDIT_PATH,
    STEP19B1_AUDIT_PATH,
    STEP19B2_AUDIT_PATH,
    WEIGHT_TABLE_PATH,
    FOLD_METRICS_PATH,
    SUMMARY_PATH,
    DESCRIPTIVE_BEST_PATH,
    PREDICTIONS_PATH,
    BASELINE_REPRO_PATH,
    UNIFORM_VS_STEP17_PATH,
    SAMPLE_WEIGHT_SUPPORT_PATH,
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
    "STEP 19B3 COMPLETED SUCCESSFULLY"
)

print(
    "Recent-year weighting experiment completed.\n"
    "All weighting strategies, including the uniform reference, were refit\n"
    "through the same sample_weight API path for a fair comparison.\n"
    "The original Step 17 unweighted fit was retained only as an external\n"
    "reproduction/reference comparison.\n"
    "All models used the full 93-feature configuration.\n"
    "Frozen Step 17 tuned hyperparameters were reused without retuning.\n"
    "Recency weights were derived only from training-year age inside each fold.\n"
    "2019 reduced to equal weighting because only 2018 was available.\n"
    "Primary weighting comparison covers 2020-2023.\n"
    "Annual prediction bias was tracked in addition to MAE/RMSE/R².\n"
    "2024 reserved validation remained untouched.\n"
    "2025 final lockbox remained untouched.\n"
    "No feature subset, bias correction, ensemble, or final model selection occurred.\n"
    "Review the complete output before moving to Step 19B4."
)
