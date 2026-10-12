
"""
STEP 19B2 - TRAIN-ONLY STABLE FEATURE SUBSET EXPERIMENT
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Step 19A found that several predictors change their relationship with LST
across years. Step 19B2 tests whether a temporally stable feature subset
improves strict rolling-origin generalization.

CRITICAL LEAKAGE RULE
---------------------
For validation year t, feature stability is estimated ONLY from years < t.

Example:
    Validation 2021
    -> feature stability uses 2018, 2019, 2020 only
    -> 2021 target is not used for feature selection

    Validation 2023
    -> feature stability uses 2018-2022 only
    -> 2023 target is not used for feature selection

2024 and 2025 are NEVER loaded.

STRATEGIES
----------
1. full_93_reference
   - Uses the finalized Step 17 rolling predictions.
   - No feature reduction.

2. train_only_stable
   - Keeps `year` by policy.
   - Other predictors are retained only when their year-by-year Spearman
     relationship with LST is stable inside the TRAINING years:
         rho range <= 0.40
         meaningful sign flips == 0
   - Meaningful sign threshold = |rho| >= 0.05.

3. train_only_stable_plus_core
   - Same train-only stability rule.
   - Also protects the 16 original/domain-core predictors used by Step 08.

MINIMUM HISTORY
---------------
At least 3 training years are required to estimate temporal stability.
Therefore:
    2019 (1 training year) -> full-feature fallback
    2020 (2 training years) -> full-feature fallback
    2021-2023 -> stable subset selection is applied

The primary 19B2 comparison is therefore reported for 2021-2023, while
all-five-fold metrics are also saved for completeness.

NO RETUNING
-----------
All models use the frozen Step 17 tuned hyperparameters.
This isolates the effect of feature stability and avoids another tuning round.

THIS FILE DOES NOT:
- use 2024
- use 2025
- retune hyperparameters
- perform bias correction
- perform recent-year weighting
- build ensembles
- select the final research model
"""

from pathlib import Path
import gc
import hashlib
import json
import math
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

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

SELECTION_ELIGIBLE_YEARS = [
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

STRATEGIES = [
    "full_93_reference",
    "train_only_stable",
    "train_only_stable_plus_core",
]

MIN_STABILITY_TRAIN_YEARS = 3
RHO_RANGE_MAX = 0.40
MEANINGFUL_RHO_SIGN_THRESHOLD = 0.05

METRIC_TOLERANCE = 1e-9

DOMAIN_CORE_FEATURES = [
    "year",
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
    "dist_road",
    "dist_main_road",
    "dist_water",
    "LCZ",
]


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

STABILITY_STATS_PATH = (
    METRICS_DIR
    / "19b2_train_only_feature_stability_by_fold.csv"
)

SELECTION_DECISIONS_PATH = (
    METRICS_DIR
    / "19b2_feature_selection_decisions.csv"
)

FOLD_METRICS_PATH = (
    METRICS_DIR
    / "19b2_stable_feature_fold_metrics.csv"
)

SUMMARY_PATH = (
    METRICS_DIR
    / "19b2_stable_feature_summary.csv"
)

DESCRIPTIVE_BEST_PATH = (
    METRICS_DIR
    / "19b2_descriptive_best_strategy_by_model.csv"
)

PREDICTIONS_PATH = (
    PREDICTIONS_DIR
    / "19b2_stable_feature_predictions.csv"
)

FEATURE_LISTS_PATH = (
    REPORTS_DIR
    / "19b2_selected_feature_lists.json"
)

BASELINE_REPRO_PATH = (
    REPORTS_DIR
    / "19b2_step17_reference_reproduction_check.csv"
)

AUDIT_PATH = (
    REPORTS_DIR
    / "19b2_stable_feature_subset_audit.csv"
)

REPORT_PATH = (
    REPORTS_DIR
    / "19b2_stable_feature_subset_report.txt"
)

HASH_PATH = (
    REPORTS_DIR
    / "19b2_stable_feature_subset_hashes.csv"
)

FIG_MAE_PATH = (
    FIGURES_DIR
    / "19b2_stable_feature_mean_mae.png"
)

FIG_R2_PATH = (
    FIGURES_DIR
    / "19b2_stable_feature_r2_by_year.png"
)

FIG_COUNT_PATH = (
    FIGURES_DIR
    / "19b2_selected_feature_count_by_year.png"
)

FIG_IMPROVEMENT_PATH = (
    FIGURES_DIR
    / "19b2_mae_improvement_vs_full.png"
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
    }


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
        len(
            x
        )
        < 3
        or np.std(
            x,
            ddof=0,
        )
        < 1e-12
        or np.std(
            y,
            ddof=0,
        )
        < 1e-12
    ):
        return np.nan

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        result = spearmanr(
            x,
            y,
            nan_policy="omit",
        )

    return float(
        result.statistic
    )


def meaningful_sign(
    rho,
) -> int:
    if pd.isna(
        rho
    ):
        return 0

    if (
        rho
        >= MEANINGFUL_RHO_SIGN_THRESHOLD
    ):
        return 1

    if (
        rho
        <= -MEANINGFUL_RHO_SIGN_THRESHOLD
    ):
        return -1

    return 0


def chronological_sign_flips(
    rhos,
) -> int:
    signs = [
        meaningful_sign(
            rho
        )
        for rho in rhos
    ]

    signs = [
        sign
        for sign in signs
        if sign != 0
    ]

    if len(
        signs
    ) < 2:
        return 0

    return int(
        sum(
            signs[
                index
            ]
            != signs[
                index
                - 1
            ]
            for index in range(
                1,
                len(
                    signs
                ),
            )
        )
    )


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


# =============================================================================
# 4. PREREQUISITE AUDITS
# =============================================================================

section(
    "STEP 19B2 - TRAIN-ONLY STABLE FEATURE SUBSET"
)

required_files = [
    INPUT_PATH,
    STEP17_PREDICTIONS_PATH,
    STEP17_FOLD_METRICS_PATH,
    STEP17_PARAMS_PATH,
    STEP17_AUDIT_PATH,
    STEP19A_AUDIT_PATH,
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

print(
    "PASS: Step 17 audit fully PASS."
)
print(
    "PASS: Step 19A audit fully PASS."
)
print(
    "Feature stability will be estimated inside each rolling fold "
    "using training years only."
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

missing_core = sorted(
    set(
        DOMAIN_CORE_FEATURES
    )
    - set(
        predictors
    )
)

if missing_core:
    print(
        "\nERROR: expected Step 08 domain-core features missing: "
        f"{missing_core}"
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
    f"Domain-core count  : {len(DOMAIN_CORE_FEATURES)}"
)
print(
    "2024 data          : NOT LOADED"
)
print(
    "2025 data          : NOT LOADED"
)


# =============================================================================
# 6. LOAD STEP 17 PARAMS / PREDICTIONS / FOLD METRICS
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
        "\nERROR: Step 17 best-parameter model names do not match "
        "the expected four models."
    )
    print(
        best_params.keys()
    )
    sys.exit(1)

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
    "residual_actual_minus_predicted",
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

expected_step17_rows = (
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
) != expected_step17_rows:
    print(
        f"\nERROR: Step 17 prediction rows={len(step17_pred)}, "
        f"expected={expected_step17_rows}."
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
    "No retuning will be performed in Step 19B2."
)


# =============================================================================
# 7. VERIFY STEP 17 REFERENCE PREDICTIONS
# =============================================================================

section(
    "3. STEP 17 REFERENCE REPRODUCTION CHECK"
)

required_fold_columns = {
    "model",
    "validation_year",
    "train_mae",
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

reference_rows = []

for model_name in MODEL_ORDER:
    for validation_year in VALIDATION_YEARS:
        pred_subset = step17_pred[
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

        if len(
            pred_subset
        ) != EXPECTED_UNIQUE_GRIDS:
            print(
                "\nERROR: Step 17 prediction fold row count mismatch "
                f"for {model_name}, {validation_year}."
            )
            sys.exit(1)

        current = regression_metrics(
            pred_subset[
                "actual_lst_c"
            ],
            pred_subset[
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

        diffs = {
            "mae_difference": (
                current[
                    "mae"
                ]
                - float(
                    ref[
                        "validation_mae"
                    ]
                )
            ),
            "rmse_difference": (
                current[
                    "rmse"
                ]
                - float(
                    ref[
                        "validation_rmse"
                    ]
                )
            ),
            "r2_difference": (
                current[
                    "r2"
                ]
                - float(
                    ref[
                        "validation_r2"
                    ]
                )
            ),
        }

        passed = bool(
            all(
                abs(
                    value
                )
                <= METRIC_TOLERANCE
                for value in diffs.values()
            )
        )

        reference_rows.append(
            {
                "model": model_name,
                "validation_year": validation_year,
                **diffs,
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
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
    "\nPASS: all Step 17 validation predictions reproduce "
    "their finalized fold metrics."
)


# =============================================================================
# 8. BUILD TRAIN-ONLY TEMPORAL STABILITY TABLES
# =============================================================================

section(
    "4. BUILD TRAIN-ONLY TEMPORAL STABILITY TABLES"
)

stability_rows = []
selection_rows = []
feature_lists_payload = {
    "step": "19B2",
    "minimum_training_years_for_selection": (
        MIN_STABILITY_TRAIN_YEARS
    ),
    "rho_range_max": RHO_RANGE_MAX,
    "meaningful_rho_sign_threshold": (
        MEANINGFUL_RHO_SIGN_THRESHOLD
    ),
    "domain_core_features": (
        DOMAIN_CORE_FEATURES
    ),
    "folds": {},
}

selection_by_year = {}

for validation_year in VALIDATION_YEARS:
    train_years = [
        year
        for year in DEVELOPMENT_YEARS
        if year < validation_year
    ]

    selection_applied = bool(
        len(
            train_years
        )
        >= MIN_STABILITY_TRAIN_YEARS
    )

    fold_key = str(
        validation_year
    )

    feature_lists_payload[
        "folds"
    ][
        fold_key
    ] = {
        "training_years": (
            train_years
        ),
        "selection_applied": (
            selection_applied
        ),
    }

    if not selection_applied:
        stable_features = list(
            predictors
        )

        stable_plus_core_features = list(
            predictors
        )

        for feature in predictors:
            stability_rows.append(
                {
                    "validation_year": validation_year,
                    "train_start_year": min(
                        train_years
                    ),
                    "train_end_year": max(
                        train_years
                    ),
                    "training_year_count": len(
                        train_years
                    ),
                    "feature": feature,
                    "valid_rho_years": np.nan,
                    "rho_min": np.nan,
                    "rho_max": np.nan,
                    "rho_range": np.nan,
                    "meaningful_sign_flip_count": np.nan,
                    "stable_by_train_only_rule": True,
                    "selection_applied": False,
                    "selection_reason": (
                        "full_feature_fallback_insufficient_training_years"
                    ),
                }
            )

    else:
        stable_features = [
            YEAR_COL
        ]

        stable_plus_core_features = [
            YEAR_COL
        ]

        for feature in predictors:
            if feature == YEAR_COL:
                stability_rows.append(
                    {
                        "validation_year": validation_year,
                        "train_start_year": min(
                            train_years
                        ),
                        "train_end_year": max(
                            train_years
                        ),
                        "training_year_count": len(
                            train_years
                        ),
                        "feature": feature,
                        "valid_rho_years": 0,
                        "rho_min": np.nan,
                        "rho_max": np.nan,
                        "rho_range": np.nan,
                        "meaningful_sign_flip_count": 0,
                        "stable_by_train_only_rule": True,
                        "selection_applied": True,
                        "selection_reason": (
                            "year_retained_by_policy"
                        ),
                    }
                )
                continue

            year_rhos = []

            for train_year in train_years:
                year_df = df[
                    df[
                        YEAR_COL
                    ]
                    == train_year
                ]

                rho = safe_spearman(
                    year_df[
                        feature
                    ],
                    year_df[
                        TARGET
                    ],
                )

                year_rhos.append(
                    rho
                )

            finite_rhos = [
                float(
                    rho
                )
                for rho in year_rhos
                if pd.notna(
                    rho
                )
            ]

            if finite_rhos:
                rho_min = float(
                    np.min(
                        finite_rhos
                    )
                )

                rho_max = float(
                    np.max(
                        finite_rhos
                    )
                )

                rho_range = (
                    rho_max
                    - rho_min
                )
            else:
                rho_min = np.nan
                rho_max = np.nan
                rho_range = np.nan

            sign_flips = (
                chronological_sign_flips(
                    year_rhos
                )
            )

            stable = bool(
                len(
                    finite_rhos
                )
                >= MIN_STABILITY_TRAIN_YEARS
                and rho_range
                <= RHO_RANGE_MAX
                and sign_flips
                == 0
            )

            if stable:
                stable_features.append(
                    feature
                )

            if (
                stable
                or feature
                in DOMAIN_CORE_FEATURES
            ):
                stable_plus_core_features.append(
                    feature
                )

            if stable:
                reason = (
                    "stable_train_only_relationship"
                )
            elif (
                feature
                in DOMAIN_CORE_FEATURES
            ):
                reason = (
                    "unstable_but_domain_core_protected_in_plus_core_only"
                )
            else:
                reason = (
                    "excluded_due_to_train_only_relationship_instability"
                )

            stability_rows.append(
                {
                    "validation_year": validation_year,
                    "train_start_year": min(
                        train_years
                    ),
                    "train_end_year": max(
                        train_years
                    ),
                    "training_year_count": len(
                        train_years
                    ),
                    "feature": feature,
                    "valid_rho_years": len(
                        finite_rhos
                    ),
                    "rho_min": rho_min,
                    "rho_max": rho_max,
                    "rho_range": rho_range,
                    "meaningful_sign_flip_count": (
                        sign_flips
                    ),
                    "stable_by_train_only_rule": (
                        stable
                    ),
                    "selection_applied": True,
                    "selection_reason": reason,
                }
            )

    # Preserve original predictor order.
    stable_features = [
        feature
        for feature in predictors
        if feature
        in set(
            stable_features
        )
    ]

    stable_plus_core_features = [
        feature
        for feature in predictors
        if feature
        in set(
            stable_plus_core_features
        )
    ]

    if YEAR_COL not in stable_features:
        print(
            "\nERROR: year missing from train_only_stable."
        )
        sys.exit(1)

    if YEAR_COL not in stable_plus_core_features:
        print(
            "\nERROR: year missing from train_only_stable_plus_core."
        )
        sys.exit(1)

    if selection_applied:
        if len(
            stable_features
        ) >= len(
            predictors
        ):
            print(
                f"\nWARNING: train_only_stable retained all "
                f"{len(predictors)} features for {validation_year}."
            )

        if len(
            stable_features
        ) < 2:
            print(
                "\nERROR: stable subset unexpectedly collapsed "
                "to fewer than two predictors."
            )
            sys.exit(1)

    selection_by_year[
        validation_year
    ] = {
        "full_93_reference": list(
            predictors
        ),
        "train_only_stable": (
            stable_features
        ),
        "train_only_stable_plus_core": (
            stable_plus_core_features
        ),
        "selection_applied": (
            selection_applied
        ),
        "train_years": (
            train_years
        ),
    }

    feature_lists_payload[
        "folds"
    ][
        fold_key
    ][
        "full_93_reference"
    ] = list(
        predictors
    )

    feature_lists_payload[
        "folds"
    ][
        fold_key
    ][
        "train_only_stable"
    ] = (
        stable_features
    )

    feature_lists_payload[
        "folds"
    ][
        fold_key
    ][
        "train_only_stable_plus_core"
    ] = (
        stable_plus_core_features
    )

    print(
        f"{train_years[0]}-{train_years[-1]} -> "
        f"{validation_year} | "
        f"selection_applied={selection_applied} | "
        f"full={len(predictors)} | "
        f"stable={len(stable_features)} | "
        f"stable+core={len(stable_plus_core_features)}"
    )

    for strategy in STRATEGIES:
        selected = set(
            selection_by_year[
                validation_year
            ][
                strategy
            ]
        )

        for feature in predictors:
            selection_rows.append(
                {
                    "validation_year": (
                        validation_year
                    ),
                    "train_start_year": min(
                        train_years
                    ),
                    "train_end_year": max(
                        train_years
                    ),
                    "training_year_count": len(
                        train_years
                    ),
                    "strategy": strategy,
                    "selection_applied": (
                        selection_applied
                        and strategy
                        != "full_93_reference"
                    ),
                    "feature": feature,
                    "selected": bool(
                        feature
                        in selected
                    ),
                    "domain_core": bool(
                        feature
                        in DOMAIN_CORE_FEATURES
                    ),
                }
            )

stability_df = pd.DataFrame(
    stability_rows
)

selection_df = pd.DataFrame(
    selection_rows
)

stability_df.to_csv(
    STABILITY_STATS_PATH,
    index=False,
)

selection_df.to_csv(
    SELECTION_DECISIONS_PATH,
    index=False,
)

with FEATURE_LISTS_PATH.open(
    "w",
    encoding="utf-8",
) as file_obj:
    json.dump(
        feature_lists_payload,
        file_obj,
        indent=2,
        sort_keys=True,
    )

print(
    f"\nSaved: {STABILITY_STATS_PATH}"
)
print(
    f"Saved: {SELECTION_DECISIONS_PATH}"
)
print(
    f"Saved: {FEATURE_LISTS_PATH}"
)


# =============================================================================
# 9. STRICT ROLLING-ORIGIN EVALUATION
# =============================================================================

section(
    "5. STRICT ROLLING-ORIGIN STABLE-FEATURE EVALUATION"
)

fold_metric_rows = []
prediction_frames = []

for model_name in MODEL_ORDER:
    print(
        f"\n--- {model_name} ---"
    )

    for validation_year in VALIDATION_YEARS:
        train_years = (
            selection_by_year[
                validation_year
            ][
                "train_years"
            ]
        )

        train_mask = (
            df[
                YEAR_COL
            ]
            < validation_year
        )

        val_mask = (
            df[
                YEAR_COL
            ]
            == validation_year
        )

        train_df = df[
            train_mask
        ]

        val_df = df[
            val_mask
        ]

        y_train = (
            train_df[
                TARGET
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_val = (
            val_df[
                TARGET
            ]
            .to_numpy(
                dtype=float
            )
        )

        for strategy in STRATEGIES:
            selected_features = (
                selection_by_year[
                    validation_year
                ][
                    strategy
                ]
            )

            selection_applied = bool(
                selection_by_year[
                    validation_year
                ][
                    "selection_applied"
                ]
                and strategy
                != "full_93_reference"
            )

            # ---------------------------------------------------------
            # Reference strategy and early fallback years:
            # Reuse finalized Step 17 predictions/metrics.
            # This avoids unnecessary duplicate model fits and is exact.
            # ---------------------------------------------------------
            reuse_step17 = bool(
                strategy
                == "full_93_reference"
                or not selection_by_year[
                    validation_year
                ][
                    "selection_applied"
                ]
            )

            if reuse_step17:
                reference_prediction = step17_pred[
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
                ].copy()

                reference_metric = step17_fold[
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
                    reference_metric
                ) != 1:
                    print(
                        "\nERROR: Step 17 reference metric row mismatch."
                    )
                    sys.exit(1)

                reference_metric = (
                    reference_metric.iloc[
                        0
                    ]
                )

                pred_val = (
                    reference_prediction[
                        "predicted_lst_c"
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                val_metrics = regression_metrics(
                    y_val,
                    pred_val,
                )

                train_mae = float(
                    reference_metric[
                        "train_mae"
                    ]
                )

                train_rmse = float(
                    reference_metric[
                        "train_rmse"
                    ]
                )

                train_r2 = float(
                    reference_metric[
                        "train_r2"
                    ]
                )

                train_mape = float(
                    reference_metric[
                        "train_mape_percent"
                    ]
                )

                source = (
                    "reused_finalized_step17_reference"
                    if strategy
                    == "full_93_reference"
                    else (
                        "reused_step17_full_feature_fallback_"
                        "insufficient_training_years"
                    )
                )

                area_name = (
                    reference_prediction[
                        AREA_COL
                    ]
                    .astype(str)
                    .to_numpy()
                )

                grid_ids = (
                    reference_prediction[
                        GRID_COL
                    ]
                    .astype(str)
                    .to_numpy()
                )

                actual_values = (
                    reference_prediction[
                        "actual_lst_c"
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

            else:
                X_train = train_df[
                    selected_features
                ]

                X_val = val_df[
                    selected_features
                ]

                model = build_model(
                    model_name,
                    best_params[
                        model_name
                    ],
                )

                model.fit(
                    X_train,
                    y_train,
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

                train_mae = (
                    train_metrics[
                        "mae"
                    ]
                )

                train_rmse = (
                    train_metrics[
                        "rmse"
                    ]
                )

                train_r2 = (
                    train_metrics[
                        "r2"
                    ]
                )

                train_mape = (
                    train_metrics[
                        "mape_percent"
                    ]
                )

                source = (
                    "refit_frozen_step17_model_with_train_only_subset"
                )

                area_name = (
                    val_df[
                        AREA_COL
                    ]
                    .astype(str)
                    .to_numpy()
                )

                grid_ids = (
                    val_df[
                        GRID_COL
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
                    "\nERROR: non-finite predictions found."
                )
                sys.exit(1)

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
                    "training_year_count": len(
                        train_years
                    ),
                    "train_rows": len(
                        train_df
                    ),
                    "validation_rows": len(
                        val_df
                    ),
                    "selection_applied": (
                        selection_applied
                    ),
                    "selected_feature_count": len(
                        selected_features
                    ),
                    "evaluation_source": source,
                    "train_mae": train_mae,
                    "validation_mae": (
                        val_metrics[
                            "mae"
                        ]
                    ),
                    "mae_generalization_gap": (
                        val_metrics[
                            "mae"
                        ]
                        - train_mae
                    ),
                    "train_rmse": train_rmse,
                    "validation_rmse": (
                        val_metrics[
                            "rmse"
                        ]
                    ),
                    "rmse_generalization_gap": (
                        val_metrics[
                            "rmse"
                        ]
                        - train_rmse
                    ),
                    "train_r2": train_r2,
                    "validation_r2": (
                        val_metrics[
                            "r2"
                        ]
                    ),
                    "r2_generalization_gap": (
                        train_r2
                        - val_metrics[
                            "r2"
                        ]
                    ),
                    "train_mape_percent": (
                        train_mape
                    ),
                    "validation_mape_percent": (
                        val_metrics[
                            "mape_percent"
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
                        "selection_applied": (
                            selection_applied
                        ),
                        "selected_feature_count": len(
                            selected_features
                        ),
                        GRID_COL: grid_ids,
                        AREA_COL: area_name,
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
                f"{strategy:<29} | "
                f"features={len(selected_features):>2} | "
                f"MAE={val_metrics['mae']:.4f} | "
                f"RMSE={val_metrics['rmse']:.4f} | "
                f"R²={val_metrics['r2']:.4f}"
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


# =============================================================================
# 10. STRATEGY SUMMARY
# =============================================================================

section(
    "6. STABLE-FEATURE STRATEGY SUMMARY"
)

summary_rows = []

for model_name in MODEL_ORDER:
    full_eligible = fold_metrics[
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
            == "full_93_reference"
        )
        & (
            fold_metrics[
                "validation_year"
            ]
            .isin(
                SELECTION_ELIGIBLE_YEARS
            )
        )
    ]

    full_mean_mae = float(
        full_eligible[
            "validation_mae"
        ].mean()
    )

    full_mean_rmse = float(
        full_eligible[
            "validation_rmse"
        ].mean()
    )

    full_mean_r2 = float(
        full_eligible[
            "validation_r2"
        ].mean()
    )

    full_negative_count = int(
        full_eligible[
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

        subset_eligible = subset_all[
            subset_all[
                "validation_year"
            ]
            .isin(
                SELECTION_ELIGIBLE_YEARS
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
            subset_eligible
        ) != len(
            SELECTION_ELIGIBLE_YEARS
        ):
            print(
                "\nERROR: incomplete selection-eligible summary input."
            )
            sys.exit(1)

        strategy_mean_mae = float(
            subset_eligible[
                "validation_mae"
            ].mean()
        )

        strategy_mean_rmse = float(
            subset_eligible[
                "validation_rmse"
            ].mean()
        )

        strategy_mean_r2 = float(
            subset_eligible[
                "validation_r2"
            ].mean()
        )

        strategy_negative_count = int(
            subset_eligible[
                "negative_r2"
            ].sum()
        )

        summary_rows.append(
            {
                "model": model_name,
                "strategy": strategy,
                "all_folds_mean_validation_mae": float(
                    subset_all[
                        "validation_mae"
                    ].mean()
                ),
                "all_folds_std_validation_mae": float(
                    subset_all[
                        "validation_mae"
                    ].std(
                        ddof=0
                    )
                ),
                "all_folds_mean_validation_rmse": float(
                    subset_all[
                        "validation_rmse"
                    ].mean()
                ),
                "all_folds_mean_validation_r2": float(
                    subset_all[
                        "validation_r2"
                    ].mean()
                ),
                "all_folds_negative_r2_count": int(
                    subset_all[
                        "negative_r2"
                    ].sum()
                ),
                "all_folds_worst_year_mae": float(
                    subset_all[
                        "validation_mae"
                    ].max()
                ),
                "all_folds_mean_generalization_gap": float(
                    subset_all[
                        "mae_generalization_gap"
                    ].mean()
                ),
                "eligible_2021_2023_mean_mae": (
                    strategy_mean_mae
                ),
                "eligible_2021_2023_std_mae": float(
                    subset_eligible[
                        "validation_mae"
                    ].std(
                        ddof=0
                    )
                ),
                "eligible_2021_2023_mean_rmse": (
                    strategy_mean_rmse
                ),
                "eligible_2021_2023_mean_r2": (
                    strategy_mean_r2
                ),
                "eligible_2021_2023_negative_r2_count": (
                    strategy_negative_count
                ),
                "eligible_2021_2023_worst_year_mae": float(
                    subset_eligible[
                        "validation_mae"
                    ].max()
                ),
                "eligible_2021_2023_mean_generalization_gap": float(
                    subset_eligible[
                        "mae_generalization_gap"
                    ].mean()
                ),
                "eligible_mean_selected_feature_count": float(
                    subset_eligible[
                        "selected_feature_count"
                    ].mean()
                ),
                "eligible_min_selected_feature_count": int(
                    subset_eligible[
                        "selected_feature_count"
                    ].min()
                ),
                "eligible_max_selected_feature_count": int(
                    subset_eligible[
                        "selected_feature_count"
                    ].max()
                ),
                "mean_mae_improvement_vs_full_2021_2023": (
                    full_mean_mae
                    - strategy_mean_mae
                ),
                "mean_rmse_improvement_vs_full_2021_2023": (
                    full_mean_rmse
                    - strategy_mean_rmse
                ),
                "mean_r2_improvement_vs_full_2021_2023": (
                    strategy_mean_r2
                    - full_mean_r2
                ),
                "negative_r2_count_change_vs_full_2021_2023": (
                    strategy_negative_count
                    - full_negative_count
                ),
            }
        )

summary = pd.DataFrame(
    summary_rows
)

summary[
    "within_model_19b2_mae_rank"
] = (
    summary.groupby(
        "model"
    )[
        "eligible_2021_2023_mean_mae"
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
        "within_model_19b2_mae_rank",
        "eligible_2021_2023_worst_year_mae",
        "eligible_2021_2023_mean_rmse",
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
            "eligible_2021_2023_mean_mae",
            "eligible_2021_2023_worst_year_mae",
            "eligible_2021_2023_mean_rmse",
            "eligible_2021_2023_negative_r2_count",
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
    "\nDESCRIPTIVE BEST STRATEGY WITHIN 19B2 "
    "(NOT final model selection):"
)

print(
    descriptive_best[
        [
            "model",
            "strategy",
            "eligible_2021_2023_mean_mae",
            "eligible_2021_2023_mean_rmse",
            "eligible_2021_2023_mean_r2",
            "eligible_2021_2023_negative_r2_count",
            "eligible_2021_2023_worst_year_mae",
            "eligible_mean_selected_feature_count",
            "mean_mae_improvement_vs_full_2021_2023",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# 11. FIGURES
# =============================================================================

section(
    "7. FIGURES"
)

# ---- mean MAE, 2021-2023 ----
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
            "eligible_2021_2023_mean_mae"
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
    "Mean MAE for selection-eligible years 2021-2023 (°C)"
)

ax.set_title(
    "Step 19B2 - Stable-Feature Strategy Comparison"
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


# ---- R² by validation year ----
fig, ax = plt.subplots(
    figsize=(
        12,
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
    "Step 19B2 - Mean Temporal R² by Feature Strategy"
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


# ---- selected feature counts ----
fig, ax = plt.subplots(
    figsize=(
        10,
        6,
    )
)

for strategy in [
    "train_only_stable",
    "train_only_stable_plus_core",
]:
    count_rows = (
        fold_metrics[
            (
                fold_metrics[
                    "model"
                ]
                == MODEL_ORDER[
                    0
                ]
            )
            & (
                fold_metrics[
                    "strategy"
                ]
                == strategy
            )
        ]
        .sort_values(
            "validation_year"
        )
    )

    ax.plot(
        count_rows[
            "validation_year"
        ],
        count_rows[
            "selected_feature_count"
        ],
        marker="o",
        label=strategy,
    )

ax.axhline(
    EXPECTED_PREDICTOR_COUNT,
    linewidth=1,
    label="full predictor count",
)

ax.set_xticks(
    VALIDATION_YEARS
)

ax.set_xlabel(
    "Validation year"
)

ax.set_ylabel(
    "Selected feature count"
)

ax.set_title(
    "Step 19B2 - Train-Only Stable Feature Count"
)

ax.grid(
    alpha=0.2,
)

ax.legend()

plt.tight_layout()

plt.savefig(
    FIG_COUNT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ---- MAE improvement vs full reference ----
improvement_plot = summary[
    summary[
        "strategy"
    ]
    != "full_93_reference"
].copy()

labels = (
    improvement_plot[
        "model"
    ]
    + " | "
    + improvement_plot[
        "strategy"
    ]
)

fig, ax = plt.subplots(
    figsize=(
        12,
        7,
    )
)

ax.barh(
    labels,
    improvement_plot[
        "mean_mae_improvement_vs_full_2021_2023"
    ],
)

ax.axvline(
    0,
    linewidth=1,
)

ax.set_xlabel(
    "Mean MAE improvement vs full 93 features, 2021-2023 (°C)"
)

ax.set_title(
    "Step 19B2 - Stable-Feature MAE Change vs Full Reference"
)

ax.grid(
    axis="x",
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    FIG_IMPROVEMENT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

for figure_path in [
    FIG_MAE_PATH,
    FIG_R2_PATH,
    FIG_COUNT_PATH,
    FIG_IMPROVEMENT_PATH,
]:
    print(
        f"Saved: {figure_path}"
    )


# =============================================================================
# 12. STRICT AUDIT
# =============================================================================

section(
    "8. STRICT STEP 19B2 AUDIT"
)

expected_stability_rows = (
    len(
        VALIDATION_YEARS
    )
    * EXPECTED_PREDICTOR_COUNT
)

expected_selection_rows = (
    len(
        VALIDATION_YEARS
    )
    * len(
        STRATEGIES
    )
    * EXPECTED_PREDICTOR_COUNT
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

expected_ref_rows = (
    len(
        MODEL_ORDER
    )
    * len(
        VALIDATION_YEARS
    )
)

selection_future_leakage = bool(
    (
        selection_df[
            "train_end_year"
        ]
        >= selection_df[
            "validation_year"
        ]
    ).any()
)

selection_eligible_flags = (
    fold_metrics[
        fold_metrics[
            "strategy"
        ]
        != "full_93_reference"
    ][
        [
            "validation_year",
            "selection_applied",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "validation_year"
    )
)

selection_year_policy_ok = bool(
    all(
        bool(
            row[
                "selection_applied"
            ]
        )
        == (
            int(
                row[
                    "validation_year"
                ]
            )
            in SELECTION_ELIGIBLE_YEARS
        )
        for _, row in selection_eligible_flags.iterrows()
    )
)

core_protection_ok = bool(
    selection_df[
        (
            selection_df[
                "strategy"
            ]
            == "train_only_stable_plus_core"
        )
        & (
            selection_df[
                "validation_year"
            ]
            .isin(
                SELECTION_ELIGIBLE_YEARS
            )
        )
        & (
            selection_df[
                "domain_core"
            ]
        )
    ][
        "selected"
    ].all()
)

year_always_selected = bool(
    selection_df[
        selection_df[
            "feature"
        ]
        == YEAR_COL
    ][
        "selected"
    ].all()
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
            ]
        ]
        .to_numpy(
            dtype=float
        )
    ).all()
)

full_ref_metrics = fold_metrics[
    fold_metrics[
        "strategy"
    ]
    == "full_93_reference"
]

full_reference_repro_ok = True

for _, row in full_ref_metrics.iterrows():
    ref = step17_fold[
        (
            step17_fold[
                "model"
            ]
            == row[
                "model"
            ]
        )
        & (
            step17_fold[
                "validation_year"
            ]
            == row[
                "validation_year"
            ]
        )
    ]

    if len(
        ref
    ) != 1:
        full_reference_repro_ok = False
        break

    ref = ref.iloc[
        0
    ]

    if not (
        abs(
            float(
                row[
                    "validation_mae"
                ]
            )
            - float(
                ref[
                    "validation_mae"
                ]
            )
        )
        <= METRIC_TOLERANCE
        and abs(
            float(
                row[
                    "validation_rmse"
                ]
            )
            - float(
                ref[
                    "validation_rmse"
                ]
            )
        )
        <= METRIC_TOLERANCE
        and abs(
            float(
                row[
                    "validation_r2"
                ]
            )
            - float(
                ref[
                    "validation_r2"
                ]
            )
        )
        <= METRIC_TOLERANCE
    ):
        full_reference_repro_ok = False
        break

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
        "check": "domain_core_count",
        "status": (
            "PASS"
            if len(
                DOMAIN_CORE_FEATURES
            )
            == 16
            else "FAIL"
        ),
        "details": (
            f"actual={len(DOMAIN_CORE_FEATURES)}, expected=16"
        ),
    },
    {
        "check": "step17_reference_reproduction_rows",
        "status": (
            "PASS"
            if len(
                reference_repro
            )
            == expected_ref_rows
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
            f"expected={expected_ref_rows}"
        ),
    },
    {
        "check": "full_reference_exactly_matches_step17",
        "status": (
            "PASS"
            if full_reference_repro_ok
            else "FAIL"
        ),
        "details": (
            "full_93_reference reproduces finalized Step 17 fold metrics."
        ),
    },
    {
        "check": "stability_table_rows",
        "status": (
            "PASS"
            if len(
                stability_df
            )
            == expected_stability_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(stability_df)}, "
            f"expected={expected_stability_rows}"
        ),
    },
    {
        "check": "selection_decision_rows",
        "status": (
            "PASS"
            if len(
                selection_df
            )
            == expected_selection_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(selection_df)}, "
            f"expected={expected_selection_rows}"
        ),
    },
    {
        "check": "strict_train_years_only_for_feature_selection",
        "status": (
            "PASS"
            if not selection_future_leakage
            else "FAIL"
        ),
        "details": (
            "Every feature-selection training end year is earlier "
            "than its validation year."
        ),
    },
    {
        "check": "minimum_three_year_selection_policy",
        "status": (
            "PASS"
            if selection_year_policy_ok
            else "FAIL"
        ),
        "details": (
            "2019/2020 use full-feature fallback; "
            "2021-2023 apply train-only stability selection."
        ),
    },
    {
        "check": "year_predictor_always_retained",
        "status": (
            "PASS"
            if year_always_selected
            else "FAIL"
        ),
        "details": (
            "year is retained by policy in every strategy/fold."
        ),
    },
    {
        "check": "plus_core_domain_features_protected",
        "status": (
            "PASS"
            if core_protection_ok
            else "FAIL"
        ),
        "details": (
            "All 16 domain-core predictors remain in "
            "train_only_stable_plus_core for eligible folds."
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
            "No. 2024 was not loaded by Step 19B2."
        ),
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": (
            "No. 2025 was not loaded by Step 19B2."
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
        "check": "bias_correction_applied",
        "status": "PASS",
        "details": (
            "No bias correction occurs in Step 19B2."
        ),
    },
    {
        "check": "recent_year_weighting_applied",
        "status": "PASS",
        "details": (
            "No sample weighting occurs in Step 19B2."
        ),
    },
    {
        "check": "final_model_selected",
        "status": "PASS",
        "details": (
            "No. Rankings are descriptive within 19B2 only."
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
        "\nERROR: one or more Step 19B2 audit checks failed."
    )
    sys.exit(1)


# =============================================================================
# 13. REPORT
# =============================================================================

section(
    "9. SAVE STEP 19B2 REPORT"
)

report_lines = [
    "STEP 19B2 - TRAIN-ONLY STABLE FEATURE SUBSET",
    "=" * 112,
    "",
    "PURPOSE",
    "-" * 112,
    "Test whether removing temporally unstable predictors improves",
    "strict rolling-origin development generalization.",
    "",
    "LEAKAGE CONTROL",
    "-" * 112,
    "Feature stability for validation year t was estimated only from",
    "training years strictly earlier than t.",
    "2024 was not loaded.",
    "2025 was not loaded.",
    "",
    "STABILITY RULE",
    "-" * 112,
    f"Minimum training years: {MIN_STABILITY_TRAIN_YEARS}",
    f"Maximum allowed Spearman rho range: {RHO_RANGE_MAX}",
    (
        "Meaningful sign threshold: "
        f"|rho| >= {MEANINGFUL_RHO_SIGN_THRESHOLD}"
    ),
    "Meaningful sign flip count must equal 0.",
    "year is retained by policy.",
    "",
    "EARLY-FOLD POLICY",
    "-" * 112,
    "2019 and 2020 do not have at least 3 earlier training years.",
    "Those folds therefore use the full 93-feature reference as a",
    "predeclared fallback. Primary subset comparison uses 2021-2023.",
    "",
    "STRATEGIES",
    "-" * 112,
    "full_93_reference",
    "train_only_stable",
    "train_only_stable_plus_core",
    "",
    "HYPERPARAMETERS",
    "-" * 112,
    "Frozen Step 17 tuned hyperparameters were reused without retuning.",
    "",
    "DESCRIPTIVE 19B2 RESULTS BY MODEL",
    "-" * 112,
]

for _, row in descriptive_best.iterrows():
    report_lines.append(
        f"{row['model']}: {row['strategy']} | "
        f"2021-2023 mean MAE="
        f"{row['eligible_2021_2023_mean_mae']:.6f} °C | "
        f"mean RMSE="
        f"{row['eligible_2021_2023_mean_rmse']:.6f} °C | "
        f"mean R²="
        f"{row['eligible_2021_2023_mean_r2']:.6f} | "
        f"negative-R² folds="
        f"{int(row['eligible_2021_2023_negative_r2_count'])} | "
        f"worst MAE="
        f"{row['eligible_2021_2023_worst_year_mae']:.6f} °C | "
        f"mean selected features="
        f"{row['eligible_mean_selected_feature_count']:.2f} | "
        f"MAE improvement vs full="
        f"{row['mean_mae_improvement_vs_full_2021_2023']:.6f} °C."
    )

report_lines.extend(
    [
        "",
        "INTERPRETATION POLICY",
        "-" * 112,
        "A stable subset is not automatically accepted because it has fewer",
        "features. It must improve or robustly preserve future-year metrics.",
        "A negative result is valid evidence that feature instability alone",
        "does not explain the temporal failure.",
        "",
        "NO FINAL MODEL SELECTION",
        "-" * 112,
        "Step 19B2 remains a development experiment.",
        "Its results must later be compared with bias correction,",
        "recent-year weighting, ensemble, and optional level/anomaly",
        "experiments in Step 19C before reserved-2024 validation.",
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
# 14. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "10. REPRODUCIBILITY HASHES"
)

hash_files = [
    INPUT_PATH,
    STEP17_PREDICTIONS_PATH,
    STEP17_FOLD_METRICS_PATH,
    STEP17_PARAMS_PATH,
    STEP17_AUDIT_PATH,
    STEP19A_AUDIT_PATH,
    STABILITY_STATS_PATH,
    SELECTION_DECISIONS_PATH,
    FOLD_METRICS_PATH,
    SUMMARY_PATH,
    DESCRIPTIVE_BEST_PATH,
    PREDICTIONS_PATH,
    FEATURE_LISTS_PATH,
    BASELINE_REPRO_PATH,
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
# 15. FINAL STATUS
# =============================================================================

section(
    "STEP 19B2 COMPLETED SUCCESSFULLY"
)

print(
    "Train-only stable-feature experiment completed.\n"
    "Feature stability was estimated independently inside each rolling fold.\n"
    "2019/2020 used the predeclared full-feature fallback because fewer than\n"
    "three earlier training years were available.\n"
    "2021-2023 applied train-only temporal stability selection.\n"
    "The full 93-feature reference reproduces finalized Step 17 results.\n"
    "Frozen Step 17 hyperparameters were reused without retuning.\n"
    "2024 reserved validation remained untouched.\n"
    "2025 final lockbox remained untouched.\n"
    "No bias correction, recent-year weighting, ensemble, or final model\n"
    "selection occurred.\n"
    "Review the complete output before moving to Step 19B3."
)
