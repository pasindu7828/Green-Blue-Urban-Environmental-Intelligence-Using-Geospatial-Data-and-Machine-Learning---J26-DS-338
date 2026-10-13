"""
STEP 20 - RESERVED 2024 TEMPORAL VALIDATION
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Run the FIRST untouched 2024 temporal validation using ONLY the candidate
strategies frozen by Step 19C.

Frozen Step 20 candidates:
1. Random Forest raw tuned
2. LightGBM spatial anomaly + prior-mean annual level
3. Random Forest spatial anomaly + prior-mean annual level

STRICT POLICY
-------------
- Train/final-fit candidates using 2018-2023 only.
- Predict 2024 once using the frozen Step 19C rules.
- Do NOT tune hyperparameters on 2024.
- Do NOT invent new strategies after seeing 2024.
- Do NOT load 2025.
- Do NOT retrain on 2018-2024 in this step.
- Do NOT select the final production model in this step.
  Step 21 performs final strategy selection/freeze using development + 2024 evidence.

IMPORTANT FOR THE B5 DECOMPOSITION CANDIDATES
---------------------------------------------
Training target:
    spatial_anomaly = LST_C - mean_LST_C(training year)

2024 annual level:
    mean of annual LST means from 2018-2023 only

2024 predicted anomaly:
    model prediction on 2024 predictors
    then subtract mean of the PREDICTIONS themselves

Eligible 2024 LST prediction:
    prior_mean_annual_level_2018_2023
    + centered_predicted_spatial_anomaly_2024

The actual 2024 target is accessed only AFTER all candidate predictions are
constructed. It is then used only for evaluation metrics.

2025 remains the final unseen lockbox.
"""

from pathlib import Path
import gc
import hashlib
import json
import math
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =============================================================================
# SETTINGS
# =============================================================================

TARGET = "LST_C"
GRID_COL = "grid_id"
AREA_COL = "area_name"
YEAR_COL = "year"

TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
VALIDATION_YEAR = 2024

EXPECTED_TRAIN_ROWS = 13164
EXPECTED_VALIDATION_ROWS = 2194
EXPECTED_UNIQUE_GRIDS = 2194
EXPECTED_PREDICTOR_COUNT = 93

RANDOM_STATE = 42
ZERO_MEAN_TOLERANCE = 1e-10
METRIC_TOLERANCE = 1e-9

EXPECTED_FROZEN_CANDIDATES = [
    "raw_step17",
    "lgbm_anomaly_plus_prior_mean_level",
    "rf_anomaly_plus_prior_mean_level",
]


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "tree"
    / "tree_train_2018_2023_ready.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "tree"
    / "tree_validation_2024_ready.csv"
)

STEP19C_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "19c_temporal_strategy_comparison_audit.csv"
)

STEP19C_SHORTLIST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "19c_step20_shortlist.csv"
)

STEP19C_MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "19c_step20_candidate_manifest.json"
)

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "models" / "validation_2024"

for folder in [
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

METRICS_PATH = (
    METRICS_DIR
    / "20_reserved_2024_validation_metrics.csv"
)

COMPARISON_PATH = (
    METRICS_DIR
    / "20_reserved_2024_candidate_comparison.csv"
)

ANNUAL_LEVEL_PATH = (
    METRICS_DIR
    / "20_reserved_2024_annual_level_audit.csv"
)

PREDICTIONS_PATH = (
    PREDICTIONS_DIR
    / "20_reserved_2024_predictions.csv"
)

AUDIT_PATH = (
    REPORTS_DIR
    / "20_reserved_2024_validation_audit.csv"
)

REPORT_PATH = (
    REPORTS_DIR
    / "20_reserved_2024_validation_report.txt"
)

HASH_PATH = (
    REPORTS_DIR
    / "20_reserved_2024_validation_hashes.csv"
)

FIG_MAE_PATH = (
    FIGURES_DIR
    / "20_reserved_2024_candidate_mae.png"
)

FIG_R2_PATH = (
    FIGURES_DIR
    / "20_reserved_2024_candidate_r2.png"
)

FIG_BIAS_PATH = (
    FIGURES_DIR
    / "20_reserved_2024_candidate_bias.png"
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


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file missing: {path}"
        )


def require_audit_all_pass(path, label):
    require_file(path)

    audit = pd.read_csv(path)

    if "status" not in audit.columns:
        raise RuntimeError(
            f"{label} audit has no status column"
        )

    bad = audit[
        audit["status"]
        .astype(str)
        .str.upper()
        != "PASS"
    ]

    if not bad.empty:
        print(bad.to_string(index=False))
        raise RuntimeError(
            f"{label} audit is not fully PASS"
        )


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


def build_model(model_name, hyperparameters):
    params = clean_params(
        hyperparameters
    )

    if model_name == "Random Forest":
        return RandomForestRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "LightGBM":
        return LGBMRegressor(
            **params,
            objective="regression",
            subsample_freq=1,
            importance_type="gain",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

    raise ValueError(
        f"Unsupported frozen model: {model_name}"
    )


def mape_percent(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    valid = (
        np.abs(y_true)
        > 1e-12
    )

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


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    bias = float(
        np.mean(
            y_pred
            - y_true
        )
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
        "mean_prediction_bias_pred_minus_actual_c": bias,
        "abs_mean_prediction_bias_c": abs(
            bias
        ),
    }


def centered_spatial_r2(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    y_true_centered = (
        y_true
        - np.mean(y_true)
    )

    y_pred_centered = (
        y_pred
        - np.mean(y_pred)
    )

    return float(
        r2_score(
            y_true_centered,
            y_pred_centered,
        )
    )


# =============================================================================
# 0. PREREQUISITE / FREEZE AUDIT
# =============================================================================

section(
    "STEP 20 - RESERVED 2024 TEMPORAL VALIDATION"
)

for path in [
    TRAIN_PATH,
    VALIDATION_PATH,
    STEP19C_AUDIT_PATH,
    STEP19C_SHORTLIST_PATH,
    STEP19C_MANIFEST_PATH,
]:
    require_file(path)

require_audit_all_pass(
    STEP19C_AUDIT_PATH,
    "Step 19C",
)

with STEP19C_MANIFEST_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    manifest = json.load(f)

shortlist = pd.read_csv(
    STEP19C_SHORTLIST_PATH
)

if manifest.get(
    "reserved_2024_loaded"
) is not False:
    raise RuntimeError(
        "Step 19C manifest does not confirm untouched 2024"
    )

if manifest.get(
    "final_2025_lockbox_loaded"
) is not False:
    raise RuntimeError(
        "Step 19C manifest does not confirm untouched 2025"
    )

manifest_candidates = manifest.get(
    "candidates",
    []
)

if len(manifest_candidates) != 3:
    raise RuntimeError(
        f"Frozen candidate count={len(manifest_candidates)}, expected=3"
    )

manifest_strategies = [
    str(candidate["strategy"])
    for candidate in manifest_candidates
]

if manifest_strategies != EXPECTED_FROZEN_CANDIDATES:
    raise RuntimeError(
        "Frozen Step 20 candidate order/set does not match the "
        "final Step 19C manifest.\n"
        f"Actual: {manifest_strategies}\n"
        f"Expected: {EXPECTED_FROZEN_CANDIDATES}"
    )

shortlist_strategies = (
    shortlist
    .sort_values("shortlist_order")["strategy"]
    .astype(str)
    .tolist()
)

if shortlist_strategies != manifest_strategies:
    raise RuntimeError(
        "Step 19C shortlist CSV and manifest disagree"
    )

print("PASS: Step 19C audit is fully PASS.")
print("PASS: frozen shortlist CSV matches the Step 19C manifest.")
print(f"Frozen candidates: {manifest_strategies}")
print("2024 is now opened ONLY for reserved validation.")
print("2025 remains completely untouched / NOT LOADED.")
print("No tuning or new strategy search is allowed in Step 20.")


# =============================================================================
# 1. LOAD TRAIN + RESERVED 2024 VALIDATION
# =============================================================================

section(
    "1. LOAD 2018-2023 TRAINING + RESERVED 2024 VALIDATION"
)

train_df = pd.read_csv(
    TRAIN_PATH,
    dtype={
        GRID_COL: str,
    },
)

val_df = pd.read_csv(
    VALIDATION_PATH,
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

for frame, label in [
    (train_df, "train"),
    (val_df, "validation"),
]:
    missing = sorted(
        required_columns
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            f"{label} missing required columns: {missing}"
        )

train_years = sorted(
    pd.to_numeric(
        train_df[YEAR_COL],
        errors="coerce",
    )
    .astype(int)
    .unique()
    .tolist()
)

validation_years = sorted(
    pd.to_numeric(
        val_df[YEAR_COL],
        errors="coerce",
    )
    .astype(int)
    .unique()
    .tolist()
)

if train_years != TRAIN_YEARS:
    raise RuntimeError(
        f"Training years={train_years}; expected={TRAIN_YEARS}"
    )

if validation_years != [VALIDATION_YEAR]:
    raise RuntimeError(
        f"Validation years={validation_years}; expected=[{VALIDATION_YEAR}]"
    )

if len(train_df) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError(
        f"Training rows={len(train_df)}; expected={EXPECTED_TRAIN_ROWS}"
    )

if len(val_df) != EXPECTED_VALIDATION_ROWS:
    raise RuntimeError(
        f"Validation rows={len(val_df)}; expected={EXPECTED_VALIDATION_ROWS}"
    )

if train_df[GRID_COL].nunique() != EXPECTED_UNIQUE_GRIDS:
    raise RuntimeError(
        "Unexpected training unique-grid count"
    )

if val_df[GRID_COL].nunique() != EXPECTED_UNIQUE_GRIDS:
    raise RuntimeError(
        "Unexpected validation unique-grid count"
    )

train_predictors = [
    column
    for column in train_df.columns
    if column not in {
        TARGET,
        GRID_COL,
        AREA_COL,
    }
]

val_predictors = [
    column
    for column in val_df.columns
    if column not in {
        TARGET,
        GRID_COL,
        AREA_COL,
    }
]

if train_predictors != val_predictors:
    raise RuntimeError(
        "Training and 2024 predictor schemas/order differ"
    )

predictors = train_predictors

if len(predictors) != EXPECTED_PREDICTOR_COUNT:
    raise RuntimeError(
        f"Predictor count={len(predictors)}; "
        f"expected={EXPECTED_PREDICTOR_COUNT}"
    )

for frame in [
    train_df,
    val_df,
]:
    for column in predictors + [TARGET]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

if train_df[predictors + [TARGET]].isna().any().any():
    raise RuntimeError(
        "Missing training numeric values found"
    )

if val_df[predictors + [TARGET]].isna().any().any():
    raise RuntimeError(
        "Missing 2024 numeric values found"
    )

if not np.isfinite(
    train_df[predictors + [TARGET]]
    .to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Non-finite training numeric values found"
    )

if not np.isfinite(
    val_df[predictors + [TARGET]]
    .to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Non-finite 2024 numeric values found"
    )

# Prediction matrices prepared now.
# 2024 TARGET is intentionally NOT assigned to y_val until all candidate
# predictions have been constructed.
X_train = train_df[
    predictors
].copy()

X_val_2024 = val_df[
    predictors
].copy()

print(f"Training rows       : {len(train_df):,}")
print(f"Training years      : {train_years}")
print(f"2024 validation rows: {len(val_df):,}")
print(f"Validation year     : {validation_years}")
print(f"Unique grids        : {val_df[GRID_COL].nunique():,}")
print(f"Predictors          : {len(predictors)}")
print("2024 target         : withheld from prediction construction")
print("2025 data           : NOT LOADED")


# =============================================================================
# 2. TRAINING-ONLY ANNUAL LEVEL
# =============================================================================

section(
    "2. CALCULATE FROZEN PRIOR-MEAN ANNUAL LEVEL FROM 2018-2023 ONLY"
)

annual_training_levels = (
    train_df.groupby(
        YEAR_COL,
        as_index=False,
    )
    .agg(
        annual_mean_lst_c=(
            TARGET,
            "mean",
        ),
        rows=(
            TARGET,
            "size",
        ),
    )
    .rename(
        columns={
            YEAR_COL: "year",
        }
    )
)

annual_training_levels[
    "year"
] = annual_training_levels[
    "year"
].astype(int)

if sorted(
    annual_training_levels["year"]
    .tolist()
) != TRAIN_YEARS:
    raise RuntimeError(
        "Annual training-level years are incorrect"
    )

prior_mean_level_2024 = float(
    annual_training_levels[
        "annual_mean_lst_c"
    ].mean()
)

annual_level_audit = annual_training_levels.copy()

annual_level_audit[
    "used_for_2024_prior_mean_level"
] = True

annual_level_audit[
    "forecast_2024_prior_mean_level_c"
] = prior_mean_level_2024

annual_level_audit.to_csv(
    ANNUAL_LEVEL_PATH,
    index=False,
)

print(
    annual_training_levels.to_string(
        index=False
    )
)

print(
    f"\nFrozen 2024 prior-mean annual level "
    f"= {prior_mean_level_2024:.6f} °C"
)

print(
    "IMPORTANT: this value uses 2018-2023 targets only."
)


# =============================================================================
# 3. CONSTRUCT ALL THREE 2024 PREDICTIONS BEFORE USING 2024 TARGET
# =============================================================================

section(
    "3. CONSTRUCT FROZEN 2024 PREDICTIONS"
)

prediction_store = {}
diagnostic_store = {}
saved_model_paths = {}

candidate_lookup = {
    candidate["strategy"]: candidate
    for candidate in manifest_candidates
}


# -------------------------------------------------------------------------
# Candidate 1: raw tuned Random Forest
# -------------------------------------------------------------------------

raw_candidate = candidate_lookup[
    "raw_step17"
]

raw_rf = build_model(
    raw_candidate[
        "base_model"
    ],
    raw_candidate[
        "implementation"
    ][
        "hyperparameters"
    ],
)

raw_rf.fit(
    X_train,
    train_df[TARGET].to_numpy(dtype=float),
)

raw_rf_prediction = np.asarray(
    raw_rf.predict(
        X_val_2024
    ),
    dtype=float,
)

if not np.isfinite(
    raw_rf_prediction
).all():
    raise RuntimeError(
        "Non-finite raw RF 2024 predictions"
    )

prediction_store[
    "raw_step17"
] = raw_rf_prediction

diagnostic_store[
    "raw_step17"
] = {
        "base_model": "Random Forest",
        "prediction_method": "direct_raw_lst",
        "forecast_level_c": float(
            np.mean(
                raw_rf_prediction
            )
        ),
        "raw_predicted_anomaly_mean_c": np.nan,
        "centered_predicted_anomaly_mean_c": np.nan,
    }

raw_model_path = (
    MODELS_DIR
    / "20_random_forest_raw_trained_2018_2023.joblib"
)

joblib.dump(
    raw_rf,
    raw_model_path,
)

saved_model_paths[
    "raw_step17"
] = raw_model_path

print(
    "Candidate 1 complete: Random Forest raw tuned"
)


# -------------------------------------------------------------------------
# Shared anomaly training target from 2018-2023 only
# -------------------------------------------------------------------------

training_year_mean_map = (
    train_df.groupby(
        YEAR_COL
    )[
        TARGET
    ]
    .mean()
    .to_dict()
)

train_year_level = (
    train_df[
        YEAR_COL
    ]
    .map(
        training_year_mean_map
    )
    .to_numpy(dtype=float)
)

y_train_anomaly = (
    train_df[
        TARGET
    ]
    .to_numpy(dtype=float)
    - train_year_level
)

max_abs_train_anomaly_mean = 0.0

for train_year in TRAIN_YEARS:
    mask = (
        train_df[
            YEAR_COL
        ]
        .to_numpy(dtype=int)
        == train_year
    )

    current_mean = float(
        np.mean(
            y_train_anomaly[
                mask
            ]
        )
    )

    max_abs_train_anomaly_mean = max(
        max_abs_train_anomaly_mean,
        abs(current_mean),
    )

if (
    max_abs_train_anomaly_mean
    > ZERO_MEAN_TOLERANCE
):
    raise RuntimeError(
        "Training anomaly target is not zero-mean inside year"
    )


# -------------------------------------------------------------------------
# Candidates 2 and 3: B5 decomposition
# -------------------------------------------------------------------------

for strategy in [
    "lgbm_anomaly_plus_prior_mean_level",
    "rf_anomaly_plus_prior_mean_level",
]:
    candidate = candidate_lookup[
        strategy
    ]

    implementation = candidate[
        "implementation"
    ]

    if (
        implementation.get(
            "level_rule"
        )
        != "prior_mean_level"
    ):
        raise RuntimeError(
            f"{strategy} frozen level rule changed"
        )

    model_name = candidate[
        "base_model"
    ]

    anomaly_model = build_model(
        model_name,
        implementation[
            "hyperparameters"
        ],
    )

    anomaly_model.fit(
        X_train,
        y_train_anomaly,
    )

    raw_anomaly_prediction = np.asarray(
        anomaly_model.predict(
            X_val_2024
        ),
        dtype=float,
    )

    if not np.isfinite(
        raw_anomaly_prediction
    ).all():
        raise RuntimeError(
            f"Non-finite anomaly predictions for {strategy}"
        )

    # Leakage-safe re-centering:
    # uses only 2024 model predictions, NOT actual 2024 LST.
    centered_anomaly_prediction = (
        raw_anomaly_prediction
        - float(
            np.mean(
                raw_anomaly_prediction
            )
        )
    )

    centered_mean = float(
        np.mean(
            centered_anomaly_prediction
        )
    )

    if abs(
        centered_mean
    ) > ZERO_MEAN_TOLERANCE:
        raise RuntimeError(
            f"Centered anomaly mean failed for {strategy}"
        )

    final_prediction = (
        prior_mean_level_2024
        + centered_anomaly_prediction
    )

    if not np.isfinite(
        final_prediction
    ).all():
        raise RuntimeError(
            f"Non-finite final prediction for {strategy}"
        )

    prediction_store[
        strategy
    ] = final_prediction

    diagnostic_store[
        strategy
    ] = {
        "base_model": model_name,
        "prediction_method": (
            "spatial_anomaly_plus_prior_mean_level"
        ),
        "forecast_level_c": prior_mean_level_2024,
        "raw_predicted_anomaly_mean_c": float(
            np.mean(
                raw_anomaly_prediction
            )
        ),
        "centered_predicted_anomaly_mean_c": (
            centered_mean
        ),
    }

    safe_name = (
        "lightgbm"
        if model_name == "LightGBM"
        else "random_forest"
    )

    model_path = (
        MODELS_DIR
        / f"20_{safe_name}_anomaly_trained_2018_2023.joblib"
    )

    joblib.dump(
        anomaly_model,
        model_path,
    )

    saved_model_paths[
        strategy
    ] = model_path

    print(
        f"Candidate complete: {strategy} | "
        f"raw anomaly mean="
        f"{np.mean(raw_anomaly_prediction):+.6f} °C | "
        f"centered mean={centered_mean:+.12f}"
    )

    del anomaly_model
    gc.collect()


# Freeze prediction arrays before target evaluation.
for strategy in EXPECTED_FROZEN_CANDIDATES:
    if strategy not in prediction_store:
        raise RuntimeError(
            f"Missing frozen prediction for {strategy}"
        )

    prediction_store[
        strategy
    ] = np.asarray(
        prediction_store[
            strategy
        ],
        dtype=float,
    ).copy()

print(
    "\nPASS: all three 2024 predictions constructed "
    "before 2024 target evaluation."
)


# =============================================================================
# 4. NOW OPEN 2024 TARGET FOR EVALUATION ONLY
# =============================================================================

section(
    "4. EVALUATE RESERVED 2024 TARGET"
)

y_val_2024 = val_df[
    TARGET
].to_numpy(dtype=float)

actual_2024_level = float(
    np.mean(
        y_val_2024
    )
)

print(
    f"Actual 2024 annual mean LST "
    f"(evaluation only) = {actual_2024_level:.6f} °C"
)

metric_rows = []
prediction_frames = []

display_lookup = {
    candidate["strategy"]: candidate[
        "display_name"
    ]
    for candidate in manifest_candidates
}

for strategy in EXPECTED_FROZEN_CANDIDATES:
    y_pred = prediction_store[
        strategy
    ]

    current = regression_metrics(
        y_val_2024,
        y_pred,
    )

    centered_r2 = centered_spatial_r2(
        y_val_2024,
        y_pred,
    )

    diagnostics = diagnostic_store[
        strategy
    ]

    forecast_level = float(
        diagnostics[
            "forecast_level_c"
        ]
    )

    level_error = (
        forecast_level
        - actual_2024_level
    )

    metric_rows.append(
        {
            "strategy": strategy,
            "display_name": (
                display_lookup[
                    strategy
                ]
            ),
            "base_model": (
                diagnostics[
                    "base_model"
                ]
            ),
            "prediction_method": (
                diagnostics[
                    "prediction_method"
                ]
            ),
            "train_years": "2018-2023",
            "validation_year": 2024,
            "train_rows": len(
                train_df
            ),
            "validation_rows": len(
                val_df
            ),
            "predictor_count": len(
                predictors
            ),
            "forecast_level_c": (
                forecast_level
            ),
            "actual_2024_level_c_evaluation_only": (
                actual_2024_level
            ),
            "level_error_forecast_minus_actual_c": (
                level_error
            ),
            "abs_level_error_c": abs(
                level_error
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
            "mean_prediction_bias_pred_minus_actual_c": (
                current[
                    "mean_prediction_bias_pred_minus_actual_c"
                ]
            ),
            "abs_mean_prediction_bias_c": (
                current[
                    "abs_mean_prediction_bias_c"
                ]
            ),
            "centered_spatial_r2_evaluation_only": (
                centered_r2
            ),
            "negative_r2": bool(
                current[
                    "r2"
                ]
                < 0
            ),
            "hyperparameters_retuned_on_2024": False,
            "new_strategy_created_after_2024": False,
            "eligible_for_step21_review": True,
        }
    )

    prediction_frames.append(
        pd.DataFrame(
            {
                "strategy": strategy,
                "display_name": (
                    display_lookup[
                        strategy
                    ]
                ),
                "base_model": (
                    diagnostics[
                        "base_model"
                    ]
                ),
                "validation_year": 2024,
                GRID_COL: (
                    val_df[
                        GRID_COL
                    ]
                    .astype(str)
                    .to_numpy()
                ),
                AREA_COL: (
                    val_df[
                        AREA_COL
                    ]
                    .astype(str)
                    .to_numpy()
                ),
                "actual_lst_c": (
                    y_val_2024
                ),
                "predicted_lst_c": (
                    y_pred
                ),
                "residual_actual_minus_predicted_c": (
                    y_val_2024
                    - y_pred
                ),
                "forecast_level_c": (
                    forecast_level
                ),
                "actual_2024_level_c_evaluation_only": (
                    actual_2024_level
                ),
            }
        )
    )

    print(
        f"{display_lookup[strategy]:<42} | "
        f"MAE={current['mae']:.6f} °C | "
        f"RMSE={current['rmse']:.6f} °C | "
        f"R²={current['r2']:.6f} | "
        f"MAPE={current['mape_percent']:.3f}% | "
        f"Bias={current['mean_prediction_bias_pred_minus_actual_c']:+.6f} °C | "
        f"Centered spatial R²={centered_r2:.6f}"
    )

metrics_df = pd.DataFrame(
    metric_rows
)

predictions_df = pd.concat(
    prediction_frames,
    ignore_index=True,
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

print(f"\nSaved: {METRICS_PATH}")
print(f"Saved: {PREDICTIONS_PATH}")


# =============================================================================
# 5. DESCRIPTIVE 2024 COMPARISON - NO FINAL SELECTION
# =============================================================================

section(
    "5. RESERVED 2024 DESCRIPTIVE COMPARISON"
)

comparison = (
    metrics_df.sort_values(
        [
            "mae",
            "rmse",
            "abs_mean_prediction_bias_c",
        ]
    )
    .reset_index(drop=True)
)

comparison[
    "descriptive_2024_mae_rank"
] = (
    np.arange(
        1,
        len(comparison) + 1,
    )
)

comparison[
    "final_model_selected_here"
] = False

comparison[
    "decision_note"
] = (
    "Descriptive reserved-2024 result only; "
    "final strategy decision belongs to Step 21."
)

comparison.to_csv(
    COMPARISON_PATH,
    index=False,
)

print(
    comparison[
        [
            "descriptive_2024_mae_rank",
            "display_name",
            "mae",
            "rmse",
            "r2",
            "mape_percent",
            "mean_prediction_bias_pred_minus_actual_c",
            "centered_spatial_r2_evaluation_only",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nIMPORTANT: no final model is selected in Step 20."
)
print(
    "Step 21 must review development + reserved-2024 evidence "
    "and freeze the final strategy without retuning."
)


# =============================================================================
# 6. FIGURES
# =============================================================================

section(
    "6. FIGURES"
)

plot_df = (
    comparison.sort_values(
        "descriptive_2024_mae_rank"
    )
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    plot_df[
        "display_name"
    ],
    plot_df[
        "mae"
    ],
)

ax.set_ylabel(
    "2024 MAE (°C)"
)

ax.set_title(
    "Step 20 - Reserved 2024 Validation MAE"
)

ax.tick_params(
    axis="x",
    rotation=20,
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


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    plot_df[
        "display_name"
    ],
    plot_df[
        "r2"
    ],
)

ax.axhline(
    0,
    linewidth=1,
)

ax.set_ylabel(
    "2024 R²"
)

ax.set_title(
    "Step 20 - Reserved 2024 Validation R²"
)

ax.tick_params(
    axis="x",
    rotation=20,
)

ax.grid(
    axis="y",
    alpha=0.2,
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


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    plot_df[
        "display_name"
    ],
    plot_df[
        "mean_prediction_bias_pred_minus_actual_c"
    ],
)

ax.axhline(
    0,
    linewidth=1,
)

ax.set_ylabel(
    "Mean prediction bias: predicted - actual (°C)"
)

ax.set_title(
    "Step 20 - Reserved 2024 Annual Prediction Bias"
)

ax.tick_params(
    axis="x",
    rotation=20,
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

for path in [
    FIG_MAE_PATH,
    FIG_R2_PATH,
    FIG_BIAS_PATH,
]:
    print(
        f"Saved: {path}"
    )


# =============================================================================
# 7. STRICT AUDIT
# =============================================================================

section(
    "7. STRICT STEP 20 AUDIT"
)

expected_prediction_rows = (
    len(
        EXPECTED_FROZEN_CANDIDATES
    )
    * EXPECTED_VALIDATION_ROWS
)

candidate_set_ok = bool(
    metrics_df[
        "strategy"
    ]
    .tolist()
    == EXPECTED_FROZEN_CANDIDATES
)

all_metrics_finite = bool(
    np.isfinite(
        metrics_df[
            [
                "forecast_level_c",
                "actual_2024_level_c_evaluation_only",
                "level_error_forecast_minus_actual_c",
                "abs_level_error_c",
                "mae",
                "rmse",
                "r2",
                "mape_percent",
                "mean_prediction_bias_pred_minus_actual_c",
                "abs_mean_prediction_bias_c",
                "centered_spatial_r2_evaluation_only",
            ]
        ]
        .to_numpy(dtype=float)
    ).all()
)

all_predictions_finite = bool(
    np.isfinite(
        predictions_df[
            [
                "actual_lst_c",
                "predicted_lst_c",
                "residual_actual_minus_predicted_c",
                "forecast_level_c",
                "actual_2024_level_c_evaluation_only",
            ]
        ]
        .to_numpy(dtype=float)
    ).all()
)

b5_rows = metrics_df[
    metrics_df[
        "prediction_method"
    ]
    .eq(
        "spatial_anomaly_plus_prior_mean_level"
    )
]

b5_level_exact = bool(
    np.allclose(
        b5_rows[
            "forecast_level_c"
        ]
        .to_numpy(dtype=float),
        prior_mean_level_2024,
        atol=METRIC_TOLERANCE,
    )
)

model_files_exist = bool(
    all(
        path.exists()
        for path in saved_model_paths.values()
    )
)

no_retuning_flags = bool(
    (
        metrics_df[
            "hyperparameters_retuned_on_2024"
        ]
        == False
    ).all()
)

no_new_strategy_flags = bool(
    (
        metrics_df[
            "new_strategy_created_after_2024"
        ]
        == False
    ).all()
)

no_final_selection = bool(
    (
        comparison[
            "final_model_selected_here"
        ]
        == False
    ).all()
)

audit_rows = [
    {
        "check": "step19c_audit",
        "status": "PASS",
        "details": (
            "Step 19C audit loaded and fully PASS."
        ),
    },
    {
        "check": "step19c_manifest_shortlist_match",
        "status": "PASS",
        "details": (
            "Frozen candidate manifest and shortlist CSV agree exactly."
        ),
    },
    {
        "check": "frozen_candidate_set",
        "status": (
            "PASS"
            if candidate_set_ok
            else "FAIL"
        ),
        "details": str(
            metrics_df[
                "strategy"
            ]
            .tolist()
        ),
    },
    {
        "check": "training_years",
        "status": (
            "PASS"
            if train_years
            == TRAIN_YEARS
            else "FAIL"
        ),
        "details": str(
            train_years
        ),
    },
    {
        "check": "validation_year",
        "status": (
            "PASS"
            if validation_years
            == [VALIDATION_YEAR]
            else "FAIL"
        ),
        "details": str(
            validation_years
        ),
    },
    {
        "check": "training_rows",
        "status": (
            "PASS"
            if len(train_df)
            == EXPECTED_TRAIN_ROWS
            else "FAIL"
        ),
        "details": (
            f"actual={len(train_df)}, "
            f"expected={EXPECTED_TRAIN_ROWS}"
        ),
    },
    {
        "check": "validation_rows",
        "status": (
            "PASS"
            if len(val_df)
            == EXPECTED_VALIDATION_ROWS
            else "FAIL"
        ),
        "details": (
            f"actual={len(val_df)}, "
            f"expected={EXPECTED_VALIDATION_ROWS}"
        ),
    },
    {
        "check": "predictor_count",
        "status": (
            "PASS"
            if len(predictors)
            == EXPECTED_PREDICTOR_COUNT
            else "FAIL"
        ),
        "details": (
            f"actual={len(predictors)}, "
            f"expected={EXPECTED_PREDICTOR_COUNT}"
        ),
    },
    {
        "check": "train_validation_predictor_schema",
        "status": (
            "PASS"
            if train_predictors
            == val_predictors
            else "FAIL"
        ),
        "details": (
            "Training and 2024 predictor names/order are identical."
        ),
    },
    {
        "check": "b5_2024_level_training_history_only",
        "status": (
            "PASS"
            if b5_level_exact
            else "FAIL"
        ),
        "details": (
            "Both B5 candidates use the mean of 2018-2023 "
            "annual training levels only."
        ),
    },
    {
        "check": "training_anomaly_zero_mean_by_year",
        "status": (
            "PASS"
            if max_abs_train_anomaly_mean
            <= ZERO_MEAN_TOLERANCE
            else "FAIL"
        ),
        "details": (
            f"max_abs_year_mean="
            f"{max_abs_train_anomaly_mean:.12f}"
        ),
    },
    {
        "check": "validation_anomaly_recentering",
        "status": "PASS",
        "details": (
            "B5 validation anomaly centering uses only the mean "
            "of 2024 model predictions; no 2024 target is used."
        ),
    },
    {
        "check": "predictions_constructed_before_target_evaluation",
        "status": "PASS",
        "details": (
            "All three prediction arrays were constructed and frozen "
            "before y_val_2024 was assigned for metric evaluation."
        ),
    },
    {
        "check": "metric_rows",
        "status": (
            "PASS"
            if len(metrics_df)
            == len(
                EXPECTED_FROZEN_CANDIDATES
            )
            else "FAIL"
        ),
        "details": (
            f"actual={len(metrics_df)}, "
            f"expected={len(EXPECTED_FROZEN_CANDIDATES)}"
        ),
    },
    {
        "check": "prediction_rows",
        "status": (
            "PASS"
            if len(predictions_df)
            == expected_prediction_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(predictions_df)}, "
            f"expected={expected_prediction_rows}"
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
        "check": "all_predictions_finite",
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
        "check": "saved_candidate_models",
        "status": (
            "PASS"
            if model_files_exist
            else "FAIL"
        ),
        "details": (
            "All 2018-2023 Step 20 validation-fit model artifacts saved."
        ),
    },
    {
        "check": "hyperparameter_retuning_on_2024",
        "status": (
            "PASS"
            if no_retuning_flags
            else "FAIL"
        ),
        "details": (
            "No 2024-driven hyperparameter tuning occurred."
        ),
    },
    {
        "check": "new_strategy_search_after_2024",
        "status": (
            "PASS"
            if no_new_strategy_flags
            else "FAIL"
        ),
        "details": (
            "No new strategy was created after opening 2024."
        ),
    },
    {
        "check": "2018_2024_retraining_performed",
        "status": "PASS",
        "details": (
            "No. Retraining on 2018-2024 is intentionally deferred "
            "until after Step 21 final strategy freeze."
        ),
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": (
            "No. The 2025 lockbox is not loaded or evaluated."
        ),
    },
    {
        "check": "final_model_selected",
        "status": (
            "PASS"
            if no_final_selection
            else "FAIL"
        ),
        "details": (
            "No final model is selected in Step 20; "
            "descriptive 2024 ranking only."
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
        "One or more Step 20 audit checks failed"
    )


# =============================================================================
# 8. REPORT
# =============================================================================

section(
    "8. SAVE STEP 20 REPORT"
)

report_lines = [
    "STEP 20 - RESERVED 2024 TEMPORAL VALIDATION",
    "=" * 116,
    "",
    "PURPOSE",
    "-" * 116,
    (
        "Evaluate the exact three candidates frozen in Step 19C "
        "on the first reserved unseen temporal validation year, 2024."
    ),
    "",
    "TRAIN / VALIDATION POLICY",
    "-" * 116,
    "Fit years: 2018-2023.",
    "Reserved validation year: 2024.",
    "2025 final lockbox: NOT loaded.",
    "No hyperparameter retuning.",
    "No new strategy search.",
    "No final model selection in Step 20.",
    "",
    "FROZEN CANDIDATES",
    "-" * 116,
]

for candidate in manifest_candidates:
    report_lines.append(
        f"{candidate['shortlist_order']}. "
        f"{candidate['display_name']}"
    )

report_lines.extend(
    [
        "",
        "B5 2024 ANNUAL LEVEL",
        "-" * 116,
        (
            f"Prior-mean level from 2018-2023 only = "
            f"{prior_mean_level_2024:.6f} °C"
        ),
        (
            f"Actual 2024 annual mean (evaluation only, after predictions) = "
            f"{actual_2024_level:.6f} °C"
        ),
        "",
        "RESERVED 2024 RESULTS",
        "-" * 116,
        comparison[
            [
                "descriptive_2024_mae_rank",
                "display_name",
                "mae",
                "rmse",
                "r2",
                "mape_percent",
                "mean_prediction_bias_pred_minus_actual_c",
                "centered_spatial_r2_evaluation_only",
            ]
        ].to_string(
            index=False
        ),
        "",
        "INTERPRETATION POLICY",
        "-" * 116,
        (
            "These results are the first reserved 2024 temporal evidence. "
            "They must not trigger retuning or creation of a new strategy."
        ),
        (
            "Step 21 should combine pre-2024 development evidence with "
            "this reserved 2024 evidence and freeze one final strategy."
        ),
        (
            "Only after Step 21 freeze should the final strategy be retrained "
            "on 2018-2024 for the one-time 2025 test."
        ),
    ]
)

REPORT_PATH.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)

print(f"Saved: {REPORT_PATH}")
print(f"Saved: {AUDIT_PATH}")


# =============================================================================
# 9. HASHES
# =============================================================================

section(
    "9. REPRODUCIBILITY HASHES"
)

hash_files = [
    TRAIN_PATH,
    VALIDATION_PATH,
    STEP19C_AUDIT_PATH,
    STEP19C_SHORTLIST_PATH,
    STEP19C_MANIFEST_PATH,
    ANNUAL_LEVEL_PATH,
    METRICS_PATH,
    COMPARISON_PATH,
    PREDICTIONS_PATH,
    AUDIT_PATH,
    REPORT_PATH,
    *saved_model_paths.values(),
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
# FINAL STATUS
# =============================================================================

section(
    "STEP 20 COMPLETED SUCCESSFULLY"
)

print(
    "Reserved 2024 temporal validation completed.\n"
    "Exactly the three Step 19C-frozen candidates were evaluated.\n"
    "Each candidate was fit using 2018-2023 only.\n"
    "B5 annual-level forecast used only 2018-2023 annual target history.\n"
    "B5 2024 anomaly centering used prediction values only.\n"
    "2024 target was used only after candidate predictions were constructed.\n"
    "No hyperparameter tuning or new strategy search occurred.\n"
    "No 2018-2024 retraining occurred yet.\n"
    "2025 final lockbox remained untouched.\n"
    "No final model was selected in this step.\n"
    "NEXT: Step 21 final strategy selection/freeze; then retrain the frozen "
    "strategy on 2018-2024 before the one-time 2025 test."
)
