"""
STEP 22 - FINAL UNSEEN 2025 LOCKBOX TEST
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Evaluate the FINAL strategy frozen in Step 21 on the untouched 2025 lockbox
exactly once.

FINAL FROZEN STRATEGY
---------------------
LightGBM
+ within-year spatial LST anomaly target
+ prior-mean annual thermal level
+ full frozen 93-feature configuration

STRICT RULES
------------
- DO NOT retrain in Step 22.
- DO NOT tune hyperparameters.
- DO NOT change features.
- DO NOT change annual-level rule.
- DO NOT create a new strategy after seeing 2025.
- Construct 2025 predictions BEFORE accessing the 2025 target for metrics.
- Re-center 2025 anomaly predictions using prediction values only.
- Use the Step-21-frozen 2025 prior-mean annual level.
- 2025 target is used ONLY for final evaluation.

IMPORTANT INTERPRETATION
------------------------
This is the one-time out-of-time test of the frozen LST model using the
prepared 2025 predictor set. Future 2026+ forecasting later requires future
predictor forecasting/scenarios before applying this frozen LST strategy.
"""

from pathlib import Path
import hashlib
import json
import math

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

TEST_YEAR = 2025
EXPECTED_ROWS = 2194
EXPECTED_GRIDS = 2194
EXPECTED_FEATURES = 93

FINAL_STRATEGY = "lgbm_anomaly_plus_prior_mean_level"
FINAL_BASE_MODEL = "LightGBM"

# Hashes printed by the successfully completed Step 21 / earlier frozen data.
EXPECTED_FREEZE_MANIFEST_SHA256 = (
    "2cd2fb44f0780287d7ff5ddba6b44410ef6966c8cc9cf554c0fff83c71989a8c"
)
EXPECTED_FINAL_MODEL_SHA256 = (
    "0531dc08c3f488a82ac3e8b4a7f4e17a22ac9a9a8b3c3a1ad879ef23ecb8bb7c"
)
EXPECTED_FEATURE_LIST_SHA256 = (
    "6769a195b5027850ebd458a4bef7c9da0ecc43c4969bcc81f746e46bce3be9c0"
)
EXPECTED_STEP21_AUDIT_SHA256 = (
    "f03fc7b4cbf09f91a044c0388fce42f34e9df919867adf99faad3986f2f2b649"
)
EXPECTED_2025_READY_SHA256 = (
    "37d6403434e9341077c8f5d75d10ea5a4904915b924fc2dbba4581dbe5ad9464"
)

ZERO_MEAN_TOL = 1e-10


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    ROOT
    / "data"
    / "model_ready"
    / "tree"
    / "tree_test_2025_LOCKED_ready.csv"
)

STEP21_AUDIT_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "21_final_strategy_selection_audit.csv"
)

FREEZE_MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "21_final_strategy_freeze.json"
)

FEATURE_LIST_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "21_final_feature_list.csv"
)

FINAL_MODEL_PATH = (
    ROOT
    / "models"
    / "final"
    / "21_final_lightgbm_anomaly_trained_2018_2024.joblib"
)

METRICS_DIR = ROOT / "outputs" / "metrics"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
REPORTS_DIR = ROOT / "outputs" / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"

for folder in [
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

METRICS_PATH = (
    METRICS_DIR
    / "22_final_2025_lockbox_metrics.csv"
)

PREDICTIONS_PATH = (
    PREDICTIONS_DIR
    / "22_final_2025_lockbox_predictions.csv"
)

AUDIT_PATH = (
    REPORTS_DIR
    / "22_final_2025_lockbox_audit.csv"
)

REPORT_PATH = (
    REPORTS_DIR
    / "22_final_2025_lockbox_report.txt"
)

HASH_PATH = (
    REPORTS_DIR
    / "22_final_2025_lockbox_hashes.csv"
)

FIG_SCATTER_PATH = (
    FIGURES_DIR
    / "22_final_2025_actual_vs_predicted.png"
)

FIG_RESIDUAL_PATH = (
    FIGURES_DIR
    / "22_final_2025_residual_distribution.png"
)


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 128)
    print(title)
    print("=" * 128)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file missing: {path}"
        )


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def require_hash(path, expected, label):
    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual  : {actual}\n"
            "STOP: a frozen Step-21 artifact or the locked 2025 file changed."
        )

    print(
        f"PASS hash: {label} = {actual}"
    )

    return actual


def require_audit_all_pass(path, label):
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


def centered_spatial_r2(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return float(
        r2_score(
            y_true
            - np.mean(y_true),
            y_pred
            - np.mean(y_pred),
        )
    )


# =============================================================================
# 0. VERIFY FROZEN STEP-21 ARTIFACTS BEFORE OPENING 2025
# =============================================================================

section(
    "STEP 22 - FINAL UNSEEN 2025 LOCKBOX TEST"
)

for path in [
    TEST_PATH,
    STEP21_AUDIT_PATH,
    FREEZE_MANIFEST_PATH,
    FEATURE_LIST_PATH,
    FINAL_MODEL_PATH,
]:
    require_file(path)

# Verify Step 21 first, before using the final model.
require_audit_all_pass(
    STEP21_AUDIT_PATH,
    "Step 21",
)

print("PASS: Step 21 audit is fully PASS.")

hash_step21_audit = require_hash(
    STEP21_AUDIT_PATH,
    EXPECTED_STEP21_AUDIT_SHA256,
    "Step 21 audit",
)

hash_manifest = require_hash(
    FREEZE_MANIFEST_PATH,
    EXPECTED_FREEZE_MANIFEST_SHA256,
    "Step 21 final freeze manifest",
)

hash_features = require_hash(
    FEATURE_LIST_PATH,
    EXPECTED_FEATURE_LIST_SHA256,
    "Step 21 frozen feature list",
)

hash_model = require_hash(
    FINAL_MODEL_PATH,
    EXPECTED_FINAL_MODEL_SHA256,
    "Step 21 final trained model",
)

# This confirms the test file is the same frozen file created earlier.
hash_test_input = require_hash(
    TEST_PATH,
    EXPECTED_2025_READY_SHA256,
    "2025 locked model-ready file",
)

print(
    "\nFrozen Step-21 artifacts verified unchanged."
)
print(
    "2025 lockbox may now be opened for the ONE-TIME final test."
)


# =============================================================================
# 1. LOAD FINAL FREEZE MANIFEST
# =============================================================================

section(
    "1. LOAD FINAL STEP-21 FREEZE MANIFEST"
)

with FREEZE_MANIFEST_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    freeze = json.load(f)

if freeze.get(
    "status"
) != "FINAL_STRATEGY_FROZEN_BEFORE_2025":
    raise RuntimeError(
        "Unexpected Step-21 freeze status"
    )

if freeze.get(
    "selected_strategy"
) != FINAL_STRATEGY:
    raise RuntimeError(
        "Frozen final strategy does not match expected strategy"
    )

if freeze.get(
    "selected_base_model"
) != FINAL_BASE_MODEL:
    raise RuntimeError(
        "Frozen final base model does not match expected LightGBM"
    )

if freeze.get(
    "hyperparameters_retuned_after_2024"
) is not False:
    raise RuntimeError(
        "Freeze manifest indicates post-2024 retuning"
    )

if freeze.get(
    "features_changed_after_2024"
) is not False:
    raise RuntimeError(
        "Freeze manifest indicates post-2024 feature changes"
    )

if freeze.get(
    "new_strategy_created_after_2024"
) is not False:
    raise RuntimeError(
        "Freeze manifest indicates a post-2024 strategy change"
    )

if freeze.get(
    "final_strategy_frozen_before_2025"
) is not True:
    raise RuntimeError(
        "Final strategy was not frozen before 2025"
    )

inference_rule = freeze[
    "frozen_2025_inference_rule"
]

if inference_rule.get(
    "annual_level_rule"
) != "prior_mean_level":
    raise RuntimeError(
        "Frozen annual-level rule changed"
    )

frozen_2025_level = float(
    inference_rule[
        "annual_level_forecast_c"
    ]
)

history_years = list(
    inference_rule[
        "annual_level_history_years"
    ]
)

if history_years != [
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
]:
    raise RuntimeError(
        f"Unexpected annual-level history years: {history_years}"
    )

print(
    f"Frozen final strategy    : {freeze['selected_strategy']}"
)

print(
    f"Frozen base model        : {freeze['selected_base_model']}"
)

print(
    f"Frozen feature count     : "
    f"{freeze['final_model_configuration']['feature_count']}"
)

print(
    f"Frozen training years    : "
    f"{freeze['final_model_configuration']['training_years_for_2025_test_model']}"
)

print(
    f"Frozen 2025 annual level : {frozen_2025_level:.6f} °C"
)

print(
    "No Step-22 model changes are permitted."
)


# =============================================================================
# 2. LOAD FROZEN FEATURE LIST + FINAL MODEL
# =============================================================================

section(
    "2. LOAD FROZEN FEATURE LIST + FINAL MODEL"
)

feature_df = pd.read_csv(
    FEATURE_LIST_PATH
)

required_feature_cols = {
    "feature_order",
    "feature",
}

if not required_feature_cols.issubset(
    feature_df.columns
):
    raise RuntimeError(
        "Frozen feature-list schema is invalid"
    )

feature_df = feature_df.sort_values(
    "feature_order"
).reset_index(
    drop=True
)

predictors = (
    feature_df[
        "feature"
    ]
    .astype(str)
    .tolist()
)

if len(predictors) != EXPECTED_FEATURES:
    raise RuntimeError(
        f"Frozen feature count={len(predictors)}, "
        f"expected={EXPECTED_FEATURES}"
    )

if len(
    set(
        predictors
    )
) != len(
    predictors
):
    raise RuntimeError(
        "Duplicate predictors in frozen feature list"
    )

final_model = joblib.load(
    FINAL_MODEL_PATH
)

print(
    f"Frozen predictors loaded : {len(predictors)}"
)

print(
    f"Final model artifact      : {FINAL_MODEL_PATH}"
)

print(
    "Model is loaded only. No .fit() call occurs anywhere in Step 22."
)


# =============================================================================
# 3. OPEN 2025 LOCKBOX PREDICTORS
# =============================================================================

section(
    "3. OPEN 2025 LOCKBOX PREDICTORS"
)

test_df = pd.read_csv(
    TEST_PATH,
    dtype={
        GRID_COL: str,
    },
)

required_columns = {
    TARGET,
    GRID_COL,
    AREA_COL,
    YEAR_COL,
    *predictors,
}

missing_columns = sorted(
    required_columns
    - set(
        test_df.columns
    )
)

if missing_columns:
    raise RuntimeError(
        f"2025 file missing required columns: {missing_columns}"
    )

test_years = sorted(
    pd.to_numeric(
        test_df[
            YEAR_COL
        ],
        errors="coerce",
    )
    .astype(int)
    .unique()
    .tolist()
)

if test_years != [
    TEST_YEAR
]:
    raise RuntimeError(
        f"Test years={test_years}; expected=[2025]"
    )

if len(
    test_df
) != EXPECTED_ROWS:
    raise RuntimeError(
        f"2025 rows={len(test_df)}, expected={EXPECTED_ROWS}"
    )

if test_df[
    GRID_COL
].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        "Unexpected 2025 unique-grid count"
    )

if test_df.duplicated(
    subset=[
        GRID_COL,
        YEAR_COL,
    ]
).any():
    raise RuntimeError(
        "Duplicate grid-year rows found in 2025"
    )

# Convert predictors only.
# IMPORTANT: TARGET is intentionally not assigned to y_test yet.
for column in predictors:
    test_df[
        column
    ] = pd.to_numeric(
        test_df[
            column
        ],
        errors="coerce",
    )

if test_df[
    predictors
].isna().any().any():
    raise RuntimeError(
        "Missing predictor values found in 2025"
    )

if not np.isfinite(
    test_df[
        predictors
    ].to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "Non-finite predictor values found in 2025"
    )

X_2025 = test_df[
    predictors
].copy()

print(
    f"2025 rows         : {len(test_df):,}"
)

print(
    f"2025 unique grids : {test_df[GRID_COL].nunique():,}"
)

print(
    f"Predictors        : {len(predictors)}"
)

print(
    "2025 target       : WITHHELD from prediction construction"
)


# =============================================================================
# 4. CONSTRUCT FINAL 2025 PREDICTIONS BEFORE ACCESSING TARGET
# =============================================================================

section(
    "4. CONSTRUCT FINAL FROZEN 2025 PREDICTIONS"
)

raw_anomaly_pred = np.asarray(
    final_model.predict(
        X_2025
    ),
    dtype=float,
)

if len(
    raw_anomaly_pred
) != EXPECTED_ROWS:
    raise RuntimeError(
        "Unexpected 2025 prediction count"
    )

if not np.isfinite(
    raw_anomaly_pred
).all():
    raise RuntimeError(
        "Non-finite raw anomaly predictions"
    )

raw_anomaly_mean = float(
    np.mean(
        raw_anomaly_pred
    )
)

# Frozen leakage-safe rule:
# center using prediction values only.
centered_anomaly_pred = (
    raw_anomaly_pred
    - raw_anomaly_mean
)

centered_anomaly_mean = float(
    np.mean(
        centered_anomaly_pred
    )
)

if abs(
    centered_anomaly_mean
) > ZERO_MEAN_TOL:
    raise RuntimeError(
        "Centered 2025 anomaly prediction does not have zero mean"
    )

final_2025_pred = (
    frozen_2025_level
    + centered_anomaly_pred
)

if not np.isfinite(
    final_2025_pred
).all():
    raise RuntimeError(
        "Non-finite final 2025 predictions"
    )

prediction_mean_before_target = float(
    np.mean(
        final_2025_pred
    )
)

if not np.isclose(
    prediction_mean_before_target,
    frozen_2025_level,
    atol=ZERO_MEAN_TOL,
):
    raise RuntimeError(
        "Final 2025 prediction mean does not equal frozen annual level"
    )

# Copy/freeze prediction arrays before opening target for evaluation.
raw_anomaly_pred = raw_anomaly_pred.copy()
centered_anomaly_pred = centered_anomaly_pred.copy()
final_2025_pred = final_2025_pred.copy()

print(
    f"Raw predicted anomaly mean      : {raw_anomaly_mean:+.6f} °C"
)

print(
    f"Centered anomaly mean           : {centered_anomaly_mean:+.12f} °C"
)

print(
    f"Frozen annual level             : {frozen_2025_level:.6f} °C"
)

print(
    f"Mean final 2025 prediction      : {prediction_mean_before_target:.6f} °C"
)

print(
    "\nPASS: final 2025 predictions constructed BEFORE target evaluation."
)


# =============================================================================
# 5. OPEN 2025 TARGET FOR FINAL EVALUATION ONLY
# =============================================================================

section(
    "5. FINAL ONE-TIME 2025 LOCKBOX EVALUATION"
)

test_df[
    TARGET
] = pd.to_numeric(
    test_df[
        TARGET
    ],
    errors="coerce",
)

if test_df[
    TARGET
].isna().any():
    raise RuntimeError(
        "Missing 2025 target values found"
    )

y_test = test_df[
    TARGET
].to_numpy(
    dtype=float
)

if not np.isfinite(
    y_test
).all():
    raise RuntimeError(
        "Non-finite 2025 target values found"
    )

actual_2025_level = float(
    np.mean(
        y_test
    )
)

level_error = (
    frozen_2025_level
    - actual_2025_level
)

mae = float(
    mean_absolute_error(
        y_test,
        final_2025_pred,
    )
)

rmse = float(
    math.sqrt(
        mean_squared_error(
            y_test,
            final_2025_pred,
        )
    )
)

r2 = float(
    r2_score(
        y_test,
        final_2025_pred,
    )
)

mape = mape_percent(
    y_test,
    final_2025_pred,
)

bias = float(
    np.mean(
        final_2025_pred
        - y_test
    )
)

centered_r2 = centered_spatial_r2(
    y_test,
    final_2025_pred,
)

residual = (
    y_test
    - final_2025_pred
)

metrics = pd.DataFrame(
    [
        {
            "strategy": FINAL_STRATEGY,
            "base_model": FINAL_BASE_MODEL,
            "train_years": "2018-2024",
            "test_year": 2025,
            "test_rows": len(
                test_df
            ),
            "unique_grids": test_df[
                GRID_COL
            ].nunique(),
            "predictor_count": len(
                predictors
            ),
            "frozen_2025_annual_level_c": frozen_2025_level,
            "actual_2025_annual_level_c_evaluation_only": actual_2025_level,
            "annual_level_error_forecast_minus_actual_c": level_error,
            "abs_annual_level_error_c": abs(
                level_error
            ),
            "mae_c": mae,
            "rmse_c": rmse,
            "r2": r2,
            "mape_percent": mape,
            "bias_pred_minus_actual_c": bias,
            "abs_bias_c": abs(
                bias
            ),
            "centered_spatial_r2_evaluation_only": centered_r2,
            "mean_actual_lst_c": actual_2025_level,
            "mean_predicted_lst_c": float(
                np.mean(
                    final_2025_pred
                )
            ),
            "median_absolute_error_c": float(
                np.median(
                    np.abs(
                        residual
                    )
                )
            ),
            "p90_absolute_error_c": float(
                np.quantile(
                    np.abs(
                        residual
                    ),
                    0.90,
                )
            ),
            "min_actual_lst_c": float(
                np.min(
                    y_test
                )
            ),
            "max_actual_lst_c": float(
                np.max(
                    y_test
                )
            ),
            "min_predicted_lst_c": float(
                np.min(
                    final_2025_pred
                )
            ),
            "max_predicted_lst_c": float(
                np.max(
                    final_2025_pred
                )
            ),
        }
    ]
)

metrics.to_csv(
    METRICS_PATH,
    index=False,
)

print(
    f"Actual 2025 annual mean LST : {actual_2025_level:.6f} °C"
)

print(
    f"Frozen forecast annual level: {frozen_2025_level:.6f} °C"
)

print(
    f"Annual-level error          : {level_error:+.6f} °C"
)

print(
    "\nFINAL 2025 TEST PERFORMANCE"
)

print(
    f"MAE                       : {mae:.6f} °C"
)

print(
    f"RMSE                      : {rmse:.6f} °C"
)

print(
    f"R²                        : {r2:.6f}"
)

print(
    f"MAPE                      : {mape:.6f}%"
)

print(
    f"Bias (predicted - actual) : {bias:+.6f} °C"
)

print(
    f"Centered spatial R²       : {centered_r2:.6f}"
)

print(
    "\nIMPORTANT: this is the final untouched 2025 result. "
    "Do not tune or change the model because of this result."
)


# =============================================================================
# 6. SAVE 2,194 GRID-LEVEL FINAL TEST PREDICTIONS
# =============================================================================

section(
    "6. SAVE GRID-LEVEL 2025 PREDICTIONS"
)

predictions = pd.DataFrame(
    {
        GRID_COL: test_df[
            GRID_COL
        ].astype(
            str
        ),
        AREA_COL: test_df[
            AREA_COL
        ].astype(
            str
        ),
        YEAR_COL: 2025,
        "actual_lst_c": y_test,
        "raw_predicted_spatial_anomaly_c": raw_anomaly_pred,
        "centered_predicted_spatial_anomaly_c": centered_anomaly_pred,
        "frozen_annual_level_c": frozen_2025_level,
        "predicted_lst_c": final_2025_pred,
        "residual_actual_minus_predicted_c": residual,
        "absolute_error_c": np.abs(
            residual
        ),
    }
)

predictions.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

print(
    f"Saved {len(predictions):,} grid predictions:"
)

print(
    PREDICTIONS_PATH
)


# =============================================================================
# 7. DIAGNOSTIC FIGURES
# =============================================================================

section(
    "7. FINAL TEST DIAGNOSTIC FIGURES"
)

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.scatter(
    y_test,
    final_2025_pred,
    s=12,
    alpha=0.45,
)

low = min(
    float(
        np.min(
            y_test
        )
    ),
    float(
        np.min(
            final_2025_pred
        )
    ),
)

high = max(
    float(
        np.max(
            y_test
        )
    ),
    float(
        np.max(
            final_2025_pred
        )
    ),
)

ax.plot(
    [
        low,
        high,
    ],
    [
        low,
        high,
    ],
    linewidth=1,
)

ax.set_xlabel(
    "Actual 2025 LST (°C)"
)

ax.set_ylabel(
    "Predicted 2025 LST (°C)"
)

ax.set_title(
    "Step 22 - Final 2025 Actual vs Predicted LST"
)

ax.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    FIG_SCATTER_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.hist(
    residual,
    bins=40,
)

ax.axvline(
    0,
    linewidth=1,
)

ax.set_xlabel(
    "Residual: Actual - Predicted (°C)"
)

ax.set_ylabel(
    "Grid count"
)

ax.set_title(
    "Step 22 - Final 2025 Residual Distribution"
)

ax.grid(
    axis="y",
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    FIG_RESIDUAL_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    f"Saved: {FIG_SCATTER_PATH}"
)

print(
    f"Saved: {FIG_RESIDUAL_PATH}"
)


# =============================================================================
# 8. STRICT FINAL-TEST AUDIT
# =============================================================================

section(
    "8. STRICT STEP-22 AUDIT"
)

prediction_count_ok = bool(
    len(
        predictions
    )
    == EXPECTED_ROWS
)

unique_grids_ok = bool(
    predictions[
        GRID_COL
    ].nunique()
    == EXPECTED_GRIDS
)

centered_mean_ok = bool(
    abs(
        centered_anomaly_mean
    )
    <= ZERO_MEAN_TOL
)

prediction_mean_ok = bool(
    np.isclose(
        np.mean(
            final_2025_pred
        ),
        frozen_2025_level,
        atol=ZERO_MEAN_TOL,
    )
)

metrics_finite = bool(
    np.isfinite(
        [
            actual_2025_level,
            level_error,
            mae,
            rmse,
            r2,
            mape,
            bias,
            centered_r2,
        ]
    ).all()
)

audit_rows = [
    {
        "check": "step21_audit",
        "status": "PASS",
        "details": (
            "Step 21 audit loaded and fully PASS before 2025 evaluation."
        ),
    },
    {
        "check": "step21_freeze_manifest_hash",
        "status": "PASS",
        "details": hash_manifest,
    },
    {
        "check": "step21_final_model_hash",
        "status": "PASS",
        "details": hash_model,
    },
    {
        "check": "step21_feature_list_hash",
        "status": "PASS",
        "details": hash_features,
    },
    {
        "check": "step21_audit_hash",
        "status": "PASS",
        "details": hash_step21_audit,
    },
    {
        "check": "2025_locked_input_hash",
        "status": "PASS",
        "details": hash_test_input,
    },
    {
        "check": "final_strategy_unchanged",
        "status": "PASS",
        "details": FINAL_STRATEGY,
    },
    {
        "check": "final_base_model_unchanged",
        "status": "PASS",
        "details": FINAL_BASE_MODEL,
    },
    {
        "check": "test_year",
        "status": (
            "PASS"
            if test_years
            == [
                2025
            ]
            else "FAIL"
        ),
        "details": str(
            test_years
        ),
    },
    {
        "check": "test_rows",
        "status": (
            "PASS"
            if prediction_count_ok
            else "FAIL"
        ),
        "details": (
            f"actual={len(predictions)}, "
            f"expected={EXPECTED_ROWS}"
        ),
    },
    {
        "check": "unique_grids",
        "status": (
            "PASS"
            if unique_grids_ok
            else "FAIL"
        ),
        "details": (
            f"actual={predictions[GRID_COL].nunique()}, "
            f"expected={EXPECTED_GRIDS}"
        ),
    },
    {
        "check": "frozen_feature_count",
        "status": (
            "PASS"
            if len(
                predictors
            )
            == EXPECTED_FEATURES
            else "FAIL"
        ),
        "details": (
            f"actual={len(predictors)}, "
            f"expected={EXPECTED_FEATURES}"
        ),
    },
    {
        "check": "model_retrained_in_step22",
        "status": "PASS",
        "details": (
            "No. Final Step-21 model artifact loaded and predict() only."
        ),
    },
    {
        "check": "hyperparameters_changed_in_step22",
        "status": "PASS",
        "details": (
            "No. Step-22 contains no tuning or parameter search."
        ),
    },
    {
        "check": "features_changed_in_step22",
        "status": "PASS",
        "details": (
            "No. Exact frozen feature-list order used."
        ),
    },
    {
        "check": "annual_level_rule_changed",
        "status": "PASS",
        "details": (
            f"No. Used frozen prior-mean annual level="
            f"{frozen_2025_level:.6f} °C."
        ),
    },
    {
        "check": "2025_anomaly_recentering_target_used",
        "status": "PASS",
        "details": (
            "No. Re-centering used only the mean of 2025 model predictions."
        ),
    },
    {
        "check": "centered_predicted_anomaly_mean_zero",
        "status": (
            "PASS"
            if centered_mean_ok
            else "FAIL"
        ),
        "details": (
            f"mean={centered_anomaly_mean:+.12f} °C"
        ),
    },
    {
        "check": "prediction_mean_equals_frozen_level",
        "status": (
            "PASS"
            if prediction_mean_ok
            else "FAIL"
        ),
        "details": (
            f"prediction_mean={np.mean(final_2025_pred):.12f}, "
            f"frozen_level={frozen_2025_level:.12f}"
        ),
    },
    {
        "check": "predictions_constructed_before_target_evaluation",
        "status": "PASS",
        "details": (
            "Yes. Frozen predictions were constructed and copied before "
            "y_test was assigned for final metrics."
        ),
    },
    {
        "check": "all_final_metrics_finite",
        "status": (
            "PASS"
            if metrics_finite
            else "FAIL"
        ),
        "details": str(
            metrics_finite
        ),
    },
    {
        "check": "post_2025_model_changes_allowed",
        "status": "PASS",
        "details": (
            "No. Final lockbox result must be reported as observed; "
            "it must not trigger tuning or strategy replacement."
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
        "One or more Step-22 audit checks failed"
    )


# =============================================================================
# 9. FINAL REPORT
# =============================================================================

section(
    "9. SAVE FINAL 2025 TEST REPORT"
)

report_lines = [
    "STEP 22 - FINAL UNSEEN 2025 LOCKBOX TEST",
    "=" * 116,
    "",
    "FINAL FROZEN STRATEGY",
    "-" * 116,
    f"Strategy: {FINAL_STRATEGY}",
    f"Base model: {FINAL_BASE_MODEL}",
    "Model training years: 2018-2024",
    "Target: within-year spatial LST anomaly",
    f"Frozen feature count: {len(predictors)}",
    "Annual-level rule: prior mean",
    f"Frozen 2025 annual-level forecast: {frozen_2025_level:.6f} °C",
    "",
    "FINAL UNSEEN 2025 PERFORMANCE",
    "-" * 116,
    f"MAE: {mae:.6f} °C",
    f"RMSE: {rmse:.6f} °C",
    f"R²: {r2:.6f}",
    f"MAPE: {mape:.6f}%",
    f"Bias (predicted - actual): {bias:+.6f} °C",
    f"Centered spatial R²: {centered_r2:.6f}",
    "",
    "ANNUAL LEVEL",
    "-" * 116,
    f"Frozen forecast annual mean: {frozen_2025_level:.6f} °C",
    f"Actual 2025 annual mean: {actual_2025_level:.6f} °C",
    f"Annual-level forecast error: {level_error:+.6f} °C",
    "",
    "LOCKBOX POLICY",
    "-" * 116,
    (
        "The 2025 result is the final one-time untouched test result. "
        "No model, feature, hyperparameter, or annual-level-rule changes "
        "may be made in response to this result."
    ),
    "",
    "INTERPRETATION NOTE",
    "-" * 116,
    (
        "This test evaluates the frozen LST strategy using the prepared "
        "2025 predictor set. Future 2026+ forecasting requires future "
        "predictor forecasting/scenarios before the LST model is applied."
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

print(
    f"Saved: {AUDIT_PATH}"
)


# =============================================================================
# 10. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "10. REPRODUCIBILITY HASHES"
)

hash_files = [
    TEST_PATH,
    STEP21_AUDIT_PATH,
    FREEZE_MANIFEST_PATH,
    FEATURE_LIST_PATH,
    FINAL_MODEL_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    AUDIT_PATH,
    REPORT_PATH,
    FIG_SCATTER_PATH,
    FIG_RESIDUAL_PATH,
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
    "STEP 22 COMPLETED SUCCESSFULLY"
)

print(
    "FINAL unseen 2025 lockbox test completed.\n"
    "The Step-21-frozen LightGBM anomaly + prior-mean strategy was used unchanged.\n"
    "The model was NOT retrained in Step 22.\n"
    "No hyperparameter tuning, feature changes, annual-level changes, "
    "or strategy search occurred.\n"
    "2025 predictions were constructed before accessing the target for metrics.\n"
    f"Final 2025 MAE  : {mae:.6f} °C\n"
    f"Final 2025 RMSE : {rmse:.6f} °C\n"
    f"Final 2025 R²   : {r2:.6f}\n"
    f"Final 2025 MAPE : {mape:.6f}%\n"
    f"Final 2025 bias : {bias:+.6f} °C\n"
    f"Spatial R²      : {centered_r2:.6f}\n"
    "\nIMPORTANT: report this lockbox result exactly as observed. "
    "Do not tune the model because of the 2025 result.\n"
    "NEXT: preserve this final-test evidence, then retrain the same frozen "
    "production strategy on 2018-2025 for 2026+ forecasting."
)
