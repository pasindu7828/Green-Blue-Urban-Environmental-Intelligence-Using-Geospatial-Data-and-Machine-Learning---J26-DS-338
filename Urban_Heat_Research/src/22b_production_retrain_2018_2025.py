"""
STEP 22B - PRODUCTION RETRAIN ON 2018-2025
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
After the one-time final 2025 lockbox test has been completed and preserved,
retrain the SAME frozen final strategy using all observed modeling years
2018-2025.

This creates the production spatial-anomaly model for later 2026+ forecasting.

STRICT RULES
------------
- Step 22 final 2025 test evidence must already exist and remain unchanged.
- Do NOT retune hyperparameters.
- Do NOT change the 93 frozen predictors.
- Do NOT change the selected LightGBM anomaly strategy.
- Do NOT use 2018-2025 training-fit metrics as a new test score.
- 2025 is now allowed in training ONLY because the final lockbox evaluation
  was already completed and permanently preserved in Step 22.
- Future annual thermal level forecasting is NOT decided in Step 22B.
  That is handled later in the forecasting stages.

FINAL FROZEN STRATEGY
---------------------
LightGBM
+ within-year spatial LST anomaly target
+ same frozen 93 predictors
+ exact frozen hyperparameter configuration

PRODUCTION TRAINING PERIOD
--------------------------
2018-2025
8 years x 2,194 grids = 17,552 rows
"""

from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone


# =============================================================================
# SETTINGS
# =============================================================================

TARGET = "LST_C"
GRID_COL = "grid_id"
AREA_COL = "area_name"
YEAR_COL = "year"

EXPECTED_YEARS = list(range(2018, 2026))
EXPECTED_GRIDS = 2194
EXPECTED_ROWS_PER_YEAR = 2194
EXPECTED_TOTAL_ROWS = 17552
EXPECTED_FEATURES = 93

FINAL_STRATEGY = "lgbm_anomaly_plus_prior_mean_level"
FINAL_BASE_MODEL = "LightGBM"

ZERO_TOL = 1e-10

# Frozen source hashes already established in Steps 10/21/22.
EXPECTED_TRAIN_HASH = (
    "37139f64919d50ef09ccdb8453fa48a6e522894742c4517fb1b3e47fead6fa9c"
)

EXPECTED_VALIDATION_HASH = (
    "9547a58ecabe6946df6929191a851e5761c884e8d83c12ff059b3a90aafb752b"
)

EXPECTED_TEST_HASH = (
    "37d6403434e9341077c8f5d75d10ea5a4904915b924fc2dbba4581dbe5ad9464"
)

EXPECTED_STEP21_MANIFEST_HASH = (
    "2cd2fb44f0780287d7ff5ddba6b44410ef6966c8cc9cf554c0fff83c71989a8c"
)

EXPECTED_STEP21_FEATURE_LIST_HASH = (
    "6769a195b5027850ebd458a4bef7c9da0ecc43c4969bcc81f746e46bce3be9c0"
)

EXPECTED_STEP21_FINAL_MODEL_HASH = (
    "0531dc08c3f488a82ac3e8b4a7f4e17a22ac9a9a8b3c3a1ad879ef23ecb8bb7c"
)

EXPECTED_STEP22_AUDIT_HASH = (
    "470e281aa30d7b1518c23a49f4f374068e6754f374af5d8a1a600eb5a597fda9"
)

EXPECTED_STEP22_METRICS_HASH = (
    "49f784bc32b6d5c252822a17d61e94f1958298a6482e96597ed8fd3cfac7aa61"
)

EXPECTED_STEP22_REPORT_HASH = (
    "cccb534437ca66698be98190422a3b54bf11519698b4fee9a32dec65f1965b99"
)


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    ROOT
    / "data"
    / "model_ready"
    / "tree"
    / "tree_train_2018_2023_ready.csv"
)

VALIDATION_PATH = (
    ROOT
    / "data"
    / "model_ready"
    / "tree"
    / "tree_validation_2024_ready.csv"
)

TEST_PATH = (
    ROOT
    / "data"
    / "model_ready"
    / "tree"
    / "tree_test_2025_LOCKED_ready.csv"
)

STEP21_MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "21_final_strategy_freeze.json"
)

STEP21_FEATURE_LIST_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "21_final_feature_list.csv"
)

STEP21_FINAL_MODEL_PATH = (
    ROOT
    / "models"
    / "final"
    / "21_final_lightgbm_anomaly_trained_2018_2024.joblib"
)

STEP22_AUDIT_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "22_final_2025_lockbox_audit.csv"
)

STEP22_METRICS_PATH = (
    ROOT
    / "outputs"
    / "metrics"
    / "22_final_2025_lockbox_metrics.csv"
)

STEP22_REPORT_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "22_final_2025_lockbox_report.txt"
)

MODELS_FINAL_DIR = ROOT / "models" / "final"
REPORTS_DIR = ROOT / "outputs" / "reports"
METRICS_DIR = ROOT / "outputs" / "metrics"

for folder in [
    MODELS_FINAL_DIR,
    REPORTS_DIR,
    METRICS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

PRODUCTION_MODEL_PATH = (
    MODELS_FINAL_DIR
    / "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
)

ANNUAL_HISTORY_PATH = (
    METRICS_DIR
    / "22b_production_annual_level_history_2018_2025.csv"
)

TRAINING_AUDIT_TABLE_PATH = (
    METRICS_DIR
    / "22b_production_training_year_audit.csv"
)

PRODUCTION_MANIFEST_PATH = (
    REPORTS_DIR
    / "22b_production_model_manifest.json"
)

AUDIT_PATH = (
    REPORTS_DIR
    / "22b_production_retrain_audit.csv"
)

REPORT_PATH = (
    REPORTS_DIR
    / "22b_production_retrain_report.txt"
)

HASH_PATH = (
    REPORTS_DIR
    / "22b_production_retrain_hashes.csv"
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


def verify_hash(path, expected_hash, label):
    actual = sha256_file(
        path
    )

    if actual != expected_hash:
        raise RuntimeError(
            f"{label} hash mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual  : {actual}"
        )

    print(
        f"PASS hash: {label} = {actual}"
    )

    return actual


def require_audit_all_pass(path, label):
    audit = pd.read_csv(
        path
    )

    if "status" not in audit.columns:
        raise RuntimeError(
            f"{label} audit does not contain a status column"
        )

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
            bad.to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"{label} audit is not fully PASS"
        )


def same_value(a, b):
    """
    Robust equality check for frozen hyperparameter values.
    """
    if isinstance(
        b,
        float,
    ):
        try:
            return bool(
                np.isclose(
                    float(
                        a
                    ),
                    float(
                        b
                    ),
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        except Exception:
            return False

    return a == b


# =============================================================================
# 0. VERIFY STEP 22 IS COMPLETE AND PRESERVED
# =============================================================================

section(
    "STEP 22B - PRODUCTION RETRAIN ON 2018-2025"
)

required_files = [
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH,
    STEP21_MANIFEST_PATH,
    STEP21_FEATURE_LIST_PATH,
    STEP21_FINAL_MODEL_PATH,
    STEP22_AUDIT_PATH,
    STEP22_METRICS_PATH,
    STEP22_REPORT_PATH,
]

for path in required_files:
    require_file(
        path
    )

require_audit_all_pass(
    STEP22_AUDIT_PATH,
    "Step 22",
)

print(
    "PASS: Step 22 final lockbox audit is fully PASS."
)

hash_step22_audit = verify_hash(
    STEP22_AUDIT_PATH,
    EXPECTED_STEP22_AUDIT_HASH,
    "Step 22 audit",
)

hash_step22_metrics = verify_hash(
    STEP22_METRICS_PATH,
    EXPECTED_STEP22_METRICS_HASH,
    "Step 22 final metrics",
)

hash_step22_report = verify_hash(
    STEP22_REPORT_PATH,
    EXPECTED_STEP22_REPORT_HASH,
    "Step 22 final report",
)

print(
    "\nPASS: final 2025 test evidence exists and is preserved unchanged."
)

print(
    "2025 may now be used as historical training data for production retraining."
)


# =============================================================================
# 1. VERIFY FROZEN STEP-21 STRATEGY ARTIFACTS
# =============================================================================

section(
    "1. VERIFY FROZEN STEP-21 STRATEGY"
)

hash_manifest = verify_hash(
    STEP21_MANIFEST_PATH,
    EXPECTED_STEP21_MANIFEST_HASH,
    "Step 21 freeze manifest",
)

hash_features = verify_hash(
    STEP21_FEATURE_LIST_PATH,
    EXPECTED_STEP21_FEATURE_LIST_HASH,
    "Step 21 feature list",
)

hash_step21_model = verify_hash(
    STEP21_FINAL_MODEL_PATH,
    EXPECTED_STEP21_FINAL_MODEL_HASH,
    "Step 21 final pre-test model",
)

with STEP21_MANIFEST_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    freeze = json.load(
        f
    )

if freeze.get(
    "selected_strategy"
) != FINAL_STRATEGY:
    raise RuntimeError(
        "Frozen selected strategy changed"
    )

if freeze.get(
    "selected_base_model"
) != FINAL_BASE_MODEL:
    raise RuntimeError(
        "Frozen selected base model changed"
    )

final_config = freeze.get(
    "final_model_configuration",
    {}
)

if final_config.get(
    "feature_count"
) != EXPECTED_FEATURES:
    raise RuntimeError(
        "Frozen feature count changed"
    )

frozen_hyperparameters = final_config.get(
    "hyperparameters",
    {}
)

if not frozen_hyperparameters:
    raise RuntimeError(
        "Frozen hyperparameters missing from Step-21 manifest"
    )

print(
    f"Frozen strategy     : {freeze['selected_strategy']}"
)

print(
    f"Frozen base model   : {freeze['selected_base_model']}"
)

print(
    f"Frozen feature count: {final_config['feature_count']}"
)

print(
    f"Frozen parameters   : {frozen_hyperparameters}"
)


# =============================================================================
# 2. VERIFY THE THREE MODEL-READY DATA BLOCKS
# =============================================================================

section(
    "2. VERIFY 2018-2025 MODEL-READY DATA BLOCKS"
)

hash_train = verify_hash(
    TRAIN_PATH,
    EXPECTED_TRAIN_HASH,
    "2018-2023 train-ready file",
)

hash_validation = verify_hash(
    VALIDATION_PATH,
    EXPECTED_VALIDATION_HASH,
    "2024 validation-ready file",
)

hash_test = verify_hash(
    TEST_PATH,
    EXPECTED_TEST_HASH,
    "2025 former-lockbox ready file",
)

feature_df = pd.read_csv(
    STEP21_FEATURE_LIST_PATH
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

if len(
    predictors
) != EXPECTED_FEATURES:
    raise RuntimeError(
        f"Frozen feature count={len(predictors)}, "
        f"expected={EXPECTED_FEATURES}"
    )

if len(
    predictors
) != len(
    set(
        predictors
    )
):
    raise RuntimeError(
        "Duplicate predictors in frozen feature list"
    )

train_2018_2023 = pd.read_csv(
    TRAIN_PATH,
    dtype={
        GRID_COL: str,
    },
)

validation_2024 = pd.read_csv(
    VALIDATION_PATH,
    dtype={
        GRID_COL: str,
    },
)

history_2025 = pd.read_csv(
    TEST_PATH,
    dtype={
        GRID_COL: str,
    },
)

blocks = [
    (
        "2018-2023",
        train_2018_2023,
        list(
            range(
                2018,
                2024,
            )
        ),
        13164,
    ),
    (
        "2024",
        validation_2024,
        [
            2024
        ],
        2194,
    ),
    (
        "2025",
        history_2025,
        [
            2025
        ],
        2194,
    ),
]

required_cols = {
    TARGET,
    GRID_COL,
    AREA_COL,
    YEAR_COL,
    *predictors,
}

for label, df, expected_years, expected_rows in blocks:
    missing = sorted(
        required_cols
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{label} missing columns: {missing}"
        )

    actual_years = sorted(
        pd.to_numeric(
            df[
                YEAR_COL
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if actual_years != expected_years:
        raise RuntimeError(
            f"{label} years={actual_years}, "
            f"expected={expected_years}"
        )

    if len(
        df
    ) != expected_rows:
        raise RuntimeError(
            f"{label} rows={len(df)}, expected={expected_rows}"
        )

    if df.duplicated(
        subset=[
            GRID_COL,
            YEAR_COL,
        ]
    ).any():
        raise RuntimeError(
            f"Duplicate grid-year rows in {label}"
        )

    print(
        f"{label:<10} rows={len(df):,} years={actual_years}"
    )


# =============================================================================
# 3. COMBINE ALL OBSERVED MODELING YEARS 2018-2025
# =============================================================================

section(
    "3. COMBINE 2018-2025 FOR PRODUCTION TRAINING"
)

production_df = pd.concat(
    [
        train_2018_2023,
        validation_2024,
        history_2025,
    ],
    axis=0,
    ignore_index=True,
)

production_df[
    YEAR_COL
] = pd.to_numeric(
    production_df[
        YEAR_COL
    ],
    errors="coerce",
).astype(
    int
)

production_df[
    TARGET
] = pd.to_numeric(
    production_df[
        TARGET
    ],
    errors="coerce",
)

for feature in predictors:
    production_df[
        feature
    ] = pd.to_numeric(
        production_df[
            feature
        ],
        errors="coerce",
    )

actual_years = sorted(
    production_df[
        YEAR_COL
    ].unique().tolist()
)

if actual_years != EXPECTED_YEARS:
    raise RuntimeError(
        f"Combined years={actual_years}, expected={EXPECTED_YEARS}"
    )

if len(
    production_df
) != EXPECTED_TOTAL_ROWS:
    raise RuntimeError(
        f"Combined rows={len(production_df)}, "
        f"expected={EXPECTED_TOTAL_ROWS}"
    )

if production_df[
    GRID_COL
].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        "Combined production data has unexpected unique-grid count"
    )

if production_df.duplicated(
    subset=[
        GRID_COL,
        YEAR_COL,
    ]
).any():
    raise RuntimeError(
        "Duplicate grid-year rows after combining production data"
    )

year_counts = (
    production_df.groupby(
        YEAR_COL
    )
    .size()
)

if not (
    year_counts
    == EXPECTED_ROWS_PER_YEAR
).all():
    raise RuntimeError(
        "Every production year must have exactly 2,194 rows"
    )

grid_year_counts = (
    production_df.groupby(
        GRID_COL
    )[
        YEAR_COL
    ]
    .nunique()
)

if not (
    grid_year_counts
    == len(
        EXPECTED_YEARS
    )
).all():
    raise RuntimeError(
        "Every grid must appear in all 8 production-training years"
    )

if production_df[
    [
        TARGET,
        *predictors,
    ]
].isna().any().any():
    raise RuntimeError(
        "Missing target/predictor values found in production data"
    )

if not np.isfinite(
    production_df[
        [
            TARGET,
            *predictors,
        ]
    ].to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "Non-finite target/predictor values found"
    )

print(
    f"Production rows   : {len(production_df):,}"
)

print(
    f"Production years  : {actual_years}"
)

print(
    f"Unique grids      : {production_df[GRID_COL].nunique():,}"
)

print(
    f"Frozen predictors : {len(predictors)}"
)


# =============================================================================
# 4. BUILD WITHIN-YEAR SPATIAL ANOMALY TARGET
# =============================================================================

section(
    "4. BUILD 2018-2025 WITHIN-YEAR SPATIAL ANOMALY TARGET"
)

production_df[
    "_annual_mean_lst_c"
] = production_df.groupby(
    YEAR_COL
)[
    TARGET
].transform(
    "mean"
)

production_df[
    "_spatial_anomaly_c"
] = (
    production_df[
        TARGET
    ]
    - production_df[
        "_annual_mean_lst_c"
    ]
)

annual_history = (
    production_df.groupby(
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
        rows=(
            TARGET,
            "size",
        ),
        anomaly_mean_c=(
            "_spatial_anomaly_c",
            "mean",
        ),
        anomaly_std_c=(
            "_spatial_anomaly_c",
            "std",
        ),
    )
)

max_abs_anomaly_year_mean = float(
    annual_history[
        "anomaly_mean_c"
    ]
    .abs()
    .max()
)

if max_abs_anomaly_year_mean > ZERO_TOL:
    raise RuntimeError(
        "Within-year anomaly target is not zero-mean by year"
    )

annual_history[
    "used_for_production_spatial_model"
] = True

annual_history[
    "future_annual_level_forecast_role"
] = (
    "historical input for later forecasting stage; "
    "no future level is chosen in Step 22B"
)

annual_history.to_csv(
    ANNUAL_HISTORY_PATH,
    index=False,
)

print(
    annual_history.to_string(
        index=False
    )
)

print(
    f"\nMax absolute anomaly mean by year: "
    f"{max_abs_anomaly_year_mean:.12f} °C"
)

print(
    "\nIMPORTANT: Step 22B does NOT choose a 2026/2030 annual thermal level."
)


# =============================================================================
# 5. VERIFY EXACT FROZEN MODEL CONFIGURATION
# =============================================================================

section(
    "5. VERIFY EXACT FROZEN LIGHTGBM CONFIGURATION"
)

step21_model = joblib.load(
    STEP21_FINAL_MODEL_PATH
)

source_params = step21_model.get_params(
    deep=False
)

parameter_checks = []

for name, expected_value in frozen_hyperparameters.items():
    if name not in source_params:
        raise RuntimeError(
            f"Frozen parameter '{name}' is missing from loaded Step-21 model"
        )

    actual_value = source_params[
        name
    ]

    matched = same_value(
        actual_value,
        expected_value,
    )

    parameter_checks.append(
        {
            "parameter": name,
            "manifest_value": expected_value,
            "loaded_step21_model_value": actual_value,
            "match": matched,
        }
    )

    if not matched:
        raise RuntimeError(
            f"Frozen parameter mismatch for {name}: "
            f"manifest={expected_value}, model={actual_value}"
        )

parameter_check_df = pd.DataFrame(
    parameter_checks
)

print(
    parameter_check_df.to_string(
        index=False
    )
)

print(
    "\nPASS: loaded Step-21 model matches the frozen hyperparameter manifest."
)


# =============================================================================
# 6. RETRAIN SAME FROZEN MODEL ON 2018-2025
# =============================================================================

section(
    "6. RETRAIN SAME FROZEN MODEL ON 2018-2025"
)

X_production = production_df[
    predictors
].copy()

y_production_anomaly = production_df[
    "_spatial_anomaly_c"
].to_numpy(
    dtype=float
)

# Clone preserves the estimator class and complete parameter configuration
# from the frozen Step-21 LightGBM model, but starts with an unfitted estimator.
production_model = clone(
    step21_model
)

production_model.fit(
    X_production,
    y_production_anomaly,
)

# Sanity check only; NOT a test-set performance evaluation.
train_pred_anomaly = np.asarray(
    production_model.predict(
        X_production
    ),
    dtype=float,
)

if len(
    train_pred_anomaly
) != EXPECTED_TOTAL_ROWS:
    raise RuntimeError(
        "Unexpected production-model prediction count"
    )

if not np.isfinite(
    train_pred_anomaly
).all():
    raise RuntimeError(
        "Non-finite production-model sanity predictions"
    )

joblib.dump(
    production_model,
    PRODUCTION_MODEL_PATH,
)

print(
    f"Saved production model : {PRODUCTION_MODEL_PATH}"
)

print(
    "Model target           : within-year spatial LST anomaly"
)

print(
    "Model fit years        : 2018-2025"
)

print(
    f"Model fit rows         : {EXPECTED_TOTAL_ROWS:,}"
)

print(
    "Hyperparameter tuning  : NONE"
)

print(
    "Feature changes        : NONE"
)

print(
    "\nNOTE: in-sample predictions above are only a technical sanity check."
)

print(
    "They are NOT reported as a new accuracy/test result."
)


# =============================================================================
# 7. SAVE YEAR-BY-YEAR PRODUCTION TRAINING AUDIT
# =============================================================================

section(
    "7. PRODUCTION TRAINING YEAR AUDIT"
)

training_year_audit = (
    production_df.groupby(
        YEAR_COL,
        as_index=False,
    )
    .agg(
        rows=(
            TARGET,
            "size",
        ),
        unique_grids=(
            GRID_COL,
            "nunique",
        ),
        observed_mean_lst_c=(
            TARGET,
            "mean",
        ),
        anomaly_target_mean_c=(
            "_spatial_anomaly_c",
            "mean",
        ),
        anomaly_target_std_c=(
            "_spatial_anomaly_c",
            "std",
        ),
    )
)

training_year_audit[
    "expected_rows"
] = EXPECTED_ROWS_PER_YEAR

training_year_audit[
    "row_count_pass"
] = (
    training_year_audit[
        "rows"
    ]
    == EXPECTED_ROWS_PER_YEAR
)

training_year_audit[
    "grid_count_pass"
] = (
    training_year_audit[
        "unique_grids"
    ]
    == EXPECTED_GRIDS
)

training_year_audit[
    "anomaly_zero_mean_pass"
] = (
    training_year_audit[
        "anomaly_target_mean_c"
    ]
    .abs()
    <= ZERO_TOL
)

training_year_audit.to_csv(
    TRAINING_AUDIT_TABLE_PATH,
    index=False,
)

print(
    training_year_audit.to_string(
        index=False
    )
)


# =============================================================================
# 8. SAVE PRODUCTION MODEL MANIFEST
# =============================================================================

section(
    "8. SAVE PRODUCTION MODEL MANIFEST"
)

production_model_hash = sha256_file(
    PRODUCTION_MODEL_PATH
)

manifest = {
    "step": "22B",
    "status": "PRODUCTION_SPATIAL_ANOMALY_MODEL_TRAINED",
    "purpose": (
        "Production retraining after completion of the final 2025 lockbox test"
    ),
    "selected_strategy": FINAL_STRATEGY,
    "base_model": FINAL_BASE_MODEL,
    "target": (
        "within-year spatial anomaly = LST_C - same-year annual mean LST_C"
    ),
    "feature_configuration": "frozen_full_93",
    "feature_count": len(
        predictors
    ),
    "feature_list_artifact": str(
        STEP21_FEATURE_LIST_PATH
    ),
    "source_frozen_model_artifact": str(
        STEP21_FINAL_MODEL_PATH
    ),
    "production_model_artifact": str(
        PRODUCTION_MODEL_PATH
    ),
    "production_model_sha256": production_model_hash,
    "training_years": EXPECTED_YEARS,
    "training_rows": EXPECTED_TOTAL_ROWS,
    "unique_grids": EXPECTED_GRIDS,
    "hyperparameters": frozen_hyperparameters,
    "model_configuration_source": (
        "exact clone of Step-21 frozen fitted LightGBM estimator configuration"
    ),
    "step22_final_test_preserved": True,
    "step22_test_result_recomputed": False,
    "2025_role_after_step22": (
        "historical observed training data for production retraining"
    ),
    "retuning_performed": False,
    "feature_reselection_performed": False,
    "strategy_change_performed": False,
    "future_annual_level_forecast": {
        "chosen_in_step22b": False,
        "reason": (
            "Future annual thermal level is a separate forecasting problem "
            "to be developed/validated in later forecasting stages."
        ),
        "historical_annual_level_file": str(
            ANNUAL_HISTORY_PATH
        ),
    },
    "future_inference_note": (
        "Future LST = forecast annual thermal level + centered predicted "
        "spatial anomaly. Future predictor values and annual level must be "
        "generated by the later forecasting pipeline."
    ),
}

PRODUCTION_MANIFEST_PATH.write_text(
    json.dumps(
        manifest,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        manifest,
        indent=2,
    )
)

print(
    f"\nSaved: {PRODUCTION_MANIFEST_PATH}"
)


# =============================================================================
# 9. STRICT STEP-22B AUDIT
# =============================================================================

section(
    "9. STRICT STEP-22B AUDIT"
)

all_year_rows_pass = bool(
    (
        year_counts
        == EXPECTED_ROWS_PER_YEAR
    ).all()
)

all_grid_years_pass = bool(
    (
        grid_year_counts
        == len(
            EXPECTED_YEARS
        )
    ).all()
)

all_training_audit_pass = bool(
    training_year_audit[
        [
            "row_count_pass",
            "grid_count_pass",
            "anomaly_zero_mean_pass",
        ]
    ].all().all()
)

saved_model_exists = PRODUCTION_MODEL_PATH.exists()

audit_rows = [
    {
        "check": "step22_final_lockbox_audit",
        "status": "PASS",
        "details": (
            "Step 22 audit verified fully PASS before production retraining."
        ),
    },
    {
        "check": "step22_final_evidence_hashes",
        "status": "PASS",
        "details": (
            "Step22 audit, metrics and report hashes match preserved final-test evidence."
        ),
    },
    {
        "check": "step21_strategy_unchanged",
        "status": "PASS",
        "details": FINAL_STRATEGY,
    },
    {
        "check": "step21_base_model_unchanged",
        "status": "PASS",
        "details": FINAL_BASE_MODEL,
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
            f"actual={len(predictors)}, expected={EXPECTED_FEATURES}"
        ),
    },
    {
        "check": "source_data_hashes",
        "status": "PASS",
        "details": (
            "2018-2023, 2024, and 2025 model-ready source hashes all verified."
        ),
    },
    {
        "check": "production_years",
        "status": (
            "PASS"
            if actual_years
            == EXPECTED_YEARS
            else "FAIL"
        ),
        "details": str(
            actual_years
        ),
    },
    {
        "check": "production_rows",
        "status": (
            "PASS"
            if len(
                production_df
            )
            == EXPECTED_TOTAL_ROWS
            else "FAIL"
        ),
        "details": (
            f"actual={len(production_df)}, expected={EXPECTED_TOTAL_ROWS}"
        ),
    },
    {
        "check": "unique_grids",
        "status": (
            "PASS"
            if production_df[
                GRID_COL
            ].nunique()
            == EXPECTED_GRIDS
            else "FAIL"
        ),
        "details": (
            f"actual={production_df[GRID_COL].nunique()}, "
            f"expected={EXPECTED_GRIDS}"
        ),
    },
    {
        "check": "rows_per_year",
        "status": (
            "PASS"
            if all_year_rows_pass
            else "FAIL"
        ),
        "details": (
            f"all years expected={EXPECTED_ROWS_PER_YEAR}"
        ),
    },
    {
        "check": "all_grids_present_all_8_years",
        "status": (
            "PASS"
            if all_grid_years_pass
            else "FAIL"
        ),
        "details": (
            "Each grid appears once in every year 2018-2025."
        ),
    },
    {
        "check": "within_year_anomaly_zero_mean",
        "status": (
            "PASS"
            if max_abs_anomaly_year_mean
            <= ZERO_TOL
            else "FAIL"
        ),
        "details": (
            f"max_abs_year_mean={max_abs_anomaly_year_mean:.12f}"
        ),
    },
    {
        "check": "frozen_hyperparameters_match_source_model",
        "status": (
            "PASS"
            if parameter_check_df[
                "match"
            ].all()
            else "FAIL"
        ),
        "details": (
            "All manifest hyperparameters match the frozen Step-21 model."
        ),
    },
    {
        "check": "hyperparameter_retuning",
        "status": "PASS",
        "details": (
            "None. Production estimator is cloned from the frozen Step-21 model."
        ),
    },
    {
        "check": "feature_reselection",
        "status": "PASS",
        "details": (
            "None. Exact Step-21 ordered 93-feature list reused."
        ),
    },
    {
        "check": "strategy_change",
        "status": "PASS",
        "details": (
            "None. Frozen LightGBM spatial-anomaly strategy retained."
        ),
    },
    {
        "check": "step22_test_score_recomputed_after_training",
        "status": "PASS",
        "details": (
            "No. 2025 is not re-evaluated as a test after entering production training."
        ),
    },
    {
        "check": "future_annual_level_chosen",
        "status": "PASS",
        "details": (
            "No. Future annual thermal-level forecasting is deferred to later forecasting steps."
        ),
    },
    {
        "check": "production_model_saved",
        "status": (
            "PASS"
            if saved_model_exists
            else "FAIL"
        ),
        "details": str(
            PRODUCTION_MODEL_PATH
        ),
    },
    {
        "check": "training_year_audit",
        "status": (
            "PASS"
            if all_training_audit_pass
            else "FAIL"
        ),
        "details": (
            "Row counts, grid counts and anomaly-zero-mean checks pass for all years."
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
        "One or more Step-22B audit checks failed"
    )


# =============================================================================
# 10. SAVE REPORT
# =============================================================================

section(
    "10. SAVE STEP-22B REPORT"
)

report_lines = [
    "STEP 22B - PRODUCTION RETRAIN ON 2018-2025",
    "=" * 116,
    "",
    "PURPOSE",
    "-" * 116,
    (
        "Retrain the exact final frozen LightGBM spatial-anomaly strategy "
        "on all observed modeling years after the one-time 2025 lockbox test."
    ),
    "",
    "PRODUCTION MODEL",
    "-" * 116,
    f"Strategy: {FINAL_STRATEGY}",
    f"Base model: {FINAL_BASE_MODEL}",
    "Training years: 2018-2025",
    f"Training rows: {EXPECTED_TOTAL_ROWS:,}",
    f"Unique grids: {EXPECTED_GRIDS:,}",
    f"Frozen predictors: {EXPECTED_FEATURES}",
    "Target: within-year spatial LST anomaly",
    "Hyperparameter retuning: NO",
    "Feature reselection: NO",
    "Strategy change: NO",
    "",
    "FINAL TEST PRESERVATION",
    "-" * 116,
    (
        "The Step-22 unseen 2025 lockbox result remains the final independent "
        "test evidence. It is not recomputed after production retraining."
    ),
    "",
    "FUTURE FORECASTING",
    "-" * 116,
    (
        "This production model predicts future spatial anomalies once future "
        "predictor values are available."
    ),
    (
        "A future annual thermal level is NOT selected in Step 22B. "
        "Annual-level forecasting and future predictor forecasting must be "
        "developed and validated in the later forecasting stages."
    ),
    "",
    "ANNUAL HISTORY",
    "-" * 116,
    annual_history.to_string(
        index=False
    ),
    "",
    f"Production model: {PRODUCTION_MODEL_PATH}",
    f"Production model SHA256: {production_model_hash}",
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
# 11. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "11. REPRODUCIBILITY HASHES"
)

hash_files = [
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH,
    STEP21_MANIFEST_PATH,
    STEP21_FEATURE_LIST_PATH,
    STEP21_FINAL_MODEL_PATH,
    STEP22_AUDIT_PATH,
    STEP22_METRICS_PATH,
    STEP22_REPORT_PATH,
    PRODUCTION_MODEL_PATH,
    ANNUAL_HISTORY_PATH,
    TRAINING_AUDIT_TABLE_PATH,
    PRODUCTION_MANIFEST_PATH,
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
# FINAL STATUS
# =============================================================================

section(
    "STEP 22B COMPLETED SUCCESSFULLY"
)

print(
    "Production retraining completed.\n"
    "The exact frozen LightGBM spatial-anomaly strategy was retained.\n"
    "Training now uses all observed modeling years 2018-2025.\n"
    f"Production rows: {EXPECTED_TOTAL_ROWS:,}\n"
    f"Unique grids: {EXPECTED_GRIDS:,}\n"
    f"Frozen predictors: {EXPECTED_FEATURES}\n"
    "No tuning, feature reselection, or strategy replacement occurred.\n"
    "The Step-22 2025 test result remains preserved as the final independent test.\n"
    "No future annual thermal level was chosen in Step 22B.\n"
    "\nNEXT: proceed to Step 23 historical Mann-Kendall + Sen's slope analysis."
)
