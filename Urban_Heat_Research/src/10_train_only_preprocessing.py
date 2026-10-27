"""
STEP 10 - TRAIN-ONLY PREPROCESSING (CORRECTED v2)
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Create model-ready files using preprocessing fitted ONLY on the Step 09
development/training split.

INPUT SPLITS
------------
Tree/nonlinear:
    data/splits/tree/tree_train_2018_2023.csv
    data/splits/tree/tree_validation_2024.csv
    data/splits/tree/tree_test_2025_LOCKED.csv

Linear baseline:
    data/splits/linear/linear_train_2018_2023.csv
    data/splits/linear/linear_validation_2024.csv
    data/splits/linear/linear_test_2025_LOCKED.csv

PREPROCESSING POLICY
--------------------
TREE / NONLINEAR MODELS
- Random Forest
- XGBoost
- LightGBM
- NGBoost with tree learners

No scaling is applied.
No imputation is applied because Step 09 verified zero predictor missingness
for 2018-2025.
The tree branch is therefore an audited identity preprocessing step.

LINEAR BASELINE
- 'year' is preserved unchanged because it is BOTH:
    (a) the temporal key used for split integrity, and
    (b) a legitimate time predictor selected in Step 08.
- StandardScaler is fitted ONLY on the OTHER 2018-2023 training predictors.
- The fitted scaler is then applied unchanged to:
    2024 validation predictors
    2025 lockbox predictors
- Target LST_C is NEVER scaled here.
- grid_id and area_name are NEVER used as numeric predictors.
- Integer predictor columns are explicitly converted to float before scaled
  values are assigned, avoiding pandas incompatible-dtype warnings.

ROLLING-ORIGIN RULE
-------------------
The scaler saved by this script is ONLY appropriate for the fixed
2018-2023 -> 2024 holdout workflow (and later final locked transformation).

It MUST NOT be reused inside earlier rolling-origin folds.
Example:
    train 2018-2020 -> validate 2021
must fit its own scaler on 2018-2020 inside that fold.

This script therefore also saves a preprocessing policy file that later model
scripts must obey.

LOCKBOX RULE
------------
2025 target values are copied unchanged into the locked model-ready files only
for eventual final evaluation. They are NOT summarized, scored, compared, or
used to fit any preprocessing transformation in Step 10.

This step performs NO:
- model fitting
- hyperparameter tuning
- target transformation
- 2024 model evaluation
- 2025 model evaluation
"""

from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. SETTINGS
# =============================================================================

TARGET = "LST_C"

TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
VALIDATION_YEAR = 2024
TEST_YEAR = 2025

NON_PREDICTOR_COLUMNS = [
    "grid_id",
    TARGET,
    "area_name",
]

# 'year' remains a model predictor, but it is also the temporal key.
# Therefore it is preserved unchanged in the model-ready files.
PRESERVE_UNSCALED_LINEAR_FEATURES = [
    "year",
]


# =============================================================================
# 2. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TREE_TRAIN_INPUT = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "tree"
    / "tree_train_2018_2023.csv"
)

TREE_VAL_INPUT = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "tree"
    / "tree_validation_2024.csv"
)

TREE_TEST_INPUT = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "tree"
    / "tree_test_2025_LOCKED.csv"
)

LINEAR_TRAIN_INPUT = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "linear"
    / "linear_train_2018_2023.csv"
)

LINEAR_VAL_INPUT = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "linear"
    / "linear_validation_2024.csv"
)

LINEAR_TEST_INPUT = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "linear"
    / "linear_test_2025_LOCKED.csv"
)

MODEL_READY_DIR = PROJECT_ROOT / "data" / "model_ready"
TREE_READY_DIR = MODEL_READY_DIR / "tree"
LINEAR_READY_DIR = MODEL_READY_DIR / "linear"

PREPROCESSOR_DIR = PROJECT_ROOT / "models" / "preprocessing"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

for directory in [
    MODEL_READY_DIR,
    TREE_READY_DIR,
    LINEAR_READY_DIR,
    PREPROCESSOR_DIR,
    REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

TREE_TRAIN_READY = TREE_READY_DIR / "tree_train_2018_2023_ready.csv"
TREE_VAL_READY = TREE_READY_DIR / "tree_validation_2024_ready.csv"
TREE_TEST_READY = TREE_READY_DIR / "tree_test_2025_LOCKED_ready.csv"

LINEAR_TRAIN_READY = LINEAR_READY_DIR / "linear_train_2018_2023_scaled.csv"
LINEAR_VAL_READY = LINEAR_READY_DIR / "linear_validation_2024_scaled.csv"
LINEAR_TEST_READY = LINEAR_READY_DIR / "linear_test_2025_LOCKED_scaled.csv"

LINEAR_SCALER_PATH = (
    PREPROCESSOR_DIR
    / "linear_standard_scaler_2018_2023.joblib"
)

LINEAR_SCALER_PARAMS_PATH = (
    REPORT_DIR
    / "10_linear_scaler_parameters.csv"
)

PREPROCESSING_POLICY_PATH = (
    REPORT_DIR
    / "10_preprocessing_policy.json"
)

AUDIT_PATH = (
    REPORT_DIR
    / "10_train_only_preprocessing_audit.csv"
)

HASH_PATH = (
    REPORT_DIR
    / "10_model_ready_file_hashes.csv"
)

MAIN_REPORT_PATH = (
    REPORT_DIR
    / "10_train_only_preprocessing_report.txt"
)


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 122)
    print(title)
    print("=" * 122)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        low_memory=False,
        dtype={
            "grid_id": "string",
            "area_name": "string",
        },
    )


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def validate_split(
    df: pd.DataFrame,
    name: str,
    expected_years: list[int],
) -> None:
    required = {
        "grid_id",
        "year",
        TARGET,
        "area_name",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{name}: missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            f"{name}: split is empty."
        )

    years = sorted(
        df["year"].astype(int).unique().tolist()
    )

    if years != expected_years:
        raise ValueError(
            f"{name}: expected years {expected_years}, got {years}"
        )

    duplicate_keys = int(
        df.duplicated(
            ["grid_id", "year"],
            keep=False,
        ).sum()
    )

    if duplicate_keys != 0:
        raise ValueError(
            f"{name}: duplicate grid-year rows = {duplicate_keys}"
        )


def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in NON_PREDICTOR_COLUMNS
    ]


def check_feature_schema(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    name: str,
) -> list[str]:
    train_features = get_feature_columns(train)
    val_features = get_feature_columns(validation)
    test_features = get_feature_columns(test)

    if train_features != val_features:
        raise ValueError(
            f"{name}: train and validation feature schemas differ."
        )

    if train_features != test_features:
        raise ValueError(
            f"{name}: train and test feature schemas differ."
        )

    return train_features


def check_predictor_numeric_finite(
    df: pd.DataFrame,
    feature_columns: list[str],
    name: str,
) -> None:
    non_numeric = [
        f
        for f in feature_columns
        if not pd.api.types.is_numeric_dtype(df[f])
    ]

    if non_numeric:
        raise ValueError(
            f"{name}: non-numeric predictors found: {non_numeric}"
        )

    missing_cells = int(
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    values = df[
        feature_columns
    ].to_numpy(dtype=float)

    infinite_cells = int(
        np.isinf(values).sum()
    )

    if missing_cells != 0:
        raise ValueError(
            f"{name}: predictor missing cells = {missing_cells}"
        )

    if infinite_cells != 0:
        raise ValueError(
            f"{name}: predictor infinite cells = {infinite_cells}"
        )


def validate_output_identity(
    before: pd.DataFrame,
    after: pd.DataFrame,
    name: str,
    predictors_should_match: bool,
) -> dict:
    same_rows = (
        len(before) == len(after)
    )

    same_keys = (
        before[
            ["grid_id", "year"]
        ]
        .reset_index(drop=True)
        .equals(
            after[
                ["grid_id", "year"]
            ]
            .reset_index(drop=True)
        )
    )

    same_target = np.allclose(
        before[TARGET],
        after[TARGET],
        equal_nan=True,
    )

    same_area = (
        before["area_name"]
        .reset_index(drop=True)
        .equals(
            after["area_name"]
            .reset_index(drop=True)
        )
    )

    duplicate_keys = int(
        after.duplicated(
            ["grid_id", "year"],
            keep=False,
        ).sum()
    )

    result = {
        "dataset": name,
        "same_rows": same_rows,
        "same_keys": same_keys,
        "same_target": same_target,
        "same_area": same_area,
        "duplicate_keys": duplicate_keys,
    }

    if predictors_should_match:
        feature_cols = get_feature_columns(before)

        same_predictors = np.allclose(
            before[feature_cols].to_numpy(dtype=float),
            after[feature_cols].to_numpy(dtype=float),
            equal_nan=True,
        )

        result["same_predictors"] = same_predictors

    return result


# =============================================================================
# 4. LOAD STEP 09 SPLITS
# =============================================================================

section("STEP 10 - TRAIN-ONLY PREPROCESSING")

input_paths = [
    TREE_TRAIN_INPUT,
    TREE_VAL_INPUT,
    TREE_TEST_INPUT,
    LINEAR_TRAIN_INPUT,
    LINEAR_VAL_INPUT,
    LINEAR_TEST_INPUT,
]

for path in input_paths:
    if not path.exists():
        print(f"\nERROR: required Step 09 split not found: {path}")
        sys.exit(1)

tree_train = load_csv(TREE_TRAIN_INPUT)
tree_val = load_csv(TREE_VAL_INPUT)
tree_test = load_csv(TREE_TEST_INPUT)

linear_train = load_csv(LINEAR_TRAIN_INPUT)
linear_val = load_csv(LINEAR_VAL_INPUT)
linear_test = load_csv(LINEAR_TEST_INPUT)

validate_split(
    tree_train,
    "tree_train",
    TRAIN_YEARS,
)

validate_split(
    tree_val,
    "tree_validation",
    [VALIDATION_YEAR],
)

validate_split(
    tree_test,
    "tree_test_LOCKED",
    [TEST_YEAR],
)

validate_split(
    linear_train,
    "linear_train",
    TRAIN_YEARS,
)

validate_split(
    linear_val,
    "linear_validation",
    [VALIDATION_YEAR],
)

validate_split(
    linear_test,
    "linear_test_LOCKED",
    [TEST_YEAR],
)

print(f"Tree train rows      : {len(tree_train):,}")
print(f"Tree validation rows : {len(tree_val):,}")
print(f"Tree test rows       : {len(tree_test):,}")
print(f"Linear train rows    : {len(linear_train):,}")
print(f"Linear validation rows: {len(linear_val):,}")
print(f"Linear test rows     : {len(linear_test):,}")


# =============================================================================
# 5. FEATURE SCHEMA AUDIT
# =============================================================================

section("1. FEATURE SCHEMA AUDIT")

tree_features = check_feature_schema(
    tree_train,
    tree_val,
    tree_test,
    "TREE",
)

linear_features = check_feature_schema(
    linear_train,
    linear_val,
    linear_test,
    "LINEAR",
)

print(
    f"Tree predictor count   : {len(tree_features)}"
)
print(
    f"Linear predictor count : {len(linear_features)}"
)

if "grid_id" in tree_features or "grid_id" in linear_features:
    print("\nERROR: grid_id leaked into predictor list.")
    sys.exit(1)

if "area_name" in tree_features or "area_name" in linear_features:
    print("\nERROR: area_name leaked into predictor list.")
    sys.exit(1)

if TARGET in tree_features or TARGET in linear_features:
    print("\nERROR: target leaked into predictor list.")
    sys.exit(1)

print(
    "PASS: identifiers/reporting fields/target are excluded from predictors."
)


# =============================================================================
# 6. NUMERIC / FINITE AUDIT
# =============================================================================

section("2. NUMERIC + FINITE PREDICTOR AUDIT")

for df_, features_, name_ in [
    (tree_train, tree_features, "tree_train"),
    (tree_val, tree_features, "tree_validation"),
    (tree_test, tree_features, "tree_test_LOCKED"),
    (linear_train, linear_features, "linear_train"),
    (linear_val, linear_features, "linear_validation"),
    (linear_test, linear_features, "linear_test_LOCKED"),
]:
    check_predictor_numeric_finite(
        df_,
        features_,
        name_,
    )
    print(f"{name_}: PASS")

print(
    "No imputation is required for the Step 10 fixed holdout files."
)


# =============================================================================
# 7. TREE / NONLINEAR IDENTITY PREPROCESSING
# =============================================================================

section("3. TREE / NONLINEAR PREPROCESSING")

print(
    "Policy: NO scaling and NO imputation for Random Forest / XGBoost / "
    "LightGBM / NGBoost tree learners."
)

tree_train_ready = tree_train.copy()
tree_val_ready = tree_val.copy()
tree_test_ready = tree_test.copy()

save_csv(
    tree_train_ready,
    TREE_TRAIN_READY,
)

save_csv(
    tree_val_ready,
    TREE_VAL_READY,
)

save_csv(
    tree_test_ready,
    TREE_TEST_READY,
)

print(f"Saved: {TREE_TRAIN_READY}")
print(f"Saved: {TREE_VAL_READY}")
print(f"Saved: {TREE_TEST_READY}")


# =============================================================================
# 8. LINEAR TRAIN-ONLY STANDARDIZATION
# =============================================================================

section("4. LINEAR TRAIN-ONLY STANDARDIZATION")

linear_scale_features = [
    f
    for f in linear_features
    if f not in PRESERVE_UNSCALED_LINEAR_FEATURES
]

missing_preserved = [
    f
    for f in PRESERVE_UNSCALED_LINEAR_FEATURES
    if f not in linear_features
]

if missing_preserved:
    print(
        f"\nERROR: expected preserved linear predictor(s) missing: "
        f"{missing_preserved}"
    )
    sys.exit(1)

X_linear_train = linear_train[
    linear_scale_features
].astype(float).copy()

X_linear_val = linear_val[
    linear_scale_features
].astype(float).copy()

X_linear_test = linear_test[
    linear_scale_features
].astype(float).copy()

scaler = StandardScaler(
    with_mean=True,
    with_std=True,
)

scaler.fit(
    X_linear_train
)

train_scaled = scaler.transform(
    X_linear_train
)

val_scaled = scaler.transform(
    X_linear_val
)

test_scaled = scaler.transform(
    X_linear_test
)

linear_train_ready = linear_train.copy()
linear_val_ready = linear_val.copy()
linear_test_ready = linear_test.copy()

# Cast only the columns receiving standardized float values.
for ready_df in [
    linear_train_ready,
    linear_val_ready,
    linear_test_ready,
]:
    ready_df[linear_scale_features] = (
        ready_df[linear_scale_features]
        .astype(float)
    )

linear_train_ready.loc[
    :,
    linear_scale_features,
] = train_scaled

linear_val_ready.loc[
    :,
    linear_scale_features,
] = val_scaled

linear_test_ready.loc[
    :,
    linear_scale_features,
] = test_scaled

# Explicit integrity check: year must remain the original temporal key.
for original_df, ready_df, label in [
    (linear_train, linear_train_ready, "train"),
    (linear_val, linear_val_ready, "validation"),
    (linear_test, linear_test_ready, "test_LOCKED"),
]:
    if not np.array_equal(
        original_df["year"].to_numpy(),
        ready_df["year"].to_numpy(),
    ):
        print(
            f"\nERROR: year changed during linear preprocessing for {label}."
        )
        sys.exit(1)

save_csv(
    linear_train_ready,
    LINEAR_TRAIN_READY,
)

save_csv(
    linear_val_ready,
    LINEAR_VAL_READY,
)

save_csv(
    linear_test_ready,
    LINEAR_TEST_READY,
)

joblib.dump(
    {
        "scaler": scaler,
        "scaled_feature_names": linear_scale_features,
        "preserved_unscaled_predictors": PRESERVE_UNSCALED_LINEAR_FEATURES,
        "all_linear_predictors": linear_features,
        "fit_years": TRAIN_YEARS,
        "target_scaled": False,
        "year_preserved_as_temporal_key": True,
        "grid_id_used_as_predictor": False,
        "area_name_used_as_predictor": False,
        "rolling_origin_reuse_allowed": False,
    },
    LINEAR_SCALER_PATH,
)

print(
    f"Scaler fitted on years: "
    f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]} ONLY"
)
print(f"Total linear predictors       : {len(linear_features)}")
print(f"Scaled linear predictors      : {len(linear_scale_features)}")
print(
    f"Preserved unscaled predictors : "
    f"{PRESERVE_UNSCALED_LINEAR_FEATURES}"
)
print(f"Saved: {LINEAR_TRAIN_READY}")
print(f"Saved: {LINEAR_VAL_READY}")
print(f"Saved: {LINEAR_TEST_READY}")
print(f"Saved scaler: {LINEAR_SCALER_PATH}")


# =============================================================================
# 9. SCALER PARAMETER REPORT
# =============================================================================

section("5. LINEAR SCALER PARAMETER AUDIT")

scaler_params = pd.DataFrame(
    {
        "feature": linear_scale_features,
        "train_mean": scaler.mean_,
        "train_scale": scaler.scale_,
        "train_variance": scaler.var_,
    }
)

zero_scale_features = scaler_params.loc[
    scaler_params["train_scale"] == 0,
    "feature",
].tolist()

if zero_scale_features:
    print(
        f"\nERROR: zero-scale linear features found: "
        f"{zero_scale_features}"
    )
    sys.exit(1)

scaler_params.to_csv(
    LINEAR_SCALER_PARAMS_PATH,
    index=False,
)

scaled_train_means = np.mean(
    train_scaled,
    axis=0,
)

scaled_train_stds = np.std(
    train_scaled,
    axis=0,
    ddof=0,
)

max_abs_scaled_mean = float(
    np.max(
        np.abs(
            scaled_train_means
        )
    )
)

max_abs_std_deviation = float(
    np.max(
        np.abs(
            scaled_train_stds - 1.0
        )
    )
)

print(
    f"Maximum absolute scaled training mean : "
    f"{max_abs_scaled_mean:.12f}"
)
print(
    f"Maximum |scaled std - 1|             : "
    f"{max_abs_std_deviation:.12f}"
)
print(
    f"Saved scaler parameter report        : "
    f"{LINEAR_SCALER_PARAMS_PATH}"
)

if max_abs_scaled_mean > 1e-8:
    print(
        "\nERROR: standardized training means are not sufficiently close to zero."
    )
    sys.exit(1)

if max_abs_std_deviation > 1e-8:
    print(
        "\nERROR: standardized training standard deviations are not sufficiently close to one."
    )
    sys.exit(1)


# =============================================================================
# 10. TRANSFORMED-FILE FINITE AUDIT
# =============================================================================

section("6. TRANSFORMED FILE FINITE AUDIT")

for df_, features_, name_ in [
    (
        linear_train_ready,
        linear_features,
        "linear_train_scaled",
    ),
    (
        linear_val_ready,
        linear_features,
        "linear_validation_scaled",
    ),
    (
        linear_test_ready,
        linear_features,
        "linear_test_LOCKED_scaled",
    ),
]:
    values = df_[features_].to_numpy(
        dtype=float
    )

    missing = int(
        np.isnan(values).sum()
    )

    infinite = int(
        np.isinf(values).sum()
    )

    print(
        f"{name_}: missing={missing}, infinite={infinite}"
    )

    if missing != 0 or infinite != 0:
        print(
            "\nERROR: transformed linear data contains non-finite values."
        )
        sys.exit(1)


# =============================================================================
# 11. TARGET / IDENTIFIER PRESERVATION
# =============================================================================

section("7. TARGET + IDENTIFIER PRESERVATION")

validation_rows = []

pairs = [
    (
        tree_train,
        tree_train_ready,
        "tree_train",
        True,
    ),
    (
        tree_val,
        tree_val_ready,
        "tree_validation",
        True,
    ),
    (
        tree_test,
        tree_test_ready,
        "tree_test_LOCKED",
        True,
    ),
    (
        linear_train,
        linear_train_ready,
        "linear_train_scaled",
        False,
    ),
    (
        linear_val,
        linear_val_ready,
        "linear_validation_scaled",
        False,
    ),
    (
        linear_test,
        linear_test_ready,
        "linear_test_LOCKED_scaled",
        False,
    ),
]

for before, after, name, predictors_should_match in pairs:
    result = validate_output_identity(
        before,
        after,
        name,
        predictors_should_match,
    )

    validation_rows.append(result)

validation_df = pd.DataFrame(
    validation_rows
)

print(
    validation_df.to_string(
        index=False
    )
)

required_boolean_cols = [
    "same_rows",
    "same_keys",
    "same_target",
    "same_area",
]

for col in required_boolean_cols:
    if not validation_df[col].all():
        print(
            f"\nERROR: preservation audit failed for {col}."
        )
        sys.exit(1)

if (
    validation_df["duplicate_keys"] != 0
).any():
    print(
        "\nERROR: duplicate keys found after preprocessing."
    )
    sys.exit(1)

tree_identity_rows = validation_df[
    validation_df["dataset"].str.startswith(
        "tree_"
    )
]

if not tree_identity_rows[
    "same_predictors"
].all():
    print(
        "\nERROR: tree identity preprocessing changed predictor values."
    )
    sys.exit(1)


# =============================================================================
# 12. SAVE PREPROCESSING POLICY
# =============================================================================

section("8. SAVE PREPROCESSING POLICY")

policy = {
    "step": 10,
    "name": "train_only_preprocessing",
    "temporal_protocol": {
        "training_years": TRAIN_YEARS,
        "validation_year": VALIDATION_YEAR,
        "test_year": TEST_YEAR,
    },
    "tree_nonlinear": {
        "models": [
            "Random Forest",
            "XGBoost",
            "LightGBM",
            "NGBoost tree learners",
        ],
        "scaling": "none",
        "imputation": "none because no missing predictors in 2018-2025",
        "feature_count": len(tree_features),
    },
    "linear_baseline": {
        "scaler": "StandardScaler",
        "fit_data": "2018-2023 non-year training predictors only",
        "total_feature_count": len(linear_features),
        "scaled_feature_count": len(linear_scale_features),
        "preserved_unscaled_predictors": PRESERVE_UNSCALED_LINEAR_FEATURES,
        "year_preserved_as_temporal_key": True,
        "target_scaled": False,
        "validation_transformation": "use frozen training scaler",
        "test_transformation": "use frozen training scaler",
    },
    "rolling_origin_rule": {
        "saved_global_scaler_must_not_be_used_inside_earlier_folds": True,
        "required_behavior": (
            "fit a fresh scaler/preprocessor inside each rolling-origin "
            "training fold and apply it only to that fold's validation year"
        ),
    },
    "lockbox_rule": {
        "2025_target_used_to_fit_preprocessing": False,
        "2025_model_performance_evaluated_in_step10": False,
        "2025_target_statistics_printed_in_step10": False,
    },
    "deep_learning_note": (
        "LSTM/TFT preprocessing must be handled separately with fold-aware "
        "sequence scaling later; do not automatically reuse this linear scaler."
    ),
}

PREPROCESSING_POLICY_PATH.write_text(
    json.dumps(
        policy,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    f"Saved: {PREPROCESSING_POLICY_PATH}"
)


# =============================================================================
# 13. REPRODUCIBILITY HASHES
# =============================================================================

section("9. REPRODUCIBILITY HASHES")

hash_paths = [
    TREE_TRAIN_INPUT,
    TREE_VAL_INPUT,
    TREE_TEST_INPUT,
    LINEAR_TRAIN_INPUT,
    LINEAR_VAL_INPUT,
    LINEAR_TEST_INPUT,
    TREE_TRAIN_READY,
    TREE_VAL_READY,
    TREE_TEST_READY,
    LINEAR_TRAIN_READY,
    LINEAR_VAL_READY,
    LINEAR_TEST_READY,
    LINEAR_SCALER_PATH,
]

hash_rows = []

for path in hash_paths:
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

hash_df = pd.DataFrame(
    hash_rows
)

hash_df.to_csv(
    HASH_PATH,
    index=False,
)


# =============================================================================
# 14. FINAL AUDIT
# =============================================================================

section("10. TRAIN-ONLY PREPROCESSING AUDIT")

audit_df = pd.DataFrame(
    [
        {
            "check": "tree_scaler_fitted",
            "status": "PASS",
            "details": "No scaler fitted for tree/nonlinear branch.",
        },
        {
            "check": "tree_imputation_fitted",
            "status": "PASS",
            "details": "No imputer fitted; no missing predictors in fixed modeling years.",
        },
        {
            "check": "linear_scaler_fit_scope",
            "status": "PASS",
            "details": (
                "StandardScaler fitted only on 2018-2023 non-year training "
                "predictors; raw year is preserved as the temporal key."
            ),
        },
        {
            "check": "linear_year_preserved",
            "status": "PASS",
            "details": (
                "Raw year remains unchanged in train, validation, and test files."
            ),
        },
        {
            "check": "validation_used_to_fit_scaler",
            "status": "PASS",
            "details": "No. 2024 transformed using frozen training scaler.",
        },
        {
            "check": "test_used_to_fit_scaler",
            "status": "PASS",
            "details": "No. 2025 transformed using frozen training scaler.",
        },
        {
            "check": "target_scaled_or_transformed",
            "status": "PASS",
            "details": "No. LST_C preserved unchanged.",
        },
        {
            "check": "identifiers_used_as_predictors",
            "status": "PASS",
            "details": "grid_id and area_name excluded.",
        },
        {
            "check": "rolling_origin_leakage_policy",
            "status": "PASS",
            "details": (
                "Saved 2018-2023 scaler explicitly prohibited inside earlier "
                "rolling-origin folds; fold-local fitting required."
            ),
        },
        {
            "check": "2024_model_evaluation_in_step10",
            "status": "PASS",
            "details": "No model evaluation performed.",
        },
        {
            "check": "2025_model_evaluation_in_step10",
            "status": "PASS",
            "details": "No model evaluation performed; lockbox preserved.",
        },
        {
            "check": "2025_target_statistics_inspected",
            "status": "PASS",
            "details": "No 2025 target statistics printed or used.",
        },
    ]
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


# =============================================================================
# 15. MAIN REPORT
# =============================================================================

section("11. SAVE STEP 10 REPORT")

report_lines = [
    "STEP 10 - TRAIN-ONLY PREPROCESSING REPORT",
    "=" * 104,
    "",
    "TREE / NONLINEAR BRANCH",
    "-" * 104,
    f"Predictor count: {len(tree_features)}",
    "Scaling: none",
    "Imputation: none",
    "Reason: tree models do not require scaling and Step 09 confirmed zero",
    "predictor missingness in 2018-2025.",
    "",
    "LINEAR BASELINE BRANCH",
    "-" * 104,
    f"Total predictor count: {len(linear_features)}",
    f"Scaled predictor count: {len(linear_scale_features)}",
    f"Preserved unscaled predictor(s): {PRESERVE_UNSCALED_LINEAR_FEATURES}",
    "Transformation: StandardScaler on non-year predictors",
    "Scaler fit years: 2018-2023 ONLY",
    "Raw year: preserved unchanged as temporal key and model predictor",
    "2024: non-year predictors transformed with frozen training scaler",
    "2025: non-year predictors transformed with frozen training scaler",
    "Target LST_C: not scaled",
    "",
    "SCALER VALIDATION",
    "-" * 104,
    f"Maximum absolute scaled training mean: {max_abs_scaled_mean:.12f}",
    f"Maximum |scaled std - 1|: {max_abs_std_deviation:.12f}",
    "",
    "ROLLING-ORIGIN LEAKAGE RULE",
    "-" * 104,
    "The saved 2018-2023 scaler MUST NOT be reused inside earlier rolling folds.",
    "Each rolling fold must fit preprocessing only on that fold's historical",
    "training years before transforming its validation year.",
    "",
    "LOCKBOX",
    "-" * 104,
    "2025 target was not used to fit preprocessing.",
    "2025 target performance was not evaluated.",
    "2025 target statistics were not printed.",
    "",
    "MODEL-READY OUTPUTS",
    "-" * 104,
    f"Tree train: {TREE_TRAIN_READY}",
    f"Tree validation: {TREE_VAL_READY}",
    f"Tree test LOCKED: {TREE_TEST_READY}",
    f"Linear train scaled: {LINEAR_TRAIN_READY}",
    f"Linear validation scaled: {LINEAR_VAL_READY}",
    f"Linear test LOCKED scaled: {LINEAR_TEST_READY}",
    "",
    "NEXT",
    "-" * 104,
    "Proceed to Step 11 baseline model development only after reviewing",
    "the complete Step 10 audit and model-ready files.",
]

MAIN_REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {MAIN_REPORT_PATH}")
print(f"Saved: {AUDIT_PATH}")
print(f"Saved: {HASH_PATH}")


# =============================================================================
# 16. FINAL STATUS
# =============================================================================

section("STEP 10 COMPLETED SUCCESSFULLY")

print(
    "Train-only preprocessing completed.\n"
    "Tree/nonlinear files were preserved without unnecessary scaling.\n"
    "Linear StandardScaler was fitted only on 2018-2023 non-year training predictors.\n"
    "2024 was transformed only; no evaluation was performed.\n"
    "2025 remained a locked final-test dataset; no target performance or "
    "target statistics were inspected.\n"
    "Rolling-origin models must fit preprocessing independently inside each fold.\n"
    "Review all Step 10 outputs before Step 11 baselines."
)
