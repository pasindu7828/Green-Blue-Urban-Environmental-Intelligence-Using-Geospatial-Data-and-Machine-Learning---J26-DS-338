"""
STEP 01 - PREPARE ADDITIONAL-MODEL EXPERIMENT DATA
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Prepare a leakage-safe, reproducible development dataset for a separate
supplementary experiment comparing additional model families.

IMPORTANT
---------
- Existing production src/models/outputs are NOT modified.
- Only the existing 2018-2023 tree/nonlinear development dataset is loaded.
- 2024 reserved validation is NOT loaded.
- 2025 final lockbox is NOT loaded.
- No model is trained in this step.
- No hyperparameter tuning is performed in this step.

Experiment design
-----------------
Development years:
    2018-2023

Strict expanding rolling-origin folds:
    train 2018      -> validate 2019
    train 2018-2019 -> validate 2020
    train 2018-2020 -> validate 2021
    train 2018-2021 -> validate 2022
    train 2018-2022 -> validate 2023

Later experiment scripts must use the same 93 predictors and the same
spatial-anomaly + prior-mean annual-level framework used by the finalized
LightGBM development strategy.

The validation-year target must NEVER be used to construct its prediction.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# 1. SETTINGS
# =============================================================================

TARGET = "LST_C"
GRID_COL = "grid_id"
AREA_COL = "area_name"
YEAR_COL = "year"

DEVELOPMENT_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
ROLLING_VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]

EXPECTED_ROWS = 13_164
EXPECTED_GRIDS = 2_194
EXPECTED_PREDICTORS = 93

NON_PREDICTOR_COLUMNS = [
    GRID_COL,
    TARGET,
    AREA_COL,
]


# =============================================================================
# 2. PATHS
# =============================================================================

# This file lives at:
# Urban_Heat_Research/experiments/additional_models/src/01_prepare_experiment_data.py
#
# parents[3] -> Urban_Heat_Research
# parents[1] -> experiments/additional_models
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "tree"
    / "tree_train_2018_2023_ready.csv"
)

EXPERIMENT_DATA_DIR = EXPERIMENT_ROOT / "data"
OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
METRIC_DIR = OUTPUT_DIR / "metrics"

for directory in [
    EXPERIMENT_DATA_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    METRIC_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

EXPERIMENT_DATA_PATH = (
    EXPERIMENT_DATA_DIR
    / "01_experiment_dev_2018_2023.csv"
)

FOLD_PLAN_PATH = (
    METRIC_DIR
    / "01_rolling_origin_fold_plan.csv"
)

POLICY_PATH = (
    REPORT_DIR
    / "01_experiment_policy.json"
)

AUDIT_PATH = (
    REPORT_DIR
    / "01_prepare_experiment_data_audit.csv"
)

HASH_PATH = (
    REPORT_DIR
    / "01_experiment_data_hashes.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "01_prepare_experiment_data_report.txt"
)


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        low_memory=False,
        dtype={
            GRID_COL: "string",
            AREA_COL: "string",
        },
    )


def fail(message: str) -> None:
    raise RuntimeError(message)


# =============================================================================
# 4. START
# =============================================================================

section("STEP 01 - PREPARE ADDITIONAL-MODEL EXPERIMENT DATA")

print(f"Project root       : {PROJECT_ROOT}")
print(f"Experiment root    : {EXPERIMENT_ROOT}")
print(f"Input              : {INPUT_PATH}")
print("Production src     : UNTOUCHED")
print("Production models  : UNTOUCHED")
print("Production outputs : UNTOUCHED")
print("2024 data          : NOT LOADED")
print("2025 data          : NOT LOADED")


# =============================================================================
# 5. LOAD DEVELOPMENT DATA ONLY
# =============================================================================

section("1. LOAD 2018-2023 DEVELOPMENT DATA")

if not INPUT_PATH.exists():
    fail(
        "Required development input does not exist:\n"
        f"{INPUT_PATH}\n"
        "Run/restore the finalized production preprocessing pipeline first."
    )

df = load_csv(INPUT_PATH)

required_columns = {
    GRID_COL,
    AREA_COL,
    YEAR_COL,
    TARGET,
}

missing_required = sorted(required_columns - set(df.columns))
if missing_required:
    fail("Missing required columns: " + ", ".join(missing_required))

df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="raise").astype(int)

actual_years = sorted(df[YEAR_COL].unique().tolist())
if actual_years != DEVELOPMENT_YEARS:
    fail(
        "Development years do not match the finalized design.\n"
        f"Expected: {DEVELOPMENT_YEARS}\n"
        f"Actual  : {actual_years}"
    )

if df[YEAR_COL].isin([2024, 2025]).any():
    fail("2024 or 2025 rows were unexpectedly loaded.")

predictors = [
    c
    for c in df.columns
    if c not in NON_PREDICTOR_COLUMNS
]

if len(predictors) != EXPECTED_PREDICTORS:
    fail(
        f"Expected {EXPECTED_PREDICTORS} predictors, "
        f"but found {len(predictors)}."
    )

non_numeric_predictors = [
    c
    for c in predictors
    if not pd.api.types.is_numeric_dtype(df[c])
]

if non_numeric_predictors:
    fail(
        "Non-numeric predictors found: "
        + ", ".join(non_numeric_predictors)
    )

predictor_array = df[predictors].to_numpy(dtype=float)
missing_predictor_cells = int(df[predictors].isna().sum().sum())
finite_predictors = bool(np.isfinite(predictor_array).all())

if missing_predictor_cells != 0:
    fail(f"Predictors contain {missing_predictor_cells} missing cells.")

if not finite_predictors:
    fail("Predictors contain non-finite values.")

if df[TARGET].isna().any():
    fail("Development target contains missing values.")

if not np.isfinite(
    pd.to_numeric(df[TARGET], errors="raise").to_numpy(dtype=float)
).all():
    fail("Development target contains non-finite values.")

unique_grids = int(df[GRID_COL].nunique())

if len(df) != EXPECTED_ROWS:
    fail(
        f"Expected {EXPECTED_ROWS:,} rows, "
        f"but found {len(df):,}."
    )

if unique_grids != EXPECTED_GRIDS:
    fail(
        f"Expected {EXPECTED_GRIDS:,} unique grids, "
        f"but found {unique_grids:,}."
    )

duplicate_grid_year_rows = int(
    df.duplicated(
        subset=[GRID_COL, YEAR_COL],
        keep=False,
    ).sum()
)

if duplicate_grid_year_rows != 0:
    fail(
        "Duplicate grid-year rows found: "
        f"{duplicate_grid_year_rows}"
    )

year_counts = (
    df.groupby(YEAR_COL)
    .size()
    .rename("rows")
    .reset_index()
)

if not (year_counts["rows"] == EXPECTED_GRIDS).all():
    fail(
        "Each development year must contain exactly "
        f"{EXPECTED_GRIDS:,} grid rows."
    )

print(f"Rows               : {len(df):,}")
print(f"Columns            : {len(df.columns):,}")
print(f"Unique grids       : {unique_grids:,}")
print(f"Years              : {actual_years}")
print(f"Predictors         : {len(predictors)}")
print(f"Missing predictors : {missing_predictor_cells}")
print(f"Finite predictors  : {finite_predictors}")

print("\nRows by year:")
print(year_counts.to_string(index=False))


# =============================================================================
# 6. SAVE FROZEN EXPERIMENT SNAPSHOT
# =============================================================================

section("2. SAVE FROZEN EXPERIMENT SNAPSHOT")

experiment_df = (
    df.sort_values(
        [YEAR_COL, GRID_COL],
        kind="mergesort",
    )
    .reset_index(drop=True)
)

experiment_df.to_csv(
    EXPERIMENT_DATA_PATH,
    index=False,
    encoding="utf-8",
)

print(f"Saved: {EXPERIMENT_DATA_PATH}")


# =============================================================================
# 7. BUILD STRICT ROLLING-ORIGIN FOLD PLAN
# =============================================================================

section("3. STRICT ROLLING-ORIGIN FOLD PLAN")

fold_rows = []

for validation_year in ROLLING_VALIDATION_YEARS:
    train_years = [
        y
        for y in DEVELOPMENT_YEARS
        if y < validation_year
    ]

    train_df = experiment_df[
        experiment_df[YEAR_COL].isin(train_years)
    ]

    val_df = experiment_df[
        experiment_df[YEAR_COL] == validation_year
    ]

    fold_rows.append(
        {
            "validation_year": validation_year,
            "train_start_year": min(train_years),
            "train_end_year": max(train_years),
            "train_year_count": len(train_years),
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "train_unique_grids": int(train_df[GRID_COL].nunique()),
            "validation_unique_grids": int(val_df[GRID_COL].nunique()),
            "random_kfold_used": False,
            "uses_2024": False,
            "uses_2025": False,
        }
    )

fold_plan = pd.DataFrame(fold_rows)

fold_plan.to_csv(
    FOLD_PLAN_PATH,
    index=False,
    encoding="utf-8",
)

print(
    fold_plan[
        [
            "train_start_year",
            "train_end_year",
            "validation_year",
            "train_rows",
            "validation_rows",
        ]
    ].to_string(index=False)
)

print(f"\nSaved: {FOLD_PLAN_PATH}")


# =============================================================================
# 8. EXPERIMENT POLICY
# =============================================================================

section("4. SAVE EXPERIMENT POLICY")

policy = {
    "experiment_name": "additional_model_comparison",
    "purpose": (
        "Supplementary comparison of additional model families without "
        "modifying the finalized production pipeline."
    ),
    "production_pipeline_modified": False,
    "production_model_overwritten": False,
    "target": TARGET,
    "development_years": DEVELOPMENT_YEARS,
    "rolling_validation_years": ROLLING_VALIDATION_YEARS,
    "predictor_count": len(predictors),
    "predictors": predictors,
    "non_predictors": NON_PREDICTOR_COLUMNS,
    "random_kfold_allowed": False,
    "reserved_validation_2024_loaded_in_development_experiment": False,
    "final_lockbox_2025_loaded_in_development_experiment": False,
    "primary_model_selection_metric": "mean rolling-origin MAE",
    "secondary_metrics": [
        "RMSE",
        "R2",
        "MAPE",
        "centered_spatial_R2",
        "worst_year_MAE",
        "negative_R2_fold_count",
    ],
    "required_prediction_framework": {
        "target_structure": (
            "spatial anomaly model plus causal annual-level forecast"
        ),
        "annual_level_strategy": "prior_mean_level",
        "training_anomaly_definition": (
            "training-year LST minus that training year's mean LST"
        ),
        "validation_prediction_rule": (
            "predict validation spatial anomaly using training years only; "
            "center predicted validation anomalies using prediction mean only; "
            "add annual level forecast computed only from prior training-year "
            "annual target means"
        ),
        "current_validation_year_target_allowed_for_prediction": False,
        "current_validation_year_target_allowed_for_metrics_only": True,
    },
    "additional_models_planned": [
        "CatBoostRegressor",
        "ExtraTreesRegressor",
        "HistGradientBoostingRegressor",
    ],
    "production_reference": {
        "model_family": "LightGBM",
        "development_strategy": "lgbm_anomaly_plus_prior_mean_level",
    },
    "decision_rule": (
        "Do not replace the frozen production model based on this development "
        "experiment alone. First compare on identical rolling-origin folds. "
        "Only if an additional model is clearly stronger should reserved 2024 "
        "validation be opened for one-time confirmation. 2025 remains "
        "untouched for model selection/tuning."
    ),
}

POLICY_PATH.write_text(
    json.dumps(
        policy,
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Saved: {POLICY_PATH}")


# =============================================================================
# 9. AUDIT
# =============================================================================

section("5. STRICT STEP 01 AUDIT")

audit_rows = [
    {
        "check": "development_years",
        "status": "PASS" if actual_years == DEVELOPMENT_YEARS else "FAIL",
        "details": str(actual_years),
    },
    {
        "check": "row_count",
        "status": "PASS" if len(experiment_df) == EXPECTED_ROWS else "FAIL",
        "details": f"actual={len(experiment_df)}, expected={EXPECTED_ROWS}",
    },
    {
        "check": "unique_grid_count",
        "status": "PASS" if unique_grids == EXPECTED_GRIDS else "FAIL",
        "details": f"actual={unique_grids}, expected={EXPECTED_GRIDS}",
    },
    {
        "check": "predictor_count",
        "status": "PASS" if len(predictors) == EXPECTED_PREDICTORS else "FAIL",
        "details": f"actual={len(predictors)}, expected={EXPECTED_PREDICTORS}",
    },
    {
        "check": "missing_predictors",
        "status": "PASS" if missing_predictor_cells == 0 else "FAIL",
        "details": str(missing_predictor_cells),
    },
    {
        "check": "finite_predictors",
        "status": "PASS" if finite_predictors else "FAIL",
        "details": str(finite_predictors),
    },
    {
        "check": "duplicate_grid_year_rows",
        "status": "PASS" if duplicate_grid_year_rows == 0 else "FAIL",
        "details": str(duplicate_grid_year_rows),
    },
    {
        "check": "2024_loaded_or_used",
        "status": "PASS",
        "details": "No. Only the 2018-2023 development input is loaded.",
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": "No. Only the 2018-2023 development input is loaded.",
    },
    {
        "check": "strict_rolling_origin_fold_count",
        "status": "PASS" if len(fold_plan) == 5 else "FAIL",
        "details": f"actual={len(fold_plan)}, expected=5",
    },
    {
        "check": "random_kfold_used",
        "status": "PASS",
        "details": "No. Expanding rolling-origin folds only.",
    },
    {
        "check": "production_pipeline_modified",
        "status": "PASS",
        "details": "No production src/model/output file is written by Step 01.",
    },
]

audit_df = pd.DataFrame(audit_rows)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
    encoding="utf-8",
)

print(audit_df.to_string(index=False))

if not (audit_df["status"] == "PASS").all():
    fail("Step 01 audit contains one or more FAIL rows.")


# =============================================================================
# 10. HASHES
# =============================================================================

section("6. REPRODUCIBILITY HASHES")

hash_df = pd.DataFrame(
    [
        {
            "file": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(INPUT_PATH),
            "role": "production development input",
        },
        {
            "file": str(EXPERIMENT_DATA_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(EXPERIMENT_DATA_PATH),
            "role": "frozen experiment snapshot",
        },
        {
            "file": str(FOLD_PLAN_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(FOLD_PLAN_PATH),
            "role": "rolling-origin fold plan",
        },
        {
            "file": str(POLICY_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(POLICY_PATH),
            "role": "experiment policy",
        },
    ]
)

hash_df.to_csv(
    HASH_PATH,
    index=False,
    encoding="utf-8",
)

print(hash_df.to_string(index=False))
print(f"\nSaved: {HASH_PATH}")


# =============================================================================
# 11. REPORT
# =============================================================================

section("7. SAVE STEP 01 REPORT")

report_lines = [
    "STEP 01 - PREPARE ADDITIONAL-MODEL EXPERIMENT DATA",
    "",
    f"Project root: {PROJECT_ROOT}",
    f"Experiment root: {EXPERIMENT_ROOT}",
    f"Input: {INPUT_PATH}",
    f"Experiment snapshot: {EXPERIMENT_DATA_PATH}",
    "",
    f"Rows: {len(experiment_df):,}",
    f"Unique grids: {unique_grids:,}",
    f"Years: {actual_years}",
    f"Predictors: {len(predictors)}",
    f"Missing predictor cells: {missing_predictor_cells}",
    f"Finite predictors: {finite_predictors}",
    "",
    "Strict rolling-origin validation years: " + str(ROLLING_VALIDATION_YEARS),
    "2024 reserved validation: NOT LOADED",
    "2025 final lockbox: NOT LOADED",
    "Production pipeline: UNTOUCHED",
    "",
    "Planned additional models:",
    "- CatBoostRegressor",
    "- ExtraTreesRegressor",
    "- HistGradientBoostingRegressor",
    "",
    "Primary development comparison metric: mean rolling-origin MAE",
    "Production reference: LightGBM anomaly + prior-mean annual level",
    "",
    "STEP 01 STATUS: PASS",
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {REPORT_PATH}")


section("STEP 01 COMPLETED SUCCESSFULLY")

print("Development experiment snapshot prepared.")
print("Strict expanding rolling-origin folds are frozen.")
print("2024 reserved validation remains untouched.")
print("2025 final lockbox remains untouched.")
print("No production file was modified.")
print("Next: Step 02 - CatBoost supplementary experiment.")
