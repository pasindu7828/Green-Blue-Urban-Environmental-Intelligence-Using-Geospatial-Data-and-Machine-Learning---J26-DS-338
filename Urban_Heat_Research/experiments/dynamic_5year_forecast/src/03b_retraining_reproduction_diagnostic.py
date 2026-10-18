"""
DYNAMIC 5-YEAR FORECAST - STEP 03B
==================================
Diagnostic only. Explains why a fresh dynamic LightGBM refit may differ slightly
from the stored Step-22B production model even when the same data, features and
strategy are used.

This script DOES NOT modify or replace any production model.
"""

from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]

DYNAMIC_ENGINEERED_PATH = (
    EXPERIMENT_ROOT / "outputs" / "03_dynamic_engineered_history.csv"
)
DYNAMIC_MODEL_PATH = (
    EXPERIMENT_ROOT / "models" / "03_dynamic_lightgbm_anomaly_trained_2018_2025.joblib"
)

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "model_ready" / "tree" / "tree_train_2018_2023_ready.csv"
)
VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "model_ready" / "tree" / "tree_validation_2024_ready.csv"
)
TEST_PATH = (
    PROJECT_ROOT / "data" / "model_ready" / "tree" / "tree_test_2025_LOCKED_ready.csv"
)
FEATURE_LIST_PATH = (
    PROJECT_ROOT / "outputs" / "reports" / "21_final_feature_list.csv"
)
PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "final" / "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
)

REPORT_DIR = EXPERIMENT_ROOT / "outputs" / "reports"
AUDIT_PATH = REPORT_DIR / "03b_retraining_reproduction_audit.csv"
FEATURE_DIFF_PATH = REPORT_DIR / "03b_feature_reproduction_differences.csv"
PARAM_DIFF_PATH = REPORT_DIR / "03b_model_parameter_comparison.csv"
SUMMARY_PATH = REPORT_DIR / "03b_retraining_reproduction_summary.json"

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET = "LST_C"
EXPECTED_GRIDS = 2194
EXPECTED_YEARS = list(range(2018, 2026))
EXPECTED_ROWS = EXPECTED_GRIDS * len(EXPECTED_YEARS)
FEATURE_TOL = 1e-8
TARGET_TOL = 1e-10


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 125)
    print(title)
    print("=" * 125)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")


def same_value(a, b):
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        try:
            return bool(np.isclose(float(a), float(b), rtol=0.0, atol=0.0, equal_nan=True))
        except Exception:
            pass
    return a == b


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def add(audit_rows, check, passed, details):
    status = "PASS" if passed else "FAIL"
    audit_rows.append({"check": check, "status": status, "details": details})
    print(f"{status:4s} | {check}: {details}")


# =============================================================================
# START
# =============================================================================

REPORT_DIR.mkdir(parents=True, exist_ok=True)

section("DYNAMIC 5-YEAR FORECAST - STEP 03B RETRAINING REPRODUCTION DIAGNOSTIC")

for path in [
    DYNAMIC_ENGINEERED_PATH,
    DYNAMIC_MODEL_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH,
    FEATURE_LIST_PATH,
    PRODUCTION_MODEL_PATH,
]:
    require_file(path)

# Frozen feature order
feature_df = pd.read_csv(FEATURE_LIST_PATH).sort_values("feature_order")
predictors = feature_df["feature"].astype(str).tolist()

if len(predictors) != 93:
    raise RuntimeError(f"Expected 93 frozen predictors, found {len(predictors)}")

print(f"Frozen predictors : {len(predictors)}")


# =============================================================================
# 1. LOAD ORIGINAL STEP-22B TRAINING DATA AND DYNAMIC DATA
# =============================================================================

section("1. LOAD + ALIGN ORIGINAL AND DYNAMIC TRAINING DATA")

original = pd.concat(
    [
        pd.read_csv(TRAIN_PATH, dtype={GRID_COL: str}),
        pd.read_csv(VALIDATION_PATH, dtype={GRID_COL: str}),
        pd.read_csv(TEST_PATH, dtype={GRID_COL: str}),
    ],
    ignore_index=True,
)

original[YEAR_COL] = pd.to_numeric(original[YEAR_COL], errors="raise").astype(int)
original[TARGET] = pd.to_numeric(original[TARGET], errors="raise")

for feature in predictors:
    original[feature] = pd.to_numeric(original[feature], errors="raise")

# This is the exact Step-22B target definition.
original["_annual_mean_lst_c"] = original.groupby(YEAR_COL)[TARGET].transform("mean")
original["_spatial_anomaly_c"] = original[TARGET] - original["_annual_mean_lst_c"]


dynamic = pd.read_csv(DYNAMIC_ENGINEERED_PATH, dtype={GRID_COL: str})
dynamic[YEAR_COL] = pd.to_numeric(dynamic[YEAR_COL], errors="raise").astype(int)
dynamic[TARGET] = pd.to_numeric(dynamic[TARGET], errors="raise")
dynamic = dynamic[dynamic[YEAR_COL].isin(EXPECTED_YEARS)].copy()

for feature in predictors:
    if feature not in dynamic.columns:
        raise RuntimeError(f"Dynamic engineered history is missing frozen feature: {feature}")
    dynamic[feature] = pd.to_numeric(dynamic[feature], errors="raise")

dynamic["_annual_mean_lst_c"] = dynamic.groupby(YEAR_COL)[TARGET].transform("mean")
dynamic["_spatial_anomaly_c"] = dynamic[TARGET] - dynamic["_annual_mean_lst_c"]

print(f"Original rows : {len(original):,}")
print(f"Dynamic rows  : {len(dynamic):,}")

if len(original) != EXPECTED_ROWS or len(dynamic) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} rows in both datasets; "
        f"original={len(original):,}, dynamic={len(dynamic):,}"
    )

original_keys = set(zip(original[GRID_COL], original[YEAR_COL]))
dynamic_keys = set(zip(dynamic[GRID_COL], dynamic[YEAR_COL]))

key_sets_equal = original_keys == dynamic_keys
print(f"Key set equal : {key_sets_equal}")

# Compare source row order before key sorting.
original_key_sequence = list(zip(original[GRID_COL], original[YEAR_COL]))
dynamic_key_sequence = list(zip(dynamic[GRID_COL], dynamic[YEAR_COL]))
row_order_equal = original_key_sequence == dynamic_key_sequence
print(f"Raw row order equal : {row_order_equal}")


# =============================================================================
# 2. KEY-ALIGNED FEATURE + TARGET REPRODUCTION
# =============================================================================

section("2. COMPARE ALL 93 FEATURES + TARGET BY GRID/YEAR")

# IMPORTANT:
# `year` is itself one of the frozen 93 predictors, but it is also part of the
# alignment key.  Therefore we do NOT duplicate it inside a merge payload.
# Instead, sort both datasets by the same unique key and compare row-for-row.
original_aligned = (
    original
    .sort_values([YEAR_COL, GRID_COL], kind="stable")
    .reset_index(drop=True)
)

dynamic_aligned = (
    dynamic
    .sort_values([YEAR_COL, GRID_COL], kind="stable")
    .reset_index(drop=True)
)

original_aligned_keys = list(zip(original_aligned[GRID_COL], original_aligned[YEAR_COL]))
dynamic_aligned_keys = list(zip(dynamic_aligned[GRID_COL], dynamic_aligned[YEAR_COL]))

if original_aligned_keys != dynamic_aligned_keys:
    raise RuntimeError("Key-aligned row sequences differ after sorting by year/grid_id.")

if len(original_aligned) != EXPECTED_ROWS or len(dynamic_aligned) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Aligned rows are incomplete; original={len(original_aligned):,}, "
        f"dynamic={len(dynamic_aligned):,}, expected={EXPECTED_ROWS:,}"
    )

feature_diff_rows = []

for feature in predictors:
    a = original_aligned[feature].to_numpy(dtype=float)
    b = dynamic_aligned[feature].to_numpy(dtype=float)
    diff = np.abs(a - b)
    feature_diff_rows.append(
        {
            "feature": feature,
            "max_abs_error": float(np.max(diff)),
            "mean_abs_error": float(np.mean(diff)),
            "rows_over_1e-8": int(np.count_nonzero(diff > FEATURE_TOL)),
            "status": "PASS" if float(np.max(diff)) <= FEATURE_TOL else "FAIL",
        }
    )

feature_diff_df = pd.DataFrame(feature_diff_rows).sort_values(
    "max_abs_error", ascending=False
)
feature_diff_df.to_csv(FEATURE_DIFF_PATH, index=False)

max_feature_error = float(feature_diff_df["max_abs_error"].max())
failed_features = int((feature_diff_df["status"] == "FAIL").sum())

print(feature_diff_df.head(15).to_string(index=False))
print(f"\nMax feature error : {max_feature_error:.12e}")
print(f"Failed features   : {failed_features}")

orig_target = original_aligned[TARGET].to_numpy(dtype=float)
dyn_target = dynamic_aligned[TARGET].to_numpy(dtype=float)
target_max_diff = float(np.max(np.abs(orig_target - dyn_target)))

orig_anomaly = original_aligned["_spatial_anomaly_c"].to_numpy(dtype=float)
dyn_anomaly = dynamic_aligned["_spatial_anomaly_c"].to_numpy(dtype=float)
anomaly_max_diff = float(np.max(np.abs(orig_anomaly - dyn_anomaly)))

print(f"Target max difference  : {target_max_diff:.12e} °C")
print(f"Anomaly max difference : {anomaly_max_diff:.12e} °C")


# =============================================================================
# 3. MODEL PARAMETER PARITY
# =============================================================================

section("3. COMPARE STORED PRODUCTION MODEL VS DYNAMIC MODEL PARAMETERS")

production_model = joblib.load(PRODUCTION_MODEL_PATH)
dynamic_model = joblib.load(DYNAMIC_MODEL_PATH)

prod_params = production_model.get_params(deep=False)
dyn_params = dynamic_model.get_params(deep=False)

all_param_names = sorted(set(prod_params) | set(dyn_params))
param_rows = []

for name in all_param_names:
    p = prod_params.get(name, "<MISSING>")
    d = dyn_params.get(name, "<MISSING>")
    matched = same_value(p, d)
    param_rows.append(
        {
            "parameter": name,
            "production_value": repr(p),
            "dynamic_value": repr(d),
            "match": matched,
        }
    )

param_df = pd.DataFrame(param_rows)
param_df.to_csv(PARAM_DIFF_PATH, index=False)
param_mismatch_count = int((~param_df["match"]).sum())

print(f"Parameter mismatches : {param_mismatch_count}")
if param_mismatch_count:
    print(param_df[~param_df["match"]].to_string(index=False))
else:
    print("PASS: complete estimator parameter configuration matches.")


# =============================================================================
# 4. PREDICTION COMPARISON ON IDENTICAL KEY-SORTED INPUT
# =============================================================================

section("4. COMPARE BOTH MODELS ON IDENTICAL KEY-SORTED INPUT")

X_aligned = original_aligned[predictors].copy()

pred_prod = np.asarray(production_model.predict(X_aligned), dtype=float)
pred_dyn = np.asarray(dynamic_model.predict(X_aligned), dtype=float)

stored_vs_dynamic_max_diff = float(np.max(np.abs(pred_prod - pred_dyn)))
stored_vs_dynamic_corr = corr(pred_prod, pred_dyn)

print(f"Max prediction difference : {stored_vs_dynamic_max_diff:.12e} °C")
print(f"Prediction correlation     : {stored_vs_dynamic_corr:.12f}")


# =============================================================================
# 5. FRESH REFIT USING EXACT ORIGINAL STEP-22B ROW ORDER
# =============================================================================

section("5. FRESH REFIT USING ORIGINAL STEP-22B ROW ORDER")

X_original_order = original[predictors].copy()
y_original_order = original["_spatial_anomaly_c"].to_numpy(dtype=float)

fresh_original_order_model = clone(production_model)
fresh_original_order_model.fit(X_original_order, y_original_order)

# Compare on the exact same original-order matrix.
pred_stored_original_order = np.asarray(
    production_model.predict(X_original_order), dtype=float
)
pred_fresh_original_order = np.asarray(
    fresh_original_order_model.predict(X_original_order), dtype=float
)

fresh_original_max_diff = float(
    np.max(np.abs(pred_stored_original_order - pred_fresh_original_order))
)
fresh_original_corr = corr(
    pred_stored_original_order,
    pred_fresh_original_order,
)

print(f"Stored vs fresh-original-order max diff : {fresh_original_max_diff:.12e} °C")
print(f"Stored vs fresh-original-order corr     : {fresh_original_corr:.12f}")


# =============================================================================
# 6. FRESH REFIT USING DYNAMIC ROW ORDER
# =============================================================================

section("6. FRESH REFIT USING DYNAMIC ROW ORDER")

X_dynamic_order = dynamic[predictors].copy()
y_dynamic_order = dynamic["_spatial_anomaly_c"].to_numpy(dtype=float)

fresh_dynamic_order_model = clone(production_model)
fresh_dynamic_order_model.fit(X_dynamic_order, y_dynamic_order)

pred_dynamic_saved_on_dynamic = np.asarray(
    dynamic_model.predict(X_dynamic_order), dtype=float
)
pred_fresh_dynamic_on_dynamic = np.asarray(
    fresh_dynamic_order_model.predict(X_dynamic_order), dtype=float
)

fresh_dynamic_max_diff = float(
    np.max(np.abs(pred_dynamic_saved_on_dynamic - pred_fresh_dynamic_on_dynamic))
)
fresh_dynamic_corr = corr(
    pred_dynamic_saved_on_dynamic,
    pred_fresh_dynamic_on_dynamic,
)

print(f"Saved dynamic vs fresh-dynamic-order max diff : {fresh_dynamic_max_diff:.12e} °C")
print(f"Saved dynamic vs fresh-dynamic-order corr     : {fresh_dynamic_corr:.12f}")


# =============================================================================
# 7. DECISION + AUDIT
# =============================================================================

section("7. DIAGNOSTIC DECISION")

audit_rows = []

add(
    audit_rows,
    "training_key_set_equal",
    key_sets_equal,
    f"original={len(original_keys)}, dynamic={len(dynamic_keys)}",
)
add(
    audit_rows,
    "all_93_features_reproduced",
    failed_features == 0,
    f"max_abs_error={max_feature_error:.12e}, tolerance={FEATURE_TOL}",
)
add(
    audit_rows,
    "lst_target_reproduced",
    target_max_diff <= TARGET_TOL,
    f"max_abs_error={target_max_diff:.12e}",
)
add(
    audit_rows,
    "anomaly_target_reproduced",
    anomaly_max_diff <= TARGET_TOL,
    f"max_abs_error={anomaly_max_diff:.12e}",
)
add(
    audit_rows,
    "model_parameter_parity",
    param_mismatch_count == 0,
    f"mismatches={param_mismatch_count}",
)
add(
    audit_rows,
    "original_raw_row_order_equal_dynamic",
    row_order_equal,
    "This may legitimately be FAIL; it is a diagnostic, not a methodology failure.",
)

# Fresh refit result is diagnostic evidence, not a required hard PASS.
if fresh_original_max_diff <= 1e-10:
    reproduction_class = "EXACT_REPRODUCTION_ON_ORIGINAL_ORDER"
elif fresh_original_corr >= 0.999:
    reproduction_class = "NEAR_IDENTICAL_REFIT_NOT_BITWISE_IDENTICAL"
else:
    reproduction_class = "INVESTIGATE_REFIT_DIFFERENCE"

print(f"\nReproduction classification: {reproduction_class}")

hard_failures = [
    row for row in audit_rows
    if row["status"] == "FAIL"
    and row["check"] != "original_raw_row_order_equal_dynamic"
]

summary = {
    "original_rows": int(len(original)),
    "dynamic_rows": int(len(dynamic)),
    "key_sets_equal": bool(key_sets_equal),
    "raw_row_order_equal": bool(row_order_equal),
    "max_feature_error": max_feature_error,
    "failed_feature_count": failed_features,
    "target_max_diff": target_max_diff,
    "anomaly_max_diff": anomaly_max_diff,
    "parameter_mismatch_count": param_mismatch_count,
    "stored_vs_dynamic_max_prediction_diff_c": stored_vs_dynamic_max_diff,
    "stored_vs_dynamic_prediction_correlation": stored_vs_dynamic_corr,
    "stored_vs_fresh_original_order_max_diff_c": fresh_original_max_diff,
    "stored_vs_fresh_original_order_correlation": fresh_original_corr,
    "saved_dynamic_vs_fresh_dynamic_order_max_diff_c": fresh_dynamic_max_diff,
    "saved_dynamic_vs_fresh_dynamic_order_correlation": fresh_dynamic_corr,
    "reproduction_classification": reproduction_class,
    "hard_failure_count": len(hard_failures),
}

pd.DataFrame(audit_rows).to_csv(AUDIT_PATH, index=False)
SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(f"\nFeature diff : {FEATURE_DIFF_PATH}")
print(f"Param diff   : {PARAM_DIFF_PATH}")
print(f"Audit        : {AUDIT_PATH}")
print(f"Summary      : {SUMMARY_PATH}")

if hard_failures:
    raise RuntimeError(
        f"Step 03B found {len(hard_failures)} hard reproducibility failure(s)."
    )

section("STEP 03B COMPLETED SUCCESSFULLY")
print(
    "Core training data, target, frozen feature set and estimator configuration "
    "are reproducible. Review the refit comparison above before Step 04."
)
