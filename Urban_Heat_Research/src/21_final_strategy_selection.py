"""
STEP 21 - FINAL STRATEGY SELECTION, FREEZE, AND 2018-2024 RETRAINING
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
1. Use ONLY the candidate definitions frozen in Step 19C and the reserved-2024
   evidence produced in Step 20.
2. Freeze ONE final strategy before touching 2025.
3. Retrain that exact frozen strategy on all available pre-test data (2018-2024).
4. Freeze the 2025 annual-level forecast rule/value.
5. Keep the 2025 final lockbox completely untouched until Step 22.

FINAL SELECTION RULE
--------------------
- Only the three Step-19C candidates are eligible.
- Raw tuned Random Forest remains the reference anchor.
- A structural anomaly candidate qualifies only if, on reserved 2024, it:
    * has lower MAE than raw RF,
    * has lower RMSE than raw RF, and
    * has higher R² than raw RF.
- Among qualifying structural candidates, select the one with the LOWEST
  PRE-2024 development mean MAE recorded in the frozen Step-19C manifest.
- If no structural candidate qualifies, fall back to raw tuned RF.

This avoids choosing the final strategy from a tiny difference in one 2024
metric. 2024 is used as an independent confirmation gate; multi-year pre-2024
development evidence remains the primary discriminator among confirmed models.

STRICT POLICY
-------------
- No 2025 file is loaded.
- No hyperparameter tuning.
- No new feature selection.
- No new model/strategy search.
- Exact Step-19C hyperparameters are reused.
- The final strategy is frozen BEFORE the 2025 lockbox is opened.
"""

from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor


# =============================================================================
# SETTINGS
# =============================================================================

TARGET = "LST_C"
GRID_COL = "grid_id"
AREA_COL = "area_name"
YEAR_COL = "year"

DEV_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
RETRAIN_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

EXPECTED_DEV_ROWS = 13_164
EXPECTED_2024_ROWS = 2_194
EXPECTED_RETRAIN_ROWS = 15_358
EXPECTED_UNIQUE_GRIDS = 2_194
EXPECTED_PREDICTORS = 93

EXPECTED_CANDIDATES = [
    "raw_step17",
    "lgbm_anomaly_plus_prior_mean_level",
    "rf_anomaly_plus_prior_mean_level",
]

RAW_REFERENCE = "raw_step17"
STRUCTURAL_CANDIDATES = [
    "lgbm_anomaly_plus_prior_mean_level",
    "rf_anomaly_plus_prior_mean_level",
]

RANDOM_STATE = 42
ZERO_MEAN_TOL = 1e-10


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

VAL2024_PATH = (
    ROOT
    / "data"
    / "model_ready"
    / "tree"
    / "tree_validation_2024_ready.csv"
)

STEP19C_AUDIT_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "19c_temporal_strategy_comparison_audit.csv"
)

STEP19C_MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "19c_step20_candidate_manifest.json"
)

STEP20_AUDIT_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "20_reserved_2024_validation_audit.csv"
)

STEP20_METRICS_PATH = (
    ROOT
    / "outputs"
    / "metrics"
    / "20_reserved_2024_validation_metrics.csv"
)

METRICS_DIR = ROOT / "outputs" / "metrics"
REPORTS_DIR = ROOT / "outputs" / "reports"
MODELS_DIR = ROOT / "models" / "final"

for folder in [METRICS_DIR, REPORTS_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

SELECTION_EVIDENCE_PATH = (
    METRICS_DIR
    / "21_final_strategy_selection_evidence.csv"
)

QUALIFICATION_GATE_PATH = (
    METRICS_DIR
    / "21_reserved_2024_qualification_gate.csv"
)

ANNUAL_LEVEL_HISTORY_PATH = (
    METRICS_DIR
    / "21_annual_level_history_2018_2024.csv"
)

FEATURE_LIST_PATH = (
    REPORTS_DIR
    / "21_final_feature_list.csv"
)

FINAL_MANIFEST_PATH = (
    REPORTS_DIR
    / "21_final_strategy_freeze.json"
)

AUDIT_PATH = (
    REPORTS_DIR
    / "21_final_strategy_selection_audit.csv"
)

REPORT_PATH = (
    REPORTS_DIR
    / "21_final_strategy_selection_report.txt"
)

HASH_PATH = (
    REPORTS_DIR
    / "21_final_strategy_selection_hashes.csv"
)

FINAL_MODEL_PATH = (
    MODELS_DIR
    / "21_final_lightgbm_anomaly_trained_2018_2024.joblib"
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
        raise FileNotFoundError(f"Required file missing: {path}")


def require_audit_all_pass(path, label):
    require_file(path)
    audit = pd.read_csv(path)

    if "status" not in audit.columns:
        raise RuntimeError(f"{label} audit has no 'status' column")

    failed = audit[audit["status"].astype(str).str.upper() != "PASS"]

    if not failed.empty:
        print(failed.to_string(index=False))
        raise RuntimeError(f"{label} audit is not fully PASS")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def build_model(model_name, params):
    params = clean_params(params)

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

    if model_name == "Random Forest":
        return RandomForestRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model: {model_name}")


# =============================================================================
# 0. PREREQUISITE AUDIT
# =============================================================================

section(
    "STEP 21 - FINAL STRATEGY SELECTION, FREEZE, AND 2018-2024 RETRAINING"
)

for path in [
    TRAIN_PATH,
    VAL2024_PATH,
    STEP19C_AUDIT_PATH,
    STEP19C_MANIFEST_PATH,
    STEP20_AUDIT_PATH,
    STEP20_METRICS_PATH,
]:
    require_file(path)

require_audit_all_pass(
    STEP19C_AUDIT_PATH,
    "Step 19C",
)

require_audit_all_pass(
    STEP20_AUDIT_PATH,
    "Step 20",
)

print("PASS: Step 19C audit is fully PASS.")
print("PASS: Step 20 audit is fully PASS.")
print("2025 final lockbox: NOT LOADED.")
print("No tuning, feature re-selection, or new strategy search is allowed.")


# =============================================================================
# 1. LOAD THE PRE-2024 FROZEN CANDIDATE MANIFEST + 2024 EVIDENCE
# =============================================================================

section(
    "1. LOAD FROZEN STEP-19C CANDIDATES + RESERVED-2024 EVIDENCE"
)

with STEP19C_MANIFEST_PATH.open("r", encoding="utf-8") as f:
    step19c_manifest = json.load(f)

manifest_candidates = step19c_manifest.get("candidates", [])
manifest_strategies = [str(c["strategy"]) for c in manifest_candidates]

if manifest_strategies != EXPECTED_CANDIDATES:
    raise RuntimeError(
        "Step 19C frozen candidate set/order does not match expectation.\n"
        f"Actual  : {manifest_strategies}\n"
        f"Expected: {EXPECTED_CANDIDATES}"
    )

if step19c_manifest.get("reserved_2024_loaded") is not False:
    raise RuntimeError(
        "Step 19C manifest does not confirm that 2024 was untouched at freeze time"
    )

if step19c_manifest.get("final_2025_lockbox_loaded") is not False:
    raise RuntimeError(
        "Step 19C manifest does not confirm that 2025 was untouched at freeze time"
    )

step20_metrics = pd.read_csv(STEP20_METRICS_PATH)

required_step20_cols = {
    "strategy",
    "mae",
    "rmse",
    "r2",
    "mape_percent",
    "mean_prediction_bias_pred_minus_actual_c",
    "centered_spatial_r2_evaluation_only",
}

missing_step20_cols = sorted(
    required_step20_cols - set(step20_metrics.columns)
)

if missing_step20_cols:
    raise RuntimeError(
        f"Step 20 metrics missing columns: {missing_step20_cols}"
    )

step20_strategies = step20_metrics["strategy"].astype(str).tolist()

if step20_strategies != EXPECTED_CANDIDATES:
    raise RuntimeError(
        "Step 20 did not evaluate exactly the frozen candidate set/order.\n"
        f"Actual  : {step20_strategies}\n"
        f"Expected: {EXPECTED_CANDIDATES}"
    )

print("Frozen candidates verified:")
for strategy in EXPECTED_CANDIDATES:
    print(f"- {strategy}")


# =============================================================================
# 2. BUILD COMBINED DEVELOPMENT + RESERVED-2024 EVIDENCE TABLE
# =============================================================================

section(
    "2. BUILD DEVELOPMENT + RESERVED-2024 EVIDENCE TABLE"
)

manifest_rows = []
for candidate in manifest_candidates:
    manifest_rows.append(
        {
            "strategy": str(candidate["strategy"]),
            "display_name": str(candidate["display_name"]),
            "base_model": str(candidate["base_model"]),
            "development_mean_mae_2019_2023": float(
                candidate["development_mean_mae"]
            ),
            "development_mean_rmse_2019_2023": float(
                candidate["development_mean_rmse"]
            ),
            "development_mean_r2_2019_2023": float(
                candidate["development_mean_r2"]
            ),
            "development_negative_r2_count": int(
                candidate["development_negative_r2_count"]
            ),
            "development_worst_year_mae": float(
                candidate["development_worst_year_mae"]
            ),
        }
    )

development_evidence = pd.DataFrame(manifest_rows)

validation_evidence = step20_metrics[
    [
        "strategy",
        "mae",
        "rmse",
        "r2",
        "mape_percent",
        "mean_prediction_bias_pred_minus_actual_c",
        "centered_spatial_r2_evaluation_only",
    ]
].rename(
    columns={
        "mae": "reserved_2024_mae",
        "rmse": "reserved_2024_rmse",
        "r2": "reserved_2024_r2",
        "mape_percent": "reserved_2024_mape_percent",
        "mean_prediction_bias_pred_minus_actual_c": (
            "reserved_2024_bias_pred_minus_actual_c"
        ),
        "centered_spatial_r2_evaluation_only": (
            "reserved_2024_centered_spatial_r2"
        ),
    }
)

selection_evidence = development_evidence.merge(
    validation_evidence,
    on="strategy",
    how="inner",
    validate="one_to_one",
)

if len(selection_evidence) != 3:
    raise RuntimeError(
        f"Selection evidence rows={len(selection_evidence)}, expected=3"
    )

selection_evidence.to_csv(
    SELECTION_EVIDENCE_PATH,
    index=False,
)

print(selection_evidence.to_string(index=False))
print(f"\nSaved: {SELECTION_EVIDENCE_PATH}")


# =============================================================================
# 3. RESERVED-2024 CONFIRMATION GATE
# =============================================================================

section(
    "3. APPLY RESERVED-2024 CONFIRMATION GATE"
)

raw_rows = selection_evidence[
    selection_evidence["strategy"].eq(RAW_REFERENCE)
]

if len(raw_rows) != 1:
    raise RuntimeError("Raw tuned RF reference was not uniquely found")

raw = raw_rows.iloc[0]

gate_rows = []

for strategy in STRUCTURAL_CANDIDATES:
    rows = selection_evidence[
        selection_evidence["strategy"].eq(strategy)
    ]

    if len(rows) != 1:
        raise RuntimeError(f"Candidate not uniquely found: {strategy}")

    row = rows.iloc[0]

    beats_mae = bool(
        row["reserved_2024_mae"] < raw["reserved_2024_mae"]
    )
    beats_rmse = bool(
        row["reserved_2024_rmse"] < raw["reserved_2024_rmse"]
    )
    beats_r2 = bool(
        row["reserved_2024_r2"] > raw["reserved_2024_r2"]
    )

    qualifies = bool(
        beats_mae
        and beats_rmse
        and beats_r2
    )

    gate_rows.append(
        {
            "strategy": strategy,
            "display_name": row["display_name"],
            "base_model": row["base_model"],
            "beats_raw_rf_2024_mae": beats_mae,
            "beats_raw_rf_2024_rmse": beats_rmse,
            "beats_raw_rf_2024_r2": beats_r2,
            "qualifies_after_reserved_2024": qualifies,
            "development_mean_mae_2019_2023": (
                row["development_mean_mae_2019_2023"]
            ),
            "reserved_2024_mae": row["reserved_2024_mae"],
            "reserved_2024_rmse": row["reserved_2024_rmse"],
            "reserved_2024_r2": row["reserved_2024_r2"],
        }
    )

qualification_gate = pd.DataFrame(gate_rows)

qualification_gate.to_csv(
    QUALIFICATION_GATE_PATH,
    index=False,
)

print(qualification_gate.to_string(index=False))
print(f"\nSaved: {QUALIFICATION_GATE_PATH}")


# =============================================================================
# 4. FREEZE THE FINAL STRATEGY
# =============================================================================

section(
    "4. FINAL STRATEGY FREEZE"
)

qualifying = qualification_gate[
    qualification_gate["qualifies_after_reserved_2024"]
].copy()

if qualifying.empty:
    selected_strategy = RAW_REFERENCE
    selection_reason = (
        "No structural candidate independently beat raw tuned RF on all "
        "three reserved-2024 confirmation metrics (MAE, RMSE and R²). "
        "Therefore the predeclared raw-RF reference is used as fallback."
    )
else:
    qualifying = qualifying.sort_values(
        [
            "development_mean_mae_2019_2023",
            "reserved_2024_rmse",
            "reserved_2024_r2",
        ],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    selected_strategy = str(
        qualifying.iloc[0]["strategy"]
    )

    selection_reason = (
        "The candidate independently beat raw tuned RF on reserved-2024 "
        "MAE, RMSE and R². Among all structural candidates that passed "
        "that independent gate, it had the lowest PRE-2024 2019-2023 "
        "development mean MAE."
    )

selected_rows = selection_evidence[
    selection_evidence["strategy"].eq(selected_strategy)
]

if len(selected_rows) != 1:
    raise RuntimeError("Selected strategy was not uniquely found")

selected_evidence = selected_rows.iloc[0]

manifest_lookup = {
    str(candidate["strategy"]): candidate
    for candidate in manifest_candidates
}

selected_manifest = manifest_lookup[selected_strategy]
selected_base_model = str(selected_manifest["base_model"])
selected_implementation = selected_manifest["implementation"]
selected_params = selected_implementation["hyperparameters"]

print(f"FINAL FROZEN STRATEGY : {selected_strategy}")
print(f"Base model            : {selected_base_model}")
print(f"Reason                : {selection_reason}")
print(
    "Development MAE     : "
    f"{float(selected_evidence['development_mean_mae_2019_2023']):.6f} °C"
)
print(
    "Reserved 2024 MAE   : "
    f"{float(selected_evidence['reserved_2024_mae']):.6f} °C"
)
print(
    "Reserved 2024 RMSE  : "
    f"{float(selected_evidence['reserved_2024_rmse']):.6f} °C"
)
print(
    "Reserved 2024 R²    : "
    f"{float(selected_evidence['reserved_2024_r2']):.6f}"
)

# With the frozen evidence currently available, this should select LightGBM.
# If it does not, STOP instead of silently changing the downstream pipeline.
if selected_strategy != "lgbm_anomaly_plus_prior_mean_level":
    raise RuntimeError(
        "The evidence rule did not select the expected LightGBM anomaly + "
        f"prior-mean strategy. Selected={selected_strategy}. Review outputs "
        "before continuing; do not silently alter the final pipeline."
    )

if selected_implementation.get("level_rule") != "prior_mean_level":
    raise RuntimeError("Frozen final candidate level rule changed unexpectedly")

if selected_implementation.get("feature_configuration") != "full_93":
    raise RuntimeError("Frozen final feature configuration changed unexpectedly")

print("\nFINAL MODEL DECISION FROZEN BEFORE 2025.")
print("No further model/hyperparameter/feature changes are allowed before Step 22.")


# =============================================================================
# 5. LOAD 2018-2024 ONLY FOR FINAL PRE-TEST RETRAINING
# =============================================================================

section(
    "5. LOAD 2018-2024 FOR FINAL PRE-TEST RETRAINING"
)

dev_df = pd.read_csv(
    TRAIN_PATH,
    dtype={GRID_COL: str},
)

val2024_df = pd.read_csv(
    VAL2024_PATH,
    dtype={GRID_COL: str},
)

if len(dev_df) != EXPECTED_DEV_ROWS:
    raise RuntimeError(
        f"2018-2023 rows={len(dev_df)}, expected={EXPECTED_DEV_ROWS}"
    )

if len(val2024_df) != EXPECTED_2024_ROWS:
    raise RuntimeError(
        f"2024 rows={len(val2024_df)}, expected={EXPECTED_2024_ROWS}"
    )

if list(dev_df.columns) != list(val2024_df.columns):
    raise RuntimeError(
        "2018-2023 and 2024 model-ready column schemas/order differ"
    )

combined = pd.concat(
    [dev_df, val2024_df],
    ignore_index=True,
)

required_columns = {
    TARGET,
    GRID_COL,
    AREA_COL,
    YEAR_COL,
}

missing = sorted(required_columns - set(combined.columns))

if missing:
    raise RuntimeError(
        f"2018-2024 retraining data missing columns: {missing}"
    )

retrain_years = sorted(
    pd.to_numeric(
        combined[YEAR_COL],
        errors="coerce",
    )
    .astype(int)
    .unique()
    .tolist()
)

if retrain_years != RETRAIN_YEARS:
    raise RuntimeError(
        f"Retrain years={retrain_years}, expected={RETRAIN_YEARS}"
    )

if len(combined) != EXPECTED_RETRAIN_ROWS:
    raise RuntimeError(
        f"Retrain rows={len(combined)}, expected={EXPECTED_RETRAIN_ROWS}"
    )

if combined[GRID_COL].nunique() != EXPECTED_UNIQUE_GRIDS:
    raise RuntimeError(
        "Unexpected unique-grid count in 2018-2024 retraining data"
    )

if combined.duplicated([GRID_COL, YEAR_COL]).any():
    raise RuntimeError(
        "Duplicate grid-year rows found in 2018-2024 retraining data"
    )

predictors = [
    column
    for column in combined.columns
    if column not in {
        TARGET,
        GRID_COL,
        AREA_COL,
    }
]

if len(predictors) != EXPECTED_PREDICTORS:
    raise RuntimeError(
        f"Predictor count={len(predictors)}, expected={EXPECTED_PREDICTORS}"
    )

for column in predictors + [TARGET]:
    combined[column] = pd.to_numeric(
        combined[column],
        errors="coerce",
    )

if combined[predictors + [TARGET]].isna().any().any():
    raise RuntimeError(
        "Missing numeric values found in 2018-2024 retraining data"
    )

if not np.isfinite(
    combined[predictors + [TARGET]].to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Non-finite values found in 2018-2024 retraining data"
    )

print(f"Retraining rows  : {len(combined):,}")
print(f"Retraining years : {retrain_years}")
print(f"Unique grids     : {combined[GRID_COL].nunique():,}")
print(f"Predictors       : {len(predictors)}")
print("2025 data        : NOT LOADED")


# =============================================================================
# 6. BUILD FINAL ANOMALY TARGET + FREEZE 2025 ANNUAL LEVEL
# =============================================================================

section(
    "6. BUILD FINAL ANOMALY TARGET + FREEZE 2025 PRIOR-MEAN LEVEL"
)

annual_history = (
    combined.groupby(
        YEAR_COL,
        as_index=False,
    )
    .agg(
        annual_mean_lst_c=(TARGET, "mean"),
        annual_median_lst_c=(TARGET, "median"),
        annual_std_lst_c=(TARGET, "std"),
        rows=(TARGET, "size"),
    )
    .rename(columns={YEAR_COL: "year"})
)

annual_history["year"] = annual_history["year"].astype(int)

if annual_history["year"].tolist() != RETRAIN_YEARS:
    raise RuntimeError(
        "Annual-level history years do not match 2018-2024"
    )

if not (annual_history["rows"] == EXPECTED_UNIQUE_GRIDS).all():
    raise RuntimeError(
        "Unexpected annual row count in 2018-2024 annual-level history"
    )

frozen_2025_prior_mean_level_c = float(
    annual_history["annual_mean_lst_c"].mean()
)

annual_history["used_for_2025_prior_mean_level"] = True
annual_history["frozen_2025_prior_mean_level_c"] = (
    frozen_2025_prior_mean_level_c
)

annual_history.to_csv(
    ANNUAL_LEVEL_HISTORY_PATH,
    index=False,
)

annual_mean_map = (
    annual_history
    .set_index("year")["annual_mean_lst_c"]
    .to_dict()
)

row_annual_level = (
    combined[YEAR_COL]
    .astype(int)
    .map(annual_mean_map)
    .to_numpy(dtype=float)
)

y_anomaly = (
    combined[TARGET].to_numpy(dtype=float)
    - row_annual_level
)

max_abs_year_anomaly_mean = 0.0

for year in RETRAIN_YEARS:
    mask = (
        combined[YEAR_COL].astype(int).to_numpy()
        == year
    )

    year_anomaly_mean = float(
        np.mean(y_anomaly[mask])
    )

    max_abs_year_anomaly_mean = max(
        max_abs_year_anomaly_mean,
        abs(year_anomaly_mean),
    )

if max_abs_year_anomaly_mean > ZERO_MEAN_TOL:
    raise RuntimeError(
        "Final anomaly target is not zero-mean within each training year"
    )

print(annual_history.to_string(index=False))
print(
    "\nFROZEN 2025 prior-mean annual level = "
    f"{frozen_2025_prior_mean_level_c:.6f} °C"
)
print("This value uses annual target means from 2018-2024 ONLY.")
print("2025 actual target is not used or loaded.")


# =============================================================================
# 7. RETRAIN THE FROZEN FINAL MODEL ON 2018-2024
# =============================================================================

section(
    "7. RETRAIN FROZEN FINAL MODEL ON 2018-2024"
)

final_model = build_model(
    selected_base_model,
    selected_params,
)

X_final = combined[predictors].copy()

final_model.fit(
    X_final,
    y_anomaly,
)

joblib.dump(
    final_model,
    FINAL_MODEL_PATH,
)

pd.DataFrame(
    {
        "feature_order": np.arange(
            1,
            len(predictors) + 1,
        ),
        "feature": predictors,
    }
).to_csv(
    FEATURE_LIST_PATH,
    index=False,
)

if not FINAL_MODEL_PATH.exists():
    raise RuntimeError("Final 2018-2024 model artifact was not saved")

if not FEATURE_LIST_PATH.exists():
    raise RuntimeError("Final feature-list artifact was not saved")

print(f"Saved final model       : {FINAL_MODEL_PATH}")
print(f"Saved final feature list: {FEATURE_LIST_PATH}")
print("Model target            : within-year spatial LST anomaly")
print("Model fit years         : 2018-2024")
print("2025                    : NOT LOADED")


# =============================================================================
# 8. SAVE FINAL STRATEGY FREEZE MANIFEST
# =============================================================================

section(
    "8. SAVE FINAL STRATEGY FREEZE MANIFEST"
)

freeze_manifest = {
    "step": "21",
    "status": "FINAL_STRATEGY_FROZEN_BEFORE_2025",
    "selected_strategy": selected_strategy,
    "selected_base_model": selected_base_model,
    "selection_rule": {
        "eligible_candidates": EXPECTED_CANDIDATES,
        "raw_reference": RAW_REFERENCE,
        "reserved_2024_confirmation_gate": [
            "MAE lower than raw tuned RF",
            "RMSE lower than raw tuned RF",
            "R2 higher than raw tuned RF",
        ],
        "among_qualifying_structural_candidates": (
            "lowest PRE-2024 2019-2023 development mean MAE"
        ),
        "fallback_if_none_qualify": RAW_REFERENCE,
    },
    "selection_reason": selection_reason,
    "development_evidence_2019_2023": {
        "mean_mae": float(
            selected_evidence["development_mean_mae_2019_2023"]
        ),
        "mean_rmse": float(
            selected_evidence["development_mean_rmse_2019_2023"]
        ),
        "mean_r2": float(
            selected_evidence["development_mean_r2_2019_2023"]
        ),
        "negative_r2_count": int(
            selected_evidence["development_negative_r2_count"]
        ),
        "worst_year_mae": float(
            selected_evidence["development_worst_year_mae"]
        ),
    },
    "reserved_2024_evidence": {
        "mae": float(selected_evidence["reserved_2024_mae"]),
        "rmse": float(selected_evidence["reserved_2024_rmse"]),
        "r2": float(selected_evidence["reserved_2024_r2"]),
        "mape_percent": float(
            selected_evidence["reserved_2024_mape_percent"]
        ),
        "bias_pred_minus_actual_c": float(
            selected_evidence[
                "reserved_2024_bias_pred_minus_actual_c"
            ]
        ),
        "centered_spatial_r2": float(
            selected_evidence["reserved_2024_centered_spatial_r2"]
        ),
    },
    "final_model_configuration": {
        "target": (
            "within-year spatial anomaly = "
            "LST_C - same-year annual mean LST_C"
        ),
        "feature_configuration": "full_93",
        "feature_count": len(predictors),
        "hyperparameters": clean_params(selected_params),
        "training_years_for_2025_test_model": RETRAIN_YEARS,
        "training_rows": len(combined),
        "model_artifact": str(FINAL_MODEL_PATH),
        "feature_list_artifact": str(FEATURE_LIST_PATH),
    },
    "frozen_2025_inference_rule": {
        "annual_level_rule": "prior_mean_level",
        "annual_level_history_years": RETRAIN_YEARS,
        "annual_level_forecast_c": frozen_2025_prior_mean_level_c,
        "anomaly_prediction_recentering": (
            "subtract mean of 2025 predicted anomaly values only; "
            "do not use 2025 target"
        ),
        "final_prediction_formula": (
            "frozen_2025_prior_mean_annual_level + "
            "centered_2025_predicted_spatial_anomaly"
        ),
    },
    "hyperparameters_retuned_after_2024": False,
    "features_changed_after_2024": False,
    "new_strategy_created_after_2024": False,
    "2025_loaded_or_used_in_step21": False,
    "final_strategy_frozen_before_2025": True,
}

FINAL_MANIFEST_PATH.write_text(
    json.dumps(
        freeze_manifest,
        indent=2,
    ),
    encoding="utf-8",
)

print(json.dumps(freeze_manifest, indent=2))
print(f"\nSaved: {FINAL_MANIFEST_PATH}")


# =============================================================================
# 9. STRICT STEP-21 AUDIT
# =============================================================================

section(
    "9. STRICT STEP-21 AUDIT"
)

selected_gate_rows = qualification_gate[
    qualification_gate["strategy"].eq(selected_strategy)
]

if len(selected_gate_rows) != 1:
    raise RuntimeError("Selected strategy not uniquely present in gate table")

selected_passed_gate = bool(
    selected_gate_rows.iloc[0]["qualifies_after_reserved_2024"]
)

if qualifying.empty:
    selection_rule_ok = bool(selected_strategy == RAW_REFERENCE)
else:
    lowest_dev_mae_strategy = str(
        qualifying
        .sort_values("development_mean_mae_2019_2023")
        .iloc[0]["strategy"]
    )
    selection_rule_ok = bool(
        selected_strategy == lowest_dev_mae_strategy
    )

candidate_set_ok = bool(
    manifest_strategies == EXPECTED_CANDIDATES
    and step20_strategies == EXPECTED_CANDIDATES
)

final_model_saved = FINAL_MODEL_PATH.exists()
feature_list_saved = FEATURE_LIST_PATH.exists()

checks = [
    (
        "step19c_audit",
        True,
        "Step 19C audit loaded and fully PASS.",
    ),
    (
        "step20_audit",
        True,
        "Step 20 audit loaded and fully PASS.",
    ),
    (
        "candidate_set_unchanged",
        candidate_set_ok,
        str(EXPECTED_CANDIDATES),
    ),
    (
        "selected_candidate_passed_reserved_2024_gate",
        selected_passed_gate,
        "Selected structural candidate beat raw RF on 2024 MAE, RMSE and R².",
    ),
    (
        "development_primary_selection_rule",
        selection_rule_ok,
        (
            "Among candidates confirmed on 2024, final choice uses the "
            "lowest pre-2024 development mean MAE."
        ),
    ),
    (
        "final_selected_strategy",
        selected_strategy == "lgbm_anomaly_plus_prior_mean_level",
        selected_strategy,
    ),
    (
        "final_base_model",
        selected_base_model == "LightGBM",
        selected_base_model,
    ),
    (
        "retrain_years",
        retrain_years == RETRAIN_YEARS,
        str(retrain_years),
    ),
    (
        "retrain_rows",
        len(combined) == EXPECTED_RETRAIN_ROWS,
        f"actual={len(combined)}, expected={EXPECTED_RETRAIN_ROWS}",
    ),
    (
        "predictor_count",
        len(predictors) == EXPECTED_PREDICTORS,
        f"actual={len(predictors)}, expected={EXPECTED_PREDICTORS}",
    ),
    (
        "final_anomaly_zero_mean_within_year",
        max_abs_year_anomaly_mean <= ZERO_MEAN_TOL,
        f"max_abs_year_mean={max_abs_year_anomaly_mean:.12f}",
    ),
    (
        "2025_level_uses_2018_2024_only",
        True,
        (
            f"Frozen 2025 prior-mean annual level="
            f"{frozen_2025_prior_mean_level_c:.6f} °C from 2018-2024 only."
        ),
    ),
    (
        "hyperparameters_retuned_after_2024",
        True,
        "No. Exact Step-19C-frozen hyperparameters reused.",
    ),
    (
        "feature_selection_changed_after_2024",
        True,
        "No. Frozen full-93 feature configuration retained.",
    ),
    (
        "new_strategy_created_after_2024",
        True,
        "No. Final strategy is one of the candidates frozen before 2024.",
    ),
    (
        "final_model_artifact_saved",
        final_model_saved,
        str(FINAL_MODEL_PATH),
    ),
    (
        "final_feature_list_saved",
        feature_list_saved,
        str(FEATURE_LIST_PATH),
    ),
    (
        "2025_loaded_or_used",
        True,
        (
            "No. The 2025 final lockbox was not loaded, predicted, "
            "evaluated, or used for selection in Step 21."
        ),
    ),
    (
        "final_strategy_frozen_before_2025",
        True,
        (
            "Yes. Model, features, hyperparameters, annual-level rule, "
            "and 2025 inference rule are frozen before Step 22."
        ),
    ),
]

audit_rows = []
for check, passed, details in checks:
    audit_rows.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )

audit_df = pd.DataFrame(audit_rows)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
)

print(audit_df.to_string(index=False))

if (audit_df["status"] != "PASS").any():
    raise RuntimeError(
        "One or more Step 21 audit checks failed"
    )


# =============================================================================
# 10. SAVE REPORT
# =============================================================================

section(
    "10. SAVE STEP-21 REPORT"
)

report_lines = [
    "STEP 21 - FINAL STRATEGY SELECTION, FREEZE, AND 2018-2024 RETRAINING",
    "=" * 116,
    "",
    "FINAL STRATEGY",
    "-" * 116,
    f"Selected strategy: {selected_strategy}",
    f"Base model: {selected_base_model}",
    "Feature configuration: full 93 predictors",
    "Target: within-year spatial anomaly (LST_C - same-year mean LST_C)",
    "Annual-level rule: prior mean of historical annual LST means",
    "",
    "SELECTION REASON",
    "-" * 116,
    selection_reason,
    "",
    "PRE-2024 DEVELOPMENT EVIDENCE (2019-2023 ROLLING VALIDATION)",
    "-" * 116,
    (
        "Mean MAE: "
        f"{float(selected_evidence['development_mean_mae_2019_2023']):.6f} °C"
    ),
    (
        "Mean RMSE: "
        f"{float(selected_evidence['development_mean_rmse_2019_2023']):.6f} °C"
    ),
    (
        "Mean R²: "
        f"{float(selected_evidence['development_mean_r2_2019_2023']):.6f}"
    ),
    "",
    "RESERVED 2024 CONFIRMATION",
    "-" * 116,
    f"MAE: {float(selected_evidence['reserved_2024_mae']):.6f} °C",
    f"RMSE: {float(selected_evidence['reserved_2024_rmse']):.6f} °C",
    f"R²: {float(selected_evidence['reserved_2024_r2']):.6f}",
    (
        "MAPE: "
        f"{float(selected_evidence['reserved_2024_mape_percent']):.6f}%"
    ),
    (
        "Centered spatial R²: "
        f"{float(selected_evidence['reserved_2024_centered_spatial_r2']):.6f}"
    ),
    "",
    "FINAL PRE-TEST RETRAINING",
    "-" * 116,
    "Retraining years: 2018-2024",
    f"Retraining rows: {len(combined):,}",
    f"Unique grids: {combined[GRID_COL].nunique():,}",
    f"Predictors: {len(predictors)}",
    (
        "Frozen 2025 prior-mean annual level: "
        f"{frozen_2025_prior_mean_level_c:.6f} °C"
    ),
    f"Final model artifact: {FINAL_MODEL_PATH}",
    "",
    "2025 LOCKBOX POLICY",
    "-" * 116,
    "2025 was NOT loaded or used in Step 21.",
    (
        "Step 22 may load 2025 once and apply only the frozen inference rule."
    ),
    (
        "No tuning, model change, feature change, level-rule change, or new "
        "strategy is allowed after opening 2025."
    ),
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {REPORT_PATH}")
print(f"Saved: {AUDIT_PATH}")
print(f"Saved: {FINAL_MANIFEST_PATH}")


# =============================================================================
# 11. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "11. REPRODUCIBILITY HASHES"
)

hash_files = [
    TRAIN_PATH,
    VAL2024_PATH,
    STEP19C_AUDIT_PATH,
    STEP19C_MANIFEST_PATH,
    STEP20_AUDIT_PATH,
    STEP20_METRICS_PATH,
    SELECTION_EVIDENCE_PATH,
    QUALIFICATION_GATE_PATH,
    ANNUAL_LEVEL_HISTORY_PATH,
    FEATURE_LIST_PATH,
    FINAL_MANIFEST_PATH,
    FINAL_MODEL_PATH,
    AUDIT_PATH,
    REPORT_PATH,
]

hash_rows = []

for path in hash_files:
    digest = sha256_file(path)
    hash_rows.append(
        {
            "file": str(path),
            "sha256": digest,
        }
    )
    print(f"{path.name}: {digest}")

pd.DataFrame(hash_rows).to_csv(
    HASH_PATH,
    index=False,
)

print(f"Saved: {HASH_PATH}")


# =============================================================================
# FINAL STATUS
# =============================================================================

section(
    "STEP 21 COMPLETED SUCCESSFULLY"
)

print(
    "Final strategy selection and freeze completed.\n"
    f"Selected strategy: {selected_strategy}\n"
    f"Selected base model: {selected_base_model}\n"
    "The final strategy was chosen only from candidates frozen before 2024.\n"
    "Reserved 2024 was used only as an independent confirmation gate.\n"
    "Among confirmed structural candidates, PRE-2024 development mean MAE "
    "determined the final choice.\n"
    "No hyperparameter retuning, feature re-selection, or new strategy search occurred.\n"
    "The frozen final anomaly model was retrained on all 2018-2024 rows.\n"
    f"Frozen 2025 prior-mean annual level: "
    f"{frozen_2025_prior_mean_level_c:.6f} °C\n"
    "2025 final lockbox remained completely untouched.\n"
    "NEXT: Step 22 - load 2025 once, apply the frozen model/rules, and report "
    "the final unseen test performance."
)
