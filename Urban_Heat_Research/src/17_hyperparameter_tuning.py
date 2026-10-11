
"""
STEP 17 - CORRECTED + FOCUSED HYPERPARAMETER TUNING
Urban Heat Research - Kaduwela, Sri Lanka

WHY THIS UPDATED VERSION EXISTS
-------------------------------
The first Step 17 run was scientifically useful, but two refinements are made
before finalizing tuning:

1) Reproducibility correction
   The Step 17 untuned LightGBM reference did not exactly reproduce the
   Step 14/16 untuned LightGBM aggregate metrics. This version preserves the
   original pandas/DataFrame numeric representation instead of converting the
   predictors to float32 NumPy arrays before training.

2) Focused boundary refinement
   The first tuning run showed that several selected hyperparameters sat at
   search-space boundaries (stronger regularization / simpler models). This
   version performs a small deterministic refinement around those promising
   regions, using ONLY the same 2018-2023 development period.

DATA POLICY
-----------
2015-2017 : historical context only; already used upstream to create lags/history
2018-2023 : direct development/tuning rows
2024      : NOT loaded; reserved validation
2025      : NOT loaded; final lockbox

ROLLING-ORIGIN FOLDS
--------------------
2018       -> 2019
2018-2019  -> 2020
2018-2020  -> 2021
2018-2021  -> 2022
2018-2022  -> 2023

PRIMARY OBJECTIVE
-----------------
Mean rolling-origin validation MAE.

SUPPORTING DIAGNOSTICS
----------------------
RMSE, R², MAPE, negative-R² folds, worst-year MAE, validation-MAE variability,
and train-vs-validation MAE gap.

IMPORTANT
---------
- No random K-fold.
- No 2024 or 2025.
- No final model selection.
- Untuned Step 12-15 reference configurations are reproduced BEFORE tuning.
- If an untuned reference does not reproduce Step 16 within tolerance,
  the script stops before the tuning search.
"""

from pathlib import Path
import gc
import hashlib
import json
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore


# =============================================================================
# 1. SETTINGS
# =============================================================================

RANDOM_STATE = 42
REPRO_TOLERANCE = 1e-6

TARGET = "LST_C"
YEAR_COL = "year"
ID_COLS = ["grid_id", "area_name"]

DEVELOPMENT_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]
EXPECTED_PREDICTOR_COUNT = 93


# =============================================================================
# 2. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "tree"
    / "tree_train_2018_2023_ready.csv"
)

STEP16_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "16_candidate_model_summary.csv"
)

STEP16_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "16_candidate_comparison_audit.csv"
)

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TUNED_MODELS_DIR = PROJECT_ROOT / "models" / "tuned"

for folder in [
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TUNED_MODELS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

REFERENCE_CHECK_PATH = (
    METRICS_DIR / "17_reference_reproduction_check.csv"
)

SEARCH_PATHS = {
    "Random Forest": METRICS_DIR / "17_random_forest_tuning_search.csv",
    "XGBoost": METRICS_DIR / "17_xgboost_tuning_search.csv",
    "LightGBM": METRICS_DIR / "17_lightgbm_tuning_search.csv",
    "NGBoost": METRICS_DIR / "17_ngboost_tuning_search.csv",
}

BEST_FOLD_METRICS_PATH = METRICS_DIR / "17_best_config_fold_metrics.csv"
TUNED_SUMMARY_PATH = METRICS_DIR / "17_tuned_model_summary.csv"
TUNED_VS_UNTUNED_PATH = METRICS_DIR / "17_tuned_vs_untuned.csv"

PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "17_best_tuned_rolling_predictions.csv"
)

BEST_PARAMS_PATH = REPORTS_DIR / "17_best_hyperparameters.json"
SEARCH_PLAN_PATH = REPORTS_DIR / "17_focused_search_plan.json"
AUDIT_PATH = REPORTS_DIR / "17_hyperparameter_tuning_audit.csv"
HASH_PATH = REPORTS_DIR / "17_hyperparameter_tuning_hashes.csv"
REPORT_PATH = REPORTS_DIR / "17_hyperparameter_tuning_report.txt"

FIG_COMPARE_PATH = FIGURES_DIR / "17_untuned_vs_tuned_mean_mae.png"
FIG_MAE_YEAR_PATH = FIGURES_DIR / "17_tuned_mae_by_year.png"
FIG_R2_YEAR_PATH = FIGURES_DIR / "17_tuned_r2_by_year.png"
FIG_GAP_PATH = FIGURES_DIR / "17_tuned_generalization_gap.png"


# =============================================================================
# 3. UNTUNED REFERENCE CONFIGURATIONS
# =============================================================================

UNTUNED_CONFIGS = {
    "Random Forest": {
        "n_estimators": 500,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "bootstrap": True,
        "max_samples": None,
    },
    "XGBoost": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
    },
    "LightGBM": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
    "NGBoost": {
        "n_estimators": 500,
        "learning_rate": 0.01,
        "minibatch_frac": 1.0,
        "col_sample": 1.0,
        "base_max_depth": 3,
        "base_min_samples_split": 2,
        "base_min_samples_leaf": 1,
    },
}


# =============================================================================
# 4. FOCUSED REFINEMENT CONFIGURATIONS
# =============================================================================
#
# These configurations deliberately refine around the best regions identified
# in the first Step 17 broad development search. No 2024/2025 information is
# used to define them.
#
# Each list includes the previous broad-search best configuration plus a small
# number of nearby stronger/weaker-regularization alternatives.
# =============================================================================

FOCUSED_CONFIGS = {
    "Random Forest": [
        {
            "n_estimators": 900,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_split": 5,
            "min_samples_leaf": 20,
            "bootstrap": True,
            "max_samples": 0.70,
        },
        {
            "n_estimators": 900,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_split": 5,
            "min_samples_leaf": 30,
            "bootstrap": True,
            "max_samples": 0.70,
        },
        {
            "n_estimators": 700,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_split": 10,
            "min_samples_leaf": 40,
            "bootstrap": True,
            "max_samples": 0.60,
        },
        {
            "n_estimators": 700,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_split": 10,
            "min_samples_leaf": 30,
            "bootstrap": True,
            "max_samples": 0.60,
        },
        {
            "n_estimators": 700,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_split": 10,
            "min_samples_leaf": 20,
            "bootstrap": True,
            "max_samples": 0.60,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_split": 20,
            "min_samples_leaf": 30,
            "bootstrap": True,
            "max_samples": 0.50,
        },
    ],
    "XGBoost": [
        {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_child_weight": 12,
            "subsample": 0.75,
            "colsample_bytree": 0.65,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "gamma": 0.0,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "max_depth": 1,
            "min_child_weight": 12,
            "subsample": 0.75,
            "colsample_bytree": 0.65,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "gamma": 0.0,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_child_weight": 16,
            "subsample": 0.75,
            "colsample_bytree": 0.65,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "gamma": 0.0,
        },
        {
            "n_estimators": 250,
            "learning_rate": 0.02,
            "max_depth": 2,
            "min_child_weight": 20,
            "subsample": 0.70,
            "colsample_bytree": 0.60,
            "reg_alpha": 0.5,
            "reg_lambda": 10.0,
            "gamma": 0.0,
        },
        {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 2,
            "min_child_weight": 16,
            "subsample": 0.75,
            "colsample_bytree": 0.65,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "gamma": 0.05,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.02,
            "max_depth": 3,
            "min_child_weight": 16,
            "subsample": 0.70,
            "colsample_bytree": 0.60,
            "reg_alpha": 1.0,
            "reg_lambda": 10.0,
            "gamma": 0.05,
        },
    ],
    "LightGBM": [
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 4,
            "min_child_samples": 20,
            "subsample": 0.60,
            "colsample_bytree": 0.50,
            "reg_alpha": 0.5,
            "reg_lambda": 3.0,
        },
        {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "max_depth": 3,
            "min_child_samples": 20,
            "subsample": 0.60,
            "colsample_bytree": 0.50,
            "reg_alpha": 0.5,
            "reg_lambda": 3.0,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "max_depth": 3,
            "min_child_samples": 40,
            "subsample": 0.60,
            "colsample_bytree": 0.50,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
        },
        {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 40,
            "subsample": 0.50,
            "colsample_bytree": 0.40,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
        },
        {
            "n_estimators": 250,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 4,
            "min_child_samples": 40,
            "subsample": 0.60,
            "colsample_bytree": 0.40,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 5,
            "min_child_samples": 40,
            "subsample": 0.60,
            "colsample_bytree": 0.50,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
        },
    ],
    "NGBoost": [
        {
            "n_estimators": 250,
            "learning_rate": 0.03,
            "minibatch_frac": 1.0,
            "col_sample": 1.0,
            "base_max_depth": 2,
            "base_min_samples_split": 10,
            "base_min_samples_leaf": 3,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "minibatch_frac": 1.0,
            "col_sample": 1.0,
            "base_max_depth": 1,
            "base_min_samples_split": 10,
            "base_min_samples_leaf": 3,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "minibatch_frac": 1.0,
            "col_sample": 1.0,
            "base_max_depth": 2,
            "base_min_samples_split": 10,
            "base_min_samples_leaf": 5,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.02,
            "minibatch_frac": 1.0,
            "col_sample": 1.0,
            "base_max_depth": 2,
            "base_min_samples_split": 20,
            "base_min_samples_leaf": 10,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.02,
            "minibatch_frac": 1.0,
            "col_sample": 1.0,
            "base_max_depth": 1,
            "base_min_samples_split": 10,
            "base_min_samples_leaf": 5,
        },
        {
            "n_estimators": 150,
            "learning_rate": 0.03,
            "minibatch_frac": 1.0,
            "col_sample": 1.0,
            "base_max_depth": 2,
            "base_min_samples_split": 20,
            "base_min_samples_leaf": 5,
        },
    ],
}


# =============================================================================
# 5. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 126)
    print(title)
    print("=" * 126)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def py_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def clean_params(params: dict) -> dict:
    return {str(k): py_value(v) for k, v in params.items()}


def params_to_json(params: dict) -> str:
    return json.dumps(clean_params(params), sort_keys=True, allow_nan=False)


def unique_configs(configs):
    result = []
    seen = set()

    for params in configs:
        params = clean_params(params)
        key = params_to_json(params)

        if key not in seen:
            result.append(params)
            seen.add(key)

    return result


def mape_percent(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.abs(y_true) > 1e-12

    if not np.any(valid):
        return np.nan

    return float(
        np.mean(
            np.abs(
                (y_true[valid] - y_pred[valid])
                / y_true[valid]
            )
        )
        * 100.0
    )


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": mape_percent(y_true, y_pred),
    }


def build_model(model_name: str, params: dict):
    p = clean_params(params)

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
            max_depth=int(p["base_max_depth"]),
            min_samples_split=int(p["base_min_samples_split"]),
            min_samples_leaf=int(p["base_min_samples_leaf"]),
            random_state=RANDOM_STATE,
        )

        return NGBRegressor(
            Dist=Normal,
            Score=LogScore,
            Base=base,
            natural_gradient=True,
            n_estimators=int(p["n_estimators"]),
            learning_rate=float(p["learning_rate"]),
            minibatch_frac=float(p["minibatch_frac"]),
            col_sample=float(p["col_sample"]),
            verbose=False,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# 6. LOAD DEVELOPMENT DATA ONLY
# =============================================================================

section("STEP 17 - CORRECTED + FOCUSED HYPERPARAMETER TUNING")

for required in [
    INPUT_PATH,
    STEP16_SUMMARY_PATH,
    STEP16_AUDIT_PATH,
]:
    if not required.exists():
        print(f"\nERROR: required file is missing: {required}")
        sys.exit(1)

step16_audit = pd.read_csv(STEP16_AUDIT_PATH)

if (
    "status" not in step16_audit.columns
    or (step16_audit["status"] != "PASS").any()
):
    print("\nERROR: Step 16 audit is not fully PASS. Stop Step 17.")
    sys.exit(1)

df = pd.read_csv(INPUT_PATH, dtype={"grid_id": str})
step16_summary = pd.read_csv(STEP16_SUMMARY_PATH)

required_cols = {TARGET, YEAR_COL, *ID_COLS}
missing_cols = sorted(required_cols - set(df.columns))

if missing_cols:
    print(f"\nERROR: missing required columns: {missing_cols}")
    sys.exit(1)

years = sorted(df[YEAR_COL].astype(int).unique().tolist())

if years != DEVELOPMENT_YEARS:
    print(
        f"\nERROR: input years are {years}; "
        f"expected {DEVELOPMENT_YEARS}."
    )
    sys.exit(1)

predictors = [
    c
    for c in df.columns
    if c not in {TARGET, "grid_id", "area_name"}
]

if len(predictors) != EXPECTED_PREDICTOR_COUNT:
    print(
        f"\nERROR: predictor count is {len(predictors)}; "
        f"expected {EXPECTED_PREDICTOR_COUNT}."
    )
    sys.exit(1)

# IMPORTANT CORRECTION:
# Keep the natural pandas numeric representation from the model-ready CSV.
# Do not convert the full predictor matrix to float32.
X_df = df[predictors].copy()

for col in predictors:
    X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

missing_predictor_cells = int(X_df.isna().sum().sum())

finite_predictors = bool(
    np.isfinite(
        X_df.to_numpy(dtype=np.float64)
    ).all()
)

if missing_predictor_cells != 0 or not finite_predictors:
    print("\nERROR: predictors contain missing or non-finite values.")
    sys.exit(1)

if df[TARGET].isna().any():
    print("\nERROR: target contains missing values.")
    sys.exit(1)

y = df[TARGET].to_numpy(dtype=np.float64)

grid_ids = df["grid_id"].astype(str).to_numpy()
area_names = df["area_name"].astype(str).to_numpy()
row_years = df[YEAR_COL].astype(int).to_numpy()

print(f"Project root       : {PROJECT_ROOT}")
print(f"Input              : {INPUT_PATH}")
print(f"Rows               : {len(df):,}")
print(f"Columns            : {len(df.columns)}")
print(f"Unique grids       : {df['grid_id'].nunique():,}")
print(f"Years              : {years}")
print(f"Predictors         : {len(predictors)}")
print(f"Missing predictors : {missing_predictor_cells}")
print(f"Finite predictors  : {finite_predictors}")
print("Input representation: pandas/DataFrame numeric dtypes preserved")
print("2015-2017          : historical context only")
print("2024 data          : NOT LOADED")
print("2025 data          : NOT LOADED")
print("Random K-fold      : NOT USED")
print("Primary objective  : mean rolling-origin validation MAE")


# =============================================================================
# 7. STRICT ROLLING-ORIGIN FOLDS
# =============================================================================

section("1. STRICT ROLLING-ORIGIN FOLDS")

folds = []

for validation_year in VALIDATION_YEARS:
    train_mask = df[YEAR_COL] < validation_year
    val_mask = df[YEAR_COL] == validation_year

    train_years = sorted(
        df.loc[train_mask, YEAR_COL].astype(int).unique().tolist()
    )

    expected_train_years = list(
        range(DEVELOPMENT_YEARS[0], validation_year)
    )

    if train_years != expected_train_years:
        print(
            f"\nERROR: validation {validation_year}: "
            f"train years {train_years}, "
            f"expected {expected_train_years}."
        )
        sys.exit(1)

    train_idx = np.flatnonzero(train_mask.to_numpy())
    val_idx = np.flatnonzero(val_mask.to_numpy())

    if len(train_idx) == 0 or len(val_idx) == 0:
        print(f"\nERROR: empty fold for {validation_year}.")
        sys.exit(1)

    folds.append(
        {
            "validation_year": validation_year,
            "train_years": train_years,
            "train_idx": train_idx,
            "val_idx": val_idx,
        }
    )

    print(
        f"{train_years[0]}-{train_years[-1]} "
        f"({len(train_idx):,}) -> "
        f"{validation_year} ({len(val_idx):,})"
    )

print("PASS: strict expanding-window folds created.")


# =============================================================================
# 8. CONFIGURATION EVALUATION
# =============================================================================

def evaluate_configuration(
    model_name: str,
    params: dict,
    trial_id: int,
    search_stage: str,
) -> dict:
    fold_rows = []
    start = time.perf_counter()

    for fold in folds:
        tr = fold["train_idx"]
        va = fold["val_idx"]

        # Preserve DataFrame representation for consistency with the
        # original candidate-model stages.
        X_train = X_df.iloc[tr]
        X_val = X_df.iloc[va]

        model = build_model(model_name, params)
        model.fit(X_train, y[tr])

        pred_train = np.asarray(
            model.predict(X_train),
            dtype=float,
        )

        pred_val = np.asarray(
            model.predict(X_val),
            dtype=float,
        )

        train_m = regression_metrics(y[tr], pred_train)
        val_m = regression_metrics(y[va], pred_val)

        fold_rows.append(
            {
                "validation_year": fold["validation_year"],
                "train_mae": train_m["mae"],
                "validation_mae": val_m["mae"],
                "train_rmse": train_m["rmse"],
                "validation_rmse": val_m["rmse"],
                "train_r2": train_m["r2"],
                "validation_r2": val_m["r2"],
                "train_mape_percent": train_m["mape_percent"],
                "validation_mape_percent": val_m["mape_percent"],
                "mae_gap": val_m["mae"] - train_m["mae"],
            }
        )

        del model
        gc.collect()

    fold_df = pd.DataFrame(fold_rows)

    return {
        "model": model_name,
        "trial_id": trial_id,
        "search_stage": search_stage,
        "is_untuned_reference": search_stage == "untuned_reference",
        "params_json": params_to_json(params),
        "mean_train_mae": float(fold_df["train_mae"].mean()),
        "mean_validation_mae": float(fold_df["validation_mae"].mean()),
        "median_validation_mae": float(
            fold_df["validation_mae"].median()
        ),
        "std_validation_mae": float(
            fold_df["validation_mae"].std(ddof=1)
        ),
        "best_validation_mae": float(
            fold_df["validation_mae"].min()
        ),
        "worst_validation_mae": float(
            fold_df["validation_mae"].max()
        ),
        "mean_validation_rmse": float(
            fold_df["validation_rmse"].mean()
        ),
        "mean_validation_r2": float(
            fold_df["validation_r2"].mean()
        ),
        "median_validation_r2": float(
            fold_df["validation_r2"].median()
        ),
        "minimum_validation_r2": float(
            fold_df["validation_r2"].min()
        ),
        "negative_r2_folds": int(
            (fold_df["validation_r2"] < 0).sum()
        ),
        "mean_validation_mape_percent": float(
            fold_df["validation_mape_percent"].mean()
        ),
        "mean_mae_generalization_gap": float(
            fold_df["mae_gap"].mean()
        ),
        "runtime_seconds": float(time.perf_counter() - start),
        "status": "OK",
        "error": "",
    }


# =============================================================================
# 9. PRE-TUNING UNTUNED-REFERENCE REPRODUCTION CHECK
# =============================================================================

section("2. PRE-TUNING REFERENCE REPRODUCTION CHECK")

required_step16_cols = {
    "model",
    "mean_validation_mae",
    "mean_validation_rmse",
    "mean_validation_r2",
}

missing_step16_cols = sorted(
    required_step16_cols - set(step16_summary.columns)
)

if missing_step16_cols:
    print(
        "\nERROR: Step 16 summary missing columns: "
        f"{missing_step16_cols}"
    )
    sys.exit(1)

reference_rows = []
reference_results = {}

for model_name in [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "NGBoost",
]:
    print(f"\nReproducing untuned reference: {model_name}")

    current = evaluate_configuration(
        model_name=model_name,
        params=UNTUNED_CONFIGS[model_name],
        trial_id=0,
        search_stage="untuned_reference",
    )

    step16_match = step16_summary[
        step16_summary["model"] == model_name
    ]

    if len(step16_match) != 1:
        print(
            f"\nERROR: expected exactly one Step 16 row for {model_name}."
        )
        sys.exit(1)

    ref = step16_match.iloc[0]

    mae_diff = (
        current["mean_validation_mae"]
        - float(ref["mean_validation_mae"])
    )
    rmse_diff = (
        current["mean_validation_rmse"]
        - float(ref["mean_validation_rmse"])
    )
    r2_diff = (
        current["mean_validation_r2"]
        - float(ref["mean_validation_r2"])
    )

    passed = (
        abs(mae_diff) <= REPRO_TOLERANCE
        and abs(rmse_diff) <= REPRO_TOLERANCE
        and abs(r2_diff) <= REPRO_TOLERANCE
    )

    reference_rows.append(
        {
            "model": model_name,
            "step16_mean_mae": float(ref["mean_validation_mae"]),
            "rerun_mean_mae": current["mean_validation_mae"],
            "mae_difference": mae_diff,
            "step16_mean_rmse": float(ref["mean_validation_rmse"]),
            "rerun_mean_rmse": current["mean_validation_rmse"],
            "rmse_difference": rmse_diff,
            "step16_mean_r2": float(ref["mean_validation_r2"]),
            "rerun_mean_r2": current["mean_validation_r2"],
            "r2_difference": r2_diff,
            "tolerance": REPRO_TOLERANCE,
            "status": "PASS" if passed else "FAIL",
        }
    )

    reference_results[model_name] = current

    print(
        f"Step16 MAE={float(ref['mean_validation_mae']):.9f} | "
        f"rerun MAE={current['mean_validation_mae']:.9f} | "
        f"diff={mae_diff:+.9f}"
    )
    print(
        f"Step16 RMSE={float(ref['mean_validation_rmse']):.9f} | "
        f"rerun RMSE={current['mean_validation_rmse']:.9f} | "
        f"diff={rmse_diff:+.9f}"
    )
    print(
        f"Step16 R²={float(ref['mean_validation_r2']):.9f} | "
        f"rerun R²={current['mean_validation_r2']:.9f} | "
        f"diff={r2_diff:+.9f}"
    )
    print(f"Reproduction status: {'PASS' if passed else 'FAIL'}")

reference_check_df = pd.DataFrame(reference_rows)

reference_check_df.to_csv(
    REFERENCE_CHECK_PATH,
    index=False,
)

print()
print(reference_check_df.to_string(index=False))

if (reference_check_df["status"] != "PASS").any():
    print(
        "\nERROR: at least one untuned reference does not reproduce "
        "Step 16 within tolerance."
    )
    print(
        "Tuning has NOT started. Fix the reproducibility mismatch first."
    )
    sys.exit(1)

print(
    "\nPASS: all four untuned references reproduce Step 16. "
    "Focused tuning may proceed."
)


# =============================================================================
# 10. SAVE FOCUSED SEARCH PLAN
# =============================================================================

with SEARCH_PLAN_PATH.open("w", encoding="utf-8") as f:
    json.dump(
        {
            "step": 17,
            "version": "corrected_focused_v2",
            "development_years": DEVELOPMENT_YEARS,
            "rolling_validation_years": VALIDATION_YEARS,
            "2024_used": False,
            "2025_used": False,
            "primary_objective": "mean rolling-origin validation MAE",
            "repro_tolerance": REPRO_TOLERANCE,
            "focused_configs": FOCUSED_CONFIGS,
        },
        f,
        indent=2,
        sort_keys=True,
    )


# =============================================================================
# 11. FOCUSED TUNING
# =============================================================================

def tune_one_model(model_name: str) -> pd.DataFrame:
    section(f"3. FOCUSED TUNING - {model_name.upper()}")

    # Trial 0 is reused from the successful reproduction check, so it is not
    # redundantly refit a second time.
    rows = [reference_results[model_name].copy()]

    candidate_configs = unique_configs(
        FOCUSED_CONFIGS[model_name]
    )

    # Remove exact duplicate of untuned reference if one happens to exist.
    untuned_key = params_to_json(
        UNTUNED_CONFIGS[model_name]
    )

    candidate_configs = [
        cfg
        for cfg in candidate_configs
        if params_to_json(cfg) != untuned_key
    ]

    print(
        f"Trial 0: reproduced untuned reference\n"
        f"Focused configurations to evaluate: {len(candidate_configs)}"
    )

    for offset, params in enumerate(candidate_configs, start=1):
        trial_id = offset

        print(
            f"\n{model_name} | focused trial "
            f"{offset}/{len(candidate_configs)}"
        )
        print(params_to_json(params))

        try:
            result = evaluate_configuration(
                model_name=model_name,
                params=params,
                trial_id=trial_id,
                search_stage="focused_refinement",
            )

            print(
                f"mean MAE={result['mean_validation_mae']:.4f} °C | "
                f"worst MAE={result['worst_validation_mae']:.4f} °C | "
                f"mean RMSE={result['mean_validation_rmse']:.4f} °C | "
                f"mean R²={result['mean_validation_r2']:.4f} | "
                f"negative R² folds={result['negative_r2_folds']} | "
                f"MAE gap={result['mean_mae_generalization_gap']:.4f} °C | "
                f"time={result['runtime_seconds']:.1f}s"
            )

        except Exception as exc:
            result = {
                "model": model_name,
                "trial_id": trial_id,
                "search_stage": "focused_refinement",
                "is_untuned_reference": False,
                "params_json": params_to_json(params),
                "mean_train_mae": np.nan,
                "mean_validation_mae": np.nan,
                "median_validation_mae": np.nan,
                "std_validation_mae": np.nan,
                "best_validation_mae": np.nan,
                "worst_validation_mae": np.nan,
                "mean_validation_rmse": np.nan,
                "mean_validation_r2": np.nan,
                "median_validation_r2": np.nan,
                "minimum_validation_r2": np.nan,
                "negative_r2_folds": np.nan,
                "mean_validation_mape_percent": np.nan,
                "mean_mae_generalization_gap": np.nan,
                "runtime_seconds": np.nan,
                "status": "FAILED",
                "error": repr(exc),
            }

            print(f"FAILED: {repr(exc)}")

        rows.append(result)

        # Checkpoint after every trial.
        pd.DataFrame(rows).to_csv(
            SEARCH_PATHS[model_name],
            index=False,
        )

    results = pd.DataFrame(rows)

    successful = results[
        results["status"] == "OK"
    ].copy()

    if successful.empty:
        print(f"\nERROR: all {model_name} trials failed.")
        sys.exit(1)

    successful = successful.sort_values(
        [
            "mean_validation_mae",  # primary objective
            "worst_validation_mae",
            "std_validation_mae",
            "mean_validation_rmse",
            "negative_r2_folds",
            "mean_mae_generalization_gap",
            "trial_id",
        ],
        ascending=True,
    ).reset_index(drop=True)

    best = successful.iloc[0]

    print(f"\nBEST DEVELOPMENT CONFIG - {model_name}")
    print(best["params_json"])
    print(f"Search stage         : {best['search_stage']}")
    print(f"Mean validation MAE : {best['mean_validation_mae']:.6f} °C")
    print(f"Mean validation RMSE: {best['mean_validation_rmse']:.6f} °C")
    print(f"Mean validation R²  : {best['mean_validation_r2']:.6f}")
    print(f"Worst-year MAE      : {best['worst_validation_mae']:.6f} °C")
    print(f"Negative R² folds   : {int(best['negative_r2_folds'])}")
    print(
        f"Mean MAE gap        : "
        f"{best['mean_mae_generalization_gap']:.6f} °C"
    )

    return results


all_search_results = {}
best_params = {}

for model_name in [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "NGBoost",
]:
    results = tune_one_model(model_name)

    all_search_results[model_name] = results

    successful = results[
        results["status"] == "OK"
    ].copy()

    successful = successful.sort_values(
        [
            "mean_validation_mae",
            "worst_validation_mae",
            "std_validation_mae",
            "mean_validation_rmse",
            "negative_r2_folds",
            "mean_mae_generalization_gap",
            "trial_id",
        ],
        ascending=True,
    )

    best_params[model_name] = json.loads(
        successful.iloc[0]["params_json"]
    )


# =============================================================================
# 12. RE-EVALUATE BEST CONFIGURATIONS AND SAVE FOLD PREDICTIONS
# =============================================================================

section("4. RE-EVALUATE BEST TUNED CONFIGURATIONS")

best_fold_rows = []
prediction_frames = []

for model_name, params in best_params.items():
    print(f"\nBest tuned {model_name}")
    print(params_to_json(params))

    for fold in folds:
        tr = fold["train_idx"]
        va = fold["val_idx"]

        X_train = X_df.iloc[tr]
        X_val = X_df.iloc[va]

        model = build_model(model_name, params)
        model.fit(X_train, y[tr])

        pred_train = np.asarray(
            model.predict(X_train),
            dtype=float,
        )

        pred_val = np.asarray(
            model.predict(X_val),
            dtype=float,
        )

        train_m = regression_metrics(y[tr], pred_train)
        val_m = regression_metrics(y[va], pred_val)

        best_fold_rows.append(
            {
                "model": model_name,
                "validation_year": fold["validation_year"],
                "train_start_year": min(fold["train_years"]),
                "train_end_year": max(fold["train_years"]),
                "train_rows": len(tr),
                "validation_rows": len(va),
                "train_mae": train_m["mae"],
                "validation_mae": val_m["mae"],
                "mae_generalization_gap": (
                    val_m["mae"] - train_m["mae"]
                ),
                "train_rmse": train_m["rmse"],
                "validation_rmse": val_m["rmse"],
                "rmse_generalization_gap": (
                    val_m["rmse"] - train_m["rmse"]
                ),
                "train_r2": train_m["r2"],
                "validation_r2": val_m["r2"],
                "r2_generalization_gap": (
                    train_m["r2"] - val_m["r2"]
                ),
                "train_mape_percent": train_m["mape_percent"],
                "validation_mape_percent": val_m["mape_percent"],
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "grid_id": grid_ids[va],
                    "area_name": area_names[va],
                    "validation_year": row_years[va],
                    "actual_lst_c": y[va],
                    "predicted_lst_c": pred_val,
                    "residual_actual_minus_predicted": (
                        y[va] - pred_val
                    ),
                }
            )
        )

        print(
            f"{fold['train_years'][0]}-{fold['train_years'][-1]} "
            f"-> {fold['validation_year']} | "
            f"MAE={val_m['mae']:.4f} | "
            f"RMSE={val_m['rmse']:.4f} | "
            f"R²={val_m['r2']:.4f}"
        )

        del model
        gc.collect()

best_fold_metrics = pd.DataFrame(best_fold_rows)

best_fold_metrics.to_csv(
    BEST_FOLD_METRICS_PATH,
    index=False,
)

all_predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

all_predictions.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# =============================================================================
# 13. TUNED DEVELOPMENT SUMMARY
# =============================================================================

section("5. TUNED DEVELOPMENT SUMMARY")

tuned_summary = (
    best_fold_metrics.groupby("model")
    .agg(
        folds=("validation_year", "count"),
        mean_train_mae=("train_mae", "mean"),
        mean_validation_mae=("validation_mae", "mean"),
        median_validation_mae=("validation_mae", "median"),
        std_validation_mae=("validation_mae", "std"),
        best_year_mae=("validation_mae", "min"),
        worst_year_mae=("validation_mae", "max"),
        mean_validation_rmse=("validation_rmse", "mean"),
        mean_validation_r2=("validation_r2", "mean"),
        median_validation_r2=("validation_r2", "median"),
        minimum_validation_r2=("validation_r2", "min"),
        mean_validation_mape_percent=(
            "validation_mape_percent",
            "mean",
        ),
        mean_mae_generalization_gap=(
            "mae_generalization_gap",
            "mean",
        ),
    )
    .reset_index()
)

negative_counts = (
    best_fold_metrics.assign(
        negative_r2=(
            best_fold_metrics["validation_r2"] < 0
        ).astype(int)
    )
    .groupby("model")["negative_r2"]
    .sum()
    .rename("negative_r2_folds")
    .reset_index()
)

tuned_summary = tuned_summary.merge(
    negative_counts,
    on="model",
    how="left",
)

tuned_summary[
    "development_tuned_mean_mae_rank"
] = (
    tuned_summary["mean_validation_mae"]
    .rank(method="min", ascending=True)
    .astype(int)
)

tuned_summary = tuned_summary.sort_values(
    ["development_tuned_mean_mae_rank", "model"]
).reset_index(drop=True)

tuned_summary.to_csv(
    TUNED_SUMMARY_PATH,
    index=False,
)

print(tuned_summary.to_string(index=False))

print(
    "\nIMPORTANT: these are tuned DEVELOPMENT candidates only. "
    "No final model is selected."
)


# =============================================================================
# 14. TUNED VS UNTUNED
# =============================================================================

section("6. TUNED VS UNTUNED")

untuned_ref = step16_summary[
    [
        "model",
        "mean_validation_mae",
        "mean_validation_rmse",
        "mean_validation_r2",
        "negative_r2_folds",
        "mean_mae_generalization_gap",
    ]
].rename(
    columns={
        "mean_validation_mae": "untuned_mean_validation_mae",
        "mean_validation_rmse": "untuned_mean_validation_rmse",
        "mean_validation_r2": "untuned_mean_validation_r2",
        "negative_r2_folds": "untuned_negative_r2_folds",
        "mean_mae_generalization_gap": "untuned_mean_mae_gap",
    }
)

comparison = tuned_summary.merge(
    untuned_ref,
    on="model",
    how="left",
)

comparison = comparison.rename(
    columns={
        "mean_validation_mae": "tuned_mean_validation_mae",
        "mean_validation_rmse": "tuned_mean_validation_rmse",
        "mean_validation_r2": "tuned_mean_validation_r2",
        "negative_r2_folds": "tuned_negative_r2_folds",
        "mean_mae_generalization_gap": "tuned_mean_mae_gap",
    }
)

comparison[
    "mae_improvement_untuned_minus_tuned"
] = (
    comparison["untuned_mean_validation_mae"]
    - comparison["tuned_mean_validation_mae"]
)

comparison[
    "rmse_improvement_untuned_minus_tuned"
] = (
    comparison["untuned_mean_validation_rmse"]
    - comparison["tuned_mean_validation_rmse"]
)

comparison[
    "r2_change_tuned_minus_untuned"
] = (
    comparison["tuned_mean_validation_r2"]
    - comparison["untuned_mean_validation_r2"]
)

comparison[
    "mae_gap_change_tuned_minus_untuned"
] = (
    comparison["tuned_mean_mae_gap"]
    - comparison["untuned_mean_mae_gap"]
)

comparison.to_csv(
    TUNED_VS_UNTUNED_PATH,
    index=False,
)

print(
    comparison[
        [
            "model",
            "untuned_mean_validation_mae",
            "tuned_mean_validation_mae",
            "mae_improvement_untuned_minus_tuned",
            "untuned_mean_validation_rmse",
            "tuned_mean_validation_rmse",
            "untuned_mean_validation_r2",
            "tuned_mean_validation_r2",
            "untuned_negative_r2_folds",
            "tuned_negative_r2_folds",
            "untuned_mean_mae_gap",
            "tuned_mean_mae_gap",
        ]
    ].to_string(index=False)
)


# =============================================================================
# 15. FIT FULL 2018-2023 TUNED CANDIDATES
# =============================================================================

section("7. FIT FULL 2018-2023 TUNED CANDIDATES")

MODEL_PATHS = {
    "Random Forest": (
        TUNED_MODELS_DIR / "17_random_forest_tuned_2018_2023.joblib"
    ),
    "XGBoost": (
        TUNED_MODELS_DIR / "17_xgboost_tuned_2018_2023.joblib"
    ),
    "LightGBM": (
        TUNED_MODELS_DIR / "17_lightgbm_tuned_2018_2023.joblib"
    ),
    "NGBoost": (
        TUNED_MODELS_DIR / "17_ngboost_tuned_2018_2023.joblib"
    ),
}

for model_name, params in best_params.items():
    model = build_model(model_name, params)

    model.fit(
        X_df,
        y,
    )

    artifact = {
        "model_name": model_name,
        "model": model,
        "predictors": predictors,
        "target": TARGET,
        "training_years": DEVELOPMENT_YEARS,
        "best_params": clean_params(params),
        "tuning_objective": (
            "mean rolling-origin validation MAE"
        ),
        "tuning_version": "corrected_focused_v2",
        "reserved_2024_used": False,
        "lockbox_2025_used": False,
        "step": 17,
    }

    joblib.dump(
        artifact,
        MODEL_PATHS[model_name],
    )

    print(f"Saved: {MODEL_PATHS[model_name]}")

    del model
    gc.collect()


# =============================================================================
# 16. SAVE BEST HYPERPARAMETERS
# =============================================================================

with BEST_PARAMS_PATH.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "step": 17,
            "version": "corrected_focused_v2",
            "development_years": DEVELOPMENT_YEARS,
            "rolling_validation_years": VALIDATION_YEARS,
            "primary_objective": (
                "mean rolling-origin validation MAE"
            ),
            "reference_reproduction_tolerance": REPRO_TOLERANCE,
            "2024_used": False,
            "2025_used": False,
            "best_params": {
                model_name: clean_params(params)
                for model_name, params in best_params.items()
            },
        },
        f,
        indent=2,
        sort_keys=True,
    )

print(f"Saved: {BEST_PARAMS_PATH}")


# =============================================================================
# 17. FIGURES
# =============================================================================

section("8. FIGURES")

plot_compare = comparison.sort_values(
    "tuned_mean_validation_mae",
    ascending=True,
)

x = np.arange(len(plot_compare))
width = 0.36

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    x - width / 2,
    plot_compare["untuned_mean_validation_mae"],
    width,
    label="Untuned",
)

ax.bar(
    x + width / 2,
    plot_compare["tuned_mean_validation_mae"],
    width,
    label="Tuned",
)

ax.set_xticks(x, plot_compare["model"])
ax.set_ylabel("Mean rolling validation MAE (°C)")
ax.set_title("Untuned vs Corrected/Refined Tuned MAE")
ax.grid(axis="y", alpha=0.2)
ax.legend()

plt.tight_layout()
plt.savefig(
    FIG_COMPARE_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 6))

for model_name in best_params:
    md = best_fold_metrics[
        best_fold_metrics["model"] == model_name
    ].sort_values("validation_year")

    ax.plot(
        md["validation_year"].astype(str),
        md["validation_mae"],
        marker="o",
        label=model_name,
    )

ax.set_title("Tuned Candidate MAE by Rolling Validation Year")
ax.set_xlabel("Validation Year")
ax.set_ylabel("MAE (°C)")
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()
plt.savefig(
    FIG_MAE_YEAR_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 6))

for model_name in best_params:
    md = best_fold_metrics[
        best_fold_metrics["model"] == model_name
    ].sort_values("validation_year")

    ax.plot(
        md["validation_year"].astype(str),
        md["validation_r2"],
        marker="o",
        label=model_name,
    )

ax.axhline(0, linewidth=1)
ax.set_title("Tuned Candidate R² by Rolling Validation Year")
ax.set_xlabel("Validation Year")
ax.set_ylabel("R²")
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()
plt.savefig(
    FIG_R2_YEAR_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


gap_plot = tuned_summary.sort_values(
    "mean_mae_generalization_gap",
    ascending=True,
)

fig, ax = plt.subplots(figsize=(9, 6))

ax.barh(
    gap_plot["model"],
    gap_plot["mean_mae_generalization_gap"],
)

ax.set_title("Tuned Candidate Mean MAE Generalization Gap")
ax.set_xlabel("Validation MAE - Train MAE (°C)")
ax.set_ylabel("Model")
ax.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig(
    FIG_GAP_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print(f"Saved: {FIG_COMPARE_PATH}")
print(f"Saved: {FIG_MAE_YEAR_PATH}")
print(f"Saved: {FIG_R2_YEAR_PATH}")
print(f"Saved: {FIG_GAP_PATH}")


# =============================================================================
# 18. STRICT AUDIT
# =============================================================================

section("9. STRICT STEP 17 AUDIT")

search_status_rows = []

for model_name, results in all_search_results.items():
    search_status_rows.append(
        {
            "model": model_name,
            "total_trials_including_reference": int(
                len(results)
            ),
            "successful_trials": int(
                (results["status"] == "OK").sum()
            ),
            "failed_trials": int(
                (results["status"] != "OK").sum()
            ),
            "untuned_reference_present": bool(
                results[
                    "is_untuned_reference"
                ].astype(bool).any()
            ),
            "focused_trials": int(
                (
                    results["search_stage"]
                    == "focused_refinement"
                ).sum()
            ),
        }
    )

search_status = pd.DataFrame(
    search_status_rows
)

best_fold_years = sorted(
    best_fold_metrics[
        "validation_year"
    ]
    .astype(int)
    .unique()
    .tolist()
)

expected_best_fold_rows = (
    4 * len(VALIDATION_YEARS)
)

all_searches_success = bool(
    (
        search_status["successful_trials"]
        > 0
    ).all()
)

all_reference_present = bool(
    search_status[
        "untuned_reference_present"
    ].all()
)

reference_reproduction_all_pass = bool(
    (
        reference_check_df["status"]
        == "PASS"
    ).all()
)

all_best_metrics_finite = bool(
    np.isfinite(
        best_fold_metrics[
            [
                "train_mae",
                "validation_mae",
                "train_rmse",
                "validation_rmse",
                "train_r2",
                "validation_r2",
            ]
        ].to_numpy(dtype=float)
    ).all()
)

audit_rows = [
    {
        "check": "development_input_years",
        "status": (
            "PASS"
            if years == DEVELOPMENT_YEARS
            else "FAIL"
        ),
        "details": str(years),
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
        "check": "input_representation",
        "status": "PASS",
        "details": (
            "Pandas/DataFrame numeric representation preserved; "
            "no global float32 conversion."
        ),
    },
    {
        "check": "untuned_reference_reproduction",
        "status": (
            "PASS"
            if reference_reproduction_all_pass
            else "FAIL"
        ),
        "details": (
            f"all four models within tolerance "
            f"{REPRO_TOLERANCE}"
        ),
    },
    {
        "check": "2024_loaded_or_used",
        "status": "PASS",
        "details": (
            "No. Only the 2018-2023 development input was loaded."
        ),
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": (
            "No. Only the 2018-2023 development input was loaded."
        ),
    },
    {
        "check": "random_kfold_used",
        "status": "PASS",
        "details": (
            "No. Strict expanding rolling-origin folds only."
        ),
    },
    {
        "check": "rolling_validation_years",
        "status": (
            "PASS"
            if best_fold_years
            == VALIDATION_YEARS
            else "FAIL"
        ),
        "details": str(best_fold_years),
    },
    {
        "check": "best_fold_metric_rows",
        "status": (
            "PASS"
            if len(best_fold_metrics)
            == expected_best_fold_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(best_fold_metrics)}, "
            f"expected={expected_best_fold_rows}"
        ),
    },
    {
        "check": "all_searches_have_success",
        "status": (
            "PASS"
            if all_searches_success
            else "FAIL"
        ),
        "details": search_status.to_json(
            orient="records"
        ),
    },
    {
        "check": "untuned_reference_in_each_search",
        "status": (
            "PASS"
            if all_reference_present
            else "FAIL"
        ),
        "details": str(all_reference_present),
    },
    {
        "check": "best_fold_metrics_finite",
        "status": (
            "PASS"
            if all_best_metrics_finite
            else "FAIL"
        ),
        "details": str(all_best_metrics_finite),
    },
    {
        "check": "early_stopping_used",
        "status": "PASS",
        "details": (
            "No. Reserved 2024 was not used for early stopping."
        ),
    },
    {
        "check": "uncertainty_calibration",
        "status": "PASS",
        "details": (
            "No uncertainty calibration performed in Step 17."
        ),
    },
    {
        "check": "final_model_selected",
        "status": "PASS",
        "details": (
            "No. Step 17 creates tuned development candidates only."
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

print(search_status.to_string(index=False))
print()
print(audit_df.to_string(index=False))

if (audit_df["status"] != "PASS").any():
    print(
        "\nERROR: one or more Step 17 audit checks failed."
    )
    sys.exit(1)


# =============================================================================
# 19. REPRODUCIBILITY HASHES
# =============================================================================

section("10. REPRODUCIBILITY HASHES")

hash_files = [
    INPUT_PATH,
    STEP16_SUMMARY_PATH,
    STEP16_AUDIT_PATH,
    REFERENCE_CHECK_PATH,
    SEARCH_PLAN_PATH,
    *SEARCH_PATHS.values(),
    BEST_FOLD_METRICS_PATH,
    TUNED_SUMMARY_PATH,
    TUNED_VS_UNTUNED_PATH,
    PREDICTIONS_PATH,
    BEST_PARAMS_PATH,
    *MODEL_PATHS.values(),
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

pd.DataFrame(
    hash_rows
).to_csv(
    HASH_PATH,
    index=False,
)


# =============================================================================
# 20. REPORT
# =============================================================================

section("11. SAVE STEP 17 REPORT")

report_lines = [
    "STEP 17 - CORRECTED + FOCUSED HYPERPARAMETER TUNING",
    "=" * 112,
    "",
    "WHY THIS VERSION WAS RUN",
    "-" * 112,
    "1. Preserve the original pandas/DataFrame numeric representation so the",
    "   untuned candidate references reproduce Step 16.",
    "2. Run a small focused refinement around promising boundary regions found",
    "   in the first broad development search.",
    "",
    "DATA POLICY",
    "-" * 112,
    "2015-2017: historical context used upstream for temporal feature creation.",
    "2018-2023: direct development/tuning rows.",
    "2024: NOT loaded; reserved validation remains untouched.",
    "2025: NOT loaded; final lockbox remains untouched.",
    "",
    "REFERENCE REPRODUCTION",
    "-" * 112,
]

for _, row in reference_check_df.iterrows():
    report_lines.append(
        f"{row['model']}: {row['status']} | "
        f"MAE diff={row['mae_difference']:+.12f}, "
        f"RMSE diff={row['rmse_difference']:+.12f}, "
        f"R² diff={row['r2_difference']:+.12f}."
    )

report_lines.extend(
    [
        "",
        "VALIDATION",
        "-" * 112,
        "Strict expanding rolling-origin folds:",
        "2018 -> 2019",
        "2018-2019 -> 2020",
        "2018-2020 -> 2021",
        "2018-2021 -> 2022",
        "2018-2022 -> 2023",
        "",
        "PRIMARY OBJECTIVE",
        "-" * 112,
        "Mean rolling-origin validation MAE.",
        "",
        "BEST DEVELOPMENT HYPERPARAMETERS",
        "-" * 112,
    ]
)

for model_name in [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "NGBoost",
]:
    report_lines.append(
        f"{model_name}: "
        f"{params_to_json(best_params[model_name])}"
    )

report_lines.extend(
    [
        "",
        "TUNED DEVELOPMENT RESULTS",
        "-" * 112,
    ]
)

for _, row in tuned_summary.iterrows():
    report_lines.append(
        f"{row['model']}: "
        f"mean MAE={row['mean_validation_mae']:.6f} °C, "
        f"mean RMSE={row['mean_validation_rmse']:.6f} °C, "
        f"mean R²={row['mean_validation_r2']:.6f}, "
        f"negative-R² folds={int(row['negative_r2_folds'])}, "
        f"worst-year MAE={row['worst_year_mae']:.6f} °C, "
        f"mean MAE gap={row['mean_mae_generalization_gap']:.6f} °C."
    )

report_lines.extend(
    [
        "",
        "INTERPRETATION",
        "-" * 112,
        "These remain tuned DEVELOPMENT candidates only.",
        "No final model has been selected.",
        "Do not use 2024 or 2025 to modify these hyperparameters.",
        "Next step: spatial validation of the tuned candidate configurations.",
    ]
)

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {REPORT_PATH}")
print(f"Saved: {AUDIT_PATH}")
print(f"Saved: {HASH_PATH}")


# =============================================================================
# 21. FINAL STATUS
# =============================================================================

section("STEP 17 COMPLETED SUCCESSFULLY")

print(
    "Corrected and focused hyperparameter tuning completed.\n"
    "All untuned references reproduced Step 16 within tolerance before tuning.\n"
    "Only 2018-2023 direct development rows were used.\n"
    "2015-2017 remained historical context only.\n"
    "Five strict expanding rolling-origin folds were used.\n"
    "2024 reserved validation remained untouched.\n"
    "2025 final lockbox remained untouched.\n"
    "No final model was selected.\n"
    "Review the complete Step 17 output before starting Step 18."
)
