
"""
STEP 18 - SPATIAL VALIDATION
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Evaluate whether the Step 17 tuned development candidates can generalize to
spatially unseen grid cells using the frozen Step 09 geometry-only folds.

PRIMARY VALIDATION
------------------
Reuse the 5 spatial folds created in Step 09.

For each fold:
    held-out fold grids = spatial validation set
    remaining 4 folds   = spatial training set

All 2018-2023 rows of a held-out grid remain completely absent from that fold's
training set.

SECONDARY SENSITIVITY
---------------------
Run a 210 m buffer sensitivity analysis. Training grids touching/intersecting
the 210 m buffered held-out region are removed before fitting. This is a stricter
diagnostic for local spatial dependence.

IMPORTANT RESEARCH RULES
------------------------
- Use only 2018-2023 development rows.
- 2024 is NOT loaded or used.
- 2025 is NOT loaded or used.
- Reuse Step 17 tuned hyperparameters exactly.
- No hyperparameter tuning in Step 18.
- No final model selection in Step 18.
- The Step 08 frozen 93-feature set is reused.
"""

from pathlib import Path
import gc
import hashlib
import json
import re
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

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

TARGET = "LST_C"
YEAR_COL = "year"
GRID_COL = "grid_id"
AREA_COL = "area_name"

DEVELOPMENT_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
EXPECTED_PREDICTOR_COUNT = 93
EXPECTED_UNIQUE_GRIDS = 2194
EXPECTED_SPATIAL_FOLDS = 5
EXPECTED_SPATIAL_CONTEXT_FEATURES = 18
EXPECTED_NBR_MEAN_FEATURES = 9
EXPECTED_NBR_CONTRAST_FEATURES = 9

EXPECTED_FOLD_COUNTS = {
    1: 484,
    2: 415,
    3: 531,
    4: 382,
    5: 382,
}

RUN_BUFFER_SENSITIVITY = True
BUFFER_DISTANCE_METERS = 210.0

MODEL_ORDER = [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "NGBoost",
]


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

SPATIAL_ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "spatial"
    / "spatial_fold_assignment.csv"
)

SPATIAL_GEOMETRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "spatial"
    / "spatial_fold_grid_utm44n.geojson"
)

STEP09_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "09_spatial_split_audit.csv"
)

STEP17_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "17_hyperparameter_tuning_audit.csv"
)

STEP17_PARAMS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "17_best_hyperparameters.json"
)

STEP17_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "17_tuned_model_summary.csv"
)

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

for folder in [METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR, FIGURES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

PRIMARY_METRICS_PATH = METRICS_DIR / "18_spatial_fold_metrics.csv"
PRIMARY_SUMMARY_PATH = METRICS_DIR / "18_spatial_validation_summary.csv"
YEAR_FOLD_METRICS_PATH = METRICS_DIR / "18_spatial_year_fold_metrics.csv"
BUFFER_METRICS_PATH = METRICS_DIR / "18_spatial_buffer_fold_metrics.csv"
BUFFER_SUMMARY_PATH = METRICS_DIR / "18_spatial_buffer_summary.csv"
SPATIAL_VS_TEMPORAL_PATH = (
    METRICS_DIR / "18_spatial_vs_step17_temporal_summary.csv"
)

PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "18_spatial_validation_predictions.csv"
)
BUFFER_PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "18_spatial_buffer_predictions.csv"
)

CONTEXT_FEATURE_AUDIT_PATH = (
    REPORTS_DIR / "18_spatial_context_feature_audit.csv"
)
BUFFER_GRID_AUDIT_PATH = REPORTS_DIR / "18_buffer_grid_audit.csv"
AUDIT_PATH = REPORTS_DIR / "18_spatial_validation_audit.csv"
REPORT_PATH = REPORTS_DIR / "18_spatial_validation_report.txt"
HASH_PATH = REPORTS_DIR / "18_spatial_validation_hashes.csv"

FIG_MEAN_MAE_PATH = FIGURES_DIR / "18_spatial_mean_mae_by_model.png"
FIG_FOLD_MAE_PATH = FIGURES_DIR / "18_spatial_mae_by_fold.png"
FIG_FOLD_R2_PATH = FIGURES_DIR / "18_spatial_r2_by_fold.png"
FIG_BUFFER_COMPARE_PATH = (
    FIGURES_DIR / "18_standard_vs_buffered_mae.png"
)


# =============================================================================
# 3. HELPERS
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
        ) * 100.0
    )


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": mape_percent(y_true, y_pred),
    }


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


def check_audit_all_pass(path: Path, label: str) -> None:
    if not path.exists():
        print(f"\nERROR: required {label} audit is missing: {path}")
        sys.exit(1)

    audit = pd.read_csv(path)

    if "status" not in audit.columns:
        print(f"\nERROR: {label} audit has no 'status' column.")
        sys.exit(1)

    if (audit["status"].astype(str).str.upper() != "PASS").any():
        print(f"\nERROR: {label} audit is not fully PASS.")
        print(audit.to_string(index=False))
        sys.exit(1)


def detect_fold_column(assignments: pd.DataFrame) -> str:
    preferred = []
    fallback = []

    for col in assignments.columns:
        if col == GRID_COL:
            continue

        s = assignments[col].dropna()
        if s.empty:
            continue

        if s.nunique() == EXPECTED_SPATIAL_FOLDS:
            if "fold" in col.lower():
                preferred.append(col)
            else:
                fallback.append(col)

    if len(preferred) == 1:
        return preferred[0]

    if len(preferred) > 1:
        print(f"\nERROR: multiple candidate fold columns found: {preferred}")
        sys.exit(1)

    if len(fallback) == 1:
        return fallback[0]

    print("\nERROR: could not uniquely identify Step 09 fold column.")
    print(assignments.columns.tolist())
    sys.exit(1)


def normalize_fold_ids(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        result = numeric.astype(int)
    else:
        extracted = series.astype(str).str.extract(r"(\d+)", expand=False)
        numeric = pd.to_numeric(extracted, errors="coerce")

        if numeric.isna().any():
            print("\nERROR: some spatial fold IDs could not be converted.")
            sys.exit(1)

        result = numeric.astype(int)

    unique_folds = sorted(result.unique().tolist())

    if unique_folds == [0, 1, 2, 3, 4]:
        result = result + 1
        unique_folds = sorted(result.unique().tolist())

    if unique_folds != [1, 2, 3, 4, 5]:
        print(
            f"\nERROR: normalized folds are {unique_folds}; "
            "expected [1, 2, 3, 4, 5]."
        )
        sys.exit(1)

    return result


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


def summarize_fold_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics_df.groupby("model")
        .agg(
            folds=("spatial_fold", "count"),
            mean_train_mae=("train_mae", "mean"),
            mean_validation_mae=("validation_mae", "mean"),
            median_validation_mae=("validation_mae", "median"),
            std_validation_mae=("validation_mae", "std"),
            best_fold_mae=("validation_mae", "min"),
            worst_fold_mae=("validation_mae", "max"),
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
            total_validation_rows=("validation_rows", "sum"),
        )
        .reset_index()
    )

    negative = (
        metrics_df.assign(
            negative_r2=(
                metrics_df["validation_r2"] < 0
            ).astype(int)
        )
        .groupby("model")["negative_r2"]
        .sum()
        .rename("negative_r2_folds")
        .reset_index()
    )

    summary = summary.merge(
        negative,
        on="model",
        how="left",
    )

    summary["spatial_mean_mae_rank"] = (
        summary["mean_validation_mae"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    return summary.sort_values(
        ["spatial_mean_mae_rank", "model"]
    ).reset_index(drop=True)


def get_union(geometries):
    try:
        return geometries.union_all()
    except AttributeError:
        return geometries.unary_union


# =============================================================================
# 4. PREREQUISITE AUDITS
# =============================================================================

section("STEP 18 - SPATIAL VALIDATION")

for required in [
    INPUT_PATH,
    SPATIAL_ASSIGNMENT_PATH,
    SPATIAL_GEOMETRY_PATH,
    STEP09_AUDIT_PATH,
    STEP17_AUDIT_PATH,
    STEP17_PARAMS_PATH,
    STEP17_SUMMARY_PATH,
]:
    if not required.exists():
        print(f"\nERROR: required file is missing: {required}")
        sys.exit(1)

check_audit_all_pass(
    STEP09_AUDIT_PATH,
    "Step 09 spatial split",
)

check_audit_all_pass(
    STEP17_AUDIT_PATH,
    "Step 17 hyperparameter tuning",
)

print("PASS: Step 09 audit is fully PASS.")
print("PASS: Step 17 audit is fully PASS.")


# =============================================================================
# 5. LOAD DEVELOPMENT DATA
# =============================================================================

section("1. LOAD 2018-2023 DEVELOPMENT DATA")

df = pd.read_csv(
    INPUT_PATH,
    dtype={GRID_COL: str},
)

required_cols = {
    TARGET,
    YEAR_COL,
    GRID_COL,
    AREA_COL,
}

missing_required = sorted(
    required_cols - set(df.columns)
)

if missing_required:
    print(f"\nERROR: missing required columns: {missing_required}")
    sys.exit(1)

years = sorted(
    df[YEAR_COL].astype(int).unique().tolist()
)

if years != DEVELOPMENT_YEARS:
    print(
        f"\nERROR: input years are {years}; "
        f"expected {DEVELOPMENT_YEARS}."
    )
    sys.exit(1)

if df[GRID_COL].nunique() != EXPECTED_UNIQUE_GRIDS:
    print(
        f"\nERROR: unique grid count is {df[GRID_COL].nunique()}; "
        f"expected {EXPECTED_UNIQUE_GRIDS}."
    )
    sys.exit(1)

predictors = [
    c
    for c in df.columns
    if c not in {
        TARGET,
        GRID_COL,
        AREA_COL,
    }
]

if len(predictors) != EXPECTED_PREDICTOR_COUNT:
    print(
        f"\nERROR: predictor count is {len(predictors)}; "
        f"expected {EXPECTED_PREDICTOR_COUNT}."
    )
    sys.exit(1)

X_df = df[predictors].copy()

for col in predictors:
    X_df[col] = pd.to_numeric(
        X_df[col],
        errors="coerce",
    )

missing_predictor_cells = int(
    X_df.isna().sum().sum()
)

finite_predictors = bool(
    np.isfinite(
        X_df.to_numpy(dtype=np.float64)
    ).all()
)

if missing_predictor_cells != 0:
    print(
        f"\nERROR: predictor matrix contains "
        f"{missing_predictor_cells} missing cells."
    )
    sys.exit(1)

if not finite_predictors:
    print("\nERROR: predictor matrix contains non-finite values.")
    sys.exit(1)

y = pd.to_numeric(
    df[TARGET],
    errors="coerce",
).to_numpy(dtype=np.float64)

if not np.isfinite(y).all():
    print("\nERROR: target contains missing/non-finite values.")
    sys.exit(1)

print(f"Rows               : {len(df):,}")
print(f"Columns            : {len(df.columns)}")
print(f"Unique grids       : {df[GRID_COL].nunique():,}")
print(f"Years              : {years}")
print(f"Predictors         : {len(predictors)}")
print(f"Missing predictors : {missing_predictor_cells}")
print(f"Finite predictors  : {finite_predictors}")
print("2024 data          : NOT LOADED")
print("2025 data          : NOT LOADED")


# =============================================================================
# 6. LOAD + VERIFY FROZEN STEP 09 SPATIAL FOLDS
# =============================================================================

section("2. LOAD + VERIFY FROZEN STEP 09 SPATIAL FOLDS")

assign = pd.read_csv(
    SPATIAL_ASSIGNMENT_PATH,
    dtype={GRID_COL: str},
)

if GRID_COL not in assign.columns:
    print(f"\nERROR: {GRID_COL} missing from spatial assignment.")
    sys.exit(1)

fold_col = detect_fold_column(assign)

assign = assign[
    [GRID_COL, fold_col]
].copy()

assign = assign.drop_duplicates(
    subset=[GRID_COL],
    keep="first",
)

assign["spatial_fold"] = normalize_fold_ids(
    assign[fold_col]
)

assign = assign[
    [GRID_COL, "spatial_fold"]
].copy()

if assign[GRID_COL].nunique() != EXPECTED_UNIQUE_GRIDS:
    print(
        f"\nERROR: Step 09 assignment has "
        f"{assign[GRID_COL].nunique()} unique grids; "
        f"expected {EXPECTED_UNIQUE_GRIDS}."
    )
    sys.exit(1)

fold_counts = (
    assign.groupby("spatial_fold")[GRID_COL]
    .nunique()
    .to_dict()
)

print(f"Detected Step 09 fold column: {fold_col}")

for fold_id in sorted(fold_counts):
    print(
        f"Fold {fold_id}: "
        f"{fold_counts[fold_id]:,} grids"
    )

if fold_counts != EXPECTED_FOLD_COUNTS:
    print("\nERROR: Step 09 fold counts do not match the finalized split.")
    print(f"Actual  : {fold_counts}")
    print(f"Expected: {EXPECTED_FOLD_COUNTS}")
    sys.exit(1)

df = df.merge(
    assign,
    on=GRID_COL,
    how="left",
    validate="many_to_one",
)

if df["spatial_fold"].isna().any():
    print("\nERROR: some development rows have no spatial fold.")
    sys.exit(1)

df["spatial_fold"] = df["spatial_fold"].astype(int)

fold_consistency = (
    df.groupby(GRID_COL)["spatial_fold"]
    .nunique()
)

if int(fold_consistency.max()) != 1:
    print("\nERROR: at least one grid changes fold across years.")
    sys.exit(1)

print("PASS: all rows mapped to frozen Step 09 folds.")
print("PASS: each grid keeps one fold across all years.")


# =============================================================================
# 7. LOAD STEP 17 TUNED CONFIGURATIONS
# =============================================================================

section("3. LOAD FIXED STEP 17 HYPERPARAMETERS")

with STEP17_PARAMS_PATH.open("r", encoding="utf-8") as f:
    params_payload = json.load(f)

if "best_params" not in params_payload:
    print("\nERROR: Step 17 JSON does not contain 'best_params'.")
    sys.exit(1)

best_params = params_payload["best_params"]

missing_models = [
    model
    for model in MODEL_ORDER
    if model not in best_params
]

if missing_models:
    print(
        f"\nERROR: missing Step 17 parameters for: {missing_models}"
    )
    sys.exit(1)

for model_name in MODEL_ORDER:
    print(
        f"{model_name}: "
        f"{json.dumps(clean_params(best_params[model_name]), sort_keys=True)}"
    )

print("\nPASS: Step 17 tuned configurations loaded.")
print("No tuning will be performed in Step 18.")


# =============================================================================
# 8. SPATIAL CONTEXT FEATURE AUDIT
# =============================================================================

section("4. SPATIAL CONTEXT FEATURE AUDIT")

spatial_patterns = [
    r"_nbr_mean$",
    r"_nbr_contrast$",
    r"neighbor",
    r"neighbour",
    r"local_contrast",
    r"spatial",
]

spatial_context_features = [
    c
    for c in predictors
    if any(
        re.search(pattern, c, flags=re.IGNORECASE)
        for pattern in spatial_patterns
    )
]

context_rows = []

for feature in spatial_context_features:
    context_rows.append(
        {
            "feature": feature,
            "contains_target_name_lst": (
                "lst" in feature.lower()
            ),
            "retained": True,
            "reason": (
                "Predictor-derived spatial context. Retained because "
                "the intended inference workflow assumes complete "
                "predictor surfaces across the study area."
            ),
        }
    )

context_df = pd.DataFrame(
    context_rows,
    columns=[
        "feature",
        "contains_target_name_lst",
        "retained",
        "reason",
    ],
)

context_df.to_csv(
    CONTEXT_FEATURE_AUDIT_PATH,
    index=False,
)

print(
    f"Detected spatial-context features: "
    f"{len(spatial_context_features)}"
)

nbr_mean_features = [
    c for c in spatial_context_features
    if c.lower().endswith("_nbr_mean")
]

nbr_contrast_features = [
    c for c in spatial_context_features
    if c.lower().endswith("_nbr_contrast")
]

print(
    f"Neighbor-mean features     : {len(nbr_mean_features)} "
    f"(expected {EXPECTED_NBR_MEAN_FEATURES})"
)
print(
    f"Neighbor-contrast features : {len(nbr_contrast_features)} "
    f"(expected {EXPECTED_NBR_CONTRAST_FEATURES})"
)

for feature in spatial_context_features:
    print(f"  - {feature}")

if len(spatial_context_features) != EXPECTED_SPATIAL_CONTEXT_FEATURES:
    print(
        "\nERROR: detected spatial-context feature count does not match "
        f"the finalized Step 07/08 design. actual={len(spatial_context_features)}, "
        f"expected={EXPECTED_SPATIAL_CONTEXT_FEATURES}."
    )
    sys.exit(1)

if len(nbr_mean_features) != EXPECTED_NBR_MEAN_FEATURES:
    print(
        "\nERROR: neighbor-mean feature count mismatch. "
        f"actual={len(nbr_mean_features)}, "
        f"expected={EXPECTED_NBR_MEAN_FEATURES}."
    )
    sys.exit(1)

if len(nbr_contrast_features) != EXPECTED_NBR_CONTRAST_FEATURES:
    print(
        "\nERROR: neighbor-contrast feature count mismatch. "
        f"actual={len(nbr_contrast_features)}, "
        f"expected={EXPECTED_NBR_CONTRAST_FEATURES}."
    )
    sys.exit(1)

missing_pairs = []
for mean_feature in nbr_mean_features:
    source = mean_feature[:-len("_nbr_mean")]
    contrast_feature = f"{source}_nbr_contrast"
    if contrast_feature not in spatial_context_features:
        missing_pairs.append(
            {
                "source": source,
                "missing": contrast_feature,
            }
        )

if missing_pairs:
    print(
        "\nERROR: one or more neighbor mean/contrast pairs are incomplete:"
    )
    print(missing_pairs)
    sys.exit(1)

if any(
    "lst" in feature.lower()
    for feature in spatial_context_features
):
    print(
        "\nERROR: detected a spatial-context feature containing "
        "the target name LST."
    )
    sys.exit(1)

print("PASS: no detected spatial-context feature uses LST target naming.")
print(
    "NOTE: these predictor-derived spatial context features are retained "
    "under the full-predictor-surface inference assumption."
)


# =============================================================================
# 9. PRIMARY SPATIAL VALIDATION
# =============================================================================

section("5. PRIMARY FIVE-FOLD SPATIAL VALIDATION")

primary_metric_rows = []
primary_prediction_frames = []

for model_name in MODEL_ORDER:
    print(f"\n--- {model_name} ---")

    params = best_params[model_name]

    for fold_id in range(1, EXPECTED_SPATIAL_FOLDS + 1):
        val_mask = (
            df["spatial_fold"].to_numpy()
            == fold_id
        )
        train_mask = ~val_mask

        train_idx = np.flatnonzero(train_mask)
        val_idx = np.flatnonzero(val_mask)

        train_grids = set(
            df.iloc[train_idx][GRID_COL].astype(str).unique()
        )
        val_grids = set(
            df.iloc[val_idx][GRID_COL].astype(str).unique()
        )

        overlap = train_grids.intersection(val_grids)

        if overlap:
            print(
                f"\nERROR: grid overlap in fold {fold_id}: "
                f"{list(overlap)[:10]}"
            )
            sys.exit(1)

        model = build_model(
            model_name,
            params,
        )

        start = time.perf_counter()

        model.fit(
            X_df.iloc[train_idx],
            y[train_idx],
        )

        pred_train = np.asarray(
            model.predict(
                X_df.iloc[train_idx]
            ),
            dtype=float,
        )

        pred_val = np.asarray(
            model.predict(
                X_df.iloc[val_idx]
            ),
            dtype=float,
        )

        runtime = time.perf_counter() - start

        train_m = regression_metrics(
            y[train_idx],
            pred_train,
        )

        val_m = regression_metrics(
            y[val_idx],
            pred_val,
        )

        primary_metric_rows.append(
            {
                "model": model_name,
                "spatial_fold": fold_id,
                "train_grids": len(train_grids),
                "validation_grids": len(val_grids),
                "train_rows": len(train_idx),
                "validation_rows": len(val_idx),
                "train_mae": train_m["mae"],
                "validation_mae": val_m["mae"],
                "mae_generalization_gap": (
                    val_m["mae"] - train_m["mae"]
                ),
                "train_rmse": train_m["rmse"],
                "validation_rmse": val_m["rmse"],
                "train_r2": train_m["r2"],
                "validation_r2": val_m["r2"],
                "train_mape_percent": train_m["mape_percent"],
                "validation_mape_percent": val_m["mape_percent"],
                "runtime_seconds": runtime,
            }
        )

        primary_prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    GRID_COL: (
                        df.iloc[val_idx][GRID_COL]
                        .astype(str)
                        .to_numpy()
                    ),
                    AREA_COL: (
                        df.iloc[val_idx][AREA_COL]
                        .astype(str)
                        .to_numpy()
                    ),
                    YEAR_COL: (
                        df.iloc[val_idx][YEAR_COL]
                        .astype(int)
                        .to_numpy()
                    ),
                    "spatial_fold": fold_id,
                    "actual_lst_c": y[val_idx],
                    "predicted_lst_c": pred_val,
                    "residual_actual_minus_predicted": (
                        y[val_idx] - pred_val
                    ),
                }
            )
        )

        print(
            f"Fold {fold_id} | "
            f"train grids={len(train_grids):,} | "
            f"holdout grids={len(val_grids):,} | "
            f"MAE={val_m['mae']:.4f} °C | "
            f"RMSE={val_m['rmse']:.4f} °C | "
            f"R²={val_m['r2']:.4f} | "
            f"MAPE={val_m['mape_percent']:.2f}% | "
            f"time={runtime:.1f}s"
        )

        del model
        gc.collect()

primary_metrics = pd.DataFrame(primary_metric_rows)

primary_metrics.to_csv(
    PRIMARY_METRICS_PATH,
    index=False,
)

primary_predictions = pd.concat(
    primary_prediction_frames,
    ignore_index=True,
)

primary_predictions.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# =============================================================================
# 10. YEAR-BY-YEAR METRICS INSIDE SPATIAL HOLDOUTS
# =============================================================================

section("6. YEAR-BY-YEAR SPATIAL HOLDOUT DIAGNOSTICS")

year_fold_rows = []

for model_name in MODEL_ORDER:
    model_pred = primary_predictions[
        primary_predictions["model"] == model_name
    ]

    for fold_id in range(1, EXPECTED_SPATIAL_FOLDS + 1):
        for year in DEVELOPMENT_YEARS:
            subset = model_pred[
                (
                    model_pred["spatial_fold"] == fold_id
                )
                & (
                    model_pred[YEAR_COL] == year
                )
            ]

            if subset.empty:
                print(
                    f"\nERROR: missing predictions for "
                    f"{model_name}, fold={fold_id}, year={year}"
                )
                sys.exit(1)

            metrics = regression_metrics(
                subset["actual_lst_c"].to_numpy(dtype=float),
                subset["predicted_lst_c"].to_numpy(dtype=float),
            )

            bias = float(
                (
                    subset["actual_lst_c"]
                    - subset["predicted_lst_c"]
                ).mean()
            )

            year_fold_rows.append(
                {
                    "model": model_name,
                    "spatial_fold": fold_id,
                    "year": year,
                    "rows": len(subset),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "mape_percent": metrics["mape_percent"],
                    "mean_residual_actual_minus_predicted": bias,
                }
            )

year_fold_metrics = pd.DataFrame(
    year_fold_rows
)

year_fold_metrics.to_csv(
    YEAR_FOLD_METRICS_PATH,
    index=False,
)

print(
    f"Saved {len(year_fold_metrics)} model-fold-year diagnostic rows."
)


# =============================================================================
# 11. PRIMARY SUMMARY
# =============================================================================

section("7. PRIMARY SPATIAL VALIDATION SUMMARY")

primary_summary = summarize_fold_metrics(
    primary_metrics
)

primary_summary.to_csv(
    PRIMARY_SUMMARY_PATH,
    index=False,
)

print(primary_summary.to_string(index=False))

print(
    "\nIMPORTANT: Step 18 ranking is descriptive only. "
    "No final model is selected."
)


# =============================================================================
# 12. STEP 17 TEMPORAL VS STEP 18 SPATIAL COMPARISON
# =============================================================================

section("8. STEP 17 TEMPORAL VS STEP 18 SPATIAL EVIDENCE")

step17_summary = pd.read_csv(
    STEP17_SUMMARY_PATH
)

needed_step17 = {
    "model",
    "mean_validation_mae",
    "mean_validation_rmse",
    "mean_validation_r2",
    "negative_r2_folds",
}

missing_step17 = sorted(
    needed_step17 - set(step17_summary.columns)
)

if missing_step17:
    print(
        f"\nERROR: Step 17 summary missing columns: {missing_step17}"
    )
    sys.exit(1)

temporal_view = step17_summary[
    [
        "model",
        "mean_validation_mae",
        "mean_validation_rmse",
        "mean_validation_r2",
        "negative_r2_folds",
    ]
].rename(
    columns={
        "mean_validation_mae": "step17_temporal_mean_mae",
        "mean_validation_rmse": "step17_temporal_mean_rmse",
        "mean_validation_r2": "step17_temporal_mean_r2",
        "negative_r2_folds": "step17_temporal_negative_r2_folds",
    }
)

spatial_view = primary_summary[
    [
        "model",
        "mean_validation_mae",
        "mean_validation_rmse",
        "mean_validation_r2",
        "negative_r2_folds",
    ]
].rename(
    columns={
        "mean_validation_mae": "step18_spatial_mean_mae",
        "mean_validation_rmse": "step18_spatial_mean_rmse",
        "mean_validation_r2": "step18_spatial_mean_r2",
        "negative_r2_folds": "step18_spatial_negative_r2_folds",
    }
)

spatial_vs_temporal = temporal_view.merge(
    spatial_view,
    on="model",
    how="inner",
)

spatial_vs_temporal.to_csv(
    SPATIAL_VS_TEMPORAL_PATH,
    index=False,
)

print(spatial_vs_temporal.to_string(index=False))

print(
    "\nNOTE: temporal and spatial validation answer different questions. "
    "Do not combine them into one score."
)


# =============================================================================
# 13. 210 m BUFFER SENSITIVITY
# =============================================================================

buffer_metrics = pd.DataFrame()
buffer_summary = pd.DataFrame()
buffer_predictions = pd.DataFrame()
buffer_grid_audit = pd.DataFrame()

if RUN_BUFFER_SENSITIVITY:
    section(
        f"9. BUFFER SENSITIVITY - {BUFFER_DISTANCE_METERS:.0f} m"
    )

    grid_gdf = gpd.read_file(
        SPATIAL_GEOMETRY_PATH
    )

    if GRID_COL not in grid_gdf.columns:
        print(
            f"\nERROR: {GRID_COL} missing from spatial geometry."
        )
        sys.exit(1)

    grid_gdf[GRID_COL] = (
        grid_gdf[GRID_COL].astype(str)
    )

    grid_gdf = (
        grid_gdf[
            [GRID_COL, "geometry"]
        ]
        .drop_duplicates(
            subset=[GRID_COL]
        )
        .copy()
    )

    if grid_gdf.crs is None:
        print(
            "\nERROR: geometry has no CRS; cannot run metric buffer."
        )
        sys.exit(1)

    if grid_gdf.crs.to_epsg() != 32644:
        grid_gdf = grid_gdf.to_crs(
            epsg=32644
        )

    if grid_gdf[GRID_COL].nunique() != EXPECTED_UNIQUE_GRIDS:
        print(
            f"\nERROR: geometry has "
            f"{grid_gdf[GRID_COL].nunique()} unique grids; "
            f"expected {EXPECTED_UNIQUE_GRIDS}."
        )
        sys.exit(1)

    grid_gdf = grid_gdf.merge(
        assign,
        on=GRID_COL,
        how="left",
        validate="one_to_one",
    )

    if grid_gdf["spatial_fold"].isna().any():
        print("\nERROR: some geometry grids have no spatial fold.")
        sys.exit(1)

    buffer_grid_rows = []
    buffer_metric_rows = []
    buffer_prediction_frames = []

    for fold_id in range(1, EXPECTED_SPATIAL_FOLDS + 1):
        holdout_geom = grid_gdf[
            grid_gdf["spatial_fold"] == fold_id
        ]

        train_geom = grid_gdf[
            grid_gdf["spatial_fold"] != fold_id
        ]

        heldout_union = get_union(
            holdout_geom.geometry
        )

        exclusion_zone = heldout_union.buffer(
            BUFFER_DISTANCE_METERS
        )

        intersects_buffer = (
            train_geom.geometry.intersects(
                exclusion_zone
            )
        )

        excluded_grids = set(
            train_geom.loc[
                intersects_buffer,
                GRID_COL,
            ].astype(str)
        )

        safe_train_grids = set(
            train_geom.loc[
                ~intersects_buffer,
                GRID_COL,
            ].astype(str)
        )

        holdout_grids = set(
            holdout_geom[GRID_COL].astype(str)
        )

        if safe_train_grids.intersection(holdout_grids):
            print(
                f"\nERROR: buffered train/holdout overlap in fold {fold_id}."
            )
            sys.exit(1)

        if len(safe_train_grids) == 0:
            print(
                f"\nERROR: no safe training grids after buffering fold "
                f"{fold_id}."
            )
            sys.exit(1)

        buffer_grid_rows.append(
            {
                "spatial_fold": fold_id,
                "buffer_distance_m": BUFFER_DISTANCE_METERS,
                "heldout_grids": len(holdout_grids),
                "candidate_training_grids_before_buffer": len(train_geom),
                "excluded_training_grids_due_to_buffer": len(excluded_grids),
                "training_grids_after_buffer": len(safe_train_grids),
            }
        )

        print(
            f"Fold {fold_id}: "
            f"holdout={len(holdout_grids):,} | "
            f"excluded near boundary={len(excluded_grids):,} | "
            f"safe training={len(safe_train_grids):,}"
        )

        train_mask = df[GRID_COL].isin(
            safe_train_grids
        ).to_numpy()

        val_mask = (
            df["spatial_fold"].to_numpy()
            == fold_id
        )

        train_idx = np.flatnonzero(train_mask)
        val_idx = np.flatnonzero(val_mask)

        for model_name in MODEL_ORDER:
            model = build_model(
                model_name,
                best_params[model_name],
            )

            start = time.perf_counter()

            model.fit(
                X_df.iloc[train_idx],
                y[train_idx],
            )

            pred_train = np.asarray(
                model.predict(
                    X_df.iloc[train_idx]
                ),
                dtype=float,
            )

            pred_val = np.asarray(
                model.predict(
                    X_df.iloc[val_idx]
                ),
                dtype=float,
            )

            runtime = time.perf_counter() - start

            train_m = regression_metrics(
                y[train_idx],
                pred_train,
            )

            val_m = regression_metrics(
                y[val_idx],
                pred_val,
            )

            buffer_metric_rows.append(
                {
                    "model": model_name,
                    "spatial_fold": fold_id,
                    "buffer_distance_m": BUFFER_DISTANCE_METERS,
                    "train_grids": len(safe_train_grids),
                    "excluded_training_grids": len(excluded_grids),
                    "validation_grids": len(holdout_grids),
                    "train_rows": len(train_idx),
                    "validation_rows": len(val_idx),
                    "train_mae": train_m["mae"],
                    "validation_mae": val_m["mae"],
                    "mae_generalization_gap": (
                        val_m["mae"] - train_m["mae"]
                    ),
                    "train_rmse": train_m["rmse"],
                    "validation_rmse": val_m["rmse"],
                    "train_r2": train_m["r2"],
                    "validation_r2": val_m["r2"],
                    "train_mape_percent": train_m["mape_percent"],
                    "validation_mape_percent": val_m["mape_percent"],
                    "runtime_seconds": runtime,
                }
            )

            buffer_prediction_frames.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        GRID_COL: (
                            df.iloc[val_idx][GRID_COL]
                            .astype(str)
                            .to_numpy()
                        ),
                        AREA_COL: (
                            df.iloc[val_idx][AREA_COL]
                            .astype(str)
                            .to_numpy()
                        ),
                        YEAR_COL: (
                            df.iloc[val_idx][YEAR_COL]
                            .astype(int)
                            .to_numpy()
                        ),
                        "spatial_fold": fold_id,
                        "buffer_distance_m": BUFFER_DISTANCE_METERS,
                        "actual_lst_c": y[val_idx],
                        "predicted_lst_c": pred_val,
                        "residual_actual_minus_predicted": (
                            y[val_idx] - pred_val
                        ),
                    }
                )
            )

            print(
                f"  {model_name:<14} | "
                f"MAE={val_m['mae']:.4f} °C | "
                f"RMSE={val_m['rmse']:.4f} °C | "
                f"R²={val_m['r2']:.4f} | "
                f"time={runtime:.1f}s"
            )

            del model
            gc.collect()

    buffer_grid_audit = pd.DataFrame(
        buffer_grid_rows
    )
    buffer_grid_audit.to_csv(
        BUFFER_GRID_AUDIT_PATH,
        index=False,
    )

    buffer_metrics = pd.DataFrame(
        buffer_metric_rows
    )
    buffer_metrics.to_csv(
        BUFFER_METRICS_PATH,
        index=False,
    )

    buffer_predictions = pd.concat(
        buffer_prediction_frames,
        ignore_index=True,
    )
    buffer_predictions.to_csv(
        BUFFER_PREDICTIONS_PATH,
        index=False,
    )

    buffer_summary = summarize_fold_metrics(
        buffer_metrics
    )
    buffer_summary.to_csv(
        BUFFER_SUMMARY_PATH,
        index=False,
    )

    print("\nBuffered spatial summary:")
    print(buffer_summary.to_string(index=False))


# =============================================================================
# 14. FIGURES
# =============================================================================

section("10. FIGURES")

plot_summary = primary_summary.sort_values(
    "mean_validation_mae",
    ascending=True,
)

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(
    plot_summary["model"],
    plot_summary["mean_validation_mae"],
)
ax.set_title("Step 18 Mean Spatial Holdout MAE")
ax.set_ylabel("Mean spatial validation MAE (°C)")
ax.set_xlabel("Model")
ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(FIG_MEAN_MAE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 6))

for model_name in MODEL_ORDER:
    md = primary_metrics[
        primary_metrics["model"] == model_name
    ].sort_values("spatial_fold")

    ax.plot(
        md["spatial_fold"],
        md["validation_mae"],
        marker="o",
        label=model_name,
    )

ax.set_title("Spatial Validation MAE by Frozen Step 09 Fold")
ax.set_xlabel("Spatial Fold")
ax.set_ylabel("MAE (°C)")
ax.set_xticks(range(1, EXPECTED_SPATIAL_FOLDS + 1))
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()
plt.savefig(FIG_FOLD_MAE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 6))

for model_name in MODEL_ORDER:
    md = primary_metrics[
        primary_metrics["model"] == model_name
    ].sort_values("spatial_fold")

    ax.plot(
        md["spatial_fold"],
        md["validation_r2"],
        marker="o",
        label=model_name,
    )

ax.axhline(0, linewidth=1)
ax.set_title("Spatial Validation R² by Frozen Step 09 Fold")
ax.set_xlabel("Spatial Fold")
ax.set_ylabel("R²")
ax.set_xticks(range(1, EXPECTED_SPATIAL_FOLDS + 1))
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()
plt.savefig(FIG_FOLD_R2_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)


if RUN_BUFFER_SENSITIVITY:
    standard_plot = primary_summary[
        ["model", "mean_validation_mae"]
    ].rename(
        columns={
            "mean_validation_mae": "standard_mae"
        }
    )

    buffer_plot = buffer_summary[
        ["model", "mean_validation_mae"]
    ].rename(
        columns={
            "mean_validation_mae": "buffered_mae"
        }
    )

    compare_plot = standard_plot.merge(
        buffer_plot,
        on="model",
        how="inner",
    )

    x = np.arange(len(compare_plot))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width / 2,
        compare_plot["standard_mae"],
        width,
        label="Standard spatial holdout",
    )
    ax.bar(
        x + width / 2,
        compare_plot["buffered_mae"],
        width,
        label=f"{BUFFER_DISTANCE_METERS:.0f} m buffered holdout",
    )
    ax.set_xticks(x, compare_plot["model"])
    ax.set_ylabel("Mean validation MAE (°C)")
    ax.set_title("Standard vs Buffered Spatial Validation")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        FIG_BUFFER_COMPARE_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

print(f"Saved: {FIG_MEAN_MAE_PATH}")
print(f"Saved: {FIG_FOLD_MAE_PATH}")
print(f"Saved: {FIG_FOLD_R2_PATH}")

if RUN_BUFFER_SENSITIVITY:
    print(f"Saved: {FIG_BUFFER_COMPARE_PATH}")


# =============================================================================
# 15. STRICT AUDIT
# =============================================================================

section("11. STRICT STEP 18 AUDIT")

expected_primary_metric_rows = (
    len(MODEL_ORDER)
    * EXPECTED_SPATIAL_FOLDS
)

expected_primary_prediction_rows = (
    len(df)
    * len(MODEL_ORDER)
)

finite_primary_metrics = bool(
    np.isfinite(
        primary_metrics[
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

all_grid_years_once = True

for model_name in MODEL_ORDER:
    mp = primary_predictions[
        primary_predictions["model"] == model_name
    ]

    counts = (
        mp.groupby(
            [GRID_COL, YEAR_COL]
        )
        .size()
    )

    if not (
        len(counts) == len(df)
        and counts.eq(1).all()
    ):
        all_grid_years_once = False
        break

buffer_pass = True
buffer_details = "Buffer sensitivity disabled."

if RUN_BUFFER_SENSITIVITY:
    expected_buffer_rows = (
        len(MODEL_ORDER)
        * EXPECTED_SPATIAL_FOLDS
    )

    finite_buffer_metrics = bool(
        np.isfinite(
            buffer_metrics[
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

    buffer_pass = bool(
        len(buffer_metrics) == expected_buffer_rows
        and finite_buffer_metrics
        and (
            buffer_grid_audit["training_grids_after_buffer"] > 0
        ).all()
    )

    buffer_details = (
        f"rows={len(buffer_metrics)}, "
        f"expected={expected_buffer_rows}, "
        f"finite={finite_buffer_metrics}, "
        f"buffer={BUFFER_DISTANCE_METERS}m"
    )

audit_rows = [
    {
        "check": "step09_spatial_split_audit",
        "status": "PASS",
        "details": "Step 09 audit loaded and fully PASS.",
    },
    {
        "check": "step17_tuning_audit",
        "status": "PASS",
        "details": "Step 17 audit loaded and fully PASS.",
    },
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
            if len(predictors) == EXPECTED_PREDICTOR_COUNT
            else "FAIL"
        ),
        "details": (
            f"actual={len(predictors)}, "
            f"expected={EXPECTED_PREDICTOR_COUNT}"
        ),
    },
    {
        "check": "unique_grid_count",
        "status": (
            "PASS"
            if df[GRID_COL].nunique() == EXPECTED_UNIQUE_GRIDS
            else "FAIL"
        ),
        "details": (
            f"actual={df[GRID_COL].nunique()}, "
            f"expected={EXPECTED_UNIQUE_GRIDS}"
        ),
    },
    {
        "check": "frozen_step09_fold_counts",
        "status": (
            "PASS"
            if fold_counts == EXPECTED_FOLD_COUNTS
            else "FAIL"
        ),
        "details": (
            f"actual={fold_counts}, "
            f"expected={EXPECTED_FOLD_COUNTS}"
        ),
    },
    {
        "check": "grid_fold_consistency",
        "status": (
            "PASS"
            if int(fold_consistency.max()) == 1
            else "FAIL"
        ),
        "details": "Each grid has exactly one fold across all years.",
    },
    {
        "check": "2024_loaded_or_used",
        "status": "PASS",
        "details": "No. Only 2018-2023 development rows were loaded.",
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": "No. Only 2018-2023 development rows were loaded.",
    },
    {
        "check": "hyperparameters_retuned",
        "status": "PASS",
        "details": "No. Step 17 tuned configurations were reused exactly.",
    },
    {
        "check": "primary_fold_metric_rows",
        "status": (
            "PASS"
            if len(primary_metrics) == expected_primary_metric_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(primary_metrics)}, "
            f"expected={expected_primary_metric_rows}"
        ),
    },
    {
        "check": "primary_prediction_rows",
        "status": (
            "PASS"
            if len(primary_predictions)
            == expected_primary_prediction_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(primary_predictions)}, "
            f"expected={expected_primary_prediction_rows}"
        ),
    },
    {
        "check": "each_grid_year_predicted_once_per_model",
        "status": "PASS" if all_grid_years_once else "FAIL",
        "details": str(all_grid_years_once),
    },
    {
        "check": "primary_metrics_finite",
        "status": "PASS" if finite_primary_metrics else "FAIL",
        "details": str(finite_primary_metrics),
    },
    {
        "check": "spatial_context_feature_count",
        "status": (
            "PASS"
            if len(spatial_context_features)
            == EXPECTED_SPATIAL_CONTEXT_FEATURES
            else "FAIL"
        ),
        "details": (
            f"actual={len(spatial_context_features)}, "
            f"expected={EXPECTED_SPATIAL_CONTEXT_FEATURES}"
        ),
    },
    {
        "check": "spatial_neighbor_mean_count",
        "status": (
            "PASS"
            if len(nbr_mean_features)
            == EXPECTED_NBR_MEAN_FEATURES
            else "FAIL"
        ),
        "details": (
            f"actual={len(nbr_mean_features)}, "
            f"expected={EXPECTED_NBR_MEAN_FEATURES}"
        ),
    },
    {
        "check": "spatial_neighbor_contrast_count",
        "status": (
            "PASS"
            if len(nbr_contrast_features)
            == EXPECTED_NBR_CONTRAST_FEATURES
            else "FAIL"
        ),
        "details": (
            f"actual={len(nbr_contrast_features)}, "
            f"expected={EXPECTED_NBR_CONTRAST_FEATURES}"
        ),
    },
    {
        "check": "spatial_context_target_name_audit",
        "status": (
            "PASS"
            if not any(
                "lst" in feature.lower()
                for feature in spatial_context_features
            )
            else "FAIL"
        ),
        "details": (
            "No detected spatial-context predictor contains LST target naming."
        ),
    },
    {
        "check": "buffer_sensitivity",
        "status": "PASS" if buffer_pass else "FAIL",
        "details": buffer_details,
    },
    {
        "check": "final_model_selected",
        "status": "PASS",
        "details": (
            "No. Step 18 provides spatial development robustness evidence only."
        ),
    },
    {
        "check": "nested_feature_selection_claim",
        "status": "PASS",
        "details": (
            "No fully nested spatial feature-selection claim is made; "
            "Step 08 frozen features are treated as development-frozen."
        ),
    },
]

audit_df = pd.DataFrame(audit_rows)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
)

print(audit_df.to_string(index=False))

if (audit_df["status"] != "PASS").any():
    print("\nERROR: one or more Step 18 audit checks failed.")
    sys.exit(1)


# =============================================================================
# 16. REPRODUCIBILITY HASHES
# =============================================================================

section("12. REPRODUCIBILITY HASHES")

hash_files = [
    INPUT_PATH,
    SPATIAL_ASSIGNMENT_PATH,
    SPATIAL_GEOMETRY_PATH,
    STEP09_AUDIT_PATH,
    STEP17_AUDIT_PATH,
    STEP17_PARAMS_PATH,
    STEP17_SUMMARY_PATH,
    PRIMARY_METRICS_PATH,
    PRIMARY_SUMMARY_PATH,
    YEAR_FOLD_METRICS_PATH,
    SPATIAL_VS_TEMPORAL_PATH,
    CONTEXT_FEATURE_AUDIT_PATH,
    PREDICTIONS_PATH,
    AUDIT_PATH,
]

if RUN_BUFFER_SENSITIVITY:
    hash_files.extend(
        [
            BUFFER_METRICS_PATH,
            BUFFER_SUMMARY_PATH,
            BUFFER_GRID_AUDIT_PATH,
            BUFFER_PREDICTIONS_PATH,
        ]
    )

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
# 17. REPORT
# =============================================================================

section("13. SAVE STEP 18 REPORT")

report_lines = [
    "STEP 18 - SPATIAL VALIDATION",
    "=" * 112,
    "",
    "PURPOSE",
    "-" * 112,
    "Evaluate spatial generalization of the fixed Step 17 tuned candidates",
    "using the five geometry-only folds frozen in Step 09.",
    "",
    "DATA POLICY",
    "-" * 112,
    "2018-2023: development rows used.",
    "2024: NOT loaded; reserved validation remains untouched.",
    "2025: NOT loaded; final lockbox remains untouched.",
    "",
    "PRIMARY SPATIAL SPLIT",
    "-" * 112,
    f"Fold counts: {EXPECTED_FOLD_COUNTS}",
    "All years of a held-out grid were excluded from that fold's training set.",
    "",
    "MODEL POLICY",
    "-" * 112,
    "Step 17 tuned hyperparameters were reused exactly.",
    "No tuning was performed in Step 18.",
    "No final model was selected.",
    "",
    "FEATURE-SELECTION INTERPRETATION",
    "-" * 112,
    "The 93-feature set was already frozen in Step 08 using development evidence.",
    "Therefore Step 18 is a development robustness analysis conditional on that",
    "frozen feature set, not a fully nested spatial feature-selection estimate.",
    "",
    "SPATIAL-CONTEXT FEATURE ASSUMPTION",
    "-" * 112,
    f"Detected spatial-context features: {len(spatial_context_features)}.",
    "These are predictor-derived rather than LST-target-derived.",
    "They are retained under the assumption that the corresponding predictor",
    "surfaces are available across the full study area at inference time.",
    "",
    "PRIMARY SPATIAL RESULTS",
    "-" * 112,
]

for _, row in primary_summary.iterrows():
    report_lines.append(
        f"{row['model']}: "
        f"mean MAE={row['mean_validation_mae']:.6f} °C, "
        f"mean RMSE={row['mean_validation_rmse']:.6f} °C, "
        f"mean R²={row['mean_validation_r2']:.6f}, "
        f"negative-R² folds={int(row['negative_r2_folds'])}, "
        f"worst-fold MAE={row['worst_fold_mae']:.6f} °C, "
        f"mean MAE gap={row['mean_mae_generalization_gap']:.6f} °C."
    )

if RUN_BUFFER_SENSITIVITY:
    report_lines.extend(
        [
            "",
            f"{BUFFER_DISTANCE_METERS:.0f} m BUFFER SENSITIVITY RESULTS",
            "-" * 112,
        ]
    )

    for _, row in buffer_summary.iterrows():
        report_lines.append(
            f"{row['model']}: "
            f"mean MAE={row['mean_validation_mae']:.6f} °C, "
            f"mean RMSE={row['mean_validation_rmse']:.6f} °C, "
            f"mean R²={row['mean_validation_r2']:.6f}, "
            f"negative-R² folds={int(row['negative_r2_folds'])}."
        )

report_lines.extend(
    [
        "",
        "INTERPRETATION",
        "-" * 112,
        "Step 18 provides spatial development robustness evidence.",
        "The Step 18 descriptive ranking is not final model selection.",
        "Temporal instability from Step 17 is carried forward to Step 19",
        "for dedicated temporal-shift diagnosis and improvement.",
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
# 18. FINAL STATUS
# =============================================================================

section("STEP 18 COMPLETED SUCCESSFULLY")

print(
    "Primary five-fold spatial validation completed.\n"
    f"Frozen Step 09 fold counts verified: {EXPECTED_FOLD_COUNTS}.\n"
    "Step 17 tuned hyperparameters reused without retuning.\n"
    "Only 2018-2023 development rows were used.\n"
    "2024 reserved validation remained untouched.\n"
    "2025 final lockbox remained untouched.\n"
    f"Buffer sensitivity run: {RUN_BUFFER_SENSITIVITY} "
    f"({BUFFER_DISTANCE_METERS:.0f} m when enabled).\n"
    "No final model was selected.\n"
    "Review the complete Step 18 output before starting Step 19."
)
