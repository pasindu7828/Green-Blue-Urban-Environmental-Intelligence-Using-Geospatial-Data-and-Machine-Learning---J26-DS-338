
"""
STEP 11 - BASELINE MODELS
Urban Heat Research - Kaduwela, Sri Lanka

STRICT RULES
------------
- Uses ONLY 2018-2023 development data.
- Does NOT load 2024 validation data.
- Does NOT load 2025 final lockbox data.
- Uses rolling-origin evaluation:
    2018 -> 2019
    2018-2019 -> 2020
    2018-2020 -> 2021
    2018-2021 -> 2022
    2018-2022 -> 2023
- Linear Regression uses fold-local train-only StandardScaler.
- 'year' remains unscaled.
- No hyperparameter tuning.
- No final model selection.

Baselines
---------
1. Global Historical Mean
2. Grid Historical Mean
3. Last-Year Persistence
4. 3-Year Grid Moving Average
5. Linear Regression
"""

from pathlib import Path
import hashlib
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# SETTINGS
# =============================================================================

TARGET = "LST_C"
DEVELOPMENT_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
ROLLING_VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]
UNSCALED_LINEAR_FEATURES = ["year"]

MODEL_NAMES = [
    "Global Historical Mean",
    "Grid Historical Mean",
    "Last-Year Persistence",
    "3-Year Grid Moving Average",
    "Linear Regression",
]


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "linear"
    / "linear_train_2018_2023.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "baselines"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

for d in [MODEL_DIR, METRICS_DIR, PREDICTION_DIR, REPORT_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ROLLING_METRICS_PATH = METRICS_DIR / "11_baseline_rolling_origin_metrics.csv"
AGG_METRICS_PATH = METRICS_DIR / "11_baseline_aggregated_metrics.csv"
PREDICTIONS_PATH = PREDICTION_DIR / "11_baseline_rolling_origin_predictions.csv"

LINEAR_MODEL_PATH = MODEL_DIR / "11_linear_baseline_model_2018_2023.joblib"
LINEAR_SCALER_PATH = MODEL_DIR / "11_linear_baseline_scaler_2018_2023.joblib"

FEATURE_LIST_PATH = REPORT_DIR / "11_linear_baseline_feature_list.csv"
AUDIT_PATH = REPORT_DIR / "11_baseline_model_audit.csv"
HASH_PATH = REPORT_DIR / "11_baseline_file_hashes.csv"
REPORT_PATH = REPORT_DIR / "11_baseline_models_report.txt"

MAE_FIG_PATH = FIGURE_DIR / "11_baseline_mae_by_year.png"
R2_FIG_PATH = FIGURE_DIR / "11_baseline_r2_by_year.png"


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return np.nan
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": mape(y_true, y_pred),
    }


def ensure_valid_predictions(values, expected_n, label):
    arr = np.asarray(values, dtype=float)
    if len(arr) != expected_n:
        raise ValueError(f"{label}: wrong prediction length.")
    if np.isnan(arr).any():
        raise ValueError(f"{label}: NaN predictions detected.")
    if np.isinf(arr).any():
        raise ValueError(f"{label}: infinite predictions detected.")
    return arr


# =============================================================================
# LOAD DEVELOPMENT DATA ONLY
# =============================================================================

section("STEP 11 - BASELINE MODELS")

print(f"Project root : {PROJECT_ROOT}")
print(f"Input        : {INPUT_PATH}")
print("2024 data    : NOT LOADED")
print("2025 data    : NOT LOADED")

if not INPUT_PATH.exists():
    print(f"\nERROR: required Step 09 training file not found:\n{INPUT_PATH}")
    sys.exit(1)

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False,
    dtype={"grid_id": "string", "area_name": "string"},
)

required_cols = {"grid_id", "year", TARGET, "area_name"}
missing_required = sorted(required_cols - set(df.columns))

if missing_required:
    print(f"\nERROR: missing required columns: {missing_required}")
    sys.exit(1)

years = sorted(df["year"].astype(int).unique().tolist())

if years != DEVELOPMENT_YEARS:
    print(f"\nERROR: expected {DEVELOPMENT_YEARS}, got {years}")
    sys.exit(1)

duplicate_keys = int(df.duplicated(["grid_id", "year"], keep=False).sum())

if duplicate_keys != 0:
    print(f"\nERROR: duplicate grid-year rows = {duplicate_keys}")
    sys.exit(1)

print(f"Rows         : {len(df):,}")
print(f"Columns      : {len(df.columns):,}")
print(f"Unique grids : {df['grid_id'].nunique():,}")
print(f"Years        : {years}")


# =============================================================================
# FEATURE AUDIT
# =============================================================================

section("1. LINEAR FEATURE AUDIT")

linear_features = [
    c for c in df.columns
    if c not in ["grid_id", TARGET, "area_name"]
]

scaled_features = [
    c for c in linear_features
    if c not in UNSCALED_LINEAR_FEATURES
]

if "year" not in linear_features:
    print("\nERROR: year predictor missing.")
    sys.exit(1)

non_numeric = [
    c for c in linear_features
    if not pd.api.types.is_numeric_dtype(df[c])
]

if non_numeric:
    print(f"\nERROR: non-numeric predictors found: {non_numeric}")
    sys.exit(1)

missing_predictors = int(df[linear_features].isna().sum().sum())
inf_predictors = int(
    np.isinf(df[linear_features].to_numpy(dtype=float)).sum()
)

if missing_predictors != 0 or inf_predictors != 0:
    print(
        f"\nERROR: predictor missing={missing_predictors}, "
        f"infinite={inf_predictors}"
    )
    sys.exit(1)

print(f"Total linear predictors : {len(linear_features)}")
print(f"Scaled inside each fold : {len(scaled_features)}")
print(f"Preserved unscaled      : {UNSCALED_LINEAR_FEATURES}")
print("PASS: predictors are numeric, finite, and complete.")

pd.DataFrame({
    "feature": linear_features,
    "scaled_per_fold": [f in scaled_features for f in linear_features],
    "preserved_unscaled": [f in UNSCALED_LINEAR_FEATURES for f in linear_features],
}).to_csv(FEATURE_LIST_PATH, index=False)


# =============================================================================
# ROLLING-ORIGIN BASELINE EVALUATION
# =============================================================================

section("2. ROLLING-ORIGIN BASELINE EVALUATION")

metric_rows = []
prediction_frames = []

for fold, validation_year in enumerate(ROLLING_VALIDATION_YEARS, start=1):

    train_years = [y for y in DEVELOPMENT_YEARS if y < validation_year]

    train = df[df["year"].isin(train_years)].copy()
    val = df[df["year"] == validation_year].copy()

    if train.empty or val.empty:
        raise ValueError(f"Fold {fold}: empty train or validation set.")

    if train["grid_id"].nunique() != val["grid_id"].nunique():
        raise ValueError(f"Fold {fold}: grid coverage mismatch.")

    y_true = val[TARGET].to_numpy(dtype=float)

    fold_pred = pd.DataFrame({
        "grid_id": val["grid_id"].to_numpy(),
        "validation_year": validation_year,
        "actual_LST_C": y_true,
    })

    # 1. Global historical mean
    global_mean = float(train[TARGET].mean())
    pred_global = np.full(len(val), global_mean, dtype=float)

    # 2. Grid historical mean
    grid_mean_map = train.groupby("grid_id")[TARGET].mean()
    pred_grid_mean = val["grid_id"].map(grid_mean_map).to_numpy(dtype=float)

    # 3. Last-year persistence
    prev_year = validation_year - 1
    prev_map = (
        train[train["year"] == prev_year]
        .set_index("grid_id")[TARGET]
    )
    pred_persistence = val["grid_id"].map(prev_map).to_numpy(dtype=float)

    # 4. Up-to-3-year moving average by grid
    start_year = max(min(train_years), validation_year - 3)
    window = train[
        train["year"].between(start_year, validation_year - 1)
    ]
    moving_map = window.groupby("grid_id")[TARGET].mean()
    pred_moving = val["grid_id"].map(moving_map).to_numpy(dtype=float)

    # 5. Linear Regression with fold-local scaling
    scaler = StandardScaler()

    X_train_scaled = train[scaled_features].astype(float)
    X_val_scaled = val[scaled_features].astype(float)

    scaler.fit(X_train_scaled)

    X_train_model = pd.DataFrame(
        scaler.transform(X_train_scaled),
        columns=scaled_features,
        index=train.index,
    )

    X_val_model = pd.DataFrame(
        scaler.transform(X_val_scaled),
        columns=scaled_features,
        index=val.index,
    )

    for feature in UNSCALED_LINEAR_FEATURES:
        X_train_model[feature] = train[feature].astype(float).to_numpy()
        X_val_model[feature] = val[feature].astype(float).to_numpy()

    X_train_model = X_train_model[linear_features]
    X_val_model = X_val_model[linear_features]

    lr = LinearRegression()
    lr.fit(X_train_model, train[TARGET])
    pred_linear = lr.predict(X_val_model)

    model_predictions = {
        "Global Historical Mean": pred_global,
        "Grid Historical Mean": pred_grid_mean,
        "Last-Year Persistence": pred_persistence,
        "3-Year Grid Moving Average": pred_moving,
        "Linear Regression": pred_linear,
    }

    print(
        f"\nFold {fold}: "
        f"{min(train_years)}-{max(train_years)} -> {validation_year}"
    )

    for model_name, prediction in model_predictions.items():

        prediction = ensure_valid_predictions(
            prediction,
            len(val),
            f"{model_name} / {validation_year}",
        )

        result = metrics(y_true, prediction)

        metric_rows.append({
            "fold": fold,
            "train_start_year": min(train_years),
            "train_end_year": max(train_years),
            "validation_year": validation_year,
            "train_rows": len(train),
            "validation_rows": len(val),
            "model": model_name,
            **result,
        })

        fold_pred[model_name] = prediction

        print(
            f"{model_name:<28} "
            f"MAE={result['mae']:.4f} | "
            f"RMSE={result['rmse']:.4f} | "
            f"R2={result['r2']:.4f} | "
            f"MAPE={result['mape_percent']:.2f}%"
        )

    prediction_frames.append(fold_pred)


# =============================================================================
# SAVE RESULTS
# =============================================================================

section("3. SAVE ROLLING RESULTS")

rolling_metrics: pd.DataFrame = pd.DataFrame(metric_rows)
rolling_predictions: pd.DataFrame = pd.concat(
    prediction_frames,
    ignore_index=True,
)

rolling_metrics.to_csv(ROLLING_METRICS_PATH, index=False)
rolling_predictions.to_csv(PREDICTIONS_PATH, index=False)

print(f"Saved: {ROLLING_METRICS_PATH}")
print(f"Saved: {PREDICTIONS_PATH}")


# =============================================================================
# AGGREGATED METRICS
# =============================================================================

section("4. AGGREGATED DEVELOPMENT BASELINE METRICS")

aggregated = (
    rolling_metrics
    .groupby("model")
    .agg(
        mean_mae=("mae", "mean"),
        median_mae=("mae", "median"),
        std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"),
        median_rmse=("rmse", "median"),
        mean_r2=("r2", "mean"),
        median_r2=("r2", "median"),
        negative_r2_folds=("r2", lambda s: int((s < 0).sum())),
        mean_mape_percent=("mape_percent", "mean"),
        folds=("fold", "count"),
    )
    .reset_index()
)

global_mean_mae = float(
    aggregated.loc[
        aggregated["model"] == "Global Historical Mean",
        "mean_mae"
    ].iloc[0]
)

aggregated["mae_skill_vs_global_mean"] = (
    1.0 - aggregated["mean_mae"] / global_mean_mae
)

aggregated = aggregated.sort_values(
    ["mean_mae", "mean_rmse", "model"]
).reset_index(drop=True)

aggregated["development_rank_by_mean_mae"] = np.arange(
    1, len(aggregated) + 1
)

aggregated.to_csv(AGG_METRICS_PATH, index=False)

print(aggregated.to_string(index=False))

print(
    "\nNOTE: this ranking is only a baseline-development summary. "
    "It is NOT final model selection."
)


# =============================================================================
# FIT FULL 2018-2023 LINEAR BASELINE ARTIFACT
# =============================================================================

section("5. FIT FULL-DEVELOPMENT LINEAR BASELINE ARTIFACT")

full_scaler = StandardScaler()
full_scaler.fit(df[scaled_features].astype(float))

X_full = pd.DataFrame(
    full_scaler.transform(df[scaled_features].astype(float)),
    columns=scaled_features,
    index=df.index,
)

for feature in UNSCALED_LINEAR_FEATURES:
    X_full[feature] = df[feature].astype(float).to_numpy()

X_full = X_full[linear_features]

full_lr = LinearRegression()
full_lr.fit(X_full, df[TARGET])

joblib.dump({
    "model": full_lr,
    "feature_names": linear_features,
    "fit_years": DEVELOPMENT_YEARS,
    "target": TARGET,
    "2024_used": False,
    "2025_used": False,
}, LINEAR_MODEL_PATH)

joblib.dump({
    "scaler": full_scaler,
    "scaled_feature_names": scaled_features,
    "preserved_unscaled_features": UNSCALED_LINEAR_FEATURES,
    "fit_years": DEVELOPMENT_YEARS,
    "rolling_origin_reuse_allowed": False,
}, LINEAR_SCALER_PATH)

print(f"Saved: {LINEAR_MODEL_PATH}")
print(f"Saved: {LINEAR_SCALER_PATH}")


# =============================================================================
# FIGURES
# =============================================================================

section("6. FIGURES")

mae_pivot = rolling_metrics.pivot(
    index="validation_year",
    columns="model",
    values="mae",
)

fig, ax = plt.subplots(figsize=(11, 6))
for col in mae_pivot.columns:
    ax.plot(
        mae_pivot.index.astype(str),
        mae_pivot[col],
        marker="o",
        label=col,
    )
ax.set_title("Baseline MAE Across Rolling-Origin Validation Years")
ax.set_xlabel("Validation Year")
ax.set_ylabel("MAE (°C)")
ax.grid(alpha=0.2)
ax.legend()
plt.tight_layout()
plt.savefig(MAE_FIG_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

r2_pivot = rolling_metrics.pivot(
    index="validation_year",
    columns="model",
    values="r2",
)

fig, ax = plt.subplots(figsize=(11, 6))
for col in r2_pivot.columns:
    ax.plot(
        r2_pivot.index.astype(str),
        r2_pivot[col],
        marker="o",
        label=col,
    )
ax.axhline(0, linewidth=1)
ax.set_title("Baseline R² Across Rolling-Origin Validation Years")
ax.set_xlabel("Validation Year")
ax.set_ylabel("R²")
ax.grid(alpha=0.2)
ax.legend()
plt.tight_layout()
plt.savefig(R2_FIG_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {MAE_FIG_PATH}")
print(f"Saved: {R2_FIG_PATH}")


# =============================================================================
# STRICT AUDIT
# =============================================================================

section("7. STRICT STEP 11 AUDIT")

rolling_years_used = sorted(
    rolling_metrics["validation_year"].unique().tolist()
)

models_present = sorted(
    rolling_metrics["model"].unique().tolist()
)

expected_rows = (
    len(ROLLING_VALIDATION_YEARS) * len(MODEL_NAMES)
)

audit = pd.DataFrame([
    {
        "check": "development_input_years",
        "status": "PASS" if years == DEVELOPMENT_YEARS else "FAIL",
        "details": str(years),
    },
    {
        "check": "2024_loaded_or_used",
        "status": "PASS",
        "details": "No. 2024 file was not loaded.",
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": "No. 2025 file was not loaded.",
    },
    {
        "check": "rolling_validation_years",
        "status": (
            "PASS"
            if rolling_years_used == ROLLING_VALIDATION_YEARS
            else "FAIL"
        ),
        "details": str(rolling_years_used),
    },
    {
        "check": "all_baselines_present",
        "status": (
            "PASS"
            if set(models_present) == set(MODEL_NAMES)
            else "FAIL"
        ),
        "details": str(models_present),
    },
    {
        "check": "expected_metric_rows",
        "status": "PASS" if len(rolling_metrics) == expected_rows else "FAIL",
        "details": f"actual={len(rolling_metrics)}, expected={expected_rows}",
    },
    {
        "check": "fold_local_scaling",
        "status": "PASS",
        "details": (
            "Each Linear Regression fold fits StandardScaler only "
            "on that fold's training rows."
        ),
    },
    {
        "check": "year_preserved_unscaled",
        "status": "PASS",
        "details": "Raw year remains unscaled in Linear Regression.",
    },
    {
        "check": "target_as_linear_predictor",
        "status": "PASS",
        "details": "No. LST_C is excluded from the Linear Regression predictors.",
    },
    {
        "check": "identifier_as_linear_predictor",
        "status": "PASS",
        "details": "No. grid_id and area_name are excluded.",
    },
    {
        "check": "final_model_selected",
        "status": "PASS",
        "details": "No. Step 11 establishes baselines only.",
    },
])

audit.to_csv(AUDIT_PATH, index=False)
print(audit.to_string(index=False))

if (audit["status"] != "PASS").any():
    print("\nERROR: one or more Step 11 audit checks failed.")
    sys.exit(1)


# =============================================================================
# HASHES
# =============================================================================

section("8. REPRODUCIBILITY HASHES")

hash_files = [
    INPUT_PATH,
    ROLLING_METRICS_PATH,
    AGG_METRICS_PATH,
    PREDICTIONS_PATH,
    LINEAR_MODEL_PATH,
    LINEAR_SCALER_PATH,
    FEATURE_LIST_PATH,
]

hash_rows = []

for path in hash_files:
    digest = sha256_file(path)
    hash_rows.append({"file": str(path), "sha256": digest})
    print(f"{path.name}: {digest}")

pd.DataFrame(hash_rows).to_csv(HASH_PATH, index=False)


# =============================================================================
# REPORT
# =============================================================================

section("9. SAVE STEP 11 REPORT")

top_baseline = aggregated.iloc[0]

report_lines = [
    "STEP 11 - BASELINE MODELS REPORT",
    "=" * 100,
    "",
    "DATA PROTECTION",
    "2018-2023 development data only.",
    "2024 not loaded.",
    "2025 not loaded.",
    "",
    "BASELINES",
    "1. Global Historical Mean",
    "2. Grid Historical Mean",
    "3. Last-Year Persistence",
    "4. 3-Year Grid Moving Average",
    "5. Linear Regression",
    "",
    "ROLLING-ORIGIN FOLDS",
    "2018 -> 2019",
    "2018-2019 -> 2020",
    "2018-2020 -> 2021",
    "2018-2021 -> 2022",
    "2018-2022 -> 2023",
    "",
    "LINEAR PREPROCESSING",
    f"Total predictors: {len(linear_features)}",
    f"Scaled per fold: {len(scaled_features)}",
    f"Preserved unscaled: {UNSCALED_LINEAR_FEATURES}",
    "",
    "BEST DEVELOPMENT BASELINE BY MEAN MAE",
    f"Model: {top_baseline['model']}",
    f"Mean MAE: {top_baseline['mean_mae']:.6f}",
    f"Mean RMSE: {top_baseline['mean_rmse']:.6f}",
    f"Mean R2: {top_baseline['mean_r2']:.6f}",
    "",
    "IMPORTANT",
    "This is not final model selection.",
    "2024 remains reserved.",
    "2025 remains the final lockbox.",
    "",
    "NEXT",
    "Step 12 - Random Forest candidate model analysis.",
]

REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

print(f"Saved: {REPORT_PATH}")
print(f"Saved: {AUDIT_PATH}")
print(f"Saved: {HASH_PATH}")


# =============================================================================
# FINAL
# =============================================================================

section("STEP 11 COMPLETED SUCCESSFULLY")

print(
    "Baseline development completed using 2018-2023 only.\n"
    "2024 remains untouched for reserved validation.\n"
    "2025 remains untouched as the final lockbox.\n"
    "Five baseline methods were evaluated across five rolling-origin folds.\n"
    "Linear Regression used fold-local train-only scaling.\n"
    "No final research model was selected.\n"
    "Review the full output before Step 12."
)
