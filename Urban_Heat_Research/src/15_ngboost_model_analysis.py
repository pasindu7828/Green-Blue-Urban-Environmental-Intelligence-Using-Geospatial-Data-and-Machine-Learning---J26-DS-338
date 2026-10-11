
"""
STEP 15 - NGBOOST CANDIDATE MODEL ANALYSIS
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Evaluate NGBoost as a FIXED, UNTUNED probabilistic candidate model using only
the 2018-2023 development period.

STRICT PROTECTION
-----------------
2024 validation data is NOT loaded.
2025 final lockbox data is NOT loaded.

ROLLING-ORIGIN FOLDS
--------------------
2018       -> 2019
2018-2019  -> 2020
2018-2020  -> 2021
2018-2021  -> 2022
2018-2022  -> 2023

WHAT THIS STEP DOES
-------------------
- Fit the same predefined NGBoost configuration in every rolling fold.
- Calculate point-prediction MAE, RMSE, R² and MAPE.
- Quantify train-vs-validation generalization gaps.
- Evaluate NGBoost probabilistic outputs using Gaussian predictive mean/std.
- Calculate NLL, interval coverage and interval width for 80%, 90% and 95%.
- Compare NGBoost against Step 11 baselines.
- Save descriptive references against RF, XGBoost and LightGBM.
- Save fold-level predictions, residuals and uncertainty intervals.
- Calculate fold-level validation permutation importance using point MAE.
- Fit and save ONE full 2018-2023 NGBoost candidate artifact.

WHAT THIS STEP DOES NOT DO
--------------------------
- No hyperparameter search/tuning.
- No uncertainty calibration.
- No 2024 evaluation.
- No 2025 evaluation.
- No final model selection.
- No decision yet that NGBoost will be the final uncertainty method.
"""

from pathlib import Path
import hashlib
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

try:
    import ngboost
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import LogScore
except Exception as exc:
    print("ERROR: ngboost could not be imported.")
    print("Install/check ngboost inside the project virtual environment.")
    print(f"Original error: {exc}")
    sys.exit(1)


# =============================================================================
# 1. SETTINGS
# =============================================================================

TARGET = "LST_C"
SEED = 42

DEVELOPMENT_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
ROLLING_VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]

# Fixed shallow-tree base learner for candidate screening.
# This is NOT the result of hyperparameter optimization.
BASE_LEARNER_PARAMS = {
    "criterion": "squared_error",
    "max_depth": 3,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": SEED,
}

# Fixed candidate NGBoost configuration.
# This is intentionally NOT tuned in Step 15.
NGB_PARAMS_SERIALIZABLE = {
    "Dist": "Normal",
    "Score": "LogScore",
    "natural_gradient": True,
    "n_estimators": 500,
    "learning_rate": 0.01,
    "minibatch_frac": 1.0,
    "col_sample": 1.0,
    "verbose": False,
    "random_state": SEED,
}

PERMUTATION_REPEATS = 3
INTERVAL_LEVELS = [0.80, 0.90, 0.95]
STD_EPS = 1e-8


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

BASELINE_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "11_baseline_rolling_origin_metrics.csv"
)

RF_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "12_random_forest_rolling_origin_metrics.csv"
)

XGB_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "13_xgboost_rolling_origin_metrics.csv"
)

LGBM_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "14_lightgbm_rolling_origin_metrics.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "candidates"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

for d in [MODEL_DIR, METRICS_DIR, PREDICTION_DIR, REPORT_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ROLLING_METRICS_PATH = (
    METRICS_DIR / "15_ngboost_rolling_origin_metrics.csv"
)

AGG_METRICS_PATH = (
    METRICS_DIR / "15_ngboost_aggregated_metrics.csv"
)

PROB_METRICS_PATH = (
    METRICS_DIR / "15_ngboost_probabilistic_metrics.csv"
)

BASELINE_COMPARISON_PATH = (
    METRICS_DIR / "15_ngboost_vs_baselines.csv"
)

MODEL_REFERENCE_PATH = (
    METRICS_DIR / "15_ngboost_vs_rf_xgb_lgbm_reference.csv"
)

PREDICTIONS_PATH = (
    PREDICTION_DIR / "15_ngboost_rolling_predictions_with_uncertainty.csv"
)

RESIDUAL_SUMMARY_PATH = (
    REPORT_DIR / "15_ngboost_residual_summary.csv"
)

PERM_FOLD_PATH = (
    REPORT_DIR / "15_ngboost_permutation_importance_by_fold.csv"
)

PERM_AGG_PATH = (
    REPORT_DIR / "15_ngboost_permutation_importance_aggregated.csv"
)

MODEL_PATH = (
    MODEL_DIR / "15_ngboost_candidate_2018_2023.joblib"
)

PARAMS_PATH = (
    REPORT_DIR / "15_ngboost_candidate_parameters.json"
)

AUDIT_PATH = (
    REPORT_DIR / "15_ngboost_audit.csv"
)

HASH_PATH = (
    REPORT_DIR / "15_ngboost_file_hashes.csv"
)

REPORT_PATH = (
    REPORT_DIR / "15_ngboost_model_report.txt"
)

MAE_FIG_PATH = (
    FIGURE_DIR / "15_ngboost_mae_by_year.png"
)

R2_FIG_PATH = (
    FIGURE_DIR / "15_ngboost_r2_by_year.png"
)

RESIDUAL_FIG_PATH = (
    FIGURE_DIR / "15_ngboost_residuals_by_year.png"
)

IMPORTANCE_FIG_PATH = (
    FIGURE_DIR / "15_ngboost_permutation_importance_top25.png"
)

COVERAGE_FIG_PATH = (
    FIGURE_DIR / "15_ngboost_interval_coverage.png"
)


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 124)
    print(title)
    print("=" * 124)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mape_percent(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return np.nan

    return float(
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100.0
    )


def calculate_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": mape_percent(y_true, y_pred),
    }


def check_array(arr, expected_length, label, positive_only=False) -> np.ndarray:
    values = np.asarray(arr, dtype=float)

    if len(values) != expected_length:
        raise ValueError(
            f"{label}: length {len(values)} != {expected_length}"
        )

    if np.isnan(values).any():
        raise ValueError(f"{label}: NaN values detected.")

    if np.isinf(values).any():
        raise ValueError(f"{label}: infinite values detected.")

    if positive_only and np.any(values <= 0):
        raise ValueError(f"{label}: non-positive values detected.")

    return values


def make_ngboost_model() -> NGBRegressor:
    base = DecisionTreeRegressor(
        **BASE_LEARNER_PARAMS
    )

    return NGBRegressor(
        Dist=Normal,
        Score=LogScore,
        Base=base,
        natural_gradient=NGB_PARAMS_SERIALIZABLE["natural_gradient"],
        n_estimators=NGB_PARAMS_SERIALIZABLE["n_estimators"],
        learning_rate=NGB_PARAMS_SERIALIZABLE["learning_rate"],
        minibatch_frac=NGB_PARAMS_SERIALIZABLE["minibatch_frac"],
        col_sample=NGB_PARAMS_SERIALIZABLE["col_sample"],
        verbose=NGB_PARAMS_SERIALIZABLE["verbose"],
        random_state=NGB_PARAMS_SERIALIZABLE["random_state"],
    )


def extract_normal_distribution(model, X, label):
    dist = model.pred_dist(X)

    mu = check_array(
        dist.loc,
        len(X),
        f"{label} predictive mean",
    )

    sigma = check_array(
        dist.scale,
        len(X),
        f"{label} predictive std",
        positive_only=True,
    )

    sigma = np.maximum(
        sigma,
        STD_EPS,
    )

    return mu, sigma


def normal_nll(y_true, mu, sigma) -> float:
    y_true = np.asarray(y_true, dtype=float)
    log_density = norm.logpdf(
        y_true,
        loc=mu,
        scale=sigma,
    )
    return float(
        -np.mean(log_density)
    )


def interval_metrics(y_true, mu, sigma, level):
    alpha = 1.0 - level
    z = norm.ppf(
        1.0 - alpha / 2.0
    )

    lower = mu - z * sigma
    upper = mu + z * sigma

    y_true = np.asarray(y_true, dtype=float)

    covered = (
        (y_true >= lower)
        & (y_true <= upper)
    )

    coverage = float(
        covered.mean()
    )

    width = upper - lower

    return {
        "lower": lower,
        "upper": upper,
        "coverage": coverage,
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
    }


# =============================================================================
# 4. LOAD DEVELOPMENT DATA ONLY
# =============================================================================

section("STEP 15 - NGBOOST CANDIDATE MODEL ANALYSIS")

print(f"Project root         : {PROJECT_ROOT}")
print(f"Tree input           : {INPUT_PATH}")
print(f"Step 11 baselines    : {BASELINE_METRICS_PATH}")
print(f"Step 12 RF metrics   : {RF_METRICS_PATH}")
print(f"Step 13 XGB metrics  : {XGB_METRICS_PATH}")
print(f"Step 14 LGBM metrics : {LGBM_METRICS_PATH}")
print(f"NGBoost version      : {getattr(ngboost, '__version__', 'unknown')}")
print("2024 data            : NOT LOADED")
print("2025 data            : NOT LOADED")

required_files = [
    INPUT_PATH,
    BASELINE_METRICS_PATH,
    RF_METRICS_PATH,
    XGB_METRICS_PATH,
    LGBM_METRICS_PATH,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("\nERROR: required prior-step files are missing:")
    for item in missing_files:
        print(f" - {item}")
    sys.exit(1)

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False,
    dtype={
        "grid_id": "string",
        "area_name": "string",
    },
)

baseline_metrics = pd.read_csv(BASELINE_METRICS_PATH)
rf_metrics = pd.read_csv(RF_METRICS_PATH)
xgb_metrics = pd.read_csv(XGB_METRICS_PATH)
lgbm_metrics = pd.read_csv(LGBM_METRICS_PATH)

required = {
    "grid_id",
    "year",
    TARGET,
    "area_name",
}

missing_required = sorted(
    required - set(df.columns)
)

if missing_required:
    print(
        f"\nERROR: missing required columns: {missing_required}"
    )
    sys.exit(1)

actual_years = sorted(
    df["year"].astype(int).unique().tolist()
)

if actual_years != DEVELOPMENT_YEARS:
    print(
        f"\nERROR: expected development years {DEVELOPMENT_YEARS}, "
        f"got {actual_years}"
    )
    sys.exit(1)

duplicate_keys = int(
    df.duplicated(
        ["grid_id", "year"],
        keep=False,
    ).sum()
)

if duplicate_keys != 0:
    print(
        f"\nERROR: duplicate grid-year rows = {duplicate_keys}"
    )
    sys.exit(1)

print(f"Rows         : {len(df):,}")
print(f"Columns      : {len(df.columns):,}")
print(f"Unique grids : {df['grid_id'].nunique():,}")
print(f"Years        : {actual_years}")


# =============================================================================
# 5. FEATURE AUDIT
# =============================================================================

section("1. NGBOOST FEATURE AUDIT")

feature_columns = [
    c
    for c in df.columns
    if c not in [
        "grid_id",
        TARGET,
        "area_name",
    ]
]

if TARGET in feature_columns:
    print("\nERROR: target leaked into predictors.")
    sys.exit(1)

if "grid_id" in feature_columns or "area_name" in feature_columns:
    print(
        "\nERROR: identifier/reporting field leaked into predictors."
    )
    sys.exit(1)

non_numeric = [
    c
    for c in feature_columns
    if not pd.api.types.is_numeric_dtype(df[c])
]

if non_numeric:
    print(
        f"\nERROR: non-numeric NGBoost predictors found: {non_numeric}"
    )
    sys.exit(1)

missing_cells = int(
    df[feature_columns].isna().sum().sum()
)

infinite_cells = int(
    np.isinf(
        df[feature_columns].to_numpy(dtype=float)
    ).sum()
)

if missing_cells != 0 or infinite_cells != 0:
    print(
        f"\nERROR: predictor missing={missing_cells}, "
        f"infinite={infinite_cells}"
    )
    sys.exit(1)

print(f"Predictor count : {len(feature_columns)}")
print(f"Missing cells   : {missing_cells}")
print(f"Infinite cells  : {infinite_cells}")
print("Scaling         : NONE")
print("Imputation      : NONE")
print("PASS: finalized tree/nonlinear feature set is model-ready.")


# =============================================================================
# 6. FIXED CANDIDATE CONFIGURATION
# =============================================================================

section("2. FIXED UNTUNED NGBOOST CONFIGURATION")

print("Base learner:")
print(
    json.dumps(
        BASE_LEARNER_PARAMS,
        indent=2,
    )
)

print("\nNGBoost:")
print(
    json.dumps(
        NGB_PARAMS_SERIALIZABLE,
        indent=2,
    )
)

PARAMS_PATH.write_text(
    json.dumps(
        {
            "model": "NGBRegressor",
            "ngboost_version": getattr(
                ngboost,
                "__version__",
                "unknown",
            ),
            "purpose": "untuned probabilistic candidate analysis",
            "distribution": "Normal",
            "score": "LogScore",
            "base_learner": {
                "model": "DecisionTreeRegressor",
                "parameters": BASE_LEARNER_PARAMS,
            },
            "parameters": NGB_PARAMS_SERIALIZABLE,
            "hyperparameter_search_performed": False,
            "uncertainty_calibration_performed": False,
            "development_years": DEVELOPMENT_YEARS,
            "2024_used": False,
            "2025_used": False,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(
    "\nIMPORTANT: these parameters are fixed for Step 15. "
    "No parameter search or uncertainty calibration is performed here."
)


# =============================================================================
# 7. ROLLING-ORIGIN EVALUATION
# =============================================================================

section("3. ROLLING-ORIGIN NGBOOST EVALUATION")

metric_rows = []
probability_rows = []
prediction_frames = []
permutation_rows = []

for fold, validation_year in enumerate(
    ROLLING_VALIDATION_YEARS,
    start=1,
):
    train_years = [
        y
        for y in DEVELOPMENT_YEARS
        if y < validation_year
    ]

    train = df[
        df["year"].isin(train_years)
    ].copy()

    val = df[
        df["year"] == validation_year
    ].copy()

    if train.empty or val.empty:
        raise ValueError(
            f"Fold {fold}: empty train or validation set."
        )

    X_train = train[feature_columns]
    y_train = train[TARGET]

    X_val = val[feature_columns]
    y_val = val[TARGET]

    model = make_ngboost_model()

    model.fit(
        X_train,
        y_train,
    )

    train_mu, train_sigma = extract_normal_distribution(
        model,
        X_train,
        f"Fold {fold} train",
    )

    val_mu, val_sigma = extract_normal_distribution(
        model,
        X_val,
        f"Fold {fold} validation",
    )

    train_result = calculate_metrics(
        y_train.to_numpy(dtype=float),
        train_mu,
    )

    val_result = calculate_metrics(
        y_val.to_numpy(dtype=float),
        val_mu,
    )

    mae_gap = (
        val_result["mae"]
        - train_result["mae"]
    )

    rmse_gap = (
        val_result["rmse"]
        - train_result["rmse"]
    )

    r2_gap = (
        train_result["r2"]
        - val_result["r2"]
    )

    metric_rows.append(
        {
            "fold": fold,
            "train_start_year": min(train_years),
            "train_end_year": max(train_years),
            "validation_year": validation_year,
            "train_rows": len(train),
            "validation_rows": len(val),
            "train_mae": train_result["mae"],
            "validation_mae": val_result["mae"],
            "mae_generalization_gap": mae_gap,
            "train_rmse": train_result["rmse"],
            "validation_rmse": val_result["rmse"],
            "rmse_generalization_gap": rmse_gap,
            "train_r2": train_result["r2"],
            "validation_r2": val_result["r2"],
            "r2_generalization_gap": r2_gap,
            "train_mape_percent": train_result["mape_percent"],
            "validation_mape_percent": val_result["mape_percent"],
        }
    )

    y_train_np = y_train.to_numpy(dtype=float)
    y_val_np = y_val.to_numpy(dtype=float)

    train_nll = normal_nll(
        y_train_np,
        train_mu,
        train_sigma,
    )

    val_nll = normal_nll(
        y_val_np,
        val_mu,
        val_sigma,
    )

    train_interval_results = {}
    val_interval_results = {}

    for level in INTERVAL_LEVELS:
        train_interval_results[level] = interval_metrics(
            y_train_np,
            train_mu,
            train_sigma,
            level,
        )

        val_interval_results[level] = interval_metrics(
            y_val_np,
            val_mu,
            val_sigma,
            level,
        )

    probability_row = {
        "fold": fold,
        "train_start_year": min(train_years),
        "train_end_year": max(train_years),
        "validation_year": validation_year,
        "train_nll": train_nll,
        "validation_nll": val_nll,
        "train_mean_predicted_std": float(
            np.mean(train_sigma)
        ),
        "validation_mean_predicted_std": float(
            np.mean(val_sigma)
        ),
    }

    for level in INTERVAL_LEVELS:
        pct = int(round(level * 100))

        probability_row[
            f"train_coverage_{pct}"
        ] = train_interval_results[level]["coverage"]

        probability_row[
            f"validation_coverage_{pct}"
        ] = val_interval_results[level]["coverage"]

        probability_row[
            f"train_mean_width_{pct}"
        ] = train_interval_results[level]["mean_width"]

        probability_row[
            f"validation_mean_width_{pct}"
        ] = val_interval_results[level]["mean_width"]

    probability_rows.append(
        probability_row
    )

    residual = (
        y_val_np
        - val_mu
    )

    fold_predictions = pd.DataFrame(
        {
            "grid_id": val["grid_id"].to_numpy(),
            "area_name": val["area_name"].to_numpy(),
            "validation_year": validation_year,
            "actual_LST_C": y_val_np,
            "predicted_mean_LST_C": val_mu,
            "predicted_std_LST_C": val_sigma,
            "residual_actual_minus_predicted": residual,
            "absolute_error": np.abs(residual),
        }
    )

    for level in INTERVAL_LEVELS:
        pct = int(round(level * 100))

        fold_predictions[
            f"lower_{pct}"
        ] = val_interval_results[level]["lower"]

        fold_predictions[
            f"upper_{pct}"
        ] = val_interval_results[level]["upper"]

        fold_predictions[
            f"covered_{pct}"
        ] = (
            (y_val_np >= val_interval_results[level]["lower"])
            & (y_val_np <= val_interval_results[level]["upper"])
        )

    prediction_frames.append(
        fold_predictions
    )

    print(
        f"\nFold {fold}: "
        f"{min(train_years)}-{max(train_years)} -> {validation_year}"
    )

    print(
        f"Train point : MAE={train_result['mae']:.4f} | "
        f"RMSE={train_result['rmse']:.4f} | "
        f"R2={train_result['r2']:.4f} | "
        f"MAPE={train_result['mape_percent']:.2f}%"
    )

    print(
        f"Val point   : MAE={val_result['mae']:.4f} | "
        f"RMSE={val_result['rmse']:.4f} | "
        f"R2={val_result['r2']:.4f} | "
        f"MAPE={val_result['mape_percent']:.2f}%"
    )

    print(
        f"Gap         : MAE={mae_gap:.4f} | "
        f"RMSE={rmse_gap:.4f} | "
        f"R2={r2_gap:.4f}"
    )

    print(
        f"Probabilistic: Val NLL={val_nll:.4f} | "
        f"Mean predicted std={np.mean(val_sigma):.4f} °C"
    )

    for level in INTERVAL_LEVELS:
        pct = int(round(level * 100))
        result = val_interval_results[level]

        print(
            f"  {pct}% interval: "
            f"coverage={result['coverage'] * 100:.2f}% | "
            f"mean width={result['mean_width']:.4f} °C"
        )

    # Point-prediction permutation importance on the validation year.
    # This is diagnostic only; no feature elimination/tuning happens here.
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=UserWarning,
        )

        perm = permutation_importance(
            model,
            X_val,
            y_val,
            scoring="neg_mean_absolute_error",
            n_repeats=PERMUTATION_REPEATS,
            random_state=SEED + fold,
            n_jobs=-1,
        )

    for feature, mean_imp, std_imp in zip(
        feature_columns,
        perm.importances_mean,
        perm.importances_std,
    ):
        permutation_rows.append(
            {
                "fold": fold,
                "validation_year": validation_year,
                "feature": feature,
                "permutation_importance_mean": float(mean_imp),
                "permutation_importance_std": float(std_imp),
            }
        )


# =============================================================================
# 8. SAVE ROLLING RESULTS
# =============================================================================

section("4. SAVE ROLLING RESULTS")

rolling_metrics = pd.DataFrame(
    metric_rows
)

prob_metrics = pd.DataFrame(
    probability_rows
)

rolling_predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

permutation_fold = pd.DataFrame(
    permutation_rows
)

rolling_metrics.to_csv(
    ROLLING_METRICS_PATH,
    index=False,
)

prob_metrics.to_csv(
    PROB_METRICS_PATH,
    index=False,
)

rolling_predictions.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

permutation_fold.to_csv(
    PERM_FOLD_PATH,
    index=False,
)

print(f"Saved: {ROLLING_METRICS_PATH}")
print(f"Saved: {PROB_METRICS_PATH}")
print(f"Saved: {PREDICTIONS_PATH}")
print(f"Saved: {PERM_FOLD_PATH}")


# =============================================================================
# 9. AGGREGATED NGBOOST POINT METRICS
# =============================================================================

section("5. AGGREGATED NGBOOST POINT METRICS")

aggregated_ngb = pd.DataFrame(
    [
        {
            "model": "NGBoost",
            "folds": len(rolling_metrics),
            "mean_train_mae": rolling_metrics["train_mae"].mean(),
            "mean_validation_mae": rolling_metrics["validation_mae"].mean(),
            "median_validation_mae": rolling_metrics["validation_mae"].median(),
            "std_validation_mae": rolling_metrics["validation_mae"].std(),
            "mean_validation_rmse": rolling_metrics["validation_rmse"].mean(),
            "median_validation_rmse": rolling_metrics["validation_rmse"].median(),
            "mean_validation_r2": rolling_metrics["validation_r2"].mean(),
            "median_validation_r2": rolling_metrics["validation_r2"].median(),
            "negative_r2_folds": int(
                (rolling_metrics["validation_r2"] < 0).sum()
            ),
            "mean_validation_mape_percent": (
                rolling_metrics["validation_mape_percent"].mean()
            ),
            "mean_mae_generalization_gap": (
                rolling_metrics["mae_generalization_gap"].mean()
            ),
            "mean_rmse_generalization_gap": (
                rolling_metrics["rmse_generalization_gap"].mean()
            ),
            "mean_r2_generalization_gap": (
                rolling_metrics["r2_generalization_gap"].mean()
            ),
            "mean_validation_nll": prob_metrics["validation_nll"].mean(),
            "mean_validation_predicted_std": (
                prob_metrics["validation_mean_predicted_std"].mean()
            ),
            "mean_validation_coverage_80": (
                prob_metrics["validation_coverage_80"].mean()
            ),
            "mean_validation_coverage_90": (
                prob_metrics["validation_coverage_90"].mean()
            ),
            "mean_validation_coverage_95": (
                prob_metrics["validation_coverage_95"].mean()
            ),
            "mean_validation_width_80": (
                prob_metrics["validation_mean_width_80"].mean()
            ),
            "mean_validation_width_90": (
                prob_metrics["validation_mean_width_90"].mean()
            ),
            "mean_validation_width_95": (
                prob_metrics["validation_mean_width_95"].mean()
            ),
        }
    ]
)

aggregated_ngb.to_csv(
    AGG_METRICS_PATH,
    index=False,
)

print(
    aggregated_ngb.to_string(
        index=False
    )
)


# =============================================================================
# 10. COMPARE WITH STEP 11 BASELINES
# =============================================================================

section("6. NGBOOST VS STEP 11 BASELINES")

required_baseline_cols = {
    "validation_year",
    "model",
    "mae",
    "rmse",
    "r2",
    "mape_percent",
}

if not required_baseline_cols.issubset(
    baseline_metrics.columns
):
    print(
        "\nERROR: Step 11 baseline metrics file has unexpected schema."
    )
    sys.exit(1)

comparison_rows = []

for _, ngb_row in rolling_metrics.iterrows():
    year = int(
        ngb_row["validation_year"]
    )

    year_baselines = baseline_metrics[
        baseline_metrics["validation_year"] == year
    ].copy()

    if year_baselines.empty:
        print(
            f"\nERROR: no Step 11 baseline metrics found for {year}."
        )
        sys.exit(1)

    best_baseline = (
        year_baselines
        .sort_values(
            ["mae", "rmse", "model"],
            ascending=[True, True, True],
        )
        .iloc[0]
    )

    grid_baseline_rows = year_baselines[
        year_baselines["model"] == "Grid Historical Mean"
    ]

    if grid_baseline_rows.empty:
        print(
            f"\nERROR: Grid Historical Mean missing for {year}."
        )
        sys.exit(1)

    grid_baseline = grid_baseline_rows.iloc[0]

    comparison_rows.append(
        {
            "validation_year": year,
            "ngb_mae": ngb_row["validation_mae"],
            "ngb_rmse": ngb_row["validation_rmse"],
            "ngb_r2": ngb_row["validation_r2"],
            "best_baseline_model_that_year": best_baseline["model"],
            "best_baseline_mae": best_baseline["mae"],
            "ngb_mae_improvement_vs_best_baseline": (
                best_baseline["mae"]
                - ngb_row["validation_mae"]
            ),
            "ngb_beats_best_baseline_mae": (
                ngb_row["validation_mae"]
                < best_baseline["mae"]
            ),
            "grid_historical_mean_mae": grid_baseline["mae"],
            "ngb_mae_improvement_vs_grid_historical_mean": (
                grid_baseline["mae"]
                - ngb_row["validation_mae"]
            ),
            "ngb_beats_grid_historical_mean_mae": (
                ngb_row["validation_mae"]
                < grid_baseline["mae"]
            ),
        }
    )

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    BASELINE_COMPARISON_PATH,
    index=False,
)

print(
    comparison_df.to_string(
        index=False
    )
)

ngb_beats_best_count = int(
    comparison_df[
        "ngb_beats_best_baseline_mae"
    ].sum()
)

ngb_beats_grid_count = int(
    comparison_df[
        "ngb_beats_grid_historical_mean_mae"
    ].sum()
)

print(
    f"\nNGBoost beats the best Step 11 baseline by MAE in "
    f"{ngb_beats_best_count}/{len(comparison_df)} folds."
)

print(
    f"NGBoost beats Grid Historical Mean by MAE in "
    f"{ngb_beats_grid_count}/{len(comparison_df)} folds."
)


# =============================================================================
# 11. DESCRIPTIVE REFERENCE VS RF/XGB/LGBM
# =============================================================================

section("7. NGBOOST VS RF / XGBOOST / LIGHTGBM REFERENCE")

required_candidate_cols = {
    "validation_year",
    "validation_mae",
    "validation_rmse",
    "validation_r2",
}

for label, candidate_df in [
    ("Random Forest", rf_metrics),
    ("XGBoost", xgb_metrics),
    ("LightGBM", lgbm_metrics),
]:
    if not required_candidate_cols.issubset(
        candidate_df.columns
    ):
        print(
            f"\nERROR: {label} metrics schema mismatch."
        )
        sys.exit(1)

model_reference = (
    rolling_metrics[
        [
            "validation_year",
            "validation_mae",
            "validation_rmse",
            "validation_r2",
        ]
    ]
    .rename(
        columns={
            "validation_mae": "ngb_mae",
            "validation_rmse": "ngb_rmse",
            "validation_r2": "ngb_r2",
        }
    )
    .merge(
        rf_metrics[
            [
                "validation_year",
                "validation_mae",
                "validation_rmse",
                "validation_r2",
            ]
        ].rename(
            columns={
                "validation_mae": "rf_mae",
                "validation_rmse": "rf_rmse",
                "validation_r2": "rf_r2",
            }
        ),
        on="validation_year",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        xgb_metrics[
            [
                "validation_year",
                "validation_mae",
                "validation_rmse",
                "validation_r2",
            ]
        ].rename(
            columns={
                "validation_mae": "xgb_mae",
                "validation_rmse": "xgb_rmse",
                "validation_r2": "xgb_r2",
            }
        ),
        on="validation_year",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        lgbm_metrics[
            [
                "validation_year",
                "validation_mae",
                "validation_rmse",
                "validation_r2",
            ]
        ].rename(
            columns={
                "validation_mae": "lgbm_mae",
                "validation_rmse": "lgbm_rmse",
                "validation_r2": "lgbm_r2",
            }
        ),
        on="validation_year",
        how="inner",
        validate="one_to_one",
    )
)

for short_name in [
    "rf",
    "xgb",
    "lgbm",
]:
    model_reference[
        f"ngb_mae_minus_{short_name}_mae"
    ] = (
        model_reference["ngb_mae"]
        - model_reference[f"{short_name}_mae"]
    )

    model_reference[
        f"ngb_lower_mae_than_{short_name}"
    ] = (
        model_reference["ngb_mae"]
        < model_reference[f"{short_name}_mae"]
    )

model_reference.to_csv(
    MODEL_REFERENCE_PATH,
    index=False,
)

print(
    model_reference.to_string(
        index=False
    )
)

ngb_lower_rf_count = int(
    model_reference[
        "ngb_lower_mae_than_rf"
    ].sum()
)

ngb_lower_xgb_count = int(
    model_reference[
        "ngb_lower_mae_than_xgb"
    ].sum()
)

ngb_lower_lgbm_count = int(
    model_reference[
        "ngb_lower_mae_than_lgbm"
    ].sum()
)

print(
    f"\nNGBoost has lower MAE than RF in "
    f"{ngb_lower_rf_count}/{len(model_reference)} folds."
)

print(
    f"NGBoost has lower MAE than XGBoost in "
    f"{ngb_lower_xgb_count}/{len(model_reference)} folds."
)

print(
    f"NGBoost has lower MAE than LightGBM in "
    f"{ngb_lower_lgbm_count}/{len(model_reference)} folds."
)

print(
    "NOTE: these are descriptive development-stage references only. "
    "Formal multi-model comparison remains Step 16."
)


# =============================================================================
# 12. RESIDUAL DIAGNOSTICS
# =============================================================================

section("8. RESIDUAL DIAGNOSTICS")

residual_summary = (
    rolling_predictions
    .groupby("validation_year")
    .agg(
        residual_mean=(
            "residual_actual_minus_predicted",
            "mean",
        ),
        residual_median=(
            "residual_actual_minus_predicted",
            "median",
        ),
        residual_std=(
            "residual_actual_minus_predicted",
            "std",
        ),
        mean_absolute_error=(
            "absolute_error",
            "mean",
        ),
        p90_absolute_error=(
            "absolute_error",
            lambda s: float(np.percentile(s, 90)),
        ),
        p95_absolute_error=(
            "absolute_error",
            lambda s: float(np.percentile(s, 95)),
        ),
        max_absolute_error=(
            "absolute_error",
            "max",
        ),
        mean_predicted_std=(
            "predicted_std_LST_C",
            "mean",
        ),
    )
    .reset_index()
)

residual_summary.to_csv(
    RESIDUAL_SUMMARY_PATH,
    index=False,
)

print(
    residual_summary.to_string(
        index=False
    )
)


# =============================================================================
# 13. AGGREGATE PERMUTATION IMPORTANCE
# =============================================================================

section("9. PERMUTATION IMPORTANCE STABILITY")

perm_agg = (
    permutation_fold
    .groupby("feature")
    .agg(
        mean_importance=(
            "permutation_importance_mean",
            "mean",
        ),
        median_importance=(
            "permutation_importance_mean",
            "median",
        ),
        std_importance=(
            "permutation_importance_mean",
            "std",
        ),
        positive_fold_fraction=(
            "permutation_importance_mean",
            lambda s: float((s > 0).mean()),
        ),
        negative_fold_fraction=(
            "permutation_importance_mean",
            lambda s: float((s < 0).mean()),
        ),
        min_importance=(
            "permutation_importance_mean",
            "min",
        ),
        max_importance=(
            "permutation_importance_mean",
            "max",
        ),
    )
    .reset_index()
)

perm_agg["robust_importance_score"] = (
    perm_agg["median_importance"].clip(lower=0)
    * perm_agg["positive_fold_fraction"]
)

perm_agg = perm_agg.sort_values(
    [
        "robust_importance_score",
        "mean_importance",
        "feature",
    ],
    ascending=[False, False, True],
).reset_index(drop=True)

perm_agg["rank"] = np.arange(
    1,
    len(perm_agg) + 1,
)

perm_agg.to_csv(
    PERM_AGG_PATH,
    index=False,
)

print("Top 25 robust permutation features:")
print(
    perm_agg.head(25).to_string(
        index=False
    )
)


# =============================================================================
# 14. FIT FULL 2018-2023 NGBOOST CANDIDATE ARTIFACT
# =============================================================================

section("10. FIT FULL 2018-2023 NGBOOST CANDIDATE")

full_model = make_ngboost_model()

full_model.fit(
    df[feature_columns],
    df[TARGET],
)

joblib.dump(
    {
        "model": full_model,
        "feature_names": feature_columns,
        "fit_years": DEVELOPMENT_YEARS,
        "target": TARGET,
        "distribution": "Normal",
        "score": "LogScore",
        "base_learner_parameters": BASE_LEARNER_PARAMS,
        "ngboost_parameters": NGB_PARAMS_SERIALIZABLE,
        "hyperparameter_tuned": False,
        "uncertainty_calibrated": False,
        "2024_used": False,
        "2025_used": False,
        "purpose": (
            "Untuned NGBoost probabilistic candidate artifact for later "
            "reserved validation/comparison."
        ),
    },
    MODEL_PATH,
)

print(
    f"Saved candidate model: {MODEL_PATH}"
)


# =============================================================================
# 15. FIGURES
# =============================================================================

section("11. FIGURES")

# Train vs validation MAE
fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    rolling_metrics["validation_year"].astype(str),
    rolling_metrics["train_mae"],
    marker="o",
    label="Train MAE",
)

ax.plot(
    rolling_metrics["validation_year"].astype(str),
    rolling_metrics["validation_mae"],
    marker="o",
    label="Validation MAE",
)

ax.set_title(
    "NGBoost Train vs Validation MAE"
)
ax.set_xlabel(
    "Validation Year"
)
ax.set_ylabel(
    "MAE (°C)"
)
ax.grid(
    alpha=0.2
)
ax.legend()

plt.tight_layout()
plt.savefig(
    MAE_FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Train vs validation R²
fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    rolling_metrics["validation_year"].astype(str),
    rolling_metrics["train_r2"],
    marker="o",
    label="Train R²",
)

ax.plot(
    rolling_metrics["validation_year"].astype(str),
    rolling_metrics["validation_r2"],
    marker="o",
    label="Validation R²",
)

ax.axhline(
    0,
    linewidth=1,
)

ax.set_title(
    "NGBoost Train vs Validation R²"
)
ax.set_xlabel(
    "Validation Year"
)
ax.set_ylabel(
    "R²"
)
ax.grid(
    alpha=0.2
)
ax.legend()

plt.tight_layout()
plt.savefig(
    R2_FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Residual boxplot
fig, ax = plt.subplots(
    figsize=(10, 6)
)

year_groups = [
    rolling_predictions.loc[
        rolling_predictions["validation_year"] == year,
        "residual_actual_minus_predicted",
    ].to_numpy()
    for year in ROLLING_VALIDATION_YEARS
]

ax.boxplot(
    year_groups,
    tick_labels=[
        str(year)
        for year in ROLLING_VALIDATION_YEARS
    ],
)

ax.axhline(
    0,
    linewidth=1,
)

ax.set_title(
    "NGBoost Validation Residuals by Year"
)
ax.set_xlabel(
    "Validation Year"
)
ax.set_ylabel(
    "Residual (Actual - Predicted) °C"
)
ax.grid(
    axis="y",
    alpha=0.2,
)

plt.tight_layout()
plt.savefig(
    RESIDUAL_FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Robust permutation importance
top_perm = (
    perm_agg
    .head(25)
    .sort_values(
        "robust_importance_score",
        ascending=True,
    )
)

fig, ax = plt.subplots(
    figsize=(10, 9)
)

ax.barh(
    top_perm["feature"],
    top_perm["robust_importance_score"],
)

ax.set_title(
    "NGBoost Robust Rolling Permutation Importance - Top 25"
)
ax.set_xlabel(
    "Robust importance score"
)
ax.set_ylabel(
    "Feature"
)
ax.grid(
    axis="x",
    alpha=0.2,
)

plt.tight_layout()
plt.savefig(
    IMPORTANCE_FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Interval coverage by year
fig, ax = plt.subplots(
    figsize=(10, 6)
)

years_as_text = (
    prob_metrics["validation_year"]
    .astype(str)
)

for level in INTERVAL_LEVELS:
    pct = int(
        round(level * 100)
    )

    ax.plot(
        years_as_text,
        prob_metrics[
            f"validation_coverage_{pct}"
        ] * 100.0,
        marker="o",
        label=f"Observed {pct}% interval coverage",
    )

    ax.axhline(
        pct,
        linewidth=1,
        linestyle="--",
    )

ax.set_title(
    "NGBoost Validation Interval Coverage by Year"
)
ax.set_xlabel(
    "Validation Year"
)
ax.set_ylabel(
    "Observed Coverage (%)"
)
ax.set_ylim(
    0,
    100,
)
ax.grid(
    alpha=0.2
)
ax.legend()

plt.tight_layout()
plt.savefig(
    COVERAGE_FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print(f"Saved: {MAE_FIG_PATH}")
print(f"Saved: {R2_FIG_PATH}")
print(f"Saved: {RESIDUAL_FIG_PATH}")
print(f"Saved: {IMPORTANCE_FIG_PATH}")
print(f"Saved: {COVERAGE_FIG_PATH}")


# =============================================================================
# 16. STRICT AUDIT
# =============================================================================

section("12. STRICT STEP 15 AUDIT")

validation_years_used = sorted(
    rolling_metrics[
        "validation_year"
    ].unique().tolist()
)

expected_rows = len(
    ROLLING_VALIDATION_YEARS
)

all_point_predictions_finite = bool(
    np.isfinite(
        rolling_predictions[
            [
                "actual_LST_C",
                "predicted_mean_LST_C",
                "residual_actual_minus_predicted",
                "absolute_error",
            ]
        ].to_numpy(dtype=float)
    ).all()
)

all_uncertainty_finite = bool(
    np.isfinite(
        rolling_predictions[
            [
                "predicted_std_LST_C",
                "lower_80",
                "upper_80",
                "lower_90",
                "upper_90",
                "lower_95",
                "upper_95",
            ]
        ].to_numpy(dtype=float)
    ).all()
)

all_std_positive = bool(
    (
        rolling_predictions[
            "predicted_std_LST_C"
        ] > 0
    ).all()
)

all_intervals_ordered = bool(
    (
        rolling_predictions["lower_80"]
        <= rolling_predictions["upper_80"]
    ).all()
    and (
        rolling_predictions["lower_90"]
        <= rolling_predictions["upper_90"]
    ).all()
    and (
        rolling_predictions["lower_95"]
        <= rolling_predictions["upper_95"]
    ).all()
)

reference_complete = (
    len(model_reference)
    == expected_rows
)

prob_metrics_complete = (
    len(prob_metrics)
    == expected_rows
)

coverage_valid = True

for pct in [80, 90, 95]:
    coverage_valid = coverage_valid and bool(
        (
            prob_metrics[
                f"validation_coverage_{pct}"
            ].between(0.0, 1.0)
        ).all()
    )

audit_df = pd.DataFrame(
    [
        {
            "check": "development_input_years",
            "status": (
                "PASS"
                if actual_years == DEVELOPMENT_YEARS
                else "FAIL"
            ),
            "details": str(actual_years),
        },
        {
            "check": "2024_loaded_or_used",
            "status": "PASS",
            "details": "No. 2024 validation data was not loaded.",
        },
        {
            "check": "2025_loaded_or_used",
            "status": "PASS",
            "details": "No. 2025 lockbox data was not loaded.",
        },
        {
            "check": "rolling_validation_years",
            "status": (
                "PASS"
                if validation_years_used == ROLLING_VALIDATION_YEARS
                else "FAIL"
            ),
            "details": str(validation_years_used),
        },
        {
            "check": "expected_fold_rows",
            "status": (
                "PASS"
                if len(rolling_metrics) == expected_rows
                else "FAIL"
            ),
            "details": (
                f"actual={len(rolling_metrics)}, "
                f"expected={expected_rows}"
            ),
        },
        {
            "check": "target_in_predictors",
            "status": "PASS",
            "details": "No. LST_C excluded from NGBoost predictors.",
        },
        {
            "check": "identifiers_in_predictors",
            "status": "PASS",
            "details": "No. grid_id and area_name excluded.",
        },
        {
            "check": "scaling_or_imputation",
            "status": "PASS",
            "details": "None applied to NGBoost branch.",
        },
        {
            "check": "hyperparameter_search",
            "status": "PASS",
            "details": "No. Fixed candidate configuration only.",
        },
        {
            "check": "uncertainty_calibration",
            "status": "PASS",
            "details": "No. Raw candidate-stage NGBoost uncertainty only.",
        },
        {
            "check": "finite_point_predictions",
            "status": (
                "PASS"
                if all_point_predictions_finite
                else "FAIL"
            ),
            "details": str(all_point_predictions_finite),
        },
        {
            "check": "finite_uncertainty_outputs",
            "status": (
                "PASS"
                if all_uncertainty_finite
                else "FAIL"
            ),
            "details": str(all_uncertainty_finite),
        },
        {
            "check": "positive_predictive_std",
            "status": (
                "PASS"
                if all_std_positive
                else "FAIL"
            ),
            "details": str(all_std_positive),
        },
        {
            "check": "interval_bounds_ordered",
            "status": (
                "PASS"
                if all_intervals_ordered
                else "FAIL"
            ),
            "details": str(all_intervals_ordered),
        },
        {
            "check": "coverage_values_valid",
            "status": (
                "PASS"
                if coverage_valid
                else "FAIL"
            ),
            "details": str(coverage_valid),
        },
        {
            "check": "probabilistic_metrics_complete",
            "status": (
                "PASS"
                if prob_metrics_complete
                else "FAIL"
            ),
            "details": f"rows={len(prob_metrics)}",
        },
        {
            "check": "prior_model_reference_complete",
            "status": (
                "PASS"
                if reference_complete
                else "FAIL"
            ),
            "details": f"rows={len(model_reference)}",
        },
        {
            "check": "final_model_selected",
            "status": "PASS",
            "details": "No. NGBoost remains a candidate model.",
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

if (
    audit_df["status"] != "PASS"
).any():
    print(
        "\nERROR: one or more Step 15 audit checks failed."
    )
    sys.exit(1)


# =============================================================================
# 17. REPRODUCIBILITY HASHES
# =============================================================================

section("13. REPRODUCIBILITY HASHES")

hash_files = [
    INPUT_PATH,
    BASELINE_METRICS_PATH,
    RF_METRICS_PATH,
    XGB_METRICS_PATH,
    LGBM_METRICS_PATH,
    ROLLING_METRICS_PATH,
    AGG_METRICS_PATH,
    PROB_METRICS_PATH,
    BASELINE_COMPARISON_PATH,
    MODEL_REFERENCE_PATH,
    PREDICTIONS_PATH,
    RESIDUAL_SUMMARY_PATH,
    PERM_FOLD_PATH,
    PERM_AGG_PATH,
    MODEL_PATH,
    PARAMS_PATH,
]

hash_rows = []

for path in hash_files:
    digest = sha256_file(
        path
    )

    hash_rows.append(
        {
            "file": str(path),
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


# =============================================================================
# 18. REPORT
# =============================================================================

section("14. SAVE STEP 15 REPORT")

agg_row = aggregated_ngb.iloc[0]

report_lines = [
    "STEP 15 - NGBOOST CANDIDATE MODEL REPORT",
    "=" * 108,
    "",
    "DATA PROTECTION",
    "-" * 108,
    "Used: 2018-2023 development data only.",
    "2024: NOT loaded.",
    "2025: NOT loaded.",
    "",
    "MODEL",
    "-" * 108,
    "NGBRegressor",
    f"NGBoost version: {getattr(ngboost, '__version__', 'unknown')}",
    "Distribution: Normal",
    "Score: LogScore",
    "Status: UNTUNED probabilistic candidate model",
    f"Base learner parameters: {BASE_LEARNER_PARAMS}",
    f"NGBoost parameters: {NGB_PARAMS_SERIALIZABLE}",
    "",
    "ROLLING-ORIGIN DEVELOPMENT VALIDATION",
    "-" * 108,
    "2018 -> 2019",
    "2018-2019 -> 2020",
    "2018-2020 -> 2021",
    "2018-2021 -> 2022",
    "2018-2022 -> 2023",
    "",
    "AGGREGATED POINT PERFORMANCE",
    "-" * 108,
    f"Mean validation MAE: {agg_row['mean_validation_mae']:.6f}",
    f"Median validation MAE: {agg_row['median_validation_mae']:.6f}",
    f"Mean validation RMSE: {agg_row['mean_validation_rmse']:.6f}",
    f"Mean validation R²: {agg_row['mean_validation_r2']:.6f}",
    f"Median validation R²: {agg_row['median_validation_r2']:.6f}",
    f"Negative-R² folds: {int(agg_row['negative_r2_folds'])}",
    f"Mean MAE generalization gap: {agg_row['mean_mae_generalization_gap']:.6f}",
    "",
    "PROBABILISTIC CANDIDATE DIAGNOSTICS",
    "-" * 108,
    f"Mean validation NLL: {agg_row['mean_validation_nll']:.6f}",
    f"Mean predicted std: {agg_row['mean_validation_predicted_std']:.6f} °C",
    f"Mean observed coverage for nominal 80% interval: {agg_row['mean_validation_coverage_80'] * 100:.2f}%",
    f"Mean observed coverage for nominal 90% interval: {agg_row['mean_validation_coverage_90'] * 100:.2f}%",
    f"Mean observed coverage for nominal 95% interval: {agg_row['mean_validation_coverage_95'] * 100:.2f}%",
    f"Mean width of nominal 80% interval: {agg_row['mean_validation_width_80']:.6f} °C",
    f"Mean width of nominal 90% interval: {agg_row['mean_validation_width_90']:.6f} °C",
    f"Mean width of nominal 95% interval: {agg_row['mean_validation_width_95']:.6f} °C",
    "",
    "BASELINE COMPARISON",
    "-" * 108,
    (
        f"NGBoost beats the best Step 11 baseline by validation MAE in "
        f"{ngb_beats_best_count}/{len(comparison_df)} folds."
    ),
    (
        f"NGBoost beats Grid Historical Mean by validation MAE in "
        f"{ngb_beats_grid_count}/{len(comparison_df)} folds."
    ),
    "",
    "DEVELOPMENT REFERENCES",
    "-" * 108,
    (
        f"NGBoost has lower validation MAE than Random Forest in "
        f"{ngb_lower_rf_count}/{len(model_reference)} folds."
    ),
    (
        f"NGBoost has lower validation MAE than XGBoost in "
        f"{ngb_lower_xgb_count}/{len(model_reference)} folds."
    ),
    (
        f"NGBoost has lower validation MAE than LightGBM in "
        f"{ngb_lower_lgbm_count}/{len(model_reference)} folds."
    ),
    "These are descriptive only. Formal candidate comparison is Step 16.",
    "",
    "IMPORTANT",
    "-" * 108,
    "This step does NOT tune NGBoost.",
    "This step does NOT calibrate NGBoost uncertainty.",
    "Raw NGBoost intervals must not be treated as final uncertainty intervals.",
    "This step does NOT select NGBoost as the final model or final uncertainty method.",
    "Negative R², poor coverage, wide intervals or generalization gaps are retained as research findings.",
    "",
    "NEXT",
    "-" * 108,
    "Proceed to Step 16 Candidate Model Comparison only after reviewing",
    "the complete Step 15 output and resolving any Step 15 issues.",
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
print(
    f"Saved: {HASH_PATH}"
)


# =============================================================================
# 19. FINAL STATUS
# =============================================================================

section(
    "STEP 15 COMPLETED SUCCESSFULLY"
)

print(
    "Untuned NGBoost probabilistic candidate analysis completed using 2018-2023 only.\n"
    "2024 remains untouched for reserved validation.\n"
    "2025 remains untouched as the final lockbox.\n"
    "Five rolling-origin folds were evaluated.\n"
    "Point metrics, residuals, raw probabilistic intervals, coverage, NLL, "
    "baseline comparison, prior-model references and permutation importance "
    "were saved.\n"
    "No hyperparameter tuning, uncertainty calibration, or final model "
    "selection was performed.\n"
    "Review the complete Step 15 output before Step 16."
)
