"""
STEP 19B1 - LEAKAGE-SAFE TEMPORAL BIAS CORRECTION
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Step 19A showed that the strongest temporal failure is annual-level prediction
bias. This experiment tests whether that bias can be reduced using ONLY bias
information available from EARLIER rolling-origin validation years.

Research protections
--------------------
- Uses finalized Step 17 rolling predictions for 2019-2023 only.
- Eligible corrections use only earlier out-of-sample annual residual bias.
- The current validation year's target is NEVER used to construct an eligible
  correction.
- 2024 is NOT loaded or used.
- 2025 is NOT loaded or used.
- No hyperparameter tuning.
- No feature selection.
- No model retraining.
- No final model selection.

Eligible correction rules
-------------------------
1. raw_no_correction
2. prior_mean_residual
3. prior_median_residual
4. last_prior_residual
5. prior_ewma_residual_alpha_0_50

For validation year t, annual residual bias is:

    b_y = mean(actual_y - predicted_y)

using only earlier validation years y < t. The correction is added to every grid
prediction in year t:

    corrected_prediction = raw_prediction + estimated_prior_residual_bias

2019 has no earlier out-of-sample validation year, so all eligible correction
rules use zero correction for 2019. Therefore strategy comparison is reported
for both:
- all five folds (2019-2023)
- correction-eligible folds only (2020-2023)

Oracle diagnostic
-----------------
A CURRENT-YEAR mean-residual shift is calculated only as an ORACLE diagnostic.
It uses the current year's target, so it is explicitly INELIGIBLE for strategy
selection or deployment. It should reproduce Step 19A centered spatial R² and
shows the theoretical gain available if the annual thermal level were known.
"""

from pathlib import Path
import hashlib
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# 1. SETTINGS
# =============================================================================

GRID_COL = "grid_id"

MODEL_ORDER = [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "NGBoost",
]

VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]
CORRECTION_ELIGIBLE_YEARS = [2020, 2021, 2022, 2023]

EXPECTED_UNIQUE_GRIDS = 2194
EXPECTED_RAW_PREDICTION_ROWS = (
    len(MODEL_ORDER) * len(VALIDATION_YEARS) * EXPECTED_UNIQUE_GRIDS
)

EWMA_ALPHA = 0.50

ELIGIBLE_STRATEGIES = [
    "raw_no_correction",
    "prior_mean_residual",
    "prior_median_residual",
    "last_prior_residual",
    "prior_ewma_residual_alpha_0_50",
]

ORACLE_STRATEGY = "oracle_current_year_mean_residual_DIAGNOSTIC_ONLY"

METRIC_TOLERANCE = 1e-9
ORACLE_CENTERED_R2_TOLERANCE = 1e-9


# =============================================================================
# 2. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEP17_PREDICTIONS_PATH = (
    PROJECT_ROOT / "outputs" / "predictions" / "17_best_tuned_rolling_predictions.csv"
)
STEP17_FOLD_METRICS_PATH = (
    PROJECT_ROOT / "outputs" / "metrics" / "17_best_config_fold_metrics.csv"
)
STEP17_AUDIT_PATH = (
    PROJECT_ROOT / "outputs" / "reports" / "17_hyperparameter_tuning_audit.csv"
)
STEP19A_AUDIT_PATH = (
    PROJECT_ROOT / "outputs" / "reports" / "19a_temporal_diagnosis_audit.csv"
)
STEP19A_YEAR_DIAG_PATH = (
    PROJECT_ROOT / "outputs" / "metrics" / "19a_year_level_prediction_diagnostics.csv"
)

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

for folder in [METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR, FIGURES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

CORRECTION_HISTORY_PATH = METRICS_DIR / "19b1_bias_correction_history.csv"
FOLD_METRICS_PATH = METRICS_DIR / "19b1_bias_correction_fold_metrics.csv"
SUMMARY_PATH = METRICS_DIR / "19b1_bias_correction_summary.csv"
BEST_BY_MODEL_PATH = METRICS_DIR / "19b1_best_bias_correction_by_model.csv"
ORACLE_DIAGNOSTIC_PATH = METRICS_DIR / "19b1_oracle_level_shift_diagnostic.csv"

BASELINE_REPRO_PATH = REPORTS_DIR / "19b1_raw_baseline_reproduction_check.csv"
PREDICTIONS_PATH = PREDICTIONS_DIR / "19b1_bias_correction_predictions.csv"
AUDIT_PATH = REPORTS_DIR / "19b1_temporal_bias_correction_audit.csv"
REPORT_PATH = REPORTS_DIR / "19b1_temporal_bias_correction_report.txt"
HASH_PATH = REPORTS_DIR / "19b1_temporal_bias_correction_hashes.csv"

FIG_MAE_PATH = FIGURES_DIR / "19b1_bias_correction_mean_mae.png"
FIG_R2_BY_YEAR_PATH = FIGURES_DIR / "19b1_bias_correction_r2_by_year.png"
FIG_CORRECTION_PATH = FIGURES_DIR / "19b1_applied_correction_by_year.png"
FIG_ORACLE_GAP_PATH = FIGURES_DIR / "19b1_raw_vs_oracle_r2.png"


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 126)
    print(title)
    print("=" * 126)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_audit_all_pass(path: Path, label: str) -> None:
    if not path.exists():
        print(f"\nERROR: required {label} audit is missing: {path}")
        sys.exit(1)

    audit = pd.read_csv(path)
    if "status" not in audit.columns:
        print(f"\nERROR: {label} audit has no 'status' column.")
        sys.exit(1)

    bad = audit[audit["status"].astype(str).str.upper() != "PASS"]
    if not bad.empty:
        print(f"\nERROR: {label} audit is not fully PASS.")
        print(bad.to_string(index=False))
        sys.exit(1)


def mape_percent(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.abs(y_true) > 1e-12
    if not np.any(valid):
        return np.nan
    return float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100.0)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": mape_percent(y_true, y_pred),
        "mean_residual_actual_minus_predicted": float(np.mean(residual)),
        "mean_prediction_bias_pred_minus_actual": float(np.mean(y_pred - y_true)),
    }


def ewma_prior_residual(residual_history, alpha: float) -> float:
    values = [float(v) for v in residual_history]
    if not values:
        return 0.0
    estimate = values[0]
    for value in values[1:]:
        estimate = alpha * value + (1.0 - alpha) * estimate
    return float(estimate)


def get_correction(strategy: str, prior_residual_means) -> float:
    prior = np.asarray(prior_residual_means, dtype=float)

    if strategy == "raw_no_correction":
        return 0.0
    if len(prior) == 0:
        return 0.0
    if strategy == "prior_mean_residual":
        return float(np.mean(prior))
    if strategy == "prior_median_residual":
        return float(np.median(prior))
    if strategy == "last_prior_residual":
        return float(prior[-1])
    if strategy == "prior_ewma_residual_alpha_0_50":
        return ewma_prior_residual(prior, EWMA_ALPHA)

    raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# 4. PREREQUISITE AUDITS
# =============================================================================

section("STEP 19B1 - LEAKAGE-SAFE TEMPORAL BIAS CORRECTION")

required_files = [
    STEP17_PREDICTIONS_PATH,
    STEP17_FOLD_METRICS_PATH,
    STEP17_AUDIT_PATH,
    STEP19A_AUDIT_PATH,
    STEP19A_YEAR_DIAG_PATH,
]

for path in required_files:
    if not path.exists():
        print(f"\nERROR: required file is missing: {path}")
        sys.exit(1)

check_audit_all_pass(STEP17_AUDIT_PATH, "Step 17")
check_audit_all_pass(STEP19A_AUDIT_PATH, "Step 19A")

print("PASS: Step 17 audit fully PASS.")
print("PASS: Step 19A audit fully PASS.")
print("19B1 tests only predefined leakage-safe bias-correction rules.")
print("2024 and 2025 are not loaded.")


# =============================================================================
# 5. LOAD + VERIFY FINALIZED STEP 17 ROLLING PREDICTIONS
# =============================================================================

section("1. LOAD STEP 17 ROLLING PREDICTIONS")

pred = pd.read_csv(STEP17_PREDICTIONS_PATH, dtype={GRID_COL: str})

required_prediction_columns = {
    "model",
    GRID_COL,
    "validation_year",
    "actual_lst_c",
    "predicted_lst_c",
    "residual_actual_minus_predicted",
}

missing_prediction_columns = sorted(required_prediction_columns - set(pred.columns))
if missing_prediction_columns:
    print(f"\nERROR: Step 17 prediction file is missing: {missing_prediction_columns}")
    sys.exit(1)

pred["validation_year"] = pd.to_numeric(pred["validation_year"], errors="coerce").astype(int)
for column in ["actual_lst_c", "predicted_lst_c", "residual_actual_minus_predicted"]:
    pred[column] = pd.to_numeric(pred[column], errors="coerce")

if len(pred) != EXPECTED_RAW_PREDICTION_ROWS:
    print(
        f"\nERROR: prediction rows={len(pred):,}; "
        f"expected={EXPECTED_RAW_PREDICTION_ROWS:,}."
    )
    sys.exit(1)

actual_models = sorted(pred["model"].unique().tolist())
if actual_models != sorted(MODEL_ORDER):
    print(f"\nERROR: models={actual_models}; expected={sorted(MODEL_ORDER)}.")
    sys.exit(1)

actual_years = sorted(pred["validation_year"].unique().tolist())
if actual_years != VALIDATION_YEARS:
    print(f"\nERROR: validation years={actual_years}; expected={VALIDATION_YEARS}.")
    sys.exit(1)

if not np.isfinite(
    pred[["actual_lst_c", "predicted_lst_c", "residual_actual_minus_predicted"]]
    .to_numpy(dtype=float)
).all():
    print("\nERROR: non-finite prediction values found.")
    sys.exit(1)

key_counts = pred.groupby(["model", GRID_COL, "validation_year"]).size()
if not (len(key_counts) == EXPECTED_RAW_PREDICTION_ROWS and key_counts.eq(1).all()):
    print("\nERROR: duplicate or missing model-grid-year keys.")
    sys.exit(1)

residual_reconstruction_error = float(
    np.max(
        np.abs(
            (pred["actual_lst_c"] - pred["predicted_lst_c"])
            - pred["residual_actual_minus_predicted"]
        )
    )
)

if residual_reconstruction_error > METRIC_TOLERANCE:
    print("\nERROR: saved residual column does not equal actual - predicted.")
    print(f"Maximum reconstruction error: {residual_reconstruction_error}")
    sys.exit(1)

print(f"Rows            : {len(pred):,}")
print(f"Models          : {actual_models}")
print(f"Years           : {actual_years}")
print(f"Residual max Δ  : {residual_reconstruction_error:.12f}")
print("2024            : NOT LOADED")
print("2025            : NOT LOADED")


# =============================================================================
# 6. VERIFY RAW PREDICTIONS STILL REPRODUCE STEP 17
# =============================================================================

section("2. RAW BASELINE REPRODUCTION CHECK")

step17_fold = pd.read_csv(STEP17_FOLD_METRICS_PATH)
required_step17_columns = {
    "model",
    "validation_year",
    "validation_mae",
    "validation_rmse",
    "validation_r2",
}

missing_step17_columns = sorted(required_step17_columns - set(step17_fold.columns))
if missing_step17_columns:
    print(f"\nERROR: Step 17 fold metric file missing: {missing_step17_columns}")
    sys.exit(1)

baseline_repro_rows = []

for model_name in MODEL_ORDER:
    for year in VALIDATION_YEARS:
        subset = pred[(pred["model"] == model_name) & (pred["validation_year"] == year)]
        metrics = regression_metrics(subset["actual_lst_c"], subset["predicted_lst_c"])

        ref = step17_fold[
            (step17_fold["model"] == model_name)
            & (step17_fold["validation_year"] == year)
        ]
        if len(ref) != 1:
            print(
                f"\nERROR: expected exactly one Step 17 row for "
                f"{model_name}, {year}; found {len(ref)}."
            )
            sys.exit(1)

        ref = ref.iloc[0]
        mae_diff = metrics["mae"] - float(ref["validation_mae"])
        rmse_diff = metrics["rmse"] - float(ref["validation_rmse"])
        r2_diff = metrics["r2"] - float(ref["validation_r2"])

        status = "PASS" if (
            abs(mae_diff) <= METRIC_TOLERANCE
            and abs(rmse_diff) <= METRIC_TOLERANCE
            and abs(r2_diff) <= METRIC_TOLERANCE
        ) else "FAIL"

        baseline_repro_rows.append({
            "model": model_name,
            "validation_year": year,
            "step17_mae": float(ref["validation_mae"]),
            "recomputed_mae": metrics["mae"],
            "mae_difference": mae_diff,
            "step17_rmse": float(ref["validation_rmse"]),
            "recomputed_rmse": metrics["rmse"],
            "rmse_difference": rmse_diff,
            "step17_r2": float(ref["validation_r2"]),
            "recomputed_r2": metrics["r2"],
            "r2_difference": r2_diff,
            "status": status,
        })

baseline_repro = pd.DataFrame(baseline_repro_rows)
baseline_repro.to_csv(BASELINE_REPRO_PATH, index=False)
print(baseline_repro.to_string(index=False))

if (baseline_repro["status"] != "PASS").any():
    print("\nERROR: raw Step 17 prediction reproduction failed.")
    sys.exit(1)

print("\nPASS: all 20 raw model-year predictions reproduce Step 17.")


# =============================================================================
# 7. BUILD STRICT PRIOR-YEAR BIAS HISTORY
# =============================================================================

section("3. BUILD PRIOR OUT-OF-SAMPLE BIAS HISTORY")

annual_raw_bias = (
    pred.groupby(["model", "validation_year"], as_index=False)
    .agg(
        annual_mean_residual_actual_minus_predicted=(
            "residual_actual_minus_predicted",
            "mean",
        ),
        actual_year_mean_c=("actual_lst_c", "mean"),
        raw_predicted_year_mean_c=("predicted_lst_c", "mean"),
    )
)

history_rows = []

for model_name in MODEL_ORDER:
    model_bias = (
        annual_raw_bias[annual_raw_bias["model"] == model_name]
        .sort_values("validation_year")
        .copy()
    )
    bias_lookup = dict(zip(
        model_bias["validation_year"],
        model_bias["annual_mean_residual_actual_minus_predicted"],
    ))

    for year in VALIDATION_YEARS:
        prior_years = [prior_year for prior_year in VALIDATION_YEARS if prior_year < year]
        prior_residuals = [float(bias_lookup[prior_year]) for prior_year in prior_years]

        for strategy in ELIGIBLE_STRATEGIES:
            correction = get_correction(strategy, prior_residuals)
            max_source_year = max(prior_years) if prior_years else np.nan
            leakage_safe = bool((not prior_years) or (max_source_year < year))

            history_rows.append({
                "model": model_name,
                "validation_year": year,
                "strategy": strategy,
                "strategy_eligible_for_comparison": True,
                "prior_history_year_count": len(prior_years),
                "prior_history_years": (
                    ",".join(str(prior_year) for prior_year in prior_years)
                    if prior_years else ""
                ),
                "max_source_year": max_source_year,
                "uses_current_validation_year_target": False,
                "leakage_safe_source_year_check": leakage_safe,
                "estimated_correction_actual_minus_predicted_c": correction,
            })

history_df = pd.DataFrame(history_rows)
history_df.to_csv(CORRECTION_HISTORY_PATH, index=False)
print(history_df.to_string(index=False))

if not history_df["leakage_safe_source_year_check"].all():
    print("\nERROR: at least one correction uses a non-prior source year.")
    sys.exit(1)

print(
    "\nPASS: every eligible correction uses only earlier "
    "out-of-sample validation-year bias."
)


# =============================================================================
# 8. APPLY ELIGIBLE CORRECTIONS
# =============================================================================

section("4. APPLY ELIGIBLE CORRECTIONS")

prediction_frames = []
fold_metric_rows = []

for model_name in MODEL_ORDER:
    print(f"\n--- {model_name} ---")
    model_pred = pred[pred["model"] == model_name].copy()

    for strategy in ELIGIBLE_STRATEGIES:
        print(f"\nStrategy: {strategy}")

        for year in VALIDATION_YEARS:
            subset = model_pred[model_pred["validation_year"] == year].copy()

            history_row = history_df[
                (history_df["model"] == model_name)
                & (history_df["validation_year"] == year)
                & (history_df["strategy"] == strategy)
            ]

            if len(history_row) != 1:
                print("\nERROR: expected exactly one correction history row.")
                sys.exit(1)

            history_row = history_row.iloc[0]
            correction = float(
                history_row["estimated_correction_actual_minus_predicted_c"]
            )

            raw_prediction = subset["predicted_lst_c"].to_numpy(dtype=float)
            corrected_prediction = raw_prediction + correction
            y_true = subset["actual_lst_c"].to_numpy(dtype=float)

            metrics = regression_metrics(y_true, corrected_prediction)
            raw_metrics = regression_metrics(y_true, raw_prediction)

            prediction_frames.append(pd.DataFrame({
                "model": model_name,
                GRID_COL: subset[GRID_COL].astype(str).to_numpy(),
                "validation_year": year,
                "strategy": strategy,
                "strategy_eligible_for_comparison": True,
                "prior_history_year_count": int(history_row["prior_history_year_count"]),
                "correction_actual_minus_predicted_c": correction,
                "actual_lst_c": y_true,
                "raw_predicted_lst_c": raw_prediction,
                "corrected_predicted_lst_c": corrected_prediction,
                "raw_residual_actual_minus_predicted": y_true - raw_prediction,
                "corrected_residual_actual_minus_predicted": y_true - corrected_prediction,
            }))

            fold_metric_rows.append({
                "model": model_name,
                "validation_year": year,
                "strategy": strategy,
                "strategy_eligible_for_comparison": True,
                "correction_eligible_year": bool(year in CORRECTION_ELIGIBLE_YEARS),
                "prior_history_year_count": int(history_row["prior_history_year_count"]),
                "correction_actual_minus_predicted_c": correction,
                "raw_mae": raw_metrics["mae"],
                "corrected_mae": metrics["mae"],
                "mae_improvement_raw_minus_corrected": raw_metrics["mae"] - metrics["mae"],
                "raw_rmse": raw_metrics["rmse"],
                "corrected_rmse": metrics["rmse"],
                "rmse_improvement_raw_minus_corrected": raw_metrics["rmse"] - metrics["rmse"],
                "raw_r2": raw_metrics["r2"],
                "corrected_r2": metrics["r2"],
                "r2_improvement_corrected_minus_raw": metrics["r2"] - raw_metrics["r2"],
                "corrected_mape_percent": metrics["mape_percent"],
                "raw_mean_prediction_bias_pred_minus_actual": raw_metrics[
                    "mean_prediction_bias_pred_minus_actual"
                ],
                "corrected_mean_prediction_bias_pred_minus_actual": metrics[
                    "mean_prediction_bias_pred_minus_actual"
                ],
                "corrected_negative_r2": bool(metrics["r2"] < 0),
            })

            print(
                f"{year} | correction={correction:+.3f} °C | "
                f"MAE {raw_metrics['mae']:.3f} -> {metrics['mae']:.3f} | "
                f"R² {raw_metrics['r2']:.4f} -> {metrics['r2']:.4f}"
            )

eligible_predictions = pd.concat(prediction_frames, ignore_index=True)
eligible_predictions.to_csv(PREDICTIONS_PATH, index=False)

fold_metrics = pd.DataFrame(fold_metric_rows)
fold_metrics.to_csv(FOLD_METRICS_PATH, index=False)


# =============================================================================
# 9. SUMMARY OF ELIGIBLE STRATEGIES
# =============================================================================

section("5. ELIGIBLE STRATEGY SUMMARY")

summary_rows = []

for model_name in MODEL_ORDER:
    for strategy in ELIGIBLE_STRATEGIES:
        subset_all = fold_metrics[
            (fold_metrics["model"] == model_name)
            & (fold_metrics["strategy"] == strategy)
        ]
        subset_eligible = subset_all[subset_all["correction_eligible_year"]]

        if len(subset_all) != len(VALIDATION_YEARS):
            print("\nERROR: incomplete all-fold strategy summary input.")
            sys.exit(1)
        if len(subset_eligible) != len(CORRECTION_ELIGIBLE_YEARS):
            print("\nERROR: incomplete correction-eligible strategy summary input.")
            sys.exit(1)

        summary_rows.append({
            "model": model_name,
            "strategy": strategy,
            "strategy_eligible_for_comparison": True,
            "all_folds_mean_mae": float(subset_all["corrected_mae"].mean()),
            "all_folds_std_mae": float(subset_all["corrected_mae"].std(ddof=0)),
            "all_folds_mean_rmse": float(subset_all["corrected_rmse"].mean()),
            "all_folds_mean_r2": float(subset_all["corrected_r2"].mean()),
            "all_folds_negative_r2_count": int(subset_all["corrected_negative_r2"].sum()),
            "all_folds_worst_mae": float(subset_all["corrected_mae"].max()),
            "eligible_2020_2023_mean_mae": float(subset_eligible["corrected_mae"].mean()),
            "eligible_2020_2023_std_mae": float(subset_eligible["corrected_mae"].std(ddof=0)),
            "eligible_2020_2023_mean_rmse": float(subset_eligible["corrected_rmse"].mean()),
            "eligible_2020_2023_mean_r2": float(subset_eligible["corrected_r2"].mean()),
            "eligible_2020_2023_negative_r2_count": int(
                subset_eligible["corrected_negative_r2"].sum()
            ),
            "eligible_2020_2023_worst_mae": float(subset_eligible["corrected_mae"].max()),
            "eligible_2020_2023_mean_mae_improvement_vs_raw": float(
                subset_eligible["mae_improvement_raw_minus_corrected"].mean()
            ),
            "eligible_2020_2023_mean_rmse_improvement_vs_raw": float(
                subset_eligible["rmse_improvement_raw_minus_corrected"].mean()
            ),
            "eligible_2020_2023_mean_r2_improvement_vs_raw": float(
                subset_eligible["r2_improvement_corrected_minus_raw"].mean()
            ),
        })

summary = pd.DataFrame(summary_rows)
summary["within_model_strategy_rank"] = (
    summary.groupby("model")["eligible_2020_2023_mean_mae"]
    .rank(method="min", ascending=True)
    .astype(int)
)
summary = summary.sort_values(
    [
        "model",
        "within_model_strategy_rank",
        "eligible_2020_2023_worst_mae",
        "eligible_2020_2023_mean_rmse",
    ]
).reset_index(drop=True)
summary.to_csv(SUMMARY_PATH, index=False)
print(summary.to_string(index=False))

best_by_model = (
    summary.sort_values(
        [
            "model",
            "eligible_2020_2023_mean_mae",
            "eligible_2020_2023_worst_mae",
            "eligible_2020_2023_mean_rmse",
            "eligible_2020_2023_negative_r2_count",
        ],
        ascending=[True, True, True, True, True],
    )
    .groupby("model", as_index=False)
    .first()
)
best_by_model.to_csv(BEST_BY_MODEL_PATH, index=False)

print(
    "\nBEST PREDEFINED BIAS-CORRECTION RULE WITHIN EACH MODEL "
    "(development experiment only):"
)
print(
    best_by_model[
        [
            "model",
            "strategy",
            "eligible_2020_2023_mean_mae",
            "eligible_2020_2023_mean_rmse",
            "eligible_2020_2023_mean_r2",
            "eligible_2020_2023_negative_r2_count",
            "eligible_2020_2023_worst_mae",
            "eligible_2020_2023_mean_mae_improvement_vs_raw",
        ]
    ].to_string(index=False)
)
print(
    "\nIMPORTANT: this is NOT final model selection. It only identifies "
    "the best predefined correction rule inside this 19B1 development experiment."
)


# =============================================================================
# 10. ORACLE CURRENT-YEAR LEVEL SHIFT - DIAGNOSTIC ONLY
# =============================================================================

section("6. ORACLE ANNUAL-LEVEL SHIFT - DIAGNOSTIC ONLY")

step19a_diag = pd.read_csv(STEP19A_YEAR_DIAG_PATH)
required_19a_cols = {"model", "validation_year", "centered_spatial_r2"}
missing_19a_cols = sorted(required_19a_cols - set(step19a_diag.columns))
if missing_19a_cols:
    print(f"\nERROR: Step 19A diagnostic file missing: {missing_19a_cols}")
    sys.exit(1)

oracle_rows = []

for model_name in MODEL_ORDER:
    for year in VALIDATION_YEARS:
        subset = pred[
            (pred["model"] == model_name)
            & (pred["validation_year"] == year)
        ]

        y_true = subset["actual_lst_c"].to_numpy(dtype=float)
        raw_pred = subset["predicted_lst_c"].to_numpy(dtype=float)
        oracle_correction = float(np.mean(y_true - raw_pred))
        oracle_pred = raw_pred + oracle_correction

        raw_metrics = regression_metrics(y_true, raw_pred)
        oracle_metrics = regression_metrics(y_true, oracle_pred)

        diag_row = step19a_diag[
            (step19a_diag["model"] == model_name)
            & (step19a_diag["validation_year"] == year)
        ]
        if len(diag_row) != 1:
            print(
                f"\nERROR: expected one Step 19A diagnostic row for "
                f"{model_name}, {year}."
            )
            sys.exit(1)

        centered_r2 = float(diag_row.iloc[0]["centered_spatial_r2"])
        oracle_vs_centered_diff = oracle_metrics["r2"] - centered_r2

        oracle_rows.append({
            "model": model_name,
            "validation_year": year,
            "strategy": ORACLE_STRATEGY,
            "strategy_eligible_for_comparison": False,
            "uses_current_validation_year_target": True,
            "oracle_correction_actual_minus_predicted_c": oracle_correction,
            "raw_mae": raw_metrics["mae"],
            "oracle_mae": oracle_metrics["mae"],
            "raw_rmse": raw_metrics["rmse"],
            "oracle_rmse": oracle_metrics["rmse"],
            "raw_r2": raw_metrics["r2"],
            "oracle_r2": oracle_metrics["r2"],
            "step19a_centered_spatial_r2": centered_r2,
            "oracle_minus_centered_r2_difference": oracle_vs_centered_diff,
            "oracle_negative_r2": bool(oracle_metrics["r2"] < 0),
        })

oracle_df = pd.DataFrame(oracle_rows)
oracle_df.to_csv(ORACLE_DIAGNOSTIC_PATH, index=False)
print(oracle_df.to_string(index=False))

oracle_repro_max_diff = float(
    np.max(np.abs(oracle_df["oracle_minus_centered_r2_difference"]))
)
if oracle_repro_max_diff > ORACLE_CENTERED_R2_TOLERANCE:
    print(
        "\nERROR: oracle annual-level shift does not reproduce "
        "Step 19A centered spatial R²."
    )
    print(f"Max difference={oracle_repro_max_diff}")
    sys.exit(1)

print(
    "\nPASS: oracle level shift reproduces Step 19A centered spatial-pattern R²."
)
print(
    "REMINDER: oracle result uses CURRENT-YEAR TARGET information and is "
    "DIAGNOSTIC ONLY. It is excluded from eligible strategy rankings."
)


# =============================================================================
# 11. FIGURES
# =============================================================================

section("7. FIGURES")

# Mean eligible-fold MAE by correction strategy
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(ELIGIBLE_STRATEGIES))
width = 0.8 / len(MODEL_ORDER)

for model_index, model_name in enumerate(MODEL_ORDER):
    model_summary = (
        summary[summary["model"] == model_name]
        .set_index("strategy")
        .reindex(ELIGIBLE_STRATEGIES)
    )
    offset = (model_index - (len(MODEL_ORDER) - 1) / 2) * width
    ax.bar(
        x + offset,
        model_summary["eligible_2020_2023_mean_mae"],
        width,
        label=model_name,
    )

ax.set_xticks(x, ELIGIBLE_STRATEGIES, rotation=20, ha="right")
ax.set_ylabel("Mean MAE, correction-eligible years 2020-2023 (°C)")
ax.set_title("Step 19B1 - Leakage-Safe Bias-Correction Comparison")
ax.grid(axis="y", alpha=0.2)
ax.legend()
plt.tight_layout()
plt.savefig(FIG_MAE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

# R² by year for best eligible correction per model
fig, ax = plt.subplots(figsize=(11, 7))
for model_name in MODEL_ORDER:
    best_strategy = best_by_model.loc[
        best_by_model["model"] == model_name,
        "strategy",
    ].iloc[0]

    model_fold = fold_metrics[
        (fold_metrics["model"] == model_name)
        & (fold_metrics["strategy"] == best_strategy)
    ].sort_values("validation_year")

    ax.plot(
        model_fold["validation_year"],
        model_fold["corrected_r2"],
        marker="o",
        label=f"{model_name} - {best_strategy}",
    )

ax.axhline(0, linewidth=1)
ax.set_xticks(VALIDATION_YEARS)
ax.set_xlabel("Validation year")
ax.set_ylabel("R²")
ax.set_title("Step 19B1 - R² of Best Eligible Bias-Correction Rule per Model")
ax.grid(alpha=0.2)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_R2_BY_YEAR_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

# Applied correction by year
fig, ax = plt.subplots(figsize=(11, 7))
for strategy in ELIGIBLE_STRATEGIES:
    strategy_history = (
        history_df[history_df["strategy"] == strategy]
        .groupby("validation_year", as_index=False)[
            "estimated_correction_actual_minus_predicted_c"
        ]
        .mean()
    )
    ax.plot(
        strategy_history["validation_year"],
        strategy_history["estimated_correction_actual_minus_predicted_c"],
        marker="o",
        label=strategy,
    )

ax.axhline(0, linewidth=1)
ax.set_xticks(VALIDATION_YEARS)
ax.set_xlabel("Validation year")
ax.set_ylabel("Mean applied correction across models (°C)")
ax.set_title("Step 19B1 - Leakage-Safe Applied Annual Bias Correction")
ax.grid(alpha=0.2)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_CORRECTION_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

# Raw versus oracle R²
oracle_plot = (
    oracle_df.groupby("validation_year", as_index=False)
    .agg(
        raw_r2=("raw_r2", "mean"),
        oracle_r2=("oracle_r2", "mean"),
    )
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    oracle_plot["validation_year"],
    oracle_plot["raw_r2"],
    marker="o",
    label="Raw tuned prediction",
)
ax.plot(
    oracle_plot["validation_year"],
    oracle_plot["oracle_r2"],
    marker="o",
    label="Oracle current-year level shift - diagnostic only",
)
ax.axhline(0, linewidth=1)
ax.set_xticks(VALIDATION_YEARS)
ax.set_xlabel("Validation year")
ax.set_ylabel("Mean R² across models")
ax.set_title("Step 19B1 - Raw vs Oracle Annual-Level Shift")
ax.grid(alpha=0.2)
ax.legend()
plt.tight_layout()
plt.savefig(FIG_ORACLE_GAP_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

for figure_path in [FIG_MAE_PATH, FIG_R2_BY_YEAR_PATH, FIG_CORRECTION_PATH, FIG_ORACLE_GAP_PATH]:
    print(f"Saved: {figure_path}")


# =============================================================================
# 12. STRICT AUDIT
# =============================================================================

section("8. STRICT STEP 19B1 AUDIT")

expected_history_rows = len(MODEL_ORDER) * len(VALIDATION_YEARS) * len(ELIGIBLE_STRATEGIES)
expected_fold_metric_rows = expected_history_rows
expected_prediction_rows = EXPECTED_RAW_PREDICTION_ROWS * len(ELIGIBLE_STRATEGIES)
expected_summary_rows = len(MODEL_ORDER) * len(ELIGIBLE_STRATEGIES)
expected_oracle_rows = len(MODEL_ORDER) * len(VALIDATION_YEARS)

history_year_safety = bool(history_df["leakage_safe_source_year_check"].all())
current_year_target_used_by_eligible = bool(
    history_df["uses_current_validation_year_target"].any()
)
oracle_excluded_from_eligible = bool(
    not oracle_df["strategy_eligible_for_comparison"].astype(bool).any()
)

all_metrics_finite = bool(
    np.isfinite(
        fold_metrics[
            [
                "correction_actual_minus_predicted_c",
                "raw_mae",
                "corrected_mae",
                "raw_rmse",
                "corrected_rmse",
                "raw_r2",
                "corrected_r2",
                "corrected_mape_percent",
            ]
        ].to_numpy(dtype=float)
    ).all()
)

raw_strategy_rows = fold_metrics[fold_metrics["strategy"] == "raw_no_correction"]
raw_strategy_no_change = bool(
    (np.abs(raw_strategy_rows["mae_improvement_raw_minus_corrected"]) <= METRIC_TOLERANCE).all()
    and (np.abs(raw_strategy_rows["rmse_improvement_raw_minus_corrected"]) <= METRIC_TOLERANCE).all()
    and (np.abs(raw_strategy_rows["r2_improvement_corrected_minus_raw"]) <= METRIC_TOLERANCE).all()
)

year_2019_zero_corrections = bool(
    (
        np.abs(
            history_df.loc[
                history_df["validation_year"] == 2019,
                "estimated_correction_actual_minus_predicted_c",
            ]
        )
        <= METRIC_TOLERANCE
    ).all()
)

audit_rows = [
    {
        "check": "step17_audit",
        "status": "PASS",
        "details": "Step 17 audit loaded and fully PASS.",
    },
    {
        "check": "step19a_audit",
        "status": "PASS",
        "details": "Step 19A audit loaded and fully PASS.",
    },
    {
        "check": "raw_prediction_rows",
        "status": "PASS" if len(pred) == EXPECTED_RAW_PREDICTION_ROWS else "FAIL",
        "details": f"actual={len(pred)}, expected={EXPECTED_RAW_PREDICTION_ROWS}",
    },
    {
        "check": "raw_step17_metric_reproduction",
        "status": "PASS" if (baseline_repro["status"] == "PASS").all() else "FAIL",
        "details": f"PASS rows={int((baseline_repro['status'] == 'PASS').sum())}/20",
    },
    {
        "check": "residual_reconstruction",
        "status": "PASS" if residual_reconstruction_error <= METRIC_TOLERANCE else "FAIL",
        "details": f"max_abs_error={residual_reconstruction_error:.12f}",
    },
    {
        "check": "correction_history_rows",
        "status": "PASS" if len(history_df) == expected_history_rows else "FAIL",
        "details": f"actual={len(history_df)}, expected={expected_history_rows}",
    },
    {
        "check": "strict_prior_year_source_only",
        "status": "PASS" if history_year_safety else "FAIL",
        "details": "Every eligible correction source year is earlier than its validation year.",
    },
    {
        "check": "eligible_current_year_target_used",
        "status": "PASS" if not current_year_target_used_by_eligible else "FAIL",
        "details": "No eligible correction uses the current validation year target.",
    },
    {
        "check": "2019_no_prior_history_zero_correction",
        "status": "PASS" if year_2019_zero_corrections else "FAIL",
        "details": "All eligible strategies use zero correction in 2019.",
    },
    {
        "check": "fold_metric_rows",
        "status": "PASS" if len(fold_metrics) == expected_fold_metric_rows else "FAIL",
        "details": f"actual={len(fold_metrics)}, expected={expected_fold_metric_rows}",
    },
    {
        "check": "prediction_rows",
        "status": "PASS" if len(eligible_predictions) == expected_prediction_rows else "FAIL",
        "details": f"actual={len(eligible_predictions)}, expected={expected_prediction_rows}",
    },
    {
        "check": "summary_rows",
        "status": "PASS" if len(summary) == expected_summary_rows else "FAIL",
        "details": f"actual={len(summary)}, expected={expected_summary_rows}",
    },
    {
        "check": "best_rule_rows",
        "status": "PASS" if len(best_by_model) == len(MODEL_ORDER) else "FAIL",
        "details": f"actual={len(best_by_model)}, expected={len(MODEL_ORDER)}",
    },
    {
        "check": "all_eligible_metrics_finite",
        "status": "PASS" if all_metrics_finite else "FAIL",
        "details": str(all_metrics_finite),
    },
    {
        "check": "raw_strategy_is_identity",
        "status": "PASS" if raw_strategy_no_change else "FAIL",
        "details": "raw_no_correction exactly reproduces raw predictions.",
    },
    {
        "check": "oracle_rows",
        "status": "PASS" if len(oracle_df) == expected_oracle_rows else "FAIL",
        "details": f"actual={len(oracle_df)}, expected={expected_oracle_rows}",
    },
    {
        "check": "oracle_excluded_from_eligible_ranking",
        "status": "PASS" if oracle_excluded_from_eligible else "FAIL",
        "details": "Oracle current-year correction is diagnostic only.",
    },
    {
        "check": "oracle_reproduces_19a_centered_r2",
        "status": "PASS" if oracle_repro_max_diff <= ORACLE_CENTERED_R2_TOLERANCE else "FAIL",
        "details": f"max_abs_difference={oracle_repro_max_diff:.12f}",
    },
    {
        "check": "2024_loaded_or_used",
        "status": "PASS",
        "details": "No. 2024 was not loaded by Step 19B1.",
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": "No. 2025 was not loaded by Step 19B1.",
    },
    {
        "check": "model_retraining_performed",
        "status": "PASS",
        "details": "No. Step 19B1 is a post-processing experiment on finalized Step 17 rolling predictions.",
    },
    {
        "check": "hyperparameter_tuning_performed",
        "status": "PASS",
        "details": "No hyperparameter tuning occurs.",
    },
    {
        "check": "feature_selection_performed",
        "status": "PASS",
        "details": "No feature selection occurs.",
    },
    {
        "check": "final_model_selected",
        "status": "PASS",
        "details": "No final model is selected. Best rule is only within the predefined 19B1 development experiment.",
    },
]

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(AUDIT_PATH, index=False)
print(audit_df.to_string(index=False))

if (audit_df["status"] != "PASS").any():
    print("\nERROR: one or more Step 19B1 audit checks failed.")
    sys.exit(1)


# =============================================================================
# 13. REPORT
# =============================================================================

section("9. SAVE STEP 19B1 REPORT")

report_lines = [
    "STEP 19B1 - LEAKAGE-SAFE TEMPORAL BIAS CORRECTION",
    "=" * 112,
    "",
    "PURPOSE",
    "-" * 112,
    "Test whether annual-level temporal bias can be reduced using only",
    "earlier out-of-sample rolling-origin residual bias.",
    "",
    "DATA POLICY",
    "-" * 112,
    "Step 17 finalized rolling predictions: 2019-2023 only.",
    "2024 reserved validation: NOT loaded.",
    "2025 final lockbox: NOT loaded.",
    "",
    "ELIGIBLE STRATEGIES",
    "-" * 112,
]

report_lines.extend(ELIGIBLE_STRATEGIES)
report_lines.extend([
    "",
    "CORRECTION RULE",
    "-" * 112,
    "For validation year t, eligible corrections use only annual",
    "actual-minus-predicted residual means from earlier validation years.",
    "The current validation year's target is never used to construct an",
    "eligible correction.",
    "",
    "2019 HANDLING",
    "-" * 112,
    "2019 has no earlier out-of-sample rolling-origin residual history.",
    "Therefore every eligible strategy uses zero correction in 2019.",
    "Primary within-experiment comparison also reports 2020-2023 separately.",
    "",
    "BEST PREDEFINED RULE WITHIN EACH MODEL",
    "-" * 112,
])

for _, row in best_by_model.iterrows():
    report_lines.append(
        f"{row['model']}: {row['strategy']} | "
        f"2020-2023 mean MAE={row['eligible_2020_2023_mean_mae']:.6f} °C | "
        f"mean RMSE={row['eligible_2020_2023_mean_rmse']:.6f} °C | "
        f"mean R²={row['eligible_2020_2023_mean_r2']:.6f} | "
        f"negative-R² folds={int(row['eligible_2020_2023_negative_r2_count'])} | "
        f"worst MAE={row['eligible_2020_2023_worst_mae']:.6f} °C | "
        f"mean MAE improvement vs raw="
        f"{row['eligible_2020_2023_mean_mae_improvement_vs_raw']:.6f} °C."
    )

report_lines.extend([
    "",
    "ORACLE DIAGNOSTIC",
    "-" * 112,
    "The oracle current-year mean-residual correction uses the current",
    "year target and is therefore INVALID for deployment or strategy",
    "selection. It is included only to quantify the theoretical benefit",
    "of perfectly knowing the annual thermal-level offset.",
    f"Maximum difference between oracle R² and Step 19A centered R²: "
    f"{oracle_repro_max_diff:.12f}.",
    "",
    "INTERPRETATION",
    "-" * 112,
    "A leakage-safe correction is useful only if historical bias carries",
    "forward reliably. If eligible corrections fail or worsen performance,",
    "that is a genuine finding that annual bias changes too abruptly to be",
    "fixed by simple historical residual calibration.",
    "",
    "NO FINAL SELECTION",
    "-" * 112,
    "Any within-model best rule reported here is only the best predefined",
    "19B1 development rule. It must later be compared against the stable",
    "feature, recent-year weighting, ensemble, and optional level/anomaly",
    "experiments in Step 19C before reserved-2024 validation.",
])

REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
print(f"Saved: {REPORT_PATH}")
print(f"Saved: {AUDIT_PATH}")


# =============================================================================
# 14. REPRODUCIBILITY HASHES
# =============================================================================

section("10. REPRODUCIBILITY HASHES")

hash_files = [
    STEP17_PREDICTIONS_PATH,
    STEP17_FOLD_METRICS_PATH,
    STEP17_AUDIT_PATH,
    STEP19A_AUDIT_PATH,
    STEP19A_YEAR_DIAG_PATH,
    CORRECTION_HISTORY_PATH,
    FOLD_METRICS_PATH,
    SUMMARY_PATH,
    BEST_BY_MODEL_PATH,
    ORACLE_DIAGNOSTIC_PATH,
    BASELINE_REPRO_PATH,
    PREDICTIONS_PATH,
    AUDIT_PATH,
    REPORT_PATH,
]

hash_rows = []
for path in hash_files:
    digest = sha256_file(path)
    hash_rows.append({"file": str(path), "sha256": digest})
    print(f"{path.name}: {digest}")

pd.DataFrame(hash_rows).to_csv(HASH_PATH, index=False)
print(f"Saved: {HASH_PATH}")


# =============================================================================
# 15. FINAL STATUS
# =============================================================================

section("STEP 19B1 COMPLETED SUCCESSFULLY")

print(
    "Leakage-safe temporal bias-correction experiment completed.\n"
    "Raw Step 17 predictions were reproduced before correction.\n"
    "Eligible corrections used only earlier out-of-sample validation-year bias.\n"
    "2019 used zero correction because no earlier OOF residual history exists.\n"
    "2020-2023 were compared as correction-eligible years.\n"
    "The oracle current-year level shift was calculated only as an explicitly\n"
    "ineligible diagnostic upper bound and reproduced Step 19A centered R².\n"
    "2024 reserved validation remained untouched.\n"
    "2025 final lockbox remained untouched.\n"
    "No model retraining, tuning, feature selection, or final model selection occurred.\n"
    "Review the complete output before moving to Step 19B2."
)
