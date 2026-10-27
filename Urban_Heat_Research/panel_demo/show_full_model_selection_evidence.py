from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

# =============================================================================
# URBAN HEAT - FULL MODEL SELECTION PANEL DEMO (READ ONLY)
# =============================================================================
# Recommended location:
#   Urban_Heat_Research/panel_demo/show_full_model_selection_evidence.py
#
# This script only READS already-generated evidence files.
# It does NOT:
#   - train/retrain a model
#   - tune hyperparameters
#   - call original src/*.py scripts
#   - regenerate research outputs
#   - overwrite any CSV/JSON/model file
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FORECASTING_DIR = PROJECT_ROOT / "outputs" / "forecasting"

FILES = {
    "baseline": METRICS_DIR / "16_baseline_summary.csv",
    "candidate_vs_baseline": METRICS_DIR / "16_candidate_vs_fixed_baselines.csv",
    "step16": METRICS_DIR / "16_candidate_model_summary.csv",
    "step17": METRICS_DIR / "17_tuned_model_summary.csv",
    "step19c": METRICS_DIR / "19c_step20_shortlist.csv",
    "step20": METRICS_DIR / "20_reserved_2024_validation_metrics.csv",
    "step21": METRICS_DIR / "21_final_strategy_selection_evidence.csv",
    "freeze": REPORTS_DIR / "21_final_strategy_freeze.json",
    "step25_annual_metrics": FORECASTING_DIR / "25_annual_level_baseline_metrics_by_horizon.csv",
    "step28_annual_forecast": FORECASTING_DIR / "28_future_annual_thermal_level_2026_2030.csv",
}


def section(title: str) -> None:
    print("\n" + "=" * 122)
    print(title)
    print("=" * 122)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required saved evidence file is missing:\n{path}\n\n"
            "This panel demo is intentionally READ ONLY, so it will not regenerate the missing file."
        )


def existing_columns(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    return [c for c in wanted if c in df.columns]


def show(df: pd.DataFrame, wanted: list[str], sort_by: str | None = None) -> None:
    cols = existing_columns(df, wanted)
    view = df[cols].copy() if cols else df.copy()
    if sort_by and sort_by in view.columns:
        view = view.sort_values(sort_by)
    print(view.round(6).to_string(index=False))


def find_step27_annual_summary() -> tuple[Path | None, pd.DataFrame | None]:
    """Find the saved Step-27 annual reserved-comparison summary without assuming its filename."""
    required = {
        "model_group",
        "method",
        "horizon_count",
        "mean_horizon_MAE_C",
        "mean_horizon_RMSE_C",
    }

    for path in sorted(FORECASTING_DIR.glob("27_*.csv")):
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue

        if required.issubset(set(header.columns)):
            return path, pd.read_csv(path)

    return None, None


def find_step27_annual_fold_metrics() -> tuple[Path | None, pd.DataFrame | None]:
    """Fallback: find the 80-row reserved annual fold file and rebuild the summary in memory."""
    required = {
        "model_group",
        "method",
        "horizon_years",
        "actual_annual_mean_lst_c",
        "predicted_annual_mean_lst_c",
    }

    for path in sorted(FORECASTING_DIR.glob("27_*.csv")):
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue

        if required.issubset(set(header.columns)):
            return path, pd.read_csv(path)

    return None, None


def rebuild_reserved_annual_summary(folds: pd.DataFrame) -> pd.DataFrame:
    """Recreate Step-27 equal-horizon summary in memory only; nothing is written."""
    import numpy as np

    rows = []
    grouped = folds.groupby(["model_group", "method", "horizon_years"], dropna=False)

    for (group, method, horizon), g in grouped:
        y = g["actual_annual_mean_lst_c"].astype(float).to_numpy()
        p = g["predicted_annual_mean_lst_c"].astype(float).to_numpy()
        err = p - y
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        nz = np.abs(y) > 1e-12
        mape = float(np.mean(np.abs(err[nz] / y[nz])) * 100.0) if nz.any() else np.nan
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        bias = float(np.mean(err))

        rows.append({
            "model_group": group,
            "method": method,
            "horizon_years": horizon,
            "MAE_C": mae,
            "RMSE_C": rmse,
            "MAPE_pct": mape,
            "R2": r2,
            "Bias_C": bias,
        })

    horizon = pd.DataFrame(rows)

    return (
        horizon.groupby(["model_group", "method"], as_index=False)
        .agg(
            horizon_count=("horizon_years", "nunique"),
            mean_horizon_MAE_C=("MAE_C", "mean"),
            mean_horizon_RMSE_C=("RMSE_C", "mean"),
            mean_horizon_MAPE_pct=("MAPE_pct", "mean"),
            mean_horizon_R2=("R2", "mean"),
            mean_abs_horizon_Bias_C=("Bias_C", lambda s: float(np.mean(np.abs(s)))),
        )
    )


section("URBAN HEAT - FULL MODEL SELECTION READ-ONLY PANEL EVIDENCE")
print(f"Project root : {PROJECT_ROOT}")
print("Mode         : READ ONLY")
print("Training     : NO")
print("Retraining   : NO")
print("Tuning       : NO")
print("Output write : NO")
print("Original src : NOT EXECUTED")
print(
    "\nPurpose: safely show the complete model-selection story: "
    "baselines -> ML candidates -> tuning -> direct/raw vs anomaly -> reserved 2024 -> final freeze -> annual-level method selection."
)

# -----------------------------------------------------------------------------
# 1. Baseline comparison
# -----------------------------------------------------------------------------
require(FILES["baseline"])
baseline = pd.read_csv(FILES["baseline"])

section("1. SIMPLE BASELINE MODEL COMPARISON")
show(
    baseline,
    [
        "model",
        "folds",
        "mean_mae",
        "median_mae",
        "std_mae",
        "worst_year_mae",
        "mean_rmse",
        "mean_r2",
        "mean_mape_percent",
        "negative_r2_folds",
        "development_mean_mae_rank",
    ],
    sort_by="development_mean_mae_rank",
)
print(
    "\nExplanation: these are simple reference forecasting methods. "
    "They provide the benchmark that the machine-learning candidates should improve on."
)

# Optional saved direct comparison table
if FILES["candidate_vs_baseline"].exists():
    candidate_vs_baseline = pd.read_csv(FILES["candidate_vs_baseline"])
    section("1A. ML CANDIDATES VS FIXED BASELINES")
    show(
        candidate_vs_baseline,
        [
            "candidate_model",
            "baseline_model",
            "candidate_mean_mae",
            "baseline_mean_mae",
            "candidate_minus_baseline_mean_mae",
            "candidate_lower_mean_mae",
        ],
    )
    print(
        "\nReading rule: candidate_lower_mean_mae = True means the ML candidate "
        "had a lower development MAE than that baseline."
    )

# -----------------------------------------------------------------------------
# 2. Candidate model comparison
# -----------------------------------------------------------------------------
require(FILES["step16"])
step16 = pd.read_csv(FILES["step16"])

section("2. STEP 16 - UNTUNED ML CANDIDATE MODEL COMPARISON")
show(
    step16,
    [
        "model",
        "folds",
        "mean_mae",
        "std_mae",
        "worst_year_mae",
        "mean_rmse",
        "mean_r2",
        "mean_mape_percent",
        "negative_r2_folds",
        "development_mean_mae_rank",
        "mae_year_wins",
    ],
    sort_by="development_mean_mae_rank",
)
print(
    "\nExplanation: Random Forest, XGBoost, LightGBM and NGBoost were compared "
    "using the same rolling-origin development years. This stage did not select the final model."
)

# -----------------------------------------------------------------------------
# 3. Tuned candidate comparison
# -----------------------------------------------------------------------------
require(FILES["step17"])
step17 = pd.read_csv(FILES["step17"])

section("3. STEP 17 - TUNED ML MODEL COMPARISON")
show(
    step17,
    [
        "model",
        "mean_mae",
        "std_mae",
        "mean_rmse",
        "mean_r2",
        "negative_r2_count",
        "negative_r2_folds",
        "worst_year_mae",
        "minimum_r2",
        "development_mean_mae_rank",
        "descriptive_mean_mae_rank",
    ],
    sort_by="mean_mae",
)
print(
    "\nExplanation: tuning used development data only. Reserved 2024 and final 2025 "
    "were not used for tuning."
)

# -----------------------------------------------------------------------------
# 4. Direct/raw LST vs anomaly strategy - development shortlist
# -----------------------------------------------------------------------------
require(FILES["step19c"])
step19c = pd.read_csv(FILES["step19c"])

section("4. DIRECT/RAW LST VS ANOMALY-BASED STRATEGY - DEVELOPMENT EVIDENCE")
show(
    step19c,
    [
        "shortlist_order",
        "source_step",
        "strategy",
        "display_name",
        "base_model",
        "mean_mae",
        "mean_rmse",
        "mean_r2",
        "negative_r2_count",
        "worst_year_mae",
        "shortlist_role",
    ],
    sort_by="shortlist_order",
)
print("\nDIRECT/RAW : predicts absolute LST_C directly.")
print(
    "ANOMALY    : predicts within-year spatial LST deviation, then adds a separately "
    "forecast annual thermal level."
)
print("At this stage the final strategy was still NOT selected.")

# -----------------------------------------------------------------------------
# 5. Reserved 2024 raw-vs-anomaly validation
# -----------------------------------------------------------------------------
require(FILES["step20"])
step20 = pd.read_csv(FILES["step20"])

section("5. STEP 20 - RESERVED 2024 RAW VS ANOMALY COMPARISON")
show(
    step20,
    [
        "strategy",
        "display_name",
        "base_model",
        "prediction_method",
        "mae",
        "rmse",
        "r2",
        "mape_percent",
        "mean_prediction_bias_pred_minus_actual_c",
        "centered_spatial_r2_evaluation_only",
    ],
    sort_by="mae",
)

if "strategy" in step20.columns:
    raw = step20[step20["strategy"].astype(str).eq("raw_step17")]
    anom = step20[
        step20["strategy"].astype(str).eq("lgbm_anomaly_plus_prior_mean_level")
    ]

    if not raw.empty and not anom.empty:
        raw = raw.iloc[0]
        anom = anom.iloc[0]

        section("5A. SIMPLE DIRECT VS SELECTED ANOMALY DIFFERENCE")
        print(f"Direct/raw RF 2024 MAE     : {float(raw['mae']):.6f} C")
        print(f"LightGBM anomaly 2024 MAE  : {float(anom['mae']):.6f} C")
        print(f"MAE improvement            : {float(raw['mae']) - float(anom['mae']):.6f} C")
        print()
        print(f"Direct/raw RF 2024 RMSE    : {float(raw['rmse']):.6f} C")
        print(f"LightGBM anomaly 2024 RMSE : {float(anom['rmse']):.6f} C")
        print(f"RMSE improvement           : {float(raw['rmse']) - float(anom['rmse']):.6f} C")
        print()
        print(f"Direct/raw RF 2024 R2      : {float(raw['r2']):.6f}")
        print(f"LightGBM anomaly 2024 R2   : {float(anom['r2']):.6f}")
        print(f"R2 improvement             : {float(anom['r2']) - float(raw['r2']):.6f}")

print(
    "\nInterpretation: the anomaly-based LightGBM beat the raw tuned Random Forest "
    "on reserved-2024 MAE, RMSE and R2."
)

# -----------------------------------------------------------------------------
# 6. Final strategy selection evidence
# -----------------------------------------------------------------------------
require(FILES["step21"])
step21 = pd.read_csv(FILES["step21"])

section("6. STEP 21 - FINAL STRATEGY SELECTION EVIDENCE")
show(
    step21,
    [
        "strategy",
        "display_name",
        "base_model",
        "development_mean_mae_2019_2023",
        "development_mean_rmse_2019_2023",
        "development_mean_r2_2019_2023",
        "development_negative_r2_count",
        "development_worst_year_mae",
        "reserved_2024_mae",
        "reserved_2024_rmse",
        "reserved_2024_r2",
        "reserved_2024_mape_percent",
        "reserved_2024_bias_pred_minus_actual_c",
        "reserved_2024_centered_spatial_r2",
    ],
    sort_by="reserved_2024_mae",
)

section("7. FINAL FROZEN SPATIAL-MODEL DECISION")
print("Final strategy : lgbm_anomaly_plus_prior_mean_level")
print("Base model     : LightGBM")
print("Final decision : frozen before opening the 2025 lockbox")

if FILES["freeze"].exists():
    try:
        with FILES["freeze"].open("r", encoding="utf-8") as f:
            freeze = json.load(f)
        print("\nSaved Step-21 freeze metadata:")
        print(json.dumps(freeze, indent=2))
    except Exception as exc:
        print(f"\nFreeze JSON exists but could not be displayed: {exc}")

# -----------------------------------------------------------------------------
# 8. Annual thermal-level baseline comparison (Step 25)
# -----------------------------------------------------------------------------
require(FILES["step25_annual_metrics"])
annual25 = pd.read_csv(FILES["step25_annual_metrics"])

section("8. STEP 25 - ANNUAL THERMAL-LEVEL BASELINE COMPARISON (H1-H5)")
show(
    annual25,
    [
        "method",
        "horizon_years",
        "fold_count",
        "MAE_C",
        "RMSE_C",
        "MAPE_pct",
        "R2",
        "Bias_C",
    ],
    sort_by="horizon_years",
)

annual25_summary = (
    annual25.groupby("method", as_index=False)
    .agg(
        horizon_count=("horizon_years", "nunique"),
        mean_horizon_MAE_C=("MAE_C", "mean"),
        mean_horizon_RMSE_C=("RMSE_C", "mean"),
        mean_horizon_MAPE_pct=("MAPE_pct", "mean"),
        mean_horizon_R2=("R2", "mean"),
        mean_abs_horizon_Bias_C=("Bias_C", lambda s: s.abs().mean()),
    )
    .sort_values(["mean_horizon_MAE_C", "mean_horizon_RMSE_C"])
)

section("8A. STEP 25 - EQUAL-HORIZON BASELINE SUMMARY")
print(annual25_summary.round(6).to_string(index=False))
print(
    "\nInterpretation: the initial annual-level baseline experiment compared "
    "Persistence, Expanding Mean, Rolling-3 Mean, Drift and OLS Linear Trend. "
    "This was development/hindcast evidence, not the final reserved selection."
)

# -----------------------------------------------------------------------------
# 9. Reserved 2024-2025 annual-level comparison (Step 27)
# -----------------------------------------------------------------------------
step27_path, annual27 = find_step27_annual_summary()

if annual27 is None:
    fold_path, fold_df = find_step27_annual_fold_metrics()
    if fold_df is None:
        raise FileNotFoundError(
            "Could not locate the saved Step-27 annual reserved-validation evidence under:\n"
            f"{FORECASTING_DIR}\n\n"
            "This panel demo is READ ONLY and will not regenerate Step 27."
        )
    annual27 = rebuild_reserved_annual_summary(fold_df)
    step27_path = fold_path
    step27_source_note = "Rebuilt in memory from saved Step-27 reserved fold rows (no file written)."
else:
    step27_source_note = "Loaded the saved Step-27 annual summary directly."

section("9. STEP 27 - RESERVED 2024-2025 ANNUAL-LEVEL FINAL COMPARISON")
annual27_view = annual27.copy()
if "mean_horizon_MAE_C" in annual27_view.columns:
    annual27_view = annual27_view.sort_values(
        ["mean_horizon_MAE_C", "mean_horizon_RMSE_C"]
    )
show(
    annual27_view,
    [
        "model_group",
        "method",
        "horizon_count",
        "mean_horizon_MAE_C",
        "mean_horizon_RMSE_C",
        "mean_horizon_MAPE_pct",
        "mean_horizon_R2",
        "mean_abs_horizon_Bias_C",
        "selection_status",
        "selection_rule",
    ],
)
print(f"\nEvidence source: {step27_path}")
print(step27_source_note)
print(
    "\nEight fixed annual-level methods were compared across H1-H5 using reserved "
    "2024-2025 evidence: Persistence, Expanding Mean, Rolling-3 Mean, Drift, "
    "OLS Linear Trend, Simple Exponential Smoothing, Damped Holt Trend, and "
    "Theil-Sen Robust Trend."
)

# Final winner from the saved/rebuilt reserved comparison
if "mean_horizon_MAE_C" in annual27.columns:
    winner = annual27.sort_values(
        ["mean_horizon_MAE_C", "mean_horizon_RMSE_C", "mean_abs_horizon_Bias_C"]
    ).iloc[0]

    section("9A. FINAL ANNUAL THERMAL-LEVEL METHOD")
    print(f"Selected method             : {winner['method']}")
    if "model_group" in winner.index:
        print(f"Source group                : {winner['model_group']}")
    print(f"Reserved mean-horizon MAE   : {float(winner['mean_horizon_MAE_C']):.6f} C")
    print(f"Reserved mean-horizon RMSE  : {float(winner['mean_horizon_RMSE_C']):.6f} C")
    if "mean_horizon_MAPE_pct" in winner.index:
        print(f"Reserved mean-horizon MAPE  : {float(winner['mean_horizon_MAPE_pct']):.6f} %")
    print(
        "Selection rule             : lowest reserved mean-horizon MAE; "
        "tie-break RMSE then absolute bias"
    )

# -----------------------------------------------------------------------------
# 10. Show that Step 28 actually uses the frozen annual-level winner
# -----------------------------------------------------------------------------
if FILES["step28_annual_forecast"].exists():
    future_level = pd.read_csv(FILES["step28_annual_forecast"])
    section("10. STEP 28 - OPERATIONAL USE OF THE SELECTED ANNUAL-LEVEL METHOD")
    show(
        future_level,
        [
            "year",
            "forecast_origin_year",
            "horizon_years",
            "annual_level_method",
            "forecast_annual_mean_lst_c",
        ],
        sort_by="year",
    )
    print(
        "\nFinal future-grid formulation: forecast annual thermal level + "
        "centered LightGBM spatial anomaly."
    )

section("11. COMPLETE PANEL STORY")
print(
    "1) I established simple forecasting baselines.\n"
    "2) I compared Random Forest, XGBoost, LightGBM and NGBoost on the same rolling-origin development folds.\n"
    "3) I tuned the ML candidates using development data only.\n"
    "4) I compared direct/raw LST prediction with a level + spatial-anomaly design.\n"
    "5) I independently checked the shortlisted spatial strategies on reserved 2024.\n"
    "6) LightGBM anomaly + prior-mean annual level was selected and frozen before the 2025 lockbox.\n"
    "7) I separately evaluated five annual thermal-level baseline methods across H1-H5.\n"
    "8) I then compared eight fixed annual-level methods on reserved 2024-2025 evidence.\n"
    "9) Expanding Mean was selected by lowest reserved mean-horizon MAE, with RMSE and absolute bias as tie-breakers.\n"
    "10) Step 28 uses the frozen annual-level method together with the frozen centered LightGBM spatial anomaly."
)

section("READ-ONLY PANEL DEMO COMPLETED")
print("No model was trained or retrained.")
print("No hyperparameter tuning was performed.")
print("No research output file was regenerated or overwritten.")
print("Original Urban_Heat_Research/src remained untouched.")

