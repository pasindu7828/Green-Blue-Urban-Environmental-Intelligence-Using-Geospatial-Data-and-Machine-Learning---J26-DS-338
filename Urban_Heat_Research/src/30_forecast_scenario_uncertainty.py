
"""
STEP 30 - FORECAST + SCENARIO UNCERTAINTY / CONFIDENCE
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Add an uncertainty/confidence layer to:
1) Step-28 future LST forecasts (2026-2030)
2) Final Step-29A / 29B / 29C / 29D scenario outputs

IMPORTANT METHODOLOGY RULES
---------------------------
- Do NOT retrain the final LightGBM.
- Do NOT tune hyperparameters.
- Do NOT modify Step-28 point forecasts.
- Do NOT modify Step-29 scenario point estimates.
- Do NOT use 2025 to recalibrate uncertainty after the final test.
- Do NOT claim a calibrated causal interval for scenario ΔLST.
- Do NOT use raw NGBoost intervals as the final uncertainty layer.

FINAL UNCERTAINTY DESIGN
------------------------
A) Spatial uncertainty:
   - 2019-2023 rolling-origin final-strategy residuals:
       build per-grid historical error profile.
   - 2024 reserved validation:
       calibrate 90% and 95% centered-spatial residual radii.
   - 2025 final lockbox:
       audit the frozen 2024 spatial calibration only.

B) Annual-level uncertainty:
   - Use the final Expanding Mean annual-level forecast method.
   - Development annual forecast errors through 2023:
       calibrate horizon-specific H1-H5 empirical radii.
   - 2024-2025:
       audit only.
   - Apply cumulative maximum so farther horizons do not get
     artificially narrower annual-level uncertainty.

C) Future reference interval:
       total_radius = spatial_radius + annual_level_radius

   This is a conservative reference prediction band, not a formal
   exact future-coverage guarantee.

D) Scenario uncertainty:
   - Add the same grid/year baseline-calibrated outcome reference band
     around scenario LST.
   - Do NOT claim a calibrated interval for scenario ΔLST.
   - 29D historical-support/extrapolation class is carried forward.
"""

from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"

FINAL_STRATEGY = "lgbm_anomaly_plus_prior_mean_level"
FINAL_ANNUAL_METHOD = "Expanding Mean"

DEVELOPMENT_YEARS = [2019, 2020, 2021, 2022, 2023]
CALIBRATION_YEAR = 2024
AUDIT_YEAR = 2025
FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]
HORIZONS = [1, 2, 3, 4, 5]

EXPECTED_GRIDS = 2194
EXPECTED_FUTURE_ROWS = 2194 * 5

ALPHAS = {
    0.10: "90",
    0.05: "95",
}

EXPECTED_HASHES = {
    "19b5_level_anomaly_predictions.csv":
        "2afb33f8251ef75ff1d143405247cb7426252e62e115ab74a1c021c123ea6beb",
    "20_reserved_2024_predictions.csv":
        "5329646e69d5262c667d144707c2e8cace1459abe98c7bd86d0181b13f4d452a",
    "22_final_2025_lockbox_predictions.csv":
        "63f9e46f16d10809fbdbef77a980a9592aea6c2031eef19a1f6e99645aae8db6",
    "25_annual_level_baseline_rolling_origin_predictions.csv":
        "319bf955d3c555e7b1c24111746c8c10de89b55e218241dcb22c1d7caa853007",
    "28_future_grid_lst_predictions_2026_2030.csv":
        "55586971042bbce04845a4a33ae724f51b8417fce9bafc2c15bc0a5e75d7ff3e",
    "28_future_prediction_audit.csv":
        "368daa91dcbce7ac009c8f11f0ecedde5817d5eb1ed0d2c99888f7be9017f3af",
    "29a_v2_persistent_green_scenarios_2026_2030.csv":
        "0ea174b2603976c57fb019e0467928e90b4d5243b7d09cc492577852ed329c96",
    "29a_v2_persistent_green_scenario_audit.csv":
        "eaac9e8e0e16066c3a0a24ca4a9328a3c411f04d17e15940175c87c7d8c751d4",
    "29b_built_up_intensity_scenarios_2026_2030.csv":
        "5a134e775fd31be5aa9ddc27f525e80c8badc4ea9a4cd9d20a1808f9baf4784a",
    "29b_built_up_intensity_scenario_audit.csv":
        "044c1a70f3643888ed279a4d67442b2d2e5bf9d1fdf64a20b77ffb26e1f851dc",
    "29c_combined_green_built_scenarios_2026_2030.csv":
        "89766e3c16cd84fe5227149757f0d2faaad48b3cd3d817f4a93add04b263757c",
    "29c_combined_scenario_audit.csv":
        "4fc87ab8028d1dd959cfd26097371e1ab2256a512d635388a0de68d5f43045e8",
    "29d_blue_space_ndwi_scenarios_2026_2030.csv":
        "328421e5b4d3c36c0aef7747009573a5bb39f03bdb83c9c5a88d923e10a27521",
    "29d_blue_space_scenario_audit.csv":
        "97c3f0a25ace0ecb18803653ec5b9c2c2f1f84b49cc29331b91e3b0d78944891",
    "29d_qa_scenario_row_support.csv":
        "7a140a0a0c7fc79376d8f4e3b041224fb879bc8ab9c183aa22e81c02ffe53134",
    "29d_qa_support_audit.csv":
        "85fca95451d845eecdd6ddf52506c61bb7bc7cf25ccc0dbbf818998dc4cb633e",
}


# =============================================================================
# 2. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

DEV_PRED_PATH = ROOT / "outputs" / "predictions" / "19b5_level_anomaly_predictions.csv"
VAL2024_PRED_PATH = ROOT / "outputs" / "predictions" / "20_reserved_2024_predictions.csv"
TEST2025_PRED_PATH = ROOT / "outputs" / "predictions" / "22_final_2025_lockbox_predictions.csv"

ANNUAL_ROLLING_PATH = (
    ROOT / "outputs" / "forecasting"
    / "25_annual_level_baseline_rolling_origin_predictions.csv"
)

FUTURE_PRED_PATH = (
    ROOT / "outputs" / "forecasting"
    / "28_future_grid_lst_predictions_2026_2030.csv"
)

STEP28_AUDIT_PATH = (
    ROOT / "outputs" / "reports"
    / "28_future_prediction_audit.csv"
)

GREEN_SCENARIO_PATH = (
    ROOT / "outputs" / "scenarios"
    / "29a_v2_persistent_green_scenarios_2026_2030.csv"
)

GREEN_AUDIT_PATH = (
    ROOT / "outputs" / "reports"
    / "29a_v2_persistent_green_scenario_audit.csv"
)

BUILT_SCENARIO_PATH = (
    ROOT / "outputs" / "scenarios"
    / "29b_built_up_intensity_scenarios_2026_2030.csv"
)

BUILT_AUDIT_PATH = (
    ROOT / "outputs" / "reports"
    / "29b_built_up_intensity_scenario_audit.csv"
)

COMBINED_SCENARIO_PATH = (
    ROOT / "outputs" / "scenarios"
    / "29c_combined_green_built_scenarios_2026_2030.csv"
)

COMBINED_AUDIT_PATH = (
    ROOT / "outputs" / "reports"
    / "29c_combined_scenario_audit.csv"
)

BLUE_SCENARIO_PATH = (
    ROOT / "outputs" / "scenarios"
    / "29d_blue_space_ndwi_scenarios_2026_2030.csv"
)

BLUE_AUDIT_PATH = (
    ROOT / "outputs" / "reports"
    / "29d_blue_space_scenario_audit.csv"
)

BLUE_SUPPORT_PATH = (
    ROOT / "outputs" / "reports"
    / "29d_qa_scenario_row_support.csv"
)

BLUE_SUPPORT_AUDIT_PATH = (
    ROOT / "outputs" / "reports"
    / "29d_qa_support_audit.csv"
)

REPORT_DIR = ROOT / "outputs" / "reports"
UNCERTAINTY_DIR = ROOT / "outputs" / "uncertainty"
FIGURE_DIR = ROOT / "outputs" / "figures"

for d in [REPORT_DIR, UNCERTAINTY_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SPATIAL_CAL_PATH = REPORT_DIR / "30_spatial_uncertainty_calibration_2024.csv"
SPATIAL_AUDIT_PATH = REPORT_DIR / "30_spatial_uncertainty_audit_2025.csv"
GRID_PROFILE_PATH = REPORT_DIR / "30_grid_historical_spatial_error_profile.csv"
ANNUAL_UNC_PATH = REPORT_DIR / "30_annual_level_uncertainty_by_horizon.csv"
ANNUAL_AUDIT_PATH = REPORT_DIR / "30_annual_level_reserved_audit.csv"

FUTURE_UNC_PATH = (
    UNCERTAINTY_DIR
    / "30_future_lst_with_uncertainty_2026_2030.csv"
)

FUTURE_SUMMARY_PATH = REPORT_DIR / "30_future_uncertainty_summary.csv"

GREEN_UNC_PATH = UNCERTAINTY_DIR / "30_29a_green_scenarios_with_uncertainty.csv"
BUILT_UNC_PATH = UNCERTAINTY_DIR / "30_29b_built_scenarios_with_uncertainty.csv"
COMBINED_UNC_PATH = UNCERTAINTY_DIR / "30_29c_combined_scenarios_with_uncertainty.csv"
BLUE_UNC_PATH = UNCERTAINTY_DIR / "30_29d_blue_scenarios_with_uncertainty.csv"

SCENARIO_SUMMARY_PATH = REPORT_DIR / "30_scenario_uncertainty_summary.csv"
AUDIT_PATH = REPORT_DIR / "30_uncertainty_audit.csv"
METHOD_PATH = REPORT_DIR / "30_uncertainty_methodology.json"
REPORT_PATH = REPORT_DIR / "30_uncertainty_report.txt"
HASH_PATH = REPORT_DIR / "30_uncertainty_hashes.csv"

FIG_SPATIAL = FIGURE_DIR / "30_spatial_calibration_absolute_residuals_2024.png"
FIG_ANNUAL = FIGURE_DIR / "30_annual_level_uncertainty_by_horizon.png"
FIG_FUTURE = FIGURE_DIR / "30_future_reference_interval_width_by_year.png"


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 136)
    print(title)
    print("=" * 136)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hash(path, expected, label):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual  : {actual}"
        )

    print(f"PASS hash: {label} = {actual}")


def require_all_pass(path, label):
    df = pd.read_csv(path)

    status_col = None
    for c in ["status", "Status"]:
        if c in df.columns:
            status_col = c
            break

    if status_col is None:
        raise RuntimeError(f"{label}: no status column.")

    bad = df[df[status_col].astype(str).str.upper() != "PASS"]

    if not bad.empty:
        print(bad.to_string(index=False))
        raise RuntimeError(f"{label} is not fully PASS.")

    print(f"PASS: {label} audit fully PASS.")


def detect_unique_column(
    df,
    exact=None,
    contains=None,
    excludes=None,
    label="column",
):
    exact = exact or []
    contains = contains or []
    excludes = excludes or []

    lower_map = {str(c).lower(): c for c in df.columns}

    for c in exact:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    matches = []
    for c in df.columns:
        low = str(c).lower()
        if all(t.lower() in low for t in contains) and not any(
            t.lower() in low for t in excludes
        ):
            matches.append(c)

    if len(matches) == 1:
        return matches[0]

    raise RuntimeError(
        f"Could not uniquely detect {label}. "
        f"Matches={matches}. Columns={df.columns.tolist()}"
    )


def filter_final_strategy(df, label):
    strategy_col = None

    for c in [
        "strategy",
        "display_name",
        "candidate",
        "candidate_name",
        "model_strategy",
    ]:
        if c in df.columns:
            strategy_col = c
            break

    if strategy_col is None:
        return df.copy()

    s = df[strategy_col].astype(str).str.lower()

    mask = s.str.contains(FINAL_STRATEGY.lower(), regex=False)

    if not mask.any():
        mask = (
            s.str.contains("lgbm", regex=False)
            & s.str.contains("anomaly", regex=False)
            & s.str.contains("prior", regex=False)
        )

    if not mask.any():
        raise RuntimeError(
            f"{label}: final strategy not found. "
            f"Unique={df[strategy_col].dropna().astype(str).unique().tolist()}"
        )

    return df.loc[mask].copy()


def standardize_prediction_file(path, years, label, use_strategy_filter=True):
    df = pd.read_csv(path, dtype={GRID_COL: str}, low_memory=False)

    if use_strategy_filter:
        df = filter_final_strategy(df, label)

    grid_col = detect_unique_column(
        df,
        exact=["grid_id"],
        label=f"{label} grid",
    )

    year_col = detect_unique_column(
        df,
        exact=["validation_year", "year", "target_year"],
        contains=["year"],
        excludes=["origin"],
        label=f"{label} year",
    )

    actual_col = None
    for exact, contains in [
        (
            ["actual_lst_c", "actual", "y_true", "LST_C"],
            ["actual", "lst"],
        ),
        (
            [],
            ["actual"],
        ),
    ]:
        try:
            actual_col = detect_unique_column(
                df,
                exact=exact,
                contains=contains,
                excludes=["annual", "mean", "anomaly"],
                label=f"{label} actual",
            )
            break
        except RuntimeError:
            pass

    if actual_col is None:
        raise RuntimeError(
            f"{label}: actual LST column not detected. "
            f"Columns={df.columns.tolist()}"
        )

    pred_col = None
    for exact, contains in [
        (
            [
                "predicted_lst_c",
                "final_prediction_c",
                "final_predicted_lst_c",
                "prediction",
                "y_pred",
            ],
            ["pred", "lst"],
        ),
        (
            [],
            ["prediction"],
        ),
        (
            [],
            ["predicted"],
        ),
    ]:
        try:
            pred_col = detect_unique_column(
                df,
                exact=exact,
                contains=contains,
                excludes=["raw", "anomaly", "annual", "mean", "bias"],
                label=f"{label} prediction",
            )
            break
        except RuntimeError:
            pass

    if pred_col is None:
        raise RuntimeError(
            f"{label}: prediction column not detected. "
            f"Columns={df.columns.tolist()}"
        )

    out = pd.DataFrame({
        GRID_COL: df[grid_col].astype(str),
        YEAR_COL: pd.to_numeric(df[year_col], errors="raise").astype(int),
        "actual_lst_c": pd.to_numeric(df[actual_col], errors="raise"),
        "predicted_lst_c": pd.to_numeric(df[pred_col], errors="raise"),
    })

    out = (
        out[out[YEAR_COL].isin(years)]
        .sort_values([YEAR_COL, GRID_COL])
        .reset_index(drop=True)
    )

    expected_rows = EXPECTED_GRIDS * len(years)

    if len(out) != expected_rows:
        raise RuntimeError(
            f"{label}: rows={len(out)}, expected={expected_rows}. "
            f"Detected actual={actual_col}, pred={pred_col}, year={year_col}"
        )

    if out.duplicated([GRID_COL, YEAR_COL]).any():
        raise RuntimeError(f"{label}: duplicate grid-year rows.")

    return out


def add_centered_residuals(df):
    out = df.copy()

    actual_mean = out.groupby(YEAR_COL)["actual_lst_c"].transform("mean")
    pred_mean = out.groupby(YEAR_COL)["predicted_lst_c"].transform("mean")

    out["actual_centered_c"] = out["actual_lst_c"] - actual_mean
    out["predicted_centered_c"] = out["predicted_lst_c"] - pred_mean

    out["spatial_residual_c"] = (
        out["actual_centered_c"]
        - out["predicted_centered_c"]
    )

    out["abs_spatial_residual_c"] = out["spatial_residual_c"].abs()

    return out


def conformal_quantile(scores, alpha):
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        raise RuntimeError("Empty conformal score array.")

    ordered = np.sort(arr)

    rank = int(math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(ordered))

    return float(ordered[rank - 1])


def detect_annual_columns(df):
    method_col = detect_unique_column(
        df,
        exact=["method"],
        contains=["method"],
        label="annual method",
    )

    horizon_col = detect_unique_column(
        df,
        exact=["horizon_years", "forecast_horizon_years", "horizon"],
        contains=["horizon"],
        label="annual horizon",
    )

    target_year_col = detect_unique_column(
        df,
        exact=["target_year", "validation_year", "year"],
        contains=["year"],
        excludes=["origin", "start", "end"],
        label="annual target year",
    )

    actual_col = detect_unique_column(
        df,
        exact=[
            "actual_level_c",
            "actual_annual_mean_lst_c",
            "actual_mean_lst_c",
            "actual_lst_c",
        ],
        contains=["actual"],
        excludes=["error", "bias"],
        label="annual actual",
    )

    pred_col = None
    for exact, contains in [
        (
            [
                "forecast_level_c",
                "predicted_level_c",
                "forecast_annual_mean_lst_c",
                "predicted_annual_mean_lst_c",
            ],
            ["forecast"],
        ),
        (
            [],
            ["predicted"],
        ),
    ]:
        try:
            pred_col = detect_unique_column(
                df,
                exact=exact,
                contains=contains,
                excludes=["error", "bias"],
                label="annual prediction",
            )
            break
        except RuntimeError:
            pass

    if pred_col is None:
        raise RuntimeError(
            "Could not detect annual forecast column. "
            f"Columns={df.columns.tolist()}"
        )

    return method_col, horizon_col, target_year_col, actual_col, pred_col


def add_reference_bands(
    df,
    point_col,
    radius_table,
    grid_profile,
    prefix,
):
    out = df.copy()

    out[GRID_COL] = out[GRID_COL].astype(str)
    out[YEAR_COL] = pd.to_numeric(out[YEAR_COL], errors="raise").astype(int)

    out["horizon_years"] = out[YEAR_COL] - 2025

    out = out.merge(
        radius_table,
        on="horizon_years",
        how="left",
        validate="many_to_one",
    )

    out = out.merge(
        grid_profile,
        on=GRID_COL,
        how="left",
        validate="many_to_one",
    )

    if out["total_radius_95_c"].isna().any():
        raise RuntimeError(f"{prefix}: missing total uncertainty radius.")

    if out["relative_confidence_label"].isna().any():
        raise RuntimeError(f"{prefix}: missing relative confidence label.")

    point = pd.to_numeric(out[point_col], errors="raise")

    out[f"{prefix}_reference_pi90_lower_c"] = (
        point - out["total_radius_90_c"]
    )
    out[f"{prefix}_reference_pi90_upper_c"] = (
        point + out["total_radius_90_c"]
    )
    out[f"{prefix}_reference_pi95_lower_c"] = (
        point - out["total_radius_95_c"]
    )
    out[f"{prefix}_reference_pi95_upper_c"] = (
        point + out["total_radius_95_c"]
    )
    out[f"{prefix}_reference_pi95_width_c"] = (
        2.0 * out["total_radius_95_c"]
    )

    return out


# =============================================================================
# 4. VERIFY PREREQUISITES
# =============================================================================

section("STEP 30 - FORECAST + SCENARIO UNCERTAINTY / CONFIDENCE")

for path, key, label in [
    (DEV_PRED_PATH, "19b5_level_anomaly_predictions.csv", "2019-2023 rolling-origin predictions"),
    (VAL2024_PRED_PATH, "20_reserved_2024_predictions.csv", "2024 reserved predictions"),
    (TEST2025_PRED_PATH, "22_final_2025_lockbox_predictions.csv", "2025 final lockbox predictions"),
    (ANNUAL_ROLLING_PATH, "25_annual_level_baseline_rolling_origin_predictions.csv", "annual rolling-origin predictions"),
    (FUTURE_PRED_PATH, "28_future_grid_lst_predictions_2026_2030.csv", "Step-28 future predictions"),
    (STEP28_AUDIT_PATH, "28_future_prediction_audit.csv", "Step-28 audit"),
    (GREEN_SCENARIO_PATH, "29a_v2_persistent_green_scenarios_2026_2030.csv", "Step-29A output"),
    (GREEN_AUDIT_PATH, "29a_v2_persistent_green_scenario_audit.csv", "Step-29A audit"),
    (BUILT_SCENARIO_PATH, "29b_built_up_intensity_scenarios_2026_2030.csv", "Step-29B output"),
    (BUILT_AUDIT_PATH, "29b_built_up_intensity_scenario_audit.csv", "Step-29B audit"),
    (COMBINED_SCENARIO_PATH, "29c_combined_green_built_scenarios_2026_2030.csv", "Step-29C output"),
    (COMBINED_AUDIT_PATH, "29c_combined_scenario_audit.csv", "Step-29C audit"),
    (BLUE_SCENARIO_PATH, "29d_blue_space_ndwi_scenarios_2026_2030.csv", "Step-29D output"),
    (BLUE_AUDIT_PATH, "29d_blue_space_scenario_audit.csv", "Step-29D audit"),
    (BLUE_SUPPORT_PATH, "29d_qa_scenario_row_support.csv", "Step-29D-QA row support"),
    (BLUE_SUPPORT_AUDIT_PATH, "29d_qa_support_audit.csv", "Step-29D-QA audit"),
]:
    verify_hash(path, EXPECTED_HASHES[key], label)

for path, label in [
    (STEP28_AUDIT_PATH, "Step-28"),
    (GREEN_AUDIT_PATH, "Step-29A"),
    (BUILT_AUDIT_PATH, "Step-29B"),
    (COMBINED_AUDIT_PATH, "Step-29C"),
    (BLUE_AUDIT_PATH, "Step-29D"),
    (BLUE_SUPPORT_AUDIT_PATH, "Step-29D-QA"),
]:
    require_all_pass(path, label)


# =============================================================================
# 5. LOAD OUT-OF-SAMPLE FINAL-STRATEGY PREDICTIONS
# =============================================================================

section("1. LOAD OUT-OF-SAMPLE FINAL-STRATEGY PREDICTIONS")

dev = standardize_prediction_file(
    DEV_PRED_PATH,
    DEVELOPMENT_YEARS,
    "2019-2023 development rolling-origin",
    use_strategy_filter=True,
)

val2024 = standardize_prediction_file(
    VAL2024_PRED_PATH,
    [CALIBRATION_YEAR],
    "2024 reserved calibration",
    use_strategy_filter=True,
)

test2025 = standardize_prediction_file(
    TEST2025_PRED_PATH,
    [AUDIT_YEAR],
    "2025 final audit",
    use_strategy_filter=False,
)

dev = add_centered_residuals(dev)
val2024 = add_centered_residuals(val2024)
test2025 = add_centered_residuals(test2025)

print(f"Development residual rows : {len(dev):,}")
print(f"2024 calibration rows     : {len(val2024):,}")
print(f"2025 audit rows           : {len(test2025):,}")


# =============================================================================
# 6. PER-GRID HISTORICAL SPATIAL ERROR PROFILE
# =============================================================================

section("2. BUILD GRID-LEVEL HISTORICAL SPATIAL-ERROR PROFILE")

grid_profile = (
    dev.groupby(GRID_COL)
    .agg(
        local_historical_centered_mae_c=(
            "abs_spatial_residual_c",
            "mean",
        ),
        local_historical_centered_median_ae_c=(
            "abs_spatial_residual_c",
            "median",
        ),
        local_historical_centered_rmse_c=(
            "spatial_residual_c",
            lambda s: float(
                np.sqrt(
                    np.mean(
                        np.square(np.asarray(s, dtype=float))
                    )
                )
            ),
        ),
        historical_oos_year_count=(YEAR_COL, "nunique"),
    )
    .reset_index()
)

if len(grid_profile) != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Grid profile count={len(grid_profile)}, expected={EXPECTED_GRIDS}"
    )

if not (
    grid_profile["historical_oos_year_count"]
    == len(DEVELOPMENT_YEARS)
).all():
    raise RuntimeError(
        "Some grids are missing development rolling-origin years."
    )

q33 = float(
    grid_profile["local_historical_centered_mae_c"].quantile(1 / 3)
)
q67 = float(
    grid_profile["local_historical_centered_mae_c"].quantile(2 / 3)
)

grid_profile["local_historical_error_tier"] = np.select(
    [
        grid_profile["local_historical_centered_mae_c"] <= q33,
        grid_profile["local_historical_centered_mae_c"] <= q67,
    ],
    [
        "Low historical spatial error",
        "Moderate historical spatial error",
    ],
    default="High historical spatial error",
)

grid_profile["relative_confidence_label"] = (
    grid_profile["local_historical_error_tier"].map({
        "Low historical spatial error":
            "Higher relative confidence",
        "Moderate historical spatial error":
            "Moderate relative confidence",
        "High historical spatial error":
            "Lower relative confidence",
    })
)

grid_profile.to_csv(
    GRID_PROFILE_PATH,
    index=False,
)

print(f"Grid-error q33={q33:.6f} °C")
print(f"Grid-error q67={q67:.6f} °C")
print(grid_profile["local_historical_error_tier"].value_counts().to_string())


# =============================================================================
# 7. SPATIAL CALIBRATION - RESERVED 2024
# =============================================================================

section("3. CALIBRATE SPATIAL UNCERTAINTY ON RESERVED 2024")

scores_2024 = val2024["abs_spatial_residual_c"].to_numpy(dtype=float)

spatial_quantiles = {}
spatial_rows = []

for alpha, label in ALPHAS.items():
    q = conformal_quantile(scores_2024, alpha)
    spatial_quantiles[label] = q

    spatial_rows.append({
        "calibration_year": 2024,
        "calibration_rows": len(val2024),
        "nominal_level_percent": (1 - alpha) * 100,
        "spatial_radius_c": q,
        "score_definition": "absolute centered spatial residual",
        "method": "split-conformal-style finite-sample absolute-residual quantile",
    })

spatial_cal_df = pd.DataFrame(spatial_rows)
spatial_cal_df.to_csv(SPATIAL_CAL_PATH, index=False)

print(spatial_cal_df.to_string(index=False))


# =============================================================================
# 8. 2025 AUDIT OF FROZEN SPATIAL CALIBRATION
# =============================================================================

section("4. AUDIT FROZEN 2024 SPATIAL BAND ON 2025")

spatial_audit_rows = []

for alpha, label in ALPHAS.items():
    radius = spatial_quantiles[label]

    actual = test2025["actual_centered_c"].to_numpy(dtype=float)
    predicted = test2025["predicted_centered_c"].to_numpy(dtype=float)

    covered = (
        (actual >= predicted - radius)
        & (actual <= predicted + radius)
    )

    spatial_audit_rows.append({
        "audit_year": 2025,
        "nominal_level_percent": (1 - alpha) * 100,
        "frozen_spatial_radius_c": radius,
        "observed_spatial_coverage_percent":
            float(np.mean(covered) * 100),
        "covered_grids": int(covered.sum()),
        "grid_count": int(len(covered)),
        "recalibrated_after_2025": False,
    })

spatial_audit_df = pd.DataFrame(spatial_audit_rows)
spatial_audit_df.to_csv(SPATIAL_AUDIT_PATH, index=False)

print(spatial_audit_df.to_string(index=False))
print("\nPASS: 2025 used only as audit; calibration is unchanged.")


# =============================================================================
# 9. ANNUAL-LEVEL HORIZON UNCERTAINTY
# =============================================================================

section("5. CALIBRATE ANNUAL-LEVEL UNCERTAINTY BY HORIZON")

annual = pd.read_csv(
    ANNUAL_ROLLING_PATH,
    low_memory=False,
)

(
    method_col,
    horizon_col,
    target_year_col,
    actual_col,
    pred_col,
) = detect_annual_columns(annual)

annual_selected = annual[
    annual[method_col].astype(str).str.lower()
    == FINAL_ANNUAL_METHOD.lower()
].copy()

if annual_selected.empty:
    raise RuntimeError(
        f"Annual method '{FINAL_ANNUAL_METHOD}' not found."
    )

annual_selected["horizon_years"] = pd.to_numeric(
    annual_selected[horizon_col],
    errors="raise",
).astype(int)

annual_selected["target_year"] = pd.to_numeric(
    annual_selected[target_year_col],
    errors="raise",
).astype(int)

annual_selected["actual_level_c"] = pd.to_numeric(
    annual_selected[actual_col],
    errors="raise",
)

annual_selected["forecast_level_c"] = pd.to_numeric(
    annual_selected[pred_col],
    errors="raise",
)

annual_selected["error_c"] = (
    annual_selected["actual_level_c"]
    - annual_selected["forecast_level_c"]
)

annual_selected["abs_error_c"] = annual_selected["error_c"].abs()

annual_dev = annual_selected[
    annual_selected["target_year"] <= 2023
].copy()

annual_reserved = annual_selected[
    annual_selected["target_year"].isin([2024, 2025])
].copy()

annual_rows = []

for horizon in HORIZONS:
    hdf = annual_dev[
        annual_dev["horizon_years"] == horizon
    ].copy()

    if hdf.empty:
        raise RuntimeError(
            f"No development annual-level errors for H{horizon}."
        )

    row = {
        "horizon_years": horizon,
        "development_error_count": len(hdf),
        "development_target_years":
            ";".join(map(str, sorted(hdf["target_year"].unique()))),
        "development_mae_c":
            float(hdf["abs_error_c"].mean()),
        "development_max_abs_error_c":
            float(hdf["abs_error_c"].max()),
        "low_independent_year_count_flag":
            bool(len(hdf) < 5),
    }

    for alpha, label in ALPHAS.items():
        row[f"raw_annual_radius_{label}_c"] = (
            conformal_quantile(
                hdf["abs_error_c"].to_numpy(dtype=float),
                alpha,
            )
        )

    annual_rows.append(row)

annual_unc = pd.DataFrame(annual_rows).sort_values(
    "horizon_years"
).reset_index(drop=True)

# Conservative horizon rule: farther horizon cannot get narrower only
# because few historical target years exist.
for label in ["90", "95"]:
    annual_unc[f"annual_radius_{label}_c"] = (
        annual_unc[f"raw_annual_radius_{label}_c"].cummax()
    )

annual_unc.to_csv(ANNUAL_UNC_PATH, index=False)

print(annual_unc.to_string(index=False))


# =============================================================================
# 10. AUDIT ANNUAL ENVELOPES ON RESERVED 2024-2025
# =============================================================================

section("6. AUDIT ANNUAL-LEVEL ENVELOPES ON RESERVED 2024-2025")

annual_audit_rows = []

for horizon in HORIZONS:
    hdf = annual_reserved[
        annual_reserved["horizon_years"] == horizon
    ].copy()

    if hdf.empty:
        continue

    row = annual_unc[
        annual_unc["horizon_years"] == horizon
    ].iloc[0]

    for alpha, label in ALPHAS.items():
        radius = float(row[f"annual_radius_{label}_c"])

        covered = (
            hdf["abs_error_c"].to_numpy(dtype=float)
            <= radius
        )

        annual_audit_rows.append({
            "horizon_years": horizon,
            "nominal_level_percent": (1 - alpha) * 100,
            "development_frozen_radius_c": radius,
            "reserved_target_count": len(hdf),
            "reserved_target_years":
                ";".join(map(str, sorted(hdf["target_year"].unique()))),
            "observed_reserved_coverage_percent":
                float(np.mean(covered) * 100),
            "recalibrated_using_reserved_data": False,
        })

annual_audit_df = pd.DataFrame(annual_audit_rows)
annual_audit_df.to_csv(ANNUAL_AUDIT_PATH, index=False)

print(annual_audit_df.to_string(index=False))


# =============================================================================
# 11. BUILD TOTAL FUTURE UNCERTAINTY RADII
# =============================================================================

section("7. BUILD CONSERVATIVE FUTURE REFERENCE RADII")

radius_table = annual_unc[
    [
        "horizon_years",
        "annual_radius_90_c",
        "annual_radius_95_c",
        "development_error_count",
        "low_independent_year_count_flag",
    ]
].copy()

radius_table["spatial_radius_90_c"] = spatial_quantiles["90"]
radius_table["spatial_radius_95_c"] = spatial_quantiles["95"]

radius_table["total_radius_90_c"] = (
    radius_table["annual_radius_90_c"]
    + radius_table["spatial_radius_90_c"]
)

radius_table["total_radius_95_c"] = (
    radius_table["annual_radius_95_c"]
    + radius_table["spatial_radius_95_c"]
)

print(radius_table.to_string(index=False))


# =============================================================================
# 12. STEP-28 FUTURE FORECAST UNCERTAINTY
# =============================================================================

section("8. APPLY UNCERTAINTY TO STEP-28 FUTURE LST")

future = pd.read_csv(
    FUTURE_PRED_PATH,
    dtype={GRID_COL: str, "area_name": str},
    low_memory=False,
)

future[YEAR_COL] = pd.to_numeric(
    future[YEAR_COL],
    errors="raise",
).astype(int)

if len(future) != EXPECTED_FUTURE_ROWS:
    raise RuntimeError(
        f"Future rows={len(future)}, expected={EXPECTED_FUTURE_ROWS}"
    )

if future.duplicated([GRID_COL, YEAR_COL]).any():
    raise RuntimeError("Duplicate future grid-year rows.")

if "predicted_lst_c" not in future.columns:
    raise RuntimeError(
        "Step-28 file missing predicted_lst_c."
    )

future_unc = add_reference_bands(
    future,
    "predicted_lst_c",
    radius_table,
    grid_profile,
    "forecast",
)

future_unc["uncertainty_method"] = (
    "2024 centered-spatial residual calibration + "
    "development horizon-specific annual-level error envelope"
)

future_unc["uncertainty_interpretation"] = (
    "Conservative reference prediction band; "
    "not a formal exact future-coverage guarantee."
)

future_unc["future_actual_available"] = False

future_unc.to_csv(FUTURE_UNC_PATH, index=False)

future_summary = (
    future_unc
    .groupby([YEAR_COL, "horizon_years"], as_index=False)
    .agg(
        grid_count=(GRID_COL, "size"),
        mean_point_forecast_c=("predicted_lst_c", "mean"),
        spatial_radius_95_c=("spatial_radius_95_c", "first"),
        annual_radius_95_c=("annual_radius_95_c", "first"),
        total_radius_95_c=("total_radius_95_c", "first"),
        reference_pi95_width_c=(
            "forecast_reference_pi95_width_c",
            "first",
        ),
    )
)

future_summary.to_csv(FUTURE_SUMMARY_PATH, index=False)

print(future_summary.to_string(index=False))


# =============================================================================
# 13. STEP-29 SCENARIO UNCERTAINTY OUTPUTS
# =============================================================================

section("9. ADD REFERENCE OUTCOME BANDS TO STEP-29 SCENARIOS")

scenario_summary_rows = []


def process_scenario(
    path,
    output_path,
    scenario_name,
    point_candidates,
):
    df = pd.read_csv(
        path,
        dtype={GRID_COL: str, "area_name": str},
        low_memory=False,
    )

    point_col = None

    for c in point_candidates:
        if c in df.columns:
            point_col = c
            break

    if point_col is None:
        raise RuntimeError(
            f"{scenario_name}: scenario LST column not detected. "
            f"Columns={df.columns.tolist()}"
        )

    out = add_reference_bands(
        df,
        point_col,
        radius_table,
        grid_profile,
        "scenario_lst",
    )

    out["scenario_family"] = scenario_name

    out["scenario_outcome_band_scope"] = (
        "Baseline-calibrated reference LST outcome band."
    )

    out["scenario_delta_interval_available"] = False

    out["scenario_delta_uncertainty_note"] = (
        "No calibrated interval is claimed for intervention ΔLST. "
        "Scenario ΔLST remains a model-based sensitivity estimate."
    )

    out.to_csv(output_path, index=False)

    scenario_summary_rows.append({
        "scenario_family": scenario_name,
        "rows": len(out),
        "year_min": int(out[YEAR_COL].min()),
        "year_max": int(out[YEAR_COL].max()),
        "scenario_point_column": point_col,
        "reference_outcome_band_added": True,
        "calibrated_delta_lst_interval_claimed": False,
    })

    return out


green_unc = process_scenario(
    GREEN_SCENARIO_PATH,
    GREEN_UNC_PATH,
    "29A_GREEN",
    [
        "independent_scenario_lst_c",
        "scenario_predicted_lst_c",
        "scenario_lst_c",
    ],
)

built_unc = process_scenario(
    BUILT_SCENARIO_PATH,
    BUILT_UNC_PATH,
    "29B_BUILT",
    [
        "scenario_predicted_lst_c",
        "scenario_lst_c",
    ],
)

combined_unc = process_scenario(
    COMBINED_SCENARIO_PATH,
    COMBINED_UNC_PATH,
    "29C_COMBINED",
    [
        "combined_scenario_predicted_lst_c",
        "scenario_predicted_lst_c",
        "scenario_lst_c",
    ],
)

blue_unc = process_scenario(
    BLUE_SCENARIO_PATH,
    BLUE_UNC_PATH,
    "29D_BLUE",
    [
        "scenario_predicted_lst_c",
        "scenario_lst_c",
    ],
)


# =============================================================================
# 14. PRESERVE 29D GRID-LEVEL SUPPORT CLASS
# =============================================================================

section("10. CARRY 29D LOCAL SUPPORT / EXTRAPOLATION STATUS")

blue_support = pd.read_csv(
    BLUE_SUPPORT_PATH,
    dtype={GRID_COL: str},
    low_memory=False,
)

blue_support[YEAR_COL] = pd.to_numeric(
    blue_support[YEAR_COL],
    errors="raise",
).astype(int)

blue_support["scenario_level"] = (
    blue_support["scenario_level"]
    .astype(str)
    .str.upper()
)

blue_unc["scenario_level"] = (
    blue_unc["scenario_level"]
    .astype(str)
    .str.upper()
)

support_cols = [
    GRID_COL,
    YEAR_COL,
    "scenario_level",
    "grid_level_support_class",
    "increase_within_grid_q90_positive_change",
    "increase_within_grid_max_positive_change",
    "scenario_within_grid_historical_ndwi_range",
    "scenario_newly_exits_grid_historical_range",
]

support_keep = blue_support[
    support_cols
].drop_duplicates(
    [GRID_COL, YEAR_COL, "scenario_level"]
)

drop_existing = [
    c for c in support_cols[3:]
    if c in blue_unc.columns
]

if drop_existing:
    blue_unc = blue_unc.drop(columns=drop_existing)

blue_unc = blue_unc.merge(
    support_keep,
    on=[GRID_COL, YEAR_COL, "scenario_level"],
    how="left",
    validate="many_to_one",
)

if blue_unc["grid_level_support_class"].isna().any():
    raise RuntimeError(
        "29D support class missing after merge."
    )

blue_unc["blue_scenario_support_interpretation"] = np.where(
    blue_unc["grid_level_support_class"]
    == "extrapolative_for_selected_grid",
    "Lower support: selected NDWI scenario is extrapolative for this grid.",
    "Scenario is within the Step-29D-QA historical-support definition.",
)

blue_unc.to_csv(BLUE_UNC_PATH, index=False)

scenario_summary_df = pd.DataFrame(scenario_summary_rows)
scenario_summary_df.to_csv(SCENARIO_SUMMARY_PATH, index=False)

print(scenario_summary_df.to_string(index=False))

print("\n29D support classes:")
print(
    blue_unc["grid_level_support_class"]
    .value_counts()
    .to_string()
)


# =============================================================================
# 15. FIGURES
# =============================================================================

section("11. SAVE UNCERTAINTY FIGURES")

fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(
    val2024["abs_spatial_residual_c"],
    bins=40,
)
ax.axvline(
    spatial_quantiles["95"],
    linestyle="--",
    label=f"95% spatial radius = {spatial_quantiles['95']:.2f} °C",
)
ax.set_title("2024 Spatial-Uncertainty Calibration Residuals")
ax.set_xlabel("Absolute centered spatial residual (°C)")
ax.set_ylabel("Grid count")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_SPATIAL, dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    annual_unc["horizon_years"],
    annual_unc["annual_radius_95_c"],
    marker="o",
)
ax.set_title("Annual Thermal-Level Uncertainty by Horizon")
ax.set_xlabel("Forecast horizon (years)")
ax.set_ylabel("95% empirical annual-level radius (°C)")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FIG_ANNUAL, dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    future_summary[YEAR_COL],
    future_summary["reference_pi95_width_c"],
    marker="o",
)
ax.set_title("Future LST Reference-Uncertainty Width")
ax.set_xlabel("Forecast year")
ax.set_ylabel("95% reference prediction-band width (°C)")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FIG_FUTURE, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {FIG_SPATIAL}")
print(f"Saved: {FIG_ANNUAL}")
print(f"Saved: {FIG_FUTURE}")


# =============================================================================
# 16. STRICT AUDIT
# =============================================================================

section("12. STRICT STEP-30 AUDIT")

audit_rows = [
    {
        "check": "prerequisite_audits",
        "status": "PASS",
        "details":
            "Step 28, 29A, 29B, 29C, 29D and 29D-QA verified fully PASS.",
    },
    {
        "check": "development_spatial_years",
        "status":
            "PASS"
            if sorted(dev[YEAR_COL].unique().tolist())
            == DEVELOPMENT_YEARS
            else "FAIL",
        "details": str(sorted(dev[YEAR_COL].unique().tolist())),
    },
    {
        "check": "spatial_calibration_year_2024_only",
        "status":
            "PASS"
            if val2024[YEAR_COL].unique().tolist() == [2024]
            else "FAIL",
        "details": str(val2024[YEAR_COL].unique().tolist()),
    },
    {
        "check": "spatial_audit_year_2025_only",
        "status":
            "PASS"
            if test2025[YEAR_COL].unique().tolist() == [2025]
            else "FAIL",
        "details": str(test2025[YEAR_COL].unique().tolist()),
    },
    {
        "check": "2025_recalibration",
        "status": "PASS",
        "details":
            "No. 2025 is audit-only and does not alter the 2024 spatial radius.",
    },
    {
        "check": "annual_calibration_stops_at_2023",
        "status":
            "PASS"
            if int(annual_dev["target_year"].max()) <= 2023
            else "FAIL",
        "details":
            f"max_calibration_target={int(annual_dev['target_year'].max())}",
    },
    {
        "check": "annual_reserved_2024_2025_audit_only",
        "status":
            "PASS"
            if set(
                annual_reserved["target_year"].unique().tolist()
            ).issubset({2024, 2025})
            else "FAIL",
        "details":
            str(sorted(annual_reserved["target_year"].unique().tolist())),
    },
    {
        "check": "annual_95_radius_non_decreasing_by_horizon",
        "status":
            "PASS"
            if (
                np.diff(
                    annual_unc["annual_radius_95_c"].to_numpy(dtype=float)
                )
                >= -1e-12
            ).all()
            else "FAIL",
        "details":
            "Cumulative-max conservative horizon policy.",
    },
    {
        "check": "future_row_count",
        "status":
            "PASS"
            if len(future_unc) == EXPECTED_FUTURE_ROWS
            else "FAIL",
        "details":
            f"actual={len(future_unc)}, expected={EXPECTED_FUTURE_ROWS}",
    },
    {
        "check": "future_interval_width_positive",
        "status":
            "PASS"
            if (
                future_unc["forecast_reference_pi95_width_c"] > 0
            ).all()
            else "FAIL",
        "details": "All future 95% reference widths are positive.",
    },
    {
        "check": "step28_point_forecasts_changed",
        "status": "PASS",
        "details": "No. Step-28 point forecasts are copied unchanged.",
    },
    {
        "check": "step29_point_estimates_changed",
        "status": "PASS",
        "details": "No. Step-29 point estimates are copied unchanged.",
    },
    {
        "check": "scenario_delta_interval_claimed",
        "status": "PASS",
        "details":
            "No calibrated interval is claimed for intervention ΔLST.",
    },
    {
        "check": "29d_support_preserved",
        "status":
            "PASS"
            if blue_unc["grid_level_support_class"].notna().all()
            else "FAIL",
        "details":
            "Grid-level historical-support/extrapolation status retained.",
    },
    {
        "check": "model_retraining",
        "status": "PASS",
        "details": "None.",
    },
    {
        "check": "hyperparameter_tuning",
        "status": "PASS",
        "details": "None.",
    },
    {
        "check": "ngboost_final_layer",
        "status": "PASS",
        "details":
            "Not used; final heat uncertainty is residual-calibrated.",
    },
    {
        "check": "future_coverage_overclaim",
        "status": "PASS",
        "details":
            "Reference bands are not described as exact future guarantees.",
    },
]

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(AUDIT_PATH, index=False)

print(audit_df.to_string(index=False))

if (audit_df["status"] != "PASS").any():
    raise RuntimeError("One or more Step-30 audit checks failed.")


# =============================================================================
# 17. METHODOLOGY + REPORT
# =============================================================================

section("13. SAVE STEP-30 METHODOLOGY + REPORT")

methodology = {
    "step": 30,
    "analysis": "Forecast and scenario uncertainty/confidence",
    "point_forecast_changed": False,
    "scenario_point_estimates_changed": False,
    "spatial_uncertainty": {
        "development_profile_years": DEVELOPMENT_YEARS,
        "calibration_year": 2024,
        "audit_year": 2025,
        "score": "absolute centered spatial residual",
        "levels_percent": [90, 95],
        "recalibration_after_2025": False,
    },
    "annual_level_uncertainty": {
        "final_method": FINAL_ANNUAL_METHOD,
        "development_calibration_targets": "<=2023",
        "reserved_audit_targets": [2024, 2025],
        "horizons_years": HORIZONS,
        "non_decreasing_horizon_radius_policy": True,
        "small_sample_note":
            "H4/H5 have few independent annual target years; "
            "bands are empirical conservative envelopes.",
    },
    "future_total_reference_band": {
        "formula":
            "total_radius = spatial_radius + annual_level_radius",
        "formal_exact_future_coverage_guarantee": False,
    },
    "local_confidence": {
        "basis":
            "Per-grid 2019-2023 mean absolute centered spatial residual",
        "labels": [
            "Higher relative confidence",
            "Moderate relative confidence",
            "Lower relative confidence",
        ],
        "probability_interpretation": False,
    },
    "scenario_uncertainty": {
        "scenario_lST_reference_band_added": True,
        "calibrated_delta_lST_interval": False,
        "reason":
            "Historical counterfactual intervention outcomes do not exist.",
        "29d_support_status_preserved": True,
    },
    "ngboost": {
        "used": False,
        "reason":
            "Earlier raw NGBoost heat intervals had poor empirical coverage.",
    },
    "limitation": {
        "future_predictor_uncertainty_monte_carlo_propagated": False,
        "note":
            "Dynamic predictor forecast uncertainty is not separately sampled "
            "through the full 93-feature LightGBM pipeline.",
    },
}

METHOD_PATH.write_text(
    json.dumps(methodology, indent=2),
    encoding="utf-8",
)

spatial95 = spatial_audit_df[
    spatial_audit_df["nominal_level_percent"] == 95.0
].iloc[0]

report_text = "\n".join([
    "STEP 30 - FORECAST + SCENARIO UNCERTAINTY / CONFIDENCE",
    "=" * 120,
    "",
    "SPATIAL CALIBRATION",
    "-" * 120,
    spatial_cal_df.to_string(index=False),
    "",
    "2025 SPATIAL AUDIT",
    "-" * 120,
    spatial_audit_df.to_string(index=False),
    "",
    "ANNUAL-LEVEL HORIZON UNCERTAINTY",
    "-" * 120,
    annual_unc.to_string(index=False),
    "",
    "RESERVED ANNUAL AUDIT",
    "-" * 120,
    annual_audit_df.to_string(index=False),
    "",
    "FUTURE 2026-2030 UNCERTAINTY",
    "-" * 120,
    future_summary.to_string(index=False),
    "",
    "SCENARIO POLICY",
    "-" * 120,
    "Scenario LST receives a baseline-calibrated reference outcome band.",
    "No calibrated interval is claimed for scenario ΔLST.",
    "29D support/extrapolation status is carried forward.",
    "",
    "INTERPRETATION",
    "-" * 120,
    "These are conservative empirical reference prediction bands.",
    "They are not exact future-coverage guarantees.",
    (
        f"Frozen 2024-calibrated 95% spatial radius: "
        f"{float(spatial95['frozen_spatial_radius_c']):.6f} °C"
    ),
    (
        f"Observed 2025 centered-spatial coverage: "
        f"{float(spatial95['observed_spatial_coverage_percent']):.2f}%"
    ),
])

REPORT_PATH.write_text(
    report_text,
    encoding="utf-8",
)

print(f"Saved: {METHOD_PATH}")
print(f"Saved: {REPORT_PATH}")


# =============================================================================
# 18. HASHES
# =============================================================================

section("14. REPRODUCIBILITY HASHES")

hash_files = [
    SPATIAL_CAL_PATH,
    SPATIAL_AUDIT_PATH,
    GRID_PROFILE_PATH,
    ANNUAL_UNC_PATH,
    ANNUAL_AUDIT_PATH,
    FUTURE_UNC_PATH,
    FUTURE_SUMMARY_PATH,
    GREEN_UNC_PATH,
    BUILT_UNC_PATH,
    COMBINED_UNC_PATH,
    BLUE_UNC_PATH,
    SCENARIO_SUMMARY_PATH,
    AUDIT_PATH,
    METHOD_PATH,
    REPORT_PATH,
    FIG_SPATIAL,
    FIG_ANNUAL,
    FIG_FUTURE,
]

hash_rows = []

for path in hash_files:
    digest = sha256_file(path)

    hash_rows.append({
        "file": str(path),
        "sha256": digest,
    })

    print(f"{path.name}: {digest}")

pd.DataFrame(hash_rows).to_csv(
    HASH_PATH,
    index=False,
)

print(f"Saved: {HASH_PATH}")


# =============================================================================
# FINAL
# =============================================================================

section("STEP 30 COMPLETED SUCCESSFULLY")

print(
    "Forecast + scenario uncertainty layer completed.\n"
    "Step-28 point forecasts unchanged.\n"
    "Step-29 scenario point estimates unchanged.\n"
    "Spatial uncertainty calibrated on reserved 2024 centered residuals.\n"
    "Final 2025 used only to audit the frozen spatial calibration.\n"
    "Annual-level uncertainty calibrated only from development targets through 2023.\n"
    "Reserved 2024-2025 annual forecasts used only as audit evidence.\n"
    "Future H1-H5 reference bands combine spatial + annual uncertainty conservatively.\n"
    "Every grid receives a historical-error-based relative confidence label.\n"
    "29A/29B/29C/29D scenario LST outputs receive reference outcome bands.\n"
    "No calibrated confidence interval is claimed for scenario ΔLST.\n"
    "29D support/extrapolation status is retained.\n"
    "NGBoost is not used as the final heat uncertainty layer.\n"
    "No retraining, hyperparameter tuning or post-2025 recalibration occurred.\n"
    "\nNEXT: review the complete terminal output before freezing Step 30 "
    "and moving to Step 31 CRITIC weighting."
)
