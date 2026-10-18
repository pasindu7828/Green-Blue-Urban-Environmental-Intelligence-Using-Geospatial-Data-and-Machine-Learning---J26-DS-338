"""
DYNAMIC 5-YEAR FORECAST - STEP 04
=================================
Project the validated annual Kaduwela thermal level and base predictors for
Y+1 ... Y+5, where Y is the latest observed year in Step 01 validated history.

This preserves the Step-27 selected forecasting strategy used by original
Step 28:
- Annual thermal level: Expanding Mean
- NDVI: Rolling-3 Mean
- NDBI: Rolling-3 Mean
- NDWI: Rolling-3 Mean
- EVI: Rolling-3 Mean
- Albedo: Bounded Theil-Sen Trend
- NDBI_ADJ: Rolling-3 Mean
- NDVI_ADJ: Rolling-3 Mean
- NightLights: Rolling-3 Mean
- green_mask: Rolling-3 Mean

Static/structural predictors are carried forward from the latest observed year.
No model inference occurs in this step.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]

INPUT_PATH = EXPERIMENT_ROOT / "outputs" / "01_validated_history.csv"
STEP01_SUMMARY = EXPERIMENT_ROOT / "outputs" / "reports" / "01_upload_validation_summary.json"
STEP27_FREEZE = PROJECT_ROOT / "outputs" / "reports" / "27_forecast_strategy_freeze_for_step28.json"

PREDICTION_DIR = EXPERIMENT_ROOT / "outputs" / "predictions"
REPORT_DIR = EXPERIMENT_ROOT / "outputs" / "reports"

REFERENCE_BASE_2026_2030 = (
    PROJECT_ROOT / "outputs" / "forecasting"
    / "28_future_base_predictors_2026_2030.csv"
)
REFERENCE_ANNUAL_2026_2030 = (
    PROJECT_ROOT / "outputs" / "forecasting"
    / "28_future_annual_thermal_level_2026_2030.csv"
)

AUDIT_PATH = REPORT_DIR / "04_dynamic_future_input_projection_audit.csv"
METHOD_PATH = REPORT_DIR / "04_dynamic_future_input_projection_methodology.json"
SUMMARY_PATH = REPORT_DIR / "04_dynamic_future_input_projection_summary.json"


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET_COL = "LST_C"
EXPECTED_GRIDS = 2194
FORECAST_HORIZON = 5
REPRO_TOL = 1e-8

TEMPORAL_SOURCES = [
    "NDVI",
    "NDBI",
    "NDWI",
    "EVI",
    "Albedo",
    "NDBI_ADJ",
    "NDVI_ADJ",
    "NightLights",
    "green_mask",
]

STATIC_PREDICTORS = [
    "building_mask",
    "road_mask",
]

STRUCTURAL_PREDICTORS = [
    "dist_road",
    "dist_main_road",
    "dist_water",
    "LCZ",
]

EXPECTED_SELECTION = {
    "annual_level": "Expanding Mean",
    "dynamic_predictors": {
        "NDVI": "Rolling-3 Mean",
        "NDBI": "Rolling-3 Mean",
        "NDWI": "Rolling-3 Mean",
        "EVI": "Rolling-3 Mean",
        "Albedo": "Bounded Theil-Sen Trend",
        "NDBI_ADJ": "Rolling-3 Mean",
        "NDVI_ADJ": "Rolling-3 Mean",
        "NightLights": "Rolling-3 Mean",
        "green_mask": "Rolling-3 Mean",
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 125)
    print(title)
    print("=" * 125)


def add_audit(rows, check, passed, details):
    status = "PASS" if passed else "FAIL"
    rows.append({"check": check, "status": status, "details": details})
    print(f"{status:4s} | {check}: {details}")


def theil_sen_forecast_matrix(history_values, history_years, target_year):
    """Exact vectorized rule used by original Step 28."""
    x = np.asarray(history_years, dtype=float)
    y = np.asarray(history_values, dtype=float)

    pair_slopes = []
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            pair_slopes.append(
                (y[:, j] - y[:, i]) / (x[j] - x[i])
            )

    slopes = np.median(
        np.stack(pair_slopes, axis=1),
        axis=1,
    )

    intercepts = (
        np.median(y, axis=1)
        - slopes * np.median(x)
    )

    return intercepts + slopes * float(target_year)


def forecast_dynamic_predictor(wide, history_years, method, target_year):
    history = wide[history_years].to_numpy(dtype=float)

    if method == "Rolling-3 Mean":
        prediction = np.mean(history[:, -3:], axis=1)

    elif method == "Bounded Theil-Sen Trend":
        prediction = theil_sen_forecast_matrix(
            history_values=history,
            history_years=history_years,
            target_year=target_year,
        )
        # Preserve original Step-28 global historical bound rule.
        prediction = np.clip(
            prediction,
            float(np.min(history)),
            float(np.max(history)),
        )

    elif method == "Persistence":
        prediction = history[:, -1].copy()

    elif method == "Bounded OLS Trend":
        x = np.asarray(history_years, dtype=float)
        x_mean = float(np.mean(x))
        x_centered = x - x_mean
        denominator = float(np.sum(x_centered ** 2))

        y_mean = np.mean(history, axis=1)
        slopes = (
            np.sum(
                (history - y_mean[:, None]) * x_centered[None, :],
                axis=1,
            )
            / denominator
        )

        prediction = (
            y_mean
            + slopes * (float(target_year) - x_mean)
        )

        prediction = np.clip(
            prediction,
            float(np.min(history)),
            float(np.max(history)),
        )

    else:
        raise ValueError(f"Unsupported frozen predictor method: {method}")

    return prediction


def max_numeric_diff_by_key(left, right, columns):
    left = left.copy()
    right = right.copy()
    left[GRID_COL] = left[GRID_COL].astype(str)
    right[GRID_COL] = right[GRID_COL].astype(str)

    l = left[[GRID_COL, YEAR_COL] + columns].sort_values([YEAR_COL, GRID_COL]).reset_index(drop=True)
    r = right[[GRID_COL, YEAR_COL] + columns].sort_values([YEAR_COL, GRID_COL]).reset_index(drop=True)

    if not l[[GRID_COL, YEAR_COL]].equals(r[[GRID_COL, YEAR_COL]]):
        raise RuntimeError("Reference reproduction keys do not align.")

    diffs = {}
    for column in columns:
        a = pd.to_numeric(l[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(r[column], errors="coerce").to_numpy(dtype=float)
        diffs[column] = float(np.nanmax(np.abs(a - b)))

    return diffs


# =============================================================================
# START
# =============================================================================

PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

section("DYNAMIC 5-YEAR FORECAST - STEP 04 FUTURE INPUT PROJECTION")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Step 01 validated history missing: {INPUT_PATH}")

if not STEP01_SUMMARY.exists():
    raise FileNotFoundError(f"Step 01 summary missing: {STEP01_SUMMARY}")

if not STEP27_FREEZE.exists():
    raise FileNotFoundError(f"Step 27 frozen strategy missing: {STEP27_FREEZE}")

with STEP01_SUMMARY.open("r", encoding="utf-8") as f:
    step01 = json.load(f)

if step01.get("status") != "PASS":
    raise RuntimeError("Step 01 upload validation is not PASS.")

with STEP27_FREEZE.open("r", encoding="utf-8") as f:
    freeze = json.load(f)

annual_method = freeze["annual_level"]["selected_method"]
selected_methods = freeze["dynamic_predictors"]["selected_methods"]

if annual_method != EXPECTED_SELECTION["annual_level"]:
    raise RuntimeError(f"Unexpected annual-level method: {annual_method}")

if selected_methods != EXPECTED_SELECTION["dynamic_predictors"]:
    raise RuntimeError(
        "Frozen Step-27 dynamic predictor methods differ from expected finalized selection."
    )

base = pd.read_csv(
    INPUT_PATH,
    dtype={GRID_COL: str, "area_name": str},
    low_memory=False,
)
base[YEAR_COL] = pd.to_numeric(base[YEAR_COL], errors="raise").astype(int)

history_years = sorted(base[YEAR_COL].unique().tolist())
latest_year = int(history_years[-1])
future_years = list(range(latest_year + 1, latest_year + FORECAST_HORIZON + 1))
expected_future_rows = EXPECTED_GRIDS * FORECAST_HORIZON

print(f"Observed history       : {history_years[0]}-{history_years[-1]}")
print(f"Forecast origin year   : {latest_year}")
print(f"Future years           : {future_years}")
print(f"Annual method          : {annual_method}")
print("Dynamic predictor methods:")
for predictor, method in selected_methods.items():
    print(f"  {predictor:12s} -> {method}")


# =============================================================================
# 1. VALIDATE OBSERVED INPUT
# =============================================================================

section("1. VALIDATE OBSERVED INPUT")

audit_rows = []

add_audit(
    audit_rows,
    "unique_grid_count",
    base[GRID_COL].nunique() == EXPECTED_GRIDS,
    f"actual={base[GRID_COL].nunique()}, expected={EXPECTED_GRIDS}",
)

rows_per_year = base.groupby(YEAR_COL)[GRID_COL].nunique()
add_audit(
    audit_rows,
    "balanced_observed_grid_panel",
    bool((rows_per_year == EXPECTED_GRIDS).all()),
    f"min={rows_per_year.min()}, max={rows_per_year.max()}",
)

missing_required = [
    c for c in (
        [GRID_COL, YEAR_COL, TARGET_COL, "area_name"]
        + TEMPORAL_SOURCES
        + STATIC_PREDICTORS
        + STRUCTURAL_PREDICTORS
    )
    if c not in base.columns
]
add_audit(
    audit_rows,
    "required_base_predictors_present",
    not missing_required,
    f"missing={missing_required}",
)

if missing_required or any(x["status"] == "FAIL" for x in audit_rows):
    pd.DataFrame(audit_rows).to_csv(AUDIT_PATH, index=False)
    raise RuntimeError("Observed input failed Step 04 prerequisites.")


# =============================================================================
# 2. PROJECT DYNAMIC + STATIC BASE PREDICTORS
# =============================================================================

section(f"2. PROJECT BASE PREDICTORS FOR {future_years[0]}-{future_years[-1]}")

base_sorted = (
    base.sort_values([GRID_COL, YEAR_COL])
    .reset_index(drop=True)
)

grid_order = sorted(base_sorted[GRID_COL].unique().tolist())

latest_observed = (
    base_sorted[base_sorted[YEAR_COL] == latest_year]
    .copy()
    .set_index(GRID_COL)
    .loc[grid_order]
)

# Build predictor-wide tables once from observed data only.
wide_by_predictor = {}
for predictor in TEMPORAL_SOURCES:
    wide = (
        base_sorted.pivot(
            index=GRID_COL,
            columns=YEAR_COL,
            values=predictor,
        )
        .loc[grid_order, history_years]
    )
    wide_by_predictor[predictor] = wide

future_frames = []

for future_year in future_years:
    frame = pd.DataFrame(
        {
            GRID_COL: grid_order,
            YEAR_COL: future_year,
        }
    ).set_index(GRID_COL)

    for predictor, method in selected_methods.items():
        frame[predictor] = forecast_dynamic_predictor(
            wide=wide_by_predictor[predictor],
            history_years=history_years,
            method=method,
            target_year=future_year,
        )

    for predictor in STATIC_PREDICTORS + STRUCTURAL_PREDICTORS:
        frame[predictor] = latest_observed[predictor].astype(float)

    frame["area_name"] = latest_observed["area_name"].astype(str)

    frame = frame.reset_index()
    frame[TARGET_COL] = np.nan
    future_frames.append(frame)

future_base = pd.concat(future_frames, ignore_index=True)

if len(future_base) != expected_future_rows:
    raise RuntimeError(
        f"Future row mismatch: actual={len(future_base)}, expected={expected_future_rows}"
    )

if future_base.duplicated([GRID_COL, YEAR_COL]).any():
    raise RuntimeError("Duplicate future grid-year rows detected.")

required_future_predictors = TEMPORAL_SOURCES + STATIC_PREDICTORS + STRUCTURAL_PREDICTORS
missing_future_values = int(future_base[required_future_predictors].isna().sum().sum())
if missing_future_values:
    raise RuntimeError(f"Projected future predictors contain {missing_future_values} missing cells.")

future_base_path = (
    PREDICTION_DIR
    / f"04_future_base_predictors_{future_years[0]}_{future_years[-1]}.csv"
)
future_base.to_csv(future_base_path, index=False)

for year in future_years:
    future_base[future_base[YEAR_COL] == year].to_csv(
        PREDICTION_DIR / f"04_future_base_predictors_{year}.csv",
        index=False,
    )

print(f"Future base rows : {len(future_base):,}")
print(f"Saved            : {future_base_path}")
print("\nFuture predictor annual means:")
print(future_base.groupby(YEAR_COL)[TEMPORAL_SOURCES].mean().to_string())


# =============================================================================
# 3. PROJECT ANNUAL KADUWELA THERMAL LEVEL
# =============================================================================

section("3. PROJECT ANNUAL KADUWELA THERMAL LEVEL")

annual_history = (
    base.groupby(YEAR_COL, as_index=False)
    .agg(
        annual_mean_lst_c=(TARGET_COL, "mean"),
        annual_median_lst_c=(TARGET_COL, "median"),
        annual_std_lst_c=(TARGET_COL, "std"),
        grid_count=(GRID_COL, "nunique"),
    )
    .sort_values(YEAR_COL)
    .reset_index(drop=True)
)

if annual_method != "Expanding Mean":
    raise RuntimeError(
        "Dynamic Step 04 currently preserves the finalized Expanding Mean annual-level strategy."
    )

forecast_annual_mean = float(annual_history["annual_mean_lst_c"].mean())

annual_future = pd.DataFrame(
    {
        YEAR_COL: future_years,
        "forecast_origin_year": latest_year,
        "horizon_years": [year - latest_year for year in future_years],
        "annual_level_method": annual_method,
        "forecast_annual_mean_lst_c": forecast_annual_mean,
    }
)

annual_history_path = (
    PREDICTION_DIR
    / f"04_observed_annual_thermal_level_{history_years[0]}_{latest_year}.csv"
)
annual_future_path = (
    PREDICTION_DIR
    / f"04_future_annual_thermal_level_{future_years[0]}_{future_years[-1]}.csv"
)

annual_history.to_csv(annual_history_path, index=False)
annual_future.to_csv(annual_future_path, index=False)

print(annual_future.to_string(index=False))
print(f"\nObserved annual series : {annual_history_path}")
print(f"Future annual levels   : {annual_future_path}")


# =============================================================================
# 4. ORIGINAL 2025 -> 2026-2030 REPRODUCTION CHECK
# =============================================================================

section("4. ORIGINAL 2025 BASELINE REPRODUCTION CHECK")

reference_status = "NOT_APPLICABLE"
max_base_repro_error = None
max_annual_repro_error = None

if latest_year == 2025 and history_years[0] == 2015:
    if not REFERENCE_BASE_2026_2030.exists():
        raise FileNotFoundError(
            f"Original Step-28 future base file missing: {REFERENCE_BASE_2026_2030}"
        )
    if not REFERENCE_ANNUAL_2026_2030.exists():
        raise FileNotFoundError(
            f"Original Step-28 annual-level file missing: {REFERENCE_ANNUAL_2026_2030}"
        )

    ref_base = pd.read_csv(
        REFERENCE_BASE_2026_2030,
        dtype={GRID_COL: str, "area_name": str},
        low_memory=False,
    )
    ref_base[YEAR_COL] = pd.to_numeric(ref_base[YEAR_COL], errors="raise").astype(int)

    numeric_compare_cols = TEMPORAL_SOURCES + STATIC_PREDICTORS + STRUCTURAL_PREDICTORS
    diffs = max_numeric_diff_by_key(future_base, ref_base, numeric_compare_cols)
    max_base_repro_error = max(diffs.values())

    dyn_area = (
        future_base[[GRID_COL, YEAR_COL, "area_name"]]
        .sort_values([YEAR_COL, GRID_COL])
        .reset_index(drop=True)
    )
    ref_area = (
        ref_base[[GRID_COL, YEAR_COL, "area_name"]]
        .sort_values([YEAR_COL, GRID_COL])
        .reset_index(drop=True)
    )
    area_equal = dyn_area.equals(ref_area)

    ref_annual = pd.read_csv(REFERENCE_ANNUAL_2026_2030)
    ref_annual[YEAR_COL] = pd.to_numeric(ref_annual[YEAR_COL], errors="raise").astype(int)

    annual_cmp = (
        annual_future[[YEAR_COL, "forecast_annual_mean_lst_c"]]
        .merge(
            ref_annual[[YEAR_COL, "forecast_annual_mean_lst_c"]],
            on=YEAR_COL,
            suffixes=("_dynamic", "_reference"),
            validate="one_to_one",
        )
    )
    max_annual_repro_error = float(
        np.max(
            np.abs(
                annual_cmp["forecast_annual_mean_lst_c_dynamic"].to_numpy(dtype=float)
                - annual_cmp["forecast_annual_mean_lst_c_reference"].to_numpy(dtype=float)
            )
        )
    )

    print(f"Max base-predictor reproduction error : {max_base_repro_error:.12e}")
    print(f"Area-name reproduction                : {area_equal}")
    print(f"Max annual-level reproduction error   : {max_annual_repro_error:.12e} °C")

    if max_base_repro_error > REPRO_TOL or not area_equal or max_annual_repro_error > REPRO_TOL:
        raise RuntimeError(
            "Dynamic Step 04 does not reproduce the validated original 2026-2030 input projection."
        )

    reference_status = "PASS"
    print("PASS: dynamic projection exactly reproduces original Step-28 inputs within tolerance.")
else:
    print(
        "Reference reproduction check not applicable because uploaded history "
        "extends beyond the original 2015-2025 period."
    )


# =============================================================================
# 5. AUDIT + METHODOLOGY
# =============================================================================

section("5. SAVE AUDIT + METHODOLOGY")

add_audit(
    audit_rows,
    "step27_strategy_preserved",
    annual_method == EXPECTED_SELECTION["annual_level"]
    and selected_methods == EXPECTED_SELECTION["dynamic_predictors"],
    f"annual={annual_method}; dynamic={selected_methods}",
)
add_audit(
    audit_rows,
    "future_year_window",
    future_years == list(range(latest_year + 1, latest_year + 6)),
    f"origin={latest_year}, future={future_years}",
)
add_audit(
    audit_rows,
    "future_row_count",
    len(future_base) == expected_future_rows,
    f"actual={len(future_base)}, expected={expected_future_rows}",
)
add_audit(
    audit_rows,
    "future_predictor_missing_cells",
    missing_future_values == 0,
    f"count={missing_future_values}",
)
add_audit(
    audit_rows,
    "original_2025_projection_reproduction",
    reference_status != "FAIL",
    (
        f"status={reference_status}, max_base_error={max_base_repro_error}, "
        f"max_annual_error={max_annual_repro_error}"
    ),
)

failed_audit = [x for x in audit_rows if x["status"] == "FAIL"]
pd.DataFrame(audit_rows).to_csv(AUDIT_PATH, index=False)

methodology = {
    "step": 4,
    "analysis": "Dynamic rolling five-year future input projection",
    "observed_period": f"{history_years[0]}-{latest_year}",
    "forecast_origin_year": latest_year,
    "future_years": future_years,
    "forecast_horizon_years": [1, 2, 3, 4, 5],
    "annual_level_method": annual_method,
    "dynamic_predictor_methods": selected_methods,
    "static_predictor_policy": "Carry latest observed value forward for building_mask and road_mask.",
    "structural_predictor_policy": "Carry latest observed dist_road, dist_main_road, dist_water and LCZ forward.",
    "area_name_policy": "Carry latest observed grid-to-area mapping forward.",
    "future_engineered_feature_policy": (
        "Do not forecast engineered features independently. Step 05 must combine observed + "
        "projected base predictors and rebuild the exact Step-07 feature engineering sequence."
    ),
    "future_target_policy": "LST_C is unknown for projected future rows and remains null until model inference.",
    "original_2025_reproduction_status": reference_status,
}

METHOD_PATH.write_text(json.dumps(methodology, indent=2), encoding="utf-8")

summary = {
    "status": "PASS" if not failed_audit else "FAIL",
    "latest_observed_year": latest_year,
    "forecast_years": future_years,
    "future_rows": int(len(future_base)),
    "future_base_path": str(future_base_path),
    "annual_future_path": str(annual_future_path),
    "reference_reproduction_status": reference_status,
    "max_base_reproduction_error": max_base_repro_error,
    "max_annual_reproduction_error": max_annual_repro_error,
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(pd.DataFrame(audit_rows).to_string(index=False))
print(f"\nAudit       : {AUDIT_PATH}")
print(f"Methodology : {METHOD_PATH}")
print(f"Summary     : {SUMMARY_PATH}")

if failed_audit:
    raise RuntimeError(f"Step 04 failed {len(failed_audit)} audit check(s).")

section("STEP 04 COMPLETED SUCCESSFULLY")
print(
    f"Validated future input projection generated for {future_years[0]}-{future_years[-1]}."
)
print(
    "NEXT: rebuild exact Step-07 future engineered features, run the selected model, "
    "and generate Y+1 to Y+5 grid LST forecasts."
)


if __name__ == "__main__":
    pass
