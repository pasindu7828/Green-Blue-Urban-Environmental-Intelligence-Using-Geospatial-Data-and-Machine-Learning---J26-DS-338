"""
STEP 34 - DASHBOARD DATA EXPORT
================================

Purpose
-------
Create a clean, dashboard/backend-ready data layer from the frozen Urban Heat
research outputs without retraining models or changing any scientific result.

This step:
1. discovers the project root from this script location (no hard-coded drive),
2. verifies required upstream audits,
3. builds one stable grid -> area_name mapping,
4. prepares historical heat data (2015-2025),
5. prepares future heat + uncertainty + XAI + trajectory + priority data
   (2026-2030),
6. attaches the primary planning scenario outputs used by the dashboard,
7. prepares cooling-priority and area-comparison tables,
8. prepares WGS84 GeoJSON map layers,
9. writes a dashboard manifest, audit, methodology, report and hashes,
10. confirms upstream input files were not modified.

Important methodological rules
------------------------------
- NO model retraining.
- NO hyperparameter tuning.
- NO new feature/indicator selection.
- NO CRITIC recalculation.
- NO TOPSIS recalculation.
- NO arbitrary High/Medium/Low priority thresholds are invented.
- area_name and geometry are display / filtering metadata, not new model inputs.
- Step-29 scenario outputs remain model-based planning sensitivities, not causal
  guarantees.
- Step-30 intervals are reference prediction/outcome bands and are not claimed
  as calibrated causal intervals for scenario delta-LST.

Expected project placement
--------------------------
Urban_Heat_Research/
└── src/
    └── 34_dashboard_data_export.py

Run from project root:
    python src/34_dashboard_data_export.py
"""

from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError as exc:
    raise ImportError(
        "geopandas is required for Step 34. Install it with: pip install geopandas"
    ) from exc


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
AREA_COL = "area_name"
TARGET_COL = "LST_C"

EXPECTED_GRIDS = 2194
HIST_YEARS = list(range(2015, 2026))
FUTURE_YEARS = list(range(2026, 2031))
EXPECTED_HIST_ROWS = EXPECTED_GRIDS * len(HIST_YEARS)      # 24,134
EXPECTED_FUTURE_ROWS = EXPECTED_GRIDS * len(FUTURE_YEARS)  # 10,970

# Dashboard primary scenario choices are inherited from the frozen scenario
# designs. They are not new optimization decisions in Step 34.
GREEN_PRIMARY_PP = 5
BUILT_PRIMARY_LEVEL = "MODERATE"
COMBINED_PRIMARY_GREEN_PP = 10
COMBINED_PRIMARY_BUILT_LEVEL = "MODERATE"
BLUE_PRIMARY_LEVEL = "MODERATE"

ROOT = Path(__file__).resolve().parents[1]

# Core historical / spatial metadata
BASE_PATH = ROOT / "data" / "processed" / "kaduwela_preprocessed_base.csv"
GRID_WGS84_PATH = ROOT / "data" / "processed" / "kaduwela_grid_geometry_wgs84.geojson"

# Trajectory
TRAJECTORY_PATH = ROOT / "outputs" / "metrics" / "23b_grid_heat_trajectory_classes.csv"
TRAJECTORY_AUDIT = ROOT / "outputs" / "reports" / "23b_heat_trajectory_audit.csv"

# Forecast + future explainability
FUTURE_PRED_PATH = ROOT / "outputs" / "forecasting" / "28_future_grid_lst_predictions_2026_2030.csv"
STEP28_AUDIT = ROOT / "outputs" / "reports" / "28_future_prediction_audit.csv"
FUTURE_XAI_PATH = ROOT / "outputs" / "explainability" / "28b_future_local_explanation_summary_2026_2030.csv"
STEP28B_AUDIT = ROOT / "outputs" / "reports" / "28b_future_shap_audit.csv"

# Forecast/scenario uncertainty
FUTURE_UNC_PATH = ROOT / "outputs" / "uncertainty" / "30_future_lst_with_uncertainty_2026_2030.csv"
GREEN_UNC_PATH = ROOT / "outputs" / "uncertainty" / "30_29a_green_scenarios_with_uncertainty.csv"
BUILT_UNC_PATH = ROOT / "outputs" / "uncertainty" / "30_29b_built_scenarios_with_uncertainty.csv"
COMBINED_UNC_PATH = ROOT / "outputs" / "uncertainty" / "30_29c_combined_scenarios_with_uncertainty.csv"
BLUE_UNC_PATH = ROOT / "outputs" / "uncertainty" / "30_29d_blue_scenarios_with_uncertainty.csv"
STEP30_AUDIT = ROOT / "outputs" / "reports" / "30_uncertainty_audit.csv"

# Priority intelligence
PRIORITY_PATH = ROOT / "outputs" / "priority" / "32_topsis_priority_ranking.csv"
STEP32_AUDIT = ROOT / "outputs" / "reports" / "32_topsis_priority_ranking_audit.csv"
STABILITY_PATH = ROOT / "outputs" / "priority" / "33_grid_rank_stability.csv"
STEP33_AUDIT = ROOT / "outputs" / "reports" / "33_priority_sensitivity_validation_audit.csv"

# Step 34 output layer
DASHBOARD_DIR = ROOT / "outputs" / "dashboard"
DASHBOARD_MAP_DIR = DASHBOARD_DIR / "maps"
DASHBOARD_SCENARIO_DIR = DASHBOARD_DIR / "scenarios"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [DASHBOARD_DIR, DASHBOARD_MAP_DIR, DASHBOARD_SCENARIO_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Main dashboard outputs
GRID_META_OUT = DASHBOARD_DIR / "34_grid_metadata.csv"
GRID_GEOJSON_OUT = DASHBOARD_MAP_DIR / "34_grid_metadata_wgs84.geojson"
HIST_OUT = DASHBOARD_DIR / "34_historical_heat_2015_2025.csv"
FUTURE_MASTER_OUT = DASHBOARD_DIR / "34_future_dashboard_master_2026_2030.csv"
FUTURE_GEOJSON_OUT = DASHBOARD_MAP_DIR / "34_future_dashboard_master_2026_2030_wgs84.geojson"
PRIORITY_OUT = DASHBOARD_DIR / "34_cooling_priority_dashboard.csv"
PRIORITY_GEOJSON_OUT = DASHBOARD_MAP_DIR / "34_cooling_priority_wgs84.geojson"
AREA_HEAT_OUT = DASHBOARD_DIR / "34_area_heat_summary_2015_2030.csv"
AREA_PRIORITY_OUT = DASHBOARD_DIR / "34_area_priority_summary.csv"
CURRENT_VS_FUTURE_OUT = DASHBOARD_DIR / "34_current_vs_forecast_2025_2030.csv"

# Scenario dashboard outputs
GREEN_DASH_OUT = DASHBOARD_SCENARIO_DIR / "34_green_scenarios_2026_2030.csv"
BUILT_DASH_OUT = DASHBOARD_SCENARIO_DIR / "34_built_scenarios_2026_2030.csv"
COMBINED_DASH_OUT = DASHBOARD_SCENARIO_DIR / "34_combined_scenarios_2026_2030.csv"
BLUE_DASH_OUT = DASHBOARD_SCENARIO_DIR / "34_blue_scenarios_2026_2030.csv"
PRIMARY_SCENARIOS_OUT = DASHBOARD_SCENARIO_DIR / "34_primary_scenarios_2026_2030.csv"

# QA / documentation
MANIFEST_OUT = DASHBOARD_DIR / "34_dashboard_manifest.json"
AUDIT_OUT = REPORT_DIR / "34_dashboard_data_export_audit.csv"
METHOD_OUT = REPORT_DIR / "34_dashboard_data_export_methodology.json"
REPORT_OUT = REPORT_DIR / "34_dashboard_data_export_report.txt"
HASH_OUT = REPORT_DIR / "34_dashboard_data_export_hashes.csv"


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 136)
    print(title)
    print("=" * 136)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def require_columns(df: pd.DataFrame, cols, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def require_audit_all_pass(path: Path, label: str) -> None:
    require_file(path, label)
    df = pd.read_csv(path, low_memory=False)
    if "status" not in df.columns:
        raise RuntimeError(f"{label} audit missing 'status' column: {path}")
    status = df["status"].astype(str).str.strip().str.upper()
    bad = df.loc[status != "PASS"]
    if len(bad) > 0:
        raise RuntimeError(
            f"{label} audit is not fully PASS. Non-PASS rows:\n{bad.to_string(index=False)}"
        )


def normalize_ids_years(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    require_columns(out, [GRID_COL], label)
    out[GRID_COL] = out[GRID_COL].astype(str).str.strip()
    if YEAR_COL in out.columns:
        out[YEAR_COL] = pd.to_numeric(out[YEAR_COL], errors="raise").astype(int)
    if AREA_COL in out.columns:
        out[AREA_COL] = out[AREA_COL].astype(str).str.strip()
    return out


def finite_check(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(values).all())


def safe_keep(df: pd.DataFrame, columns) -> pd.DataFrame:
    return df[[c for c in columns if c in df.columns]].copy()


def first_existing(df: pd.DataFrame, candidates, label: str):
    for col in candidates:
        if col in df.columns:
            return col
    raise RuntimeError(f"{label}: none of the expected columns exists: {candidates}")


def add_audit(audit_rows, check: str, status: bool, details: str) -> None:
    audit_rows.append({
        "check": check,
        "status": "PASS" if bool(status) else "FAIL",
        "details": details,
    })


def register_manifest(manifest_rows, path: Path, role: str, df=None, notes: str = "") -> None:
    item = {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "role": role,
        "notes": notes,
    }
    if df is not None:
        item["rows"] = int(len(df))
        item["columns"] = list(df.columns)
    manifest_rows.append(item)


def standardize_scenario_family(df: pd.DataFrame, family: str) -> pd.DataFrame:
    out = normalize_ids_years(df, family)
    require_columns(out, [GRID_COL, YEAR_COL, AREA_COL], family)
    out["scenario_family_dashboard"] = family
    return out


# =============================================================================
# 0. START + VERIFY REQUIRED INPUTS/AUDITS
# =============================================================================

section("STEP 34 - DASHBOARD DATA EXPORT")
print(f"Project root: {ROOT}")
print("No hard-coded drive path is used.")
print("No predictive model retraining or decision-model recalculation is performed.")

required_inputs = [
    (BASE_PATH, "historical processed base"),
    (GRID_WGS84_PATH, "WGS84 grid geometry"),
    (TRAJECTORY_PATH, "Step23B trajectory"),
    (FUTURE_PRED_PATH, "Step28 future predictions"),
    (FUTURE_XAI_PATH, "Step28B future local explanations"),
    (FUTURE_UNC_PATH, "Step30 future uncertainty"),
    (GREEN_UNC_PATH, "Step30 green scenario uncertainty"),
    (BUILT_UNC_PATH, "Step30 built scenario uncertainty"),
    (COMBINED_UNC_PATH, "Step30 combined scenario uncertainty"),
    (BLUE_UNC_PATH, "Step30 blue scenario uncertainty"),
    (PRIORITY_PATH, "Step32 TOPSIS ranking"),
    (STABILITY_PATH, "Step33 grid rank stability"),
]

for path, label in required_inputs:
    require_file(path, label)

for path, label in [
    (TRAJECTORY_AUDIT, "Step23B"),
    (STEP28_AUDIT, "Step28"),
    (STEP28B_AUDIT, "Step28B"),
    (STEP30_AUDIT, "Step30"),
    (STEP32_AUDIT, "Step32"),
    (STEP33_AUDIT, "Step33"),
]:
    require_audit_all_pass(path, label)
    print(f"PASS: {label} audit fully PASS")

# Hash all inputs before export so Step34 can prove read-only behavior.
input_paths = [p for p, _ in required_inputs]
input_hash_before = {str(p): sha256_file(p) for p in input_paths}


# =============================================================================
# 1. GRID -> AREA METADATA + GEOMETRY
# =============================================================================

section("1. BUILD STABLE GRID / AREA / GEOMETRY METADATA")

base = normalize_ids_years(
    pd.read_csv(BASE_PATH, dtype={GRID_COL: str, AREA_COL: str}, low_memory=False),
    "historical base",
)
require_columns(base, [GRID_COL, YEAR_COL, AREA_COL, TARGET_COL], "historical base")

# Verify that each grid belongs to exactly one area across the historical panel.
area_counts = base.groupby(GRID_COL)[AREA_COL].nunique(dropna=False)
if (area_counts != 1).any():
    bad_grids = area_counts[area_counts != 1]
    raise RuntimeError(
        "area_name is not stable one-to-one with grid_id. Problem grids:\n"
        + bad_grids.head(20).to_string()
    )

grid_meta = (
    base[[GRID_COL, AREA_COL]]
    .drop_duplicates()
    .sort_values(GRID_COL)
    .reset_index(drop=True)
)

if len(grid_meta) != EXPECTED_GRIDS or grid_meta[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Grid metadata must contain exactly {EXPECTED_GRIDS} unique grids; "
        f"rows={len(grid_meta)}, unique={grid_meta[GRID_COL].nunique()}"
    )

grid = gpd.read_file(GRID_WGS84_PATH)
require_columns(grid, [GRID_COL, "geometry"], "grid geometry")
grid[GRID_COL] = grid[GRID_COL].astype(str).str.strip()

if len(grid) != EXPECTED_GRIDS or grid[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Geometry must contain exactly {EXPECTED_GRIDS} unique grids; "
        f"rows={len(grid)}, unique={grid[GRID_COL].nunique()}"
    )

if grid.crs is None:
    raise RuntimeError("Grid geometry has no CRS.")

# Force API/map output to WGS84 if necessary.
if str(grid.crs).upper() not in {"EPSG:4326", "WGS84"}:
    grid = grid.to_crs(epsg=4326)

grid_geo = grid[[GRID_COL, "geometry"]].merge(
    grid_meta,
    on=GRID_COL,
    how="left",
    validate="one_to_one",
)
grid_geo = gpd.GeoDataFrame(grid_geo, geometry="geometry", crs=grid.crs)

if grid_geo[AREA_COL].isna().any():
    raise RuntimeError("Some geometry rows could not be joined to area_name.")

# Add light map metadata that is safe and useful to a backend/frontend.
# Use geographic bounding-box center only as a display coordinate, not for distance analysis.
bounds = grid_geo.geometry.bounds
coords = pd.DataFrame({
    GRID_COL: grid_geo[GRID_COL].astype(str).to_numpy(),
    "display_lon": (bounds["minx"].to_numpy() + bounds["maxx"].to_numpy()) / 2.0,
    "display_lat": (bounds["miny"].to_numpy() + bounds["maxy"].to_numpy()) / 2.0,
})
grid_meta_out = grid_meta.merge(coords, on=GRID_COL, how="left", validate="one_to_one")

grid_meta_out.to_csv(GRID_META_OUT, index=False)
grid_geo.to_file(GRID_GEOJSON_OUT, driver="GeoJSON")

print(f"Grid metadata rows : {len(grid_meta_out):,}")
print(f"Areas              : {grid_meta_out[AREA_COL].nunique():,}")
print(f"Map CRS            : {grid_geo.crs}")
print(f"Saved              : {GRID_META_OUT}")
print(f"Saved              : {GRID_GEOJSON_OUT}")


# =============================================================================
# 2. HISTORICAL DASHBOARD DATA 2015-2025
# =============================================================================

section("2. EXPORT HISTORICAL HEAT DATA 2015-2025")

hist = base[base[YEAR_COL].isin(HIST_YEARS)].copy()

if len(hist) != EXPECTED_HIST_ROWS:
    raise RuntimeError(
        f"Historical rows={len(hist):,}, expected={EXPECTED_HIST_ROWS:,}."
    )
if hist.duplicated([GRID_COL, YEAR_COL]).any():
    raise RuntimeError("Historical dashboard data has duplicate grid-year rows.")

hist_cols = [
    GRID_COL,
    AREA_COL,
    YEAR_COL,
    TARGET_COL,
    "NDVI",
    "NDBI",
    "NDWI",
    "EVI",
    "Albedo",
    "NightLights",
    "green_mask",
    "building_mask",
    "road_mask",
    "dist_road",
    "dist_main_road",
    "dist_water",
    "LCZ",
]
hist = safe_keep(hist, hist_cols)
hist = hist.rename(columns={TARGET_COL: "actual_lst_c"})
hist = hist.sort_values([YEAR_COL, GRID_COL]).reset_index(drop=True)
hist.to_csv(HIST_OUT, index=False)

print(f"Historical rows : {len(hist):,}")
print(f"Years           : {hist[YEAR_COL].min()}-{hist[YEAR_COL].max()}")
print(f"Saved           : {HIST_OUT}")


# =============================================================================
# 3. LOAD FUTURE FORECAST + UNCERTAINTY
# =============================================================================

section("3. LOAD FUTURE FORECAST + UNCERTAINTY")

future_unc = normalize_ids_years(
    pd.read_csv(FUTURE_UNC_PATH, dtype={GRID_COL: str, AREA_COL: str}, low_memory=False),
    "Step30 future uncertainty",
)
require_columns(
    future_unc,
    [GRID_COL, YEAR_COL, "predicted_lst_c"],
    "Step30 future uncertainty",
)

future_unc = future_unc[future_unc[YEAR_COL].isin(FUTURE_YEARS)].copy()
if len(future_unc) != EXPECTED_FUTURE_ROWS:
    raise RuntimeError(
        f"Future uncertainty rows={len(future_unc):,}, expected={EXPECTED_FUTURE_ROWS:,}."
    )
if future_unc.duplicated([GRID_COL, YEAR_COL]).any():
    raise RuntimeError("Future uncertainty has duplicate grid-year rows.")

# Independently verify Step30 retained Step28 point forecasts unchanged.
step28 = normalize_ids_years(
    pd.read_csv(FUTURE_PRED_PATH, dtype={GRID_COL: str, AREA_COL: str}, low_memory=False),
    "Step28 future predictions",
)
require_columns(step28, [GRID_COL, YEAR_COL, "predicted_lst_c"], "Step28 future predictions")

compare = future_unc[[GRID_COL, YEAR_COL, "predicted_lst_c"]].merge(
    step28[[GRID_COL, YEAR_COL, "predicted_lst_c"]].rename(
        columns={"predicted_lst_c": "step28_predicted_lst_c"}
    ),
    on=[GRID_COL, YEAR_COL],
    how="left",
    validate="one_to_one",
)
if compare["step28_predicted_lst_c"].isna().any():
    raise RuntimeError("Step30 future rows do not fully match Step28 future predictions.")
max_pred_diff = float(
    np.max(
        np.abs(
            compare["predicted_lst_c"].to_numpy(dtype=float)
            - compare["step28_predicted_lst_c"].to_numpy(dtype=float)
        )
    )
)
if max_pred_diff > 1e-10:
    raise RuntimeError(
        f"Step30 point forecasts do not reproduce Step28. max_diff={max_pred_diff:.3e}"
    )

# Canonical area_name always comes from stable historical grid metadata.
future_unc = future_unc.drop(columns=[AREA_COL], errors="ignore").merge(
    grid_meta,
    on=GRID_COL,
    how="left",
    validate="many_to_one",
)
if future_unc[AREA_COL].isna().any():
    raise RuntimeError("Some future rows are missing area_name after canonical metadata join.")

print(f"Future rows            : {len(future_unc):,}")
print(f"Future years           : {sorted(future_unc[YEAR_COL].unique().tolist())}")
print(f"Step28 reproduction max diff: {max_pred_diff:.3e}")


# =============================================================================
# 4. TRAJECTORY + PRIORITY + SENSITIVITY
# =============================================================================

section("4. JOIN TRAJECTORY + COOLING PRIORITY + RANK STABILITY")

trajectory = normalize_ids_years(
    pd.read_csv(TRAJECTORY_PATH, dtype={GRID_COL: str, AREA_COL: str}, low_memory=False),
    "Step23B trajectory",
)
require_columns(
    trajectory,
    [GRID_COL, "trajectory_class", "relative_sen_slope_percentile_per_year"],
    "Step23B trajectory",
)
if len(trajectory) != EXPECTED_GRIDS or trajectory[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError("Step23B trajectory must contain exactly one row per grid.")

trajectory_keep = safe_keep(
    trajectory,
    [
        GRID_COL,
        "trajectory_class",
        "baseline_relative_heat_median",
        "recent_relative_heat_median",
        "long_term_relative_heat_median",
        "relative_sen_slope_percentile_per_year",
        "relative_direction_operational",
        "sen_slope_c_per_year",
        "mk_p_value",
        "mk_fdr_q_value",
        "absolute_lst_sen_direction",
        "trajectory_scope",
        "trajectory_is_absolute_warming_claim",
        "trajectory_is_statistical_significance_claim",
        "trajectory_rule_version",
    ],
)

priority = normalize_ids_years(
    pd.read_csv(PRIORITY_PATH, dtype={GRID_COL: str}, low_memory=False),
    "Step32 priority ranking",
)
require_columns(
    priority,
    [GRID_COL, "topsis_rank", "topsis_priority_score", "priority_percentile"],
    "Step32 priority ranking",
)
if len(priority) != EXPECTED_GRIDS or priority[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError("Step32 ranking must contain exactly one row per grid.")

stability = normalize_ids_years(
    pd.read_csv(STABILITY_PATH, dtype={GRID_COL: str}, low_memory=False),
    "Step33 rank stability",
)
require_columns(stability, [GRID_COL, "baseline_rank", "mean_sensitivity_rank"], "Step33 rank stability")
if len(stability) != EXPECTED_GRIDS or stability[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError("Step33 stability must contain exactly one row per grid.")

priority_dashboard = (
    grid_meta
    .merge(priority, on=GRID_COL, how="left", validate="one_to_one")
    .merge(stability, on=GRID_COL, how="left", validate="one_to_one")
    .merge(trajectory_keep, on=GRID_COL, how="left", validate="one_to_one")
)

if priority_dashboard[["topsis_rank", "topsis_priority_score", "trajectory_class"]].isna().any().any():
    raise RuntimeError("Priority dashboard join produced missing core values.")

priority_dashboard = priority_dashboard.sort_values("topsis_rank").reset_index(drop=True)
priority_dashboard.to_csv(PRIORITY_OUT, index=False)

priority_geo = grid_geo[[GRID_COL, AREA_COL, "geometry"]].merge(
    priority_dashboard.drop(columns=[AREA_COL]),
    on=GRID_COL,
    how="left",
    validate="one_to_one",
)
priority_geo = gpd.GeoDataFrame(priority_geo, geometry="geometry", crs=grid_geo.crs)
priority_geo.to_file(PRIORITY_GEOJSON_OUT, driver="GeoJSON")

print(f"Priority rows       : {len(priority_dashboard):,}")
print(f"Rank range          : {priority_dashboard['topsis_rank'].min()}-{priority_dashboard['topsis_rank'].max()}")
print(f"Saved               : {PRIORITY_OUT}")
print(f"Saved               : {PRIORITY_GEOJSON_OUT}")


# =============================================================================
# 5. FUTURE XAI SUMMARY
# =============================================================================

section("5. JOIN FUTURE XAI SUMMARY")

xai = normalize_ids_years(
    pd.read_csv(FUTURE_XAI_PATH, dtype={GRID_COL: str}, low_memory=False),
    "Step28B future local explanation summary",
)
require_columns(
    xai,
    [GRID_COL, YEAR_COL, "top1_heating_feature", "top1_cooling_feature"],
    "Step28B future local explanation summary",
)
if len(xai) != EXPECTED_FUTURE_ROWS:
    raise RuntimeError(
        f"Future XAI summary rows={len(xai):,}, expected={EXPECTED_FUTURE_ROWS:,}."
    )
if xai.duplicated([GRID_COL, YEAR_COL]).any():
    raise RuntimeError("Future XAI summary has duplicate grid-year rows.")

xai_keep = safe_keep(
    xai,
    [
        GRID_COL,
        YEAR_COL,
        "top1_heating_feature",
        "top1_heating_shap_c",
        "top2_heating_feature",
        "top2_heating_shap_c",
        "top3_heating_feature",
        "top3_heating_shap_c",
        "top1_cooling_feature",
        "top1_cooling_shap_c",
        "top2_cooling_feature",
        "top2_cooling_shap_c",
        "top3_cooling_feature",
        "top3_cooling_shap_c",
    ],
)

print(f"XAI grid-year rows: {len(xai_keep):,}")


# =============================================================================
# 6. PRIMARY SCENARIOS + CLEAN SCENARIO EXPORTS
# =============================================================================

section("6. PREPARE SCENARIO DASHBOARD EXPORTS")

green = standardize_scenario_family(
    pd.read_csv(GREEN_UNC_PATH, dtype={GRID_COL: str, AREA_COL: str}, low_memory=False),
    "GREEN",
)
built = standardize_scenario_family(
    pd.read_csv(BUILT_UNC_PATH, dtype={GRID_COL: str, AREA_COL: str}, low_memory=False),
    "BUILT",
)
combined = standardize_scenario_family(
    pd.read_csv(COMBINED_UNC_PATH, dtype={GRID_COL: str, AREA_COL: str}, low_memory=False),
    "COMBINED",
)
blue = standardize_scenario_family(
    pd.read_csv(BLUE_UNC_PATH, dtype={GRID_COL: str, AREA_COL: str}, low_memory=False),
    "BLUE",
)

# Canonicalize area_name from grid metadata in all scenario exports.
def canonicalize_area(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.drop(columns=[AREA_COL], errors="ignore")
        .merge(grid_meta, on=GRID_COL, how="left", validate="many_to_one")
    )

green = canonicalize_area(green)
built = canonicalize_area(built)
combined = canonicalize_area(combined)
blue = canonicalize_area(blue)

for name, df in [("green", green), ("built", built), ("combined", combined), ("blue", blue)]:
    if df[AREA_COL].isna().any():
        raise RuntimeError(f"{name}: missing area_name after canonical metadata join.")
    if not set(df[YEAR_COL].unique()).issubset(set(FUTURE_YEARS)):
        raise RuntimeError(f"{name}: contains unexpected years.")

# Preserve all columns in these clean dashboard scenario tables. This allows the
# backend to expose full scenario detail later without returning to research folders.
green.to_csv(GREEN_DASH_OUT, index=False)
built.to_csv(BUILT_DASH_OUT, index=False)
combined.to_csv(COMBINED_DASH_OUT, index=False)
blue.to_csv(BLUE_DASH_OUT, index=False)

# Select the frozen primary scenario from each family.
require_columns(green, ["requested_green_increase_pp"], "green scenarios")
green_primary = green.loc[
    pd.to_numeric(green["requested_green_increase_pp"], errors="coerce") == GREEN_PRIMARY_PP
].copy()

require_columns(built, ["scenario_level"], "built scenarios")
built_primary = built.loc[
    built["scenario_level"].astype(str).str.upper() == BUILT_PRIMARY_LEVEL
].copy()

require_columns(
    combined,
    ["green_increase_pp_requested", "built_up_level"],
    "combined scenarios",
)
combined_primary = combined.loc[
    (pd.to_numeric(combined["green_increase_pp_requested"], errors="coerce") == COMBINED_PRIMARY_GREEN_PP)
    & (combined["built_up_level"].astype(str).str.upper() == COMBINED_PRIMARY_BUILT_LEVEL)
].copy()

require_columns(blue, ["scenario_level"], "blue scenarios")
blue_primary = blue.loc[
    blue["scenario_level"].astype(str).str.upper() == BLUE_PRIMARY_LEVEL
].copy()

for name, df in [
    ("green primary", green_primary),
    ("built primary", built_primary),
    ("combined primary", combined_primary),
    ("blue primary", blue_primary),
]:
    if len(df) != EXPECTED_FUTURE_ROWS:
        raise RuntimeError(
            f"{name} must have exactly {EXPECTED_FUTURE_ROWS:,} grid-year rows; got {len(df):,}."
        )
    if df.duplicated([GRID_COL, YEAR_COL]).any():
        raise RuntimeError(f"{name} contains duplicate grid-year rows.")

# Extract standardized scenario point estimate / benefit / uncertainty fields.
def scenario_primary_view(df, prefix, point_candidates, benefit_candidates):
    point_col = first_existing(df, point_candidates, f"{prefix} point estimate")
    benefit_col = first_existing(df, benefit_candidates, f"{prefix} cooling benefit")

    view = df[[GRID_COL, YEAR_COL]].copy()
    view[f"{prefix}_scenario_lst_c"] = pd.to_numeric(df[point_col], errors="raise")
    view[f"{prefix}_cooling_benefit_c"] = pd.to_numeric(df[benefit_col], errors="raise")

    # Step30 uses the common scenario_lst_* names regardless of family.
    for src, dst in [
        ("scenario_lst_reference_pi95_lower_c", f"{prefix}_reference_pi95_lower_c"),
        ("scenario_lst_reference_pi95_upper_c", f"{prefix}_reference_pi95_upper_c"),
        ("scenario_lst_reference_pi95_width_c", f"{prefix}_reference_pi95_width_c"),
        ("relative_confidence_label", f"{prefix}_relative_confidence_label"),
    ]:
        if src in df.columns:
            view[dst] = df[src].to_numpy()

    return view

green_view = scenario_primary_view(
    green_primary,
    "green",
    ["scenario_predicted_lst_c", "independent_scenario_lst_c", "scenario_lst_c"],
    ["cooling_benefit_c"],
)
built_view = scenario_primary_view(
    built_primary,
    "built",
    ["scenario_predicted_lst_c", "scenario_lst_c"],
    ["cooling_benefit_c"],
)
combined_view = scenario_primary_view(
    combined_primary,
    "combined",
    ["combined_scenario_predicted_lst_c", "scenario_predicted_lst_c", "scenario_lst_c"],
    ["combined_cooling_benefit_c", "cooling_benefit_c"],
)
blue_view = scenario_primary_view(
    blue_primary,
    "blue",
    ["scenario_predicted_lst_c", "scenario_lst_c"],
    ["cooling_benefit_c"],
)

primary_scenarios = (
    green_view
    .merge(built_view, on=[GRID_COL, YEAR_COL], how="inner", validate="one_to_one")
    .merge(combined_view, on=[GRID_COL, YEAR_COL], how="inner", validate="one_to_one")
    .merge(blue_view, on=[GRID_COL, YEAR_COL], how="inner", validate="one_to_one")
    .merge(grid_meta, on=GRID_COL, how="left", validate="many_to_one")
)
primary_scenarios.to_csv(PRIMARY_SCENARIOS_OUT, index=False)

print(f"Green scenarios rows    : {len(green):,}")
print(f"Built scenarios rows    : {len(built):,}")
print(f"Combined scenarios rows : {len(combined):,}")
print(f"Blue scenarios rows     : {len(blue):,}")
print(f"Primary scenario rows   : {len(primary_scenarios):,}")


# =============================================================================
# 7. BUILD FUTURE DASHBOARD MASTER 2026-2030
# =============================================================================

section("7. BUILD FUTURE DASHBOARD MASTER 2026-2030")

# Historical 2025 actual per grid for direct current-vs-forecast comparison.
actual_2025 = hist.loc[hist[YEAR_COL] == 2025, [GRID_COL, "actual_lst_c"]].copy()
if len(actual_2025) != EXPECTED_GRIDS:
    raise RuntimeError("2025 actual table must contain exactly one row per grid.")
actual_2025 = actual_2025.rename(columns={"actual_lst_c": "actual_2025_lst_c"})

# Priority/stability fields are grid-level, so replicate safely across future years.
priority_future_keep = safe_keep(
    priority_dashboard,
    [
        GRID_COL,
        "topsis_rank",
        "topsis_priority_score",
        "priority_percentile",
        "mean_sensitivity_rank",
        "median_sensitivity_rank",
        "best_sensitivity_rank",
        "worst_sensitivity_rank",
        "mean_abs_rank_shift",
        "median_abs_rank_shift",
        "p95_abs_rank_shift",
        "max_abs_rank_shift",
        "pct_scenarios_in_top10",
        "pct_scenarios_in_top50",
        "pct_scenarios_in_top100",
        "pct_scenarios_in_top10pct",
        "pct_scenarios_in_top25pct",
    ],
)

future_master = (
    future_unc
    .merge(actual_2025, on=GRID_COL, how="left", validate="many_to_one")
    .merge(trajectory_keep, on=GRID_COL, how="left", validate="many_to_one")
    .merge(priority_future_keep, on=GRID_COL, how="left", validate="many_to_one")
    .merge(xai_keep, on=[GRID_COL, YEAR_COL], how="left", validate="one_to_one")
    .merge(
        primary_scenarios.drop(columns=[AREA_COL], errors="ignore"),
        on=[GRID_COL, YEAR_COL],
        how="left",
        validate="one_to_one",
    )
)

future_master["change_from_2025_grid_actual_c"] = (
    pd.to_numeric(future_master["predicted_lst_c"], errors="raise")
    - pd.to_numeric(future_master["actual_2025_lst_c"], errors="raise")
)

# Friendly aliases for the main forecast interval; keep source fields too.
if "forecast_reference_pi95_lower_c" in future_master.columns:
    future_master["forecast_pi95_lower_c"] = future_master["forecast_reference_pi95_lower_c"]
if "forecast_reference_pi95_upper_c" in future_master.columns:
    future_master["forecast_pi95_upper_c"] = future_master["forecast_reference_pi95_upper_c"]
if "forecast_reference_pi95_width_c" in future_master.columns:
    future_master["forecast_pi95_width_c"] = future_master["forecast_reference_pi95_width_c"]

if len(future_master) != EXPECTED_FUTURE_ROWS:
    raise RuntimeError(
        f"Future dashboard master rows={len(future_master):,}, expected={EXPECTED_FUTURE_ROWS:,}."
    )
if future_master.duplicated([GRID_COL, YEAR_COL]).any():
    raise RuntimeError("Future dashboard master has duplicate grid-year rows.")

required_master_fields = [
    GRID_COL,
    AREA_COL,
    YEAR_COL,
    "predicted_lst_c",
    "trajectory_class",
    "topsis_rank",
    "topsis_priority_score",
    "top1_heating_feature",
    "top1_cooling_feature",
    "combined_scenario_lst_c",
    "combined_cooling_benefit_c",
]
require_columns(future_master, required_master_fields, "future dashboard master")
if future_master[required_master_fields].isna().any().any():
    missing_counts = future_master[required_master_fields].isna().sum()
    raise RuntimeError(
        "Future dashboard master has missing required values:\n"
        + missing_counts[missing_counts > 0].to_string()
    )

future_master = future_master.sort_values([YEAR_COL, "topsis_rank", GRID_COL]).reset_index(drop=True)
future_master.to_csv(FUTURE_MASTER_OUT, index=False)

# Map-ready future layer. Repeating geometry across 5 years is intentional so a
# simple API/frontend can filter by year and draw a complete heat map directly.
future_geo = grid_geo[[GRID_COL, "geometry"]].merge(
    future_master,
    on=GRID_COL,
    how="right",
    validate="one_to_many",
)
future_geo = gpd.GeoDataFrame(future_geo, geometry="geometry", crs=grid_geo.crs)
future_geo.to_file(FUTURE_GEOJSON_OUT, driver="GeoJSON")

print(f"Future master rows : {len(future_master):,}")
print(f"Future map features: {len(future_geo):,}")
print(f"Saved              : {FUTURE_MASTER_OUT}")
print(f"Saved              : {FUTURE_GEOJSON_OUT}")


# =============================================================================
# 8. CURRENT (2025) VS FORECAST (2026-2030)
# =============================================================================

section("8. BUILD CURRENT VS FORECAST TABLE")

current_2025 = hist.loc[hist[YEAR_COL] == 2025, [GRID_COL, AREA_COL, "actual_lst_c"]].copy()
current_2025 = current_2025.rename(columns={"actual_lst_c": "actual_2025_lst_c"})

future_compare = future_master[
    [
        GRID_COL,
        YEAR_COL,
        "predicted_lst_c",
        "change_from_2025_grid_actual_c",
    ]
].copy()

current_vs_future = current_2025.merge(
    future_compare,
    on=GRID_COL,
    how="left",
    validate="one_to_many",
)
current_vs_future.to_csv(CURRENT_VS_FUTURE_OUT, index=False)

print(f"Current-vs-future rows: {len(current_vs_future):,}")
print(f"Saved                 : {CURRENT_VS_FUTURE_OUT}")


# =============================================================================
# 9. AREA-LEVEL SUMMARY FOR COMPARISON PAGE
# =============================================================================

section("9. BUILD AREA-LEVEL HEAT + PRIORITY SUMMARIES")

hist_area = (
    hist.groupby([AREA_COL, YEAR_COL], as_index=False)
    .agg(
        grid_count=(GRID_COL, "nunique"),
        mean_lst_c=("actual_lst_c", "mean"),
        median_lst_c=("actual_lst_c", "median"),
        min_lst_c=("actual_lst_c", "min"),
        max_lst_c=("actual_lst_c", "max"),
    )
)
hist_area["value_type"] = "ACTUAL"

future_agg_spec = {
    "grid_count": (GRID_COL, "nunique"),
    "mean_lst_c": ("predicted_lst_c", "mean"),
    "median_lst_c": ("predicted_lst_c", "median"),
    "min_lst_c": ("predicted_lst_c", "min"),
    "max_lst_c": ("predicted_lst_c", "max"),
}
if "forecast_pi95_lower_c" in future_master.columns:
    future_agg_spec["mean_forecast_pi95_lower_c"] = ("forecast_pi95_lower_c", "mean")
if "forecast_pi95_upper_c" in future_master.columns:
    future_agg_spec["mean_forecast_pi95_upper_c"] = ("forecast_pi95_upper_c", "mean")

future_area = future_master.groupby([AREA_COL, YEAR_COL], as_index=False).agg(**future_agg_spec)
future_area["value_type"] = "FORECAST"

area_heat = pd.concat([hist_area, future_area], ignore_index=True, sort=False)
area_heat = area_heat.sort_values([AREA_COL, YEAR_COL]).reset_index(drop=True)
area_heat.to_csv(AREA_HEAT_OUT, index=False)

priority_tmp = priority_dashboard.copy()
priority_tmp["is_top100"] = priority_tmp["topsis_rank"] <= 100
priority_tmp["is_top10pct"] = priority_tmp["priority_percentile"] >= 90.0

area_priority = (
    priority_tmp.groupby(AREA_COL, as_index=False)
    .agg(
        grid_count=(GRID_COL, "nunique"),
        mean_priority_score=("topsis_priority_score", "mean"),
        median_priority_score=("topsis_priority_score", "median"),
        best_priority_rank=("topsis_rank", "min"),
        median_priority_rank=("topsis_rank", "median"),
        top100_grid_count=("is_top100", "sum"),
        top10pct_grid_count=("is_top10pct", "sum"),
    )
)
area_priority["top100_grid_percent"] = 100.0 * area_priority["top100_grid_count"] / area_priority["grid_count"]
area_priority["top10pct_grid_percent"] = 100.0 * area_priority["top10pct_grid_count"] / area_priority["grid_count"]
area_priority = area_priority.sort_values(
    ["mean_priority_score", AREA_COL], ascending=[False, True]
).reset_index(drop=True)
area_priority.to_csv(AREA_PRIORITY_OUT, index=False)

print(f"Area heat rows      : {len(area_heat):,}")
print(f"Area priority rows  : {len(area_priority):,}")
print(f"Saved               : {AREA_HEAT_OUT}")
print(f"Saved               : {AREA_PRIORITY_OUT}")


# =============================================================================
# 10. DASHBOARD MANIFEST
# =============================================================================

section("10. WRITE DASHBOARD MANIFEST")

manifest_rows = []
register_manifest(manifest_rows, GRID_META_OUT, "grid metadata", grid_meta_out,
                  "Stable grid_id -> area_name mapping plus display coordinates.")
register_manifest(manifest_rows, GRID_GEOJSON_OUT, "base map geometry", notes="WGS84 geometry, one feature per grid.")
register_manifest(manifest_rows, HIST_OUT, "historical heat", hist,
                  "Historical actual LST and key environmental variables, 2015-2025.")
register_manifest(manifest_rows, FUTURE_MASTER_OUT, "future dashboard master", future_master,
                  "Forecast + uncertainty + trajectory + XAI + priority + primary scenarios, 2026-2030.")
register_manifest(manifest_rows, FUTURE_GEOJSON_OUT, "future heat map layer", notes="WGS84 grid-year GeoJSON, 2026-2030.")
register_manifest(manifest_rows, PRIORITY_OUT, "cooling priority", priority_dashboard,
                  "Step32 TOPSIS ranking plus Step33 stability and trajectory metadata; no classes invented.")
register_manifest(manifest_rows, PRIORITY_GEOJSON_OUT, "cooling priority map", notes="WGS84 one feature per grid.")
register_manifest(manifest_rows, AREA_HEAT_OUT, "area heat comparison", area_heat,
                  "Actual 2015-2025 + forecast 2026-2030 area summaries.")
register_manifest(manifest_rows, AREA_PRIORITY_OUT, "area priority comparison", area_priority,
                  "Area-level descriptive aggregation of grid priority scores/ranks.")
register_manifest(manifest_rows, CURRENT_VS_FUTURE_OUT, "current vs forecast", current_vs_future,
                  "Per-grid actual 2025 versus forecasts 2026-2030.")
register_manifest(manifest_rows, GREEN_DASH_OUT, "green scenario catalog", green,
                  "Full Step30-enriched green scenario family.")
register_manifest(manifest_rows, BUILT_DASH_OUT, "built-up scenario catalog", built,
                  "Full Step30-enriched built-up scenario family.")
register_manifest(manifest_rows, COMBINED_DASH_OUT, "combined scenario catalog", combined,
                  "Full Step30-enriched combined green+built-up scenario family.")
register_manifest(manifest_rows, BLUE_DASH_OUT, "blue-space scenario catalog", blue,
                  "Full Step30-enriched blue-space/NDWI scenario family.")
register_manifest(manifest_rows, PRIMARY_SCENARIOS_OUT, "primary scenario comparison", primary_scenarios,
                  "Compact primary scenario comparison for each grid-year.")

manifest = {
    "step": "34_dashboard_data_export",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "project_root_resolution": "Path(__file__).resolve().parents[1]",
    "project_root_at_run": str(ROOT),
    "grid_count": EXPECTED_GRIDS,
    "historical_years": HIST_YEARS,
    "future_years": FUTURE_YEARS,
    "priority_classes_frozen": False,
    "scenario_causality_claim": False,
    "frontend_note": (
        "Use grid_id as the stable join key. area_name is display/filter metadata. "
        "GeoJSON geometry is WGS84."
    ),
    "primary_scenario_choices": {
        "green_requested_increase_pp": GREEN_PRIMARY_PP,
        "built_level": BUILT_PRIMARY_LEVEL,
        "combined_green_increase_pp": COMBINED_PRIMARY_GREEN_PP,
        "combined_built_level": COMBINED_PRIMARY_BUILT_LEVEL,
        "blue_level": BLUE_PRIMARY_LEVEL,
    },
    "files": manifest_rows,
}

with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"Saved: {MANIFEST_OUT}")


# =============================================================================
# 11. STRICT STEP 34 AUDIT
# =============================================================================

section("11. STRICT STEP 34 AUDIT")

audit_rows = []

add_audit(audit_rows, "step23b_audit", True, "Step23B audit verified fully PASS.")
add_audit(audit_rows, "step28_audit", True, "Step28 audit verified fully PASS.")
add_audit(audit_rows, "step28b_audit", True, "Step28B audit verified fully PASS.")
add_audit(audit_rows, "step30_audit", True, "Step30 audit verified fully PASS.")
add_audit(audit_rows, "step32_audit", True, "Step32 audit verified fully PASS.")
add_audit(audit_rows, "step33_audit", True, "Step33 audit verified fully PASS.")

add_audit(
    audit_rows,
    "grid_count",
    len(grid_meta) == EXPECTED_GRIDS and grid_meta[GRID_COL].nunique() == EXPECTED_GRIDS,
    f"rows={len(grid_meta)}, unique_grids={grid_meta[GRID_COL].nunique()}, expected={EXPECTED_GRIDS}",
)
add_audit(
    audit_rows,
    "grid_area_one_to_one",
    bool((area_counts == 1).all()),
    "Each grid_id maps to exactly one historical area_name.",
)
add_audit(
    audit_rows,
    "geometry_complete",
    len(grid_geo) == EXPECTED_GRIDS and grid_geo.geometry.notna().all(),
    f"rows={len(grid_geo)}, CRS={grid_geo.crs}",
)
add_audit(
    audit_rows,
    "historical_rows_complete",
    len(hist) == EXPECTED_HIST_ROWS and not hist.duplicated([GRID_COL, YEAR_COL]).any(),
    f"rows={len(hist)}, expected={EXPECTED_HIST_ROWS}",
)
add_audit(
    audit_rows,
    "historical_years_exact",
    sorted(hist[YEAR_COL].unique().tolist()) == HIST_YEARS,
    f"years={sorted(hist[YEAR_COL].unique().tolist())}",
)
add_audit(
    audit_rows,
    "future_rows_complete",
    len(future_master) == EXPECTED_FUTURE_ROWS and not future_master.duplicated([GRID_COL, YEAR_COL]).any(),
    f"rows={len(future_master)}, expected={EXPECTED_FUTURE_ROWS}",
)
add_audit(
    audit_rows,
    "future_years_exact",
    sorted(future_master[YEAR_COL].unique().tolist()) == FUTURE_YEARS,
    f"years={sorted(future_master[YEAR_COL].unique().tolist())}",
)
add_audit(
    audit_rows,
    "step28_forecast_reproduced_in_step30",
    max_pred_diff <= 1e-10,
    f"max_abs_difference={max_pred_diff:.3e}",
)
add_audit(
    audit_rows,
    "trajectory_join_complete",
    future_master["trajectory_class"].notna().all(),
    "Trajectory class available for every future grid-year.",
)
add_audit(
    audit_rows,
    "xai_join_complete",
    future_master["top1_heating_feature"].notna().all()
    and future_master["top1_cooling_feature"].notna().all(),
    "Future local XAI summary available for every grid-year.",
)
add_audit(
    audit_rows,
    "priority_join_complete",
    future_master["topsis_rank"].notna().all()
    and future_master["topsis_priority_score"].notna().all(),
    "Step32 continuous priority rank/score available for every future grid-year.",
)
add_audit(
    audit_rows,
    "rank_stability_join_complete",
    future_master["mean_sensitivity_rank"].notna().all(),
    "Step33 rank-stability information available for every future grid-year.",
)
add_audit(
    audit_rows,
    "primary_scenario_join_complete",
    all(
        future_master[c].notna().all()
        for c in [
            "green_scenario_lst_c",
            "built_scenario_lst_c",
            "combined_scenario_lst_c",
            "blue_scenario_lst_c",
        ]
    ),
    "All four primary scenario families available for every future grid-year.",
)
add_audit(
    audit_rows,
    "area_name_complete",
    future_master[AREA_COL].notna().all()
    and priority_dashboard[AREA_COL].notna().all(),
    "Canonical area_name joined to final dashboard datasets.",
)
add_audit(
    audit_rows,
    "priority_scores_finite",
    finite_check(priority_dashboard["topsis_priority_score"]),
    "All TOPSIS priority scores are finite.",
)
add_audit(
    audit_rows,
    "future_predictions_finite",
    finite_check(future_master["predicted_lst_c"]),
    "All future LST point forecasts are finite.",
)
add_audit(
    audit_rows,
    "no_priority_class_thresholds_invented",
    True,
    "Step34 exports continuous Step32 rank/score; no new High/Medium/Low thresholds are created.",
)
add_audit(
    audit_rows,
    "no_model_retraining",
    True,
    "Step34 contains no estimator fitting or model retraining.",
)
add_audit(
    audit_rows,
    "no_critic_or_topsis_recalculation",
    True,
    "Step34 only reads Step32/33 priority outputs; CRITIC/TOPSIS are not recalculated.",
)
add_audit(
    audit_rows,
    "scenario_claim_scope_preserved",
    True,
    "Scenario fields are exported as model-based planning sensitivities, not causal guarantees.",
)

# Re-hash all upstream inputs after export.
input_hash_after = {str(p): sha256_file(p) for p in input_paths}
upstream_unchanged = input_hash_before == input_hash_after
add_audit(
    audit_rows,
    "upstream_files_unchanged",
    upstream_unchanged,
    "All Step34 input hashes are unchanged before vs after export.",
)

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(AUDIT_OUT, index=False)
print(audit_df.to_string(index=False))

if not (audit_df["status"] == "PASS").all():
    raise RuntimeError("Step34 strict audit contains FAIL rows. Review audit before continuing.")


# =============================================================================
# 12. METHODOLOGY + REPORT + HASHES
# =============================================================================

section("12. SAVE METHODOLOGY, REPORT AND HASHES")

methodology = {
    "step": "34_dashboard_data_export",
    "purpose": "Convert frozen research outputs into a clean dashboard/backend data contract.",
    "path_strategy": "All paths are relative to PROJECT_ROOT = Path(__file__).resolve().parents[1].",
    "join_key": GRID_COL,
    "area_name_role": "display/filter/grouping metadata only; not introduced as a model or TOPSIS criterion",
    "geometry_role": "WGS84 map geometry only",
    "historical_export": {
        "years": HIST_YEARS,
        "rows_expected": EXPECTED_HIST_ROWS,
    },
    "future_export": {
        "years": FUTURE_YEARS,
        "rows_expected": EXPECTED_FUTURE_ROWS,
        "contains": [
            "Step28 point forecast",
            "Step30 forecast reference uncertainty",
            "Step23B trajectory",
            "Step28B local SHAP summary",
            "Step32 TOPSIS rank/score",
            "Step33 sensitivity/rank stability",
            "Step30-enriched primary scenario summaries",
        ],
    },
    "primary_scenarios": {
        "green": f"+{GREEN_PRIMARY_PP} percentage-point persistent green scenario",
        "built": BUILT_PRIMARY_LEVEL,
        "combined": f"green +{COMBINED_PRIMARY_GREEN_PP}pp + built {COMBINED_PRIMARY_BUILT_LEVEL}",
        "blue": BLUE_PRIMARY_LEVEL,
    },
    "priority_policy": {
        "source": "Step32 TOPSIS + Step33 sensitivity",
        "new_thresholds_created": False,
        "new_priority_classes_created": False,
    },
    "scientific_guardrails": [
        "No model retraining.",
        "No feature or indicator reselection.",
        "No CRITIC or TOPSIS recalculation.",
        "No arbitrary priority classes invented.",
        "Scenario outputs remain model-based sensitivities, not causal guarantees.",
        "Step30 scenario uncertainty is a reference outcome band; no calibrated interval is claimed for intervention delta-LST.",
    ],
}
with open(METHOD_OUT, "w", encoding="utf-8") as f:
    json.dump(methodology, f, indent=2, ensure_ascii=False)

report_lines = [
    "STEP 34 - DASHBOARD DATA EXPORT",
    "",
    f"Project root: {ROOT}",
    f"Grid count: {EXPECTED_GRIDS}",
    f"Area count: {grid_meta[AREA_COL].nunique()}",
    f"Historical rows: {len(hist)} ({HIST_YEARS[0]}-{HIST_YEARS[-1]})",
    f"Future master rows: {len(future_master)} ({FUTURE_YEARS[0]}-{FUTURE_YEARS[-1]})",
    f"Priority rows: {len(priority_dashboard)}",
    f"Future map features: {len(future_geo)}",
    "",
    "PRIMARY DASHBOARD SOURCES",
    "- Historical: kaduwela_preprocessed_base.csv",
    "- Forecast: Step28 point forecasts carried through Step30 uncertainty",
    "- Trajectory: Step23B relative spatial heat trajectory",
    "- XAI: Step28B future local SHAP summary",
    "- Scenarios: Step30-enriched Step29 scenario outputs",
    "- Priority: Step32 TOPSIS + Step33 sensitivity/rank stability",
    "",
    "GUARDRAILS",
    "- No model retraining performed.",
    "- No CRITIC/TOPSIS recalculation performed.",
    "- No priority classes or new thresholds invented.",
    "- area_name and geometry are metadata for dashboard use.",
    "- Scenario outputs remain model-based planning sensitivity, not causal guarantees.",
    "",
    "MAIN FRONTEND/BACKEND FILES",
    f"- {FUTURE_MASTER_OUT.relative_to(ROOT)}",
    f"- {HIST_OUT.relative_to(ROOT)}",
    f"- {PRIORITY_OUT.relative_to(ROOT)}",
    f"- {GRID_GEOJSON_OUT.relative_to(ROOT)}",
    f"- {FUTURE_GEOJSON_OUT.relative_to(ROOT)}",
    f"- {AREA_HEAT_OUT.relative_to(ROOT)}",
    f"- {PRIMARY_SCENARIOS_OUT.relative_to(ROOT)}",
]
REPORT_OUT.write_text("\n".join(report_lines), encoding="utf-8")

# Hash all created Step34 files (except hash file itself), plus record upstream input hashes.
created_files = [
    GRID_META_OUT,
    GRID_GEOJSON_OUT,
    HIST_OUT,
    FUTURE_MASTER_OUT,
    FUTURE_GEOJSON_OUT,
    PRIORITY_OUT,
    PRIORITY_GEOJSON_OUT,
    AREA_HEAT_OUT,
    AREA_PRIORITY_OUT,
    CURRENT_VS_FUTURE_OUT,
    GREEN_DASH_OUT,
    BUILT_DASH_OUT,
    COMBINED_DASH_OUT,
    BLUE_DASH_OUT,
    PRIMARY_SCENARIOS_OUT,
    MANIFEST_OUT,
    AUDIT_OUT,
    METHOD_OUT,
    REPORT_OUT,
]

hash_rows = []
for path in input_paths:
    hash_rows.append({
        "file_role": "upstream_input",
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_file(path),
    })
for path in created_files:
    hash_rows.append({
        "file_role": "step34_output",
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_file(path),
    })
pd.DataFrame(hash_rows).to_csv(HASH_OUT, index=False)

print(f"Saved: {AUDIT_OUT}")
print(f"Saved: {METHOD_OUT}")
print(f"Saved: {REPORT_OUT}")
print(f"Saved: {HASH_OUT}")


# =============================================================================
# 13. FINAL SUMMARY
# =============================================================================

section("STEP 34 COMPLETED SUCCESSFULLY")

print(f"Dashboard root             : {DASHBOARD_DIR}")
print(f"Grid metadata rows         : {len(grid_meta_out):,}")
print(f"Historical grid-year rows  : {len(hist):,}")
print(f"Future dashboard rows      : {len(future_master):,}")
print(f"Cooling-priority rows      : {len(priority_dashboard):,}")
print(f"Primary scenario rows      : {len(primary_scenarios):,}")
print(f"Area heat summary rows     : {len(area_heat):,}")
print(f"Area priority rows         : {len(area_priority):,}")
print("All strict Step34 audit checks: PASS")
print("No upstream research file was modified.")
print("No model retraining performed.")
print("No CRITIC/TOPSIS recalculation performed.")
print("No arbitrary priority class thresholds created.")
print("NEXT: FastAPI/backend integration can use outputs/dashboard as the stable data layer.")
