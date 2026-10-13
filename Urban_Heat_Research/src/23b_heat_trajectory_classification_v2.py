"""
STEP 23B v2 - ROBUST FIVE-CLASS RELATIVE HEAT TRAJECTORY CLASSIFICATION
Urban Heat Research - Kaduwela, Sri Lanka
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import theilslopes

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET = "LST_C"
AREA_COL = "area_name"

EXPECTED_ROWS = 24134
EXPECTED_GRIDS = 2194
EXPECTED_YEARS = list(range(2015, 2026))

BASELINE_YEARS = [2015, 2016, 2017, 2018, 2019]
RECENT_YEARS = [2021, 2022, 2023, 2024, 2025]

HOT_THRESHOLD = 0.75
STABLE_SPLIT = 0.50
RELATIVE_SLOPE_THRESHOLD = 0.01

CLASS_ORDER = [
    "Persistent Cool",
    "Diminishing",
    "Emerging",
    "Intensifying",
    "Persistent Hot",
]

EXPECTED_BASE_HASH = "0e030b2ec0b5c4080cd4226bec798189a178aa45efd9962d19f38dc9cb170f46"
EXPECTED_STEP23_GRID_HASH = "d972bacb111b2bce3ef40a739d3980e61ed2263682adfdc9c2ebf8678a800eb8"
EXPECTED_STEP23_SENSITIVITY_HASH = "fe38083cfcf11f8b4ef19c6c4b58184ec4a94c10969b30bed69394da033f0618"
EXPECTED_STEP23_AUDIT_HASH = "5b6c983793d2b1922d0e59c710bc6ccb69c1ef5c12f217b9b7749fb0d2eea02c"
EXPECTED_GRID_WGS84_HASH = "6b758ddaae19b54219404b86740472038cfd21391cf349ea766124678050f608"

ROOT = Path(__file__).resolve().parents[1]

BASE_DATA_PATH = ROOT / "data" / "processed" / "kaduwela_preprocessed_base.csv"
STEP23_GRID_PATH = ROOT / "outputs" / "metrics" / "23_grid_mann_kendall_sens_slope.csv"
STEP23_SENSITIVITY_PATH = ROOT / "outputs" / "metrics" / "23_2017_sensitivity_comparison.csv"
STEP23_AUDIT_PATH = ROOT / "outputs" / "reports" / "23_mann_kendall_sens_slope_audit.csv"
GRID_WGS84_PATH = ROOT / "data" / "processed" / "kaduwela_grid_geometry_wgs84.geojson"

METRICS_DIR = ROOT / "outputs" / "metrics"
REPORTS_DIR = ROOT / "outputs" / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"
MAPS_DIR = ROOT / "outputs" / "maps"

for folder in [METRICS_DIR, REPORTS_DIR, FIGURES_DIR, MAPS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TRAJECTORY_PATH = METRICS_DIR / "23b_grid_heat_trajectory_classes.csv"
CLASS_COUNTS_PATH = METRICS_DIR / "23b_trajectory_class_counts.csv"
YEARLY_PERCENTILES_PATH = METRICS_DIR / "23b_grid_year_relative_heat_percentiles.csv"
NO2017_COMPARISON_PATH = METRICS_DIR / "23b_no2017_trajectory_sensitivity.csv"
THRESHOLD_SENSITIVITY_PATH = METRICS_DIR / "23b_threshold_sensitivity_summary.csv"
MAP_PATH = MAPS_DIR / "23b_heat_trajectory_classes_wgs84.geojson"

CLASS_MAP_PATH = FIGURES_DIR / "23b_heat_trajectory_class_map.png"
CLASS_COUNT_FIG_PATH = FIGURES_DIR / "23b_heat_trajectory_class_counts.png"
RELATIVE_SLOPE_FIG_PATH = FIGURES_DIR / "23b_relative_heat_sen_slope_distribution.png"
BASELINE_RECENT_FIG_PATH = FIGURES_DIR / "23b_baseline_vs_recent_relative_heat.png"

METHOD_PATH = REPORTS_DIR / "23b_heat_trajectory_methodology.json"
AUDIT_PATH = REPORTS_DIR / "23b_heat_trajectory_audit.csv"
REPORT_PATH = REPORTS_DIR / "23b_heat_trajectory_report.txt"
HASH_PATH = REPORTS_DIR / "23b_heat_trajectory_hashes.csv"


def section(title):
    print("\n" + "=" * 128)
    print(title)
    print("=" * 128)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")


def verify_hash(path, expected, label):
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch.\nExpected: {expected}\nActual  : {actual}"
        )
    print(f"PASS hash: {label} = {actual}")
    return actual


def require_audit_all_pass(path, label):
    audit = pd.read_csv(path)
    if "status" not in audit.columns:
        raise RuntimeError(f"{label} audit has no status column")
    bad = audit[audit["status"].astype(str).str.upper() != "PASS"]
    if not bad.empty:
        print(bad.to_string(index=False))
        raise RuntimeError(f"{label} audit is not fully PASS")


def add_yearly_relative_heat_percentile(df):
    out = df.copy()
    out["relative_heat_percentile"] = (
        out.groupby(YEAR_COL)[TARGET]
        .rank(method="average", pct=True)
    )
    return out


def classify_trajectory(
    baseline_score,
    recent_score,
    long_score,
    relative_sen_slope,
    hot_threshold=HOT_THRESHOLD,
    stable_split=STABLE_SPLIT,
    slope_threshold=RELATIVE_SLOPE_THRESHOLD,
):
    if (
        relative_sen_slope >= slope_threshold
        and baseline_score < hot_threshold
        and recent_score >= hot_threshold
    ):
        return "Emerging"

    if relative_sen_slope >= slope_threshold:
        return "Intensifying"

    if relative_sen_slope <= -slope_threshold:
        return "Diminishing"

    if long_score >= stable_split:
        return "Persistent Hot"

    return "Persistent Cool"


def slope_direction(slope, threshold=RELATIVE_SLOPE_THRESHOLD):
    if slope >= threshold:
        return "Increasing"
    if slope <= -threshold:
        return "Decreasing"
    return "Stable"


def build_grid_trajectory_features(
    percentile_data,
    baseline_years,
    recent_years,
    slope_threshold=RELATIVE_SLOPE_THRESHOLD,
    hot_threshold=HOT_THRESHOLD,
):
    rows = []

    for grid_id, group in percentile_data.groupby(GRID_COL, sort=True):
        g = group.sort_values(YEAR_COL).copy()

        x = g[YEAR_COL].to_numpy(dtype=float)
        y = g["relative_heat_percentile"].to_numpy(dtype=float)

        if len(g) < 3:
            raise RuntimeError(f"Grid {grid_id} has too few years for Sen slope")

        sen = theilslopes(y, x, alpha=0.95)

        relative_slope = float(sen.slope)
        relative_intercept = float(sen.intercept)
        relative_ci_low = float(sen.low_slope)
        relative_ci_high = float(sen.high_slope)

        baseline = g[g[YEAR_COL].isin(baseline_years)]
        recent = g[g[YEAR_COL].isin(recent_years)]

        if baseline.empty or recent.empty:
            raise RuntimeError(f"Grid {grid_id} has missing baseline/recent years")

        baseline_score = float(baseline["relative_heat_percentile"].median())
        recent_score = float(recent["relative_heat_percentile"].median())
        long_score = float(g["relative_heat_percentile"].median())
        mean_score = float(g["relative_heat_percentile"].mean())

        area_values = g[AREA_COL].dropna().astype(str).unique().tolist()
        if len(area_values) != 1:
            raise RuntimeError(f"Grid {grid_id} has inconsistent area_name")

        trajectory_class = classify_trajectory(
            baseline_score=baseline_score,
            recent_score=recent_score,
            long_score=long_score,
            relative_sen_slope=relative_slope,
            hot_threshold=hot_threshold,
            stable_split=STABLE_SPLIT,
            slope_threshold=slope_threshold,
        )

        rows.append(
            {
                GRID_COL: str(grid_id),
                AREA_COL: area_values[0],
                "baseline_relative_heat_median": baseline_score,
                "recent_relative_heat_median": recent_score,
                "recent_minus_baseline_relative_heat": recent_score - baseline_score,
                "long_term_relative_heat_median": long_score,
                "long_term_relative_heat_mean": mean_score,
                "relative_sen_slope_percentile_per_year": relative_slope,
                "relative_sen_intercept": relative_intercept,
                "relative_sen_slope_ci95_low": relative_ci_low,
                "relative_sen_slope_ci95_high": relative_ci_high,
                "relative_direction_operational": slope_direction(
                    relative_slope, threshold=slope_threshold
                ),
                "relative_slope_ci_excludes_zero": bool(
                    (relative_ci_low > 0.0) or (relative_ci_high < 0.0)
                ),
                "trajectory_class": trajectory_class,
            }
        )

    return pd.DataFrame(rows)


section("STEP 23B v2 - ROBUST FIVE-CLASS RELATIVE HEAT TRAJECTORY CLASSIFICATION")

for path in [
    BASE_DATA_PATH,
    STEP23_GRID_PATH,
    STEP23_SENSITIVITY_PATH,
    STEP23_AUDIT_PATH,
    GRID_WGS84_PATH,
]:
    require_file(path)

require_audit_all_pass(STEP23_AUDIT_PATH, "Step 23")
print("PASS: Step 23 audit is fully PASS.")

verify_hash(BASE_DATA_PATH, EXPECTED_BASE_HASH, "Processed historical data")
verify_hash(STEP23_GRID_PATH, EXPECTED_STEP23_GRID_HASH, "Step 23 grid trend results")
verify_hash(
    STEP23_SENSITIVITY_PATH,
    EXPECTED_STEP23_SENSITIVITY_HASH,
    "Step 23 2017 sensitivity results",
)
verify_hash(STEP23_AUDIT_PATH, EXPECTED_STEP23_AUDIT_HASH, "Step 23 audit")
verify_hash(GRID_WGS84_PATH, EXPECTED_GRID_WGS84_HASH, "WGS84 grid geometry")

print("\nIMPORTANT methodological revision:")
print("- trajectory direction now uses Theil-Sen slope of yearly relative heat percentile")
print("- ALL years 2015-2025 contribute to trajectory direction")
print("- 2015-2019 and 2021-2025 are robust median status windows")
print("- Step-23 absolute LST MK/Sen results remain separate evidence")


section("1. LOAD + VALIDATE HISTORICAL DATA AND STEP-23 RESULTS")

data = pd.read_csv(BASE_DATA_PATH, dtype={GRID_COL: str})
trend = pd.read_csv(STEP23_GRID_PATH, dtype={GRID_COL: str})

required_data_cols = {GRID_COL, YEAR_COL, TARGET, AREA_COL}
missing_data = sorted(required_data_cols - set(data.columns))
if missing_data:
    raise RuntimeError(f"Historical data missing columns: {missing_data}")

required_trend_cols = {
    GRID_COL,
    "mk_p_value",
    "mk_fdr_q_value",
    "mk_tau",
    "sen_slope_c_per_year",
    "sen_slope_ci95_low",
    "sen_slope_ci95_high",
    "trend_raw_alpha_0_05",
    "trend_fdr_alpha_0_05",
}
missing_trend = sorted(required_trend_cols - set(trend.columns))
if missing_trend:
    raise RuntimeError(f"Step-23 trend file missing columns: {missing_trend}")

data[YEAR_COL] = pd.to_numeric(data[YEAR_COL], errors="coerce").astype(int)
data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")

if len(data) != EXPECTED_ROWS:
    raise RuntimeError(f"Rows={len(data)}, expected={EXPECTED_ROWS}")
if data[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError("Historical grid count mismatch")
if trend[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError("Step-23 trend result count mismatch")
if sorted(data[YEAR_COL].unique().tolist()) != EXPECTED_YEARS:
    raise RuntimeError("Historical year range mismatch")
if data.duplicated([GRID_COL, YEAR_COL]).any():
    raise RuntimeError("Duplicate grid-year rows found")

year_counts = data.groupby(GRID_COL)[YEAR_COL].nunique()
if year_counts.min() != 11 or year_counts.max() != 11:
    raise RuntimeError("Every grid must have exactly 11 historical years")

print(f"Historical rows : {len(data):,}")
print(f"Unique grids    : {data[GRID_COL].nunique():,}")
print(f"Years           : {EXPECTED_YEARS}")
print(f"Years/grid      : {year_counts.min()} to {year_counts.max()}")


section("2. CALCULATE YEARLY RELATIVE HEAT PERCENTILES")

percentile_data = add_yearly_relative_heat_percentile(
    data[[GRID_COL, AREA_COL, YEAR_COL, TARGET]].copy()
)

if not percentile_data["relative_heat_percentile"].between(0, 1).all():
    raise RuntimeError("Relative percentile outside [0,1]")

year_audit = (
    percentile_data.groupby(YEAR_COL, as_index=False)
    .agg(
        grid_count=(GRID_COL, "size"),
        min_percentile=("relative_heat_percentile", "min"),
        median_percentile=("relative_heat_percentile", "median"),
        max_percentile=("relative_heat_percentile", "max"),
    )
)

print(year_audit.to_string(index=False))
percentile_data.to_csv(YEARLY_PERCENTILES_PATH, index=False)
print(f"\nSaved: {YEARLY_PERCENTILES_PATH}")


section("3. PRIMARY ALL-YEAR RELATIVE TRAJECTORY CLASSIFICATION")

features = build_grid_trajectory_features(
    percentile_data=percentile_data,
    baseline_years=BASELINE_YEARS,
    recent_years=RECENT_YEARS,
)

trajectory = features.merge(
    trend[
        [
            GRID_COL,
            "mk_p_value",
            "mk_fdr_q_value",
            "mk_tau",
            "sen_slope_c_per_year",
            "sen_slope_ci95_low",
            "sen_slope_ci95_high",
            "trend_raw_alpha_0_05",
            "trend_fdr_alpha_0_05",
        ]
    ],
    on=GRID_COL,
    how="left",
    validate="one_to_one",
)

if trajectory["mk_p_value"].isna().any():
    raise RuntimeError("Step-23 statistics failed to join for some grids")

trajectory["absolute_lst_sen_direction"] = np.select(
    [
        trajectory["sen_slope_c_per_year"] > 0,
        trajectory["sen_slope_c_per_year"] < 0,
    ],
    ["Increasing", "Decreasing"],
    default="Flat",
)

trajectory["relative_vs_absolute_direction_agree"] = (
    (
        (trajectory["relative_direction_operational"] == "Increasing")
        & (trajectory["absolute_lst_sen_direction"] == "Increasing")
    )
    |
    (
        (trajectory["relative_direction_operational"] == "Decreasing")
        & (trajectory["absolute_lst_sen_direction"] == "Decreasing")
    )
    |
    (trajectory["relative_direction_operational"] == "Stable")
)

trajectory["trajectory_scope"] = "relative spatial heat trajectory within Kaduwela"
trajectory["trajectory_is_absolute_warming_claim"] = False
trajectory["trajectory_is_statistical_significance_claim"] = False
trajectory["trajectory_rule_version"] = "23B_relative_sen_v2"

trajectory = trajectory.sort_values(GRID_COL).reset_index(drop=True)

if len(trajectory) != EXPECTED_GRIDS:
    raise RuntimeError("Trajectory output must have 2,194 grids")
if trajectory[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError("Duplicate/missing grid IDs in trajectory output")
if trajectory["trajectory_class"].isna().any():
    raise RuntimeError("At least one grid has no trajectory class")

unexpected_classes = sorted(
    set(trajectory["trajectory_class"].unique()) - set(CLASS_ORDER)
)
if unexpected_classes:
    raise RuntimeError(f"Unexpected trajectory classes: {unexpected_classes}")

trajectory.to_csv(TRAJECTORY_PATH, index=False)

class_counts = (
    trajectory["trajectory_class"]
    .value_counts()
    .reindex(CLASS_ORDER, fill_value=0)
    .rename_axis("trajectory_class")
    .reset_index(name="grid_count")
)
class_counts["percent_of_grids"] = (
    class_counts["grid_count"] / EXPECTED_GRIDS * 100.0
)
class_counts.to_csv(CLASS_COUNTS_PATH, index=False)

print("Reference rules:")
print(f"- hot threshold                  : {HOT_THRESHOLD:.2f}")
print(f"- stable warm/cool split         : {STABLE_SPLIT:.2f}")
print(f"- relative Sen slope threshold   : ±{RELATIVE_SLOPE_THRESHOLD:.3f} percentile/year")
print(f"- baseline heat-status window    : {BASELINE_YEARS}")
print(f"- recent heat-status window      : {RECENT_YEARS}")

print("\nTrajectory class distribution:")
print(class_counts.to_string(index=False))
print(f"\nSaved: {TRAJECTORY_PATH}")
print(f"Saved: {CLASS_COUNTS_PATH}")


section("4. RELATIVE-TRAJECTORY + ABSOLUTE-LST EVIDENCE DIAGNOSTICS")

evidence_summary = (
    trajectory.groupby("trajectory_class", as_index=False)
    .agg(
        grid_count=(GRID_COL, "count"),
        mean_relative_sen_slope=("relative_sen_slope_percentile_per_year", "mean"),
        median_relative_sen_slope=("relative_sen_slope_percentile_per_year", "median"),
        relative_slope_ci_excludes_zero_count=("relative_slope_ci_excludes_zero", "sum"),
        mean_absolute_lst_sen_slope_c_per_year=("sen_slope_c_per_year", "mean"),
        median_absolute_lst_sen_slope_c_per_year=("sen_slope_c_per_year", "median"),
        raw_absolute_mk_significant_count=("mk_p_value", lambda x: int((x < 0.05).sum())),
        fdr_absolute_mk_significant_count=("mk_fdr_q_value", lambda x: int((x < 0.05).sum())),
        relative_vs_absolute_direction_agree_count=("relative_vs_absolute_direction_agree", "sum"),
    )
)

evidence_summary["relative_slope_ci_excludes_zero_percent"] = (
    evidence_summary["relative_slope_ci_excludes_zero_count"]
    / evidence_summary["grid_count"] * 100.0
)
evidence_summary["relative_vs_absolute_direction_agree_percent"] = (
    evidence_summary["relative_vs_absolute_direction_agree_count"]
    / evidence_summary["grid_count"] * 100.0
)

print(evidence_summary.to_string(index=False))
print("\nINTERPRETATION:")
print("- relative Sen slope defines the planning trajectory class")
print("- absolute Step-23 Sen slope describes absolute LST trend separately")
print("- disagreement is allowed because they answer different questions")
print("- no FDR-significant absolute trend is converted into a significant claim")


section("5. NO-2017 ROBUSTNESS OF THE FULL RELATIVE-TRAJECTORY METHOD")

no2017 = percentile_data[percentile_data[YEAR_COL] != 2017].copy()
no2017 = add_yearly_relative_heat_percentile(
    no2017[[GRID_COL, AREA_COL, YEAR_COL, TARGET]].copy()
)

baseline_no2017 = [2015, 2016, 2018, 2019]

no2017_features = build_grid_trajectory_features(
    percentile_data=no2017,
    baseline_years=baseline_no2017,
    recent_years=RECENT_YEARS,
)

no2017_features = no2017_features[
    [
        GRID_COL,
        "baseline_relative_heat_median",
        "recent_relative_heat_median",
        "long_term_relative_heat_median",
        "relative_sen_slope_percentile_per_year",
        "trajectory_class",
    ]
].rename(
    columns={
        "baseline_relative_heat_median": "without_2017_baseline_relative_heat_median",
        "recent_relative_heat_median": "without_2017_recent_relative_heat_median",
        "long_term_relative_heat_median": "without_2017_long_term_relative_heat_median",
        "relative_sen_slope_percentile_per_year": "without_2017_relative_sen_slope",
        "trajectory_class": "without_2017_trajectory_class",
    }
)

no2017_comparison = trajectory[
    [
        GRID_COL,
        AREA_COL,
        "trajectory_class",
        "baseline_relative_heat_median",
        "recent_relative_heat_median",
        "long_term_relative_heat_median",
        "relative_sen_slope_percentile_per_year",
    ]
].merge(
    no2017_features,
    on=GRID_COL,
    how="left",
    validate="one_to_one",
)

no2017_comparison["trajectory_class_same_without_2017"] = (
    no2017_comparison["trajectory_class"]
    == no2017_comparison["without_2017_trajectory_class"]
)

no2017_comparison["absolute_relative_slope_change_without_2017"] = (
    no2017_comparison["relative_sen_slope_percentile_per_year"]
    - no2017_comparison["without_2017_relative_sen_slope"]
).abs()

no2017_comparison.to_csv(NO2017_COMPARISON_PATH, index=False)

pct_same_no2017 = float(
    no2017_comparison["trajectory_class_same_without_2017"].mean() * 100.0
)
median_slope_change_no2017 = float(
    no2017_comparison["absolute_relative_slope_change_without_2017"].median()
)

print(f"Trajectory class unchanged without 2017: {pct_same_no2017:.2f}%")
print(
    "Median absolute change in relative Sen slope without 2017: "
    f"{median_slope_change_no2017:.6f} percentile/year"
)
print(f"Saved: {NO2017_COMPARISON_PATH}")


section("6. FIXED THRESHOLD SENSITIVITY")

hot_variants = [0.70, 0.75, 0.80]
slope_variants = [0.008, 0.010, 0.012]
reference_labels = trajectory["trajectory_class"].to_numpy()
sensitivity_rows = []

for hot_threshold in hot_variants:
    for slope_threshold in slope_variants:
        labels = np.array(
            [
                classify_trajectory(
                    baseline_score=baseline,
                    recent_score=recent,
                    long_score=long_score,
                    relative_sen_slope=slope,
                    hot_threshold=hot_threshold,
                    stable_split=STABLE_SPLIT,
                    slope_threshold=slope_threshold,
                )
                for baseline, recent, long_score, slope in zip(
                    trajectory["baseline_relative_heat_median"],
                    trajectory["recent_relative_heat_median"],
                    trajectory["long_term_relative_heat_median"],
                    trajectory["relative_sen_slope_percentile_per_year"],
                )
            ],
            dtype=object,
        )

        row = {
            "hot_threshold": hot_threshold,
            "stable_split": STABLE_SPLIT,
            "relative_slope_threshold": slope_threshold,
            "is_reference_configuration": bool(
                np.isclose(hot_threshold, HOT_THRESHOLD)
                and np.isclose(slope_threshold, RELATIVE_SLOPE_THRESHOLD)
            ),
            "pct_same_as_reference_classification": float(
                np.mean(labels == reference_labels) * 100.0
            ),
        }

        for class_name in CLASS_ORDER:
            row[f"count_{class_name.lower().replace(' ', '_')}"] = int(
                np.sum(labels == class_name)
            )

        sensitivity_rows.append(row)

threshold_sensitivity = pd.DataFrame(sensitivity_rows)
threshold_sensitivity.to_csv(THRESHOLD_SENSITIVITY_PATH, index=False)

print(threshold_sensitivity.to_string(index=False))
print(f"\nSaved: {THRESHOLD_SENSITIVITY_PATH}")


section("7. BUILD MAP-READY TRAJECTORY OUTPUT")

grid = gpd.read_file(GRID_WGS84_PATH)
if GRID_COL not in grid.columns:
    raise RuntimeError("Grid geometry missing grid_id")

grid[GRID_COL] = grid[GRID_COL].astype(str)

if len(grid) != EXPECTED_GRIDS:
    raise RuntimeError("Grid geometry row count mismatch")

map_data = grid[[GRID_COL, "geometry"]].merge(
    trajectory,
    on=GRID_COL,
    how="left",
    validate="one_to_one",
)

map_data = gpd.GeoDataFrame(map_data, geometry="geometry", crs=grid.crs)

if map_data["trajectory_class"].isna().any():
    raise RuntimeError("Some map grids have no trajectory class")

map_data.to_file(MAP_PATH, driver="GeoJSON")
print(f"Map rows : {len(map_data):,}")
print(f"Map CRS  : {map_data.crs}")
print(f"Saved    : {MAP_PATH}")


section("8. SAVE TRAJECTORY FIGURES")

fig, ax = plt.subplots(figsize=(9, 8))
map_data.plot(column="trajectory_class", categorical=True, legend=True, ax=ax)
ax.set_title("Kaduwela Relative Spatial Heat Trajectories (2015-2025)")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(CLASS_MAP_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(class_counts["trajectory_class"], class_counts["grid_count"])
ax.set_ylabel("Grid count")
ax.set_title("Relative Heat Trajectory Class Distribution")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(CLASS_COUNT_FIG_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(trajectory["relative_sen_slope_percentile_per_year"], bins=50)
ax.axvline(-RELATIVE_SLOPE_THRESHOLD, linewidth=1)
ax.axvline(RELATIVE_SLOPE_THRESHOLD, linewidth=1)
ax.set_xlabel("Relative heat Sen slope (percentile/year)")
ax.set_ylabel("Grid count")
ax.set_title("Distribution of All-Year Relative Heat Sen Slopes")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(RELATIVE_SLOPE_FIG_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    trajectory["baseline_relative_heat_median"],
    trajectory["recent_relative_heat_median"],
    s=12,
    alpha=0.45,
)
ax.axhline(HOT_THRESHOLD, linewidth=1)
ax.axvline(HOT_THRESHOLD, linewidth=1)
ax.plot([0, 1], [0, 1], linewidth=1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Baseline relative heat median (2015-2019)")
ax.set_ylabel("Recent relative heat median (2021-2025)")
ax.set_title("Baseline vs Recent Relative Heat Status")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(BASELINE_RECENT_FIG_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

for path in [
    CLASS_MAP_PATH,
    CLASS_COUNT_FIG_PATH,
    RELATIVE_SLOPE_FIG_PATH,
    BASELINE_RECENT_FIG_PATH,
]:
    print(f"Saved: {path}")


section("9. SAVE TRAJECTORY METHODOLOGY")

methodology = {
    "step": "23B",
    "version": "relative_sen_v2",
    "analysis": "Robust five-class relative spatial heat trajectory classification",
    "historical_period": "2015-2025",
    "classes": CLASS_ORDER,
    "classification_scope": (
        "relative spatial heat trajectory within Kaduwela; "
        "not an absolute warming/cooling significance claim"
    ),
    "relative_heat_definition": (
        "within-year percentile rank of observed LST_C among all 2,194 grids"
    ),
    "direction_definition": (
        "Theil-Sen slope of yearly relative heat percentile across all available years"
    ),
    "baseline_status_window": BASELINE_YEARS,
    "recent_status_window": RECENT_YEARS,
    "status_statistic": "median",
    "reference_thresholds": {
        "hot_threshold": HOT_THRESHOLD,
        "stable_warm_cool_split": STABLE_SPLIT,
        "relative_sen_slope_threshold_percentile_per_year": RELATIVE_SLOPE_THRESHOLD,
        "approximate_decadal_relative_change_equivalent": RELATIVE_SLOPE_THRESHOLD * 10,
    },
    "rules": [
        "Emerging: relative Sen slope >= +0.01/year, baseline < 0.75, recent >= 0.75",
        "Intensifying: relative Sen slope >= +0.01/year and not Emerging",
        "Diminishing: relative Sen slope <= -0.01/year",
        "Persistent Hot: abs(relative Sen slope) < 0.01/year and long-term relative heat median >= 0.50",
        "Persistent Cool: abs(relative Sen slope) < 0.01/year and long-term relative heat median < 0.50",
    ],
    "step23_absolute_lst_role": (
        "Mann-Kendall p/q and absolute LST Sen slope retained as separate "
        "inferential/descriptive evidence; not overwritten by trajectory labels"
    ),
    "2017_policy": {
        "primary": "2017 retained",
        "secondary": "full relative-trajectory method repeated with 2017 omitted",
    },
    "threshold_sensitivity": {
        "hot_thresholds": hot_variants,
        "relative_slope_thresholds": slope_variants,
        "stable_split": STABLE_SPLIT,
        "purpose": (
            "robustness only; reference thresholds are not selected by optimizing class counts"
        ),
    },
}

METHOD_PATH.write_text(json.dumps(methodology, indent=2), encoding="utf-8")
print(json.dumps(methodology, indent=2))


section("10. STRICT STEP-23B v2 AUDIT")

class_count_sum = int(class_counts["grid_count"].sum())

one_class_per_grid = bool(
    len(trajectory) == EXPECTED_GRIDS
    and trajectory[GRID_COL].nunique() == EXPECTED_GRIDS
    and trajectory["trajectory_class"].notna().all()
)

all_year_relative_slope_finite = bool(
    np.isfinite(
        trajectory[
            [
                "relative_sen_slope_percentile_per_year",
                "relative_sen_slope_ci95_low",
                "relative_sen_slope_ci95_high",
            ]
        ].to_numpy(dtype=float)
    ).all()
)

map_alignment = bool(
    len(map_data) == EXPECTED_GRIDS
    and map_data[GRID_COL].nunique() == EXPECTED_GRIDS
)

audit_rows = [
    {"check": "step23_audit", "status": "PASS", "details": "Step 23 audit verified fully PASS."},
    {"check": "input_hashes", "status": "PASS", "details": "All frozen Step-23/base/geometry input hashes verified."},
    {"check": "all_2015_2025_years_used_for_relative_direction", "status": "PASS", "details": "Relative Theil-Sen direction uses all 11 years 2015-2025."},
    {"check": "trajectory_grid_count", "status": "PASS" if len(trajectory) == EXPECTED_GRIDS else "FAIL", "details": f"actual={len(trajectory)}, expected={EXPECTED_GRIDS}"},
    {"check": "exactly_one_class_per_grid", "status": "PASS" if one_class_per_grid else "FAIL", "details": "Every grid has one non-null class."},
    {"check": "class_count_sum", "status": "PASS" if class_count_sum == EXPECTED_GRIDS else "FAIL", "details": f"sum={class_count_sum}, expected={EXPECTED_GRIDS}"},
    {"check": "only_expected_five_classes", "status": "PASS" if not unexpected_classes else "FAIL", "details": str(sorted(trajectory["trajectory_class"].unique().tolist()))},
    {"check": "relative_sen_slopes_finite", "status": "PASS" if all_year_relative_slope_finite else "FAIL", "details": str(all_year_relative_slope_finite)},
    {"check": "reference_thresholds_fixed", "status": "PASS", "details": f"hot={HOT_THRESHOLD}, stable_split={STABLE_SPLIT}, relative_slope={RELATIVE_SLOPE_THRESHOLD}"},
    {"check": "absolute_step23_evidence_preserved", "status": "PASS", "details": "Original Step-23 p/q/Sen fields are attached unchanged."},
    {"check": "trajectory_scope_explicit", "status": "PASS", "details": "Labels explicitly describe relative spatial heat within Kaduwela."},
    {"check": "trajectory_not_absolute_significance_claim", "status": "PASS", "details": "No trajectory class is presented as an absolute MK significance result."},
    {"check": "2017_preserved_primary", "status": "PASS", "details": "Primary all-year trajectory includes 2017."},
    {"check": "2017_sensitivity_saved", "status": "PASS", "details": f"No-2017 class agreement={pct_same_no2017:.2f}%."},
    {"check": "threshold_sensitivity_saved", "status": "PASS" if len(threshold_sensitivity) == 9 else "FAIL", "details": f"configurations={len(threshold_sensitivity)}"},
    {"check": "map_alignment", "status": "PASS" if map_alignment else "FAIL", "details": f"map_rows={len(map_data)}, expected={EXPECTED_GRIDS}"},
    {"check": "forecast_model_used_or_modified", "status": "PASS", "details": "No. Historical observed LST + Step23 evidence only."},
]

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(AUDIT_PATH, index=False)

print(audit_df.to_string(index=False))

if (audit_df["status"] != "PASS").any():
    raise RuntimeError("One or more Step-23B v2 audit checks failed")


section("11. SAVE STEP-23B v2 REPORT")

report_lines = [
    "STEP 23B v2 - ROBUST FIVE-CLASS RELATIVE HEAT TRAJECTORY CLASSIFICATION",
    "=" * 116,
    "",
    "METHOD",
    "-" * 116,
    "Relative heat = within-year percentile rank of observed LST_C.",
    "Trajectory direction = Theil-Sen slope across all years 2015-2025.",
    "Baseline heat status = median percentile over 2015-2019.",
    "Recent heat status = median percentile over 2021-2025.",
    f"Hot threshold = {HOT_THRESHOLD:.2f}.",
    f"Stable split = {STABLE_SPLIT:.2f}.",
    f"Operational direction threshold = ±{RELATIVE_SLOPE_THRESHOLD:.3f} percentile/year.",
    "",
    "CLASS DISTRIBUTION",
    "-" * 116,
    class_counts.to_string(index=False),
    "",
    "EVIDENCE DIAGNOSTICS",
    "-" * 116,
    evidence_summary.to_string(index=False),
    "",
    "NO-2017 ROBUSTNESS",
    "-" * 116,
    f"Trajectory class unchanged without 2017: {pct_same_no2017:.2f}%",
    f"Median absolute relative-Sen-slope change without 2017: {median_slope_change_no2017:.6f} percentile/year",
    "",
    "INTERPRETATION",
    "-" * 116,
    "These labels describe each grid's relative spatial heat trajectory inside Kaduwela. They do not claim statistically significant absolute LST warming/cooling.",
    "Step-23 absolute LST Mann-Kendall and Sen-slope evidence is preserved in the per-grid output and should be reported separately.",
]

REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

print(f"Saved: {REPORT_PATH}")
print(f"Saved: {AUDIT_PATH}")


section("12. REPRODUCIBILITY HASHES")

hash_files = [
    BASE_DATA_PATH,
    STEP23_GRID_PATH,
    STEP23_SENSITIVITY_PATH,
    STEP23_AUDIT_PATH,
    GRID_WGS84_PATH,
    TRAJECTORY_PATH,
    CLASS_COUNTS_PATH,
    YEARLY_PERCENTILES_PATH,
    NO2017_COMPARISON_PATH,
    THRESHOLD_SENSITIVITY_PATH,
    MAP_PATH,
    CLASS_MAP_PATH,
    CLASS_COUNT_FIG_PATH,
    RELATIVE_SLOPE_FIG_PATH,
    BASELINE_RECENT_FIG_PATH,
    METHOD_PATH,
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


section("STEP 23B v2 COMPLETED SUCCESSFULLY")

print(
    "Exactly one relative spatial heat trajectory class was assigned to every grid.\n"
    "Trajectory direction uses robust all-year Theil-Sen slope of yearly relative heat percentile.\n"
    "All years 2015-2025 contribute to direction.\n"
    "Baseline/recent status uses 5-year medians for greater robustness.\n"
    "Step-23 absolute LST MK/Sen evidence is preserved separately.\n"
    "Primary analysis includes 2017; no-2017 sensitivity was saved.\n"
    "Threshold sensitivity was saved without optimizing the class distribution.\n"
    f"Total classified grids: {len(trajectory):,}\n"
    "\nNEXT: review class distribution, no-2017 robustness and threshold sensitivity before finalizing Step 23B."
)
