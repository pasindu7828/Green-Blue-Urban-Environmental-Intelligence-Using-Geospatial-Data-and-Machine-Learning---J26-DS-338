"""
STEP 03 - DATA QUALITY ANALYSIS
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Perform a deeper data-quality audit after Step 01 (data understanding)
and Step 02 (target validation).

This step focuses on:
1. schema and type quality
2. missing / infinite / blank values
3. grid_id + year integrity
4. domain/range checks
5. static-feature consistency through time
6. year-wise feature stability
7. focused investigation of the unusual 2017 LST pattern
8. large grid-level temporal jumps
9. persistent-hotspot consistency
10. CSV ↔ grid-geometry alignment
11. basic geometry / CRS metadata quality

IMPORTANT
---------
- This script DOES NOT modify the raw CSV.
- It DOES NOT delete outliers.
- It DOES NOT impute, scale, normalize, transform, or winsorize values.
- It DOES NOT train models.
- 2017 anomalies are investigated, not automatically treated as errors.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except ImportError:
    gpd = None


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "kaduwela_master_raw.csv"
GRID_GEOJSON_PATH = PROJECT_ROOT / "data" / "raw" / "kaduwela_grid_geometry.geojson"
BOUNDARY_KML_PATH = (
    PROJECT_ROOT / "data" / "raw" / "boundary" / "kaduwela_kmc_boundary.kml"
)

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MAIN_REPORT_PATH = REPORT_DIR / "03_data_quality_report.txt"
QUALITY_SUMMARY_PATH = REPORT_DIR / "03_data_quality_summary.csv"
SCHEMA_PATH = REPORT_DIR / "03_schema_quality.csv"
DOMAIN_PATH = REPORT_DIR / "03_domain_checks.csv"
STATIC_PATH = REPORT_DIR / "03_static_feature_consistency.csv"
YEAR_FEATURE_PATH = REPORT_DIR / "03_year_feature_summary.csv"

ANOMALY_2017_PATH = REPORT_DIR / "03_2017_low_lst_anomaly_comparison.csv"
ANOMALY_2017_AREA_PATH = REPORT_DIR / "03_2017_low_lst_area_counts.csv"
ANOMALY_2017_EFFECT_PATH = REPORT_DIR / "03_2017_anomaly_feature_effects.csv"
ANOMALY_2017_CHANGE_PATH = (
    REPORT_DIR / "03_2017_anomaly_feature_change_summary.csv"
)

TEMPORAL_JUMPS_PATH = REPORT_DIR / "03_large_temporal_jump_candidates.csv"
HOTSPOT_PATH = REPORT_DIR / "03_persistent_hotspot_candidates.csv"

GEOMETRY_PATH = REPORT_DIR / "03_grid_geometry_quality.csv"
GEOMETRY_ID_PATH = REPORT_DIR / "03_grid_id_alignment.csv"
BOUNDARY_PATH = REPORT_DIR / "03_boundary_geometry_quality.csv"

ANOMALY_PROFILE_FIG = FIGURE_DIR / "03_2017_low_lst_profiles.png"
HOTSPOT_PROFILE_FIG = FIGURE_DIR / "03_persistent_hotspot_profiles.png"


# =============================================================================
# 2. EXPECTED SCHEMA
# =============================================================================

EXPECTED_COLUMNS = [
    "grid_id",
    "year",
    "LST_C",
    "NDVI",
    "NDBI",
    "NDWI",
    "EVI",
    "Albedo",
    "NDBI_ADJ",
    "NDVI_ADJ",
    "NightLights",
    "green_mask",
    "building_mask",
    "road_mask",
    "dist_road",
    "dist_main_road",
    "dist_water",
    "LCZ",
    "area_name",
]

NUMERIC_COLUMNS = [
    "year",
    "LST_C",
    "NDVI",
    "NDBI",
    "NDWI",
    "EVI",
    "Albedo",
    "NDBI_ADJ",
    "NDVI_ADJ",
    "NightLights",
    "green_mask",
    "building_mask",
    "road_mask",
    "dist_road",
    "dist_main_road",
    "dist_water",
    "LCZ",
]

FEATURE_COLUMNS = [
    "NDVI",
    "NDBI",
    "NDWI",
    "EVI",
    "Albedo",
    "NDBI_ADJ",
    "NDVI_ADJ",
    "NightLights",
    "green_mask",
    "building_mask",
    "road_mask",
    "dist_road",
    "dist_main_road",
    "dist_water",
    "LCZ",
]

SATELLITE_DYNAMIC_FEATURES = [
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

# These should normally be stable for a fixed grid if they were generated
# from one static GIS layer. Any changes are reported for investigation.
STATIC_DIAGNOSTIC_COLUMNS = [
    "area_name",
    "dist_road",
    "dist_main_road",
    "dist_water",
    "building_mask",
    "road_mask",
]


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float:
    """
    Standardized mean difference using pooled SD.
    Used only as a descriptive diagnostic, not as a significance test.
    """
    a = safe_numeric(a).replace([np.inf, -np.inf], np.nan).dropna()
    b = safe_numeric(b).replace([np.inf, -np.inf], np.nan).dropna()

    if len(a) < 2 or len(b) < 2:
        return np.nan

    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)

    pooled_var = (
        ((len(a) - 1) * var_a + (len(b) - 1) * var_b)
        / (len(a) + len(b) - 2)
    )

    if pooled_var <= 0 or np.isnan(pooled_var):
        return np.nan

    return (a.mean() - b.mean()) / np.sqrt(pooled_var)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# 4. LOAD MASTER DATA
# =============================================================================

section("STEP 03 - DATA QUALITY ANALYSIS")

print(f"Project root : {PROJECT_ROOT}")
print(f"Master CSV   : {RAW_DATA_PATH}")

if not RAW_DATA_PATH.exists():
    print("\nERROR: Master CSV not found.")
    print(f"Expected: {RAW_DATA_PATH}")
    sys.exit(1)

try:
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
except Exception as exc:
    print(f"\nERROR: Could not read master CSV.\n{exc}")
    sys.exit(1)

if df.empty:
    print("\nERROR: Master dataset is empty.")
    sys.exit(1)

print("Master dataset loaded successfully.")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# =============================================================================
# 5. SCHEMA / TYPE QUALITY
# =============================================================================

section("1. SCHEMA AND TYPE QUALITY")

schema_rows = []

missing_expected_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
unexpected_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

for col in EXPECTED_COLUMNS:
    present = col in df.columns

    if present:
        dtype = str(df[col].dtype)
        non_null = int(df[col].notna().sum())
        missing = int(df[col].isna().sum())

        if col in NUMERIC_COLUMNS:
            converted = safe_numeric(df[col])
            parse_failures = int((df[col].notna() & converted.isna()).sum())
        else:
            parse_failures = 0
    else:
        dtype = None
        non_null = 0
        missing = len(df)
        parse_failures = None

    schema_rows.append(
        {
            "column": col,
            "present": present,
            "dtype": dtype,
            "non_null_count": non_null,
            "missing_count": missing,
            "numeric_parse_failures": parse_failures,
        }
    )

schema_quality = pd.DataFrame(schema_rows)
schema_quality.to_csv(SCHEMA_PATH, index=False)

print(f"Missing expected columns : {missing_expected_columns}")
print(f"Unexpected columns       : {unexpected_columns}")
print("\nSchema summary:")
print(schema_quality.to_string(index=False))


# =============================================================================
# 6. GLOBAL MISSING / INFINITE / BLANK QUALITY
# =============================================================================

section("2. MISSING, INFINITE, AND BLANK VALUES")

total_missing = int(df.isna().sum().sum())

numeric_present = [c for c in NUMERIC_COLUMNS if c in df.columns]

if numeric_present:
    numeric_frame = df[numeric_present].apply(pd.to_numeric, errors="coerce")
    total_positive_inf = int(np.isposinf(numeric_frame.to_numpy(dtype=float)).sum())
    total_negative_inf = int(np.isneginf(numeric_frame.to_numpy(dtype=float)).sum())
else:
    numeric_frame = pd.DataFrame()
    total_positive_inf = 0
    total_negative_inf = 0

blank_grid_ids = 0
blank_area_names = 0

if "grid_id" in df.columns:
    blank_grid_ids = int(
        df["grid_id"].astype(str).str.strip().eq("").sum()
    )

if "area_name" in df.columns:
    blank_area_names = int(
        df["area_name"].astype(str).str.strip().eq("").sum()
    )

print(f"Total missing cells       : {total_missing:,}")
print(f"Positive infinity values  : {total_positive_inf:,}")
print(f"Negative infinity values  : {total_negative_inf:,}")
print(f"Blank grid_id values      : {blank_grid_ids:,}")
print(f"Blank area_name values    : {blank_area_names:,}")


# =============================================================================
# 7. KEY INTEGRITY
# =============================================================================

section("3. GRID-YEAR KEY INTEGRITY")

full_duplicates = int(df.duplicated().sum())

duplicate_grid_year_rows = None
unique_grid_year_pairs = None

if {"grid_id", "year"}.issubset(df.columns):
    duplicate_grid_year_rows = int(
        df.duplicated(["grid_id", "year"], keep=False).sum()
    )
    unique_grid_year_pairs = int(
        df[["grid_id", "year"]].drop_duplicates().shape[0]
    )

    print(f"Full duplicate rows                : {full_duplicates:,}")
    print(f"Unique grid_id + year combinations : {unique_grid_year_pairs:,}")
    print(f"Rows in duplicate grid-year keys   : {duplicate_grid_year_rows:,}")
else:
    print("grid_id/year columns missing; key validation could not run.")


# =============================================================================
# 8. DOMAIN / RANGE CHECKS
# =============================================================================

section("4. DOMAIN AND RANGE CHECKS")

domain_rows = []

def add_range_check(
    column: str,
    lower=None,
    upper=None,
    lower_inclusive=True,
    upper_inclusive=True,
    check_name=None,
    severity="warning",
    note="",
):
    if column not in df.columns:
        domain_rows.append(
            {
                "column": column,
                "check": check_name or "range_check",
                "severity": severity,
                "lower_expected": lower,
                "upper_expected": upper,
                "violation_count": None,
                "violation_percent": None,
                "observed_min": None,
                "observed_max": None,
                "note": "Column not present.",
            }
        )
        return

    s = safe_numeric(df[column])
    finite = s.replace([np.inf, -np.inf], np.nan).dropna()

    violation = pd.Series(False, index=s.index)

    if lower is not None:
        if lower_inclusive:
            violation |= s < lower
        else:
            violation |= s <= lower

    if upper is not None:
        if upper_inclusive:
            violation |= s > upper
        else:
            violation |= s >= upper

    violation_count = int(violation.fillna(False).sum())

    domain_rows.append(
        {
            "column": column,
            "check": check_name or "range_check",
            "severity": severity,
            "lower_expected": lower,
            "upper_expected": upper,
            "violation_count": violation_count,
            "violation_percent": (
                100 * violation_count / len(df) if len(df) else np.nan
            ),
            "observed_min": finite.min() if not finite.empty else np.nan,
            "observed_max": finite.max() if not finite.empty else np.nan,
            "note": note,
        }
    )


# Standard remote-sensing index ranges.
add_range_check(
    "NDVI", -1, 1, note="Standard normalized-difference range."
)
add_range_check(
    "NDBI", -1, 1, note="Standard normalized-difference range."
)
add_range_check(
    "NDWI", -1, 1, note="Standard normalized-difference range."
)
add_range_check(
    "NDBI_ADJ", -1, 1, note="Neighbourhood mean of NDBI should remain in [-1,1]."
)
add_range_check(
    "NDVI_ADJ", -1, 1, note="Neighbourhood mean of NDVI should remain in [-1,1]."
)

# EVI can occasionally exceed [-1, 1], so a wider sanity range is used.
add_range_check(
    "EVI",
    -1,
    2,
    check_name="broad_plausibility_range",
    note="Wide diagnostic range; no automatic deletion if violated.",
)

add_range_check(
    "Albedo", 0, 1, note="Physical proportion/reflection diagnostic."
)
add_range_check(
    "NightLights",
    0,
    None,
    note="Negative radiance values would require investigation.",
)
add_range_check(
    "green_mask",
    0,
    1,
    note="Grid-aggregated green proportion should remain between 0 and 1.",
)

for distance_col in ["dist_road", "dist_main_road", "dist_water"]:
    add_range_check(
        distance_col,
        0,
        None,
        note="Distances must not be negative.",
    )

# LCZ codes are class labels 1-17 in the source product, but grid aggregation
# may produce fractional means. Range violations are errors; fractions are a
# separate methodological flag below.
add_range_check(
    "LCZ",
    1,
    17,
    note="LCZ source classes are in the 1-17 range; fractional values are checked separately.",
)

# Very broad LST sanity range. This is not an outlier-removal rule.
add_range_check(
    "LST_C",
    -20,
    80,
    check_name="broad_temperature_sanity_range",
    note="Broad Celsius sanity check only; statistical extremes are handled separately.",
)

# Binary checks
for binary_col in ["building_mask", "road_mask"]:
    if binary_col in df.columns:
        s = safe_numeric(df[binary_col])
        invalid_binary = int((~s.isin([0, 1]) & s.notna()).sum())

        domain_rows.append(
            {
                "column": binary_col,
                "check": "binary_0_1_check",
                "severity": "warning",
                "lower_expected": 0,
                "upper_expected": 1,
                "violation_count": invalid_binary,
                "violation_percent": 100 * invalid_binary / len(df),
                "observed_min": s.min(),
                "observed_max": s.max(),
                "note": "Expected binary GIS indicator.",
            }
        )

# Year range based on research period
add_range_check(
    "year",
    2015,
    2025,
    check_name="research_period_check",
    note="Expected study period is 2015-2025.",
)

domain_checks = pd.DataFrame(domain_rows)
domain_checks.to_csv(DOMAIN_PATH, index=False)

print(domain_checks.to_string(index=False))


# =============================================================================
# 9. LCZ AGGREGATION QUALITY FLAG
# =============================================================================

section("5. LCZ AGGREGATION DIAGNOSTIC")

lcz_fractional_count = 0
lcz_fractional_percent = 0.0

if "LCZ" in df.columns:
    lcz = safe_numeric(df["LCZ"])
    finite_lcz = lcz.replace([np.inf, -np.inf], np.nan).dropna()

    fractional_mask = (
        finite_lcz.sub(finite_lcz.round()).abs() > 1e-9
    )

    lcz_fractional_count = int(fractional_mask.sum())
    lcz_fractional_percent = (
        100 * lcz_fractional_count / len(finite_lcz)
        if len(finite_lcz)
        else 0.0
    )

    print(f"LCZ unique numeric values : {finite_lcz.nunique():,}")
    print(f"Fractional LCZ values     : {lcz_fractional_count:,}")
    print(f"Fractional LCZ percent    : {lcz_fractional_percent:.2f}%")
    print(
        "Interpretation: fractional LCZ values are not automatically invalid. "
        "They suggest the class raster may have been averaged within 210 m grids. "
        "Later feature analysis must decide whether to keep this continuous aggregate "
        "or derive a dominant LCZ class."
    )
else:
    print("LCZ column not present.")


# =============================================================================
# 10. STATIC-FEATURE CONSISTENCY THROUGH TIME
# =============================================================================

section("6. STATIC-FEATURE CONSISTENCY THROUGH TIME")

static_rows = []

if "grid_id" in df.columns:
    for col in STATIC_DIAGNOSTIC_COLUMNS:
        if col not in df.columns:
            continue

        per_grid_unique = (
            df.groupby("grid_id")[col]
            .nunique(dropna=False)
        )

        changing_grids = int((per_grid_unique > 1).sum())
        max_unique_per_grid = int(per_grid_unique.max())

        static_rows.append(
            {
                "column": col,
                "unique_values_overall": int(df[col].nunique(dropna=False)),
                "grids_with_more_than_one_value": changing_grids,
                "percent_grids_changing": (
                    100 * changing_grids / per_grid_unique.shape[0]
                    if per_grid_unique.shape[0]
                    else np.nan
                ),
                "max_unique_values_within_one_grid": max_unique_per_grid,
                "interpretation": (
                    "Investigate if changing; expected to be stable if created from one static GIS layer."
                ),
            }
        )

static_consistency = pd.DataFrame(static_rows)
static_consistency.to_csv(STATIC_PATH, index=False)

print(static_consistency.to_string(index=False))


# =============================================================================
# 11. YEAR-WISE FEATURE SUMMARY / DRIFT SCREEN
# =============================================================================

section("7. YEAR-WISE FEATURE SUMMARY")

year_feature_rows = []

if "year" in df.columns:
    for year, group in df.groupby("year", sort=True):
        for col in ["LST_C"] + FEATURE_COLUMNS:
            if col not in group.columns:
                continue

            s = safe_numeric(group[col]).replace([np.inf, -np.inf], np.nan)

            year_feature_rows.append(
                {
                    "year": year,
                    "feature": col,
                    "count": int(s.notna().sum()),
                    "mean": s.mean(),
                    "median": s.median(),
                    "std": s.std(),
                    "min": s.min(),
                    "q05": s.quantile(0.05),
                    "q25": s.quantile(0.25),
                    "q75": s.quantile(0.75),
                    "q95": s.quantile(0.95),
                    "max": s.max(),
                }
            )

year_feature_summary = pd.DataFrame(year_feature_rows)
year_feature_summary.to_csv(YEAR_FEATURE_PATH, index=False)

print(
    "Saved detailed year-wise feature statistics for LST_C and all available predictors."
)


# =============================================================================
# 12. FOCUSED 2017 LOW-LST ANOMALY INVESTIGATION
# =============================================================================

section("8. FOCUSED 2017 LOW-LST ANOMALY INVESTIGATION")

anomaly_2017 = pd.DataFrame()
anomaly_effects = pd.DataFrame()
anomaly_change_summary = pd.DataFrame()
anomaly_area_counts = pd.DataFrame()

required_2017_years = {2016, 2017, 2018}
available_years = set(df["year"].dropna().astype(int).unique())

if required_2017_years.issubset(available_years):
    data_2017 = df[df["year"] == 2017].copy()
    lst_2017 = safe_numeric(data_2017["LST_C"])

    q1_2017 = lst_2017.quantile(0.25)
    q3_2017 = lst_2017.quantile(0.75)
    iqr_2017 = q3_2017 - q1_2017

    lower_fence_2017 = q1_2017 - 1.5 * iqr_2017
    upper_fence_2017 = q3_2017 + 1.5 * iqr_2017

    low_2017 = data_2017[lst_2017 < lower_fence_2017].copy()
    high_2017 = data_2017[lst_2017 > upper_fence_2017].copy()

    anomalous_grid_ids = set(low_2017["grid_id"].astype(str))

    print(f"2017 Q1                  : {q1_2017:.6f}")
    print(f"2017 Q3                  : {q3_2017:.6f}")
    print(f"2017 IQR                 : {iqr_2017:.6f}")
    print(f"2017 lower IQR fence     : {lower_fence_2017:.6f}")
    print(f"2017 upper IQR fence     : {upper_fence_2017:.6f}")
    print(f"2017 low-LST candidates  : {len(low_2017):,}")
    print(f"2017 high-LST candidates : {len(high_2017):,}")

    # Compare flagged grids in 2016, 2017, 2018.
    compare_cols = [
        "grid_id",
        "year",
        "LST_C",
        "NDVI",
        "NDBI",
        "NDWI",
        "EVI",
        "Albedo",
        "NDBI_ADJ",
        "NDVI_ADJ",
        "NightLights",
        "green_mask",
        "building_mask",
        "road_mask",
        "dist_road",
        "dist_main_road",
        "dist_water",
        "LCZ",
        "area_name",
    ]

    compare_cols = [c for c in compare_cols if c in df.columns]

    anomaly_2017 = (
        df[
            df["grid_id"].astype(str).isin(anomalous_grid_ids)
            & df["year"].isin([2016, 2017, 2018])
        ][compare_cols]
        .sort_values(["grid_id", "year"])
        .copy()
    )

    anomaly_2017.to_csv(ANOMALY_2017_PATH, index=False)

    # Count affected grids by area.
    if "area_name" in low_2017.columns:
        anomaly_area_counts = (
            low_2017.groupby("area_name")
            .agg(
                low_2017_grid_count=("grid_id", "nunique"),
                mean_low_2017_lst=("LST_C", "mean"),
                min_low_2017_lst=("LST_C", "min"),
            )
            .reset_index()
            .sort_values(
                ["low_2017_grid_count", "mean_low_2017_lst"],
                ascending=[False, True],
            )
        )
        anomaly_area_counts.to_csv(ANOMALY_2017_AREA_PATH, index=False)

        print("\nAreas containing the most 2017 low-LST candidates:")
        print(anomaly_area_counts.head(15).to_string(index=False))

    # Compare 2017 anomalous grids vs other 2017 grids.
    effect_rows = []
    anomaly_mask_2017 = data_2017["grid_id"].astype(str).isin(anomalous_grid_ids)

    for col in ["LST_C"] + SATELLITE_DYNAMIC_FEATURES:
        if col not in data_2017.columns:
            continue

        anomalous_values = safe_numeric(
            data_2017.loc[anomaly_mask_2017, col]
        )
        normal_values = safe_numeric(
            data_2017.loc[~anomaly_mask_2017, col]
        )

        effect_rows.append(
            {
                "feature": col,
                "anomaly_mean": anomalous_values.mean(),
                "other_2017_mean": normal_values.mean(),
                "mean_difference": (
                    anomalous_values.mean() - normal_values.mean()
                ),
                "anomaly_median": anomalous_values.median(),
                "other_2017_median": normal_values.median(),
                "standardized_mean_difference": standardized_mean_difference(
                    anomalous_values, normal_values
                ),
            }
        )

    anomaly_effects = pd.DataFrame(effect_rows)
    anomaly_effects.to_csv(ANOMALY_2017_EFFECT_PATH, index=False)

    print("\n2017 anomalous-grid feature comparison:")
    print(anomaly_effects.to_string(index=False))

    # Quantify changes from 2016 -> 2017 -> 2018 on anomalous grids.
    change_rows = []

    for col in ["LST_C"] + SATELLITE_DYNAMIC_FEATURES:
        if col not in anomaly_2017.columns:
            continue

        pivot = (
            anomaly_2017.pivot_table(
                index="grid_id",
                columns="year",
                values=col,
                aggfunc="first",
            )
        )

        if not {2016, 2017, 2018}.issubset(pivot.columns):
            continue

        change_16_17 = pivot[2017] - pivot[2016]
        change_17_18 = pivot[2018] - pivot[2017]

        change_rows.append(
            {
                "feature": col,
                "grid_count": int(pivot[[2016, 2017, 2018]].dropna().shape[0]),
                "mean_2016": pivot[2016].mean(),
                "mean_2017": pivot[2017].mean(),
                "mean_2018": pivot[2018].mean(),
                "mean_change_2016_to_2017": change_16_17.mean(),
                "median_change_2016_to_2017": change_16_17.median(),
                "mean_abs_change_2016_to_2017": change_16_17.abs().mean(),
                "mean_change_2017_to_2018": change_17_18.mean(),
                "median_change_2017_to_2018": change_17_18.median(),
                "mean_abs_change_2017_to_2018": change_17_18.abs().mean(),
            }
        )

    anomaly_change_summary = pd.DataFrame(change_rows)
    anomaly_change_summary.to_csv(ANOMALY_2017_CHANGE_PATH, index=False)

    print("\n2016 → 2017 → 2018 change summary for flagged grids:")
    print(anomaly_change_summary.to_string(index=False))

else:
    lower_fence_2017 = np.nan
    anomalous_grid_ids = set()
    print("2016/2017/2018 are not all present; focused anomaly analysis skipped.")


# =============================================================================
# 13. LARGE GRID-LEVEL TEMPORAL JUMPS
# =============================================================================

section("9. LARGE GRID-LEVEL TEMPORAL JUMPS")

jump_columns = [
    "grid_id",
    "year",
    "LST_C",
    "NDVI",
    "NDBI",
    "NDWI",
    "EVI",
    "Albedo",
    "NightLights",
    "green_mask",
    "area_name",
]

jump_columns = [c for c in jump_columns if c in df.columns]

temporal = df[jump_columns].sort_values(["grid_id", "year"]).copy()

temporal["previous_year"] = temporal.groupby("grid_id")["year"].shift(1)
temporal["previous_lst"] = temporal.groupby("grid_id")["LST_C"].shift(1)
temporal["lst_change"] = temporal["LST_C"] - temporal["previous_lst"]
temporal["abs_lst_change"] = temporal["lst_change"].abs()

for col in SATELLITE_DYNAMIC_FEATURES:
    if col in temporal.columns:
        previous = temporal.groupby("grid_id")[col].shift(1)
        temporal[f"{col}_change"] = temporal[col] - previous

valid_abs_changes = temporal["abs_lst_change"].dropna()

if not valid_abs_changes.empty:
    jump_threshold_99 = valid_abs_changes.quantile(0.99)

    large_jumps = (
        temporal[temporal["abs_lst_change"] >= jump_threshold_99]
        .copy()
        .sort_values("abs_lst_change", ascending=False)
    )

    large_jumps.to_csv(TEMPORAL_JUMPS_PATH, index=False)

    print(f"99th percentile absolute LST change: {jump_threshold_99:.6f}")
    print(f"Large-jump candidate rows           : {len(large_jumps):,}")
    print("\nTop 15 temporal jumps:")
    print(large_jumps.head(15).to_string(index=False))
else:
    jump_threshold_99 = np.nan
    large_jumps = pd.DataFrame()
    print("No valid temporal changes available.")


# =============================================================================
# 14. PERSISTENT HOTSPOT CONSISTENCY
# =============================================================================

section("10. PERSISTENT HOTSPOT CONSISTENCY")

hotspot_source = df[["grid_id", "year", "LST_C"]].copy()
if "area_name" in df.columns:
    hotspot_source["area_name"] = df["area_name"]

year_99 = (
    hotspot_source.groupby("year")["LST_C"]
    .quantile(0.99)
    .rename("year_lst_99th_percentile")
)

hotspot_source = hotspot_source.join(year_99, on="year")
hotspot_source["top_1_percent_in_year"] = (
    hotspot_source["LST_C"] >= hotspot_source["year_lst_99th_percentile"]
)

aggregation = {
    "top_1_percent_in_year": "sum",
    "LST_C": ["mean", "max"],
}

hotspot_summary = (
    hotspot_source.groupby("grid_id")
    .agg(aggregation)
)

hotspot_summary.columns = [
    "top_1_percent_year_count",
    "mean_lst_all_years",
    "max_lst_all_years",
]

hotspot_summary = hotspot_summary.reset_index()

if "area_name" in hotspot_source.columns:
    stable_area = (
        hotspot_source.groupby("grid_id")["area_name"]
        .first()
        .reset_index()
    )
    hotspot_summary = hotspot_summary.merge(
        stable_area,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

hotspot_summary["persistent_hotspot_candidate"] = (
    hotspot_summary["top_1_percent_year_count"] >= 3
)

persistent_hotspots = (
    hotspot_summary[
        hotspot_summary["persistent_hotspot_candidate"]
    ]
    .copy()
    .sort_values(
        ["top_1_percent_year_count", "max_lst_all_years"],
        ascending=[False, False],
    )
)

persistent_hotspots.to_csv(HOTSPOT_PATH, index=False)

print(
    "Persistent-hotspot rule: grid appears in its year's hottest 1% "
    "for at least 3 different years."
)
print(f"Persistent hotspot candidates: {len(persistent_hotspots):,}")

if not persistent_hotspots.empty:
    print("\nTop persistent-hotspot candidates:")
    print(persistent_hotspots.head(20).to_string(index=False))


# =============================================================================
# 15. GRID GEOJSON QUALITY + CSV ALIGNMENT
# =============================================================================

section("11. GRID GEOJSON QUALITY AND CSV ALIGNMENT")

geometry_quality_rows = []
geometry_id_alignment = pd.DataFrame()

grid_crs_suspect = False
csv_grids_missing_geometry = 0
geometry_grids_missing_csv = 0
invalid_geometry_count = None
empty_geometry_count = None
duplicate_geometry_grid_ids = None

if GRID_GEOJSON_PATH.exists() and gpd is not None:
    try:
        grid_gdf = gpd.read_file(GRID_GEOJSON_PATH)

        if "grid_id" not in grid_gdf.columns:
            geometry_quality_rows.append(
                {
                    "metric": "grid_id_column_present",
                    "value": False,
                    "status": "REVIEW_REQUIRED",
                    "note": "GeoJSON has no grid_id column.",
                }
            )
        else:
            grid_gdf = grid_gdf.copy()
            grid_gdf["grid_id_str"] = grid_gdf["grid_id"].astype(str)
            csv_grid_ids = set(df["grid_id"].astype(str))
            geometry_grid_ids = set(grid_gdf["grid_id_str"])

            missing_in_geometry = sorted(
                csv_grid_ids - geometry_grid_ids
            )
            missing_in_csv = sorted(
                geometry_grid_ids - csv_grid_ids
            )

            csv_grids_missing_geometry = len(missing_in_geometry)
            geometry_grids_missing_csv = len(missing_in_csv)

            alignment_rows = []

            for gid in missing_in_geometry:
                alignment_rows.append(
                    {
                        "grid_id": gid,
                        "alignment_issue": "present_in_csv_missing_in_geojson",
                    }
                )

            for gid in missing_in_csv:
                alignment_rows.append(
                    {
                        "grid_id": gid,
                        "alignment_issue": "present_in_geojson_missing_in_csv",
                    }
                )

            geometry_id_alignment = pd.DataFrame(alignment_rows)

            if geometry_id_alignment.empty:
                geometry_id_alignment = pd.DataFrame(
                    columns=["grid_id", "alignment_issue"]
                )

            geometry_id_alignment.to_csv(
                GEOMETRY_ID_PATH,
                index=False,
            )

            duplicate_geometry_grid_ids = int(
                grid_gdf.duplicated("grid_id_str", keep=False).sum()
            )

        invalid_geometry_count = int(
            (~grid_gdf.geometry.is_valid).sum()
        )
        empty_geometry_count = int(
            grid_gdf.geometry.is_empty.sum()
        )

        geometry_types = sorted(
            grid_gdf.geometry.geom_type.dropna().unique().tolist()
        )

        bounds = grid_gdf.total_bounds
        minx, miny, maxx, maxy = bounds

        crs_string = str(grid_gdf.crs)

        # Detect obviously impossible longitude/latitude coordinates if metadata
        # claims a geographic CRS.
        if grid_gdf.crs is not None and grid_gdf.crs.is_geographic:
            if (
                abs(minx) > 180
                or abs(maxx) > 180
                or abs(miny) > 90
                or abs(maxy) > 90
            ):
                grid_crs_suspect = True

        geometry_quality_rows.extend(
            [
                {
                    "metric": "geojson_row_count",
                    "value": len(grid_gdf),
                    "status": "INFO",
                    "note": "",
                },
                {
                    "metric": "geojson_unique_grid_ids",
                    "value": (
                        grid_gdf["grid_id_str"].nunique()
                        if "grid_id_str" in grid_gdf.columns
                        else np.nan
                    ),
                    "status": "INFO",
                    "note": "",
                },
                {
                    "metric": "duplicate_geojson_grid_id_rows",
                    "value": duplicate_geometry_grid_ids,
                    "status": (
                        "PASS"
                        if duplicate_geometry_grid_ids == 0
                        else "REVIEW_REQUIRED"
                    ),
                    "note": "",
                },
                {
                    "metric": "invalid_geometries",
                    "value": invalid_geometry_count,
                    "status": (
                        "PASS"
                        if invalid_geometry_count == 0
                        else "REVIEW_REQUIRED"
                    ),
                    "note": "",
                },
                {
                    "metric": "empty_geometries",
                    "value": empty_geometry_count,
                    "status": (
                        "PASS"
                        if empty_geometry_count == 0
                        else "REVIEW_REQUIRED"
                    ),
                    "note": "",
                },
                {
                    "metric": "geometry_types",
                    "value": ", ".join(geometry_types),
                    "status": "INFO",
                    "note": "",
                },
                {
                    "metric": "reported_crs",
                    "value": crs_string,
                    "status": (
                        "INVESTIGATE"
                        if grid_crs_suspect
                        else "INFO"
                    ),
                    "note": (
                        "CRS metadata appears geographic but coordinate magnitudes "
                        "look projected."
                        if grid_crs_suspect
                        else ""
                    ),
                },
                {
                    "metric": "bounds_minx",
                    "value": minx,
                    "status": "INFO",
                    "note": "",
                },
                {
                    "metric": "bounds_miny",
                    "value": miny,
                    "status": "INFO",
                    "note": "",
                },
                {
                    "metric": "bounds_maxx",
                    "value": maxx,
                    "status": "INFO",
                    "note": "",
                },
                {
                    "metric": "bounds_maxy",
                    "value": maxy,
                    "status": "INFO",
                    "note": "",
                },
                {
                    "metric": "csv_grids_missing_geometry",
                    "value": csv_grids_missing_geometry,
                    "status": (
                        "PASS"
                        if csv_grids_missing_geometry == 0
                        else "REVIEW_REQUIRED"
                    ),
                    "note": "",
                },
                {
                    "metric": "geometry_grids_missing_csv",
                    "value": geometry_grids_missing_csv,
                    "status": (
                        "PASS"
                        if geometry_grids_missing_csv == 0
                        else "REVIEW_REQUIRED"
                    ),
                    "note": "",
                },
            ]
        )

        print(f"GeoJSON rows                    : {len(grid_gdf):,}")
        print(f"Invalid geometries              : {invalid_geometry_count:,}")
        print(f"Empty geometries                : {empty_geometry_count:,}")
        print(f"Reported CRS                    : {crs_string}")
        print(f"Bounds                          : {bounds}")
        print(f"CSV grids missing from GeoJSON  : {csv_grids_missing_geometry:,}")
        print(f"GeoJSON grids missing from CSV  : {geometry_grids_missing_csv:,}")
        print(f"CRS metadata suspicion          : {grid_crs_suspect}")

    except Exception as exc:
        warnings.warn(f"Grid GeoJSON quality check failed: {exc}")
        geometry_quality_rows.append(
            {
                "metric": "geojson_read",
                "value": "FAILED",
                "status": "REVIEW_REQUIRED",
                "note": str(exc),
            }
        )

elif not GRID_GEOJSON_PATH.exists():
    print("Grid GeoJSON not found. Geometry-quality check skipped.")
    geometry_quality_rows.append(
        {
            "metric": "geojson_file_present",
            "value": False,
            "status": "REVIEW_REQUIRED",
            "note": str(GRID_GEOJSON_PATH),
        }
    )
else:
    print("GeoPandas unavailable. Geometry-quality check skipped.")
    geometry_quality_rows.append(
        {
            "metric": "geopandas_available",
            "value": False,
            "status": "REVIEW_REQUIRED",
            "note": "Install geopandas to run spatial quality checks.",
        }
    )

geometry_quality = pd.DataFrame(geometry_quality_rows)
geometry_quality.to_csv(GEOMETRY_PATH, index=False)


# =============================================================================
# 16. BOUNDARY KML BASIC QUALITY
# =============================================================================

section("12. KADUWELA BOUNDARY KML BASIC QUALITY")

boundary_rows = []

if BOUNDARY_KML_PATH.exists() and gpd is not None:
    try:
        boundary_gdf = gpd.read_file(BOUNDARY_KML_PATH)

        boundary_invalid = int(
            (~boundary_gdf.geometry.is_valid).sum()
        )
        boundary_empty = int(
            boundary_gdf.geometry.is_empty.sum()
        )

        boundary_rows.extend(
            [
                {
                    "metric": "boundary_feature_count",
                    "value": len(boundary_gdf),
                    "status": "INFO",
                    "note": "",
                },
                {
                    "metric": "boundary_invalid_geometries",
                    "value": boundary_invalid,
                    "status": (
                        "PASS"
                        if boundary_invalid == 0
                        else "REVIEW_REQUIRED"
                    ),
                    "note": "",
                },
                {
                    "metric": "boundary_empty_geometries",
                    "value": boundary_empty,
                    "status": (
                        "PASS"
                        if boundary_empty == 0
                        else "REVIEW_REQUIRED"
                    ),
                    "note": "",
                },
                {
                    "metric": "boundary_reported_crs",
                    "value": str(boundary_gdf.crs),
                    "status": "INFO",
                    "note": "",
                },
                {
                    "metric": "boundary_geometry_types",
                    "value": ", ".join(
                        sorted(
                            boundary_gdf.geometry.geom_type
                            .dropna()
                            .unique()
                            .tolist()
                        )
                    ),
                    "status": "INFO",
                    "note": "",
                },
            ]
        )

        print(f"Boundary features         : {len(boundary_gdf):,}")
        print(f"Invalid boundary geometry : {boundary_invalid:,}")
        print(f"Empty boundary geometry   : {boundary_empty:,}")
        print(f"Boundary CRS              : {boundary_gdf.crs}")

    except Exception as exc:
        warnings.warn(f"Boundary KML quality check failed: {exc}")
        boundary_rows.append(
            {
                "metric": "boundary_read",
                "value": "FAILED",
                "status": "INVESTIGATE",
                "note": str(exc),
            }
        )
else:
    boundary_rows.append(
        {
            "metric": "boundary_file_present",
            "value": BOUNDARY_KML_PATH.exists(),
            "status": "INVESTIGATE",
            "note": (
                "Boundary check skipped because file or GeoPandas is unavailable."
            ),
        }
    )

boundary_quality = pd.DataFrame(boundary_rows)
boundary_quality.to_csv(BOUNDARY_PATH, index=False)


# =============================================================================
# 17. FIGURE - 2017 LOW-LST GRID PROFILES
# =============================================================================

section("13. GENERATING DATA-QUALITY FIGURES")

if anomalous_grid_ids:
    lowest_2017_ids = (
        df[
            (df["year"] == 2017)
            & df["grid_id"].astype(str).isin(anomalous_grid_ids)
        ]
        .nsmallest(10, "LST_C")["grid_id"]
        .astype(str)
        .tolist()
    )

    profile = df[
        df["grid_id"].astype(str).isin(lowest_2017_ids)
    ][["grid_id", "year", "LST_C"]].copy()

    plt.figure(figsize=(11, 7))

    for grid_id, group in profile.groupby("grid_id"):
        group = group.sort_values("year")
        plt.plot(
            group["year"],
            group["LST_C"],
            marker="o",
            label=str(grid_id),
        )

    plt.axvline(2017, linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Land Surface Temperature (°C)")
    plt.title("LST Profiles of the 10 Lowest-LST Grids in 2017")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.2)
    save_figure(ANOMALY_PROFILE_FIG)

    print(f"Saved: {ANOMALY_PROFILE_FIG}")
else:
    print("No 2017 low-LST anomaly profile figure created.")


# =============================================================================
# 18. FIGURE - PERSISTENT HOTSPOT PROFILES
# =============================================================================

if not persistent_hotspots.empty:
    top_hot_ids = (
        persistent_hotspots.head(5)["grid_id"]
        .astype(str)
        .tolist()
    )

    hot_profile = df[
        df["grid_id"].astype(str).isin(top_hot_ids)
    ][["grid_id", "year", "LST_C"]].copy()

    plt.figure(figsize=(11, 7))

    for grid_id, group in hot_profile.groupby("grid_id"):
        group = group.sort_values("year")
        plt.plot(
            group["year"],
            group["LST_C"],
            marker="o",
            label=str(grid_id),
        )

    plt.xlabel("Year")
    plt.ylabel("Land Surface Temperature (°C)")
    plt.title("LST Profiles of Top Persistent-Hotspot Candidates")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    save_figure(HOTSPOT_PROFILE_FIG)

    print(f"Saved: {HOTSPOT_PROFILE_FIG}")
else:
    print("No persistent-hotspot profile figure created.")


# =============================================================================
# 19. BUILD QUALITY SUMMARY
# =============================================================================

section("14. DATA QUALITY SUMMARY")

domain_violation_total = int(
    pd.to_numeric(
        domain_checks["violation_count"],
        errors="coerce",
    )
    .fillna(0)
    .sum()
)

numeric_parse_failure_total = int(
    pd.to_numeric(
        schema_quality["numeric_parse_failures"],
        errors="coerce",
    )
    .fillna(0)
    .sum()
)

critical_structural_issue_count = (
    len(missing_expected_columns)
    + total_missing
    + total_positive_inf
    + total_negative_inf
    + blank_grid_ids
    + (duplicate_grid_year_rows or 0)
    + csv_grids_missing_geometry
)

investigation_flags = []

if lcz_fractional_count > 0:
    investigation_flags.append(
        "LCZ contains fractional grid-aggregated values."
    )

if anomalous_grid_ids:
    investigation_flags.append(
        f"2017 has {len(anomalous_grid_ids)} low-LST IQR candidate grids."
    )

if not large_jumps.empty:
    investigation_flags.append(
        f"{len(large_jumps)} grid-year rows are in the top 1% of absolute LST jumps."
    )

if grid_crs_suspect:
    investigation_flags.append(
        "Grid GeoJSON CRS metadata appears inconsistent with coordinate magnitudes."
    )

if not persistent_hotspots.empty:
    investigation_flags.append(
        f"{len(persistent_hotspots)} persistent-hotspot candidates were identified."
    )

quality_status = (
    "PASS_WITH_INVESTIGATION_FLAGS"
    if critical_structural_issue_count == 0
    else "REVIEW_REQUIRED"
)

quality_summary = pd.DataFrame(
    [
        {
            "overall_status": quality_status,
            "rows": len(df),
            "columns": len(df.columns),
            "missing_expected_columns": len(missing_expected_columns),
            "total_missing_cells": total_missing,
            "numeric_parse_failures": numeric_parse_failure_total,
            "positive_infinity_values": total_positive_inf,
            "negative_infinity_values": total_negative_inf,
            "full_duplicate_rows": full_duplicates,
            "duplicate_grid_year_rows": duplicate_grid_year_rows,
            "domain_range_violation_total": domain_violation_total,
            "lcz_fractional_count": lcz_fractional_count,
            "low_2017_anomaly_grid_count": len(anomalous_grid_ids),
            "large_temporal_jump_candidate_count": len(large_jumps),
            "persistent_hotspot_candidate_count": len(persistent_hotspots),
            "csv_grids_missing_geometry": csv_grids_missing_geometry,
            "geometry_grids_missing_csv": geometry_grids_missing_csv,
            "grid_crs_metadata_suspect": grid_crs_suspect,
            "investigation_flag_count": len(investigation_flags),
        }
    ]
)

quality_summary.to_csv(QUALITY_SUMMARY_PATH, index=False)

print(quality_summary.to_string(index=False))

print("\nInvestigation flags:")
if investigation_flags:
    for i, flag in enumerate(investigation_flags, start=1):
        print(f"{i}. {flag}")
else:
    print("None.")


# =============================================================================
# 20. SAVE MAIN TEXT REPORT
# =============================================================================

section("15. SAVING DATA QUALITY REPORT")

static_change_lines = []

if not static_consistency.empty:
    for _, row in static_consistency.iterrows():
        static_change_lines.append(
            f"{row['column']}: grids changing = "
            f"{int(row['grids_with_more_than_one_value'])}"
        )

report_lines = [
    "STEP 03 - DATA QUALITY ANALYSIS REPORT",
    "=" * 90,
    "",
    f"Input file: {RAW_DATA_PATH}",
    f"Rows: {len(df):,}",
    f"Columns: {len(df.columns):,}",
    f"Overall status: {quality_status}",
    "",
    "STRUCTURAL QUALITY",
    "-" * 90,
    f"Missing expected columns: {missing_expected_columns}",
    f"Unexpected columns: {unexpected_columns}",
    f"Total missing cells: {total_missing:,}",
    f"Numeric parse failures: {numeric_parse_failure_total:,}",
    f"Positive infinity values: {total_positive_inf:,}",
    f"Negative infinity values: {total_negative_inf:,}",
    f"Full duplicate rows: {full_duplicates:,}",
    f"Duplicate grid-year rows: {duplicate_grid_year_rows}",
    "",
    "DOMAIN CHECKS",
    "-" * 90,
    f"Total flagged domain/range violations: {domain_violation_total:,}",
    "No value was automatically changed because of a range diagnostic.",
    "",
    "LCZ QUALITY",
    "-" * 90,
    f"Fractional LCZ values: {lcz_fractional_count:,}",
    f"Fractional LCZ percent: {lcz_fractional_percent:.4f}%",
    "Fractional LCZ suggests grid aggregation of categorical LCZ values.",
    "This is a methodological investigation flag, not an automatic data error.",
    "",
    "STATIC FEATURE CONSISTENCY",
    "-" * 90,
    *static_change_lines,
    "",
    "2017 TARGET ANOMALY INVESTIGATION",
    "-" * 90,
    f"2017 low-LST IQR candidate grids: {len(anomalous_grid_ids):,}",
    f"2017 lower IQR fence: {lower_fence_2017}",
    "The affected grids were compared across 2016, 2017 and 2018.",
    "Dynamic satellite features were also compared to determine whether the unusual",
    "LST behaviour is isolated to LST or accompanied by broader feature changes.",
    "",
    "TEMPORAL JUMPS",
    "-" * 90,
    f"99th percentile absolute LST jump threshold: {jump_threshold_99}",
    f"Large temporal jump candidates: {len(large_jumps):,}",
    "",
    "PERSISTENT HOTSPOTS",
    "-" * 90,
    f"Persistent-hotspot candidates: {len(persistent_hotspots):,}",
    "Rule: hottest 1% within year for at least 3 separate years.",
    "",
    "SPATIAL FILE ALIGNMENT",
    "-" * 90,
    f"CSV grids missing geometry: {csv_grids_missing_geometry:,}",
    f"Geometry grids missing CSV data: {geometry_grids_missing_csv:,}",
    f"Invalid grid geometries: {invalid_geometry_count}",
    f"Empty grid geometries: {empty_geometry_count}",
    f"Grid CRS metadata suspect: {grid_crs_suspect}",
    "",
    "INVESTIGATION FLAGS",
    "-" * 90,
]

if investigation_flags:
    report_lines.extend(
        [f"{i}. {flag}" for i, flag in enumerate(investigation_flags, start=1)]
    )
else:
    report_lines.append("None.")

report_lines.extend(
    [
        "",
        "IMPORTANT",
        "-" * 90,
        "Step 03 is diagnostic only.",
        "No raw value was deleted, imputed, capped, scaled, normalized, transformed,",
        "or otherwise modified.",
        "Any corrective action must be justified in Step 04 preprocessing after",
        "reviewing these quality findings.",
    ]
)

MAIN_REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {MAIN_REPORT_PATH}")
print(f"Saved: {QUALITY_SUMMARY_PATH}")
print(f"Saved: {SCHEMA_PATH}")
print(f"Saved: {DOMAIN_PATH}")
print(f"Saved: {STATIC_PATH}")
print(f"Saved: {YEAR_FEATURE_PATH}")
print(f"Saved: {ANOMALY_2017_PATH}")

if not anomaly_area_counts.empty:
    print(f"Saved: {ANOMALY_2017_AREA_PATH}")

if not anomaly_effects.empty:
    print(f"Saved: {ANOMALY_2017_EFFECT_PATH}")

if not anomaly_change_summary.empty:
    print(f"Saved: {ANOMALY_2017_CHANGE_PATH}")

print(f"Saved: {TEMPORAL_JUMPS_PATH}")
print(f"Saved: {HOTSPOT_PATH}")
print(f"Saved: {GEOMETRY_PATH}")
print(f"Saved: {GEOMETRY_ID_PATH}")
print(f"Saved: {BOUNDARY_PATH}")


# =============================================================================
# 21. FINAL STATUS
# =============================================================================

section("STEP 03 COMPLETED SUCCESSFULLY")

print(
    "Data-quality analysis completed without modifying the raw dataset.\n"
    "Review the terminal output and all Step 03 reports before starting Step 04 preprocessing."
)
