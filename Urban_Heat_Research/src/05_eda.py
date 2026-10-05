"""
STEP 05 - EXPLORATORY DATA ANALYSIS (EDA)
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Explore the processed base dataset before feature engineering and modeling.

STRICT MODEL-DEVELOPMENT RULE
-----------------------------
The final year 2025 is treated as the final lockbox test period.

Therefore:
- 2025 remains in the processed base dataset.
- Step 05 reports its row count only for structural confirmation.
- All EDA that could influence modeling decisions uses 2015-2024 only.
- Correlations, distributions, feature-target relationships, area comparisons,
  and spatial heat maps in this step are based on 2015-2024.
- 2025 is NOT used to choose features or model settings.

This step DOES NOT:
- remove rows or outliers
- impute or scale data
- run Mann-Kendall / Sen's slope
- run Moran's I
- perform feature selection
- split/train/tune models
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "kaduwela_preprocessed_base.csv"
GRID_WGS84_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kaduwela_grid_geometry_wgs84.geojson"
)

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MAP_DIR = PROJECT_ROOT / "outputs" / "maps"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "05_eda_report.txt"
DESCRIPTIVE_PATH = REPORT_DIR / "05_descriptive_statistics.csv"
SKEWNESS_PATH = REPORT_DIR / "05_skewness_kurtosis.csv"
YEARLY_PATH = REPORT_DIR / "05_yearly_summary_2015_2024.csv"
AREA_PATH = REPORT_DIR / "05_area_summary_2015_2024.csv"
AREA_YEAR_PATH = REPORT_DIR / "05_area_year_lst_pivot_2015_2024.csv"
PEARSON_PATH = REPORT_DIR / "05_pearson_correlation_matrix.csv"
SPEARMAN_PATH = REPORT_DIR / "05_spearman_correlation_matrix.csv"
TARGET_CORR_PATH = REPORT_DIR / "05_feature_target_correlations.csv"
FEATURE_YEAR_PATH = REPORT_DIR / "05_feature_year_summary_2015_2024.csv"
FEATURE_RANGE_PATH = REPORT_DIR / "05_feature_range_summary.csv"
LOCKBOX_PATH = REPORT_DIR / "05_lockbox_status.csv"

TARGET_HIST_PATH = FIGURE_DIR / "05_lst_distribution_2015_2024.png"
TARGET_BOXPLOT_PATH = FIGURE_DIR / "05_lst_boxplot_2015_2024.png"
YEAR_BOXPLOT_PATH = FIGURE_DIR / "05_lst_by_year_boxplot_2015_2024.png"
YEAR_MEAN_PATH = FIGURE_DIR / "05_lst_yearly_mean_median_2015_2024.png"
YEAR_STD_PATH = FIGURE_DIR / "05_lst_yearly_variability_2015_2024.png"
CORR_HEATMAP_PATH = FIGURE_DIR / "05_pearson_correlation_heatmap.png"
TARGET_CORR_BAR_PATH = FIGURE_DIR / "05_feature_target_correlation_bar.png"
AREA_TOP_PATH = FIGURE_DIR / "05_hottest_areas_mean_lst.png"
AREA_BOTTOM_PATH = FIGURE_DIR / "05_coolest_areas_mean_lst.png"
LCZ_DIST_PATH = FIGURE_DIR / "05_lcz_distribution.png"
MAP_MEAN_LST_PATH = MAP_DIR / "05_spatial_mean_lst_2015_2024.png"
MAP_2024_LST_PATH = MAP_DIR / "05_spatial_lst_2024.png"


# =============================================================================
# 2. COLUMN GROUPS
# =============================================================================

TARGET = "LST_C"

NUMERIC_FEATURES = [
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

CONTINUOUS_FEATURES = [
    "NDVI",
    "NDBI",
    "NDWI",
    "EVI",
    "Albedo",
    "NDBI_ADJ",
    "NDVI_ADJ",
    "NightLights",
    "green_mask",
    "dist_road",
    "dist_main_road",
    "dist_water",
    "LCZ",
]

BINARY_FEATURES = [
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


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def safe_filename(text: str) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


# =============================================================================
# 4. LOAD DATA
# =============================================================================

section("STEP 05 - EXPLORATORY DATA ANALYSIS")

print(f"Project root     : {PROJECT_ROOT}")
print(f"Processed input : {DATA_PATH}")

if not DATA_PATH.exists():
    print("\nERROR: Processed base dataset was not found.")
    print("Run Step 04 first.")
    sys.exit(1)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
    dtype={"grid_id": "string", "area_name": "string"},
)

if df.empty:
    print("\nERROR: Processed dataset is empty.")
    sys.exit(1)

required = {"grid_id", "year", TARGET, "area_name"} | set(NUMERIC_FEATURES)
missing_required = sorted(required - set(df.columns))

if missing_required:
    print(f"\nERROR: Missing required columns: {missing_required}")
    sys.exit(1)

print(f"Rows            : {len(df):,}")
print(f"Columns         : {len(df.columns):,}")
print(f"Unique grids    : {df['grid_id'].nunique():,}")
print(f"Years           : {sorted(df['year'].unique().tolist())}")


# =============================================================================
# 5. LOCKBOX PROTECTION
# =============================================================================

section("1. 2025 LOCKBOX PROTECTION")

development_df = df[df["year"] <= 2024].copy()
lockbox_df = df[df["year"] == 2025].copy()

print(f"Development EDA period : 2015-2024")
print(f"Development EDA rows   : {len(development_df):,}")
print(f"2025 lockbox rows      : {len(lockbox_df):,}")
print("2025 target distributions/relationships are NOT explored in Step 05.")

expected_dev_rows = df["grid_id"].nunique() * 10

if len(development_df) != expected_dev_rows:
    warnings.warn(
        f"Expected {expected_dev_rows:,} rows for 2015-2024, "
        f"but found {len(development_df):,}."
    )

lockbox_status = pd.DataFrame(
    [
        {
            "development_period": "2015-2024",
            "development_rows": len(development_df),
            "final_lockbox_year": 2025,
            "lockbox_rows": len(lockbox_df),
            "lockbox_used_in_correlations": False,
            "lockbox_used_in_feature_target_plots": False,
            "lockbox_used_in_area_rankings": False,
            "lockbox_used_in_spatial_eda_maps": False,
        }
    ]
)

lockbox_status.to_csv(LOCKBOX_PATH, index=False)


# =============================================================================
# 6. DESCRIPTIVE STATISTICS
# =============================================================================

section("2. DESCRIPTIVE STATISTICS - 2015 TO 2024")

eda_numeric_cols = ["year", TARGET] + NUMERIC_FEATURES

descriptive = (
    development_df[eda_numeric_cols]
    .describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
    .T
    .reset_index()
    .rename(columns={"index": "variable"})
)

descriptive.to_csv(DESCRIPTIVE_PATH, index=False)

print(
    descriptive[
        ["variable", "count", "mean", "std", "min", "50%", "max"]
    ].to_string(index=False)
)


# =============================================================================
# 7. SKEWNESS / KURTOSIS
# =============================================================================

section("3. SKEWNESS AND KURTOSIS")

shape_rows = []

for col in [TARGET] + NUMERIC_FEATURES:
    s = pd.to_numeric(development_df[col], errors="coerce").dropna()

    shape_rows.append(
        {
            "variable": col,
            "skewness": s.skew(),
            "excess_kurtosis": s.kurt(),
            "absolute_skewness": abs(s.skew()),
            "strong_skew_flag": abs(s.skew()) >= 1.0,
        }
    )

shape_summary = (
    pd.DataFrame(shape_rows)
    .sort_values("absolute_skewness", ascending=False)
)

shape_summary.to_csv(SKEWNESS_PATH, index=False)

print(shape_summary.to_string(index=False))


# =============================================================================
# 8. FEATURE RANGE / UNIQUENESS SUMMARY
# =============================================================================

section("4. FEATURE RANGE AND UNIQUENESS SUMMARY")

range_rows = []

for col in [TARGET] + NUMERIC_FEATURES:
    s = pd.to_numeric(development_df[col], errors="coerce")

    range_rows.append(
        {
            "feature": col,
            "count": int(s.notna().sum()),
            "unique_count": int(s.nunique(dropna=True)),
            "min": s.min(),
            "q01": s.quantile(0.01),
            "q05": s.quantile(0.05),
            "median": s.median(),
            "q95": s.quantile(0.95),
            "q99": s.quantile(0.99),
            "max": s.max(),
            "zero_count": int((s == 0).sum()),
            "zero_percent": 100 * float((s == 0).mean()),
        }
    )

feature_range_summary = pd.DataFrame(range_rows)
feature_range_summary.to_csv(FEATURE_RANGE_PATH, index=False)

print(feature_range_summary.to_string(index=False))


# =============================================================================
# 9. YEARLY TARGET SUMMARY
# =============================================================================

section("5. YEARLY LST SUMMARY")

yearly_summary = (
    development_df.groupby("year")[TARGET]
    .agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
    )
    .reset_index()
)

yearly_summary["q05"] = (
    development_df.groupby("year")[TARGET]
    .quantile(0.05)
    .values
)

yearly_summary["q25"] = (
    development_df.groupby("year")[TARGET]
    .quantile(0.25)
    .values
)

yearly_summary["q75"] = (
    development_df.groupby("year")[TARGET]
    .quantile(0.75)
    .values
)

yearly_summary["q95"] = (
    development_df.groupby("year")[TARGET]
    .quantile(0.95)
    .values
)

yearly_summary.to_csv(YEARLY_PATH, index=False)

print(yearly_summary.to_string(index=False))


# =============================================================================
# 10. YEAR-WISE FEATURE SUMMARY
# =============================================================================

section("6. YEAR-WISE FEATURE SUMMARY")

feature_year_rows = []

for year, group in development_df.groupby("year", sort=True):
    for col in [TARGET] + NUMERIC_FEATURES:
        s = pd.to_numeric(group[col], errors="coerce")

        feature_year_rows.append(
            {
                "year": year,
                "feature": col,
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

feature_year_summary = pd.DataFrame(feature_year_rows)
feature_year_summary.to_csv(FEATURE_YEAR_PATH, index=False)

print(f"Saved detailed year-wise feature summary: {FEATURE_YEAR_PATH}")


# =============================================================================
# 11. AREA-LEVEL EDA
# =============================================================================

section("7. AREA-LEVEL LST SUMMARY")

area_summary = (
    development_df.groupby("area_name")
    .agg(
        row_count=(TARGET, "size"),
        unique_grids=("grid_id", "nunique"),
        mean_lst=(TARGET, "mean"),
        median_lst=(TARGET, "median"),
        std_lst=(TARGET, "std"),
        min_lst=(TARGET, "min"),
        max_lst=(TARGET, "max"),
    )
    .reset_index()
    .sort_values("mean_lst", ascending=False)
)

area_summary.to_csv(AREA_PATH, index=False)

area_year_pivot = development_df.pivot_table(
    index="area_name",
    columns="year",
    values=TARGET,
    aggfunc="mean",
)

area_year_pivot.to_csv(AREA_YEAR_PATH)

print("Top 10 areas by mean LST (2015-2024):")
print(area_summary.head(10).to_string(index=False))

print("\nBottom 10 areas by mean LST (2015-2024):")
print(
    area_summary.tail(10)
    .sort_values("mean_lst")
    .to_string(index=False)
)


# =============================================================================
# 12. CORRELATION ANALYSIS
# =============================================================================

section("8. CORRELATION ANALYSIS")

corr_columns = [TARGET, "year"] + NUMERIC_FEATURES

pearson_corr = development_df[corr_columns].corr(method="pearson")
spearman_corr = development_df[corr_columns].corr(method="spearman")

pearson_corr.to_csv(PEARSON_PATH)
spearman_corr.to_csv(SPEARMAN_PATH)

target_corr_rows = []

for feature in ["year"] + NUMERIC_FEATURES:
    target_corr_rows.append(
        {
            "feature": feature,
            "pearson_with_lst": pearson_corr.loc[feature, TARGET],
            "spearman_with_lst": spearman_corr.loc[feature, TARGET],
            "abs_pearson": abs(pearson_corr.loc[feature, TARGET]),
            "abs_spearman": abs(spearman_corr.loc[feature, TARGET]),
        }
    )

target_correlations = (
    pd.DataFrame(target_corr_rows)
    .sort_values("abs_spearman", ascending=False)
)

target_correlations.to_csv(TARGET_CORR_PATH, index=False)

print(
    target_correlations[
        ["feature", "pearson_with_lst", "spearman_with_lst"]
    ].to_string(index=False)
)

print(
    "\nNOTE: Correlation does not prove causation and does not by itself "
    "determine feature selection."
)


# =============================================================================
# 13. TARGET DISTRIBUTION FIGURES
# =============================================================================

section("9. TARGET DISTRIBUTION FIGURES")

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(development_df[TARGET], bins=55)
plt.xlabel("Land Surface Temperature (°C)")
plt.ylabel("Frequency")
plt.title("LST_C Distribution - Development Period (2015-2024)")
plt.grid(alpha=0.2)
save_figure(TARGET_HIST_PATH)

# Overall boxplot
plt.figure(figsize=(9, 5))
plt.boxplot(
    development_df[TARGET].dropna(),
    orientation="horizontal",
)
plt.xlabel("Land Surface Temperature (°C)")
plt.title("LST_C Boxplot - Development Period (2015-2024)")
plt.grid(axis="x", alpha=0.2)
save_figure(TARGET_BOXPLOT_PATH)

# Year-wise boxplot
years = sorted(development_df["year"].unique().tolist())
year_values = [
    development_df.loc[
        development_df["year"] == year,
        TARGET,
    ].dropna().values
    for year in years
]

plt.figure(figsize=(12, 6))
plt.boxplot(
    year_values,
    tick_labels=[str(y) for y in years],
)
plt.xlabel("Year")
plt.ylabel("Land Surface Temperature (°C)")
plt.title("Year-wise LST_C Distribution - 2015-2024")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.2)
save_figure(YEAR_BOXPLOT_PATH)

# Yearly mean/median
plt.figure(figsize=(10, 6))
plt.plot(
    yearly_summary["year"],
    yearly_summary["mean"],
    marker="o",
    label="Mean",
)
plt.plot(
    yearly_summary["year"],
    yearly_summary["median"],
    marker="o",
    label="Median",
)
plt.xlabel("Year")
plt.ylabel("Land Surface Temperature (°C)")
plt.title("Yearly Mean and Median LST_C - 2015-2024")
plt.legend()
plt.grid(alpha=0.2)
save_figure(YEAR_MEAN_PATH)

# Yearly variability
plt.figure(figsize=(10, 6))
plt.plot(
    yearly_summary["year"],
    yearly_summary["std"],
    marker="o",
)
plt.xlabel("Year")
plt.ylabel("LST Standard Deviation (°C)")
plt.title("Yearly Spatial Variability of LST_C - 2015-2024")
plt.grid(alpha=0.2)
save_figure(YEAR_STD_PATH)

print(f"Saved: {TARGET_HIST_PATH}")
print(f"Saved: {TARGET_BOXPLOT_PATH}")
print(f"Saved: {YEAR_BOXPLOT_PATH}")
print(f"Saved: {YEAR_MEAN_PATH}")
print(f"Saved: {YEAR_STD_PATH}")


# =============================================================================
# 14. CORRELATION FIGURES
# =============================================================================

section("10. CORRELATION FIGURES")

# Pearson correlation matrix heatmap using matplotlib only.
matrix = pearson_corr.to_numpy()
labels = pearson_corr.columns.tolist()

plt.figure(figsize=(13, 11))
image = plt.imshow(
    matrix,
    aspect="auto",
    vmin=-1,
    vmax=1,
)
plt.colorbar(image, label="Pearson correlation")
plt.xticks(
    np.arange(len(labels)),
    labels,
    rotation=90,
    fontsize=8,
)
plt.yticks(
    np.arange(len(labels)),
    labels,
    fontsize=8,
)
plt.title("Pearson Correlation Matrix - 2015-2024")
save_figure(CORR_HEATMAP_PATH)

# Target correlation bar plot
ordered_corr = target_correlations.sort_values(
    "pearson_with_lst",
    ascending=True,
)

plt.figure(figsize=(10, 8))
plt.barh(
    ordered_corr["feature"],
    ordered_corr["pearson_with_lst"],
)
plt.xlabel("Pearson Correlation with LST_C")
plt.ylabel("Feature")
plt.title("Feature Correlation with LST_C - 2015-2024")
plt.axvline(0, linewidth=1)
plt.grid(axis="x", alpha=0.2)
save_figure(TARGET_CORR_BAR_PATH)

print(f"Saved: {CORR_HEATMAP_PATH}")
print(f"Saved: {TARGET_CORR_BAR_PATH}")


# =============================================================================
# 15. FEATURE DISTRIBUTION FIGURES
# =============================================================================

section("11. FEATURE DISTRIBUTION FIGURES")

feature_distribution_paths = []

for feature in NUMERIC_FEATURES:
    out_path = FIGURE_DIR / f"05_distribution_{safe_filename(feature)}.png"

    plt.figure(figsize=(9, 6))
    plt.hist(
        development_df[feature].dropna(),
        bins=45,
    )
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature} - 2015-2024")
    plt.grid(alpha=0.2)
    save_figure(out_path)

    feature_distribution_paths.append(out_path)

print(
    f"Saved {len(feature_distribution_paths)} feature-distribution figures."
)


# =============================================================================
# 16. FEATURE-vs-LST RELATIONSHIP FIGURES
# =============================================================================

section("12. FEATURE-vs-LST RELATIONSHIP FIGURES")

relationship_paths = []

# Hexbin is used for continuous variables because 21,940 observations would
# heavily overlap in ordinary scatter plots.
for feature in CONTINUOUS_FEATURES:
    out_path = (
        FIGURE_DIR
        / f"05_relationship_{safe_filename(feature)}_vs_LST_C.png"
    )

    x = development_df[feature]
    y = development_df[TARGET]

    valid = x.notna() & y.notna()

    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(
        x[valid],
        y[valid],
        gridsize=45,
        mincnt=1,
    )
    plt.colorbar(hb, label="Observation count")
    plt.xlabel(feature)
    plt.ylabel("LST_C (°C)")
    plt.title(f"{feature} vs LST_C - 2015-2024")
    plt.grid(alpha=0.15)
    save_figure(out_path)

    relationship_paths.append(out_path)

# Binary masks: use boxplots instead of scatter.
for feature in BINARY_FEATURES:
    out_path = (
        FIGURE_DIR
        / f"05_relationship_{safe_filename(feature)}_vs_LST_C.png"
    )

    values_0 = development_df.loc[
        development_df[feature] == 0,
        TARGET,
    ].dropna()

    values_1 = development_df.loc[
        development_df[feature] == 1,
        TARGET,
    ].dropna()

    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [values_0, values_1],
        tick_labels=["0", "1"],
    )
    plt.xlabel(feature)
    plt.ylabel("LST_C (°C)")
    plt.title(f"{feature} vs LST_C - 2015-2024")
    plt.grid(axis="y", alpha=0.2)
    save_figure(out_path)

    relationship_paths.append(out_path)

print(
    f"Saved {len(relationship_paths)} feature-target relationship figures."
)


# =============================================================================
# 17. AREA FIGURES
# =============================================================================

section("13. AREA-LEVEL FIGURES")

top_15 = area_summary.head(15).sort_values("mean_lst")
bottom_15 = area_summary.tail(15).sort_values("mean_lst", ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(
    top_15["area_name"],
    top_15["mean_lst"],
)
plt.xlabel("Mean LST_C (°C)")
plt.ylabel("Area")
plt.title("15 Highest Mean-LST Areas - 2015-2024")
plt.grid(axis="x", alpha=0.2)
save_figure(AREA_TOP_PATH)

plt.figure(figsize=(10, 8))
plt.barh(
    bottom_15["area_name"],
    bottom_15["mean_lst"],
)
plt.xlabel("Mean LST_C (°C)")
plt.ylabel("Area")
plt.title("15 Lowest Mean-LST Areas - 2015-2024")
plt.grid(axis="x", alpha=0.2)
save_figure(AREA_BOTTOM_PATH)

print(f"Saved: {AREA_TOP_PATH}")
print(f"Saved: {AREA_BOTTOM_PATH}")


# =============================================================================
# 18. LCZ EDA - NO RECLASSIFICATION
# =============================================================================

section("14. LCZ EXPLORATORY VIEW")

plt.figure(figsize=(9, 6))
plt.hist(
    development_df["LCZ"].dropna(),
    bins=50,
)
plt.xlabel("LCZ aggregated value")
plt.ylabel("Frequency")
plt.title("Distribution of Aggregated LCZ Values - 2015-2024")
plt.grid(alpha=0.2)
save_figure(LCZ_DIST_PATH)

fractional_lcz = int(
    (
        development_df["LCZ"]
        .sub(development_df["LCZ"].round())
        .abs()
        > 1e-9
    ).sum()
)

print(f"Fractional LCZ rows in development period: {fractional_lcz:,}")
print("LCZ remains unchanged; this figure is descriptive only.")
print(f"Saved: {LCZ_DIST_PATH}")


# =============================================================================
# 19. 2017 VISUAL DIAGNOSTIC
# =============================================================================

section("15. 2017 VISUAL DIAGNOSTIC")

# We already know from Step 03 that 2017 is unusual. Here we visualize the
# same development-period feature means without changing any observation.
selected_2017_features = [
    "LST_C",
    "NDVI",
    "NDWI",
    "EVI",
    "Albedo",
    "NDVI_ADJ",
    "green_mask",
]

for feature in selected_2017_features:
    yearly_feature = (
        development_df.groupby("year")[feature]
        .mean()
        .reset_index()
    )

    out_path = (
        FIGURE_DIR
        / f"05_yearly_mean_{safe_filename(feature)}_2015_2024.png"
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        yearly_feature["year"],
        yearly_feature[feature],
        marker="o",
    )
    plt.axvline(2017, linestyle="--")
    plt.xlabel("Year")
    plt.ylabel(feature)
    plt.title(f"Yearly Mean {feature} - 2015-2024")
    plt.grid(alpha=0.2)
    save_figure(out_path)

print(
    f"Saved yearly diagnostic plots for {len(selected_2017_features)} "
    "features, with 2017 marked."
)


# =============================================================================
# 20. SPATIAL EDA MAPS - DESCRIPTIVE ONLY
# =============================================================================

section("16. SPATIAL EDA MAPS")

spatial_maps_created = 0

if GRID_WGS84_PATH.exists():
    try:
        grid = gpd.read_file(GRID_WGS84_PATH)

        if "grid_id" not in grid.columns:
            raise ValueError("Processed grid geometry has no grid_id column.")

        grid = grid.copy()
        grid["grid_id"] = grid["grid_id"].astype(str).str.strip()

        # Multi-year mean LST per grid: development period only
        grid_mean_lst = (
            development_df.groupby("grid_id")[TARGET]
            .mean()
            .rename("mean_lst_2015_2024")
            .reset_index()
        )
        grid_mean_lst["grid_id"] = grid_mean_lst["grid_id"].astype(str)

        spatial_mean = grid.merge(
            grid_mean_lst,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )

        if spatial_mean["mean_lst_2015_2024"].isna().any():
            raise ValueError(
                "Some grid geometries did not receive mean LST values."
            )

        plt.figure(figsize=(10, 10))
        ax = spatial_mean.plot(
            column="mean_lst_2015_2024",
            legend=True,
            figsize=(10, 10),
        )
        ax.set_axis_off()
        ax.set_title("Mean LST_C by Grid - 2015-2024")
        plt.savefig(
            MAP_MEAN_LST_PATH,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        spatial_maps_created += 1

        # Latest development year map = 2024, NOT 2025.
        lst_2024 = development_df[
            development_df["year"] == 2024
        ][["grid_id", TARGET]].copy()

        lst_2024["grid_id"] = lst_2024["grid_id"].astype(str)

        spatial_2024 = grid.merge(
            lst_2024,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )

        if spatial_2024[TARGET].isna().any():
            raise ValueError(
                "Some grid geometries did not receive 2024 LST values."
            )

        plt.figure(figsize=(10, 10))
        ax = spatial_2024.plot(
            column=TARGET,
            legend=True,
            figsize=(10, 10),
        )
        ax.set_axis_off()
        ax.set_title("LST_C by Grid - 2024")
        plt.savefig(
            MAP_2024_LST_PATH,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        spatial_maps_created += 1

        print(f"Saved: {MAP_MEAN_LST_PATH}")
        print(f"Saved: {MAP_2024_LST_PATH}")

    except Exception as exc:
        warnings.warn(f"Spatial EDA map generation failed: {exc}")
else:
    warnings.warn(
        f"Processed WGS84 grid not found: {GRID_WGS84_PATH}"
    )

print(
    "NOTE: These are descriptive maps only. "
    "Formal spatial autocorrelation (Moran's I) belongs in Step 06."
)


# =============================================================================
# 21. TEXT REPORT
# =============================================================================

section("17. SAVE EDA REPORT")

top_positive = (
    target_correlations
    .sort_values("pearson_with_lst", ascending=False)
    .head(5)
)

top_negative = (
    target_correlations
    .sort_values("pearson_with_lst", ascending=True)
    .head(5)
)

strong_skew = shape_summary[
    shape_summary["strong_skew_flag"]
]["variable"].tolist()

report_lines = [
    "STEP 05 - EXPLORATORY DATA ANALYSIS REPORT",
    "=" * 90,
    "",
    f"Input dataset: {DATA_PATH}",
    "",
    "LOCKBOX POLICY",
    "-" * 90,
    "Model-development EDA period: 2015-2024",
    f"Development rows: {len(development_df):,}",
    "Final lockbox year: 2025",
    f"2025 rows held out from modeling-oriented EDA: {len(lockbox_df):,}",
    "",
    "TARGET SUMMARY - 2015-2024",
    "-" * 90,
    f"Mean LST_C: {development_df[TARGET].mean():.6f}",
    f"Median LST_C: {development_df[TARGET].median():.6f}",
    f"Std LST_C: {development_df[TARGET].std():.6f}",
    f"Minimum LST_C: {development_df[TARGET].min():.6f}",
    f"Maximum LST_C: {development_df[TARGET].max():.6f}",
    "",
    "STRONGLY SKEWED VARIABLES (|skew| >= 1)",
    "-" * 90,
    ", ".join(strong_skew) if strong_skew else "None",
    "",
    "TOP POSITIVE PEARSON RELATIONSHIPS WITH LST_C",
    "-" * 90,
]

for _, row in top_positive.iterrows():
    report_lines.append(
        f"{row['feature']}: {row['pearson_with_lst']:.6f}"
    )

report_lines.extend(
    [
        "",
        "TOP NEGATIVE PEARSON RELATIONSHIPS WITH LST_C",
        "-" * 90,
    ]
)

for _, row in top_negative.iterrows():
    report_lines.append(
        f"{row['feature']}: {row['pearson_with_lst']:.6f}"
    )

report_lines.extend(
    [
        "",
        "AREA EDA",
        "-" * 90,
        f"Number of areas: {development_df['area_name'].nunique():,}",
        f"Highest mean-LST area: {area_summary.iloc[0]['area_name']}",
        f"Highest area mean LST: {area_summary.iloc[0]['mean_lst']:.6f}",
        f"Lowest mean-LST area: {area_summary.iloc[-1]['area_name']}",
        f"Lowest area mean LST: {area_summary.iloc[-1]['mean_lst']:.6f}",
        "",
        "LCZ",
        "-" * 90,
        f"Fractional LCZ rows in 2015-2024: {fractional_lcz:,}",
        "LCZ was not rounded or reclassified in Step 05.",
        "",
        "2017",
        "-" * 90,
        "2017 remains in the development dataset.",
        "It is visualized and documented but not removed.",
        "Formal robustness/sensitivity testing will be done later.",
        "",
        "SPATIAL EDA",
        "-" * 90,
        f"Spatial maps created: {spatial_maps_created}",
        "Moran's I was NOT run in Step 05.",
        "",
        "IMPORTANT INTERPRETATION RULES",
        "-" * 90,
        "Correlation does not prove causation.",
        "EDA relationships do not automatically determine feature selection.",
        "No row, outlier, or feature was removed in Step 05.",
        "No train/test split or model fitting was performed.",
        "Mann-Kendall and Sen's slope are reserved for the formal trend-analysis stage.",
        "Moran's I is reserved for Step 06 spatial-temporal analysis.",
    ]
)

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {REPORT_PATH}")
print(f"Saved: {DESCRIPTIVE_PATH}")
print(f"Saved: {SKEWNESS_PATH}")
print(f"Saved: {YEARLY_PATH}")
print(f"Saved: {AREA_PATH}")
print(f"Saved: {AREA_YEAR_PATH}")
print(f"Saved: {PEARSON_PATH}")
print(f"Saved: {SPEARMAN_PATH}")
print(f"Saved: {TARGET_CORR_PATH}")
print(f"Saved: {FEATURE_YEAR_PATH}")
print(f"Saved: {FEATURE_RANGE_PATH}")
print(f"Saved: {LOCKBOX_PATH}")


# =============================================================================
# 22. FINAL STATUS
# =============================================================================

section("STEP 05 COMPLETED SUCCESSFULLY")

print(
    "Exploratory Data Analysis completed using the development period 2015-2024.\n"
    "The 2025 lockbox was not used for modeling-oriented EDA.\n"
    "Review the terminal output, reports, figures, and maps before Step 06."
)
