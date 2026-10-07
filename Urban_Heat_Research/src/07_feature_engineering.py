"""
STEP 07 - FEATURE ENGINEERING
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Create scientifically meaningful, leakage-aware features for later model
development while preserving the 2025 final lockbox.

FEATURE-ENGINEERING PRINCIPLES
------------------------------
1. No feature is created from the current/future LST target.
2. No same-year neighbor LST is used.
3. Temporal features are created only from predictor history.
4. Historical rolling features use shifted values, so they never look forward.
5. Spatial features use independent environmental predictors only.
6. No scaler, imputer, PCA, feature selector, or model is fitted here.
7. 2025 features may be deterministically generated from predictor data, but
   the 2025 target is never inspected or used to make feature decisions.
8. Structural missing values created by lags are preserved for later
   train-only preprocessing.

Outputs
-------
data/processed/kaduwela_engineered_features.csv

Reports
-------
07_feature_engineering_report.txt
07_feature_dictionary.csv
07_engineered_missingness.csv
07_engineered_missingness_by_year.csv
07_leakage_audit.csv
07_feature_count_summary.csv
07_spatial_feature_audit.csv
07_temporal_feature_audit.csv
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

from libpysal.weights import Queen, lag_spatial


# =============================================================================
# 1. SETTINGS
# =============================================================================

DEVELOPMENT_START_YEAR = 2015
DEVELOPMENT_END_YEAR = 2024
LOCKBOX_YEAR = 2025

TARGET = "LST_C"

# Dynamic environmental predictors suitable for temporal history features.
TEMPORAL_SOURCE_FEATURES = [
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

# Same-year neighborhood features are created only from non-target predictors.
SPATIAL_SOURCE_FEATURES = [
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

STATIC_DISTANCE_FEATURES = [
    "dist_road",
    "dist_main_road",
    "dist_water",
]


# =============================================================================
# 2. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kaduwela_preprocessed_base.csv"
)

GRID_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kaduwela_grid_geometry_utm44n.geojson"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kaduwela_engineered_features.csv"
)

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_REPORT = REPORT_DIR / "07_feature_engineering_report.txt"
FEATURE_DICTIONARY_PATH = REPORT_DIR / "07_feature_dictionary.csv"
MISSINGNESS_PATH = REPORT_DIR / "07_engineered_missingness.csv"
MISSINGNESS_BY_YEAR_PATH = REPORT_DIR / "07_engineered_missingness_by_year.csv"
LEAKAGE_AUDIT_PATH = REPORT_DIR / "07_leakage_audit.csv"
COUNT_SUMMARY_PATH = REPORT_DIR / "07_feature_count_summary.csv"
SPATIAL_AUDIT_PATH = REPORT_DIR / "07_spatial_feature_audit.csv"
TEMPORAL_AUDIT_PATH = REPORT_DIR / "07_temporal_feature_audit.csv"


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112)


def add_dictionary_row(
    rows,
    feature,
    category,
    source_features,
    formula,
    leakage_status,
    availability_note,
):
    rows.append(
        {
            "feature": feature,
            "category": category,
            "source_features": source_features,
            "formula_or_definition": formula,
            "leakage_status": leakage_status,
            "availability_note": availability_note,
        }
    )


def structural_missing_reason(feature: str) -> str:
    if "_lag1" in feature:
        return "Expected for first year of each grid because no previous-year predictor exists."
    if "_lag2" in feature:
        return "Expected for first two years of each grid."
    if "_hist3_mean" in feature:
        return "Expected for first year; later rows use only available prior predictor history."
    if "_hist3_std" in feature:
        return "Expected until at least two prior predictor observations exist."
    if "_hist3_slope" in feature:
        return "Expected until three prior predictor observations exist."
    if "_delta1" in feature:
        return "Expected for first year because previous-year predictor is unavailable."
    return "Not expected unless source data are missing."


# =============================================================================
# 4. LOAD INPUTS
# =============================================================================

section("STEP 07 - FEATURE ENGINEERING")

print(f"Project root : {PROJECT_ROOT}")
print(f"Input data   : {INPUT_PATH}")
print(f"Grid geometry: {GRID_PATH}")

if not INPUT_PATH.exists():
    print("\nERROR: Step 04 processed base dataset not found.")
    sys.exit(1)

if not GRID_PATH.exists():
    print("\nERROR: Corrected UTM grid geometry not found.")
    sys.exit(1)

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False,
    dtype={"grid_id": "string", "area_name": "string"},
)

grid = gpd.read_file(GRID_PATH)

if df.empty or grid.empty:
    print("\nERROR: Input data or geometry is empty.")
    sys.exit(1)

required_columns = {
    "grid_id",
    "year",
    TARGET,
    "area_name",
    "building_mask",
    "road_mask",
    "LCZ",
    *TEMPORAL_SOURCE_FEATURES,
    *STATIC_DISTANCE_FEATURES,
}

missing_required = sorted(required_columns - set(df.columns))

if missing_required:
    print(f"\nERROR: Missing required columns: {missing_required}")
    sys.exit(1)

df = df.copy()
grid = grid.copy()

df["grid_id"] = df["grid_id"].astype(str).str.strip()
grid["grid_id"] = grid["grid_id"].astype(str).str.strip()

print(f"Rows before engineering   : {len(df):,}")
print(f"Columns before engineering: {len(df.columns):,}")
print(f"Unique grids              : {df['grid_id'].nunique():,}")
print(f"Years                     : {sorted(df['year'].unique().tolist())}")


# =============================================================================
# 5. LOCKBOX / TARGET-LEAKAGE SAFEGUARD
# =============================================================================

section("1. LOCKBOX AND TARGET-LEAKAGE SAFEGUARDS")

development_rows = int(
    df["year"].between(
        DEVELOPMENT_START_YEAR,
        DEVELOPMENT_END_YEAR,
    ).sum()
)

lockbox_rows = int((df["year"] == LOCKBOX_YEAR).sum())

print(f"Development period rows : {development_rows:,}")
print(f"2025 lockbox rows       : {lockbox_rows:,}")
print(
    "The 2025 target is preserved in the file but is NOT used to engineer "
    "features or make feature decisions."
)

original_target = df[["grid_id", "year", TARGET]].copy()

feature_dictionary = []

# Original columns are retained. Mark key roles clearly.
for col in df.columns:
    if col == "grid_id":
        category = "identifier"
        leakage = "exclude_from_model"
        note = "Grid identifier; retain for joins only."
    elif col == "area_name":
        category = "reporting_group"
        leakage = "exclude_from_model_by_default"
        note = "GN/area label retained for reporting; do not encode automatically."
    elif col == TARGET:
        category = "target"
        leakage = "target_only"
        note = "Prediction target; never used to create engineered predictors in Step 07."
    elif col == "year":
        category = "time"
        leakage = "safe_deterministic"
        note = "Calendar year."
    else:
        category = "original_predictor"
        leakage = "candidate_predictor"
        note = "Original processed predictor."

    add_dictionary_row(
        feature_dictionary,
        col,
        category,
        col,
        "Original Step 04 column",
        leakage,
        note,
    )


# =============================================================================
# 6. DETERMINISTIC ROW-WISE FEATURES
# =============================================================================

section("2. DETERMINISTIC ROW-WISE FEATURES")

engineered = df.copy()

# Time index - deterministic and does not learn anything from data.
engineered["year_index"] = (
    engineered["year"] - DEVELOPMENT_START_YEAR
)

add_dictionary_row(
    feature_dictionary,
    "year_index",
    "temporal_deterministic",
    "year",
    f"year - {DEVELOPMENT_START_YEAR}",
    "safe_deterministic",
    "Available for all rows including 2025 without using target information.",
)

# Vegetation / built-form interactions.
engineered["vegetation_deficit"] = 1.0 - engineered["green_mask"]

add_dictionary_row(
    feature_dictionary,
    "vegetation_deficit",
    "environmental_derived",
    "green_mask",
    "1 - green_mask",
    "safe_deterministic",
    "Higher values indicate lower grid-level green proportion.",
)

engineered["ndbi_minus_ndvi"] = (
    engineered["NDBI"] - engineered["NDVI"]
)

add_dictionary_row(
    feature_dictionary,
    "ndbi_minus_ndvi",
    "environmental_interaction",
    "NDBI, NDVI",
    "NDBI - NDVI",
    "safe_deterministic",
    "Contrast feature only; not treated as a named established index.",
)

engineered["ndbi_x_vegetation_deficit"] = (
    engineered["NDBI"]
    * engineered["vegetation_deficit"]
)

add_dictionary_row(
    feature_dictionary,
    "ndbi_x_vegetation_deficit",
    "environmental_interaction",
    "NDBI, vegetation_deficit",
    "NDBI * vegetation_deficit",
    "safe_deterministic",
    "Interaction between built-up spectral signal and vegetation deficit.",
)

engineered["ndvi_x_green_mask"] = (
    engineered["NDVI"]
    * engineered["green_mask"]
)

add_dictionary_row(
    feature_dictionary,
    "ndvi_x_green_mask",
    "environmental_interaction",
    "NDVI, green_mask",
    "NDVI * green_mask",
    "safe_deterministic",
    "Interaction between vegetation index and green proportion.",
)

engineered["ndbi_x_building_mask"] = (
    engineered["NDBI"]
    * engineered["building_mask"]
)

add_dictionary_row(
    feature_dictionary,
    "ndbi_x_building_mask",
    "environmental_interaction",
    "NDBI, building_mask",
    "NDBI * building_mask",
    "safe_deterministic",
    "Built spectral signal conditioned by mapped building presence.",
)

engineered["nightlights_x_building_mask"] = (
    engineered["NightLights"]
    * engineered["building_mask"]
)

add_dictionary_row(
    feature_dictionary,
    "nightlights_x_building_mask",
    "environmental_interaction",
    "NightLights, building_mask",
    "NightLights * building_mask",
    "safe_deterministic",
    "Urban-activity proxy conditioned by building presence.",
)

# Fixed formula transformations: no parameters are fitted from the dataset.
for feature in STATIC_DISTANCE_FEATURES:
    new_col = f"log1p_{feature}"
    engineered[new_col] = np.log1p(engineered[feature])

    add_dictionary_row(
        feature_dictionary,
        new_col,
        "distance_transform",
        feature,
        f"log(1 + {feature})",
        "safe_deterministic",
        "Fixed monotonic transform; no train/test statistics are learned.",
    )

print("Created deterministic features:")
print(
    [
        "year_index",
        "vegetation_deficit",
        "ndbi_minus_ndvi",
        "ndbi_x_vegetation_deficit",
        "ndvi_x_green_mask",
        "ndbi_x_building_mask",
        "nightlights_x_building_mask",
        *[f"log1p_{x}" for x in STATIC_DISTANCE_FEATURES],
    ]
)


# =============================================================================
# 7. TEMPORAL HISTORY FEATURES - PREDICTORS ONLY
# =============================================================================

section("3. TEMPORAL HISTORY FEATURES - STRICTLY PAST PREDICTORS")

# Work in grid-year order to guarantee causal shifting.
engineered = engineered.sort_values(
    ["grid_id", "year"]
).reset_index(drop=True)

temporal_feature_names = []

for source in TEMPORAL_SOURCE_FEATURES:
    grouped = engineered.groupby(
        "grid_id",
        sort=False,
    )[source]

    lag1 = grouped.shift(1)
    lag2 = grouped.shift(2)
    lag3 = grouped.shift(3)

    lag1_col = f"{source}_lag1"
    lag2_col = f"{source}_lag2"
    mean_col = f"{source}_hist3_mean"
    std_col = f"{source}_hist3_std"
    slope_col = f"{source}_hist3_slope"
    delta_col = f"{source}_delta1"

    engineered[lag1_col] = lag1
    engineered[lag2_col] = lag2

    # Historical window contains ONLY t-1, t-2, t-3 predictors.
    hist_matrix = pd.concat(
        [lag1, lag2, lag3],
        axis=1,
    )

    engineered[mean_col] = hist_matrix.mean(
        axis=1,
        skipna=True,
    )

    engineered[std_col] = hist_matrix.std(
        axis=1,
        skipna=True,
        ddof=1,
    )

    # For equally spaced historical years t-3,t-2,t-1, OLS slope equals
    # (latest - earliest) / 2 when all three observations exist.
    engineered[slope_col] = np.where(
        lag1.notna() & lag2.notna() & lag3.notna(),
        (lag1 - lag3) / 2.0,
        np.nan,
    )

    # Current predictor change from previous year.
    # This is safe because it uses predictor values, not the target.
    engineered[delta_col] = (
        engineered[source] - lag1
    )

    temporal_feature_names.extend(
        [
            lag1_col,
            lag2_col,
            mean_col,
            std_col,
            slope_col,
            delta_col,
        ]
    )

    add_dictionary_row(
        feature_dictionary,
        lag1_col,
        "temporal_history",
        source,
        f"{source}(t-1)",
        "safe_historical_predictor",
        "Unavailable for the first year of each grid.",
    )

    add_dictionary_row(
        feature_dictionary,
        lag2_col,
        "temporal_history",
        source,
        f"{source}(t-2)",
        "safe_historical_predictor",
        "Unavailable for the first two years of each grid.",
    )

    add_dictionary_row(
        feature_dictionary,
        mean_col,
        "temporal_history",
        source,
        f"mean of available {source}(t-1), {source}(t-2), {source}(t-3)",
        "safe_historical_predictor",
        "Uses only prior-year predictor values; first year is structurally missing.",
    )

    add_dictionary_row(
        feature_dictionary,
        std_col,
        "temporal_history",
        source,
        f"sample std of available {source}(t-1), {source}(t-2), {source}(t-3)",
        "safe_historical_predictor",
        "Requires at least two historical predictor observations.",
    )

    add_dictionary_row(
        feature_dictionary,
        slope_col,
        "temporal_history",
        source,
        f"3-year historical slope from t-3 to t-1 = ({source}(t-1)-{source}(t-3))/2",
        "safe_historical_predictor",
        "Requires three previous annual predictor observations.",
    )

    add_dictionary_row(
        feature_dictionary,
        delta_col,
        "temporal_change",
        source,
        f"{source}(t) - {source}(t-1)",
        "safe_predictor_change",
        "Uses current predictor plus previous predictor; first year is structurally missing.",
    )

print(
    f"Created {len(temporal_feature_names)} temporal-history/change features "
    f"from {len(TEMPORAL_SOURCE_FEATURES)} source predictors."
)


# =============================================================================
# 8. SPATIAL NEIGHBOR FEATURES - NON-TARGET PREDICTORS ONLY
# =============================================================================

section("4. SPATIAL NEIGHBOR FEATURES - PREDICTORS ONLY")

if grid.crs is None or not grid.crs.is_projected:
    print(
        "\nERROR: Step 07 requires the corrected projected grid geometry."
    )
    sys.exit(1)

if grid["grid_id"].duplicated().any():
    print("\nERROR: Duplicate grid IDs in geometry.")
    sys.exit(1)

grid = (
    grid.sort_values("grid_id")
    .set_index("grid_id", drop=False)
)

csv_grid_ids = set(engineered["grid_id"].unique())
geo_grid_ids = set(grid["grid_id"].unique())

if csv_grid_ids != geo_grid_ids:
    print("\nERROR: CSV and geometry grid IDs do not match.")
    sys.exit(1)

w = Queen.from_dataframe(
    grid,
    use_index=True,
)

w.transform = "R"

if w.islands:
    print(
        f"\nERROR: Queen spatial weights contain {len(w.islands)} islands."
    )
    sys.exit(1)

spatial_feature_names = []
spatial_frames = []

for year in sorted(engineered["year"].unique()):
    yearly = (
        engineered.loc[
            engineered["year"] == year,
            ["grid_id"] + SPATIAL_SOURCE_FEATURES,
        ]
        .copy()
        .set_index("grid_id")
        .reindex(w.id_order)
    )

    if yearly.isna().any().any():
        print(
            f"\nERROR: Missing source predictors during spatial alignment for {year}."
        )
        sys.exit(1)

    spatial_year = pd.DataFrame(
        {
            "grid_id": w.id_order,
            "year": year,
        }
    )

    for source in SPATIAL_SOURCE_FEATURES:
        values = yearly[source].to_numpy(dtype=float)

        neighbor_mean = lag_spatial(
            w,
            values,
        )

        mean_col = f"{source}_nbr_mean"
        contrast_col = f"{source}_nbr_contrast"

        spatial_year[mean_col] = neighbor_mean
        spatial_year[contrast_col] = (
            values - neighbor_mean
        )

        if mean_col not in spatial_feature_names:
            spatial_feature_names.extend(
                [mean_col, contrast_col]
            )

    spatial_frames.append(spatial_year)

spatial_features = pd.concat(
    spatial_frames,
    ignore_index=True,
)

engineered = engineered.merge(
    spatial_features,
    on=["grid_id", "year"],
    how="left",
    validate="one_to_one",
)

for source in SPATIAL_SOURCE_FEATURES:
    mean_col = f"{source}_nbr_mean"
    contrast_col = f"{source}_nbr_contrast"

    add_dictionary_row(
        feature_dictionary,
        mean_col,
        "spatial_predictor_context",
        source,
        f"Queen row-standardized mean of neighboring-grid {source}",
        "safe_non_target_spatial",
        "Uses same-year non-target predictor values from adjacent grids.",
    )

    add_dictionary_row(
        feature_dictionary,
        contrast_col,
        "spatial_predictor_context",
        source,
        f"{source} - neighboring mean {source}",
        "safe_non_target_spatial",
        "Local predictor contrast; contains no LST target information.",
    )

print(
    f"Created {len(spatial_feature_names)} spatial predictor-context features."
)
print(
    "No spatial lag or neighbor mean of LST_C was created."
)


# =============================================================================
# 9. RESTORE READABLE YEAR-GRID ORDER
# =============================================================================

section("5. FINAL ENGINEERED DATASET ASSEMBLY")

engineered = engineered.sort_values(
    ["year", "grid_id"]
).reset_index(drop=True)

rows_after = len(engineered)
columns_after = len(engineered.columns)

print(f"Rows after engineering   : {rows_after:,}")
print(f"Columns after engineering: {columns_after:,}")

if rows_after != len(df):
    print("\nERROR: Row count changed during feature engineering.")
    sys.exit(1)

if engineered.duplicated(["grid_id", "year"], keep=False).any():
    print("\nERROR: Duplicate grid-year rows created.")
    sys.exit(1)


# =============================================================================
# 10. TARGET PRESERVATION CHECK
# =============================================================================

section("6. TARGET PRESERVATION / LEAKAGE CHECK")

target_after = engineered[
    ["grid_id", "year", TARGET]
].copy()

target_check = original_target.merge(
    target_after,
    on=["grid_id", "year"],
    how="outer",
    suffixes=("_before", "_after"),
    validate="one_to_one",
)

target_changed = int(
    (
        ~np.isclose(
            target_check[f"{TARGET}_before"],
            target_check[f"{TARGET}_after"],
            equal_nan=True,
        )
    ).sum()
)

# Search engineered feature names for any accidental target-derived predictor.
new_feature_names = [
    c for c in engineered.columns
    if c not in df.columns
]

suspicious_target_names = [
    c
    for c in new_feature_names
    if "lst" in c.lower()
    or TARGET.lower() in c.lower()
]

print(f"Target values changed            : {target_changed:,}")
print(f"Target-derived new feature names : {suspicious_target_names}")

if target_changed > 0:
    print("\nERROR: Target values changed unexpectedly.")
    sys.exit(1)

if suspicious_target_names:
    print(
        "\nERROR: A newly engineered feature appears target-derived."
    )
    sys.exit(1)

print("PASS: no engineered predictor was created from LST_C.")


# =============================================================================
# 11. TEMPORAL CAUSALITY AUDIT
# =============================================================================

section("7. TEMPORAL FEATURE CAUSALITY AUDIT")

temporal_audit_rows = []

# Verify lag1 against previous-year original source value for every source.
for source in TEMPORAL_SOURCE_FEATURES:
    test = engineered[
        ["grid_id", "year", source, f"{source}_lag1"]
    ].sort_values(["grid_id", "year"]).copy()

    expected_lag1 = (
        test.groupby("grid_id")[source]
        .shift(1)
    )

    actual_lag1 = test[f"{source}_lag1"]

    mismatch = ~(
        np.isclose(
            expected_lag1,
            actual_lag1,
            equal_nan=True,
        )
    )

    mismatch_count = int(mismatch.sum())

    temporal_audit_rows.append(
        {
            "feature_source": source,
            "lag1_mismatch_count": mismatch_count,
            "causal_shift_verified": mismatch_count == 0,
        }
    )

temporal_audit = pd.DataFrame(
    temporal_audit_rows
)

temporal_audit.to_csv(
    TEMPORAL_AUDIT_PATH,
    index=False,
)

print(temporal_audit.to_string(index=False))

if not temporal_audit["causal_shift_verified"].all():
    print("\nERROR: Temporal feature causality audit failed.")
    sys.exit(1)


# =============================================================================
# 12. SPATIAL FEATURE AUDIT
# =============================================================================

section("8. SPATIAL FEATURE AUDIT")

spatial_audit_rows = []

for source in SPATIAL_SOURCE_FEATURES:
    mean_col = f"{source}_nbr_mean"
    contrast_col = f"{source}_nbr_contrast"

    mean_missing = int(
        engineered[mean_col].isna().sum()
    )

    contrast_missing = int(
        engineered[contrast_col].isna().sum()
    )

    reconstruction_error = (
        engineered[source]
        - engineered[mean_col]
        - engineered[contrast_col]
    ).abs().max()

    spatial_audit_rows.append(
        {
            "source_feature": source,
            "neighbor_mean_missing": mean_missing,
            "neighbor_contrast_missing": contrast_missing,
            "max_contrast_reconstruction_error": reconstruction_error,
            "uses_target": False,
        }
    )

spatial_audit = pd.DataFrame(
    spatial_audit_rows
)

spatial_audit.to_csv(
    SPATIAL_AUDIT_PATH,
    index=False,
)

print(spatial_audit.to_string(index=False))

if (
    spatial_audit["neighbor_mean_missing"].sum() > 0
    or spatial_audit["neighbor_contrast_missing"].sum() > 0
):
    print("\nERROR: Unexpected missing spatial features.")
    sys.exit(1)


# =============================================================================
# 13. ENGINEERED MISSINGNESS AUDIT
# =============================================================================

section("9. STRUCTURAL MISSINGNESS AUDIT")

missing_rows = []

for col in engineered.columns:
    missing_count = int(
        engineered[col].isna().sum()
    )

    if missing_count > 0:
        missing_rows.append(
            {
                "feature": col,
                "missing_count": missing_count,
                "missing_percent": (
                    100 * missing_count / len(engineered)
                ),
                "expected_structural_reason": structural_missing_reason(col),
            }
        )

missingness = pd.DataFrame(
    missing_rows
)

if missingness.empty:
    missingness = pd.DataFrame(
        columns=[
            "feature",
            "missing_count",
            "missing_percent",
            "expected_structural_reason",
        ]
    )

missingness.to_csv(
    MISSINGNESS_PATH,
    index=False,
)

print(
    f"Engineered columns with structural missingness: "
    f"{len(missingness):,}"
)

if not missingness.empty:
    print(missingness.to_string(index=False))

# Missingness by year for engineered features only.
missing_by_year_rows = []

for year, group in engineered.groupby("year"):
    for col in new_feature_names:
        count = int(group[col].isna().sum())

        if count > 0:
            missing_by_year_rows.append(
                {
                    "year": year,
                    "feature": col,
                    "missing_count": count,
                    "row_count": len(group),
                    "missing_percent": 100 * count / len(group),
                }
            )

missingness_by_year = pd.DataFrame(
    missing_by_year_rows
)

if missingness_by_year.empty:
    missingness_by_year = pd.DataFrame(
        columns=[
            "year",
            "feature",
            "missing_count",
            "row_count",
            "missing_percent",
        ]
    )

missingness_by_year.to_csv(
    MISSINGNESS_BY_YEAR_PATH,
    index=False,
)

print(
    "\nImportant: these lag/history missing values are NOT imputed in Step 07. "
    "Any fitted imputation must occur later using training data only."
)


# =============================================================================
# 14. FEATURE COUNT SUMMARY
# =============================================================================

section("10. FEATURE COUNT SUMMARY")

deterministic_features = [
    "year_index",
    "vegetation_deficit",
    "ndbi_minus_ndvi",
    "ndbi_x_vegetation_deficit",
    "ndvi_x_green_mask",
    "ndbi_x_building_mask",
    "nightlights_x_building_mask",
    *[f"log1p_{x}" for x in STATIC_DISTANCE_FEATURES],
]

count_summary = pd.DataFrame(
    [
        {
            "feature_group": "original_columns",
            "count": len(df.columns),
        },
        {
            "feature_group": "deterministic_derived",
            "count": len(deterministic_features),
        },
        {
            "feature_group": "temporal_history_and_change",
            "count": len(temporal_feature_names),
        },
        {
            "feature_group": "spatial_predictor_context",
            "count": len(spatial_feature_names),
        },
        {
            "feature_group": "new_engineered_features_total",
            "count": len(new_feature_names),
        },
        {
            "feature_group": "final_columns_total",
            "count": len(engineered.columns),
        },
    ]
)

count_summary.to_csv(
    COUNT_SUMMARY_PATH,
    index=False,
)

print(count_summary.to_string(index=False))


# =============================================================================
# 15. LEAKAGE AUDIT
# =============================================================================

section("11. LEAKAGE AUDIT")

leakage_audit = pd.DataFrame(
    [
        {
            "check": "current_or_future_LST_used_to_create_predictors",
            "status": "PASS",
            "details": "No engineered predictor uses LST_C.",
        },
        {
            "check": "same_year_neighbor_LST_used",
            "status": "PASS",
            "details": "No neighbor/spatial lag of LST_C exists in engineered predictors.",
        },
        {
            "check": "future_predictor_values_used_in_temporal_features",
            "status": "PASS",
            "details": "Lag/history features use shift(1), shift(2), shift(3) only.",
        },
        {
            "check": "rolling_windows_include_current_predictor",
            "status": "PASS",
            "details": "Historical mean/std/slope are based on t-1,t-2,t-3 only.",
        },
        {
            "check": "2025_target_used_for_feature_decisions",
            "status": "PASS",
            "details": "2025 target was not inspected for feature engineering decisions.",
        },
        {
            "check": "scaling_or_imputation_fitted_before_split",
            "status": "PASS",
            "details": "No fitted preprocessing was performed.",
        },
        {
            "check": "grid_id_used_as_numeric_predictor",
            "status": "PASS",
            "details": "grid_id is retained only as an identifier.",
        },
        {
            "check": "area_name_encoded_as_predictor",
            "status": "PASS",
            "details": "area_name is retained for reporting only and is not encoded.",
        },
    ]
)

leakage_audit.to_csv(
    LEAKAGE_AUDIT_PATH,
    index=False,
)

print(leakage_audit.to_string(index=False))


# =============================================================================
# 16. FEATURE DICTIONARY
# =============================================================================

section("12. SAVE FEATURE DICTIONARY")

feature_dictionary_df = pd.DataFrame(
    feature_dictionary
)

# Keep dictionary in final column order.
feature_order = {
    feature: i
    for i, feature in enumerate(engineered.columns)
}

feature_dictionary_df["column_order"] = (
    feature_dictionary_df["feature"]
    .map(feature_order)
)

feature_dictionary_df = (
    feature_dictionary_df
    .sort_values("column_order")
    .reset_index(drop=True)
)

feature_dictionary_df.to_csv(
    FEATURE_DICTIONARY_PATH,
    index=False,
)

print(
    f"Feature dictionary rows: {len(feature_dictionary_df):,}"
)


# =============================================================================
# 17. SAVE ENGINEERED DATASET
# =============================================================================

section("13. SAVE ENGINEERED DATASET")

engineered.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)

print(f"Saved: {OUTPUT_PATH}")


# =============================================================================
# 18. POST-SAVE VALIDATION
# =============================================================================

section("14. POST-SAVE VALIDATION")

roundtrip = pd.read_csv(
    OUTPUT_PATH,
    low_memory=False,
    dtype={"grid_id": "string", "area_name": "string"},
)

same_rows = len(roundtrip) == len(df)
same_keys = (
    roundtrip[["grid_id", "year"]]
    .drop_duplicates()
    .shape[0]
    == df[["grid_id", "year"]]
    .drop_duplicates()
    .shape[0]
)

same_target = np.allclose(
    roundtrip[TARGET],
    engineered[TARGET],
    equal_nan=True,
)

duplicate_keys = int(
    roundtrip.duplicated(
        ["grid_id", "year"],
        keep=False,
    ).sum()
)

print(f"Same row count        : {same_rows}")
print(f"Same unique key count : {same_keys}")
print(f"Same target values    : {same_target}")
print(f"Duplicate key rows    : {duplicate_keys:,}")

if not all(
    [
        same_rows,
        same_keys,
        same_target,
        duplicate_keys == 0,
    ]
):
    print("\nERROR: Post-save validation failed.")
    sys.exit(1)


# =============================================================================
# 19. MAIN REPORT
# =============================================================================

section("15. SAVE FEATURE ENGINEERING REPORT")

report_lines = [
    "STEP 07 - FEATURE ENGINEERING REPORT",
    "=" * 95,
    "",
    f"Input dataset: {INPUT_PATH}",
    f"Output dataset: {OUTPUT_PATH}",
    "",
    "DATASET SIZE",
    "-" * 95,
    f"Rows before: {len(df):,}",
    f"Rows after: {len(engineered):,}",
    f"Columns before: {len(df.columns):,}",
    f"Columns after: {len(engineered.columns):,}",
    f"New engineered features: {len(new_feature_names):,}",
    "",
    "FEATURE GROUPS",
    "-" * 95,
    f"Deterministic derived features: {len(deterministic_features):,}",
    f"Temporal history/change features: {len(temporal_feature_names):,}",
    f"Spatial non-target context features: {len(spatial_feature_names):,}",
    "",
    "TEMPORAL FEATURES",
    "-" * 95,
    "Temporal source predictors:",
    ", ".join(TEMPORAL_SOURCE_FEATURES),
    "",
    "For each source predictor, Step 07 created:",
    "- lag 1 year",
    "- lag 2 years",
    "- historical 3-year mean using t-1,t-2,t-3",
    "- historical 3-year standard deviation",
    "- historical 3-year slope",
    "- current-vs-previous-year predictor change",
    "",
    "SPATIAL FEATURES",
    "-" * 95,
    "Queen-contiguity row-standardized neighbor means were created for:",
    ", ".join(SPATIAL_SOURCE_FEATURES),
    "A local contrast = current predictor - neighbor predictor mean was also created.",
    "No LST_C spatial lag or neighbor-LST feature was created.",
    "",
    "STRUCTURAL MISSINGNESS",
    "-" * 95,
    f"Columns with structural missingness: {len(missingness):,}",
    "Lag/history missingness is expected in early years and was preserved.",
    "No imputation was performed.",
    "",
    "LOCKBOX / LEAKAGE",
    "-" * 95,
    f"Development period: {DEVELOPMENT_START_YEAR}-{DEVELOPMENT_END_YEAR}",
    f"Final lockbox year: {LOCKBOX_YEAR}",
    "No current/future target information was used to create predictors.",
    "No scaler, imputer, PCA, feature selector, or model was fitted.",
    "2025 target was not used for feature engineering decisions.",
    "",
    "IMPORTANT FOR LATER MODELING",
    "-" * 95,
    "grid_id must remain an identifier, not a numeric predictor.",
    "area_name should remain a reporting/grouping field unless a later justified spatial encoding is designed.",
    "Same-year spatial predictor context is built only from non-target covariates.",
    "Any imputation/scaling must be fitted on training data only.",
    "Feature selection belongs in Step 08.",
]

MAIN_REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {MAIN_REPORT}")
print(f"Saved: {FEATURE_DICTIONARY_PATH}")
print(f"Saved: {MISSINGNESS_PATH}")
print(f"Saved: {MISSINGNESS_BY_YEAR_PATH}")
print(f"Saved: {LEAKAGE_AUDIT_PATH}")
print(f"Saved: {COUNT_SUMMARY_PATH}")
print(f"Saved: {SPATIAL_AUDIT_PATH}")
print(f"Saved: {TEMPORAL_AUDIT_PATH}")


# =============================================================================
# 20. FINAL STATUS
# =============================================================================

section("STEP 07 COMPLETED SUCCESSFULLY")

print(
    "Leakage-aware feature engineering completed.\n"
    "The engineered dataset contains deterministic, temporal-history, and "
    "non-target spatial-context features.\n"
    "Review all Step 07 outputs before starting Step 08 Feature Analysis / Selection."
)
