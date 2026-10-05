"""
STEP 04 - BASE PREPROCESSING
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Create a clean, reproducible BASE processed dataset without leaking target
information or making unsupported corrections.

This step performs only safe preprocessing actions:
1. validate the raw schema again
2. preserve grid_id exactly as an identifier
3. standardize safe string formatting in the processed copy
4. enforce numeric types where justified
5. verify missing / infinite / duplicate / domain conditions
6. preserve legitimate LST extremes and the 2017 anomaly
7. preserve LCZ exactly as supplied (fractional LCZ is not changed yet)
8. correct the GRID GEOJSON CRS metadata in a processed spatial copy
9. create projected and WGS84 geometry outputs
10. save reproducibility hashes and preprocessing audit reports

IMPORTANT
---------
- data/raw/ is READ ONLY.
- No row is deleted.
- No outlier is removed.
- No target-based quality flag is added to the modeling dataset.
- No imputation is fitted here.
- No scaling/normalization is fitted here.
- No feature selection is performed here.
- No train/validation/test split is performed here.
- 2025 remains untouched for later final model evaluation.
"""

from pathlib import Path
import hashlib
import json
import re
import sys
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = RAW_DIR / "kaduwela_master_raw.csv"
RAW_GRID = RAW_DIR / "kaduwela_grid_geometry.geojson"
RAW_BOUNDARY = RAW_DIR / "boundary" / "kaduwela_kmc_boundary.kml"

PROCESSED_CSV = PROCESSED_DIR / "kaduwela_preprocessed_base.csv"

GRID_UTM = PROCESSED_DIR / "kaduwela_grid_geometry_utm44n.geojson"
GRID_WGS84 = PROCESSED_DIR / "kaduwela_grid_geometry_wgs84.geojson"

BOUNDARY_WGS84 = PROCESSED_DIR / "kaduwela_boundary_wgs84.geojson"
BOUNDARY_UTM = PROCESSED_DIR / "kaduwela_boundary_utm44n.geojson"

REPORT_PATH = REPORT_DIR / "04_preprocessing_report.txt"
AUDIT_PATH = REPORT_DIR / "04_preprocessing_audit.csv"
HASH_PATH = REPORT_DIR / "04_file_hashes.json"
METADATA_PATH = REPORT_DIR / "04_preprocessing_metadata.json"


# =============================================================================
# 2. CONSTANTS
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

# The grid coordinates observed in Step 03 are projected coordinates around
# easting ~378-395 km and northing ~758-768 km. Kaduwela lies in UTM zone 44N.
CORRECT_GRID_CRS = "EPSG:32644"
WEB_MAP_CRS = "EPSG:4326"

GRID_ID_PATTERN = re.compile(r"^\d+,\d+$")


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def count_inf(df: pd.DataFrame, columns: list[str]) -> tuple[int, int]:
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    arr = numeric.to_numpy(dtype=float)
    return int(np.isposinf(arr).sum()), int(np.isneginf(arr).sum())


def audit_row(check, before, after, status, action, note=""):
    return {
        "check": check,
        "before": before,
        "after": after,
        "status": status,
        "action": action,
        "note": note,
    }


def validate_kaduwela_wgs84(gdf_wgs84: gpd.GeoDataFrame, label: str) -> None:
    """
    Broad geographic sanity check for Sri Lanka / Kaduwela after reprojection.
    This prevents silently accepting the wrong projected CRS.
    """
    minx, miny, maxx, maxy = gdf_wgs84.total_bounds

    # Broad Sri Lanka / western-region sanity window.
    if not (
        79.0 <= minx <= 82.0
        and 79.0 <= maxx <= 82.0
        and 5.0 <= miny <= 10.0
        and 5.0 <= maxy <= 10.0
    ):
        raise ValueError(
            f"{label} failed WGS84 sanity check after CRS conversion. "
            f"Bounds were {gdf_wgs84.total_bounds}. "
            "Do not continue until the source CRS is confirmed."
        )


# =============================================================================
# 4. LOAD RAW CSV
# =============================================================================

section("STEP 04 - BASE PREPROCESSING")

print(f"Project root : {PROJECT_ROOT}")
print(f"Raw CSV      : {RAW_CSV}")

if not RAW_CSV.exists():
    print(f"\nERROR: Raw CSV not found: {RAW_CSV}")
    sys.exit(1)

raw_hash_before = sha256_file(RAW_CSV)

try:
    raw_df = pd.read_csv(RAW_CSV, low_memory=False)
except Exception as exc:
    print(f"\nERROR: Could not read raw CSV.\n{exc}")
    sys.exit(1)

if raw_df.empty:
    print("\nERROR: Raw dataset is empty.")
    sys.exit(1)

print("Raw dataset loaded successfully.")
print(f"Rows    : {len(raw_df):,}")
print(f"Columns : {len(raw_df.columns):,}")
print(f"SHA256  : {raw_hash_before}")


# =============================================================================
# 5. FAIL-FAST SCHEMA VALIDATION
# =============================================================================

section("1. FAIL-FAST SCHEMA VALIDATION")

missing_columns = sorted(set(EXPECTED_COLUMNS) - set(raw_df.columns))
extra_columns = sorted(set(raw_df.columns) - set(EXPECTED_COLUMNS))

print(f"Missing expected columns : {missing_columns}")
print(f"Unexpected columns       : {extra_columns}")

if missing_columns:
    print("\nERROR: Required columns are missing. Preprocessing stopped.")
    sys.exit(1)

# Require exact expected schema to prevent silent pipeline drift.
if list(raw_df.columns) != EXPECTED_COLUMNS:
    print("\nERROR: Column order/schema differs from the approved master schema.")
    print("Preprocessing stopped to avoid silent pipeline drift.")
    sys.exit(1)


# =============================================================================
# 6. RAW INTEGRITY SNAPSHOT
# =============================================================================

section("2. RAW INTEGRITY SNAPSHOT")

raw_rows = len(raw_df)
raw_columns = len(raw_df.columns)
raw_missing = int(raw_df.isna().sum().sum())
raw_full_duplicates = int(raw_df.duplicated().sum())
raw_key_duplicates = int(
    raw_df.duplicated(["grid_id", "year"], keep=False).sum()
)

raw_positive_inf, raw_negative_inf = count_inf(raw_df, NUMERIC_COLUMNS)

raw_unique_grids = int(raw_df["grid_id"].nunique())
raw_unique_years = int(raw_df["year"].nunique())
raw_unique_grid_year = int(
    raw_df[["grid_id", "year"]].drop_duplicates().shape[0]
)

print(f"Rows                       : {raw_rows:,}")
print(f"Columns                    : {raw_columns:,}")
print(f"Missing cells              : {raw_missing:,}")
print(f"Full duplicate rows        : {raw_full_duplicates:,}")
print(f"Duplicate grid-year rows   : {raw_key_duplicates:,}")
print(f"Positive infinity values   : {raw_positive_inf:,}")
print(f"Negative infinity values   : {raw_negative_inf:,}")
print(f"Unique grids               : {raw_unique_grids:,}")
print(f"Unique years               : {raw_unique_years:,}")
print(f"Unique grid-year pairs     : {raw_unique_grid_year:,}")


# =============================================================================
# 7. CREATE WORKING COPY - RAW DATA REMAINS UNCHANGED
# =============================================================================

section("3. SAFE BASE PREPROCESSING")

processed = raw_df.copy(deep=True)

# -------------------------------------------------------------------------
# 7.1 grid_id - preserve identifier text
# -------------------------------------------------------------------------
grid_before = processed["grid_id"].copy()

processed["grid_id"] = processed["grid_id"].astype("string").str.strip()

grid_id_whitespace_changes = int(
    (grid_before.astype(str) != processed["grid_id"].astype(str)).sum()
)

invalid_grid_id_format = int(
    (~processed["grid_id"].fillna("").map(
        lambda x: bool(GRID_ID_PATTERN.fullmatch(str(x)))
    )).sum()
)

if invalid_grid_id_format > 0:
    print(
        f"\nERROR: {invalid_grid_id_format} grid_id values do not match "
        "the expected 'number,number' structure."
    )
    sys.exit(1)

# -------------------------------------------------------------------------
# 7.2 area_name - only trim leading/trailing whitespace
# -------------------------------------------------------------------------
area_before = processed["area_name"].copy()

processed["area_name"] = processed["area_name"].astype("string").str.strip()

area_whitespace_changes = int(
    (area_before.astype(str) != processed["area_name"].astype(str)).sum()
)

# IMPORTANT: no spelling correction is performed here.
# Names such as "Battaramulla Sout" or "Udumulla2" are retained unless an
# authoritative source confirms a correction.

# -------------------------------------------------------------------------
# 7.3 enforce numeric parsing
# -------------------------------------------------------------------------
numeric_parse_failures = {}

for col in NUMERIC_COLUMNS:
    original_non_null = processed[col].notna()
    converted = pd.to_numeric(processed[col], errors="coerce")

    failures = int(
        (original_non_null & converted.isna()).sum()
    )

    numeric_parse_failures[col] = failures

    if failures > 0:
        print(
            f"\nERROR: {failures} numeric parsing failures detected in {col}."
        )
        sys.exit(1)

    processed[col] = converted

# year must be integer-valued and within approved research period
year_fractional = int(
    ((processed["year"] % 1) != 0).sum()
)

if year_fractional > 0:
    print("\nERROR: Non-integer year values detected.")
    sys.exit(1)

processed["year"] = processed["year"].astype("int64")

if not processed["year"].between(2015, 2025).all():
    print("\nERROR: Year outside approved 2015-2025 period detected.")
    sys.exit(1)

print(f"grid_id whitespace changes : {grid_id_whitespace_changes:,}")
print(f"area_name whitespace changes: {area_whitespace_changes:,}")
print(f"Numeric parse failures      : {sum(numeric_parse_failures.values()):,}")


# =============================================================================
# 8. MISSING / INFINITY POLICY
# =============================================================================

section("4. MISSING / INFINITY POLICY")

processed_missing = int(processed.isna().sum().sum())
processed_positive_inf, processed_negative_inf = count_inf(
    processed,
    NUMERIC_COLUMNS,
)

print(f"Missing cells             : {processed_missing:,}")
print(f"Positive infinity values  : {processed_positive_inf:,}")
print(f"Negative infinity values  : {processed_negative_inf:,}")

if processed_missing > 0:
    print(
        "\nERROR: Missing values exist. Step 03 reported none, so this is "
        "unexpected. No automatic imputation will be performed."
    )
    sys.exit(1)

if processed_positive_inf > 0 or processed_negative_inf > 0:
    print(
        "\nERROR: Infinite numeric values exist. No automatic replacement "
        "will be performed."
    )
    sys.exit(1)

print(
    "No imputation was required. No zero-filling or statistical imputation "
    "was performed."
)


# =============================================================================
# 9. DUPLICATE POLICY
# =============================================================================

section("5. DUPLICATE POLICY")

processed_full_duplicates = int(processed.duplicated().sum())
processed_key_duplicates = int(
    processed.duplicated(["grid_id", "year"], keep=False).sum()
)

print(f"Full duplicate rows      : {processed_full_duplicates:,}")
print(f"Duplicate grid-year rows : {processed_key_duplicates:,}")

if processed_full_duplicates > 0 or processed_key_duplicates > 0:
    print(
        "\nERROR: Duplicate records exist. No automatic drop is allowed "
        "without investigation."
    )
    sys.exit(1)

print("No duplicate removal was required.")


# =============================================================================
# 10. DOMAIN VALIDATION - NO CLIPPING
# =============================================================================

section("6. DOMAIN VALIDATION - NO CLIPPING")

domain_checks = {
    "NDVI": (-1, 1),
    "NDBI": (-1, 1),
    "NDWI": (-1, 1),
    "EVI": (-1, 2),
    "Albedo": (0, 1),
    "NDBI_ADJ": (-1, 1),
    "NDVI_ADJ": (-1, 1),
    "NightLights": (0, None),
    "green_mask": (0, 1),
    "building_mask": (0, 1),
    "road_mask": (0, 1),
    "dist_road": (0, None),
    "dist_main_road": (0, None),
    "dist_water": (0, None),
    "LCZ": (1, 17),
    "LST_C": (-20, 80),
}

domain_violation_counts = {}

for col, (lower, upper) in domain_checks.items():
    s = processed[col]

    invalid = pd.Series(False, index=processed.index)

    if lower is not None:
        invalid |= s < lower

    if upper is not None:
        invalid |= s > upper

    count = int(invalid.sum())
    domain_violation_counts[col] = count

    print(f"{col:<16}: {count:>6,} violations")

if sum(domain_violation_counts.values()) > 0:
    print(
        "\nERROR: Domain violations were found. "
        "No clipping is performed automatically."
    )
    sys.exit(1)

# binary indicator check
for col in ["building_mask", "road_mask"]:
    invalid_binary = int((~processed[col].isin([0, 1])).sum())

    if invalid_binary > 0:
        print(f"\nERROR: {col} contains non-binary values.")
        sys.exit(1)

print("\nAll approved domain checks passed. No clipping was performed.")


# =============================================================================
# 11. OUTLIER / 2017 POLICY
# =============================================================================

section("7. OUTLIER AND 2017 ANOMALY POLICY")

lst_min_before = float(raw_df["LST_C"].min())
lst_max_before = float(raw_df["LST_C"].max())

lst_min_after = float(processed["LST_C"].min())
lst_max_after = float(processed["LST_C"].max())

print(f"LST minimum preserved : {lst_min_after:.6f} °C")
print(f"LST maximum preserved : {lst_max_after:.6f} °C")

rows_2017 = int((processed["year"] == 2017).sum())

print(f"2017 rows preserved   : {rows_2017:,}")

if lst_min_before != lst_min_after or lst_max_before != lst_max_after:
    print("\nERROR: LST extrema changed unexpectedly.")
    sys.exit(1)

print(
    "\nPolicy:"
    "\n- No LST outlier was removed."
    "\n- The 2017 anomaly was preserved."
    "\n- No target-derived anomaly flag was added to the modeling dataset."
    "\n- Sensitivity analysis will be performed later, after the main pipeline."
)


# =============================================================================
# 12. LCZ POLICY
# =============================================================================

section("8. LCZ POLICY")

lcz = processed["LCZ"]
lcz_fractional = int(
    ((lcz - lcz.round()).abs() > 1e-9).sum()
)

print(f"LCZ fractional rows preserved: {lcz_fractional:,}")

print(
    "\nLCZ is kept exactly as supplied in the base processed dataset."
    "\nWe are NOT rounding it and NOT inventing a dominant class."
    "\nThe LCZ representation decision is postponed to feature analysis, "
    "because Step 03 showed that grid aggregation produced fractional values."
)


# =============================================================================
# 13. LEAKAGE-SAFE PREPROCESSING POLICY
# =============================================================================

section("9. LEAKAGE-SAFE PREPROCESSING POLICY")

print(
    "NOT performed in Step 04:"
    "\n- StandardScaler / MinMaxScaler"
    "\n- mean/median/KNN imputation"
    "\n- PCA"
    "\n- feature selection"
    "\n- target encoding"
    "\n- train/validation/test splitting"
    "\n- model fitting"
    "\n- hyperparameter tuning"
)

print(
    "\nReason: transformations that learn from data must later be fitted "
    "only on the training portion to avoid leakage."
)


# =============================================================================
# 14. SAVE BASE PROCESSED CSV
# =============================================================================

section("10. SAVE BASE PROCESSED CSV")

processed.to_csv(
    PROCESSED_CSV,
    index=False,
    encoding="utf-8",
)

processed_hash = sha256_file(PROCESSED_CSV)

print(f"Saved: {PROCESSED_CSV}")
print(f"SHA256: {processed_hash}")


# =============================================================================
# 15. POST-SAVE ROUND-TRIP VALIDATION
# =============================================================================

section("11. POST-SAVE ROUND-TRIP VALIDATION")

roundtrip = pd.read_csv(
    PROCESSED_CSV,
    low_memory=False,
    dtype={"grid_id": "string", "area_name": "string"},
)

roundtrip_rows = len(roundtrip)
roundtrip_columns = len(roundtrip.columns)
roundtrip_key_duplicates = int(
    roundtrip.duplicated(["grid_id", "year"], keep=False).sum()
)
roundtrip_missing = int(roundtrip.isna().sum().sum())

same_row_count = roundtrip_rows == raw_rows
same_column_count = roundtrip_columns == raw_columns
same_grid_count = roundtrip["grid_id"].nunique() == raw_unique_grids
same_year_count = roundtrip["year"].nunique() == raw_unique_years
same_key_count = (
    roundtrip[["grid_id", "year"]].drop_duplicates().shape[0]
    == raw_unique_grid_year
)

# Key and target preservation are especially important.
raw_key_target = (
    raw_df[["grid_id", "year", "LST_C"]]
    .copy()
)
raw_key_target["grid_id"] = raw_key_target["grid_id"].astype(str).str.strip()

roundtrip_key_target = (
    roundtrip[["grid_id", "year", "LST_C"]]
    .copy()
)
roundtrip_key_target["grid_id"] = (
    roundtrip_key_target["grid_id"].astype(str).str.strip()
)

same_key_target = raw_key_target.equals(roundtrip_key_target)

print(f"Same row count          : {same_row_count}")
print(f"Same column count       : {same_column_count}")
print(f"Same unique grid count  : {same_grid_count}")
print(f"Same unique year count  : {same_year_count}")
print(f"Same grid-year key count: {same_key_count}")
print(f"Same grid/year/LST data : {same_key_target}")
print(f"Missing after save      : {roundtrip_missing:,}")
print(f"Duplicate key rows      : {roundtrip_key_duplicates:,}")

if not all(
    [
        same_row_count,
        same_column_count,
        same_grid_count,
        same_year_count,
        same_key_count,
        same_key_target,
        roundtrip_missing == 0,
        roundtrip_key_duplicates == 0,
    ]
):
    print("\nERROR: Processed CSV failed round-trip validation.")
    sys.exit(1)


# =============================================================================
# 16. GRID GEOJSON CRS CORRECTION - PROCESSED COPY ONLY
# =============================================================================

section("12. GRID GEOJSON CRS CORRECTION")

grid_processed_ok = False
grid_rows = None
grid_invalid = None

if RAW_GRID.exists():
    try:
        grid = gpd.read_file(RAW_GRID)

        if "grid_id" not in grid.columns:
            raise ValueError("Grid GeoJSON does not contain grid_id.")

        grid = grid.copy()
        grid["grid_id"] = grid["grid_id"].astype(str).str.strip()

        grid_rows = len(grid)
        grid_invalid = int((~grid.geometry.is_valid).sum())

        if grid_invalid > 0:
            raise ValueError(
                f"Grid GeoJSON contains {grid_invalid} invalid geometries."
            )

        csv_ids = set(processed["grid_id"].astype(str))
        geo_ids = set(grid["grid_id"].astype(str))

        if csv_ids != geo_ids:
            raise ValueError(
                "CSV grid IDs and GeoJSON grid IDs do not match exactly."
            )

        print(f"Raw grid reported CRS : {grid.crs}")
        print(f"Raw grid bounds       : {grid.total_bounds}")

        # STEP 03 proved the metadata says EPSG:4326 while coordinates are projected.
        # We correct metadata on a processed copy only.
        grid_utm = grid.set_crs(
            CORRECT_GRID_CRS,
            allow_override=True,
        )

        grid_wgs84 = grid_utm.to_crs(WEB_MAP_CRS)

        # Confirm the corrected CRS transforms into a plausible Sri Lankan location.
        validate_kaduwela_wgs84(
            grid_wgs84,
            "Grid geometry",
        )

        grid_utm.to_file(
            GRID_UTM,
            driver="GeoJSON",
        )

        grid_wgs84.to_file(
            GRID_WGS84,
            driver="GeoJSON",
        )

        print(f"Corrected projected CRS : {grid_utm.crs}")
        print(f"WGS84 bounds            : {grid_wgs84.total_bounds}")
        print(f"Saved projected grid    : {GRID_UTM}")
        print(f"Saved web-map grid      : {GRID_WGS84}")

        grid_processed_ok = True

    except Exception as exc:
        warnings.warn(
            f"Grid geometry preprocessing failed: {exc}"
        )
        print(
            "\nSpatial preprocessing was not silently continued. "
            "Review the warning before using geometry in later spatial analysis."
        )
else:
    warnings.warn(
        f"Grid GeoJSON not found: {RAW_GRID}"
    )


# =============================================================================
# 17. BOUNDARY CONVERSION
# =============================================================================

section("13. BOUNDARY SPATIAL PREPARATION")

boundary_processed_ok = False

if RAW_BOUNDARY.exists():
    try:
        boundary = gpd.read_file(RAW_BOUNDARY)

        if boundary.empty:
            raise ValueError("Boundary KML contains no geometry.")

        invalid_boundary = int(
            (~boundary.geometry.is_valid).sum()
        )

        if invalid_boundary > 0:
            raise ValueError(
                f"Boundary contains {invalid_boundary} invalid geometries."
            )

        if boundary.crs is None:
            # KML is normally WGS84, but fail-safe rather than guessing silently.
            raise ValueError(
                "Boundary CRS is missing. Confirm CRS before conversion."
            )

        boundary_wgs84 = boundary.to_crs(WEB_MAP_CRS)
        validate_kaduwela_wgs84(
            boundary_wgs84,
            "Kaduwela boundary",
        )

        boundary_utm = boundary_wgs84.to_crs(
            CORRECT_GRID_CRS
        )

        boundary_wgs84.to_file(
            BOUNDARY_WGS84,
            driver="GeoJSON",
        )

        boundary_utm.to_file(
            BOUNDARY_UTM,
            driver="GeoJSON",
        )

        print(f"Boundary input CRS      : {boundary.crs}")
        print(f"Boundary WGS84 bounds   : {boundary_wgs84.total_bounds}")
        print(f"Saved boundary WGS84    : {BOUNDARY_WGS84}")
        print(f"Saved boundary UTM      : {BOUNDARY_UTM}")

        boundary_processed_ok = True

    except Exception as exc:
        warnings.warn(
            f"Boundary preprocessing failed: {exc}"
        )
else:
    warnings.warn(
        f"Boundary KML not found: {RAW_BOUNDARY}"
    )


# =============================================================================
# 18. VERIFY RAW CSV WAS NOT MODIFIED
# =============================================================================

section("14. RAW FILE IMMUTABILITY CHECK")

raw_hash_after = sha256_file(RAW_CSV)
raw_unchanged = raw_hash_before == raw_hash_after

print(f"Raw SHA256 before : {raw_hash_before}")
print(f"Raw SHA256 after  : {raw_hash_after}")
print(f"Raw file unchanged: {raw_unchanged}")

if not raw_unchanged:
    print(
        "\nCRITICAL ERROR: Raw dataset hash changed during preprocessing."
    )
    sys.exit(1)


# =============================================================================
# 19. BUILD PREPROCESSING AUDIT
# =============================================================================

section("15. PREPROCESSING AUDIT")

audit = []

audit.append(
    audit_row(
        "row_count",
        raw_rows,
        len(processed),
        "PASS" if raw_rows == len(processed) else "FAIL",
        "preserved",
    )
)

audit.append(
    audit_row(
        "column_count",
        raw_columns,
        len(processed.columns),
        "PASS" if raw_columns == len(processed.columns) else "FAIL",
        "preserved",
    )
)

audit.append(
    audit_row(
        "missing_cells",
        raw_missing,
        processed_missing,
        "PASS" if processed_missing == 0 else "REVIEW",
        "no imputation",
    )
)

audit.append(
    audit_row(
        "duplicate_grid_year_rows",
        raw_key_duplicates,
        processed_key_duplicates,
        "PASS" if processed_key_duplicates == 0 else "FAIL",
        "no duplicate removal required",
    )
)

audit.append(
    audit_row(
        "grid_id_whitespace_changes",
        0,
        grid_id_whitespace_changes,
        "PASS",
        "trim only",
        "Identifier text otherwise preserved.",
    )
)

audit.append(
    audit_row(
        "area_name_whitespace_changes",
        0,
        area_whitespace_changes,
        "PASS",
        "trim only",
        "No spelling correction applied.",
    )
)

audit.append(
    audit_row(
        "LST_min",
        lst_min_before,
        lst_min_after,
        "PASS" if lst_min_before == lst_min_after else "FAIL",
        "preserved",
    )
)

audit.append(
    audit_row(
        "LST_max",
        lst_max_before,
        lst_max_after,
        "PASS" if lst_max_before == lst_max_after else "FAIL",
        "preserved",
    )
)

audit.append(
    audit_row(
        "2017_rows",
        int((raw_df["year"] == 2017).sum()),
        rows_2017,
        "PASS",
        "preserved",
        "No anomaly removal.",
    )
)

audit.append(
    audit_row(
        "LCZ_fractional_rows",
        lcz_fractional,
        lcz_fractional,
        "PASS_WITH_NOTE",
        "preserved",
        "LCZ representation decision postponed.",
    )
)

audit.append(
    audit_row(
        "raw_file_hash",
        raw_hash_before,
        raw_hash_after,
        "PASS" if raw_unchanged else "FAIL",
        "read only",
    )
)

audit.append(
    audit_row(
        "grid_geometry_processed",
        str(RAW_GRID.exists()),
        str(grid_processed_ok),
        "PASS" if grid_processed_ok else "REVIEW",
        "processed-copy CRS correction",
        "Raw geometry not modified.",
    )
)

audit.append(
    audit_row(
        "boundary_processed",
        str(RAW_BOUNDARY.exists()),
        str(boundary_processed_ok),
        "PASS" if boundary_processed_ok else "REVIEW",
        "GeoJSON conversion",
        "Raw boundary not modified.",
    )
)

audit_df = pd.DataFrame(audit)
audit_df.to_csv(AUDIT_PATH, index=False)

print(audit_df.to_string(index=False))


# =============================================================================
# 20. REPRODUCIBILITY METADATA + HASHES
# =============================================================================

section("16. REPRODUCIBILITY METADATA")

hashes = {
    "raw_master_csv_sha256": raw_hash_before,
    "processed_base_csv_sha256": processed_hash,
}

if RAW_GRID.exists():
    hashes["raw_grid_geojson_sha256"] = sha256_file(RAW_GRID)

if RAW_BOUNDARY.exists():
    hashes["raw_boundary_kml_sha256"] = sha256_file(RAW_BOUNDARY)

if GRID_UTM.exists():
    hashes["processed_grid_utm44n_sha256"] = sha256_file(GRID_UTM)

if GRID_WGS84.exists():
    hashes["processed_grid_wgs84_sha256"] = sha256_file(GRID_WGS84)

HASH_PATH.write_text(
    json.dumps(hashes, indent=2),
    encoding="utf-8",
)

metadata = {
    "step": "04_base_preprocessing",
    "raw_dataset": str(RAW_CSV),
    "processed_dataset": str(PROCESSED_CSV),
    "rows_before": raw_rows,
    "rows_after": len(processed),
    "columns_before": raw_columns,
    "columns_after": len(processed.columns),
    "unique_grids": raw_unique_grids,
    "unique_years": raw_unique_years,
    "study_period": [2015, 2025],
    "target": "LST_C",
    "raw_modified": False,
    "rows_removed": 0,
    "outliers_removed": 0,
    "2017_rows_removed": 0,
    "imputation_performed": False,
    "scaling_performed": False,
    "feature_selection_performed": False,
    "train_test_split_performed": False,
    "lcz_rounded_or_reclassified": False,
    "grid_corrected_projected_crs": CORRECT_GRID_CRS if grid_processed_ok else None,
    "grid_web_map_crs": WEB_MAP_CRS if grid_processed_ok else None,
    "notes": [
        "2017 anomaly intentionally preserved for later sensitivity analysis.",
        "Persistent high-LST grids intentionally preserved.",
        "LCZ fractional values intentionally preserved pending feature-analysis decision.",
        "Any fitted preprocessing must later be trained on training data only.",
    ],
}

METADATA_PATH.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print(f"Saved: {HASH_PATH}")
print(f"Saved: {METADATA_PATH}")


# =============================================================================
# 21. MAIN TEXT REPORT
# =============================================================================

section("17. SAVE PREPROCESSING REPORT")

report_lines = [
    "STEP 04 - BASE PREPROCESSING REPORT",
    "=" * 90,
    "",
    f"Raw input: {RAW_CSV}",
    f"Processed output: {PROCESSED_CSV}",
    "",
    "DATASET PRESERVATION",
    "-" * 90,
    f"Rows before: {raw_rows:,}",
    f"Rows after: {len(processed):,}",
    f"Columns before: {raw_columns:,}",
    f"Columns after: {len(processed.columns):,}",
    f"Unique grids: {raw_unique_grids:,}",
    f"Unique years: {raw_unique_years:,}",
    f"Duplicate grid-year rows after: {processed_key_duplicates:,}",
    f"Missing cells after: {processed_missing:,}",
    "",
    "SAFE TRANSFORMATIONS",
    "-" * 90,
    f"grid_id whitespace-only changes: {grid_id_whitespace_changes:,}",
    f"area_name whitespace-only changes: {area_whitespace_changes:,}",
    "Numeric parsing validated.",
    "Year validated as integer in 2015-2025.",
    "",
    "OUTLIER POLICY",
    "-" * 90,
    f"LST minimum preserved: {lst_min_after:.6f}",
    f"LST maximum preserved: {lst_max_after:.6f}",
    f"2017 rows preserved: {rows_2017:,}",
    "No statistical outlier was removed or capped.",
    "No target-derived anomaly flag was added to the modeling dataset.",
    "",
    "LCZ POLICY",
    "-" * 90,
    f"Fractional LCZ rows preserved: {lcz_fractional:,}",
    "LCZ was not rounded, reclassified, or converted to a dominant class.",
    "That decision is postponed to feature analysis.",
    "",
    "SPATIAL PREPROCESSING",
    "-" * 90,
    f"Grid processed successfully: {grid_processed_ok}",
    f"Corrected grid CRS in processed copy: {CORRECT_GRID_CRS if grid_processed_ok else 'N/A'}",
    f"Web-map grid CRS: {WEB_MAP_CRS if grid_processed_ok else 'N/A'}",
    f"Boundary processed successfully: {boundary_processed_ok}",
    "",
    "LEAKAGE SAFEGUARDS",
    "-" * 90,
    "No scaler fitted.",
    "No imputer fitted.",
    "No feature selection performed.",
    "No target encoding performed.",
    "No model fitted.",
    "No train/validation/test split performed.",
    "2025 remains reserved for later final model evaluation.",
    "",
    "RAW IMMUTABILITY",
    "-" * 90,
    f"Raw SHA256 before: {raw_hash_before}",
    f"Raw SHA256 after: {raw_hash_after}",
    f"Raw file unchanged: {raw_unchanged}",
    "",
    "STATUS",
    "-" * 90,
    "BASE PREPROCESSING COMPLETE.",
    "The processed base dataset is suitable for Step 05 EDA.",
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {REPORT_PATH}")
print(f"Saved: {AUDIT_PATH}")


# =============================================================================
# 22. FINAL STATUS
# =============================================================================

section("STEP 04 COMPLETED SUCCESSFULLY")

print(
    "Base preprocessing completed without deleting observations or leaking target information.\n"
    f"Use this processed dataset from now on:\n{PROCESSED_CSV}\n\n"
    "Next step: Step 05 - Exploratory Data Analysis (EDA)."
)
