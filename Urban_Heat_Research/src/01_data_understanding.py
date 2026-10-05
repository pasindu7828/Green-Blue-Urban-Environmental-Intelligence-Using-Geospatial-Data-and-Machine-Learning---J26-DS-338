"""
STEP 01 - DATA UNDERSTANDING
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Understand the raw master dataset before any cleaning, transformation,
feature engineering, splitting, or model training.

IMPORTANT:
- This script DOES NOT modify the raw dataset.
- It only reads the data, checks its structure, and saves audit reports.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "kaduwela_master_raw.csv"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OVERVIEW_REPORT_PATH = REPORT_DIR / "01_dataset_overview.txt"
COLUMN_SUMMARY_PATH = REPORT_DIR / "01_column_summary.csv"
MISSING_VALUES_PATH = REPORT_DIR / "01_missing_values.csv"
NUMERIC_STATS_PATH = REPORT_DIR / "01_numeric_statistics.csv"
YEAR_SUMMARY_PATH = REPORT_DIR / "01_year_summary.csv"
GRID_COVERAGE_PATH = REPORT_DIR / "01_grid_coverage_by_year.csv"
AREA_SUMMARY_PATH = REPORT_DIR / "01_area_summary.csv"
GRID_COMPLETENESS_PATH = REPORT_DIR / "01_grid_temporal_completeness.csv"
GRID_AREA_CONSISTENCY_PATH = REPORT_DIR / "01_grid_area_consistency.csv"


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def section(title: str) -> None:
    """Print a clear section heading."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def safe_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text safely."""
    path.write_text(text, encoding="utf-8")


# =============================================================================
# 3. LOAD RAW DATA
# =============================================================================

section("STEP 01 - DATA UNDERSTANDING")

print(f"Project root : {PROJECT_ROOT}")
print(f"Input file   : {RAW_DATA_PATH}")

if not RAW_DATA_PATH.exists():
    print("\nERROR: Raw master CSV was not found.")
    print("Expected location:")
    print(RAW_DATA_PATH)
    sys.exit(1)

try:
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
except Exception as exc:
    print(f"\nERROR: Could not read the CSV.\n{exc}")
    sys.exit(1)

if df.empty:
    print("\nERROR: The dataset is empty.")
    sys.exit(1)

print("\nDataset loaded successfully.")


# =============================================================================
# 4. BASIC DATASET STRUCTURE
# =============================================================================

section("1. BASIC DATASET STRUCTURE")

n_rows, n_cols = df.shape

print(f"Rows                 : {n_rows:,}")
print(f"Columns              : {n_cols:,}")
print(f"Memory usage         : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nColumn names:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:>2}. {col}")

print("\nData types:")
print(df.dtypes.to_string())


# =============================================================================
# 5. SAMPLE RECORDS
# =============================================================================

section("2. SAMPLE RECORDS")

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

print("\nLast 5 rows:")
print(df.tail().to_string(index=False))


# =============================================================================
# 6. COLUMN-LEVEL SUMMARY
# =============================================================================

section("3. COLUMN-LEVEL SUMMARY")

column_summary_rows = []

for col in df.columns:
    series = df[col]

    column_summary_rows.append(
        {
            "column": col,
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_percent": round(series.isna().mean() * 100, 4),
            "unique_count": int(series.nunique(dropna=True)),
            "duplicate_value_count": int(series.notna().sum() - series.nunique(dropna=True)),
        }
    )

column_summary = pd.DataFrame(column_summary_rows)
column_summary.to_csv(COLUMN_SUMMARY_PATH, index=False)

print(column_summary.to_string(index=False))


# =============================================================================
# 7. MISSING VALUES
# =============================================================================

section("4. MISSING VALUES")

missing_summary = pd.DataFrame(
    {
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percent": (df.isna().mean().values * 100),
    }
)

missing_summary["missing_percent"] = missing_summary["missing_percent"].round(4)
missing_summary = missing_summary.sort_values(
    by=["missing_count", "column"],
    ascending=[False, True]
)

missing_summary.to_csv(MISSING_VALUES_PATH, index=False)

total_missing = int(df.isna().sum().sum())

print(f"Total missing cells: {total_missing:,}")
print(missing_summary.to_string(index=False))


# =============================================================================
# 8. DUPLICATES AND KEY STRUCTURE
# =============================================================================

section("5. DUPLICATES AND GRID-YEAR KEY CHECK")

full_duplicate_count = int(df.duplicated().sum())
print(f"Fully duplicated rows: {full_duplicate_count:,}")

grid_year_duplicate_count = None

if {"grid_id", "year"}.issubset(df.columns):
    grid_year_duplicate_count = int(
        df.duplicated(subset=["grid_id", "year"], keep=False).sum()
    )

    unique_grid_year_pairs = int(
        df[["grid_id", "year"]].drop_duplicates().shape[0]
    )

    print(f"Unique grid_id + year pairs           : {unique_grid_year_pairs:,}")
    print(f"Rows involved in duplicate grid-years: {grid_year_duplicate_count:,}")
else:
    print("grid_id and/or year column is missing, so the grid-year key cannot be checked.")


# =============================================================================
# 9. TEMPORAL COVERAGE
# =============================================================================

section("6. TEMPORAL COVERAGE")

year_summary = pd.DataFrame()
grid_coverage_by_year = pd.DataFrame()

if "year" in df.columns:
    valid_years = pd.to_numeric(df["year"], errors="coerce")
    unique_years = sorted(valid_years.dropna().astype(int).unique().tolist())

    print(f"Available years : {unique_years}")

    if unique_years:
        print(f"First year      : {min(unique_years)}")
        print(f"Last year       : {max(unique_years)}")
        print(f"Number of years : {len(unique_years)}")

    year_summary = (
        df.groupby("year", dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values("year")
    )
    year_summary.to_csv(YEAR_SUMMARY_PATH, index=False)

    print("\nRows per year:")
    print(year_summary.to_string(index=False))

    if "grid_id" in df.columns:
        grid_coverage_by_year = (
            df.groupby("year", dropna=False)["grid_id"]
            .nunique()
            .reset_index(name="unique_grid_count")
            .sort_values("year")
        )
        grid_coverage_by_year.to_csv(GRID_COVERAGE_PATH, index=False)

        print("\nUnique grids per year:")
        print(grid_coverage_by_year.to_string(index=False))
else:
    print("Column 'year' was not found.")


# =============================================================================
# 10. SPATIAL UNIT / GRID COVERAGE
# =============================================================================

section("7. GRID COVERAGE")

unique_grids = None
grid_completeness = pd.DataFrame()

if "grid_id" in df.columns:
    unique_grids = int(df["grid_id"].nunique(dropna=True))
    print(f"Unique grid cells: {unique_grids:,}")

    if "year" in df.columns:
        grid_completeness = (
            df.groupby("grid_id")["year"]
            .nunique()
            .reset_index(name="observed_year_count")
        )

        expected_year_count = int(df["year"].nunique(dropna=True))
        grid_completeness["expected_year_count"] = expected_year_count
        grid_completeness["is_temporally_complete"] = (
            grid_completeness["observed_year_count"] == expected_year_count
        )

        grid_completeness.to_csv(GRID_COMPLETENESS_PATH, index=False)

        complete_grids = int(grid_completeness["is_temporally_complete"].sum())
        incomplete_grids = int((~grid_completeness["is_temporally_complete"]).sum())

        print(f"Expected years per grid : {expected_year_count}")
        print(f"Complete grids           : {complete_grids:,}")
        print(f"Incomplete grids         : {incomplete_grids:,}")

        expected_rows = unique_grids * expected_year_count
        print(f"Expected rows from grid × year structure: {expected_rows:,}")
        print(f"Actual rows                           : {len(df):,}")
        print(f"Difference                            : {len(df) - expected_rows:,}")
else:
    print("Column 'grid_id' was not found.")


# =============================================================================
# 11. AREA / GN DIVISION COVERAGE
# =============================================================================

section("8. AREA / GN DIVISION COVERAGE")

area_summary = pd.DataFrame()
grid_area_consistency = pd.DataFrame()

if "area_name" in df.columns:
    unique_areas = int(df["area_name"].nunique(dropna=True))

    print(f"Unique area names : {unique_areas:,}")
    print(f"Missing area names: {int(df['area_name'].isna().sum()):,}")

    area_summary = (
        df.groupby("area_name", dropna=False)
        .agg(
            row_count=("grid_id", "size") if "grid_id" in df.columns else ("area_name", "size"),
            unique_grid_count=("grid_id", "nunique") if "grid_id" in df.columns else ("area_name", "size"),
        )
        .reset_index()
        .sort_values(["unique_grid_count", "area_name"], ascending=[False, True])
    )

    area_summary.to_csv(AREA_SUMMARY_PATH, index=False)

    print("\nArea summary:")
    print(area_summary.to_string(index=False))

    # A fixed grid should normally keep the same area assignment through all years.
    if "grid_id" in df.columns:
        grid_area_consistency = (
            df.groupby("grid_id")["area_name"]
            .nunique(dropna=True)
            .reset_index(name="unique_area_count")
        )

        grid_area_consistency["area_assignment_consistent"] = (
            grid_area_consistency["unique_area_count"] <= 1
        )

        grid_area_consistency.to_csv(GRID_AREA_CONSISTENCY_PATH, index=False)

        inconsistent_grids = int(
            (~grid_area_consistency["area_assignment_consistent"]).sum()
        )

        print(
            f"\nGrids assigned to more than one area across years: "
            f"{inconsistent_grids:,}"
        )
else:
    print("Column 'area_name' was not found.")


# =============================================================================
# 12. NUMERIC DATA UNDERSTANDING
# =============================================================================

section("9. NUMERIC VARIABLE SUMMARY")

numeric_df = df.select_dtypes(include=[np.number])

if not numeric_df.empty:
    numeric_stats = numeric_df.describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T

    numeric_stats["missing_count"] = numeric_df.isna().sum()
    numeric_stats["unique_count"] = numeric_df.nunique(dropna=True)

    # Count positive and negative infinity separately.
    numeric_array = numeric_df.to_numpy(dtype=float)
    pos_inf_counts = np.isposinf(numeric_array).sum(axis=0)
    neg_inf_counts = np.isneginf(numeric_array).sum(axis=0)

    numeric_stats["positive_infinity_count"] = pos_inf_counts
    numeric_stats["negative_infinity_count"] = neg_inf_counts

    numeric_stats.to_csv(NUMERIC_STATS_PATH)

    print(numeric_stats.to_string())
else:
    numeric_stats = pd.DataFrame()
    print("No numeric columns were detected.")


# =============================================================================
# 13. TARGET VARIABLE QUICK VIEW
# =============================================================================

section("10. TARGET VARIABLE QUICK VIEW")

if "LST_C" in df.columns:
    lst = pd.to_numeric(df["LST_C"], errors="coerce")

    print(f"Target variable : LST_C")
    print(f"Non-null values : {lst.notna().sum():,}")
    print(f"Missing values  : {lst.isna().sum():,}")
    print(f"Minimum         : {lst.min():.4f}")
    print(f"Maximum         : {lst.max():.4f}")
    print(f"Mean            : {lst.mean():.4f}")
    print(f"Median          : {lst.median():.4f}")
    print(f"Std. deviation  : {lst.std():.4f}")
else:
    print("Target column 'LST_C' was not found.")


# =============================================================================
# 14. CONSTANT / VERY LOW-VARIATION COLUMNS
# =============================================================================

section("11. CONSTANT / LOW-UNIQUENESS COLUMNS")

constant_columns = [
    col for col in df.columns if df[col].nunique(dropna=False) <= 1
]

low_uniqueness_columns = [
    col
    for col in df.columns
    if 1 < df[col].nunique(dropna=False) <= 10
]

print(f"Constant columns: {constant_columns}")
print(f"Columns with 2-10 unique values: {low_uniqueness_columns}")


# =============================================================================
# 15. BUILD TEXT AUDIT REPORT
# =============================================================================

section("12. SAVING DATA-UNDERSTANDING REPORTS")

report_lines = [
    "STEP 01 - DATA UNDERSTANDING REPORT",
    "=" * 80,
    "",
    f"Input file: {RAW_DATA_PATH}",
    f"Rows: {n_rows:,}",
    f"Columns: {n_cols:,}",
    f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
    "",
    "COLUMNS",
    "-" * 80,
]

for i, col in enumerate(df.columns, start=1):
    report_lines.append(f"{i:>2}. {col} [{df[col].dtype}]")

report_lines.extend(
    [
        "",
        "CORE STRUCTURE",
        "-" * 80,
        f"Fully duplicated rows: {full_duplicate_count:,}",
        f"Total missing cells: {total_missing:,}",
        f"Constant columns: {constant_columns}",
        f"Low-uniqueness columns (2-10 values): {low_uniqueness_columns}",
    ]
)

if unique_grids is not None:
    report_lines.append(f"Unique grid cells: {unique_grids:,}")

if "year" in df.columns:
    unique_years_report = sorted(
        pd.to_numeric(df["year"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    report_lines.append(f"Years: {unique_years_report}")

if "area_name" in df.columns:
    report_lines.append(
        f"Unique area names: {df['area_name'].nunique(dropna=True):,}"
    )

if grid_year_duplicate_count is not None:
    report_lines.append(
        f"Rows involved in duplicate grid_id + year keys: "
        f"{grid_year_duplicate_count:,}"
    )

if "LST_C" in df.columns:
    lst = pd.to_numeric(df["LST_C"], errors="coerce")
    report_lines.extend(
        [
            "",
            "TARGET - LST_C",
            "-" * 80,
            f"Minimum: {lst.min():.6f}",
            f"Maximum: {lst.max():.6f}",
            f"Mean: {lst.mean():.6f}",
            f"Median: {lst.median():.6f}",
            f"Std. deviation: {lst.std():.6f}",
        ]
    )

report_lines.extend(
    [
        "",
        "IMPORTANT",
        "-" * 80,
        "No cleaning, imputation, scaling, feature engineering, splitting,",
        "outlier removal, or model training was performed in Step 01.",
        "The raw dataset was read only.",
    ]
)

safe_write_text(OVERVIEW_REPORT_PATH, "\n".join(report_lines))


# =============================================================================
# 16. FINAL STATUS
# =============================================================================

print(f"Saved: {OVERVIEW_REPORT_PATH}")
print(f"Saved: {COLUMN_SUMMARY_PATH}")
print(f"Saved: {MISSING_VALUES_PATH}")
print(f"Saved: {NUMERIC_STATS_PATH}")
print(f"Saved: {YEAR_SUMMARY_PATH}")
print(f"Saved: {GRID_COVERAGE_PATH}")
print(f"Saved: {AREA_SUMMARY_PATH}")
print(f"Saved: {GRID_COMPLETENESS_PATH}")
print(f"Saved: {GRID_AREA_CONSISTENCY_PATH}")

section("STEP 01 COMPLETED SUCCESSFULLY")

print(
    "Next step: review these results first. "
    "Do NOT start cleaning or model training until the Step 01 findings are checked."
)
