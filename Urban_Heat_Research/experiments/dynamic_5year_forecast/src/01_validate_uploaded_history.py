from pathlib import Path
import json
import hashlib
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

EXPERIMENT_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]

UPLOAD_DIR = EXPERIMENT_ROOT / "uploads"
OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"

REFERENCE_HISTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kaduwela_preprocessed_base.csv"
)

VALIDATED_OUTPUT = (
    OUTPUT_DIR
    / "01_validated_history.csv"
)

AUDIT_OUTPUT = (
    REPORT_DIR
    / "01_upload_validation_audit.csv"
)

SUMMARY_OUTPUT = (
    REPORT_DIR
    / "01_upload_validation_summary.json"
)

YEAR_OUTPUT = (
    REPORT_DIR
    / "01_uploaded_year_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_GRIDS = 2194
EXPECTED_START_YEAR = 2015
MINIMUM_LATEST_YEAR = 2025
FORECAST_HORIZON = 5

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

TEXT_COLUMNS = [
    "grid_id",
    "area_name",
]

NUMERIC_COLUMNS = [
    column
    for column in EXPECTED_COLUMNS
    if column not in TEXT_COLUMNS
]


# ============================================================
# HELPERS
# ============================================================

def section(title):

    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def audit(checks, name, passed, details):

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    checks.append(
        {
            "check": name,
            "status": status,
            "details": details,
        }
    )

    print(
        f"{status:4s} | "
        f"{name}: "
        f"{details}"
    )


# ============================================================
# PREPARE DIRECTORIES
# ============================================================

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FIND UPLOADED CSV
# ============================================================

section(
    "DYNAMIC 5-YEAR FORECAST - STEP 01 UPLOAD VALIDATION"
)

uploaded_files = sorted(
    UPLOAD_DIR.glob("*.csv")
)

if len(uploaded_files) == 0:

    raise FileNotFoundError(
        f"No CSV found inside:\n{UPLOAD_DIR}"
    )

if len(uploaded_files) > 1:

    print(
        "\nMore than one CSV exists in uploads:"
    )

    for file in uploaded_files:
        print(f"  {file.name}")

    raise RuntimeError(
        "Keep exactly one input CSV in uploads."
    )

INPUT_PATH = uploaded_files[0]

print(
    f"Project root      : "
    f"{PROJECT_ROOT}"
)

print(
    f"Experiment root   : "
    f"{EXPERIMENT_ROOT}"
)

print(
    f"Uploaded CSV      : "
    f"{INPUT_PATH}"
)

print(
    f"Reference history : "
    f"{REFERENCE_HISTORY}"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_PATH,
    dtype={
        "grid_id": str,
        "area_name": str,
    },
    low_memory=False,
)

checks = []

input_hash = sha256_file(
    INPUT_PATH
)


# ============================================================
# 1. SCHEMA VALIDATION
# ============================================================

section(
    "1. SCHEMA VALIDATION"
)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

missing_columns = [
    column
    for column in EXPECTED_COLUMNS
    if column not in df.columns
]

unexpected_columns = [
    column
    for column in df.columns
    if column not in EXPECTED_COLUMNS
]

audit(
    checks,
    "required_columns",
    len(missing_columns) == 0,
    f"missing={missing_columns}",
)

audit(
    checks,
    "unexpected_columns",
    len(unexpected_columns) == 0,
    f"unexpected={unexpected_columns}",
)

if (
    missing_columns
    or unexpected_columns
):

    raise RuntimeError(
        "Uploaded CSV schema does not match "
        "the canonical Urban Heat schema."
    )

df = df[
    EXPECTED_COLUMNS
].copy()


# ============================================================
# 2. TYPE / MISSING VALUE VALIDATION
# ============================================================

section(
    "2. TYPE AND VALUE VALIDATION"
)

for column in TEXT_COLUMNS:

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
    )


parse_failures = {}

for column in NUMERIC_COLUMNS:

    original_non_missing = (
        df[column]
        .notna()
    )

    converted = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    failed = (
        original_non_missing
        &
        converted.isna()
    )

    if failed.any():

        parse_failures[column] = int(
            failed.sum()
        )

    df[column] = converted


audit(
    checks,
    "numeric_parse",
    len(parse_failures) == 0,
    (
        "all numeric columns parsed"
        if not parse_failures
        else str(parse_failures)
    ),
)


missing_cells = int(
    df.isna()
    .sum()
    .sum()
)

audit(
    checks,
    "missing_cells",
    missing_cells == 0,
    f"count={missing_cells}",
)


numeric_values = (
    df[NUMERIC_COLUMNS]
    .to_numpy(
        dtype=float
    )
)

infinity_count = int(
    np.isinf(
        numeric_values
    ).sum()
)

audit(
    checks,
    "infinite_values",
    infinity_count == 0,
    f"count={infinity_count}",
)


if (
    parse_failures
    or missing_cells > 0
    or infinity_count > 0
):

    raise RuntimeError(
        "Uploaded data contains invalid values."
    )


year_values = (
    df["year"]
    .to_numpy(
        dtype=float
    )
)

non_integer_years = int(
    np.count_nonzero(
        np.abs(
            year_values
            -
            np.round(
                year_values
            )
        )
        >
        1e-12
    )
)

audit(
    checks,
    "integer_year_values",
    non_integer_years == 0,
    f"non_integer_rows={non_integer_years}",
)

if non_integer_years > 0:

    raise RuntimeError(
        "Year column contains non-integer values."
    )

df["year"] = (
    df["year"]
    .astype(int)
)


# ============================================================
# 3. DUPLICATE VALIDATION
# ============================================================

section(
    "3. DUPLICATE VALIDATION"
)

full_duplicates = int(
    df.duplicated()
    .sum()
)

duplicate_grid_year = int(
    df.duplicated(
        subset=[
            "grid_id",
            "year",
        ],
        keep=False,
    ).sum()
)

audit(
    checks,
    "full_duplicate_rows",
    full_duplicates == 0,
    f"count={full_duplicates}",
)

audit(
    checks,
    "duplicate_grid_year_rows",
    duplicate_grid_year == 0,
    f"rows={duplicate_grid_year}",
)


# ============================================================
# 4. YEAR STRUCTURE
# ============================================================

section(
    "4. OBSERVED YEAR STRUCTURE"
)

years = sorted(
    df["year"]
    .unique()
    .tolist()
)

first_year = int(
    years[0]
)

latest_year = int(
    years[-1]
)

expected_years = list(
    range(
        EXPECTED_START_YEAR,
        latest_year + 1,
    )
)

audit(
    checks,
    "start_year",
    first_year
    ==
    EXPECTED_START_YEAR,
    (
        f"actual={first_year}, "
        f"expected={EXPECTED_START_YEAR}"
    ),
)

audit(
    checks,
    "contiguous_years",
    years
    ==
    expected_years,
    f"years={years}",
)

audit(
    checks,
    "latest_year_supported",
    latest_year
    >=
    MINIMUM_LATEST_YEAR,
    (
        f"latest={latest_year}, "
        f"minimum={MINIMUM_LATEST_YEAR}"
    ),
)


forecast_years = list(
    range(
        latest_year + 1,
        latest_year
        +
        FORECAST_HORIZON
        +
        1,
    )
)

print(
    f"\nLatest observed year : "
    f"{latest_year}"
)

print(
    f"Next 5 years        : "
    f"{forecast_years}"
)


# ============================================================
# 5. GRID PANEL VALIDATION
# ============================================================

section(
    "5. GRID PANEL VALIDATION"
)

unique_grids = int(
    df["grid_id"]
    .nunique()
)

audit(
    checks,
    "unique_grid_count",
    unique_grids
    ==
    EXPECTED_GRIDS,
    (
        f"actual={unique_grids}, "
        f"expected={EXPECTED_GRIDS}"
    ),
)


year_summary = (
    df.groupby(
        "year"
    )
    .agg(
        rows=(
            "grid_id",
            "size",
        ),
        unique_grids=(
            "grid_id",
            "nunique",
        ),
        mean_lst_c=(
            "LST_C",
            "mean",
        ),
        median_lst_c=(
            "LST_C",
            "median",
        ),
    )
    .reset_index()
)


invalid_years = (
    year_summary[
        (
            year_summary["rows"]
            !=
            EXPECTED_GRIDS
        )
        |
        (
            year_summary["unique_grids"]
            !=
            EXPECTED_GRIDS
        )
    ]
)

audit(
    checks,
    "balanced_grid_year_panel",
    invalid_years.empty,
    (
        f"required rows/year="
        f"{EXPECTED_GRIDS}"
    ),
)


# ============================================================
# 6. CANONICAL GRID ALIGNMENT
# ============================================================

section(
    "6. CANONICAL GRID ALIGNMENT"
)

reference = pd.read_csv(
    REFERENCE_HISTORY,
    dtype={
        "grid_id": str,
        "area_name": str,
    },
    usecols=[
        "grid_id",
        "area_name",
    ],
)

reference["grid_id"] = (
    reference["grid_id"]
    .astype("string")
    .str.strip()
)

reference["area_name"] = (
    reference["area_name"]
    .astype("string")
    .str.strip()
)


reference_grids = set(
    reference[
        "grid_id"
    ].unique()
)

uploaded_grids = set(
    df[
        "grid_id"
    ].unique()
)


missing_reference_grids = (
    reference_grids
    -
    uploaded_grids
)

unexpected_grids = (
    uploaded_grids
    -
    reference_grids
)

audit(
    checks,
    "canonical_grid_set",
    (
        len(
            missing_reference_grids
        )
        ==
        0
        and
        len(
            unexpected_grids
        )
        ==
        0
    ),
    (
        f"missing={len(missing_reference_grids)}, "
        f"unexpected={len(unexpected_grids)}"
    ),
)


area_count_per_grid = (
    df.groupby(
        "grid_id"
    )[
        "area_name"
    ]
    .nunique()
)

inconsistent_areas = int(
    (
        area_count_per_grid
        !=
        1
    ).sum()
)

audit(
    checks,
    "one_area_per_grid",
    inconsistent_areas == 0,
    (
        f"inconsistent_grids="
        f"{inconsistent_areas}"
    ),
)


# ============================================================
# 7. FINAL DECISION
# ============================================================

section(
    "7. FINAL VALIDATION DECISION"
)

failed_checks = [
    item
    for item in checks
    if item["status"]
    ==
    "FAIL"
]


summary = {
    "input_file":
        str(INPUT_PATH),

    "input_sha256":
        input_hash,

    "rows":
        int(len(df)),

    "columns":
        int(len(df.columns)),

    "unique_grids":
        unique_grids,

    "observed_years":
        years,

    "latest_observed_year":
        latest_year,

    "forecast_horizon_years":
        FORECAST_HORIZON,

    "forecast_years":
        forecast_years,

    "failed_check_count":
        len(failed_checks),

    "status":
        (
            "PASS"
            if not failed_checks
            else "FAIL"
        ),
}


pd.DataFrame(
    checks
).to_csv(
    AUDIT_OUTPUT,
    index=False,
)

year_summary.to_csv(
    YEAR_OUTPUT,
    index=False,
)

with open(
    SUMMARY_OUTPUT,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=2,
    )


if failed_checks:

    print(
        "\nVALIDATION RESULT: FAIL"
    )

    print(
        f"Failed checks: "
        f"{len(failed_checks)}"
    )

    raise RuntimeError(
        "Uploaded history cannot be used "
        "for forecasting."
    )


# ============================================================
# SAVE VALIDATED HISTORY
# ============================================================

df = (
    df.sort_values(
        [
            "year",
            "grid_id",
        ]
    )
    .reset_index(
        drop=True
    )
)

df.to_csv(
    VALIDATED_OUTPUT,
    index=False,
)

validated_hash = sha256_file(
    VALIDATED_OUTPUT
)


print(
    "\nPASS: uploaded history "
    "is valid."
)

print(
    f"Observed coverage      : "
    f"{first_year}-{latest_year}"
)

print(
    f"Latest observed year   : "
    f"{latest_year}"
)

print(
    f"Automatic forecast     : "
    f"{forecast_years[0]}"
    f"-"
    f"{forecast_years[-1]}"
)

print(
    f"Rows                   : "
    f"{len(df):,}"
)

print(
    f"Unique grids           : "
    f"{unique_grids:,}"
)

print(
    f"Validated history      : "
    f"{VALIDATED_OUTPUT}"
)

print(
    f"Validated SHA256       : "
    f"{validated_hash}"
)

print(
    f"Audit                  : "
    f"{AUDIT_OUTPUT}"
)

print(
    f"Summary                : "
    f"{SUMMARY_OUTPUT}"
)


section(
    "STEP 01 COMPLETED SUCCESSFULLY"
)

print(
    "NEXT: refresh heat trajectories "
    "using the validated observed history."
)