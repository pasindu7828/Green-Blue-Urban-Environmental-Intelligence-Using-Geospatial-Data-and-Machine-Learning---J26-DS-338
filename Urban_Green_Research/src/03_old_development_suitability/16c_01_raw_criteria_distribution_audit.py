
"""
STEP 16C-A - RAW SUITABILITY CRITERIA DISTRIBUTION AUDIT

Component:
Urban Green Transition, Urbanization Pressure and
Conservation Intelligence

Purpose:
1. Analyze distributions of Step 16B raw criteria.
2. Assess missing, zero, negative and extreme values.
3. Verify the frozen Step 16B input and terrain provenance.
4. Save reproducible statistics and QA reports.

Important:
- Read-only access to frozen Step 16B inputs.
- No criterion selection or removal.
- No normalization or standardization.
- No AHP weights or suitability scores.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "development_suitability"
    / "raw_criteria"
    / "16b_raw_suitability_criteria_100m.csv"
)

STEP16B_QA = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "16b_raw_suitability_criteria_qa.csv"
)

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

STATS_CSV = (
    REPORT_DIR / "16c_a_raw_distribution_statistics.csv"
)

QA_CSV = (
    REPORT_DIR / "16c_a_raw_distribution_qa.csv"
)

SUMMARY_JSON = (
    REPORT_DIR / "16c_a_raw_distribution_summary.json"
)

EXPECTED_CELLS = 9240
EXPECTED_ZONAL = 9201
EXPECTED_FALLBACK = 39


# ============================================================
# 2. RAW CANDIDATE VARIABLES
# ============================================================

CRITERIA = [
    ("C1", "c1_major_road_distance_m", "m"),

    ("C2", "c2_road_density_250m_km_per_km2",
     "km/km2"),
    ("C2", "c2_road_density_500m_km_per_km2",
     "km/km2"),
    ("C2", "c2_road_density_1000m_km_per_km2",
     "km/km2"),

    ("C3", "c3_building_footprint_pct_cell", "%"),
    ("C3", "c3_context_building_pct_250m", "%"),
    ("C3", "c3_context_building_pct_500m", "%"),

    ("C4", "c4_slope_mean_deg", "degrees"),
    ("C5", "c5_elevation_mean_m", "m"),
]

VARIABLES = [item[1] for item in CRITERIA]

NONNEGATIVE_COLUMNS = [
    name for criterion, name, unit in CRITERIA
    if criterion != "C5"
]

PERCENT_COLUMNS = [
    name for criterion, name, unit in CRITERIA
    if unit == "%"
]


# ============================================================
# 3. INPUT VALIDATION
# ============================================================

print("=" * 90)
print("STEP 16C-A - RAW CRITERIA DISTRIBUTION AUDIT")
print("=" * 90)

if not INPUT_CSV.is_file():
    raise FileNotFoundError(INPUT_CSV)

if not STEP16B_QA.is_file():
    raise FileNotFoundError(STEP16B_QA)

df = pd.read_csv(INPUT_CSV)
previous_qa = pd.read_csv(STEP16B_QA)

required_columns = VARIABLES + [
    "grid_id",
    "terrain_extraction_method",
    "terrain_fallback_applied",
    "terrain_fallback_distance_m",
    "c4_slope_pixel_count",
    "c5_elevation_pixel_count",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

numeric_columns = list(dict.fromkeys(
    VARIABLES + [
        "terrain_fallback_distance_m",
        "c4_slope_pixel_count",
        "c5_elevation_pixel_count",
    ]
))

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

input_hash = hashlib.sha256(
    INPUT_CSV.read_bytes()
).hexdigest()

print(f"\nInput dataset: {INPUT_CSV.name}")
print(f"Grid cells: {len(df):,}")
print(f"Candidate variables: {len(VARIABLES)}")


# ============================================================
# 4. DISTRIBUTION STATISTICS
# ============================================================

records = []

for criterion, variable, unit in CRITERIA:

    values = df[variable]
    valid_values = values.dropna()

    q1 = valid_values.quantile(0.25)
    q3 = valid_values.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    # Exploratory outlier indicators only.
    # No observations are removed.
    outlier_count = int(
        (
            (valid_values < lower_fence)
            | (valid_values > upper_fence)
        ).sum()
    )

    records.append({
        "criterion": criterion,
        "variable": variable,
        "unit": unit,
        "total_rows": len(df),
        "valid_count": int(values.notna().sum()),
        "missing_count": int(values.isna().sum()),
        "mean": float(valid_values.mean()),
        "std": float(valid_values.std()),
        "min": float(valid_values.min()),
        "p01": float(valid_values.quantile(0.01)),
        "p05": float(valid_values.quantile(0.05)),
        "p25": float(q1),
        "median": float(valid_values.median()),
        "p75": float(q3),
        "p95": float(valid_values.quantile(0.95)),
        "p99": float(valid_values.quantile(0.99)),
        "max": float(valid_values.max()),
        "iqr": float(iqr),
        "skewness": float(valid_values.skew()),
        "zero_count": int((values == 0).sum()),
        "zero_pct": float((values == 0).mean() * 100),
        "negative_count": int((values < 0).sum()),
        "negative_pct": float(
            (values < 0).mean() * 100
        ),
        "iqr_outlier_count": outlier_count,
        "iqr_outlier_pct": float(
            outlier_count / len(valid_values) * 100
        ),
    })

stats_df = pd.DataFrame(records)


# ============================================================
# 5. TERRAIN PROVENANCE CHECK
# ============================================================

method_counts = (
    df["terrain_extraction_method"]
    .value_counts()
    .to_dict()
)

zonal_count = int(
    method_counts.get("ZONAL_GEDTM", 0)
)

fallback_count = int(
    method_counts.get(
        "NEAREST_VALID_GEDTM_PROXY", 0
    )
)

flagged = (
    df["terrain_fallback_applied"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1"])
)

proxy_method = (
    df["terrain_extraction_method"]
    == "NEAREST_VALID_GEDTM_PROXY"
)

proxy = df.loc[proxy_method]

fallback_max_distance = (
    float(proxy["terrain_fallback_distance_m"].max())
    if len(proxy) > 0
    else None
)


# ============================================================
# 6. QA VALIDATION
# ============================================================

checks = []


def add_check(name, passed, details):
    checks.append({
        "check": name,
        "passed": bool(passed),
        "details": str(details),
    })


add_check(
    "Expected 9240 grid cells",
    len(df) == EXPECTED_CELLS,
    f"Actual: {len(df)}",
)

add_check(
    "Unique non-null grid IDs",
    df["grid_id"].is_unique
    and df["grid_id"].notna().all(),
    "Each record must identify one grid cell",
)

add_check(
    "Previous Step 16B QA PASS",
    not previous_qa.empty
    and previous_qa.iloc[0][
        "overall_step16b_status"
    ] == "PASS",
    "Input must originate from accepted Step 16B",
)

add_check(
    "All candidate values finite",
    bool(
        np.isfinite(
            df[VARIABLES].to_numpy(dtype=float)
        ).all()
    ),
    "All nine raw variables must be complete",
)

add_check(
    "Nonnegative C1-C4 values",
    bool(
        df[NONNEGATIVE_COLUMNS].ge(0).all().all()
    ),
    "Negative elevation is allowed for C5",
)

add_check(
    "Building percentages in 0-100 range",
    bool(
        df[PERCENT_COLUMNS]
        .ge(0).all().all()
        and
        df[PERCENT_COLUMNS]
        .le(100).all().all()
    ),
    "Applies to C3 percentage variables",
)

add_check(
    "Expected GEDTM terrain sources",
    zonal_count == EXPECTED_ZONAL
    and fallback_count == EXPECTED_FALLBACK
    and zonal_count + fallback_count == len(df),
    f"Zonal: {zonal_count}; proxy: {fallback_count}",
)

add_check(
    "Fallback flags agree with method",
    bool((flagged == proxy_method).all()),
    "Fallback flags must identify exactly the proxy cells",
)

add_check(
    "Fallback within 30 metres",
    fallback_max_distance is not None
    and proxy["terrain_fallback_distance_m"]
    .notna().all()
    and bool(
        proxy["terrain_fallback_distance_m"]
        .le(30).all()
    ),
    f"Maximum observed distance: {fallback_max_distance}",
)

add_check(
    "Proxy cells retain zero zonal pixel counts",
    bool(
        proxy["c4_slope_pixel_count"].eq(0).all()
        and proxy["c5_elevation_pixel_count"].eq(0).all()
    ),
    "Proxy estimates must not be described as zonal pixels",
)

qa_df = pd.DataFrame(checks)

overall_pass = bool(qa_df["passed"].all())

status = "PASS" if overall_pass else "REVIEW_REQUIRED"


# ============================================================
# 7. SAVE REPRODUCIBLE REPORTS
# ============================================================

REPORT_DIR.mkdir(parents=True, exist_ok=True)

stats_df.to_csv(STATS_CSV, index=False)
qa_df.to_csv(QA_CSV, index=False)

summary = {
    "step": "16C-A",
    "name": "Raw Suitability Criteria Distribution Audit",
    "status": status,
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "input_csv": INPUT_CSV.relative_to(
        PROJECT_ROOT
    ).as_posix(),
    "input_sha256": input_hash,
    "grid_cells": int(len(df)),
    "candidate_variables": len(VARIABLES),
    "criterion_groups": ["C1", "C2", "C3", "C4", "C5"],
    "terrain_provenance": {
        "zonal_gedtm": zonal_count,
        "nearest_valid_gedtm_proxy": fallback_count,
        "maximum_proxy_distance_m": fallback_max_distance,
    },
    "qa_checks_total": len(checks),
    "qa_checks_passed": int(qa_df["passed"].sum()),
    "methodological_restrictions": {
        "normalization_performed": False,
        "standardization_performed": False,
        "criteria_selected_or_removed": False,
        "ahp_weights_assigned": False,
        "suitability_score_calculated": False,
    },
    "notes": [
        "Zero concentrations and skewness are descriptive.",
        "IQR outliers are flagged, not removed.",
        "Negative GEDTM elevation is permitted.",
        "C5 criterion direction remains undecided.",
        "The 39 terrain proxy values are not in-polygon zonal measurements.",
        "Step 16B inputs are not modified.",
    ],
    "outputs": {
        "statistics_csv": STATS_CSV.name,
        "qa_csv": QA_CSV.name,
        "summary_json": SUMMARY_JSON.name,
    },
}

with open(
    SUMMARY_JSON, "w", encoding="utf-8"
) as file:
    json.dump(
        summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 8. TERMINAL SUMMARY
# ============================================================

print("\n=== DISTRIBUTION STATISTICS ===")

display_columns = [
    "criterion",
    "variable",
    "mean",
    "median",
    "std",
    "min",
    "p95",
    "max",
    "zero_pct",
    "skewness",
    "iqr_outlier_pct",
]

print(
    stats_df[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\n=== STEP 16C-A QA ===")

for _, row in qa_df.iterrows():
    result = "PASS" if row["passed"] else "FAIL"
    print(f"{row['check']}: {result}")

print("\n=== TERRAIN SOURCE ===")
print(f"Original zonal cells : {zonal_count}")
print(f"GEDTM proxy cells    : {fallback_count}")

print("\n=== GENERATED REPORTS ===")
for path in [STATS_CSV, QA_CSV, SUMMARY_JSON]:
    print(path.relative_to(PROJECT_ROOT))

print("\nOverall Step 16C-A status:", status)
print("No criterion selection or standardization performed.")

if not overall_pass:
    raise SystemExit(
        "Step 16C-A requires review. Inspect the QA CSV."
    )
