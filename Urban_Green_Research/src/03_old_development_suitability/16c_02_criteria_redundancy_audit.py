
"""
STEP 16C-B - CRITERIA CORRELATION AND REDUNDANCY AUDIT

Research component:
Urban Green Transition, Urbanization Pressure and
Conservation Intelligence

Purpose:
1. Compare C2 road-density alternatives.
2. Compare C3 building-context measurements.
3. Evaluate correlations across C1-C5.
4. Flag highly correlated pairs for methodological review.
5. Save reproducible matrices, pairwise results and QA.

Rules:
- Use frozen Step 16B raw criteria.
- Verify the completed Step 16C-A input.
- No automatic criterion removal.
- No AHP weighting or standardization.
- Correlation does not establish causation.
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

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

PREVIOUS_SUMMARY = (
    REPORT_DIR / "16c_a_raw_distribution_summary.json"
)

PEARSON_CSV = (
    REPORT_DIR / "16c_b_pearson_correlation_matrix.csv"
)

SPEARMAN_CSV = (
    REPORT_DIR / "16c_b_spearman_correlation_matrix.csv"
)

PAIRS_CSV = (
    REPORT_DIR / "16c_b_correlation_pairwise_review.csv"
)

QA_CSV = (
    REPORT_DIR / "16c_b_correlation_qa.csv"
)

SUMMARY_JSON = (
    REPORT_DIR / "16c_b_correlation_summary.json"
)

EXPECTED_CELLS = 9240
HIGH_CORRELATION_THRESHOLD = 0.85


# ============================================================
# 2. CRITERION GROUPS
# ============================================================

C1 = [
    "c1_major_road_distance_m"
]

C2 = [
    "c2_road_density_250m_km_per_km2",
    "c2_road_density_500m_km_per_km2",
    "c2_road_density_1000m_km_per_km2",
]

C3 = [
    "c3_building_footprint_pct_cell",
    "c3_context_building_pct_250m",
    "c3_context_building_pct_500m",
]

C4 = [
    "c4_slope_mean_deg"
]

C5 = [
    "c5_elevation_mean_m"
]

GROUPS = {
    "C1": C1,
    "C2": C2,
    "C3": C3,
    "C4": C4,
    "C5": C5,
}

VARIABLES = [
    variable
    for group in GROUPS.values()
    for variable in group
]

VARIABLE_GROUP = {
    variable: criterion
    for criterion, columns in GROUPS.items()
    for variable in columns
}


# ============================================================
# 3. LOAD AND VERIFY INPUT
# ============================================================

print("=" * 95)
print("STEP 16C-B - CRITERIA CORRELATION AND REDUNDANCY AUDIT")
print("=" * 95)

if not INPUT_CSV.is_file():
    raise FileNotFoundError(INPUT_CSV)

if not PREVIOUS_SUMMARY.is_file():
    raise FileNotFoundError(PREVIOUS_SUMMARY)

df = pd.read_csv(INPUT_CSV)

with open(
    PREVIOUS_SUMMARY, encoding="utf-8"
) as file:
    previous_summary = json.load(file)

input_hash = hashlib.sha256(
    INPUT_CSV.read_bytes()
).hexdigest()

missing_columns = [
    column
    for column in ["grid_id"] + VARIABLES
    if column not in df.columns
]

if missing_columns:
    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

data = df[VARIABLES].apply(
    pd.to_numeric,
    errors="coerce"
)

print(f"\nInput: {INPUT_CSV.name}")
print(f"Grid cells: {len(df):,}")
print(f"Candidate measurements: {len(VARIABLES)}")


# ============================================================
# 4. INPUT QUALITY CHECKS
# ============================================================

checks = []


def add_check(name, passed, details):
    checks.append({
        "check": name,
        "passed": bool(passed),
        "details": str(details),
    })


add_check(
    "Expected grid count",
    len(df) == EXPECTED_CELLS,
    f"Actual: {len(df)}",
)

add_check(
    "Unique and valid grid IDs",
    df["grid_id"].notna().all()
    and df["grid_id"].is_unique,
    "One record per 100m grid cell",
)

add_check(
    "Step 16C-A status PASS",
    previous_summary.get("status") == "PASS",
    "Previous distribution audit must have passed",
)

add_check(
    "Frozen input checksum agrees with Step 16C-A",
    input_hash == previous_summary.get("input_sha256"),
    "Prevents silent changes to the analyzed input",
)

add_check(
    "All nine measurements finite",
    bool(
        np.isfinite(
            data.to_numpy(dtype=float)
        ).all()
    ),
    "Missing or infinite values are not permitted",
)

add_check(
    "All measurements nonconstant",
    bool(
        (data.nunique(dropna=True) > 1).all()
    ),
    "Constant columns cannot yield valid correlations",
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

initial_qa = pd.DataFrame(checks)

if not initial_qa["passed"].all():
    initial_qa.to_csv(QA_CSV, index=False)
    raise RuntimeError(
        "Input validation failed. Review: "
        + str(QA_CSV)
    )


# ============================================================
# 5. CORRELATION MATRICES
# ============================================================

pearson = data.corr(
    method="pearson"
)

spearman = data.corr(
    method="spearman"
)

print("\n=== C2: ROAD-DENSITY PEARSON ===")
print(
    pearson.loc[C2, C2]
    .round(4)
    .to_string()
)

print("\n=== C2: ROAD-DENSITY SPEARMAN ===")
print(
    spearman.loc[C2, C2]
    .round(4)
    .to_string()
)

print("\n=== C3: BUILDING-CONTEXT PEARSON ===")
print(
    pearson.loc[C3, C3]
    .round(4)
    .to_string()
)

print("\n=== C3: BUILDING-CONTEXT SPEARMAN ===")
print(
    spearman.loc[C3, C3]
    .round(4)
    .to_string()
)


# ============================================================
# 6. PAIRWISE REDUNDANCY REVIEW
# ============================================================

pair_records = []

for i, variable_a in enumerate(VARIABLES):

    for variable_b in VARIABLES[i + 1:]:

        group_a = VARIABLE_GROUP[variable_a]
        group_b = VARIABLE_GROUP[variable_b]

        pearson_r = float(
            pearson.loc[variable_a, variable_b]
        )

        spearman_rho = float(
            spearman.loc[variable_a, variable_b]
        )

        if group_a == group_b == "C2":
            comparison_scope = "C2_RADIUS_ALTERNATIVES"

        elif group_a == group_b == "C3":
            comparison_scope = "C3_BUILDING_MEASUREMENTS"

        else:
            comparison_scope = "CROSS_CRITERION"

        high_pearson = (
            abs(pearson_r)
            >= HIGH_CORRELATION_THRESHOLD
        )

        high_spearman = (
            abs(spearman_rho)
            >= HIGH_CORRELATION_THRESHOLD
        )

        pair_records.append({
            "criterion_a": group_a,
            "criterion_b": group_b,
            "variable_a": variable_a,
            "variable_b": variable_b,
            "comparison_scope": comparison_scope,
            "pearson_r": pearson_r,
            "spearman_rho": spearman_rho,
            "abs_pearson": abs(pearson_r),
            "abs_spearman": abs(spearman_rho),
            "high_pearson_flag": bool(high_pearson),
            "high_spearman_flag": bool(high_spearman),
            "review_flag": bool(
                high_pearson or high_spearman
            ),
            "decision": "REVIEW_ONLY",
        })

pairs_df = pd.DataFrame(pair_records)

pairs_df = pairs_df.sort_values(
    ["review_flag", "abs_spearman"],
    ascending=[False, False],
).reset_index(drop=True)

flagged_pairs = pairs_df[
    pairs_df["review_flag"]
].copy()

expected_pairs = (
    len(VARIABLES)
    * (len(VARIABLES) - 1)
    // 2
)


# ============================================================
# 7. CORRELATION QA
# ============================================================

def matrix_is_valid(matrix):
    values = matrix.to_numpy(dtype=float)

    return bool(
        np.isfinite(values).all()
        and np.allclose(
            values,
            values.T,
            atol=1e-10,
        )
        and np.allclose(
            np.diag(values),
            1.0,
            atol=1e-10,
        )
        and (np.abs(values) <= 1.0000000001).all()
    )


add_check(
    "Pearson matrix valid",
    matrix_is_valid(pearson),
    "Finite, symmetric, diagonal equal to one",
)

add_check(
    "Spearman matrix valid",
    matrix_is_valid(spearman),
    "Finite, symmetric, diagonal equal to one",
)

add_check(
    "Complete pairwise comparison",
    len(pairs_df) == expected_pairs,
    f"Expected: {expected_pairs}; actual: {len(pairs_df)}",
)

add_check(
    "No automatic criterion removal",
    bool(
        pairs_df["decision"].eq(
            "REVIEW_ONLY"
        ).all()
    ),
    "Correlation flags require methodological review",
)

qa_df = pd.DataFrame(checks)

overall_pass = bool(
    qa_df["passed"].all()
)

status = (
    "PASS"
    if overall_pass
    else "REVIEW_REQUIRED"
)

qa_df.to_csv(
    QA_CSV,
    index=False,
)

if not overall_pass:
    raise RuntimeError(
        "Correlation QA failed. Review " + str(QA_CSV)
    )


# ============================================================
# 8. SAVE REPRODUCIBLE OUTPUTS
# ============================================================

pearson.to_csv(
    PEARSON_CSV,
    index_label="variable",
)

spearman.to_csv(
    SPEARMAN_CSV,
    index_label="variable",
)

pairs_df.to_csv(
    PAIRS_CSV,
    index=False,
)

summary = {
    "step": "16C-B",
    "name": "Criteria Correlation and Redundancy Audit",
    "status": status,
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "input_csv": INPUT_CSV.relative_to(
        PROJECT_ROOT
    ).as_posix(),
    "input_sha256": input_hash,
    "grid_cells": int(len(df)),
    "measurement_count": len(VARIABLES),
    "pair_count": len(pairs_df),
    "high_correlation_review_threshold":
        HIGH_CORRELATION_THRESHOLD,
    "flagged_pair_count": len(flagged_pairs),
    "flagged_pairs_by_scope": {
        scope: int(
            (flagged_pairs["comparison_scope"] == scope).sum()
        )
        for scope in [
            "C2_RADIUS_ALTERNATIVES",
            "C3_BUILDING_MEASUREMENTS",
            "CROSS_CRITERION",
        ]
    },
    "correlation_methods": [
        "Pearson",
        "Spearman",
    ],
    "qa_checks_total": len(checks),
    "qa_checks_passed": int(qa_df["passed"].sum()),
    "methodological_notes": [
        "A correlation flag is not an exclusion decision.",
        "The 0.85 threshold is an exploratory review rule.",
        "C3 cell occupation and neighbourhood context are conceptually different.",
        "Spatial dependence means grid cells should not be assumed statistically independent.",
        "No significance tests or causal conclusions are made.",
        "No AHP weights or suitability scores are calculated.",
        "Criteria selection and standardization are deferred.",
    ],
    "outputs": {
        "pearson_matrix": PEARSON_CSV.name,
        "spearman_matrix": SPEARMAN_CSV.name,
        "pairwise_review": PAIRS_CSV.name,
        "qa_report": QA_CSV.name,
        "summary": SUMMARY_JSON.name,
    },
}

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 9. TERMINAL RESULTS
# ============================================================

print("\n=== HIGH-CORRELATION PAIRS FOR REVIEW ===")

if flagged_pairs.empty:
    print("No pairs exceeded the review threshold.")

else:
    print(
        flagged_pairs[
            [
                "variable_a",
                "variable_b",
                "pearson_r",
                "spearman_rho",
                "comparison_scope",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

print("\n=== TOP 10 PAIRS BY ABSOLUTE SPEARMAN ===")

print(
    pairs_df.sort_values(
        "abs_spearman",
        ascending=False,
    )[
        [
            "variable_a",
            "variable_b",
            "spearman_rho",
            "review_flag",
        ]
    ]
    .head(10)
    .round(4)
    .to_string(index=False)
)

print("\n=== STEP 16C-B QA ===")

for _, row in qa_df.iterrows():
    result = (
        "PASS"
        if row["passed"]
        else "FAIL"
    )

    print(f"{row['check']}: {result}")

print("\n=== GENERATED REPORTS ===")

for path in [
    PEARSON_CSV,
    SPEARMAN_CSV,
    PAIRS_CSV,
    QA_CSV,
    SUMMARY_JSON,
]:
    print(path.relative_to(PROJECT_ROOT))

print("\nGrid cells:", len(df))
print("Pairwise comparisons:", len(pairs_df))
print("Flagged pairs:", len(flagged_pairs))
print("Overall Step 16C-B status:", status)

print("\nNo criterion removed.")
print("No normalization or AHP weighting performed.")
