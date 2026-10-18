
"""
STEP 16C-G - SCORING METHOD SENSITIVITY AUDIT

Research Component:
Urban Green Transition, Urbanization Pressure and
Conservation Intelligence

Purpose:
1. Compare experimental scoring methods from Step 16C-F.
2. Measure score changes for every grid cell.
3. Identify cells sensitive to scoring method choice.
4. Save reproducible reports and validation results.

IMPORTANT:
- No final scoring method selected.
- No final criterion directions selected.
- No AHP weights assigned.
- No combined suitability score calculated.
- Frozen research datasets are not modified.
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

ROOT = Path(__file__).resolve().parents[2]

RAW_INPUT = (
    ROOT / "data/processed/development_suitability"
    / "raw_criteria"
    / "16b_raw_suitability_criteria_100m.csv"
)

REPORTS = ROOT / "outputs/reports"

SCORES_INPUT = (
    REPORTS
    / "16c_f_candidate_scoring_scenarios_100m.csv"
)

PREVIOUS_SUMMARY = (
    REPORTS
    / "16c_f_scoring_method_summary.json"
)

COMPARISON_CSV = (
    REPORTS
    / "16c_g_scoring_sensitivity_comparison.csv"
)

SENSITIVE_CSV = (
    REPORTS
    / "16c_g_scoring_sensitive_cells.csv"
)

QA_CSV = (
    REPORTS
    / "16c_g_scoring_sensitivity_qa.csv"
)

SUMMARY_JSON = (
    REPORTS
    / "16c_g_scoring_sensitivity_summary.json"
)

EXPECTED_CELLS = 9240

# Exploratory threshold, not an official planning rule.
SENSITIVITY_THRESHOLD = 0.20


# ============================================================
# 2. EXPERIMENTAL SCENARIOS
# ============================================================

SCENARIOS = [
    (
        "C1",
        "DECREASING",
        "c1_major_road_distance_m",
    ),
    (
        "C2",
        "INCREASING",
        "c2_road_density_500m_km_per_km2",
    ),
    (
        "C3",
        "INCREASING",
        "c3_context_building_pct_500m",
    ),
    (
        "C3",
        "DECREASING",
        "c3_context_building_pct_500m",
    ),
    (
        "C4",
        "DECREASING",
        "c4_slope_mean_deg",
    ),
]


# ============================================================
# 3. START AND LOAD INPUTS
# ============================================================

print("=" * 95)
print("STEP 16C-G - SCORING METHOD SENSITIVITY AUDIT")
print("=" * 95)

for path in [
    RAW_INPUT,
    SCORES_INPUT,
    PREVIOUS_SUMMARY,
]:
    if not path.is_file():
        raise FileNotFoundError(path)

raw = pd.read_csv(RAW_INPUT)

scores = pd.read_csv(SCORES_INPUT)

with PREVIOUS_SUMMARY.open(
    encoding="utf-8"
) as file:
    previous = json.load(file)

raw_hash = hashlib.sha256(
    RAW_INPUT.read_bytes()
).hexdigest()

scores_hash = hashlib.sha256(
    SCORES_INPUT.read_bytes()
).hexdigest()


# ============================================================
# 4. INITIAL QUALITY CHECKS
# ============================================================

checks = []


def check(name, passed, details):
    checks.append({
        "check": name,
        "passed": bool(passed),
        "details": str(details),
    })


check(
    "Raw dataset has 9240 unique grid cells",
    len(raw) == EXPECTED_CELLS
    and raw["grid_id"].is_unique
    and raw["grid_id"].notna().all(),
    f"Actual: {len(raw)}",
)

check(
    "Experimental scores have 9240 unique grid cells",
    len(scores) == EXPECTED_CELLS
    and scores["grid_id"].is_unique
    and scores["grid_id"].notna().all(),
    f"Actual: {len(scores)}",
)

check(
    "Previous Step 16C-F QA passed",
    previous.get("qa_status") == "PASS",
    "Step 16C-F must be completed",
)

check(
    "Previous scoring methods remain experimental",
    previous.get("methodology_status") == "EXPERIMENTAL",
    "No final method has been selected",
)

check(
    "Frozen raw dataset checksum matches",
    raw_hash == previous.get("input_sha256"),
    "Verify unchanged Step 16B input",
)

check(
    "Raw and score grid IDs match in order",
    raw["grid_id"].equals(scores["grid_id"]),
    "Prevent incorrect comparisons between cells",
)

expected_score_columns = []

for criterion, direction, variable in SCENARIOS:
    prefix = criterion.lower()
    suffix = direction.lower()

    expected_score_columns.extend([
        f"{prefix}_pctclip_{suffix}",
        f"{prefix}_rank_{suffix}",
    ])

required_raw_columns = [
    item[2] for item in SCENARIOS
]

check(
    "All experimental score columns exist",
    all(
        column in scores.columns
        for column in expected_score_columns
    ),
    "Expected ten score columns",
)

check(
    "All raw criterion columns exist",
    all(
        column in raw.columns
        for column in required_raw_columns
    ),
    "C1, C2, C3 and C4 available",
)

REPORTS.mkdir(
    parents=True,
    exist_ok=True,
)

if not all(item["passed"] for item in checks):
    pd.DataFrame(checks).to_csv(
        QA_CSV, index=False
    )
    raise RuntimeError(
        "Initial QA failed. Inspect the QA CSV."
    )


# ============================================================
# 5. COMPARE EXPERIMENTAL METHODS
# ============================================================

comparison_records = []
sensitive_records = []

for criterion, direction, raw_column in SCENARIOS:

    prefix = criterion.lower()
    suffix = direction.lower()

    capped_col = (
        f"{prefix}_pctclip_{suffix}"
    )

    rank_col = (
        f"{prefix}_rank_{suffix}"
    )

    capped = pd.to_numeric(
        scores[capped_col],
        errors="coerce",
    )

    ranked = pd.to_numeric(
        scores[rank_col],
        errors="coerce",
    )

    difference = (capped - ranked).abs()

    comparison_records.append({
        "criterion": criterion,
        "direction": direction,
        "raw_variable": raw_column,

        "method_a": "PERCENTILE_CAPPED",
        "method_b": "PERCENTILE_RANK",

        "mean_abs_difference":
            float(difference.mean()),

        "median_abs_difference":
            float(difference.median()),

        "p95_abs_difference":
            float(difference.quantile(0.95)),

        "max_abs_difference":
            float(difference.max()),

        "rmse_difference":
            float(np.sqrt(
                np.mean(difference ** 2)
            )),

        "pearson_score_correlation":
            float(capped.corr(
                ranked,
                method="pearson",
            )),

        "spearman_score_correlation":
            float(capped.corr(
                ranked,
                method="spearman",
            )),

        "capped_mean":
            float(capped.mean()),

        "rank_mean":
            float(ranked.mean()),

        "capped_zero_pct":
            float(capped.eq(0).mean() * 100),

        "capped_one_pct":
            float(capped.eq(1).mean() * 100),

        "rank_zero_pct":
            float(ranked.eq(0).mean() * 100),

        "rank_one_pct":
            float(ranked.eq(1).mean() * 100),

        "sensitive_cell_count":
            int(
                (difference >= SENSITIVITY_THRESHOLD).sum()
            ),

        "sensitive_cell_pct":
            float(
                (difference >= SENSITIVITY_THRESHOLD).mean()
                * 100
            ),

        "status": "EXPERIMENT_ONLY",
    })

    sensitive_mask = (
        difference >= SENSITIVITY_THRESHOLD
    )

    sensitive_indices = np.flatnonzero(
        sensitive_mask.to_numpy()
    )

    for index in sensitive_indices:

        sensitive_records.append({
            "grid_id": raw.iloc[index]["grid_id"],
            "criterion": criterion,
            "direction": direction,
            "raw_variable": raw_column,

            "raw_value": float(
                raw.iloc[index][raw_column]
            ),

            "percentile_capped_score": float(
                capped.iloc[index]
            ),

            "percentile_rank_score": float(
                ranked.iloc[index]
            ),

            "absolute_difference": float(
                difference.iloc[index]
            ),

            "review_reason":
                "SCORING_METHOD_SENSITIVITY",
        })


comparison = pd.DataFrame(
    comparison_records
)

sensitive = pd.DataFrame(
    sensitive_records,
    columns=[
        "grid_id",
        "criterion",
        "direction",
        "raw_variable",
        "raw_value",
        "percentile_capped_score",
        "percentile_rank_score",
        "absolute_difference",
        "review_reason",
    ],
)


# ============================================================
# 6. FINAL VALIDATION
# ============================================================

numeric_score_values = scores[
    expected_score_columns
].apply(
    pd.to_numeric,
    errors="coerce",
)

check(
    "Experimental scores are finite",
    bool(
        np.isfinite(
            numeric_score_values.to_numpy(dtype=float)
        ).all()
    ),
    "No missing or infinite scores",
)

check(
    "Experimental scores remain within 0-1",
    bool(
        numeric_score_values.ge(0).all().all()
        and numeric_score_values.le(1).all().all()
    ),
    "Valid score range",
)

check(
    "Exactly five scenario comparisons",
    len(comparison) == len(SCENARIOS),
    f"Actual: {len(comparison)}",
)

check(
    "All comparison metrics are finite",
    bool(
        np.isfinite(
            comparison[
                [
                    "mean_abs_difference",
                    "median_abs_difference",
                    "p95_abs_difference",
                    "max_abs_difference",
                    "rmse_difference",
                    "pearson_score_correlation",
                    "spearman_score_correlation",
                    "sensitive_cell_pct",
                ]
            ].to_numpy(dtype=float)
        ).all()
    ),
    "All scenario metrics valid",
)

check(
    "Sensitive-cell totals match",
    len(sensitive)
    == int(comparison["sensitive_cell_count"].sum()),
    "Detailed records match summary",
)

check(
    "C3 opposite directions are complementary",
    bool(
        np.allclose(
            numeric_score_values[
                "c3_pctclip_increasing"
            ]
            + numeric_score_values[
                "c3_pctclip_decreasing"
            ],
            1.0,
            atol=1e-10,
        )
        and np.allclose(
            numeric_score_values[
                "c3_rank_increasing"
            ]
            + numeric_score_values[
                "c3_rank_decreasing"
            ],
            1.0,
            atol=1e-10,
        )
    ),
    "Opposite C3 scenarios must be consistent",
)

check(
    "No final scoring decision selected",
    bool(
        comparison["status"].eq(
            "EXPERIMENT_ONLY"
        ).all()
    ),
    "Comparison only",
)

check(
    "Frozen raw input remains unchanged",
    hashlib.sha256(
        RAW_INPUT.read_bytes()
    ).hexdigest() == raw_hash,
    "Read-only analysis",
)

check(
    "Experimental scoring input remains unchanged",
    hashlib.sha256(
        SCORES_INPUT.read_bytes()
    ).hexdigest() == scores_hash,
    "Read-only analysis",
)

qa = pd.DataFrame(checks)

overall_pass = bool(qa["passed"].all())

status = (
    "PASS"
    if overall_pass
    else "REVIEW_REQUIRED"
)

qa.to_csv(
    QA_CSV,
    index=False,
)

if not overall_pass:
    print(qa.to_string(index=False))
    raise RuntimeError(
        "Step 16C-G validation failed."
    )


# ============================================================
# 7. SAVE REPORTS
# ============================================================

comparison.to_csv(
    COMPARISON_CSV,
    index=False,
)

sensitive.to_csv(
    SENSITIVE_CSV,
    index=False,
)

summary = {
    "step": "16C-G",
    "name": "Scoring Method Sensitivity Audit",
    "qa_status": status,
    "methodology_status": "EXPERIMENTAL",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "grid_cells": int(len(raw)),
    "input_sha256": raw_hash,
    "scoring_input_sha256": scores_hash,
    "scenario_count": len(SCENARIOS),
    "sensitivity_threshold":
        SENSITIVITY_THRESHOLD,
    "sensitivity_threshold_note": (
        "An exploratory score-difference flag, "
        "not a planning threshold."
    ),
    "qa_checks": len(qa),
    "qa_passed": int(qa["passed"].sum()),
    "important_notes": [
        "No final scoring method selected.",
        "No combined suitability score calculated.",
        "The 0.20 difference flag is exploratory.",
        "Rank scaling depends on relative grid ranks.",
        "Percentile-capped scaling can create tied scores.",
        "C3 opposite directions remain experimental.",
        "C5 elevation has not been standardized.",
        "No AHP weights or zoning rules applied.",
    ],
    "outputs": {
        "comparison": COMPARISON_CSV.name,
        "sensitive_cells": SENSITIVE_CSV.name,
        "qa": QA_CSV.name,
        "summary": SUMMARY_JSON.name,
    },
}

with SUMMARY_JSON.open(
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
# 8. TERMINAL RESULTS
# ============================================================

print("\n=== SCORING SENSITIVITY RESULTS ===")

print(
    comparison[
        [
            "criterion",
            "direction",
            "mean_abs_difference",
            "p95_abs_difference",
            "max_abs_difference",
            "pearson_score_correlation",
            "spearman_score_correlation",
            "sensitive_cell_count",
            "sensitive_cell_pct",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

print("\n=== QA RESULTS ===")

for _, row in qa.iterrows():
    result = "PASS" if row["passed"] else "FAIL"
    print(f"{row['check']}: {result}")

print("\n=== GENERATED REPORTS ===")

for path in [
    COMPARISON_CSV,
    SENSITIVE_CSV,
    QA_CSV,
    SUMMARY_JSON,
]:
    print(path.relative_to(ROOT))

print("\nOverall Step 16C-G QA:", status)
print("Methodology: EXPERIMENTAL")
print("No suitability weights or final scores calculated.")
print("=" * 95)
