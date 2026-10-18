
"""
STEP 16C-F - CRITERIA SCORING METHOD DESIGN

Research:
Urban Green Transition, Urbanization Pressure and
Conservation Intelligence

Purpose:
1. Test alternative 0-1 transformations for raw criteria.
2. Compare percentile-capped and percentile-rank methods.
3. Examine both possible directions for C3.
4. Keep C5 elevation unscored until justified.
5. Save candidate scores and reproducible QA.

IMPORTANT:
- Candidate scores are experiments only.
- No production suitability score is calculated.
- No AHP weights are assigned.
- No official UDA thresholds are invented.
- Step 16B and previous results remain unchanged.
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

INPUT = (
    ROOT
    / "data/processed/development_suitability"
    / "raw_criteria"
    / "16b_raw_suitability_criteria_100m.csv"
)

REPORTS = ROOT / "outputs/reports"

PREVIOUS_D = (
    REPORTS
    / "16c_d_criteria_selection_summary.json"
)

PREVIOUS_E = (
    REPORTS
    / "16c_e_planning_evidence_summary.json"
)

PREVIOUS_REGISTRY = (
    REPORTS
    / "16c_d_provisional_criteria_registry.csv"
)

SCORES_CSV = (
    REPORTS
    / "16c_f_candidate_scoring_scenarios_100m.csv"
)

METHODS_CSV = (
    REPORTS
    / "16c_f_scoring_method_comparison.csv"
)

DIRECTIONS_CSV = (
    REPORTS
    / "16c_f_direction_design_registry.csv"
)

QA_CSV = (
    REPORTS
    / "16c_f_scoring_method_qa.csv"
)

SUMMARY_JSON = (
    REPORTS
    / "16c_f_scoring_method_summary.json"
)


# ============================================================
# 2. CRITERION VARIABLES
# ============================================================

C1 = "c1_major_road_distance_m"

C2 = "c2_road_density_500m_km_per_km2"

C3 = "c3_context_building_pct_500m"

C4 = "c4_slope_mean_deg"

C5 = "c5_elevation_mean_m"

VARIABLES = [C1, C2, C3, C4, C5]


# ============================================================
# 3. START
# ============================================================

print("=" * 95)
print("STEP 16C-F - CRITERIA SCORING METHOD DESIGN")
print("=" * 95)

required_files = [
    INPUT,
    PREVIOUS_D,
    PREVIOUS_E,
    PREVIOUS_REGISTRY,
]

for path in required_files:
    if not path.is_file():
        raise FileNotFoundError(path)


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


previous_d = read_json(PREVIOUS_D)
previous_e = read_json(PREVIOUS_E)

previous_registry = pd.read_csv(
    PREVIOUS_REGISTRY
)

df = pd.read_csv(INPUT)

input_hash = hashlib.sha256(
    INPUT.read_bytes()
).hexdigest()

missing = [
    column
    for column in ["grid_id"] + VARIABLES
    if column not in df.columns
]

if missing:
    raise RuntimeError(
        "Missing input columns: " + ", ".join(missing)
    )

data = df[VARIABLES].apply(
    pd.to_numeric,
    errors="coerce"
)

print("\nInput:", INPUT.name)
print("Grid cells:", len(df))
print("Candidate criteria:", len(VARIABLES))


# ============================================================
# 4. INITIAL QA
# ============================================================

checks = []


def check(name, passed, detail):
    checks.append({
        "check": name,
        "passed": bool(passed),
        "detail": str(detail),
    })


check(
    "9240 unique grid cells",
    len(df) == 9240
    and df["grid_id"].is_unique
    and df["grid_id"].notna().all(),
    f"Actual rows: {len(df)}",
)

check(
    "Step 16C-D QA passed",
    previous_d.get("qa_status") == "PASS",
    "Previous criteria review",
)

check(
    "Step 16C-E QA passed",
    previous_e.get("qa_status") == "PASS",
    "Previous planning evidence audit",
)

check(
    "Frozen input matches Step 16C-D",
    input_hash == previous_d.get("input_sha256"),
    "SHA256 comparison",
)

check(
    "Frozen input matches Step 16C-E",
    input_hash
    == previous_e.get("frozen_16b_csv_sha256"),
    "SHA256 comparison",
)

check(
    "All five raw criteria contain finite values",
    bool(
        np.isfinite(
            data.to_numpy(dtype=float)
        ).all()
    ),
    "No missing or infinite values",
)

check(
    "All five raw criteria vary across cells",
    bool(
        (data.nunique() > 1).all()
    ),
    "Nonconstant variables required",
)

expected_roles = {
    C1: "PROVISIONAL_CORE",
    C2: "PROVISIONAL_CORE",
    C3: "PROVISIONAL_CORE",
    C4: "PROVISIONAL_CORE",
    C5: "PENDING_JUSTIFICATION",
}

registry_roles = (
    previous_registry
    .set_index("variable")["role"]
    .to_dict()
)

check(
    "Previous provisional roles preserved",
    all(
        registry_roles.get(variable) == role
        for variable, role in expected_roles.items()
    ),
    "C5 must remain pending",
)

REPORTS.mkdir(
    parents=True,
    exist_ok=True
)

if not all(item["passed"] for item in checks):
    pd.DataFrame(checks).to_csv(
        QA_CSV, index=False
    )
    raise RuntimeError(
        "Initial QA failed. Review QA CSV."
    )


# ============================================================
# 5. STANDARDIZATION CANDIDATE FUNCTIONS
# ============================================================

def percentile_capped_score(
    values,
    direction,
    lower_percentile=0.05,
    upper_percentile=0.95,
):
    """
    Convert raw values to 0-1 using percentile limits.

    This is an experimental statistical transformation,
    NOT an official planning threshold.
    """

    lower = float(
        values.quantile(lower_percentile)
    )

    upper = float(
        values.quantile(upper_percentile)
    )

    if upper <= lower:
        raise ValueError(
            "Invalid percentile range."
        )

    clipped = values.clip(
        lower=lower,
        upper=upper
    )

    score = (
        (clipped - lower)
        / (upper - lower)
    )

    if direction == "DECREASING":
        score = 1.0 - score

    elif direction != "INCREASING":
        raise ValueError(
            "Unsupported direction."
        )

    return score.clip(0.0, 1.0), lower, upper


def percentile_rank_score(values, direction):
    """
    Rank-based score, using average ranks for ties.

    The scaling is relative to the study dataset.
    """

    if len(values) <= 1:
        raise ValueError(
            "At least two values are required."
        )

    rank = values.rank(
        method="average"
    )

    score = (
        (rank - 1.0)
        / (len(values) - 1.0)
    )

    if direction == "DECREASING":
        score = 1.0 - score

    elif direction != "INCREASING":
        raise ValueError(
            "Unsupported direction."
        )

    return score.clip(0.0, 1.0)


# ============================================================
# 6. DEFINE EXPERIMENTAL SCENARIOS
# ============================================================

SCENARIOS = [
    {
        "criterion": "C1",
        "variable": C1,
        "direction": "DECREASING",
        "status": "PROVISIONAL_DIRECTION",
    },
    {
        "criterion": "C2",
        "variable": C2,
        "direction": "INCREASING",
        "status": "PROVISIONAL_DIRECTION",
    },
    {
        "criterion": "C3",
        "variable": C3,
        "direction": "INCREASING",
        "status": "ALTERNATIVE_ONLY",
    },
    {
        "criterion": "C3",
        "variable": C3,
        "direction": "DECREASING",
        "status": "ALTERNATIVE_ONLY",
    },
    {
        "criterion": "C4",
        "variable": C4,
        "direction": "DECREASING",
        "status": "PROVISIONAL_DIRECTION",
    },
]

scores = pd.DataFrame({
    "grid_id": df["grid_id"]
})

comparison_records = []

generated_columns = []


# ============================================================
# 7. CALCULATE CANDIDATE SCORES
# ============================================================

for scenario in SCENARIOS:

    criterion = scenario["criterion"]
    variable = scenario["variable"]
    direction = scenario["direction"]

    values = data[variable]

    direction_name = direction.lower()

    score_capped, lower, upper = (
        percentile_capped_score(
            values,
            direction,
        )
    )

    score_rank = percentile_rank_score(
        values,
        direction,
    )

    capped_column = (
        f"{criterion.lower()}_pctclip_"
        f"{direction_name}"
    )

    rank_column = (
        f"{criterion.lower()}_rank_"
        f"{direction_name}"
    )

    scores[capped_column] = (
        score_capped.to_numpy()
    )

    scores[rank_column] = (
        score_rank.to_numpy()
    )

    generated_columns.extend([
        capped_column,
        rank_column,
    ])

    difference = (
        score_capped - score_rank
    ).abs()

    for method, column, result in [
        (
            "PERCENTILE_CAPPED",
            capped_column,
            score_capped,
        ),
        (
            "PERCENTILE_RANK",
            rank_column,
            score_rank,
        ),
    ]:

        comparison_records.append({
            "criterion": criterion,
            "variable": variable,
            "direction": direction,
            "scenario_status":
                scenario["status"],
            "method": method,
            "output_column": column,
            "raw_p05": lower,
            "raw_p95": upper,
            "score_min": float(result.min()),
            "score_mean": float(result.mean()),
            "score_median": float(
                result.median()
            ),
            "score_max": float(result.max()),
            "score_std": float(result.std()),
            "score_at_zero_pct": float(
                result.eq(0).mean() * 100
            ),
            "score_at_one_pct": float(
                result.eq(1).mean() * 100
            ),
            "mean_method_difference": float(
                difference.mean()
            ),
            "p95_method_difference": float(
                difference.quantile(0.95)
            ),
            "decision": "EXPERIMENT_ONLY",
        })


comparison = pd.DataFrame(
    comparison_records
)


# ============================================================
# 8. METHOD AND DIRECTION CONSISTENCY QA
# ============================================================

def monotonic_check(
    raw_values,
    score_values,
    direction,
):
    """
    Verify that a score respects its stated direction.
    """

    order = np.argsort(
        raw_values.to_numpy(),
        kind="mergesort",
    )

    ordered_score = (
        score_values.to_numpy()[order]
    )

    differences = np.diff(
        ordered_score
    )

    if direction == "INCREASING":
        return bool(
            (differences >= -1e-10).all()
        )

    return bool(
        (differences <= 1e-10).all()
    )


monotonic_results = []

for scenario in SCENARIOS:

    criterion = scenario["criterion"]
    direction = scenario["direction"]
    variable = scenario["variable"]

    for method in [
        "pctclip",
        "rank",
    ]:

        column = (
            f"{criterion.lower()}_{method}_"
            f"{direction.lower()}"
        )

        passed = monotonic_check(
            data[variable],
            scores[column],
            direction,
        )

        monotonic_results.append(
            passed
        )


check(
    "Expected ten experimental score columns",
    len(generated_columns) == 10
    and len(set(generated_columns)) == 10,
    f"Actual: {len(generated_columns)}",
)

check(
    "All scores are finite and within 0-1",
    bool(
        np.isfinite(
            scores[generated_columns]
            .to_numpy(dtype=float)
        ).all()
        and
        scores[generated_columns]
        .ge(0).all().all()
        and
        scores[generated_columns]
        .le(1).all().all()
    ),
    "Candidate score range validation",
)

check(
    "All candidate scores nonconstant",
    bool(
        (
            scores[generated_columns].nunique()
            > 1
        ).all()
    ),
    "Experimental outputs must vary",
)

check(
    "All score directions are monotonic",
    all(monotonic_results),
    (
        f"Passed {sum(monotonic_results)}"
        f"/{len(monotonic_results)}"
    ),
)

check(
    "C3 opposite scenarios remain complementary",
    bool(
        np.allclose(
            scores["c3_pctclip_increasing"]
            + scores["c3_pctclip_decreasing"],
            1.0,
            atol=1e-10,
        )
    ),
    "Opposite interpretations tested consistently",
)

check(
    "Elevation has no experimental score",
    not any(
        column.startswith("c5_")
        for column in generated_columns
    ),
    "C5 requires further justification",
)

check(
    "No final scoring decision assigned",
    comparison["decision"]
    .eq("EXPERIMENT_ONLY").all(),
    "No production scoring formula chosen",
)

check(
    "Frozen input unchanged during analysis",
    hashlib.sha256(
        INPUT.read_bytes()
    ).hexdigest() == input_hash,
    "Read-only source verification",
)


# ============================================================
# 9. CRITERION DIRECTION DESIGN REGISTRY
# ============================================================

direction_records = [
    {
        "criterion": "C1",
        "variable": C1,
        "direction": "DECREASING",
        "status": "PROVISIONAL",
        "reason": (
            "Shorter distance to major roads "
            "generally supports accessibility."
        ),
        "pending_decision": (
            "Choose final scoring method; "
            "check physical access and road restrictions."
        ),
    },
    {
        "criterion": "C2",
        "variable": C2,
        "direction": "INCREASING",
        "status": "PROVISIONAL",
        "reason": (
            "Higher nearby road density may "
            "support general accessibility."
        ),
        "pending_decision": (
            "Justify the 500m scale and scoring method."
        ),
    },
    {
        "criterion": "C3",
        "variable": C3,
        "direction": "UNDECIDED",
        "status": "NEEDS_RESEARCH_JUSTIFICATION",
        "reason": (
            "Existing development may show available "
            "services but also land saturation."
        ),
        "pending_decision": (
            "Justify positive, negative or "
            "non-monotonic interpretation using "
            "planning evidence and sensitivity tests."
        ),
    },
    {
        "criterion": "C4",
        "variable": C4,
        "direction": "DECREASING",
        "status": "PROVISIONAL",
        "reason": (
            "Lower slope generally makes "
            "construction easier."
        ),
        "pending_decision": (
            "Choose planning-supported scoring "
            "thresholds or a relative scaling method."
        ),
    },
    {
        "criterion": "C5",
        "variable": C5,
        "direction": "UNDECIDED",
        "status": "NEEDS_RESEARCH_JUSTIFICATION",
        "reason": (
            "Elevation may relate to local terrain "
            "and protection conditions but has no "
            "universal suitability direction."
        ),
        "pending_decision": (
            "Decide whether elevation should be "
            "a core criterion or separate screening "
            "indicator, while retaining its raw data."
        ),
    },
]

direction_registry = pd.DataFrame(
    direction_records
)


# ============================================================
# 10. FINAL QA STATUS
# ============================================================

qa = pd.DataFrame(checks)

overall_pass = bool(
    qa["passed"].all()
)

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
    print(
        "\nQA FAILED:"
    )
    print(qa.to_string(index=False))
    raise RuntimeError(
        "Step 16C-F QA failed."
    )


# ============================================================
# 11. SAVE OUTPUTS
# ============================================================

scores.to_csv(
    SCORES_CSV,
    index=False,
)

comparison.to_csv(
    METHODS_CSV,
    index=False,
)

direction_registry.to_csv(
    DIRECTIONS_CSV,
    index=False,
)


# ============================================================
# 12. SUMMARY JSON
# ============================================================

summary = {
    "step": "16C-F",
    "name": "Criteria Scoring Method Design",
    "qa_status": status,
    "methodology_status": "EXPERIMENTAL",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "input_sha256": input_hash,
    "grid_cells": int(len(df)),
    "tested_criteria": [
        "C1", "C2", "C3", "C4"
    ],
    "pending_criterion": "C5",
    "experimental_score_columns":
        len(generated_columns),
    "experimental_scenarios":
        len(SCENARIOS),
    "scoring_methods": [
        "PERCENTILE_CAPPED",
        "PERCENTILE_RANK",
    ],
    "percentile_cap_range": [
        0.05, 0.95
    ],
    "qa_checks": len(qa),
    "qa_passed": int(
        qa["passed"].sum()
    ),
    "methodology_notes": [
        (
            "Candidate scores are experimental "
            "and are not final suitability scores."
        ),
        (
            "5th and 95th percentiles are statistical "
            "comparison limits, not official UDA rules."
        ),
        (
            "C3 increasing and decreasing directions "
            "are both tested without selecting one."
        ),
        (
            "C5 remains unscored until its role "
            "is academically justified."
        ),
        (
            "The 500m C2/C3 measurements remain "
            "provisional spatial-scale choices."
        ),
        (
            "No AHP weights, final suitability scores "
            "or zoning restrictions were applied."
        ),
        (
            "The frozen Step 16B input is unchanged."
        ),
    ],
    "outputs": {
        "candidate_scores": SCORES_CSV.name,
        "method_comparison": METHODS_CSV.name,
        "direction_registry": DIRECTIONS_CSV.name,
        "qa": QA_CSV.name,
        "summary": SUMMARY_JSON.name,
    },
}

with SUMMARY_JSON.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 13. TERMINAL REPORT
# ============================================================

print("\n=== SCORING METHOD COMPARISON ===")

display_columns = [
    "criterion",
    "direction",
    "method",
    "score_mean",
    "score_std",
    "score_at_zero_pct",
    "score_at_one_pct",
    "mean_method_difference",
]

print(
    comparison[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\n=== CRITERION DIRECTIONS ===")

print(
    direction_registry[
        [
            "criterion",
            "direction",
            "status",
        ]
    ].to_string(index=False)
)

print("\n=== STEP 16C-F QA ===")

for _, row in qa.iterrows():
    result = (
        "PASS"
        if row["passed"]
        else "FAIL"
    )
    print(
        f"{row['check']}: {result}"
    )

print("\n=== GENERATED REPORTS ===")

for path in [
    SCORES_CSV,
    METHODS_CSV,
    DIRECTIONS_CSV,
    QA_CSV,
    SUMMARY_JSON,
]:
    print(path.relative_to(ROOT))

print("\nOverall Step 16C-F QA:", status)
print("Methodology status: EXPERIMENTAL")

print(
    "No final suitability score "
    "or AHP weights calculated."
)

print("=" * 95)
