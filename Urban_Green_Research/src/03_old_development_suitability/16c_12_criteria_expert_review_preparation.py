
"""
STEP 16C-L - CRITERIA EVIDENCE AND EXPERT REVIEW PREPARATION

Research component:
Urban Green Transition, Urbanization Pressure
and Conservation Intelligence

Purpose:
1. Review existing criteria evidence.
2. Prepare an expert-review brief.
3. Prepare unanswered criteria decision forms.
4. Prepare alternative AHP pairwise questionnaires.
5. Preserve existing research data and reports.

IMPORTANT:
No expert judgments, AHP weights, final suitability
scores, or legal planning decisions are generated.
"""

from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REPORTS = ROOT / "outputs" / "reports"

RAW = (
    ROOT
    / "data/processed/development_suitability/raw_criteria"
    / "16b_raw_suitability_criteria_100m.csv"
)

D = REPORTS / "16c_d_provisional_criteria_registry.csv"
E = REPORTS / "16c_e_planning_evidence_summary.json"
F = REPORTS / "16c_f_scoring_method_summary.json"
G = REPORTS / "16c_g_scoring_sensitivity_summary.json"
J = REPORTS / "16c_j_zoning_reconciliation_summary.json"
K = REPORTS / "16c_k_criterion_robustness_summary.json"

K_COMP = (
    REPORTS / "16c_k_scenario_rank_stability.csv"
)

OUT_BRIEF = (
    REPORTS / "16c_l_expert_review_brief.md"
)

OUT_CRITERIA = (
    REPORTS / "16c_l_expert_criteria_review_template.csv"
)

OUT_AHP = (
    REPORTS / "16c_l_ahp_pairwise_template.csv"
)

OUT_EVIDENCE = (
    REPORTS / "16c_l_criteria_evidence_snapshot.csv"
)

OUT_QA = (
    REPORTS / "16c_l_expert_review_qa.csv"
)

OUT_SUMMARY = (
    REPORTS / "16c_l_expert_review_summary.json"
)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def sha(path):

    digest = hashlib.sha256()

    with path.open("rb") as source:

        for block in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as source:

        return json.load(source)


def save_template_unless_exists(df, path):
    """
    Never overwrite a form that an expert might
    already have started filling in.
    """

    if not path.exists():

        df.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        return "CREATED"

    old = pd.read_csv(
        path,
        dtype="string",
        encoding="utf-8-sig",
    )

    if (
        list(old.columns) != list(df.columns)
        or len(old) != len(df)
    ):

        raise RuntimeError(
            "Existing template structure differs: "
            f"{path}"
        )

    editable = {
        "expert_decision",
        "expert_direction",
        "expert_reason",
        "reviewer_name",
        "reviewer_role",
        "review_date",
        "higher_importance",
        "importance_strength_1_to_9",
        "justification",
    }

    fixed = [
        column
        for column in df.columns
        if column not in editable
    ]

    expected = (
        df[fixed]
        .fillna("")
        .astype(str)
        .reset_index(drop=True)
    )

    actual = (
        old[fixed]
        .fillna("")
        .astype(str)
        .reset_index(drop=True)
    )

    if not expected.equals(actual):

        raise RuntimeError(
            "Existing template questions differ. "
            f"File preserved: {path}"
        )

    return "PRESERVED_EXISTING"


# ============================================================
# 3. MAIN
# ============================================================

def main():

    print("=" * 88)

    print(
        "STEP 16C-L - CRITERIA EVIDENCE "
        "AND EXPERT REVIEW PREPARATION"
    )

    print("=" * 88)

    required_files = [
        RAW, D, E, F, G, J, K, K_COMP
    ]

    for path in required_files:

        if not path.is_file():

            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    e, f, g, j, k = [
        load_json(path)
        for path in (E, F, G, J, K)
    ]

    raw_digest = sha(RAW)

    k_comp_digest = sha(K_COMP)

    checks = []

    def check(label, passed, detail=""):

        checks.append({
            "check": label,
            "passed": bool(passed),
            "detail": str(detail),
        })

    # ========================================================
    # 4. VERIFY PREVIOUS RESEARCH
    # ========================================================

    check(
        "Prior 16C-E/F/G/J/K audits passed",
        all(
            result.get("qa_status") == "PASS"
            for result in (e, f, g, j, k)
        ),
    )

    check(
        "Step 16B raw dataset unchanged",
        all(
            value == raw_digest
            for value in [
                f.get("input_sha256"),
                g.get("input_sha256"),
                j.get("frozen_input_sha256"),
                k.get("frozen_raw_sha256"),
            ]
        ),
    )

    check(
        "Step 16C-K remains diagnostic",
        k.get("methodology_status")
        == "DIAGNOSTIC_ONLY_FINAL_CRITERIA_PENDING",
    )

    check(
        "Provisional zoning audit completed",
        j.get("qa_status") == "PASS",
        "Zoning remains legally unverified",
    )

    # ========================================================
    # 5. VERIFY CRITERION REGISTRY
    # ========================================================

    registry = pd.read_csv(D)

    expected_roles = {
        "c1_major_road_distance_m":
            "PROVISIONAL_CORE",

        "c2_road_density_500m_km_per_km2":
            "PROVISIONAL_CORE",

        "c3_context_building_pct_500m":
            "PROVISIONAL_CORE",

        "c4_slope_mean_deg":
            "PROVISIONAL_CORE",

        "c5_elevation_mean_m":
            "PENDING_JUSTIFICATION",
    }

    roles_valid = (
        {"variable", "role"}.issubset(
            registry.columns
        )
    )

    if roles_valid:

        recorded_roles = (
            registry
            .set_index("variable")["role"]
            .to_dict()
        )

        roles_valid = all(
            recorded_roles.get(variable) == role
            for variable, role
            in expected_roles.items()
        )

    check(
        "Previous criterion roles preserved",
        roles_valid,
    )

    # ========================================================
    # 6. RETRIEVE ROBUSTNESS EVIDENCE
    # ========================================================

    comparison = pd.read_csv(K_COMP)

    comparison = comparison.set_index(
        "comparison"
    )

    direction_rows = [
        "C3_direction_pctclip",
        "C3_direction_rank",
    ]

    check(
        "Two C3 direction comparisons available",
        all(
            name in comparison.index
            for name in direction_rows
        ),
    )

    if not all(
        item["passed"]
        for item in checks
    ):

        pd.DataFrame(checks).to_csv(
            OUT_QA,
            index=False,
        )

        raise RuntimeError(
            "Source QA failed. "
            "Inspect the 16C-L QA CSV."
        )

    def metric(row, column):

        return float(
            comparison.loc[row, column]
        )

    # ========================================================
    # 7. DEFINE CRITERIA FOR EXPERT REVIEW
    # ========================================================

    criteria = [
        (
            "C1",
            "c1_major_road_distance_m",
            "Accessibility to major roads",
            "Core candidate; shorter generally preferable",
            (
                "Is major-road distance a reasonable "
                "accessibility proxy? Consider actual "
                "access, buffers and road restrictions."
            ),
            "DECREASING",
        ),

        (
            "C2",
            "c2_road_density_500m_km_per_km2",
            "Neighbourhood road availability (500 m)",
            "Core candidate; direction provisional",
            (
                "Does greater road density generally "
                "help? Could there be saturation? "
                "Is the 500 m scale suitable?"
            ),
            "INCREASING_PROVISIONAL",
        ),

        (
            "C3",
            "c3_context_building_pct_500m",
            "Existing built-up context (500 m)",
            (
                "Provisional core in Step 16C-D; "
                "conditional candidate after sensitivity"
            ),
            (
                "Should existing buildings indicate "
                "infrastructure availability, saturation "
                "or a non-linear relationship? "
                "Include or exclude?"
            ),
            "UNDECIDED",
        ),

        (
            "C4",
            "c4_slope_mean_deg",
            "Terrain slope (degrees)",
            "Core candidate; gentler terrain preferable",
            (
                "Is slope suitable for general urban "
                "development assessment? Should "
                "engineering thresholds be supported "
                "by additional evidence?"
            ),
            "DECREASING",
        ),

        (
            "C5",
            "c5_elevation_mean_m",
            "Local elevation (metres)",
            "Context/screening candidate; role undecided",
            (
                "Is elevation independently relevant "
                "across Kaduwela? Do not generalize "
                "zone-specific elevation regulations."
            ),
            "UNDECIDED",
        ),
    ]

    # ========================================================
    # 8. CREATE CRITERIA EVIDENCE SNAPSHOT
    # ========================================================

    evidence = pd.DataFrame([
        {
            "criterion": code,
            "frozen_raw_variable": variable,
            "provisional_model_role": role,
            "working_direction": direction,
            "question_for_expert": question,
            "evidence_origin":
                "Steps 16C-D/E/F/G/K and GIS audits H-J",
            "decision_status":
                "PENDING_EXPERT_OR_DOCUMENTED_ALTERNATIVE",
        }
        for (
            code,
            variable,
            title,
            role,
            question,
            direction,
        ) in criteria
    ])

    # ========================================================
    # 9. CREATE BLANK CRITERIA REVIEW FORM
    # ========================================================

    response = pd.DataFrame([
        {
            "criterion": code,
            "criterion_description": title,
            "proposed_status_for_review": role,
            "review_question": question,

            "expert_decision": "",
            "expert_direction": "",
            "expert_reason": "",

            "reviewer_name": "",
            "reviewer_role": "",
            "review_date": "",
        }
        for (
            code,
            variable,
            title,
            role,
            question,
            direction,
        ) in criteria
    ])

    # ========================================================
    # 10. CREATE BLANK AHP QUESTIONNAIRES
    # ========================================================

    rows = []

    scenarios = [
        (
            "BASE_3_CRITERIA",
            ["C1", "C2", "C4"],
        ),

        (
            "CONDITIONAL_4_WITH_C3",
            ["C1", "C2", "C3", "C4"],
        ),
    ]

    for scenario, criterion_set in scenarios:

        for criterion_a, criterion_b in combinations(
            criterion_set,
            2,
        ):

            rows.append({
                "scenario": scenario,

                "criterion_a": criterion_a,
                "criterion_b": criterion_b,

                "question": (
                    "For general urban development "
                    "suitability, which is more important: "
                    f"{criterion_a} or {criterion_b}, "
                    "and by how much?"
                ),

                "higher_importance": "",
                "importance_strength_1_to_9": "",
                "justification": "",

                "reviewer_name": "",
                "reviewer_role": "",
                "review_date": "",
            })

    pairwise = pd.DataFrame(rows)

    # ========================================================
    # 11. TEMPLATE QA
    # ========================================================

    check(
        "Five criteria represented",
        len(response) == 5
        and response["criterion"].is_unique,
    )

    check(
        "Both AHP questionnaires have complete pairs",
        pairwise.groupby(
            "scenario"
        ).size().to_dict()
        == {
            "BASE_3_CRITERIA": 3,
            "CONDITIONAL_4_WITH_C3": 6,
        },
    )

    check(
        "Generated expert response fields are blank",
        all(
            response[column].eq("").all()
            for column in [
                "expert_decision",
                "expert_reason",
            ]
        )
        and all(
            pairwise[column].eq("").all()
            for column in [
                "higher_importance",
                "importance_strength_1_to_9",
            ]
        ),
    )

    if not all(
        item["passed"]
        for item in checks
    ):

        pd.DataFrame(checks).to_csv(
            OUT_QA,
            index=False,
        )

        raise RuntimeError(
            "Template QA failed."
        )

    # Protect forms on repeated execution.

    criteria_state = save_template_unless_exists(
        response,
        OUT_CRITERIA,
    )

    ahp_state = save_template_unless_exists(
        pairwise,
        OUT_AHP,
    )

    evidence.to_csv(
        OUT_EVIDENCE,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 12. PREPARE EXPERT-REVIEW EVIDENCE
    # ========================================================

    pctclip_j = metric(
        "C3_direction_pctclip",
        "top_decile_jaccard",
    )

    rank_j = metric(
        "C3_direction_rank",
        "top_decile_jaccard",
    )

    pctclip_s = metric(
        "C3_direction_pctclip",
        "spearman_rank_correlation",
    )

    rank_s = metric(
        "C3_direction_rank",
        "spearman_rank_correlation",
    )

    brief = f"""# Expert Review Request

## Research component

Urban Green Transition, Urbanization Pressure
and Conservation Intelligence — J26-DS-338.

Study area: Kaduwela Municipal Council.

Analysis: 9,240 grid cells, nominal 100 m resolution.

## Purpose

Review the candidate criteria for a relative
general urban development suitability map using
GIS-MCDA and AHP.

This does not determine legal building permission.

## Candidate criteria

| Code | Criterion | Working direction |
|---|---|---|
| C1 | Distance to major roads | Decreasing |
| C2 | Road density, 500 m | Increasing, provisional |
| C3 | Building context, 500 m | Undecided |
| C4 | Slope in degrees | Decreasing |
| C5 | Elevation in metres | Undecided |

## Important research findings

The original raw criteria were validated across
9,240 unique grid cells.

C3 direction is sensitive:

- Percentile-capped: Spearman {pctclip_s:.4f};
  top-decile Jaccard {pctclip_j:.4f}.
- Percentile-rank: Spearman {rank_s:.4f};
  top-decile Jaccard {rank_j:.4f}.

These statistical results cannot determine which
C3 direction is scientifically correct.

The 500 m C2 and C3 neighbourhood scales remain
provisional spatial-scale choices.

C5 elevation must not receive a city-wide direction
solely from elevation rules applying to specific
conservation zones.

The provisional UDA GIS is image-derived, has
low-confidence classifications and is legally
unverified. It must not be treated as official
development permission.

Road distance is not direct parcel access.
Road density is not proof of road capacity.
Building density does not show vacant land.
Slope does not replace site engineering studies.

## Requested expert review

First complete:
16c_l_expert_criteria_review_template.csv

For each criterion, provide:

- INCLUDE, EXCLUDE, REVISE or UNCERTAIN.
- INCREASING, DECREASING, NON_MONOTONIC,
  NOT_APPLICABLE or UNCERTAIN for direction.
- A short justification.
- Actual reviewer name, role and review date.

Next, complete the appropriate alternative in:
16c_l_ahp_pairwise_template.csv

Available alternatives:

BASE_3_CRITERIA: C1, C2 and C4.

CONDITIONAL_4_WITH_C3: C1, C2, C3 and C4.

Complete only the alternative matching the
reviewed criterion decision.

If a different criterion set is approved,
the questionnaire must be regenerated.

For each pairwise comparison:

Higher importance: A, B or EQUAL.

Strength: integer 1 to 9.

AHP verbal scale:

1 = Equal importance.
3 = Moderate importance.
5 = Strong importance.
7 = Very strong importance.
9 = Extreme importance.

2, 4, 6 and 8 = Intermediate judgments.

If equal, enter EQUAL and 1.

No reciprocal fractions are required.
The later AHP implementation will calculate them.

## Methodological safeguards

Expert input must be genuinely collected and
recorded. Do not invent responses.

AHP consistency will be validated separately.
Inconsistent judgments may require review.

If genuine expert input cannot be obtained,
a transparently documented alternative using
published evidence or scenario analysis is needed.

Planning screening and legal zoning verification
remain separate from the suitability score.

No final AHP weights, criterion set or suitability
recommendations are approved by this document.
"""

    if not OUT_BRIEF.exists():

        OUT_BRIEF.write_text(
            brief,
            encoding="utf-8",
        )

        brief_state = "CREATED"

    else:

        brief_state = "PRESERVED_EXISTING"

    # ========================================================
    # 13. FINAL QA
    # ========================================================

    check(
        "Frozen raw data unchanged",
        sha(RAW) == raw_digest,
    )

    check(
        "Source comparison report unchanged",
        sha(K_COMP) == k_comp_digest,
    )

    check(
        "Expert templates safely created or preserved",
        all(
            value in {
                "CREATED",
                "PRESERVED_EXISTING",
            }
            for value in (
                criteria_state,
                ahp_state,
            )
        ),
    )

    qa = pd.DataFrame(checks)

    status = (
        "PASS"
        if qa["passed"].all()
        else "REVIEW_REQUIRED"
    )

    qa.to_csv(
        OUT_QA,
        index=False,
    )

    # ========================================================
    # 14. SUMMARY JSON
    # ========================================================

    summary = {
        "step": "16C-L",

        "qa_status": status,

        "deliverable_status":
            "EXPERT_REVIEW_PREPARED_NOT_COLLECTED",

        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "frozen_input_sha256": raw_digest,

        "criterion_count": len(response),

        "ahp_alternative_pair_count": len(pairwise),

        "c3_direction_spearman_pctclip":
            pctclip_s,

        "c3_direction_jaccard_pctclip":
            pctclip_j,

        "c3_direction_spearman_rank":
            rank_s,

        "c3_direction_jaccard_rank":
            rank_j,

        "qa_checks": len(qa),

        "qa_passed": int(
            qa["passed"].sum()
        ),

        "criteria_template_state": criteria_state,

        "ahp_template_state": ahp_state,

        "brief_state": brief_state,

        "expert_judgments_collected_by_script": False,

        "weights_assigned": False,

        "final_suitability_calculated": False,

        "outputs": [
            path.name
            for path in [
                OUT_BRIEF,
                OUT_CRITERIA,
                OUT_AHP,
                OUT_EVIDENCE,
                OUT_QA,
                OUT_SUMMARY,
            ]
        ],
    }

    OUT_SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 15. PRINT TERMINAL RESULTS
    # ========================================================

    print("\n=== EVIDENCE FROM STEP 16C-K ===")

    print(
        "C3 direction (percentile-capped): "
        f"Spearman={pctclip_s:.4f}, "
        f"top-10% Jaccard={pctclip_j:.4f}"
    )

    print(
        "C3 direction (percentile-rank): "
        f"Spearman={rank_s:.4f}, "
        f"top-10% Jaccard={rank_j:.4f}"
    )

    print("\n=== PREPARED FOR REVIEW ===")

    for path in [
        OUT_BRIEF,
        OUT_CRITERIA,
        OUT_AHP,
        OUT_EVIDENCE,
        OUT_QA,
        OUT_SUMMARY,
    ]:

        print(
            " -",
            path.relative_to(ROOT),
        )

    print(
        "Criteria template:",
        criteria_state,
    )

    print(
        "AHP questionnaire:",
        ahp_state,
    )

    print(
        f"QA status: {status} "
        f"({int(qa['passed'].sum())}/{len(qa)})"
    )

    print(
        "Expert judgments not yet collected. "
        "No AHP weights or final suitability scores."
    )

    print("=" * 88)


# ============================================================
# 16. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
