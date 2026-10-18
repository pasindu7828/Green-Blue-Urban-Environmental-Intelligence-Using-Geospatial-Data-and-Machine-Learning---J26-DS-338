
"""
J26-DS-338
STEP 16C-N - CRITERIA AND STANDARDIZATION READINESS AUDIT

Research:
Urban Green Transition, Urbanization Pressure
and Conservation Intelligence.

Purpose:
1. Verify the frozen 9240-grid criteria.
2. Check standardized scoring and reference consistency.
3. Recalculate Spearman associations for redundancy review.
4. Document directions and candidate criteria.
5. Audit real expert-review readiness.
6. Save permanent evidence for the research viva.

IMPORTANT:
- AHP is mandatory for the final suitability model.
- This script does not assign expert judgments.
- Equal contribution is only an existing diagnostic baseline.
- High correlation is a review flag, not automatic exclusion.
- C3 and C5 are not silently accepted or rejected.
"""

from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "outputs" / "reports"

RAW_FILE = (
    ROOT / "data/processed/development_suitability/"
    "raw_criteria/16b_raw_suitability_criteria_100m.csv"
)

STANDARD_FILE = REPORTS / (
    "16d_a_provisional_standardized_criteria_100m.csv"
)

STANDARD_META = REPORTS / (
    "16d_a_standardized_criteria_summary.json"
)

BASELINE_FILE = REPORTS / (
    "16d_b_provisional_suitability_baseline_100m.csv"
)

BASELINE_META = REPORTS / (
    "16d_b_provisional_baseline_summary.json"
)

REVIEW_FILE = REPORTS / (
    "16c_l_expert_criteria_review_template.csv"
)

PAIRWISE_FILE = REPORTS / (
    "16c_l_ahp_pairwise_template.csv"
)

AHP_META = REPORTS / (
    "16d_d_ahp_consistency_summary.json"
)

OUT_CRITERIA = REPORTS / (
    "16c_n_criteria_readiness_registry.csv"
)

OUT_CORRELATION = REPORTS / (
    "16c_n_criteria_spearman_matrix.csv"
)

OUT_PAIRS = REPORTS / (
    "16c_n_criteria_redundancy_review.csv"
)

OUT_QA = REPORTS / (
    "16c_n_criteria_readiness_qa.csv"
)

OUT_SUMMARY = REPORTS / (
    "16c_n_criteria_readiness_summary.json"
)

OUT_REPORT = REPORTS / (
    "16c_n_criteria_methodology_report.md"
)

EXPECTED_GRID_COUNT = 9240

# Researcher-defined screening threshold.
# This is NOT proof of redundant criteria.
REDUNDANCY_REVIEW_THRESHOLD = 0.85

CORE_SCORE_COLUMNS = {
    "C1": "s_c1_road_pc",
    "C2": "s_c2_road_density_pc",
    "C4": "s_c4_slope_pc",
}

CONTEXT_COLUMNS = {
    "C3": "c3_context_building_pct_500m",
    "C5": "c5_elevation_mean_m",
}

CRITERIA = [
    {
        "criterion": "C1",
        "description": "Distance to major roads",
        "working_direction": "DECREASING",
        "working_role": "CORE_CANDIDATE",
        "standardized_column": "s_c1_road_pc",
        "research_note": (
            "Closer major-road proximity is provisionally "
            "preferred, but it does not establish parcel access."
        ),
    },
    {
        "criterion": "C2",
        "description": "Road density within 500 metres",
        "working_direction": "INCREASING_PROVISIONAL",
        "working_role": "CORE_CANDIDATE",
        "standardized_column": "s_c2_road_density_pc",
        "research_note": (
            "Neighbourhood road availability is useful, "
            "but redundancy with C1 and saturation "
            "require review."
        ),
    },
    {
        "criterion": "C3",
        "description": "Building context within 500 metres",
        "working_direction": "UNDECIDED",
        "working_role": "CONDITIONAL_CANDIDATE",
        "standardized_column": "NOT_FINALIZED",
        "research_note": (
            "Increasing and decreasing interpretations "
            "can lead to substantially different rankings. "
            "Expert decision required."
        ),
    },
    {
        "criterion": "C4",
        "description": "Terrain slope in degrees",
        "working_direction": "DECREASING",
        "working_role": "CORE_CANDIDATE",
        "standardized_column": "s_c4_slope_pc",
        "research_note": (
            "Gentler terrain is provisionally preferred. "
            "Grid-scale terrain does not replace "
            "site-level engineering assessment."
        ),
    },
    {
        "criterion": "C5",
        "description": "Elevation in metres",
        "working_direction": "UNDECIDED",
        "working_role": "CONTEXT_CANDIDATE",
        "standardized_column": "NOT_FINALIZED",
        "research_note": (
            "A city-wide monotonic suitability direction "
            "has not been justified. Zone-specific rules "
            "cannot automatically apply everywhere."
        ),
    },
]


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_csv(path):

    frame = pd.read_csv(
        path,
        dtype={"grid_id": "string"},
    )

    if "grid_id" in frame.columns:
        frame["grid_id"] = (
            frame["grid_id"].str.strip()
        )

    return frame


def read_json(path):

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_json(path, value):

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def main():

    print("=" * 84)
    print(
        "STEP 16C-N - CRITERIA AND "
        "STANDARDIZATION READINESS"
    )
    print("=" * 84)

    sources = [
        RAW_FILE,
        STANDARD_FILE,
        STANDARD_META,
        BASELINE_FILE,
        BASELINE_META,
        REVIEW_FILE,
        PAIRWISE_FILE,
        AHP_META,
    ]

    for path in sources:

        if not path.is_file():
            raise FileNotFoundError(
                f"Required input missing: {path}"
            )

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    fingerprints = {
        str(path): sha256(path)
        for path in sources
    }

    standard_meta = read_json(STANDARD_META)
    baseline_meta = read_json(BASELINE_META)
    ahp_meta = read_json(AHP_META)

    qa = []

    def check(name, passed, detail=""):

        qa.append({
            "check": name,
            "passed": bool(passed),
            "detail": str(detail),
        })

    def checkpoint():

        failed = [
            item for item in qa
            if not item["passed"]
        ]

        if failed:

            pd.DataFrame(qa).to_csv(
                OUT_QA,
                index=False,
            )

            for item in failed:
                print(
                    "REVIEW:",
                    item["check"],
                    item["detail"],
                )

            raise RuntimeError(
                "Step 16C-N validation failed. "
                "Inspect the saved QA report."
            )

    # --------------------------------------------------------
    # 1. Verify completed research prerequisites
    # --------------------------------------------------------

    print("\n=== VERIFYING RESEARCH INPUTS ===")

    check(
        "Standardization step previously passed QA",
        standard_meta.get("qa_status") == "PASS",
    )

    check(
        "Provisional baseline previously passed QA",
        baseline_meta.get("qa_status") == "PASS",
    )

    check(
        "AHP intake technical QA passed",
        ahp_meta.get("technical_qa_status") == "PASS",
    )

    checkpoint()

    raw = read_csv(RAW_FILE)
    standard = read_csv(STANDARD_FILE)
    baseline = read_csv(BASELINE_FILE)

    for name, frame in [
        ("Raw criteria", raw),
        ("Standardized criteria", standard),
        ("Provisional baseline", baseline),
    ]:

        check(
            f"{name}: 9240 unique nonempty grid IDs",
            "grid_id" in frame.columns
            and len(frame) == EXPECTED_GRID_COUNT
            and frame["grid_id"].notna().all()
            and frame["grid_id"].is_unique,
        )

    checkpoint()

    check(
        "All research grid ID sets match",
        set(raw["grid_id"])
        == set(standard["grid_id"])
        == set(baseline["grid_id"]),
    )

    required_standard = (
        set(CORE_SCORE_COLUMNS.values())
        | set(CONTEXT_COLUMNS.values())
    )

    check(
        "Core normalized and contextual fields exist",
        required_standard.issubset(
            standard.columns
        ),
    )

    check(
        "Baseline contains reference score",
        "baseline_pc" in baseline.columns,
    )

    checkpoint()

    # --------------------------------------------------------
    # 2. Align by permanent grid identifier
    # --------------------------------------------------------

    standard = (
        standard.set_index("grid_id")
        .sort_index()
    )

    baseline = (
        baseline.set_index("grid_id")
        .reindex(standard.index)
    )

    selected = {
        **{
            code: standard[column]
            for code, column
            in CORE_SCORE_COLUMNS.items()
        },
        **{
            code: standard[column]
            for code, column
            in CONTEXT_COLUMNS.items()
        },
    }

    numeric = pd.DataFrame(selected).apply(
        pd.to_numeric,
        errors="coerce",
    )

    core = numeric[["C1", "C2", "C4"]]

    check(
        "All five candidate variables contain finite data",
        np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all(),
    )

    check(
        "Three core scores are normalized to [0,1]",
        core.ge(0).all().all()
        and core.le(1).all().all(),
    )

    check(
        "Three core scores have measurable variation",
        all(
            float(core[col].max() - core[col].min())
            > 1e-12
            for col in core.columns
        ),
    )

    baseline_scores = pd.to_numeric(
        baseline["baseline_pc"],
        errors="coerce",
    ).to_numpy(dtype=float)

    check(
        "Reference baseline matches mean of C1/C2/C4",
        np.isfinite(baseline_scores).all()
        and np.allclose(
            core.mean(axis=1).to_numpy(),
            baseline_scores,
            atol=1e-10,
        ),
    )

    checkpoint()

    # --------------------------------------------------------
    # 3. Spearman criterion association analysis
    # --------------------------------------------------------

    print("\n=== CRITERION ASSOCIATION / REDUNDANCY ===")

    # Spearman correlation examines monotonic association.
    # It does not establish identical meaning, causality,
    # or whether a criterion should be excluded.

    correlation = numeric[
        ["C1", "C2", "C3", "C4", "C5"]
    ].corr(method="spearman")

    pair_rows = []

    for a, b in combinations(
        ["C1", "C2", "C3", "C4", "C5"],
        2,
    ):

        rho = float(
            correlation.loc[a, b]
        )

        if not np.isfinite(rho):
            flag = "UNDEFINED_ASSOCIATION_REVIEW"
        elif abs(rho) >= REDUNDANCY_REVIEW_THRESHOLD:
            flag = "HIGH_ASSOCIATION_REVIEW"
        else:
            flag = "NO_HIGH_ASSOCIATION_FLAG"

        pair_rows.append({
            "criterion_a": a,
            "criterion_b": b,
            "spearman_rho": rho,
            "absolute_rho": abs(rho),
            "review_threshold":
                REDUNDANCY_REVIEW_THRESHOLD,
            "screening_flag": flag,
            "automatically_exclude_criterion": False,
            "interpretation": (
                "Statistical association only; "
                "requires conceptual and planning review."
            ),
        })

    pairs = pd.DataFrame(pair_rows)

    check(
        "Ten pairwise criterion associations calculated",
        len(pairs) == 10,
    )

    check(
        "Five-by-five association matrix generated",
        correlation.shape == (5, 5),
    )

    check(
        "Correlation values valid where defined",
        correlation.stack().between(
            -1, 1
        ).all(),
    )

    checkpoint()

    print(
        correlation.round(4).to_string()
    )

    # --------------------------------------------------------
    # 4. Review questionnaire structure and readiness
    # --------------------------------------------------------

    print("\n=== EXPERT REVIEW READINESS ===")

    review = pd.read_csv(
        REVIEW_FILE,
        dtype=str,
        keep_default_na=False,
    ).fillna("")

    pairwise = pd.read_csv(
        PAIRWISE_FILE,
        dtype=str,
        keep_default_na=False,
    ).fillna("")

    for frame in [review, pairwise]:
        for column in frame.columns:
            frame[column] = (
                frame[column]
                .astype(str)
                .str.strip()
            )

    review_fields = {
        "criterion",
        "expert_decision",
        "expert_direction",
        "expert_reason",
        "reviewer_name",
        "reviewer_role",
        "review_date",
    }

    pair_fields = {
        "scenario",
        "criterion_a",
        "criterion_b",
        "higher_importance",
        "importance_strength_1_to_9",
        "justification",
        "reviewer_name",
        "reviewer_role",
        "review_date",
    }

    check(
        "Expert criteria questionnaire has required fields",
        review_fields.issubset(
            review.columns
        ),
    )

    check(
        "AHP pairwise questionnaire has required fields",
        pair_fields.issubset(
            pairwise.columns
        ),
    )

    checkpoint()

    check(
        "Expert review contains five unique candidates",
        len(review) == 5
        and review["criterion"].is_unique
        and set(review["criterion"])
        == {"C1", "C2", "C3", "C4", "C5"},
    )

    expected_pairs = {
        "BASE_3_CRITERIA": {
            ("C1", "C2"),
            ("C1", "C4"),
            ("C2", "C4"),
        },
        "CONDITIONAL_4_WITH_C3": set(
            combinations(
                ["C1", "C2", "C3", "C4"],
                2,
            )
        ),
    }

    structure_ok = len(pairwise) == 9

    for scenario, expected in expected_pairs.items():

        rows = pairwise.loc[
            pairwise["scenario"] == scenario
        ]

        actual = list(
            zip(
                rows["criterion_a"],
                rows["criterion_b"],
            )
        )

        structure_ok = (
            structure_ok
            and len(actual) == len(expected)
            and len(set(actual)) == len(actual)
            and set(actual) == expected
        )

    check(
        "AHP questionnaire contains correct 3+6 comparisons",
        structure_ok
        and set(pairwise["scenario"])
        == set(expected_pairs),
    )

    checkpoint()

    reviewer_columns = [
        "expert_decision",
        "expert_direction",
        "expert_reason",
        "reviewer_name",
        "reviewer_role",
        "review_date",
    ]

    pair_answer_columns = [
        "higher_importance",
        "importance_strength_1_to_9",
        "justification",
        "reviewer_name",
        "reviewer_role",
        "review_date",
    ]

    criterion_rows_entered = int(
        review[reviewer_columns]
        .ne("")
        .all(axis=1)
        .sum()
    )

    pairwise_rows_entered = int(
        pairwise[pair_answer_columns]
        .ne("")
        .all(axis=1)
        .sum()
    )

    # Entered questionnaire fields are not automatically
    # authenticated or accepted as expert judgments.

    # --------------------------------------------------------
    # 5. Create criterion decision registry
    # --------------------------------------------------------

    registry = pd.DataFrame(
        CRITERIA
    )

    expert_lookup = (
        review
        .set_index("criterion")
        .reindex(registry["criterion"])
        .reset_index()
    )

    for column in reviewer_columns:

        registry[column] = (
            expert_lookup[column].to_numpy()
        )

    registry["approval_status"] = (
        "REQUIRES_AUTHENTIC_REVIEW_AND_VALIDATION"
    )

    registry["final_ahp_weight"] = (
        "NOT_AVAILABLE"
    )

    registry["final_scoring_status"] = (
        "PENDING_CRITERIA_APPROVAL"
    )

    c3_score_columns = [
        col
        for col in standard.columns
        if col.lower().startswith("s_c3")
    ]

    registry.loc[
        registry["criterion"] == "C3",
        "research_note",
    ] += (
        " Available standardized candidates: "
        + (
            ", ".join(c3_score_columns)
            if c3_score_columns
            else "not identified"
        )
        + "."
    )

    # --------------------------------------------------------
    # 6. Final QA and provenance
    # --------------------------------------------------------

    check(
        "All five candidate criteria appear in registry",
        len(registry) == 5
        and registry["criterion"].is_unique,
    )

    check(
        "No AHP weights fabricated by this audit",
        registry["final_ahp_weight"]
        .eq("NOT_AVAILABLE")
        .all(),
    )

    check(
        "All source research files unchanged",
        all(
            sha256(path) == fingerprints[str(path)]
            for path in sources
        ),
    )

    checkpoint()

    # --------------------------------------------------------
    # 7. Save reproducible outputs
    # --------------------------------------------------------

    registry.to_csv(
        OUT_CRITERIA,
        index=False,
    )

    correlation.to_csv(
        OUT_CORRELATION,
        index_label="criterion",
    )

    pairs.to_csv(
        OUT_PAIRS,
        index=False,
    )

    pd.DataFrame(qa).to_csv(
        OUT_QA,
        index=False,
    )

    high_pairs = pairs.loc[
        pairs["screening_flag"]
        == "HIGH_ASSOCIATION_REVIEW"
    ]

    summary = {
        "step": "16C-N",
        "qa_status": "PASS",
        "methodology": "GIS_MCDA_AHP_MANDATORY",
        "grid_count": EXPECTED_GRID_COUNT,
        "candidate_criteria_count": 5,
        "core_working_criteria": [
            "C1", "C2", "C4"
        ],
        "conditional_criterion": "C3",
        "context_criterion": "C5",
        "criteria_final_approval":
            "PENDING",
        "expert_criteria_rows_fully_entered":
            criterion_rows_entered,
        "pairwise_rows_fully_entered":
            pairwise_rows_entered,
        "ahp_status": ahp_meta.get(
            "expert_input_status"
        ),
        "correlation_method": "Spearman",
        "association_review_threshold":
            REDUNDANCY_REVIEW_THRESHOLD,
        "high_association_pair_count":
            len(high_pairs),
        "high_association_pairs":
            high_pairs[
                [
                    "criterion_a",
                    "criterion_b",
                    "spearman_rho",
                ]
            ].to_dict(orient="records"),
        "c3_standardized_candidates":
            c3_score_columns,
        "final_weighted_score_created":
            False,
        "source_hashes": {
            path.name: fingerprints[str(path)]
            for path in sources
        },
        "qa_checks": len(qa),
        "qa_passed": len(qa),
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    save_json(
        OUT_SUMMARY,
        summary,
    )

    # Permanent, concise viva-oriented methodology report.

    lines = [
        "# Step 16C-N: Criteria Readiness Audit",
        "",
        "## Purpose",
        "",
        "Verify the criteria prepared for mandatory "
        "GIS-MCDA with AHP development suitability.",
        "",
        "## Verified study data",
        "",
        f"- Grid cells: {EXPECTED_GRID_COUNT}",
        "- Core working criteria: C1, C2, C4",
        "- Conditional criterion: C3",
        "- Contextual criterion: C5",
        "- Core scoring uses existing percentile-capped "
        "normalization.",
        "",
        "## Redundancy assessment",
        "",
        "Spearman correlation was recalculated across "
        "the five candidate variables.",
        "",
        f"Review threshold: absolute rho >= "
        f"{REDUNDANCY_REVIEW_THRESHOLD}.",
        "",
        "The threshold is a screening convention, "
        "not a rule for automatically excluding criteria.",
        "",
        "High-association pairs:",
        "",
    ]

    if len(high_pairs) == 0:

        lines.append(
            "- None exceeded the selected threshold."
        )

    else:

        for _, row in high_pairs.iterrows():

            lines.append(
                f"- {row['criterion_a']} / "
                f"{row['criterion_b']}: "
                f"rho = {row['spearman_rho']:.4f}"
            )

    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- C1 and C2 may both represent aspects of "
        "road accessibility; statistical association "
        "alone cannot resolve double-counting concerns.",
        "- C3 direction remains unresolved.",
        "- C5 does not have an approved city-wide "
        "monotonic scoring direction.",
        "- The current reference baseline is not "
        "the final AHP-weighted model.",
        "- Provisional zoning and conservation screening "
        "do not establish legal building permission.",
        "",
        "## AHP requirements",
        "",
        f"- Criteria review rows fully entered: "
        f"{criterion_rows_entered}/5",
        f"- Pairwise questionnaire rows fully entered: "
        f"{pairwise_rows_entered}/9",
        f"- AHP intake status: "
        f"{summary['ahp_status']}",
        "",
        "Final approved criteria and genuine AHP "
        "judgments are still required before weighted "
        "suitability calculation.",
        "",
        "## What this audit validates",
        "",
        "Input consistency, scoring availability, "
        "reference reproducibility, criterion "
        "associations, questionnaire structure "
        "and file integrity.",
        "",
        "This audit does not validate expert judgments, "
        "legal development suitability or external "
        "prediction performance.",
        "",
    ])

    OUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 8. Terminal summary
    # --------------------------------------------------------

    print("\n=== CRITERION REGISTRY ===")

    print(
        registry[
            [
                "criterion",
                "working_direction",
                "working_role",
                "approval_status",
            ]
        ].to_string(index=False)
    )

    print("\n=== REDUNDANCY SCREENING ===")

    print(
        pairs[
            [
                "criterion_a",
                "criterion_b",
                "spearman_rho",
                "screening_flag",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n=== AHP READINESS ===")

    print(
        "Criteria review rows entered:",
        f"{criterion_rows_entered}/5"
    )

    print(
        "Pairwise rows entered:",
        f"{pairwise_rows_entered}/9"
    )

    print(
        "AHP status:",
        summary["ahp_status"]
    )

    print("\n=== QA RESULTS ===")

    for item in qa:

        print(
            "PASS:"
            if item["passed"]
            else "REVIEW:",
            item["check"],
        )

    print(
        f"\nOverall QA: PASS ({len(qa)}/{len(qa)})"
    )

    print("\n=== GENERATED REPORTS ===")

    for path in [
        OUT_CRITERIA,
        OUT_CORRELATION,
        OUT_PAIRS,
        OUT_QA,
        OUT_SUMMARY,
        OUT_REPORT,
    ]:

        print(path.relative_to(ROOT))

    print(
        "\nSTATUS: TECHNICAL CRITERIA PREPARATION "
        "VERIFIED; EXPERT APPROVAL PENDING"
    )

    print("=" * 84)


if __name__ == "__main__":
    main()
