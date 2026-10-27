
"""
STEP 16C-K - CRITERION INCLUSION AND SCORING ROBUSTNESS

General urban development suitability (Kaduwela, 100-m grid).
This is a reproducible decision-support experiment, not a final model.

No AHP weights are estimated. No planning permissions are inferred.
No UDA zoning is used as a score, and frozen inputs are read-only.
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
REPORTS = ROOT / "outputs" / "reports"

RAW = (
    ROOT
    / "data/processed/development_suitability/raw_criteria"
    / "16b_raw_suitability_criteria_100m.csv"
)

CANDIDATES = (
    REPORTS
    / "16c_f_candidate_scoring_scenarios_100m.csv"
)

F_SUMMARY = (
    REPORTS / "16c_f_scoring_method_summary.json"
)

G_SUMMARY = (
    REPORTS / "16c_g_scoring_sensitivity_summary.json"
)

J_SUMMARY = (
    REPORTS / "16c_j_zoning_reconciliation_summary.json"
)

OUT_CELLS = (
    REPORTS
    / "16c_k_criterion_inclusion_scenarios_100m.csv"
)

OUT_COMPARISON = (
    REPORTS / "16c_k_scenario_rank_stability.csv"
)

OUT_DECISIONS = (
    REPORTS / "16c_k_pending_methodology_decisions.csv"
)

OUT_QA = (
    REPORTS / "16c_k_criterion_robustness_qa.csv"
)

OUT_SUMMARY = (
    REPORTS / "16c_k_criterion_robustness_summary.json"
)

EXPECTED_ROWS = 9240

# Diagnostic only; not an official planning threshold.
TOP_FRACTION = 0.10


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def file_sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as source:

        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):

    with path.open(encoding="utf-8") as source:
        return json.load(source)


def top_decile_mask(values):
    """
    Include all grid cells tied at the 90th percentile.
    Avoid arbitrary selection based on grid ID order.
    """

    cutoff = float(
        values.quantile(1 - TOP_FRACTION)
    )

    return values >= cutoff - 1e-12


def compare_scenarios(
    name,
    a_name,
    b_name,
    scored,
):

    a = scored[a_name]
    b = scored[b_name]

    difference = (a - b).abs()

    top_a = top_decile_mask(a)
    top_b = top_decile_mask(b)

    intersection = int(
        (top_a & top_b).sum()
    )

    union = int(
        (top_a | top_b).sum()
    )

    disagreement = int(
        (top_a ^ top_b).sum()
    )

    rank_a = a.rank(method="average")
    rank_b = b.rank(method="average")

    return {
        "comparison": name,

        "scenario_a": a_name,
        "scenario_b": b_name,

        "mean_absolute_score_change":
            float(difference.mean()),

        "p95_absolute_score_change":
            float(difference.quantile(0.95)),

        "max_absolute_score_change":
            float(difference.max()),

        "spearman_rank_correlation":
            float(a.corr(
                b,
                method="spearman",
            )),

        "mean_absolute_rank_fraction":
            float(
                (rank_a - rank_b)
                .abs()
                .mean()
                / (len(a) - 1)
            ),

        "top_decile_a_count": int(
            top_a.sum()
        ),

        "top_decile_b_count": int(
            top_b.sum()
        ),

        "top_decile_jaccard":
            float(intersection / union)
            if union else np.nan,

        "top_decile_membership_disagreement_count":
            disagreement,

        "top_decile_membership_disagreement_pct":
            float(
                disagreement / len(a) * 100
            ),

        "decision_status": "DIAGNOSTIC_ONLY",
    }


# ============================================================
# 3. MAIN
# ============================================================

def main():

    print("=" * 93)
    print(
        "STEP 16C-K - CRITERION INCLUSION AND SCORING ROBUSTNESS"
    )
    print("=" * 93)

    required_files = [
        RAW,
        CANDIDATES,
        F_SUMMARY,
        G_SUMMARY,
        J_SUMMARY,
    ]

    for path in required_files:

        if not path.is_file():

            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    # --------------------------------------------------------
    # 3.1 LOAD PREVIOUS RESULTS
    # --------------------------------------------------------

    f = load_json(F_SUMMARY)
    g = load_json(G_SUMMARY)
    j = load_json(J_SUMMARY)

    raw_hash = file_sha256(RAW)
    candidate_hash = file_sha256(CANDIDATES)

    raw = pd.read_csv(
        RAW,
        usecols=["grid_id"],
        dtype={"grid_id": "string"},
    )

    candidates = pd.read_csv(
        CANDIDATES,
        dtype={"grid_id": "string"},
    )

    checks = []

    def check(name, passed, detail):

        checks.append({
            "check": name,
            "passed": bool(passed),
            "detail": str(detail),
        })

    # --------------------------------------------------------
    # 3.2 CHECK INPUT CONSISTENCY
    # --------------------------------------------------------

    check(
        "Step 16C-F and G passed",
        f.get("qa_status") == "PASS"
        and g.get("qa_status") == "PASS",
        "Previous scoring experiments",
    )

    check(
        "Step 16C-J passed",
        j.get("qa_status") == "PASS",
        "Zoning coverage reconciled",
    )

    check(
        "Earlier scenarios remain experimental",
        f.get("methodology_status") == "EXPERIMENTAL"
        and g.get("methodology_status") == "EXPERIMENTAL",
        "No final score",
    )

    check(
        "All raw-input checksums match",
        all(
            value == raw_hash
            for value in [
                f.get("input_sha256"),
                g.get("input_sha256"),
                j.get("frozen_input_sha256"),
            ]
        ),
        "Frozen Step 16B data",
    )

    check(
        "Step 16C-G used these exact candidate scores",
        candidate_hash
        == g.get("scoring_input_sha256"),
        "Experimental CSV checksum",
    )

    check(
        "9240 unique, aligned grid cells",
        len(raw) == len(candidates) == EXPECTED_ROWS
        and raw["grid_id"].notna().all()
        and raw["grid_id"].is_unique
        and candidates["grid_id"].is_unique
        and raw["grid_id"].equals(
            candidates["grid_id"]
        ),
        "Exact row-to-grid matching",
    )

    check(
        "Supplied zoning coverage fully reconciled",
        j.get("reconciled_metric_count")
        == j.get("total_compared_metrics")
        == 10,
        "Zoning remains separate from criterion scoring",
    )

    required_columns = [
        "c1_pctclip_decreasing",
        "c2_pctclip_increasing",
        "c4_pctclip_decreasing",
        "c3_pctclip_increasing",
        "c3_pctclip_decreasing",
        "c1_rank_decreasing",
        "c2_rank_increasing",
        "c4_rank_decreasing",
        "c3_rank_increasing",
        "c3_rank_decreasing",
    ]

    check(
        "Ten needed experimental score columns available",
        all(
            column in candidates.columns
            for column in required_columns
        ),
        "C1, C2, C3, C4",
    )

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    initial_qa = pd.DataFrame(checks)

    if not initial_qa["passed"].all():

        initial_qa.to_csv(
            OUT_QA,
            index=False,
        )

        raise RuntimeError(
            "Initial QA failed. Review Step 16C-K QA CSV."
        )

    # --------------------------------------------------------
    # 3.3 VALIDATE EXPERIMENTAL SCORE VALUES
    # --------------------------------------------------------

    numeric = candidates[
        required_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    valid_scores = bool(
        np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all()
        and numeric.ge(0).all().all()
        and numeric.le(1).all().all()
    )

    check(
        "Input scores finite and between zero and one",
        valid_scores,
        "Scores from Step 16C-F",
    )

    if not valid_scores:

        pd.DataFrame(checks).to_csv(
            OUT_QA,
            index=False,
        )

        raise RuntimeError(
            "Invalid experimental score data."
        )

    # ========================================================
    # 4. BUILD DIAGNOSTIC SCENARIOS
    # ========================================================

    # These are equal-contribution diagnostic averages.
    #
    # Three-core example: C1 + C2 + C4, each divided by 3.
    # Four-core example: C1 + C2 + C3 + C4, divided by 4.
    #
    # These are NOT expert AHP weights or final scores.

    scored = pd.DataFrame({
        "grid_id": raw["grid_id"]
    })

    for method in [
        "pctclip",
        "rank",
    ]:

        base = (
            numeric[f"c1_{method}_decreasing"]
            + numeric[f"c2_{method}_increasing"]
            + numeric[f"c4_{method}_decreasing"]
        )

        # Scenario A: Without C3
        scored[
            f"{method}_three_core_diagnostic"
        ] = base / 3.0

        # Scenario B: C3 increasing
        scored[
            f"{method}_four_core_c3_increasing_diagnostic"
        ] = (
            base
            + numeric[f"c3_{method}_increasing"]
        ) / 4.0

        # Scenario C: C3 decreasing
        scored[
            f"{method}_four_core_c3_decreasing_diagnostic"
        ] = (
            base
            + numeric[f"c3_{method}_decreasing"]
        ) / 4.0

    scenario_columns = [
        column
        for column in scored.columns
        if column != "grid_id"
    ]

    # ========================================================
    # 5. DEFINE SCENARIO COMPARISONS
    # ========================================================

    pairs = []

    for method in [
        "pctclip",
        "rank",
    ]:

        base = (
            f"{method}_three_core_diagnostic"
        )

        increasing = (
            f"{method}_four_core_c3_increasing_diagnostic"
        )

        decreasing = (
            f"{method}_four_core_c3_decreasing_diagnostic"
        )

        pairs.extend([
            (
                f"C3_inclusion_increasing_{method}",
                base,
                increasing,
            ),
            (
                f"C3_inclusion_decreasing_{method}",
                base,
                decreasing,
            ),
            (
                f"C3_direction_{method}",
                increasing,
                decreasing,
            ),
        ])

    # Also compare the two scoring methods.

    for label, suffix in [
        (
            "method_three_core",
            "three_core_diagnostic",
        ),
        (
            "method_four_core_C3_increasing",
            "four_core_c3_increasing_diagnostic",
        ),
        (
            "method_four_core_C3_decreasing",
            "four_core_c3_decreasing_diagnostic",
        ),
    ]:

        pairs.append((
            label,
            f"pctclip_{suffix}",
            f"rank_{suffix}",
        ))

    comparisons = pd.DataFrame([
        compare_scenarios(
            label,
            scenario_a,
            scenario_b,
            scored,
        )
        for label, scenario_a, scenario_b in pairs
    ])

    # ========================================================
    # 6. RECORD REMAINING METHODOLOGY DECISIONS
    # ========================================================

    decisions = pd.DataFrame([
        {
            "criterion": "C1",
            "working_role": "CORE_CANDIDATE",
            "working_direction": "DECREASING",
            "decision_needed": (
                "Choose percentile-capped versus rank "
                "scoring; review road access limitations."
            ),
        },
        {
            "criterion": "C2",
            "working_role": "CORE_CANDIDATE",
            "working_direction": "INCREASING_PROVISIONAL",
            "decision_needed": (
                "Confirm 500m road-density scale "
                "and possible saturation."
            ),
        },
        {
            "criterion": "C3",
            "working_role": "CONDITIONAL_CANDIDATE",
            "working_direction": "UNDECIDED",
            "decision_needed": (
                "Expert or planning justification required "
                "before choosing a development-context "
                "benefit direction."
            ),
        },
        {
            "criterion": "C4",
            "working_role": "CORE_CANDIDATE",
            "working_direction": "DECREASING",
            "decision_needed": (
                "Choose slope scoring and review "
                "terrain-proxy-cell sensitivity."
            ),
        },
        {
            "criterion": "C5",
            "working_role": "CONTEXT_OR_SCREENING_CANDIDATE",
            "working_direction": "UNDECIDED",
            "decision_needed": (
                "No city-wide monotonic elevation direction "
                "established. Do not assume flood risk "
                "or general 25m/30m thresholds."
            ),
        },
    ])

    # ========================================================
    # 7. FINAL QA
    # ========================================================

    check(
        "Exactly six diagnostic composite columns",
        len(scenario_columns) == 6
        and len(set(scenario_columns)) == 6,
        "Three for each scoring method",
    )

    check(
        "All diagnostic scenario values valid 0-1",
        bool(
            np.isfinite(
                scored[scenario_columns]
                .to_numpy(dtype=float)
            ).all()
            and scored[scenario_columns]
            .ge(0).all().all()
            and scored[scenario_columns]
            .le(1).all().all()
        ),
        "Read-only provisional scenarios",
    )

    check(
        "Scenarios show spatial variation",
        bool(
            scored[scenario_columns]
            .nunique()
            .gt(1)
            .all()
        ),
        "All grid scenarios nonconstant",
    )

    check(
        "Exactly nine scenario comparisons",
        len(comparisons) == 9,
        "Inclusion, direction and method comparisons",
    )

    metric_columns = [
        "mean_absolute_score_change",
        "p95_absolute_score_change",
        "max_absolute_score_change",
        "spearman_rank_correlation",
        "mean_absolute_rank_fraction",
        "top_decile_jaccard",
        "top_decile_membership_disagreement_pct",
    ]

    check(
        "All ranking-comparison metrics finite",
        bool(
            np.isfinite(
                comparisons[metric_columns]
                .to_numpy(dtype=float)
            ).all()
        ),
        "No NaN or infinity",
    )

    check(
        "All top-decile statistics valid",
        bool(
            comparisons["top_decile_jaccard"]
            .between(0, 1).all()
            and comparisons[
                "top_decile_membership_disagreement_pct"
            ].between(0, 100).all()
            and comparisons[
                "top_decile_a_count"
            ].ge(1).all()
            and comparisons[
                "top_decile_b_count"
            ].ge(1).all()
        ),
        "Ties included at p90",
    )

    check(
        "Opposite C3 scenarios are represented",
        set(decisions["criterion"])
        == {"C1", "C2", "C3", "C4", "C5"}
        and decisions.loc[
            decisions["criterion"] == "C3",
            "working_direction"
        ].iloc[0] == "UNDECIDED",
        "No forced C3 decision",
    )

    check(
        "C5 not included in diagnostic composites",
        not any(
            "c5" in column.lower()
            for column in scenario_columns
        ),
        "Zone-specific restrictions not generalized",
    )

    check(
        "Comparisons explicitly labelled diagnostic only",
        bool(
            comparisons["decision_status"]
            .eq("DIAGNOSTIC_ONLY")
            .all()
        ),
        "These are not final AHP scores",
    )

    check(
        "Frozen raw input unchanged",
        file_sha256(RAW) == raw_hash,
        "SHA256 stable",
    )

    check(
        "Candidate scoring input unchanged",
        file_sha256(CANDIDATES) == candidate_hash,
        "SHA256 stable",
    )

    qa = pd.DataFrame(checks)

    status = (
        "PASS"
        if bool(qa["passed"].all())
        else "REVIEW_REQUIRED"
    )

    qa.to_csv(
        OUT_QA,
        index=False,
    )

    if status != "PASS":

        print(qa.to_string(index=False))

        raise RuntimeError(
            "Step 16C-K QA failed. "
            "Other outputs were not overwritten."
        )

    # ========================================================
    # 8. SAVE RESEARCH OUTPUTS
    # ========================================================

    scored.to_csv(
        OUT_CELLS,
        index=False,
    )

    comparisons.to_csv(
        OUT_COMPARISON,
        index=False,
    )

    decisions.to_csv(
        OUT_DECISIONS,
        index=False,
    )

    summary = {
        "step": "16C-K",

        "qa_status": status,

        "methodology_status":
            "DIAGNOSTIC_ONLY_FINAL_CRITERIA_PENDING",

        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "grid_count": len(raw),

        "frozen_raw_sha256": raw_hash,

        "candidate_scoring_sha256": candidate_hash,

        "scenario_count": len(scenario_columns),

        "scenario_comparisons": len(comparisons),

        "qa_checks": len(qa),

        "qa_passed": int(
            qa["passed"].sum()
        ),

        "interpretation": [
            (
                "Equal-contribution averages (1/3 or 1/4) "
                "are diagnostic scenarios, not stakeholder "
                "AHP weights or final GIS-MCDA scores."
            ),
            (
                "Top decile is the upper ten percent of "
                "illustrative scores, with threshold ties "
                "included."
            ),
            (
                "C3 sensitivity cannot prove which direction "
                "is substantively correct; planning or "
                "expert justification is required."
            ),
            (
                "C5 remains outside weighted criteria "
                "until a defensible city-wide role "
                "is established."
            ),
            (
                "UDA zoning remains provisional and legally "
                "unverified, and is not included in these "
                "diagnostic scores."
            ),
            (
                "500m C2/C3 radii and both scoring methods "
                "remain provisional."
            ),
            (
                "Current building footprints must not be "
                "used as past predictors for historical "
                "transition evaluation because this could "
                "introduce look-ahead bias."
            ),
        ],

        "outputs": {
            "grid_scenarios": OUT_CELLS.name,
            "scenario_comparison": OUT_COMPARISON.name,
            "pending_decisions": OUT_DECISIONS.name,
            "qa": OUT_QA.name,
            "summary": OUT_SUMMARY.name,
        },
    }

    with OUT_SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # 9. TERMINAL REPORT
    # ========================================================

    print("\n=== SCENARIO ROBUSTNESS ===")

    print(
        comparisons[
            [
                "comparison",
                "mean_absolute_score_change",
                "spearman_rank_correlation",
                "top_decile_jaccard",
                "top_decile_membership_disagreement_pct",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\n=== QA RESULTS ===")

    for entry in checks:

        outcome = (
            "PASS"
            if entry["passed"]
            else "REVIEW"
        )

        print(
            f"{outcome}: {entry['check']}"
        )

    print("\n=== GENERATED REPORTS ===")

    for path in [
        OUT_CELLS,
        OUT_COMPARISON,
        OUT_DECISIONS,
        OUT_QA,
        OUT_SUMMARY,
    ]:

        print(
            path.relative_to(ROOT)
        )

    print("\nOverall QA:", status)

    print(
        "Methodology: DIAGNOSTIC ONLY. "
        "No final model, AHP or legal permission."
    )

    print("=" * 93)


# ============================================================
# 10. EXECUTE
# ============================================================

if __name__ == "__main__":
    main()
