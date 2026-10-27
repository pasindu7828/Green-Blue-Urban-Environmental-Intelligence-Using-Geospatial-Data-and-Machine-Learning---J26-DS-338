
"""
STEP 16D-D - REAL EXPERT AHP JUDGMENT INTAKE

Purpose:
1. Validate the existing expert-review templates.
2. Determine the expert-approved criterion set.
3. Check genuine AHP pairwise responses.
4. Calculate principal-eigenvector AHP weights.
5. Calculate the Saaty consistency ratio.
6. Reject incomplete or inconsistent judgments.

No artificial expert opinions are generated.

Blank forms produce PENDING_EXPERT_REVIEW.
No final suitability score is calculated.

One reviewer per questionnaire is supported.
Multiple reviewers require a separate aggregation process.
"""

from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REPORTS = ROOT / "outputs/reports"

REVIEW = (
    REPORTS
    / "16c_l_expert_criteria_review_template.csv"
)

PAIRWISE = (
    REPORTS
    / "16c_l_ahp_pairwise_template.csv"
)

PREP_SUMMARY = (
    REPORTS
    / "16c_l_expert_review_summary.json"
)

BASE_SUMMARY = (
    REPORTS
    / "16d_b_provisional_baseline_summary.json"
)

SPATIAL_SUMMARY = (
    REPORTS
    / "16d_c_spatial_baseline_summary.json"
)

OUT_READINESS = (
    REPORTS
    / "16d_d_ahp_input_readiness.csv"
)

OUT_QA = (
    REPORTS
    / "16d_d_ahp_intake_qa.csv"
)

OUT_WEIGHTS = (
    REPORTS
    / "16d_d_validated_expert_ahp_weights.csv"
)

OUT_SUMMARY = (
    REPORTS
    / "16d_d_ahp_consistency_summary.json"
)


# ============================================================
# 2. AHP CONFIGURATION
# ============================================================

SETS = {
    "BASE_3_CRITERIA": [
        "C1", "C2", "C4"
    ],

    "CONDITIONAL_4_WITH_C3": [
        "C1", "C2", "C3", "C4"
    ],
}

EXPECTED_DIRECTIONS = {
    "C1": "DECREASING",
    "C2": "INCREASING",
    "C4": "DECREASING",
}

REVIEW_EDITABLE = [
    "expert_decision",
    "expert_direction",
    "expert_reason",
    "reviewer_name",
    "reviewer_role",
    "review_date",
]

PAIR_EDITABLE = [
    "higher_importance",
    "importance_strength_1_to_9",
    "justification",
    "reviewer_name",
    "reviewer_role",
    "review_date",
]

DECISIONS = {
    "INCLUDE",
    "EXCLUDE",
    "REVISE",
    "UNCERTAIN",
}

DIRECTIONS = {
    "INCREASING",
    "DECREASING",
    "NON_MONOTONIC",
    "NOT_APPLICABLE",
    "UNCERTAIN",
}

# Standard Saaty random index values
# for the supported matrix sizes.

RI = {
    3: 0.58,
    4: 0.90,
}

CR_LIMIT = 0.10


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as source:

        for block in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_form(path):

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    for column in df.columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    return df


def save_json(path, record):

    path.write_text(
        json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# 4. MAIN
# ============================================================

def main():

    print("=" * 88)

    print(
        "STEP 16D-D - REAL EXPERT "
        "AHP JUDGMENT INTAKE"
    )

    print("=" * 88)

    required = [
        REVIEW,
        PAIRWISE,
        PREP_SUMMARY,
        BASE_SUMMARY,
        SPATIAL_SUMMARY,
    ]

    for path in required:

        if not path.is_file():

            raise FileNotFoundError(path)

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    originals = {
        str(path): sha256(path)
        for path in required
    }

    summaries = [
        json.loads(
            path.read_text(encoding="utf-8")
        )
        for path in [
            PREP_SUMMARY,
            BASE_SUMMARY,
            SPATIAL_SUMMARY,
        ]
    ]

    r = read_form(REVIEW)

    p = read_form(PAIRWISE)

    qa = []

    def check(name, condition, detail=""):

        qa.append({
            "check": name,
            "passed": bool(condition),
            "detail": str(detail),
        })

    def guard():

        if not all(
            item["passed"]
            for item in qa
        ):

            pd.DataFrame(qa).to_csv(
                OUT_QA,
                index=False,
            )

            for item in qa:

                if not item["passed"]:

                    print(
                        "REVIEW:",
                        item["check"],
                        item["detail"],
                    )

            raise RuntimeError(
                "Input structure QA failed. "
                "Review Step 16D-D QA report."
            )

    # ========================================================
    # 5. PREREQUISITE VALIDATION
    # ========================================================

    check(
        "16C-L, 16D-B and 16D-C prerequisite QA passed",
        all(
            item.get("qa_status") == "PASS"
            for item in summaries
        ),
    )

    check(
        "Research baseline remains provisional",
        summaries[1].get("methodology_status")
        == "PROVISIONAL_REFERENCE_BASELINE_ONLY",
    )

    check(
        "Expert preparation did not invent judgments",
        summaries[0].get(
            "expert_judgments_collected_by_script"
        ) is False,
    )

    check(
        "Review form contains five expected criteria",
        "criterion" in r.columns
        and len(r) == 5
        and set(r["criterion"])
        == {"C1", "C2", "C3", "C4", "C5"}
        and r["criterion"].is_unique,
    )

    check(
        "Required editable review columns available",
        set(REVIEW_EDITABLE).issubset(
            r.columns
        ),
    )

    check(
        "Pairwise form has required columns",
        {
            "scenario",
            "criterion_a",
            "criterion_b",
            *PAIR_EDITABLE,
        }.issubset(p.columns),
    )

    # ========================================================
    # 6. CHECK AHP QUESTIONNAIRE STRUCTURE
    # ========================================================

    required_pair_columns = {
        "scenario",
        "criterion_a",
        "criterion_b",
    }

    if required_pair_columns.issubset(
        p.columns
    ):

        structure_valid = all(

            len(
                p[p["scenario"] == scenario]
            ) == len(
                set(combinations(codes, 2))
            )

            and set(
                map(
                    tuple,
                    p.loc[
                        p["scenario"] == scenario,
                        [
                            "criterion_a",
                            "criterion_b",
                        ],
                    ].to_numpy(),
                )
            ) == set(combinations(codes, 2))

            for scenario, codes
            in SETS.items()

        )

        structure_valid = (
            structure_valid
            and len(p) == 9
            and set(p["scenario"]) == set(SETS)
        )

    else:

        structure_valid = False

    check(
        "AHP alternatives contain exactly 3 and 6 required comparisons",
        structure_valid,
    )

    guard()

    # ========================================================
    # 7. CHECK EXPERT REVIEW COMPLETENESS
    # ========================================================

    r = (
        r.set_index("criterion")
        .reindex([
            "C1", "C2", "C3", "C4", "C5"
        ])
    )

    review_work_started = (
        r[REVIEW_EDITABLE]
        .ne("")
        .any()
        .any()
    )

    review_ready = bool(

        r["expert_decision"]
        .str.upper()
        .isin(DECISIONS)
        .all()

        and r["expert_reason"]
        .ne("")
        .all()

        and r[
            [
                "reviewer_name",
                "reviewer_role",
                "review_date",
            ]
        ].ne("").all().all()

        and all(

            (
                r.loc[
                    criterion,
                    "expert_direction",
                ].upper() in DIRECTIONS
            )

            if r.loc[
                criterion,
                "expert_decision",
            ].upper() == "INCLUDE"

            else True

            for criterion in r.index

        )
    )

    # ========================================================
    # 8. CHECK PAIRWISE FORM READINESS
    # ========================================================

    readiness_records = []

    for scenario, codes in SETS.items():

        selected = p[
            p["scenario"] == scenario
        ]

        complete_rows = (
            selected[PAIR_EDITABLE]
            .ne("")
            .all(axis=1)
        )

        partially_completed = (
            selected[PAIR_EDITABLE]
            .ne("")
            .any(axis=1)
            & ~complete_rows
        )

        readiness_records.append({
            "scenario": scenario,

            "criteria": ",".join(codes),

            "total_pairs": len(selected),

            "fully_answered_pairs":
                int(complete_rows.sum()),

            "partially_answered_pairs":
                int(partially_completed.sum()),

            "all_pairs_answered":
                bool(complete_rows.all()),

            "eligible_for_calculation":
                False,
        })

    readiness = pd.DataFrame(
        readiness_records
    )

    # ========================================================
    # 9. INITIAL PROCESSING STATUS
    # ========================================================

    active = None

    computed = None

    status = "PENDING_EXPERT_REVIEW"

    explanation = (
        "Criteria review form has not been "
        "fully and validly completed."
    )

    # ========================================================
    # 10. DETERMINE APPROVED CRITERION SET
    # ========================================================

    if review_ready:

        decisions = (
            r["expert_decision"]
            .str.upper()
        )

        approved = [
            code
            for code in r.index
            if decisions.loc[code] == "INCLUDE"
        ]

        if decisions.isin([
            "REVISE",
            "UNCERTAIN",
        ]).any():

            status = "CRITERIA_REDRAFT_REQUIRED"

            explanation = (
                "At least one criterion is marked "
                "REVISE or UNCERTAIN. Resolve all "
                "decisions before AHP."
            )

        elif set(approved) == set(
            SETS["BASE_3_CRITERIA"]
        ):

            active = "BASE_3_CRITERIA"

        elif set(approved) == set(
            SETS["CONDITIONAL_4_WITH_C3"]
        ):

            active = "CONDITIONAL_4_WITH_C3"

        else:

            status = "CRITERIA_REDRAFT_REQUIRED"

            explanation = (
                f"Reviewed criterion set {approved} "
                "is not covered by the questionnaires."
            )

        # ====================================================
        # 11. CHECK SCORING DIRECTIONS
        # ====================================================

        if active is not None:

            base_directions_match = all(

                r.loc[
                    code,
                    "expert_direction",
                ].upper() == direction

                for code, direction
                in EXPECTED_DIRECTIONS.items()
            )

            c3_direction_ok = (

                active == "BASE_3_CRITERIA"

                or r.loc[
                    "C3",
                    "expert_direction",
                ].upper() in {
                    "INCREASING",
                    "DECREASING",
                }
            )

            if not (
                base_directions_match
                and c3_direction_ok
            ):

                status = "SCORING_REDRAFT_REQUIRED"

                explanation = (
                    "Expert directions differ from "
                    "existing scoring assumptions. "
                    "Revise normalization before "
                    "using AHP weights."
                )

            else:

                selected = p[
                    p["scenario"] == active
                ]

                # ============================================
                # 12. CHECK COMPLETE AHP RESPONSES
                # ============================================

                pairwise_complete = bool(
                    selected[PAIR_EDITABLE]
                    .ne("")
                    .all()
                    .all()
                )

                if not pairwise_complete:

                    status = (
                        "PENDING_PAIRWISE_JUDGMENTS"
                    )

                    explanation = (
                        f"Selected questionnaire {active} "
                        f"needs all {len(selected)} "
                        "comparisons completed."
                    )

                else:

                    errors = []

                    for index, entry in selected.iterrows():

                        side = (
                            entry["higher_importance"]
                            .upper()
                        )

                        strength_string = (
                            entry[
                                "importance_strength_1_to_9"
                            ]
                        )

                        if (
                            not strength_string.isdecimal()
                            or not 1
                            <= int(strength_string)
                            <= 9
                        ):

                            errors.append(
                                f"Row {index + 1}: "
                                "invalid Saaty scale"
                            )

                            continue

                        strength = int(
                            strength_string
                        )

                        if (
                            side not in {
                                "A",
                                "B",
                                "EQUAL",
                            }

                            or (
                                side == "EQUAL"
                                and strength != 1
                            )

                            or (
                                side in {"A", "B"}
                                and strength == 1
                            )
                        ):

                            errors.append(
                                f"Row {index + 1}: "
                                "importance/strength "
                                "contradiction"
                            )

                    # ========================================
                    # 13. VERIFY REVIEWER CONSISTENCY
                    # ========================================

                    pair_reviewers = {

                        (
                            row["reviewer_name"]
                            .casefold(),

                            row["reviewer_role"]
                            .casefold(),
                        )

                        for _, row
                        in selected.iterrows()
                    }

                    criteria_reviewers = {

                        (
                            row["reviewer_name"]
                            .casefold(),

                            row["reviewer_role"]
                            .casefold(),
                        )

                        for _, row
                        in r.iterrows()
                    }

                    if (
                        len(pair_reviewers) != 1
                        or pair_reviewers
                        != criteria_reviewers
                    ):

                        errors.append(
                            "Criteria and pairwise forms "
                            "must belong to the same "
                            "single reviewer."
                        )

                    if errors:

                        status = (
                            "JUDGMENT_REVIEW_REQUIRED"
                        )

                        explanation = "; ".join(
                            errors
                        )

                    else:

                        # ====================================
                        # 14. CONSTRUCT RECIPROCAL AHP MATRIX
                        # ====================================

                        codes = SETS[active]

                        indices = {
                            code: i
                            for i, code
                            in enumerate(codes)
                        }

                        n = len(codes)

                        matrix = np.ones(
                            (n, n),
                            dtype=float,
                        )

                        for _, entry in selected.iterrows():

                            i = indices[
                                entry["criterion_a"]
                            ]

                            j = indices[
                                entry["criterion_b"]
                            ]

                            side = (
                                entry["higher_importance"]
                                .upper()
                            )

                            strength = int(
                                entry[
                                    "importance_strength_1_to_9"
                                ]
                            )

                            if side == "EQUAL":

                                value = 1.0

                            elif side == "A":

                                value = float(
                                    strength
                                )

                            else:

                                value = (
                                    1.0 / strength
                                )

                            matrix[i, j] = value

                            matrix[j, i] = (
                                1.0 / value
                            )

                        # ====================================
                        # 15. PRINCIPAL EIGENVECTOR METHOD
                        # ====================================

                        eigenvalues, eigenvectors = (
                            np.linalg.eig(matrix)
                        )

                        index = int(
                            np.argmax(
                                eigenvalues.real
                            )
                        )

                        lambda_max = float(
                            eigenvalues[index].real
                        )

                        vector = (
                            eigenvectors[:, index]
                            .real
                        )

                        if vector.sum() < 0:

                            vector = -vector

                        if not np.all(
                            vector > 0
                        ):

                            raise RuntimeError(
                                "Principal eigenvector "
                                "must be positive."
                            )

                        weights = (
                            vector / vector.sum()
                        )

                        # ====================================
                        # 16. CONSISTENCY VALIDATION
                        # ====================================

                        ci = max(
                            0.0,

                            (
                                lambda_max - n
                            ) / (n - 1),
                        )

                        cr = (
                            ci / RI[n]
                        )

                        computed = {
                            "scenario": active,

                            "criteria": codes,

                            "matrix":
                                matrix.tolist(),

                            "weights": {

                                code: float(weight)

                                for code, weight
                                in zip(
                                    codes,
                                    weights,
                                )
                            },

                            "lambda_max":
                                lambda_max,

                            "consistency_index":
                                float(ci),

                            "consistency_ratio":
                                float(cr),

                            "consistency_limit":
                                CR_LIMIT,

                            "random_index":
                                RI[n],
                        }

                        if cr <= CR_LIMIT + 1e-10:

                            status = (
                                "EXPERT_AHP_VALIDATED"
                            )

                            explanation = (
                                "Completed expert input "
                                "passed structural and "
                                "AHP consistency checks."
                            )

                        else:

                            status = (
                                "CONSISTENCY_REVIEW_REQUIRED"
                            )

                            explanation = (
                                f"AHP CR {cr:.4f} "
                                "exceeds 0.10. "
                                "Ask the reviewer to "
                                "reconsider the judgments."
                            )

    # ========================================================
    # 17. RECORD SCENARIO READINESS
    # ========================================================

    if active is not None:

        readiness.loc[
            readiness["scenario"] == active,

            "eligible_for_calculation",

        ] = (
            status == "EXPERT_AHP_VALIDATED"
        )

    # ========================================================
    # 18. SAVE WEIGHTS ONLY WHEN VALIDATED
    # ========================================================

    if status == "EXPERT_AHP_VALIDATED":

        weights_table = pd.DataFrame([
            {
                "criterion": code,

                "ahp_weight":
                    computed["weights"][code],

                "scenario": active,

                "reviewer_name":
                    r.iloc[0]["reviewer_name"],

                "reviewer_role":
                    r.iloc[0]["reviewer_role"],

                "review_csv_sha256":
                    originals[str(REVIEW)],

                "pairwise_csv_sha256":
                    originals[str(PAIRWISE)],
            }

            for code in computed["criteria"]
        ])

        weights_table.to_csv(
            OUT_WEIGHTS,
            index=False,
        )

        check(
            "Validated AHP weights sum to one",

            abs(
                weights_table[
                    "ahp_weight"
                ].sum() - 1
            ) < 1e-10,
        )

        calculated_matrix = np.asarray(
            computed["matrix"]
        )

        check(
            "AHP matrix is positive and reciprocal",

            np.all(
                calculated_matrix > 0
            )

            and np.allclose(
                calculated_matrix
                * calculated_matrix.T,
                1,
            ),
        )

    else:

        check(
            "No unverified AHP weights calculated in this run",
            computed is None
            or status != "EXPERT_AHP_VALIDATED",
        )

    # ========================================================
    # 19. FINAL INPUT INTEGRITY CHECK
    # ========================================================

    check(
        "Expert forms and prerequisite summaries unchanged",

        all(
            sha256(path)
            == originals[str(path)]

            for path in required
        ),
    )

    # ========================================================
    # 20. SAVE REPORTS
    # ========================================================

    pd.DataFrame(qa).to_csv(
        OUT_QA,
        index=False,
    )

    readiness.to_csv(
        OUT_READINESS,
        index=False,
    )

    technical_status = (
        "PASS"

        if all(
            item["passed"]
            for item in qa
        )

        else "REVIEW_REQUIRED"
    )

    result = {
        "step": "16D-D",

        "technical_qa_status":
            technical_status,

        "expert_input_status":
            status,

        "selected_scenario":
            active,

        "criterion_review_complete":
            review_ready,

        "criterion_review_has_any_input":
            bool(review_work_started),

        "explanation":
            explanation,

        "input_sha256": {

            path.name: originals[str(path)]

            for path in required
        },

        "ahp": computed,

        "weights_current_and_validated":
            status == "EXPERT_AHP_VALIDATED",

        "warning": (
            "Do not use pre-existing weight files "
            "unless this summary validates "
            "the current expert inputs."
        ),

        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "qa_checks":
            len(qa),

        "qa_passed":
            sum(
                item["passed"]
                for item in qa
            ),

        "outputs": [
            OUT_QA.name,
            OUT_READINESS.name,
            OUT_SUMMARY.name,
        ] + (

            [OUT_WEIGHTS.name]

            if status == "EXPERT_AHP_VALIDATED"

            else []
        ),
    }

    save_json(
        OUT_SUMMARY,
        result,
    )

    # ========================================================
    # 21. PRINT TERMINAL RESULTS
    # ========================================================

    print("\n=== REAL EXPERT INPUT STATUS ===")

    print(
        "Criteria form complete:",
        review_ready,
    )

    print(
        readiness[
            [
                "scenario",
                "total_pairs",
                "fully_answered_pairs",
                "partially_answered_pairs",
            ]
        ].to_string(index=False)
    )

    print(
        "Selected scenario:",
        active or "NONE",
    )

    print(
        "Status:",
        status,
    )

    print(
        "Explanation:",
        explanation,
    )

    if computed is not None:

        print(
            "Computed weights:",
            computed["weights"],
        )

        print(
            "Consistency ratio:",
            round(
                computed["consistency_ratio"],
                6,
            ),
        )

    print("\n=== QA ===")

    for item in qa:

        print(
            "PASS:"
            if item["passed"]
            else "REVIEW:",

            item["check"],
        )

    print("\n=== GENERATED REPORTS ===")

    for path in [
        OUT_READINESS,
        OUT_QA,
        OUT_SUMMARY,
    ]:

        print(
            path.relative_to(ROOT)
        )

    if status == "EXPERT_AHP_VALIDATED":

        print(
            OUT_WEIGHTS.relative_to(ROOT)
        )

    print(
        "\nNo final suitability score calculated."
    )

    print("=" * 88)


# ============================================================
# 22. EXECUTE
# ============================================================

if __name__ == "__main__":
    main()
