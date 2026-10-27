
"""
J26-DS-338 | Section 03 | Expert AHP Response Documentation

Record the researcher's report of an Engineer's AHP judgments.

Consultation date: 2026-10-17
Scenario: BASE_3_CRITERIA
Criteria: C1, C2, C4

Actions:
1. Load the original nine-question AHP template.
2. Record the three latest reported expert responses.
3. Preserve the six unanswered conditional questions.
4. Construct the reciprocal AHP pairwise matrix.
5. Calculate principal-eigenvector AHP weights.
6. Calculate lambda_max, CI and CR.
7. Export the completed responses, matrix, weights and QA.

No existing provisional suitability results are modified.
The original AHP questionnaire remains unchanged.
"""

from pathlib import Path
from datetime import date
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REPORT = ROOT / "outputs" / "reports"

TEMPLATE = (
    REPORT / "16c_l_ahp_pairwise_template.csv"
)

OUT_RESPONSES = (
    REPORT / "16d_e_expert_ahp_completed_responses.csv"
)

OUT_MATRIX = (
    REPORT / "16d_e_expert_ahp_pairwise_matrix.csv"
)

OUT_WEIGHTS = (
    REPORT / "16d_e_expert_ahp_weights.csv"
)

OUT_QA = (
    REPORT / "16d_e_expert_ahp_consistency_qa.json"
)


# ============================================================
# 2. EXPERT INFORMATION
# ============================================================

EXPERT_ROLE = "Engineer"

CONSULTATION_DATE = "2026-10-17"

CRITERIA = ["C1", "C2", "C4"]

NAMES = {
    "C1": "Major Road Proximity",
    "C2": "Local Road Density",
    "C4": "Terrain Slope",
}


# ============================================================
# 3. LATEST REPORTED EXPERT RESPONSES
# ============================================================

# These are the exact latest judgments reported by
# the researcher.
#
# Do not change ratings or justifications without
# receiving an actual correction from the expert.
#
# EQUAL = Equal importance, strength 1.

ANSWERS = {

    ("C1", "C2"): (
        "C2",
        3,
        (
            "Local road density improves accessibility "
            "to land and supports urban development."
        ),
    ),

    ("C1", "C4"): (
        "EQUAL",
        1,
        (
            "Major road access supports connectivity, "
            "while terrain slope affects construction "
            "feasibility and costs."
        ),
    ),

    ("C2", "C4"): (
        "C2",
        3,
        (
            "Local road density improves access, "
            "while slope affects construction feasibility. "
            "Flood risk also depends on drainage "
            "and elevation."
        ),
    ),
}


RESPONSE_FIELDS = [
    "higher_importance",
    "importance_strength_1_to_9",
    "justification",
    "reviewer_name",
    "reviewer_role",
    "review_date",
]

REQUIRED_FIELDS = [
    "scenario",
    "criterion_a",
    "criterion_b",
    "question",
    *RESPONSE_FIELDS,
]


# ============================================================
# 4. HELPER FUNCTION
# ============================================================

def sha256(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


# ============================================================
# 5. MAIN WORKFLOW
# ============================================================

def main():

    print("=" * 78)

    print(
        "SECTION 03 | EXPERT AHP RESPONSE "
        "RECORD AND CONSISTENCY"
    )

    print("=" * 78)

    # --------------------------------------------------------
    # CHECK INPUT AND OUTPUT FILES
    # --------------------------------------------------------

    if not TEMPLATE.is_file():

        raise FileNotFoundError(TEMPLATE)

    for path in [
        OUT_RESPONSES,
        OUT_MATRIX,
        OUT_WEIGHTS,
        OUT_QA,
    ]:

        if path.exists():

            raise FileExistsError(
                "Output already exists, not overwriting: "
                f"{path}"
            )

    date.fromisoformat(
        CONSULTATION_DATE
    )

    original_hash = sha256(
        TEMPLATE
    )

    original = pd.read_csv(
        TEMPLATE
    )

    missing = sorted(
        set(REQUIRED_FIELDS)
        - set(original.columns)
    )

    if missing:

        raise ValueError(
            f"Questionnaire missing columns: {missing}"
        )

    # ========================================================
    # STEP 1. VERIFY ORIGINAL QUESTIONNAIRE
    # ========================================================

    print("\n[1] TEMPLATE VERIFIED")

    base = original.scenario.eq(
        "BASE_3_CRITERIA"
    )

    conditional = original.scenario.eq(
        "CONDITIONAL_4_WITH_C3"
    )

    actual_pairs = list(
        zip(
            original.loc[
                base, "criterion_a"
            ],
            original.loc[
                base, "criterion_b"
            ],
        )
    )

    if (
        len(original) != 9
        or base.sum() != 3
        or conditional.sum() != 6
        or len(set(actual_pairs)) != 3
        or set(actual_pairs) != set(ANSWERS)
    ):

        raise ValueError(
            "Unexpected template scenarios "
            "or criterion pairs"
        )

    print(
        "Questionnaire rows: 9 | "
        "Baseline: 3 | Conditional: 6"
    )

    # ========================================================
    # STEP 2. RECORD EXPERT ANSWERS
    # ========================================================

    print(
        "\n[2] RECORD LATEST EXPERT RESPONSES"
    )

    completed = original.copy(
        deep=True
    )

    # Cast answer columns to object so that
    # text and numeric ratings can be assigned.
    #
    # This may change pandas column dtypes.
    # We handle that correctly in the comparison
    # of conditional questionnaire rows.

    for field in RESPONSE_FIELDS:

        completed[field] = (
            completed[field].astype(object)
        )

    for row_index, row in original.loc[
        base
    ].iterrows():

        a = str(row.criterion_a)
        b = str(row.criterion_b)

        higher, strength, explanation = (
            ANSWERS[(a, b)]
        )

        if (
            not isinstance(strength, int)
            or not 1 <= strength <= 9
        ):

            raise ValueError(
                f"Invalid strength: {a} vs {b}"
            )

        if (
            higher == "EQUAL"
            and strength != 1
        ):

            raise ValueError(
                "Equal importance requires strength 1"
            )

        if higher not in (
            a,
            b,
            "EQUAL",
        ):

            raise ValueError(
                f"Invalid higher importance: {higher}"
            )

        completed.at[
            row_index,
            "higher_importance",
        ] = higher

        completed.at[
            row_index,
            "importance_strength_1_to_9",
        ] = strength

        completed.at[
            row_index,
            "justification",
        ] = explanation

        # Expert name was not provided.
        # Never invent expert identity.

        completed.at[
            row_index,
            "reviewer_name",
        ] = pd.NA

        completed.at[
            row_index,
            "reviewer_role",
        ] = EXPERT_ROLE

        completed.at[
            row_index,
            "review_date",
        ] = CONSULTATION_DATE

        print(
            f"{a} vs {b}: "
            f"{higher}, strength {strength}"
        )

        print(
            f"  Reason: {explanation}"
        )

    # ========================================================
    # FIXED: CHECK CONDITIONAL QUESTIONS
    # ========================================================

    # Previous code used DataFrame.equals().
    #
    # The comparison could fail because pandas
    # also compares column dtypes.
    #
    # Instead, compare values after consistently
    # converting both sides to string.
    #
    # Missing values are represented by a
    # common placeholder during comparison.

    for field in original.columns:

        before = (
            original.loc[
                conditional,
                field,
            ]
            .astype("string")
            .fillna("<MISSING>")
        )

        after = (
            completed.loc[
                conditional,
                field,
            ]
            .astype("string")
            .fillna("<MISSING>")
        )

        if not before.equals(after):

            raise ValueError(
                "Conditional questionnaire changed: "
                f"{field}"
            )

    if not completed.loc[
        base,
        "justification",
    ].notna().all():

        raise ValueError(
            "Incomplete baseline justifications"
        )

    print(
        "[PASS] All six conditional questions unchanged"
    )

    # ========================================================
    # STEP 3. BUILD AHP PAIRWISE COMPARISON MATRIX
    # ========================================================

    print(
        "\n[3] BUILD AHP PAIRWISE MATRIX"
    )

    positions = {
        criterion: i
        for i, criterion in enumerate(CRITERIA)
    }

    matrix = np.eye(
        3,
        dtype=float,
    )

    for (
        a,
        b,
    ), (
        higher,
        strength,
        _,
    ) in ANSWERS.items():

        if higher == "EQUAL":

            ab_value = 1.0

        elif higher == a:

            ab_value = float(strength)

        else:

            ab_value = 1.0 / strength

        i = positions[a]
        j = positions[b]

        matrix[i, j] = ab_value

        matrix[j, i] = (
            1.0 / ab_value
        )

    # Reciprocal matrix validation.

    if not np.allclose(
        matrix * matrix.T,
        1.0,
        atol=1e-12,
        rtol=0,
    ):

        raise ValueError(
            "AHP matrix is not reciprocal"
        )

    matrix_df = pd.DataFrame(
        matrix,
        index=CRITERIA,
        columns=CRITERIA,
    )

    matrix_df.index.name = (
        "criterion"
    )

    print(
        matrix_df.to_string()
    )

    print(
        "[PASS] Pairwise matrix is reciprocal"
    )

    # ========================================================
    # STEP 4. CALCULATE AHP WEIGHTS
    # ========================================================

    print(
        "\n[4] CALCULATE PRINCIPAL "
        "EIGENVECTOR AND CONSISTENCY"
    )

    eigenvalues, eigenvectors = (
        np.linalg.eig(matrix)
    )

    principal = int(
        np.argmax(
            eigenvalues.real
        )
    )

    lambda_max = float(
        eigenvalues[principal].real
    )

    raw_weights = (
        eigenvectors[
            :,
            principal,
        ].real
    )

    if raw_weights.sum() < 0:

        raw_weights = -raw_weights

    if not np.all(
        raw_weights > 0
    ):

        raise ValueError(
            "Principal eigenvector is not positive"
        )

    weights = (
        raw_weights / raw_weights.sum()
    )

    # ========================================================
    # STEP 5. CALCULATE CONSISTENCY RATIO
    # ========================================================

    n = len(CRITERIA)

    ci = max(
        0.0,
        (lambda_max - n) / (n - 1),
    )

    # Saaty Random Index for n = 3.

    ri = 0.58

    cr = ci / ri

    expected_weights = np.array([
        0.20,
        0.60,
        0.20,
    ])

    if not np.allclose(
        weights,
        expected_weights,
        atol=1e-10,
        rtol=0,
    ):

        raise ValueError(
            "Unexpected weights; "
            "verify pairwise judgments"
        )

    if cr >= 0.10:

        raise ValueError(
            "Consistency Ratio >= 0.10: "
            "expert review required"
        )

    weight_df = pd.DataFrame({
        "criterion": CRITERIA,

        "criterion_name": [
            NAMES[c]
            for c in CRITERIA
        ],

        "expert_ahp_weight": weights,
    })

    print(
        weight_df.to_string(index=False)
    )

    print(
        f"Lambda max: {lambda_max:.6f} | "
        f"CI: {ci:.6f} | "
        f"CR: {cr:.6f}"
    )

    print(
        "[PASS] AHP weights and consistency verified"
    )

    # ========================================================
    # STEP 6. VERIFY ORIGINAL TEMPLATE
    # ========================================================

    if sha256(TEMPLATE) != original_hash:

        raise RuntimeError(
            "Original questionnaire changed "
            "before export"
        )

    # ========================================================
    # STEP 7. EXPORT COMPLETED AHP RESPONSES
    # ========================================================

    print("\n[5] EXPORT REPORTS")

    completed.to_csv(
        OUT_RESPONSES,
        index=False,
    )

    matrix_df.to_csv(
        OUT_MATRIX
    )

    weight_df.to_csv(
        OUT_WEIGHTS,
        index=False,
    )

    # ========================================================
    # STEP 8. CREATE AHP QA / METHODOLOGY REPORT
    # ========================================================

    qa = {

        "section":
            "03_development_suitability",

        "scenario":
            "BASE_3_CRITERIA",

        "status": (
            "PASS_USER_REPORTED_EXPERT_AHP_"
            "NUMERICAL_QA_FINAL_SUITABILITY_PENDING"
        ),

        "consultation_date":
            CONSULTATION_DATE,

        "expert_role":
            EXPERT_ROLE,

        "expert_name":
            None,

        "response_provenance": (
            "Expert judgments communicated to "
            "researcher; researcher-transcribed"
        ),

        "independently_signed_questionnaire":
            False,

        "judgments": [

            {
                "criterion_a": a,

                "criterion_b": b,

                "higher_importance": h,

                "strength": s,

                "exact_justification": j,
            }

            for (
                a,
                b,
            ), (
                h,
                s,
                j,
            ) in ANSWERS.items()

        ],

        "matrix_criterion_order":
            CRITERIA,

        "pairwise_matrix":
            matrix.tolist(),

        "weights": {

            c: float(weights[i])

            for i, c in enumerate(CRITERIA)

        },

        "lambda_max":
            lambda_max,

        "consistency_index":
            ci,

        "random_index_n3":
            ri,

        "consistency_ratio":
            cr,

        "consistency_acceptable":
            bool(cr < 0.10),

        "method":
            "AHP_PRINCIPAL_EIGENVECTOR",

        "original_template_sha256":
            original_hash,

        "original_template_unchanged":
            sha256(TEMPLATE) == original_hash,

        "conditional_four_criterion_model_completed":
            False,

        "final_weighted_suitability_calculated":
            False,

        "final_suitability_sensitivity_completed":
            False,

        "outputs": {

            "completed_responses": (
                OUT_RESPONSES
                .relative_to(ROOT)
                .as_posix()
            ),

            "pairwise_matrix": (
                OUT_MATRIX
                .relative_to(ROOT)
                .as_posix()
            ),

            "criterion_weights": (
                OUT_WEIGHTS
                .relative_to(ROOT)
                .as_posix()
            ),

        },

    }

    OUT_QA.write_text(
        json.dumps(
            qa,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # STEP 9. FINAL TERMINAL SUMMARY
    # ========================================================

    print(
        "Completed responses:",
        OUT_RESPONSES,
    )

    print(
        "Pairwise matrix:",
        OUT_MATRIX,
    )

    print(
        "AHP weights:",
        OUT_WEIGHTS,
    )

    print(
        "QA report:",
        OUT_QA,
    )

    print("\n" + "=" * 78)

    print(
        "PASS_USER_REPORTED_EXPERT_AHP_"
        "NUMERICAL_QA_FINAL_SUITABILITY_PENDING"
    )

    print(
        "Original template untouched: YES"
    )

    print(
        "Final Development Suitability recalculated: NO"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
