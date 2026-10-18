
"""
J26-DS-338 | STEP 16D-T

Isolated tests of the actual AHP intake implementation.

All comparison values are SYNTHETIC MATHEMATICAL TESTS.
They are NOT real expert judgments and must never
be used as final research weights.

Runs the existing AHP script in temporary test folders.
"""

from contextlib import redirect_stdout
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import hashlib
import importlib.util
import io
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / (
    "src/03_development_suitability/"
    "16d_04_ahp_judgment_intake.py"
)

REPORTS = ROOT / "outputs/reports"

OUT_CASES = REPORTS / (
    "16d_t_ahp_engine_unit_test_cases.csv"
)

OUT_SUMMARY = REPORTS / (
    "16d_t_ahp_engine_unit_test_summary.json"
)

LIVE_REVIEW = REPORTS / (
    "16c_l_expert_criteria_review_template.csv"
)

LIVE_PAIRS = REPORTS / (
    "16c_l_ahp_pairwise_template.csv"
)

SCENARIOS = {
    "BASE_3_CRITERIA": [
        "C1", "C2", "C4"
    ],
    "CONDITIONAL_4_WITH_C3": [
        "C1", "C2", "C3", "C4"
    ],
}


def sha(path):

    h = hashlib.sha256()

    with path.open("rb") as stream:
        for part in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(part)

    return h.hexdigest()


def write_json(path, value):

    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def make_fixture(folder, case):

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    blank = case == "blank_forms"

    four = (
        case == "consistent_four_criteria"
    )

    reviewer = (
        "SYNTHETIC_TEST_ONLY_NOT_AN_EXPERT"
    )

    role = (
        "AUTOMATED_MATHEMATICAL_TEST_FIXTURE"
    )

    day = "2099-01-01"

    included = (
        {"C1", "C2", "C3", "C4"}
        if four
        else {"C1", "C2", "C4"}
    )

    directions = {
        "C1": "DECREASING",
        "C2": "INCREASING",
        "C3": "INCREASING",
        "C4": "DECREASING",
        "C5": "NOT_APPLICABLE",
    }

    reviews = []

    for code in [
        "C1", "C2", "C3", "C4", "C5"
    ]:

        reviews.append({
            "criterion": code,

            "expert_decision": (
                ""
                if blank
                else (
                    "INCLUDE"
                    if code in included
                    else "EXCLUDE"
                )
            ),

            "expert_direction": (
                ""
                if blank
                else directions[code]
            ),

            "expert_reason": (
                ""
                if blank
                else "Synthetic test only"
            ),

            "reviewer_name": (
                "" if blank else reviewer
            ),

            "reviewer_role": (
                "" if blank else role
            ),

            "review_date": (
                "" if blank else day
            ),
        })

    pd.DataFrame(reviews).to_csv(
        folder / "review.csv",
        index=False,
    )

    # Perfectly consistent test comparisons:
    # Three-criterion priority ratios: 4:2:1
    # Four-criterion priority ratios: 8:4:2:1

    three_choices = {
        ("C1", "C2"): ("A", 2),
        ("C1", "C4"): ("A", 4),
        ("C2", "C4"): ("A", 2),
    }

    four_choices = {
        ("C1", "C2"): ("A", 2),
        ("C1", "C3"): ("A", 4),
        ("C1", "C4"): ("A", 8),
        ("C2", "C3"): ("A", 2),
        ("C2", "C4"): ("A", 4),
        ("C3", "C4"): ("A", 2),
    }

    if case == "inconsistent_three_criteria":

        three_choices = {
            ("C1", "C2"): ("A", 9),
            ("C1", "C4"): ("B", 9),
            ("C2", "C4"): ("A", 9),
        }

    if case == "invalid_scale":

        three_choices[
            ("C1", "C2")
        ] = ("A", 10)

    active = (
        "CONDITIONAL_4_WITH_C3"
        if four
        else "BASE_3_CRITERIA"
    )

    pair_rows = []

    for scenario, codes in SCENARIOS.items():

        for a, b in combinations(codes, 2):

            fill = (
                not blank
                and scenario == active
            )

            if fill:
                choices = (
                    four_choices
                    if four
                    else three_choices
                )

                side, strength = choices[(a, b)]

            else:
                side, strength = "", ""

            pair_rows.append({
                "scenario": scenario,
                "criterion_a": a,
                "criterion_b": b,

                "higher_importance": side,

                "importance_strength_1_to_9":
                    strength,

                "justification": (
                    "Synthetic mathematical test"
                    if fill
                    else ""
                ),

                "reviewer_name": (
                    reviewer if fill else ""
                ),

                "reviewer_role": (
                    role if fill else ""
                ),

                "review_date": (
                    day if fill else ""
                ),
            })

    pd.DataFrame(pair_rows).to_csv(
        folder / "pairs.csv",
        index=False,
    )

    # Synthetic prerequisite metadata,
    # also isolated from real research.

    write_json(
        folder / "prep.json",
        {
            "qa_status": "PASS",
            "expert_judgments_collected_by_script":
                False,
        },
    )

    write_json(
        folder / "base.json",
        {
            "qa_status": "PASS",
            "methodology_status":
                "PROVISIONAL_REFERENCE_BASELINE_ONLY",
        },
    )

    write_json(
        folder / "spatial.json",
        {"qa_status": "PASS"},
    )


def run_fixture(case, temporary_root):

    case_dir = temporary_root / case

    make_fixture(case_dir, case)

    # Load the REAL production AHP calculation code.

    spec = importlib.util.spec_from_file_location(
        "ahp_engine_" + case,
        SRC,
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            "Could not load AHP implementation."
        )

    engine = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(engine)

    # Redirect all production engine file paths
    # to the temporary test case folder.

    replacements = {
        "ROOT": case_dir,
        "REPORTS": case_dir,

        "REVIEW": case_dir / "review.csv",
        "PAIRWISE": case_dir / "pairs.csv",

        "PREP_SUMMARY": case_dir / "prep.json",
        "BASE_SUMMARY": case_dir / "base.json",
        "SPATIAL_SUMMARY":
            case_dir / "spatial.json",

        "OUT_READINESS":
            case_dir / "readiness.csv",

        "OUT_QA": case_dir / "qa.csv",

        "OUT_WEIGHTS":
            case_dir / "weights.csv",

        "OUT_SUMMARY":
            case_dir / "result.json",
    }

    for name, value in replacements.items():
        setattr(engine, name, value)

    # Execute the actual engine.
    # Capture its verbose terminal output.

    capture = io.StringIO()

    with redirect_stdout(capture):
        engine.main()

    result = json.loads(
        (case_dir / "result.json").read_text(
            encoding="utf-8"
        )
    )

    qa = pd.read_csv(
        case_dir / "qa.csv"
    )

    if result.get(
        "technical_qa_status"
    ) != "PASS":

        raise AssertionError(
            "Production engine technical QA failed."
        )

    if not qa["passed"].astype(str).str.lower().eq(
        "true"
    ).all():

        raise AssertionError(
            "A production engine QA check failed."
        )

    status = result.get(
        "expert_input_status"
    )

    weights_file = (
        case_dir / "weights.csv"
    )

    # ========================================================
    # TEST CASE 1 - BLANK QUESTIONNAIRES
    # ========================================================

    if case == "blank_forms":

        assert status == "PENDING_EXPERT_REVIEW"

        assert result.get("ahp") is None

        assert not weights_file.exists()

    # ========================================================
    # TEST CASE 2/3 - CONSISTENT AHP MATRICES
    # ========================================================

    elif case in {
        "consistent_three_criteria",
        "consistent_four_criteria",
    }:

        assert status == "EXPERT_AHP_VALIDATED"

        assert weights_file.is_file()

        ahp = result["ahp"]

        assert abs(
            ahp["consistency_ratio"]
        ) < 1e-9

        if case == "consistent_three_criteria":

            expected_codes = [
                "C1", "C2", "C4"
            ]

            expected = (
                np.array([4, 2, 1], dtype=float)
                / 7
            )

        else:

            expected_codes = [
                "C1", "C2", "C3", "C4"
            ]

            expected = (
                np.array([8, 4, 2, 1], dtype=float)
                / 15
            )

        assert ahp["criteria"] == expected_codes

        observed = np.array([
            ahp["weights"][code]
            for code in expected_codes
        ])

        assert np.allclose(
            observed,
            expected,
            atol=1e-8,
        )

        saved = pd.read_csv(
            weights_file
        ).set_index("criterion")

        assert np.allclose(
            saved.loc[
                expected_codes,
                "ahp_weight",
            ].to_numpy(dtype=float),
            expected,
            atol=1e-8,
        )

    # ========================================================
    # TEST CASE 4 - INCONSISTENT JUDGMENTS
    # ========================================================

    elif case == "inconsistent_three_criteria":

        assert (
            status
            == "CONSISTENCY_REVIEW_REQUIRED"
        )

        assert (
            result["ahp"]["consistency_ratio"]
            > 0.10
        )

        assert not weights_file.exists()

    # ========================================================
    # TEST CASE 5 - INVALID SAATY SCALE
    # ========================================================

    elif case == "invalid_scale":

        assert (
            status
            == "JUDGMENT_REVIEW_REQUIRED"
        )

        assert not weights_file.exists()

    else:

        raise RuntimeError(
            "Unknown test fixture."
        )

    return {
        "test_case": case,
        "result": "PASS",
        "engine_status": status,
        "consistency_ratio": (
            result.get("ahp", {})
            .get("consistency_ratio")
            if result.get("ahp")
            else None
        ),
        "engine_qa_checks": len(qa),
        "weights_generated_in_temp_only":
            weights_file.is_file(),
    }


def main():

    print("=" * 82)

    print(
        "STEP 16D-T - AHP CALCULATION ENGINE UNIT TESTS"
    )

    print("=" * 82)

    print(
        "ALL COMPARISON VALUES ARE SYNTHETIC TEST DATA."
    )

    if not SRC.is_file():
        raise FileNotFoundError(SRC)

    for path in [
        LIVE_REVIEW,
        LIVE_PAIRS,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Confirm actual forms remain unchanged.

    original_hashes = {
        str(path): sha(path)
        for path in [
            LIVE_REVIEW,
            LIVE_PAIRS,
        ]
    }

    cases = [
        "blank_forms",
        "consistent_three_criteria",
        "consistent_four_criteria",
        "inconsistent_three_criteria",
        "invalid_scale",
    ]

    results = []

    # All synthetic questionnaires, weights and
    # engine outputs live inside this directory.
    # Python deletes it after tests finish.

    with TemporaryDirectory(
        prefix="j26_ahp_unit_test_"
    ) as temporary:

        temp_root = Path(temporary)

        for case in cases:

            try:

                record = run_fixture(
                    case,
                    temp_root,
                )

            except Exception as exc:

                record = {
                    "test_case": case,
                    "result": "FAIL",
                    "error": repr(exc),
                }

            results.append(record)

            print(
                f"{record['result']}: {case}"
            )

            if "error" in record:
                print(
                    "  Reason:",
                    record["error"],
                )

    unchanged = all(
        sha(path)
        == original_hashes[str(path)]

        for path in [
            LIVE_REVIEW,
            LIVE_PAIRS,
        ]
    )

    results.append({
        "test_case":
            "real_expert_forms_unchanged",

        "result":
            "PASS" if unchanged else "FAIL",
    })

    print(
        f"{'PASS' if unchanged else 'FAIL'}: "
        "real_expert_forms_unchanged"
    )

    table = pd.DataFrame(results)

    table.to_csv(
        OUT_CASES,
        index=False,
    )

    passed = int(
        table["result"].eq("PASS").sum()
    )

    overall = (
        "PASS"
        if passed == len(table)
        else "FAIL"
    )

    summary = {
        "step": "16D-T",
        "status": overall,

        "purpose":
            "Isolated tests of the real AHP intake code",

        "tests_passed": passed,
        "tests_total": len(table),

        "production_engine_sha256":
            sha(SRC),

        "real_questionnaires_modified":
            not unchanged,

        "all_test_judgments_synthetic":
            True,

        "real_ahp_weights_created":
            False,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "outputs": [
            OUT_CASES.name,
            OUT_SUMMARY.name,
        ],
    }

    write_json(
        OUT_SUMMARY,
        summary,
    )

    print("\n=== TEST SUMMARY ===")

    print(
        f"Overall QA: {overall} "
        f"({passed}/{len(table)})"
    )

    print("\n=== GENERATED REPORTS ===")

    print(
        OUT_CASES.relative_to(ROOT)
    )

    print(
        OUT_SUMMARY.relative_to(ROOT)
    )

    print(
        "\nNo actual expert judgments "
        "or final AHP weights were created."
    )

    if overall != "PASS":

        raise SystemExit(
            "AHP unit tests require review."
        )

    print("=" * 82)


if __name__ == "__main__":
    main()
