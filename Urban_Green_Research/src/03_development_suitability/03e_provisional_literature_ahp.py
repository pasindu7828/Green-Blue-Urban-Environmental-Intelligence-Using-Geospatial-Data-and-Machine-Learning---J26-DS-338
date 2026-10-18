
"""
J26-DS-338 | Development Suitability
Stages 3.6-3.7: Provisional Literature-Informed AHP

Purpose:
- Build an explicitly documented 4x4 AHP comparison matrix.
- Distinguish published comparisons from researcher assumptions.
- Calculate principal-eigenvector weights, lambda max, CI and CR.
- Validate mathematical consistency.
- Save a compact, reproducible evidence package.

IMPORTANT:
This is a PROVISIONAL cross-study scenario.
It is NOT a Kaduwela expert judgment or approved final weighting.
No GIS suitability score is calculated.
"""

from pathlib import Path
import csv
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "outputs/reports/development_suitability"

STANDARDIZATION_QA = REPORT / "03d_standardization_qa.json"

KEL = "https://doi.org/10.3390/land13081288"
SUZHOU = "https://doi.org/10.3390/land14040836"

CRITERIA = {
    "C1": "Higher-order road accessibility",
    "C2": "Local road density (500 m)",
    "C3": "Surrounding building context (500 m)",
    "C4": "Mean terrain slope",
}

# A/B = relative importance of criterion A over criterion B.
#
# Adapted published values:
# Kelowna 2024, Table 3
# Suzhou 2025, Appendix Table A8
#
# The two unsourced comparisons below are explicitly marked
# as assumptions. Never report them as published expert values.

PAIRS = [
    {
        "a": "C1", "b": "C2", "value": 2.0,
        "reported_value": "2",
        "evidence": "SCENARIO_ASSUMPTION",
        "source": "",
        "source_table": "",
        "original_comparison": "Not independently verified",
        "limitation": "Requires independent expert review",
    },
    {
        "a": "C1", "b": "C3", "value": 3.0,
        "reported_value": "3",
        "evidence": "PUBLISHED_VALUE_ADAPTED",
        "source": KEL,
        "source_table": "Table 3",
        "original_comparison":
            "Proximity to Roads / Proximity to Residential",
        "limitation":
            "Residential proximity differs from building footprint context",
    },
    {
        "a": "C1", "b": "C4", "value": 8.0,
        "reported_value": "8",
        "evidence": "PUBLISHED_VALUE_ADAPTED",
        "source": KEL,
        "source_table": "Table 3",
        "original_comparison": "Proximity to Roads / Slope",
        "limitation":
            "Transferred from Kelowna urban-suitability context",
    },
    {
        "a": "C2", "b": "C3", "value": 1.0 / 3.0,
        "reported_value": "1/3",
        "evidence": "PUBLISHED_VALUE_ADAPTED",
        "source": SUZHOU,
        "source_table": "Appendix Table A8",
        "original_comparison":
            "Road Network Density / Building Density",
        "limitation":
            "Original study evaluates sponge-city construction",
    },
    {
        "a": "C2", "b": "C4", "value": 4.0,
        "reported_value": "4",
        "evidence": "SCENARIO_ASSUMPTION",
        "source": "",
        "source_table": "",
        "original_comparison": "Not independently verified",
        "limitation": "Requires independent expert review",
    },
    {
        "a": "C3", "b": "C4", "value": 7.0,
        "reported_value": "7",
        "evidence": "PUBLISHED_VALUE_ADAPTED",
        "source": KEL,
        "source_table": "Table 3",
        "original_comparison":
            "Proximity to Residential / Slope",
        "limitation":
            "Residential proximity differs from building footprint context",
    },
]


def calculate_ahp(matrix):
    """Principal right-eigenvector AHP calculation."""

    n = matrix.shape[0]

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    index = int(np.argmax(eigenvalues.real))

    eigenvalue = eigenvalues[index]
    eigenvector = eigenvectors[:, index]

    if abs(eigenvalue.imag) > 1e-9:
        raise ValueError("Unexpected complex principal eigenvalue")

    if np.max(np.abs(eigenvector.imag)) > 1e-9:
        raise ValueError("Unexpected complex priority vector")

    weights = eigenvector.real

    if weights.sum() < 0:
        weights = -weights

    if np.any(weights <= 0):
        raise ValueError("Invalid non-positive AHP weights")

    weights = weights / weights.sum()

    lambda_max = float(eigenvalue.real)
    ci = (lambda_max - n) / (n - 1)

    # Saaty's commonly used Random Index for n = 4.
    ri = 0.90
    cr = ci / ri

    residual = np.max(
        np.abs(matrix @ weights - lambda_max * weights)
    )

    return {
        "weights": weights,
        "lambda_max": lambda_max,
        "CI": float(ci),
        "RI": ri,
        "CR": float(cr),
        "eigenvector_residual": float(residual),
    }


def main():

    print("=" * 65)
    print("STAGES 3.6-3.7 | PROVISIONAL LITERATURE AHP")
    print("=" * 65)

    # Check previous standardized-criteria stage.
    if not STANDARDIZATION_QA.is_file():
        raise FileNotFoundError(
            "Stage 3.5 QA report not found"
        )

    previous = json.loads(
        STANDARDIZATION_QA.read_text(encoding="utf-8")
    )

    if previous.get("qa_status") != "PASS":
        raise ValueError(
            "Stage 3.5 has not passed QA"
        )

    names = list(CRITERIA)
    n = len(names)
    expected_pairs = n * (n - 1) // 2

    if len(PAIRS) != expected_pairs:
        raise ValueError("Incorrect number of AHP comparisons")

    matrix = np.eye(n, dtype=float)
    seen = set()

    print("\n[1] PAIRWISE COMPARISONS")

    for item in PAIRS:

        a = item["a"]
        b = item["b"]
        value = float(item["value"])

        if a not in CRITERIA or b not in CRITERIA or a == b:
            raise ValueError("Invalid criterion pair")

        if not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid comparison value")

        pair = frozenset((a, b))

        if pair in seen:
            raise ValueError("Duplicate pairwise comparison")

        seen.add(pair)

        # Standard Saaty 1-9 scale or reciprocal.
        magnitude = max(value, 1.0 / value)
        if (
            magnitude > 9
            or not np.isclose(magnitude, round(magnitude))
        ):
            raise ValueError(
                f"Comparison outside Saaty scale: {a}/{b}"
            )

        i = names.index(a)
        j = names.index(b)

        matrix[i, j] = value
        matrix[j, i] = 1.0 / value

        print(
            f"{a}/{b} = {item['reported_value']:<4} "
            f"[{item['evidence']}]"
        )

    print("\n[2] PAIRWISE MATRIX")

    print("         " + " ".join(f"{x:>9}" for x in names))

    for name, row in zip(names, matrix):
        print(f"{name:<8} " + " ".join(f"{x:9.4f}" for x in row))

    # --------------------------------------------------
    # AHP mathematics
    # --------------------------------------------------

    print("\n[3] AHP WEIGHT CALCULATION")

    result = calculate_ahp(matrix)
    weights = result["weights"]

    for key, weight in zip(names, weights):
        print(f"{key}: {weight:.8f} ({weight * 100:.4f}%)")

    print(f"\nLambda max: {result['lambda_max']:.8f}")
    print(f"CI: {result['CI']:.8f}")
    print(f"RI: {result['RI']:.2f}")
    print(f"CR: {result['CR']:.8f}")
    print(f"CR percentage: {result['CR'] * 100:.4f}%")

    # --------------------------------------------------
    # QA
    # --------------------------------------------------

    print("\n[4] MATHEMATICAL AND PROVENANCE QA")

    source_count = sum(
        p["evidence"] == "PUBLISHED_VALUE_ADAPTED"
        for p in PAIRS
    )

    assumption_count = sum(
        p["evidence"] == "SCENARIO_ASSUMPTION"
        for p in PAIRS
    )

    checks = {
        "previous_standardization_pass": True,
        "four_criteria": n == 4,
        "six_unique_comparisons": len(seen) == 6,
        "diagonal_equals_one": bool(
            np.allclose(np.diag(matrix), 1.0)
        ),
        "positive_matrix": bool(np.all(matrix > 0)),
        "reciprocal_matrix": bool(
            np.allclose(matrix * matrix.T, 1.0)
        ),
        "weights_positive": bool(np.all(weights > 0)),
        "weights_sum_one": bool(
            np.isclose(weights.sum(), 1.0)
        ),
        "eigenvector_residual_small": (
            result["eigenvector_residual"] < 1e-8
        ),
        "lambda_at_least_n": (
            result["lambda_max"] >= n - 1e-10
        ),
        "consistency_ratio_below_0_10": (
            0 <= result["CR"] < 0.10
        ),
        "four_adapted_published_values_identified": (
            source_count == 4
        ),
        "two_assumptions_explicitly_identified": (
            assumption_count == 2
        ),
    }

    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    numerical_pass = all(checks.values())

    status = (
        "NUMERICALLY_CONSISTENT_PROVISIONAL_ONLY"
        if numerical_pass
        else "REVIEW_REQUIRED"
    )

    # --------------------------------------------------
    # Save only two compact reports
    # --------------------------------------------------

    REPORT.mkdir(parents=True, exist_ok=True)

    pairs_path = (
        REPORT / "03e_provisional_ahp_pairwise.csv"
    )

    audit_path = (
        REPORT / "03e_provisional_ahp_audit.json"
    )

    with pairs_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(PAIRS[0])
        )
        writer.writeheader()
        writer.writerows(PAIRS)

    audit = {
        "stages": ["3.6", "3.7"],
        "status": status,
        "judgment_origin": (
            "Cross-study adapted literature values plus "
            "explicit researcher scenario assumptions"
        ),
        "is_new_expert_assessment": False,
        "is_approved_final_ahp": False,
        "criteria_order": names,
        "criteria_names": CRITERIA,
        "pairwise_matrix": matrix.tolist(),
        "pairwise_judgments": PAIRS,
        "principal_eigenvector_weights": {
            key: float(value)
            for key, value in zip(names, weights)
        },
        "lambda_max": result["lambda_max"],
        "consistency_index": result["CI"],
        "random_index": result["RI"],
        "consistency_ratio": result["CR"],
        "consistency_threshold": 0.10,
        "checks": checks,
        "published_adapted_comparisons": source_count,
        "unsourced_scenario_comparisons": assumption_count,
        "source_references": [
            {
                "study": "Kelowna urban suitability (2024)",
                "doi": KEL,
                "table": "Table 3",
            },
            {
                "study": "Suzhou sponge-city construction (2025)",
                "doi": SUZHOU,
                "table": "Appendix Table A8",
            },
        ],
        "limitations": [
            "The scenario is not Kaduwela-specific expert AHP.",
            "Two comparisons have no verified published source.",
            "Published criteria and objectives are not identical.",
            "Low CR does not prove external validity.",
            "Supervisor approval or genuine expert assessment "
            "is still needed for final methodology compliance.",
            "No suitability score was calculated in this stage.",
        ],
    }

    audit_path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print("\n[5] GENERATED EVIDENCE")
    print(pairs_path)
    print(audit_path)

    print("\n" + "=" * 65)
    print("AHP STATUS:", status)
    print("Expert assessment: NOT COMPLETED")
    print("Final approved weights: NOT ESTABLISHED")
    print("Suitability scoring: NOT PERFORMED")
    print("=" * 65)

    if not numerical_pass:
        raise SystemExit(
            "Provisional matrix failed QA; review required."
        )


if __name__ == "__main__":
    main()
