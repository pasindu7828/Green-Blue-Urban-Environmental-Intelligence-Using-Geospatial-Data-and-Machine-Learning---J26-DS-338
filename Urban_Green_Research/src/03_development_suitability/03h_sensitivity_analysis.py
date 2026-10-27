
"""
J26-DS-338 | Section 03 | Stage 3.10

Provisional Development Suitability Sensitivity Analysis

Tests:
1. AHP weight perturbation: +/-20% per criterion
2. Alternative C2/C3 neighbourhood radii
3. Alternative percentile standardization anchors
4. Two assumed AHP pairwise judgments
5. Equal-interval versus quantile classification

Outputs:
- 03h_sensitivity_scenarios.csv
- 03h_sensitivity_qa.json

No source data, existing scores or maps are modified.
Not an independent accuracy assessment.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed/development_suitability"
REPORT = ROOT / "outputs/reports/development_suitability"

RAW_FILE = DATA / "raw_criteria/03b_raw_criteria_100m.csv"

STD_FILE = (
    DATA / "standardized_criteria/"
    "03d_standardized_criteria_100m.csv"
)

WEIGHTED_FILE = (
    DATA / "weighted_suitability/"
    "03f_provisional_weighted_suitability_100m.csv"
)

CLASS_FILE = (
    DATA / "weighted_suitability/"
    "03g_provisional_suitability_classes_100m.csv"
)

AHP_FILE = REPORT / "03e_provisional_ahp_audit.json"

MCDA_QA_FILE = (
    REPORT / "03f_provisional_weighted_suitability_qa.json"
)

CLASS_QA_FILE = (
    REPORT / "03g_suitability_classification_qa.json"
)

CRITERIA = ["C1", "C2", "C3", "C4"]

STD_COLS = [
    "c1_road_access_score",
    "c2_road_context_score",
    "c3_development_context_score",
    "c4_slope_score",
]

RAW_BASE = [
    "c1_major_road_distance_m",
    "c2_road_density_500m_km_per_km2",
    "c3_context_building_pct_500m",
    "c4_slope_mean_deg",
]

DIRECTION = ["COST", "BENEFIT", "BENEFIT", "COST"]

BOUNDARIES = np.array([0.2, 0.4, 0.6, 0.8])


def scale(values, direction, low=0.05, high=0.95):

    lower, upper = np.quantile(values, [low, high])

    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or upper <= lower
    ):
        raise ValueError("Invalid percentile limits")

    if direction == "BENEFIT":
        scores = (values - lower) / (upper - lower)
    else:
        scores = (upper - values) / (upper - lower)

    return np.clip(scores, 0.0, 1.0)


def ahp_weights(matrix):

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    index = int(np.argmax(eigenvalues.real))

    value = eigenvalues[index]
    vector = eigenvectors[:, index]

    if (
        abs(value.imag) > 1e-9
        or np.max(np.abs(vector.imag)) > 1e-9
    ):
        raise ValueError("Complex AHP eigenpair")

    weights = vector.real

    if weights.sum() < 0:
        weights = -weights

    if not (weights > 0).all():
        raise ValueError("Nonpositive AHP weights")

    weights /= weights.sum()

    ci = (value.real - len(weights)) / (len(weights) - 1)
    cr = ci / 0.90

    return weights, float(cr)


def check_table(path):

    df = pd.read_csv(path)

    if (
        len(df) != 9240
        or df["grid_id"].isna().any()
        or not df["grid_id"].is_unique
    ):
        raise ValueError(f"Unexpected grid IDs: {path}")

    return df.set_index("grid_id")


def main():

    print("=" * 65)
    print("STAGE 3.10 - GIS-MCDA SENSITIVITY ANALYSIS")
    print("=" * 65)

    files = [
        RAW_FILE,
        STD_FILE,
        WEIGHTED_FILE,
        CLASS_FILE,
        AHP_FILE,
        MCDA_QA_FILE,
        CLASS_QA_FILE,
    ]

    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)

    ahp = json.loads(AHP_FILE.read_text(encoding="utf-8"))
    mcda_qa = json.loads(
        MCDA_QA_FILE.read_text(encoding="utf-8")
    )
    class_qa = json.loads(
        CLASS_QA_FILE.read_text(encoding="utf-8")
    )

    if (
        mcda_qa.get("status") != "PASS_PROVISIONAL"
        or class_qa.get("status") != "PASS_PROVISIONAL"
    ):
        raise ValueError("Stages 3.8 and 3.9 must pass")

    if (
        ahp.get("status")
        != "NUMERICALLY_CONSISTENT_PROVISIONAL_ONLY"
        or ahp.get("is_approved_final_ahp") is not False
    ):
        raise ValueError("Unexpected AHP provenance")

    if ahp["criteria_order"] != CRITERIA:
        raise ValueError("Unexpected criterion order")

    weighted = check_table(WEIGHTED_FILE)
    ids = weighted.index

    raw = check_table(RAW_FILE).loc[ids]
    standard = check_table(STD_FILE).loc[ids]
    classified = check_table(CLASS_FILE).loc[ids]

    X = standard[STD_COLS].to_numpy(dtype=float)

    w = np.array(
        [
            ahp["principal_eigenvector_weights"][key]
            for key in CRITERIA
        ],
        dtype=float,
    )

    base = weighted[
        "provisional_suitability_score"
    ].to_numpy(dtype=float)

    areas = weighted[
        "kmc_overlap_area_m2"
    ].to_numpy(dtype=float)

    baseline_classes = np.digitize(
        base, BOUNDARIES
    ) + 1

    matrix = np.array(
        ahp["pairwise_matrix"], dtype=float
    )

    recovered_w, recovered_cr = ahp_weights(matrix)

    recon_error = float(
        np.max(np.abs(X @ w - base))
    )

    # ---------------------------------------------------
    # 1. Baseline QA
    # ---------------------------------------------------

    print("\n[1] BASELINE VALIDATION")

    reproduced_X = np.column_stack([
        scale(
            raw[col].to_numpy(dtype=float),
            direction
        )
        for col, direction in zip(RAW_BASE, DIRECTION)
    ])

    checks = {
        "9240_aligned_grid_ids": all(
            len(d) == 9240
            for d in (raw, standard, weighted, classified)
        ),
        "finite_bounded_criteria": bool(
            np.isfinite(X).all()
            and ((X >= 0) & (X <= 1)).all()
        ),
        "valid_area": bool(
            np.isfinite(areas).all()
            and (areas > 0).all()
        ),
        "ahp_reproduced": bool(
            np.allclose(w, recovered_w, atol=1e-10)
            and np.isclose(
                recovered_cr,
                ahp["consistency_ratio"],
                atol=1e-10,
            )
        ),
        "baseline_score_reproduced": (
            recon_error < 1e-10
        ),
        "standardization_reproduced": bool(
            np.allclose(X, reproduced_X, atol=1e-10)
        ),
        "baseline_classes_reproduced": bool(
            np.array_equal(
                baseline_classes,
                classified[
                    "suitability_class_code"
                ].to_numpy(dtype=int),
            )
        ),
        "study_area_reconciled": (
            abs(
                float(areas.sum())
                - float(class_qa["kmc_area_km2"]) * 1e6
            ) <= 1.0
        ),
    }

    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(
        "Maximum baseline score difference:",
        f"{recon_error:.3e}"
    )

    if not all(checks.values()):
        raise SystemExit("Baseline QA failed")

    # ---------------------------------------------------
    # 2. Scenario comparison helper
    # ---------------------------------------------------

    n = len(base)
    base_high = baseline_classes >= 4

    top_k = int(np.ceil(n * 0.10))

    base_top = set(
        np.argsort(base, kind="stable")[-top_k:].tolist()
    )

    records = []
    inconsistent_ahp_scenarios = []

    def evaluate(
        name,
        category,
        scores,
        *,
        weights=None,
        cr=None,
        classes=None,
        change="",
    ):

        scores = np.asarray(scores, dtype=float)

        scenario_weights = (
            w if weights is None
            else np.asarray(weights, dtype=float)
        )

        if (
            scores.shape != (n,)
            or not np.isfinite(scores).all()
            or (scores < -1e-10).any()
            or (scores > 1 + 1e-10).any()
            or not np.isclose(
                scenario_weights.sum(), 1.0
            )
            or (scenario_weights <= 0).any()
        ):
            raise ValueError(f"Invalid scenario: {name}")

        new_classes = (
            np.digitize(scores, BOUNDARIES) + 1
            if classes is None
            else np.asarray(classes, dtype=int)
        )

        if (
            new_classes.shape != (n,)
            or not np.isin(
                new_classes, [1, 2, 3, 4, 5]
            ).all()
        ):
            raise ValueError(
                f"Invalid class values: {name}"
            )

        new_high = new_classes >= 4

        intersection = float(
            areas[base_high & new_high].sum()
        )

        union = float(
            areas[base_high | new_high].sum()
        )

        high_jaccard = (
            intersection / union if union > 0 else 1.0
        )

        new_top = set(
            np.argsort(
                scores, kind="stable"
            )[-top_k:].tolist()
        )

        correlation = float(
            spearmanr(base, scores).statistic
        )

        changed = new_classes != baseline_classes

        records.append({
            "scenario": name,
            "type": category,
            "changed_parameter": change,
            "spearman_rank_vs_baseline": correlation,
            "mean_absolute_score_change": float(
                np.mean(np.abs(scores - base))
            ),
            "p95_absolute_score_change": float(
                np.quantile(
                    np.abs(scores - base), 0.95
                )
            ),
            "class_changed_cells_pct": float(
                100 * changed.mean()
            ),
            "class_changed_kmc_area_pct": float(
                100 * areas[changed].sum() / areas.sum()
            ),
            "high_very_high_area_pct": float(
                100 * areas[new_high].sum() / areas.sum()
            ),
            "high_area_overlap_jaccard": high_jaccard,
            "top_decile_cell_overlap_pct": (
                100 * len(base_top & new_top) / top_k
            ),
            "ahp_cr": (
                None if cr is None else float(cr)
            ),
            "ahp_cr_below_0_10": (
                None if cr is None
                else bool(cr < 0.10)
            ),
        })

        if cr is not None and cr >= 0.10:
            inconsistent_ahp_scenarios.append(name)

    # ---------------------------------------------------
    # 3. AHP weight perturbation
    # ---------------------------------------------------

    print("\n[2] AHP WEIGHT SENSITIVITY")

    for i, criterion in enumerate(CRITERIA):

        for factor, label in [
            (0.8, "MINUS20"),
            (1.2, "PLUS20"),
        ]:

            alt_w = w.copy()
            alt_w[i] *= factor
            alt_w /= alt_w.sum()

            evaluate(
                f"WEIGHT_{criterion}_{label}",
                "weight_perturbation",
                X @ alt_w,
                weights=alt_w,
                change=(
                    f"{criterion} x{factor:g}; "
                    "weights renormalized"
                ),
            )

    # ---------------------------------------------------
    # 4. Alternative neighbourhood radii
    # ---------------------------------------------------

    print("\n[3] NEIGHBOURHOOD RADIUS SENSITIVITY")

    alternatives = [
        (
            "C2_250M",
            {1: "c2_road_density_250m_km_per_km2"},
        ),
        (
            "C2_1000M",
            {1: "c2_road_density_1000m_km_per_km2"},
        ),
        (
            "C3_250M",
            {2: "c3_context_building_pct_250m"},
        ),
        (
            "C2_C3_250M",
            {
                1: "c2_road_density_250m_km_per_km2",
                2: "c3_context_building_pct_250m",
            },
        ),
    ]

    for name, changes in alternatives:

        XX = X.copy()

        for index, raw_col in changes.items():

            XX[:, index] = scale(
                raw[raw_col].to_numpy(dtype=float),
                DIRECTION[index],
            )

        evaluate(
            name,
            "radius",
            XX @ w,
            change="; ".join(changes.values()),
        )

    # ---------------------------------------------------
    # 5. Standardization sensitivity
    # ---------------------------------------------------

    print("\n[4] STANDARDIZATION SENSITIVITY")

    for low, high, name in [
        (0.10, 0.90, "ANCHORS_P10_P90"),
        (0.01, 0.99, "ANCHORS_P01_P99"),
    ]:

        XX = np.column_stack([
            scale(
                raw[col].to_numpy(dtype=float),
                direction,
                low,
                high,
            )
            for col, direction in zip(
                RAW_BASE, DIRECTION
            )
        ])

        evaluate(
            name,
            "standardization",
            XX @ w,
            change=f"P{low*100:.0f}-P{high*100:.0f}",
        )

    # ---------------------------------------------------
    # 6. Unsourced AHP judgment stress tests
    # ---------------------------------------------------

    print("\n[5] ASSUMED AHP JUDGMENT SENSITIVITY")

    pairwise_tests = [
        (0, 1, 2.0, (1.0, 3.0), "C1_C2"),
        (1, 3, 4.0, (2.0, 6.0), "C2_C4"),
    ]

    for i, j, original, options, label in pairwise_tests:

        if not np.isclose(matrix[i, j], original):
            raise ValueError(
                f"Unexpected original AHP judgment: {label}"
            )

        for alternative in options:

            new_matrix = matrix.copy()

            new_matrix[i, j] = alternative
            new_matrix[j, i] = 1.0 / alternative

            alternative_weights, alternative_cr = (
                ahp_weights(new_matrix)
            )

            evaluate(
                f"PAIR_{label}_{alternative:g}",
                "assumed_pairwise_stress",
                X @ alternative_weights,
                weights=alternative_weights,
                cr=alternative_cr,
                change=(
                    f"{label}: {original:g} "
                    f"to {alternative:g}"
                ),
            )

    # ---------------------------------------------------
    # 7. Classification sensitivity
    # ---------------------------------------------------

    print("\n[6] CLASSIFICATION SENSITIVITY")

    quantile_boundaries = np.quantile(
        base, [0.2, 0.4, 0.6, 0.8]
    )

    quantile_classes = np.digitize(
        base, quantile_boundaries
    ) + 1

    evaluate(
        "CLASS_QUINTILES",
        "classification",
        base,
        classes=quantile_classes,
        change=(
            "Five equal-frequency classes instead "
            "of five equal score intervals"
        ),
    )

    # ---------------------------------------------------
    # 8. Validate all scenarios
    # ---------------------------------------------------

    print("\n[7] SENSITIVITY VALIDATION")

    results = pd.DataFrame(records)

    checks["expected_19_scenarios"] = (
        len(results) == 19
    )

    numeric_columns = [
        "spearman_rank_vs_baseline",
        "mean_absolute_score_change",
        "class_changed_cells_pct",
        "class_changed_kmc_area_pct",
        "high_very_high_area_pct",
        "high_area_overlap_jaccard",
        "top_decile_cell_overlap_pct",
    ]

    checks["all_results_finite"] = bool(
        np.isfinite(
            results[numeric_columns].to_numpy(dtype=float)
        ).all()
    )

    checks["jaccard_within_0_1"] = bool(
        results[
            "high_area_overlap_jaccard"
        ].between(0, 1).all()
    )

    checks["no_final_expert_approval_claim"] = (
        ahp["is_approved_final_ahp"] is False
    )

    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if not all(checks.values()):
        raise SystemExit(
            "Sensitivity QA failed. No reports exported."
        )

    # ---------------------------------------------------
    # 9. Summarize results
    # ---------------------------------------------------

    print("\n[8] SENSITIVITY RESULTS")

    display_columns = [
        "scenario",
        "spearman_rank_vs_baseline",
        "class_changed_kmc_area_pct",
        "high_very_high_area_pct",
        "high_area_overlap_jaccard",
        "ahp_cr",
    ]

    print(
        results[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nPairwise scenarios with CR >= 0.10:")
    print(
        inconsistent_ahp_scenarios
        if inconsistent_ahp_scenarios
        else "None"
    )

    # ---------------------------------------------------
    # 10. Export two reports
    # ---------------------------------------------------

    REPORT.mkdir(parents=True, exist_ok=True)

    csv_out = REPORT / "03h_sensitivity_scenarios.csv"
    qa_out = REPORT / "03h_sensitivity_qa.json"

    results.to_csv(csv_out, index=False)

    qa = {
        "stage": "3.10",
        "status": "PASS_PROVISIONAL",
        "baseline_provenance": (
            "Literature-adapted provisional AHP "
            "with two scenario assumptions"
        ),
        "scenario_count": len(results),
        "baseline_high_very_high_area_pct": float(
            100 * areas[base_high].sum() / areas.sum()
        ),
        "baseline_score_reproduction_max_error": (
            recon_error
        ),
        "checks": checks,
        "inconsistent_alternative_ahp_scenarios": (
            inconsistent_ahp_scenarios
        ),
        "limitations": [
            "Sensitivity is not predictive accuracy.",
            "Sensitivity is not external ground truth.",
            "Equal-frequency classes differ conceptually "
            "from equal-interval classes.",
            "AHP scenarios with CR >= 0.10 are "
            "diagnostic only, not acceptable replacements.",
            "Genuine expert AHP is not yet completed.",
            "Legal and environmental exclusions are "
            "not represented by the suitability score.",
            "Final methodology approval remains pending.",
        ],
    }

    qa_out.write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )

    print("\nGenerated reports:")
    print(csv_out)
    print(qa_out)

    print("\n" + "=" * 65)
    print("STAGE 3.10: PASS_PROVISIONAL")
    print("Sensitivity scenarios:", len(results))
    print("Final expert-approved AHP: NOT COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
