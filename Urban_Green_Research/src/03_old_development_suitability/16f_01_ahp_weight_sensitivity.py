
"""
J26-DS-338
STEP 16F-01: AHP WEIGHT SENSITIVITY ANALYSIS

Approved methodology: GIS-MCDA with genuine expert AHP.

Current status:
    Expert responses pending.
    Do not calculate final sensitivity or invent weights.

When genuine AHP results become available:
    - Use final Step 16E AHP-weighted scores.
    - Perturb approved AHP weights by +/-10% and +/-20%.
    - Renormalize weight vectors.
    - Recalculate suitability scores.
    - Compare rank, top-decile and relative-class stability.
    - Preserve zoning and conservation review information.
    - Export reproducible CSV, GIS, QA and JSON reports.

This is internal robustness testing, not external validation.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "outputs" / "reports"

MODEL_STATUS = REPORTS / (
    "16e_01_ahp_weighted_status.json"
)

AHP_STATUS = REPORTS / (
    "16d_d_ahp_consistency_summary.json"
)

MODEL_CSV = REPORTS / (
    "16e_01_ahp_weighted_grid_100m.csv"
)

MODEL_GPKG = REPORTS / (
    "16e_01_ahp_weighted_grid_100m.gpkg"
)

OUT_QA = REPORTS / (
    "16f_01_ahp_sensitivity_qa.csv"
)

OUT_STATUS = REPORTS / (
    "16f_01_ahp_sensitivity_status.json"
)

OUT_SCENARIOS = REPORTS / (
    "16f_01_ahp_sensitivity_scenarios.csv"
)

OUT_GRID = REPORTS / (
    "16f_01_ahp_sensitivity_grid_100m.csv"
)

OUT_GPKG = REPORTS / (
    "16f_01_ahp_sensitivity_grid_100m.gpkg"
)

OUT_TRANSITIONS = REPORTS / (
    "16f_01_ahp_class_transitions.csv"
)

N_GRIDS = 9240
EPSG = 32644
CR_LIMIT = 0.10

# Relative changes to one AHP weight before normalization.
PERTURBATIONS = [-0.20, -0.10, 0.10, 0.20]

# Descriptive stability threshold, not probability.
STABILITY_THRESHOLD = 0.75

CLASS_NAMES = [
    "RELATIVE_VERY_LOW",
    "RELATIVE_LOW",
    "RELATIVE_MODERATE",
    "RELATIVE_HIGH",
    "RELATIVE_VERY_HIGH",
]


# ============================================================
# 2. HELPERS
# ============================================================

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


def file_hash(path):

    h = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def parse_boolean(series):

    value = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    if not value.isin(
        ["true", "false", "1", "0"]
    ).all():
        raise ValueError(
            f"Invalid boolean column: {series.name}"
        )

    return value.isin(
        ["true", "1"]
    ).to_numpy(dtype=bool)


def relative_classes(scores, ids):
    """
    Reproduce Step 16E deterministic quintiles.
    Tie-break by stable grid ID.
    """

    order = np.lexsort(
        (ids, scores)
    )

    classes = np.empty(
        len(scores),
        dtype=int,
    )

    classes[order] = (
        np.arange(len(scores))
        * 5
        // len(scores)
    )

    return np.array(
        CLASS_NAMES
    )[classes]


def top_decile(scores, ids):
    """
    Select exactly the highest-ranked 10%.
    Tie-break by grid ID.
    """

    order = np.lexsort(
        (ids, -scores)
    )

    result = np.zeros(
        len(scores),
        dtype=bool,
    )

    result[
        order[:len(scores) // 10]
    ] = True

    return result


def spearman(reference_ranks, scores):

    other_ranks = (
        pd.Series(scores)
        .rank(method="average")
        .to_numpy(dtype=float)
    )

    x = reference_ranks - reference_ranks.mean()
    y = other_ranks - other_ranks.mean()

    denominator = np.sqrt(
        np.sum(x ** 2) * np.sum(y ** 2)
    )

    if denominator <= 0:
        raise ValueError(
            "Rank correlation undefined."
        )

    return float(
        np.dot(x, y) / denominator
    )


def generate_weight_scenarios(codes, weights):

    scenarios = [
        {
            "scenario": "AHP_REFERENCE",
            "criterion": "REFERENCE",
            "relative_change": 0.0,
            "weights": weights.copy(),
        }
    ]

    for index, code in enumerate(codes):

        for change in PERTURBATIONS:

            adjusted = weights.copy()

            adjusted[index] *= (
                1 + change
            )

            adjusted /= adjusted.sum()

            direction = (
                "PLUS"
                if change > 0
                else "MINUS"
            )

            scenarios.append({
                "scenario": (
                    f"{code}_{direction}"
                    f"{abs(int(round(change * 100)))}PCT"
                ),
                "criterion": code,
                "relative_change": change,
                "weights": adjusted,
            })

    return scenarios


# ============================================================
# 3. MAIN
# ============================================================

def main():

    print("=" * 88)
    print(
        "STEP 16F-01 - POST-AHP "
        "WEIGHT SENSITIVITY"
    )
    print("=" * 88)

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        MODEL_STATUS,
        AHP_STATUS,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    model_meta = read_json(MODEL_STATUS)
    ahp_meta = read_json(AHP_STATUS)

    qa = []

    def check(name, passed, detail=""):

        qa.append({
            "check": name,
            "passed": bool(passed),
            "detail": str(detail),
        })

    def verify():

        failures = [
            row for row in qa
            if not row["passed"]
        ]

        if failures:

            pd.DataFrame(qa).to_csv(
                OUT_QA,
                index=False,
            )

            for row in failures:
                print(
                    "REVIEW:",
                    row["check"],
                    row["detail"],
                )

            raise RuntimeError(
                "Step 16F validation failed. "
                "Inspect QA report."
            )

    def finish(state, explanation, extra=None):

        verify()

        result = {
            "step": "16F-01",
            "status": state,
            "methodology": "GIS_MCDA_EXPERT_AHP",
            "explanation": explanation,
            "sensitivity_completed": (
                state == "POST_AHP_SENSITIVITY_QA_PASS"
            ),
            "external_validation": "NOT_COMPLETED",
            "legal_permission": "NOT_ASSESSED",
            "qa_checks": len(qa),
            "qa_passed": len(qa),
            "created_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        result.update(extra or {})

        pd.DataFrame(qa).to_csv(
            OUT_QA,
            index=False,
        )

        save_json(
            OUT_STATUS,
            result,
        )

        print("\n=== STATUS ===")
        print("Status:", state)
        print("Explanation:", explanation)
        print(
            f"QA: {len(qa)}/{len(qa)} PASS"
        )

        return result

    # ========================================================
    # 4. VERIFY METHODOLOGY AND READINESS
    # ========================================================

    print("\n=== CHECKING AHP READINESS ===")

    print(
        "Step 16E status:",
        model_meta.get("status"),
    )

    print(
        "Expert AHP status:",
        ahp_meta.get("expert_input_status"),
    )

    check(
        "Step 16E uses mandatory AHP methodology",
        model_meta.get("methodology")
        == "GIS_MCDA_AHP_MANDATORY",
    )

    check(
        "AHP intake technical QA passed",
        ahp_meta.get("technical_qa_status")
        == "PASS",
    )

    check(
        "Step 16E validation checks passed",
        int(
            model_meta.get("qa_checks", 0)
        ) > 0
        and model_meta.get("qa_checks")
        == model_meta.get("qa_passed"),
    )

    verify()

    # ========================================================
    # 5. STOP SAFELY IF EXPERT AHP IS UNAVAILABLE
    # ========================================================

    if model_meta.get("status") != (
        "AHP_WEIGHTED_INTERNAL_QA_PASS"
    ):

        check(
            "Uncompleted AHP model is correctly identified",
            model_meta.get(
                "weighted_model_available"
            ) is False,
        )

        finish(
            "BLOCKED_AWAITING_AHP_WEIGHTED_MODEL",

            "Final Step 16E expert-AHP weighted "
            "model is not available. No post-AHP "
            "sensitivity scenarios were calculated.",

            extra={
                "expert_status":
                    ahp_meta.get(
                        "expert_input_status"
                    ),
                "model_status":
                    model_meta.get("status"),
                "scenario_count": 0,
                "final_outputs_generated": False,
            },
        )

        print(
            "\nWaiting for validated AHP weights "
            "and the completed Step 16E model."
        )

        print("=" * 88)
        return

    # ========================================================
    # 6. VALIDATE FINAL AHP MODEL
    # ========================================================

    check(
        "Genuine expert AHP was validated",
        ahp_meta.get("expert_input_status")
        == "EXPERT_AHP_VALIDATED"
        and ahp_meta.get(
            "weights_current_and_validated"
        ) is True,
    )

    check(
        "Step 16E confirms final weighted outputs",
        model_meta.get("weighted_model_available")
        is True
        and model_meta.get(
            "final_outputs_generated"
        ) is True,
    )

    consistency_ratio = model_meta.get(
        "consistency_ratio"
    )

    check(
        "AHP Consistency Ratio acceptable",
        isinstance(
            consistency_ratio,
            (int, float),
        )
        and 0 <= consistency_ratio
        <= CR_LIMIT + 1e-10,
    )

    verify()

    for path in [MODEL_CSV, MODEL_GPKG]:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Verify that the sources used by the final
    # AHP model have not changed since Step 16E.

    provenance = model_meta.get(
        "input_sha256", {}
    )

    check(
        "Final model contains input provenance",
        len(provenance) >= 7,
    )

    verify()

    changed_sources = []

    for filename, recorded_hash in provenance.items():

        source = REPORTS / filename

        if (
            not source.is_file()
            or file_hash(source) != recorded_hash
        ):
            changed_sources.append(filename)

    check(
        "Final model source files unchanged",
        len(changed_sources) == 0,
        str(changed_sources),
    )

    verify()

    source_hashes = {
        path.name: file_hash(path)
        for path in [MODEL_CSV, MODEL_GPKG]
    }

    # ========================================================
    # 7. LOAD FINAL WEIGHTED DATA
    # ========================================================

    print("\n=== LOADING AHP-WEIGHTED RESULTS ===")

    data = pd.read_csv(
        MODEL_CSV,
        dtype={"grid_id": "string"},
    )

    gis = gpd.read_file(
        MODEL_GPKG,
        layer="ahp_weighted_suitability",
    )

    data["grid_id"] = (
        data["grid_id"].str.strip()
    )

    gis["grid_id"] = (
        gis["grid_id"]
        .astype("string")
        .str.strip()
    )

    check(
        "CSV contains 9240 unique grid IDs",
        len(data) == N_GRIDS
        and data["grid_id"].notna().all()
        and data["grid_id"].is_unique,
    )

    check(
        "GIS contains 9240 unique grid IDs",
        len(gis) == N_GRIDS
        and gis["grid_id"].notna().all()
        and gis["grid_id"].is_unique,
    )

    check(
        "Grid ID sets agree",
        set(data["grid_id"])
        == set(gis["grid_id"]),
    )

    check(
        "Final GIS has valid EPSG:32644 polygons",
        gis.crs is not None
        and gis.crs.to_epsg() == EPSG
        and gis.geometry.notna().all()
        and gis.geometry.is_valid.all(),
    )

    verify()

    data = (
        data.sort_values("grid_id")
        .reset_index(drop=True)
    )

    gis = (
        gis.set_index("grid_id")
        .loc[data["grid_id"]]
        .reset_index()
    )

    # ========================================================
    # 8. RECOVER VALIDATED AHP CRITERIA AND WEIGHTS
    # ========================================================

    codes = model_meta.get(
        "approved_criteria"
    )

    check(
        "Three or four approved AHP criteria",
        isinstance(codes, list)
        and len(codes) in [3, 4]
        and len(codes) == len(set(codes)),
    )

    verify()

    check(
        "Criterion set matches AHP intake",
        codes == (
            ahp_meta.get("ahp") or {}
        ).get("criteria"),
    )

    required = {
        "grid_id",
        "ahp_weighted_score",
        "ahp_relative_class",
        "ahp_model_status",
        "development_permission",
        "planning_review_needed",
        "conservation_water_pct",
    }

    for code in codes:
        required.add(
            f"ahp_weight_{code.lower()}"
        )
        required.add(
            f"ahp_contribution_{code.lower()}"
        )

    check(
        "All AHP and planning data fields exist",
        required.issubset(data.columns),
    )

    verify()

    weight_dict = model_meta.get(
        "weights", {}
    )

    check(
        "AHP weights available for approved criteria",
        set(weight_dict) == set(codes),
    )

    verify()

    weights = np.array([
        float(weight_dict[code])
        for code in codes
    ])

    check(
        "AHP weights are positive and sum to one",
        np.isfinite(weights).all()
        and np.all(weights > 0)
        and np.isclose(
            weights.sum(), 1.0,
            atol=1e-10,
        ),
    )

    verify()

    # ========================================================
    # 9. RECONSTRUCT NORMALIZED CRITERION MATRIX
    # ========================================================

    # The original Step 16E model stores each
    # normalized criterion's weighted contribution.
    #
    # contribution_ij = criterion_score_ij * weight_j
    #
    # Recover criterion scores without guessing
    # any C3 normalization direction.

    contributions = np.column_stack([
        pd.to_numeric(
            data[f"ahp_contribution_{code.lower()}"],
            errors="coerce",
        ).to_numpy(dtype=float)
        for code in codes
    ])

    X = contributions / weights[None, :]

    reference = pd.to_numeric(
        data["ahp_weighted_score"],
        errors="coerce",
    ).to_numpy(dtype=float)

    ids = data["grid_id"].astype(str).to_numpy()

    reference_classes = (
        data["ahp_relative_class"]
        .astype(str)
        .to_numpy()
    )

    review = parse_boolean(
        data["planning_review_needed"]
    )

    conservation = pd.to_numeric(
        data["conservation_water_pct"],
        errors="coerce",
    ).to_numpy(dtype=float)

    check(
        "Recovered criterion scores finite and normalized",
        np.isfinite(X).all()
        and np.all(X >= -1e-10)
        and np.all(X <= 1 + 1e-10),
    )

    check(
        "AHP weighted scores reproduced",
        np.allclose(
            X @ weights,
            reference,
            atol=1e-10,
        ),
    )

    check(
        "Final GIS scores agree with CSV",
        np.allclose(
            pd.to_numeric(
                gis["ahp_weighted_score"]
            ),
            reference,
            atol=1e-10,
        ),
    )

    check(
        "Final relative classes reproduced",
        np.array_equal(
            relative_classes(reference, ids),
            reference_classes,
        ),
    )

    check(
        "Final model and legal status correct",
        data["ahp_model_status"].eq(
            "EXPERT_AHP_WEIGHTED_INTERNAL_QA"
        ).all()
        and data["development_permission"].eq(
            "NOT_ASSESSED"
        ).all(),
    )

    check(
        "Conservation screening is finite",
        np.isfinite(conservation).all(),
    )

    verify()

    # ========================================================
    # 10. GENERATE POST-AHP SENSITIVITY SCENARIOS
    # ========================================================

    print("\n=== GENERATING WEIGHT PERTURBATIONS ===")

    scenarios = generate_weight_scenarios(
        codes, weights
    )

    check(
        "Correct sensitivity scenario count",
        len(scenarios)
        == 1 + 4 * len(codes),
    )

    check(
        "All sensitivity weights valid",
        all(
            np.all(s["weights"] > 0)
            and np.isclose(
                s["weights"].sum(),
                1.0,
                atol=1e-12,
            )
            for s in scenarios
        ),
    )

    verify()

    # ========================================================
    # 11. REFERENCE RANKS AND HIGH-RANKED CELLS
    # ========================================================

    reference_ranks = (
        pd.Series(reference)
        .rank(method="average")
        .to_numpy(dtype=float)
    )

    reference_top = top_decile(
        reference, ids
    )

    scenario_rows = []
    transition_rows = []
    all_top_masks = []
    all_scores = []

    # ========================================================
    # 12. EVALUATE EACH SCENARIO
    # ========================================================

    for item in scenarios:

        scenario_name = item["scenario"]
        scenario_weights = item["weights"]

        scores = X @ scenario_weights

        classes = relative_classes(
            scores, ids
        )

        top = top_decile(
            scores, ids
        )

        all_top_masks.append(top)
        all_scores.append(scores)

        intersection_count = int(
            np.count_nonzero(
                top & reference_top
            )
        )

        union_count = int(
            np.count_nonzero(
                top | reference_top
            )
        )

        jaccard = (
            intersection_count / union_count
        )

        class_change_count = int(
            np.count_nonzero(
                classes != reference_classes
            )
        )

        row = {
            "scenario_id": scenario_name,

            "perturbed_criterion":
                item["criterion"],

            "relative_change_before_normalization_pct":
                100 * item["relative_change"],

            "scenario_type": (
                "AHP_REFERENCE"
                if item["criterion"] == "REFERENCE"
                else "SENSITIVITY_NOT_EXPERT_JUDGMENT"
            ),

            "spearman_vs_ahp_reference":
                spearman(
                    reference_ranks, scores
                ),

            "top_decile_count":
                int(top.sum()),

            "top_decile_jaccard":
                float(jaccard),

            "top_decile_overlap_count":
                intersection_count,

            "class_changed_cells":
                class_change_count,

            "class_changed_pct":
                100 * class_change_count / N_GRIDS,

            "mean_abs_score_change":
                float(
                    np.abs(
                        scores - reference
                    ).mean()
                ),

            "top_decile_planning_review_cells":
                int(
                    np.count_nonzero(
                        top & review
                    )
                ),

            "top_decile_conservation_intersection_cells":
                int(
                    np.count_nonzero(
                        top & (conservation > 0)
                    )
                ),
        }

        for i, code in enumerate(codes):
            row[f"weight_{code.lower()}"] = (
                float(scenario_weights[i])
            )

        scenario_rows.append(row)

        # Complete 5x5 relative-class transition
        # matrix for each non-reference scenario.

        if scenario_name != "AHP_REFERENCE":

            for original_class in CLASS_NAMES:

                original_mask = (
                    reference_classes
                    == original_class
                )

                for new_class in CLASS_NAMES:

                    count = int(
                        np.count_nonzero(
                            original_mask
                            & (classes == new_class)
                        )
                    )

                    transition_rows.append({
                        "scenario_id": scenario_name,
                        "reference_class": original_class,
                        "sensitivity_class": new_class,
                        "grid_cells": count,
                    })

    # ========================================================
    # 13. GRID-LEVEL ROBUSTNESS INDICATORS
    # ========================================================

    scenario_table = pd.DataFrame(
        scenario_rows
    )

    top_matrix = np.column_stack(
        all_top_masks
    )

    score_matrix = np.column_stack(
        all_scores
    )

    frequency = top_matrix.mean(axis=1)

    grid_result = data[
        [
            "grid_id",
            "ahp_weighted_score",
            "ahp_relative_class",
            "planning_review_needed",
            "conservation_water_pct",
        ]
    ].copy()

    grid_result[
        "top10_sensitivity_frequency"
    ] = frequency

    grid_result[
        "top10_in_all_scenarios"
    ] = top_matrix.all(axis=1)

    grid_result[
        "top10_frequency_at_least_75pct"
    ] = frequency >= STABILITY_THRESHOLD

    grid_result[
        "sensitivity_score_min"
    ] = score_matrix.min(axis=1)

    grid_result[
        "sensitivity_score_max"
    ] = score_matrix.max(axis=1)

    grid_result[
        "sensitivity_score_range"
    ] = (
        grid_result["sensitivity_score_max"]
        - grid_result["sensitivity_score_min"]
    )

    grid_result[
        "scenario_count"
    ] = len(scenarios)

    grid_result[
        "sensitivity_status"
    ] = "POST_AHP_INTERNAL_ROBUSTNESS"

    transitions = pd.DataFrame(
        transition_rows
    )

    # ========================================================
    # 14. FINAL RESEARCH VALIDATION
    # ========================================================

    check(
        "Scenario report has expected rows",
        len(scenario_table)
        == 1 + 4 * len(codes),
    )

    check(
        "Every scenario contains 924 top-decile grids",
        scenario_table[
            "top_decile_count"
        ].eq(924).all(),
    )

    check(
        "Spearman values within valid bounds",
        scenario_table[
            "spearman_vs_ahp_reference"
        ].between(
            -1 - 1e-10,
            1 + 1e-10,
        ).all(),
    )

    check(
        "Top-decile Jaccard values valid",
        scenario_table[
            "top_decile_jaccard"
        ].between(0, 1).all(),
    )

    check(
        "Sensitivity frequencies valid",
        np.isfinite(frequency).all()
        and np.all(
            (frequency >= 0)
            & (frequency <= 1)
        ),
    )

    check(
        "Original AHP scores are unchanged",
        np.allclose(
            score_matrix[:, 0],
            reference,
            atol=1e-10,
        ),
    )

    check(
        "Each class transition covers all 9240 grids",
        all(
            transitions.loc[
                transitions["scenario_id"]
                == item["scenario"],
                "grid_cells",
            ].sum() == N_GRIDS

            for item in scenarios
            if item["scenario"] != "AHP_REFERENCE"
        ),
    )

    check(
        "Final AHP input outputs remain unchanged",
        all(
            file_hash(path)
            == source_hashes[path.name]
            for path in [MODEL_CSV, MODEL_GPKG]
        ),
    )

    verify()

    # ========================================================
    # 15. PREPARE GIS LAYER
    # ========================================================

    additional_columns = [
        "grid_id",
        "top10_sensitivity_frequency",
        "top10_in_all_scenarios",
        "top10_frequency_at_least_75pct",
        "sensitivity_score_min",
        "sensitivity_score_max",
        "sensitivity_score_range",
        "scenario_count",
        "sensitivity_status",
    ]

    mapped = gis.merge(
        grid_result[additional_columns],
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    staging = OUT_GPKG.with_name(
        OUT_GPKG.stem + "_staging.gpkg"
    )

    if staging.exists():
        staging.unlink()

    mapped.to_file(
        staging,
        layer="post_ahp_sensitivity",
        driver="GPKG",
        index=False,
    )

    saved = gpd.read_file(
        staging,
        layer="post_ahp_sensitivity",
    )

    check(
        "Sensitivity GIS retains all valid grids",
        len(saved) == N_GRIDS
        and saved["grid_id"].is_unique
        and saved.geometry.is_valid.all(),
    )

    check(
        "GIS retains grid robustness indicators",
        np.allclose(
            saved[
                "top10_sensitivity_frequency"
            ].to_numpy(dtype=float),
            frequency,
            atol=1e-10,
        ),
    )

    verify()

    del saved

    staging.replace(OUT_GPKG)

    # ========================================================
    # 16. SAVE FINAL REPORTS
    # ========================================================

    scenario_table.to_csv(
        OUT_SCENARIOS,
        index=False,
    )

    grid_result.to_csv(
        OUT_GRID,
        index=False,
    )

    transitions.to_csv(
        OUT_TRANSITIONS,
        index=False,
    )

    sensitivity_only = (
        scenario_table.loc[
            scenario_table["scenario_id"]
            != "AHP_REFERENCE"
        ]
    )

    finish(
        "POST_AHP_SENSITIVITY_QA_PASS",

        "Sensitivity analysis around validated expert "
        "AHP weights passed internal QA. "
        "External validation remains separate.",

        extra={
            "approved_criteria": codes,

            "scenario_count": len(scenarios),

            "perturbations": PERTURBATIONS,

            "minimum_spearman":
                float(
                    sensitivity_only[
                        "spearman_vs_ahp_reference"
                    ].min()
                ),

            "minimum_top_decile_jaccard":
                float(
                    sensitivity_only[
                        "top_decile_jaccard"
                    ].min()
                ),

            "grids_top10_all_scenarios":
                int(
                    top_matrix.all(axis=1).sum()
                ),

            "grids_top10_frequency_ge_75pct":
                int(
                    (frequency >= 0.75).sum()
                ),

            "final_outputs_generated": True,

            "output_files": [
                OUT_SCENARIOS.name,
                OUT_GRID.name,
                OUT_GPKG.name,
                OUT_TRANSITIONS.name,
            ],
        },
    )

    print("\n=== SENSITIVITY SUMMARY ===")

    print(
        scenario_table[
            [
                "scenario_id",
                "spearman_vs_ahp_reference",
                "top_decile_jaccard",
                "class_changed_pct",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n=== SAVED RESEARCH REPORTS ===")

    for path in [
        OUT_SCENARIOS,
        OUT_GRID,
        OUT_GPKG,
        OUT_TRANSITIONS,
        OUT_QA,
        OUT_STATUS,
    ]:
        print(path.relative_to(ROOT))

    print("=" * 88)


if __name__ == "__main__":
    main()
