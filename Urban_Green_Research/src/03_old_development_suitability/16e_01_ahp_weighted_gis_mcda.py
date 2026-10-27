
"""
J26-DS-338
STEP 16E-01 - AHP-WEIGHTED GIS-MCDA SUITABILITY

Approved research methodology: GIS-MCDA with AHP.

Inputs:
- Validated standardized suitability criteria
- Validated grid geometries and GN integration
- Genuine expert AHP judgments and consistency results

If expert AHP is pending:
- Validate prepared research inputs
- Save a permanent QA and status report
- Do NOT calculate or publish weighted suitability

If valid AHP is available:
- Verify approved criteria and directions
- Independently check AHP consistency
- Calculate weighted GIS-MCDA scores
- Export reproducible GIS and CSV results

No invented weights or legal planning permissions.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# 1. INPUT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "outputs" / "reports"

STANDARD_CSV = REPORTS / (
    "16d_a_provisional_standardized_criteria_100m.csv"
)

STANDARD_META = REPORTS / (
    "16d_a_standardized_criteria_summary.json"
)

GRID_GPKG = REPORTS / (
    "16d_h_grid_with_gn_divisions_100m.gpkg"
)

GN_META = REPORTS / (
    "16d_h_gn_integration_summary.json"
)

REVIEW_FORM = REPORTS / (
    "16c_l_expert_criteria_review_template.csv"
)

PAIRWISE_FORM = REPORTS / (
    "16c_l_ahp_pairwise_template.csv"
)

AHP_META = REPORTS / (
    "16d_d_ahp_consistency_summary.json"
)

AHP_WEIGHTS = REPORTS / (
    "16d_d_validated_expert_ahp_weights.csv"
)


# ============================================================
# 2. OUTPUT PATHS
# ============================================================

OUT_QA = REPORTS / (
    "16e_01_ahp_weighted_qa.csv"
)

OUT_STATUS = REPORTS / (
    "16e_01_ahp_weighted_status.json"
)

OUT_CSV = REPORTS / (
    "16e_01_ahp_weighted_grid_100m.csv"
)

OUT_GPKG = REPORTS / (
    "16e_01_ahp_weighted_grid_100m.gpkg"
)

OUT_CLASSES = REPORTS / (
    "16e_01_ahp_weighted_class_summary.csv"
)


# ============================================================
# 3. METHODOLOGY CONFIGURATION
# ============================================================

N_GRIDS = 9240
EXPECTED_EPSG = 32644
CR_LIMIT = 0.10

RI = {
    3: 0.58,
    4: 0.90,
}

SCENARIOS = {
    "BASE_3_CRITERIA": [
        "C1", "C2", "C4"
    ],
    "CONDITIONAL_4_WITH_C3": [
        "C1", "C2", "C3", "C4"
    ],
}

CORE_SCORE_COLUMNS = {
    "C1": "s_c1_road_pc",
    "C2": "s_c2_road_density_pc",
    "C4": "s_c4_slope_pc",
}

CORE_DIRECTIONS = {
    "C1": "DECREASING",
    "C2": "INCREASING",
    "C4": "DECREASING",
}

# C3 is intentionally NOT guessed.
# If an expert approves C3, its direction-specific
# standardized column must be confirmed from the
# actual frozen Step 16D-A dataset before scoring.
#
# These None values are intentional safeguards.

C3_SCORE_COLUMNS = {
    "INCREASING": None,
    "DECREASING": None,
}

CLASSES = [
    "RELATIVE_VERY_LOW",
    "RELATIVE_LOW",
    "RELATIVE_MODERATE",
    "RELATIVE_HIGH",
    "RELATIVE_VERY_HIGH",
]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as stream:

        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):

    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path, value):

    with path.open(
        "w",
        encoding="utf-8",
    ) as stream:

        json.dump(
            value,
            stream,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 88)
    print(
        "STEP 16E-01 - AHP-WEIGHTED GIS-MCDA"
    )
    print("=" * 88)

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    inputs = [
        STANDARD_CSV,
        STANDARD_META,
        GRID_GPKG,
        GN_META,
        REVIEW_FORM,
        PAIRWISE_FORM,
        AHP_META,
    ]

    for path in inputs:

        if not path.is_file():

            raise FileNotFoundError(
                f"Missing research input: {path}"
            )

    original_hashes = {
        path.name: sha256(path)
        for path in inputs
    }

    standard_meta = load_json(STANDARD_META)
    gn_meta = load_json(GN_META)
    ahp_meta = load_json(AHP_META)

    checks = []

    def check(name, passed):

        checks.append({
            "check": name,
            "passed": bool(passed),
        })

    def save_qa():

        pd.DataFrame(checks).to_csv(
            OUT_QA,
            index=False,
        )

    def require_pass():

        failed = [
            item for item in checks
            if not item["passed"]
        ]

        if failed:

            save_qa()

            for item in failed:

                print(
                    "REVIEW:",
                    item["check"],
                )

            raise RuntimeError(
                "Step 16E input QA failed. "
                "See 16e_01_ahp_weighted_qa.csv"
            )

    def save_status(
        state,
        explanation,
        extra=None,
    ):
        """
        Safe status writer.

        'extra' is an ordinary Python dictionary.
        This avoids incorrect keyword syntax
        in function calls.
        """

        require_pass()

        additional = (
            extra if extra is not None else {}
        )

        result = {
            "step": "16E-01",

            "status": state,

            "methodology":
                "GIS_MCDA_AHP_MANDATORY",

            "explanation": explanation,

            "expert_input_status":
                ahp_meta.get(
                    "expert_input_status"
                ),

            "weighted_model_available":
                state == "AHP_WEIGHTED_INTERNAL_QA_PASS",

            "external_validation":
                "NOT_COMPLETED",

            "legal_permission":
                "NOT_ASSESSED",

            "qa_checks":
                len(checks),

            "qa_passed":
                sum(
                    item["passed"]
                    for item in checks
                ),

            "input_sha256":
                original_hashes,

            "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            **additional,
        }

        save_qa()
        save_json(OUT_STATUS, result)

        print("\n=== EXECUTION STATUS ===")

        print("Status:", state)

        print(
            "Explanation:",
            explanation,
        )

        print(
            f"QA: {result['qa_passed']}/"
            f"{result['qa_checks']} PASS"
        )

        return result

    # ========================================================
    # 6. PREREQUISITE VALIDATION
    # ========================================================

    print("\n=== VALIDATING PREREQUISITES ===")

    check(
        "Standardized criteria QA passed",
        standard_meta.get("qa_status")
        == "PASS",
    )

    check(
        "GN integration QA passed",
        gn_meta.get("qa_status")
        == "PASS",
    )

    check(
        "AHP intake technical QA passed",
        ahp_meta.get("technical_qa_status")
        == "PASS",
    )

    stored_hashes = ahp_meta.get(
        "input_sha256", {}
    )

    check(
        "Expert review form matches AHP intake",
        stored_hashes.get(REVIEW_FORM.name)
        == sha256(REVIEW_FORM),
    )

    check(
        "Pairwise form matches AHP intake",
        stored_hashes.get(PAIRWISE_FORM.name)
        == sha256(PAIRWISE_FORM),
    )

    require_pass()

    # ========================================================
    # 7. LOAD RESEARCH DATA
    # ========================================================

    print("\n=== LOADING RESEARCH DATA ===")

    criteria = pd.read_csv(
        STANDARD_CSV,
        dtype={"grid_id": "string"},
    )

    grid = gpd.read_file(
        GRID_GPKG,
        layer="suitability_grid_with_gn",
    )

    criteria["grid_id"] = (
        criteria["grid_id"].str.strip()
    )

    grid["grid_id"] = (
        grid["grid_id"]
        .astype("string")
        .str.strip()
    )

    check(
        "Criteria contain 9240 unique grid IDs",
        len(criteria) == N_GRIDS
        and criteria["grid_id"].notna().all()
        and criteria["grid_id"].is_unique,
    )

    check(
        "GIS contains 9240 unique grid IDs",
        len(grid) == N_GRIDS
        and grid["grid_id"].notna().all()
        and grid["grid_id"].is_unique,
    )

    check(
        "Grid ID sets match",
        set(criteria["grid_id"])
        == set(grid["grid_id"]),
    )

    check(
        "GIS uses EPSG:32644",
        grid.crs is not None
        and grid.crs.to_epsg() == EXPECTED_EPSG,
    )

    check(
        "Grid polygon geometries are valid",
        grid.geometry.notna().all()
        and (~grid.geometry.is_empty).all()
        and grid.geometry.is_valid.all(),
    )

    check(
        "Core standardized criterion columns exist",
        set(
            CORE_SCORE_COLUMNS.values()
        ).issubset(criteria.columns),
    )

    check(
        "Reference score and status columns exist",
        {
            "suitability_score",
            "model_status",
            "development_permission",
            "cell_area_m2",
        }.issubset(grid.columns),
    )

    require_pass()

    # ========================================================
    # 8. ALIGN ALL DATA BY GRID ID
    # ========================================================

    grid = (
        grid.sort_values("grid_id")
        .reset_index(drop=True)
    )

    criteria = (
        criteria
        .set_index("grid_id")
        .loc[grid["grid_id"]]
    )

    core_columns = [
        CORE_SCORE_COLUMNS[c]
        for c in ["C1", "C2", "C4"]
    ]

    core_scores = (
        criteria[core_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    reference_scores = pd.to_numeric(
        grid["suitability_score"],
        errors="coerce",
    ).to_numpy(dtype=float)

    area_m2 = pd.to_numeric(
        grid["cell_area_m2"],
        errors="coerce",
    ).to_numpy(dtype=float)

    check(
        "Core scores finite and between 0 and 1",
        np.isfinite(core_scores).all()
        and np.all(
            (core_scores >= 0)
            & (core_scores <= 1)
        ),
    )

    check(
        "Provisional baseline reproduced exactly",
        np.allclose(
            core_scores.mean(axis=1),
            reference_scores,
            atol=1e-10,
        ),
    )

    check(
        "Reference model is not incorrectly labelled AHP",
        grid["model_status"].eq(
            "PROVISIONAL_REFERENCE_NOT_AHP"
        ).all(),
    )

    check(
        "Development permissions are unassessed",
        grid["development_permission"]
        .eq("NOT_ASSESSED").all(),
    )

    check(
        "Grid areas valid and preserve Kaduwela area",
        np.isfinite(area_m2).all()
        and np.all(area_m2 > 0)
        and abs(
            area_m2.sum() / 10000
            - float(gn_meta["grid_area_ha"])
        ) < 0.05,
    )

    check(
        "Original input files unchanged",
        all(
            sha256(path)
            == original_hashes[path.name]
            for path in inputs
        ),
    )

    require_pass()

    # ========================================================
    # 9. EXPERT AHP READINESS GATE
    # ========================================================

    print("\n=== EXPERT AHP READINESS ===")

    ahp_status = ahp_meta.get(
        "expert_input_status"
    )

    print(
        "Current expert status:",
        ahp_status,
    )

    if ahp_status != "EXPERT_AHP_VALIDATED":

        c3_candidates = [
            column
            for column in criteria.columns
            if "c3" in column.lower()
        ]

        save_status(
            "BLOCKED_AWAITING_VALID_EXPERT_AHP",

            "GIS criteria and provisional reference "
            "passed validation. Genuine consistent "
            "expert AHP weights are not available. "
            "Weighted suitability was not calculated.",

            extra={
                "candidate_c3_columns":
                    c3_candidates,

                "final_outputs_generated":
                    False,

                "existing_weighted_outputs":
                    "DO_NOT_USE_UNLESS_CURRENT_STATUS_VALID",
            },
        )

        print(
            "\nNo new AHP-weighted GIS outputs created."
        )

        print(
            "Waiting for genuine expert judgments."
        )

        print("=" * 88)

        return

    # ========================================================
    # 10. VERIFY ACTUAL AHP WEIGHTS
    # ========================================================

    print("\n=== VERIFYING EXPERT AHP ===")

    if not AHP_WEIGHTS.is_file():

        raise FileNotFoundError(
            "Expert AHP status is validated, "
            "but the weights CSV is missing."
        )

    ahp = ahp_meta.get("ahp") or {}

    scenario = ahp.get("scenario")

    approved = SCENARIOS.get(scenario)

    check(
        "Approved AHP scenario is supported",
        approved is not None,
    )

    require_pass()

    check(
        "Approved criteria match AHP scenario",
        ahp.get("criteria") == approved,
    )

    check(
        "AHP intake validated the current weights",
        ahp_meta.get(
            "weights_current_and_validated"
        ) is True,
    )

    weight_df = pd.read_csv(
        AHP_WEIGHTS,
        dtype={
            "criterion": str,
            "scenario": str,
        },
    )

    check(
        "Weights contain exact approved criteria",
        len(weight_df) == len(approved)
        and weight_df["criterion"].is_unique
        and set(weight_df["criterion"])
        == set(approved),
    )

    check(
        "Weight scenario matches approval",
        weight_df["scenario"]
        .eq(scenario).all(),
    )

    check(
        "Weights match genuine input form hashes",
        weight_df["review_csv_sha256"]
        .eq(sha256(REVIEW_FORM)).all()
        and weight_df["pairwise_csv_sha256"]
        .eq(sha256(PAIRWISE_FORM)).all(),
    )

    require_pass()

    weights = (
        weight_df
        .set_index("criterion")
        .loc[approved, "ahp_weight"]
        .astype(float)
        .to_numpy(dtype=float)
    )

    matrix = np.asarray(
        ahp["matrix"],
        dtype=float,
    )

    n = len(approved)

    check(
        "AHP weights positive and sum to one",
        np.isfinite(weights).all()
        and np.all(weights > 0)
        and np.isclose(
            weights.sum(),
            1,
            atol=1e-10,
        ),
    )

    check(
        "Weights agree with AHP intake summary",
        all(
            np.isclose(
                weights[i],
                float(
                    ahp["weights"][code]
                ),
                atol=1e-10,
            )
            for i, code in enumerate(approved)
        ),
    )

    check(
        "AHP matrix positive and reciprocal",
        matrix.shape == (n, n)
        and np.isfinite(matrix).all()
        and np.all(matrix > 0)
        and np.allclose(
            matrix * matrix.T,
            1,
            atol=1e-10,
        ),
    )

    require_pass()

    # ========================================================
    # 11. INDEPENDENT CONSISTENCY RATIO CHECK
    # ========================================================

    eigenvalues = np.linalg.eigvals(matrix)

    lambda_max = float(
        eigenvalues.real.max()
    )

    ci = max(
        0.0,
        (lambda_max - n) / (n - 1),
    )

    cr = float(
        ci / RI[n]
    )

    check(
        "AHP CR is at most 0.10",
        cr <= CR_LIMIT + 1e-10,
    )

    check(
        "Recomputed CR agrees with intake",
        np.isclose(
            cr,
            float(
                ahp["consistency_ratio"]
            ),
            atol=1e-8,
        ),
    )

    check(
        "Weights satisfy principal eigenvector relation",
        np.allclose(
            matrix @ weights,
            lambda_max * weights,
            atol=1e-7,
        ),
    )

    require_pass()

    # ========================================================
    # 12. VERIFY CRITERIA DECISIONS AND DIRECTIONS
    # ========================================================

    review = pd.read_csv(
        REVIEW_FORM,
        dtype=str,
        keep_default_na=False,
    )

    for column in review.columns:

        review[column] = (
            review[column].str.strip()
        )

    review = review.set_index("criterion")

    decisions = (
        review["expert_decision"]
        .str.upper()
    )

    all_criteria = {
        "C1", "C2", "C3", "C4", "C5"
    }

    check(
        "Criteria inclusion matches expert review",
        all(
            decisions.loc[code] == "INCLUDE"
            for code in approved
        )
        and all(
            decisions.loc[code] == "EXCLUDE"
            for code in (
                all_criteria - set(approved)
            )
        ),
    )

    check(
        "Core directions match standardized scoring",
        all(
            review.loc[
                code,
                "expert_direction"
            ].upper() == direction

            for code, direction
            in CORE_DIRECTIONS.items()
        ),
    )

    require_pass()

    # ========================================================
    # 13. SELECT APPROVED STANDARDIZED CRITERIA
    # ========================================================

    selected_columns = []

    for code in approved:

        if code in CORE_SCORE_COLUMNS:

            selected_columns.append(
                CORE_SCORE_COLUMNS[code]
            )

        elif code == "C3":

            direction = review.loc[
                "C3",
                "expert_direction",
            ].upper()

            c3_column = (
                C3_SCORE_COLUMNS.get(direction)
            )

            if (
                c3_column is None
                or c3_column not in criteria.columns
            ):

                save_status(
                    "BLOCKED_C3_COLUMN_CONFIRMATION",

                    "C3 was approved, but its exact "
                    "direction-specific standardized "
                    "column must be verified before "
                    "weighted scoring.",

                    extra={
                        "approved_c3_direction":
                            direction,

                        "candidate_c3_columns": [
                            col
                            for col in criteria.columns
                            if "c3" in col.lower()
                        ],

                        "final_outputs_generated":
                            False,
                    },
                )

                return

            selected_columns.append(
                c3_column
            )

    X = (
        criteria[selected_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    check(
        "All approved criterion scores valid",
        np.isfinite(X).all()
        and np.all(
            (X >= 0)
            & (X <= 1)
        ),
    )

    require_pass()

    # ========================================================
    # 14. CALCULATE AHP-WEIGHTED SUITABILITY
    # ========================================================

    print("\n=== CALCULATING WEIGHTED SUITABILITY ===")

    weighted_scores = X @ weights

    contributions = X * weights

    check(
        "AHP suitability scores in valid range",
        np.isfinite(weighted_scores).all()
        and np.all(
            (weighted_scores >= -1e-12)
            & (weighted_scores <= 1 + 1e-12)
        ),
    )

    check(
        "Weighted scores equal criterion contributions",
        np.allclose(
            weighted_scores,
            contributions.sum(axis=1),
            atol=1e-12,
        ),
    )

    require_pass()

    # ========================================================
    # 15. CREATE RELATIVE QUINTILE CLASSES
    # ========================================================

    # Deterministic sorting by score and grid ID.
    # These classes are relative to Kaduwela.
    # They are NOT legal suitability thresholds.

    order = np.lexsort((
        grid["grid_id"].astype(str).to_numpy(),
        weighted_scores,
    ))

    group_numbers = np.empty(
        N_GRIDS,
        dtype=int,
    )

    group_numbers[order] = (
        np.arange(N_GRIDS)
        * 5
        // N_GRIDS
    )

    classes = np.asarray(
        CLASSES
    )[group_numbers]

    ranks = (
        pd.Series(weighted_scores)
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
        .to_numpy()
    )

    check(
        "Five relative classes contain 1848 grids each",
        all(
            int(
                (classes == name).sum()
            ) == 1848
            for name in CLASSES
        ),
    )

    require_pass()

    # ========================================================
    # 16. PREPARE FINAL MODEL GIS DATA
    # ========================================================

    output = grid.copy()

    output["ahp_weighted_score"] = (
        weighted_scores
    )

    output["ahp_relative_class"] = classes

    output["ahp_rank_desc"] = ranks

    output["ahp_scenario"] = scenario

    output["ahp_consistency_ratio"] = cr

    output["ahp_model_status"] = (
        "EXPERT_AHP_WEIGHTED_INTERNAL_QA"
    )

    # Existing suitability_score column remains
    # the OLD equal-weight reference score.
    # ahp_weighted_score is the NEW model result.

    for i, code in enumerate(approved):

        output[
            f"ahp_weight_{code.lower()}"
        ] = float(weights[i])

        output[
            f"ahp_contribution_{code.lower()}"
        ] = contributions[:, i]

    # ========================================================
    # 17. CLASS SUMMARY
    # ========================================================

    class_summary = (
        output.drop(columns="geometry")
        .groupby(
            "ahp_relative_class",
            as_index=False,
        )
        .agg(
            grid_cells=(
                "grid_id",
                "size",
            ),

            mean_score=(
                "ahp_weighted_score",
                "mean",
            ),

            min_score=(
                "ahp_weighted_score",
                "min",
            ),

            max_score=(
                "ahp_weighted_score",
                "max",
            ),

            area_m2=(
                "cell_area_m2",
                "sum",
            ),
        )
    )

    class_summary["area_ha"] = (
        class_summary["area_m2"]
        / 10000
    )

    # ========================================================
    # 18. EXPORT AND VERIFY GIS
    # ========================================================

    temporary = OUT_GPKG.with_name(
        OUT_GPKG.stem + "_staging.gpkg"
    )

    if temporary.exists():
        temporary.unlink()

    output.to_file(
        temporary,
        layer="ahp_weighted_suitability",
        driver="GPKG",
        index=False,
    )

    saved = gpd.read_file(
        temporary,
        layer="ahp_weighted_suitability",
    )

    check(
        "Exported GIS retains 9240 unique grids",
        len(saved) == N_GRIDS
        and saved["grid_id"].is_unique
        and saved.geometry.notna().all()
        and saved.geometry.is_valid.all(),
    )

    saved_scores = (
        saved
        .set_index("grid_id")
        .reindex(
            grid["grid_id"]
        )["ahp_weighted_score"]
        .to_numpy(dtype=float)
    )

    check(
        "Exported GIS retains calculated AHP scores",
        np.allclose(
            saved_scores,
            weighted_scores,
            atol=1e-10,
        ),
    )

    check(
        "All frozen input files unchanged",
        all(
            sha256(path)
            == original_hashes[path.name]
            for path in inputs
        ),
    )

    require_pass()

    del saved

    temporary.replace(OUT_GPKG)

    output.drop(
        columns="geometry"
    ).to_csv(
        OUT_CSV,
        index=False,
    )

    class_summary.to_csv(
        OUT_CLASSES,
        index=False,
    )

    # ========================================================
    # 19. SAVE FINAL INTERNAL VALIDATION STATUS
    # ========================================================

    report = save_status(
        "AHP_WEIGHTED_INTERNAL_QA_PASS",

        "Actual expert AHP weights passed "
        "provenance, consistency, scoring "
        "and GIS output checks. Independent "
        "external validation is still pending.",

        extra={
            "scenario": scenario,

            "approved_criteria": approved,

            "weights": {
                code: float(weights[i])
                for i, code in enumerate(approved)
            },

            "consistency_ratio": cr,

            "score_min":
                float(weighted_scores.min()),

            "score_max":
                float(weighted_scores.max()),

            "score_mean":
                float(weighted_scores.mean()),

            "final_outputs_generated": True,

            "output_files": [
                OUT_CSV.name,
                OUT_GPKG.name,
                OUT_CLASSES.name,
            ],
        },
    )

    print("\n=== WEIGHTED MODEL OUTPUTS ===")

    for path in [
        OUT_CSV,
        OUT_GPKG,
        OUT_CLASSES,
        OUT_QA,
        OUT_STATUS,
    ]:

        print(path.relative_to(ROOT))

    print(
        "Consistency Ratio:",
        report["consistency_ratio"],
    )

    print(
        "STATUS: AHP internal validation passed."
    )

    print(
        "External validation remains pending."
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
