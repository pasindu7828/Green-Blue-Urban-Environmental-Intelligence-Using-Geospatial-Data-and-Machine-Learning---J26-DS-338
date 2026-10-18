
"""
STEP 16D-B - PROVISIONAL SUITABILITY BASELINE

Research component:
Urban Green Transition, Urbanization Pressure
and Conservation Intelligence

Study area:
Kaduwela Municipal Council, Sri Lanka

PURPOSE
1. Calculate a transparent three-criterion reference baseline.
2. Compare percentile-capped and percentile-rank scoring.
3. Prepare a provisional grid-level suitability map.
4. Attach separate zoning and conservation review indicators.
5. Validate against previously completed research steps.
6. Preserve all original research inputs.

METHODOLOGY
Baseline = (C1 + C2 + C4) / 3

The equal contributions are illustrative reference assumptions.
They are NOT AHP expert weights.

C3 is excluded because its inclusion/direction remains undecided.
C5 remains raw context without a justified city-wide direction.

All classifications are relative analytical descriptions.
They are NOT legal development suitability approvals.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REPORTS = ROOT / "outputs/reports"

STANDARD_CSV = (
    REPORTS
    / "16d_a_provisional_standardized_criteria_100m.csv"
)

STANDARD_GPKG = (
    REPORTS
    / "16d_a_provisional_standardized_criteria_100m.gpkg"
)

STANDARD_SUMMARY = (
    REPORTS
    / "16d_a_standardized_criteria_summary.json"
)

SCREENING_CSV = (
    REPORTS
    / "16c_m_grid_zoning_screening_100m.csv"
)

SCREENING_SUMMARY = (
    REPORTS
    / "16c_m_grid_zoning_screening_summary.json"
)

PREVIOUS_K_CSV = (
    REPORTS
    / "16c_k_criterion_inclusion_scenarios_100m.csv"
)

PREVIOUS_K_SUMMARY = (
    REPORTS
    / "16c_k_criterion_robustness_summary.json"
)

EXPERT_SUMMARY = (
    REPORTS
    / "16c_l_expert_review_summary.json"
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

OUT_CSV = (
    REPORTS
    / "16d_b_provisional_suitability_baseline_100m.csv"
)

OUT_GPKG = (
    REPORTS
    / "16d_b_provisional_suitability_baseline_100m.gpkg"
)

OUT_CLASSES = (
    REPORTS
    / "16d_b_relative_suitability_class_summary.csv"
)

OUT_SENSITIVITY = (
    REPORTS
    / "16d_b_baseline_method_sensitivity.csv"
)

OUT_QA = (
    REPORTS
    / "16d_b_provisional_baseline_qa.csv"
)

OUT_SUMMARY = (
    REPORTS
    / "16d_b_provisional_baseline_summary.json"
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

EXPECTED_GRID_COUNT = 9240
EXPECTED_EPSG = 32644

# Analytical threshold for checking rank changes.
# NOT a government or development-planning threshold.
SCORE_DIFFERENCE_REVIEW = 0.20

# For recording grid fragments smaller than half
# the nominal 100 m x 100 m cell.
# NOT a minimum legal land parcel size.
HALF_CELL_AREA_M2 = 5000.0

CRITERIA = [
    ("C1", "s_c1_road_pc", "sens_c1_rank"),
    ("C2", "s_c2_road_density_pc", "sens_c2_rank"),
    ("C4", "s_c4_slope_pc", "sens_c4_rank"),
]

CLASS_LABELS = [
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

    with path.open("rb") as source:

        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path):

    with path.open(
        encoding="utf-8"
    ) as source:

        return json.load(source)


def top_decile_mask(series):

    # Include ties at the cutoff.
    cutoff = float(
        series.quantile(0.90)
    )

    return series >= cutoff - 1e-12


def validate_ids(df):

    return bool(
        len(df) == EXPECTED_GRID_COUNT
        and "grid_id" in df.columns
        and df["grid_id"].notna().all()
        and df["grid_id"].is_unique
    )


def read_table(path):

    df = pd.read_csv(
        path,
        dtype={"grid_id": "string"},
    )

    df["grid_id"] = (
        df["grid_id"]
        .str.strip()
    )

    return df


def safe_max_difference(a, b):

    values_a = np.asarray(
        a,
        dtype=float,
    )

    values_b = np.asarray(
        b,
        dtype=float,
    )

    if (
        len(values_a) != len(values_b)
        or not np.isfinite(values_a).all()
        or not np.isfinite(values_b).all()
    ):

        return float("inf")

    return float(
        np.max(
            np.abs(values_a - values_b)
        )
    )


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 94)
    print(
        "STEP 16D-B - PROVISIONAL "
        "SUITABILITY BASELINE AND GIS VALIDATION"
    )
    print("=" * 94)

    required = [
        STANDARD_CSV,
        STANDARD_GPKG,
        STANDARD_SUMMARY,
        SCREENING_CSV,
        SCREENING_SUMMARY,
        PREVIOUS_K_CSV,
        PREVIOUS_K_SUMMARY,
        EXPERT_SUMMARY,
    ]

    for path in required:

        if not path.is_file():

            raise FileNotFoundError(
                f"Missing input: {path}"
            )

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_hashes = {
        str(path): sha256(path)
        for path in required
    }

    previous_d = read_json(
        STANDARD_SUMMARY
    )

    previous_m = read_json(
        SCREENING_SUMMARY
    )

    previous_k = read_json(
        PREVIOUS_K_SUMMARY
    )

    expert = read_json(
        EXPERT_SUMMARY
    )

    checks = []

    def check(name, passed, detail=""):

        checks.append({
            "check": name,
            "passed": bool(passed),
            "detail": str(detail),
        })

    def checkpoint():

        if not all(
            item["passed"]
            for item in checks
        ):

            pd.DataFrame(checks).to_csv(
                OUT_QA,
                index=False,
            )

            print("\nQA REQUIRES REVIEW")

            for item in checks:

                if not item["passed"]:

                    print(
                        "REVIEW:",
                        item["check"],
                        item["detail"],
                    )

            raise RuntimeError(
                "Step 16D-B validation failed. "
                "See QA report."
            )

    # ========================================================
    # 6. VERIFY PREVIOUS STEPS
    # ========================================================

    print("\n=== VERIFYING RESEARCH INPUTS ===")

    check(
        "Step 16D-A standardized criteria QA passed",
        previous_d.get("qa_status") == "PASS",
    )

    check(
        "Step 16C-M zoning screening QA passed",
        previous_m.get("qa_status") == "PASS",
    )

    check(
        "Step 16C-K robustness QA passed",
        previous_k.get("qa_status") == "PASS",
    )

    check(
        "Expert-review preparation QA passed",
        expert.get("qa_status") == "PASS",
    )

    check(
        "Standardization remains provisional",
        previous_d.get("status")
        == "PROVISIONAL_STANDARDIZATION_ONLY",
    )

    check(
        "Previous comparison remains diagnostic",
        previous_k.get("methodology_status")
        == "DIAGNOSTIC_ONLY_FINAL_CRITERIA_PENDING",
    )

    check(
        "Existing data describes same research grid",
        previous_d.get("grid_cells") == EXPECTED_GRID_COUNT
        and previous_m.get("grid_count") == EXPECTED_GRID_COUNT
        and previous_k.get("grid_count") == EXPECTED_GRID_COUNT,
    )

    checkpoint()

    # ========================================================
    # 7. LOAD STANDARDIZED DATA
    # ========================================================

    standardized = read_table(
        STANDARD_CSV
    )

    previous_scenarios = read_table(
        PREVIOUS_K_CSV
    )

    screening = read_table(
        SCREENING_CSV
    )

    grid = gpd.read_file(
        STANDARD_GPKG,
        layer="provisional_standardized_criteria",
    )

    grid["grid_id"] = (
        grid["grid_id"]
        .astype("string")
        .str.strip()
    )

    for name, df in [
        ("Standardized CSV", standardized),
        ("Previous scenarios", previous_scenarios),
        ("Screening CSV", screening),
        ("Standardized GIS", grid),
    ]:

        check(
            f"{name} has 9240 unique grid IDs",
            validate_ids(df),
        )

    checkpoint()

    grid_ids = set(
        standardized["grid_id"]
    )

    check(
        "All grid ID sets match",
        all(
            set(df["grid_id"]) == grid_ids
            for df in [
                previous_scenarios,
                screening,
                grid,
            ]
        ),
    )

    check(
        "Standardized geometry valid and EPSG:32644",
        grid.crs is not None
        and grid.crs.to_epsg() == EXPECTED_EPSG
        and grid.geometry.notna().all()
        and (~grid.geometry.is_empty).all()
        and grid.geometry.is_valid.all(),
    )

    checkpoint()

    # ========================================================
    # 8. ALIGN ALL INPUTS BY GRID ID
    # ========================================================

    previous_scenarios = (
        previous_scenarios
        .set_index("grid_id")
        .reindex(standardized["grid_id"])
    )

    screening = (
        screening
        .set_index("grid_id")
        .reindex(standardized["grid_id"])
    )

    aligned_geometry = (
        grid
        .set_index("grid_id")
        .geometry
        .reindex(standardized["grid_id"])
    )

    check(
        "All aligned spatial geometries available",
        aligned_geometry.notna().all(),
    )

    checkpoint()

    # ========================================================
    # 9. VALIDATE REQUIRED CRITERION COLUMNS
    # ========================================================

    criterion_columns = []

    for _, primary, alternative in CRITERIA:

        criterion_columns.extend([
            primary,
            alternative,
        ])

    extra_columns = [
        "c3_context_building_pct_500m",
        "c5_elevation_mean_m",
        "cell_area_m2",
        "fraction_of_100m_square",
        "is_partial_boundary_cell",
    ]

    check(
        "Six baseline criterion score columns exist",
        all(
            column in standardized.columns
            for column in criterion_columns
        ),
    )

    check(
        "C3, C5 and boundary attributes available",
        all(
            column in standardized.columns
            for column in extra_columns
        ),
    )

    screening_columns = [
        "classified_pct",
        "unclassified_pct",
        "unmapped_pct",
        "ambiguous_pct",
        "conservation_water_pct",
        "highland_approx_pct",
        "coverage_label",
        "review_flags",
        "review_needed",
        "zoning_legal_status",
    ]

    check(
        "Separate planning-screening attributes available",
        all(
            column in screening.columns
            for column in screening_columns
        ),
    )

    previous_columns = [
        "pctclip_three_core_diagnostic",
        "rank_three_core_diagnostic",
    ]

    check(
        "Previous three-core reference scores available",
        all(
            column in previous_scenarios.columns
            for column in previous_columns
        ),
    )

    checkpoint()

    # ========================================================
    # 10. COMPUTE REFERENCE SCORES
    # ========================================================

    print(
        "\n=== COMPUTING THREE-CRITERION BASELINE ==="
    )

    numeric_scores = standardized[
        criterion_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    check(
        "All contributing criterion scores valid",
        bool(
            np.isfinite(
                numeric_scores.to_numpy(dtype=float)
            ).all()
            and numeric_scores.ge(0).all().all()
            and numeric_scores.le(1).all().all()
        ),
    )

    checkpoint()

    # Equal contribution is an explicit REFERENCE
    # assumption, not an expert AHP judgment.

    primary_columns = [
        criterion[1]
        for criterion in CRITERIA
    ]

    alternative_columns = [
        criterion[2]
        for criterion in CRITERIA
    ]

    baseline_pc = (
        numeric_scores[primary_columns]
        .mean(axis=1)
    )

    baseline_rank = (
        numeric_scores[alternative_columns]
        .mean(axis=1)
    )

    baseline_difference = (
        baseline_pc - baseline_rank
    ).abs()

    # ========================================================
    # 11. CHECK AGAINST STEP 16C-K
    # ========================================================

    prior_pc = pd.to_numeric(
        previous_scenarios[
            "pctclip_three_core_diagnostic"
        ],
        errors="coerce",
    )

    prior_rank = pd.to_numeric(
        previous_scenarios[
            "rank_three_core_diagnostic"
        ],
        errors="coerce",
    )

    pc_difference = safe_max_difference(
        baseline_pc,
        prior_pc,
    )

    rank_difference = safe_max_difference(
        baseline_rank,
        prior_rank,
    )

    check(
        "Percentile-capped baseline reproduces Step 16C-K",
        pc_difference < 1e-10,
        f"Maximum difference: {pc_difference:.12g}",
    )

    check(
        "Percentile-rank baseline reproduces Step 16C-K",
        rank_difference < 1e-10,
        f"Maximum difference: {rank_difference:.12g}",
    )

    check(
        "Both baseline scores valid between zero and one",
        bool(
            np.isfinite(
                baseline_pc.to_numpy()
            ).all()
            and np.isfinite(
                baseline_rank.to_numpy()
            ).all()
            and baseline_pc.between(0, 1).all()
            and baseline_rank.between(0, 1).all()
        ),
    )

    checkpoint()

    # ========================================================
    # 12. RELATIVE DESCRIPTIVE CLASSES
    # ========================================================

    # Classes are based on within-study ranking,
    # not fixed physical or legal suitability levels.
    #
    # Average ranks preserve ties.

    percentile_position = (
        baseline_pc.rank(
            method="average",
            pct=True,
        )
    )

    relative_class = pd.cut(
        percentile_position,
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=CLASS_LABELS,
        include_lowest=True,
    )

    check(
        "Every grid assigned a relative baseline class",
        relative_class.notna().all(),
    )

    checkpoint()

    # ========================================================
    # 13. METHOD-SENSITIVITY ANALYSIS
    # ========================================================

    print(
        "\n=== CHECKING BASELINE ROBUSTNESS ==="
    )

    spearman = float(
        baseline_pc.corr(
            baseline_rank,
            method="spearman",
        )
    )

    pearson = float(
        baseline_pc.corr(
            baseline_rank,
            method="pearson",
        )
    )

    top_pc = top_decile_mask(
        baseline_pc
    )

    top_rank = top_decile_mask(
        baseline_rank
    )

    top_intersection = int(
        (top_pc & top_rank).sum()
    )

    top_union = int(
        (top_pc | top_rank).sum()
    )

    top_disagreement = int(
        (top_pc ^ top_rank).sum()
    )

    top_jaccard = (
        float(top_intersection / top_union)
        if top_union
        else float("nan")
    )

    comparison = pd.DataFrame([
        {
            "comparison":
                "PERCENTILE_CAPPED_VS_RANK",

            "criterion_set":
                "C1_C2_C4",

            "contribution_assumption":
                "EQUAL_REFERENCE_ONLY",

            "mean_absolute_score_difference":
                float(baseline_difference.mean()),

            "median_absolute_score_difference":
                float(baseline_difference.median()),

            "p95_absolute_score_difference":
                float(
                    baseline_difference.quantile(0.95)
                ),

            "max_absolute_score_difference":
                float(baseline_difference.max()),

            "pearson_correlation": pearson,

            "spearman_correlation": spearman,

            "top_decile_pc_count":
                int(top_pc.sum()),

            "top_decile_rank_count":
                int(top_rank.sum()),

            "top_decile_jaccard": top_jaccard,

            "top_decile_disagreement_cells":
                top_disagreement,

            "top_decile_disagreement_pct":
                float(
                    100 * top_disagreement
                    / EXPECTED_GRID_COUNT
                ),

            "cells_score_difference_ge_0_20":
                int(
                    (
                        baseline_difference
                        >= SCORE_DIFFERENCE_REVIEW
                    ).sum()
                ),

            "status": "PROVISIONAL_DIAGNOSTIC",
        }
    ])

    check(
        "Sensitivity comparison metrics finite",
        bool(
            np.isfinite(
                comparison[
                    [
                        "mean_absolute_score_difference",
                        "max_absolute_score_difference",
                        "pearson_correlation",
                        "spearman_correlation",
                        "top_decile_jaccard",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
    )

    checkpoint()

    # ========================================================
    # 14. CREATE SPATIAL BASELINE OUTPUT
    # ========================================================

    print(
        "\n=== PREPARING SPATIAL BASELINE DATA ==="
    )

    output = gpd.GeoDataFrame(
        {
            "grid_id":
                standardized["grid_id"].to_numpy()
        },
        geometry=gpd.GeoSeries(
            aligned_geometry.to_numpy(),
            crs=grid.crs,
        ),
        crs=grid.crs,
    )

    output["baseline_pc"] = (
        baseline_pc.to_numpy(dtype=float)
    )

    output["baseline_rank"] = (
        baseline_rank.to_numpy(dtype=float)
    )

    output["baseline_difference"] = (
        baseline_difference.to_numpy(dtype=float)
    )

    output["relative_class"] = (
        relative_class.astype(str).to_numpy()
    )

    output["top_decile_pc"] = (
        top_pc.to_numpy(dtype=bool)
    )

    output["top_decile_rank"] = (
        top_rank.to_numpy(dtype=bool)
    )

    # Retain each contributing criterion for XAI-like
    # transparent breakdown. These are criterion
    # contributions, NOT model SHAP explanations.

    for criterion, pc_column, rank_column in CRITERIA:

        output[
            f"{criterion.lower()}_pc"
        ] = (
            numeric_scores[pc_column]
            .to_numpy(dtype=float)
        )

        output[
            f"{criterion.lower()}_rank"
        ] = (
            numeric_scores[rank_column]
            .to_numpy(dtype=float)
        )

        output[
            f"{criterion.lower()}_reference_contribution"
        ] = (
            numeric_scores[pc_column]
            .to_numpy(dtype=float) / 3.0
        )

    output["c3_raw_building_context"] = (
        pd.to_numeric(
            standardized[
                "c3_context_building_pct_500m"
            ],
            errors="coerce",
        ).to_numpy(dtype=float)
    )

    output["c5_raw_elevation_m"] = (
        pd.to_numeric(
            standardized[
                "c5_elevation_mean_m"
            ],
            errors="coerce",
        ).to_numpy(dtype=float)
    )

    output["cell_area_m2"] = (
        pd.to_numeric(
            standardized["cell_area_m2"],
            errors="coerce",
        ).to_numpy(dtype=float)
    )

    output["cell_fraction"] = (
        pd.to_numeric(
            standardized["fraction_of_100m_square"],
            errors="coerce",
        ).to_numpy(dtype=float)
    )

    output["partial_boundary_cell"] = (
        standardized[
            "is_partial_boundary_cell"
        ].astype(bool).to_numpy()
    )

    output["small_boundary_fragment"] = (
        output["cell_area_m2"]
        < HALF_CELL_AREA_M2
    )

    # Attach the planning screening information.
    # None of these columns enters the baseline formula.

    for column in screening_columns:

        output[column] = (
            screening[column].to_numpy()
        )

    if "main_zone_zone_name" in screening.columns:

        output["main_zone_zone_name"] = (
            screening[
                "main_zone_zone_name"
            ].fillna(
                "NOT_CLASSIFIED"
            ).astype(str).to_numpy()
        )

    output["score_type"] = (
        "PROVISIONAL_EQUAL_CONTRIBUTION"
    )

    output["legal_permission"] = (
        "NOT_ASSESSED"
    )

    # ========================================================
    # 15. FINAL GRID QUALITY ASSURANCE
    # ========================================================

    check(
        "Output contains 9240 unique grid cells",
        validate_ids(output),
    )

    check(
        "Output geometries valid",
        bool(
            output.geometry.notna().all()
            and (~output.geometry.is_empty).all()
            and output.geometry.is_valid.all()
        ),
    )

    actual_areas = (
        output.geometry.area
        .to_numpy(dtype=float)
    )

    recorded_areas = (
        output["cell_area_m2"]
        .to_numpy(dtype=float)
    )

    max_area_difference = (
        safe_max_difference(
            actual_areas,
            recorded_areas,
        )
    )

    check(
        "Recorded cell areas match actual polygons",
        max_area_difference <= 0.01,
        f"Maximum error: {max_area_difference:.8f} m2",
    )

    check(
        "Baseline equals the sum of reference contributions",
        np.allclose(
            output["baseline_pc"],
            (
                output["c1_reference_contribution"]
                + output["c2_reference_contribution"]
                + output["c4_reference_contribution"]
            ),
            atol=1e-12,
        ),
    )

    check(
        "C3 and C5 excluded from baseline formula",
        (
            previous_d.get("c3")
            == "TWO_UNSELECTED_ALTERNATIVE_DIRECTIONS"
            and previous_d.get("c5")
            == "RAW_ONLY_NO_CITYWIDE_SUITABILITY_DIRECTION_ESTABLISHED"
        ),
        "Raw C3 and C5 retained for review only",
    )

    check(
        "Zoning retains unverified legal status",
        (
            output["zoning_legal_status"]
            .astype(str)
            .eq("UNVERIFIED")
            .all()
        ),
    )

    check(
        "No development permission classification produced",
        output["legal_permission"]
        .eq("NOT_ASSESSED")
        .all(),
    )

    check(
        "All original inputs unchanged",
        all(
            sha256(path)
            == source_hashes[str(path)]
            for path in required
        ),
    )

    checkpoint()

    # ========================================================
    # 16. SUMMARIZE RELATIVE CLASSES
    # ========================================================

    class_summary = (
        output.groupby(
            "relative_class",
            dropna=False,
        )
        .agg(
            grid_cells=("grid_id", "size"),

            area_ha=(
                "cell_area_m2",
                lambda values:
                    float(values.sum()) / 10000,
            ),

            average_reference_score=(
                "baseline_pc",
                "mean",
            ),

            planning_review_cells=(
                "review_needed",
                "sum",
            ),

            small_fragments=(
                "small_boundary_fragment",
                "sum",
            ),
        )
        .reset_index()
    )

    class_summary["cell_pct"] = (
        100
        * class_summary["grid_cells"]
        / EXPECTED_GRID_COUNT
    )

    check(
        "Relative classes summarize all cells",
        int(class_summary["grid_cells"].sum())
        == EXPECTED_GRID_COUNT,
    )

    check(
        "Relative classes preserve complete KMC area",
        abs(
            class_summary["area_ha"].sum()
            - previous_d["grid_area_ha"]
        ) < 0.05,
    )

    checkpoint()

    # ========================================================
    # 17. VALIDATE GIS ROUND-TRIP
    # ========================================================

    staging_path = OUT_GPKG.with_name(
        OUT_GPKG.stem + "_staging.gpkg"
    )

    if staging_path.exists():

        staging_path.unlink()

    output.to_file(
        staging_path,
        layer="provisional_suitability_baseline",
        driver="GPKG",
        index=False,
    )

    saved = gpd.read_file(
        staging_path,
        layer="provisional_suitability_baseline",
    )

    check(
        "Saved GIS retains 9240 unique grid IDs",
        validate_ids(saved),
    )

    check(
        "Saved GIS retains baseline values",
        safe_max_difference(
            saved["baseline_pc"],
            output["baseline_pc"],
        ) < 1e-9,
    )

    check(
        "Saved GIS retains valid geometries",
        bool(
            saved.geometry.notna().all()
            and saved.geometry.is_valid.all()
        ),
    )

    checkpoint()

    del saved

    # ========================================================
    # 18. SAVE FINAL OUTPUTS
    # ========================================================

    os.replace(
        staging_path,
        OUT_GPKG,
    )

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

    comparison.to_csv(
        OUT_SENSITIVITY,
        index=False,
    )

    qa = pd.DataFrame(checks)

    qa.to_csv(
        OUT_QA,
        index=False,
    )

    # ========================================================
    # 19. SUMMARY JSON
    # ========================================================

    summary = {
        "step": "16D-B",

        "qa_status": "PASS",

        "methodology_status":
            "PROVISIONAL_REFERENCE_BASELINE_ONLY",

        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "grid_count": EXPECTED_GRID_COUNT,

        "crs": f"EPSG:{EXPECTED_EPSG}",

        "criterion_set": [
            "C1",
            "C2",
            "C4",
        ],

        "reference_assumption": {
            "C1": 1 / 3,
            "C2": 1 / 3,
            "C4": 1 / 3,
        },

        "assumption_status":
            "ILLUSTRATIVE_NOT_AHP_WEIGHTS",

        "reference_formula":
            "(C1 + C2 + C4) / 3",

        "primary_method":
            "PERCENTILE_CAPPED_P05_P95",

        "sensitivity_method":
            "PERCENTILE_RANK",

        "raw_c3_retained": True,

        "raw_c5_retained": True,

        "expert_ahp_weights_calculated": False,

        "final_suitability_model_calculated": False,

        "planning_exclusions_applied": False,

        "baseline_min": float(
            baseline_pc.min()
        ),

        "baseline_mean": float(
            baseline_pc.mean()
        ),

        "baseline_max": float(
            baseline_pc.max()
        ),

        "baseline_spearman_between_methods":
            spearman,

        "top_decile_method_jaccard":
            top_jaccard,

        "partial_boundary_cells": int(
            output["partial_boundary_cell"].sum()
        ),

        "small_boundary_fragments": int(
            output["small_boundary_fragment"].sum()
        ),

        "planning_review_cells": int(
            output["review_needed"].sum()
        ),

        "qa_checks": len(qa),

        "qa_passed": int(
            qa["passed"].sum()
        ),

        "source_sha256": {
            Path(path).name: digest
            for path, digest
            in source_hashes.items()
        },

        "limitations": [
            (
                "Equal criterion contributions are a "
                "reference assumption, not AHP weights."
            ),
            (
                "C3 and C5 remain outside the "
                "baseline pending justification."
            ),
            (
                "Relative classes are study-specific "
                "ranking categories, not official "
                "planning suitability thresholds."
            ),
            (
                "Partial boundary cells include very "
                "small polygon fragments that are not "
                "independently verified buildable land."
            ),
            (
                "Planning screening is attached as "
                "separate context and never alters "
                "the reference baseline score."
            ),
            (
                "Provisional UDA layers do not establish "
                "legal development permission."
            ),
            (
                "Further criterion review, expert AHP, "
                "sensitivity testing and independent "
                "validation remain necessary."
            ),
        ],

        "outputs": [
            path.name
            for path in [
                OUT_CSV,
                OUT_GPKG,
                OUT_CLASSES,
                OUT_SENSITIVITY,
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
    # 20. TERMINAL REPORT
    # ========================================================

    print("\n=== PROVISIONAL BASELINE ===")

    print(
        "Grid cells:",
        len(output)
    )

    print(
        "Baseline minimum:",
        round(float(baseline_pc.min()), 4)
    )

    print(
        "Baseline mean:",
        round(float(baseline_pc.mean()), 4)
    )

    print(
        "Baseline maximum:",
        round(float(baseline_pc.max()), 4)
    )

    print(
        "Spearman correlation (methods):",
        round(spearman, 4)
    )

    print(
        "Top-decile Jaccard:",
        round(top_jaccard, 4)
    )

    print(
        "Small boundary fragments:",
        summary["small_boundary_fragments"]
    )

    print(
        "Planning-review cells:",
        summary["planning_review_cells"]
    )

    print("\n=== RELATIVE CLASS SUMMARY ===")

    print(
        class_summary[
            [
                "relative_class",
                "grid_cells",
                "area_ha",
                "average_reference_score",
                "planning_review_cells",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\n=== QA RESULTS ===")

    for item in checks:

        result = (
            "PASS"
            if item["passed"]
            else "FAIL"
        )

        print(
            f"{result}: {item['check']}"
        )

    print(
        f"\nOverall QA: PASS "
        f"({int(qa['passed'].sum())}/{len(qa)})"
    )

    print("\n=== GENERATED REPORTS ===")

    for path in [
        OUT_CSV,
        OUT_GPKG,
        OUT_CLASSES,
        OUT_SENSITIVITY,
        OUT_QA,
        OUT_SUMMARY,
    ]:

        print(
            path.relative_to(ROOT)
        )

    print(
        "\nSTATUS: PROVISIONAL REFERENCE BASELINE"
    )

    print(
        "No expert AHP weights, final suitability "
        "model or legal permissions calculated."
    )

    print("=" * 94)


# ============================================================
# 21. EXECUTE
# ============================================================

if __name__ == "__main__":
    main()
