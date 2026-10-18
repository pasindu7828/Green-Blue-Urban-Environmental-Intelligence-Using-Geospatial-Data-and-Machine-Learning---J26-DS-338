
"""
J26-DS-338 | SECTION 03
STEP 16E-02 - EXPERT-AHP WEIGHTED DEVELOPMENT SUITABILITY

Inputs
------
16c_m: Validated, re-standardized C1/C2/C4 GIS criteria
16d_e: Researcher-transcribed engineer AHP judgments

Weights
-------
C1: Major Road Proximity = 0.20
C2: Local Road Density = 0.60
C4: Terrain Slope = 0.20

Method
------
GIS-MCDA weighted linear combination:
    S = 0.20*C1 + 0.60*C2 + 0.20*C4

Safety
------
- Read latest 16c_m data, NEVER outdated 03d scores.
- Re-verify matrix, weights, responses and consistency.
- Preserve all 9,240 grids and spatial geometry.
- Never overwrite existing final outputs.
- Never overwrite provisional results or Section 04.
- Do not invent suitability classes or flood-risk scores.
- Sensitivity analysis remains a separate next stage.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT / "data/processed/development_suitability"
)

REPORT = ROOT / "outputs/reports"

STD_CSV = (
    BASE / "standardized_criteria"
    / "16c_m_final_criteria_standardized_100m.csv"
)

STD_GPKG = (
    BASE / "standardized_criteria"
    / "16c_m_final_criteria_standardized_100m.gpkg"
)

STD_LAYER = "final_criteria_standardized_100m"

STD_QA = (
    REPORT
    / "16c_m_final_criteria_standardization_qa.json"
)

AHP_RESPONSES = (
    REPORT / "16d_e_expert_ahp_completed_responses.csv"
)

AHP_MATRIX = (
    REPORT / "16d_e_expert_ahp_pairwise_matrix.csv"
)

AHP_WEIGHTS = (
    REPORT / "16d_e_expert_ahp_weights.csv"
)

AHP_QA = (
    REPORT / "16d_e_expert_ahp_consistency_qa.json"
)

OUT_DIR = BASE / "final_expert_ahp"

OUT_CSV = (
    OUT_DIR
    / "16e_02_expert_ahp_weighted_suitability_100m.csv"
)

OUT_GPKG = (
    OUT_DIR
    / "16e_02_expert_ahp_weighted_suitability_100m.gpkg"
)

OUT_LAYER = "expert_ahp_suitability_100m"

OUT_QA = (
    REPORT / "16e_02_expert_ahp_weighted_qa.json"
)

CRITERIA = ["C1", "C2", "C4"]

SCORES = {
    "C1": "c1_road_access_score",
    "C2": "c2_road_context_score",
    "C4": "c4_slope_score",
}

RAW_FIELDS = {
    "C1": "c1_major_road_distance_m",
    "C2": "c2_road_density_500m_km_per_km2",
    "C4": "c4_slope_mean_deg",
}

WEIGHTS_EXPECTED = np.array([0.2, 0.6, 0.2])

EXPECTED_MATRIX = np.array([
    [1.0, 1.0 / 3.0, 1.0],
    [3.0, 1.0, 3.0],
    [1.0, 1.0 / 3.0, 1.0],
])

# The three latest expert judgments.
# Used to verify the stored questionnaire.

EXPECTED_ANSWERS = {
    ("C1", "C2"): ("C2", 3),
    ("C1", "C4"): ("EQUAL", 1),
    ("C2", "C4"): ("C2", 3),
}

EXPECTED_REASONS = {
    ("C1", "C2"):
        "Local road density improves accessibility "
        "to land and supports urban development.",

    ("C1", "C4"):
        "Major road access supports connectivity, "
        "while terrain slope affects construction "
        "feasibility and costs.",

    ("C2", "C4"):
        "Local road density improves access, "
        "while slope affects construction feasibility. "
        "Flood risk also depends on drainage and elevation.",
}

METADATA = [
    "grid_id",
    "kmc_overlap_area_m2",
    "kmc_overlap_fraction",
    "boundary_cell",
    "coverage_group",
    "terrain_fallback_applied",
    "terrain_extraction_method",
    "terrain_fallback_distance_m",
]

FINAL_SCORE = "development_suitability_score"


def require(condition, label):
    passed = bool(condition)

    print(
        f"[{'PASS' if passed else 'FAIL'}] {label}"
    )

    if not passed:
        raise ValueError(
            f"QA FAILED: {label}"
        )


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def numeric(df, column):
    if column not in df.columns:
        raise KeyError(
            f"Required field not found: {column}"
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def validate_grid(df, label):
    require(
        len(df) == 9240,
        f"{label}: 9,240 rows",
    )

    require(
        "grid_id" in df.columns
        and df.grid_id.notna().all()
        and df.grid_id.is_unique,
        f"{label}: unique grid IDs",
    )


def compact_text(value):
    # Compare expert justifications despite accidental
    # whitespace formatting during transcription.
    # Actual numerical judgments may never be changed.
    return "".join(str(value).split())


def score_summary(values):
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "standard_deviation": float(
            np.std(values, ddof=0)
        ),
    }


def main():

    print("=" * 78)
    print(
        "SECTION 03 | FINAL EXPERT-AHP "
        "WEIGHTED DEVELOPMENT SUITABILITY"
    )
    print("=" * 78)

    inputs = [
        STD_CSV,
        STD_GPKG,
        STD_QA,
        AHP_RESPONSES,
        AHP_MATRIX,
        AHP_WEIGHTS,
        AHP_QA,
    ]

    outputs = [
        OUT_CSV,
        OUT_GPKG,
        OUT_QA,
    ]

    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    for path in outputs:
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {path}"
            )

    original_hashes = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in inputs
    }

    # ======================================================
    # 1. VERIFY FINAL STANDARDIZATION
    # ======================================================

    print("\n[1] VERIFY STAGE 16C-M")

    std_qa = json.loads(
        STD_QA.read_text(encoding="utf-8")
    )

    require(
        std_qa.get("status")
        == (
            "PASS_FINAL_SELECTED_CRITERIA_STANDARDIZED_"
            "AHP_WEIGHTING_PENDING"
        ),
        "16C-M standardization QA passed",
    )

    require(
        std_qa.get("grid_cells") == 9240,
        "16C-M grid count",
    )

    require(
        std_qa.get("crs") == "EPSG:32644",
        "16C-M coordinate system",
    )

    require(
        std_qa.get("selected_criteria") == CRITERIA,
        "Selected criteria C1/C2/C4",
    )

    require(
        std_qa.get("historical_03d_method_reproduced")
        is True,
        "Original normalization method reproduced",
    )

    require(
        std_qa.get("ahp_weights_applied") is False,
        "Standardization has no premature AHP weighting",
    )

    require(
        std_qa.get("final_suitability_calculated") is False,
        "Standardization has no premature final score",
    )

    recorded_outputs = std_qa.get("outputs", {})

    require(
        recorded_outputs.get("csv_sha256")
        == sha256(STD_CSV),
        "Standardized CSV hash matches QA",
    )

    require(
        recorded_outputs.get("gpkg_sha256")
        == sha256(STD_GPKG),
        "Standardized GIS hash matches QA",
    )

    # ======================================================
    # 2. VERIFY EXPERT AHP
    # ======================================================

    print("\n[2] VERIFY EXPERT AHP EVIDENCE")

    ahp_qa = json.loads(
        AHP_QA.read_text(encoding="utf-8")
    )

    require(
        ahp_qa.get("status")
        == (
            "PASS_USER_REPORTED_EXPERT_AHP_"
            "NUMERICAL_QA_FINAL_SUITABILITY_PENDING"
        ),
        "Expert-AHP numerical QA passed",
    )

    require(
        ahp_qa.get("scenario")
        == "BASE_3_CRITERIA",
        "Three-criterion scenario verified",
    )

    require(
        ahp_qa.get("consultation_date")
        == "2026-10-17"
        and ahp_qa.get("expert_role") == "Engineer",
        "Expert consultation metadata verified",
    )

    require(
        ahp_qa.get("matrix_criterion_order") == CRITERIA,
        "AHP matrix criterion order verified",
    )

    responses = pd.read_csv(AHP_RESPONSES)

    base_responses = responses.loc[
        responses["scenario"] == "BASE_3_CRITERIA"
    ].copy()

    require(
        len(base_responses) == 3,
        "Exactly three baseline expert responses",
    )

    observed = {}
    spacing_warnings = []

    for row in base_responses.itertuples(index=False):

        pair = (
            str(row.criterion_a),
            str(row.criterion_b),
        )

        require(
            pair in EXPECTED_ANSWERS,
            f"Known expert pair {pair}",
        )

        observation = (
            str(row.higher_importance),
            int(row.importance_strength_1_to_9),
        )

        observed[pair] = observation

        require(
            str(row.reviewer_role) == "Engineer"
            and str(row.review_date) == "2026-10-17",
            f"Expert metadata for {pair}",
        )

        require(
            pd.notna(row.justification),
            f"Expert justification exists for {pair}",
        )

        require(
            compact_text(row.justification)
            == compact_text(EXPECTED_REASONS[pair]),
            f"Latest expert justification agrees for {pair}",
        )

        if str(row.justification) != EXPECTED_REASONS[pair]:
            spacing_warnings.append(
                f"{pair}: whitespace transcription differs"
            )

    require(
        observed == EXPECTED_ANSWERS,
        "All three expert numerical judgments verified",
    )

    conditional = responses.loc[
        responses["scenario"]
        == "CONDITIONAL_4_WITH_C3"
    ]

    require(
        len(conditional) == 6
        and conditional[
            [
                "higher_importance",
                "importance_strength_1_to_9",
                "justification",
            ]
        ].isna().all().all(),
        "Conditional four-criterion answers remain blank",
    )

    weight_table = pd.read_csv(AHP_WEIGHTS)

    require(
        len(weight_table) == 3
        and weight_table["criterion"].is_unique
        and set(weight_table["criterion"]) == set(CRITERIA),
        "Exactly three uniquely named criterion weights",
    )

    weight_table = (
        weight_table
        .set_index("criterion")
        .loc[CRITERIA]
    )

    weights = numeric(
        weight_table,
        "expert_ahp_weight",
    )

    require(
        np.isfinite(weights).all()
        and np.allclose(
            weights,
            WEIGHTS_EXPECTED,
            atol=1e-12,
            rtol=0,
        )
        and np.isclose(
            weights.sum(),
            1.0,
            atol=1e-12,
        ),
        "Expert weights = 0.20 / 0.60 / 0.20",
    )

    require(
        all(
            np.isclose(
                float(ahp_qa["weights"][criterion]),
                weights[i],
                atol=1e-12,
            )
            for i, criterion in enumerate(CRITERIA)
        ),
        "AHP weight file agrees with QA record",
    )

    matrix_df = pd.read_csv(
        AHP_MATRIX,
        index_col=0,
    )

    require(
        set(matrix_df.index) == set(CRITERIA)
        and set(matrix_df.columns) == set(CRITERIA),
        "AHP matrix labels valid",
    )

    matrix = (
        matrix_df.loc[CRITERIA, CRITERIA]
        .to_numpy(dtype=float)
    )

    require(
        np.allclose(
            matrix,
            EXPECTED_MATRIX,
            atol=1e-12,
            rtol=0,
        ),
        "AHP pairwise matrix matches judgments",
    )

    require(
        np.allclose(
            matrix * matrix.T,
            1.0,
            atol=1e-12,
            rtol=0,
        ),
        "AHP matrix is reciprocal",
    )

    require(
        np.allclose(
            matrix @ weights,
            3.0 * weights,
            atol=1e-12,
            rtol=0,
        ),
        "Expert eigenvector weights reproduced",
    )

    eigenvalues = np.linalg.eigvals(matrix)

    lambda_max = float(
        np.max(eigenvalues.real)
    )

    ci = max(
        0.0,
        (lambda_max - 3.0) / 2.0,
    )

    cr = ci / 0.58

    require(
        cr < 0.10
        and np.isclose(
            cr,
            float(ahp_qa["consistency_ratio"]),
            atol=1e-10,
        ),
        "Expert consistency ratio verified",
    )

    print("C1 weight:", f"{weights[0]:.6f}")
    print("C2 weight:", f"{weights[1]:.6f}")
    print("C4 weight:", f"{weights[2]:.6f}")
    print("Consistency Ratio:", f"{cr:.6f}")

    if spacing_warnings:
        print(
            "[NOTE] Some previously transcribed "
            "justifications have whitespace differences."
        )
        print(
            "Correct expert wording will be preserved "
            "in this QA report."
        )

    # ======================================================
    # 3. LOAD FINAL STANDARDIZED GRID DATA
    # ======================================================

    print("\n[3] LOAD FINAL STANDARDIZED CRITERIA")

    standardized = pd.read_csv(STD_CSV)

    validate_grid(
        standardized,
        "16C-M standardized CSV",
    )

    require(
        standardized["selected_criteria_set"]
        .eq("C1_C2_C4").all(),
        "All rows use the selected three criteria",
    )

    require(
        standardized["standardization_method"]
        .eq("P05_P95_CLIPPED_MINMAX").all(),
        "Standardization method preserved",
    )

    for column in (
        METADATA
        + list(RAW_FIELDS.values())
        + list(SCORES.values())
    ):
        require(
            column in standardized.columns,
            f"Required standardized field: {column}",
        )

    X = np.column_stack([
        numeric(standardized, SCORES[c])
        for c in CRITERIA
    ])

    require(
        X.shape == (9240, 3),
        "Standardized matrix 9,240 x 3",
    )

    require(
        np.isfinite(X).all(),
        "All standardized scores finite",
    )

    require(
        np.all(X >= 0)
        and np.all(X <= 1),
        "All standardized scores within [0,1]",
    )

    # ======================================================
    # 4. CALCULATE AHP WEIGHTED SUITABILITY
    # ======================================================

    print("\n[4] CALCULATE FINAL SUITABILITY SCORES")

    contributions = X * weights[None, :]

    suitability = contributions.sum(axis=1)

    require(
        suitability.shape == (9240,)
        and np.isfinite(suitability).all(),
        "9,240 finite suitability scores",
    )

    require(
        np.all(suitability >= -1e-12)
        and np.all(suitability <= 1 + 1e-12),
        "Suitability scores bounded within [0,1]",
    )

    independently_calculated = (
        0.20 * X[:, 0]
        + 0.60 * X[:, 1]
        + 0.20 * X[:, 2]
    )

    require(
        np.allclose(
            suitability,
            independently_calculated,
            atol=1e-12,
            rtol=0,
        ),
        "Weighted suitability formula reproduced",
    )

    # ======================================================
    # 5. BUILD FINAL OUTPUT TABLE
    # ======================================================

    print("\n[5] BUILD OUTPUT DATASET")

    selected_columns = (
        METADATA
        + list(RAW_FIELDS.values())
        + list(SCORES.values())
    )

    result = standardized[
        selected_columns
    ].copy()

    for i, criterion in enumerate(CRITERIA):

        prefix = criterion.lower()

        result[
            f"{prefix}_expert_ahp_weight"
        ] = float(weights[i])

        result[
            f"{prefix}_weighted_contribution"
        ] = contributions[:, i]

    result[FINAL_SCORE] = suitability

    result["suitability_method"] = (
        "GIS_MCDA_EXPERT_AHP_WEIGHTED_LINEAR_COMBINATION"
    )

    result["suitability_criteria_set"] = (
        "C1_C2_C4"
    )

    result["suitability_status"] = (
        "EXPERT_AHP_WEIGHTED_SENSITIVITY_PENDING"
    )

    result["suitability_class"] = (
        "NOT_YET_CLASSIFIED"
    )

    require(
        len(result) == 9240
        and result.grid_id.is_unique,
        "Output has 9,240 unique grids",
    )

    require(
        not any(
            column.startswith("c3_")
            for column in result.columns
        ),
        "Excluded C3 is not included in weighted model",
    )

    require(
        result["suitability_class"]
        .eq("NOT_YET_CLASSIFIED").all(),
        "No unapproved classification thresholds used",
    )

    # ======================================================
    # 6. VALIDATE GIS SOURCE
    # ======================================================

    print("\n[6] VALIDATE FINAL GIS GEOMETRY")

    geo = gpd.read_file(
        STD_GPKG,
        layer=STD_LAYER,
    )

    validate_grid(
        geo,
        "16C-M standardized GIS",
    )

    require(
        geo.crs is not None
        and geo.crs.to_epsg() == 32644,
        "GIS CRS is EPSG:32644",
    )

    require(
        geo.geometry.notna().all()
        and not geo.geometry.is_empty.any()
        and geo.geometry.is_valid.all(),
        "All GIS geometries are valid",
    )

    require(
        set(geo.grid_id) == set(result.grid_id),
        "GIS and CSV grid IDs match exactly",
    )

    geo_aligned = (
        geo.set_index("grid_id")
        .loc[result.grid_id]
    )

    for criterion in CRITERIA:

        column = SCORES[criterion]

        require(
            column in geo_aligned.columns
            and np.allclose(
                numeric(geo_aligned, column),
                numeric(result, column),
                atol=1e-12,
                rtol=0,
            ),
            f"{criterion}: CSV and GIS scores agree",
        )

    source_area = (
        geo_aligned.geometry.area
        .to_numpy(dtype=float)
    )

    csv_area = numeric(
        result,
        "kmc_overlap_area_m2",
    )

    require(
        np.allclose(
            source_area,
            csv_area,
            atol=1.0,
            rtol=0,
        ),
        "GIS geometry agrees with clipped grid area",
    )

    study_area = float(
        source_area.sum() / 1e6
    )

    require(
        np.isclose(
            study_area,
            88.82921581108494,
            atol=0.001,
            rtol=0,
        ),
        "GIS study area verified",
    )

    output_geo = geo[
        ["grid_id", "geometry"]
    ].merge(
        result,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    validate_grid(
        output_geo,
        "Final expert-AHP GIS",
    )

    # ======================================================
    # 7. EXPORT AND VERIFY
    # ======================================================

    print("\n[7] EXPORT NEW SUITABILITY OUTPUTS")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix="16e_02_build_",
        dir=OUT_DIR,
    ) as temp_folder:

        temp_folder = Path(temp_folder)

        temp_csv = temp_folder / OUT_CSV.name
        temp_gpkg = temp_folder / OUT_GPKG.name

        result.to_csv(
            temp_csv,
            index=False,
        )

        output_geo.to_file(
            temp_gpkg,
            layer=OUT_LAYER,
            driver="GPKG",
            engine="pyogrio",
        )

        exported_csv = pd.read_csv(temp_csv)

        exported_geo = gpd.read_file(
            temp_gpkg,
            layer=OUT_LAYER,
        )

        validate_grid(
            exported_csv,
            "Exported suitability CSV",
        )

        validate_grid(
            exported_geo,
            "Exported suitability GIS",
        )

        require(
            exported_geo.crs is not None
            and exported_geo.crs.to_epsg() == 32644,
            "Exported GIS coordinate system",
        )

        require(
            np.allclose(
                numeric(exported_csv, FINAL_SCORE),
                suitability,
                atol=1e-12,
                rtol=0,
            ),
            "Exported CSV preserves suitability scores",
        )

        expected_by_id = (
            result.set_index("grid_id")[FINAL_SCORE]
        )

        exported_by_id = (
            exported_geo.set_index("grid_id")[FINAL_SCORE]
        ).loc[expected_by_id.index]

        require(
            np.allclose(
                expected_by_id.to_numpy(dtype=float),
                exported_by_id.to_numpy(dtype=float),
                atol=1e-12,
                rtol=0,
            ),
            "Exported GIS preserves suitability scores",
        )

        for path in outputs:
            if path.exists():
                raise FileExistsError(
                    f"Unexpected existing output: {path}"
                )

        temp_csv.replace(OUT_CSV)
        temp_gpkg.replace(OUT_GPKG)

    # ======================================================
    # 8. VERIFY SOURCE INTEGRITY
    # ======================================================

    print("\n[8] VERIFY INPUTS UNCHANGED")

    for path in inputs:

        relative = (
            path.relative_to(ROOT).as_posix()
        )

        require(
            sha256(path) == original_hashes[relative],
            f"Unmodified source: {path.name}",
        )

    # ======================================================
    # 9. WRITE QA REPORT
    # ======================================================

    print("\n[9] SAVE FINAL WEIGHTING QA")

    qa = {
        "section": "03_development_suitability",
        "stage": "16e_02",
        "status": (
            "PASS_EXPERT_AHP_WEIGHTED_SUITABILITY_"
            "SENSITIVITY_PENDING"
        ),
        "grid_cells": 9240,
        "crs": "EPSG:32644",
        "study_area_km2": study_area,
        "method": (
            "GIS_MCDA_EXPERT_AHP_"
            "WEIGHTED_LINEAR_COMBINATION"
        ),
        "formula": (
            "S=0.20*C1+0.60*C2+0.20*C4"
        ),
        "criteria": {
            criterion: {
                "raw_field": RAW_FIELDS[criterion],
                "score_field": SCORES[criterion],
                "expert_weight": float(weights[i]),
            }
            for i, criterion in enumerate(CRITERIA)
        },
        "expert_consultation": {
            "date": "2026-10-17",
            "role": "Engineer",
            "provenance": (
                "Researcher-transcribed expert judgments"
            ),
            "name_supplied": False,
            "signed_questionnaire_available": False,
            "pairwise_judgments": [
                {
                    "criterion_a": a,
                    "criterion_b": b,
                    "higher_importance": h,
                    "strength": strength,
                    "justification_as_reported": (
                        EXPECTED_REASONS[(a, b)]
                    ),
                }
                for (a, b), (h, strength)
                in EXPECTED_ANSWERS.items()
            ],
            "original_response_spacing_notes": (
                spacing_warnings
            ),
            "consistency_ratio": float(cr),
        },
        "source_standardization": (
            "16c_m_final_criteria_standardized_100m"
        ),
        "normalization_method": (
            "Empirical P05/P95 clipped min-max"
        ),
        "suitability_score_distribution": (
            score_summary(suitability)
        ),
        "final_suitability_classes_assigned": False,
        "sensitivity_analysis_completed": False,
        "flood_risk_explicitly_modelled": False,
        "provisional_results_overwritten": False,
        "section04_modified": False,
        "input_sha256": original_hashes,
        "outputs": {
            "csv": (
                OUT_CSV.relative_to(ROOT).as_posix()
            ),
            "gpkg": (
                OUT_GPKG.relative_to(ROOT).as_posix()
            ),
            "csv_sha256": sha256(OUT_CSV),
            "gpkg_sha256": sha256(OUT_GPKG),
        },
    }

    OUT_QA.write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )

    # ======================================================
    # 10. TERMINAL SUMMARY
    # ======================================================

    print("\nFINAL SUITABILITY DISTRIBUTION")

    for key, value in score_summary(suitability).items():
        if key == "count":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")

    print("\n" + "=" * 78)

    print(
        "PASS_EXPERT_AHP_WEIGHTED_SUITABILITY_"
        "SENSITIVITY_PENDING"
    )

    print("Grid cells:", len(result))
    print("Weights: C1=0.20, C2=0.60, C4=0.20")
    print(f"AHP Consistency Ratio: {cr:.6f}")
    print("Suitability classification: PENDING")
    print("Sensitivity analysis: PENDING")
    print("Provisional results overwritten: NO")
    print("Section 04 modified: NO")

    print("\nCSV:", OUT_CSV)
    print("GPKG:", OUT_GPKG)
    print("QA:", OUT_QA)

    print("=" * 78)


if __name__ == "__main__":
    main()
