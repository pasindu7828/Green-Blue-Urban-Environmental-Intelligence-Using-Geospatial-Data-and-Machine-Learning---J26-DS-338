
"""
J26-DS-338 | Section 03
STEP 16G-01 - FINAL EXPERT-AHP SUITABILITY RELEASE

Validates:
    - Genuine researcher-reported AHP consultation evidence
    - Final C1/C2/C4 standardized and weighted scores
    - 9,240 cells and EPSG:32644 geometry
    - Published CSV/GIS score consistency
    - All 13 post-AHP sensitivity scenarios
    - Rank stability and top-group retention
    - Source file hashes and provenance

Final deliverable:
    Continuous development suitability score in [0,1].

No arbitrary HIGH/MEDIUM/LOW suitability classification.
No retraining or changes to expert judgments.
No modification of frozen Section 04.

Only one new JSON release report is created.
"""

from pathlib import Path
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT
    / "data/processed/development_suitability"
    / "final_expert_ahp"
)

REPORT = ROOT / "outputs/reports"

SCORE_CSV = (
    BASE
    / "16e_02_expert_ahp_weighted_suitability_100m.csv"
)

SCORE_GPKG = (
    BASE
    / "16e_02_expert_ahp_weighted_suitability_100m.gpkg"
)

SCORE_QA = (
    REPORT / "16e_02_expert_ahp_weighted_qa.json"
)

SENS_CSV = (
    BASE
    / "16f_02_sensitivity_scenario_summary.csv"
)

STABILITY_CSV = (
    BASE
    / "16f_02_sensitivity_grid_stability.csv"
)

SENS_QA = (
    REPORT / "16f_02_expert_ahp_sensitivity_qa.json"
)

AHP_QA = (
    REPORT / "16d_e_expert_ahp_consistency_qa.json"
)

AHP_WEIGHTS = (
    REPORT / "16d_e_expert_ahp_weights.csv"
)

STANDARDIZATION_QA = (
    REPORT
    / "16c_m_final_criteria_standardization_qa.json"
)

OUT_QA = (
    REPORT
    / "16g_01_final_expert_ahp_suitability_release.json"
)

LAYER = "expert_ahp_suitability_100m"

CRITERIA = ["C1", "C2", "C4"]

SCORE_FIELDS = [
    "c1_road_access_score",
    "c2_road_context_score",
    "c4_slope_score",
]

WEIGHTS = np.array([0.20, 0.60, 0.20])

FINAL_FIELD = "development_suitability_score"

EXPECTED_CELLS = 9240
EXPECTED_AREA_KM2 = 88.82921581108494


def require(condition, description):

    passed = bool(condition)

    print(
        f"[{'PASS' if passed else 'FAIL'}] "
        f"{description}"
    )

    if not passed:
        raise ValueError(
            f"Final QA failed: {description}"
        )


def sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def numeric(frame, column):

    if column not in frame.columns:
        raise ValueError(
            f"Missing column: {column}"
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def check_grid(frame, name):

    require(
        len(frame) == EXPECTED_CELLS,
        f"{name}: 9,240 rows",
    )

    require(
        "grid_id" in frame.columns
        and frame["grid_id"].notna().all()
        and frame["grid_id"].is_unique,
        f"{name}: unique nonmissing grid IDs",
    )


def ranks(scores, grid_ids):

    # Highest suitability = rank 1.
    # grid_id breaks any exact score ties.

    order = np.lexsort(
        (grid_ids, -scores)
    )

    result = np.empty(
        len(scores),
        dtype=np.int32,
    )

    result[order] = np.arange(
        1,
        len(scores) + 1,
    )

    return result


def main():

    print("=" * 78)
    print(
        "SECTION 03 | STEP 16G-01 "
        "FINAL EXPERT-AHP SUITABILITY QA"
    )
    print("=" * 78)

    inputs = [
        SCORE_CSV,
        SCORE_GPKG,
        SCORE_QA,
        SENS_CSV,
        STABILITY_CSV,
        SENS_QA,
        AHP_QA,
        AHP_WEIGHTS,
        STANDARDIZATION_QA,
    ]

    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    if OUT_QA.exists():
        raise FileExistsError(
            f"Release already exists: {OUT_QA}"
        )

    hashes = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in inputs
    }

    # ======================================================
    # 1. VERIFY SOURCE QA STATUS
    # ======================================================

    print("\n[1] VERIFY RESEARCH STAGE RESULTS")

    score_qa = load_json(SCORE_QA)
    sens_qa = load_json(SENS_QA)
    ahp_qa = load_json(AHP_QA)
    std_qa = load_json(STANDARDIZATION_QA)

    require(
        std_qa.get("status")
        == (
            "PASS_FINAL_SELECTED_CRITERIA_STANDARDIZED_"
            "AHP_WEIGHTING_PENDING"
        ),
        "Latest criteria standardization passed",
    )

    require(
        score_qa.get("status")
        == (
            "PASS_EXPERT_AHP_WEIGHTED_SUITABILITY_"
            "SENSITIVITY_PENDING"
        ),
        "Expert-weighted suitability passed",
    )

    require(
        sens_qa.get("status")
        == (
            "PASS_POST_AHP_SENSITIVITY_WITH_"
            "DOCUMENTED_STRESS_TESTS_"
            "FINAL_RELEASE_PENDING"
        ),
        "Post-AHP sensitivity passed",
    )

    require(
        ahp_qa.get("consistency_acceptable") is True,
        "Expert AHP consistency accepted",
    )

    require(
        ahp_qa.get("scenario")
        == "BASE_3_CRITERIA",
        "Three-criterion expert AHP scenario",
    )

    require(
        ahp_qa.get("consultation_date")
        == "2026-10-17",
        "Expert consultation date verified",
    )

    require(
        ahp_qa.get("expert_role") == "Engineer",
        "Expert designation verified",
    )

    require(
        float(ahp_qa["consistency_ratio"]) < 0.10,
        "AHP consistency ratio below 0.10",
    )

    # ======================================================
    # 2. VERIFY CROSS-STAGE SOURCE HASHES
    # ======================================================

    print("\n[2] VERIFY SOURCE PROVENANCE")

    score_input_hashes = score_qa.get(
        "input_sha256", {}
    )

    sens_input_hashes = sens_qa.get(
        "source_sha256", {}
    )

    for path in [
        AHP_QA,
        AHP_WEIGHTS,
        STANDARDIZATION_QA,
    ]:

        key = path.relative_to(ROOT).as_posix()

        require(
            score_input_hashes.get(key)
            == sha256(path),
            f"Weighted model source verified: {path.name}",
        )

    for path in [
        SCORE_CSV,
        SCORE_GPKG,
        SCORE_QA,
        AHP_QA,
        AHP_WEIGHTS,
    ]:

        key = path.relative_to(ROOT).as_posix()

        require(
            sens_input_hashes.get(key)
            == sha256(path),
            f"Sensitivity source verified: {path.name}",
        )

    require(
        score_qa["outputs"]["csv_sha256"]
        == sha256(SCORE_CSV),
        "Final suitability CSV hash",
    )

    require(
        score_qa["outputs"]["gpkg_sha256"]
        == sha256(SCORE_GPKG),
        "Final suitability GIS hash",
    )

    require(
        sens_qa["outputs"]["scenario_summary_sha256"]
        == sha256(SENS_CSV),
        "Sensitivity scenario CSV hash",
    )

    require(
        sens_qa["outputs"]["grid_stability_sha256"]
        == sha256(STABILITY_CSV),
        "Grid stability CSV hash",
    )

    # ======================================================
    # 3. VERIFY WEIGHTED SUITABILITY
    # ======================================================

    print("\n[3] VERIFY FINAL SUITABILITY")

    frame = pd.read_csv(SCORE_CSV)

    check_grid(
        frame,
        "Final suitability",
    )

    grid_ids = frame["grid_id"].astype(str).to_numpy()

    weight_frame = pd.read_csv(
        AHP_WEIGHTS
    )

    require(
        len(weight_frame) == 3
        and weight_frame["criterion"].is_unique
        and set(weight_frame["criterion"]) == set(CRITERIA),
        "Expert weight table contains C1/C2/C4",
    )

    ordered_weights = (
        weight_frame
        .set_index("criterion")
        .loc[CRITERIA]["expert_ahp_weight"]
        .to_numpy(dtype=float)
    )

    require(
        np.allclose(
            ordered_weights,
            WEIGHTS,
            atol=1e-12,
            rtol=0,
        ),
        "Expert weights remain 0.20/0.60/0.20",
    )

    X = np.column_stack([
        numeric(frame, field)
        for field in SCORE_FIELDS
    ])

    baseline = numeric(
        frame,
        FINAL_FIELD,
    )

    require(
        np.isfinite(X).all()
        and np.isfinite(baseline).all(),
        "All criterion and final scores finite",
    )

    require(
        np.all(X >= 0)
        and np.all(X <= 1)
        and np.all(baseline >= 0)
        and np.all(baseline <= 1),
        "All scores are between 0 and 1",
    )

    require(
        np.allclose(
            X @ ordered_weights,
            baseline,
            atol=1e-12,
            rtol=0,
        ),
        "Final expert-weighted formula reproduced",
    )

    require(
        frame["suitability_criteria_set"]
        .eq("C1_C2_C4").all(),
        "Only C1/C2/C4 used for final model",
    )

    require(
        frame["suitability_class"]
        .eq("NOT_YET_CLASSIFIED").all(),
        "No arbitrary suitability classes assigned",
    )

    area = numeric(
        frame,
        "kmc_overlap_area_m2",
    )

    require(
        np.isfinite(area).all()
        and np.all(area > 0),
        "Valid positive clipped grid areas",
    )

    require(
        np.isclose(
            area.sum() / 1e6,
            EXPECTED_AREA_KM2,
            atol=0.001,
            rtol=0,
        ),
        "Kaduwela study area preserved",
    )

    # ======================================================
    # 4. VERIFY FINAL GIS
    # ======================================================

    print("\n[4] VERIFY FINAL GIS DELIVERABLE")

    geo = gpd.read_file(
        SCORE_GPKG,
        layer=LAYER,
    )

    check_grid(
        geo,
        "Final suitability GIS",
    )

    require(
        geo.crs is not None
        and geo.crs.to_epsg() == 32644,
        "Final GIS CRS EPSG:32644",
    )

    require(
        geo.geometry.notna().all()
        and not geo.geometry.is_empty.any()
        and geo.geometry.is_valid.all(),
        "Final GIS geometries valid",
    )

    require(
        set(geo["grid_id"])
        == set(frame["grid_id"]),
        "CSV/GIS grid identities match",
    )

    geo_ordered = (
        geo.set_index("grid_id")
        .loc[frame["grid_id"]]
    )

    require(
        np.allclose(
            numeric(geo_ordered, FINAL_FIELD),
            baseline,
            atol=1e-12,
            rtol=0,
        ),
        "GIS and CSV final suitability scores match",
    )

    require(
        np.allclose(
            geo_ordered.geometry.area.to_numpy(),
            area,
            atol=1.0,
            rtol=0,
        ),
        "GIS geometry agrees with grid area",
    )

    # ======================================================
    # 5. VALIDATE POST-AHP SCENARIOS
    # ======================================================

    print("\n[5] VALIDATE SENSITIVITY SCENARIOS")

    scenarios = pd.read_csv(
        SENS_CSV
    )

    stability = pd.read_csv(
        STABILITY_CSV
    )

    require(
        len(scenarios) == 13
        and scenarios["scenario_id"].is_unique,
        "13 unique sensitivity scenarios",
    )

    check_grid(
        stability,
        "Grid stability",
    )

    stability = (
        stability
        .set_index("grid_id")
        .loc[frame["grid_id"]]
    )

    require(
        np.allclose(
            numeric(
                stability,
                "baseline_development_suitability_score",
            ),
            baseline,
            atol=1e-12,
            rtol=0,
        ),
        "Sensitivity baseline matches final scores",
    )

    require(
        int(scenarios["scenario_type"]
            .eq("BASELINE").sum()) == 1
        and int(scenarios["scenario_type"]
                .eq(
                    "HYPOTHETICAL_WEIGHT_STRESS_TEST"
                ).sum()) == 12,
        "One baseline plus twelve stress tests",
    )

    expected_scenario_ids = {"BASE_EXPERT_AHP"}

    for source in CRITERIA:
        for target in CRITERIA:
            if source == target:
                continue

            for points in (5, 10):
                expected_scenario_ids.add(
                    f"TRANSFER_{points:02d}"
                    f"PP_{source}_TO_{target}"
                )

    require(
        set(scenarios["scenario_id"])
        == expected_scenario_ids,
        "All intended weight-transfer scenarios present",
    )

    baseline_rank = ranks(
        baseline,
        grid_ids,
    )

    require(
        np.array_equal(
            numeric(
                stability,
                "baseline_suitability_rank",
            ).astype(int),
            baseline_rank,
        ),
        "Stored baseline ranks reproduced",
    )

    top10_size = int(np.ceil(
        EXPECTED_CELLS * 0.10
    ))

    top20_size = int(np.ceil(
        EXPECTED_CELLS * 0.20
    ))

    baseline_top10 = baseline_rank <= top10_size
    baseline_top20 = baseline_rank <= top20_size

    count_top10 = np.zeros(
        EXPECTED_CELLS,
        dtype=int,
    )

    count_top20 = np.zeros(
        EXPECTED_CELLS,
        dtype=int,
    )

    maximum_change = np.zeros(
        EXPECTED_CELLS,
        dtype=float,
    )

    maximum_rank_shift = np.zeros(
        EXPECTED_CELLS,
        dtype=int,
    )

    worst_spearman = 1.0
    lowest_top10 = 1.0
    lowest_top20 = 1.0

    for row in scenarios.itertuples(index=False):

        scenario_weights = np.array([
            float(row.C1_weight),
            float(row.C2_weight),
            float(row.C4_weight),
        ])

        require(
            np.isfinite(scenario_weights).all()
            and np.all(scenario_weights >= 0)
            and np.isclose(
                scenario_weights.sum(),
                1.0,
                atol=1e-12,
            ),
            f"Valid scenario weights: {row.scenario_id}",
        )

        scenario_scores = X @ scenario_weights

        scenario_ranks = ranks(
            scenario_scores,
            grid_ids,
        )

        scenario_top10 = (
            scenario_ranks <= top10_size
        )

        scenario_top20 = (
            scenario_ranks <= top20_size
        )

        count_top10 += scenario_top10
        count_top20 += scenario_top20

        changes = np.abs(
            scenario_scores - baseline
        )

        shifts = np.abs(
            scenario_ranks - baseline_rank
        )

        maximum_change = np.maximum(
            maximum_change,
            changes,
        )

        maximum_rank_shift = np.maximum(
            maximum_rank_shift,
            shifts,
        )

        spearman = float(
            pd.Series(baseline).corr(
                pd.Series(scenario_scores),
                method="spearman",
            )
        )

        retention10 = float(
            np.sum(
                baseline_top10 & scenario_top10
            ) / top10_size
        )

        retention20 = float(
            np.sum(
                baseline_top20 & scenario_top20
            ) / top20_size
        )

        require(
            np.isclose(
                spearman,
                float(row.spearman_score_correlation),
                atol=1e-9,
            ),
            f"Spearman reproduced: {row.scenario_id}",
        )

        require(
            np.isclose(
                retention10,
                float(
                    row.baseline_top10_retention_fraction
                ),
                atol=1e-12,
            )
            and np.isclose(
                retention20,
                float(
                    row.baseline_top20_retention_fraction
                ),
                atol=1e-12,
            ),
            f"Top-group retention reproduced: {row.scenario_id}",
        )

        require(
            np.isclose(
                changes.max(),
                float(
                    row.maximum_absolute_score_change
                ),
                atol=1e-12,
            ),
            f"Maximum score change reproduced: {row.scenario_id}",
        )

        if row.scenario_id != "BASE_EXPERT_AHP":
            worst_spearman = min(
                worst_spearman,
                spearman,
            )

            lowest_top10 = min(
                lowest_top10,
                retention10,
            )

            lowest_top20 = min(
                lowest_top20,
                retention20,
            )

    # ======================================================
    # 6. VERIFY GRID-LEVEL SENSITIVITY
    # ======================================================

    print("\n[6] VERIFY GRID-LEVEL STABILITY")

    require(
        np.array_equal(
            numeric(
                stability,
                "top10pct_scenario_count",
            ).astype(int),
            count_top10,
        ),
        "Top 10% scenario membership reproduced",
    )

    require(
        np.array_equal(
            numeric(
                stability,
                "top20pct_scenario_count",
            ).astype(int),
            count_top20,
        ),
        "Top 20% scenario membership reproduced",
    )

    require(
        np.allclose(
            numeric(
                stability,
                "maximum_absolute_score_change",
            ),
            maximum_change,
            atol=1e-12,
            rtol=0,
        ),
        "Grid maximum score changes reproduced",
    )

    require(
        np.array_equal(
            numeric(
                stability,
                "maximum_absolute_rank_shift",
            ).astype(int),
            maximum_rank_shift,
        ),
        "Grid maximum rank shifts reproduced",
    )

    require(
        np.isclose(
            worst_spearman,
            float(
                sens_qa[
                    "overall_robustness_diagnostics"
                ][
                    "worst_scenario_spearman_correlation"
                ]
            ),
            atol=1e-9,
        ),
        "Worst-case Spearman agrees with QA",
    )

    require(
        np.isclose(
            lowest_top10,
            float(
                sens_qa[
                    "overall_robustness_diagnostics"
                ][
                    "smallest_top_decile_retention_fraction"
                ]
            ),
            atol=1e-12,
        ),
        "Worst-case top-decile retention agrees with QA",
    )

    # ======================================================
    # 7. PRESERVE SOURCE FILES
    # ======================================================

    print("\n[7] VERIFY FILE INTEGRITY")

    for path in inputs:

        key = path.relative_to(ROOT).as_posix()

        require(
            sha256(path) == hashes[key],
            f"Input unchanged: {path.name}",
        )

    # ======================================================
    # 8. CREATE FINAL RELEASE REPORT
    # ======================================================

    print("\n[8] CREATE SECTION 03 RELEASE REPORT")

    release = {
        "project": "J26-DS-338",
        "component": (
            "Urban Green Transition, Urbanization "
            "Pressure and Conservation Intelligence"
        ),
        "section": "03_development_suitability",
        "stage": "16g_01",
        "status": (
            "PASS_SECTION03_FINAL_CONTINUOUS_"
            "SUITABILITY_WITH_DOCUMENTED_LIMITATIONS"
        ),
        "study_area": "Kaduwela Municipal Council",
        "grid_cells": EXPECTED_CELLS,
        "crs": "EPSG:32644",
        "study_area_km2": float(
            area.sum() / 1e6
        ),
        "method": (
            "GIS_MCDA_EXPERT_AHP_"
            "WEIGHTED_LINEAR_COMBINATION"
        ),
        "formula": (
            "S=0.20*C1+0.60*C2+0.20*C4"
        ),
        "suitability_field": FINAL_FIELD,
        "score_range": [0.0, 1.0],
        "criteria": {
            "C1": {
                "name": "Major Road Proximity",
                "direction": "COST",
                "weight": 0.20,
            },
            "C2": {
                "name": "Local Road Density 500m",
                "direction": "BENEFIT",
                "weight": 0.60,
            },
            "C4": {
                "name": "Terrain Slope",
                "direction": "COST",
                "weight": 0.20,
            },
        },
        "expert_consultation": {
            "role": "Engineer",
            "date": "2026-10-17",
            "source": (
                "Researcher-transcribed expert responses"
            ),
            "expert_name_recorded": False,
            "independently_signed": False,
            "consistency_ratio": float(
                ahp_qa["consistency_ratio"]
            ),
        },
        "suitability_distribution": {
            "minimum": float(baseline.min()),
            "p05": float(
                np.quantile(baseline, 0.05)
            ),
            "median": float(
                np.median(baseline)
            ),
            "mean": float(baseline.mean()),
            "p95": float(
                np.quantile(baseline, 0.95)
            ),
            "maximum": float(baseline.max()),
        },
        "sensitivity": {
            "scenario_count": 13,
            "hypothetical_scenarios": 12,
            "worst_spearman": worst_spearman,
            "minimum_top10_retention": lowest_top10,
            "minimum_top20_retention": lowest_top20,
            "real_expert_weights_changed": False,
            "robustness_interpretation": (
                "High overall ranking agreement; "
                "some changes near top-ranked boundaries."
            ),
        },
        "model_scope": {
            "continuous_suitability_completed": True,
            "final_policy_suitability_classes": False,
            "flood_risk_explicitly_modelled": False,
            "zoning_legal_compliance_modelled": False,
            "usable_for_section05_continuous_DCI": True,
        },
        "limitations": [
            (
                "Development suitability represents "
                "only the selected three criteria."
            ),
            (
                "Expert weights are researcher-transcribed "
                "and not independently signed."
            ),
            (
                "Drainage, flood hazard and official "
                "land-use permissions are not directly "
                "modelled by this suitability score."
            ),
            (
                "Hypothetical sensitivity scenarios "
                "are not additional expert judgments."
            ),
            (
                "No arbitrary final HIGH/MEDIUM/LOW "
                "suitability categories are asserted."
            ),
            (
                "The score is decision-support evidence, "
                "not legal permission for development."
            ),
        ],
        "final_deliverables": {
            "suitability_csv": (
                SCORE_CSV.relative_to(ROOT).as_posix()
            ),
            "suitability_gpkg": (
                SCORE_GPKG.relative_to(ROOT).as_posix()
            ),
            "suitability_gpkg_layer": LAYER,
            "sensitivity_scenarios": (
                SENS_CSV.relative_to(ROOT).as_posix()
            ),
            "sensitivity_grid_stability": (
                STABILITY_CSV.relative_to(ROOT).as_posix()
            ),
        },
        "input_sha256": hashes,
        "historical_blocker_note": (
            "Older 16e_01 and 16f_01 status reports "
            "refer to earlier provisional stages. "
            "This new release documents the verified "
            "later expert-AHP calculation and sensitivity."
        ),
        "section04_outputs_modified": False,
    }

    # The report is only written after all
    # verification checks have passed.

    OUT_QA.write_text(
        json.dumps(
            release,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)

    print(
        "PASS_SECTION03_FINAL_CONTINUOUS_"
        "SUITABILITY_WITH_DOCUMENTED_LIMITATIONS"
    )

    print("Final suitability grids:", EXPECTED_CELLS)

    print(
        "Mean suitability:",
        f"{baseline.mean():.6f}",
    )

    print(
        "Worst Spearman:",
        f"{worst_spearman:.6f}",
    )

    print(
        "Minimum top 10% retention:",
        f"{lowest_top10:.6f}",
    )

    print("Expert AHP weights unchanged: YES")
    print("Final GIS verified: YES")
    print("Suitable for Section 05 continuous DCI: YES")
    print("Section 04 modified: NO")

    print("\nRelease report:", OUT_QA)

    print("=" * 78)


if __name__ == "__main__":
    main()
