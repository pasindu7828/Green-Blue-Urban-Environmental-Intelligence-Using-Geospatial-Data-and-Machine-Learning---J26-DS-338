
"""
J26-DS-338 | Section 05 | Step 18B
Common-Grid Integration

Integrates:
1. Final expert-AHP Development Suitability
2. Frozen 2026-2028 Urbanization Pressure
3. Frozen Conservation Intelligence
4. Historical green-loss supporting evidence

Important:
- No model training or new predictions.
- No Development Conflict Index calculated here.
- No Conservation Priority recalculation.
- Missing pressure values remain missing.
- No changes to Sections 02, 03 or 04.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed"
REPORT = ROOT / "outputs/reports"

SUITABILITY = (
    DATA
    / "development_suitability/final_expert_ahp"
    / "16e_02_expert_ahp_weighted_suitability_100m.csv"
)

SUITABILITY_RELEASE = (
    REPORT
    / "16g_01_final_expert_ahp_suitability_release.json"
)

PRESSURE = (
    DATA
    / "urbanization_pressure/inference"
    / "14c_2026_2028_urbanization_pressure_scores.csv"
)

ELIGIBILITY = (
    DATA
    / "urbanization_pressure/inference"
    / "14b_2025_inference_eligibility.csv"
)

CONSERVATION = (
    DATA
    / "conservation_intelligence/final_gis"
    / "17o_conservation_intelligence_final.gpkg"
)

CONSERVATION_LAYER = "primary_conservation_priority"

CONSERVATION_QA = (
    REPORT
    / "conservation_intelligence"
    / "17o_conservation_final_qa.json"
)

PREVIOUS_AUDIT = (
    REPORT
    / "planning_conflict_intervention"
    / "17a_input_contract_audit.json"
)

PREVIOUS_REGISTRY = (
    REPORT
    / "planning_conflict_intervention"
    / "17a_input_contract_registry.csv"
)

OUT_DIR = (
    DATA
    / "planning_conflict_intervention/common_grid"
)

OUT_CSV = (
    OUT_DIR
    / "18b_common_grid_integration_100m.csv"
)

OUT_GPKG = (
    OUT_DIR
    / "18b_common_grid_integration_100m.gpkg"
)

OUT_LAYER = "planning_conflict_common_grid_100m"

OUT_QA = (
    REPORT
    / "planning_conflict_intervention"
    / "18b_common_grid_integration_qa.json"
)


# ============================================================
# 2. FROZEN RESEARCH CONTRACT
# ============================================================

N = 9240
AREA_KM2 = 88.82921581108494

PRESSURE_DOMAIN_N = 4620
GI_N = 8727
CP_N = 4609

THRESHOLD = 0.715


# ============================================================
# 3. HELPERS
# ============================================================

def must(condition, label):

    passed = bool(condition)

    print(
        f"[{'PASS' if passed else 'FAIL'}] {label}"
    )

    if not passed:
        raise ValueError(
            f"Step 18B QA failed: {label}"
        )


def digest(path):

    h = hashlib.sha256()

    with path.open("rb") as f:
        for piece in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(piece)

    return h.hexdigest()


def unique_ids(table, expected_n, label):

    must(
        len(table) == expected_n,
        f"{label}: {expected_n:,} records",
    )

    must(
        "grid_id" in table.columns
        and table["grid_id"].notna().all()
        and table["grid_id"].is_unique,
        f"{label}: unique grid_id",
    )


def numbers(table, field):

    if field not in table.columns:
        raise KeyError(
            f"Missing required field: {field}"
        )

    return pd.to_numeric(
        table[field],
        errors="coerce",
    ).to_numpy(dtype=float)


def require_fields(table, names, label):

    missing = sorted(
        set(names) - set(table.columns)
    )

    must(
        not missing,
        f"{label}: required fields present"
        f" ({', '.join(missing) if missing else 'OK'})",
    )


def validate_range(values, name):

    present = np.isfinite(values)

    must(
        np.all(np.isnan(values) | present),
        f"{name}: no infinite values",
    )

    must(
        np.all(
            (values[present] >= -1e-12)
            & (values[present] <= 1 + 1e-12)
        ),
        f"{name}: available scores in [0,1]",
    )


# ============================================================
# 4. MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "SECTION 05 | STEP 18B - "
        "FINAL COMMON-GRID INTEGRATION"
    )
    print("=" * 78)

    sources = [
        SUITABILITY,
        SUITABILITY_RELEASE,
        PRESSURE,
        ELIGIBILITY,
        CONSERVATION,
        CONSERVATION_QA,
        PREVIOUS_AUDIT,
        PREVIOUS_REGISTRY,
    ]

    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    for destination in [
        OUT_CSV,
        OUT_GPKG,
        OUT_QA,
    ]:
        if destination.exists():
            raise FileExistsError(
                f"Output already exists: {destination}"
            )

    hashes = {
        p.relative_to(ROOT).as_posix(): digest(p)
        for p in sources
    }

    # ========================================================
    # STEP 1 - VERIFY FINAL SOURCE RELEASES
    # ========================================================

    print("\n[1] VERIFY FINAL SOURCE RELEASES")

    suit_release = json.loads(
        SUITABILITY_RELEASE.read_text(
            encoding="utf-8"
        )
    )

    conservation_qa = json.loads(
        CONSERVATION_QA.read_text(
            encoding="utf-8"
        )
    )

    earlier_audit = json.loads(
        PREVIOUS_AUDIT.read_text(
            encoding="utf-8"
        )
    )

    must(
        suit_release.get("status")
        == (
            "PASS_SECTION03_FINAL_CONTINUOUS_"
            "SUITABILITY_WITH_DOCUMENTED_LIMITATIONS"
        ),
        "Section 03 continuous suitability release passed",
    )

    must(
        suit_release.get("grid_cells") == N
        and suit_release.get("crs") == "EPSG:32644",
        "Section 03 grid and CRS verified",
    )

    must(
        "PASS_FINAL_GIS_PACKAGE_WITH_DOCUMENTED_LIMITATIONS"
        in json.dumps(conservation_qa, default=str),
        "Frozen conservation GIS QA passed",
    )

    must(
        earlier_audit.get("status") in (
            "PASS_INPUT_AUDIT_WITH_EXPERT_AHP_BLOCKER",
            "PASS_INPUT_AUDIT",
        ),
        "Historical Section 05 input audit recognized",
    )

    registry = pd.read_csv(
        PREVIOUS_REGISTRY
    )

    require_fields(
        registry,
        ["path", "sha256"],
        "18A frozen registry",
    )

    for p in (
        PRESSURE,
        ELIGIBILITY,
        CONSERVATION,
    ):

        rel = p.relative_to(ROOT).as_posix()

        normalized = (
            registry["path"]
            .astype(str)
            .str.replace("\\", "/", regex=False)
        )

        rows = registry.loc[
            normalized == rel
        ]

        if len(rows):
            must(
                len(rows) == 1
                and str(rows.iloc[0]["sha256"]) == digest(p),
                f"18A frozen hash unchanged: {p.name}",
            )

    print(
        "[INFO] Previous AHP blocker superseded "
        "by final Section 03 release."
    )

    # ========================================================
    # STEP 2 - LOAD VERIFIED SOURCE DATA
    # ========================================================

    print("\n[2] READ AND VERIFY SOURCE TABLES")

    suit = pd.read_csv(
        SUITABILITY
    )

    press = pd.read_csv(
        PRESSURE
    )

    eligible = pd.read_csv(
        ELIGIBILITY
    )

    geo = gpd.read_file(
        CONSERVATION,
        layer=CONSERVATION_LAYER,
    )

    unique_ids(
        suit, N, "Final suitability"
    )

    unique_ids(
        eligible, N, "Pressure eligibility"
    )

    unique_ids(
        press,
        PRESSURE_DOMAIN_N,
        "Frozen pressure",
    )

    unique_ids(
        geo, N, "Frozen conservation GIS"
    )

    must(
        geo.crs is not None
        and geo.crs.to_epsg() == 32644,
        "Frozen GIS EPSG:32644",
    )

    must(
        geo.geometry.notna().all()
        and not geo.geometry.is_empty.any()
        and geo.geometry.is_valid.all(),
        "Frozen conservation geometries valid",
    )

    grid_ids = set(
        geo["grid_id"]
    )

    must(
        set(suit.grid_id)
        == set(eligible.grid_id)
        == grid_ids,
        "Full-grid IDs match exactly",
    )

    must(
        set(press.grid_id).issubset(grid_ids),
        "Pressure scores belong to KMC grids",
    )

    # ========================================================
    # STEP 3 - SELECT ACTUAL SOURCE COLUMNS
    # ========================================================

    suit_fields = [
        "grid_id",
        "kmc_overlap_area_m2",
        "kmc_overlap_fraction",
        "boundary_cell",
        "coverage_group",
        "terrain_fallback_applied",
        "c1_road_access_score",
        "c2_road_context_score",
        "c4_slope_score",
        "c1_weighted_contribution",
        "c2_weighted_contribution",
        "c4_weighted_contribution",
        "development_suitability_score",
        "suitability_criteria_set",
        "suitability_class",
    ]

    pressure_fields = [
        "grid_id",
        "origin_year",
        "pressure_period",
        "urbanization_pressure_score",
        "pressure_percentile",
        "pressure_rank",
        "operating_threshold",
        "threshold_flag",
        "threshold_status",
        "score_interpretation",
    ]

    eligible_fields = [
        "grid_id",
        "origin_year",
        "pressure_period",
        "final_inference_ready",
        "spatial_overlap_eligible",
        "coverage_eligible",
        "current_nonbuilt_eligible",
        "past_stability_eligible",
        "predictor_complete",
    ]

    conservation_fields = [
        "grid_id",
        "geometry",
        "kmc_overlap_area_m2",
        "current_green_condition_status",
        "green_importance_status",
        "green_importance_frozen",
        "green_importance_index_0_1",
        "primary_conservation_priority_0_1",
        "primary_priority_status",
        "primary_priority_rank_common_domain",
        "primary_priority_percentile_common_domain",
        "future_pressure_domain_status",
        "future_urbanization_pressure_score",
        "future_pressure_origin_year",
        "future_pressure_period",
        "future_operating_threshold",
        "future_threshold_flag",
        "integration_evidence_status",
        "historical_g2b_robust_exposure_share_250m",
        "historical_g2b_robust_exposure_share_500m",
        "historical_g2b_strict_exposure_share_250m",
        "historical_g2b_strict_exposure_share_500m",
        "historical_g2b_robust_nearest_pixel_centre_m",
    ]

    require_fields(
        suit, suit_fields, "Suitability"
    )

    require_fields(
        press, pressure_fields, "Pressure"
    )

    require_fields(
        eligible, eligible_fields, "Eligibility"
    )

    require_fields(
        geo, conservation_fields, "Conservation"
    )

    # ========================================================
    # STEP 4 - VERIFY PRESSURE DOMAIN AND AHP SUITABILITY
    # ========================================================

    print(
        "\n[3] VERIFY PRESSURE DOMAIN "
        "AND FINAL SUITABILITY"
    )

    ready_raw = (
        eligible["final_inference_ready"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    must(
        ready_raw.isin(
            ("true", "false", "1", "0")
        ).all(),
        "Pressure eligibility flags unambiguous",
    )

    eligible["final_inference_ready"] = (
        ready_raw.isin(("true", "1"))
    )

    ready_ids = set(
        eligible.loc[
            eligible.final_inference_ready,
            "grid_id",
        ]
    )

    must(
        len(ready_ids) == PRESSURE_DOMAIN_N
        and ready_ids == set(press.grid_id),
        "Exactly 4,620 scored cells match eligible domain",
    )

    must(
        eligible["origin_year"].eq(2025).all()
        and eligible["pressure_period"].eq("2026-2028").all()
        and press["origin_year"].eq(2025).all()
        and press["pressure_period"].eq("2026-2028").all(),
        "Frozen pressure origin and period verified",
    )

    must(
        np.allclose(
            numbers(press, "operating_threshold"),
            THRESHOLD,
            atol=1e-12,
            rtol=0,
        ),
        "Frozen pressure threshold = 0.715",
    )

    must(
        int(
            np.count_nonzero(
                numbers(press, "threshold_flag") == 1
            )
        ) == 255,
        "255 frozen threshold-flagged cells",
    )

    must(
        press["score_interpretation"].eq(
            "MODEL_DERIVED_PRESSURE_SCORE_"
            "NOT_CALIBRATED_PROBABILITY"
        ).all(),
        "Pressure is not a calibrated probability",
    )

    score = numbers(
        suit,
        "development_suitability_score",
    )

    must(
        np.isfinite(score).all(),
        "Final suitability complete for all grids",
    )

    validate_range(
        score, "Development suitability"
    )

    must(
        suit["suitability_criteria_set"]
        .eq("C1_C2_C4").all(),
        "Final expert-AHP criteria C1/C2/C4",
    )

    must(
        suit["suitability_class"]
        .eq("NOT_YET_CLASSIFIED").all(),
        "No fabricated suitability classes",
    )

    contributions = (
        numbers(suit, "c1_weighted_contribution")
        + numbers(suit, "c2_weighted_contribution")
        + numbers(suit, "c4_weighted_contribution")
    )

    must(
        np.allclose(
            contributions,
            score,
            atol=1e-12,
            rtol=0,
        ),
        "Final suitability score reproduced",
    )

    suit_area = numbers(
        suit, "kmc_overlap_area_m2"
    )

    must(
        np.isfinite(suit_area).all()
        and np.all(suit_area > 0)
        and np.isclose(
            suit_area.sum() / 1e6,
            AREA_KM2,
            atol=0.001,
            rtol=0,
        ),
        "Suitability clipped areas valid",
    )

    # ========================================================
    # STEP 5 - COMMON-GRID JOIN
    # ========================================================

    print("\n[4] JOIN ALL SOURCES BY GRID ID")

    suit_join = (
        suit[suit_fields]
        .rename(columns={
            "kmc_overlap_area_m2":
                "suitability_kmc_overlap_area_m2",
        })
    )

    elig_join = (
        eligible[eligible_fields]
        .rename(columns={
            "origin_year": "eligibility_origin_year",
            "pressure_period": "eligibility_pressure_period",
        })
    )

    press_join = (
        press[pressure_fields]
        .rename(columns={
            "origin_year": "pressure_origin_year",
        })
    )

    combined = (
        geo[conservation_fields]
        .merge(
            suit_join,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            elig_join,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            press_join,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
    )

    unique_ids(
        combined, N, "Integrated common grid"
    )

    must(
        np.isfinite(
            numbers(
                combined,
                "development_suitability_score",
            )
        ).all(),
        "9,240 final suitability scores preserved",
    )

    must(
        np.allclose(
            numbers(combined, "kmc_overlap_area_m2"),
            numbers(
                combined,
                "suitability_kmc_overlap_area_m2",
            ),
            atol=1e-6,
            rtol=1e-9,
        ),
        "Suitability/conservation clipped areas match",
    )

    must(
        np.allclose(
            combined.geometry.area.to_numpy(dtype=float),
            numbers(combined, "kmc_overlap_area_m2"),
            atol=1.0,
            rtol=0,
        ),
        "Grid geometry agrees with clipped areas",
    )

    must(
        np.isclose(
            combined.geometry.area.sum() / 1e6,
            AREA_KM2,
            atol=0.001,
            rtol=0,
        ),
        "KMC study area preserved",
    )

    # ========================================================
    # STEP 6 - PRESERVE MISSING-DOMAIN RULES
    # ========================================================

    print("\n[5] VALIDATE COMMON EVIDENCE DOMAINS")

    domain = (
        combined["final_inference_ready"]
        .astype(bool)
        .to_numpy()
    )

    pressure_values = numbers(
        combined,
        "urbanization_pressure_score",
    )

    gi_values = numbers(
        combined,
        "green_importance_index_0_1",
    )

    cp_values = numbers(
        combined,
        "primary_conservation_priority_0_1",
    )

    frozen_pressure = numbers(
        combined,
        "future_urbanization_pressure_score",
    )

    validate_range(
        pressure_values, "Frozen pressure"
    )

    validate_range(
        gi_values, "Frozen Green Importance"
    )

    validate_range(
        cp_values, "Frozen Conservation Priority"
    )

    must(
        np.array_equal(
            np.isfinite(pressure_values),
            domain,
        ),
        "Pressure exists exactly inside eligible domain",
    )

    must(
        np.isnan(
            pressure_values[~domain]
        ).all(),
        "Out-of-domain pressure remains missing",
    )

    must(
        np.count_nonzero(
            np.isfinite(gi_values)
        ) == GI_N,
        "8,727 frozen Green Importance values preserved",
    )

    must(
        np.count_nonzero(
            np.isfinite(cp_values)
        ) == CP_N,
        "4,609 frozen Conservation Priority values preserved",
    )

    must(
        np.all(
            np.isnan(frozen_pressure) | domain
        ),
        "Conservation GIS has no out-of-domain pressure",
    )

    valid_frozen = np.isfinite(
        frozen_pressure
    )

    must(
        np.allclose(
            frozen_pressure[valid_frozen],
            pressure_values[valid_frozen],
            atol=1e-9,
            rtol=0,
        ),
        "Frozen pressure agrees across Sections 02/04",
    )

    common_domain = (
        domain & np.isfinite(gi_values)
    )

    has_cp = np.isfinite(
        cp_values
    )

    must(
        np.array_equal(
            common_domain,
            has_cp,
        ),
        "Frozen CP available in common P/GI domain",
    )

    # Verification only:
    # CP is imported unchanged from Section 04.
    # This does not recompute or replace CP.

    must(
        np.allclose(
            cp_values[has_cp],
            gi_values[has_cp]
            * pressure_values[has_cp],
            atol=1e-8,
            rtol=0,
        ),
        "Frozen CP formula verified without replacement",
    )

    must(
        np.count_nonzero(
            domain & ~np.isfinite(gi_values)
        ) == 11,
        "11 pressure-only grids lack GI",
    )

    must(
        np.count_nonzero(
            ~domain & np.isfinite(gi_values)
        ) == 4118,
        "4,118 GI-only grids outside pressure domain",
    )

    must(
        np.count_nonzero(
            ~domain & ~np.isfinite(gi_values)
        ) == 502,
        "502 grids lack both pressure and GI",
    )

    # ========================================================
    # STEP 7 - EXPLICIT INTEGRATION STATUSES
    # ========================================================

    print("\n[6] ASSIGN INTEGRATION DOMAIN STATUSES")

    combined["pressure_domain_status_section05"] = (
        np.where(
            domain,
            "IN_MODEL_DOMAIN",
            "NOT_IN_DOMAIN",
        )
    )

    combined["suitability_input_status_section05"] = (
        "FINAL_EXPERT_AHP_AVAILABLE"
    )

    combined["dci_input_status"] = (
        np.where(
            domain,
            "READY_FOR_18C_DCI",
            "PRESSURE_NOT_IN_DOMAIN",
        )
    )

    combined["cci_input_status"] = (
        np.where(
            has_cp,
            "FROZEN_CP_AVAILABLE",
            "FROZEN_CP_UNAVAILABLE",
        )
    )

    combined["joint_evidence_status"] = (
        np.select(
            [
                domain & has_cp,
                domain & ~has_cp,
                ~domain & np.isfinite(gi_values),
                ~domain & ~np.isfinite(gi_values),
            ],
            [
                "P_S_GI_CP_COMMON_DOMAIN",
                "P_S_ONLY_GI_CP_UNAVAILABLE",
                "S_GI_ONLY_PRESSURE_NOT_IN_DOMAIN",
                "S_ONLY_PRESSURE_GI_CP_UNAVAILABLE",
            ],
            default="UNEXPECTED",
        )
    )

    must(
        not combined[
            "joint_evidence_status"
        ].eq("UNEXPECTED").any(),
        "All grids have valid evidence-domain status",
    )

    must(
        int(
            combined["dci_input_status"]
            .eq("READY_FOR_18C_DCI")
            .sum()
        ) == 4620,
        "4,620 grids ready for Development Conflict Index",
    )

    must(
        int(
            combined["cci_input_status"]
            .eq("FROZEN_CP_AVAILABLE")
            .sum()
        ) == 4609,
        "4,609 frozen conservation conflict inputs",
    )

    print("\nEvidence domain distribution:")
    print(
        combined["joint_evidence_status"]
        .value_counts()
        .to_string()
    )

    print(
        "\n[INFO] DCI is not calculated in Step 18B."
    )

    print(
        "[INFO] CP is not multiplied by pressure again."
    )

    print(
        "[INFO] Historical loss remains supporting evidence."
    )

    # ========================================================
    # STEP 8 - EXPORT NEW OUTPUTS
    # ========================================================

    print("\n[7] EXPORT AND VERIFY COMMON-GRID OUTPUTS")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_QA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix="18b_build_",
        dir=OUT_DIR,
    ) as folder:

        folder = Path(folder)

        temp_csv = folder / OUT_CSV.name
        temp_gpkg = folder / OUT_GPKG.name

        combined.drop(
            columns="geometry"
        ).to_csv(
            temp_csv,
            index=False,
        )

        combined.to_file(
            temp_gpkg,
            layer=OUT_LAYER,
            driver="GPKG",
            engine="pyogrio",
        )

        csv_check = pd.read_csv(
            temp_csv
        )

        geo_check = gpd.read_file(
            temp_gpkg,
            layer=OUT_LAYER,
        )

        unique_ids(
            csv_check, N, "Exported common-grid CSV"
        )

        unique_ids(
            geo_check, N, "Exported common-grid GIS"
        )

        must(
            geo_check.crs is not None
            and geo_check.crs.to_epsg() == 32644,
            "Exported GIS EPSG:32644",
        )

        must(
            np.allclose(
                numbers(
                    csv_check,
                    "development_suitability_score",
                ),
                numbers(
                    combined,
                    "development_suitability_score",
                ),
                atol=1e-12,
                rtol=0,
            ),
            "Exported suitability scores preserved",
        )

        must(
            np.allclose(
                numbers(
                    csv_check,
                    "urbanization_pressure_score",
                ),
                pressure_values,
                atol=1e-12,
                rtol=0,
                equal_nan=True,
            ),
            "Exported pressure values/missingness preserved",
        )

        must(
            np.allclose(
                numbers(
                    csv_check,
                    "primary_conservation_priority_0_1",
                ),
                cp_values,
                atol=1e-12,
                rtol=0,
                equal_nan=True,
            ),
            "Exported Conservation Priority preserved",
        )

        must(
            set(geo_check.grid_id)
            == set(combined.grid_id),
            "Exported GIS grid IDs unchanged",
        )

        must(
            geo_check.geometry.is_valid.all(),
            "Exported GIS geometry valid",
        )

        for path in [
            OUT_CSV,
            OUT_GPKG,
            OUT_QA,
        ]:
            if path.exists():
                raise FileExistsError(
                    f"Output appeared during run: {path}"
                )

        temp_csv.replace(
            OUT_CSV
        )

        temp_gpkg.replace(
            OUT_GPKG
        )

    # ========================================================
    # STEP 9 - VERIFY FROZEN INPUT FILES
    # ========================================================

    print("\n[8] VERIFY SOURCE FILE INTEGRITY")

    for p in sources:

        relative = (
            p.relative_to(ROOT).as_posix()
        )

        must(
            digest(p) == hashes[relative],
            f"Source unchanged: {p.name}",
        )

    # ========================================================
    # STEP 10 - FINAL QA REPORT
    # ========================================================

    print("\n[9] GENERATE STEP 18B QA REPORT")

    qa = {
        "section": "05_planning_conflict_intervention",
        "stage": "18b_common_grid_integration",
        "status": (
            "PASS_COMMON_GRID_FINAL_EXPERT_AHP_INTEGRATION"
        ),
        "grid_cells": N,
        "crs": "EPSG:32644",
        "study_area_km2": float(
            combined.geometry.area.sum() / 1e6
        ),
        "pressure_origin_year": 2025,
        "pressure_period": "2026-2028",
        "pressure_threshold_frozen": THRESHOLD,
        "pressure_not_a_calibrated_probability": True,
        "suitability_release": "16g_01",
        "suitability_criteria": "C1_C2_C4",
        "expert_weights": {
            "C1": 0.20,
            "C2": 0.60,
            "C4": 0.20,
        },
        "suitability_classes_assigned": False,
        "domain_counts": {
            "pressure_in_domain": int(
                domain.sum()
            ),
            "pressure_not_in_domain": int(
                (~domain).sum()
            ),
            "green_importance_scored": int(
                np.isfinite(gi_values).sum()
            ),
            "green_importance_unavailable": int(
                np.isnan(gi_values).sum()
            ),
            "frozen_conservation_priority_available": int(
                has_cp.sum()
            ),
            "frozen_conservation_priority_unavailable": int(
                (~has_cp).sum()
            ),
            "dci_eligible_for_next_stage": int(
                domain.sum()
            ),
            "joint_evidence_status_counts": (
                combined[
                    "joint_evidence_status"
                ]
                .value_counts()
                .to_dict()
            ),
        },
        "methodology_notes": [
            (
                "Pressure and conservation data are "
                "frozen outputs, not recomputed."
            ),
            (
                "Out-of-domain pressure remains missing, "
                "never imputed as zero."
            ),
            (
                "Frozen Conservation Priority is not "
                "multiplied by pressure again."
            ),
            (
                "Historical green-to-built exposure is "
                "supporting evidence only."
            ),
            (
                "No Development Conflict Index or "
                "intervention class is calculated in 18B."
            ),
            (
                "Development Suitability is continuous "
                "expert-AHP GIS-MCDA, not a legal decision."
            ),
            (
                "Expert responses are researcher-transcribed "
                "and not independently signed."
            ),
            (
                "The older 18A AHP blocker is historical "
                "and superseded by the Section 03 release."
            ),
        ],
        "input_sha256": hashes,
        "outputs": {
            "csv": (
                OUT_CSV.relative_to(ROOT).as_posix()
            ),
            "gpkg": (
                OUT_GPKG.relative_to(ROOT).as_posix()
            ),
            "layer": OUT_LAYER,
            "csv_sha256": digest(OUT_CSV),
            "gpkg_sha256": digest(OUT_GPKG),
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
    # STEP 11 - TERMINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 78)

    print(
        "PASS_COMMON_GRID_FINAL_EXPERT_AHP_INTEGRATION"
    )

    print("KMC grid cells:", N)
    print("Pressure in domain:", int(domain.sum()))
    print("Pressure not in domain:", int((~domain).sum()))
    print("Frozen CP available:", int(has_cp.sum()))

    print(
        "Suitability scores:",
        int(np.isfinite(score).sum()),
    )

    print("DCI calculated: NO - NEXT STEP 18C")
    print("Frozen Section 04 modified: NO")

    print("\nCSV:", OUT_CSV)
    print("GPKG:", OUT_GPKG)
    print("QA:", OUT_QA)

    print("=" * 78)


if __name__ == "__main__":
    main()
