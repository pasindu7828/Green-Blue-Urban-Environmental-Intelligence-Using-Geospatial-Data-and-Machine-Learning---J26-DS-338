
"""
STEP 16D-G - DEVELOPMENT SUITABILITY DASHBOARD REFERENCE RELEASE

Combines validated Step 16D-F data and raw context variables
into a stable, dashboard-ready GeoPackage/CSV contract.

This is a PROVISIONAL GIS-MCDA reference model, NOT:
- Expert-weighted AHP
- Independently validated development prediction
- Legal development permission
- Parcel-level buildability assessment

No original research files are modified.
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
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "outputs" / "reports"

INTEGRATED_CSV = (
    REPORTS
    / "16d_f_integrated_suitability_intelligence_100m.csv"
)

INTEGRATED_GPKG = (
    REPORTS
    / "16d_f_integrated_suitability_intelligence_100m.gpkg"
)

INTEGRATED_META = (
    REPORTS
    / "16d_f_integration_summary.json"
)

STANDARD_CSV = (
    REPORTS
    / "16d_a_provisional_standardized_criteria_100m.csv"
)

STANDARD_META = (
    REPORTS
    / "16d_a_standardized_criteria_summary.json"
)

AHP_META = (
    REPORTS
    / "16d_d_ahp_consistency_summary.json"
)

REVIEW_FORM = (
    REPORTS
    / "16c_l_expert_criteria_review_template.csv"
)

AHP_FORM = (
    REPORTS
    / "16c_l_ahp_pairwise_template.csv"
)

OUT_CSV = (
    REPORTS
    / "16d_g_dashboard_reference_100m.csv"
)

OUT_GPKG = (
    REPORTS
    / "16d_g_dashboard_reference_100m.gpkg"
)

OUT_CONTRACT = (
    REPORTS
    / "16d_g_dashboard_data_contract.json"
)

OUT_QA = (
    REPORTS
    / "16d_g_dashboard_release_qa.csv"
)

OUT_SUMMARY = (
    REPORTS
    / "16d_g_dashboard_release_summary.json"
)

N = 9240
CRS = 32644


# ============================================================
# 2. HELPERS
# ============================================================

def digest(path):

    h = hashlib.sha256()

    with path.open("rb") as stream:

        for part in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(part)

    return h.hexdigest()


def read_json(path):

    with path.open(
        encoding="utf-8"
    ) as stream:
        return json.load(stream)


def read_csv(path):

    df = pd.read_csv(
        path,
        dtype={"grid_id": "string"},
    )

    df["grid_id"] = (
        df["grid_id"].str.strip()
    )

    return df


def as_bool(series):

    values = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    if not values.isin(
        ["true", "false", "1", "0"]
    ).all():

        raise ValueError(
            f"Invalid boolean values: {series.name}"
        )

    return values.isin(
        ["true", "1"]
    ).to_numpy(dtype=bool)


# ============================================================
# 3. MAIN
# ============================================================

def main():

    print("=" * 88)
    print(
        "STEP 16D-G - DASHBOARD-READY "
        "PROVISIONAL SUITABILITY RELEASE"
    )
    print("=" * 88)

    inputs = [
        INTEGRATED_CSV,
        INTEGRATED_GPKG,
        INTEGRATED_META,
        STANDARD_CSV,
        STANDARD_META,
        AHP_META,
        REVIEW_FORM,
        AHP_FORM,
    ]

    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_hashes = {
        str(path): digest(path)
        for path in inputs
    }

    f_meta = read_json(INTEGRATED_META)
    a_meta = read_json(STANDARD_META)
    d_meta = read_json(AHP_META)

    checks = []

    def check(name, passed, detail=""):

        checks.append({
            "check": name,
            "passed": bool(passed),
            "detail": str(detail),
        })

    def checkpoint():

        failed = [
            row for row in checks
            if not row["passed"]
        ]

        if failed:

            pd.DataFrame(checks).to_csv(
                OUT_QA,
                index=False,
            )

            for row in failed:

                print(
                    "REVIEW:",
                    row["check"],
                    row["detail"],
                )

            raise RuntimeError(
                "Step 16D-G QA failed. "
                "Review QA CSV."
            )

    # ========================================================
    # 4. RESEARCH PROVENANCE CHECKS
    # ========================================================

    check(
        "Step 16D-F integration passed",
        f_meta.get("qa_status") == "PASS",
    )

    check(
        "Step 16D-A standardization passed",
        a_meta.get("qa_status") == "PASS",
    )

    check(
        "AHP intake technical checks passed",
        d_meta.get("technical_qa_status") == "PASS",
    )

    check(
        "Integrated output remains a reference model",
        f_meta.get("methodology_status")
        == "INTEGRATION_READY_PROVISIONAL_REFERENCE_ONLY",
    )

    check(
        "Study grid contains 9240 cells",
        f_meta.get("grid_count") == N
        and a_meta.get("grid_cells") == N,
    )

    # If someone edits the expert forms after running
    # Step 16D-D, the intake report must be refreshed.

    stored_hashes = d_meta.get(
        "input_sha256", {}
    )

    check(
        "AHP intake report matches current expert forms",
        stored_hashes.get(REVIEW_FORM.name)
        == digest(REVIEW_FORM)
        and stored_hashes.get(AHP_FORM.name)
        == digest(AHP_FORM),
        "If failed, rerun Step 16D-D first.",
    )

    checkpoint()

    # ========================================================
    # 5. LOAD AND ALIGN VERIFIED DATA
    # ========================================================

    print("\nLoading validated research outputs...")

    integrated = read_csv(INTEGRATED_CSV)
    standard = read_csv(STANDARD_CSV)

    gis = gpd.read_file(
        INTEGRATED_GPKG,
        layer="provisional_suitability_intelligence",
    )

    gis["grid_id"] = (
        gis["grid_id"]
        .astype("string")
        .str.strip()
    )

    for label, frame in [
        ("Integrated CSV", integrated),
        ("Standardized CSV", standard),
        ("Integrated GeoPackage", gis),
    ]:

        check(
            f"{label}: 9240 unique non-null IDs",
            len(frame) == N
            and frame["grid_id"].notna().all()
            and frame["grid_id"].is_unique,
        )

    check(
        "All source grid ID sets match",
        set(integrated["grid_id"])
        == set(standard["grid_id"])
        == set(gis["grid_id"]),
    )

    check(
        "GIS polygons valid and EPSG:32644",
        gis.crs is not None
        and gis.crs.to_epsg() == CRS
        and gis.geometry.notna().all()
        and (~gis.geometry.is_empty).all()
        and gis.geometry.is_valid.all(),
    )

    checkpoint()

    std = (
        standard
        .set_index("grid_id")
        .reindex(integrated["grid_id"])
    )

    geom = (
        gis
        .set_index("grid_id")
        .geometry
        .reindex(integrated["grid_id"])
    )

    required_integrated = [
        "reference_score",
        "relative_class",
        "reference_rank_desc",
        "reference_top_decile",
        "c1_road_proximity_score",
        "c2_road_density_score",
        "c4_slope_score",
        "weakest_baseline_criterion",
        "top10_frequency_pctclip",
        "top10_frequency_rank",
        "high_stability_both_methods",
        "cell_area_m2",
        "small_boundary_fragment",
        "planning_review_needed",
        "review_flags",
        "coverage_label",
        "classified_pct",
        "unclassified_pct",
        "unmapped_pct",
        "ambiguous_pct",
        "conservation_water_pct",
        "highland_approx_pct",
        "provisional_zone_name",
        "legal_zoning_status",
        "development_permission",
    ]

    required_standard = [
        "c3_context_building_pct_500m",
        "c5_elevation_mean_m",
    ]

    check(
        "All integrated contract fields available",
        set(required_integrated).issubset(
            integrated.columns
        ),
    )

    check(
        "C3 and C5 raw context available",
        set(required_standard).issubset(
            std.columns
        ),
    )

    check(
        "All grid geometries aligned",
        geom.notna().all(),
    )

    checkpoint()

    # ========================================================
    # 6. PREPARE DASHBOARD VARIABLES
    # ========================================================

    score = pd.to_numeric(
        integrated["reference_score"],
        errors="coerce",
    ).to_numpy(dtype=float)

    c1 = pd.to_numeric(
        integrated["c1_road_proximity_score"],
        errors="coerce",
    ).to_numpy(dtype=float)

    c2 = pd.to_numeric(
        integrated["c2_road_density_score"],
        errors="coerce",
    ).to_numpy(dtype=float)

    c4 = pd.to_numeric(
        integrated["c4_slope_score"],
        errors="coerce",
    ).to_numpy(dtype=float)

    stability = pd.to_numeric(
        integrated["top10_frequency_pctclip"],
        errors="coerce",
    ).to_numpy(dtype=float)

    stability_rank = pd.to_numeric(
        integrated["top10_frequency_rank"],
        errors="coerce",
    ).to_numpy(dtype=float)

    area = pd.to_numeric(
        integrated["cell_area_m2"],
        errors="coerce",
    ).to_numpy(dtype=float)

    top = as_bool(
        integrated["reference_top_decile"]
    )

    review = as_bool(
        integrated["planning_review_needed"]
    )

    stable = as_bool(
        integrated["high_stability_both_methods"]
    )

    small = as_bool(
        integrated["small_boundary_fragment"]
    )

    numeric = np.column_stack([
        score,
        c1, c2, c4,
        stability, stability_rank,
        area,
    ])

    check(
        "All core dashboard measurements finite",
        np.isfinite(numeric).all(),
    )

    check(
        "All scores and stability frequencies within 0-1",
        np.all(
            (numeric[:, :6] >= 0)
            & (numeric[:, :6] <= 1)
        ),
    )

    check(
        "Reference score reproduces C1/C2/C4 mean",
        np.allclose(
            score,
            (c1 + c2 + c4) / 3,
            atol=1e-10,
        ),
    )

    check(
        "Polygon areas match preserved research geometry",
        np.allclose(
            geom.area.to_numpy(dtype=float),
            area,
            atol=0.01,
        ),
    )

    check(
        "Provisional zoning remains legally unverified",
        integrated["legal_zoning_status"]
        .eq("UNVERIFIED").all()
        and integrated["development_permission"]
        .eq("NOT_ASSESSED").all(),
    )

    checkpoint()

    # ========================================================
    # 7. BUILD RELEASE DATASET
    # ========================================================

    print("Creating dashboard field contract...")

    out = pd.DataFrame({
        "grid_id":
            integrated["grid_id"].to_numpy(),

        "suitability_score":
            score,

        "relative_suitability_class":
            integrated["relative_class"].to_numpy(),

        "reference_rank":
            pd.to_numeric(
                integrated["reference_rank_desc"],
                errors="raise",
            ).to_numpy(dtype=int),

        "top_decile_reference":
            top,

        "c1_road_proximity":
            c1,

        "c2_road_density":
            c2,

        "c4_slope":
            c4,

        "limiting_criterion":
            integrated[
                "weakest_baseline_criterion"
            ].to_numpy(),

        "c3_building_context_raw_pct":
            pd.to_numeric(
                std[
                    "c3_context_building_pct_500m"
                ],
                errors="raise",
            ).to_numpy(dtype=float),

        "c5_elevation_raw_m":
            pd.to_numeric(
                std["c5_elevation_mean_m"],
                errors="raise",
            ).to_numpy(dtype=float),

        "weight_stability_pctclip":
            stability,

        "weight_stability_rank":
            stability_rank,

        "high_stability_both_methods":
            stable,

        "cell_area_m2":
            area,

        "small_boundary_fragment":
            small,

        "planning_review_needed":
            review,

        "review_flags":
            integrated[
                "review_flags"
            ].fillna("").to_numpy(),

        "coverage_label":
            integrated[
                "coverage_label"
            ].to_numpy(),

        "classified_pct":
            integrated["classified_pct"].to_numpy(),

        "unclassified_pct":
            integrated["unclassified_pct"].to_numpy(),

        "unmapped_pct":
            integrated["unmapped_pct"].to_numpy(),

        "ambiguous_pct":
            integrated["ambiguous_pct"].to_numpy(),

        "conservation_water_pct":
            integrated[
                "conservation_water_pct"
            ].to_numpy(),

        "highland_approx_pct":
            integrated[
                "highland_approx_pct"
            ].to_numpy(),

        "provisional_zone_name":
            integrated[
                "provisional_zone_name"
            ].fillna(
                "NOT_CLASSIFIED"
            ).to_numpy(),

        "model_status":
            "PROVISIONAL_REFERENCE_NOT_AHP",

        "zoning_legal_status":
            "UNVERIFIED",

        "development_permission":
            "NOT_ASSESSED",
    })

    # GN names must come from the real GN boundary
    # GIS overlay in the subsequent integration step.
    # Do not infer GN divisions from a grid ID.

    # ========================================================
    # 8. RELEASE VALIDATION
    # ========================================================

    check(
        "Dashboard records contain 9240 unique IDs",
        len(out) == N
        and out["grid_id"].is_unique,
    )

    check(
        "Top-decile count matches Step 16D-F",
        int(top.sum())
        == f_meta.get(
            "provisional_top_decile_cells"
        ),
    )

    check(
        "Top-decile review overlap matches Step 16D-F",
        int((top & review).sum())
        == f_meta.get(
            "provisional_top_decile_with_review"
        ),
    )

    check(
        "High-stability count matches Step 16D-F",
        int(stable.sum())
        == f_meta.get(
            "high_stability_both_methods_cells"
        ),
    )

    check(
        "Review flag count matches Step 16D-F",
        int(review.sum())
        == f_meta.get(
            "planning_review_cells"
        ),
    )

    check(
        "Study area matches previous integrated dataset",
        abs(
            area.sum() / 10000
            - float(f_meta["total_area_ha"])
        ) < 0.05,
    )

    check(
        "No AHP approval or legal permission asserted",
        out["model_status"].eq(
            "PROVISIONAL_REFERENCE_NOT_AHP"
        ).all()
        and out["development_permission"].eq(
            "NOT_ASSESSED"
        ).all(),
    )

    check(
        "All original inputs unchanged",
        all(
            digest(path) == original_hashes[str(path)]
            for path in inputs
        ),
    )

    checkpoint()

    # ========================================================
    # 9. GIS EXPORT AND ROUND-TRIP TEST
    # ========================================================

    output_gis = gpd.GeoDataFrame(
        out,
        geometry=gpd.GeoSeries(
            geom.to_numpy(),
            crs=gis.crs,
        ),
        crs=gis.crs,
    )

    staging = OUT_GPKG.with_name(
        OUT_GPKG.stem + "_staging.gpkg"
    )

    if staging.exists():
        staging.unlink()

    output_gis.to_file(
        staging,
        layer="suitability_reference_dashboard",
        driver="GPKG",
        index=False,
    )

    saved = gpd.read_file(
        staging,
        layer="suitability_reference_dashboard",
    )

    check(
        "GIS export retains all 9240 grid IDs",
        len(saved) == N
        and saved["grid_id"].is_unique
        and saved.geometry.notna().all()
        and saved.geometry.is_valid.all(),
    )

    check(
        "GIS export preserves suitability scores",
        np.allclose(
            saved["suitability_score"],
            out["suitability_score"],
            atol=1e-10,
        ),
    )

    checkpoint()

    del saved

    os.replace(
        staging,
        OUT_GPKG,
    )

    out.to_csv(
        OUT_CSV,
        index=False,
    )

    # ========================================================
    # 10. BACKEND / FRONTEND DATA CONTRACT
    # ========================================================

    contract = {
        "contract_version": "16D-G-1.0",

        "layer":
            "suitability_reference_dashboard",

        "grid_count": N,

        "crs": "EPSG:32644",

        "coordinate_note": (
            "Convert geometries to EPSG:4326 "
            "when serving GeoJSON for web maps."
        ),

        "model_status":
            "PROVISIONAL_REFERENCE_NOT_AHP",

        "scoring": {
            "formula":
                "(C1 + C2 + C4) / 3",

            "weights": {
                "C1": 1 / 3,
                "C2": 1 / 3,
                "C4": 1 / 3,
            },

            "weight_source":
                "ILLUSTRATIVE_EQUAL_CONTRIBUTION_NOT_EXPERT",

            "score_range": [0, 1],

            "class_type":
                "WITHIN_STUDY_RELATIVE_QUINTILE",

            "no_final_ahp_claim": True,
        },

        "dashboard_field_meanings": {
            "grid_id":
                "Stable original 100m grid identifier",

            "suitability_score":
                "Provisional three-criterion reference score",

            "relative_suitability_class":
                "Relative quintile within Kaduwela",

            "reference_rank":
                "Descending reference score rank",

            "c1_road_proximity":
                "Normalized road-distance score",

            "c2_road_density":
                "Normalized road-density score",

            "c4_slope":
                "Normalized terrain-slope score",

            "limiting_criterion":
                "Lowest standardized core criterion; not a causal SHAP explanation",

            "weight_stability_pctclip":
                "Fraction of 120 exploratory weight scenarios ranking this cell in the top decile",

            "weight_stability_rank":
                "Equivalent frequency under rank normalization",

            "high_stability_both_methods":
                "Top-decile frequency at least 0.75 for both methods",

            "planning_review_needed":
                "At least one provisional screening flag",

            "review_flags":
                "Reasons for provisional planning review",

            "conservation_water_pct":
                "Approximate percentage overlapping provisional conservation/water review layers",

            "provisional_zone_name":
                "Largest-area provisional zoning classification",

            "small_boundary_fragment":
                "Original clipped grid area below 5000m2; not a parcel or minimum legal plot threshold",

            "c3_building_context_raw_pct":
                "Raw context only; excluded from reference scoring",

            "c5_elevation_raw_m":
                "Raw elevation only; excluded from reference scoring",
        },

        "important_limitations": [
            "Not expert-validated AHP",
            "Not a development approval tool",
            "Provisional zoning remains legally unverified",
            "No automatic conservation exclusions",
            "Grid cells are not individual land parcels",
            "GN division labels are not yet joined",
            "No external ground-truth suitability validation claimed",
        ],
    }

    OUT_CONTRACT.write_text(
        json.dumps(
            contract,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(checks).to_csv(
        OUT_QA,
        index=False,
    )

    # ========================================================
    # 11. SUMMARY
    # ========================================================

    summary = {
        "step": "16D-G",

        "qa_status": "PASS",

        "release_status":
            "DASHBOARD_READY_PROVISIONAL_REFERENCE",

        "generated_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "grid_count": N,

        "area_ha":
            float(area.sum() / 10000),

        "top_decile_cells":
            int(top.sum()),

        "top_decile_with_review":
            int((top & review).sum()),

        "top_decile_review_pct":
            float(
                100
                * (top & review).sum()
                / top.sum()
            ),

        "high_stability_both_methods":
            int(stable.sum()),

        "small_boundary_fragments":
            int(small.sum()),

        "expert_ahp_status_at_run":
            d_meta.get(
                "expert_input_status"
            ),

        "uses_validated_expert_ahp":
            False,

        "independent_external_validation":
            "NOT_YET_COMPLETED",

        "qa_checks":
            len(checks),

        "qa_passed":
            sum(
                row["passed"]
                for row in checks
            ),

        "original_input_sha256": {
            path.name: original_hashes[str(path)]
            for path in inputs
        },

        "outputs": [
            path.name
            for path in [
                OUT_CSV,
                OUT_GPKG,
                OUT_CONTRACT,
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
    # 12. TERMINAL REPORT
    # ========================================================

    print("\n=== DASHBOARD RELEASE ===")

    print("Grid cells:", N)

    print(
        "Area (ha):",
        round(
            summary["area_ha"],
            4,
        )
    )

    print(
        "Top-decile cells:",
        summary["top_decile_cells"]
    )

    print(
        "Top-decile cells with review flags:",
        summary["top_decile_with_review"]
    )

    print(
        "Top-decile review percentage:",
        round(
            summary["top_decile_review_pct"],
            2,
        )
    )

    print(
        "High-stability cells (both methods):",
        summary["high_stability_both_methods"]
    )

    print(
        "Small boundary fragments:",
        summary["small_boundary_fragments"]
    )

    print(
        "AHP input status:",
        summary["expert_ahp_status_at_run"]
    )

    print("\n=== QA ===")

    for row in checks:

        print(
            "PASS:"
            if row["passed"]
            else "REVIEW:",
            row["check"],
        )

    print(
        f"\nOverall QA: PASS "
        f"({len(checks)}/{len(checks)})"
    )

    print("\n=== OUTPUTS ===")

    for path in [
        OUT_CSV,
        OUT_GPKG,
        OUT_CONTRACT,
        OUT_QA,
        OUT_SUMMARY,
    ]:

        print(
            path.relative_to(ROOT)
        )

    print(
        "\nSTATUS: DASHBOARD-READY PROVISIONAL REFERENCE"
    )

    print(
        "No final AHP, independent external "
        "validation or legal approval claimed."
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
