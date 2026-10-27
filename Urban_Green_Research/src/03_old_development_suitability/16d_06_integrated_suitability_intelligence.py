
"""
STEP 16D-F — CONSOLIDATED PROVISIONAL SUITABILITY INTELLIGENCE

Consolidates existing Step 16D-B/C/E and Step 16C-M results
for 9,240 Kaduwela grid cells.

No new weighting, AHP judgments, legal exclusions,
or final suitability recommendations are generated.
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

REPORTS = ROOT / "outputs" / "reports"

BASE_CSV = (
    REPORTS
    / "16d_b_provisional_suitability_baseline_100m.csv"
)

BASE_GPKG = (
    REPORTS
    / "16d_b_provisional_suitability_baseline_100m.gpkg"
)

SPATIAL_CSV = (
    REPORTS
    / "16d_c_grid_spatial_diagnostics_100m.csv"
)

WEIGHT_CSV = (
    REPORTS
    / "16d_e_grid_weight_robustness_100m.csv"
)

WEIGHT_METHODS = (
    REPORTS
    / "16d_e_weight_sensitivity_method_summary.csv"
)

ZONING_CSV = (
    REPORTS
    / "16c_m_grid_zoning_screening_100m.csv"
)

BASE_META = (
    REPORTS
    / "16d_b_provisional_baseline_summary.json"
)

SPATIAL_META = (
    REPORTS
    / "16d_c_spatial_baseline_summary.json"
)

WEIGHT_META = (
    REPORTS
    / "16d_e_weight_sensitivity_summary.json"
)

ZONING_META = (
    REPORTS
    / "16c_m_grid_zoning_screening_summary.json"
)

AHP_META = (
    REPORTS
    / "16d_d_ahp_consistency_summary.json"
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

OUT_CSV = (
    REPORTS
    / "16d_f_integrated_suitability_intelligence_100m.csv"
)

OUT_GPKG = (
    REPORTS
    / "16d_f_integrated_suitability_intelligence_100m.gpkg"
)

OUT_CLASSES = (
    REPORTS
    / "16d_f_class_intelligence_summary.csv"
)

OUT_TOP = (
    REPORTS
    / "16d_f_top_decile_planning_review_register.csv"
)

OUT_QA = (
    REPORTS
    / "16d_f_integration_qa.csv"
)

OUT_META = (
    REPORTS
    / "16d_f_integration_summary.json"
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

N = 9240

# Fraction of tested weight scenarios.
# This is NOT a probability or legal threshold.
STABILITY_CUTOFF = 0.75


# ============================================================
# 4. HELPERS
# ============================================================

def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):

            h.update(chunk)

    return h.hexdigest()


def load_csv(path):

    df = pd.read_csv(
        path,
        dtype={"grid_id": "string"},
    )

    if "grid_id" in df.columns:

        df["grid_id"] = (
            df["grid_id"].str.strip()
        )

    return df


def boolean_values(series):

    text = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    accepted = text.isin([
        "true",
        "false",
        "1",
        "0",
    ])

    if not bool(accepted.all()):

        raise ValueError(
            f"Unrecognized boolean values "
            f"in {series.name}"
        )

    return text.isin([
        "true",
        "1",
    ]).to_numpy(dtype=bool)


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 88)

    print(
        "STEP 16D-F - INTEGRATED PROVISIONAL "
        "SUITABILITY INTELLIGENCE"
    )

    print("=" * 88)

    inputs = [
        BASE_CSV,
        BASE_GPKG,
        SPATIAL_CSV,
        WEIGHT_CSV,
        WEIGHT_METHODS,
        ZONING_CSV,
        BASE_META,
        SPATIAL_META,
        WEIGHT_META,
        ZONING_META,
        AHP_META,
    ]

    for path in inputs:

        if not path.is_file():

            raise FileNotFoundError(
                f"Required input missing: {path}"
            )

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    digests = {
        str(path): sha256(path)
        for path in inputs
    }

    b_meta, c_meta, e_meta, m_meta, d_meta = [

        json.loads(
            p.read_text(
                encoding="utf-8"
            )
        )

        for p in [
            BASE_META,
            SPATIAL_META,
            WEIGHT_META,
            ZONING_META,
            AHP_META,
        ]
    ]

    checks = []

    def check(name, success, details=""):

        checks.append({
            "check": name,
            "passed": bool(success),
            "detail": str(details),
        })

    def validate():

        failures = [
            r
            for r in checks
            if not r["passed"]
        ]

        if failures:

            pd.DataFrame(checks).to_csv(
                OUT_QA,
                index=False,
            )

            for r in failures:

                print(
                    "REVIEW:",
                    r["check"],
                    r["detail"],
                )

            raise RuntimeError(
                "Step 16D-F QA needs review; "
                "see QA CSV. No inputs changed."
            )

    # ========================================================
    # 6. VERIFY PREVIOUS STEPS
    # ========================================================

    check(
        "16D-B/C/E and 16C-M passed technical QA",

        all(
            x.get("qa_status") == "PASS"

            for x in (
                b_meta,
                c_meta,
                e_meta,
                m_meta,
            )
        ),
    )

    check(
        "AHP intake technical QA passed",

        d_meta.get("technical_qa_status")
        == "PASS",
    )

    check(
        "Existing results are explicitly provisional",

        b_meta.get("methodology_status")
        == "PROVISIONAL_REFERENCE_BASELINE_ONLY"

        and e_meta.get("methodology_status")
        == "NON_EXPERT_EXPLORATORY_WEIGHT_SENSITIVITY_ONLY",
    )

    check(
        "Inputs reference 9240 grid cells",

        b_meta.get("grid_count")
        == c_meta.get("grid_count")
        == e_meta.get("grid_cells")
        == m_meta.get("grid_count")
        == N,
    )

    check(
        "120 sensitivity weight vectors per normalization",

        e_meta.get("unique_weight_vectors")
        == 120

        and e_meta.get(
            "total_weight_method_scenarios"
        ) == 240,
    )

    validate()

    # ========================================================
    # 7. LOAD COMPLETED RESEARCH OUTPUTS
    # ========================================================

    print(
        "\nReading and aligning completed "
        "research outputs..."
    )

    base = load_csv(BASE_CSV)

    spatial = load_csv(SPATIAL_CSV)

    weight = load_csv(WEIGHT_CSV)

    zoning = load_csv(ZONING_CSV)

    methods = pd.read_csv(
        WEIGHT_METHODS
    )

    gis = gpd.read_file(
        BASE_GPKG,
        layer="provisional_suitability_baseline",
    )

    gis["grid_id"] = (
        gis["grid_id"]
        .astype("string")
        .str.strip()
    )

    for label, data in [
        ("baseline", base),
        ("spatial", spatial),
        ("weight sensitivity", weight),
        ("zoning", zoning),
        ("GIS", gis),
    ]:

        check(
            f"{label} has 9240 unique non-null grid IDs",

            len(data) == N

            and data["grid_id"].notna().all()

            and data["grid_id"].is_unique,
        )

    validate()

    check(
        "All source grid-ID sets identical",

        all(
            set(data["grid_id"])
            == set(base["grid_id"])

            for data in (
                spatial,
                weight,
                zoning,
                gis,
            )
        ),
    )

    check(
        "GIS layer is valid EPSG:32644",

        gis.crs is not None

        and gis.crs.to_epsg() == 32644

        and gis.geometry.notna().all()

        and (
            ~gis.geometry.is_empty
        ).all()

        and gis.geometry.is_valid.all(),
    )

    validate()

    # ========================================================
    # 8. ALIGN DATA BY GRID ID
    # ========================================================

    def aligned(df):

        return (
            df
            .set_index("grid_id")
            .reindex(base["grid_id"])
        )

    sp = aligned(spatial)

    wt = aligned(weight)

    zn = aligned(zoning)

    geom = aligned(gis).geometry

    # ========================================================
    # 9. CHECK REQUIRED FIELDS
    # ========================================================

    required_base = [
        "baseline_pc",
        "baseline_rank",
        "relative_class",
        "top_decile_pc",
        "top_decile_rank",
        "cell_area_m2",
        "small_boundary_fragment",
        "c1_pc",
        "c2_pc",
        "c4_pc",
        "c1_reference_contribution",
        "c2_reference_contribution",
        "c4_reference_contribution",
        "review_needed",
        "zoning_legal_status",
        "legal_permission",
    ]

    required_weight = [
        "baseline_pc_equal_reference",
        "baseline_rank_equal_reference",
        "pctclip_top10_frequency",
        "rank_top10_frequency",
        "pctclip_top10_in_every_scenario",
        "rank_top10_in_every_scenario",
        "pctclip_top10_frequency_ge_75pct",
        "rank_top10_frequency_ge_75pct",
        "planning_review",
        "small_boundary_fragment",
        "pctclip_score_min",
        "pctclip_score_max",
        "rank_score_min",
        "rank_score_max",
    ]

    required_spatial = [
        "rook_neighbor_count",
        "nominal_1km_block",
        "top_decile_both_methods",
        "planning_review",
    ]

    required_zoning = [
        "classified_pct",
        "unclassified_pct",
        "unmapped_pct",
        "conservation_water_pct",
        "ambiguous_pct",
        "highland_approx_pct",
        "coverage_label",
        "review_flags",
        "review_needed",
        "main_zone_zone_name",
        "zoning_legal_status",
    ]

    check(
        "All required baseline fields exist",

        set(required_base).issubset(
            base.columns
        ),
    )

    check(
        "All required weight-sensitivity fields exist",

        set(required_weight).issubset(
            wt.columns
        ),
    )

    check(
        "All required spatial-diagnostic fields exist",

        set(required_spatial).issubset(
            sp.columns
        ),
    )

    check(
        "All required planning-screening fields exist",

        set(required_zoning).issubset(
            zn.columns
        ),
    )

    check(
        "All polygon geometries aligned",

        geom.notna().all(),
    )

    validate()

    # ========================================================
    # 10. NUMERIC DATA HELPER
    # ========================================================

    def nums(df, names):

        a = (
            df[names]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(dtype=float)
        )

        if not np.isfinite(a).all():

            raise ValueError(
                "Missing/nonfinite numeric fields: "
                f"{names}"
            )

        return a

    # ========================================================
    # 11. PREPARE MODEL AND SCREENING VALUES
    # ========================================================

    scores = nums(
        base,
        [
            "baseline_pc",
            "baseline_rank",
            "c1_pc",
            "c2_pc",
            "c4_pc",
        ],
    )

    contributions = nums(
        base,
        [
            "c1_reference_contribution",
            "c2_reference_contribution",
            "c4_reference_contribution",
        ],
    )

    frequencies = nums(
        wt,
        [
            "pctclip_top10_frequency",
            "rank_top10_frequency",
        ],
    )

    area = nums(
        base,
        ["cell_area_m2"],
    )[:, 0]

    zones = nums(
        zn,
        [
            "classified_pct",
            "unclassified_pct",
            "unmapped_pct",
        ],
    )

    review = boolean_values(
        base["review_needed"]
    )

    wt_review = boolean_values(
        wt["planning_review"]
    )

    sp_review = boolean_values(
        sp["planning_review"]
    )

    zone_review = boolean_values(
        zn["review_needed"]
    )

    small = boolean_values(
        base["small_boundary_fragment"]
    )

    top_pc = boolean_values(
        base["top_decile_pc"]
    )

    top_rank = boolean_values(
        base["top_decile_rank"]
    )

    # ========================================================
    # 12. CROSS-SOURCE VALIDATION
    # ========================================================

    check(
        "Scores and sensitivity frequencies within 0 to 1",

        np.all(
            (scores >= 0)
            & (scores <= 1)
        )

        and np.all(
            (frequencies >= 0)
            & (frequencies <= 1)
        ),
    )

    check(
        "Equal-contribution formula reproduced",

        np.allclose(
            scores[:, 0],
            contributions.sum(axis=1),
            atol=1e-10,
        )

        and np.allclose(
            scores[:, 0],
            scores[:, 2:5].mean(axis=1),
            atol=1e-10,
        ),
    )

    check(
        "Sensitivity references reproduce baseline",

        np.allclose(
            scores[:, 0],

            nums(
                wt,
                ["baseline_pc_equal_reference"],
            )[:, 0],

            atol=1e-10,
        )

        and np.allclose(
            scores[:, 1],

            nums(
                wt,
                ["baseline_rank_equal_reference"],
            )[:, 0],

            atol=1e-10,
        ),
    )

    check(
        "Planning review flags agree across all research steps",

        np.array_equal(
            review,
            wt_review,
        )

        and np.array_equal(
            review,
            sp_review,
        )

        and np.array_equal(
            review,
            zone_review,
        ),
    )

    check(
        "Weight scenario stability flags reproduce frequencies",

        np.array_equal(
            boolean_values(
                wt["pctclip_top10_frequency_ge_75pct"]
            ),

            frequencies[:, 0]
            >= STABILITY_CUTOFF,
        )

        and np.array_equal(
            boolean_values(
                wt["rank_top10_frequency_ge_75pct"]
            ),

            frequencies[:, 1]
            >= STABILITY_CUTOFF,
        )

        and np.array_equal(
            boolean_values(
                wt["pctclip_top10_in_every_scenario"]
            ),

            frequencies[:, 0] == 1,
        )

        and np.array_equal(
            boolean_values(
                wt["rank_top10_in_every_scenario"]
            ),

            frequencies[:, 1] == 1,
        ),
    )

    check(
        "High-decile agreement reproduces spatial diagnostics",

        np.array_equal(
            top_pc & top_rank,

            boolean_values(
                sp["top_decile_both_methods"]
            ),
        ),
    )

    check(
        "Zoning coverage fractions sum to 100%",

        np.allclose(
            zones.sum(axis=1),
            100,
            atol=0.001,
        ),
    )

    check(
        "Zoning is legally unverified; permissions remain unassessed",

        base["zoning_legal_status"]
        .eq("UNVERIFIED")
        .all()

        and zn["zoning_legal_status"]
        .eq("UNVERIFIED")
        .all()

        and base["legal_permission"]
        .eq("NOT_ASSESSED")
        .all(),
    )

    check(
        "Polygon areas match recorded clipped-cell areas",

        np.allclose(
            geom.area.to_numpy(dtype=float),
            area,
            atol=0.01,
        ),
    )

    check(
        "Whole study area maintained",

        abs(
            area.sum() / 10000

            - float(
                c_meta["study_area_ha"]
            )
        ) < 0.05,
    )

    check(
        "Sensitivity file matches recorded methods",

        set(
            methods["normalization_method"]
        ) == {
            "pctclip",
            "rank",
        }

        and all(
            len(
                methods[
                    methods["normalization_method"]
                    == x
                ]
            ) == 1

            for x in (
                "pctclip",
                "rank",
            )
        ),
    )

    validate()

    # ========================================================
    # 13. CREATE INTEGRATED GRID DATASET
    # ========================================================

    print(
        "Consolidating explanatory scores, "
        "sensitivity, and provisional screening..."
    )

    out = pd.DataFrame({

        "grid_id":
            base["grid_id"].to_numpy(),

        "reference_score":
            scores[:, 0],

        "rank_method_score":
            scores[:, 1],

        "relative_class":
            base["relative_class"].to_numpy(),

        "reference_rank_desc":
            pd.Series(
                scores[:, 0]
            ).rank(
                method="min",
                ascending=False,
            ).astype(int).to_numpy(),

        "reference_top_decile":
            top_pc,

        "both_methods_top_decile":
            top_pc & top_rank,

        "c1_road_proximity_score":
            scores[:, 2],

        "c2_road_density_score":
            scores[:, 3],

        "c4_slope_score":
            scores[:, 4],

        "c1_equal_reference_contribution":
            contributions[:, 0],

        "c2_equal_reference_contribution":
            contributions[:, 1],

        "c4_equal_reference_contribution":
            contributions[:, 2],

        "weakest_baseline_criterion":
            np.array([
                "C1_ROAD_DISTANCE",
                "C2_ROAD_DENSITY",
                "C4_SLOPE",
            ])[
                np.argmin(
                    scores[:, 2:5],
                    axis=1,
                )
            ],

        "top10_frequency_pctclip":
            frequencies[:, 0],

        "top10_frequency_rank":
            frequencies[:, 1],

        "high_stability_both_methods":
            (
                frequencies[:, 0]
                >= STABILITY_CUTOFF
            )

            & (
                frequencies[:, 1]
                >= STABILITY_CUTOFF
            ),

        "top10_in_all_120_pctclip":
            frequencies[:, 0] == 1,

        "top10_in_all_120_rank":
            frequencies[:, 1] == 1,

        "pctclip_score_min_weight_scenarios":
            nums(
                wt,
                ["pctclip_score_min"],
            )[:, 0],

        "pctclip_score_max_weight_scenarios":
            nums(
                wt,
                ["pctclip_score_max"],
            )[:, 0],

        "rank_score_min_weight_scenarios":
            nums(
                wt,
                ["rank_score_min"],
            )[:, 0],

        "rank_score_max_weight_scenarios":
            nums(
                wt,
                ["rank_score_max"],
            )[:, 0],

        "nominal_1km_block":
            sp["nominal_1km_block"].to_numpy(),

        "rook_neighbor_count":
            nums(
                sp,
                ["rook_neighbor_count"],
            )[:, 0].astype(int),

        "cell_area_m2":
            area,

        "small_boundary_fragment":
            small,

        "planning_review_needed":
            review,

        "review_flags":
            zn["review_flags"].to_numpy(),

        "coverage_label":
            zn["coverage_label"].to_numpy(),

        "classified_pct":
            zones[:, 0],

        "unclassified_pct":
            zones[:, 1],

        "unmapped_pct":
            zones[:, 2],

        "ambiguous_pct":
            nums(
                zn,
                ["ambiguous_pct"],
            )[:, 0],

        "conservation_water_pct":
            nums(
                zn,
                ["conservation_water_pct"],
            )[:, 0],

        "highland_approx_pct":
            nums(
                zn,
                ["highland_approx_pct"],
            )[:, 0],

        "provisional_zone_name":
            zn["main_zone_zone_name"]
            .fillna("NOT_CLASSIFIED")
            .to_numpy(),

        "legal_zoning_status":
            "UNVERIFIED",

        "development_permission":
            "NOT_ASSESSED",

        "model_status":
            "PROVISIONAL_REFERENCE_GIS_MCDA_NOT_AHP",
    })

    # Additional available zoning attributes.

    for previous, new in [
        (
            "main_zone_group_name",
            "provisional_zone_group",
        ),

        (
            "main_zone_zone_id",
            "provisional_zone_id",
        ),
    ]:

        if previous in zn.columns:

            out[new] = (
                zn[previous]
                .fillna("NOT_CLASSIFIED")
                .astype(str)
                .to_numpy()
            )

    # ========================================================
    # 14. CLASS-LEVEL INTELLIGENCE SUMMARY
    # ========================================================

    cls = (
        out
        .groupby(
            "relative_class",
            as_index=False,
        )
        .agg(
            cells=(
                "grid_id",
                "size",
            ),

            clipped_area_ha=(
                "cell_area_m2",
                lambda x: x.sum() / 10000,
            ),

            mean_reference_score=(
                "reference_score",
                "mean",
            ),

            planning_review_cells=(
                "planning_review_needed",
                "sum",
            ),

            conservation_water_intersecting_cells=(
                "conservation_water_pct",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),

            high_stability_both_methods_cells=(
                "high_stability_both_methods",
                "sum",
            ),

            small_boundary_fragments=(
                "small_boundary_fragment",
                "sum",
            ),
        )
    )

    cls["planning_review_pct_of_cells"] = (
        100
        * cls["planning_review_cells"]
        / cls["cells"]
    )

    # ========================================================
    # 15. TOP-DECILE PLANNING REVIEW REGISTER
    # ========================================================

    # Contains ALL reference top-decile cells,
    # including cells with review flags.
    #
    # This is NOT an automatically recommended
    # list of buildable development sites.

    top = (
        out.loc[
            out["reference_top_decile"]
        ]
        .sort_values(
            [
                "reference_score",
                "grid_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 16. FINAL DATA VALIDATION
    # ========================================================

    check(
        "Integrated grid covers all unique cells",

        len(out) == N

        and out["grid_id"].is_unique,
    )

    check(
        "Class summaries cover all grid cells",

        cls["cells"].sum() == N,
    )

    check(
        "Top-decile register reproduces original reference mask",

        len(top) == int(
            top_pc.sum()
        ),
    )

    check(
        "Planning-review cell total agrees with prior report",

        int(
            out["planning_review_needed"].sum()
        )

        == b_meta.get("planning_review_cells")

        == m_meta.get("any_review_cells"),
    )

    check(
        "Sensitivity stability summaries reproduce Step 16D-E",

        all(

            int(
                out[
                    f"top10_in_all_120_{name}"
                ].sum()
            )

            == int(
                methods
                .set_index("normalization_method")
                .loc[
                    name,
                    "top10_in_all_scenarios_cells",
                ]
            )

            for name in (
                "pctclip",
                "rank",
            )
        )

        and all(

            int(
                (
                    frequencies[:, i]
                    >= STABILITY_CUTOFF
                ).sum()
            )

            == int(
                methods
                .set_index("normalization_method")
                .loc[
                    name,
                    "top10_frequency_at_least_75pct_cells",
                ]
            )

            for i, name in enumerate(
                (
                    "pctclip",
                    "rank",
                )
            )
        ),
    )

    check(
        "Source input files remain unchanged",

        all(
            sha256(p) == digests[str(p)]
            for p in inputs
        ),
    )

    validate()

    # ========================================================
    # 17. EXPORT GIS LAYER
    # ========================================================

    # Reuse original Step 16D-B geometries.
    # Do not regenerate the Kaduwela grid.

    mapped = gpd.GeoDataFrame(
        out,

        geometry=gpd.GeoSeries(
            geom.to_numpy(),
            crs=gis.crs,
        ),

        crs=gis.crs,
    )

    temporary = OUT_GPKG.with_name(
        OUT_GPKG.stem + "_staging.gpkg"
    )

    if temporary.exists():

        temporary.unlink()

    mapped.to_file(
        temporary,
        layer="provisional_suitability_intelligence",
        driver="GPKG",
        index=False,
    )

    saved = gpd.read_file(
        temporary,
        layer="provisional_suitability_intelligence",
    )

    check(
        "GIS round-trip retains 9240 valid polygons and IDs",

        len(saved) == N

        and saved["grid_id"].is_unique

        and saved.geometry.is_valid.all()

        and saved.geometry.notna().all(),
    )

    check(
        "GIS round-trip retains reference and robustness values",

        np.allclose(
            saved["reference_score"],
            out["reference_score"],
            atol=1e-10,
        )

        and np.allclose(
            saved["top10_frequency_pctclip"],
            out["top10_frequency_pctclip"],
            atol=1e-10,
        ),
    )

    validate()

    del saved

    os.replace(
        temporary,
        OUT_GPKG,
    )

    # ========================================================
    # 18. SAVE TABULAR OUTPUTS
    # ========================================================

    out.to_csv(
        OUT_CSV,
        index=False,
    )

    cls.to_csv(
        OUT_CLASSES,
        index=False,
    )

    top.to_csv(
        OUT_TOP,
        index=False,
    )

    qa = pd.DataFrame(
        checks
    )

    qa.to_csv(
        OUT_QA,
        index=False,
    )

    # ========================================================
    # 19. SAVE RESEARCH SUMMARY
    # ========================================================

    summary = {
        "step": "16D-F",

        "qa_status": "PASS",

        "methodology_status":
            "INTEGRATION_READY_PROVISIONAL_REFERENCE_ONLY",

        "generated_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "grid_count":
            N,

        "total_area_ha":
            float(
                area.sum() / 10000
            ),

        "reference_criteria": [
            "C1",
            "C2",
            "C4",
        ],

        "equal_contribution_reference_not_expert_AHP":
            True,

        "expert_intake_status_at_run":
            d_meta.get(
                "expert_input_status"
            ),

        "final_AHP_weights_used":
            False,

        "legal_exclusions_applied":
            False,

        "high_stability_cutoff":
            STABILITY_CUTOFF,

        "high_stability_cutoff_meaning":
            (
                "Fraction of tested scenario weight "
                "vectors; not a probability"
            ),

        "provisional_top_decile_cells":
            int(
                top_pc.sum()
            ),

        "provisional_top_decile_with_review":
            int(
                (
                    top_pc & review
                ).sum()
            ),

        "high_stability_both_methods_cells":
            int(
                out[
                    "high_stability_both_methods"
                ].sum()
            ),

        "planning_review_cells":
            int(
                review.sum()
            ),

        "qa_checks":
            len(checks),

        "qa_passed":
            int(
                qa["passed"].sum()
            ),

        "inputs_sha256": {
            p.name: digests[str(p)]
            for p in inputs
        },

        "limitations": [
            (
                "Scores are provisional and not "
                "validated AHP weights or final suitability."
            ),

            (
                "Sensitivity frequencies are fractions "
                "of exploratory weight vectors, "
                "not probabilities."
            ),

            (
                "Review flags and screenshot-derived "
                "zoning are not official legal "
                "restrictions or permits."
            ),

            (
                "Current high rankings do not establish "
                "vacant buildable parcels."
            ),

            (
                "C3 direction, C5 inclusion, and "
                "actual expert judgments remain unresolved."
            ),

            (
                "GN division names are not inferred "
                "from grid identifiers."
            ),
        ],

        "outputs": [
            p.name

            for p in [
                OUT_CSV,
                OUT_GPKG,
                OUT_CLASSES,
                OUT_TOP,
                OUT_QA,
                OUT_META,
            ]
        ],
    }

    OUT_META.write_text(
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

    print(
        "\n=== CONSOLIDATED PROVISIONAL RESULTS ==="
    )

    print(
        "Grid cells:",
        len(out)
    )

    print(
        "Area (ha):",
        round(
            summary["total_area_ha"],
            4,
        )
    )

    print(
        "Reference top-decile cells:",
        summary["provisional_top_decile_cells"]
    )

    print(
        "Top-decile cells with planning-review flags:",
        summary["provisional_top_decile_with_review"]
    )

    print(
        "High stability under both normalizations:",
        summary["high_stability_both_methods_cells"]
    )

    print(
        "\n=== CLASS INTELLIGENCE ==="
    )

    print(
        cls[
            [
                "relative_class",
                "cells",
                "planning_review_cells",
                "high_stability_both_methods_cells",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== QA ==="
    )

    for record in checks:

        print(
            "PASS:"
            if record["passed"]
            else "REVIEW:",

            record["check"],
        )

    print(
        f"Overall QA: PASS "
        f"({len(checks)}/{len(checks)})"
    )

    print(
        "\n=== OUTPUTS ==="
    )

    for path in [
        OUT_CSV,
        OUT_GPKG,
        OUT_CLASSES,
        OUT_TOP,
        OUT_QA,
        OUT_META,
    ]:

        print(
            path.relative_to(ROOT)
        )

    print(
        "\nNo final AHP model or legal "
        "planning decision created."
    )

    print("=" * 88)


# ============================================================
# 21. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
