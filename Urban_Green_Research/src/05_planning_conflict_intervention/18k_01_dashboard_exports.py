
"""
J26-DS-338 | Section 05 | Step 5.11 (18K-01)
BACKEND-READY DASHBOARD DATA EXPORTS

Uses ONLY final released Section 05 GeoPackage and its audited metadata.
Creates flat CSV records and WGS84 GeoJSON map layers.
Does not train, infer, recalculate indices, or assign legal classes.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import force_2d


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/planning_conflict_intervention"
REPORT = ROOT / "outputs/reports/planning_conflict_intervention"
DASH = ROOT / "outputs/dashboard/planning_conflict_intervention"

FINAL_GIS = DATA / "final_gis/18j_planning_conflict_final.gpkg"
FINAL_QA = REPORT / "18j_planning_conflict_final_gis_qa.json"
SENSITIVITY_QA = REPORT / "18h_planning_review_and_hotspot_sensitivity_qa.json"

GRID_CSV = DASH / "grid_records.csv"
GRID_MAP = DASH / "grid_map.geojson"
GN_CSV = DASH / "gn_summary.csv"
GN_MAP = DASH / "gn_map.geojson"
HOT_CSV = DASH / "hotspot_groups.csv"
HOT_MAP = DASH / "hotspot_map.geojson"
METADATA = DASH / "metadata.json"
MANIFEST = DASH / "manifest.json"
EXPORT_QA = REPORT / "18k_01_dashboard_exports_qa.json"

GRID_N = 9240
GN_N = 57
CLUSTER_N = 673
AREA_KM2 = 88.82921581108494


# ============================================================
# HELPERS
# ============================================================

def check(condition, message):

    passed = bool(condition)

    print(f"[{'PASS' if passed else 'FAIL'}] {message}")

    if not passed:
        raise ValueError(f"Step 18K-01 failed: {message}")


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def rel(path):

    return path.relative_to(ROOT).as_posix()


def num(df, col):

    check(
        col in df.columns,
        f"Required field: {col}",
    )

    return pd.to_numeric(
        df[col],
        errors="coerce",
    ).to_numpy(dtype=float)


def check_ids(df, key, expected, label):

    check(
        key in df
        and len(df) == expected
        and df[key].notna().all()
        and df[key].is_unique,
        f"{label}: {expected:,} unique identifiers",
    )


def bool_values(df, col, expected_domain=None):

    check(
        col in df.columns,
        f"Boolean field: {col}",
    )

    raw = (
        df[col]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    check(
        raw.dropna().isin(
            ["true", "false", "1", "0", "1.0", "0.0"]
        ).all(),
        f"{col}: boolean/null values only",
    )

    if expected_domain is not None:

        check(
            np.array_equal(
                raw.notna().to_numpy(),
                expected_domain,
            ),
            f"{col}: unavailable values preserved",
        )

    return raw.isin(
        ["true", "1", "1.0"]
    ).to_numpy(dtype=bool)


def to_wgs84(gdf):

    converted = gdf.copy()

    converted["geometry"] = (
        converted.geometry.map(force_2d)
    )

    return converted.to_crs("EPSG:4326")


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():

    print("=" * 76)
    print(
        "SECTION 05 | STEP 5.11 | "
        "BACKEND-READY DASHBOARD EXPORTS"
    )
    print("=" * 76)

    sources = (
        FINAL_GIS,
        FINAL_QA,
        SENSITIVITY_QA,
    )

    outputs = (
        GRID_CSV,
        GRID_MAP,
        GN_CSV,
        GN_MAP,
        HOT_CSV,
        HOT_MAP,
        METADATA,
        MANIFEST,
        EXPORT_QA,
    )

    for p in sources:
        if not p.is_file():
            raise FileNotFoundError(p)

    for p in outputs:
        if p.exists():
            raise FileExistsError(
                f"Not overwriting previous output: {p}"
            )

    source_hashes = {
        rel(p): sha256(p)
        for p in sources
    }

    # ========================================================
    # 1. VERIFY FINAL RESEARCH RELEASE
    # ========================================================

    print("\n[1] VERIFY FINAL GIS RELEASE")

    qa = json.loads(
        FINAL_QA.read_text(encoding="utf-8")
    )

    sens = json.loads(
        SENSITIVITY_QA.read_text(
            encoding="utf-8"
        )
    )

    check(
        qa.get("status")
        == (
            "PASS_SECTION05_GIS_PACKAGE_WITH_"
            "DOCUMENTED_LIMITATIONS_DASHBOARD_PENDING"
        ),
        "Section 05 final GIS QA passed",
    )

    check(
        sens.get("status")
        == (
            "PASS_REVIEW_AND_SPATIAL_HOTSPOT_"
            "SENSITIVITY_WITH_BASELINE_REPRODUCTION"
        ),
        "Combined sensitivity release passed",
    )

    check(
        qa["output"]["sha256"]
        == sha256(FINAL_GIS),
        "Final GeoPackage matches release hash",
    )

    check(
        qa["inputs_sha256"].get(
            rel(SENSITIVITY_QA)
        ) == sha256(SENSITIVITY_QA),
        "Final package references exact sensitivity report",
    )

    check(
        set(qa["output"]["layers"]) == {
            "planning_grid_100m",
            "gn_planning_summary",
            "exploratory_spatial_groups",
        },
        "Final three-layer GIS contract verified",
    )

    # ========================================================
    # 2. READ FINAL GIS LAYERS
    # ========================================================

    print("\n[2] READ AND VERIFY RELEASED GIS LAYERS")

    grid = gpd.read_file(
        FINAL_GIS,
        layer="planning_grid_100m",
    )

    gn = gpd.read_file(
        FINAL_GIS,
        layer="gn_planning_summary",
    )

    groups = gpd.read_file(
        FINAL_GIS,
        layer="exploratory_spatial_groups",
    )

    for data, key, n, label in (

        (
            grid,
            "grid_id",
            GRID_N,
            "Planning grids",
        ),

        (
            gn,
            "gn_id",
            GN_N,
            "GN divisions",
        ),

        (
            groups,
            "cluster_id",
            CLUSTER_N,
            "Spatial groups",
        ),

    ):

        check_ids(
            data,
            key,
            n,
            label,
        )

        check(
            data.crs is not None
            and data.crs.to_epsg() == 32644,
            f"{label}: source EPSG:32644",
        )

        check(
            data.geometry.notna().all()
            and not data.geometry.is_empty.any()
            and data.geometry.is_valid.all(),
            f"{label}: geometries valid",
        )

    # ========================================================
    # 3. VALIDATE FROZEN MODEL OUTPUTS
    # ========================================================

    pressure = num(
        grid,
        "urbanization_pressure_score",
    )

    suitability = num(
        grid,
        "development_suitability_score",
    )

    dci = num(
        grid,
        "development_conflict_index_0_1",
    )

    cp = num(
        grid,
        "frozen_conservation_priority_0_1",
    )

    pvalid = np.isfinite(pressure)
    cpvalid = np.isfinite(cp)

    check(
        pvalid.sum() == 4620
        and cpvalid.sum() == 4609,
        "Pressure and conservation valid domains intact",
    )

    check(
        np.array_equal(
            np.isfinite(dci),
            pvalid,
        )
        and np.all(
            ~cpvalid | pvalid
        ),
        "Missingness preserved; no out-of-domain conflict",
    )

    check(
        np.isfinite(suitability).all()
        and np.all(
            (suitability >= 0)
            & (suitability <= 1)
        )
        and np.allclose(
            dci[pvalid],
            pressure[pvalid]
            * (1 - suitability[pvalid]),
            atol=1e-12,
            rtol=0,
        ),
        "Final expert-AHP suitability and DCI reproduced",
    )

    check(
        np.isclose(
            grid.geometry.area.sum() / 1e6,
            AREA_KM2,
            atol=0.001,
            rtol=0,
        ),
        "Final grid study area intact",
    )

    check(
        int(
            grid.review_rule_id.eq(
                "R02_JOINT_CONCERN"
            ).sum()
        ) == 7
        and int(
            grid.review_rule_id.eq(
                "R05_MANAGED_GROWTH_FEASIBILITY"
            ).sum()
        ) == 1,
        "Original exploratory planning review signals intact",
    )

    # ========================================================
    # 4. VALIDATE GN COVERAGE
    # ========================================================

    check(
        int(
            grid.gn_assignment_status.eq(
                "GN_BOUNDARY_GAP_UNASSIGNED"
            ).sum()
        ) == 206,
        "206 unassigned GN grids remain explicit",
    )

    check(
        grid.loc[
            grid.gn_assignment_status.eq(
                "GN_BOUNDARY_GAP_UNASSIGNED"
            ),
            "gn_name",
        ].isna().all(),
        "No fabricated GN names",
    )

    check(
        abs(
            float(qa["gn_boundary_uncovered_km2"])
            - 2.133336
        ) < 1e-5,
        "GN coverage gap retained in release metadata",
    )

    # ========================================================
    # 5. VALIDATE HOTSPOTS
    # ========================================================

    hot = bool_values(
        groups,
        "is_candidate_hotspot",
    )

    check(
        int(hot.sum()) == 58
        and int(
            (
                hot
                & groups.index_type.eq(
                    "DEVELOPMENT_CONFLICT"
                ).to_numpy()
            ).sum()
        ) == 36
        and int(
            (
                hot
                & groups.index_type.eq(
                    "CONSERVATION_PRIORITY"
                ).to_numpy()
            ).sum()
        ) == 22,
        "58 baseline candidates: 36 DCI and 22 CP",
    )

    dc_hot = bool_values(
        grid,
        "dci_candidate_hotspot",
        pvalid,
    )

    cp_hot = bool_values(
        grid,
        "cp_candidate_hotspot",
        cpvalid,
    )

    check(
        int(dc_hot.sum()) == 135
        and int(cp_hot.sum()) == 81,
        "Baseline hotspot cell membership preserved",
    )

    # ========================================================
    # 6. PREPARE MAP DATA
    # ========================================================

    print(
        "\n[3] PREPARE FRONTEND MAP FIELD SELECTIONS"
    )

    grid_map_fields = [
        "grid_id",
        "gn_id",
        "gn_name",
        "gn_assignment_status",
        "gn_covered_fraction",
        "pressure_domain_status",
        "review_rule_id",
        "review_stream",
        "urbanization_pressure_score",
        "pressure_threshold_flag_frozen",
        "development_suitability_score",
        "development_conflict_index_0_1",
        "frozen_conservation_priority_0_1",
        "dci_candidate_hotspot",
        "cp_candidate_hotspot",
        "dci_cluster_id",
        "cp_cluster_id",
        "review_rule_changes_in_8_stress_scenarios",
        "dci_candidate_hotspot_presence_in_9_scenarios",
        "cp_candidate_hotspot_presence_in_9_scenarios",
        "geometry",
    ]

    gn_map_fields = [
        "gn_id",
        "gn_name",
        "gn_area_inside_kmc_km2",
        "mapped_grid_area_km2",
        "suitability_area_weighted_mean",
        "pressure_area_weighted_mean",
        "dci_area_weighted_mean",
        "dci_valid_area_km2",
        "conservation_area_weighted_mean",
        "conservation_valid_area_km2",
        "dci_hotspot_gn_area_ha",
        "cp_hotspot_gn_area_ha",
        "dci_hotspot_share_of_gn_pct",
        "cp_hotspot_share_of_gn_pct",
        "geometry",
    ]

    hotspot_map_fields = [
        "cluster_id",
        "index_type",
        "screen_status",
        "is_candidate_hotspot",
        "grid_count",
        "area_ha",
        "area_km2",
        "mean_screening_score_area_weighted",
        "dominant_intersected_gn_name",
        "intersected_gn_names",
        "gn_uncovered_area_m2",
        "geometry",
    ]

    for frame, fields, name in (

        (
            grid,
            grid_map_fields,
            "Grid map",
        ),

        (
            gn,
            gn_map_fields,
            "GN map",
        ),

        (
            groups,
            hotspot_map_fields,
            "Hotspot map",
        ),

    ):

        check(
            set(fields).issubset(
                frame.columns
            ),
            f"{name}: required map properties exist",
        )

    grid_map = to_wgs84(
        grid[grid_map_fields]
    )

    gn_map = to_wgs84(
        gn[gn_map_fields]
    )

    # Map only candidate hotspots.
    # Keep all 673 spatial groups in CSV.
    hot_map = to_wgs84(
        groups.loc[
            hot,
            hotspot_map_fields,
        ].copy()
    )

    for data, label in (
        (grid_map, "Grid"),
        (gn_map, "GN"),
        (hot_map, "Hotspot"),
    ):

        check(
            data.crs.to_epsg() == 4326
            and data.geometry.is_valid.all(),
            f"{label} GeoJSON prepared in EPSG:4326",
        )

    # ========================================================
    # 7. STAGE DASHBOARD OUTPUTS
    # ========================================================

    print(
        "\n[4] STAGE CSV, GEOJSON AND DOCUMENTED METADATA"
    )

    DASH.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPORT_QA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix="18k_build_",
        dir=DASH,
    ) as directory:

        tmp = Path(directory)

        temp = {
            p: tmp / p.name
            for p in outputs
            if p != EXPORT_QA
        }

        # Full tabular research records
        grid.drop(
            columns="geometry"
        ).to_csv(
            temp[GRID_CSV],
            index=False,
        )

        gn.drop(
            columns="geometry"
        ).to_csv(
            temp[GN_CSV],
            index=False,
        )

        groups.drop(
            columns="geometry"
        ).to_csv(
            temp[HOT_CSV],
            index=False,
        )

        # Lightweight spatial map properties
        for layer, dest in (

            (
                grid_map,
                GRID_MAP,
            ),

            (
                gn_map,
                GN_MAP,
            ),

            (
                hot_map,
                HOT_MAP,
            ),

        ):

            layer.to_file(
                temp[dest],
                driver="GeoJSON",
                engine="pyogrio",
            )

        # ====================================================
        # 8. DOCUMENT DASHBOARD CONTRACT
        # ====================================================

        metadata = {

            "project":
                "J26-DS-338",

            "section":
                "05_planning_conflict_intervention",

            "release_stage":
                "18j_final_planning_conflict_gis",

            "pressure_origin_year":
                2025,

            "pressure_period":
                "2026-2028",

            "source_crs":
                "EPSG:32644",

            "map_crs":
                "EPSG:4326",

            "counts": {

                "grid_cells":
                    GRID_N,

                "gn_divisions":
                    GN_N,

                "all_screened_spatial_groups":
                    CLUSTER_N,

                "candidate_spatial_groups":
                    58,

                "baseline_dci_candidate_groups":
                    36,

                "baseline_cp_candidate_groups":
                    22,

                "dci_valid_cells":
                    4620,

                "cp_valid_cells":
                    4609,

                "pressure_out_of_domain_cells":
                    4620,

                "gn_unassigned_cells":
                    206,
            },

            "gn_uncovered_area_km2":
                float(
                    qa["gn_boundary_uncovered_km2"]
                ),

            "score_meanings": {

                "urbanization_pressure_score": (
                    "Frozen model-derived relative score; "
                    "NOT a calibrated probability"
                ),

                "development_suitability_score": (
                    "Continuous expert-AHP GIS-MCDA using "
                    "road proximity, local road density and slope"
                ),

                "development_conflict_index_0_1": (
                    "Pressure * (1 - suitability), "
                    "only in pressure-model domain"
                ),

                "frozen_conservation_priority_0_1": (
                    "Frozen Section04 conservation priority "
                    "(CCI); already includes pressure"
                ),
            },

            "hotspot_screening": {

                "rank_cutoff": (
                    "Top 10% within each index's valid domain"
                ),

                "minimum_edge_connected_cells":
                    3,

                "geojson_contains": (
                    "Only the 58 baseline candidate hotspots; "
                    "CSV retains all 673 screened groups"
                ),

                "statistical_significance_tested":
                    False,
            },

            "sensitivity": {

                "review_joint_concern_range":
                    qa[
                        "planning_rule_sensitivity_joint_concern_range"
                    ],

                "review_feasibility_range":
                    qa[
                        "planning_rule_sensitivity_feasibility_range"
                    ],

                "dci_candidate_hotspot_count_range":
                    qa[
                        "sensitivity_dci_hotspot_count_range"
                    ],

                "cp_candidate_hotspot_count_range":
                    qa[
                        "sensitivity_cp_hotspot_count_range"
                    ],

                "screening_choices_materially_affect_hotspot_counts":
                    True,
            },

            "limitations": [

                (
                    "Missing scores are unknown/unavailable, "
                    "never zero."
                ),

                (
                    "GN boundaries leave 2.133336 square "
                    "kilometres unattributed."
                ),

                (
                    "Dominant GN name is for display; "
                    "original GN summaries use fractional "
                    "polygon intersection."
                ),

                (
                    "DCI and conservation priority are separate; "
                    "do not add or multiply them together."
                ),

                (
                    "Hotspots are exploratory, parameter-sensitive "
                    "clusters, not statistically significant hotspots."
                ),

                (
                    "Review classes are not official zoning rules "
                    "or development permissions."
                ),

                (
                    "Historical green-to-built detections do not "
                    "prove illegal development."
                ),

                (
                    "Drainage, flood hazard, infrastructure, "
                    "legal restrictions and field checks are "
                    "not resolved by these outputs."
                ),
            ],
        }

        temp[METADATA].write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # ====================================================
        # 9. VERIFY EXPORTS BEFORE RELEASE
        # ====================================================

        print(
            "\n[5] VERIFY EXPORTED FILES BEFORE RELEASE"
        )

        for dest, id_field, expected, label in (

            (
                GRID_CSV,
                "grid_id",
                GRID_N,
                "Grid CSV",
            ),

            (
                GN_CSV,
                "gn_id",
                GN_N,
                "GN CSV",
            ),

            (
                HOT_CSV,
                "cluster_id",
                CLUSTER_N,
                "Spatial group CSV",
            ),

        ):

            df = pd.read_csv(
                temp[dest],
                dtype=(
                    {"gn_id": "string"}
                    if id_field == "gn_id"
                    else None
                ),
            )

            check_ids(
                df,
                id_field,
                expected,
                label,
            )

        for dest, field, expected, label in (

            (
                GRID_MAP,
                "grid_id",
                GRID_N,
                "Grid GeoJSON",
            ),

            (
                GN_MAP,
                "gn_id",
                GN_N,
                "GN GeoJSON",
            ),

            (
                HOT_MAP,
                "cluster_id",
                58,
                "Hotspot GeoJSON",
            ),

        ):

            readback = gpd.read_file(
                temp[dest]
            )

            check_ids(
                readback,
                field,
                expected,
                label,
            )

            check(
                readback.crs is not None
                and readback.crs.to_epsg() == 4326
                and readback.geometry.is_valid.all(),
                f"{label}: valid WGS84 geometries",
            )

            if dest == GRID_MAP:

                check(
                    np.array_equal(
                        np.isfinite(
                            num(
                                readback,
                                "development_conflict_index_0_1",
                            )
                        ),
                        np.isfinite(
                            num(
                                grid_map,
                                "development_conflict_index_0_1",
                            )
                        ),
                    )
                    and np.isfinite(
                        num(
                            readback,
                            "development_conflict_index_0_1",
                        )
                    ).sum() == 4620,
                    (
                        "Grid GeoJSON preserves 4,620 "
                        "DCI values and nulls"
                    ),
                )

                check(
                    np.isfinite(
                        num(
                            readback,
                            "frozen_conservation_priority_0_1",
                        )
                    ).sum() == 4609,
                    (
                        "Grid GeoJSON preserves 4,609 "
                        "conservation values"
                    ),
                )

                check(
                    int(
                        readback.gn_name.isna().sum()
                    ) >= 206,
                    "Unassigned GN map properties stay null",
                )

            if dest == HOT_MAP:

                check(
                    bool_values(
                        readback,
                        "is_candidate_hotspot",
                    ).all(),
                    (
                        "Hotspot GeoJSON excludes "
                        "small screened groups"
                    ),
                )

        check(
            int(
                np.isfinite(
                    num(
                        pd.read_csv(temp[GRID_CSV]),
                        "development_conflict_index_0_1",
                    )
                ).sum()
            ) == 4620,
            "Full grid CSV preserves DCI valid domain",
        )

        check(
            json.loads(
                temp[METADATA].read_text(
                    encoding="utf-8"
                )
            )["counts"]["gn_divisions"] == 57,
            "Metadata roundtrip verified",
        )

        # ====================================================
        # 10. CREATE EXPORT MANIFEST
        # ====================================================

        deliverables = (
            GRID_CSV,
            GRID_MAP,
            GN_CSV,
            GN_MAP,
            HOT_CSV,
            HOT_MAP,
            METADATA,
        )

        manifest = {

            "project":
                "J26-DS-338",

            "section":
                "05_planning_conflict_intervention",

            "release_status": (
                "TECHNICAL_EXPORT_VALIDATED_WITH_"
                "DOCUMENTED_LIMITATIONS"
            ),

            "files": {

                p.name: {

                    "relative_path":
                        rel(p),

                    "sha256":
                        sha256(temp[p]),

                    "bytes":
                        temp[p].stat().st_size,

                }

                for p in deliverables
            },

            "model_inference_performed_for_dashboard_export":
                False,

            "backend_routes_implemented":
                False,
        }

        temp[MANIFEST].write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

        check(
            len(manifest["files"]) == 7,
            "Seven exported content files registered",
        )

        for p in outputs:

            if p.exists():
                raise FileExistsError(
                    f"Output appeared during processing: {p}"
                )

        for p in (*deliverables, MANIFEST):

            temp[p].replace(p)

    # ========================================================
    # 11. VERIFY SOURCE AND OUTPUT INTEGRITY
    # ========================================================

    print("\n[6] VERIFY FINAL SOURCE INTEGRITY")

    for p in sources:

        check(
            sha256(p) == source_hashes[rel(p)],
            f"Unmodified source: {p.name}",
        )

    for filename, info in manifest["files"].items():

        check(
            sha256(DASH / filename)
            == info["sha256"],
            f"Released file hash: {filename}",
        )

    # ========================================================
    # 12. FINAL EXPORT QA
    # ========================================================

    result = {

        "stage":
            "18k_01_dashboard_exports",

        "status": (
            "PASS_SECTION05_BACKEND_READY_"
            "DASHBOARD_EXPORTS_WITH_DOCUMENTED_LIMITATIONS"
        ),

        "source_release":
            qa["status"],

        "source_sha256":
            source_hashes,

        "output_sha256": {

            rel(p): sha256(p)

            for p in (
                *deliverables,
                MANIFEST,
            )
        },

        "counts":
            metadata["counts"],

        "gn_uncovered_area_km2":
            metadata["gn_uncovered_area_km2"],

        "map_crs":
            "EPSG:4326",

        "source_crs":
            "EPSG:32644",

        "scores_modified":
            False,

        "backend_routes_implemented":
            False,

        "review_and_hotspot_limitations":
            metadata["limitations"],
    }

    EXPORT_QA.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 13. TERMINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 76)

    print(
        result["status"]
    )

    print(
        "Grids: 9,240 | GN divisions: 57 | "
        "All spatial groups: 673"
    )

    print(
        "Map candidate hotspots: "
        "58 (36 DCI + 22 CP)"
    )

    print(
        "Valid DCI: 4,620 | Valid CP: 4,609"
    )

    print(
        "GN boundary gap preserved: "
        "2.133336 km2"
    )

    print(
        "Research models/scores changed: NO"
    )

    print(
        "Backend endpoints implemented: "
        "NO - exports ready for wiring"
    )

    print(
        "\nDashboard directory:",
        DASH,
    )

    print(
        "Manifest:",
        MANIFEST,
    )

    print(
        "QA report:",
        EXPORT_QA,
    )

    print("=" * 76)


if __name__ == "__main__":
    main()
