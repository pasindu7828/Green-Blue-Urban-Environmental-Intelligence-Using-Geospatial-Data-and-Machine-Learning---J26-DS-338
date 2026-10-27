
"""
J26-DS-338 | SECTION 05 | STEP 18I-01
GN DIVISION INTEGRATION AND AREA-WEIGHTED SUMMARIES

Inputs:
    57 GN boundaries
    KMC boundary
    Frozen Step 18B grid GIS
    Verified Step 18G planning review queues

Method:
    - Intersect actual grid polygons with GN polygons.
    - Use intersected areas for GN statistics.
    - Assign a dominant GN per grid for map popups.
    - Preserve uncovered GN-boundary gaps.
    - Preserve missing DCI / Conservation Priority.
    - Never use old provisional suitability scores.

Outputs:
    GN-grid fractional intersections CSV
    Grid-to-GN assignment CSV
    GN area-weighted summary CSV
    GN summary GeoPackage
    Reproducibility QA JSON
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import force_2d, union_all


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed"

REPORT = (
    ROOT
    / "outputs/reports/planning_conflict_intervention"
)

GN_SOURCE = (
    DATA
    / "boundaries/gn_divisions/"
    "kaduwela_gn_divisions_utm44n.gpkg"
)

KMC_SOURCE = (
    DATA
    / "boundaries/kmc/"
    "kaduwela_kmc_boundary_utm44n.gpkg"
)

GRID_SOURCE = (
    DATA
    / "planning_conflict_intervention/common_grid/"
    "18b_common_grid_integration_100m.gpkg"
)

GRID_QA = (
    REPORT / "18b_common_grid_integration_qa.json"
)

QUEUE_SOURCE = (
    DATA
    / "planning_conflict_intervention/"
    "intervention_prioritization/"
    "18g_separate_planning_review_queues_100m.csv"
)

QUEUE_QA = (
    REPORT
    / "18g_separate_planning_review_queues_qa.json"
)

OUT_DIR = (
    DATA
    / "planning_conflict_intervention/gn_hotspots"
)

OUT_INTERSECTIONS = (
    OUT_DIR / "18i_01_gn_grid_intersections.csv"
)

OUT_ASSIGNMENTS = (
    OUT_DIR / "18i_01_grid_gn_assignment_100m.csv"
)

OUT_SUMMARY = (
    OUT_DIR / "18i_01_gn_area_weighted_summary.csv"
)

OUT_GPKG = (
    OUT_DIR / "18i_01_gn_area_weighted_summary.gpkg"
)

OUT_QA = (
    REPORT / "18i_01_gn_area_weighted_summary_qa.json"
)

GN_LAYER = "gn_divisions"
KMC_LAYER = "kmc_boundary"
GRID_LAYER = "planning_conflict_common_grid_100m"
OUT_LAYER = "gn_planning_evidence_summary"

N = 9240
GN_N = 57
EXPECTED_AREA = 88.82921581108494

METRICS = {
    "suitability": "development_suitability_score",
    "pressure": "urbanization_pressure_score",
    "dci": "development_conflict_index_0_1",
    "conservation": "frozen_conservation_priority_0_1",
}


def require(condition, label):

    ok = bool(condition)

    print(
        f"[{'PASS' if ok else 'FAIL'}] {label}"
    )

    if not ok:
        raise ValueError(
            f"18I-01 QA failed: {label}"
        )


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def numeric(df, field):

    return pd.to_numeric(
        df[field],
        errors="coerce",
    ).to_numpy(dtype=float)


def validate_unique(df, field, count, label):

    require(
        len(df) == count
        and field in df.columns
        and df[field].notna().all()
        and df[field].is_unique,
        f"{label}: {count:,} unique records",
    )


def check_crs(gdf, label):

    require(
        gdf.crs is not None
        and gdf.crs.to_epsg() == 32644,
        f"{label}: EPSG:32644",
    )


def remove_z(gdf):

    result = gdf.copy()

    result["geometry"] = [
        force_2d(geom) if geom is not None else None
        for geom in result.geometry
    ]

    return result


def weighted_mean(df, value_field):

    values = numeric(df, value_field)
    weights = numeric(df, "intersection_area_m2")

    valid = np.isfinite(values) & (weights > 0)

    if not valid.any():
        return np.nan

    return float(
        np.average(
            values[valid],
            weights=weights[valid],
        )
    )


def main():

    print("=" * 78)
    print(
        "SECTION 05 | STEP 18I-01 "
        "GN DIVISION AREA-WEIGHTED SUMMARIES"
    )
    print("=" * 78)

    inputs = [
        GN_SOURCE,
        KMC_SOURCE,
        GRID_SOURCE,
        GRID_QA,
        QUEUE_SOURCE,
        QUEUE_QA,
    ]

    outputs = [
        OUT_INTERSECTIONS,
        OUT_ASSIGNMENTS,
        OUT_SUMMARY,
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

    hashes = {
        p.relative_to(ROOT).as_posix(): sha256(p)
        for p in inputs
    }

    # ========================================================
    # 1. VERIFY RESEARCH PROVENANCE
    # ========================================================

    print("\n[1] VERIFY FROZEN RESEARCH SOURCES")

    grid_qa = json.loads(
        GRID_QA.read_text(encoding="utf-8")
    )

    queue_qa = json.loads(
        QUEUE_QA.read_text(encoding="utf-8")
    )

    require(
        grid_qa.get("status")
        == "PASS_COMMON_GRID_FINAL_EXPERT_AHP_INTEGRATION",
        "18B grid integration passed",
    )

    require(
        queue_qa.get("status")
        == "PASS_SEPARATE_REVIEW_QUEUE_RANKINGS_SENSITIVITY_PENDING",
        "18G review queues passed",
    )

    require(
        grid_qa["outputs"]["gpkg_sha256"]
        == sha256(GRID_SOURCE),
        "18B GIS hash verified",
    )

    require(
        queue_qa["output"]["sha256"]
        == sha256(QUEUE_SOURCE),
        "18G review CSV hash verified",
    )

    # ========================================================
    # 2. LOAD GN, KMC AND GRID GEOMETRY
    # ========================================================

    print("\n[2] LOAD ACTUAL GN AND KMC BOUNDARIES")

    gn = gpd.read_file(
        GN_SOURCE,
        layer=GN_LAYER,
    )

    kmc = gpd.read_file(
        KMC_SOURCE,
        layer=KMC_LAYER,
    )

    grid = gpd.read_file(
        GRID_SOURCE,
        layer=GRID_LAYER,
    )

    queue = pd.read_csv(QUEUE_SOURCE)

    check_crs(gn, "GN")
    check_crs(kmc, "KMC")
    check_crs(grid, "Grid")

    validate_unique(
        gn, "id", GN_N, "GN divisions"
    )

    validate_unique(
        grid, "grid_id", N, "Grid GIS"
    )

    validate_unique(
        queue, "grid_id", N, "Review queues"
    )

    require(
        len(kmc) == 1,
        "One processed KMC study boundary",
    )

    require(
        gn.geometry.notna().all()
        and gn.geometry.is_valid.all()
        and not gn.geometry.is_empty.any(),
        "GN geometries valid",
    )

    require(
        grid.geometry.notna().all()
        and grid.geometry.is_valid.all()
        and not grid.geometry.is_empty.any(),
        "Grid geometries valid",
    )

    require(
        set(grid.grid_id) == set(queue.grid_id),
        "Grid and review queue identities match",
    )

    gn = remove_z(gn)
    kmc = remove_z(kmc)
    grid = remove_z(grid)

    gn = gn.rename(
        columns={
            "id": "gn_id",
            "GN_NAME": "gn_name",
        }
    )

    gn["gn_id"] = (
        gn["gn_id"].astype("string").str.strip()
    )

    gn["gn_name"] = (
        gn["gn_name"].astype("string").str.strip()
    )

    require(
        gn["gn_id"].notna().all()
        and gn["gn_id"].ne("").all()
        and gn["gn_id"].is_unique
        and gn["gn_name"].notna().all()
        and gn["gn_name"].ne("").all(),
        "57 valid GN names and IDs",
    )

    kmc_shape = union_all(
        kmc.geometry.to_numpy()
    )

    kmc_area = float(kmc_shape.area)

    require(
        np.isclose(
            kmc_area / 1e6,
            EXPECTED_AREA,
            atol=0.01,
            rtol=0,
        ),
        "Processed KMC area agrees with research area",
    )

    print(
        "KMC boundary area:",
        f"{kmc_area / 1e6:.6f} km2",
    )

    # ========================================================
    # 3. CLIP GN BOUNDARIES TO KMC
    # ========================================================

    print("\n[3] CHECK GN-TO-KMC BOUNDARY COVERAGE")

    gn["geometry"] = [
        geom.intersection(kmc_shape)
        for geom in gn.geometry
    ]

    gn["gn_area_inside_kmc_m2"] = (
        gn.geometry.area
    )

    require(
        np.all(
            gn["gn_area_inside_kmc_m2"] > 0
        ),
        "All 57 GN divisions intersect KMC",
    )

    gn_union = union_all(
        gn.geometry.to_numpy()
    )

    gn_sum = float(
        gn.geometry.area.sum()
    )

    gn_union_area = float(
        gn_union.area
    )

    gn_overlap_area = max(
        0.0,
        gn_sum - gn_union_area,
    )

    gn_gap_area = float(
        kmc_shape.difference(gn_union).area
    )

    print(
        "GN union inside KMC:",
        f"{gn_union_area / 1e6:.6f} km2",
    )

    print(
        "KMC not covered by GN boundaries:",
        f"{gn_gap_area / 1e6:.6f} km2",
    )

    print(
        "GN polygon overlap:",
        f"{gn_overlap_area:.3f} m2",
    )

    # Avoid silently counting significant
    # overlapping GN polygon areas twice.
    require(
        gn_overlap_area <= 100.0,
        "No material overlapping GN polygon areas",
    )

    # GN gaps are preserved and reported,
    # not filled using nearest neighbours.

    # ========================================================
    # 4. ALIGN FINAL SECTION 05 RESEARCH
    # ========================================================

    print("\n[4] VERIFY FINAL RESEARCH GRID SCORES")

    queue = (
        queue.set_index("grid_id")
        .loc[grid.grid_id]
        .reset_index()
    )

    required = [
        "grid_id",
        "development_suitability_score",
        "urbanization_pressure_score",
        "development_conflict_index_0_1",
        "frozen_conservation_priority_0_1",
        "pressure_threshold_flag_frozen",
        "review_rule_id",
    ]

    require(
        set(required).issubset(queue.columns),
        "Required 18G planning evidence available",
    )

    require(
        np.isfinite(
            numeric(
                queue,
                "development_suitability_score",
            )
        ).all(),
        "Suitability exists for all grids",
    )

    require(
        int(
            np.isfinite(
                numeric(
                    queue,
                    "urbanization_pressure_score",
                )
            ).sum()
        ) == 4620,
        "Pressure domain = 4,620",
    )

    require(
        int(
            np.isfinite(
                numeric(
                    queue,
                    "frozen_conservation_priority_0_1",
                )
            ).sum()
        ) == 4609,
        "Frozen CP domain = 4,609",
    )

    require(
        np.allclose(
            numeric(
                queue,
                "development_conflict_index_0_1",
            ),
            numeric(
                grid,
                "urbanization_pressure_score",
            ) * (
                1 - numeric(
                    grid,
                    "development_suitability_score",
                )
            ),
            atol=1e-12,
            rtol=0,
            equal_nan=True,
        ),
        "DCI agrees with final integrated grid",
    )

    source_grid_areas = numeric(
        grid,
        "kmc_overlap_area_m2",
    )

    actual_grid_areas = (
        grid.geometry.area.to_numpy(dtype=float)
    )

    require(
        np.allclose(
            actual_grid_areas,
            source_grid_areas,
            atol=1.0,
            rtol=0,
        ),
        "Grid polygons match documented clipped areas",
    )

    require(
        np.isclose(
            actual_grid_areas.sum() / 1e6,
            EXPECTED_AREA,
            atol=0.001,
            rtol=0,
        ),
        "9,240 grids cover study area",
    )

    # ========================================================
    # 5. FRACTIONAL GN OVERLAY
    # ========================================================

    print("\n[5] INTERSECT GRIDS WITH GN DIVISIONS")

    intersections = gpd.overlay(
        grid[["grid_id", "geometry"]],
        gn[["gn_id", "gn_name", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )

    intersections[
        "intersection_area_m2"
    ] = intersections.geometry.area

    intersections = intersections.loc[
        intersections["intersection_area_m2"] > 1e-6
    ].copy()

    pieces = (
        intersections[
            [
                "grid_id",
                "gn_id",
                "gn_name",
                "intersection_area_m2",
            ]
        ]
        .groupby(
            ["grid_id", "gn_id", "gn_name"],
            as_index=False,
            dropna=False,
        )["intersection_area_m2"]
        .sum()
    )

    require(
        len(pieces) > 0
        and pieces["grid_id"].isin(
            grid.grid_id
        ).all(),
        "Real GN-grid polygon intersections generated",
    )

    covered = (
        pieces.groupby("grid_id")[
            "intersection_area_m2"
        ].sum()
    )

    grid_coverage = (
        grid[
            ["grid_id", "kmc_overlap_area_m2"]
        ]
        .copy()
    )

    grid_coverage[
        "gn_covered_area_m2"
    ] = (
        grid_coverage["grid_id"]
        .map(covered)
        .fillna(0.0)
    )

    grid_coverage[
        "gn_uncovered_area_m2"
    ] = (
        grid_coverage["kmc_overlap_area_m2"]
        - grid_coverage["gn_covered_area_m2"]
    )

    require(
        np.all(
            grid_coverage["gn_uncovered_area_m2"]
            >= -1.0
        ),
        "GN intersections do not materially double-count grids",
    )

    grid_coverage[
        "gn_uncovered_area_m2"
    ] = (
        grid_coverage["gn_uncovered_area_m2"]
        .clip(lower=0)
    )

    grid_coverage[
        "gn_covered_fraction"
    ] = (
        grid_coverage["gn_covered_area_m2"]
        / grid_coverage["kmc_overlap_area_m2"]
    ).clip(0, 1)

    covered_total = float(
        grid_coverage.gn_covered_area_m2.sum()
    )

    uncovered_total = float(
        grid_coverage.gn_uncovered_area_m2.sum()
    )

    print(
        "Grid area covered by GN:",
        f"{covered_total / 1e6:.6f} km2",
    )

    print(
        "Grid area lacking GN coverage:",
        f"{uncovered_total / 1e6:.6f} km2",
    )

    require(
        np.isclose(
            (covered_total + uncovered_total) / 1e6,
            EXPECTED_AREA,
            atol=0.001,
            rtol=0,
        ),
        "Covered and uncovered areas reconcile",
    )

    # ========================================================
    # 6. DOMINANT GN FOR FRONTEND GRID POPUPS
    # ========================================================

    print("\n[6] ASSIGN DOMINANT GN PER GRID")

    sorted_pieces = pieces.sort_values(
        [
            "grid_id",
            "intersection_area_m2",
            "gn_id",
        ],
        ascending=[True, False, True],
    )

    dominant = (
        sorted_pieces
        .drop_duplicates("grid_id")
        [
            [
                "grid_id",
                "gn_id",
                "gn_name",
                "intersection_area_m2",
            ]
        ]
        .rename(
            columns={
                "intersection_area_m2":
                    "dominant_gn_area_m2",
            }
        )
    )

    gn_count = (
        pieces.groupby("grid_id")[
            "gn_id"
        ].nunique()
    )

    assignments = (
        grid_coverage
        .merge(
            dominant,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
    )

    assignments[
        "gn_intersection_count"
    ] = (
        assignments["grid_id"]
        .map(gn_count)
        .fillna(0)
        .astype(int)
    )

    assignments[
        "gn_assignment_status"
    ] = np.select(
        [
            assignments["gn_intersection_count"].eq(0),
            assignments["gn_covered_fraction"].lt(
                0.999
            ),
            assignments["gn_intersection_count"].gt(1),
        ],
        [
            "GN_BOUNDARY_GAP_UNASSIGNED",
            "PARTIAL_GN_COVERAGE_DOMINANT_ASSIGNED",
            "MULTIPLE_GN_DOMINANT_ASSIGNED",
        ],
        default="SINGLE_GN_ASSIGNED",
    )

    assignments[
        "dominant_gn_fraction_of_grid"
    ] = (
        assignments["dominant_gn_area_m2"]
        / assignments["kmc_overlap_area_m2"]
    )

    validate_unique(
        assignments,
        "grid_id",
        N,
        "Grid GN assignments",
    )

    require(
        assignments.loc[
            assignments["gn_intersection_count"].eq(0),
            "gn_name",
        ].isna().all(),
        "Uncovered grids have no invented GN name",
    )

    print(
        "\nGrid GN assignment statuses:"
    )

    print(
        assignments["gn_assignment_status"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # 7. AREA-WEIGHTED GN SUMMARIES
    # ========================================================

    print("\n[7] CALCULATE GN PLANNING SUMMARIES")

    analysis_fields = [
        "grid_id",
        *METRICS.values(),
        "pressure_threshold_flag_frozen",
        "review_rule_id",
    ]

    pieces = pieces.merge(
        queue[analysis_fields],
        on="grid_id",
        how="left",
        validate="many_to_one",
    )

    require(
        pieces[
            "development_suitability_score"
        ].notna().all(),
        "All GN intersections retain suitability",
    )

    gn_records = []

    for record in gn.itertuples():

        subset = pieces.loc[
            pieces["gn_id"] == record.gn_id
        ]

        total_area = float(
            subset["intersection_area_m2"].sum()
        )

        gn_area = float(
            record.gn_area_inside_kmc_m2
        )

        row = {
            "gn_id": str(record.gn_id),
            "gn_name": str(record.gn_name),
            "gn_area_inside_kmc_km2":
                gn_area / 1e6,
            "gn_grid_intersection_count":
                int(len(subset)),
            "unique_intersecting_grid_count":
                int(subset.grid_id.nunique()),
            "mapped_grid_area_km2":
                total_area / 1e6,
            "mapped_gn_coverage_pct":
                (
                    100.0 * total_area / gn_area
                    if gn_area > 0
                    else np.nan
                ),
        }

        for short, field in METRICS.items():

            vals = numeric(
                subset, field
            ) if len(subset) else np.array([])

            valid = np.isfinite(vals)

            valid_area = float(
                subset.loc[
                    valid,
                    "intersection_area_m2",
                ].sum()
            ) if len(subset) else 0.0

            row[
                f"{short}_valid_area_km2"
            ] = valid_area / 1e6

            row[
                f"{short}_area_weighted_mean"
            ] = (
                weighted_mean(subset, field)
                if len(subset)
                else np.nan
            )

        flags = (
            numeric(
                subset,
                "pressure_threshold_flag_frozen",
            )
            if len(subset)
            else np.array([])
        )

        selected = (
            np.isfinite(flags)
            & (flags == 1)
        )

        row[
            "pressure_flagged_intersection_area_ha"
        ] = float(
            subset.loc[
                selected,
                "intersection_area_m2",
            ].sum() / 1e4
        ) if len(subset) else 0.0

        row[
            "pressure_flagged_intersecting_grid_count"
        ] = (
            int(subset.loc[selected, "grid_id"].nunique())
            if len(subset)
            else 0
        )

        for code, label in [
            ("R02_JOINT_CONCERN", "joint_concern"),
            (
                "R03_CONSERVATION_SAFEGUARD",
                "conservation_safeguard",
            ),
            (
                "R05_MANAGED_GROWTH_FEASIBILITY",
                "managed_growth_screen",
            ),
            (
                "R01_CONSERVATION_EVIDENCE_GAP",
                "conservation_evidence_gap",
            ),
        ]:

            selected = (
                subset["review_rule_id"].eq(code)
                if len(subset)
                else pd.Series(dtype=bool)
            )

            row[
                f"{label}_area_ha"
            ] = (
                float(
                    subset.loc[
                        selected,
                        "intersection_area_m2",
                    ].sum() / 1e4
                )
                if len(subset)
                else 0.0
            )

            row[
                f"{label}_intersecting_grid_count"
            ] = (
                int(
                    subset.loc[
                        selected,
                        "grid_id",
                    ].nunique()
                )
                if len(subset)
                else 0
            )

        row["gn_summary_method"] = (
            "POLYGON_INTERSECTION_AREA_WEIGHTED"
        )

        row["development_approval_inferred"] = False

        gn_records.append(row)

    summary = pd.DataFrame(
        gn_records
    )

    validate_unique(
        summary,
        "gn_id",
        GN_N,
        "GN division summary",
    )

    require(
        np.isclose(
            summary[
                "mapped_grid_area_km2"
            ].sum(),
            covered_total / 1e6,
            atol=1e-5,
            rtol=0,
        ),
        "GN mapped areas reconcile to grid intersections",
    )

    require(
        np.all(
            summary["mapped_gn_coverage_pct"]
            <= 100.1
        ),
        "GN summaries do not materially exceed GN area",
    )

    summary_gis = gn[
        ["gn_id", "gn_name", "geometry"]
    ].merge(
        summary,
        on=["gn_id", "gn_name"],
        how="left",
        validate="one_to_one",
    )

    require(
        len(summary_gis) == GN_N,
        "GN GIS preserves all 57 divisions",
    )

    print(
        "GN summary divisions:",
        len(summary),
    )

    print(
        "GN divisions with valid DCI area:",
        int(
            summary["dci_valid_area_km2"]
            .gt(0).sum()
        ),
    )

    # ========================================================
    # 8. EXPORT RESULTS
    # ========================================================

    print("\n[8] EXPORT AND VERIFY RESULTS")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix="18i01_build_",
        dir=OUT_DIR,
    ) as folder:

        temp = Path(folder)

        temp_intersections = (
            temp / OUT_INTERSECTIONS.name
        )

        temp_assignments = (
            temp / OUT_ASSIGNMENTS.name
        )

        temp_summary = (
            temp / OUT_SUMMARY.name
        )

        temp_gpkg = (
            temp / OUT_GPKG.name
        )

        pieces[
            [
                "grid_id",
                "gn_id",
                "gn_name",
                "intersection_area_m2",
            ]
        ].to_csv(
            temp_intersections,
            index=False,
        )

        assignments.to_csv(
            temp_assignments,
            index=False,
        )

        summary.to_csv(
            temp_summary,
            index=False,
        )

        summary_gis.to_file(
            temp_gpkg,
            layer=OUT_LAYER,
            driver="GPKG",
            engine="pyogrio",
        )

        check_pieces = pd.read_csv(
            temp_intersections
        )

        check_assignments = pd.read_csv(
            temp_assignments
        )

        check_summary = pd.read_csv(
            temp_summary
        )

        check_gis = gpd.read_file(
            temp_gpkg,
            layer=OUT_LAYER,
        )

        validate_unique(
            check_assignments,
            "grid_id",
            N,
            "Exported assignments",
        )

        validate_unique(
            check_summary,
            "gn_id",
            GN_N,
            "Exported GN summary",
        )

        validate_unique(
            check_gis,
            "gn_id",
            GN_N,
            "Exported GN GIS",
        )

        require(
            check_gis.crs is not None
            and check_gis.crs.to_epsg() == 32644,
            "Exported GN GIS EPSG:32644",
        )

        require(
            np.isclose(
                numeric(
                    check_pieces,
                    "intersection_area_m2",
                ).sum(),
                covered_total,
                atol=0.01,
                rtol=0,
            ),
            "Exported fractional GN intersections preserved",
        )

        require(
            np.allclose(
                numeric(
                    check_summary,
                    "dci_area_weighted_mean",
                ),
                numeric(
                    summary,
                    "dci_area_weighted_mean",
                ),
                atol=1e-12,
                rtol=0,
                equal_nan=True,
            ),
            "Exported DCI GN means preserved",
        )

        for path in outputs:

            if path.exists():
                raise FileExistsError(
                    f"Output appeared during run: {path}"
                )

        temp_intersections.replace(
            OUT_INTERSECTIONS
        )

        temp_assignments.replace(
            OUT_ASSIGNMENTS
        )

        temp_summary.replace(
            OUT_SUMMARY
        )

        temp_gpkg.replace(
            OUT_GPKG
        )

    # ========================================================
    # 9. VERIFY INPUT INTEGRITY
    # ========================================================

    print("\n[9] VERIFY INPUTS UNCHANGED")

    for path in inputs:

        key = path.relative_to(ROOT).as_posix()

        require(
            sha256(path) == hashes[key],
            f"Source unchanged: {path.name}",
        )

    # ========================================================
    # 10. QA REPORT
    # ========================================================

    print("\n[10] GENERATE GN INTEGRATION QA")

    qa = {
        "section":
            "05_planning_conflict_intervention",

        "stage":
            "18i_01_gn_area_weighted_summary",

        "status":
            "PASS_GN_AREA_WEIGHTED_INTEGRATION_WITH_COVERAGE_AUDIT",

        "gn_divisions": GN_N,

        "grid_cells": N,

        "crs": "EPSG:32644",

        "kmc_area_km2":
            kmc_area / 1e6,

        "gn_union_covered_kmc_km2":
            gn_union_area / 1e6,

        "kmc_not_covered_by_gn_km2":
            gn_gap_area / 1e6,

        "gn_boundary_overlap_m2":
            gn_overlap_area,

        "grid_area_covered_by_gn_km2":
            covered_total / 1e6,

        "grid_area_uncovered_by_gn_km2":
            uncovered_total / 1e6,

        "gn_grid_intersection_records":
            len(pieces),

        "grid_gn_assignment_counts":
            {
                str(k): int(v)
                for k, v in (
                    assignments[
                        "gn_assignment_status"
                    ]
                    .value_counts()
                    .items()
                )
            },

        "gn_area_weighted_metrics":
            list(METRICS.values()),

        "gn_summary_method":
            "FRACTIONAL_POLYGON_INTERSECTION_AREA_WEIGHTS",

        "grid_popup_gn_method":
            "DOMINANT_INTERSECTED_GN_AREA",

        "uncovered_gn_area_imputed":
            False,

        "historical_provisional_suitability_used":
            False,

        "expert_ahp_weights_modified":
            False,

        "dci_or_frozen_cp_modified":
            False,

        "official_planning_classifications_assigned":
            False,

        "limitations": [
            (
                "GN divisions may not completely "
                "cover the KMC boundary. "
                "Any uncovered area is explicitly reported."
            ),
            (
                "A grid crossing multiple GN divisions "
                "contributes to each GN in proportion "
                "to its actual intersected area."
            ),
            (
                "Dominant GN is for grid identification "
                "and map popups, not a replacement "
                "for fractional GN statistics."
            ),
            (
                "Counts of intersecting grids across GN "
                "divisions are not additive; intersected "
                "areas are the aggregation basis."
            ),
            (
                "Missing DCI and frozen CP are excluded "
                "from their respective weighted averages, "
                "not interpreted as zero."
            ),
            (
                "These GN statistics are research "
                "decision-support, not official "
                "development-permission determinations."
            ),
        ],

        "input_sha256":
            hashes,

        "outputs": {
            str(p.relative_to(ROOT).as_posix()):
                sha256(p)
            for p in [
                OUT_INTERSECTIONS,
                OUT_ASSIGNMENTS,
                OUT_SUMMARY,
                OUT_GPKG,
            ]
        },
    }

    OUT_QA.write_text(
        json.dumps(
            qa,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print(qa["status"])

    print("GN divisions:", GN_N)
    print("Grid cells:", N)

    print(
        "GN boundary coverage gaps:",
        f"{gn_gap_area / 1e6:.6f} km2",
    )

    print(
        "Grid area without GN attribution:",
        f"{uncovered_total / 1e6:.6f} km2",
    )

    print(
        "GN divisions with DCI evidence:",
        int(
            summary["dci_valid_area_km2"]
            .gt(0).sum()
        ),
    )

    print("Frozen DCI and CP changed: NO")
    print("Old suitability outputs used: NO")

    print("\nIntersections:", OUT_INTERSECTIONS)
    print("Grid assignments:", OUT_ASSIGNMENTS)
    print("GN summary:", OUT_SUMMARY)
    print("GN GIS:", OUT_GPKG)
    print("QA:", OUT_QA)
    print("=" * 78)


if __name__ == "__main__":
    main()
