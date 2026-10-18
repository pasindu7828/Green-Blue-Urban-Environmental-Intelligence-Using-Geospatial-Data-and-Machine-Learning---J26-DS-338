
"""
STEP 16D-H2 - GN DIVISION INTEGRATION

J26-DS-338: Urban Green Transition,
Urbanization Pressure and Conservation Intelligence.

Inputs:
- Verified 16D-G provisional dashboard release
- Existing 57 Kaduwela GN polygons

Method:
- Maximum polygon-intersection area for grid GN assignment
- Area-weighted GN-level descriptive summaries
- Explicit GN boundary gap and multiple-GN flags

No final AHP model, legal development classifications,
or new grid geometries are created.
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

GRID_GPKG = (
    REPORTS / "16d_g_dashboard_reference_100m.gpkg"
)

GRID_CSV = (
    REPORTS / "16d_g_dashboard_reference_100m.csv"
)

GRID_META = (
    REPORTS / "16d_g_dashboard_release_summary.json"
)

GN_GPKG = (
    ROOT / "data/processed/boundaries/"
    "gn_divisions/kaduwela_gn_divisions_utm44n.gpkg"
)

OUT_GRID_CSV = (
    REPORTS / "16d_h_grid_with_gn_divisions_100m.csv"
)

OUT_GRID_GPKG = (
    REPORTS / "16d_h_grid_with_gn_divisions_100m.gpkg"
)

OUT_GN_CSV = (
    REPORTS / "16d_h_gn_area_weighted_summary.csv"
)

OUT_GN_GPKG = (
    REPORTS / "16d_h_gn_area_weighted_summary.gpkg"
)

OUT_ALLOCATIONS = (
    REPORTS / "16d_h_gn_grid_intersection_audit.csv"
)

OUT_QA = (
    REPORTS / "16d_h_gn_integration_qa.csv"
)

OUT_META = (
    REPORTS / "16d_h_gn_integration_summary.json"
)

EXPECTED_GRID = 9240
EXPECTED_GN = 57
EPSG = 32644

# Numeric intersection tolerance, NOT a legal rule.
MIN_INTERSECTION_M2 = 1e-8

# A meaningful GN coverage gap is larger than this.
GAP_MEASUREMENT_TOLERANCE_M2 = 0.01

# Descriptive indicator for weak GN assignment.
# Not a legal or administrative threshold.
LOW_GN_COVERAGE_FRACTION = 0.50


# ============================================================
# 2. HELPERS
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
            f"Invalid boolean data: {series.name}"
        )

    return values.isin(
        ["true", "1"]
    ).to_numpy(dtype=bool)


def save_json(path, value):

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# 3. MAIN
# ============================================================

def main():

    print("=" * 90)
    print("STEP 16D-H2 - GN DIVISION INTEGRATION")
    print("=" * 90)

    sources = [
        GRID_GPKG,
        GRID_CSV,
        GRID_META,
        GN_GPKG,
    ]

    for path in sources:

        if not path.is_file():
            raise FileNotFoundError(path)

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    hashes = {
        str(p): sha256(p)
        for p in sources
    }

    meta = json.loads(
        GRID_META.read_text(
            encoding="utf-8"
        )
    )

    qa = []

    def check(name, passed, detail=""):

        qa.append({
            "check": name,
            "passed": bool(passed),
            "detail": str(detail),
        })

    def checkpoint():

        failed = [
            item
            for item in qa
            if not item["passed"]
        ]

        if failed:

            pd.DataFrame(qa).to_csv(
                OUT_QA,
                index=False,
            )

            for item in failed:
                print(
                    "REVIEW:",
                    item["check"],
                    item["detail"],
                )

            raise RuntimeError(
                "GN integration requires review. "
                "Inspect the QA CSV."
            )

    # ========================================================
    # 4. LOAD AND VERIFY SOURCE DATA
    # ========================================================

    check(
        "Step 16D-G passed",
        meta.get("qa_status") == "PASS",
    )

    check(
        "Source remains a provisional reference release",
        meta.get("release_status")
        == "DASHBOARD_READY_PROVISIONAL_REFERENCE",
    )

    checkpoint()

    print("\nLoading 100m grids and GN polygons...")

    grid = gpd.read_file(
        GRID_GPKG,
        layer="suitability_reference_dashboard",
    )

    csv = pd.read_csv(
        GRID_CSV,
        dtype={"grid_id": "string"},
    )

    gn = gpd.read_file(
        GN_GPKG,
        layer="gn_divisions",
    )

    grid["grid_id"] = (
        grid["grid_id"]
        .astype("string")
        .str.strip()
    )

    csv["grid_id"] = (
        csv["grid_id"].str.strip()
    )

    check(
        "Grid GeoPackage contains 9240 unique IDs",
        len(grid) == EXPECTED_GRID
        and grid["grid_id"].notna().all()
        and grid["grid_id"].is_unique,
    )

    check(
        "Dashboard CSV contains 9240 unique IDs",
        len(csv) == EXPECTED_GRID
        and csv["grid_id"].notna().all()
        and csv["grid_id"].is_unique,
    )

    check(
        "Grid ID sets agree",
        set(grid["grid_id"])
        == set(csv["grid_id"]),
    )

    check(
        "GN source contains 57 polygon features",
        len(gn) == EXPECTED_GN
        and gn.geom_type.isin([
            "Polygon",
            "MultiPolygon",
        ]).all(),
    )

    check(
        "GN source contains verified fields",
        {"id", "GN_NAME"}.issubset(
            gn.columns
        ),
    )

    check(
        "Both datasets have projected CRS",
        grid.crs is not None
        and gn.crs is not None
        and grid.crs.to_epsg() == EPSG
        and gn.crs.to_epsg() == EPSG,
    )

    check(
        "Grid and GN geometries valid",
        grid.geometry.notna().all()
        and gn.geometry.notna().all()
        and (~grid.geometry.is_empty).all()
        and (~gn.geometry.is_empty).all()
        and grid.geometry.is_valid.all()
        and gn.geometry.is_valid.all(),
    )

    checkpoint()

    gn["id"] = (
        gn["id"]
        .astype("string")
        .str.strip()
    )

    gn["GN_NAME"] = (
        gn["GN_NAME"]
        .astype("string")
        .str.strip()
    )

    check(
        "57 nonempty unique GN IDs and names",
        gn["id"].notna().all()
        and gn["GN_NAME"].notna().all()
        and gn["id"].ne("").all()
        and gn["GN_NAME"].ne("").all()
        and gn["id"].is_unique
        and gn["GN_NAME"].is_unique,
    )

    # Work in the exact order of the CSV source.
    grid = (
        grid
        .set_index("grid_id")
        .loc[csv["grid_id"]]
        .reset_index()
    )

    check(
        "GPKG and CSV scores agree by grid_id",
        np.allclose(
            pd.to_numeric(
                grid["suitability_score"],
                errors="coerce",
            ),
            pd.to_numeric(
                csv["suitability_score"],
                errors="coerce",
            ),
            atol=1e-10,
        ),
    )

    checkpoint()

    grid = grid.reset_index(drop=True)
    gn = gn.reset_index(drop=True)

    grid_area = (
        grid.geometry.area
        .to_numpy(dtype=float)
    )

    score = pd.to_numeric(
        grid["suitability_score"],
        errors="coerce",
    ).to_numpy(dtype=float)

    review = as_bool(
        grid["planning_review_needed"]
    )

    top = as_bool(
        grid["top_decile_reference"]
    )

    stable = as_bool(
        grid["high_stability_both_methods"]
    )

    very_high = (
        grid["relative_suitability_class"]
        .astype(str)
        .eq("RELATIVE_VERY_HIGH")
        .to_numpy()
    )

    check(
        "All scores and cell areas finite",
        np.isfinite(score).all()
        and np.isfinite(grid_area).all()
        and (grid_area > 0).all(),
    )

    check(
        "Study area preserved",
        abs(
            grid_area.sum() / 10000
            - float(meta["area_ha"])
        ) <= 0.05,
    )

    checkpoint()

    # ========================================================
    # 5. MEASURE GN UNION COVERAGE
    # ========================================================

    print("\nMeasuring GN coverage...")

    gn_union = gn.geometry.union_all()

    covered_area = (
        grid.geometry
        .intersection(gn_union)
        .area
        .to_numpy(dtype=float)
    )

    uncovered_area = np.maximum(
        0.0,
        grid_area - covered_area,
    )

    coverage_ha = float(
        covered_area.sum() / 10000
    )

    coverage_pct = float(
        100
        * covered_area.sum()
        / grid_area.sum()
    )

    # Compare to the user's completed inspection.
    # Tolerance allows for rounded printed values.
    check(
        "GN area reproduces the inspected coverage",
        abs(
            coverage_ha - 8669.588
        ) <= 0.1,
        f"Actual GN coverage: {coverage_ha:.4f} ha",
    )

    check(
        "GN union does not exceed any grid area",
        np.all(
            covered_area <= grid_area + 0.01
        ),
    )

    checkpoint()

    # ========================================================
    # 6. GENERATE GRID/GN POLYGON INTERSECTIONS
    # ========================================================

    print("\nCalculating grid-to-GN intersections...")

    # Spatial join identifies candidate polygon pairs.
    # Actual shared polygon AREA determines the label.

    joined = gpd.sjoin(
        grid[["grid_id", "geometry"]],
        gn[["id", "GN_NAME", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    if joined.empty:

        raise RuntimeError(
            "No spatial intersections found."
        )

    grid_positions = joined.index.to_numpy(
        dtype=int
    )

    gn_positions = (
        joined["index_right"]
        .to_numpy(dtype=int)
    )

    left_geometries = gpd.GeoSeries(
        grid.geometry.iloc[
            grid_positions
        ].to_numpy(),
        crs=grid.crs,
    )

    right_geometries = gpd.GeoSeries(
        gn.geometry.iloc[
            gn_positions
        ].to_numpy(),
        crs=grid.crs,
    )

    intersection_area = (
        left_geometries
        .intersection(
            right_geometries,
            align=False,
        )
        .area
        .to_numpy(dtype=float)
    )

    pairs = pd.DataFrame({
        "grid_pos":
            grid_positions,

        "gn_pos":
            gn_positions,

        "grid_id":
            grid["grid_id"].iloc[
                grid_positions
            ].to_numpy(),

        "gn_id":
            gn["id"].iloc[
                gn_positions
            ].astype(str).to_numpy(),

        "gn_name":
            gn["GN_NAME"].iloc[
                gn_positions
            ].astype(str).to_numpy(),

        "intersection_area_m2":
            intersection_area,
    })

    # Remove line and point contacts without area.
    pairs = (
        pairs.loc[
            pairs["intersection_area_m2"]
            > MIN_INTERSECTION_M2
        ]
        .reset_index(drop=True)
    )

    if pairs.empty:
        raise RuntimeError(
            "No positive-area GN intersections."
        )

    # ========================================================
    # 7. CHECK OVERLAPPING GN POLYGONS
    # ========================================================

    pair_area_by_cell = (
        pairs
        .groupby("grid_pos")[
            "intersection_area_m2"
        ]
        .sum()
        .reindex(
            range(EXPECTED_GRID),
            fill_value=0.0,
        )
        .to_numpy(dtype=float)
    )

    excess_area = np.maximum(
        0.0,
        pair_area_by_cell - covered_area,
    )

    total_excess_ha = float(
        excess_area.sum() / 10000
    )

    check(
        "No material double-counting from overlapping GN polygons",
        total_excess_ha <= 0.01,
        (
            "Duplicate overlap estimate: "
            f"{total_excess_ha:.6f} ha"
        ),
    )

    # GN source polygons must not be assumed
    # to partition Kaduwela unless checked.
    checkpoint()

    # ========================================================
    # 8. ASSIGN DOMINANT GN BY LARGEST AREA
    # ========================================================

    counts = (
        pairs
        .groupby("grid_pos")
        .size()
        .reindex(
            range(EXPECTED_GRID),
            fill_value=0,
        )
        .to_numpy(dtype=int)
    )

    dominant = (
        pairs
        .sort_values(
            [
                "grid_pos",
                "intersection_area_m2",
                "gn_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop_duplicates("grid_pos")
        .set_index("grid_pos")
    )

    positions = pd.Index(
        range(EXPECTED_GRID)
    )

    gn_id = (
        dominant["gn_id"]
        .reindex(positions)
    )

    gn_name = (
        dominant["gn_name"]
        .reindex(positions)
    )

    dominant_area = (
        dominant["intersection_area_m2"]
        .reindex(positions)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    gn_covered_pct = (
        100 * covered_area / grid_area
    )

    gn_dominant_pct = (
        100 * dominant_area / grid_area
    )

    has_gap = (
        uncovered_area
        > GAP_MEASUREMENT_TOLERANCE_M2
    )

    status = np.select(
        [
            counts == 0,

            covered_area / grid_area
            < LOW_GN_COVERAGE_FRACTION,

            counts > 1,

            has_gap,
        ],
        [
            "NO_GN_INTERSECTION",

            "LOW_GN_COVERAGE",

            "MULTIPLE_GN_INTERSECTIONS",

            "PARTIAL_SINGLE_GN_COVERAGE",
        ],
        default="SINGLE_GN_COVERAGE",
    )

    # ========================================================
    # 9. BUILD GRID-LEVEL GIS OUTPUT
    # ========================================================

    out = grid.copy()

    out["gn_id"] = (
        gn_id.astype(object).to_numpy()
    )

    out["gn_name"] = (
        gn_name.astype(object).to_numpy()
    )

    out["gn_assignment_method"] = (
        "MAXIMUM_INTERSECTION_AREA"
    )

    out["gn_assignment_status"] = status

    out["gn_intersection_count"] = counts

    out["gn_covered_area_m2"] = covered_area

    out["gn_uncovered_area_m2"] = uncovered_area

    out["gn_covered_pct_cell"] = (
        np.round(
            gn_covered_pct,
            6,
        )
    )

    out["gn_dominant_pct_cell"] = (
        np.round(
            gn_dominant_pct,
            6,
        )
    )

    out["gn_multiple_intersections"] = (
        counts > 1
    )

    out["gn_partial_gap"] = has_gap

    out["gn_unassigned"] = (
        counts == 0
    )

    # Preserve original suitability model status,
    # all review flags and legal-status fields.

    # ========================================================
    # 10. GN-LEVEL AREA-WEIGHTED AGGREGATION
    # ========================================================

    print("\nCalculating area-weighted GN summaries...")

    pos = pairs["grid_pos"].to_numpy(
        dtype=int
    )

    shared_area = (
        pairs["intersection_area_m2"]
        .to_numpy(dtype=float)
    )

    pairs["weighted_score_area"] = (
        shared_area * score[pos]
    )

    pairs["planning_review_area_m2"] = (
        shared_area * review[pos]
    )

    pairs["top_decile_area_m2"] = (
        shared_area * top[pos]
    )

    pairs["very_high_class_area_m2"] = (
        shared_area * very_high[pos]
    )

    pairs["high_stability_area_m2"] = (
        shared_area * stable[pos]
    )

    aggregated = (
        pairs
        .groupby("gn_pos")
        .agg(
            gn_grid_intersection_count=(
                "grid_id",
                "size",
            ),

            gn_covered_area_m2=(
                "intersection_area_m2",
                "sum",
            ),

            weighted_score_sum=(
                "weighted_score_area",
                "sum",
            ),

            planning_review_area_m2=(
                "planning_review_area_m2",
                "sum",
            ),

            top_decile_area_m2=(
                "top_decile_area_m2",
                "sum",
            ),

            very_high_class_area_m2=(
                "very_high_class_area_m2",
                "sum",
            ),

            high_stability_area_m2=(
                "high_stability_area_m2",
                "sum",
            ),
        )
    )

    gn_out = gn[
        ["id", "GN_NAME", "geometry"]
    ].copy()

    gn_out["gn_pos"] = np.arange(
        len(gn_out)
    )

    gn_out = gn_out.join(
        aggregated,
        on="gn_pos",
    )

    gn_out["dominant_grid_count"] = (
        gn_out["id"]
        .map(
            out["gn_id"]
            .value_counts()
        )
        .fillna(0)
        .astype(int)
    )

    gn_out["covered_area_ha"] = (
        gn_out["gn_covered_area_m2"]
        .fillna(0)
        / 10000
    )

    gn_out["mean_reference_score"] = (
        gn_out["weighted_score_sum"]
        / gn_out["gn_covered_area_m2"]
    )

    for field in [
        "planning_review",
        "top_decile",
        "very_high_class",
        "high_stability",
    ]:

        gn_out[field + "_area_pct"] = (
            100
            * gn_out[
                field + "_area_m2"
            ]
            / gn_out[
                "gn_covered_area_m2"
            ]
        )

    gn_out["summary_method"] = (
        "AREA_WEIGHTED_OVER_GRID_GN_INTERSECTIONS"
    )

    gn_out["model_status"] = (
        "PROVISIONAL_REFERENCE_NOT_AHP"
    )

    # GN polygons are preserved as provided.
    # The summary area refers ONLY to the portions
    # overlapping the Kaduwela research grid.

    # ========================================================
    # 11. FINAL QA
    # ========================================================

    check(
        "All 9240 original grids preserved",
        len(out) == EXPECTED_GRID
        and out["grid_id"].is_unique,
    )

    check(
        "All 57 GN divisions preserved",
        len(gn_out) == EXPECTED_GN
        and gn_out["id"].is_unique,
    )

    check(
        "Every assigned GN ID exists in source",
        set(
            out["gn_id"].dropna()
        ).issubset(
            set(gn["id"])
        ),
    )

    check(
        "Cells without GN intersection have no invented GN name",
        out.loc[
            out["gn_unassigned"],
            "gn_name",
        ].isna().all(),
    )

    check(
        "GN coverage and dominant shares between 0 and 100%",
        np.all(
            gn_covered_pct >= -1e-6
        )
        and np.all(
            gn_covered_pct <= 100 + 1e-6
        )
        and np.all(
            gn_dominant_pct >= -1e-6
        )
        and np.all(
            gn_dominant_pct <= 100 + 1e-6
        ),
    )

    check(
        "Dominant GN share does not exceed GN coverage",
        np.all(
            gn_dominant_pct
            <= gn_covered_pct + 1e-5
        ),
    )

    check(
        "Area-weighted GN summaries account for source coverage",
        abs(
            float(
                gn_out["gn_covered_area_m2"].sum()
            )
            - float(covered_area.sum())
        ) <= 100,
    )

    check(
        "GN summary scores are finite and in [0,1]",
        gn_out["mean_reference_score"]
        .notna().all()
        and np.isfinite(
            gn_out["mean_reference_score"]
            .to_numpy(dtype=float)
        ).all()
        and gn_out["mean_reference_score"]
        .between(0, 1).all(),
    )

    check(
        "Original reference scores remain unchanged",
        np.allclose(
            pd.to_numeric(
                out["suitability_score"]
            ),
            score,
            atol=1e-12,
        ),
    )

    check(
        "Original reference model remains explicitly provisional",
        out["model_status"].eq(
            "PROVISIONAL_REFERENCE_NOT_AHP"
        ).all(),
    )

    check(
        "Original source files unchanged",
        all(
            sha256(path)
            == hashes[str(path)]
            for path in sources
        ),
    )

    checkpoint()

    # ========================================================
    # 12. VALIDATE GIS OUTPUTS BEFORE PUBLISHING
    # ========================================================

    grid_stage = OUT_GRID_GPKG.with_name(
        OUT_GRID_GPKG.stem + "_staging.gpkg"
    )

    gn_stage = OUT_GN_GPKG.with_name(
        OUT_GN_GPKG.stem + "_staging.gpkg"
    )

    for temporary in [
        grid_stage,
        gn_stage,
    ]:

        if temporary.exists():
            temporary.unlink()

    out.to_file(
        grid_stage,
        layer="suitability_grid_with_gn",
        driver="GPKG",
        index=False,
    )

    gn_out.to_file(
        gn_stage,
        layer="gn_suitability_summary",
        driver="GPKG",
        index=False,
    )

    grid_test = gpd.read_file(
        grid_stage
    )

    gn_test = gpd.read_file(
        gn_stage
    )

    check(
        "Exported grid GeoPackage has 9240 records",
        len(grid_test) == EXPECTED_GRID
        and grid_test["grid_id"].is_unique,
    )

    check(
        "Exported GN GeoPackage has 57 records",
        len(gn_test) == EXPECTED_GN
        and gn_test["id"].is_unique,
    )

    check(
        "GIS exports have valid original polygons",
        grid_test.geometry.is_valid.all()
        and gn_test.geometry.is_valid.all(),
    )

    check(
        "GIS export retains all reference scores",
        np.allclose(
            pd.to_numeric(
                grid_test["suitability_score"]
            ),
            score,
            atol=1e-10,
        ),
    )

    checkpoint()

    del grid_test, gn_test

    os.replace(
        grid_stage,
        OUT_GRID_GPKG,
    )

    os.replace(
        gn_stage,
        OUT_GN_GPKG,
    )

    # ========================================================
    # 13. SAVE TABLES
    # ========================================================

    out.drop(
        columns="geometry"
    ).to_csv(
        OUT_GRID_CSV,
        index=False,
    )

    gn_out.drop(
        columns=["geometry", "gn_pos"]
    ).to_csv(
        OUT_GN_CSV,
        index=False,
    )

    pairs[
        [
            "grid_id",
            "gn_id",
            "gn_name",
            "intersection_area_m2",
        ]
    ].to_csv(
        OUT_ALLOCATIONS,
        index=False,
    )

    pd.DataFrame(qa).to_csv(
        OUT_QA,
        index=False,
    )

    # ========================================================
    # 14. SUMMARY JSON
    # ========================================================

    summary = {
        "step": "16D-H2",

        "qa_status": "PASS",

        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "grid_count": EXPECTED_GRID,

        "source_gn_division_count": EXPECTED_GN,

        "grid_area_ha": float(
            grid_area.sum() / 10000
        ),

        "gn_covered_area_ha":
            coverage_ha,

        "gn_coverage_pct":
            coverage_pct,

        "gn_uncovered_area_ha": float(
            uncovered_area.sum() / 10000
        ),

        "assigned_grid_cells": int(
            (counts > 0).sum()
        ),

        "unassigned_grid_cells": int(
            (counts == 0).sum()
        ),

        "multiple_gn_intersection_cells": int(
            (counts > 1).sum()
        ),

        "low_gn_coverage_cells": int(
            (
                (covered_area > 0)
                & (
                    covered_area / grid_area
                    < LOW_GN_COVERAGE_FRACTION
                )
            ).sum()
        ),

        "cells_with_gn_coverage_gap": int(
            has_gap.sum()
        ),

        "estimated_overlapping_gn_allocation_ha":
            total_excess_ha,

        "gn_assignment_method":
            "MAXIMUM_POLYGON_INTERSECTION_AREA",

        "gn_summary_method":
            "AREA_WEIGHTED_GRID_GN_INTERSECTIONS",

        "model_status":
            "PROVISIONAL_REFERENCE_NOT_AHP",

        "gn_coverage_complete":
            False,

        "expert_ahp_validated":
            False,

        "qa_checks": len(qa),

        "qa_passed": sum(
            record["passed"]
            for record in qa
        ),

        "source_sha256": {
            path.name: hashes[str(path)]
            for path in sources
        },

        "limitations": [
            "GN polygons cover approximately 97.6% of the Kaduwela grid area",
            "Uncovered portions are not assigned invented GN names",
            "Dominant GN labels do not mean a cell lies completely inside that GN",
            "GN-level scores are area-weighted descriptive summaries",
            "GN coverage is independent from provisional UDA zoning coverage",
            "Suitability remains a provisional reference model, not expert AHP",
            "Grid cells do not represent legally verified buildable parcels",
            "No development permission is determined",
        ],

        "outputs": [
            path.name
            for path in [
                OUT_GRID_CSV,
                OUT_GRID_GPKG,
                OUT_GN_CSV,
                OUT_GN_GPKG,
                OUT_ALLOCATIONS,
                OUT_QA,
                OUT_META,
            ]
        ],
    }

    save_json(
        OUT_META,
        summary,
    )

    # ========================================================
    # 15. TERMINAL RESULTS
    # ========================================================

    print("\n=== GN COVERAGE ===")

    print(
        "GN divisions:",
        EXPECTED_GN
    )

    print(
        "Suitability grids:",
        len(out)
    )

    print(
        "GN-covered area (ha):",
        round(coverage_ha, 4)
    )

    print(
        "GN coverage (%):",
        round(coverage_pct, 4)
    )

    print(
        "GN-uncovered area (ha):",
        round(
            summary["gn_uncovered_area_ha"],
            4,
        )
    )

    print(
        "Grids with a GN assignment:",
        summary["assigned_grid_cells"]
    )

    print(
        "Grids without a GN assignment:",
        summary["unassigned_grid_cells"]
    )

    print(
        "Grids intersecting multiple GN polygons:",
        summary["multiple_gn_intersection_cells"]
    )

    print(
        "Grids with less than 50% GN coverage:",
        summary["low_gn_coverage_cells"]
    )

    print(
        "\n=== QA RESULTS ==="
    )

    for record in qa:

        print(
            (
                "PASS: "
                if record["passed"]
                else "REVIEW: "
            ) + record["check"]
        )

    print(
        f"\nOverall QA: PASS "
        f"({len(qa)}/{len(qa)})"
    )

    print("\n=== SAVED REPORTS ===")

    for path in [
        OUT_GRID_CSV,
        OUT_GRID_GPKG,
        OUT_GN_CSV,
        OUT_GN_GPKG,
        OUT_ALLOCATIONS,
        OUT_QA,
        OUT_META,
    ]:

        print(
            path.relative_to(ROOT)
        )

    print(
        "\nOriginal grids, suitability scores "
        "and source GN polygons are unchanged."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
