
"""
STEP 16C-M1 - GRID GEOMETRY DIFFERENCE DIAGNOSTIC

Component:
Urban Green Transition, Urbanization Pressure
and Conservation Intelligence

Purpose:
1. Compare suitability and modelling grid geometries.
2. Measure differences by matching grid_id.
3. Check polygon area, overlap and centroid differences.
4. Examine both grids against the supplied KMC boundary.
5. Produce permanent CSV/JSON diagnostic evidence.

IMPORTANT:
- Never modifies either original grid.
- Does not assume the geometries must be identical.
- Does not change any frozen raw criteria.
- No suitability scoring, AHP or zoning classification.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sqlite3

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REPORTS = ROOT / "outputs/reports"

SUITABILITY_GRID = (
    ROOT
    / "data/processed/development_suitability"
    / "raw_criteria/16b_raw_suitability_criteria_100m.gpkg"
)

MODELLING_GRID = (
    ROOT
    / "data/processed/urbanization_pressure"
    / "grid/kaduwela_100m_modelling_grid.gpkg"
)

FROZEN_CSV = (
    ROOT
    / "data/processed/development_suitability"
    / "raw_criteria/16b_raw_suitability_criteria_100m.csv"
)

KMC_BOUNDARY = (
    ROOT
    / "data/raw/uda/zoning_provisional"
    / "provided_package_extracted/New data"
    / "Kaduwela_MC_Boundary.gpkg"
)

PREVIOUS_SUMMARY = (
    REPORTS / "16c_j_zoning_reconciliation_summary.json"
)

OUT_DIFFERENCES = (
    REPORTS / "16c_m1_grid_geometry_differences.csv"
)

OUT_QA = (
    REPORTS / "16c_m1_grid_geometry_qa.csv"
)

OUT_SUMMARY = (
    REPORTS / "16c_m1_grid_geometry_summary.json"
)

EXPECTED_ROWS = 9240
TARGET_EPSG = 32644

# Diagnostic measurement tolerance only.
# This is NOT a development or planning rule.
DIFFERENCE_TOLERANCE_M2 = 0.1


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as source:
        for block in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_json(path):

    with path.open(encoding="utf-8") as source:
        return json.load(source)


def identify_layer(path):

    with sqlite3.connect(str(path)) as conn:

        records = conn.execute(
            "SELECT table_name "
            "FROM gpkg_contents "
            "WHERE data_type = 'features' "
            "ORDER BY table_name"
        ).fetchall()

    names = [record[0] for record in records]

    if len(names) == 1:
        return names[0]

    candidates = [
        name
        for name in names
        if "grid" in name.lower()
        or "100m" in name.lower()
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"Cannot identify unique grid layer in "
        f"{path.name}: {names}"
    )


def load_grid(path):

    layer = identify_layer(path)

    gdf = gpd.read_file(
        path,
        layer=layer,
    )

    id_columns = [
        col for col in gdf.columns
        if col.lower() == "grid_id"
    ]

    if len(id_columns) != 1:

        raise RuntimeError(
            f"Grid ID column not found in {path.name}"
        )

    gdf = gdf.rename(
        columns={id_columns[0]: "grid_id"}
    )

    gdf["grid_id"] = (
        gdf["grid_id"]
        .astype("string")
        .str.strip()
    )

    original_crs = (
        gdf.crs.to_string()
        if gdf.crs
        else "MISSING"
    )

    return gdf, layer, original_crs


def geometry_valid(gdf):

    return bool(
        gdf.crs is not None
        and gdf.geometry.notna().all()
        and (~gdf.geometry.is_empty).all()
        and gdf.geometry.is_valid.all()
        and (gdf.geometry.area > 0).all()
    )


def union_all(geometries):

    if hasattr(geometries, "union_all"):
        return geometries.union_all()

    return geometries.unary_union


def stats(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


# ============================================================
# 3. MAIN PROCESS
# ============================================================

def main():

    print("=" * 94)
    print(
        "STEP 16C-M1 - GRID GEOMETRY DIFFERENCE DIAGNOSTIC"
    )
    print("=" * 94)

    required = [
        SUITABILITY_GRID,
        MODELLING_GRID,
        FROZEN_CSV,
        KMC_BOUNDARY,
        PREVIOUS_SUMMARY,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_hashes = {
        path.name: sha256(path)
        for path in required
        if path != PREVIOUS_SUMMARY
    }

    previous = read_json(PREVIOUS_SUMMARY)

    checks = []

    def check(name, passed, detail=""):

        checks.append({
            "check": name,
            "passed": bool(passed),
            "detail": str(detail),
        })

    check(
        "Previous zoning reconciliation passed",
        previous.get("qa_status") == "PASS",
    )

    check(
        "Frozen CSV checksum matches Step 16C-J",
        sha256(FROZEN_CSV)
        == previous.get("frozen_input_sha256"),
    )

    # ========================================================
    # 4. READ EXISTING GRIDS
    # ========================================================

    print("\n=== INPUT GRID FILES ===")

    suitability, s_layer, s_crs = load_grid(
        SUITABILITY_GRID
    )

    modelling, m_layer, m_crs = load_grid(
        MODELLING_GRID
    )

    print(
        "Suitability grid:",
        SUITABILITY_GRID.name,
    )
    print(
        "  Layer:", s_layer,
        "| CRS:", s_crs,
        "| Rows:", len(suitability),
    )

    print(
        "Modelling grid:",
        MODELLING_GRID.name,
    )
    print(
        "  Layer:", m_layer,
        "| CRS:", m_crs,
        "| Rows:", len(modelling),
    )

    check(
        "Suitability grid has 9240 unique IDs",
        len(suitability) == EXPECTED_ROWS
        and suitability["grid_id"].notna().all()
        and suitability["grid_id"].is_unique,
    )

    check(
        "Modelling grid has 9240 unique IDs",
        len(modelling) == EXPECTED_ROWS
        and modelling["grid_id"].notna().all()
        and modelling["grid_id"].is_unique,
    )

    raw_ids = pd.read_csv(
        FROZEN_CSV,
        usecols=["grid_id"],
        dtype={"grid_id": "string"},
    )["grid_id"].str.strip()

    check(
        "Frozen CSV has 9240 unique IDs",
        len(raw_ids) == EXPECTED_ROWS
        and raw_ids.notna().all()
        and raw_ids.is_unique,
    )

    check(
        "All three grid ID sets match",
        set(suitability["grid_id"])
        == set(modelling["grid_id"])
        == set(raw_ids),
    )

    check(
        "Suitability grid has valid geometry",
        geometry_valid(suitability),
    )

    check(
        "Modelling grid has valid geometry",
        geometry_valid(modelling),
    )

    if not all(item["passed"] for item in checks):

        pd.DataFrame(checks).to_csv(
            OUT_QA,
            index=False,
        )

        raise RuntimeError(
            "Initial grid validation failed."
        )

    # ========================================================
    # 5. PROJECT AND ALIGN BY GRID ID
    # ========================================================

    suitability = suitability.to_crs(
        epsg=TARGET_EPSG
    )

    modelling = modelling.to_crs(
        epsg=TARGET_EPSG
    )

    a = suitability.geometry.reset_index(
        drop=True
    )

    model_lookup = (
        modelling
        .set_index("grid_id")
        .geometry
    )

    aligned_model = (
        model_lookup
        .reindex(suitability["grid_id"])
    )

    b = gpd.GeoSeries(
        aligned_model.to_numpy(),
        crs=suitability.crs,
    )

    check(
        "Modelling geometries aligned without missing IDs",
        b.notna().all(),
    )

    # ========================================================
    # 6. MEASURE ACTUAL GEOMETRY DIFFERENCES
    # ========================================================

    area_a = a.area.to_numpy(
        dtype=float
    )

    area_b = b.area.to_numpy(
        dtype=float
    )

    shared = a.intersection(
        b,
        align=False,
    )

    shared_area = shared.area.to_numpy(
        dtype=float
    )

    # Area of symmetric difference:
    # area(A) + area(B) - 2 * area(A intersection B)

    difference_area = np.maximum(
        0.0,
        area_a + area_b - 2.0 * shared_area,
    )

    union_area = np.maximum(
        0.0,
        area_a + area_b - shared_area,
    )

    iou = np.divide(
        shared_area,
        union_area,
        out=np.zeros_like(shared_area),
        where=union_area > 0,
    )

    centroid_distance = (
        a.centroid.distance(
            b.centroid,
            align=False,
        )
        .to_numpy(dtype=float)
    )

    absolute_area_change = np.abs(
        area_a - area_b
    )

    tolerance_mask = (
        difference_area
        > DIFFERENCE_TOLERANCE_M2
    )

    # ========================================================
    # 7. CHECK KMC BOUNDARY RELATIONSHIPS
    # ========================================================

    boundary_gdf = gpd.read_file(
        KMC_BOUNDARY,
        layer="Kaduwela_MC_Boundary",
    )

    if (
        boundary_gdf.crs is None
        or boundary_gdf.geometry.isna().any()
        or not boundary_gdf.geometry.is_valid.all()
    ):

        raise RuntimeError(
            "KMC reference boundary is invalid."
        )

    boundary_gdf = boundary_gdf.to_crs(
        epsg=TARGET_EPSG
    )

    boundary = union_all(
        boundary_gdf.geometry
    )

    inside_a = (
        a.intersection(boundary)
        .area
        .to_numpy(dtype=float)
    )

    inside_b = (
        b.intersection(boundary)
        .area
        .to_numpy(dtype=float)
    )

    outside_a = np.maximum(
        0.0,
        area_a - inside_a,
    )

    outside_b = np.maximum(
        0.0,
        area_b - inside_b,
    )

    # ========================================================
    # 8. BUILD PER-CELL DIAGNOSTIC REPORT
    # ========================================================

    result = pd.DataFrame({
        "grid_id": suitability[
            "grid_id"
        ].to_numpy(),

        "suitability_area_m2": area_a,

        "modelling_area_m2": area_b,

        "intersection_area_m2": shared_area,

        "absolute_area_change_m2":
            absolute_area_change,

        "symmetric_difference_m2":
            difference_area,

        "intersection_over_union": iou,

        "centroid_distance_m":
            centroid_distance,

        "suitability_outside_kmc_m2":
            outside_a,

        "modelling_outside_kmc_m2":
            outside_b,

        "geometry_difference_over_0_1m2":
            tolerance_mask,
    })

    # ========================================================
    # 9. DIAGNOSTIC QA
    # ========================================================

    numeric_columns = [
        "suitability_area_m2",
        "modelling_area_m2",
        "intersection_area_m2",
        "absolute_area_change_m2",
        "symmetric_difference_m2",
        "intersection_over_union",
        "centroid_distance_m",
        "suitability_outside_kmc_m2",
        "modelling_outside_kmc_m2",
    ]

    numeric = result[
        numeric_columns
    ].to_numpy(dtype=float)

    check(
        "All diagnostic measurements are finite",
        bool(
            np.isfinite(numeric).all()
        ),
    )

    check(
        "Intersection-over-union within 0 to 1",
        bool(
            np.all(iou >= -1e-8)
            and np.all(iou <= 1 + 1e-8)
        ),
    )

    check(
        "Intersection cannot exceed either polygon area",
        bool(
            np.all(shared_area <= area_a + 1e-5)
            and np.all(shared_area <= area_b + 1e-5)
        ),
    )

    check(
        "9240 grid comparisons produced",
        len(result) == EXPECTED_ROWS
        and result["grid_id"].is_unique,
    )

    check(
        "Original inputs unchanged",
        all(
            sha256(path)
            == source_hashes[path.name]
            for path in required
            if path.name in source_hashes
        ),
    )

    qa = pd.DataFrame(checks)

    status = (
        "PASS"
        if qa["passed"].all()
        else "REVIEW_REQUIRED"
    )

    # ========================================================
    # 10. SAVE OUTPUTS
    # ========================================================

    result.to_csv(
        OUT_DIFFERENCES,
        index=False,
    )

    qa.to_csv(
        OUT_QA,
        index=False,
    )

    summary = {
        "step": "16C-M1",

        "qa_status": status,

        "analysis_status":
            "DIAGNOSTIC_NO_GEOMETRY_REPLACEMENT",

        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "grid_count": len(result),

        "target_crs": f"EPSG:{TARGET_EPSG}",

        "original_suitability_crs": s_crs,

        "original_modelling_crs": m_crs,

        "suitability_layer": s_layer,

        "modelling_layer": m_layer,

        "source_hashes": source_hashes,

        "geometry_difference_tolerance_m2":
            DIFFERENCE_TOLERANCE_M2,

        "cells_above_tolerance": int(
            tolerance_mask.sum()
        ),

        "cells_above_1m2": int(
            (difference_area > 1).sum()
        ),

        "cells_above_100m2": int(
            (difference_area > 100).sum()
        ),

        "cells_above_500m2": int(
            (difference_area > 500).sum()
        ),

        "cells_iou_below_0_99": int(
            (iou < 0.99).sum()
        ),

        "cells_iou_below_0_90": int(
            (iou < 0.90).sum()
        ),

        "cells_suitability_zero_kmc_overlap":
            int((inside_a <= 0).sum()),

        "cells_modelling_zero_kmc_overlap":
            int((inside_b <= 0).sum()),

        "cells_suitability_outside_kmc":
            int((outside_a > 0.1).sum()),

        "cells_modelling_outside_kmc":
            int((outside_b > 0.1).sum()),

        "suitability_total_area_ha":
            float(area_a.sum() / 10000),

        "modelling_total_area_ha":
            float(area_b.sum() / 10000),

        "suitability_area_within_kmc_ha":
            float(inside_a.sum() / 10000),

        "modelling_area_within_kmc_ha":
            float(inside_b.sum() / 10000),

        "suitability_grid_area_stats_m2":
            stats(area_a),

        "modelling_grid_area_stats_m2":
            stats(area_b),

        "symmetric_difference_stats_m2":
            stats(difference_area),

        "iou_stats": stats(iou),

        "centroid_distance_stats_m":
            stats(centroid_distance),

        "qa_checks": len(qa),

        "qa_passed": int(
            qa["passed"].sum()
        ),

        "important_note": (
            "Different geometry does not automatically "
            "mean invalid research data. The results "
            "must be interpreted before continuing "
            "the zoning overlay."
        ),

        "outputs": [
            OUT_DIFFERENCES.name,
            OUT_QA.name,
            OUT_SUMMARY.name,
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
    # 11. PRINT RESULTS
    # ========================================================

    print("\n=== GEOMETRY DIFFERENCE SUMMARY ===")

    print(
        "Total grid cells:",
        len(result),
    )

    print(
        "Cells with difference > 0.1 m2:",
        summary["cells_above_tolerance"],
    )

    print(
        "Cells with difference > 1 m2:",
        summary["cells_above_1m2"],
    )

    print(
        "Cells with difference > 100 m2:",
        summary["cells_above_100m2"],
    )

    print(
        "Cells with difference > 500 m2:",
        summary["cells_above_500m2"],
    )

    print(
        "Maximum symmetric difference (m2):",
        round(
            float(difference_area.max()),
            4,
        ),
    )

    print(
        "Median symmetric difference (m2):",
        round(
            float(np.median(difference_area)),
            4,
        ),
    )

    print(
        "Minimum IoU:",
        round(float(iou.min()), 6),
    )

    print(
        "Median IoU:",
        round(float(np.median(iou)), 6),
    )

    print(
        "Cells with IoU < 0.99:",
        summary["cells_iou_below_0_99"],
    )

    print(
        "Cells with IoU < 0.90:",
        summary["cells_iou_below_0_90"],
    )

    print(
        "Maximum centroid distance (m):",
        round(
            float(centroid_distance.max()),
            4,
        ),
    )

    print("\n=== AREA COMPARISON ===")

    print(
        "Suitability grid total area (ha):",
        round(
            summary["suitability_total_area_ha"],
            4,
        ),
    )

    print(
        "Modelling grid total area (ha):",
        round(
            summary["modelling_total_area_ha"],
            4,
        ),
    )

    print(
        "Suitability cells outside KMC:",
        summary["cells_suitability_outside_kmc"],
    )

    print(
        "Modelling cells outside KMC:",
        summary["cells_modelling_outside_kmc"],
    )

    print(
        "Suitability cells with zero KMC overlap:",
        summary["cells_suitability_zero_kmc_overlap"],
    )

    print(
        "Modelling cells with zero KMC overlap:",
        summary["cells_modelling_zero_kmc_overlap"],
    )

    print("\n=== TOP 10 GEOMETRY DIFFERENCES ===")

    print(
        result.nlargest(
            10,
            "symmetric_difference_m2",
        )[
            [
                "grid_id",
                "suitability_area_m2",
                "modelling_area_m2",
                "symmetric_difference_m2",
                "intersection_over_union",
                "centroid_distance_m",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\n=== QA RESULTS ===")

    for item in checks:

        print(
            "PASS" if item["passed"] else "REVIEW",
            ":",
            item["check"],
        )

    print(
        "\nOverall diagnostic QA:",
        status,
    )

    print("\n=== GENERATED REPORTS ===")

    for path in [
        OUT_DIFFERENCES,
        OUT_QA,
        OUT_SUMMARY,
    ]:

        print(
            path.relative_to(ROOT)
        )

    print(
        "\nNo original grid or research input "
        "was modified."
    )

    print("=" * 94)


# ============================================================
# 12. EXECUTE
# ============================================================

if __name__ == "__main__":
    main()
