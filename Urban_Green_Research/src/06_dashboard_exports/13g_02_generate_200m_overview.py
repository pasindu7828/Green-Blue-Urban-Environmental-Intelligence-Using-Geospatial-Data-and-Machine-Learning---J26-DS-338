
"""
Urban Green - Step 13G-2
Validated 200 m Dashboard Overview

Research component:
Urban Green Transition, Urbanization Pressure
and Conservation Intelligence

Purpose:
Generate 200 m dashboard visualization polygons
from the frozen original 100 m research grids.

IMPORTANT:
- Does not change original 100 m research results.
- Does not retrain any machine learning model.
- Does not recalculate expert AHP weights.
- Does not modify conservation or planning models.
- Does not modify other members' components.
- 200 m scores are area-weighted visualization means,
  NOT new 200 m model predictions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from shapely import transform as shapely_transform, union_all
from shapely.geometry import mapping, shape
from pyproj import Transformer

import json
import math
import re
import sqlite3


# =====================================================
# 1. PROJECT PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT
    / "outputs"
    / "dashboard"
    / "planning_conflict_intervention"
    / "grid_map.geojson"
)

GIS = (
    ROOT
    / "data"
    / "processed"
    / "planning_conflict_intervention"
    / "final_gis"
    / "18j_planning_conflict_final.gpkg"
)

GIS_LAYER = "planning_grid_100m"

DESTINATION = (
    ROOT
    / "outputs"
    / "dashboard"
    / "urban_green_200m_overview"
)

OUT_GEOJSON = (
    DESTINATION
    / "grid_map_200m.geojson"
)

OUT_REPORT = (
    DESTINATION
    / "grid_map_200m_qa.json"
)


# =====================================================
# 2. RESEARCH SCORE FIELDS
# =====================================================

FIELDS = {
    "pressure":
        "urbanization_pressure_score",

    "suitability":
        "development_suitability_score",

    "conservation":
        "frozen_conservation_priority_0_1",

    "conflict":
        "development_conflict_index_0_1",
}

EXPECTED_SCORED = {
    "pressure": 4620,
    "suitability": 9240,
    "conservation": 4609,
    "conflict": 4620,
}

PATTERN = re.compile(
    r"^G100_R(\d+)_C(\d+)$"
)


# =====================================================
# 3. VALIDATION HELPERS
# =====================================================

def check(condition, message):
    if not condition:
        raise AssertionError(message)


def is_numeric(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


# =====================================================
# 4. MAIN EXPORTER
# =====================================================

def main():

    print(
        "\n=== 13G-2 | 200 m DISPLAY EXPORT "
        "(SOURCE READ-ONLY) ===",
        flush=True,
    )

    # -------------------------------------------------
    # 4A. VERIFY INPUT FILES
    # -------------------------------------------------

    check(
        SOURCE.is_file(),
        f"Source GeoJSON missing: {SOURCE}",
    )

    check(
        GIS.is_file(),
        f"Final GIS missing: {GIS}",
    )

    check(
        not OUT_GEOJSON.exists()
        and not OUT_REPORT.exists(),
        "Output exists. Refusing to overwrite "
        "without review.",
    )

    data = json.loads(
        SOURCE.read_text(encoding="utf-8")
    )

    original = data.get("features", [])

    check(
        data.get("type") == "FeatureCollection",
        "Not a FeatureCollection",
    )

    check(
        len(original) == 9240,
        f"Expected 9240 source grids, "
        f"received {len(original)}",
    )

    print(
        f"Source grids: {len(original)}"
    )

    # -------------------------------------------------
    # 4B. LOAD ORIGINAL GRID AREAS
    # -------------------------------------------------

    # The dashboard GeoJSON does not include
    # kmc_overlap_area_m2.
    #
    # Read the original research areas from the
    # validated planning GeoPackage using grid_id.

    with sqlite3.connect(GIS) as db:

        columns = {
            row[1]
            for row in db.execute(
                f'PRAGMA table_info("{GIS_LAYER}")'
            )
        }

        check(
            {
                "grid_id",
                "kmc_overlap_area_m2",
            }.issubset(columns),
            f"Required area columns missing "
            f"from {GIS_LAYER}: {columns}",
        )

        raw_areas = db.execute(
            f'SELECT grid_id, kmc_overlap_area_m2 '
            f'FROM "{GIS_LAYER}"'
        ).fetchall()

    areas = {}

    for grid_id, area in raw_areas:

        check(
            grid_id not in areas,
            f"Duplicate GIS area ID: {grid_id}",
        )

        check(
            is_numeric(area) and area > 0,
            f"Invalid area for {grid_id}: {area}",
        )

        areas[grid_id] = float(area)

    check(
        len(areas) == 9240,
        f"Expected 9240 GIS areas, "
        f"got {len(areas)}",
    )

    print(
        "Original grid areas loaded:",
        len(areas),
    )

    # -------------------------------------------------
    # 4C. PREPARE COORDINATE SYSTEMS
    # -------------------------------------------------

    # Source dashboard GeoJSON: EPSG:4326
    # Area-validation GIS CRS: EPSG:32644

    to_utm = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32644",
        always_xy=True,
    )

    to_wgs84 = Transformer.from_crs(
        "EPSG:32644",
        "EPSG:4326",
        always_xy=True,
    )

    # -------------------------------------------------
    # 4D. VALIDATE ORIGINAL GEOMETRIES
    # -------------------------------------------------

    groups = defaultdict(list)

    projected_source = []

    seen = set()

    mismatched_area = []

    source_valid_counts = Counter()

    print(
        "Checking every original geometry "
        "and joining areas...",
        flush=True,
    )

    for feature in original:

        props = feature.get("properties") or {}

        grid_id = str(
            props.get("grid_id", "")
        )

        check(
            grid_id not in seen,
            f"Duplicate source grid: {grid_id}",
        )

        seen.add(grid_id)

        check(
            grid_id in areas,
            f"No area joined for {grid_id}",
        )

        match = PATTERN.fullmatch(grid_id)

        check(
            match is not None,
            f"Unrecognized 100 m grid ID: {grid_id}",
        )

        source_geom = shape(
            feature["geometry"]
        )

        check(
            source_geom.is_valid
            and not source_geom.is_empty,
            f"Invalid source geometry: {grid_id}",
        )

        # Convert to projected metres for
        # geometry and area validation.

        geom = shapely_transform(
            source_geom,
            to_utm.transform,
            interleaved=False,
        )

        check(
            geom.is_valid and geom.area > 0,
            f"Invalid projected source "
            f"geometry: {grid_id}",
        )

        # Check whether the map polygon area
        # agrees with the frozen GIS record.

        delta = abs(
            geom.area - areas[grid_id]
        )

        if delta > max(
            25.0,
            0.02 * areas[grid_id],
        ):
            mismatched_area.append(
                (
                    grid_id,
                    round(geom.area, 2),
                    round(areas[grid_id], 2),
                )
            )

        # Original row/column indexing.
        r = int(match.group(1))
        c = int(match.group(2))

        # Combine a 2 x 2 group of
        # neighboring original 100 m grids.
        key = (
            r // 2,
            c // 2,
        )

        groups[key].append(
            {
                "id": grid_id,
                "geom": geom,
                "area": areas[grid_id],
                "properties": props,
            }
        )

        projected_source.append(geom)

        for metric, field in FIELDS.items():

            if is_numeric(props.get(field)):

                check(
                    0 <= props[field] <= 1,
                    f"Unexpected {metric} score "
                    f"outside [0,1] in {grid_id}",
                )

                source_valid_counts[metric] += 1

    # -------------------------------------------------
    # 4E. VERIFY ORIGINAL DATA COVERAGE
    # -------------------------------------------------

    check(
        seen == set(areas),
        "Source IDs and area IDs differ",
    )

    check(
        not mismatched_area,
        "Source geometries disagree with "
        "final-GIS overlap areas; examples: "
        + str(mismatched_area[:5]),
    )

    check(
        len(groups) == 2410,
        f"Expected 2410 groups, got {len(groups)}",
    )

    for metric, expected in EXPECTED_SCORED.items():

        check(
            source_valid_counts[metric] == expected,
            f"{metric} scored-source count changed: "
            f"{source_valid_counts[metric]}",
        )

    group_sizes = Counter(
        len(records)
        for records in groups.values()
    )

    check(
        dict(group_sizes) == {
            1: 60,
            2: 88,
            3: 44,
            4: 2218,
        },
        "Unexpected grouped source-cell "
        f"distribution: {dict(group_sizes)}",
    )

    print(
        f"Input QA PASS: {len(groups)} overview "
        "groups, source coverage unchanged",
        flush=True,
    )

    # -------------------------------------------------
    # 4F. PREPARE OVERVIEW OUTPUT
    # -------------------------------------------------

    result_features = []

    projected_overview = []

    coverage_counts = {
        metric: Counter()
        for metric in FIELDS
    }

    overview_geom_total = 0.0

    original_reported_area = sum(
        areas.values()
    )

    print(
        "Dissolving true polygon footprints "
        "(no artificial squares)...",
        flush=True,
    )

    # -------------------------------------------------
    # 4G. DISSOLVE NEIGHBORING GRID GEOMETRIES
    # -------------------------------------------------

    for index, (
        (group_r, group_c),
        records,
    ) in enumerate(
        sorted(groups.items()),
        1,
    ):

        joined_geom = union_all(
            [
                record["geom"]
                for record in records
            ]
        )

        overview_id = (
            f"OV200_R{group_r:03d}_C{group_c:03d}"
        )

        check(
            joined_geom.is_valid
            and not joined_geom.is_empty,
            f"Invalid dissolved geometry: "
            f"{overview_id}",
        )

        # Check original cells do not
        # overlap unexpectedly.

        summed_geom_area = sum(
            record["geom"].area
            for record in records
        )

        lost_overlap = (
            summed_geom_area
            - joined_geom.area
        )

        check(
            abs(lost_overlap)
            <= max(
                10.0,
                summed_geom_area * 0.0005,
            ),
            f"Unexpected overlap inside "
            f"{overview_id}: "
            f"{lost_overlap:.3f} m2",
        )

        x0, y0, x1, y1 = joined_geom.bounds

        check(
            x1 - x0 <= 225
            and y1 - y0 <= 225,
            "Not spatially compatible with "
            f"200 m overview: {overview_id}",
        )

        group_area = sum(
            record["area"]
            for record in records
        )

        check(
            group_area > 0,
            f"Invalid area: {overview_id}",
        )

        # ---------------------------------------------
        # 4H. OVERVIEW IDENTIFICATION
        # ---------------------------------------------

        props = {
            "overview_id": overview_id,

            "display_resolution_m": 200,

            "original_resolution_m": 100,

            "source_cell_count": len(records),

            "source_grid_ids": sorted(
                record["id"]
                for record in records
            ),

            "source_overlap_area_m2": round(
                group_area,
                6,
            ),

            "display_only": True,
        }

        # ---------------------------------------------
        # 4I. AREA-WEIGHTED SCORE AGGREGATION
        # ---------------------------------------------

        for metric, field in FIELDS.items():

            # Select only records with an
            # actual numeric research score.

            eligible = [
                record
                for record in records
                if is_numeric(
                    record["properties"].get(field)
                )
            ]

            valid_area = sum(
                record["area"]
                for record in eligible
            )

            if valid_area > 0:

                mean = (
                    sum(
                        record["area"]
                        * record["properties"][field]
                        for record in eligible
                    )
                    / valid_area
                )

            else:
                mean = None

            count = len(eligible)

            if count == 0:
                status = "NONE"

            elif count == len(records):
                status = "COMPLETE"

            else:
                status = "PARTIAL"

            coverage_counts[metric][status] += 1

            # Same field name as original map,
            # but this is a display-only mean.

            props[field] = (
                round(mean, 10)
                if mean is not None
                else None
            )

            props[
                f"{metric}_valid_cell_count"
            ] = count

            props[
                f"{metric}_coverage_fraction"
            ] = round(
                valid_area / group_area,
                8,
            )

            props[
                f"{metric}_coverage_status"
            ] = status

        # ---------------------------------------------
        # 4J. CONVERT GEOMETRY TO WGS84 GEOJSON
        # ---------------------------------------------

        output_geom = shapely_transform(
            joined_geom,
            to_wgs84.transform,
            interleaved=False,
        )

        result_features.append(
            {
                "type": "Feature",

                "geometry": mapping(
                    output_geom
                ),

                "properties": props,
            }
        )

        projected_overview.append(
            joined_geom
        )

        overview_geom_total += (
            joined_geom.area
        )

        if index % 500 == 0:

            print(
                f"Processed {index}/{len(groups)} "
                "overview polygons",
                flush=True,
            )

    # =================================================
    # 5. INDEPENDENT GIS FOOTPRINT QA
    # =================================================

    print(
        "\nValidating full projected "
        "geometry coverage...",
        flush=True,
    )

    # Union all original GIS polygons.
    source_footprint = union_all(
        projected_source
    )

    # Union all aggregated GIS polygons.
    overview_footprint = union_all(
        projected_overview
    )

    # Compare the actual shapes.
    lost_or_added_area = (
        source_footprint
        .symmetric_difference(
            overview_footprint
        )
        .area
    )

    # Check overlap between display polygons.
    cross_group_overlap = (
        overview_geom_total
        - overview_footprint.area
    )

    source_geometric_total = sum(
        geom.area
        for geom in projected_source
    )

    area_disagreement = abs(
        source_geometric_total
        - original_reported_area
    )

    check(
        lost_or_added_area <= 5.0,
        "Overview footprint differs "
        f"from source by "
        f"{lost_or_added_area:.3f} m2",
    )

    check(
        cross_group_overlap <= max(
            10.0,
            original_reported_area * 0.00001,
        ),
        "Overlapping dissolved display "
        f"polygons: {cross_group_overlap:.3f} m2",
    )

    check(
        area_disagreement
        <= original_reported_area * 0.0005,
        "GeoJSON geometry versus GPKG area "
        f"mismatch: {area_disagreement:.3f} m2",
    )

    check(
        abs(
            original_reported_area / 1e6
            - 88.829216
        ) < 0.05,
        "Frozen source KMC area unexpected",
    )

    check(
        len(result_features) == 2410,
        "Output grid count unexpected",
    )

    check(
        sum(
            feature["properties"][
                "source_cell_count"
            ]
            for feature in result_features
        ) == 9240,
        "Original cells not fully accounted for",
    )

    # =================================================
    # 6. PRINT FINAL GIS QA
    # =================================================

    print("\n=== FINAL QA ===")

    print(
        "Original cells: 9240"
    )

    print(
        "Overview polygons: 2410"
    )

    print(
        "Reported source area:",
        f"{original_reported_area / 1e6:.6f}",
        "km2",
    )

    print(
        "Footprint symmetric difference:",
        f"{lost_or_added_area:.6f}",
        "m2",
    )

    print(
        "Cross-group overlap:",
        f"{cross_group_overlap:.6f}",
        "m2",
    )

    for metric in FIELDS:

        print(
            f"{metric}: "
            f"{dict(coverage_counts[metric])}"
        )

    # =================================================
    # 7. CREATE QA REPORT
    # =================================================

    report = {
        "status": "PASS_STEP_13G_02_GIS_OVERVIEW",

        "generated_utc":
            datetime.now(timezone.utc).isoformat(),

        "source_geojson": str(SOURCE),

        "source_geojson_sha256":
            sha256(SOURCE.read_bytes()).hexdigest(),

        "research_area_source": str(GIS),

        "research_area_layer": GIS_LAYER,

        "crs_for_area_validation": "EPSG:32644",

        "output_crs": "EPSG:4326",

        "research_resolution_m": 100,

        "display_resolution_m": 200,

        "display_only_not_model_predictions": True,

        "source_cell_count": len(original),

        "overview_group_count": len(result_features),

        "group_sizes": {
            str(key): value
            for key, value in sorted(
                group_sizes.items()
            )
        },

        "source_area_km2": round(
            original_reported_area / 1e6,
            9,
        ),

        "projected_geometry_area_km2": round(
            source_geometric_total / 1e6,
            9,
        ),

        "footprint_symmetric_difference_m2": round(
            lost_or_added_area,
            6,
        ),

        "cross_group_overlap_m2": round(
            cross_group_overlap,
            6,
        ),

        "source_scored_cells": dict(
            source_valid_counts
        ),

        "overview_group_coverage": {
            metric: {
                key: counts[key]
                for key in (
                    "COMPLETE",
                    "PARTIAL",
                    "NONE",
                )
            }
            for metric, counts
            in coverage_counts.items()
        },

        "aggregation": (
            "Mean of available original 100 m "
            "cell scores, weighted by original "
            "KMC-overlap area; no missing-value "
            "substitution"
        ),

        "geometry": (
            "Dissolve existing original 100 m "
            "polygons grouped by 2x2 row/column "
            "indices; no bounding rectangle expansion"
        ),
    }

    output_geojson = {
        "type": "FeatureCollection",
        "features": result_features,
    }

    # =================================================
    # 8. WRITE OUTPUTS AFTER ALL QA PASSES
    # =================================================

    # Frozen source files are never changed.
    # Only dashboard-specific output files
    # are created in a separate directory.

    DESTINATION.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_geojson = (
        OUT_GEOJSON.with_suffix(
            ".geojson.tmp"
        )
    )

    temp_report = (
        OUT_REPORT.with_suffix(
            ".json.tmp"
        )
    )

    try:

        temp_geojson.write_text(
            json.dumps(
                output_geojson,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

        temp_report.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Confirm both files can be read back.

        written = json.loads(
            temp_geojson.read_text(
                encoding="utf-8"
            )
        )

        check(
            len(written["features"]) == 2410,
            "Written GeoJSON read-back "
            "verification failed",
        )

        written_report = json.loads(
            temp_report.read_text(
                encoding="utf-8"
            )
        )

        check(
            written_report["status"]
            == "PASS_STEP_13G_02_GIS_OVERVIEW",
            "Written report read-back "
            "verification failed",
        )

        # Publish validated output files.

        temp_geojson.replace(
            OUT_GEOJSON
        )

        temp_report.replace(
            OUT_REPORT
        )

    finally:

        temp_geojson.unlink(
            missing_ok=True
        )

        temp_report.unlink(
            missing_ok=True
        )

    # =================================================
    # 9. TERMINAL SUCCESS
    # =================================================

    print(
        "\n[PASS] STEP 13G-2 "
        "GENERATED AND QA VERIFIED"
    )

    print(
        f"GeoJSON: {OUT_GEOJSON}"
    )

    print(
        f"QA report: {OUT_REPORT}"
    )

    print(
        "Source research/model files "
        "were not modified."
    )


# =====================================================
# 10. EXECUTION ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
