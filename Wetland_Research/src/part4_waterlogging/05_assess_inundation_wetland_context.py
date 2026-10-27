
from pathlib import Path
from datetime import datetime, timezone
import json
import xml.etree.ElementTree as ET

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from rasterio.features import geometry_mask
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from shapely.validation import explain_validity

# ==========================================================
# PART 4 - SEASONAL WATERLOGGING SUSCEPTIBILITY
# SCRIPT 05 - INUNDATION / WETLAND CONTEXT ASSESSMENT
#
# 1. Compare E001/E002 with date-near Part 1 wetlands.
# 2. Inspect 2020 six-class land-cover context.
# 3. Save auditable metrics and limitations.
#
# IMPORTANT:
# No negative or confirmed waterlogging labels created.
# No original data or other Part outputs modified.
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

EVENT_DIR = ROOT / "data/raw/waterlogging_events"

BOUNDARY = (
    ROOT / "data/raw/boundaries/Kaduwela TO Area.kml"
)

PART1 = (
    ROOT / "data/processed/part1_historical_change/"
    "historical_wetland_clean_observations_utm44n.gpkg"
)

LULC = (
    ROOT / "data/processed/part3_future_projection/"
    "LULC_2020_6Class_CommonMask.tif"
)

REPORT_DIR = ROOT / "outputs/reports/part4_waterlogging"
METRIC_DIR = ROOT / "outputs/metrics/part4_waterlogging"

EVENTS = {
    "E001": "2018-05-29",
    "E002": "2018-11-25",
}

MAX_REFERENCE_GAP_DAYS = 90
CRS_ANALYSIS = "EPSG:32644"

CLASSES = {
    1: "Wetland_Marsh",
    2: "Open_Water",
    3: "Built_Up",
    4: "Paddy_Agriculture",
    5: "Other_Vegetation",
    6: "Bare_Fill_Disturbed",
}


def read_coordinates(element):
    if element is None or not element.text:
        raise ValueError("Missing KML coordinates")

    points = []

    for token in element.text.strip().split():
        parts = token.split(",")

        if len(parts) < 2:
            raise ValueError("Malformed coordinate")

        lon = float(parts[0])
        lat = float(parts[1])

        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Invalid geographic coordinates")

        points.append((lon, lat))

    if len(set(points)) < 3:
        raise ValueError("Insufficient polygon vertices")

    return points



def read_event_polygon(event_id):
    """
    Read main event KML and repair invalid polygons
    for analysis only.

    Original geometry validity and repair decisions
    are recorded for the permanent audit.
    """
    path = EVENT_DIR / f"{event_id}.kml"

    root = ET.parse(path).getroot()

    polygons = []
    audit_records = []

    for index, polygon_node in enumerate(
        root.findall(".//{*}Polygon"),
        start=1,
    ):
        outer = polygon_node.find(
            "./{*}outerBoundaryIs/"
            "{*}LinearRing/{*}coordinates"
        )

        holes = []

        for inner in polygon_node.findall(
            "./{*}innerBoundaryIs/"
            "{*}LinearRing/{*}coordinates"
        ):
            holes.append(read_coordinates(inner))

        original = Polygon(
            read_coordinates(outer),
            holes=holes,
        )

        originally_valid = (
            original.is_valid
            and not original.is_empty
        )

        issue = (
            "None"
            if originally_valid
            else explain_validity(original)
        )

        if originally_valid:
            geometry = original
            repair_applied = False
        else:
            geometry = original.buffer(0)
            repair_applied = True

        if (
            geometry.is_empty
            or not geometry.is_valid
            or geometry.area <= 0
            or geometry.geom_type not in (
                "Polygon", "MultiPolygon"
            )
        ):
            raise ValueError(
                f"{event_id} polygon {index}: "
                "repair failed or geometry unusable"
            )

        polygons.append(geometry)

        audit_records.append({
            "event_id": event_id,
            "polygon_index": index,
            "originally_valid": originally_valid,
            "original_issue": issue,
            "repair_applied": repair_applied,
            "repair_method": (
                "buffer(0)"
                if repair_applied else "none"
            ),
            "original_geometry_type":
                original.geom_type,
            "analysis_geometry_type":
                geometry.geom_type,
            "geometry_type_changed": (
                original.geom_type != geometry.geom_type
            ),
            "analysis_geometry_valid":
                geometry.is_valid,
            "analysis_geometry_empty":
                geometry.is_empty,
            "review_required": repair_applied,
        })

    if not polygons:
        raise ValueError(
            f"No polygons in {path.name}"
        )

    union_wgs84 = unary_union(polygons)

    if (
        union_wgs84.is_empty
        or not union_wgs84.is_valid
    ):
        raise ValueError(
            f"{event_id}: unusable union geometry"
        )

    union_projected = gpd.GeoSeries(
        [union_wgs84],
        crs="EPSG:4326",
    ).to_crs(CRS_ANALYSIS).iloc[0]

    print(
        "Originally valid polygons:",
        sum(r["originally_valid"] for r in audit_records)
    )

    print(
        "Polygons repaired:",
        sum(r["repair_applied"] for r in audit_records)
    )

    print(
        "Geometry type changes:",
        sum(
            r["geometry_type_changed"]
            for r in audit_records
        )
    )

    return union_projected, audit_records


def select_nearby_wetlands(observations, event_date):
    """
    One nearest observation per wetland, within +/-90 days.
    On equal gaps, prefer earlier over later imagery.
    """
    target = pd.Timestamp(event_date)

    candidates = observations.copy()

    candidates["_signed_gap"] = (
        candidates["observation_date"] - target
    ).dt.days

    candidates["_absolute_gap"] = (
        candidates["_signed_gap"].abs()
    )

    candidates = candidates[
        candidates["_absolute_gap"]
        <= MAX_REFERENCE_GAP_DAYS
    ].copy()

    candidates["_is_future"] = (
        candidates["_signed_gap"] > 0
    )

    candidates = candidates.sort_values(
        [
            "wetland_id",
            "_absolute_gap",
            "_is_future",
            "observation_date",
        ]
    )

    return candidates.drop_duplicates(
        subset=["wetland_id"],
        keep="first",
    ).copy()


def main():

    print("=" * 70)
    print("PART 4 - HISTORICAL INUNDATION / WETLAND CONTEXT")
    print("SCRIPT 05")
    print("=" * 70)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    # Load and validate analysis references.
    gn = gpd.read_file(
        BOUNDARY,
        layer="GN Boundary Colombo District (#2)",
    ).to_crs(CRS_ANALYSIS)

    if len(gn) != 56:
        raise ValueError(
            f"Expected 56 GN features, found {len(gn)}"
        )

    study_area = unary_union(gn.geometry)

    observations = gpd.read_file(PART1)

    if observations.crs is None:
        raise ValueError("Part 1 reference CRS missing")

    observations = observations.to_crs(CRS_ANALYSIS)

    observations["observation_date"] = (
        pd.to_datetime(
            observations["observation_date"],
            errors="raise",
        )
    )

    summary_rows = []
    reference_rows = []
    lulc_rows = []
    geometry_audit_rows = []

    with rasterio.open(LULC) as src:

        if str(src.crs) != CRS_ANALYSIS:
            raise ValueError("2020 LULC CRS mismatch")

        landcover = src.read(1)

        pixel_area_ha = (
            abs(src.res[0] * src.res[1]) / 10000
        )

        for event_id, event_date in EVENTS.items():

            print("\n" + "-" * 70)
            print("EVENT:", event_id, event_date)

            event_union, repair_records = read_event_polygon(event_id)

            geometry_audit_rows.extend(repair_records)

            # Precise polygon geometry inside the GN study area.
            event_inside = event_union.intersection(
                study_area
            )

            if event_inside.is_empty:
                raise ValueError(
                    f"{event_id} has no study-area overlap"
                )

            event_area_ha = (
                event_inside.area / 10000
            )

            # ----------------------------------------------
            # PART 1: Time-restricted historical references
            # ----------------------------------------------

            selected = select_nearby_wetlands(
                observations,
                event_date,
            )

            reference_geometries = []

            for _, row in selected.iterrows():

                geom = row.geometry

                if geom is None or geom.is_empty:
                    raise ValueError(
                        "Empty Part 1 reference geometry"
                    )

                if not geom.is_valid:
                    raise ValueError(
                        "Invalid Part 1 reference geometry"
                    )

                geom = geom.intersection(study_area)
                reference_geometries.append(geom)

                overlap = geom.intersection(
                    event_inside
                ).area / 10000
                distance_m = geom.distance(event_inside)

                reference_rows.append({
                    "event_id": event_id,
                    "event_date": event_date,
                    "wetland_id": row["wetland_id"],
                    "reference_date":
                        row["observation_date"].date().isoformat(),
                    "signed_gap_days":
                        int(row["_signed_gap"]),
                    "absolute_gap_days":
                        int(row["_absolute_gap"]),
                    "reference_area_ha":
                        round(geom.area / 10000, 6),
                    "event_overlap_ha":
                        round(overlap, 6),
                    "distance_to_event_m":
                        round(distance_m, 4),
                })

            if reference_geometries:

                reference_union = unary_union(
                    reference_geometries
                )

                overlap_ha = (
                    event_inside
                    .intersection(reference_union)
                    .area / 10000
                )

                outside_reference_ha = (
                    event_area_ha - overlap_ha
                )

                overlap_pct = (
                    100 * overlap_ha / event_area_ha
                )

            else:
                # No appropriate temporal reference.
                # Do NOT treat missing evidence as zero
                # actual wetlands.
                overlap_ha = None
                outside_reference_ha = None
                overlap_pct = None

            # ----------------------------------------------
            # PART 3: 2020 land-cover context
            # ----------------------------------------------

            sampled_mask = geometry_mask(
                [mapping(event_inside)],
                out_shape=landcover.shape,
                transform=src.transform,
                invert=True,
                all_touched=False,
            )

            sampled = landcover[sampled_mask]

            valid = sampled[
                np.isin(sampled, list(CLASSES))
            ]

            if len(valid) == 0:
                raise ValueError(
                    f"No valid 2020 LULC pixels for {event_id}"
                )

            class_counts = {}

            for class_id, class_name in CLASSES.items():

                count = int(
                    np.count_nonzero(valid == class_id)
                )

                class_counts[class_id] = count

                lulc_rows.append({
                    "event_id": event_id,
                    "event_date": event_date,
                    "lulc_year": 2020,
                    "class_id": class_id,
                    "class_name": class_name,
                    "pixel_count": count,
                    "sampled_area_ha":
                        round(count * pixel_area_ha, 4),
                    "percent_valid_sampled_pixels":
                        round(
                            100 * count / len(valid),
                            4,
                        ),
                    "method":
                        "30m raster cell-center sampling",
                })

            wetland_water_count = (
                class_counts[1] + class_counts[2]
            )

            selected_days = (
                selected["_absolute_gap"].tolist()
            )

            summary = {
                "event_id": event_id,
                "event_date": event_date,
                "event_inside_study_area_ha":
                    round(event_area_ha, 4),
                "part1_reference_window_days":
                    MAX_REFERENCE_GAP_DAYS,
                "part1_nearest_reference_wetlands":
                    len(selected),
                "part1_references_within_30_days":
                    sum(
                        d <= 30 for d in selected_days
                    ),
                "part1_exact_date_references":
                    sum(
                        d == 0 for d in selected_days
                    ),
                "part1_reference_overlap_ha":
                    (
                        round(overlap_ha, 4)
                        if overlap_ha is not None
                        else None
                    ),
                "outside_selected_part1_reference_ha":
                    (
                        round(outside_reference_ha, 4)
                        if outside_reference_ha is not None
                        else None
                    ),
                "part1_reference_overlap_percent":
                    (
                        round(overlap_pct, 4)
                        if overlap_pct is not None
                        else None
                    ),
                "lulc_2020_valid_sampled_pixels":
                    len(valid),
                "lulc_2020_nodata_sampled_pixels":
                    int(len(sampled) - len(valid)),
                "lulc_2020_wetland_openwater_percent":
                    round(
                        100 * wetland_water_count
                        / len(valid),
                        4,
                    ),
            }

            summary_rows.append(summary)

            print(
                "Event area inside Kaduwela:",
                round(event_area_ha, 2),
                "ha"
            )
            print(
                "Nearest Part 1 reference wetlands:",
                len(selected)
            )
            print(
                "References within 30 days:",
                summary[
                    "part1_references_within_30_days"
                ]
            )
            print(
                "Part 1 spatial overlap:",
                (
                    f"{overlap_ha:.2f} ha"
                    if overlap_ha is not None
                    else "UNAVAILABLE"
                )
            )
            print(
                "2020 LULC valid sampled pixels:",
                len(valid)
            )
            print(
                "2020 wetland/open-water context:",
                summary[
                    "lulc_2020_wetland_openwater_percent"
                ],
                "%"
            )

    # ------------------------------------------------------
    # Save reproducible analysis evidence
    # ------------------------------------------------------

    geometry_audit_path = (
        METRIC_DIR
        / "script05_event_geometry_repair_audit.csv"
    )

    pd.DataFrame(geometry_audit_rows).to_csv(
        geometry_audit_path,
        index=False,
    )

    print("Saved geometry audit:", geometry_audit_path)

    summary_path = (
        METRIC_DIR
        / "script05_event_context_summary.csv"
    )

    reference_path = (
        METRIC_DIR
        / "script05_selected_wetland_references.csv"
    )

    lulc_path = (
        METRIC_DIR
        / "script05_event_lulc_2020_context.csv"
    )

    report_path = (
        REPORT_DIR
        / "script05_inundation_wetland_context.json"
    )

    pd.DataFrame(summary_rows).to_csv(
        summary_path,
        index=False,
    )

    pd.DataFrame(reference_rows).to_csv(
        reference_path,
        index=False,
    )

    pd.DataFrame(lulc_rows).to_csv(
        lulc_path,
        index=False,
    )

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage":
            "Historical inundation versus wetland context",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "analysis_crs": CRS_ANALYSIS,
        "events": summary_rows,
        "part1_reference_policy": {
            "source":
                "Part 1 clean historical wetland observations",
            "maximum_temporal_gap_days":
                MAX_REFERENCE_GAP_DAYS,
            "selection":
                "One nearest observation per wetland",
            "tie_break":
                "Prefer earlier imagery for equal gaps",
            "interpretation":
                "Selected wetlands are incomplete "
                "historical context, not exhaustive "
                "wetland ground truth.",
        },
        "lulc_reference_policy": {
            "source":
                "Part 3 historical LULC 2020",
            "spatial_resolution_m": 30,
            "sampling": "Raster cell centers",
            "interpretation":
                "2020 land-cover context only; "
                "not confirmed 2018 land cover.",
        },
        "interpretation_limitations": [
            "Only 25 selected wetlands are represented "
            "in Part 1.",
            "No Part 1 observations are available "
            "within 90 days of E001.",
            "Part 1 observations near E002 may "
            "share imagery with the event polygons "
            "and are not independent validation.",
            "2020 Open Water is not proof of permanent "
            "water coverage in 2018.",
            "2020 Wetland/Marsh is not proof of "
            "waterlogging during either event.",
            "Raster sampled areas are approximate "
            "because boundaries cross 30m cells.",
            "Cloud cover and manual image "
            "interpretation introduce uncertainty.",
            "Outside sampled wetlands does not mean "
            "confirmed non-wetland.",
            "No confirmed waterlogging or "
            "non-waterlogging labels are assigned.",
        ],
        "status":
            "CONTEXT_ASSESSMENT_COMPLETE_"
            "LABEL_RELIABILITY_PENDING",
        "raw_data_modified": False,

        "geometry_repair_policy": (
        "Invalid event polygons were repaired "
        "in memory using buffer(0). Original "
        "KML geometries were not modified. "
        "Repaired boundaries still require "
        "interpretation-quality review."
    ),
    "geometry_repair_summary": {
        "total_polygons": len(geometry_audit_rows),
        "originally_invalid": sum(
            not r["originally_valid"]
            for r in geometry_audit_rows
        ),
        "repaired": sum(
            r["repair_applied"]
            for r in geometry_audit_rows
        ),
        "geometry_type_changes": sum(
            r["geometry_type_changed"]
            for r in geometry_audit_rows
        ),
        "spatial_proximity_interpretation": (
            "Exact geometric overlap and distance to the "
            "historical event footprint are reported separately. "
            "A small positive distance indicates proximity, "
            "not confirmed overlap or independent evidence "
            "of waterlogging. Manual digitization uncertainty "
            "must be considered."
        ),
    },
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 05 - SUMMARY")
    print("=" * 70)

    print("Events assessed:", len(summary_rows))
    print(
        "Part 1 reference observations selected:",
        len(reference_rows)
    )
    print("LULC context rows:", len(lulc_rows))

    print("\nSaved:", summary_path)
    print("Saved:", reference_path)
    print("Saved:", lulc_path)
    print("Saved:", report_path)

    print(
        "\nCONTEXT ASSESSMENT: COMPLETE"
    )
    print("WATERLOGGING LABEL RELIABILITY: PENDING")
    print("RAW DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
