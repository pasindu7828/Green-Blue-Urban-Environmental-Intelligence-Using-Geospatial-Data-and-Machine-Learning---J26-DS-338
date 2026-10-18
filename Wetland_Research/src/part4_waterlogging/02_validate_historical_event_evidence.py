
from pathlib import Path
from datetime import datetime, timezone
import json
import xml.etree.ElementTree as ET

import geopandas as gpd
import pandas as pd

from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform, unary_union


# ============================================================
# PART 4 - SEASONAL WATERLOGGING SUSCEPTIBILITY
# SCRIPT 02 - HISTORICAL EVENT EVIDENCE VALIDATION
#
# Main E001/E002 KMLs are primary evidence.
# Individual KML files are cross-checked for duplication.
#
# Structural/spatial validation only.
# No negative waterlogging labels are generated.
# Raw KML files are not modified.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

EVENT_DIR = ROOT / "data/raw/waterlogging_events"

BOUNDARY_FILE = (
    ROOT / "data/raw/boundaries/Kaduwela TO Area.kml"
)

GN_LAYER = "GN Boundary Colombo District (#2)"

METRIC_DIR = (
    ROOT / "outputs/metrics/part4_waterlogging"
)

REPORT_DIR = (
    ROOT / "outputs/reports/part4_waterlogging"
)

EVENTS = {
    "E001": {
        "date": "2018-05-29",
        "folder": "E001_2018_05_29",
    },
    "E002": {
        "date": "2018-11-25",
        "folder": "E002_2018_11_25",
    },
}

CRS_GEOGRAPHIC = "EPSG:4326"
CRS_ANALYSIS = "EPSG:32644"

transformer = Transformer.from_crs(
    CRS_GEOGRAPHIC,
    CRS_ANALYSIS,
    always_xy=True,
)


def read_ring(element):
    if element is None or not element.text:
        raise ValueError("Missing polygon coordinates")

    points = []

    for token in element.text.strip().split():
        values = token.split(",")

        if len(values) < 2:
            raise ValueError("Invalid coordinate token")

        lon = float(values[0])
        lat = float(values[1])

        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Coordinates outside lon/lat range")

        points.append((lon, lat))

    if len(set(points)) < 3:
        raise ValueError("Fewer than three distinct vertices")

    return points


def parse_kml(path):
    """
    Read Polygon geometries from KML placemarks.
    Handles polygon holes and multiple polygons.

    A placemark may contain more than one polygon.
    """

    root = ET.parse(path).getroot()

    records = []

    placemarks = root.findall(".//{*}Placemark")

    for placemark_number, placemark in enumerate(
        placemarks, start=1
    ):

        name_element = placemark.find("./{*}name")

        name = (
            name_element.text.strip()
            if name_element is not None
            and name_element.text
            else f"Unnamed_{placemark_number}"
        )

        polygons = placemark.findall(".//{*}Polygon")

        for polygon_number, element in enumerate(
            polygons, start=1
        ):

            try:
                outer = element.find(
                    "./{*}outerBoundaryIs/"
                    "{*}LinearRing/{*}coordinates"
                )

                exterior = read_ring(outer)

                holes = []

                for inner in element.findall(
                    "./{*}innerBoundaryIs/"
                    "{*}LinearRing/{*}coordinates"
                ):
                    holes.append(read_ring(inner))

                geometry = Polygon(
                    exterior,
                    holes=holes,
                )

                records.append({
                    "placemark": name,
                    "placemark_number": placemark_number,
                    "polygon_number": polygon_number,
                    "geometry": geometry,
                    "parse_error": None,
                })

            except Exception as exc:
                records.append({
                    "placemark": name,
                    "placemark_number": placemark_number,
                    "polygon_number": polygon_number,
                    "geometry": None,
                    "parse_error": str(exc),
                })

    return records, len(placemarks)


def valid_for_area(geom):
    """
    Attempt validity repair for measurements only.
    This does not modify or replace original KML data.
    """

    if geom is None or geom.is_empty:
        return None

    if geom.is_valid:
        candidate = geom
    else:
        candidate = geom.buffer(0)

    if (
        candidate.is_empty
        or not candidate.is_valid
        or candidate.area <= 0
        or candidate.geom_type not in
        ("Polygon", "MultiPolygon")
    ):
        return None

    return candidate


def projected(geom):
    return transform(transformer.transform, geom)


def count_duplicate_geometries(geometries):
    """
    Count repeated geometries using geometric equality.
    Does not count overlapping but different polygons
    as exact duplicates.
    """

    unique = []
    duplicates = 0

    for geom in geometries:
        if any(geom.equals(other) for other in unique):
            duplicates += 1
        else:
            unique.append(geom)

    return duplicates


def main():

    print("=" * 70)
    print("PART 4 - HISTORICAL INUNDATION EVIDENCE VALIDATION")
    print("SCRIPT 02")
    print("=" * 70)

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load study-area GN polygons
    # --------------------------------------------------------

    gn = gpd.read_file(
        BOUNDARY_FILE,
        layer=GN_LAYER,
    )

    if len(gn) != 56:
        raise ValueError(
            f"Expected 56 GN features, found {len(gn)}"
        )

    if gn.crs is None:
        raise ValueError("GN CRS is missing")

    gn = gn.to_crs(CRS_ANALYSIS)

    if gn.geometry.isna().any():
        raise ValueError("Missing GN geometry")

    gn_geometries = []

    for geom in gn.geometry:
        valid_geom = valid_for_area(geom)

        if valid_geom is None:
            raise ValueError("Unusable GN geometry")

        gn_geometries.append(valid_geom)

    study_area = unary_union(gn_geometries)

    if study_area.is_empty:
        raise ValueError("Empty study area")

    print("\nSTUDY AREA")
    print("GN features:", len(gn))
    print("Analysis CRS:", CRS_ANALYSIS)
    print(
        "GN union area:",
        f"{study_area.area / 10000:.2f} ha"
    )

    # --------------------------------------------------------
    # 2. Inspect each historical event
    # --------------------------------------------------------

    polygon_rows = []
    event_rows = []
    issues = []
    event_unions = {}

    for event_id, info in EVENTS.items():

        print("\n" + "-" * 70)
        print(
            f"EVENT {event_id} - {info['date']}"
        )
        print("-" * 70)

        main_path = EVENT_DIR / f"{event_id}.kml"

        if not main_path.is_file():
            raise FileNotFoundError(main_path)

        main_records, main_placemark_count = parse_kml(
            main_path
        )

        child_folder = EVENT_DIR / info["folder"]

        child_paths = sorted(
            child_folder.glob("*.kml")
        )

        if not child_paths:
            issues.append(
                f"{event_id}: no individual KML files"
            )

        main_projected = []
        main_geographic = []

        invalid_count = 0
        parse_errors = 0
        outside_count = 0

        for index, record in enumerate(
            main_records, start=1
        ):

            raw_geom = record["geometry"]

            parse_error = record["parse_error"]

            if parse_error is not None:
                parse_errors += 1

            originally_valid = (
                raw_geom is not None
                and not raw_geom.is_empty
                and raw_geom.is_valid
            )

            usable = valid_for_area(raw_geom)

            area_ha = None
            inside_ha = None
            outside_ha = None
            inside_fraction = None

            if usable is None:
                invalid_count += 1

            else:
                main_geographic.append(usable)

                geom_utm = projected(usable)

                main_projected.append(geom_utm)

                area_ha = geom_utm.area / 10000

                clipped_area = (
                    geom_utm.intersection(study_area)
                ).area

                inside_ha = clipped_area / 10000

                outside_ha = (
                    geom_utm.area - clipped_area
                ) / 10000

                inside_fraction = (
                    clipped_area / geom_utm.area
                    if geom_utm.area > 0
                    else None
                )

                if inside_ha <= 0:
                    outside_count += 1

            polygon_rows.append({
                "event_id": event_id,
                "event_date": info["date"],
                "polygon_index": index,
                "placemark": record["placemark"],
                "placemark_number":
                    record["placemark_number"],
                "polygon_number":
                    record["polygon_number"],
                "original_geometry_valid":
                    originally_valid,
                "usable_for_spatial_audit":
                    usable is not None,
                "parse_error": parse_error,
                "total_area_ha": area_ha,
                "inside_study_area_ha": inside_ha,
                "outside_study_area_ha": outside_ha,
                "fraction_inside_study_area":
                    inside_fraction,
            })

        # ----------------------------------------------------
        # 3. Compare individual KMLs to main event KML
        # ----------------------------------------------------

        child_geometries = []
        child_parse_errors = 0

        for child_path in child_paths:

            try:
                child_records, _ = parse_kml(
                    child_path
                )

                for record in child_records:
                    usable = valid_for_area(
                        record["geometry"]
                    )

                    if usable is not None:
                        child_geometries.append(usable)
                    else:
                        child_parse_errors += 1

            except Exception:
                child_parse_errors += 1

        child_matches = sum(
            any(
                child_geom.equals(main_geom)
                for main_geom in main_geographic
            )
            for child_geom in child_geometries
        )

        duplicate_main = count_duplicate_geometries(
            main_geographic
        )

        if main_projected:
            union = unary_union(main_projected)

            total_individual_area_ha = (
                sum(g.area for g in main_projected)
                / 10000
            )

            unique_union_ha = union.area / 10000

            inside_union = union.intersection(
                study_area
            )

            inside_union_ha = (
                inside_union.area / 10000
            )

            outside_union_ha = (
                union.area - inside_union.area
            ) / 10000

            within_event_overlap_ha = (
                total_individual_area_ha
                - unique_union_ha
            )

            event_unions[event_id] = inside_union

        else:
            unique_union_ha = 0
            inside_union_ha = 0
            outside_union_ha = 0
            within_event_overlap_ha = 0

        event_rows.append({
            "event_id": event_id,
            "event_date": info["date"],
            "main_placemarks": main_placemark_count,
            "main_polygon_entries": len(main_records),
            "usable_main_polygons":
                len(main_projected),
            "main_invalid_or_unusable":
                invalid_count,
            "main_parse_errors": parse_errors,
            "main_duplicate_geometries":
                duplicate_main,
            "fully_outside_polygon_count":
                outside_count,
            "individual_kml_files":
                len(child_paths),
            "usable_child_polygons":
                len(child_geometries),
            "child_parse_or_geometry_errors":
                child_parse_errors,
            "child_polygons_matching_main":
                child_matches,
            "unique_event_union_ha":
                unique_union_ha,
            "union_inside_study_area_ha":
                inside_union_ha,
            "union_outside_study_area_ha":
                outside_union_ha,
            "within_event_overlap_ha":
                within_event_overlap_ha,
        })

        print(
            "Main KML placemarks:",
            main_placemark_count
        )
        print(
            "Main polygon entries:",
            len(main_records)
        )
        print(
            "Usable main polygons:",
            len(main_projected)
        )
        print(
            "Invalid/unusable:",
            invalid_count
        )
        print(
            "Exact duplicate main geometries:",
            duplicate_main
        )
        print(
            "Child geometries matching main:",
            f"{child_matches}/{len(child_geometries)}"
        )
        print(
            "Unique mapped area:",
            f"{unique_union_ha:.2f} ha"
        )
        print(
            "Mapped area inside study area:",
            f"{inside_union_ha:.2f} ha"
        )
        print(
            "Mapped area outside study area:",
            f"{outside_union_ha:.2f} ha"
        )
        print(
            "Within-event polygon overlap:",
            f"{within_event_overlap_ha:.2f} ha"
        )

        if invalid_count or parse_errors:
            issues.append(
                f"{event_id}: unusable or unparsed polygons"
            )

        if child_parse_errors:
            issues.append(
                f"{event_id}: child KML inspection errors"
            )

        if not main_projected:
            issues.append(
                f"{event_id}: no usable event polygons"
            )

        if child_geometries and (
            child_matches != len(child_geometries)
        ):
            issues.append(
                f"{event_id}: some child polygons differ "
                "from main event geometries"
            )

    # --------------------------------------------------------
    # 4. Cross-event geographic overlap
    # --------------------------------------------------------

    cross_event_overlap_ha = None

    if all(key in event_unions for key in EVENTS):

        cross_event_overlap_ha = (
            event_unions["E001"]
            .intersection(event_unions["E002"])
            .area / 10000
        )

    print("\n" + "=" * 70)
    print("CROSS-EVENT COMPARISON")
    print("=" * 70)

    print(
        "E001/E002 spatial overlap:",
        cross_event_overlap_ha,
        "ha"
    )

    print("Independent event dates:", len(EVENTS))

    # --------------------------------------------------------
    # 5. Save scientific audit evidence
    # --------------------------------------------------------

    polygon_path = (
        METRIC_DIR
        / "script02_event_polygon_audit.csv"
    )

    event_path = (
        METRIC_DIR
        / "script02_event_summary.csv"
    )

    report_path = (
        REPORT_DIR
        / "script02_event_validation.json"
    )

    pd.DataFrame(polygon_rows).to_csv(
        polygon_path,
        index=False,
    )

    pd.DataFrame(event_rows).to_csv(
        event_path,
        index=False,
    )

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "Historical event spatial evidence audit",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "event_dates_from_collection_metadata": {
            event_id: info["date"]
            for event_id, info in EVENTS.items()
        },
        "date_note": (
            "Event dates are assigned from dataset "
            "collection metadata and filenames; "
            "they are not independently proven by "
            "the polygon geometries."
        ),
        "primary_geometry_source": (
            "Main E001.kml and E002.kml files"
        ),
        "individual_file_role": (
            "Duplicate/consistency cross-check only"
        ),
        "analysis_crs": CRS_ANALYSIS,
        "gn_divisions": len(gn),
        "event_summaries": event_rows,
        "cross_event_overlap_ha":
            cross_event_overlap_ha,
        "issues": issues,
        "raw_data_modified": False,
        "negative_labels_created": False,
        "supervised_ml_feasibility": "NOT_YET_ASSESSED",
        "scientific_limitations": [
            "Only two known event dates are available.",
            "Mapped polygons may represent interpreted "
            "inundation rather than verified waterlogging.",
            "Polygon absence cannot be interpreted "
            "automatically as no waterlogging.",
            "Event polygons may represent spatially "
            "correlated observations.",
            "Geometry validity does not establish "
            "correct environmental interpretation.",
            "No rainfall-event relationship is validated "
            "in this script.",
            "No supervised model accuracy is established.",
        ],
        "validation_status": (
            "REVIEW_REQUIRED" if issues
            else "SPATIAL_STRUCTURE_PASS_"
                 "EVIDENCE_RELIABILITY_PENDING"
        ),
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 02 - VALIDATION SUMMARY")
    print("=" * 70)

    for issue in issues:
        print("REVIEW:", issue)

    print("Issues identified:", len(issues))
    print("Saved:", polygon_path)
    print("Saved:", event_path)
    print("Saved:", report_path)

    print(
        "\nEVENT SPATIAL AUDIT:",
        report["validation_status"]
    )

    print("LABEL RELIABILITY: NOT YET ESTABLISHED")
    print("ML FEASIBILITY: NOT YET ASSESSED")
    print("RAW DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
