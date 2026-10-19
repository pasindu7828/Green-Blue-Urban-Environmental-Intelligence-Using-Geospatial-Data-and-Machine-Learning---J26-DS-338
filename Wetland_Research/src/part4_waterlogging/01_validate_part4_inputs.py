import os
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import csv
import json
import sqlite3
import xml.etree.ElementTree as ET

import rasterio


# ==========================================================
# PART 4 - SEASONAL WATERLOGGING SUSCEPTIBILITY
# SCRIPT 01 - INITIAL INPUT INVENTORY AND VALIDATION
#
# Read-only inspection of raw data.
# Saves reproducible input validation evidence.
#
# This is NOT a scientific validation of waterlogging
# labels or rainfall measurements.
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw"

EVENT_DIR = RAW / "waterlogging_events"
RAIN_DIR = RAW / "rainfall" / "imerg"
DEM_DIR = RAW / "dem"
OSM_DIR = RAW / "osm"
BOUNDARY_DIR = RAW / "boundaries"
KMC_DIR = RAW / "kmc"

REPORT_DIR = ROOT / "outputs/reports/part4_waterlogging"
METRIC_DIR = ROOT / "outputs/metrics/part4_waterlogging"

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


def relative_path(path):
    return str(path.relative_to(ROOT))


def inspect_kml(path):
    """
    Count KML placemarks and their geometry elements.

    A placemark can contain multiple geometry elements.
    These are structural counts, not confirmed event labels.
    """
    counts = Counter()
    errors = []

    try:
        parser = ET.iterparse(path, events=("end",))

        for _, element in parser:
            tag = element.tag.rsplit("}", 1)[-1]

            if tag != "Placemark":
                continue

            counts["placemarks"] += 1
            found = set()

            for child in element.iter():
                child_tag = child.tag.rsplit("}", 1)[-1]

                if child_tag in {
                    "Polygon",
                    "LineString",
                    "Point",
                    "MultiGeometry",
                }:
                    found.add(child_tag)

            for geometry_type in found:
                counts[geometry_type] += 1

            element.clear()

    except Exception as exc:
        errors.append(
            f"{type(exc).__name__}: {str(exc)}"
        )

    return {
        "placemarks": counts["placemarks"],
        "polygon_placemarks": counts["Polygon"],
        "line_placemarks": counts["LineString"],
        "point_placemarks": counts["Point"],
        "multigeometry_placemarks": counts["MultiGeometry"],
        "parse_errors": errors,
    }


def inspect_dem(path):
    with rasterio.open(path) as src:
        return {
            "file": relative_path(path),
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "resolution": list(src.res),
            "nodata": src.nodata,
            "bounds": [
                float(v) for v in src.bounds
            ],
        }


def inspect_geopackage(path):
    """
    Inspect standard GeoPackage metadata directly using
    read-only SQLite. Does not change the GeoPackage.
    """
    layers = []

    uri = path.resolve().as_uri() + "?mode=ro"

    with sqlite3.connect(uri, uri=True) as connection:

        query = """
            SELECT
                c.table_name,
                c.data_type,
                g.geometry_type_name,
                g.srs_id
            FROM gpkg_contents AS c
            LEFT JOIN gpkg_geometry_columns AS g
                ON c.table_name = g.table_name
            ORDER BY c.table_name
        """

        for table, data_type, geom_type, srs in (
            connection.execute(query)
        ):
            safe_table = table.replace('"', '""')

            count = connection.execute(
                f'SELECT COUNT(*) FROM "{safe_table}"'
            ).fetchone()[0]

            layers.append({
                "layer": table,
                "data_type": data_type,
                "geometry_type": geom_type,
                "srs_id": srs,
                "record_count": int(count),
            })

    return {
        "file": relative_path(path),
        "layers": layers,
    }


def main():

    print("=" * 68)
    print("PART 4 - INITIAL INPUT VALIDATION")
    print("=" * 68)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    checks = {}
    event_records = []
    rainfall_records = []
    dem_records = []
    gpkg_records = []
    issues = []

    # ------------------------------------------------------
    # 1. Historical inundation evidence
    # ------------------------------------------------------

    print("\n1. HISTORICAL EVENT EVIDENCE")

    for event_id, info in EVENTS.items():

        main_file = EVENT_DIR / f"{event_id}.kml"
        event_folder = EVENT_DIR / info["folder"]

        main_exists = main_file.is_file()
        child_files = (
            sorted(event_folder.glob("*.kml"))
            if event_folder.exists()
            else []
        )

        checks[f"{event_id}_main_kml_exists"] = main_exists
        checks[f"{event_id}_child_kml_count"] = len(child_files)

        print(
            f"{event_id} ({info['date']}): "
            f"main={main_exists}, "
            f"child KMLs={len(child_files)}"
        )

        if not main_exists or not child_files:
            issues.append(
                f"{event_id}: missing event evidence"
            )

        files = (
            ([main_file] if main_exists else [])
            + child_files
        )

        for path in files:

            inspection = inspect_kml(path)

            record = {
                "event_id": event_id,
                "event_date": info["date"],
                "file": relative_path(path),
                "role": (
                    "main"
                    if path == main_file
                    else "individual"
                ),
                "size_bytes": path.stat().st_size,
                **inspection,
            }

            event_records.append(record)

            if inspection["parse_errors"]:
                issues.append(
                    f"KML parse error: {relative_path(path)}"
                )

        main_results = [
            record for record in event_records
            if record["event_id"] == event_id
            and record["role"] == "main"
        ]

        if main_results:
            print(
                "  Main KML placemarks:",
                main_results[0]["placemarks"]
            )
            print(
                "  Main KML polygon placemarks:",
                main_results[0]["polygon_placemarks"]
            )

    # ------------------------------------------------------
    # 2. Rainfall inventory
    # ------------------------------------------------------

    print("\n2. IMERG RAINFALL INVENTORY")

    rainfall_groups = {
        "E001_daily": RAIN_DIR / "E001",
        "E001_half_hourly":
            RAIN_DIR / "E001 half-hourly rainfall",
        "E002_daily": RAIN_DIR / "E002",
        "E002_half_hourly":
            RAIN_DIR / "E002 half-hourly rainfall",
    }

    
    for group_name, folder in rainfall_groups.items():

        # Use os.scandir because pathlib.is_file()
        # may incorrectly return False for long
        # Windows paths in this repository.

        if folder.exists():
            with os.scandir(folder) as entries:
                files = sorted(
                    [
                        entry
                        for entry in entries
                        if entry.is_file()
                    ],
                    key=lambda entry: entry.name.lower()
                )
        else:
            files = []

        counts = Counter(
            Path(entry.name).suffix.lower()
            for entry in files
        )

        checks[f"{group_name}_file_counts"] = dict(counts)

        print(
            group_name,
            "| CSV:", counts[".csv"],
            "| NC:", counts[".nc"],
            "| PNG:", counts[".png"]
        )

        if counts[".csv"] < 1 or counts[".nc"] < 1:
            issues.append(
                f"{group_name}: expected CSV/NetCDF missing"
            )

        for entry in files:

            rainfall_records.append({
                "group": group_name,
                "file": relative_path(
                    Path(entry.path)
                ),
                "extension": Path(entry.name).suffix.lower(),
                "size_bytes": entry.stat().st_size,
            })

    # ------------------------------------------------------
    # 3. DEM metadata
    # ------------------------------------------------------

    print("\n3. TERRAIN DATA")

    for name in [
        "GEDTM30_Kaduwela_Buffer.tif",
        "NASADEM_Kaduwela_Buffer.tif",
    ]:

        path = DEM_DIR / name

        if path.is_file():
            try:
                result = inspect_dem(path)
                dem_records.append(result)

                print(
                    name,
                    "| CRS:", result["crs"],
                    "| Resolution:", result["resolution"]
                )
            except Exception as exc:
                issues.append(
                    f"DEM inspection failed: {name}: {exc}"
                )
        else:
            issues.append(f"DEM missing: {name}")

    # ------------------------------------------------------
    # 4. OSM GeoPackage metadata
    # ------------------------------------------------------

    print("\n4. OSM AND DRAINAGE LAYERS")

    for name in [
        "Kaduwela_OSM_Roads_Buildings_2026.gpkg",
        "Kaduwela_OSM_Water_Drainage_2026.gpkg",
    ]:

        path = OSM_DIR / name

        if path.is_file():
            try:
                result = inspect_geopackage(path)
                gpkg_records.append(result)

                print(name)

                for layer in result["layers"]:
                    print(
                        "  ",
                        layer["layer"],
                        "|",
                        layer["geometry_type"],
                        "|",
                        layer["record_count"],
                        "records"
                    )
            except Exception as exc:
                issues.append(
                    f"GeoPackage inspection failed: "
                    f"{name}: {exc}"
                )
        else:
            issues.append(f"GeoPackage missing: {name}")

    # ------------------------------------------------------
    # 5. Boundary and supporting data
    # ------------------------------------------------------

    print("\n5. BOUNDARY AND KMC SUPPORTING DATA")

    boundary = (
        BOUNDARY_DIR / "Kaduwela TO Area.kml"
    )

    checks["boundary_file_exists"] = boundary.is_file()

    print("Kaduwela boundary exists:", boundary.is_file())

    if not boundary.is_file():
        issues.append("Kaduwela boundary missing")

    kmc_files = (
        list(KMC_DIR.rglob("*"))
        if KMC_DIR.exists()
        else []
    )

    kmc_files = [
        file for file in kmc_files if file.is_file()
    ]

    checks["kmc_file_count"] = len(kmc_files)

    print("KMC support files:", len(kmc_files))
    print(
        "KMC/CRIP support data are optional contextual "
        "inputs at this stage."
    )

    # ------------------------------------------------------
    # 6. Save validation evidence
    # ------------------------------------------------------

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "Initial source inventory",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "source_data_modified": False,
        "checks": checks,
        "event_kml_file_count": len(event_records),
        "rainfall_file_count": len(rainfall_records),
        "dem_metadata": dem_records,
        "geopackage_metadata": gpkg_records,
        "issues": issues,
        "scientific_validation_pending": [
            "Event polygon geometry validity and coverage",
            "Duplicate and overlapping event evidence",
            "Event evidence interpretation and reliability",
            "IMERG timestamps, units and missing values",
            "Rainfall temporal completeness",
            "DEM valid-data coverage",
            "GN boundary and feature-grid alignment",
            "Supervised ML label feasibility",
            "Independent susceptibility model evaluation",
        ],
        "conclusion": (
            "This audit establishes source availability "
            "and structural metadata only. It does not "
            "validate waterlogging labels or model accuracy."
        ),
    }

    json_path = (
        REPORT_DIR / "script01_part4_input_audit.json"
    )

    event_csv = (
        METRIC_DIR / "script01_event_kml_inventory.csv"
    )

    rainfall_csv = (
        METRIC_DIR / "script01_imerg_file_inventory.csv"
    )

    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    if event_records:
        with event_csv.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "event_id",
                    "event_date",
                    "file",
                    "role",
                    "size_bytes",
                    "placemarks",
                    "polygon_placemarks",
                    "line_placemarks",
                    "point_placemarks",
                    "multigeometry_placemarks",
                    "parse_errors",
                ],
            )
            writer.writeheader()
            writer.writerows(event_records)

    if rainfall_records:
        with rainfall_csv.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "group",
                    "file",
                    "extension",
                    "size_bytes",
                ],
            )
            writer.writeheader()
            writer.writerows(rainfall_records)

    # ------------------------------------------------------
    # 7. Final status
    # ------------------------------------------------------

    print("\n" + "=" * 68)
    print("SCRIPT 01 - INITIAL AUDIT SUMMARY")
    print("=" * 68)

    print("Event KML files inspected:", len(event_records))
    print("IMERG files inventoried:", len(rainfall_records))
    print("DEM datasets inspected:", len(dem_records))
    print("OSM GeoPackages inspected:", len(gpkg_records))
    print("Issues identified:", len(issues))

    for issue in issues:
        print("REVIEW:", issue)

    print("\nSaved:", json_path)
    print("Saved:", event_csv)
    print("Saved:", rainfall_csv)

    print("\nSOURCE INVENTORY:",
          "REVIEW REQUIRED" if issues else "PASS")

    print("SCIENTIFIC EVENT VALIDATION: PENDING")
    print("MODEL FEASIBILITY: NOT YET ASSESSED")
    print("RAW DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
