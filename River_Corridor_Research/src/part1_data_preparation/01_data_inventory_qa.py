
"""
J26-DS-338 - River Corridor Research

STEP 09: DATA INVENTORY AND INITIAL QUALITY ASSESSMENT

Purpose:
    Inspect collected satellite and GIS datasets
    without modifying original research data.

Inputs:
    data/raw/landsat/
    data/raw/sentinel2/
    data/raw/dem/
    data/raw/jrc/
    data/raw/osm/
    data/raw/boundaries/

Outputs:
    outputs/reports/01_data_inventory.csv
    outputs/reports/01_data_inventory_qa.json
"""

from pathlib import Path
from datetime import datetime
import csv
import json

import rasterio
import geopandas as gpd


# ==================================================
# 1. PROJECT PATH CONFIGURATION
# ==================================================

# Script is inside src/part1_data_preparation/
# parents[2] points to River_Corridor_Research/

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. EXPECTED DATASETS
# ==================================================

EXPECTED_FILES = {
    "landsat": 18,
    "sentinel2": 16,
    "dem": 1,
    "jrc": 1,
    "osm": 2,
    "boundaries": 2
}

RASTER_EXTENSIONS = {".tif", ".tiff"}

VECTOR_EXTENSIONS = {
    ".geojson",
    ".gpkg",
    ".shp",
    ".kml"
}

SUPPORTED_EXTENSIONS = (
    RASTER_EXTENSIONS | VECTOR_EXTENSIONS
)

records = []
issues = []
dataset_counts = {}


# ==================================================
# 3. START DATA INVENTORY
# ==================================================

print("=" * 75)
print("J26-DS-338 | RIVER CORRIDOR RESEARCH")
print("STEP 09 - DATA INVENTORY AND INITIAL QA")
print("=" * 75)

print(f"\nProject Root: {PROJECT_ROOT}")
print(f"Raw Data Directory: {RAW_DIR}")


# ==================================================
# 4. INSPECT EACH DATASET
# ==================================================

for dataset, expected_count in EXPECTED_FILES.items():

    folder = RAW_DIR / dataset

    print("\n" + "-" * 75)
    print(f"DATASET: {dataset.upper()}")
    print("-" * 75)

    if not folder.exists():
        issues.append(
            f"Missing dataset folder: {folder}"
        )
        files = []
    else:
        files = sorted(
            [
                file
                for file in folder.rglob("*")
                if file.is_file()
                and file.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ]
        )

    actual_count = len(files)
    dataset_counts[dataset] = actual_count

    print(f"Expected files: {expected_count}")
    print(f"Actual files  : {actual_count}")

    # Boundaries can contain additional KMC GIS files.
    if dataset == "boundaries":
        count_valid = actual_count >= expected_count
    else:
        count_valid = actual_count == expected_count

    if not count_valid:
        warning = (
            f"{dataset}: expected "
            f"{expected_count}, found {actual_count}"
        )
        issues.append(warning)
        print(f"WARNING: {warning}")

    # ----------------------------------------------
    # INSPECT INDIVIDUAL FILES
    # ----------------------------------------------

    for file in files:

        record = {
            "dataset": dataset,
            "filename": file.name,
            "relative_path": file.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "file_size_mb": round(
                file.stat().st_size / (1024 * 1024),
                3
            ),
            "file_type": "",
            "status": "OK",
            "crs": "",
            "band_count": "",
            "band_names": "",
            "width_pixels": "",
            "height_pixels": "",
            "pixel_resolution": "",
            "feature_count": "",
            "geometry_types": "",
            "missing_geometries": "",
            "invalid_geometries": "",
            "notes": ""
        }

        try:

            # ======================================
            # RASTER FILE INSPECTION
            # ======================================

            if file.suffix.lower() in RASTER_EXTENSIONS:

                with rasterio.open(file) as src:

                    record["file_type"] = "Raster"
                    record["crs"] = str(src.crs)
                    record["band_count"] = src.count

                    record["band_names"] = str(
                        src.descriptions
                    )

                    record["width_pixels"] = src.width
                    record["height_pixels"] = src.height

                    record["pixel_resolution"] = str(
                        src.res
                    )

                    record["notes"] = (
                        f"nodata={src.nodata}; "
                        f"dtype={src.dtypes}"
                    )

                    if src.crs is None:
                        warning = (
                            f"Missing CRS: {file.name}"
                        )
                        issues.append(warning)
                        record["status"] = "WARNING"

                    if src.count < 1:
                        warning = (
                            f"No raster bands: {file.name}"
                        )
                        issues.append(warning)
                        record["status"] = "WARNING"

                    print(
                        f"OK | {file.name} | "
                        f"Bands: {src.count} | "
                        f"Resolution: {src.res}"
                    )

            # ======================================
            # VECTOR GIS FILE INSPECTION
            # ======================================

            elif file.suffix.lower() in VECTOR_EXTENSIONS:

                gdf = gpd.read_file(file)

                record["file_type"] = "Vector"
                record["crs"] = str(gdf.crs)
                record["feature_count"] = len(gdf)

                record["geometry_types"] = str(
                    gdf.geom_type.value_counts().to_dict()
                )

                missing_geometry_count = int(
                    gdf.geometry.isna().sum()
                )

                invalid_geometry_count = int(
                    (~gdf.geometry.is_valid).sum()
                )

                record["missing_geometries"] = (
                    missing_geometry_count
                )

                record["invalid_geometries"] = (
                    invalid_geometry_count
                )

                if gdf.crs is None:
                    issues.append(
                        f"Missing CRS: {file.name}"
                    )
                    record["status"] = "WARNING"

                if gdf.empty:
                    issues.append(
                        f"Empty vector dataset: {file.name}"
                    )
                    record["status"] = "WARNING"

                if invalid_geometry_count > 0:
                    issues.append(
                        f"Invalid geometries: "
                        f"{file.name} "
                        f"({invalid_geometry_count})"
                    )
                    record["status"] = "WARNING"

                print(
                    f"OK | {file.name} | "
                    f"Features: {len(gdf)} | "
                    f"CRS: {gdf.crs}"
                )

        # ==========================================
        # HANDLE UNREADABLE FILES
        # ==========================================

        except Exception as error:

            record["status"] = "ERROR"
            record["notes"] = str(error)

            issues.append(
                f"Unreadable file: {file.name}: {error}"
            )

            print(f"ERROR | {file.name}")
            print(f"Reason: {error}")

        records.append(record)


# ==================================================
# 5. EXPORT DETAILED INVENTORY CSV
# ==================================================

print("\n" + "=" * 75)
print("SAVING DATA INVENTORY REPORTS")
print("=" * 75)

csv_path = REPORT_DIR / "01_data_inventory.csv"

fieldnames = [
    "dataset",
    "filename",
    "relative_path",
    "file_size_mb",
    "file_type",
    "status",
    "crs",
    "band_count",
    "band_names",
    "width_pixels",
    "height_pixels",
    "pixel_resolution",
    "feature_count",
    "geometry_types",
    "missing_geometries",
    "invalid_geometries",
    "notes"
]

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(records)


# ==================================================
# 6. GENERATE QA SUMMARY
# ==================================================

total_files = len(records)

readable_files = sum(
    record["status"] != "ERROR"
    for record in records
)

error_files = sum(
    record["status"] == "ERROR"
    for record in records
)

warning_files = sum(
    record["status"] == "WARNING"
    for record in records
)

summary = {
    "project": "J26-DS-338",
    "component": "River Corridor Research",
    "step": "09 - Data Inventory and Initial QA",
    "generated_at": datetime.now().isoformat(),
    "expected_counts": EXPECTED_FILES,
    "actual_counts": dataset_counts,
    "total_files": total_files,
    "readable_files": readable_files,
    "error_files": error_files,
    "warning_files": warning_files,
    "total_issues": len(issues),
    "issues": issues,
    "qa_scope": (
        "File readability, metadata, CRS, "
        "dimensions and vector geometry checks. "
        "Pixel-level quality and river-water "
        "accuracy are not yet validated."
    )
}

json_path = (
    REPORT_DIR / "01_data_inventory_qa.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as json_file:

    json.dump(
        summary,
        json_file,
        indent=4,
        ensure_ascii=False
    )


# ==================================================
# 7. PRINT FINAL SUMMARY
# ==================================================

print("\n" + "=" * 75)
print("STEP 09 - FINAL DATA INVENTORY SUMMARY")
print("=" * 75)

for dataset, count in dataset_counts.items():
    print(
        f"{dataset.upper():15s}: "
        f"{count} files"
    )

print("-" * 75)

print(f"Total files     : {total_files}")
print(f"Readable files  : {readable_files}")
print(f"Error files     : {error_files}")
print(f"Warning files   : {warning_files}")
print(f"Total QA issues : {len(issues)}")

if issues:
    print("\nQA ISSUES:")
    for issue in issues:
        print(f" - {issue}")
else:
    print("\nNo initial inventory issues detected.")

print("\nOUTPUT FILES:")
print(csv_path)
print(json_path)

print("\nSTEP 09 FINISHED")
print("=" * 75)
