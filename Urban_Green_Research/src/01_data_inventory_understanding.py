"""
STEP 01 - DATA INVENTORY AND UNDERSTANDING

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Validate that all documented research datasets are available.
2. Inspect raster metadata.
3. Inspect vector layers and metadata.
4. Verify Landsat annual coverage from 2010 to 2026.
5. Create reproducible data-understanding reports.

IMPORTANT:
This script is READ-ONLY.
It does not modify any raw research dataset.
"""

from pathlib import Path
import re

import pandas as pd
import rasterio
import pyogrio


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

INVENTORY_FILE = DATA_DIR / "data_inventory.csv"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def relative_path(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def inspect_raster(path: Path, dataset_name: str) -> dict:
    """
    Read raster metadata without modifying the raster.
    """

    record = {
        "dataset_name": dataset_name,
        "path": relative_path(path),
        "status": "UNKNOWN",
        "crs": None,
        "bands": None,
        "width": None,
        "height": None,
        "resolution_x": None,
        "resolution_y": None,
        "dtype": None,
        "nodata": None,
        "bounds_left": None,
        "bounds_bottom": None,
        "bounds_right": None,
        "bounds_top": None,
    }

    if not path.exists():
        record["status"] = "MISSING"
        return record

    try:
        with rasterio.open(path) as src:
            record.update(
                {
                    "status": "OK",
                    "crs": str(src.crs),
                    "bands": src.count,
                    "width": src.width,
                    "height": src.height,
                    "resolution_x": src.res[0],
                    "resolution_y": src.res[1],
                    "dtype": str(src.dtypes[0]) if src.count > 0 else None,
                    "nodata": src.nodata,
                    "bounds_left": src.bounds.left,
                    "bounds_bottom": src.bounds.bottom,
                    "bounds_right": src.bounds.right,
                    "bounds_top": src.bounds.top,
                }
            )

    except Exception as error:
        record["status"] = f"ERROR: {error}"

    return record


def inspect_vector(path: Path, dataset_name: str) -> list[dict]:
    """
    Inspect vector layers without modifying the source file.
    Supports Shapefile, GeoPackage, GeoJSON and KML where GDAL permits.
    """

    records = []

    if not path.exists():
        records.append(
            {
                "dataset_name": dataset_name,
                "path": relative_path(path),
                "layer_name": None,
                "geometry_type": None,
                "feature_count": None,
                "field_count": None,
                "crs": None,
                "status": "MISSING",
            }
        )
        return records

    try:
        layers = pyogrio.list_layers(path)

        for layer_info in layers:
            layer_name = str(layer_info[0])

            try:
                info = pyogrio.read_info(path, layer=layer_name)

                fields = info.get("fields", [])

                records.append(
                    {
                        "dataset_name": dataset_name,
                        "path": relative_path(path),
                        "layer_name": layer_name,
                        "geometry_type": info.get("geometry_type"),
                        "feature_count": info.get("features"),
                        "field_count": len(fields),
                        "crs": str(info.get("crs")),
                        "status": "OK",
                    }
                )

            except Exception as layer_error:
                records.append(
                    {
                        "dataset_name": dataset_name,
                        "path": relative_path(path),
                        "layer_name": layer_name,
                        "geometry_type": None,
                        "feature_count": None,
                        "field_count": None,
                        "crs": None,
                        "status": f"ERROR: {layer_error}",
                    }
                )

    except Exception as error:
        records.append(
            {
                "dataset_name": dataset_name,
                "path": relative_path(path),
                "layer_name": None,
                "geometry_type": None,
                "feature_count": None,
                "field_count": None,
                "crs": None,
                "status": f"ERROR: {error}",
            }
        )

    return records


# =============================================================================
# 3. READ DATA INVENTORY
# =============================================================================

print("=" * 100)
print("STEP 01 - DATA INVENTORY AND UNDERSTANDING")
print("=" * 100)

if not INVENTORY_FILE.exists():
    raise FileNotFoundError(
        f"Data inventory file was not found:\n{INVENTORY_FILE}"
    )

inventory_df = pd.read_csv(INVENTORY_FILE)

print(f"\nInventory file loaded: {relative_path(INVENTORY_FILE)}")
print(f"Inventory records: {len(inventory_df)}")


# =============================================================================
# 4. CHECK INVENTORY PATHS
# =============================================================================

inventory_records = []

for _, row in inventory_df.iterrows():

    dataset_id = row["dataset_id"]
    dataset_name = row["dataset_name"]
    documented_path = str(row["local_path"])

    full_path = PROJECT_ROOT / documented_path

    exists = full_path.exists()

    if exists and full_path.is_file():
        object_type = "FILE"
        size_mb = full_path.stat().st_size / (1024 ** 2)
        contained_files = 1

    elif exists and full_path.is_dir():
        object_type = "DIRECTORY"
        contained_files = sum(
            1 for item in full_path.rglob("*") if item.is_file()
        )
        size_mb = None

    else:
        object_type = "MISSING"
        contained_files = 0
        size_mb = None

    inventory_records.append(
        {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "documented_path": documented_path,
            "exists": exists,
            "object_type": object_type,
            "contained_files": contained_files,
            "file_size_mb": size_mb,
        }
    )

inventory_status_df = pd.DataFrame(inventory_records)


# =============================================================================
# 5. LANDSAT 2010-2026 CHECK
# =============================================================================

print("\n" + "-" * 100)
print("LANDSAT ANNUAL COMPOSITE CHECK")
print("-" * 100)

LANDSAT_DIR = RAW_DIR / "landsat" / "annual_composites"

landsat_files = sorted(
    LANDSAT_DIR.glob("Kaduwela_Landsat_*.tif")
)

expected_years = list(range(2010, 2027))
found_years = []

for file_path in landsat_files:
    match = re.search(r"(20\d{2})", file_path.stem)

    if match:
        found_years.append(int(match.group(1)))

missing_years = sorted(set(expected_years) - set(found_years))
unexpected_years = sorted(set(found_years) - set(expected_years))

print(f"Expected years : {expected_years[0]}-{expected_years[-1]}")
print(f"Expected files : {len(expected_years)}")
print(f"Found files    : {len(landsat_files)}")
print(f"Found years    : {found_years}")
print(f"Missing years  : {missing_years if missing_years else 'None'}")
print(
    f"Unexpected years: "
    f"{unexpected_years if unexpected_years else 'None'}"
)


# =============================================================================
# 6. RASTER DATA UNDERSTANDING
# =============================================================================

print("\n" + "-" * 100)
print("RASTER DATA INSPECTION")
print("-" * 100)

raster_records = []

# Landsat annual composites
for file_path in landsat_files:

    match = re.search(r"(20\d{2})", file_path.stem)
    year = match.group(1) if match else "Unknown"

    raster_records.append(
        inspect_raster(
            file_path,
            f"Landsat Annual Composite {year}"
        )
    )


# GEDTM30
raster_records.append(
    inspect_raster(
        RAW_DIR
        / "dem"
        / "gedtm30"
        / "GEDTM30_Kaduwela_Buffer.tif",
        "GEDTM30 DEM",
    )
)


# NASADEM
raster_records.append(
    inspect_raster(
        RAW_DIR
        / "dem"
        / "nasadem"
        / "NASADEM_Kaduwela_Buffer.tif",
        "NASADEM",
    )
)


# ESA WorldCover
raster_records.append(
    inspect_raster(
        RAW_DIR
        / "reference_lulc"
        / "esa_worldcover_2021"
        / "Kaduwela_ESA_WorldCover_2021.tif",
        "ESA WorldCover 2021",
    )
)

raster_df = pd.DataFrame(raster_records)

print(raster_df[
    [
        "dataset_name",
        "status",
        "crs",
        "bands",
        "width",
        "height",
        "resolution_x",
        "resolution_y",
    ]
].to_string(index=False))


# =============================================================================
# 7. VECTOR DATA UNDERSTANDING
# =============================================================================

print("\n" + "-" * 100)
print("VECTOR DATA INSPECTION")
print("-" * 100)

vector_records = []


# KMC boundary
vector_records.extend(
    inspect_vector(
        RAW_DIR
        / "boundaries"
        / "kmc"
        / "kmc boundry.kml",
        "KMC Boundary",
    )
)


# Original GN division source
vector_records.extend(
    inspect_vector(
        RAW_DIR
        / "boundaries"
        / "gn_divisions"
        / "Kaduwela TO Area.kml",
        "Kaduwela GN Division Source",
    )
)


# Processed 57-GN working dataset
vector_records.extend(
    inspect_vector(
        PROCESSED_DIR
        / "boundaries"
        / "gn_divisions"
        / "kaduwela_gn_divisions.shp",
        "Kaduwela GN Divisions Processed",
    )
)


# OSM roads/buildings GeoPackage
vector_records.extend(
    inspect_vector(
        RAW_DIR
        / "osm"
        / "roads_buildings"
        / "Kaduwela_OSM_Roads_Buildings_2026.gpkg",
        "HOT OSM Roads and Buildings",
    )
)


# =============================================================================
# 8. UDA PROVISIONAL ZONING PACKAGE
# =============================================================================

print("\n" + "-" * 100)
print("UDA PROVISIONAL ZONING PACKAGE")
print("-" * 100)

UDA_ZONING_DIR = RAW_DIR / "uda" / "zoning_provisional"

uda_vector_files = []

if UDA_ZONING_DIR.exists():

    uda_vector_files.extend(
        UDA_ZONING_DIR.rglob("*.gpkg")
    )

    uda_vector_files.extend(
        UDA_ZONING_DIR.rglob("*.geojson")
    )

uda_vector_files = sorted(set(uda_vector_files))

print(
    f"UDA vector files discovered: "
    f"{len(uda_vector_files)}"
)

for file_path in uda_vector_files:

    print(f"Inspecting: {file_path.name}")

    vector_records.extend(
        inspect_vector(
            file_path,
            f"UDA Provisional - {file_path.stem}",
        )
    )


vector_df = pd.DataFrame(vector_records)

if not vector_df.empty:
    print(
        vector_df[
            [
                "dataset_name",
                "layer_name",
                "geometry_type",
                "feature_count",
                "field_count",
                "crs",
                "status",
            ]
        ].to_string(index=False)
    )


# =============================================================================
# 9. SAVE REPORTS
# =============================================================================

inventory_report = (
    REPORT_DIR
    / "01_data_inventory_validation.csv"
)

raster_report = (
    REPORT_DIR
    / "01_raster_inventory.csv"
)

vector_report = (
    REPORT_DIR
    / "01_vector_layer_inventory.csv"
)

inventory_status_df.to_csv(
    inventory_report,
    index=False
)

raster_df.to_csv(
    raster_report,
    index=False
)

vector_df.to_csv(
    vector_report,
    index=False
)


# =============================================================================
# 10. FINAL SUMMARY
# =============================================================================

missing_inventory = inventory_status_df[
    inventory_status_df["exists"] == False
]

raster_errors = raster_df[
    raster_df["status"] != "OK"
]

vector_errors = vector_df[
    vector_df["status"] != "OK"
] if not vector_df.empty else pd.DataFrame()


print("\n" + "=" * 100)
print("STEP 01 SUMMARY")
print("=" * 100)

print(
    f"Inventory datasets documented : "
    f"{len(inventory_status_df)}"
)

print(
    f"Missing inventory datasets    : "
    f"{len(missing_inventory)}"
)

print(
    f"Landsat annual files          : "
    f"{len(landsat_files)} / 17"
)

print(
    f"Missing Landsat years         : "
    f"{missing_years if missing_years else 'None'}"
)

print(
    f"Raster inspection errors      : "
    f"{len(raster_errors)}"
)

print(
    f"Vector inspection errors      : "
    f"{len(vector_errors)}"
)

print("\nReports created:")

print(
    f" - {relative_path(inventory_report)}"
)

print(
    f" - {relative_path(raster_report)}"
)

print(
    f" - {relative_path(vector_report)}"
)

print("\nIMPORTANT:")
print("No raw datasets were modified.")

print("=" * 100)