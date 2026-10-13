from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import rasterio
from rasterio.warp import transform_bounds


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

SENTINEL_DIR = RAW_DIR / "sentinel2"
DEM_DIR = RAW_DIR / "dem"
OSM_DIR = RAW_DIR / "osm"
LULC_DIR = RAW_DIR / "lulc_reference"
BOUNDARY_DIR = RAW_DIR / "boundaries"

PART1_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part1_historical_change"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "part2_candidate_wetland"
)

INTERIM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# EXPECTED INPUT FILES
# ============================================================

S2_DRY = (
    SENTINEL_DIR
    / "Sentinel2_2025_KMC_Dry_9Band.tif"
)

S2_WET = (
    SENTINEL_DIR
    / "Sentinel2_2025_KMC_Wet_9Band.tif"
)

GEDTM = (
    DEM_DIR
    / "GEDTM30_Kaduwela_Buffer.tif"
)

NASADEM = (
    DEM_DIR
    / "NASADEM_Kaduwela_Buffer.tif"
)

OSM_ROADS_BUILDINGS = (
    OSM_DIR
    / "Kaduwela_OSM_Roads_Buildings_2026.gpkg"
)

OSM_WATER_DRAINAGE = (
    OSM_DIR
    / "Kaduwela_OSM_Water_Drainage_2026.gpkg"
)

OSM_CLIP = (
    OSM_DIR
    / "clipping_boundary.geojson"
)

LULC_2025 = (
    LULC_DIR
    / "LULC_Reference_Samples_2025.kmz"
)

BOUNDARY_KML = (
    BOUNDARY_DIR
    / "Kaduwela TO Area.kml"
)

KNOWN_WETLANDS = (
    PART1_DIR
    / "wetland_latest_status_gn_wgs84.geojson"
)

AUDIT_OUTPUT = (
    INTERIM_DIR
    / "part2_input_audit.csv"
)


# ============================================================
# HELPERS
# ============================================================

audit_rows = []


def add_check(
    category,
    item,
    status,
    details,
):
    audit_rows.append(
        {
            "category": category,
            "item": item,
            "status": status,
            "details": details,
        }
    )


def inspect_raster(
    path,
    label,
):
    if not path.exists():

        add_check(
            "Raster",
            label,
            "FAIL",
            "File not found",
        )

        print(
            f"\n{label}: FILE NOT FOUND"
        )

        return None

    with rasterio.open(
        path
    ) as src:

        information = {
            "path": path,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "transform": src.transform,
            "bounds": src.bounds,
            "resolution": src.res,
            "dtype": src.dtypes,
            "nodata": src.nodata,
            "descriptions": src.descriptions,
        }

        print(
            f"\n--- {label} ---"
        )

        print(
            f"CRS: {src.crs}"
        )

        print(
            f"Size: "
            f"{src.width} x {src.height}"
        )

        print(
            f"Bands: {src.count}"
        )

        print(
            f"Resolution: {src.res}"
        )

        print(
            f"NoData: {src.nodata}"
        )

        print(
            f"Bounds: {src.bounds}"
        )

        print(
            "Band descriptions:"
        )

        for index, description in enumerate(
            src.descriptions,
            start=1,
        ):
            print(
                f"  Band {index}: "
                f"{description}"
            )

        add_check(
            "Raster",
            label,
            "PASS",
            (
                f"{src.count} bands; "
                f"{src.crs}; "
                f"{src.width}x{src.height}; "
                f"resolution={src.res}"
            ),
        )

        return information


def inspect_geopackage(
    path,
    label,
):
    if not path.exists():

        add_check(
            "Vector",
            label,
            "FAIL",
            "File not found",
        )

        print(
            f"\n{label}: FILE NOT FOUND"
        )

        return

    print(
        f"\n--- {label} ---"
    )

    layers = pyogrio.list_layers(
        path
    )

    print(
        f"Layers: {len(layers)}"
    )

    for layer in layers:

        layer_name = layer[0]

        try:

            info = pyogrio.read_info(
                path,
                layer=layer_name,
            )

            feature_count = info.get(
                "features"
            )

            geometry_type = info.get(
                "geometry_type"
            )

            crs = info.get(
                "crs"
            )

            print(
                f"  {layer_name}"
            )

            print(
                f"    Features: "
                f"{feature_count}"
            )

            print(
                f"    Geometry: "
                f"{geometry_type}"
            )

            print(
                f"    CRS: {crs}"
            )

        except Exception as error:

            print(
                f"    Could not inspect: "
                f"{error}"
            )

    add_check(
        "Vector",
        label,
        "PASS",
        f"{len(layers)} layer(s) available",
    )


def inspect_geojson(
    path,
    label,
):
    if not path.exists():

        add_check(
            "Vector",
            label,
            "FAIL",
            "File not found",
        )

        print(
            f"\n{label}: FILE NOT FOUND"
        )

        return

    print(
        f"\n--- {label} ---"
    )

    try:

        data = gpd.read_file(
            path
        )

        print(
            f"Features: {len(data)}"
        )

        print(
            f"CRS: {data.crs}"
        )

        print(
            "Geometry types:"
        )

        print(
            data.geometry
            .geom_type
            .value_counts()
            .to_string()
        )

        print(
            f"Bounds: "
            f"{data.total_bounds}"
        )

        add_check(
            "Vector",
            label,
            "PASS",
            (
                f"{len(data)} features; "
                f"{data.crs}"
            ),
        )

    except Exception as error:

        print(
            f"ERROR: {error}"
        )

        add_check(
            "Vector",
            label,
            "FAIL",
            str(error),
        )


def inspect_kml_layers(
    path,
):
    print(
        "\n--- KMC / GN Boundary KML ---"
    )

    if not path.exists():

        add_check(
            "Boundary",
            "Kaduwela TO Area.kml",
            "FAIL",
            "File not found",
        )

        return

    layers = pyogrio.list_layers(
        path
    )

    for layer in layers:

        print(
            f"  {layer[0]} "
            f"[{layer[1]}]"
        )

    add_check(
        "Boundary",
        "Kaduwela TO Area.kml",
        "PASS",
        f"{len(layers)} layers",
    )


def inspect_lulc_kmz(
    kmz_path,
):
    print(
        "\n--- 2025 LULC Reference Samples ---"
    )

    if not kmz_path.exists():

        add_check(
            "Reference",
            "LULC 2025",
            "FAIL",
            "File not found",
        )

        return

    with TemporaryDirectory() as temp_dir:

        with zipfile.ZipFile(
            kmz_path,
            "r",
        ) as archive:

            archive.extractall(
                temp_dir
            )

        kml_files = list(
            Path(temp_dir).rglob(
                "*.kml"
            )
        )

        if not kml_files:

            print(
                "No KML found inside KMZ."
            )

            add_check(
                "Reference",
                "LULC 2025",
                "FAIL",
                "KMZ contains no KML",
            )

            return

        kml_file = kml_files[0]

        tree = ET.parse(
            kml_file
        )

        root = tree.getroot()

        ns = {
            "kml":
                "http://www.opengis.net/kml/2.2"
        }

        placemarks = root.findall(
            ".//kml:Placemark",
            ns,
        )

        point_count = 0

        for placemark in placemarks:

            point = placemark.find(
                ".//kml:Point",
                ns,
            )

            if point is not None:
                point_count += 1

        print(
            f"Total placemarks: "
            f"{len(placemarks)}"
        )

        print(
            f"Point samples: "
            f"{point_count}"
        )

        print(
            "\nFolder / class-like counts:"
        )

        folder_results = []

        for folder in root.findall(
            ".//kml:Folder",
            ns,
        ):

            name_element = folder.find(
                "kml:name",
                ns,
            )

            if name_element is None:
                continue

            folder_name = (
                name_element.text
                if name_element.text
                else "Unnamed"
            )

            direct_points = 0

            for placemark in folder.findall(
                "kml:Placemark",
                ns,
            ):

                if (
                    placemark.find(
                        ".//kml:Point",
                        ns,
                    )
                    is not None
                ):
                    direct_points += 1

            if direct_points > 0:

                folder_results.append(
                    (
                        folder_name,
                        direct_points,
                    )
                )

        for folder_name, count in folder_results:

            print(
                f"  {folder_name}: "
                f"{count}"
            )

        expected_count = 375

        status = (
            "PASS"
            if point_count == expected_count
            else "WARNING"
        )

        add_check(
            "Reference",
            "LULC 2025",
            status,
            (
                f"{point_count} point samples; "
                f"expected approximately "
                f"{expected_count}"
            ),
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 INPUT VALIDATION"
    )

    print(
        "========================================"
    )

    # ========================================================
    # 1. SENTINEL-2
    # ========================================================

    dry = inspect_raster(
        S2_DRY,
        "Sentinel-2 Dry 2025",
    )

    wet = inspect_raster(
        S2_WET,
        "Sentinel-2 Wet 2025",
    )

    if (
        dry is not None
        and wet is not None
    ):

        same_crs = (
            dry["crs"]
            == wet["crs"]
        )

        same_dimensions = (
            dry["width"]
            == wet["width"]
            and
            dry["height"]
            == wet["height"]
        )

        same_transform = np.allclose(
            tuple(
                dry["transform"]
            ),
            tuple(
                wet["transform"]
            ),
        )

        same_band_count = (
            dry["count"]
            == wet["count"]
            == 9
        )

        print(
            "\nSentinel wet/dry alignment:"
        )

        print(
            f"Same CRS: "
            f"{same_crs}"
        )

        print(
            f"Same dimensions: "
            f"{same_dimensions}"
        )

        print(
            f"Same transform: "
            f"{same_transform}"
        )

        print(
            f"Both have 9 bands: "
            f"{same_band_count}"
        )

        add_check(
            "Alignment",
            "Sentinel wet/dry",
            (
                "PASS"
                if (
                    same_crs
                    and
                    same_dimensions
                    and
                    same_transform
                    and
                    same_band_count
                )
                else "FAIL"
            ),
            (
                f"same_crs={same_crs}; "
                f"same_dimensions={same_dimensions}; "
                f"same_transform={same_transform}; "
                f"9bands={same_band_count}"
            ),
        )

    # ========================================================
    # 2. DEM
    # ========================================================

    gedtm = inspect_raster(
        GEDTM,
        "GEDTM30 Primary DEM",
    )

    nasadem = inspect_raster(
        NASADEM,
        "NASADEM Backup DEM",
    )

    # --------------------------------------------------------
    # CHECK PRIMARY DEM COVERS SENTINEL STUDY AREA
    # --------------------------------------------------------

    if (
        dry is not None
        and gedtm is not None
    ):

        dem_bounds_in_s2 = transform_bounds(
            gedtm["crs"],
            dry["crs"],
            *gedtm["bounds"],
            densify_pts=21,
        )

        s2_bounds = dry[
            "bounds"
        ]

        covers = (
            dem_bounds_in_s2[0]
            <= s2_bounds.left
            and
            dem_bounds_in_s2[1]
            <= s2_bounds.bottom
            and
            dem_bounds_in_s2[2]
            >= s2_bounds.right
            and
            dem_bounds_in_s2[3]
            >= s2_bounds.top
        )

        print(
            "\nPrimary DEM coverage:"
        )

        print(
            f"GEDTM covers complete "
            f"Sentinel extent: "
            f"{covers}"
        )

        add_check(
            "Coverage",
            "GEDTM vs Sentinel",
            (
                "PASS"
                if covers
                else "WARNING"
            ),
            (
                "Complete Sentinel coverage"
                if covers
                else
                "DEM does not fully cover "
                "Sentinel raster extent"
            ),
        )

    # ========================================================
    # 3. OSM
    # ========================================================

    inspect_geopackage(
        OSM_ROADS_BUILDINGS,
        "OSM Roads / Buildings",
    )

    inspect_geopackage(
        OSM_WATER_DRAINAGE,
        "OSM Water / Drainage",
    )

    inspect_geojson(
        OSM_CLIP,
        "OSM Clipping Boundary",
    )

    # ========================================================
    # 4. LULC REFERENCE
    # ========================================================

    inspect_lulc_kmz(
        LULC_2025
    )

    # ========================================================
    # 5. KMC / GN BOUNDARIES
    # ========================================================

    inspect_kml_layers(
        BOUNDARY_KML
    )

    # ========================================================
    # 6. KNOWN WETLANDS FROM PART 1
    # ========================================================

    print(
        "\n--- Known Wetlands from Part 1 ---"
    )

    if KNOWN_WETLANDS.exists():

        wetlands = gpd.read_file(
            KNOWN_WETLANDS
        )

        unique_wetlands = (
            wetlands[
                "wetland_id"
            ]
            .nunique()
        )

        print(
            f"Features: "
            f"{len(wetlands)}"
        )

        print(
            f"Unique wetland IDs: "
            f"{unique_wetlands}"
        )

        print(
            f"CRS: "
            f"{wetlands.crs}"
        )

        print(
            "Geometry types:"
        )

        print(
            wetlands.geometry
            .geom_type
            .value_counts()
            .to_string()
        )

        add_check(
            "Training reference",
            "Known wetlands",
            (
                "PASS"
                if unique_wetlands == 25
                else "WARNING"
            ),
            (
                f"{unique_wetlands} "
                f"unique wetlands"
            ),
        )

    else:

        print(
            "Known wetland file not found."
        )

        add_check(
            "Training reference",
            "Known wetlands",
            "FAIL",
            "Part 1 latest wetland layer missing",
        )

    # ========================================================
    # SAVE AUDIT
    # ========================================================

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    print(
        "\n========================================"
    )

    print(
        "AUDIT SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        audit.to_string(
            index=False
        )
    )

    failed = (
        audit[
            "status"
        ]
        == "FAIL"
    ).sum()

    warnings = (
        audit[
            "status"
        ]
        == "WARNING"
    ).sum()

    print(
        f"\nFailed checks: "
        f"{failed}"
    )

    print(
        f"Warnings: "
        f"{warnings}"
    )

    print(
        f"\nAudit saved to:\n"
        f"{AUDIT_OUTPUT}"
    )

    if failed == 0:

        print(
            "\nPART 2 INPUT STATUS: "
            "READY FOR PREPROCESSING"
        )

    else:

        print(
            "\nPART 2 INPUT STATUS: "
            "FIX REQUIRED BEFORE PREPROCESSING"
        )


if __name__ == "__main__":
    main()