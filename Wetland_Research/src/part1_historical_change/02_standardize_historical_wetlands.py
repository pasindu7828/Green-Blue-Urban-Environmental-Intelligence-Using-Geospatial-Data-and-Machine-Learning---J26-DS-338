from pathlib import Path
import re
import xml.etree.ElementTree as ET
from datetime import date

import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon
from shapely import make_valid


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "historical_wetlands"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part1_historical_change"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "part1_historical_change"
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CRS
# ============================================================

# Google Earth / KML coordinates
WGS84 = "EPSG:4326"

# Metric CRS used for Kaduwela area calculations
UTM44N = "EPSG:32644"


# ============================================================
# KML SETTINGS
# ============================================================

KML_NS = {
    "k": "http://www.opengis.net/kml/2.2"
}

ID_RE = re.compile(
    r"W\d{3}",
    re.IGNORECASE
)

DATE_RE = re.compile(
    r"(?P<year>\d{4})_(?P<month>\d{1,2})_(?P<day>\d{1,2})"
)


# ============================================================
# SEASON CLASSIFICATION
# ============================================================

def season_from_month(month: int) -> str:

    if month in (12, 1, 2):
        return "Northeast monsoon"

    if month in (3, 4):
        return "First inter-monsoon"

    if month in (5, 6, 7, 8, 9):
        return "Southwest monsoon"

    return "Second inter-monsoon"


# ============================================================
# PARSE DATE / PATCH LABEL
# ============================================================

def parse_date_from_name(name: str):

    match = DATE_RE.search(name)

    if not match:
        return None, None

    try:
        observation_date = date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )

    except ValueError:
        return None, None

    remainder = name[match.end():].strip("_- ")

    patch_label = remainder if remainder else None

    return observation_date, patch_label


# ============================================================
# READ KML COORDINATES
# ============================================================

def parse_coordinate_text(text):

    """
    Convert KML coordinate text into [(x, y), ...].

    KML normally stores:
        longitude,latitude,altitude
    """

    if not text:
        return []

    coordinates = []

    for token in text.strip().split():

        parts = token.split(",")

        if len(parts) < 2:
            continue

        try:
            longitude = float(parts[0])
            latitude = float(parts[1])

            coordinates.append(
                (longitude, latitude)
            )

        except ValueError:
            continue

    return coordinates


# ============================================================
# EXTRACT POLYGON FROM KML
# ============================================================

def polygon_from_kml_element(polygon_element):

    outer_element = polygon_element.find(
        ".//k:outerBoundaryIs/"
        "k:LinearRing/"
        "k:coordinates",
        KML_NS,
    )

    if outer_element is None:
        return None

    outer_coords = parse_coordinate_text(
        outer_element.text
    )

    if len(outer_coords) < 4:
        return None

    holes = []

    inner_elements = polygon_element.findall(
        ".//k:innerBoundaryIs/"
        "k:LinearRing/"
        "k:coordinates",
        KML_NS,
    )

    for inner in inner_elements:

        inner_coords = parse_coordinate_text(
            inner.text
        )

        if len(inner_coords) >= 4:
            holes.append(inner_coords)

    try:
        polygon = Polygon(
            shell=outer_coords,
            holes=holes,
        )

    except Exception:
        return None

    return polygon


# ============================================================
# MAIN PROCESS
# ============================================================

def main():

    kml_files = sorted(
        RAW_DIR.glob("W*_Historical_Wetland.kml")
    )

    if not kml_files:
        raise FileNotFoundError(
            f"No KML files found in:\n{RAW_DIR}"
        )

    spatial_rows = []
    issue_rows = []

    # --------------------------------------------------------
    # READ ALL KML FILES
    # --------------------------------------------------------

    for kml_path in kml_files:

        file_id_match = ID_RE.search(
            kml_path.stem
        )

        if file_id_match is None:
            continue

        # Source filename becomes the canonical wetland ID.
        canonical_wetland_id = (
            file_id_match.group(0).upper()
        )

        try:
            root = ET.parse(
                kml_path
            ).getroot()

        except ET.ParseError as error:

            issue_rows.append(
                {
                    "source_file":
                        kml_path.name,

                    "placemark_name":
                        None,

                    "issue":
                        f"KML_XML_ERROR: {error}",
                }
            )

            continue

        placemarks = root.findall(
            ".//k:Placemark",
            KML_NS,
        )

        for placemark in placemarks:

            placemark_name = placemark.findtext(
                "k:name",
                default="",
                namespaces=KML_NS,
            ).strip()

            name_id_match = ID_RE.search(
                placemark_name
            )

            original_name_id = (
                name_id_match.group(0).upper()
                if name_id_match
                else None
            )

            observation_date, patch_label = (
                parse_date_from_name(
                    placemark_name
                )
            )

            if observation_date is None:

                issue_rows.append(
                    {
                        "source_file":
                            kml_path.name,

                        "placemark_name":
                            placemark_name,

                        "issue":
                            "DATE_PARSE_FAILED",
                    }
                )

                continue

            # Record naming inconsistencies,
            # but do NOT alter the raw KML.
            if (
                original_name_id is not None
                and original_name_id
                != canonical_wetland_id
            ):

                issue_rows.append(
                    {
                        "source_file":
                            kml_path.name,

                        "placemark_name":
                            placemark_name,

                        "issue":
                            (
                                "WETLAND_ID_MISMATCH_"
                                "CANONICAL_ID_USED"
                            ),
                    }
                )

            polygon_elements = placemark.findall(
                ".//k:Polygon",
                KML_NS,
            )

            if not polygon_elements:

                issue_rows.append(
                    {
                        "source_file":
                            kml_path.name,

                        "placemark_name":
                            placemark_name,

                        "issue":
                            "NO_POLYGON_GEOMETRY",
                    }
                )

                continue

            # One placemark may theoretically
            # contain more than one polygon.
            for polygon_index, polygon_element in enumerate(
                polygon_elements,
                start=1,
            ):

                geometry = polygon_from_kml_element(
                    polygon_element
                )

                if geometry is None:

                    issue_rows.append(
                        {
                            "source_file":
                                kml_path.name,

                            "placemark_name":
                                placemark_name,

                            "issue":
                                (
                                    "POLYGON_COORDINATE_"
                                    "PARSE_FAILED"
                                ),
                        }
                    )

                    continue

                original_valid = geometry.is_valid

                if not original_valid:
                    geometry = make_valid(
                        geometry
                    )

                if geometry.is_empty:

                    issue_rows.append(
                        {
                            "source_file":
                                kml_path.name,

                            "placemark_name":
                                placemark_name,

                            "issue":
                                (
                                    "EMPTY_GEOMETRY_"
                                    "AFTER_REPAIR"
                                ),
                        }
                    )

                    continue

                spatial_rows.append(
                    {
                        "wetland_id":
                            canonical_wetland_id,

                        "original_name_id":
                            original_name_id,

                        "source_file":
                            kml_path.name,

                        "placemark_name":
                            placemark_name,

                        "observation_date":
                            observation_date.isoformat(),

                        "year":
                            observation_date.year,

                        "month":
                            observation_date.month,

                        "day":
                            observation_date.day,

                        "season":
                            season_from_month(
                                observation_date.month
                            ),

                        "patch_label":
                            patch_label,

                        "polygon_index":
                            polygon_index,

                        "original_geometry_valid":
                            original_valid,

                        "geometry":
                            geometry,
                    }
                )

    # ========================================================
    # BUILD WGS84 GEODATAFRAME
    # ========================================================

    gdf_wgs84 = gpd.GeoDataFrame(
        spatial_rows,
        geometry="geometry",
        crs=WGS84,
    )

    if gdf_wgs84.empty:
        raise RuntimeError(
            "No polygon geometries were successfully parsed."
        )

    # ========================================================
    # REMOVE EXACT DUPLICATE GEOMETRY RECORDS
    # ========================================================

    gdf_wgs84["geometry_wkb"] = (
        gdf_wgs84.geometry.apply(
            lambda geom: geom.wkb_hex
        )
    )

    duplicate_mask = gdf_wgs84.duplicated(
        subset=[
            "wetland_id",
            "observation_date",
            "geometry_wkb",
        ],
        keep="first",
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:

        duplicate_rows = (
            gdf_wgs84.loc[
                duplicate_mask,
                [
                    "source_file",
                    "placemark_name",
                ],
            ]
        )

        for _, row in duplicate_rows.iterrows():

            issue_rows.append(
                {
                    "source_file":
                        row["source_file"],

                    "placemark_name":
                        row["placemark_name"],

                    "issue":
                        "EXACT_DUPLICATE_GEOMETRY_REMOVED",
                }
            )

        gdf_wgs84 = (
            gdf_wgs84.loc[
                ~duplicate_mask
            ].copy()
        )

    gdf_wgs84.drop(
        columns=["geometry_wkb"],
        inplace=True,
    )

    # ========================================================
    # CONVERT TO METRIC CRS
    # ========================================================

    gdf_utm = gdf_wgs84.to_crs(
        UTM44N
    )

    # Patch-level measurements
    gdf_utm["patch_area_m2"] = (
        gdf_utm.geometry.area
    )

    gdf_utm["patch_area_ha"] = (
        gdf_utm["patch_area_m2"]
        / 10000.0
    )

    gdf_utm["patch_perimeter_m"] = (
        gdf_utm.geometry.length
    )

    # ========================================================
    # TRANSFER METRICS TO WGS84 VERSION
    # ========================================================

    gdf_wgs84[
        "patch_area_m2"
    ] = gdf_utm[
        "patch_area_m2"
    ].to_numpy()

    gdf_wgs84[
        "patch_area_ha"
    ] = gdf_utm[
        "patch_area_ha"
    ].to_numpy()

    gdf_wgs84[
        "patch_perimeter_m"
    ] = gdf_utm[
        "patch_perimeter_m"
    ].to_numpy()

    # ========================================================
    # SAVE STANDARDIZED SPATIAL OUTPUTS
    # ========================================================

    wgs84_geojson_path = (
        PROCESSED_DIR
        / "historical_wetland_patches_wgs84.geojson"
    )

    gdf_wgs84.to_file(
        wgs84_geojson_path,
        driver="GeoJSON",
    )

    utm_gpkg_path = (
        PROCESSED_DIR
        / "historical_wetland_patches_utm44n.gpkg"
    )

    gdf_utm.to_file(
        utm_gpkg_path,
        layer="historical_wetland_patches",
        driver="GPKG",
    )

    # ========================================================
    # SAVE PATCH METRICS TABLE
    # ========================================================

    metrics_columns = [
        "wetland_id",
        "original_name_id",
        "source_file",
        "placemark_name",
        "observation_date",
        "year",
        "month",
        "day",
        "season",
        "patch_label",
        "polygon_index",
        "original_geometry_valid",
        "patch_area_m2",
        "patch_area_ha",
        "patch_perimeter_m",
    ]

    patch_metrics = (
        gdf_utm[
            metrics_columns
        ].copy()
    )

    patch_metrics_path = (
        PROCESSED_DIR
        / "historical_wetland_patch_metrics.csv"
    )

    patch_metrics.to_csv(
        patch_metrics_path,
        index=False,
    )

    # ========================================================
    # SAVE STANDARDIZATION ISSUES
    # ========================================================

    issues_df = pd.DataFrame(
        issue_rows
    )

    issues_path = (
        INTERIM_DIR
        / "historical_wetland_standardization_issues.csv"
    )

    issues_df.to_csv(
        issues_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    invalid_original_count = int(
        (~gdf_wgs84[
            "original_geometry_valid"
        ]).sum()
    )

    print(
        "\n========================================"
    )

    print(
        "HISTORICAL WETLAND SPATIAL STANDARDIZATION"
    )

    print(
        "========================================"
    )

    print(
        f"KML files processed: {len(kml_files)}"
    )

    print(
        f"Standardized polygon features: {len(gdf_wgs84)}"
    )

    print(
        "Wetlands represented: "
        f"{gdf_wgs84['wetland_id'].nunique()}"
    )

    print(
        "Observation dates represented: "
        f"{gdf_wgs84['observation_date'].nunique()}"
    )

    print(
        "Originally invalid geometries repaired: "
        f"{invalid_original_count}"
    )

    print(
        "Exact duplicate geometries removed: "
        f"{duplicate_count}"
    )

    print(
        "Remaining invalid geometries: "
        f"{(~gdf_wgs84.geometry.is_valid).sum()}"
    )

    print(
        "\nMetric CRS used: EPSG:32644"
    )

    print(
        "Area unit: hectares"
    )

    print(
        "\nOutputs:"
    )

    print(
        f"WGS84 GeoJSON:\n{wgs84_geojson_path}"
    )

    print(
        f"\nUTM GeoPackage:\n{utm_gpkg_path}"
    )

    print(
        f"\nPatch metrics CSV:\n{patch_metrics_path}"
    )

    print(
        f"\nIssues CSV:\n{issues_path}"
    )

    print(
        "\nNOTE:"
        "\nThe raw KML files were not modified."
        "\nSource filename IDs are used as the canonical wetland IDs."
    )


if __name__ == "__main__":
    main()