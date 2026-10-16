"""
STEP 04 - SPATIAL PREPROCESSING

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Standardize the KMC boundary to EPSG:32644.
2. Standardize the processed 57-GN working layer to EPSG:32644.
3. Mask annual Landsat composites (2010-2026) to the confirmed KMC boundary.
4. Reproduce OSM vehicular-road and building filtering in code.
5. Clip roads and buildings to the KMC study area.
6. Reproject and clip GEDTM30 and NASADEM to the KMC study area.
7. Standardize ESA WorldCover 2021 to the KMC study area.
8. Produce reproducible preprocessing reports.

IMPORTANT:
- Raw datasets are NEVER overwritten.
- All new spatial datasets are written to data/processed/.
- EPSG:32644 is used as the common metric analysis CRS.
- UDA provisional zoning is NOT used as a core ML input here.
- UDA planning-reference processing will be handled separately later.
"""

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.mask import mask as raster_mask
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.features import geometry_mask

from shapely.geometry import mapping


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -------------------------------------------------------------------------
# Raw / source datasets
# -------------------------------------------------------------------------

KMC_RAW = (
    RAW_DIR
    / "boundaries"
    / "kmc"
    / "kmc boundry.kml"
)

GN_SOURCE_PROCESSED = (
    PROCESSED_DIR
    / "boundaries"
    / "gn_divisions"
    / "kaduwela_gn_divisions.shp"
)

LANDSAT_RAW_DIR = (
    RAW_DIR
    / "landsat"
    / "annual_composites"
)

OSM_RAW = (
    RAW_DIR
    / "osm"
    / "roads_buildings"
    / "Kaduwela_OSM_Roads_Buildings_2026.gpkg"
)

GEDTM30_RAW = (
    RAW_DIR
    / "dem"
    / "gedtm30"
    / "GEDTM30_Kaduwela_Buffer.tif"
)

NASADEM_RAW = (
    RAW_DIR
    / "dem"
    / "nasadem"
    / "NASADEM_Kaduwela_Buffer.tif"
)

WORLDCOVER_RAW = (
    RAW_DIR
    / "reference_lulc"
    / "esa_worldcover_2021"
    / "Kaduwela_ESA_WorldCover_2021.tif"
)


# -------------------------------------------------------------------------
# Processed output directories
# -------------------------------------------------------------------------

PROCESSED_BOUNDARY_DIR = (
    PROCESSED_DIR
    / "boundaries"
    / "kmc"
)

PROCESSED_GN_DIR = (
    PROCESSED_DIR
    / "boundaries"
    / "gn_divisions"
)

PROCESSED_LANDSAT_DIR = (
    PROCESSED_DIR
    / "landsat"
    / "annual_composites_kmc"
)

PROCESSED_OSM_DIR = (
    PROCESSED_DIR
    / "osm"
)

PROCESSED_DEM_DIR = (
    PROCESSED_DIR
    / "dem"
)

PROCESSED_REFERENCE_DIR = (
    PROCESSED_DIR
    / "reference_lulc"
)


for directory in [
    PROCESSED_BOUNDARY_DIR,
    PROCESSED_GN_DIR,
    PROCESSED_LANDSAT_DIR,
    PROCESSED_OSM_DIR,
    PROCESSED_DEM_DIR,
    PROCESSED_REFERENCE_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# -------------------------------------------------------------------------
# Final processed file paths
# -------------------------------------------------------------------------

KMC_OUTPUT = (
    PROCESSED_BOUNDARY_DIR
    / "kaduwela_kmc_boundary_utm44n.gpkg"
)

GN_OUTPUT = (
    PROCESSED_GN_DIR
    / "kaduwela_gn_divisions_utm44n.gpkg"
)

ROADS_OUTPUT = (
    PROCESSED_OSM_DIR
    / "kaduwela_roads_2026.gpkg"
)

BUILDINGS_OUTPUT = (
    PROCESSED_OSM_DIR
    / "kaduwela_buildings_2026.gpkg"
)

GEDTM30_OUTPUT = (
    PROCESSED_DEM_DIR
    / "gedtm30_kmc_30m.tif"
)

NASADEM_OUTPUT = (
    PROCESSED_DEM_DIR
    / "nasadem_kmc_30m.tif"
)

WORLDCOVER_OUTPUT = (
    PROCESSED_REFERENCE_DIR
    / "esa_worldcover_2021_kmc_10m.tif"
)


# =============================================================================
# 2. CONSTANTS
# =============================================================================

ANALYSIS_CRS = "EPSG:32644"

WGS84 = "EPSG:4326"


# Vehicular road classes selected during the earlier QGIS review.
ROAD_CLASSES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "residential",
    "living_street",
    "unclassified",
    "road",
    "service",
}


# Previous QGIS processing counts.
# These are comparison references only.
QGIS_REFERENCE_ROADS = 6543
QGIS_REFERENCE_BUILDINGS = 33673


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """
    Return a project-relative path when possible.
    """

    try:

        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:

        return str(path)


def remove_existing_file(path: Path):
    """
    Remove an existing output file before rewriting it.

    Only processed/output files are passed to this function.
    """

    if path.exists():
        path.unlink()


def safe_percent(
    numerator: float,
    denominator: float
) -> float:
    """
    Safely calculate a percentage.
    """

    if denominator == 0:
        return 0.0

    return float(
        numerator
        / denominator
        * 100
    )


def aligned_bounds(
    bounds,
    resolution
):
    """
    Align projected bounds to a regular raster grid.
    """

    minx, miny, maxx, maxy = bounds

    aligned_minx = (
        math.floor(
            minx / resolution
        )
        * resolution
    )

    aligned_miny = (
        math.floor(
            miny / resolution
        )
        * resolution
    )

    aligned_maxx = (
        math.ceil(
            maxx / resolution
        )
        * resolution
    )

    aligned_maxy = (
        math.ceil(
            maxy / resolution
        )
        * resolution
    )

    return (
        aligned_minx,
        aligned_miny,
        aligned_maxx,
        aligned_maxy,
    )


def clean_text_field(
    series: pd.Series
) -> pd.Series:
    """
    Convert attribute values to normalized nullable strings.
    """

    cleaned = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    return cleaned


def remove_exact_geometry_duplicates(
    gdf: gpd.GeoDataFrame
):
    """
    Remove exact duplicate geometries.

    Returns:
    cleaned_gdf,
    removed_count
    """

    duplicate_mask = (
        gdf.geometry.duplicated()
    )

    removed_count = int(
        duplicate_mask.sum()
    )

    cleaned = (
        gdf.loc[
            ~duplicate_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        cleaned,
        removed_count
    )


def save_gpkg(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    layer_name: str
):
    """
    Save a GeoDataFrame as a single-layer GeoPackage.
    """

    remove_existing_file(
        output_path
    )

    gdf.to_file(
        output_path,
        layer=layer_name,
        driver="GPKG",
    )


# =============================================================================
# 4. START
# =============================================================================

print("=" * 100)
print("STEP 04 - SPATIAL PREPROCESSING")
print("=" * 100)

print(
    f"\nProject root : "
    f"{PROJECT_ROOT}"
)

print(
    f"Analysis CRS : "
    f"{ANALYSIS_CRS}"
)


summary_records = []


# =============================================================================
# 5. KMC BOUNDARY PREPROCESSING
# =============================================================================

print("\n" + "=" * 100)
print("KMC BOUNDARY PREPROCESSING")
print("=" * 100)


if not KMC_RAW.exists():

    raise FileNotFoundError(
        f"KMC boundary not found:\n"
        f"{KMC_RAW}"
    )


kmc_raw_gdf = gpd.read_file(
    KMC_RAW
)


if kmc_raw_gdf.crs is None:

    raise ValueError(
        "KMC boundary CRS is missing."
    )


if not bool(
    kmc_raw_gdf.geometry.is_valid.all()
):

    raise ValueError(
        "KMC boundary contains invalid geometry."
    )


kmc_projected = (
    kmc_raw_gdf
    .to_crs(
        ANALYSIS_CRS
    )
)


kmc_union_geometry = (
    kmc_projected
    .geometry
    .union_all()
)


kmc_processed = (
    gpd.GeoDataFrame(
        {
            "study_area": [
                "Kaduwela Municipal Council"
            ]
        },
        geometry=[
            kmc_union_geometry
        ],
        crs=ANALYSIS_CRS,
    )
)


kmc_processed[
    "area_km2"
] = (
    kmc_processed
    .geometry
    .area
    / 1_000_000
)


save_gpkg(
    kmc_processed,
    KMC_OUTPUT,
    "kmc_boundary"
)


kmc_area_km2 = float(
    kmc_processed[
        "area_km2"
    ].iloc[0]
)


print(
    f"Input CRS       : "
    f"{kmc_raw_gdf.crs}"
)

print(
    f"Output CRS      : "
    f"{kmc_processed.crs}"
)

print(
    f"Output features : "
    f"{len(kmc_processed)}"
)

print(
    f"KMC area        : "
    f"{kmc_area_km2:.6f} km²"
)

print(
    f"Saved           : "
    f"{relative_path(KMC_OUTPUT)}"
)


summary_records.append(
    {
        "dataset": (
            "KMC Boundary"
        ),
        "operation": (
            "Reproject and dissolve"
        ),
        "input_count": (
            len(kmc_raw_gdf)
        ),
        "output_count": (
            len(kmc_processed)
        ),
        "output_crs": (
            str(
                kmc_processed.crs
            )
        ),
        "output_resolution": (
            None
        ),
        "valid_percent": (
            None
        ),
        "output_path": (
            relative_path(
                KMC_OUTPUT
            )
        ),
        "status": (
            "OK"
        ),
    }
)


# =============================================================================
# 6. GN DIVISION PREPROCESSING
# =============================================================================

print("\n" + "=" * 100)
print("57 GN DIVISION PREPROCESSING")
print("=" * 100)


if not GN_SOURCE_PROCESSED.exists():

    raise FileNotFoundError(
        f"57-GN working dataset not found:\n"
        f"{GN_SOURCE_PROCESSED}"
    )


gn_gdf = gpd.read_file(
    GN_SOURCE_PROCESSED
)


if gn_gdf.crs is None:

    raise ValueError(
        "GN division CRS is missing."
    )


if len(gn_gdf) != 57:

    raise ValueError(
        f"Expected 57 GN divisions, "
        f"found {len(gn_gdf)}."
    )


if not bool(
    gn_gdf.geometry.is_valid.all()
):

    raise ValueError(
        "GN dataset contains invalid geometry."
    )


gn_projected = (
    gn_gdf
    .to_crs(
        ANALYSIS_CRS
    )
    .copy()
)


gn_projected[
    "area_km2"
] = (
    gn_projected
    .geometry
    .area
    / 1_000_000
)


save_gpkg(
    gn_projected,
    GN_OUTPUT,
    "gn_divisions"
)


print(
    f"Input features  : "
    f"{len(gn_gdf)}"
)

print(
    f"Output features : "
    f"{len(gn_projected)}"
)

print(
    f"Output CRS      : "
    f"{gn_projected.crs}"
)

print(
    f"Saved           : "
    f"{relative_path(GN_OUTPUT)}"
)


summary_records.append(
    {
        "dataset": (
            "57 GN Divisions"
        ),
        "operation": (
            "Reproject and add area"
        ),
        "input_count": (
            len(gn_gdf)
        ),
        "output_count": (
            len(gn_projected)
        ),
        "output_crs": (
            str(
                gn_projected.crs
            )
        ),
        "output_resolution": (
            None
        ),
        "valid_percent": (
            None
        ),
        "output_path": (
            relative_path(
                GN_OUTPUT
            )
        ),
        "status": (
            "OK"
        ),
    }
)


# =============================================================================
# 7. LANDSAT MASKING TO CONFIRMED KMC BOUNDARY
# =============================================================================

print("\n" + "=" * 100)
print("LANDSAT KMC MASKING")
print("=" * 100)


landsat_files = sorted(
    LANDSAT_RAW_DIR.glob(
        "Kaduwela_Landsat_*.tif"
    )
)


if len(landsat_files) != 17:

    raise ValueError(
        f"Expected 17 Landsat files, "
        f"found {len(landsat_files)}."
    )


landsat_output_records = []


for landsat_file in landsat_files:

    year_match = re.search(
        r"(20\d{2})",
        landsat_file.stem
    )

    year = (
        int(
            year_match.group(1)
        )
        if year_match
        else None
    )


    output_path = (
        PROCESSED_LANDSAT_DIR
        / landsat_file.name
    )


    with rasterio.open(
        landsat_file
    ) as src:

        if src.crs is None:

            raise ValueError(
                f"Missing CRS: "
                f"{landsat_file.name}"
            )


        kmc_in_raster_crs = (
            kmc_processed
            .to_crs(
                src.crs
            )
        )


        geometries = [
            mapping(
                geometry
            )
            for geometry
            in kmc_in_raster_crs.geometry
        ]


        source_dtype = np.dtype(
            src.dtypes[0]
        )


        if np.issubdtype(
            source_dtype,
            np.floating
        ):

            output_nodata = -9999.0

        else:

            output_nodata = (
                np.iinfo(
                    source_dtype
                ).min
            )


        masked_data, masked_transform = (
            raster_mask(
                src,
                geometries,
                crop=True,
                all_touched=False,
                filled=True,
                nodata=output_nodata,
            )
        )


        output_meta = (
            src.meta.copy()
        )


        output_meta.update(
            {
                "driver": (
                    "GTiff"
                ),
                "height": (
                    masked_data.shape[1]
                ),
                "width": (
                    masked_data.shape[2]
                ),
                "transform": (
                    masked_transform
                ),
                "nodata": (
                    output_nodata
                ),
                "compress": (
                    "deflate"
                ),
            }
        )


        remove_existing_file(
            output_path
        )


        with rasterio.open(
            output_path,
            "w",
            **output_meta
        ) as dst:

            dst.write(
                masked_data
            )


            for band_number in range(
                1,
                src.count + 1
            ):

                description = (
                    src.descriptions[
                        band_number - 1
                    ]
                )

                if description:

                    dst.set_band_description(
                        band_number,
                        description
                    )


        first_band = (
            masked_data[0]
            .astype(
                "float64"
            )
        )


        if np.issubdtype(
            source_dtype,
            np.floating
        ):

            valid_mask = (
                np.isfinite(
                    first_band
                )
                &
                (
                    first_band
                    != output_nodata
                )
            )

        else:

            valid_mask = (
                first_band
                != output_nodata
            )


        valid_pixels = int(
            np.count_nonzero(
                valid_mask
            )
        )


        total_pixels = int(
            first_band.size
        )


        valid_percent = (
            safe_percent(
                valid_pixels,
                total_pixels
            )
        )


        landsat_output_records.append(
            {
                "year": (
                    year
                ),
                "bands": (
                    src.count
                ),
                "width": (
                    masked_data.shape[2]
                ),
                "height": (
                    masked_data.shape[1]
                ),
                "crs": (
                    str(
                        src.crs
                    )
                ),
                "resolution_x": (
                    src.res[0]
                ),
                "resolution_y": (
                    src.res[1]
                ),
                "valid_bbox_percent_band1": (
                    valid_percent
                ),
                "output_path": (
                    relative_path(
                        output_path
                    )
                ),
            }
        )


        print(
            f"{year}: "
            f"{masked_data.shape[2]} x "
            f"{masked_data.shape[1]}, "
            f"{src.count} bands -> OK"
        )


landsat_output_df = pd.DataFrame(
    landsat_output_records
)


summary_records.append(
    {
        "dataset": (
            "Landsat Annual Composites"
        ),
        "operation": (
            "Mask/crop to confirmed KMC boundary"
        ),
        "input_count": (
            len(landsat_files)
        ),
        "output_count": (
            len(
                landsat_output_df
            )
        ),
        "output_crs": (
            ANALYSIS_CRS
        ),
        "output_resolution": (
            "30 m"
        ),
        "valid_percent": (
            float(
                landsat_output_df[
                    "valid_bbox_percent_band1"
                ].min()
            )
        ),
        "output_path": (
            relative_path(
                PROCESSED_LANDSAT_DIR
            )
        ),
        "status": (
            "OK"
        ),
    }
)


# =============================================================================
# 8. OSM ROAD AND BUILDING PREPROCESSING
# =============================================================================

print("\n" + "=" * 100)
print("OSM ROAD AND BUILDING PREPROCESSING")
print("=" * 100)


if not OSM_RAW.exists():

    raise FileNotFoundError(
        f"OSM GeoPackage not found:\n"
        f"{OSM_RAW}"
    )


osm_gdf = gpd.read_file(
    OSM_RAW
)


if osm_gdf.crs is None:

    raise ValueError(
        "OSM CRS is missing."
    )


print(
    f"Raw OSM features : "
    f"{len(osm_gdf)}"
)


if "highway" not in osm_gdf.columns:

    raise ValueError(
        "OSM 'highway' field is missing."
    )


if "building" not in osm_gdf.columns:

    raise ValueError(
        "OSM 'building' field is missing."
    )


geometry_type = (
    osm_gdf
    .geometry
    .geom_type
)


highway_clean = (
    clean_text_field(
        osm_gdf[
            "highway"
        ]
    )
    .fillna("")
)


building_clean = (
    clean_text_field(
        osm_gdf[
            "building"
        ]
    )
    .fillna("")
)


# -------------------------------------------------------------------------
# Select vehicular road line features
# -------------------------------------------------------------------------

road_geometry_mask = (
    geometry_type.isin(
        [
            "LineString",
            "MultiLineString",
        ]
    )
)


road_class_mask = (
    highway_clean.isin(
        ROAD_CLASSES
    )
)


roads_candidates = (
    osm_gdf.loc[
        road_geometry_mask
        &
        road_class_mask
    ]
    .copy()
)


roads_candidates[
    "highway_clean"
] = (
    highway_clean.loc[
        roads_candidates.index
    ]
)


roads_candidate_count = len(
    roads_candidates
)


# -------------------------------------------------------------------------
# Select building polygon features
# -------------------------------------------------------------------------

building_geometry_mask = (
    geometry_type.isin(
        [
            "Polygon",
            "MultiPolygon",
        ]
    )
)


building_attribute_mask = (
    (
        building_clean != ""
    )
    &
    (
        building_clean != "no"
    )
)


building_candidates = (
    osm_gdf.loc[
        building_geometry_mask
        &
        building_attribute_mask
    ]
    .copy()
)


building_candidates[
    "building_clean"
] = (
    building_clean.loc[
        building_candidates.index
    ]
)


building_candidate_count = len(
    building_candidates
)


print(
    f"Road candidates before KMC clip     : "
    f"{roads_candidate_count}"
)

print(
    f"Building candidates before KMC clip : "
    f"{building_candidate_count}"
)


# -------------------------------------------------------------------------
# Reproject to analysis CRS
# -------------------------------------------------------------------------

roads_projected = (
    roads_candidates
    .to_crs(
        ANALYSIS_CRS
    )
)


buildings_projected = (
    building_candidates
    .to_crs(
        ANALYSIS_CRS
    )
)


# -------------------------------------------------------------------------
# Clip to confirmed KMC boundary
# -------------------------------------------------------------------------

roads_clipped = gpd.clip(
    roads_projected,
    kmc_processed,
    keep_geom_type=False,
)


buildings_clipped = gpd.clip(
    buildings_projected,
    kmc_processed,
    keep_geom_type=False,
)


# -------------------------------------------------------------------------
# Remove empty geometries
# -------------------------------------------------------------------------

roads_clipped = (
    roads_clipped.loc[
        roads_clipped.geometry.notna()
        &
        (
            ~roads_clipped.geometry.is_empty
        )
    ]
    .copy()
)


buildings_clipped = (
    buildings_clipped.loc[
        buildings_clipped.geometry.notna()
        &
        (
            ~buildings_clipped.geometry.is_empty
        )
    ]
    .copy()
)


# -------------------------------------------------------------------------
# Keep only correct final geometry types
# -------------------------------------------------------------------------

roads_clipped = (
    roads_clipped.loc[
        roads_clipped
        .geometry
        .geom_type
        .isin(
            [
                "LineString",
                "MultiLineString",
            ]
        )
    ]
    .copy()
)


buildings_clipped = (
    buildings_clipped.loc[
        buildings_clipped
        .geometry
        .geom_type
        .isin(
            [
                "Polygon",
                "MultiPolygon",
            ]
        )
    ]
    .copy()
)


roads_before_dedup = len(
    roads_clipped
)

buildings_before_dedup = len(
    buildings_clipped
)


# -------------------------------------------------------------------------
# Remove exact duplicate geometries
# -------------------------------------------------------------------------

roads_clipped, road_duplicates_removed = (
    remove_exact_geometry_duplicates(
        roads_clipped
    )
)


buildings_clipped, building_duplicates_removed = (
    remove_exact_geometry_duplicates(
        buildings_clipped
    )
)


# -------------------------------------------------------------------------
# Final geometry validation
# -------------------------------------------------------------------------

if not bool(
    roads_clipped.geometry.is_valid.all()
):

    raise ValueError(
        "Invalid road geometry remained "
        "after preprocessing."
    )


if not bool(
    buildings_clipped.geometry.is_valid.all()
):

    raise ValueError(
        "Invalid building geometry remained "
        "after preprocessing."
    )


# -------------------------------------------------------------------------
# Add useful metric attributes
# -------------------------------------------------------------------------

roads_clipped[
    "length_m"
] = (
    roads_clipped
    .geometry
    .length
)


buildings_clipped[
    "area_m2"
] = (
    buildings_clipped
    .geometry
    .area
)


roads_clipped = (
    roads_clipped
    .reset_index(
        drop=True
    )
)


buildings_clipped = (
    buildings_clipped
    .reset_index(
        drop=True
    )
)


save_gpkg(
    roads_clipped,
    ROADS_OUTPUT,
    "roads_2026"
)


save_gpkg(
    buildings_clipped,
    BUILDINGS_OUTPUT,
    "buildings_2026"
)


final_road_count = len(
    roads_clipped
)

final_building_count = len(
    buildings_clipped
)


print("\nFINAL OSM RESULTS")

print(
    f"Roads clipped before dedup      : "
    f"{roads_before_dedup}"
)

print(
    f"Road duplicates removed         : "
    f"{road_duplicates_removed}"
)

print(
    f"Final roads                     : "
    f"{final_road_count}"
)

print(
    f"Previous QGIS road reference    : "
    f"{QGIS_REFERENCE_ROADS}"
)

print(
    f"Road count difference           : "
    f"{final_road_count - QGIS_REFERENCE_ROADS}"
)


print(
    f"\nBuildings clipped before dedup  : "
    f"{buildings_before_dedup}"
)

print(
    f"Building duplicates removed     : "
    f"{building_duplicates_removed}"
)

print(
    f"Final buildings                 : "
    f"{final_building_count}"
)

print(
    f"Previous QGIS building reference: "
    f"{QGIS_REFERENCE_BUILDINGS}"
)

print(
    f"Building count difference       : "
    f"{final_building_count - QGIS_REFERENCE_BUILDINGS}"
)


print(
    f"\nRoads saved     : "
    f"{relative_path(ROADS_OUTPUT)}"
)

print(
    f"Buildings saved : "
    f"{relative_path(BUILDINGS_OUTPUT)}"
)


# -------------------------------------------------------------------------
# OSM road-class counts
# -------------------------------------------------------------------------

road_class_counts_df = (
    roads_clipped[
        "highway_clean"
    ]
    .value_counts()
    .rename_axis(
        "highway_class"
    )
    .reset_index(
        name="feature_count"
    )
)


summary_records.append(
    {
        "dataset": (
            "OSM Roads 2026"
        ),
        "operation": (
            "Select vehicular classes, reproject, clip, deduplicate"
        ),
        "input_count": (
            roads_candidate_count
        ),
        "output_count": (
            final_road_count
        ),
        "output_crs": (
            ANALYSIS_CRS
        ),
        "output_resolution": (
            None
        ),
        "valid_percent": (
            None
        ),
        "output_path": (
            relative_path(
                ROADS_OUTPUT
            )
        ),
        "status": (
            "OK"
        ),
    }
)


summary_records.append(
    {
        "dataset": (
            "OSM Buildings 2026"
        ),
        "operation": (
            "Select building polygons, reproject, clip, deduplicate"
        ),
        "input_count": (
            building_candidate_count
        ),
        "output_count": (
            final_building_count
        ),
        "output_crs": (
            ANALYSIS_CRS
        ),
        "output_resolution": (
            None
        ),
        "valid_percent": (
            None
        ),
        "output_path": (
            relative_path(
                BUILDINGS_OUTPUT
            )
        ),
        "status": (
            "OK"
        ),
    }
)


# =============================================================================
# 9. CONTINUOUS RASTER PREPROCESSING FUNCTION
# =============================================================================

def preprocess_continuous_raster(
    source_path: Path,
    output_path: Path,
    dataset_name: str,
    target_resolution: float = 30.0,
):
    """
    Reproject a continuous raster to EPSG:32644,
    align to a metric grid, and mask to KMC.

    Bilinear resampling is used for continuous elevation data.
    """

    if not source_path.exists():

        raise FileNotFoundError(
            f"{dataset_name} not found:\n"
            f"{source_path}"
        )


    kmc_bounds = (
        kmc_processed
        .total_bounds
    )


    minx, miny, maxx, maxy = (
        aligned_bounds(
            kmc_bounds,
            target_resolution
        )
    )


    width = int(
        round(
            (
                maxx - minx
            )
            / target_resolution
        )
    )


    height = int(
        round(
            (
                maxy - miny
            )
            / target_resolution
        )
    )


    target_transform = (
        from_origin(
            minx,
            maxy,
            target_resolution,
            target_resolution,
        )
    )


    output_nodata = -9999.0


    destination = np.full(
        (
            height,
            width
        ),
        output_nodata,
        dtype="float32",
    )


    with rasterio.open(
        source_path
    ) as src:

        source_data = (
            src.read(
                1
            )
        )


        reproject(
            source=source_data,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=target_transform,
            dst_crs=ANALYSIS_CRS,
            dst_nodata=output_nodata,
            resampling=Resampling.bilinear,
        )


    inside_kmc_mask = (
        geometry_mask(
            [
                mapping(
                    kmc_union_geometry
                )
            ],
            out_shape=(
                height,
                width
            ),
            transform=target_transform,
            invert=True,
            all_touched=False,
        )
    )


    destination[
        ~inside_kmc_mask
    ] = output_nodata


    valid_inside = (
        inside_kmc_mask
        &
        np.isfinite(
            destination
        )
        &
        (
            destination
            != output_nodata
        )
    )


    inside_count = int(
        np.count_nonzero(
            inside_kmc_mask
        )
    )


    valid_count = int(
        np.count_nonzero(
            valid_inside
        )
    )


    valid_percent = (
        safe_percent(
            valid_count,
            inside_count
        )
    )


    output_meta = {
        "driver": (
            "GTiff"
        ),
        "height": (
            height
        ),
        "width": (
            width
        ),
        "count": (
            1
        ),
        "dtype": (
            "float32"
        ),
        "crs": (
            ANALYSIS_CRS
        ),
        "transform": (
            target_transform
        ),
        "nodata": (
            output_nodata
        ),
        "compress": (
            "deflate"
        ),
    }


    remove_existing_file(
        output_path
    )


    with rasterio.open(
        output_path,
        "w",
        **output_meta
    ) as dst:

        dst.write(
            destination,
            1
        )


    valid_values = (
        destination[
            valid_inside
        ]
    )


    minimum = (
        float(
            np.min(
                valid_values
            )
        )
        if valid_values.size
        else None
    )


    maximum = (
        float(
            np.max(
                valid_values
            )
        )
        if valid_values.size
        else None
    )


    mean = (
        float(
            np.mean(
                valid_values
            )
        )
        if valid_values.size
        else None
    )


    print(
        f"{dataset_name}: "
        f"{width} x {height}, "
        f"{target_resolution:.0f} m, "
        f"valid KMC = {valid_percent:.4f}%"
    )


    return {
        "dataset": (
            dataset_name
        ),
        "operation": (
            "Reproject, resample and mask to KMC"
        ),
        "input_count": (
            1
        ),
        "output_count": (
            1
        ),
        "output_crs": (
            ANALYSIS_CRS
        ),
        "output_resolution": (
            f"{target_resolution:g} m"
        ),
        "valid_percent": (
            valid_percent
        ),
        "output_path": (
            relative_path(
                output_path
            )
        ),
        "status": (
            "OK"
        ),
        "minimum": (
            minimum
        ),
        "maximum": (
            maximum
        ),
        "mean": (
            mean
        ),
    }


# =============================================================================
# 10. DEM PREPROCESSING
# =============================================================================

print("\n" + "=" * 100)
print("DEM PREPROCESSING")
print("=" * 100)


gedtm_summary = (
    preprocess_continuous_raster(
        GEDTM30_RAW,
        GEDTM30_OUTPUT,
        "GEDTM30",
        target_resolution=30.0,
    )
)


nasadem_summary = (
    preprocess_continuous_raster(
        NASADEM_RAW,
        NASADEM_OUTPUT,
        "NASADEM",
        target_resolution=30.0,
    )
)


summary_records.append(
    gedtm_summary
)


summary_records.append(
    nasadem_summary
)


# =============================================================================
# 11. ESA WORLDCOVER PREPROCESSING
# =============================================================================

print("\n" + "=" * 100)
print("ESA WORLDCOVER PREPROCESSING")
print("=" * 100)


def preprocess_worldcover_native_grid(
    source_path: Path,
    output_path: Path,
):
    """
    Preserve the native ESA WorldCover grid and class values.

    The source raster is already:
    - EPSG:32644
    - 10 m resolution

    Therefore no reprojection or resampling is performed.
    The raster is only masked/cropped to the confirmed KMC boundary.

    Value 0 is retained as background/no-data.
    """

    if not source_path.exists():

        raise FileNotFoundError(
            f"ESA WorldCover not found:\n"
            f"{source_path}"
        )


    with rasterio.open(
        source_path
    ) as src:

        if str(src.crs) != ANALYSIS_CRS:

            raise ValueError(
                f"Unexpected WorldCover CRS: {src.crs}. "
                f"Expected {ANALYSIS_CRS}."
            )


        if not (
            math.isclose(
                src.res[0],
                10.0,
                abs_tol=1e-9,
            )
            and
            math.isclose(
                src.res[1],
                10.0,
                abs_tol=1e-9,
            )
        ):

            raise ValueError(
                f"Unexpected WorldCover resolution: "
                f"{src.res}"
            )


        kmc_in_worldcover_crs = (
            kmc_processed
            .to_crs(
                src.crs
            )
        )


        geometries = [
            mapping(
                geometry
            )
            for geometry
            in kmc_in_worldcover_crs.geometry
        ]


        masked_data, masked_transform = (
            raster_mask(
                src,
                geometries,
                crop=True,
                all_touched=False,
                filled=True,
                nodata=0,
            )
        )


        output_meta = (
            src.meta.copy()
        )


        output_meta.update(
            {
                "driver": "GTiff",
                "height": masked_data.shape[1],
                "width": masked_data.shape[2],
                "transform": masked_transform,
                "nodata": 0,
                "compress": "deflate",
            }
        )


        remove_existing_file(
            output_path
        )


        with rasterio.open(
            output_path,
            "w",
            **output_meta
        ) as dst:

            dst.write(
                masked_data
            )


    output_array = (
        masked_data[0]
    )


    # Create a KMC-only mask using the cropped WorldCover grid.
    # This prevents pixels outside the irregular KMC boundary
    # from being counted in the coverage denominator.
    inside_kmc_mask = (
        geometry_mask(
            [
                mapping(
                    kmc_union_geometry
                )
            ],
            out_shape=(
                masked_data.shape[1],
                masked_data.shape[2],
            ),
            transform=masked_transform,
            invert=True,
            all_touched=False,
        )
    )


    inside_kmc_pixels = int(
        np.count_nonzero(
            inside_kmc_mask
        )
    )


    valid_mask = (
        inside_kmc_mask
        &
        (
            output_array != 0
        )
    )


    valid_pixels = int(
        np.count_nonzero(
            valid_mask
        )
    )


    background_inside_kmc = (
        inside_kmc_pixels
        - valid_pixels
    )


    valid_percent = (
        safe_percent(
            valid_pixels,
            inside_kmc_pixels
        )
    )


    background_percent = (
        safe_percent(
            background_inside_kmc,
            inside_kmc_pixels
        )
    )


    unique_classes = sorted(
        [
            int(value)
            for value
            in np.unique(
                output_array[
                    valid_mask
                ]
            )
        ]
    )


    expected_classes = {
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        95,
        100,
    }


    unexpected_classes = sorted(
        set(
            unique_classes
        )
        -
        expected_classes
    )


    print(
        f"ESA WorldCover 2021: "
        f"{masked_data.shape[2]} x "
        f"{masked_data.shape[1]}, "
        f"10 m"
    )

    print(
        f"Valid KMC class coverage : "
        f"{valid_percent:.4f}%"
    )

    print(
        f"KMC background/no-data   : "
        f"{background_inside_kmc} pixels "
        f"({background_percent:.6f}%)"
    )

    print(
        f"Class values        : "
        f"{unique_classes}"
    )

    print(
        f"Unexpected classes  : "
        f"{unexpected_classes if unexpected_classes else 'None'}"
    )


    if unexpected_classes:

        raise ValueError(
            f"Unexpected WorldCover class values "
            f"found after preprocessing: "
            f"{unexpected_classes}"
        )


    return {
        "dataset": "ESA WorldCover 2021",
        "operation": (
            "Native-grid KMC mask/crop; "
            "no reprojection or resampling"
        ),
        "input_count": 1,
        "output_count": 1,
        "output_crs": ANALYSIS_CRS,
        "output_resolution": "10 m",
        "valid_percent": valid_percent,
        "output_path": (
            relative_path(
                output_path
            )
        ),
        "status": "OK",
        "minimum": None,
        "maximum": None,
        "mean": None,
    }


worldcover_summary = (
    preprocess_worldcover_native_grid(
        WORLDCOVER_RAW,
        WORLDCOVER_OUTPUT,
    )
)


summary_records.append(
    worldcover_summary
)


# =============================================================================
# 12. POST-PROCESSING VALIDATION
# =============================================================================

print("\n" + "=" * 100)
print("POST-PROCESSING VALIDATION")
print("=" * 100)


# -------------------------------------------------------------------------
# Vector validation
# -------------------------------------------------------------------------

processed_kmc_check = gpd.read_file(
    KMC_OUTPUT
)

processed_gn_check = gpd.read_file(
    GN_OUTPUT
)

processed_roads_check = gpd.read_file(
    ROADS_OUTPUT
)

processed_buildings_check = gpd.read_file(
    BUILDINGS_OUTPUT
)


vector_outputs_valid = all(
    [
        str(
            processed_kmc_check.crs
        ) == ANALYSIS_CRS,
        str(
            processed_gn_check.crs
        ) == ANALYSIS_CRS,
        str(
            processed_roads_check.crs
        ) == ANALYSIS_CRS,
        str(
            processed_buildings_check.crs
        ) == ANALYSIS_CRS,
        len(
            processed_gn_check
        ) == 57,
        bool(
            processed_kmc_check
            .geometry
            .is_valid
            .all()
        ),
        bool(
            processed_gn_check
            .geometry
            .is_valid
            .all()
        ),
        bool(
            processed_roads_check
            .geometry
            .is_valid
            .all()
        ),
        bool(
            processed_buildings_check
            .geometry
            .is_valid
            .all()
        ),
    ]
)


# -------------------------------------------------------------------------
# Landsat validation
# -------------------------------------------------------------------------

processed_landsat_files = sorted(
    PROCESSED_LANDSAT_DIR.glob(
        "Kaduwela_Landsat_*.tif"
    )
)


landsat_post_records = []


reference_landsat_grid = None


for file_path in (
    processed_landsat_files
):

    with rasterio.open(
        file_path
    ) as src:

        grid = (
            str(src.crs),
            src.width,
            src.height,
            src.count,
            src.transform,
            src.res,
        )


        if reference_landsat_grid is None:

            reference_landsat_grid = (
                grid
            )

            consistent = True

        else:

            consistent = (
                grid[0]
                ==
                reference_landsat_grid[0]
                and
                grid[1]
                ==
                reference_landsat_grid[1]
                and
                grid[2]
                ==
                reference_landsat_grid[2]
                and
                grid[3]
                ==
                reference_landsat_grid[3]
                and
                grid[4].almost_equals(
                    reference_landsat_grid[4]
                )
                and
                math.isclose(
                    grid[5][0],
                    reference_landsat_grid[5][0],
                    abs_tol=1e-9,
                )
                and
                math.isclose(
                    grid[5][1],
                    reference_landsat_grid[5][1],
                    abs_tol=1e-9,
                )
            )


        landsat_post_records.append(
            {
                "file": (
                    file_path.name
                ),
                "crs": (
                    str(src.crs)
                ),
                "bands": (
                    src.count
                ),
                "width": (
                    src.width
                ),
                "height": (
                    src.height
                ),
                "resolution_x": (
                    src.res[0]
                ),
                "resolution_y": (
                    src.res[1]
                ),
                "grid_consistent": (
                    consistent
                ),
            }
        )


landsat_post_df = pd.DataFrame(
    landsat_post_records
)


landsat_outputs_valid = (
    len(
        processed_landsat_files
    )
    == 17
    and
    bool(
        landsat_post_df[
            "grid_consistent"
        ].all()
    )
    and
    bool(
        (
            landsat_post_df[
                "bands"
            ]
            == 6
        ).all()
    )
    and
    bool(
        (
            landsat_post_df[
                "crs"
            ]
            == ANALYSIS_CRS
        ).all()
    )
)


# -------------------------------------------------------------------------
# Raster output validation
# -------------------------------------------------------------------------

raster_validation_records = []


for dataset_name, path, expected_res in [
    (
        "GEDTM30",
        GEDTM30_OUTPUT,
        30.0,
    ),
    (
        "NASADEM",
        NASADEM_OUTPUT,
        30.0,
    ),
    (
        "ESA WorldCover 2021",
        WORLDCOVER_OUTPUT,
        10.0,
    ),
]:

    with rasterio.open(
        path
    ) as src:

        resolution_ok = (
            math.isclose(
                src.res[0],
                expected_res,
                abs_tol=1e-9,
            )
            and
            math.isclose(
                src.res[1],
                expected_res,
                abs_tol=1e-9,
            )
        )


        crs_ok = (
            str(
                src.crs
            )
            == ANALYSIS_CRS
        )


        raster_validation_records.append(
            {
                "dataset": (
                    dataset_name
                ),
                "crs": (
                    str(
                        src.crs
                    )
                ),
                "resolution_x": (
                    src.res[0]
                ),
                "resolution_y": (
                    src.res[1]
                ),
                "crs_ok": (
                    crs_ok
                ),
                "resolution_ok": (
                    resolution_ok
                ),
            }
        )


raster_validation_df = pd.DataFrame(
    raster_validation_records
)


raster_outputs_valid = bool(
    (
        raster_validation_df[
            "crs_ok"
        ]
        &
        raster_validation_df[
            "resolution_ok"
        ]
    ).all()
)


print(
    f"Vector outputs valid   : "
    f"{vector_outputs_valid}"
)

print(
    f"Landsat outputs valid  : "
    f"{landsat_outputs_valid}"
)

print(
    f"Raster outputs valid   : "
    f"{raster_outputs_valid}"
)


# =============================================================================
# 13. SAVE REPORTS
# =============================================================================

summary_df = pd.DataFrame(
    summary_records
)


SUMMARY_REPORT = (
    REPORT_DIR
    / "04_spatial_preprocessing_summary.csv"
)


OSM_CLASS_REPORT = (
    REPORT_DIR
    / "04_osm_road_class_counts.csv"
)


LANDSAT_REPORT = (
    REPORT_DIR
    / "04_processed_landsat_grid_validation.csv"
)


RASTER_REPORT = (
    REPORT_DIR
    / "04_processed_raster_validation.csv"
)


summary_df.to_csv(
    SUMMARY_REPORT,
    index=False
)


road_class_counts_df.to_csv(
    OSM_CLASS_REPORT,
    index=False
)


landsat_post_df.to_csv(
    LANDSAT_REPORT,
    index=False
)


raster_validation_df.to_csv(
    RASTER_REPORT,
    index=False
)


# =============================================================================
# 14. FINAL SUMMARY
# =============================================================================

overall_preprocessing_pass = all(
    [
        vector_outputs_valid,
        landsat_outputs_valid,
        raster_outputs_valid,
    ]
)


print("\n" + "=" * 100)
print("STEP 04 SUMMARY")
print("=" * 100)


print(
    f"KMC area                     : "
    f"{kmc_area_km2:.6f} km²"
)

print(
    f"Processed GN features        : "
    f"{len(gn_projected)}"
)

print(
    f"Processed Landsat years      : "
    f"{len(processed_landsat_files)} / 17"
)

print(
    f"Final vehicular roads        : "
    f"{final_road_count}"
)

print(
    f"Previous QGIS road reference : "
    f"{QGIS_REFERENCE_ROADS}"
)

print(
    f"Final buildings              : "
    f"{final_building_count}"
)

print(
    f"Previous QGIS building ref   : "
    f"{QGIS_REFERENCE_BUILDINGS}"
)

print(
    f"Vector outputs valid         : "
    f"{vector_outputs_valid}"
)

print(
    f"Landsat outputs valid        : "
    f"{landsat_outputs_valid}"
)

print(
    f"Raster outputs valid         : "
    f"{raster_outputs_valid}"
)

print(
    f"\nOverall preprocessing status : "
    f"{'PASS' if overall_preprocessing_pass else 'REVIEW REQUIRED'}"
)


print("\nReports created:")

print(
    f" - "
    f"{relative_path(SUMMARY_REPORT)}"
)

print(
    f" - "
    f"{relative_path(OSM_CLASS_REPORT)}"
)

print(
    f" - "
    f"{relative_path(LANDSAT_REPORT)}"
)

print(
    f" - "
    f"{relative_path(RASTER_REPORT)}"
)


print("\nIMPORTANT:")

print(
    "1. No raw source dataset was overwritten."
)

print(
    "2. All metric vector outputs use EPSG:32644."
)

print(
    "3. Landsat data were only re-masked/cropped; "
    "spectral values were not resampled."
)

print(
    "4. DEM rasters were reprojected using bilinear resampling."
)

print(
    "5. WorldCover retained its native EPSG:32644, 10 m grid; "
    "only KMC masking/cropping was applied and no resampling was performed."
)

print(
    "6. OSM road/building data represent current infrastructure. "
    "They must not be used as historical predictors without "
    "explicit temporal-leakage controls."
)

print(
    "7. UDA provisional zoning was intentionally excluded from "
    "this core preprocessing step."
)

print("=" * 100)