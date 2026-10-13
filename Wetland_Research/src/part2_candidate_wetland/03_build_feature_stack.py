from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling

from scipy.ndimage import (
    distance_transform_edt,
    uniform_filter,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

SENTINEL_DIR = RAW_DIR / "sentinel2"
DEM_DIR = RAW_DIR / "dem"
OSM_DIR = RAW_DIR / "osm"
BOUNDARY_DIR = RAW_DIR / "boundaries"

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part2_candidate_wetland"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "part2_candidate_wetland"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INTERIM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUTS
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

OSM_ROADS_BUILDINGS = (
    OSM_DIR
    / "Kaduwela_OSM_Roads_Buildings_2026.gpkg"
)

OSM_WATER_DRAINAGE = (
    OSM_DIR
    / "Kaduwela_OSM_Water_Drainage_2026.gpkg"
)

BOUNDARY_KML = (
    BOUNDARY_DIR
    / "Kaduwela TO Area.kml"
)

GN_LAYER = (
    "GN Boundary Colombo District (#2)"
)


# ============================================================
# OUTPUTS
# ============================================================

FEATURE_STACK_OUTPUT = (
    PROCESSED_DIR
    / "feature_stack_10m.tif"
)

STUDY_MASK_OUTPUT = (
    PROCESSED_DIR
    / "study_mask_10m.tif"
)

STUDY_BOUNDARY_OUTPUT = (
    PROCESSED_DIR
    / "study_boundary_wgs84.geojson"
)

FEATURE_METADATA_OUTPUT = (
    PROCESSED_DIR
    / "feature_stack_metadata.csv"
)

FEATURE_SUMMARY_OUTPUT = (
    INTERIM_DIR
    / "feature_stack_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_NODATA = -9999.0

SENTINEL_BAND_NAMES = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "ndvi",
    "mndwi",
    "ndbi",
]


# ============================================================
# HELPERS
# ============================================================

def read_sentinel(path):

    with rasterio.open(path) as src:

        data = src.read().astype(
            np.float32
        )

        if src.nodata is not None:

            data[
                data == src.nodata
            ] = np.nan

        profile = src.profile.copy()

        return (
            data,
            profile,
            src.transform,
            src.crs,
            src.width,
            src.height,
        )


def repair_invalid_geometries(
    gdf,
    label,
):

    invalid = (
        ~gdf.geometry.is_valid
    )

    invalid_count = int(
        invalid.sum()
    )

    print(
        f"{label} invalid geometries: "
        f"{invalid_count}"
    )

    if invalid_count > 0:

        gdf.loc[
            invalid,
            "geometry",
        ] = (
            gdf.loc[
                invalid,
                "geometry",
            ]
            .make_valid()
        )

    return gdf


def rasterize_binary(
    geometries,
    shape,
    transform,
    all_touched=True,
):

    geometry_list = [
        geometry
        for geometry in geometries
        if (
            geometry is not None
            and
            not geometry.is_empty
        )
    ]

    if not geometry_list:

        return np.zeros(
            shape,
            dtype=np.uint8,
        )

    return rasterize(
        [
            (
                geometry,
                1,
            )
            for geometry
            in geometry_list
        ],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )


def distance_from_mask(
    binary_mask,
    pixel_size,
):

    if not binary_mask.any():

        raise RuntimeError(
            "Cannot calculate distance: "
            "source mask contains no pixels."
        )

    return distance_transform_edt(
        binary_mask == 0,
        sampling=(
            pixel_size,
            pixel_size,
        ),
    ).astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("PART 2 FEATURE STACK")
    print("========================================")

    # ========================================================
    # 1. LOAD SENTINEL MASTER GRID
    # ========================================================

    (
        dry,
        dry_profile,
        transform,
        crs,
        width,
        height,
    ) = read_sentinel(
        S2_DRY
    )

    (
        wet,
        wet_profile,
        wet_transform,
        wet_crs,
        wet_width,
        wet_height,
    ) = read_sentinel(
        S2_WET
    )

    if (
        crs != wet_crs
        or
        width != wet_width
        or
        height != wet_height
        or
        not np.allclose(
            tuple(transform),
            tuple(wet_transform),
        )
    ):

        raise RuntimeError(
            "Dry and wet Sentinel rasters "
            "are not aligned."
        )

    pixel_size = abs(
        transform.a
    )

    print(
        f"Master CRS: {crs}"
    )

    print(
        f"Grid: {width} x {height}"
    )

    print(
        f"Pixel size: "
        f"{pixel_size:.2f} m"
    )

    # ========================================================
    # 2. BUILD STUDY AREA FROM GN POLYGONS
    # ========================================================

    gn = gpd.read_file(
        BOUNDARY_KML,
        layer=GN_LAYER,
    )

    print(
        f"\nGN polygons loaded: "
        f"{len(gn)}"
    )

    gn = repair_invalid_geometries(
        gn,
        "GN",
    )

    gn_utm = gn.to_crs(
        crs
    )

    study_geometry = (
        gn_utm.geometry
        .union_all()
    )

    study_mask = rasterize(
        [
            (
                study_geometry,
                1,
            )
        ],
        out_shape=(
            height,
            width,
        ),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )

    study_pixels = int(
        study_mask.sum()
    )

    rasterized_study_area_km2 = (
        study_pixels
        * pixel_size
        * pixel_size
        / 1_000_000.0
    )

    print(
        f"Study mask pixels: "
        f"{study_pixels:,}"
    )

    print(
        f"Rasterized study area: "
        f"{rasterized_study_area_km2:.2f} km²"
    )

    # Save dissolved boundary.
    study_boundary = gpd.GeoDataFrame(
        {
            "study_area": [
                "Kaduwela_GN_union"
            ]
        },
        geometry=[
            study_geometry
        ],
        crs=crs,
    )

    study_boundary_wgs84 = (
        study_boundary
        .to_crs(
            "EPSG:4326"
        )
    )

    study_boundary_wgs84.to_file(
        STUDY_BOUNDARY_OUTPUT,
        driver="GeoJSON",
    )

    # ========================================================
    # 3. SENTINEL VALID PIXELS
    # ========================================================

    dry_valid = np.all(
        np.isfinite(dry),
        axis=0,
    )

    wet_valid = np.all(
        np.isfinite(wet),
        axis=0,
    )

    sentinel_valid = (
        dry_valid
        &
        wet_valid
    )

    print(
        f"\nSentinel valid pixels "
        f"inside study area: "
        f"{int((sentinel_valid & (study_mask == 1)).sum()):,}"
    )

    # ========================================================
    # 4. REPROJECT GEDTM TO MASTER GRID
    # ========================================================

    print(
        "\nReprojecting GEDTM..."
    )

    dem = np.full(
        (
            height,
            width,
        ),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(
        GEDTM
    ) as dem_src:

        reproject(
            source=rasterio.band(
                dem_src,
                1,
            ),
            destination=dem,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            src_nodata=dem_src.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    dem_valid_inside = (
        np.isfinite(dem)
        &
        (study_mask == 1)
    )

    print(
        f"DEM valid pixels inside study area: "
        f"{int(dem_valid_inside.sum()):,}"
    )

    # ========================================================
    # 5. SLOPE
    #
    # GEDTM remains a ~30 m source even though it is
    # represented on the 10 m master grid.
    # ========================================================

    print(
        "Calculating slope..."
    )

    dz_dy, dz_dx = np.gradient(
        dem,
        pixel_size,
        pixel_size,
    )

    slope_degrees = np.degrees(
        np.arctan(
            np.sqrt(
                (
                    dz_dx
                    ** 2
                )
                +
                (
                    dz_dy
                    ** 2
                )
            )
        )
    ).astype(
        np.float32
    )

    # ========================================================
    # 6. LOAD OSM ROADS / BUILDINGS
    # ========================================================

    print(
        "\nPreparing OSM roads/buildings..."
    )

    osm = gpd.read_file(
        OSM_ROADS_BUILDINGS
    )

    osm = osm.to_crs(
        crs
    )

    # Buildings:
    # only building-tagged polygons.
    buildings = osm[
        (
            osm[
                "building"
            ].notna()
        )
        &
        (
            osm.geometry
            .geom_type
            .isin(
                [
                    "Polygon",
                    "MultiPolygon",
                ]
            )
        )
    ].copy()

    buildings = repair_invalid_geometries(
        buildings,
        "Building",
    )

    # Roads:
    # only highway-tagged line geometries.
    roads = osm[
        (
            osm[
                "highway"
            ].notna()
        )
        &
        (
            osm.geometry
            .geom_type
            .isin(
                [
                    "LineString",
                    "MultiLineString",
                ]
            )
        )
    ].copy()

    roads = repair_invalid_geometries(
        roads,
        "Road",
    )

    print(
        f"Building polygons selected: "
        f"{len(buildings):,}"
    )

    print(
        f"Road lines selected: "
        f"{len(roads):,}"
    )

    # ========================================================
    # 7. LOAD OSM DRAINAGE
    # ========================================================

    print(
        "\nPreparing OSM drainage..."
    )

    drainage = gpd.read_file(
        OSM_WATER_DRAINAGE
    )

    drainage = drainage.to_crs(
        crs
    )

    drainage = drainage[
        (
            drainage[
                "waterway"
            ].notna()
        )
        &
        (
            drainage.geometry
            .geom_type
            .isin(
                [
                    "LineString",
                    "MultiLineString",
                ]
            )
        )
    ].copy()

    drainage = repair_invalid_geometries(
        drainage,
        "Drainage",
    )

    print(
        f"Drainage lines selected: "
        f"{len(drainage):,}"
    )

    print(
        "Drainage categories:"
    )

    print(
        drainage[
            "waterway"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # 8. RASTERIZE OSM FEATURES
    # ========================================================

    print(
        "\nRasterizing spatial context..."
    )

    shape = (
        height,
        width,
    )

    building_mask = rasterize_binary(
        buildings.geometry,
        shape,
        transform,
        all_touched=True,
    )

    road_mask = rasterize_binary(
        roads.geometry,
        shape,
        transform,
        all_touched=True,
    )

    drainage_mask = rasterize_binary(
        drainage.geometry,
        shape,
        transform,
        all_touched=True,
    )

    print(
        f"Building pixels: "
        f"{int(building_mask.sum()):,}"
    )

    print(
        f"Road pixels: "
        f"{int(road_mask.sum()):,}"
    )

    print(
        f"Drainage pixels: "
        f"{int(drainage_mask.sum()):,}"
    )

    # ========================================================
    # 9. DISTANCE FEATURES
    # ========================================================

    print(
        "\nCalculating distance features..."
    )

    distance_to_building_m = (
        distance_from_mask(
            building_mask,
            pixel_size,
        )
    )

    distance_to_road_m = (
        distance_from_mask(
            road_mask,
            pixel_size,
        )
    )

    distance_to_drainage_m = (
        distance_from_mask(
            drainage_mask,
            pixel_size,
        )
    )

    # ========================================================
    # 10. BUILDING DENSITY
    #
    # 31 x 31 pixels ~= 310 m x 310 m neighbourhood.
    # ========================================================

    building_fraction_300m = (
        uniform_filter(
            building_mask.astype(
                np.float32
            ),
            size=31,
            mode="constant",
            cval=0.0,
        )
    )

    # ========================================================
    # 11. CREATE FEATURE LIST
    # ========================================================

    feature_names = []
    feature_arrays = []
    feature_sources = []

    # Dry Sentinel bands.
    for index, band_name in enumerate(
        SENTINEL_BAND_NAMES
    ):

        feature_names.append(
            f"dry_{band_name}"
        )

        feature_arrays.append(
            dry[index]
        )

        feature_sources.append(
            "Sentinel-2 dry composite"
        )

    # Wet Sentinel bands.
    for index, band_name in enumerate(
        SENTINEL_BAND_NAMES
    ):

        feature_names.append(
            f"wet_{band_name}"
        )

        feature_arrays.append(
            wet[index]
        )

        feature_sources.append(
            "Sentinel-2 wet composite"
        )

    # Seasonal index differences.
    ndvi_index = (
        SENTINEL_BAND_NAMES
        .index(
            "ndvi"
        )
    )

    mndwi_index = (
        SENTINEL_BAND_NAMES
        .index(
            "mndwi"
        )
    )

    ndbi_index = (
        SENTINEL_BAND_NAMES
        .index(
            "ndbi"
        )
    )

    feature_names.extend(
        [
            "delta_ndvi_wet_minus_dry",
            "delta_mndwi_wet_minus_dry",
            "delta_ndbi_wet_minus_dry",
        ]
    )

    feature_arrays.extend(
        [
            wet[
                ndvi_index
            ]
            -
            dry[
                ndvi_index
            ],

            wet[
                mndwi_index
            ]
            -
            dry[
                mndwi_index
            ],

            wet[
                ndbi_index
            ]
            -
            dry[
                ndbi_index
            ],
        ]
    )

    feature_sources.extend(
        [
            "Derived Sentinel seasonal change",
            "Derived Sentinel seasonal change",
            "Derived Sentinel seasonal change",
        ]
    )

    # Terrain.
    feature_names.extend(
        [
            "elevation_m",
            "slope_degrees",
        ]
    )

    feature_arrays.extend(
        [
            dem,
            slope_degrees,
        ]
    )

    feature_sources.extend(
        [
            "GEDTM30",
            "Derived from GEDTM30",
        ]
    )

    # Spatial context.
    feature_names.extend(
        [
            "distance_to_drainage_m",
            "distance_to_road_m",
            "distance_to_building_m",
            "building_fraction_300m",
        ]
    )

    feature_arrays.extend(
        [
            distance_to_drainage_m,
            distance_to_road_m,
            distance_to_building_m,
            building_fraction_300m,
        ]
    )

    feature_sources.extend(
        [
            "OSM water/drainage",
            "OSM highways",
            "OSM buildings",
            "OSM buildings",
        ]
    )

    # ========================================================
    # 12. STACK FEATURES
    # ========================================================

    feature_stack = np.stack(
        feature_arrays,
        axis=0,
    ).astype(
        np.float32
    )

    all_features_valid = np.all(
        np.isfinite(
            feature_stack
        ),
        axis=0,
    )

    valid_analysis_mask = (
        (study_mask == 1)
        &
        sentinel_valid
        &
        all_features_valid
    )

    valid_pixels = int(
        valid_analysis_mask.sum()
    )

    valid_area_km2 = (
        valid_pixels
        * pixel_size
        * pixel_size
        / 1_000_000.0
    )

    print(
        "\n========================================"
    )

    print(
        "FEATURE STACK SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Feature count: "
        f"{len(feature_names)}"
    )

    print(
        f"Valid analysis pixels: "
        f"{valid_pixels:,}"
    )

    print(
        f"Valid analysis area: "
        f"{valid_area_km2:.2f} km²"
    )

    # ========================================================
    # 13. SAVE STUDY MASK
    # ========================================================

    mask_profile = dry_profile.copy()

    mask_profile.update(
        {
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
            "compress": "deflate",
        }
    )

    with rasterio.open(
        STUDY_MASK_OUTPUT,
        "w",
        **mask_profile,
    ) as dst:

        dst.write(
            valid_analysis_mask.astype(
                np.uint8
            ),
            1,
        )

        dst.set_band_description(
            1,
            "valid_analysis_mask",
        )

    # ========================================================
    # 14. SAVE FEATURE STACK
    # ========================================================

    output_profile = dry_profile.copy()

    output_profile.update(
        {
            "count":
                len(
                    feature_names
                ),

            "dtype":
                "float32",

            "nodata":
                OUTPUT_NODATA,

            "compress":
                "deflate",

            "tiled":
                True,
        }
    )

    with rasterio.open(
        FEATURE_STACK_OUTPUT,
        "w",
        **output_profile,
    ) as dst:

        for band_index, (
            feature_name,
            array,
        ) in enumerate(
            zip(
                feature_names,
                feature_arrays,
            ),
            start=1,
        ):

            output_array = np.where(
                valid_analysis_mask,
                array,
                OUTPUT_NODATA,
            ).astype(
                np.float32
            )

            dst.write(
                output_array,
                band_index,
            )

            dst.set_band_description(
                band_index,
                feature_name,
            )

    # ========================================================
    # 15. FEATURE METADATA
    # ========================================================

    metadata = pd.DataFrame(
        {
            "band":
                range(
                    1,
                    len(
                        feature_names
                    )
                    + 1,
                ),

            "feature":
                feature_names,

            "source":
                feature_sources,
        }
    )

    metadata.to_csv(
        FEATURE_METADATA_OUTPUT,
        index=False,
    )

    # ========================================================
    # 16. FEATURE STATISTICS
    # ========================================================

    summary_rows = []

    for feature_name, array in zip(
        feature_names,
        feature_arrays,
    ):

        values = array[
            valid_analysis_mask
        ]

        summary_rows.append(
            {
                "feature":
                    feature_name,

                "count":
                    len(
                        values
                    ),

                "min":
                    float(
                        np.min(
                            values
                        )
                    ),

                "p05":
                    float(
                        np.percentile(
                            values,
                            5,
                        )
                    ),

                "median":
                    float(
                        np.median(
                            values
                        )
                    ),

                "mean":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "p95":
                    float(
                        np.percentile(
                            values,
                            95,
                        )
                    ),

                "max":
                    float(
                        np.max(
                            values
                        )
                    ),
            }
        )

    feature_summary = pd.DataFrame(
        summary_rows
    )

    feature_summary.to_csv(
        FEATURE_SUMMARY_OUTPUT,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\nFeatures:"
    )

    print(
        metadata.to_string(
            index=False
        )
    )

    print(
        "\nFeature ranges:"
    )

    print(
        feature_summary[
            [
                "feature",
                "min",
                "median",
                "max",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        FEATURE_STACK_OUTPUT
    )

    print(
        STUDY_MASK_OUTPUT
    )

    print(
        STUDY_BOUNDARY_OUTPUT
    )

    print(
        FEATURE_METADATA_OUTPUT
    )

    print(
        FEATURE_SUMMARY_OUTPUT
    )

    print(
        "\nIMPORTANT:"
        "\nThe 10 m grid is the analysis/output grid."
        "\nGEDTM terrain information remains based on "
        "the original ~30 m DEM source; resampling does "
        "not create new terrain detail."
        "\nOSM proximity features represent mapped OSM "
        "features and should not be interpreted as a "
        "complete municipal infrastructure inventory."
    )


if __name__ == "__main__":
    main()