"""
STEP 12B - TERRAIN PREDICTOR ENGINEERING

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Derive static terrain predictors from the processed GEDTM30 DEM.
2. Calculate elevation and slope statistics for each permanent 100 m grid cell.
3. Respect the exact KMC overlap of boundary grid cells.
4. Join terrain predictors to the Step 12A causal LULC predictor dataset.
5. Keep terrain features temporally safe and free from future information.

Primary DEM:
- GEDTM30 processed 30 m raster.

NASADEM:
- retained as an independent terrain sensitivity/reference dataset.
- not duplicated as a model predictor in this step.

Terrain predictors:
- elevation_mean_m
- elevation_std_m
- elevation_min_m
- elevation_max_m
- elevation_range_m
- slope_mean_deg
- slope_std_deg
- slope_max_deg

IMPORTANT:
- Terrain is treated as static.
- No future LULC information is used.
- No OSM roads or buildings are used.
- Grid ID and coordinates remain metadata only.
"""

from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from shapely.geometry import box


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PREDICTOR_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "predictors"
    / "12a_causal_lulc_predictors.csv"
)

GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "grid"
    / "kaduwela_100m_modelling_grid.gpkg"
)

KMC_BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "boundaries"
    / "kmc"
    / "kaduwela_kmc_boundary_utm44n.gpkg"
)

GEDTM_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "gedtm30_kmc_30m.tif"
)

NASADEM_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "nasadem_kmc_30m.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "predictors"
)

TERRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "terrain"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TERRAIN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. OUTPUTS
# =============================================================================

TERRAIN_GRID_CSV = (
    TERRAIN_DIR
    / "kaduwela_100m_static_terrain_predictors.csv"
)

COMBINED_PREDICTOR_CSV = (
    OUTPUT_DIR
    / "12b_causal_lulc_terrain_predictors.csv"
)

FEATURE_LIST_CSV = (
    REPORT_DIR
    / "12b_combined_predictor_feature_list.csv"
)

TERRAIN_QA_CSV = (
    REPORT_DIR
    / "12b_terrain_predictor_qa.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "12b_terrain_predictor_summary.csv"
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

TARGET_CRS = "EPSG:32644"

EXPECTED_GRID_CELLS = 9240

EXPECTED_MODEL_ROWS = 51310

EXPECTED_POSITIVE_ROWS = 844

AREA_TOLERANCE_M2 = 1e-5


TERRAIN_FEATURES = [
    "elevation_mean_m",
    "elevation_std_m",
    "elevation_min_m",
    "elevation_max_m",
    "elevation_range_m",
    "slope_mean_deg",
    "slope_std_deg",
    "slope_max_deg",
]


METADATA_COLUMNS = [
    "grid_id",
    "grid_row",
    "grid_col",
    "origin_year",
    "urban_conversion_target",
    "kmc_overlap_fraction",
    "boundary_cell",
]


# =============================================================================
# 4. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """Return project-relative path where possible."""

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def candidate_pixel_range(
    geometry,
    transform,
    width,
    height,
):
    """
    Return candidate raster rows and columns overlapping a geometry.

    Assumes north-up raster with no rotation.
    """

    minx, miny, maxx, maxy = (
        geometry.bounds
    )

    pixel_width = float(
        transform.a
    )

    pixel_height = float(
        abs(
            transform.e
        )
    )

    col_start = math.floor(
        (
            minx
            -
            transform.c
        )
        /
        pixel_width
    )

    col_stop = math.ceil(
        (
            maxx
            -
            transform.c
        )
        /
        pixel_width
    )

    row_start = math.floor(
        (
            transform.f
            -
            maxy
        )
        /
        pixel_height
    )

    row_stop = math.ceil(
        (
            transform.f
            -
            miny
        )
        /
        pixel_height
    )

    col_start = max(
        0,
        col_start,
    )

    row_start = max(
        0,
        row_start,
    )

    col_stop = min(
        width,
        col_stop,
    )

    row_stop = min(
        height,
        row_stop,
    )

    return (
        row_start,
        row_stop,
        col_start,
        col_stop,
    )


def pixel_bounds(
    row,
    col,
    transform,
):
    """Return bounds of one raster pixel."""

    left = (
        transform.c
        +
        col
        *
        transform.a
    )

    right = (
        left
        +
        transform.a
    )

    top = (
        transform.f
        +
        row
        *
        transform.e
    )

    bottom = (
        top
        +
        transform.e
    )

    return (
        min(
            left,
            right,
        ),
        min(
            bottom,
            top,
        ),
        max(
            left,
            right,
        ),
        max(
            bottom,
            top,
        ),
    )


def rectangle_overlap_area(
    first_bounds,
    second_bounds,
):
    """Return overlap area between two axis-aligned rectangles."""

    a_minx, a_miny, a_maxx, a_maxy = (
        first_bounds
    )

    b_minx, b_miny, b_maxx, b_maxy = (
        second_bounds
    )

    overlap_width = max(
        0.0,
        min(
            a_maxx,
            b_maxx,
        )
        -
        max(
            a_minx,
            b_minx,
        ),
    )

    overlap_height = max(
        0.0,
        min(
            a_maxy,
            b_maxy,
        )
        -
        max(
            a_miny,
            b_miny,
        ),
    )

    return (
        overlap_width
        *
        overlap_height
    )


def finite_difference_derivative(
    array,
    spacing,
    axis,
):
    """
    Calculate a finite-difference derivative.

    Uses:
    - central difference when both neighbours exist,
    - forward difference when only next neighbour exists,
    - backward difference when only previous neighbour exists.

    This avoids unnecessarily losing DEM boundary pixels.
    """

    array = np.asarray(
        array,
        dtype=float,
    )

    derivative = np.full(
        array.shape,
        np.nan,
        dtype=float,
    )

    if axis == 1:

        previous_value = np.full_like(
            array,
            np.nan,
        )

        next_value = np.full_like(
            array,
            np.nan,
        )

        previous_value[
            :,
            1:
        ] = array[
            :,
            :-1
        ]

        next_value[
            :,
            :-1
        ] = array[
            :,
            1:
        ]

    elif axis == 0:

        previous_value = np.full_like(
            array,
            np.nan,
        )

        next_value = np.full_like(
            array,
            np.nan,
        )

        previous_value[
            1:,
            :
        ] = array[
            :-1,
            :
        ]

        next_value[
            :-1,
            :
        ] = array[
            1:,
            :
        ]

    else:

        raise ValueError(
            "axis must be 0 or 1"
        )


    center_valid = np.isfinite(
        array
    )

    previous_valid = np.isfinite(
        previous_value
    )

    next_valid = np.isfinite(
        next_value
    )


    central_mask = (
        center_valid
        &
        previous_valid
        &
        next_valid
    )

    derivative[
        central_mask
    ] = (
        next_value[
            central_mask
        ]
        -
        previous_value[
            central_mask
        ]
    ) / (
        2.0
        *
        spacing
    )


    forward_mask = (
        center_valid
        &
        next_valid
        &
        ~central_mask
    )

    derivative[
        forward_mask
    ] = (
        next_value[
            forward_mask
        ]
        -
        array[
            forward_mask
        ]
    ) / spacing


    backward_mask = (
        center_valid
        &
        previous_valid
        &
        ~central_mask
        &
        ~forward_mask
    )

    derivative[
        backward_mask
    ] = (
        array[
            backward_mask
        ]
        -
        previous_value[
            backward_mask
        ]
    ) / spacing


    return derivative


def weighted_stats(
    values,
    weights,
):
    """Calculate area-weighted statistics."""

    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    valid = (
        np.isfinite(
            values
        )
        &
        np.isfinite(
            weights
        )
        &
        (
            weights
            >
            0
        )
    )

    if not np.any(
        valid
    ):

        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "valid_area_m2": 0.0,
        }


    valid_values = (
        values[
            valid
        ]
    )

    valid_weights = (
        weights[
            valid
        ]
    )


    weight_sum = float(
        valid_weights.sum()
    )


    weighted_mean = float(
        np.average(
            valid_values,
            weights=valid_weights,
        )
    )


    weighted_variance = float(
        np.average(
            (
                valid_values
                -
                weighted_mean
            )
            **
            2,
            weights=valid_weights,
        )
    )


    return {
        "mean": (
            weighted_mean
        ),
        "std": (
            float(
                np.sqrt(
                    max(
                        weighted_variance,
                        0.0,
                    )
                )
            )
        ),
        "min": (
            float(
                np.min(
                    valid_values
                )
            )
        ),
        "max": (
            float(
                np.max(
                    valid_values
                )
            )
        ),
        "valid_area_m2": (
            weight_sum
        ),
    }


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 12B - TERRAIN PREDICTOR ENGINEERING")
print("=" * 100)


for required_file in [
    INPUT_PREDICTOR_CSV,
    GRID_FILE,
    KMC_BOUNDARY_FILE,
    GEDTM_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


print(
    f"\nPrimary DEM       : "
    f"{relative_path(GEDTM_FILE)}"
)

print(
    f"NASADEM reference : "
    f"{relative_path(NASADEM_FILE)}"
)

print(
    "NASADEM used as model predictor : False"
)


# =============================================================================
# 6. LOAD STEP 12A DATASET
# =============================================================================

predictor_12a_df = pd.read_csv(
    INPUT_PREDICTOR_CSV
)


if len(
    predictor_12a_df
) != EXPECTED_MODEL_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_MODEL_ROWS:,} Step 12A rows, "
        f"found {len(predictor_12a_df):,}."
    )


if int(
    predictor_12a_df[
        "urban_conversion_target"
    ].sum()
) != EXPECTED_POSITIVE_ROWS:

    raise RuntimeError(
        "Step 12A positive target count does not match Step 11F."
    )


if predictor_12a_df.duplicated(
    subset=[
        "grid_id",
        "origin_year",
    ]
).any():

    raise RuntimeError(
        "Duplicate grid/origin rows detected in Step 12A."
    )


print(
    f"\nStep 12A rows     : "
    f"{len(predictor_12a_df):,}"
)

print(
    f"Positive rows     : "
    f"{predictor_12a_df['urban_conversion_target'].sum():,}"
)


# =============================================================================
# 7. LOAD GRID
# =============================================================================

grid_gdf = gpd.read_file(
    GRID_FILE
)


if grid_gdf.crs is None:

    raise RuntimeError(
        "100 m grid has no CRS."
    )


if str(
    grid_gdf.crs
).upper() != TARGET_CRS:

    grid_gdf = grid_gdf.to_crs(
        TARGET_CRS
    )


if len(
    grid_gdf
) != EXPECTED_GRID_CELLS:

    raise RuntimeError(
        f"Expected {EXPECTED_GRID_CELLS} grid cells, "
        f"found {len(grid_gdf)}."
    )


if grid_gdf[
    "grid_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate grid IDs found."
    )


# =============================================================================
# 8. LOAD KMC BOUNDARY
# =============================================================================

kmc_gdf = gpd.read_file(
    KMC_BOUNDARY_FILE
)


if kmc_gdf.crs is None:

    raise RuntimeError(
        "KMC boundary has no CRS."
    )


if str(
    kmc_gdf.crs
).upper() != TARGET_CRS:

    kmc_gdf = kmc_gdf.to_crs(
        TARGET_CRS
    )


kmc_geometry = (
    kmc_gdf.geometry
    .union_all()
)


if kmc_geometry.is_empty:

    raise RuntimeError(
        "KMC boundary geometry is empty."
    )


# =============================================================================
# 9. LOAD GEDTM DEM
# =============================================================================

with rasterio.open(
    GEDTM_FILE
) as src:

    if src.crs is None:

        raise RuntimeError(
            "GEDTM raster has no CRS."
        )


    if str(
        src.crs
    ).upper() != TARGET_CRS:

        raise RuntimeError(
            f"Unexpected GEDTM CRS: "
            f"{src.crs}"
        )


    if src.count != 1:

        raise RuntimeError(
            "GEDTM raster must contain one band."
        )


    if (
        abs(
            src.transform.b
        )
        >
        1e-12
        or
        abs(
            src.transform.d
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "Rotated DEM grids are not supported."
        )


    dem_masked = src.read(
        1,
        masked=True,
    )


    dem_array = np.asarray(
        dem_masked.filled(
            np.nan
        ),
        dtype=float,
    )


    dem_transform = (
        src.transform
    )

    dem_width = (
        src.width
    )

    dem_height = (
        src.height
    )

    pixel_width_m = float(
        abs(
            src.transform.a
        )
    )

    pixel_height_m = float(
        abs(
            src.transform.e
        )
    )

    dem_nodata = (
        src.nodata
    )


print(
    f"\nGEDTM grid        : "
    f"{dem_width} x {dem_height}"
)

print(
    f"GEDTM resolution  : "
    f"{pixel_width_m:.3f} m x "
    f"{pixel_height_m:.3f} m"
)

print(
    f"GEDTM NoData      : "
    f"{dem_nodata}"
)


# =============================================================================
# 10. CALCULATE SLOPE
# =============================================================================

dz_dx = finite_difference_derivative(
    dem_array,
    pixel_width_m,
    axis=1,
)


dz_dy = finite_difference_derivative(
    dem_array,
    pixel_height_m,
    axis=0,
)


slope_array = np.degrees(
    np.arctan(
        np.sqrt(
            dz_dx
            **
            2
            +
            dz_dy
            **
            2
        )
    )
)


slope_array[
    ~np.isfinite(
        dem_array
    )
] = np.nan


valid_slope_values = (
    slope_array[
        np.isfinite(
            slope_array
        )
    ]
)


if len(
    valid_slope_values
) == 0:

    raise RuntimeError(
        "No valid slope values were calculated."
    )


if np.min(
    valid_slope_values
) < 0:

    raise RuntimeError(
        "Negative slope values detected."
    )


if np.max(
    valid_slope_values
) > 90.0 + 1e-9:

    raise RuntimeError(
        "Slope values exceed 90 degrees."
    )


print(
    f"Slope valid pixels : "
    f"{len(valid_slope_values):,}"
)

print(
    f"Slope range        : "
    f"{np.min(valid_slope_values):.3f} - "
    f"{np.max(valid_slope_values):.3f} degrees"
)


# =============================================================================
# 11. AGGREGATE TERRAIN TO 100 M GRID
# =============================================================================

print("\n" + "=" * 100)
print("AGGREGATING TERRAIN TO 100 M GRID")
print("=" * 100)


terrain_records = []

geometry_reconstruction_errors = []


for counter, (
    _,
    grid_row,
) in enumerate(
    grid_gdf.iterrows(),
    start=1,
):

    grid_id = (
        grid_row[
            "grid_id"
        ]
    )


    cell_geometry = (
        grid_row.geometry
    )


    boundary_cell = bool(
        grid_row[
            "boundary_cell"
        ]
    )


    kmc_overlap_area_m2 = float(
        grid_row[
            "kmc_overlap_area_m2"
        ]
    )


    if boundary_cell:

        analysis_geometry = (
            cell_geometry.intersection(
                kmc_geometry
            )
        )

    else:

        analysis_geometry = (
            cell_geometry
        )


    if analysis_geometry.is_empty:

        raise RuntimeError(
            f"Empty grid/KMC geometry for {grid_id}."
        )


    (
        row_start,
        row_stop,
        col_start,
        col_stop,
    ) = candidate_pixel_range(
        analysis_geometry,
        dem_transform,
        dem_width,
        dem_height,
    )


    value_rows = []

    value_cols = []

    overlap_areas = []


    cell_bounds = (
        cell_geometry.bounds
    )


    for raster_row in range(
        row_start,
        row_stop,
    ):

        for raster_col in range(
            col_start,
            col_stop,
        ):

            current_pixel_bounds = pixel_bounds(
                raster_row,
                raster_col,
                dem_transform,
            )


            if boundary_cell:

                current_pixel_geometry = box(
                    *current_pixel_bounds
                )


                overlap_area_m2 = float(
                    analysis_geometry.intersection(
                        current_pixel_geometry
                    ).area
                )

            else:

                overlap_area_m2 = float(
                    rectangle_overlap_area(
                        cell_bounds,
                        current_pixel_bounds,
                    )
                )


            if overlap_area_m2 <= 0:

                continue


            value_rows.append(
                raster_row
            )

            value_cols.append(
                raster_col
            )

            overlap_areas.append(
                overlap_area_m2
            )


    if len(
        overlap_areas
    ) == 0:

        raise RuntimeError(
            f"No DEM overlap found for {grid_id}."
        )


    rows_array = np.asarray(
        value_rows,
        dtype=np.int32,
    )

    cols_array = np.asarray(
        value_cols,
        dtype=np.int32,
    )

    areas_array = np.asarray(
        overlap_areas,
        dtype=float,
    )


    reconstructed_area_m2 = float(
        areas_array.sum()
    )


    reconstruction_error_m2 = (
        reconstructed_area_m2
        -
        kmc_overlap_area_m2
    )


    geometry_reconstruction_errors.append(
        abs(
            reconstruction_error_m2
        )
    )


    elevation_values = (
        dem_array[
            rows_array,
            cols_array,
        ]
    )


    slope_values = (
        slope_array[
            rows_array,
            cols_array,
        ]
    )


    elevation_stats = weighted_stats(
        elevation_values,
        areas_array,
    )


    slope_stats = weighted_stats(
        slope_values,
        areas_array,
    )


    elevation_coverage_fraction = (
        elevation_stats[
            "valid_area_m2"
        ]
        /
        kmc_overlap_area_m2
        if kmc_overlap_area_m2
        >
        0
        else
        np.nan
    )


    slope_coverage_fraction = (
        slope_stats[
            "valid_area_m2"
        ]
        /
        kmc_overlap_area_m2
        if kmc_overlap_area_m2
        >
        0
        else
        np.nan
    )


    terrain_records.append(
        {
            "grid_id": (
                grid_id
            ),
            "grid_row": int(
                grid_row[
                    "grid_row"
                ]
            ),
            "grid_col": int(
                grid_row[
                    "grid_col"
                ]
            ),
            "boundary_cell": (
                boundary_cell
            ),
            "kmc_overlap_fraction": float(
                grid_row[
                    "kmc_overlap_fraction"
                ]
            ),
            "elevation_mean_m": (
                elevation_stats[
                    "mean"
                ]
            ),
            "elevation_std_m": (
                elevation_stats[
                    "std"
                ]
            ),
            "elevation_min_m": (
                elevation_stats[
                    "min"
                ]
            ),
            "elevation_max_m": (
                elevation_stats[
                    "max"
                ]
            ),
            "elevation_range_m": (
                elevation_stats[
                    "max"
                ]
                -
                elevation_stats[
                    "min"
                ]
                if (
                    np.isfinite(
                        elevation_stats[
                            "max"
                        ]
                    )
                    and
                    np.isfinite(
                        elevation_stats[
                            "min"
                        ]
                    )
                )
                else
                np.nan
            ),
            "slope_mean_deg": (
                slope_stats[
                    "mean"
                ]
            ),
            "slope_std_deg": (
                slope_stats[
                    "std"
                ]
            ),
            "slope_max_deg": (
                slope_stats[
                    "max"
                ]
            ),
            "elevation_valid_coverage_fraction": (
                elevation_coverage_fraction
            ),
            "slope_valid_coverage_fraction": (
                slope_coverage_fraction
            ),
        }
    )


    if (
        counter
        %
        1000
        ==
        0
        or
        counter
        ==
        len(
            grid_gdf
        )
    ):

        print(
            f"Processed "
            f"{counter}/{len(grid_gdf)} "
            f"grid cells"
        )


terrain_df = pd.DataFrame(
    terrain_records
)


# =============================================================================
# 12. TERRAIN GRID QA
# =============================================================================

maximum_geometry_error_m2 = float(
    max(
        geometry_reconstruction_errors
    )
)


geometry_reconstruction_valid = (
    maximum_geometry_error_m2
    <=
    AREA_TOLERANCE_M2
)


duplicate_terrain_grid_ids = int(
    terrain_df[
        "grid_id"
    ]
    .duplicated()
    .sum()
)


terrain_feature_missing_all_grid = int(
    terrain_df[
        TERRAIN_FEATURES
    ]
    .isna()
    .sum()
    .sum()
)


minimum_elevation_coverage = float(
    terrain_df[
        "elevation_valid_coverage_fraction"
    ]
    .min()
)


minimum_slope_coverage = float(
    terrain_df[
        "slope_valid_coverage_fraction"
    ]
    .min()
)


maximum_slope = float(
    terrain_df[
        "slope_max_deg"
    ]
    .max()
)


# =============================================================================
# 13. JOIN TERRAIN TO STEP 12A
# =============================================================================

terrain_merge_columns = (
    [
        "grid_id",
    ]
    +
    TERRAIN_FEATURES
    +
    [
        "elevation_valid_coverage_fraction",
        "slope_valid_coverage_fraction",
    ]
)


combined_df = predictor_12a_df.merge(
    terrain_df[
        terrain_merge_columns
    ],
    on="grid_id",
    how="left",
    validate="many_to_one",
)


if len(
    combined_df
) != len(
    predictor_12a_df
):

    raise RuntimeError(
        "Terrain merge changed the number of modelling rows."
    )


# =============================================================================
# 14. FINAL FEATURE LIST
# =============================================================================

all_feature_columns = [
    column
    for column in combined_df.columns
    if column not in (
        METADATA_COLUMNS
        +
        [
            "elevation_valid_coverage_fraction",
            "slope_valid_coverage_fraction",
        ]
    )
]


# Coverage columns are QA fields, not ML predictors.
if (
    "elevation_valid_coverage_fraction"
    in
    all_feature_columns
    or
    "slope_valid_coverage_fraction"
    in
    all_feature_columns
):

    raise RuntimeError(
        "Terrain coverage QA fields entered the feature list."
    )


for prohibited_column in [
    "grid_id",
    "grid_row",
    "grid_col",
]:

    if prohibited_column in all_feature_columns:

        raise RuntimeError(
            f"Spatial identifier leaked into model feature list: "
            f"{prohibited_column}"
        )


# =============================================================================
# 15. MODEL-DATA QA
# =============================================================================

missing_terrain_values_target_rows = int(
    combined_df[
        TERRAIN_FEATURES
    ]
    .isna()
    .sum()
    .sum()
)


infinite_terrain_values_target_rows = int(
    np.isinf(
        combined_df[
            TERRAIN_FEATURES
        ]
        .to_numpy(
            dtype=float
        )
    ).sum()
)


target_unchanged = bool(
    np.array_equal(
        combined_df[
            "urban_conversion_target"
        ].to_numpy(),
        predictor_12a_df[
            "urban_conversion_target"
        ].to_numpy(),
    )
)


positive_count_unchanged = (
    int(
        combined_df[
            "urban_conversion_target"
        ].sum()
    )
    ==
    EXPECTED_POSITIVE_ROWS
)


duplicate_model_rows = int(
    combined_df.duplicated(
        subset=[
            "grid_id",
            "origin_year",
        ]
    ).sum()
)


terrain_values_finite_for_model = (
    missing_terrain_values_target_rows
    ==
    0
    and
    infinite_terrain_values_target_rows
    ==
    0
)


slope_bounds_valid = bool(
    (
        combined_df[
            "slope_mean_deg"
        ]
        >=
        0
    ).all()
    and
    (
        combined_df[
            "slope_max_deg"
        ]
        <=
        90.0
        +
        1e-9
    ).all()
)


static_terrain_consistent = True


for feature_name in TERRAIN_FEATURES:

    unique_per_grid = (
        combined_df.groupby(
            "grid_id"
        )[
            feature_name
        ]
        .nunique(
            dropna=False
        )
    )


    if (
        unique_per_grid
        >
        1
    ).any():

        static_terrain_consistent = (
            False
        )

        break


# =============================================================================
# 16. FEATURE LIST REPORT
# =============================================================================

feature_list_records = []


for feature_name in all_feature_columns:

    if feature_name in TERRAIN_FEATURES:

        feature_group = (
            "STATIC_TERRAIN"
        )

        temporal_role = (
            "STATIC"
        )

    else:

        feature_group = (
            "CAUSAL_RAW_LULC"
        )

        temporal_role = (
            "T_MINUS_2_TO_T"
        )


    feature_list_records.append(
        {
            "feature_name": (
                feature_name
            ),
            "feature_group": (
                feature_group
            ),
            "temporal_role": (
                temporal_role
            ),
            "future_information_used": (
                False
            ),
            "stabilized_lulc_used": (
                False
            ),
            "osm_2026_used": (
                False
            ),
        }
    )


feature_list_df = pd.DataFrame(
    feature_list_records
)


# =============================================================================
# 17. FINAL STATUS
# =============================================================================

overall_status = (
    "PASS"
    if all(
        [
            geometry_reconstruction_valid,
            duplicate_terrain_grid_ids
            ==
            0,
            len(
                terrain_df
            )
            ==
            EXPECTED_GRID_CELLS,
            len(
                combined_df
            )
            ==
            EXPECTED_MODEL_ROWS,
            missing_terrain_values_target_rows
            ==
            0,
            infinite_terrain_values_target_rows
            ==
            0,
            target_unchanged,
            positive_count_unchanged,
            duplicate_model_rows
            ==
            0,
            slope_bounds_valid,
            static_terrain_consistent,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 18. REPORTS
# =============================================================================

qa_df = pd.DataFrame(
    [
        {
            "terrain_grid_cells": (
                len(
                    terrain_df
                )
            ),
            "model_rows": (
                len(
                    combined_df
                )
            ),
            "total_model_features": (
                len(
                    all_feature_columns
                )
            ),
            "terrain_feature_count": (
                len(
                    TERRAIN_FEATURES
                )
            ),
            "duplicate_terrain_grid_ids": (
                duplicate_terrain_grid_ids
            ),
            "duplicate_model_grid_origin_rows": (
                duplicate_model_rows
            ),
            "maximum_geometry_reconstruction_error_m2": (
                maximum_geometry_error_m2
            ),
            "minimum_elevation_coverage_fraction_all_grid": (
                minimum_elevation_coverage
            ),
            "minimum_slope_coverage_fraction_all_grid": (
                minimum_slope_coverage
            ),
            "missing_terrain_values_target_rows": (
                missing_terrain_values_target_rows
            ),
            "infinite_terrain_values_target_rows": (
                infinite_terrain_values_target_rows
            ),
            "target_unchanged": (
                target_unchanged
            ),
            "positive_count_unchanged": (
                positive_count_unchanged
            ),
            "slope_bounds_valid": (
                slope_bounds_valid
            ),
            "static_terrain_consistent_across_origins": (
                static_terrain_consistent
            ),
            "nasadem_used_as_model_predictor": (
                False
            ),
            "osm_2026_used": (
                False
            ),
            "overall_step12b_status": (
                overall_status
            ),
        }
    ]
)


final_summary_df = pd.DataFrame(
    [
        {
            "input_predictor_dataset": (
                "12a_causal_lulc_predictors.csv"
            ),
            "primary_dem": (
                "GEDTM30"
            ),
            "nasadem_role": (
                "Sensitivity/reference only"
            ),
            "terrain_static": (
                True
            ),
            "lulc_feature_count_before_terrain": (
                len(
                    all_feature_columns
                )
                -
                len(
                    TERRAIN_FEATURES
                )
            ),
            "terrain_feature_count": (
                len(
                    TERRAIN_FEATURES
                )
            ),
            "combined_feature_count": (
                len(
                    all_feature_columns
                )
            ),
            "model_rows": (
                len(
                    combined_df
                )
            ),
            "positive_target_rows": int(
                combined_df[
                    "urban_conversion_target"
                ].sum()
            ),
            "future_information_used": (
                False
            ),
            "centered_stabilized_lulc_used": (
                False
            ),
            "osm_2026_used": (
                False
            ),
            "overall_step12b_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 19. SAVE OUTPUTS
# =============================================================================

terrain_df.to_csv(
    TERRAIN_GRID_CSV,
    index=False,
)


combined_df.to_csv(
    COMBINED_PREDICTOR_CSV,
    index=False,
)


feature_list_df.to_csv(
    FEATURE_LIST_CSV,
    index=False,
)


qa_df.to_csv(
    TERRAIN_QA_CSV,
    index=False,
)


final_summary_df.to_csv(
    FINAL_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 20. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("STATIC TERRAIN SUMMARY")
print("=" * 100)


print(
    terrain_df[
        TERRAIN_FEATURES
    ]
    .describe()
    .T.to_string()
)


print("\n" + "=" * 100)
print("COMBINED PREDICTOR DATASET")
print("=" * 100)


print(
    f"Rows                           : "
    f"{len(combined_df):,}"
)

print(
    f"Original LULC features         : "
    f"{len(all_feature_columns) - len(TERRAIN_FEATURES)}"
)

print(
    f"Terrain features added         : "
    f"{len(TERRAIN_FEATURES)}"
)

print(
    f"Total predictor features       : "
    f"{len(all_feature_columns)}"
)

print(
    f"Positive target rows           : "
    f"{combined_df['urban_conversion_target'].sum():,}"
)


print("\nTerrain feature names:")

for feature_name in TERRAIN_FEATURES:

    print(
        f" - {feature_name}"
    )


print("\n" + "=" * 100)
print("STEP 12B QA")
print("=" * 100)


print(
    f"Terrain grid cells                    : "
    f"{len(terrain_df):,}"
)

print(
    f"Duplicate terrain grid IDs            : "
    f"{duplicate_terrain_grid_ids}"
)

print(
    f"Maximum geometry reconstruction error : "
    f"{maximum_geometry_error_m2:.10f} m²"
)

print(
    f"Minimum elevation coverage            : "
    f"{minimum_elevation_coverage:.6f}"
)

print(
    f"Minimum slope coverage                : "
    f"{minimum_slope_coverage:.6f}"
)

print(
    f"Missing terrain values in model rows  : "
    f"{missing_terrain_values_target_rows}"
)

print(
    f"Infinite terrain values               : "
    f"{infinite_terrain_values_target_rows}"
)

print(
    f"Target unchanged                      : "
    f"{target_unchanged}"
)

print(
    f"Positive count unchanged              : "
    f"{positive_count_unchanged}"
)

print(
    f"Slope bounds valid                    : "
    f"{slope_bounds_valid}"
)

print(
    f"Terrain static across origin years    : "
    f"{static_terrain_consistent}"
)

print(
    f"NASADEM used as model predictor       : "
    f"False"
)

print(
    f"OSM 2026 used                         : "
    f"False"
)

print(
    f"Overall Step 12B status               : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(TERRAIN_GRID_CSV)}"
)

print(
    f" - {relative_path(COMBINED_PREDICTOR_CSV)}"
)

print(
    f" - {relative_path(FEATURE_LIST_CSV)}"
)

print(
    f" - {relative_path(TERRAIN_QA_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. GEDTM30 is the primary terrain source."
)

print(
    "2. NASADEM remains a sensitivity/reference DEM and is not duplicated as a model feature."
)

print(
    "3. Terrain predictors are static and do not introduce temporal leakage."
)

print(
    "4. No 2026 OSM roads or buildings were added."
)

print(
    "5. The target remains exactly the final Step 11F causal target."
)

print(
    "6. The next step will create causal spatial-context predictors before model training."
)

print("=" * 100)