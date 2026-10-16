"""
STEP 11B - AGGREGATE ANNUAL LULC TO 100 M MODELLING GRID

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Connect the permanent 100 m modelling grid with annual LULC maps.
2. Calculate exact area-weighted LULC composition inside every grid cell.
3. Process both RAW and STABILIZED annual LULC products.
4. Preserve KMC boundary overlap for partial boundary cells.
5. Produce one grid-level LULC table for each year from 2010-2026.
6. Keep RAW and STABILIZED information clearly separated.

Important:
- No urbanization-pressure target is created here.
- No predictor engineering is performed here.
- RAW annual LULC does not use future-year smoothing.
- STABILIZED LULC for 2011-2024 is future-informed because it uses
  a centered three-year temporal majority.
- Exact polygon/pixel overlap area is used instead of simple pixel counts.
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

RAW_LULC_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lulc"
    / "annual_maps"
    / "raw"
)

STABILIZED_LULC_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lulc"
    / "annual_maps"
    / "stabilized_mode_3yr"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "lulc_grid"
    / "annual_fractions"
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

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. OUTPUT REPORTS
# =============================================================================

YEAR_SUMMARY_CSV = (
    REPORT_DIR
    / "11b_lulc_100m_year_product_summary.csv"
)

RAW_STABILIZED_COMPARISON_CSV = (
    REPORT_DIR
    / "11b_raw_vs_stabilized_100m_comparison.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "11b_lulc_100m_aggregation_summary.csv"
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

EXPECTED_YEARS = list(
    range(
        2010,
        2027,
    )
)

CLASS_IDS = [
    1,
    2,
    3,
    4,
]

CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}

TARGET_CRS = "EPSG:32644"

EXPECTED_GRID_CELLS = 9240

AREA_TOLERANCE_M2 = 1e-5

FRACTION_TOLERANCE = 1e-8


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


def raw_lulc_path(year: int) -> Path:
    """Return RAW annual LULC map path."""

    return (
        RAW_LULC_DIR
        /
        f"Kaduwela_LULC_RAW_{year}.tif"
    )


def stabilized_lulc_path(year: int) -> Path:
    """Return STABILIZED annual LULC map path."""

    return (
        STABILIZED_LULC_DIR
        /
        f"Kaduwela_LULC_STABILIZED_{year}.tif"
    )


def output_year_csv(year: int) -> Path:
    """Return yearly 100 m aggregation table path."""

    return (
        OUTPUT_DIR
        /
        f"kaduwela_100m_lulc_fractions_{year}.csv"
    )


def future_informed_status(
    product_type: str,
    year: int,
) -> bool:
    """
    Centered stabilization uses year+1 information for 2011-2024.

    RAW maps do not use centered future-year smoothing.
    """

    return bool(
        product_type
        ==
        "STABILIZED_ANALYTICAL"
        and
        2011
        <=
        year
        <=
        2024
    )


def candidate_pixel_range(
    geometry,
    transform,
    width,
    height,
):
    """
    Return candidate raster row/column range overlapping a geometry.

    Assumes a north-up raster with no rotation.
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
    """Return bounds of one north-up raster pixel."""

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


    x_min = min(
        left,
        right,
    )

    x_max = max(
        left,
        right,
    )

    y_min = min(
        bottom,
        top,
    )

    y_max = max(
        bottom,
        top,
    )


    return (
        x_min,
        y_min,
        x_max,
        y_max,
    )


def rectangle_overlap_area(
    first_bounds,
    second_bounds,
):
    """
    Exact overlap area between two axis-aligned rectangles.
    """

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


def aggregate_one_grid_cell(
    class_array,
    rows,
    cols,
    overlap_areas,
    kmc_overlap_area_m2,
):
    """
    Calculate exact area-weighted class composition.

    Fractions are calculated using valid classified LULC area.
    Raster NoData class 0 is excluded.

    Coverage fraction records how much of the KMC portion of the
    100 m grid cell has valid LULC classification.
    """

    class_values = (
        class_array[
            rows,
            cols,
        ]
    )


    valid_class_mask = np.isin(
        class_values,
        CLASS_IDS,
    )


    valid_lulc_area_m2 = float(
        overlap_areas[
            valid_class_mask
        ].sum()
    )


    class_areas = {}


    for class_id in CLASS_IDS:

        class_areas[
            class_id
        ] = float(
            overlap_areas[
                class_values
                ==
                class_id
            ].sum()
        )


    if valid_lulc_area_m2 > 0:

        class_fractions = {
            class_id:
            (
                class_areas[
                    class_id
                ]
                /
                valid_lulc_area_m2
            )
            for class_id in CLASS_IDS
        }


        dominant_class_id = max(
            CLASS_IDS,
            key=lambda class_id:
            class_areas[
                class_id
            ],
        )


        dominant_class_name = (
            CLASS_DEFINITIONS[
                dominant_class_id
            ]
        )


        dominant_fraction = float(
            class_fractions[
                dominant_class_id
            ]
        )

    else:

        class_fractions = {
            class_id: np.nan
            for class_id in CLASS_IDS
        }


        dominant_class_id = (
            np.nan
        )

        dominant_class_name = (
            "NO_VALID_LULC"
        )

        dominant_fraction = (
            np.nan
        )


    valid_lulc_coverage_fraction = (
        valid_lulc_area_m2
        /
        kmc_overlap_area_m2
        if kmc_overlap_area_m2
        >
        0
        else
        np.nan
    )


    return {
        "valid_lulc_area_m2": (
            valid_lulc_area_m2
        ),
        "valid_lulc_coverage_fraction": (
            valid_lulc_coverage_fraction
        ),
        "green_area_m2": (
            class_areas[
                1
            ]
        ),
        "built_area_m2": (
            class_areas[
                2
            ]
        ),
        "open_bare_area_m2": (
            class_areas[
                3
            ]
        ),
        "water_area_m2": (
            class_areas[
                4
            ]
        ),
        "green_fraction": (
            class_fractions[
                1
            ]
        ),
        "built_fraction": (
            class_fractions[
                2
            ]
        ),
        "open_bare_fraction": (
            class_fractions[
                3
            ]
        ),
        "water_fraction": (
            class_fractions[
                4
            ]
        ),
        "dominant_class_id": (
            dominant_class_id
        ),
        "dominant_class_name": (
            dominant_class_name
        ),
        "dominant_class_fraction": (
            dominant_fraction
        ),
    }


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 11B - AGGREGATE ANNUAL LULC TO 100 M MODELLING GRID")
print("=" * 100)


for required_file in [
    GRID_FILE,
    KMC_BOUNDARY_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


for year in EXPECTED_YEARS:

    for required_raster in [
        raw_lulc_path(
            year
        ),
        stabilized_lulc_path(
            year
        ),
    ]:

        if not required_raster.exists():

            raise FileNotFoundError(
                f"Required annual LULC raster not found:\n"
                f"{required_raster}"
            )


# =============================================================================
# 6. LOAD GRID
# =============================================================================

grid_gdf = gpd.read_file(
    GRID_FILE
)


if grid_gdf.crs is None:

    raise RuntimeError(
        "100 m modelling grid has no CRS."
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
        "Duplicate grid IDs detected."
    )


print(
    f"\n100 m modelling cells : "
    f"{len(grid_gdf)}"
)


# =============================================================================
# 7. LOAD KMC BOUNDARY
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


kmc_area_m2 = float(
    kmc_geometry.area
)


print(
    f"KMC area             : "
    f"{kmc_area_m2 / 1_000_000:.6f} km²"
)


# =============================================================================
# 8. LOAD REFERENCE RASTER GRID
# =============================================================================

reference_raster = raw_lulc_path(
    2021
)


with rasterio.open(
    reference_raster
) as src:

    reference_crs = (
        src.crs
    )

    reference_transform = (
        src.transform
    )

    reference_width = (
        src.width
    )

    reference_height = (
        src.height
    )

    reference_nodata = (
        src.nodata
    )


if str(
    reference_crs
).upper() != TARGET_CRS:

    raise RuntimeError(
        f"Unexpected LULC CRS: "
        f"{reference_crs}"
    )


if (
    abs(
        reference_transform.b
    )
    >
    1e-12
    or
    abs(
        reference_transform.d
    )
    >
    1e-12
):

    raise RuntimeError(
        "Rotated raster grids are not supported by this aggregation step."
    )


print(
    f"Reference raster grid : "
    f"{reference_width} x {reference_height}"
)

print(
    f"Reference NoData      : "
    f"{reference_nodata}"
)


# =============================================================================
# 9. PRECOMPUTE EXACT RASTER PIXEL OVERLAPS FOR EACH 100 M CELL
# =============================================================================

print("\n" + "=" * 100)
print("PRECOMPUTING EXACT GRID / RASTER PIXEL OVERLAPS")
print("=" * 100)


pixel_contributions = {}

geometry_reconstruction_errors = []


for counter, (
    dataframe_index,
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


    full_cell_geometry = (
        grid_row.geometry
    )


    boundary_cell = bool(
        grid_row[
            "boundary_cell"
        ]
    )


    expected_kmc_overlap_area_m2 = float(
        grid_row[
            "kmc_overlap_area_m2"
        ]
    )


    if boundary_cell:

        analysis_geometry = (
            full_cell_geometry.intersection(
                kmc_geometry
            )
        )

    else:

        analysis_geometry = (
            full_cell_geometry
        )


    if analysis_geometry.is_empty:

        raise RuntimeError(
            f"Empty KMC/grid intersection for {grid_id}."
        )


    (
        row_start,
        row_stop,
        col_start,
        col_stop,
    ) = candidate_pixel_range(
        analysis_geometry,
        reference_transform,
        reference_width,
        reference_height,
    )


    contribution_rows = []

    contribution_cols = []

    contribution_areas = []


    cell_bounds = (
        full_cell_geometry.bounds
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
                reference_transform,
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


            contribution_rows.append(
                raster_row
            )

            contribution_cols.append(
                raster_col
            )

            contribution_areas.append(
                overlap_area_m2
            )


    if len(
        contribution_areas
    ) == 0:

        raise RuntimeError(
            f"No raster pixel overlap found for {grid_id}."
        )


    rows_array = np.asarray(
        contribution_rows,
        dtype=np.int32,
    )


    cols_array = np.asarray(
        contribution_cols,
        dtype=np.int32,
    )


    areas_array = np.asarray(
        contribution_areas,
        dtype=np.float64,
    )


    reconstructed_area_m2 = float(
        areas_array.sum()
    )


    reconstruction_error_m2 = (
        reconstructed_area_m2
        -
        expected_kmc_overlap_area_m2
    )


    geometry_reconstruction_errors.append(
        abs(
            reconstruction_error_m2
        )
    )


    pixel_contributions[
        grid_id
    ] = {
        "rows": (
            rows_array
        ),
        "cols": (
            cols_array
        ),
        "areas_m2": (
            areas_array
        ),
    }


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
            f"Prepared "
            f"{counter}/{len(grid_gdf)} "
            f"grid cells"
        )


maximum_geometry_reconstruction_error_m2 = float(
    max(
        geometry_reconstruction_errors
    )
)


geometry_reconstruction_valid = bool(
    maximum_geometry_reconstruction_error_m2
    <=
    AREA_TOLERANCE_M2
)


print(
    f"\nMaximum grid-area reconstruction error : "
    f"{maximum_geometry_reconstruction_error_m2:.10f} m²"
)

print(
    f"Geometry reconstruction valid          : "
    f"{geometry_reconstruction_valid}"
)


if not geometry_reconstruction_valid:

    raise RuntimeError(
        "Raster-pixel overlap geometry does not reconstruct "
        "the 100 m grid/KMC intersection accurately."
    )


# =============================================================================
# 10. PROCESS EACH YEAR
# =============================================================================

year_summary_records = []

raw_stabilized_comparison_records = []

total_output_rows = 0

all_fraction_sum_errors = []

all_coverage_values = []

overall_duplicate_rows = 0


print("\n" + "=" * 100)
print("ANNUAL 100 M LULC AGGREGATION")
print("=" * 100)


for year in EXPECTED_YEARS:

    print(
        f"\nProcessing {year}..."
    )


    raw_path = raw_lulc_path(
        year
    )

    stabilized_path = (
        stabilized_lulc_path(
            year
        )
    )


    with rasterio.open(
        raw_path
    ) as raw_src:

        if (
            raw_src.crs
            !=
            reference_crs
            or
            raw_src.width
            !=
            reference_width
            or
            raw_src.height
            !=
            reference_height
            or
            not raw_src.transform.almost_equals(
                reference_transform
            )
        ):

            raise RuntimeError(
                f"RAW raster grid mismatch for {year}."
            )


        raw_array = (
            raw_src.read(
                1
            )
            .astype(
                np.uint8
            )
        )


    with rasterio.open(
        stabilized_path
    ) as stabilized_src:

        if (
            stabilized_src.crs
            !=
            reference_crs
            or
            stabilized_src.width
            !=
            reference_width
            or
            stabilized_src.height
            !=
            reference_height
            or
            not stabilized_src.transform.almost_equals(
                reference_transform
            )
        ):

            raise RuntimeError(
                f"STABILIZED raster grid mismatch for {year}."
            )


        stabilized_array = (
            stabilized_src.read(
                1
            )
            .astype(
                np.uint8
            )
        )


    for (
        product_type,
        class_array,
    ) in [
        (
            "RAW_RF",
            raw_array,
        ),
        (
            "STABILIZED_ANALYTICAL",
            stabilized_array,
        ),
    ]:

        invalid_values = (
            set(
                int(
                    value
                )
                for value in np.unique(
                    class_array
                )
            )
            -
            {
                0,
                1,
                2,
                3,
                4,
            }
        )


        if invalid_values:

            raise RuntimeError(
                f"Invalid class values in "
                f"{product_type} {year}: "
                f"{sorted(invalid_values)}"
            )


    yearly_records = []


    # Keep these for RAW-vs-STABILIZED comparison.
    yearly_product_frames = {}


    for (
        product_type,
        class_array,
    ) in [
        (
            "RAW_RF",
            raw_array,
        ),
        (
            "STABILIZED_ANALYTICAL",
            stabilized_array,
        ),
    ]:

        product_records = []


        for _, grid_row in (
            grid_gdf.iterrows()
        ):

            grid_id = (
                grid_row[
                    "grid_id"
                ]
            )


            contribution = (
                pixel_contributions[
                    grid_id
                ]
            )


            kmc_overlap_area_m2 = float(
                grid_row[
                    "kmc_overlap_area_m2"
                ]
            )


            aggregated = (
                aggregate_one_grid_cell(
                    class_array=class_array,
                    rows=contribution[
                        "rows"
                    ],
                    cols=contribution[
                        "cols"
                    ],
                    overlap_areas=contribution[
                        "areas_m2"
                    ],
                    kmc_overlap_area_m2=(
                        kmc_overlap_area_m2
                    ),
                )
            )


            record = {
                "grid_id": (
                    grid_id
                ),
                "grid_row": (
                    int(
                        grid_row[
                            "grid_row"
                        ]
                    )
                ),
                "grid_col": (
                    int(
                        grid_row[
                            "grid_col"
                        ]
                    )
                ),
                "year": (
                    year
                ),
                "product_type": (
                    product_type
                ),
                "future_informed": (
                    future_informed_status(
                        product_type,
                        year,
                    )
                ),
                "provisional_year": (
                    year
                    ==
                    2026
                ),
                "boundary_cell": (
                    bool(
                        grid_row[
                            "boundary_cell"
                        ]
                    )
                ),
                "kmc_overlap_area_m2": (
                    kmc_overlap_area_m2
                ),
                "kmc_overlap_fraction": (
                    float(
                        grid_row[
                            "kmc_overlap_fraction"
                        ]
                    )
                ),
            }


            record.update(
                aggregated
            )


            product_records.append(
                record
            )


        product_df = pd.DataFrame(
            product_records
        )


        expected_rows = len(
            grid_gdf
        )


        if len(
            product_df
        ) != expected_rows:

            raise RuntimeError(
                f"{product_type} {year}: expected "
                f"{expected_rows} rows, found "
                f"{len(product_df)}."
            )


        duplicate_rows = int(
            product_df[
                "grid_id"
            ]
            .duplicated()
            .sum()
        )


        overall_duplicate_rows += (
            duplicate_rows
        )


        valid_fraction_rows = (
            product_df[
                "valid_lulc_area_m2"
            ]
            >
            0
        )


        fraction_sum = (
            product_df.loc[
                valid_fraction_rows,
                [
                    "green_fraction",
                    "built_fraction",
                    "open_bare_fraction",
                    "water_fraction",
                ],
            ]
            .sum(
                axis=1
            )
        )


        maximum_fraction_sum_error = (
            float(
                np.max(
                    np.abs(
                        fraction_sum
                        -
                        1.0
                    )
                )
            )
            if len(
                fraction_sum
            )
            >
            0
            else
            np.nan
        )


        if np.isfinite(
            maximum_fraction_sum_error
        ):

            all_fraction_sum_errors.append(
                maximum_fraction_sum_error
            )


        coverage_values = (
            product_df[
                "valid_lulc_coverage_fraction"
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )


        if len(
            coverage_values
        ) > 0:

            all_coverage_values.extend(
                coverage_values.tolist()
            )


        invalid_coverage_count = int(
            (
                (
                    product_df[
                        "valid_lulc_coverage_fraction"
                    ]
                    <
                    -
                    FRACTION_TOLERANCE
                )
                |
                (
                    product_df[
                        "valid_lulc_coverage_fraction"
                    ]
                    >
                    1.0
                    +
                    FRACTION_TOLERANCE
                )
            )
            .fillna(
                False
            )
            .sum()
        )


        if invalid_coverage_count > 0:

            raise RuntimeError(
                f"{product_type} {year}: "
                f"{invalid_coverage_count} invalid "
                f"coverage fractions."
            )


        zero_coverage_cells = int(
            (
                product_df[
                    "valid_lulc_area_m2"
                ]
                <=
                0
            ).sum()
        )


        total_valid_lulc_area_m2 = float(
            product_df[
                "valid_lulc_area_m2"
            ].sum()
        )


        total_kmc_overlap_area_m2 = float(
            product_df[
                "kmc_overlap_area_m2"
            ].sum()
        )


        weighted_coverage_percent = (
            total_valid_lulc_area_m2
            /
            total_kmc_overlap_area_m2
            *
            100
        )


        year_summary_records.append(
            {
                "year": (
                    year
                ),
                "product_type": (
                    product_type
                ),
                "grid_rows": (
                    len(
                        product_df
                    )
                ),
                "future_informed": (
                    future_informed_status(
                        product_type,
                        year,
                    )
                ),
                "provisional_year": (
                    year
                    ==
                    2026
                ),
                "zero_valid_lulc_cells": (
                    zero_coverage_cells
                ),
                "cells_coverage_ge_90_percent": int(
                    (
                        product_df[
                            "valid_lulc_coverage_fraction"
                        ]
                        >=
                        0.90
                    ).sum()
                ),
                "cells_coverage_ge_75_percent": int(
                    (
                        product_df[
                            "valid_lulc_coverage_fraction"
                        ]
                        >=
                        0.75
                    ).sum()
                ),
                "mean_cell_coverage_fraction": (
                    float(
                        product_df[
                            "valid_lulc_coverage_fraction"
                        ].mean()
                    )
                ),
                "median_cell_coverage_fraction": (
                    float(
                        product_df[
                            "valid_lulc_coverage_fraction"
                        ].median()
                    )
                ),
                "weighted_kmc_coverage_percent": (
                    weighted_coverage_percent
                ),
                "maximum_class_fraction_sum_error": (
                    maximum_fraction_sum_error
                ),
                "duplicate_grid_ids": (
                    duplicate_rows
                ),
            }
        )


        yearly_records.extend(
            product_records
        )


        yearly_product_frames[
            product_type
        ] = (
            product_df
        )


    # =========================================================================
    # RAW VS STABILIZED GRID-LEVEL COMPARISON
    # =========================================================================

    raw_df = (
        yearly_product_frames[
            "RAW_RF"
        ][
            [
                "grid_id",
                "green_fraction",
                "built_fraction",
                "open_bare_fraction",
                "water_fraction",
            ]
        ]
        .copy()
    )


    stabilized_df = (
        yearly_product_frames[
            "STABILIZED_ANALYTICAL"
        ][
            [
                "grid_id",
                "green_fraction",
                "built_fraction",
                "open_bare_fraction",
                "water_fraction",
            ]
        ]
        .copy()
    )


    comparison_df = raw_df.merge(
        stabilized_df,
        on="grid_id",
        how="inner",
        suffixes=(
            "_raw",
            "_stabilized",
        ),
        validate="one_to_one",
    )


    valid_comparison = (
        comparison_df[
            "built_fraction_raw"
        ].notna()
        &
        comparison_df[
            "built_fraction_stabilized"
        ].notna()
    )


    built_difference = np.abs(
        comparison_df.loc[
            valid_comparison,
            "built_fraction_raw",
        ]
        -
        comparison_df.loc[
            valid_comparison,
            "built_fraction_stabilized",
        ]
    )


    green_difference = np.abs(
        comparison_df.loc[
            valid_comparison,
            "green_fraction_raw",
        ]
        -
        comparison_df.loc[
            valid_comparison,
            "green_fraction_stabilized",
        ]
    )


    changed_cells = int(
        (
            built_difference
            >
            FRACTION_TOLERANCE
        ).sum()
    )


    raw_stabilized_comparison_records.append(
        {
            "year": (
                year
            ),
            "future_informed_stabilized_product": (
                2011
                <=
                year
                <=
                2024
            ),
            "comparable_grid_cells": (
                int(
                    valid_comparison.sum()
                )
            ),
            "cells_with_built_fraction_change": (
                changed_cells
            ),
            "percent_cells_with_built_fraction_change": (
                changed_cells
                /
                int(
                    valid_comparison.sum()
                )
                *
                100
                if int(
                    valid_comparison.sum()
                )
                >
                0
                else
                np.nan
            ),
            "mean_absolute_built_fraction_difference": (
                float(
                    built_difference.mean()
                )
                if len(
                    built_difference
                )
                >
                0
                else
                np.nan
            ),
            "maximum_absolute_built_fraction_difference": (
                float(
                    built_difference.max()
                )
                if len(
                    built_difference
                )
                >
                0
                else
                np.nan
            ),
            "mean_absolute_green_fraction_difference": (
                float(
                    green_difference.mean()
                )
                if len(
                    green_difference
                )
                >
                0
                else
                np.nan
            ),
            "maximum_absolute_green_fraction_difference": (
                float(
                    green_difference.max()
                )
                if len(
                    green_difference
                )
                >
                0
                else
                np.nan
            ),
        }
    )


    # =========================================================================
    # SAVE YEARLY TABLE
    # =========================================================================

    yearly_df = pd.DataFrame(
        yearly_records
    )


    duplicate_year_product_grid = int(
        yearly_df.duplicated(
            subset=[
                "grid_id",
                "year",
                "product_type",
            ]
        ).sum()
    )


    if duplicate_year_product_grid > 0:

        raise RuntimeError(
            f"{year}: duplicate "
            f"(grid_id, year, product_type) rows found."
        )


    output_csv = output_year_csv(
        year
    )


    yearly_df.to_csv(
        output_csv,
        index=False,
        float_format="%.10f",
    )


    total_output_rows += len(
        yearly_df
    )


    print(
        f"  Saved {len(yearly_df):,} rows"
        f" -> {relative_path(output_csv)}"
    )


# =============================================================================
# 11. BUILD REPORT TABLES
# =============================================================================

year_summary_df = pd.DataFrame(
    year_summary_records
)


raw_stabilized_comparison_df = pd.DataFrame(
    raw_stabilized_comparison_records
)


maximum_fraction_sum_error_all = (
    float(
        max(
            all_fraction_sum_errors
        )
    )
    if len(
        all_fraction_sum_errors
    )
    >
    0
    else
    np.nan
)


minimum_coverage_fraction_all = (
    float(
        min(
            all_coverage_values
        )
    )
    if len(
        all_coverage_values
    )
    >
    0
    else
    np.nan
)


maximum_coverage_fraction_all = (
    float(
        max(
            all_coverage_values
        )
    )
    if len(
        all_coverage_values
    )
    >
    0
    else
    np.nan
)


expected_total_rows = (
    EXPECTED_GRID_CELLS
    *
    len(
        EXPECTED_YEARS
    )
    *
    2
)


yearly_files_generated = len(
    list(
        OUTPUT_DIR.glob(
            "kaduwela_100m_lulc_fractions_*.csv"
        )
    )
)


edge_years_identical = bool(
    (
        raw_stabilized_comparison_df.loc[
            raw_stabilized_comparison_df[
                "year"
            ].isin(
                [
                    2010,
                    2025,
                    2026,
                ]
            ),
            "cells_with_built_fraction_change",
        ]
        ==
        0
    ).all()
)


interior_stabilized_years_marked_future_informed = bool(
    year_summary_df.loc[
        (
            year_summary_df[
                "product_type"
            ]
            ==
            "STABILIZED_ANALYTICAL"
        )
        &
        year_summary_df[
            "year"
        ].between(
            2011,
            2024,
        ),
        "future_informed",
    ].all()
)


raw_years_not_future_informed = bool(
    (
        ~year_summary_df.loc[
            year_summary_df[
                "product_type"
            ]
            ==
            "RAW_RF",
            "future_informed",
        ]
    ).all()
)


class_fraction_sums_valid = bool(
    (
        maximum_fraction_sum_error_all
        <=
        FRACTION_TOLERANCE
    )
)


coverage_values_valid = bool(
    (
        minimum_coverage_fraction_all
        >=
        -
        FRACTION_TOLERANCE
    )
    and
    (
        maximum_coverage_fraction_all
        <=
        1.0
        +
        FRACTION_TOLERANCE
    )
)


row_count_valid = (
    total_output_rows
    ==
    expected_total_rows
)


year_file_count_valid = (
    yearly_files_generated
    ==
    len(
        EXPECTED_YEARS
    )
)


overall_status = (
    "PASS"
    if all(
        [
            geometry_reconstruction_valid,
            class_fraction_sums_valid,
            coverage_values_valid,
            row_count_valid,
            year_file_count_valid,
            overall_duplicate_rows
            ==
            0,
            edge_years_identical,
            interior_stabilized_years_marked_future_informed,
            raw_years_not_future_informed,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 12. FINAL SUMMARY
# =============================================================================

final_summary_df = pd.DataFrame(
    [
        {
            "grid_cells": (
                EXPECTED_GRID_CELLS
            ),
            "years": (
                len(
                    EXPECTED_YEARS
                )
            ),
            "products_per_year": (
                2
            ),
            "expected_output_rows": (
                expected_total_rows
            ),
            "actual_output_rows": (
                total_output_rows
            ),
            "yearly_csv_files": (
                yearly_files_generated
            ),
            "maximum_geometry_reconstruction_error_m2": (
                maximum_geometry_reconstruction_error_m2
            ),
            "maximum_class_fraction_sum_error": (
                maximum_fraction_sum_error_all
            ),
            "minimum_valid_lulc_coverage_fraction": (
                minimum_coverage_fraction_all
            ),
            "maximum_valid_lulc_coverage_fraction": (
                maximum_coverage_fraction_all
            ),
            "duplicate_grid_rows": (
                overall_duplicate_rows
            ),
            "raw_and_stabilized_identical_2010_2025_2026": (
                edge_years_identical
            ),
            "stabilized_2011_2024_marked_future_informed": (
                interior_stabilized_years_marked_future_informed
            ),
            "raw_product_marked_not_future_informed": (
                raw_years_not_future_informed
            ),
            "urbanization_target_created": (
                False
            ),
            "predictors_created": (
                False
            ),
            "overall_step11b_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 13. SAVE REPORTS
# =============================================================================

year_summary_df.to_csv(
    YEAR_SUMMARY_CSV,
    index=False,
)


raw_stabilized_comparison_df.to_csv(
    RAW_STABILIZED_COMPARISON_CSV,
    index=False,
)


final_summary_df.to_csv(
    FINAL_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 14. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("YEAR / PRODUCT SUMMARY")
print("=" * 100)


print(
    year_summary_df[
        [
            "year",
            "product_type",
            "future_informed",
            "provisional_year",
            "zero_valid_lulc_cells",
            "cells_coverage_ge_90_percent",
            "weighted_kmc_coverage_percent",
            "maximum_class_fraction_sum_error",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("RAW VS STABILIZED 100 M COMPARISON")
print("=" * 100)


print(
    raw_stabilized_comparison_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 11B QA")
print("=" * 100)


print(
    f"Expected output rows                   : "
    f"{expected_total_rows:,}"
)

print(
    f"Actual output rows                     : "
    f"{total_output_rows:,}"
)

print(
    f"Yearly CSV files                       : "
    f"{yearly_files_generated}"
)

print(
    f"Maximum geometry reconstruction error  : "
    f"{maximum_geometry_reconstruction_error_m2:.10f} m²"
)

print(
    f"Maximum class-fraction sum error        : "
    f"{maximum_fraction_sum_error_all:.12f}"
)

print(
    f"Minimum LULC coverage fraction          : "
    f"{minimum_coverage_fraction_all:.6f}"
)

print(
    f"Maximum LULC coverage fraction          : "
    f"{maximum_coverage_fraction_all:.6f}"
)

print(
    f"Duplicate grid rows                     : "
    f"{overall_duplicate_rows}"
)

print(
    f"2010/2025/2026 RAW=STABILIZED           : "
    f"{edge_years_identical}"
)

print(
    f"STABILIZED 2011-2024 future-informed    : "
    f"{interior_stabilized_years_marked_future_informed}"
)

print(
    f"RAW product future-informed             : "
    f"{not raw_years_not_future_informed}"
)

print(
    f"Urbanization target created             : "
    f"False"
)

print(
    f"Predictors created                      : "
    f"False"
)

print(
    f"Overall Step 11B status                 : "
    f"{overall_status}"
)


print("\nGenerated annual grid tables:")

print(
    f" - {relative_path(OUTPUT_DIR)}"
)

print(
    "   kaduwela_100m_lulc_fractions_2010.csv"
)

print(
    "   ..."
)

print(
    "   kaduwela_100m_lulc_fractions_2026.csv"
)


print("\nGenerated reports:")

print(
    f" - {relative_path(YEAR_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(RAW_STABILIZED_COMPARISON_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. Each 100 m cell now has area-weighted annual LULC fractions."
)

print(
    "2. RAW and STABILIZED products remain separate."
)

print(
    "3. STABILIZED 2011-2024 is explicitly marked as future-informed."
)

print(
    "4. RAW LULC will be important when constructing leakage-safe ML history."
)

print(
    "5. Low-coverage boundary cells are not deleted in this step."
)

print(
    "6. A minimum grid-coverage rule will be selected before ML target construction."
)

print(
    "7. No urbanization-pressure target has been created yet."
)

print("=" * 100)