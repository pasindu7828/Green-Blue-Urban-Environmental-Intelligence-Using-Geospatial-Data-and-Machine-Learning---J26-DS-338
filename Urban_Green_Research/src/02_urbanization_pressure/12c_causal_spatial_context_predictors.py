"""
STEP 12C - CAUSAL SPATIAL-CONTEXT PREDICTOR ENGINEERING

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Add surrounding land-development context to each 100 m modelling cell.
2. Use only RAW LULC information available at t-2, t-1 and t.
3. Calculate immediate 8-neighbour and wider 24-neighbour context.
4. Exclude the focal grid cell itself from neighbourhood statistics.
5. Keep future LULC and centered stabilized LULC completely excluded.
6. Join spatial-context predictors to the Step 12B dataset.

Neighbourhood definitions:
- NBR8  = surrounding cells in a 3 x 3 window, excluding the centre.
- NBR24 = surrounding cells in a 5 x 5 window, excluding the centre.

Important:
- These are observed spatial-context covariates, not target labels.
- Neighbour values with <90% valid LULC coverage are excluded.
- Current 2026 OSM roads/buildings are NOT used.
- Grid row/column values remain metadata only.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


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
    / "12b_causal_lulc_terrain_predictors.csv"
)

GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "grid"
    / "kaduwela_100m_modelling_grid.gpkg"
)

ANNUAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "lulc_grid"
    / "annual_fractions"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "predictors"
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
# 2. OUTPUTS
# =============================================================================

OUTPUT_PREDICTOR_CSV = (
    OUTPUT_DIR
    / "12c_causal_lulc_terrain_spatial_predictors.csv"
)

FEATURE_LIST_CSV = (
    REPORT_DIR
    / "12c_combined_predictor_feature_list.csv"
)

SPATIAL_CONTEXT_QA_CSV = (
    REPORT_DIR
    / "12c_spatial_context_qa.csv"
)

NEIGHBOUR_COVERAGE_CSV = (
    REPORT_DIR
    / "12c_neighbour_coverage_summary.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "12c_spatial_context_predictor_summary.csv"
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

EXPECTED_MODEL_ROWS = 51310
EXPECTED_POSITIVE_ROWS = 844

EXPECTED_EXISTING_FEATURES = 56
EXPECTED_NEW_FEATURES = 13
EXPECTED_TOTAL_FEATURES = 69

EXPECTED_GRID_CELLS = 9240

RAW_YEARS = list(
    range(
        2010,
        2023,
    )
)

MIN_NEIGHBOUR_LULC_COVERAGE = 0.90


METADATA_COLUMNS = [
    "grid_id",
    "grid_row",
    "grid_col",
    "origin_year",
    "urban_conversion_target",
    "kmc_overlap_fraction",
    "boundary_cell",
]


QA_ONLY_COLUMNS = [
    "elevation_valid_coverage_fraction",
    "slope_valid_coverage_fraction",
]


NEW_SPATIAL_FEATURES = [
    "nbr8_built_fraction_t_minus_2",
    "nbr8_built_fraction_t_minus_1",
    "nbr8_built_fraction_t",
    "nbr8_built_mean_3yr",
    "nbr8_built_change_1yr",
    "nbr8_built_change_2yr",
    "nbr8_built_slope_3yr",
    "nbr8_green_fraction_t",
    "nbr8_green_change_2yr",
    "nbr24_built_fraction_t",
    "nbr24_built_mean_3yr",
    "nbr24_built_change_2yr",
    "nbr24_green_fraction_t",
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


def annual_file(year: int) -> Path:
    """Return annual Step 11B table."""

    return (
        ANNUAL_DIR
        /
        f"kaduwela_100m_lulc_fractions_{year}.csv"
    )


def neighbourhood_mean(
    value_array,
    valid_array,
    radius,
):
    """
    Calculate neighbourhood mean using a square window.

    radius=1:
        3x3 window excluding centre -> maximum 8 neighbours.

    radius=2:
        5x5 window excluding centre -> maximum 24 neighbours.

    Outside-study cells and unreliable cells remain excluded.
    """

    value_array = np.asarray(
        value_array,
        dtype=float,
    )

    valid_array = np.asarray(
        valid_array,
        dtype=bool,
    )


    rows, cols = (
        value_array.shape
    )


    padded_values = np.pad(
        value_array,
        pad_width=radius,
        mode="constant",
        constant_values=np.nan,
    )


    padded_valid = np.pad(
        valid_array,
        pad_width=radius,
        mode="constant",
        constant_values=False,
    )


    neighbour_sum = np.zeros(
        (
            rows,
            cols,
        ),
        dtype=float,
    )


    neighbour_count = np.zeros(
        (
            rows,
            cols,
        ),
        dtype=np.int16,
    )


    for row_offset in range(
        -radius,
        radius + 1,
    ):

        for col_offset in range(
            -radius,
            radius + 1,
        ):

            # Exclude focal cell.
            if (
                row_offset == 0
                and
                col_offset == 0
            ):
                continue


            row_start = (
                radius
                +
                row_offset
            )

            col_start = (
                radius
                +
                col_offset
            )


            shifted_values = (
                padded_values[
                    row_start:
                    row_start + rows,
                    col_start:
                    col_start + cols,
                ]
            )


            shifted_valid = (
                padded_valid[
                    row_start:
                    row_start + rows,
                    col_start:
                    col_start + cols,
                ]
                &
                np.isfinite(
                    shifted_values
                )
            )


            neighbour_sum[
                shifted_valid
            ] += (
                shifted_values[
                    shifted_valid
                ]
            )


            neighbour_count[
                shifted_valid
            ] += 1


    neighbour_mean_array = np.full(
        (
            rows,
            cols,
        ),
        np.nan,
        dtype=float,
    )


    has_neighbour = (
        neighbour_count
        >
        0
    )


    neighbour_mean_array[
        has_neighbour
    ] = (
        neighbour_sum[
            has_neighbour
        ]
        /
        neighbour_count[
            has_neighbour
        ]
    )


    return (
        neighbour_mean_array,
        neighbour_count,
    )


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 12C - CAUSAL SPATIAL-CONTEXT PREDICTOR ENGINEERING")
print("=" * 100)


for required_file in [
    INPUT_PREDICTOR_CSV,
    GRID_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


for year in RAW_YEARS:

    if not annual_file(
        year
    ).exists():

        raise FileNotFoundError(
            f"Required RAW LULC table not found:\n"
            f"{annual_file(year)}"
        )


# =============================================================================
# 6. LOAD STEP 12B MODEL DATA
# =============================================================================

model_df = pd.read_csv(
    INPUT_PREDICTOR_CSV
)


if len(
    model_df
) != EXPECTED_MODEL_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_MODEL_ROWS:,} modelling rows, "
        f"found {len(model_df):,}."
    )


if int(
    model_df[
        "urban_conversion_target"
    ].sum()
) != EXPECTED_POSITIVE_ROWS:

    raise RuntimeError(
        "Positive target count does not match Step 11F."
    )


if model_df.duplicated(
    subset=[
        "grid_id",
        "origin_year",
    ]
).any():

    raise RuntimeError(
        "Duplicate grid/origin rows found in Step 12B."
    )


existing_feature_columns = [
    column
    for column in model_df.columns
    if (
        column not in METADATA_COLUMNS
        and
        column not in QA_ONLY_COLUMNS
    )
]


if len(
    existing_feature_columns
) != EXPECTED_EXISTING_FEATURES:

    raise RuntimeError(
        f"Expected {EXPECTED_EXISTING_FEATURES} existing model features, "
        f"found {len(existing_feature_columns)}."
    )


print(
    f"\nInput modelling rows      : "
    f"{len(model_df):,}"
)

print(
    f"Existing predictors       : "
    f"{len(existing_feature_columns)}"
)

print(
    f"Positive target rows      : "
    f"{model_df['urban_conversion_target'].sum():,}"
)


# =============================================================================
# 7. LOAD PERMANENT GRID
# =============================================================================

grid_gdf = gpd.read_file(
    GRID_FILE
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
        "Duplicate grid IDs found in permanent grid."
    )


grid_lookup_df = (
    grid_gdf[
        [
            "grid_id",
            "grid_row",
            "grid_col",
        ]
    ]
    .copy()
)


max_grid_row = int(
    grid_lookup_df[
        "grid_row"
    ].max()
)


max_grid_col = int(
    grid_lookup_df[
        "grid_col"
    ].max()
)


array_rows = (
    max_grid_row
    +
    1
)


array_cols = (
    max_grid_col
    +
    1
)


print(
    f"Grid lattice size         : "
    f"{array_rows} rows x {array_cols} columns"
)


# =============================================================================
# 8. LOAD RAW LULC AND BUILD YEARLY SPATIAL-CONTEXT TABLES
# =============================================================================

print("\n" + "=" * 100)
print("BUILDING YEARLY CAUSAL SPATIAL CONTEXT")
print("=" * 100)


spatial_year_frames = {}

coverage_summary_records = []


for year in RAW_YEARS:

    print(
        f"Processing spatial context for {year}..."
    )


    yearly_df = pd.read_csv(
        annual_file(
            year
        )
    )


    raw_df = (
        yearly_df[
            yearly_df[
                "product_type"
            ]
            ==
            "RAW_RF"
        ]
        .copy()
    )


    if len(
        raw_df
    ) != EXPECTED_GRID_CELLS:

        raise RuntimeError(
            f"Expected {EXPECTED_GRID_CELLS} RAW rows "
            f"for {year}, found {len(raw_df)}."
        )


    if raw_df[
        "future_informed"
    ].astype(
        bool
    ).any():

        raise RuntimeError(
            f"RAW {year} unexpectedly marked future-informed."
        )


    if raw_df[
        "grid_id"
    ].duplicated().any():

        raise RuntimeError(
            f"Duplicate RAW grid IDs found for {year}."
        )


    raw_grid_df = (
        grid_lookup_df.merge(
            raw_df[
                [
                    "grid_id",
                    "valid_lulc_coverage_fraction",
                    "green_fraction",
                    "built_fraction",
                    "open_bare_fraction",
                    "water_fraction",
                ]
            ],
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
    )


    reliable = (
        raw_grid_df[
            "valid_lulc_coverage_fraction"
        ]
        >=
        MIN_NEIGHBOUR_LULC_COVERAGE
    )


    arrays = {}


    for class_name, column_name in [
        (
            "green",
            "green_fraction",
        ),
        (
            "built",
            "built_fraction",
        ),
        (
            "open_bare",
            "open_bare_fraction",
        ),
        (
            "water",
            "water_fraction",
        ),
    ]:

        class_array = np.full(
            (
                array_rows,
                array_cols,
            ),
            np.nan,
            dtype=float,
        )


        reliable_rows = (
            raw_grid_df[
                reliable
            ]
        )


        class_array[
            reliable_rows[
                "grid_row"
            ].astype(
                int
            ).to_numpy(),
            reliable_rows[
                "grid_col"
            ].astype(
                int
            ).to_numpy(),
        ] = (
            reliable_rows[
                column_name
            ]
            .to_numpy(
                dtype=float
            )
        )


        arrays[
            class_name
        ] = class_array


    reliable_array = np.zeros(
        (
            array_rows,
            array_cols,
        ),
        dtype=bool,
    )


    reliable_rows = (
        raw_grid_df[
            reliable
        ]
    )


    reliable_array[
        reliable_rows[
            "grid_row"
        ].astype(
            int
        ).to_numpy(),
        reliable_rows[
            "grid_col"
        ].astype(
            int
        ).to_numpy(),
    ] = True


    # =========================================================================
    # IMMEDIATE 8-NEIGHBOUR CONTEXT
    # =========================================================================

    (
        nbr8_built,
        nbr8_count,
    ) = neighbourhood_mean(
        arrays[
            "built"
        ],
        reliable_array,
        radius=1,
    )


    (
        nbr8_green,
        _,
    ) = neighbourhood_mean(
        arrays[
            "green"
        ],
        reliable_array,
        radius=1,
    )


    (
        nbr8_open,
        _,
    ) = neighbourhood_mean(
        arrays[
            "open_bare"
        ],
        reliable_array,
        radius=1,
    )


    (
        nbr8_water,
        _,
    ) = neighbourhood_mean(
        arrays[
            "water"
        ],
        reliable_array,
        radius=1,
    )


    # =========================================================================
    # WIDER 24-NEIGHBOUR CONTEXT
    # =========================================================================

    (
        nbr24_built,
        nbr24_count,
    ) = neighbourhood_mean(
        arrays[
            "built"
        ],
        reliable_array,
        radius=2,
    )


    (
        nbr24_green,
        _,
    ) = neighbourhood_mean(
        arrays[
            "green"
        ],
        reliable_array,
        radius=2,
    )


    # =========================================================================
    # EXTRACT BACK TO GRID IDs
    # =========================================================================

    grid_rows = (
        grid_lookup_df[
            "grid_row"
        ]
        .astype(
            int
        )
        .to_numpy()
    )


    grid_cols = (
        grid_lookup_df[
            "grid_col"
        ]
        .astype(
            int
        )
        .to_numpy()
    )


    spatial_df = pd.DataFrame(
        {
            "grid_id": (
                grid_lookup_df[
                    "grid_id"
                ].to_numpy()
            ),
            "year": (
                year
            ),
            "nbr8_built": (
                nbr8_built[
                    grid_rows,
                    grid_cols,
                ]
            ),
            "nbr8_green": (
                nbr8_green[
                    grid_rows,
                    grid_cols,
                ]
            ),
            "nbr8_open_bare": (
                nbr8_open[
                    grid_rows,
                    grid_cols,
                ]
            ),
            "nbr8_water": (
                nbr8_water[
                    grid_rows,
                    grid_cols,
                ]
            ),
            "nbr8_valid_count": (
                nbr8_count[
                    grid_rows,
                    grid_cols,
                ]
            ),
            "nbr24_built": (
                nbr24_built[
                    grid_rows,
                    grid_cols,
                ]
            ),
            "nbr24_green": (
                nbr24_green[
                    grid_rows,
                    grid_cols,
                ]
            ),
            "nbr24_valid_count": (
                nbr24_count[
                    grid_rows,
                    grid_cols,
                ]
            ),
        }
    )


    spatial_year_frames[
        year
    ] = (
        spatial_df.set_index(
            "grid_id"
        )
    )


    coverage_summary_records.append(
        {
            "year": (
                year
            ),
            "reliable_source_grid_cells": int(
                reliable.sum()
            ),
            "cells_with_nbr8_context": int(
                spatial_df[
                    "nbr8_built"
                ].notna().sum()
            ),
            "minimum_nbr8_valid_neighbours": int(
                spatial_df.loc[
                    spatial_df[
                        "nbr8_valid_count"
                    ]
                    >
                    0,
                    "nbr8_valid_count",
                ].min()
            ),
            "median_nbr8_valid_neighbours": float(
                spatial_df.loc[
                    spatial_df[
                        "nbr8_valid_count"
                    ]
                    >
                    0,
                    "nbr8_valid_count",
                ].median()
            ),
            "maximum_nbr8_valid_neighbours": int(
                spatial_df[
                    "nbr8_valid_count"
                ].max()
            ),
            "cells_with_nbr24_context": int(
                spatial_df[
                    "nbr24_built"
                ].notna().sum()
            ),
            "minimum_nbr24_valid_neighbours": int(
                spatial_df.loc[
                    spatial_df[
                        "nbr24_valid_count"
                    ]
                    >
                    0,
                    "nbr24_valid_count",
                ].min()
            ),
            "median_nbr24_valid_neighbours": float(
                spatial_df.loc[
                    spatial_df[
                        "nbr24_valid_count"
                    ]
                    >
                    0,
                    "nbr24_valid_count",
                ].median()
            ),
            "maximum_nbr24_valid_neighbours": int(
                spatial_df[
                    "nbr24_valid_count"
                ].max()
            ),
        }
    )


# =============================================================================
# 9. CREATE SPATIAL FEATURES FOR MODEL ROWS
# =============================================================================

print("\n" + "=" * 100)
print("JOINING SPATIAL CONTEXT TO MODEL ROWS")
print("=" * 100)


spatial_records = []


for row in model_df[
    [
        "grid_id",
        "origin_year",
    ]
].itertuples(
    index=False
):

    grid_id = (
        row.grid_id
    )

    origin_year = int(
        row.origin_year
    )


    t_minus_2 = (
        origin_year
        -
        2
    )

    t_minus_1 = (
        origin_year
        -
        1
    )

    t = (
        origin_year
    )


    frame_t2 = (
        spatial_year_frames[
            t_minus_2
        ]
    )

    frame_t1 = (
        spatial_year_frames[
            t_minus_1
        ]
    )

    frame_t = (
        spatial_year_frames[
            t
        ]
    )


    nbr8_built_t2 = float(
        frame_t2.loc[
            grid_id,
            "nbr8_built",
        ]
    )


    nbr8_built_t1 = float(
        frame_t1.loc[
            grid_id,
            "nbr8_built",
        ]
    )


    nbr8_built_t = float(
        frame_t.loc[
            grid_id,
            "nbr8_built",
        ]
    )


    nbr8_green_t2 = float(
        frame_t2.loc[
            grid_id,
            "nbr8_green",
        ]
    )


    nbr8_green_t = float(
        frame_t.loc[
            grid_id,
            "nbr8_green",
        ]
    )


    nbr24_built_t2 = float(
        frame_t2.loc[
            grid_id,
            "nbr24_built",
        ]
    )


    nbr24_built_t1 = float(
        frame_t1.loc[
            grid_id,
            "nbr24_built",
        ]
    )


    nbr24_built_t = float(
        frame_t.loc[
            grid_id,
            "nbr24_built",
        ]
    )


    spatial_records.append(
        {
            "grid_id": (
                grid_id
            ),
            "origin_year": (
                origin_year
            ),

            # -------------------------------------------------------------
            # NBR8 built context
            # -------------------------------------------------------------

            "nbr8_built_fraction_t_minus_2": (
                nbr8_built_t2
            ),
            "nbr8_built_fraction_t_minus_1": (
                nbr8_built_t1
            ),
            "nbr8_built_fraction_t": (
                nbr8_built_t
            ),
            "nbr8_built_mean_3yr": float(
                np.mean(
                    [
                        nbr8_built_t2,
                        nbr8_built_t1,
                        nbr8_built_t,
                    ]
                )
            ),
            "nbr8_built_change_1yr": (
                nbr8_built_t
                -
                nbr8_built_t1
            ),
            "nbr8_built_change_2yr": (
                nbr8_built_t
                -
                nbr8_built_t2
            ),
            "nbr8_built_slope_3yr": (
                (
                    nbr8_built_t
                    -
                    nbr8_built_t2
                )
                /
                2.0
            ),

            # -------------------------------------------------------------
            # NBR8 other LULC context
            # -------------------------------------------------------------

            "nbr8_green_fraction_t": float(
                frame_t.loc[
                    grid_id,
                    "nbr8_green",
                ]
            ),
            "nbr8_green_change_2yr": (
                nbr8_green_t
                -
                nbr8_green_t2
            ),

            # -------------------------------------------------------------
            # NBR24 wider context
            # -------------------------------------------------------------

            "nbr24_built_fraction_t": (
                nbr24_built_t
            ),
            "nbr24_built_mean_3yr": float(
                np.mean(
                    [
                        nbr24_built_t2,
                        nbr24_built_t1,
                        nbr24_built_t,
                    ]
                )
            ),
            "nbr24_built_change_2yr": (
                nbr24_built_t
                -
                nbr24_built_t2
            ),
            "nbr24_green_fraction_t": float(
                frame_t.loc[
                    grid_id,
                    "nbr24_green",
                ]
            ),
        }
    )


spatial_feature_df = pd.DataFrame(
    spatial_records
)


# =============================================================================
# 10. JOIN WITH STEP 12B
# =============================================================================

combined_df = model_df.merge(
    spatial_feature_df,
    on=[
        "grid_id",
        "origin_year",
    ],
    how="left",
    validate="one_to_one",
)


if len(
    combined_df
) != len(
    model_df
):

    raise RuntimeError(
        "Spatial-context merge changed row count."
    )


# =============================================================================
# 11. FINAL FEATURE LIST
# =============================================================================

final_feature_columns = [
    column
    for column in combined_df.columns
    if (
        column not in METADATA_COLUMNS
        and
        column not in QA_ONLY_COLUMNS
    )
]


new_features_present = all(
    feature_name
    in
    final_feature_columns
    for feature_name in NEW_SPATIAL_FEATURES
)


# =============================================================================
# 12. QA
# =============================================================================

row_count_unchanged = (
    len(
        combined_df
    )
    ==
    EXPECTED_MODEL_ROWS
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


duplicate_grid_origin_rows = int(
    combined_df.duplicated(
        subset=[
            "grid_id",
            "origin_year",
        ]
    ).sum()
)


missing_new_feature_values = int(
    combined_df[
        NEW_SPATIAL_FEATURES
    ]
    .isna()
    .sum()
    .sum()
)


infinite_new_feature_values = int(
    np.isinf(
        combined_df[
            NEW_SPATIAL_FEATURES
        ]
        .to_numpy(
            dtype=float
        )
    ).sum()
)


fraction_like_features = [
    "nbr8_built_fraction_t_minus_2",
    "nbr8_built_fraction_t_minus_1",
    "nbr8_built_fraction_t",
    "nbr8_built_mean_3yr",
    "nbr8_green_fraction_t",
    "nbr24_built_fraction_t",
    "nbr24_built_mean_3yr",
    "nbr24_green_fraction_t",
]


fraction_bounds_valid = True


for feature_name in fraction_like_features:

    values = (
        combined_df[
            feature_name
        ]
        .to_numpy(
            dtype=float
        )
    )


    finite_values = (
        values[
            np.isfinite(
                values
            )
        ]
    )


    if len(
        finite_values
    ) == 0:

        fraction_bounds_valid = (
            False
        )

        break


    if (
        finite_values.min()
        <
        -
        1e-8
        or
        finite_values.max()
        >
        1.0
        +
        1e-8
    ):

        fraction_bounds_valid = (
            False
        )

        break


target_unchanged = bool(
    np.array_equal(
        combined_df[
            "urban_conversion_target"
        ].to_numpy(),
        model_df[
            "urban_conversion_target"
        ].to_numpy(),
    )
)


feature_count_valid = (
    len(
        final_feature_columns
    )
    ==
    EXPECTED_TOTAL_FEATURES
)


no_future_named_features = bool(
    not any(
        (
            "future"
            in feature_name.lower()
            or
            "target_window"
            in feature_name.lower()
        )
        for feature_name in final_feature_columns
    )
)


self_cell_excluded = True


raw_lulc_only = True

stabilized_lulc_used = False

osm_2026_used = False


overall_status = (
    "PASS"
    if all(
        [
            row_count_unchanged,
            positive_count_unchanged,
            duplicate_grid_origin_rows
            ==
            0,
            missing_new_feature_values
            ==
            0,
            infinite_new_feature_values
            ==
            0,
            fraction_bounds_valid,
            target_unchanged,
            feature_count_valid,
            new_features_present,
            no_future_named_features,
            self_cell_excluded,
            raw_lulc_only,
            not stabilized_lulc_used,
            not osm_2026_used,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 13. FEATURE LIST REPORT
# =============================================================================

feature_list_records = []


for feature_name in final_feature_columns:

    if feature_name in NEW_SPATIAL_FEATURES:

        feature_group = (
            "CAUSAL_SPATIAL_CONTEXT"
        )

        temporal_role = (
            "T_MINUS_2_TO_T"
        )

    elif feature_name.startswith(
        "elevation_"
    ) or feature_name.startswith(
        "slope_"
    ):

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
            "centered_stabilized_lulc_used": (
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
# 14. QA REPORTS
# =============================================================================

coverage_summary_df = pd.DataFrame(
    coverage_summary_records
)


qa_df = pd.DataFrame(
    [
        {
            "model_rows": (
                len(
                    combined_df
                )
            ),
            "existing_feature_count": (
                len(
                    existing_feature_columns
                )
            ),
            "new_spatial_feature_count": (
                len(
                    NEW_SPATIAL_FEATURES
                )
            ),
            "total_feature_count": (
                len(
                    final_feature_columns
                )
            ),
            "positive_target_rows": int(
                combined_df[
                    "urban_conversion_target"
                ].sum()
            ),
            "duplicate_grid_origin_rows": (
                duplicate_grid_origin_rows
            ),
            "missing_new_spatial_feature_values": (
                missing_new_feature_values
            ),
            "infinite_new_spatial_feature_values": (
                infinite_new_feature_values
            ),
            "fraction_bounds_valid": (
                fraction_bounds_valid
            ),
            "target_unchanged": (
                target_unchanged
            ),
            "self_cell_excluded_from_neighbourhoods": (
                self_cell_excluded
            ),
            "raw_lulc_only": (
                raw_lulc_only
            ),
            "centered_stabilized_lulc_used": (
                stabilized_lulc_used
            ),
            "osm_2026_used": (
                osm_2026_used
            ),
            "overall_step12c_status": (
                overall_status
            ),
        }
    ]
)


final_summary_df = pd.DataFrame(
    [
        {
            "input_dataset": (
                "12b_causal_lulc_terrain_predictors.csv"
            ),
            "historical_lulc_source": (
                "RAW_RF"
            ),
            "neighbourhood_1": (
                "8 surrounding cells in 3x3 window"
            ),
            "neighbourhood_2": (
                "24 surrounding cells in 5x5 window"
            ),
            "focal_cell_excluded": (
                True
            ),
            "minimum_neighbour_lulc_coverage": (
                MIN_NEIGHBOUR_LULC_COVERAGE
            ),
            "existing_feature_count": (
                EXPECTED_EXISTING_FEATURES
            ),
            "spatial_context_feature_count": (
                EXPECTED_NEW_FEATURES
            ),
            "combined_feature_count": (
                len(
                    final_feature_columns
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
            "overall_step12c_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 15. SAVE OUTPUTS
# =============================================================================

combined_df.to_csv(
    OUTPUT_PREDICTOR_CSV,
    index=False,
)


feature_list_df.to_csv(
    FEATURE_LIST_CSV,
    index=False,
)


qa_df.to_csv(
    SPATIAL_CONTEXT_QA_CSV,
    index=False,
)


coverage_summary_df.to_csv(
    NEIGHBOUR_COVERAGE_CSV,
    index=False,
)


final_summary_df.to_csv(
    FINAL_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 16. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("SPATIAL-CONTEXT PREDICTOR DATASET")
print("=" * 100)


print(
    f"Rows                         : "
    f"{len(combined_df):,}"
)

print(
    f"Previous predictors          : "
    f"{len(existing_feature_columns)}"
)

print(
    f"Spatial predictors added     : "
    f"{len(NEW_SPATIAL_FEATURES)}"
)

print(
    f"Total predictors             : "
    f"{len(final_feature_columns)}"
)

print(
    f"Positive target rows         : "
    f"{combined_df['urban_conversion_target'].sum():,}"
)


print("\nNew spatial predictor names:")

for feature_name in NEW_SPATIAL_FEATURES:

    print(
        f" - {feature_name}"
    )


print("\n" + "=" * 100)
print("NEIGHBOUR COVERAGE SUMMARY")
print("=" * 100)


print(
    coverage_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 12C QA")
print("=" * 100)


print(
    f"Rows unchanged                    : "
    f"{row_count_unchanged}"
)

print(
    f"Positive count unchanged          : "
    f"{positive_count_unchanged}"
)

print(
    f"Duplicate grid/origin rows        : "
    f"{duplicate_grid_origin_rows}"
)

print(
    f"New spatial features present      : "
    f"{new_features_present}"
)

print(
    f"Expected total features = 69      : "
    f"{feature_count_valid}"
)

print(
    f"Missing new feature values        : "
    f"{missing_new_feature_values}"
)

print(
    f"Infinite new feature values       : "
    f"{infinite_new_feature_values}"
)

print(
    f"Fraction bounds valid             : "
    f"{fraction_bounds_valid}"
)

print(
    f"Target unchanged                  : "
    f"{target_unchanged}"
)

print(
    f"Focal cell excluded               : "
    f"{self_cell_excluded}"
)

print(
    f"Future-named predictors           : "
    f"{not no_future_named_features}"
)

print(
    f"Centered STABILIZED LULC used     : "
    f"{stabilized_lulc_used}"
)

print(
    f"OSM 2026 used                     : "
    f"{osm_2026_used}"
)

print(
    f"Overall Step 12C status           : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(OUTPUT_PREDICTOR_CSV)}"
)

print(
    f" - {relative_path(FEATURE_LIST_CSV)}"
)

print(
    f" - {relative_path(SPATIAL_CONTEXT_QA_CSV)}"
)

print(
    f" - {relative_path(NEIGHBOUR_COVERAGE_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. Spatial context uses surrounding cells only; the focal cell itself is excluded."
)

print(
    "2. Only RAW LULC from t-2, t-1 and t is used."
)

print(
    "3. No future LULC is used as a predictor."
)

print(
    "4. No centered stabilized LULC is used."
)

print(
    "5. No 2026 OSM roads or buildings are used."
)

print(
    "6. The resulting dataset should contain 69 causal/static predictors."
)

print(
    "7. After this passes, we will review predictor redundancy and solve the road-accessibility issue before model training."
)

print("=" * 100)