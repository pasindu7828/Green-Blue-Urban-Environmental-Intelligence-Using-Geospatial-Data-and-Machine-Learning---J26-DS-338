"""
STEP 14B - CREATE 2025 PRODUCTION INFERENCE FEATURE TABLE

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Create the final production inference feature table for origin year 2025.
2. Use only RAW LULC from 2023, 2024 and 2025.
3. Apply the same predictor-side eligibility domain used during training.
4. Add the same static terrain predictors used by the production model.
5. Add the same causal spatial-context predictors.
6. Select exactly the frozen 55 predictors from Step 12E.
7. Do NOT use provisional 2026 data.

Inference meaning:
- Origin year = 2025
- Predictors = information available through 2025
- Pressure horizon = persistent majority-Built conversion during 2026-2028

Eligibility:
- KMC overlap >= 50%
- RAW LULC coverage >= 90% in 2023, 2024 and 2025
- 2025 Built-up <= 0.33
- 2025 Green + Open/Bare >= 0.50
- At least 2 of 2023, 2024 and 2025 satisfy the same non-built condition

IMPORTANT:
- 2026 is NOT loaded.
- STABILIZED LULC is NOT used.
- No future information is used.
- OSM 2026 is NOT used.
- The model is NOT scored in this step.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNUAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "lulc_grid"
    / "annual_fractions"
)

GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "grid"
    / "kaduwela_100m_modelling_grid.gpkg"
)

TERRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "terrain"
    / "kaduwela_100m_static_terrain_predictors.csv"
)

FEATURE_LIST_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "12e_final_predictor_feature_list.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "inference"
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

INFERENCE_FEATURES_CSV = (
    OUTPUT_DIR
    / "14b_2025_inference_features.csv"
)

ELIGIBILITY_CSV = (
    OUTPUT_DIR
    / "14b_2025_inference_eligibility.csv"
)

FEATURE_QA_CSV = (
    REPORT_DIR
    / "14b_2025_inference_feature_qa.csv"
)

ELIGIBILITY_SUMMARY_CSV = (
    REPORT_DIR
    / "14b_2025_inference_eligibility_summary.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "14b_2025_inference_summary.csv"
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

ORIGIN_YEAR = 2025

PAST_YEARS = [
    2023,
    2024,
    2025,
]

PRESSURE_PERIOD = "2026-2028"

EXPECTED_GRID_CELLS = 9240

EXPECTED_FEATURES = 55

MIN_KMC_OVERLAP = 0.50

MIN_LULC_COVERAGE = 0.90

CURRENT_MAX_BUILT = 0.33

CURRENT_MIN_GREEN_PLUS_OPEN = 0.50

PAST_REQUIRED_NONBUILT_VOTES = 2

MIN_NEIGHBOUR_LULC_COVERAGE = 0.90


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
    """Return one Step 11B annual LULC fraction table."""

    return (
        ANNUAL_DIR
        /
        f"kaduwela_100m_lulc_fractions_{year}.csv"
    )


def create_matrix(
    long_df,
    column_name,
):
    """Create grid_id x year matrix."""

    return (
        long_df.pivot(
            index="grid_id",
            columns="year",
            values=column_name,
        )
        .sort_index()
    )


def neighbourhood_mean(
    value_array,
    valid_array,
    radius,
):
    """
    Calculate square-window neighbourhood mean.

    radius=1:
        3x3 window excluding focal cell -> maximum 8 neighbours.

    radius=2:
        5x5 window excluding focal cell -> maximum 24 neighbours.
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
print("STEP 14B - CREATE 2025 PRODUCTION INFERENCE FEATURE TABLE")
print("=" * 100)


required_files = [
    GRID_FILE,
    TERRAIN_FILE,
    FEATURE_LIST_FILE,
]


for year in PAST_YEARS:

    required_files.append(
        annual_file(
            year
        )
    )


for required_file in required_files:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


print(
    f"\nInference origin year : "
    f"{ORIGIN_YEAR}"
)

print(
    f"RAW LULC years        : "
    f"{PAST_YEARS}"
)

print(
    f"Pressure period       : "
    f"{PRESSURE_PERIOD}"
)

print(
    "2026 used             : False"
)

print(
    "STABILIZED LULC used  : False"
)

print(
    "OSM 2026 used         : False"
)


# =============================================================================
# 6. LOAD PERMANENT GRID
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
        "Duplicate grid IDs detected."
    )


grid_df = (
    grid_gdf[
        [
            "grid_id",
            "grid_row",
            "grid_col",
            "boundary_cell",
            "kmc_overlap_area_m2",
            "kmc_overlap_fraction",
        ]
    ]
    .copy()
)


grid_df[
    "grid_row"
] = (
    grid_df[
        "grid_row"
    ]
    .astype(
        int
    )
)


grid_df[
    "grid_col"
] = (
    grid_df[
        "grid_col"
    ]
    .astype(
        int
    )
)


# =============================================================================
# 7. LOAD 2023-2025 RAW LULC ONLY
# =============================================================================

raw_frames = []


for year in PAST_YEARS:

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
        "grid_id"
    ].duplicated().any():

        raise RuntimeError(
            f"Duplicate RAW grid IDs in {year}."
        )


    if raw_df[
        "future_informed"
    ].astype(
        bool
    ).any():

        raise RuntimeError(
            f"RAW {year} unexpectedly marked future-informed."
        )


    raw_frames.append(
        raw_df
    )


raw_long_df = pd.concat(
    raw_frames,
    ignore_index=True,
)


if set(
    raw_long_df[
        "year"
    ].unique()
) != set(
    PAST_YEARS
):

    raise RuntimeError(
        "Unexpected LULC year set."
    )


# =============================================================================
# 8. CREATE GRID x YEAR MATRICES
# =============================================================================

green_wide = create_matrix(
    raw_long_df,
    "green_fraction",
)

built_wide = create_matrix(
    raw_long_df,
    "built_fraction",
)

open_wide = create_matrix(
    raw_long_df,
    "open_bare_fraction",
)

water_wide = create_matrix(
    raw_long_df,
    "water_fraction",
)

coverage_wide = create_matrix(
    raw_long_df,
    "valid_lulc_coverage_fraction",
)


green_open_wide = (
    green_wide
    +
    open_wide
)


# =============================================================================
# 9. CREATE 2025 PREDICTOR-SIDE ELIGIBILITY
# =============================================================================

static_grid = (
    grid_df.set_index(
        "grid_id"
    )
    .loc[
        built_wide.index
    ]
)


spatial_ok = (
    static_grid[
        "kmc_overlap_fraction"
    ]
    >=
    MIN_KMC_OVERLAP
)


coverage_ok = (
    coverage_wide[
        PAST_YEARS
    ]
    >=
    MIN_LULC_COVERAGE
).all(
    axis=1
)


current_nonbuilt_ok = (
    (
        built_wide[
            ORIGIN_YEAR
        ]
        <=
        CURRENT_MAX_BUILT
    )
    &
    (
        green_open_wide[
            ORIGIN_YEAR
        ]
        >=
        CURRENT_MIN_GREEN_PLUS_OPEN
    )
)


past_nonbuilt_condition = (
    (
        built_wide[
            PAST_YEARS
        ]
        <=
        CURRENT_MAX_BUILT
    )
    &
    (
        green_open_wide[
            PAST_YEARS
        ]
        >=
        CURRENT_MIN_GREEN_PLUS_OPEN
    )
)


past_nonbuilt_votes = (
    past_nonbuilt_condition.sum(
        axis=1
    )
)


past_stable_nonbuilt = (
    past_nonbuilt_votes
    >=
    PAST_REQUIRED_NONBUILT_VOTES
)


initial_eligible = (
    spatial_ok
    &
    coverage_ok
    &
    current_nonbuilt_ok
    &
    past_stable_nonbuilt
)


eligibility_df = pd.DataFrame(
    {
        "grid_id": (
            built_wide.index
        ),
        "origin_year": (
            ORIGIN_YEAR
        ),
        "pressure_period": (
            PRESSURE_PERIOD
        ),
        "kmc_overlap_fraction": (
            static_grid[
                "kmc_overlap_fraction"
            ].to_numpy()
        ),
        "minimum_2023_2025_lulc_coverage": (
            coverage_wide[
                PAST_YEARS
            ]
            .min(
                axis=1
            )
            .to_numpy()
        ),
        "built_fraction_2025": (
            built_wide[
                ORIGIN_YEAR
            ]
            .to_numpy()
        ),
        "green_plus_open_fraction_2025": (
            green_open_wide[
                ORIGIN_YEAR
            ]
            .to_numpy()
        ),
        "past_nonbuilt_votes_2023_2025": (
            past_nonbuilt_votes.to_numpy()
        ),
        "spatial_overlap_eligible": (
            spatial_ok.to_numpy()
        ),
        "coverage_eligible": (
            coverage_ok.to_numpy()
        ),
        "current_nonbuilt_eligible": (
            current_nonbuilt_ok.to_numpy()
        ),
        "past_stability_eligible": (
            past_stable_nonbuilt.to_numpy()
        ),
        "initial_inference_eligible": (
            initial_eligible.to_numpy()
        ),
    }
)


print("\n" + "=" * 100)
print("2025 PREDICTOR-SIDE ELIGIBILITY")
print("=" * 100)


print(
    f"All permanent grid cells          : "
    f"{EXPECTED_GRID_CELLS:,}"
)

print(
    f"KMC overlap >= 50%                : "
    f"{int(spatial_ok.sum()):,}"
)

print(
    f"2023-2025 coverage >= 90%         : "
    f"{int(coverage_ok.sum()):,}"
)

print(
    f"2025 current non-built condition  : "
    f"{int(current_nonbuilt_ok.sum()):,}"
)

print(
    f"Past 2-of-3 stability             : "
    f"{int(past_stable_nonbuilt.sum()):,}"
)

print(
    f"Initial inference-eligible cells  : "
    f"{int(initial_eligible.sum()):,}"
)


# =============================================================================
# 10. ENGINEER CELL-LEVEL CAUSAL LULC FEATURES
# =============================================================================

class_matrices = {
    "green": (
        green_wide
    ),
    "built": (
        built_wide
    ),
    "open_bare": (
        open_wide
    ),
    "water": (
        water_wide
    ),
}


lulc_feature_records = []


for grid_id in built_wide.index[
    initial_eligible
]:

    record = {
        "grid_id": (
            grid_id
        ),
        "origin_year": (
            ORIGIN_YEAR
        ),
        "pressure_period": (
            PRESSURE_PERIOD
        ),
    }


    for class_name, matrix in class_matrices.items():

        t_minus_2 = float(
            matrix.loc[
                grid_id,
                2023,
            ]
        )

        t_minus_1 = float(
            matrix.loc[
                grid_id,
                2024,
            ]
        )

        current = float(
            matrix.loc[
                grid_id,
                2025,
            ]
        )


        values = np.array(
            [
                t_minus_2,
                t_minus_1,
                current,
            ],
            dtype=float,
        )


        record[
            f"{class_name}_fraction_t_minus_2"
        ] = (
            t_minus_2
        )

        record[
            f"{class_name}_fraction_t_minus_1"
        ] = (
            t_minus_1
        )

        record[
            f"{class_name}_fraction_t"
        ] = (
            current
        )

        record[
            f"{class_name}_mean_3yr"
        ] = float(
            np.mean(
                values
            )
        )

        record[
            f"{class_name}_min_3yr"
        ] = float(
            np.min(
                values
            )
        )

        record[
            f"{class_name}_max_3yr"
        ] = float(
            np.max(
                values
            )
        )

        record[
            f"{class_name}_change_1yr"
        ] = (
            current
            -
            t_minus_1
        )

        record[
            f"{class_name}_change_2yr"
        ] = (
            current
            -
            t_minus_2
        )

        # Kept temporarily for reconstruction consistency.
        # It will not enter the final frozen feature list if Step 12E removed it.
        record[
            f"{class_name}_slope_3yr"
        ] = (
            (
                current
                -
                t_minus_2
            )
            /
            2.0
        )

        record[
            f"{class_name}_range_3yr"
        ] = float(
            np.max(
                values
            )
            -
            np.min(
                values
            )
        )


    # -------------------------------------------------------------------------
    # Cross-class derived features from Step 12A
    # -------------------------------------------------------------------------

    green_t = (
        record[
            "green_fraction_t"
        ]
    )

    built_t = (
        record[
            "built_fraction_t"
        ]
    )

    open_t = (
        record[
            "open_bare_fraction_t"
        ]
    )

    water_t = (
        record[
            "water_fraction_t"
        ]
    )

    green_t_minus_1 = (
        record[
            "green_fraction_t_minus_1"
        ]
    )

    green_t_minus_2 = (
        record[
            "green_fraction_t_minus_2"
        ]
    )

    built_t_minus_1 = (
        record[
            "built_fraction_t_minus_1"
        ]
    )

    built_t_minus_2 = (
        record[
            "built_fraction_t_minus_2"
        ]
    )


    record[
        "terrestrial_nonbuilt_fraction_t"
    ] = (
        green_t
        +
        open_t
    )

    record[
        "nonwater_fraction_t"
    ] = (
        1.0
        -
        water_t
    )

    record[
        "built_minus_green_t"
    ] = (
        built_t
        -
        green_t
    )

    record[
        "built_green_ratio_t"
    ] = (
        built_t
        /
        (
            green_t
            +
            1e-6
        )
    )

    record[
        "built_growth_vs_green_loss_1yr"
    ] = (
        (
            built_t
            -
            built_t_minus_1
        )
        -
        (
            green_t
            -
            green_t_minus_1
        )
    )

    record[
        "built_growth_vs_green_loss_2yr"
    ] = (
        (
            built_t
            -
            built_t_minus_2
        )
        -
        (
            green_t
            -
            green_t_minus_2
        )
    )


    # QA only, not final model predictors.
    record[
        "origin_lulc_coverage"
    ] = float(
        coverage_wide.loc[
            grid_id,
            ORIGIN_YEAR,
        ]
    )

    record[
        "minimum_past_3yr_lulc_coverage"
    ] = float(
        coverage_wide.loc[
            grid_id,
            PAST_YEARS,
        ].min()
    )


    lulc_feature_records.append(
        record
    )


lulc_feature_df = pd.DataFrame(
    lulc_feature_records
)


# =============================================================================
# 11. BUILD 2023-2025 SPATIAL CONTEXT
# =============================================================================

print("\n" + "=" * 100)
print("BUILDING 2023-2025 SPATIAL CONTEXT")
print("=" * 100)


max_grid_row = int(
    grid_df[
        "grid_row"
    ].max()
)

max_grid_col = int(
    grid_df[
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


spatial_year_frames = {}


for year in PAST_YEARS:

    year_raw = (
        raw_long_df[
            raw_long_df[
                "year"
            ]
            ==
            year
        ][
            [
                "grid_id",
                "valid_lulc_coverage_fraction",
                "green_fraction",
                "built_fraction",
            ]
        ]
        .copy()
    )


    year_grid = grid_df[
        [
            "grid_id",
            "grid_row",
            "grid_col",
        ]
    ].merge(
        year_raw,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )


    reliable = (
        year_grid[
            "valid_lulc_coverage_fraction"
        ]
        >=
        MIN_NEIGHBOUR_LULC_COVERAGE
    )


    reliable_rows = (
        year_grid[
            reliable
        ]
    )


    reliable_array = np.zeros(
        (
            array_rows,
            array_cols,
        ),
        dtype=bool,
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


    built_array = np.full(
        (
            array_rows,
            array_cols,
        ),
        np.nan,
        dtype=float,
    )


    green_array = np.full(
        (
            array_rows,
            array_cols,
        ),
        np.nan,
        dtype=float,
    )


    built_array[
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
            "built_fraction"
        ]
        .to_numpy(
            dtype=float
        )
    )


    green_array[
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
            "green_fraction"
        ]
        .to_numpy(
            dtype=float
        )
    )


    (
        nbr8_built,
        nbr8_count,
    ) = neighbourhood_mean(
        built_array,
        reliable_array,
        radius=1,
    )


    (
        nbr8_green,
        _,
    ) = neighbourhood_mean(
        green_array,
        reliable_array,
        radius=1,
    )


    (
        nbr24_built,
        nbr24_count,
    ) = neighbourhood_mean(
        built_array,
        reliable_array,
        radius=2,
    )


    (
        nbr24_green,
        _,
    ) = neighbourhood_mean(
        green_array,
        reliable_array,
        radius=2,
    )


    grid_rows = (
        grid_df[
            "grid_row"
        ]
        .astype(
            int
        )
        .to_numpy()
    )


    grid_cols = (
        grid_df[
            "grid_col"
        ]
        .astype(
            int
        )
        .to_numpy()
    )


    spatial_year_frames[
        year
    ] = (
        pd.DataFrame(
            {
                "grid_id": (
                    grid_df[
                        "grid_id"
                    ].to_numpy()
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
        .set_index(
            "grid_id"
        )
    )


    print(
        f"{year}: reliable source cells="
        f"{int(reliable.sum()):,}, "
        f"cells with nbr8="
        f"{int(np.isfinite(nbr8_built[grid_rows, grid_cols]).sum()):,}, "
        f"cells with nbr24="
        f"{int(np.isfinite(nbr24_built[grid_rows, grid_cols]).sum()):,}"
    )


# =============================================================================
# 12. CREATE SPATIAL-CONTEXT FEATURES FOR ELIGIBLE CELLS
# =============================================================================

spatial_records = []


for grid_id in lulc_feature_df[
    "grid_id"
]:

    f2023 = (
        spatial_year_frames[
            2023
        ]
    )

    f2024 = (
        spatial_year_frames[
            2024
        ]
    )

    f2025 = (
        spatial_year_frames[
            2025
        ]
    )


    nbr8_built_2023 = float(
        f2023.loc[
            grid_id,
            "nbr8_built",
        ]
    )

    nbr8_built_2024 = float(
        f2024.loc[
            grid_id,
            "nbr8_built",
        ]
    )

    nbr8_built_2025 = float(
        f2025.loc[
            grid_id,
            "nbr8_built",
        ]
    )


    nbr8_green_2023 = float(
        f2023.loc[
            grid_id,
            "nbr8_green",
        ]
    )

    nbr8_green_2025 = float(
        f2025.loc[
            grid_id,
            "nbr8_green",
        ]
    )


    nbr24_built_2023 = float(
        f2023.loc[
            grid_id,
            "nbr24_built",
        ]
    )

    nbr24_built_2024 = float(
        f2024.loc[
            grid_id,
            "nbr24_built",
        ]
    )

    nbr24_built_2025 = float(
        f2025.loc[
            grid_id,
            "nbr24_built",
        ]
    )


    spatial_records.append(
        {
            "grid_id": (
                grid_id
            ),

            "nbr8_built_fraction_t_minus_2": (
                nbr8_built_2023
            ),

            "nbr8_built_fraction_t_minus_1": (
                nbr8_built_2024
            ),

            "nbr8_built_fraction_t": (
                nbr8_built_2025
            ),

            "nbr8_built_mean_3yr": float(
                np.mean(
                    [
                        nbr8_built_2023,
                        nbr8_built_2024,
                        nbr8_built_2025,
                    ]
                )
            ),

            "nbr8_built_change_1yr": (
                nbr8_built_2025
                -
                nbr8_built_2024
            ),

            "nbr8_built_change_2yr": (
                nbr8_built_2025
                -
                nbr8_built_2023
            ),

            "nbr8_built_slope_3yr": (
                (
                    nbr8_built_2025
                    -
                    nbr8_built_2023
                )
                /
                2.0
            ),

            "nbr8_green_fraction_t": (
                nbr8_green_2025
            ),

            "nbr8_green_change_2yr": (
                nbr8_green_2025
                -
                nbr8_green_2023
            ),

            "nbr24_built_fraction_t": (
                nbr24_built_2025
            ),

            "nbr24_built_mean_3yr": float(
                np.mean(
                    [
                        nbr24_built_2023,
                        nbr24_built_2024,
                        nbr24_built_2025,
                    ]
                )
            ),

            "nbr24_built_change_2yr": (
                nbr24_built_2025
                -
                nbr24_built_2023
            ),

            "nbr24_green_fraction_t": float(
                f2025.loc[
                    grid_id,
                    "nbr24_green",
                ]
            ),

            "nbr8_valid_count_2025": int(
                f2025.loc[
                    grid_id,
                    "nbr8_valid_count",
                ]
            ),

            "nbr24_valid_count_2025": int(
                f2025.loc[
                    grid_id,
                    "nbr24_valid_count",
                ]
            ),
        }
    )


spatial_feature_df = pd.DataFrame(
    spatial_records
)


# =============================================================================
# 13. LOAD STATIC TERRAIN
# =============================================================================

terrain_df = pd.read_csv(
    TERRAIN_FILE
)


if len(
    terrain_df
) != EXPECTED_GRID_CELLS:

    raise RuntimeError(
        f"Expected {EXPECTED_GRID_CELLS} terrain rows, "
        f"found {len(terrain_df)}."
    )


if terrain_df[
    "grid_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate terrain grid IDs detected."
    )


terrain_columns = [
    "grid_id",
    "elevation_mean_m",
    "elevation_std_m",
    "elevation_min_m",
    "elevation_max_m",
    "elevation_range_m",
    "slope_mean_deg",
    "slope_std_deg",
    "slope_max_deg",
    "elevation_valid_coverage_fraction",
    "slope_valid_coverage_fraction",
]


# =============================================================================
# 14. MERGE ALL 2025 PREDICTOR GROUPS
# =============================================================================

combined_df = (
    lulc_feature_df.merge(
        spatial_feature_df,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        terrain_df[
            terrain_columns
        ],
        on="grid_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        grid_df[
            [
                "grid_id",
                "grid_row",
                "grid_col",
                "boundary_cell",
                "kmc_overlap_fraction",
            ]
        ],
        on="grid_id",
        how="left",
        validate="one_to_one",
    )
)


# =============================================================================
# 15. LOAD FROZEN FINAL 55-FEATURE LIST
# =============================================================================

feature_list_df = pd.read_csv(
    FEATURE_LIST_FILE
)


if len(
    feature_list_df
) != EXPECTED_FEATURES:

    raise RuntimeError(
        f"Expected {EXPECTED_FEATURES} frozen features, "
        f"found {len(feature_list_df)}."
    )


final_feature_columns = (
    feature_list_df[
        "feature_name"
    ]
    .astype(
        str
    )
    .tolist()
)


missing_final_features = [
    feature_name
    for feature_name in final_feature_columns
    if feature_name not in combined_df.columns
]


if missing_final_features:

    raise RuntimeError(
        "The following frozen production features "
        "could not be reconstructed:\n"
        +
        "\n".join(
            missing_final_features
        )
    )


# =============================================================================
# 16. FINAL PREDICTOR COMPLETENESS
# =============================================================================

feature_matrix = (
    combined_df[
        final_feature_columns
    ]
    .astype(
        float
    )
)


row_has_missing_predictor = (
    feature_matrix.isna().any(
        axis=1
    )
)


row_has_infinite_predictor = (
    np.isinf(
        feature_matrix.to_numpy(
            dtype=float
        )
    ).any(
        axis=1
    )
)


predictor_complete = (
    ~row_has_missing_predictor
    &
    ~row_has_infinite_predictor
)


combined_df[
    "predictor_complete"
] = (
    predictor_complete
)


final_inference_df = (
    combined_df[
        combined_df[
            "predictor_complete"
        ]
    ]
    .copy()
)


# =============================================================================
# 17. FINAL OUTPUT COLUMN ORDER
# =============================================================================

metadata_columns = [
    "grid_id",
    "grid_row",
    "grid_col",
    "origin_year",
    "pressure_period",
    "boundary_cell",
    "kmc_overlap_fraction",
    "origin_lulc_coverage",
    "minimum_past_3yr_lulc_coverage",
    "nbr8_valid_count_2025",
    "nbr24_valid_count_2025",
    "elevation_valid_coverage_fraction",
    "slope_valid_coverage_fraction",
    "predictor_complete",
]


final_inference_df = (
    final_inference_df[
        metadata_columns
        +
        final_feature_columns
    ]
    .copy()
)


# =============================================================================
# 18. UPDATE ELIGIBILITY TABLE WITH FINAL READINESS
# =============================================================================

readiness_lookup = (
    combined_df[
        [
            "grid_id",
            "predictor_complete",
        ]
    ]
    .set_index(
        "grid_id"
    )[
        "predictor_complete"
    ]
)


eligibility_df[
    "predictor_complete"
] = (
    eligibility_df[
        "grid_id"
    ]
    .map(
        readiness_lookup
    )
    .fillna(
        False
    )
    .astype(
        bool
    )
)


eligibility_df[
    "final_inference_ready"
] = (
    eligibility_df[
        "initial_inference_eligible"
    ]
    &
    eligibility_df[
        "predictor_complete"
    ]
)


# =============================================================================
# 19. QA
# =============================================================================

initial_eligible_count = int(
    eligibility_df[
        "initial_inference_eligible"
    ].sum()
)


final_ready_count = int(
    eligibility_df[
        "final_inference_ready"
    ].sum()
)


excluded_due_predictor_missing = (
    initial_eligible_count
    -
    final_ready_count
)


final_missing_values = int(
    final_inference_df[
        final_feature_columns
    ]
    .isna()
    .sum()
    .sum()
)


final_infinite_values = int(
    np.isinf(
        final_inference_df[
            final_feature_columns
        ]
        .to_numpy(
            dtype=float
        )
    ).sum()
)


feature_count_valid = (
    len(
        final_feature_columns
    )
    ==
    EXPECTED_FEATURES
)


all_origin_year_2025 = bool(
    (
        final_inference_df[
            "origin_year"
        ]
        ==
        ORIGIN_YEAR
    ).all()
)


no_2026_columns = bool(
    not any(
        "2026"
        in
        column
        for column in final_inference_df.columns
    )
)


future_named_features = [
    feature_name
    for feature_name in final_feature_columns
    if (
        "future"
        in feature_name.lower()
        or
        "target_window"
        in feature_name.lower()
    )
]


no_future_named_features = (
    len(
        future_named_features
    )
    ==
    0
)


grid_ids_unique = bool(
    not final_inference_df[
        "grid_id"
    ].duplicated().any()
)


all_final_rows_initially_eligible = bool(
    final_inference_df[
        "grid_id"
    ].isin(
        eligibility_df.loc[
            eligibility_df[
                "initial_inference_eligible"
            ],
            "grid_id",
        ]
    ).all()
)


production_model_scored = False

provisional_2026_used = False

stabilized_lulc_used = False

osm_2026_used = False


overall_status = (
    "PASS"
    if all(
        [
            final_ready_count
            >
            0,
            final_missing_values
            ==
            0,
            final_infinite_values
            ==
            0,
            feature_count_valid,
            all_origin_year_2025,
            no_2026_columns,
            no_future_named_features,
            grid_ids_unique,
            all_final_rows_initially_eligible,
            not production_model_scored,
            not provisional_2026_used,
            not stabilized_lulc_used,
            not osm_2026_used,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 20. REPORTS
# =============================================================================

eligibility_summary_df = pd.DataFrame(
    [
        {
            "all_grid_cells": (
                EXPECTED_GRID_CELLS
            ),
            "kmc_overlap_eligible_cells": int(
                spatial_ok.sum()
            ),
            "past_lulc_coverage_eligible_cells": int(
                coverage_ok.sum()
            ),
            "current_nonbuilt_eligible_cells": int(
                current_nonbuilt_ok.sum()
            ),
            "past_stability_eligible_cells": int(
                past_stable_nonbuilt.sum()
            ),
            "initial_inference_eligible_cells": (
                initial_eligible_count
            ),
            "excluded_due_incomplete_predictors": (
                excluded_due_predictor_missing
            ),
            "final_inference_ready_cells": (
                final_ready_count
            ),
        }
    ]
)


feature_qa_df = pd.DataFrame(
    [
        {
            "origin_year": (
                ORIGIN_YEAR
            ),
            "pressure_period": (
                PRESSURE_PERIOD
            ),
            "final_inference_rows": (
                len(
                    final_inference_df
                )
            ),
            "frozen_feature_count": (
                len(
                    final_feature_columns
                )
            ),
            "missing_final_feature_values": (
                final_missing_values
            ),
            "infinite_final_feature_values": (
                final_infinite_values
            ),
            "grid_ids_unique": (
                grid_ids_unique
            ),
            "all_rows_match_training_domain_eligibility": (
                all_final_rows_initially_eligible
            ),
            "raw_lulc_years": (
                "2023,2024,2025"
            ),
            "provisional_2026_used": (
                provisional_2026_used
            ),
            "centered_stabilized_lulc_used": (
                stabilized_lulc_used
            ),
            "osm_2026_used": (
                osm_2026_used
            ),
            "production_model_scored": (
                production_model_scored
            ),
            "overall_step14b_status": (
                overall_status
            ),
        }
    ]
)


final_summary_df = pd.DataFrame(
    [
        {
            "stage": (
                "2025 production inference feature construction"
            ),
            "inference_origin_year": (
                ORIGIN_YEAR
            ),
            "predictor_history": (
                "2023-2025 RAW LULC"
            ),
            "pressure_target_period": (
                PRESSURE_PERIOD
            ),
            "target_meaning": (
                "Pressure for persistent majority-Built "
                "conversion during next 3 years"
            ),
            "initial_eligible_cells": (
                initial_eligible_count
            ),
            "final_inference_ready_cells": (
                final_ready_count
            ),
            "frozen_predictor_count": (
                len(
                    final_feature_columns
                )
            ),
            "future_information_used": (
                False
            ),
            "provisional_2026_used": (
                False
            ),
            "centered_stabilized_lulc_used": (
                False
            ),
            "osm_2026_used": (
                False
            ),
            "model_scored_in_this_step": (
                False
            ),
            "overall_step14b_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 21. SAVE OUTPUTS
# =============================================================================

final_inference_df.to_csv(
    INFERENCE_FEATURES_CSV,
    index=False,
)


eligibility_df.to_csv(
    ELIGIBILITY_CSV,
    index=False,
)


feature_qa_df.to_csv(
    FEATURE_QA_CSV,
    index=False,
)


eligibility_summary_df.to_csv(
    ELIGIBILITY_SUMMARY_CSV,
    index=False,
)


final_summary_df.to_csv(
    FINAL_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 22. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("FINAL 2025 INFERENCE FEATURE TABLE")
print("=" * 100)


print(
    f"Initial eligible cells             : "
    f"{initial_eligible_count:,}"
)

print(
    f"Excluded - incomplete predictors   : "
    f"{excluded_due_predictor_missing:,}"
)

print(
    f"Final inference-ready cells        : "
    f"{final_ready_count:,}"
)

print(
    f"Frozen predictor count             : "
    f"{len(final_feature_columns)}"
)

print(
    f"Missing final feature values       : "
    f"{final_missing_values}"
)

print(
    f"Infinite final feature values      : "
    f"{final_infinite_values}"
)


print("\n" + "=" * 100)
print("STEP 14B QA")
print("=" * 100)


print(
    f"Feature count = 55                 : "
    f"{feature_count_valid}"
)

print(
    f"All origin years = 2025            : "
    f"{all_origin_year_2025}"
)

print(
    f"Unique inference grid IDs          : "
    f"{grid_ids_unique}"
)

print(
    f"All rows satisfy inference domain  : "
    f"{all_final_rows_initially_eligible}"
)

print(
    f"Future-named model features        : "
    f"{not no_future_named_features}"
)

print(
    f"2026 used                          : "
    f"{provisional_2026_used}"
)

print(
    f"Centered STABILIZED LULC used      : "
    f"{stabilized_lulc_used}"
)

print(
    f"OSM 2026 used                      : "
    f"{osm_2026_used}"
)

print(
    f"Production model scored            : "
    f"{production_model_scored}"
)

print(
    f"Overall Step 14B status            : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(INFERENCE_FEATURES_CSV)}"
)

print(
    f" - {relative_path(ELIGIBILITY_CSV)}"
)

print(
    f" - {relative_path(FEATURE_QA_CSV)}"
)

print(
    f" - {relative_path(ELIGIBILITY_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. This table uses only information available through the 2025 origin."
)

print(
    "2. 2026 provisional Landsat data is completely excluded."
)

print(
    "3. The same predictor-side non-built eligibility domain used during training is applied."
)

print(
    "4. The final table contains exactly the frozen 55 production predictors."
)

print(
    "5. No urbanization-pressure score has been generated yet."
)

print(
    "6. The next step will apply the Step 14A production XGBoost model to these cells."
)

print(
    "7. The resulting map will represent 2026-2028 urbanization pressure, not a guaranteed conversion map."
)

print("=" * 100)