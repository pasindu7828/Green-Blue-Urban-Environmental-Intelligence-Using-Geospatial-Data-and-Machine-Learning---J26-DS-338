"""
STEP 12A - CAUSAL HISTORICAL LULC PREDICTOR ENGINEERING

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Join the FINAL causal urbanization target with historical RAW LULC data.
2. Create leakage-safe LULC predictors using only:
      t-2, t-1, t
3. Create current-state, lag, mean and trend features.
4. Keep future target information completely separate.
5. Produce the first predictor table for later ML modelling.

IMPORTANT:
- RAW LULC only.
- STABILIZED LULC is NOT used.
- Future LULC t+1, t+2, t+3 is NOT used as predictor information.
- Target columns are retained only as labels/metadata.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "target"
    / "urbanization_conversion_target_3yr_final_causal.csv"
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

PREDICTOR_DATASET_CSV = (
    OUTPUT_DIR
    / "12a_causal_lulc_predictors.csv"
)

FEATURE_LIST_CSV = (
    REPORT_DIR
    / "12a_causal_lulc_predictor_feature_list.csv"
)

FEATURE_QA_CSV = (
    REPORT_DIR
    / "12a_causal_lulc_predictor_qa.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "12a_causal_lulc_predictor_summary.csv"
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

REQUIRED_RAW_YEARS = list(
    range(
        2010,
        2023,
    )
)

LULC_CLASSES = [
    "green",
    "built",
    "open_bare",
    "water",
]

EXPECTED_TARGET_ROWS = 51310

EXPECTED_POSITIVE_ROWS = 844


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
    """Return Step 11B annual table."""

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


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 12A - CAUSAL HISTORICAL LULC PREDICTOR ENGINEERING")
print("=" * 100)


if not TARGET_FILE.exists():

    raise FileNotFoundError(
        f"Final Step 11F target not found:\n"
        f"{TARGET_FILE}"
    )


# =============================================================================
# 6. LOAD FINAL TARGET
# =============================================================================

target_df = pd.read_csv(
    TARGET_FILE
)


if len(
    target_df
) != EXPECTED_TARGET_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_TARGET_ROWS:,} target rows, "
        f"found {len(target_df):,}."
    )


positive_rows = int(
    target_df[
        "urban_conversion_target"
    ].sum()
)


if positive_rows != EXPECTED_POSITIVE_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_POSITIVE_ROWS} positive rows, "
        f"found {positive_rows}."
    )


if target_df.duplicated(
    subset=[
        "grid_id",
        "origin_year",
    ]
).any():

    raise RuntimeError(
        "Duplicate grid/origin rows in target dataset."
    )


print(
    f"\nFinal target rows     : "
    f"{len(target_df):,}"
)

print(
    f"Positive target rows  : "
    f"{positive_rows:,}"
)


# =============================================================================
# 7. LOAD RAW ANNUAL LULC DATA
# =============================================================================

raw_frames = []


for year in REQUIRED_RAW_YEARS:

    input_file = annual_file(
        year
    )


    if not input_file.exists():

        raise FileNotFoundError(
            f"Required annual LULC table not found:\n"
            f"{input_file}"
        )


    yearly_df = pd.read_csv(
        input_file
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
    ) == 0:

        raise RuntimeError(
            f"No RAW_RF data found for {year}."
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
            f"Duplicate RAW grid IDs found in {year}."
        )


    raw_frames.append(
        raw_df
    )


raw_long_df = pd.concat(
    raw_frames,
    ignore_index=True,
)


print(
    f"RAW predictor years   : "
    f"{REQUIRED_RAW_YEARS[0]}-{REQUIRED_RAW_YEARS[-1]}"
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


# =============================================================================
# 9. ENGINEER CAUSAL LULC FEATURES
# =============================================================================

records = []


for row in target_df.itertuples(
    index=False
):

    grid_id = (
        row.grid_id
    )

    origin_year = int(
        row.origin_year
    )


    lag2_year = (
        origin_year
        -
        2
    )

    lag1_year = (
        origin_year
        -
        1
    )

    current_year = (
        origin_year
    )


    required_years = [
        lag2_year,
        lag1_year,
        current_year,
    ]


    record = {
        "grid_id": (
            grid_id
        ),
        "grid_row": int(
            row.grid_row
        ),
        "grid_col": int(
            row.grid_col
        ),
        "origin_year": (
            origin_year
        ),
        "urban_conversion_target": int(
            row.urban_conversion_target
        ),
        "kmc_overlap_fraction": float(
            row.kmc_overlap_fraction
        ),
        "boundary_cell": bool(
            row.boundary_cell
        ),
    }


    # =========================================================================
    # CLASS-SPECIFIC FEATURES
    # =========================================================================

    for class_name in LULC_CLASSES:

        matrix = (
            class_matrices[
                class_name
            ]
        )


        lag2_value = float(
            matrix.loc[
                grid_id,
                lag2_year,
            ]
        )


        lag1_value = float(
            matrix.loc[
                grid_id,
                lag1_year,
            ]
        )


        current_value = float(
            matrix.loc[
                grid_id,
                current_year,
            ]
        )


        three_year_values = np.array(
            [
                lag2_value,
                lag1_value,
                current_value,
            ],
            dtype=float,
        )


        record[
            f"{class_name}_fraction_t_minus_2"
        ] = (
            lag2_value
        )


        record[
            f"{class_name}_fraction_t_minus_1"
        ] = (
            lag1_value
        )


        record[
            f"{class_name}_fraction_t"
        ] = (
            current_value
        )


        record[
            f"{class_name}_mean_3yr"
        ] = float(
            np.mean(
                three_year_values
            )
        )


        record[
            f"{class_name}_min_3yr"
        ] = float(
            np.min(
                three_year_values
            )
        )


        record[
            f"{class_name}_max_3yr"
        ] = float(
            np.max(
                three_year_values
            )
        )


        record[
            f"{class_name}_change_1yr"
        ] = (
            current_value
            -
            lag1_value
        )


        record[
            f"{class_name}_change_2yr"
        ] = (
            current_value
            -
            lag2_value
        )


        # For equally spaced t-2, t-1, t:
        # linear endpoint slope over two years.
        record[
            f"{class_name}_slope_3yr"
        ] = (
            (
                current_value
                -
                lag2_value
            )
            /
            2.0
        )


        record[
            f"{class_name}_range_3yr"
        ] = (
            float(
                np.max(
                    three_year_values
                )
                -
                np.min(
                    three_year_values
                )
            )
        )


    # =========================================================================
    # CROSS-CLASS CAUSAL FEATURES
    # =========================================================================

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


    green_lag1 = (
        record[
            "green_fraction_t_minus_1"
        ]
    )


    built_lag1 = (
        record[
            "built_fraction_t_minus_1"
        ]
    )


    green_lag2 = (
        record[
            "green_fraction_t_minus_2"
        ]
    )


    built_lag2 = (
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
            built_lag1
        )
        -
        (
            green_t
            -
            green_lag1
        )
    )


    record[
        "built_growth_vs_green_loss_2yr"
    ] = (
        (
            built_t
            -
            built_lag2
        )
        -
        (
            green_t
            -
            green_lag2
        )
    )


    record[
        "origin_lulc_coverage"
    ] = float(
        coverage_wide.loc[
            grid_id,
            origin_year,
        ]
    )


    record[
        "minimum_past_3yr_lulc_coverage"
    ] = float(
        coverage_wide.loc[
            grid_id,
            required_years,
        ].min()
    )


    records.append(
        record
    )


predictor_df = pd.DataFrame(
    records
)


# =============================================================================
# 10. DEFINE FEATURE LIST
# =============================================================================

metadata_columns = [
    "grid_id",
    "grid_row",
    "grid_col",
    "origin_year",
    "urban_conversion_target",
    "kmc_overlap_fraction",
    "boundary_cell",
]


feature_columns = [
    column
    for column in predictor_df.columns
    if column not in metadata_columns
]


# Do not use grid coordinates / IDs as model predictors.
for prohibited_column in [
    "grid_id",
    "grid_row",
    "grid_col",
]:

    if prohibited_column in feature_columns:

        raise RuntimeError(
            f"Spatial identifier leaked into feature list: "
            f"{prohibited_column}"
        )


# =============================================================================
# 11. FEATURE LIST REPORT
# =============================================================================

feature_list_records = []


for feature_name in feature_columns:

    if "_t_minus_2" in feature_name:

        temporal_role = (
            "PAST_T_MINUS_2"
        )

    elif "_t_minus_1" in feature_name:

        temporal_role = (
            "PAST_T_MINUS_1"
        )

    elif feature_name.endswith(
        "_t"
    ):

        temporal_role = (
            "CURRENT_T"
        )

    elif (
        "3yr"
        in feature_name
        or
        "change"
        in feature_name
        or
        "growth"
        in feature_name
    ):

        temporal_role = (
            "DERIVED_FROM_T_MINUS_2_TO_T"
        )

    else:

        temporal_role = (
            "CURRENT_OR_PAST_DERIVED"
        )


    feature_list_records.append(
        {
            "feature_name": (
                feature_name
            ),
            "feature_group": (
                "CAUSAL_RAW_LULC"
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
        }
    )


feature_list_df = pd.DataFrame(
    feature_list_records
)


# =============================================================================
# 12. QA
# =============================================================================

row_count_matches_target = (
    len(
        predictor_df
    )
    ==
    len(
        target_df
    )
)


duplicate_grid_origin_rows = int(
    predictor_df.duplicated(
        subset=[
            "grid_id",
            "origin_year",
        ]
    ).sum()
)


target_match = bool(
    np.array_equal(
        predictor_df[
            "urban_conversion_target"
        ].to_numpy(),
        target_df[
            "urban_conversion_target"
        ].to_numpy(),
    )
)


positive_count_matches = (
    int(
        predictor_df[
            "urban_conversion_target"
        ].sum()
    )
    ==
    EXPECTED_POSITIVE_ROWS
)


missing_feature_values = int(
    predictor_df[
        feature_columns
    ].isna().sum().sum()
)


infinite_feature_values = int(
    np.isinf(
        predictor_df[
            feature_columns
        ].to_numpy(
            dtype=float
        )
    ).sum()
)


all_fraction_bounds_valid = True


fraction_feature_columns = [
    column
    for column in feature_columns
    if (
        "fraction"
        in column
        and
        "change"
        not in column
    )
]


for column in fraction_feature_columns:

    values = (
        predictor_df[
            column
        ]
        .to_numpy(
            dtype=float
        )
    )


    if (
        np.nanmin(
            values
        )
        <
        -
        1e-8
        or
        np.nanmax(
            values
        )
        >
        1.0
        +
        1e-8
    ):

        all_fraction_bounds_valid = (
            False
        )

        break


no_future_column_names = bool(
    not any(
        (
            "future"
            in column.lower()
            or
            "target_window"
            in column.lower()
        )
        for column in feature_columns
    )
)


raw_only = bool(
    feature_list_df[
        "stabilized_lulc_used"
    ].eq(
        False
    ).all()
)


no_future_information = bool(
    feature_list_df[
        "future_information_used"
    ].eq(
        False
    ).all()
)


overall_status = (
    "PASS"
    if all(
        [
            row_count_matches_target,
            duplicate_grid_origin_rows
            ==
            0,
            target_match,
            positive_count_matches,
            missing_feature_values
            ==
            0,
            infinite_feature_values
            ==
            0,
            all_fraction_bounds_valid,
            no_future_column_names,
            raw_only,
            no_future_information,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 13. QA REPORT
# =============================================================================

qa_df = pd.DataFrame(
    [
        {
            "predictor_rows": (
                len(
                    predictor_df
                )
            ),
            "target_rows": (
                len(
                    target_df
                )
            ),
            "feature_count": (
                len(
                    feature_columns
                )
            ),
            "duplicate_grid_origin_rows": (
                duplicate_grid_origin_rows
            ),
            "missing_feature_values": (
                missing_feature_values
            ),
            "infinite_feature_values": (
                infinite_feature_values
            ),
            "target_matches_step11f": (
                target_match
            ),
            "positive_count_matches_step11f": (
                positive_count_matches
            ),
            "fraction_bounds_valid": (
                all_fraction_bounds_valid
            ),
            "future_named_feature_columns": (
                not no_future_column_names
            ),
            "raw_lulc_only": (
                raw_only
            ),
            "future_information_used": (
                not no_future_information
            ),
            "overall_step12a_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 14. FINAL SUMMARY
# =============================================================================

final_summary_df = pd.DataFrame(
    [
        {
            "input_target": (
                "FINAL_CAUSAL_PAST_2_OF_3"
            ),
            "predictor_source": (
                "RAW_RF LULC"
            ),
            "predictor_time_window": (
                "t-2,t-1,t"
            ),
            "predictor_rows": (
                len(
                    predictor_df
                )
            ),
            "feature_count": (
                len(
                    feature_columns
                )
            ),
            "positive_target_rows": int(
                predictor_df[
                    "urban_conversion_target"
                ].sum()
            ),
            "centered_stabilized_lulc_used": (
                False
            ),
            "future_lulc_used_as_predictor": (
                False
            ),
            "grid_id_used_as_predictor": (
                False
            ),
            "grid_coordinates_used_as_predictor": (
                False
            ),
            "terrain_predictors_added": (
                False
            ),
            "road_predictors_added": (
                False
            ),
            "building_predictors_added": (
                False
            ),
            "overall_step12a_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 15. SAVE OUTPUTS
# =============================================================================

predictor_df.to_csv(
    PREDICTOR_DATASET_CSV,
    index=False,
)


feature_list_df.to_csv(
    FEATURE_LIST_CSV,
    index=False,
)


qa_df.to_csv(
    FEATURE_QA_CSV,
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
print("CAUSAL LULC PREDICTOR DATASET")
print("=" * 100)


print(
    f"Rows                         : "
    f"{len(predictor_df):,}"
)

print(
    f"Features                     : "
    f"{len(feature_columns)}"
)

print(
    f"Positive target rows         : "
    f"{predictor_df['urban_conversion_target'].sum():,}"
)


print("\nFeature names:")

for feature_name in feature_columns:

    print(
        f" - {feature_name}"
    )


print("\n" + "=" * 100)
print("STEP 12A QA")
print("=" * 100)


print(
    f"Rows match Step 11F              : "
    f"{row_count_matches_target}"
)

print(
    f"Duplicate grid/origin rows       : "
    f"{duplicate_grid_origin_rows}"
)

print(
    f"Target matches Step 11F          : "
    f"{target_match}"
)

print(
    f"Positive count matches           : "
    f"{positive_count_matches}"
)

print(
    f"Missing feature values           : "
    f"{missing_feature_values}"
)

print(
    f"Infinite feature values          : "
    f"{infinite_feature_values}"
)

print(
    f"Fraction bounds valid            : "
    f"{all_fraction_bounds_valid}"
)

print(
    f"Future-named feature columns     : "
    f"{not no_future_column_names}"
)

print(
    f"Centered STABILIZED LULC used    : "
    f"{not raw_only}"
)

print(
    f"Future information used          : "
    f"{not no_future_information}"
)

print(
    f"Overall Step 12A status          : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(PREDICTOR_DATASET_CSV)}"
)

print(
    f" - {relative_path(FEATURE_LIST_CSV)}"
)

print(
    f" - {relative_path(FEATURE_QA_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. These predictors use only RAW LULC from t-2, t-1 and t."
)

print(
    "2. No future target-window LULC is used as a predictor."
)

print(
    "3. No centered stabilized LULC is used."
)

print(
    "4. Grid IDs and grid coordinates are metadata only, not predictors."
)

print(
    "5. DEM terrain predictors will be added separately."
)

print(
    "6. Current 2026 OSM roads/buildings have NOT been added because they require temporal-leakage review."
)

print("=" * 100)