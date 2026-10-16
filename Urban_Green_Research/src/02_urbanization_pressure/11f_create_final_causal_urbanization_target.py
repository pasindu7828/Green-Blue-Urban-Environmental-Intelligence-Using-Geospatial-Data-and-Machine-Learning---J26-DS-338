"""
STEP 11F - CREATE FINAL CAUSAL URBANIZATION CONVERSION TARGET

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Replace the preliminary single-origin Step 11D target with the
   causally stabilized target selected after Step 11E.
2. Use RAW annual LULC only.
3. Require a stable non-built origin state using t-2, t-1 and t.
4. Require persistent majority Built-up evidence using t+1, t+2 and t+3.
5. Produce the final ML target dataset for later predictor engineering.

FINAL TARGET RULE
-----------------
Spatial / data eligibility:
- KMC overlap >= 50%
- valid LULC coverage >= 90% for all required past and future years

Current origin state:
- Built-up fraction <= 0.33 at t
- Green + Open/Bare fraction >= 0.50 at t

Past stability:
- at least 2 of [t-2, t-1, t] satisfy:
      Built-up <= 0.33
      Green + Open/Bare >= 0.50

Future conversion:
- at least 2 of [t+1, t+2, t+3] have:
      Built-up >= 0.50

Target:
0 = no persistent majority-Built conversion in next three years
1 = persistent majority-Built conversion in next three years

IMPORTANT:
- RAW LULC only.
- Centered STABILIZED LULC is not used.
- Future information is used only for the outcome label.
- Future information must never be used as a predictor.
- 2026 is excluded because it is provisional.
"""

from pathlib import Path

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "target"
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

FINAL_TARGET_CSV = (
    OUTPUT_DIR
    / "urbanization_conversion_target_3yr_final_causal.csv"
)

GRID_EVENT_SUMMARY_CSV = (
    OUTPUT_DIR
    / "urbanization_conversion_grid_event_summary_final_causal.csv"
)

YEAR_SUMMARY_CSV = (
    REPORT_DIR
    / "11f_final_causal_target_by_origin_year.csv"
)

EXCLUSION_SUMMARY_CSV = (
    REPORT_DIR
    / "11f_final_causal_target_exclusion_summary.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "11f_final_causal_target_summary.csv"
)


# =============================================================================
# 3. FINAL LOCKED TARGET RULE
# =============================================================================

COMPLETE_YEARS = list(
    range(
        2010,
        2026,
    )
)

# Need t-2, t-1, t, t+1, t+2, t+3.
ORIGIN_YEARS = list(
    range(
        2012,
        2023,
    )
)

MIN_KMC_OVERLAP = 0.50

MIN_LULC_COVERAGE = 0.90

CURRENT_MAX_BUILT = 0.33

CURRENT_MIN_GREEN_PLUS_OPEN = 0.50

PAST_REQUIRED_NONBUILT_VOTES = 2

FUTURE_MIN_BUILT = 0.50

FUTURE_REQUIRED_BUILT_VOTES = 2

TARGET_HORIZON_YEARS = 3


# Expected values from Step 11E Variant B.
EXPECTED_TOTAL_ELIGIBLE = 51310
EXPECTED_TOTAL_POSITIVE = 844


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
    """Return Step 11B annual LULC fraction CSV."""

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
print("STEP 11F - CREATE FINAL CAUSAL URBANIZATION CONVERSION TARGET")
print("=" * 100)


print("\nFinal target definition:")

print(
    f"KMC overlap                         >= "
    f"{MIN_KMC_OVERLAP:.2f}"
)

print(
    f"Required-year LULC coverage         >= "
    f"{MIN_LULC_COVERAGE:.2f}"
)

print(
    f"Origin Built-up                     <= "
    f"{CURRENT_MAX_BUILT:.2f}"
)

print(
    f"Origin Green + Open/Bare            >= "
    f"{CURRENT_MIN_GREEN_PLUS_OPEN:.2f}"
)

print(
    f"Past stable non-built votes         >= "
    f"{PAST_REQUIRED_NONBUILT_VOTES} of 3"
)

print(
    f"Future Built-up                     >= "
    f"{FUTURE_MIN_BUILT:.2f}"
)

print(
    f"Future persistent Built-up votes    >= "
    f"{FUTURE_REQUIRED_BUILT_VOTES} of 3"
)

print(
    f"Origin years                        : "
    f"{ORIGIN_YEARS[0]}-{ORIGIN_YEARS[-1]}"
)

print(
    "STABILIZED LULC used                : False"
)

print(
    "2026 used                           : False"
)


# =============================================================================
# 6. LOAD RAW ANNUAL LULC GRID TABLES
# =============================================================================

raw_frames = []


for year in COMPLETE_YEARS:

    input_file = annual_file(
        year
    )


    if not input_file.exists():

        raise FileNotFoundError(
            f"Required Step 11B file not found:\n"
            f"{input_file}"
        )


    yearly_df = pd.read_csv(
        input_file
    )


    required_columns = {
        "grid_id",
        "grid_row",
        "grid_col",
        "year",
        "product_type",
        "future_informed",
        "boundary_cell",
        "kmc_overlap_area_m2",
        "kmc_overlap_fraction",
        "valid_lulc_coverage_fraction",
        "green_fraction",
        "built_fraction",
        "open_bare_fraction",
        "water_fraction",
    }


    missing_columns = (
        required_columns
        -
        set(
            yearly_df.columns
        )
    )


    if missing_columns:

        raise ValueError(
            f"{year} missing required columns: "
            f"{sorted(missing_columns)}"
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
            f"No RAW_RF rows found for {year}."
        )


    if raw_df[
        "grid_id"
    ].duplicated().any():

        raise RuntimeError(
            f"Duplicate RAW grid IDs found for {year}."
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


grid_count = int(
    raw_long_df[
        "grid_id"
    ].nunique()
)


expected_raw_rows = (
    grid_count
    *
    len(
        COMPLETE_YEARS
    )
)


if len(
    raw_long_df
) != expected_raw_rows:

    raise RuntimeError(
        f"Expected {expected_raw_rows:,} RAW rows, "
        f"found {len(raw_long_df):,}."
    )


print(
    f"\nGrid cells loaded                   : "
    f"{grid_count:,}"
)

print(
    f"RAW historical grid-year rows       : "
    f"{len(raw_long_df):,}"
)


# =============================================================================
# 7. CREATE GRID x YEAR MATRICES
# =============================================================================

built_wide = create_matrix(
    raw_long_df,
    "built_fraction",
)

green_wide = create_matrix(
    raw_long_df,
    "green_fraction",
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
# 8. STATIC GRID ATTRIBUTES
# =============================================================================

static_df = (
    raw_long_df[
        [
            "grid_id",
            "grid_row",
            "grid_col",
            "boundary_cell",
            "kmc_overlap_area_m2",
            "kmc_overlap_fraction",
        ]
    ]
    .drop_duplicates(
        subset="grid_id"
    )
    .set_index(
        "grid_id"
    )
    .loc[
        built_wide.index
    ]
)


if len(
    static_df
) != grid_count:

    raise RuntimeError(
        "Static grid extraction failed."
    )


# =============================================================================
# 9. BUILD FINAL TARGET
# =============================================================================

target_records = []

year_summary_records = []

exclusion_records = []


for origin_year in ORIGIN_YEARS:

    past_years = [
        origin_year - 2,
        origin_year - 1,
        origin_year,
    ]


    future_years = [
        origin_year + 1,
        origin_year + 2,
        origin_year + 3,
    ]


    required_years = (
        past_years
        +
        future_years
    )


    # -------------------------------------------------------------------------
    # Spatial eligibility
    # -------------------------------------------------------------------------

    spatial_ok = (
        static_df[
            "kmc_overlap_fraction"
        ]
        >=
        MIN_KMC_OVERLAP
    )


    # -------------------------------------------------------------------------
    # Coverage eligibility
    # -------------------------------------------------------------------------

    coverage_ok = (
        coverage_wide[
            required_years
        ]
        >=
        MIN_LULC_COVERAGE
    ).all(
        axis=1
    )


    # -------------------------------------------------------------------------
    # Origin state
    # -------------------------------------------------------------------------

    origin_built = (
        built_wide[
            origin_year
        ]
    )


    origin_green = (
        green_wide[
            origin_year
        ]
    )


    origin_open = (
        open_wide[
            origin_year
        ]
    )


    origin_water = (
        water_wide[
            origin_year
        ]
    )


    origin_green_open = (
        green_open_wide[
            origin_year
        ]
    )


    origin_nonbuilt_ok = (
        (
            origin_built
            <=
            CURRENT_MAX_BUILT
        )
        &
        (
            origin_green_open
            >=
            CURRENT_MIN_GREEN_PLUS_OPEN
        )
    )


    # -------------------------------------------------------------------------
    # Past stable non-built evidence
    # -------------------------------------------------------------------------

    past_nonbuilt_condition = (
        (
            built_wide[
                past_years
            ]
            <=
            CURRENT_MAX_BUILT
        )
        &
        (
            green_open_wide[
                past_years
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


    # -------------------------------------------------------------------------
    # Final eligible modelling population
    # -------------------------------------------------------------------------

    eligible = (
        spatial_ok
        &
        coverage_ok
        &
        origin_nonbuilt_ok
        &
        past_stable_nonbuilt
    )


    # -------------------------------------------------------------------------
    # Future outcome
    # -------------------------------------------------------------------------

    future_built_condition = (
        built_wide[
            future_years
        ]
        >=
        FUTURE_MIN_BUILT
    )


    future_built_votes = (
        future_built_condition.sum(
            axis=1
        )
    )


    target = (
        future_built_votes
        >=
        FUTURE_REQUIRED_BUILT_VOTES
    ).astype(
        np.uint8
    )


    positive = (
        eligible
        &
        (
            target
            ==
            1
        )
    )


    eligible_ids = (
        built_wide.index[
            eligible
        ]
    )


    # -------------------------------------------------------------------------
    # Save final target rows
    # -------------------------------------------------------------------------

    for grid_id in eligible_ids:

        target_records.append(
            {
                "grid_id": (
                    grid_id
                ),
                "grid_row": int(
                    static_df.loc[
                        grid_id,
                        "grid_row",
                    ]
                ),
                "grid_col": int(
                    static_df.loc[
                        grid_id,
                        "grid_col",
                    ]
                ),
                "origin_year": (
                    origin_year
                ),
                "past_window_start_year": (
                    past_years[
                        0
                    ]
                ),
                "past_window_end_year": (
                    past_years[
                        -1
                    ]
                ),
                "target_window_start_year": (
                    future_years[
                        0
                    ]
                ),
                "target_window_end_year": (
                    future_years[
                        -1
                    ]
                ),
                "target_horizon_years": (
                    TARGET_HORIZON_YEARS
                ),
                "urban_conversion_target": int(
                    target.loc[
                        grid_id
                    ]
                ),
                "kmc_overlap_fraction": float(
                    static_df.loc[
                        grid_id,
                        "kmc_overlap_fraction",
                    ]
                ),
                "boundary_cell": bool(
                    static_df.loc[
                        grid_id,
                        "boundary_cell",
                    ]
                ),
                "origin_lulc_coverage": float(
                    coverage_wide.loc[
                        grid_id,
                        origin_year,
                    ]
                ),
                "origin_green_fraction": float(
                    origin_green.loc[
                        grid_id
                    ]
                ),
                "origin_built_fraction": float(
                    origin_built.loc[
                        grid_id
                    ]
                ),
                "origin_open_bare_fraction": float(
                    origin_open.loc[
                        grid_id
                    ]
                ),
                "origin_water_fraction": float(
                    origin_water.loc[
                        grid_id
                    ]
                ),
                "origin_green_plus_open_fraction": float(
                    origin_green_open.loc[
                        grid_id
                    ]
                ),
                "past_year_1": (
                    past_years[
                        0
                    ]
                ),
                "past_year_1_built_fraction": float(
                    built_wide.loc[
                        grid_id,
                        past_years[
                            0
                        ],
                    ]
                ),
                "past_year_2": (
                    past_years[
                        1
                    ]
                ),
                "past_year_2_built_fraction": float(
                    built_wide.loc[
                        grid_id,
                        past_years[
                            1
                        ],
                    ]
                ),
                "past_year_3": (
                    past_years[
                        2
                    ]
                ),
                "past_year_3_built_fraction": float(
                    built_wide.loc[
                        grid_id,
                        past_years[
                            2
                        ],
                    ]
                ),
                "past_nonbuilt_votes": int(
                    past_nonbuilt_votes.loc[
                        grid_id
                    ]
                ),
                "required_past_nonbuilt_votes": (
                    PAST_REQUIRED_NONBUILT_VOTES
                ),
                "future_year_1": (
                    future_years[
                        0
                    ]
                ),
                "future_year_1_built_fraction": float(
                    built_wide.loc[
                        grid_id,
                        future_years[
                            0
                        ],
                    ]
                ),
                "future_year_2": (
                    future_years[
                        1
                    ]
                ),
                "future_year_2_built_fraction": float(
                    built_wide.loc[
                        grid_id,
                        future_years[
                            1
                        ],
                    ]
                ),
                "future_year_3": (
                    future_years[
                        2
                    ]
                ),
                "future_year_3_built_fraction": float(
                    built_wide.loc[
                        grid_id,
                        future_years[
                            2
                        ],
                    ]
                ),
                "future_built_votes": int(
                    future_built_votes.loc[
                        grid_id
                    ]
                ),
                "required_future_built_votes": (
                    FUTURE_REQUIRED_BUILT_VOTES
                ),
                "target_source": (
                    "RAW_RF"
                ),
                "centered_stabilized_lulc_used": (
                    False
                ),
                "future_information_used_as_predictor": (
                    False
                ),
                "provisional_2026_used": (
                    False
                ),
            }
        )


    eligible_count = int(
        eligible.sum()
    )


    positive_count = int(
        positive.sum()
    )


    negative_count = (
        eligible_count
        -
        positive_count
    )


    year_summary_records.append(
        {
            "origin_year": (
                origin_year
            ),
            "eligible_grid_cells": (
                eligible_count
            ),
            "positive_conversion_cells": (
                positive_count
            ),
            "negative_cells": (
                negative_count
            ),
            "positive_rate_percent": (
                positive_count
                /
                eligible_count
                *
                100
                if eligible_count
                >
                0
                else
                np.nan
            ),
        }
    )


    # -------------------------------------------------------------------------
    # Sequential exclusion audit
    # -------------------------------------------------------------------------

    exclusion_records.append(
        {
            "origin_year": (
                origin_year
            ),
            "all_grid_cells": (
                grid_count
            ),
            "excluded_kmc_overlap": int(
                (
                    ~spatial_ok
                ).sum()
            ),
            "excluded_required_year_coverage_after_spatial": int(
                (
                    spatial_ok
                    &
                    ~coverage_ok
                ).sum()
            ),
            "excluded_origin_nonbuilt_rule_after_coverage": int(
                (
                    spatial_ok
                    &
                    coverage_ok
                    &
                    ~origin_nonbuilt_ok
                ).sum()
            ),
            "excluded_past_stability_after_origin_rule": int(
                (
                    spatial_ok
                    &
                    coverage_ok
                    &
                    origin_nonbuilt_ok
                    &
                    ~past_stable_nonbuilt
                ).sum()
            ),
            "final_eligible_grid_cells": (
                eligible_count
            ),
            "positive_conversion_cells": (
                positive_count
            ),
            "negative_cells": (
                negative_count
            ),
        }
    )


# =============================================================================
# 10. CREATE FINAL DATAFRAME
# =============================================================================

target_df = pd.DataFrame(
    target_records
)


if len(
    target_df
) == 0:

    raise RuntimeError(
        "Final causal target dataset is empty."
    )


duplicate_grid_origin_rows = int(
    target_df.duplicated(
        subset=[
            "grid_id",
            "origin_year",
        ]
    ).sum()
)


if duplicate_grid_origin_rows > 0:

    raise RuntimeError(
        "Duplicate grid/origin target rows detected."
    )


year_summary_df = pd.DataFrame(
    year_summary_records
)


exclusion_df = pd.DataFrame(
    exclusion_records
)


# =============================================================================
# 11. PER-GRID EVENT SUMMARY
# =============================================================================

grid_event_df = (
    target_df.groupby(
        "grid_id",
        as_index=False,
    )
    .agg(
        number_of_eligible_origins=(
            "origin_year",
            "count",
        ),
        number_of_positive_origins=(
            "urban_conversion_target",
            "sum",
        ),
        first_eligible_origin_year=(
            "origin_year",
            "min",
        ),
        last_eligible_origin_year=(
            "origin_year",
            "max",
        ),
    )
)


positive_df = (
    target_df[
        target_df[
            "urban_conversion_target"
        ]
        ==
        1
    ]
)


first_positive_origin = (
    positive_df.groupby(
        "grid_id"
    )[
        "origin_year"
    ]
    .min()
)


grid_event_df[
    "first_positive_origin_year"
] = (
    grid_event_df[
        "grid_id"
    ]
    .map(
        first_positive_origin
    )
)


grid_event_df[
    "ever_positive"
] = (
    grid_event_df[
        "number_of_positive_origins"
    ]
    >
    0
)


# =============================================================================
# 12. QA
# =============================================================================

total_eligible = len(
    target_df
)


total_positive = int(
    target_df[
        "urban_conversion_target"
    ].sum()
)


total_negative = (
    total_eligible
    -
    total_positive
)


positive_rate = (
    total_positive
    /
    total_eligible
    *
    100
)


target_values_valid = set(
    target_df[
        "urban_conversion_target"
    ]
    .unique()
    .tolist()
).issubset(
    {
        0,
        1,
    }
)


origin_years_valid = (
    sorted(
        target_df[
            "origin_year"
        ]
        .unique()
        .tolist()
    )
    ==
    ORIGIN_YEARS
)


past_windows_valid = bool(
    (
        target_df[
            "past_window_start_year"
        ]
        ==
        target_df[
            "origin_year"
        ]
        -
        2
    ).all()
    and
    (
        target_df[
            "past_window_end_year"
        ]
        ==
        target_df[
            "origin_year"
        ]
    ).all()
)


future_windows_valid = bool(
    (
        target_df[
            "target_window_start_year"
        ]
        ==
        target_df[
            "origin_year"
        ]
        +
        1
    ).all()
    and
    (
        target_df[
            "target_window_end_year"
        ]
        ==
        target_df[
            "origin_year"
        ]
        +
        3
    ).all()
)


origin_rule_valid = bool(
    (
        target_df[
            "origin_built_fraction"
        ]
        <=
        CURRENT_MAX_BUILT
        +
        1e-10
    ).all()
    and
    (
        target_df[
            "origin_green_plus_open_fraction"
        ]
        >=
        CURRENT_MIN_GREEN_PLUS_OPEN
        -
        1e-10
    ).all()
)


past_vote_rule_valid = bool(
    (
        target_df[
            "past_nonbuilt_votes"
        ]
        >=
        PAST_REQUIRED_NONBUILT_VOTES
    ).all()
)


future_vote_rule_valid = bool(
    (
        target_df.loc[
            target_df[
                "urban_conversion_target"
            ]
            ==
            1,
            "future_built_votes",
        ]
        >=
        FUTURE_REQUIRED_BUILT_VOTES
    ).all()
    and
    (
        target_df.loc[
            target_df[
                "urban_conversion_target"
            ]
            ==
            0,
            "future_built_votes",
        ]
        <
        FUTURE_REQUIRED_BUILT_VOTES
    ).all()
)


kmc_overlap_valid = bool(
    (
        target_df[
            "kmc_overlap_fraction"
        ]
        >=
        MIN_KMC_OVERLAP
        -
        1e-10
    ).all()
)


no_future_predictor_usage = bool(
    (
        ~target_df[
            "future_information_used_as_predictor"
        ].astype(
            bool
        )
    ).all()
)


no_stabilized_usage = bool(
    (
        ~target_df[
            "centered_stabilized_lulc_used"
        ].astype(
            bool
        )
    ).all()
)


no_2026_usage = bool(
    (
        ~target_df[
            "provisional_2026_used"
        ].astype(
            bool
        )
    ).all()
    and
    target_df[
        "target_window_end_year"
    ].max()
    <=
    2025
)


matches_step11e_variant_b = (
    total_eligible
    ==
    EXPECTED_TOTAL_ELIGIBLE
    and
    total_positive
    ==
    EXPECTED_TOTAL_POSITIVE
)


overall_status = (
    "PASS"
    if all(
        [
            duplicate_grid_origin_rows
            ==
            0,
            target_values_valid,
            origin_years_valid,
            past_windows_valid,
            future_windows_valid,
            origin_rule_valid,
            past_vote_rule_valid,
            future_vote_rule_valid,
            kmc_overlap_valid,
            no_future_predictor_usage,
            no_stabilized_usage,
            no_2026_usage,
            matches_step11e_variant_b,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 13. FINAL SUMMARY
# =============================================================================

final_summary_df = pd.DataFrame(
    [
        {
            "target_name": (
                "Causally stable non-built to persistent majority-Built "
                "conversion within next 3 years"
            ),
            "target_version": (
                "FINAL_CAUSAL_PAST_2_OF_3"
            ),
            "target_source": (
                "RAW_RF annual LULC fractions"
            ),
            "origin_years": (
                "2012-2022"
            ),
            "past_window": (
                "t-2,t-1,t"
            ),
            "past_required_nonbuilt_votes": (
                PAST_REQUIRED_NONBUILT_VOTES
            ),
            "future_window": (
                "t+1,t+2,t+3"
            ),
            "future_required_built_votes": (
                FUTURE_REQUIRED_BUILT_VOTES
            ),
            "minimum_kmc_overlap_fraction": (
                MIN_KMC_OVERLAP
            ),
            "minimum_lulc_coverage_fraction": (
                MIN_LULC_COVERAGE
            ),
            "origin_max_built_fraction": (
                CURRENT_MAX_BUILT
            ),
            "origin_min_green_plus_open_fraction": (
                CURRENT_MIN_GREEN_PLUS_OPEN
            ),
            "future_min_built_fraction": (
                FUTURE_MIN_BUILT
            ),
            "total_eligible_grid_years": (
                total_eligible
            ),
            "positive_grid_years": (
                total_positive
            ),
            "negative_grid_years": (
                total_negative
            ),
            "positive_rate_percent": (
                positive_rate
            ),
            "unique_eligible_grid_cells": int(
                target_df[
                    "grid_id"
                ].nunique()
            ),
            "unique_ever_positive_grid_cells": int(
                positive_df[
                    "grid_id"
                ].nunique()
            ),
            "centered_stabilized_lulc_used": (
                False
            ),
            "future_information_used_as_predictor": (
                False
            ),
            "provisional_2026_used": (
                False
            ),
            "matches_step11e_variant_b": (
                matches_step11e_variant_b
            ),
            "predictors_created": (
                False
            ),
            "overall_step11f_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 14. SAVE OUTPUTS
# =============================================================================

target_df.to_csv(
    FINAL_TARGET_CSV,
    index=False,
)


grid_event_df.to_csv(
    GRID_EVENT_SUMMARY_CSV,
    index=False,
)


year_summary_df.to_csv(
    YEAR_SUMMARY_CSV,
    index=False,
)


exclusion_df.to_csv(
    EXCLUSION_SUMMARY_CSV,
    index=False,
)


final_summary_df.to_csv(
    FINAL_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 15. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("FINAL CAUSAL TARGET DISTRIBUTION")
print("=" * 100)


print(
    f"Eligible grid-year rows             : "
    f"{total_eligible:,}"
)

print(
    f"Positive conversion rows            : "
    f"{total_positive:,}"
)

print(
    f"Negative rows                       : "
    f"{total_negative:,}"
)

print(
    f"Positive rate                       : "
    f"{positive_rate:.3f}%"
)

print(
    f"Unique eligible grid cells          : "
    f"{target_df['grid_id'].nunique():,}"
)

print(
    f"Unique ever-positive grid cells     : "
    f"{positive_df['grid_id'].nunique():,}"
)


print("\n" + "=" * 100)
print("FINAL TARGET BY ORIGIN YEAR")
print("=" * 100)


print(
    year_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 11F QA")
print("=" * 100)


print(
    f"Duplicate grid/origin rows           : "
    f"{duplicate_grid_origin_rows}"
)

print(
    f"Target values only 0/1               : "
    f"{target_values_valid}"
)

print(
    f"Origin years 2012-2022 valid         : "
    f"{origin_years_valid}"
)

print(
    f"Past windows valid                   : "
    f"{past_windows_valid}"
)

print(
    f"Future windows valid                 : "
    f"{future_windows_valid}"
)

print(
    f"Origin non-built rule valid          : "
    f"{origin_rule_valid}"
)

print(
    f"Past 2-of-3 stability valid          : "
    f"{past_vote_rule_valid}"
)

print(
    f"Future 2-of-3 persistence valid      : "
    f"{future_vote_rule_valid}"
)

print(
    f"KMC overlap rule valid               : "
    f"{kmc_overlap_valid}"
)

print(
    f"Centered STABILIZED LULC used        : "
    f"{not no_stabilized_usage}"
)

print(
    f"Future information used as predictor : "
    f"{not no_future_predictor_usage}"
)

print(
    f"2026 used                            : "
    f"{not no_2026_usage}"
)

print(
    f"Matches Step 11E Variant B           : "
    f"{matches_step11e_variant_b}"
)

print(
    f"Predictors created                   : "
    f"False"
)

print(
    f"Overall Step 11F status              : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(FINAL_TARGET_CSV)}"
)

print(
    f" - {relative_path(GRID_EVENT_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(YEAR_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(EXCLUSION_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. This is now the final causal urbanization-conversion target."
)

print(
    "2. Step 11D is retained as a preliminary target/audit product only."
)

print(
    "3. Past stability uses only information available at or before the origin year."
)

print(
    "4. Future years are used only to create the outcome label."
)

print(
    "5. Centered stabilized LULC is not used."
)

print(
    "6. 2026 provisional data is not used."
)

print(
    "7. The next stage can begin predictor engineering using only information available at or before each origin year."
)

print("=" * 100)