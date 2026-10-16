"""
STEP 11D - CREATE FINAL URBANIZATION CONVERSION TARGET

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Create the final 100 m urbanization-conversion target.
2. Use RAW annual LULC only.
3. Use only complete historical years 2010-2025.
4. Exclude provisional 2026.
5. Define current eligible non-built cells.
6. Label whether each eligible cell becomes majority Built-up
   persistently within the next three years.
7. Produce a clean grid-year target table for later ML modelling.

Final target definition:
- KMC overlap >= 50%
- LULC coverage >= 90% for origin and next three years
- current Built-up fraction <= 0.33
- current Green + Open/Bare fraction >= 0.50
- future Built-up fraction >= 0.50
- future condition must occur in at least 2 of next 3 years

Target:
0 = no persistent majority-Built conversion in next 3 years
1 = persistent majority-Built conversion in next 3 years

IMPORTANT:
- RAW LULC only.
- STABILIZED future-informed maps are NOT used.
- 2026 is excluded.
- Future years are used only to define the outcome label.
- They must never be used as predictors for the same origin.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. PROJECT PATHS
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

TARGET_DATASET_CSV = (
    OUTPUT_DIR
    / "urbanization_conversion_target_3yr.csv"
)

GRID_EVENT_SUMMARY_CSV = (
    OUTPUT_DIR
    / "urbanization_conversion_grid_event_summary.csv"
)

YEAR_SUMMARY_CSV = (
    REPORT_DIR
    / "11d_urbanization_target_by_origin_year.csv"
)

EXCLUSION_SUMMARY_CSV = (
    REPORT_DIR
    / "11d_urbanization_target_exclusion_summary.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "11d_urbanization_target_summary.csv"
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

ORIGIN_YEARS = list(
    range(
        2010,
        2023,
    )
)

FUTURE_HORIZON_YEARS = 3

MIN_KMC_OVERLAP = 0.50

MIN_LULC_COVERAGE = 0.90

CURRENT_MAX_BUILT = 0.33

CURRENT_MIN_GREEN_PLUS_OPEN = 0.50

FUTURE_MIN_BUILT = 0.50

FUTURE_REQUIRED_VOTES = 2


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


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 11D - CREATE FINAL URBANIZATION CONVERSION TARGET")
print("=" * 100)

print("\nLocked target definition:")

print(
    f"KMC overlap                     >= "
    f"{MIN_KMC_OVERLAP:.2f}"
)

print(
    f"LULC coverage                   >= "
    f"{MIN_LULC_COVERAGE:.2f}"
)

print(
    f"Current Built-up                <= "
    f"{CURRENT_MAX_BUILT:.2f}"
)

print(
    f"Current Green + Open/Bare       >= "
    f"{CURRENT_MIN_GREEN_PLUS_OPEN:.2f}"
)

print(
    f"Future Built-up                 >= "
    f"{FUTURE_MIN_BUILT:.2f}"
)

print(
    f"Future persistence              >= "
    f"{FUTURE_REQUIRED_VOTES} of 3 years"
)

print(
    f"Origin years                    : "
    f"{ORIGIN_YEARS[0]}-{ORIGIN_YEARS[-1]}"
)

print(
    f"2026 used                       : False"
)


# =============================================================================
# 6. LOAD RAW LULC TABLES
# =============================================================================

raw_frames = []


for year in COMPLETE_YEARS:

    path = annual_file(
        year
    )


    if not path.exists():

        raise FileNotFoundError(
            f"Required Step 11B file not found:\n"
            f"{path}"
        )


    yearly_df = pd.read_csv(
        path
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
            f"Duplicate grid IDs in RAW {year}."
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


expected_rows = (
    grid_count
    *
    len(
        COMPLETE_YEARS
    )
)


if len(
    raw_long_df
) != expected_rows:

    raise RuntimeError(
        "Unexpected RAW grid-year row count."
    )


print(
    f"\nGrid cells loaded               : "
    f"{grid_count:,}"
)

print(
    f"RAW historical grid-year rows   : "
    f"{len(raw_long_df):,}"
)


# =============================================================================
# 7. CREATE GRID × YEAR MATRICES
# =============================================================================

def create_matrix(column_name):
    """Create grid_id × year matrix."""

    return (
        raw_long_df.pivot(
            index="grid_id",
            columns="year",
            values=column_name,
        )
        .sort_index()
    )


built_wide = create_matrix(
    "built_fraction"
)

green_wide = create_matrix(
    "green_fraction"
)

open_wide = create_matrix(
    "open_bare_fraction"
)

water_wide = create_matrix(
    "water_fraction"
)

coverage_wide = create_matrix(
    "valid_lulc_coverage_fraction"
)


# =============================================================================
# 8. STATIC GRID INFORMATION
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


# =============================================================================
# 9. BUILD TARGET DATASET
# =============================================================================

target_records = []

exclusion_records = []


for origin_year in ORIGIN_YEARS:

    future_years = [
        origin_year + 1,
        origin_year + 2,
        origin_year + 3,
    ]


    current_built = (
        built_wide[
            origin_year
        ]
    )


    current_green = (
        green_wide[
            origin_year
        ]
    )


    current_open = (
        open_wide[
            origin_year
        ]
    )


    current_water = (
        water_wide[
            origin_year
        ]
    )


    current_nonbuilt_terrestrial = (
        current_green
        +
        current_open
    )


    sufficient_kmc_overlap = (
        static_df[
            "kmc_overlap_fraction"
        ]
        >=
        MIN_KMC_OVERLAP
    )


    required_coverage_years = (
        [
            origin_year
        ]
        +
        future_years
    )


    sufficient_lulc_coverage = (
        coverage_wide[
            required_coverage_years
        ]
        >=
        MIN_LULC_COVERAGE
    ).all(
        axis=1
    )


    current_built_eligible = (
        current_built
        <=
        CURRENT_MAX_BUILT
    )


    current_nonbuilt_eligible = (
        current_nonbuilt_terrestrial
        >=
        CURRENT_MIN_GREEN_PLUS_OPEN
    )


    labelable = (
        sufficient_kmc_overlap
        &
        sufficient_lulc_coverage
        &
        current_built_eligible
        &
        current_nonbuilt_eligible
    )


    future_built_df = (
        built_wide[
            future_years
        ]
    )


    future_votes = (
        future_built_df
        >=
        FUTURE_MIN_BUILT
    ).sum(
        axis=1
    )


    target = (
        future_votes
        >=
        FUTURE_REQUIRED_VOTES
    ).astype(
        np.uint8
    )


    positive = (
        labelable
        &
        (
            target
            ==
            1
        )
    )


    # -------------------------------------------------------------------------
    # Save only labelable grid-year observations
    # -------------------------------------------------------------------------

    labelable_ids = (
        built_wide.index[
            labelable
        ]
    )


    for grid_id in labelable_ids:

        first_future_year = (
            future_years[
                0
            ]
        )

        second_future_year = (
            future_years[
                1
            ]
        )

        third_future_year = (
            future_years[
                2
            ]
        )


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
                "target_horizon_years": (
                    FUTURE_HORIZON_YEARS
                ),
                "target_window_start_year": (
                    first_future_year
                ),
                "target_window_end_year": (
                    third_future_year
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
                    current_green.loc[
                        grid_id
                    ]
                ),
                "origin_built_fraction": float(
                    current_built.loc[
                        grid_id
                    ]
                ),
                "origin_open_bare_fraction": float(
                    current_open.loc[
                        grid_id
                    ]
                ),
                "origin_water_fraction": float(
                    current_water.loc[
                        grid_id
                    ]
                ),
                "origin_green_plus_open_fraction": float(
                    current_nonbuilt_terrestrial.loc[
                        grid_id
                    ]
                ),
                "future_year_1": (
                    first_future_year
                ),
                "future_year_1_built_fraction": float(
                    built_wide.loc[
                        grid_id,
                        first_future_year,
                    ]
                ),
                "future_year_2": (
                    second_future_year
                ),
                "future_year_2_built_fraction": float(
                    built_wide.loc[
                        grid_id,
                        second_future_year,
                    ]
                ),
                "future_year_3": (
                    third_future_year
                ),
                "future_year_3_built_fraction": float(
                    built_wide.loc[
                        grid_id,
                        third_future_year,
                    ]
                ),
                "future_built_threshold": (
                    FUTURE_MIN_BUILT
                ),
                "future_built_votes": int(
                    future_votes.loc[
                        grid_id
                    ]
                ),
                "required_future_votes": (
                    FUTURE_REQUIRED_VOTES
                ),
                "target_source": (
                    "RAW_RF"
                ),
                "future_informed_predictors_used": (
                    False
                ),
                "provisional_2026_used": (
                    False
                ),
            }
        )


    # -------------------------------------------------------------------------
    # Exclusion audit
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
                    ~sufficient_kmc_overlap
                ).sum()
            ),
            "excluded_required_lulc_coverage": int(
                (
                    sufficient_kmc_overlap
                    &
                    ~sufficient_lulc_coverage
                ).sum()
            ),
            "excluded_current_built_above_threshold": int(
                (
                    sufficient_kmc_overlap
                    &
                    sufficient_lulc_coverage
                    &
                    ~current_built_eligible
                ).sum()
            ),
            "excluded_current_nonbuilt_below_threshold": int(
                (
                    sufficient_kmc_overlap
                    &
                    sufficient_lulc_coverage
                    &
                    current_built_eligible
                    &
                    ~current_nonbuilt_eligible
                ).sum()
            ),
            "labelable_grid_cells": int(
                labelable.sum()
            ),
            "positive_grid_cells": int(
                positive.sum()
            ),
            "negative_grid_cells": int(
                labelable.sum()
                -
                positive.sum()
            ),
        }
    )


# =============================================================================
# 10. FINAL TARGET TABLE
# =============================================================================

target_df = pd.DataFrame(
    target_records
)


if len(
    target_df
) == 0:

    raise RuntimeError(
        "Final target dataset is empty."
    )


duplicate_target_rows = int(
    target_df.duplicated(
        subset=[
            "grid_id",
            "origin_year",
        ]
    ).sum()
)


if duplicate_target_rows > 0:

    raise RuntimeError(
        "Duplicate grid_id/origin_year target rows detected."
    )


# =============================================================================
# 11. YEAR SUMMARY
# =============================================================================

year_summary_df = (
    target_df.groupby(
        "origin_year",
        as_index=False,
    )
    .agg(
        labelable_grid_cells=(
            "grid_id",
            "count",
        ),
        positive_conversion_cells=(
            "urban_conversion_target",
            "sum",
        ),
    )
)


year_summary_df[
    "negative_cells"
] = (
    year_summary_df[
        "labelable_grid_cells"
    ]
    -
    year_summary_df[
        "positive_conversion_cells"
    ]
)


year_summary_df[
    "positive_rate_percent"
] = (
    year_summary_df[
        "positive_conversion_cells"
    ]
    /
    year_summary_df[
        "labelable_grid_cells"
    ]
    *
    100
)


# =============================================================================
# 12. PER-GRID EVENT SUMMARY
# =============================================================================

grid_event_df = (
    target_df.groupby(
        "grid_id",
        as_index=False,
    )
    .agg(
        number_of_labelable_origins=(
            "origin_year",
            "count",
        ),
        number_of_positive_origins=(
            "urban_conversion_target",
            "sum",
        ),
        first_labelable_origin_year=(
            "origin_year",
            "min",
        ),
        last_labelable_origin_year=(
            "origin_year",
            "max",
        ),
    )
)


positive_rows = (
    target_df[
        target_df[
            "urban_conversion_target"
        ]
        ==
        1
    ]
)


first_positive_year = (
    positive_rows.groupby(
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
        first_positive_year
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
# 13. QA
# =============================================================================

total_labelable = len(
    target_df
)


total_positive = int(
    target_df[
        "urban_conversion_target"
    ].sum()
)


total_negative = (
    total_labelable
    -
    total_positive
)


positive_rate = (
    total_positive
    /
    total_labelable
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


all_origin_years_present = (
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


all_future_windows_valid = bool(
    (
        target_df[
            "target_window_end_year"
        ]
        ==
        target_df[
            "origin_year"
        ]
        +
        FUTURE_HORIZON_YEARS
    ).all()
)


no_2026_used = bool(
    target_df[
        "target_window_end_year"
    ].max()
    <=
    2025
)


all_current_built_eligible = bool(
    (
        target_df[
            "origin_built_fraction"
        ]
        <=
        CURRENT_MAX_BUILT
        +
        1e-10
    ).all()
)


all_current_nonbuilt_eligible = bool(
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


all_kmc_overlap_valid = bool(
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
        FUTURE_REQUIRED_VOTES
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
        FUTURE_REQUIRED_VOTES
    ).all()
)


target_matches_step11c = (
    total_labelable
    ==
    63452
    and
    total_positive
    ==
    1543
)


overall_status = (
    "PASS"
    if all(
        [
            duplicate_target_rows
            ==
            0,
            target_values_valid,
            all_origin_years_present,
            all_future_windows_valid,
            no_2026_used,
            all_current_built_eligible,
            all_current_nonbuilt_eligible,
            all_kmc_overlap_valid,
            future_vote_rule_valid,
            target_matches_step11c,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 14. FINAL SUMMARY
# =============================================================================

final_summary_df = pd.DataFrame(
    [
        {
            "target_name": (
                "Persistent majority-Built conversion within next 3 years"
            ),
            "target_source": (
                "RAW_RF annual LULC fractions"
            ),
            "origin_years": (
                "2010-2022"
            ),
            "last_target_year": (
                2025
            ),
            "provisional_2026_used": (
                False
            ),
            "minimum_kmc_overlap_fraction": (
                MIN_KMC_OVERLAP
            ),
            "minimum_lulc_coverage_fraction": (
                MIN_LULC_COVERAGE
            ),
            "current_max_built_fraction": (
                CURRENT_MAX_BUILT
            ),
            "current_min_green_plus_open_fraction": (
                CURRENT_MIN_GREEN_PLUS_OPEN
            ),
            "future_min_built_fraction": (
                FUTURE_MIN_BUILT
            ),
            "future_required_votes_out_of_3": (
                FUTURE_REQUIRED_VOTES
            ),
            "total_labelable_grid_years": (
                total_labelable
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
            "unique_labelable_grid_cells": (
                int(
                    target_df[
                        "grid_id"
                    ].nunique()
                )
            ),
            "unique_ever_positive_grid_cells": (
                int(
                    positive_rows[
                        "grid_id"
                    ].nunique()
                )
            ),
            "duplicate_grid_origin_rows": (
                duplicate_target_rows
            ),
            "matches_step11c_selected_rule": (
                target_matches_step11c
            ),
            "predictors_created": (
                False
            ),
            "overall_step11d_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 15. SAVE OUTPUTS
# =============================================================================

target_df.to_csv(
    TARGET_DATASET_CSV,
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


pd.DataFrame(
    exclusion_records
).to_csv(
    EXCLUSION_SUMMARY_CSV,
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
print("TARGET CLASS DISTRIBUTION")
print("=" * 100)


print(
    f"Labelable grid-year rows       : "
    f"{total_labelable:,}"
)

print(
    f"Positive conversion rows       : "
    f"{total_positive:,}"
)

print(
    f"Negative rows                  : "
    f"{total_negative:,}"
)

print(
    f"Positive rate                  : "
    f"{positive_rate:.3f}%"
)

print(
    f"Unique labelable grid cells    : "
    f"{target_df['grid_id'].nunique():,}"
)

print(
    f"Unique ever-positive cells     : "
    f"{positive_rows['grid_id'].nunique():,}"
)


print("\n" + "=" * 100)
print("TARGET BY ORIGIN YEAR")
print("=" * 100)


print(
    year_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 11D QA")
print("=" * 100)


print(
    f"Duplicate grid/origin rows          : "
    f"{duplicate_target_rows}"
)

print(
    f"Target values only 0/1              : "
    f"{target_values_valid}"
)

print(
    f"All origin years 2010-2022 present  : "
    f"{all_origin_years_present}"
)

print(
    f"Future windows exactly 3 years      : "
    f"{all_future_windows_valid}"
)

print(
    f"2026 excluded                       : "
    f"{no_2026_used}"
)

print(
    f"Current Built-up eligibility valid  : "
    f"{all_current_built_eligible}"
)

print(
    f"Current non-built eligibility valid : "
    f"{all_current_nonbuilt_eligible}"
)

print(
    f"KMC overlap rule valid              : "
    f"{all_kmc_overlap_valid}"
)

print(
    f"Future persistence labels valid     : "
    f"{future_vote_rule_valid}"
)

print(
    f"Matches Step 11C selected rule      : "
    f"{target_matches_step11c}"
)

print(
    f"Predictors created                  : "
    f"False"
)

print(
    f"Overall Step 11D status             : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(TARGET_DATASET_CSV)}"
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
    "1. This is the final historical urbanization-conversion target definition."
)

print(
    "2. Target = 1 means persistent majority-Built conversion within the next three complete years."
)

print(
    "3. Only RAW LULC was used."
)

print(
    "4. Future years are used only to define the outcome and will not become predictors."
)

print(
    "5. The target is imbalanced, so future RF/XGBoost training and evaluation must handle class imbalance carefully."
)

print(
    "6. Predictor engineering starts only after this target passes QA."
)

print("=" * 100)