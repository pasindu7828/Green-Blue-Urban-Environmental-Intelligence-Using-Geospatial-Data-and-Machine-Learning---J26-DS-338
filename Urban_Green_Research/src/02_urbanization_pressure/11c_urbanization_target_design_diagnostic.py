"""
STEP 11C - URBANIZATION TARGET DESIGN DIAGNOSTIC

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Use RAW annual 100 m LULC fractions only.
2. Inspect grid coverage suitability before ML target construction.
3. Inspect annual Built-up fraction distributions.
4. Test multiple reasonable non-built -> persistent Built-up target rules.
5. Quantify positive / negative class balance by origin year.
6. Do NOT automatically choose or export the final ML target.

Important:
- STABILIZED LULC is intentionally NOT used here.
- 2026 is excluded because it is provisional.
- Candidate future windows use the next three complete years.
- This is target-design sensitivity analysis only.
"""

from pathlib import Path
import itertools

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

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. OUTPUTS
# =============================================================================

COVERAGE_SENSITIVITY_CSV = (
    REPORT_DIR
    / "11c_grid_coverage_threshold_sensitivity.csv"
)

BUILT_DISTRIBUTION_CSV = (
    REPORT_DIR
    / "11c_raw_built_fraction_distribution.csv"
)

TARGET_RULE_SENSITIVITY_CSV = (
    REPORT_DIR
    / "11c_candidate_target_rule_sensitivity.csv"
)

TARGET_RULE_YEAR_CSV = (
    REPORT_DIR
    / "11c_candidate_target_rule_by_origin_year.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "11c_target_design_diagnostic_summary.csv"
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

COMPLETE_YEARS = list(
    range(
        2010,
        2026,
    )
)

# 2026 is deliberately excluded because it is provisional.

TARGET_HORIZON_YEARS = 3

LAST_POSSIBLE_ORIGIN_YEAR = (
    max(
        COMPLETE_YEARS
    )
    -
    TARGET_HORIZON_YEARS
)

ORIGIN_YEARS = list(
    range(
        min(
            COMPLETE_YEARS
        ),
        LAST_POSSIBLE_ORIGIN_YEAR
        +
        1,
    )
)


# -------------------------------------------------------------------------
# Coverage sensitivity
# -------------------------------------------------------------------------

KMC_OVERLAP_THRESHOLDS = [
    0.25,
    0.50,
    0.75,
]

MIN_VALID_LULC_COVERAGE = 0.90


# -------------------------------------------------------------------------
# Candidate target thresholds
# -------------------------------------------------------------------------

CURRENT_MAX_BUILT_THRESHOLDS = [
    0.25,
    0.33,
    0.40,
]

FUTURE_MIN_BUILT_THRESHOLDS = [
    0.50,
    0.60,
]

FUTURE_PERSISTENCE_VOTES = [
    2,
    3,
]

PRIMARY_DIAGNOSTIC_KMC_OVERLAP = 0.50

TERRESTRIAL_NONBUILT_MIN_FRACTION = 0.50


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
    """Return one yearly 100 m LULC fraction CSV."""

    return (
        ANNUAL_DIR
        /
        f"kaduwela_100m_lulc_fractions_{year}.csv"
    )


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 11C - URBANIZATION TARGET DESIGN DIAGNOSTIC")
print("=" * 100)

print(
    f"\nComplete historical years : "
    f"{COMPLETE_YEARS[0]}-{COMPLETE_YEARS[-1]}"
)

print(
    f"2026 excluded             : "
    f"True (provisional)"
)

print(
    f"Diagnostic future window  : "
    f"{TARGET_HORIZON_YEARS} years"
)

print(
    f"Possible origin years     : "
    f"{ORIGIN_YEARS[0]}-{ORIGIN_YEARS[-1]}"
)


# =============================================================================
# 6. LOAD RAW 100 M LULC TABLES
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
        "provisional_year",
        "boundary_cell",
        "kmc_overlap_area_m2",
        "kmc_overlap_fraction",
        "valid_lulc_area_m2",
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
            f"{year} is missing required columns: "
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
        "future_informed"
    ].astype(
        bool
    ).any():

        raise RuntimeError(
            f"RAW data unexpectedly marked future-informed "
            f"for {year}."
        )


    if raw_df[
        "grid_id"
    ].duplicated().any():

        raise RuntimeError(
            f"Duplicate RAW grid IDs found for {year}."
        )


    raw_frames.append(
        raw_df
    )


raw_long_df = pd.concat(
    raw_frames,
    ignore_index=True,
)


expected_grid_cells = int(
    raw_long_df[
        "grid_id"
    ].nunique()
)


expected_rows = (
    expected_grid_cells
    *
    len(
        COMPLETE_YEARS
    )
)


if len(
    raw_long_df
) != expected_rows:

    raise RuntimeError(
        f"Expected {expected_rows:,} RAW rows, "
        f"found {len(raw_long_df):,}."
    )


print(
    f"\nUnique 100 m grid cells : "
    f"{expected_grid_cells:,}"
)

print(
    f"RAW grid-year rows       : "
    f"{len(raw_long_df):,}"
)


# =============================================================================
# 7. STATIC GRID ATTRIBUTES
# =============================================================================

static_grid_df = (
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
        subset=[
            "grid_id"
        ]
    )
    .copy()
)


if len(
    static_grid_df
) != expected_grid_cells:

    raise RuntimeError(
        "Static grid attribute extraction failed."
    )


# =============================================================================
# 8. COVERAGE THRESHOLD SENSITIVITY
# =============================================================================

coverage_records = []


coverage_wide = (
    raw_long_df.pivot(
        index="grid_id",
        columns="year",
        values="valid_lulc_coverage_fraction",
    )
)


for overlap_threshold in (
    KMC_OVERLAP_THRESHOLDS
):

    spatial_grid_ids = set(
        static_grid_df.loc[
            static_grid_df[
                "kmc_overlap_fraction"
            ]
            >=
            overlap_threshold,
            "grid_id",
        ]
    )


    spatial_count = len(
        spatial_grid_ids
    )


    all_year_coverage_valid = (
        coverage_wide.loc[
            list(
                spatial_grid_ids
            ),
            COMPLETE_YEARS,
        ]
        >=
        MIN_VALID_LULC_COVERAGE
    ).all(
        axis=1
    )


    all_year_valid_count = int(
        all_year_coverage_valid.sum()
    )


    # Target-relevant years include all possible origins plus their future years.
    target_relevant_years = sorted(
        set(
            ORIGIN_YEARS
        )
        |
        {
            origin
            +
            offset
            for origin in ORIGIN_YEARS
            for offset in range(
                1,
                TARGET_HORIZON_YEARS
                +
                1,
            )
        }
    )


    target_period_coverage_valid = (
        coverage_wide.loc[
            list(
                spatial_grid_ids
            ),
            target_relevant_years,
        ]
        >=
        MIN_VALID_LULC_COVERAGE
    ).all(
        axis=1
    )


    target_period_valid_count = int(
        target_period_coverage_valid.sum()
    )


    coverage_records.append(
        {
            "kmc_overlap_threshold": (
                overlap_threshold
            ),
            "minimum_lulc_coverage_threshold": (
                MIN_VALID_LULC_COVERAGE
            ),
            "spatially_retained_grid_cells": (
                spatial_count
            ),
            "percent_of_all_grid_cells": (
                spatial_count
                /
                expected_grid_cells
                *
                100
            ),
            "cells_meeting_coverage_all_2010_2025": (
                all_year_valid_count
            ),
            "percent_meeting_coverage_all_2010_2025": (
                all_year_valid_count
                /
                expected_grid_cells
                *
                100
            ),
            "cells_meeting_target_period_coverage": (
                target_period_valid_count
            ),
            "percent_meeting_target_period_coverage": (
                target_period_valid_count
                /
                expected_grid_cells
                *
                100
            ),
        }
    )


coverage_sensitivity_df = pd.DataFrame(
    coverage_records
)


# =============================================================================
# 9. BUILT-UP FRACTION DISTRIBUTION
# =============================================================================

built_distribution_records = []


distribution_df = (
    raw_long_df[
        raw_long_df[
            "kmc_overlap_fraction"
        ]
        >=
        PRIMARY_DIAGNOSTIC_KMC_OVERLAP
    ]
    .copy()
)


distribution_df = (
    distribution_df[
        distribution_df[
            "valid_lulc_coverage_fraction"
        ]
        >=
        MIN_VALID_LULC_COVERAGE
    ]
)


for year in COMPLETE_YEARS:

    year_df = (
        distribution_df[
            distribution_df[
                "year"
            ]
            ==
            year
        ]
    )


    built_values = (
        year_df[
            "built_fraction"
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )


    if len(
        built_values
    ) == 0:

        raise RuntimeError(
            f"No usable Built-up fractions for {year}."
        )


    built_distribution_records.append(
        {
            "year": (
                year
            ),
            "usable_grid_cells": (
                len(
                    built_values
                )
            ),
            "mean_built_fraction": (
                float(
                    np.mean(
                        built_values
                    )
                )
            ),
            "p10": (
                float(
                    np.quantile(
                        built_values,
                        0.10,
                    )
                )
            ),
            "p25": (
                float(
                    np.quantile(
                        built_values,
                        0.25,
                    )
                )
            ),
            "median": (
                float(
                    np.quantile(
                        built_values,
                        0.50,
                    )
                )
            ),
            "p75": (
                float(
                    np.quantile(
                        built_values,
                        0.75,
                    )
                )
            ),
            "p90": (
                float(
                    np.quantile(
                        built_values,
                        0.90,
                    )
                )
            ),
            "p95": (
                float(
                    np.quantile(
                        built_values,
                        0.95,
                    )
                )
            ),
            "percent_built_ge_25": (
                float(
                    np.mean(
                        built_values
                        >=
                        0.25
                    )
                    *
                    100
                )
            ),
            "percent_built_ge_33": (
                float(
                    np.mean(
                        built_values
                        >=
                        0.33
                    )
                    *
                    100
                )
            ),
            "percent_built_ge_40": (
                float(
                    np.mean(
                        built_values
                        >=
                        0.40
                    )
                    *
                    100
                )
            ),
            "percent_built_ge_50": (
                float(
                    np.mean(
                        built_values
                        >=
                        0.50
                    )
                    *
                    100
                )
            ),
            "percent_built_ge_60": (
                float(
                    np.mean(
                        built_values
                        >=
                        0.60
                    )
                    *
                    100
                )
            ),
            "percent_built_ge_75": (
                float(
                    np.mean(
                        built_values
                        >=
                        0.75
                    )
                    *
                    100
                )
            ),
        }
    )


built_distribution_df = pd.DataFrame(
    built_distribution_records
)


# =============================================================================
# 10. CREATE WIDE RAW MATRICES
# =============================================================================

def make_year_matrix(
    value_column,
):
    """Create grid x year matrix."""

    return (
        raw_long_df.pivot(
            index="grid_id",
            columns="year",
            values=value_column,
        )
        .sort_index()
    )


built_wide = make_year_matrix(
    "built_fraction"
)

green_wide = make_year_matrix(
    "green_fraction"
)

open_wide = make_year_matrix(
    "open_bare_fraction"
)

water_wide = make_year_matrix(
    "water_fraction"
)

coverage_wide = make_year_matrix(
    "valid_lulc_coverage_fraction"
)


static_indexed = (
    static_grid_df.set_index(
        "grid_id"
    )
    .loc[
        built_wide.index
    ]
)


# =============================================================================
# 11. CANDIDATE TARGET RULE SENSITIVITY
# =============================================================================

rule_year_records = []


candidate_rules = list(
    itertools.product(
        CURRENT_MAX_BUILT_THRESHOLDS,
        FUTURE_MIN_BUILT_THRESHOLDS,
        FUTURE_PERSISTENCE_VOTES,
    )
)


for (
    current_max_built,
    future_min_built,
    required_future_votes,
) in candidate_rules:

    rule_id = (
        f"CUR_LE_{current_max_built:.2f}"
        f"__FUT_GE_{future_min_built:.2f}"
        f"__{required_future_votes}OF3"
    )


    for origin_year in ORIGIN_YEARS:

        future_years = [
            origin_year
            +
            1,
            origin_year
            +
            2,
            origin_year
            +
            3,
        ]


        # ---------------------------------------------------------------------
        # Spatial eligibility
        # ---------------------------------------------------------------------

        sufficient_kmc_overlap = (
            static_indexed[
                "kmc_overlap_fraction"
            ]
            >=
            PRIMARY_DIAGNOSTIC_KMC_OVERLAP
        )


        # ---------------------------------------------------------------------
        # Data availability: origin + all 3 future years
        # ---------------------------------------------------------------------

        coverage_years = (
            [
                origin_year
            ]
            +
            future_years
        )


        adequate_lulc_coverage = (
            coverage_wide[
                coverage_years
            ]
            >=
            MIN_VALID_LULC_COVERAGE
        ).all(
            axis=1
        )


        # ---------------------------------------------------------------------
        # Current non-built terrestrial eligibility
        # ---------------------------------------------------------------------

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


        current_terrestrial_nonbuilt = (
            current_green
            +
            current_open
        )


        current_nonbuilt_eligible = (
            (
                current_built
                <=
                current_max_built
            )
            &
            (
                current_terrestrial_nonbuilt
                >=
                TERRESTRIAL_NONBUILT_MIN_FRACTION
            )
        )


        # ---------------------------------------------------------------------
        # Final labelable population
        # ---------------------------------------------------------------------

        labelable = (
            sufficient_kmc_overlap
            &
            adequate_lulc_coverage
            &
            current_nonbuilt_eligible
        )


        labelable_count = int(
            labelable.sum()
        )


        # ---------------------------------------------------------------------
        # Future persistent Built-up state
        # ---------------------------------------------------------------------

        future_built = (
            built_wide[
                future_years
            ]
        )


        future_built_votes = (
            future_built
            >=
            future_min_built
        ).sum(
            axis=1
        )


        positive = (
            labelable
            &
            (
                future_built_votes
                >=
                required_future_votes
            )
        )


        positive_count = int(
            positive.sum()
        )


        negative_count = (
            labelable_count
            -
            positive_count
        )


        positive_rate = (
            positive_count
            /
            labelable_count
            *
            100
            if labelable_count
            >
            0
            else
            np.nan
        )


        rule_year_records.append(
            {
                "rule_id": (
                    rule_id
                ),
                "origin_year": (
                    origin_year
                ),
                "future_year_1": (
                    future_years[
                        0
                    ]
                ),
                "future_year_2": (
                    future_years[
                        1
                    ]
                ),
                "future_year_3": (
                    future_years[
                        2
                    ]
                ),
                "kmc_overlap_threshold": (
                    PRIMARY_DIAGNOSTIC_KMC_OVERLAP
                ),
                "minimum_lulc_coverage": (
                    MIN_VALID_LULC_COVERAGE
                ),
                "current_max_built_fraction": (
                    current_max_built
                ),
                "current_min_green_plus_open_fraction": (
                    TERRESTRIAL_NONBUILT_MIN_FRACTION
                ),
                "future_min_built_fraction": (
                    future_min_built
                ),
                "future_required_votes_out_of_3": (
                    required_future_votes
                ),
                "labelable_grid_cells": (
                    labelable_count
                ),
                "positive_conversion_cells": (
                    positive_count
                ),
                "negative_cells": (
                    negative_count
                ),
                "positive_rate_percent": (
                    positive_rate
                ),
            }
        )


rule_year_df = pd.DataFrame(
    rule_year_records
)


# =============================================================================
# 12. AGGREGATE EACH CANDIDATE RULE
# =============================================================================

rule_summary_records = []


for rule_id, group_df in (
    rule_year_df.groupby(
        "rule_id",
        sort=False,
    )
):

    first_row = (
        group_df.iloc[
            0
        ]
    )


    total_labelable = int(
        group_df[
            "labelable_grid_cells"
        ].sum()
    )


    total_positive = int(
        group_df[
            "positive_conversion_cells"
        ].sum()
    )


    total_negative = int(
        group_df[
            "negative_cells"
        ].sum()
    )


    overall_positive_rate = (
        total_positive
        /
        total_labelable
        *
        100
        if total_labelable
        >
        0
        else
        np.nan
    )


    positive_year_counts = (
        group_df[
            "positive_conversion_cells"
        ]
        .to_numpy(
            dtype=int
        )
    )


    rule_summary_records.append(
        {
            "rule_id": (
                rule_id
            ),
            "current_max_built_fraction": (
                first_row[
                    "current_max_built_fraction"
                ]
            ),
            "future_min_built_fraction": (
                first_row[
                    "future_min_built_fraction"
                ]
            ),
            "future_required_votes_out_of_3": (
                first_row[
                    "future_required_votes_out_of_3"
                ]
            ),
            "origin_year_count": (
                len(
                    group_df
                )
            ),
            "total_labelable_grid_years": (
                total_labelable
            ),
            "total_positive_grid_years": (
                total_positive
            ),
            "total_negative_grid_years": (
                total_negative
            ),
            "overall_positive_rate_percent": (
                overall_positive_rate
            ),
            "minimum_positive_cells_in_one_origin_year": (
                int(
                    positive_year_counts.min()
                )
            ),
            "median_positive_cells_per_origin_year": (
                float(
                    np.median(
                        positive_year_counts
                    )
                )
            ),
            "maximum_positive_cells_in_one_origin_year": (
                int(
                    positive_year_counts.max()
                )
            ),
            "origin_years_with_zero_positive_cells": (
                int(
                    np.sum(
                        positive_year_counts
                        ==
                        0
                    )
                )
            ),
        }
    )


rule_summary_df = pd.DataFrame(
    rule_summary_records
)


rule_summary_df = (
    rule_summary_df.sort_values(
        [
            "current_max_built_fraction",
            "future_min_built_fraction",
            "future_required_votes_out_of_3",
        ]
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 13. QA
# =============================================================================

raw_only_used = bool(
    (
        raw_long_df[
            "product_type"
        ]
        ==
        "RAW_RF"
    ).all()
)


no_provisional_2026_used = bool(
    raw_long_df[
        "year"
    ].max()
    ==
    2025
)


fraction_columns = [
    "green_fraction",
    "built_fraction",
    "open_bare_fraction",
    "water_fraction",
]


valid_fraction_rows = (
    raw_long_df[
        "valid_lulc_area_m2"
    ]
    >
    0
)


fraction_totals = (
    raw_long_df.loc[
        valid_fraction_rows,
        fraction_columns,
    ]
    .sum(
        axis=1
    )
)


fraction_sum_valid = bool(
    np.allclose(
        fraction_totals.to_numpy(
            dtype=float
        ),
        1.0,
        atol=1e-8,
    )
)


all_positive_counts_valid = bool(
    (
        rule_year_df[
            "positive_conversion_cells"
        ]
        <=
        rule_year_df[
            "labelable_grid_cells"
        ]
    ).all()
)


all_negative_counts_valid = bool(
    (
        rule_year_df[
            "negative_cells"
        ]
        >=
        0
    ).all()
)


final_target_created = False


overall_status = (
    "PASS"
    if all(
        [
            raw_only_used,
            no_provisional_2026_used,
            fraction_sum_valid,
            all_positive_counts_valid,
            all_negative_counts_valid,
            not final_target_created,
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
            "raw_lulc_only": (
                raw_only_used
            ),
            "historical_years_used": (
                "2010-2025"
            ),
            "provisional_2026_excluded": (
                no_provisional_2026_used
            ),
            "diagnostic_future_window_years": (
                TARGET_HORIZON_YEARS
            ),
            "possible_origin_years": (
                f"{ORIGIN_YEARS[0]}-{ORIGIN_YEARS[-1]}"
            ),
            "candidate_rule_count": (
                len(
                    candidate_rules
                )
            ),
            "primary_diagnostic_kmc_overlap_threshold": (
                PRIMARY_DIAGNOSTIC_KMC_OVERLAP
            ),
            "minimum_valid_lulc_coverage": (
                MIN_VALID_LULC_COVERAGE
            ),
            "current_nonbuilt_terrestrial_min_fraction": (
                TERRESTRIAL_NONBUILT_MIN_FRACTION
            ),
            "final_target_rule_selected": (
                False
            ),
            "final_target_created": (
                False
            ),
            "overall_step11c_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 15. SAVE OUTPUTS
# =============================================================================

coverage_sensitivity_df.to_csv(
    COVERAGE_SENSITIVITY_CSV,
    index=False,
)


built_distribution_df.to_csv(
    BUILT_DISTRIBUTION_CSV,
    index=False,
)


rule_summary_df.to_csv(
    TARGET_RULE_SENSITIVITY_CSV,
    index=False,
)


rule_year_df.to_csv(
    TARGET_RULE_YEAR_CSV,
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
print("GRID COVERAGE THRESHOLD SENSITIVITY")
print("=" * 100)


print(
    coverage_sensitivity_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("RAW BUILT-UP FRACTION DISTRIBUTION")
print("=" * 100)


print(
    built_distribution_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("CANDIDATE TARGET RULE SENSITIVITY")
print("=" * 100)


print(
    rule_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 11C QA")
print("=" * 100)


print(
    f"RAW LULC only used                  : "
    f"{raw_only_used}"
)

print(
    f"2026 provisional excluded           : "
    f"{no_provisional_2026_used}"
)

print(
    f"Class fractions sum to 1            : "
    f"{fraction_sum_valid}"
)

print(
    f"Positive counts valid               : "
    f"{all_positive_counts_valid}"
)

print(
    f"Negative counts valid               : "
    f"{all_negative_counts_valid}"
)

print(
    f"Candidate rules tested              : "
    f"{len(candidate_rules)}"
)

print(
    f"Final target rule selected          : "
    f"False"
)

print(
    f"Final target created                : "
    f"False"
)

print(
    f"Overall Step 11C status             : "
    f"{overall_status}"
)


print("\nGenerated reports:")

print(
    f" - {relative_path(COVERAGE_SENSITIVITY_CSV)}"
)

print(
    f" - {relative_path(BUILT_DISTRIBUTION_CSV)}"
)

print(
    f" - {relative_path(TARGET_RULE_SENSITIVITY_CSV)}"
)

print(
    f" - {relative_path(TARGET_RULE_YEAR_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. No final urbanization-pressure target was created."
)

print(
    "2. Only RAW annual LULC was used, so centered future-informed stabilization was excluded."
)

print(
    "3. 2026 was excluded because it is provisional."
)

print(
    "4. Candidate target rules compare different current/future Built-up thresholds."
)

print(
    "5. The next step will select one defensible target rule from these observed results."
)

print(
    "6. Predictor engineering has not started yet."
)

print("=" * 100)