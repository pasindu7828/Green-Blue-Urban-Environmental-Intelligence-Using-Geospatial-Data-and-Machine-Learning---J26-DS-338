"""
STEP 11E - CAUSAL URBANIZATION TARGET STABILITY AUDIT

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Audit the Step 11D single-origin urbanization target.
2. Test whether positive labels are sensitive to one-year RAW LULC fluctuations.
3. Compare three leakage-safe target definitions:
      A. SINGLE_ORIGIN
      B. PAST_2_OF_3
      C. PAST_2_OF_3_PLUS_FINAL
4. Use RAW annual LULC only.
5. Use past years only for current-state stability.
6. Use future years only for outcome definition.
7. Do NOT overwrite the Step 11D target.

Important:
- Centered STABILIZED maps are not used.
- 2026 is not used.
- This is diagnostic only.
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

VARIANT_SUMMARY_CSV = (
    REPORT_DIR
    / "11e_causal_target_variant_summary.csv"
)

VARIANT_YEAR_CSV = (
    REPORT_DIR
    / "11e_causal_target_variant_by_origin_year.csv"
)

POSITIVE_SURVIVAL_CSV = (
    REPORT_DIR
    / "11e_step11d_positive_survival_summary.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "11e_causal_target_stability_audit_summary.csv"
)


# =============================================================================
# 3. LOCKED THRESHOLDS FROM STEP 11D
# =============================================================================

COMPLETE_YEARS = list(
    range(
        2010,
        2026,
    )
)

# Need t-2, t-1, t and t+1, t+2, t+3.
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

FUTURE_MIN_BUILT = 0.50
FUTURE_REQUIRED_VOTES = 2

PAST_REQUIRED_VOTES = 2


# =============================================================================
# 4. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def annual_file(year: int) -> Path:

    return (
        ANNUAL_DIR
        /
        f"kaduwela_100m_lulc_fractions_{year}.csv"
    )


def create_matrix(
    long_df,
    column_name,
):

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
print("STEP 11E - CAUSAL URBANIZATION TARGET STABILITY AUDIT")
print("=" * 100)

print(
    f"\nAudit origin years : "
    f"{ORIGIN_YEARS[0]}-{ORIGIN_YEARS[-1]}"
)

print(
    "Past window        : t-2, t-1, t"
)

print(
    "Future window      : t+1, t+2, t+3"
)

print(
    "RAW LULC only      : True"
)

print(
    "2026 used          : False"
)


# =============================================================================
# 6. LOAD RAW GRID-YEAR DATA
# =============================================================================

raw_frames = []


for year in COMPLETE_YEARS:

    path = annual_file(
        year
    )


    if not path.exists():

        raise FileNotFoundError(
            f"Missing Step 11B file:\n{path}"
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
            f"No RAW_RF data for {year}."
        )


    if raw_df[
        "grid_id"
    ].duplicated().any():

        raise RuntimeError(
            f"Duplicate RAW grid IDs for {year}."
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


print(
    f"\nGrid cells loaded : "
    f"{grid_count:,}"
)


# =============================================================================
# 7. MATRICES
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

coverage_wide = create_matrix(
    raw_long_df,
    "valid_lulc_coverage_fraction",
)


static_df = (
    raw_long_df[
        [
            "grid_id",
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


green_open_wide = (
    green_wide
    +
    open_wide
)


# =============================================================================
# 8. BUILD TARGET VARIANTS
# =============================================================================

variant_year_records = []

positive_membership_records = []


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


    # -------------------------------------------------------------------------
    # Static spatial eligibility
    # -------------------------------------------------------------------------

    spatial_ok = (
        static_df[
            "kmc_overlap_fraction"
        ]
        >=
        MIN_KMC_OVERLAP
    )


    # -------------------------------------------------------------------------
    # Coverage: all past + all future years
    # -------------------------------------------------------------------------

    required_years = (
        past_years
        +
        future_years
    )


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
    # Current Step 11D origin eligibility
    # -------------------------------------------------------------------------

    current_nonbuilt = (
        (
            built_wide[
                origin_year
            ]
            <=
            CURRENT_MAX_BUILT
        )
        &
        (
            green_open_wide[
                origin_year
            ]
            >=
            CURRENT_MIN_GREEN_PLUS_OPEN
        )
    )


    # -------------------------------------------------------------------------
    # Past causal stability
    # -------------------------------------------------------------------------

    past_nonbuilt_votes = (
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
    ).sum(
        axis=1
    )


    past_stable_nonbuilt = (
        past_nonbuilt_votes
        >=
        PAST_REQUIRED_VOTES
    )


    # -------------------------------------------------------------------------
    # Future conversion outcome
    # -------------------------------------------------------------------------

    future_built_votes = (
        built_wide[
            future_years
        ]
        >=
        FUTURE_MIN_BUILT
    ).sum(
        axis=1
    )


    future_persistent_built = (
        future_built_votes
        >=
        FUTURE_REQUIRED_VOTES
    )


    final_future_year_built = (
        built_wide[
            future_years[
                -1
            ]
        ]
        >=
        FUTURE_MIN_BUILT
    )


    # =========================================================================
    # VARIANT A - SINGLE ORIGIN
    # =========================================================================

    eligible_a = (
        spatial_ok
        &
        coverage_ok
        &
        current_nonbuilt
    )


    positive_a = (
        eligible_a
        &
        future_persistent_built
    )


    # =========================================================================
    # VARIANT B - CURRENT + PAST 2 OF 3
    # =========================================================================

    eligible_b = (
        spatial_ok
        &
        coverage_ok
        &
        current_nonbuilt
        &
        past_stable_nonbuilt
    )


    positive_b = (
        eligible_b
        &
        future_persistent_built
    )


    # =========================================================================
    # VARIANT C - PAST STABLE + FUTURE PERSISTENT + FINAL YEAR BUILT
    # =========================================================================

    eligible_c = (
        eligible_b
    )


    positive_c = (
        eligible_c
        &
        future_persistent_built
        &
        final_future_year_built
    )


    variants = [
        (
            "A_SINGLE_ORIGIN",
            eligible_a,
            positive_a,
        ),
        (
            "B_PAST_2_OF_3",
            eligible_b,
            positive_b,
        ),
        (
            "C_PAST_2_OF_3_PLUS_FINAL",
            eligible_c,
            positive_c,
        ),
    ]


    for (
        variant_name,
        eligible,
        positive,
    ) in variants:

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


        positive_rate = (
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
        )


        variant_year_records.append(
            {
                "variant": (
                    variant_name
                ),
                "origin_year": (
                    origin_year
                ),
                "past_year_1": (
                    past_years[
                        0
                    ]
                ),
                "past_year_2": (
                    past_years[
                        1
                    ]
                ),
                "past_year_3": (
                    past_years[
                        2
                    ]
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
                "eligible_grid_cells": (
                    eligible_count
                ),
                "positive_cells": (
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


    # -------------------------------------------------------------------------
    # Track whether Step 11D-like positives survive stricter variants.
    # -------------------------------------------------------------------------

    step11d_positive_ids = set(
        built_wide.index[
            positive_a
        ]
    )


    b_positive_ids = set(
        built_wide.index[
            positive_b
        ]
    )


    c_positive_ids = set(
        built_wide.index[
            positive_c
        ]
    )


    positive_membership_records.append(
        {
            "origin_year": (
                origin_year
            ),
            "single_origin_positive_cells": (
                len(
                    step11d_positive_ids
                )
            ),
            "survive_past_2_of_3": (
                len(
                    step11d_positive_ids
                    &
                    b_positive_ids
                )
            ),
            "survive_past_2_of_3_plus_final": (
                len(
                    step11d_positive_ids
                    &
                    c_positive_ids
                )
            ),
            "removed_by_past_stability": (
                len(
                    step11d_positive_ids
                    -
                    b_positive_ids
                )
            ),
            "removed_by_final_year_requirement_after_past_filter": (
                len(
                    b_positive_ids
                    -
                    c_positive_ids
                )
            ),
        }
    )


# =============================================================================
# 9. DATAFRAMES
# =============================================================================

variant_year_df = pd.DataFrame(
    variant_year_records
)


positive_survival_df = pd.DataFrame(
    positive_membership_records
)


# =============================================================================
# 10. VARIANT SUMMARY
# =============================================================================

variant_summary_records = []


for variant_name, group_df in (
    variant_year_df.groupby(
        "variant",
        sort=False,
    )
):

    total_eligible = int(
        group_df[
            "eligible_grid_cells"
        ].sum()
    )


    total_positive = int(
        group_df[
            "positive_cells"
        ].sum()
    )


    total_negative = int(
        group_df[
            "negative_cells"
        ].sum()
    )


    positive_rate = (
        total_positive
        /
        total_eligible
        *
        100
        if total_eligible
        >
        0
        else
        np.nan
    )


    yearly_positive_counts = (
        group_df[
            "positive_cells"
        ]
        .to_numpy(
            dtype=float
        )
    )


    yearly_positive_rates = (
        group_df[
            "positive_rate_percent"
        ]
        .to_numpy(
            dtype=float
        )
    )


    variant_summary_records.append(
        {
            "variant": (
                variant_name
            ),
            "origin_year_count": (
                len(
                    group_df
                )
            ),
            "total_eligible_grid_years": (
                total_eligible
            ),
            "total_positive_grid_years": (
                total_positive
            ),
            "total_negative_grid_years": (
                total_negative
            ),
            "overall_positive_rate_percent": (
                positive_rate
            ),
            "minimum_positive_cells_per_year": (
                int(
                    np.min(
                        yearly_positive_counts
                    )
                )
            ),
            "median_positive_cells_per_year": (
                float(
                    np.median(
                        yearly_positive_counts
                    )
                )
            ),
            "maximum_positive_cells_per_year": (
                int(
                    np.max(
                        yearly_positive_counts
                    )
                )
            ),
            "minimum_positive_rate_percent": (
                float(
                    np.min(
                        yearly_positive_rates
                    )
                )
            ),
            "maximum_positive_rate_percent": (
                float(
                    np.max(
                        yearly_positive_rates
                    )
                )
            ),
            "standard_deviation_positive_rate_percent": (
                float(
                    np.std(
                        yearly_positive_rates,
                        ddof=1,
                    )
                )
            ),
        }
    )


variant_summary_df = pd.DataFrame(
    variant_summary_records
)


# =============================================================================
# 11. SURVIVAL SUMMARY
# =============================================================================

single_total = int(
    positive_survival_df[
        "single_origin_positive_cells"
    ].sum()
)


past_survive_total = int(
    positive_survival_df[
        "survive_past_2_of_3"
    ].sum()
)


final_survive_total = int(
    positive_survival_df[
        "survive_past_2_of_3_plus_final"
    ].sum()
)


past_survival_percent = (
    past_survive_total
    /
    single_total
    *
    100
    if single_total
    >
    0
    else
    np.nan
)


final_survival_percent = (
    final_survive_total
    /
    single_total
    *
    100
    if single_total
    >
    0
    else
    np.nan
)


# =============================================================================
# 12. QA
# =============================================================================

raw_only_used = True

no_2026_used = (
    max(
        COMPLETE_YEARS
    )
    ==
    2025
)


all_counts_valid = bool(
    (
        variant_year_df[
            "positive_cells"
        ]
        <=
        variant_year_df[
            "eligible_grid_cells"
        ]
    ).all()
    and
    (
        variant_year_df[
            "negative_cells"
        ]
        >=
        0
    ).all()
)


b_never_exceeds_a = bool(
    variant_year_df.pivot(
        index="origin_year",
        columns="variant",
        values="positive_cells",
    )[
        "B_PAST_2_OF_3"
    ]
    .le(
        variant_year_df.pivot(
            index="origin_year",
            columns="variant",
            values="positive_cells",
        )[
            "A_SINGLE_ORIGIN"
        ]
    )
    .all()
)


c_never_exceeds_b = bool(
    variant_year_df.pivot(
        index="origin_year",
        columns="variant",
        values="positive_cells",
    )[
        "C_PAST_2_OF_3_PLUS_FINAL"
    ]
    .le(
        variant_year_df.pivot(
            index="origin_year",
            columns="variant",
            values="positive_cells",
        )[
            "B_PAST_2_OF_3"
        ]
    )
    .all()
)


final_target_changed = False


overall_status = (
    "PASS"
    if all(
        [
            raw_only_used,
            no_2026_used,
            all_counts_valid,
            b_never_exceeds_a,
            c_never_exceeds_b,
            not final_target_changed,
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
            "raw_lulc_only": (
                True
            ),
            "centered_stabilized_lulc_used": (
                False
            ),
            "provisional_2026_used": (
                False
            ),
            "audit_origin_years": (
                f"{ORIGIN_YEARS[0]}-{ORIGIN_YEARS[-1]}"
            ),
            "past_window_years": (
                3
            ),
            "past_required_nonbuilt_votes": (
                PAST_REQUIRED_VOTES
            ),
            "future_window_years": (
                3
            ),
            "future_required_built_votes": (
                FUTURE_REQUIRED_VOTES
            ),
            "single_origin_positive_rows": (
                single_total
            ),
            "past_stable_positive_rows": (
                past_survive_total
            ),
            "past_stable_plus_final_positive_rows": (
                final_survive_total
            ),
            "past_stable_survival_percent_of_single": (
                past_survival_percent
            ),
            "past_stable_plus_final_survival_percent_of_single": (
                final_survival_percent
            ),
            "step11d_target_overwritten": (
                False
            ),
            "final_target_rule_changed": (
                False
            ),
            "overall_step11e_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 14. SAVE
# =============================================================================

variant_summary_df.to_csv(
    VARIANT_SUMMARY_CSV,
    index=False,
)


variant_year_df.to_csv(
    VARIANT_YEAR_CSV,
    index=False,
)


positive_survival_df.to_csv(
    POSITIVE_SURVIVAL_CSV,
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
print("TARGET VARIANT SUMMARY")
print("=" * 100)


print(
    variant_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("TARGET VARIANTS BY ORIGIN YEAR")
print("=" * 100)


print(
    variant_year_df[
        [
            "variant",
            "origin_year",
            "eligible_grid_cells",
            "positive_cells",
            "positive_rate_percent",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 11D-LIKE POSITIVE SURVIVAL")
print("=" * 100)


print(
    positive_survival_df.to_string(
        index=False
    )
)


print(
    f"\nSingle-origin positives              : "
    f"{single_total:,}"
)

print(
    f"Survive past 2-of-3 stability        : "
    f"{past_survive_total:,} "
    f"({past_survival_percent:.2f}%)"
)

print(
    f"Survive past stability + final year  : "
    f"{final_survive_total:,} "
    f"({final_survival_percent:.2f}%)"
)


print("\n" + "=" * 100)
print("STEP 11E QA")
print("=" * 100)


print(
    f"RAW LULC only                  : "
    f"{raw_only_used}"
)

print(
    f"2026 excluded                  : "
    f"{no_2026_used}"
)

print(
    f"Counts valid                   : "
    f"{all_counts_valid}"
)

print(
    f"B positives <= A positives     : "
    f"{b_never_exceeds_a}"
)

print(
    f"C positives <= B positives     : "
    f"{c_never_exceeds_b}"
)

print(
    f"Step 11D overwritten           : "
    f"{final_target_changed}"
)

print(
    f"Overall Step 11E status        : "
    f"{overall_status}"
)


print("\nGenerated reports:")

print(
    f" - {relative_path(VARIANT_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(VARIANT_YEAR_CSV)}"
)

print(
    f" - {relative_path(POSITIVE_SURVIVAL_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. This is an audit only; Step 11D target was not overwritten."
)

print(
    "2. Variant B uses only past RAW years to confirm stable non-built origin state."
)

print(
    "3. Variant C additionally requires the final future year to remain majority Built-up."
)

print(
    "4. No future information is used as a predictor."
)

print(
    "5. We will inspect these results before deciding whether Step 11D should remain final."
)

print("=" * 100)