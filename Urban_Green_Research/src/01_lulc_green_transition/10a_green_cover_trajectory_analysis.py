"""
STEP 10A - GREEN-COVER TRAJECTORY ANALYSIS

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Analyse the annual stabilized analytical LULC series.
2. Treat 2011-2024 as the core retrospective trend period.
3. Keep 2010 and 2025 as contextual raw edge years.
4. Keep 2026 explicitly provisional.
5. Quantify long-term Green / Vegetation and Built-up trajectories.
6. Calculate robust Sen slopes without treating annual fluctuations
   as confirmed pixel-level land conversion.

IMPORTANT:
- This step analyses annual class composition only.
- It does NOT yet identify persistent pixel-level transitions.
- Percentage is the primary temporal measure because valid raster
  coverage differs slightly in a few years.
- Area in km2 is retained for planning interpretation.
- No model training or classification changes occur.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

INPUT_CSV = (
    REPORT_DIR
    / "08_annual_lulc_class_area_summary.csv"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. OUTPUTS
# =============================================================================

ANNUAL_TRAJECTORY_CSV = (
    REPORT_DIR
    / "10a_green_cover_annual_trajectory.csv"
)

CORE_TREND_SUMMARY_CSV = (
    REPORT_DIR
    / "10a_lulc_core_trend_summary.csv"
)

PERIOD_STATUS_CSV = (
    REPORT_DIR
    / "10a_lulc_period_status_summary.csv"
)

GREEN_KEY_SUMMARY_CSV = (
    REPORT_DIR
    / "10a_green_cover_key_summary.csv"
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

CORE_START_YEAR = 2011
CORE_END_YEAR = 2024

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


# =============================================================================
# 4. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """Return a project-relative path where possible."""

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def sen_slope(
    years,
    values,
):
    """
    Calculate the Theil-Sen median pairwise slope.

    This is used descriptively as a robust long-term trend measure.
    """

    years = np.asarray(
        years,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )


    if len(
        years
    ) != len(
        values
    ):

        raise ValueError(
            "Years and values must have the same length."
        )


    if len(
        years
    ) < 2:

        return np.nan


    slopes = []


    for first_index in range(
        len(
            years
        )
        -
        1
    ):

        for second_index in range(
            first_index
            +
            1,
            len(
                years
            ),
        ):

            year_difference = (
                years[
                    second_index
                ]
                -
                years[
                    first_index
                ]
            )


            if year_difference == 0:

                continue


            value_difference = (
                values[
                    second_index
                ]
                -
                values[
                    first_index
                ]
            )


            slopes.append(
                value_difference
                /
                year_difference
            )


    if len(
        slopes
    ) == 0:

        return np.nan


    return float(
        np.median(
            slopes
        )
    )


def trend_direction(
    slope,
    tolerance=1e-12,
):
    """Simple descriptive trend direction."""

    if not np.isfinite(
        slope
    ):

        return "UNDEFINED"

    if slope > tolerance:

        return "INCREASING"

    if slope < -tolerance:

        return "DECREASING"

    return "STABLE"


def year_status(year: int) -> str:
    """Assign methodological status to each annual map."""

    if year == 2010:

        return "CONTEXT_RAW_EDGE_YEAR"

    if (
        CORE_START_YEAR
        <=
        year
        <=
        CORE_END_YEAR
    ):

        return "CORE_STABILIZED_RETROSPECTIVE"

    if year == 2025:

        return "CONTEXT_RAW_EDGE_YEAR"

    if year == 2026:

        return "PROVISIONAL_PARTIAL_YEAR"

    raise ValueError(
        f"Unexpected year: {year}"
    )


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 10A - GREEN-COVER TRAJECTORY ANALYSIS")
print("=" * 100)


if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"Required Step 08 summary not found:\n"
        f"{INPUT_CSV}"
    )


# =============================================================================
# 6. LOAD STABILIZED ANALYTICAL PRODUCT
# =============================================================================

input_df = pd.read_csv(
    INPUT_CSV
)


required_columns = {
    "year",
    "product_type",
    "class_id",
    "class_name",
    "valid_pixels",
    "class_pixels",
    "class_area_km2",
    "class_percent",
    "stabilization_applied",
    "provisional_year",
}


missing_columns = (
    required_columns
    -
    set(
        input_df.columns
    )
)


if missing_columns:

    raise ValueError(
        "Input dataset is missing required columns: "
        f"{sorted(missing_columns)}"
    )


analysis_df = (
    input_df[
        input_df[
            "product_type"
        ]
        ==
        "STABILIZED_ANALYTICAL"
    ]
    .copy()
)


years_found = sorted(
    analysis_df[
        "year"
    ]
    .astype(int)
    .unique()
    .tolist()
)


if years_found != EXPECTED_YEARS:

    raise ValueError(
        f"Expected stabilized analytical years "
        f"2010-2026, found: {years_found}"
    )


print(
    f"\nAnnual years available : "
    f"{years_found[0]}-{years_found[-1]}"
)

print(
    f"Core trend period      : "
    f"{CORE_START_YEAR}-{CORE_END_YEAR}"
)


# =============================================================================
# 7. BUILD WIDE ANNUAL TRAJECTORY TABLE
# =============================================================================

percent_wide = (
    analysis_df.pivot(
        index="year",
        columns="class_id",
        values="class_percent",
    )
    .rename(
        columns={
            1: "green_percent",
            2: "built_percent",
            3: "open_bare_percent",
            4: "water_percent",
        }
    )
    .reset_index()
)


area_wide = (
    analysis_df.pivot(
        index="year",
        columns="class_id",
        values="class_area_km2",
    )
    .rename(
        columns={
            1: "green_area_km2",
            2: "built_area_km2",
            3: "open_bare_area_km2",
            4: "water_area_km2",
        }
    )
    .reset_index()
)


pixel_wide = (
    analysis_df.pivot(
        index="year",
        columns="class_id",
        values="class_pixels",
    )
    .rename(
        columns={
            1: "green_pixels",
            2: "built_pixels",
            3: "open_bare_pixels",
            4: "water_pixels",
        }
    )
    .reset_index()
)


valid_pixel_df = (
    analysis_df.groupby(
        "year",
        as_index=False,
    )[
        "valid_pixels"
    ]
    .first()
)


annual_df = (
    percent_wide.merge(
        area_wide,
        on="year",
        how="left",
    )
    .merge(
        pixel_wide,
        on="year",
        how="left",
    )
    .merge(
        valid_pixel_df,
        on="year",
        how="left",
    )
    .sort_values(
        "year"
    )
    .reset_index(
        drop=True
    )
)


annual_df[
    "period_status"
] = [
    year_status(
        int(
            year
        )
    )
    for year in annual_df[
        "year"
    ]
]


annual_df[
    "included_in_core_trend"
] = (
    annual_df[
        "year"
    ]
    .between(
        CORE_START_YEAR,
        CORE_END_YEAR,
    )
)


annual_df[
    "provisional_year"
] = (
    annual_df[
        "year"
    ]
    ==
    2026
)


# =============================================================================
# 8. YEAR-TO-YEAR DESCRIPTIVE CHANGES
# =============================================================================

for column in [
    "green_percent",
    "built_percent",
    "open_bare_percent",
    "water_percent",
]:

    annual_df[
        f"{column}_change_from_previous_year_pp"
    ] = (
        annual_df[
            column
        ]
        .diff()
    )


for column in [
    "green_area_km2",
    "built_area_km2",
    "open_bare_area_km2",
    "water_area_km2",
]:

    annual_df[
        f"{column}_change_from_previous_year_km2"
    ] = (
        annual_df[
            column
        ]
        .diff()
    )


annual_df[
    "green_to_built_percent_ratio"
] = (
    annual_df[
        "green_percent"
    ]
    /
    annual_df[
        "built_percent"
    ]
)


# =============================================================================
# 9. CORE PERIOD
# =============================================================================

core_df = (
    annual_df[
        annual_df[
            "included_in_core_trend"
        ]
    ]
    .copy()
)


if len(
    core_df
) != 14:

    raise ValueError(
        f"Expected 14 core years from 2011-2024, "
        f"found {len(core_df)}."
    )


# =============================================================================
# 10. CLASS-LEVEL CORE TREND SUMMARY
# =============================================================================

trend_records = []


class_columns = {
    1: (
        "green_percent",
        "green_area_km2",
    ),
    2: (
        "built_percent",
        "built_area_km2",
    ),
    3: (
        "open_bare_percent",
        "open_bare_area_km2",
    ),
    4: (
        "water_percent",
        "water_area_km2",
    ),
}


for class_id, (
    percent_column,
    area_column,
) in class_columns.items():

    years = (
        core_df[
            "year"
        ]
        .to_numpy(
            dtype=float
        )
    )


    percent_values = (
        core_df[
            percent_column
        ]
        .to_numpy(
            dtype=float
        )
    )


    area_values = (
        core_df[
            area_column
        ]
        .to_numpy(
            dtype=float
        )
    )


    percent_slope = sen_slope(
        years,
        percent_values,
    )


    area_slope = sen_slope(
        years,
        area_values,
    )


    start_percent = float(
        percent_values[
            0
        ]
    )

    end_percent = float(
        percent_values[
            -1
        ]
    )


    start_area = float(
        area_values[
            0
        ]
    )

    end_area = float(
        area_values[
            -1
        ]
    )


    net_percent_point_change = (
        end_percent
        -
        start_percent
    )


    net_area_change = (
        end_area
        -
        start_area
    )


    relative_percent_change = (
        net_percent_point_change
        /
        start_percent
        *
        100
        if start_percent
        !=
        0
        else
        np.nan
    )


    minimum_index = int(
        np.argmin(
            percent_values
        )
    )

    maximum_index = int(
        np.argmax(
            percent_values
        )
    )


    mean_absolute_annual_change = float(
        np.mean(
            np.abs(
                np.diff(
                    percent_values
                )
            )
        )
    )


    trend_records.append(
        {
            "class_id": (
                class_id
            ),
            "class_name": (
                CLASS_DEFINITIONS[
                    class_id
                ]
            ),
            "core_start_year": (
                CORE_START_YEAR
            ),
            "core_end_year": (
                CORE_END_YEAR
            ),
            "core_year_count": (
                len(
                    core_df
                )
            ),
            "start_percent": (
                start_percent
            ),
            "end_percent": (
                end_percent
            ),
            "net_change_percent_points": (
                net_percent_point_change
            ),
            "relative_change_percent_of_start": (
                relative_percent_change
            ),
            "start_area_km2": (
                start_area
            ),
            "end_area_km2": (
                end_area
            ),
            "net_area_change_km2": (
                net_area_change
            ),
            "sen_slope_percent_points_per_year": (
                percent_slope
            ),
            "sen_slope_area_km2_per_year": (
                area_slope
            ),
            "sen_trend_direction": (
                trend_direction(
                    percent_slope
                )
            ),
            "minimum_percent": (
                float(
                    np.min(
                        percent_values
                    )
                )
            ),
            "minimum_percent_year": (
                int(
                    years[
                        minimum_index
                    ]
                )
            ),
            "maximum_percent": (
                float(
                    np.max(
                        percent_values
                    )
                )
            ),
            "maximum_percent_year": (
                int(
                    years[
                        maximum_index
                    ]
                )
            ),
            "mean_percent": (
                float(
                    np.mean(
                        percent_values
                    )
                )
            ),
            "standard_deviation_percent": (
                float(
                    np.std(
                        percent_values,
                        ddof=1,
                    )
                )
            ),
            "mean_absolute_annual_change_pp": (
                mean_absolute_annual_change
            ),
        }
    )


trend_summary_df = pd.DataFrame(
    trend_records
)


# =============================================================================
# 11. PERIOD-STATUS SUMMARY
# =============================================================================

period_status_records = []


for year in EXPECTED_YEARS:

    row = (
        annual_df[
            annual_df[
                "year"
            ]
            ==
            year
        ]
        .iloc[
            0
        ]
    )


    period_status_records.append(
        {
            "year": (
                year
            ),
            "period_status": (
                row[
                    "period_status"
                ]
            ),
            "included_in_core_trend": (
                bool(
                    row[
                        "included_in_core_trend"
                    ]
                )
            ),
            "green_percent": (
                float(
                    row[
                        "green_percent"
                    ]
                )
            ),
            "built_percent": (
                float(
                    row[
                        "built_percent"
                    ]
                )
            ),
            "interpretation": (
                "Core stabilized retrospective year"
                if (
                    CORE_START_YEAR
                    <=
                    year
                    <=
                    CORE_END_YEAR
                )
                else
                (
                    "Provisional partial-year context only"
                    if year
                    ==
                    2026
                    else
                    "Raw temporal edge-year context only"
                )
            ),
        }
    )


period_status_df = pd.DataFrame(
    period_status_records
)


# =============================================================================
# 12. GREEN-COVER KEY SUMMARY
# =============================================================================

green_summary = (
    trend_summary_df[
        trend_summary_df[
            "class_id"
        ]
        ==
        1
    ]
    .iloc[
        0
    ]
)


built_summary = (
    trend_summary_df[
        trend_summary_df[
            "class_id"
        ]
        ==
        2
    ]
    .iloc[
        0
    ]
)


green_key_summary_df = pd.DataFrame(
    [
        {
            "core_period": (
                f"{CORE_START_YEAR}-{CORE_END_YEAR}"
            ),
            "primary_trend_measure": (
                "LULC class percentage"
            ),
            "green_start_percent": (
                green_summary[
                    "start_percent"
                ]
            ),
            "green_end_percent": (
                green_summary[
                    "end_percent"
                ]
            ),
            "green_net_change_percent_points": (
                green_summary[
                    "net_change_percent_points"
                ]
            ),
            "green_net_area_change_km2": (
                green_summary[
                    "net_area_change_km2"
                ]
            ),
            "green_sen_slope_pp_per_year": (
                green_summary[
                    "sen_slope_percent_points_per_year"
                ]
            ),
            "green_trend_direction": (
                green_summary[
                    "sen_trend_direction"
                ]
            ),
            "built_start_percent": (
                built_summary[
                    "start_percent"
                ]
            ),
            "built_end_percent": (
                built_summary[
                    "end_percent"
                ]
            ),
            "built_net_change_percent_points": (
                built_summary[
                    "net_change_percent_points"
                ]
            ),
            "built_net_area_change_km2": (
                built_summary[
                    "net_area_change_km2"
                ]
            ),
            "built_sen_slope_pp_per_year": (
                built_summary[
                    "sen_slope_percent_points_per_year"
                ]
            ),
            "built_trend_direction": (
                built_summary[
                    "sen_trend_direction"
                ]
            ),
            "year_2010_used_in_core_trend": (
                False
            ),
            "year_2025_used_in_core_trend": (
                False
            ),
            "year_2026_used_in_core_trend": (
                False
            ),
            "year_2026_provisional": (
                True
            ),
            "pixel_level_persistent_transition_analyzed": (
                False
            ),
        }
    ]
)


# =============================================================================
# 13. QA
# =============================================================================

class_percent_totals = (
    annual_df[
        [
            "green_percent",
            "built_percent",
            "open_bare_percent",
            "water_percent",
        ]
    ]
    .sum(
        axis=1
    )
)


annual_percent_totals_valid = bool(
    np.allclose(
        class_percent_totals.to_numpy(),
        100.0,
        atol=1e-6,
    )
)


all_core_years_stabilized = bool(
    analysis_df[
        analysis_df[
            "year"
        ].between(
            CORE_START_YEAR,
            CORE_END_YEAR,
        )
    ][
        "stabilization_applied"
    ].all()
)


year_2010_excluded = bool(
    ~annual_df.loc[
        annual_df[
            "year"
        ]
        ==
        2010,
        "included_in_core_trend",
    ].iloc[
        0
    ]
)


year_2025_excluded = bool(
    ~annual_df.loc[
        annual_df[
            "year"
        ]
        ==
        2025,
        "included_in_core_trend",
    ].iloc[
        0
    ]
)


year_2026_excluded = bool(
    ~annual_df.loc[
        annual_df[
            "year"
        ]
        ==
        2026,
        "included_in_core_trend",
    ].iloc[
        0
    ]
)


overall_status = (
    "PASS"
    if all(
        [
            annual_percent_totals_valid,
            all_core_years_stabilized,
            year_2010_excluded,
            year_2025_excluded,
            year_2026_excluded,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 14. SAVE OUTPUTS
# =============================================================================

annual_df.to_csv(
    ANNUAL_TRAJECTORY_CSV,
    index=False,
)


trend_summary_df.to_csv(
    CORE_TREND_SUMMARY_CSV,
    index=False,
)


period_status_df.to_csv(
    PERIOD_STATUS_CSV,
    index=False,
)


green_key_summary_df.to_csv(
    GREEN_KEY_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 15. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("CORE 2011-2024 LULC TREND SUMMARY")
print("=" * 100)


print(
    trend_summary_df[
        [
            "class_id",
            "class_name",
            "start_percent",
            "end_percent",
            "net_change_percent_points",
            "net_area_change_km2",
            "sen_slope_percent_points_per_year",
            "sen_trend_direction",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("ANNUAL GREEN / BUILT TRAJECTORY")
print("=" * 100)


print(
    annual_df[
        [
            "year",
            "period_status",
            "green_percent",
            "built_percent",
            "green_area_km2",
            "built_area_km2",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("GREEN-COVER KEY RESULT")
print("=" * 100)


print(
    f"Core period                    : "
    f"{CORE_START_YEAR}-{CORE_END_YEAR}"
)

print(
    f"Green start                    : "
    f"{float(green_summary['start_percent']):.3f}%"
)

print(
    f"Green end                      : "
    f"{float(green_summary['end_percent']):.3f}%"
)

print(
    f"Green net change               : "
    f"{float(green_summary['net_change_percent_points']):+.3f} percentage points"
)

print(
    f"Green net area change          : "
    f"{float(green_summary['net_area_change_km2']):+.3f} km²"
)

print(
    f"Green Sen slope                : "
    f"{float(green_summary['sen_slope_percent_points_per_year']):+.4f} pp/year"
)

print(
    f"Green trend direction          : "
    f"{green_summary['sen_trend_direction']}"
)

print(
    f"Built-up start                 : "
    f"{float(built_summary['start_percent']):.3f}%"
)

print(
    f"Built-up end                   : "
    f"{float(built_summary['end_percent']):.3f}%"
)

print(
    f"Built-up net change            : "
    f"{float(built_summary['net_change_percent_points']):+.3f} percentage points"
)

print(
    f"Built-up Sen slope             : "
    f"{float(built_summary['sen_slope_percent_points_per_year']):+.4f} pp/year"
)

print(
    f"Built-up trend direction       : "
    f"{built_summary['sen_trend_direction']}"
)


print("\n" + "=" * 100)
print("STEP 10A QA")
print("=" * 100)


print(
    f"Annual class totals = 100%     : "
    f"{annual_percent_totals_valid}"
)

print(
    f"All 2011-2024 maps stabilized  : "
    f"{all_core_years_stabilized}"
)

print(
    f"2010 excluded from core trend  : "
    f"{year_2010_excluded}"
)

print(
    f"2025 excluded from core trend  : "
    f"{year_2025_excluded}"
)

print(
    f"2026 excluded from core trend  : "
    f"{year_2026_excluded}"
)

print(
    f"2026 provisional               : "
    f"True"
)

print(
    f"Overall Step 10A status        : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(ANNUAL_TRAJECTORY_CSV)}"
)

print(
    f" - {relative_path(CORE_TREND_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(PERIOD_STATUS_CSV)}"
)

print(
    f" - {relative_path(GREEN_KEY_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. The formal retrospective trend period is 2011-2024."
)

print(
    "2. 2010 and 2025 are retained only as raw edge-year context."
)

print(
    "3. 2026 is provisional and excluded from formal historical trend estimation."
)

print(
    "4. Sen slope describes robust long-term direction; it does not prove that every annual change is real."
)

print(
    "5. Pixel-level persistent Green-to-Built transition mapping has NOT yet been performed."
)

print(
    "6. That spatial transition analysis will be the next step."
)

print("=" * 100)