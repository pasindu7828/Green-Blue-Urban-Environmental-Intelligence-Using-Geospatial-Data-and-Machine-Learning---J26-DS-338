from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part1_historical_change"
)

INPUT_CSV = (
    PROCESSED_DIR
    / "wetland_obs_rainfall.csv"
)

OUTPUT_CSV = (
    PROCESSED_DIR
    / "historical_wetland_change_pairs.csv"
)


# ============================================================
# HELPER
# ============================================================

def relative_difference(a, b):
    """
    Symmetric relative difference.

    Returns the absolute difference as a proportion
    of the mean magnitude of the two values.
    """

    if pd.isna(a) or pd.isna(b):
        return np.nan

    mean_value = (abs(a) + abs(b)) / 2.0

    if mean_value == 0:
        return 0.0

    return abs(b - a) / mean_value


# ============================================================
# MAIN
# ============================================================

def main():

    observations = pd.read_csv(
        INPUT_CSV
    )

    observations[
        "observation_date"
    ] = pd.to_datetime(
        observations[
            "observation_date"
        ]
    )

    observations = (
        observations
        .sort_values(
            [
                "wetland_id",
                "observation_date",
            ]
        )
        .reset_index(drop=True)
    )

    rows = []

    # --------------------------------------------------------
    # BUILD CONSECUTIVE COMPARISONS WITHIN EACH WETLAND
    # --------------------------------------------------------

    for wetland_id, group in observations.groupby(
        "wetland_id"
    ):

        group = (
            group
            .sort_values(
                "observation_date"
            )
            .reset_index(drop=True)
        )

        for i in range(
            len(group) - 1
        ):

            earlier = group.iloc[i]
            later = group.iloc[i + 1]

            start_date = earlier[
                "observation_date"
            ]

            end_date = later[
                "observation_date"
            ]

            interval_days = (
                end_date - start_date
            ).days

            interval_years = (
                interval_days / 365.25
            )

            # ------------------------------------------------
            # AREA CHANGE
            # ------------------------------------------------

            start_area = earlier[
                "raw_total_area_ha"
            ]

            end_area = later[
                "raw_total_area_ha"
            ]

            area_change_ha = (
                end_area - start_area
            )

            if start_area > 0:

                area_change_percent = (
                    area_change_ha
                    / start_area
                    * 100.0
                )

            else:
                area_change_percent = np.nan

            if interval_years > 0:

                annual_area_change_ha = (
                    area_change_ha
                    / interval_years
                )

                annual_area_change_percent = (
                    area_change_percent
                    / interval_years
                )

            else:

                annual_area_change_ha = np.nan
                annual_area_change_percent = np.nan

            # ------------------------------------------------
            # FRAGMENTATION CHANGE
            # ------------------------------------------------

            patch_count_change = (
                later[
                    "analysis_patch_count"
                ]
                -
                earlier[
                    "analysis_patch_count"
                ]
            )

            lpi_change = (
                later[
                    "lpi_percent"
                ]
                -
                earlier[
                    "lpi_percent"
                ]
            )

            patch_density_change = (
                later[
                    "patch_density_per_ha"
                ]
                -
                earlier[
                    "patch_density_per_ha"
                ]
            )

            # ------------------------------------------------
            # SEASON / RAINFALL CONTEXT
            # ------------------------------------------------

            same_season = (
                earlier["season"]
                == later["season"]
            )

            rain_1d_diff_mm = abs(
                later["rain_1d_mm"]
                -
                earlier["rain_1d_mm"]
            )

            rain_7d_diff_mm = abs(
                later["rain_7d_mm"]
                -
                earlier["rain_7d_mm"]
            )

            rain_30d_diff_mm = abs(
                later["rain_30d_mm"]
                -
                earlier["rain_30d_mm"]
            )

            rain_7d_relative_diff = (
                relative_difference(
                    earlier["rain_7d_mm"],
                    later["rain_7d_mm"],
                )
            )

            rain_30d_relative_diff = (
                relative_difference(
                    earlier["rain_30d_mm"],
                    later["rain_30d_mm"],
                )
            )

            # ------------------------------------------------
            # SIMPLE DIRECTION LABEL
            # ------------------------------------------------

            if area_change_ha < 0:
                area_direction = "Decrease"

            elif area_change_ha > 0:
                area_direction = "Increase"

            else:
                area_direction = "No change"

            rows.append(
                {
                    "wetland_id":
                        wetland_id,

                    "start_date":
                        start_date.strftime(
                            "%Y-%m-%d"
                        ),

                    "end_date":
                        end_date.strftime(
                            "%Y-%m-%d"
                        ),

                    "interval_days":
                        interval_days,

                    "interval_years":
                        interval_years,

                    "start_season":
                        earlier["season"],

                    "end_season":
                        later["season"],

                    "same_season":
                        same_season,

                    # Area
                    "start_area_ha":
                        start_area,

                    "end_area_ha":
                        end_area,

                    "area_change_ha":
                        area_change_ha,

                    "area_change_percent":
                        area_change_percent,

                    "annual_area_change_ha":
                        annual_area_change_ha,

                    "annual_area_change_percent":
                        annual_area_change_percent,

                    "area_direction":
                        area_direction,

                    # Fragmentation
                    "start_patch_count":
                        earlier[
                            "analysis_patch_count"
                        ],

                    "end_patch_count":
                        later[
                            "analysis_patch_count"
                        ],

                    "patch_count_change":
                        patch_count_change,

                    "start_lpi_percent":
                        earlier[
                            "lpi_percent"
                        ],

                    "end_lpi_percent":
                        later[
                            "lpi_percent"
                        ],

                    "lpi_change":
                        lpi_change,

                    "start_patch_density":
                        earlier[
                            "patch_density_per_ha"
                        ],

                    "end_patch_density":
                        later[
                            "patch_density_per_ha"
                        ],

                    "patch_density_change":
                        patch_density_change,

                    # Rainfall
                    "start_rain_1d_mm":
                        earlier[
                            "rain_1d_mm"
                        ],

                    "end_rain_1d_mm":
                        later[
                            "rain_1d_mm"
                        ],

                    "rain_1d_diff_mm":
                        rain_1d_diff_mm,

                    "start_rain_7d_mm":
                        earlier[
                            "rain_7d_mm"
                        ],

                    "end_rain_7d_mm":
                        later[
                            "rain_7d_mm"
                        ],

                    "rain_7d_diff_mm":
                        rain_7d_diff_mm,

                    "rain_7d_relative_diff":
                        rain_7d_relative_diff,

                    "start_rain_30d_mm":
                        earlier[
                            "rain_30d_mm"
                        ],

                    "end_rain_30d_mm":
                        later[
                            "rain_30d_mm"
                        ],

                    "rain_30d_diff_mm":
                        rain_30d_diff_mm,

                    "rain_30d_relative_diff":
                        rain_30d_relative_diff,
                }
            )

    comparisons = pd.DataFrame(
        rows
    )

    comparisons.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========================================")
    print("HISTORICAL WETLAND CHANGE PAIRS")
    print("========================================")

    print(
        f"Wetlands represented: "
        f"{comparisons['wetland_id'].nunique()}"
    )

    print(
        f"Consecutive comparison pairs: "
        f"{len(comparisons)}"
    )

    print(
        f"Same-season comparisons: "
        f"{comparisons['same_season'].sum()}"
    )

    print(
        f"Different-season comparisons: "
        f"{(~comparisons['same_season']).sum()}"
    )

    print("\nArea direction:")

    print(
        comparisons[
            "area_direction"
        ]
        .value_counts()
        .to_string()
    )

    print("\nArea change range:")

    print(
        f"Minimum: "
        f"{comparisons['area_change_percent'].min():.2f}%"
    )

    print(
        f"Maximum: "
        f"{comparisons['area_change_percent'].max():.2f}%"
    )

    print("\nRainfall difference ranges:")

    print(
        f"7-day absolute difference: "
        f"{comparisons['rain_7d_diff_mm'].min():.2f} - "
        f"{comparisons['rain_7d_diff_mm'].max():.2f} mm"
    )

    print(
        f"30-day absolute difference: "
        f"{comparisons['rain_30d_diff_mm'].min():.2f} - "
        f"{comparisons['rain_30d_diff_mm'].max():.2f} mm"
    )

    print("\nOutput:")
    print(OUTPUT_CSV)

    print(
        "\nNOTE:"
        "\nThese are raw consecutive comparisons."
        "\nNo pair has yet been classified as "
        "seasonally/rainfall comparable."
        "\nThe next stage will assess comparison quality."
    )


if __name__ == "__main__":
    main()