from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


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

PAIR_INPUT = (
    PROCESSED_DIR
    / "historical_wetland_change_pairs_quality.csv"
)

OBS_INPUT = (
    PROCESSED_DIR
    / "wetland_obs_rainfall.csv"
)

SPATIAL_INPUT = (
    PROCESSED_DIR
    / "wetland_obs_rainfall_wgs84.geojson"
)

SUMMARY_OUTPUT = (
    PROCESSED_DIR
    / "wetland_change_summary.csv"
)

LATEST_MAP_OUTPUT = (
    PROCESSED_DIR
    / "wetland_latest_status_wgs84.geojson"
)


# ============================================================
# HELPERS
# ============================================================

def parse_bool(series):
    """
    Safely convert CSV True/False values to boolean.
    """

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def percent_change(start_value, end_value):

    if pd.isna(start_value) or pd.isna(end_value):
        return np.nan

    if start_value == 0:
        return np.nan

    return (
        (end_value - start_value)
        / start_value
        * 100.0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    pairs = pd.read_csv(
        PAIR_INPUT
    )

    observations = pd.read_csv(
        OBS_INPUT
    )

    # --------------------------------------------------------
    # CLEAN TYPES
    # --------------------------------------------------------

    pairs["manual_qa_required"] = parse_bool(
        pairs["manual_qa_required"]
    )

    observations["observation_date"] = (
        pd.to_datetime(
            observations["observation_date"]
        )
    )

    # --------------------------------------------------------
    # DEFINE CORE / TRUSTED COMPARISONS
    # --------------------------------------------------------

    trusted = pairs[
        (
            pairs["comparison_quality"]
            .isin(
                [
                    "High",
                    "Moderate",
                ]
            )
        )
        &
        (
            ~pairs["manual_qa_required"]
        )
    ].copy()

    print("\n========================================")
    print("WETLAND HISTORICAL CHANGE SUMMARY")
    print("========================================")

    print(
        f"All comparison pairs: "
        f"{len(pairs)}"
    )

    print(
        f"High/Moderate pairs: "
        f"{pairs['comparison_quality'].isin(['High', 'Moderate']).sum()}"
    )

    print(
        f"Core trusted pairs after QA screening: "
        f"{len(trusted)}"
    )

    # --------------------------------------------------------
    # CREATE ONE SUMMARY RECORD PER WETLAND
    # --------------------------------------------------------

    summary_rows = []

    wetland_ids = sorted(
        observations[
            "wetland_id"
        ].dropna().unique()
    )

    for wetland_id in wetland_ids:

        obs = (
            observations[
                observations["wetland_id"]
                == wetland_id
            ]
            .sort_values(
                "observation_date"
            )
            .copy()
        )

        wetland_pairs = pairs[
            pairs["wetland_id"]
            == wetland_id
        ].copy()

        trusted_pairs = trusted[
            trusted["wetland_id"]
            == wetland_id
        ].copy()

        earliest = obs.iloc[0]
        latest = obs.iloc[-1]

        earliest_area = (
            earliest["raw_total_area_ha"]
        )

        latest_area = (
            latest["raw_total_area_ha"]
        )

        endpoint_change_ha = (
            latest_area
            - earliest_area
        )

        endpoint_change_percent = (
            percent_change(
                earliest_area,
                latest_area,
            )
        )

        high_count = int(
            (
                wetland_pairs[
                    "comparison_quality"
                ]
                == "High"
            ).sum()
        )

        moderate_count = int(
            (
                wetland_pairs[
                    "comparison_quality"
                ]
                == "Moderate"
            ).sum()
        )

        low_count = int(
            (
                wetland_pairs[
                    "comparison_quality"
                ]
                == "Low"
            ).sum()
        )

        trusted_count = len(
            trusted_pairs
        )

        # ----------------------------------------------------
        # TRUSTED CHANGE EVIDENCE
        # ----------------------------------------------------

        if trusted_count > 0:

            decrease_count = int(
                (
                    trusted_pairs[
                        "area_change_ha"
                    ] < 0
                ).sum()
            )

            increase_count = int(
                (
                    trusted_pairs[
                        "area_change_ha"
                    ] > 0
                ).sum()
            )

            median_change_percent = (
                trusted_pairs[
                    "area_change_percent"
                ].median()
            )

            median_annual_change_percent = (
                trusted_pairs[
                    "annual_area_change_percent"
                ].median()
            )

            median_lpi_change = (
                trusted_pairs[
                    "lpi_change"
                ].median()
            )

            median_patch_count_change = (
                trusted_pairs[
                    "patch_count_change"
                ].median()
            )

            median_patch_density_change = (
                trusted_pairs[
                    "patch_density_change"
                ].median()
            )

        else:

            decrease_count = 0
            increase_count = 0

            median_change_percent = np.nan
            median_annual_change_percent = np.nan
            median_lpi_change = np.nan
            median_patch_count_change = np.nan
            median_patch_density_change = np.nan

        # ----------------------------------------------------
        # LONG-TERM CHANGE INTERPRETATION
        #
        # Require at least two trusted comparisons.
        # Direction must be supported both by:
        #   1. majority of trusted intervals
        #   2. median annual change direction
        #
        # Otherwise report mixed/variable evidence.
        # ----------------------------------------------------

        if trusted_count < 2:

            change_status = (
                "Insufficient comparable evidence"
            )

        elif (
            decrease_count > increase_count
            and
            median_annual_change_percent < 0
        ):

            change_status = (
                "Predominantly decreasing"
            )

        elif (
            increase_count > decrease_count
            and
            median_annual_change_percent > 0
        ):

            change_status = (
                "Predominantly increasing"
            )

        else:

            change_status = (
                "Mixed / variable"
            )

        # ----------------------------------------------------
        # FRAGMENTATION INTERPRETATION
        # ----------------------------------------------------

        if trusted_count < 2:

            fragmentation_status = (
                "Insufficient comparable evidence"
            )

        elif (
            median_lpi_change < 0
            and
            median_patch_count_change > 0
        ):

            fragmentation_status = (
                "Increasing fragmentation"
            )

        elif (
            median_lpi_change > 0
            and
            median_patch_count_change < 0
        ):

            fragmentation_status = (
                "Increasing consolidation"
            )

        else:

            fragmentation_status = (
                "Mixed / limited fragmentation change"
            )

        # ----------------------------------------------------
        # CURRENT / LATEST CONDITION
        # ----------------------------------------------------

        summary_rows.append(
            {
                "wetland_id":
                    wetland_id,

                "first_observation_date":
                    earliest[
                        "observation_date"
                    ].strftime(
                        "%Y-%m-%d"
                    ),

                "latest_observation_date":
                    latest[
                        "observation_date"
                    ].strftime(
                        "%Y-%m-%d"
                    ),

                "observation_count":
                    len(obs),

                "all_comparison_count":
                    len(wetland_pairs),

                "high_quality_pairs":
                    high_count,

                "moderate_quality_pairs":
                    moderate_count,

                "low_quality_pairs":
                    low_count,

                "trusted_pair_count":
                    trusted_count,

                "trusted_decrease_count":
                    decrease_count,

                "trusted_increase_count":
                    increase_count,

                "first_area_ha":
                    earliest_area,

                "latest_area_ha":
                    latest_area,

                # Descriptive endpoint only.
                "raw_endpoint_change_ha":
                    endpoint_change_ha,

                "raw_endpoint_change_percent":
                    endpoint_change_percent,

                "median_trusted_change_percent":
                    median_change_percent,

                "median_trusted_annual_change_percent":
                    median_annual_change_percent,

                "median_trusted_lpi_change":
                    median_lpi_change,

                "median_trusted_patch_count_change":
                    median_patch_count_change,

                "median_trusted_patch_density_change":
                    median_patch_density_change,

                "change_status":
                    change_status,

                "fragmentation_status":
                    fragmentation_status,

                "latest_patch_count":
                    latest[
                        "analysis_patch_count"
                    ],

                "latest_lpi_percent":
                    latest[
                        "lpi_percent"
                    ],

                "latest_season":
                    latest[
                        "season"
                    ],

                "latest_rain_7d_mm":
                    latest[
                        "rain_7d_mm"
                    ],

                "latest_rain_30d_mm":
                    latest[
                        "rain_30d_mm"
                    ],
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # SAVE SUMMARY TABLE
    # --------------------------------------------------------

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # CREATE LATEST WETLAND MAP FOR FRONTEND
    # --------------------------------------------------------

    spatial = gpd.read_file(
        SPATIAL_INPUT
    )

    spatial["observation_date"] = (
        pd.to_datetime(
            spatial["observation_date"]
        )
    )

    latest_indices = (
        spatial
        .groupby(
            "wetland_id"
        )[
            "observation_date"
        ]
        .idxmax()
    )

    latest_spatial = (
        spatial
        .loc[
            latest_indices
        ]
        .copy()
    )

    latest_spatial[
        "observation_date"
    ] = latest_spatial[
        "observation_date"
    ].dt.strftime(
        "%Y-%m-%d"
    )

    latest_spatial = (
        latest_spatial
        .merge(
            summary,
            on="wetland_id",
            how="left",
            validate="one_to_one",
            suffixes=(
                "",
                "_summary",
            ),
        )
    )

    latest_spatial.to_file(
        LATEST_MAP_OUTPUT,
        driver="GeoJSON",
    )

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\nWetland change status:"
    )

    print(
        summary[
            "change_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nFragmentation status:"
    )

    print(
        summary[
            "fragmentation_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nTrusted evidence per wetland:"
    )

    print(
        summary[
            [
                "wetland_id",
                "trusted_pair_count",
                "trusted_decrease_count",
                "trusted_increase_count",
                "change_status",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print("\nOutputs:")

    print(
        SUMMARY_OUTPUT
    )

    print(
        LATEST_MAP_OUTPUT
    )

    print(
        "\nIMPORTANT:"
        "\n'Raw endpoint change' is descriptive only."
        "\nThe primary change status uses environmentally "
        "comparable, non-QA-flagged intervals."
        "\nPredominantly decreasing does not by itself "
        "prove a specific cause such as urban development."
    )


if __name__ == "__main__":
    main()