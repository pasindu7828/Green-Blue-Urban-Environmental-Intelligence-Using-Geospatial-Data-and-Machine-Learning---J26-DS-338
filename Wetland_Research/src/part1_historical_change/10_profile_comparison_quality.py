from pathlib import Path

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

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "part1_historical_change"
)

INPUT_CSV = (
    PROCESSED_DIR
    / "historical_wetland_change_pairs.csv"
)

PROFILE_OUTPUT = (
    INTERIM_DIR
    / "comparison_quality_profile.csv"
)

DATE_PAIR_OUTPUT = (
    INTERIM_DIR
    / "unique_date_pair_rainfall_profile.csv"
)

EXTREME_OUTPUT = (
    INTERIM_DIR
    / "extreme_area_change_pairs.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    pairs = pd.read_csv(
        INPUT_CSV
    )

    # --------------------------------------------------------
    # PREPARE BASIC QA VARIABLES
    # --------------------------------------------------------

    pairs["abs_area_change_percent"] = (
        pairs["area_change_percent"].abs()
    )

    pairs["small_start_area"] = (
        pairs["start_area_ha"] < 0.05
    )

    # --------------------------------------------------------
    # UNIQUE DATE PAIRS
    #
    # Many wetlands can share exactly the same imagery dates.
    # We do not want those repeated dates to artificially
    # dominate the rainfall threshold calculation.
    # --------------------------------------------------------

    unique_date_pairs = (
        pairs
        .drop_duplicates(
            subset=[
                "start_date",
                "end_date",
            ]
        )
        .copy()
    )

    unique_date_pairs.to_csv(
        DATE_PAIR_OUTPUT,
        index=False,
    )

    same_season = unique_date_pairs[
        unique_date_pairs["same_season"] == True
    ].copy()

    # --------------------------------------------------------
    # EMPIRICAL RAINFALL DIFFERENCE QUANTILES
    # --------------------------------------------------------

    rainfall_fields = [
        "rain_7d_diff_mm",
        "rain_30d_diff_mm",
        "rain_7d_relative_diff",
        "rain_30d_relative_diff",
    ]

    quantiles = [
        0.25,
        0.50,
        0.75,
        0.90,
    ]

    profile_rows = []

    for field in rainfall_fields:

        for q in quantiles:

            profile_rows.append(
                {
                    "subset":
                        "same_season_unique_date_pairs",

                    "variable":
                        field,

                    "quantile":
                        q,

                    "value":
                        same_season[
                            field
                        ].quantile(q),
                }
            )

    profile = pd.DataFrame(
        profile_rows
    )

    profile.to_csv(
        PROFILE_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # EXTREME AREA CHANGE QA
    # --------------------------------------------------------

    extreme = (
        pairs
        .sort_values(
            "abs_area_change_percent",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    extreme.to_csv(
        EXTREME_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # TERMINAL REPORT
    # --------------------------------------------------------

    print("\n========================================")
    print("COMPARISON QUALITY PROFILE")
    print("========================================")

    print(
        f"All wetland comparison pairs: "
        f"{len(pairs)}"
    )

    print(
        f"Unique date pairs: "
        f"{len(unique_date_pairs)}"
    )

    print(
        f"Unique same-season date pairs: "
        f"{len(same_season)}"
    )

    print("\nSame-season rainfall difference quantiles:")

    for field in rainfall_fields:

        print(
            f"\n{field}"
        )

        values = (
            same_season[
                field
            ]
            .quantile(
                quantiles
            )
        )

        for q, value in values.items():

            print(
                f"  Q{int(q * 100)}: "
                f"{value:.4f}"
            )

    # --------------------------------------------------------
    # INTERVAL PROFILE
    # --------------------------------------------------------

    print("\nComparison interval (years):")

    print(
        pairs[
            "interval_years"
        ]
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
            ]
        )
        .to_string()
    )

    # --------------------------------------------------------
    # SMALL STARTING AREA QA
    # --------------------------------------------------------

    print("\nSmall starting-area comparisons:")

    print(
        "< 0.05 ha:",
        int(
            pairs[
                "small_start_area"
            ].sum()
        )
    )

    # --------------------------------------------------------
    # EXTREME CHANGE QA
    # --------------------------------------------------------

    print(
        "\nTop 20 absolute area changes:"
    )

    print(
        extreme[
            [
                "wetland_id",
                "start_date",
                "end_date",
                "start_area_ha",
                "end_area_ha",
                "area_change_percent",
                "same_season",
                "rain_7d_diff_mm",
                "rain_30d_diff_mm",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print("\nOutputs:")

    print(PROFILE_OUTPUT)
    print(DATE_PAIR_OUTPUT)
    print(EXTREME_OUTPUT)

    print(
        "\nNOTE:"
        "\nNo rainfall-comparability thresholds "
        "have been applied yet."
        "\nThis script only profiles the empirical "
        "distribution so the next-stage thresholds "
        "can be data-informed."
    )


if __name__ == "__main__":
    main()