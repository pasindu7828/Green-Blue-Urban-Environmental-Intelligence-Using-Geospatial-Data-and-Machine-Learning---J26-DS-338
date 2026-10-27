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

OUTPUT_CSV = (
    PROCESSED_DIR
    / "historical_wetland_change_pairs_quality.csv"
)

QA_OUTPUT = (
    INTERIM_DIR
    / "historical_change_manual_qa.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    pairs = pd.read_csv(
        INPUT_CSV
    )

    # --------------------------------------------------------
    # UNIQUE SAME-SEASON DATE PAIRS
    #
    # Rainfall thresholds are calculated from unique date
    # combinations so dates shared by several wetlands do
    # not receive extra weight.
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

    same_season_dates = (
        unique_date_pairs[
            unique_date_pairs[
                "same_season"
            ] == True
        ]
        .copy()
    )

    # --------------------------------------------------------
    # DATA-INFORMED RAINFALL THRESHOLDS
    # --------------------------------------------------------

    rain7_q50 = (
        same_season_dates[
            "rain_7d_diff_mm"
        ].quantile(0.50)
    )

    rain7_q75 = (
        same_season_dates[
            "rain_7d_diff_mm"
        ].quantile(0.75)
    )

    rain30_q50 = (
        same_season_dates[
            "rain_30d_diff_mm"
        ].quantile(0.50)
    )

    rain30_q75 = (
        same_season_dates[
            "rain_30d_diff_mm"
        ].quantile(0.75)
    )

    # --------------------------------------------------------
    # RAINFALL SIMILARITY CLASS
    # --------------------------------------------------------

    def rainfall_similarity(row):

        if (
            row["rain_7d_diff_mm"]
            <= rain7_q50
            and
            row["rain_30d_diff_mm"]
            <= rain30_q50
        ):
            return "High"

        if (
            row["rain_7d_diff_mm"]
            <= rain7_q75
            and
            row["rain_30d_diff_mm"]
            <= rain30_q75
        ):
            return "Moderate"

        return "Low"

    pairs[
        "rainfall_similarity"
    ] = pairs.apply(
        rainfall_similarity,
        axis=1,
    )

    # --------------------------------------------------------
    # OVERALL ENVIRONMENTAL COMPARABILITY
    # --------------------------------------------------------

    def comparison_quality(row):

        if not row["same_season"]:
            return "Low"

        if (
            row[
                "rainfall_similarity"
            ] == "High"
        ):
            return "High"

        if (
            row[
                "rainfall_similarity"
            ] == "Moderate"
        ):
            return "Moderate"

        return "Low"

    pairs[
        "comparison_quality"
    ] = pairs.apply(
        comparison_quality,
        axis=1,
    )

    # --------------------------------------------------------
    # TEMPORAL CONTEXT
    #
    # This does NOT decide whether change is genuine.
    # It only describes how far apart the images are.
    # --------------------------------------------------------

    def temporal_context(days):

        if days < 90:
            return "Short-term"

        if days < 365:
            return "Within-year"

        return "Interannual"

    pairs[
        "temporal_context"
    ] = pairs[
        "interval_days"
    ].apply(
        temporal_context
    )

    # --------------------------------------------------------
    # EXTREME CHANGE QA
    #
    # Use dataset-derived percentiles rather than manually
    # inventing an extreme-change threshold.
    # --------------------------------------------------------

    pairs[
        "abs_area_change_percent"
    ] = pairs[
        "area_change_percent"
    ].abs()

    extreme_threshold = (
        pairs[
            "abs_area_change_percent"
        ].quantile(0.95)
    )

    pairs[
        "extreme_change_flag"
    ] = (
        pairs[
            "abs_area_change_percent"
        ]
        >= extreme_threshold
    )

    # --------------------------------------------------------
    # SMALL STARTING-AREA QA
    #
    # Again use the lower 10th percentile from this dataset.
    # --------------------------------------------------------

    small_area_threshold = (
        pairs[
            "start_area_ha"
        ].quantile(0.10)
    )

    pairs[
        "small_start_area_flag"
    ] = (
        pairs[
            "start_area_ha"
        ]
        <= small_area_threshold
    )

    # --------------------------------------------------------
    # MANUAL REVIEW FLAG
    #
    # These cases are not automatically rejected.
    # They simply require visual review before interpretation.
    # --------------------------------------------------------

    pairs[
        "manual_qa_required"
    ] = (
        pairs[
            "extreme_change_flag"
        ]
        |
        pairs[
            "small_start_area_flag"
        ]
        |
        (
            pairs[
                "temporal_context"
            ] == "Short-term"
        )
    )

    # --------------------------------------------------------
    # ADD THRESHOLDS TO OUTPUT FOR TRANSPARENCY
    # --------------------------------------------------------

    pairs[
        "rain7_q50_threshold_mm"
    ] = rain7_q50

    pairs[
        "rain7_q75_threshold_mm"
    ] = rain7_q75

    pairs[
        "rain30_q50_threshold_mm"
    ] = rain30_q50

    pairs[
        "rain30_q75_threshold_mm"
    ] = rain30_q75

    pairs[
        "extreme_change_threshold_percent"
    ] = extreme_threshold

    pairs[
        "small_start_area_threshold_ha"
    ] = small_area_threshold

    # --------------------------------------------------------
    # SAVE FULL OUTPUT
    # --------------------------------------------------------

    pairs.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE MANUAL QA SUBSET
    # --------------------------------------------------------

    qa = pairs[
        pairs[
            "manual_qa_required"
        ]
    ].copy()

    qa.to_csv(
        QA_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\n========================================")
    print("HISTORICAL COMPARISON QUALITY")
    print("========================================")

    print("\nRainfall thresholds derived from data:")

    print(
        f"7-day Q50:  "
        f"{rain7_q50:.4f} mm"
    )

    print(
        f"7-day Q75:  "
        f"{rain7_q75:.4f} mm"
    )

    print(
        f"30-day Q50: "
        f"{rain30_q50:.4f} mm"
    )

    print(
        f"30-day Q75: "
        f"{rain30_q75:.4f} mm"
    )

    print(
        f"\nExtreme-change threshold "
        f"(95th percentile): "
        f"{extreme_threshold:.2f}%"
    )

    print(
        f"Small starting-area threshold "
        f"(10th percentile): "
        f"{small_area_threshold:.4f} ha"
    )

    print("\nComparison quality:")

    print(
        pairs[
            "comparison_quality"
        ]
        .value_counts()
        .to_string()
    )

    print("\nTemporal context:")

    print(
        pairs[
            "temporal_context"
        ]
        .value_counts()
        .to_string()
    )

    print("\nManual QA:")

    print(
        f"Pairs requiring review: "
        f"{len(qa)} / {len(pairs)}"
    )

    print(
        f"Extreme-change pairs: "
        f"{pairs['extreme_change_flag'].sum()}"
    )

    print(
        f"Small-start-area pairs: "
        f"{pairs['small_start_area_flag'].sum()}"
    )

    print(
        f"Short-term pairs: "
        f"{(pairs['temporal_context'] == 'Short-term').sum()}"
    )

    print("\nOutputs:")

    print(OUTPUT_CSV)
    print(QA_OUTPUT)

    print(
        "\nIMPORTANT:"
        "\nHigh/Moderate/Low represents "
        "environmental comparability, not "
        "proof of degradation."
        "\nFlagged comparisons remain in the "
        "dataset and are not automatically deleted."
    )


if __name__ == "__main__":
    main()