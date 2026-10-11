from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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

OUTPUT_TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "part1"
)

OUTPUT_FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "part1"
)

OUTPUT_TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUTS
# ============================================================

SUMMARY_INPUT = (
    PROCESSED_DIR
    / "wetland_change_summary.csv"
)

PAIR_INPUT = (
    PROCESSED_DIR
    / "historical_wetland_change_pairs_quality.csv"
)

SPATIAL_INPUT = (
    PROCESSED_DIR
    / "trusted_spatial_change_metrics.csv"
)

GN_INPUT = (
    PROCESSED_DIR
    / "wetland_gn_summary.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

FINAL_SUMMARY_OUTPUT = (
    OUTPUT_TABLE_DIR
    / "wetland_backend_summary.csv"
)

VALIDATION_OUTPUT = (
    OUTPUT_TABLE_DIR
    / "backend_validation.csv"
)

PP1_OVERVIEW_OUTPUT = (
    OUTPUT_TABLE_DIR
    / "pp1_backend_overview.txt"
)


# ============================================================
# HELPERS
# ============================================================

def parse_bool(series):

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


def add_validation(
    rows,
    check_name,
    result,
    details,
):

    rows.append(
        {
            "check":
                check_name,

            "status":
                "PASS"
                if result
                else "FAIL",

            "details":
                details,
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("PART 1 BACKEND FINALIZATION")
    print("========================================")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    summary = pd.read_csv(
        SUMMARY_INPUT
    )

    pairs = pd.read_csv(
        PAIR_INPUT
    )

    spatial = pd.read_csv(
        SPATIAL_INPUT
    )

    gn = pd.read_csv(
        GN_INPUT
    )

    pairs[
        "manual_qa_required"
    ] = parse_bool(
        pairs[
            "manual_qa_required"
        ]
    )

    # --------------------------------------------------------
    # REBUILD TRUSTED PAIR SUBSET
    # --------------------------------------------------------

    trusted_pairs = pairs[
        (
            pairs[
                "comparison_quality"
            ].isin(
                [
                    "High",
                    "Moderate",
                ]
            )
        )
        &
        (
            ~pairs[
                "manual_qa_required"
            ]
        )
    ].copy()

    # ========================================================
    # BACKEND VALIDATION
    # ========================================================

    validation_rows = []

    add_validation(
        validation_rows,
        "Known wetland count",
        summary[
            "wetland_id"
        ].nunique() == 25,
        (
            f"{summary['wetland_id'].nunique()} "
            f"wetlands found"
        ),
    )

    add_validation(
        validation_rows,
        "Trusted pair count",
        len(
            trusted_pairs
        ) == 89,
        (
            f"{len(trusted_pairs)} "
            f"trusted pairs found"
        ),
    )

    add_validation(
        validation_rows,
        "Spatial pair count",
        len(
            spatial
        ) == len(
            trusted_pairs
        ),
        (
            f"{len(spatial)} spatial pairs, "
            f"{len(trusted_pairs)} trusted pairs"
        ),
    )

    add_validation(
        validation_rows,
        "GN assignment coverage",
        gn[
            "wetland_id"
        ].nunique() == 25,
        (
            f"{gn['wetland_id'].nunique()} "
            f"wetlands assigned to GN divisions"
        ),
    )

    missing_primary_gn = (
        gn[
            "primary_gn"
        ]
        .isna()
        .sum()
    )

    add_validation(
        validation_rows,
        "Missing primary GN",
        missing_primary_gn == 0,
        (
            f"{missing_primary_gn} "
            f"wetlands missing primary GN"
        ),
    )

    max_identity_error = (
        spatial[
            "geometry_identity_error_m2"
        ].max()
    )

    add_validation(
        validation_rows,
        "Spatial identity error",
        max_identity_error
        < 0.001,
        (
            f"Maximum error = "
            f"{max_identity_error:.9f} m²"
        ),
    )

    duplicate_summary_ids = (
        summary[
            "wetland_id"
        ]
        .duplicated()
        .sum()
    )

    add_validation(
        validation_rows,
        "Duplicate summary IDs",
        duplicate_summary_ids == 0,
        (
            f"{duplicate_summary_ids} "
            f"duplicate wetland IDs"
        ),
    )

    # --------------------------------------------------------
    # VERIFY SAME TRUSTED PAIRS EXIST IN SPATIAL RESULTS
    # --------------------------------------------------------

    trusted_keys = set(
        zip(
            trusted_pairs[
                "wetland_id"
            ].astype(str),

            trusted_pairs[
                "start_date"
            ].astype(str),

            trusted_pairs[
                "end_date"
            ].astype(str),
        )
    )

    spatial_keys = set(
        zip(
            spatial[
                "wetland_id"
            ].astype(str),

            spatial[
                "start_date"
            ].astype(str),

            spatial[
                "end_date"
            ].astype(str),
        )
    )

    missing_spatial_keys = (
        trusted_keys
        - spatial_keys
    )

    extra_spatial_keys = (
        spatial_keys
        - trusted_keys
    )

    add_validation(
        validation_rows,
        "Trusted/spatial pair correspondence",
        (
            len(
                missing_spatial_keys
            ) == 0
            and
            len(
                extra_spatial_keys
            ) == 0
        ),
        (
            f"Missing = "
            f"{len(missing_spatial_keys)}, "
            f"extra = "
            f"{len(extra_spatial_keys)}"
        ),
    )

    validation = pd.DataFrame(
        validation_rows
    )

    validation.to_csv(
        VALIDATION_OUTPUT,
        index=False,
    )

    # ========================================================
    # SPATIAL TRANSITION SUMMARY PER WETLAND
    # ========================================================

    spatial_summary = (
        spatial
        .groupby(
            "wetland_id"
        )
        .agg(
            trusted_spatial_pairs=(
                "wetland_id",
                "size",
            ),

            pairwise_stable_area_sum_ha=(
                "stable_area_ha",
                "sum",
            ),

            pairwise_loss_area_sum_ha=(
                "analysis_loss_area_ha",
                "sum",
            ),

            pairwise_gain_area_sum_ha=(
                "analysis_gain_area_ha",
                "sum",
            ),

            median_loss_percent=(
                "loss_percent_of_start",
                "median",
            ),

            median_gain_percent=(
                "gain_percent_of_start",
                "median",
            ),

            median_turnover_percent=(
                "turnover_percent_of_start",
                "median",
            ),
        )
        .reset_index()
    )

    spatial_summary[
        "pairwise_net_transition_sum_ha"
    ] = (
        spatial_summary[
            "pairwise_gain_area_sum_ha"
        ]
        -
        spatial_summary[
            "pairwise_loss_area_sum_ha"
        ]
    )

    # ========================================================
    # COMBINE FINAL WETLAND BACKEND SUMMARY
    # ========================================================

    final_summary = (
        summary
        .merge(
            gn[
                [
                    "wetland_id",
                    "primary_gn",
                    "primary_gn_overlap_percent",
                    "gn_count",
                    "all_gn_divisions",
                ]
            ],
            on="wetland_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            spatial_summary,
            on="wetland_id",
            how="left",
            validate="one_to_one",
        )
    )

    final_summary.to_csv(
        FINAL_SUMMARY_OUTPUT,
        index=False,
    )

    # ========================================================
    # PP1 SUMMARY STATISTICS
    # ========================================================

    total_raw_loss = (
        spatial[
            "raw_loss_area_ha"
        ].sum()
    )

    total_raw_gain = (
        spatial[
            "raw_gain_area_ha"
        ].sum()
    )

    total_analysis_loss = (
        spatial[
            "analysis_loss_area_ha"
        ].sum()
    )

    total_analysis_gain = (
        spatial[
            "analysis_gain_area_ha"
        ].sum()
    )

    loss_removed_percent = (
        (
            total_raw_loss
            - total_analysis_loss
        )
        /
        total_raw_loss
        * 100.0
    )

    gain_removed_percent = (
        (
            total_raw_gain
            - total_analysis_gain
        )
        /
        total_raw_gain
        * 100.0
    )

    status_counts = (
        summary[
            "change_status"
        ]
        .value_counts()
    )

    fragmentation_counts = (
        summary[
            "fragmentation_status"
        ]
        .value_counts()
    )

    all_validation_passed = (
        validation[
            "status"
        ]
        .eq(
            "PASS"
        )
        .all()
    )

    # ========================================================
    # CHART 1 - CHANGE STATUS
    # ========================================================

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    status_counts.plot(
        kind="bar"
    )

    plt.title(
        "Historical Wetland Change Status"
    )

    plt.xlabel(
        "Change Status"
    )

    plt.ylabel(
        "Number of Wetlands"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE_DIR
        / "change_status.png",
        dpi=300,
    )

    plt.close()

    # ========================================================
    # CHART 2 - FRAGMENTATION STATUS
    # ========================================================

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    fragmentation_counts.plot(
        kind="bar"
    )

    plt.title(
        "Historical Fragmentation Status"
    )

    plt.xlabel(
        "Fragmentation Status"
    )

    plt.ylabel(
        "Number of Wetlands"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE_DIR
        / "fragmentation_status.png",
        dpi=300,
    )

    plt.close()

    # ========================================================
    # CHART 3 - TRUSTED PAIRS
    # ========================================================

    trusted_plot = (
        final_summary
        .sort_values(
            "wetland_id"
        )
    )

    plt.figure(
        figsize=(
            12,
            5,
        )
    )

    plt.bar(
        trusted_plot[
            "wetland_id"
        ],
        trusted_plot[
            "trusted_pair_count"
        ],
    )

    plt.title(
        "Trusted Historical Comparison Pairs by Wetland"
    )

    plt.xlabel(
        "Wetland ID"
    )

    plt.ylabel(
        "Trusted Pair Count"
    )

    plt.xticks(
        rotation=90
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE_DIR
        / "trusted_pairs.png",
        dpi=300,
    )

    plt.close()

    # ========================================================
    # CHART 4 - PAIRWISE LOSS / GAIN
    # ========================================================

    transition_plot = (
        final_summary
        .sort_values(
            "wetland_id"
        )
        .copy()
    )

    plt.figure(
        figsize=(
            13,
            6,
        )
    )

    x = range(
        len(
            transition_plot
        )
    )

    width = 0.4

    plt.bar(
        [
            value - width / 2
            for value in x
        ],
        transition_plot[
            "pairwise_loss_area_sum_ha"
        ],
        width=width,
        label="Pairwise loss",
    )

    plt.bar(
        [
            value + width / 2
            for value in x
        ],
        transition_plot[
            "pairwise_gain_area_sum_ha"
        ],
        width=width,
        label="Pairwise gain",
    )

    plt.title(
        "Trusted Pairwise Wetland Loss and Gain"
    )

    plt.xlabel(
        "Wetland ID"
    )

    plt.ylabel(
        "Transition Area Sum (ha)"
    )

    plt.xticks(
        list(x),
        transition_plot[
            "wetland_id"
        ],
        rotation=90,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE_DIR
        / "loss_gain.png",
        dpi=300,
    )

    plt.close()

    # ========================================================
    # WRITE SIMPLE PP1 BACKEND OVERVIEW
    # ========================================================

    with open(
        PP1_OVERVIEW_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PART 1 - HISTORICAL WETLAND CHANGE\n"
        )

        file.write(
            "==================================\n\n"
        )

        file.write(
            f"Known wetlands analysed: "
            f"{summary['wetland_id'].nunique()}\n"
        )

        file.write(
            f"Total historical observations: "
            f"448\n"
        )

        file.write(
            f"Raw consecutive comparison pairs: "
            f"{len(pairs)}\n"
        )

        file.write(
            f"High/Moderate comparison pairs: "
            f"{pairs['comparison_quality'].isin(['High', 'Moderate']).sum()}\n"
        )

        file.write(
            f"Core trusted comparison pairs: "
            f"{len(trusted_pairs)}\n"
        )

        file.write(
            f"GN divisions containing known wetlands: "
            f"{gn['primary_gn'].nunique()}\n\n"
        )

        file.write(
            "Change status counts:\n"
        )

        file.write(
            status_counts.to_string()
        )

        file.write(
            "\n\nFragmentation status counts:\n"
        )

        file.write(
            fragmentation_counts.to_string()
        )

        file.write(
            "\n\nTrusted pairwise spatial transitions:\n"
        )

        file.write(
            f"Raw loss sum: "
            f"{total_raw_loss:.4f} ha\n"
        )

        file.write(
            f"Raw gain sum: "
            f"{total_raw_gain:.4f} ha\n"
        )

        file.write(
            f"MMU-filtered loss sum: "
            f"{total_analysis_loss:.4f} ha\n"
        )

        file.write(
            f"MMU-filtered gain sum: "
            f"{total_analysis_gain:.4f} ha\n"
        )

        file.write(
            f"Loss removed by MMU: "
            f"{loss_removed_percent:.2f}%\n"
        )

        file.write(
            f"Gain removed by MMU: "
            f"{gain_removed_percent:.2f}%\n"
        )

        file.write(
            "\nIMPORTANT INTERPRETATION:\n"
        )

        file.write(
            "Loss/gain sums are pairwise transition totals "
            "across multiple observation intervals. "
            "They are not unique cumulative land-loss or "
            "land-gain totals for Kaduwela.\n"
        )

        file.write(
            "Historical status describes mapped wetland "
            "boundary behaviour under environmentally "
            "comparable observations. It does not by itself "
            "identify the cause of change.\n"
        )

        file.write(
            "\nBackend validation: "
        )

        file.write(
            "PASS\n"
            if all_validation_passed
            else "CHECK REQUIRED\n"
        )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print("\nBackend validation:")

    print(
        validation.to_string(
            index=False
        )
    )

    print(
        "\nAll validation checks passed:",
        all_validation_passed,
    )

    print(
        "\nFinal change status:"
    )

    print(
        status_counts.to_string()
    )

    print(
        "\nMMU impact on trusted spatial transitions:"
    )

    print(
        f"Loss removed: "
        f"{loss_removed_percent:.2f}%"
    )

    print(
        f"Gain removed: "
        f"{gain_removed_percent:.2f}%"
    )

    print(
        "\nFinal backend outputs:"
    )

    print(
        FINAL_SUMMARY_OUTPUT
    )

    print(
        VALIDATION_OUTPUT
    )

    print(
        PP1_OVERVIEW_OUTPUT
    )

    print(
        OUTPUT_FIGURE_DIR
    )

    if all_validation_passed:

        print(
            "\nPART 1 BACKEND STATUS: COMPLETE"
        )

    else:

        print(
            "\nPART 1 BACKEND STATUS: "
            "VALIDATION CHECK REQUIRED"
        )


if __name__ == "__main__":
    main()