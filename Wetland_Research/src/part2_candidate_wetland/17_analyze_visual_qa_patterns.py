from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "part2_candidate_wetland"
)


# ============================================================
# INPUTS
# ============================================================

QA_INPUT = (
    REPORT_DIR
    / "16_candidate_visual_qa_completed.csv"
)

ALL_CANDIDATES_INPUT = (
    REPORT_DIR
    / "15_candidate_qa_stratification.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

ACTION_DISTRIBUTION_OUTPUT = (
    REPORT_DIR
    / "17_qa_action_attribute_distribution.csv"
)

AREA_BIN_OUTPUT = (
    REPORT_DIR
    / "17_qa_by_area_bin.csv"
)

PROBABILITY_BIN_OUTPUT = (
    REPORT_DIR
    / "17_qa_by_probability_bin.csv"
)

P80_BIN_OUTPUT = (
    REPORT_DIR
    / "17_qa_by_p80_support_bin.csv"
)

COMBINED_BIN_OUTPUT = (
    REPORT_DIR
    / "17_qa_area_probability_crosscheck.csv"
)

POPULATION_PROFILE_OUTPUT = (
    REPORT_DIR
    / "17_all_candidate_profile_by_area_probability.csv"
)

AUDIT_OUTPUT = (
    REPORT_DIR
    / "17_visual_qa_pattern_analysis_audit.json"
)


# ============================================================
# BIN DEFINITIONS
#
# These are descriptive reporting bins only.
# They do NOT alter the model or define new classification
# thresholds.
# ============================================================

AREA_BINS = [
    0.10,
    0.20,
    0.50,
    1.00,
    5.00,
    np.inf,
]

AREA_LABELS = [
    "0.10-<0.20",
    "0.20-<0.50",
    "0.50-<1.00",
    "1.00-<5.00",
    ">=5.00",
]

PROBABILITY_BINS = [
    0.00,
    0.65,
    0.75,
    0.85,
    1.01,
]

PROBABILITY_LABELS = [
    "<0.65",
    "0.65-<0.75",
    "0.75-<0.85",
    ">=0.85",
]

P80_BINS = [
    -0.001,
    0.10,
    0.30,
    0.50,
    0.70,
    1.001,
]

P80_LABELS = [
    "<0.10",
    "0.10-<0.30",
    "0.30-<0.50",
    "0.50-<0.70",
    ">=0.70",
]


# ============================================================
# HELPERS
# ============================================================

def build_action_summary(
    dataframe,
    group_column,
):

    summary = (
        dataframe.groupby(
            [
                group_column,
                "qa_final_action",
            ],
            observed=False,
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    if (
        "retain_candidate"
        not in
        summary.columns
    ):

        summary[
            "retain_candidate"
        ] = 0

    if (
        "flag_false_positive"
        not in
        summary.columns
    ):

        summary[
            "flag_false_positive"
        ] = 0

    summary[
        "reviewed_total"
    ] = (
        summary[
            "retain_candidate"
        ]
        +
        summary[
            "flag_false_positive"
        ]
    )

    summary[
        "retain_percentage_in_reviewed_sample"
    ] = np.where(
        summary[
            "reviewed_total"
        ]
        > 0,

        100.0
        *
        summary[
            "retain_candidate"
        ]
        /
        summary[
            "reviewed_total"
        ],

        np.nan,
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 VISUAL QA PATTERN ANALYSIS"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nThis is descriptive analysis only."
        "\nNo model is trained."
        "\nNo probability threshold is changed."
        "\nNo candidate polygon is automatically removed."
    )

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    qa = pd.read_csv(
        QA_INPUT
    )

    all_candidates = pd.read_csv(
        ALL_CANDIDATES_INPUT
    )

    print(
        f"\nCompleted QA records: "
        f"{len(qa)}"
    )

    print(
        f"All candidate records: "
        f"{len(all_candidates):,}"
    )

    if (
        len(qa)
        != 60
    ):

        raise RuntimeError(
            "Expected 60 completed QA records."
        )

    # ========================================================
    # 2. CHECK REQUIRED FIELDS
    # ========================================================

    required_fields = [
        "candidate_id",
        "area_ha",
        "mean_probability",
        "fraction_ge_0p80",
        "mean_dry_ndvi",
        "mean_wet_ndvi",
        "mean_dry_mndwi",
        "mean_wet_mndwi",
        "mean_elevation_m",
        "mean_distance_to_drainage_m",
        "qa_surface_signal",
        "qa_final_action",
    ]

    missing_fields = [
        field
        for field
        in required_fields
        if field
        not in qa.columns
    ]

    if (
        missing_fields
    ):

        raise RuntimeError(
            f"Missing QA fields: "
            f"{missing_fields}"
        )

    # ========================================================
    # 3. DERIVED VARIABLES
    # ========================================================

    for dataframe in [
        qa,
        all_candidates,
    ]:

        dataframe[
            "mean_seasonal_ndvi_analysis"
        ] = (
            dataframe[
                "mean_dry_ndvi"
            ]
            +
            dataframe[
                "mean_wet_ndvi"
            ]
        ) / 2.0

        dataframe[
            "maximum_seasonal_mndwi_analysis"
        ] = dataframe[
            [
                "mean_dry_mndwi",
                "mean_wet_mndwi",
            ]
        ].max(
            axis=1
        )

    # ========================================================
    # 4. ATTRIBUTE DISTRIBUTIONS BY MANUAL QA ACTION
    # ========================================================

    diagnostic_fields = [
        "area_ha",
        "mean_probability",
        "fraction_ge_0p70",
        "fraction_ge_0p80",
        "fraction_ge_0p90",
        "area_ge_0p80_ha",
        "mean_seasonal_ndvi_analysis",
        "maximum_seasonal_mndwi_analysis",
        "mean_elevation_m",
        "mean_distance_to_drainage_m",
    ]

    distribution_rows = []

    for action in [
        "retain_candidate",
        "flag_false_positive",
    ]:

        subset = (
            qa[
                qa[
                    "qa_final_action"
                ]
                ==
                action
            ]
        )

        for field in (
            diagnostic_fields
        ):

            values = (
                subset[
                    field
                ]
                .dropna()
            )

            distribution_rows.append(
                {
                    "qa_final_action":
                        action,

                    "attribute":
                        field,

                    "n":
                        len(
                            values
                        ),

                    "minimum":
                        float(
                            values.min()
                        ),

                    "q25":
                        float(
                            values.quantile(
                                0.25
                            )
                        ),

                    "median":
                        float(
                            values.median()
                        ),

                    "mean":
                        float(
                            values.mean()
                        ),

                    "q75":
                        float(
                            values.quantile(
                                0.75
                            )
                        ),

                    "maximum":
                        float(
                            values.max()
                        ),
                }
            )

    distribution_table = pd.DataFrame(
        distribution_rows
    )

    distribution_table.to_csv(
        ACTION_DISTRIBUTION_OUTPUT,
        index=False,
    )

    # ========================================================
    # 5. AREA BINS
    # ========================================================

    qa[
        "analysis_area_bin"
    ] = pd.cut(
        qa[
            "area_ha"
        ],
        bins=AREA_BINS,
        labels=AREA_LABELS,
        right=False,
        include_lowest=True,
    )

    area_summary = build_action_summary(
        qa,
        "analysis_area_bin",
    )

    area_summary.to_csv(
        AREA_BIN_OUTPUT,
        index=False,
    )

    # ========================================================
    # 6. MEAN PROBABILITY BINS
    # ========================================================

    qa[
        "analysis_probability_bin"
    ] = pd.cut(
        qa[
            "mean_probability"
        ],
        bins=PROBABILITY_BINS,
        labels=PROBABILITY_LABELS,
        right=False,
        include_lowest=True,
    )

    probability_summary = (
        build_action_summary(
            qa,
            "analysis_probability_bin",
        )
    )

    probability_summary.to_csv(
        PROBABILITY_BIN_OUTPUT,
        index=False,
    )

    # ========================================================
    # 7. >=0.80 SUPPORT-FRACTION BINS
    # ========================================================

    qa[
        "analysis_p80_bin"
    ] = pd.cut(
        qa[
            "fraction_ge_0p80"
        ],
        bins=P80_BINS,
        labels=P80_LABELS,
        right=False,
        include_lowest=True,
    )

    p80_summary = build_action_summary(
        qa,
        "analysis_p80_bin",
    )

    p80_summary.to_csv(
        P80_BIN_OUTPUT,
        index=False,
    )

    # ========================================================
    # 8. AREA x PROBABILITY CROSS-CHECK
    # ========================================================

    combined = (
        qa.groupby(
            [
                "analysis_area_bin",
                "analysis_probability_bin",
                "qa_final_action",
            ],
            observed=False,
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    if (
        "retain_candidate"
        not in
        combined.columns
    ):

        combined[
            "retain_candidate"
        ] = 0

    if (
        "flag_false_positive"
        not in
        combined.columns
    ):

        combined[
            "flag_false_positive"
        ] = 0

    combined[
        "reviewed_total"
    ] = (
        combined[
            "retain_candidate"
        ]
        +
        combined[
            "flag_false_positive"
        ]
    )

    combined[
        "retain_percentage_in_reviewed_sample"
    ] = np.where(
        combined[
            "reviewed_total"
        ]
        > 0,

        100.0
        *
        combined[
            "retain_candidate"
        ]
        /
        combined[
            "reviewed_total"
        ],

        np.nan,
    )

    combined.to_csv(
        COMBINED_BIN_OUTPUT,
        index=False,
    )

    # ========================================================
    # 9. APPLY SAME BINS TO ALL 1,097 CANDIDATES
    #
    # This does NOT classify them.
    # It only shows how many full-population candidates occupy
    # each descriptive area/probability combination.
    # ========================================================

    all_candidates[
        "analysis_area_bin"
    ] = pd.cut(
        all_candidates[
            "area_ha"
        ],
        bins=AREA_BINS,
        labels=AREA_LABELS,
        right=False,
        include_lowest=True,
    )

    all_candidates[
        "analysis_probability_bin"
    ] = pd.cut(
        all_candidates[
            "mean_probability"
        ],
        bins=PROBABILITY_BINS,
        labels=PROBABILITY_LABELS,
        right=False,
        include_lowest=True,
    )

    population_profile = (
        all_candidates.groupby(
            [
                "analysis_area_bin",
                "analysis_probability_bin",
            ],
            observed=False,
        )
        .agg(
            candidates=(
                "candidate_id",
                "count",
            ),

            total_area_ha=(
                "area_ha",
                "sum",
            ),

            mean_probability=(
                "mean_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    population_profile.to_csv(
        POPULATION_PROFILE_OUTPUT,
        index=False,
    )

    # ========================================================
    # 10. TERMINAL COMPARISON TABLE
    # ========================================================

    comparison = (
        distribution_table[
            distribution_table[
                "attribute"
            ]
            .isin(
                [
                    "area_ha",
                    "mean_probability",
                    "fraction_ge_0p80",
                    "area_ge_0p80_ha",
                    "mean_seasonal_ndvi_analysis",
                    "maximum_seasonal_mndwi_analysis",
                    "mean_distance_to_drainage_m",
                ]
            )
        ]
        [
            [
                "qa_final_action",
                "attribute",
                "n",
                "q25",
                "median",
                "mean",
                "q75",
            ]
        ]
        .copy()
    )

    # ========================================================
    # 11. AUDIT
    # ========================================================

    audit = {
        "qa_records":
            int(
                len(
                    qa
                )
            ),

        "all_candidate_records":
            int(
                len(
                    all_candidates
                )
            ),

        "analysis_type":
            "descriptive_only",

        "trained_secondary_classifier":
            False,

        "changed_xgboost_model":
            False,

        "changed_frozen_probability_threshold":
            False,

        "automatically_removed_candidates":
            False,

        "important_warning":
            (
                "The 60-object visual QA set was selected "
                "using a stratified design. Percentages in "
                "the diagnostic bins describe the reviewed "
                "sample only and must not be treated as "
                "unbiased population-wide probabilities."
            ),
    }

    with open(
        AUDIT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit,
            file,
            indent=4,
        )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "KEY ATTRIBUTE DISTRIBUTIONS"
    )

    print(
        "========================================"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "QA RESULTS BY COMPONENT AREA"
    )

    print(
        "========================================"
    )

    print(
        area_summary.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "QA RESULTS BY MEAN MODEL PROBABILITY"
    )

    print(
        "========================================"
    )

    print(
        probability_summary.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "QA RESULTS BY >=0.80 SUPPORT FRACTION"
    )

    print(
        "========================================"
    )

    print(
        p80_summary.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "AREA x PROBABILITY CROSS-CHECK"
    )

    print(
        "========================================"
    )

    print(
        combined[
            combined[
                "reviewed_total"
            ]
            > 0
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "ALL 1,097 CANDIDATE PROFILE"
    )

    print(
        "========================================"
    )

    print(
        population_profile[
            population_profile[
                "candidates"
            ]
            > 0
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        ACTION_DISTRIBUTION_OUTPUT
    )

    print(
        AREA_BIN_OUTPUT
    )

    print(
        PROBABILITY_BIN_OUTPUT
    )

    print(
        P80_BIN_OUTPUT
    )

    print(
        COMBINED_BIN_OUTPUT
    )

    print(
        POPULATION_PROFILE_OUTPUT
    )

    print(
        AUDIT_OUTPUT
    )

    print(
        "\n========================================"
    )

    print(
        "VISUAL QA PATTERN ANALYSIS: COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nNo automatic candidate filtering has "
        "been performed."
        "\nWe will use these diagnostics to decide "
        "whether a conservative confidence-tier "
        "system is justified."
    )


if __name__ == "__main__":
    main()