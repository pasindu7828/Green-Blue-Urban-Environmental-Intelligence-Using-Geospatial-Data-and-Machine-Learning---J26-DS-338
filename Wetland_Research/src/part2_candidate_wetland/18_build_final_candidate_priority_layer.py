from pathlib import Path
import json

import geopandas as gpd
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

MAP_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "maps"
    / "part2_candidate_wetland"
)


# ============================================================
# INPUTS
# ============================================================

ALL_CANDIDATES_INPUT = (
    REPORT_DIR
    / "15_candidate_qa_stratification.csv"
)

MANUAL_QA_INPUT = (
    REPORT_DIR
    / "16_candidate_visual_qa_completed.csv"
)

CANDIDATE_UTM_INPUT = (
    MAP_DIR
    / "14_candidate_qa_polygons_mmu_0p10ha_utm.geojson"
)

CANDIDATE_WGS84_INPUT = (
    MAP_DIR
    / "14_candidate_qa_polygons_mmu_0p10ha_wgs84.geojson"
)


# ============================================================
# OUTPUTS
# ============================================================

FINAL_TABLE_OUTPUT = (
    REPORT_DIR
    / "18_final_candidate_priority_table.csv"
)

PRIORITY_SUMMARY_OUTPUT = (
    REPORT_DIR
    / "18_candidate_priority_summary.csv"
)

MANUAL_QA_CROSSCHECK_OUTPUT = (
    REPORT_DIR
    / "18_priority_manual_qa_crosscheck.csv"
)

FINAL_UTM_OUTPUT = (
    MAP_DIR
    / "18_final_candidate_priority_utm.geojson"
)

FINAL_WGS84_OUTPUT = (
    MAP_DIR
    / "18_final_candidate_priority_wgs84.geojson"
)

PRIMARY_UTM_OUTPUT = (
    MAP_DIR
    / "18_primary_candidate_layer_utm.geojson"
)

PRIMARY_WGS84_OUTPUT = (
    MAP_DIR
    / "18_primary_candidate_layer_wgs84.geojson"
)

REVIEW_UTM_OUTPUT = (
    MAP_DIR
    / "18_review_candidate_layer_utm.geojson"
)

FALSE_POSITIVE_UTM_OUTPUT = (
    MAP_DIR
    / "18_manual_false_positive_flags_utm.geojson"
)

AUDIT_OUTPUT = (
    REPORT_DIR
    / "18_final_candidate_priority_audit.json"
)


# ============================================================
# PRIORITY RULES
#
# IMPORTANT:
#
# These are transparent screening/prioritization rules.
# They are NOT a second ML classifier.
#
# HIGH:
#   area >= 5 ha
#   AND mean model probability >= 0.75
#
# MEDIUM:
#   area >= 1 ha AND probability >= 0.75
#   OR
#   area >= 5 ha AND probability >= 0.65
#
# REVIEW:
#   all remaining MMU-filtered candidate polygons
#
# Manual visual QA results override reporting status only for
# the 60 polygons that were actually reviewed.
# ============================================================

HIGH_MIN_AREA_HA = 5.0
HIGH_MIN_PROBABILITY = 0.75

MEDIUM_MIN_AREA_HA = 1.0
MEDIUM_MIN_PROBABILITY = 0.75

LARGE_LOWER_SUPPORT_AREA_HA = 5.0
LARGE_LOWER_SUPPORT_MIN_PROBABILITY = 0.65


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 FINAL CANDIDATE PRIORITY LAYER"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nThis script does NOT create a new classifier."
        "\nThe frozen XGBoost model and 0.50 threshold "
        "remain unchanged."
        "\nPriority tiers are for candidate screening "
        "and reporting only."
    )

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    candidates = pd.read_csv(
        ALL_CANDIDATES_INPUT
    )

    manual_qa = pd.read_csv(
        MANUAL_QA_INPUT
    )

    candidate_utm = gpd.read_file(
        CANDIDATE_UTM_INPUT
    )

    candidate_wgs84 = gpd.read_file(
        CANDIDATE_WGS84_INPUT
    )

    print(
        f"\nAll MMU-filtered candidates: "
        f"{len(candidates):,}"
    )

    print(
        f"Manual QA reviewed candidates: "
        f"{len(manual_qa):,}"
    )

    if len(candidates) != 1097:
        raise RuntimeError(
            "Expected 1,097 candidate records."
        )

    if len(manual_qa) != 60:
        raise RuntimeError(
            "Expected 60 completed manual QA records."
        )

    # ========================================================
    # 2. AUTOMATIC SCREENING PRIORITY
    # ========================================================

    candidates[
        "screening_priority"
    ] = "Review"

    high_mask = (
        (
            candidates[
                "area_ha"
            ]
            >=
            HIGH_MIN_AREA_HA
        )
        &
        (
            candidates[
                "mean_probability"
            ]
            >=
            HIGH_MIN_PROBABILITY
        )
    )

    medium_mask = (
        (
            (
                candidates[
                    "area_ha"
                ]
                >=
                MEDIUM_MIN_AREA_HA
            )
            &
            (
                candidates[
                    "mean_probability"
                ]
                >=
                MEDIUM_MIN_PROBABILITY
            )
        )
        |
        (
            (
                candidates[
                    "area_ha"
                ]
                >=
                LARGE_LOWER_SUPPORT_AREA_HA
            )
            &
            (
                candidates[
                    "mean_probability"
                ]
                >=
                LARGE_LOWER_SUPPORT_MIN_PROBABILITY
            )
        )
    )

    candidates.loc[
        medium_mask,
        "screening_priority",
    ] = "Medium"

    candidates.loc[
        high_mask,
        "screening_priority",
    ] = "High"

    # ========================================================
    # 3. SURFACE-SIGNAL WARNING
    #
    # This does NOT remove polygons.
    # ========================================================

    candidates[
        "surface_review_flag"
    ] = "none"

    candidates.loc[
        candidates[
            "qa_surface_signal"
        ]
        ==
        "water_signal",
        "surface_review_flag",
    ] = "water_confusion_risk"

    candidates.loc[
        candidates[
            "qa_surface_signal"
        ]
        ==
        "vegetation_signal",
        "surface_review_flag",
    ] = "vegetation_confusion_risk"

    candidates.loc[
        candidates[
            "qa_surface_signal"
        ]
        ==
        "mixed_signal",
        "surface_review_flag",
    ] = "mixed_surface_signal"

    # ========================================================
    # 4. MANUAL QA LOOKUP
    # ========================================================

    qa_fields = (
        manual_qa[
            [
                "candidate_id",
                "qa_final_visual_category",
                "qa_final_action",
                "qa_visual_confidence",
                "qa_visual_notes",
            ]
        ]
        .copy()
    )

    candidates = (
        candidates.merge(
            qa_fields,
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
    )

    candidates[
        "manual_qa_reviewed"
    ] = (
        candidates[
            "qa_final_action"
        ]
        .notna()
    )

    # ========================================================
    # 5. FINAL REPORTING STATUS
    #
    # Manual visual QA overrides only reviewed polygons.
    # Unreviewed polygons remain priority-based.
    # ========================================================

    candidates[
        "candidate_reporting_status"
    ] = (
        candidates[
            "screening_priority"
        ]
        .map(
            {
                "High":
                    "high_priority_unreviewed_candidate",

                "Medium":
                    "medium_priority_unreviewed_candidate",

                "Review":
                    "review_priority_unreviewed_candidate",
            }
        )
    )

    reviewed_retain = (
        candidates[
            "qa_final_action"
        ]
        ==
        "retain_candidate"
    )

    reviewed_false = (
        candidates[
            "qa_final_action"
        ]
        ==
        "flag_false_positive"
    )

    candidates.loc[
        reviewed_retain,
        "candidate_reporting_status",
    ] = "manual_qa_retained_candidate"

    candidates.loc[
        reviewed_false,
        "candidate_reporting_status",
    ] = "manual_qa_false_positive_flag"

    # ========================================================
    # 6. PRIMARY DISPLAY FLAG
    #
    # Primary map:
    # - manually retained candidates
    # - unreviewed High candidates
    # - unreviewed Medium candidates
    #
    # Review-priority candidates remain available separately.
    # Manually flagged false positives are not in primary map.
    # ========================================================

    candidates[
        "include_primary_candidate_layer"
    ] = False

    candidates.loc[
        reviewed_retain,
        "include_primary_candidate_layer",
    ] = True

    unreviewed_primary = (
        (
            ~candidates[
                "manual_qa_reviewed"
            ]
        )
        &
        (
            candidates[
                "screening_priority"
            ]
            .isin(
                [
                    "High",
                    "Medium",
                ]
            )
        )
    )

    candidates.loc[
        unreviewed_primary,
        "include_primary_candidate_layer",
    ] = True

    # ========================================================
    # 7. SAVE FINAL TABLE
    # ========================================================

    candidates.to_csv(
        FINAL_TABLE_OUTPUT,
        index=False,
    )

    # ========================================================
    # 8. PRIORITY SUMMARY
    # ========================================================

    priority_summary = (
        candidates.groupby(
            "screening_priority"
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

            manually_reviewed=(
                "manual_qa_reviewed",
                "sum",
            ),

            primary_layer_count=(
                "include_primary_candidate_layer",
                "sum",
            ),
        )
        .reset_index()
    )

    priority_order = {
        "High": 1,
        "Medium": 2,
        "Review": 3,
    }

    priority_summary[
        "_order"
    ] = (
        priority_summary[
            "screening_priority"
        ]
        .map(
            priority_order
        )
    )

    priority_summary = (
        priority_summary
        .sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
    )

    priority_summary.to_csv(
        PRIORITY_SUMMARY_OUTPUT,
        index=False,
    )

    # ========================================================
    # 9. MANUAL QA CROSSCHECK
    #
    # Descriptive only.
    # Do NOT interpret as unbiased population accuracy.
    # ========================================================

    reviewed_only = (
        candidates[
            candidates[
                "manual_qa_reviewed"
            ]
        ]
        .copy()
    )

    manual_crosscheck = (
        reviewed_only.groupby(
            [
                "screening_priority",
                "qa_final_action",
            ]
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
        manual_crosscheck.columns
    ):
        manual_crosscheck[
            "retain_candidate"
        ] = 0

    if (
        "flag_false_positive"
        not in
        manual_crosscheck.columns
    ):
        manual_crosscheck[
            "flag_false_positive"
        ] = 0

    manual_crosscheck[
        "reviewed_total"
    ] = (
        manual_crosscheck[
            "retain_candidate"
        ]
        +
        manual_crosscheck[
            "flag_false_positive"
        ]
    )

    manual_crosscheck[
        "retain_percentage_in_reviewed_sample"
    ] = np.where(
        manual_crosscheck[
            "reviewed_total"
        ]
        > 0,

        100.0
        *
        manual_crosscheck[
            "retain_candidate"
        ]
        /
        manual_crosscheck[
            "reviewed_total"
        ],

        np.nan,
    )

    manual_crosscheck.to_csv(
        MANUAL_QA_CROSSCHECK_OUTPUT,
        index=False,
    )

    # ========================================================
    # 10. JOIN ATTRIBUTES TO SPATIAL LAYERS
    # ========================================================

    output_fields = [
        "candidate_id",
        "screening_priority",
        "surface_review_flag",
        "manual_qa_reviewed",
        "qa_final_visual_category",
        "qa_final_action",
        "qa_visual_confidence",
        "candidate_reporting_status",
        "include_primary_candidate_layer",
    ]

    def build_spatial_output(
        spatial_dataframe,
    ):

        drop_fields = [
            field
            for field
            in output_fields
            if (
                field
                in
                spatial_dataframe.columns
                and
                field
                !=
                "candidate_id"
            )
        ]

        result = (
            spatial_dataframe
            .drop(
                columns=drop_fields,
                errors="ignore",
            )
            .merge(
                candidates[
                    output_fields
                ],
                on="candidate_id",
                how="left",
                validate="one_to_one",
            )
        )

        if (
            result[
                "screening_priority"
            ]
            .isna()
            .any()
        ):
            raise RuntimeError(
                "Spatial priority join failed."
            )

        return result

    final_utm = build_spatial_output(
        candidate_utm
    )

    final_wgs84 = build_spatial_output(
        candidate_wgs84
    )

    # ========================================================
    # 11. SAVE FULL PRIORITY LAYERS
    # ========================================================

    final_utm.to_file(
        FINAL_UTM_OUTPUT,
        driver="GeoJSON",
    )

    final_wgs84.to_file(
        FINAL_WGS84_OUTPUT,
        driver="GeoJSON",
    )

    # ========================================================
    # 12. PRIMARY CANDIDATE LAYER
    # ========================================================

    primary_utm = (
        final_utm[
            final_utm[
                "include_primary_candidate_layer"
            ]
            ==
            True
        ]
        .copy()
    )

    primary_wgs84 = (
        final_wgs84[
            final_wgs84[
                "include_primary_candidate_layer"
            ]
            ==
            True
        ]
        .copy()
    )

    primary_utm.to_file(
        PRIMARY_UTM_OUTPUT,
        driver="GeoJSON",
    )

    primary_wgs84.to_file(
        PRIMARY_WGS84_OUTPUT,
        driver="GeoJSON",
    )

    # ========================================================
    # 13. REVIEW-PRIORITY LAYER
    # ========================================================

    review_utm = (
        final_utm[
            (
                final_utm[
                    "candidate_reporting_status"
                ]
                ==
                "review_priority_unreviewed_candidate"
            )
        ]
        .copy()
    )

    review_utm.to_file(
        REVIEW_UTM_OUTPUT,
        driver="GeoJSON",
    )

    # ========================================================
    # 14. MANUAL FALSE-POSITIVE FLAGS
    # ========================================================

    false_positive_utm = (
        final_utm[
            final_utm[
                "candidate_reporting_status"
            ]
            ==
            "manual_qa_false_positive_flag"
        ]
        .copy()
    )

    false_positive_utm.to_file(
        FALSE_POSITIVE_UTM_OUTPUT,
        driver="GeoJSON",
    )

    # ========================================================
    # 15. AUDIT
    # ========================================================

    audit = {
        "input_candidates":
            int(
                len(
                    candidates
                )
            ),

        "manual_qa_reviewed":
            int(
                candidates[
                    "manual_qa_reviewed"
                ]
                .sum()
            ),

        "high_priority_rule":
            (
                "area_ha >= 5.0 AND "
                "mean_probability >= 0.75"
            ),

        "medium_priority_rule":
            (
                "(area_ha >= 1.0 AND "
                "mean_probability >= 0.75) OR "
                "(area_ha >= 5.0 AND "
                "mean_probability >= 0.65)"
            ),

        "review_priority_rule":
            "all remaining MMU-filtered candidates",

        "xgboost_threshold_changed":
            False,

        "model_retrained":
            False,

        "secondary_classifier_trained":
            False,

        "automatic_open_water_removal":
            False,

        "important_note":
            (
                "Priority tiers are screening and reporting "
                "categories only. They do not represent "
                "official wetland confirmation."
            ),

        "manual_qa_note":
            (
                "Only the 60 polygons in the stratified "
                "visual-QA sample receive manual visual "
                "retain/false-positive flags."
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
        "CANDIDATE PRIORITY SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        priority_summary.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "MANUAL QA x PRIORITY CROSS-CHECK"
    )

    print(
        "========================================"
    )

    print(
        manual_crosscheck.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "FINAL LAYER COUNTS"
    )

    print(
        "========================================"
    )

    print(
        f"All candidate polygons: "
        f"{len(final_utm):,}"
    )

    print(
        f"Primary candidate polygons: "
        f"{len(primary_utm):,}"
    )

    print(
        f"Review-priority unreviewed polygons: "
        f"{len(review_utm):,}"
    )

    print(
        f"Manual QA false-positive flags: "
        f"{len(false_positive_utm):,}"
    )

    print(
        "\nSurface review flags:"
    )

    print(
        candidates[
            "surface_review_flag"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nReporting-status counts:"
    )

    print(
        candidates[
            "candidate_reporting_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nOutputs:"
    )

    print(
        FINAL_TABLE_OUTPUT
    )

    print(
        PRIORITY_SUMMARY_OUTPUT
    )

    print(
        MANUAL_QA_CROSSCHECK_OUTPUT
    )

    print(
        FINAL_UTM_OUTPUT
    )

    print(
        FINAL_WGS84_OUTPUT
    )

    print(
        PRIMARY_UTM_OUTPUT
    )

    print(
        PRIMARY_WGS84_OUTPUT
    )

    print(
        REVIEW_UTM_OUTPUT
    )

    print(
        FALSE_POSITIVE_UTM_OUTPUT
    )

    print(
        AUDIT_OUTPUT
    )

    print(
        "\n========================================"
    )

    print(
        "FINAL CANDIDATE PRIORITY LAYER: COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nHigh / Medium / Review are candidate "
        "screening priorities, not wetland classes."
        "\nManually flagged false positives are only "
        "the reviewed polygons for which imagery "
        "showed clear non-wetland/open-water dominance."
        "\nNo unreviewed candidate has been automatically "
        "declared a false positive."
    )


if __name__ == "__main__":
    main()