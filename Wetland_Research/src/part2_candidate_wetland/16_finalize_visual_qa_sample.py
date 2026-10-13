from pathlib import Path
import json

import geopandas as gpd
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

QA_SAMPLE_INPUT = (
    REPORT_DIR
    / "15_candidate_visual_qa_sample.csv"
)

QA_SPATIAL_INPUT = (
    MAP_DIR
    / "15_candidate_visual_qa_sample_wgs84.geojson"
)


# ============================================================
# OUTPUTS
# ============================================================

COMPLETED_QA_OUTPUT = (
    REPORT_DIR
    / "16_candidate_visual_qa_completed.csv"
)

CATEGORY_SUMMARY_OUTPUT = (
    REPORT_DIR
    / "16_visual_qa_category_summary.csv"
)

ACTION_SUMMARY_OUTPUT = (
    REPORT_DIR
    / "16_visual_qa_action_summary.csv"
)

GROUP_SUMMARY_OUTPUT = (
    REPORT_DIR
    / "16_visual_qa_by_sample_group.csv"
)

SIGNAL_SUMMARY_OUTPUT = (
    REPORT_DIR
    / "16_visual_qa_by_surface_signal.csv"
)

COMPLETED_SPATIAL_OUTPUT = (
    MAP_DIR
    / "16_candidate_visual_qa_completed_wgs84.geojson"
)

AUDIT_OUTPUT = (
    REPORT_DIR
    / "16_visual_qa_completion_audit.json"
)


# ============================================================
# FINAL MANUAL VISUAL-QA DECISIONS
#
# IMPORTANT:
# These labels describe only what was observed during
# visual QA of the 60-polygon stratified sample.
#
# "retain_candidate" DOES NOT mean confirmed wetland.
#
# "flag_false_positive" means the imagery strongly suggests
# the object should not be treated as an additional wetland
# candidate without further evidence.
# ============================================================

QA_DECISIONS = {

    # --------------------------------------------------------
    # TOP PRIORITY / PREVIOUSLY REVIEWED
    # --------------------------------------------------------

    "CW_01849": {
        "category": "wetland_like_lowland_mixed",
        "action": "retain_candidate",
        "confidence": "High",
        "note": "Persistent wetland-like or seasonally wet lowland complex; mixed with paddy or other lowland vegetation.",
    },

    "CW_05540": {
        "category": "open_water_dominated",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Permanent open-water dominated area associated with Diyawanna Lake.",
    },

    "CW_03482": {
        "category": "open_water_water_edge_dominated",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Large water and drainage corridor with vegetated margins.",
    },

    "CW_04215": {
        "category": "wetland_like_lowland_mixed",
        "action": "retain_candidate",
        "confidence": "High",
        "note": "Persistent low-lying vegetated complex with strong paddy or agricultural influence.",
    },

    "CW_02680": {
        "category": "wetland_like_lowland_mixed",
        "action": "retain_candidate",
        "confidence": "High",
        "note": "Persistent seasonally wet lowland complex mixed with paddy or agriculture.",
    },

    "CW_00648": {
        "category": "wetland_like_lowland_mixed",
        "action": "retain_candidate",
        "confidence": "High",
        "note": "Persistent lowland vegetation and paddy-like landscape; not open-water dominated.",
    },

    "CW_04161": {
        "category": "wetland_like_lowland_mixed",
        "action": "retain_candidate",
        "confidence": "High",
        "note": "Broad low-lying vegetated area with field-like and seasonally wet characteristics.",
    },

    "CW_07144": {
        "category": "wetland_like_lowland_mixed",
        "action": "retain_candidate",
        "confidence": "High",
        "note": "Lowland complex strongly influenced by paddy or agriculture.",
    },

    "CW_02346": {
        "category": "wetland_like_lowland_mixed",
        "action": "retain_candidate",
        "confidence": "High",
        "note": "Vegetated lowland corridor with agricultural and urban-edge influence.",
    },

    "CW_04706": {
        "category": "wetland_like_lowland_mixed",
        "action": "retain_candidate",
        "confidence": "High",
        "note": "Wetland-like lowland complex with strong paddy influence.",
    },

    "CW_00138": {
        "category": "wetland_like_water_edge_mixed",
        "action": "retain_candidate",
        "confidence": "High",
        "note": "Wetland-like lowland with open-water patches; not dominated entirely by open water.",
    },

    "CW_00757": {
        "category": "wetland_like_lowland_mixed",
        "action": "retain_candidate",
        "confidence": "High",
        "note": "Wetland-like lowland mixture.",
    },

    "CW_01185": {
        "category": "wetland_like_lowland_mixed",
        "action": "retain_candidate",
        "confidence": "High",
        "note": "Lowland wetland-like vegetation mixed with disturbed and built-up edges.",
    },

    "CW_01605": {
        "category": "wetland_like_lowland_mixed",
        "action": "retain_candidate",
        "confidence": "Medium-High",
        "note": "Wetland-like lowland with substantial disturbed and bare-land influence.",
    },

    "CW_03301": {
        "category": "wetland_like_lowland_mixed",
        "action": "retain_candidate",
        "confidence": "High",
        "note": "Wetland-like lowland mixture.",
    },

    # --------------------------------------------------------
    # WATER-SIGNAL SAMPLE
    # --------------------------------------------------------

    "CW_08096": {
        "category": "other_non_wetland_paddy_agriculture",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Clearly structured paddy or agricultural fields.",
    },

    "CW_07755": {
        "category": "other_non_wetland_paddy_agriculture",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Paddy or agriculture dominated.",
    },

    "CW_10206": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Built-up and residential vegetation dominated.",
    },

    "CW_07103": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential or institutional landscaped area.",
    },

    "CW_08008": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential or institutional landscaped area.",
    },

    "CW_08869": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Built-up and residential vegetation dominated.",
    },

    "CW_00442": {
        "category": "wetland_like_water_edge_mixed",
        "action": "retain_candidate",
        "confidence": "Medium-High",
        "note": "Vegetated water-margin or riparian candidate; not dominated by open water.",
    },

    "CW_00028": {
        "category": "other_non_wetland_ordinary_vegetation",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Plantation or residential river-edge vegetation without clear wetland characteristics.",
    },

    "CW_02311": {
        "category": "wetland_like_water_edge_mixed",
        "action": "retain_candidate",
        "confidence": "Medium-High",
        "note": "Vegetated water-edge area with plausible wetland-margin characteristics.",
    },

    "CW_06562": {
        "category": "other_non_wetland_paddy_agriculture",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Managed paddy or agricultural fields dominate the candidate.",
    },

    "CW_00045": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential buildings and ordinary tree cover dominate.",
    },

    "CW_00279": {
        "category": "open_water_water_edge_dominated",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Open water and water-edge area dominate the candidate.",
    },

    # --------------------------------------------------------
    # VEGETATION-SIGNAL SAMPLE
    # --------------------------------------------------------

    "CW_06334": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential area with ordinary dense vegetation.",
    },

    "CW_01178": {
        "category": "wetland_like_lowland_mixed",
        "action": "retain_candidate",
        "confidence": "Medium-High",
        "note": "Broad open and vegetated lowland with plausible wetland-like characteristics.",
    },

    "CW_10111": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential and garden vegetation dominated.",
    },

    "CW_09615": {
        "category": "other_non_wetland_paddy_agriculture",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Managed paddy or agricultural plots dominate.",
    },

    "CW_08353": {
        "category": "other_non_wetland_ordinary_vegetation",
        "action": "flag_false_positive",
        "confidence": "Medium-High",
        "note": "Ordinary vegetation mixed with residential land.",
    },

    "CW_06493": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential and agricultural-edge vegetation.",
    },

    "CW_07749": {
        "category": "other_non_wetland_ordinary_vegetation",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Grass or yard vegetation surrounded by development.",
    },

    "CW_04521": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential or garden vegetation.",
    },

    "CW_00614": {
        "category": "other_non_wetland_paddy_agriculture",
        "action": "flag_false_positive",
        "confidence": "Medium-High",
        "note": "Plantation or agricultural vegetation dominates.",
    },

    "CW_08504": {
        "category": "other_non_wetland_ordinary_vegetation",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Ordinary vegetation near residential or agricultural land.",
    },

    "CW_00546": {
        "category": "other_non_wetland_ordinary_vegetation",
        "action": "flag_false_positive",
        "confidence": "Medium-High",
        "note": "Dense plantation or ordinary vegetation.",
    },

    "CW_03863": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Built-up and residential vegetation dominated.",
    },

    # --------------------------------------------------------
    # MIXED-SIGNAL SAMPLE
    # --------------------------------------------------------

    "CW_00105": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential garden and tree cover.",
    },

    "CW_07065": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential or wooded garden area.",
    },

    "CW_01176": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential and ordinary tree cover.",
    },

    "CW_01264": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential and garden vegetation.",
    },

    "CW_06130": {
        "category": "other_non_wetland_disturbed_bare",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Disturbed or bare land mixed with built-up development.",
    },

    "CW_09093": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential and ordinary vegetation.",
    },

    "CW_04969": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Vegetation strip embedded among buildings.",
    },

    "CW_09740": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential and ordinary vegetation.",
    },

    "CW_02768": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential or institutional vegetation.",
    },

    "CW_03367": {
        "category": "other_non_wetland_ordinary_vegetation",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Scrub or ordinary vegetation with road and disturbed-land influence; confirmed using multiple dates.",
    },

    "CW_08907": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "Medium-High",
        "note": "Residential vegetation dominated.",
    },

    "CW_06773": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Small tree-covered patch surrounded by residences.",
    },

    # --------------------------------------------------------
    # LOWER-SUPPORT SAMPLE
    # --------------------------------------------------------

    "CW_03834": {
        "category": "other_non_wetland_paddy_agriculture",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Plantation or agricultural land dominates.",
    },

    "CW_10345": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential and ordinary vegetation dominated; confirmed using multiple dates.",
    },

    "CW_03725": {
        "category": "open_water_disturbed_mixed",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Open-water and bare or disturbed land dominate.",
    },

    "CW_02991": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential and ordinary tree cover.",
    },

    "CW_07129": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential and garden vegetation.",
    },

    "CW_08749": {
        "category": "other_non_wetland_disturbed_bare",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Bare or disturbed land mixed with residential vegetation.",
    },

    "CW_00194": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Road, residential land and tree cover dominate.",
    },

    "CW_09966": {
        "category": "other_non_wetland_ordinary_vegetation",
        "action": "flag_false_positive",
        "confidence": "Medium-High",
        "note": "Ordinary dense vegetation mixed with buildings.",
    },

    "CW_04922": {
        "category": "other_non_wetland_built_residential",
        "action": "flag_false_positive",
        "confidence": "High",
        "note": "Residential and garden vegetation.",
    },
}


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 FINALIZE VISUAL QA SAMPLE"
    )

    print(
        "========================================"
    )

    qa = pd.read_csv(
        QA_SAMPLE_INPUT
    )

    spatial = gpd.read_file(
        QA_SPATIAL_INPUT
    )

    print(
        f"\nQA sample records: "
        f"{len(qa)}"
    )

    print(
        f"Spatial QA records: "
        f"{len(spatial)}"
    )

    # ========================================================
    # 1. VALIDATE THE 60 MANUAL DECISIONS
    # ========================================================

    sample_ids = set(
        qa[
            "candidate_id"
        ]
        .astype(str)
    )

    decision_ids = set(
        QA_DECISIONS.keys()
    )

    missing_decisions = (
        sample_ids
        -
        decision_ids
    )

    extra_decisions = (
        decision_ids
        -
        sample_ids
    )

    if (
        missing_decisions
        or
        extra_decisions
    ):

        raise RuntimeError(
            "\nManual QA decision mismatch."
            f"\nMissing: {sorted(missing_decisions)}"
            f"\nExtra: {sorted(extra_decisions)}"
        )

    if (
        len(
            QA_DECISIONS
        )
        != 60
    ):

        raise RuntimeError(
            f"Expected 60 QA decisions, "
            f"found {len(QA_DECISIONS)}."
        )

    print(
        "\nManual QA decision coverage: PASS"
    )

    # ========================================================
    # 2. APPLY FINAL MANUAL QA FIELDS
    # ========================================================

    qa[
        "manual_review_status"
    ] = "reviewed"

    qa[
        "qa_final_visual_category"
    ] = (
        qa[
            "candidate_id"
        ]
        .map(
            lambda candidate_id:
                QA_DECISIONS[
                    candidate_id
                ][
                    "category"
                ]
        )
    )

    qa[
        "qa_final_action"
    ] = (
        qa[
            "candidate_id"
        ]
        .map(
            lambda candidate_id:
                QA_DECISIONS[
                    candidate_id
                ][
                    "action"
                ]
        )
    )

    qa[
        "qa_visual_confidence"
    ] = (
        qa[
            "candidate_id"
        ]
        .map(
            lambda candidate_id:
                QA_DECISIONS[
                    candidate_id
                ][
                    "confidence"
                ]
        )
    )

    qa[
        "qa_visual_notes"
    ] = (
        qa[
            "candidate_id"
        ]
        .map(
            lambda candidate_id:
                QA_DECISIONS[
                    candidate_id
                ][
                    "note"
                ]
        )
    )

    qa.to_csv(
        COMPLETED_QA_OUTPUT,
        index=False,
    )

    # ========================================================
    # 3. CATEGORY SUMMARY
    # ========================================================

    category_summary = (
        qa.groupby(
            "qa_final_visual_category"
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
        .sort_values(
            "candidates",
            ascending=False,
        )
    )

    category_summary[
        "sample_percentage"
    ] = (
        100.0
        *
        category_summary[
            "candidates"
        ]
        /
        len(qa)
    )

    category_summary.to_csv(
        CATEGORY_SUMMARY_OUTPUT,
        index=False,
    )

    # ========================================================
    # 4. BROAD ACTION SUMMARY
    # ========================================================

    action_summary = (
        qa.groupby(
            "qa_final_action"
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

    action_summary[
        "sample_percentage"
    ] = (
        100.0
        *
        action_summary[
            "candidates"
        ]
        /
        len(qa)
    )

    action_summary.to_csv(
        ACTION_SUMMARY_OUTPUT,
        index=False,
    )

    # ========================================================
    # 5. SAMPLE-GROUP SUMMARY
    #
    # This is important because the 60 records were selected
    # using a STRATIFIED design.
    # ========================================================

    group_summary = (
        qa.groupby(
            [
                "visual_qa_sample_group",
                "qa_final_action",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    action_columns = [
        column
        for column
        in group_summary.columns
        if column
        !=
        "visual_qa_sample_group"
    ]

    group_summary[
        "sample_group_total"
    ] = (
        group_summary[
            action_columns
        ]
        .sum(
            axis=1
        )
    )

    if (
        "retain_candidate"
        in
        group_summary.columns
    ):

        group_summary[
            "retain_percentage_within_group"
        ] = (
            100.0
            *
            group_summary[
                "retain_candidate"
            ]
            /
            group_summary[
                "sample_group_total"
            ]
        )

    group_summary.to_csv(
        GROUP_SUMMARY_OUTPUT,
        index=False,
    )

    # ========================================================
    # 6. SURFACE-SIGNAL SUMMARY
    # ========================================================

    signal_summary = (
        qa.groupby(
            [
                "qa_surface_signal",
                "qa_final_action",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    signal_action_columns = [
        column
        for column
        in signal_summary.columns
        if column
        !=
        "qa_surface_signal"
    ]

    signal_summary[
        "signal_total"
    ] = (
        signal_summary[
            signal_action_columns
        ]
        .sum(
            axis=1
        )
    )

    if (
        "retain_candidate"
        in
        signal_summary.columns
    ):

        signal_summary[
            "retain_percentage_within_signal"
        ] = (
            100.0
            *
            signal_summary[
                "retain_candidate"
            ]
            /
            signal_summary[
                "signal_total"
            ]
        )

    signal_summary.to_csv(
        SIGNAL_SUMMARY_OUTPUT,
        index=False,
    )

    # ========================================================
    # 7. ADD QA RESULTS TO SPATIAL LAYER
    # ========================================================

    spatial = (
        spatial.drop(
            columns=[
                "manual_review_status",
                "manual_qa_observation",
            ],
            errors="ignore",
        )
        .merge(
            qa[
                [
                    "candidate_id",
                    "manual_review_status",
                    "qa_final_visual_category",
                    "qa_final_action",
                    "qa_visual_confidence",
                    "qa_visual_notes",
                ]
            ],
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
    )

    if (
        spatial[
            "qa_final_action"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "One or more spatial QA features "
            "failed to receive a manual QA result."
        )

    spatial.to_file(
        COMPLETED_SPATIAL_OUTPUT,
        driver="GeoJSON",
    )

    # ========================================================
    # 8. AUDIT
    # ========================================================

    retained_count = int(
        (
            qa[
                "qa_final_action"
            ]
            ==
            "retain_candidate"
        ).sum()
    )

    flagged_count = int(
        (
            qa[
                "qa_final_action"
            ]
            ==
            "flag_false_positive"
        ).sum()
    )

    audit = {
        "visual_qa_sample_size":
            len(qa),

        "all_sample_records_reviewed":
            True,

        "retained_in_visual_qa":
            retained_count,

        "flagged_false_positive_in_visual_qa":
            flagged_count,

        "sampling_design":
            "stratified_visual_qa_sample",

        "important_statistical_warning":
            (
                "The overall retained/flagged percentages "
                "from this 60-polygon sample MUST NOT be "
                "treated as population-wide false-positive "
                "or accuracy estimates for all 1,097 "
                "candidate polygons because the sample was "
                "intentionally stratified by priority, "
                "spectral signal and model support."
            ),

        "important_interpretation_warning":
            (
                "retain_candidate means visually plausible "
                "candidate only. It does not mean confirmed "
                "or officially recognized wetland."
            ),

        "model_tuning_after_test":
            False,
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
        "VISUAL QA ACTION SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        action_summary.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "VISUAL QA CATEGORY SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        category_summary.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "RESULTS BY STRATIFIED SAMPLE GROUP"
    )

    print(
        "========================================"
    )

    print(
        group_summary.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "RESULTS BY QA SURFACE SIGNAL"
    )

    print(
        "========================================"
    )

    print(
        signal_summary.to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        COMPLETED_QA_OUTPUT
    )

    print(
        CATEGORY_SUMMARY_OUTPUT
    )

    print(
        ACTION_SUMMARY_OUTPUT
    )

    print(
        GROUP_SUMMARY_OUTPUT
    )

    print(
        SIGNAL_SUMMARY_OUTPUT
    )

    print(
        COMPLETED_SPATIAL_OUTPUT
    )

    print(
        AUDIT_OUTPUT
    )

    print(
        "\n========================================"
    )

    print(
        "VISUAL QA SAMPLE: FINALIZED"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nAll 60 stratified QA candidates now have "
        "a completed visual interpretation."
        "\nDo NOT interpret the overall 60-sample "
        "false-positive percentage as the false-positive "
        "rate of all 1,097 candidates."
        "\nNo model or threshold has been changed."
    )


if __name__ == "__main__":
    main()