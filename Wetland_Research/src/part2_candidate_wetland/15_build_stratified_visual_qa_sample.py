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

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUTS
# ============================================================

CANDIDATE_TABLE_INPUT = (
    REPORT_DIR
    / "14_candidate_qa_polygon_table.csv"
)

CANDIDATE_WGS84_INPUT = (
    MAP_DIR
    / "14_candidate_qa_polygons_mmu_0p10ha_wgs84.geojson"
)


# ============================================================
# OUTPUTS
# ============================================================

STRATIFICATION_OUTPUT = (
    REPORT_DIR
    / "15_candidate_qa_stratification.csv"
)

VISUAL_QA_SAMPLE_OUTPUT = (
    REPORT_DIR
    / "15_candidate_visual_qa_sample.csv"
)

VISUAL_QA_SAMPLE_GEOJSON = (
    MAP_DIR
    / "15_candidate_visual_qa_sample_wgs84.geojson"
)

MANUAL_ANCHOR_OUTPUT = (
    REPORT_DIR
    / "15_manual_qa_anchor_records.csv"
)

AUDIT_OUTPUT = (
    REPORT_DIR
    / "15_candidate_visual_qa_sampling_audit.json"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TOTAL_QA_SAMPLE = 60

TOP_PRIORITY_SAMPLE = 15

WATER_SIGNAL_SAMPLE = 12

VEGETATION_SIGNAL_SAMPLE = 12

MIXED_SIGNAL_SAMPLE = 12

LOWER_SUPPORT_SAMPLE = 9


# ============================================================
# MANUALLY REVIEWED ANCHOR EXAMPLES
#
# These observations are NOT used to train the model or
# determine the spectral thresholds.
#
# They simply preserve the three visual checks already done.
# ============================================================

MANUAL_ANCHORS = {
    "CW_01849": {
        "manual_review_status":
            "reviewed",

        "manual_qa_observation":
            (
                "strong_wetland_like_mixed_lowland"
            ),

        "manual_qa_note":
            (
                "Persistent open/vegetated lowland visible "
                "across historical imagery; likely mixed "
                "with paddy or other lowland vegetation. "
                "Not independently confirmed as wetland."
            ),
    },

    "CW_05540": {
        "manual_review_status":
            "reviewed",

        "manual_qa_observation":
            (
                "open_water_dominated"
            ),

        "manual_qa_note":
            (
                "Google Earth imagery confirms that much "
                "of the component follows Diyawanna Lake "
                "and permanent open-water surfaces."
            ),
    },

    "CW_03482": {
        "manual_review_status":
            "reviewed",

        "manual_qa_observation":
            (
                "open_water_water_edge_dominated"
            ),

        "manual_qa_note":
            (
                "Google Earth imagery shows a large "
                "water/drainage corridor with vegetated "
                "margins and mixed wetland-edge areas."
            ),
    },
}


# ============================================================
# HELPER: SAMPLE WITHOUT DUPLICATING ALREADY SELECTED IDS
# ============================================================

def sample_candidates(
    dataframe,
    n,
    selected_ids,
    random_seed,
):

    available = (
        dataframe[
            ~dataframe[
                "candidate_id"
            ]
            .isin(
                selected_ids
            )
        ]
        .copy()
    )

    if (
        len(available)
        == 0
    ):

        return available

    sample_n = min(
        n,
        len(available),
    )

    return (
        available
        .sample(
            n=sample_n,
            random_state=random_seed,
        )
        .copy()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 STRATIFIED VISUAL QA SAMPLE"
    )

    print(
        "========================================"
    )

    print(
        f"Target QA sample size: "
        f"{TOTAL_QA_SAMPLE}"
    )

    print(
        "\nIMPORTANT:"
        "\nThis script does NOT classify polygons as "
        "confirmed wetlands."
        "\nIt only creates representative visual-QA strata."
        "\nNo model retraining or threshold tuning occurs."
    )

    # ========================================================
    # 1. LOAD CANDIDATE DATA
    # ========================================================

    candidates = pd.read_csv(
        CANDIDATE_TABLE_INPUT
    )

    candidate_map = gpd.read_file(
        CANDIDATE_WGS84_INPUT
    )

    print(
        f"\nCandidate table records: "
        f"{len(candidates):,}"
    )

    print(
        f"Candidate map records: "
        f"{len(candidate_map):,}"
    )

    if (
        len(candidates)
        !=
        len(candidate_map)
    ):

        raise RuntimeError(
            "Candidate table and spatial layer "
            "have different record counts."
        )

    if (
        candidates[
            "candidate_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate candidate IDs found."
        )

    # ========================================================
    # 2. DERIVE SIMPLE QA DESCRIPTORS
    #
    # These are descriptive variables only.
    # ========================================================

    candidates[
        "mean_seasonal_ndvi"
    ] = (
        candidates[
            [
                "mean_dry_ndvi",
                "mean_wet_ndvi",
            ]
        ]
        .mean(
            axis=1
        )
    )

    candidates[
        "maximum_seasonal_mndwi"
    ] = (
        candidates[
            [
                "mean_dry_mndwi",
                "mean_wet_mndwi",
            ]
        ]
        .max(
            axis=1
        )
    )

    candidates[
        "minimum_seasonal_mndwi"
    ] = (
        candidates[
            [
                "mean_dry_mndwi",
                "mean_wet_mndwi",
            ]
        ]
        .min(
            axis=1
        )
    )

    candidates[
        "seasonal_ndvi_change_abs"
    ] = (
        candidates[
            "mean_wet_ndvi"
        ]
        -
        candidates[
            "mean_dry_ndvi"
        ]
    ).abs()

    candidates[
        "seasonal_mndwi_change_abs"
    ] = (
        candidates[
            "mean_wet_mndwi"
        ]
        -
        candidates[
            "mean_dry_mndwi"
        ]
    ).abs()

    # ========================================================
    # 3. DATASET-RELATIVE QUANTILES
    #
    # These thresholds are NOT ecological wetland thresholds.
    #
    # They are used only to divide the 1,097 polygons into
    # contrasting groups for representative visual checking.
    # ========================================================

    ndvi_q25 = float(
        candidates[
            "mean_seasonal_ndvi"
        ]
        .quantile(
            0.25
        )
    )

    ndvi_q50 = float(
        candidates[
            "mean_seasonal_ndvi"
        ]
        .quantile(
            0.50
        )
    )

    ndvi_q75 = float(
        candidates[
            "mean_seasonal_ndvi"
        ]
        .quantile(
            0.75
        )
    )

    mndwi_q25 = float(
        candidates[
            "maximum_seasonal_mndwi"
        ]
        .quantile(
            0.25
        )
    )

    mndwi_q50 = float(
        candidates[
            "maximum_seasonal_mndwi"
        ]
        .quantile(
            0.50
        )
    )

    mndwi_q75 = float(
        candidates[
            "maximum_seasonal_mndwi"
        ]
        .quantile(
            0.75
        )
    )

    probability_q25 = float(
        candidates[
            "mean_probability"
        ]
        .quantile(
            0.25
        )
    )

    probability_q75 = float(
        candidates[
            "mean_probability"
        ]
        .quantile(
            0.75
        )
    )

    area_q25 = float(
        candidates[
            "area_ha"
        ]
        .quantile(
            0.25
        )
    )

    area_q75 = float(
        candidates[
            "area_ha"
        ]
        .quantile(
            0.75
        )
    )

    print(
        "\nDataset-relative QA thresholds:"
    )

    print(
        f"NDVI Q25 / Q50 / Q75: "
        f"{ndvi_q25:.4f} / "
        f"{ndvi_q50:.4f} / "
        f"{ndvi_q75:.4f}"
    )

    print(
        f"MNDWI Q25 / Q50 / Q75: "
        f"{mndwi_q25:.4f} / "
        f"{mndwi_q50:.4f} / "
        f"{mndwi_q75:.4f}"
    )

    print(
        f"Mean probability Q25 / Q75: "
        f"{probability_q25:.4f} / "
        f"{probability_q75:.4f}"
    )

    print(
        f"Area Q25 / Q75: "
        f"{area_q25:.4f} / "
        f"{area_q75:.4f} ha"
    )

    # ========================================================
    # 4. SPECTRAL QA SIGNAL GROUPS
    #
    # WATER-LIKE:
    # relatively high MNDWI + relatively low vegetation
    #
    # VEGETATION-LIKE:
    # relatively high NDVI + lower MNDWI
    #
    # Everything else remains MIXED.
    #
    # Again: these are QA strata, NOT land-cover classes.
    # ========================================================

    water_signal = (
        (
            candidates[
                "maximum_seasonal_mndwi"
            ]
            >=
            mndwi_q75
        )
        &
        (
            candidates[
                "mean_seasonal_ndvi"
            ]
            <=
            ndvi_q50
        )
    )

    vegetation_signal = (
        (
            candidates[
                "mean_seasonal_ndvi"
            ]
            >=
            ndvi_q75
        )
        &
        (
            candidates[
                "maximum_seasonal_mndwi"
            ]
            <=
            mndwi_q50
        )
    )

    candidates[
        "qa_surface_signal"
    ] = "mixed_signal"

    candidates.loc[
        water_signal,
        "qa_surface_signal",
    ] = "water_signal"

    candidates.loc[
        vegetation_signal,
        "qa_surface_signal",
    ] = "vegetation_signal"

    # ========================================================
    # 5. MODEL-SUPPORT TIER
    #
    # Dataset-relative only.
    # ========================================================

    candidates[
        "qa_probability_tier"
    ] = "middle_support"

    candidates.loc[
        candidates[
            "mean_probability"
        ]
        >=
        probability_q75,
        "qa_probability_tier",
    ] = "upper_quartile_support"

    candidates.loc[
        candidates[
            "mean_probability"
        ]
        <=
        probability_q25,
        "qa_probability_tier",
    ] = "lower_quartile_support"

    # ========================================================
    # 6. SIZE TIER
    # ========================================================

    candidates[
        "qa_size_tier"
    ] = "medium"

    candidates.loc[
        candidates[
            "area_ha"
        ]
        >=
        area_q75,
        "qa_size_tier",
    ] = "large"

    candidates.loc[
        candidates[
            "area_ha"
        ]
        <=
        area_q25,
        "qa_size_tier",
    ] = "small"

    # ========================================================
    # 7. PRESERVE CURRENT MANUAL QA ANCHORS
    # ========================================================

    candidates[
        "manual_review_status"
    ] = "pending"

    candidates[
        "manual_qa_observation"
    ] = ""

    candidates[
        "manual_qa_note"
    ] = ""

    anchor_rows = []

    for (
        candidate_id,
        anchor_information,
    ) in MANUAL_ANCHORS.items():

        candidate_match = (
            candidates[
                "candidate_id"
            ]
            ==
            candidate_id
        )

        if (
            not
            candidate_match.any()
        ):

            raise RuntimeError(
                f"Manual QA anchor "
                f"{candidate_id} "
                f"was not found."
            )

        candidates.loc[
            candidate_match,
            "manual_review_status",
        ] = anchor_information[
            "manual_review_status"
        ]

        candidates.loc[
            candidate_match,
            "manual_qa_observation",
        ] = anchor_information[
            "manual_qa_observation"
        ]

        candidates.loc[
            candidate_match,
            "manual_qa_note",
        ] = anchor_information[
            "manual_qa_note"
        ]

        anchor_row = (
            candidates.loc[
                candidate_match
            ]
            .iloc[0]
            .to_dict()
        )

        anchor_rows.append(
            anchor_row
        )

    manual_anchor_table = pd.DataFrame(
        anchor_rows
    )

    manual_anchor_table.to_csv(
        MANUAL_ANCHOR_OUTPUT,
        index=False,
    )

    # ========================================================
    # 8. SAVE FULL STRATIFICATION
    # ========================================================

    candidates.to_csv(
        STRATIFICATION_OUTPUT,
        index=False,
    )

    # ========================================================
    # 9. BUILD 60-POLYGON VISUAL QA SAMPLE
    #
    # Group 1:
    # Top 15 model-priority polygons.
    #
    # Groups 2-4:
    # Representative spectral signal groups.
    #
    # Group 5:
    # Lower-support polygons so visual QA is not biased only
    # toward the strongest model predictions.
    # ========================================================

    selected_parts = []

    selected_ids = set()

    # --------------------------------------------------------
    # A. TOP PRIORITY
    # --------------------------------------------------------

    top_priority = (
        candidates
        .sort_values(
            "qa_order"
        )
        .head(
            TOP_PRIORITY_SAMPLE
        )
        .copy()
    )

    top_priority[
        "visual_qa_sample_group"
    ] = "A_top_priority"

    selected_parts.append(
        top_priority
    )

    selected_ids.update(
        top_priority[
            "candidate_id"
        ]
        .tolist()
    )

    # --------------------------------------------------------
    # B. WATER-SIGNAL SAMPLE
    # --------------------------------------------------------

    water_pool = (
        candidates[
            candidates[
                "qa_surface_signal"
            ]
            ==
            "water_signal"
        ]
    )

    water_sample = sample_candidates(
        water_pool,
        WATER_SIGNAL_SAMPLE,
        selected_ids,
        RANDOM_SEED + 1,
    )

    water_sample[
        "visual_qa_sample_group"
    ] = "B_water_signal"

    selected_parts.append(
        water_sample
    )

    selected_ids.update(
        water_sample[
            "candidate_id"
        ]
        .tolist()
    )

    # --------------------------------------------------------
    # C. VEGETATION-SIGNAL SAMPLE
    # --------------------------------------------------------

    vegetation_pool = (
        candidates[
            candidates[
                "qa_surface_signal"
            ]
            ==
            "vegetation_signal"
        ]
    )

    vegetation_sample = sample_candidates(
        vegetation_pool,
        VEGETATION_SIGNAL_SAMPLE,
        selected_ids,
        RANDOM_SEED + 2,
    )

    vegetation_sample[
        "visual_qa_sample_group"
    ] = "C_vegetation_signal"

    selected_parts.append(
        vegetation_sample
    )

    selected_ids.update(
        vegetation_sample[
            "candidate_id"
        ]
        .tolist()
    )

    # --------------------------------------------------------
    # D. MIXED-SIGNAL SAMPLE
    # --------------------------------------------------------

    mixed_pool = (
        candidates[
            candidates[
                "qa_surface_signal"
            ]
            ==
            "mixed_signal"
        ]
    )

    mixed_sample = sample_candidates(
        mixed_pool,
        MIXED_SIGNAL_SAMPLE,
        selected_ids,
        RANDOM_SEED + 3,
    )

    mixed_sample[
        "visual_qa_sample_group"
    ] = "D_mixed_signal"

    selected_parts.append(
        mixed_sample
    )

    selected_ids.update(
        mixed_sample[
            "candidate_id"
        ]
        .tolist()
    )

    # --------------------------------------------------------
    # E. LOWER-SUPPORT SAMPLE
    # --------------------------------------------------------

    lower_pool = (
        candidates[
            candidates[
                "qa_probability_tier"
            ]
            ==
            "lower_quartile_support"
        ]
    )

    lower_sample = sample_candidates(
        lower_pool,
        LOWER_SUPPORT_SAMPLE,
        selected_ids,
        RANDOM_SEED + 4,
    )

    lower_sample[
        "visual_qa_sample_group"
    ] = "E_lower_support"

    selected_parts.append(
        lower_sample
    )

    selected_ids.update(
        lower_sample[
            "candidate_id"
        ]
        .tolist()
    )

    # ========================================================
    # 10. COMBINE SAMPLE
    # ========================================================

    qa_sample = pd.concat(
        selected_parts,
        axis=0,
        ignore_index=True,
    )

    qa_sample = (
        qa_sample
        .drop_duplicates(
            subset=[
                "candidate_id"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Fill sample if any stratum had too few candidates.
    # --------------------------------------------------------

    if (
        len(qa_sample)
        <
        TOTAL_QA_SAMPLE
    ):

        remaining_needed = (
            TOTAL_QA_SAMPLE
            -
            len(qa_sample)
        )

        remaining_pool = (
            candidates[
                ~candidates[
                    "candidate_id"
                ]
                .isin(
                    qa_sample[
                        "candidate_id"
                    ]
                )
            ]
            .copy()
        )

        fallback = (
            remaining_pool
            .sample(
                n=min(
                    remaining_needed,
                    len(
                        remaining_pool
                    ),
                ),
                random_state=(
                    RANDOM_SEED + 5
                ),
            )
            .copy()
        )

        fallback[
            "visual_qa_sample_group"
        ] = "F_fallback_fill"

        qa_sample = pd.concat(
            [
                qa_sample,
                fallback,
            ],
            axis=0,
            ignore_index=True,
        )

    # --------------------------------------------------------
    # Hard cap at target size.
    # --------------------------------------------------------

    qa_sample = (
        qa_sample
        .head(
            TOTAL_QA_SAMPLE
        )
        .copy()
    )

    qa_sample[
        "visual_qa_sequence"
    ] = np.arange(
        1,
        len(qa_sample) + 1,
    )

    # ========================================================
    # 11. ADD EMPTY MANUAL REVIEW FIELDS
    #
    # These will be filled later from Google Earth QA.
    # ========================================================

    qa_sample[
        "qa_final_visual_category"
    ] = ""

    qa_sample[
        "qa_imagery_dates_checked"
    ] = ""

    qa_sample[
        "qa_visual_confidence"
    ] = ""

    qa_sample[
        "qa_visual_notes"
    ] = ""

    # Keep existing anchor observations.
    for (
        candidate_id,
        anchor_information,
    ) in MANUAL_ANCHORS.items():

        anchor_mask = (
            qa_sample[
                "candidate_id"
            ]
            ==
            candidate_id
        )

        if (
            anchor_mask.any()
        ):

            qa_sample.loc[
                anchor_mask,
                "qa_visual_notes",
            ] = anchor_information[
                "manual_qa_note"
            ]

    qa_sample.to_csv(
        VISUAL_QA_SAMPLE_OUTPUT,
        index=False,
    )

    # ========================================================
    # 12. BUILD SPATIAL QA SAMPLE
    # ========================================================

    spatial_sample = (
        candidate_map[
            candidate_map[
                "candidate_id"
            ]
            .isin(
                qa_sample[
                    "candidate_id"
                ]
            )
        ]
        .copy()
    )

    sample_attribute_columns = [
        "candidate_id",
        "visual_qa_sequence",
        "visual_qa_sample_group",
        "qa_surface_signal",
        "qa_probability_tier",
        "qa_size_tier",
        "manual_review_status",
        "manual_qa_observation",
    ]

    spatial_sample = (
        spatial_sample.drop(
            columns=[
                column
                for column
                in sample_attribute_columns
                if (
                    column
                    in spatial_sample.columns
                    and
                    column
                    != "candidate_id"
                )
            ],
            errors="ignore",
        )
        .merge(
            qa_sample[
                sample_attribute_columns
            ],
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
    )

    spatial_sample = (
        spatial_sample
        .sort_values(
            "visual_qa_sequence"
        )
        .reset_index(
            drop=True
        )
    )

    spatial_sample.to_file(
        VISUAL_QA_SAMPLE_GEOJSON,
        driver="GeoJSON",
    )

    # ========================================================
    # 13. QA
    # ========================================================

    if (
        qa_sample[
            "candidate_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate candidates exist "
            "in visual QA sample."
        )

    if (
        len(
            spatial_sample
        )
        !=
        len(
            qa_sample
        )
    ):

        raise RuntimeError(
            "Spatial QA sample count does "
            "not match table sample count."
        )

    # ========================================================
    # 14. SUMMARY TABLES
    # ========================================================

    surface_counts = (
        candidates[
            "qa_surface_signal"
        ]
        .value_counts()
    )

    sample_group_counts = (
        qa_sample[
            "visual_qa_sample_group"
        ]
        .value_counts()
    )

    reviewed_anchor_count = int(
        (
            qa_sample[
                "manual_review_status"
            ]
            ==
            "reviewed"
        ).sum()
    )

    # ========================================================
    # 15. AUDIT
    # ========================================================

    audit = {
        "random_seed":
            RANDOM_SEED,

        "all_candidate_count":
            int(
                len(
                    candidates
                )
            ),

        "target_visual_qa_sample_size":
            TOTAL_QA_SAMPLE,

        "actual_visual_qa_sample_size":
            int(
                len(
                    qa_sample
                )
            ),

        "already_reviewed_anchor_count":
            reviewed_anchor_count,

        "manual_anchor_candidate_ids":
            list(
                MANUAL_ANCHORS.keys()
            ),

        "sampling_design":
            {
                "top_priority":
                    TOP_PRIORITY_SAMPLE,

                "water_signal":
                    WATER_SIGNAL_SAMPLE,

                "vegetation_signal":
                    VEGETATION_SIGNAL_SAMPLE,

                "mixed_signal":
                    MIXED_SIGNAL_SAMPLE,

                "lower_support":
                    LOWER_SUPPORT_SAMPLE,
            },

        "dataset_relative_thresholds":
            {
                "ndvi_q25":
                    ndvi_q25,

                "ndvi_q50":
                    ndvi_q50,

                "ndvi_q75":
                    ndvi_q75,

                "mndwi_q25":
                    mndwi_q25,

                "mndwi_q50":
                    mndwi_q50,

                "mndwi_q75":
                    mndwi_q75,

                "probability_q25":
                    probability_q25,

                "probability_q75":
                    probability_q75,

                "area_q25_ha":
                    area_q25,

                "area_q75_ha":
                    area_q75,
            },

        "important_note":
            (
                "QA strata are dataset-relative sampling "
                "groups only. They are not wetland classes "
                "and must not be interpreted as automatic "
                "confirmation or rejection rules."
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
        "FULL CANDIDATE QA STRATIFICATION"
    )

    print(
        "========================================"
    )

    print(
        "\nSurface-signal groups:"
    )

    print(
        surface_counts.to_string()
    )

    print(
        "\n========================================"
    )

    print(
        "VISUAL QA SAMPLE"
    )

    print(
        "========================================"
    )

    print(
        f"Selected polygons: "
        f"{len(qa_sample)}"
    )

    print(
        f"Already reviewed anchors: "
        f"{reviewed_anchor_count}"
    )

    print(
        "\nSample-group counts:"
    )

    print(
        sample_group_counts.to_string()
    )

    print(
        "\nSelected candidates:"
    )

    display_columns = [
        "visual_qa_sequence",
        "candidate_id",
        "visual_qa_sample_group",
        "qa_surface_signal",
        "qa_probability_tier",
        "qa_size_tier",
        "area_ha",
        "mean_probability",
        "mean_seasonal_ndvi",
        "maximum_seasonal_mndwi",
        "manual_review_status",
        "manual_qa_observation",
    ]

    print(
        qa_sample[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        STRATIFICATION_OUTPUT
    )

    print(
        VISUAL_QA_SAMPLE_OUTPUT
    )

    print(
        VISUAL_QA_SAMPLE_GEOJSON
    )

    print(
        MANUAL_ANCHOR_OUTPUT
    )

    print(
        AUDIT_OUTPUT
    )

    print(
        "\n========================================"
    )

    print(
        "STRATIFIED VISUAL QA SAMPLE: COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nDo NOT visually inspect all 1,097 candidates."
        "\nThe 60-polygons sample is designed to cover "
        "contrasting candidate conditions."
        "\nThe three candidates already reviewed are "
        "preserved as anchor examples."
        "\nVisual QA must not be used to retune the "
        "locked XGBoost model or threshold."
    )


if __name__ == "__main__":
    main()