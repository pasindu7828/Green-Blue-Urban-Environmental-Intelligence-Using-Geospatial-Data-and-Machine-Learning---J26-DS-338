"""
STEP 12E - FINALIZE URBANIZATION-PRESSURE PREDICTOR SET

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Start from the Step 12C dataset containing 69 predictors.
2. Remove exact/deterministic and strongly redundant predictors identified
   by the unsupervised Step 12D audit.
3. Reclassify LULC coverage variables as QA fields rather than ML predictors.
4. Produce the final frozen predictor dataset and feature list.
5. Do NOT use the target to select predictors.

Expected result:
- 51,310 modelling rows
- 844 positive target rows
- 55 final ML predictors

IMPORTANT:
- No target-based feature selection is performed.
- No future information is introduced.
- No centered stabilized LULC is used.
- No 2026 OSM roads/buildings are used.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "predictors"
    / "12c_causal_lulc_terrain_spatial_predictors.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "predictors"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. OUTPUTS
# =============================================================================

FINAL_DATASET_CSV = (
    OUTPUT_DIR
    / "12e_final_urbanization_pressure_predictors.csv"
)

FINAL_FEATURE_LIST_CSV = (
    REPORT_DIR
    / "12e_final_predictor_feature_list.csv"
)

REMOVAL_DECISIONS_CSV = (
    REPORT_DIR
    / "12e_predictor_removal_decisions.csv"
)

FINAL_CORRELATION_CSV = (
    REPORT_DIR
    / "12e_final_high_correlation_pairs.csv"
)

QA_CSV = (
    REPORT_DIR
    / "12e_final_predictor_set_qa.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "12e_final_predictor_set_summary.csv"
)


# =============================================================================
# 3. EXPECTED VALUES
# =============================================================================

EXPECTED_ROWS = 51310
EXPECTED_POSITIVE_ROWS = 844

EXPECTED_INPUT_FEATURES = 69
EXPECTED_FINAL_FEATURES = 55

HIGH_CORRELATION_THRESHOLD = 0.98


# =============================================================================
# 4. COLUMN GROUPS
# =============================================================================

METADATA_COLUMNS = [
    "grid_id",
    "grid_row",
    "grid_col",
    "origin_year",
    "urban_conversion_target",
    "kmc_overlap_fraction",
    "boundary_cell",
]


# These remain in the dataset for QA only.
# They must NOT enter RF/XGBoost.
QA_ONLY_COLUMNS = [
    "elevation_valid_coverage_fraction",
    "slope_valid_coverage_fraction",
    "origin_lulc_coverage",
    "minimum_past_3yr_lulc_coverage",
]


# Predictors physically removed from the final ML dataset.
DROP_PREDICTORS = {
    "green_slope_3yr": (
        "Exact deterministic duplicate of green_change_2yr / 2; "
        "retain green_change_2yr for simpler interpretation."
    ),

    "built_slope_3yr": (
        "Exact deterministic duplicate of built_change_2yr / 2; "
        "retain built_change_2yr."
    ),

    "open_bare_slope_3yr": (
        "Exact deterministic duplicate of open_bare_change_2yr / 2; "
        "retain open_bare_change_2yr."
    ),

    "water_slope_3yr": (
        "Exact deterministic duplicate of water_change_2yr / 2; "
        "retain water_change_2yr."
    ),

    "nbr8_built_slope_3yr": (
        "Exact deterministic duplicate of nbr8_built_change_2yr / 2; "
        "retain neighbourhood change."
    ),

    "terrestrial_nonbuilt_fraction_t": (
        "Deterministic combination of green_fraction_t and "
        "open_bare_fraction_t."
    ),

    "nonwater_fraction_t": (
        "Exact deterministic transformation of water_fraction_t."
    ),

    "built_minus_green_t": (
        "Deterministic combination of built_fraction_t and "
        "green_fraction_t."
    ),

    "built_green_ratio_t": (
        "Very strongly rank-correlated with current Built-up fraction "
        "and less stable/interpretable near very small Green fractions."
    ),

    "open_bare_max_3yr": (
        "Near-redundant with open_bare_mean_3yr in Step 12D; "
        "retain the 3-year mean."
    ),

    "water_max_3yr": (
        "Near-redundant with water_mean_3yr in Step 12D; "
        "retain the 3-year mean."
    ),

    "elevation_std_m": (
        "Very highly correlated with terrain slope/range; retain more "
        "directly interpretable elevation and slope descriptors."
    ),
}


QA_RECLASSIFICATION_REASONS = {
    "origin_lulc_coverage": (
        "Data-quality / observation-coverage variable. Retained for QA "
        "but excluded from ML to avoid learning sensor-coverage patterns."
    ),

    "minimum_past_3yr_lulc_coverage": (
        "Data-quality / observation-coverage variable. Retained for QA "
        "but excluded from ML."
    ),
}


# =============================================================================
# 5. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """Return project-relative path where possible."""

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 12E - FINALIZE URBANIZATION-PRESSURE PREDICTOR SET")
print("=" * 100)


if not INPUT_DATASET.exists():

    raise FileNotFoundError(
        f"Step 12C dataset not found:\n"
        f"{INPUT_DATASET}"
    )


# =============================================================================
# 7. LOAD STEP 12C DATA
# =============================================================================

df = pd.read_csv(
    INPUT_DATASET
)


if len(
    df
) != EXPECTED_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} rows, "
        f"found {len(df):,}."
    )


positive_count = int(
    df[
        "urban_conversion_target"
    ].sum()
)


if positive_count != EXPECTED_POSITIVE_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_POSITIVE_ROWS} positive rows, "
        f"found {positive_count}."
    )


if df.duplicated(
    subset=[
        "grid_id",
        "origin_year",
    ]
).any():

    raise RuntimeError(
        "Duplicate grid/origin rows found."
    )


input_feature_columns = [
    column
    for column in df.columns
    if (
        column not in METADATA_COLUMNS
        and
        column not in [
            "elevation_valid_coverage_fraction",
            "slope_valid_coverage_fraction",
        ]
    )
]


if len(
    input_feature_columns
) != EXPECTED_INPUT_FEATURES:

    raise RuntimeError(
        f"Expected {EXPECTED_INPUT_FEATURES} Step 12C predictors, "
        f"found {len(input_feature_columns)}."
    )


print(
    f"\nRows                  : "
    f"{len(df):,}"
)

print(
    f"Positive target rows  : "
    f"{positive_count:,}"
)

print(
    f"Input predictors      : "
    f"{len(input_feature_columns)}"
)


# =============================================================================
# 8. VERIFY REMOVAL COLUMNS
# =============================================================================

missing_drop_columns = [
    column
    for column in DROP_PREDICTORS
    if column not in df.columns
]


if missing_drop_columns:

    raise RuntimeError(
        "Expected predictors for removal are missing:\n"
        +
        "\n".join(
            missing_drop_columns
        )
    )


missing_qa_columns = [
    column
    for column in QA_ONLY_COLUMNS
    if column not in df.columns
]


if missing_qa_columns:

    raise RuntimeError(
        "Expected QA columns are missing:\n"
        +
        "\n".join(
            missing_qa_columns
        )
    )


# =============================================================================
# 9. CREATE FINAL DATASET
# =============================================================================

final_df = df.drop(
    columns=list(
        DROP_PREDICTORS.keys()
    )
).copy()


# =============================================================================
# 10. DEFINE FINAL MODEL FEATURES
# =============================================================================

final_feature_columns = [
    column
    for column in final_df.columns
    if (
        column not in METADATA_COLUMNS
        and
        column not in QA_ONLY_COLUMNS
    )
]


if len(
    final_feature_columns
) != EXPECTED_FINAL_FEATURES:

    raise RuntimeError(
        f"Expected {EXPECTED_FINAL_FEATURES} final predictors, "
        f"found {len(final_feature_columns)}."
    )


# =============================================================================
# 11. REMOVAL DECISION REPORT
# =============================================================================

removal_records = []


for feature_name, reason in DROP_PREDICTORS.items():

    removal_records.append(
        {
            "feature_name": (
                feature_name
            ),
            "decision": (
                "REMOVE_FROM_FINAL_DATASET"
            ),
            "reason": (
                reason
            ),
            "target_used_for_decision": (
                False
            ),
        }
    )


for feature_name, reason in QA_RECLASSIFICATION_REASONS.items():

    removal_records.append(
        {
            "feature_name": (
                feature_name
            ),
            "decision": (
                "RETAIN_AS_QA_ONLY_NOT_MODEL_FEATURE"
            ),
            "reason": (
                reason
            ),
            "target_used_for_decision": (
                False
            ),
        }
    )


removal_decision_df = pd.DataFrame(
    removal_records
)


# =============================================================================
# 12. FINAL FEATURE GROUPS
# =============================================================================

feature_list_records = []


for feature_name in final_feature_columns:

    if feature_name.startswith(
        "nbr8_"
    ) or feature_name.startswith(
        "nbr24_"
    ):

        feature_group = (
            "CAUSAL_SPATIAL_CONTEXT"
        )

        temporal_role = (
            "T_MINUS_2_TO_T"
        )

    elif feature_name.startswith(
        "elevation_"
    ) or feature_name.startswith(
        "slope_"
    ):

        feature_group = (
            "STATIC_TERRAIN"
        )

        temporal_role = (
            "STATIC"
        )

    else:

        feature_group = (
            "CAUSAL_RAW_LULC"
        )

        temporal_role = (
            "T_MINUS_2_TO_T"
        )


    feature_list_records.append(
        {
            "feature_name": (
                feature_name
            ),
            "feature_group": (
                feature_group
            ),
            "temporal_role": (
                temporal_role
            ),
            "included_in_final_ml": (
                True
            ),
            "future_information_used": (
                False
            ),
            "centered_stabilized_lulc_used": (
                False
            ),
            "osm_2026_used": (
                False
            ),
        }
    )


final_feature_list_df = pd.DataFrame(
    feature_list_records
)


# =============================================================================
# 13. FINAL CORRELATION AUDIT
# =============================================================================

feature_matrix = (
    final_df[
        final_feature_columns
    ]
    .astype(
        float
    )
)


pearson_corr = feature_matrix.corr(
    method="pearson"
)


spearman_corr = feature_matrix.corr(
    method="spearman"
)


high_corr_records = []


for i in range(
    len(
        final_feature_columns
    )
):

    feature_a = (
        final_feature_columns[
            i
        ]
    )


    for j in range(
        i + 1,
        len(
            final_feature_columns
        ),
    ):

        feature_b = (
            final_feature_columns[
                j
            ]
        )


        pearson_value = float(
            pearson_corr.loc[
                feature_a,
                feature_b,
            ]
        )


        spearman_value = float(
            spearman_corr.loc[
                feature_a,
                feature_b,
            ]
        )


        maximum_absolute_correlation = max(
            abs(
                pearson_value
            ),
            abs(
                spearman_value
            ),
        )


        if (
            maximum_absolute_correlation
            >=
            HIGH_CORRELATION_THRESHOLD
        ):

            high_corr_records.append(
                {
                    "feature_a": (
                        feature_a
                    ),
                    "feature_b": (
                        feature_b
                    ),
                    "pearson_correlation": (
                        pearson_value
                    ),
                    "spearman_correlation": (
                        spearman_value
                    ),
                    "maximum_absolute_correlation": (
                        maximum_absolute_correlation
                    ),
                }
            )


final_high_corr_df = pd.DataFrame(
    high_corr_records,
    columns=[
        "feature_a",
        "feature_b",
        "pearson_correlation",
        "spearman_correlation",
        "maximum_absolute_correlation",
    ],
)


if len(
    final_high_corr_df
) > 0:

    final_high_corr_df = (
        final_high_corr_df.sort_values(
            "maximum_absolute_correlation",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# 14. QA
# =============================================================================

row_count_unchanged = (
    len(
        final_df
    )
    ==
    EXPECTED_ROWS
)


positive_count_unchanged = (
    int(
        final_df[
            "urban_conversion_target"
        ].sum()
    )
    ==
    EXPECTED_POSITIVE_ROWS
)


target_unchanged = bool(
    np.array_equal(
        final_df[
            "urban_conversion_target"
        ].to_numpy(),
        df[
            "urban_conversion_target"
        ].to_numpy(),
    )
)


duplicate_grid_origin_rows = int(
    final_df.duplicated(
        subset=[
            "grid_id",
            "origin_year",
        ]
    ).sum()
)


missing_final_feature_values = int(
    final_df[
        final_feature_columns
    ]
    .isna()
    .sum()
    .sum()
)


infinite_final_feature_values = int(
    np.isinf(
        final_df[
            final_feature_columns
        ]
        .to_numpy(
            dtype=float
        )
    ).sum()
)


removed_predictors_absent = all(
    column
    not in
    final_df.columns
    for column in DROP_PREDICTORS
)


qa_columns_not_in_feature_list = all(
    column
    not in
    final_feature_columns
    for column in QA_ONLY_COLUMNS
)


target_not_in_feature_list = (
    "urban_conversion_target"
    not in
    final_feature_columns
)


grid_identifiers_not_in_feature_list = all(
    column
    not in
    final_feature_columns
    for column in [
        "grid_id",
        "grid_row",
        "grid_col",
    ]
)


no_future_named_predictors = bool(
    not any(
        (
            "future"
            in column.lower()
            or
            "target_window"
            in column.lower()
        )
        for column in final_feature_columns
    )
)


feature_count_valid = (
    len(
        final_feature_columns
    )
    ==
    EXPECTED_FINAL_FEATURES
)


overall_status = (
    "PASS"
    if all(
        [
            row_count_unchanged,
            positive_count_unchanged,
            target_unchanged,
            duplicate_grid_origin_rows
            ==
            0,
            missing_final_feature_values
            ==
            0,
            infinite_final_feature_values
            ==
            0,
            removed_predictors_absent,
            qa_columns_not_in_feature_list,
            target_not_in_feature_list,
            grid_identifiers_not_in_feature_list,
            no_future_named_predictors,
            feature_count_valid,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 15. QA REPORT
# =============================================================================

qa_df = pd.DataFrame(
    [
        {
            "rows": (
                len(
                    final_df
                )
            ),
            "input_feature_count": (
                EXPECTED_INPUT_FEATURES
            ),
            "physically_removed_feature_count": (
                len(
                    DROP_PREDICTORS
                )
            ),
            "qa_only_reclassified_feature_count": (
                len(
                    QA_RECLASSIFICATION_REASONS
                )
            ),
            "final_ml_feature_count": (
                len(
                    final_feature_columns
                )
            ),
            "positive_target_rows": (
                int(
                    final_df[
                        "urban_conversion_target"
                    ].sum()
                )
            ),
            "duplicate_grid_origin_rows": (
                duplicate_grid_origin_rows
            ),
            "missing_final_feature_values": (
                missing_final_feature_values
            ),
            "infinite_final_feature_values": (
                infinite_final_feature_values
            ),
            "target_unchanged": (
                target_unchanged
            ),
            "removed_predictors_absent": (
                removed_predictors_absent
            ),
            "qa_columns_excluded_from_ml": (
                qa_columns_not_in_feature_list
            ),
            "target_excluded_from_ml": (
                target_not_in_feature_list
            ),
            "grid_identifiers_excluded_from_ml": (
                grid_identifiers_not_in_feature_list
            ),
            "remaining_high_correlation_pair_count_abs_ge_0_98": (
                len(
                    final_high_corr_df
                )
            ),
            "target_used_for_feature_selection": (
                False
            ),
            "osm_2026_used": (
                False
            ),
            "overall_step12e_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 16. FINAL SUMMARY
# =============================================================================

group_counts = (
    final_feature_list_df[
        "feature_group"
    ]
    .value_counts()
    .to_dict()
)


final_summary_df = pd.DataFrame(
    [
        {
            "input_dataset": (
                "12c_causal_lulc_terrain_spatial_predictors.csv"
            ),
            "initial_predictor_count": (
                EXPECTED_INPUT_FEATURES
            ),
            "final_predictor_count": (
                len(
                    final_feature_columns
                )
            ),
            "causal_raw_lulc_features": (
                int(
                    group_counts.get(
                        "CAUSAL_RAW_LULC",
                        0,
                    )
                )
            ),
            "static_terrain_features": (
                int(
                    group_counts.get(
                        "STATIC_TERRAIN",
                        0,
                    )
                )
            ),
            "causal_spatial_context_features": (
                int(
                    group_counts.get(
                        "CAUSAL_SPATIAL_CONTEXT",
                        0,
                    )
                )
            ),
            "model_rows": (
                len(
                    final_df
                )
            ),
            "positive_target_rows": (
                int(
                    final_df[
                        "urban_conversion_target"
                    ].sum()
                )
            ),
            "target_used_for_feature_selection": (
                False
            ),
            "future_information_used": (
                False
            ),
            "centered_stabilized_lulc_used": (
                False
            ),
            "osm_2026_used": (
                False
            ),
            "predictor_set_frozen_after_this_step": (
                overall_status
                ==
                "PASS"
            ),
            "overall_step12e_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 17. SAVE OUTPUTS
# =============================================================================

final_df.to_csv(
    FINAL_DATASET_CSV,
    index=False,
)


final_feature_list_df.to_csv(
    FINAL_FEATURE_LIST_CSV,
    index=False,
)


removal_decision_df.to_csv(
    REMOVAL_DECISIONS_CSV,
    index=False,
)


final_high_corr_df.to_csv(
    FINAL_CORRELATION_CSV,
    index=False,
)


qa_df.to_csv(
    QA_CSV,
    index=False,
)


final_summary_df.to_csv(
    FINAL_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 18. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("PREDICTOR DECISIONS")
print("=" * 100)


print(
    removal_decision_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("FINAL PREDICTOR SET")
print("=" * 100)


print(
    f"Initial predictors               : "
    f"{EXPECTED_INPUT_FEATURES}"
)

print(
    f"Physically removed               : "
    f"{len(DROP_PREDICTORS)}"
)

print(
    f"Reclassified as QA-only          : "
    f"{len(QA_RECLASSIFICATION_REASONS)}"
)

print(
    f"Final ML predictors              : "
    f"{len(final_feature_columns)}"
)

print(
    f"Rows                             : "
    f"{len(final_df):,}"
)

print(
    f"Positive cases                   : "
    f"{final_df['urban_conversion_target'].sum():,}"
)


print("\nFinal feature groups:")

for group_name, count in (
    final_feature_list_df[
        "feature_group"
    ]
    .value_counts()
    .items()
):

    print(
        f" - {group_name}: "
        f"{count}"
    )


print("\n" + "=" * 100)
print("REMAINING HIGH CORRELATION PAIRS |r| >= 0.98")
print("=" * 100)


if len(
    final_high_corr_df
) == 0:

    print(
        "None."
    )

else:

    print(
        final_high_corr_df.to_string(
            index=False
        )
    )


print("\n" + "=" * 100)
print("STEP 12E QA")
print("=" * 100)


print(
    f"Rows unchanged                     : "
    f"{row_count_unchanged}"
)

print(
    f"Positive count unchanged           : "
    f"{positive_count_unchanged}"
)

print(
    f"Target unchanged                   : "
    f"{target_unchanged}"
)

print(
    f"Duplicate grid/origin rows         : "
    f"{duplicate_grid_origin_rows}"
)

print(
    f"Missing final feature values       : "
    f"{missing_final_feature_values}"
)

print(
    f"Infinite final feature values      : "
    f"{infinite_final_feature_values}"
)

print(
    f"Removed predictors absent          : "
    f"{removed_predictors_absent}"
)

print(
    f"QA columns excluded from ML        : "
    f"{qa_columns_not_in_feature_list}"
)

print(
    f"Target excluded from ML            : "
    f"{target_not_in_feature_list}"
)

print(
    f"Grid IDs/coordinates excluded      : "
    f"{grid_identifiers_not_in_feature_list}"
)

print(
    f"Final feature count = 55           : "
    f"{feature_count_valid}"
)

print(
    f"Future-named predictors            : "
    f"{not no_future_named_predictors}"
)

print(
    f"Target used for feature selection  : "
    f"False"
)

print(
    f"OSM 2026 used                      : "
    f"False"
)

print(
    f"Overall Step 12E status            : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(FINAL_DATASET_CSV)}"
)

print(
    f" - {relative_path(FINAL_FEATURE_LIST_CSV)}"
)

print(
    f" - {relative_path(REMOVAL_DECISIONS_CSV)}"
)

print(
    f" - {relative_path(FINAL_CORRELATION_CSV)}"
)

print(
    f" - {relative_path(QA_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. The target was not used to choose predictors."
)

print(
    "2. Exact/deterministic redundancy was reduced for cleaner RF/XGBoost and SHAP interpretation."
)

print(
    "3. LULC coverage variables remain only as QA fields, not model predictors."
)

print(
    "4. Current 2026 OSM data remains excluded from the historical pressure model."
)

print(
    "5. If this step passes, the predictor set is frozen before validation/model training."
)

print(
    "6. The next step will create leakage-aware temporal and spatial validation partitions."
)

print("=" * 100)