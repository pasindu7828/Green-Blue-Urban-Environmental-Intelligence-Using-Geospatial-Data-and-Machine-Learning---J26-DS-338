"""
STEP 12D - PREDICTOR REDUNDANCY AUDIT

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Audit the current 69 causal/static predictor features.
2. Detect exact duplicate predictors.
3. Detect constant / near-constant predictors.
4. Detect very highly correlated predictor pairs.
5. Identify deterministic feature relationships that may make
   interpretation / SHAP unstable.
6. Do NOT use the target for feature selection.
7. Do NOT automatically delete predictors in this step.

Input:
- Step 12C causal LULC + terrain + spatial-context dataset.

Important:
- This is an UNSUPERVISED feature audit.
- The target is not used to rank or select features.
- No model is trained.
- No feature is automatically removed.
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

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. OUTPUTS
# =============================================================================

FEATURE_PROFILE_CSV = (
    REPORT_DIR
    / "12d_predictor_feature_profile.csv"
)

EXACT_DUPLICATE_CSV = (
    REPORT_DIR
    / "12d_exact_duplicate_predictor_pairs.csv"
)

HIGH_CORRELATION_CSV = (
    REPORT_DIR
    / "12d_high_correlation_predictor_pairs.csv"
)

DETERMINISTIC_RELATIONSHIP_CSV = (
    REPORT_DIR
    / "12d_known_deterministic_relationship_audit.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "12d_predictor_redundancy_audit_summary.csv"
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

EXPECTED_ROWS = 51310
EXPECTED_POSITIVES = 844
EXPECTED_FEATURES = 69

NEAR_CONSTANT_STD_THRESHOLD = 1e-8

HIGH_CORRELATION_THRESHOLD = 0.98

EXACT_TOLERANCE = 1e-10


METADATA_COLUMNS = [
    "grid_id",
    "grid_row",
    "grid_col",
    "origin_year",
    "urban_conversion_target",
    "kmc_overlap_fraction",
    "boundary_cell",
]


QA_ONLY_COLUMNS = [
    "elevation_valid_coverage_fraction",
    "slope_valid_coverage_fraction",
]


# =============================================================================
# 4. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """Return project-relative path where possible."""

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def max_abs_difference(
    first,
    second,
):
    """Return maximum absolute finite difference."""

    first = np.asarray(
        first,
        dtype=float,
    )

    second = np.asarray(
        second,
        dtype=float,
    )

    valid = (
        np.isfinite(
            first
        )
        &
        np.isfinite(
            second
        )
    )

    if not np.any(
        valid
    ):
        return np.nan

    return float(
        np.max(
            np.abs(
                first[
                    valid
                ]
                -
                second[
                    valid
                ]
            )
        )
    )


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 12D - PREDICTOR REDUNDANCY AUDIT")
print("=" * 100)


if not INPUT_DATASET.exists():

    raise FileNotFoundError(
        f"Step 12C dataset not found:\n"
        f"{INPUT_DATASET}"
    )


# =============================================================================
# 6. LOAD DATA
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


if positive_count != EXPECTED_POSITIVES:

    raise RuntimeError(
        f"Expected {EXPECTED_POSITIVES} positive rows, "
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


feature_columns = [
    column
    for column in df.columns
    if (
        column not in METADATA_COLUMNS
        and
        column not in QA_ONLY_COLUMNS
    )
]


if len(
    feature_columns
) != EXPECTED_FEATURES:

    raise RuntimeError(
        f"Expected {EXPECTED_FEATURES} model predictors, "
        f"found {len(feature_columns)}."
    )


print(
    f"\nRows                 : "
    f"{len(df):,}"
)

print(
    f"Positive target rows : "
    f"{positive_count:,}"
)

print(
    f"Predictors audited   : "
    f"{len(feature_columns)}"
)


# =============================================================================
# 7. BASIC FEATURE PROFILE
# =============================================================================

feature_profile_records = []


for feature_name in feature_columns:

    values = (
        df[
            feature_name
        ]
        .to_numpy(
            dtype=float
        )
    )


    finite_values = (
        values[
            np.isfinite(
                values
            )
        ]
    )


    missing_count = int(
        np.isnan(
            values
        ).sum()
    )


    infinite_count = int(
        np.isinf(
            values
        ).sum()
    )


    unique_count = int(
        pd.Series(
            values
        )
        .nunique(
            dropna=True
        )
    )


    if len(
        finite_values
    ) > 0:

        mean_value = float(
            np.mean(
                finite_values
            )
        )

        std_value = float(
            np.std(
                finite_values,
                ddof=0,
            )
        )

        minimum = float(
            np.min(
                finite_values
            )
        )

        maximum = float(
            np.max(
                finite_values
            )
        )

    else:

        mean_value = np.nan
        std_value = np.nan
        minimum = np.nan
        maximum = np.nan


    is_constant = (
        unique_count
        <=
        1
    )


    is_near_constant = (
        np.isfinite(
            std_value
        )
        and
        std_value
        <=
        NEAR_CONSTANT_STD_THRESHOLD
    )


    feature_profile_records.append(
        {
            "feature_name": (
                feature_name
            ),
            "missing_count": (
                missing_count
            ),
            "infinite_count": (
                infinite_count
            ),
            "unique_count": (
                unique_count
            ),
            "mean": (
                mean_value
            ),
            "std": (
                std_value
            ),
            "min": (
                minimum
            ),
            "max": (
                maximum
            ),
            "is_constant": (
                is_constant
            ),
            "is_near_constant": (
                is_near_constant
            ),
        }
    )


feature_profile_df = pd.DataFrame(
    feature_profile_records
)


constant_features = (
    feature_profile_df.loc[
        feature_profile_df[
            "is_constant"
        ],
        "feature_name",
    ]
    .tolist()
)


near_constant_features = (
    feature_profile_df.loc[
        feature_profile_df[
            "is_near_constant"
        ],
        "feature_name",
    ]
    .tolist()
)


# =============================================================================
# 8. EXACT DUPLICATE FEATURE AUDIT
# =============================================================================

exact_duplicate_records = []


for i in range(
    len(
        feature_columns
    )
):

    feature_a = (
        feature_columns[
            i
        ]
    )


    values_a = (
        df[
            feature_a
        ]
        .to_numpy(
            dtype=float
        )
    )


    for j in range(
        i + 1,
        len(
            feature_columns
        ),
    ):

        feature_b = (
            feature_columns[
                j
            ]
        )


        values_b = (
            df[
                feature_b
            ]
            .to_numpy(
                dtype=float
            )
        )


        difference = max_abs_difference(
            values_a,
            values_b,
        )


        if (
            np.isfinite(
                difference
            )
            and
            difference
            <=
            EXACT_TOLERANCE
        ):

            exact_duplicate_records.append(
                {
                    "feature_a": (
                        feature_a
                    ),
                    "feature_b": (
                        feature_b
                    ),
                    "maximum_absolute_difference": (
                        difference
                    ),
                }
            )


exact_duplicate_df = pd.DataFrame(
    exact_duplicate_records,
    columns=[
        "feature_a",
        "feature_b",
        "maximum_absolute_difference",
    ],
)


# =============================================================================
# 9. HIGH CORRELATION AUDIT
# =============================================================================

feature_matrix = (
    df[
        feature_columns
    ]
    .astype(
        float
    )
)


pearson_corr = (
    feature_matrix.corr(
        method="pearson"
    )
)


spearman_corr = (
    feature_matrix.corr(
        method="spearman"
    )
)


high_correlation_records = []


for i in range(
    len(
        feature_columns
    )
):

    feature_a = (
        feature_columns[
            i
        ]
    )


    for j in range(
        i + 1,
        len(
            feature_columns
        ),
    ):

        feature_b = (
            feature_columns[
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

            high_correlation_records.append(
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


high_correlation_df = pd.DataFrame(
    high_correlation_records,
    columns=[
        "feature_a",
        "feature_b",
        "pearson_correlation",
        "spearman_correlation",
        "maximum_absolute_correlation",
    ],
)


if len(
    high_correlation_df
) > 0:

    high_correlation_df = (
        high_correlation_df.sort_values(
            "maximum_absolute_correlation",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# 10. KNOWN DETERMINISTIC RELATIONSHIPS
# =============================================================================

relationship_records = []


def add_relationship(
    relationship_name,
    expected_values,
    actual_values,
):

    difference = max_abs_difference(
        expected_values,
        actual_values,
    )


    relationship_records.append(
        {
            "relationship": (
                relationship_name
            ),
            "maximum_absolute_error": (
                difference
            ),
            "exact_within_tolerance": (
                bool(
                    np.isfinite(
                        difference
                    )
                    and
                    difference
                    <=
                    EXACT_TOLERANCE
                )
            ),
        }
    )


for class_name in [
    "green",
    "built",
    "open_bare",
    "water",
]:

    add_relationship(
        relationship_name=(
            f"{class_name}_slope_3yr "
            f"= {class_name}_change_2yr / 2"
        ),
        expected_values=(
            df[
                f"{class_name}_change_2yr"
            ]
            /
            2.0
        ),
        actual_values=(
            df[
                f"{class_name}_slope_3yr"
            ]
        ),
    )


add_relationship(
    relationship_name=(
        "nbr8_built_slope_3yr "
        "= nbr8_built_change_2yr / 2"
    ),
    expected_values=(
        df[
            "nbr8_built_change_2yr"
        ]
        /
        2.0
    ),
    actual_values=(
        df[
            "nbr8_built_slope_3yr"
        ]
    ),
)


add_relationship(
    relationship_name=(
        "terrestrial_nonbuilt_fraction_t "
        "= green_fraction_t + open_bare_fraction_t"
    ),
    expected_values=(
        df[
            "green_fraction_t"
        ]
        +
        df[
            "open_bare_fraction_t"
        ]
    ),
    actual_values=(
        df[
            "terrestrial_nonbuilt_fraction_t"
        ]
    ),
)


add_relationship(
    relationship_name=(
        "nonwater_fraction_t "
        "= 1 - water_fraction_t"
    ),
    expected_values=(
        1.0
        -
        df[
            "water_fraction_t"
        ]
    ),
    actual_values=(
        df[
            "nonwater_fraction_t"
        ]
    ),
)


add_relationship(
    relationship_name=(
        "built_minus_green_t "
        "= built_fraction_t - green_fraction_t"
    ),
    expected_values=(
        df[
            "built_fraction_t"
        ]
        -
        df[
            "green_fraction_t"
        ]
    ),
    actual_values=(
        df[
            "built_minus_green_t"
        ]
    ),
)


deterministic_relationship_df = pd.DataFrame(
    relationship_records
)


# =============================================================================
# 11. COMPOSITION-SUM AUDIT
# =============================================================================

current_fraction_sum = (
    df[
        "green_fraction_t"
    ]
    +
    df[
        "built_fraction_t"
    ]
    +
    df[
        "open_bare_fraction_t"
    ]
    +
    df[
        "water_fraction_t"
    ]
)


current_composition_max_error = float(
    np.max(
        np.abs(
            current_fraction_sum
            -
            1.0
        )
    )
)


composition_sum_valid = (
    current_composition_max_error
    <=
    1e-8
)


# =============================================================================
# 12. QA
# =============================================================================

missing_feature_values = int(
    feature_matrix.isna().sum().sum()
)


infinite_feature_values = int(
    np.isinf(
        feature_matrix.to_numpy(
            dtype=float
        )
    ).sum()
)


target_not_used_in_feature_matrix = (
    "urban_conversion_target"
    not in
    feature_columns
)


grid_identifiers_not_features = all(
    prohibited_column
    not in
    feature_columns
    for prohibited_column in [
        "grid_id",
        "grid_row",
        "grid_col",
    ]
)


overall_status = (
    "PASS"
    if all(
        [
            len(
                df
            )
            ==
            EXPECTED_ROWS,
            len(
                feature_columns
            )
            ==
            EXPECTED_FEATURES,
            positive_count
            ==
            EXPECTED_POSITIVES,
            missing_feature_values
            ==
            0,
            infinite_feature_values
            ==
            0,
            target_not_used_in_feature_matrix,
            grid_identifiers_not_features,
            composition_sum_valid,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 13. FINAL SUMMARY
# =============================================================================

final_summary_df = pd.DataFrame(
    [
        {
            "model_rows": (
                len(
                    df
                )
            ),
            "predictor_count": (
                len(
                    feature_columns
                )
            ),
            "positive_target_rows": (
                positive_count
            ),
            "constant_feature_count": (
                len(
                    constant_features
                )
            ),
            "near_constant_feature_count": (
                len(
                    near_constant_features
                )
            ),
            "exact_duplicate_pair_count": (
                len(
                    exact_duplicate_df
                )
            ),
            "high_correlation_pair_count_abs_ge_0_98": (
                len(
                    high_correlation_df
                )
            ),
            "known_deterministic_relationship_count": (
                len(
                    deterministic_relationship_df
                )
            ),
            "current_lulc_composition_max_sum_error": (
                current_composition_max_error
            ),
            "target_used_for_feature_selection": (
                False
            ),
            "features_removed_in_this_step": (
                False
            ),
            "osm_2026_used": (
                False
            ),
            "overall_step12d_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 14. SAVE REPORTS
# =============================================================================

feature_profile_df.to_csv(
    FEATURE_PROFILE_CSV,
    index=False,
)


exact_duplicate_df.to_csv(
    EXACT_DUPLICATE_CSV,
    index=False,
)


high_correlation_df.to_csv(
    HIGH_CORRELATION_CSV,
    index=False,
)


deterministic_relationship_df.to_csv(
    DETERMINISTIC_RELATIONSHIP_CSV,
    index=False,
)


final_summary_df.to_csv(
    FINAL_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 15. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("FEATURE PROFILE")
print("=" * 100)


print(
    f"Constant predictors                 : "
    f"{len(constant_features)}"
)

if len(
    constant_features
) > 0:

    for feature_name in constant_features:

        print(
            f" - {feature_name}"
        )


print(
    f"\nNear-constant predictors            : "
    f"{len(near_constant_features)}"
)

if len(
    near_constant_features
) > 0:

    for feature_name in near_constant_features:

        print(
            f" - {feature_name}"
        )


print("\n" + "=" * 100)
print("EXACT DUPLICATE PREDICTOR PAIRS")
print("=" * 100)


if len(
    exact_duplicate_df
) == 0:

    print(
        "No exact duplicate predictor pairs."
    )

else:

    print(
        exact_duplicate_df.to_string(
            index=False
        )
    )


print("\n" + "=" * 100)
print("HIGH CORRELATION PAIRS |r| >= 0.98")
print("=" * 100)


if len(
    high_correlation_df
) == 0:

    print(
        "No predictor pairs reached the threshold."
    )

else:

    print(
        high_correlation_df.to_string(
            index=False
        )
    )


print("\n" + "=" * 100)
print("KNOWN DETERMINISTIC RELATIONSHIPS")
print("=" * 100)


print(
    deterministic_relationship_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 12D QA")
print("=" * 100)


print(
    f"Rows                              : "
    f"{len(df):,}"
)

print(
    f"Predictors audited                : "
    f"{len(feature_columns)}"
)

print(
    f"Positive target rows              : "
    f"{positive_count:,}"
)

print(
    f"Missing predictor values          : "
    f"{missing_feature_values}"
)

print(
    f"Infinite predictor values         : "
    f"{infinite_feature_values}"
)

print(
    f"Target excluded from feature audit: "
    f"{target_not_used_in_feature_matrix}"
)

print(
    f"Grid IDs/coordinates excluded     : "
    f"{grid_identifiers_not_features}"
)

print(
    f"Current LULC fractions sum to 1   : "
    f"{composition_sum_valid}"
)

print(
    f"Max current composition error     : "
    f"{current_composition_max_error:.12f}"
)

print(
    f"Features removed                  : "
    f"False"
)

print(
    f"OSM 2026 used                     : "
    f"False"
)

print(
    f"Overall Step 12D status           : "
    f"{overall_status}"
)


print("\nGenerated reports:")

print(
    f" - {relative_path(FEATURE_PROFILE_CSV)}"
)

print(
    f" - {relative_path(EXACT_DUPLICATE_CSV)}"
)

print(
    f" - {relative_path(HIGH_CORRELATION_CSV)}"
)

print(
    f" - {relative_path(DETERMINISTIC_RELATIONSHIP_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. No predictor was deleted automatically."
)

print(
    "2. This audit does not use the target for feature selection."
)

print(
    "3. Exact and near-deterministic predictors can make SHAP explanations unstable."
)

print(
    "4. We will select one representative feature from redundant groups only after reviewing this output."
)

print(
    "5. 2026 OSM roads/buildings remain excluded from the historical pressure model."
)

print(
    "6. Current OSM road accessibility can be used later in present-day Development Suitability / planning analysis."
)

print("=" * 100)