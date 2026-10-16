"""
STEP 14D - CREATE RELATIVE URBANIZATION-PRESSURE CATEGORIES

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Convert the continuous Step 14C pressure output into transparent
   visualization / dashboard categories.
2. Preserve the locked threshold as the strongest operational category.
3. Use relative percentile bands below that threshold.
4. Keep cells outside the model inference domain clearly separate.
5. Produce area-weighted category summaries for Kaduwela.

Category rules:
- NOT_IN_INFERENCE_DOMAIN
    Cell was not scored by the urbanization-conversion model.

For eligible scored cells:
- LOW_RELATIVE
    pressure percentile < 50

- MODERATE_RELATIVE
    50 <= pressure percentile < 75

- ELEVATED_RELATIVE
    75 <= pressure percentile < 90

- HIGH_RELATIVE
    pressure percentile >= 90
    AND score < locked threshold

- THRESHOLD_FLAGGED
    score >= locked operating threshold

Important interpretation:
- Percentile classes are RELATIVE visualization categories.
- They are not calibrated probabilities.
- THRESHOLD_FLAGGED uses the locked development threshold.
- NOT_IN_INFERENCE_DOMAIN does NOT mean low pressure.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRESSURE_GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "inference"
    / "14c_2026_2028_urbanization_pressure_grid.gpkg"
)

PRESSURE_GRID_LAYER = (
    "urbanization_pressure_2026_2028"
)

PRESSURE_SCORES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "inference"
    / "14c_2026_2028_urbanization_pressure_scores.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "inference"
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

CATEGORIZED_GRID_FILE = (
    OUTPUT_DIR
    / "14d_2026_2028_pressure_categories.gpkg"
)

CATEGORIZED_GRID_LAYER = (
    "pressure_categories_2026_2028"
)

CATEGORIZED_SCORES_CSV = (
    OUTPUT_DIR
    / "14d_2026_2028_pressure_categories.csv"
)

CATEGORY_SUMMARY_CSV = (
    REPORT_DIR
    / "14d_pressure_category_summary.csv"
)

CATEGORY_RULES_CSV = (
    REPORT_DIR
    / "14d_pressure_category_rules.csv"
)

QA_CSV = (
    REPORT_DIR
    / "14d_pressure_category_qa.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "14d_pressure_category_analysis_summary.csv"
)


# =============================================================================
# 3. EXPECTED VALUES
# =============================================================================

EXPECTED_GRID_CELLS = 9240

EXPECTED_SCORED_CELLS = 4620

EXPECTED_FLAGGED_CELLS = 255

EXPECTED_THRESHOLD = 0.715

EXPECTED_ORIGIN_YEAR = 2025

EXPECTED_PRESSURE_PERIOD = "2026-2028"


# =============================================================================
# 4. CATEGORY CONSTANTS
# =============================================================================

P50 = 50.0

P75 = 75.0

P90 = 90.0


CATEGORY_ORDER = {
    "LOW_RELATIVE": 1,
    "MODERATE_RELATIVE": 2,
    "ELEVATED_RELATIVE": 3,
    "HIGH_RELATIVE": 4,
    "THRESHOLD_FLAGGED": 5,
    "NOT_IN_INFERENCE_DOMAIN": 0,
}


# =============================================================================
# 5. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """Return a project-relative path where possible."""

    try:

        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:

        return str(path)


def assign_pressure_category(row):
    """
    Assign one mutually exclusive pressure category.

    The locked threshold has priority over percentile categories.
    """

    if not bool(
        row[
            "final_inference_ready"
        ]
    ):

        return (
            "NOT_IN_INFERENCE_DOMAIN"
        )


    score = float(
        row[
            "urbanization_pressure_score"
        ]
    )


    percentile = float(
        row[
            "pressure_percentile"
        ]
    )


    threshold = float(
        row[
            "operating_threshold"
        ]
    )


    # -------------------------------------------------------------------------
    # Operational threshold has first priority.
    # -------------------------------------------------------------------------

    if score >= threshold:

        return (
            "THRESHOLD_FLAGGED"
        )


    # -------------------------------------------------------------------------
    # Relative visualization bands below threshold.
    # -------------------------------------------------------------------------

    if percentile >= P90:

        return (
            "HIGH_RELATIVE"
        )


    if percentile >= P75:

        return (
            "ELEVATED_RELATIVE"
        )


    if percentile >= P50:

        return (
            "MODERATE_RELATIVE"
        )


    return (
        "LOW_RELATIVE"
    )


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 14D - CREATE RELATIVE URBANIZATION-PRESSURE CATEGORIES")
print("=" * 100)


for required_file in [
    PRESSURE_GRID_FILE,
    PRESSURE_SCORES_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 7. LOAD STEP 14C GIS GRID
# =============================================================================

pressure_gdf = gpd.read_file(
    PRESSURE_GRID_FILE,
    layer=PRESSURE_GRID_LAYER,
)


if len(
    pressure_gdf
) != EXPECTED_GRID_CELLS:

    raise RuntimeError(
        f"Expected {EXPECTED_GRID_CELLS:,} grid cells, "
        f"found {len(pressure_gdf):,}."
    )


if pressure_gdf[
    "grid_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate grid IDs detected."
    )


required_columns = [
    "grid_id",
    "kmc_overlap_area_m2",
    "kmc_overlap_fraction",
    "final_inference_ready",
    "urbanization_pressure_score",
    "pressure_percentile",
    "pressure_rank",
    "operating_threshold",
    "threshold_flag",
    "map_status",
]


missing_columns = [
    column
    for column in required_columns
    if column not in pressure_gdf.columns
]


if missing_columns:

    raise RuntimeError(
        "Required Step 14C columns are missing:\n"
        +
        "\n".join(
            missing_columns
        )
    )


# =============================================================================
# 8. VERIFY STEP 14C STATE
# =============================================================================

scored_mask = (
    pressure_gdf[
        "urbanization_pressure_score"
    ]
    .notna()
)


scored_count = int(
    scored_mask.sum()
)


if scored_count != EXPECTED_SCORED_CELLS:

    raise RuntimeError(
        f"Expected {EXPECTED_SCORED_CELLS:,} scored cells, "
        f"found {scored_count:,}."
    )


flagged_count_before = int(
    pressure_gdf[
        "threshold_flag"
    ]
    .fillna(
        0
    )
    .astype(
        int
    )
    .sum()
)


if flagged_count_before != EXPECTED_FLAGGED_CELLS:

    raise RuntimeError(
        f"Expected {EXPECTED_FLAGGED_CELLS} Step 14C flagged cells, "
        f"found {flagged_count_before}."
    )


scored_threshold_values = (
    pressure_gdf.loc[
        scored_mask,
        "operating_threshold",
    ]
    .dropna()
    .unique()
)


if len(
    scored_threshold_values
) != 1:

    raise RuntimeError(
        "Expected exactly one locked operating threshold."
    )


locked_threshold = float(
    scored_threshold_values[
        0
    ]
)


if not np.isclose(
    locked_threshold,
    EXPECTED_THRESHOLD,
    atol=0.0,
    rtol=0.0,
):

    raise RuntimeError(
        f"Expected locked threshold {EXPECTED_THRESHOLD}, "
        f"found {locked_threshold}."
    )


print(
    f"\nPermanent grid cells     : "
    f"{len(pressure_gdf):,}"
)

print(
    f"Scored eligible cells    : "
    f"{scored_count:,}"
)

print(
    f"Locked threshold         : "
    f"{locked_threshold:.3f}"
)

print(
    f"Threshold-flagged cells  : "
    f"{flagged_count_before:,}"
)


# =============================================================================
# 9. CREATE CATEGORIES
# =============================================================================

pressure_gdf[
    "pressure_category"
] = (
    pressure_gdf.apply(
        assign_pressure_category,
        axis=1,
    )
)


pressure_gdf[
    "pressure_category_order"
] = (
    pressure_gdf[
        "pressure_category"
    ]
    .map(
        CATEGORY_ORDER
    )
    .astype(
        int
    )
)


pressure_gdf[
    "pressure_category_basis"
] = np.where(
    pressure_gdf[
        "pressure_category"
    ]
    ==
    "THRESHOLD_FLAGGED",
    "LOCKED_MODEL_THRESHOLD",
    np.where(
        pressure_gdf[
            "pressure_category"
        ]
        ==
        "NOT_IN_INFERENCE_DOMAIN",
        "MODEL_DOMAIN_EXCLUSION",
        "RELATIVE_PERCENTILE_BAND",
    ),
)


pressure_gdf[
    "category_is_probability_class"
] = (
    False
)


pressure_gdf[
    "origin_year"
] = (
    EXPECTED_ORIGIN_YEAR
)


pressure_gdf[
    "pressure_period"
] = (
    EXPECTED_PRESSURE_PERIOD
)


# =============================================================================
# 10. CREATE AREA-WEIGHTED CATEGORY SUMMARY
# =============================================================================

summary_records = []


for category_name in sorted(
    CATEGORY_ORDER,
    key=lambda category:
    CATEGORY_ORDER[
        category
    ],
):

    category_df = (
        pressure_gdf[
            pressure_gdf[
                "pressure_category"
            ]
            ==
            category_name
        ]
    )


    cell_count = len(
        category_df
    )


    overlap_area_km2 = float(
        category_df[
            "kmc_overlap_area_m2"
        ]
        .sum()
        /
        1_000_000.0
    )


    scored_category = (
        category_name
        !=
        "NOT_IN_INFERENCE_DOMAIN"
    )


    if scored_category:

        mean_score = float(
            category_df[
                "urbanization_pressure_score"
            ]
            .mean()
        )

        median_score = float(
            category_df[
                "urbanization_pressure_score"
            ]
            .median()
        )

        minimum_score = float(
            category_df[
                "urbanization_pressure_score"
            ]
            .min()
        )

        maximum_score = float(
            category_df[
                "urbanization_pressure_score"
            ]
            .max()
        )

        mean_percentile = float(
            category_df[
                "pressure_percentile"
            ]
            .mean()
        )

    else:

        mean_score = np.nan
        median_score = np.nan
        minimum_score = np.nan
        maximum_score = np.nan
        mean_percentile = np.nan


    summary_records.append(
        {
            "pressure_category": (
                category_name
            ),
            "category_order": (
                CATEGORY_ORDER[
                    category_name
                ]
            ),
            "grid_cells": (
                cell_count
            ),
            "percent_of_all_grid_cells": (
                cell_count
                /
                EXPECTED_GRID_CELLS
                *
                100.0
            ),
            "kmc_overlap_area_km2": (
                overlap_area_km2
            ),
            "mean_pressure_score": (
                mean_score
            ),
            "median_pressure_score": (
                median_score
            ),
            "minimum_pressure_score": (
                minimum_score
            ),
            "maximum_pressure_score": (
                maximum_score
            ),
            "mean_pressure_percentile": (
                mean_percentile
            ),
        }
    )


category_summary_df = pd.DataFrame(
    summary_records
)


eligible_area_km2 = float(
    pressure_gdf.loc[
        pressure_gdf[
            "final_inference_ready"
        ].astype(
            bool
        ),
        "kmc_overlap_area_m2",
    ]
    .sum()
    /
    1_000_000.0
)


category_summary_df[
    "percent_of_eligible_kmc_overlap_area"
] = np.where(
    category_summary_df[
        "pressure_category"
    ]
    !=
    "NOT_IN_INFERENCE_DOMAIN",
    (
        category_summary_df[
            "kmc_overlap_area_km2"
        ]
        /
        eligible_area_km2
        *
        100.0
    ),
    np.nan,
)


# =============================================================================
# 11. CATEGORY RULES REPORT
# =============================================================================

category_rules_df = pd.DataFrame(
    [
        {
            "pressure_category": (
                "LOW_RELATIVE"
            ),
            "rule": (
                "Eligible; percentile < 50; score below locked threshold"
            ),
            "basis": (
                "Relative percentile visualization band"
            ),
            "probability_interpretation": (
                False
            ),
        },
        {
            "pressure_category": (
                "MODERATE_RELATIVE"
            ),
            "rule": (
                "Eligible; 50 <= percentile < 75; "
                "score below locked threshold"
            ),
            "basis": (
                "Relative percentile visualization band"
            ),
            "probability_interpretation": (
                False
            ),
        },
        {
            "pressure_category": (
                "ELEVATED_RELATIVE"
            ),
            "rule": (
                "Eligible; 75 <= percentile < 90; "
                "score below locked threshold"
            ),
            "basis": (
                "Relative percentile visualization band"
            ),
            "probability_interpretation": (
                False
            ),
        },
        {
            "pressure_category": (
                "HIGH_RELATIVE"
            ),
            "rule": (
                "Eligible; percentile >= 90; "
                "score below locked threshold"
            ),
            "basis": (
                "Relative percentile visualization band"
            ),
            "probability_interpretation": (
                False
            ),
        },
        {
            "pressure_category": (
                "THRESHOLD_FLAGGED"
            ),
            "rule": (
                f"Eligible; score >= locked threshold "
                f"{locked_threshold:.3f}"
            ),
            "basis": (
                "Locked development operating threshold"
            ),
            "probability_interpretation": (
                False
            ),
        },
        {
            "pressure_category": (
                "NOT_IN_INFERENCE_DOMAIN"
            ),
            "rule": (
                "Cell did not satisfy production-model inference eligibility"
            ),
            "basis": (
                "Model-domain exclusion"
            ),
            "probability_interpretation": (
                False
            ),
        },
    ]
)


# =============================================================================
# 12. CREATE SCORED-CELL CSV
# =============================================================================

categorized_scores_df = (
    pressure_gdf.loc[
        pressure_gdf[
            "final_inference_ready"
        ].astype(
            bool
        ),
        [
            "grid_id",
            "grid_row",
            "grid_col",
            "origin_year",
            "pressure_period",
            "urbanization_pressure_score",
            "pressure_percentile",
            "pressure_rank",
            "operating_threshold",
            "threshold_flag",
            "pressure_category",
            "pressure_category_order",
            "pressure_category_basis",
            "kmc_overlap_fraction",
            "kmc_overlap_area_m2",
        ],
    ]
    .copy()
)


categorized_scores_df = (
    categorized_scores_df.sort_values(
        [
            "pressure_category_order",
            "urbanization_pressure_score",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 13. QA
# =============================================================================

valid_categories = set(
    CATEGORY_ORDER.keys()
)


observed_categories = set(
    pressure_gdf[
        "pressure_category"
    ].unique()
)


category_values_valid = bool(
    observed_categories.issubset(
        valid_categories
    )
)


all_expected_categories_present = bool(
    valid_categories.issubset(
        observed_categories
    )
)


eligible_count_after = int(
    (
        pressure_gdf[
            "pressure_category"
        ]
        !=
        "NOT_IN_INFERENCE_DOMAIN"
    ).sum()
)


eligible_count_valid = (
    eligible_count_after
    ==
    EXPECTED_SCORED_CELLS
)


flagged_category_count = int(
    (
        pressure_gdf[
            "pressure_category"
        ]
        ==
        "THRESHOLD_FLAGGED"
    ).sum()
)


flagged_count_valid = (
    flagged_category_count
    ==
    EXPECTED_FLAGGED_CELLS
)


threshold_category_exact = bool(
    (
        (
            pressure_gdf.loc[
                scored_mask,
                "urbanization_pressure_score",
            ]
            >=
            locked_threshold
        )
        ==
        (
            pressure_gdf.loc[
                scored_mask,
                "pressure_category",
            ]
            ==
            "THRESHOLD_FLAGGED"
        )
    ).all()
)


ineligible_category_valid = bool(
    (
        pressure_gdf.loc[
            ~pressure_gdf[
                "final_inference_ready"
            ].astype(
                bool
            ),
            "pressure_category",
        ]
        ==
        "NOT_IN_INFERENCE_DOMAIN"
    ).all()
)


ineligible_still_has_no_score = bool(
    pressure_gdf.loc[
        ~pressure_gdf[
            "final_inference_ready"
        ].astype(
            bool
        ),
        "urbanization_pressure_score",
    ]
    .isna()
    .all()
)


scored_cells_have_category = bool(
    pressure_gdf.loc[
        pressure_gdf[
            "final_inference_ready"
        ].astype(
            bool
        ),
        "pressure_category",
    ]
    .notna()
    .all()
)


category_count_total_valid = (
    int(
        category_summary_df[
            "grid_cells"
        ].sum()
    )
    ==
    EXPECTED_GRID_CELLS
)


area_reconstruction_km2 = float(
    category_summary_df[
        "kmc_overlap_area_km2"
    ].sum()
)


original_kmc_overlap_area_km2 = float(
    pressure_gdf[
        "kmc_overlap_area_m2"
    ].sum()
    /
    1_000_000.0
)


area_difference_m2 = float(
    abs(
        area_reconstruction_km2
        -
        original_kmc_overlap_area_km2
    )
    *
    1_000_000.0
)


area_reconstruction_valid = (
    area_difference_m2
    <=
    0.01
)


threshold_changed = False

model_retrained = False

score_recalculated = False

probability_classes_claimed = False


overall_status = (
    "PASS"
    if all(
        [
            category_values_valid,
            all_expected_categories_present,
            eligible_count_valid,
            flagged_count_valid,
            threshold_category_exact,
            ineligible_category_valid,
            ineligible_still_has_no_score,
            scored_cells_have_category,
            category_count_total_valid,
            area_reconstruction_valid,
            not threshold_changed,
            not model_retrained,
            not score_recalculated,
            not probability_classes_claimed,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 14. QA REPORT
# =============================================================================

qa_df = pd.DataFrame(
    [
        {
            "permanent_grid_cells": (
                len(
                    pressure_gdf
                )
            ),
            "scored_eligible_cells": (
                eligible_count_after
            ),
            "threshold_flagged_cells": (
                flagged_category_count
            ),
            "category_values_valid": (
                category_values_valid
            ),
            "all_expected_categories_present": (
                all_expected_categories_present
            ),
            "eligible_count_valid": (
                eligible_count_valid
            ),
            "flagged_count_valid": (
                flagged_count_valid
            ),
            "threshold_category_matches_locked_threshold": (
                threshold_category_exact
            ),
            "ineligible_category_valid": (
                ineligible_category_valid
            ),
            "ineligible_cells_still_have_no_score": (
                ineligible_still_has_no_score
            ),
            "all_scored_cells_have_category": (
                scored_cells_have_category
            ),
            "category_counts_sum_to_full_grid": (
                category_count_total_valid
            ),
            "kmc_area_reconstruction_difference_m2": (
                area_difference_m2
            ),
            "kmc_area_reconstruction_valid": (
                area_reconstruction_valid
            ),
            "threshold_changed": (
                threshold_changed
            ),
            "model_retrained": (
                model_retrained
            ),
            "pressure_score_recalculated": (
                score_recalculated
            ),
            "categories_claimed_as_probability_classes": (
                probability_classes_claimed
            ),
            "overall_step14d_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 15. FINAL SUMMARY
# =============================================================================

final_summary_df = pd.DataFrame(
    [
        {
            "stage": (
                "Urbanization-pressure visualization categorization"
            ),
            "origin_year": (
                EXPECTED_ORIGIN_YEAR
            ),
            "pressure_period": (
                EXPECTED_PRESSURE_PERIOD
            ),
            "scored_cells": (
                EXPECTED_SCORED_CELLS
            ),
            "locked_threshold": (
                locked_threshold
            ),
            "threshold_flagged_cells": (
                flagged_category_count
            ),
            "relative_percentile_breaks": (
                "50,75,90"
            ),
            "category_scheme": (
                "LOW_RELATIVE;MODERATE_RELATIVE;"
                "ELEVATED_RELATIVE;HIGH_RELATIVE;"
                "THRESHOLD_FLAGGED"
            ),
            "not_in_domain_kept_separate": (
                True
            ),
            "categories_are_calibrated_probability_bands": (
                False
            ),
            "model_or_threshold_changed": (
                False
            ),
            "overall_step14d_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 16. SAVE OUTPUTS
# =============================================================================

if CATEGORIZED_GRID_FILE.exists():

    CATEGORIZED_GRID_FILE.unlink()


pressure_gdf.to_file(
    CATEGORIZED_GRID_FILE,
    layer=CATEGORIZED_GRID_LAYER,
    driver="GPKG",
)


categorized_scores_df.to_csv(
    CATEGORIZED_SCORES_CSV,
    index=False,
)


category_summary_df.to_csv(
    CATEGORY_SUMMARY_CSV,
    index=False,
)


category_rules_df.to_csv(
    CATEGORY_RULES_CSV,
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
# 17. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("PRESSURE CATEGORY SUMMARY")
print("=" * 100)


print(
    category_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("CATEGORY RULES")
print("=" * 100)


print(
    category_rules_df[
        [
            "pressure_category",
            "rule",
            "basis",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 14D QA")
print("=" * 100)


print(
    f"Category values valid                 : "
    f"{category_values_valid}"
)

print(
    f"All expected categories present       : "
    f"{all_expected_categories_present}"
)

print(
    f"Eligible count remains 4,620          : "
    f"{eligible_count_valid}"
)

print(
    f"Threshold-flagged count remains 255   : "
    f"{flagged_count_valid}"
)

print(
    f"Flagged category matches threshold    : "
    f"{threshold_category_exact}"
)

print(
    f"Ineligible cells kept separate        : "
    f"{ineligible_category_valid}"
)

print(
    f"Ineligible cells still have no score  : "
    f"{ineligible_still_has_no_score}"
)

print(
    f"Every scored cell has a category      : "
    f"{scored_cells_have_category}"
)

print(
    f"Category counts reconstruct grid      : "
    f"{category_count_total_valid}"
)

print(
    f"KMC area reconstruction valid         : "
    f"{area_reconstruction_valid}"
)

print(
    f"KMC area difference                  : "
    f"{area_difference_m2:.6f} m²"
)

print(
    f"Threshold changed                    : "
    f"{threshold_changed}"
)

print(
    f"Model retrained                      : "
    f"{model_retrained}"
)

print(
    f"Pressure scores recalculated         : "
    f"{score_recalculated}"
)

print(
    f"Probability classes claimed          : "
    f"{probability_classes_claimed}"
)

print(
    f"Overall Step 14D status              : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(CATEGORIZED_GRID_FILE)}"
)

print(
    f"   Layer: {CATEGORIZED_GRID_LAYER}"
)

print(
    f" - {relative_path(CATEGORIZED_SCORES_CSV)}"
)

print(
    f" - {relative_path(CATEGORY_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(CATEGORY_RULES_CSV)}"
)

print(
    f" - {relative_path(QA_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. THRESHOLD_FLAGGED is based on the locked model threshold, not a percentile guess."
)

print(
    "2. LOW/MODERATE/ELEVATED/HIGH are relative visualization bands among eligible Kaduwela cells."
)

print(
    "3. These categories are not calibrated conversion probabilities."
)

print(
    "4. NOT_IN_INFERENCE_DOMAIN must not be interpreted as low urbanization pressure."
)

print(
    "5. No model, score, threshold or final performance metric was changed."
)

print(
    "6. After this passes, the urbanization-pressure prediction stage can be frozen and we can move to XAI/SHAP explanations."
)

print("=" * 100)