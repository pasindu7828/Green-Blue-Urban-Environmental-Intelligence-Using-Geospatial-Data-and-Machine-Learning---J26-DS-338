"""
STEP 05F - MANUAL LULC REFERENCE QUALITY-AUDIT EVALUATION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Read the manually reviewed 40-point HIGH-confidence quality-audit layer.
2. Validate that all 40 points were reviewed.
3. Count VERIFIED, CORRECTED and REJECTED results.
4. Calculate per-class and overall audit agreement.
5. Identify corrected and rejected audit samples.
6. Create a training-action table for the audited points.
7. Keep this audit separate from final model accuracy validation.

Interpretation:
VERIFIED
    Manual inspection agrees with the original reference class.

CORRECTED
    Manual inspection indicates a different clear LULC class.

REJECTED
    The 30 m pixel is mixed, unclear or unsuitable as a clean reference sample.

IMPORTANT:
- This is a reference-quality audit.
- It is NOT final model accuracy validation.
- No classifier is trained here.
- The 399-point training candidate pool is NOT modified in this step.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lulc"
    / "reference_samples"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. INPUT
# =============================================================================

AUDIT_REVIEW_GPKG = (
    REFERENCE_DIR
    / "lulc_high_confidence_quality_audit_review_40_2021.gpkg"
)


# =============================================================================
# 3. OUTPUTS
# =============================================================================

AUDIT_RESULTS_CSV = (
    REFERENCE_DIR
    / "lulc_high_confidence_quality_audit_results_40_2021.csv"
)

TRAINING_ACTIONS_CSV = (
    REFERENCE_DIR
    / "lulc_manual_audit_training_actions_2021.csv"
)

CLASS_SUMMARY_CSV = (
    REPORT_DIR
    / "05f_manual_audit_class_summary.csv"
)

OVERALL_SUMMARY_CSV = (
    REPORT_DIR
    / "05f_manual_audit_overall_summary.csv"
)

CORRECTIONS_CSV = (
    REPORT_DIR
    / "05f_manual_audit_corrections.csv"
)

REJECTIONS_CSV = (
    REPORT_DIR
    / "05f_manual_audit_rejections.csv"
)


# =============================================================================
# 4. CONSTANTS
# =============================================================================

ANALYSIS_CRS = "EPSG:32644"

EXPECTED_TOTAL = 40

EXPECTED_PER_CLASS = 10


CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}


ALLOWED_REVIEW_STATUS = {
    "VERIFIED",
    "CORRECTED",
    "REJECTED",
}


# =============================================================================
# 5. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """
    Return a project-relative path when possible.
    """

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def safe_percent(
    numerator: float,
    denominator: float,
) -> float:
    """
    Calculate percentage safely.
    """

    if denominator == 0:
        return np.nan

    return float(
        numerator
        / denominator
        * 100
    )


def clean_text_series(series: pd.Series) -> pd.Series:
    """
    Convert text-like values to clean strings.
    """

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 05F - MANUAL LULC REFERENCE QUALITY-AUDIT EVALUATION")
print("=" * 100)


if not AUDIT_REVIEW_GPKG.exists():

    raise FileNotFoundError(
        f"Editable manual-audit GeoPackage not found:\n"
        f"{AUDIT_REVIEW_GPKG}"
    )


audit_gdf = gpd.read_file(
    AUDIT_REVIEW_GPKG
)


print(
    f"\nInput audit layer : "
    f"{relative_path(AUDIT_REVIEW_GPKG)}"
)

print(
    f"Input features    : "
    f"{len(audit_gdf)}"
)

print(
    f"CRS               : "
    f"{audit_gdf.crs}"
)


# =============================================================================
# 7. INPUT VALIDATION
# =============================================================================

if str(
    audit_gdf.crs
) != ANALYSIS_CRS:

    raise ValueError(
        f"Unexpected CRS: {audit_gdf.crs}. "
        f"Expected {ANALYSIS_CRS}."
    )


if len(
    audit_gdf
) != EXPECTED_TOTAL:

    raise ValueError(
        f"Expected {EXPECTED_TOTAL} audit points, "
        f"but found {len(audit_gdf)}."
    )


required_columns = {
    "sample_id",
    "class_id",
    "class_name",
    "review_status",
    "manual_class_id",
    "manual_class_name",
    "review_notes",
}


missing_columns = (
    required_columns
    -
    set(
        audit_gdf.columns
    )
)


if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )


if audit_gdf[
    "sample_id"
].duplicated().any():

    raise ValueError(
        "Duplicate sample IDs found "
        "in the audit layer."
    )


audit_gdf[
    "class_id"
] = pd.to_numeric(
    audit_gdf[
        "class_id"
    ],
    errors="raise",
).astype(int)


audit_gdf[
    "review_status"
] = (
    clean_text_series(
        audit_gdf[
            "review_status"
        ]
    )
    .str.upper()
)


audit_gdf[
    "manual_class_name"
] = clean_text_series(
    audit_gdf[
        "manual_class_name"
    ]
)


audit_gdf[
    "review_notes"
] = clean_text_series(
    audit_gdf[
        "review_notes"
    ]
)


# =============================================================================
# 8. CHECK THAT ALL 40 POINTS WERE REVIEWED
# =============================================================================

print("\n" + "=" * 100)
print("MANUAL REVIEW COMPLETENESS CHECK")
print("=" * 100)


status_counts = (
    audit_gdf[
        "review_status"
    ]
    .value_counts(
        dropna=False
    )
)


print(
    status_counts.to_string()
)


invalid_status_mask = (
    ~audit_gdf[
        "review_status"
    ].isin(
        ALLOWED_REVIEW_STATUS
    )
)


invalid_status_count = int(
    invalid_status_mask.sum()
)


if invalid_status_count > 0:

    print(
        "\nPoints with invalid or unfinished review status:"
    )

    print(
        audit_gdf.loc[
            invalid_status_mask,
            [
                "sample_id",
                "class_id",
                "class_name",
                "review_status",
            ],
        ].to_string(
            index=False
        )
    )

    raise ValueError(
        f"{invalid_status_count} audit points are still "
        f"PENDING, blank, or contain an invalid review_status. "
        f"Complete all 40 manual reviews before Step 05F."
    )


# =============================================================================
# 9. VALIDATE MANUAL LABELS
# =============================================================================

manual_class_numeric = pd.to_numeric(
    audit_gdf[
        "manual_class_id"
    ],
    errors="coerce",
)


# VERIFIED must have a manual class and it should match the original class.
verified_mask = (
    audit_gdf[
        "review_status"
    ]
    == "VERIFIED"
)


verified_missing_manual = (
    verified_mask
    &
    manual_class_numeric.isna()
)


if verified_missing_manual.any():

    bad_ids = (
        audit_gdf.loc[
            verified_missing_manual,
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    raise ValueError(
        "VERIFIED points must have manual_class_id. "
        f"Missing for: {bad_ids}"
    )


verified_mismatch = (
    verified_mask
    &
    (
        manual_class_numeric
        != audit_gdf[
            "class_id"
        ]
    )
)


if verified_mismatch.any():

    bad_ids = (
        audit_gdf.loc[
            verified_mismatch,
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    raise ValueError(
        "VERIFIED points must keep the same class_id. "
        f"Mismatch for: {bad_ids}"
    )


# CORRECTED must have another valid class.
corrected_mask = (
    audit_gdf[
        "review_status"
    ]
    == "CORRECTED"
)


corrected_missing_manual = (
    corrected_mask
    &
    manual_class_numeric.isna()
)


if corrected_missing_manual.any():

    bad_ids = (
        audit_gdf.loc[
            corrected_missing_manual,
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    raise ValueError(
        "CORRECTED points must have manual_class_id. "
        f"Missing for: {bad_ids}"
    )


corrected_invalid_class = (
    corrected_mask
    &
    (
        ~manual_class_numeric.isin(
            CLASS_DEFINITIONS.keys()
        )
    )
)


if corrected_invalid_class.any():

    bad_ids = (
        audit_gdf.loc[
            corrected_invalid_class,
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    raise ValueError(
        "CORRECTED points contain an invalid manual class. "
        f"Check: {bad_ids}"
    )


corrected_same_as_original = (
    corrected_mask
    &
    (
        manual_class_numeric
        == audit_gdf[
            "class_id"
        ]
    )
)


if corrected_same_as_original.any():

    bad_ids = (
        audit_gdf.loc[
            corrected_same_as_original,
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    raise ValueError(
        "CORRECTED points must have a different class "
        "from the original reference class. "
        f"Check: {bad_ids}"
    )


# REJECTED points can keep manual class fields empty.
rejected_mask = (
    audit_gdf[
        "review_status"
    ]
    == "REJECTED"
)


# =============================================================================
# 10. CHECK ORIGINAL CLASS BALANCE
# =============================================================================

original_class_counts = (
    audit_gdf[
        "class_id"
    ]
    .value_counts()
    .sort_index()
)


all_original_class_counts_correct = True


for class_id in (
    CLASS_DEFINITIONS.keys()
):

    count = int(
        original_class_counts.get(
            class_id,
            0
        )
    )

    if count != EXPECTED_PER_CLASS:

        all_original_class_counts_correct = False


if not all_original_class_counts_correct:

    raise ValueError(
        "The audit no longer contains exactly "
        "10 original samples per class."
    )


# =============================================================================
# 11. CREATE TRAINING ACTION
# =============================================================================

training_actions = []


for _, row in (
    audit_gdf.iterrows()
):

    status = row[
        "review_status"
    ]


    if status == "VERIFIED":

        action = (
            "KEEP_ORIGINAL_LABEL"
        )

        final_manual_class_id = int(
            row[
                "class_id"
            ]
        )

        final_manual_class_name = (
            CLASS_DEFINITIONS[
                final_manual_class_id
            ]
        )


    elif status == "CORRECTED":

        action = (
            "RELABEL_MANUALLY"
        )

        final_manual_class_id = int(
            pd.to_numeric(
                row[
                    "manual_class_id"
                ],
                errors="raise",
            )
        )

        final_manual_class_name = (
            CLASS_DEFINITIONS[
                final_manual_class_id
            ]
        )


    elif status == "REJECTED":

        action = (
            "EXCLUDE_FROM_TRAINING"
        )

        final_manual_class_id = pd.NA

        final_manual_class_name = ""


    else:

        raise ValueError(
            f"Unexpected review status: "
            f"{status}"
        )


    training_actions.append(
        {
            "sample_id": (
                row[
                    "sample_id"
                ]
            ),
            "original_class_id": (
                int(
                    row[
                        "class_id"
                    ]
                )
            ),
            "original_class_name": (
                row[
                    "class_name"
                ]
            ),
            "review_status": (
                status
            ),
            "manual_class_id": (
                final_manual_class_id
            ),
            "manual_class_name": (
                final_manual_class_name
            ),
            "training_action": (
                action
            ),
            "review_notes": (
                row[
                    "review_notes"
                ]
            ),
        }
    )


training_actions_df = pd.DataFrame(
    training_actions
)


training_actions_df[
    "manual_class_id"
] = (
    training_actions_df[
        "manual_class_id"
    ]
    .astype(
        "Int64"
    )
)


# =============================================================================
# 12. OVERALL AUDIT COUNTS
# =============================================================================

verified_count = int(
    verified_mask.sum()
)

corrected_count = int(
    corrected_mask.sum()
)

rejected_count = int(
    rejected_mask.sum()
)


accepted_count = (
    verified_count
    +
    corrected_count
)


original_label_agreement_percent = (
    safe_percent(
        verified_count,
        EXPECTED_TOTAL,
    )
)


correction_percent = (
    safe_percent(
        corrected_count,
        EXPECTED_TOTAL,
    )
)


rejection_percent = (
    safe_percent(
        rejected_count,
        EXPECTED_TOTAL,
    )
)


manual_usable_percent = (
    safe_percent(
        accepted_count,
        EXPECTED_TOTAL,
    )
)


# Agreement among only points where a clear manual class
# could be assigned.
if accepted_count > 0:

    classifiable_agreement_percent = (
        safe_percent(
            verified_count,
            accepted_count,
        )
    )

else:

    classifiable_agreement_percent = (
        np.nan
    )


# =============================================================================
# 13. PER-CLASS AUDIT SUMMARY
# =============================================================================

class_summary_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    class_df = (
        audit_gdf[
            audit_gdf[
                "class_id"
            ]
            == class_id
        ]
        .copy()
    )


    total = len(
        class_df
    )


    class_verified = int(
        (
            class_df[
                "review_status"
            ]
            == "VERIFIED"
        ).sum()
    )


    class_corrected = int(
        (
            class_df[
                "review_status"
            ]
            == "CORRECTED"
        ).sum()
    )


    class_rejected = int(
        (
            class_df[
                "review_status"
            ]
            == "REJECTED"
        ).sum()
    )


    class_accepted = (
        class_verified
        +
        class_corrected
    )


    class_summary_records.append(
        {
            "class_id": (
                class_id
            ),
            "class_name": (
                class_name
            ),
            "audit_points": (
                total
            ),
            "verified": (
                class_verified
            ),
            "corrected": (
                class_corrected
            ),
            "rejected": (
                class_rejected
            ),
            "original_label_agreement_percent": (
                safe_percent(
                    class_verified,
                    total,
                )
            ),
            "correction_percent": (
                safe_percent(
                    class_corrected,
                    total,
                )
            ),
            "rejection_percent": (
                safe_percent(
                    class_rejected,
                    total,
                )
            ),
            "manual_usable_percent": (
                safe_percent(
                    class_accepted,
                    total,
                )
            ),
            "classifiable_agreement_percent": (
                safe_percent(
                    class_verified,
                    class_accepted,
                )
                if class_accepted > 0
                else np.nan
            ),
        }
    )


class_summary_df = pd.DataFrame(
    class_summary_records
)


# =============================================================================
# 14. CORRECTION AND REJECTION TABLES
# =============================================================================

corrections_df = (
    training_actions_df[
        training_actions_df[
            "review_status"
        ]
        == "CORRECTED"
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


rejections_df = (
    training_actions_df[
        training_actions_df[
            "review_status"
        ]
        == "REJECTED"
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 15. OVERALL SUMMARY
# =============================================================================

overall_status = (
    "PASS"
    if (
        len(
            audit_gdf
        )
        == EXPECTED_TOTAL
        and
        invalid_status_count
        == 0
        and
        all_original_class_counts_correct
    )
    else
    "REVIEW_REQUIRED"
)


overall_summary_df = pd.DataFrame(
    [
        {
            "audit_type": (
                "REFERENCE_QUALITY_AUDIT"
            ),
            "total_audit_points": (
                EXPECTED_TOTAL
            ),
            "verified": (
                verified_count
            ),
            "corrected": (
                corrected_count
            ),
            "rejected": (
                rejected_count
            ),
            "manual_usable_points": (
                accepted_count
            ),
            "original_label_agreement_percent": (
                original_label_agreement_percent
            ),
            "classifiable_agreement_percent": (
                classifiable_agreement_percent
            ),
            "correction_percent": (
                correction_percent
            ),
            "rejection_percent": (
                rejection_percent
            ),
            "manual_usable_percent": (
                manual_usable_percent
            ),
            "all_40_reviews_completed": (
                True
            ),
            "exactly_10_original_points_per_class": (
                all_original_class_counts_correct
            ),
            "final_model_accuracy_validation": (
                False
            ),
            "overall_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 16. SAVE AUDIT RESULT COPY
# =============================================================================

audit_results_df = (
    audit_gdf
    .drop(
        columns="geometry"
    )
    .copy()
)


audit_results_df.to_csv(
    AUDIT_RESULTS_CSV,
    index=False,
)


training_actions_df.to_csv(
    TRAINING_ACTIONS_CSV,
    index=False,
)


class_summary_df.to_csv(
    CLASS_SUMMARY_CSV,
    index=False,
)


overall_summary_df.to_csv(
    OVERALL_SUMMARY_CSV,
    index=False,
)


corrections_df.to_csv(
    CORRECTIONS_CSV,
    index=False,
)


rejections_df.to_csv(
    REJECTIONS_CSV,
    index=False,
)


# =============================================================================
# 17. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("MANUAL AUDIT - CLASS SUMMARY")
print("=" * 100)


print(
    class_summary_df[
        [
            "class_id",
            "class_name",
            "audit_points",
            "verified",
            "corrected",
            "rejected",
            "original_label_agreement_percent",
            "manual_usable_percent",
        ]
    ]
    .to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("OVERALL MANUAL AUDIT RESULT")
print("=" * 100)


print(
    f"Total audit points                  : "
    f"{EXPECTED_TOTAL}"
)

print(
    f"VERIFIED                            : "
    f"{verified_count}"
)

print(
    f"CORRECTED                           : "
    f"{corrected_count}"
)

print(
    f"REJECTED                            : "
    f"{rejected_count}"
)

print(
    f"Original-label agreement            : "
    f"{original_label_agreement_percent:.2f}%"
)

print(
    f"Agreement among classifiable points : "
    f"{classifiable_agreement_percent:.2f}%"
)

print(
    f"Correction rate                     : "
    f"{correction_percent:.2f}%"
)

print(
    f"Rejection rate                      : "
    f"{rejection_percent:.2f}%"
)

print(
    f"Manually usable audit points        : "
    f"{accepted_count} / {EXPECTED_TOTAL}"
)

print(
    f"Manual usable rate                  : "
    f"{manual_usable_percent:.2f}%"
)


# =============================================================================
# 18. SHOW CORRECTED POINTS
# =============================================================================

print("\n" + "=" * 100)
print("CORRECTED AUDIT POINTS")
print("=" * 100)


if corrections_df.empty:

    print(
        "No corrected points."
    )

else:

    print(
        corrections_df[
            [
                "sample_id",
                "original_class_name",
                "manual_class_name",
                "review_notes",
            ]
        ]
        .to_string(
            index=False
        )
    )


# =============================================================================
# 19. SHOW REJECTED POINTS
# =============================================================================

print("\n" + "=" * 100)
print("REJECTED AUDIT POINTS")
print("=" * 100)


if rejections_df.empty:

    print(
        "No rejected points."
    )

else:

    print(
        rejections_df[
            [
                "sample_id",
                "original_class_name",
                "review_notes",
            ]
        ]
        .to_string(
            index=False
        )
    )


# =============================================================================
# 20. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("STEP 05F SUMMARY")
print("=" * 100)


print(
    f"All 40 reviews completed       : "
    f"True"
)

print(
    f"10 original points per class   : "
    f"{all_original_class_counts_correct}"
)

print(
    f"Training pool modified         : "
    f"False"
)

print(
    f"Final model validation         : "
    f"False"
)

print(
    f"Overall Step 05F status        : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(AUDIT_RESULTS_CSV)}"
)

print(
    f" - {relative_path(TRAINING_ACTIONS_CSV)}"
)

print(
    f" - {relative_path(CLASS_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(OVERALL_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(CORRECTIONS_CSV)}"
)

print(
    f" - {relative_path(REJECTIONS_CSV)}"
)


print("\nIMPORTANT INTERPRETATION:")

print(
    "1. Original-label agreement = VERIFIED / all 40 audit points."
)

print(
    "2. CORRECTED means a different clear manual class was identified."
)

print(
    "3. REJECTED means the pixel is mixed, unclear or unsuitable "
    "as a clean reference sample."
)

print(
    "4. The audit measures the quality of the reference-generation workflow; "
    "it is not final model accuracy."
)

print(
    "5. The 399-point Step 05D pool has not been modified yet."
)

print(
    "6. A later step can use the training-action table to remove rejected "
    "audit samples and handle corrected samples before model training."
)

print("=" * 100)