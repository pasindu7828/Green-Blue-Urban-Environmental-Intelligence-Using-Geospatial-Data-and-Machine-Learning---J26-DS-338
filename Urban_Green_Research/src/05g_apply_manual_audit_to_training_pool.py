"""
STEP 05G - APPLY MANUAL AUDIT TO LULC TRAINING POOL

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Read the 399 HIGH-confidence Step 05D training-candidate pool.
2. Read the Step 05F manual-audit training actions.
3. Keep VERIFIED audit samples with their original labels.
4. Relabel CORRECTED audit samples using the manual class.
5. Remove REJECTED audit samples.
6. Leave unaudited HIGH-confidence candidates unchanged.
7. Recalculate final class counts and balanced class weights.
8. Produce a curated training-reference pool for later LULC modelling.

IMPORTANT:
- This step does NOT train the Random Forest.
- Only manually audited samples are changed based on manual evidence.
- Unaudited samples remain HIGH-confidence automated reference candidates.
- The manual audit is a quality audit, not final model accuracy validation.
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
# 2. INPUTS
# =============================================================================

TRAINING_POOL_GPKG = (
    REFERENCE_DIR
    / "lulc_high_confidence_training_candidates_2021.gpkg"
)

AUDIT_ACTIONS_CSV = (
    REFERENCE_DIR
    / "lulc_manual_audit_training_actions_2021.csv"
)


# =============================================================================
# 3. OUTPUTS
# =============================================================================

CURATED_TRAINING_GPKG = (
    REFERENCE_DIR
    / "lulc_curated_training_reference_2021.gpkg"
)

CURATED_TRAINING_CSV = (
    REFERENCE_DIR
    / "lulc_curated_training_reference_2021.csv"
)

CLASS_SUMMARY_CSV = (
    REPORT_DIR
    / "05g_curated_training_class_summary.csv"
)

CLASS_WEIGHTS_CSV = (
    REPORT_DIR
    / "05g_curated_training_class_weights.csv"
)

AUDIT_APPLICATION_CSV = (
    REPORT_DIR
    / "05g_manual_audit_application_summary.csv"
)

REMOVED_SAMPLES_CSV = (
    REPORT_DIR
    / "05g_removed_training_samples.csv"
)

RELABELED_SAMPLES_CSV = (
    REPORT_DIR
    / "05g_relabeled_training_samples.csv"
)

PROCESS_SUMMARY_CSV = (
    REPORT_DIR
    / "05g_curated_training_pool_summary.csv"
)


# =============================================================================
# 4. CONSTANTS
# =============================================================================

ANALYSIS_CRS = "EPSG:32644"


CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}


VALID_ACTIONS = {
    "KEEP_ORIGINAL_LABEL",
    "RELABEL_MANUALLY",
    "EXCLUDE_FROM_TRAINING",
}


# =============================================================================
# 5. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """
    Return project-relative path when possible.
    """

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def remove_existing_file(path: Path):
    """
    Remove only generated output files.
    """

    if path.exists():
        path.unlink()


def safe_percent(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safely calculate a percentage.
    """

    if denominator == 0:
        return np.nan

    return float(
        numerator
        / denominator
        * 100
    )


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 05G - APPLY MANUAL AUDIT TO LULC TRAINING POOL")
print("=" * 100)


for required_path in [
    TRAINING_POOL_GPKG,
    AUDIT_ACTIONS_CSV,
]:

    if not required_path.exists():

        raise FileNotFoundError(
            f"Required input not found:\n"
            f"{required_path}"
        )


# =============================================================================
# 7. LOAD STEP 05D TRAINING POOL
# =============================================================================

training_gdf = gpd.read_file(
    TRAINING_POOL_GPKG
)


print(
    f"\nTraining pool : "
    f"{relative_path(TRAINING_POOL_GPKG)}"
)

print(
    f"Features      : "
    f"{len(training_gdf)}"
)

print(
    f"CRS           : "
    f"{training_gdf.crs}"
)


if str(
    training_gdf.crs
) != ANALYSIS_CRS:

    raise ValueError(
        f"Unexpected training CRS: "
        f"{training_gdf.crs}. "
        f"Expected {ANALYSIS_CRS}."
    )


if training_gdf[
    "sample_id"
].duplicated().any():

    raise ValueError(
        "Duplicate sample IDs exist "
        "in the Step 05D training pool."
    )


if training_gdf[
    "landsat_pixel_id"
].duplicated().any():

    raise ValueError(
        "Duplicate Landsat pixels exist "
        "in the Step 05D training pool."
    )


training_gdf[
    "class_id"
] = pd.to_numeric(
    training_gdf[
        "class_id"
    ],
    errors="raise",
).astype(int)


# =============================================================================
# 8. LOAD STEP 05F AUDIT ACTIONS
# =============================================================================

audit_df = pd.read_csv(
    AUDIT_ACTIONS_CSV
)


print(
    f"\nAudit actions : "
    f"{relative_path(AUDIT_ACTIONS_CSV)}"
)

print(
    f"Audit records : "
    f"{len(audit_df)}"
)


required_audit_columns = {
    "sample_id",
    "original_class_id",
    "original_class_name",
    "review_status",
    "manual_class_id",
    "manual_class_name",
    "training_action",
    "review_notes",
}


missing_columns = (
    required_audit_columns
    -
    set(
        audit_df.columns
    )
)


if missing_columns:

    raise ValueError(
        f"Audit action file is missing columns: "
        f"{sorted(missing_columns)}"
    )


if audit_df[
    "sample_id"
].duplicated().any():

    raise ValueError(
        "Duplicate sample IDs exist "
        "in the audit action file."
    )


audit_df[
    "training_action"
] = (
    audit_df[
        "training_action"
    ]
    .astype(str)
    .str.strip()
    .str.upper()
)


invalid_actions = (
    ~audit_df[
        "training_action"
    ].isin(
        VALID_ACTIONS
    )
)


if invalid_actions.any():

    bad_actions = (
        audit_df.loc[
            invalid_actions,
            [
                "sample_id",
                "training_action",
            ],
        ]
    )

    raise ValueError(
        "Invalid training actions found:\n"
        f"{bad_actions.to_string(index=False)}"
    )


# =============================================================================
# 9. CHECK THAT ALL AUDITED SAMPLES EXIST IN THE TRAINING POOL
# =============================================================================

training_sample_ids = set(
    training_gdf[
        "sample_id"
    ].astype(str)
)


audit_sample_ids = set(
    audit_df[
        "sample_id"
    ].astype(str)
)


missing_audit_samples = (
    audit_sample_ids
    -
    training_sample_ids
)


if missing_audit_samples:

    raise ValueError(
        "Some audited samples are not present "
        "in the Step 05D training pool:\n"
        f"{sorted(missing_audit_samples)}"
    )


# =============================================================================
# 10. PRESERVE ORIGINAL CLASS INFORMATION
# =============================================================================

curated_gdf = (
    training_gdf
    .copy()
    .reset_index(
        drop=True
    )
)


curated_gdf[
    "pre_audit_class_id"
] = (
    curated_gdf[
        "class_id"
    ]
)


curated_gdf[
    "pre_audit_class_name"
] = (
    curated_gdf[
        "class_name"
    ]
)


curated_gdf[
    "manual_audit_status"
] = (
    "NOT_AUDITED"
)


curated_gdf[
    "manual_audit_action"
] = (
    "NO_MANUAL_CHANGE"
)


curated_gdf[
    "manual_audit_notes"
] = ""


curated_gdf[
    "manually_audited"
] = False


curated_gdf[
    "label_after_audit"
] = (
    curated_gdf[
        "class_name"
    ]
)


# =============================================================================
# 11. VALIDATE ORIGINAL CLASSES AGAINST AUDIT FILE
# =============================================================================

merged_check = (
    audit_df[
        [
            "sample_id",
            "original_class_id",
        ]
    ]
    .merge(
        training_gdf[
            [
                "sample_id",
                "class_id",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
)


merged_check[
    "original_class_id"
] = pd.to_numeric(
    merged_check[
        "original_class_id"
    ],
    errors="raise",
).astype(int)


original_class_mismatch = (
    merged_check[
        "original_class_id"
    ]
    !=
    merged_check[
        "class_id"
    ]
)


if original_class_mismatch.any():

    bad_rows = (
        merged_check.loc[
            original_class_mismatch
        ]
    )

    raise ValueError(
        "Original class in Step 05F does not match "
        "the Step 05D training pool:\n"
        f"{bad_rows.to_string(index=False)}"
    )


# =============================================================================
# 12. APPLY VERIFIED ACTIONS
# =============================================================================

verified_actions = (
    audit_df[
        audit_df[
            "training_action"
        ]
        == "KEEP_ORIGINAL_LABEL"
    ]
    .copy()
)


for _, action_row in (
    verified_actions.iterrows()
):

    sample_id = str(
        action_row[
            "sample_id"
        ]
    )


    mask = (
        curated_gdf[
            "sample_id"
        ].astype(str)
        == sample_id
    )


    curated_gdf.loc[
        mask,
        "manual_audit_status"
    ] = (
        "VERIFIED"
    )


    curated_gdf.loc[
        mask,
        "manual_audit_action"
    ] = (
        "KEEP_ORIGINAL_LABEL"
    )


    curated_gdf.loc[
        mask,
        "manual_audit_notes"
    ] = (
        str(
            action_row[
                "review_notes"
            ]
        )
        if pd.notna(
            action_row[
                "review_notes"
            ]
        )
        else ""
    )


    curated_gdf.loc[
        mask,
        "manually_audited"
    ] = True


# =============================================================================
# 13. APPLY CORRECTED / RELABELED ACTIONS
# =============================================================================

relabeled_actions = (
    audit_df[
        audit_df[
            "training_action"
        ]
        == "RELABEL_MANUALLY"
    ]
    .copy()
)


relabeled_records = []


for _, action_row in (
    relabeled_actions.iterrows()
):

    sample_id = str(
        action_row[
            "sample_id"
        ]
    )


    manual_class_id = int(
        pd.to_numeric(
            action_row[
                "manual_class_id"
            ],
            errors="raise",
        )
    )


    if manual_class_id not in (
        CLASS_DEFINITIONS
    ):

        raise ValueError(
            f"Invalid manual class ID "
            f"{manual_class_id} for "
            f"{sample_id}."
        )


    manual_class_name = (
        CLASS_DEFINITIONS[
            manual_class_id
        ]
    )


    mask = (
        curated_gdf[
            "sample_id"
        ].astype(str)
        == sample_id
    )


    original_row = (
        curated_gdf.loc[
            mask
        ]
        .iloc[0]
    )


    original_class_id = int(
        original_row[
            "class_id"
        ]
    )


    original_class_name = str(
        original_row[
            "class_name"
        ]
    )


    curated_gdf.loc[
        mask,
        "class_id"
    ] = (
        manual_class_id
    )


    curated_gdf.loc[
        mask,
        "class_name"
    ] = (
        manual_class_name
    )


    curated_gdf.loc[
        mask,
        "manual_audit_status"
    ] = (
        "CORRECTED"
    )


    curated_gdf.loc[
        mask,
        "manual_audit_action"
    ] = (
        "RELABEL_MANUALLY"
    )


    curated_gdf.loc[
        mask,
        "manual_audit_notes"
    ] = (
        str(
            action_row[
                "review_notes"
            ]
        )
        if pd.notna(
            action_row[
                "review_notes"
            ]
        )
        else ""
    )


    curated_gdf.loc[
        mask,
        "manually_audited"
    ] = True


    curated_gdf.loc[
        mask,
        "label_after_audit"
    ] = (
        manual_class_name
    )


    relabeled_records.append(
        {
            "sample_id": (
                sample_id
            ),
            "original_class_id": (
                original_class_id
            ),
            "original_class_name": (
                original_class_name
            ),
            "new_class_id": (
                manual_class_id
            ),
            "new_class_name": (
                manual_class_name
            ),
            "review_notes": (
                action_row[
                    "review_notes"
                ]
            ),
        }
    )


# =============================================================================
# 14. IDENTIFY REJECTED SAMPLES
# =============================================================================

rejected_actions = (
    audit_df[
        audit_df[
            "training_action"
        ]
        == "EXCLUDE_FROM_TRAINING"
    ]
    .copy()
)


rejected_sample_ids = (
    rejected_actions[
        "sample_id"
    ]
    .astype(str)
    .tolist()
)


removed_records = []


for sample_id in (
    rejected_sample_ids
):

    rejected_row = (
        curated_gdf[
            curated_gdf[
                "sample_id"
            ].astype(str)
            == sample_id
        ]
        .iloc[0]
    )


    audit_action_row = (
        rejected_actions[
            rejected_actions[
                "sample_id"
            ].astype(str)
            == sample_id
        ]
        .iloc[0]
    )


    removed_records.append(
        {
            "sample_id": (
                sample_id
            ),
            "class_id": (
                int(
                    rejected_row[
                        "class_id"
                    ]
                )
            ),
            "class_name": (
                rejected_row[
                    "class_name"
                ]
            ),
            "landsat_pixel_id": (
                rejected_row[
                    "landsat_pixel_id"
                ]
            ),
            "reason": (
                "MANUAL_AUDIT_REJECTED"
            ),
            "review_notes": (
                audit_action_row[
                    "review_notes"
                ]
            ),
        }
    )


# =============================================================================
# 15. REMOVE REJECTED SAMPLES
# =============================================================================

curated_gdf = (
    curated_gdf[
        ~curated_gdf[
            "sample_id"
        ]
        .astype(str)
        .isin(
            rejected_sample_ids
        )
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 16. FINAL TRAINING-REFERENCE METADATA
# =============================================================================

curated_gdf[
    "training_reference_status"
] = (
    "CURATED_REFERENCE"
)


curated_gdf[
    "reference_source"
] = np.where(
    curated_gdf[
        "manually_audited"
    ],
    (
        "High-confidence automated reference "
        "+ manual quality audit"
    ),
    (
        "High-confidence automated reference"
    ),
)


curated_gdf[
    "independent_ground_truth"
] = False


curated_gdf[
    "final_model_validation_sample"
] = False


# =============================================================================
# 17. VALIDATE FINAL CURATED POOL
# =============================================================================

final_total = len(
    curated_gdf
)


expected_final_total = (
    len(
        training_gdf
    )
    -
    len(
        rejected_sample_ids
    )
)


if final_total != expected_final_total:

    raise ValueError(
        f"Unexpected final pool size. "
        f"Expected {expected_final_total}, "
        f"found {final_total}."
    )


remaining_rejected_ids = set(
    rejected_sample_ids
).intersection(
    set(
        curated_gdf[
            "sample_id"
        ].astype(str)
    )
)


if remaining_rejected_ids:

    raise ValueError(
        "Rejected samples remain in curated pool:\n"
        f"{sorted(remaining_rejected_ids)}"
    )


if curated_gdf[
    "sample_id"
].duplicated().any():

    raise ValueError(
        "Duplicate sample IDs exist "
        "after audit application."
    )


if curated_gdf[
    "landsat_pixel_id"
].duplicated().any():

    raise ValueError(
        "Duplicate Landsat pixels exist "
        "after audit application."
    )


invalid_final_classes = (
    ~curated_gdf[
        "class_id"
    ].isin(
        CLASS_DEFINITIONS.keys()
    )
)


if invalid_final_classes.any():

    raise ValueError(
        "Invalid class IDs exist "
        "in the final curated pool."
    )


all_classes_present = (
    set(
        curated_gdf[
            "class_id"
        ].unique()
    )
    ==
    set(
        CLASS_DEFINITIONS.keys()
    )
)


# =============================================================================
# 18. FINAL CLASS COUNTS
# =============================================================================

class_summary_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    before_count = int(
        (
            training_gdf[
                "class_id"
            ]
            == class_id
        ).sum()
    )


    after_count = int(
        (
            curated_gdf[
                "class_id"
            ]
            == class_id
        ).sum()
    )


    manually_audited_count = int(
        (
            (
                curated_gdf[
                    "class_id"
                ]
                == class_id
            )
            &
            (
                curated_gdf[
                    "manually_audited"
                ]
                == True
            )
        ).sum()
    )


    unaudited_count = int(
        (
            (
                curated_gdf[
                    "class_id"
                ]
                == class_id
            )
            &
            (
                curated_gdf[
                    "manually_audited"
                ]
                == False
            )
        ).sum()
    )


    class_summary_records.append(
        {
            "class_id": (
                class_id
            ),
            "class_name": (
                class_name
            ),
            "before_audit_count": (
                before_count
            ),
            "after_audit_count": (
                after_count
            ),
            "net_change": (
                after_count
                -
                before_count
            ),
            "percent_of_final_pool": (
                safe_percent(
                    after_count,
                    final_total,
                )
            ),
            "manually_audited_retained": (
                manually_audited_count
            ),
            "unaudited_high_confidence": (
                unaudited_count
            ),
        }
    )


class_summary_df = pd.DataFrame(
    class_summary_records
)


# =============================================================================
# 19. RECALCULATE BALANCED CLASS WEIGHTS
# =============================================================================

number_of_classes = len(
    CLASS_DEFINITIONS
)


class_weight_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    class_count = int(
        (
            curated_gdf[
                "class_id"
            ]
            == class_id
        ).sum()
    )


    if class_count == 0:

        class_weight = np.nan

    else:

        class_weight = float(
            final_total
            /
            (
                number_of_classes
                *
                class_count
            )
        )


    class_weight_records.append(
        {
            "class_id": (
                class_id
            ),
            "class_name": (
                class_name
            ),
            "sample_count": (
                class_count
            ),
            "balanced_class_weight": (
                class_weight
            ),
        }
    )


class_weights_df = pd.DataFrame(
    class_weight_records
)


# Remove old Step 05D class-weight field if present,
# then add the updated Step 05G weight.

if (
    "balanced_class_weight"
    in curated_gdf.columns
):

    curated_gdf.drop(
        columns=[
            "balanced_class_weight"
        ],
        inplace=True,
    )


curated_gdf = (
    curated_gdf
    .merge(
        class_weights_df[
            [
                "class_id",
                "balanced_class_weight",
            ]
        ],
        on="class_id",
        how="left",
        validate="many_to_one",
    )
)


# =============================================================================
# 20. AUDIT APPLICATION SUMMARY
# =============================================================================

verified_count = len(
    verified_actions
)

corrected_count = len(
    relabeled_actions
)

rejected_count = len(
    rejected_actions
)

audited_total = len(
    audit_df
)

unaudited_retained = int(
    (
        curated_gdf[
            "manually_audited"
        ]
        == False
    ).sum()
)


audit_application_df = pd.DataFrame(
    [
        {
            "input_training_candidates": (
                len(
                    training_gdf
                )
            ),
            "audited_samples": (
                audited_total
            ),
            "verified_kept": (
                verified_count
            ),
            "corrected_relabeled": (
                corrected_count
            ),
            "rejected_removed": (
                rejected_count
            ),
            "unaudited_high_confidence_retained": (
                unaudited_retained
            ),
            "final_curated_training_reference_count": (
                final_total
            ),
        }
    ]
)


# =============================================================================
# 21. CREATE REMOVED / RELABELED REPORTS
# =============================================================================

removed_df = pd.DataFrame(
    removed_records
)


relabeled_df = pd.DataFrame(
    relabeled_records
)


# =============================================================================
# 22. PROCESS SUMMARY
# =============================================================================

largest_class_count = int(
    class_summary_df[
        "after_audit_count"
    ].max()
)


smallest_class_count = int(
    class_summary_df[
        "after_audit_count"
    ].min()
)


imbalance_ratio = float(
    largest_class_count
    /
    smallest_class_count
)


open_bare_count = int(
    (
        curated_gdf[
            "class_id"
        ]
        == 3
    ).sum()
)


if open_bare_count < 20:

    open_bare_status = (
        "LOW_SAMPLE_WARNING"
    )

elif open_bare_count < 40:

    open_bare_status = (
        "LIMITED_BUT_USABLE_WITH_CAUTION"
    )

else:

    open_bare_status = (
        "ADEQUATE_FOR_PRELIMINARY_TRAINING"
    )


overall_pass = all(
    [
        final_total
        == expected_final_total,
        len(
            remaining_rejected_ids
        )
        == 0,
        not curated_gdf[
            "sample_id"
        ].duplicated().any(),
        not curated_gdf[
            "landsat_pixel_id"
        ].duplicated().any(),
        all_classes_present,
    ]
)


process_summary_df = pd.DataFrame(
    [
        {
            "step05d_input_candidates": (
                len(
                    training_gdf
                )
            ),
            "step05f_audited_samples": (
                audited_total
            ),
            "verified_kept": (
                verified_count
            ),
            "corrected_relabeled": (
                corrected_count
            ),
            "rejected_removed": (
                rejected_count
            ),
            "final_curated_pool": (
                final_total
            ),
            "largest_class_count": (
                largest_class_count
            ),
            "smallest_class_count": (
                smallest_class_count
            ),
            "class_imbalance_ratio": (
                imbalance_ratio
            ),
            "open_bare_count": (
                open_bare_count
            ),
            "open_bare_status": (
                open_bare_status
            ),
            "all_four_classes_present": (
                all_classes_present
            ),
            "unique_sample_ids": (
                not curated_gdf[
                    "sample_id"
                ].duplicated().any()
            ),
            "unique_landsat_pixels": (
                not curated_gdf[
                    "landsat_pixel_id"
                ].duplicated().any()
            ),
            "final_model_validation_dataset": (
                False
            ),
            "overall_status": (
                "PASS"
                if overall_pass
                else
                "REVIEW_REQUIRED"
            ),
        }
    ]
)


# =============================================================================
# 23. SAVE OUTPUTS
# =============================================================================

remove_existing_file(
    CURATED_TRAINING_GPKG
)


curated_gdf.to_file(
    CURATED_TRAINING_GPKG,
    layer="curated_training_reference_2021",
    driver="GPKG",
)


curated_gdf.drop(
    columns="geometry"
).to_csv(
    CURATED_TRAINING_CSV,
    index=False,
)


class_summary_df.to_csv(
    CLASS_SUMMARY_CSV,
    index=False,
)


class_weights_df.to_csv(
    CLASS_WEIGHTS_CSV,
    index=False,
)


audit_application_df.to_csv(
    AUDIT_APPLICATION_CSV,
    index=False,
)


removed_df.to_csv(
    REMOVED_SAMPLES_CSV,
    index=False,
)


relabeled_df.to_csv(
    RELABELED_SAMPLES_CSV,
    index=False,
)


process_summary_df.to_csv(
    PROCESS_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 24. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("MANUAL AUDIT APPLICATION")
print("=" * 100)


print(
    f"Input Step 05D candidates       : "
    f"{len(training_gdf)}"
)

print(
    f"Audited samples                 : "
    f"{audited_total}"
)

print(
    f"VERIFIED kept                   : "
    f"{verified_count}"
)

print(
    f"CORRECTED relabeled             : "
    f"{corrected_count}"
)

print(
    f"REJECTED removed                : "
    f"{rejected_count}"
)

print(
    f"Unaudited HIGH samples retained : "
    f"{unaudited_retained}"
)


# =============================================================================
# 25. REMOVED SAMPLES
# =============================================================================

print("\n" + "=" * 100)
print("REMOVED TRAINING SAMPLES")
print("=" * 100)


if removed_df.empty:

    print(
        "No samples removed."
    )

else:

    print(
        removed_df.to_string(
            index=False
        )
    )


# =============================================================================
# 26. RELABELED SAMPLES
# =============================================================================

print("\n" + "=" * 100)
print("RELABELED TRAINING SAMPLES")
print("=" * 100)


if relabeled_df.empty:

    print(
        "No samples relabeled."
    )

else:

    print(
        relabeled_df.to_string(
            index=False
        )
    )


# =============================================================================
# 27. FINAL CLASS DISTRIBUTION
# =============================================================================

print("\n" + "=" * 100)
print("FINAL CURATED TRAINING-REFERENCE CLASS DISTRIBUTION")
print("=" * 100)


print(
    class_summary_df.to_string(
        index=False
    )
)


# =============================================================================
# 28. UPDATED CLASS WEIGHTS
# =============================================================================

print("\n" + "=" * 100)
print("UPDATED BALANCED CLASS WEIGHTS")
print("=" * 100)


print(
    class_weights_df.to_string(
        index=False
    )
)


# =============================================================================
# 29. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("STEP 05G SUMMARY")
print("=" * 100)


print(
    f"Original Step 05D pool          : "
    f"{len(training_gdf)}"
)

print(
    f"Rejected samples removed        : "
    f"{rejected_count}"
)

print(
    f"Corrected samples relabeled     : "
    f"{corrected_count}"
)

print(
    f"Final curated reference pool    : "
    f"{final_total}"
)

print(
    f"All four classes present        : "
    f"{all_classes_present}"
)

print(
    f"Unique sample IDs               : "
    f"{not curated_gdf['sample_id'].duplicated().any()}"
)

print(
    f"Unique Landsat pixels           : "
    f"{not curated_gdf['landsat_pixel_id'].duplicated().any()}"
)

print(
    f"Open / Bare samples             : "
    f"{open_bare_count}"
)

print(
    f"Open / Bare status              : "
    f"{open_bare_status}"
)

print(
    f"Class imbalance ratio           : "
    f"{imbalance_ratio:.4f}"
)

print(
    f"Overall Step 05G status         : "
    f"{'PASS' if overall_pass else 'REVIEW REQUIRED'}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(CURATED_TRAINING_GPKG)}"
)

print(
    f" - {relative_path(CURATED_TRAINING_CSV)}"
)

print(
    f" - {relative_path(CLASS_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(CLASS_WEIGHTS_CSV)}"
)

print(
    f" - {relative_path(AUDIT_APPLICATION_CSV)}"
)

print(
    f" - {relative_path(REMOVED_SAMPLES_CSV)}"
)

print(
    f" - {relative_path(RELABELED_SAMPLES_CSV)}"
)

print(
    f" - {relative_path(PROCESS_SUMMARY_CSV)}"
)


print("\nIMPORTANT INTERPRETATION:")

print(
    "1. The final pool contains curated HIGH-confidence reference samples."
)

print(
    "2. Only manually audited samples were removed or relabeled."
)

print(
    "3. Unaudited candidates remain automated HIGH-confidence references."
)

print(
    "4. Manually audited points are explicitly marked in the output."
)

print(
    "5. This is still not an independent ground-truth validation dataset."
)

print(
    "6. No LULC classifier was trained in Step 05G."
)

print(
    "7. Final model validation must remain separate from training data."
)

print("=" * 100)