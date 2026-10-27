"""
STEP 05D - HIGH-CONFIDENCE LULC TRAINING CANDIDATE POOL

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Read the Step 05C purity-screened candidate dataset.
2. Keep only HIGH-confidence candidates.
3. Ensure each Landsat 30 m pixel appears only once.
4. Detect any conflicting labels within the same Landsat pixel.
5. Preserve all legitimate high-confidence samples without artificial balancing.
6. Calculate class weights for later ML training.
7. Produce the final high-confidence TRAINING-CANDIDATE pool.

Research LULC Classes:
1 - Green / Vegetation
2 - Built-up
3 - Open / Bare
4 - Water

IMPORTANT:
- This is a TRAINING-CANDIDATE pool, not final model training data yet.
- No classifier is trained in this step.
- No synthetic labels are created.
- No class is artificially oversampled or undersampled here.
- Final independent validation samples will be created separately.
- High confidence is automated screening evidence, not independent ground truth.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REFERENCE_DIR = (
    PROCESSED_DIR
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

SCREENED_GPKG = (
    REFERENCE_DIR
    / "lulc_candidate_purity_screened_2021.gpkg"
)


# =============================================================================
# 3. OUTPUTS
# =============================================================================

TRAINING_POOL_GPKG = (
    REFERENCE_DIR
    / "lulc_high_confidence_training_candidates_2021.gpkg"
)

TRAINING_POOL_CSV = (
    REFERENCE_DIR
    / "lulc_high_confidence_training_candidates_2021.csv"
)

CLASS_SUMMARY_REPORT = (
    REPORT_DIR
    / "05d_training_candidate_class_summary.csv"
)

CLASS_WEIGHT_REPORT = (
    REPORT_DIR
    / "05d_training_candidate_class_weights.csv"
)

DUPLICATE_PIXEL_REPORT = (
    REPORT_DIR
    / "05d_duplicate_landsat_pixel_check.csv"
)

PROCESS_SUMMARY_REPORT = (
    REPORT_DIR
    / "05d_high_confidence_training_pool_summary.csv"
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
    Remove only a generated output file.
    """

    if path.exists():
        path.unlink()


def safe_percent(
    numerator: float,
    denominator: float
) -> float:
    """
    Safely calculate percentage.
    """

    if denominator == 0:
        return 0.0

    return float(
        numerator
        / denominator
        * 100
    )


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 05D - HIGH-CONFIDENCE LULC TRAINING CANDIDATE POOL")
print("=" * 100)


if not SCREENED_GPKG.exists():

    raise FileNotFoundError(
        f"Step 05C screened dataset not found:\n"
        f"{SCREENED_GPKG}"
    )


screened_gdf = gpd.read_file(
    SCREENED_GPKG
)


print(
    f"\nInput screened dataset : "
    f"{relative_path(SCREENED_GPKG)}"
)

print(
    f"Input candidates       : "
    f"{len(screened_gdf)}"
)

print(
    f"CRS                    : "
    f"{screened_gdf.crs}"
)


# =============================================================================
# 7. VALIDATE INPUT STRUCTURE
# =============================================================================

if str(
    screened_gdf.crs
) != ANALYSIS_CRS:

    raise ValueError(
        f"Unexpected CRS: "
        f"{screened_gdf.crs}. "
        f"Expected {ANALYSIS_CRS}."
    )


required_columns = {
    "sample_id",
    "class_id",
    "class_name",
    "screen_confidence",
    "high_confidence_candidate",
    "landsat_row",
    "landsat_col",
    "landsat_valid",
    "wc_valid_cells",
    "wc_matching_cells",
    "wc_pixel_purity",
    "max_spectral_robust_z",
    "spectral_nearest_class",
    "spectral_class_agreement",
    "spectral_margin",
}


missing_columns = (
    required_columns
    -
    set(
        screened_gdf.columns
    )
)


if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )


if screened_gdf[
    "sample_id"
].duplicated().any():

    raise ValueError(
        "Duplicate sample IDs exist "
        "in the Step 05C dataset."
    )


screened_gdf[
    "class_id"
] = pd.to_numeric(
    screened_gdf[
        "class_id"
    ],
    errors="raise",
).astype(int)


screened_gdf[
    "landsat_row"
] = pd.to_numeric(
    screened_gdf[
        "landsat_row"
    ],
    errors="raise",
).astype(int)


screened_gdf[
    "landsat_col"
] = pd.to_numeric(
    screened_gdf[
        "landsat_col"
    ],
    errors="raise",
).astype(int)


# =============================================================================
# 8. SCREENING DISTRIBUTION BEFORE FILTERING
# =============================================================================

print("\n" + "=" * 100)
print("STEP 05C CONFIDENCE DISTRIBUTION")
print("=" * 100)


confidence_counts = (
    screened_gdf[
        "screen_confidence"
    ]
    .value_counts()
)


for confidence_name in [
    "HIGH",
    "MEDIUM",
    "LOW",
]:

    count = int(
        confidence_counts.get(
            confidence_name,
            0
        )
    )

    print(
        f"{confidence_name:<8}: "
        f"{count}"
    )


# =============================================================================
# 9. SELECT ONLY HIGH-CONFIDENCE CANDIDATES
# =============================================================================

high_gdf = (
    screened_gdf[
        screened_gdf[
            "screen_confidence"
        ]
        .astype(str)
        .str.upper()
        .eq("HIGH")
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 100)
print("HIGH-CONFIDENCE FILTER")
print("=" * 100)


print(
    f"Input candidates       : "
    f"{len(screened_gdf)}"
)

print(
    f"HIGH-confidence kept   : "
    f"{len(high_gdf)}"
)

print(
    f"Candidates removed     : "
    f"{len(screened_gdf) - len(high_gdf)}"
)


if len(
    high_gdf
) == 0:

    raise ValueError(
        "No HIGH-confidence candidates found."
    )


# =============================================================================
# 10. BASIC HIGH-CONFIDENCE VALIDATION
# =============================================================================

invalid_landsat_high = int(
    (
        high_gdf[
            "landsat_valid"
        ]
        == False
    ).sum()
)


spectral_disagreement_high = int(
    (
        high_gdf[
            "spectral_class_agreement"
        ]
        == False
    ).sum()
)


if invalid_landsat_high > 0:

    raise ValueError(
        f"{invalid_landsat_high} HIGH-confidence candidates "
        f"have invalid Landsat pixels."
    )


if spectral_disagreement_high > 0:

    raise ValueError(
        f"{spectral_disagreement_high} HIGH-confidence candidates "
        f"do not agree with their nearest spectral class."
    )


# =============================================================================
# 11. CREATE UNIQUE LANDSAT PIXEL ID
# =============================================================================

high_gdf[
    "landsat_pixel_id"
] = (
    high_gdf[
        "landsat_row"
    ].astype(str)
    +
    "_"
    +
    high_gdf[
        "landsat_col"
    ].astype(str)
)


# =============================================================================
# 12. CHECK DUPLICATE LANDSAT PIXELS
# =============================================================================

pixel_counts = (
    high_gdf[
        "landsat_pixel_id"
    ]
    .value_counts()
)


duplicate_pixel_ids = (
    pixel_counts[
        pixel_counts > 1
    ]
    .index
    .tolist()
)


duplicate_records = (
    high_gdf[
        high_gdf[
            "landsat_pixel_id"
        ].isin(
            duplicate_pixel_ids
        )
    ]
    .copy()
)


duplicate_pixel_count = len(
    duplicate_pixel_ids
)


duplicate_candidate_count = len(
    duplicate_records
)


# =============================================================================
# 13. CHECK CONFLICTING LABELS INSIDE DUPLICATE PIXELS
# =============================================================================

conflicting_pixel_ids = []


if duplicate_pixel_count > 0:

    for pixel_id, group in (
        duplicate_records.groupby(
            "landsat_pixel_id"
        )
    ):

        unique_classes = (
            group[
                "class_id"
            ]
            .unique()
        )

        if len(
            unique_classes
        ) > 1:

            conflicting_pixel_ids.append(
                pixel_id
            )


conflicting_pixel_count = len(
    conflicting_pixel_ids
)


print("\n" + "=" * 100)
print("LANDSAT PIXEL DUPLICATE CHECK")
print("=" * 100)


print(
    f"Duplicate Landsat pixel IDs    : "
    f"{duplicate_pixel_count}"
)

print(
    f"Candidates in duplicate pixels : "
    f"{duplicate_candidate_count}"
)

print(
    f"Conflicting-label pixels        : "
    f"{conflicting_pixel_count}"
)


# =============================================================================
# 14. SAVE DUPLICATE PIXEL REPORT
# =============================================================================

if duplicate_records.empty:

    duplicate_report_df = pd.DataFrame(
        columns=[
            "sample_id",
            "class_id",
            "class_name",
            "landsat_pixel_id",
            "landsat_row",
            "landsat_col",
            "wc_pixel_purity",
            "max_spectral_robust_z",
            "spectral_margin",
            "conflicting_label_pixel",
        ]
    )

else:

    duplicate_report_df = (
        duplicate_records[
            [
                "sample_id",
                "class_id",
                "class_name",
                "landsat_pixel_id",
                "landsat_row",
                "landsat_col",
                "wc_pixel_purity",
                "max_spectral_robust_z",
                "spectral_margin",
            ]
        ]
        .copy()
    )


    duplicate_report_df[
        "conflicting_label_pixel"
    ] = (
        duplicate_report_df[
            "landsat_pixel_id"
        ].isin(
            conflicting_pixel_ids
        )
    )


duplicate_report_df.to_csv(
    DUPLICATE_PIXEL_REPORT,
    index=False,
)


# =============================================================================
# 15. DEDUPLICATE LANDSAT PIXELS
# =============================================================================

# If more than one candidate belongs to the same Landsat pixel,
# retain only the strongest candidate.
#
# Selection order:
# 1. Higher WorldCover pixel purity
# 2. Lower class-specific spectral robust-z
# 3. Larger positive spectral margin
# 4. sample_id for deterministic tie-breaking


high_gdf[
    "_sort_wc_purity"
] = pd.to_numeric(
    high_gdf[
        "wc_pixel_purity"
    ],
    errors="coerce",
).fillna(
    -np.inf
)


high_gdf[
    "_sort_robust_z"
] = pd.to_numeric(
    high_gdf[
        "max_spectral_robust_z"
    ],
    errors="coerce",
).fillna(
    np.inf
)


high_gdf[
    "_sort_margin"
] = pd.to_numeric(
    high_gdf[
        "spectral_margin"
    ],
    errors="coerce",
).fillna(
    -np.inf
)


high_sorted = (
    high_gdf
    .sort_values(
        by=[
            "landsat_pixel_id",
            "_sort_wc_purity",
            "_sort_robust_z",
            "_sort_margin",
            "sample_id",
        ],
        ascending=[
            True,
            False,
            True,
            False,
            True,
        ],
    )
    .copy()
)


training_pool_gdf = (
    high_sorted
    .drop_duplicates(
        subset=[
            "landsat_pixel_id"
        ],
        keep="first",
    )
    .copy()
    .reset_index(
        drop=True
    )
)


removed_duplicate_candidates = (
    len(high_gdf)
    -
    len(training_pool_gdf)
)


training_pool_gdf.drop(
    columns=[
        "_sort_wc_purity",
        "_sort_robust_z",
        "_sort_margin",
    ],
    inplace=True,
)


# =============================================================================
# 16. FINAL UNIQUE-PIXEL VALIDATION
# =============================================================================

remaining_duplicate_pixels = int(
    training_pool_gdf[
        "landsat_pixel_id"
    ]
    .duplicated()
    .sum()
)


if remaining_duplicate_pixels > 0:

    raise ValueError(
        "Duplicate Landsat pixels remain "
        "after deduplication."
    )


# =============================================================================
# 17. ADD TRAINING-POOL METADATA
# =============================================================================

training_pool_gdf[
    "pool_role"
] = (
    "HIGH_CONFIDENCE_TRAINING_CANDIDATE"
)


training_pool_gdf[
    "reference_year"
] = 2021


training_pool_gdf[
    "label_source"
] = (
    "WorldCover-derived + automated purity/spectral screening"
)


training_pool_gdf[
    "independent_ground_truth"
] = False


training_pool_gdf[
    "final_training_status"
] = (
    "CANDIDATE_NOT_YET_FINAL"
)


training_pool_gdf[
    "future_validation_status"
] = (
    "NOT_ASSIGNED"
)


# =============================================================================
# 18. CLASS DISTRIBUTION
# =============================================================================

class_summary_records = []


total_training_candidates = len(
    training_pool_gdf
)


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    class_df = (
        training_pool_gdf[
            training_pool_gdf[
                "class_id"
            ]
            == class_id
        ]
    )


    count = len(
        class_df
    )


    class_summary_records.append(
        {
            "class_id": (
                class_id
            ),
            "class_name": (
                class_name
            ),
            "training_candidates": (
                count
            ),
            "percent_of_pool": (
                safe_percent(
                    count,
                    total_training_candidates
                )
            ),
            "mean_wc_pixel_purity": (
                float(
                    class_df[
                        "wc_pixel_purity"
                    ].mean()
                )
                if count > 0
                else np.nan
            ),
            "median_wc_pixel_purity": (
                float(
                    class_df[
                        "wc_pixel_purity"
                    ].median()
                )
                if count > 0
                else np.nan
            ),
            "mean_max_spectral_robust_z": (
                float(
                    class_df[
                        "max_spectral_robust_z"
                    ].mean()
                )
                if count > 0
                else np.nan
            ),
            "mean_spectral_margin": (
                float(
                    class_df[
                        "spectral_margin"
                    ].mean()
                )
                if count > 0
                else np.nan
            ),
        }
    )


class_summary_df = pd.DataFrame(
    class_summary_records
)


# =============================================================================
# 19. CALCULATE CLASS WEIGHTS
# =============================================================================

# Balanced class-weight formula:
#
# weight(class) = N / (K * n_class)
#
# where:
# N = total samples
# K = number of classes
# n_class = samples in that class
#
# This allows later models to pay more attention to rare classes
# without creating synthetic labels.


number_of_classes = len(
    CLASS_DEFINITIONS
)


class_weight_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    class_count = int(
        (
            training_pool_gdf[
                "class_id"
            ]
            == class_id
        ).sum()
    )


    if class_count == 0:

        class_weight = np.nan

    else:

        class_weight = float(
            total_training_candidates
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


class_weight_df = pd.DataFrame(
    class_weight_records
)


training_pool_gdf = (
    training_pool_gdf
    .merge(
        class_weight_df[
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
# 20. CLASS IMBALANCE DIAGNOSTICS
# =============================================================================

class_counts = (
    class_weight_df[
        "sample_count"
    ]
    .to_numpy()
)


positive_class_counts = (
    class_counts[
        class_counts > 0
    ]
)


if len(
    positive_class_counts
) > 0:

    largest_class_count = int(
        np.max(
            positive_class_counts
        )
    )

    smallest_class_count = int(
        np.min(
            positive_class_counts
        )
    )

    imbalance_ratio = float(
        largest_class_count
        /
        smallest_class_count
    )

else:

    largest_class_count = 0
    smallest_class_count = 0
    imbalance_ratio = np.nan


# =============================================================================
# 21. OPEN / BARE CHECK
# =============================================================================

open_bare_count = int(
    (
        training_pool_gdf[
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


# =============================================================================
# 22. FINAL POOL CHECKS
# =============================================================================

all_classes_present = (
    set(
        training_pool_gdf[
            "class_id"
        ].unique()
    )
    ==
    set(
        CLASS_DEFINITIONS.keys()
    )
)


all_high_confidence = bool(
    (
        training_pool_gdf[
            "screen_confidence"
        ]
        .astype(str)
        .str.upper()
        ==
        "HIGH"
    ).all()
)


all_landsat_valid = bool(
    (
        training_pool_gdf[
            "landsat_valid"
        ]
        == True
    ).all()
)


all_spectral_agreement = bool(
    (
        training_pool_gdf[
            "spectral_class_agreement"
        ]
        == True
    ).all()
)


overall_pass = all(
    [
        all_classes_present,
        all_high_confidence,
        all_landsat_valid,
        all_spectral_agreement,
        remaining_duplicate_pixels == 0,
        conflicting_pixel_count == 0,
    ]
)


# =============================================================================
# 23. SAVE OUTPUTS
# =============================================================================

remove_existing_file(
    TRAINING_POOL_GPKG
)


training_pool_gdf.to_file(
    TRAINING_POOL_GPKG,
    layer="high_confidence_training_candidates",
    driver="GPKG",
)


training_pool_gdf.drop(
    columns="geometry"
).to_csv(
    TRAINING_POOL_CSV,
    index=False,
)


class_summary_df.to_csv(
    CLASS_SUMMARY_REPORT,
    index=False,
)


class_weight_df.to_csv(
    CLASS_WEIGHT_REPORT,
    index=False,
)


# =============================================================================
# 24. PROCESS SUMMARY
# =============================================================================

process_summary_df = pd.DataFrame(
    [
        {
            "step05c_total_candidates": (
                len(
                    screened_gdf
                )
            ),
            "step05c_high_confidence_candidates": (
                len(
                    high_gdf
                )
            ),
            "duplicate_landsat_pixel_ids": (
                duplicate_pixel_count
            ),
            "candidates_removed_by_pixel_deduplication": (
                removed_duplicate_candidates
            ),
            "conflicting_label_pixels": (
                conflicting_pixel_count
            ),
            "final_training_candidate_pool": (
                len(
                    training_pool_gdf
                )
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
            "open_bare_candidate_count": (
                open_bare_count
            ),
            "open_bare_status": (
                open_bare_status
            ),
            "all_classes_present": (
                all_classes_present
            ),
            "all_high_confidence": (
                all_high_confidence
            ),
            "all_landsat_valid": (
                all_landsat_valid
            ),
            "all_spectral_agreement": (
                all_spectral_agreement
            ),
            "unique_landsat_pixels": (
                remaining_duplicate_pixels
                == 0
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


process_summary_df.to_csv(
    PROCESS_SUMMARY_REPORT,
    index=False,
)


# =============================================================================
# 25. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("FINAL TRAINING-CANDIDATE CLASS DISTRIBUTION")
print("=" * 100)


print(
    class_summary_df[
        [
            "class_id",
            "class_name",
            "training_candidates",
            "percent_of_pool",
            "mean_wc_pixel_purity",
        ]
    ]
    .to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("BALANCED CLASS WEIGHTS FOR LATER MODEL TRAINING")
print("=" * 100)


print(
    class_weight_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("CLASS IMBALANCE CHECK")
print("=" * 100)


print(
    f"Largest class count     : "
    f"{largest_class_count}"
)

print(
    f"Smallest class count    : "
    f"{smallest_class_count}"
)

print(
    f"Imbalance ratio         : "
    f"{imbalance_ratio:.4f}"
)

print(
    f"Open / Bare candidates  : "
    f"{open_bare_count}"
)

print(
    f"Open / Bare status      : "
    f"{open_bare_status}"
)


# =============================================================================
# 26. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("STEP 05D SUMMARY")
print("=" * 100)


print(
    f"Original screened candidates   : "
    f"{len(screened_gdf)}"
)

print(
    f"HIGH-confidence candidates     : "
    f"{len(high_gdf)}"
)

print(
    f"Duplicate Landsat pixels       : "
    f"{duplicate_pixel_count}"
)

print(
    f"Removed duplicate candidates   : "
    f"{removed_duplicate_candidates}"
)

print(
    f"Conflicting-label pixels        : "
    f"{conflicting_pixel_count}"
)

print(
    f"Final training candidate pool  : "
    f"{len(training_pool_gdf)}"
)

print(
    f"All four classes present       : "
    f"{all_classes_present}"
)

print(
    f"All candidates HIGH confidence : "
    f"{all_high_confidence}"
)

print(
    f"All Landsat pixels valid       : "
    f"{all_landsat_valid}"
)

print(
    f"All spectral classes agree     : "
    f"{all_spectral_agreement}"
)

print(
    f"Unique Landsat pixels          : "
    f"{remaining_duplicate_pixels == 0}"
)

print(
    f"\nOverall Step 05D status        : "
    f"{'PASS' if overall_pass else 'REVIEW REQUIRED'}"
)


print("\nOutputs created:")

print(
    f" - "
    f"{relative_path(TRAINING_POOL_GPKG)}"
)

print(
    f" - "
    f"{relative_path(TRAINING_POOL_CSV)}"
)

print(
    f" - "
    f"{relative_path(CLASS_SUMMARY_REPORT)}"
)

print(
    f" - "
    f"{relative_path(CLASS_WEIGHT_REPORT)}"
)

print(
    f" - "
    f"{relative_path(DUPLICATE_PIXEL_REPORT)}"
)

print(
    f" - "
    f"{relative_path(PROCESS_SUMMARY_REPORT)}"
)


print("\nIMPORTANT INTERPRETATION:")

print(
    "1. These records are HIGH-confidence training CANDIDATES, "
    "not independently verified ground truth."
)

print(
    "2. MEDIUM and LOW candidates were excluded from this pool."
)

print(
    "3. One record is retained per Landsat 30 m pixel."
)

print(
    "4. Class imbalance is preserved rather than artificially "
    "changing the real candidate counts."
)

print(
    "5. Balanced class weights were calculated for later model training."
)

print(
    "6. Open / Bare remains the rarest class and must be evaluated carefully."
)

print(
    "7. Independent validation data will be prepared separately "
    "and must not be mixed into training."
)

print(
    "8. No classifier was trained in Step 05D."
)

print("=" * 100)