"""
STEP 05B - MANUAL LULC QUALITY-AUDIT SUBSET SELECTION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Read the current 100-point manual-validation review layer.
2. Reduce the manual inspection workload to a practical 40-point subset.
3. Select 10 spatially distributed points from each LULC class.
4. Preserve already completed manual reviews such as C1_001 and C1_003.
5. Create a separate generated quality-audit subset for QGIS review.

Classes:
1 - Green / Vegetation
2 - Built-up
3 - Open / Bare
4 - Water

IMPORTANT:
- The 40 points are a MANUAL REFERENCE-QUALITY AUDIT subset.
- They are not claimed to be the final model accuracy-validation sample.
- The existing 100-point review GeoPackage is never modified.
- Already completed manual reviews are preserved.
- Temporal stability remains a separate later assessment.
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
# 2. INPUT / OUTPUT FILES
# =============================================================================

# Current editable 100-point review layer.
# C1_001 and C1_003 should already be saved here.
INPUT_GPKG = (
    REFERENCE_DIR
    / "lulc_manual_validation_review_2021.gpkg"
)


# Generated 40-point quality-audit subset.
# Do NOT manually edit this generated file.
OUTPUT_GPKG = (
    REFERENCE_DIR
    / "lulc_manual_quality_audit_subset_40_2021.gpkg"
)

OUTPUT_CSV = (
    REFERENCE_DIR
    / "lulc_manual_quality_audit_subset_40_2021.csv"
)

SUMMARY_REPORT = (
    REPORT_DIR
    / "05b_manual_quality_audit_subset_40_summary.csv"
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

ANALYSIS_CRS = "EPSG:32644"

SAMPLES_PER_CLASS = 10

EXPECTED_TOTAL = 40


CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}


# These are the two points already manually checked.
REQUIRED_REVIEWED_SAMPLES = {
    "C1_001",
    "C1_003",
}


COMPLETED_REVIEW_STATUSES = {
    "VERIFIED",
    "CORRECTED",
    "REJECTED",
}


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def relative_path(path: Path) -> str:
    """
    Return project-relative path where possible.
    """

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def remove_existing_file(path: Path):
    """
    Remove only an existing generated output file.
    """

    if path.exists():
        path.unlink()


def normalized_status_series(
    series: pd.Series
) -> pd.Series:
    """
    Normalize manual-review status values.
    """

    return (
        series
        .astype("string")
        .fillna("PENDING")
        .str.strip()
        .str.upper()
    )


def nearest_neighbour_distances(
    gdf: gpd.GeoDataFrame
) -> np.ndarray:
    """
    Calculate nearest-neighbour distance for each point.
    """

    coordinates = np.column_stack(
        [
            gdf.geometry.x.to_numpy(),
            gdf.geometry.y.to_numpy(),
        ]
    )


    if len(coordinates) < 2:
        return np.array(
            [np.nan]
        )


    differences = (
        coordinates[:, None, :]
        -
        coordinates[None, :, :]
    )


    distances = np.sqrt(
        np.sum(
            differences ** 2,
            axis=2
        )
    )


    np.fill_diagonal(
        distances,
        np.inf
    )


    return distances.min(
        axis=1
    )


def select_spatially_spread_points(
    class_gdf: gpd.GeoDataFrame,
    n_samples: int,
) -> gpd.GeoDataFrame:
    """
    Select spatially distributed points using deterministic
    farthest-point sampling.

    Any already manually reviewed samples are preserved first.
    Remaining samples are selected to maximize spatial spread.
    """

    class_gdf = (
        class_gdf
        .sort_values(
            "sample_id"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )


    if len(class_gdf) < n_samples:

        raise ValueError(
            f"Requested {n_samples} samples, "
            f"but only {len(class_gdf)} are available."
        )


    review_status = (
        normalized_status_series(
            class_gdf[
                "review_status"
            ]
        )
    )


    reviewed_indices = (
        class_gdf.index[
            review_status.isin(
                COMPLETED_REVIEW_STATUSES
            )
        ]
        .tolist()
    )


    if len(reviewed_indices) > n_samples:

        raise ValueError(
            f"This class already contains "
            f"{len(reviewed_indices)} manually reviewed points, "
            f"which is greater than the target of {n_samples}. "
            f"Do not continue until this is reviewed."
        )


    coordinates = np.column_stack(
        [
            class_gdf.geometry.x.to_numpy(),
            class_gdf.geometry.y.to_numpy(),
        ]
    )


    selected_indices = list(
        reviewed_indices
    )


    # -------------------------------------------------------------------------
    # If no manually reviewed point exists in this class,
    # start from the candidate nearest the class spatial centre.
    # -------------------------------------------------------------------------

    if len(selected_indices) == 0:

        centre = np.mean(
            coordinates,
            axis=0
        )


        distances_to_centre = np.sqrt(
            np.sum(
                (
                    coordinates
                    -
                    centre
                )
                ** 2,
                axis=1
            )
        )


        first_index = int(
            np.argmin(
                distances_to_centre
            )
        )


        selected_indices.append(
            first_index
        )


    # -------------------------------------------------------------------------
    # Farthest-point sampling
    # -------------------------------------------------------------------------

    while len(
        selected_indices
    ) < n_samples:

        selected_coordinates = (
            coordinates[
                selected_indices
            ]
        )


        differences = (
            coordinates[:, None, :]
            -
            selected_coordinates[
                None, :, :
            ]
        )


        distances = np.sqrt(
            np.sum(
                differences ** 2,
                axis=2
            )
        )


        minimum_distance = (
            distances.min(
                axis=1
            )
        )


        # Prevent already-selected rows from being selected again.
        minimum_distance[
            selected_indices
        ] = -1


        next_index = int(
            np.argmax(
                minimum_distance
            )
        )


        selected_indices.append(
            next_index
        )


    selected = (
        class_gdf
        .iloc[
            selected_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    return selected


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 05B - MANUAL LULC QUALITY-AUDIT SUBSET SELECTION")
print("=" * 100)


if not INPUT_GPKG.exists():

    raise FileNotFoundError(
        f"Input 100-point review GeoPackage not found:\n"
        f"{INPUT_GPKG}"
    )


review_gdf = gpd.read_file(
    INPUT_GPKG
)


print(
    f"\nInput review layer : "
    f"{relative_path(INPUT_GPKG)}"
)

print(
    f"Input features     : "
    f"{len(review_gdf)}"
)

print(
    f"CRS                : "
    f"{review_gdf.crs}"
)


# =============================================================================
# 6. INPUT VALIDATION
# =============================================================================

if str(
    review_gdf.crs
) != ANALYSIS_CRS:

    raise ValueError(
        f"Unexpected CRS: "
        f"{review_gdf.crs}. "
        f"Expected {ANALYSIS_CRS}."
    )


if len(
    review_gdf
) != 100:

    raise ValueError(
        f"Expected the current 100-point review layer, "
        f"but found {len(review_gdf)} features."
    )


required_columns = {
    "sample_id",
    "class_id",
    "class_name",
    "review_status",
    "manual_class_id",
    "manual_class_name",
    "temporal_stability",
    "training_eligible",
    "validation_eligible",
    "review_notes",
}


missing_columns = (
    required_columns
    -
    set(
        review_gdf.columns
    )
)


if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )


if review_gdf[
    "sample_id"
].duplicated().any():

    raise ValueError(
        "Duplicate sample_id values exist "
        "in the input review layer."
    )


review_gdf[
    "class_id"
] = pd.to_numeric(
    review_gdf[
        "class_id"
    ],
    errors="raise",
).astype(int)


# =============================================================================
# 7. VERIFY SAVED MANUAL REVIEWS
# =============================================================================

print("\n" + "=" * 100)
print("CHECKING EXISTING MANUAL REVIEWS")
print("=" * 100)


status_normalized = (
    normalized_status_series(
        review_gdf[
            "review_status"
        ]
    )
)


reviewed_mask = (
    status_normalized.isin(
        COMPLETED_REVIEW_STATUSES
    )
)


reviewed_gdf = (
    review_gdf.loc[
        reviewed_mask
    ]
    .copy()
)


print(
    f"Already reviewed points : "
    f"{len(reviewed_gdf)}"
)


if len(reviewed_gdf) > 0:

    print(
        reviewed_gdf[
            [
                "sample_id",
                "class_id",
                "class_name",
                "review_status",
                "manual_class_id",
                "manual_class_name",
            ]
        ]
        .sort_values(
            "sample_id"
        )
        .to_string(
            index=False
        )
    )


for required_sample in sorted(
    REQUIRED_REVIEWED_SAMPLES
):

    record = (
        review_gdf[
            review_gdf[
                "sample_id"
            ]
            == required_sample
        ]
    )


    if len(record) != 1:

        raise ValueError(
            f"Required reviewed sample "
            f"{required_sample} was not found exactly once."
        )


    status = str(
        record[
            "review_status"
        ].iloc[0]
    ).strip().upper()


    if status not in COMPLETED_REVIEW_STATUSES:

        raise ValueError(
            f"{required_sample} is currently '{status}', "
            f"not a completed manual review.\n"
            f"Go back to QGIS, save the layer edits, "
            f"then run this script again."
        )


    print(
        f"{required_sample}: "
        f"{status} -> preserved"
    )


# =============================================================================
# 8. CLASS COUNTS IN CURRENT 100-POINT LAYER
# =============================================================================

print("\n" + "=" * 100)
print("CURRENT 100-POINT CLASS COUNTS")
print("=" * 100)


current_class_counts = (
    review_gdf[
        "class_id"
    ]
    .value_counts()
    .sort_index()
)


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    available = int(
        current_class_counts.get(
            class_id,
            0
        )
    )


    print(
        f"{class_id} - {class_name}: "
        f"{available}"
    )


    if available < SAMPLES_PER_CLASS:

        raise ValueError(
            f"Not enough points for "
            f"{class_name}. "
            f"Available={available}, "
            f"required={SAMPLES_PER_CLASS}."
        )


# =============================================================================
# 9. SELECT 10 POINTS PER CLASS
# =============================================================================

print("\n" + "=" * 100)
print("SELECTING 40-POINT QUALITY-AUDIT SUBSET")
print("=" * 100)


selected_frames = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    class_gdf = (
        review_gdf[
            review_gdf[
                "class_id"
            ]
            == class_id
        ]
        .copy()
    )


    class_status = (
        normalized_status_series(
            class_gdf[
                "review_status"
            ]
        )
    )


    reviewed_in_class = int(
        class_status.isin(
            COMPLETED_REVIEW_STATUSES
        ).sum()
    )


    selected_class = (
        select_spatially_spread_points(
            class_gdf,
            SAMPLES_PER_CLASS,
        )
    )


    selected_class[
        "subset_role"
    ] = "MANUAL_QUALITY_AUDIT"


    selected_class[
        "subset_target_per_class"
    ] = SAMPLES_PER_CLASS


    selected_class[
        "subset_selection"
    ] = (
        "preserve_reviewed_then_farthest_point"
    )


    selected_frames.append(
        selected_class
    )


    print(
        f"\n{class_name}"
    )

    print(
        f"Available in 100-point layer : "
        f"{len(class_gdf)}"
    )

    print(
        f"Already reviewed preserved   : "
        f"{reviewed_in_class}"
    )

    print(
        f"Selected for audit           : "
        f"{len(selected_class)}"
    )


# =============================================================================
# 10. COMBINE SELECTED SUBSET
# =============================================================================

audit_gdf = pd.concat(
    selected_frames,
    ignore_index=True,
)


audit_gdf = gpd.GeoDataFrame(
    audit_gdf,
    geometry="geometry",
    crs=ANALYSIS_CRS,
)


audit_gdf = (
    audit_gdf
    .sort_values(
        [
            "class_id",
            "sample_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


audit_gdf[
    "audit_order"
] = np.arange(
    1,
    len(audit_gdf) + 1
)


# =============================================================================
# 11. SPATIAL DISTRIBUTION METRICS
# =============================================================================

audit_gdf[
    "audit_nn_distance_m"
] = np.nan


summary_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    class_mask = (
        audit_gdf[
            "class_id"
        ]
        == class_id
    )


    class_subset = (
        audit_gdf.loc[
            class_mask
        ]
        .copy()
    )


    nearest_distances = (
        nearest_neighbour_distances(
            class_subset
        )
    )


    audit_gdf.loc[
        class_mask,
        "audit_nn_distance_m",
    ] = nearest_distances


    review_status = (
        normalized_status_series(
            class_subset[
                "review_status"
            ]
        )
    )


    already_reviewed = int(
        review_status.isin(
            COMPLETED_REVIEW_STATUSES
        ).sum()
    )


    summary_records.append(
        {
            "class_id": (
                class_id
            ),
            "class_name": (
                class_name
            ),
            "input_100_count": int(
                (
                    review_gdf[
                        "class_id"
                    ]
                    == class_id
                ).sum()
            ),
            "selected_audit_points": (
                len(class_subset)
            ),
            "already_reviewed_preserved": (
                already_reviewed
            ),
            "remaining_to_review": (
                len(class_subset)
                - already_reviewed
            ),
            "minimum_nn_distance_m": float(
                np.nanmin(
                    nearest_distances
                )
            ),
            "mean_nn_distance_m": float(
                np.nanmean(
                    nearest_distances
                )
            ),
            "maximum_nn_distance_m": float(
                np.nanmax(
                    nearest_distances
                )
            ),
        }
    )


summary_df = pd.DataFrame(
    summary_records
)


# =============================================================================
# 12. FINAL QUALITY CHECKS
# =============================================================================

total_correct = (
    len(
        audit_gdf
    )
    == EXPECTED_TOTAL
)


class_counts = (
    audit_gdf[
        "class_id"
    ]
    .value_counts()
    .to_dict()
)


all_class_counts_correct = all(
    class_counts.get(
        class_id,
        0
    )
    ==
    SAMPLES_PER_CLASS

    for class_id
    in CLASS_DEFINITIONS
)


duplicate_ids = int(
    audit_gdf[
        "sample_id"
    ]
    .duplicated()
    .sum()
)


required_samples_included = (
    REQUIRED_REVIEWED_SAMPLES
    .issubset(
        set(
            audit_gdf[
                "sample_id"
            ]
        )
    )
)


audit_status = (
    normalized_status_series(
        audit_gdf[
            "review_status"
        ]
    )
)


already_reviewed_total = int(
    audit_status.isin(
        COMPLETED_REVIEW_STATUSES
    ).sum()
)


remaining_manual_checks = (
    len(audit_gdf)
    -
    already_reviewed_total
)


overall_pass = all(
    [
        total_correct,
        all_class_counts_correct,
        duplicate_ids == 0,
        required_samples_included,
    ]
)


# =============================================================================
# 13. SAVE GENERATED OUTPUTS
# =============================================================================

remove_existing_file(
    OUTPUT_GPKG
)


audit_gdf.to_file(
    OUTPUT_GPKG,
    layer="manual_quality_audit_40",
    driver="GPKG",
)


audit_csv_df = (
    audit_gdf
    .drop(
        columns="geometry"
    )
    .copy()
)


audit_csv_df.to_csv(
    OUTPUT_CSV,
    index=False,
)


summary_df.to_csv(
    SUMMARY_REPORT,
    index=False,
)


# =============================================================================
# 14. PRINT CLASS COUNTS
# =============================================================================

print("\n" + "=" * 100)
print("FINAL 40-POINT AUDIT COUNTS")
print("=" * 100)


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    count = int(
        (
            audit_gdf[
                "class_id"
            ]
            == class_id
        ).sum()
    )


    print(
        f"{class_id} - "
        f"{class_name}: "
        f"{count}"
    )


print("\nSpatial distribution summary:")

print(
    summary_df.to_string(
        index=False
    )
)


# =============================================================================
# 15. VERIFY PRESERVED REVIEW RECORDS
# =============================================================================

print("\n" + "=" * 100)
print("PRESERVED MANUAL REVIEWS")
print("=" * 100)


preserved_records = (
    audit_gdf[
        audit_gdf[
            "sample_id"
        ].isin(
            REQUIRED_REVIEWED_SAMPLES
        )
    ]
    [
        [
            "sample_id",
            "class_id",
            "class_name",
            "review_status",
            "manual_class_id",
            "manual_class_name",
            "review_notes",
        ]
    ]
    .sort_values(
        "sample_id"
    )
)


print(
    preserved_records.to_string(
        index=False
    )
)


# =============================================================================
# 16. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("STEP 05B SUMMARY")
print("=" * 100)


print(
    f"Input 100-point review layer : "
    f"{len(review_gdf)}"
)

print(
    f"Final audit subset           : "
    f"{len(audit_gdf)}"
)

print(
    f"Points per class             : "
    f"{SAMPLES_PER_CLASS}"
)

print(
    f"Already reviewed preserved   : "
    f"{already_reviewed_total}"
)

print(
    f"Remaining manual checks      : "
    f"{remaining_manual_checks}"
)

print(
    f"C1_001 and C1_003 included   : "
    f"{required_samples_included}"
)

print(
    f"Duplicate sample IDs         : "
    f"{duplicate_ids}"
)

print(
    f"All class counts correct     : "
    f"{all_class_counts_correct}"
)

print(
    f"Overall subset status        : "
    f"{'PASS' if overall_pass else 'REVIEW REQUIRED'}"
)


print("\nGenerated outputs:")

print(
    f" - "
    f"{relative_path(OUTPUT_GPKG)}"
)

print(
    f" - "
    f"{relative_path(OUTPUT_CSV)}"
)

print(
    f" - "
    f"{relative_path(SUMMARY_REPORT)}"
)


print("\nIMPORTANT:")

print(
    "1. This 40-point set is a manual reference-quality audit subset."
)

print(
    "2. It is not yet the final model accuracy-validation dataset."
)

print(
    "3. Already completed reviews are preserved from the 100-point layer."
)

print(
    "4. The generated GeoPackage should not be manually edited directly."
)

print(
    "5. Create a separate editable QGIS copy before continuing manual review."
)

print(
    "6. Temporal stability remains NOT_CHECKED until a later dedicated step."
)

print("=" * 100)