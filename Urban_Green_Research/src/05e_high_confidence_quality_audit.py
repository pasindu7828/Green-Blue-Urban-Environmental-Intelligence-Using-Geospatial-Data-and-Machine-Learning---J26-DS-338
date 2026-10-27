"""
STEP 05E - HIGH-CONFIDENCE LULC REFERENCE QUALITY AUDIT PREPARATION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Read the HIGH-confidence, unique-Landsat-pixel training candidate pool.
2. Select a small spatially distributed manual quality-audit subset.
3. Select 10 points per LULC class (40 total).
4. Keep the audit separate from final model accuracy validation.
5. Create fields needed for later manual visual checking.

Research LULC Classes:
1 - Green / Vegetation
2 - Built-up
3 - Open / Bare
4 - Water

IMPORTANT:
- This is a QUALITY AUDIT of the filtered reference-generation method.
- It is NOT the final Random Forest accuracy-validation dataset.
- The audit does NOT prove that every candidate is correct.
- HIGH confidence is automated evidence, not independent ground truth.
- No classifier is trained in this step.
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

TRAINING_POOL_GPKG = (
    REFERENCE_DIR
    / "lulc_high_confidence_training_candidates_2021.gpkg"
)


# =============================================================================
# 3. OUTPUTS
# =============================================================================

AUDIT_GPKG = (
    REFERENCE_DIR
    / "lulc_high_confidence_quality_audit_40_2021.gpkg"
)

AUDIT_CSV = (
    REFERENCE_DIR
    / "lulc_high_confidence_quality_audit_40_2021.csv"
)

AUDIT_SUMMARY_CSV = (
    REPORT_DIR
    / "05e_high_confidence_quality_audit_summary.csv"
)

PROCESS_SUMMARY_CSV = (
    REPORT_DIR
    / "05e_high_confidence_quality_audit_process_summary.csv"
)


# =============================================================================
# 4. CONSTANTS
# =============================================================================

ANALYSIS_CRS = "EPSG:32644"

POINTS_PER_CLASS = 10

EXPECTED_TOTAL = 40


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
    Remove only generated output files.
    """

    if path.exists():
        path.unlink()


def euclidean_distance_matrix(
    coordinates_a,
    coordinates_b,
):
    """
    Calculate Euclidean distances between two sets
    of projected coordinates.
    """

    difference = (
        coordinates_a[:, None, :]
        -
        coordinates_b[None, :, :]
    )

    return np.sqrt(
        np.sum(
            difference ** 2,
            axis=2,
        )
    )


# =============================================================================
# 6. DETERMINISTIC SPATIAL FARTHest-POINT SELECTION
# =============================================================================

def select_spatially_distributed_points(
    class_gdf: gpd.GeoDataFrame,
    n_points: int,
) -> gpd.GeoDataFrame:
    """
    Select spatially distributed points using deterministic
    farthest-point sampling.

    Procedure:
    1. Start with the point farthest from the class centroid.
    2. Repeatedly choose the point with the greatest distance
       to its nearest already-selected point.
    3. Use sample_id as deterministic tie-breaker.

    This avoids random sampling and makes the workflow reproducible.
    """

    if len(class_gdf) < n_points:

        raise ValueError(
            f"Requested {n_points} points, "
            f"but only {len(class_gdf)} are available."
        )


    working = (
        class_gdf
        .copy()
        .reset_index(
            drop=True
        )
    )


    coordinates = np.column_stack(
        [
            working.geometry.x.to_numpy(),
            working.geometry.y.to_numpy(),
        ]
    )


    centroid = np.mean(
        coordinates,
        axis=0,
    )


    centroid_distances = np.sqrt(
        np.sum(
            (
                coordinates
                -
                centroid
            )
            ** 2,
            axis=1,
        )
    )


    max_centroid_distance = np.max(
        centroid_distances
    )


    first_candidates = np.where(
        np.isclose(
            centroid_distances,
            max_centroid_distance,
        )
    )[0]


    if len(first_candidates) > 1:

        candidate_sample_ids = (
            working.iloc[
                first_candidates
            ][
                "sample_id"
            ]
            .astype(str)
            .to_numpy()
        )

        first_index = int(
            first_candidates[
                np.argmin(
                    candidate_sample_ids
                )
            ]
        )

    else:

        first_index = int(
            first_candidates[0]
        )


    selected_indices = [
        first_index
    ]


    while len(
        selected_indices
    ) < n_points:

        selected_coordinates = (
            coordinates[
                selected_indices
            ]
        )


        distances = (
            euclidean_distance_matrix(
                coordinates,
                selected_coordinates,
            )
        )


        nearest_selected_distance = (
            distances.min(
                axis=1
            )
        )


        nearest_selected_distance[
            selected_indices
        ] = -np.inf


        maximum_distance = np.max(
            nearest_selected_distance
        )


        candidate_indices = np.where(
            np.isclose(
                nearest_selected_distance,
                maximum_distance,
            )
        )[0]


        if len(
            candidate_indices
        ) > 1:

            candidate_sample_ids = (
                working.iloc[
                    candidate_indices
                ][
                    "sample_id"
                ]
                .astype(str)
                .to_numpy()
            )

            chosen_index = int(
                candidate_indices[
                    np.argmin(
                        candidate_sample_ids
                    )
                ]
            )

        else:

            chosen_index = int(
                candidate_indices[0]
            )


        selected_indices.append(
            chosen_index
        )


    selected = (
        working
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
# 7. NEAREST-NEIGHBOUR SUMMARY
# =============================================================================

def nearest_neighbour_summary(
    class_gdf: gpd.GeoDataFrame,
):
    """
    Calculate nearest-neighbour distances among the selected
    points of one class.
    """

    if len(
        class_gdf
    ) < 2:

        return (
            np.nan,
            np.nan,
            np.nan,
        )


    coordinates = np.column_stack(
        [
            class_gdf.geometry.x.to_numpy(),
            class_gdf.geometry.y.to_numpy(),
        ]
    )


    distances = (
        euclidean_distance_matrix(
            coordinates,
            coordinates,
        )
    )


    np.fill_diagonal(
        distances,
        np.inf,
    )


    nearest_distances = (
        distances.min(
            axis=1
        )
    )


    return (
        float(
            np.min(
                nearest_distances
            )
        ),
        float(
            np.mean(
                nearest_distances
            )
        ),
        float(
            np.max(
                nearest_distances
            )
        ),
    )


# =============================================================================
# 8. START
# =============================================================================

print("=" * 100)
print("STEP 05E - HIGH-CONFIDENCE LULC REFERENCE QUALITY AUDIT PREPARATION")
print("=" * 100)


if not TRAINING_POOL_GPKG.exists():

    raise FileNotFoundError(
        f"Step 05D training candidate pool not found:\n"
        f"{TRAINING_POOL_GPKG}"
    )


training_gdf = gpd.read_file(
    TRAINING_POOL_GPKG
)


print(
    f"\nInput training-candidate pool : "
    f"{relative_path(TRAINING_POOL_GPKG)}"
)

print(
    f"Input features                : "
    f"{len(training_gdf)}"
)

print(
    f"CRS                           : "
    f"{training_gdf.crs}"
)


# =============================================================================
# 9. INPUT VALIDATION
# =============================================================================

if str(
    training_gdf.crs
) != ANALYSIS_CRS:

    raise ValueError(
        f"Unexpected CRS: {training_gdf.crs}. "
        f"Expected {ANALYSIS_CRS}."
    )


required_columns = {
    "sample_id",
    "class_id",
    "class_name",
    "screen_confidence",
    "landsat_pixel_id",
    "wc_pixel_purity",
    "max_spectral_robust_z",
    "spectral_margin",
    "spectral_class_agreement",
}


missing_columns = (
    required_columns
    -
    set(
        training_gdf.columns
    )
)


if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )


if training_gdf[
    "sample_id"
].duplicated().any():

    raise ValueError(
        "Duplicate sample IDs exist "
        "in the Step 05D pool."
    )


if training_gdf[
    "landsat_pixel_id"
].duplicated().any():

    raise ValueError(
        "Duplicate Landsat pixels exist "
        "in the Step 05D pool."
    )


if not (
    training_gdf[
        "screen_confidence"
    ]
    .astype(str)
    .str.upper()
    .eq("HIGH")
    .all()
):

    raise ValueError(
        "Step 05D input contains "
        "non-HIGH-confidence candidates."
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
# 10. INPUT CLASS COUNTS
# =============================================================================

print("\n" + "=" * 100)
print("STEP 05D HIGH-CONFIDENCE CLASS COUNTS")
print("=" * 100)


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    class_count = int(
        (
            training_gdf[
                "class_id"
            ]
            == class_id
        ).sum()
    )


    print(
        f"{class_id} - "
        f"{class_name:<20}: "
        f"{class_count}"
    )


    if class_count < POINTS_PER_CLASS:

        raise ValueError(
            f"Class {class_id} has only "
            f"{class_count} candidates. "
            f"Cannot select {POINTS_PER_CLASS} "
            f"audit points."
        )


# =============================================================================
# 11. SELECT SPATIALLY DISTRIBUTED AUDIT POINTS
# =============================================================================

print("\n" + "=" * 100)
print("SELECTING SPATIALLY DISTRIBUTED 40-POINT AUDIT")
print("=" * 100)


selected_class_layers = []

summary_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    class_gdf = (
        training_gdf[
            training_gdf[
                "class_id"
            ]
            == class_id
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    selected = (
        select_spatially_distributed_points(
            class_gdf=class_gdf,
            n_points=POINTS_PER_CLASS,
        )
    )


    (
        minimum_nn_distance,
        mean_nn_distance,
        maximum_nn_distance,
    ) = nearest_neighbour_summary(
        selected
    )


    selected_class_layers.append(
        selected
    )


    summary_records.append(
        {
            "class_id": class_id,
            "class_name": class_name,
            "available_high_confidence_candidates": len(
                class_gdf
            ),
            "selected_audit_points": len(
                selected
            ),
            "minimum_nn_distance_m": minimum_nn_distance,
            "mean_nn_distance_m": mean_nn_distance,
            "maximum_nn_distance_m": maximum_nn_distance,
            "mean_wc_pixel_purity": float(
                selected[
                    "wc_pixel_purity"
                ].mean()
            ),
            "minimum_wc_pixel_purity": float(
                selected[
                    "wc_pixel_purity"
                ].min()
            ),
        }
    )


    print(
        f"\n{class_name}"
    )

    print(
        f"Available HIGH candidates : "
        f"{len(class_gdf)}"
    )

    print(
        f"Selected audit points     : "
        f"{len(selected)}"
    )

    print(
        f"Minimum NN distance       : "
        f"{minimum_nn_distance:.3f} m"
    )

    print(
        f"Mean NN distance          : "
        f"{mean_nn_distance:.3f} m"
    )


# =============================================================================
# 12. COMBINE AUDIT SUBSET
# =============================================================================

audit_gdf = gpd.GeoDataFrame(
    pd.concat(
        selected_class_layers,
        ignore_index=True,
    ),
    geometry="geometry",
    crs=ANALYSIS_CRS,
)


audit_gdf = (
    audit_gdf
    .sort_values(
        by=[
            "class_id",
            "sample_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 13. ADD MANUAL-AUDIT FIELDS
# =============================================================================

audit_gdf[
    "audit_role"
] = (
    "REFERENCE_QUALITY_AUDIT"
)


audit_gdf[
    "review_status"
] = (
    "PENDING"
)


audit_gdf[
    "manual_class_id"
] = pd.Series(
    [pd.NA] * len(
        audit_gdf
    ),
    dtype="Int64",
)


audit_gdf[
    "manual_class_name"
] = ""


audit_gdf[
    "imagery_source"
] = ""


audit_gdf[
    "imagery_date"
] = ""


audit_gdf[
    "temporal_match"
] = (
    "NOT_CHECKED"
)


audit_gdf[
    "review_notes"
] = ""


audit_gdf[
    "training_eligibility"
] = (
    "NOT_DECIDED"
)


audit_gdf[
    "validation_eligibility"
] = (
    "NOT_FOR_MODEL_VALIDATION"
)


# =============================================================================
# 14. FINAL AUDIT VALIDATION
# =============================================================================

audit_count = len(
    audit_gdf
)


class_counts = (
    audit_gdf[
        "class_id"
    ]
    .value_counts()
    .sort_index()
)


all_class_counts_correct = True


for class_id in (
    CLASS_DEFINITIONS.keys()
):

    count = int(
        class_counts.get(
            class_id,
            0
        )
    )

    if count != POINTS_PER_CLASS:

        all_class_counts_correct = False


duplicate_sample_ids = int(
    audit_gdf[
        "sample_id"
    ]
    .duplicated()
    .sum()
)


duplicate_landsat_pixels = int(
    audit_gdf[
        "landsat_pixel_id"
    ]
    .duplicated()
    .sum()
)


all_high_confidence = bool(
    audit_gdf[
        "screen_confidence"
    ]
    .astype(str)
    .str.upper()
    .eq("HIGH")
    .all()
)


overall_pass = all(
    [
        audit_count == EXPECTED_TOTAL,
        all_class_counts_correct,
        duplicate_sample_ids == 0,
        duplicate_landsat_pixels == 0,
        all_high_confidence,
    ]
)


# =============================================================================
# 15. SUMMARY TABLE
# =============================================================================

summary_df = pd.DataFrame(
    summary_records
)


print("\n" + "=" * 100)
print("FINAL 40-POINT AUDIT DISTRIBUTION")
print("=" * 100)


print(
    summary_df.to_string(
        index=False
    )
)


# =============================================================================
# 16. LIST SELECTED SAMPLE IDS
# =============================================================================

print("\n" + "=" * 100)
print("SELECTED AUDIT SAMPLE IDS")
print("=" * 100)


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    sample_ids = (
        audit_gdf[
            audit_gdf[
                "class_id"
            ]
            == class_id
        ][
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )


    print(
        f"\n{class_id} - {class_name}:"
    )

    print(
        ", ".join(
            sample_ids
        )
    )


# =============================================================================
# 17. SAVE OUTPUTS
# =============================================================================

remove_existing_file(
    AUDIT_GPKG
)


audit_gdf.to_file(
    AUDIT_GPKG,
    layer="high_confidence_quality_audit_40",
    driver="GPKG",
)


audit_gdf.drop(
    columns="geometry"
).to_csv(
    AUDIT_CSV,
    index=False,
)


summary_df.to_csv(
    AUDIT_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 18. PROCESS SUMMARY
# =============================================================================

process_summary_df = pd.DataFrame(
    [
        {
            "input_high_confidence_training_candidates": len(
                training_gdf
            ),
            "audit_points_per_class": POINTS_PER_CLASS,
            "total_audit_points": audit_count,
            "all_class_counts_correct": all_class_counts_correct,
            "duplicate_sample_ids": duplicate_sample_ids,
            "duplicate_landsat_pixels": duplicate_landsat_pixels,
            "all_audit_points_high_confidence": all_high_confidence,
            "audit_type": (
                "REFERENCE_QUALITY_AUDIT"
            ),
            "final_model_validation_dataset": False,
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
    PROCESS_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 19. FINAL TERMINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("STEP 05E SUMMARY")
print("=" * 100)


print(
    f"Input HIGH-confidence pool     : "
    f"{len(training_gdf)}"
)

print(
    f"Audit points per class         : "
    f"{POINTS_PER_CLASS}"
)

print(
    f"Final audit subset             : "
    f"{audit_count}"
)

print(
    f"All class counts correct       : "
    f"{all_class_counts_correct}"
)

print(
    f"Duplicate sample IDs           : "
    f"{duplicate_sample_ids}"
)

print(
    f"Duplicate Landsat pixels       : "
    f"{duplicate_landsat_pixels}"
)

print(
    f"All points HIGH confidence     : "
    f"{all_high_confidence}"
)

print(
    f"Overall Step 05E status        : "
    f"{'PASS' if overall_pass else 'REVIEW REQUIRED'}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(AUDIT_GPKG)}"
)

print(
    f" - {relative_path(AUDIT_CSV)}"
)

print(
    f" - {relative_path(AUDIT_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(PROCESS_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. This 40-point subset is a manual reference-quality audit."
)

print(
    "2. It was selected only from HIGH-confidence Step 05D candidates."
)

print(
    "3. It is not the final model accuracy-validation dataset."
)

print(
    "4. Do not manually edit the generated GeoPackage directly."
)

print(
    "5. Create a separate editable QGIS copy before manual review."
)

print(
    "6. Prefer historical high-resolution imagery close to 2021 "
    "when performing visual review."
)

print(
    "7. If exact 2021 imagery is unavailable, record the imagery "
    "source/date and temporal uncertainty instead of assuming a match."
)

print(
    "8. Rejected or corrected audit points will be handled in a "
    "later audit-evaluation step."
)

print("=" * 100)