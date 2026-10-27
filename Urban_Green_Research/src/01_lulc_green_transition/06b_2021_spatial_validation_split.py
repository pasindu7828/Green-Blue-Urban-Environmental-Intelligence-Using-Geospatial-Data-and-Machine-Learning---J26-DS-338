"""
STEP 06B - 2021 SPATIALLY SAFE TRAIN / VALIDATION SPLIT

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Read the 2021 LULC model feature table from Step 06A.
2. Divide samples into fixed 1 km spatial blocks.
3. Reserve complete spatial blocks for validation.
4. Keep approximately 80% of samples for training and 20% for validation.
5. Ensure every LULC class occurs in both subsets.
6. Ensure every validation class occurs across at least TWO spatial blocks.
7. Export reproducible split assignments and QA reports.

IMPORTANT:
- Individual samples are NOT randomly divided.
- Complete spatial blocks remain together.
- Validation data must remain untouched during model tuning.
- Coordinates are used only for spatial splitting and QA.
- Coordinates are NOT model predictors.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.model_selection import GroupShuffleSplit


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lulc"
    / "model_data"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

MODEL_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. INPUT
# =============================================================================

INPUT_FEATURE_CSV = (
    MODEL_DATA_DIR
    / "lulc_2021_model_features.csv"
)


# =============================================================================
# 3. OUTPUTS
# =============================================================================

SPLIT_CSV = (
    MODEL_DATA_DIR
    / "lulc_2021_spatial_train_validation_split.csv"
)

TRAIN_CSV = (
    MODEL_DATA_DIR
    / "lulc_2021_training_features.csv"
)

VALIDATION_CSV = (
    MODEL_DATA_DIR
    / "lulc_2021_validation_features.csv"
)

SPLIT_GPKG = (
    MODEL_DATA_DIR
    / "lulc_2021_spatial_split_points.gpkg"
)

SPLIT_SUMMARY_CSV = (
    REPORT_DIR
    / "06b_2021_spatial_split_summary.csv"
)

CLASS_SUMMARY_CSV = (
    REPORT_DIR
    / "06b_2021_spatial_split_class_summary.csv"
)

BLOCK_SUMMARY_CSV = (
    REPORT_DIR
    / "06b_2021_spatial_block_summary.csv"
)

CANDIDATE_SEARCH_CSV = (
    REPORT_DIR
    / "06b_2021_spatial_split_candidate_search.csv"
)


# =============================================================================
# 4. CONSTANTS
# =============================================================================

ANALYSIS_CRS = "EPSG:32644"

EXPECTED_SAMPLE_COUNT = 397

SPATIAL_BLOCK_SIZE_M = 1000

TARGET_VALIDATION_FRACTION = 0.20
MIN_VALIDATION_FRACTION = 0.15
MAX_VALIDATION_FRACTION = 0.25

MIN_VALIDATION_SAMPLES_PER_CLASS = 3
MIN_TRAINING_SAMPLES_PER_CLASS = 10

# NEW IMPORTANT CONSTRAINT
MIN_VALIDATION_BLOCKS_PER_CLASS = 2

N_CANDIDATE_SPLITS = 10000

RANDOM_STATE = 2026


CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}


MODEL_FEATURES = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "ndvi",
    "ndbi",
    "mndwi",
    "bsi",
]


# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================

def relative_path(path: Path) -> str:
    """Return project-relative path when possible."""

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def class_counts(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """Return sample counts for every expected LULC class."""

    return (
        dataframe[
            "class_id"
        ]
        .value_counts()
        .reindex(
            CLASS_DEFINITIONS.keys(),
            fill_value=0,
        )
        .astype(int)
    )


def class_proportions(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """Return class proportions."""

    counts = class_counts(
        dataframe
    )

    total = int(
        counts.sum()
    )

    if total == 0:
        return pd.Series(
            0.0,
            index=counts.index,
        )

    return counts / total


def class_block_counts(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Return number of unique spatial blocks occupied by each class.
    """

    results = {}

    for class_id in CLASS_DEFINITIONS:

        results[
            class_id
        ] = int(
            dataframe.loc[
                dataframe[
                    "class_id"
                ]
                == class_id,
                "spatial_block_id",
            ].nunique()
        )

    return pd.Series(
        results,
        dtype=int,
    )


def minimum_cross_split_distance(
    training_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> float:
    """
    Minimum Euclidean point-to-point distance between
    training and validation samples.

    Diagnostic only:
    neighbouring 1 km blocks can share boundaries,
    so this distance does NOT have to equal 1 km.
    """

    train_coordinates = (
        training_df[
            [
                "easting",
                "northing",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    validation_coordinates = (
        validation_df[
            [
                "easting",
                "northing",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    differences = (
        train_coordinates[
            :,
            np.newaxis,
            :
        ]
        -
        validation_coordinates[
            np.newaxis,
            :,
            :
        ]
    )

    squared_distances = np.sum(
        differences ** 2,
        axis=2,
    )

    return float(
        np.sqrt(
            np.min(
                squared_distances
            )
        )
    )


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 06B - 2021 SPATIALLY SAFE TRAIN / VALIDATION SPLIT")
print("=" * 100)


if not INPUT_FEATURE_CSV.exists():

    raise FileNotFoundError(
        f"Step 06A feature table not found:\n"
        f"{INPUT_FEATURE_CSV}"
    )


# =============================================================================
# 7. LOAD FEATURE TABLE
# =============================================================================

feature_df = pd.read_csv(
    INPUT_FEATURE_CSV
)


print(
    f"\nInput feature table : "
    f"{relative_path(INPUT_FEATURE_CSV)}"
)

print(
    f"Input samples       : "
    f"{len(feature_df)}"
)


# =============================================================================
# 8. VALIDATE INPUT
# =============================================================================

required_columns = {
    "sample_id",
    "class_id",
    "class_name",
    "easting",
    "northing",
    "landsat_row",
    "landsat_col",
    "landsat_pixel_id",
    *MODEL_FEATURES,
}


missing_columns = (
    required_columns
    -
    set(
        feature_df.columns
    )
)


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        f"{sorted(missing_columns)}"
    )


if len(
    feature_df
) != EXPECTED_SAMPLE_COUNT:

    raise ValueError(
        f"Expected {EXPECTED_SAMPLE_COUNT} samples, "
        f"but found {len(feature_df)}."
    )


feature_df[
    "class_id"
] = pd.to_numeric(
    feature_df[
        "class_id"
    ],
    errors="raise",
).astype(int)


feature_df[
    "easting"
] = pd.to_numeric(
    feature_df[
        "easting"
    ],
    errors="raise",
)


feature_df[
    "northing"
] = pd.to_numeric(
    feature_df[
        "northing"
    ],
    errors="raise",
)


if feature_df[
    "sample_id"
].duplicated().any():

    raise ValueError(
        "Duplicate sample IDs exist."
    )


if feature_df[
    "landsat_pixel_id"
].duplicated().any():

    raise ValueError(
        "Duplicate Landsat pixel IDs exist."
    )


if feature_df[
    MODEL_FEATURES
].isna().any().any():

    raise ValueError(
        "Missing model feature values exist."
    )


if set(
    feature_df[
        "class_id"
    ].unique()
) != set(
    CLASS_DEFINITIONS.keys()
):

    raise ValueError(
        "Expected all four LULC classes."
    )


# =============================================================================
# 9. CREATE FIXED 1 KM SPATIAL BLOCKS
# =============================================================================

print("\n" + "=" * 100)
print("CREATING 1 KM SPATIAL BLOCKS")
print("=" * 100)


feature_df[
    "spatial_block_x"
] = np.floor(
    feature_df[
        "easting"
    ]
    /
    SPATIAL_BLOCK_SIZE_M
).astype(int)


feature_df[
    "spatial_block_y"
] = np.floor(
    feature_df[
        "northing"
    ]
    /
    SPATIAL_BLOCK_SIZE_M
).astype(int)


feature_df[
    "spatial_block_id"
] = (
    "B"
    +
    str(
        SPATIAL_BLOCK_SIZE_M
    )
    +
    "_E"
    +
    feature_df[
        "spatial_block_x"
    ].astype(str)
    +
    "_N"
    +
    feature_df[
        "spatial_block_y"
    ].astype(str)
)


total_blocks = int(
    feature_df[
        "spatial_block_id"
    ].nunique()
)


print(
    f"Spatial block size : "
    f"{SPATIAL_BLOCK_SIZE_M} m x "
    f"{SPATIAL_BLOCK_SIZE_M} m"
)

print(
    f"Total blocks       : "
    f"{total_blocks}"
)


# =============================================================================
# 10. CLASS COVERAGE ACROSS ALL BLOCKS
# =============================================================================

print("\n" + "=" * 100)
print("CLASS COVERAGE ACROSS SPATIAL BLOCKS")
print("=" * 100)


overall_class_block_counts = class_block_counts(
    feature_df
)


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    sample_count = int(
        (
            feature_df[
                "class_id"
            ]
            == class_id
        ).sum()
    )

    block_count = int(
        overall_class_block_counts.loc[
            class_id
        ]
    )

    print(
        f"{class_id} - "
        f"{class_name:<20}: "
        f"{sample_count:>3} samples across "
        f"{block_count:>3} blocks"
    )


# =============================================================================
# 11. SEARCH FOR SPATIALLY ADEQUATE HOLDOUT
# =============================================================================

print("\n" + "=" * 100)
print("SEARCHING FOR A REPRODUCIBLE SPATIAL HOLDOUT")
print("=" * 100)


print(
    f"Minimum validation samples/class : "
    f"{MIN_VALIDATION_SAMPLES_PER_CLASS}"
)

print(
    f"Minimum validation blocks/class  : "
    f"{MIN_VALIDATION_BLOCKS_PER_CLASS}"
)


overall_proportions = class_proportions(
    feature_df
)


splitter = GroupShuffleSplit(
    n_splits=N_CANDIDATE_SPLITS,
    test_size=TARGET_VALIDATION_FRACTION,
    random_state=RANDOM_STATE,
)


candidate_records = []

best_candidate = None


for candidate_number, (
    training_indices,
    validation_indices,
) in enumerate(
    splitter.split(
        feature_df,
        y=feature_df[
            "class_id"
        ],
        groups=feature_df[
            "spatial_block_id"
        ],
    ),
    start=1,
):

    training_candidate = (
        feature_df.iloc[
            training_indices
        ]
    )

    validation_candidate = (
        feature_df.iloc[
            validation_indices
        ]
    )


    validation_fraction = (
        len(
            validation_candidate
        )
        /
        len(
            feature_df
        )
    )


    training_counts = class_counts(
        training_candidate
    )

    validation_counts = class_counts(
        validation_candidate
    )


    training_block_counts = class_block_counts(
        training_candidate
    )

    validation_block_counts = class_block_counts(
        validation_candidate
    )


    training_blocks = set(
        training_candidate[
            "spatial_block_id"
        ]
    )

    validation_blocks = set(
        validation_candidate[
            "spatial_block_id"
        ]
    )


    shared_blocks = (
        training_blocks
        &
        validation_blocks
    )


    enough_training_samples = bool(
        (
            training_counts
            >= MIN_TRAINING_SAMPLES_PER_CLASS
        ).all()
    )


    enough_validation_samples = bool(
        (
            validation_counts
            >= MIN_VALIDATION_SAMPLES_PER_CLASS
        ).all()
    )


    enough_validation_blocks = bool(
        (
            validation_block_counts
            >= MIN_VALIDATION_BLOCKS_PER_CLASS
        ).all()
    )


    validation_size_acceptable = bool(
        MIN_VALIDATION_FRACTION
        <= validation_fraction
        <= MAX_VALIDATION_FRACTION
    )


    no_shared_blocks = (
        len(
            shared_blocks
        )
        == 0
    )


    training_proportions = class_proportions(
        training_candidate
    )

    validation_proportions = class_proportions(
        validation_candidate
    )


    validation_balance_error = float(
        np.mean(
            np.abs(
                validation_proportions.to_numpy()
                -
                overall_proportions.to_numpy()
            )
        )
    )


    training_balance_error = float(
        np.mean(
            np.abs(
                training_proportions.to_numpy()
                -
                overall_proportions.to_numpy()
            )
        )
    )


    validation_size_error = abs(
        validation_fraction
        -
        TARGET_VALIDATION_FRACTION
    )


    score = float(
        (
            2.0
            *
            validation_size_error
        )
        +
        validation_balance_error
        +
        (
            0.5
            *
            training_balance_error
        )
    )


    candidate_is_valid = all(
        [
            enough_training_samples,
            enough_validation_samples,
            enough_validation_blocks,
            validation_size_acceptable,
            no_shared_blocks,
        ]
    )


    candidate_record = {
        "candidate_number": (
            candidate_number
        ),
        "training_samples": (
            len(
                training_candidate
            )
        ),
        "validation_samples": (
            len(
                validation_candidate
            )
        ),
        "validation_fraction": (
            validation_fraction
        ),
        "training_blocks": (
            len(
                training_blocks
            )
        ),
        "validation_blocks": (
            len(
                validation_blocks
            )
        ),
        "shared_blocks": (
            len(
                shared_blocks
            )
        ),
        "validation_size_error": (
            validation_size_error
        ),
        "validation_balance_error": (
            validation_balance_error
        ),
        "training_balance_error": (
            training_balance_error
        ),
        "score": (
            score
        ),
        "candidate_valid": (
            candidate_is_valid
        ),
    }


    for class_id in CLASS_DEFINITIONS:

        candidate_record[
            f"train_class_{class_id}_samples"
        ] = int(
            training_counts.loc[
                class_id
            ]
        )

        candidate_record[
            f"validation_class_{class_id}_samples"
        ] = int(
            validation_counts.loc[
                class_id
            ]
        )

        candidate_record[
            f"train_class_{class_id}_blocks"
        ] = int(
            training_block_counts.loc[
                class_id
            ]
        )

        candidate_record[
            f"validation_class_{class_id}_blocks"
        ] = int(
            validation_block_counts.loc[
                class_id
            ]
        )


    candidate_records.append(
        candidate_record
    )


    if not candidate_is_valid:
        continue


    if (
        best_candidate is None
        or
        score
        <
        best_candidate[
            "score"
        ]
    ):

        best_candidate = {
            "candidate_number": (
                candidate_number
            ),
            "training_indices": (
                training_indices.copy()
            ),
            "validation_indices": (
                validation_indices.copy()
            ),
            "score": (
                score
            ),
        }


candidate_search_df = pd.DataFrame(
    candidate_records
)


valid_candidate_count = int(
    candidate_search_df[
        "candidate_valid"
    ].sum()
)


print(
    f"Candidate splits tested : "
    f"{N_CANDIDATE_SPLITS}"
)

print(
    f"Valid candidates        : "
    f"{valid_candidate_count}"
)


if best_candidate is None:

    candidate_search_df.to_csv(
        CANDIDATE_SEARCH_CSV,
        index=False,
    )

    raise RuntimeError(
        "No acceptable spatial split was found. "
        "Do not continue to model training."
    )


# =============================================================================
# 12. APPLY BEST SPLIT
# =============================================================================

feature_df[
    "split"
] = "UNASSIGNED"


feature_df.loc[
    best_candidate[
        "training_indices"
    ],
    "split",
] = "TRAIN"


feature_df.loc[
    best_candidate[
        "validation_indices"
    ],
    "split",
] = "VALIDATION"


training_df = (
    feature_df[
        feature_df[
            "split"
        ]
        == "TRAIN"
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


validation_df = (
    feature_df[
        feature_df[
            "split"
        ]
        == "VALIDATION"
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 13. FINAL VALIDATION
# =============================================================================

training_blocks = set(
    training_df[
        "spatial_block_id"
    ]
)


validation_blocks = set(
    validation_df[
        "spatial_block_id"
    ]
)


shared_blocks = (
    training_blocks
    &
    validation_blocks
)


training_class_counts = class_counts(
    training_df
)


validation_class_counts = class_counts(
    validation_df
)


training_class_blocks = class_block_counts(
    training_df
)


validation_class_blocks = class_block_counts(
    validation_df
)


validation_fraction = (
    len(
        validation_df
    )
    /
    len(
        feature_df
    )
)


all_samples_assigned = (
    len(
        training_df
    )
    +
    len(
        validation_df
    )
    ==
    len(
        feature_df
    )
)


no_shared_blocks = (
    len(
        shared_blocks
    )
    == 0
)


enough_training_samples = bool(
    (
        training_class_counts
        >= MIN_TRAINING_SAMPLES_PER_CLASS
    ).all()
)


enough_validation_samples = bool(
    (
        validation_class_counts
        >= MIN_VALIDATION_SAMPLES_PER_CLASS
    ).all()
)


enough_validation_blocks = bool(
    (
        validation_class_blocks
        >= MIN_VALIDATION_BLOCKS_PER_CLASS
    ).all()
)


validation_fraction_valid = bool(
    MIN_VALIDATION_FRACTION
    <= validation_fraction
    <= MAX_VALIDATION_FRACTION
)


minimum_distance_m = (
    minimum_cross_split_distance(
        training_df,
        validation_df,
    )
)


overall_pass = all(
    [
        all_samples_assigned,
        no_shared_blocks,
        enough_training_samples,
        enough_validation_samples,
        enough_validation_blocks,
        validation_fraction_valid,
    ]
)


# =============================================================================
# 14. CLASS SUMMARY
# =============================================================================

class_summary_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    total_samples = int(
        (
            feature_df[
                "class_id"
            ]
            == class_id
        ).sum()
    )


    train_samples = int(
        training_class_counts.loc[
            class_id
        ]
    )


    validation_samples = int(
        validation_class_counts.loc[
            class_id
        ]
    )


    class_summary_records.append(
        {
            "class_id": (
                class_id
            ),
            "class_name": (
                class_name
            ),
            "total_samples": (
                total_samples
            ),
            "training_samples": (
                train_samples
            ),
            "validation_samples": (
                validation_samples
            ),
            "training_percent_of_class": (
                train_samples
                /
                total_samples
                *
                100
            ),
            "validation_percent_of_class": (
                validation_samples
                /
                total_samples
                *
                100
            ),
            "total_spatial_blocks": (
                int(
                    overall_class_block_counts.loc[
                        class_id
                    ]
                )
            ),
            "training_spatial_blocks": (
                int(
                    training_class_blocks.loc[
                        class_id
                    ]
                )
            ),
            "validation_spatial_blocks": (
                int(
                    validation_class_blocks.loc[
                        class_id
                    ]
                )
            ),
        }
    )


class_summary_df = pd.DataFrame(
    class_summary_records
)


# =============================================================================
# 15. BLOCK SUMMARY
# =============================================================================

block_records = []


for block_id, block_df in (
    feature_df.groupby(
        "spatial_block_id"
    )
):

    block_splits = (
        block_df[
            "split"
        ].unique()
    )


    if len(
        block_splits
    ) != 1:

        raise RuntimeError(
            f"Block {block_id} contains "
            "multiple split assignments."
        )


    block_record = {
        "spatial_block_id": (
            block_id
        ),
        "spatial_block_x": (
            int(
                block_df[
                    "spatial_block_x"
                ].iloc[
                    0
                ]
            )
        ),
        "spatial_block_y": (
            int(
                block_df[
                    "spatial_block_y"
                ].iloc[
                    0
                ]
            )
        ),
        "split": (
            block_splits[
                0
            ]
        ),
        "total_samples": (
            len(
                block_df
            )
        ),
    }


    for class_id in CLASS_DEFINITIONS:

        block_record[
            f"class_{class_id}_samples"
        ] = int(
            (
                block_df[
                    "class_id"
                ]
                == class_id
            ).sum()
        )


    block_records.append(
        block_record
    )


block_summary_df = (
    pd.DataFrame(
        block_records
    )
    .sort_values(
        [
            "split",
            "spatial_block_id",
        ]
    )
)


# =============================================================================
# 16. PROCESS SUMMARY
# =============================================================================

split_summary_df = pd.DataFrame(
    [
        {
            "total_samples": (
                len(
                    feature_df
                )
            ),
            "spatial_block_size_m": (
                SPATIAL_BLOCK_SIZE_M
            ),
            "total_spatial_blocks": (
                total_blocks
            ),
            "candidate_splits_tested": (
                N_CANDIDATE_SPLITS
            ),
            "valid_candidate_splits": (
                valid_candidate_count
            ),
            "selected_candidate_number": (
                best_candidate[
                    "candidate_number"
                ]
            ),
            "selected_candidate_score": (
                best_candidate[
                    "score"
                ]
            ),
            "training_samples": (
                len(
                    training_df
                )
            ),
            "validation_samples": (
                len(
                    validation_df
                )
            ),
            "training_fraction": (
                len(
                    training_df
                )
                /
                len(
                    feature_df
                )
            ),
            "validation_fraction": (
                validation_fraction
            ),
            "training_blocks": (
                len(
                    training_blocks
                )
            ),
            "validation_blocks": (
                len(
                    validation_blocks
                )
            ),
            "shared_blocks": (
                len(
                    shared_blocks
                )
            ),
            "minimum_validation_blocks_per_class_required": (
                MIN_VALIDATION_BLOCKS_PER_CLASS
            ),
            "minimum_actual_validation_class_blocks": (
                int(
                    validation_class_blocks.min()
                )
            ),
            "minimum_train_validation_point_distance_m": (
                minimum_distance_m
            ),
            "validation_holdout_selection_used_model_performance": (
                False
            ),
            "coordinates_used_as_model_predictors": (
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
# 17. SAVE CSV OUTPUTS
# =============================================================================

feature_df.to_csv(
    SPLIT_CSV,
    index=False,
)


training_df.to_csv(
    TRAIN_CSV,
    index=False,
)


validation_df.to_csv(
    VALIDATION_CSV,
    index=False,
)


split_summary_df.to_csv(
    SPLIT_SUMMARY_CSV,
    index=False,
)


class_summary_df.to_csv(
    CLASS_SUMMARY_CSV,
    index=False,
)


block_summary_df.to_csv(
    BLOCK_SUMMARY_CSV,
    index=False,
)


candidate_search_df.to_csv(
    CANDIDATE_SEARCH_CSV,
    index=False,
)


# =============================================================================
# 18. SAVE GIS REVIEW LAYER
# =============================================================================

split_gdf = gpd.GeoDataFrame(
    feature_df.copy(),
    geometry=gpd.points_from_xy(
        feature_df[
            "easting"
        ],
        feature_df[
            "northing"
        ],
    ),
    crs=ANALYSIS_CRS,
)


if SPLIT_GPKG.exists():

    SPLIT_GPKG.unlink()


split_gdf.to_file(
    SPLIT_GPKG,
    layer="lulc_2021_spatial_split",
    driver="GPKG",
)


# =============================================================================
# 19. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("SELECTED SPATIAL SPLIT")
print("=" * 100)


print(
    f"Selected candidate        : "
    f"{best_candidate['candidate_number']}"
)

print(
    f"Selection score           : "
    f"{best_candidate['score']:.6f}"
)

print(
    f"Training samples          : "
    f"{len(training_df)} "
    f"({len(training_df) / len(feature_df) * 100:.2f}%)"
)

print(
    f"Validation samples        : "
    f"{len(validation_df)} "
    f"({validation_fraction * 100:.2f}%)"
)

print(
    f"Training blocks           : "
    f"{len(training_blocks)}"
)

print(
    f"Validation blocks         : "
    f"{len(validation_blocks)}"
)

print(
    f"Shared blocks             : "
    f"{len(shared_blocks)}"
)

print(
    f"Minimum train/validation "
    f"point distance            : "
    f"{minimum_distance_m:.3f} m"
)


print("\n" + "=" * 100)
print("CLASS DISTRIBUTION")
print("=" * 100)


print(
    class_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 06B FINAL VALIDATION")
print("=" * 100)


print(
    f"All samples assigned                   : "
    f"{all_samples_assigned}"
)

print(
    f"No shared spatial blocks               : "
    f"{no_shared_blocks}"
)

print(
    f"Enough training samples per class      : "
    f"{enough_training_samples}"
)

print(
    f"Enough validation samples per class    : "
    f"{enough_validation_samples}"
)

print(
    f">= {MIN_VALIDATION_BLOCKS_PER_CLASS} validation blocks per class         : "
    f"{enough_validation_blocks}"
)

print(
    f"Validation fraction valid              : "
    f"{validation_fraction_valid}"
)

print(
    f"Coordinates used as ML predictors      : "
    f"False"
)

print(
    f"Model performance used to choose split : "
    f"False"
)

print(
    f"Overall Step 06B status                : "
    f"{'PASS' if overall_pass else 'REVIEW REQUIRED'}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(SPLIT_CSV)}"
)

print(
    f" - {relative_path(TRAIN_CSV)}"
)

print(
    f" - {relative_path(VALIDATION_CSV)}"
)

print(
    f" - {relative_path(SPLIT_GPKG)}"
)

print(
    f" - {relative_path(SPLIT_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(CLASS_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(BLOCK_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(CANDIDATE_SEARCH_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. Complete 1 km blocks are assigned entirely to TRAIN or VALIDATION."
)

print(
    "2. Every LULC class must occur in at least two validation spatial blocks."
)

print(
    "3. The holdout was selected using structural class/spatial criteria, not model performance."
)

print(
    "4. Validation data must remain untouched during Random Forest tuning."
)

print(
    "5. This remains an internal spatial holdout, not an external independent ground-truth dataset."
)

print("=" * 100)