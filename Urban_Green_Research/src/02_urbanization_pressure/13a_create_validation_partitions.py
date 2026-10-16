"""
STEP 13A - CREATE LEAKAGE-AWARE TEMPORAL AND SPATIAL VALIDATION PARTITIONS

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Freeze validation partitions BEFORE RF/XGBoost training.
2. Create rolling time-ordered validation folds.
3. Respect 3-year target-label availability.
4. Keep 2022 as the final untouched temporal test.
5. Create 1 km spatial blocks from the permanent 100 m grid.
6. Assign each 1 km block to one of five deterministic spatial folds.
7. Keep every observation from the same physical block in the same
   spatial fold across all origin years.

Temporal design:
Fold T1:
    train      = 2012-2014
    embargo    = 2015-2016
    validation = 2017

Fold T2:
    train      = 2012-2016
    embargo    = 2017-2018
    validation = 2019

Fold T3:
    train      = 2012-2018
    embargo    = 2019-2020
    validation = 2021

Final temporal test:
    train      = 2012-2019
    embargo    = 2020-2021
    test       = 2022

Why the embargo exists:
The target for origin year t requires t+1, t+2 and t+3.
Therefore, at validation/test year V, only origins <= V-3 have
fully observable target labels.

Spatial design:
- 100 m cells are grouped into 1 km x 1 km blocks.
- A block contains up to 10 x 10 permanent 100 m cells.
- Five spatial folds are assigned independently of the target.
- Same grid cell / same 1 km block never enters different spatial folds.

IMPORTANT:
- No model is trained here.
- No target-based optimization is performed here.
- The final 2022 temporal test remains untouched.
- "1 km block validation" does NOT mean all train/test cells are
  separated by at least 1 km.
"""

from pathlib import Path
import hashlib

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
    / "12e_final_urbanization_pressure_predictors.csv"
)

FEATURE_LIST_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "12e_final_predictor_feature_list.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "splits"
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

BASE_ASSIGNMENT_CSV = (
    OUTPUT_DIR
    / "13a_validation_base_assignments.csv"
)

TEMPORAL_MANIFEST_CSV = (
    OUTPUT_DIR
    / "13a_temporal_validation_manifest.csv"
)

TEMPORAL_SUMMARY_CSV = (
    REPORT_DIR
    / "13a_temporal_validation_fold_summary.csv"
)

SPATIAL_SUMMARY_CSV = (
    REPORT_DIR
    / "13a_spatial_validation_fold_summary.csv"
)

FINAL_TEST_SUMMARY_CSV = (
    REPORT_DIR
    / "13a_final_temporal_test_summary.csv"
)

QA_CSV = (
    REPORT_DIR
    / "13a_validation_partition_qa.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "13a_validation_design_summary.csv"
)


# =============================================================================
# 3. EXPECTED DATASET VALUES
# =============================================================================

EXPECTED_ROWS = 51310
EXPECTED_POSITIVES = 844
EXPECTED_FEATURES = 55

TARGET_HORIZON_YEARS = 3

SPATIAL_BLOCK_SIZE_CELLS = 10
SPATIAL_FOLD_COUNT = 5

FINAL_TEST_ORIGIN_YEAR = 2022
FINAL_TRAIN_MAX_ORIGIN_YEAR = 2019


# =============================================================================
# 4. TEMPORAL CV DESIGN
# =============================================================================

TEMPORAL_FOLDS = [
    {
        "fold_id": "T1_VALIDATE_2017",
        "validation_year": 2017,
        "train_min_year": 2012,
        "train_max_year": 2014,
    },
    {
        "fold_id": "T2_VALIDATE_2019",
        "validation_year": 2019,
        "train_min_year": 2012,
        "train_max_year": 2016,
    },
    {
        "fold_id": "T3_VALIDATE_2021",
        "validation_year": 2021,
        "train_min_year": 2012,
        "train_max_year": 2018,
    },
]


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


def deterministic_spatial_fold(
    block_id: str,
    number_of_folds: int,
) -> int:
    """
    Stable target-independent fold assignment.

    Python's built-in hash() is intentionally NOT used because its result
    can change between Python sessions.

    SHA-256 gives a reproducible assignment.
    """

    digest = hashlib.sha256(
        block_id.encode(
            "utf-8"
        )
    ).digest()

    integer_value = int.from_bytes(
        digest[
            :8
        ],
        byteorder="big",
        signed=False,
    )

    return int(
        integer_value
        %
        number_of_folds
    )


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 13A - CREATE LEAKAGE-AWARE TEMPORAL AND SPATIAL VALIDATION PARTITIONS")
print("=" * 100)


for required_file in [
    INPUT_DATASET,
    FEATURE_LIST_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 7. LOAD FINAL FROZEN PREDICTOR DATASET
# =============================================================================

df = pd.read_csv(
    INPUT_DATASET
)


feature_list_df = pd.read_csv(
    FEATURE_LIST_FILE
)


if len(
    df
) != EXPECTED_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} modelling rows, "
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


if len(
    feature_list_df
) != EXPECTED_FEATURES:

    raise RuntimeError(
        f"Expected {EXPECTED_FEATURES} frozen predictors, "
        f"found {len(feature_list_df)}."
    )


if df.duplicated(
    subset=[
        "grid_id",
        "origin_year",
    ]
).any():

    raise RuntimeError(
        "Duplicate grid/origin rows detected."
    )


print(
    f"\nModelling rows         : "
    f"{len(df):,}"
)

print(
    f"Positive rows          : "
    f"{positive_count:,}"
)

print(
    f"Frozen ML predictors   : "
    f"{len(feature_list_df)}"
)

print(
    f"Origin-year range      : "
    f"{df['origin_year'].min()}-"
    f"{df['origin_year'].max()}"
)


# =============================================================================
# 8. CREATE STABLE ROW ID
# =============================================================================

df[
    "row_id"
] = (
    df[
        "grid_id"
    ].astype(
        str
    )
    +
    "__Y"
    +
    df[
        "origin_year"
    ].astype(
        int
    ).astype(
        str
    )
)


if df[
    "row_id"
].duplicated().any():

    raise RuntimeError(
        "Generated row IDs are not unique."
    )


# =============================================================================
# 9. CREATE 1 KM SPATIAL BLOCKS
# =============================================================================

df[
    "grid_row"
] = (
    df[
        "grid_row"
    ].astype(
        int
    )
)


df[
    "grid_col"
] = (
    df[
        "grid_col"
    ].astype(
        int
    )
)


df[
    "block_1km_row"
] = (
    df[
        "grid_row"
    ]
    //
    SPATIAL_BLOCK_SIZE_CELLS
)


df[
    "block_1km_col"
] = (
    df[
        "grid_col"
    ]
    //
    SPATIAL_BLOCK_SIZE_CELLS
)


df[
    "spatial_block_1km"
] = (
    "B1K_R"
    +
    df[
        "block_1km_row"
    ]
    .astype(
        int
    )
    .astype(
        str
    )
    .str
    .zfill(
        2
    )
    +
    "_C"
    +
    df[
        "block_1km_col"
    ]
    .astype(
        int
    )
    .astype(
        str
    )
    .str
    .zfill(
        2
    )
)


unique_blocks = (
    df[
        "spatial_block_1km"
    ]
    .drop_duplicates()
    .sort_values()
    .tolist()
)


block_to_fold = {
    block_id:
    deterministic_spatial_fold(
        block_id,
        SPATIAL_FOLD_COUNT,
    )
    for block_id in unique_blocks
}


df[
    "spatial_fold"
] = (
    df[
        "spatial_block_1km"
    ]
    .map(
        block_to_fold
    )
    .astype(
        int
    )
)


print(
    f"\nUnique 1 km blocks     : "
    f"{len(unique_blocks)}"
)

print(
    f"Spatial folds          : "
    f"{SPATIAL_FOLD_COUNT}"
)


# =============================================================================
# 10. VERIFY STATIC SPATIAL ASSIGNMENT
# =============================================================================

grid_block_counts = (
    df.groupby(
        "grid_id"
    )[
        "spatial_block_1km"
    ]
    .nunique()
)


grid_fold_counts = (
    df.groupby(
        "grid_id"
    )[
        "spatial_fold"
    ]
    .nunique()
)


grid_has_one_block = bool(
    (
        grid_block_counts
        ==
        1
    ).all()
)


grid_has_one_spatial_fold = bool(
    (
        grid_fold_counts
        ==
        1
    ).all()
)


block_fold_counts = (
    df.groupby(
        "spatial_block_1km"
    )[
        "spatial_fold"
    ]
    .nunique()
)


block_has_one_fold = bool(
    (
        block_fold_counts
        ==
        1
    ).all()
)


# =============================================================================
# 11. FINAL TEST ROLE
# =============================================================================

def final_test_role(
    origin_year: int,
) -> str:

    if origin_year <= FINAL_TRAIN_MAX_ORIGIN_YEAR:

        return "TRAIN"

    if (
        FINAL_TRAIN_MAX_ORIGIN_YEAR
        <
        origin_year
        <
        FINAL_TEST_ORIGIN_YEAR
    ):

        return "EMBARGO"

    if origin_year == FINAL_TEST_ORIGIN_YEAR:

        return "TEST"

    return "NOT_USED"


df[
    "final_temporal_test_role"
] = (
    df[
        "origin_year"
    ]
    .astype(
        int
    )
    .map(
        final_test_role
    )
)


# =============================================================================
# 12. BASE ASSIGNMENT TABLE
# =============================================================================

base_assignment_columns = [
    "row_id",
    "grid_id",
    "grid_row",
    "grid_col",
    "origin_year",
    "urban_conversion_target",
    "block_1km_row",
    "block_1km_col",
    "spatial_block_1km",
    "spatial_fold",
    "final_temporal_test_role",
]


base_assignment_df = (
    df[
        base_assignment_columns
    ]
    .copy()
)


# =============================================================================
# 13. TEMPORAL VALIDATION MANIFEST
# =============================================================================

temporal_manifest_records = []

temporal_summary_records = []

temporal_fold_qa_records = []


for fold in TEMPORAL_FOLDS:

    fold_id = (
        fold[
            "fold_id"
        ]
    )

    validation_year = int(
        fold[
            "validation_year"
        ]
    )

    train_min_year = int(
        fold[
            "train_min_year"
        ]
    )

    train_max_year = int(
        fold[
            "train_max_year"
        ]
    )


    expected_train_max_from_label_availability = (
        validation_year
        -
        TARGET_HORIZON_YEARS
    )


    if (
        train_max_year
        !=
        expected_train_max_from_label_availability
    ):

        raise RuntimeError(
            f"{fold_id}: train_max_year should equal "
            f"validation_year - {TARGET_HORIZON_YEARS}."
        )


    train_mask = (
        df[
            "origin_year"
        ].between(
            train_min_year,
            train_max_year,
        )
    )


    validation_mask = (
        df[
            "origin_year"
        ]
        ==
        validation_year
    )


    embargo_mask = (
        (
            df[
                "origin_year"
            ]
            >
            train_max_year
        )
        &
        (
            df[
                "origin_year"
            ]
            <
            validation_year
        )
    )


    future_mask = (
        df[
            "origin_year"
        ]
        >
        validation_year
    )


    for current_role, current_mask in [
        (
            "TRAIN",
            train_mask,
        ),
        (
            "EMBARGO",
            embargo_mask,
        ),
        (
            "VALIDATION",
            validation_mask,
        ),
        (
            "FUTURE_NOT_USED",
            future_mask,
        ),
    ]:

        role_df = (
            df.loc[
                current_mask,
                [
                    "row_id",
                    "grid_id",
                    "origin_year",
                    "urban_conversion_target",
                    "spatial_block_1km",
                    "spatial_fold",
                ],
            ]
            .copy()
        )


        role_df[
            "fold_id"
        ] = (
            fold_id
        )


        role_df[
            "role"
        ] = (
            current_role
        )


        temporal_manifest_records.extend(
            role_df[
                [
                    "fold_id",
                    "role",
                    "row_id",
                    "grid_id",
                    "origin_year",
                    "urban_conversion_target",
                    "spatial_block_1km",
                    "spatial_fold",
                ]
            ].to_dict(
                orient="records"
            )
        )


    train_df = (
        df.loc[
            train_mask
        ]
    )


    validation_df = (
        df.loc[
            validation_mask
        ]
    )


    embargo_df = (
        df.loc[
            embargo_mask
        ]
    )


    train_target_end_max = int(
        (
            train_df[
                "origin_year"
            ]
            +
            TARGET_HORIZON_YEARS
        ).max()
    )


    label_availability_valid = (
        train_target_end_max
        <=
        validation_year
    )


    train_validation_row_overlap = len(
        set(
            train_df[
                "row_id"
            ]
        )
        &
        set(
            validation_df[
                "row_id"
            ]
        )
    )


    train_positive = int(
        train_df[
            "urban_conversion_target"
        ].sum()
    )


    validation_positive = int(
        validation_df[
            "urban_conversion_target"
        ].sum()
    )


    temporal_summary_records.append(
        {
            "fold_id": (
                fold_id
            ),
            "train_origin_year_start": (
                train_min_year
            ),
            "train_origin_year_end": (
                train_max_year
            ),
            "train_target_information_available_through_year": (
                train_target_end_max
            ),
            "embargo_origin_year_start": (
                train_max_year
                +
                1
            ),
            "embargo_origin_year_end": (
                validation_year
                -
                1
            ),
            "validation_origin_year": (
                validation_year
            ),
            "train_rows": (
                len(
                    train_df
                )
            ),
            "train_positive_rows": (
                train_positive
            ),
            "train_positive_rate_percent": (
                train_positive
                /
                len(
                    train_df
                )
                *
                100
            ),
            "embargo_rows": (
                len(
                    embargo_df
                )
            ),
            "validation_rows": (
                len(
                    validation_df
                )
            ),
            "validation_positive_rows": (
                validation_positive
            ),
            "validation_positive_rate_percent": (
                validation_positive
                /
                len(
                    validation_df
                )
                *
                100
            ),
        }
    )


    temporal_fold_qa_records.append(
        {
            "fold_id": (
                fold_id
            ),
            "label_availability_valid": (
                label_availability_valid
            ),
            "train_validation_row_overlap": (
                train_validation_row_overlap
            ),
            "train_positive_rows_gt_0": (
                train_positive
                >
                0
            ),
            "validation_positive_rows_gt_0": (
                validation_positive
                >
                0
            ),
        }
    )


temporal_manifest_df = pd.DataFrame(
    temporal_manifest_records
)


temporal_summary_df = pd.DataFrame(
    temporal_summary_records
)


temporal_fold_qa_df = pd.DataFrame(
    temporal_fold_qa_records
)


# =============================================================================
# 14. FINAL TEMPORAL TEST SUMMARY
# =============================================================================

final_train_df = (
    df[
        df[
            "final_temporal_test_role"
        ]
        ==
        "TRAIN"
    ]
)


final_embargo_df = (
    df[
        df[
            "final_temporal_test_role"
        ]
        ==
        "EMBARGO"
    ]
)


final_test_df = (
    df[
        df[
            "final_temporal_test_role"
        ]
        ==
        "TEST"
    ]
)


final_train_target_end_max = int(
    (
        final_train_df[
            "origin_year"
        ]
        +
        TARGET_HORIZON_YEARS
    ).max()
)


final_label_availability_valid = (
    final_train_target_end_max
    <=
    FINAL_TEST_ORIGIN_YEAR
)


final_train_test_row_overlap = len(
    set(
        final_train_df[
            "row_id"
        ]
    )
    &
    set(
        final_test_df[
            "row_id"
        ]
    )
)


final_train_positive = int(
    final_train_df[
        "urban_conversion_target"
    ].sum()
)


final_test_positive = int(
    final_test_df[
        "urban_conversion_target"
    ].sum()
)


final_test_summary_df = pd.DataFrame(
    [
        {
            "final_test_origin_year": (
                FINAL_TEST_ORIGIN_YEAR
            ),
            "train_origin_year_start": int(
                final_train_df[
                    "origin_year"
                ].min()
            ),
            "train_origin_year_end": int(
                final_train_df[
                    "origin_year"
                ].max()
            ),
            "train_target_information_available_through_year": (
                final_train_target_end_max
            ),
            "embargo_origin_years": (
                "2020-2021"
            ),
            "train_rows": (
                len(
                    final_train_df
                )
            ),
            "train_positive_rows": (
                final_train_positive
            ),
            "train_positive_rate_percent": (
                final_train_positive
                /
                len(
                    final_train_df
                )
                *
                100
            ),
            "embargo_rows": (
                len(
                    final_embargo_df
                )
            ),
            "test_rows": (
                len(
                    final_test_df
                )
            ),
            "test_positive_rows": (
                final_test_positive
            ),
            "test_positive_rate_percent": (
                final_test_positive
                /
                len(
                    final_test_df
                )
                *
                100
            ),
            "label_availability_valid": (
                final_label_availability_valid
            ),
            "train_test_row_overlap": (
                final_train_test_row_overlap
            ),
        }
    ]
)


# =============================================================================
# 15. SPATIAL VALIDATION SUMMARY
# =============================================================================

# Spatial CV is built only from the development/training universe.
# The final temporal test year 2022 is not used.
#
# Origins 2020-2021 are also excluded from this tuning universe because
# they are embargoed for the final 2022 back-test.

spatial_development_df = (
    df[
        df[
            "origin_year"
        ]
        <=
        FINAL_TRAIN_MAX_ORIGIN_YEAR
    ]
    .copy()
)


spatial_summary_records = []


for spatial_fold in range(
    SPATIAL_FOLD_COUNT
):

    validation_mask = (
        spatial_development_df[
            "spatial_fold"
        ]
        ==
        spatial_fold
    )


    training_mask = (
        spatial_development_df[
            "spatial_fold"
        ]
        !=
        spatial_fold
    )


    spatial_train_df = (
        spatial_development_df[
            training_mask
        ]
    )


    spatial_validation_df = (
        spatial_development_df[
            validation_mask
        ]
    )


    train_blocks = set(
        spatial_train_df[
            "spatial_block_1km"
        ]
    )


    validation_blocks = set(
        spatial_validation_df[
            "spatial_block_1km"
        ]
    )


    shared_blocks = (
        train_blocks
        &
        validation_blocks
    )


    validation_positive = int(
        spatial_validation_df[
            "urban_conversion_target"
        ].sum()
    )


    training_positive = int(
        spatial_train_df[
            "urban_conversion_target"
        ].sum()
    )


    spatial_summary_records.append(
        {
            "spatial_fold": (
                spatial_fold
            ),
            "development_origin_years": (
                "2012-2019"
            ),
            "train_1km_blocks": (
                len(
                    train_blocks
                )
            ),
            "validation_1km_blocks": (
                len(
                    validation_blocks
                )
            ),
            "shared_1km_blocks": (
                len(
                    shared_blocks
                )
            ),
            "train_rows": (
                len(
                    spatial_train_df
                )
            ),
            "train_positive_rows": (
                training_positive
            ),
            "validation_rows": (
                len(
                    spatial_validation_df
                )
            ),
            "validation_positive_rows": (
                validation_positive
            ),
            "validation_positive_rate_percent": (
                validation_positive
                /
                len(
                    spatial_validation_df
                )
                *
                100
                if len(
                    spatial_validation_df
                )
                >
                0
                else
                np.nan
            ),
        }
    )


spatial_summary_df = pd.DataFrame(
    spatial_summary_records
)


# =============================================================================
# 16. FINAL TEST PROTECTION QA
# =============================================================================

development_fold_rows = (
    temporal_manifest_df[
        temporal_manifest_df[
            "role"
        ].isin(
            [
                "TRAIN",
                "VALIDATION",
            ]
        )
    ]
)


final_test_used_in_temporal_cv = bool(
    (
        development_fold_rows[
            "origin_year"
        ]
        ==
        FINAL_TEST_ORIGIN_YEAR
    ).any()
)


spatial_cv_uses_final_test = bool(
    (
        spatial_development_df[
            "origin_year"
        ]
        ==
        FINAL_TEST_ORIGIN_YEAR
    ).any()
)


# =============================================================================
# 17. GLOBAL QA
# =============================================================================

temporal_label_availability_valid = bool(
    temporal_fold_qa_df[
        "label_availability_valid"
    ].all()
)


temporal_row_overlap_zero = bool(
    (
        temporal_fold_qa_df[
            "train_validation_row_overlap"
        ]
        ==
        0
    ).all()
)


all_temporal_train_folds_have_positives = bool(
    temporal_fold_qa_df[
        "train_positive_rows_gt_0"
    ].all()
)


all_temporal_validation_folds_have_positives = bool(
    temporal_fold_qa_df[
        "validation_positive_rows_gt_0"
    ].all()
)


all_spatial_folds_have_positives = bool(
    (
        spatial_summary_df[
            "validation_positive_rows"
        ]
        >
        0
    ).all()
)


spatial_shared_blocks_zero = bool(
    (
        spatial_summary_df[
            "shared_1km_blocks"
        ]
        ==
        0
    ).all()
)


all_spatial_folds_present = (
    sorted(
        df[
            "spatial_fold"
        ]
        .unique()
        .tolist()
    )
    ==
    list(
        range(
            SPATIAL_FOLD_COUNT
        )
    )
)


final_test_has_positives = (
    final_test_positive
    >
    0
)


final_test_protected = bool(
    not final_test_used_in_temporal_cv
    and
    not spatial_cv_uses_final_test
)


overall_status = (
    "PASS"
    if all(
        [
            grid_has_one_block,
            grid_has_one_spatial_fold,
            block_has_one_fold,
            temporal_label_availability_valid,
            temporal_row_overlap_zero,
            all_temporal_train_folds_have_positives,
            all_temporal_validation_folds_have_positives,
            all_spatial_folds_have_positives,
            spatial_shared_blocks_zero,
            all_spatial_folds_present,
            final_label_availability_valid,
            final_train_test_row_overlap
            ==
            0,
            final_test_has_positives,
            final_test_protected,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 18. QA REPORT
# =============================================================================

qa_df = pd.DataFrame(
    [
        {
            "model_rows": (
                len(
                    df
                )
            ),
            "positive_rows": (
                positive_count
            ),
            "frozen_feature_count": (
                len(
                    feature_list_df
                )
            ),
            "unique_1km_blocks": (
                len(
                    unique_blocks
                )
            ),
            "spatial_fold_count": (
                SPATIAL_FOLD_COUNT
            ),
            "each_grid_has_one_1km_block": (
                grid_has_one_block
            ),
            "each_grid_has_one_spatial_fold": (
                grid_has_one_spatial_fold
            ),
            "each_1km_block_has_one_spatial_fold": (
                block_has_one_fold
            ),
            "all_spatial_folds_present": (
                all_spatial_folds_present
            ),
            "spatial_train_validation_shared_blocks_zero": (
                spatial_shared_blocks_zero
            ),
            "all_spatial_validation_folds_have_positive_cases": (
                all_spatial_folds_have_positives
            ),
            "temporal_label_availability_valid": (
                temporal_label_availability_valid
            ),
            "temporal_train_validation_row_overlap_zero": (
                temporal_row_overlap_zero
            ),
            "all_temporal_train_folds_have_positive_cases": (
                all_temporal_train_folds_have_positives
            ),
            "all_temporal_validation_folds_have_positive_cases": (
                all_temporal_validation_folds_have_positives
            ),
            "final_test_label_availability_valid": (
                final_label_availability_valid
            ),
            "final_train_test_row_overlap": (
                final_train_test_row_overlap
            ),
            "final_test_positive_cases_gt_0": (
                final_test_has_positives
            ),
            "final_test_2022_protected_from_cv": (
                final_test_protected
            ),
            "models_trained": (
                False
            ),
            "overall_step13a_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 19. FINAL SUMMARY
# =============================================================================

final_summary_df = pd.DataFrame(
    [
        {
            "validation_design": (
                "Rolling temporal validation + independent 1km spatial block CV"
            ),
            "target_horizon_years": (
                TARGET_HORIZON_YEARS
            ),
            "temporal_cv_fold_count": (
                len(
                    TEMPORAL_FOLDS
                )
            ),
            "temporal_validation_years": (
                "2017,2019,2021"
            ),
            "final_temporal_test_year": (
                FINAL_TEST_ORIGIN_YEAR
            ),
            "final_temporal_train_origin_years": (
                "2012-2019"
            ),
            "final_temporal_embargo_origin_years": (
                "2020-2021"
            ),
            "spatial_block_size": (
                "1km x 1km"
            ),
            "spatial_fold_count": (
                SPATIAL_FOLD_COUNT
            ),
            "spatial_cv_development_origin_years": (
                "2012-2019"
            ),
            "spatial_assignment_target_independent": (
                True
            ),
            "final_test_used_for_tuning": (
                False
            ),
            "models_trained": (
                False
            ),
            "overall_step13a_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 20. SAVE OUTPUTS
# =============================================================================

base_assignment_df.to_csv(
    BASE_ASSIGNMENT_CSV,
    index=False,
)


temporal_manifest_df.to_csv(
    TEMPORAL_MANIFEST_CSV,
    index=False,
)


temporal_summary_df.to_csv(
    TEMPORAL_SUMMARY_CSV,
    index=False,
)


spatial_summary_df.to_csv(
    SPATIAL_SUMMARY_CSV,
    index=False,
)


final_test_summary_df.to_csv(
    FINAL_TEST_SUMMARY_CSV,
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
# 21. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("TEMPORAL VALIDATION FOLDS")
print("=" * 100)


print(
    temporal_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("FINAL UNTOUCHED TEMPORAL TEST")
print("=" * 100)


print(
    final_test_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("1 KM SPATIAL BLOCK FOLDS")
print("=" * 100)


print(
    spatial_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 13A QA")
print("=" * 100)


print(
    f"Each grid has one 1 km block             : "
    f"{grid_has_one_block}"
)

print(
    f"Each grid has one spatial fold           : "
    f"{grid_has_one_spatial_fold}"
)

print(
    f"Each 1 km block has one fold             : "
    f"{block_has_one_fold}"
)

print(
    f"All 5 spatial folds present              : "
    f"{all_spatial_folds_present}"
)

print(
    f"Spatial shared blocks train/validation   : "
    f"{not spatial_shared_blocks_zero}"
)

print(
    f"All spatial folds have positives         : "
    f"{all_spatial_folds_have_positives}"
)

print(
    f"Temporal label availability valid        : "
    f"{temporal_label_availability_valid}"
)

print(
    f"Temporal train/validation overlap        : "
    f"{not temporal_row_overlap_zero}"
)

print(
    f"All temporal validation folds positives  : "
    f"{all_temporal_validation_folds_have_positives}"
)

print(
    f"Final test label availability valid      : "
    f"{final_label_availability_valid}"
)

print(
    f"Final train/test row overlap             : "
    f"{final_train_test_row_overlap}"
)

print(
    f"Final test has positive cases            : "
    f"{final_test_has_positives}"
)

print(
    f"2022 final test protected from tuning    : "
    f"{final_test_protected}"
)

print(
    f"Models trained                           : "
    f"False"
)

print(
    f"Overall Step 13A status                  : "
    f"{overall_status}"
)


print("\nGenerated split files:")

print(
    f" - {relative_path(BASE_ASSIGNMENT_CSV)}"
)

print(
    f" - {relative_path(TEMPORAL_MANIFEST_CSV)}"
)


print("\nGenerated reports:")

print(
    f" - {relative_path(TEMPORAL_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(SPATIAL_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(FINAL_TEST_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(QA_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. The final predictor set was not changed."
)

print(
    "2. No model was trained in this step."
)

print(
    "3. 2022 is reserved as the untouched final temporal test."
)

print(
    "4. Origins 2020-2021 are embargoed when back-testing the 2022 prediction."
)

print(
    "5. Temporal folds only use training labels that would be observable by the validation year."
)

print(
    "6. Spatial folds are assigned using 1 km blocks and are independent of the target."
)

print(
    "7. All observations from the same 1 km block stay in the same spatial fold."
)

print(
    "8. This is 1 km block validation; it does not mean every train/test cell is physically 1 km apart."
)

print(
    "9. After this passes, RF/XGBoost tuning can begin without touching the final 2022 test."
)

print("=" * 100)