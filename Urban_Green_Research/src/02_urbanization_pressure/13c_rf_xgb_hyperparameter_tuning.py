"""
STEP 13C - RF + XGBOOST DEVELOPMENT-ONLY HYPERPARAMETER TUNING

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Tune Random Forest and XGBoost using DEVELOPMENT validation only.
2. Use the frozen 55 predictors.
3. Use the frozen Step 13A temporal and spatial validation folds.
4. Keep the final 2022 temporal test completely untouched.
5. Optimize primarily for PR-AUC because the positive class is rare.
6. Select one parameter configuration per model.

Model-selection score:
    balanced_development_pr_auc
        = (mean temporal PR-AUC + mean spatial PR-AUC) / 2

Why:
- Temporal and spatial generalization are both important.
- There are 3 temporal folds and 5 spatial folds.
- Averaging the two validation families prevents the larger spatial
  fold count from automatically dominating model selection.

Tie breakers:
1. Higher minimum PR-AUC across all development folds.
2. Lower mean Brier score.

IMPORTANT:
- 2022 is NOT evaluated.
- Classification threshold is NOT tuned here.
- No SMOTE is used.
- XGBoost scale_pos_weight is calculated independently for each
  training fold.
- Final fitted production models are NOT created here.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from xgboost import XGBClassifier


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
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

BASE_ASSIGNMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "splits"
    / "13a_validation_base_assignments.csv"
)

TEMPORAL_MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "splits"
    / "13a_temporal_validation_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "model_development"
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

FOLD_RESULTS_CSV = (
    REPORT_DIR
    / "13c_tuning_fold_results.csv"
)

CANDIDATE_SUMMARY_CSV = (
    REPORT_DIR
    / "13c_tuning_candidate_summary.csv"
)

BEST_PARAMS_CSV = (
    REPORT_DIR
    / "13c_best_hyperparameters.csv"
)

BEST_PARAMS_JSON = (
    OUTPUT_DIR
    / "13c_best_hyperparameters.json"
)

QA_CSV = (
    REPORT_DIR
    / "13c_hyperparameter_tuning_qa.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "13c_hyperparameter_tuning_summary.csv"
)


# =============================================================================
# 3. EXPECTED VALUES
# =============================================================================

EXPECTED_ROWS = 51310
EXPECTED_POSITIVES = 844
EXPECTED_FEATURES = 55

FINAL_TEST_YEAR = 2022
SPATIAL_DEVELOPMENT_MAX_YEAR = 2019

RANDOM_STATE = 2026


# =============================================================================
# 4. RANDOM FOREST CANDIDATES
# =============================================================================

RF_CANDIDATES = [
    {
        "candidate_id": "RF01_BASELINE",
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "max_samples": None,
    },
    {
        "candidate_id": "RF02_DEPTH12",
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "max_samples": None,
    },
    {
        "candidate_id": "RF03_DEPTH16",
        "n_estimators": 600,
        "max_depth": 16,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "max_samples": None,
    },
    {
        "candidate_id": "RF04_LEAF4",
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_split": 8,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "max_samples": None,
    },
    {
        "candidate_id": "RF05_HALF_FEATURES",
        "n_estimators": 600,
        "max_depth": 14,
        "min_samples_split": 8,
        "min_samples_leaf": 4,
        "max_features": 0.50,
        "max_samples": None,
    },
    {
        "candidate_id": "RF06_CONSERVATIVE",
        "n_estimators": 500,
        "max_depth": 8,
        "min_samples_split": 10,
        "min_samples_leaf": 6,
        "max_features": "sqrt",
        "max_samples": None,
    },
    {
        "candidate_id": "RF07_BOOTSTRAP80",
        "n_estimators": 600,
        "max_depth": 12,
        "min_samples_split": 8,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "max_samples": 0.80,
    },
    {
        "candidate_id": "RF08_HALF_FEATURES_LEAF6",
        "n_estimators": 600,
        "max_depth": 12,
        "min_samples_split": 10,
        "min_samples_leaf": 6,
        "max_features": 0.50,
        "max_samples": 0.80,
    },
]


# =============================================================================
# 5. XGBOOST CANDIDATES
# =============================================================================

XGB_CANDIDATES = [
    {
        "candidate_id": "XGB01_BASELINE",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 2.0,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
    },
    {
        "candidate_id": "XGB02_DEPTH3",
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 2.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
    },
    {
        "candidate_id": "XGB03_DEPTH4_REG",
        "n_estimators": 600,
        "learning_rate": 0.04,
        "max_depth": 4,
        "min_child_weight": 4.0,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.10,
        "reg_lambda": 2.0,
        "gamma": 0.0,
    },
    {
        "candidate_id": "XGB04_DEPTH5_REG",
        "n_estimators": 500,
        "learning_rate": 0.04,
        "max_depth": 5,
        "min_child_weight": 5.0,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.20,
        "reg_lambda": 3.0,
        "gamma": 0.05,
    },
    {
        "candidate_id": "XGB05_DEPTH3_STRONG_REG",
        "n_estimators": 700,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 5.0,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.20,
        "reg_lambda": 4.0,
        "gamma": 0.10,
    },
    {
        "candidate_id": "XGB06_DEPTH4_FAST",
        "n_estimators": 400,
        "learning_rate": 0.08,
        "max_depth": 4,
        "min_child_weight": 3.0,
        "subsample": 0.80,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
        "gamma": 0.0,
    },
    {
        "candidate_id": "XGB07_DEPTH6_REG",
        "n_estimators": 500,
        "learning_rate": 0.04,
        "max_depth": 6,
        "min_child_weight": 6.0,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.30,
        "reg_lambda": 4.0,
        "gamma": 0.10,
    },
    {
        "candidate_id": "XGB08_CONSERVATIVE",
        "n_estimators": 600,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 8.0,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.50,
        "reg_lambda": 5.0,
        "gamma": 0.20,
    },
]


# =============================================================================
# 6. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def calculate_scale_pos_weight(y_train):

    y_train = np.asarray(
        y_train,
        dtype=np.uint8,
    )

    positives = int(
        np.sum(
            y_train
            ==
            1
        )
    )

    negatives = int(
        np.sum(
            y_train
            ==
            0
        )
    )

    if positives <= 0:
        raise RuntimeError(
            "Training fold has zero positive cases."
        )

    return float(
        negatives
        /
        positives
    )


def evaluate_probabilities(
    y_true,
    probability,
):

    y_true = np.asarray(
        y_true,
        dtype=np.uint8,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    if len(
        np.unique(
            y_true
        )
    ) != 2:

        raise RuntimeError(
            "Validation data must contain both classes."
        )

    return {
        "pr_auc": float(
            average_precision_score(
                y_true,
                probability,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probability,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probability,
            )
        ),
    }


def build_model(
    model_name,
    candidate,
    y_train,
):

    params = {
        key: value
        for key, value in candidate.items()
        if key != "candidate_id"
    }

    if model_name == "RandomForest":

        return (
            RandomForestClassifier(
                criterion="gini",
                class_weight="balanced_subsample",
                bootstrap=True,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                **params,
            ),
            np.nan,
        )

    if model_name == "XGBoost":

        scale_pos_weight = calculate_scale_pos_weight(
            y_train
        )

        return (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                scale_pos_weight=scale_pos_weight,
                verbosity=0,
                **params,
            ),
            scale_pos_weight,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# =============================================================================
# 7. START
# =============================================================================

print("=" * 100)
print("STEP 13C - RF + XGBOOST DEVELOPMENT-ONLY HYPERPARAMETER TUNING")
print("=" * 100)


for required_file in [
    DATASET_FILE,
    FEATURE_LIST_FILE,
    BASE_ASSIGNMENT_FILE,
    TEMPORAL_MANIFEST_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 8. LOAD DATA
# =============================================================================

df = pd.read_csv(
    DATASET_FILE
)

feature_list_df = pd.read_csv(
    FEATURE_LIST_FILE
)

base_assignment_df = pd.read_csv(
    BASE_ASSIGNMENT_FILE
)

temporal_manifest_df = pd.read_csv(
    TEMPORAL_MANIFEST_FILE
)


if len(
    df
) != EXPECTED_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} rows, "
        f"found {len(df):,}."
    )


if int(
    df[
        "urban_conversion_target"
    ].sum()
) != EXPECTED_POSITIVES:

    raise RuntimeError(
        "Positive target count mismatch."
    )


if len(
    feature_list_df
) != EXPECTED_FEATURES:

    raise RuntimeError(
        "Frozen feature count mismatch."
    )


feature_columns = (
    feature_list_df[
        "feature_name"
    ]
    .astype(
        str
    )
    .tolist()
)


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
    ]
    .astype(
        int
    )
    .astype(
        str
    )
)


if df[
    "row_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate row IDs detected."
    )


df = df.set_index(
    "row_id",
    drop=False,
)


print(
    f"\nRows               : "
    f"{len(df):,}"
)

print(
    f"Positive rows      : "
    f"{df['urban_conversion_target'].sum():,}"
)

print(
    f"Frozen predictors  : "
    f"{len(feature_columns)}"
)

print(
    f"RF candidates      : "
    f"{len(RF_CANDIDATES)}"
)

print(
    f"XGB candidates     : "
    f"{len(XGB_CANDIDATES)}"
)

print(
    "Final 2022 test    : UNTOUCHED"
)


# =============================================================================
# 9. PREPARE TEMPORAL FOLDS
# =============================================================================

temporal_folds = []


temporal_fold_ids = (
    temporal_manifest_df.loc[
        temporal_manifest_df[
            "role"
        ]
        ==
        "VALIDATION",
        "fold_id",
    ]
    .drop_duplicates()
    .tolist()
)


for fold_id in temporal_fold_ids:

    fold_manifest = (
        temporal_manifest_df[
            temporal_manifest_df[
                "fold_id"
            ]
            ==
            fold_id
        ]
    )


    train_ids = (
        fold_manifest.loc[
            fold_manifest[
                "role"
            ]
            ==
            "TRAIN",
            "row_id",
        ]
        .tolist()
    )


    validation_ids = (
        fold_manifest.loc[
            fold_manifest[
                "role"
            ]
            ==
            "VALIDATION",
            "row_id",
        ]
        .tolist()
    )


    train_df = df.loc[
        train_ids
    ]


    validation_df = df.loc[
        validation_ids
    ]


    if (
        validation_df[
            "origin_year"
        ]
        ==
        FINAL_TEST_YEAR
    ).any():

        raise RuntimeError(
            "2022 leaked into temporal tuning."
        )


    temporal_folds.append(
        {
            "fold_id": (
                fold_id
            ),
            "X_train": (
                train_df[
                    feature_columns
                ]
                .to_numpy(
                    dtype=np.float32
                )
            ),
            "y_train": (
                train_df[
                    "urban_conversion_target"
                ]
                .to_numpy(
                    dtype=np.uint8
                )
            ),
            "X_validation": (
                validation_df[
                    feature_columns
                ]
                .to_numpy(
                    dtype=np.float32
                )
            ),
            "y_validation": (
                validation_df[
                    "urban_conversion_target"
                ]
                .to_numpy(
                    dtype=np.uint8
                )
            ),
        }
    )


# =============================================================================
# 10. PREPARE SPATIAL FOLDS
# =============================================================================

assignment_lookup = (
    base_assignment_df[
        [
            "row_id",
            "spatial_block_1km",
            "spatial_fold",
        ]
    ]
    .drop_duplicates(
        subset="row_id"
    )
)


development_df = (
    df[
        df[
            "origin_year"
        ]
        <=
        SPATIAL_DEVELOPMENT_MAX_YEAR
    ]
    .reset_index(
        drop=True
    )
)


development_df = development_df.merge(
    assignment_lookup,
    on="row_id",
    how="left",
    validate="one_to_one",
)


if (
    development_df[
        "origin_year"
    ]
    ==
    FINAL_TEST_YEAR
).any():

    raise RuntimeError(
        "2022 leaked into spatial tuning."
    )


spatial_folds = []


for spatial_fold in sorted(
    development_df[
        "spatial_fold"
    ]
    .astype(
        int
    )
    .unique()
):

    train_df = (
        development_df[
            development_df[
                "spatial_fold"
            ]
            !=
            spatial_fold
        ]
    )


    validation_df = (
        development_df[
            development_df[
                "spatial_fold"
            ]
            ==
            spatial_fold
        ]
    )


    train_blocks = set(
        train_df[
            "spatial_block_1km"
        ]
    )

    validation_blocks = set(
        validation_df[
            "spatial_block_1km"
        ]
    )


    if len(
        train_blocks
        &
        validation_blocks
    ) != 0:

        raise RuntimeError(
            f"Spatial fold {spatial_fold} contains shared blocks."
        )


    spatial_folds.append(
        {
            "fold_id": (
                f"S{spatial_fold}"
            ),
            "X_train": (
                train_df[
                    feature_columns
                ]
                .to_numpy(
                    dtype=np.float32
                )
            ),
            "y_train": (
                train_df[
                    "urban_conversion_target"
                ]
                .to_numpy(
                    dtype=np.uint8
                )
            ),
            "X_validation": (
                validation_df[
                    feature_columns
                ]
                .to_numpy(
                    dtype=np.float32
                )
            ),
            "y_validation": (
                validation_df[
                    "urban_conversion_target"
                ]
                .to_numpy(
                    dtype=np.uint8
                )
            ),
        }
    )


print(
    f"\nTemporal tuning folds : "
    f"{len(temporal_folds)}"
)

print(
    f"Spatial tuning folds  : "
    f"{len(spatial_folds)}"
)


# =============================================================================
# 11. EVALUATE ALL CANDIDATES
# =============================================================================

fold_records = []


all_models = [
    (
        "RandomForest",
        RF_CANDIDATES,
    ),
    (
        "XGBoost",
        XGB_CANDIDATES,
    ),
]


for model_name, candidate_list in all_models:

    print("\n" + "=" * 100)
    print(
        f"TUNING {model_name.upper()}"
    )
    print("=" * 100)


    for candidate_number, candidate in enumerate(
        candidate_list,
        start=1,
    ):

        candidate_id = (
            candidate[
                "candidate_id"
            ]
        )


        print(
            f"\n{model_name} candidate "
            f"{candidate_number}/{len(candidate_list)}: "
            f"{candidate_id}"
        )


        for validation_type, folds in [
            (
                "TEMPORAL",
                temporal_folds,
            ),
            (
                "SPATIAL_1KM",
                spatial_folds,
            ),
        ]:

            for fold in folds:

                model, scale_pos_weight = build_model(
                    model_name=model_name,
                    candidate=candidate,
                    y_train=fold[
                        "y_train"
                    ],
                )


                start_time = (
                    time.perf_counter()
                )


                model.fit(
                    fold[
                        "X_train"
                    ],
                    fold[
                        "y_train"
                    ],
                )


                training_seconds = (
                    time.perf_counter()
                    -
                    start_time
                )


                probability = (
                    model.predict_proba(
                        fold[
                            "X_validation"
                        ]
                    )[
                        :,
                        1
                    ]
                )


                metrics = evaluate_probabilities(
                    y_true=fold[
                        "y_validation"
                    ],
                    probability=probability,
                )


                fold_records.append(
                    {
                        "model": (
                            model_name
                        ),
                        "candidate_id": (
                            candidate_id
                        ),
                        "validation_type": (
                            validation_type
                        ),
                        "fold_id": (
                            fold[
                                "fold_id"
                            ]
                        ),
                        "train_rows": (
                            len(
                                fold[
                                    "y_train"
                                ]
                            )
                        ),
                        "train_positive_rows": int(
                            fold[
                                "y_train"
                            ].sum()
                        ),
                        "validation_rows": (
                            len(
                                fold[
                                    "y_validation"
                                ]
                            )
                        ),
                        "validation_positive_rows": int(
                            fold[
                                "y_validation"
                            ].sum()
                        ),
                        "scale_pos_weight": (
                            scale_pos_weight
                        ),
                        "training_seconds": (
                            training_seconds
                        ),
                        **metrics,
                    }
                )


        candidate_rows = pd.DataFrame(
            [
                row
                for row in fold_records
                if (
                    row[
                        "model"
                    ]
                    ==
                    model_name
                    and
                    row[
                        "candidate_id"
                    ]
                    ==
                    candidate_id
                )
            ]
        )


        temporal_mean = float(
            candidate_rows.loc[
                candidate_rows[
                    "validation_type"
                ]
                ==
                "TEMPORAL",
                "pr_auc",
            ].mean()
        )


        spatial_mean = float(
            candidate_rows.loc[
                candidate_rows[
                    "validation_type"
                ]
                ==
                "SPATIAL_1KM",
                "pr_auc",
            ].mean()
        )


        balanced_score = (
            temporal_mean
            +
            spatial_mean
        ) / 2.0


        print(
            f"  Temporal mean PR-AUC : "
            f"{temporal_mean:.5f}"
        )

        print(
            f"  Spatial mean PR-AUC  : "
            f"{spatial_mean:.5f}"
        )

        print(
            f"  Balanced dev score   : "
            f"{balanced_score:.5f}"
        )


fold_results_df = pd.DataFrame(
    fold_records
)


# =============================================================================
# 12. BUILD CANDIDATE SUMMARY
# =============================================================================

candidate_summary_records = []


for (
    model_name,
    candidate_id,
), candidate_df in fold_results_df.groupby(
    [
        "model",
        "candidate_id",
    ],
    sort=False,
):

    temporal_df = (
        candidate_df[
            candidate_df[
                "validation_type"
            ]
            ==
            "TEMPORAL"
        ]
    )


    spatial_df = (
        candidate_df[
            candidate_df[
                "validation_type"
            ]
            ==
            "SPATIAL_1KM"
        ]
    )


    temporal_pr_auc_mean = float(
        temporal_df[
            "pr_auc"
        ].mean()
    )


    temporal_pr_auc_std = float(
        temporal_df[
            "pr_auc"
        ].std(
            ddof=1
        )
    )


    spatial_pr_auc_mean = float(
        spatial_df[
            "pr_auc"
        ].mean()
    )


    spatial_pr_auc_std = float(
        spatial_df[
            "pr_auc"
        ].std(
            ddof=1
        )
    )


    balanced_development_pr_auc = (
        temporal_pr_auc_mean
        +
        spatial_pr_auc_mean
    ) / 2.0


    minimum_fold_pr_auc = float(
        candidate_df[
            "pr_auc"
        ].min()
    )


    mean_brier_score = float(
        candidate_df[
            "brier_score"
        ].mean()
    )


    candidate_summary_records.append(
        {
            "model": (
                model_name
            ),
            "candidate_id": (
                candidate_id
            ),
            "temporal_pr_auc_mean": (
                temporal_pr_auc_mean
            ),
            "temporal_pr_auc_std": (
                temporal_pr_auc_std
            ),
            "spatial_pr_auc_mean": (
                spatial_pr_auc_mean
            ),
            "spatial_pr_auc_std": (
                spatial_pr_auc_std
            ),
            "balanced_development_pr_auc": (
                balanced_development_pr_auc
            ),
            "minimum_fold_pr_auc": (
                minimum_fold_pr_auc
            ),
            "mean_roc_auc": float(
                candidate_df[
                    "roc_auc"
                ].mean()
            ),
            "mean_brier_score": (
                mean_brier_score
            ),
            "mean_training_seconds": float(
                candidate_df[
                    "training_seconds"
                ].mean()
            ),
        }
    )


candidate_summary_df = pd.DataFrame(
    candidate_summary_records
)


candidate_summary_df = (
    candidate_summary_df.sort_values(
        [
            "model",
            "balanced_development_pr_auc",
            "minimum_fold_pr_auc",
            "mean_brier_score",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 13. SELECT BEST CANDIDATE PER MODEL
# =============================================================================

best_parameter_records = []

best_parameter_json = {}


candidate_lookup = {
    "RandomForest": {
        candidate[
            "candidate_id"
        ]:
        candidate
        for candidate in RF_CANDIDATES
    },
    "XGBoost": {
        candidate[
            "candidate_id"
        ]:
        candidate
        for candidate in XGB_CANDIDATES
    },
}


for model_name in [
    "RandomForest",
    "XGBoost",
]:

    model_summary = (
        candidate_summary_df[
            candidate_summary_df[
                "model"
            ]
            ==
            model_name
        ]
        .sort_values(
            [
                "balanced_development_pr_auc",
                "minimum_fold_pr_auc",
                "mean_brier_score",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    best_row = (
        model_summary.iloc[
            0
        ]
    )


    best_candidate_id = (
        best_row[
            "candidate_id"
        ]
    )


    best_candidate = (
        candidate_lookup[
            model_name
        ][
            best_candidate_id
        ]
    )


    clean_params = {
        key: value
        for key, value in best_candidate.items()
        if key != "candidate_id"
    }


    best_parameter_json[
        model_name
    ] = {
        "candidate_id": (
            best_candidate_id
        ),
        "parameters": (
            clean_params
        ),
        "balanced_development_pr_auc": float(
            best_row[
                "balanced_development_pr_auc"
            ]
        ),
        "temporal_pr_auc_mean": float(
            best_row[
                "temporal_pr_auc_mean"
            ]
        ),
        "spatial_pr_auc_mean": float(
            best_row[
                "spatial_pr_auc_mean"
            ]
        ),
        "minimum_fold_pr_auc": float(
            best_row[
                "minimum_fold_pr_auc"
            ]
        ),
    }


    best_parameter_records.append(
        {
            "model": (
                model_name
            ),
            "best_candidate_id": (
                best_candidate_id
            ),
            "balanced_development_pr_auc": float(
                best_row[
                    "balanced_development_pr_auc"
                ]
            ),
            "temporal_pr_auc_mean": float(
                best_row[
                    "temporal_pr_auc_mean"
                ]
            ),
            "spatial_pr_auc_mean": float(
                best_row[
                    "spatial_pr_auc_mean"
                ]
            ),
            "minimum_fold_pr_auc": float(
                best_row[
                    "minimum_fold_pr_auc"
                ]
            ),
            "mean_brier_score": float(
                best_row[
                    "mean_brier_score"
                ]
            ),
            "parameters_json": json.dumps(
                clean_params,
                sort_keys=True,
            ),
        }
    )


best_params_df = pd.DataFrame(
    best_parameter_records
)


# =============================================================================
# 14. QA
# =============================================================================

expected_total_fits = (
    (
        len(
            RF_CANDIDATES
        )
        +
        len(
            XGB_CANDIDATES
        )
    )
    *
    (
        len(
            temporal_folds
        )
        +
        len(
            spatial_folds
        )
    )
)


actual_total_fits = len(
    fold_results_df
)


fit_count_valid = (
    actual_total_fits
    ==
    expected_total_fits
)


all_metrics_finite = bool(
    np.isfinite(
        fold_results_df[
            [
                "pr_auc",
                "roc_auc",
                "brier_score",
            ]
        ].to_numpy(
            dtype=float
        )
    ).all()
)


both_models_present = (
    set(
        best_params_df[
            "model"
        ].tolist()
    )
    ==
    {
        "RandomForest",
        "XGBoost",
    }
)


all_candidates_have_8_folds = bool(
    (
        fold_results_df.groupby(
            [
                "model",
                "candidate_id",
            ]
        )
        .size()
        ==
        8
    ).all()
)


final_2022_test_used = False

threshold_tuned = False

final_model_fitted = False


overall_status = (
    "PASS"
    if all(
        [
            fit_count_valid,
            all_metrics_finite,
            both_models_present,
            all_candidates_have_8_folds,
            not final_2022_test_used,
            not threshold_tuned,
            not final_model_fitted,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 15. REPORTS
# =============================================================================

qa_df = pd.DataFrame(
    [
        {
            "rf_candidate_count": (
                len(
                    RF_CANDIDATES
                )
            ),
            "xgb_candidate_count": (
                len(
                    XGB_CANDIDATES
                )
            ),
            "temporal_fold_count": (
                len(
                    temporal_folds
                )
            ),
            "spatial_fold_count": (
                len(
                    spatial_folds
                )
            ),
            "expected_total_model_fits": (
                expected_total_fits
            ),
            "actual_total_model_fits": (
                actual_total_fits
            ),
            "fit_count_valid": (
                fit_count_valid
            ),
            "all_candidates_have_8_development_folds": (
                all_candidates_have_8_folds
            ),
            "all_metrics_finite": (
                all_metrics_finite
            ),
            "best_candidate_for_each_model_found": (
                both_models_present
            ),
            "selection_metric": (
                "Equal-weight mean of temporal and spatial PR-AUC"
            ),
            "final_2022_test_used": (
                final_2022_test_used
            ),
            "threshold_tuned": (
                threshold_tuned
            ),
            "final_model_fitted": (
                final_model_fitted
            ),
            "overall_step13c_status": (
                overall_status
            ),
        }
    ]
)


final_summary_df = pd.DataFrame(
    [
        {
            "stage": (
                "Development-only hyperparameter tuning"
            ),
            "primary_metric": (
                "PR-AUC / Average Precision"
            ),
            "selection_score": (
                "(Temporal mean PR-AUC + Spatial mean PR-AUC) / 2"
            ),
            "rf_candidates": (
                len(
                    RF_CANDIDATES
                )
            ),
            "xgb_candidates": (
                len(
                    XGB_CANDIDATES
                )
            ),
            "development_folds_per_candidate": (
                8
            ),
            "final_2022_test_used": (
                False
            ),
            "smote_used": (
                False
            ),
            "classification_threshold_selected": (
                False
            ),
            "final_model_selected_between_rf_xgb": (
                False
            ),
            "overall_step13c_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 16. SAVE
# =============================================================================

fold_results_df.to_csv(
    FOLD_RESULTS_CSV,
    index=False,
)


candidate_summary_df.to_csv(
    CANDIDATE_SUMMARY_CSV,
    index=False,
)


best_params_df.to_csv(
    BEST_PARAMS_CSV,
    index=False,
)


with open(
    BEST_PARAMS_JSON,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        best_parameter_json,
        file,
        indent=2,
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
print("CANDIDATE SUMMARY")
print("=" * 100)


print(
    candidate_summary_df[
        [
            "model",
            "candidate_id",
            "temporal_pr_auc_mean",
            "temporal_pr_auc_std",
            "spatial_pr_auc_mean",
            "spatial_pr_auc_std",
            "balanced_development_pr_auc",
            "minimum_fold_pr_auc",
            "mean_brier_score",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("BEST DEVELOPMENT PARAMETERS")
print("=" * 100)


print(
    best_params_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 13C QA")
print("=" * 100)


print(
    f"Expected model fits                  : "
    f"{expected_total_fits}"
)

print(
    f"Actual model fits                    : "
    f"{actual_total_fits}"
)

print(
    f"Fit count valid                      : "
    f"{fit_count_valid}"
)

print(
    f"Every candidate has 8 folds          : "
    f"{all_candidates_have_8_folds}"
)

print(
    f"All metric values finite             : "
    f"{all_metrics_finite}"
)

print(
    f"Best RF and XGB candidates found     : "
    f"{both_models_present}"
)

print(
    f"2022 final test used                 : "
    f"{final_2022_test_used}"
)

print(
    f"Threshold tuned                      : "
    f"{threshold_tuned}"
)

print(
    f"Final model fitted                   : "
    f"{final_model_fitted}"
)

print(
    f"Overall Step 13C status              : "
    f"{overall_status}"
)


print("\nGenerated reports:")

print(
    f" - {relative_path(FOLD_RESULTS_CSV)}"
)

print(
    f" - {relative_path(CANDIDATE_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(BEST_PARAMS_CSV)}"
)

print(
    f" - {relative_path(BEST_PARAMS_JSON)}"
)

print(
    f" - {relative_path(QA_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. Hyperparameters were selected using development validation only."
)

print(
    "2. Temporal and spatial PR-AUC receive equal weight in candidate selection."
)

print(
    "3. The 2022 final test remains untouched."
)

print(
    "4. No classification threshold was selected in this step."
)

print(
    "5. No final production model was fitted."
)

print(
    "6. The next step will regenerate OOF probabilities using only the selected RF/XGB configurations and select a development-only decision threshold."
)

print("=" * 100)