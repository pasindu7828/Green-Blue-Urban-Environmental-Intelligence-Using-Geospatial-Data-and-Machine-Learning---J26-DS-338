"""
STEP 13B - RF + XGBOOST DEVELOPMENT BASELINE EVALUATION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Evaluate baseline Random Forest and XGBoost models.
2. Use the frozen 55-predictor set.
3. Use the frozen Step 13A temporal and spatial validation design.
4. Keep the final 2022 temporal test completely untouched.
5. Handle severe class imbalance using model-specific class weighting.
6. Produce development-only performance reports and out-of-fold predictions.

Development validation:
TEMPORAL:
    T1 -> validate 2017
    T2 -> validate 2019
    T3 -> validate 2021

SPATIAL:
    5 independent 1 km block validation folds
    development origins 2012-2019 only

Final 2022 test:
    NOT LOADED INTO MODEL EVALUATION
    NOT USED FOR THRESHOLD SELECTION
    NOT USED FOR HYPERPARAMETER SELECTION

Models:
- Random Forest
- XGBoost

Primary metric:
- PR-AUC / Average Precision

Additional metrics:
- ROC-AUC
- Brier Score
- Log Loss
- Precision @ 0.50
- Recall @ 0.50
- F1 @ 0.50
- Balanced Accuracy @ 0.50

IMPORTANT:
- This is BASELINE development evaluation, not final model selection.
- Hyperparameter tuning happens after this baseline passes.
"""

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# =============================================================================
# 1. PROJECT PATHS
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

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "urbanization_pressure"
    / "baseline"
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

MODEL_DIR.mkdir(
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

FOLD_METRICS_CSV = (
    REPORT_DIR
    / "13b_baseline_fold_metrics.csv"
)

MODEL_SUMMARY_CSV = (
    REPORT_DIR
    / "13b_baseline_model_summary.csv"
)

TEMPORAL_OOF_CSV = (
    OUTPUT_DIR
    / "13b_temporal_oof_predictions.csv"
)

SPATIAL_OOF_CSV = (
    OUTPUT_DIR
    / "13b_spatial_oof_predictions.csv"
)

QA_CSV = (
    REPORT_DIR
    / "13b_baseline_evaluation_qa.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "13b_baseline_evaluation_summary.csv"
)


# =============================================================================
# 3. EXPECTED VALUES
# =============================================================================

EXPECTED_ROWS = 51310
EXPECTED_POSITIVE_ROWS = 844
EXPECTED_FEATURES = 55

FINAL_TEST_YEAR = 2022
SPATIAL_DEVELOPMENT_MAX_YEAR = 2019

RANDOM_STATE = 2026

CLASSIFICATION_THRESHOLD = 0.50


# =============================================================================
# 4. MODEL DEFINITIONS
# =============================================================================

RF_BASE_PARAMS = {
    "n_estimators": 500,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "bootstrap": True,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


XGB_BASE_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 2.0,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


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


def calculate_scale_pos_weight(y_train):
    """
    XGBoost imbalance weight calculated ONLY from current training fold.
    """

    y_train = np.asarray(
        y_train,
        dtype=int,
    )

    positive = int(
        np.sum(
            y_train
            ==
            1
        )
    )

    negative = int(
        np.sum(
            y_train
            ==
            0
        )
    )

    if positive <= 0:

        raise RuntimeError(
            "Training fold contains no positive cases."
        )

    return float(
        negative
        /
        positive
    )


def calculate_metrics(
    y_true,
    probability,
    threshold=0.50,
):
    """Calculate threshold-free and fixed-threshold metrics."""

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )


    prediction = (
        probability
        >=
        threshold
    ).astype(
        np.uint8
    )


    if len(
        np.unique(
            y_true
        )
    ) < 2:

        raise RuntimeError(
            "Validation fold must contain both target classes."
        )


    return {
        "pr_auc_average_precision": float(
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
        "log_loss": float(
            log_loss(
                y_true,
                probability,
                labels=[
                    0,
                    1,
                ],
            )
        ),
        "precision_at_0_50": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "recall_at_0_50": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "f1_at_0_50": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "balanced_accuracy_at_0_50": float(
            balanced_accuracy_score(
                y_true,
                prediction,
            )
        ),
        "predicted_positive_count_at_0_50": int(
            prediction.sum()
        ),
        "mean_predicted_probability": float(
            np.mean(
                probability
            )
        ),
    }


def fit_and_evaluate(
    model_name,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """
    Fit one baseline model and return probabilities + metadata.
    """

    if model_name == "RandomForest":

        model = RandomForestClassifier(
            **RF_BASE_PARAMS
        )

        imbalance_parameter = np.nan


    elif model_name == "XGBoost":

        scale_pos_weight = calculate_scale_pos_weight(
            y_train
        )


        current_params = (
            XGB_BASE_PARAMS.copy()
        )


        current_params[
            "scale_pos_weight"
        ] = (
            scale_pos_weight
        )


        model = XGBClassifier(
            **current_params
        )


        imbalance_parameter = (
            scale_pos_weight
        )


    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


    start_time = time.perf_counter()


    model.fit(
        X_train,
        y_train,
    )


    training_seconds = (
        time.perf_counter()
        -
        start_time
    )


    probability = (
        model.predict_proba(
            X_validation
        )[
            :,
            1
        ]
    )


    return (
        model,
        probability,
        training_seconds,
        imbalance_parameter,
    )


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 13B - RF + XGBOOST DEVELOPMENT BASELINE EVALUATION")
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
# 7. LOAD DATA
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
) != EXPECTED_POSITIVE_ROWS:

    raise RuntimeError(
        "Positive target count does not match Step 11F."
    )


if len(
    feature_list_df
) != EXPECTED_FEATURES:

    raise RuntimeError(
        f"Expected {EXPECTED_FEATURES} frozen predictors, "
        f"found {len(feature_list_df)}."
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


missing_features = [
    feature
    for feature in feature_columns
    if feature not in df.columns
]


if missing_features:

    raise RuntimeError(
        "Frozen features missing from dataset:\n"
        +
        "\n".join(
            missing_features
        )
    )


if df[
    feature_columns
].isna().any().any():

    raise RuntimeError(
        "Missing values exist in frozen predictors."
    )


if np.isinf(
    df[
        feature_columns
    ].to_numpy(
        dtype=float
    )
).any():

    raise RuntimeError(
        "Infinite values exist in frozen predictors."
    )


# =============================================================================
# 8. BUILD STABLE ROW ID
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
        "Dataset row IDs are not unique."
    )


df = df.set_index(
    "row_id",
    drop=False,
)


print(
    f"\nRows                 : "
    f"{len(df):,}"
)

print(
    f"Positive rows        : "
    f"{df['urban_conversion_target'].sum():,}"
)

print(
    f"Frozen predictors    : "
    f"{len(feature_columns)}"
)

print(
    "Final 2022 test used : False"
)


# =============================================================================
# 9. VERIFY 2022 FINAL TEST IS NOT PART OF DEVELOPMENT MANIFEST
# =============================================================================

development_manifest = (
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


final_test_in_temporal_development = bool(
    (
        development_manifest[
            "origin_year"
        ]
        ==
        FINAL_TEST_YEAR
    ).any()
)


if final_test_in_temporal_development:

    raise RuntimeError(
        "2022 final test leaked into temporal development manifest."
    )


# =============================================================================
# 10. TEMPORAL BASELINE EVALUATION
# =============================================================================

print("\n" + "=" * 100)
print("TEMPORAL DEVELOPMENT VALIDATION")
print("=" * 100)


fold_metric_records = []

temporal_oof_records = []


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

    print(
        f"\nTemporal fold: {fold_id}"
    )


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


    train_df = (
        df.loc[
            train_ids
        ]
    )


    validation_df = (
        df.loc[
            validation_ids
        ]
    )


    if (
        train_df[
            "origin_year"
        ].max()
        >=
        validation_df[
            "origin_year"
        ].min()
    ):

        raise RuntimeError(
            f"{fold_id}: temporal ordering invalid."
        )


    if (
        validation_df[
            "origin_year"
        ]
        ==
        FINAL_TEST_YEAR
    ).any():

        raise RuntimeError(
            f"{fold_id}: final 2022 test was used."
        )


    X_train = (
        train_df[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    y_train = (
        train_df[
            "urban_conversion_target"
        ]
        .to_numpy(
            dtype=np.uint8
        )
    )


    X_validation = (
        validation_df[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    y_validation = (
        validation_df[
            "urban_conversion_target"
        ]
        .to_numpy(
            dtype=np.uint8
        )
    )


    print(
        f"  Train rows      : "
        f"{len(train_df):,} "
        f"(positive={y_train.sum():,})"
    )

    print(
        f"  Validation rows : "
        f"{len(validation_df):,} "
        f"(positive={y_validation.sum():,})"
    )


    for model_name in [
        "RandomForest",
        "XGBoost",
    ]:

        (
            model,
            probability,
            training_seconds,
            imbalance_parameter,
        ) = fit_and_evaluate(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
        )


        metrics = calculate_metrics(
            y_true=y_validation,
            probability=probability,
            threshold=CLASSIFICATION_THRESHOLD,
        )


        fold_metric_records.append(
            {
                "validation_type": (
                    "TEMPORAL"
                ),
                "fold_id": (
                    fold_id
                ),
                "model": (
                    model_name
                ),
                "train_rows": (
                    len(
                        train_df
                    )
                ),
                "train_positive_rows": int(
                    y_train.sum()
                ),
                "validation_rows": (
                    len(
                        validation_df
                    )
                ),
                "validation_positive_rows": int(
                    y_validation.sum()
                ),
                "class_weight_method": (
                    "balanced_subsample"
                    if model_name
                    ==
                    "RandomForest"
                    else
                    "scale_pos_weight"
                ),
                "xgb_scale_pos_weight": (
                    imbalance_parameter
                ),
                "training_seconds": (
                    training_seconds
                ),
                **metrics,
            }
        )


        for (
            row_id,
            grid_id,
            origin_year,
            true_target,
            probability_value,
        ) in zip(
            validation_df[
                "row_id"
            ],
            validation_df[
                "grid_id"
            ],
            validation_df[
                "origin_year"
            ],
            y_validation,
            probability,
        ):

            temporal_oof_records.append(
                {
                    "validation_type": (
                        "TEMPORAL"
                    ),
                    "fold_id": (
                        fold_id
                    ),
                    "model": (
                        model_name
                    ),
                    "row_id": (
                        row_id
                    ),
                    "grid_id": (
                        grid_id
                    ),
                    "origin_year": int(
                        origin_year
                    ),
                    "actual_target": int(
                        true_target
                    ),
                    "predicted_probability": float(
                        probability_value
                    ),
                }
            )


        print(
            f"  {model_name:<12} "
            f"PR-AUC={metrics['pr_auc_average_precision']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f} | "
            f"F1@0.50={metrics['f1_at_0_50']:.4f}"
        )


# =============================================================================
# 11. SPATIAL BASELINE EVALUATION
# =============================================================================

print("\n" + "=" * 100)
print("1 KM SPATIAL BLOCK DEVELOPMENT VALIDATION")
print("=" * 100)


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


if development_df[
    "spatial_fold"
].isna().any():

    raise RuntimeError(
        "Spatial fold assignment missing for development rows."
    )


if (
    development_df[
        "origin_year"
    ]
    ==
    FINAL_TEST_YEAR
).any():

    raise RuntimeError(
        "2022 final test entered spatial model development."
    )


spatial_oof_records = []


for spatial_fold in sorted(
    development_df[
        "spatial_fold"
    ]
    .astype(
        int
    )
    .unique()
):

    print(
        f"\nSpatial fold: {spatial_fold}"
    )


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


    shared_blocks = (
        train_blocks
        &
        validation_blocks
    )


    if len(
        shared_blocks
    ) != 0:

        raise RuntimeError(
            f"Spatial fold {spatial_fold}: "
            f"shared 1 km blocks detected."
        )


    X_train = (
        train_df[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    y_train = (
        train_df[
            "urban_conversion_target"
        ]
        .to_numpy(
            dtype=np.uint8
        )
    )


    X_validation = (
        validation_df[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    y_validation = (
        validation_df[
            "urban_conversion_target"
        ]
        .to_numpy(
            dtype=np.uint8
        )
    )


    print(
        f"  Train rows      : "
        f"{len(train_df):,} "
        f"(positive={y_train.sum():,})"
    )

    print(
        f"  Validation rows : "
        f"{len(validation_df):,} "
        f"(positive={y_validation.sum():,})"
    )

    print(
        f"  Shared blocks   : "
        f"{len(shared_blocks)}"
    )


    for model_name in [
        "RandomForest",
        "XGBoost",
    ]:

        (
            model,
            probability,
            training_seconds,
            imbalance_parameter,
        ) = fit_and_evaluate(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
        )


        metrics = calculate_metrics(
            y_true=y_validation,
            probability=probability,
            threshold=CLASSIFICATION_THRESHOLD,
        )


        fold_metric_records.append(
            {
                "validation_type": (
                    "SPATIAL_1KM"
                ),
                "fold_id": (
                    f"S{spatial_fold}"
                ),
                "model": (
                    model_name
                ),
                "train_rows": (
                    len(
                        train_df
                    )
                ),
                "train_positive_rows": int(
                    y_train.sum()
                ),
                "validation_rows": (
                    len(
                        validation_df
                    )
                ),
                "validation_positive_rows": int(
                    y_validation.sum()
                ),
                "class_weight_method": (
                    "balanced_subsample"
                    if model_name
                    ==
                    "RandomForest"
                    else
                    "scale_pos_weight"
                ),
                "xgb_scale_pos_weight": (
                    imbalance_parameter
                ),
                "training_seconds": (
                    training_seconds
                ),
                **metrics,
            }
        )


        for (
            row_id,
            grid_id,
            origin_year,
            block_id,
            true_target,
            probability_value,
        ) in zip(
            validation_df[
                "row_id"
            ],
            validation_df[
                "grid_id"
            ],
            validation_df[
                "origin_year"
            ],
            validation_df[
                "spatial_block_1km"
            ],
            y_validation,
            probability,
        ):

            spatial_oof_records.append(
                {
                    "validation_type": (
                        "SPATIAL_1KM"
                    ),
                    "fold_id": (
                        f"S{spatial_fold}"
                    ),
                    "model": (
                        model_name
                    ),
                    "row_id": (
                        row_id
                    ),
                    "grid_id": (
                        grid_id
                    ),
                    "origin_year": int(
                        origin_year
                    ),
                    "spatial_block_1km": (
                        block_id
                    ),
                    "actual_target": int(
                        true_target
                    ),
                    "predicted_probability": float(
                        probability_value
                    ),
                }
            )


        print(
            f"  {model_name:<12} "
            f"PR-AUC={metrics['pr_auc_average_precision']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f} | "
            f"F1@0.50={metrics['f1_at_0_50']:.4f}"
        )


# =============================================================================
# 12. CREATE RESULT TABLES
# =============================================================================

fold_metrics_df = pd.DataFrame(
    fold_metric_records
)


temporal_oof_df = pd.DataFrame(
    temporal_oof_records
)


spatial_oof_df = pd.DataFrame(
    spatial_oof_records
)


# =============================================================================
# 13. MODEL SUMMARY
# =============================================================================

summary_metric_columns = [
    "pr_auc_average_precision",
    "roc_auc",
    "brier_score",
    "log_loss",
    "precision_at_0_50",
    "recall_at_0_50",
    "f1_at_0_50",
    "balanced_accuracy_at_0_50",
]


summary_records = []


for (
    validation_type,
    model_name,
), group_df in fold_metrics_df.groupby(
    [
        "validation_type",
        "model",
    ],
    sort=True,
):

    record = {
        "validation_type": (
            validation_type
        ),
        "model": (
            model_name
        ),
        "fold_count": (
            len(
                group_df
            )
        ),
    }


    for metric_name in summary_metric_columns:

        metric_values = (
            group_df[
                metric_name
            ]
            .to_numpy(
                dtype=float
            )
        )


        record[
            f"{metric_name}_mean"
        ] = float(
            np.mean(
                metric_values
            )
        )


        record[
            f"{metric_name}_std"
        ] = float(
            np.std(
                metric_values,
                ddof=1,
            )
            if len(
                metric_values
            )
            >
            1
            else
            0.0
        )


    summary_records.append(
        record
    )


model_summary_df = pd.DataFrame(
    summary_records
)


# =============================================================================
# 14. QA
# =============================================================================

expected_temporal_prediction_rows = 0


for fold_id in temporal_fold_ids:

    fold_validation_rows = (
        temporal_manifest_df[
            (
                temporal_manifest_df[
                    "fold_id"
                ]
                ==
                fold_id
            )
            &
            (
                temporal_manifest_df[
                    "role"
                ]
                ==
                "VALIDATION"
            )
        ]
    )


    expected_temporal_prediction_rows += (
        len(
            fold_validation_rows
        )
        *
        2
    )


temporal_prediction_count_valid = (
    len(
        temporal_oof_df
    )
    ==
    expected_temporal_prediction_rows
)


expected_spatial_prediction_rows = (
    len(
        development_df
    )
    *
    2
)


spatial_prediction_count_valid = (
    len(
        spatial_oof_df
    )
    ==
    expected_spatial_prediction_rows
)


final_test_present_temporal_oof = bool(
    (
        temporal_oof_df[
            "origin_year"
        ]
        ==
        FINAL_TEST_YEAR
    ).any()
)


final_test_present_spatial_oof = bool(
    (
        spatial_oof_df[
            "origin_year"
        ]
        ==
        FINAL_TEST_YEAR
    ).any()
)


all_metric_values_finite = bool(
    np.isfinite(
        fold_metrics_df[
            summary_metric_columns
        ].to_numpy(
            dtype=float
        )
    ).all()
)


probability_bounds_valid = bool(
    (
        temporal_oof_df[
            "predicted_probability"
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    )
    and
    (
        spatial_oof_df[
            "predicted_probability"
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    )
)


both_models_present = (
    set(
        fold_metrics_df[
            "model"
        ].unique()
    )
    ==
    {
        "RandomForest",
        "XGBoost",
    }
)


temporal_fold_count_valid = (
    fold_metrics_df[
        fold_metrics_df[
            "validation_type"
        ]
        ==
        "TEMPORAL"
    ][
        "fold_id"
    ]
    .nunique()
    ==
    3
)


spatial_fold_count_valid = (
    fold_metrics_df[
        fold_metrics_df[
            "validation_type"
        ]
        ==
        "SPATIAL_1KM"
    ][
        "fold_id"
    ]
    .nunique()
    ==
    5
)


final_test_untouched = bool(
    not final_test_present_temporal_oof
    and
    not final_test_present_spatial_oof
    and
    not final_test_in_temporal_development
)


models_saved_for_final_use = False


overall_status = (
    "PASS"
    if all(
        [
            temporal_prediction_count_valid,
            spatial_prediction_count_valid,
            not final_test_present_temporal_oof,
            not final_test_present_spatial_oof,
            all_metric_values_finite,
            probability_bounds_valid,
            both_models_present,
            temporal_fold_count_valid,
            spatial_fold_count_valid,
            final_test_untouched,
            not models_saved_for_final_use,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 15. QA REPORT
# =============================================================================

qa_df = pd.DataFrame(
    [
        {
            "dataset_rows": (
                len(
                    df
                )
            ),
            "frozen_feature_count": (
                len(
                    feature_columns
                )
            ),
            "positive_target_rows": int(
                df[
                    "urban_conversion_target"
                ].sum()
            ),
            "temporal_fold_count": (
                3
            ),
            "spatial_fold_count": (
                5
            ),
            "temporal_oof_prediction_rows": (
                len(
                    temporal_oof_df
                )
            ),
            "temporal_oof_row_count_valid": (
                temporal_prediction_count_valid
            ),
            "spatial_oof_prediction_rows": (
                len(
                    spatial_oof_df
                )
            ),
            "spatial_oof_row_count_valid": (
                spatial_prediction_count_valid
            ),
            "both_models_present": (
                both_models_present
            ),
            "all_metric_values_finite": (
                all_metric_values_finite
            ),
            "probability_bounds_valid": (
                probability_bounds_valid
            ),
            "final_2022_test_present_in_temporal_oof": (
                final_test_present_temporal_oof
            ),
            "final_2022_test_present_in_spatial_oof": (
                final_test_present_spatial_oof
            ),
            "final_2022_test_untouched": (
                final_test_untouched
            ),
            "classification_threshold": (
                CLASSIFICATION_THRESHOLD
            ),
            "threshold_tuned_in_this_step": (
                False
            ),
            "hyperparameters_tuned_in_this_step": (
                False
            ),
            "final_models_saved": (
                models_saved_for_final_use
            ),
            "overall_step13b_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 16. FINAL SUMMARY
# =============================================================================

final_summary_df = pd.DataFrame(
    [
        {
            "stage": (
                "Development baseline evaluation"
            ),
            "models": (
                "RandomForest;XGBoost"
            ),
            "primary_metric": (
                "PR-AUC / Average Precision"
            ),
            "temporal_validation_years": (
                "2017,2019,2021"
            ),
            "spatial_validation": (
                "5-fold 1km block validation"
            ),
            "spatial_development_origin_years": (
                "2012-2019"
            ),
            "final_2022_test_used": (
                False
            ),
            "smote_used": (
                False
            ),
            "rf_imbalance_handling": (
                "balanced_subsample"
            ),
            "xgb_imbalance_handling": (
                "training-fold scale_pos_weight"
            ),
            "hyperparameter_tuning_completed": (
                False
            ),
            "threshold_selection_completed": (
                False
            ),
            "final_model_selected": (
                False
            ),
            "overall_step13b_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 17. SAVE OUTPUTS
# =============================================================================

fold_metrics_df.to_csv(
    FOLD_METRICS_CSV,
    index=False,
)


model_summary_df.to_csv(
    MODEL_SUMMARY_CSV,
    index=False,
)


temporal_oof_df.to_csv(
    TEMPORAL_OOF_CSV,
    index=False,
)


spatial_oof_df.to_csv(
    SPATIAL_OOF_CSV,
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
# 18. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("BASELINE FOLD METRICS")
print("=" * 100)


display_columns = [
    "validation_type",
    "fold_id",
    "model",
    "train_rows",
    "train_positive_rows",
    "validation_rows",
    "validation_positive_rows",
    "pr_auc_average_precision",
    "roc_auc",
    "brier_score",
    "precision_at_0_50",
    "recall_at_0_50",
    "f1_at_0_50",
    "balanced_accuracy_at_0_50",
]


print(
    fold_metrics_df[
        display_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("BASELINE MODEL SUMMARY")
print("=" * 100)


summary_display_columns = [
    "validation_type",
    "model",
    "fold_count",
    "pr_auc_average_precision_mean",
    "pr_auc_average_precision_std",
    "roc_auc_mean",
    "roc_auc_std",
    "brier_score_mean",
    "f1_at_0_50_mean",
    "balanced_accuracy_at_0_50_mean",
]


print(
    model_summary_df[
        summary_display_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 13B QA")
print("=" * 100)


print(
    f"Frozen predictors                     : "
    f"{len(feature_columns)}"
)

print(
    f"Temporal folds                        : "
    f"{temporal_fold_count_valid}"
)

print(
    f"Spatial folds                         : "
    f"{spatial_fold_count_valid}"
)

print(
    f"Temporal OOF row count valid          : "
    f"{temporal_prediction_count_valid}"
)

print(
    f"Spatial OOF row count valid           : "
    f"{spatial_prediction_count_valid}"
)

print(
    f"Both RF and XGBoost evaluated         : "
    f"{both_models_present}"
)

print(
    f"Metric values finite                  : "
    f"{all_metric_values_finite}"
)

print(
    f"Probability values within [0,1]       : "
    f"{probability_bounds_valid}"
)

print(
    f"2022 in temporal OOF                  : "
    f"{final_test_present_temporal_oof}"
)

print(
    f"2022 in spatial OOF                   : "
    f"{final_test_present_spatial_oof}"
)

print(
    f"2022 final test untouched             : "
    f"{final_test_untouched}"
)

print(
    f"Threshold tuned                       : "
    f"False"
)

print(
    f"Hyperparameters tuned                 : "
    f"False"
)

print(
    f"Final model selected                  : "
    f"False"
)

print(
    f"Overall Step 13B status               : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(FOLD_METRICS_CSV)}"
)

print(
    f" - {relative_path(MODEL_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(TEMPORAL_OOF_CSV)}"
)

print(
    f" - {relative_path(SPATIAL_OOF_CSV)}"
)

print(
    f" - {relative_path(QA_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. These are baseline development results only."
)

print(
    "2. PR-AUC is the primary metric because the target is strongly imbalanced."
)

print(
    "3. Accuracy is intentionally not used as the primary performance measure."
)

print(
    "4. The 0.50 classification threshold is only a fixed diagnostic threshold."
)

print(
    "5. The classification threshold has NOT been tuned."
)

print(
    "6. Hyperparameters have NOT been tuned."
)

print(
    "7. 2022 final temporal test remains completely untouched."
)

print(
    "8. No SMOTE or synthetic oversampling was used."
)

print(
    "9. The next step will compare the RF/XGBoost baseline behaviour and then perform development-only hyperparameter tuning."
)

print("=" * 100)