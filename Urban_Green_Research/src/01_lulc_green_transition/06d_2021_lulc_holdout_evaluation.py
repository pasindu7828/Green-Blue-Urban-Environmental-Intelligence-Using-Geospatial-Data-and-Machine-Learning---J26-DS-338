"""
STEP 06D - 2021 LULC FINAL SPATIAL HOLDOUT EVALUATION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Load the frozen Random Forest model from Step 06C.
2. Load the untouched spatial validation subset from Step 06B.
3. Verify train/validation spatial separation.
4. Evaluate the model once on the spatial holdout.
5. Calculate overall and class-level performance metrics.
6. Produce a confusion matrix and sample-level predictions.
7. Record misclassified samples for later interpretation.

IMPORTANT:
- This is the first model-performance evaluation using the Step 06B holdout.
- The holdout must NOT be used for further hyperparameter tuning.
- Coordinates remain metadata only and are not predictors.
- Metrics represent performance against the curated internal reference labels.
- This is NOT an external independent ground-truth accuracy assessment.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
)


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

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "lulc"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. INPUTS
# =============================================================================

MODEL_FILE = (
    MODEL_DIR
    / "lulc_random_forest_2021.joblib"
)

TRAINING_CSV = (
    MODEL_DATA_DIR
    / "lulc_2021_training_features.csv"
)

VALIDATION_CSV = (
    MODEL_DATA_DIR
    / "lulc_2021_validation_features.csv"
)


# =============================================================================
# 3. OUTPUTS
# =============================================================================

OVERALL_METRICS_CSV = (
    REPORT_DIR
    / "06d_rf_holdout_overall_metrics.csv"
)

CLASS_METRICS_CSV = (
    REPORT_DIR
    / "06d_rf_holdout_class_metrics.csv"
)

CONFUSION_MATRIX_CSV = (
    REPORT_DIR
    / "06d_rf_holdout_confusion_matrix.csv"
)

PREDICTIONS_CSV = (
    REPORT_DIR
    / "06d_rf_holdout_predictions.csv"
)

MISCLASSIFICATIONS_CSV = (
    REPORT_DIR
    / "06d_rf_holdout_misclassifications.csv"
)

EVALUATION_SUMMARY_CSV = (
    REPORT_DIR
    / "06d_rf_holdout_evaluation_summary.csv"
)


# =============================================================================
# 4. CONSTANTS
# =============================================================================

CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}

CLASS_IDS = [
    1,
    2,
    3,
    4,
]


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


def safe_mean(series: pd.Series) -> float:
    """
    Safe numerical mean.
    """

    if len(series) == 0:
        return np.nan

    return float(
        pd.to_numeric(
            series,
            errors="coerce",
        ).mean()
    )


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 06D - 2021 LULC FINAL SPATIAL HOLDOUT EVALUATION")
print("=" * 100)


# =============================================================================
# 7. CHECK INPUT FILES
# =============================================================================

for required_file in [
    MODEL_FILE,
    TRAINING_CSV,
    VALIDATION_CSV,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required input not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 8. LOAD FROZEN MODEL
# =============================================================================

model_bundle = joblib.load(
    MODEL_FILE
)


required_bundle_keys = {
    "model",
    "features",
    "class_definitions",
    "best_parameters",
    "best_cv_macro_f1",
}


missing_bundle_keys = (
    required_bundle_keys
    -
    set(
        model_bundle.keys()
    )
)


if missing_bundle_keys:

    raise ValueError(
        "Model bundle is missing required keys: "
        f"{sorted(missing_bundle_keys)}"
    )


model = (
    model_bundle[
        "model"
    ]
)

MODEL_FEATURES = list(
    model_bundle[
        "features"
    ]
)


print(
    f"\nModel              : "
    f"{model_bundle.get('model_name', 'RandomForestClassifier')}"
)

print(
    f"Training year      : "
    f"{model_bundle.get('training_year', 2021)}"
)

print(
    f"Model features     : "
    f"{len(MODEL_FEATURES)}"
)

print(
    f"Spatial CV Macro F1: "
    f"{float(model_bundle['best_cv_macro_f1']):.6f}"
)


# =============================================================================
# 9. LOAD TRAINING + HOLDOUT DATA
# =============================================================================

training_df = pd.read_csv(
    TRAINING_CSV
)

validation_df = pd.read_csv(
    VALIDATION_CSV
)


print("\n" + "=" * 100)
print("DATASETS")
print("=" * 100)


print(
    f"Training samples   : "
    f"{len(training_df)}"
)

print(
    f"Validation samples : "
    f"{len(validation_df)}"
)


# =============================================================================
# 10. VALIDATE REQUIRED COLUMNS
# =============================================================================

required_columns = {
    "sample_id",
    "class_id",
    "class_name",
    "spatial_block_id",
    "landsat_pixel_id",
    *MODEL_FEATURES,
}


for dataset_name, dataframe in [
    (
        "training",
        training_df,
    ),
    (
        "validation",
        validation_df,
    ),
]:

    missing_columns = (
        required_columns
        -
        set(
            dataframe.columns
        )
    )

    if missing_columns:

        raise ValueError(
            f"{dataset_name} dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )


# =============================================================================
# 11. STRUCTURAL HOLDOUT INTEGRITY CHECK
# =============================================================================

training_sample_ids = set(
    training_df[
        "sample_id"
    ].astype(str)
)

validation_sample_ids = set(
    validation_df[
        "sample_id"
    ].astype(str)
)


shared_sample_ids = (
    training_sample_ids
    &
    validation_sample_ids
)


training_pixel_ids = set(
    training_df[
        "landsat_pixel_id"
    ].astype(str)
)

validation_pixel_ids = set(
    validation_df[
        "landsat_pixel_id"
    ].astype(str)
)


shared_pixel_ids = (
    training_pixel_ids
    &
    validation_pixel_ids
)


training_blocks = set(
    training_df[
        "spatial_block_id"
    ].astype(str)
)

validation_blocks = set(
    validation_df[
        "spatial_block_id"
    ].astype(str)
)


shared_blocks = (
    training_blocks
    &
    validation_blocks
)


if shared_sample_ids:

    raise RuntimeError(
        "Training/validation sample leakage detected."
    )


if shared_pixel_ids:

    raise RuntimeError(
        "Training/validation Landsat-pixel leakage detected."
    )


if shared_blocks:

    raise RuntimeError(
        "Training/validation spatial-block leakage detected."
    )


print(
    f"\nShared sample IDs        : "
    f"{len(shared_sample_ids)}"
)

print(
    f"Shared Landsat pixels    : "
    f"{len(shared_pixel_ids)}"
)

print(
    f"Shared 1 km blocks       : "
    f"{len(shared_blocks)}"
)

print(
    "Spatial holdout integrity: PASS"
)


# =============================================================================
# 12. VALIDATE HOLDOUT LABELS + FEATURES
# =============================================================================

validation_df[
    "class_id"
] = pd.to_numeric(
    validation_df[
        "class_id"
    ],
    errors="raise",
).astype(int)


if set(
    validation_df[
        "class_id"
    ].unique()
) != set(
    CLASS_IDS
):

    raise ValueError(
        "All four LULC classes must be present "
        "in the holdout dataset."
    )


if validation_df[
    MODEL_FEATURES
].isna().any().any():

    raise ValueError(
        "Missing predictor values exist in the holdout."
    )


X_validation = (
    validation_df[
        MODEL_FEATURES
    ]
    .copy()
)


y_true = (
    validation_df[
        "class_id"
    ]
    .to_numpy(
        dtype=int
    )
)


# =============================================================================
# 13. VERIFY MODEL CLASS ORDER
# =============================================================================

model_classes = [
    int(
        value
    )
    for value in model.classes_
]


if set(
    model_classes
) != set(
    CLASS_IDS
):

    raise ValueError(
        f"Unexpected model classes: "
        f"{model_classes}"
    )


# =============================================================================
# 14. ONE-TIME HOLDOUT PREDICTION
# =============================================================================

print("\n" + "=" * 100)
print("RUNNING ONE-TIME SPATIAL HOLDOUT PREDICTION")
print("=" * 100)


y_pred = (
    model.predict(
        X_validation
    )
    .astype(int)
)


y_proba = (
    model.predict_proba(
        X_validation
    )
)


# =============================================================================
# 15. OVERALL METRICS
# =============================================================================

accuracy = float(
    accuracy_score(
        y_true,
        y_pred,
    )
)


balanced_accuracy = float(
    balanced_accuracy_score(
        y_true,
        y_pred,
    )
)


macro_precision = float(
    precision_score(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        average="macro",
        zero_division=0,
    )
)


macro_recall = float(
    recall_score(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        average="macro",
        zero_division=0,
    )
)


macro_f1 = float(
    f1_score(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        average="macro",
        zero_division=0,
    )
)


weighted_f1 = float(
    f1_score(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        average="weighted",
        zero_division=0,
    )
)


cohen_kappa = float(
    cohen_kappa_score(
        y_true,
        y_pred,
        labels=CLASS_IDS,
    )
)


multiclass_log_loss = float(
    log_loss(
        y_true,
        y_proba,
        labels=model_classes,
    )
)


correct_prediction_count = int(
    np.sum(
        y_true
        ==
        y_pred
    )
)


misclassification_count = int(
    np.sum(
        y_true
        !=
        y_pred
    )
)


# =============================================================================
# 16. CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=CLASS_IDS,
)


confusion_matrix_df = pd.DataFrame(
    cm,
    index=[
        (
            f"Actual_{class_id}_"
            f"{CLASS_DEFINITIONS[class_id]}"
        )
        for class_id in CLASS_IDS
    ],
    columns=[
        (
            f"Predicted_{class_id}_"
            f"{CLASS_DEFINITIONS[class_id]}"
        )
        for class_id in CLASS_IDS
    ],
)


confusion_matrix_df.index.name = (
    "actual_class"
)


# =============================================================================
# 17. PER-CLASS METRICS
# =============================================================================

(
    class_precision,
    class_recall,
    class_f1,
    class_support,
) = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=CLASS_IDS,
    zero_division=0,
)


class_metric_records = []


for index, class_id in enumerate(
    CLASS_IDS
):

    class_validation_df = (
        validation_df[
            validation_df[
                "class_id"
            ]
            == class_id
        ]
    )


    actual_count = int(
        class_support[
            index
        ]
    )


    correct_count = int(
        cm[
            index,
            index,
        ]
    )


    validation_block_count = int(
        class_validation_df[
            "spatial_block_id"
        ].nunique()
    )


    class_metric_records.append(
        {
            "class_id": (
                class_id
            ),
            "class_name": (
                CLASS_DEFINITIONS[
                    class_id
                ]
            ),
            "validation_samples": (
                actual_count
            ),
            "validation_spatial_blocks": (
                validation_block_count
            ),
            "correct_predictions": (
                correct_count
            ),
            "precision_user_accuracy": (
                float(
                    class_precision[
                        index
                    ]
                )
            ),
            "recall_producer_accuracy": (
                float(
                    class_recall[
                        index
                    ]
                )
            ),
            "f1_score": (
                float(
                    class_f1[
                        index
                    ]
                )
            ),
        }
    )


class_metrics_df = pd.DataFrame(
    class_metric_records
)


# =============================================================================
# 18. SAMPLE-LEVEL PREDICTIONS
# =============================================================================

prediction_df = (
    validation_df.copy()
)


prediction_df[
    "actual_class_id"
] = y_true


prediction_df[
    "actual_class_name"
] = [
    CLASS_DEFINITIONS[
        int(
            class_id
        )
    ]
    for class_id in y_true
]


prediction_df[
    "predicted_class_id"
] = y_pred


prediction_df[
    "predicted_class_name"
] = [
    CLASS_DEFINITIONS[
        int(
            class_id
        )
    ]
    for class_id in y_pred
]


prediction_df[
    "correct_prediction"
] = (
    y_true
    ==
    y_pred
)


# =============================================================================
# 19. ADD RF PROBABILITIES
# =============================================================================

class_to_probability_column = {}


for probability_index, class_id in enumerate(
    model_classes
):

    probability_column = (
        f"rf_probability_class_{class_id}"
    )

    prediction_df[
        probability_column
    ] = (
        y_proba[
            :,
            probability_index
        ]
    )

    class_to_probability_column[
        class_id
    ] = probability_index


prediction_df[
    "rf_max_probability"
] = np.max(
    y_proba,
    axis=1,
)


true_class_probabilities = []


for row_index, actual_class_id in enumerate(
    y_true
):

    probability_index = (
        class_to_probability_column[
            int(
                actual_class_id
            )
        ]
    )

    true_class_probabilities.append(
        float(
            y_proba[
                row_index,
                probability_index,
            ]
        )
    )


prediction_df[
    "rf_true_class_probability"
] = (
    true_class_probabilities
)


# =============================================================================
# 20. MISCLASSIFICATIONS
# =============================================================================

misclassification_df = (
    prediction_df[
        ~prediction_df[
            "correct_prediction"
        ]
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 21. PROBABILITY DIAGNOSTICS
# =============================================================================

mean_max_probability = safe_mean(
    prediction_df[
        "rf_max_probability"
    ]
)


mean_true_class_probability = safe_mean(
    prediction_df[
        "rf_true_class_probability"
    ]
)


correct_probability_mean = safe_mean(
    prediction_df.loc[
        prediction_df[
            "correct_prediction"
        ],
        "rf_max_probability",
    ]
)


incorrect_probability_mean = safe_mean(
    prediction_df.loc[
        ~prediction_df[
            "correct_prediction"
        ],
        "rf_max_probability",
    ]
)


# =============================================================================
# 22. OVERALL METRICS TABLE
# =============================================================================

overall_metrics_df = pd.DataFrame(
    [
        {
            "validation_samples": (
                len(
                    validation_df
                )
            ),
            "validation_spatial_blocks": (
                len(
                    validation_blocks
                )
            ),
            "correct_predictions": (
                correct_prediction_count
            ),
            "misclassifications": (
                misclassification_count
            ),
            "accuracy": (
                accuracy
            ),
            "balanced_accuracy": (
                balanced_accuracy
            ),
            "macro_precision": (
                macro_precision
            ),
            "macro_recall": (
                macro_recall
            ),
            "macro_f1": (
                macro_f1
            ),
            "weighted_f1": (
                weighted_f1
            ),
            "cohen_kappa": (
                cohen_kappa
            ),
            "multiclass_log_loss": (
                multiclass_log_loss
            ),
            "mean_rf_max_probability": (
                mean_max_probability
            ),
            "mean_rf_true_class_probability": (
                mean_true_class_probability
            ),
            "mean_max_probability_correct_predictions": (
                correct_probability_mean
            ),
            "mean_max_probability_misclassifications": (
                incorrect_probability_mean
            ),
        }
    ]
)


# =============================================================================
# 23. EVALUATION SUMMARY
# =============================================================================

minimum_class_support = int(
    class_metrics_df[
        "validation_samples"
    ].min()
)


minimum_class_blocks = int(
    class_metrics_df[
        "validation_spatial_blocks"
    ].min()
)


evaluation_summary_df = pd.DataFrame(
    [
        {
            "model_name": (
                "RandomForestClassifier"
            ),
            "training_year": (
                2021
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
            "training_spatial_blocks": (
                len(
                    training_blocks
                )
            ),
            "validation_spatial_blocks": (
                len(
                    validation_blocks
                )
            ),
            "shared_spatial_blocks": (
                len(
                    shared_blocks
                )
            ),
            "best_training_spatial_cv_macro_f1": (
                float(
                    model_bundle[
                        "best_cv_macro_f1"
                    ]
                )
            ),
            "holdout_accuracy": (
                accuracy
            ),
            "holdout_balanced_accuracy": (
                balanced_accuracy
            ),
            "holdout_macro_f1": (
                macro_f1
            ),
            "holdout_cohen_kappa": (
                cohen_kappa
            ),
            "minimum_validation_samples_in_any_class": (
                minimum_class_support
            ),
            "minimum_validation_blocks_in_any_class": (
                minimum_class_blocks
            ),
            "validation_used_for_hyperparameter_tuning": (
                False
            ),
            "coordinates_used_as_predictors": (
                False
            ),
            "evaluation_reference_type": (
                "Internal spatial holdout from curated reference dataset"
            ),
            "evaluation_process_status": (
                "PASS"
            ),
        }
    ]
)


# =============================================================================
# 24. SAVE OUTPUTS
# =============================================================================

overall_metrics_df.to_csv(
    OVERALL_METRICS_CSV,
    index=False,
)


class_metrics_df.to_csv(
    CLASS_METRICS_CSV,
    index=False,
)


confusion_matrix_df.to_csv(
    CONFUSION_MATRIX_CSV,
)


prediction_df.to_csv(
    PREDICTIONS_CSV,
    index=False,
)


misclassification_df.to_csv(
    MISCLASSIFICATIONS_CSV,
    index=False,
)


evaluation_summary_df.to_csv(
    EVALUATION_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 25. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("SPATIAL HOLDOUT OVERALL PERFORMANCE")
print("=" * 100)


print(
    f"Validation samples        : "
    f"{len(validation_df)}"
)

print(
    f"Validation blocks         : "
    f"{len(validation_blocks)}"
)

print(
    f"Correct predictions       : "
    f"{correct_prediction_count}"
)

print(
    f"Misclassifications        : "
    f"{misclassification_count}"
)

print(
    f"Accuracy                  : "
    f"{accuracy:.6f}"
)

print(
    f"Balanced Accuracy         : "
    f"{balanced_accuracy:.6f}"
)

print(
    f"Macro Precision           : "
    f"{macro_precision:.6f}"
)

print(
    f"Macro Recall              : "
    f"{macro_recall:.6f}"
)

print(
    f"Macro F1                  : "
    f"{macro_f1:.6f}"
)

print(
    f"Weighted F1               : "
    f"{weighted_f1:.6f}"
)

print(
    f"Cohen's Kappa             : "
    f"{cohen_kappa:.6f}"
)

print(
    f"Multiclass Log Loss       : "
    f"{multiclass_log_loss:.6f}"
)


print("\n" + "=" * 100)
print("PER-CLASS HOLDOUT PERFORMANCE")
print("=" * 100)


print(
    class_metrics_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("CONFUSION MATRIX")
print("=" * 100)


print(
    confusion_matrix_df.to_string()
)


print("\n" + "=" * 100)
print("RF PROBABILITY DIAGNOSTICS")
print("=" * 100)


print(
    f"Mean maximum probability          : "
    f"{mean_max_probability:.6f}"
)

print(
    f"Mean true-class probability       : "
    f"{mean_true_class_probability:.6f}"
)

print(
    f"Mean max probability - correct    : "
    f"{correct_probability_mean:.6f}"
)

print(
    f"Mean max probability - incorrect  : "
    f"{incorrect_probability_mean:.6f}"
)


print("\n" + "=" * 100)
print("STEP 06D SUMMARY")
print("=" * 100)


print(
    f"Training/validation shared IDs     : "
    f"{len(shared_sample_ids)}"
)

print(
    f"Training/validation shared pixels  : "
    f"{len(shared_pixel_ids)}"
)

print(
    f"Training/validation shared blocks  : "
    f"{len(shared_blocks)}"
)

print(
    f"Training spatial CV Macro F1       : "
    f"{float(model_bundle['best_cv_macro_f1']):.6f}"
)

print(
    f"Final holdout Macro F1             : "
    f"{macro_f1:.6f}"
)

print(
    f"Minimum validation class samples   : "
    f"{minimum_class_support}"
)

print(
    f"Minimum validation class blocks    : "
    f"{minimum_class_blocks}"
)

print(
    f"Evaluation process status           : "
    f"PASS"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(OVERALL_METRICS_CSV)}"
)

print(
    f" - {relative_path(CLASS_METRICS_CSV)}"
)

print(
    f" - {relative_path(CONFUSION_MATRIX_CSV)}"
)

print(
    f" - {relative_path(PREDICTIONS_CSV)}"
)

print(
    f" - {relative_path(MISCLASSIFICATIONS_CSV)}"
)

print(
    f" - {relative_path(EVALUATION_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. The spatial validation holdout has now been used for final evaluation."
)

print(
    "2. Do NOT tune the Random Forest using these holdout results."
)

print(
    "3. Precision corresponds to User's Accuracy and recall corresponds to Producer's Accuracy."
)

print(
    "4. Open / Bare has only four validation samples, so its class-specific metric has high uncertainty."
)

print(
    "5. RF probabilities are uncalibrated model-confidence diagnostics, not guaranteed probabilities."
)

print(
    "6. These results measure agreement with the curated internal reference labels, not external ground truth."
)

print(
    "7. The next stage is temporal-transfer assessment before annual 2010-2026 LULC classification."
)

print("=" * 100)