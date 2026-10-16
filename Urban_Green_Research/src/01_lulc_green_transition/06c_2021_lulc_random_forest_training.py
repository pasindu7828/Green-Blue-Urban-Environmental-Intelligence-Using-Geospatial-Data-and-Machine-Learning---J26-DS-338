"""
STEP 06C - 2021 LULC RANDOM FOREST MODEL TRAINING

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Load ONLY the training subset produced by Step 06B.
2. Train a Random Forest LULC classifier.
3. Tune model hyperparameters using spatially grouped cross-validation.
4. Keep complete 1 km spatial blocks together within each CV fold.
5. Use macro F1 as the primary tuning metric because the classes are imbalanced.
6. Save the best model and training diagnostics.

IMPORTANT:
- The Step 06B validation holdout is NOT used in this script.
- The final holdout evaluation will be performed separately in Step 06D.
- Coordinates are NOT model predictors.
- Spatial block IDs are used only for grouped cross-validation.
- Training performance is diagnostic only and is not final model accuracy.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.utils.class_weight import compute_class_weight


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

# Current file:
# Urban_Green_Research/
#   src/
#     01_lulc_green_transition/
#       06c_2021_lulc_random_forest_training.py
#
# parents[2] = Urban_Green_Research

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


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. INPUT
# =============================================================================

TRAINING_CSV = (
    MODEL_DATA_DIR
    / "lulc_2021_training_features.csv"
)


# The validation file is defined only so we can confirm that
# the holdout exists. It is NOT loaded into memory in this script.

VALIDATION_CSV = (
    MODEL_DATA_DIR
    / "lulc_2021_validation_features.csv"
)


# =============================================================================
# 3. OUTPUTS
# =============================================================================

MODEL_FILE = (
    MODEL_DIR
    / "lulc_random_forest_2021.joblib"
)


BEST_PARAMETERS_JSON = (
    REPORT_DIR
    / "06c_rf_best_hyperparameters.json"
)


CV_FOLD_SUMMARY_CSV = (
    REPORT_DIR
    / "06c_rf_spatial_cv_fold_summary.csv"
)


CV_SEARCH_RESULTS_CSV = (
    REPORT_DIR
    / "06c_rf_hyperparameter_search_results.csv"
)


FEATURE_IMPORTANCE_CSV = (
    REPORT_DIR
    / "06c_rf_feature_importance.csv"
)


TRAINING_SUMMARY_CSV = (
    REPORT_DIR
    / "06c_rf_training_summary.csv"
)


TRAINING_CLASS_SUMMARY_CSV = (
    REPORT_DIR
    / "06c_rf_training_class_summary.csv"
)


# =============================================================================
# 4. CONSTANTS
# =============================================================================

RANDOM_STATE = 2026

N_CV_FOLDS = 4

N_RANDOM_SEARCH_ITERATIONS = 40


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
# 5. HELPER
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


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 06C - 2021 LULC RANDOM FOREST MODEL TRAINING")
print("=" * 100)


# =============================================================================
# 7. CHECK INPUTS
# =============================================================================

if not TRAINING_CSV.exists():

    raise FileNotFoundError(
        f"Training dataset not found:\n"
        f"{TRAINING_CSV}"
    )


if not VALIDATION_CSV.exists():

    raise FileNotFoundError(
        f"Step 06B validation holdout not found:\n"
        f"{VALIDATION_CSV}"
    )


print(
    "\nValidation holdout exists."
)

print(
    "IMPORTANT: Validation holdout will NOT be loaded or used in Step 06C."
)


# =============================================================================
# 8. LOAD TRAINING DATA ONLY
# =============================================================================

training_df = pd.read_csv(
    TRAINING_CSV
)


print("\n" + "=" * 100)
print("TRAINING DATA")
print("=" * 100)


print(
    f"Training dataset : "
    f"{relative_path(TRAINING_CSV)}"
)

print(
    f"Training samples : "
    f"{len(training_df)}"
)


# =============================================================================
# 9. VALIDATE TRAINING DATA
# =============================================================================

required_columns = {
    "sample_id",
    "class_id",
    "class_name",
    "spatial_block_id",
    *MODEL_FEATURES,
}


missing_columns = (
    required_columns
    -
    set(
        training_df.columns
    )
)


if missing_columns:

    raise ValueError(
        "Training dataset is missing required columns: "
        f"{sorted(missing_columns)}"
    )


if training_df[
    "sample_id"
].duplicated().any():

    raise ValueError(
        "Duplicate sample IDs exist in the training dataset."
    )


if training_df[
    MODEL_FEATURES
].isna().any().any():

    raise ValueError(
        "Missing model feature values exist."
    )


training_df[
    "class_id"
] = pd.to_numeric(
    training_df[
        "class_id"
    ],
    errors="raise",
).astype(int)


unexpected_classes = (
    set(
        training_df[
            "class_id"
        ].unique()
    )
    -
    set(
        CLASS_DEFINITIONS.keys()
    )
)


if unexpected_classes:

    raise ValueError(
        f"Unexpected class IDs: "
        f"{sorted(unexpected_classes)}"
    )


if set(
    training_df[
        "class_id"
    ].unique()
) != set(
    CLASS_DEFINITIONS.keys()
):

    raise ValueError(
        "Not all four LULC classes are present "
        "in the training dataset."
    )


# =============================================================================
# 10. PREPARE X, y AND GROUPS
# =============================================================================

X = (
    training_df[
        MODEL_FEATURES
    ]
    .copy()
)


y = (
    training_df[
        "class_id"
    ]
    .copy()
)


groups = (
    training_df[
        "spatial_block_id"
    ]
    .copy()
)


print(
    f"Model predictors : "
    f"{len(MODEL_FEATURES)}"
)

print(
    f"Spatial blocks   : "
    f"{groups.nunique()}"
)


print(
    "\nPredictor columns:"
)


for feature in MODEL_FEATURES:

    print(
        f" - {feature}"
    )


# =============================================================================
# 11. TRAINING CLASS DISTRIBUTION
# =============================================================================

print("\n" + "=" * 100)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 100)


class_summary_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    class_mask = (
        y
        == class_id
    )


    sample_count = int(
        class_mask.sum()
    )


    block_count = int(
        training_df.loc[
            class_mask,
            "spatial_block_id",
        ].nunique()
    )


    class_summary_records.append(
        {
            "class_id": class_id,
            "class_name": class_name,
            "training_samples": sample_count,
            "training_spatial_blocks": block_count,
            "training_percent": (
                sample_count
                /
                len(
                    training_df
                )
                *
                100
            ),
        }
    )


class_summary_df = pd.DataFrame(
    class_summary_records
)


print(
    class_summary_df.to_string(
        index=False
    )
)


# =============================================================================
# 12. DISPLAY CLASS WEIGHTS
# =============================================================================

class_ids = np.array(
    sorted(
        CLASS_DEFINITIONS.keys()
    )
)


calculated_class_weights = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=y.to_numpy(),
)


class_weight_display = {
    int(
        class_id
    ): float(
        weight
    )

    for class_id, weight in zip(
        class_ids,
        calculated_class_weights,
    )
}


print(
    "\nApproximate balanced class weights "
    "for the complete training subset:"
)


for class_id, weight in (
    class_weight_display.items()
):

    print(
        f"{class_id} - "
        f"{CLASS_DEFINITIONS[class_id]:<20}: "
        f"{weight:.6f}"
    )


print(
    "\nThe Random Forest itself will use "
    "class_weight='balanced_subsample'."
)


# =============================================================================
# 13. CREATE SPATIAL GROUPED CROSS-VALIDATION
# =============================================================================

print("\n" + "=" * 100)
print("SPATIALLY GROUPED CROSS-VALIDATION")
print("=" * 100)


cv = StratifiedGroupKFold(
    n_splits=N_CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


cv_splits = list(
    cv.split(
        X,
        y,
        groups,
    )
)


cv_fold_records = []


for fold_number, (
    fold_train_indices,
    fold_test_indices,
) in enumerate(
    cv_splits,
    start=1,
):

    fold_train_df = (
        training_df.iloc[
            fold_train_indices
        ]
    )


    fold_test_df = (
        training_df.iloc[
            fold_test_indices
        ]
    )


    fold_train_groups = set(
        fold_train_df[
            "spatial_block_id"
        ]
    )


    fold_test_groups = set(
        fold_test_df[
            "spatial_block_id"
        ]
    )


    shared_groups = (
        fold_train_groups
        &
        fold_test_groups
    )


    train_class_counts = (
        fold_train_df[
            "class_id"
        ]
        .value_counts()
        .reindex(
            CLASS_DEFINITIONS.keys(),
            fill_value=0,
        )
    )


    test_class_counts = (
        fold_test_df[
            "class_id"
        ]
        .value_counts()
        .reindex(
            CLASS_DEFINITIONS.keys(),
            fill_value=0,
        )
    )


    all_classes_train = bool(
        (
            train_class_counts
            >
            0
        ).all()
    )


    all_classes_test = bool(
        (
            test_class_counts
            >
            0
        ).all()
    )


    no_group_overlap = (
        len(
            shared_groups
        )
        == 0
    )


    cv_fold_records.append(
        {
            "fold": fold_number,
            "train_samples": len(
                fold_train_df
            ),
            "test_samples": len(
                fold_test_df
            ),
            "train_blocks": len(
                fold_train_groups
            ),
            "test_blocks": len(
                fold_test_groups
            ),
            "shared_blocks": len(
                shared_groups
            ),
            "all_classes_train": (
                all_classes_train
            ),
            "all_classes_test": (
                all_classes_test
            ),
            "train_green": int(
                train_class_counts.loc[
                    1
                ]
            ),
            "train_built": int(
                train_class_counts.loc[
                    2
                ]
            ),
            "train_open_bare": int(
                train_class_counts.loc[
                    3
                ]
            ),
            "train_water": int(
                train_class_counts.loc[
                    4
                ]
            ),
            "test_green": int(
                test_class_counts.loc[
                    1
                ]
            ),
            "test_built": int(
                test_class_counts.loc[
                    2
                ]
            ),
            "test_open_bare": int(
                test_class_counts.loc[
                    3
                ]
            ),
            "test_water": int(
                test_class_counts.loc[
                    4
                ]
            ),
        }
    )


cv_fold_df = pd.DataFrame(
    cv_fold_records
)


print(
    cv_fold_df.to_string(
        index=False
    )
)


if not cv_fold_df[
    "all_classes_train"
].all():

    raise RuntimeError(
        "At least one CV training fold "
        "is missing an LULC class."
    )


if not cv_fold_df[
    "all_classes_test"
].all():

    raise RuntimeError(
        "At least one CV test fold "
        "is missing an LULC class. "
        "Reduce the number of CV folds."
    )


if not (
    cv_fold_df[
        "shared_blocks"
    ]
    == 0
).all():

    raise RuntimeError(
        "Spatial block leakage detected "
        "between CV train and test partitions."
    )


print(
    "\nSpatial CV validation: PASS"
)


# =============================================================================
# 14. RANDOM FOREST BASE MODEL
# =============================================================================

base_model = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
    bootstrap=True,
    n_jobs=-1,
)


# =============================================================================
# 15. HYPERPARAMETER SEARCH SPACE
# =============================================================================

parameter_distributions = {
    "n_estimators": [
        300,
        500,
        700,
        900,
    ],
    "max_depth": [
        None,
        10,
        15,
        20,
        30,
    ],
    "min_samples_split": [
        2,
        4,
        6,
        10,
    ],
    "min_samples_leaf": [
        1,
        2,
        3,
        4,
    ],
    "max_features": [
        "sqrt",
        "log2",
        0.5,
        0.75,
    ],
    "criterion": [
        "gini",
        "entropy",
        "log_loss",
    ],
}


# =============================================================================
# 16. RANDOMIZED SPATIAL CV SEARCH
# =============================================================================

print("\n" + "=" * 100)
print("RANDOM FOREST HYPERPARAMETER SEARCH")
print("=" * 100)


print(
    f"Random search candidates : "
    f"{N_RANDOM_SEARCH_ITERATIONS}"
)

print(
    f"Spatial CV folds         : "
    f"{N_CV_FOLDS}"
)

print(
    f"Primary selection metric : "
    f"Macro F1"
)


search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=parameter_distributions,
    n_iter=N_RANDOM_SEARCH_ITERATIONS,
    scoring={
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
    },
    refit="f1_macro",
    cv=cv_splits,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
    error_score="raise",
)


search.fit(
    X,
    y,
    groups=groups,
)


best_model = (
    search.best_estimator_
)


best_parameters = (
    search.best_params_
)


best_cv_macro_f1 = float(
    search.best_score_
)


# =============================================================================
# 17. SAVE COMPLETE CV SEARCH RESULTS
# =============================================================================

cv_results_df = pd.DataFrame(
    search.cv_results_
)


selected_result_columns = [
    column
    for column in cv_results_df.columns
    if (
        column.startswith(
            "param_"
        )
        or
        column.startswith(
            "mean_"
        )
        or
        column.startswith(
            "std_"
        )
        or
        column.startswith(
            "rank_"
        )
    )
]


cv_results_export_df = (
    cv_results_df[
        selected_result_columns
    ]
    .sort_values(
        "rank_test_f1_macro"
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 18. TRAINING-SET DIAGNOSTICS
# =============================================================================

# These metrics are NOT final evaluation results.
# They simply verify that the fitted model operates correctly.

training_predictions = (
    best_model.predict(
        X
    )
)


training_accuracy = float(
    accuracy_score(
        y,
        training_predictions,
    )
)


training_balanced_accuracy = float(
    balanced_accuracy_score(
        y,
        training_predictions,
    )
)


training_macro_f1 = float(
    f1_score(
        y,
        training_predictions,
        average="macro",
        zero_division=0,
    )
)


# =============================================================================
# 19. FEATURE IMPORTANCE
# =============================================================================

feature_importance_df = pd.DataFrame(
    {
        "feature": MODEL_FEATURES,
        "importance": (
            best_model.feature_importances_
        ),
    }
).sort_values(
    "importance",
    ascending=False,
).reset_index(
    drop=True
)


feature_importance_df[
    "rank"
] = (
    np.arange(
        1,
        len(
            feature_importance_df
        )
        + 1,
    )
)


feature_importance_df = (
    feature_importance_df[
        [
            "rank",
            "feature",
            "importance",
        ]
    ]
)


# =============================================================================
# 20. TRAINING SUMMARY
# =============================================================================

training_summary_df = pd.DataFrame(
    [
        {
            "training_samples": (
                len(
                    training_df
                )
            ),
            "training_spatial_blocks": (
                groups.nunique()
            ),
            "model_feature_count": (
                len(
                    MODEL_FEATURES
                )
            ),
            "cv_method": (
                "StratifiedGroupKFold"
            ),
            "cv_folds": (
                N_CV_FOLDS
            ),
            "random_search_iterations": (
                N_RANDOM_SEARCH_ITERATIONS
            ),
            "primary_selection_metric": (
                "macro_f1"
            ),
            "best_cv_macro_f1": (
                best_cv_macro_f1
            ),
            "training_accuracy_diagnostic": (
                training_accuracy
            ),
            "training_balanced_accuracy_diagnostic": (
                training_balanced_accuracy
            ),
            "training_macro_f1_diagnostic": (
                training_macro_f1
            ),
            "class_weight_method": (
                "balanced_subsample"
            ),
            "validation_holdout_used_for_tuning": (
                False
            ),
            "coordinates_used_as_predictors": (
                False
            ),
            "overall_status": (
                "PASS"
            ),
        }
    ]
)


# =============================================================================
# 21. MODEL BUNDLE
# =============================================================================

model_bundle = {
    "model": best_model,
    "model_name": (
        "RandomForestClassifier"
    ),
    "training_year": (
        2021
    ),
    "features": (
        MODEL_FEATURES
    ),
    "class_definitions": (
        CLASS_DEFINITIONS
    ),
    "best_parameters": (
        best_parameters
    ),
    "best_cv_macro_f1": (
        best_cv_macro_f1
    ),
    "random_state": (
        RANDOM_STATE
    ),
    "training_samples": (
        len(
            training_df
        )
    ),
    "training_spatial_blocks": (
        groups.nunique()
    ),
    "spatial_cv_folds": (
        N_CV_FOLDS
    ),
    "validation_holdout_used": (
        False
    ),
}


# =============================================================================
# 22. SAVE OUTPUTS
# =============================================================================

joblib.dump(
    model_bundle,
    MODEL_FILE,
)


with open(
    BEST_PARAMETERS_JSON,
    "w",
    encoding="utf-8",
) as json_file:

    json.dump(
        best_parameters,
        json_file,
        indent=4,
    )


cv_fold_df.to_csv(
    CV_FOLD_SUMMARY_CSV,
    index=False,
)


cv_results_export_df.to_csv(
    CV_SEARCH_RESULTS_CSV,
    index=False,
)


feature_importance_df.to_csv(
    FEATURE_IMPORTANCE_CSV,
    index=False,
)


training_summary_df.to_csv(
    TRAINING_SUMMARY_CSV,
    index=False,
)


class_summary_df.to_csv(
    TRAINING_CLASS_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 23. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("BEST RANDOM FOREST CONFIGURATION")
print("=" * 100)


for parameter_name, parameter_value in (
    best_parameters.items()
):

    print(
        f"{parameter_name:<25}: "
        f"{parameter_value}"
    )


print(
    f"\nBest spatial CV Macro F1 : "
    f"{best_cv_macro_f1:.6f}"
)


print("\n" + "=" * 100)
print("TRAINING DIAGNOSTICS - NOT FINAL MODEL ACCURACY")
print("=" * 100)


print(
    f"Training Accuracy          : "
    f"{training_accuracy:.6f}"
)

print(
    f"Training Balanced Accuracy : "
    f"{training_balanced_accuracy:.6f}"
)

print(
    f"Training Macro F1          : "
    f"{training_macro_f1:.6f}"
)


print("\n" + "=" * 100)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 100)


print(
    feature_importance_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 06C SUMMARY")
print("=" * 100)


print(
    f"Training samples             : "
    f"{len(training_df)}"
)

print(
    f"Training spatial blocks      : "
    f"{groups.nunique()}"
)

print(
    f"Spatial CV folds             : "
    f"{N_CV_FOLDS}"
)

print(
    f"Hyperparameter candidates    : "
    f"{N_RANDOM_SEARCH_ITERATIONS}"
)

print(
    f"Validation holdout used      : "
    f"False"
)

print(
    f"Coordinates used as features : "
    f"False"
)

print(
    f"Overall Step 06C status      : "
    f"PASS"
)


print("\nGenerated outputs:")


print(
    f" - {relative_path(MODEL_FILE)}"
)

print(
    f" - {relative_path(BEST_PARAMETERS_JSON)}"
)

print(
    f" - {relative_path(CV_FOLD_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(CV_SEARCH_RESULTS_CSV)}"
)

print(
    f" - {relative_path(FEATURE_IMPORTANCE_CSV)}"
)

print(
    f" - {relative_path(TRAINING_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(TRAINING_CLASS_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. The 79-sample spatial holdout has NOT been used."
)

print(
    "2. Hyperparameter selection used only the 318 training samples."
)

print(
    "3. Cross-validation kept complete spatial blocks together."
)

print(
    "4. Macro F1 was used as the primary model-selection metric."
)

print(
    "5. Training accuracy is diagnostic only and must not be reported as final accuracy."
)

print(
    "6. Final spatial holdout evaluation will be performed in Step 06D."
)

print("=" * 100)