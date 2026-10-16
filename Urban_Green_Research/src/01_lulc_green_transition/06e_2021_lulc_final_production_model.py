"""
STEP 06E - FINAL 2021 LULC PRODUCTION MODEL REFIT

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Load all 397 curated 2021 reference samples.
2. Load the evaluated Step 06C model specification.
3. Reuse EXACTLY the locked Random Forest hyperparameters.
4. Fit a final production Random Forest using all curated samples.
5. Save this model separately for annual 2010-2026 mapping.

IMPORTANT:
- No hyperparameter tuning occurs in this step.
- The Step 06D holdout performance is NOT recalculated.
- The previously reported holdout metrics remain the evaluation evidence.
- This refitted model is for production mapping only.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


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

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "lulc"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
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
# 2. INPUTS
# =============================================================================

ALL_FEATURES_CSV = (
    MODEL_DATA_DIR
    / "lulc_2021_model_features.csv"
)

EVALUATED_MODEL_FILE = (
    MODEL_DIR
    / "lulc_random_forest_2021.joblib"
)


# =============================================================================
# 3. OUTPUTS
# =============================================================================

FINAL_MODEL_FILE = (
    MODEL_DIR
    / "lulc_random_forest_2021_final_production.joblib"
)

FINAL_MODEL_SUMMARY_CSV = (
    REPORT_DIR
    / "06e_final_production_model_summary.csv"
)

FINAL_CLASS_SUMMARY_CSV = (
    REPORT_DIR
    / "06e_final_production_model_class_summary.csv"
)

FINAL_FEATURE_IMPORTANCE_CSV = (
    REPORT_DIR
    / "06e_final_production_model_feature_importance.csv"
)

FINAL_MODEL_PARAMETERS_JSON = (
    REPORT_DIR
    / "06e_final_production_model_parameters.json"
)


# =============================================================================
# 4. CONSTANTS
# =============================================================================

EXPECTED_SAMPLE_COUNT = 397

RANDOM_STATE = 2026


CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}


# =============================================================================
# 5. HELPER
# =============================================================================

def relative_path(path: Path) -> str:
    """Return project-relative path when possible."""

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
print("STEP 06E - FINAL 2021 LULC PRODUCTION MODEL REFIT")
print("=" * 100)


for required_file in [
    ALL_FEATURES_CSV,
    EVALUATED_MODEL_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required input not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 7. LOAD EVALUATED MODEL SPECIFICATION
# =============================================================================

evaluated_bundle = joblib.load(
    EVALUATED_MODEL_FILE
)


required_bundle_keys = {
    "model",
    "features",
    "best_parameters",
    "best_cv_macro_f1",
}


missing_keys = (
    required_bundle_keys
    -
    set(
        evaluated_bundle.keys()
    )
)


if missing_keys:

    raise ValueError(
        f"Evaluated model bundle is missing keys: "
        f"{sorted(missing_keys)}"
    )


MODEL_FEATURES = list(
    evaluated_bundle[
        "features"
    ]
)


locked_parameters = dict(
    evaluated_bundle[
        "best_parameters"
    ]
)


print(
    "\nLocked evaluated model specification:"
)


for parameter_name, parameter_value in (
    locked_parameters.items()
):

    print(
        f"{parameter_name:<25}: "
        f"{parameter_value}"
    )


print(
    f"\nPreviously recorded spatial CV Macro F1 : "
    f"{float(evaluated_bundle['best_cv_macro_f1']):.6f}"
)


# =============================================================================
# 8. LOAD ALL CURATED 2021 FEATURES
# =============================================================================

feature_df = pd.read_csv(
    ALL_FEATURES_CSV
)


print("\n" + "=" * 100)
print("FINAL PRODUCTION TRAINING DATA")
print("=" * 100)


print(
    f"Input dataset  : "
    f"{relative_path(ALL_FEATURES_CSV)}"
)

print(
    f"Total samples  : "
    f"{len(feature_df)}"
)


if len(
    feature_df
) != EXPECTED_SAMPLE_COUNT:

    raise ValueError(
        f"Expected {EXPECTED_SAMPLE_COUNT} samples, "
        f"but found {len(feature_df)}."
    )


required_columns = {
    "sample_id",
    "class_id",
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
        f"Input feature dataset is missing columns: "
        f"{sorted(missing_columns)}"
    )


if feature_df[
    "sample_id"
].duplicated().any():

    raise ValueError(
        "Duplicate sample IDs exist."
    )


if feature_df[
    MODEL_FEATURES
].isna().any().any():

    raise ValueError(
        "Missing predictor values exist."
    )


feature_df[
    "class_id"
] = pd.to_numeric(
    feature_df[
        "class_id"
    ],
    errors="raise",
).astype(int)


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
# 9. CLASS DISTRIBUTION
# =============================================================================

class_summary_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    count = int(
        (
            feature_df[
                "class_id"
            ]
            == class_id
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
            "production_training_samples": (
                count
            ),
            "percent_of_training_data": (
                count
                /
                len(
                    feature_df
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
# 10. PREPARE FINAL TRAINING MATRICES
# =============================================================================

X = (
    feature_df[
        MODEL_FEATURES
    ]
    .copy()
)


y = (
    feature_df[
        "class_id"
    ]
    .copy()
)


# =============================================================================
# 11. BUILD FINAL MODEL USING LOCKED PARAMETERS
# =============================================================================

print("\n" + "=" * 100)
print("REFITTING FINAL PRODUCTION MODEL")
print("=" * 100)


final_model = RandomForestClassifier(
    **locked_parameters,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
    bootstrap=True,
    n_jobs=-1,
)


final_model.fit(
    X,
    y,
)


print(
    "Final production model fitted successfully."
)


# =============================================================================
# 12. TRAINING-ONLY SANITY CHECK
# =============================================================================

training_predictions = (
    final_model.predict(
        X
    )
)


training_correct = int(
    np.sum(
        training_predictions
        ==
        y.to_numpy()
    )
)


training_accuracy_diagnostic = float(
    training_correct
    /
    len(
        y
    )
)


print(
    f"Training-only diagnostic accuracy : "
    f"{training_accuracy_diagnostic:.6f}"
)

print(
    "This value is NOT a validation or accuracy-assessment result."
)


# =============================================================================
# 13. FEATURE IMPORTANCE
# =============================================================================

feature_importance_df = pd.DataFrame(
    {
        "feature": (
            MODEL_FEATURES
        ),
        "importance": (
            final_model.feature_importances_
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
] = np.arange(
    1,
    len(
        feature_importance_df
    )
    + 1,
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
# 14. BUILD PRODUCTION MODEL BUNDLE
# =============================================================================

production_bundle = {
    "model": (
        final_model
    ),
    "model_name": (
        "RandomForestClassifier"
    ),
    "model_role": (
        "Final production LULC mapping model"
    ),
    "training_year": (
        2021
    ),
    "training_samples": (
        len(
            feature_df
        )
    ),
    "features": (
        MODEL_FEATURES
    ),
    "class_definitions": (
        CLASS_DEFINITIONS
    ),
    "locked_hyperparameters": (
        locked_parameters
    ),
    "random_state": (
        RANDOM_STATE
    ),
    "class_weight": (
        "balanced_subsample"
    ),
    "hyperparameters_retuned_after_holdout": (
        False
    ),
    "holdout_reused_for_accuracy_estimation": (
        False
    ),
    "source_evaluation_model": (
        relative_path(
            EVALUATED_MODEL_FILE
        )
    ),
    "recorded_evaluation_cv_macro_f1": (
        float(
            evaluated_bundle[
                "best_cv_macro_f1"
            ]
        )
    ),
}


# =============================================================================
# 15. SUMMARY
# =============================================================================

summary_df = pd.DataFrame(
    [
        {
            "model_name": (
                "RandomForestClassifier"
            ),
            "model_role": (
                "Final production LULC mapping model"
            ),
            "training_year": (
                2021
            ),
            "production_training_samples": (
                len(
                    feature_df
                )
            ),
            "model_features": (
                len(
                    MODEL_FEATURES
                )
            ),
            "hyperparameters_locked_from_step_06c": (
                True
            ),
            "hyperparameters_retuned_after_holdout": (
                False
            ),
            "holdout_reused_for_model_selection": (
                False
            ),
            "training_accuracy_diagnostic": (
                training_accuracy_diagnostic
            ),
            "reported_accuracy_source": (
                "Step 06D pre-refit spatial holdout evaluation"
            ),
            "production_model_status": (
                "READY"
            ),
        }
    ]
)


# =============================================================================
# 16. SAVE OUTPUTS
# =============================================================================

joblib.dump(
    production_bundle,
    FINAL_MODEL_FILE,
)


summary_df.to_csv(
    FINAL_MODEL_SUMMARY_CSV,
    index=False,
)


class_summary_df.to_csv(
    FINAL_CLASS_SUMMARY_CSV,
    index=False,
)


feature_importance_df.to_csv(
    FINAL_FEATURE_IMPORTANCE_CSV,
    index=False,
)


with open(
    FINAL_MODEL_PARAMETERS_JSON,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "locked_hyperparameters": (
                locked_parameters
            ),
            "random_state": (
                RANDOM_STATE
            ),
            "class_weight": (
                "balanced_subsample"
            ),
            "training_samples": (
                len(
                    feature_df
                )
            ),
            "features": (
                MODEL_FEATURES
            ),
        },
        file,
        indent=4,
    )


# =============================================================================
# 17. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("FINAL PRODUCTION FEATURE IMPORTANCE")
print("=" * 100)


print(
    feature_importance_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 06E SUMMARY")
print("=" * 100)


print(
    f"Curated samples used          : "
    f"{len(feature_df)}"
)

print(
    f"Model predictors              : "
    f"{len(MODEL_FEATURES)}"
)

print(
    f"Hyperparameters retuned       : "
    f"False"
)

print(
    f"Holdout reused for selection  : "
    f"False"
)

print(
    f"Production model status       : "
    f"READY"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(FINAL_MODEL_FILE)}"
)

print(
    f" - {relative_path(FINAL_MODEL_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(FINAL_CLASS_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(FINAL_FEATURE_IMPORTANCE_CSV)}"
)

print(
    f" - {relative_path(FINAL_MODEL_PARAMETERS_JSON)}"
)


print("\nIMPORTANT:")

print(
    "1. This production model uses all 397 curated 2021 reference samples."
)

print(
    "2. Hyperparameters were copied unchanged from the evaluated Step 06C model."
)

print(
    "3. Step 06D holdout results remain the official internal spatial evaluation evidence."
)

print(
    "4. The production model must NOT be evaluated again on the former holdout as if it were independent."
)

print(
    "5. This model will be used for temporal-transfer diagnostics and annual LULC mapping."
)

print("=" * 100)