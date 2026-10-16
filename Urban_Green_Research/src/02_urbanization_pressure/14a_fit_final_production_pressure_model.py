"""
STEP 14A - FIT FINAL PRODUCTION URBANIZATION-PRESSURE MODEL

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Fit the locked XGBoost configuration on ALL available labelled history.
2. Use all final causal labelled origins from 2012-2022.
3. Use the frozen 55-predictor set.
4. Preserve the locked operating threshold from Step 13D.
5. Save the final production model for future urbanization-pressure inference.

Important distinction:
- Step 13E remains the OFFICIAL independent performance evaluation.
- This production refit uses the former 2022 test data only AFTER
  final evaluation was completed and frozen.
- No new validation accuracy is calculated from the production training data.
- The production model score is an urbanization-pressure score,
  NOT a calibrated real-world probability.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

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

LOCKED_MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "model_development"
    / "13d_locked_model_decision.json"
)

FINAL_EVALUATION_SUMMARY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "13e_final_2022_evaluation_summary.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "urbanization_pressure"
    / "production"
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
# 2. OUTPUTS
# =============================================================================

PRODUCTION_MODEL_FILE = (
    MODEL_DIR
    / "14a_xgboost_final_production.joblib"
)

TRAINING_SUMMARY_CSV = (
    REPORT_DIR
    / "14a_production_training_summary.csv"
)

QA_CSV = (
    REPORT_DIR
    / "14a_production_model_qa.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "14a_production_model_summary.csv"
)


# =============================================================================
# 3. EXPECTED VALUES
# =============================================================================

EXPECTED_ROWS = 51310

EXPECTED_POSITIVES = 844

EXPECTED_FEATURES = 55

EXPECTED_ORIGIN_MIN = 2012

EXPECTED_ORIGIN_MAX = 2022

EXPECTED_LOCKED_THRESHOLD = 0.715

RANDOM_STATE = 2026


# =============================================================================
# 4. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """Return project-relative path where possible."""

    try:

        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:

        return str(path)


def calculate_scale_pos_weight(
    y,
):
    """Calculate production imbalance weight."""

    y = np.asarray(
        y,
        dtype=np.uint8,
    )

    positives = int(
        np.sum(
            y
            ==
            1
        )
    )

    negatives = int(
        np.sum(
            y
            ==
            0
        )
    )

    if positives <= 0:

        raise RuntimeError(
            "Production training data contains zero positive cases."
        )

    return float(
        negatives
        /
        positives
    )


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 14A - FIT FINAL PRODUCTION URBANIZATION-PRESSURE MODEL")
print("=" * 100)


for required_file in [
    DATASET_FILE,
    FEATURE_LIST_FILE,
    LOCKED_MODEL_FILE,
    FINAL_EVALUATION_SUMMARY_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 6. VERIFY FINAL EVALUATION WAS COMPLETED FIRST
# =============================================================================

final_evaluation_df = pd.read_csv(
    FINAL_EVALUATION_SUMMARY_FILE
)


if len(
    final_evaluation_df
) != 1:

    raise RuntimeError(
        "Unexpected Step 13E final-evaluation summary structure."
    )


final_evaluation_status = str(
    final_evaluation_df.loc[
        0,
        "overall_step13e_status",
    ]
)


if final_evaluation_status != "PASS":

    raise RuntimeError(
        "Step 13E final evaluation has not passed. "
        "Production refit must not continue."
    )


print(
    "\nStep 13E final evaluation status : "
    f"{final_evaluation_status}"
)

print(
    "Official final evaluation frozen  : True"
)


# =============================================================================
# 7. LOAD LOCKED MODEL DECISION
# =============================================================================

with open(
    LOCKED_MODEL_FILE,
    "r",
    encoding="utf-8",
) as file:

    locked_decision = json.load(
        file
    )


selected_model = (
    locked_decision[
        "selected_model"
    ]
)


selected_candidate = (
    locked_decision[
        "selected_candidate_id"
    ]
)


selected_parameters = (
    locked_decision[
        "parameters"
    ]
)


locked_threshold = float(
    locked_decision[
        "operating_threshold"
    ]
)


if selected_model != "XGBoost":

    raise RuntimeError(
        f"Expected locked XGBoost model, "
        f"found {selected_model}."
    )


if not np.isclose(
    locked_threshold,
    EXPECTED_LOCKED_THRESHOLD,
    atol=0.0,
    rtol=0.0,
):

    raise RuntimeError(
        f"Expected locked threshold "
        f"{EXPECTED_LOCKED_THRESHOLD}, "
        f"found {locked_threshold}."
    )


print("\nLOCKED MODEL CONFIGURATION")

print(
    f"Model                  : "
    f"{selected_model}"
)

print(
    f"Candidate              : "
    f"{selected_candidate}"
)

print(
    f"Operating threshold    : "
    f"{locked_threshold:.3f}"
)

print(
    "Hyperparameters changed: False"
)

print(
    "Threshold changed      : False"
)


# =============================================================================
# 8. LOAD FROZEN DATASET / FEATURE LIST
# =============================================================================

df = pd.read_csv(
    DATASET_FILE
)


feature_list_df = pd.read_csv(
    FEATURE_LIST_FILE
)


if len(
    df
) != EXPECTED_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} rows, "
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
        f"Expected {EXPECTED_FEATURES} frozen features, "
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
    feature_name
    for feature_name in feature_columns
    if feature_name not in df.columns
]


if missing_features:

    raise RuntimeError(
        "Frozen production predictors are missing:\n"
        +
        "\n".join(
            missing_features
        )
    )


if df[
    feature_columns
].isna().any().any():

    raise RuntimeError(
        "Missing values detected in frozen production predictors."
    )


if np.isinf(
    df[
        feature_columns
    ].to_numpy(
        dtype=float
    )
).any():

    raise RuntimeError(
        "Infinite values detected in frozen production predictors."
    )


origin_min = int(
    df[
        "origin_year"
    ].min()
)


origin_max = int(
    df[
        "origin_year"
    ].max()
)


if (
    origin_min
    !=
    EXPECTED_ORIGIN_MIN
    or
    origin_max
    !=
    EXPECTED_ORIGIN_MAX
):

    raise RuntimeError(
        f"Unexpected production origin-year range: "
        f"{origin_min}-{origin_max}."
    )


print("\n" + "=" * 100)
print("PRODUCTION TRAINING DATA")
print("=" * 100)


print(
    f"Training origins       : "
    f"{origin_min}-{origin_max}"
)

print(
    f"Training rows          : "
    f"{len(df):,}"
)

print(
    f"Positive rows          : "
    f"{positive_count:,}"
)

print(
    f"Negative rows          : "
    f"{len(df) - positive_count:,}"
)

print(
    f"Frozen predictors      : "
    f"{len(feature_columns)}"
)


# =============================================================================
# 9. PREPARE PRODUCTION TRAINING MATRIX
# =============================================================================

X = (
    df[
        feature_columns
    ]
    .to_numpy(
        dtype=np.float32
    )
)


y = (
    df[
        "urban_conversion_target"
    ]
    .to_numpy(
        dtype=np.uint8
    )
)


production_scale_pos_weight = (
    calculate_scale_pos_weight(
        y
    )
)


print(
    f"Production scale_pos_weight : "
    f"{production_scale_pos_weight:.6f}"
)


# =============================================================================
# 10. FIT FINAL PRODUCTION MODEL
# =============================================================================

production_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=0,
    scale_pos_weight=production_scale_pos_weight,
    **selected_parameters,
)


print(
    "\nFitting production model on all "
    "available labelled history..."
)


production_model.fit(
    X,
    y,
)


print(
    "Production model fitted."
)


# =============================================================================
# 11. SAVE MODEL BUNDLE
# =============================================================================

production_bundle = {
    "model": (
        production_model
    ),
    "model_name": (
        selected_model
    ),
    "candidate_id": (
        selected_candidate
    ),
    "parameters": (
        selected_parameters
    ),
    "scale_pos_weight": (
        production_scale_pos_weight
    ),
    "feature_names": (
        feature_columns
    ),
    "operating_threshold": (
        locked_threshold
    ),
    "training_origin_years": (
        "2012-2022"
    ),
    "training_rows": (
        len(
            df
        )
    ),
    "training_positive_rows": (
        positive_count
    ),
    "target_definition": (
        "Stable non-built to persistent majority-Built "
        "conversion within next 3 years"
    ),
    "target_horizon_years": (
        3
    ),
    "official_evaluation_source": (
        "Step 13E one-time final 2022 temporal evaluation"
    ),
    "raw_model_score_is_calibrated_probability": (
        False
    ),
    "model_role": (
        "FINAL_PRODUCTION_URBANIZATION_PRESSURE_MODEL"
    ),
}


joblib.dump(
    production_bundle,
    PRODUCTION_MODEL_FILE,
)


# =============================================================================
# 12. SERIALIZATION QA
# =============================================================================

loaded_bundle = joblib.load(
    PRODUCTION_MODEL_FILE
)


loaded_model = (
    loaded_bundle[
        "model"
    ]
)


sample_size = min(
    250,
    len(
        X
    ),
)


original_sample_scores = (
    production_model.predict_proba(
        X[
            :sample_size
        ]
    )[
        :,
        1
    ]
)


loaded_sample_scores = (
    loaded_model.predict_proba(
        X[
            :sample_size
        ]
    )[
        :,
        1
    ]
)


serialization_max_score_difference = float(
    np.max(
        np.abs(
            original_sample_scores
            -
            loaded_sample_scores
        )
    )
)


serialization_valid = (
    serialization_max_score_difference
    <=
    1e-12
)


sample_probability_bounds_valid = bool(
    np.all(
        (
            loaded_sample_scores
            >=
            0.0
        )
        &
        (
            loaded_sample_scores
            <=
            1.0
        )
    )
)


# =============================================================================
# 13. TRAINING SUMMARY
# =============================================================================

training_summary_df = pd.DataFrame(
    [
        {
            "production_model": (
                selected_model
            ),
            "candidate_id": (
                selected_candidate
            ),
            "training_origin_start": (
                origin_min
            ),
            "training_origin_end": (
                origin_max
            ),
            "training_rows": (
                len(
                    df
                )
            ),
            "positive_rows": (
                positive_count
            ),
            "negative_rows": (
                len(
                    df
                )
                -
                positive_count
            ),
            "positive_rate_percent": (
                positive_count
                /
                len(
                    df
                )
                *
                100
            ),
            "feature_count": (
                len(
                    feature_columns
                )
            ),
            "scale_pos_weight": (
                production_scale_pos_weight
            ),
            "operating_threshold": (
                locked_threshold
            ),
            "training_performance_reported_as_validation_accuracy": (
                False
            ),
            "official_performance_source": (
                "Step 13E"
            ),
        }
    ]
)


# =============================================================================
# 14. QA
# =============================================================================

row_count_valid = (
    len(
        df
    )
    ==
    EXPECTED_ROWS
)


positive_count_valid = (
    positive_count
    ==
    EXPECTED_POSITIVES
)


feature_count_valid = (
    len(
        feature_columns
    )
    ==
    EXPECTED_FEATURES
)


origin_range_valid = (
    origin_min
    ==
    EXPECTED_ORIGIN_MIN
    and
    origin_max
    ==
    EXPECTED_ORIGIN_MAX
)


locked_model_valid = (
    selected_model
    ==
    "XGBoost"
)


locked_threshold_valid = bool(
    np.isclose(
        locked_threshold,
        EXPECTED_LOCKED_THRESHOLD,
        atol=0.0,
        rtol=0.0,
    )
)


production_file_exists = (
    PRODUCTION_MODEL_FILE.exists()
)


final_evaluation_reused_for_tuning = False

new_validation_performed = False

training_accuracy_reported = False


overall_status = (
    "PASS"
    if all(
        [
            row_count_valid,
            positive_count_valid,
            feature_count_valid,
            origin_range_valid,
            locked_model_valid,
            locked_threshold_valid,
            production_file_exists,
            serialization_valid,
            sample_probability_bounds_valid,
            not final_evaluation_reused_for_tuning,
            not new_validation_performed,
            not training_accuracy_reported,
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
            "training_rows_valid": (
                row_count_valid
            ),
            "positive_count_valid": (
                positive_count_valid
            ),
            "frozen_feature_count_valid": (
                feature_count_valid
            ),
            "training_origin_range_valid": (
                origin_range_valid
            ),
            "locked_model_unchanged": (
                locked_model_valid
            ),
            "locked_threshold_unchanged": (
                locked_threshold_valid
            ),
            "production_model_file_exists": (
                production_file_exists
            ),
            "serialization_valid": (
                serialization_valid
            ),
            "serialization_max_score_difference": (
                serialization_max_score_difference
            ),
            "sample_score_bounds_valid": (
                sample_probability_bounds_valid
            ),
            "step13e_used_for_retuning": (
                final_evaluation_reused_for_tuning
            ),
            "new_validation_performed": (
                new_validation_performed
            ),
            "training_accuracy_reported_as_performance": (
                training_accuracy_reported
            ),
            "overall_step14a_status": (
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
                "Final production model refit"
            ),
            "production_model": (
                selected_model
            ),
            "production_candidate": (
                selected_candidate
            ),
            "training_origins": (
                "2012-2022"
            ),
            "all_available_labelled_history_used": (
                True
            ),
            "training_rows": (
                len(
                    df
                )
            ),
            "positive_training_rows": (
                positive_count
            ),
            "frozen_predictor_count": (
                len(
                    feature_columns
                )
            ),
            "locked_operating_threshold": (
                locked_threshold
            ),
            "official_evaluation_remains_step13e": (
                True
            ),
            "production_refit_has_new_independent_accuracy": (
                False
            ),
            "raw_score_calibrated_probability": (
                False
            ),
            "ready_for_current_pressure_inference": (
                overall_status
                ==
                "PASS"
            ),
            "overall_step14a_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 17. SAVE REPORTS
# =============================================================================

training_summary_df.to_csv(
    TRAINING_SUMMARY_CSV,
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
print("FINAL PRODUCTION MODEL")
print("=" * 100)


print(
    f"Model                         : "
    f"{selected_model}"
)

print(
    f"Candidate                     : "
    f"{selected_candidate}"
)

print(
    f"Training origins              : "
    f"{origin_min}-{origin_max}"
)

print(
    f"Training rows                 : "
    f"{len(df):,}"
)

print(
    f"Positive training rows        : "
    f"{positive_count:,}"
)

print(
    f"Frozen predictors             : "
    f"{len(feature_columns)}"
)

print(
    f"Production scale_pos_weight   : "
    f"{production_scale_pos_weight:.6f}"
)

print(
    f"Locked operating threshold    : "
    f"{locked_threshold:.3f}"
)

print(
    f"Official evaluation source    : "
    f"Step 13E"
)


print("\n" + "=" * 100)
print("STEP 14A QA")
print("=" * 100)


print(
    f"Training row count valid              : "
    f"{row_count_valid}"
)

print(
    f"Positive count valid                  : "
    f"{positive_count_valid}"
)

print(
    f"Frozen feature count valid            : "
    f"{feature_count_valid}"
)

print(
    f"Training origin range 2012-2022       : "
    f"{origin_range_valid}"
)

print(
    f"Locked model unchanged                : "
    f"{locked_model_valid}"
)

print(
    f"Locked threshold unchanged            : "
    f"{locked_threshold_valid}"
)

print(
    f"Production model saved                : "
    f"{production_file_exists}"
)

print(
    f"Saved/reloaded model scores identical : "
    f"{serialization_valid}"
)

print(
    f"Max serialization score difference    : "
    f"{serialization_max_score_difference:.12f}"
)

print(
    f"Sample scores inside [0,1]            : "
    f"{sample_probability_bounds_valid}"
)

print(
    f"Step 13E reused for tuning            : "
    f"{final_evaluation_reused_for_tuning}"
)

print(
    f"New validation performed              : "
    f"{new_validation_performed}"
)

print(
    f"Training accuracy reported            : "
    f"{training_accuracy_reported}"
)

print(
    f"Overall Step 14A status               : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(PRODUCTION_MODEL_FILE)}"
)

print(
    f" - {relative_path(TRAINING_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(QA_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. Step 13E remains the official independent final performance evaluation."
)

print(
    "2. The production model now uses all labelled origins from 2012-2022."
)

print(
    "3. No training-set accuracy is reported as model performance."
)

print(
    "4. The locked XGBoost hyperparameters and operating threshold were not changed."
)

print(
    "5. The production raw score is an urbanization-pressure score, not a calibrated probability."
)

print(
    "6. The next step will construct a leakage-safe 2025 inference feature table using 2023, 2024 and 2025 RAW LULC."
)

print(
    "7. The 2025 pressure origin will represent pressure for persistent conversion during 2026-2028."
)

print("=" * 100)