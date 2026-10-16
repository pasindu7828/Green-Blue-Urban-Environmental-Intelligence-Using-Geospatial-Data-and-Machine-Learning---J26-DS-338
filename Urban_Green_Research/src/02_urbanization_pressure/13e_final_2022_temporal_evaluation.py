"""
STEP 13E - ONE-TIME FINAL 2022 TEMPORAL EVALUATION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Perform the ONE-TIME final temporal evaluation on origin year 2022.
2. Use the model and threshold locked BEFORE seeing the final test.
3. Train only on final training origins 2012-2019.
4. Keep 2020-2021 embargoed.
5. Evaluate only the already-selected XGBoost model.
6. Do NOT compare alternative models on the final test.
7. Do NOT retune hyperparameters or threshold after this evaluation.

Locked development decision:
- Model: XGBoost
- Candidate: selected in Step 13D
- Threshold: selected from temporal development OOF in Step 13D

Final back-test interpretation:
- 2022 predictors use information available through 2022.
- 2022 target describes persistent majority-Built conversion
  during 2023-2025.

IMPORTANT:
- Final test results must not be used for additional tuning.
- Raw model scores are urbanization-pressure scores and are NOT
  claimed to be perfectly calibrated probabilities.
- This fitted model is the FINAL-EVALUATION model, not yet the
  later production model trained on all available labelled history.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
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

LOCKED_MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "model_development"
    / "13d_locked_model_decision.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "final_evaluation"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "urbanization_pressure"
    / "final_evaluation"
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

TEST_PREDICTIONS_CSV = (
    OUTPUT_DIR
    / "13e_final_2022_test_predictions.csv"
)

FINAL_METRICS_CSV = (
    REPORT_DIR
    / "13e_final_2022_test_metrics.csv"
)

SCORE_SUMMARY_CSV = (
    REPORT_DIR
    / "13e_final_2022_score_summary.csv"
)

QA_CSV = (
    REPORT_DIR
    / "13e_final_2022_evaluation_qa.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "13e_final_2022_evaluation_summary.csv"
)

EVALUATION_MODEL_FILE = (
    MODEL_DIR
    / "13e_xgboost_2022_final_evaluation.joblib"
)


# =============================================================================
# 3. EXPECTED VALUES
# =============================================================================

EXPECTED_DATASET_ROWS = 51310
EXPECTED_POSITIVE_ROWS = 844
EXPECTED_FEATURES = 55

EXPECTED_TRAIN_ROWS = 37801
EXPECTED_TRAIN_POSITIVES = 725

EXPECTED_EMBARGO_ROWS = 8912

EXPECTED_TEST_ROWS = 4597
EXPECTED_TEST_POSITIVES = 45

FINAL_TEST_YEAR = 2022

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


def calculate_scale_pos_weight(y_train):
    """Calculate imbalance weight from final training data only."""

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
            "Final training set contains zero positive cases."
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
print("STEP 13E - ONE-TIME FINAL 2022 TEMPORAL EVALUATION")
print("=" * 100)


for required_file in [
    DATASET_FILE,
    FEATURE_LIST_FILE,
    BASE_ASSIGNMENT_FILE,
    LOCKED_MODEL_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 6. LOAD LOCKED MODEL DECISION
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
        f"Expected locked XGBoost model, found "
        f"{selected_model}."
    )


if not (
    0.0
    <
    locked_threshold
    <
    1.0
):

    raise RuntimeError(
        "Locked threshold is outside (0,1)."
    )


print("\nLOCKED BEFORE FINAL TEST")

print(
    f"Selected model      : "
    f"{selected_model}"
)

print(
    f"Selected candidate  : "
    f"{selected_candidate}"
)

print(
    f"Locked threshold    : "
    f"{locked_threshold:.3f}"
)

print(
    "Threshold retuned   : False"
)

print(
    "Alternative model compared on test : False"
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


assignment_df = pd.read_csv(
    BASE_ASSIGNMENT_FILE
)


if len(
    df
) != EXPECTED_DATASET_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_DATASET_ROWS:,} rows, "
        f"found {len(df):,}."
    )


if int(
    df[
        "urban_conversion_target"
    ].sum()
) != EXPECTED_POSITIVE_ROWS:

    raise RuntimeError(
        "Overall positive target count mismatch."
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
        "Frozen features missing from modelling dataset:\n"
        +
        "\n".join(
            missing_features
        )
    )


if df[
    feature_columns
].isna().any().any():

    raise RuntimeError(
        "Missing values detected in frozen predictors."
    )


if np.isinf(
    df[
        feature_columns
    ].to_numpy(
        dtype=float
    )
).any():

    raise RuntimeError(
        "Infinite values detected in frozen predictors."
    )


# =============================================================================
# 8. CREATE / VERIFY ROW ID
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
        "Duplicate row IDs detected."
    )


assignment_lookup = (
    assignment_df[
        [
            "row_id",
            "spatial_block_1km",
            "spatial_fold",
            "final_temporal_test_role",
        ]
    ]
    .drop_duplicates(
        subset="row_id"
    )
)


df = df.merge(
    assignment_lookup,
    on="row_id",
    how="left",
    validate="one_to_one",
)


if df[
    "final_temporal_test_role"
].isna().any():

    raise RuntimeError(
        "Some modelling rows are missing final-test roles."
    )


# =============================================================================
# 9. CREATE FINAL TRAIN / EMBARGO / TEST
# =============================================================================

train_df = (
    df[
        df[
            "final_temporal_test_role"
        ]
        ==
        "TRAIN"
    ]
    .copy()
)


embargo_df = (
    df[
        df[
            "final_temporal_test_role"
        ]
        ==
        "EMBARGO"
    ]
    .copy()
)


test_df = (
    df[
        df[
            "final_temporal_test_role"
        ]
        ==
        "TEST"
    ]
    .copy()
)


if len(
    train_df
) != EXPECTED_TRAIN_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_TRAIN_ROWS:,} final training rows, "
        f"found {len(train_df):,}."
    )


if int(
    train_df[
        "urban_conversion_target"
    ].sum()
) != EXPECTED_TRAIN_POSITIVES:

    raise RuntimeError(
        "Final training positive count mismatch."
    )


if len(
    embargo_df
) != EXPECTED_EMBARGO_ROWS:

    raise RuntimeError(
        "Final embargo row count mismatch."
    )


if len(
    test_df
) != EXPECTED_TEST_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_TEST_ROWS:,} test rows, "
        f"found {len(test_df):,}."
    )


if int(
    test_df[
        "urban_conversion_target"
    ].sum()
) != EXPECTED_TEST_POSITIVES:

    raise RuntimeError(
        "Final test positive count mismatch."
    )


if not (
    test_df[
        "origin_year"
    ]
    ==
    FINAL_TEST_YEAR
).all():

    raise RuntimeError(
        "Final test contains an unexpected origin year."
    )


train_test_row_overlap = len(
    set(
        train_df[
            "row_id"
        ]
    )
    &
    set(
        test_df[
            "row_id"
        ]
    )
)


if train_test_row_overlap != 0:

    raise RuntimeError(
        "Train/test row overlap detected."
    )


print("\n" + "=" * 100)
print("FINAL TEMPORAL SPLIT")
print("=" * 100)


print(
    f"Train origins        : "
    f"{train_df['origin_year'].min()}-"
    f"{train_df['origin_year'].max()}"
)

print(
    f"Train rows           : "
    f"{len(train_df):,}"
)

print(
    f"Train positives      : "
    f"{train_df['urban_conversion_target'].sum():,}"
)

print(
    f"Embargo origins      : "
    f"{embargo_df['origin_year'].min()}-"
    f"{embargo_df['origin_year'].max()}"
)

print(
    f"Embargo rows         : "
    f"{len(embargo_df):,}"
)

print(
    f"Final test origin    : "
    f"{FINAL_TEST_YEAR}"
)

print(
    f"Final test rows      : "
    f"{len(test_df):,}"
)

print(
    f"Final test positives : "
    f"{test_df['urban_conversion_target'].sum():,}"
)


# =============================================================================
# 10. PREPARE FINAL TRAINING DATA
# =============================================================================

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


X_test = (
    test_df[
        feature_columns
    ]
    .to_numpy(
        dtype=np.float32
    )
)


y_test = (
    test_df[
        "urban_conversion_target"
    ]
    .to_numpy(
        dtype=np.uint8
    )
)


final_scale_pos_weight = (
    calculate_scale_pos_weight(
        y_train
    )
)


print(
    f"\nFinal training scale_pos_weight : "
    f"{final_scale_pos_weight:.6f}"
)


# =============================================================================
# 11. FIT LOCKED MODEL ON FINAL TRAINING SET
# =============================================================================

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=0,
    scale_pos_weight=final_scale_pos_weight,
    **selected_parameters,
)


print("\nFitting locked XGBoost model...")


model.fit(
    X_train,
    y_train,
)


print(
    "Model fitted."
)


# =============================================================================
# 12. ONE-TIME FINAL TEST SCORING
# =============================================================================

print("\nRunning ONE-TIME 2022 final test...")


test_scores = (
    model.predict_proba(
        X_test
    )[
        :,
        1
    ]
)


test_predictions = (
    test_scores
    >=
    locked_threshold
).astype(
    np.uint8
)


# =============================================================================
# 13. FINAL THRESHOLD-FREE METRICS
# =============================================================================

test_prevalence = float(
    np.mean(
        y_test
    )
)


pr_auc = float(
    average_precision_score(
        y_test,
        test_scores,
    )
)


roc_auc = float(
    roc_auc_score(
        y_test,
        test_scores,
    )
)


brier_score = float(
    brier_score_loss(
        y_test,
        test_scores,
    )
)


final_log_loss = float(
    log_loss(
        y_test,
        test_scores,
        labels=[
            0,
            1,
        ],
    )
)


pr_auc_lift_over_prevalence = (
    pr_auc
    /
    test_prevalence
    if test_prevalence
    >
    0
    else
    np.nan
)


# =============================================================================
# 14. FINAL LOCKED-THRESHOLD METRICS
# =============================================================================

precision = float(
    precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )
)


recall = float(
    recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    )
)


f1 = float(
    f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    )
)


balanced_accuracy = float(
    balanced_accuracy_score(
        y_test,
        test_predictions,
    )
)


accuracy = float(
    accuracy_score(
        y_test,
        test_predictions,
    )
)


mcc = float(
    matthews_corrcoef(
        y_test,
        test_predictions,
    )
)


tn, fp, fn, tp = (
    confusion_matrix(
        y_test,
        test_predictions,
        labels=[
            0,
            1,
        ],
    )
    .ravel()
)


specificity = float(
    tn
    /
    (
        tn
        +
        fp
    )
    if (
        tn
        +
        fp
    )
    >
    0
    else
    np.nan
)


predicted_positive_count = int(
    test_predictions.sum()
)


predicted_positive_rate = float(
    np.mean(
        test_predictions
    )
    *
    100
)


# =============================================================================
# 15. SCORE DISTRIBUTION SUMMARY
# =============================================================================

positive_scores = (
    test_scores[
        y_test
        ==
        1
    ]
)


negative_scores = (
    test_scores[
        y_test
        ==
        0
    ]
)


score_summary_df = pd.DataFrame(
    [
        {
            "group": (
                "ACTUAL_POSITIVE"
            ),
            "count": (
                len(
                    positive_scores
                )
            ),
            "mean_score": float(
                np.mean(
                    positive_scores
                )
            ),
            "median_score": float(
                np.median(
                    positive_scores
                )
            ),
            "p25_score": float(
                np.quantile(
                    positive_scores,
                    0.25,
                )
            ),
            "p75_score": float(
                np.quantile(
                    positive_scores,
                    0.75,
                )
            ),
            "minimum_score": float(
                np.min(
                    positive_scores
                )
            ),
            "maximum_score": float(
                np.max(
                    positive_scores
                )
            ),
        },
        {
            "group": (
                "ACTUAL_NEGATIVE"
            ),
            "count": (
                len(
                    negative_scores
                )
            ),
            "mean_score": float(
                np.mean(
                    negative_scores
                )
            ),
            "median_score": float(
                np.median(
                    negative_scores
                )
            ),
            "p25_score": float(
                np.quantile(
                    negative_scores,
                    0.25,
                )
            ),
            "p75_score": float(
                np.quantile(
                    negative_scores,
                    0.75,
                )
            ),
            "minimum_score": float(
                np.min(
                    negative_scores
                )
            ),
            "maximum_score": float(
                np.max(
                    negative_scores
                )
            ),
        },
    ]
)


# =============================================================================
# 16. TEST PREDICTION TABLE
# =============================================================================

test_prediction_df = pd.DataFrame(
    {
        "row_id": (
            test_df[
                "row_id"
            ].to_numpy()
        ),
        "grid_id": (
            test_df[
                "grid_id"
            ].to_numpy()
        ),
        "origin_year": (
            test_df[
                "origin_year"
            ].astype(
                int
            ).to_numpy()
        ),
        "spatial_block_1km": (
            test_df[
                "spatial_block_1km"
            ].to_numpy()
        ),
        "spatial_fold": (
            test_df[
                "spatial_fold"
            ].astype(
                int
            ).to_numpy()
        ),
        "actual_target": (
            y_test
        ),
        "raw_model_score": (
            test_scores
        ),
        "locked_threshold": (
            locked_threshold
        ),
        "predicted_class": (
            test_predictions
        ),
    }
)


# =============================================================================
# 17. FINAL METRICS TABLE
# =============================================================================

final_metrics_df = pd.DataFrame(
    [
        {
            "evaluation": (
                "FINAL_UNTOUCHED_TEMPORAL_TEST"
            ),
            "test_origin_year": (
                FINAL_TEST_YEAR
            ),
            "target_future_period": (
                "2023-2025"
            ),
            "selected_model": (
                selected_model
            ),
            "selected_candidate": (
                selected_candidate
            ),
            "locked_threshold": (
                locked_threshold
            ),
            "train_rows": (
                len(
                    train_df
                )
            ),
            "train_positive_rows": int(
                y_train.sum()
            ),
            "test_rows": (
                len(
                    test_df
                )
            ),
            "test_positive_rows": int(
                y_test.sum()
            ),
            "test_positive_prevalence": (
                test_prevalence
            ),
            "pr_auc_average_precision": (
                pr_auc
            ),
            "pr_auc_lift_over_prevalence": (
                pr_auc_lift_over_prevalence
            ),
            "roc_auc": (
                roc_auc
            ),
            "brier_score": (
                brier_score
            ),
            "log_loss": (
                final_log_loss
            ),
            "precision_at_locked_threshold": (
                precision
            ),
            "recall_at_locked_threshold": (
                recall
            ),
            "f1_at_locked_threshold": (
                f1
            ),
            "balanced_accuracy_at_locked_threshold": (
                balanced_accuracy
            ),
            "specificity_at_locked_threshold": (
                specificity
            ),
            "accuracy_secondary_only": (
                accuracy
            ),
            "matthews_correlation_coefficient": (
                mcc
            ),
            "true_negative": int(
                tn
            ),
            "false_positive": int(
                fp
            ),
            "false_negative": int(
                fn
            ),
            "true_positive": int(
                tp
            ),
            "predicted_positive_count": (
                predicted_positive_count
            ),
            "predicted_positive_rate_percent": (
                predicted_positive_rate
            ),
            "raw_score_is_calibrated_probability": (
                False
            ),
        }
    ]
)


# =============================================================================
# 18. QA
# =============================================================================

probability_bounds_valid = bool(
    np.all(
        (
            test_scores
            >=
            0.0
        )
        &
        (
            test_scores
            <=
            1.0
        )
    )
)


metric_values_finite = bool(
    np.isfinite(
        np.array(
            [
                pr_auc,
                roc_auc,
                brier_score,
                final_log_loss,
                precision,
                recall,
                f1,
                balanced_accuracy,
                specificity,
                accuracy,
                mcc,
            ],
            dtype=float,
        )
    ).all()
)


confusion_total_valid = (
    int(
        tn
        +
        fp
        +
        fn
        +
        tp
    )
    ==
    EXPECTED_TEST_ROWS
)


actual_positive_count_valid = (
    int(
        fn
        +
        tp
    )
    ==
    EXPECTED_TEST_POSITIVES
)


locked_threshold_unchanged = bool(
    np.isclose(
        locked_threshold,
        float(
            locked_decision[
                "operating_threshold"
            ]
        ),
        atol=0.0,
        rtol=0.0,
    )
)


selected_model_unchanged = (
    selected_model
    ==
    locked_decision[
        "selected_model"
    ]
)


alternative_model_tested = False

hyperparameters_retuned = False

threshold_retuned = False

final_test_run_completed = True


overall_status = (
    "PASS"
    if all(
        [
            len(
                train_df
            )
            ==
            EXPECTED_TRAIN_ROWS,
            int(
                y_train.sum()
            )
            ==
            EXPECTED_TRAIN_POSITIVES,
            len(
                embargo_df
            )
            ==
            EXPECTED_EMBARGO_ROWS,
            len(
                test_df
            )
            ==
            EXPECTED_TEST_ROWS,
            int(
                y_test.sum()
            )
            ==
            EXPECTED_TEST_POSITIVES,
            train_test_row_overlap
            ==
            0,
            probability_bounds_valid,
            metric_values_finite,
            confusion_total_valid,
            actual_positive_count_valid,
            locked_threshold_unchanged,
            selected_model_unchanged,
            not alternative_model_tested,
            not hyperparameters_retuned,
            not threshold_retuned,
            final_test_run_completed,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 19. SAVE FINAL-EVALUATION MODEL
# =============================================================================

model_bundle = {
    "model": (
        model
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
        final_scale_pos_weight
    ),
    "feature_names": (
        feature_columns
    ),
    "locked_threshold": (
        locked_threshold
    ),
    "training_origin_years": (
        "2012-2019"
    ),
    "final_test_origin_year": (
        FINAL_TEST_YEAR
    ),
    "final_test_future_period": (
        "2023-2025"
    ),
    "model_role": (
        "FINAL_EVALUATION_MODEL_NOT_PRODUCTION_MODEL"
    ),
}


joblib.dump(
    model_bundle,
    EVALUATION_MODEL_FILE,
)


# =============================================================================
# 20. QA / SUMMARY REPORTS
# =============================================================================

qa_df = pd.DataFrame(
    [
        {
            "train_rows": (
                len(
                    train_df
                )
            ),
            "train_positive_rows": int(
                y_train.sum()
            ),
            "embargo_rows": (
                len(
                    embargo_df
                )
            ),
            "test_rows": (
                len(
                    test_df
                )
            ),
            "test_positive_rows": int(
                y_test.sum()
            ),
            "train_test_row_overlap": (
                train_test_row_overlap
            ),
            "probability_bounds_valid": (
                probability_bounds_valid
            ),
            "metric_values_finite": (
                metric_values_finite
            ),
            "confusion_matrix_total_valid": (
                confusion_total_valid
            ),
            "actual_positive_count_valid": (
                actual_positive_count_valid
            ),
            "selected_model_unchanged": (
                selected_model_unchanged
            ),
            "locked_threshold_unchanged": (
                locked_threshold_unchanged
            ),
            "alternative_model_tested_on_final_test": (
                alternative_model_tested
            ),
            "hyperparameters_retuned_after_final_test": (
                hyperparameters_retuned
            ),
            "threshold_retuned_after_final_test": (
                threshold_retuned
            ),
            "final_test_run_completed": (
                final_test_run_completed
            ),
            "overall_step13e_status": (
                overall_status
            ),
        }
    ]
)


final_summary_df = pd.DataFrame(
    [
        {
            "stage": (
                "One-time untouched final temporal evaluation"
            ),
            "selected_model": (
                selected_model
            ),
            "selected_candidate": (
                selected_candidate
            ),
            "locked_threshold": (
                locked_threshold
            ),
            "training_origins": (
                "2012-2019"
            ),
            "embargo_origins": (
                "2020-2021"
            ),
            "final_test_origin": (
                2022
            ),
            "final_test_target_period": (
                "2023-2025"
            ),
            "final_test_rows": (
                len(
                    test_df
                )
            ),
            "final_test_positive_rows": int(
                y_test.sum()
            ),
            "pr_auc_average_precision": (
                pr_auc
            ),
            "roc_auc": (
                roc_auc
            ),
            "precision": (
                precision
            ),
            "recall": (
                recall
            ),
            "f1": (
                f1
            ),
            "balanced_accuracy": (
                balanced_accuracy
            ),
            "raw_score_calibrated_probability": (
                False
            ),
            "model_or_threshold_changed_after_test": (
                False
            ),
            "evaluation_model_is_production_model": (
                False
            ),
            "overall_step13e_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 21. SAVE OUTPUTS
# =============================================================================

test_prediction_df.to_csv(
    TEST_PREDICTIONS_CSV,
    index=False,
)


final_metrics_df.to_csv(
    FINAL_METRICS_CSV,
    index=False,
)


score_summary_df.to_csv(
    SCORE_SUMMARY_CSV,
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
# 22. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("FINAL 2022 THRESHOLD-FREE PERFORMANCE")
print("=" * 100)


print(
    f"Test prevalence                 : "
    f"{test_prevalence:.6f} "
    f"({test_prevalence * 100:.3f}%)"
)

print(
    f"PR-AUC / Average Precision      : "
    f"{pr_auc:.6f}"
)

print(
    f"PR-AUC lift over prevalence     : "
    f"{pr_auc_lift_over_prevalence:.3f}x"
)

print(
    f"ROC-AUC                         : "
    f"{roc_auc:.6f}"
)

print(
    f"Brier score                     : "
    f"{brier_score:.6f}"
)

print(
    f"Log loss                        : "
    f"{final_log_loss:.6f}"
)


print("\n" + "=" * 100)
print("FINAL 2022 LOCKED-THRESHOLD PERFORMANCE")
print("=" * 100)


print(
    f"Locked threshold                : "
    f"{locked_threshold:.3f}"
)

print(
    f"Precision                       : "
    f"{precision:.6f}"
)

print(
    f"Recall                          : "
    f"{recall:.6f}"
)

print(
    f"F1                              : "
    f"{f1:.6f}"
)

print(
    f"Balanced accuracy               : "
    f"{balanced_accuracy:.6f}"
)

print(
    f"Specificity                     : "
    f"{specificity:.6f}"
)

print(
    f"MCC                             : "
    f"{mcc:.6f}"
)

print(
    f"Accuracy (secondary only)       : "
    f"{accuracy:.6f}"
)

print(
    f"Predicted positive cells        : "
    f"{predicted_positive_count:,}"
)

print(
    f"Predicted positive rate         : "
    f"{predicted_positive_rate:.3f}%"
)


print("\nConfusion matrix:")

print(
    f"TN = {tn:,}"
)

print(
    f"FP = {fp:,}"
)

print(
    f"FN = {fn:,}"
)

print(
    f"TP = {tp:,}"
)


print("\n" + "=" * 100)
print("FINAL SCORE DISTRIBUTION")
print("=" * 100)


print(
    score_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 13E QA")
print("=" * 100)


print(
    f"Train rows valid                     : "
    f"{len(train_df) == EXPECTED_TRAIN_ROWS}"
)

print(
    f"Train positive count valid           : "
    f"{int(y_train.sum()) == EXPECTED_TRAIN_POSITIVES}"
)

print(
    f"Embargo rows valid                   : "
    f"{len(embargo_df) == EXPECTED_EMBARGO_ROWS}"
)

print(
    f"Final test rows valid                : "
    f"{len(test_df) == EXPECTED_TEST_ROWS}"
)

print(
    f"Final test positive count valid      : "
    f"{int(y_test.sum()) == EXPECTED_TEST_POSITIVES}"
)

print(
    f"Train/test row overlap               : "
    f"{train_test_row_overlap}"
)

print(
    f"Probability bounds valid             : "
    f"{probability_bounds_valid}"
)

print(
    f"Metrics finite                       : "
    f"{metric_values_finite}"
)

print(
    f"Locked model unchanged               : "
    f"{selected_model_unchanged}"
)

print(
    f"Locked threshold unchanged           : "
    f"{locked_threshold_unchanged}"
)

print(
    f"Alternative model tested             : "
    f"{alternative_model_tested}"
)

print(
    f"Hyperparameters retuned              : "
    f"{hyperparameters_retuned}"
)

print(
    f"Threshold retuned                    : "
    f"{threshold_retuned}"
)

print(
    f"Overall Step 13E status              : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(TEST_PREDICTIONS_CSV)}"
)

print(
    f" - {relative_path(FINAL_METRICS_CSV)}"
)

print(
    f" - {relative_path(SCORE_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(QA_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(EVALUATION_MODEL_FILE)}"
)


print("\nIMPORTANT:")

print(
    "1. This is the one-time final 2022 temporal test."
)

print(
    "2. The selected model and threshold were locked before this evaluation."
)

print(
    "3. Do NOT retune hyperparameters or threshold using this result."
)

print(
    "4. Do NOT switch to Random Forest because of the final-test result."
)

print(
    "5. The 2022 origin evaluates conversion outcomes observed during 2023-2025."
)

print(
    "6. Accuracy is secondary because the target prevalence is below 1% in the final test."
)

print(
    "7. PR-AUC should be interpreted relative to the final-test prevalence."
)

print(
    "8. Raw model scores remain pressure scores rather than calibrated real-world probabilities."
)

print(
    "9. This saved model is the final-evaluation model, not the later production model."
)

print("=" * 100)