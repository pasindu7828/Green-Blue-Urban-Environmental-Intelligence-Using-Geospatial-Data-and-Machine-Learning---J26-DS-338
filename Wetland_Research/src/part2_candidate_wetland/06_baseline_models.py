from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "part2_candidate_wetland"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part2_candidate_wetland"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "candidate_wetland"
    / "baseline"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "part2_candidate_wetland"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "part2_candidate_wetland"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "part2_candidate_wetland"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUTS
# ============================================================

TRAIN_INPUT = (
    SPLIT_DIR
    / "train_samples.csv"
)

VALIDATION_INPUT = (
    SPLIT_DIR
    / "validation_samples.csv"
)

FEATURE_METADATA_INPUT = (
    PROCESSED_DIR
    / "feature_stack_metadata.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

SCALER_OUTPUT = (
    MODEL_DIR
    / "baseline_standard_scaler.joblib"
)

LOGISTIC_MODEL_OUTPUT = (
    MODEL_DIR
    / "logistic_regression_baseline.joblib"
)

DUMMY_MODEL_OUTPUT = (
    MODEL_DIR
    / "dummy_prior_baseline.joblib"
)

METRICS_OUTPUT = (
    METRICS_DIR
    / "06_baseline_validation_metrics.csv"
)

PREDICTIONS_OUTPUT = (
    PREDICTION_DIR
    / "06_baseline_validation_predictions.csv"
)

COEFFICIENT_OUTPUT = (
    REPORT_DIR
    / "06_logistic_regression_coefficients.csv"
)

AUDIT_OUTPUT = (
    REPORT_DIR
    / "06_baseline_model_audit.json"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

DEFAULT_THRESHOLD = 0.50


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
    model_name,
):

    predictions = (
        probabilities
        >= threshold
    ).astype(
        int
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )

    metrics = {
        "model":
            model_name,

        "threshold":
            threshold,

        "samples":
            len(
                y_true
            ),

        "accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predictions,
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities,
            ),

        "tn":
            int(
                tn
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "tp":
            int(
                tp
            ),
    }

    return (
        metrics,
        predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 BASELINE MODELS"
    )

    print(
        "========================================"
    )

    # ========================================================
    # 1. LOAD TRAIN / VALIDATION ONLY
    # ========================================================

    train = pd.read_csv(
        TRAIN_INPUT
    )

    validation = pd.read_csv(
        VALIDATION_INPUT
    )

    print(
        f"Train samples: "
        f"{len(train)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation)}"
    )

    print(
        "\nTEST SET STATUS: LOCKED / NOT LOADED"
    )

    # ========================================================
    # 2. LOAD FEATURE LIST
    # ========================================================

    feature_metadata = pd.read_csv(
        FEATURE_METADATA_INPUT
    )

    feature_names = (
        feature_metadata[
            "feature"
        ]
        .tolist()
    )

    print(
        f"\nFeature count: "
        f"{len(feature_names)}"
    )

    missing_train_columns = [
        feature
        for feature
        in feature_names
        if feature
        not in train.columns
    ]

    missing_validation_columns = [
        feature
        for feature
        in feature_names
        if feature
        not in validation.columns
    ]

    if (
        missing_train_columns
        or
        missing_validation_columns
    ):

        raise RuntimeError(
            "Feature columns missing from "
            "train or validation data."
        )

    # ========================================================
    # 3. PREPARE MATRICES
    # ========================================================

    X_train = (
        train[
            feature_names
        ]
        .astype(
            np.float64
        )
    )

    y_train = (
        train[
            "target"
        ]
        .astype(
            int
        )
    )

    X_validation = (
        validation[
            feature_names
        ]
        .astype(
            np.float64
        )
    )

    y_validation = (
        validation[
            "target"
        ]
        .astype(
            int
        )
    )

    # ========================================================
    # 4. DATA QA
    # ========================================================

    train_missing = int(
        X_train
        .isna()
        .sum()
        .sum()
    )

    validation_missing = int(
        X_validation
        .isna()
        .sum()
        .sum()
    )

    train_infinite = int(
        np.isinf(
            X_train.to_numpy()
        ).sum()
    )

    validation_infinite = int(
        np.isinf(
            X_validation.to_numpy()
        ).sum()
    )

    print(
        "\nData QA:"
    )

    print(
        f"Train missing values: "
        f"{train_missing}"
    )

    print(
        f"Validation missing values: "
        f"{validation_missing}"
    )

    print(
        f"Train infinite values: "
        f"{train_infinite}"
    )

    print(
        f"Validation infinite values: "
        f"{validation_infinite}"
    )

    if (
        train_missing > 0
        or
        validation_missing > 0
        or
        train_infinite > 0
        or
        validation_infinite > 0
    ):

        raise RuntimeError(
            "Missing or infinite feature values "
            "detected. Fix data before modelling."
        )

    print(
        "\nClass distribution:"
    )

    print(
        f"Train positive: "
        f"{int((y_train == 1).sum())}"
    )

    print(
        f"Train negative: "
        f"{int((y_train == 0).sum())}"
    )

    print(
        f"Validation positive: "
        f"{int((y_validation == 1).sum())}"
    )

    print(
        f"Validation negative: "
        f"{int((y_validation == 0).sum())}"
    )

    # ========================================================
    # 5. DUMMY PRIOR BASELINE
    #
    # Gives us a deliberately simple reference model.
    # ========================================================

    print(
        "\nTraining Dummy prior baseline..."
    )

    dummy_model = DummyClassifier(
        strategy="prior",
        random_state=RANDOM_SEED,
    )

    dummy_model.fit(
        X_train,
        y_train,
    )

    dummy_probability = (
        dummy_model
        .predict_proba(
            X_validation
        )[
            :,
            1
        ]
    )

    (
        dummy_metrics,
        dummy_predictions,
    ) = calculate_metrics(
        y_validation,
        dummy_probability,
        DEFAULT_THRESHOLD,
        "Dummy_Prior",
    )

    joblib.dump(
        dummy_model,
        DUMMY_MODEL_OUTPUT,
    )

    # ========================================================
    # 6. TRAIN-ONLY STANDARDIZATION
    #
    # The scaler is fitted ONLY on training data.
    # Validation data is transformed using training
    # parameters to prevent preprocessing leakage.
    # ========================================================

    print(
        "Fitting train-only scaler..."
    )

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_validation_scaled = (
        scaler.transform(
            X_validation
        )
    )

    joblib.dump(
        scaler,
        SCALER_OUTPUT,
    )

    # ========================================================
    # 7. LOGISTIC REGRESSION BASELINE
    # ========================================================

    print(
        "Training Logistic Regression baseline..."
    )

    logistic_model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_SEED,
    )

    logistic_model.fit(
        X_train_scaled,
        y_train,
    )

    logistic_probability = (
        logistic_model
        .predict_proba(
            X_validation_scaled
        )[
            :,
            1
        ]
    )

    (
        logistic_metrics,
        logistic_predictions,
    ) = calculate_metrics(
        y_validation,
        logistic_probability,
        DEFAULT_THRESHOLD,
        "Logistic_Regression",
    )

    joblib.dump(
        logistic_model,
        LOGISTIC_MODEL_OUTPUT,
    )

    # ========================================================
    # 8. LOGISTIC COEFFICIENTS
    # ========================================================

    coefficient_table = pd.DataFrame(
        {
            "feature":
                feature_names,

            "coefficient":
                logistic_model
                .coef_[
                    0
                ],
        }
    )

    coefficient_table[
        "absolute_coefficient"
    ] = (
        coefficient_table[
            "coefficient"
        ]
        .abs()
    )

    coefficient_table = (
        coefficient_table
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    coefficient_table.to_csv(
        COEFFICIENT_OUTPUT,
        index=False,
    )

    # ========================================================
    # 9. SAVE VALIDATION PREDICTIONS
    # ========================================================

    prediction_table = validation[
        [
            "sample_id",
            "sample_source",
            "source_site_id",
            "reference_label",
            "wetland_id",
            "target",
            "spatial_group_id",
            "spatial_block_id",
        ]
    ].copy()

    prediction_table[
        "dummy_probability"
    ] = (
        dummy_probability
    )

    prediction_table[
        "dummy_prediction_0_50"
    ] = (
        dummy_predictions
    )

    prediction_table[
        "logistic_probability"
    ] = (
        logistic_probability
    )

    prediction_table[
        "logistic_prediction_0_50"
    ] = (
        logistic_predictions
    )

    prediction_table.to_csv(
        PREDICTIONS_OUTPUT,
        index=False,
    )

    # ========================================================
    # 10. SAVE METRICS
    # ========================================================

    metrics_table = pd.DataFrame(
        [
            dummy_metrics,
            logistic_metrics,
        ]
    )

    metrics_table.to_csv(
        METRICS_OUTPUT,
        index=False,
    )

    # ========================================================
    # 11. AUDIT
    # ========================================================

    audit = {
        "random_seed":
            RANDOM_SEED,

        "test_set_loaded":
            False,

        "default_threshold":
            DEFAULT_THRESHOLD,

        "feature_count":
            len(
                feature_names
            ),

        "train_samples":
            len(
                train
            ),

        "validation_samples":
            len(
                validation
            ),

        "train_positive":
            int(
                (
                    y_train
                    == 1
                ).sum()
            ),

        "train_negative":
            int(
                (
                    y_train
                    == 0
                ).sum()
            ),

        "validation_positive":
            int(
                (
                    y_validation
                    == 1
                ).sum()
            ),

        "validation_negative":
            int(
                (
                    y_validation
                    == 0
                ).sum()
            ),

        "scaler_fit_scope":
            "train_only",

        "dummy_strategy":
            "prior",

        "logistic_solver":
            "liblinear",

        "logistic_C":
            1.0,

        "logistic_penalty":
            "l2",
    }

    with open(
        AUDIT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit,
            file,
            indent=4,
        )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "========================================"
    )

    display_columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    print(
        metrics_table[
            display_columns
        ]
        .to_string(
            index=False,
        )
    )

    print(
        "\nTop Logistic Regression coefficients:"
    )

    print(
        coefficient_table[
            [
                "feature",
                "coefficient",
                "absolute_coefficient",
            ]
        ]
        .head(
            15
        )
        .to_string(
            index=False,
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        SCALER_OUTPUT
    )

    print(
        LOGISTIC_MODEL_OUTPUT
    )

    print(
        DUMMY_MODEL_OUTPUT
    )

    print(
        METRICS_OUTPUT
    )

    print(
        PREDICTIONS_OUTPUT
    )

    print(
        COEFFICIENT_OUTPUT
    )

    print(
        AUDIT_OUTPUT
    )

    print(
        "\n========================================"
    )

    print(
        "BASELINE MODEL STATUS: COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nThe test set was not loaded."
        "\nAll scaling parameters were learned only "
        "from training data."
        "\nThe 0.50 threshold is only a baseline "
        "reference threshold."
        "\nFinal threshold selection will be performed "
        "later using validation data only."
    )


if __name__ == "__main__":
    main()