from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold

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
    / "svm"
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

BEST_MODEL_OUTPUT = (
    MODEL_DIR
    / "svm_best_validation_model.joblib"
)

CANDIDATE_METRICS_OUTPUT = (
    METRICS_DIR
    / "08_svm_candidate_metrics.csv"
)

VALIDATION_PREDICTIONS_OUTPUT = (
    PREDICTION_DIR
    / "08_svm_validation_predictions.csv"
)

DIRECT_REFERENCE_METRICS_OUTPUT = (
    METRICS_DIR
    / "08_svm_direct_reference_metrics.csv"
)

CALIBRATION_FOLD_OUTPUT = (
    REPORT_DIR
    / "08_svm_calibration_fold_audit.csv"
)

MODEL_AUDIT_OUTPUT = (
    REPORT_DIR
    / "08_svm_model_audit.json"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

DEFAULT_THRESHOLD = 0.50

CALIBRATION_FOLDS = 5


# ============================================================
# SVM CANDIDATES
#
# The feature set is standardized inside each training fold.
#
# We deliberately use a small candidate set because the
# training dataset is limited and spatially grouped.
# ============================================================

SVM_CANDIDATES = [
    {
        "candidate":
            "SVM_01",

        "kernel":
            "linear",

        "C":
            0.1,

        "gamma":
            "scale",

        "class_weight":
            None,
    },

    {
        "candidate":
            "SVM_02",

        "kernel":
            "linear",

        "C":
            1.0,

        "gamma":
            "scale",

        "class_weight":
            None,
    },

    {
        "candidate":
            "SVM_03",

        "kernel":
            "rbf",

        "C":
            1.0,

        "gamma":
            "scale",

        "class_weight":
            None,
    },

    {
        "candidate":
            "SVM_04",

        "kernel":
            "rbf",

        "C":
            10.0,

        "gamma":
            "scale",

        "class_weight":
            None,
    },

    {
        "candidate":
            "SVM_05",

        "kernel":
            "rbf",

        "C":
            10.0,

        "gamma":
            0.05,

        "class_weight":
            None,
    },

    {
        "candidate":
            "SVM_06",

        "kernel":
            "rbf",

        "C":
            10.0,

        "gamma":
            "scale",

        "class_weight":
            "balanced",
    },
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
    candidate_name,
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

    return {
        "candidate":
            candidate_name,

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


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 SVM MODEL"
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
    # 2. FEATURE LIST
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

    # ========================================================
    # 3. MATRICES
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

    groups_train = (
        train[
            "spatial_group_id"
        ]
        .astype(
            str
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
    # 4. QA
    # ========================================================

    if (
        X_train
        .isna()
        .any()
        .any()
        or
        X_validation
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "Missing values found."
        )

    if (
        np.isinf(
            X_train.to_numpy()
        ).any()
        or
        np.isinf(
            X_validation.to_numpy()
        ).any()
    ):

        raise RuntimeError(
            "Infinite values found."
        )

    print(
        "\nData QA: PASS"
    )

    print(
        f"Train positive / negative: "
        f"{int((y_train == 1).sum())} / "
        f"{int((y_train == 0).sum())}"
    )

    print(
        f"Validation positive / negative: "
        f"{int((y_validation == 1).sum())} / "
        f"{int((y_validation == 0).sum())}"
    )

    print(
        f"Training spatial groups: "
        f"{groups_train.nunique()}"
    )

    # ========================================================
    # 5. SPATIALLY GROUPED CALIBRATION FOLDS
    #
    # SVM decision values are converted to probabilities using
    # sigmoid calibration.
    #
    # Calibration folds remain spatially grouped so samples
    # from the same spatial group do not appear in both sides
    # of the same calibration fold.
    # ========================================================

    calibration_cv = StratifiedGroupKFold(
        n_splits=CALIBRATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    calibration_splits = list(
        calibration_cv.split(
            X_train,
            y_train,
            groups_train,
        )
    )

    calibration_audit_rows = []

    for fold_number, (
        fit_indices,
        calibration_indices,
    ) in enumerate(
        calibration_splits,
        start=1,
    ):

        y_fit = (
            y_train
            .iloc[
                fit_indices
            ]
        )

        y_calibration = (
            y_train
            .iloc[
                calibration_indices
            ]
        )

        fit_groups = set(
            groups_train
            .iloc[
                fit_indices
            ]
        )

        calibration_groups = set(
            groups_train
            .iloc[
                calibration_indices
            ]
        )

        overlap = (
            fit_groups
            &
            calibration_groups
        )

        fit_positive = int(
            (
                y_fit
                == 1
            ).sum()
        )

        fit_negative = int(
            (
                y_fit
                == 0
            ).sum()
        )

        calibration_positive = int(
            (
                y_calibration
                == 1
            ).sum()
        )

        calibration_negative = int(
            (
                y_calibration
                == 0
            ).sum()
        )

        if (
            len(
                overlap
            )
            != 0
        ):

            raise RuntimeError(
                "Spatial-group overlap detected "
                "inside SVM calibration folds."
            )

        if (
            fit_positive == 0
            or
            fit_negative == 0
            or
            calibration_positive == 0
            or
            calibration_negative == 0
        ):

            raise RuntimeError(
                "A calibration fold does not contain "
                "both target classes."
            )

        calibration_audit_rows.append(
            {
                "fold":
                    fold_number,

                "fit_samples":
                    len(
                        fit_indices
                    ),

                "calibration_samples":
                    len(
                        calibration_indices
                    ),

                "fit_positive":
                    fit_positive,

                "fit_negative":
                    fit_negative,

                "calibration_positive":
                    calibration_positive,

                "calibration_negative":
                    calibration_negative,

                "fit_spatial_groups":
                    len(
                        fit_groups
                    ),

                "calibration_spatial_groups":
                    len(
                        calibration_groups
                    ),

                "overlapping_spatial_groups":
                    len(
                        overlap
                    ),
            }
        )

    calibration_audit = pd.DataFrame(
        calibration_audit_rows
    )

    calibration_audit.to_csv(
        CALIBRATION_FOLD_OUTPUT,
        index=False,
    )

    print(
        "\nCalibration fold audit:"
    )

    print(
        calibration_audit
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 6. TRAIN CANDIDATE SVM MODELS
    # ========================================================

    metric_rows = []

    candidate_models = {}

    candidate_probabilities = {}

    print(
        "\nTraining candidate SVM models..."
    )

    for parameters in SVM_CANDIDATES:

        candidate_name = (
            parameters[
                "candidate"
            ]
        )

        print(
            f"  {candidate_name}..."
        )

        # ----------------------------------------------------
        # Pipeline:
        #
        # StandardScaler is learned only from each fit fold.
        # SVC then operates on standardized features.
        # ----------------------------------------------------

        base_pipeline = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "svc",
                    SVC(
                        kernel=parameters[
                            "kernel"
                        ],

                        C=parameters[
                            "C"
                        ],

                        gamma=parameters[
                            "gamma"
                        ],

                        class_weight=parameters[
                            "class_weight"
                        ],

                        probability=False,

                        cache_size=1000,

                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

        # ----------------------------------------------------
        # Spatially grouped sigmoid probability calibration.
        # ----------------------------------------------------

        calibrated_model = (
            CalibratedClassifierCV(
                estimator=base_pipeline,
                method="sigmoid",
                cv=calibration_splits,
            )
        )

        calibrated_model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            calibrated_model
            .predict_proba(
                X_validation
            )[
                :,
                1
            ]
        )

        metrics = calculate_metrics(
            y_validation,
            probabilities,
            DEFAULT_THRESHOLD,
            candidate_name,
        )

        metrics[
            "kernel"
        ] = parameters[
            "kernel"
        ]

        metrics[
            "C"
        ] = parameters[
            "C"
        ]

        metrics[
            "gamma"
        ] = str(
            parameters[
                "gamma"
            ]
        )

        metrics[
            "class_weight"
        ] = str(
            parameters[
                "class_weight"
            ]
        )

        metric_rows.append(
            metrics
        )

        candidate_models[
            candidate_name
        ] = calibrated_model

        candidate_probabilities[
            candidate_name
        ] = probabilities

    # ========================================================
    # 7. SELECT BEST VALIDATION CANDIDATE
    #
    # Same selection rule as Random Forest:
    #
    # 1. Balanced accuracy
    # 2. PR-AUC
    # 3. ROC-AUC
    # ========================================================

    metrics_table = pd.DataFrame(
        metric_rows
    )

    metrics_table = (
        metrics_table
        .sort_values(
            [
                "balanced_accuracy",
                "pr_auc",
                "roc_auc",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    best_candidate = (
        metrics_table
        .iloc[
            0
        ][
            "candidate"
        ]
    )

    best_model = (
        candidate_models[
            best_candidate
        ]
    )

    best_probabilities = (
        candidate_probabilities[
            best_candidate
        ]
    )

    print(
        f"\nBest validation candidate: "
        f"{best_candidate}"
    )

    # ========================================================
    # 8. SAVE BEST MODEL
    # ========================================================

    joblib.dump(
        best_model,
        BEST_MODEL_OUTPUT,
    )

    # ========================================================
    # 9. SAVE VALIDATION PREDICTIONS
    # ========================================================

    best_predictions = (
        best_probabilities
        >= DEFAULT_THRESHOLD
    ).astype(
        int
    )

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
        "svm_probability"
    ] = (
        best_probabilities
    )

    prediction_table[
        "svm_prediction_0_50"
    ] = (
        best_predictions
    )

    prediction_table[
        "selected_candidate"
    ] = (
        best_candidate
    )

    prediction_table.to_csv(
        VALIDATION_PREDICTIONS_OUTPUT,
        index=False,
    )

    # ========================================================
    # 10. DIRECT 2025 REFERENCE METRICS
    # ========================================================

    direct_mask = (
        validation[
            "sample_source"
        ]
        ==
        "lulc_reference_2025"
    )

    direct_validation = (
        validation[
            direct_mask
        ]
    )

    direct_y = (
        direct_validation[
            "target"
        ]
        .astype(
            int
        )
    )

    direct_probabilities = (
        best_probabilities[
            direct_mask.to_numpy()
        ]
    )

    direct_metrics = (
        calculate_metrics(
            direct_y,
            direct_probabilities,
            DEFAULT_THRESHOLD,
            best_candidate,
        )
    )

    direct_metrics[
        "evaluation_subset"
    ] = (
        "direct_2025_lulc_reference"
    )

    pd.DataFrame(
        [
            direct_metrics
        ]
    ).to_csv(
        DIRECT_REFERENCE_METRICS_OUTPUT,
        index=False,
    )

    # ========================================================
    # 11. SAVE CANDIDATE METRICS
    # ========================================================

    metrics_table.to_csv(
        CANDIDATE_METRICS_OUTPUT,
        index=False,
    )

    # ========================================================
    # 12. AUDIT
    # ========================================================

    best_row = (
        metrics_table
        .iloc[
            0
        ]
        .to_dict()
    )

    audit = {
        "random_seed":
            RANDOM_SEED,

        "test_set_loaded":
            False,

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

        "training_spatial_groups":
            int(
                groups_train
                .nunique()
            ),

        "calibration_method":
            "sigmoid",

        "calibration_folds":
            CALIBRATION_FOLDS,

        "calibration_split_method":
            "StratifiedGroupKFold",

        "candidate_count":
            len(
                SVM_CANDIDATES
            ),

        "selection_metric_primary":
            "balanced_accuracy",

        "selection_metric_tiebreak_1":
            "pr_auc",

        "selection_metric_tiebreak_2":
            "roc_auc",

        "selected_candidate":
            best_candidate,

        "default_threshold":
            DEFAULT_THRESHOLD,

        "selected_candidate_metrics":
            {
                key:
                    (
                        value.item()
                        if isinstance(
                            value,
                            np.generic,
                        )
                        else value
                    )
                for (
                    key,
                    value,
                ) in best_row.items()
            },
    }

    with open(
        MODEL_AUDIT_OUTPUT,
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
        "SVM VALIDATION RESULTS"
    )

    print(
        "========================================"
    )

    result_columns = [
        "candidate",
        "kernel",
        "C",
        "gamma",
        "class_weight",
        "balanced_accuracy",
        "accuracy",
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
            result_columns
        ]
        .to_string(
            index=False,
        )
    )

    print(
        "\nSelected model:"
    )

    print(
        best_candidate
    )

    print(
        "\nDirect 2025 LULC validation metrics:"
    )

    print(
        pd.DataFrame(
            [
                direct_metrics
            ]
        )[
            [
                "samples",
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
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        BEST_MODEL_OUTPUT
    )

    print(
        CANDIDATE_METRICS_OUTPUT
    )

    print(
        VALIDATION_PREDICTIONS_OUTPUT
    )

    print(
        DIRECT_REFERENCE_METRICS_OUTPUT
    )

    print(
        CALIBRATION_FOLD_OUTPUT
    )

    print(
        MODEL_AUDIT_OUTPUT
    )

    print(
        "\n========================================"
    )

    print(
        "SVM STATUS: COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nThe test set was not loaded."
        "\nFeature scaling is performed inside the "
        "training pipeline."
        "\nProbability calibration uses spatially "
        "grouped training folds."
        "\nThe 0.50 threshold remains provisional."
        "\nFinal threshold selection will happen only "
        "after model-family comparison."
    )


if __name__ == "__main__":
    main()