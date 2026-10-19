from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

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
    / "random_forest"
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
    / "random_forest_best_validation_model.joblib"
)

CANDIDATE_METRICS_OUTPUT = (
    METRICS_DIR
    / "07_random_forest_candidate_metrics.csv"
)

VALIDATION_PREDICTIONS_OUTPUT = (
    PREDICTION_DIR
    / "07_random_forest_validation_predictions.csv"
)

FEATURE_IMPORTANCE_OUTPUT = (
    REPORT_DIR
    / "07_random_forest_feature_importance.csv"
)

MODEL_AUDIT_OUTPUT = (
    REPORT_DIR
    / "07_random_forest_model_audit.json"
)

DIRECT_REFERENCE_METRICS_OUTPUT = (
    METRICS_DIR
    / "07_random_forest_direct_reference_metrics.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

DEFAULT_THRESHOLD = 0.50


# ============================================================
# CANDIDATE MODELS
#
# Kept deliberately small because the training dataset is
# relatively limited and spatially grouped.
# ============================================================

RF_CANDIDATES = [
    {
        "candidate":
            "RF_01",

        "n_estimators":
            500,

        "max_depth":
            None,

        "min_samples_leaf":
            1,

        "max_features":
            "sqrt",

        "class_weight":
            None,
    },

    {
        "candidate":
            "RF_02",

        "n_estimators":
            500,

        "max_depth":
            12,

        "min_samples_leaf":
            2,

        "max_features":
            "sqrt",

        "class_weight":
            None,
    },

    {
        "candidate":
            "RF_03",

        "n_estimators":
            700,

        "max_depth":
            8,

        "min_samples_leaf":
            3,

        "max_features":
            "sqrt",

        "class_weight":
            None,
    },

    {
        "candidate":
            "RF_04",

        "n_estimators":
            700,

        "max_depth":
            10,

        "min_samples_leaf":
            4,

        "max_features":
            0.5,

        "class_weight":
            None,
    },

    {
        "candidate":
            "RF_05",

        "n_estimators":
            700,

        "max_depth":
            10,

        "min_samples_leaf":
            3,

        "max_features":
            "sqrt",

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
        "PART 2 RANDOM FOREST MODEL"
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
        X_train.isna().any().any()
        or
        X_validation.isna().any().any()
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

    # ========================================================
    # 5. TRAIN CANDIDATE RANDOM FORESTS
    # ========================================================

    metric_rows = []

    candidate_models = {}

    candidate_probabilities = {}

    print(
        "\nTraining candidate models..."
    )

    for parameters in RF_CANDIDATES:

        candidate_name = (
            parameters[
                "candidate"
            ]
        )

        print(
            f"  {candidate_name}..."
        )

        model = RandomForestClassifier(
            n_estimators=parameters[
                "n_estimators"
            ],

            max_depth=parameters[
                "max_depth"
            ],

            min_samples_leaf=parameters[
                "min_samples_leaf"
            ],

            max_features=parameters[
                "max_features"
            ],

            class_weight=parameters[
                "class_weight"
            ],

            random_state=RANDOM_SEED,

            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            model
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
            "n_estimators"
        ] = parameters[
            "n_estimators"
        ]

        metrics[
            "max_depth"
        ] = parameters[
            "max_depth"
        ]

        metrics[
            "min_samples_leaf"
        ] = parameters[
            "min_samples_leaf"
        ]

        metrics[
            "max_features"
        ] = str(
            parameters[
                "max_features"
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
        ] = model

        candidate_probabilities[
            candidate_name
        ] = probabilities

    # ========================================================
    # 6. SELECT BEST VALIDATION CANDIDATE
    #
    # Primary criterion:
    # balanced accuracy
    #
    # Tie breakers:
    # PR-AUC, then ROC-AUC
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
    # 7. SAVE BEST MODEL
    # ========================================================

    joblib.dump(
        best_model,
        BEST_MODEL_OUTPUT,
    )

    # ========================================================
    # 8. FEATURE IMPORTANCE
    # ========================================================

    feature_importance = pd.DataFrame(
        {
            "feature":
                feature_names,

            "importance":
                best_model
                .feature_importances_,
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    feature_importance[
        "rank"
    ] = (
        np.arange(
            1,
            len(
                feature_importance
            )
            + 1,
        )
    )

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_OUTPUT,
        index=False,
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
        "rf_probability"
    ] = (
        best_probabilities
    )

    prediction_table[
        "rf_prediction_0_50"
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
    #
    # This is reported separately because supplementary
    # polygon-positive pixels are spatially correlated within
    # known wetland sites.
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

        "candidate_count":
            len(
                RF_CANDIDATES
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
        "RANDOM FOREST VALIDATION RESULTS"
    )

    print(
        "========================================"
    )

    result_columns = [
        "candidate",
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
        "\nTop 15 feature importances:"
    )

    print(
        feature_importance[
            [
                "rank",
                "feature",
                "importance",
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
        BEST_MODEL_OUTPUT
    )

    print(
        CANDIDATE_METRICS_OUTPUT
    )

    print(
        VALIDATION_PREDICTIONS_OUTPUT
    )

    print(
        FEATURE_IMPORTANCE_OUTPUT
    )

    print(
        DIRECT_REFERENCE_METRICS_OUTPUT
    )

    print(
        MODEL_AUDIT_OUTPUT
    )

    print(
        "\n========================================"
    )

    print(
        "RANDOM FOREST STATUS: COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nThe test set was not loaded."
        "\nThe 0.50 threshold is still provisional."
        "\nHyperparameter candidates were compared only "
        "using the validation set."
        "\nFinal probability-threshold selection will "
        "happen after model-family comparison."
    )


if __name__ == "__main__":
    main()