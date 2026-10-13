from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from xgboost import XGBClassifier

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
    / "final"
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

TEST_INPUT = (
    SPLIT_DIR
    / "test_samples.csv"
)

FEATURE_METADATA_INPUT = (
    PROCESSED_DIR
    / "feature_stack_metadata.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

FINAL_MODEL_OUTPUT = (
    MODEL_DIR
    / "xgboost_final_model.joblib"
)

FINAL_NATIVE_MODEL_OUTPUT = (
    MODEL_DIR
    / "xgboost_final_model.json"
)

TEST_METRICS_OUTPUT = (
    METRICS_DIR
    / "11_final_test_metrics.csv"
)

DIRECT_REFERENCE_METRICS_OUTPUT = (
    METRICS_DIR
    / "11_final_test_direct_reference_metrics.csv"
)

REFERENCE_DIAGNOSTICS_OUTPUT = (
    METRICS_DIR
    / "11_final_test_reference_label_diagnostics.csv"
)

KNOWN_WETLAND_SITE_OUTPUT = (
    METRICS_DIR
    / "11_final_test_known_wetland_site_diagnostics.csv"
)

TEST_PREDICTIONS_OUTPUT = (
    PREDICTION_DIR
    / "11_final_test_predictions.csv"
)

FEATURE_IMPORTANCE_OUTPUT = (
    REPORT_DIR
    / "11_final_xgboost_feature_importance.csv"
)

FINAL_AUDIT_OUTPUT = (
    REPORT_DIR
    / "11_final_test_evaluation_audit.json"
)

COMPLETION_MARKER = (
    REPORT_DIR
    / "11_FINAL_TEST_COMPLETED.json"
)


# ============================================================
# FROZEN MODEL CONFIGURATION
#
# These values come from validation-only model selection.
# They MUST NOT be changed after viewing the test results.
# ============================================================

RANDOM_SEED = 42

SELECTED_MODEL_FAMILY = "XGBoost"

SELECTED_CANDIDATE = "XGB_03"

SELECTED_THRESHOLD = 0.50

FINAL_PARAMETERS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 2,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "gamma": 0.10,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


# ============================================================
# METRICS
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
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        )
        .ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "model":
            model_name,

        "threshold":
            float(threshold),

        "samples":
            len(y_true),

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

        "specificity":
            specificity,

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
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 FINAL TEST EVALUATION"
    )

    print(
        "========================================"
    )

    # ========================================================
    # PROTECT AGAINST ACCIDENTAL RE-EVALUATION
    # ========================================================

    if COMPLETION_MARKER.exists():

        raise RuntimeError(
            "\nFINAL TEST EVALUATION HAS ALREADY "
            "BEEN COMPLETED.\n"
            "Do not repeatedly evaluate or tune models "
            "against the locked test set."
        )

    print(
        "\nFROZEN DEVELOPMENT DECISIONS:"
    )

    print(
        f"Model family: "
        f"{SELECTED_MODEL_FAMILY}"
    )

    print(
        f"Candidate: "
        f"{SELECTED_CANDIDATE}"
    )

    print(
        f"Threshold: "
        f"{SELECTED_THRESHOLD:.2f}"
    )

    # ========================================================
    # 1. LOAD TRAIN + VALIDATION
    #
    # Validation has completed its development role.
    # We now combine train and validation and refit the
    # frozen model configuration on all development data.
    # ========================================================

    train = pd.read_csv(
        TRAIN_INPUT
    )

    validation = pd.read_csv(
        VALIDATION_INPUT
    )

    development = pd.concat(
        [
            train,
            validation,
        ],
        axis=0,
        ignore_index=True,
    )

    print(
        f"\nTrain samples: "
        f"{len(train)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation)}"
    )

    print(
        f"Final development samples: "
        f"{len(development)}"
    )

    # ========================================================
    # 2. OPEN LOCKED TEST SET
    #
    # This is the first final evaluation use of the test set.
    # ========================================================

    test = pd.read_csv(
        TEST_INPUT
    )

    print(
        f"\nLOCKED TEST SET OPENED: "
        f"{len(test)} samples"
    )

    # ========================================================
    # 3. SPATIAL LEAKAGE AUDIT
    # ========================================================

    development_groups = set(
        development[
            "spatial_group_id"
        ].astype(str)
    )

    test_groups = set(
        test[
            "spatial_group_id"
        ].astype(str)
    )

    overlapping_groups = (
        development_groups
        &
        test_groups
    )

    development_blocks = set(
        development[
            "spatial_block_id"
        ].astype(str)
    )

    test_blocks = set(
        test[
            "spatial_block_id"
        ].astype(str)
    )

    overlapping_blocks = (
        development_blocks
        &
        test_blocks
    )

    development_wetlands = set(
        development.loc[
            development[
                "sample_source"
            ]
            ==
            "known_wetland_polygon",
            "wetland_id",
        ]
        .dropna()
        .astype(str)
    )

    test_wetlands = set(
        test.loc[
            test[
                "sample_source"
            ]
            ==
            "known_wetland_polygon",
            "wetland_id",
        ]
        .dropna()
        .astype(str)
    )

    overlapping_wetlands = (
        development_wetlands
        &
        test_wetlands
    )

    print(
        "\nSpatial leakage audit:"
    )

    print(
        f"Overlapping spatial groups: "
        f"{len(overlapping_groups)}"
    )

    print(
        f"Overlapping spatial blocks: "
        f"{len(overlapping_blocks)}"
    )

    print(
        f"Overlapping known wetlands: "
        f"{len(overlapping_wetlands)}"
    )

    if (
        len(overlapping_groups) != 0
        or
        len(overlapping_blocks) != 0
        or
        len(overlapping_wetlands) != 0
    ):

        raise RuntimeError(
            "Test-set spatial leakage detected."
        )

    # ========================================================
    # 4. FEATURE LIST
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
    # 5. MATRICES
    # ========================================================

    X_development = (
        development[
            feature_names
        ]
        .astype(np.float32)
    )

    y_development = (
        development[
            "target"
        ]
        .astype(int)
    )

    X_test = (
        test[
            feature_names
        ]
        .astype(np.float32)
    )

    y_test = (
        test[
            "target"
        ]
        .astype(int)
    )

    # ========================================================
    # 6. DATA QA
    # ========================================================

    development_missing = int(
        X_development
        .isna()
        .sum()
        .sum()
    )

    test_missing = int(
        X_test
        .isna()
        .sum()
        .sum()
    )

    development_infinite = int(
        np.isinf(
            X_development.to_numpy()
        ).sum()
    )

    test_infinite = int(
        np.isinf(
            X_test.to_numpy()
        ).sum()
    )

    if (
        development_missing > 0
        or
        test_missing > 0
        or
        development_infinite > 0
        or
        test_infinite > 0
    ):

        raise RuntimeError(
            "Missing or infinite feature "
            "values detected."
        )

    print(
        "\nData QA: PASS"
    )

    print(
        f"Development positive / negative: "
        f"{int((y_development == 1).sum())} / "
        f"{int((y_development == 0).sum())}"
    )

    print(
        f"Test positive / negative: "
        f"{int((y_test == 1).sum())} / "
        f"{int((y_test == 0).sum())}"
    )

    # ========================================================
    # 7. REFIT FROZEN FINAL MODEL
    #
    # Hyperparameters remain exactly those selected before
    # opening the test set.
    # ========================================================

    print(
        "\nTraining final frozen XGBoost model "
        "on train + validation..."
    )

    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",

        n_estimators=FINAL_PARAMETERS[
            "n_estimators"
        ],

        learning_rate=FINAL_PARAMETERS[
            "learning_rate"
        ],

        max_depth=FINAL_PARAMETERS[
            "max_depth"
        ],

        min_child_weight=FINAL_PARAMETERS[
            "min_child_weight"
        ],

        subsample=FINAL_PARAMETERS[
            "subsample"
        ],

        colsample_bytree=FINAL_PARAMETERS[
            "colsample_bytree"
        ],

        gamma=FINAL_PARAMETERS[
            "gamma"
        ],

        reg_alpha=FINAL_PARAMETERS[
            "reg_alpha"
        ],

        reg_lambda=FINAL_PARAMETERS[
            "reg_lambda"
        ],

        random_state=RANDOM_SEED,

        n_jobs=-1,

        verbosity=0,
    )

    final_model.fit(
        X_development,
        y_development,
    )

    # ========================================================
    # 8. ONE-TIME TEST PREDICTION
    # ========================================================

    test_probabilities = (
        final_model
        .predict_proba(
            X_test
        )[:, 1]
    )

    test_predictions = (
        test_probabilities
        >= SELECTED_THRESHOLD
    ).astype(int)

    # ========================================================
    # 9. FINAL TEST METRICS
    # ========================================================

    test_metrics = (
        calculate_metrics(
            y_test,
            test_probabilities,
            SELECTED_THRESHOLD,
            SELECTED_CANDIDATE,
        )
    )

    pd.DataFrame(
        [
            test_metrics
        ]
    ).to_csv(
        TEST_METRICS_OUTPUT,
        index=False,
    )

    # ========================================================
    # 10. DIRECT 2025 REFERENCE TEST METRICS
    # ========================================================

    direct_mask = (
        test[
            "sample_source"
        ]
        ==
        "lulc_reference_2025"
    ).to_numpy()

    direct_y = (
        y_test.to_numpy()[
            direct_mask
        ]
    )

    direct_probabilities = (
        test_probabilities[
            direct_mask
        ]
    )

    direct_metrics = (
        calculate_metrics(
            direct_y,
            direct_probabilities,
            SELECTED_THRESHOLD,
            SELECTED_CANDIDATE,
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
    # 11. SAVE TEST PREDICTIONS
    # ========================================================

    prediction_table = (
        test[
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
        ]
        .copy()
    )

    prediction_table[
        "selected_model"
    ] = (
        SELECTED_CANDIDATE
    )

    prediction_table[
        "probability"
    ] = (
        test_probabilities
    )

    prediction_table[
        "threshold"
    ] = (
        SELECTED_THRESHOLD
    )

    prediction_table[
        "prediction"
    ] = (
        test_predictions
    )

    prediction_table[
        "correct"
    ] = (
        prediction_table[
            "target"
        ]
        ==
        prediction_table[
            "prediction"
        ]
    )

    prediction_table.to_csv(
        TEST_PREDICTIONS_OUTPUT,
        index=False,
    )

    # ========================================================
    # 12. REFERENCE-LABEL DIAGNOSTICS
    # ========================================================

    diagnostic_rows = []

    for (
        reference_label,
        subset,
    ) in prediction_table.groupby(
        "reference_label",
        dropna=False,
    ):

        targets = (
            subset[
                "target"
            ]
            .astype(int)
        )

        predictions = (
            subset[
                "prediction"
            ]
            .astype(int)
        )

        tp = int(
            (
                (targets == 1)
                &
                (predictions == 1)
            ).sum()
        )

        tn = int(
            (
                (targets == 0)
                &
                (predictions == 0)
            ).sum()
        )

        fp = int(
            (
                (targets == 0)
                &
                (predictions == 1)
            ).sum()
        )

        fn = int(
            (
                (targets == 1)
                &
                (predictions == 0)
            ).sum()
        )

        diagnostic_rows.append(
            {
                "reference_label":
                    reference_label,

                "samples":
                    len(subset),

                "mean_probability":
                    subset[
                        "probability"
                    ].mean(),

                "predicted_positive":
                    int(
                        (
                            predictions == 1
                        ).sum()
                    ),

                "predicted_negative":
                    int(
                        (
                            predictions == 0
                        ).sum()
                    ),

                "correct":
                    int(
                        subset[
                            "correct"
                        ].sum()
                    ),

                "accuracy":
                    subset[
                        "correct"
                    ].mean(),

                "tp":
                    tp,

                "tn":
                    tn,

                "fp":
                    fp,

                "fn":
                    fn,
            }
        )

    reference_diagnostics = (
        pd.DataFrame(
            diagnostic_rows
        )
    )

    reference_diagnostics.to_csv(
        REFERENCE_DIAGNOSTICS_OUTPUT,
        index=False,
    )

    # ========================================================
    # 13. KNOWN-WETLAND SITE-LEVEL DIAGNOSTICS
    #
    # Supplementary known-wetland pixels from one site are
    # correlated. Site-level summaries are therefore reported
    # separately instead of treating every pixel as an
    # independent wetland site.
    # ========================================================

    known_polygon_predictions = (
        prediction_table[
            prediction_table[
                "sample_source"
            ]
            ==
            "known_wetland_polygon"
        ]
        .copy()
    )

    known_site_diagnostics = (
        known_polygon_predictions
        .groupby(
            "wetland_id"
        )
        .agg(
            samples=(
                "sample_id",
                "count",
            ),

            mean_probability=(
                "probability",
                "mean",
            ),

            min_probability=(
                "probability",
                "min",
            ),

            max_probability=(
                "probability",
                "max",
            ),

            predicted_positive=(
                "prediction",
                "sum",
            ),
        )
        .reset_index()
    )

    known_site_diagnostics[
        "predicted_negative"
    ] = (
        known_site_diagnostics[
            "samples"
        ]
        -
        known_site_diagnostics[
            "predicted_positive"
        ]
    )

    known_site_diagnostics[
        "positive_fraction"
    ] = (
        known_site_diagnostics[
            "predicted_positive"
        ]
        /
        known_site_diagnostics[
            "samples"
        ]
    )

    known_site_diagnostics.to_csv(
        KNOWN_WETLAND_SITE_OUTPUT,
        index=False,
    )

    # ========================================================
    # 14. FEATURE IMPORTANCE
    # ========================================================

    booster = (
        final_model
        .get_booster()
    )

    raw_gain = (
        booster.get_score(
            importance_type="gain"
        )
    )

    importance_rows = []

    for (
        index,
        feature_name,
    ) in enumerate(
        feature_names
    ):

        importance = raw_gain.get(
            feature_name,
            raw_gain.get(
                f"f{index}",
                0.0,
            ),
        )

        importance_rows.append(
            {
                "feature":
                    feature_name,

                "gain_importance":
                    float(
                        importance
                    ),
            }
        )

    feature_importance = pd.DataFrame(
        importance_rows
    )

    total_gain = (
        feature_importance[
            "gain_importance"
        ]
        .sum()
    )

    if (
        total_gain > 0
    ):

        feature_importance[
            "normalized_gain"
        ] = (
            feature_importance[
                "gain_importance"
            ]
            /
            total_gain
        )

    else:

        feature_importance[
            "normalized_gain"
        ] = 0.0

    feature_importance = (
        feature_importance
        .sort_values(
            "normalized_gain",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    feature_importance[
        "rank"
    ] = np.arange(
        1,
        len(feature_importance) + 1,
    )

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_OUTPUT,
        index=False,
    )

    # ========================================================
    # 15. SAVE FINAL MODEL
    # ========================================================

    joblib.dump(
        final_model,
        FINAL_MODEL_OUTPUT,
    )

    final_model.save_model(
        FINAL_NATIVE_MODEL_OUTPUT
    )

    # ========================================================
    # 16. FINAL AUDIT
    # ========================================================

    audit = {
        "evaluation_type":
            "one_time_locked_test_evaluation",

        "xgboost_version":
            xgb.__version__,

        "random_seed":
            RANDOM_SEED,

        "selected_model_family":
            SELECTED_MODEL_FAMILY,

        "selected_candidate":
            SELECTED_CANDIDATE,

        "selected_threshold":
            SELECTED_THRESHOLD,

        "model_parameters":
            FINAL_PARAMETERS,

        "final_training_scope":
            "train_plus_validation",

        "train_samples":
            len(train),

        "validation_samples":
            len(validation),

        "development_samples":
            len(development),

        "test_samples":
            len(test),

        "feature_count":
            len(feature_names),

        "spatial_group_overlap":
            len(overlapping_groups),

        "spatial_block_overlap":
            len(overlapping_blocks),

        "known_wetland_overlap":
            len(overlapping_wetlands),

        "test_metrics":
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
                ) in test_metrics.items()
            },

        "direct_reference_test_metrics":
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
                ) in direct_metrics.items()
            },

        "warning":
            (
                "Test results must not be used "
                "for further hyperparameter or "
                "threshold tuning."
            ),
    }

    with open(
        FINAL_AUDIT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit,
            file,
            indent=4,
        )

    # ========================================================
    # 17. COMPLETION MARKER
    #
    # Written only after successful completion.
    # Prevents accidental repeated evaluation.
    # ========================================================

    completion_record = {
        "status":
            "FINAL_TEST_COMPLETED",

        "model":
            SELECTED_CANDIDATE,

        "threshold":
            SELECTED_THRESHOLD,

        "test_samples":
            len(test),

        "important":
            (
                "Do not tune the model or threshold "
                "using these test results."
            ),
    }

    with open(
        COMPLETION_MARKER,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            completion_record,
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
        "FINAL LOCKED TEST RESULTS"
    )

    print(
        "========================================"
    )

    print(
        pd.DataFrame(
            [
                test_metrics
            ]
        )[
            [
                "samples",
                "threshold",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "specificity",
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
        "\nDirect 2025 LULC test metrics:"
    )

    print(
        pd.DataFrame(
            [
                direct_metrics
            ]
        )[
            [
                "samples",
                "threshold",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "specificity",
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
        "\nReference-label diagnostics:"
    )

    print(
        reference_diagnostics
        .to_string(
            index=False
        )
    )

    print(
        "\nKnown-wetland site diagnostics:"
    )

    print(
        known_site_diagnostics
        .to_string(
            index=False
        )
    )

    print(
        "\nTop 15 final feature importances:"
    )

    print(
        feature_importance[
            [
                "rank",
                "feature",
                "gain_importance",
                "normalized_gain",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        FINAL_MODEL_OUTPUT
    )

    print(
        FINAL_NATIVE_MODEL_OUTPUT
    )

    print(
        TEST_METRICS_OUTPUT
    )

    print(
        DIRECT_REFERENCE_METRICS_OUTPUT
    )

    print(
        REFERENCE_DIAGNOSTICS_OUTPUT
    )

    print(
        KNOWN_WETLAND_SITE_OUTPUT
    )

    print(
        TEST_PREDICTIONS_OUTPUT
    )

    print(
        FEATURE_IMPORTANCE_OUTPUT
    )

    print(
        FINAL_AUDIT_OUTPUT
    )

    print(
        COMPLETION_MARKER
    )

    print(
        "\n========================================"
    )

    print(
        "FINAL TEST EVALUATION: COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nThe locked test set has now been used."
        "\nDo NOT change model hyperparameters or "
        "threshold based on these results."
        "\nThe final model was refitted using the "
        "combined train + validation development data."
    )


if __name__ == "__main__":
    main()