from pathlib import Path
import json

import numpy as np
import pandas as pd

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

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "part2_candidate_wetland"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "part2_candidate_wetland"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "part2_candidate_wetland"
)

METRICS_DIR.mkdir(
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

VALIDATION_INPUT = (
    SPLIT_DIR
    / "validation_samples.csv"
)

BASELINE_PREDICTIONS = (
    PREDICTION_DIR
    / "06_baseline_validation_predictions.csv"
)

RF_PREDICTIONS = (
    PREDICTION_DIR
    / "07_random_forest_validation_predictions.csv"
)

SVM_PREDICTIONS = (
    PREDICTION_DIR
    / "08_svm_validation_predictions.csv"
)

XGB_PREDICTIONS = (
    PREDICTION_DIR
    / "09_xgboost_validation_predictions.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

MODEL_COMPARISON_OUTPUT = (
    METRICS_DIR
    / "10_model_family_comparison.csv"
)

THRESHOLD_SEARCH_OUTPUT = (
    METRICS_DIR
    / "10_selected_model_threshold_search.csv"
)

DIRECT_REFERENCE_METRICS_OUTPUT = (
    METRICS_DIR
    / "10_selected_model_direct_reference_metrics.csv"
)

FINAL_VALIDATION_PREDICTIONS_OUTPUT = (
    PREDICTION_DIR
    / "10_selected_model_validation_predictions.csv"
)

REFERENCE_DIAGNOSTICS_OUTPUT = (
    METRICS_DIR
    / "10_selected_model_reference_label_diagnostics.csv"
)

MODEL_SELECTION_AUDIT_OUTPUT = (
    REPORT_DIR
    / "10_model_selection_and_threshold_audit.json"
)


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_THRESHOLD = 0.50

THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01


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

    recall = (
        recall_score(
            y_true,
            predictions,
            zero_division=0,
        )
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
            recall,

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
# LOAD AND ALIGN PREDICTIONS
# ============================================================

def load_probability_column(
    path,
    probability_column,
    expected_sample_ids,
):

    prediction_data = pd.read_csv(
        path
    )

    if (
        "sample_id"
        not in prediction_data.columns
    ):

        raise RuntimeError(
            f"sample_id missing from {path}"
        )

    if (
        probability_column
        not in prediction_data.columns
    ):

        raise RuntimeError(
            f"{probability_column} missing from {path}"
        )

    if (
        prediction_data[
            "sample_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"Duplicate sample IDs found in {path}"
        )

    prediction_data = (
        prediction_data
        .set_index(
            "sample_id"
        )
    )

    missing_ids = (
        set(
            expected_sample_ids
        )
        -
        set(
            prediction_data.index
        )
    )

    extra_ids = (
        set(
            prediction_data.index
        )
        -
        set(
            expected_sample_ids
        )
    )

    if (
        missing_ids
        or
        extra_ids
    ):

        raise RuntimeError(
            f"Sample-ID mismatch in {path}"
        )

    probabilities = (
        prediction_data
        .loc[
            expected_sample_ids,
            probability_column,
        ]
        .to_numpy(
            dtype=float
        )
    )

    if (
        np.isnan(
            probabilities
        ).any()
        or
        np.isinf(
            probabilities
        ).any()
    ):

        raise RuntimeError(
            f"Invalid probabilities in {path}"
        )

    if (
        (
            probabilities < 0
        ).any()
        or
        (
            probabilities > 1
        ).any()
    ):

        raise RuntimeError(
            f"Probabilities outside [0, 1] in {path}"
        )

    return probabilities


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 MODEL SELECTION + THRESHOLD"
    )

    print(
        "========================================"
    )

    # ========================================================
    # 1. LOAD VALIDATION SET ONLY
    # ========================================================

    validation = pd.read_csv(
        VALIDATION_INPUT
    )

    print(
        f"Validation samples: "
        f"{len(validation)}"
    )

    print(
        "\nTEST SET STATUS: LOCKED / NOT LOADED"
    )

    if (
        validation[
            "sample_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate sample IDs found "
            "in validation data."
        )

    expected_sample_ids = (
        validation[
            "sample_id"
        ]
        .tolist()
    )

    y_validation = (
        validation[
            "target"
        ]
        .astype(int)
        .to_numpy()
    )

    # ========================================================
    # 2. LOAD VALIDATION PROBABILITIES
    # ========================================================

    logistic_probability = (
        load_probability_column(
            BASELINE_PREDICTIONS,
            "logistic_probability",
            expected_sample_ids,
        )
    )

    rf_probability = (
        load_probability_column(
            RF_PREDICTIONS,
            "rf_probability",
            expected_sample_ids,
        )
    )

    svm_probability = (
        load_probability_column(
            SVM_PREDICTIONS,
            "svm_probability",
            expected_sample_ids,
        )
    )

    xgb_probability = (
        load_probability_column(
            XGB_PREDICTIONS,
            "xgboost_probability",
            expected_sample_ids,
        )
    )

    probability_map = {
        "Logistic_Regression":
            logistic_probability,

        "Random_Forest":
            rf_probability,

        "SVM":
            svm_probability,

        "XGBoost":
            xgb_probability,
    }

    print(
        "\nPrediction alignment QA: PASS"
    )

    # ========================================================
    # 3. FORMAL MODEL-FAMILY COMPARISON
    #
    # All models are compared at the same provisional
    # threshold of 0.50.
    #
    # Selection hierarchy:
    # 1. Balanced accuracy
    # 2. PR-AUC
    # 3. ROC-AUC
    # ========================================================

    comparison_rows = []

    for (
        model_name,
        probabilities,
    ) in probability_map.items():

        metrics = calculate_metrics(
            y_validation,
            probabilities,
            DEFAULT_THRESHOLD,
            model_name,
        )

        comparison_rows.append(
            metrics
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison = (
        comparison
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

    comparison[
        "validation_rank"
    ] = np.arange(
        1,
        len(comparison) + 1,
    )

    selected_model = (
        comparison
        .iloc[0][
            "model"
        ]
    )

    selected_probabilities = (
        probability_map[
            selected_model
        ]
    )

    comparison.to_csv(
        MODEL_COMPARISON_OUTPUT,
        index=False,
    )

    # ========================================================
    # 4. THRESHOLD SEARCH
    #
    # Threshold tuning is performed only AFTER the model
    # family has been selected.
    #
    # Primary criterion:
    # balanced accuracy
    #
    # Tie breakers:
    # F1 score
    # threshold closest to 0.50
    # ========================================================

    thresholds = np.round(
        np.arange(
            THRESHOLD_MIN,
            THRESHOLD_MAX
            +
            THRESHOLD_STEP / 2,
            THRESHOLD_STEP,
        ),
        2,
    )

    threshold_rows = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_validation,
            selected_probabilities,
            threshold,
            selected_model,
        )

        metrics[
            "distance_from_0_50"
        ] = abs(
            threshold
            -
            DEFAULT_THRESHOLD
        )

        threshold_rows.append(
            metrics
        )

    threshold_table = pd.DataFrame(
        threshold_rows
    )

    threshold_table = (
        threshold_table
        .sort_values(
            [
                "balanced_accuracy",
                "f1",
                "distance_from_0_50",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    threshold_table[
        "threshold_rank"
    ] = np.arange(
        1,
        len(
            threshold_table
        )
        + 1,
    )

    selected_threshold = float(
        threshold_table
        .iloc[0][
            "threshold"
        ]
    )

    threshold_table.to_csv(
        THRESHOLD_SEARCH_OUTPUT,
        index=False,
    )

    # ========================================================
    # 5. FINAL VALIDATION METRICS
    # ========================================================

    final_validation_metrics = (
        calculate_metrics(
            y_validation,
            selected_probabilities,
            selected_threshold,
            selected_model,
        )
    )

    final_predictions = (
        selected_probabilities
        >= selected_threshold
    ).astype(int)

    # ========================================================
    # 6. DIRECT 2025 REFERENCE CHECK
    # ========================================================

    direct_mask = (
        validation[
            "sample_source"
        ]
        ==
        "lulc_reference_2025"
    ).to_numpy()

    direct_y = (
        y_validation[
            direct_mask
        ]
    )

    direct_probabilities = (
        selected_probabilities[
            direct_mask
        ]
    )

    direct_metrics = (
        calculate_metrics(
            direct_y,
            direct_probabilities,
            selected_threshold,
            selected_model,
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
    # 7. SAVE FINAL VALIDATION PREDICTIONS
    # ========================================================

    final_prediction_table = (
        validation[
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

    final_prediction_table[
        "selected_model"
    ] = (
        selected_model
    )

    final_prediction_table[
        "probability"
    ] = (
        selected_probabilities
    )

    final_prediction_table[
        "selected_threshold"
    ] = (
        selected_threshold
    )

    final_prediction_table[
        "prediction"
    ] = (
        final_predictions
    )

    final_prediction_table[
        "correct"
    ] = (
        final_prediction_table[
            "prediction"
        ]
        ==
        final_prediction_table[
            "target"
        ]
    )

    final_prediction_table.to_csv(
        FINAL_VALIDATION_PREDICTIONS_OUTPUT,
        index=False,
    )

    # ========================================================
    # 8. REFERENCE-LABEL DIAGNOSTICS
    # ========================================================

    diagnostic_rows = []

    for (
        reference_label,
        subset,
    ) in final_prediction_table.groupby(
        "reference_label",
        dropna=False,
    ):

        target_values = (
            subset[
                "target"
            ]
            .astype(int)
        )

        prediction_values = (
            subset[
                "prediction"
            ]
            .astype(int)
        )

        tp = int(
            (
                (target_values == 1)
                &
                (prediction_values == 1)
            ).sum()
        )

        tn = int(
            (
                (target_values == 0)
                &
                (prediction_values == 0)
            ).sum()
        )

        fp = int(
            (
                (target_values == 0)
                &
                (prediction_values == 1)
            ).sum()
        )

        fn = int(
            (
                (target_values == 1)
                &
                (prediction_values == 0)
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
                            prediction_values == 1
                        ).sum()
                    ),

                "predicted_negative":
                    int(
                        (
                            prediction_values == 0
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

    reference_diagnostics = pd.DataFrame(
        diagnostic_rows
    )

    reference_diagnostics.to_csv(
        REFERENCE_DIAGNOSTICS_OUTPUT,
        index=False,
    )

    # ========================================================
    # 9. AUDIT
    # ========================================================

    audit = {
        "test_set_loaded":
            False,

        "model_family_selection_scope":
            "validation_only",

        "model_family_selection_threshold":
            DEFAULT_THRESHOLD,

        "model_family_primary_metric":
            "balanced_accuracy",

        "model_family_tiebreak_1":
            "pr_auc",

        "model_family_tiebreak_2":
            "roc_auc",

        "selected_model_family":
            selected_model,

        "threshold_selection_scope":
            "validation_only",

        "threshold_search_min":
            THRESHOLD_MIN,

        "threshold_search_max":
            THRESHOLD_MAX,

        "threshold_search_step":
            THRESHOLD_STEP,

        "threshold_primary_metric":
            "balanced_accuracy",

        "threshold_tiebreak_1":
            "f1",

        "threshold_tiebreak_2":
            "distance_from_0.50",

        "selected_threshold":
            selected_threshold,

        "final_validation_metrics":
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
                ) in (
                    final_validation_metrics
                    .items()
                )
            },
    }

    with open(
        MODEL_SELECTION_AUDIT_OUTPUT,
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
        "MODEL-FAMILY COMPARISON @ 0.50"
    )

    print(
        "========================================"
    )

    display_columns = [
        "validation_rank",
        "model",
        "balanced_accuracy",
        "accuracy",
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

    print(
        comparison[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nSelected model family:"
    )

    print(
        selected_model
    )

    print(
        "\n========================================"
    )

    print(
        "THRESHOLD SELECTION"
    )

    print(
        "========================================"
    )

    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        "\nTop 10 threshold candidates:"
    )

    threshold_display = [
        "threshold_rank",
        "threshold",
        "balanced_accuracy",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    print(
        threshold_table[
            threshold_display
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nFinal selected-model validation metrics:"
    )

    print(
        pd.DataFrame(
            [
                final_validation_metrics
            ]
        )[
            [
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
        "\nDirect 2025 LULC metrics "
        "at selected threshold:"
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
        "\nOutputs:"
    )

    print(
        MODEL_COMPARISON_OUTPUT
    )

    print(
        THRESHOLD_SEARCH_OUTPUT
    )

    print(
        DIRECT_REFERENCE_METRICS_OUTPUT
    )

    print(
        FINAL_VALIDATION_PREDICTIONS_OUTPUT
    )

    print(
        REFERENCE_DIAGNOSTICS_OUTPUT
    )

    print(
        MODEL_SELECTION_AUDIT_OUTPUT
    )

    print(
        "\n========================================"
    )

    print(
        "MODEL SELECTION STATUS: COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nThe test set was not loaded."
        "\nModel-family selection used validation "
        "data only."
        "\nThreshold selection used validation "
        "data only."
        "\nThe selected model family and threshold "
        "must be frozen before final test evaluation."
    )


if __name__ == "__main__":
    main()