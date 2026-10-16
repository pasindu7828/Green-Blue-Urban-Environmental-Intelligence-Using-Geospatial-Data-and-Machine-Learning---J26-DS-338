"""
STEP 13D - DEVELOPMENT-ONLY MODEL LOCKING AND THRESHOLD SELECTION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Load the best RF and XGBoost configurations selected in Step 13C.
2. Regenerate temporal and spatial out-of-fold scores.
3. Lock the development-winning algorithm using the PREDEFINED
   Step 13C balanced-development PR-AUC criterion.
4. Select an operating classification threshold using TEMPORAL
   development validation only.
5. Evaluate that threshold on spatial development folds as a robustness check.
6. Keep the final 2022 temporal test completely untouched.

Model selection:
- Highest Step 13C balanced development PR-AUC.
- This decision is made BEFORE final 2022 evaluation.

Threshold selection:
- Candidate thresholds are tested from 0.005 to 0.995.
- Primary criterion = mean F1 across temporal validation folds.
- Tie breakers:
    1. Higher minimum temporal-fold F1
    2. Higher mean balanced accuracy
    3. Higher mean precision

IMPORTANT:
- 2022 is NOT evaluated here.
- Threshold is selected from temporal development OOF only.
- Spatial OOF is used only as a robustness check.
- Raw class-weighted model scores are NOT claimed to be
  perfectly calibrated probabilities.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
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

TEMPORAL_MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "splits"
    / "13a_temporal_validation_manifest.csv"
)

BEST_PARAMS_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "model_development"
    / "13c_best_hyperparameters.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "model_development"
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

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. OUTPUTS
# =============================================================================

TEMPORAL_OOF_CSV = (
    OUTPUT_DIR
    / "13d_selected_config_temporal_oof.csv"
)

SPATIAL_OOF_CSV = (
    OUTPUT_DIR
    / "13d_selected_config_spatial_oof.csv"
)

THRESHOLD_SCAN_CSV = (
    REPORT_DIR
    / "13d_temporal_threshold_scan.csv"
)

SELECTED_THRESHOLD_CSV = (
    REPORT_DIR
    / "13d_selected_thresholds.csv"
)

THRESHOLD_FOLD_METRICS_CSV = (
    REPORT_DIR
    / "13d_selected_threshold_fold_metrics.csv"
)

MODEL_DECISION_CSV = (
    REPORT_DIR
    / "13d_development_model_decision.csv"
)

MODEL_DECISION_JSON = (
    OUTPUT_DIR
    / "13d_locked_model_decision.json"
)

QA_CSV = (
    REPORT_DIR
    / "13d_threshold_selection_qa.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "13d_threshold_selection_summary.csv"
)


# =============================================================================
# 3. EXPECTED VALUES
# =============================================================================

EXPECTED_ROWS = 51310
EXPECTED_POSITIVES = 844
EXPECTED_FEATURES = 55

FINAL_TEST_YEAR = 2022
SPATIAL_DEVELOPMENT_MAX_YEAR = 2019

RANDOM_STATE = 2026


# =============================================================================
# 4. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def scale_pos_weight(y):

    y = np.asarray(
        y,
        dtype=np.uint8,
    )

    positives = int(
        np.sum(
            y == 1
        )
    )

    negatives = int(
        np.sum(
            y == 0
        )
    )

    if positives <= 0:
        raise RuntimeError(
            "Training fold has zero positive cases."
        )

    return float(
        negatives
        /
        positives
    )


def build_model(
    model_name,
    parameters,
    y_train,
):

    params = dict(
        parameters
    )

    if model_name == "RandomForest":

        return RandomForestClassifier(
            criterion="gini",
            class_weight="balanced_subsample",
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **params,
        )


    if model_name == "XGBoost":

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
            scale_pos_weight=scale_pos_weight(
                y_train
            ),
            **params,
        )


    raise ValueError(
        f"Unknown model: {model_name}"
    )


def threshold_metrics(
    y_true,
    scores,
    threshold,
):

    y_true = np.asarray(
        y_true,
        dtype=np.uint8,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    predictions = (
        scores
        >=
        threshold
    ).astype(
        np.uint8
    )


    return {
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "predicted_positive_count": int(
            predictions.sum()
        ),
    }


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 13D - DEVELOPMENT-ONLY MODEL LOCKING AND THRESHOLD SELECTION")
print("=" * 100)


for required_file in [
    DATASET_FILE,
    FEATURE_LIST_FILE,
    BASE_ASSIGNMENT_FILE,
    TEMPORAL_MANIFEST_FILE,
    BEST_PARAMS_JSON,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 6. LOAD DATA
# =============================================================================

df = pd.read_csv(
    DATASET_FILE
)

feature_list_df = pd.read_csv(
    FEATURE_LIST_FILE
)

base_assignment_df = pd.read_csv(
    BASE_ASSIGNMENT_FILE
)

temporal_manifest_df = pd.read_csv(
    TEMPORAL_MANIFEST_FILE
)


with open(
    BEST_PARAMS_JSON,
    "r",
    encoding="utf-8",
) as file:

    best_params = json.load(
        file
    )


if len(
    df
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Unexpected dataset row count."
    )


if int(
    df[
        "urban_conversion_target"
    ].sum()
) != EXPECTED_POSITIVES:

    raise RuntimeError(
        "Positive target count mismatch."
    )


if len(
    feature_list_df
) != EXPECTED_FEATURES:

    raise RuntimeError(
        "Frozen feature count mismatch."
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


df = df.set_index(
    "row_id",
    drop=False,
)


print(
    f"\nRows                : "
    f"{len(df):,}"
)

print(
    f"Positive rows       : "
    f"{df['urban_conversion_target'].sum():,}"
)

print(
    f"Frozen predictors   : "
    f"{len(feature_columns)}"
)


# =============================================================================
# 7. LOCK DEVELOPMENT-WINNING MODEL
# =============================================================================

model_names = [
    "RandomForest",
    "XGBoost",
]


for model_name in model_names:

    if model_name not in best_params:

        raise RuntimeError(
            f"Missing Step 13C parameters for {model_name}."
        )


development_winner = max(
    model_names,
    key=lambda model_name:
    float(
        best_params[
            model_name
        ][
            "balanced_development_pr_auc"
        ]
    ),
)


winner_score = float(
    best_params[
        development_winner
    ][
        "balanced_development_pr_auc"
    ]
)


print("\n" + "=" * 100)
print("STEP 13C DEVELOPMENT MODEL DECISION")
print("=" * 100)


for model_name in model_names:

    print(
        f"{model_name:<12} "
        f"candidate="
        f"{best_params[model_name]['candidate_id']} | "
        f"balanced development PR-AUC="
        f"{best_params[model_name]['balanced_development_pr_auc']:.6f}"
    )


print(
    f"\nDevelopment winner : "
    f"{development_winner}"
)

print(
    f"Winner candidate   : "
    f"{best_params[development_winner]['candidate_id']}"
)

print(
    f"Winner score       : "
    f"{winner_score:.6f}"
)


# =============================================================================
# 8. GENERATE TEMPORAL OOF SCORES
# =============================================================================

temporal_oof_records = []


temporal_fold_ids = (
    temporal_manifest_df.loc[
        temporal_manifest_df[
            "role"
        ]
        ==
        "VALIDATION",
        "fold_id",
    ]
    .drop_duplicates()
    .tolist()
)


print("\n" + "=" * 100)
print("REGENERATING TEMPORAL OOF SCORES")
print("=" * 100)


for model_name in model_names:

    parameters = (
        best_params[
            model_name
        ][
            "parameters"
        ]
    )


    for fold_id in temporal_fold_ids:

        fold_manifest = (
            temporal_manifest_df[
                temporal_manifest_df[
                    "fold_id"
                ]
                ==
                fold_id
            ]
        )


        train_ids = (
            fold_manifest.loc[
                fold_manifest[
                    "role"
                ]
                ==
                "TRAIN",
                "row_id",
            ]
            .tolist()
        )


        validation_ids = (
            fold_manifest.loc[
                fold_manifest[
                    "role"
                ]
                ==
                "VALIDATION",
                "row_id",
            ]
            .tolist()
        )


        train_df = df.loc[
            train_ids
        ]


        validation_df = df.loc[
            validation_ids
        ]


        if (
            validation_df[
                "origin_year"
            ]
            ==
            FINAL_TEST_YEAR
        ).any():

            raise RuntimeError(
                "2022 leaked into temporal OOF."
            )


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


        X_validation = (
            validation_df[
                feature_columns
            ]
            .to_numpy(
                dtype=np.float32
            )
        )


        y_validation = (
            validation_df[
                "urban_conversion_target"
            ]
            .to_numpy(
                dtype=np.uint8
            )
        )


        model = build_model(
            model_name=model_name,
            parameters=parameters,
            y_train=y_train,
        )


        model.fit(
            X_train,
            y_train,
        )


        scores = (
            model.predict_proba(
                X_validation
            )[
                :,
                1
            ]
        )


        pr_auc = float(
            average_precision_score(
                y_validation,
                scores,
            )
        )


        roc_auc = float(
            roc_auc_score(
                y_validation,
                scores,
            )
        )


        print(
            f"{model_name:<12} "
            f"{fold_id:<18} "
            f"PR-AUC={pr_auc:.5f} "
            f"ROC-AUC={roc_auc:.5f}"
        )


        for (
            row_id,
            grid_id,
            origin_year,
            actual,
            score,
        ) in zip(
            validation_df[
                "row_id"
            ],
            validation_df[
                "grid_id"
            ],
            validation_df[
                "origin_year"
            ],
            y_validation,
            scores,
        ):

            temporal_oof_records.append(
                {
                    "model": (
                        model_name
                    ),
                    "fold_id": (
                        fold_id
                    ),
                    "row_id": (
                        row_id
                    ),
                    "grid_id": (
                        grid_id
                    ),
                    "origin_year": int(
                        origin_year
                    ),
                    "actual_target": int(
                        actual
                    ),
                    "raw_model_score": float(
                        score
                    ),
                }
            )


temporal_oof_df = pd.DataFrame(
    temporal_oof_records
)


# =============================================================================
# 9. GENERATE SPATIAL OOF SCORES
# =============================================================================

assignment_lookup = (
    base_assignment_df[
        [
            "row_id",
            "spatial_block_1km",
            "spatial_fold",
        ]
    ]
    .drop_duplicates(
        subset="row_id"
    )
)


development_df = (
    df[
        df[
            "origin_year"
        ]
        <=
        SPATIAL_DEVELOPMENT_MAX_YEAR
    ]
    .reset_index(
        drop=True
    )
)


development_df = development_df.merge(
    assignment_lookup,
    on="row_id",
    how="left",
    validate="one_to_one",
)


if (
    development_df[
        "origin_year"
    ]
    ==
    FINAL_TEST_YEAR
).any():

    raise RuntimeError(
        "2022 leaked into spatial OOF."
    )


spatial_oof_records = []


print("\n" + "=" * 100)
print("REGENERATING SPATIAL OOF SCORES")
print("=" * 100)


for model_name in model_names:

    parameters = (
        best_params[
            model_name
        ][
            "parameters"
        ]
    )


    for spatial_fold in sorted(
        development_df[
            "spatial_fold"
        ]
        .astype(
            int
        )
        .unique()
    ):

        train_df = (
            development_df[
                development_df[
                    "spatial_fold"
                ]
                !=
                spatial_fold
            ]
        )


        validation_df = (
            development_df[
                development_df[
                    "spatial_fold"
                ]
                ==
                spatial_fold
            ]
        )


        train_blocks = set(
            train_df[
                "spatial_block_1km"
            ]
        )


        validation_blocks = set(
            validation_df[
                "spatial_block_1km"
            ]
        )


        if len(
            train_blocks
            &
            validation_blocks
        ) != 0:

            raise RuntimeError(
                f"Shared blocks in spatial fold {spatial_fold}."
            )


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


        X_validation = (
            validation_df[
                feature_columns
            ]
            .to_numpy(
                dtype=np.float32
            )
        )


        y_validation = (
            validation_df[
                "urban_conversion_target"
            ]
            .to_numpy(
                dtype=np.uint8
            )
        )


        model = build_model(
            model_name=model_name,
            parameters=parameters,
            y_train=y_train,
        )


        model.fit(
            X_train,
            y_train,
        )


        scores = (
            model.predict_proba(
                X_validation
            )[
                :,
                1
            ]
        )


        print(
            f"{model_name:<12} "
            f"S{spatial_fold} | "
            f"PR-AUC="
            f"{average_precision_score(y_validation, scores):.5f}"
        )


        for (
            row_id,
            grid_id,
            origin_year,
            block_id,
            actual,
            score,
        ) in zip(
            validation_df[
                "row_id"
            ],
            validation_df[
                "grid_id"
            ],
            validation_df[
                "origin_year"
            ],
            validation_df[
                "spatial_block_1km"
            ],
            y_validation,
            scores,
        ):

            spatial_oof_records.append(
                {
                    "model": (
                        model_name
                    ),
                    "fold_id": (
                        f"S{spatial_fold}"
                    ),
                    "row_id": (
                        row_id
                    ),
                    "grid_id": (
                        grid_id
                    ),
                    "origin_year": int(
                        origin_year
                    ),
                    "spatial_block_1km": (
                        block_id
                    ),
                    "actual_target": int(
                        actual
                    ),
                    "raw_model_score": float(
                        score
                    ),
                }
            )


spatial_oof_df = pd.DataFrame(
    spatial_oof_records
)


# =============================================================================
# 10. TEMPORAL THRESHOLD SCAN
# =============================================================================

threshold_values = np.round(
    np.arange(
        0.005,
        1.000,
        0.005,
    ),
    3,
)


threshold_scan_records = []


print("\n" + "=" * 100)
print("TEMPORAL DEVELOPMENT THRESHOLD SCAN")
print("=" * 100)


for model_name in model_names:

    model_oof = (
        temporal_oof_df[
            temporal_oof_df[
                "model"
            ]
            ==
            model_name
        ]
    )


    for threshold in threshold_values:

        fold_metrics = []


        for fold_id, fold_df in (
            model_oof.groupby(
                "fold_id"
            )
        ):

            metrics = threshold_metrics(
                y_true=fold_df[
                    "actual_target"
                ],
                scores=fold_df[
                    "raw_model_score"
                ],
                threshold=threshold,
            )


            fold_metrics.append(
                metrics
            )


        mean_f1 = float(
            np.mean(
                [
                    item[
                        "f1"
                    ]
                    for item in fold_metrics
                ]
            )
        )


        minimum_f1 = float(
            np.min(
                [
                    item[
                        "f1"
                    ]
                    for item in fold_metrics
                ]
            )
        )


        mean_precision = float(
            np.mean(
                [
                    item[
                        "precision"
                    ]
                    for item in fold_metrics
                ]
            )
        )


        mean_recall = float(
            np.mean(
                [
                    item[
                        "recall"
                    ]
                    for item in fold_metrics
                ]
            )
        )


        mean_balanced_accuracy = float(
            np.mean(
                [
                    item[
                        "balanced_accuracy"
                    ]
                    for item in fold_metrics
                ]
            )
        )


        threshold_scan_records.append(
            {
                "model": (
                    model_name
                ),
                "threshold": float(
                    threshold
                ),
                "temporal_fold_count": (
                    len(
                        fold_metrics
                    )
                ),
                "mean_f1": (
                    mean_f1
                ),
                "minimum_fold_f1": (
                    minimum_f1
                ),
                "mean_precision": (
                    mean_precision
                ),
                "mean_recall": (
                    mean_recall
                ),
                "mean_balanced_accuracy": (
                    mean_balanced_accuracy
                ),
            }
        )


threshold_scan_df = pd.DataFrame(
    threshold_scan_records
)


# =============================================================================
# 11. SELECT THRESHOLD PER MODEL
# =============================================================================

selected_threshold_records = []


for model_name in model_names:

    model_scan = (
        threshold_scan_df[
            threshold_scan_df[
                "model"
            ]
            ==
            model_name
        ]
        .sort_values(
            [
                "mean_f1",
                "minimum_fold_f1",
                "mean_balanced_accuracy",
                "mean_precision",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    best_row = (
        model_scan.iloc[
            0
        ]
    )


    selected_threshold_records.append(
        {
            "model": (
                model_name
            ),
            "selected_threshold": float(
                best_row[
                    "threshold"
                ]
            ),
            "mean_temporal_f1": float(
                best_row[
                    "mean_f1"
                ]
            ),
            "minimum_temporal_fold_f1": float(
                best_row[
                    "minimum_fold_f1"
                ]
            ),
            "mean_temporal_precision": float(
                best_row[
                    "mean_precision"
                ]
            ),
            "mean_temporal_recall": float(
                best_row[
                    "mean_recall"
                ]
            ),
            "mean_temporal_balanced_accuracy": float(
                best_row[
                    "mean_balanced_accuracy"
                ]
            ),
            "threshold_selected_using_2022": (
                False
            ),
        }
    )


selected_threshold_df = pd.DataFrame(
    selected_threshold_records
)


print(
    selected_threshold_df.to_string(
        index=False
    )
)


# =============================================================================
# 12. EVALUATE SELECTED THRESHOLDS ON TEMPORAL + SPATIAL OOF
# =============================================================================

threshold_fold_records = []


for model_name in model_names:

    threshold = float(
        selected_threshold_df.loc[
            selected_threshold_df[
                "model"
            ]
            ==
            model_name,
            "selected_threshold",
        ].iloc[
            0
        ]
    )


    for validation_type, oof_df in [
        (
            "TEMPORAL",
            temporal_oof_df,
        ),
        (
            "SPATIAL_1KM",
            spatial_oof_df,
        ),
    ]:

        model_oof = (
            oof_df[
                oof_df[
                    "model"
                ]
                ==
                model_name
            ]
        )


        for fold_id, fold_df in (
            model_oof.groupby(
                "fold_id"
            )
        ):

            metrics = threshold_metrics(
                y_true=fold_df[
                    "actual_target"
                ],
                scores=fold_df[
                    "raw_model_score"
                ],
                threshold=threshold,
            )


            threshold_fold_records.append(
                {
                    "model": (
                        model_name
                    ),
                    "validation_type": (
                        validation_type
                    ),
                    "fold_id": (
                        fold_id
                    ),
                    "selected_threshold": (
                        threshold
                    ),
                    "actual_positive_count": int(
                        fold_df[
                            "actual_target"
                        ].sum()
                    ),
                    **metrics,
                }
            )


threshold_fold_df = pd.DataFrame(
    threshold_fold_records
)


# =============================================================================
# 13. DEVELOPMENT MODEL DECISION
# =============================================================================

winner_threshold = float(
    selected_threshold_df.loc[
        selected_threshold_df[
            "model"
        ]
        ==
        development_winner,
        "selected_threshold",
    ].iloc[
        0
    ]
)


model_decision_df = pd.DataFrame(
    [
        {
            "selected_model": (
                development_winner
            ),
            "selected_candidate_id": (
                best_params[
                    development_winner
                ][
                    "candidate_id"
                ]
            ),
            "development_selection_metric": (
                "balanced_development_pr_auc"
            ),
            "development_selection_score": (
                winner_score
            ),
            "selected_operating_threshold": (
                winner_threshold
            ),
            "threshold_selection_source": (
                "2017, 2019 and 2021 temporal development OOF"
            ),
            "threshold_selection_metric": (
                "Mean temporal-fold F1"
            ),
            "final_2022_test_used_for_model_selection": (
                False
            ),
            "final_2022_test_used_for_threshold_selection": (
                False
            ),
            "raw_score_is_calibrated_probability": (
                False
            ),
        }
    ]
)


locked_model_json = {
    "selected_model": (
        development_winner
    ),
    "selected_candidate_id": (
        best_params[
            development_winner
        ][
            "candidate_id"
        ]
    ),
    "parameters": (
        best_params[
            development_winner
        ][
            "parameters"
        ]
    ),
    "development_selection_metric": (
        "balanced_development_pr_auc"
    ),
    "development_selection_score": (
        winner_score
    ),
    "operating_threshold": (
        winner_threshold
    ),
    "threshold_selection_source": (
        "Temporal development OOF: 2017, 2019, 2021"
    ),
    "final_2022_test_used": (
        False
    ),
    "raw_model_score_is_calibrated_probability": (
        False
    ),
}


# =============================================================================
# 14. QA
# =============================================================================

expected_temporal_rows = (
    temporal_manifest_df[
        temporal_manifest_df[
            "role"
        ]
        ==
        "VALIDATION"
    ].shape[
        0
    ]
    *
    len(
        model_names
    )
)


temporal_oof_count_valid = (
    len(
        temporal_oof_df
    )
    ==
    expected_temporal_rows
)


development_row_count = len(
    development_df
)


expected_spatial_rows = (
    development_row_count
    *
    len(
        model_names
    )
)


spatial_oof_count_valid = (
    len(
        spatial_oof_df
    )
    ==
    expected_spatial_rows
)


temporal_2022_present = bool(
    (
        temporal_oof_df[
            "origin_year"
        ]
        ==
        FINAL_TEST_YEAR
    ).any()
)


spatial_2022_present = bool(
    (
        spatial_oof_df[
            "origin_year"
        ]
        ==
        FINAL_TEST_YEAR
    ).any()
)


thresholds_valid = bool(
    selected_threshold_df[
        "selected_threshold"
    ]
    .between(
        0.0,
        1.0,
        inclusive="neither",
    )
    .all()
)


threshold_metric_values_finite = bool(
    np.isfinite(
        selected_threshold_df[
            [
                "selected_threshold",
                "mean_temporal_f1",
                "mean_temporal_precision",
                "mean_temporal_recall",
                "mean_temporal_balanced_accuracy",
            ]
        ].to_numpy(
            dtype=float
        )
    ).all()
)


winner_exists = (
    development_winner
    in
    model_names
)


final_test_untouched = bool(
    not temporal_2022_present
    and
    not spatial_2022_present
)


model_fitted_for_final_test = False


overall_status = (
    "PASS"
    if all(
        [
            temporal_oof_count_valid,
            spatial_oof_count_valid,
            thresholds_valid,
            threshold_metric_values_finite,
            winner_exists,
            final_test_untouched,
            not model_fitted_for_final_test,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 15. SUMMARY
# =============================================================================

qa_df = pd.DataFrame(
    [
        {
            "temporal_oof_rows": (
                len(
                    temporal_oof_df
                )
            ),
            "temporal_oof_count_valid": (
                temporal_oof_count_valid
            ),
            "spatial_oof_rows": (
                len(
                    spatial_oof_df
                )
            ),
            "spatial_oof_count_valid": (
                spatial_oof_count_valid
            ),
            "selected_thresholds_valid": (
                thresholds_valid
            ),
            "threshold_metric_values_finite": (
                threshold_metric_values_finite
            ),
            "development_winner_found": (
                winner_exists
            ),
            "temporal_2022_present": (
                temporal_2022_present
            ),
            "spatial_2022_present": (
                spatial_2022_present
            ),
            "final_2022_test_untouched": (
                final_test_untouched
            ),
            "raw_score_claimed_as_calibrated_probability": (
                False
            ),
            "final_test_model_fitted": (
                model_fitted_for_final_test
            ),
            "overall_step13d_status": (
                overall_status
            ),
        }
    ]
)


final_summary_df = pd.DataFrame(
    [
        {
            "selected_model": (
                development_winner
            ),
            "selected_candidate": (
                best_params[
                    development_winner
                ][
                    "candidate_id"
                ]
            ),
            "development_selection_score": (
                winner_score
            ),
            "operating_threshold": (
                winner_threshold
            ),
            "threshold_selected_using": (
                "Temporal development OOF only"
            ),
            "temporal_threshold_years": (
                "2017,2019,2021"
            ),
            "spatial_oof_used_for_threshold_selection": (
                False
            ),
            "spatial_oof_used_as_robustness_check": (
                True
            ),
            "final_2022_test_used": (
                False
            ),
            "raw_model_score_calibrated": (
                False
            ),
            "overall_step13d_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 16. SAVE
# =============================================================================

temporal_oof_df.to_csv(
    TEMPORAL_OOF_CSV,
    index=False,
)


spatial_oof_df.to_csv(
    SPATIAL_OOF_CSV,
    index=False,
)


threshold_scan_df.to_csv(
    THRESHOLD_SCAN_CSV,
    index=False,
)


selected_threshold_df.to_csv(
    SELECTED_THRESHOLD_CSV,
    index=False,
)


threshold_fold_df.to_csv(
    THRESHOLD_FOLD_METRICS_CSV,
    index=False,
)


model_decision_df.to_csv(
    MODEL_DECISION_CSV,
    index=False,
)


with open(
    MODEL_DECISION_JSON,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        locked_model_json,
        file,
        indent=2,
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
# 17. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("SELECTED THRESHOLDS")
print("=" * 100)


print(
    selected_threshold_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("SELECTED-THRESHOLD FOLD PERFORMANCE")
print("=" * 100)


print(
    threshold_fold_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("LOCKED DEVELOPMENT MODEL")
print("=" * 100)


print(
    model_decision_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 13D QA")
print("=" * 100)


print(
    f"Temporal OOF row count valid      : "
    f"{temporal_oof_count_valid}"
)

print(
    f"Spatial OOF row count valid       : "
    f"{spatial_oof_count_valid}"
)

print(
    f"Selected thresholds valid         : "
    f"{thresholds_valid}"
)

print(
    f"Threshold metrics finite          : "
    f"{threshold_metric_values_finite}"
)

print(
    f"Development winner found          : "
    f"{winner_exists}"
)

print(
    f"2022 in temporal OOF              : "
    f"{temporal_2022_present}"
)

print(
    f"2022 in spatial OOF               : "
    f"{spatial_2022_present}"
)

print(
    f"2022 final test untouched         : "
    f"{final_test_untouched}"
)

print(
    f"Raw score calibrated probability  : "
    f"False"
)

print(
    f"Final-test model fitted           : "
    f"{model_fitted_for_final_test}"
)

print(
    f"Overall Step 13D status           : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(TEMPORAL_OOF_CSV)}"
)

print(
    f" - {relative_path(SPATIAL_OOF_CSV)}"
)

print(
    f" - {relative_path(THRESHOLD_SCAN_CSV)}"
)

print(
    f" - {relative_path(SELECTED_THRESHOLD_CSV)}"
)

print(
    f" - {relative_path(THRESHOLD_FOLD_METRICS_CSV)}"
)

print(
    f" - {relative_path(MODEL_DECISION_CSV)}"
)

print(
    f" - {relative_path(MODEL_DECISION_JSON)}"
)

print(
    f" - {relative_path(QA_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. The algorithm was selected using Step 13C development PR-AUC only."
)

print(
    "2. The classification threshold was selected using temporal development OOF only."
)

print(
    "3. Spatial OOF was used only to check threshold robustness."
)

print(
    "4. The final 2022 test is still completely untouched."
)

print(
    "5. The selected threshold replaces the arbitrary 0.50 diagnostic threshold."
)

print(
    "6. Raw class-weighted model scores are pressure scores, not yet calibrated probabilities."
)

print(
    "7. After this passes, the selected model/threshold are locked before the one-time 2022 final evaluation."
)

print("=" * 100)