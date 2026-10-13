from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "part2_candidate_wetland"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "part2_candidate_wetland"
)


TRAIN_FILE = (
    DATA_DIR
    / "splits"
    / "part2_candidate_wetland"
    / "train_samples.csv"
)

VALIDATION_FILE = (
    DATA_DIR
    / "splits"
    / "part2_candidate_wetland"
    / "validation_samples.csv"
)

FEATURE_METADATA = (
    DATA_DIR
    / "processed"
    / "part2_candidate_wetland"
    / "feature_stack_metadata.csv"
)


OUTPUT_CSV = (
    REPORT_DIR
    / "23_xgboost_train_validation_learning_curve.csv"
)

OUTPUT_PLOT = (
    FIGURE_DIR
    / "23_xgboost_train_validation_learning_curve.png"
)


# ============================================================
# FIXED MODEL SETTINGS
#
# These are the already-selected XGB_03 settings.
# We are NOT tuning using the test set.
# ============================================================

TREE_COUNTS = [
    10,
    25,
    50,
    75,
    100,
    150,
    200,
    250,
    300,
]

THRESHOLD = 0.50


# ============================================================
# HELPER
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
):
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return {
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
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "XGBOOST OVERFITTING DIAGNOSTIC"
    )

    print(
        "TRAINING + VALIDATION ONLY"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train = pd.read_csv(
        TRAIN_FILE
    )

    validation = pd.read_csv(
        VALIDATION_FILE
    )

    feature_metadata = pd.read_csv(
        FEATURE_METADATA
    )

    features = (
        feature_metadata[
            "feature"
        ]
        .tolist()
    )

    print(
        f"\nTraining samples: "
        f"{len(train)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation)}"
    )

    print(
        f"Features: "
        f"{len(features)}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The locked test set is NOT loaded "
        "or used in this diagnostic."
    )

    X_train = train[
        features
    ]

    y_train = train[
        "target"
    ]

    X_validation = validation[
        features
    ]

    y_validation = validation[
        "target"
    ]

    rows = []

    # --------------------------------------------------------
    # Train same XGB_03 structure using different numbers
    # of boosting trees.
    #
    # Purpose:
    # See when training performance keeps improving while
    # validation performance stops improving.
    # --------------------------------------------------------

    for n_estimators in TREE_COUNTS:

        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        train_probability = (
            model.predict_proba(
                X_train
            )[:, 1]
        )

        validation_probability = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        train_metrics = calculate_metrics(
            y_train,
            train_probability,
        )

        validation_metrics = calculate_metrics(
            y_validation,
            validation_probability,
        )

        row = {
            "n_estimators":
                n_estimators,

            "train_accuracy":
                train_metrics[
                    "accuracy"
                ],

            "validation_accuracy":
                validation_metrics[
                    "accuracy"
                ],

            "accuracy_gap":
                (
                    train_metrics[
                        "accuracy"
                    ]
                    -
                    validation_metrics[
                        "accuracy"
                    ]
                ),

            "train_balanced_accuracy":
                train_metrics[
                    "balanced_accuracy"
                ],

            "validation_balanced_accuracy":
                validation_metrics[
                    "balanced_accuracy"
                ],

            "balanced_accuracy_gap":
                (
                    train_metrics[
                        "balanced_accuracy"
                    ]
                    -
                    validation_metrics[
                        "balanced_accuracy"
                    ]
                ),

            "train_precision":
                train_metrics[
                    "precision"
                ],

            "validation_precision":
                validation_metrics[
                    "precision"
                ],

            "train_recall":
                train_metrics[
                    "recall"
                ],

            "validation_recall":
                validation_metrics[
                    "recall"
                ],

            "train_f1":
                train_metrics[
                    "f1"
                ],

            "validation_f1":
                validation_metrics[
                    "f1"
                ],
        }

        rows.append(
            row
        )

        print(
            f"\nTrees: {n_estimators}"
        )

        print(
            f"Train Accuracy: "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"Train Balanced Accuracy: "
            f"{train_metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Validation Balanced Accuracy: "
            f"{validation_metrics['balanced_accuracy']:.4f}"
        )

    results = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Plot balanced accuracy learning behaviour
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        results[
            "n_estimators"
        ],
        results[
            "train_balanced_accuracy"
        ],
        marker="o",
        label="Training",
    )

    plt.plot(
        results[
            "n_estimators"
        ],
        results[
            "validation_balanced_accuracy"
        ],
        marker="o",
        label="Validation",
    )

    plt.xlabel(
        "Number of XGBoost Trees"
    )

    plt.ylabel(
        "Balanced Accuracy"
    )

    plt.title(
        "XGB_03 Training vs Validation Learning Behaviour"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT,
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # Best validation point
    # --------------------------------------------------------

    best_index = (
        results[
            "validation_balanced_accuracy"
        ]
        .idxmax()
    )

    best_row = results.loc[
        best_index
    ]

    final_row = (
        results[
            results[
                "n_estimators"
            ]
            ==
            300
        ]
        .iloc[0]
    )

    print(
        "\n========================================"
    )

    print(
        "DIAGNOSTIC SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        "\nBest validation balanced accuracy:"
    )

    print(
        f"Trees: "
        f"{int(best_row['n_estimators'])}"
    )

    print(
        f"Training BA: "
        f"{best_row['train_balanced_accuracy']:.4f}"
    )

    print(
        f"Validation BA: "
        f"{best_row['validation_balanced_accuracy']:.4f}"
    )

    print(
        f"Gap: "
        f"{best_row['balanced_accuracy_gap']:.4f}"
    )

    print(
        "\nOriginal XGB_03 at 300 trees:"
    )

    print(
        f"Training Accuracy: "
        f"{final_row['train_accuracy']:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{final_row['validation_accuracy']:.4f}"
    )

    print(
        f"Training Balanced Accuracy: "
        f"{final_row['train_balanced_accuracy']:.4f}"
    )

    print(
        f"Validation Balanced Accuracy: "
        f"{final_row['validation_balanced_accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy Gap: "
        f"{final_row['balanced_accuracy_gap']:.4f}"
    )

    print(
        "\nOutputs:"
    )

    print(
        OUTPUT_CSV
    )

    print(
        OUTPUT_PLOT
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This script is diagnostic only."
    )

    print(
        "It does NOT change the frozen final model, "
        "threshold, test result, or candidate outputs."
    )

    print(
        "The locked test set was not used."
    )

    print(
        "\n========================================"
    )

    print(
        "OVERFITTING DIAGNOSTIC: COMPLETE"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()