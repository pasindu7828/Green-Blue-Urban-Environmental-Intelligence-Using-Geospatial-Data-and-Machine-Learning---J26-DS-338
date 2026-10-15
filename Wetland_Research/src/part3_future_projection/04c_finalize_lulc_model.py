
from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd
import xgboost

from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PART 3 - HISTORICAL LULC CLASSIFICATION
# SCRIPT 04C - FROZEN MODEL AND HELD-OUT EVALUATION
#
# The model configuration was selected using validation.
# The test partition must NOT be used for further tuning.
#
# Model output classes: 0-5
# Research/GIS class IDs: 1-6
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data/interim/part3_future_projection"
)

MODEL_DIR = (
    ROOT / "models/part3_future_projection"
)

METRIC_DIR = (
    ROOT / "outputs/metrics/part3_future_projection"
)

REPORT_DIR = (
    ROOT / "outputs/reports/part3_future_projection"
)

INPUT_FILE = DATA_DIR / "lulc_spatial_split.csv"

FEATURES = [
    "blue", "green", "red",
    "nir", "swir1", "swir2",
    "ndvi", "mndwi", "ndbi"
]

CLASS_NAMES = {
    1: "Wetland_Marsh",
    2: "Open_Water",
    3: "Built_Up",
    4: "Paddy_Agriculture",
    5: "Other_Vegetation",
    6: "Bare_Fill_Disturbed",
}

CLASSES = list(CLASS_NAMES)
YEARS = [2015, 2020, 2025]

WETLAND_WEIGHT_MULTIPLIER = 1.5
SEED = 42

FROZEN_PARAMETERS = {
    "objective": "multi:softprob",
    "num_class": 6,
    "n_estimators": 250,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.85,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "random_state": SEED,
    "n_jobs": 4,
}


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred,
                labels=CLASSES,
                average="weighted",
                zero_division=0,
            )
        ),
        "wetland_precision": float(
            precision_score(
                y_true == 1,
                y_pred == 1,
                zero_division=0,
            )
        ),
        "wetland_recall": float(
            recall_score(
                y_true == 1,
                y_pred == 1,
                zero_division=0,
            )
        ),
        "wetland_f1": float(
            f1_score(
                y_true == 1,
                y_pred == 1,
                zero_division=0,
            )
        ),
        "wetland_support": int(
            (y_true == 1).sum()
        ),
        "wetlands_correctly_detected": int(
            ((y_true == 1) & (y_pred == 1)).sum()
        ),
        "paddy_incorrectly_predicted_as_wetland": int(
            ((y_true == 4) & (y_pred == 1)).sum()
        ),
    }


def validate_input(df):

    required = FEATURES + [
        "year",
        "class_id",
        "split",
        "spatial_group",
        "raster_row",
        "raster_col",
        "sample_id",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing input columns: {missing}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Missing required input values."
        )

    if not np.isfinite(
        df[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite spectral features detected."
        )

    expected_counts = {
        "train": 485,
        "validation": 347,
        "test": 234,
    }

    if df["split"].value_counts().to_dict() != expected_counts:
        raise ValueError(
            "Input partition sizes do not match "
            "the validated frozen spatial split."
        )

    if df.duplicated(
        ["year", "raster_row", "raster_col"]
    ).any():
        raise ValueError(
            "Duplicate pixel-year records detected."
        )

    if set(df["year"]) != set(YEARS):
        raise ValueError("Unexpected historical years.")

    if set(df["class_id"]) != set(CLASSES):
        raise ValueError("Unexpected LULC classes.")

    training_groups = set(
        df.loc[
            df["split"].isin(["train", "validation"]),
            "spatial_group",
        ]
    )

    test_groups = set(
        df.loc[
            df["split"] == "test",
            "spatial_group",
        ]
    )

    if training_groups & test_groups:
        raise ValueError(
            "Spatial groups overlap with held-out test data."
        )

    print("Input checks: PASS")
    print("Frozen spatial partition: confirmed")


def build_year_class_results(y_true, y_pred, years):

    rows = []

    for year in YEARS:

        mask = years == year

        report = classification_report(
            y_true[mask],
            y_pred[mask],
            labels=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        for class_id in CLASSES:

            result = report[str(class_id)]

            rows.append({
                "year": year,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "precision": float(result["precision"]),
                "recall": float(result["recall"]),
                "f1": float(result["f1-score"]),
                "support": int(result["support"]),
            })

    return pd.DataFrame(rows)


def main():

    print("=" * 65)
    print("PART 3 - HISTORICAL LULC CLASSIFICATION")
    print("SCRIPT 04C - FINAL HELD-OUT EVALUATION")
    print("=" * 65)

    df = pd.read_csv(INPUT_FILE)

    validate_input(df)

    # --------------------------------------------------------
    # 1. PREPARE FROZEN TRAINING AND TEST PARTITIONS
    # --------------------------------------------------------

    development = df[
        df["split"].isin(["train", "validation"])
    ].copy()

    test = df[
        df["split"] == "test"
    ].copy()

    X_development = development[
        FEATURES
    ].to_numpy(dtype=np.float32)

    y_development = development[
        "class_id"
    ].to_numpy(dtype=int) - 1

    X_test = test[
        FEATURES
    ].to_numpy(dtype=np.float32)

    y_test = test[
        "class_id"
    ].to_numpy(dtype=int)

    print("\nDATASET")
    print("-" * 50)
    print("Training + validation:", len(development))
    print("Held-out test:", len(test))
    print("Spectral features:", len(FEATURES))
    print("LULC classes:", len(CLASSES))

    print("\nSelected algorithm: XGBoost Moderate")
    print(
        "Frozen wetland weight multiplier:",
        WETLAND_WEIGHT_MULTIPLIER
    )

    # --------------------------------------------------------
    # 2. BUILD WEIGHTS FROM DEVELOPMENT DATA ONLY
    # --------------------------------------------------------

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_development,
    )

    sample_weights[
        y_development == 0
    ] *= WETLAND_WEIGHT_MULTIPLIER

    # --------------------------------------------------------
    # 3. TRAIN FROZEN CONFIGURATION
    # --------------------------------------------------------

    print("\nTraining frozen XGBoost model...")

    model = XGBClassifier(
        **FROZEN_PARAMETERS
    )

    model.fit(
        X_development,
        y_development,
        sample_weight=sample_weights,
    )

    print("Training completed.")

    # --------------------------------------------------------
    # 4. ONE-TIME HELD-OUT TEST EVALUATION
    # --------------------------------------------------------

    print("\nEvaluating frozen model on held-out test...")

    predictions = (
        model.predict(X_test).astype(int) + 1
    )

    summary = evaluate(
        y_test,
        predictions,
    )

    print("\n" + "=" * 65)
    print("HELD-OUT TEST RESULTS")
    print("=" * 65)

    for key, value in summary.items():

        if isinstance(value, float):
            print(f"{key:42s}: {value:.4f}")
        else:
            print(f"{key:42s}: {value}")

    # --------------------------------------------------------
    # 5. CLASS-SPECIFIC EVALUATION
    # --------------------------------------------------------

    print("\nCLASSIFICATION REPORT")
    print("-" * 65)

    report = classification_report(
        y_test,
        predictions,
        labels=CLASSES,
        target_names=[
            CLASS_NAMES[c] for c in CLASSES
        ],
        digits=4,
        zero_division=0,
    )

    print(report)

    # --------------------------------------------------------
    # 6. CONFUSION MATRIX
    # --------------------------------------------------------

    confusion = confusion_matrix(
        y_test,
        predictions,
        labels=CLASSES,
    )

    confusion_df = pd.DataFrame(
        confusion,
        index=[
            f"actual_{c}_{CLASS_NAMES[c]}"
            for c in CLASSES
        ],
        columns=[
            f"predicted_{c}_{CLASS_NAMES[c]}"
            for c in CLASSES
        ],
    )

    print("\nCONFUSION MATRIX")
    print(confusion_df.to_string())

    # --------------------------------------------------------
    # 7. YEAR-SPECIFIC EVALUATION
    # --------------------------------------------------------

    year_class = build_year_class_results(
        y_test,
        predictions,
        test["year"].to_numpy(),
    )

    wetland_years = year_class[
        year_class["class_id"] == 1
    ].copy()

    print("\nWETLAND PERFORMANCE BY YEAR")

    print(
        wetland_years[
            [
                "year", "precision",
                "recall", "f1", "support",
            ]
        ].round(4).to_string(index=False)
    )

    # --------------------------------------------------------
    # 8. SAVE IMPORTANT RESEARCH EVIDENCE
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRIC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "lulc_xgb_weighted_final_candidate.joblib"
    )

    summary_path = (
        METRIC_DIR
        / "script04c_heldout_test_metrics.json"
    )

    confusion_path = (
        METRIC_DIR
        / "script04c_heldout_confusion_matrix.csv"
    )

    yearly_path = (
        METRIC_DIR
        / "script04c_heldout_class_year_metrics.csv"
    )

    predictions_path = (
        METRIC_DIR
        / "script04c_heldout_predictions.csv"
    )

    report_path = (
        REPORT_DIR
        / "script04c_heldout_evaluation_report.txt"
    )

    metadata_path = (
        REPORT_DIR
        / "script04c_final_candidate_metadata.json"
    )

    joblib.dump(model, model_path)

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    confusion_df.to_csv(confusion_path)

    year_class.to_csv(
        yearly_path,
        index=False,
    )

    test_predictions = test[
        [
            "sample_id",
            "year",
            "class_id",
            "raster_row",
            "raster_col",
            "spatial_group",
        ]
    ].copy()

    test_predictions["predicted_class_id"] = predictions

    test_predictions["correct"] = (
        test_predictions["class_id"]
        == test_predictions["predicted_class_id"]
    )

    test_predictions.to_csv(
        predictions_path,
        index=False,
    )

    metadata = {
        "component": "Part 3 - Future Wetland Boundary Projection",
        "stage": "Final historical LULC classifier evaluation",
        "created_utc":
            datetime.now(timezone.utc).isoformat(),
        "selected_model": "XGBoost Moderate",
        "selected_before_test_evaluation": True,
        "selected_configuration": FROZEN_PARAMETERS,
        "wetland_weight_multiplier":
            WETLAND_WEIGHT_MULTIPLIER,
        "selection_basis": (
            "Highest observed validation macro-F1 "
            "among tested weighted configurations, "
            "with wetland-specific error trade-offs reviewed."
        ),
        "training_includes_original_validation": True,
        "development_observations": len(development),
        "heldout_test_observations": len(test),
        "features_in_order": FEATURES,
        "research_class_ids": CLASS_NAMES,
        "model_class_encoding": (
            "XGBoost output 0-5; add 1 for GIS classes 1-6."
        ),
        "heldout_test_metrics": summary,
        "model_acceptance": "PENDING_RESEARCH_REVIEW",
        "software": {
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            "Single buffered geographic hold-out.",
            "Small year-specific held-out class sample sizes.",
            "Historical reference interpretation uncertainty.",
            "Four documented placemark-name/year discrepancies.",
            "Classification errors can propagate into "
            "CA-Markov transitions and projections.",
            "Test results must not be used for further tuning.",
        ],
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    text_lines = [
        "PART 3 - HELD-OUT LULC CLASSIFIER EVALUATION",
        "=" * 65,
        "",
        "Frozen model: XGBoost Moderate",
        "Wetland weighting multiplier: 1.5",
        f"Development observations: {len(development)}",
        f"Held-out test observations: {len(test)}",
        "",
        "OVERALL TEST METRICS:",
    ]

    for key, value in summary.items():
        text_lines.append(f"{key}: {value}")

    text_lines.extend([
        "",
        "CLASSIFICATION REPORT:",
        report,
        "",
        "WETLAND METRICS BY YEAR:",
        wetland_years.to_string(index=False),
        "",
        "IMPORTANT LIMITATIONS:",
        "The historical LULC maps remain estimated maps.",
        "Low wetland recall may produce unreliable loss/gain.",
        "Model acceptance requires research review.",
        "Do not tune further using these test results.",
    ])

    report_path.write_text(
        "\n".join(text_lines) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 9. FINAL TERMINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SCRIPT 04C COMPLETED")
    print("=" * 65)

    print("Saved model:", model_path)
    print("Saved metrics:", summary_path)
    print("Saved confusion matrix:", confusion_path)
    print("Saved year/class metrics:", yearly_path)
    print("Saved test predictions:", predictions_path)
    print("Saved report:", report_path)
    print("Saved metadata:", metadata_path)

    print("\nMODEL ACCEPTANCE: PENDING RESULTS REVIEW")
    print("HELD-OUT TEST EVALUATION: COMPLETED")
    print("DO NOT RETUNE USING TEST RESULTS")


if __name__ == "__main__":
    main()
