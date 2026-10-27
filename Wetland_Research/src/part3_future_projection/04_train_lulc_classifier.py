
from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
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
# SCRIPT 04 - RANDOM FOREST MODEL DEVELOPMENT
#
# IMPORTANT:
# - Train using TRAIN observations only.
# - Select model using VALIDATION observations only.
# - Do not evaluate the held-out TEST set in this script.
# - Use identical spectral features for all three years.
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "interim" / "part3_future_projection"
)

MODEL_DIR = (
    ROOT / "models" / "part3_future_projection"
)

METRIC_DIR = (
    ROOT / "outputs" / "metrics" / "part3_future_projection"
)

REPORT_DIR = (
    ROOT / "outputs" / "reports" / "part3_future_projection"
)

INPUT_FILE = DATA_DIR / "lulc_spatial_split.csv"

FEATURES = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "ndvi",
    "mndwi",
    "ndbi",
]

CLASSES = [1, 2, 3, 4, 5, 6]

CLASS_NAMES = {
    1: "Wetland_Marsh",
    2: "Open_Water",
    3: "Built_Up",
    4: "Paddy_Agriculture",
    5: "Other_Vegetation",
    6: "Bare_Fill_Disturbed",
}

YEARS = [2015, 2020, 2025]

RANDOM_STATE = 42


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CLASSES,
                average="weighted",
                zero_division=0,
            )
        ),
        "wetland_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=[1],
                average="macro",
                zero_division=0,
            )
        ),
        "wetland_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=[1],
                average="macro",
                zero_division=0,
            )
        ),
        "wetland_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[1],
                average="macro",
                zero_division=0,
            )
        ),
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * 55)

    for key, value in metrics.items():
        print(f"{key:25s}: {value:.4f}")


def class_year_report(y_true, y_pred, years):
    records = []

    for year in YEARS:
        mask = years == year

        actual = y_true[mask]
        predicted = y_pred[mask]

        report = classification_report(
            actual,
            predicted,
            labels=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        for class_id in CLASSES:
            class_result = report[str(class_id)]

            records.append({
                "year": year,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "precision": float(
                    class_result["precision"]
                ),
                "recall": float(
                    class_result["recall"]
                ),
                "f1": float(
                    class_result["f1-score"]
                ),
                "support": int(
                    class_result["support"]
                ),
            })

    return pd.DataFrame(records)


# ============================================================
# DATA VALIDATION
# ============================================================

def load_and_validate_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required_columns = (
        FEATURES
        + [
            "sample_id",
            "year",
            "class_id",
            "raster_row",
            "raster_col",
            "spatial_group",
            "split",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if len(df) != 1066:
        raise ValueError(
            f"Expected 1066 records, found {len(df)}"
        )

    expected_counts = {
        "train": 485,
        "validation": 347,
        "test": 234,
    }

    actual_counts = (
        df["split"].value_counts().to_dict()
    )

    if actual_counts != expected_counts:
        raise ValueError(
            f"Unexpected split counts: {actual_counts}"
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Missing values detected in required columns"
        )

    if not np.isfinite(
        df[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite Landsat features detected"
        )

    if set(df["year"]) != set(YEARS):
        raise ValueError(
            "Unexpected historical years"
        )

    if set(df["class_id"]) != set(CLASSES):
        raise ValueError(
            "Unexpected LULC class IDs"
        )

    if df.duplicated(
        subset=[
            "year",
            "raster_row",
            "raster_col",
        ]
    ).any():
        raise ValueError(
            "Duplicate pixel-year observations detected"
        )

    groups = {
        name: set(
            df.loc[
                df["split"] == name,
                "spatial_group"
            ]
        )
        for name in expected_counts
    }

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        if groups[first] & groups[second]:
            raise ValueError(
                f"Spatial group overlap: {first}/{second}"
            )

    return df


# ============================================================
# MAIN MODEL DEVELOPMENT
# ============================================================

def main():
    print("=" * 65)
    print("PART 3 - FUTURE WETLAND BOUNDARY PROJECTION")
    print("SCRIPT 04 - RANDOM FOREST LULC CLASSIFICATION")
    print("=" * 65)

    df = load_and_validate_data()

    train = df[
        df["split"] == "train"
    ].copy()

    validation = df[
        df["split"] == "validation"
    ].copy()

    test_count = int(
        (df["split"] == "test").sum()
    )

    X_train = train[FEATURES].to_numpy(
        dtype=np.float32
    )
    y_train = train["class_id"].to_numpy()

    X_val = validation[FEATURES].to_numpy(
        dtype=np.float32
    )
    y_val = validation["class_id"].to_numpy()

    val_years = validation["year"].to_numpy()

    print("\nDATASET SUMMARY")
    print("-" * 55)

    print("Training observations:", len(train))
    print("Validation observations:", len(validation))
    print("Reserved test observations:", test_count)
    print("Landsat features:", len(FEATURES))
    print("Land-cover classes:", len(CLASSES))

    print("\nIMPORTANT:")
    print("The held-out TEST partition is not evaluated.")
    print("Model selection uses VALIDATION only.")

    # --------------------------------------------------------
    # 1. MAJORITY-CLASS BASELINE
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("BASELINE CLASSIFIER")
    print("=" * 65)

    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(
        X_train,
        y_train
    )

    baseline_predictions = baseline.predict(
        X_val
    )

    baseline_metrics = calculate_metrics(
        y_val,
        baseline_predictions
    )

    print_metrics(
        "MAJORITY-CLASS BASELINE - VALIDATION",
        baseline_metrics
    )

    # --------------------------------------------------------
    # 2. PREDEFINED RANDOM FOREST CONFIGURATIONS
    # --------------------------------------------------------

    # A small predefined comparison is used to limit
    # unnecessary model-selection complexity.
    #
    # No test data are used for configuration selection.

    configurations = {
        "RF_01": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },

        "RF_02": {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },

        "RF_03": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
    }

    trained_models = {}
    comparison_rows = []

    print("\n" + "=" * 65)
    print("RANDOM FOREST MODEL COMPARISON")
    print("=" * 65)

    for model_name, parameters in configurations.items():
        print(f"\nTraining {model_name}...")

        model = RandomForestClassifier(
            **parameters,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train
        )

        train_predictions = model.predict(
            X_train
        )

        val_predictions = model.predict(
            X_val
        )

        train_metrics = calculate_metrics(
            y_train,
            train_predictions
        )

        val_metrics = calculate_metrics(
            y_val,
            val_predictions
        )

        trained_models[model_name] = model

        comparison_rows.append({
            "model": model_name,
            "train_accuracy":
                train_metrics["accuracy"],
            "train_macro_f1":
                train_metrics["macro_f1"],
            "validation_accuracy":
                val_metrics["accuracy"],
            "validation_balanced_accuracy":
                val_metrics["balanced_accuracy"],
            "validation_macro_f1":
                val_metrics["macro_f1"],
            "validation_weighted_f1":
                val_metrics["weighted_f1"],
            "validation_wetland_precision":
                val_metrics["wetland_precision"],
            "validation_wetland_recall":
                val_metrics["wetland_recall"],
            "validation_wetland_f1":
                val_metrics["wetland_f1"],
            "train_validation_macro_f1_gap":
                train_metrics["macro_f1"]
                - val_metrics["macro_f1"],
        })

        print(
            "Training macro-F1:",
            round(
                train_metrics["macro_f1"], 4
            )
        )

        print(
            "Validation macro-F1:",
            round(
                val_metrics["macro_f1"], 4
            )
        )

        print(
            "Validation balanced accuracy:",
            round(
                val_metrics["balanced_accuracy"], 4
            )
        )

        print(
            "Validation Wetland F1:",
            round(
                val_metrics["wetland_f1"], 4
            )
        )

    # --------------------------------------------------------
    # 3. SELECT USING VALIDATION RESULTS ONLY
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        comparison_rows
    )

    # Primary: validation macro-F1.
    # Tie-breakers: validation wetland F1,
    # then validation balanced accuracy.
    #
    # This rule is predetermined and uses no test results.

    ranking = comparison.sort_values(
        [
            "validation_macro_f1",
            "validation_wetland_f1",
            "validation_balanced_accuracy",
            "model",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    selected_name = str(
        ranking.iloc[0]["model"]
    )

    selected_model = trained_models[
        selected_name
    ]

    selected_predictions = selected_model.predict(
        X_val
    )

    selected_metrics = calculate_metrics(
        y_val,
        selected_predictions
    )

    print("\n" + "=" * 65)
    print("SELECTED MODEL - VALIDATION RESULTS")
    print("=" * 65)

    print("Selected model:", selected_name)

    print(
        "\nSelection rule: highest validation macro-F1, "
        "followed by Wetland F1 and balanced accuracy."
    )

    print_metrics(
        "SELECTED RANDOM FOREST",
        selected_metrics
    )

    # --------------------------------------------------------
    # 4. CLASS-WISE AND YEAR-WISE EVALUATION
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("VALIDATION - CLASSIFICATION REPORT")
    print("=" * 65)

    report_text = classification_report(
        y_val,
        selected_predictions,
        labels=CLASSES,
        target_names=[
            CLASS_NAMES[i]
            for i in CLASSES
        ],
        digits=4,
        zero_division=0,
    )

    print(report_text)

    yearly_results = class_year_report(
        y_val,
        selected_predictions,
        val_years,
    )

    print("\nYEAR-SPECIFIC WETLAND PERFORMANCE")

    wetland_rows = yearly_results[
        yearly_results["class_id"] == 1
    ]

    print(
        wetland_rows[
            [
                "year",
                "precision",
                "recall",
                "f1",
                "support",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 5. VALIDATION CONFUSION MATRIX
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_val,
        selected_predictions,
        labels=CLASSES,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"actual_{i}_{CLASS_NAMES[i]}"
            for i in CLASSES
        ],
        columns=[
            f"predicted_{i}_{CLASS_NAMES[i]}"
            for i in CLASSES
        ],
    )

    # --------------------------------------------------------
    # 6. FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance":
            selected_model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    print("\nFEATURE IMPORTANCE")

    print(
        importance_df.to_string(index=False)
    )

    # --------------------------------------------------------
    # 7. SAVE IMPORTANT RESEARCH OUTPUTS
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METRIC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODEL_DIR / "lulc_selected_random_forest.joblib"
    )

    comparison_path = (
        METRIC_DIR / "script04_rf_model_comparison.csv"
    )

    confusion_path = (
        METRIC_DIR / "script04_validation_confusion_matrix.csv"
    )

    yearly_path = (
        METRIC_DIR / "script04_validation_class_year_metrics.csv"
    )

    importance_path = (
        METRIC_DIR / "script04_feature_importance.csv"
    )

    prediction_path = (
        METRIC_DIR / "script04_validation_predictions.csv"
    )

    metadata_path = (
        REPORT_DIR / "script04_lulc_model_metadata.json"
    )

    report_path = (
        REPORT_DIR / "script04_lulc_validation_report.txt"
    )

    joblib.dump(
        selected_model,
        model_path
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    matrix_df.to_csv(
        confusion_path
    )

    yearly_results.to_csv(
        yearly_path,
        index=False
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    validation_output = validation[
        [
            "sample_id",
            "year",
            "class_id",
            "raster_row",
            "raster_col",
            "spatial_group",
        ]
    ].copy()

    validation_output["predicted_class_id"] = (
        selected_predictions
    )

    validation_output["correct_prediction"] = (
        validation_output["class_id"]
        == validation_output["predicted_class_id"]
    )

    validation_output.to_csv(
        prediction_path,
        index=False
    )

    metadata = {
        "component": "Part 3 - Future Wetland Boundary Projection",
        "stage": "Historical LULC classification",
        "created_utc":
            datetime.now(timezone.utc).isoformat(),
        "classifier": "RandomForestClassifier",
        "selected_model": selected_name,
        "selected_parameters":
            selected_model.get_params(),
        "model_selection_rule": (
            "Highest validation macro-F1; "
            "tie-breakers: wetland F1, balanced accuracy."
        ),
        "feature_columns": FEATURES,
        "class_definitions": {
            str(k): v
            for k, v in CLASS_NAMES.items()
        },
        "historical_years": YEARS,
        "training_observations": len(train),
        "validation_observations":
            len(validation),
        "reserved_test_observations":
            test_count,
        "held_out_test_evaluated": False,
        "spatial_split_method": (
            "1 km geographic blocks with "
            "150 m pixel-centre exclusion buffer"
        ),
        "random_state": RANDOM_STATE,
        "validation_baseline": baseline_metrics,
        "selected_validation_metrics":
            selected_metrics,
        "software": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            "The LULC labels originate from historical "
            "reference interpretation and require careful "
            "consideration of temporal label uncertainty.",
            "Four original placemark names have a year prefix "
            "that disagrees with their KMZ dataset year.",
            "Spatial buffering reduces proximity but does "
            "not guarantee full statistical independence.",
            "Class-specific performance depends on the number "
            "of independent validation locations.",
            "The 2025 classified LULC map will be an estimated "
            "historical reference, not perfect ground truth "
            "for CA-Markov hindcast evaluation.",
            "The held-out test data have not been used "
            "for training or model selection.",
        ],
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    text_lines = [
        "PART 3 - HISTORICAL LULC CLASSIFICATION",
        "=" * 60,
        "",
        f"Selected model: {selected_name}",
        "Classification method: Random Forest",
        "Historical years: 2015, 2020, 2025",
        f"Training observations: {len(train)}",
        f"Validation observations: {len(validation)}",
        f"Reserved test observations: {test_count}",
        "",
        "MODEL SELECTION:",
        "Validation macro-F1, then Wetland F1,",
        "then balanced accuracy.",
        "",
        "MAJORITY-CLASS BASELINE:",
    ]

    for key, value in baseline_metrics.items():
        text_lines.append(
            f"{key}: {value:.4f}"
        )

    text_lines.extend([
        "",
        "SELECTED MODEL VALIDATION:",
    ])

    for key, value in selected_metrics.items():
        text_lines.append(
            f"{key}: {value:.4f}"
        )

    text_lines.extend([
        "",
        "CLASSIFICATION REPORT:",
        report_text,
        "",
        "RESEARCH LIMITATIONS:",
        "Historical reference labels contain some naming",
        "discrepancies and temporal uncertainty.",
        "The spatial buffer does not prove full independence.",
        "The held-out test data remain reserved.",
        "",
        "SCRIPT 04: TRAINING AND VALIDATION COMPLETED",
        "MODEL ACCEPTANCE: REQUIRES RESULTS REVIEW",
    ])

    report_path.write_text(
        "\n".join(text_lines) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 8. FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SCRIPT 04 FINAL SUMMARY")
    print("=" * 65)

    print("Selected model:", selected_name)

    print(
        "Baseline validation macro-F1:",
        round(baseline_metrics["macro_f1"], 4)
    )

    print(
        "Selected validation macro-F1:",
        round(selected_metrics["macro_f1"], 4)
    )

    print(
        "Selected validation balanced accuracy:",
        round(
            selected_metrics["balanced_accuracy"],
            4
        )
    )

    print(
        "Selected validation Wetland F1:",
        round(
            selected_metrics["wetland_f1"],
            4
        )
    )

    print("\nSaved model:", model_path)
    print("Saved model comparison:", comparison_path)
    print("Saved confusion matrix:", confusion_path)
    print("Saved per-year metrics:", yearly_path)
    print("Saved feature importance:", importance_path)
    print("Saved validation predictions:", prediction_path)
    print("Saved model metadata:", metadata_path)
    print("Saved validation report:", report_path)

    print("\nSCRIPT 04 TRAINING AND VALIDATION: COMPLETED")
    print("MODEL ACCEPTANCE: PENDING RESULTS REVIEW")
    print("HELD-OUT TEST: NOT EVALUATED")


if __name__ == "__main__":
    main()
