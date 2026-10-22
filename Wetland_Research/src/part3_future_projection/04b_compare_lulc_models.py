
from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# ============================================================
# PART 3 - HISTORICAL LULC CLASSIFICATION
# SCRIPT 04B - CONTROLLED SVM / XGBOOST COMPARISON
#
# Train partition: training only
# Model comparison: validation only
# Final test partition: strictly reserved
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/interim/part3_future_projection"
MODEL_DIR = ROOT / "models/part3_future_projection"
METRIC_DIR = ROOT / "outputs/metrics/part3_future_projection"
REPORT_DIR = ROOT / "outputs/reports/part3_future_projection"

INPUT_FILE = DATA_DIR / "lulc_spatial_split.csv"
RF_FILE = METRIC_DIR / "script04_rf_model_comparison.csv"

FEATURES = [
    "blue", "green", "red", "nir", "swir1",
    "swir2", "ndvi", "mndwi", "ndbi"
]

CLASSES = [1, 2, 3, 4, 5, 6]
YEARS = [2015, 2020, 2025]

SEED = 42


def metrics(y_true, y_pred):
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASSES,
            zero_division=0
        )
    )

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
                zero_division=0
            )
        ),
        "wetland_precision": float(precision[0]),
        "wetland_recall": float(recall[0]),
        "wetland_f1": float(f1[0]),
        "wetland_support": int(support[0]),
        "false_wetland_from_paddy": int(
            ((y_true == 4) & (y_pred == 1)).sum()
        ),
        "wetlands_predicted_as_vegetation": int(
            ((y_true == 1) & (y_pred == 5)).sum()
        ),
        "wetlands_predicted_as_paddy": int(
            ((y_true == 1) & (y_pred == 4)).sum()
        ),
        "wetlands_predicted_as_water": int(
            ((y_true == 1) & (y_pred == 2)).sum()
        ),
    }


def check_data(df):
    required = FEATURES + [
        "year", "class_id", "split",
        "spatial_group",
        "raster_row", "raster_col"
    ]

    if not set(required).issubset(df.columns):
        raise ValueError("Missing required columns.")

    if df[required].isna().any().any():
        raise ValueError("Missing data detected.")

    if len(df) != 1066:
        raise ValueError("Unexpected dataset size.")

    expected = {
        "train": 485,
        "validation": 347,
        "test": 234
    }

    if df["split"].value_counts().to_dict() != expected:
        raise ValueError("Unexpected split distribution.")

    if df.duplicated(
        ["year", "raster_row", "raster_col"]
    ).any():
        raise ValueError("Duplicate pixel-year observations.")

    for name in expected:
        part = df[df["split"] == name]

        if set(part["year"]) != set(YEARS):
            raise ValueError(f"Missing years in {name}.")

        if set(part["class_id"]) != set(CLASSES):
            raise ValueError(f"Missing classes in {name}.")

    train_groups = set(
        df.loc[df["split"] == "train", "spatial_group"]
    )
    val_groups = set(
        df.loc[df["split"] == "validation", "spatial_group"]
    )
    test_groups = set(
        df.loc[df["split"] == "test", "spatial_group"]
    )

    if (
        train_groups & val_groups
        or train_groups & test_groups
        or val_groups & test_groups
    ):
        raise ValueError("Spatial block overlap detected.")

    if not np.isfinite(
        df[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite spectral features.")


def main():
    print("=" * 65)
    print("PART 3 - HISTORICAL LULC CLASSIFICATION")
    print("SCRIPT 04B - SVM AND XGBOOST COMPARISON")
    print("=" * 65)

    df = pd.read_csv(INPUT_FILE)
    check_data(df)

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()

    X_train = train[FEATURES].to_numpy(
        dtype=np.float32
    )
    y_train = train["class_id"].to_numpy(
        dtype=int
    )

    X_val = val[FEATURES].to_numpy(
        dtype=np.float32
    )
    y_val = val["class_id"].to_numpy(
        dtype=int
    )

    # XGBoost requires zero-based multiclass labels.
    y_train_xgb = y_train - 1

    balanced_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train_xgb
    )

    print("\nTraining:", len(train))
    print("Validation:", len(val))
    print("Reserved test:", 234)
    print("Features:", len(FEATURES))

    print("\nTEST DATA WILL NOT BE USED.")

    # --------------------------------------------------------
    # PREDEFINED MODELS
    # --------------------------------------------------------

    models = {
        "SVM_LINEAR": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="linear",
                C=1.0,
                class_weight="balanced"
            ))
        ]),

        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced"
            ))
        ]),

        "XGB_SHALLOW": XGBClassifier(
            objective="multi:softprob",
            num_class=6,
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.85,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=4
        ),

        "XGB_MODERATE": XGBClassifier(
            objective="multi:softprob",
            num_class=6,
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.85,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=4
        ),
    }

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    prediction_rows = []
    yearly_rows = []

    # --------------------------------------------------------
    # TRAIN AND COMPARE
    # --------------------------------------------------------

    for name, model in models.items():
        print("\n" + "-" * 65)
        print("MODEL:", name)
        print("-" * 65)

        is_xgb = name.startswith("XGB")

        if is_xgb:
            model.fit(
                X_train,
                y_train_xgb,
                sample_weight=balanced_weights
            )

            train_pred = model.predict(X_train).astype(int) + 1
            val_pred = model.predict(X_val).astype(int) + 1

        else:
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

        train_metrics = metrics(y_train, train_pred)
        val_metrics = metrics(y_val, val_pred)

        row = {
            "model": name,
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_balanced_accuracy":
                val_metrics["balanced_accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_wetland_precision":
                val_metrics["wetland_precision"],
            "validation_wetland_recall":
                val_metrics["wetland_recall"],
            "validation_wetland_f1":
                val_metrics["wetland_f1"],
            "train_validation_macro_f1_gap":
                train_metrics["macro_f1"]
                - val_metrics["macro_f1"],
            "false_wetland_from_paddy":
                val_metrics["false_wetland_from_paddy"],
            "wetlands_predicted_as_vegetation":
                val_metrics["wetlands_predicted_as_vegetation"],
            "wetlands_predicted_as_paddy":
                val_metrics["wetlands_predicted_as_paddy"],
            "wetlands_predicted_as_water":
                val_metrics["wetlands_predicted_as_water"],
        }

        comparison_rows.append(row)

        print(
            "Train macro-F1:",
            round(train_metrics["macro_f1"], 4)
        )
        print(
            "Validation macro-F1:",
            round(val_metrics["macro_f1"], 4)
        )
        print(
            "Validation balanced accuracy:",
            round(val_metrics["balanced_accuracy"], 4)
        )
        print(
            "Wetland precision:",
            round(val_metrics["wetland_precision"], 4)
        )
        print(
            "Wetland recall:",
            round(val_metrics["wetland_recall"], 4)
        )
        print(
            "Wetland F1:",
            round(val_metrics["wetland_f1"], 4)
        )
        print(
            "Paddy incorrectly predicted as wetland:",
            val_metrics["false_wetland_from_paddy"]
        )

        print("\nVALIDATION CONFUSION MATRIX")
        matrix = confusion_matrix(
            y_val, val_pred, labels=CLASSES
        )
        print(matrix)

        # Save predictions for reproducible validation.
        for i, prediction in enumerate(val_pred):
            prediction_rows.append({
                "model": name,
                "year": int(val.iloc[i]["year"]),
                "sample_id": str(val.iloc[i]["sample_id"]),
                "actual_class": int(y_val[i]),
                "predicted_class": int(prediction),
                "correct": bool(y_val[i] == prediction),
            })

        # Year-specific wetland evaluation.
        for year in YEARS:
            year_mask = (
                val["year"].to_numpy() == year
            )

            year_metrics = metrics(
                y_val[year_mask],
                val_pred[year_mask]
            )

            yearly_rows.append({
                "model": name,
                "year": year,
                "wetland_precision":
                    year_metrics["wetland_precision"],
                "wetland_recall":
                    year_metrics["wetland_recall"],
                "wetland_f1":
                    year_metrics["wetland_f1"],
                "wetland_support":
                    year_metrics["wetland_support"],
            })

        joblib.dump(
            model,
            MODEL_DIR / f"lulc_candidate_{name.lower()}.joblib"
        )

    # --------------------------------------------------------
    # SAVE COMPARISON EVIDENCE
    # --------------------------------------------------------

    comparison = pd.DataFrame(comparison_rows)

    # Add the existing RF results to the comparison table.
    # RF models are not retrained here.
    if not RF_FILE.is_file():
        raise FileNotFoundError(RF_FILE)

    rf = pd.read_csv(RF_FILE)

    required_rf_columns = {
        "model",
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_f1",
        "validation_wetland_precision",
        "validation_wetland_recall",
        "validation_wetland_f1",
    }

    if not required_rf_columns.issubset(rf.columns):
        raise ValueError(
            "Existing Random Forest comparison has missing columns."
        )

    all_models = pd.concat(
        [rf, comparison],
        ignore_index=True
    )

    all_models = all_models.sort_values(
        ["validation_macro_f1", "validation_wetland_f1"],
        ascending=False,
        kind="stable"
    ).reset_index(drop=True)

    comparison_path = (
        METRIC_DIR / "script04b_all_model_comparison.csv"
    )

    prediction_path = (
        METRIC_DIR / "script04b_validation_predictions.csv"
    )

    yearly_path = (
        METRIC_DIR / "script04b_wetland_year_metrics.csv"
    )

    metadata_path = (
        REPORT_DIR / "script04b_model_comparison_metadata.json"
    )

    all_models.to_csv(comparison_path, index=False)

    pd.DataFrame(prediction_rows).to_csv(
        prediction_path, index=False
    )

    pd.DataFrame(yearly_rows).to_csv(
        yearly_path, index=False
    )

    metadata = {
        "stage": "Historical LULC classifier comparison",
        "timestamp_utc":
            datetime.now(timezone.utc).isoformat(),
        "models_compared": list(models.keys()),
        "existing_reference_models": rf["model"].tolist(),
        "features": FEATURES,
        "training_samples": len(train),
        "validation_samples": len(val),
        "held_out_test_samples": 234,
        "test_evaluated": False,
        "selection_status": "PENDING_RESEARCH_REVIEW",
        "selection_basis": (
            "Macro-F1, wetland-specific precision/recall/F1, "
            "class confusion and model complexity."
        ),
        "limitations": [
            "Comparison uses one buffered spatial split.",
            "150 m separation does not prove spatial independence.",
            "Wetland/Paddy/Vegetation spectral confusion remains "
            "a key research concern.",
            "Four source sample-name/year mismatches remain "
            "documented as temporal-reference uncertainty.",
            "No held-out test observations were used.",
        ],
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("ALL MODEL VALIDATION COMPARISON")
    print("=" * 65)

    display_cols = [
        "model",
        "validation_macro_f1",
        "validation_balanced_accuracy",
        "validation_wetland_precision",
        "validation_wetland_recall",
        "validation_wetland_f1",
    ]

    print(
        all_models[display_cols]
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved model comparison:", comparison_path)
    print("Saved validation predictions:", prediction_path)
    print("Saved yearly wetland metrics:", yearly_path)
    print("Saved metadata:", metadata_path)

    print("\nSCRIPT 04B: COMPLETED")
    print("FINAL MODEL ACCEPTANCE: PENDING REVIEW")
    print("HELD-OUT TEST: NOT EVALUATED")


if __name__ == "__main__":
    main()
