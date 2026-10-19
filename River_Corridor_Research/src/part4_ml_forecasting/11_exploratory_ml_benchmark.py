
"""
J26-DS-338 - River Corridor Research

STEP 16: EXPERIMENTAL ML BENCHMARK

Compare:
1. Persistence baseline
2. Training mean-change baseline
3. Random Forest Regressor

Target:
    Next-year classified water-width proxy change.

IMPORTANT:
- Pilot labels are not independently validated.
- This is NOT a riverbank erosion predictor.
- Strict chronological validation is used.
- Test years do not influence model selection.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data/processed/morphology"
    / "10_next_year_pilot_pairs.csv"
)

REPORT_DIR = ROOT / "outputs/reports"
MODEL_READY_DIR = ROOT / "data/model_ready"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "outputs/figures"

for folder in [
    REPORT_DIR,
    MODEL_READY_DIR,
    MODEL_DIR,
    FIG_DIR
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

FEATURES = [
    "current_width_m",
    "chainage_m",
    "reach_number",
    "base_year"
]

RANDOM_STATE = 42
MAX_TRANSECT_WIDTH_M = 300.0


# ==========================================
# 2. DATA LOADING AND QUALITY CHECKS
# ==========================================

print("=" * 75)
print("STEP 16 - EXPERIMENTAL ML BENCHMARK")
print("=" * 75)

df = pd.read_csv(INPUT_FILE)

required = [
    "transect_id",
    "reach_id",
    "chainage_m",
    "current_width_m",
    "next_width_m",
    "base_year",
    "target_year",
    "pilot_change_m",
    "validated",
    "ml_training_approved"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

if df.duplicated(
    ["transect_id", "target_year"]
).any():
    raise ValueError(
        "Duplicate transect-target year pairs."
    )

if not (
    df["target_year"] == df["base_year"] + 1
).all():
    raise ValueError(
        "Non-consecutive temporal pairs detected."
    )

# The input is explicitly an unvalidated proxy.
# Fail if its provenance has unexpectedly changed.
for flag_column in [
    "validated",
    "ml_training_approved"
]:
    if (
        df[flag_column]
        .astype(str)
        .str.lower()
        .eq("true")
        .any()
    ):
        raise ValueError(
            f"Unexpected TRUE flag: {flag_column}"
        )

df["reach_number"] = pd.to_numeric(
    df["reach_id"]
    .astype(str)
    .str.replace(
        "R", "",
        regex=False
    ),
    errors="raise"
)

df["observed_change_m"] = (
    df["next_width_m"]
    - df["current_width_m"]
)

if not np.allclose(
    df["observed_change_m"],
    df["pilot_change_m"],
    atol=0.11
):
    raise ValueError(
        "Observed change does not match input target."
    )

numeric_columns = FEATURES + [
    "next_width_m",
    "observed_change_m"
]

if not np.isfinite(
    df[numeric_columns].to_numpy(
        dtype=float
    )
).all():
    raise ValueError(
        "Missing or non-finite modeling values."
    )

if (
    (df["current_width_m"] < 60).any()
    or (df["next_width_m"] < 60).any()
    or (df["current_width_m"] > 300).any()
    or (df["next_width_m"] > 300).any()
):
    raise ValueError(
        "Unexpected widths outside screened range."
    )

print(f"Input pilot pairs: {len(df)}")
print(
    f"Unique transects: {df['transect_id'].nunique()}"
)


# ==========================================
# 3. CHRONOLOGICAL SPLIT
# ==========================================

train = df[
    df["target_year"].between(
        2015, 2022
    )
].copy()

validation = df[
    df["target_year"] == 2023
].copy()

test = df[
    df["target_year"].isin(
        [2024, 2025]
    )
].copy()

excluded = df[
    ~df.index.isin(
        train.index.union(
            validation.index
        ).union(test.index)
    )
].copy()

for name, subset in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:
    if subset.empty:
        raise ValueError(
            f"{name} dataset is empty."
        )

if (
    train["target_year"].max()
    >= validation["target_year"].min()
):
    raise ValueError(
        "Train/validation temporal leakage."
    )

if (
    validation["target_year"].max()
    >= test["target_year"].min()
):
    raise ValueError(
        "Validation/test temporal leakage."
    )

print("\nCHRONOLOGICAL SPLIT")
print(f"Train       : {len(train)}")
print(f"Validation  : {len(validation)}")
print(f"Test        : {len(test)}")
print(f"Excluded    : {len(excluded)}")

split_qa = pd.DataFrame([
    {
        "split": name,
        "records": len(subset),
        "first_target_year": (
            int(subset["target_year"].min())
            if not subset.empty else None
        ),
        "last_target_year": (
            int(subset["target_year"].max())
            if not subset.empty else None
        ),
        "purpose": purpose
    }
    for name, subset, purpose in [
        (
            "train", train,
            "FIT_EXPERIMENTAL_MODEL"
        ),
        (
            "validation", validation,
            "MODEL_SELECTION"
        ),
        (
            "test", test,
            "FINAL_HOLDOUT_EVALUATION"
        ),
        (
            "excluded", excluded,
            "EARLY_SENSOR_PERIOD"
        )
    ]
])

split_qa.to_csv(
    REPORT_DIR / "11_ml_split_qa.csv",
    index=False
)


# ==========================================
# 4. TARGET SIGNAL DIAGNOSTICS
# ==========================================

def target_profile(name, subset):

    values = subset[
        "observed_change_m"
    ].to_numpy()

    return {
        "split": name,
        "records": len(values),
        "median_change_m": float(
            np.median(values)
        ),
        "mean_change_m": float(
            np.mean(values)
        ),
        "std_change_m": float(
            np.std(values)
        ),
        "near_zero_change_pct": float(
            100 * np.mean(
                np.isclose(
                    values, 0, atol=0.05
                )
            )
        ),
        "abs_change_ge_30m_pct": float(
            100 * np.mean(
                np.abs(values) >= 30
            )
        )
    }


target_report = pd.DataFrame([
    target_profile("train", train),
    target_profile(
        "validation_2023",
        validation
    ),
    target_profile("test", test)
])

target_report.to_csv(
    REPORT_DIR / "11_ml_target_qa.csv",
    index=False
)

print("\nTARGET SIGNAL QA")
print(target_report.to_string(index=False))


# ==========================================
# 5. TRAIN RANDOM FOREST
# ==========================================

X_train = train[FEATURES]
y_train = train["observed_change_m"]

model = RandomForestRegressor(
    n_estimators=250,
    max_depth=8,
    min_samples_leaf=8,
    max_features=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)

training_mean_change = float(
    y_train.mean()
)

print("\nRandom Forest training completed.")
print(f"Number of trees: {model.n_estimators}")
print(f"Training samples: {len(train)}")


# ==========================================
# 6. PREDICTION FUNCTIONS
# ==========================================

def predict_methods(subset):

    current_width = subset[
        "current_width_m"
    ].to_numpy()

    # Baseline 1: No detected change.
    persistence = current_width.copy()

    # Baseline 2: Training-period mean change.
    mean_baseline = np.clip(
        current_width + training_mean_change,
        0,
        MAX_TRANSECT_WIDTH_M
    )

    # Experimental RF predicts the change,
    # not directly the future river width.
    rf_change = model.predict(
        subset[FEATURES]
    )

    rf_width = np.clip(
        current_width + rf_change,
        0,
        MAX_TRANSECT_WIDTH_M
    )

    return {
        "Persistence": persistence,
        "TrainingMeanChange": mean_baseline,
        "RandomForest": rf_width
    }


# ==========================================
# 7. EVALUATION FUNCTION
# ==========================================

def evaluate(
    split_name,
    subset,
    method_name,
    predictions
):

    observed = subset[
        "next_width_m"
    ].to_numpy()

    observed_change = subset[
        "observed_change_m"
    ].to_numpy()

    errors = observed - predictions

    changed = (
        np.abs(observed_change) >= 30.0
    )

    return {
        "split": split_name,
        "method": method_name,
        "samples": len(subset),
        "mae_m": float(
            mean_absolute_error(
                observed,
                predictions
            )
        ),
        "rmse_m": float(
            np.sqrt(
                mean_squared_error(
                    observed,
                    predictions
                )
            )
        ),
        "r2_width": float(
            r2_score(
                observed,
                predictions
            )
        ),
        "mean_error_m": float(
            np.mean(errors)
        ),
        "changed_cases_ge_30m": int(
            changed.sum()
        ),
        "mae_on_changed_cases_m": (
            float(
                np.mean(
                    np.abs(errors[changed])
                )
            )
            if changed.any()
            else np.nan
        ),
        "label_status": (
            "UNVALIDATED_PILOT_PROXY"
        )
    }


# ==========================================
# 8. VALIDATION AND FINAL TEST
# ==========================================

evaluation_sets = [
    ("validation_2023", validation),
    (
        "test_2024",
        test[test["target_year"] == 2024]
    ),
    (
        "test_2025",
        test[test["target_year"] == 2025]
    ),
    ("test_2024_2025", test)
]

metrics_rows = []

for split_name, subset in evaluation_sets:

    predictions = predict_methods(subset)

    for method, predicted_width in predictions.items():

        metrics_rows.append(
            evaluate(
                split_name,
                subset,
                method,
                predicted_width
            )
        )

metrics = pd.DataFrame(metrics_rows)

# Select a method using 2023 validation only.
validation_metrics = metrics[
    metrics["split"] == "validation_2023"
]

best_row = validation_metrics.sort_values(
    "mae_m",
    kind="stable"
).iloc[0]

selected_method = str(
    best_row["method"]
)

# Do not choose or tune based on final test.
print("\nSELECTED USING 2023 VALIDATION:")
print(selected_method)

metrics_path = (
    REPORT_DIR / "11_ml_benchmark_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)

print("\nBENCHMARK RESULTS")
print(
    metrics[
        [
            "split",
            "method",
            "samples",
            "mae_m",
            "rmse_m",
            "r2_width",
            "mae_on_changed_cases_m"
        ]
    ].to_string(index=False)
)


# ==========================================
# 9. EXPORT VALIDATION / TEST PREDICTIONS
# ==========================================

evaluation_data = pd.concat(
    [validation, test],
    ignore_index=True
).sort_values(
    ["target_year", "transect_id"]
).reset_index(drop=True)

all_predictions = predict_methods(
    evaluation_data
)

prediction_output = evaluation_data[
    [
        "transect_id",
        "reach_id",
        "chainage_m",
        "base_year",
        "target_year",
        "current_width_m",
        "next_width_m",
        "observed_change_m"
    ]
].copy()

for method, predicted in all_predictions.items():

    prediction_output[
        f"{method}_predicted_width_m"
    ] = np.round(predicted, 3)

    prediction_output[
        f"{method}_predicted_change_m"
    ] = np.round(
        predicted
        - evaluation_data[
            "current_width_m"
        ].to_numpy(),
        3
    )

prediction_output["dataset_split"] = np.where(
    prediction_output["target_year"] == 2023,
    "VALIDATION",
    "TEST"
)

prediction_output["label_status"] = (
    "UNVALIDATED_PILOT_PROXY"
)

prediction_output["physical_erosion_prediction"] = False

predictions_path = (
    MODEL_READY_DIR
    / "11_ml_holdout_predictions.csv"
)

prediction_output.to_csv(
    predictions_path,
    index=False
)


# ==========================================
# 10. SAVE EXPERIMENTAL MODEL
# ==========================================

model_path = (
    MODEL_DIR / "11_rf_pilot_change.joblib"
)

joblib.dump({
    "model": model,
    "feature_columns": FEATURES,
    "target": "pilot_change_m",
    "training_target_years": [2015, 2022],
    "validation_target_year": 2023,
    "test_target_years": [2024, 2025],
    "selected_method": selected_method,
    "label_status": "UNVALIDATED_PILOT_PROXY",
    "production_approved": False
}, model_path)

metadata_path = (
    REPORT_DIR / "11_ml_model_metadata.json"
)

metadata = {
    "experiment": "NEXT_YEAR_PILOT_WIDTH_CHANGE",
    "algorithm": "RandomForestRegressor",
    "baseline_methods": [
        "Persistence",
        "TrainingMeanChange"
    ],
    "feature_columns": FEATURES,
    "train_records": len(train),
    "validation_records": len(validation),
    "test_records": len(test),
    "selected_by_validation": selected_method,
    "hyperparameter_tuning": False,
    "independent_ground_truth": False,
    "physical_erosion_model": False,
    "production_approved": False
}

metadata_path.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)


# ==========================================
# 11. CREATE BENCHMARK FIGURE
# ==========================================

plot_data = metrics[
    metrics["split"].isin([
        "validation_2023",
        "test_2024",
        "test_2025"
    ])
]

figure_table = plot_data.pivot(
    index="split",
    columns="method",
    values="mae_m"
)

fig, ax = plt.subplots(
    figsize=(11, 6)
)

figure_table.plot(
    kind="bar",
    ax=ax,
    rot=0
)

ax.set_ylabel(
    "MAE (metres)"
)

ax.set_xlabel(
    "Chronological evaluation period"
)

ax.set_title(
    "Exploratory Pilot-Width Forecast Benchmark"
)

ax.grid(
    axis="y",
    alpha=0.2
)

fig.tight_layout()

fig_path = (
    FIG_DIR / "11_ml_benchmark.png"
)

fig.savefig(
    fig_path,
    dpi=150
)

plt.close(fig)


# ==========================================
# 12. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 75)
print("STEP 16 - FINAL SUMMARY")
print("=" * 75)

print("Experimental model trained: YES")
print("Validation-selected method:", selected_method)
print("Validation year: 2023")
print("Test years: 2024, 2025")

print("\nSaved outputs:")
print(metrics_path)
print(predictions_path)
print(model_path)
print(metadata_path)
print(fig_path)

print("\nIMPORTANT LIMITATIONS:")
print("Labels are classified-water width proxies.")
print("They are not independently verified.")
print("Spatially adjacent transects are correlated.")
print("The evaluation tests future years,")
print("not new independent river locations.")
print("This is not a validated erosion predictor.")
print("Production approval remains FALSE.")

print("\nSTEP 16 FINISHED")
print("=" * 75)
