"""
STEP 15A - PRODUCTION XGBOOST SHAP EXPLANATIONS

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Explain the frozen Step 14A production XGBoost model.
2. Use the exact Step 14B 2025-origin inference features.
3. Preserve the Step 14C/14D 2026-2028 pressure results.
4. Generate global SHAP feature importance.
5. Generate local grid-cell explanations.
6. Generate focused explanations for threshold-flagged cells.
7. Generate presentation-ready SHAP plots.

Interpretation:
- Origin year = 2025.
- Pressure period = 2026-2028.
- SHAP explains the production XGBoost RAW MARGIN.
- Positive SHAP value -> pushes model output toward higher pressure.
- Negative SHAP value -> pushes model output toward lower pressure.
- SHAP values are NOT percentage-point probability contributions.
- Existing urbanization-pressure scores are NOT recalculated or changed.

IMPORTANT:
- No model retraining.
- No hyperparameter tuning.
- No threshold tuning.
- No feature selection.
- No use of target labels.
- No use of provisional 2026 predictors.
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from scipy.special import expit


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "urbanization_pressure"
    / "production"
    / "14a_xgboost_final_production.joblib"
)

INFERENCE_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "inference"
    / "14b_2025_inference_features.csv"
)

CATEGORIZED_SCORES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "inference"
    / "14d_2026_2028_pressure_categories.csv"
)

FEATURE_LIST_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "12e_final_predictor_feature_list.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "xai"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "urbanization_pressure"
    / "xai"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. OUTPUTS
# =============================================================================

SHAP_VALUES_CSV = (
    OUTPUT_DIR
    / "15a_2025_inference_shap_values.csv"
)

LOCAL_TOP_EXPLANATIONS_CSV = (
    OUTPUT_DIR
    / "15a_local_top_shap_explanations.csv"
)

FLAGGED_LOCAL_EXPLANATIONS_CSV = (
    OUTPUT_DIR
    / "15a_flagged_cell_top_shap_explanations.csv"
)

GLOBAL_IMPORTANCE_CSV = (
    REPORT_DIR
    / "15a_global_shap_importance.csv"
)

FLAGGED_GLOBAL_IMPORTANCE_CSV = (
    REPORT_DIR
    / "15a_flagged_global_shap_importance.csv"
)

QA_CSV = (
    REPORT_DIR
    / "15a_shap_qa.csv"
)

SUMMARY_CSV = (
    REPORT_DIR
    / "15a_shap_summary.csv"
)

GLOBAL_BAR_PNG = (
    FIGURE_DIR
    / "15a_global_shap_importance_top20.png"
)

GLOBAL_BEESWARM_PNG = (
    FIGURE_DIR
    / "15a_global_shap_beeswarm_top20.png"
)

TOP_CELL_WATERFALL_PNG = (
    FIGURE_DIR
    / "15a_top_pressure_cell_shap_waterfall.png"
)


# =============================================================================
# 3. EXPECTED VALUES
# =============================================================================

EXPECTED_INFERENCE_ROWS = 4620

EXPECTED_FEATURES = 55

EXPECTED_FLAGGED_CELLS = 255

EXPECTED_THRESHOLD = 0.715

EXPECTED_ORIGIN_YEAR = 2025

EXPECTED_PRESSURE_PERIOD = "2026-2028"

EXPECTED_MODEL_NAME = "XGBoost"

EXPECTED_MODEL_ROLE = (
    "FINAL_PRODUCTION_URBANIZATION_PRESSURE_MODEL"
)

TOP_LOCAL_FEATURES = 8

GLOBAL_PLOT_FEATURES = 20

ADDITIVITY_TOLERANCE = 1e-4

SCORE_REPRODUCTION_TOLERANCE = 1e-6


# =============================================================================
# 4. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """Return project-relative path where possible."""

    try:

        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:

        return str(path)


def direction_from_shap(
    shap_value,
):
    """Human-readable SHAP direction."""

    if shap_value > 0:

        return "INCREASES_PRESSURE"

    if shap_value < 0:

        return "DECREASES_PRESSURE"

    return "NEUTRAL"


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 15A - PRODUCTION XGBOOST SHAP EXPLANATIONS")
print("=" * 100)

print(
    f"\nSHAP version : "
    f"{shap.__version__}"
)


for required_file in [
    MODEL_FILE,
    INFERENCE_FEATURES_FILE,
    CATEGORIZED_SCORES_FILE,
    FEATURE_LIST_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 6. LOAD PRODUCTION MODEL
# =============================================================================

production_bundle = joblib.load(
    MODEL_FILE
)


required_bundle_keys = {
    "model",
    "model_name",
    "candidate_id",
    "feature_names",
    "operating_threshold",
    "model_role",
}


missing_bundle_keys = (
    required_bundle_keys
    -
    set(
        production_bundle.keys()
    )
)


if missing_bundle_keys:

    raise RuntimeError(
        "Production model bundle is missing required keys:\n"
        +
        "\n".join(
            sorted(
                missing_bundle_keys
            )
        )
    )


model = (
    production_bundle[
        "model"
    ]
)


model_name = str(
    production_bundle[
        "model_name"
    ]
)


candidate_id = str(
    production_bundle[
        "candidate_id"
    ]
)


model_feature_names = list(
    production_bundle[
        "feature_names"
    ]
)


locked_threshold = float(
    production_bundle[
        "operating_threshold"
    ]
)


model_role = str(
    production_bundle[
        "model_role"
    ]
)


if model_name != EXPECTED_MODEL_NAME:

    raise RuntimeError(
        f"Expected {EXPECTED_MODEL_NAME}, "
        f"found {model_name}."
    )


if model_role != EXPECTED_MODEL_ROLE:

    raise RuntimeError(
        f"Unexpected production model role:\n"
        f"{model_role}"
    )


if not np.isclose(
    locked_threshold,
    EXPECTED_THRESHOLD,
    atol=0.0,
    rtol=0.0,
):

    raise RuntimeError(
        f"Expected locked threshold "
        f"{EXPECTED_THRESHOLD}, "
        f"found {locked_threshold}."
    )


print("\nPRODUCTION MODEL")

print(
    f"Model                 : "
    f"{model_name}"
)

print(
    f"Candidate             : "
    f"{candidate_id}"
)

print(
    f"Locked threshold      : "
    f"{locked_threshold:.3f}"
)

print(
    f"Model role            : "
    f"{model_role}"
)


# =============================================================================
# 7. LOAD FROZEN FEATURE LIST
# =============================================================================

feature_list_df = pd.read_csv(
    FEATURE_LIST_FILE
)


if len(
    feature_list_df
) != EXPECTED_FEATURES:

    raise RuntimeError(
        f"Expected {EXPECTED_FEATURES} frozen features, "
        f"found {len(feature_list_df)}."
    )


frozen_feature_names = (
    feature_list_df[
        "feature_name"
    ]
    .astype(
        str
    )
    .tolist()
)


feature_order_exact_match = (
    frozen_feature_names
    ==
    model_feature_names
)


if not feature_order_exact_match:

    raise RuntimeError(
        "Frozen Step 12E feature order does not match "
        "the production model."
    )


# =============================================================================
# 8. LOAD 2025 INFERENCE FEATURES
# =============================================================================

inference_df = pd.read_csv(
    INFERENCE_FEATURES_FILE
)


if len(
    inference_df
) != EXPECTED_INFERENCE_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_INFERENCE_ROWS:,} inference rows, "
        f"found {len(inference_df):,}."
    )


if inference_df[
    "grid_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate inference grid IDs detected."
    )


if not (
    inference_df[
        "origin_year"
    ]
    ==
    EXPECTED_ORIGIN_YEAR
).all():

    raise RuntimeError(
        "Unexpected inference origin year."
    )


if not (
    inference_df[
        "pressure_period"
    ]
    .astype(
        str
    )
    ==
    EXPECTED_PRESSURE_PERIOD
).all():

    raise RuntimeError(
        "Unexpected pressure period."
    )


missing_features = [
    feature_name
    for feature_name in frozen_feature_names
    if feature_name not in inference_df.columns
]


if missing_features:

    raise RuntimeError(
        "Frozen features missing from inference table:\n"
        +
        "\n".join(
            missing_features
        )
    )


X_df = (
    inference_df[
        frozen_feature_names
    ]
    .astype(
        np.float32
    )
    .copy()
)


if X_df.isna().any().any():

    raise RuntimeError(
        "Missing values detected in SHAP inference matrix."
    )


if np.isinf(
    X_df.to_numpy(
        dtype=float
    )
).any():

    raise RuntimeError(
        "Infinite values detected in SHAP inference matrix."
    )


# =============================================================================
# 9. LOAD FROZEN PRESSURE RESULTS / CATEGORIES
# =============================================================================

pressure_df = pd.read_csv(
    CATEGORIZED_SCORES_FILE
)


if len(
    pressure_df
) != EXPECTED_INFERENCE_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_INFERENCE_ROWS:,} categorized scores, "
        f"found {len(pressure_df):,}."
    )


if pressure_df[
    "grid_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate categorized pressure grid IDs detected."
    )


required_pressure_columns = [
    "grid_id",
    "urbanization_pressure_score",
    "pressure_percentile",
    "pressure_rank",
    "operating_threshold",
    "threshold_flag",
    "pressure_category",
]


missing_pressure_columns = [
    column
    for column in required_pressure_columns
    if column not in pressure_df.columns
]


if missing_pressure_columns:

    raise RuntimeError(
        "Required Step 14D pressure columns are missing:\n"
        +
        "\n".join(
            missing_pressure_columns
        )
    )


metadata_df = (
    inference_df[
        [
            "grid_id",
            "grid_row",
            "grid_col",
            "origin_year",
            "pressure_period",
        ]
    ]
    .merge(
        pressure_df[
            required_pressure_columns
        ],
        on="grid_id",
        how="left",
        validate="one_to_one",
    )
)


if metadata_df[
    "urbanization_pressure_score"
].isna().any():

    raise RuntimeError(
        "Some inference cells have no frozen pressure score."
    )


flagged_count = int(
    metadata_df[
        "threshold_flag"
    ]
    .astype(
        int
    )
    .sum()
)


if flagged_count != EXPECTED_FLAGGED_CELLS:

    raise RuntimeError(
        f"Expected {EXPECTED_FLAGGED_CELLS} flagged cells, "
        f"found {flagged_count}."
    )


print("\nEXPLANATION INPUT")

print(
    f"Inference cells       : "
    f"{len(inference_df):,}"
)

print(
    f"Frozen predictors     : "
    f"{len(frozen_feature_names)}"
)

print(
    f"Threshold-flagged     : "
    f"{flagged_count:,}"
)

print(
    f"Origin                : "
    f"{EXPECTED_ORIGIN_YEAR}"
)

print(
    f"Pressure period       : "
    f"{EXPECTED_PRESSURE_PERIOD}"
)


# =============================================================================
# 10. VERIFY PRODUCTION SCORES BEFORE SHAP
# =============================================================================

reproduced_scores = (
    model.predict_proba(
        X_df
    )[
        :,
        1
    ]
)


saved_scores = (
    metadata_df[
        "urbanization_pressure_score"
    ]
    .to_numpy(
        dtype=float
    )
)


score_max_difference = float(
    np.max(
        np.abs(
            reproduced_scores
            -
            saved_scores
        )
    )
)


score_reproduction_valid = (
    score_max_difference
    <=
    SCORE_REPRODUCTION_TOLERANCE
)


if not score_reproduction_valid:

    raise RuntimeError(
        "Production scores could not be reproduced exactly enough. "
        f"Maximum difference = {score_max_difference:.12f}"
    )


print(
    f"\nMaximum saved-vs-reproduced score difference : "
    f"{score_max_difference:.12f}"
)


# =============================================================================
# 11. CALCULATE RAW MODEL MARGINS
# =============================================================================

raw_margin = (
    model.predict(
        X_df,
        output_margin=True,
    )
)


raw_margin = np.asarray(
    raw_margin,
    dtype=float,
).reshape(
    -1
)


score_from_margin = expit(
    raw_margin
)


margin_score_max_difference = float(
    np.max(
        np.abs(
            score_from_margin
            -
            reproduced_scores
        )
    )
)


margin_score_consistency_valid = (
    margin_score_max_difference
    <=
    SCORE_REPRODUCTION_TOLERANCE
)


# =============================================================================
# 12. CREATE TREE SHAP EXPLAINER
# =============================================================================

print("\n" + "=" * 100)
print("CALCULATING TREE SHAP VALUES")
print("=" * 100)

print(
    "Explaining raw XGBoost model margin..."
)


explainer = shap.TreeExplainer(
    model,
    model_output="raw",
)


explanation = explainer(
    X_df
)


shap_values = np.asarray(
    explanation.values,
    dtype=float,
)


base_values = np.asarray(
    explanation.base_values,
    dtype=float,
)


# =============================================================================
# 13. NORMALIZE SHAP OUTPUT SHAPE
# =============================================================================

if shap_values.ndim == 3:

    if shap_values.shape[
        2
    ] == 1:

        shap_values = shap_values[
            :,
            :,
            0
        ]

    else:

        raise RuntimeError(
            "Unexpected multi-output SHAP shape: "
            f"{shap_values.shape}"
        )


if shap_values.ndim != 2:

    raise RuntimeError(
        f"Expected 2D SHAP values, "
        f"found shape {shap_values.shape}."
    )


if shap_values.shape != (
    EXPECTED_INFERENCE_ROWS,
    EXPECTED_FEATURES,
):

    raise RuntimeError(
        "Unexpected SHAP matrix shape:\n"
        f"{shap_values.shape}"
    )


if base_values.ndim == 0:

    base_values = np.full(
        EXPECTED_INFERENCE_ROWS,
        float(
            base_values
        ),
        dtype=float,
    )


elif base_values.ndim == 1:

    if len(
        base_values
    ) == 1:

        base_values = np.full(
            EXPECTED_INFERENCE_ROWS,
            float(
                base_values[
                    0
                ]
            ),
            dtype=float,
        )

    elif len(
        base_values
    ) != EXPECTED_INFERENCE_ROWS:

        raise RuntimeError(
            "Unexpected SHAP base-value length."
        )


else:

    base_values = (
        base_values.reshape(
            EXPECTED_INFERENCE_ROWS,
            -1,
        )
    )


    if base_values.shape[
        1
    ] != 1:

        raise RuntimeError(
            "Unexpected multidimensional SHAP base values."
        )


    base_values = base_values[
        :,
        0
    ]


print(
    f"SHAP matrix shape     : "
    f"{shap_values.shape}"
)


# =============================================================================
# 14. SHAP ADDITIVITY QA
# =============================================================================

reconstructed_margin = (
    base_values
    +
    shap_values.sum(
        axis=1
    )
)


additivity_error = np.abs(
    reconstructed_margin
    -
    raw_margin
)


maximum_additivity_error = float(
    np.max(
        additivity_error
    )
)


mean_additivity_error = float(
    np.mean(
        additivity_error
    )
)


additivity_valid = (
    maximum_additivity_error
    <=
    ADDITIVITY_TOLERANCE
)


if not additivity_valid:

    raise RuntimeError(
        "SHAP additivity check failed. "
        f"Maximum error = {maximum_additivity_error:.12f}"
    )


print(
    f"Maximum SHAP additivity error : "
    f"{maximum_additivity_error:.12f}"
)

print(
    f"Mean SHAP additivity error    : "
    f"{mean_additivity_error:.12f}"
)


# =============================================================================
# 15. GLOBAL SHAP IMPORTANCE
# =============================================================================

mean_abs_shap = np.mean(
    np.abs(
        shap_values
    ),
    axis=0,
)


mean_signed_shap = np.mean(
    shap_values,
    axis=0,
)


positive_mean_shap = np.mean(
    np.where(
        shap_values > 0,
        shap_values,
        0.0,
    ),
    axis=0,
)


negative_mean_shap = np.mean(
    np.where(
        shap_values < 0,
        shap_values,
        0.0,
    ),
    axis=0,
)


total_mean_abs_shap = float(
    np.sum(
        mean_abs_shap
    )
)


if total_mean_abs_shap <= 0:

    raise RuntimeError(
        "Global SHAP importance sum is zero."
    )


global_importance_df = pd.DataFrame(
    {
        "feature_name": (
            frozen_feature_names
        ),
        "mean_absolute_shap_raw_margin": (
            mean_abs_shap
        ),
        "mean_signed_shap_raw_margin": (
            mean_signed_shap
        ),
        "mean_positive_shap_raw_margin": (
            positive_mean_shap
        ),
        "mean_negative_shap_raw_margin": (
            negative_mean_shap
        ),
    }
)


global_importance_df[
    "importance_percent"
] = (
    global_importance_df[
        "mean_absolute_shap_raw_margin"
    ]
    /
    total_mean_abs_shap
    *
    100.0
)


global_importance_df = (
    global_importance_df.sort_values(
        "mean_absolute_shap_raw_margin",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


global_importance_df[
    "global_importance_rank"
] = (
    np.arange(
        1,
        len(
            global_importance_df
        )
        +
        1,
    )
)


global_importance_df = global_importance_df[
    [
        "global_importance_rank",
        "feature_name",
        "mean_absolute_shap_raw_margin",
        "importance_percent",
        "mean_signed_shap_raw_margin",
        "mean_positive_shap_raw_margin",
        "mean_negative_shap_raw_margin",
    ]
]


# =============================================================================
# 16. FLAGGED-CELL GLOBAL SHAP IMPORTANCE
# =============================================================================

flagged_mask = (
    metadata_df[
        "threshold_flag"
    ]
    .astype(
        int
    )
    .to_numpy()
    ==
    1
)


flagged_shap_values = (
    shap_values[
        flagged_mask
    ]
)


if flagged_shap_values.shape[
    0
] != EXPECTED_FLAGGED_CELLS:

    raise RuntimeError(
        "Unexpected number of flagged SHAP rows."
    )


flagged_mean_abs_shap = np.mean(
    np.abs(
        flagged_shap_values
    ),
    axis=0,
)


flagged_mean_signed_shap = np.mean(
    flagged_shap_values,
    axis=0,
)


flagged_total_mean_abs = float(
    np.sum(
        flagged_mean_abs_shap
    )
)


flagged_global_importance_df = pd.DataFrame(
    {
        "feature_name": (
            frozen_feature_names
        ),
        "mean_absolute_shap_raw_margin": (
            flagged_mean_abs_shap
        ),
        "mean_signed_shap_raw_margin": (
            flagged_mean_signed_shap
        ),
    }
)


flagged_global_importance_df[
    "importance_percent"
] = (
    flagged_global_importance_df[
        "mean_absolute_shap_raw_margin"
    ]
    /
    flagged_total_mean_abs
    *
    100.0
)


flagged_global_importance_df = (
    flagged_global_importance_df.sort_values(
        "mean_absolute_shap_raw_margin",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


flagged_global_importance_df[
    "flagged_importance_rank"
] = (
    np.arange(
        1,
        len(
            flagged_global_importance_df
        )
        +
        1,
    )
)


flagged_global_importance_df = (
    flagged_global_importance_df[
        [
            "flagged_importance_rank",
            "feature_name",
            "mean_absolute_shap_raw_margin",
            "importance_percent",
            "mean_signed_shap_raw_margin",
        ]
    ]
)


# =============================================================================
# 17. SAVE FULL SHAP MATRIX
# =============================================================================

shap_column_names = [
    f"shap__{feature_name}"
    for feature_name in frozen_feature_names
]


shap_matrix_df = pd.DataFrame(
    shap_values,
    columns=shap_column_names,
)


shap_output_df = pd.concat(
    [
        metadata_df.reset_index(
            drop=True
        ),
        pd.DataFrame(
            {
                "raw_model_margin": (
                    raw_margin
                ),
                "shap_base_value_raw_margin": (
                    base_values
                ),
                "shap_reconstructed_raw_margin": (
                    reconstructed_margin
                ),
                "shap_additivity_abs_error": (
                    additivity_error
                ),
            }
        ),
        shap_matrix_df,
    ],
    axis=1,
)


# =============================================================================
# 18. LOCAL TOP EXPLANATIONS FOR EVERY CELL
# =============================================================================

print("\nCreating local top-feature explanations...")


local_records = []


feature_values_array = (
    X_df.to_numpy(
        dtype=float
    )
)


for row_index in range(
    len(
        inference_df
    )
):

    row_shap = (
        shap_values[
            row_index
        ]
    )


    absolute_order = np.argsort(
        np.abs(
            row_shap
        )
    )[
        ::-1
    ]


    top_indices = absolute_order[
        :TOP_LOCAL_FEATURES
    ]


    row_meta = (
        metadata_df.iloc[
            row_index
        ]
    )


    for local_rank, feature_index in enumerate(
        top_indices,
        start=1,
    ):

        shap_value = float(
            row_shap[
                feature_index
            ]
        )


        local_records.append(
            {
                "grid_id": (
                    row_meta[
                        "grid_id"
                    ]
                ),
                "origin_year": int(
                    row_meta[
                        "origin_year"
                    ]
                ),
                "pressure_period": (
                    row_meta[
                        "pressure_period"
                    ]
                ),
                "urbanization_pressure_score": float(
                    row_meta[
                        "urbanization_pressure_score"
                    ]
                ),
                "pressure_percentile": float(
                    row_meta[
                        "pressure_percentile"
                    ]
                ),
                "pressure_rank": int(
                    row_meta[
                        "pressure_rank"
                    ]
                ),
                "pressure_category": (
                    row_meta[
                        "pressure_category"
                    ]
                ),
                "threshold_flag": int(
                    row_meta[
                        "threshold_flag"
                    ]
                ),
                "local_importance_rank": (
                    local_rank
                ),
                "feature_name": (
                    frozen_feature_names[
                        feature_index
                    ]
                ),
                "feature_value": float(
                    feature_values_array[
                        row_index,
                        feature_index,
                    ]
                ),
                "shap_value_raw_margin": (
                    shap_value
                ),
                "absolute_shap_value": float(
                    abs(
                        shap_value
                    )
                ),
                "direction": (
                    direction_from_shap(
                        shap_value
                    )
                ),
            }
        )


local_explanations_df = pd.DataFrame(
    local_records
)


flagged_local_explanations_df = (
    local_explanations_df[
        local_explanations_df[
            "threshold_flag"
        ]
        ==
        1
    ]
    .copy()
)


# =============================================================================
# 19. CREATE PRESENTATION FIGURE - GLOBAL BAR
# =============================================================================

print(
    "Generating global SHAP bar plot..."
)


plt.figure(
    figsize=(
        10,
        8,
    )
)


shap.summary_plot(
    shap_values,
    X_df,
    feature_names=frozen_feature_names,
    plot_type="bar",
    max_display=GLOBAL_PLOT_FEATURES,
    show=False,
)


plt.title(
    "Global Urbanization Pressure Drivers\n"
    "2025 Origin → 2026–2028 Pressure",
    fontsize=12,
)


plt.tight_layout()


plt.savefig(
    GLOBAL_BAR_PNG,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =============================================================================
# 20. CREATE PRESENTATION FIGURE - BEESWARM
# =============================================================================

print(
    "Generating global SHAP beeswarm plot..."
)


plt.figure(
    figsize=(
        11,
        9,
    )
)


shap.summary_plot(
    shap_values,
    X_df,
    feature_names=frozen_feature_names,
    max_display=GLOBAL_PLOT_FEATURES,
    show=False,
)


plt.title(
    "SHAP Distribution of Urbanization Pressure Drivers\n"
    "2025 Origin → 2026–2028 Pressure",
    fontsize=12,
)


plt.tight_layout()


plt.savefig(
    GLOBAL_BEESWARM_PNG,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =============================================================================
# 21. CREATE TOP-PRESSURE CELL WATERFALL
# =============================================================================

top_pressure_index = int(
    metadata_df[
        "pressure_rank"
    ]
    .astype(
        int
    )
    .idxmin()
)


top_grid_id = str(
    metadata_df.loc[
        top_pressure_index,
        "grid_id",
    ]
)


top_pressure_score = float(
    metadata_df.loc[
        top_pressure_index,
        "urbanization_pressure_score",
    ]
)


print(
    f"Generating SHAP waterfall for top pressure cell: "
    f"{top_grid_id}"
)


single_explanation = shap.Explanation(
    values=(
        shap_values[
            top_pressure_index
        ]
    ),
    base_values=float(
        base_values[
            top_pressure_index
        ]
    ),
    data=(
        feature_values_array[
            top_pressure_index
        ]
    ),
    feature_names=(
        frozen_feature_names
    ),
)


shap.plots.waterfall(
    single_explanation,
    max_display=12,
    show=False,
)


plt.title(
    f"Why Grid {top_grid_id} Has High Urbanization Pressure\n"
    f"Pressure Score = {top_pressure_score:.3f}",
    fontsize=11,
)


plt.tight_layout()


plt.savefig(
    TOP_CELL_WATERFALL_PNG,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =============================================================================
# 22. QA
# =============================================================================

shap_matrix_finite = bool(
    np.isfinite(
        shap_values
    ).all()
)


base_values_finite = bool(
    np.isfinite(
        base_values
    ).all()
)


shap_shape_valid = (
    shap_values.shape
    ==
    (
        EXPECTED_INFERENCE_ROWS,
        EXPECTED_FEATURES,
    )
)


local_rows_expected = (
    EXPECTED_INFERENCE_ROWS
    *
    TOP_LOCAL_FEATURES
)


local_row_count_valid = (
    len(
        local_explanations_df
    )
    ==
    local_rows_expected
)


flagged_local_rows_expected = (
    EXPECTED_FLAGGED_CELLS
    *
    TOP_LOCAL_FEATURES
)


flagged_local_row_count_valid = (
    len(
        flagged_local_explanations_df
    )
    ==
    flagged_local_rows_expected
)


global_importance_sum = float(
    global_importance_df[
        "importance_percent"
    ].sum()
)


global_importance_sum_valid = bool(
    np.isclose(
        global_importance_sum,
        100.0,
        atol=1e-8,
    )
)


flagged_importance_sum = float(
    flagged_global_importance_df[
        "importance_percent"
    ].sum()
)


flagged_importance_sum_valid = bool(
    np.isclose(
        flagged_importance_sum,
        100.0,
        atol=1e-8,
    )
)


figures_exist = bool(
    GLOBAL_BAR_PNG.exists()
    and
    GLOBAL_BEESWARM_PNG.exists()
    and
    TOP_CELL_WATERFALL_PNG.exists()
)


model_retrained = False

threshold_changed = False

pressure_scores_changed = False

feature_selection_changed = False

target_labels_used = False

provisional_2026_predictors_used = False

shap_values_claimed_as_probability_change = False


overall_status = (
    "PASS"
    if all(
        [
            feature_order_exact_match,
            score_reproduction_valid,
            margin_score_consistency_valid,
            shap_shape_valid,
            shap_matrix_finite,
            base_values_finite,
            additivity_valid,
            local_row_count_valid,
            flagged_local_row_count_valid,
            global_importance_sum_valid,
            flagged_importance_sum_valid,
            figures_exist,
            not model_retrained,
            not threshold_changed,
            not pressure_scores_changed,
            not feature_selection_changed,
            not target_labels_used,
            not provisional_2026_predictors_used,
            not shap_values_claimed_as_probability_change,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 23. QA REPORT
# =============================================================================

qa_df = pd.DataFrame(
    [
        {
            "shap_version": (
                shap.__version__
            ),
            "inference_rows": (
                len(
                    inference_df
                )
            ),
            "frozen_feature_count": (
                len(
                    frozen_feature_names
                )
            ),
            "feature_order_exact_match": (
                feature_order_exact_match
            ),
            "saved_pressure_score_reproduction_valid": (
                score_reproduction_valid
            ),
            "max_saved_score_difference": (
                score_max_difference
            ),
            "raw_margin_probability_consistency_valid": (
                margin_score_consistency_valid
            ),
            "max_margin_score_difference": (
                margin_score_max_difference
            ),
            "shap_matrix_shape_valid": (
                shap_shape_valid
            ),
            "shap_values_finite": (
                shap_matrix_finite
            ),
            "base_values_finite": (
                base_values_finite
            ),
            "shap_additivity_valid": (
                additivity_valid
            ),
            "maximum_shap_additivity_error": (
                maximum_additivity_error
            ),
            "mean_shap_additivity_error": (
                mean_additivity_error
            ),
            "local_explanation_row_count_valid": (
                local_row_count_valid
            ),
            "flagged_local_explanation_row_count_valid": (
                flagged_local_row_count_valid
            ),
            "global_importance_sum_valid": (
                global_importance_sum_valid
            ),
            "flagged_importance_sum_valid": (
                flagged_importance_sum_valid
            ),
            "presentation_figures_created": (
                figures_exist
            ),
            "model_retrained": (
                model_retrained
            ),
            "threshold_changed": (
                threshold_changed
            ),
            "pressure_scores_changed": (
                pressure_scores_changed
            ),
            "feature_selection_changed": (
                feature_selection_changed
            ),
            "target_labels_used": (
                target_labels_used
            ),
            "provisional_2026_predictors_used": (
                provisional_2026_predictors_used
            ),
            "shap_claimed_as_probability_change": (
                shap_values_claimed_as_probability_change
            ),
            "overall_step15a_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 24. SUMMARY REPORT
# =============================================================================

top_10_features = (
    global_importance_df.head(
        10
    )[
        "feature_name"
    ]
    .tolist()
)


top_10_flagged_features = (
    flagged_global_importance_df.head(
        10
    )[
        "feature_name"
    ]
    .tolist()
)


summary_df = pd.DataFrame(
    [
        {
            "stage": (
                "Production XGBoost SHAP explanation"
            ),
            "origin_year": (
                EXPECTED_ORIGIN_YEAR
            ),
            "pressure_period": (
                EXPECTED_PRESSURE_PERIOD
            ),
            "explained_cells": (
                EXPECTED_INFERENCE_ROWS
            ),
            "threshold_flagged_cells": (
                EXPECTED_FLAGGED_CELLS
            ),
            "frozen_predictors": (
                EXPECTED_FEATURES
            ),
            "shap_model_output": (
                "RAW_XGBOOST_MARGIN"
            ),
            "shap_probability_interpretation": (
                False
            ),
            "top_pressure_grid_id": (
                top_grid_id
            ),
            "top_pressure_score": (
                top_pressure_score
            ),
            "top_10_global_features": (
                ";".join(
                    top_10_features
                )
            ),
            "top_10_flagged_features": (
                ";".join(
                    top_10_flagged_features
                )
            ),
            "model_or_threshold_changed": (
                False
            ),
            "overall_step15a_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 25. SAVE OUTPUTS
# =============================================================================

shap_output_df.to_csv(
    SHAP_VALUES_CSV,
    index=False,
)


local_explanations_df.to_csv(
    LOCAL_TOP_EXPLANATIONS_CSV,
    index=False,
)


flagged_local_explanations_df.to_csv(
    FLAGGED_LOCAL_EXPLANATIONS_CSV,
    index=False,
)


global_importance_df.to_csv(
    GLOBAL_IMPORTANCE_CSV,
    index=False,
)


flagged_global_importance_df.to_csv(
    FLAGGED_GLOBAL_IMPORTANCE_CSV,
    index=False,
)


qa_df.to_csv(
    QA_CSV,
    index=False,
)


summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 26. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("GLOBAL SHAP IMPORTANCE - TOP 20")
print("=" * 100)


print(
    global_importance_df.head(
        20
    ).to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("THRESHOLD-FLAGGED CELLS - TOP 20 SHAP FEATURES")
print("=" * 100)


print(
    flagged_global_importance_df.head(
        20
    ).to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("TOP PRESSURE CELL LOCAL EXPLANATION")
print("=" * 100)


print(
    local_explanations_df[
        local_explanations_df[
            "grid_id"
        ]
        ==
        top_grid_id
    ]
    .sort_values(
        "local_importance_rank"
    )
    .to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 15A QA")
print("=" * 100)


print(
    f"SHAP version                         : "
    f"{shap.__version__}"
)

print(
    f"Feature order exact match            : "
    f"{feature_order_exact_match}"
)

print(
    f"Saved pressure scores reproduced     : "
    f"{score_reproduction_valid}"
)

print(
    f"Max saved-score difference           : "
    f"{score_max_difference:.12f}"
)

print(
    f"Raw margin -> score consistency      : "
    f"{margin_score_consistency_valid}"
)

print(
    f"SHAP matrix shape valid              : "
    f"{shap_shape_valid}"
)

print(
    f"SHAP values finite                   : "
    f"{shap_matrix_finite}"
)

print(
    f"SHAP additivity valid                : "
    f"{additivity_valid}"
)

print(
    f"Maximum additivity error             : "
    f"{maximum_additivity_error:.12f}"
)

print(
    f"Local explanation rows valid         : "
    f"{local_row_count_valid}"
)

print(
    f"Flagged explanation rows valid       : "
    f"{flagged_local_row_count_valid}"
)

print(
    f"Global importance sums to 100%       : "
    f"{global_importance_sum_valid}"
)

print(
    f"Flagged importance sums to 100%      : "
    f"{flagged_importance_sum_valid}"
)

print(
    f"Presentation figures created         : "
    f"{figures_exist}"
)

print(
    f"Model retrained                      : "
    f"{model_retrained}"
)

print(
    f"Threshold changed                    : "
    f"{threshold_changed}"
)

print(
    f"Pressure scores changed              : "
    f"{pressure_scores_changed}"
)

print(
    f"Target labels used                   : "
    f"{target_labels_used}"
)

print(
    f"2026 predictors used                 : "
    f"{provisional_2026_predictors_used}"
)

print(
    f"SHAP treated as probability change   : "
    f"{shap_values_claimed_as_probability_change}"
)

print(
    f"Overall Step 15A status              : "
    f"{overall_status}"
)


print("\nGenerated outputs:")


for output_path in [
    SHAP_VALUES_CSV,
    LOCAL_TOP_EXPLANATIONS_CSV,
    FLAGGED_LOCAL_EXPLANATIONS_CSV,
    GLOBAL_IMPORTANCE_CSV,
    FLAGGED_GLOBAL_IMPORTANCE_CSV,
    QA_CSV,
    SUMMARY_CSV,
    GLOBAL_BAR_PNG,
    GLOBAL_BEESWARM_PNG,
    TOP_CELL_WATERFALL_PNG,
]:

    print(
        f" - {relative_path(output_path)}"
    )


print("\nIMPORTANT:")

print(
    "1. SHAP explains the frozen production XGBoost model; no model was retrained."
)

print(
    "2. Positive SHAP values push the model toward higher urbanization pressure."
)

print(
    "3. Negative SHAP values push the model toward lower urbanization pressure."
)

print(
    "4. SHAP values are raw-margin contributions, not probability-percentage contributions."
)

print(
    "5. The original 2026-2028 pressure scores and the locked 0.715 threshold were not changed."
)

print(
    "6. Global SHAP shows which predictors matter across all eligible Kaduwela cells."
)

print(
    "7. Local SHAP shows why an individual 100 m grid cell received its pressure score."
)

print(
    "8. After this passes, Step 15B will convert technical feature names into understandable explanation domains for the dashboard and panel."
)

print("=" * 100)