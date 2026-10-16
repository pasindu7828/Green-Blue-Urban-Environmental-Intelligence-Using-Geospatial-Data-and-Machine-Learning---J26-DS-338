"""
STEP 14C - GENERATE 2026-2028 URBANIZATION PRESSURE SCORES AND GIS MAP

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Load the final Step 14A production XGBoost model.
2. Load the Step 14B leakage-safe 2025 inference feature table.
3. Score all inference-eligible 100 m cells.
4. Apply the locked operating threshold without changing it.
5. Calculate relative pressure rank / percentile.
6. Join scores back to the permanent 100 m grid.
7. Export GIS-ready pressure outputs.

Interpretation:
- Origin year = 2025
- Predictor information = through 2025 only
- Pressure horizon = 2026-2028
- Score = model-derived urbanization-pressure score
- Score is NOT claimed to be a calibrated probability.
- Threshold flag uses the locked development threshold of 0.715.
- Ineligible cells are not assigned a model score.

IMPORTANT:
- No model training occurs here.
- No threshold tuning occurs here.
- No 2026 Landsat predictor data is used.
- No OSM 2026 data is used.
- No final-test metric is recalculated.
"""

from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd


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

ELIGIBILITY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "inference"
    / "14b_2025_inference_eligibility.csv"
)

GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "grid"
    / "kaduwela_100m_modelling_grid.gpkg"
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
    / "inference"
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

PRESSURE_SCORES_CSV = (
    OUTPUT_DIR
    / "14c_2026_2028_urbanization_pressure_scores.csv"
)

PRESSURE_GRID_GPKG = (
    OUTPUT_DIR
    / "14c_2026_2028_urbanization_pressure_grid.gpkg"
)

PRESSURE_GPKG_LAYER = (
    "urbanization_pressure_2026_2028"
)

SCORE_DISTRIBUTION_CSV = (
    REPORT_DIR
    / "14c_pressure_score_distribution.csv"
)

TOP_100_CSV = (
    REPORT_DIR
    / "14c_top_100_pressure_cells.csv"
)

QA_CSV = (
    REPORT_DIR
    / "14c_pressure_scoring_qa.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "14c_pressure_scoring_summary.csv"
)


# =============================================================================
# 3. EXPECTED VALUES
# =============================================================================

EXPECTED_GRID_CELLS = 9240

EXPECTED_INFERENCE_ROWS = 4620

EXPECTED_FEATURES = 55

EXPECTED_ORIGIN_YEAR = 2025

EXPECTED_PRESSURE_PERIOD = "2026-2028"

EXPECTED_MODEL_NAME = "XGBoost"

EXPECTED_MODEL_ROLE = (
    "FINAL_PRODUCTION_URBANIZATION_PRESSURE_MODEL"
)

EXPECTED_THRESHOLD = 0.715


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


def percentile_0_100(values):
    """
    Convert continuous scores to relative percentile ranks.

    Lowest score  -> approximately 0
    Highest score -> 100

    Tied scores receive the same average percentile.
    """

    values = pd.Series(
        values,
        dtype=float,
    )

    n = len(
        values
    )

    if n <= 1:
        return pd.Series(
            np.full(
                n,
                100.0,
            ),
            index=values.index,
        )

    ascending_rank = values.rank(
        method="average",
        ascending=True,
    )

    return (
        (
            ascending_rank
            -
            1.0
        )
        /
        (
            n
            -
            1.0
        )
        *
        100.0
    )


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 14C - GENERATE 2026-2028 URBANIZATION PRESSURE SCORES AND GIS MAP")
print("=" * 100)


for required_file in [
    MODEL_FILE,
    INFERENCE_FEATURES_FILE,
    ELIGIBILITY_FILE,
    GRID_FILE,
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
        "Production model bundle is missing keys:\n"
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
        f"Expected production model {EXPECTED_MODEL_NAME}, "
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


print(
    f"\nProduction model      : "
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
        "Production model feature order does not exactly match "
        "the frozen Step 12E feature list."
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
        "Inference table contains an unexpected origin year."
    )


if not (
    inference_df[
        "pressure_period"
    ].astype(
        str
    )
    ==
    EXPECTED_PRESSURE_PERIOD
).all():

    raise RuntimeError(
        "Unexpected pressure period in inference table."
    )


missing_features = [
    feature_name
    for feature_name in frozen_feature_names
    if feature_name not in inference_df.columns
]


if missing_features:

    raise RuntimeError(
        "Frozen production features missing from inference table:\n"
        +
        "\n".join(
            missing_features
        )
    )


X_inference = (
    inference_df[
        frozen_feature_names
    ]
    .to_numpy(
        dtype=np.float32
    )
)


if np.isnan(
    X_inference
).any():

    raise RuntimeError(
        "Missing values found in inference matrix."
    )


if np.isinf(
    X_inference
).any():

    raise RuntimeError(
        "Infinite values found in inference matrix."
    )


print("\n" + "=" * 100)
print("PRODUCTION INFERENCE INPUT")
print("=" * 100)


print(
    f"Inference origin       : "
    f"{EXPECTED_ORIGIN_YEAR}"
)

print(
    f"Pressure period        : "
    f"{EXPECTED_PRESSURE_PERIOD}"
)

print(
    f"Eligible cells         : "
    f"{len(inference_df):,}"
)

print(
    f"Frozen predictors      : "
    f"{len(frozen_feature_names)}"
)

print(
    f"Feature order matches  : "
    f"{feature_order_exact_match}"
)


# =============================================================================
# 9. SCORE ELIGIBLE CELLS
# =============================================================================

print(
    "\nScoring production inference cells..."
)


pressure_scores = (
    model.predict_proba(
        X_inference
    )[
        :,
        1
    ]
)


pressure_scores = np.asarray(
    pressure_scores,
    dtype=float,
)


if len(
    pressure_scores
) != len(
    inference_df
):

    raise RuntimeError(
        "Prediction count does not match inference row count."
    )


if not np.isfinite(
    pressure_scores
).all():

    raise RuntimeError(
        "Non-finite production scores detected."
    )


if not np.all(
    (
        pressure_scores
        >=
        0.0
    )
    &
    (
        pressure_scores
        <=
        1.0
    )
):

    raise RuntimeError(
        "Production scores outside [0,1] detected."
    )


print(
    "Production scoring complete."
)


# =============================================================================
# 10. CREATE PRESSURE OUTPUT TABLE
# =============================================================================

prediction_df = pd.DataFrame(
    {
        "grid_id": (
            inference_df[
                "grid_id"
            ].to_numpy()
        ),
        "grid_row": (
            inference_df[
                "grid_row"
            ]
            .astype(
                int
            )
            .to_numpy()
        ),
        "grid_col": (
            inference_df[
                "grid_col"
            ]
            .astype(
                int
            )
            .to_numpy()
        ),
        "origin_year": (
            EXPECTED_ORIGIN_YEAR
        ),
        "pressure_period": (
            EXPECTED_PRESSURE_PERIOD
        ),
        "urbanization_pressure_score": (
            pressure_scores
        ),
    }
)


prediction_df[
    "pressure_percentile"
] = (
    percentile_0_100(
        prediction_df[
            "urbanization_pressure_score"
        ]
    )
    .to_numpy(
        dtype=float
    )
)


prediction_df[
    "pressure_rank"
] = (
    prediction_df[
        "urbanization_pressure_score"
    ]
    .rank(
        method="min",
        ascending=False,
    )
    .astype(
        int
    )
)


prediction_df[
    "operating_threshold"
] = (
    locked_threshold
)


prediction_df[
    "threshold_flag"
] = (
    prediction_df[
        "urbanization_pressure_score"
    ]
    >=
    locked_threshold
).astype(
    np.uint8
)


prediction_df[
    "threshold_status"
] = np.where(
    prediction_df[
        "threshold_flag"
    ]
    ==
    1,
    "FLAGGED_BY_LOCKED_THRESHOLD",
    "BELOW_LOCKED_THRESHOLD",
)


prediction_df[
    "score_interpretation"
] = (
    "MODEL_DERIVED_PRESSURE_SCORE_NOT_CALIBRATED_PROBABILITY"
)


prediction_df = prediction_df.sort_values(
    [
        "pressure_rank",
        "grid_id",
    ]
).reset_index(
    drop=True
)


# =============================================================================
# 11. SCORE DISTRIBUTION
# =============================================================================

score_series = (
    prediction_df[
        "urbanization_pressure_score"
    ]
)


flagged_count = int(
    prediction_df[
        "threshold_flag"
    ].sum()
)


flagged_percent = float(
    flagged_count
    /
    len(
        prediction_df
    )
    *
    100
)


score_distribution_df = pd.DataFrame(
    [
        {
            "inference_cells": (
                len(
                    prediction_df
                )
            ),
            "minimum_score": float(
                score_series.min()
            ),
            "p05_score": float(
                score_series.quantile(
                    0.05
                )
            ),
            "p10_score": float(
                score_series.quantile(
                    0.10
                )
            ),
            "p25_score": float(
                score_series.quantile(
                    0.25
                )
            ),
            "median_score": float(
                score_series.median()
            ),
            "mean_score": float(
                score_series.mean()
            ),
            "p75_score": float(
                score_series.quantile(
                    0.75
                )
            ),
            "p90_score": float(
                score_series.quantile(
                    0.90
                )
            ),
            "p95_score": float(
                score_series.quantile(
                    0.95
                )
            ),
            "p99_score": float(
                score_series.quantile(
                    0.99
                )
            ),
            "maximum_score": float(
                score_series.max()
            ),
            "locked_threshold": (
                locked_threshold
            ),
            "flagged_cells": (
                flagged_count
            ),
            "flagged_percent_of_eligible": (
                flagged_percent
            ),
        }
    ]
)


# =============================================================================
# 12. TOP 100 PRESSURE CELLS
# =============================================================================

top_100_df = (
    prediction_df[
        [
            "pressure_rank",
            "grid_id",
            "grid_row",
            "grid_col",
            "origin_year",
            "pressure_period",
            "urbanization_pressure_score",
            "pressure_percentile",
            "operating_threshold",
            "threshold_flag",
            "threshold_status",
        ]
    ]
    .head(
        100
    )
    .copy()
)


# =============================================================================
# 13. LOAD FULL GRID AND ELIGIBILITY
# =============================================================================

grid_gdf = gpd.read_file(
    GRID_FILE
)


if len(
    grid_gdf
) != EXPECTED_GRID_CELLS:

    raise RuntimeError(
        f"Expected {EXPECTED_GRID_CELLS} permanent grid cells, "
        f"found {len(grid_gdf)}."
    )


if grid_gdf[
    "grid_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate permanent grid IDs detected."
    )


eligibility_df = pd.read_csv(
    ELIGIBILITY_FILE
)


if len(
    eligibility_df
) != EXPECTED_GRID_CELLS:

    raise RuntimeError(
        f"Expected {EXPECTED_GRID_CELLS} eligibility rows, "
        f"found {len(eligibility_df)}."
    )


if eligibility_df[
    "grid_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate eligibility grid IDs detected."
    )


final_ready_count = int(
    eligibility_df[
        "final_inference_ready"
    ]
    .astype(
        bool
    )
    .sum()
)


if final_ready_count != EXPECTED_INFERENCE_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_INFERENCE_ROWS:,} final-ready cells, "
        f"found {final_ready_count:,}."
    )


# =============================================================================
# 14. CREATE FULL GIS PRESSURE GRID
# =============================================================================

eligibility_columns = [
    "grid_id",
    "spatial_overlap_eligible",
    "coverage_eligible",
    "current_nonbuilt_eligible",
    "past_stability_eligible",
    "initial_inference_eligible",
    "predictor_complete",
    "final_inference_ready",
]


pressure_join_columns = [
    "grid_id",
    "urbanization_pressure_score",
    "pressure_percentile",
    "pressure_rank",
    "operating_threshold",
    "threshold_flag",
    "threshold_status",
]


pressure_grid_gdf = (
    grid_gdf.merge(
        eligibility_df[
            eligibility_columns
        ],
        on="grid_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        prediction_df[
            pressure_join_columns
        ],
        on="grid_id",
        how="left",
        validate="one_to_one",
    )
)


pressure_grid_gdf[
    "origin_year"
] = (
    EXPECTED_ORIGIN_YEAR
)


pressure_grid_gdf[
    "pressure_period"
] = (
    EXPECTED_PRESSURE_PERIOD
)


pressure_grid_gdf[
    "map_status"
] = np.where(
    pressure_grid_gdf[
        "final_inference_ready"
    ]
    .astype(
        bool
    ),
    np.where(
        pressure_grid_gdf[
            "threshold_flag"
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
        ==
        1,
        "ELIGIBLE_FLAGGED",
        "ELIGIBLE_BELOW_THRESHOLD",
    ),
    "NOT_IN_INFERENCE_DOMAIN",
)


pressure_grid_gdf[
    "score_is_calibrated_probability"
] = (
    False
)


# =============================================================================
# 15. GIS QA
# =============================================================================

gis_score_count = int(
    pressure_grid_gdf[
        "urbanization_pressure_score"
    ]
    .notna()
    .sum()
)


gis_flagged_count = int(
    (
        pressure_grid_gdf[
            "threshold_flag"
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
        ==
        1
    )
    .sum()
)


gis_score_count_matches = (
    gis_score_count
    ==
    EXPECTED_INFERENCE_ROWS
)


gis_flag_count_matches = (
    gis_flagged_count
    ==
    flagged_count
)


all_scored_cells_are_ready = bool(
    pressure_grid_gdf.loc[
        pressure_grid_gdf[
            "urbanization_pressure_score"
        ].notna(),
        "final_inference_ready",
    ]
    .astype(
        bool
    )
    .all()
)


ineligible_cells_have_no_scores = bool(
    pressure_grid_gdf.loc[
        ~pressure_grid_gdf[
            "final_inference_ready"
        ]
        .astype(
            bool
        ),
        "urbanization_pressure_score",
    ]
    .isna()
    .all()
)


# =============================================================================
# 16. SAVE GIS OUTPUT
# =============================================================================

if PRESSURE_GRID_GPKG.exists():

    PRESSURE_GRID_GPKG.unlink()


pressure_grid_gdf.to_file(
    PRESSURE_GRID_GPKG,
    layer=PRESSURE_GPKG_LAYER,
    driver="GPKG",
)


# =============================================================================
# 17. FINAL QA
# =============================================================================

score_count_valid = (
    len(
        prediction_df
    )
    ==
    EXPECTED_INFERENCE_ROWS
)


grid_ids_unique = bool(
    not prediction_df[
        "grid_id"
    ]
    .duplicated()
    .any()
)


score_bounds_valid = bool(
    prediction_df[
        "urbanization_pressure_score"
    ]
    .between(
        0.0,
        1.0,
        inclusive="both",
    )
    .all()
)


percentile_bounds_valid = bool(
    prediction_df[
        "pressure_percentile"
    ]
    .between(
        0.0,
        100.0,
        inclusive="both",
    )
    .all()
)


threshold_flag_valid = bool(
    set(
        prediction_df[
            "threshold_flag"
        ]
        .unique()
        .tolist()
    ).issubset(
        {
            0,
            1,
        }
    )
)


threshold_application_valid = bool(
    (
        (
            prediction_df[
                "urbanization_pressure_score"
            ]
            >=
            locked_threshold
        )
        .astype(
            np.uint8
        )
        ==
        prediction_df[
            "threshold_flag"
        ]
    )
    .all()
)


threshold_unchanged = bool(
    np.isclose(
        locked_threshold,
        EXPECTED_THRESHOLD,
        atol=0.0,
        rtol=0.0,
    )
)


model_retrained = False

threshold_retuned = False

provisional_2026_used = False

osm_2026_used = False

calibrated_probability_claimed = False


overall_status = (
    "PASS"
    if all(
        [
            score_count_valid,
            grid_ids_unique,
            score_bounds_valid,
            percentile_bounds_valid,
            threshold_flag_valid,
            threshold_application_valid,
            threshold_unchanged,
            feature_order_exact_match,
            gis_score_count_matches,
            gis_flag_count_matches,
            all_scored_cells_are_ready,
            ineligible_cells_have_no_scores,
            PRESSURE_GRID_GPKG.exists(),
            not model_retrained,
            not threshold_retuned,
            not provisional_2026_used,
            not osm_2026_used,
            not calibrated_probability_claimed,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 18. QA REPORT
# =============================================================================

qa_df = pd.DataFrame(
    [
        {
            "permanent_grid_cells": (
                EXPECTED_GRID_CELLS
            ),
            "inference_ready_cells": (
                len(
                    prediction_df
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
            "score_count_valid": (
                score_count_valid
            ),
            "grid_ids_unique": (
                grid_ids_unique
            ),
            "score_bounds_valid": (
                score_bounds_valid
            ),
            "percentile_bounds_valid": (
                percentile_bounds_valid
            ),
            "threshold_flag_valid": (
                threshold_flag_valid
            ),
            "threshold_application_valid": (
                threshold_application_valid
            ),
            "locked_threshold_unchanged": (
                threshold_unchanged
            ),
            "flagged_cells": (
                flagged_count
            ),
            "gis_scored_cell_count": (
                gis_score_count
            ),
            "gis_score_count_matches": (
                gis_score_count_matches
            ),
            "gis_flag_count_matches": (
                gis_flag_count_matches
            ),
            "all_scored_cells_are_inference_ready": (
                all_scored_cells_are_ready
            ),
            "ineligible_cells_have_no_score": (
                ineligible_cells_have_no_scores
            ),
            "model_retrained": (
                model_retrained
            ),
            "threshold_retuned": (
                threshold_retuned
            ),
            "provisional_2026_predictor_used": (
                provisional_2026_used
            ),
            "osm_2026_used": (
                osm_2026_used
            ),
            "score_claimed_as_calibrated_probability": (
                calibrated_probability_claimed
            ),
            "overall_step14c_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 19. FINAL SUMMARY
# =============================================================================

final_summary_df = pd.DataFrame(
    [
        {
            "stage": (
                "Production urbanization-pressure inference"
            ),
            "origin_year": (
                EXPECTED_ORIGIN_YEAR
            ),
            "pressure_period": (
                EXPECTED_PRESSURE_PERIOD
            ),
            "production_model": (
                model_name
            ),
            "candidate_id": (
                candidate_id
            ),
            "eligible_scored_cells": (
                len(
                    prediction_df
                )
            ),
            "locked_operating_threshold": (
                locked_threshold
            ),
            "flagged_cells": (
                flagged_count
            ),
            "flagged_percent_of_eligible": (
                flagged_percent
            ),
            "minimum_pressure_score": float(
                score_series.min()
            ),
            "median_pressure_score": float(
                score_series.median()
            ),
            "mean_pressure_score": float(
                score_series.mean()
            ),
            "maximum_pressure_score": float(
                score_series.max()
            ),
            "raw_score_is_calibrated_probability": (
                False
            ),
            "official_model_evaluation_source": (
                "Step 13E"
            ),
            "new_accuracy_calculated": (
                False
            ),
            "overall_step14c_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 20. SAVE TABLE OUTPUTS
# =============================================================================

prediction_df.to_csv(
    PRESSURE_SCORES_CSV,
    index=False,
)


score_distribution_df.to_csv(
    SCORE_DISTRIBUTION_CSV,
    index=False,
)


top_100_df.to_csv(
    TOP_100_CSV,
    index=False,
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
# 21. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("2026-2028 URBANIZATION PRESSURE RESULT")
print("=" * 100)


print(
    f"Scored eligible cells          : "
    f"{len(prediction_df):,}"
)

print(
    f"Minimum pressure score         : "
    f"{score_series.min():.6f}"
)

print(
    f"Median pressure score          : "
    f"{score_series.median():.6f}"
)

print(
    f"Mean pressure score            : "
    f"{score_series.mean():.6f}"
)

print(
    f"90th percentile score          : "
    f"{score_series.quantile(0.90):.6f}"
)

print(
    f"95th percentile score          : "
    f"{score_series.quantile(0.95):.6f}"
)

print(
    f"Maximum pressure score         : "
    f"{score_series.max():.6f}"
)

print(
    f"Locked threshold               : "
    f"{locked_threshold:.3f}"
)

print(
    f"Cells flagged by threshold     : "
    f"{flagged_count:,}"
)

print(
    f"Flagged % of eligible cells    : "
    f"{flagged_percent:.3f}%"
)


print("\n" + "=" * 100)
print("TOP 20 PRESSURE CELLS")
print("=" * 100)


print(
    top_100_df.head(
        20
    ).to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("GIS MAP STATUS")
print("=" * 100)


print(
    pressure_grid_gdf[
        "map_status"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


print("\n" + "=" * 100)
print("STEP 14C QA")
print("=" * 100)


print(
    f"Scored row count valid              : "
    f"{score_count_valid}"
)

print(
    f"Unique scored grid IDs              : "
    f"{grid_ids_unique}"
)

print(
    f"Frozen feature order exact match    : "
    f"{feature_order_exact_match}"
)

print(
    f"Score bounds [0,1] valid            : "
    f"{score_bounds_valid}"
)

print(
    f"Percentile bounds [0,100] valid     : "
    f"{percentile_bounds_valid}"
)

print(
    f"Threshold flag values valid         : "
    f"{threshold_flag_valid}"
)

print(
    f"Locked threshold applied correctly  : "
    f"{threshold_application_valid}"
)

print(
    f"Locked threshold unchanged          : "
    f"{threshold_unchanged}"
)

print(
    f"GIS scored cell count matches       : "
    f"{gis_score_count_matches}"
)

print(
    f"GIS flagged cell count matches      : "
    f"{gis_flag_count_matches}"
)

print(
    f"All scored cells inference-ready    : "
    f"{all_scored_cells_are_ready}"
)

print(
    f"Ineligible cells have no score      : "
    f"{ineligible_cells_have_no_scores}"
)

print(
    f"Model retrained                     : "
    f"{model_retrained}"
)

print(
    f"Threshold retuned                   : "
    f"{threshold_retuned}"
)

print(
    f"2026 predictor data used            : "
    f"{provisional_2026_used}"
)

print(
    f"OSM 2026 used                       : "
    f"{osm_2026_used}"
)

print(
    f"Score claimed as calibrated prob.   : "
    f"{calibrated_probability_claimed}"
)

print(
    f"Overall Step 14C status             : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(PRESSURE_SCORES_CSV)}"
)

print(
    f" - {relative_path(PRESSURE_GRID_GPKG)}"
)

print(
    f"   Layer: {PRESSURE_GPKG_LAYER}"
)

print(
    f" - {relative_path(SCORE_DISTRIBUTION_CSV)}"
)

print(
    f" - {relative_path(TOP_100_CSV)}"
)

print(
    f" - {relative_path(QA_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. These are 2026-2028 urbanization-pressure scores from a 2025 origin."
)

print(
    "2. The score is a model-derived pressure score, not a guaranteed conversion probability."
)

print(
    "3. The operating threshold remains the locked 0.715 threshold selected before final testing."
)

print(
    "4. Cells outside the inference domain are kept in the GIS layer but receive no model score."
)

print(
    "5. Pressure percentile is only a relative ranking among eligible Kaduwela cells."
)

print(
    "6. No Low/Medium/High/Critical classes were invented in this step."
)

print(
    "7. No model retraining, threshold tuning or new accuracy calculation occurred."
)

print(
    "8. After this passes, we will inspect the real score distribution before defining any dashboard pressure categories."
)

print("=" * 100)