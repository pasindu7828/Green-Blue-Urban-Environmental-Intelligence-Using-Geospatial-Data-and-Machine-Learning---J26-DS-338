"""
PRIORITY VALIDATION 05
HISTORICAL RETROSPECTIVE PRIORITY BACKTEST
===========================================

PURPOSE
-------
Test whether the finalized priority methodology would have produced useful
priority rankings when applied retrospectively using only information
available before an evaluation year.

Backtests:
    History <= 2023 -> construct 2024 priority -> compare with actual 2024
    History <= 2024 -> construct 2025 priority -> compare with actual 2025

IMPORTANT
---------
This is a one-year retrospective analogue of the final 2026-2030 priority
framework.

Six decision dimensions are retained:

C1 Future Heat Severity
    -> next-year forecast LST

C2 Trend Urgency
    -> relative heat-percentile Theil-Sen slope using history only

C3 Vegetation / Green Deficit
    -> 1 - next-year forecast green_mask

C4 Built-up / Urban Pressure
    -> next-year forecast NDBI

C5 Blue-space / Water Deficit
    -> distance to water

C6 Modeled Intervention Benefit
    -> next-year modeled cooling benefit from:
       +10 percentage-point green cover
       + data-derived MODERATE NDBI reduction

No actual target-year LST or target-year environmental predictors are used
to construct the ranking.

This is retrospective / hindcast validation, not causal validation and not
a ground-truth priority accuracy test.
"""

from pathlib import Path
import json
import math
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from scipy.stats import (
    theilslopes,
    spearmanr,
    kendalltau,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from lightgbm import LGBMRegressor
from libpysal.weights import Queen


# ============================================================
# SETTINGS
# ============================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
AREA_COL = "area_name"
TARGET = "LST_C"

EXPECTED_GRIDS = 2194
EXPECTED_FEATURES = 93

BACKTESTS = [
    {
        "cutoff_year": 2023,
        "target_year": 2024,
    },
    {
        "cutoff_year": 2024,
        "target_year": 2025,
    },
]

TEMPORAL_SOURCES = [
    "NDVI",
    "NDBI",
    "NDWI",
    "EVI",
    "Albedo",
    "NDBI_ADJ",
    "NDVI_ADJ",
    "NightLights",
    "green_mask",
]

STATIC_PREDICTORS = [
    "building_mask",
    "road_mask",
]

STRUCTURAL_PREDICTORS = [
    "dist_road",
    "dist_main_road",
    "dist_water",
    "LCZ",
]

TOL = 1e-12
RANDOM_STATE = 42

GREEN_SCENARIO_PP = 10

TOP10_FRAC = 0.10
TOP25_FRAC = 0.25


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[4]

BASE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "kaduwela_preprocessed_base.csv"
)

ENGINEERED_PATH = (
    ROOT
    / "data"
    / "processed"
    / "kaduwela_engineered_features.csv"
)

GRID_UTM_PATH = (
    ROOT
    / "data"
    / "processed"
    / "kaduwela_grid_geometry_utm44n.geojson"
)

TRAIN_PATH = (
    ROOT
    / "data"
    / "model_ready"
    / "tree"
    / "tree_train_2018_2023_ready.csv"
)

VALIDATION_2024_PATH = (
    ROOT
    / "data"
    / "model_ready"
    / "tree"
    / "tree_validation_2024_ready.csv"
)

STEP19C_MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "19c_step20_candidate_manifest.json"
)

STEP27_FREEZE_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "27_forecast_strategy_freeze_for_step28.json"
)

FINAL_CRITIC_PATH = (
    ROOT
    / "outputs"
    / "priority"
    / "31e_critic_weights.csv"
)


VALIDATION_DIR = (
    ROOT
    / "experiments"
    / "validation"
    / "priority_validation"
)

OUTPUT_DIR = VALIDATION_DIR / "outputs"
FIGURE_DIR = VALIDATION_DIR / "figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


METRICS_PATH = (
    OUTPUT_DIR
    / "05_historical_priority_backtest_metrics.csv"
)

GRID_RESULTS_PATH = (
    OUTPUT_DIR
    / "05_historical_priority_grid_results.csv"
)

WEIGHTS_PATH = (
    OUTPUT_DIR
    / "05_historical_priority_critic_weights.csv"
)

QUINTILE_PATH = (
    OUTPUT_DIR
    / "05_historical_priority_quintile_summary.csv"
)

DECISION_MATRIX_PATH = (
    OUTPUT_DIR
    / "05_historical_priority_decision_matrices.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "05_historical_priority_backtest_summary.csv"
)

FIGURE_PATH = (
    FIGURE_DIR
    / "05_historical_priority_actual_heat_by_quintile.png"
)


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print(
        "\n"
        + "=" * 128
    )

    print(title)

    print(
        "=" * 128
    )


def require_file(path, label):

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def require_columns(
    df,
    columns,
    label,
):

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{label} missing columns: "
            f"{missing}"
        )


def clean_params(params):

    clean = {}

    for key, value in params.items():

        if isinstance(
            value,
            np.integer,
        ):
            value = int(value)

        elif isinstance(
            value,
            np.floating,
        ):
            value = float(value)

        elif isinstance(
            value,
            np.bool_,
        ):
            value = bool(value)

        clean[str(key)] = value

    return clean


def build_lightgbm(
    hyperparameters,
):

    params = clean_params(
        hyperparameters
    )

    return LGBMRegressor(
        **params,
        objective="regression",
        subsample_freq=1,
        importance_type="gain",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


# ============================================================
# GEOMETRY / FEATURE ENGINEERING
# ============================================================

def build_queen_neighbors(
    grid_utm,
):

    ordered = (
        grid_utm[
            [
                GRID_COL,
                "geometry",
            ]
        ]
        .copy()
        .sort_values(
            GRID_COL
        )
        .reset_index(
            drop=True
        )
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        weights = Queen.from_dataframe(
            ordered,
            ids=ordered[
                GRID_COL
            ].tolist(),
            silence_warnings=True,
        )

    neighbors = {
        str(grid_id):
            [
                str(n)
                for n in nbrs
            ]

        for grid_id, nbrs
        in weights.neighbors.items()
    }

    isolates = [
        grid_id
        for grid_id, nbrs
        in neighbors.items()
        if len(nbrs) == 0
    ]

    if isolates:

        raise RuntimeError(
            "Queen neighbour structure "
            "contains isolated grids."
        )

    return neighbors


def add_neighbor_features(
    df,
    neighbors,
):

    out = df.copy()

    for source in TEMPORAL_SOURCES:

        mean_col = (
            f"{source}_nbr_mean"
        )

        contrast_col = (
            f"{source}_nbr_contrast"
        )

        result = np.empty(
            len(out),
            dtype=float,
        )

        for year, index in (
            out.groupby(
                YEAR_COL
            ).groups.items()
        ):

            year_rows = out.loc[
                index,
                [
                    GRID_COL,
                    source,
                ],
            ]

            lookup = dict(
                zip(
                    year_rows[
                        GRID_COL
                    ].astype(str),

                    year_rows[
                        source
                    ].astype(float),
                )
            )

            means = []

            for grid_id in (
                year_rows[
                    GRID_COL
                ].astype(str)
            ):

                nbr_values = [
                    lookup[nbr]
                    for nbr
                    in neighbors[
                        str(grid_id)
                    ]
                ]

                means.append(
                    float(
                        np.mean(
                            nbr_values
                        )
                    )
                )

            result[
                np.asarray(index)
            ] = np.asarray(
                means,
                dtype=float,
            )

        out[
            mean_col
        ] = result

        out[
            contrast_col
        ] = (
            out[source]
            - out[mean_col]
        )

    return out


def hist3_slope(values):

    arr = np.asarray(
        values,
        dtype=float,
    )

    if (
        len(arr) != 3
        or np.isnan(
            arr
        ).any()
    ):

        return np.nan

    return float(
        (
            arr[2]
            - arr[0]
        )
        / 2.0
    )


def infer_year_index_relation(
    engineered,
    cutoff_year,
):

    pairs = (
        engineered[
            engineered[
                YEAR_COL
            ]
            <= cutoff_year
        ][
            [
                YEAR_COL,
                "year_index",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            YEAR_COL
        )
    )

    if len(
        pairs
    ) < 2:

        raise RuntimeError(
            "Too few year_index pairs."
        )

    x = pairs[
        YEAR_COL
    ].to_numpy(
        dtype=float
    )

    y = pairs[
        "year_index"
    ].to_numpy(
        dtype=float
    )

    slope, intercept = (
        np.polyfit(
            x,
            y,
            1,
        )
    )

    reconstructed = (
        intercept
        + slope * x
    )

    max_error = float(
        np.max(
            np.abs(
                reconstructed
                - y
            )
        )
    )

    if max_error > 1e-10:

        raise RuntimeError(
            "year_index linear "
            "reproduction failed."
        )

    return (
        float(slope),
        float(intercept),
    )


def add_engineered_features(
    combined_base,
    neighbors,
    year_index_slope,
    year_index_intercept,
):

    out = (
        combined_base
        .copy()
        .sort_values(
            [
                GRID_COL,
                YEAR_COL,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    out[
        "year_index"
    ] = (
        year_index_intercept
        + year_index_slope
        * out[
            YEAR_COL
        ].astype(float)
    )

    out[
        "vegetation_deficit"
    ] = (
        1.0
        - out[
            "green_mask"
        ]
    )

    out[
        "ndbi_minus_ndvi"
    ] = (
        out["NDBI"]
        - out["NDVI"]
    )

    out[
        "ndbi_x_vegetation_deficit"
    ] = (
        out["NDBI"]
        * out[
            "vegetation_deficit"
        ]
    )

    out[
        "ndvi_x_green_mask"
    ] = (
        out["NDVI"]
        * out[
            "green_mask"
        ]
    )

    out[
        "ndbi_x_building_mask"
    ] = (
        out["NDBI"]
        * out[
            "building_mask"
        ]
    )

    out[
        "nightlights_x_building_mask"
    ] = (
        out["NightLights"]
        * out[
            "building_mask"
        ]
    )

    for source in [
        "dist_road",
        "dist_main_road",
        "dist_water",
    ]:

        out[
            f"log1p_{source}"
        ] = np.log1p(
            out[
                source
            ].astype(float)
        )

    grouped = out.groupby(
        GRID_COL,
        sort=False,
    )

    for source in TEMPORAL_SOURCES:

        out[
            f"{source}_lag1"
        ] = (
            grouped[
                source
            ]
            .shift(1)
        )

        out[
            f"{source}_lag2"
        ] = (
            grouped[
                source
            ]
            .shift(2)
        )

        shifted = (
            grouped[
                source
            ]
            .shift(1)
        )

        out[
            f"{source}_hist3_mean"
        ] = (
            shifted
            .groupby(
                out[
                    GRID_COL
                ],
                sort=False,
            )
            .rolling(
                window=3,
                min_periods=1,
            )
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
            .sort_index()
        )

        out[
            f"{source}_hist3_std"
        ] = (
            shifted
            .groupby(
                out[
                    GRID_COL
                ],
                sort=False,
            )
            .rolling(
                window=3,
                min_periods=2,
            )
            .std(
                ddof=1
            )
            .reset_index(
                level=0,
                drop=True,
            )
            .sort_index()
        )

        out[
            f"{source}_hist3_slope"
        ] = (
            shifted
            .groupby(
                out[
                    GRID_COL
                ],
                sort=False,
            )
            .rolling(
                window=3,
                min_periods=3,
            )
            .apply(
                hist3_slope,
                raw=True,
            )
            .reset_index(
                level=0,
                drop=True,
            )
            .sort_index()
        )

        out[
            f"{source}_delta1"
        ] = (
            out[
                source
            ]
            - out[
                f"{source}_lag1"
            ]
        )

    out = add_neighbor_features(
        out,
        neighbors,
    )

    return out


# ============================================================
# FORECASTING
# ============================================================

def theil_sen_forecast_matrix(
    history_values,
    history_years,
    target_year,
):

    x = np.asarray(
        history_years,
        dtype=float,
    )

    y = np.asarray(
        history_values,
        dtype=float,
    )

    pair_slopes = []

    for i in range(
        len(x) - 1
    ):

        for j in range(
            i + 1,
            len(x),
        ):

            pair_slopes.append(
                (
                    y[:, j]
                    - y[:, i]
                )
                /
                (
                    x[j]
                    - x[i]
                )
            )

    slopes = np.median(
        np.stack(
            pair_slopes,
            axis=1,
        ),
        axis=1,
    )

    intercepts = (
        np.median(
            y,
            axis=1,
        )
        - slopes
        * np.median(x)
    )

    return (
        intercepts
        + slopes
        * float(
            target_year
        )
    )


def forecast_dynamic_predictor(
    wide,
    method,
    history_years,
    target_year,
):

    history = (
        wide[
            history_years
        ]
        .to_numpy(
            dtype=float
        )
    )

    if method == "Rolling-3 Mean":

        prediction = np.mean(
            history[
                :,
                -3:
            ],
            axis=1,
        )

    elif (
        method
        == "Bounded Theil-Sen Trend"
    ):

        prediction = (
            theil_sen_forecast_matrix(
                history_values=
                    history,

                history_years=
                    history_years,

                target_year=
                    target_year,
            )
        )

        prediction = np.clip(
            prediction,
            float(
                np.min(
                    history
                )
            ),
            float(
                np.max(
                    history
                )
            ),
        )

    elif method == "Persistence":

        prediction = (
            history[
                :,
                -1
            ].copy()
        )

    else:

        raise ValueError(
            f"Unsupported predictor "
            f"method: {method}"
        )

    return prediction


def build_next_year_base(
    history,
    target_year,
    selected_methods,
):

    history = (
        history
        .sort_values(
            [
                GRID_COL,
                YEAR_COL,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    history_years = sorted(
        history[
            YEAR_COL
        ]
        .unique()
        .tolist()
    )

    grid_order = sorted(
        history[
            GRID_COL
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    latest_year = max(
        history_years
    )

    latest = (
        history[
            history[
                YEAR_COL
            ]
            == latest_year
        ]
        .copy()
        .set_index(
            GRID_COL
        )
        .loc[
            grid_order
        ]
    )

    frame = pd.DataFrame(
        {
            GRID_COL:
                grid_order,

            YEAR_COL:
                target_year,
        }
    ).set_index(
        GRID_COL
    )

    for predictor, method in (
        selected_methods.items()
    ):

        wide = (
            history
            .pivot(
                index=GRID_COL,
                columns=YEAR_COL,
                values=predictor,
            )
            .loc[
                grid_order,
                history_years,
            ]
        )

        frame[
            predictor
        ] = (
            forecast_dynamic_predictor(
                wide=wide,
                method=method,
                history_years=
                    history_years,
                target_year=
                    target_year,
            )
        )

    for predictor in (
        STATIC_PREDICTORS
    ):

        frame[
            predictor
        ] = (
            latest[
                predictor
            ].astype(float)
        )

    for predictor in (
        STRUCTURAL_PREDICTORS
    ):

        frame[
            predictor
        ] = (
            latest[
                predictor
            ].astype(float)
        )

    frame[
        AREA_COL
    ] = (
        latest[
            AREA_COL
        ].astype(str)
    )

    frame = (
        frame
        .reset_index()
    )

    frame[
        TARGET
    ] = np.nan

    return frame


# ============================================================
# TREND
# ============================================================

def historical_relative_sen_slope(
    history,
):

    d = history[
        [
            GRID_COL,
            YEAR_COL,
            TARGET,
        ]
    ].copy()

    d[
        "relative_heat_percentile"
    ] = (
        d.groupby(
            YEAR_COL
        )[
            TARGET
        ]
        .rank(
            method="average",
            pct=True,
        )
    )

    rows = []

    for grid_id, group in (
        d.groupby(
            GRID_COL,
            sort=True,
        )
    ):

        g = group.sort_values(
            YEAR_COL
        )

        x = g[
            YEAR_COL
        ].to_numpy(
            dtype=float
        )

        y = g[
            "relative_heat_percentile"
        ].to_numpy(
            dtype=float
        )

        sen = theilslopes(
            y,
            x,
            alpha=0.95,
        )

        rows.append(
            {
                GRID_COL:
                    str(grid_id),

                "relative_heat_sen_slope":
                    float(
                        sen.slope
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SCENARIO
# ============================================================

def derive_moderate_ndbi_reduction(
    history,
):

    d = (
        history[
            [
                GRID_COL,
                YEAR_COL,
                "NDBI",
            ]
        ]
        .sort_values(
            [
                GRID_COL,
                YEAR_COL,
            ]
        )
        .copy()
    )

    d[
        "previous_year"
    ] = (
        d.groupby(
            GRID_COL
        )[
            YEAR_COL
        ]
        .shift(1)
    )

    d[
        "d_ndbi"
    ] = (
        d.groupby(
            GRID_COL
        )[
            "NDBI"
        ]
        .diff()
    )

    transitions = (
        d[
            d[
                "previous_year"
            ].notna()
        ]
        .copy()
    )

    year_gap = (
        transitions[
            YEAR_COL
        ]
        - transitions[
            "previous_year"
        ]
    )

    if not (
        year_gap == 1
    ).all():

        raise RuntimeError(
            "NDBI history contains "
            "non-adjacent transitions."
        )

    magnitude = float(
        transitions[
            "d_ndbi"
        ]
        .abs()
        .quantile(
            0.75
        )
    )

    if magnitude <= 0:

        raise RuntimeError(
            "Invalid MODERATE NDBI "
            "reduction."
        )

    return magnitude


def apply_combined_scenario(
    X_base,
    history,
    moderate_ndbi_reduction,
):

    X = X_base.copy()

    # -----------------------------
    # Green +10 percentage points
    # -----------------------------

    baseline_green = (
        X_base[
            "green_mask"
        ]
        .to_numpy(
            dtype=float
        )
    )

    scenario_green = np.clip(
        baseline_green
        + (
            GREEN_SCENARIO_PP
            / 100.0
        ),
        0.0,
        1.0,
    )

    X[
        "green_mask"
    ] = scenario_green


    # -----------------------------
    # NDBI MODERATE reduction
    # -----------------------------

    historical_ndbi_min = float(
        history[
            "NDBI"
        ].min()
    )

    historical_ndbi_max = float(
        history[
            "NDBI"
        ].max()
    )

    baseline_ndbi = (
        X_base[
            "NDBI"
        ]
        .to_numpy(
            dtype=float
        )
    )

    scenario_ndbi = np.clip(
        baseline_ndbi
        - moderate_ndbi_reduction,
        historical_ndbi_min,
        historical_ndbi_max,
    )

    X[
        "NDBI"
    ] = scenario_ndbi


    # -----------------------------
    # Green temporal dependency
    # First future year:
    # lag/history values remain past-only.
    # delta1 changes.
    # -----------------------------

    if (
        "green_mask_delta1"
        in X.columns
    ):

        if (
            "green_mask_lag1"
            not in X.columns
        ):

            raise RuntimeError(
                "green_mask_lag1 "
                "required."
            )

        X[
            "green_mask_delta1"
        ] = (
            scenario_green
            - X_base[
                "green_mask_lag1"
            ].to_numpy(
                dtype=float
            )
        )


    # -----------------------------
    # NDBI temporal dependency
    # -----------------------------

    if (
        "NDBI_delta1"
        in X.columns
    ):

        if (
            "NDBI_lag1"
            not in X.columns
        ):

            raise RuntimeError(
                "NDBI_lag1 required."
            )

        X[
            "NDBI_delta1"
        ] = (
            scenario_ndbi
            - X_base[
                "NDBI_lag1"
            ].to_numpy(
                dtype=float
            )
        )


    # -----------------------------
    # Green current-year features
    # -----------------------------

    scenario_green_deficit = (
        1.0
        - scenario_green
    )

    if (
        "vegetation_deficit"
        in X.columns
    ):

        X[
            "vegetation_deficit"
        ] = (
            scenario_green_deficit
        )

    if (
        "ndvi_x_green_mask"
        in X.columns
    ):

        X[
            "ndvi_x_green_mask"
        ] = (
            X_base[
                "NDVI"
            ].to_numpy(
                dtype=float
            )
            * scenario_green
        )

    if (
        "green_mask_nbr_contrast"
        in X.columns
    ):

        if (
            "green_mask_nbr_mean"
            in X_base.columns
        ):

            nbr_mean = (
                X_base[
                    "green_mask_nbr_mean"
                ]
                .to_numpy(
                    dtype=float
                )
            )

        else:

            nbr_mean = (
                baseline_green
                - X_base[
                    "green_mask_nbr_contrast"
                ]
                .to_numpy(
                    dtype=float
                )
            )

        X[
            "green_mask_nbr_contrast"
        ] = (
            scenario_green
            - nbr_mean
        )


    # -----------------------------
    # NDBI current-year features
    # -----------------------------

    if (
        "ndbi_minus_ndvi"
        in X.columns
    ):

        X[
            "ndbi_minus_ndvi"
        ] = (
            scenario_ndbi
            - X_base[
                "NDVI"
            ].to_numpy(
                dtype=float
            )
        )

    if (
        "ndbi_x_building_mask"
        in X.columns
    ):

        X[
            "ndbi_x_building_mask"
        ] = (
            scenario_ndbi
            * X_base[
                "building_mask"
            ].to_numpy(
                dtype=float
            )
        )

    if (
        "NDBI_nbr_contrast"
        in X.columns
    ):

        if (
            "NDBI_nbr_mean"
            in X_base.columns
        ):

            nbr_mean = (
                X_base[
                    "NDBI_nbr_mean"
                ]
                .to_numpy(
                    dtype=float
                )
            )

        else:

            nbr_mean = (
                baseline_ndbi
                - X_base[
                    "NDBI_nbr_contrast"
                ]
                .to_numpy(
                    dtype=float
                )
            )

        X[
            "NDBI_nbr_contrast"
        ] = (
            scenario_ndbi
            - nbr_mean
        )


    # -----------------------------
    # True joint green x NDBI feature
    # -----------------------------

    if (
        "ndbi_x_vegetation_deficit"
        in X.columns
    ):

        X[
            "ndbi_x_vegetation_deficit"
        ] = (
            scenario_ndbi
            * scenario_green_deficit
        )

    return X


# ============================================================
# CRITIC + TOPSIS
# ============================================================

def minmax_priority_matrix(
    raw,
    criteria,
):

    normalized = pd.DataFrame(
        {
            GRID_COL:
                raw[
                    GRID_COL
                ].astype(str)
        }
    )

    for criterion in criteria:

        values = pd.to_numeric(
            raw[
                criterion
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        min_value = float(
            np.min(
                values
            )
        )

        max_value = float(
            np.max(
                values
            )
        )

        span = (
            max_value
            - min_value
        )

        if span <= TOL:

            raise RuntimeError(
                f"{criterion}: "
                "no cross-grid variation."
            )

        normalized[
            criterion
        ] = (
            values
            - min_value
        ) / span

    return normalized


def critic_weights(
    normalized,
    criteria,
):

    X = (
        normalized[
            criteria
        ]
        .astype(float)
    )

    corr = X.corr(
        method="pearson"
    )

    if corr.isna().any().any():

        raise RuntimeError(
            "CRITIC correlation "
            "contains NaN."
        )

    std_population = X.std(
        axis=0,
        ddof=0,
    )

    conflict = (
        1.0
        - corr
    ).sum(
        axis=1
    )

    information = (
        std_population
        * conflict
    )

    information = information.clip(
        lower=0.0
    )

    total = float(
        information.sum()
    )

    if total <= TOL:

        raise RuntimeError(
            "Invalid CRITIC "
            "information total."
        )

    return (
        information
        / total
    ).to_numpy(
        dtype=float
    )


def topsis(
    normalized,
    criteria,
    weights,
):

    X = (
        normalized[
            criteria
        ]
        .to_numpy(
            dtype=float
        )
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    weights = (
        weights
        / weights.sum()
    )

    weighted = (
        X
        * weights.reshape(
            1,
            -1,
        )
    )

    ideal_best = np.max(
        weighted,
        axis=0,
    )

    ideal_worst = np.min(
        weighted,
        axis=0,
    )

    d_best = np.sqrt(
        np.sum(
            (
                weighted
                - ideal_best
            ) ** 2,
            axis=1,
        )
    )

    d_worst = np.sqrt(
        np.sum(
            (
                weighted
                - ideal_worst
            ) ** 2,
            axis=1,
        )
    )

    score = (
        d_worst
        /
        (
            d_best
            + d_worst
        )
    )

    order = pd.DataFrame(
        {
            "row_index":
                np.arange(
                    len(score)
                ),

            GRID_COL:
                normalized[
                    GRID_COL
                ].astype(str),

            "score":
                score,
        }
    )

    order = order.sort_values(
        [
            "score",
            GRID_COL,
        ],
        ascending=[
            False,
            True,
        ],
        kind="mergesort",
    )

    rank = np.empty(
        len(score),
        dtype=int,
    )

    rank[
        order[
            "row_index"
        ].to_numpy(
            dtype=int
        )
    ] = np.arange(
        1,
        len(score) + 1,
    )

    return (
        score,
        rank,
    )


# ============================================================
# OUTCOME METRICS
# ============================================================

def exact_descending_rank(
    values,
    grid_ids,
):

    d = pd.DataFrame(
        {
            "row":
                np.arange(
                    len(values)
                ),

            GRID_COL:
                np.asarray(
                    grid_ids,
                    dtype=str,
                ),

            "value":
                np.asarray(
                    values,
                    dtype=float,
                ),
        }
    )

    d = d.sort_values(
        [
            "value",
            GRID_COL,
        ],
        ascending=[
            False,
            True,
        ],
        kind="mergesort",
    )

    ranks = np.empty(
        len(d),
        dtype=int,
    )

    ranks[
        d[
            "row"
        ].to_numpy(
            dtype=int
        )
    ] = np.arange(
        1,
        len(d) + 1,
    )

    return ranks


def top_overlap(
    priority_rank,
    actual_heat_rank,
    k,
):

    priority_set = set(
        np.flatnonzero(
            priority_rank
            <= k
        )
    )

    heat_set = set(
        np.flatnonzero(
            actual_heat_rank
            <= k
        )
    )

    return float(
        len(
            priority_set
            & heat_set
        )
        / k
    )


# ============================================================
# VERIFY / LOAD
# ============================================================

section(
    "PRIORITY VALIDATION 05 - HISTORICAL RETROSPECTIVE BACKTEST"
)


for path, label in [
    (
        BASE_PATH,
        "Historical base",
    ),
    (
        ENGINEERED_PATH,
        "Engineered historical data",
    ),
    (
        GRID_UTM_PATH,
        "UTM grid",
    ),
    (
        TRAIN_PATH,
        "2018-2023 model-ready train",
    ),
    (
        VALIDATION_2024_PATH,
        "2024 model-ready validation",
    ),
    (
        STEP19C_MANIFEST_PATH,
        "Step 19C pre-2024 candidate manifest",
    ),
    (
        STEP27_FREEZE_PATH,
        "Step 27 forecasting freeze",
    ),
]:

    require_file(
        path,
        label,
    )


base = pd.read_csv(
    BASE_PATH,
    dtype={
        GRID_COL: str,
        AREA_COL: str,
    },
    low_memory=False,
)

engineered = pd.read_csv(
    ENGINEERED_PATH,
    dtype={
        GRID_COL: str,
        AREA_COL: str,
    },
    low_memory=False,
)

train_2018_2023 = pd.read_csv(
    TRAIN_PATH,
    dtype={
        GRID_COL: str,
        AREA_COL: str,
    },
    low_memory=False,
)

validation_2024 = pd.read_csv(
    VALIDATION_2024_PATH,
    dtype={
        GRID_COL: str,
        AREA_COL: str,
    },
    low_memory=False,
)


for frame in [
    base,
    engineered,
    train_2018_2023,
    validation_2024,
]:

    frame[
        YEAR_COL
    ] = pd.to_numeric(
        frame[
            YEAR_COL
        ],
        errors="raise",
    ).astype(int)


require_columns(
    base,
    [
        GRID_COL,
        YEAR_COL,
        AREA_COL,
        TARGET,
        *TEMPORAL_SOURCES,
        *STATIC_PREDICTORS,
        *STRUCTURAL_PREDICTORS,
    ],
    "Historical base",
)


if (
    base[
        GRID_COL
    ].nunique()
    != EXPECTED_GRIDS
):

    raise RuntimeError(
        "Historical base does not "
        "contain 2,194 grids."
    )


predictors = [
    column
    for column in (
        train_2018_2023.columns
    )
    if column not in {
        TARGET,
        GRID_COL,
        AREA_COL,
    }
]


if len(
    predictors
) != EXPECTED_FEATURES:

    raise RuntimeError(
        f"Predictor count="
        f"{len(predictors)}, "
        f"expected="
        f"{EXPECTED_FEATURES}."
    )


if [
    column
    for column in (
        validation_2024.columns
    )
    if column not in {
        TARGET,
        GRID_COL,
        AREA_COL,
    }
] != predictors:

    raise RuntimeError(
        "2018-2023 and 2024 "
        "predictor schemas differ."
    )


with STEP19C_MANIFEST_PATH.open(
    "r",
    encoding="utf-8",
) as f:

    manifest19c = json.load(
        f
    )


candidate = next(
    (
        item
        for item
        in manifest19c[
            "candidates"
        ]
        if item[
            "strategy"
        ]
        ==
        "lgbm_anomaly_plus_prior_mean_level"
    ),
    None,
)


if candidate is None:

    raise RuntimeError(
        "Pre-2024 frozen LightGBM "
        "anomaly candidate not found."
    )


lgbm_params = (
    candidate[
        "implementation"
    ][
        "hyperparameters"
    ]
)


with STEP27_FREEZE_PATH.open(
    "r",
    encoding="utf-8",
) as f:

    freeze27 = json.load(
        f
    )


selected_methods = (
    freeze27[
        "dynamic_predictors"
    ][
        "selected_methods"
    ]
)


expected_methods = {
    "NDVI":
        "Rolling-3 Mean",

    "NDBI":
        "Rolling-3 Mean",

    "NDWI":
        "Rolling-3 Mean",

    "EVI":
        "Rolling-3 Mean",

    "Albedo":
        "Bounded Theil-Sen Trend",

    "NDBI_ADJ":
        "Rolling-3 Mean",

    "NDVI_ADJ":
        "Rolling-3 Mean",

    "NightLights":
        "Rolling-3 Mean",

    "green_mask":
        "Rolling-3 Mean",
}


if selected_methods != expected_methods:

    raise RuntimeError(
        "Unexpected Step-27 "
        "forecast methods."
    )


grid_utm = gpd.read_file(
    GRID_UTM_PATH
)

grid_utm[
    GRID_COL
] = (
    grid_utm[
        GRID_COL
    ]
    .astype(str)
)


neighbors = build_queen_neighbors(
    grid_utm
)


print(
    f"Historical rows : "
    f"{len(base):,}"
)

print(
    f"Unique grids    : "
    f"{base[GRID_COL].nunique():,}"
)

print(
    f"Frozen features : "
    f"{len(predictors)}"
)

print(
    "Backtests       : "
    "2023->2024, 2024->2025"
)


# ============================================================
# RUN BACKTESTS
# ============================================================

all_grid_results = []
all_metric_rows = []
all_weight_rows = []
all_quintile_rows = []
all_decision_rows = []


CRITERIA = [
    "future_heat_severity",
    "relative_trend_urgency",
    "green_deficit",
    "built_up_pressure",
    "blue_space_deficit",
    "modeled_intervention_benefit",
]


for test in BACKTESTS:

    cutoff_year = int(
        test[
            "cutoff_year"
        ]
    )

    target_year = int(
        test[
            "target_year"
        ]
    )


    section(
        f"BACKTEST: HISTORY <= {cutoff_year} -> ACTUAL {target_year}"
    )


    # --------------------------------------------------------
    # Historical information available at ranking origin
    # --------------------------------------------------------

    history = (
        base[
            base[
                YEAR_COL
            ]
            <= cutoff_year
        ]
        .copy()
    )


    expected_years = list(
        range(
            2015,
            cutoff_year + 1,
        )
    )


    if sorted(
        history[
            YEAR_COL
        ]
        .unique()
        .tolist()
    ) != expected_years:

        raise RuntimeError(
            f"{cutoff_year}: historical "
            "years are incomplete."
        )


    if (
        history[
            GRID_COL
        ].nunique()
        != EXPECTED_GRIDS
    ):

        raise RuntimeError(
            f"{cutoff_year}: grid "
            "count mismatch."
        )


    # --------------------------------------------------------
    # Train anomaly model using only eligible past years
    # --------------------------------------------------------

    if cutoff_year == 2023:

        model_train = (
            train_2018_2023
            .copy()
        )

    elif cutoff_year == 2024:

        model_train = pd.concat(
            [
                train_2018_2023,
                validation_2024,
            ],
            ignore_index=True,
        )

    else:

        raise ValueError(
            "Unsupported cutoff year."
        )


    model_years = sorted(
        model_train[
            YEAR_COL
        ]
        .unique()
        .tolist()
    )


    if max(
        model_years
    ) != cutoff_year:

        raise RuntimeError(
            "Training data includes "
            "incorrect final year."
        )


    for column in (
        predictors
        + [TARGET]
    ):

        model_train[
            column
        ] = pd.to_numeric(
            model_train[
                column
            ],
            errors="raise",
        )


    if model_train[
        predictors
        + [TARGET]
    ].isna().any().any():

        raise RuntimeError(
            "Missing model-training "
            "values found."
        )


    annual_mean_train = (
        model_train
        .groupby(
            YEAR_COL
        )[
            TARGET
        ]
        .transform(
            "mean"
        )
    )


    y_anomaly = (
        model_train[
            TARGET
        ].to_numpy(
            dtype=float
        )
        - annual_mean_train.to_numpy(
            dtype=float
        )
    )


    model = build_lightgbm(
        lgbm_params
    )


    model.fit(
        model_train[
            predictors
        ],
        y_anomaly,
    )


    print(
        f"Model training years : "
        f"{model_years}"
    )


    # --------------------------------------------------------
    # Forecast target-year base predictors
    # --------------------------------------------------------

    target_base = (
        build_next_year_base(
            history=history,
            target_year=
                target_year,
            selected_methods=
                selected_methods,
        )
    )


    if (
        target_base[
            GRID_COL
        ].nunique()
        != EXPECTED_GRIDS
    ):

        raise RuntimeError(
            "Target forecast base "
            "grid count mismatch."
        )


    # --------------------------------------------------------
    # Reconstruct target-year 93-feature matrix
    # --------------------------------------------------------

    year_index_slope, (
        year_index_intercept
    ) = infer_year_index_relation(
        engineered,
        cutoff_year,
    )


    combined_base = pd.concat(
        [
            history,
            target_base,
        ],
        ignore_index=True,
    )


    reconstructed = (
        add_engineered_features(
            combined_base=
                combined_base,

            neighbors=
                neighbors,

            year_index_slope=
                year_index_slope,

            year_index_intercept=
                year_index_intercept,
        )
    )


    target_features = (
        reconstructed[
            reconstructed[
                YEAR_COL
            ]
            == target_year
        ]
        .copy()
        .sort_values(
            GRID_COL
        )
        .reset_index(
            drop=True
        )
    )


    if len(
        target_features
    ) != EXPECTED_GRIDS:

        raise RuntimeError(
            "Target feature matrix "
            "row count mismatch."
        )


    missing_predictors = [
        p
        for p in predictors
        if p not in (
            target_features.columns
        )
    ]


    if missing_predictors:

        raise RuntimeError(
            "Target feature matrix "
            "missing predictors:\n"
            + str(
                missing_predictors
            )
        )


    X_target = (
        target_features[
            predictors
        ]
        .copy()
    )


    for feature in predictors:

        X_target[
            feature
        ] = pd.to_numeric(
            X_target[
                feature
            ],
            errors="raise",
        )


    if X_target.isna().any().any():

        bad = (
            X_target
            .columns[
                X_target
                .isna()
                .any()
            ]
            .tolist()
        )

        raise RuntimeError(
            "Target-year forecast "
            "features contain missing "
            f"values: {bad}"
        )


    if not np.isfinite(
        X_target.to_numpy(
            dtype=float
        )
    ).all():

        raise RuntimeError(
            "Target feature matrix "
            "contains non-finite values."
        )


    # --------------------------------------------------------
    # C1 - Target-year forecast LST
    # --------------------------------------------------------

    raw_anomaly = np.asarray(
        model.predict(
            X_target
        ),
        dtype=float,
    )


    centered_anomaly = (
        raw_anomaly
        - float(
            np.mean(
                raw_anomaly
            )
        )
    )


    annual_history = (
        history
        .groupby(
            YEAR_COL,
            as_index=False,
        )[
            TARGET
        ]
        .mean()
    )


    # Final frozen annual strategy = Expanding Mean
    annual_level_forecast = float(
        annual_history[
            TARGET
        ].mean()
    )


    predicted_target_lst = (
        annual_level_forecast
        + centered_anomaly
    )


    # --------------------------------------------------------
    # C2 - Relative trend through cutoff only
    # --------------------------------------------------------

    relative_trend = (
        historical_relative_sen_slope(
            history
        )
        .sort_values(
            GRID_COL
        )
        .reset_index(
            drop=True
        )
    )


    if not (
        relative_trend[
            GRID_COL
        ].astype(str)
        .equals(
            target_features[
                GRID_COL
            ].astype(str)
        )
    ):

        raise RuntimeError(
            "Trend and target feature "
            "grid order mismatch."
        )


    # --------------------------------------------------------
    # C6 - Combined modeled intervention benefit
    # --------------------------------------------------------

    moderate_ndbi = (
        derive_moderate_ndbi_reduction(
            history
        )
    )


    X_scenario = (
        apply_combined_scenario(
            X_base=
                X_target,

            history=
                history,

            moderate_ndbi_reduction=
                moderate_ndbi,
        )
    )


    if X_scenario.isna().any().any():

        raise RuntimeError(
            "Scenario feature matrix "
            "contains missing values."
        )


    scenario_raw = np.asarray(
        model.predict(
            X_scenario
        ),
        dtype=float,
    )


    combined_cooling_benefit = -(
        scenario_raw
        - raw_anomaly
    )


    print(
        f"MODERATE NDBI reduction "
        f"(past-only Q75) = "
        f"{moderate_ndbi:.6f}"
    )


    # --------------------------------------------------------
    # Build historical analogue of six-criterion matrix
    # --------------------------------------------------------

    target_base_sorted = (
        target_base
        .sort_values(
            GRID_COL
        )
        .reset_index(
            drop=True
        )
    )


    raw_decision = pd.DataFrame(
        {
            GRID_COL:
                target_features[
                    GRID_COL
                ].astype(str),

            "future_heat_severity":
                predicted_target_lst,

            "relative_trend_urgency":
                relative_trend[
                    "relative_heat_sen_slope"
                ].to_numpy(
                    dtype=float
                ),

            "green_deficit":
                (
                    1.0
                    - target_base_sorted[
                        "green_mask"
                    ].to_numpy(
                        dtype=float
                    )
                ),

            "built_up_pressure":
                target_base_sorted[
                    "NDBI"
                ].to_numpy(
                    dtype=float
                ),

            "blue_space_deficit":
                target_base_sorted[
                    "dist_water"
                ].to_numpy(
                    dtype=float
                ),

            "modeled_intervention_benefit":
                combined_cooling_benefit,
        }
    )


    normalized = (
        minmax_priority_matrix(
            raw_decision,
            CRITERIA,
        )
    )


    weights = critic_weights(
        normalized,
        CRITERIA,
    )


    priority_score, priority_rank = (
        topsis(
            normalized,
            CRITERIA,
            weights,
        )
    )


    raw_decision[
        "priority_score"
    ] = priority_score

    raw_decision[
        "priority_rank"
    ] = priority_rank

    raw_decision[
        "cutoff_year"
    ] = cutoff_year

    raw_decision[
        "target_year"
    ] = target_year


    # --------------------------------------------------------
    # Save weight evidence
    # --------------------------------------------------------

    for criterion, weight in zip(
        CRITERIA,
        weights,
    ):

        all_weight_rows.append(
            {
                "cutoff_year":
                    cutoff_year,

                "target_year":
                    target_year,

                "criterion":
                    criterion,

                "critic_weight":
                    float(
                        weight
                    ),

                "critic_weight_pct":
                    100.0
                    * float(
                        weight
                    ),
            }
        )


    # --------------------------------------------------------
    # Open target-year actual observations ONLY NOW
    # --------------------------------------------------------

    actual = (
        base[
            base[
                YEAR_COL
            ]
            == target_year
        ][
            [
                GRID_COL,
                TARGET,
            ]
        ]
        .copy()
        .sort_values(
            GRID_COL
        )
        .reset_index(
            drop=True
        )
    )


    if len(
        actual
    ) != EXPECTED_GRIDS:

        raise RuntimeError(
            "Actual target-year "
            "row count mismatch."
        )


    if not actual[
        GRID_COL
    ].astype(str).equals(
        raw_decision[
            GRID_COL
        ].astype(str)
    ):

        raise RuntimeError(
            "Actual and ranking "
            "grid order mismatch."
        )


    actual_lst = actual[
        TARGET
    ].to_numpy(
        dtype=float
    )


    actual_heat_percentile = (
        pd.Series(
            actual_lst
        )
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(
            dtype=float
        )
    )


    actual_heat_rank = (
        exact_descending_rank(
            actual_lst,
            actual[
                GRID_COL
            ].to_numpy(
                dtype=str
            ),
        )
    )


    # Cutoff-year observed heat for change diagnostic
    cutoff_actual = (
        base[
            base[
                YEAR_COL
            ]
            == cutoff_year
        ][
            [
                GRID_COL,
                TARGET,
            ]
        ]
        .copy()
        .sort_values(
            GRID_COL
        )
        .reset_index(
            drop=True
        )
    )


    if not cutoff_actual[
        GRID_COL
    ].astype(str).equals(
        actual[
            GRID_COL
        ].astype(str)
    ):

        raise RuntimeError(
            "Cutoff/target grid "
            "alignment failed."
        )


    actual_change = (
        actual_lst
        - cutoff_actual[
            TARGET
        ].to_numpy(
            dtype=float
        )
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    n = EXPECTED_GRIDS

    top10_k = int(
        math.ceil(
            n
            * TOP10_FRAC
        )
    )

    top25_k = int(
        math.ceil(
            n
            * TOP25_FRAC
        )
    )


    score_spearman = float(
        spearmanr(
            priority_score,
            actual_heat_percentile,
        ).statistic
    )


    score_kendall = float(
        kendalltau(
            priority_score,
            actual_heat_percentile,
        ).statistic
    )


    change_spearman = float(
        spearmanr(
            priority_score,
            actual_change,
        ).statistic
    )


    top10_overlap = top_overlap(
        priority_rank,
        actual_heat_rank,
        top10_k,
    )


    top25_overlap = top_overlap(
        priority_rank,
        actual_heat_rank,
        top25_k,
    )


    top100_overlap = top_overlap(
        priority_rank,
        actual_heat_rank,
        100,
    )


    top10_mask = (
        priority_rank
        <= top10_k
    )

    bottom10_mask = (
        priority_rank
        > (
            n
            - top10_k
        )
    )


    top10_mean_actual_lst = float(
        np.mean(
            actual_lst[
                top10_mask
            ]
        )
    )


    bottom10_mean_actual_lst = float(
        np.mean(
            actual_lst[
                bottom10_mask
            ]
        )
    )


    top10_minus_bottom10_lst = (
        top10_mean_actual_lst
        - bottom10_mean_actual_lst
    )


    forecast_mae = float(
        mean_absolute_error(
            actual_lst,
            predicted_target_lst,
        )
    )


    forecast_rmse = float(
        math.sqrt(
            mean_squared_error(
                actual_lst,
                predicted_target_lst,
            )
        )
    )


    forecast_r2 = float(
        r2_score(
            actual_lst,
            predicted_target_lst,
        )
    )


    metric_row = {
        "cutoff_year":
            cutoff_year,

        "target_year":
            target_year,

        "ranking_constructed_without_target_year_actuals":
            True,

        "model_training_end_year":
            cutoff_year,

        "annual_level_history_end_year":
            cutoff_year,

        "moderate_ndbi_reduction_past_only_q75":
            moderate_ndbi,

        "forecast_mae_c":
            forecast_mae,

        "forecast_rmse_c":
            forecast_rmse,

        "forecast_r2":
            forecast_r2,

        "priority_score_vs_actual_heat_spearman":
            score_spearman,

        "priority_score_vs_actual_heat_kendall":
            score_kendall,

        "priority_score_vs_actual_lst_change_spearman":
            change_spearman,

        "top100_priority_vs_actual_hottest_overlap_pct":
            100.0
            * top100_overlap,

        "top10pct_priority_vs_actual_hottest_overlap_pct":
            100.0
            * top10_overlap,

        "top25pct_priority_vs_actual_hottest_overlap_pct":
            100.0
            * top25_overlap,

        "top10pct_priority_mean_actual_lst_c":
            top10_mean_actual_lst,

        "bottom10pct_priority_mean_actual_lst_c":
            bottom10_mean_actual_lst,

        "top10_minus_bottom10_actual_lst_c":
            top10_minus_bottom10_lst,
    }


    all_metric_rows.append(
        metric_row
    )


    # --------------------------------------------------------
    # Grid-level result
    # --------------------------------------------------------

    grid_result = (
        raw_decision.copy()
    )


    grid_result[
        "predicted_target_lst_c"
    ] = predicted_target_lst

    grid_result[
        "actual_target_lst_c"
    ] = actual_lst

    grid_result[
        "actual_target_heat_percentile"
    ] = actual_heat_percentile

    grid_result[
        "actual_target_heat_rank"
    ] = actual_heat_rank

    grid_result[
        "actual_lst_change_from_cutoff_c"
    ] = actual_change

    grid_result[
        "is_top10pct_priority"
    ] = top10_mask

    grid_result[
        "is_actual_top10pct_hottest"
    ] = (
        actual_heat_rank
        <= top10_k
    )


    # Priority quintile:
    # rank 1 = highest priority.
    grid_result[
        "priority_quintile"
    ] = pd.qcut(
        grid_result[
            "priority_rank"
        ],
        q=5,
        labels=[
            "Q5 Highest Priority",
            "Q4",
            "Q3",
            "Q2",
            "Q1 Lowest Priority",
        ],
    )


    all_grid_results.append(
        grid_result
    )


    # --------------------------------------------------------
    # Quintile summary
    # --------------------------------------------------------

    quintile_summary = (
        grid_result
        .groupby(
            "priority_quintile",
            observed=True,
        )
        .agg(
            grid_count=(
                GRID_COL,
                "size",
            ),

            mean_priority_score=(
                "priority_score",
                "mean",
            ),

            mean_actual_target_lst_c=(
                "actual_target_lst_c",
                "mean",
            ),

            median_actual_target_lst_c=(
                "actual_target_lst_c",
                "median",
            ),

            mean_actual_heat_percentile=(
                "actual_target_heat_percentile",
                "mean",
            ),

            mean_actual_lst_change_c=(
                "actual_lst_change_from_cutoff_c",
                "mean",
            ),
        )
        .reset_index()
    )


    quintile_summary[
        "cutoff_year"
    ] = cutoff_year

    quintile_summary[
        "target_year"
    ] = target_year


    all_quintile_rows.append(
        quintile_summary
    )


    all_decision_rows.append(
        raw_decision
    )


    # --------------------------------------------------------
    # Terminal result
    # --------------------------------------------------------

    print(
        "\nRetrospective validation metrics:"
    )

    print(
        f"  Forecast MAE                  : "
        f"{forecast_mae:.4f} °C"
    )

    print(
        f"  Forecast RMSE                 : "
        f"{forecast_rmse:.4f} °C"
    )

    print(
        f"  Forecast R²                   : "
        f"{forecast_r2:.4f}"
    )

    print(
        f"  Priority vs actual heat rho   : "
        f"{score_spearman:.4f}"
    )

    print(
        f"  Priority vs actual heat tau   : "
        f"{score_kendall:.4f}"
    )

    print(
        f"  Top-100 hot-grid overlap      : "
        f"{100 * top100_overlap:.2f}%"
    )

    print(
        f"  Top-10% hot-grid overlap      : "
        f"{100 * top10_overlap:.2f}%"
    )

    print(
        f"  Top-25% hot-grid overlap      : "
        f"{100 * top25_overlap:.2f}%"
    )

    print(
        f"  Top10 - Bottom10 actual LST   : "
        f"{top10_minus_bottom10_lst:+.4f} °C"
    )

    print(
        "\nCRITIC weights:"
    )

    for criterion, weight in zip(
        CRITERIA,
        weights,
    ):

        print(
            f"  {criterion:32s} "
            f"{weight:.6f}"
        )


# ============================================================
# COMBINE OUTPUTS
# ============================================================

metrics_df = pd.DataFrame(
    all_metric_rows
)

grid_results_df = pd.concat(
    all_grid_results,
    ignore_index=True,
)

weights_df = pd.DataFrame(
    all_weight_rows
)

quintile_df = pd.concat(
    all_quintile_rows,
    ignore_index=True,
)

decision_df = pd.concat(
    all_decision_rows,
    ignore_index=True,
)


metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)

grid_results_df.to_csv(
    GRID_RESULTS_PATH,
    index=False,
)

weights_df.to_csv(
    WEIGHTS_PATH,
    index=False,
)

quintile_df.to_csv(
    QUINTILE_PATH,
    index=False,
)

decision_df.to_csv(
    DECISION_MATRIX_PATH,
    index=False,
)


# ============================================================
# COMBINED SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "backtest_count":
                len(
                    metrics_df
                ),

            "mean_priority_vs_actual_heat_spearman":
                float(
                    metrics_df[
                        "priority_score_vs_actual_heat_spearman"
                    ].mean()
                ),

            "minimum_priority_vs_actual_heat_spearman":
                float(
                    metrics_df[
                        "priority_score_vs_actual_heat_spearman"
                    ].min()
                ),

            "mean_top100_hot_overlap_pct":
                float(
                    metrics_df[
                        "top100_priority_vs_actual_hottest_overlap_pct"
                    ].mean()
                ),

            "mean_top10pct_hot_overlap_pct":
                float(
                    metrics_df[
                        "top10pct_priority_vs_actual_hottest_overlap_pct"
                    ].mean()
                ),

            "mean_top25pct_hot_overlap_pct":
                float(
                    metrics_df[
                        "top25pct_priority_vs_actual_hottest_overlap_pct"
                    ].mean()
                ),

            "mean_top10_minus_bottom10_actual_lst_c":
                float(
                    metrics_df[
                        "top10_minus_bottom10_actual_lst_c"
                    ].mean()
                ),

            "interpretation":
                (
                    "Retrospective one-year hindcast "
                    "validation of the final priority "
                    "framework. Rankings were constructed "
                    "without target-year observed LST or "
                    "target-year observed environmental "
                    "predictors."
                ),
        }
    ]
)


summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# FIGURE
# ============================================================

plot_df = (
    quintile_df.copy()
)

quintile_order = [
    "Q1 Lowest Priority",
    "Q2",
    "Q3",
    "Q4",
    "Q5 Highest Priority",
]


fig, ax = plt.subplots(
    figsize=(
        10,
        6,
    )
)


for target_year in sorted(
    plot_df[
        "target_year"
    ].unique()
):

    part = (
        plot_df[
            plot_df[
                "target_year"
            ]
            == target_year
        ]
        .copy()
    )

    part[
        "priority_quintile"
    ] = pd.Categorical(
        part[
            "priority_quintile"
        ],
        categories=
            quintile_order,
        ordered=True,
    )

    part = part.sort_values(
        "priority_quintile"
    )

    ax.plot(
        part[
            "priority_quintile"
        ].astype(str),

        part[
            "mean_actual_target_lst_c"
        ],

        marker="o",

        label=
            f"Actual {target_year}",
    )


ax.set_xlabel(
    "Retrospective Priority Quintile"
)

ax.set_ylabel(
    "Mean Actual Next-Year LST (°C)"
)

ax.set_title(
    "Observed Next-Year Heat Across Retrospective Priority Quintiles"
)

ax.tick_params(
    axis="x",
    rotation=20,
)

ax.legend()

fig.tight_layout()


fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FINAL TERMINAL SUMMARY
# ============================================================

section(
    "HISTORICAL PRIORITY BACKTEST SUMMARY"
)


print(
    metrics_df.to_string(
        index=False
    )
)


print(
    "\nCombined:"
)

print(
    summary_df.T.to_string(
        header=False
    )
)


print(
    "\nIMPORTANT INTERPRETATION:"
)

print(
    "- This is retrospective one-year "
    "hindcast validation."
)

print(
    "- Target-year actual LST and target-year "
    "observed predictor values were NOT used "
    "to construct the ranking."
)

print(
    "- Higher priority is expected to be "
    "associated with stronger later heat "
    "burden, but priority is not identical "
    "to heat rank because it also includes "
    "trend, green, built-up, water-access "
    "and intervention-benefit dimensions."
)

print(
    "- Therefore these metrics provide "
    "temporal outcome relevance evidence, "
    "not ground-truth priority accuracy."
)


print(
    "\nSaved:"
)

print(METRICS_PATH)
print(GRID_RESULTS_PATH)
print(WEIGHTS_PATH)
print(QUINTILE_PATH)
print(DECISION_MATRIX_PATH)
print(SUMMARY_PATH)
print(FIGURE_PATH)


print(
    "\nHISTORICAL RETROSPECTIVE "
    "PRIORITY BACKTEST COMPLETE."
)