"""
STEP 29A V2 - PERSISTENT GREEN-COVER WHAT-IF COOLING SCENARIO
Urban Heat Research - Kaduwela, Sri Lanka

WHY V2?
-------
Step 29A V1 changed green cover independently in each future year while keeping
that grid's future history at the baseline trajectory. That is useful as a
direct perturbation diagnostic, but it is not the most realistic planning
intervention.

V2 assumes the planner implements the green-cover intervention in 2026 and
MAINTAINS it through 2030.

For every independently simulated grid:

    2026 baseline green + scenario increase
    2027 baseline green + same scenario increase
    2028 baseline green + same scenario increase
    2029 baseline green + same scenario increase
    2030 baseline green + same scenario increase

All values are clipped to [0, 1].

The grid's future green-cover temporal features are then rebuilt year-by-year
using the scenario trajectory:

    green_mask_lag1
    green_mask_lag2
    green_mask_hist3_mean
    green_mask_hist3_std
    green_mask_hist3_slope
    green_mask_delta1

This means, for example, the 2026 scenario value becomes part of the 2027
history, and the intervention progressively enters the rolling 3-year history.

IMPORTANT LOCAL-SCENARIO SEMANTICS
----------------------------------
Each grid is still an INDEPENDENT planner-selected counterfactual.

When Grid A is simulated:
- Grid A receives the persistent intervention from 2026 onward.
- Its neighbours remain at their Step-28 baseline future forecasts.
- green_mask_nbr_mean therefore remains baseline.
- green_mask_nbr_contrast is recalculated for Grid A.

The batch of 2,194 grids is NOT interpreted as simultaneous citywide greening.

FINAL LST EFFECT
----------------
The frozen Step-22B LightGBM predicts the raw spatial-anomaly component.

For each grid-year:

    local_anomaly_delta
      = scenario_raw_anomaly - baseline_raw_anomaly

    scenario_LST
      = Step-28 baseline LST + local_anomaly_delta

    cooling_benefit
      = baseline LST - scenario LST
      = -local_anomaly_delta

No scenario re-centering is performed because these rows are independent local
counterfactuals rather than one simultaneous citywide intervention.

INTERPRETATION
--------------
The output is modeled statistical sensitivity, NOT proof of causal or guaranteed
physical cooling. The model is predictive and was not trained as a causal
intervention model.

No result is forced to be cooling or monotonic.
No model is retrained.
Step-28 baseline forecasts remain unchanged.
Uncertainty remains Step 30.
"""

from pathlib import Path
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except ImportError as exc:
    raise ImportError(
        "\ngeopandas is required for Step 29A V2.\n"
        "Install it with:\n"
        "    pip install geopandas\n"
    ) from exc


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"

HIST_YEARS_REQUIRED = [2023, 2024, 2025]
FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]
INTERVENTION_START_YEAR = 2026

EXPECTED_GRIDS = 2194
EXPECTED_FUTURE_ROWS = EXPECTED_GRIDS * len(FUTURE_YEARS)
EXPECTED_FEATURES = 93

SCENARIO_INCREASE_PP = [5, 10, 20]
PRIMARY_SCENARIO_PP = 5

GREEN_MIN = 0.0
GREEN_MAX = 1.0

FORMULA_TOL = 1e-8
PREDICTION_REPRO_TOL = 1e-10
UNCHANGED_FEATURE_TOL = 1e-12

EXPECTED_HASHES = {
    "kaduwela_preprocessed_base.csv":
        "0e030b2ec0b5c4080cd4226bec798189a178aa45efd9962d19f38dc9cb170f46",
    "21_final_feature_list.csv":
        "6769a195b5027850ebd458a4bef7c9da0ecc43c4969bcc81f746e46bce3be9c0",
    "22b_production_lightgbm_anomaly_trained_2018_2025.joblib":
        "a3548c58ba0a708c81b4f6471e9183aa70ce9572c716ee55bf2eddafe183e24f",
    "28_future_model_features_2026_2030.csv":
        "d1e68a2e5d063f231d7a4d379e2534005ad5873f3a6017d3c4f28de9b8acba38",
    "28_future_grid_lst_predictions_2026_2030.csv":
        "55586971042bbce04845a4a33ae724f51b8417fce9bafc2c15bc0a5e75d7ff3e",
    "28_future_prediction_audit.csv":
        "368daa91dcbce7ac009c8f11f0ecedde5817d5eb1ed0d2c99888f7be9017f3af",
    "kaduwela_grid_geometry_wgs84.geojson":
        "6b758ddaae19b54219404b86740472038cfd21391cf349ea766124678050f608",
}


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_PATH = (
    ROOT / "data" / "processed"
    / "kaduwela_preprocessed_base.csv"
)

FEATURE_LIST_PATH = (
    ROOT / "outputs" / "reports"
    / "21_final_feature_list.csv"
)

MODEL_PATH = (
    ROOT / "models" / "final"
    / "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
)

FUTURE_FEATURE_PATH = (
    ROOT / "outputs" / "forecasting"
    / "28_future_model_features_2026_2030.csv"
)

FUTURE_PRED_PATH = (
    ROOT / "outputs" / "forecasting"
    / "28_future_grid_lst_predictions_2026_2030.csv"
)

STEP28_AUDIT_PATH = (
    ROOT / "outputs" / "reports"
    / "28_future_prediction_audit.csv"
)

GRID_WGS84_PATH = (
    ROOT / "data" / "processed"
    / "kaduwela_grid_geometry_wgs84.geojson"
)

BOUNDARY_KML_PATH = (
    ROOT / "data" / "raw" / "boundary"
    / "kaduwela_kmc_boundary.kml"
)

SCENARIO_DIR = ROOT / "outputs" / "scenarios"
MAP_DIR = ROOT / "outputs" / "maps"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [
    SCENARIO_DIR,
    MAP_DIR,
    FIGURE_DIR,
    REPORT_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# V2 outputs are deliberately separate from V1 diagnostic outputs.
ALL_SCENARIOS_PATH = (
    SCENARIO_DIR
    / "29a_v2_persistent_green_scenarios_2026_2030.csv"
)

PRIMARY_SCENARIO_PATH = (
    SCENARIO_DIR
    / f"29a_v2_persistent_green_primary_{PRIMARY_SCENARIO_PP}pp_2026_2030.csv"
)

SCENARIO_FEATURE_PATH = (
    SCENARIO_DIR
    / "29a_v2_persistent_green_scenario_features_2026_2030.csv"
)

SUMMARY_PATH = (
    SCENARIO_DIR
    / "29a_v2_persistent_green_scenario_summary.csv"
)

TEMPORAL_DIAGNOSTIC_PATH = (
    SCENARIO_DIR
    / "29a_v2_persistent_green_temporal_diagnostic.csv"
)

MONOTONIC_PATH = (
    SCENARIO_DIR
    / "29a_v2_persistent_green_response_monotonicity.csv"
)

BASELINE_RECON_PATH = (
    REPORT_DIR
    / "29a_v2_baseline_green_feature_reconstruction_audit.csv"
)

AUDIT_PATH = (
    REPORT_DIR
    / "29a_v2_persistent_green_scenario_audit.csv"
)

METHOD_PATH = (
    REPORT_DIR
    / "29a_v2_persistent_green_scenario_methodology.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "29a_v2_persistent_green_scenario_report.txt"
)

HASH_PATH = (
    REPORT_DIR
    / "29a_v2_persistent_green_scenario_hashes.csv"
)

SUMMARY_FIGURE_PATH = (
    FIGURE_DIR
    / "29a_v2_mean_cooling_benefit_by_green_scenario.png"
)


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 132)
    print(title)
    print("=" * 132)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def verify_hash(path, expected, label):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual  : {actual}"
        )

    print(
        f"PASS hash: {label} = {actual}"
    )

    return actual


def require_all_pass(path, label):
    audit = pd.read_csv(path)

    if "status" not in audit.columns:
        raise RuntimeError(
            f"{label} audit has no status column."
        )

    bad = audit[
        audit["status"].astype(str).str.upper()
        != "PASS"
    ]

    if not bad.empty:
        print(
            bad.to_string(index=False)
        )

        raise RuntimeError(
            f"{label} audit is not fully PASS."
        )


def detect_feature_column(feature_df):
    for conventional in [
        "feature",
        "feature_name",
        "predictor",
    ]:
        if conventional in feature_df.columns:
            values = (
                feature_df[conventional]
                .dropna()
                .astype(str)
                .tolist()
            )

            if len(values) == EXPECTED_FEATURES:
                return conventional

    for column in feature_df.columns:
        values = (
            feature_df[column]
            .dropna()
            .astype(str)
            .tolist()
        )

        if (
            len(values) == EXPECTED_FEATURES
            and len(set(values)) == EXPECTED_FEATURES
        ):
            return column

    raise RuntimeError(
        "Could not identify the frozen 93-feature column."
    )


def sample_std_three(values):
    """
    Step-07 hist3_std uses pandas rolling std with ddof=1.
    Here V2 always has three previous years available for 2026-2030.
    """
    return float(
        np.std(
            np.asarray(values, dtype=float),
            ddof=1,
        )
    )


def slope_three(oldest, middle, newest):
    """
    Exact slope for three equally spaced annual points x=[0,1,2].
    For three points, OLS slope = (newest - oldest) / 2.
    """
    return float(
        (newest - oldest)
        / 2.0
    )


def build_green_temporal_features(
    historical_green,
    future_green,
):
    """
    historical_green:
        DataFrame columns grid_id, year, green_mask containing 2023-2025 actual.

    future_green:
        DataFrame columns grid_id, year, green_mask containing 2026-2030
        baseline OR scenario future green trajectory.

    Returns one row per future grid-year with:
        green_mask
        green_mask_lag1
        green_mask_lag2
        green_mask_hist3_mean
        green_mask_hist3_std
        green_mask_hist3_slope
        green_mask_delta1
    """

    hist_lookup = {
        (str(row[GRID_COL]), int(row[YEAR_COL])):
            float(row["green_mask"])
        for _, row in historical_green.iterrows()
    }

    future_lookup = {
        (str(row[GRID_COL]), int(row[YEAR_COL])):
            float(row["green_mask"])
        for _, row in future_green.iterrows()
    }

    grid_ids = sorted(
        future_green[GRID_COL]
        .astype(str)
        .unique()
        .tolist()
    )

    rows = []

    for grid_id in grid_ids:
        series = {}

        for year in HIST_YEARS_REQUIRED:
            key = (
                grid_id,
                year,
            )

            if key not in hist_lookup:
                raise RuntimeError(
                    f"Missing historical green value for {key}"
                )

            series[year] = hist_lookup[key]

        for year in FUTURE_YEARS:
            key = (
                grid_id,
                year,
            )

            if key not in future_lookup:
                raise RuntimeError(
                    f"Missing future green value for {key}"
                )

            current = future_lookup[key]

            prev1 = series[
                year - 1
            ]

            prev2 = series[
                year - 2
            ]

            prev3 = series[
                year - 3
            ]

            prior_chronological = [
                prev3,
                prev2,
                prev1,
            ]

            rows.append(
                {
                    GRID_COL: grid_id,
                    YEAR_COL: year,
                    "green_mask": current,
                    "green_mask_lag1": prev1,
                    "green_mask_lag2": prev2,
                    "green_mask_hist3_mean":
                        float(
                            np.mean(
                                prior_chronological
                            )
                        ),
                    "green_mask_hist3_std":
                        sample_std_three(
                            prior_chronological
                        ),
                    "green_mask_hist3_slope":
                        slope_three(
                            prev3,
                            prev2,
                            prev1,
                        ),
                    "green_mask_delta1":
                        float(
                            current
                            - prev1
                        ),
                }
            )

            # Persistence of the intervention:
            # the current scenario value becomes future history.
            series[year] = current

    result = pd.DataFrame(
        rows
    )

    result = (
        result
        .sort_values(
            [
                YEAR_COL,
                GRID_COL,
            ]
        )
        .reset_index(drop=True)
    )

    return result


def load_boundary_wgs84():
    if not BOUNDARY_KML_PATH.exists():
        return None

    try:
        boundary = gpd.read_file(
            BOUNDARY_KML_PATH
        )

        if boundary.empty:
            return None

        if boundary.crs is None:
            boundary = boundary.set_crs(
                "EPSG:4326"
            )
        else:
            boundary = boundary.to_crs(
                "EPSG:4326"
            )

        return boundary

    except Exception as exc:
        print(
            "WARNING: real Kaduwela KML boundary could not be loaded."
        )

        print(
            f"  {type(exc).__name__}: {exc}"
        )

        return None


# =============================================================================
# 0. VERIFY FROZEN INPUTS
# =============================================================================

section(
    "STEP 29A V2 - PERSISTENT GREEN-COVER WHAT-IF COOLING SCENARIO"
)

verify_hash(
    BASE_PATH,
    EXPECTED_HASHES[
        "kaduwela_preprocessed_base.csv"
    ],
    "Canonical historical base data",
)

verify_hash(
    FEATURE_LIST_PATH,
    EXPECTED_HASHES[
        "21_final_feature_list.csv"
    ],
    "Frozen Step-21 feature list",
)

verify_hash(
    MODEL_PATH,
    EXPECTED_HASHES[
        "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
    ],
    "Frozen Step-22B production LightGBM",
)

verify_hash(
    FUTURE_FEATURE_PATH,
    EXPECTED_HASHES[
        "28_future_model_features_2026_2030.csv"
    ],
    "Step-28 future frozen-feature matrix",
)

verify_hash(
    FUTURE_PRED_PATH,
    EXPECTED_HASHES[
        "28_future_grid_lst_predictions_2026_2030.csv"
    ],
    "Step-28 future grid-LST predictions",
)

verify_hash(
    STEP28_AUDIT_PATH,
    EXPECTED_HASHES[
        "28_future_prediction_audit.csv"
    ],
    "Step-28 strict audit",
)

verify_hash(
    GRID_WGS84_PATH,
    EXPECTED_HASHES[
        "kaduwela_grid_geometry_wgs84.geojson"
    ],
    "Corrected WGS84 grid geometry",
)

require_all_pass(
    STEP28_AUDIT_PATH,
    "Step 28",
)

print(
    "\nPASS: Step-28 audit is fully PASS."
)

print(
    "\nPersistent intervention design:"
)

print(
    f"  Intervention starts: {INTERVENTION_START_YEAR}"
)

for pp in SCENARIO_INCREASE_PP:
    marker = (
        " [PRIMARY]"
        if pp == PRIMARY_SCENARIO_PP
        else ""
    )

    print(
        f"  +{pp} percentage points green cover, maintained through 2030{marker}"
    )

print(
    "\nEach grid is evaluated as an independent persistent local counterfactual."
)


# =============================================================================
# 1. LOAD HISTORICAL GREEN + FUTURE BASELINE
# =============================================================================

section(
    "1. LOAD HISTORICAL GREEN HISTORY + FROZEN FUTURE BASELINE"
)

feature_df = pd.read_csv(
    FEATURE_LIST_PATH
)

feature_col = detect_feature_column(
    feature_df
)

ordered_features = (
    feature_df[
        feature_col
    ]
    .dropna()
    .astype(str)
    .tolist()
)

if len(ordered_features) != EXPECTED_FEATURES:
    raise RuntimeError(
        f"Feature count={len(ordered_features)}, "
        f"expected={EXPECTED_FEATURES}"
    )

base = pd.read_csv(
    BASE_PATH,
    dtype={
        GRID_COL: str,
        "area_name": str,
    },
)

base[YEAR_COL] = pd.to_numeric(
    base[YEAR_COL],
    errors="raise",
).astype(int)

historical_green = (
    base[
        base[
            YEAR_COL
        ].isin(
            HIST_YEARS_REQUIRED
        )
    ][
        [
            GRID_COL,
            YEAR_COL,
            "green_mask",
        ]
    ]
    .copy()
)

historical_green[
    "green_mask"
] = pd.to_numeric(
    historical_green[
        "green_mask"
    ],
    errors="raise",
)

expected_hist_rows = (
    EXPECTED_GRIDS
    * len(
        HIST_YEARS_REQUIRED
    )
)

if len(historical_green) != expected_hist_rows:
    raise RuntimeError(
        f"Historical green rows={len(historical_green)}, "
        f"expected={expected_hist_rows}"
    )

if historical_green.duplicated(
    [
        GRID_COL,
        YEAR_COL,
    ]
).any():
    raise RuntimeError(
        "Duplicate historical green grid-year rows."
    )

future = pd.read_csv(
    FUTURE_FEATURE_PATH,
    dtype={
        GRID_COL: str,
        "area_name": str,
    },
)

pred = pd.read_csv(
    FUTURE_PRED_PATH,
    dtype={
        GRID_COL: str,
        "area_name": str,
    },
)

for frame, label in [
    (future, "future features"),
    (pred, "future predictions"),
]:
    frame[YEAR_COL] = pd.to_numeric(
        frame[YEAR_COL],
        errors="raise",
    ).astype(int)

    if len(frame) != EXPECTED_FUTURE_ROWS:
        raise RuntimeError(
            f"{label}: rows={len(frame)}, "
            f"expected={EXPECTED_FUTURE_ROWS}"
        )

    if sorted(
        frame[
            YEAR_COL
        ]
        .unique()
        .tolist()
    ) != FUTURE_YEARS:
        raise RuntimeError(
            f"{label}: unexpected years."
        )

    if frame.duplicated(
        [
            GRID_COL,
            YEAR_COL,
        ]
    ).any():
        raise RuntimeError(
            f"{label}: duplicate grid-year rows."
        )

future = (
    future
    .sort_values(
        [
            YEAR_COL,
            GRID_COL,
        ]
    )
    .reset_index(drop=True)
)

pred = (
    pred
    .sort_values(
        [
            YEAR_COL,
            GRID_COL,
        ]
    )
    .reset_index(drop=True)
)

if not future[
    [
        GRID_COL,
        YEAR_COL,
    ]
].equals(
    pred[
        [
            GRID_COL,
            YEAR_COL,
        ]
    ]
):
    raise RuntimeError(
        "Future feature and prediction keys do not align."
    )

missing_features = [
    feature
    for feature in ordered_features
    if feature not in future.columns
]

if missing_features:
    raise RuntimeError(
        "Step-28 future matrix is missing frozen features:\n"
        + str(
            missing_features
        )
    )

X_base = future[
    ordered_features
].copy()

for feature in ordered_features:
    X_base[
        feature
    ] = pd.to_numeric(
        X_base[
            feature
        ],
        errors="raise",
    )

if X_base.isna().any().any():
    raise RuntimeError(
        "Step-28 frozen future matrix contains missing values."
    )

if not np.isfinite(
    X_base.to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "Step-28 frozen future matrix contains non-finite values."
    )

if "green_mask" not in X_base.columns:
    raise RuntimeError(
        "green_mask is not present in the frozen 93-feature model."
    )

baseline_future_green = future[
    [
        GRID_COL,
        YEAR_COL,
    ]
].copy()

baseline_future_green[
    "green_mask"
] = X_base[
    "green_mask"
].to_numpy(
    dtype=float
)

print(
    f"Historical green rows : {len(historical_green):,}"
)

print(
    f"Future baseline rows  : {len(future):,}"
)

print(
    f"Unique grids          : {future[GRID_COL].nunique():,}"
)

print(
    f"Frozen features       : {len(ordered_features)}"
)


# =============================================================================
# 2. STRICTLY RECONSTRUCT BASELINE GREEN TEMPORAL FEATURES
# =============================================================================

section(
    "2. STRICT BASELINE GREEN-TEMPORAL RECONSTRUCTION CHECK"
)

baseline_green_rebuilt = build_green_temporal_features(
    historical_green=historical_green,
    future_green=baseline_future_green,
)

if not baseline_green_rebuilt[
    [
        GRID_COL,
        YEAR_COL,
    ]
].equals(
    future[
        [
            GRID_COL,
            YEAR_COL,
        ]
    ]
):
    raise RuntimeError(
        "Rebuilt baseline green feature keys do not align with Step 28."
    )

candidate_temporal_features = [
    "green_mask_lag1",
    "green_mask_lag2",
    "green_mask_hist3_mean",
    "green_mask_hist3_std",
    "green_mask_hist3_slope",
    "green_mask_delta1",
]

candidate_current_features = [
    "vegetation_deficit",
    "ndbi_x_vegetation_deficit",
    "ndvi_x_green_mask",
    "green_mask_nbr_contrast",
]

retained_green_temporal = [
    feature
    for feature in candidate_temporal_features
    if feature in X_base.columns
]

retained_green_current = [
    feature
    for feature in candidate_current_features
    if feature in X_base.columns
]

print(
    "Retained green temporal features:"
)

for feature in retained_green_temporal:
    print(
        f"  {feature}"
    )

print(
    "\nRetained current-year green-dependent features:"
)

for feature in retained_green_current:
    print(
        f"  {feature}"
    )

recon_rows = []

# Temporal features
for feature in retained_green_temporal:
    error = (
        X_base[
            feature
        ].to_numpy(
            dtype=float
        )
        - baseline_green_rebuilt[
            feature
        ].to_numpy(
            dtype=float
        )
    )

    max_error = float(
        np.max(
            np.abs(
                error
            )
        )
    )

    recon_rows.append(
        {
            "feature": feature,
            "group":
                "green_temporal",
            "status":
                "PASS"
                if max_error
                <= FORMULA_TOL
                else "FAIL",
            "max_abs_error":
                max_error,
            "tolerance":
                FORMULA_TOL,
        }
    )

# Current-year derived features
baseline_green = X_base[
    "green_mask"
].to_numpy(
    dtype=float
)

if "vegetation_deficit" in X_base.columns:
    error = (
        X_base[
            "vegetation_deficit"
        ].to_numpy(
            dtype=float
        )
        - (
            1.0
            - baseline_green
        )
    )

    max_error = float(
        np.max(
            np.abs(
                error
            )
        )
    )

    recon_rows.append(
        {
            "feature":
                "vegetation_deficit",
            "group":
                "current_green_derived",
            "status":
                "PASS"
                if max_error
                <= FORMULA_TOL
                else "FAIL",
            "max_abs_error":
                max_error,
            "tolerance":
                FORMULA_TOL,
        }
    )

if "ndbi_x_vegetation_deficit" in X_base.columns:
    if "NDBI" not in X_base.columns:
        raise RuntimeError(
            "ndbi_x_vegetation_deficit retained but NDBI unavailable."
        )

    expected = (
        X_base[
            "NDBI"
        ].to_numpy(
            dtype=float
        )
        * (
            1.0
            - baseline_green
        )
    )

    error = (
        X_base[
            "ndbi_x_vegetation_deficit"
        ].to_numpy(
            dtype=float
        )
        - expected
    )

    max_error = float(
        np.max(
            np.abs(
                error
            )
        )
    )

    recon_rows.append(
        {
            "feature":
                "ndbi_x_vegetation_deficit",
            "group":
                "current_green_derived",
            "status":
                "PASS"
                if max_error
                <= FORMULA_TOL
                else "FAIL",
            "max_abs_error":
                max_error,
            "tolerance":
                FORMULA_TOL,
        }
    )

if "ndvi_x_green_mask" in X_base.columns:
    if "NDVI" not in X_base.columns:
        raise RuntimeError(
            "ndvi_x_green_mask retained but NDVI unavailable."
        )

    expected = (
        X_base[
            "NDVI"
        ].to_numpy(
            dtype=float
        )
        * baseline_green
    )

    error = (
        X_base[
            "ndvi_x_green_mask"
        ].to_numpy(
            dtype=float
        )
        - expected
    )

    max_error = float(
        np.max(
            np.abs(
                error
            )
        )
    )

    recon_rows.append(
        {
            "feature":
                "ndvi_x_green_mask",
            "group":
                "current_green_derived",
            "status":
                "PASS"
                if max_error
                <= FORMULA_TOL
                else "FAIL",
            "max_abs_error":
                max_error,
            "tolerance":
                FORMULA_TOL,
        }
    )

# For a local intervention, neighbour mean stays baseline.
# We verify the baseline contrast relation first.
if "green_mask_nbr_contrast" in X_base.columns:
    if "green_mask_nbr_mean" in X_base.columns:
        baseline_neighbor_mean = X_base[
            "green_mask_nbr_mean"
        ].to_numpy(
            dtype=float
        )
    else:
        # If mean was excluded but contrast retained, infer baseline neighbour mean
        # from baseline current value - baseline contrast.
        baseline_neighbor_mean = (
            baseline_green
            - X_base[
                "green_mask_nbr_contrast"
            ].to_numpy(
                dtype=float
            )
        )

    expected = (
        baseline_green
        - baseline_neighbor_mean
    )

    error = (
        X_base[
            "green_mask_nbr_contrast"
        ].to_numpy(
            dtype=float
        )
        - expected
    )

    max_error = float(
        np.max(
            np.abs(
                error
            )
        )
    )

    recon_rows.append(
        {
            "feature":
                "green_mask_nbr_contrast",
            "group":
                "current_green_derived",
            "status":
                "PASS"
                if max_error
                <= FORMULA_TOL
                else "FAIL",
            "max_abs_error":
                max_error,
            "tolerance":
                FORMULA_TOL,
        }
    )

baseline_recon_audit = pd.DataFrame(
    recon_rows
)

baseline_recon_audit.to_csv(
    BASELINE_RECON_PATH,
    index=False,
)

print(
    baseline_recon_audit.to_string(
        index=False
    )
)

if baseline_recon_audit.empty:
    raise RuntimeError(
        "No green-dependent frozen feature was available for reconstruction."
    )

if (
    baseline_recon_audit[
        "status"
    ]
    != "PASS"
).any():
    raise RuntimeError(
        "Green temporal/current feature reconstruction does not match Step 28."
    )

print(
    "\nPASS: Step-28 green-dependent feature construction reproduced."
)


# =============================================================================
# 3. REPRODUCE FROZEN STEP-28 BASELINE LIGHTGBM
# =============================================================================

section(
    "3. REPRODUCE FROZEN STEP-28 BASELINE LIGHTGBM OUTPUT"
)

model = joblib.load(
    MODEL_PATH
)

baseline_raw_model = np.asarray(
    model.predict(
        X_base
    ),
    dtype=float,
)

baseline_raw_saved = pd.to_numeric(
    pred[
        "raw_spatial_anomaly_prediction_c"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)

baseline_predicted_lst = pd.to_numeric(
    pred[
        "predicted_lst_c"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)

baseline_centered_anomaly = pd.to_numeric(
    pred[
        "centered_spatial_anomaly_c"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)

annual_level = pd.to_numeric(
    pred[
        "forecast_annual_mean_lst_c"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)

baseline_reproduction_error = float(
    np.max(
        np.abs(
            baseline_raw_model
            - baseline_raw_saved
        )
    )
)

print(
    f"Max raw prediction reproduction error: "
    f"{baseline_reproduction_error:.12e} °C"
)

if (
    baseline_reproduction_error
    > PREDICTION_REPRO_TOL
):
    raise RuntimeError(
        "Could not reproduce frozen Step-28 LightGBM output."
    )

print(
    "PASS: frozen Step-28 baseline LightGBM output reproduced."
)


# =============================================================================
# 4. RUN PERSISTENT LOCAL GREEN-COVER SCENARIOS
# =============================================================================

section(
    "4. RUN PERSISTENT 2026-2030 LOCAL GREEN-COVER SCENARIOS"
)

scenario_rows = []
scenario_feature_exports = []
scenario_temporal_diagnostics = []
scenario_feature_change_audits = []

allowed_changed_features = {
    "green_mask",
    *retained_green_temporal,
    *retained_green_current,
}

# green_mask_nbr_mean intentionally remains baseline for local intervention.
allowed_changed_features.discard(
    "green_mask_nbr_mean"
)

for pp in SCENARIO_INCREASE_PP:
    print(
        f"\nRunning persistent +{pp} pp scenario..."
    )

    requested_fraction = (
        pp
        / 100.0
    )

    scenario_future_green = (
        baseline_future_green
        .copy()
    )

    scenario_future_green[
        "baseline_green_mask"
    ] = scenario_future_green[
        "green_mask"
    ]

    scenario_future_green[
        "green_mask"
    ] = np.clip(
        scenario_future_green[
            "baseline_green_mask"
        ].to_numpy(
            dtype=float
        )
        + requested_fraction,
        GREEN_MIN,
        GREEN_MAX,
    )

    scenario_future_green[
        "actual_green_increase_pp"
    ] = (
        scenario_future_green[
            "green_mask"
        ]
        - scenario_future_green[
            "baseline_green_mask"
        ]
    ) * 100.0

    scenario_future_green[
        "green_upper_bound_limited"
    ] = (
        scenario_future_green[
            "actual_green_increase_pp"
        ]
        < pp
        - 1e-10
    )

    scenario_green_rebuilt = build_green_temporal_features(
        historical_green=historical_green,
        future_green=scenario_future_green[
            [
                GRID_COL,
                YEAR_COL,
                "green_mask",
            ]
        ],
    )

    if not scenario_green_rebuilt[
        [
            GRID_COL,
            YEAR_COL,
        ]
    ].equals(
        future[
            [
                GRID_COL,
                YEAR_COL,
            ]
        ]
    ):
        raise RuntimeError(
            f"+{pp} pp rebuilt scenario keys do not align."
        )

    X_scenario = X_base.copy()

    # ---------------------------------------------------------------------
    # 4A. Persistent scenario direct current green
    # ---------------------------------------------------------------------

    scenario_green_array = scenario_green_rebuilt[
        "green_mask"
    ].to_numpy(
        dtype=float
    )

    X_scenario[
        "green_mask"
    ] = scenario_green_array

    # ---------------------------------------------------------------------
    # 4B. Rebuild ALL retained green temporal features
    # ---------------------------------------------------------------------

    for feature in retained_green_temporal:
        X_scenario[
            feature
        ] = scenario_green_rebuilt[
            feature
        ].to_numpy(
            dtype=float
        )

    # ---------------------------------------------------------------------
    # 4C. Rebuild retained current-year green-derived features
    # ---------------------------------------------------------------------

    scenario_vegetation_deficit = (
        1.0
        - scenario_green_array
    )

    if "vegetation_deficit" in X_scenario.columns:
        X_scenario[
            "vegetation_deficit"
        ] = scenario_vegetation_deficit

    if "ndbi_x_vegetation_deficit" in X_scenario.columns:
        X_scenario[
            "ndbi_x_vegetation_deficit"
        ] = (
            X_scenario[
                "NDBI"
            ].to_numpy(
                dtype=float
            )
            * scenario_vegetation_deficit
        )

    if "ndvi_x_green_mask" in X_scenario.columns:
        X_scenario[
            "ndvi_x_green_mask"
        ] = (
            X_scenario[
                "NDVI"
            ].to_numpy(
                dtype=float
            )
            * scenario_green_array
        )

    if "green_mask_nbr_contrast" in X_scenario.columns:
        if "green_mask_nbr_mean" in X_base.columns:
            local_baseline_neighbor_mean = X_base[
                "green_mask_nbr_mean"
            ].to_numpy(
                dtype=float
            )
        else:
            local_baseline_neighbor_mean = (
                baseline_green
                - X_base[
                    "green_mask_nbr_contrast"
                ].to_numpy(
                    dtype=float
                )
            )

        X_scenario[
            "green_mask_nbr_contrast"
        ] = (
            scenario_green_array
            - local_baseline_neighbor_mean
        )

    # ---------------------------------------------------------------------
    # 4D. Strict scenario-matrix checks
    # ---------------------------------------------------------------------

    if X_scenario.isna().any().any():
        raise RuntimeError(
            f"+{pp} pp scenario produced missing values."
        )

    if not np.isfinite(
        X_scenario.to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            f"+{pp} pp scenario produced non-finite values."
        )

    changed_features = []

    for feature in ordered_features:
        max_change = float(
            np.max(
                np.abs(
                    X_scenario[
                        feature
                    ].to_numpy(
                        dtype=float
                    )
                    - X_base[
                        feature
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
        )

        if max_change > UNCHANGED_FEATURE_TOL:
            changed_features.append(
                feature
            )

    unexpected_changed = sorted(
        set(
            changed_features
        )
        - allowed_changed_features
    )

    if unexpected_changed:
        raise RuntimeError(
            f"+{pp} pp scenario changed unexpected features: "
            f"{unexpected_changed}"
        )

    scenario_feature_change_audits.append(
        {
            "scenario_increase_pp":
                pp,
            "changed_feature_count":
                len(
                    changed_features
                ),
            "changed_features":
                "; ".join(
                    sorted(
                        changed_features
                    )
                ),
            "unexpected_changed_feature_count":
                len(
                    unexpected_changed
                ),
            "status":
                "PASS",
        }
    )

    # ---------------------------------------------------------------------
    # 4E. Predict scenario raw anomaly and local LST change
    # ---------------------------------------------------------------------

    scenario_raw = np.asarray(
        model.predict(
            X_scenario
        ),
        dtype=float,
    )

    local_anomaly_delta = (
        scenario_raw
        - baseline_raw_model
    )

    scenario_lst = (
        baseline_predicted_lst
        + local_anomaly_delta
    )

    cooling_benefit = (
        baseline_predicted_lst
        - scenario_lst
    )

    # ---------------------------------------------------------------------
    # 4F. Export scenario model-feature matrix for reproducibility
    # ---------------------------------------------------------------------

    scenario_feature_export = pd.concat(
        [
            future[
                [
                    GRID_COL,
                    YEAR_COL,
                    "area_name",
                ]
            ].reset_index(
                drop=True
            ),
            pd.DataFrame(
                {
                    "scenario_id":
                        f"persistent_green_plus_{pp}pp",
                    "requested_green_increase_pp":
                        pp,
                    "intervention_start_year":
                        INTERVENTION_START_YEAR,
                },
                index=np.arange(
                    EXPECTED_FUTURE_ROWS
                ),
            ),
            X_scenario.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    scenario_feature_exports.append(
        scenario_feature_export
    )

    # ---------------------------------------------------------------------
    # 4G. Main scenario output
    # ---------------------------------------------------------------------

    scenario_green_meta = (
        scenario_future_green
        .sort_values(
            [
                YEAR_COL,
                GRID_COL,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if not scenario_green_meta[
        [
            GRID_COL,
            YEAR_COL,
        ]
    ].equals(
        future[
            [
                GRID_COL,
                YEAR_COL,
            ]
        ]
    ):
        raise RuntimeError(
            f"+{pp} pp scenario metadata order mismatch."
        )

    for i in range(
        EXPECTED_FUTURE_ROWS
    ):
        benefit = float(
            cooling_benefit[
                i
            ]
        )

        if benefit > 1e-12:
            direction = "cooling"
        elif benefit < -1e-12:
            direction = "warming"
        else:
            direction = "no_change"

        scenario_rows.append(
            {
                GRID_COL:
                    str(
                        future.at[
                            i,
                            GRID_COL,
                        ]
                    ),
                YEAR_COL:
                    int(
                        future.at[
                            i,
                            YEAR_COL,
                        ]
                    ),
                "area_name":
                    str(
                        future.at[
                            i,
                            "area_name",
                        ]
                    ),
                "scenario_id":
                    f"persistent_green_plus_{pp}pp",
                "intervention_start_year":
                    INTERVENTION_START_YEAR,
                "requested_green_increase_pp":
                    pp,
                "baseline_green_mask":
                    float(
                        scenario_green_meta.at[
                            i,
                            "baseline_green_mask",
                        ]
                    ),
                "scenario_green_mask":
                    float(
                        scenario_green_meta.at[
                            i,
                            "green_mask",
                        ]
                    ),
                "actual_green_increase_pp":
                    float(
                        scenario_green_meta.at[
                            i,
                            "actual_green_increase_pp",
                        ]
                    ),
                "green_upper_bound_limited":
                    bool(
                        scenario_green_meta.at[
                            i,
                            "green_upper_bound_limited",
                        ]
                    ),
                "baseline_predicted_lst_c":
                    float(
                        baseline_predicted_lst[
                            i
                        ]
                    ),
                "forecast_annual_mean_lst_c":
                    float(
                        annual_level[
                            i
                        ]
                    ),
                "baseline_centered_spatial_anomaly_c":
                    float(
                        baseline_centered_anomaly[
                            i
                        ]
                    ),
                "baseline_raw_spatial_anomaly_c":
                    float(
                        baseline_raw_model[
                            i
                        ]
                    ),
                "scenario_raw_spatial_anomaly_c":
                    float(
                        scenario_raw[
                            i
                        ]
                    ),
                "local_anomaly_delta_c":
                    float(
                        local_anomaly_delta[
                            i
                        ]
                    ),
                "scenario_predicted_lst_c":
                    float(
                        scenario_lst[
                            i
                        ]
                    ),
                "cooling_benefit_c":
                    benefit,
                "scenario_effect_direction":
                    direction,
            }
        )

    # ---------------------------------------------------------------------
    # 4H. Temporal diagnostic by year
    # ---------------------------------------------------------------------

    diag = pd.concat(
        [
            future[
                [
                    GRID_COL,
                    YEAR_COL,
                ]
            ].reset_index(
                drop=True
            ),
            scenario_green_rebuilt.drop(
                columns=[
                    GRID_COL,
                    YEAR_COL,
                ]
            ).reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    for year in FUTURE_YEARS:
        year_diag = diag[
            diag[
                YEAR_COL
            ] == year
        ]

        scenario_temporal_diagnostics.append(
            {
                YEAR_COL:
                    year,
                "requested_green_increase_pp":
                    pp,
                "mean_green_mask":
                    float(
                        year_diag[
                            "green_mask"
                        ].mean()
                    ),
                "mean_green_lag1":
                    float(
                        year_diag[
                            "green_mask_lag1"
                        ].mean()
                    ),
                "mean_green_lag2":
                    float(
                        year_diag[
                            "green_mask_lag2"
                        ].mean()
                    ),
                "mean_green_hist3_mean":
                    float(
                        year_diag[
                            "green_mask_hist3_mean"
                        ].mean()
                    ),
                "mean_green_hist3_std":
                    float(
                        year_diag[
                            "green_mask_hist3_std"
                        ].mean()
                    ),
                "mean_green_hist3_slope":
                    float(
                        year_diag[
                            "green_mask_hist3_slope"
                        ].mean()
                    ),
                "mean_green_delta1":
                    float(
                        year_diag[
                            "green_mask_delta1"
                        ].mean()
                    ),
            }
        )

    print(
        f"  Mean actual green increase : "
        f"{scenario_green_meta['actual_green_increase_pp'].mean():.4f} pp"
    )

    print(
        f"  Upper-bound limited rows   : "
        f"{int(scenario_green_meta['green_upper_bound_limited'].sum()):,}"
    )

    print(
        f"  Mean cooling benefit       : "
        f"{cooling_benefit.mean():.6f} °C"
    )

    print(
        f"  Median cooling benefit     : "
        f"{np.median(cooling_benefit):.6f} °C"
    )

    print(
        f"  Min / Max benefit          : "
        f"{cooling_benefit.min():.6f} / "
        f"{cooling_benefit.max():.6f} °C"
    )


scenario_df = pd.DataFrame(
    scenario_rows
)

scenario_feature_df = pd.concat(
    scenario_feature_exports,
    ignore_index=True,
)

temporal_diagnostic_df = pd.DataFrame(
    scenario_temporal_diagnostics
)

scenario_change_audit_df = pd.DataFrame(
    scenario_feature_change_audits
)

expected_scenario_rows = (
    EXPECTED_FUTURE_ROWS
    * len(
        SCENARIO_INCREASE_PP
    )
)

if len(
    scenario_df
) != expected_scenario_rows:
    raise RuntimeError(
        f"Scenario rows={len(scenario_df)}, "
        f"expected={expected_scenario_rows}"
    )

if len(
    scenario_feature_df
) != expected_scenario_rows:
    raise RuntimeError(
        f"Scenario feature rows={len(scenario_feature_df)}, "
        f"expected={expected_scenario_rows}"
    )

scenario_df.to_csv(
    ALL_SCENARIOS_PATH,
    index=False,
)

scenario_feature_df.to_csv(
    SCENARIO_FEATURE_PATH,
    index=False,
)

primary_df = (
    scenario_df[
        scenario_df[
            "requested_green_increase_pp"
        ]
        == PRIMARY_SCENARIO_PP
    ]
    .copy()
)

primary_df.to_csv(
    PRIMARY_SCENARIO_PATH,
    index=False,
)

temporal_diagnostic_df.to_csv(
    TEMPORAL_DIAGNOSTIC_PATH,
    index=False,
)

for year in FUTURE_YEARS:
    scenario_df[
        scenario_df[
            YEAR_COL
        ]
        == year
    ].to_csv(
        SCENARIO_DIR
        / f"29a_v2_persistent_green_scenarios_{year}.csv",
        index=False,
    )

    for pp in SCENARIO_INCREASE_PP:
        scenario_df[
            (
                scenario_df[
                    YEAR_COL
                ]
                == year
            )
            & (
                scenario_df[
                    "requested_green_increase_pp"
                ]
                == pp
            )
        ].to_csv(
            SCENARIO_DIR
            / f"29a_v2_persistent_green_{pp}pp_{year}.csv",
            index=False,
        )

print(
    f"\nSaved: {ALL_SCENARIOS_PATH}"
)

print(
    f"Saved: {SCENARIO_FEATURE_PATH}"
)

print(
    f"Saved: {TEMPORAL_DIAGNOSTIC_PATH}"
)


# =============================================================================
# 5. SCENARIO SUMMARY
# =============================================================================

section(
    "5. PERSISTENT GREEN SCENARIO SUMMARY"
)

summary = (
    scenario_df
    .groupby(
        [
            YEAR_COL,
            "requested_green_increase_pp",
        ],
        as_index=False,
    )
    .agg(
        grid_count=(
            GRID_COL,
            "size",
        ),
        mean_actual_green_increase_pp=(
            "actual_green_increase_pp",
            "mean",
        ),
        upper_bound_limited_grid_count=(
            "green_upper_bound_limited",
            "sum",
        ),
        mean_baseline_lst_c=(
            "baseline_predicted_lst_c",
            "mean",
        ),
        mean_independent_scenario_lst_c=(
            "scenario_predicted_lst_c",
            "mean",
        ),
        mean_cooling_benefit_c=(
            "cooling_benefit_c",
            "mean",
        ),
        median_cooling_benefit_c=(
            "cooling_benefit_c",
            "median",
        ),
        min_cooling_benefit_c=(
            "cooling_benefit_c",
            "min",
        ),
        max_cooling_benefit_c=(
            "cooling_benefit_c",
            "max",
        ),
        std_cooling_benefit_c=(
            "cooling_benefit_c",
            "std",
        ),
    )
)

direction_counts = (
    scenario_df
    .pivot_table(
        index=[
            YEAR_COL,
            "requested_green_increase_pp",
        ],
        columns=
            "scenario_effect_direction",
        values=
            GRID_COL,
        aggfunc=
            "count",
        fill_value=
            0,
    )
    .reset_index()
)

for column in [
    "cooling",
    "warming",
    "no_change",
]:
    if column not in direction_counts.columns:
        direction_counts[
            column
        ] = 0

direction_counts = direction_counts.rename(
    columns={
        "cooling":
            "cooling_grid_count",
        "warming":
            "warming_grid_count",
        "no_change":
            "no_change_grid_count",
    }
)

summary = summary.merge(
    direction_counts,
    on=[
        YEAR_COL,
        "requested_green_increase_pp",
    ],
    how="left",
    validate="one_to_one",
)

summary[
    "cooling_grid_pct"
] = (
    summary[
        "cooling_grid_count"
    ]
    / summary[
        "grid_count"
    ]
    * 100.0
)

summary[
    "warming_grid_pct"
] = (
    summary[
        "warming_grid_count"
    ]
    / summary[
        "grid_count"
    ]
    * 100.0
)

summary[
    "no_change_grid_pct"
] = (
    summary[
        "no_change_grid_count"
    ]
    / summary[
        "grid_count"
    ]
    * 100.0
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nIMPORTANT:"
)

print(
    "These are summaries of independently evaluated local persistent "
    "interventions, not a simultaneous citywide scenario."
)


# =============================================================================
# 6. MONOTONICITY DIAGNOSTIC
# =============================================================================

section(
    "6. GREEN-INCREASE RESPONSE MONOTONICITY DIAGNOSTIC"
)

benefit_wide = (
    scenario_df
    .pivot(
        index=[
            GRID_COL,
            YEAR_COL,
        ],
        columns=
            "requested_green_increase_pp",
        values=
            "cooling_benefit_c",
    )
    .reset_index()
)

b5 = benefit_wide[
    5
].to_numpy(
    dtype=float
)

b10 = benefit_wide[
    10
].to_numpy(
    dtype=float
)

b20 = benefit_wide[
    20
].to_numpy(
    dtype=float
)

monotonic_non_decreasing = (
    (
        b10
        >= b5
        - 1e-12
    )
    & (
        b20
        >= b10
        - 1e-12
    )
)

benefit_wide[
    "cooling_benefit_5pp_c"
] = b5

benefit_wide[
    "cooling_benefit_10pp_c"
] = b10

benefit_wide[
    "cooling_benefit_20pp_c"
] = b20

benefit_wide[
    "monotonic_non_decreasing"
] = monotonic_non_decreasing

benefit_wide.to_csv(
    MONOTONIC_PATH,
    index=False,
)

monotonic_pct = float(
    monotonic_non_decreasing.mean()
    * 100.0
)

print(
    f"Grid-year cases with non-decreasing modeled cooling "
    f"from +5 -> +10 -> +20 pp: {monotonic_pct:.2f}%"
)

print(
    "No monotonic response is forced; genuine tree-model behavior is retained."
)


# =============================================================================
# 7. FRONTEND-READY GEOJSON + MAPS
# =============================================================================

section(
    "7. SAVE FRONTEND-READY PERSISTENT-SCENARIO GEOJSON + MAPS"
)

grid = gpd.read_file(
    GRID_WGS84_PATH
)

grid[
    GRID_COL
] = grid[
    GRID_COL
].astype(
    str
)

if grid[
    GRID_COL
].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        "WGS84 grid geometry count mismatch."
    )

grid = grid[
    [
        GRID_COL,
        "geometry",
    ]
].copy()

boundary = load_boundary_wgs84()

geojson_files = []
map_files = []

for year in FUTURE_YEARS:
    year_base = (
        pred[
            pred[
                YEAR_COL
            ]
            == year
        ][
            [
                GRID_COL,
                YEAR_COL,
                "area_name",
                "predicted_lst_c",
                "forecast_annual_mean_lst_c",
                "centered_spatial_anomaly_c",
            ]
        ]
        .copy()
        .rename(
            columns={
                "predicted_lst_c":
                    "baseline_predicted_lst_c",
            }
        )
    )

    year_merge = year_base.copy()

    for pp in SCENARIO_INCREASE_PP:
        subset = (
            scenario_df[
                (
                    scenario_df[
                        YEAR_COL
                    ]
                    == year
                )
                & (
                    scenario_df[
                        "requested_green_increase_pp"
                    ]
                    == pp
                )
            ][
                [
                    GRID_COL,
                    "scenario_green_mask",
                    "actual_green_increase_pp",
                    "scenario_predicted_lst_c",
                    "cooling_benefit_c",
                    "scenario_effect_direction",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "scenario_green_mask":
                        f"green_mask_{pp}pp",
                    "actual_green_increase_pp":
                        f"actual_green_increase_{pp}pp",
                    "scenario_predicted_lst_c":
                        f"scenario_lst_{pp}pp_c",
                    "cooling_benefit_c":
                        f"cooling_benefit_{pp}pp_c",
                    "scenario_effect_direction":
                        f"effect_{pp}pp",
                }
            )
        )

        year_merge = year_merge.merge(
            subset,
            on=GRID_COL,
            how="left",
            validate="one_to_one",
        )

    year_gdf = grid.merge(
        year_merge,
        on=GRID_COL,
        how="left",
        validate="one_to_one",
    )

    if year_gdf[
        "baseline_predicted_lst_c"
    ].isna().any():
        raise RuntimeError(
            f"{year}: GeoJSON merge produced missing baseline rows."
        )

    geo_path = (
        MAP_DIR
        / f"29a_v2_persistent_green_scenarios_{year}_wgs84.geojson"
    )

    year_gdf.to_file(
        geo_path,
        driver="GeoJSON",
    )

    geojson_files.append(
        geo_path
    )

    for pp in SCENARIO_INCREASE_PP:
        column = (
            f"cooling_benefit_{pp}pp_c"
        )

        fig, ax = plt.subplots(
            figsize=(
                10,
                10,
            )
        )

        values = year_gdf[
            column
        ].to_numpy(
            dtype=float
        )

        max_abs = float(
            np.nanmax(
                np.abs(
                    values
                )
            )
        )

        if max_abs == 0:
            max_abs = 1e-6

        year_gdf.plot(
            column=column,
            ax=ax,
            legend=True,
            edgecolor="none",
            cmap="RdBu",
            vmin=-max_abs,
            vmax=max_abs,
            legend_kwds={
                "label":
                    "Modeled local cooling benefit (°C)\n"
                    "(positive = cooling, negative = warming)",
                "shrink":
                    0.72,
            },
        )

        if boundary is not None:
            boundary.boundary.plot(
                ax=ax,
                linewidth=1.4,
                color="black",
            )
        else:
            year_gdf.dissolve().boundary.plot(
                ax=ax,
                linewidth=1.4,
                color="black",
            )

        ax.set_title(
            (
                f"Kaduwela Persistent Green-Cover Scenario — {year}\n"
                f"+{pp} percentage points from 2026 onward"
            ),
            fontsize=14,
            pad=12,
        )

        ax.set_axis_off()

        plt.tight_layout()

        map_path = (
            FIGURE_DIR
            / (
                f"29a_v2_persistent_green_cooling_benefit_"
                f"{year}_{pp}pp.png"
            )
        )

        plt.savefig(
            map_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        map_files.append(
            map_path
        )

    print(
        f"{year}: saved GeoJSON + "
        f"{len(SCENARIO_INCREASE_PP)} persistent-scenario maps"
    )


# =============================================================================
# 8. SUMMARY FIGURE
# =============================================================================

section(
    "8. SAVE PERSISTENT-SCENARIO SUMMARY FIGURE"
)

plot_summary = (
    summary.groupby(
        "requested_green_increase_pp",
        as_index=False,
    )[
        "mean_cooling_benefit_c"
    ]
    .mean()
)

fig, ax = plt.subplots(
    figsize=(
        8,
        6,
    )
)

ax.plot(
    plot_summary[
        "requested_green_increase_pp"
    ],
    plot_summary[
        "mean_cooling_benefit_c"
    ],
    marker="o",
)

ax.axhline(
    0.0,
    linewidth=1.0,
    color="black",
)

ax.set_xlabel(
    "Persistent green-cover increase (percentage points)"
)

ax.set_ylabel(
    "Mean modeled local cooling benefit (°C)"
)

ax.set_title(
    "Persistent Green-Cover Scenario Sensitivity — 2026–2030"
)

ax.grid(
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    SUMMARY_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    f"Saved: {SUMMARY_FIGURE_PATH}"
)


# =============================================================================
# 9. STRICT STEP-29A V2 AUDIT
# =============================================================================

section(
    "9. STRICT STEP-29A V2 AUDIT"
)

all_scenario_green = scenario_df[
    "scenario_green_mask"
].to_numpy(
    dtype=float
)

all_benefit = scenario_df[
    "cooling_benefit_c"
].to_numpy(
    dtype=float
)

all_local_delta = scenario_df[
    "local_anomaly_delta_c"
].to_numpy(
    dtype=float
)

all_scenario_lst = scenario_df[
    "scenario_predicted_lst_c"
].to_numpy(
    dtype=float
)

scenario_identity_error = float(
    np.max(
        np.abs(
            all_benefit
            + all_local_delta
        )
    )
)

audit_rows = [
    {
        "check":
            "step28_audit",
        "status":
            "PASS",
        "details":
            "Step-28 strict audit verified fully PASS.",
    },
    {
        "check":
            "historical_base_hash",
        "status":
            "PASS",
        "details":
            EXPECTED_HASHES[
                "kaduwela_preprocessed_base.csv"
            ],
    },
    {
        "check":
            "production_model_hash",
        "status":
            "PASS",
        "details":
            EXPECTED_HASHES[
                "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
            ],
    },
    {
        "check":
            "future_feature_hash",
        "status":
            "PASS",
        "details":
            EXPECTED_HASHES[
                "28_future_model_features_2026_2030.csv"
            ],
    },
    {
        "check":
            "future_prediction_hash",
        "status":
            "PASS",
        "details":
            EXPECTED_HASHES[
                "28_future_grid_lst_predictions_2026_2030.csv"
            ],
    },
    {
        "check":
            "baseline_green_feature_reconstruction",
        "status":
            "PASS"
            if (
                baseline_recon_audit[
                    "status"
                ]
                == "PASS"
            ).all()
            else "FAIL",
        "details":
            f"features_checked={len(baseline_recon_audit)}",
    },
    {
        "check":
            "baseline_raw_prediction_reproduction",
        "status":
            "PASS"
            if baseline_reproduction_error
            <= PREDICTION_REPRO_TOL
            else "FAIL",
        "details":
            f"max_error_c={baseline_reproduction_error:.12e}",
    },
    {
        "check":
            "persistent_intervention_start_year",
        "status":
            "PASS"
            if INTERVENTION_START_YEAR
            == 2026
            else "FAIL",
        "details":
            str(
                INTERVENTION_START_YEAR
            ),
    },
    {
        "check":
            "scenario_levels",
        "status":
            "PASS"
            if SCENARIO_INCREASE_PP
            == [
                5,
                10,
                20,
            ]
            else "FAIL",
        "details":
            str(
                SCENARIO_INCREASE_PP
            ),
    },
    {
        "check":
            "scenario_row_count",
        "status":
            "PASS"
            if len(
                scenario_df
            )
            == expected_scenario_rows
            else "FAIL",
        "details":
            (
                f"actual={len(scenario_df)}, "
                f"expected={expected_scenario_rows}"
            ),
    },
    {
        "check":
            "scenario_feature_row_count",
        "status":
            "PASS"
            if len(
                scenario_feature_df
            )
            == expected_scenario_rows
            else "FAIL",
        "details":
            (
                f"actual={len(scenario_feature_df)}, "
                f"expected={expected_scenario_rows}"
            ),
    },
    {
        "check":
            "scenario_green_bounds",
        "status":
            "PASS"
            if (
                np.min(
                    all_scenario_green
                )
                >= GREEN_MIN
                - 1e-12
                and np.max(
                    all_scenario_green
                )
                <= GREEN_MAX
                + 1e-12
            )
            else "FAIL",
        "details":
            (
                f"min={np.min(all_scenario_green):.6f}, "
                f"max={np.max(all_scenario_green):.6f}"
            ),
    },
    {
        "check":
            "future_green_history_propagated",
        "status":
            "PASS",
        "details":
            (
                "Scenario 2026 values become 2027 lag/history; "
                "scenario history continues recursively through 2030."
            ),
    },
    {
        "check":
            "only_intended_frozen_features_changed",
        "status":
            "PASS"
            if (
                scenario_change_audit_df[
                    "unexpected_changed_feature_count"
                ]
                == 0
            ).all()
            else "FAIL",
        "details":
            "; ".join(
                scenario_change_audit_df[
                    "changed_features"
                ].astype(
                    str
                ).tolist()
            ),
    },
    {
        "check":
            "scenario_outputs_finite",
        "status":
            "PASS"
            if (
                np.isfinite(
                    all_scenario_lst
                ).all()
                and np.isfinite(
                    all_benefit
                ).all()
            )
            else "FAIL",
        "details":
            f"rows={len(scenario_df)}",
    },
    {
        "check":
            "cooling_benefit_identity",
        "status":
            "PASS"
            if scenario_identity_error
            <= PREDICTION_REPRO_TOL
            else "FAIL",
        "details":
            f"max_error_c={scenario_identity_error:.12e}",
    },
    {
        "check":
            "local_neighbor_semantics",
        "status":
            "PASS",
        "details":
            (
                "Neighbour mean remains Step-28 baseline; "
                "only selected-grid contrast changes."
            ),
    },
    {
        "check":
            "scenario_recentering",
        "status":
            "PASS",
        "details":
            (
                "Not applied. Local scenario uses raw-anomaly delta "
                "for the same grid-year."
            ),
    },
    {
        "check":
            "annual_thermal_level_changed",
        "status":
            "PASS",
        "details":
            "No. Frozen Step-28 annual level retained.",
    },
    {
        "check":
            "model_retraining",
        "status":
            "PASS",
        "details":
            "None. Frozen Step-22B model used for inference only.",
    },
    {
        "check":
            "step28_baseline_modified",
        "status":
            "PASS",
        "details":
            "No. V2 scenario outputs saved separately.",
    },
    {
        "check":
            "cooling_or_monotonicity_forced",
        "status":
            "PASS",
        "details":
            "No. Genuine model responses retained.",
    },
    {
        "check":
            "causal_claim",
        "status":
            "PASS",
        "details":
            "No. Scenario is explicitly model-based statistical sensitivity.",
    },
    {
        "check":
            "uncertainty_added_here",
        "status":
            "PASS",
        "details":
            "No. Uncertainty/confidence remains Step 30.",
    },
    {
        "check":
            "yearly_geojson_count",
        "status":
            "PASS"
            if len(
                geojson_files
            )
            == len(
                FUTURE_YEARS
            )
            else "FAIL",
        "details":
            (
                f"saved={len(geojson_files)}, "
                f"expected={len(FUTURE_YEARS)}"
            ),
    },
    {
        "check":
            "scenario_map_count",
        "status":
            "PASS"
            if len(
                map_files
            )
            == len(
                FUTURE_YEARS
            )
            * len(
                SCENARIO_INCREASE_PP
            )
            else "FAIL",
        "details":
            (
                f"saved={len(map_files)}, "
                f"expected={len(FUTURE_YEARS) * len(SCENARIO_INCREASE_PP)}"
            ),
    },
]

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
)

print(
    audit_df.to_string(
        index=False
    )
)

if (
    audit_df[
        "status"
    ]
    != "PASS"
).any():
    raise RuntimeError(
        "One or more Step-29A V2 audit checks failed."
    )


# =============================================================================
# 10. METHODOLOGY + REPORT
# =============================================================================

section(
    "10. SAVE STEP-29A V2 METHODOLOGY + REPORT"
)

methodology = {
    "step":
        "29A V2",
    "version":
        "persistent_local_green_cover_intervention_v2",
    "purpose":
        "Temporally coherent persistent local green-cover what-if simulation",
    "intervention_start_year":
        INTERVENTION_START_YEAR,
    "future_years":
        FUTURE_YEARS,
    "scenario_levels_percentage_points":
        SCENARIO_INCREASE_PP,
    "primary_scenario_percentage_points":
        PRIMARY_SCENARIO_PP,
    "scenario_scope":
        (
            "Independent local grid-cell intervention beginning in 2026 "
            "and maintained through 2030."
        ),
    "scenario_green_formula":
        (
            "scenario green_mask(t) = clip("
            "Step-28 baseline green_mask(t) + scenario increase, 0, 1)"
        ),
    "temporal_reconstruction":
        {
            "lag1":
                "scenario/history value at t-1",
            "lag2":
                "scenario/history value at t-2",
            "hist3_mean":
                "mean of t-1,t-2,t-3 only",
            "hist3_std":
                "sample standard deviation of t-1,t-2,t-3 only, ddof=1",
            "hist3_slope":
                "OLS slope over t-3,t-2,t-1 = (most_recent-oldest)/2",
            "delta1":
                "current scenario green_mask - lag1",
        },
    "current_year_dependent_features_rebuilt":
        [
            feature
            for feature in [
                "vegetation_deficit",
                "ndbi_x_vegetation_deficit",
                "ndvi_x_green_mask",
                "green_mask_nbr_contrast",
            ]
            if feature in X_base.columns
        ],
    "retained_green_temporal_features_rebuilt":
        retained_green_temporal,
    "local_neighbor_policy":
        (
            "Neighbouring grids remain at Step-28 baseline forecasts. "
            "green_mask_nbr_mean stays baseline; selected-grid "
            "green_mask_nbr_contrast is updated."
        ),
    "model":
        str(
            MODEL_PATH
        ),
    "model_retraining":
        False,
    "step28_baseline_modified":
        False,
    "scenario_recentering":
        False,
    "local_effect_formula":
        (
            "scenario raw anomaly - baseline raw anomaly"
        ),
    "scenario_lst_formula":
        (
            "Step-28 baseline predicted LST + local anomaly delta"
        ),
    "cooling_benefit_formula":
        (
            "Step-28 baseline predicted LST - scenario predicted LST"
        ),
    "uncertainty":
        "Deferred to Step 30.",
    "interpretation_warning":
        (
            "Model-based statistical sensitivity only; not causal proof "
            "or guaranteed physical cooling."
        ),
    "v1_relationship":
        (
            "V1 is retained as a direct same-year perturbation diagnostic. "
            "V2 is the preferred planning scenario because intervention "
            "history is propagated from 2026 through 2030."
        ),
}

METHOD_PATH.write_text(
    json.dumps(
        methodology,
        indent=2,
    ),
    encoding="utf-8",
)

overall_summary = (
    scenario_df
    .groupby(
        "requested_green_increase_pp",
        as_index=False,
    )
    .agg(
        rows=(
            GRID_COL,
            "size",
        ),
        mean_actual_green_increase_pp=(
            "actual_green_increase_pp",
            "mean",
        ),
        mean_cooling_benefit_c=(
            "cooling_benefit_c",
            "mean",
        ),
        median_cooling_benefit_c=(
            "cooling_benefit_c",
            "median",
        ),
        min_cooling_benefit_c=(
            "cooling_benefit_c",
            "min",
        ),
        max_cooling_benefit_c=(
            "cooling_benefit_c",
            "max",
        ),
    )
)

report_lines = [
    "STEP 29A V2 - PERSISTENT GREEN-COVER WHAT-IF COOLING SCENARIO",
    "=" * 118,
    "",
    "WHY V2",
    "-" * 118,
    (
        "V2 applies the green-cover intervention from 2026 onward and "
        "propagates scenario values into future lag/history features."
    ),
    (
        "This is temporally more coherent than the V1 direct same-year "
        "perturbation diagnostic."
    ),
    "",
    "SCENARIO DESIGN",
    "-" * 118,
    (
        "Primary +5 percentage points; sensitivity +10 and +20 percentage "
        "points; values clipped to [0,1]."
    ),
    (
        "Each grid is an independent local intervention. Neighbouring grids "
        "remain at their Step-28 baseline future state."
    ),
    "",
    "OVERALL RESULT",
    "-" * 118,
    overall_summary.to_string(
        index=False
    ),
    "",
    "YEAR-BY-YEAR RESULT",
    "-" * 118,
    summary.to_string(
        index=False
    ),
    "",
    "GREEN TEMPORAL DIAGNOSTIC",
    "-" * 118,
    temporal_diagnostic_df.to_string(
        index=False
    ),
    "",
    "MONOTONICITY DIAGNOSTIC",
    "-" * 118,
    (
        f"{monotonic_pct:.2f}% of grid-year cases show non-decreasing "
        "modeled cooling from +5 to +10 to +20 pp."
    ),
    (
        "No monotonicity is forced or corrected."
    ),
    "",
    "INTERPRETATION",
    "-" * 118,
    (
        "Positive cooling_benefit_c means the persistent scenario is cooler "
        "than the same grid-year Step-28 baseline."
    ),
    (
        "Negative cooling_benefit_c means the predictive model associates "
        "the scenario with warming for that grid-year."
    ),
    (
        "Results are statistical model sensitivity, not causal or guaranteed "
        "real-world cooling."
    ),
    (
        "If persistent scenarios still overwhelmingly predict warming, "
        "the frozen predictive LightGBM should not be presented as a "
        "green-intervention cooling estimator."
    ),
]

REPORT_PATH.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)

print(
    f"Saved: {METHOD_PATH}"
)

print(
    f"Saved: {REPORT_PATH}"
)


# =============================================================================
# 11. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "11. REPRODUCIBILITY HASHES"
)

hash_files = [
    BASE_PATH,
    FEATURE_LIST_PATH,
    MODEL_PATH,
    FUTURE_FEATURE_PATH,
    FUTURE_PRED_PATH,
    STEP28_AUDIT_PATH,
    GRID_WGS84_PATH,
    ALL_SCENARIOS_PATH,
    PRIMARY_SCENARIO_PATH,
    SCENARIO_FEATURE_PATH,
    SUMMARY_PATH,
    TEMPORAL_DIAGNOSTIC_PATH,
    MONOTONIC_PATH,
    BASELINE_RECON_PATH,
    AUDIT_PATH,
    METHOD_PATH,
    REPORT_PATH,
    SUMMARY_FIGURE_PATH,
]

for year in FUTURE_YEARS:
    hash_files.append(
        SCENARIO_DIR
        / f"29a_v2_persistent_green_scenarios_{year}.csv"
    )

    for pp in SCENARIO_INCREASE_PP:
        hash_files.append(
            SCENARIO_DIR
            / f"29a_v2_persistent_green_{pp}pp_{year}.csv"
        )

    hash_files.append(
        MAP_DIR
        / f"29a_v2_persistent_green_scenarios_{year}_wgs84.geojson"
    )

for path in map_files:
    hash_files.append(
        path
    )

unique_hash_files = []
seen = set()

for path in hash_files:
    path = Path(
        path
    )

    key = str(
        path
    )

    if key not in seen:
        seen.add(
            key
        )

        unique_hash_files.append(
            path
        )

hash_rows = []

for path in unique_hash_files:
    digest = sha256_file(
        path
    )

    hash_rows.append(
        {
            "file":
                str(
                    path
                ),
            "sha256":
                digest,
        }
    )

    print(
        f"{path.name}: {digest}"
    )

pd.DataFrame(
    hash_rows
).to_csv(
    HASH_PATH,
    index=False,
)

print(
    f"Saved: {HASH_PATH}"
)


# =============================================================================
# FINAL
# =============================================================================

section(
    "STEP 29A V2 COMPLETED SUCCESSFULLY"
)

print(
    "Persistent green-cover what-if scenario simulation completed.\n"
    f"Future baseline grid-years: {EXPECTED_FUTURE_ROWS:,}.\n"
    f"Scenario levels: {SCENARIO_INCREASE_PP} percentage points.\n"
    f"Primary scenario: +{PRIMARY_SCENARIO_PP} percentage points.\n"
    f"Intervention start: {INTERVENTION_START_YEAR} and maintained through 2030.\n"
    f"Total independent scenario rows: {len(scenario_df):,}.\n"
    "Historical 2023-2025 green cover was used to seed future temporal history.\n"
    "Step-28 baseline green temporal features were reconstructed before scenario use.\n"
    "Scenario 2026 values propagate into 2027+ lag/history features.\n"
    "The frozen Step-22B LightGBM was used without retraining.\n"
    "Step-28 forecasts were not modified.\n"
    "Scenario effect uses local raw-anomaly delta without batch recentering.\n"
    "No cooling or monotonic behavior was forced.\n"
    "V1 remains only as a direct-perturbation diagnostic.\n"
    "V2 is the preferred temporally coherent green-planning scenario candidate.\n"
    "Uncertainty remains Step 30.\n"
    "\nNEXT: review V2 results carefully before freezing 29A or moving to Step 29B."
)
