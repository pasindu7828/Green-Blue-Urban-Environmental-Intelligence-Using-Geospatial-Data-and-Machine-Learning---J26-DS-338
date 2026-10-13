"""
STEP 28B - FUTURE 2026-2030 SHAP EXPLAINABILITY
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Explain the already-frozen Step-28 future spatial heat predictions for
2026-2030 using the exact Step-22B production LightGBM and the exact frozen
93-feature future matrices.

IMPORTANT INTERPRETATION
------------------------
The future system is decomposed as:

    Final Future Grid LST
        = Forecast Kaduwela Annual Thermal Level
        + Centered Spatial-Anomaly Prediction

SHAP in Step 28B explains ONLY the frozen LightGBM spatial-anomaly component.

Therefore:
- positive centered SHAP -> pushes a grid hotter RELATIVE TO the Kaduwela
  annual thermal level for that future year;
- negative centered SHAP -> pushes a grid cooler RELATIVE TO the Kaduwela
  annual thermal level for that future year;
- SHAP does NOT explain why the citywide annual thermal-level forecast itself
  is 35.511... °C;
- SHAP is a model explanation, not proof of causation.

CONSISTENCY WITH STEP 24
------------------------
Step 24 defined centered SHAP as:

    centered_SHAP(feature, grid, year)
      = raw_SHAP(feature, grid, year)
        - mean raw_SHAP(feature, all grids in same year)

Identity:

    sum(centered_SHAP across 93 features)
      = centered LightGBM spatial-anomaly prediction

Step 28B uses the same definition for 2026-2030.
"""

from pathlib import Path
import hashlib
import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "\nshap is required for Step 28B.\n"
        "Install it with:\n"
        "    pip install shap\n"
    ) from exc

try:
    import geopandas as gpd
except ImportError as exc:
    raise ImportError(
        "\ngeopandas is required for Step 28B spatial outputs.\n"
        "Install it with:\n"
        "    pip install geopandas\n"
    ) from exc


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"

EXPECTED_GRIDS = 2194
FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]
EXPECTED_ROWS = EXPECTED_GRIDS * len(FUTURE_YEARS)
EXPECTED_FEATURES = 93

LOCAL_TOP_K = 10
MAP_TOP_FEATURE_COUNT = 3

RAW_RECON_TOL = 1e-8
CENTERED_RECON_TOL = 1e-8
PREDICTION_REPRO_TOL = 1e-10
CENTERING_TOL = 1e-10

EXPECTED_HASHES = {
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

FEATURE_LIST_PATH = (
    ROOT / "outputs" / "reports" / "21_final_feature_list.csv"
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

EXPLAIN_DIR = ROOT / "outputs" / "explainability"
MAP_DIR = ROOT / "outputs" / "maps"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [
    EXPLAIN_DIR,
    MAP_DIR,
    FIGURE_DIR,
    REPORT_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

RAW_SHAP_PATH = (
    EXPLAIN_DIR / "28b_future_raw_shap_values_2026_2030.csv"
)

CENTERED_SHAP_PATH = (
    EXPLAIN_DIR / "28b_future_centered_shap_values_2026_2030.csv"
)

GLOBAL_IMPORTANCE_PATH = (
    EXPLAIN_DIR / "28b_future_global_shap_importance_2026_2030.csv"
)

YEARLY_IMPORTANCE_PATH = (
    EXPLAIN_DIR / "28b_future_yearly_shap_importance.csv"
)

LOCAL_TOP10_PATH = (
    EXPLAIN_DIR / "28b_future_local_top10_shap_2026_2030.csv"
)

LOCAL_SUMMARY_PATH = (
    EXPLAIN_DIR / "28b_future_local_explanation_summary_2026_2030.csv"
)

RECON_PATH = (
    EXPLAIN_DIR / "28b_future_shap_reconstruction_by_year.csv"
)

GLOBAL_BAR_PATH = (
    FIGURE_DIR / "28b_future_global_shap_importance_bar.png"
)

AUDIT_PATH = (
    REPORT_DIR / "28b_future_shap_audit.csv"
)

METHOD_PATH = (
    REPORT_DIR / "28b_future_shap_methodology.json"
)

REPORT_PATH = (
    REPORT_DIR / "28b_future_shap_report.txt"
)

HASH_PATH = (
    REPORT_DIR / "28b_future_shap_hashes.csv"
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


def normalize_shap_array(values):
    """
    Handle SHAP-version differences for a regression model.
    """
    if isinstance(values, list):
        if len(values) != 1:
            raise RuntimeError(
                "Unexpected multi-output SHAP list for regression model."
            )

        values = values[0]

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim == 3:
        if values.shape[-1] != 1:
            raise RuntimeError(
                f"Unexpected 3-D SHAP shape: {values.shape}"
            )

        values = values[:, :, 0]

    if values.ndim != 2:
        raise RuntimeError(
            f"Unexpected SHAP array shape: {values.shape}"
        )

    return values


def scalar_expected_value(explainer):
    value = np.asarray(
        explainer.expected_value,
        dtype=float,
    ).reshape(-1)

    if len(value) != 1:
        raise RuntimeError(
            f"Unexpected SHAP expected-value shape: {value.shape}"
        )

    return float(value[0])


def safe_name(text):
    return re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        str(text),
    ).strip("_")


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
            "WARNING: could not load Kaduwela KML boundary."
        )

        print(
            f"  {type(exc).__name__}: {exc}"
        )

        return None


# =============================================================================
# 0. VERIFY FROZEN INPUTS
# =============================================================================

section(
    "STEP 28B - FUTURE 2026-2030 SHAP EXPLAINABILITY"
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


# =============================================================================
# 1. LOAD FUTURE FEATURES + PREDICTIONS
# =============================================================================

section(
    "1. LOAD FUTURE FEATURES + FROZEN PREDICTIONS"
)

feature_df = pd.read_csv(
    FEATURE_LIST_PATH
)

feature_col = detect_feature_column(
    feature_df
)

ordered_features = (
    feature_df[feature_col]
    .dropna()
    .astype(str)
    .tolist()
)

if len(ordered_features) != EXPECTED_FEATURES:
    raise RuntimeError(
        f"Feature count={len(ordered_features)}, "
        f"expected={EXPECTED_FEATURES}"
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

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(
            f"{label}: rows={len(frame)}, "
            f"expected={EXPECTED_ROWS}"
        )

    if sorted(
        frame[YEAR_COL]
        .unique()
        .tolist()
    ) != FUTURE_YEARS:
        raise RuntimeError(
            f"{label}: unexpected future years."
        )

    if frame.duplicated(
        [GRID_COL, YEAR_COL]
    ).any():
        raise RuntimeError(
            f"{label}: duplicate grid-year rows."
        )

missing_features = [
    feature
    for feature in ordered_features
    if feature not in future.columns
]

if missing_features:
    raise RuntimeError(
        "Future feature matrix is missing frozen features:\n"
        + str(missing_features)
    )

future = (
    future
    .sort_values(
        [YEAR_COL, GRID_COL]
    )
    .reset_index(drop=True)
)

pred = (
    pred
    .sort_values(
        [YEAR_COL, GRID_COL]
    )
    .reset_index(drop=True)
)

if not (
    future[
        [GRID_COL, YEAR_COL]
    ].equals(
        pred[
            [GRID_COL, YEAR_COL]
        ]
    )
):
    raise RuntimeError(
        "Future feature and prediction grid-year keys do not align."
    )

X = future[
    ordered_features
].copy()

for column in ordered_features:
    X[column] = pd.to_numeric(
        X[column],
        errors="raise",
    )

if X.isna().any().any():
    raise RuntimeError(
        "Frozen future feature matrix contains missing values."
    )

if not np.isfinite(
    X.to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Frozen future feature matrix contains non-finite values."
    )

print(
    f"Future rows      : {len(future):,}"
)

print(
    f"Unique grids     : {future[GRID_COL].nunique():,}"
)

print(
    f"Future years     : {FUTURE_YEARS}"
)

print(
    f"Frozen features  : {len(ordered_features)}"
)


# =============================================================================
# 2. REPRODUCE STEP-28 LIGHTGBM PREDICTIONS
# =============================================================================

section(
    "2. REPRODUCE STEP-28 LIGHTGBM SPATIAL-ANOMALY PREDICTIONS"
)

model = joblib.load(
    MODEL_PATH
)

model_raw_prediction = np.asarray(
    model.predict(X),
    dtype=float,
)

saved_raw_prediction = pd.to_numeric(
    pred[
        "raw_spatial_anomaly_prediction_c"
    ],
    errors="raise",
).to_numpy(dtype=float)

raw_prediction_reproduction_error = float(
    np.max(
        np.abs(
            model_raw_prediction
            - saved_raw_prediction
        )
    )
)

print(
    "Max model prediction reproduction error:"
)

print(
    f"  {raw_prediction_reproduction_error:.12e} °C"
)

if (
    raw_prediction_reproduction_error
    > PREDICTION_REPRO_TOL
):
    raise RuntimeError(
        "Step-28 raw LightGBM predictions could not be reproduced."
    )

print(
    "PASS: Step-28 raw spatial-anomaly predictions reproduced."
)


# =============================================================================
# 3. TREE SHAP - RAW CONTRIBUTIONS
# =============================================================================

section(
    "3. TREE SHAP - RAW FUTURE SPATIAL-ANOMALY CONTRIBUTIONS"
)

explainer = shap.TreeExplainer(
    model
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    raw_shap = explainer.shap_values(
        X,
        check_additivity=False,
    )

raw_shap = normalize_shap_array(
    raw_shap
)

if raw_shap.shape != (
    EXPECTED_ROWS,
    EXPECTED_FEATURES,
):
    raise RuntimeError(
        f"SHAP shape={raw_shap.shape}, "
        f"expected={(EXPECTED_ROWS, EXPECTED_FEATURES)}"
    )

base_value = scalar_expected_value(
    explainer
)

raw_reconstructed = (
    base_value
    + raw_shap.sum(axis=1)
)

raw_reconstruction_error = float(
    np.max(
        np.abs(
            raw_reconstructed
            - model_raw_prediction
        )
    )
)

print(
    f"TreeSHAP expected/base value : {base_value:.12f} °C"
)

print(
    f"Max raw SHAP reconstruction : {raw_reconstruction_error:.12e} °C"
)

if raw_reconstruction_error > RAW_RECON_TOL:
    raise RuntimeError(
        "Raw SHAP values do not reconstruct the LightGBM output."
    )

print(
    "PASS: raw TreeSHAP reconstructs the future LightGBM output."
)


# =============================================================================
# 4. CENTER SHAP WITHIN EACH FUTURE YEAR
# =============================================================================

section(
    "4. YEAR-CENTER SHAP TO MATCH STEP-28 SPATIAL-ANOMALY INFERENCE"
)

centered_shap = np.empty_like(
    raw_shap,
    dtype=float,
)

recon_rows = []

saved_centered_prediction = pd.to_numeric(
    pred[
        "centered_spatial_anomaly_c"
    ],
    errors="raise",
).to_numpy(dtype=float)

for year in FUTURE_YEARS:
    mask = (
        future[YEAR_COL]
        .to_numpy()
        == year
    )

    year_raw_shap = raw_shap[
        mask
    ]

    year_centered_shap = (
        year_raw_shap
        - year_raw_shap.mean(
            axis=0,
            keepdims=True,
        )
    )

    centered_shap[
        mask
    ] = year_centered_shap

    centered_reconstructed = (
        year_centered_shap.sum(
            axis=1
        )
    )

    target_centered = (
        saved_centered_prediction[
            mask
        ]
    )

    reconstruction_error = float(
        np.max(
            np.abs(
                centered_reconstructed
                - target_centered
            )
        )
    )

    recon_rows.append(
        {
            YEAR_COL: year,
            "rows": int(mask.sum()),
            "raw_prediction_mean_c": float(
                model_raw_prediction[
                    mask
                ].mean()
            ),
            "centered_prediction_mean_c": float(
                target_centered.mean()
            ),
            "centered_shap_sum_mean_c": float(
                centered_reconstructed.mean()
            ),
            "max_raw_shap_feature_mean_after_centering_c": float(
                np.max(
                    np.abs(
                        year_centered_shap.mean(
                            axis=0
                        )
                    )
                )
            ),
            "max_centered_reconstruction_error_c": reconstruction_error,
        }
    )

recon_df = pd.DataFrame(
    recon_rows
)

recon_df.to_csv(
    RECON_PATH,
    index=False,
)

print(
    recon_df.to_string(
        index=False
    )
)

max_centered_reconstruction_error = float(
    recon_df[
        "max_centered_reconstruction_error_c"
    ].max()
)

if (
    max_centered_reconstruction_error
    > CENTERED_RECON_TOL
):
    raise RuntimeError(
        "Centered SHAP does not reconstruct the Step-28 centered anomaly."
    )

print(
    f"\nPASS: centered SHAP reconstructs future spatial anomaly "
    f"(max error={max_centered_reconstruction_error:.12e} °C)."
)


# =============================================================================
# 5. SAVE FULL RAW + CENTERED SHAP MATRICES
# =============================================================================

section(
    "5. SAVE FULL FUTURE RAW + CENTERED SHAP MATRICES"
)

metadata = future[
    [
        GRID_COL,
        YEAR_COL,
        "area_name",
    ]
].reset_index(drop=True)

raw_columns = [
    f"raw_shap__{feature}"
    for feature in ordered_features
]

centered_columns = [
    f"centered_shap__{feature}"
    for feature in ordered_features
]

raw_export = pd.concat(
    [
        metadata,
        pd.DataFrame(
            raw_shap,
            columns=raw_columns,
        ),
    ],
    axis=1,
)

centered_export = pd.concat(
    [
        metadata,
        pd.DataFrame(
            centered_shap,
            columns=centered_columns,
        ),
    ],
    axis=1,
)

raw_export.to_csv(
    RAW_SHAP_PATH,
    index=False,
)

centered_export.to_csv(
    CENTERED_SHAP_PATH,
    index=False,
)

print(
    f"Saved: {RAW_SHAP_PATH}"
)

print(
    f"Saved: {CENTERED_SHAP_PATH}"
)


# =============================================================================
# 6. GLOBAL + YEARLY FUTURE SHAP IMPORTANCE
# =============================================================================

section(
    "6. FUTURE GLOBAL + YEARLY SHAP IMPORTANCE"
)

global_rows = []

for j, feature in enumerate(
    ordered_features
):
    values = centered_shap[
        :,
        j
    ]

    global_rows.append(
        {
            "feature": feature,
            "mean_abs_centered_shap_c": float(
                np.mean(
                    np.abs(values)
                )
            ),
            "mean_centered_shap_c": float(
                np.mean(values)
            ),
            "pct_positive_contribution": float(
                np.mean(
                    values > 0
                )
                * 100.0
            ),
            "pct_negative_contribution": float(
                np.mean(
                    values < 0
                )
                * 100.0
            ),
            "mean_abs_raw_shap_c": float(
                np.mean(
                    np.abs(
                        raw_shap[
                            :,
                            j
                        ]
                    )
                )
            ),
        }
    )

global_importance = (
    pd.DataFrame(
        global_rows
    )
    .sort_values(
        "mean_abs_centered_shap_c",
        ascending=False,
    )
    .reset_index(drop=True)
)

global_importance.insert(
    0,
    "rank",
    np.arange(
        1,
        len(global_importance) + 1,
    ),
)

global_importance.to_csv(
    GLOBAL_IMPORTANCE_PATH,
    index=False,
)

print(
    "Top 20 future global centered-SHAP features:"
)

print(
    global_importance.head(20)
    .to_string(index=False)
)

yearly_rows = []

year_array = future[
    YEAR_COL
].to_numpy()

for year in FUTURE_YEARS:
    mask = (
        year_array
        == year
    )

    for j, feature in enumerate(
        ordered_features
    ):
        values = centered_shap[
            mask,
            j
        ]

        yearly_rows.append(
            {
                YEAR_COL: year,
                "feature": feature,
                "mean_abs_centered_shap_c": float(
                    np.mean(
                        np.abs(values)
                    )
                ),
                "mean_centered_shap_c": float(
                    np.mean(values)
                ),
                "pct_positive_contribution": float(
                    np.mean(
                        values > 0
                    )
                    * 100.0
                ),
                "pct_negative_contribution": float(
                    np.mean(
                        values < 0
                    )
                    * 100.0
                ),
            }
        )

yearly_importance = pd.DataFrame(
    yearly_rows
)

yearly_importance[
    "rank_within_year"
] = (
    yearly_importance.groupby(
        YEAR_COL
    )[
        "mean_abs_centered_shap_c"
    ]
    .rank(
        method="first",
        ascending=False,
    )
    .astype(int)
)

yearly_importance = (
    yearly_importance
    .sort_values(
        [
            YEAR_COL,
            "rank_within_year",
        ]
    )
    .reset_index(drop=True)
)

yearly_importance.to_csv(
    YEARLY_IMPORTANCE_PATH,
    index=False,
)

for year in FUTURE_YEARS:
    print(
        f"\nTop 10 — {year}"
    )

    print(
        yearly_importance[
            yearly_importance[
                YEAR_COL
            ] == year
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


# =============================================================================
# 7. LOCAL TOP-10 EXPLANATIONS FOR EVERY GRID-YEAR
# =============================================================================

section(
    "7. LOCAL TOP-10 FUTURE EXPLANATIONS FOR EVERY GRID-YEAR"
)

feature_values = X.to_numpy(
    dtype=float
)

local_rows = []
summary_rows = []

for row_index in range(
    EXPECTED_ROWS
):
    contributions = centered_shap[
        row_index
    ]

    absolute_order = np.argsort(
        -np.abs(
            contributions
        )
    )[
        :LOCAL_TOP_K
    ]

    positive_order = np.argsort(
        -contributions
    )

    positive_order = [
        j
        for j in positive_order
        if contributions[j] > 0
    ]

    negative_order = np.argsort(
        contributions
    )

    negative_order = [
        j
        for j in negative_order
        if contributions[j] < 0
    ]

    key = {
        GRID_COL: str(
            future.at[
                row_index,
                GRID_COL
            ]
        ),
        YEAR_COL: int(
            future.at[
                row_index,
                YEAR_COL
            ]
        ),
        "area_name": str(
            future.at[
                row_index,
                "area_name"
            ]
        ),
    }

    for local_rank, j in enumerate(
        absolute_order,
        start=1,
    ):
        value = float(
            contributions[j]
        )

        local_rows.append(
            {
                **key,
                "local_rank_abs": local_rank,
                "feature": ordered_features[j],
                "feature_value": float(
                    feature_values[
                        row_index,
                        j
                    ]
                ),
                "centered_shap_c": value,
                "abs_centered_shap_c": abs(value),
                "direction": (
                    "heating_relative_to_annual_level"
                    if value > 0
                    else (
                        "cooling_relative_to_annual_level"
                        if value < 0
                        else "neutral"
                    )
                ),
            }
        )

    row_summary = {
        **key,
        "predicted_lst_c": float(
            pred.at[
                row_index,
                "predicted_lst_c"
            ]
        ),
        "forecast_annual_mean_lst_c": float(
            pred.at[
                row_index,
                "forecast_annual_mean_lst_c"
            ]
        ),
        "centered_spatial_anomaly_c": float(
            pred.at[
                row_index,
                "centered_spatial_anomaly_c"
            ]
        ),
    }

    for rank in range(
        1,
        4,
    ):
        if len(positive_order) >= rank:
            j = positive_order[
                rank - 1
            ]

            row_summary[
                f"top{rank}_heating_feature"
            ] = ordered_features[j]

            row_summary[
                f"top{rank}_heating_shap_c"
            ] = float(
                contributions[j]
            )

        else:
            row_summary[
                f"top{rank}_heating_feature"
            ] = ""

            row_summary[
                f"top{rank}_heating_shap_c"
            ] = np.nan

        if len(negative_order) >= rank:
            j = negative_order[
                rank - 1
            ]

            row_summary[
                f"top{rank}_cooling_feature"
            ] = ordered_features[j]

            row_summary[
                f"top{rank}_cooling_shap_c"
            ] = float(
                contributions[j]
            )

        else:
            row_summary[
                f"top{rank}_cooling_feature"
            ] = ""

            row_summary[
                f"top{rank}_cooling_shap_c"
            ] = np.nan

    summary_rows.append(
        row_summary
    )

local_top10 = pd.DataFrame(
    local_rows
)

local_summary = pd.DataFrame(
    summary_rows
)

local_top10.to_csv(
    LOCAL_TOP10_PATH,
    index=False,
)

local_summary.to_csv(
    LOCAL_SUMMARY_PATH,
    index=False,
)

for year in FUTURE_YEARS:
    local_top10[
        local_top10[
            YEAR_COL
        ] == year
    ].to_csv(
        EXPLAIN_DIR
        / f"28b_future_local_top10_shap_{year}.csv",
        index=False,
    )

    local_summary[
        local_summary[
            YEAR_COL
        ] == year
    ].to_csv(
        EXPLAIN_DIR
        / f"28b_future_local_explanation_summary_{year}.csv",
        index=False,
    )

print(
    f"All-year local top-10 rows: {len(local_top10):,}"
)

print(
    f"Expected: {EXPECTED_ROWS * LOCAL_TOP_K:,}"
)

print(
    f"Saved: {LOCAL_TOP10_PATH}"
)

print(
    f"Saved: {LOCAL_SUMMARY_PATH}"
)


# =============================================================================
# 8. REPORT-QUALITY GLOBAL / YEARLY IMPORTANCE FIGURES
# =============================================================================

section(
    "8. SAVE FUTURE SHAP IMPORTANCE FIGURES"
)

plot_global = (
    global_importance
    .head(20)
    .sort_values(
        "mean_abs_centered_shap_c",
        ascending=True,
    )
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

ax.barh(
    plot_global[
        "feature"
    ],
    plot_global[
        "mean_abs_centered_shap_c"
    ],
)

ax.set_xlabel(
    "Mean |centered SHAP| (°C)"
)

ax.set_ylabel(
    "Feature"
)

ax.set_title(
    "2026–2030 Future Spatial-Anomaly SHAP Importance"
)

ax.grid(
    axis="x",
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    GLOBAL_BAR_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    f"Saved: {GLOBAL_BAR_PATH}"
)

for year in FUTURE_YEARS:
    top = (
        yearly_importance[
            yearly_importance[
                YEAR_COL
            ] == year
        ]
        .head(15)
        .sort_values(
            "mean_abs_centered_shap_c",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        top[
            "feature"
        ],
        top[
            "mean_abs_centered_shap_c"
        ],
    )

    ax.set_xlabel(
        "Mean |centered SHAP| (°C)"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        f"Future Spatial-Anomaly SHAP Importance — {year}"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    plt.tight_layout()

    path = (
        FIGURE_DIR
        / f"28b_future_shap_importance_{year}.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {path}"
    )


# =============================================================================
# 9. FUTURE SHAP GEOJSON + SPATIAL MAPS
# =============================================================================

section(
    "9. SAVE FUTURE SHAP GEOJSON + SPATIAL MAPS"
)

grid = gpd.read_file(
    GRID_WGS84_PATH
)

grid[GRID_COL] = (
    grid[GRID_COL]
    .astype(str)
)

if grid[GRID_COL].nunique() != EXPECTED_GRIDS:
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

map_files = []
geojson_files = []

for year in FUTURE_YEARS:
    year_mask = (
        year_array
        == year
    )

    year_summary = local_summary[
        local_summary[
            YEAR_COL
        ] == year
    ].copy()

    merged = (
        grid
        .merge(
            year_summary,
            on=GRID_COL,
            how="left",
            validate="one_to_one",
        )
    )

    if merged[
        "predicted_lst_c"
    ].isna().any():
        raise RuntimeError(
            f"{year}: SHAP GeoJSON merge produced missing rows."
        )

    geo_path = (
        MAP_DIR
        / f"28b_future_shap_explanation_{year}_wgs84.geojson"
    )

    merged.to_file(
        geo_path,
        driver="GeoJSON",
    )

    geojson_files.append(
        geo_path
    )

    # Centered anomaly map
    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    merged.plot(
        column="centered_spatial_anomaly_c",
        ax=ax,
        legend=True,
        edgecolor="none",
        legend_kwds={
            "label": "Centered spatial anomaly (°C)",
            "shrink": 0.72,
        },
    )

    if boundary is not None:
        boundary.boundary.plot(
            ax=ax,
            linewidth=1.4,
        )
    else:
        merged.dissolve().boundary.plot(
            ax=ax,
            linewidth=1.4,
        )

    ax.set_title(
        f"Kaduwela Future Spatial-Anomaly Prediction — {year}",
        fontsize=14,
        pad=12,
    )

    ax.set_axis_off()

    plt.tight_layout()

    anomaly_map_path = (
        FIGURE_DIR
        / f"28b_future_centered_spatial_anomaly_{year}.png"
    )

    plt.savefig(
        anomaly_map_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    map_files.append(
        anomaly_map_path
    )

    top_features = (
        yearly_importance[
            yearly_importance[
                YEAR_COL
            ] == year
        ]
        .head(
            MAP_TOP_FEATURE_COUNT
        )[
            "feature"
        ]
        .tolist()
    )

    row_indices = np.where(
        year_mask
    )[0]

    for feature_rank, feature in enumerate(
        top_features,
        start=1,
    ):
        feature_index = ordered_features.index(
            feature
        )

        shap_lookup = pd.DataFrame(
            {
                GRID_COL: future.loc[
                    year_mask,
                    GRID_COL
                ].astype(str).to_numpy(),
                "centered_shap_c": centered_shap[
                    year_mask,
                    feature_index
                ],
            }
        )

        map_gdf = (
            grid
            .merge(
                shap_lookup,
                on=GRID_COL,
                how="left",
                validate="one_to_one",
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 10)
        )

        map_gdf.plot(
            column="centered_shap_c",
            ax=ax,
            legend=True,
            edgecolor="none",
            legend_kwds={
                "label": "Centered SHAP contribution (°C)",
                "shrink": 0.72,
            },
        )

        if boundary is not None:
            boundary.boundary.plot(
                ax=ax,
                linewidth=1.4,
            )
        else:
            map_gdf.dissolve().boundary.plot(
                ax=ax,
                linewidth=1.4,
            )

        ax.set_title(
            f"{year} SHAP Rank {feature_rank}: {feature}",
            fontsize=14,
            pad=12,
        )

        ax.set_axis_off()

        plt.tight_layout()

        map_path = (
            FIGURE_DIR
            / (
                f"28b_future_{year}_shap_map_rank"
                f"{feature_rank}_{safe_name(feature)}.png"
            )
        )

        plt.savefig(
            map_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        map_files.append(
            map_path
        )

    print(
        f"{year}: saved local GeoJSON + anomaly map + "
        f"{MAP_TOP_FEATURE_COUNT} top-feature SHAP maps"
    )


# =============================================================================
# 10. STRICT STEP-28B AUDIT
# =============================================================================

section(
    "10. STRICT STEP-28B AUDIT"
)

year_center_feature_means = []

for year in FUTURE_YEARS:
    mask = (
        year_array
        == year
    )

    year_center_feature_means.append(
        float(
            np.max(
                np.abs(
                    centered_shap[
                        mask
                    ].mean(
                        axis=0
                    )
                )
            )
        )
    )

max_feature_center_mean = max(
    year_center_feature_means
)

local_counts = (
    local_top10.groupby(
        [
            GRID_COL,
            YEAR_COL,
        ]
    )
    .size()
)

geojson_row_ok = True

for path in geojson_files:
    test = gpd.read_file(
        path
    )

    if len(test) != EXPECTED_GRIDS:
        geojson_row_ok = False
        break

audit_rows = [
    {
        "check": "step28_audit",
        "status": "PASS",
        "details": "Step-28 strict audit verified fully PASS.",
    },
    {
        "check": "production_model_hash",
        "status": "PASS",
        "details": EXPECTED_HASHES[
            "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
        ],
    },
    {
        "check": "future_feature_hash",
        "status": "PASS",
        "details": EXPECTED_HASHES[
            "28_future_model_features_2026_2030.csv"
        ],
    },
    {
        "check": "future_prediction_hash",
        "status": "PASS",
        "details": EXPECTED_HASHES[
            "28_future_grid_lst_predictions_2026_2030.csv"
        ],
    },
    {
        "check": "future_rows",
        "status": (
            "PASS"
            if len(future) == EXPECTED_ROWS
            else "FAIL"
        ),
        "details": (
            f"actual={len(future)}, expected={EXPECTED_ROWS}"
        ),
    },
    {
        "check": "future_years",
        "status": (
            "PASS"
            if sorted(
                future[YEAR_COL]
                .unique()
                .tolist()
            )
            == FUTURE_YEARS
            else "FAIL"
        ),
        "details": str(FUTURE_YEARS),
    },
    {
        "check": "unique_grids",
        "status": (
            "PASS"
            if future[GRID_COL].nunique()
            == EXPECTED_GRIDS
            else "FAIL"
        ),
        "details": (
            f"actual={future[GRID_COL].nunique()}, "
            f"expected={EXPECTED_GRIDS}"
        ),
    },
    {
        "check": "frozen_feature_count",
        "status": (
            "PASS"
            if len(ordered_features)
            == EXPECTED_FEATURES
            else "FAIL"
        ),
        "details": (
            f"actual={len(ordered_features)}, "
            f"expected={EXPECTED_FEATURES}"
        ),
    },
    {
        "check": "step28_raw_prediction_reproduction",
        "status": (
            "PASS"
            if raw_prediction_reproduction_error
            <= PREDICTION_REPRO_TOL
            else "FAIL"
        ),
        "details": (
            f"max_error_c={raw_prediction_reproduction_error:.12e}"
        ),
    },
    {
        "check": "raw_shap_reconstruction",
        "status": (
            "PASS"
            if raw_reconstruction_error
            <= RAW_RECON_TOL
            else "FAIL"
        ),
        "details": (
            f"max_error_c={raw_reconstruction_error:.12e}"
        ),
    },
    {
        "check": "centered_shap_reconstruction",
        "status": (
            "PASS"
            if max_centered_reconstruction_error
            <= CENTERED_RECON_TOL
            else "FAIL"
        ),
        "details": (
            f"max_error_c={max_centered_reconstruction_error:.12e}"
        ),
    },
    {
        "check": "centered_feature_shap_zero_mean_within_year",
        "status": (
            "PASS"
            if max_feature_center_mean
            <= CENTERING_TOL
            else "FAIL"
        ),
        "details": (
            f"max_abs_feature_year_mean_c={max_feature_center_mean:.12e}"
        ),
    },
    {
        "check": "local_top10_rows",
        "status": (
            "PASS"
            if len(local_top10)
            == EXPECTED_ROWS * LOCAL_TOP_K
            else "FAIL"
        ),
        "details": (
            f"actual={len(local_top10)}, "
            f"expected={EXPECTED_ROWS * LOCAL_TOP_K}"
        ),
    },
    {
        "check": "local_top10_per_grid_year",
        "status": (
            "PASS"
            if (
                len(local_counts)
                == EXPECTED_ROWS
                and (
                    local_counts
                    == LOCAL_TOP_K
                ).all()
            )
            else "FAIL"
        ),
        "details": (
            f"grid_years={len(local_counts)}, "
            f"expected_per_grid_year={LOCAL_TOP_K}"
        ),
    },
    {
        "check": "future_shap_geojson_rows",
        "status": (
            "PASS"
            if geojson_row_ok
            else "FAIL"
        ),
        "details": (
            f"{len(geojson_files)} yearly GeoJSON files; "
            f"{EXPECTED_GRIDS} rows expected each"
        ),
    },
    {
        "check": "future_shap_maps_saved",
        "status": (
            "PASS"
            if len(map_files)
            == len(FUTURE_YEARS)
            * (
                1
                + MAP_TOP_FEATURE_COUNT
            )
            else "FAIL"
        ),
        "details": (
            f"saved={len(map_files)}, "
            f"expected={len(FUTURE_YEARS) * (1 + MAP_TOP_FEATURE_COUNT)}"
        ),
    },
    {
        "check": "annual_level_explained_by_shap",
        "status": "PASS",
        "details": (
            "No. SHAP is explicitly restricted to the LightGBM "
            "spatial-anomaly component."
        ),
    },
    {
        "check": "model_retraining",
        "status": "PASS",
        "details": (
            "None. Frozen Step-22B production model loaded for explanation only."
        ),
    },
    {
        "check": "new_accuracy_or_test_score",
        "status": "PASS",
        "details": (
            "None. Step 28B is explainability only."
        ),
    },
    {
        "check": "causal_claim",
        "status": "PASS",
        "details": (
            "SHAP documented as model attribution, not causal evidence."
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
        "One or more Step-28B audit checks failed."
    )


# =============================================================================
# 11. METHODOLOGY + REPORT
# =============================================================================

section(
    "11. SAVE STEP-28B METHODOLOGY + REPORT"
)

methodology = {
    "step": "28B",
    "version": "future_centered_shap_v1",
    "analysis": "Future SHAP explainability for 2026-2030",
    "explained_model": str(MODEL_PATH),
    "model_type": type(model).__name__,
    "model_target": "within-year spatial LST anomaly",
    "feature_count": EXPECTED_FEATURES,
    "future_years": FUTURE_YEARS,
    "future_rows": EXPECTED_ROWS,
    "unique_grids": EXPECTED_GRIDS,
    "shap_method": "TreeSHAP",
    "raw_shap_definition": (
        "Feature contributions to the raw Step-22B LightGBM "
        "spatial-anomaly output."
    ),
    "centered_shap_definition": (
        "Within-year raw SHAP contribution minus the same feature's "
        "mean raw SHAP contribution across all 2,194 grids in that year."
    ),
    "centered_shap_identity": (
        "sum(centered feature SHAP values) = Step-28 centered "
        "model-predicted spatial anomaly."
    ),
    "future_lst_identity": (
        "Step-28 final grid LST = forecast annual Kaduwela thermal level "
        "+ centered spatial anomaly."
    ),
    "annual_level_explanation": (
        "Not attributed by SHAP. The citywide annual level is produced "
        "by the separate frozen annual forecasting branch."
    ),
    "global_importance_definition": (
        "Mean absolute centered SHAP contribution across all 10,970 "
        "2026-2030 future grid-year rows."
    ),
    "yearly_importance_definition": (
        "Mean absolute centered SHAP contribution across 2,194 grids "
        "within each future year."
    ),
    "local_top_k": LOCAL_TOP_K,
    "spatial_map_top_feature_count_per_year": MAP_TOP_FEATURE_COUNT,
    "interpretation": (
        "Positive centered SHAP pushes a grid hotter relative to "
        "Kaduwela's annual thermal level; negative centered SHAP pushes "
        "it cooler relative to that level."
    ),
    "causal_warning": (
        "SHAP explains the fitted model and forecast inputs. "
        "It does not prove causal environmental effects."
    ),
    "forecast_input_warning": (
        "2026-2030 predictor values are forecasts rather than observed "
        "future measurements, so future SHAP explanations inherit "
        "forecast-input uncertainty."
    ),
    "evaluation_warning": (
        "Step 28B creates no new accuracy, validation, or test score."
    ),
}

METHOD_PATH.write_text(
    json.dumps(
        methodology,
        indent=2,
    ),
    encoding="utf-8",
)

report_lines = [
    "STEP 28B - FUTURE 2026-2030 SHAP EXPLAINABILITY",
    "=" * 118,
    "",
    "PURPOSE",
    "-" * 118,
    (
        "Explain the frozen Step-28 future LightGBM spatial-anomaly "
        "predictions without changing the model or forecasts."
    ),
    "",
    "RECONSTRUCTION",
    "-" * 118,
    recon_df.to_string(index=False),
    "",
    "TOP 20 FUTURE GLOBAL SHAP FEATURES",
    "-" * 118,
    global_importance.head(20).to_string(index=False),
    "",
    "INTERPRETATION",
    "-" * 118,
    (
        "Positive centered SHAP means a feature pushes that grid hotter "
        "relative to Kaduwela's annual thermal level."
    ),
    (
        "Negative centered SHAP means a feature pushes that grid cooler "
        "relative to Kaduwela's annual thermal level."
    ),
    (
        "SHAP explains only the spatial-anomaly LightGBM branch. "
        "It does not explain the separate annual-level forecast."
    ),
    (
        "These are model attributions, not causal effects."
    ),
    (
        "Future SHAP uses forecast predictor values, so its interpretation "
        "inherits uncertainty from Step-28 future predictor forecasting."
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
# 12. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "12. REPRODUCIBILITY HASHES"
)

hash_files = [
    FEATURE_LIST_PATH,
    MODEL_PATH,
    FUTURE_FEATURE_PATH,
    FUTURE_PRED_PATH,
    STEP28_AUDIT_PATH,
    GRID_WGS84_PATH,
    RAW_SHAP_PATH,
    CENTERED_SHAP_PATH,
    GLOBAL_IMPORTANCE_PATH,
    YEARLY_IMPORTANCE_PATH,
    LOCAL_TOP10_PATH,
    LOCAL_SUMMARY_PATH,
    RECON_PATH,
    GLOBAL_BAR_PATH,
    METHOD_PATH,
    AUDIT_PATH,
    REPORT_PATH,
]

for year in FUTURE_YEARS:
    hash_files.append(
        EXPLAIN_DIR
        / f"28b_future_local_top10_shap_{year}.csv"
    )

    hash_files.append(
        EXPLAIN_DIR
        / f"28b_future_local_explanation_summary_{year}.csv"
    )

    hash_files.append(
        MAP_DIR
        / f"28b_future_shap_explanation_{year}_wgs84.geojson"
    )

    hash_files.append(
        FIGURE_DIR
        / f"28b_future_shap_importance_{year}.png"
    )

for path in map_files:
    hash_files.append(
        path
    )

# Remove duplicates while preserving order.
unique_hash_files = []

seen = set()

for path in hash_files:
    path = Path(path)

    if str(path) not in seen:
        seen.add(
            str(path)
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
            "file": str(path),
            "sha256": digest,
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
    "STEP 28B COMPLETED SUCCESSFULLY"
)

print(
    "Future SHAP explainability completed for 2026-2030.\n"
    f"Future explained rows: {EXPECTED_ROWS:,}.\n"
    f"Unique grids/year: {EXPECTED_GRIDS:,}.\n"
    f"Frozen predictors: {EXPECTED_FEATURES}.\n"
    "TreeSHAP reconstructed the raw future LightGBM spatial-anomaly output.\n"
    "Year-centered SHAP reconstructed the exact Step-28 centered spatial anomaly.\n"
    "Global and yearly future SHAP importance tables were saved.\n"
    "Top-10 local explanations were saved for every future grid-year.\n"
    "Per-year frontend-ready local explanation summaries and GeoJSON files were saved.\n"
    "Report-quality spatial SHAP maps were saved for each future year.\n"
    "The Step-22B production model was not retrained or modified.\n"
    "No new accuracy/test result was created.\n"
    "\nNEXT: review Step-28B outputs before starting Step 29 cooling scenario simulation."
)
