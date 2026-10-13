"""
STEP 28 - OPERATIONAL 2026-2030 FUTURE HEAT PREDICTION
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Generate the frozen operational forecasts for 2026-2030 after Step 27:

1) Forecast the selected dynamic BASE predictors for all 2,194 grids.
2) Keep static / structural predictors according to the Step-27 freeze.
3) Reconstruct the original Step-07 engineered features exactly.
4) Feed the exact frozen 93-feature matrix to the Step-22B production LightGBM.
5) Center the raw spatial-anomaly predictions within each future year.
6) Add the selected annual Kaduwela thermal-level forecast.
7) Save future predictor tables, full model-feature matrices, grid LST forecasts,
   GeoJSON outputs, and report-quality future maps.

FINAL EQUATION
--------------
Future Grid LST
    = Forecast Kaduwela Annual Thermal Level
    + Centered Step-22B Spatial-Anomaly Prediction

FROZEN STEP-27 STRATEGY
-----------------------
Annual level:
    Expanding Mean

Dynamic predictors:
    NDVI         -> Rolling-3 Mean
    NDBI         -> Rolling-3 Mean
    NDWI         -> Rolling-3 Mean
    EVI          -> Rolling-3 Mean
    Albedo       -> Bounded Theil-Sen Trend
    NDBI_ADJ     -> Rolling-3 Mean
    NDVI_ADJ     -> Rolling-3 Mean
    NightLights  -> Rolling-3 Mean
    green_mask   -> Rolling-3 Mean

Static / frozen:
    building_mask, road_mask

Structural:
    dist_road, dist_main_road, dist_water, LCZ, area_name

IMPORTANT
---------
- The Step-22B LightGBM is NOT retrained.
- 2026-2030 forecast rows are NOT treated as observed data.
- The annual-level and predictor strategies are NOT reselected here.
- Engineered features are NOT forecast independently.
- Predictor forecasts use the same fixed-origin H1-H5 interpretation validated
  in Step 27: all operational forecasts originate from observed history through
  2025. The future base-predictor sequence is then used only to reconstruct
  future lag/history/delta/spatial-context features.

STRICT FEATURE-ENGINEERING REPRODUCTION
---------------------------------------
Before any future LST prediction is accepted, this script rebuilds Step-07
features for historical 2025 and compares them against
kaduwela_engineered_features.csv. If the historical reconstruction does not
match within tolerance, Step 28 stops instead of silently producing an
incompatible future feature matrix.
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
        "\ngeopandas is required for Step 28.\n"
        "Install it with:\n"
        "    pip install geopandas\n"
    ) from exc

try:
    from libpysal.weights import Queen
except ImportError as exc:
    raise ImportError(
        "\nlibpysal is required to reproduce Step-07 Queen-neighbour features.\n"
        "Install it with:\n"
        "    pip install libpysal\n"
    ) from exc


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET_COL = "LST_C"

EXPECTED_BASE_ROWS = 24134
EXPECTED_GRIDS = 2194
EXPECTED_HIST_YEARS = list(range(2015, 2026))
FUTURE_YEARS = list(range(2026, 2031))
EXPECTED_FUTURE_ROWS = EXPECTED_GRIDS * len(FUTURE_YEARS)

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
    "area_name",
]

EXPECTED_STEP27_SELECTION = {
    "annual_level": "Expanding Mean",
    "dynamic_predictors": {
        "NDVI": "Rolling-3 Mean",
        "NDBI": "Rolling-3 Mean",
        "NDWI": "Rolling-3 Mean",
        "EVI": "Rolling-3 Mean",
        "Albedo": "Bounded Theil-Sen Trend",
        "NDBI_ADJ": "Rolling-3 Mean",
        "NDVI_ADJ": "Rolling-3 Mean",
        "NightLights": "Rolling-3 Mean",
        "green_mask": "Rolling-3 Mean",
    },
}

EXPECTED_HASHES = {
    "kaduwela_preprocessed_base.csv":
        "0e030b2ec0b5c4080cd4226bec798189a178aa45efd9962d19f38dc9cb170f46",
    "21_final_feature_list.csv":
        "6769a195b5027850ebd458a4bef7c9da0ecc43c4969bcc81f746e46bce3be9c0",
    "22b_production_lightgbm_anomaly_trained_2018_2025.joblib":
        "a3548c58ba0a708c81b4f6471e9183aa70ce9572c716ee55bf2eddafe183e24f",
    "27_forecast_strategy_freeze_for_step28.json":
        "2d7ffeccc409d3300d0f261c79735ac14fe37fc93b6ac9631ebb4dfc542afc9b",
    "27_forecast_validation_audit.csv":
        "a6ceb741ac36237111afcc7d4ec31a0853ce08dd5b0f5e43b383c60721df00d0",
    "kaduwela_grid_geometry_wgs84.geojson":
        "6b758ddaae19b54219404b86740472038cfd21391cf349ea766124678050f608",
}

RECONSTRUCTION_TOL = 1e-8
ANOMALY_CENTER_TOL = 1e-10


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_PATH = (
    ROOT / "data" / "processed" / "kaduwela_preprocessed_base.csv"
)

ENGINEERED_HIST_PATH = (
    ROOT / "data" / "processed" / "kaduwela_engineered_features.csv"
)

GRID_UTM_PATH = (
    ROOT / "data" / "processed" / "kaduwela_grid_geometry_utm44n.geojson"
)

GRID_WGS84_PATH = (
    ROOT / "data" / "processed" / "kaduwela_grid_geometry_wgs84.geojson"
)

BOUNDARY_KML_PATH = (
    ROOT / "data" / "raw" / "boundary" / "kaduwela_kmc_boundary.kml"
)

FEATURE_LIST_PATH = (
    ROOT / "outputs" / "reports" / "21_final_feature_list.csv"
)

MODEL_PATH = (
    ROOT / "models" / "final"
    / "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
)

STEP27_FREEZE_PATH = (
    ROOT / "outputs" / "reports"
    / "27_forecast_strategy_freeze_for_step28.json"
)

STEP27_AUDIT_PATH = (
    ROOT / "outputs" / "reports"
    / "27_forecast_validation_audit.csv"
)

ANNUAL_SERIES_PATH = (
    ROOT / "outputs" / "forecasting"
    / "25_annual_kaduwela_mean_lst_2015_2025.csv"
)

FORECAST_DIR = ROOT / "outputs" / "forecasting"
MAP_DIR = ROOT / "outputs" / "maps"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [
    FORECAST_DIR,
    MAP_DIR,
    FIGURE_DIR,
    REPORT_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

FUTURE_BASE_PATH = (
    FORECAST_DIR / "28_future_base_predictors_2026_2030.csv"
)

FUTURE_FEATURE_PATH = (
    FORECAST_DIR / "28_future_model_features_2026_2030.csv"
)

ANNUAL_LEVEL_PATH = (
    FORECAST_DIR / "28_future_annual_thermal_level_2026_2030.csv"
)

FUTURE_LST_PATH = (
    FORECAST_DIR / "28_future_grid_lst_predictions_2026_2030.csv"
)

YEAR_SUMMARY_PATH = (
    FORECAST_DIR / "28_future_year_summary.csv"
)

FEATURE_RECON_AUDIT_PATH = (
    REPORT_DIR / "28_feature_reconstruction_audit_2025.csv"
)

AUDIT_PATH = (
    REPORT_DIR / "28_future_prediction_audit.csv"
)

METHOD_PATH = (
    REPORT_DIR / "28_future_prediction_methodology.json"
)

REPORT_PATH = (
    REPORT_DIR / "28_future_prediction_report.txt"
)

HASH_PATH = (
    REPORT_DIR / "28_future_prediction_hashes.csv"
)

COMBINED_GEOJSON_PATH = (
    MAP_DIR / "28_future_heat_predictions_2026_2030_wgs84.geojson"
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
        raise FileNotFoundError(f"Missing required file: {path}")

    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual  : {actual}"
        )

    print(f"PASS hash: {label} = {actual}")
    return actual


def require_all_pass(path, label):
    audit = pd.read_csv(path)

    if "status" not in audit.columns:
        raise RuntimeError(f"{label} audit has no status column.")

    bad = audit[
        audit["status"].astype(str).str.upper() != "PASS"
    ]

    if not bad.empty:
        print(bad.to_string(index=False))
        raise RuntimeError(f"{label} audit is not fully PASS.")


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

            if len(values) == 93:
                return conventional

    for column in feature_df.columns:
        values = (
            feature_df[column]
            .dropna()
            .astype(str)
            .tolist()
        )

        if len(values) == 93 and len(set(values)) == 93:
            return column

    raise RuntimeError(
        "Could not identify the 93-feature column in Step-21 feature list."
    )


def infer_year_index_relation(engineered):
    """
    Step 08 recorded year_index as an exact linear transform of year.
    Infer that exact transform from the historical engineered dataset rather
    than assuming a specific offset.
    """
    pairs = (
        engineered[
            [YEAR_COL, "year_index"]
        ]
        .drop_duplicates()
        .sort_values(YEAR_COL)
    )

    if len(pairs) != len(EXPECTED_HIST_YEARS):
        raise RuntimeError(
            "Could not infer year_index: unexpected unique year/year_index pairs."
        )

    x = pairs[YEAR_COL].to_numpy(dtype=float)
    y = pairs["year_index"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    reconstructed = intercept + slope * x

    max_error = float(
        np.max(
            np.abs(
                reconstructed - y
            )
        )
    )

    if max_error > 1e-10:
        raise RuntimeError(
            "year_index is not reproduced by one exact linear transform."
        )

    return float(slope), float(intercept)


def build_queen_neighbors(grid_utm):
    """
    Reproduce Step-07 Queen contiguity from the corrected projected grid.
    """
    ordered = (
        grid_utm[
            [GRID_COL, "geometry"]
        ]
        .copy()
        .sort_values(GRID_COL)
        .reset_index(drop=True)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights = Queen.from_dataframe(
            ordered,
            ids=ordered[GRID_COL].tolist(),
            silence_warnings=True,
        )

    neighbors = {
        str(grid_id): [str(n) for n in nbrs]
        for grid_id, nbrs in weights.neighbors.items()
    }

    isolates = [
        grid_id
        for grid_id, nbrs in neighbors.items()
        if len(nbrs) == 0
    ]

    if isolates:
        raise RuntimeError(
            f"Queen adjacency produced {len(isolates)} isolated grids. "
            "Step-07 historical spatial features had no missing neighbour means."
        )

    return neighbors


def add_neighbor_features(df, neighbors):
    out = df.copy()

    for source in TEMPORAL_SOURCES:
        mean_col = f"{source}_nbr_mean"
        contrast_col = f"{source}_nbr_contrast"

        result = np.empty(len(out), dtype=float)

        for year, index in out.groupby(YEAR_COL).groups.items():
            year_rows = out.loc[index, [GRID_COL, source]]

            value_lookup = dict(
                zip(
                    year_rows[GRID_COL].astype(str),
                    year_rows[source].astype(float),
                )
            )

            means = []

            for grid_id in year_rows[GRID_COL].astype(str):
                nbr_values = [
                    value_lookup[nbr]
                    for nbr in neighbors[str(grid_id)]
                ]

                means.append(
                    float(np.mean(nbr_values))
                )

            result[np.asarray(index)] = np.asarray(means, dtype=float)

        out[mean_col] = result
        out[contrast_col] = (
            out[source] - out[mean_col]
        )

    return out


def hist3_slope(values):
    """
    Slope through exactly three equally spaced PRIOR values.
    For x=[0,1,2], OLS slope simplifies to (last-first)/2.
    """
    arr = np.asarray(values, dtype=float)

    if len(arr) != 3 or np.isnan(arr).any():
        return np.nan

    return float(
        (arr[2] - arr[0]) / 2.0
    )


def add_step07_features(
    combined_base,
    neighbors,
    year_index_slope,
    year_index_intercept,
):
    out = (
        combined_base
        .copy()
        .sort_values(
            [GRID_COL, YEAR_COL]
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Deterministic row-wise features
    # -------------------------------------------------------------------------
    out["year_index"] = (
        year_index_intercept
        + year_index_slope * out[YEAR_COL].astype(float)
    )

    out["vegetation_deficit"] = (
        1.0 - out["green_mask"]
    )

    out["ndbi_minus_ndvi"] = (
        out["NDBI"] - out["NDVI"]
    )

    out["ndbi_x_vegetation_deficit"] = (
        out["NDBI"]
        * out["vegetation_deficit"]
    )

    out["ndvi_x_green_mask"] = (
        out["NDVI"]
        * out["green_mask"]
    )

    out["ndbi_x_building_mask"] = (
        out["NDBI"]
        * out["building_mask"]
    )

    out["nightlights_x_building_mask"] = (
        out["NightLights"]
        * out["building_mask"]
    )

    for source in [
        "dist_road",
        "dist_main_road",
        "dist_water",
    ]:
        out[f"log1p_{source}"] = np.log1p(
            out[source].astype(float)
        )

    # -------------------------------------------------------------------------
    # Temporal features: strictly past predictors
    # -------------------------------------------------------------------------
    grouped = out.groupby(
        GRID_COL,
        sort=False,
    )

    for source in TEMPORAL_SOURCES:
        out[f"{source}_lag1"] = (
            grouped[source]
            .shift(1)
        )

        out[f"{source}_lag2"] = (
            grouped[source]
            .shift(2)
        )

        shifted = (
            grouped[source]
            .shift(1)
        )

        out[f"{source}_hist3_mean"] = (
            shifted
            .groupby(out[GRID_COL], sort=False)
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

        out[f"{source}_hist3_std"] = (
            shifted
            .groupby(out[GRID_COL], sort=False)
            .rolling(
                window=3,
                min_periods=2,
            )
            .std(ddof=1)
            .reset_index(
                level=0,
                drop=True,
            )
            .sort_index()
        )

        out[f"{source}_hist3_slope"] = (
            shifted
            .groupby(out[GRID_COL], sort=False)
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

        out[f"{source}_delta1"] = (
            out[source]
            - out[f"{source}_lag1"]
        )

    # -------------------------------------------------------------------------
    # Same-year predictor-only Queen-neighbour context
    # -------------------------------------------------------------------------
    out = add_neighbor_features(
        out,
        neighbors,
    )

    return out


def theil_sen_forecast_matrix(
    history_values,
    history_years,
    target_year,
):
    x = np.asarray(history_years, dtype=float)
    y = np.asarray(history_values, dtype=float)

    pair_slopes = []

    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            pair_slopes.append(
                (y[:, j] - y[:, i])
                / (x[j] - x[i])
            )

    slopes = np.median(
        np.stack(pair_slopes, axis=1),
        axis=1,
    )

    intercepts = (
        np.median(y, axis=1)
        - slopes * np.median(x)
    )

    return (
        intercepts
        + slopes * float(target_year)
    )


def forecast_dynamic_predictor(
    wide,
    method,
    target_year,
):
    """
    Fixed operational origin = 2025.
    This matches Step-27 direct H1-H5 validation semantics.
    """
    history_years = EXPECTED_HIST_YEARS
    history = wide[history_years].to_numpy(dtype=float)

    if method == "Rolling-3 Mean":
        prediction = np.mean(
            history[:, -3:],
            axis=1,
        )

    elif method == "Bounded Theil-Sen Trend":
        prediction = theil_sen_forecast_matrix(
            history_values=history,
            history_years=history_years,
            target_year=target_year,
        )

        prediction = np.clip(
            prediction,
            float(np.min(history)),
            float(np.max(history)),
        )

    elif method == "Persistence":
        prediction = history[:, -1].copy()

    elif method == "Bounded OLS Trend":
        x = np.asarray(
            history_years,
            dtype=float,
        )

        x_mean = float(np.mean(x))
        x_centered = x - x_mean
        denominator = float(
            np.sum(x_centered ** 2)
        )

        y_mean = np.mean(
            history,
            axis=1,
        )

        slopes = (
            np.sum(
                (
                    history - y_mean[:, None]
                )
                * x_centered[None, :],
                axis=1,
            )
            / denominator
        )

        prediction = (
            y_mean
            + slopes
            * (
                float(target_year)
                - x_mean
            )
        )

        prediction = np.clip(
            prediction,
            float(np.min(history)),
            float(np.max(history)),
        )

    else:
        raise ValueError(
            f"Unsupported frozen predictor method: {method}"
        )

    return prediction


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
            "WARNING: Could not read Kaduwela KML boundary for PNG outline:"
        )
        print(
            f"  {type(exc).__name__}: {exc}"
        )
        return None


# =============================================================================
# 0. VERIFY FROZEN INPUTS
# =============================================================================

section(
    "STEP 28 - OPERATIONAL 2026-2030 FUTURE HEAT PREDICTION"
)

verify_hash(
    BASE_PATH,
    EXPECTED_HASHES["kaduwela_preprocessed_base.csv"],
    "Canonical preprocessed base data",
)

verify_hash(
    FEATURE_LIST_PATH,
    EXPECTED_HASHES["21_final_feature_list.csv"],
    "Frozen Step-21 ordered feature list",
)

verify_hash(
    MODEL_PATH,
    EXPECTED_HASHES[
        "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
    ],
    "Frozen Step-22B production LightGBM",
)

verify_hash(
    STEP27_FREEZE_PATH,
    EXPECTED_HASHES[
        "27_forecast_strategy_freeze_for_step28.json"
    ],
    "Frozen Step-27 forecast strategy",
)

verify_hash(
    STEP27_AUDIT_PATH,
    EXPECTED_HASHES[
        "27_forecast_validation_audit.csv"
    ],
    "Step-27 strict audit",
)

verify_hash(
    GRID_WGS84_PATH,
    EXPECTED_HASHES[
        "kaduwela_grid_geometry_wgs84.geojson"
    ],
    "Corrected WGS84 grid geometry",
)

require_all_pass(
    STEP27_AUDIT_PATH,
    "Step 27",
)

print(
    "\nPASS: Step-27 audit is fully PASS."
)


# =============================================================================
# 1. LOAD + VERIFY STEP-27 FREEZE
# =============================================================================

section(
    "1. LOAD + VERIFY STEP-27 FROZEN STRATEGY"
)

with STEP27_FREEZE_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    freeze = json.load(f)

annual_method = freeze[
    "annual_level"
][
    "selected_method"
]

selected_methods = freeze[
    "dynamic_predictors"
][
    "selected_methods"
]

if annual_method != EXPECTED_STEP27_SELECTION["annual_level"]:
    raise RuntimeError(
        f"Unexpected frozen annual method: {annual_method}"
    )

if selected_methods != EXPECTED_STEP27_SELECTION["dynamic_predictors"]:
    raise RuntimeError(
        "Frozen dynamic-predictor methods differ from the finalized Step-27 selection."
    )

print(
    f"Annual thermal-level method: {annual_method}"
)

print(
    "\nDynamic predictor methods:"
)

for predictor, method in selected_methods.items():
    print(
        f"  {predictor:12s} -> {method}"
    )

print(
    "\nStep-22B policy:"
)

print(
    freeze["spatial_anomaly_model_policy"]
)


# =============================================================================
# 2. LOAD HISTORICAL BASE + ENGINEERED DATA + GEOMETRY
# =============================================================================

section(
    "2. LOAD HISTORICAL DATA + GEOMETRY"
)

base = pd.read_csv(
    BASE_PATH,
    dtype={
        GRID_COL: str,
        "area_name": str,
    },
)

if len(base) != EXPECTED_BASE_ROWS:
    raise RuntimeError(
        f"Base rows={len(base)}, expected={EXPECTED_BASE_ROWS}"
    )

if base[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Base grids={base[GRID_COL].nunique()}, expected={EXPECTED_GRIDS}"
    )

base[YEAR_COL] = pd.to_numeric(
    base[YEAR_COL],
    errors="raise",
).astype(int)

if sorted(base[YEAR_COL].unique().tolist()) != EXPECTED_HIST_YEARS:
    raise RuntimeError(
        "Historical base years are not exactly 2015-2025."
    )

engineered_hist = pd.read_csv(
    ENGINEERED_HIST_PATH,
    dtype={
        GRID_COL: str,
        "area_name": str,
    },
)

if len(engineered_hist) != EXPECTED_BASE_ROWS:
    raise RuntimeError(
        "Historical engineered dataset row count mismatch."
    )

grid_utm = gpd.read_file(
    GRID_UTM_PATH
)

grid_wgs84 = gpd.read_file(
    GRID_WGS84_PATH
)

for gdf, label in [
    (grid_utm, "UTM grid"),
    (grid_wgs84, "WGS84 grid"),
]:
    if GRID_COL not in gdf.columns:
        raise RuntimeError(
            f"{label} is missing grid_id."
        )

    gdf[GRID_COL] = (
        gdf[GRID_COL]
        .astype(str)
    )

    if gdf[GRID_COL].nunique() != EXPECTED_GRIDS:
        raise RuntimeError(
            f"{label} unique grids != {EXPECTED_GRIDS}"
        )

print(
    f"Base rows             : {len(base):,}"
)

print(
    f"Engineered hist rows  : {len(engineered_hist):,}"
)

print(
    f"Unique grids          : {base[GRID_COL].nunique():,}"
)

print(
    f"Historical years      : {EXPECTED_HIST_YEARS}"
)

print(
    f"Future years          : {FUTURE_YEARS}"
)


# =============================================================================
# 3. BUILD QUEEN NEIGHBOUR STRUCTURE
# =============================================================================

section(
    "3. BUILD STEP-07 QUEEN NEIGHBOUR STRUCTURE"
)

neighbors = build_queen_neighbors(
    grid_utm
)

neighbor_counts = np.array(
    [
        len(neighbors[str(grid_id)])
        for grid_id in sorted(neighbors)
    ],
    dtype=int,
)

print(
    f"Queen neighbour grids : {len(neighbors):,}"
)

print(
    f"Min neighbours        : {neighbor_counts.min()}"
)

print(
    f"Median neighbours     : {np.median(neighbor_counts):.1f}"
)

print(
    f"Max neighbours        : {neighbor_counts.max()}"
)


# =============================================================================
# 4. HISTORICAL FEATURE-ENGINEERING REPRODUCTION CHECK
# =============================================================================

section(
    "4. STRICT HISTORICAL FEATURE-ENGINEERING REPRODUCTION CHECK"
)

year_index_slope, year_index_intercept = infer_year_index_relation(
    engineered_hist
)

print(
    "Inferred exact year_index transform:"
)

print(
    f"  year_index = {year_index_intercept:.12f} "
    f"+ ({year_index_slope:.12f} * year)"
)

rebuilt_hist = add_step07_features(
    combined_base=base,
    neighbors=neighbors,
    year_index_slope=year_index_slope,
    year_index_intercept=year_index_intercept,
)

historical_2025 = (
    engineered_hist[
        engineered_hist[YEAR_COL] == 2025
    ]
    .copy()
    .sort_values(GRID_COL)
    .reset_index(drop=True)
)

rebuilt_2025 = (
    rebuilt_hist[
        rebuilt_hist[YEAR_COL] == 2025
    ]
    .copy()
    .sort_values(GRID_COL)
    .reset_index(drop=True)
)

if historical_2025[GRID_COL].tolist() != rebuilt_2025[GRID_COL].tolist():
    raise RuntimeError(
        "2025 grid order/key mismatch in feature reconstruction."
    )

original_base_columns = set(base.columns)

engineered_columns = [
    column
    for column in engineered_hist.columns
    if (
        column not in original_base_columns
        and column not in [GRID_COL, YEAR_COL, TARGET_COL, "area_name"]
    )
]

reconstruction_rows = []

for column in engineered_columns:
    if column not in rebuilt_2025.columns:
        reconstruction_rows.append(
            {
                "feature": column,
                "status": "MISSING_REBUILT",
                "max_abs_error": np.nan,
            }
        )
        continue

    expected = pd.to_numeric(
        historical_2025[column],
        errors="coerce",
    ).to_numpy(dtype=float)

    actual = pd.to_numeric(
        rebuilt_2025[column],
        errors="coerce",
    ).to_numpy(dtype=float)

    same_nan = (
        np.isnan(expected)
        == np.isnan(actual)
    ).all()

    finite_mask = (
        np.isfinite(expected)
        & np.isfinite(actual)
    )

    if finite_mask.any():
        max_error = float(
            np.max(
                np.abs(
                    expected[finite_mask]
                    - actual[finite_mask]
                )
            )
        )
    else:
        max_error = 0.0

    status = (
        "PASS"
        if same_nan
        and max_error <= RECONSTRUCTION_TOL
        else "FAIL"
    )

    reconstruction_rows.append(
        {
            "feature": column,
            "status": status,
            "max_abs_error": max_error,
        }
    )

reconstruction_audit = pd.DataFrame(
    reconstruction_rows
)

reconstruction_audit.to_csv(
    FEATURE_RECON_AUDIT_PATH,
    index=False,
)

failed_reconstruction = reconstruction_audit[
    reconstruction_audit["status"] != "PASS"
]

print(
    "Largest historical 2025 reconstruction errors:"
)

print(
    reconstruction_audit.sort_values(
        "max_abs_error",
        ascending=False,
    )
    .head(20)
    .to_string(index=False)
)

if not failed_reconstruction.empty:
    print(
        "\nFAILED reconstructed features:"
    )

    print(
        failed_reconstruction.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Historical Step-07 feature reconstruction failed. "
        "Do not generate future LST until the exact feature logic is reproduced."
    )

print(
    f"\nPASS: all {len(reconstruction_audit)} engineered features "
    f"reproduce historical 2025 within tolerance {RECONSTRUCTION_TOL}."
)


# =============================================================================
# 5. GENERATE FROZEN 2026-2030 BASE-PREDICTOR FORECASTS
# =============================================================================

section(
    "5. GENERATE 2026-2030 BASE-PREDICTOR FORECASTS"
)

base_sorted = (
    base
    .sort_values(
        [GRID_COL, YEAR_COL]
    )
    .reset_index(drop=True)
)

grid_order = sorted(
    base_sorted[GRID_COL]
    .unique()
    .tolist()
)

latest_2025 = (
    base_sorted[
        base_sorted[YEAR_COL] == 2025
    ]
    .copy()
    .set_index(GRID_COL)
    .loc[grid_order]
)

future_year_frames = []

for future_year in FUTURE_YEARS:
    frame = pd.DataFrame(
        {
            GRID_COL: grid_order,
            YEAR_COL: future_year,
        }
    ).set_index(
        GRID_COL
    )

    for predictor, method in selected_methods.items():
        wide = (
            base_sorted.pivot(
                index=GRID_COL,
                columns=YEAR_COL,
                values=predictor,
            )
            .loc[
                grid_order,
                EXPECTED_HIST_YEARS,
            ]
        )

        prediction = forecast_dynamic_predictor(
            wide=wide,
            method=method,
            target_year=future_year,
        )

        frame[predictor] = prediction

    for predictor in STATIC_PREDICTORS:
        frame[predictor] = (
            latest_2025[predictor]
            .astype(float)
        )

    for predictor in [
        "dist_road",
        "dist_main_road",
        "dist_water",
        "LCZ",
    ]:
        frame[predictor] = (
            latest_2025[predictor]
            .astype(float)
        )

    frame["area_name"] = (
        latest_2025["area_name"]
        .astype(str)
    )

    frame = (
        frame
        .reset_index()
    )

    frame[TARGET_COL] = np.nan

    future_year_frames.append(
        frame
    )

future_base = pd.concat(
    future_year_frames,
    ignore_index=True,
)

if len(future_base) != EXPECTED_FUTURE_ROWS:
    raise RuntimeError(
        f"Future base rows={len(future_base)}, expected={EXPECTED_FUTURE_ROWS}"
    )

if future_base.duplicated(
    [GRID_COL, YEAR_COL]
).any():
    raise RuntimeError(
        "Duplicate grid-year future base rows found."
    )

future_base.to_csv(
    FUTURE_BASE_PATH,
    index=False,
)

for year in FUTURE_YEARS:
    year_base_path = (
        FORECAST_DIR
        / f"28_future_base_predictors_{year}.csv"
    )

    future_base[
        future_base[YEAR_COL] == year
    ].to_csv(
        year_base_path,
        index=False,
    )

print(
    f"Future base rows: {len(future_base):,}"
)

print(
    f"Saved: {FUTURE_BASE_PATH}"
)

print(
    "\nFuture predictor annual means:"
)

print(
    future_base.groupby(YEAR_COL)[
        TEMPORAL_SOURCES
    ]
    .mean()
    .to_string()
)


# =============================================================================
# 6. RECONSTRUCT COMPLETE FUTURE ENGINEERED FEATURE MATRIX
# =============================================================================

section(
    "6. RECONSTRUCT COMPLETE FUTURE ENGINEERED FEATURE MATRIX"
)

combined_base = pd.concat(
    [
        base_sorted,
        future_base[
            base_sorted.columns
        ],
    ],
    ignore_index=True,
)

combined_engineered = add_step07_features(
    combined_base=combined_base,
    neighbors=neighbors,
    year_index_slope=year_index_slope,
    year_index_intercept=year_index_intercept,
)

future_engineered = (
    combined_engineered[
        combined_engineered[YEAR_COL]
        .isin(FUTURE_YEARS)
    ]
    .copy()
    .sort_values(
        [YEAR_COL, GRID_COL]
    )
    .reset_index(drop=True)
)

if len(future_engineered) != EXPECTED_FUTURE_ROWS:
    raise RuntimeError(
        "Future engineered row count mismatch."
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

if len(ordered_features) != 93:
    raise RuntimeError(
        f"Frozen feature count={len(ordered_features)}, expected=93"
    )

missing_features = [
    feature
    for feature in ordered_features
    if feature not in future_engineered.columns
]

if missing_features:
    raise RuntimeError(
        f"Future engineered data is missing frozen features: {missing_features}"
    )

X_future = future_engineered[
    ordered_features
].copy()

for column in ordered_features:
    X_future[column] = pd.to_numeric(
        X_future[column],
        errors="raise",
    )

if X_future.isna().any().any():
    missing = (
        X_future
        .isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ]

    raise RuntimeError(
        "Future 93-feature matrix contains missing values:\n"
        + missing.to_string()
    )

if not np.isfinite(
    X_future.to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Future 93-feature matrix contains non-finite values."
    )

# Export model features without duplicating the "year" column.
# "year" is already one of the frozen 93 LightGBM predictors, while it is also
# needed as metadata for yearly CSV filtering. Keep it only once.
export_predictor_columns = [
    feature
    for feature in ordered_features
    if feature != YEAR_COL
]

future_feature_export = pd.concat(
    [
        future_engineered[
            [
                GRID_COL,
                YEAR_COL,
                "area_name",
            ]
        ].reset_index(drop=True),
        X_future[
            export_predictor_columns
        ].reset_index(drop=True),
    ],
    axis=1,
)

duplicate_export_columns = (
    future_feature_export.columns[
        future_feature_export.columns.duplicated()
    ]
    .tolist()
)

if duplicate_export_columns:
    raise RuntimeError(
        "Duplicate columns in future feature export: "
        f"{duplicate_export_columns}"
    )

# 3 metadata columns + 92 remaining frozen predictors;
# frozen "year" is already represented once in metadata.
expected_export_columns = 3 + len(ordered_features) - 1

if future_feature_export.shape[1] != expected_export_columns:
    raise RuntimeError(
        "Unexpected future feature export column count: "
        f"actual={future_feature_export.shape[1]}, "
        f"expected={expected_export_columns}"
    )

future_feature_export.to_csv(
    FUTURE_FEATURE_PATH,
    index=False,
)

for year in FUTURE_YEARS:
    year_feature_path = (
        FORECAST_DIR
        / f"28_future_model_features_{year}.csv"
    )

    year_mask = (
        future_feature_export[YEAR_COL].astype(int)
        == int(year)
    )

    future_feature_export.loc[
        year_mask
    ].to_csv(
        year_feature_path,
        index=False,
    )

print(
    f"Future model-feature rows : {len(X_future):,}"
)

print(
    f"Frozen feature count      : {len(ordered_features)}"
)

print(
    f"Missing feature cells     : {int(X_future.isna().sum().sum())}"
)

print(
    f"Saved: {FUTURE_FEATURE_PATH}"
)


# =============================================================================
# 7. FORECAST ANNUAL KADUWELA THERMAL LEVEL
# =============================================================================

section(
    "7. FORECAST ANNUAL KADUWELA THERMAL LEVEL"
)

annual_history = pd.read_csv(
    ANNUAL_SERIES_PATH
)

annual_history[YEAR_COL] = pd.to_numeric(
    annual_history[YEAR_COL],
    errors="raise",
).astype(int)

if annual_method != "Expanding Mean":
    raise RuntimeError(
        "Step 28 currently expects the frozen Step-27 annual method Expanding Mean."
    )

historical_annual_mean = float(
    annual_history[
        "annual_mean_lst_c"
    ].mean()
)

annual_future = pd.DataFrame(
    {
        YEAR_COL: FUTURE_YEARS,
        "forecast_origin_year": 2025,
        "horizon_years": [
            year - 2025
            for year in FUTURE_YEARS
        ],
        "annual_level_method": annual_method,
        "forecast_annual_mean_lst_c": historical_annual_mean,
    }
)

annual_future.to_csv(
    ANNUAL_LEVEL_PATH,
    index=False,
)

print(
    annual_future.to_string(
        index=False
    )
)

print(
    f"\nSaved: {ANNUAL_LEVEL_PATH}"
)


# =============================================================================
# 8. FROZEN STEP-22B LIGHTGBM SPATIAL-ANOMALY INFERENCE
# =============================================================================

section(
    "8. FROZEN STEP-22B LIGHTGBM SPATIAL-ANOMALY INFERENCE"
)

model = joblib.load(
    MODEL_PATH
)

raw_anomaly = np.asarray(
    model.predict(
        X_future
    ),
    dtype=float,
)

prediction_table = (
    future_engineered[
        [
            GRID_COL,
            YEAR_COL,
            "area_name",
        ]
    ]
    .copy()
    .reset_index(drop=True)
)

prediction_table[
    "raw_spatial_anomaly_prediction_c"
] = raw_anomaly

prediction_table[
    "centered_spatial_anomaly_c"
] = (
    prediction_table[
        "raw_spatial_anomaly_prediction_c"
    ]
    - prediction_table.groupby(
        YEAR_COL
    )[
        "raw_spatial_anomaly_prediction_c"
    ]
    .transform("mean")
)

annual_level_lookup = (
    annual_future
    .set_index(YEAR_COL)[
        "forecast_annual_mean_lst_c"
    ]
    .to_dict()
)

prediction_table[
    "forecast_annual_mean_lst_c"
] = prediction_table[
    YEAR_COL
].map(
    annual_level_lookup
)

prediction_table[
    "predicted_lst_c"
] = (
    prediction_table[
        "forecast_annual_mean_lst_c"
    ]
    + prediction_table[
        "centered_spatial_anomaly_c"
    ]
)

prediction_table[
    "change_from_2025_actual_mean_c"
] = (
    prediction_table[
        "predicted_lst_c"
    ]
    - float(
        annual_history.loc[
            annual_history[YEAR_COL] == 2025,
            "annual_mean_lst_c",
        ].iloc[0]
    )
)

prediction_table.to_csv(
    FUTURE_LST_PATH,
    index=False,
)

for year in FUTURE_YEARS:
    year_lst_path = (
        FORECAST_DIR
        / f"28_future_grid_lst_predictions_{year}.csv"
    )

    prediction_table[
        prediction_table[YEAR_COL] == year
    ].to_csv(
        year_lst_path,
        index=False,
    )

print(
    f"Predicted future grid rows: {len(prediction_table):,}"
)

print(
    f"Saved: {FUTURE_LST_PATH}"
)


# =============================================================================
# 9. YEAR-LEVEL FORECAST AUDIT / SUMMARY
# =============================================================================

section(
    "9. FUTURE YEAR SUMMARY"
)

summary_rows = []

for year in FUTURE_YEARS:
    part = prediction_table[
        prediction_table[YEAR_COL] == year
    ]

    summary_rows.append(
        {
            YEAR_COL: year,
            "grid_count": int(len(part)),
            "annual_level_forecast_c": float(
                part[
                    "forecast_annual_mean_lst_c"
                ].iloc[0]
            ),
            "mean_raw_anomaly_c": float(
                part[
                    "raw_spatial_anomaly_prediction_c"
                ].mean()
            ),
            "mean_centered_anomaly_c": float(
                part[
                    "centered_spatial_anomaly_c"
                ].mean()
            ),
            "mean_final_predicted_lst_c": float(
                part[
                    "predicted_lst_c"
                ].mean()
            ),
            "median_final_predicted_lst_c": float(
                part[
                    "predicted_lst_c"
                ].median()
            ),
            "min_final_predicted_lst_c": float(
                part[
                    "predicted_lst_c"
                ].min()
            ),
            "max_final_predicted_lst_c": float(
                part[
                    "predicted_lst_c"
                ].max()
            ),
            "std_final_predicted_lst_c": float(
                part[
                    "predicted_lst_c"
                ].std(ddof=0)
            ),
        }
    )

year_summary = pd.DataFrame(
    summary_rows
)

year_summary.to_csv(
    YEAR_SUMMARY_PATH,
    index=False,
)

print(
    year_summary.to_string(
        index=False
    )
)


# =============================================================================
# 10. GEOJSON + REPORT-QUALITY MAPS
# =============================================================================

section(
    "10. SAVE FUTURE GEOJSON + MAPS"
)

grid_wgs84_clean = (
    grid_wgs84[
        [GRID_COL, "geometry"]
    ]
    .copy()
)

combined_geo_rows = []

boundary = load_boundary_wgs84()

global_vmin = float(
    prediction_table[
        "predicted_lst_c"
    ].min()
)

global_vmax = float(
    prediction_table[
        "predicted_lst_c"
    ].max()
)

for year in FUTURE_YEARS:
    year_pred = prediction_table[
        prediction_table[YEAR_COL] == year
    ].copy()

    year_base = future_base[
        future_base[YEAR_COL] == year
    ][
        [
            GRID_COL,
            *TEMPORAL_SOURCES,
        ]
    ].copy()

    merged = (
        grid_wgs84_clean
        .merge(
            year_pred,
            on=GRID_COL,
            how="left",
            validate="one_to_one",
        )
        .merge(
            year_base,
            on=GRID_COL,
            how="left",
            validate="one_to_one",
        )
    )

    if merged[
        "predicted_lst_c"
    ].isna().any():
        raise RuntimeError(
            f"{year}: GeoJSON merge produced missing predictions."
        )

    year_geojson_path = (
        MAP_DIR
        / f"28_future_heat_{year}_wgs84.geojson"
    )

    merged.to_file(
        year_geojson_path,
        driver="GeoJSON",
    )

    combined_geo_rows.append(
        merged
    )

    # Heat map
    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    merged.plot(
        column="predicted_lst_c",
        ax=ax,
        legend=True,
        vmin=global_vmin,
        vmax=global_vmax,
        edgecolor="none",
        legend_kwds={
            "label": "Predicted LST (°C)",
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
        f"Kaduwela Predicted Land Surface Temperature — {year}",
        fontsize=14,
        pad=12,
    )

    ax.set_axis_off()

    plt.tight_layout()

    map_png = (
        FIGURE_DIR
        / f"28_future_lst_map_{year}.png"
    )

    plt.savefig(
        map_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"{year}: saved GeoJSON + 300-dpi LST map"
    )

combined_geo = pd.concat(
    combined_geo_rows,
    ignore_index=True,
)

combined_geo = gpd.GeoDataFrame(
    combined_geo,
    geometry="geometry",
    crs=grid_wgs84.crs,
)

combined_geo.to_file(
    COMBINED_GEOJSON_PATH,
    driver="GeoJSON",
)

print(
    f"\nSaved combined GeoJSON: {COMBINED_GEOJSON_PATH}"
)


# =============================================================================
# 11. STRICT STEP-28 AUDIT
# =============================================================================

section(
    "11. STRICT STEP-28 AUDIT"
)

center_means = (
    prediction_table.groupby(
        YEAR_COL
    )[
        "centered_spatial_anomaly_c"
    ]
    .mean()
    .abs()
)

final_mean_error = []

for year in FUTURE_YEARS:
    part = prediction_table[
        prediction_table[YEAR_COL] == year
    ]

    final_mean = float(
        part["predicted_lst_c"].mean()
    )

    annual_level = float(
        part[
            "forecast_annual_mean_lst_c"
        ].iloc[0]
    )

    final_mean_error.append(
        abs(
            final_mean - annual_level
        )
    )

feature_finite = bool(
    np.isfinite(
        X_future.to_numpy(dtype=float)
    ).all()
)

model_unchanged = bool(
    sha256_file(MODEL_PATH)
    == EXPECTED_HASHES[
        "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
    ]
)

audit_rows = [
    {
        "check": "step27_audit",
        "status": "PASS",
        "details": "Step-27 strict audit verified fully PASS.",
    },
    {
        "check": "step27_strategy_hash",
        "status": "PASS",
        "details": EXPECTED_HASHES[
            "27_forecast_strategy_freeze_for_step28.json"
        ],
    },
    {
        "check": "production_model_hash_before_after",
        "status": (
            "PASS"
            if model_unchanged
            else "FAIL"
        ),
        "details": sha256_file(MODEL_PATH),
    },
    {
        "check": "historical_feature_reconstruction",
        "status": (
            "PASS"
            if failed_reconstruction.empty
            else "FAIL"
        ),
        "details": (
            f"features_checked={len(reconstruction_audit)}, "
            f"max_error={reconstruction_audit['max_abs_error'].max():.12e}"
        ),
    },
    {
        "check": "future_years",
        "status": (
            "PASS"
            if sorted(
                future_base[YEAR_COL]
                .unique()
                .tolist()
            )
            == FUTURE_YEARS
            else "FAIL"
        ),
        "details": str(FUTURE_YEARS),
    },
    {
        "check": "future_base_rows",
        "status": (
            "PASS"
            if len(future_base)
            == EXPECTED_FUTURE_ROWS
            else "FAIL"
        ),
        "details": (
            f"actual={len(future_base)}, "
            f"expected={EXPECTED_FUTURE_ROWS}"
        ),
    },
    {
        "check": "future_rows_per_year",
        "status": (
            "PASS"
            if (
                future_base.groupby(YEAR_COL)[GRID_COL]
                .nunique()
                == EXPECTED_GRIDS
            ).all()
            else "FAIL"
        ),
        "details": f"{EXPECTED_GRIDS} grids/year",
    },
    {
        "check": "frozen_feature_count",
        "status": (
            "PASS"
            if len(ordered_features) == 93
            else "FAIL"
        ),
        "details": f"actual={len(ordered_features)}, expected=93",
    },
    {
        "check": "future_feature_matrix_finite",
        "status": (
            "PASS"
            if feature_finite
            else "FAIL"
        ),
        "details": (
            f"rows={len(X_future)}, cols={X_future.shape[1]}"
        ),
    },
    {
        "check": "future_feature_matrix_no_missing",
        "status": (
            "PASS"
            if not X_future.isna().any().any()
            else "FAIL"
        ),
        "details": (
            f"missing_cells={int(X_future.isna().sum().sum())}"
        ),
    },
    {
        "check": "future_feature_export_unique_columns",
        "status": (
            "PASS"
            if not future_feature_export.columns.duplicated().any()
            else "FAIL"
        ),
        "details": (
            f"export_columns={future_feature_export.shape[1]}, "
            f"duplicate_columns={int(future_feature_export.columns.duplicated().sum())}"
        ),
    },
    {
        "check": "centered_anomaly_zero_mean_each_year",
        "status": (
            "PASS"
            if float(center_means.max())
            <= ANOMALY_CENTER_TOL
            else "FAIL"
        ),
        "details": (
            f"max_abs_year_mean={float(center_means.max()):.12e}"
        ),
    },
    {
        "check": "final_year_mean_equals_annual_level",
        "status": (
            "PASS"
            if max(final_mean_error)
            <= ANOMALY_CENTER_TOL
            else "FAIL"
        ),
        "details": (
            f"max_abs_error={max(final_mean_error):.12e}"
        ),
    },
    {
        "check": "annual_method_frozen",
        "status": (
            "PASS"
            if annual_method == "Expanding Mean"
            else "FAIL"
        ),
        "details": annual_method,
    },
    {
        "check": "dynamic_methods_match_step27",
        "status": (
            "PASS"
            if selected_methods
            == EXPECTED_STEP27_SELECTION[
                "dynamic_predictors"
            ]
            else "FAIL"
        ),
        "details": json.dumps(selected_methods),
    },
    {
        "check": "production_model_retraining",
        "status": "PASS",
        "details": "None. Existing Step-22B model loaded for inference only.",
    },
    {
        "check": "future_rows_treated_as_observed",
        "status": "PASS",
        "details": "No. 2026-2030 rows are explicitly forecast/pseudo-future inputs.",
    },
    {
        "check": "engineered_features_independently_forecast",
        "status": "PASS",
        "details": "No. They were reconstructed from base predictor history.",
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
    audit_df["status"] != "PASS"
).any():
    raise RuntimeError(
        "One or more Step-28 strict audit checks failed."
    )


# =============================================================================
# 12. METHODOLOGY + REPORT
# =============================================================================

section(
    "12. SAVE STEP-28 METHODOLOGY + REPORT"
)

methodology = {
    "step": 28,
    "analysis": "Operational future heat prediction 2026-2030",
    "forecast_origin_year": 2025,
    "future_years": FUTURE_YEARS,
    "annual_level_method": annual_method,
    "dynamic_predictor_methods": selected_methods,
    "static_predictors": STATIC_PREDICTORS,
    "structural_predictors": STRUCTURAL_PREDICTORS,
    "predictor_forecast_semantics": (
        "Direct H1-H5 forecasts from observed history through 2025, matching "
        "Step-27 validation. Forecast future values are subsequently used to "
        "reconstruct future temporal features."
    ),
    "feature_engineering": {
        "historical_reproduction_required": True,
        "reconstruction_tolerance": RECONSTRUCTION_TOL,
        "temporal_sources": TEMPORAL_SOURCES,
        "history_rule": "lag/history features are based on prior predictor values",
        "spatial_rule": "same-year predictor-only Queen neighbour means/contrasts",
        "target_derived_spatial_features": False,
    },
    "production_model": {
        "artifact": str(MODEL_PATH),
        "sha256": EXPECTED_HASHES[
            "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
        ],
        "retrained_in_step28": False,
        "feature_count": 93,
    },
    "spatial_anomaly_centering": (
        "Subtract the mean raw anomaly prediction across all 2,194 grids "
        "separately in every future year."
    ),
    "final_equation": (
        "predicted grid LST = forecast annual Kaduwela thermal level "
        "+ centered spatial anomaly"
    ),
    "uncertainty_note": (
        "Step 28 is point forecasting. Formal calibrated forecast uncertainty "
        "is handled in the later uncertainty stage; predictor and annual-level "
        "forecast errors can compound."
    ),
    "interpretation_note": (
        "2026-2030 results are model-based projections under the frozen "
        "historical forecasting assumptions, not observed future temperatures."
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
    "STEP 28 - OPERATIONAL 2026-2030 FUTURE HEAT PREDICTION",
    "=" * 118,
    "",
    "FROZEN STRATEGY",
    "-" * 118,
    f"Annual level: {annual_method}",
    json.dumps(
        selected_methods,
        indent=2,
    ),
    "",
    "ANNUAL THERMAL-LEVEL FORECAST",
    "-" * 118,
    annual_future.to_string(index=False),
    "",
    "FUTURE YEAR SUMMARY",
    "-" * 118,
    year_summary.to_string(index=False),
    "",
    "FEATURE RECONSTRUCTION",
    "-" * 118,
    (
        f"Historical 2025 engineered features reproduced: "
        f"{len(reconstruction_audit)} checked; "
        f"max absolute error="
        f"{reconstruction_audit['max_abs_error'].max():.12e}"
    ),
    "",
    "INTERPRETATION",
    "-" * 118,
    (
        "Future grid LST is the frozen annual Kaduwela thermal-level forecast "
        "plus the centered Step-22B production LightGBM spatial anomaly."
    ),
    (
        "The production LightGBM was not retrained on 2026-2030 pseudo-future data."
    ),
    (
        "Future predictor values are forecasts. Therefore future LST inherits "
        "uncertainty from both predictor forecasting and annual-level forecasting."
    ),
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(
    f"Saved: {METHOD_PATH}"
)

print(
    f"Saved: {REPORT_PATH}"
)


# =============================================================================
# 13. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "13. REPRODUCIBILITY HASHES"
)

hash_files = [
    BASE_PATH,
    ENGINEERED_HIST_PATH,
    GRID_UTM_PATH,
    GRID_WGS84_PATH,
    FEATURE_LIST_PATH,
    MODEL_PATH,
    STEP27_FREEZE_PATH,
    STEP27_AUDIT_PATH,
    FUTURE_BASE_PATH,
    FUTURE_FEATURE_PATH,
    ANNUAL_LEVEL_PATH,
    FUTURE_LST_PATH,
    YEAR_SUMMARY_PATH,
    FEATURE_RECON_AUDIT_PATH,
    COMBINED_GEOJSON_PATH,
    METHOD_PATH,
    AUDIT_PATH,
    REPORT_PATH,
]

for year in FUTURE_YEARS:
    hash_files.append(
        FORECAST_DIR / f"28_future_base_predictors_{year}.csv"
    )

    hash_files.append(
        FORECAST_DIR / f"28_future_model_features_{year}.csv"
    )

    hash_files.append(
        FORECAST_DIR / f"28_future_grid_lst_predictions_{year}.csv"
    )

    hash_files.append(
        MAP_DIR / f"28_future_heat_{year}_wgs84.geojson"
    )

    hash_files.append(
        FIGURE_DIR / f"28_future_lst_map_{year}.png"
    )

hash_rows = []

for path in hash_files:
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
    "STEP 28 COMPLETED SUCCESSFULLY"
)

print(
    "Operational 2026-2030 future heat prediction completed.\n"
    "Step-27 frozen annual and predictor forecasting strategies were used unchanged.\n"
    f"Future base-predictor rows: {len(future_base):,}.\n"
    f"Future frozen-feature rows: {len(X_future):,} x {X_future.shape[1]} predictors.\n"
    "Historical Step-07 feature engineering was reproduced before future inference.\n"
    "The frozen Step-22B LightGBM was used for inference only and was not retrained.\n"
    "Raw future spatial anomalies were centered independently within each future year.\n"
    "Final future grid LST = annual level + centered spatial anomaly.\n"
    "Combined and separate yearly 2026-2030 CSVs, GeoJSON and 300-dpi map outputs were saved.\n"
    "\nNEXT: review Step-28 outputs carefully before Step 28B future SHAP."
)
