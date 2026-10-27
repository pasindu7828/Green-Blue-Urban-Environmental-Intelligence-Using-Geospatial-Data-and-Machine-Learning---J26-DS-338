"""
DYNAMIC 5-YEAR FORECAST - STEP 05
=================================
Rebuild exact Step-07 engineered features for Y+1..Y+5, run the
Step-03 versioned LightGBM spatial-anomaly model, center the anomaly within
each future year, add the Step-04 annual thermal level, and export CSV/GeoJSON.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError as exc:
    raise ImportError("geopandas is required: pip install geopandas") from exc

try:
    from libpysal.weights import Queen
except ImportError as exc:
    raise ImportError("libpysal is required: pip install libpysal") from exc


# =============================================================================
# PATHS / SETTINGS
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]

OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"
MAP_DIR = OUTPUT_ROOT / "maps"
REPORT_DIR = OUTPUT_ROOT / "reports"

VALIDATED_HISTORY_PATH = OUTPUT_ROOT / "01_validated_history.csv"
STEP01_SUMMARY_PATH = REPORT_DIR / "01_upload_validation_summary.json"
STEP03_MANIFEST_PATH = REPORT_DIR / "03_dynamic_model_manifest.json"
DYNAMIC_ENGINEERED_HISTORY_PATH = OUTPUT_ROOT / "03_dynamic_engineered_history.csv"
CANONICAL_ENGINEERED_HISTORY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "kaduwela_engineered_features.csv"
)

FEATURE_LIST_PATH = PROJECT_ROOT / "outputs" / "reports" / "21_final_feature_list.csv"
GRID_UTM_PATH = PROJECT_ROOT / "data" / "processed" / "kaduwela_grid_geometry_utm44n.geojson"
GRID_WGS84_PATH = PROJECT_ROOT / "data" / "processed" / "kaduwela_grid_geometry_wgs84.geojson"

REFERENCE_FEATURE_PATH = PROJECT_ROOT / "outputs" / "forecasting" / "28_future_model_features_2026_2030.csv"
REFERENCE_PREDICTION_PATH = PROJECT_ROOT / "outputs" / "forecasting" / "28_future_grid_lst_predictions_2026_2030.csv"

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET_COL = "LST_C"
EXPECTED_GRIDS = 2194
EXPECTED_FEATURES = 93
FORECAST_HORIZON = 5
REPRO_TOL = 1e-8
ANOMALY_CENTER_TOL = 1e-10

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


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 125)
    print(title)
    print("=" * 125)


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def detect_feature_column(feature_df):
    if {"feature_order", "feature"}.issubset(feature_df.columns):
        values = feature_df["feature"].dropna().astype(str).tolist()
        if len(values) == EXPECTED_FEATURES:
            return "feature"

    for conventional in ["feature", "feature_name", "predictor"]:
        if conventional in feature_df.columns:
            values = feature_df[conventional].dropna().astype(str).tolist()
            if len(values) == EXPECTED_FEATURES:
                return conventional

    for column in feature_df.columns:
        values = feature_df[column].dropna().astype(str).tolist()
        if len(values) == EXPECTED_FEATURES and len(set(values)) == EXPECTED_FEATURES:
            return column

    raise RuntimeError("Could not identify the ordered 93-feature column.")


def infer_year_index_relation(engineered):
    pairs = (
        engineered[[YEAR_COL, "year_index"]]
        .drop_duplicates()
        .sort_values(YEAR_COL)
    )

    x = pairs[YEAR_COL].to_numpy(dtype=float)
    y = pairs["year_index"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    reconstructed = intercept + slope * x
    max_error = float(np.max(np.abs(reconstructed - y)))

    if max_error > 1e-10:
        raise RuntimeError("year_index relation could not be reproduced exactly.")

    return float(slope), float(intercept)


def build_queen_neighbors(grid_utm):
    ordered = (
        grid_utm[[GRID_COL, "geometry"]]
        .copy()
        .sort_values(GRID_COL)
        .reset_index(drop=True)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights = Queen.from_dataframe(
            ordered,
            ids=ordered[GRID_COL].astype(str).tolist(),
            silence_warnings=True,
        )

    neighbors = {
        str(grid_id): [str(n) for n in nbrs]
        for grid_id, nbrs in weights.neighbors.items()
    }

    isolates = [g for g, nbrs in neighbors.items() if len(nbrs) == 0]
    if isolates:
        raise RuntimeError(
            f"Queen adjacency produced {len(isolates)} isolated grids."
        )

    return neighbors


def add_neighbor_features(df, neighbors):
    out = df.copy()

    for source in TEMPORAL_SOURCES:
        mean_col = f"{source}_nbr_mean"
        contrast_col = f"{source}_nbr_contrast"
        result = np.empty(len(out), dtype=float)

        for _, index in out.groupby(YEAR_COL).groups.items():
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
                means.append(float(np.mean(nbr_values)))

            result[np.asarray(index)] = np.asarray(means, dtype=float)

        out[mean_col] = result
        out[contrast_col] = out[source] - out[mean_col]

    return out


def hist3_slope(values):
    arr = np.asarray(values, dtype=float)

    if len(arr) != 3 or np.isnan(arr).any():
        return np.nan

    return float((arr[2] - arr[0]) / 2.0)


def add_step07_features(
    combined_base,
    neighbors,
    year_index_slope,
    year_index_intercept,
):
    out = (
        combined_base
        .copy()
        .sort_values([GRID_COL, YEAR_COL])
        .reset_index(drop=True)
    )

    # Deterministic row-wise features
    out["year_index"] = (
        year_index_intercept
        + year_index_slope * out[YEAR_COL].astype(float)
    )

    out["vegetation_deficit"] = 1.0 - out["green_mask"]
    out["ndbi_minus_ndvi"] = out["NDBI"] - out["NDVI"]
    out["ndbi_x_vegetation_deficit"] = (
        out["NDBI"] * out["vegetation_deficit"]
    )
    out["ndvi_x_green_mask"] = out["NDVI"] * out["green_mask"]
    out["ndbi_x_building_mask"] = out["NDBI"] * out["building_mask"]
    out["nightlights_x_building_mask"] = (
        out["NightLights"] * out["building_mask"]
    )

    for source in ["dist_road", "dist_main_road", "dist_water"]:
        out[f"log1p_{source}"] = np.log1p(out[source].astype(float))

    # Strictly-past temporal features
    grouped = out.groupby(GRID_COL, sort=False)

    for source in TEMPORAL_SOURCES:
        out[f"{source}_lag1"] = grouped[source].shift(1)
        out[f"{source}_lag2"] = grouped[source].shift(2)

        shifted = grouped[source].shift(1)

        out[f"{source}_hist3_mean"] = (
            shifted
            .groupby(out[GRID_COL], sort=False)
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .sort_index()
        )

        out[f"{source}_hist3_std"] = (
            shifted
            .groupby(out[GRID_COL], sort=False)
            .rolling(window=3, min_periods=2)
            .std(ddof=1)
            .reset_index(level=0, drop=True)
            .sort_index()
        )

        out[f"{source}_hist3_slope"] = (
            shifted
            .groupby(out[GRID_COL], sort=False)
            .rolling(window=3, min_periods=3)
            .apply(hist3_slope, raw=True)
            .reset_index(level=0, drop=True)
            .sort_index()
        )

        out[f"{source}_delta1"] = out[source] - out[f"{source}_lag1"]

    # Same-year predictor-only Queen-neighbour context
    out = add_neighbor_features(out, neighbors)

    return out


def max_numeric_diff_by_key(left, right, columns):
    left_sorted = (
        left[[GRID_COL, YEAR_COL] + columns]
        .copy()
        .sort_values([YEAR_COL, GRID_COL])
        .reset_index(drop=True)
    )

    right_sorted = (
        right[[GRID_COL, YEAR_COL] + columns]
        .copy()
        .sort_values([YEAR_COL, GRID_COL])
        .reset_index(drop=True)
    )

    left_keys = left_sorted[[GRID_COL, YEAR_COL]].astype(str)
    right_keys = right_sorted[[GRID_COL, YEAR_COL]].astype(str)

    if not left_keys.equals(right_keys):
        raise RuntimeError("Key mismatch in baseline reproduction comparison.")

    max_error = 0.0

    for column in columns:
        a = pd.to_numeric(left_sorted[column], errors="raise").to_numpy(dtype=float)
        b = pd.to_numeric(right_sorted[column], errors="raise").to_numpy(dtype=float)
        max_error = max(max_error, float(np.max(np.abs(a - b))))

    return max_error


# =============================================================================
# MAIN
# =============================================================================

def main():
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    section("DYNAMIC 5-YEAR FORECAST - STEP 05 GRID LST FORECAST GENERATION - PRECISION FIX V2")

    required = [
        VALIDATED_HISTORY_PATH,
        STEP01_SUMMARY_PATH,
        STEP03_MANIFEST_PATH,
        DYNAMIC_ENGINEERED_HISTORY_PATH,
        CANONICAL_ENGINEERED_HISTORY_PATH,
        FEATURE_LIST_PATH,
        GRID_UTM_PATH,
        GRID_WGS84_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    step01 = load_json(STEP01_SUMMARY_PATH)
    latest_year = int(step01["latest_observed_year"])
    future_years = list(
        range(latest_year + 1, latest_year + FORECAST_HORIZON + 1)
    )
    expected_future_rows = EXPECTED_GRIDS * len(future_years)

    future_base_path = (
        PREDICTION_DIR
        / f"04_future_base_predictors_{future_years[0]}_{future_years[-1]}.csv"
    )
    annual_level_path = (
        PREDICTION_DIR
        / f"04_future_annual_thermal_level_{future_years[0]}_{future_years[-1]}.csv"
    )

    for path in [future_base_path, annual_level_path]:
        if not path.exists():
            raise FileNotFoundError(f"Step 04 output missing: {path}")

    step03_manifest = load_json(STEP03_MANIFEST_PATH)
    dynamic_model_path = Path(step03_manifest["updated_model"])

    if not dynamic_model_path.exists():
        raise FileNotFoundError(
            f"Step 03 versioned model missing: {dynamic_model_path}"
        )

    print(f"Observed history : 2015-{latest_year}")
    print(f"Future years     : {future_years}")
    print(f"Model            : {dynamic_model_path.name}")

    # -------------------------------------------------------------------------
    section("1. LOAD OBSERVED + PROJECTED BASE INPUTS")

    historical = pd.read_csv(
        VALIDATED_HISTORY_PATH,
        dtype={GRID_COL: str, "area_name": str},
        low_memory=False,
        float_precision="round_trip",
    )

    future_base = pd.read_csv(
        future_base_path,
        dtype={GRID_COL: str, "area_name": str},
        low_memory=False,
        float_precision="round_trip",
    )

    historical[YEAR_COL] = pd.to_numeric(
        historical[YEAR_COL], errors="raise"
    ).astype(int)
    future_base[YEAR_COL] = pd.to_numeric(
        future_base[YEAR_COL], errors="raise"
    ).astype(int)

    if len(future_base) != expected_future_rows:
        raise RuntimeError(
            f"Future base rows={len(future_base)}, expected={expected_future_rows}"
        )

    if sorted(future_base[YEAR_COL].unique().tolist()) != future_years:
        raise RuntimeError("Step 04 future-year window mismatch.")

    base_columns = historical.columns.tolist()
    missing_base_columns = [c for c in base_columns if c not in future_base.columns]
    if missing_base_columns:
        raise RuntimeError(
            f"Future base missing canonical columns: {missing_base_columns}"
        )

    future_base = future_base[base_columns].copy()

    print(f"Historical rows  : {len(historical):,}")
    print(f"Future base rows : {len(future_base):,}")

    # -------------------------------------------------------------------------
    section("2. BUILD STEP-07 QUEEN NEIGHBOUR STRUCTURE")

    grid_utm = gpd.read_file(GRID_UTM_PATH)
    grid_utm[GRID_COL] = grid_utm[GRID_COL].astype(str)

    neighbors = build_queen_neighbors(grid_utm)
    neighbor_counts = np.asarray(
        [len(neighbors[str(g)]) for g in sorted(neighbors)],
        dtype=int,
    )

    print(f"Queen neighbour grids : {len(neighbors):,}")
    print(f"Min neighbours        : {neighbor_counts.min()}")
    print(f"Median neighbours     : {np.median(neighbor_counts):.1f}")
    print(f"Max neighbours        : {neighbor_counts.max()}")

    # -------------------------------------------------------------------------
    section("3. REBUILD EXACT STEP-07 FUTURE ENGINEERED FEATURES")

    # IMPORTANT: infer the structural year_index transform from the same
    # canonical Step-07 engineered history used by the original Step-28.
    # Re-inferring it from Step-03's regenerated CSV performs a second
    # floating-point regression and introduces a tiny (~1e-9) drift that can
    # flip LightGBM tree thresholds.
    canonical_engineered_history = pd.read_csv(
        CANONICAL_ENGINEERED_HISTORY_PATH,
        dtype={GRID_COL: str, "area_name": str},
        low_memory=False,
        float_precision="round_trip",
    )

    year_index_slope, year_index_intercept = infer_year_index_relation(
        canonical_engineered_history
    )

    print(
        "year_index relation : "
        f"{year_index_intercept:.12f} + "
        f"({year_index_slope:.12f} * year)"
    )

    combined_base = pd.concat(
        [historical, future_base],
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
            combined_engineered[YEAR_COL].isin(future_years)
        ]
        .copy()
        .sort_values([YEAR_COL, GRID_COL])
        .reset_index(drop=True)
    )

    if len(future_engineered) != expected_future_rows:
        raise RuntimeError("Future engineered row count mismatch.")

    feature_df = pd.read_csv(FEATURE_LIST_PATH)
    feature_col = detect_feature_column(feature_df)
    ordered_features = (
        feature_df[feature_col]
        .dropna()
        .astype(str)
        .tolist()
    )

    if len(ordered_features) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Frozen feature count={len(ordered_features)}, expected={EXPECTED_FEATURES}"
        )

    missing_features = [
        feature
        for feature in ordered_features
        if feature not in future_engineered.columns
    ]
    if missing_features:
        raise RuntimeError(
            f"Future engineered data missing frozen features: {missing_features}"
        )

    X_future = future_engineered[ordered_features].copy()

    for column in ordered_features:
        X_future[column] = pd.to_numeric(X_future[column], errors="raise")

    missing_cells = int(X_future.isna().sum().sum())
    if missing_cells:
        raise RuntimeError(
            f"Future 93-feature matrix contains {missing_cells} missing cells."
        )

    if not np.isfinite(X_future.to_numpy(dtype=float)).all():
        raise RuntimeError("Future 93-feature matrix contains non-finite values.")

    export_predictor_columns = [
        feature
        for feature in ordered_features
        if feature != YEAR_COL
    ]

    future_feature_export = pd.concat(
        [
            future_engineered[[GRID_COL, YEAR_COL, "area_name"]]
            .reset_index(drop=True),
            X_future[export_predictor_columns]
            .reset_index(drop=True),
        ],
        axis=1,
    )

    if future_feature_export.columns.duplicated().any():
        raise RuntimeError("Duplicate columns found in future feature export.")

    future_feature_path = (
        PREDICTION_DIR
        / f"05_future_model_features_{future_years[0]}_{future_years[-1]}.csv"
    )
    future_feature_export.to_csv(future_feature_path, index=False)

    for year in future_years:
        future_feature_export[
            future_feature_export[YEAR_COL] == year
        ].to_csv(
            PREDICTION_DIR / f"05_future_model_features_{year}.csv",
            index=False,
        )

    print(f"Future engineered rows : {len(future_engineered):,}")
    print(f"Frozen feature count   : {len(ordered_features)}")
    print(f"Missing feature cells  : {missing_cells}")
    print(f"Saved                  : {future_feature_path}")

    # -------------------------------------------------------------------------
    section("4. RUN VERSIONED LIGHTGBM SPATIAL-ANOMALY INFERENCE")

    model = joblib.load(dynamic_model_path)

    raw_anomaly = np.asarray(
        model.predict(X_future),
        dtype=float,
    )

    if not np.isfinite(raw_anomaly).all():
        raise RuntimeError("Model produced non-finite future predictions.")

    prediction_table = (
        future_engineered[[GRID_COL, YEAR_COL, "area_name"]]
        .copy()
        .reset_index(drop=True)
    )

    prediction_table["forecast_origin_year"] = latest_year
    prediction_table["horizon_years"] = (
        prediction_table[YEAR_COL] - latest_year
    )
    prediction_table["raw_spatial_anomaly_prediction_c"] = raw_anomaly
    prediction_table["centered_spatial_anomaly_c"] = (
        prediction_table["raw_spatial_anomaly_prediction_c"]
        - prediction_table.groupby(YEAR_COL)[
            "raw_spatial_anomaly_prediction_c"
        ].transform("mean")
    )

    center_means = (
        prediction_table.groupby(YEAR_COL)["centered_spatial_anomaly_c"]
        .mean()
        .abs()
    )
    max_center_mean = float(center_means.max())

    if max_center_mean > ANOMALY_CENTER_TOL:
        raise RuntimeError(
            "Centered spatial anomalies are not zero mean by future year."
        )

    annual_future = pd.read_csv(annual_level_path)
    annual_future[YEAR_COL] = pd.to_numeric(
        annual_future[YEAR_COL], errors="raise"
    ).astype(int)

    annual_lookup = (
        annual_future
        .set_index(YEAR_COL)["forecast_annual_mean_lst_c"]
        .to_dict()
    )

    prediction_table["forecast_annual_mean_lst_c"] = (
        prediction_table[YEAR_COL].map(annual_lookup)
    )

    if prediction_table["forecast_annual_mean_lst_c"].isna().any():
        raise RuntimeError("Missing annual thermal level for a future year.")

    prediction_table["predicted_lst_c"] = (
        prediction_table["forecast_annual_mean_lst_c"]
        + prediction_table["centered_spatial_anomaly_c"]
    )

    origin_actual_mean = float(
        historical.loc[
            historical[YEAR_COL] == latest_year,
            TARGET_COL,
        ].mean()
    )

    prediction_table[
        f"change_from_{latest_year}_actual_mean_c"
    ] = (
        prediction_table["predicted_lst_c"]
        - origin_actual_mean
    )

    future_prediction_path = (
        PREDICTION_DIR
        / f"05_dynamic_grid_lst_predictions_{future_years[0]}_{future_years[-1]}.csv"
    )
    prediction_table.to_csv(future_prediction_path, index=False)

    for year in future_years:
        prediction_table[
            prediction_table[YEAR_COL] == year
        ].to_csv(
            PREDICTION_DIR / f"05_dynamic_grid_lst_predictions_{year}.csv",
            index=False,
        )

    print(f"Future prediction rows : {len(prediction_table):,}")
    print(f"Max centered year mean : {max_center_mean:.12e} °C")
    print(f"Saved                  : {future_prediction_path}")

    # -------------------------------------------------------------------------
    section("5. FUTURE YEAR SUMMARY")

    year_summary = (
        prediction_table.groupby(YEAR_COL)
        .agg(
            grid_count=(GRID_COL, "nunique"),
            annual_level_forecast_c=("forecast_annual_mean_lst_c", "first"),
            mean_raw_anomaly_c=("raw_spatial_anomaly_prediction_c", "mean"),
            mean_centered_anomaly_c=("centered_spatial_anomaly_c", "mean"),
            mean_final_predicted_lst_c=("predicted_lst_c", "mean"),
            median_final_predicted_lst_c=("predicted_lst_c", "median"),
            min_final_predicted_lst_c=("predicted_lst_c", "min"),
            max_final_predicted_lst_c=("predicted_lst_c", "max"),
        )
        .reset_index()
    )

    year_summary_path = (
        PREDICTION_DIR
        / f"05_dynamic_future_year_summary_{future_years[0]}_{future_years[-1]}.csv"
    )
    year_summary.to_csv(year_summary_path, index=False)
    print(year_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    section("6. ORIGINAL 2015-2025 BASELINE REPRODUCTION CHECK")

    feature_repro_status = "NOT_APPLICABLE"
    forecast_repro_status = "NOT_APPLICABLE"
    feature_max_error = None
    forecast_max_error = None

    if latest_year == 2025:
        if not REFERENCE_FEATURE_PATH.exists():
            raise FileNotFoundError(
                f"Reference Step-28 feature matrix missing: {REFERENCE_FEATURE_PATH}"
            )
        if not REFERENCE_PREDICTION_PATH.exists():
            raise FileNotFoundError(
                f"Reference Step-28 forecast file missing: {REFERENCE_PREDICTION_PATH}"
            )

        reference_features = pd.read_csv(
            REFERENCE_FEATURE_PATH,
            dtype={GRID_COL: str, "area_name": str},
            low_memory=False,
            float_precision="round_trip",
        )

        compare_features = [
            feature
            for feature in ordered_features
            if feature != YEAR_COL
        ]

        feature_max_error = max_numeric_diff_by_key(
            future_feature_export,
            reference_features,
            compare_features,
        )

        feature_repro_status = (
            "PASS" if feature_max_error <= REPRO_TOL else "FAIL"
        )

        reference_predictions = pd.read_csv(
            REFERENCE_PREDICTION_PATH,
            dtype={GRID_COL: str, "area_name": str},
            low_memory=False,
            float_precision="round_trip",
        )

        forecast_compare_columns = [
            "raw_spatial_anomaly_prediction_c",
            "centered_spatial_anomaly_c",
            "forecast_annual_mean_lst_c",
            "predicted_lst_c",
        ]

        forecast_max_error = max_numeric_diff_by_key(
            prediction_table,
            reference_predictions,
            forecast_compare_columns,
        )

        forecast_repro_status = (
            "PASS" if forecast_max_error <= REPRO_TOL else "FAIL"
        )

        print(
            "Max future-feature reproduction error : "
            f"{feature_max_error:.12e}"
        )
        print(
            "Max forecast reproduction error       : "
            f"{forecast_max_error:.12e} °C"
        )

        if feature_repro_status != "PASS" or forecast_repro_status != "PASS":
            raise RuntimeError(
                "Dynamic Step 05 does not reproduce the validated original "
                "2026-2030 Step-28 outputs within tolerance."
            )

        print(
            "PASS: dynamic feature reconstruction and forecasts exactly reproduce "
            "the validated original Step-28 baseline within tolerance."
        )
    else:
        print(
            "Reference reproduction check not applicable because observed history "
            "extends beyond 2025."
        )

    # -------------------------------------------------------------------------
    section("7. EXPORT FIVE WGS84 FORECAST GEOJSON LAYERS")

    grid_wgs84 = gpd.read_file(GRID_WGS84_PATH)
    grid_wgs84[GRID_COL] = grid_wgs84[GRID_COL].astype(str)

    if grid_wgs84[GRID_COL].nunique() != EXPECTED_GRIDS:
        raise RuntimeError("WGS84 grid geometry count mismatch.")

    grid_wgs84_clean = grid_wgs84[[GRID_COL, "geometry"]].copy()
    map_files = []
    combined_parts = []

    for year in future_years:
        year_pred = prediction_table[
            prediction_table[YEAR_COL] == year
        ].copy()

        year_base = future_base[
            future_base[YEAR_COL] == year
        ][[GRID_COL] + TEMPORAL_SOURCES].copy()

        year_map = (
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

        if year_map["predicted_lst_c"].isna().any():
            raise RuntimeError(
                f"Geometry/prediction join failed for {year}."
            )

        year_geojson_path = (
            MAP_DIR / f"05_dynamic_future_heat_{year}_wgs84.geojson"
        )
        year_map.to_file(year_geojson_path, driver="GeoJSON")

        combined_parts.append(year_map)
        map_files.append(
            {
                "year": int(year),
                "path": str(year_geojson_path),
                "rows": int(len(year_map)),
                "crs": str(year_map.crs),
            }
        )

        print(
            f"{year}: rows={len(year_map):,}, CRS={year_map.crs} -> "
            f"{year_geojson_path.name}"
        )

    combined_geo = gpd.GeoDataFrame(
        pd.concat(combined_parts, ignore_index=True),
        geometry="geometry",
        crs=grid_wgs84.crs,
    )

    combined_geojson_path = (
        MAP_DIR
        / f"05_dynamic_future_heat_{future_years[0]}_{future_years[-1]}_wgs84.geojson"
    )
    combined_geo.to_file(combined_geojson_path, driver="GeoJSON")

    map_manifest_path = (
        MAP_DIR
        / f"05_dynamic_future_map_manifest_{future_years[0]}_{future_years[-1]}.json"
    )

    with map_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "forecast_origin_year": latest_year,
                "future_years": future_years,
                "combined_geojson": str(combined_geojson_path),
                "layers": map_files,
            },
            file,
            indent=2,
        )

    print(f"Combined GeoJSON : {combined_geojson_path}")

    # -------------------------------------------------------------------------
    section("8. FINAL STEP-05 AUDIT")

    final_mean_errors = []
    for year in future_years:
        part = prediction_table[
            prediction_table[YEAR_COL] == year
        ]
        final_mean_errors.append(
            abs(
                float(part["predicted_lst_c"].mean())
                - float(part["forecast_annual_mean_lst_c"].iloc[0])
            )
        )

    audit_rows = [
        {
            "check": "future_year_window",
            "status": "PASS",
            "details": f"origin={latest_year}, future={future_years}",
        },
        {
            "check": "future_feature_rows",
            "status": "PASS" if len(future_engineered) == expected_future_rows else "FAIL",
            "details": f"actual={len(future_engineered)}, expected={expected_future_rows}",
        },
        {
            "check": "frozen_feature_count",
            "status": "PASS" if len(ordered_features) == EXPECTED_FEATURES else "FAIL",
            "details": f"actual={len(ordered_features)}, expected={EXPECTED_FEATURES}",
        },
        {
            "check": "future_feature_matrix_finite",
            "status": "PASS" if np.isfinite(X_future.to_numpy(dtype=float)).all() else "FAIL",
            "details": f"rows={len(X_future)}, cols={X_future.shape[1]}",
        },
        {
            "check": "centered_anomaly_zero_mean_each_year",
            "status": "PASS" if max_center_mean <= ANOMALY_CENTER_TOL else "FAIL",
            "details": f"max_abs_year_mean={max_center_mean:.12e}",
        },
        {
            "check": "final_year_mean_equals_annual_level",
            "status": "PASS" if max(final_mean_errors) <= ANOMALY_CENTER_TOL else "FAIL",
            "details": f"max_abs_error={max(final_mean_errors):.12e}",
        },
        {
            "check": "model_retraining_in_step05",
            "status": "PASS",
            "details": "None. Step-05 performs inference only with the Step-03 versioned model.",
        },
        {
            "check": "engineered_features_independently_forecast",
            "status": "PASS",
            "details": "No. Exact Step-07 features rebuilt from observed + Step-04 base predictors.",
        },
        {
            "check": "original_2025_feature_reproduction",
            "status": feature_repro_status,
            "details": (
                f"max_abs_error={feature_max_error}"
                if feature_max_error is not None
                else "Not applicable for histories beyond 2025"
            ),
        },
        {
            "check": "original_2025_forecast_reproduction",
            "status": forecast_repro_status,
            "details": (
                f"max_abs_error={forecast_max_error}"
                if forecast_max_error is not None
                else "Not applicable for histories beyond 2025"
            ),
        },
        {
            "check": "yearly_geojson_count",
            "status": "PASS" if len(map_files) == FORECAST_HORIZON else "FAIL",
            "details": f"saved={len(map_files)}, expected={FORECAST_HORIZON}",
        },
    ]

    audit_df = pd.DataFrame(audit_rows)
    audit_path = REPORT_DIR / "05_dynamic_forecast_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    print(audit_df.to_string(index=False))

    failed = audit_df[audit_df["status"] == "FAIL"]
    if not failed.empty:
        raise RuntimeError("Step 05 final audit contains FAIL rows.")

    methodology = {
        "step": 5,
        "analysis": "Rolling five-year urban heat forecast generation",
        "forecast_origin_year": latest_year,
        "future_years": future_years,
        "forecast_horizon_years": [1, 2, 3, 4, 5],
        "model_source": str(dynamic_model_path),
        "model_target": "within-year spatial LST anomaly",
        "feature_count": len(ordered_features),
        "feature_engineering": "Exact Step-07 reconstruction",
        "spatial_context": "Queen-contiguity same-year predictor context",
        "annual_level_source": str(annual_level_path),
        "future_base_source": str(future_base_path),
        "final_equation": (
            "predicted_lst_c = forecast_annual_mean_lst_c + centered_spatial_anomaly_c"
        ),
        "pseudo_future_rows_used_for_model_training": False,
        "feature_reselection_performed": False,
        "hyperparameter_tuning_performed": False,
        "baseline_reproduction_status": {
            "feature_reproduction": feature_repro_status,
            "forecast_reproduction": forecast_repro_status,
        },
        "yearly_map_layers": map_files,
        "combined_geojson": str(combined_geojson_path),
    }

    methodology_path = REPORT_DIR / "05_dynamic_forecast_methodology.json"
    with methodology_path.open("w", encoding="utf-8") as file:
        json.dump(methodology, file, indent=2)

    summary = {
        "status": "PASS",
        "forecast_origin_year": latest_year,
        "future_years": future_years,
        "future_rows": int(len(prediction_table)),
        "feature_count": int(len(ordered_features)),
        "model_path": str(dynamic_model_path),
        "future_feature_path": str(future_feature_path),
        "future_prediction_path": str(future_prediction_path),
        "combined_geojson_path": str(combined_geojson_path),
        "audit_path": str(audit_path),
    }

    summary_path = REPORT_DIR / "05_dynamic_forecast_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\nFeature matrix : {future_feature_path}")
    print(f"Forecast CSV   : {future_prediction_path}")
    print(f"Year summary   : {year_summary_path}")
    print(f"Map manifest   : {map_manifest_path}")
    print(f"Audit          : {audit_path}")
    print(f"Methodology    : {methodology_path}")
    print(f"Summary        : {summary_path}")

    section("STEP 05 COMPLETED SUCCESSFULLY")
    print(
        f"Dynamic grid LST forecasts generated for "
        f"{future_years[0]}-{future_years[-1]}."
    )
    print(
        "NEXT: final dynamic-service audit, then backend/API integration."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "!" * 125)
        print("STEP 05 FAILED")
        print("!" * 125)
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
