"""
Dynamic Urban Heat scenario simulation service.

Research-consistent frontend scenario engine for:
- GREEN: persistent green-cover increase (final Step 29A V2 mechanism)
- BUILT: persistent NDBI reduction (Step 29B)
- BLUE: persistent NDWI increase (Step 29D)
- COMBINED: true joint GREEN + BUILT feature-vector simulation (Step 29C)

Important:
- Uses the frozen Step-22B production LightGBM.
- Uses Step-28 future model features/predictions as the baseline.
- Does not retrain any model.
- Does not modify research outputs on disk.
- Dynamic custom values are restricted to the research-supported magnitude envelope.
- Outputs are model-based planning sensitivities, not causal guarantees.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.config import URBAN_HEAT_ROOT


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
AREA_COL = "area_name"

HIST_YEARS = list(range(2015, 2026))
HISTORY_SEED_YEARS = [2023, 2024, 2025]
FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]

EXPECTED_FEATURES = 93

GREEN_RESEARCH_LEVELS_PP = [5.0, 10.0, 20.0]
GREEN_CUSTOM_MIN_PP = 0.0
GREEN_CUSTOM_MAX_PP = 20.0
GREEN_MIN = 0.0
GREEN_MAX = 1.0

SCENARIO_QUANTILES = {
    "LOW": 0.50,
    "MODERATE": 0.75,
    "HIGH": 0.90,
}

PREDICTION_REPRO_TOL = 1e-10
FEATURE_CHANGE_TOL = 1e-12
ZERO_TOL = 1e-10

BASE_PATH = (
    URBAN_HEAT_ROOT
    / "data"
    / "processed"
    / "kaduwela_preprocessed_base.csv"
)

FEATURE_LIST_PATH = (
    URBAN_HEAT_ROOT
    / "outputs"
    / "reports"
    / "21_final_feature_list.csv"
)

MODEL_PATH = (
    URBAN_HEAT_ROOT
    / "models"
    / "final"
    / "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
)

FUTURE_FEATURE_PATH = (
    URBAN_HEAT_ROOT
    / "outputs"
    / "forecasting"
    / "28_future_model_features_2026_2030.csv"
)

FUTURE_PRED_PATH = (
    URBAN_HEAT_ROOT
    / "outputs"
    / "forecasting"
    / "28_future_grid_lst_predictions_2026_2030.csv"
)


# =============================================================================
# HELPERS
# =============================================================================


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")



def _detect_feature_column(feature_df: pd.DataFrame) -> str:
    for candidate in ["feature", "feature_name", "predictor"]:
        if candidate in feature_df.columns:
            values = feature_df[candidate].dropna().astype(str).tolist()
            if len(values) == EXPECTED_FEATURES:
                return candidate

    for column in feature_df.columns:
        values = feature_df[column].dropna().astype(str).tolist()
        if (
            len(values) == EXPECTED_FEATURES
            and len(set(values)) == EXPECTED_FEATURES
        ):
            return column

    raise RuntimeError(
        f"Could not identify the frozen {EXPECTED_FEATURES}-feature column."
    )



def _sample_std_three(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1))



def _slope_three(oldest: float, middle: float, newest: float) -> float:
    del middle  # formula for x=[0,1,2] uses oldest/newest only
    return float((newest - oldest) / 2.0)



def _effect_direction(delta_c: float) -> str:
    if delta_c < -ZERO_TOL:
        return "cooling"
    if delta_c > ZERO_TOL:
        return "warming"
    return "no_change"



def _nonadditivity_direction(value: float) -> str:
    if value < -ZERO_TOL:
        return "more_cooling_than_additive"
    if value > ZERO_TOL:
        return "more_warming_than_additive"
    return "approximately_additive"



def _as_float(value: Any) -> float:
    return float(pd.to_numeric(value, errors="raise"))



def _build_temporal_features(
    historical_seed: pd.DataFrame,
    future_values: pd.DataFrame,
    variable: str,
) -> pd.DataFrame:
    """Rebuild strict-past lag/rolling features for one grid, 2026-2030."""

    hist_lookup = {
        int(row[YEAR_COL]): float(row[variable])
        for _, row in historical_seed.iterrows()
    }

    future_lookup = {
        int(row[YEAR_COL]): float(row[variable])
        for _, row in future_values.iterrows()
    }

    series: dict[int, float] = {}

    for year in HISTORY_SEED_YEARS:
        if year not in hist_lookup:
            raise RuntimeError(
                f"Missing historical {variable} value for year {year}."
            )
        series[year] = hist_lookup[year]

    rows: list[dict[str, Any]] = []

    for year in FUTURE_YEARS:
        if year not in future_lookup:
            raise RuntimeError(
                f"Missing future {variable} value for year {year}."
            )

        current = future_lookup[year]
        prev1 = series[year - 1]
        prev2 = series[year - 2]
        prev3 = series[year - 3]
        prior = [prev3, prev2, prev1]

        rows.append(
            {
                YEAR_COL: year,
                variable: current,
                f"{variable}_lag1": prev1,
                f"{variable}_lag2": prev2,
                f"{variable}_hist3_mean": float(np.mean(prior)),
                f"{variable}_hist3_std": _sample_std_three(prior),
                f"{variable}_hist3_slope": _slope_three(
                    prev3,
                    prev2,
                    prev1,
                ),
                f"{variable}_delta1": float(current - prev1),
            }
        )

        # Persistent intervention: current scenario value becomes future history.
        series[year] = current

    return pd.DataFrame(rows).sort_values(YEAR_COL).reset_index(drop=True)



def _changed_features(
    scenario: pd.DataFrame,
    baseline: pd.DataFrame,
    ordered_features: list[str],
) -> set[str]:
    changed: set[str] = set()

    for feature in ordered_features:
        max_change = float(
            np.max(
                np.abs(
                    scenario[feature].to_numpy(dtype=float)
                    - baseline[feature].to_numpy(dtype=float)
                )
            )
        )
        if max_change > FEATURE_CHANGE_TOL:
            changed.add(feature)

    return changed



def _match_level(value: float, lookup: dict[str, float]) -> str | None:
    for label, magnitude in lookup.items():
        if np.isclose(value, magnitude, rtol=0.0, atol=1e-12):
            return label
    return None


# =============================================================================
# CACHED RESEARCH ASSETS
# =============================================================================


@lru_cache(maxsize=1)
def _load_assets() -> dict[str, Any]:
    for path, label in [
        (BASE_PATH, "historical processed base"),
        (FEATURE_LIST_PATH, "frozen Step-21 feature list"),
        (MODEL_PATH, "frozen Step-22B production LightGBM"),
        (FUTURE_FEATURE_PATH, "Step-28 future feature matrix"),
        (FUTURE_PRED_PATH, "Step-28 future predictions"),
    ]:
        _require_file(path, label)

    feature_df = pd.read_csv(FEATURE_LIST_PATH)
    feature_col = _detect_feature_column(feature_df)
    ordered_features = (
        feature_df[feature_col].dropna().astype(str).tolist()
    )

    if len(ordered_features) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Frozen feature count={len(ordered_features)}, "
            f"expected={EXPECTED_FEATURES}."
        )

    base = pd.read_csv(
        BASE_PATH,
        dtype={GRID_COL: str, AREA_COL: str},
        low_memory=False,
    )
    future = pd.read_csv(
        FUTURE_FEATURE_PATH,
        dtype={GRID_COL: str, AREA_COL: str},
        low_memory=False,
    )
    pred = pd.read_csv(
        FUTURE_PRED_PATH,
        dtype={GRID_COL: str, AREA_COL: str},
        low_memory=False,
    )

    for frame, label in [
        (base, "historical base"),
        (future, "future features"),
        (pred, "future predictions"),
    ]:
        if GRID_COL not in frame.columns or YEAR_COL not in frame.columns:
            raise RuntimeError(f"{label} is missing grid_id/year.")
        frame[GRID_COL] = frame[GRID_COL].astype(str).str.strip()
        frame[YEAR_COL] = pd.to_numeric(
            frame[YEAR_COL], errors="raise"
        ).astype(int)

    future = (
        future.sort_values([YEAR_COL, GRID_COL])
        .reset_index(drop=True)
    )
    pred = (
        pred.sort_values([YEAR_COL, GRID_COL])
        .reset_index(drop=True)
    )

    if not future[[GRID_COL, YEAR_COL]].equals(
        pred[[GRID_COL, YEAR_COL]]
    ):
        raise RuntimeError(
            "Step-28 feature and prediction grid-year keys do not align."
        )

    missing_features = [
        feature
        for feature in ordered_features
        if feature not in future.columns
    ]
    if missing_features:
        raise RuntimeError(
            "Step-28 future matrix is missing frozen model features: "
            + str(missing_features)
        )

    X_base = future[ordered_features].copy()
    for feature in ordered_features:
        X_base[feature] = pd.to_numeric(
            X_base[feature], errors="raise"
        )

    if X_base.isna().any().any():
        raise RuntimeError("Step-28 future feature matrix contains missing values.")
    if not np.isfinite(X_base.to_numpy(dtype=float)).all():
        raise RuntimeError("Step-28 future feature matrix contains non-finite values.")

    model = joblib.load(MODEL_PATH)

    # Strictly confirm that backend inference reproduces the frozen Step-28 model.
    baseline_raw_model = np.asarray(model.predict(X_base), dtype=float)
    baseline_raw_saved = pd.to_numeric(
        pred["raw_spatial_anomaly_prediction_c"], errors="raise"
    ).to_numpy(dtype=float)

    max_reproduction_error = float(
        np.max(np.abs(baseline_raw_model - baseline_raw_saved))
    )
    if max_reproduction_error > PREDICTION_REPRO_TOL:
        raise RuntimeError(
            "Backend scenario service did not reproduce the frozen Step-28 "
            f"raw model predictions. max_error={max_reproduction_error:.12e}"
        )

    # Historical supports and data-derived scenario magnitudes.
    historical = base[base[YEAR_COL].isin(HIST_YEARS)].copy()

    for column in ["green_mask", "NDBI", "NDWI"]:
        if column not in historical.columns:
            raise RuntimeError(f"Historical base missing required column: {column}")
        historical[column] = pd.to_numeric(
            historical[column], errors="raise"
        )

    hist_sorted = historical[[GRID_COL, YEAR_COL, "NDBI", "NDWI"]].copy()
    hist_sorted = hist_sorted.sort_values([GRID_COL, YEAR_COL])
    hist_sorted["previous_year"] = (
        hist_sorted.groupby(GRID_COL)[YEAR_COL].shift(1)
    )
    hist_sorted["d_ndbi"] = (
        hist_sorted.groupby(GRID_COL)["NDBI"].diff()
    )
    hist_sorted["d_ndwi"] = (
        hist_sorted.groupby(GRID_COL)["NDWI"].diff()
    )

    transitions = hist_sorted[
        hist_sorted[YEAR_COL].between(2016, 2025)
    ].copy()

    if not (
        transitions[YEAR_COL] - transitions["previous_year"] == 1
    ).all():
        raise RuntimeError(
            "Historical annual transitions are not true adjacent-year transitions."
        )

    ndbi_abs_changes = transitions["d_ndbi"].abs().dropna()
    ndwi_positive_changes = transitions.loc[
        transitions["d_ndwi"] > 0,
        "d_ndwi",
    ].dropna()

    built_levels = {
        label: float(ndbi_abs_changes.quantile(q))
        for label, q in SCENARIO_QUANTILES.items()
    }
    blue_levels = {
        label: float(ndwi_positive_changes.quantile(q))
        for label, q in SCENARIO_QUANTILES.items()
    }

    return {
        "base": base,
        "future": future,
        "pred": pred,
        "X_base": X_base,
        "model": model,
        "ordered_features": ordered_features,
        "built_levels": built_levels,
        "blue_levels": blue_levels,
        "ndbi_min": float(historical["NDBI"].min()),
        "ndbi_max": float(historical["NDBI"].max()),
        "ndwi_min": float(historical["NDWI"].min()),
        "ndwi_max": float(historical["NDWI"].max()),
        "baseline_reproduction_max_error": max_reproduction_error,
    }


# =============================================================================
# GRID BASELINE EXTRACTION
# =============================================================================



def _grid_context(grid_id: str) -> dict[str, Any]:
    assets = _load_assets()
    grid_id = str(grid_id).strip()

    future = assets["future"]
    pred = assets["pred"]
    X_all = assets["X_base"]

    mask = future[GRID_COL] == grid_id
    indices = future.index[mask].tolist()

    if not indices:
        raise KeyError(f"Unknown grid_id: {grid_id}")

    grid_future = future.loc[indices].copy().reset_index(drop=True)
    grid_pred = pred.loc[indices].copy().reset_index(drop=True)
    X_base = X_all.loc[indices].copy().reset_index(drop=True)

    if grid_future[YEAR_COL].tolist() != FUTURE_YEARS:
        raise RuntimeError(
            f"Grid {grid_id} does not contain the complete 2026-2030 future series."
        )

    base = assets["base"]
    historical_seed = (
        base[
            (base[GRID_COL] == grid_id)
            & (base[YEAR_COL].isin(HISTORY_SEED_YEARS))
        ]
        .copy()
        .sort_values(YEAR_COL)
        .reset_index(drop=True)
    )

    if historical_seed[YEAR_COL].tolist() != HISTORY_SEED_YEARS:
        raise RuntimeError(
            f"Grid {grid_id} does not contain complete 2023-2025 history."
        )

    area_name = (
        str(grid_future.iloc[0][AREA_COL])
        if AREA_COL in grid_future.columns
        else None
    )

    return {
        "assets": assets,
        "grid_id": grid_id,
        "area_name": area_name,
        "grid_future": grid_future,
        "grid_pred": grid_pred,
        "X_base": X_base,
        "historical_seed": historical_seed,
    }


# =============================================================================
# FAMILY-SPECIFIC FEATURE BUILDERS
# =============================================================================



def _build_green_matrix(
    context: dict[str, Any],
    green_increase_pp: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    value = float(green_increase_pp)
    if not GREEN_CUSTOM_MIN_PP <= value <= GREEN_CUSTOM_MAX_PP:
        raise ValueError(
            f"green_increase_pp must be between {GREEN_CUSTOM_MIN_PP} "
            f"and {GREEN_CUSTOM_MAX_PP}."
        )

    X_base = context["X_base"]
    historical_seed = context["historical_seed"]
    ordered_features = context["assets"]["ordered_features"]

    if "green_mask" not in X_base.columns:
        raise RuntimeError("green_mask is not in the frozen feature list.")

    baseline_green = X_base["green_mask"].to_numpy(dtype=float)
    scenario_green = np.clip(
        baseline_green + value / 100.0,
        GREEN_MIN,
        GREEN_MAX,
    )

    future_values = pd.DataFrame(
        {
            YEAR_COL: FUTURE_YEARS,
            "green_mask": scenario_green,
        }
    )
    seed = historical_seed[[YEAR_COL, "green_mask"]].copy()
    seed["green_mask"] = pd.to_numeric(seed["green_mask"], errors="raise")

    rebuilt = _build_temporal_features(
        historical_seed=seed,
        future_values=future_values,
        variable="green_mask",
    )

    X = X_base.copy()
    X["green_mask"] = rebuilt["green_mask"].to_numpy(dtype=float)

    temporal_candidates = [
        "green_mask_lag1",
        "green_mask_lag2",
        "green_mask_hist3_mean",
        "green_mask_hist3_std",
        "green_mask_hist3_slope",
        "green_mask_delta1",
    ]
    current_candidates = [
        "vegetation_deficit",
        "ndbi_x_vegetation_deficit",
        "ndvi_x_green_mask",
        "green_mask_nbr_contrast",
    ]

    for feature in temporal_candidates:
        if feature in X.columns:
            X[feature] = rebuilt[feature].to_numpy(dtype=float)

    vegetation_deficit = 1.0 - scenario_green

    if "vegetation_deficit" in X.columns:
        X["vegetation_deficit"] = vegetation_deficit

    if "ndbi_x_vegetation_deficit" in X.columns:
        X["ndbi_x_vegetation_deficit"] = (
            X["NDBI"].to_numpy(dtype=float) * vegetation_deficit
        )

    if "ndvi_x_green_mask" in X.columns:
        X["ndvi_x_green_mask"] = (
            X["NDVI"].to_numpy(dtype=float) * scenario_green
        )

    if "green_mask_nbr_contrast" in X.columns:
        if "green_mask_nbr_mean" in X_base.columns:
            nbr_mean = X_base["green_mask_nbr_mean"].to_numpy(dtype=float)
        else:
            nbr_mean = (
                baseline_green
                - X_base["green_mask_nbr_contrast"].to_numpy(dtype=float)
            )
        X["green_mask_nbr_contrast"] = scenario_green - nbr_mean

    changed = _changed_features(X, X_base, ordered_features)
    allowed = {
        "green_mask",
        *[f for f in temporal_candidates if f in X.columns],
        *[f for f in current_candidates if f in X.columns],
    }
    allowed.discard("green_mask_nbr_mean")

    unexpected = sorted(changed - allowed)
    if unexpected:
        raise RuntimeError(
            "Green scenario changed unexpected frozen features: "
            + str(unexpected)
        )

    actual_pp = (scenario_green - baseline_green) * 100.0

    matched = None
    for level in GREEN_RESEARCH_LEVELS_PP:
        if np.isclose(value, level, rtol=0.0, atol=1e-12):
            matched = f"{int(level)}PP"
            break

    support_class = (
        "BASELINE_NO_CHANGE"
        if np.isclose(value, 0.0, atol=1e-12)
        else (
            "PRECOMPUTED_RESEARCH_LEVEL"
            if matched is not None
            else "CUSTOM_WITHIN_RESEARCH_ENVELOPE"
        )
    )

    return X, {
        "requested_green_increase_pp": value,
        "baseline_green": baseline_green,
        "scenario_green": scenario_green,
        "actual_green_increase_pp": actual_pp,
        "support_bound_limited": bool(
            np.any(actual_pp < value - 1e-10)
        ),
        "support_class": support_class,
        "matched_research_level": matched,
        "changed_features": sorted(changed),
    }



def _build_built_matrix(
    context: dict[str, Any],
    ndbi_reduction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    value = float(ndbi_reduction)
    assets = context["assets"]
    high = assets["built_levels"]["HIGH"]

    if not 0.0 <= value <= high + 1e-12:
        raise ValueError(
            "ndbi_reduction must be between 0 and the research HIGH/Q90 "
            f"magnitude ({high:.12f})."
        )

    X_base = context["X_base"]
    historical_seed = context["historical_seed"]
    ordered_features = assets["ordered_features"]

    if "NDBI" not in X_base.columns:
        raise RuntimeError("NDBI is not in the frozen feature list.")

    baseline_ndbi = X_base["NDBI"].to_numpy(dtype=float)
    scenario_ndbi = np.clip(
        baseline_ndbi - value,
        assets["ndbi_min"],
        assets["ndbi_max"],
    )

    future_values = pd.DataFrame(
        {YEAR_COL: FUTURE_YEARS, "NDBI": scenario_ndbi}
    )
    seed = historical_seed[[YEAR_COL, "NDBI"]].copy()
    seed["NDBI"] = pd.to_numeric(seed["NDBI"], errors="raise")

    rebuilt = _build_temporal_features(
        historical_seed=seed,
        future_values=future_values,
        variable="NDBI",
    )

    X = X_base.copy()
    X["NDBI"] = rebuilt["NDBI"].to_numpy(dtype=float)

    temporal_candidates = [
        "NDBI_lag1",
        "NDBI_lag2",
        "NDBI_hist3_mean",
        "NDBI_hist3_std",
        "NDBI_hist3_slope",
        "NDBI_delta1",
    ]
    current_candidates = [
        "ndbi_minus_ndvi",
        "ndbi_x_vegetation_deficit",
        "ndbi_x_building_mask",
        "NDBI_nbr_contrast",
    ]

    for feature in temporal_candidates:
        if feature in X.columns:
            X[feature] = rebuilt[feature].to_numpy(dtype=float)

    if "ndbi_minus_ndvi" in X.columns:
        X["ndbi_minus_ndvi"] = (
            scenario_ndbi - X["NDVI"].to_numpy(dtype=float)
        )

    if "ndbi_x_vegetation_deficit" in X.columns:
        X["ndbi_x_vegetation_deficit"] = (
            scenario_ndbi
            * (1.0 - X["green_mask"].to_numpy(dtype=float))
        )

    if "ndbi_x_building_mask" in X.columns:
        X["ndbi_x_building_mask"] = (
            scenario_ndbi * X["building_mask"].to_numpy(dtype=float)
        )

    if "NDBI_nbr_contrast" in X.columns:
        if "NDBI_nbr_mean" in X_base.columns:
            nbr_mean = X_base["NDBI_nbr_mean"].to_numpy(dtype=float)
        else:
            nbr_mean = (
                baseline_ndbi
                - X_base["NDBI_nbr_contrast"].to_numpy(dtype=float)
            )
        X["NDBI_nbr_contrast"] = scenario_ndbi - nbr_mean

    changed = _changed_features(X, X_base, ordered_features)
    allowed = {
        "NDBI",
        *[f for f in temporal_candidates if f in X.columns],
        *[f for f in current_candidates if f in X.columns],
    }
    allowed.discard("NDBI_nbr_mean")

    unexpected = sorted(changed - allowed)
    if unexpected:
        raise RuntimeError(
            "Built-up scenario changed unexpected frozen features: "
            + str(unexpected)
        )

    actual = baseline_ndbi - scenario_ndbi
    matched = _match_level(value, assets["built_levels"])
    support_class = (
        "BASELINE_NO_CHANGE"
        if np.isclose(value, 0.0, atol=1e-12)
        else (
            "PRECOMPUTED_RESEARCH_LEVEL"
            if matched is not None
            else "CUSTOM_WITHIN_RESEARCH_ENVELOPE"
        )
    )

    return X, {
        "requested_ndbi_reduction": value,
        "baseline_ndbi": baseline_ndbi,
        "scenario_ndbi": scenario_ndbi,
        "actual_ndbi_reduction": actual,
        "support_bound_limited": bool(
            np.any(actual < value - 1e-12)
        ),
        "support_class": support_class,
        "matched_research_level": matched,
        "changed_features": sorted(changed),
    }



def _build_blue_matrix(
    context: dict[str, Any],
    ndwi_increase: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    value = float(ndwi_increase)
    assets = context["assets"]
    high = assets["blue_levels"]["HIGH"]

    if not 0.0 <= value <= high + 1e-12:
        raise ValueError(
            "ndwi_increase must be between 0 and the research HIGH/Q90 "
            f"magnitude ({high:.12f})."
        )

    X_base = context["X_base"]
    historical_seed = context["historical_seed"]
    ordered_features = assets["ordered_features"]

    if "NDWI" not in X_base.columns:
        raise RuntimeError("NDWI is not in the frozen feature list.")

    baseline_ndwi = X_base["NDWI"].to_numpy(dtype=float)
    scenario_ndwi = np.clip(
        baseline_ndwi + value,
        assets["ndwi_min"],
        assets["ndwi_max"],
    )

    future_values = pd.DataFrame(
        {YEAR_COL: FUTURE_YEARS, "NDWI": scenario_ndwi}
    )
    seed = historical_seed[[YEAR_COL, "NDWI"]].copy()
    seed["NDWI"] = pd.to_numeric(seed["NDWI"], errors="raise")

    rebuilt = _build_temporal_features(
        historical_seed=seed,
        future_values=future_values,
        variable="NDWI",
    )

    X = X_base.copy()
    X["NDWI"] = rebuilt["NDWI"].to_numpy(dtype=float)

    temporal_candidates = [
        "NDWI_lag1",
        "NDWI_lag2",
        "NDWI_hist3_mean",
        "NDWI_hist3_std",
        "NDWI_hist3_slope",
        "NDWI_delta1",
    ]
    current_candidates = ["NDWI_nbr_contrast"]

    for feature in temporal_candidates:
        if feature in X.columns:
            X[feature] = rebuilt[feature].to_numpy(dtype=float)

    if "NDWI_nbr_contrast" in X.columns:
        if "NDWI_nbr_mean" in X_base.columns:
            nbr_mean = X_base["NDWI_nbr_mean"].to_numpy(dtype=float)
        else:
            nbr_mean = (
                baseline_ndwi
                - X_base["NDWI_nbr_contrast"].to_numpy(dtype=float)
            )
        X["NDWI_nbr_contrast"] = scenario_ndwi - nbr_mean

    # Structural distance-to-water context remains frozen.
    for frozen_feature in ["dist_water", "log1p_dist_water"]:
        if frozen_feature in X.columns:
            max_change = float(
                np.max(
                    np.abs(
                        X[frozen_feature].to_numpy(dtype=float)
                        - X_base[frozen_feature].to_numpy(dtype=float)
                    )
                )
            )
            if max_change > FEATURE_CHANGE_TOL:
                raise RuntimeError(
                    f"Blue scenario changed frozen structural feature: {frozen_feature}"
                )

    changed = _changed_features(X, X_base, ordered_features)
    allowed = {
        "NDWI",
        *[f for f in temporal_candidates if f in X.columns],
        *[f for f in current_candidates if f in X.columns],
    }
    allowed.discard("dist_water")
    allowed.discard("log1p_dist_water")

    unexpected = sorted(changed - allowed)
    if unexpected:
        raise RuntimeError(
            "Blue scenario changed unexpected frozen features: "
            + str(unexpected)
        )

    actual = scenario_ndwi - baseline_ndwi
    matched = _match_level(value, assets["blue_levels"])
    support_class = (
        "BASELINE_NO_CHANGE"
        if np.isclose(value, 0.0, atol=1e-12)
        else (
            "PRECOMPUTED_RESEARCH_LEVEL"
            if matched is not None
            else "CUSTOM_WITHIN_RESEARCH_ENVELOPE"
        )
    )

    return X, {
        "requested_ndwi_increase": value,
        "baseline_ndwi": baseline_ndwi,
        "scenario_ndwi": scenario_ndwi,
        "actual_ndwi_increase": actual,
        "support_bound_limited": bool(
            np.any(actual < value - 1e-12)
        ),
        "support_class": support_class,
        "matched_research_level": matched,
        "changed_features": sorted(changed),
    }


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================



def get_dynamic_scenario_options() -> dict[str, Any]:
    """Return frontend-safe dynamic scenario bounds and canonical research levels."""
    assets = _load_assets()

    return {
        "years": FUTURE_YEARS,
        "green": {
            "input_name": "green_increase_pp",
            "unit": "percentage_points",
            "min": GREEN_CUSTOM_MIN_PP,
            "max": GREEN_CUSTOM_MAX_PP,
            "precomputed_research_levels": GREEN_RESEARCH_LEVELS_PP,
            "variable": "green_mask",
        },
        "built": {
            "input_name": "ndbi_reduction",
            "unit": "ndbi_index_units",
            "min": 0.0,
            "max": assets["built_levels"]["HIGH"],
            "research_levels": assets["built_levels"],
            "variable": "NDBI",
            "wording": "built-up intensity reduction",
        },
        "blue": {
            "input_name": "ndwi_increase",
            "unit": "ndwi_index_units",
            "min": 0.0,
            "max": assets["blue_levels"]["HIGH"],
            "research_levels": assets["blue_levels"],
            "variable": "NDWI",
            "wording": "blue-space / water-wetness intensity increase",
        },
        "combined": {
            "supported_combination": "GREEN + BUILT",
            "blue_in_combined": False,
        },
        "model": {
            "file": MODEL_PATH.name,
            "type": type(assets["model"]).__name__,
            "feature_count": len(assets["ordered_features"]),
            "baseline_reproduction_max_error": assets[
                "baseline_reproduction_max_error"
            ],
        },
        "interpretation": (
            "Dynamic outputs are model-based planning sensitivities within the "
            "research-supported scenario envelope. They are not causal guarantees."
        ),
    }



def simulate_custom_scenario(
    grid_id: str,
    year: int,
    family: str,
    *,
    green_increase_pp: float | None = None,
    ndbi_reduction: float | None = None,
    ndwi_increase: float | None = None,
) -> dict[str, Any]:
    """
    Simulate one planner-selected local scenario for one grid and target year.

    family:
        GREEN | BUILT | BLUE | COMBINED

    COMBINED means GREEN + BUILT, matching Step 29C.
    """

    year = int(year)
    if year not in FUTURE_YEARS:
        raise ValueError("year must be between 2026 and 2030.")

    family = str(family).strip().upper()
    if family not in {"GREEN", "BUILT", "BLUE", "COMBINED"}:
        raise ValueError(
            "family must be one of GREEN, BUILT, BLUE, COMBINED."
        )

    context = _grid_context(grid_id)
    assets = context["assets"]
    model = assets["model"]
    X_base = context["X_base"]
    pred = context["grid_pred"]

    target_idx = FUTURE_YEARS.index(year)
    baseline_lst = _as_float(pred.iloc[target_idx]["predicted_lst_c"])
    baseline_raw = _as_float(
        pred.iloc[target_idx]["raw_spatial_anomaly_prediction_c"]
    )

    common = {
        "grid_id": context["grid_id"],
        "area_name": context["area_name"],
        "year": year,
        "scenario_family": family,
        "baseline_predicted_lst_c": baseline_lst,
        "baseline_raw_spatial_anomaly_c": baseline_raw,
        "model_file": MODEL_PATH.name,
        "model_type": type(model).__name__,
        "model_feature_count": len(assets["ordered_features"]),
        "scenario_uncertainty_available": False,
        "uncertainty_note": (
            "No new formal uncertainty interval is generated for arbitrary dynamic "
            "custom scenario deltas. Step-30 uncertainty remains the reference "
            "uncertainty framework for frozen/precomputed outputs."
        ),
        "interpretation_warning": (
            "Model-based planning sensitivity only; not a guaranteed causal "
            "physical cooling/heating effect."
        ),
    }

    if family == "GREEN":
        if green_increase_pp is None:
            raise ValueError("green_increase_pp is required for GREEN.")

        Xs, meta = _build_green_matrix(context, green_increase_pp)
        scenario_raw_all = np.asarray(model.predict(Xs), dtype=float)
        base_raw_all = np.asarray(model.predict(X_base), dtype=float)
        delta_all = scenario_raw_all - base_raw_all

        delta = float(delta_all[target_idx])
        scenario_lst = baseline_lst + delta

        return {
            **common,
            "scenario_id": f"CUSTOM_GREEN_{float(green_increase_pp):g}PP",
            "requested_green_increase_pp": meta[
                "requested_green_increase_pp"
            ],
            "baseline_green_mask": float(meta["baseline_green"][target_idx]),
            "scenario_green_mask": float(meta["scenario_green"][target_idx]),
            "actual_green_increase_pp": float(
                meta["actual_green_increase_pp"][target_idx]
            ),
            "support_bound_limited": meta["support_bound_limited"],
            "support_class": meta["support_class"],
            "matched_research_level": meta["matched_research_level"],
            "scenario_predicted_lst_c": scenario_lst,
            "modeled_lst_change_c": delta,
            "cooling_benefit_c": -delta,
            "scenario_effect_direction": _effect_direction(delta),
            "changed_features": meta["changed_features"],
        }

    if family == "BUILT":
        if ndbi_reduction is None:
            raise ValueError("ndbi_reduction is required for BUILT.")

        Xs, meta = _build_built_matrix(context, ndbi_reduction)
        scenario_raw_all = np.asarray(model.predict(Xs), dtype=float)
        base_raw_all = np.asarray(model.predict(X_base), dtype=float)
        delta_all = scenario_raw_all - base_raw_all

        delta = float(delta_all[target_idx])
        scenario_lst = baseline_lst + delta

        return {
            **common,
            "scenario_id": f"CUSTOM_BUILT_{float(ndbi_reduction):g}",
            "requested_ndbi_reduction": meta["requested_ndbi_reduction"],
            "baseline_ndbi": float(meta["baseline_ndbi"][target_idx]),
            "scenario_ndbi": float(meta["scenario_ndbi"][target_idx]),
            "actual_ndbi_reduction": float(
                meta["actual_ndbi_reduction"][target_idx]
            ),
            "support_bound_limited": meta["support_bound_limited"],
            "support_class": meta["support_class"],
            "matched_research_level": meta["matched_research_level"],
            "scenario_predicted_lst_c": scenario_lst,
            "modeled_lst_change_c": delta,
            "cooling_benefit_c": -delta,
            "scenario_effect_direction": _effect_direction(delta),
            "changed_features": meta["changed_features"],
        }

    if family == "BLUE":
        if ndwi_increase is None:
            raise ValueError("ndwi_increase is required for BLUE.")

        Xs, meta = _build_blue_matrix(context, ndwi_increase)
        scenario_raw_all = np.asarray(model.predict(Xs), dtype=float)
        base_raw_all = np.asarray(model.predict(X_base), dtype=float)
        delta_all = scenario_raw_all - base_raw_all

        delta = float(delta_all[target_idx])
        scenario_lst = baseline_lst + delta

        dist_water = None
        if "dist_water" in X_base.columns:
            dist_water = float(X_base.iloc[target_idx]["dist_water"])

        return {
            **common,
            "scenario_id": f"CUSTOM_BLUE_{float(ndwi_increase):g}",
            "requested_ndwi_increase": meta["requested_ndwi_increase"],
            "baseline_ndwi": float(meta["baseline_ndwi"][target_idx]),
            "scenario_ndwi": float(meta["scenario_ndwi"][target_idx]),
            "actual_ndwi_increase": float(
                meta["actual_ndwi_increase"][target_idx]
            ),
            "dist_water_m": dist_water,
            "support_bound_limited": meta["support_bound_limited"],
            "support_class": meta["support_class"],
            "matched_research_level": meta["matched_research_level"],
            "scenario_predicted_lst_c": scenario_lst,
            "modeled_lst_change_c": delta,
            "cooling_benefit_c": -delta,
            "scenario_effect_direction": _effect_direction(delta),
            "changed_features": meta["changed_features"],
        }

    # COMBINED = GREEN + BUILT, following Step 29C true-joint logic.
    if green_increase_pp is None:
        raise ValueError("green_increase_pp is required for COMBINED.")
    if ndbi_reduction is None:
        raise ValueError("ndbi_reduction is required for COMBINED.")

    Xg, green_meta = _build_green_matrix(context, green_increase_pp)
    Xb, built_meta = _build_built_matrix(context, ndbi_reduction)

    green_changed = set(green_meta["changed_features"])
    built_changed = set(built_meta["changed_features"])

    allowed_shared: set[str] = set()
    if "ndbi_x_vegetation_deficit" in assets["ordered_features"]:
        allowed_shared.add("ndbi_x_vegetation_deficit")

    unexpected_shared = (green_changed & built_changed) - allowed_shared
    if unexpected_shared:
        raise RuntimeError(
            "Unexpected overlap between green and built feature mechanisms: "
            + str(sorted(unexpected_shared))
        )

    Xc = X_base.copy()

    for feature in sorted(green_changed - allowed_shared):
        Xc[feature] = Xg[feature].to_numpy(dtype=float)

    for feature in sorted(built_changed - allowed_shared):
        Xc[feature] = Xb[feature].to_numpy(dtype=float)

    if "ndbi_x_vegetation_deficit" in allowed_shared:
        Xc["ndbi_x_vegetation_deficit"] = (
            Xc["NDBI"].to_numpy(dtype=float)
            * (1.0 - Xc["green_mask"].to_numpy(dtype=float))
        )

    base_raw_all = np.asarray(model.predict(X_base), dtype=float)
    green_raw_all = np.asarray(model.predict(Xg), dtype=float)
    built_raw_all = np.asarray(model.predict(Xb), dtype=float)
    combined_raw_all = np.asarray(model.predict(Xc), dtype=float)

    green_delta_all = green_raw_all - base_raw_all
    built_delta_all = built_raw_all - base_raw_all
    combined_delta_all = combined_raw_all - base_raw_all

    green_delta = float(green_delta_all[target_idx])
    built_delta = float(built_delta_all[target_idx])
    combined_delta = float(combined_delta_all[target_idx])
    additive_reference = green_delta + built_delta
    nonadditivity = combined_delta - additive_reference

    combined_changed = _changed_features(
        Xc,
        X_base,
        assets["ordered_features"],
    )

    return {
        **common,
        "scenario_id": (
            f"CUSTOM_COMBINED_G{float(green_increase_pp):g}PP_"
            f"B{float(ndbi_reduction):g}"
        ),
        "requested_green_increase_pp": green_meta[
            "requested_green_increase_pp"
        ],
        "actual_green_increase_pp": float(
            green_meta["actual_green_increase_pp"][target_idx]
        ),
        "baseline_green_mask": float(green_meta["baseline_green"][target_idx]),
        "scenario_green_mask": float(green_meta["scenario_green"][target_idx]),
        "green_support_bound_limited": green_meta["support_bound_limited"],
        "green_support_class": green_meta["support_class"],
        "green_matched_research_level": green_meta["matched_research_level"],
        "requested_ndbi_reduction": built_meta["requested_ndbi_reduction"],
        "actual_ndbi_reduction": float(
            built_meta["actual_ndbi_reduction"][target_idx]
        ),
        "baseline_ndbi": float(built_meta["baseline_ndbi"][target_idx]),
        "scenario_ndbi": float(built_meta["scenario_ndbi"][target_idx]),
        "built_support_bound_limited": built_meta["support_bound_limited"],
        "built_support_class": built_meta["support_class"],
        "built_matched_research_level": built_meta["matched_research_level"],
        "combined_scenario_predicted_lst_c": baseline_lst + combined_delta,
        "combined_modeled_lst_change_c": combined_delta,
        "combined_cooling_benefit_c": -combined_delta,
        "combined_effect_direction": _effect_direction(combined_delta),
        "green_only_modeled_lst_change_c": green_delta,
        "built_only_modeled_lst_change_c": built_delta,
        "additive_expected_lst_change_c": additive_reference,
        "model_non_additivity_c": nonadditivity,
        "non_additivity_direction": _nonadditivity_direction(nonadditivity),
        "changed_features": sorted(combined_changed),
        "non_additivity_note": (
            "Model non-additivity is a model-behaviour diagnostic, not causal synergy."
        ),
    }
