from functools import lru_cache

import json

from pathlib import Path

from typing import Any



import pandas as pd



from backend.config import (

    GRID_METADATA_FILE,

    HISTORICAL_HEAT_FILE,

    FUTURE_HEAT_FILE,

    COOLING_PRIORITY_FILE,

    CURRENT_VS_FORECAST_FILE,

    AREA_HEAT_SUMMARY_FILE,

    AREA_PRIORITY_SUMMARY_FILE,

    DASHBOARD_MANIFEST_FILE,

    GRID_METADATA_GEOJSON,

    FUTURE_HEAT_GEOJSON,

    COOLING_PRIORITY_GEOJSON,

    HEAT_SCENARIOS_DIR,

    HISTORICAL_START_YEAR,

    HISTORICAL_END_YEAR,

    FORECAST_START_YEAR,

    FORECAST_END_YEAR,

)





# ============================================================

# URBAN HEAT CONSTANTS

# ============================================================



GRID_ID_COLUMN = "grid_id"

AREA_COLUMN = "area_name"

YEAR_COLUMN = "year"



EXPECTED_GRID_COUNT = 2194



PRIMARY_SCENARIOS_FILE = (

    HEAT_SCENARIOS_DIR

    / "34_primary_scenarios_2026_2030.csv"

)



GREEN_SCENARIOS_FILE = (

    HEAT_SCENARIOS_DIR

    / "34_green_scenarios_2026_2030.csv"

)



BUILT_SCENARIOS_FILE = (

    HEAT_SCENARIOS_DIR

    / "34_built_scenarios_2026_2030.csv"

)



COMBINED_SCENARIOS_FILE = (

    HEAT_SCENARIOS_DIR

    / "34_combined_scenarios_2026_2030.csv"

)



BLUE_SCENARIOS_FILE = (

    HEAT_SCENARIOS_DIR

    / "34_blue_scenarios_2026_2030.csv"

)





# ============================================================

# INTERNAL HELPERS

# ============================================================



def _require_file(path: Path, label: str) -> None:

    """

    Make sure a required Step 34 dashboard file exists.

    """

    if not path.exists():

        raise FileNotFoundError(

            f"{label} was not found: {path}"

        )





def _prepare_dataframe(

    path: Path,

    label: str,

) -> pd.DataFrame:

    """

    Read a CSV safely and standardize common Urban Heat columns.

    """

    _require_file(path, label)



    df = pd.read_csv(

        path,

        dtype={GRID_ID_COLUMN: str},

        low_memory=False,

    )



    if GRID_ID_COLUMN in df.columns:

        df[GRID_ID_COLUMN] = (

            df[GRID_ID_COLUMN]

            .astype(str)

            .str.strip()

        )



    if AREA_COLUMN in df.columns:

        df[AREA_COLUMN] = (

            df[AREA_COLUMN]

            .astype(str)

            .str.strip()

        )



    if YEAR_COLUMN in df.columns:

        df[YEAR_COLUMN] = pd.to_numeric(

            df[YEAR_COLUMN],

            errors="raise",

        ).astype(int)



    return df





def _normalize_trajectory_class(value):

    """

    Standardize trajectory-class labels for frontend/API use.

    """

    if value is None or pd.isna(value):

        return None



    value = str(value).strip()



    mapping = {

        "PersistentCool": "Persistent Cool",

        "Persistent Cool": "Persistent Cool",

        "PersistentHot": "Persistent Hot",

        "Persistent Hot": "Persistent Hot",

        "Diminishing": "Diminishing",

        "Emerging": "Emerging",

        "Intensifying": "Intensifying",

    }



    return mapping.get(value, value)





def _records(df: pd.DataFrame) -> list[dict[str, Any]]:

    """

    Convert a DataFrame into JSON-safe records.



    NaN / infinity values are converted to None before FastAPI

    sends them to the frontend.



    Trajectory labels are also standardized for frontend use.

    """

    safe_df = df.copy()



    safe_df = safe_df.replace(

        [float("inf"), float("-inf")],

        pd.NA,

    )



    if "trajectory_class" in safe_df.columns:

        safe_df["trajectory_class"] = (

            safe_df["trajectory_class"]

            .apply(_normalize_trajectory_class)

        )



    safe_df = safe_df.astype(object).where(

        pd.notna(safe_df),

        None,

    )



    return safe_df.to_dict(orient="records")





def _load_geojson(path: Path, label: str) -> dict[str, Any]:

    """

    Read a Step 34 GeoJSON file using Python's built-in JSON

    library. GeoPandas is not required in the backend.

    """

    _require_file(path, label)



    with path.open(

        "r",

        encoding="utf-8",

    ) as file:

        return json.load(file)





# ============================================================

# CACHED STEP 34 DATA LOADERS

# ============================================================



@lru_cache(maxsize=1)

def load_grid_metadata() -> pd.DataFrame:

    df = _prepare_dataframe(

        GRID_METADATA_FILE,

        "Urban Heat grid metadata",

    )



    required = {

        GRID_ID_COLUMN,

        AREA_COLUMN,

    }



    missing = required - set(df.columns)



    if missing:

        raise RuntimeError(

            f"Grid metadata is missing columns: {sorted(missing)}"

        )



    if df[GRID_ID_COLUMN].nunique() != EXPECTED_GRID_COUNT:

        raise RuntimeError(

            "Grid metadata does not contain the expected "

            f"{EXPECTED_GRID_COUNT} unique grids."

        )



    return df





@lru_cache(maxsize=1)

def load_historical_heat() -> pd.DataFrame:

    df = _prepare_dataframe(

        HISTORICAL_HEAT_FILE,

        "Urban Heat historical dashboard data",

    )



    required = {

        GRID_ID_COLUMN,

        AREA_COLUMN,

        YEAR_COLUMN,

        "actual_lst_c",

    }



    missing = required - set(df.columns)



    if missing:

        raise RuntimeError(

            f"Historical heat data is missing columns: "

            f"{sorted(missing)}"

        )



    return df





@lru_cache(maxsize=1)

def load_future_heat() -> pd.DataFrame:

    df = _prepare_dataframe(

        FUTURE_HEAT_FILE,

        "Urban Heat future dashboard master",

    )



    required = {

        GRID_ID_COLUMN,

        AREA_COLUMN,

        YEAR_COLUMN,

        "predicted_lst_c",

        "trajectory_class",

        "topsis_rank",

        "topsis_priority_score",

    }



    missing = required - set(df.columns)



    if missing:

        raise RuntimeError(

            f"Future dashboard data is missing columns: "

            f"{sorted(missing)}"

        )



    return df





@lru_cache(maxsize=1)

def load_cooling_priority() -> pd.DataFrame:

    df = _prepare_dataframe(

        COOLING_PRIORITY_FILE,

        "Urban Heat cooling priority data",

    )



    required = {

        GRID_ID_COLUMN,

        AREA_COLUMN,

        "topsis_rank",

        "topsis_priority_score",

    }



    missing = required - set(df.columns)



    if missing:

        raise RuntimeError(

            f"Cooling priority data is missing columns: "

            f"{sorted(missing)}"

        )



    return df





@lru_cache(maxsize=1)

def load_current_vs_forecast() -> pd.DataFrame:

    return _prepare_dataframe(

        CURRENT_VS_FORECAST_FILE,

        "Urban Heat current-vs-forecast data",

    )





@lru_cache(maxsize=1)

def load_area_heat_summary() -> pd.DataFrame:

    return _prepare_dataframe(

        AREA_HEAT_SUMMARY_FILE,

        "Urban Heat area heat summary",

    )





@lru_cache(maxsize=1)

def load_area_priority_summary() -> pd.DataFrame:

    return _prepare_dataframe(

        AREA_PRIORITY_SUMMARY_FILE,

        "Urban Heat area priority summary",

    )





@lru_cache(maxsize=1)

def load_primary_scenarios() -> pd.DataFrame:

    return _prepare_dataframe(

        PRIMARY_SCENARIOS_FILE,

        "Urban Heat primary scenarios",

    )





@lru_cache(maxsize=1)

def load_green_scenarios() -> pd.DataFrame:

    return _prepare_dataframe(

        GREEN_SCENARIOS_FILE,

        "Urban Heat green scenarios",

    )





@lru_cache(maxsize=1)

def load_built_scenarios() -> pd.DataFrame:

    return _prepare_dataframe(

        BUILT_SCENARIOS_FILE,

        "Urban Heat built-up scenarios",

    )





@lru_cache(maxsize=1)

def load_combined_scenarios() -> pd.DataFrame:

    return _prepare_dataframe(

        COMBINED_SCENARIOS_FILE,

        "Urban Heat combined scenarios",

    )





@lru_cache(maxsize=1)

def load_blue_scenarios() -> pd.DataFrame:

    return _prepare_dataframe(

        BLUE_SCENARIOS_FILE,

        "Urban Heat blue-space scenarios",

    )





# ============================================================

# GRID / AREA LOOKUPS

# ============================================================



def get_grid_metadata(

    grid_id: str,

) -> dict[str, Any] | None:

    df = load_grid_metadata()



    row = df[

        df[GRID_ID_COLUMN] == str(grid_id).strip()

    ]



    if row.empty:

        return None



    return _records(row.head(1))[0]





def get_area_names() -> list[str]:

    df = load_grid_metadata()



    return sorted(

        df[AREA_COLUMN]

        .dropna()

        .astype(str)

        .unique()

        .tolist()

    )





# ============================================================

# HISTORICAL DATA

# ============================================================



def get_historical_by_year(

    year: int,

) -> list[dict[str, Any]]:

    if year < HISTORICAL_START_YEAR or year > HISTORICAL_END_YEAR:

        raise ValueError(

            f"Historical year must be between "

            f"{HISTORICAL_START_YEAR} and "

            f"{HISTORICAL_END_YEAR}."

        )



    df = load_historical_heat()



    result = df[

        df[YEAR_COLUMN] == int(year)

    ].copy()



    return _records(result)





def get_grid_history(

    grid_id: str,

) -> list[dict[str, Any]]:

    df = load_historical_heat()



    result = df[

        df[GRID_ID_COLUMN] == str(grid_id).strip()

    ].copy()



    result = result.sort_values(YEAR_COLUMN)



    return _records(result)





# ============================================================

# FUTURE FORECAST DATA

# ============================================================



def get_forecast_by_year(

    year: int,

) -> list[dict[str, Any]]:

    if year < FORECAST_START_YEAR or year > FORECAST_END_YEAR:

        raise ValueError(

            f"Forecast year must be between "

            f"{FORECAST_START_YEAR} and "

            f"{FORECAST_END_YEAR}."

        )



    df = load_future_heat()



    result = df[

        df[YEAR_COLUMN] == int(year)

    ].copy()



    return _records(result)





def get_grid_forecast(

    grid_id: str,

    year: int,

) -> dict[str, Any] | None:

    if year < FORECAST_START_YEAR or year > FORECAST_END_YEAR:

        raise ValueError(

            f"Forecast year must be between "

            f"{FORECAST_START_YEAR} and "

            f"{FORECAST_END_YEAR}."

        )



    df = load_future_heat()



    result = df[

        (

            df[GRID_ID_COLUMN]

            == str(grid_id).strip()

        )

        & (

            df[YEAR_COLUMN]

            == int(year)

        )

    ]



    if result.empty:

        return None



    return _records(result.head(1))[0]





def get_grid_all_future_years(

    grid_id: str,

) -> list[dict[str, Any]]:

    df = load_future_heat()



    result = df[

        df[GRID_ID_COLUMN] == str(grid_id).strip()

    ].copy()



    result = result.sort_values(YEAR_COLUMN)



    return _records(result)





# ============================================================

# COOLING PRIORITY

# ============================================================



def get_priority_ranking(

    limit: int | None = None,

    area_name: str | None = None,

) -> list[dict[str, Any]]:

    df = load_cooling_priority().copy()



    if area_name:

        df = df[

            df[AREA_COLUMN].str.casefold()

            == str(area_name).strip().casefold()

        ]



    df = df.sort_values(

        "topsis_rank",

        ascending=True,

    )



    if limit is not None:

        if limit <= 0:

            raise ValueError(

                "Priority result limit must be greater than 0."

            )



        df = df.head(limit)



    return _records(df)





def get_grid_priority(

    grid_id: str,

) -> dict[str, Any] | None:

    df = load_cooling_priority()



    result = df[

        df[GRID_ID_COLUMN] == str(grid_id).strip()

    ]



    if result.empty:

        return None



    return _records(result.head(1))[0]





# ============================================================

# SCENARIO DATA

# ============================================================



def get_primary_scenario(

    grid_id: str,

    year: int,

) -> dict[str, Any] | None:

    if year < FORECAST_START_YEAR or year > FORECAST_END_YEAR:

        raise ValueError(

            f"Scenario year must be between "

            f"{FORECAST_START_YEAR} and "

            f"{FORECAST_END_YEAR}."

        )



    df = load_primary_scenarios()



    result = df[

        (

            df[GRID_ID_COLUMN]

            == str(grid_id).strip()

        )

        & (

            df[YEAR_COLUMN]

            == int(year)

        )

    ]



    if result.empty:

        return None



    return _records(result.head(1))[0]





# ============================================================

# AREA SUMMARIES

# ============================================================



def get_area_heat(

    area_name: str | None = None,

    year: int | None = None,

) -> list[dict[str, Any]]:

    df = load_area_heat_summary().copy()



    if area_name:

        df = df[

            df[AREA_COLUMN].str.casefold()

            == str(area_name).strip().casefold()

        ]



    if year is not None:

        df = df[

            df[YEAR_COLUMN] == int(year)

        ]



    return _records(df)





def get_area_priority(

    area_name: str | None = None,

) -> list[dict[str, Any]]:

    df = load_area_priority_summary().copy()



    if area_name:

        df = df[

            df[AREA_COLUMN].str.casefold()

            == str(area_name).strip().casefold()

        ]



    return _records(df)





# ============================================================

# MAP / GEOJSON DATA

# ============================================================



@lru_cache(maxsize=1)

def get_grid_geojson() -> dict[str, Any]:

    return _load_geojson(

        GRID_METADATA_GEOJSON,

        "Urban Heat grid GeoJSON",

    )





@lru_cache(maxsize=1)

def get_priority_geojson() -> dict[str, Any]:

    return _load_geojson(

        COOLING_PRIORITY_GEOJSON,

        "Urban Heat cooling priority GeoJSON",

    )





@lru_cache(maxsize=1)

def _load_future_geojson() -> dict[str, Any]:

    return _load_geojson(

        FUTURE_HEAT_GEOJSON,

        "Urban Heat future heat GeoJSON",

    )





def get_future_geojson_by_year(

    year: int,

) -> dict[str, Any]:

    if year < FORECAST_START_YEAR or year > FORECAST_END_YEAR:

        raise ValueError(

            f"Forecast map year must be between "

            f"{FORECAST_START_YEAR} and "

            f"{FORECAST_END_YEAR}."

        )



    geojson = _load_future_geojson()



    features = []



    for feature in geojson.get("features", []):

        properties = feature.get(

            "properties",

            {},

        )



        feature_year = properties.get(YEAR_COLUMN)



        try:

            feature_year = int(feature_year)

        except (TypeError, ValueError):

            continue



        if feature_year == int(year):

            features.append(feature)



    return {

        "type": "FeatureCollection",

        "features": features,

    }





# ============================================================

# DASHBOARD MANIFEST / DATA HEALTH

# ============================================================



def get_dashboard_manifest() -> dict[str, Any]:

    _require_file(

        DASHBOARD_MANIFEST_FILE,

        "Urban Heat dashboard manifest",

    )



    with DASHBOARD_MANIFEST_FILE.open(

        "r",

        encoding="utf-8",

    ) as file:

        return json.load(file)





def get_data_health() -> dict[str, Any]:

    """

    Check whether the most important Step 34 dashboard files

    exist and can be loaded by the backend.

    """



    files = {

        "grid_metadata": GRID_METADATA_FILE,

        "historical_heat": HISTORICAL_HEAT_FILE,

        "future_heat": FUTURE_HEAT_FILE,

        "cooling_priority": COOLING_PRIORITY_FILE,

        "current_vs_forecast": CURRENT_VS_FORECAST_FILE,

        "area_heat_summary": AREA_HEAT_SUMMARY_FILE,

        "area_priority_summary": AREA_PRIORITY_SUMMARY_FILE,

        "primary_scenarios": PRIMARY_SCENARIOS_FILE,

        "grid_geojson": GRID_METADATA_GEOJSON,

        "future_geojson": FUTURE_HEAT_GEOJSON,

        "priority_geojson": COOLING_PRIORITY_GEOJSON,

    }



    file_status = {

        name: {

            "exists": path.exists(),

            "path": str(path),

        }

        for name, path in files.items()

    }



    all_files_exist = all(

        item["exists"]

        for item in file_status.values()

    )



    result = {

        "status": (

            "ok"

            if all_files_exist

            else "missing_files"

        ),

        "component": "Urban Heat Research",

        "expected_grid_count": EXPECTED_GRID_COUNT,

        "historical_year_range": [

            HISTORICAL_START_YEAR,

            HISTORICAL_END_YEAR,

        ],

        "forecast_year_range": [

            FORECAST_START_YEAR,

            FORECAST_END_YEAR,

        ],

        "files": file_status,

    }



    if all_files_exist:

        result["loaded_counts"] = {

            "grid_metadata_rows": len(

                load_grid_metadata()

            ),

            "historical_rows": len(

                load_historical_heat()

            ),

            "future_rows": len(

                load_future_heat()

            ),

            "priority_rows": len(

                load_cooling_priority()

            ),

            "primary_scenario_rows": len(

                load_primary_scenarios()

            ),

        }



    return result





# ============================================================

# CACHE CONTROL

# ============================================================



def clear_urban_heat_cache() -> None:

    """

    Clear cached dashboard data.



    Useful if Step 34 outputs are regenerated while the

    backend server is still running.

    """



    load_grid_metadata.cache_clear()

    load_historical_heat.cache_clear()

    load_future_heat.cache_clear()

    load_cooling_priority.cache_clear()

    load_current_vs_forecast.cache_clear()

    load_area_heat_summary.cache_clear()

    load_area_priority_summary.cache_clear()

    load_primary_scenarios.cache_clear()

    load_green_scenarios.cache_clear()

    load_built_scenarios.cache_clear()

    load_combined_scenarios.cache_clear()

    load_blue_scenarios.cache_clear()



    get_grid_geojson.cache_clear()

    get_priority_geojson.cache_clear()

    _load_future_geojson.cache_clear()



# =====================================================================

# FRONTEND-FOCUSED READ FUNCTIONS

# =====================================================================





def get_current_vs_forecast(year=None, area_name=None):

    """

    Return 2025 actual LST compared with a selected future forecast year.

    """

    df = load_current_vs_forecast().copy()



    if year is not None:

        df = df[df["year"] == int(year)]



    if area_name is not None:

        df = df[

            df["area_name"].str.casefold()

            == str(area_name).strip().casefold()

        ]



    df = df.sort_values(

        ["year", "grid_id"]

    )



    return _records(df)





def get_grid_current_vs_forecast(grid_id, year=None):

    """

    Return current-vs-future comparison for one grid.

    """

    df = load_current_vs_forecast().copy()



    df = df[df["grid_id"] == str(grid_id)]



    if year is not None:

        df = df[df["year"] == int(year)]



    if df.empty:

        return None



    df = df.sort_values("year")



    return _records(df)





# =====================================================================

# XAI

# =====================================================================





def get_grid_xai(grid_id, year):

    """

    Return future SHAP explanation for one grid and one year.

    """

    df = load_future_heat().copy()



    row = df[

        (df["grid_id"] == str(grid_id))

        & (df["year"] == int(year))

    ]



    if row.empty:

        return None



    columns = [

        "grid_id",

        "area_name",

        "year",

        "predicted_lst_c",



        "top1_heating_feature",

        "top1_heating_shap_c",

        "top2_heating_feature",

        "top2_heating_shap_c",

        "top3_heating_feature",

        "top3_heating_shap_c",



        "top1_cooling_feature",

        "top1_cooling_shap_c",

        "top2_cooling_feature",

        "top2_cooling_shap_c",

        "top3_cooling_feature",

        "top3_cooling_shap_c",

    ]



    return _records(row[columns])[0]





# =====================================================================

# UNCERTAINTY

# =====================================================================





def get_grid_uncertainty(grid_id, year):

    """

    Return forecast uncertainty information for one grid and one year.



    These intervals are model/reference uncertainty outputs from Step 30.

    """

    df = load_future_heat().copy()



    row = df[

        (df["grid_id"] == str(grid_id))

        & (df["year"] == int(year))

    ]



    if row.empty:

        return None



    columns = [

        "grid_id",

        "area_name",

        "year",

        "predicted_lst_c",



        "forecast_pi95_lower_c",

        "forecast_pi95_upper_c",

        "forecast_pi95_width_c",



        "forecast_reference_pi90_lower_c",

        "forecast_reference_pi90_upper_c",

        "forecast_reference_pi95_lower_c",

        "forecast_reference_pi95_upper_c",

        "forecast_reference_pi95_width_c",



        "annual_radius_90_c",

        "annual_radius_95_c",

        "spatial_radius_90_c",

        "spatial_radius_95_c",

        "total_radius_90_c",

        "total_radius_95_c",



        "local_historical_centered_mae_c",

        "local_historical_centered_median_ae_c",

        "local_historical_centered_rmse_c",

        "historical_oos_year_count",

        "local_historical_error_tier",

        "relative_confidence_label",



        "development_error_count",

        "low_independent_year_count_flag",

        "uncertainty_method",

        "uncertainty_interpretation",

        "future_actual_available",

    ]



    return _records(row[columns])[0]





# =====================================================================

# HEAT TRAJECTORY

# =====================================================================





def get_trajectory(area_name=None):

    """

    Return one trajectory classification record per grid.



    Trajectory information is grid-level information and is repeated

    across future forecast years in the dashboard master table.

    Therefore duplicates are removed by grid_id.

    """

    df = load_future_heat().copy()



    if area_name is not None:

        df = df[

            df["area_name"].str.casefold()

            == str(area_name).strip().casefold()

        ]



    columns = [

        "grid_id",

        "area_name",

        "trajectory_class",



        "baseline_relative_heat_median",

        "recent_relative_heat_median",

        "long_term_relative_heat_median",



        "relative_sen_slope_percentile_per_year",

        "relative_direction_operational",



        "sen_slope_c_per_year",

        "mk_p_value",

        "mk_fdr_q_value",

        "absolute_lst_sen_direction",



        "trajectory_scope",

        "trajectory_is_absolute_warming_claim",

        "trajectory_is_statistical_significance_claim",

        "trajectory_rule_version",

    ]



    result = (

        df[columns]

        .drop_duplicates(subset=["grid_id"])

        .sort_values("grid_id")

    )



    return _records(result)





def get_grid_trajectory(grid_id):

    """

    Return trajectory classification for one grid.

    """

    df = load_future_heat().copy()



    row = df[df["grid_id"] == str(grid_id)]



    if row.empty:

        return None



    columns = [

        "grid_id",

        "area_name",

        "trajectory_class",



        "baseline_relative_heat_median",

        "recent_relative_heat_median",

        "long_term_relative_heat_median",



        "relative_sen_slope_percentile_per_year",

        "relative_direction_operational",



        "sen_slope_c_per_year",

        "mk_p_value",

        "mk_fdr_q_value",

        "absolute_lst_sen_direction",



        "trajectory_scope",

        "trajectory_is_absolute_warming_claim",

        "trajectory_is_statistical_significance_claim",

        "trajectory_rule_version",

    ]



    return _records(

        row[columns].drop_duplicates(subset=["grid_id"])

    )[0]





# =====================================================================

# SCENARIO FILTERING HELPERS

# =====================================================================





def _filter_scenario_dataframe(

    df,

    year=None,

    grid_id=None,

    area_name=None,

    scenario_id=None,

):

    """

    Common filtering logic for precomputed Step 29 scenario tables.

    """

    result = df.copy()



    if year is not None:

        result = result[result["year"] == int(year)]



    if grid_id is not None:

        result = result[result["grid_id"] == str(grid_id)]



    if area_name is not None:

        result = result[

            result["area_name"].str.casefold()

            == str(area_name).strip().casefold()

        ]



    if scenario_id is not None:

        result = result[

            result["scenario_id"].astype(str)

            == str(scenario_id)

        ]



    return result





# =====================================================================

# GREEN SCENARIOS

# =====================================================================





def get_green_scenario_results(

    year=None,

    grid_id=None,

    area_name=None,

    scenario_id=None,

):

    df = _filter_scenario_dataframe(

        load_green_scenarios(),

        year=year,

        grid_id=grid_id,

        area_name=area_name,

        scenario_id=scenario_id,

    )



    df = df.sort_values(

        ["year", "grid_id", "scenario_id"]

    )



    return _records(df)





# =====================================================================

# BUILT-UP SCENARIOS

# =====================================================================





def get_built_scenario_results(

    year=None,

    grid_id=None,

    area_name=None,

    scenario_id=None,

):

    df = _filter_scenario_dataframe(

        load_built_scenarios(),

        year=year,

        grid_id=grid_id,

        area_name=area_name,

        scenario_id=scenario_id,

    )



    df = df.sort_values(

        ["year", "grid_id", "scenario_id"]

    )



    return _records(df)





# =====================================================================

# COMBINED SCENARIOS

# =====================================================================





def get_combined_scenario_results(

    year=None,

    grid_id=None,

    area_name=None,

    scenario_id=None,

):

    df = _filter_scenario_dataframe(

        load_combined_scenarios(),

        year=year,

        grid_id=grid_id,

        area_name=area_name,

        scenario_id=scenario_id,

    )



    df = df.sort_values(

        ["year", "grid_id", "scenario_id"]

    )



    return _records(df)





# =====================================================================

# BLUE SCENARIOS

# =====================================================================





def get_blue_scenario_results(

    year=None,

    grid_id=None,

    area_name=None,

    scenario_id=None,

):

    df = _filter_scenario_dataframe(

        load_blue_scenarios(),

        year=year,

        grid_id=grid_id,

        area_name=area_name,

        scenario_id=scenario_id,

    )



    df = df.sort_values(

        ["year", "grid_id", "scenario_id"]

    )



    return _records(df)





# =====================================================================

# ALL PRECOMPUTED SCENARIOS FOR ONE GRID

# =====================================================================





def get_grid_scenarios(grid_id, year=None):

    """

    Return every Step 29 precomputed scenario family for one grid.

    """

    grid_id = str(grid_id)



    green = get_green_scenario_results(

        grid_id=grid_id,

        year=year,

    )



    built = get_built_scenario_results(

        grid_id=grid_id,

        year=year,

    )



    combined = get_combined_scenario_results(

        grid_id=grid_id,

        year=year,

    )



    blue = get_blue_scenario_results(

        grid_id=grid_id,

        year=year,

    )



    if not any([green, built, combined, blue]):

        return None



    return {

        "grid_id": grid_id,

        "year": int(year) if year is not None else None,

        "green": green,

        "built": built,

        "combined": combined,

        "blue": blue,

    }



# =====================================================================

# LIGHTWEIGHT MAP / GEOJSON HELPERS

# =====================================================================





def _build_feature_collection(features, source_geojson=None):

    """

    Build a clean GeoJSON FeatureCollection.



    Preserve optional top-level metadata such as CRS when available.

    """

    result = {

        "type": "FeatureCollection",

        "features": features,

    }



    if source_geojson:

        for key in ("crs", "name", "bbox"):

            if key in source_geojson:

                result[key] = source_geojson[key]



    return result





# =====================================================================

# HISTORICAL HEAT MAP

# =====================================================================





def get_historical_geojson_by_year(year):

    """

    Join one historical year of actual LST data to the stable

    2,194-grid WGS84 geometry layer.

    """

    year = int(year)



    if year < 2015 or year > 2025:

        raise ValueError(

            "Historical map year must be between 2015 and 2025."

        )



    historical_records = get_historical_by_year(year)



    historical_lookup = {

        str(record["grid_id"]): record

        for record in historical_records

    }



    grid_geojson = get_grid_geojson()



    features = []



    for feature in grid_geojson.get("features", []):

        base_properties = feature.get("properties", {})

        grid_id = str(base_properties.get("grid_id"))



        historical = historical_lookup.get(grid_id)



        if historical is None:

            continue



        properties = {

            "grid_id": grid_id,

            "area_name": historical.get(

                "area_name",

                base_properties.get("area_name"),

            ),

            "year": year,

            "actual_lst_c": historical.get("actual_lst_c"),

        }



        features.append(

            {

                "type": "Feature",

                "geometry": feature.get("geometry"),

                "properties": properties,

            }

        )



    return _build_feature_collection(

        features,

        source_geojson=grid_geojson,

    )





# =====================================================================

# LIGHTWEIGHT FUTURE FORECAST MAP

# =====================================================================





def get_future_geojson_lite(year):

    """

    Return a lightweight forecast GeoJSON for frontend map rendering.



    Detailed XAI, uncertainty and scenario information should be loaded

    only after the user clicks a grid.

    """

    source = get_future_geojson_by_year(year)



    keep_properties = [

        "grid_id",

        "area_name",

        "year",

        "predicted_lst_c",

        "actual_2025_lst_c",

        "change_from_2025_grid_actual_c",

        "trajectory_class",

        "relative_confidence_label",

        "forecast_pi95_lower_c",

        "forecast_pi95_upper_c",

        "topsis_rank",

        "topsis_priority_score",

        "priority_percentile",

    ]



    features = []



    for feature in source.get("features", []):

        original = feature.get("properties", {})



        properties = {

            key: original.get(key)

            for key in keep_properties

        }


        properties["trajectory_class"] = _normalize_trajectory_class(

            properties.get("trajectory_class")

        )



        features.append(

            {

                "type": "Feature",

                "geometry": feature.get("geometry"),

                "properties": properties,

            }

        )



    return _build_feature_collection(

        features,

        source_geojson=source,

    )





# =====================================================================

# TRAJECTORY MAP

# =====================================================================





def get_trajectory_geojson():

    """

    Return one grid polygon per trajectory classification.



    The trajectory layer is relative to the Kaduwela spatial context

    and is not a statistical-significance or absolute-warming claim.

    """

    source = get_priority_geojson()



    keep_properties = [

        "grid_id",

        "area_name",

        "trajectory_class",

        "baseline_relative_heat_median",

        "recent_relative_heat_median",

        "long_term_relative_heat_median",

        "relative_sen_slope_percentile_per_year",

        "relative_direction_operational",

        "sen_slope_c_per_year",

        "mk_p_value",

        "mk_fdr_q_value",

        "absolute_lst_sen_direction",

        "trajectory_scope",

        "trajectory_is_absolute_warming_claim",

        "trajectory_is_statistical_significance_claim",

        "trajectory_rule_version",

    ]



    features = []



    for feature in source.get("features", []):

        original = feature.get("properties", {})



        properties = {

            key: original.get(key)

            for key in keep_properties

        }


        properties["trajectory_class"] = _normalize_trajectory_class(

            properties.get("trajectory_class")

        )



        features.append(

            {

                "type": "Feature",

                "geometry": feature.get("geometry"),

                "properties": properties,

            }

        )



    return _build_feature_collection(

        features,

        source_geojson=source,

    )





# =====================================================================

# LIGHTWEIGHT PRIORITY MAP

# =====================================================================





def get_priority_geojson_lite():

    """

    Return only the cooling-priority fields needed for frontend mapping.

    """

    source = get_priority_geojson()



    keep_properties = [

        "grid_id",

        "area_name",

        "topsis_priority_score",

        "topsis_rank",

        "priority_percentile",

        "trajectory_class",

        "mean_sensitivity_rank",

        "worst_sensitivity_rank",

        "mean_abs_rank_shift",

        "pct_scenarios_in_top100",

    ]



    features = []



    for feature in source.get("features", []):

        original = feature.get("properties", {})



        properties = {

            key: original.get(key)

            for key in keep_properties

        }


        properties["trajectory_class"] = _normalize_trajectory_class(

            properties.get("trajectory_class")

        )



        features.append(

            {

                "type": "Feature",

                "geometry": feature.get("geometry"),

                "properties": properties,

            }

        )



    return _build_feature_collection(

        features,

        source_geojson=source,

    )