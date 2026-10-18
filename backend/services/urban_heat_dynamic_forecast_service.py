from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import URBAN_HEAT_ROOT


# ============================================================
# DYNAMIC FIVE-YEAR FORECAST PATHS
# ============================================================

DYNAMIC_FORECAST_ROOT = (
    URBAN_HEAT_ROOT
    / "experiments"
    / "dynamic_5year_forecast"
)

DYNAMIC_OUTPUT_DIR = (
    DYNAMIC_FORECAST_ROOT
    / "outputs"
)

DYNAMIC_REPORT_DIR = (
    DYNAMIC_OUTPUT_DIR
    / "reports"
)

DYNAMIC_PREDICTION_DIR = (
    DYNAMIC_OUTPUT_DIR
    / "predictions"
)

DYNAMIC_MAP_DIR = (
    DYNAMIC_OUTPUT_DIR
    / "maps"
)

DYNAMIC_TRAJECTORY_DIR = (
    DYNAMIC_OUTPUT_DIR
    / "trajectory"
)

DYNAMIC_UPLOAD_DIR = (
    DYNAMIC_FORECAST_ROOT
    / "uploads"
)

DYNAMIC_MODEL_DIR = (
    DYNAMIC_FORECAST_ROOT
    / "models"
)

DYNAMIC_SOURCE_DIR = (
    DYNAMIC_FORECAST_ROOT
    / "src"
)


FINAL_MANIFEST_FILE = (
    DYNAMIC_REPORT_DIR
    / "06_dynamic_service_manifest.json"
)

FINAL_AUDIT_FILE = (
    DYNAMIC_REPORT_DIR
    / "06_final_dynamic_service_audit.csv"
)

FINAL_HASH_FILE = (
    DYNAMIC_REPORT_DIR
    / "06_dynamic_service_hashes.csv"
)

VALIDATED_HISTORY_FILE = (
    DYNAMIC_OUTPUT_DIR
    / "01_validated_history.csv"
)

TRAJECTORY_FILE = (
    DYNAMIC_TRAJECTORY_DIR
    / "02_dynamic_trajectory_classes.csv"
)

TRAJECTORY_GEOJSON_FILE = (
    DYNAMIC_TRAJECTORY_DIR
    / "02_dynamic_trajectory_wgs84.geojson"
)


EXPECTED_GRID_COUNT = 2194
EXPECTED_FORECAST_HORIZON = 5


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _require_file(
    path: Path,
    label: str,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"{label} was not found: {path}"
        )


def _read_json(
    path: Path,
    label: str,
) -> dict[str, Any]:

    _require_file(
        path,
        label,
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def _read_csv(
    path: Path,
    label: str,
) -> pd.DataFrame:

    _require_file(
        path,
        label,
    )

    df = pd.read_csv(
        path,
        dtype={
            "grid_id": str,
            "area_name": str,
        },
        low_memory=False,
    )

    if "grid_id" in df.columns:

        df["grid_id"] = (
            df["grid_id"]
            .astype(str)
            .str.strip()
        )

    if "area_name" in df.columns:

        df["area_name"] = (
            df["area_name"]
            .astype(str)
            .str.strip()
        )

    if "year" in df.columns:

        df["year"] = pd.to_numeric(
            df["year"],
            errors="raise",
        ).astype(int)

    return df


def _records(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:

    clean = (
        df
        .astype(object)
        .where(
            pd.notna(df),
            None,
        )
    )

    return clean.to_dict(
        orient="records"
    )


def _load_final_manifest() -> dict[str, Any]:

    return _read_json(
        FINAL_MANIFEST_FILE,
        "Dynamic forecast final manifest",
    )


def _get_forecast_years() -> list[int]:

    manifest = _load_final_manifest()

    years = manifest.get(
        "forecast_years"
    )

    if not years:

        latest_year = manifest.get(
            "latest_observed_year"
        )

        if latest_year is None:

            raise RuntimeError(
                "Dynamic manifest does not contain "
                "forecast_years or latest_observed_year."
            )

        latest_year = int(
            latest_year
        )

        years = list(
            range(
                latest_year + 1,
                latest_year
                + EXPECTED_FORECAST_HORIZON
                + 1,
            )
        )

    years = [
        int(year)
        for year in years
    ]

    if len(years) != EXPECTED_FORECAST_HORIZON:

        raise RuntimeError(
            "Dynamic forecast manifest does not contain "
            "exactly five forecast years."
        )

    return years


def _prediction_file() -> Path:

    years = _get_forecast_years()

    return (
        DYNAMIC_PREDICTION_DIR
        / (
            "05_dynamic_grid_lst_predictions_"
            f"{years[0]}_{years[-1]}.csv"
        )
    )


def _summary_file() -> Path:

    years = _get_forecast_years()

    return (
        DYNAMIC_PREDICTION_DIR
        / (
            "05_dynamic_future_year_summary_"
            f"{years[0]}_{years[-1]}.csv"
        )
    )


def _combined_map_file() -> Path:

    years = _get_forecast_years()

    return (
        DYNAMIC_MAP_DIR
        / (
            "05_dynamic_future_heat_"
            f"{years[0]}_{years[-1]}"
            "_wgs84.geojson"
        )
    )


def _year_map_file(
    year: int,
) -> Path:

    return (
        DYNAMIC_MAP_DIR
        / (
            "05_dynamic_future_heat_"
            f"{int(year)}"
            "_wgs84.geojson"
        )
    )


# ============================================================
# STATUS / MANIFEST
# ============================================================

def get_dynamic_forecast_status() -> dict[str, Any]:

    manifest = _load_final_manifest()

    forecast_years = (
        _get_forecast_years()
    )

    latest_year = int(
        manifest[
            "latest_observed_year"
        ]
    )

    files = {
        "validated_history":
            VALIDATED_HISTORY_FILE,

        "trajectory":
            TRAJECTORY_FILE,

        "trajectory_geojson":
            TRAJECTORY_GEOJSON_FILE,

        "forecast_predictions":
            _prediction_file(),

        "forecast_summary":
            _summary_file(),

        "combined_forecast_map":
            _combined_map_file(),

        "final_audit":
            FINAL_AUDIT_FILE,

        "final_hashes":
            FINAL_HASH_FILE,
    }

    file_status = {
        name: {
            "exists":
                path.exists(),

            "path":
                str(path),
        }
        for name, path
        in files.items()
    }

    all_required_files_exist = all(
        item["exists"]
        for item in file_status.values()
    )

    module_status = manifest.get(
        "status",
        "UNKNOWN",
    )

    ready = (
        module_status
        == "READY_FOR_API_INTEGRATION"
        and
        all_required_files_exist
    )

    return {
        "status":
            "ready"
            if ready
            else "not_ready",

        "module":
            "Rolling Five-Year Urban Heat Forecasting Service",

        "module_status":
            module_status,

        "latest_observed_year":
            latest_year,

        "forecast_years":
            forecast_years,

        "forecast_start_year":
            forecast_years[0],

        "forecast_end_year":
            forecast_years[-1],

        "forecast_horizon_years":
            EXPECTED_FORECAST_HORIZON,

        "expected_grid_count":
            EXPECTED_GRID_COUNT,

        "all_required_files_exist":
            all_required_files_exist,

        "files":
            file_status,
    }


def get_dynamic_forecast_manifest() -> dict[str, Any]:

    return _load_final_manifest()


# ============================================================
# FORECAST DATA
# ============================================================

def get_dynamic_forecast_by_year(
    year: int,
) -> list[dict[str, Any]]:

    year = int(
        year
    )

    valid_years = (
        _get_forecast_years()
    )

    if year not in valid_years:

        raise ValueError(
            "Dynamic forecast year must be one of "
            f"{valid_years}."
        )

    df = _read_csv(
        _prediction_file(),
        "Dynamic five-year forecast predictions",
    )

    result = (
        df[
            df["year"]
            == year
        ]
        .copy()
        .sort_values(
            "grid_id"
        )
    )

    if len(result) != EXPECTED_GRID_COUNT:

        raise RuntimeError(
            f"Forecast year {year} contains "
            f"{len(result)} rows; "
            f"expected {EXPECTED_GRID_COUNT}."
        )

    return _records(
        result
    )


def get_dynamic_grid_forecast(
    grid_id: str,
    year: int,
) -> dict[str, Any] | None:

    year = int(
        year
    )

    valid_years = (
        _get_forecast_years()
    )

    if year not in valid_years:

        raise ValueError(
            "Dynamic forecast year must be one of "
            f"{valid_years}."
        )

    df = _read_csv(
        _prediction_file(),
        "Dynamic five-year forecast predictions",
    )

    row = df[
        (
            df["grid_id"]
            == str(grid_id).strip()
        )
        &
        (
            df["year"]
            == year
        )
    ]

    if row.empty:

        return None

    return _records(
        row.head(1)
    )[0]


def get_dynamic_forecast_summary() -> list[dict[str, Any]]:

    df = _read_csv(
        _summary_file(),
        "Dynamic forecast year summary",
    )

    return _records(
        df.sort_values(
            "year"
        )
    )


# ============================================================
# TRAJECTORY DATA
# ============================================================

def get_dynamic_trajectory(
    area_name: str | None = None,
) -> list[dict[str, Any]]:

    df = _read_csv(
        TRAJECTORY_FILE,
        "Dynamic heat trajectory",
    )

    if area_name is not None:

        df = df[
            df["area_name"]
            .str.casefold()
            ==
            str(area_name)
            .strip()
            .casefold()
        ]

    df = df.sort_values(
        "grid_id"
    )

    return _records(
        df
    )


def get_dynamic_grid_trajectory(
    grid_id: str,
) -> dict[str, Any] | None:

    df = _read_csv(
        TRAJECTORY_FILE,
        "Dynamic heat trajectory",
    )

    row = df[
        df["grid_id"]
        ==
        str(grid_id).strip()
    ]

    if row.empty:

        return None

    return _records(
        row.head(1)
    )[0]


# ============================================================
# GEOJSON MAP DATA
# ============================================================

def get_dynamic_forecast_map(
    year: int,
) -> dict[str, Any]:

    year = int(
        year
    )

    valid_years = (
        _get_forecast_years()
    )

    if year not in valid_years:

        raise ValueError(
            "Dynamic forecast map year must be one of "
            f"{valid_years}."
        )

    path = _year_map_file(
        year
    )

    return _read_json(
        path,
        f"Dynamic forecast GeoJSON for {year}",
    )


def get_dynamic_combined_forecast_map() -> dict[str, Any]:

    return _read_json(
        _combined_map_file(),
        "Dynamic combined forecast GeoJSON",
    )


def get_dynamic_trajectory_map() -> dict[str, Any]:

    return _read_json(
        TRAJECTORY_GEOJSON_FILE,
        "Dynamic trajectory GeoJSON",
    )