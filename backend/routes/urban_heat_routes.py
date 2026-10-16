from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.config import (
    HISTORICAL_START_YEAR,
    HISTORICAL_END_YEAR,
    FORECAST_START_YEAR,
    FORECAST_END_YEAR,
)

from backend.services.urban_heat_data_service import (
    get_data_health,
    get_dashboard_manifest,
    get_area_names,
    get_grid_metadata,
    get_historical_by_year,
    get_grid_history,
    get_forecast_by_year,
    get_grid_forecast,
    get_grid_all_future_years,
    get_priority_ranking,
    get_grid_priority,
    get_primary_scenario,
    get_area_heat,
    get_area_priority,
    get_grid_geojson,
    get_future_geojson_by_year,
    get_priority_geojson,

    get_current_vs_forecast,
    get_grid_current_vs_forecast,
    get_grid_xai,
    get_grid_uncertainty,
    get_trajectory,
    get_grid_trajectory,
    get_green_scenario_results,
    get_built_scenario_results,
    get_combined_scenario_results,
    get_blue_scenario_results,
    get_grid_scenarios,

    get_historical_geojson_by_year,
    get_future_geojson_lite,
    get_trajectory_geojson,
    get_priority_geojson_lite,
)


# ============================================================
# URBAN HEAT ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/urban-heat",
    tags=["Urban Heat"],
)


# ============================================================
# HEALTH / SYSTEM STATUS
# ============================================================

@router.get("/health")
def urban_heat_health():
    """
    Basic Urban Heat backend health check.
    """
    return {
        "status": "ok",
        "component": "Urban Heat Research",
        "message": "Urban Heat API routes are working.",
    }


@router.get("/data-health")
def urban_heat_data_health():
    """
    Check whether the Step 34 dashboard files are available
    and readable by the backend.
    """
    try:
        return get_data_health()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Urban Heat data health check failed: {exc}",
        )


@router.get("/manifest")
def urban_heat_manifest():
    """
    Return the Step 34 dashboard data manifest.
    """
    try:
        return get_dashboard_manifest()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load dashboard manifest: {exc}",
        )


# ============================================================
# GRID METADATA / AREAS
# ============================================================

@router.get("/areas")
def urban_heat_areas():
    """
    Return the available Urban Heat area names.
    """
    try:
        areas = get_area_names()

        return {
            "count": len(areas),
            "areas": areas,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load area names: {exc}",
        )


@router.get("/grid-metadata/{grid_id}")
def urban_heat_grid_metadata(grid_id: str):
    """
    Return metadata for one grid.
    """
    try:
        result = get_grid_metadata(grid_id)

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Grid '{grid_id}' was not found.",
            )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load grid metadata: {exc}",
        )


# ============================================================
# HISTORICAL HEAT
# ============================================================

@router.get("/historical")
def urban_heat_historical(
    year: int = Query(
        ...,
        ge=HISTORICAL_START_YEAR,
        le=HISTORICAL_END_YEAR,
        description="Historical year from 2015 to 2025.",
    )
):
    """
    Return all grid-level historical heat records
    for one selected year.
    """
    try:
        records = get_historical_by_year(year)

        return {
            "year": year,
            "count": len(records),
            "data": records,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load historical heat data: {exc}",
        )


@router.get("/history/{grid_id}")
def urban_heat_grid_history(grid_id: str):
    """
    Return the 2015-2025 historical LST time series
    for one grid.
    """
    try:
        records = get_grid_history(grid_id)

        if not records:
            raise HTTPException(
                status_code=404,
                detail=f"No historical data found for grid '{grid_id}'.",
            )

        return {
            "grid_id": grid_id,
            "count": len(records),
            "data": records,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load grid history: {exc}",
        )


# ============================================================
# FUTURE FORECAST
# ============================================================

@router.get("/forecast")
def urban_heat_forecast(
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
        description="Forecast year from 2026 to 2030.",
    )
):
    """
    Return forecast information for all 2,194 grids
    for one selected future year.
    """
    try:
        records = get_forecast_by_year(year)

        return {
            "year": year,
            "count": len(records),
            "data": records,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load forecast data: {exc}",
        )


@router.get("/forecast/grid/{grid_id}")
def urban_heat_grid_forecast(
    grid_id: str,
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
        description="Forecast year from 2026 to 2030.",
    ),
):
    """
    Return the future forecast record for one grid
    and one selected future year.
    """
    try:
        result = get_grid_forecast(
            grid_id=grid_id,
            year=year,
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No forecast found for grid '{grid_id}' "
                    f"in year {year}."
                ),
            )

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load grid forecast: {exc}",
        )


@router.get("/forecast/grid/{grid_id}/all")
def urban_heat_grid_all_forecasts(grid_id: str):
    """
    Return all 2026-2030 forecasts for one grid.
    """
    try:
        records = get_grid_all_future_years(grid_id)

        if not records:
            raise HTTPException(
                status_code=404,
                detail=f"No future forecasts found for grid '{grid_id}'.",
            )

        return {
            "grid_id": grid_id,
            "count": len(records),
            "data": records,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load grid forecasts: {exc}",
        )


# ============================================================
# GRID CLICK DETAILS
# ============================================================

@router.get("/grid/{grid_id}")
def urban_heat_grid_details(
    grid_id: str,
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
        description="Selected future year from 2026 to 2030.",
    ),
):
    """
    Main frontend grid-click endpoint.

    Returns:
    - grid metadata
    - selected-year forecast
    - TOPSIS cooling priority
    - selected-year primary scenario information
    """
    try:
        metadata = get_grid_metadata(grid_id)

        if metadata is None:
            raise HTTPException(
                status_code=404,
                detail=f"Grid '{grid_id}' was not found.",
            )

        forecast = get_grid_forecast(
            grid_id=grid_id,
            year=year,
        )

        priority = get_grid_priority(grid_id)

        scenario = get_primary_scenario(
            grid_id=grid_id,
            year=year,
        )

        if forecast is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No future forecast found for grid "
                    f"'{grid_id}' in {year}."
                ),
            )

        return {
            "grid_id": grid_id,
            "year": year,
            "metadata": metadata,
            "forecast": forecast,
            "priority": priority,
            "primary_scenario": scenario,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load grid details: {exc}",
        )


# ============================================================
# COOLING PRIORITY
# ============================================================

@router.get("/priority")
def urban_heat_priority(
    limit: Optional[int] = Query(
        default=None,
        ge=1,
        description="Optional number of top-ranked grids to return.",
    ),
    area_name: Optional[str] = Query(
        default=None,
        description="Optional area filter.",
    ),
):
    """
    Return the Step 32 TOPSIS cooling-priority ranking.
    """
    try:
        records = get_priority_ranking(
            limit=limit,
            area_name=area_name,
        )

        return {
            "count": len(records),
            "area_name": area_name,
            "data": records,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load cooling-priority data: {exc}",
        )


@router.get("/priority/grid/{grid_id}")
def urban_heat_grid_priority(grid_id: str):
    """
    Return cooling-priority information for one grid.
    """
    try:
        result = get_grid_priority(grid_id)

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"No priority record found for grid '{grid_id}'.",
            )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load grid priority: {exc}",
        )


# ============================================================
# SCENARIOS
# ============================================================

@router.get("/scenario/grid/{grid_id}")
def urban_heat_grid_scenario(
    grid_id: str,
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
        description="Scenario year from 2026 to 2030.",
    ),
):
    """
    Return the selected primary planning scenarios
    for one grid and future year.
    """
    try:
        result = get_primary_scenario(
            grid_id=grid_id,
            year=year,
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No primary scenario data found for "
                    f"grid '{grid_id}' in {year}."
                ),
            )

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load scenario data: {exc}",
        )


# ============================================================
# AREA-LEVEL DATA
# ============================================================

@router.get("/areas/heat")
def urban_heat_area_heat(
    area_name: Optional[str] = Query(
        default=None,
        description="Optional area-name filter.",
    ),
    year: Optional[int] = Query(
        default=None,
        ge=HISTORICAL_START_YEAR,
        le=FORECAST_END_YEAR,
        description="Optional year from 2015 to 2030.",
    ),
):
    """
    Return area-level heat summaries for 2015-2030.
    """
    try:
        records = get_area_heat(
            area_name=area_name,
            year=year,
        )

        return {
            "count": len(records),
            "area_name": area_name,
            "year": year,
            "data": records,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load area heat summary: {exc}",
        )


@router.get("/areas/priority")
def urban_heat_area_priority(
    area_name: Optional[str] = Query(
        default=None,
        description="Optional area-name filter.",
    )
):
    """
    Return area-level cooling-priority summaries.
    """
    try:
        records = get_area_priority(
            area_name=area_name,
        )

        return {
            "count": len(records),
            "area_name": area_name,
            "data": records,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load area priority summary: {exc}",
        )


# ============================================================
# MAP / GEOJSON ENDPOINTS
# ============================================================

@router.get("/map/grids")
def urban_heat_grid_map():
    """
    Return the 2,194 grid polygons in WGS84 GeoJSON.

    This is the stable base geometry layer for the frontend map.
    """
    try:
        return get_grid_geojson()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load grid GeoJSON: {exc}",
        )


@router.get("/map/forecast")
def urban_heat_forecast_map(
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
        description="Forecast-map year from 2026 to 2030.",
    )
):
    """
    Return forecast GeoJSON for one selected future year.

    The React frontend can color each polygon using the
    predicted LST value stored in its properties.
    """
    try:
        geojson = get_future_geojson_by_year(year)

        return geojson

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load forecast GeoJSON: {exc}",
        )


@router.get("/map/priority")
def urban_heat_priority_map():
    """
    Return the cooling-priority GeoJSON layer.
    """
    try:
        return get_priority_geojson()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load priority GeoJSON: {exc}",
        )

# =====================================================================
# CURRENT VS FUTURE FORECAST
# =====================================================================


@router.get("/current-vs-forecast")
def current_vs_forecast(
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
    ),
    area_name: Optional[str] = Query(default=None),
):
    data = get_current_vs_forecast(
        year=year,
        area_name=area_name,
    )

    return {
        "year": year,
        "area_name": area_name,
        "count": len(data),
        "data": data,
    }


@router.get("/current-vs-forecast/grid/{grid_id}")
def current_vs_forecast_grid(
    grid_id: str,
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
    ),
):
    data = get_grid_current_vs_forecast(
        grid_id=grid_id,
        year=year,
    )

    if data is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Current-vs-forecast data not found for "
                f"grid '{grid_id}' and year {year}."
            ),
        )

    return {
        "grid_id": grid_id,
        "year": year,
        "data": data,
    }


# =====================================================================
# XAI / SHAP
# =====================================================================


@router.get("/xai/grid/{grid_id}")
def grid_xai(
    grid_id: str,
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
    ),
):
    data = get_grid_xai(
        grid_id=grid_id,
        year=year,
    )

    if data is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"XAI data not found for "
                f"grid '{grid_id}' and year {year}."
            ),
        )

    return data


# =====================================================================
# FORECAST UNCERTAINTY
# =====================================================================


@router.get("/uncertainty/grid/{grid_id}")
def grid_uncertainty(
    grid_id: str,
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
    ),
):
    data = get_grid_uncertainty(
        grid_id=grid_id,
        year=year,
    )

    if data is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Uncertainty data not found for "
                f"grid '{grid_id}' and year {year}."
            ),
        )

    return data


# =====================================================================
# HEAT TRAJECTORY
# =====================================================================


@router.get("/trajectory")
def trajectory(
    area_name: Optional[str] = Query(default=None),
):
    data = get_trajectory(
        area_name=area_name,
    )

    return {
        "area_name": area_name,
        "count": len(data),
        "data": data,
        "interpretation_note": (
            "Trajectory classes describe relative spatial heat "
            "trajectories within Kaduwela and must not be interpreted "
            "as statistical significance or absolute warming claims."
        ),
    }


@router.get("/trajectory/grid/{grid_id}")
def trajectory_grid(grid_id: str):
    data = get_grid_trajectory(grid_id)

    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Trajectory data not found for grid '{grid_id}'.",
        )

    return data


# =====================================================================
# PRECOMPUTED GREEN SCENARIOS
# =====================================================================


@router.get("/scenarios/green")
def green_scenarios(
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
    ),
    grid_id: Optional[str] = Query(default=None),
    area_name: Optional[str] = Query(default=None),
    scenario_id: Optional[str] = Query(default=None),
):
    data = get_green_scenario_results(
        year=year,
        grid_id=grid_id,
        area_name=area_name,
        scenario_id=scenario_id,
    )

    return {
        "scenario_family": "green",
        "year": year,
        "grid_id": grid_id,
        "area_name": area_name,
        "scenario_id": scenario_id,
        "count": len(data),
        "data": data,
        "interpretation_note": (
            "Results are precomputed model-based planning scenarios, "
            "not causal estimates."
        ),
    }


# =====================================================================
# PRECOMPUTED BUILT-UP SCENARIOS
# =====================================================================


@router.get("/scenarios/built")
def built_scenarios(
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
    ),
    grid_id: Optional[str] = Query(default=None),
    area_name: Optional[str] = Query(default=None),
    scenario_id: Optional[str] = Query(default=None),
):
    data = get_built_scenario_results(
        year=year,
        grid_id=grid_id,
        area_name=area_name,
        scenario_id=scenario_id,
    )

    return {
        "scenario_family": "built",
        "year": year,
        "grid_id": grid_id,
        "area_name": area_name,
        "scenario_id": scenario_id,
        "count": len(data),
        "data": data,
        "interpretation_note": (
            "Results are precomputed model-based planning scenarios, "
            "not causal estimates."
        ),
    }


# =====================================================================
# PRECOMPUTED COMBINED SCENARIOS
# =====================================================================


@router.get("/scenarios/combined")
def combined_scenarios(
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
    ),
    grid_id: Optional[str] = Query(default=None),
    area_name: Optional[str] = Query(default=None),
    scenario_id: Optional[str] = Query(default=None),
):
    data = get_combined_scenario_results(
        year=year,
        grid_id=grid_id,
        area_name=area_name,
        scenario_id=scenario_id,
    )

    return {
        "scenario_family": "combined",
        "year": year,
        "grid_id": grid_id,
        "area_name": area_name,
        "scenario_id": scenario_id,
        "count": len(data),
        "data": data,
        "interpretation_note": (
            "Results are precomputed model-based planning scenarios, "
            "not causal estimates."
        ),
    }


# =====================================================================
# PRECOMPUTED BLUE SCENARIOS
# =====================================================================


@router.get("/scenarios/blue")
def blue_scenarios(
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
    ),
    grid_id: Optional[str] = Query(default=None),
    area_name: Optional[str] = Query(default=None),
    scenario_id: Optional[str] = Query(default=None),
):
    data = get_blue_scenario_results(
        year=year,
        grid_id=grid_id,
        area_name=area_name,
        scenario_id=scenario_id,
    )

    return {
        "scenario_family": "blue",
        "year": year,
        "grid_id": grid_id,
        "area_name": area_name,
        "scenario_id": scenario_id,
        "count": len(data),
        "data": data,
        "interpretation_note": (
            "Results are precomputed model-based planning scenarios, "
            "not causal estimates."
        ),
    }


# =====================================================================
# ALL PRECOMPUTED SCENARIOS FOR ONE GRID
# =====================================================================


@router.get("/scenarios/grid/{grid_id}")
def all_grid_scenarios(
    grid_id: str,
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
    ),
):
    data = get_grid_scenarios(
        grid_id=grid_id,
        year=year,
    )

    if data is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Scenario data not found for "
                f"grid '{grid_id}' and year {year}."
            ),
        )

    return {
        **data,
        "interpretation_note": (
            "Scenario results are model-based planning estimates "
            "derived from precomputed Step 29 scenarios."
        ),
    }

# =====================================================================
# LIGHTWEIGHT / FRONTEND MAP ENDPOINTS
# =====================================================================


@router.get("/map/historical")
def urban_heat_historical_map(
    year: int = Query(
        ...,
        ge=HISTORICAL_START_YEAR,
        le=HISTORICAL_END_YEAR,
        description="Historical map year from 2015 to 2025.",
    )
):
    """
    Return historical actual-LST GeoJSON for one selected year.
    """
    try:
        return get_historical_geojson_by_year(year)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load historical heat map: {exc}",
        )


@router.get("/map/forecast-lite")
def urban_heat_forecast_map_lite(
    year: int = Query(
        ...,
        ge=FORECAST_START_YEAR,
        le=FORECAST_END_YEAR,
        description="Forecast-map year from 2026 to 2030.",
    )
):
    """
    Return lightweight forecast GeoJSON for frontend rendering.
    """
    try:
        return get_future_geojson_lite(year)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load lightweight forecast map: {exc}",
        )


@router.get("/map/trajectory")
def urban_heat_trajectory_map():
    """
    Return trajectory-class GeoJSON for all Urban Heat grids.
    """
    try:
        return get_trajectory_geojson()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load trajectory map: {exc}",
        )


@router.get("/map/priority-lite")
def urban_heat_priority_map_lite():
    """
    Return lightweight cooling-priority GeoJSON.
    """
    try:
        return get_priority_geojson_lite()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load lightweight priority map: {exc}",
        )