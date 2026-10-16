"""FastAPI routes for Urban Heat dynamic scenario simulation."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.urban_heat_schemas import UrbanHeatScenarioRequest
from backend.services.urban_heat_scenario_service import (
    get_dynamic_scenario_options,
    simulate_custom_scenario,
)


router = APIRouter(
    prefix="/api/urban-heat/scenario",
    tags=["Urban Heat - Dynamic Scenarios"],
)


@router.get("/options")
def dynamic_scenario_options():
    """Return supported years, scenario variables, bounds, and research levels."""
    try:
        return get_dynamic_scenario_options()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/simulate")
def dynamic_scenario_simulate(request: UrbanHeatScenarioRequest):
    """Run one research-consistent dynamic scenario for one grid and target year."""
    try:
        return simulate_custom_scenario(
            grid_id=request.grid_id,
            year=request.year,
            family=request.scenario_family.value,
            green_increase_pp=request.green_increase_pp,
            ndbi_reduction=request.ndbi_reduction,
            ndwi_increase=request.ndwi_increase,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
