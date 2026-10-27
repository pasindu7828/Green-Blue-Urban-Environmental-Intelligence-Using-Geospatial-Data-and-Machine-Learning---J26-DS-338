from typing import Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    UploadFile,
    File,
)

from backend.services.urban_heat_dynamic_forecast_service import (
    get_dynamic_forecast_status,
    get_dynamic_forecast_manifest,
    get_dynamic_forecast_by_year,
    get_dynamic_grid_forecast,
    get_dynamic_forecast_summary,
    get_dynamic_trajectory,
    get_dynamic_grid_trajectory,
    get_dynamic_forecast_map,
    get_dynamic_combined_forecast_map,
    get_dynamic_trajectory_map,
)

from backend.services.urban_heat_dynamic_forecast_pipeline_service import (
    stage_dynamic_history_upload,
    validate_staged_dynamic_history,
    run_dynamic_forecast_pipeline,
    get_dynamic_pipeline_state,
)

# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/urban-heat/dynamic-forecast",
    tags=["Urban Heat - Dynamic Forecast"],
)


# ============================================================
# STATUS / MANIFEST
# ============================================================

@router.get("/status")
def dynamic_forecast_status():

    try:

        return get_dynamic_forecast_status()

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Dynamic forecast status check failed: "
                f"{exc}"
            ),
        )


@router.get("/manifest")
def dynamic_forecast_manifest():

    try:

        return get_dynamic_forecast_manifest()

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load dynamic forecast manifest: "
                f"{exc}"
            ),
        )


# ============================================================
# FORECAST SUMMARY
# ============================================================

@router.get("/summary")
def dynamic_forecast_summary():

    try:

        records = get_dynamic_forecast_summary()

        return {
            "count": len(records),
            "records": records,
        }

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load dynamic forecast summary: "
                f"{exc}"
            ),
        )


# ============================================================
# FORECAST DATA
# ============================================================

@router.get("/forecast")
def dynamic_forecast_by_year(
    year: int = Query(
        ...,
        description=(
            "One of the currently available "
            "dynamic five-year forecast years."
        ),
    ),
):

    try:

        records = get_dynamic_forecast_by_year(
            year=year,
        )

        return {
            "year": year,
            "count": len(records),
            "records": records,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load dynamic forecast: "
                f"{exc}"
            ),
        )


@router.get("/forecast/grid/{grid_id}")
def dynamic_grid_forecast(
    grid_id: str,
    year: int = Query(
        ...,
        description="Dynamic forecast year.",
    ),
):

    try:

        result = get_dynamic_grid_forecast(
            grid_id=grid_id,
            year=year,
        )

        if result is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"No dynamic forecast found for "
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

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load grid forecast: "
                f"{exc}"
            ),
        )


# ============================================================
# TRAJECTORY DATA
# ============================================================

@router.get("/trajectory")
def dynamic_trajectory(
    area_name: Optional[str] = Query(
        default=None,
        description=(
            "Optional area-name filter."
        ),
    ),
):

    try:

        records = get_dynamic_trajectory(
            area_name=area_name,
        )

        return {
            "count": len(records),
            "area_name": area_name,
            "records": records,
        }

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load dynamic trajectory data: "
                f"{exc}"
            ),
        )


@router.get("/trajectory/grid/{grid_id}")
def dynamic_grid_trajectory(
    grid_id: str,
):

    try:

        result = get_dynamic_grid_trajectory(
            grid_id=grid_id,
        )

        if result is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"No dynamic trajectory found for "
                    f"grid '{grid_id}'."
                ),
            )

        return result

    except HTTPException:

        raise

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load grid trajectory: "
                f"{exc}"
            ),
        )


# ============================================================
# MAP / GEOJSON
# ============================================================

@router.get("/map")
def dynamic_forecast_map(
    year: int = Query(
        ...,
        description="Dynamic forecast map year.",
    ),
):

    try:

        return get_dynamic_forecast_map(
            year=year,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load dynamic forecast map: "
                f"{exc}"
            ),
        )


@router.get("/map/combined")
def dynamic_combined_forecast_map():

    try:

        return get_dynamic_combined_forecast_map()

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load combined dynamic forecast map: "
                f"{exc}"
            ),
        )


@router.get("/map/trajectory")
def dynamic_trajectory_map():

    try:

        return get_dynamic_trajectory_map()

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load dynamic trajectory map: "
                f"{exc}"
            ),
        )

# ============================================================
# CSV UPLOAD
# ============================================================

@router.post("/upload")
def dynamic_forecast_upload(
    file: UploadFile = File(...),
):

    try:

        result = stage_dynamic_history_upload(
            source_file=file.file,
            original_filename=file.filename,
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Dynamic forecast upload failed: "
                f"{exc}"
            ),
        )

    finally:

        file.file.close()

@router.post("/validate-upload")
def dynamic_forecast_validate_upload():
    try:
        return validate_staged_dynamic_history()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Dynamic forecast validation failed: "
                f"{exc}"
            ),
        )


@router.post("/run")
def dynamic_forecast_run():

    try:

        return run_dynamic_forecast_pipeline()

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except RuntimeError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Dynamic forecast pipeline failed: "
                f"{exc}"
            ),
        )

@router.get("/run-status")
def dynamic_forecast_run_status():

    try:

        return get_dynamic_pipeline_state()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to read dynamic pipeline state: "
                f"{exc}"
            ),
        )