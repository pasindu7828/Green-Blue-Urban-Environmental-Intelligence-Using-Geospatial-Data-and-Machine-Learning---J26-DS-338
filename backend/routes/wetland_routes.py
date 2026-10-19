from fastapi import APIRouter, HTTPException
from fastapi import Response

from backend.services.wetland_data_service import (
    get_part1_summary,
    get_part1_map,
    get_part1_validation,
    get_part2_primary_candidates,
    get_part2_all_candidates,
    get_part2_gn_summary,
    get_part3_gn_summary,
    get_part3_wetland_map,
    get_part3_scenario_map,
    get_part4_gn_summary,
    get_part4_raster_metadata,
    get_part4_raster_png,
    get_part1_historical_years,
    get_part1_historical_map_by_year,
    get_part1_wetland_observations,
    get_part1_comparison_evidence,
    get_part3_focused_scenario_map,
    get_part3_focused_gn_summary,
    get_part3_individual_future_comparison,

)

# ============================================================
# WETLAND RESEARCH API ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/wetlands",
    tags=["Wetland Research - Part 1"],
)


# ============================================================
# PART 1 - HISTORICAL WETLAND CHANGE
# ============================================================

@router.get("/part1/summary")
def part1_summary():
    """
    Return historical change summaries
    for the 25 monitored wetlands.
    """
    return {
        "component": "Part 1 - Historical Wetland Change",
        "total_wetlands": 25,
        "data": get_part1_summary(),
    }


@router.get("/part1/map")
def part1_map():
    """
    Return existing WGS84 GeoJSON containing
    the latest boundaries of 25 wetlands.
    """
    return get_part1_map()


@router.get("/part1/validation")
def part1_validation():
    """
    Return existing Part 1 backend
    validation results.
    """
    return {
        "component": "Part 1 - Historical Wetland Change",
        "checks": get_part1_validation(),
    }


@router.get("/part1/wetlands/{wetland_id}")
def part1_wetland_details(wetland_id: str):
    """
    Return the historical analysis summary
    for one selected wetland.
    """
    wetland_id = wetland_id.strip().upper()

    for record in get_part1_summary():
        if record["wetland_id"] == wetland_id:
            return record

    raise HTTPException(
        status_code=404,
        detail=f"Wetland {wetland_id} not found",
    )


# ============================================================
# PART 1 - AVAILABLE HISTORICAL OBSERVATION YEARS
# ============================================================

@router.get(
    "/part1/historical/years",
    tags=["Wetland Research - Part 1"],
)
def part1_historical_years():
    """
    Return available historical observation years,
    observation counts, and distinct wetland counts.
    """
    return get_part1_historical_years()

# ============================================================
# PART 1 - HISTORICAL WETLAND MAP BY YEAR
# ============================================================

@router.get(
    "/part1/historical/map/{year}",
    tags=["Wetland Research - Part 1"],
)
def part1_historical_map_by_year(year: int):
    """
    Return historical wetland polygons for the selected year.
    Each wetland uses its latest actual observation
    within that year.
    """
    return get_part1_historical_map_by_year(year)


# ============================================================
# PART 1 - INDIVIDUAL WETLAND HISTORICAL OBSERVATIONS
# ============================================================

@router.get(
    "/part1/historical/wetlands/{wetland_id}/observations",
    tags=["Wetland Research - Part 1"],
)
def part1_individual_wetland_observations(wetland_id: str):
    """
    Return actual historical observation polygons,
    dates and measurements for one monitored wetland.
    """
    result = get_part1_wetland_observations(wetland_id)

    if not result["features"]:
        raise HTTPException(
            status_code=404,
            detail=f"No historical observations found for {wetland_id}",
        )

    return result


# ============================================================
# PART 1 - HISTORICAL COMPARISON EVIDENCE
# ============================================================

@router.get(
    "/part1/historical/wetlands/{wetland_id}/comparison-evidence",
    tags=["Wetland Research - Part 1"],
)
def part1_historical_comparison_evidence(
    wetland_id: str,
    before_date: str,
    after_date: str,
):
    """
    Retrieve existing quality-assessed evidence for an exact
    historical wetland observation pair.

    Query parameters:
      before_date: YYYY-MM-DD
      after_date: YYYY-MM-DD
    """
    return get_part1_comparison_evidence(
        wetland_id=wetland_id,
        before_date=before_date,
        after_date=after_date,
    )

# ============================================================
# PART 2 - CANDIDATE WETLAND DETECTION
# ============================================================

@router.get(
    "/part2/primary-candidates",
    tags=["Wetland Research - Part 2"],
)
def part2_primary_candidates():
    """
    Return the 48 primary candidate wetland polygons.

    These are candidate locations, not independently
    confirmed wetlands.
    """
    return get_part2_primary_candidates()


@router.get(
    "/part2/all-candidates",
    tags=["Wetland Research - Part 2"],
)
def part2_all_candidates():
    """
    Return the broader candidate dataset.

    Includes primary and other candidate polygons
    retained in the research outputs.
    """
    return get_part2_all_candidates()


@router.get(
    "/part2/gn-summary",
    tags=["Wetland Research - Part 2"],
)
def part2_gn_summary():
    """
    Return candidate wetland statistics
    for Kaduwela's 56 GN divisions.
    """
    return get_part2_gn_summary()


# ============================================================
# PART 3 - FUTURE WETLAND PROJECTION API
# ============================================================

@router.get(
    "/part3/gn-summary",
    tags=["Wetland Research - Part 3"],
)
def part3_gn_summary():
    """
    Return 56 GN divisions with historical wetland
    statistics and exploratory 2030 scenario results.
    """
    return get_part3_gn_summary()


@router.get(
    "/part3/wetland-map/{year}",
    tags=["Wetland Research - Part 3"],
)
def part3_wetland_map(year: int):
    """
    Return wetland polygons for:
    2025 - Estimated wetland distribution
    2030 - Exploratory CA-Markov scenario
    """

    if year not in (2025, 2030):
        raise HTTPException(
            status_code=400,
            detail="Only 2025 and 2030 wetland maps are available",
        )

    return get_part3_wetland_map(year)


@router.get(
    "/part3/scenario/{change_type}",
    tags=["Wetland Research - Part 3"],
)
def part3_scenario_map(change_type: str):
    """
    Return exploratory wetland scenario changes:
    gain, loss or persistence.
    """

    if change_type not in ("gain", "loss", "persistence"):
        raise HTTPException(
            status_code=400,
            detail="Change type must be gain, loss, or persistence",
        )

    return get_part3_scenario_map(change_type)


# ============================================================
# PART 3 — FOCUSED GN SCENARIO SUMMARY
# ============================================================

@router.get("/part3/focused-gn-summary")
def part3_focused_gn_summary():
    """
    Return GN-wise area and spatial difference statistics
    calculated from the focused 2025 and 2030 model patches.

    Does not modify original research datasets.
    """
    return get_part3_focused_gn_summary()


# ============================================================
# PART 3 — INDIVIDUAL WETLAND FUTURE COMPARISON
# ============================================================

@router.get("/part3/individual-future-comparison/{wetland_id}")
def part3_individual_future_comparison(wetland_id: str):
    """
    Return a historical wetland boundary and nearby
    exploratory 2030 scenario patches.

    No original research data is modified.
    """
    try:
        return get_part3_individual_future_comparison(wetland_id)
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

# ============================================================
# PART 4 - SEASONAL WATERLOGGING SUSCEPTIBILITY API
# ============================================================

@router.get(
    "/part4/gn-summary",
    tags=["Wetland Research - Part 4"],
)
def part4_gn_summary():
    """
    Return waterlogging susceptibility statistics and
    geographic boundaries for all 56 GN divisions.
    """
    return get_part4_gn_summary()

# ============================================================
# PART 3 - FOCUSED 2030 WETLAND SCENARIO
# ============================================================

@router.get(
    "/part3/focused-scenario",
    tags=["Wetland Research - Part 3"],
)
def part3_focused_scenario():
    """
    Return a focused view of the existing exploratory
    2030 wetland scenario.

    Uses the 25 historical wetlands, 48 primary
    screened candidates, provisional 50 m buffer,
    and existing five-pixel display rule.

    Does not modify or validate the prediction model.
    """
    return get_part3_focused_scenario_map()

# ============================================================
# PART 4 - RASTER MAP API
# ============================================================

@router.get(
    "/part4/raster/{map_type}/metadata",
    tags=["Wetland Research - Part 4"],
)
def part4_raster_metadata(map_type: str):
    """
    Return geographic bounds, raster dimensions,
    classification legend and NoData information.
    """

    if map_type not in ("main", "terrain", "span"):
        raise HTTPException(
            status_code=400,
            detail="Map type must be main, terrain, or span",
        )

    return get_part4_raster_metadata(map_type)


@router.get(
    "/part4/raster/{map_type}/png",
    tags=["Wetland Research - Part 4"],
    response_class=Response,
)
def part4_raster_png(map_type: str):
    """
    Return a transparent, geographically reprojected
    PNG image of the selected susceptibility raster.
    """

    if map_type not in ("main", "terrain", "span"):
        raise HTTPException(
            status_code=400,
            detail="Map type must be main, terrain, or span",
        )

    png_bytes = get_part4_raster_png(map_type)

    return Response(
        content=png_bytes,
        media_type="image/png",
    )
