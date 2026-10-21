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
