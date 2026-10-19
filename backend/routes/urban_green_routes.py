
"""
Urban Green Research API Routes
Project: J26-DS-338

Read-only access to validated Urban Green Research
planning, suitability, conservation and spatial outputs.

These endpoints do not retrain models or recalculate
research scores.
"""

from typing import Literal
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend.services.urban_green_data_service import (
    get_manifest,
    get_metadata,
    get_grid_records,
    get_grid_record,
    get_gn_summaries,
    get_gn_summary,
    get_hotspot_groups,
    get_candidate_hotspot_groups,
    get_map_path,
    validate_dashboard_exports,
)


# ============================================================
# HISTORICAL GREEN-COVER RESEARCH SERVICE
# ============================================================

from backend.services.urban_green_historical_service import (
    validate_historical_reports,
    get_annual_trajectory,
    get_green_key_summary,
    get_annual_class_areas,
    get_annual_lulc_qa,
    get_period_status,
    get_transition_reporting_decision,
    get_transition_comparison,
    get_transition_analysis,
    get_transition_breakdown,
    get_transition_confidence,
)

# ============================================================
# HISTORICAL GIS MAP SERVICE
# ============================================================

from backend.services.urban_green_historical_map_service import (
    get_historical_map_catalog,
    get_historical_map_metadata,
    get_historical_map_image_path,
    get_historical_map_health,
)

# ============================================================
# SECTION 02 - URBANIZATION PRESSURE SERVICE
# ============================================================

from backend.services.urban_green_pressure_service import (
    get_pressure_overview,
    get_pressure_grid,
    get_pressure_grids,
)

# ============================================================
# SECTION 02 - PRESSURE GIS MAP SERVICE
# ============================================================

from backend.services.urban_green_pressure_map_service import (
    get_pressure_map_metadata,
    get_pressure_map_file_path,
)

# ============================================================
# SECTION 02 - GLOBAL EXPLAINABLE AI SERVICE
# ============================================================

from backend.services.urban_green_xai_service import (
    get_pressure_xai_overview,
    get_pressure_xai_features,
    get_pressure_xai_domains,
)


# ============================================================
# URBAN GREEN ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/urban-green",
    tags=["Urban Green Research"],
)


# ============================================================
# 1. API HEALTH
# ============================================================

@router.get("/health")
def urban_green_health():
    return {
        "status": "ok",
        "component": "Urban Green Research",
        "project": "J26-DS-338",
        "mode": "read-only",
        "message": "Urban Green API router is working.",
    }


# ============================================================
# 2. DATA INTEGRITY
# ============================================================

@router.get("/data-health")
def urban_green_data_health():
    """
    Check all seven manifest-listed research files
    using byte counts and SHA-256 hashes.
    """
    try:
        return validate_dashboard_exports()

    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Urban Green data verification failed: {exc}",
        ) from exc


# ============================================================
# 3. MANIFEST
# ============================================================

@router.get("/manifest")
def urban_green_manifest():
    return get_manifest()


# ============================================================
# 4. RESEARCH METADATA
# ============================================================

@router.get("/metadata")
def urban_green_metadata():
    return get_metadata()


# ============================================================
# 5. GRID RECORDS WITH PAGINATION AND FILTERING
# ============================================================

@router.get("/planning/grids")
def urban_green_grids(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    gn_id: str | None = None,
    review_stream: str | None = None,
):
    """
    Return existing grid records.

    Filtering and pagination only.
    No research score recalculation.
    """
    records = get_grid_records()

    if gn_id is not None:
        gn_id = gn_id.strip().upper()
        records = [
            record for record in records
            if record["gn_id"] == gn_id
        ]

    if review_stream is not None:
        review_stream = review_stream.strip().upper()
        records = [
            record for record in records
            if record["review_stream"] == review_stream
        ]

    total = len(records)

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "data": list(records[offset:offset + limit]),
    }


# ============================================================
# 6. SINGLE GRID DETAILS
# ============================================================

@router.get("/planning/grids/{grid_id}")
def urban_green_grid_details(grid_id: str):
    """
    Return recorded research indicators and planning
    review guidance for one selected 100 m grid.
    """
    grid_id = grid_id.strip().upper()

    record = get_grid_record(grid_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Urban Green grid {grid_id} not found",
        )

    return record


# ============================================================
# 7. GRID MAP
# ============================================================

@router.get("/planning/map")
def urban_green_grid_map():
    """
    Serve the original validated CRS84 grid GeoJSON.

    Contains 9,240 grid features.
    """
    return FileResponse(
        path=get_map_path("grid"),
        media_type="application/geo+json",
    )


# ============================================================
# 8. GN DIVISION SUMMARIES
# ============================================================

@router.get("/planning/gn-divisions")
def urban_green_gn_divisions():
    """
    Return existing area-weighted summaries
    for 57 GN divisions.
    """
    records = get_gn_summaries()

    return {
        "total": len(records),
        "data": list(records),
    }


# ============================================================
# 9. SINGLE GN DIVISION DETAILS
# ============================================================

@router.get("/planning/gn-divisions/{gn_id}")
def urban_green_gn_details(gn_id: str):
    gn_id = gn_id.strip().upper()

    record = get_gn_summary(gn_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Urban Green GN division {gn_id} not found",
        )

    return record


# ============================================================
# 10. GN DIVISION MAP
# ============================================================

@router.get("/planning/gn-map")
def urban_green_gn_map():
    return FileResponse(
        path=get_map_path("gn"),
        media_type="application/geo+json",
    )


# ============================================================
# 11. EXPLORATORY SPATIAL GROUPS
# ============================================================

@router.get("/planning/hotspots")
def urban_green_hotspots(
    candidates_only: bool = True,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    index_type: str | None = None,
):
    """
    Return exploratory spatial groups.

    candidates_only=True:
        Only the 58 candidate groups.

    candidates_only=False:
        All 673 screened groups.

    These are NOT statistically significant hotspots.
    """
    if candidates_only:
        records = get_candidate_hotspot_groups()
    else:
        records = list(get_hotspot_groups())

    if index_type is not None:
        index_type = index_type.strip().upper()

        records = [
            record for record in records
            if record["index_type"] == index_type
        ]

    total = len(records)

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "candidates_only": candidates_only,
        "statistical_significance_tested": False,
        "data": list(records[offset:offset + limit]),
    }


# ============================================================
# 12. CANDIDATE SPATIAL GROUP MAP
# ============================================================

@router.get("/planning/hotspot-map")
def urban_green_hotspot_map():
    """
    Serve 58 candidate spatial-group polygons.

    Classification is exploratory and sensitive
    to screening thresholds.
    """
    return FileResponse(
        path=get_map_path("hotspot"),
        media_type="application/geo+json",
    )


# ============================================================
# SECTION 01 - HISTORICAL GREEN-COVER RESEARCH API
# ============================================================

# All endpoints read finalized research reports.
# No model retraining or research recalculation.


# ============================================================
# 1. HISTORICAL DATA HEALTH
# ============================================================

@router.get(
    "/historical/data-health",
    tags=["Urban Green - Historical Analysis"],
)
def urban_green_historical_data_health():
    """
    Verify the ten finalized historical research reports.
    """
    try:
        return validate_historical_reports()

    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Historical data validation failed: {exc}",
        ) from exc


# ============================================================
# 2. ANNUAL GREEN-COVER TRAJECTORY
# ============================================================

@router.get(
    "/historical/trajectory",
    tags=["Urban Green - Historical Analysis"],
)
def urban_green_historical_trajectory(
    core_only: bool = False,
):
    """
    Return 2010-2026 annual green-cover statistics.

    core_only=True restricts results to the validated
    primary trend period, 2011-2024.

    The 2026 data is provisional and not part
    of the primary historical trend.
    """
    records = get_annual_trajectory(
        core_only=core_only
    )

    return {
        "total": len(records),
        "core_only": core_only,
        "primary_core_period": "2011-2024",
        "data": records,
    }


# ============================================================
# 3. HISTORICAL GREEN-COVER SUMMARY
# ============================================================

@router.get(
    "/historical/summary",
    tags=["Urban Green - Historical Analysis"],
)
def urban_green_historical_summary():
    """
    Return the saved green-cover and built-up
    historical trend statistics.
    """
    return get_green_key_summary()


# ============================================================
# 4. ANNUAL LULC CLASS AREAS
# ============================================================

@router.get(
    "/historical/class-areas",
    tags=["Urban Green - Historical Analysis"],
)
def urban_green_historical_class_areas(
    year: int | None = Query(
        default=None,
        ge=2010,
        le=2026,
    ),
    product_type: str | None = None,
):
    """
    Return recorded LULC class-area statistics.

    Optional filters:
    - year
    - product_type

    Product types come from the original research CSV.
    """
    records = get_annual_class_areas()

    if year is not None:
        records = [
            record
            for record in records
            if record["year"] == year
        ]

    if product_type is not None:
        selected_type = product_type.strip().upper()

        records = [
            record
            for record in records
            if str(
                record["product_type"]
            ).upper() == selected_type
        ]

    return {
        "total": len(records),
        "year": year,
        "product_type": product_type,
        "data": records,
    }


# ============================================================
# 5. ANNUAL PERIOD STATUS
# ============================================================

@router.get(
    "/historical/period-status",
    tags=["Urban Green - Historical Analysis"],
)
def urban_green_historical_period_status():
    """
    Explain which years are core, context,
    excluded or provisional.
    """
    records = get_period_status()

    return {
        "total": len(records),
        "data": records,
    }


# ============================================================
# 6. ANNUAL LULC QUALITY ASSURANCE
# ============================================================

@router.get(
    "/historical/annual-qa",
    tags=["Urban Green - Historical Analysis"],
)
def urban_green_historical_annual_qa():
    """
    Return saved Step 09 annual LULC QA results.
    """
    return get_annual_lulc_qa()


# ============================================================
# 7. TRANSITION REPORTING METHODOLOGY
# ============================================================

@router.get(
    "/historical/transitions/method",
    tags=["Urban Green - Historical Transitions"],
)
def urban_green_transition_method():
    """
    Return the locked historical transition method.

    ROBUST_2_OF_3 is primary.
    STRICT_3_OF_3 is the conservative subset.
    """
    return get_transition_reporting_decision()


# ============================================================
# 8. ROBUST AND STRICT TRANSITION COMPARISON
# ============================================================

@router.get(
    "/historical/transitions/comparison",
    tags=["Urban Green - Historical Transitions"],
)
def urban_green_transition_comparison():
    """
    Compare two saved transition estimates.

    The robust/strict difference is NOT a
    statistical confidence interval.
    """
    records = get_transition_comparison()
    decision = get_transition_reporting_decision()

    return {
        "primary_method": (
            decision[
                "historical_transition_primary_rule"
            ]
        ),
        "conservative_method": (
            decision[
                "historical_transition_sensitivity_rule"
            ]
        ),
        "statistical_confidence_interval": False,
        "total": len(records),
        "data": records,
    }


# ============================================================
# 9. TRANSITION ANALYSIS SUMMARY
# ============================================================

@router.get(
    "/historical/transitions/analysis",
    tags=["Urban Green - Historical Transitions"],
)
def urban_green_transition_analysis():
    """
    Return the saved consensus-window analysis,
    including common valid area and transition QA.
    """
    return get_transition_analysis()


# ============================================================
# 10. TRANSITION CLASS BREAKDOWN
# ============================================================

@router.get(
    "/historical/transitions/breakdown",
    tags=["Urban Green - Historical Transitions"],
)
def urban_green_transition_breakdown():
    records = get_transition_breakdown()

    return {
        "total": len(records),
        "data": records,
    }


# ============================================================
# 11. TRANSITION CONFIRMATION COMPARISON
# ============================================================

@router.get(
    "/historical/transitions/confidence",
    tags=["Urban Green - Historical Transitions"],
)
def urban_green_transition_confidence():
    """
    Return the saved robust-versus-strict
    confirmation comparison.

    Not a calibrated confidence interval.
    """
    records = get_transition_confidence()

    return {
        "total": len(records),
        "statistical_confidence_interval": False,
        "data": records,
    }


# ============================================================
# SECTION 01 - HISTORICAL GIS MAP API
# ============================================================
# Serves only approved historical PNG overlays.
# No raster processing, model training or file writes.


# ============================================================
# 1. HISTORICAL GIS MAP HEALTH
# ============================================================

@router.get(
    "/historical/maps/health",
    tags=["Urban Green - Historical GIS Maps"],
)
def urban_green_historical_map_health():
    """
    Validate the historical GIS map catalog.
    """
    try:
        return get_historical_map_health()

    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Historical GIS map catalog is unavailable.",
        ) from exc


# ============================================================
# 2. HISTORICAL GIS MAP CATALOG
# ============================================================

@router.get(
    "/historical/maps",
    tags=["Urban Green - Historical GIS Maps"],
)
def urban_green_historical_map_catalog():
    """
    List all 38 historical maps.

    Includes annual LULC, historical transitions,
    confidence categories, bounds and image URLs.
    """
    try:
        return get_historical_map_catalog()

    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Historical GIS map catalog is unavailable.",
        ) from exc


# ============================================================
# 3. INDIVIDUAL MAP METADATA
# ============================================================

@router.get(
    "/historical/maps/{map_id}",
    tags=["Urban Green - Historical GIS Maps"],
)
def urban_green_historical_map_metadata(
    map_id: str,
):
    """
    Return metadata for one approved historical map.

    Includes:
    - source raster
    - geographic bounds
    - classification legend
    - year or transition method
    - provisional/core period flags
    - image URL
    """
    try:
        return get_historical_map_metadata(map_id)

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Historical map not found.",
        ) from exc

    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Historical GIS map catalog is unavailable.",
        ) from exc


# ============================================================
# 4. INDIVIDUAL MAP PNG IMAGE
# ============================================================

@router.get(
    "/historical/maps/{map_id}/image",
    tags=["Urban Green - Historical GIS Maps"],
    response_class=FileResponse,
)
def urban_green_historical_map_image(
    map_id: str,
):
    """
    Return the validated georeferenced PNG overlay.

    The caller selects a registered map ID, not
    a filesystem path.

    Geographic bounds and legend are available
    from the corresponding metadata endpoint.
    """
    try:
        image_path = get_historical_map_image_path(
            map_id
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Historical map not found.",
        ) from exc

    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Historical map image is unavailable.",
        ) from exc

    return FileResponse(
        path=image_path,
        media_type="image/png",
    )

# ============================================================
# SECTION 02 - URBANIZATION PRESSURE API
# ============================================================
#
# Final production model: XGBoost
# Origin year: 2025
# Pressure period: 2026-2028
# Locked threshold: 0.715
#
# Scores are NOT calibrated probabilities.
# SHAP values explain RAW XGBoost model margins.
#
# Read-only access to validated research results.
# No retraining, model inference or recalculation.
# ============================================================


# 1. URBANIZATION PRESSURE OVERVIEW

@router.get(
    "/pressure/overview",
    tags=["Urban Green - Urbanization Pressure"],
)
def urban_green_pressure_overview():
    """
    Return the final Urbanization Pressure summary.

    Includes inference coverage, flagged grids,
    model metadata, category counts and research QA.
    """
    try:
        return get_pressure_overview()

    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Urbanization Pressure data is unavailable.",
        ) from exc


# 2. PAGINATED PRESSURE GRID RECORDS

@router.get(
    "/pressure/grids",
    tags=["Urban Green - Urbanization Pressure"],
)
def urban_green_pressure_grids(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    inference_ready: bool | None = Query(default=None),
):
    """
    Return paginated pressure grid records.

    Optional inference_ready filter:
    - True: only scored eligible cells
    - False: only outside-domain cells
    - None: all permanent grid cells

    Outside-domain scores are returned as null.
    """
    try:
        return get_pressure_grids(
            offset=offset,
            limit=limit,
            inference_ready=inference_ready,
        )

    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Urbanization Pressure data is unavailable.",
        ) from exc


# 3. INDIVIDUAL PRESSURE GRID AND XAI DETAILS

@router.get(
    "/pressure/grids/{grid_id}",
    tags=["Urban Green - Urbanization Pressure"],
)
def urban_green_pressure_grid_detail(
    grid_id: str,
):
    """
    Return one permanent grid cell's pressure results.

    For eligible grids:
    - model score and percentile
    - threshold status
    - relative pressure category
    - dominant SHAP explanation domains
    - top three SHAP features

    For outside-domain grids:
    - eligibility information
    - score/category/explanation remain null
    """
    try:
        return get_pressure_grid(grid_id)

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Urbanization Pressure grid not found.",
        ) from exc

    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Urbanization Pressure data is unavailable.",
        ) from exc

# ============================================================
# SECTION 02 - URBANIZATION PRESSURE GIS MAP API
# ============================================================
#
# Source: Final Step 14D pressure GIS export
# Origin: 2025
# Prediction period: 2026-2028
# CRS: EPSG:4326
# Permanent grids: 9240
# Scored grids: 4620
# Outside-domain grids: 4620
#
# READ ONLY - NO MODEL INFERENCE OR RECALCULATION
# ============================================================


# 1. PRESSURE GIS MAP METADATA

@router.get(
    "/pressure/map/metadata",
    tags=["Urban Green - Urbanization Pressure GIS"],
)
def urban_green_pressure_map_metadata():
    """
    Return frontend-ready Pressure GIS metadata.

    Includes:
    - Map URL
    - Geographic bounds
    - CRS
    - Six category colors and counts
    - Scored / outside-domain coverage
    - Locked model threshold
    - Prediction-period interpretation
    """
    try:
        return get_pressure_map_metadata()

    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Urbanization Pressure GIS metadata is unavailable.",
        ) from exc


# 2. PRESSURE GIS GEOJSON MAP

@router.get(
    "/pressure/map",
    tags=["Urban Green - Urbanization Pressure GIS"],
    response_class=FileResponse,
    responses={
        200: {
            "description": (
                "Validated Urbanization Pressure GeoJSON"
            ),
            "content": {
                "application/geo+json": {}
            },
        }
    },
)
def urban_green_pressure_map():
    """
    Return the validated 9240-feature GeoJSON map.

    All original grid IDs and pressure attributes
    are preserved.

    Outside-domain cells retain null scores.

    The map can be joined to grid-detail APIs
    using the grid_id property.
    """
    try:
        map_path = get_pressure_map_file_path()

    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Urbanization Pressure GIS map is unavailable.",
        ) from exc

    return FileResponse(
        path=map_path,
        media_type="application/geo+json",
    )

# ============================================================
# SECTION 02 - GLOBAL EXPLAINABLE AI API
# ============================================================
#
# Research sources:
#   Step 15A - Production SHAP
#   Step 15B - Explanation domains
#
# Groups:
#   all_scored         : 4620 eligible grids
#   threshold_flagged  : 255 flagged grids
#
# SHAP output space: RAW_XGBOOST_MARGIN
# Importance basis: Normalized mean absolute SHAP
#
# NOT probability contributions.
# NOT real-world causal explanations.
#
# READ ONLY - NO SHAP RECALCULATION OR MODEL TRAINING
# ============================================================


XAIGroup = Literal[
    "all_scored",
    "threshold_flagged",
]


# 1. GLOBAL XAI OVERVIEW

@router.get(
    "/pressure/xai/overview",
    tags=["Urban Green - Explainable AI"],
)
def urban_green_pressure_xai_overview():
    """
    Return the final Global XAI research overview.

    Includes:
    - Available explanation groups
    - Number of explained grids
    - Feature and domain counts
    - Leading feature and domain
    - Final research QA statuses
    - Scientific interpretation
    """
    try:
        return get_pressure_xai_overview()

    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Urbanization Pressure XAI data is unavailable.",
        ) from exc


# 2. RANKED GLOBAL FEATURE IMPORTANCE

@router.get(
    "/pressure/xai/features",
    tags=["Urban Green - Explainable AI"],
)
def urban_green_pressure_xai_features(
    group: XAIGroup = Query(default="all_scored"),
    limit: int = Query(default=10, ge=1, le=55),
):
    """
    Return ranked SHAP feature importance.

    group:
      all_scored: 4620 inference-eligible grids
      threshold_flagged: 255 flagged grids

    limit:
      Number of ranked features to return (1-55).

    Importance percentages are normalized within
    each analysis group.

    They are not probability changes.
    """
    try:
        return get_pressure_xai_features(
            group=group,
            limit=limit,
        )

    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Urbanization Pressure feature importance is unavailable.",
        ) from exc


# 3. GLOBAL EXPLANATION-DOMAIN IMPORTANCE

@router.get(
    "/pressure/xai/domains",
    tags=["Urban Green - Explainable AI"],
)
def urban_green_pressure_xai_domains(
    group: XAIGroup = Query(default="all_scored"),
):
    """
    Return importance of five explanation domains.

    Available groups:
    - all_scored
    - threshold_flagged

    Includes:
    - Domain rank
    - Feature count
    - Absolute SHAP importance
    - Mean signed SHAP raw margin
    - Importance percentage

    Mean signed SHAP values indicate model-output
    direction, not real-world causality.
    """
    try:
        return get_pressure_xai_domains(
            group=group,
        )

    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Urbanization Pressure domain explanations are unavailable.",
        ) from exc


# ============================================================
# FINAL DEVELOPMENT SUITABILITY + CONSERVATION APIs
# ============================================================

from fastapi import Query as _DecisionQuery
from fastapi import HTTPException as _DecisionHTTPException
from fastapi.responses import FileResponse as _DecisionFileResponse

from backend.services.urban_green_decision_service import (
    get_decision_overview,
    get_decision_grid,
    get_decision_grids,
    get_decision_map_metadata,
    get_decision_map_file,
)


def _decision_api(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)

    except KeyError as exc:
        raise _DecisionHTTPException(
            status_code=404,
            detail="Grid not found.",
        ) from exc

    except (OSError, ValueError, TypeError) as exc:
        raise _DecisionHTTPException(
            status_code=503,
            detail="Verified Urban Green research data unavailable.",
        ) from exc


# DEVELOPMENT SUITABILITY

@router.get("/suitability/overview",
            tags=["Urban Green - Development Suitability"])
def urban_green_final_suitability_overview():
    return _decision_api(get_decision_overview, "suitability")


@router.get("/suitability/grids",
            tags=["Urban Green - Development Suitability"])
def urban_green_final_suitability_grids(
    offset: int = _DecisionQuery(0, ge=0),
    limit: int = _DecisionQuery(100, ge=1, le=200),
):
    return _decision_api(
        get_decision_grids, "suitability",
        offset=offset, limit=limit,
    )


@router.get("/suitability/grids/{grid_id}",
            tags=["Urban Green - Development Suitability"])
def urban_green_final_suitability_grid(grid_id: str):
    return _decision_api(
        get_decision_grid, "suitability", grid_id,
    )


@router.get("/suitability/map/metadata",
            tags=["Urban Green - Development Suitability"])
def urban_green_final_suitability_map_metadata():
    return _decision_api(
        get_decision_map_metadata, "suitability",
    )


@router.get("/suitability/map",
            tags=["Urban Green - Development Suitability"])
def urban_green_final_suitability_map():
    path = _decision_api(
        get_decision_map_file, "suitability",
    )
    return _DecisionFileResponse(
        path, media_type="application/geo+json"
    )


# CONSERVATION INTELLIGENCE

@router.get("/conservation/overview",
            tags=["Urban Green - Conservation Intelligence"])
def urban_green_final_conservation_overview():
    return _decision_api(get_decision_overview, "conservation")


@router.get("/conservation/grids",
            tags=["Urban Green - Conservation Intelligence"])
def urban_green_final_conservation_grids(
    offset: int = _DecisionQuery(0, ge=0),
    limit: int = _DecisionQuery(100, ge=1, le=200),
):
    return _decision_api(
        get_decision_grids, "conservation",
        offset=offset, limit=limit,
    )


@router.get("/conservation/grids/{grid_id}",
            tags=["Urban Green - Conservation Intelligence"])
def urban_green_final_conservation_grid(grid_id: str):
    return _decision_api(
        get_decision_grid, "conservation", grid_id,
    )


@router.get("/conservation/map/metadata",
            tags=["Urban Green - Conservation Intelligence"])
def urban_green_final_conservation_map_metadata():
    return _decision_api(
        get_decision_map_metadata, "conservation",
    )


@router.get("/conservation/map",
            tags=["Urban Green - Conservation Intelligence"])
def urban_green_final_conservation_map():
    path = _decision_api(
        get_decision_map_file, "conservation",
    )
    return _DecisionFileResponse(
        path, media_type="application/geo+json"
    )
