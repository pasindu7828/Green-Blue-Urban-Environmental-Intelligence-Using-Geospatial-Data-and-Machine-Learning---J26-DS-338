
"""
J26-DS-338 - River Corridor Research
STEP 18: Research Dashboard API

Read-only endpoints serving completed research outputs.

IMPORTANT:
- No model retraining inside API requests.
- No independently validated riverbank erosion claims.
- No modifications to other research components.
"""

import csv
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse


router = APIRouter(
    prefix="/api/river-corridor",
    tags=["River Corridor Research"]
)

# backend/routes/this_file.py -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = PROJECT_ROOT / "River_Corridor_Research"

DATA_DIR = RESEARCH_ROOT / "data/processed/morphology"
DASHBOARD_DIR = RESEARCH_ROOT / "outputs/dashboard"
REPORT_DIR = RESEARCH_ROOT / "outputs/reports"

FILES = {
    "summary": (
        DASHBOARD_DIR / "12_projection_summary.json"
    ),
    "projections": (
        DASHBOARD_DIR / "12_projection.csv"
    ),
    "geojson": (
        DASHBOARD_DIR / "12_projection.geojson"
    ),
    "metrics": (
        REPORT_DIR / "11_ml_benchmark_metrics.csv"
    ),
    "annual_qa": (
        REPORT_DIR / "10_annual_pilot_qa.csv"
    ),
    "historical": (
        DATA_DIR / "10_annual_pilot_widths.csv"
    ),
    "projection_qa": (
        REPORT_DIR / "12_projection_qa.csv"
    ),
}

INTEGER_FIELDS = {
    "year",
    "base_year",
    "target_year",
    "samples",
    "records",
    "total_transects",
    "pilot_screened",
    "low_resolution",
    "centre_not_water",
    "other_qa_flags",
    "changed_cases_ge_30m",
    "current_year",
    "current_screened_transects",
    "evaluated_pairs",
}

FLOAT_FIELDS = {
    "chainage_m",
    "pilot_width_m",
    "current_width_m",
    "next_width_m",
    "persistence_2026_m",
    "rf_candidate_2026_m",
    "rf_candidate_change_m",
    "pilot_change_m",
    "mae_m",
    "rmse_m",
    "r2_width",
    "mean_error_m",
    "mae_on_changed_cases_m",
    "screened_pct",
    "evaluation_coverage_pct",
}

BOOLEAN_FIELDS = {
    "screened",
    "validated",
    "ml_training_approved",
    "production_approved",
    "physical_erosion_prediction",
}


def require_file(file_key: str) -> Path:
    """Check that a research output exists."""
    path = FILES[file_key]

    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Research output unavailable: {path.name}. "
                "Generate the corresponding research "
                "outputs before using this endpoint."
            )
        )

    return path



def convert_value(key, value):
    """Convert CSV values to JSON-friendly types."""
    if value is None:
        return None

    value = str(value).strip()

    if value.lower() in {
        "", "nan", "none", "null"
    }:
        return None

    # Historical CSV uses True/False.
    # Annual QA CSV uses integer counts
    # under the same column name.
    if key == "pilot_screened":
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"

        return int(value)

    if key in BOOLEAN_FIELDS:
        return value.lower() == "true"

    if key in INTEGER_FIELDS:
        return int(value)

    if key in FLOAT_FIELDS:
        return float(value)

    return value



def read_csv(file_key: str):
    """Read an exported CSV without modifying it."""
    path = require_file(file_key)

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:
            reader = csv.DictReader(file)

            return [
                {
                    key: convert_value(key, value)
                    for key, value in row.items()
                }
                for row in reader
            ]

    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to read {path.name}: "
                f"{type(exc).__name__}"
            )
        ) from exc


def read_json(file_key: str):
    """Read an exported JSON file."""
    path = require_file(file_key)

    try:
        with path.open(
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to read {path.name}: "
                f"{type(exc).__name__}"
            )
        ) from exc


# ==========================================
# 1. RESEARCH OUTPUT STATUS
# ==========================================

@router.get("/status")
def river_corridor_status():

    available = {
        name: path.is_file()
        for name, path in FILES.items()
    }

    return {
        "component": "River Corridor Research",
        "status": (
            "READY"
            if all(available.values())
            else "INCOMPLETE_OUTPUTS"
        ),
        "available_outputs": available,
        "research_stage": "PP1_EXPERIMENTAL",
        "validated_erosion_model": False,
        "production_approved": False,
    }


# ==========================================
# 2. DASHBOARD SUMMARY
# ==========================================

@router.get("/summary")
def river_corridor_summary():

    return read_json("summary")


# ==========================================
# 3. 2026 PROJECTIONS
# ==========================================

@router.get("/projections")
def river_corridor_projections(
    screened_only: bool = False
):

    records = read_csv("projections")

    if screened_only:
        records = [
            row for row in records
            if row.get("screened") is True
        ]

    return {
        "target_year": 2026,
        "record_count": len(records),
        "selected_method": "Persistence",
        "prediction_type": (
            "EXPERIMENTAL_WATER_WIDTH_PROXY"
        ),
        "records": records
    }


# ==========================================
# 4. GEOSPATIAL MAP DATA
# ==========================================

@router.get("/geojson")
def river_corridor_geojson():

    path = require_file("geojson")

    return FileResponse(
        path,
        media_type="application/geo+json",
        filename="river_corridor_projection.geojson",
        content_disposition_type="inline"
    )


# ==========================================
# 5. MODEL COMPARISON
# ==========================================

@router.get("/metrics")
def river_corridor_metrics():

    records = read_csv("metrics")

    return {
        "comparison": (
            "Persistence vs TrainingMeanChange "
            "vs RandomForest"
        ),
        "selection_basis": (
            "2023 validation MAE"
        ),
        "selected_method": "Persistence",
        "measurement": (
            "Satellite-derived pilot water-width proxy"
        ),
        "independent_ground_truth": False,
        "records": records
    }


# ==========================================
# 6. HISTORICAL ANNUAL QA
# ==========================================

@router.get("/annual-qa")
def river_corridor_annual_qa():

    records = read_csv("annual_qa")

    return {
        "observation_years": len(records),
        "records": records,
        "validated_erosion_measurements": False
    }


# ==========================================
# 7. HISTORICAL TRANSECT MEASUREMENTS
# ==========================================

@router.get("/history")
def river_corridor_history(
    year: int | None = Query(
        default=None,
        ge=2010,
        le=2025
    ),
    transect_id: str | None = None
):

    records = read_csv("historical")

    if year is not None:
        records = [
            row for row in records
            if row["year"] == year
        ]

    if transect_id is not None:
        records = [
            row for row in records
            if row["transect_id"] == transect_id
        ]

    return {
        "record_count": len(records),
        "measurement_status": (
            "PRELIMINARY_UNVALIDATED"
        ),
        "records": records
    }


# ==========================================
# 8. PROJECTION COVERAGE QA
# ==========================================

@router.get("/projection-qa")
def river_corridor_projection_qa():

    records = read_csv("projection_qa")

    return {
        "records": records,
        "note": (
            "Historical evaluation only includes "
            "transects passing QA in both years."
        )
    }


# ==========================================
# 9. INDIVIDUAL TRANSECT DETAILS
# ==========================================

@router.get("/transects/{transect_id}")
def river_corridor_transect_details(
    transect_id: str
):

    projections = read_csv("projections")

    matched = [
        row for row in projections
        if row["transect_id"] == transect_id
    ]

    if not matched:
        raise HTTPException(
            status_code=404,
            detail="Transect not found."
        )

    history = [
        row
        for row in read_csv("historical")
        if row["transect_id"] == transect_id
    ]

    history.sort(
        key=lambda row: row["year"]
    )

    return {
        "transect_id": transect_id,
        "projection": matched[0],
        "history": history,
        "independent_ground_truth": False,
        "physical_erosion_prediction": False
    }


# ================================================================
# STEP 31: RIVER CORRIDOR INTELLIGENCE (PP1 research integration)
# New routes are additive; original nine endpoints remain unchanged.
# These outputs are experimental and must not be represented as
# legal encroachment, surveyed bank erosion, or validated risk.
# ================================================================

import io
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from fastapi.responses import Response
from shapely.geometry import mapping

RCPI_GPKG = RESEARCH_ROOT / "data/processed/rcpi_v1/30_preliminary_rcpi_28_reaches.gpkg"
CENTRELINE_GEOJSON = RESEARCH_ROOT / "data/raw/boundaries/kelani_river_centreline.geojson"
WATER_MASK_DIR = RESEARCH_ROOT / "data/processed/sentinel2_change_v2"
OSM_DIR = RESEARCH_ROOT / "data/raw/osm"

INTELLIGENCE_FILES = {
    "rcpi": REPORT_DIR / "30_rcpi_preliminary_all_reaches.csv",
    "vegetation": REPORT_DIR / "28_s2_riparian_change_by_reach.csv",
    "vegetation_annual": REPORT_DIR / "28_s2_annual_riparian_vegetation.csv",
    "urban": REPORT_DIR / "29_osm_pressure_by_reach.csv",
    "width": REPORT_DIR / "27_s2_transect_width_change_pairs.csv",
    "width_annual": REPORT_DIR / "27_s2_transect_widths_2018_2025.csv",
    "model_metrics": REPORT_DIR / "32b_main_forecast_model_metrics.csv",
    "model_selection": REPORT_DIR / "32b_main_forecast_model_selection.csv",
    "future": REPORT_DIR / "32b_main_forecast_2026_predictions.csv",
}


def _intelligence_frame(key: str) -> pd.DataFrame:
    path = INTELLIGENCE_FILES[key]
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"River research output missing: {path.name}. Run the corresponding research step.",
        )
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"River research report unreadable: {path.name} ({type(exc).__name__})",
        ) from exc


def _json_records(frame: pd.DataFrame) -> list:
    """Pandas handles nullable numbers/booleans before FastAPI serialization."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _check_reach(reach_id: str) -> str:
    if not re.fullmatch(r"R\d{3}", reach_id):
        raise HTTPException(status_code=422, detail="Expected reach ID such as R020")
    if not (1 <= int(reach_id[1:]) <= 28):
        raise HTTPException(status_code=404, detail="Reach outside study corridor")
    return reach_id


def _filtered_pairs(frame: pd.DataFrame, reach_id: str):
    return frame[
        (frame["reach_id"] == reach_id)
        & (frame["year_a"].isin([2018, 2020]))
        & (frame["year_b"].isin([2024, 2025]))
    ]


@router.get("/intelligence/overview")
def river_intelligence_overview():
    """Full study panel: no inferred values or prediction probabilities."""
    rcpi = _intelligence_frame("rcpi")
    metrics = _intelligence_frame("model_metrics")
    selection = _intelligence_frame("model_selection")
    urban = _intelligence_frame("urban")
    vegetation = _intelligence_frame("vegetation")
    vegetation = vegetation[
        (vegetation["zone"] == "0-100m")
        & (vegetation["year_a"].isin([2018, 2020]))
        & (vegetation["year_b"].isin([2024, 2025]))
    ]
    vegetation = vegetation[[
        "reach_id", "year_a", "year_b", "net_vegetation_change_ha",
        "vegetation_loss_ha", "vegetation_gain_ha", "matched_land_coverage_pct",
    ]]

    if len(rcpi) != 28 or rcpi["reach_id"].nunique() != 28:
        raise HTTPException(status_code=500, detail="Expected exactly 28 unique RCPI reaches")

    class_counts = rcpi["priority_class"].value_counts().to_dict()
    ordered = rcpi.sort_values("reach_id")
    fields = [
        "reach_id", "priority_class", "priority_rank_eligible_only", "rcpi_eligible",
        "equal_weight_baseline", "morphology_normalized", "vegetation_normalized",
        "urban_normalized", "morphology_emphasis", "vegetation_emphasis", "urban_emphasis",
        "median_abs_width_change_m", "median_signed_width_change_m", "matched_transects",
        "total_transects", "width_match_pct", "matched_land_coverage_pct",
        "vegetation_coverage_caution_below85", "reason_not_scored",
        "vegetation_loss_ha", "vegetation_gain_ha", "net_vegetation_change_ha",
        "median_ndvi_delta", "building_footprint_pct", "road_density_km_per_km2",
        "building_features_intersecting", "zone_area_ha", "qa_flags", "interpretation",
    ]
    chosen = [c for c in fields if c in ordered.columns]
    return {
        "metadata": {
            "study": "Kelani River Corridor · Kaduwela area",
            "research_stage": "PP1_EXPLORATORY",
            "river_length_km": 13.53691,
            "reaches": 28,
            "transects": 271,
            "scored_reaches": int(ordered["rcpi_eligible"].sum()),
            "unscored_reaches": int((~ordered["rcpi_eligible"]).sum()),
            "priority_counts": class_counts,
            "rcpi_period": "2020–2024",
            "urban_source": "OSM snapshot — not verified historical urban growth",
            "limitations": (
                "Satellite-derived water edges are not surveyed banklines. "
                "Vegetation threshold changes are not confirmed deforestation. "
                "RCPI is a relative experimental planning index, not validated legal or physical risk."
            ),
        },
        "reaches": _json_records(ordered[chosen]),
        "vegetation_comparisons": _json_records(vegetation),
        "models": {
            "metrics": _json_records(metrics),
            "selection": _json_records(selection),
            "validation_year": 2024,
            "test_year": 2025,
            "note": "Selected on 2024 validation, not 2025 test; persistence and training mean are baselines.",
        },
        "urban_features_total": int(urban["building_features_intersecting"].sum()),
    }


@router.get("/intelligence/reaches.geojson")
def river_intelligence_reach_geometry():
    if not RCPI_GPKG.is_file():
        raise HTTPException(status_code=503, detail="Step 30 RCPI GeoPackage not found")
    try:
        layer = gpd.read_file(RCPI_GPKG)
        if layer.crs is None or layer["reach_id"].nunique() != 28:
            raise ValueError("Missing CRS or unexpected reach count")
        # Simplify in *metres*, before conversion to WGS84.
        layer = layer.to_crs("EPSG:32644")
        layer["geometry"] = layer.geometry.simplify(3.0, preserve_topology=True)
        layer = layer.to_crs("EPSG:4326")
        cols = [
            "reach_id", "priority_class", "rcpi_eligible", "equal_weight_baseline",
            "building_footprint_pct", "road_density_km_per_km2",
            "matched_transects", "net_vegetation_change_ha", "geometry",
        ]
        layer = layer[[c for c in cols if c in layer.columns]]
        return json.loads(layer.to_json(drop_id=True, na="null"))
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=f"RCPI layer unavailable: {type(exc).__name__}") from exc


@router.get("/intelligence/centreline.geojson")
def river_intelligence_centreline():
    if not CENTRELINE_GEOJSON.is_file():
        raise HTTPException(status_code=503, detail="River centreline file not found")
    return FileResponse(CENTRELINE_GEOJSON, media_type="application/geo+json")


@router.get("/intelligence/reach/{reach_id}")
def river_intelligence_reach_details(reach_id: str):
    reach_id = _check_reach(reach_id)
    annual_width = _intelligence_frame("width_annual")
    annual_width = annual_width[annual_width["reach_id"] == reach_id]
    width_series = []
    for year, group in annual_width.groupby("year"):
        quality = group[group["qa_pass"] == 1]
        width_series.append({
            "year": int(year),
            "median_width_m": float(quality["apparent_width_screened_m"].median()) if len(quality) else None,
            "screened": int(len(quality)),
            "total": int(len(group)),
        })

    vegetation = _intelligence_frame("vegetation_annual")
    vegetation = vegetation[
        (vegetation["reach_id"] == reach_id) & (vegetation["zone"] == "0-100m")
    ].sort_values("year")
    vegetation_cols = [c for c in [
        "year", "observation_date", "median_ndvi", "median_evi",
        "observed_veg_area_ha", "veg_pct_of_valid_land", "observed_land_coverage_pct",
    ] if c in vegetation.columns]

    w = _filtered_pairs(_intelligence_frame("width"), reach_id)
    pair_width = []
    for (year_a, year_b), group in w.groupby(["year_a", "year_b"]):
        good = group[group["qa_matched"] == 1]
        pair_width.append({
            "year_a": int(year_a), "year_b": int(year_b),
            "matched": int(len(good)), "total": int(len(group)),
            "median_signed_delta_m": float(good["apparent_width_delta_m"].median()) if len(good) else None,
            "median_abs_delta_m": float(good["apparent_width_delta_m"].abs().median()) if len(good) else None,
            "positive_over20": int((good["apparent_width_delta_m"] > 20).sum()),
            "negative_under_minus20": int((good["apparent_width_delta_m"] < -20).sum()),
        })

    veg = _filtered_pairs(_intelligence_frame("vegetation"), reach_id)
    veg = veg[veg["zone"] == "0-100m"]
    veg_cols = [c for c in [
        "year_a", "year_b", "date_a", "date_b", "matched_land_coverage_pct",
        "median_ndvi_delta", "vegetation_loss_ha", "vegetation_gain_ha",
        "net_vegetation_change_ha", "qa_flags",
    ] if c in veg.columns]

    forecasts = _intelligence_frame("future")
    forecasts = forecasts[forecasts["reach_id"] == reach_id]
    return {
        "reach_id": reach_id,
        "width_annual": width_series,
        "vegetation_annual": _json_records(vegetation[vegetation_cols]),
        "width_pairs": pair_width,
        "vegetation_pairs": _json_records(veg[veg_cols]),
        "forecasts_2026": _json_records(forecasts),
    }


# ==========================================
# Sentinel-2 10 m transect evidence endpoints
# ==========================================
# Use exactly the same reference transect geometries and annual width
# measurements as Step 27, NOT the old Step 17 Landsat-pilot GeoJSON.
# These are exploratory classified-water measurements, not surveyed banks.

TRANSECT_REFERENCE_GPKG = (
    RESEARCH_ROOT / "data/processed/transects/01_transects_50m.gpkg"
)


def _verify_reference_transect_links(gdf: gpd.GeoDataFrame, observations: pd.DataFrame):
    """Fail closed when map reach IDs disagree with width research outputs."""
    required_geom = {"transect_id", "reach_id"}
    required_obs = {"transect_id", "reach_id", "year", "qa_pass"}
    if not required_geom.issubset(gdf.columns) or not required_obs.issubset(observations.columns):
        raise HTTPException(status_code=500, detail="Missing transect reference schema")
    if len(gdf) != 271 or gdf["transect_id"].nunique() != 271:
        raise HTTPException(status_code=500, detail="Expected 271 unique Step 27 reference transects")
    if gdf.crs is None or not gdf.geometry.is_valid.all() or gdf.geometry.is_empty.any():
        raise HTTPException(status_code=500, detail="Invalid/missing CRS or geometry in reference transects")
    lookup = observations[["transect_id", "reach_id"]].drop_duplicates()
    if lookup["transect_id"].nunique() != 271 or len(lookup) != 271:
        raise HTTPException(status_code=500, detail="Unexpected Step 27 transect/reach membership")
    check = gdf[["transect_id", "reach_id"]].merge(
        lookup, on="transect_id", how="left", suffixes=("_map", "_width"),
        validate="one_to_one",
    )
    mismatch = check[check["reach_id_map"] != check["reach_id_width"]]
    if not mismatch.empty:
        examples = ", ".join(mismatch["transect_id"].head(4).astype(str).tolist())
        raise HTTPException(status_code=500, detail=f"Step 27 transect/reach mismatch: {examples}")


@router.get("/intelligence/transects.geojson")
def river_intelligence_reference_transects():
    """Verified Step 27 lines; legacy pilot geometry must not drive new widths."""
    if not TRANSECT_REFERENCE_GPKG.is_file():
        raise HTTPException(status_code=503, detail="Step 27 reference transects GeoPackage missing")
    try:
        gdf = gpd.read_file(TRANSECT_REFERENCE_GPKG)
        observations = _intelligence_frame("width_annual")
        _verify_reference_transect_links(gdf, observations)
        quality = observations.groupby("transect_id")["qa_pass"].sum().to_dict()
        gdf["qa_usable_years"] = gdf["transect_id"].map(quality).fillna(0).astype(int)
        gdf = gdf.to_crs("EPSG:4326")
        wanted = ["transect_id", "reach_id", "chainage_m", "qa_usable_years", "geometry"]
        gdf = gdf[[c for c in wanted if c in gdf.columns]]
        return json.loads(gdf.to_json(drop_id=True, na="null"))
    except HTTPException:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=f"Transect reference unavailable ({type(exc).__name__})") from exc


@router.get("/intelligence/transects/{transect_id}/annual")
def river_intelligence_transect_annual(transect_id: str):
    """Actual 2018-2025 QA-screened Sentinel-2 observations for one transect."""
    if not re.fullmatch(r"T\d{4}", transect_id) or not 1 <= int(transect_id[1:]) <= 271:
        raise HTTPException(status_code=422, detail="Expected transect ID T0001 to T0271")
    annual = _intelligence_frame("width_annual")
    subset = annual[annual["transect_id"] == transect_id].copy()
    if len(subset) != 8 or subset["year"].nunique() != 8:
        raise HTTPException(status_code=503, detail="Incomplete Step 27 annual transect observations")
    if subset["reach_id"].nunique() != 1:
        raise HTTPException(status_code=500, detail="Conflicting Step 27 transect reach membership")
    subset = subset.sort_values("year")
    subset["qa_pass"] = subset["qa_pass"].astype(int)
    subset["screened_width_m"] = subset["apparent_width_screened_m"].where(
        subset["qa_pass"] == 1
    )
    cols = [
        "year", "reach_id", "qa_pass", "qa_status", "qa_flags",
        "screened_width_m", "observed_valid_transect_pct",
        "start_side_water_edge_offset_m", "end_side_water_edge_offset_m",
    ]
    return {
        "transect_id": transect_id,
        "reach_id": str(subset["reach_id"].iloc[0]),
        "source": "Step 27 · Sentinel-2 NDWI+MNDWI agreement · 10 m grid",
        "units": "metres",
        "qa_usable_years": int(subset["qa_pass"].sum()),
        "total_years": 8,
        "observations": _json_records(subset[cols]),
        "interpretation": "Exploratory apparent water width; not a surveyed bankline or confirmed erosion.",
    }


@router.get("/intelligence/osm/{reach_id}")
def river_intelligence_osm_overlay(reach_id: str):
    """Read-only, reach-local imagery/context; display counts NOT the pressure denominators."""
    reach_id = _check_reach(reach_id)
    if not RCPI_GPKG.is_file():
        raise HTTPException(status_code=503, detail="Reach polygons unavailable")
    reach = gpd.read_file(RCPI_GPKG)
    match = reach[reach["reach_id"] == reach_id]
    if len(match) != 1:
        raise HTTPException(status_code=404, detail="Reach not found")
    near = match.to_crs("EPSG:32644").geometry.iloc[0].buffer(120)
    near_gs = gpd.GeoSeries([near], crs="EPSG:32644").to_crs("EPSG:4326")
    bbox = tuple(near_gs.total_bounds)
    payload = {}
    for key, filename, kind, fields in [
        ("buildings", "kelani_osm_buildings_raw.geojson", "polygon", ["building", "name"]),
        ("roads", "kelani_osm_roads_raw.geojson", "line", ["highway", "name"]),
    ]:
        path = OSM_DIR / filename
        if not path.is_file():
            payload[key] = {"type": "FeatureCollection", "features": []}
            continue
        gdf = gpd.read_file(path, bbox=bbox)
        if gdf.crs is None:
            raise HTTPException(status_code=500, detail=f"Missing OSM CRS: {filename}")
        types = ["Polygon", "MultiPolygon"] if kind == "polygon" else ["LineString", "MultiLineString"]
        gdf = gdf[gdf.geom_type.isin(types)].to_crs("EPSG:32644")
        gdf = gdf[gdf.geometry.intersects(near)].head(650)
        gdf = gdf.to_crs("EPSG:4326")
        kept = [c for c in fields if c in gdf.columns]
        payload[key] = json.loads(gdf[kept + ["geometry"]].to_json(drop_id=True, na="null"))
    return {
        "reach_id": reach_id,
        "display_area": "Selected reach + 120m context buffer",
        "note": "Map inspection features are not legal encroachments or a separate measured pressure count.",
        **payload,
    }


@router.get("/intelligence/water-catalog")
def river_intelligence_water_catalog():
    """Optional research raster overlay metadata (actual masks, distinct from online basemap)."""
    from rasterio import open as raster_open
    from rasterio.warp import transform_bounds

    available = []
    for year in range(2018, 2026):
        path = WATER_MASK_DIR / f"22_s2_{year}_water_agreement_10m.tif"
        if not path.is_file():
            continue
        try:
            with raster_open(path) as ds:
                west, south, east, north = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
            available.append({"year": year, "bounds": [[south, west], [north, east]]})
        except (OSError, ValueError):
            continue
    return {
        "layers": available,
        "method": "Strict NDWI + MNDWI water agreement; satellite derived, not a surveyed bank boundary.",
    }


@router.get("/intelligence/water-overlay/{year}.png")
def river_intelligence_water_image(year: int):
    from PIL import Image
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    if year < 2018 or year > 2025:
        raise HTTPException(status_code=404, detail="Year outside Sentinel-2 study window")
    path = WATER_MASK_DIR / f"22_s2_{year}_water_agreement_10m.tif"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Water mask not available")
    with rasterio.open(path) as ds:
        dst_transform, width, height = calculate_default_transform(
            ds.crs, "EPSG:4326", ds.width, ds.height, *ds.bounds
        )
        water = np.full((height, width), 255, dtype="uint8")
        reproject(
            source=rasterio.band(ds, 1), destination=water,
            src_transform=ds.transform, src_crs=ds.crs,
            dst_transform=dst_transform, dst_crs="EPSG:4326",
            src_nodata=255, dst_nodata=255, resampling=Resampling.nearest,
        )
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    mask = water == 1
    rgba[mask] = (34, 211, 238, 205)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    return Response(
        content=buffer.getvalue(), media_type="image/png",
        headers={"Cache-Control": "public, max-age=1800"},
    )
