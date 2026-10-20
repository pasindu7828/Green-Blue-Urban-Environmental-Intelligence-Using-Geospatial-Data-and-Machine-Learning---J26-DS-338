
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
