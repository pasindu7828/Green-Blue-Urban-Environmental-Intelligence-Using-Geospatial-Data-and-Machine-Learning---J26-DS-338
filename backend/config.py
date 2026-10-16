from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

# backend/config.py -> parent = backend
# backend/config.py -> parents[1] = main project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# COMPONENT ROOT DIRECTORIES
# ============================================================

# Urban Heat Research component
URBAN_HEAT_ROOT = PROJECT_ROOT / "Urban_Heat_Research"


# ============================================================
# URBAN HEAT DASHBOARD OUTPUTS
# ============================================================

HEAT_DASHBOARD_DIR = URBAN_HEAT_ROOT / "outputs" / "dashboard"

HEAT_MAPS_DIR = HEAT_DASHBOARD_DIR / "maps"
HEAT_SCENARIOS_DIR = HEAT_DASHBOARD_DIR / "scenarios"


# ============================================================
# STEP 34 DASHBOARD FILES
# ============================================================

GRID_METADATA_FILE = (
    HEAT_DASHBOARD_DIR
    / "34_grid_metadata.csv"
)

HISTORICAL_HEAT_FILE = (
    HEAT_DASHBOARD_DIR
    / "34_historical_heat_2015_2025.csv"
)

FUTURE_HEAT_FILE = (
    HEAT_DASHBOARD_DIR
    / "34_future_dashboard_master_2026_2030.csv"
)

COOLING_PRIORITY_FILE = (
    HEAT_DASHBOARD_DIR
    / "34_cooling_priority_dashboard.csv"
)

CURRENT_VS_FORECAST_FILE = (
    HEAT_DASHBOARD_DIR
    / "34_current_vs_forecast_2025_2030.csv"
)

AREA_HEAT_SUMMARY_FILE = (
    HEAT_DASHBOARD_DIR
    / "34_area_heat_summary_2015_2030.csv"
)

AREA_PRIORITY_SUMMARY_FILE = (
    HEAT_DASHBOARD_DIR
    / "34_area_priority_summary.csv"
)

DASHBOARD_MANIFEST_FILE = (
    HEAT_DASHBOARD_DIR
    / "34_dashboard_manifest.json"
)


# ============================================================
# GEOJSON / MAP FILES
# ============================================================

GRID_METADATA_GEOJSON = (
    HEAT_MAPS_DIR
    / "34_grid_metadata_wgs84.geojson"
)

FUTURE_HEAT_GEOJSON = (
    HEAT_MAPS_DIR
    / "34_future_dashboard_master_2026_2030_wgs84.geojson"
)

COOLING_PRIORITY_GEOJSON = (
    HEAT_MAPS_DIR
    / "34_cooling_priority_wgs84.geojson"
)


# ============================================================
# BASIC PROJECT SETTINGS
# ============================================================

API_TITLE = "Green-Blue Urban Environmental Intelligence API"
API_VERSION = "1.0.0"

HISTORICAL_START_YEAR = 2015
HISTORICAL_END_YEAR = 2025

FORECAST_START_YEAR = 2026
FORECAST_END_YEAR = 2030