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
# WETLAND RESEARCH COMPONENT
# ============================================================

WETLAND_ROOT = PROJECT_ROOT / "Wetland_Research"


# ============================================================
# PART 1 - HISTORICAL WETLAND CHANGE
# ============================================================

WETLAND_PART1_PROCESSED_DIR = (
    WETLAND_ROOT
    / "data"
    / "processed"
    / "part1_historical_change"
)

# Part 1 - Historical observation polygons
# Contains 448 cleaned historical observation records.
WETLAND_PART1_HISTORICAL_OBSERVATIONS_FILE = (
    WETLAND_PART1_PROCESSED_DIR
    / "historical_wetland_clean_observations_wgs84.geojson"
)

WETLAND_PART1_TABLES_DIR = (
    WETLAND_ROOT
    / "outputs"
    / "tables"
    / "part1"
)

WETLAND_PART1_SUMMARY_FILE = (
    WETLAND_PART1_TABLES_DIR
    / "wetland_backend_summary.csv"
)

WETLAND_PART1_MAP_FILE = (
    WETLAND_PART1_PROCESSED_DIR
    / "wetland_latest_status_wgs84.geojson"
)

WETLAND_PART1_VALIDATION_FILE = (
    WETLAND_PART1_TABLES_DIR
    / "backend_validation.csv"
)


# ============================================================
# PART 2 - CANDIDATE WETLAND DETECTION
# ============================================================

WETLAND_PART2_MAPS_DIR = (
    WETLAND_ROOT
    / "outputs"
    / "maps"
    / "part2_candidate_wetland"
)

# 48 primary candidate wetland polygons
WETLAND_PART2_PRIMARY_MAP_FILE = (
    WETLAND_PART2_MAPS_DIR
    / "20_primary_candidate_layer_wgs84_repaired.geojson"
)

# 1,097 broader candidate polygons
WETLAND_PART2_ALL_CANDIDATES_FILE = (
    WETLAND_PART2_MAPS_DIR
    / "20_final_candidate_priority_wgs84_repaired.geojson"
)

# 56 GN divisions with candidate wetland summaries
WETLAND_PART2_GN_SUMMARY_FILE = (
    WETLAND_PART2_MAPS_DIR
    / "22_gn_primary_candidate_summary_wgs84.geojson"
)


# ============================================================
# PART 3 - FUTURE WETLAND PROJECTION
# ============================================================

WETLAND_PART3_PROCESSED_DIR = (
    WETLAND_ROOT
    / "data"
    / "processed"
    / "part3_future_projection"
)

# Estimated wetland distribution in 2025
WETLAND_PART3_2025_FILE = (
    WETLAND_PART3_PROCESSED_DIR
    / "Part3_wetland_2025_WGS84.gpkg"
)

# Exploratory CA-Markov wetland scenario for 2030
WETLAND_PART3_2030_FILE = (
    WETLAND_PART3_PROCESSED_DIR
    / "Part3_wetland_2030_WGS84.gpkg"
)

# Scenario change layers
WETLAND_PART3_GAIN_FILE = (
    WETLAND_PART3_PROCESSED_DIR
    / "Part3_scenario_gain_WGS84.gpkg"
)

WETLAND_PART3_LOSS_FILE = (
    WETLAND_PART3_PROCESSED_DIR
    / "Part3_scenario_loss_WGS84.gpkg"
)

WETLAND_PART3_PERSISTENCE_FILE = (
    WETLAND_PART3_PROCESSED_DIR
    / "Part3_persistence_WGS84.gpkg"
)

# GN-wise historical and future scenario statistics
WETLAND_PART3_GN_SUMMARY_FILE = (
    WETLAND_PART3_PROCESSED_DIR
    / "GN_Wise_Wetland_Scenario_WGS84.gpkg"
)


# ============================================================
# PART 4 - SEASONAL WATERLOGGING SUSCEPTIBILITY
# ============================================================

WETLAND_PART4_PROCESSED_DIR = (
    WETLAND_ROOT / "data" / "processed" / "part4_waterlogging"
)


# ============================================================
# PART 4 - SUSCEPTIBILITY RASTER MAPS
# ============================================================

WETLAND_PART4_MAIN_CLASS_FILE = (
    WETLAND_PART4_PROCESSED_DIR
    / "13_Terrain_Urban_Susceptibility_Class_30m.tif"
)

WETLAND_PART4_TERRAIN_CLASS_FILE = (
    WETLAND_PART4_PROCESSED_DIR
    / "13_Terrain_Only_Susceptibility_Class_30m.tif"
)

WETLAND_PART4_SCENARIO_SPAN_FILE = (
    WETLAND_PART4_PROCESSED_DIR
    / "17_Scenario_Class_Span_30m.tif"
)


WETLAND_PART4_GN_SUMMARY_FILE = (
    WETLAND_PART4_PROCESSED_DIR
    / "18_GN_Waterlogging_Susceptibility.gpkg"
)

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