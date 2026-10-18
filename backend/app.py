from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import API_TITLE, API_VERSION
from backend.routes.urban_heat_routes import router as urban_heat_router

from backend.routes.wetland_routes import router as wetland_router

from backend.routes.urban_heat_scenario_routes import (
    router as urban_heat_scenario_router,
)

from backend.routes.urban_heat_dynamic_forecast_routes import (
    router as urban_heat_dynamic_forecast_router,
)

# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Common backend API for the Green-Blue Urban "
        "Environmental Intelligence research system."
    ),
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

# These origins will allow the React frontend to communicate
# with this FastAPI backend during development.
allowed_origins = [
    "http://localhost:5173",   # Vite default
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Green-Blue Urban Environmental Intelligence API",
        "version": API_VERSION,
        "docs": "/docs",
    }

app.include_router(urban_heat_router)
app.include_router(urban_heat_scenario_router)
app.include_router(urban_heat_dynamic_forecast_router)
app.include_router(wetland_router)