
# River Corridor Research

## Project
Green-Blue Urban Environmental Intelligence
Using Geospatial Data and Machine Learning

Group ID: J26-DS-338

## Research Component
Urban River Corridor Morphological Change
and Riverbank Encroachment Risk Intelligence

## Study Area
Selected Kelani River Corridor, Kaduwela,
Sri Lanka.

## Research Objectives
1. Analyze historical river corridor morphological changes.
2. Identify river-water boundary and width changes.
3. Analyze vegetation and urbanization pressure.
4. Develop and evaluate future-change prediction models.
5. Develop a River Corridor Planning Priority Index (RCPI).
6. Present research results through an interactive Web GIS.

## Data Sources
- Landsat 5, 7 and 8 imagery (2010-2025)
- Sentinel-2 imagery (2018-2025)
- Copernicus DEM
- JRC Global Surface Water
- OpenStreetMap buildings and roads
- KMC administrative boundary
- Candidate river centreline and study corridor

## Research Workflow
Data Collection
-> Data Quality Assessment
-> Satellite Preprocessing
-> River-Water Extraction
-> Morphological Change Analysis
-> Environmental Feature Engineering
-> Structured Master Dataset
-> ML Training and Evaluation
-> Future Change Prediction
-> RCPI Development
-> Web GIS Integration

## Current Status
Core raw data collection substantially completed.
Preprocessing and model development in progress.

## Data Quality Notes
- Landsat 7 observations in 2012/2013 contain gaps.
- Current OSM features are not historical observations.
- JRC is a supplementary water reference.
- River-water boundary change is not automatically
  evidence of physical riverbank erosion.
- Independent riverbank validation is pending.

## Technology
Python, Google Earth Engine, QGIS,
Scikit-learn, FastAPI and React.

## Repository
This folder contains only the River Corridor
research component.

Shared backend and frontend are maintained
at the group repository root.
