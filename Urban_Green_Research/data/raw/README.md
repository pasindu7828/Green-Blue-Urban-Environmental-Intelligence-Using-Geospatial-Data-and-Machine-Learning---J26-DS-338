# Urban Green Research - Raw Data

This directory contains the source and reference datasets used for the Urban Green Transition, Urbanization Pressure and Conservation Intelligence research component.

## Raw Data Policy

Raw datasets are preserved without modification.

Cleaning, clipping, reprojection, feature extraction, reclassification, rasterization and other transformations must be performed through reproducible processing scripts and written to the `data/processed/` directory.

Large and restricted raw datasets are excluded from GitHub using `.gitignore`.

---

## Dataset Groups

### 1. Administrative Boundaries

#### KMC Boundary
Path:
`boundaries/kmc/`

Source:
Kaduwela Municipal Council / Urban Development Authority supporting data.

Purpose:
Defines the primary study-area boundary.

#### GN Divisions

Raw source path:
`boundaries/gn_divisions/Kaduwela TO Area.kml`

The original KMC/UDA source contains a 56-feature GN boundary layer.

Processed working dataset:
`../processed/boundaries/gn_divisions/kaduwela_gn_divisions.shp`

The processed dataset contains 57 GN divisions after the boundary preparation and cleaning workflow.

Purpose:
Used for GN-level reporting, aggregation and spatial summaries.

Important:
The original KML is retained unchanged. The 57-feature Shapefile is treated as a processed working dataset and must not be presented as the untouched original source.

---

### 2. Landsat Annual Composites

Path:
`landsat/annual_composites/`

Period:
2010-2026.

Purpose:
Historical land-use/land-cover classification and green-cover / urbanization change analysis.

Notes:
- Annual surface-reflectance composites were prepared using Google Earth Engine.
- Spatial resolution: 30 m.
- Export CRS: EPSG:32644.
- The 2026 composite represents a partial year and must be interpreted as provisional.

---

### 3. OSM Roads and Buildings

Path:
`osm/roads_buildings/`

Source:
OpenStreetMap data obtained through the HOT Export Tool.

Purpose:
Used later to derive accessibility, road-density, building-density and urban-development features.

Important:
The original GeoPackage is preserved unchanged. Filtering and clipping will be reproduced through code.

---

### 4. Terrain Data

#### GEDTM30
Path:
`dem/gedtm30/`

Role:
Primary terrain dataset.

#### NASADEM
Path:
`dem/nasadem/`

Role:
Secondary cross-dataset terrain consistency reference.

Important:
NASADEM is not treated as independent surveyed ground truth.

---

### 5. ESA WorldCover 2021

Path:
`reference_lulc/esa_worldcover_2021/`

Purpose:
Independent reference for checking general consistency of the Landsat-based land-cover classification.

Important:
ESA WorldCover is an external model-generated land-cover product and is not treated as ground truth.

---

### 6. UDA Development Plan

Path:
`uda/development_plan/`

Document:
Kaduwela Development Plan 2023-2033.

Role:
Official planning-document source for zoning categories, planning regulations, land-use information, conservation information and urban-centre context.

---

### 7. UDA Provisional Zoning Package

Path:
`uda/zoning_provisional/`

Role:
Spatial planning-reference dataset supplied for research use.

Important limitations:
- This package is not the original authoritative UDA GIS database.
- It must not be presented as authoritative legal zoning data.
- It is used for planning-context, screening and research interpretation.
- The workflow must allow the authoritative UDA layer to replace it later if access becomes available.

The original supplied ZIP is preserved unchanged.

---

### 8. Supporting UDA/KMC Files

Path:
`uda/supporting_files/`

These files are retained as supporting reference material and will only be used when their relevance to the component is clearly established.

---

## Reproducibility Principle

The research follows this workflow:

Raw Data
→ Programmatic Validation
→ Preprocessing
→ Processed Data
→ Feature Engineering
→ Model-Ready Data
→ Modelling
→ Validation
→ Decision-Support Outputs

Original raw datasets must never be overwritten.