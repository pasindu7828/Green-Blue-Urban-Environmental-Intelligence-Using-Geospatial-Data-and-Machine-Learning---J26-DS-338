"""
STEP 16A - DEVELOPMENT SUITABILITY CRITERIA AND DATA READINESS AUDIT

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Audit all datasets available for the independent Development Suitability stage.
2. Confirm file existence, CRS, spatial coverage and basic GIS integrity.
3. Separate CORE candidate datasets from SENSITIVITY/REFERENCE datasets.
4. Keep provisional UDA planning GIS as CONTEXT-ONLY.
5. Explicitly exclude Urbanization Pressure, SHAP/XAI and later Conservation
   outputs from the suitability score.
6. Create a transparent criterion registry before any GIS-MCDA/AHP weighting.

IMPORTANT:
- No suitability score is calculated in this step.
- No AHP weights are assigned.
- No pressure-model output is used.
- No SHAP output is used.
- No conservation-priority output is used.
- Provisional UDA zoning is NOT treated as an authoritative legal constraint.
"""

from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from shapely.geometry import box


# =============================================================================
# 1. PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# 2. CORE INPUT PATHS
# =============================================================================

GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "grid"
    / "kaduwela_100m_modelling_grid.gpkg"
)

ROADS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "osm"
    / "kaduwela_roads_2026.gpkg"
)

BUILDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "osm"
    / "kaduwela_buildings_2026.gpkg"
)

GEDTM_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "gedtm30_kmc_30m.tif"
)

NASADEM_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "nasadem_kmc_30m.tif"
)

WORLDCOVER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_lulc"
    / "esa_worldcover_2021_kmc_10m.tif"
)


# =============================================================================
# 3. PROVISIONAL UDA PATHS
# =============================================================================

UDA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "uda"
    / "zoning_provisional"
    / "provided_package_extracted"
    / "New data"
)

UDA_HIGH_LAND_APPROX = (
    UDA_ROOT
    / "Kaduwela_2_High_Land_Conservation_Zones_APPROXIMATE.geojson"
)

UDA_CONSERVATION_REVIEW = (
    UDA_ROOT
    / "Kaduwela_Conservation_Review_Screening_PROVISIONAL.gpkg"
)

UDA_AUDITED_PROVISIONAL = (
    UDA_ROOT
    / "Kaduwela_UDA_2024_AUDITED_PROVISIONAL.gpkg"
)

UDA_PORTABLE_ZIP = (
    UDA_ROOT
    / "Kaduwela_Zoning_QA_Reviewed_Portable_Package.zip"
)

UDA_QGIS_PROJECT = (
    UDA_ROOT
    / "Kaduwela_Zoning_Research_PORTABLE_REVIEW.qgz"
)

UDA_VISUAL_TIF = (
    UDA_ROOT
    / "UDA_2024_Georeferenced_Screenshot_VISUAL_ONLY.tif"
)


# =============================================================================
# 4. OUTPUT PATHS
# =============================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INVENTORY_CSV = (
    OUTPUT_DIR
    / "16a_suitability_dataset_inventory.csv"
)

VECTOR_AUDIT_CSV = (
    OUTPUT_DIR
    / "16a_suitability_vector_audit.csv"
)

RASTER_AUDIT_CSV = (
    OUTPUT_DIR
    / "16a_suitability_raster_audit.csv"
)

CRITERIA_REGISTRY_CSV = (
    OUTPUT_DIR
    / "16a_suitability_criteria_registry.csv"
)

EXCLUSION_REGISTRY_CSV = (
    OUTPUT_DIR
    / "16a_suitability_exclusion_registry.csv"
)

QA_CSV = (
    OUTPUT_DIR
    / "16a_suitability_data_readiness_qa.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "16a_suitability_data_readiness_summary.json"
)


# =============================================================================
# 5. EXPECTED REFERENCE SYSTEM
# =============================================================================

EXPECTED_ANALYSIS_EPSG = 32644

EXPECTED_GRID_CELLS = 9240


# =============================================================================
# 6. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """
    Return project-relative path where possible.
    """

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()

    except ValueError:
        return str(path)


def file_size_mb(path: Path):
    """
    Return file size in MB.
    """

    if not path.exists():
        return np.nan

    return round(
        path.stat().st_size / (1024 ** 2),
        4,
    )


def crs_to_text(crs):
    """
    Convert CRS to a readable text representation.
    """

    if crs is None:
        return None

    try:
        epsg = crs.to_epsg()

        if epsg is not None:
            return f"EPSG:{epsg}"

    except Exception:
        pass

    return str(crs)


def safe_epsg(crs):
    """
    Return EPSG integer where available.
    """

    if crs is None:
        return None

    try:
        return crs.to_epsg()

    except Exception:
        return None


def bbox_intersects(bounds_a, bounds_b):
    """
    Check whether two bounding boxes intersect.
    """

    geom_a = box(
        bounds_a[0],
        bounds_a[1],
        bounds_a[2],
        bounds_a[3],
    )

    geom_b = box(
        bounds_b[0],
        bounds_b[1],
        bounds_b[2],
        bounds_b[3],
    )

    return bool(
        geom_a.intersects(
            geom_b
        )
    )


# =============================================================================
# 7. DATASET INVENTORY DEFINITION
# =============================================================================

DATASET_REGISTRY = [
    {
        "dataset_id": "GRID_100M",
        "dataset_name": "Kaduwela 100 m modelling grid",
        "path": GRID_FILE,
        "data_type": "VECTOR",
        "role": "ANALYSIS_FRAMEWORK",
        "usage_status": "CORE",
        "allowed_use": (
            "Common 100 m spatial analysis framework for Development Suitability"
        ),
        "restriction": (
            "Grid is an analysis framework, not a suitability criterion itself"
        ),
    },
    {
        "dataset_id": "OSM_ROADS_2026",
        "dataset_name": "Kaduwela OSM roads 2026",
        "path": ROADS_FILE,
        "data_type": "VECTOR",
        "role": "SUITABILITY_INPUT_CANDIDATE",
        "usage_status": "CORE_CANDIDATE",
        "allowed_use": (
            "Road-accessibility and road-network context criteria"
        ),
        "restriction": (
            "Road classes and exact criterion definitions must be audited "
            "before standardization"
        ),
    },
    {
        "dataset_id": "OSM_BUILDINGS_2026",
        "dataset_name": "Kaduwela OSM buildings 2026",
        "path": BUILDINGS_FILE,
        "data_type": "VECTOR",
        "role": "SUITABILITY_INPUT_CANDIDATE",
        "usage_status": "CORE_CANDIDATE",
        "allowed_use": (
            "Existing-development / built-context criteria"
        ),
        "restriction": (
            "Must not be treated as historical accessibility evidence"
        ),
    },
    {
        "dataset_id": "GEDTM30",
        "dataset_name": "GEDTM30 terrain model",
        "path": GEDTM_FILE,
        "data_type": "RASTER",
        "role": "SUITABILITY_INPUT_CANDIDATE",
        "usage_status": "CORE_CANDIDATE",
        "allowed_use": (
            "Terrain criteria such as slope and elevation"
        ),
        "restriction": (
            "Criterion direction and standardization require explicit "
            "planning/literature justification"
        ),
    },
    {
        "dataset_id": "NASADEM",
        "dataset_name": "NASADEM terrain model",
        "path": NASADEM_FILE,
        "data_type": "RASTER",
        "role": "DEM_SENSITIVITY_REFERENCE",
        "usage_status": "SENSITIVITY_REFERENCE",
        "allowed_use": (
            "Sensitivity comparison against primary GEDTM30 terrain variables"
        ),
        "restriction": (
            "Not ground truth and not the primary terrain source"
        ),
    },
    {
        "dataset_id": "ESA_WORLDCOVER_2021",
        "dataset_name": "ESA WorldCover 2021",
        "path": WORLDCOVER_FILE,
        "data_type": "RASTER",
        "role": "LAND_COVER_REFERENCE",
        "usage_status": "REFERENCE_CONTEXT",
        "allowed_use": (
            "Independent land-cover reference or suitability screening support"
        ),
        "restriction": (
            "2021 reference only; not current 2026 ground truth"
        ),
    },
    {
        "dataset_id": "UDA_HIGH_LAND_APPROX",
        "dataset_name": (
            "Approximate high-land conservation zones"
        ),
        "path": UDA_HIGH_LAND_APPROX,
        "data_type": "VECTOR",
        "role": "PLANNING_CONTEXT",
        "usage_status": "CONTEXT_ONLY",
        "allowed_use": (
            "Visual planning-context overlay and later sensitivity discussion"
        ),
        "restriction": (
            "Approximate; cannot be used as a legal exclusion, "
            "ground truth or core AHP criterion"
        ),
    },
    {
        "dataset_id": "UDA_CONSERVATION_REVIEW",
        "dataset_name": (
            "Provisional conservation review screening"
        ),
        "path": UDA_CONSERVATION_REVIEW,
        "data_type": "VECTOR",
        "role": "PLANNING_CONTEXT",
        "usage_status": "CONTEXT_ONLY",
        "allowed_use": (
            "Planning-context overlay and research review only"
        ),
        "restriction": (
            "Provisional; cannot be used as authoritative zoning or "
            "legal suitability constraint"
        ),
    },
    {
        "dataset_id": "UDA_AUDITED_PROVISIONAL",
        "dataset_name": (
            "UDA 2024 audited provisional GIS"
        ),
        "path": UDA_AUDITED_PROVISIONAL,
        "data_type": "VECTOR",
        "role": "PLANNING_CONTEXT",
        "usage_status": "CONTEXT_ONLY",
        "allowed_use": (
            "Planning-context comparison and later conflict interpretation"
        ),
        "restriction": (
            "Still provisional; not an authoritative legal zoning dataset"
        ),
    },
    {
        "dataset_id": "UDA_PORTABLE_ZIP",
        "dataset_name": (
            "UDA zoning portable review package"
        ),
        "path": UDA_PORTABLE_ZIP,
        "data_type": "PACKAGE",
        "role": "DOCUMENTATION_PACKAGE",
        "usage_status": "CONTEXT_ONLY",
        "allowed_use": (
            "Traceability and research review"
        ),
        "restriction": (
            "Package itself is not used directly as a suitability criterion"
        ),
    },
    {
        "dataset_id": "UDA_QGIS_PROJECT",
        "dataset_name": (
            "UDA zoning portable QGIS review project"
        ),
        "path": UDA_QGIS_PROJECT,
        "data_type": "QGIS_PROJECT",
        "role": "VISUAL_REVIEW",
        "usage_status": "CONTEXT_ONLY",
        "allowed_use": (
            "Visual research review only"
        ),
        "restriction": (
            "Not used directly in GIS-MCDA scoring"
        ),
    },
    {
        "dataset_id": "UDA_VISUAL_SCREENSHOT",
        "dataset_name": (
            "UDA 2024 georeferenced screenshot"
        ),
        "path": UDA_VISUAL_TIF,
        "data_type": "RASTER",
        "role": "VISUAL_REFERENCE",
        "usage_status": "CONTEXT_ONLY",
        "allowed_use": (
            "Visual comparison only"
        ),
        "restriction": (
            "VISUAL_ONLY source; cannot be used as quantitative "
            "or legal GIS evidence"
        ),
    },
]


# =============================================================================
# 8. START
# =============================================================================

print("=" * 100)
print("STEP 16A - DEVELOPMENT SUITABILITY CRITERIA AND DATA READINESS AUDIT")
print("=" * 100)


# =============================================================================
# 9. BUILD DATASET INVENTORY
# =============================================================================

inventory_records = []


for dataset in DATASET_REGISTRY:

    path = dataset["path"]

    record = {
        "dataset_id": dataset["dataset_id"],
        "dataset_name": dataset["dataset_name"],
        "relative_path": relative_path(path),
        "data_type": dataset["data_type"],
        "role": dataset["role"],
        "usage_status": dataset["usage_status"],
        "allowed_use": dataset["allowed_use"],
        "restriction": dataset["restriction"],
        "file_exists": path.exists(),
        "file_size_mb": file_size_mb(path),
    }

    inventory_records.append(record)


inventory_df = pd.DataFrame(
    inventory_records
)


missing_files = (
    inventory_df[
        ~inventory_df[
            "file_exists"
        ]
    ][
        [
            "dataset_id",
            "relative_path",
        ]
    ]
)


if len(missing_files) > 0:

    print("\nWARNING - MISSING FILES")

    print(
        missing_files.to_string(
            index=False
        )
    )


# =============================================================================
# 10. LOAD PRIMARY GRID
# =============================================================================

if not GRID_FILE.exists():

    raise FileNotFoundError(
        f"Primary 100 m grid not found:\n{GRID_FILE}"
    )


grid_layers = pyogrio.list_layers(
    GRID_FILE
)


if len(grid_layers) == 0:

    raise RuntimeError(
        "No layers found in the 100 m grid GeoPackage."
    )


grid_layer = str(
    grid_layers[
        0,
        0
    ]
)


grid_gdf = gpd.read_file(
    GRID_FILE,
    layer=grid_layer,
)


if grid_gdf.empty:

    raise RuntimeError(
        "Primary 100 m grid is empty."
    )


grid_crs = grid_gdf.crs

grid_epsg = safe_epsg(
    grid_crs
)

grid_bounds = tuple(
    grid_gdf.total_bounds
)


print("\nPRIMARY ANALYSIS GRID")

print(
    f"Layer                 : "
    f"{grid_layer}"
)

print(
    f"Grid cells            : "
    f"{len(grid_gdf):,}"
)

print(
    f"CRS                   : "
    f"{crs_to_text(grid_crs)}"
)

print(
    f"Expected EPSG         : "
    f"{EXPECTED_ANALYSIS_EPSG}"
)


# =============================================================================
# 11. VECTOR AUDIT FUNCTION
# =============================================================================

def audit_vector(
    dataset_id,
    path,
    usage_status,
):
    """
    Audit all layers in a vector dataset.
    """

    records = []

    if not path.exists():

        records.append(
            {
                "dataset_id": dataset_id,
                "layer_name": None,
                "usage_status": usage_status,
                "feature_count": np.nan,
                "geometry_type": None,
                "crs": None,
                "epsg": np.nan,
                "same_crs_as_grid": False,
                "empty_geometry_count": np.nan,
                "invalid_geometry_count": np.nan,
                "spatially_overlaps_grid": False,
                "status": "MISSING",
            }
        )

        return records


    try:

        layers = pyogrio.list_layers(
            path
        )

    except Exception as exc:

        records.append(
            {
                "dataset_id": dataset_id,
                "layer_name": None,
                "usage_status": usage_status,
                "feature_count": np.nan,
                "geometry_type": None,
                "crs": None,
                "epsg": np.nan,
                "same_crs_as_grid": False,
                "empty_geometry_count": np.nan,
                "invalid_geometry_count": np.nan,
                "spatially_overlaps_grid": False,
                "status": f"LAYER_READ_ERROR: {exc}",
            }
        )

        return records


    for layer_entry in layers:

        layer_name = str(
            layer_entry[
                0
            ]
        )


        try:

            gdf = gpd.read_file(
                path,
                layer=layer_name,
            )


            feature_count = len(
                gdf
            )


            layer_crs = gdf.crs

            layer_epsg = safe_epsg(
                layer_crs
            )


            same_crs = (
                layer_epsg
                ==
                grid_epsg
            )


            if feature_count > 0:

                empty_count = int(
                    gdf.geometry.is_empty.sum()
                )


                invalid_count = int(
                    (
                        ~gdf.geometry.is_valid
                    ).sum()
                )


                geometry_types = sorted(
                    gdf.geometry.geom_type
                    .dropna()
                    .unique()
                    .tolist()
                )


                geometry_type = (
                    ";".join(
                        geometry_types
                    )
                )


                # -------------------------------------------------------------
                # Spatial overlap with grid
                # -------------------------------------------------------------

                overlap = False

                if layer_crs is not None:

                    try:

                        if layer_epsg != grid_epsg:

                            temp = gdf.to_crs(
                                grid_crs
                            )

                            layer_bounds = tuple(
                                temp.total_bounds
                            )

                        else:

                            layer_bounds = tuple(
                                gdf.total_bounds
                            )


                        overlap = bbox_intersects(
                            layer_bounds,
                            grid_bounds,
                        )

                    except Exception:

                        overlap = False


            else:

                empty_count = 0
                invalid_count = 0
                geometry_type = None
                overlap = False


            status = (
                "PASS"
                if (
                    feature_count > 0
                    and
                    layer_crs is not None
                    and
                    overlap
                )
                else
                "REVIEW_REQUIRED"
            )


            records.append(
                {
                    "dataset_id": dataset_id,
                    "layer_name": layer_name,
                    "usage_status": usage_status,
                    "feature_count": feature_count,
                    "geometry_type": geometry_type,
                    "crs": crs_to_text(
                        layer_crs
                    ),
                    "epsg": layer_epsg,
                    "same_crs_as_grid": same_crs,
                    "empty_geometry_count": empty_count,
                    "invalid_geometry_count": invalid_count,
                    "spatially_overlaps_grid": overlap,
                    "status": status,
                }
            )


        except Exception as exc:

            records.append(
                {
                    "dataset_id": dataset_id,
                    "layer_name": layer_name,
                    "usage_status": usage_status,
                    "feature_count": np.nan,
                    "geometry_type": None,
                    "crs": None,
                    "epsg": np.nan,
                    "same_crs_as_grid": False,
                    "empty_geometry_count": np.nan,
                    "invalid_geometry_count": np.nan,
                    "spatially_overlaps_grid": False,
                    "status": f"READ_ERROR: {exc}",
                }
            )


    return records


# =============================================================================
# 12. RUN VECTOR AUDITS
# =============================================================================

vector_registry = [
    (
        "GRID_100M",
        GRID_FILE,
        "CORE",
    ),
    (
        "OSM_ROADS_2026",
        ROADS_FILE,
        "CORE_CANDIDATE",
    ),
    (
        "OSM_BUILDINGS_2026",
        BUILDINGS_FILE,
        "CORE_CANDIDATE",
    ),
    (
        "UDA_HIGH_LAND_APPROX",
        UDA_HIGH_LAND_APPROX,
        "CONTEXT_ONLY",
    ),
    (
        "UDA_CONSERVATION_REVIEW",
        UDA_CONSERVATION_REVIEW,
        "CONTEXT_ONLY",
    ),
    (
        "UDA_AUDITED_PROVISIONAL",
        UDA_AUDITED_PROVISIONAL,
        "CONTEXT_ONLY",
    ),
]


vector_records = []


for dataset_id, path, usage_status in vector_registry:

    vector_records.extend(
        audit_vector(
            dataset_id=dataset_id,
            path=path,
            usage_status=usage_status,
        )
    )


vector_audit_df = pd.DataFrame(
    vector_records
)


# =============================================================================
# 13. RASTER AUDIT FUNCTION
# =============================================================================

def audit_raster(
    dataset_id,
    path,
    usage_status,
):
    """
    Audit raster CRS, dimensions, resolution and grid overlap.
    """

    if not path.exists():

        return {
            "dataset_id": dataset_id,
            "usage_status": usage_status,
            "width": np.nan,
            "height": np.nan,
            "band_count": np.nan,
            "crs": None,
            "epsg": np.nan,
            "same_crs_as_grid": False,
            "pixel_width": np.nan,
            "pixel_height": np.nan,
            "nodata": np.nan,
            "spatially_overlaps_grid": False,
            "status": "MISSING",
        }


    try:

        with rasterio.open(
            path
        ) as src:

            raster_crs = src.crs

            raster_epsg = (
                raster_crs.to_epsg()
                if raster_crs is not None
                else None
            )


            raster_bounds = (
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            )


            if raster_crs is not None:

                try:

                    raster_bounds_gdf = gpd.GeoDataFrame(
                        geometry=[
                            box(
                                *raster_bounds
                            )
                        ],
                        crs=raster_crs,
                    )


                    if raster_epsg != grid_epsg:

                        raster_bounds_gdf = (
                            raster_bounds_gdf.to_crs(
                                grid_crs
                            )
                        )


                    rb = tuple(
                        raster_bounds_gdf.total_bounds
                    )


                    overlap = bbox_intersects(
                        rb,
                        grid_bounds,
                    )

                except Exception:

                    overlap = False

            else:

                overlap = False


            same_crs = (
                raster_epsg
                ==
                grid_epsg
            )


            status = (
                "PASS"
                if (
                    raster_crs is not None
                    and
                    src.width > 0
                    and
                    src.height > 0
                    and
                    overlap
                )
                else
                "REVIEW_REQUIRED"
            )


            return {
                "dataset_id": dataset_id,
                "usage_status": usage_status,
                "width": src.width,
                "height": src.height,
                "band_count": src.count,
                "crs": crs_to_text(
                    raster_crs
                ),
                "epsg": raster_epsg,
                "same_crs_as_grid": same_crs,
                "pixel_width": abs(
                    src.transform.a
                ),
                "pixel_height": abs(
                    src.transform.e
                ),
                "nodata": src.nodata,
                "spatially_overlaps_grid": overlap,
                "status": status,
            }


    except Exception as exc:

        return {
            "dataset_id": dataset_id,
            "usage_status": usage_status,
            "width": np.nan,
            "height": np.nan,
            "band_count": np.nan,
            "crs": None,
            "epsg": np.nan,
            "same_crs_as_grid": False,
            "pixel_width": np.nan,
            "pixel_height": np.nan,
            "nodata": np.nan,
            "spatially_overlaps_grid": False,
            "status": f"READ_ERROR: {exc}",
        }


# =============================================================================
# 14. RUN RASTER AUDITS
# =============================================================================

raster_registry = [
    (
        "GEDTM30",
        GEDTM_FILE,
        "CORE_CANDIDATE",
    ),
    (
        "NASADEM",
        NASADEM_FILE,
        "SENSITIVITY_REFERENCE",
    ),
    (
        "ESA_WORLDCOVER_2021",
        WORLDCOVER_FILE,
        "REFERENCE_CONTEXT",
    ),
    (
        "UDA_VISUAL_SCREENSHOT",
        UDA_VISUAL_TIF,
        "CONTEXT_ONLY",
    ),
]


raster_records = []


for dataset_id, path, usage_status in raster_registry:

    raster_records.append(
        audit_raster(
            dataset_id=dataset_id,
            path=path,
            usage_status=usage_status,
        )
    )


raster_audit_df = pd.DataFrame(
    raster_records
)


# =============================================================================
# 15. CRITERIA REGISTRY
# =============================================================================

criteria_records = [
    {
        "criterion_id": "C1",
        "criterion_group": "Accessibility",
        "criterion_name": "Road accessibility",
        "source_dataset": "OSM_ROADS_2026",
        "candidate_measure": (
            "Distance from each 100 m grid cell to selected road classes"
        ),
        "criterion_status": "CANDIDATE_PENDING_DEFINITION",
        "expected_role": "BENEFIT_OR_COST_TO_BE_JUSTIFIED",
        "weight_assigned_now": False,
        "notes": (
            "Road hierarchy/class filtering must be inspected before "
            "distance criterion is created."
        ),
    },
    {
        "criterion_id": "C2",
        "criterion_group": "Accessibility",
        "criterion_name": "Road-network context",
        "source_dataset": "OSM_ROADS_2026",
        "candidate_measure": (
            "Road length/density around each 100 m grid cell"
        ),
        "criterion_status": "CANDIDATE_PENDING_DEFINITION",
        "expected_role": "BENEFIT_TO_BE_JUSTIFIED",
        "weight_assigned_now": False,
        "notes": (
            "Neighbourhood radius and road classes must be defined "
            "before calculation."
        ),
    },
    {
        "criterion_id": "C3",
        "criterion_group": "Existing Development Context",
        "criterion_name": "Existing built context",
        "source_dataset": "OSM_BUILDINGS_2026",
        "candidate_measure": (
            "Building footprint/density surrounding each 100 m grid cell"
        ),
        "criterion_status": "CANDIDATE_PENDING_DEFINITION",
        "expected_role": "BENEFIT_OR_SCREENING_TO_BE_JUSTIFIED",
        "weight_assigned_now": False,
        "notes": (
            "Used only for current development context, not as "
            "historical pressure evidence."
        ),
    },
    {
        "criterion_id": "C4",
        "criterion_group": "Terrain",
        "criterion_name": "Slope",
        "source_dataset": "GEDTM30",
        "candidate_measure": (
            "Mean or representative slope within each 100 m grid cell"
        ),
        "criterion_status": "CORE_CANDIDATE",
        "expected_role": "COST",
        "weight_assigned_now": False,
        "notes": (
            "Lower slopes are generally more physically suitable, "
            "but final standardization thresholds require justification."
        ),
    },
    {
        "criterion_id": "C5",
        "criterion_group": "Terrain",
        "criterion_name": "Elevation",
        "source_dataset": "GEDTM30",
        "candidate_measure": (
            "Mean elevation within each 100 m grid cell"
        ),
        "criterion_status": "CANDIDATE_PENDING_JUSTIFICATION",
        "expected_role": "DIRECTION_TO_BE_JUSTIFIED",
        "weight_assigned_now": False,
        "notes": (
            "Do not assume higher or lower elevation is automatically "
            "more suitable without planning/environmental justification."
        ),
    },
    {
        "criterion_id": "C6",
        "criterion_group": "Land-cover Context",
        "criterion_name": "Current/reference land-cover screening",
        "source_dataset": "ESA_WORLDCOVER_2021",
        "candidate_measure": (
            "Reference land-cover class or land-cover proportion"
        ),
        "criterion_status": "REFERENCE_OR_SCREENING_CANDIDATE",
        "expected_role": "SCREENING_OR_CONTEXT",
        "weight_assigned_now": False,
        "notes": (
            "WorldCover 2021 is an independent reference, not current "
            "2026 ground truth."
        ),
    },
]


criteria_df = pd.DataFrame(
    criteria_records
)


# =============================================================================
# 16. EXCLUSION / SEPARATION REGISTRY
# =============================================================================

exclusion_records = [
    {
        "source_group": "Urbanization Pressure",
        "examples": (
            "2026-2028 pressure score, pressure category, threshold flag"
        ),
        "allowed_in_core_suitability_score": False,
        "reason": (
            "Urbanization Pressure represents what may happen. "
            "Development Suitability represents what is appropriate. "
            "They must remain analytically independent."
        ),
        "later_use": (
            "Compare with suitability after independent suitability map "
            "is completed."
        ),
    },
    {
        "source_group": "SHAP / XAI",
        "examples": (
            "SHAP feature values, dominant explanation domains"
        ),
        "allowed_in_core_suitability_score": False,
        "reason": (
            "SHAP explains the pressure model and would indirectly "
            "reintroduce pressure-model information into suitability."
        ),
        "later_use": (
            "Dashboard explanation and interpretation only."
        ),
    },
    {
        "source_group": "Historical/Future Conservation Priority",
        "examples": (
            "Green importance, persistence, connectivity, conservation score"
        ),
        "allowed_in_core_suitability_score": False,
        "reason": (
            "Conservation Intelligence is a later independent synthesis stage."
        ),
        "later_use": (
            "Compare after suitability and pressure outputs are frozen."
        ),
    },
    {
        "source_group": "Provisional UDA zoning",
        "examples": (
            "Approximate/provisional zoning and georeferenced screenshot"
        ),
        "allowed_in_core_suitability_score": False,
        "reason": (
            "Current package is provisional, approximate and/or visual-only; "
            "legal authority is unverified."
        ),
        "later_use": (
            "Planning-context overlay, sensitivity discussion and later "
            "forecast-planning conflict interpretation."
        ),
    },
]


exclusion_df = pd.DataFrame(
    exclusion_records
)


# =============================================================================
# 17. QA CHECKS
# =============================================================================

core_required_ids = [
    "GRID_100M",
    "OSM_ROADS_2026",
    "OSM_BUILDINGS_2026",
    "GEDTM30",
]


core_inventory = (
    inventory_df[
        inventory_df[
            "dataset_id"
        ]
        .isin(
            core_required_ids
        )
    ]
)


core_files_exist = bool(
    core_inventory[
        "file_exists"
    ].all()
)


grid_cell_count_valid = (
    len(
        grid_gdf
    )
    ==
    EXPECTED_GRID_CELLS
)


grid_crs_valid = (
    grid_epsg
    ==
    EXPECTED_ANALYSIS_EPSG
)


core_vector_ids = [
    "GRID_100M",
    "OSM_ROADS_2026",
    "OSM_BUILDINGS_2026",
]


core_vector_audit = (
    vector_audit_df[
        vector_audit_df[
            "dataset_id"
        ]
        .isin(
            core_vector_ids
        )
    ]
)


core_vectors_readable = bool(
    (
        core_vector_audit[
            "status"
        ]
        ==
        "PASS"
    )
    .all()
)


gedtm_row = (
    raster_audit_df[
        raster_audit_df[
            "dataset_id"
        ]
        ==
        "GEDTM30"
    ]
)


gedtm_readable = bool(
    len(
        gedtm_row
    )
    ==
    1
    and
    gedtm_row.iloc[
        0
    ][
        "status"
    ]
    ==
    "PASS"
)


no_ahp_weights_assigned = bool(
    (
        criteria_df[
            "weight_assigned_now"
        ]
        ==
        False
    )
    .all()
)


pressure_excluded = bool(
    exclusion_df.loc[
        exclusion_df[
            "source_group"
        ]
        ==
        "Urbanization Pressure",
        "allowed_in_core_suitability_score",
    ]
    .eq(
        False
    )
    .all()
)


shap_excluded = bool(
    exclusion_df.loc[
        exclusion_df[
            "source_group"
        ]
        ==
        "SHAP / XAI",
        "allowed_in_core_suitability_score",
    ]
    .eq(
        False
    )
    .all()
)


conservation_excluded = bool(
    exclusion_df.loc[
        exclusion_df[
            "source_group"
        ]
        ==
        "Historical/Future Conservation Priority",
        "allowed_in_core_suitability_score",
    ]
    .eq(
        False
    )
    .all()
)


uda_core_excluded = bool(
    (
        inventory_df.loc[
            inventory_df[
                "dataset_id"
            ]
            .str.startswith(
                "UDA_"
            ),
            "usage_status",
        ]
        ==
        "CONTEXT_ONLY"
    )
    .all()
)


overall_status = (
    "PASS"
    if all(
        [
            core_files_exist,
            grid_cell_count_valid,
            grid_crs_valid,
            core_vectors_readable,
            gedtm_readable,
            no_ahp_weights_assigned,
            pressure_excluded,
            shap_excluded,
            conservation_excluded,
            uda_core_excluded,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


qa_df = pd.DataFrame(
    [
        {
            "core_files_exist": (
                core_files_exist
            ),
            "grid_cell_count": (
                len(
                    grid_gdf
                )
            ),
            "grid_cell_count_valid": (
                grid_cell_count_valid
            ),
            "grid_epsg": (
                grid_epsg
            ),
            "grid_crs_valid": (
                grid_crs_valid
            ),
            "core_vectors_readable_and_overlapping": (
                core_vectors_readable
            ),
            "gedtm_primary_dem_readable": (
                gedtm_readable
            ),
            "ahp_weights_assigned_in_step16a": (
                not no_ahp_weights_assigned
            ),
            "pressure_outputs_excluded": (
                pressure_excluded
            ),
            "shap_outputs_excluded": (
                shap_excluded
            ),
            "conservation_outputs_excluded": (
                conservation_excluded
            ),
            "uda_provisional_excluded_from_core_score": (
                uda_core_excluded
            ),
            "overall_step16a_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 18. SUMMARY
# =============================================================================

summary = {
    "stage": (
        "Development Suitability Criteria and Data Readiness Audit"
    ),
    "analysis_grid": {
        "cells": int(
            len(
                grid_gdf
            )
        ),
        "crs": crs_to_text(
            grid_crs
        ),
        "epsg": (
            int(
                grid_epsg
            )
            if grid_epsg is not None
            else None
        ),
    },
    "primary_core_candidate_sources": [
        "OSM roads 2026",
        "OSM buildings 2026",
        "GEDTM30 terrain",
    ],
    "sensitivity_reference_sources": [
        "NASADEM",
    ],
    "reference_context_sources": [
        "ESA WorldCover 2021",
    ],
    "planning_context_only_sources": [
        "UDA approximate high-land conservation zones",
        "UDA provisional conservation review",
        "UDA 2024 audited provisional GIS",
        "UDA portable review package",
        "UDA QGIS review project",
        "UDA visual-only georeferenced screenshot",
    ],
    "explicitly_excluded_from_core_suitability_score": [
        "Urbanization Pressure outputs",
        "SHAP/XAI outputs",
        "Conservation Priority outputs",
        "Provisional UDA zoning",
    ],
    "ahp_weights_assigned": False,
    "suitability_score_calculated": False,
    "overall_step16a_status": (
        overall_status
    ),
}


# =============================================================================
# 19. SAVE OUTPUTS
# =============================================================================

inventory_df.to_csv(
    INVENTORY_CSV,
    index=False,
)

vector_audit_df.to_csv(
    VECTOR_AUDIT_CSV,
    index=False,
)

raster_audit_df.to_csv(
    RASTER_AUDIT_CSV,
    index=False,
)

criteria_df.to_csv(
    CRITERIA_REGISTRY_CSV,
    index=False,
)

exclusion_df.to_csv(
    EXCLUSION_REGISTRY_CSV,
    index=False,
)

qa_df.to_csv(
    QA_CSV,
    index=False,
)

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 20. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("DATASET INVENTORY")
print("=" * 100)

print(
    inventory_df[
        [
            "dataset_id",
            "data_type",
            "usage_status",
            "file_exists",
        ]
    ]
    .to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("VECTOR AUDIT")
print("=" * 100)

print(
    vector_audit_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("RASTER AUDIT")
print("=" * 100)

print(
    raster_audit_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("SUITABILITY CRITERIA REGISTRY")
print("=" * 100)

print(
    criteria_df[
        [
            "criterion_id",
            "criterion_group",
            "criterion_name",
            "source_dataset",
            "criterion_status",
            "weight_assigned_now",
        ]
    ]
    .to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("EXCLUSION / SEPARATION REGISTRY")
print("=" * 100)

print(
    exclusion_df[
        [
            "source_group",
            "allowed_in_core_suitability_score",
            "later_use",
        ]
    ]
    .to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 16A QA")
print("=" * 100)

print(
    f"Core files exist                         : "
    f"{core_files_exist}"
)

print(
    f"100 m grid cell count = 9,240            : "
    f"{grid_cell_count_valid}"
)

print(
    f"Analysis CRS = EPSG:32644                 : "
    f"{grid_crs_valid}"
)

print(
    f"Core vectors readable and overlapping     : "
    f"{core_vectors_readable}"
)

print(
    f"Primary GEDTM30 readable                  : "
    f"{gedtm_readable}"
)

print(
    f"AHP weights assigned                      : "
    f"{not no_ahp_weights_assigned}"
)

print(
    f"Pressure outputs excluded                 : "
    f"{pressure_excluded}"
)

print(
    f"SHAP outputs excluded                     : "
    f"{shap_excluded}"
)

print(
    f"Conservation outputs excluded             : "
    f"{conservation_excluded}"
)

print(
    f"Provisional UDA excluded from core score  : "
    f"{uda_core_excluded}"
)

print(
    f"Overall Step 16A status                   : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

for output_path in [
    INVENTORY_CSV,
    VECTOR_AUDIT_CSV,
    RASTER_AUDIT_CSV,
    CRITERIA_REGISTRY_CSV,
    EXCLUSION_REGISTRY_CSV,
    QA_CSV,
    SUMMARY_JSON,
]:

    print(
        f" - {relative_path(output_path)}"
    )


print("\nIMPORTANT:")

print(
    "1. Step 16A only audits Development Suitability inputs."
)

print(
    "2. No suitability score has been calculated."
)

print(
    "3. No AHP weights have been assigned."
)

print(
    "4. OSM 2026 roads/buildings are allowed here because this is a current planning-context assessment."
)

print(
    "5. Urbanization Pressure and SHAP remain analytically separate from Development Suitability."
)

print(
    "6. GEDTM30 is the primary terrain source; NASADEM is retained for sensitivity analysis."
)

print(
    "7. ESA WorldCover 2021 is reference/context information, not current ground truth."
)

print(
    "8. All UDA zoning files remain context-only because the current package is provisional/approximate/visual-only."
)

print(
    "9. Step 16B will inspect the actual OSM road attributes and create the raw 100 m suitability criteria only after Step 16A passes."
)

print("=" * 100)