"""
STEP 02 - SPATIAL DATA VALIDATION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Validate geometry quality of key vector datasets.
2. Check spatial coverage and alignment with the KMC study area.
3. Compare KMC and GN boundary areas.
4. Check raster coverage over the study area.
5. Validate the provisional UDA zoning coverage.

IMPORTANT:
This script is READ-ONLY for source datasets.
It does not modify or overwrite any raw research data.
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# Main study-area boundary
KMC_BOUNDARY = (
    RAW_DIR
    / "boundaries"
    / "kmc"
    / "kmc boundry.kml"
)

# Processed 57-GN working layer
GN_DIVISIONS = (
    PROCESSED_DIR
    / "boundaries"
    / "gn_divisions"
    / "kaduwela_gn_divisions.shp"
)

# Terrain
GEDTM30 = (
    RAW_DIR
    / "dem"
    / "gedtm30"
    / "GEDTM30_Kaduwela_Buffer.tif"
)

NASADEM = (
    RAW_DIR
    / "dem"
    / "nasadem"
    / "NASADEM_Kaduwela_Buffer.tif"
)

# Reference land cover
WORLD_COVER = (
    RAW_DIR
    / "reference_lulc"
    / "esa_worldcover_2021"
    / "Kaduwela_ESA_WorldCover_2021.tif"
)

# Landsat directory
LANDSAT_DIR = (
    RAW_DIR
    / "landsat"
    / "annual_composites"
)

# UDA provisional zoning folder
UDA_DIR = (
    RAW_DIR
    / "uda"
    / "zoning_provisional"
)

# Standard projected CRS used for metric measurements
ANALYSIS_CRS = "EPSG:32644"


# =============================================================================
# 2. BASIC OUTPUT
# =============================================================================

print("=" * 100)
print("STEP 02 - SPATIAL DATA VALIDATION")
print("=" * 100)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Analysis CRS : {ANALYSIS_CRS}")

# =============================================================================
# 3. VECTOR VALIDATION FUNCTIONS
# =============================================================================

def load_vector(path: Path, dataset_name: str) -> gpd.GeoDataFrame:
    """
    Load a vector dataset and perform basic checks.

    This function does not modify the source file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} not found:\n{path}"
        )

    gdf = gpd.read_file(path)

    print("\n" + "-" * 100)
    print(f"LOADING VECTOR: {dataset_name}")
    print("-" * 100)

    print(f"Path            : {path.relative_to(PROJECT_ROOT)}")
    print(f"Feature count   : {len(gdf)}")
    print(f"CRS             : {gdf.crs}")
    print(f"Columns         : {list(gdf.columns)}")

    if "geometry" not in gdf.columns:
        raise ValueError(
            f"{dataset_name} does not contain a geometry column."
        )

    return gdf


def geometry_validation_summary(
    gdf: gpd.GeoDataFrame,
    dataset_name: str
) -> dict:
    """
    Calculate geometry-quality indicators.
    """

    total_features = len(gdf)

    null_geometry_count = int(
        gdf.geometry.isna().sum()
    )

    empty_geometry_count = int(
        gdf.geometry.is_empty.sum()
    )

    valid_geometry_count = int(
        gdf.geometry.is_valid.sum()
    )

    invalid_geometry_count = (
        total_features - valid_geometry_count
    )

    duplicate_geometry_count = int(
        gdf.geometry.duplicated().sum()
    )

    geometry_types = (
        gdf.geometry.geom_type
        .value_counts(dropna=False)
        .to_dict()
    )

    return {
        "dataset_name": dataset_name,
        "feature_count": total_features,
        "null_geometries": null_geometry_count,
        "empty_geometries": empty_geometry_count,
        "valid_geometries": valid_geometry_count,
        "invalid_geometries": invalid_geometry_count,
        "duplicate_geometries": duplicate_geometry_count,
        "geometry_types": str(geometry_types),
        "source_crs": str(gdf.crs),
    }


def calculate_area_km2(
    gdf: gpd.GeoDataFrame
) -> float:
    """
    Calculate total dataset area in square kilometres
    using the common projected analysis CRS.
    """

    if gdf.crs is None:
        raise ValueError(
            "Cannot calculate area because CRS is missing."
        )

    projected = gdf.to_crs(ANALYSIS_CRS)

    total_area_m2 = projected.geometry.area.sum()

    return float(total_area_m2 / 1_000_000)


def union_area_km2(
    gdf: gpd.GeoDataFrame
) -> float:
    """
    Calculate union area in square kilometres.

    Union area avoids double-counting overlapping polygons.
    """

    if gdf.crs is None:
        raise ValueError(
            "Cannot calculate union area because CRS is missing."
        )

    projected = gdf.to_crs(ANALYSIS_CRS)

    union_geometry = projected.geometry.union_all()

    return float(union_geometry.area / 1_000_000)

# =============================================================================
# 4. KMC BOUNDARY AND GN DIVISION VALIDATION
# =============================================================================

print("\n" + "=" * 100)
print("KMC BOUNDARY AND GN DIVISION VALIDATION")
print("=" * 100)

# Load datasets
kmc_gdf = load_vector(
    KMC_BOUNDARY,
    "KMC Boundary"
)

gn_gdf = load_vector(
    GN_DIVISIONS,
    "Kaduwela GN Divisions Processed"
)


# -------------------------------------------------------------------------
# Geometry-quality summaries
# -------------------------------------------------------------------------

vector_validation_records = []

vector_validation_records.append(
    geometry_validation_summary(
        kmc_gdf,
        "KMC Boundary"
    )
)

vector_validation_records.append(
    geometry_validation_summary(
        gn_gdf,
        "Kaduwela GN Divisions Processed"
    )
)


# -------------------------------------------------------------------------
# Area calculations
# -------------------------------------------------------------------------

kmc_total_area_km2 = calculate_area_km2(kmc_gdf)
kmc_union_area_km2 = union_area_km2(kmc_gdf)

gn_total_area_km2 = calculate_area_km2(gn_gdf)
gn_union_area_km2 = union_area_km2(gn_gdf)


print("\nAREA CHECK")

print(
    f"KMC total area       : "
    f"{kmc_total_area_km2:.4f} km²"
)

print(
    f"KMC union area       : "
    f"{kmc_union_area_km2:.4f} km²"
)

print(
    f"GN summed area       : "
    f"{gn_total_area_km2:.4f} km²"
)

print(
    f"GN union area        : "
    f"{gn_union_area_km2:.4f} km²"
)


# -------------------------------------------------------------------------
# GN coverage relative to KMC boundary
# -------------------------------------------------------------------------

kmc_projected = kmc_gdf.to_crs(ANALYSIS_CRS)
gn_projected = gn_gdf.to_crs(ANALYSIS_CRS)

kmc_union = kmc_projected.geometry.union_all()
gn_union = gn_projected.geometry.union_all()

gn_inside_kmc = gn_union.intersection(kmc_union)

gn_inside_kmc_area_km2 = (
    gn_inside_kmc.area / 1_000_000
)

gn_coverage_percent = (
    gn_inside_kmc_area_km2
    / kmc_union_area_km2
    * 100
)


# Area extending outside KMC
gn_outside_kmc = gn_union.difference(kmc_union)

gn_outside_kmc_area_km2 = (
    gn_outside_kmc.area / 1_000_000
)


# KMC area not covered by GN divisions
kmc_not_covered = kmc_union.difference(gn_union)

kmc_not_covered_area_km2 = (
    kmc_not_covered.area / 1_000_000
)


print("\nGN vs KMC COVERAGE")

print(
    f"GN area inside KMC         : "
    f"{gn_inside_kmc_area_km2:.4f} km²"
)

print(
    f"GN coverage of KMC         : "
    f"{gn_coverage_percent:.4f} %"
)

print(
    f"GN area outside KMC        : "
    f"{gn_outside_kmc_area_km2:.6f} km²"
)

print(
    f"KMC area not covered by GN : "
    f"{kmc_not_covered_area_km2:.6f} km²"
)


# -------------------------------------------------------------------------
# Overlap check between GN polygons
# -------------------------------------------------------------------------

gn_sum_area_m2 = gn_projected.geometry.area.sum()
gn_union_area_m2 = gn_union.area

gn_internal_overlap_area_m2 = max(
    0.0,
    gn_sum_area_m2 - gn_union_area_m2
)

gn_internal_overlap_area_km2 = (
    gn_internal_overlap_area_m2 / 1_000_000
)


print("\nGN INTERNAL OVERLAP CHECK")

print(
    f"GN summed polygon area : "
    f"{gn_sum_area_m2 / 1_000_000:.4f} km²"
)

print(
    f"GN union area          : "
    f"{gn_union_area_m2 / 1_000_000:.4f} km²"
)

print(
    f"Estimated overlap area : "
    f"{gn_internal_overlap_area_km2:.6f} km²"
)

# =============================================================================
# 5. RASTER SPATIAL COVERAGE VALIDATION
# =============================================================================

print("\n" + "=" * 100)
print("RASTER SPATIAL COVERAGE VALIDATION")
print("=" * 100)


def raster_coverage_summary(
    raster_path: Path,
    raster_name: str,
    study_area_gdf: gpd.GeoDataFrame
) -> dict:
    """
    Check whether a raster spatially covers the KMC study area.

    The raster bounds are converted into a polygon and compared
    with the KMC boundary in the raster CRS.
    """

    if not raster_path.exists():
        return {
            "raster_name": raster_name,
            "path": str(raster_path),
            "status": "MISSING",
            "raster_crs": None,
            "width": None,
            "height": None,
            "resolution_x": None,
            "resolution_y": None,
            "kmc_coverage_percent": None,
            "kmc_area_outside_raster_km2": None,
        }

    try:
        with rasterio.open(raster_path) as src:

            raster_crs = src.crs

            if raster_crs is None:
                return {
                    "raster_name": raster_name,
                    "path": str(raster_path),
                    "status": "CRS_MISSING",
                    "raster_crs": None,
                    "width": src.width,
                    "height": src.height,
                    "resolution_x": src.res[0],
                    "resolution_y": src.res[1],
                    "kmc_coverage_percent": None,
                    "kmc_area_outside_raster_km2": None,
                }

            raster_bounds_polygon = box(
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            )

            study_area_in_raster_crs = (
                study_area_gdf.to_crs(raster_crs)
            )

            study_area_union = (
                study_area_in_raster_crs
                .geometry
                .union_all()
            )

            intersection = (
                study_area_union.intersection(
                    raster_bounds_polygon
                )
            )

            study_area_projected = (
                study_area_gdf.to_crs(ANALYSIS_CRS)
            )

            study_area_area_km2 = (
                study_area_projected
                .geometry
                .union_all()
                .area
                / 1_000_000
            )

            # Reproject intersection for metric area calculation
            intersection_gdf = gpd.GeoDataFrame(
                geometry=[intersection],
                crs=raster_crs,
            ).to_crs(ANALYSIS_CRS)

            intersection_area_km2 = (
                intersection_gdf.geometry.iloc[0].area
                / 1_000_000
            )

            coverage_percent = (
                intersection_area_km2
                / study_area_area_km2
                * 100
            )

            outside_area_km2 = max(
                0.0,
                study_area_area_km2
                - intersection_area_km2
            )

            return {
                "raster_name": raster_name,
                "path": str(
                    raster_path.relative_to(PROJECT_ROOT)
                ),
                "status": "OK",
                "raster_crs": str(raster_crs),
                "width": src.width,
                "height": src.height,
                "resolution_x": src.res[0],
                "resolution_y": src.res[1],
                "kmc_coverage_percent": coverage_percent,
                "kmc_area_outside_raster_km2": outside_area_km2,
            }

    except Exception as error:
        return {
            "raster_name": raster_name,
            "path": str(raster_path),
            "status": f"ERROR: {error}",
            "raster_crs": None,
            "width": None,
            "height": None,
            "resolution_x": None,
            "resolution_y": None,
            "kmc_coverage_percent": None,
            "kmc_area_outside_raster_km2": None,
        }


raster_coverage_records = []


# -------------------------------------------------------------------------
# Landsat coverage
# -------------------------------------------------------------------------

landsat_files = sorted(
    LANDSAT_DIR.glob("Kaduwela_Landsat_*.tif")
)

for landsat_file in landsat_files:

    year = landsat_file.stem.split("_")[-1]

    raster_coverage_records.append(
        raster_coverage_summary(
            landsat_file,
            f"Landsat {year}",
            kmc_gdf,
        )
    )


# -------------------------------------------------------------------------
# DEM coverage
# -------------------------------------------------------------------------

raster_coverage_records.append(
    raster_coverage_summary(
        GEDTM30,
        "GEDTM30",
        kmc_gdf,
    )
)

raster_coverage_records.append(
    raster_coverage_summary(
        NASADEM,
        "NASADEM",
        kmc_gdf,
    )
)


# -------------------------------------------------------------------------
# ESA WorldCover coverage
# -------------------------------------------------------------------------

raster_coverage_records.append(
    raster_coverage_summary(
        WORLD_COVER,
        "ESA WorldCover 2021",
        kmc_gdf,
    )
)


raster_coverage_df = pd.DataFrame(
    raster_coverage_records
)


print(
    raster_coverage_df[
        [
            "raster_name",
            "status",
            "raster_crs",
            "resolution_x",
            "resolution_y",
            "kmc_coverage_percent",
            "kmc_area_outside_raster_km2",
        ]
    ].to_string(index=False)
)

# =============================================================================
# 6. UDA PROVISIONAL ZONING SPATIAL VALIDATION
# =============================================================================

print("\n" + "=" * 100)
print("UDA PROVISIONAL ZONING SPATIAL VALIDATION")
print("=" * 100)


# Find the main UDA provisional GeoPackage wherever it exists
uda_main_files = list(
    UDA_DIR.rglob(
        "Kaduwela_UDA_2024_AUDITED_PROVISIONAL.gpkg"
    )
)

uda_validation_record = {
    "dataset_name": "UDA QA 508 Screening With Flags",
    "feature_count": None,
    "valid_geometries": None,
    "invalid_geometries": None,
    "total_area_km2": None,
    "union_area_km2": None,
    "area_inside_kmc_km2": None,
    "kmc_coverage_percent": None,
    "area_outside_kmc_km2": None,
    "status": "NOT_CHECKED",
}


if len(uda_main_files) == 0:

    print(
        "Main UDA provisional GeoPackage was not found."
    )

    uda_validation_record["status"] = "MISSING"


else:

    uda_main_gpkg = uda_main_files[0]

    print(
        f"UDA GeoPackage found: "
        f"{uda_main_gpkg.relative_to(PROJECT_ROOT)}"
    )

    UDA_QA_LAYER = "QA_508_Screening_With_Flags"

    try:

        uda_zoning_gdf = gpd.read_file(
            uda_main_gpkg,
            layer=UDA_QA_LAYER,
        )

        print(
            f"Layer loaded       : {UDA_QA_LAYER}"
        )

        print(
            f"Feature count      : "
            f"{len(uda_zoning_gdf)}"
        )

        print(
            f"CRS                : "
            f"{uda_zoning_gdf.crs}"
        )

        # -------------------------------------------------------------
        # Geometry validity
        # -------------------------------------------------------------

        uda_geometry_summary = (
            geometry_validation_summary(
                uda_zoning_gdf,
                "UDA QA 508 Screening With Flags",
            )
        )

        vector_validation_records.append(
            uda_geometry_summary
        )

        # -------------------------------------------------------------
        # Area calculations
        # -------------------------------------------------------------

        uda_projected = (
            uda_zoning_gdf.to_crs(ANALYSIS_CRS)
        )

        uda_total_area_km2 = (
            uda_projected.geometry.area.sum()
            / 1_000_000
        )

        uda_union = (
            uda_projected.geometry.union_all()
        )

        uda_union_area_km2 = (
            uda_union.area
            / 1_000_000
        )

        # -------------------------------------------------------------
        # Compare UDA zoning with KMC study boundary
        # -------------------------------------------------------------

        uda_inside_kmc = (
            uda_union.intersection(kmc_union)
        )

        uda_inside_kmc_area_km2 = (
            uda_inside_kmc.area
            / 1_000_000
        )

        uda_coverage_percent = (
            uda_inside_kmc_area_km2
            / kmc_union_area_km2
            * 100
        )

        uda_outside_kmc = (
            uda_union.difference(kmc_union)
        )

        uda_outside_kmc_area_km2 = (
            uda_outside_kmc.area
            / 1_000_000
        )

        print("\nUDA ZONING AREA CHECK")

        print(
            f"Summed polygon area     : "
            f"{uda_total_area_km2:.4f} km²"
        )

        print(
            f"Union zoning area       : "
            f"{uda_union_area_km2:.4f} km²"
        )

        print(
            f"Area inside KMC         : "
            f"{uda_inside_kmc_area_km2:.4f} km²"
        )

        print(
            f"KMC zoning coverage     : "
            f"{uda_coverage_percent:.4f} %"
        )

        print(
            f"Zoning area outside KMC : "
            f"{uda_outside_kmc_area_km2:.6f} km²"
        )

        uda_validation_record.update(
            {
                "feature_count": len(
                    uda_zoning_gdf
                ),
                "valid_geometries": (
                    uda_geometry_summary[
                        "valid_geometries"
                    ]
                ),
                "invalid_geometries": (
                    uda_geometry_summary[
                        "invalid_geometries"
                    ]
                ),
                "total_area_km2": (
                    uda_total_area_km2
                ),
                "union_area_km2": (
                    uda_union_area_km2
                ),
                "area_inside_kmc_km2": (
                    uda_inside_kmc_area_km2
                ),
                "kmc_coverage_percent": (
                    uda_coverage_percent
                ),
                "area_outside_kmc_km2": (
                    uda_outside_kmc_area_km2
                ),
                "status": "OK",
            }
        )

    except Exception as error:

        print(
            f"UDA zoning validation error: "
            f"{error}"
        )

        uda_validation_record["status"] = (
            f"ERROR: {error}"
        )


# =============================================================================
# 7. CREATE VALIDATION TABLES
# =============================================================================

vector_validation_df = pd.DataFrame(
    vector_validation_records
)


boundary_alignment_df = pd.DataFrame(
    [
        {
            "kmc_area_km2": kmc_union_area_km2,
            "gn_summed_area_km2": gn_total_area_km2,
            "gn_union_area_km2": gn_union_area_km2,
            "gn_inside_kmc_area_km2": (
                gn_inside_kmc_area_km2
            ),
            "gn_kmc_coverage_percent": (
                gn_coverage_percent
            ),
            "gn_outside_kmc_area_km2": (
                gn_outside_kmc_area_km2
            ),
            "kmc_not_covered_by_gn_km2": (
                kmc_not_covered_area_km2
            ),
            "gn_internal_overlap_km2": (
                gn_internal_overlap_area_km2
            ),
        }
    ]
)


uda_validation_df = pd.DataFrame(
    [uda_validation_record]
)


# =============================================================================
# 8. SAVE STEP 02 REPORTS
# =============================================================================

vector_report = (
    REPORT_DIR
    / "02_vector_geometry_validation.csv"
)

boundary_report = (
    REPORT_DIR
    / "02_kmc_gn_alignment_validation.csv"
)

raster_report = (
    REPORT_DIR
    / "02_raster_coverage_validation.csv"
)

uda_report = (
    REPORT_DIR
    / "02_uda_zoning_spatial_validation.csv"
)


vector_validation_df.to_csv(
    vector_report,
    index=False,
)

boundary_alignment_df.to_csv(
    boundary_report,
    index=False,
)

raster_coverage_df.to_csv(
    raster_report,
    index=False,
)

uda_validation_df.to_csv(
    uda_report,
    index=False,
)


# =============================================================================
# 9. FINAL STEP 02 SUMMARY
# =============================================================================

vector_invalid_total = int(
    vector_validation_df[
        "invalid_geometries"
    ].fillna(0).sum()
)

raster_errors = raster_coverage_df[
    raster_coverage_df["status"] != "OK"
]

minimum_raster_coverage = (
    raster_coverage_df[
        "kmc_coverage_percent"
    ].min()
)


print("\n" + "=" * 100)
print("STEP 02 SUMMARY")
print("=" * 100)

print(
    f"KMC area                     : "
    f"{kmc_union_area_km2:.4f} km²"
)

print(
    f"GN feature count             : "
    f"{len(gn_gdf)}"
)

print(
    f"GN coverage of KMC           : "
    f"{gn_coverage_percent:.4f} %"
)

print(
    f"GN area outside KMC          : "
    f"{gn_outside_kmc_area_km2:.6f} km²"
)

print(
    f"GN internal overlap          : "
    f"{gn_internal_overlap_area_km2:.6f} km²"
)

print(
    f"Invalid vector geometries    : "
    f"{vector_invalid_total}"
)

print(
    f"Raster coverage errors       : "
    f"{len(raster_errors)}"
)

print(
    f"Minimum raster KMC coverage  : "
    f"{minimum_raster_coverage:.4f} %"
)

print(
    f"UDA zoning validation status : "
    f"{uda_validation_record['status']}"
)

if (
    uda_validation_record[
        "kmc_coverage_percent"
    ]
    is not None
):

    print(
        f"UDA zoning KMC coverage      : "
        f"{uda_validation_record['kmc_coverage_percent']:.4f} %"
    )


print("\nReports created:")

print(
    f" - "
    f"{vector_report.relative_to(PROJECT_ROOT)}"
)

print(
    f" - "
    f"{boundary_report.relative_to(PROJECT_ROOT)}"
)

print(
    f" - "
    f"{raster_report.relative_to(PROJECT_ROOT)}"
)

print(
    f" - "
    f"{uda_report.relative_to(PROJECT_ROOT)}"
)


print("\nIMPORTANT:")
print(
    "This step validated spatial quality only."
)

print(
    "No raw or processed source dataset was modified."
)

print("=" * 100)