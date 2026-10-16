"""
STEP 02B - KMC / GN BOUNDARY DIAGNOSTIC

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Compare the main KMC boundary with the KMC boundary supplied
   inside the UDA provisional package.
2. Compare the processed 57-GN union against both KMC boundaries.
3. Identify and export the spatial gaps between KMC and GN coverage.
4. Determine whether the observed ~2.13 km² difference is caused by
   a boundary-version difference or incomplete GN spatial coverage.

IMPORTANT:
This script does NOT modify any source dataset.
All created files are diagnostic outputs only.
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd
import pyogrio


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
MAP_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "maps"
    / "boundary_diagnostic"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)


# Main KMC boundary used by the research
KMC_RAW_PATH = (
    RAW_DIR
    / "boundaries"
    / "kmc"
    / "kmc boundry.kml"
)


# Processed 57-GN working dataset
GN_PATH = (
    PROCESSED_DIR
    / "boundaries"
    / "gn_divisions"
    / "kaduwela_gn_divisions.shp"
)


# UDA provisional package directory
UDA_DIR = (
    RAW_DIR
    / "uda"
    / "zoning_provisional"
)


# Metric CRS used for area calculations
ANALYSIS_CRS = "EPSG:32644"

# Geographic CRS used for readable centroid coordinates
WGS84 = "EPSG:4326"


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def load_vector(
    path: Path,
    dataset_name: str,
    layer: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Load a vector dataset without modifying the source.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} not found:\n{path}"
        )

    if layer is None:
        gdf = gpd.read_file(path)
    else:
        gdf = gpd.read_file(
            path,
            layer=layer,
        )

    if gdf.empty:
        raise ValueError(
            f"{dataset_name} contains no features."
        )

    if gdf.crs is None:
        raise ValueError(
            f"{dataset_name} has no CRS."
        )

    if "geometry" not in gdf.columns:
        raise ValueError(
            f"{dataset_name} has no geometry column."
        )

    print("\n" + "-" * 100)
    print(f"LOADED: {dataset_name}")
    print("-" * 100)

    print(
        f"Path          : "
        f"{path.relative_to(PROJECT_ROOT)}"
    )

    if layer is not None:
        print(f"Layer         : {layer}")

    print(f"Feature count : {len(gdf)}")
    print(f"CRS           : {gdf.crs}")

    return gdf


def projected_union(
    gdf: gpd.GeoDataFrame,
):
    """
    Reproject a GeoDataFrame to the analysis CRS
    and return both the projected GeoDataFrame
    and dissolved union geometry.
    """

    projected = gdf.to_crs(
        ANALYSIS_CRS
    )

    union_geometry = (
        projected
        .geometry
        .union_all()
    )

    return projected, union_geometry


def area_km2(
    geometry,
) -> float:
    """
    Calculate geometry area in square kilometres.

    Geometry must already be in ANALYSIS_CRS.
    """

    if geometry is None:
        return 0.0

    if geometry.is_empty:
        return 0.0

    return float(
        geometry.area / 1_000_000
    )


def safe_percent(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safely calculate percentage.
    """

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator * 100
    )


def export_geometry(
    geometry,
    output_path: Path,
    dataset_name: str,
):
    """
    Export a diagnostic geometry as GeoJSON.
    """

    if geometry is None:
        print(
            f"Skipped empty output: "
            f"{dataset_name}"
        )
        return

    if geometry.is_empty:
        print(
            f"Skipped empty output: "
            f"{dataset_name}"
        )
        return

    output_gdf = gpd.GeoDataFrame(
        {
            "diagnostic": [
                dataset_name
            ]
        },
        geometry=[geometry],
        crs=ANALYSIS_CRS,
    )

    output_gdf = (
        output_gdf.to_crs(WGS84)
    )

    output_gdf.to_file(
        output_path,
        driver="GeoJSON",
    )

    print(
        f"Saved: "
        f"{output_path.relative_to(PROJECT_ROOT)}"
    )


def create_gap_component_table(
    geometry,
    source_name: str,
) -> gpd.GeoDataFrame:
    """
    Split a gap geometry into individual components
    and calculate area and centroid coordinates.
    """

    if geometry is None or geometry.is_empty:

        return gpd.GeoDataFrame(
            columns=[
                "source",
                "component_id",
                "area_km2",
                "centroid_lon",
                "centroid_lat",
                "geometry",
            ],
            geometry="geometry",
            crs=ANALYSIS_CRS,
        )

    gap_gdf = gpd.GeoDataFrame(
        {
            "source": [
                source_name
            ]
        },
        geometry=[geometry],
        crs=ANALYSIS_CRS,
    )

    gap_gdf = (
        gap_gdf
        .explode(
            index_parts=False
        )
        .reset_index(drop=True)
    )

    # Remove any empty geometries
    gap_gdf = gap_gdf[
        ~gap_gdf.geometry.is_empty
    ].copy()

    gap_gdf["component_id"] = (
        range(
            1,
            len(gap_gdf) + 1
        )
    )

    gap_gdf["area_km2"] = (
        gap_gdf.geometry.area
        / 1_000_000
    )

    # Calculate centroids in projected CRS
    centroid_projected = (
        gap_gdf.geometry.centroid
    )

    centroid_gdf = (
        gpd.GeoDataFrame(
            geometry=centroid_projected,
            crs=ANALYSIS_CRS,
        )
        .to_crs(WGS84)
    )

    gap_gdf["centroid_lon"] = (
        centroid_gdf.geometry.x.values
    )

    gap_gdf["centroid_lat"] = (
        centroid_gdf.geometry.y.values
    )

    gap_gdf = gap_gdf.sort_values(
        "area_km2",
        ascending=False,
    ).reset_index(drop=True)

    return gap_gdf


# =============================================================================
# 3. START
# =============================================================================

print("=" * 100)
print("STEP 02B - KMC / GN BOUNDARY DIAGNOSTIC")
print("=" * 100)

print(
    f"\nAnalysis CRS : "
    f"{ANALYSIS_CRS}"
)


# =============================================================================
# 4. LOAD MAIN KMC BOUNDARY
# =============================================================================

kmc_raw_gdf = load_vector(
    KMC_RAW_PATH,
    "Main KMC Boundary",
)


# =============================================================================
# 5. LOAD PROCESSED 57-GN DATASET
# =============================================================================

gn_gdf = load_vector(
    GN_PATH,
    "Processed 57 GN Divisions",
)


# =============================================================================
# 6. FIND UDA-SUPPLIED KMC BOUNDARY
# =============================================================================

uda_boundary_files = list(
    UDA_DIR.rglob(
        "Kaduwela_MC_Boundary.gpkg"
    )
)

if len(uda_boundary_files) == 0:
    raise FileNotFoundError(
        "Kaduwela_MC_Boundary.gpkg "
        "was not found inside the UDA package."
    )

if len(uda_boundary_files) > 1:
    print(
        "\nWARNING:"
        "\nMore than one UDA KMC boundary file "
        "was discovered."
        "\nThe first file will be used."
    )

UDA_BOUNDARY_PATH = (
    uda_boundary_files[0]
)

uda_layers = pyogrio.list_layers(
    UDA_BOUNDARY_PATH
)

print("\nUDA KMC boundary layers discovered:")

for layer_info in uda_layers:
    print(
        f" - {layer_info[0]}"
    )


# Prefer the expected layer name
available_layer_names = [
    str(layer_info[0])
    for layer_info in uda_layers
]

if "Kaduwela_MC_Boundary" in available_layer_names:

    uda_boundary_layer = (
        "Kaduwela_MC_Boundary"
    )

else:

    uda_boundary_layer = (
        available_layer_names[0]
    )


uda_kmc_gdf = load_vector(
    UDA_BOUNDARY_PATH,
    "UDA-Supplied KMC Boundary",
    layer=uda_boundary_layer,
)


# =============================================================================
# 7. PROJECT AND DISSOLVE
# =============================================================================

kmc_raw_projected, kmc_raw_union = (
    projected_union(
        kmc_raw_gdf
    )
)

gn_projected, gn_union = (
    projected_union(
        gn_gdf
    )
)

uda_kmc_projected, uda_kmc_union = (
    projected_union(
        uda_kmc_gdf
    )
)


# =============================================================================
# 8. BASIC AREA CHECK
# =============================================================================

kmc_raw_area = area_km2(
    kmc_raw_union
)

gn_area = area_km2(
    gn_union
)

uda_kmc_area = area_km2(
    uda_kmc_union
)


print("\n" + "=" * 100)
print("BOUNDARY AREA COMPARISON")
print("=" * 100)

print(
    f"Main KMC boundary area : "
    f"{kmc_raw_area:.6f} km²"
)

print(
    f"UDA KMC boundary area  : "
    f"{uda_kmc_area:.6f} km²"
)

print(
    f"57 GN union area       : "
    f"{gn_area:.6f} km²"
)


# =============================================================================
# 9. MAIN KMC vs UDA KMC COMPARISON
# =============================================================================

kmc_uda_intersection = (
    kmc_raw_union.intersection(
        uda_kmc_union
    )
)

kmc_uda_intersection_area = (
    area_km2(
        kmc_uda_intersection
    )
)


kmc_only_area_geometry = (
    kmc_raw_union.difference(
        uda_kmc_union
    )
)

uda_only_area_geometry = (
    uda_kmc_union.difference(
        kmc_raw_union
    )
)

kmc_uda_symmetric_difference = (
    kmc_raw_union.symmetric_difference(
        uda_kmc_union
    )
)


kmc_only_area = area_km2(
    kmc_only_area_geometry
)

uda_only_area = area_km2(
    uda_only_area_geometry
)

kmc_uda_difference_area = area_km2(
    kmc_uda_symmetric_difference
)


main_kmc_covered_by_uda = safe_percent(
    kmc_uda_intersection_area,
    kmc_raw_area,
)

uda_kmc_covered_by_main = safe_percent(
    kmc_uda_intersection_area,
    uda_kmc_area,
)


print("\n" + "=" * 100)
print("MAIN KMC vs UDA KMC")
print("=" * 100)

print(
    f"Intersection area                 : "
    f"{kmc_uda_intersection_area:.6f} km²"
)

print(
    f"Main KMC covered by UDA boundary  : "
    f"{main_kmc_covered_by_uda:.6f} %"
)

print(
    f"UDA KMC covered by main boundary  : "
    f"{uda_kmc_covered_by_main:.6f} %"
)

print(
    f"Main-KMC-only area                : "
    f"{kmc_only_area:.6f} km²"
)

print(
    f"UDA-KMC-only area                 : "
    f"{uda_only_area:.6f} km²"
)

print(
    f"Total symmetric difference        : "
    f"{kmc_uda_difference_area:.6f} km²"
)


# =============================================================================
# 10. GN vs MAIN KMC
# =============================================================================

gn_inside_main_kmc = (
    gn_union.intersection(
        kmc_raw_union
    )
)

main_kmc_missing_gn = (
    kmc_raw_union.difference(
        gn_union
    )
)

gn_outside_main_kmc = (
    gn_union.difference(
        kmc_raw_union
    )
)


gn_inside_main_kmc_area = area_km2(
    gn_inside_main_kmc
)

main_kmc_missing_gn_area = area_km2(
    main_kmc_missing_gn
)

gn_outside_main_kmc_area = area_km2(
    gn_outside_main_kmc
)

gn_main_kmc_coverage = safe_percent(
    gn_inside_main_kmc_area,
    kmc_raw_area,
)


print("\n" + "=" * 100)
print("57 GN vs MAIN KMC")
print("=" * 100)

print(
    f"GN inside main KMC       : "
    f"{gn_inside_main_kmc_area:.6f} km²"
)

print(
    f"GN coverage of main KMC  : "
    f"{gn_main_kmc_coverage:.6f} %"
)

print(
    f"Main KMC missing GN      : "
    f"{main_kmc_missing_gn_area:.6f} km²"
)

print(
    f"GN outside main KMC      : "
    f"{gn_outside_main_kmc_area:.6f} km²"
)


# =============================================================================
# 11. GN vs UDA KMC
# =============================================================================

gn_inside_uda_kmc = (
    gn_union.intersection(
        uda_kmc_union
    )
)

uda_kmc_missing_gn = (
    uda_kmc_union.difference(
        gn_union
    )
)

gn_outside_uda_kmc = (
    gn_union.difference(
        uda_kmc_union
    )
)


gn_inside_uda_kmc_area = area_km2(
    gn_inside_uda_kmc
)

uda_kmc_missing_gn_area = area_km2(
    uda_kmc_missing_gn
)

gn_outside_uda_kmc_area = area_km2(
    gn_outside_uda_kmc
)

gn_uda_kmc_coverage = safe_percent(
    gn_inside_uda_kmc_area,
    uda_kmc_area,
)


print("\n" + "=" * 100)
print("57 GN vs UDA KMC")
print("=" * 100)

print(
    f"GN inside UDA KMC       : "
    f"{gn_inside_uda_kmc_area:.6f} km²"
)

print(
    f"GN coverage of UDA KMC  : "
    f"{gn_uda_kmc_coverage:.6f} %"
)

print(
    f"UDA KMC missing GN      : "
    f"{uda_kmc_missing_gn_area:.6f} km²"
)

print(
    f"GN outside UDA KMC      : "
    f"{gn_outside_uda_kmc_area:.6f} km²"
)


# =============================================================================
# 12. GAP COMPONENT ANALYSIS
# =============================================================================

main_gap_components = (
    create_gap_component_table(
        main_kmc_missing_gn,
        "Main KMC minus GN",
    )
)

uda_gap_components = (
    create_gap_component_table(
        uda_kmc_missing_gn,
        "UDA KMC minus GN",
    )
)


print("\n" + "=" * 100)
print("GN GAP COMPONENTS")
print("=" * 100)

print(
    f"Main KMC gap components : "
    f"{len(main_gap_components)}"
)

print(
    f"UDA KMC gap components  : "
    f"{len(uda_gap_components)}"
)


if not main_gap_components.empty:

    print(
        "\nLargest Main-KMC-minus-GN gaps:"
    )

    print(
        main_gap_components[
            [
                "component_id",
                "area_km2",
                "centroid_lon",
                "centroid_lat",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


# =============================================================================
# 13. EXPORT DIAGNOSTIC MAP LAYERS
# =============================================================================

print("\n" + "=" * 100)
print("EXPORTING DIAGNOSTIC LAYERS")
print("=" * 100)


export_geometry(
    main_kmc_missing_gn,
    MAP_DIR
    / "02b_main_kmc_minus_gn.geojson",
    "Main KMC area not covered by processed GN divisions",
)


export_geometry(
    uda_kmc_missing_gn,
    MAP_DIR
    / "02b_uda_kmc_minus_gn.geojson",
    "UDA KMC area not covered by processed GN divisions",
)


export_geometry(
    gn_outside_main_kmc,
    MAP_DIR
    / "02b_gn_outside_main_kmc.geojson",
    "Processed GN area outside main KMC boundary",
)


export_geometry(
    kmc_uda_symmetric_difference,
    MAP_DIR
    / "02b_main_vs_uda_kmc_difference.geojson",
    "Difference between main and UDA KMC boundaries",
)


# =============================================================================
# 14. SAVE GAP COMPONENT TABLE
# =============================================================================

gap_component_records = []


for _, row in main_gap_components.iterrows():

    gap_component_records.append(
        {
            "boundary_reference": (
                "Main KMC"
            ),
            "component_id": (
                int(row["component_id"])
            ),
            "area_km2": (
                float(row["area_km2"])
            ),
            "centroid_lon": (
                float(row["centroid_lon"])
            ),
            "centroid_lat": (
                float(row["centroid_lat"])
            ),
        }
    )


for _, row in uda_gap_components.iterrows():

    gap_component_records.append(
        {
            "boundary_reference": (
                "UDA KMC"
            ),
            "component_id": (
                int(row["component_id"])
            ),
            "area_km2": (
                float(row["area_km2"])
            ),
            "centroid_lon": (
                float(row["centroid_lon"])
            ),
            "centroid_lat": (
                float(row["centroid_lat"])
            ),
        }
    )


gap_components_df = pd.DataFrame(
    gap_component_records
)


gap_components_report = (
    REPORT_DIR
    / "02b_kmc_gn_gap_components.csv"
)


gap_components_df.to_csv(
    gap_components_report,
    index=False,
)


# =============================================================================
# 15. CREATE SUMMARY REPORT
# =============================================================================

summary_record = {
    "main_kmc_area_km2": (
        kmc_raw_area
    ),
    "uda_kmc_area_km2": (
        uda_kmc_area
    ),
    "gn_union_area_km2": (
        gn_area
    ),
    "main_uda_intersection_km2": (
        kmc_uda_intersection_area
    ),
    "main_kmc_covered_by_uda_percent": (
        main_kmc_covered_by_uda
    ),
    "uda_kmc_covered_by_main_percent": (
        uda_kmc_covered_by_main
    ),
    "main_kmc_only_km2": (
        kmc_only_area
    ),
    "uda_kmc_only_km2": (
        uda_only_area
    ),
    "main_uda_symmetric_difference_km2": (
        kmc_uda_difference_area
    ),
    "gn_coverage_main_kmc_percent": (
        gn_main_kmc_coverage
    ),
    "main_kmc_missing_gn_km2": (
        main_kmc_missing_gn_area
    ),
    "gn_outside_main_kmc_km2": (
        gn_outside_main_kmc_area
    ),
    "gn_coverage_uda_kmc_percent": (
        gn_uda_kmc_coverage
    ),
    "uda_kmc_missing_gn_km2": (
        uda_kmc_missing_gn_area
    ),
    "gn_outside_uda_kmc_km2": (
        gn_outside_uda_kmc_area
    ),
}


summary_df = pd.DataFrame(
    [summary_record]
)


summary_report = (
    REPORT_DIR
    / "02b_kmc_gn_boundary_diagnostic.csv"
)


summary_df.to_csv(
    summary_report,
    index=False,
)


# =============================================================================
# 16. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("STEP 02B SUMMARY")
print("=" * 100)

print(
    f"Main KMC area               : "
    f"{kmc_raw_area:.6f} km²"
)

print(
    f"UDA KMC area                : "
    f"{uda_kmc_area:.6f} km²"
)

print(
    f"57 GN union area            : "
    f"{gn_area:.6f} km²"
)

print(
    f"Main vs UDA boundary diff   : "
    f"{kmc_uda_difference_area:.6f} km²"
)

print(
    f"GN coverage of main KMC     : "
    f"{gn_main_kmc_coverage:.6f} %"
)

print(
    f"GN coverage of UDA KMC      : "
    f"{gn_uda_kmc_coverage:.6f} %"
)

print(
    f"Main KMC missing GN         : "
    f"{main_kmc_missing_gn_area:.6f} km²"
)

print(
    f"UDA KMC missing GN          : "
    f"{uda_kmc_missing_gn_area:.6f} km²"
)


print("\nReports created:")

print(
    f" - "
    f"{summary_report.relative_to(PROJECT_ROOT)}"
)

print(
    f" - "
    f"{gap_components_report.relative_to(PROJECT_ROOT)}"
)


print("\nDiagnostic map layers created:")

print(
    f" - outputs/maps/boundary_diagnostic/"
    f"02b_main_kmc_minus_gn.geojson"
)

print(
    f" - outputs/maps/boundary_diagnostic/"
    f"02b_uda_kmc_minus_gn.geojson"
)

print(
    f" - outputs/maps/boundary_diagnostic/"
    f"02b_gn_outside_main_kmc.geojson"
)

print(
    f" - outputs/maps/boundary_diagnostic/"
    f"02b_main_vs_uda_kmc_difference.geojson"
)


print("\nIMPORTANT:")
print(
    "No source boundary or GN dataset was modified."
)

print(
    "All generated files are diagnostic outputs only."
)

print("=" * 100)