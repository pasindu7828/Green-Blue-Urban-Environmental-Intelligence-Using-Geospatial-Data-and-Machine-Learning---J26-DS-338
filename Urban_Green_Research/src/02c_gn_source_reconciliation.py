"""
STEP 02C - GN SOURCE RECONCILIATION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Inspect the original KMC/UDA GN KML source.
2. Compare the original 56-GN layer plus the separate feature
   against the processed 57-GN working dataset.
3. Determine whether the ~2.13 km² KMC/GN gap existed in the
   original source or was introduced during preprocessing.
4. Validate that the processed 57-GN dataset preserves the
   source geometries.

IMPORTANT:
This script does not modify any source dataset.
All outputs are diagnostic only.
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


KMC_PATH = (
    RAW_DIR
    / "boundaries"
    / "kmc"
    / "kmc boundry.kml"
)

RAW_GN_KML = (
    RAW_DIR
    / "boundaries"
    / "gn_divisions"
    / "Kaduwela TO Area.kml"
)

PROCESSED_GN_PATH = (
    PROCESSED_DIR
    / "boundaries"
    / "gn_divisions"
    / "kaduwela_gn_divisions.shp"
)

ANALYSIS_CRS = "EPSG:32644"
WGS84 = "EPSG:4326"


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def area_km2(geometry):
    """Return geometry area in square kilometres."""

    if geometry is None or geometry.is_empty:
        return 0.0

    return float(
        geometry.area / 1_000_000
    )


def safe_percent(numerator, denominator):
    """Safely calculate percentage."""

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator * 100
    )


def union_projected(gdf):
    """
    Reproject to analysis CRS and return dissolved union.
    """

    projected = gdf.to_crs(
        ANALYSIS_CRS
    )

    union_geometry = (
        projected.geometry.union_all()
    )

    return projected, union_geometry


def export_geometry(
    geometry,
    output_path,
    description,
):
    """Export a diagnostic geometry."""

    if geometry is None or geometry.is_empty:
        print(
            f"Skipped empty output: {description}"
        )
        return

    output_gdf = gpd.GeoDataFrame(
        {
            "description": [description]
        },
        geometry=[geometry],
        crs=ANALYSIS_CRS,
    ).to_crs(WGS84)

    output_gdf.to_file(
        output_path,
        driver="GeoJSON",
    )

    print(
        f"Saved: "
        f"{output_path.relative_to(PROJECT_ROOT)}"
    )


# =============================================================================
# 3. START
# =============================================================================

print("=" * 100)
print("STEP 02C - GN SOURCE RECONCILIATION")
print("=" * 100)


# =============================================================================
# 4. INSPECT ORIGINAL KML LAYERS
# =============================================================================

raw_layers = pyogrio.list_layers(
    RAW_GN_KML
)

print("\nOriginal GN KML layers:")

for layer_info in raw_layers:
    print(
        f" - {layer_info[0]}"
    )


layer_names = [
    str(layer_info[0])
    for layer_info in raw_layers
]


# =============================================================================
# 5. IDENTIFY THE 56-GN SOURCE LAYER
# =============================================================================

source_56_layer = None

for layer_name in layer_names:

    info = pyogrio.read_info(
        RAW_GN_KML,
        layer=layer_name,
    )

    feature_count = info.get(
        "features"
    )

    if feature_count == 56:
        source_56_layer = layer_name
        break


if source_56_layer is None:
    raise ValueError(
        "Could not identify the original "
        "56-feature GN layer."
    )


print(
    f"\n56-GN source layer : "
    f"{source_56_layer}"
)


raw_56_gdf = gpd.read_file(
    RAW_GN_KML,
    layer=source_56_layer,
)


# =============================================================================
# 6. IDENTIFY THE SEPARATE SINGLE FEATURE
# =============================================================================

single_feature_layer = None

for layer_name in layer_names:

    info = pyogrio.read_info(
        RAW_GN_KML,
        layer=layer_name,
    )

    feature_count = info.get(
        "features"
    )

    if feature_count == 1:
        single_feature_layer = layer_name
        break


if single_feature_layer is None:
    raise ValueError(
        "Could not identify the separate "
        "single-feature GN layer."
    )


print(
    f"Separate feature layer : "
    f"{single_feature_layer}"
)


single_gdf = gpd.read_file(
    RAW_GN_KML,
    layer=single_feature_layer,
)


# =============================================================================
# 7. LOAD PROCESSED 57 GN DATASET
# =============================================================================

processed_57_gdf = gpd.read_file(
    PROCESSED_GN_PATH
)


print("\nFeature counts")

print(
    f"Original GN layer       : "
    f"{len(raw_56_gdf)}"
)

print(
    f"Separate feature        : "
    f"{len(single_gdf)}"
)

print(
    f"Expected combined count : "
    f"{len(raw_56_gdf) + len(single_gdf)}"
)

print(
    f"Processed GN count      : "
    f"{len(processed_57_gdf)}"
)


# =============================================================================
# 8. LOAD KMC BOUNDARY
# =============================================================================

kmc_gdf = gpd.read_file(
    KMC_PATH
)


_, kmc_union = union_projected(
    kmc_gdf
)


kmc_area = area_km2(
    kmc_union
)


# =============================================================================
# 9. CREATE ORIGINAL SOURCE UNION
# =============================================================================

raw_56_projected = raw_56_gdf.to_crs(
    ANALYSIS_CRS
)

single_projected = single_gdf.to_crs(
    ANALYSIS_CRS
)


raw_56_union = (
    raw_56_projected
    .geometry
    .union_all()
)

single_union = (
    single_projected
    .geometry
    .union_all()
)


combined_source_union = (
    raw_56_union.union(
        single_union
    )
)


processed_projected, processed_union = (
    union_projected(
        processed_57_gdf
    )
)


# =============================================================================
# 10. AREA COMPARISON
# =============================================================================

raw_56_area = area_km2(
    raw_56_union
)

single_area = area_km2(
    single_union
)

combined_source_area = area_km2(
    combined_source_union
)

processed_area = area_km2(
    processed_union
)


print("\n" + "=" * 100)
print("SOURCE / PROCESSED AREA COMPARISON")
print("=" * 100)

print(
    f"Original 56-GN union area : "
    f"{raw_56_area:.6f} km²"
)

print(
    f"Separate feature area     : "
    f"{single_area:.6f} km²"
)

print(
    f"Combined source union     : "
    f"{combined_source_area:.6f} km²"
)

print(
    f"Processed 57-GN union     : "
    f"{processed_area:.6f} km²"
)


# =============================================================================
# 11. ORIGINAL SOURCE vs PROCESSED 57
# =============================================================================

source_processed_intersection = (
    combined_source_union.intersection(
        processed_union
    )
)

source_only = (
    combined_source_union.difference(
        processed_union
    )
)

processed_only = (
    processed_union.difference(
        combined_source_union
    )
)

source_processed_difference = (
    combined_source_union
    .symmetric_difference(
        processed_union
    )
)


intersection_area = area_km2(
    source_processed_intersection
)

source_only_area = area_km2(
    source_only
)

processed_only_area = area_km2(
    processed_only
)

difference_area = area_km2(
    source_processed_difference
)


source_preserved_percent = (
    safe_percent(
        intersection_area,
        combined_source_area,
    )
)


print("\n" + "=" * 100)
print("ORIGINAL SOURCE vs PROCESSED 57-GN")
print("=" * 100)

print(
    f"Intersection area          : "
    f"{intersection_area:.9f} km²"
)

print(
    f"Source preserved           : "
    f"{source_preserved_percent:.9f} %"
)

print(
    f"Source-only area           : "
    f"{source_only_area:.9f} km²"
)

print(
    f"Processed-only area        : "
    f"{processed_only_area:.9f} km²"
)

print(
    f"Symmetric difference       : "
    f"{difference_area:.9f} km²"
)


# =============================================================================
# 12. ORIGINAL SOURCE vs KMC
# =============================================================================

source_inside_kmc = (
    combined_source_union.intersection(
        kmc_union
    )
)

kmc_missing_source = (
    kmc_union.difference(
        combined_source_union
    )
)

source_outside_kmc = (
    combined_source_union.difference(
        kmc_union
    )
)


source_inside_kmc_area = area_km2(
    source_inside_kmc
)

kmc_missing_source_area = area_km2(
    kmc_missing_source
)

source_outside_kmc_area = area_km2(
    source_outside_kmc
)


source_kmc_coverage = (
    safe_percent(
        source_inside_kmc_area,
        kmc_area,
    )
)


print("\n" + "=" * 100)
print("ORIGINAL GN SOURCE vs KMC")
print("=" * 100)

print(
    f"KMC area                     : "
    f"{kmc_area:.6f} km²"
)

print(
    f"Source GN inside KMC         : "
    f"{source_inside_kmc_area:.6f} km²"
)

print(
    f"Source GN coverage of KMC    : "
    f"{source_kmc_coverage:.6f} %"
)

print(
    f"KMC absent from source GN    : "
    f"{kmc_missing_source_area:.6f} km²"
)

print(
    f"Source GN outside KMC        : "
    f"{source_outside_kmc_area:.9f} km²"
)


# =============================================================================
# 13. PROCESSED GN vs KMC
# =============================================================================

processed_inside_kmc = (
    processed_union.intersection(
        kmc_union
    )
)

kmc_missing_processed = (
    kmc_union.difference(
        processed_union
    )
)


processed_inside_kmc_area = area_km2(
    processed_inside_kmc
)

kmc_missing_processed_area = area_km2(
    kmc_missing_processed
)


processed_kmc_coverage = (
    safe_percent(
        processed_inside_kmc_area,
        kmc_area,
    )
)


print("\n" + "=" * 100)
print("PROCESSED 57-GN vs KMC")
print("=" * 100)

print(
    f"Processed coverage of KMC : "
    f"{processed_kmc_coverage:.6f} %"
)

print(
    f"KMC absent from processed : "
    f"{kmc_missing_processed_area:.6f} km²"
)


# =============================================================================
# 14. EXPORT DIAGNOSTIC DIFFERENCES
# =============================================================================

print("\n" + "=" * 100)
print("EXPORTING DIAGNOSTIC OUTPUTS")
print("=" * 100)


export_geometry(
    source_processed_difference,
    MAP_DIR
    / "02c_raw_vs_processed_gn_difference.geojson",
    "Difference between original GN source and processed 57-GN dataset",
)


export_geometry(
    kmc_missing_source,
    MAP_DIR
    / "02c_kmc_missing_from_original_gn.geojson",
    "KMC area absent from original supplied GN source",
)


# =============================================================================
# 15. SAVE REPORT
# =============================================================================

summary_df = pd.DataFrame(
    [
        {
            "kmc_area_km2": (
                kmc_area
            ),
            "raw_56_feature_count": (
                len(raw_56_gdf)
            ),
            "separate_feature_count": (
                len(single_gdf)
            ),
            "combined_source_feature_count": (
                len(raw_56_gdf)
                + len(single_gdf)
            ),
            "processed_feature_count": (
                len(processed_57_gdf)
            ),
            "raw_56_union_area_km2": (
                raw_56_area
            ),
            "separate_feature_area_km2": (
                single_area
            ),
            "combined_source_union_km2": (
                combined_source_area
            ),
            "processed_57_union_km2": (
                processed_area
            ),
            "source_processed_intersection_km2": (
                intersection_area
            ),
            "source_preserved_percent": (
                source_preserved_percent
            ),
            "source_only_km2": (
                source_only_area
            ),
            "processed_only_km2": (
                processed_only_area
            ),
            "source_processed_difference_km2": (
                difference_area
            ),
            "source_kmc_coverage_percent": (
                source_kmc_coverage
            ),
            "kmc_missing_source_km2": (
                kmc_missing_source_area
            ),
            "processed_kmc_coverage_percent": (
                processed_kmc_coverage
            ),
            "kmc_missing_processed_km2": (
                kmc_missing_processed_area
            ),
        }
    ]
)


REPORT_PATH = (
    REPORT_DIR
    / "02c_gn_source_reconciliation.csv"
)


summary_df.to_csv(
    REPORT_PATH,
    index=False,
)


# =============================================================================
# 16. INTERPRETATION CHECK
# =============================================================================

# Small numerical differences can occur because of
# format conversion / geometry precision.
TOLERANCE_KM2 = 0.001


if difference_area <= TOLERANCE_KM2:

    processing_result = (
        "PASS - Processed 57-GN geometry "
        "matches the combined original GN source."
    )

else:

    processing_result = (
        "REVIEW REQUIRED - Processed 57-GN geometry "
        "differs materially from the combined source."
    )


gap_difference = abs(
    kmc_missing_source_area
    - kmc_missing_processed_area
)


if gap_difference <= TOLERANCE_KM2:

    gap_result = (
        "PASS - The KMC/GN gap already exists "
        "in the original supplied GN source."
    )

else:

    gap_result = (
        "REVIEW REQUIRED - The KMC/GN gap changed "
        "during GN processing."
    )


# =============================================================================
# 17. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("STEP 02C SUMMARY")
print("=" * 100)

print(
    f"Original source features      : "
    f"{len(raw_56_gdf) + len(single_gdf)}"
)

print(
    f"Processed GN features         : "
    f"{len(processed_57_gdf)}"
)

print(
    f"Source preserved              : "
    f"{source_preserved_percent:.9f} %"
)

print(
    f"Source/processed difference   : "
    f"{difference_area:.9f} km²"
)

print(
    f"Original-source KMC coverage  : "
    f"{source_kmc_coverage:.6f} %"
)

print(
    f"Processed-GN KMC coverage     : "
    f"{processed_kmc_coverage:.6f} %"
)

print(
    f"Original-source KMC gap       : "
    f"{kmc_missing_source_area:.6f} km²"
)

print(
    f"Processed-GN KMC gap          : "
    f"{kmc_missing_processed_area:.6f} km²"
)

print(
    f"\nProcessing validation : "
    f"{processing_result}"
)

print(
    f"Gap provenance        : "
    f"{gap_result}"
)

print(
    f"\nReport created:"
    f"\n - {REPORT_PATH.relative_to(PROJECT_ROOT)}"
)

print("\nIMPORTANT:")
print(
    "No source or processed GN dataset was modified."
)

print("=" * 100)