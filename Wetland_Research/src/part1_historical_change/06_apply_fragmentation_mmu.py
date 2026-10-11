from pathlib import Path

import pandas as pd
import geopandas as gpd

from shapely.geometry import (
    Polygon,
    MultiPolygon,
    GeometryCollection,
)
from shapely.ops import unary_union


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part1_historical_change"
)

INPUT_GPKG = (
    PROCESSED_DIR
    / "historical_wetland_observations_utm44n.gpkg"
)

OUTPUT_CSV = (
    PROCESSED_DIR
    / "historical_wetland_clean_metrics.csv"
)

OUTPUT_GPKG = (
    PROCESSED_DIR
    / "historical_wetland_clean_observations_utm44n.gpkg"
)

OUTPUT_GEOJSON = (
    PROCESSED_DIR
    / "historical_wetland_clean_observations_wgs84.geojson"
)


# ============================================================
# FRAGMENTATION MMU
# ============================================================

MMU_M2 = 25.0


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def extract_polygon_parts(geometry):

    if geometry is None or geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)

    if isinstance(geometry, GeometryCollection):

        polygons = []

        for part in geometry.geoms:
            polygons.extend(
                extract_polygon_parts(part)
            )

        return polygons

    return []


# ============================================================
# MAIN
# ============================================================

def main():

    observations = gpd.read_file(
        INPUT_GPKG,
        layer="historical_wetland_observations",
    )

    output_rows = []

    total_removed_patches = 0
    observations_changed = 0

    for _, row in observations.iterrows():

        parts = extract_polygon_parts(
            row.geometry
        )

        if not parts:
            continue

        # ----------------------------------------------------
        # RAW / ORIGINAL METRICS
        # ----------------------------------------------------

        raw_areas_m2 = [
            part.area
            for part in parts
        ]

        raw_total_area_m2 = sum(
            raw_areas_m2
        )

        raw_total_area_ha = (
            raw_total_area_m2 / 10000.0
        )

        raw_patch_count = len(
            parts
        )

        # ----------------------------------------------------
        # APPLY 25 m² MMU FOR FRAGMENTATION
        # ----------------------------------------------------

        retained_parts = [
            part
            for part in parts
            if part.area >= MMU_M2
        ]

        removed_parts = [
            part
            for part in parts
            if part.area < MMU_M2
        ]

        removed_patch_count = len(
            removed_parts
        )

        total_removed_patches += (
            removed_patch_count
        )

        if removed_patch_count > 0:
            observations_changed += 1

        # Safety fallback:
        # sensitivity analysis showed this should not occur
        # for the selected 25 m² threshold.
        if not retained_parts:
            retained_parts = parts

        filtered_geometry = unary_union(
            retained_parts
        )

        retained_areas_m2 = [
            part.area
            for part in retained_parts
        ]

        filtered_total_area_m2 = sum(
            retained_areas_m2
        )

        filtered_total_area_ha = (
            filtered_total_area_m2
            / 10000.0
        )

        filtered_patch_count = len(
            retained_parts
        )

        largest_patch_m2 = max(
            retained_areas_m2
        )

        largest_patch_ha = (
            largest_patch_m2
            / 10000.0
        )

        filtered_perimeter_m = sum(
            part.length
            for part in retained_parts
        )

        lpi_percent = (
            largest_patch_m2
            / filtered_total_area_m2
            * 100.0
        )

        patch_density_per_ha = (
            filtered_patch_count
            / filtered_total_area_ha
            if filtered_total_area_ha > 0
            else None
        )

        removed_area_m2 = (
            raw_total_area_m2
            - filtered_total_area_m2
        )

        removed_area_percent = (
            removed_area_m2
            / raw_total_area_m2
            * 100.0
            if raw_total_area_m2 > 0
            else 0.0
        )

        output_rows.append(
            {
                "wetland_id":
                    row["wetland_id"],

                "observation_date":
                    row["observation_date"],

                "year":
                    row["year"],

                "month":
                    row["month"],

                "day":
                    row["day"],

                "season":
                    row["season"],

                # Original area is preserved
                "raw_total_area_ha":
                    raw_total_area_ha,

                "raw_patch_count":
                    raw_patch_count,

                # MMU-cleaned fragmentation metrics
                "analysis_patch_count":
                    filtered_patch_count,

                "largest_patch_ha":
                    largest_patch_ha,

                "lpi_percent":
                    lpi_percent,

                "patch_density_per_ha":
                    patch_density_per_ha,

                "analysis_perimeter_m":
                    filtered_perimeter_m,

                # QA / transparency fields
                "mmu_m2":
                    MMU_M2,

                "removed_patch_count":
                    removed_patch_count,

                "removed_area_ha":
                    removed_area_m2 / 10000.0,

                "removed_area_percent":
                    removed_area_percent,

                "analysis_total_area_ha":
                    filtered_total_area_ha,

                "geometry":
                    filtered_geometry,
            }
        )

    # ========================================================
    # CREATE CLEAN GEODATAFRAME
    # ========================================================

    clean = gpd.GeoDataFrame(
        output_rows,
        geometry="geometry",
        crs=observations.crs,
    )

    clean = clean.sort_values(
        [
            "wetland_id",
            "observation_date",
        ]
    ).reset_index(drop=True)

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_columns = [
        "wetland_id",
        "observation_date",
        "year",
        "month",
        "day",
        "season",
        "raw_total_area_ha",
        "raw_patch_count",
        "analysis_patch_count",
        "largest_patch_ha",
        "lpi_percent",
        "patch_density_per_ha",
        "analysis_perimeter_m",
        "mmu_m2",
        "removed_patch_count",
        "removed_area_ha",
        "removed_area_percent",
        "analysis_total_area_ha",
    ]

    clean[
        csv_columns
    ].to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ========================================================
    # SAVE METRIC CRS VERSION
    # ========================================================

    clean.to_file(
        OUTPUT_GPKG,
        layer="historical_wetland_clean_observations",
        driver="GPKG",
    )

    # ========================================================
    # SAVE WGS84 VERSION FOR WEB / FRONTEND
    # ========================================================

    clean_wgs84 = clean.to_crs(
        "EPSG:4326"
    )

    clean_wgs84.to_file(
        OUTPUT_GEOJSON,
        driver="GeoJSON",
    )

    # ========================================================
    # QA SUMMARY
    # ========================================================

    print("\n========================================")
    print("FRAGMENTATION MMU APPLIED")
    print("========================================")

    print(
        f"MMU used: {MMU_M2:.0f} m²"
    )

    print(
        f"Observations processed: {len(clean)}"
    )

    print(
        f"Observations affected by MMU: "
        f"{observations_changed}"
    )

    print(
        f"Small fragments removed: "
        f"{total_removed_patches}"
    )

    print(
        f"Maximum area removed from any "
        f"observation: "
        f"{clean['removed_area_percent'].max():.6f}%"
    )

    print(
        f"Mean area removed: "
        f"{clean['removed_area_percent'].mean():.6f}%"
    )

    print(
        f"Minimum analysis patch count: "
        f"{clean['analysis_patch_count'].min()}"
    )

    print(
        f"Maximum analysis patch count: "
        f"{clean['analysis_patch_count'].max()}"
    )

    print(
        f"LPI range: "
        f"{clean['lpi_percent'].min():.4f}% - "
        f"{clean['lpi_percent'].max():.4f}%"
    )

    print("\nOutputs:")

    print(OUTPUT_CSV)
    print(OUTPUT_GPKG)
    print(OUTPUT_GEOJSON)

    print(
        "\nNOTE:"
        "\nOriginal wetland area is preserved in "
        "'raw_total_area_ha'."
        "\nThe 25 m² MMU is used only to prevent "
        "tiny geometry artifacts from inflating "
        "fragmentation metrics."
    )


if __name__ == "__main__":
    main()