from pathlib import Path

import pandas as pd
import geopandas as gpd

from shapely.geometry import (
    Polygon,
    MultiPolygon,
    GeometryCollection,
)


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

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "part1_historical_change"
)

INPUT_GPKG = (
    PROCESSED_DIR
    / "historical_wetland_observations_utm44n.gpkg"
)

OUTPUT_CSV = (
    INTERIM_DIR
    / "minimum_mapping_unit_sensitivity.csv"
)

SUMMARY_CSV = (
    INTERIM_DIR
    / "minimum_mapping_unit_summary.csv"
)


# ============================================================
# TEST THRESHOLDS
# ============================================================

THRESHOLDS_M2 = [
    0,
    25,
    50,
    100,
]


# ============================================================
# GEOMETRY HELPER
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

    rows = []

    for _, row in observations.iterrows():

        parts = extract_polygon_parts(
            row.geometry
        )

        original_areas = [
            part.area
            for part in parts
            if part.area > 0
        ]

        if not original_areas:
            continue

        original_total_m2 = sum(
            original_areas
        )

        original_patch_count = len(
            original_areas
        )

        for threshold in THRESHOLDS_M2:

            retained = [
                area
                for area in original_areas
                if area >= threshold
            ]

            removed = [
                area
                for area in original_areas
                if area < threshold
            ]

            retained_total_m2 = sum(
                retained
            )

            removed_total_m2 = sum(
                removed
            )

            retained_patch_count = len(
                retained
            )

            removed_patch_count = len(
                removed
            )

            if original_total_m2 > 0:

                area_removed_percent = (
                    removed_total_m2
                    / original_total_m2
                    * 100.0
                )

            else:
                area_removed_percent = 0.0

            if retained:
                largest_patch_m2 = max(
                    retained
                )

                lpi_percent = (
                    largest_patch_m2
                    / retained_total_m2
                    * 100.0
                )

            else:
                largest_patch_m2 = 0.0
                lpi_percent = None

            rows.append(
                {
                    "wetland_id":
                        row["wetland_id"],

                    "observation_date":
                        row["observation_date"],

                    "threshold_m2":
                        threshold,

                    "original_patch_count":
                        original_patch_count,

                    "retained_patch_count":
                        retained_patch_count,

                    "removed_patch_count":
                        removed_patch_count,

                    "original_total_area_ha":
                        original_total_m2 / 10000.0,

                    "retained_total_area_ha":
                        retained_total_m2 / 10000.0,

                    "removed_total_area_ha":
                        removed_total_m2 / 10000.0,

                    "area_removed_percent":
                        area_removed_percent,

                    "lpi_percent":
                        lpi_percent,
                }
            )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ========================================================
    # SUMMARY BY THRESHOLD
    # ========================================================

    summary = (
        results
        .groupby("threshold_m2")
        .agg(
            total_observations=(
                "wetland_id",
                "count",
            ),

            total_removed_patches=(
                "removed_patch_count",
                "sum",
            ),

            mean_removed_patches=(
                "removed_patch_count",
                "mean",
            ),

            mean_area_removed_percent=(
                "area_removed_percent",
                "mean",
            ),

            median_area_removed_percent=(
                "area_removed_percent",
                "median",
            ),

            max_area_removed_percent=(
                "area_removed_percent",
                "max",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    print("\n========================================")
    print("MINIMUM MAPPING UNIT SENSITIVITY")
    print("========================================")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nObservations losing >1% area:")

    for threshold in THRESHOLDS_M2[1:]:

        subset = results[
            (
                results["threshold_m2"]
                == threshold
            )
            &
            (
                results["area_removed_percent"]
                > 1.0
            )
        ]

        print(
            f"{threshold} m² threshold: "
            f"{len(subset)} observations"
        )

    print("\nObservations losing >5% area:")

    for threshold in THRESHOLDS_M2[1:]:

        subset = results[
            (
                results["threshold_m2"]
                == threshold
            )
            &
            (
                results["area_removed_percent"]
                > 5.0
            )
        ]

        print(
            f"{threshold} m² threshold: "
            f"{len(subset)} observations"
        )

    print("\nOutputs:")

    print(OUTPUT_CSV)
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()