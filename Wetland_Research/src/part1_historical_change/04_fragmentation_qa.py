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

OUTPUT_DIR = (
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
    OUTPUT_DIR
    / "fragmentation_component_qa.csv"
)

SUSPICIOUS_CSV = (
    OUTPUT_DIR
    / "fragmentation_suspicious_observations.csv"
)


# ============================================================
# HELPERS
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

        areas_m2 = sorted(
            [part.area for part in parts]
        )

        if not areas_m2:
            continue

        total_area_m2 = sum(areas_m2)

        largest_area_m2 = max(areas_m2)

        smallest_area_m2 = min(areas_m2)

        rows.append(
            {
                "wetland_id":
                    row["wetland_id"],

                "observation_date":
                    row["observation_date"],

                "patch_count":
                    len(areas_m2),

                "total_area_ha":
                    total_area_m2 / 10000.0,

                "largest_patch_ha":
                    largest_area_m2 / 10000.0,

                "smallest_patch_m2":
                    smallest_area_m2,

                "smallest_patch_ha":
                    smallest_area_m2 / 10000.0,

                "lpi_percent":
                    (
                        largest_area_m2
                        / total_area_m2
                        * 100.0
                    ),

                "patches_under_10m2":
                    sum(
                        area < 10
                        for area in areas_m2
                    ),

                "patches_under_25m2":
                    sum(
                        area < 25
                        for area in areas_m2
                    ),

                "patches_under_50m2":
                    sum(
                        area < 50
                        for area in areas_m2
                    ),

                "patches_under_100m2":
                    sum(
                        area < 100
                        for area in areas_m2
                    ),
            }
        )

    qa = pd.DataFrame(rows)

    qa.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # Observations worth manual review:
    # multiple patches but one patch dominates almost 100%.
    suspicious = qa[
        (qa["patch_count"] > 1)
        &
        (
            qa["lpi_percent"] >= 99.5
        )
    ].copy()

    suspicious.to_csv(
        SUSPICIOUS_CSV,
        index=False,
    )

    print("\n========================================")
    print("FRAGMENTATION COMPONENT QA")
    print("========================================")

    print(
        f"Observations checked: {len(qa)}"
    )

    print(
        f"Multi-patch observations: "
        f"{(qa['patch_count'] > 1).sum()}"
    )

    print(
        f"Observations with LPI >= 99.5% "
        f"and multiple patches: "
        f"{len(suspicious)}"
    )

    print("\nTiny component counts:")

    print(
        "< 10 m²:",
        int(qa["patches_under_10m2"].sum())
    )

    print(
        "< 25 m²:",
        int(qa["patches_under_25m2"].sum())
    )

    print(
        "< 50 m²:",
        int(qa["patches_under_50m2"].sum())
    )

    print(
        "< 100 m²:",
        int(qa["patches_under_100m2"].sum())
    )

    print("\nSmallest component found:")

    smallest_row = qa.loc[
        qa["smallest_patch_m2"].idxmin()
    ]

    print(
        smallest_row[
            [
                "wetland_id",
                "observation_date",
                "patch_count",
                "smallest_patch_m2",
                "lpi_percent",
            ]
        ].to_string()
    )

    if not suspicious.empty:

        print(
            "\nSuspicious multi-patch observations:"
        )

        print(
            suspicious[
                [
                    "wetland_id",
                    "observation_date",
                    "patch_count",
                    "smallest_patch_m2",
                    "lpi_percent",
                ]
            ]
            .sort_values(
                "smallest_patch_m2"
            )
            .to_string(index=False)
        )

    print("\nOutputs:")

    print(OUTPUT_CSV)
    print(SUSPICIOUS_CSV)


if __name__ == "__main__":
    main()