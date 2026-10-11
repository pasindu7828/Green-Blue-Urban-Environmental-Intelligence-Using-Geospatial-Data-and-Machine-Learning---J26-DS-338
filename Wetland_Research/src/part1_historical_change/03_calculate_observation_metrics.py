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

OUTPUT_DIR = PROCESSED_DIR

INPUT_GPKG = (
    PROCESSED_DIR
    / "historical_wetland_patches_utm44n.gpkg"
)

OBSERVATION_CSV = (
    OUTPUT_DIR
    / "historical_wetland_observation_metrics.csv"
)

OBSERVATION_GPKG = (
    OUTPUT_DIR
    / "historical_wetland_observations_utm44n.gpkg"
)

OBSERVATION_GEOJSON = (
    OUTPUT_DIR
    / "historical_wetland_observations_wgs84.geojson"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_polygon_parts(geometry):
    """
    Extract polygon components from Polygon,
    MultiPolygon or GeometryCollection geometry.
    """

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


def calculate_observation_metrics(group):
    """
    Combine all patches belonging to one wetland
    on one observation date and calculate
    fragmentation-related metrics.
    """

    geometries = [
        geometry
        for geometry in group.geometry
        if geometry is not None
        and not geometry.is_empty
    ]

    if not geometries:
        return None

    # Dissolve all geometries from the same
    # wetland and date.
    dissolved_geometry = unary_union(
        geometries
    )

    # Extract actual connected polygon components.
    polygon_parts = extract_polygon_parts(
        dissolved_geometry
    )

    if not polygon_parts:
        return None

    patch_areas_m2 = [
        polygon.area
        for polygon in polygon_parts
    ]

    total_area_m2 = sum(
        patch_areas_m2
    )

    total_area_ha = (
        total_area_m2 / 10000.0
    )

    largest_patch_m2 = max(
        patch_areas_m2
    )

    largest_patch_ha = (
        largest_patch_m2 / 10000.0
    )

    patch_count = len(
        polygon_parts
    )

    total_perimeter_m = sum(
        polygon.length
        for polygon in polygon_parts
    )

    # Largest Patch Index
    if total_area_m2 > 0:
        lpi_percent = (
            largest_patch_m2
            / total_area_m2
        ) * 100.0
    else:
        lpi_percent = None

    # Simple fragmentation indicator:
    # number of patches per hectare.
    if total_area_ha > 0:
        patch_density_per_ha = (
            patch_count
            / total_area_ha
        )
    else:
        patch_density_per_ha = None

    return {
        "wetland_id":
            group["wetland_id"].iloc[0],

        "observation_date":
            group["observation_date"].iloc[0],

        "year":
            int(group["year"].iloc[0]),

        "month":
            int(group["month"].iloc[0]),

        "day":
            int(group["day"].iloc[0]),

        "season":
            group["season"].iloc[0],

        "source_placemark_count":
            len(group),

        "patch_count":
            patch_count,

        "total_area_m2":
            total_area_m2,

        "total_area_ha":
            total_area_ha,

        "largest_patch_m2":
            largest_patch_m2,

        "largest_patch_ha":
            largest_patch_ha,

        "lpi_percent":
            lpi_percent,

        "total_perimeter_m":
            total_perimeter_m,

        "patch_density_per_ha":
            patch_density_per_ha,

        "geometry":
            dissolved_geometry,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_GPKG.exists():

        raise FileNotFoundError(
            "Standardized wetland GeoPackage "
            f"not found:\n{INPUT_GPKG}"
        )

    # --------------------------------------------------------
    # LOAD STANDARDIZED PATCHES
    # --------------------------------------------------------

    patches = gpd.read_file(
        INPUT_GPKG,
        layer="historical_wetland_patches",
    )

    print(
        "\nLoaded standardized patches:",
        len(patches),
    )

    # --------------------------------------------------------
    # GROUP BY WETLAND + OBSERVATION DATE
    # --------------------------------------------------------

    observation_rows = []

    grouped = patches.groupby(
        [
            "wetland_id",
            "observation_date",
        ],
        sort=True,
    )

    for _, group in grouped:

        result = calculate_observation_metrics(
            group
        )

        if result is not None:
            observation_rows.append(
                result
            )

    # --------------------------------------------------------
    # CREATE OBSERVATION GEODATAFRAME
    # --------------------------------------------------------

    observations = gpd.GeoDataFrame(
        observation_rows,
        geometry="geometry",
        crs=patches.crs,
    )

    observations = observations.sort_values(
        [
            "wetland_id",
            "observation_date",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # BASIC DATA QUALITY CHECKS
    # --------------------------------------------------------

    zero_area_count = int(
        (
            observations["total_area_ha"] <= 0
        ).sum()
    )

    lpi_out_of_range = int(
        (
            (
                observations["lpi_percent"] < 0
            )
            |
            (
                observations["lpi_percent"] > 100
            )
        ).sum()
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_columns = [
        "wetland_id",
        "observation_date",
        "year",
        "month",
        "day",
        "season",
        "source_placemark_count",
        "patch_count",
        "total_area_m2",
        "total_area_ha",
        "largest_patch_m2",
        "largest_patch_ha",
        "lpi_percent",
        "total_perimeter_m",
        "patch_density_per_ha",
    ]

    observations[
        csv_columns
    ].to_csv(
        OBSERVATION_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE METRIC CRS SPATIAL DATA
    # --------------------------------------------------------

    observations.to_file(
        OBSERVATION_GPKG,
        layer="historical_wetland_observations",
        driver="GPKG",
    )

    # --------------------------------------------------------
    # SAVE FRONTEND / WEB-MAP VERSION
    # --------------------------------------------------------

    observations_wgs84 = (
        observations.to_crs(
            "EPSG:4326"
        )
    )

    observations_wgs84.to_file(
        OBSERVATION_GEOJSON,
        driver="GeoJSON",
    )

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "HISTORICAL WETLAND OBSERVATION METRICS"
    )

    print(
        "========================================"
    )

    print(
        f"Wetlands represented: "
        f"{observations['wetland_id'].nunique()}"
    )

    print(
        f"Total wetland-date observations: "
        f"{len(observations)}"
    )

    print(
        f"Earliest observation: "
        f"{observations['observation_date'].min()}"
    )

    print(
        f"Latest observation: "
        f"{observations['observation_date'].max()}"
    )

    print(
        f"Zero-area observations: "
        f"{zero_area_count}"
    )

    print(
        f"LPI values outside 0-100: "
        f"{lpi_out_of_range}"
    )

    print(
        "\nTotal area range:"
    )

    print(
        f"Minimum: "
        f"{observations['total_area_ha'].min():.4f} ha"
    )

    print(
        f"Maximum: "
        f"{observations['total_area_ha'].max():.4f} ha"
    )

    print(
        "\nPatch count range:"
    )

    print(
        f"Minimum: "
        f"{observations['patch_count'].min()}"
    )

    print(
        f"Maximum: "
        f"{observations['patch_count'].max()}"
    )

    print(
        "\nOutputs:"
    )

    print(
        f"Metrics CSV:\n{OBSERVATION_CSV}"
    )

    print(
        f"\nMetric GeoPackage:\n{OBSERVATION_GPKG}"
    )

    print(
        f"\nFrontend GeoJSON:\n{OBSERVATION_GEOJSON}"
    )

    # --------------------------------------------------------
    # SHOW FIVE LARGEST OBSERVATIONS FOR QA
    # --------------------------------------------------------

    print(
        "\nFive largest observations "
        "(for QA review):"
    )

    print(
        observations[
            [
                "wetland_id",
                "observation_date",
                "total_area_ha",
                "patch_count",
                "lpi_percent",
            ]
        ]
        .nlargest(
            5,
            "total_area_ha",
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()