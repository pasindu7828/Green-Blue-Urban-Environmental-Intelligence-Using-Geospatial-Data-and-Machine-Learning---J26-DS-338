
from pathlib import Path
from datetime import datetime, timezone
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy import ndimage


# ============================================================
# PART 4 - SEASONAL WATERLOGGING SUSCEPTIBILITY
# SCRIPT 09 - URBAN BUILDING FEATURES
#
# Generates:
# 1. Approximate building footprint coverage per 30m cell
# 2. Building density in a 300m circular neighbourhood
#
# Uses 5m subpixel rasterization for footprint estimation.
# This does not improve the accuracy of the source geometry.
#
# All outputs use the established Script 06 study grid.
# No susceptibility weights or labels assigned.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

OSM_FILE = (
    ROOT / "data/raw/osm/"
    "Kaduwela_OSM_Roads_Buildings_2026.gpkg"
)

DATA_DIR = ROOT / "data/processed/part4_waterlogging"
METRIC_DIR = ROOT / "outputs/metrics/part4_waterlogging"
REPORT_DIR = ROOT / "outputs/reports/part4_waterlogging"

MASK_FILE = DATA_DIR / "06_Kaduwela_Study_Mask_30m.tif"

TARGET_CRS = "EPSG:32644"
CELL_SIZE_M = 30
SUBPIXEL_SIZE_M = 5
SCALE = CELL_SIZE_M // SUBPIXEL_SIZE_M
RADIUS_M = 300
NODATA = -9999.0


def write_raster(path, data, reference):
    profile = reference.profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=NODATA,
        compress="deflate",
        predictor=3,
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def stats(values, study_mask):
    selected = values[study_mask]

    if not np.isfinite(selected).all():
        raise ValueError("Invalid study-area feature values")

    return {
        "minimum": round(float(selected.min()), 6),
        "maximum": round(float(selected.max()), 6),
        "mean": round(float(selected.mean()), 6),
        "median": round(float(np.median(selected)), 6),
        "p95": round(
            float(np.percentile(selected, 95)), 6
        ),
    }


def main():

    print("=" * 70)
    print("PART 4 - URBAN BUILDING FEATURES")
    print("SCRIPT 09")
    print("=" * 70)

    for folder in (DATA_DIR, METRIC_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load OSM input
    # --------------------------------------------------------

    osm = gpd.read_file(OSM_FILE)

    if osm.crs is None:
        raise ValueError("OSM CRS missing")

    if (
        osm.geometry.isna().any()
        or osm.geometry.is_empty.any()
        or not osm.geometry.is_valid.all()
    ):
        raise ValueError("Invalid or missing OSM geometries")

    geometry_type = osm.geometry.geom_type

    polygon_mask = geometry_type.isin([
        "Polygon", "MultiPolygon"
    ])

    line_mask = geometry_type.isin([
        "LineString", "MultiLineString"
    ])

    building_tags = (
        osm["building"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    valid_building_tag = (
        (building_tags != "")
        & (~building_tags.isin([
            "no", "false"
        ]))
    )

    buildings = osm[
        polygon_mask & valid_building_tag
    ].copy()

    highway_lines = osm[
        line_mask & osm["highway"].notna()
    ].copy()

    if buildings.empty:
        raise ValueError("No building polygons found")

    buildings = buildings.to_crs(TARGET_CRS)

    print("\nBUILDING FEATURES")
    print("Building polygons:", len(buildings))
    print(
        "Excluded non-polygon building tags:",
        int((valid_building_tag & ~polygon_mask).sum())
    )

    print("\nROAD INVENTORY")
    print("Highway-tagged lines:", len(highway_lines))
    print(
        "Highway tags on non-lines:",
        int((osm["highway"].notna() & ~line_mask).sum())
    )

    # --------------------------------------------------------
    # 2. Load the existing 30m common analysis grid
    # --------------------------------------------------------

    with rasterio.open(MASK_FILE) as src:

        if str(src.crs) != TARGET_CRS:
            raise ValueError("Incorrect mask CRS")

        if (
            abs(src.transform.a - CELL_SIZE_M) > 1e-6
            or abs(src.transform.e + CELL_SIZE_M) > 1e-6
        ):
            raise ValueError("Unexpected grid resolution")

        study_mask = src.read(1) == 1
        height, width = src.shape

        study_pixels = int(study_mask.sum())

        if study_pixels != 94431:
            raise ValueError(
                "Study mask differs from Script 06"
            )

        # ----------------------------------------------------
        # 3. Estimate building footprint fraction
        # ----------------------------------------------------

        high_transform = (
            src.transform
            * Affine.scale(1 / SCALE, 1 / SCALE)
        )

        high_shape = (
            height * SCALE,
            width * SCALE,
        )

        print("\nRasterizing building footprints...")
        print(
            "Subpixel resolution:",
            SUBPIXEL_SIZE_M, "m"
        )

        high_buildings = rasterize(
            (
                (geometry, 1)
                for geometry in buildings.geometry
            ),
            out_shape=high_shape,
            transform=high_transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )

        # Approximate building area within each 30m cell.
        building_fraction = (
            high_buildings
            .reshape(height, SCALE, width, SCALE)
            .mean(axis=(1, 3))
            .astype(np.float64)
        )

        if (
            np.any(building_fraction < 0)
            or np.any(building_fraction > 1)
        ):
            raise ValueError(
                "Invalid building footprint fractions"
            )

        # ----------------------------------------------------
        # 4. Circular neighbourhood building density
        # ----------------------------------------------------

        radius_cells = RADIUS_M // CELL_SIZE_M

        yy, xx = np.ogrid[
            -radius_cells:radius_cells + 1,
            -radius_cells:radius_cells + 1
        ]

        footprint = (
            (xx * xx + yy * yy)
            <= radius_cells * radius_cells
        ).astype(np.float64)

        # Restrict neighbourhood calculations to the
        # study-area domain to avoid assuming unmapped
        # areas beyond Kaduwela have no buildings.
        numerator = ndimage.convolve(
            building_fraction * study_mask,
            footprint,
            mode="constant",
            cval=0,
        )

        denominator = ndimage.convolve(
            study_mask.astype(np.float64),
            footprint,
            mode="constant",
            cval=0,
        )

        if np.any(denominator[study_mask] <= 0):
            raise ValueError(
                "Invalid neighbourhood support"
            )

        local_density = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )

        if np.any(
            (local_density[study_mask] < 0)
            | (local_density[study_mask] > 1)
        ):
            raise ValueError(
                "Invalid local building density"
            )

        # Convert fractions to percentages.
        footprint_percent = building_fraction * 100
        density_percent = local_density * 100

        footprint_stats = stats(
            footprint_percent,
            study_mask,
        )

        density_stats = stats(
            density_percent,
            study_mask,
        )

        # ----------------------------------------------------
        # 5. Restrict outputs to Kaduwela study cells
        # ----------------------------------------------------

        footprint_output = np.full(
            src.shape,
            NODATA,
            dtype=np.float32,
        )

        density_output = np.full(
            src.shape,
            NODATA,
            dtype=np.float32,
        )

        footprint_output[study_mask] = (
            footprint_percent[study_mask]
        )

        density_output[study_mask] = (
            density_percent[study_mask]
        )

        footprint_path = (
            DATA_DIR
            / "09_OSM_Building_Footprint_Percent_30m.tif"
        )

        density_path = (
            DATA_DIR
            / "09_OSM_Building_Density_300m_30m.tif"
        )

        write_raster(
            footprint_path,
            footprint_output,
            src,
        )

        write_raster(
            density_path,
            density_output,
            src,
        )

    # --------------------------------------------------------
    # 6. Save road-class inventory
    # --------------------------------------------------------

    road_counts = (
        highway_lines["highway"]
        .astype("string")
        .value_counts()
        .rename_axis("highway_class")
        .reset_index(name="feature_count")
    )

    road_path = (
        METRIC_DIR
        / "script09_osm_road_class_inventory.csv"
    )

    road_counts.to_csv(
        road_path,
        index=False,
    )

    # --------------------------------------------------------
    # 7. Save feature statistics and methodology
    # --------------------------------------------------------

    metrics_path = (
        METRIC_DIR
        / "script09_urban_building_statistics.csv"
    )

    metric_rows = [
        {
            "feature": "building_footprint_percent",
            **footprint_stats,
        },
        {
            "feature": "building_density_300m_percent",
            **density_stats,
        },
    ]

    pd.DataFrame(metric_rows).to_csv(
        metrics_path,
        index=False,
    )

    report_path = (
        REPORT_DIR
        / "script09_urban_building_validation.json"
    )

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "Urban building feature preparation",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "source": str(OSM_FILE.relative_to(ROOT)),
        "source_mapping_year": 2026,
        "analysis_crs": TARGET_CRS,
        "grid_resolution_m": CELL_SIZE_M,
        "subpixel_estimation_resolution_m":
            SUBPIXEL_SIZE_M,
        "study_pixels": study_pixels,
        "building_polygon_count": len(buildings),
        "highway_line_count": len(highway_lines),
        "building_method":
            "Rasterize mapped building footprints "
            "at 5m and aggregate to approximate "
            "percentage coverage of 30m cells.",
        "density_method":
            "Mean 30m building footprint fraction "
            "within a circular 300m neighbourhood, "
            "restricted to sampled study-area cells.",
        "density_radius_m": RADIUS_M,
        "footprint_statistics": footprint_stats,
        "density_statistics": density_stats,
        "outputs": {
            "footprint_raster":
                str(footprint_path.relative_to(ROOT)),
            "density_raster":
                str(density_path.relative_to(ROOT)),
            "road_inventory":
                str(road_path.relative_to(ROOT)),
            "statistics":
                str(metrics_path.relative_to(ROOT)),
        },
        "scientific_limitations": [
            "OSM 2026 footprints are not historical "
            "2018 building observations.",
            "OSM building completeness is not "
            "independently verified.",
            "Building footprint coverage is a "
            "proxy for development intensity, "
            "not measured impervious surface.",
            "The 5m rasterization is numerical "
            "area approximation, not 5m source "
            "mapping accuracy.",
            "Near the study boundary, neighbourhood "
            "density considers only sampled Kaduwela "
            "cells, not the full external neighbourhood.",
            "Road classes are inventoried only; "
            "no road susceptibility factor is created.",
            "Urban density may correlate strongly "
            "with existing built-up land cover.",
            "No susceptibility weights, ML labels "
            "or predictions are produced.",
        ],
        "raw_data_modified": False,
        "status": "URBAN_BUILDING_FEATURE_PREPARATION_PASS",
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 09 - SUMMARY")
    print("=" * 70)

    print("Study-area pixels:", study_pixels)
    print("Building polygons processed:", len(buildings))
    print(
        "Building coverage range:",
        footprint_stats["minimum"],
        "to",
        footprint_stats["maximum"],
        "%"
    )
    print(
        "300m density range:",
        density_stats["minimum"],
        "to",
        density_stats["maximum"],
        "%"
    )

    print("\nSaved:", footprint_path)
    print("Saved:", density_path)
    print("Saved:", road_path)
    print("Saved:", metrics_path)
    print("Saved:", report_path)

    print("\nURBAN BUILDING FEATURES: PASS")
    print("ROAD DENSITY: NOT GENERATED")
    print("SUSCEPTIBILITY MODEL: NOT STARTED")
    print("RAW SOURCE DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
