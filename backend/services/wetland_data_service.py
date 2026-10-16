import csv
import json
import geopandas as gpd
import rasterio
import numpy as np
from io import BytesIO
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.warp import transform_bounds

from backend.config import (
    WETLAND_PART1_SUMMARY_FILE,
    WETLAND_PART1_MAP_FILE,
    WETLAND_PART1_VALIDATION_FILE,
    WETLAND_PART2_PRIMARY_MAP_FILE,
    WETLAND_PART2_ALL_CANDIDATES_FILE,
    WETLAND_PART2_GN_SUMMARY_FILE,
    WETLAND_PART3_GN_SUMMARY_FILE,
    WETLAND_PART3_2025_FILE,
    WETLAND_PART3_2030_FILE,
    WETLAND_PART3_GAIN_FILE,
    WETLAND_PART3_LOSS_FILE,
    WETLAND_PART3_PERSISTENCE_FILE,
    WETLAND_PART4_GN_SUMMARY_FILE,
    WETLAND_PART4_MAIN_CLASS_FILE,
    WETLAND_PART4_TERRAIN_CLASS_FILE,
    WETLAND_PART4_SCENARIO_SPAN_FILE,
)


# ============================================================
# PART 1 - HISTORICAL WETLAND CHANGE
# ============================================================

def get_part1_summary():
    """
    Read the existing historical wetland summary.

    Returns one record for each monitored wetland.
    No research calculations are repeated.
    """
    with WETLAND_PART1_SUMMARY_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


def get_part1_map():
    """
    Read the existing WGS84 GeoJSON containing
    the latest boundaries of the 25 wetlands.
    """
    with WETLAND_PART1_MAP_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)


def get_part1_validation():
    """
    Read the saved Part 1 backend validation checks.
    """
    with WETLAND_PART1_VALIDATION_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


# ============================================================
# PART 2 - CANDIDATE WETLAND DETECTION
# ============================================================

def get_part2_primary_candidates():
    """
    Return the 48 primary candidate wetland polygons.

    Includes reviewed and unreviewed candidate locations.
    """
    with WETLAND_PART2_PRIMARY_MAP_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)


def get_part2_all_candidates():
    """
    Return the broader candidate dataset containing
    1,097 candidate polygons.

    These are not all confirmed wetlands.
    """
    with WETLAND_PART2_ALL_CANDIDATES_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)


def get_part2_gn_summary():
    """
    Return candidate wetland statistics
    for Kaduwela's 56 GN divisions.
    """
    with WETLAND_PART2_GN_SUMMARY_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)


# ============================================================
# PART 3 - FUTURE WETLAND PROJECTION
# ============================================================

def get_part3_gn_summary():
    """
    Return the 56 GN divisions with historical wetland
    statistics and exploratory 2030 scenario results.

    Reads existing research outputs without recalculating
    or modifying any data.
    """

    gdf = gpd.read_file(WETLAND_PART3_GN_SUMMARY_FILE)

    # GeoJSON web maps require WGS84 coordinates.
    if gdf.crs is None:
        raise ValueError("Part 3 GN layer has no defined CRS")

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Convert GeoPandas data into a JSON-compatible
    # GeoJSON FeatureCollection.
    return json.loads(
        gdf.to_json(na="null", drop_id=True)
    )


# ============================================================
# PART 3 - 2025 AND 2030 WETLAND MAPS
# ============================================================

def get_part3_wetland_map(year: int):
    """
    Return the existing wetland polygons for 2025 or 2030.

    2025: Estimated wetland distribution.
    2030: Exploratory CA-Markov scenario.

    All polygons are preserved.
    No modelling or prediction is repeated.
    """

    files = {
        2025: WETLAND_PART3_2025_FILE,
        2030: WETLAND_PART3_2030_FILE,
    }

    if year not in files:
        raise ValueError(
            "Only 2025 and 2030 wetland maps are available"
        )

    gdf = gpd.read_file(files[year])

    if gdf.crs is None:
        raise ValueError(
            f"Part 3 wetland {year} layer has no defined CRS"
        )

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return json.loads(
        gdf.to_json(na="null", drop_id=True)
    )


# ============================================================
# PART 3 - WETLAND SCENARIO CHANGE MAPS
# ============================================================

def get_part3_scenario_map(change_type: str):
    """
    Return the existing wetland scenario change polygons.

    Supported:
    - gain: Potential wetland gain in the 2030 scenario
    - loss: Potential wetland loss in the 2030 scenario
    - persistence: Wetlands present in both 2025 and 2030

    All polygons are preserved.
    No model calculations are repeated.
    """

    files = {
        "gain": WETLAND_PART3_GAIN_FILE,
        "loss": WETLAND_PART3_LOSS_FILE,
        "persistence": WETLAND_PART3_PERSISTENCE_FILE,
    }

    if change_type not in files:
        raise ValueError(
            "Change type must be gain, loss, or persistence"
        )

    gdf = gpd.read_file(files[change_type])

    if gdf.crs is None:
        raise ValueError(
            f"Part 3 {change_type} layer has no defined CRS"
        )

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return json.loads(
        gdf.to_json(na="null", drop_id=True)
    )


# ============================================================
# PART 4 - SEASONAL WATERLOGGING SUSCEPTIBILITY
# ============================================================

def get_part4_gn_summary():
    """
    Return GN-level waterlogging susceptibility statistics
    and geographic boundaries as WGS84 GeoJSON.
    """

    gdf = gpd.read_file(
        WETLAND_PART4_GN_SUMMARY_FILE,
        layer="gn_susceptibility"
    )

    if gdf.crs is None:
        raise ValueError("Part 4 GN layer has no defined CRS")

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return json.loads(
        gdf.to_json(na="null", drop_id=True)
    )


# ============================================================
# PART 4 - RASTER MAP METADATA SERVICE
# ============================================================

def get_part4_raster_metadata(map_type: str):
    """
    Return metadata required to display a Part 4 raster
    as a geographically aligned frontend map overlay.
    """

    files = {
        "main": WETLAND_PART4_MAIN_CLASS_FILE,
        "terrain": WETLAND_PART4_TERRAIN_CLASS_FILE,
        "span": WETLAND_PART4_SCENARIO_SPAN_FILE,
    }

    if map_type not in files:
        raise ValueError(
            "Map type must be main, terrain, or span"
        )

    susceptibility_legend = {
        1: "Low",
        2: "Moderate",
        3: "High",
        4: "Very High",
    }

    span_legend = {
        0: "Same class across scenarios",
        1: "One-class difference",
        2: "Two-class difference",
        3: "Three-class difference",
    }

    path = files[map_type]

    with rasterio.open(path) as src:

        if src.crs is None:
            raise ValueError(
                f"Raster {map_type} has no defined CRS"
            )

        west, south, east, north = transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds,
            densify_pts=21,
        )

        legend = (
            span_legend
            if map_type == "span"
            else susceptibility_legend
        )

        return {
            "map_type": map_type,
            "source_crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "bounds_wgs84": {
                "west": west,
                "south": south,
                "east": east,
                "north": north,
            },
            "legend": legend,
            "interpretation": (
                "Scenario classification disagreement"
                if map_type == "span"
                else "Relative waterlogging susceptibility"
            ),
        }


# ============================================================
# PART 4 - TRANSPARENT RASTER MAP RENDERING
# ============================================================

def get_part4_raster_png(map_type: str):
    """
    Render an existing Part 4 categorical GeoTIFF as a
    transparent WGS84 PNG overlay.

    The research raster is never modified.
    """

    files = {
        "main": WETLAND_PART4_MAIN_CLASS_FILE,
        "terrain": WETLAND_PART4_TERRAIN_CLASS_FILE,
        "span": WETLAND_PART4_SCENARIO_SPAN_FILE,
    }

    if map_type not in files:
        raise ValueError(
            "Map type must be main, terrain, or span"
        )

    # RGBA: red, green, blue, opacity
    susceptibility_colors = {
        1: (46, 160, 67, 190),     # Low - Green
        2: (255, 210, 60, 190),    # Moderate - Yellow
        3: (245, 130, 35, 190),    # High - Orange
        4: (215, 45, 45, 190),     # Very High - Red
    }

    span_colors = {
        0: (90, 175, 105, 190),    # Same class
        1: (255, 215, 70, 190),    # One-class difference
        2: (245, 130, 35, 190),    # Two-class difference
        3: (200, 40, 50, 190),     # Three-class difference
    }

    colors = (
        span_colors
        if map_type == "span"
        else susceptibility_colors
    )

    with rasterio.open(files[map_type]) as src:

        if src.crs is None:
            raise ValueError(
                f"Raster {map_type} has no defined CRS"
            )

        if src.nodata is None:
            raise ValueError(
                f"Raster {map_type} has no defined NoData value"
            )

        west, south, east, north = transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds,
            densify_pts=21,
        )

        target_transform = from_bounds(
            west,
            south,
            east,
            north,
            src.width,
            src.height,
        )

        destination = np.full(
            (src.height, src.width),
            int(src.nodata),
            dtype=np.uint8,
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=target_transform,
            dst_crs="EPSG:4326",
            dst_nodata=src.nodata,
            resampling=Resampling.nearest,
        )

        # Transparent background for NoData pixels
        rgba = np.zeros(
            (src.height, src.width, 4),
            dtype=np.uint8,
        )

        # Apply colours to valid categorical values
        for class_value, color in colors.items():
            rgba[destination == class_value] = color

        # Encode PNG in memory without writing to disk
        image = Image.fromarray(rgba, mode="RGBA")
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        return buffer.getvalue()
