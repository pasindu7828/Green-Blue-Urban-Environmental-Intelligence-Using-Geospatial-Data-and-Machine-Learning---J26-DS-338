import csv
import json
import geopandas as gpd
import rasterio
import numpy as np

from shapely.ops import unary_union
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
    WETLAND_PART1_HISTORICAL_OBSERVATIONS_FILE,
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
# PART 3 - FOCUSED 2030 SCENARIO MAP
# ============================================================

def get_part3_focused_scenario_map(year=2030):
    if year not in (2025, 2030):
        raise ValueError("Focused scenario supports only 2025 or 2030")
    """
    Return a selected-area view of existing 2030 scenario
    patches associated with historical wetlands and primary
    screened candidates.

    Selection:
      - 25 historical wetland polygons
      - 48 primary candidate polygons
      - Provisional 50 m spatial buffer
      - Existing five-pixel patch display rule

    Selected patches retain their full original geometry.

    This is a visualization filter, not model correction,
    retraining, prediction validation or confirmation of
    wetland presence.
    """

    metric_crs = "EPSG:32644"
    buffer_m = 50

    # Load existing research datasets.
    historical_data = get_part1_map()
    candidate_data = get_part2_primary_candidates()
    scenario_data = get_part3_wetland_map(year)

    def to_metric_gdf(data):
        features = data.get("features", [])

        gdf = gpd.GeoDataFrame.from_features(
            features,
            crs="EPSG:4326",
        )

        return gdf.to_crs(metric_crs)

    historical = to_metric_gdf(historical_data)
    candidates = to_metric_gdf(candidate_data)
    scenario = to_metric_gdf(scenario_data)

    # Stop if required input datasets are unavailable.
    if (
        historical.empty
        or candidates.empty
        or scenario.empty
    ):
        raise ValueError(
            "Focused scenario requires historical, "
            "candidate and 2030 scenario polygons."
        )

    # Combine the selected areas without assuming that
    # 25 + 48 represents 73 independent wetland sites.
    focus_area = unary_union(
        list(historical.geometry)
        + list(candidates.geometry)
    )

    # Expand the selection area by 50 metres.
    comparison_area = focus_area.buffer(buffer_m)

    # Apply the existing five-pixel display rule.
    display_eligible = scenario.loc[
        scenario["display_ge5"]
        .fillna(0)
        .astype(int)
        == 1
    ].copy()

    # Select full polygons intersecting the comparison area.
    selected = display_eligible.loc[
        display_eligible.geometry.intersects(
            comparison_area
        )
    ].copy()

    # Convert selected geometries back to WGS84.
    selected = selected.to_crs(epsg=4326)

    result = json.loads(
        selected.to_json(na="null", drop_id=True)
    )

    # Explain the origin and limitations of this selection.
    result["metadata"] = {
        "scenario_year": year,
        "selection_type": "focused_spatial_intersection",
        "historical_wetland_count": len(historical),
        "primary_candidate_count": len(candidates),
        "buffer_distance_m": buffer_m,
        "minimum_display_pixels": 5,
        "total_scenario_patches": len(scenario),
        "display_eligible_patches": len(display_eligible),
        "selected_patches": len(selected),
        "geometry_handling": "full_original_patch_polygons",
        "validation_status": "exploratory_not_validated",
        "note": (
            "This is a selected-area visualization of the "
            f"existing exploratory {year} CA-Markov model output. "
            "Other model-generated patches are excluded from "
            "this focused response, not from the source data. "
            "The 50 m buffer is provisional and is not a "
            "validated ecological matching distance."
        ),
    }

    return result

# ============================================================
# PART 3 — FOCUSED GN SCENARIO AREA SUMMARY
# ============================================================

def get_part3_focused_gn_summary():
    """
    Calculate GN-wise 2025 and 2030 polygon areas from
    the focused model-patch selections.

    Uses full selected patch geometries, clipped to GN
    boundaries for measurement.

    These are exploratory model-display statistics,
    not validated wetland forecasts or field observations.
    """

    metric_crs = "EPSG:32644"

    gn_data = get_part3_gn_summary()
    data_2025 = get_part3_focused_scenario_map(2025)
    data_2030 = get_part3_focused_scenario_map(2030)

    def metric_gdf(data):
        return gpd.GeoDataFrame.from_features(
            data["features"],
            crs="EPSG:4326",
        ).to_crs(metric_crs)

    gn_gdf = metric_gdf(gn_data)
    patches_2025 = metric_gdf(data_2025)
    patches_2030 = metric_gdf(data_2030)

    def area_inside_gn(patches, boundary):
        intersecting = patches.loc[
            patches.geometry.intersects(boundary)
        ]

        if intersecting.empty:
            return 0.0, 0

        combined = unary_union(
            list(intersecting.geometry)
        )

        clipped = combined.intersection(boundary)

        return clipped.area / 10000, len(intersecting)


    def clipped_patch_geometry(patches, boundary):
        """Union selected patches and clip them to a GN."""
        intersecting = patches.loc[
            patches.geometry.intersects(boundary)
        ]

        if intersecting.empty:
            return None

        return unary_union(
            list(intersecting.geometry)
        ).intersection(boundary)

    rows = []

    for _, gn in gn_gdf.iterrows():
        boundary = gn.geometry

        area_2025, count_2025 = area_inside_gn(
            patches_2025, boundary
        )

        area_2030, count_2030 = area_inside_gn(
            patches_2030, boundary
        )


        # Compare the actual spatial footprints within this GN.
        shape_2025 = clipped_patch_geometry(
            patches_2025, boundary
        )
        shape_2030 = clipped_patch_geometry(
            patches_2030, boundary
        )

        if shape_2025 is None:
            loss_ha = 0.0
            gain_ha = (
                shape_2030.area / 10000
                if shape_2030 is not None else 0.0
            )
        elif shape_2030 is None:
            loss_ha = shape_2025.area / 10000
            gain_ha = 0.0
        else:
            loss_ha = (
                shape_2025.difference(shape_2030).area
                / 10000
            )
            gain_ha = (
                shape_2030.difference(shape_2025).area
                / 10000
            )

        # Keep GNs with selected coverage in either year.
        if count_2025 == 0 and count_2030 == 0:
            continue

        rows.append({
            "gn_source_id": str(gn["gn_source_id"]),
            "gn_name": str(gn["gn_name"]),
            "wetland_2025_ha": round(area_2025, 4),
            "wetland_2030_ha": round(area_2030, 4),
            "scenario_net_change_ha": round(
                area_2030 - area_2025, 4
            ),

            "scenario_loss_ha": round(loss_ha, 4),
            "scenario_gain_ha": round(gain_ha, 4),
            "focused_2025_patch_count": count_2025,
            "focused_2030_patch_count": count_2030,
            "geometry": boundary,
        })

    result_gdf = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs=metric_crs,
    ).to_crs(epsg=4326)

    result = json.loads(
        result_gdf.to_json(na="null", drop_id=True)
    )

    result["metadata"] = {
        "summary_type": "focused_gn_scenario_area",
        "gn_divisions_included": len(rows),
        "selected_2025_patches": len(patches_2025),
        "selected_2030_patches": len(patches_2030),
        "buffer_distance_m": 50,
        "minimum_display_pixels": 5,
        "area_method": "union_of_selected_polygons_clipped_to_gn",
        "geometry_handling": "full_selected_patch_geometries",
        "validation_status": "exploratory_not_validated",
        "note": (
            "Areas represent selected model-generated patches "
            "using the same focused selection rule in both years. "
            "They are not limited to the 50 m buffer footprint. "
            "Net change is the difference between selected polygon "
            "areas, not a validated prediction of wetland change."
        ),
    }

    return result

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
# PART 3 — INDIVIDUAL WETLAND FUTURE COMPARISON
# ============================================================

def get_part3_individual_future_comparison(wetland_id: str):
    """
    Find exploratory 2030 scenario patches spatially associated
    with the latest mapped boundary of one historical wetland.

    Uses the existing focused 2030 dataset and a provisional
    50 m proximity rule. No research data is modified.
    """

    metric_crs = "EPSG:32644"
    buffer_m = 50

    wetland_id = wetland_id.strip().upper()

    historical_data = get_part1_map()
    scenario_data = get_part3_focused_scenario_map(2030)

    historical = gpd.GeoDataFrame.from_features(
        historical_data["features"],
        crs="EPSG:4326",
    ).to_crs(metric_crs)

    scenario = gpd.GeoDataFrame.from_features(
        scenario_data["features"],
        crs="EPSG:4326",
    ).to_crs(metric_crs)

    selected = historical.loc[
        historical["wetland_id"].astype(str) == wetland_id
    ].copy()

    if len(selected) != 1:
        raise ValueError(
            f"Historical wetland not uniquely found: {wetland_id}"
        )

    wetland = selected.iloc[0]
    boundary = wetland.geometry

    if boundary is None or boundary.is_empty:
        raise ValueError(
            f"Historical wetland geometry unavailable: {wetland_id}"
        )

    # Identify full scenario polygons within the provisional
    # 50 m search area around this historical wetland.
    search_area = boundary.buffer(buffer_m)

    matched = scenario.loc[
        scenario.geometry.intersects(search_area)
    ].copy()


    # Clip the selected 2030 patches to the local 50 m
    # viewing area. This affects API display geometry only.
    # The original research polygons remain unchanged.
    matched["geometry"] = matched.geometry.intersection(
        search_area
    )

    # Exclude empty or zero-area intersections.
    matched = matched.loc[
        matched.geometry.notna()
        & ~matched.geometry.is_empty
        & (matched.geometry.area > 0)
    ].copy()

    historical_geojson = json.loads(
        selected.to_crs(epsg=4326).to_json(
            na="null", drop_id=True
        )
    )

    scenario_geojson = json.loads(
        matched.to_crs(epsg=4326).to_json(
            na="null", drop_id=True
        )
    )

    return {
        "wetland_id": wetland_id,
        "historical": historical_geojson,
        "scenario_2030": scenario_geojson,
        "metadata": {
            "historical_observation_date": str(
                wetland["observation_date"]
            ),
            "matching_method": "provisional_spatial_proximity",
            "buffer_distance_m": buffer_m,
            "source_focused_scenario_patches": len(scenario),
            "matched_scenario_patches": len(matched),
            "minimum_display_pixels": 5,
            "scenario_year": 2030,
            "geometry_handling": "clipped_to_provisional_50m_local_view",
            "validation_status": "exploratory_not_validated",
            "note": (
                "2030 model-generated patches are clipped to "
                "a provisional 50 m local viewing area around "
                "the historical wetland. The clipped shapes "
                "are visualization excerpts, not complete "
                "model patches or predicted future boundaries. "
                "Zero matches do not establish disappearance."
            ),

        },
    }

# ============================================================
# PART 3 — LOCAL 2025 VS 2030 SCENARIO DIFFERENCE METRICS
# ============================================================

def get_part3_local_change_metrics(wetland_id: str):
    """
    Compare complete 2025 and 2030 model outputs within
    the same provisional 50 m area surrounding a selected
    historical wetland.

    These are exploratory local model differences, not
    validated individual wetland predictions.
    """

    from shapely.geometry import GeometryCollection

    metric_crs = "EPSG:32644"
    buffer_m = 50
    wetland_id = wetland_id.strip().upper()

    def to_metric(data):
        return gpd.GeoDataFrame.from_features(
            data["features"],
            crs="EPSG:4326",
        ).to_crs(metric_crs)

    historical = to_metric(get_part1_map())

    selected = historical.loc[
        historical["wetland_id"] == wetland_id
    ]

    if len(selected) != 1:
        raise ValueError(
            f"Historical wetland not uniquely found: {wetland_id}"
        )

    historical_boundary = selected.geometry.iloc[0]

    if historical_boundary is None or historical_boundary.is_empty:
        raise ValueError("Historical wetland geometry is unavailable")

    # Use exactly the same local area for both model years.
    local_area = historical_boundary.buffer(buffer_m)

    def clipped_model_geometry(year):
        patches = to_metric(get_part3_wetland_map(year))

        nearby = patches.loc[
            patches.geometry.intersects(local_area)
        ]

        if nearby.empty:
            return GeometryCollection()

        return unary_union(
            list(nearby.geometry)
        ).intersection(local_area)

    model_2025 = clipped_model_geometry(2025)
    model_2030 = clipped_model_geometry(2030)

    persistence = model_2025.intersection(model_2030)
    loss = model_2025.difference(model_2030)
    gain = model_2030.difference(model_2025)

    def to_ha(geometry):
        return geometry.area / 10000

    area_2025 = to_ha(model_2025)
    area_2030 = to_ha(model_2030)
    persistence_ha = to_ha(persistence)
    loss_ha = to_ha(loss)
    gain_ha = to_ha(gain)
    net_change_ha = area_2030 - area_2025

    consistency_error = abs(
        (gain_ha - loss_ha) - net_change_ha
    )

    if consistency_error > 0.00000001:
        raise ValueError(
            f"Local scenario area consistency failed: {wetland_id}"
        )

    return {
        "wetland_id": wetland_id,
        "historical_observation_date": str(
            selected.iloc[0]["observation_date"]
        ),
        "model_2025_area_ha": round(area_2025, 4),
        "model_2030_area_ha": round(area_2030, 4),
        "persistence_ha": round(persistence_ha, 4),
        "modelled_loss_ha": round(loss_ha, 4),
        "modelled_gain_ha": round(gain_ha, 4),
        "net_change_ha": round(net_change_ha, 4),
        "consistency_error_ha": consistency_error,
        "buffer_distance_m": buffer_m,
        "analysis_method": "complete_model_polygons_clipped_to_same_local_area",
        "validation_status": "exploratory_not_validated",
        "interpretation": (
            "The historical wetland identifies the local "
            "analysis area. All 2025 and 2030 model polygons "
            "are compared within that same area. Results "
            "describe exploratory modelled spatial differences, "
            "not validated change to the individual wetland."
        ),
    }

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


# ============================================================
# PART 1 - HISTORICAL OBSERVATION YEARS
# ============================================================

def get_part1_historical_years():
    with open(
        WETLAND_PART1_HISTORICAL_OBSERVATIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    features = data.get("features", [])

    year_data = {}

    all_wetland_ids = set()

    for feature in features:
        properties = feature.get("properties") or {}

        year = properties.get("year")
        wetland_id = properties.get("wetland_id")

        if year is None or wetland_id is None:
            continue

        year = int(year)
        wetland_id = str(wetland_id)

        all_wetland_ids.add(wetland_id)

        if year not in year_data:
            year_data[year] = {
                "observation_count": 0,
                "wetland_ids": set(),
            }

        year_data[year]["observation_count"] += 1
        year_data[year]["wetland_ids"].add(wetland_id)

    years = []

    for year in sorted(year_data):
        values = year_data[year]

        years.append({
            "year": year,
            "observation_count": values["observation_count"],
            "wetland_count": len(values["wetland_ids"]),
        })

    return {
        "total_observations": len(features),
        "total_wetlands": len(all_wetland_ids),
        "available_years": len(years),
        "years": years,
    }


# ============================================================
# PART 1 - HISTORICAL WETLAND MAP BY YEAR
# ============================================================

def get_part1_historical_map_by_year(year: int):
    """
    Return the latest actual observation within the selected
    year for each monitored wetland.

    No interpolation, prediction or geometry modification.
    """

    with open(
        WETLAND_PART1_HISTORICAL_OBSERVATIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    latest_by_wetland = {}

    for feature in data.get("features", []):
        properties = feature.get("properties") or {}

        observation_year = properties.get("year")
        wetland_id = properties.get("wetland_id")
        observation_date = properties.get("observation_date")

        if (
            observation_year is None
            or not wetland_id
            or not observation_date
        ):
            continue

        if int(observation_year) != year:
            continue

        existing = latest_by_wetland.get(wetland_id)

        if (
            existing is None
            or observation_date
            > existing["properties"]["observation_date"]
        ):
            latest_by_wetland[wetland_id] = feature

    selected_features = [
        latest_by_wetland[wetland_id]
        for wetland_id in sorted(latest_by_wetland)
    ]

    return {
        "type": "FeatureCollection",
        "features": selected_features,
        "metadata": {
            "selected_year": year,
            "wetland_count": len(selected_features),
            "selection_method": "latest_observation_within_year",
        },
    }

# ============================================================
# PART 1 - INDIVIDUAL WETLAND HISTORICAL OBSERVATIONS
# ============================================================

def get_part1_wetland_observations(wetland_id: str):
    """
    Retrieve all cleaned historical observation polygons
    for one monitored wetland.

    Preserves actual observation dates, geometries,
    and research measurements.
    """

    wetland_id = wetland_id.strip().upper()

    with open(
        WETLAND_PART1_HISTORICAL_OBSERVATIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    selected_features = []

    for feature in data.get("features", []):
        properties = feature.get("properties") or {}

        if properties.get("wetland_id") == wetland_id:
            selected_features.append(feature)

    selected_features.sort(
        key=lambda feature:
            feature["properties"]["observation_date"]
    )

    available_years = sorted({
        int(feature["properties"]["year"])
        for feature in selected_features
    })

    return {
        "type": "FeatureCollection",
        "features": selected_features,
        "metadata": {
            "wetland_id": wetland_id,
            "observation_count": len(selected_features),
            "available_years": available_years,
            "observation_dates": [
                feature["properties"]["observation_date"]
                for feature in selected_features
            ],
        },
    }


# ============================================================
# PART 1 - HISTORICAL COMPARISON QUALITY EVIDENCE
# ============================================================

def get_part1_comparison_evidence(
    wetland_id: str,
    before_date: str,
    after_date: str,
):
    import csv

    evidence_file = (
        WETLAND_PART1_HISTORICAL_OBSERVATIONS_FILE.parent
        / "historical_wetland_change_pairs_quality.csv"
    )

    wetland_id = wetland_id.strip().upper()

    with evidence_file.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                row["wetland_id"] == wetland_id
                and row["start_date"] == before_date
                and row["end_date"] == after_date
            ):
                return {
                    "wetland_id": wetland_id,
                    "before_date": before_date,
                    "after_date": after_date,
                    "assessed": True,
                    "comparison_quality": row["comparison_quality"],
                    "same_season": (
                        row["same_season"].strip().lower() == "true"
                    ),
                    "rainfall_similarity": row["rainfall_similarity"],
                    "temporal_context": row["temporal_context"],
                    "manual_qa_required": (
                        row["manual_qa_required"].strip().lower()
                        == "true"
                    ),
                    "area_change_ha": float(row["area_change_ha"]),
                    "area_change_percent": float(
                        row["area_change_percent"]
                    ),
                    "note": (
                        "Quality classification comes from the "
                        "existing historical comparison research."
                    ),
                }

    return {
        "wetland_id": wetland_id,
        "before_date": before_date,
        "after_date": after_date,
        "assessed": False,
        "comparison_quality": None,
        "same_season": None,
        "rainfall_similarity": None,
        "temporal_context": None,
        "manual_qa_required": None,
        "area_change_ha": None,
        "area_change_percent": None,
        "note": (
            "No exact quality-assessed comparison record exists "
            "for the selected observation dates. Raw mapped "
            "area differences are not validated change evidence."
        ),
    }
