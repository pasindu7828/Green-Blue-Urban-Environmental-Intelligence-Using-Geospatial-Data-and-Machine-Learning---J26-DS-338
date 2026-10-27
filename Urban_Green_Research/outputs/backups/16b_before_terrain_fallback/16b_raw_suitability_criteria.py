"""
STEP 16B - RAW 100 m DEVELOPMENT SUITABILITY CRITERIA GENERATION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose
-------
Generate raw spatial measurements for the independent Development
Suitability GIS-MCDA stage.

Raw candidate criteria:
    C1 - Higher-order road proximity
    C2 - Local drivable road-network context
    C3 - Existing development / building context
    C4 - Terrain slope
    C5 - Terrain elevation

Important methodological rules
------------------------------
1. NO AHP weights are assigned here.
2. NO suitability score is calculated here.
3. NO criterion standardization is performed here.
4. NO Urbanization Pressure output is used.
5. NO SHAP/XAI output is used.
6. NO Conservation Priority output is used.
7. NO provisional UDA zoning is used in the core criteria.
8. All 100 m cells are clipped to the official KMC boundary before
   area-based calculations.
9. Boundary-cell denominators use the actual KMC overlap area, not the
   full 10,000 m² cell.
10. Multiple neighbourhood radii are generated as candidate alternatives.
    They are NOT separate weighted MCDA criteria.
"""

from pathlib import Path
import json
import math
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.fill import fillnodata
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import mapping


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "grid"
    / "kaduwela_100m_modelling_grid.gpkg"
)

KMC_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "boundaries"
    / "kmc"
    / "kaduwela_kmc_boundary_utm44n.gpkg"
)

ROADS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "osm"
    / "kaduwela_roads_2026.gpkg"
)

BUILDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "osm"
    / "kaduwela_buildings_2026.gpkg"
)

GEDTM_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "gedtm30_kmc_30m.tif"
)


# =============================================================================
# 2. OUTPUT PATHS
# =============================================================================

RAW_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "development_suitability"
    / "raw_criteria"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

RAW_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    RAW_OUTPUT_DIR
    / "16b_raw_suitability_criteria_100m.csv"
)

OUTPUT_GPKG = (
    RAW_OUTPUT_DIR
    / "16b_raw_suitability_criteria_100m.gpkg"
)

ROAD_CLASS_REPORT = (
    REPORT_DIR
    / "16b_road_class_definition.csv"
)

CRITERIA_DEFINITION_REPORT = (
    REPORT_DIR
    / "16b_raw_criteria_definition.csv"
)

RAW_STATS_REPORT = (
    REPORT_DIR
    / "16b_raw_criteria_descriptive_statistics.csv"
)

QA_REPORT = (
    REPORT_DIR
    / "16b_raw_suitability_criteria_qa.csv"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "16b_raw_suitability_criteria_summary.json"
)


# =============================================================================
# 3. ANALYSIS SETTINGS
# =============================================================================

EXPECTED_EPSG = 32644
EXPECTED_GRID_COUNT = 9240

# Candidate neighbourhood radii.
# These are alternative raw representations and are NOT separate AHP criteria.
ROAD_DENSITY_RADII_M = [
    250,
    500,
    1000,
]

BUILDING_CONTEXT_RADII_M = [
    250,
    500,
]


# =============================================================================
# 4. ROAD HIERARCHY
# =============================================================================

# C1:
# Higher-order road proximity.
#
# Main motorway segments are intentionally excluded because simple proximity
# to a controlled-access motorway does not necessarily provide direct access.
#
# motorway_link is retained because ramps/links represent access locations.

C1_MAJOR_ROAD_CLASSES = {
    "primary",
    "primary_link",
    "secondary",
    "tertiary",
    "tertiary_link",
    "trunk",
    "trunk_link",
    "motorway_link",
}


# C2:
# Broad local drivable road context.
#
# service roads are excluded from the primary definition because OSM service
# roads can include driveways, parking aisles and internal access segments.
#
# motorway mainline is also excluded because it does not represent ordinary
# local street accessibility.

C2_LOCAL_DRIVABLE_CLASSES = {
    "primary",
    "primary_link",
    "secondary",
    "tertiary",
    "tertiary_link",
    "trunk",
    "trunk_link",
    "motorway_link",
    "residential",
    "living_street",
    "unclassified",
}


# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================

def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def geometry_union(geo_series):
    """
    Return union of all geometries with compatibility across GeoPandas versions.
    """

    try:
        return geo_series.union_all()
    except AttributeError:
        return geo_series.unary_union


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def clipped_geometry_series(gdf, boundary_geometry):
    """
    Clip geometries to the analysis boundary using geometric intersection.
    """

    clipped = gdf.copy()

    clipped["geometry"] = clipped.geometry.intersection(
        boundary_geometry
    )

    clipped = clipped[
        ~clipped.geometry.is_empty
        &
        clipped.geometry.notna()
    ].copy()

    clipped.reset_index(
        drop=True,
        inplace=True,
    )

    return clipped


def zone_line_length_m(
    zone_geometry,
    line_gdf,
    spatial_index,
):
    """
    Calculate total line length inside a zone.
    """

    candidate_idx = spatial_index.query(
        zone_geometry,
        predicate="intersects",
    )

    if len(candidate_idx) == 0:
        return 0.0

    candidate_geoms = line_gdf.geometry.iloc[
        candidate_idx
    ]

    intersections = candidate_geoms.intersection(
        zone_geometry
    )

    total_length = float(
        intersections.length.sum()
    )

    return total_length


def zone_building_area_m2(
    zone_geometry,
    building_gdf,
    spatial_index,
):
    """
    Calculate total building footprint area inside a zone.

    Individual building intersections are summed.

    A separate QA check later confirms that calculated proportions do not
    exceed plausible limits.
    """

    candidate_idx = spatial_index.query(
        zone_geometry,
        predicate="intersects",
    )

    if len(candidate_idx) == 0:
        return 0.0

    candidate_geoms = building_gdf.geometry.iloc[
        candidate_idx
    ]

    intersections = candidate_geoms.intersection(
        zone_geometry
    )

    total_area = float(
        intersections.area.sum()
    )

    return total_area


def zone_building_count(
    zone_geometry,
    building_gdf,
    spatial_index,
):
    """
    Number of mapped building geometries intersecting a zone.

    This is retained as a diagnostic variable only.
    It is NOT the primary C3 criterion.
    """

    candidate_idx = spatial_index.query(
        zone_geometry,
        predicate="intersects",
    )

    return int(
        len(candidate_idx)
    )


def load_dem_and_slope(
    raster_path,
):
    """
    Load primary DEM and derive slope in degrees.

    Nodata cells are temporarily filled only for gradient computation.
    Original nodata locations remain nodata in the final slope array.
    """

    with rasterio.open(
        raster_path
    ) as src:

        elevation = src.read(
            1
        ).astype(
            "float64"
        )

        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        nodata = src.nodata

    valid = np.isfinite(
        elevation
    )

    if nodata is not None:
        valid &= (
            elevation
            !=
            nodata
        )

    elevation_clean = elevation.copy()

    elevation_clean[
        ~valid
    ] = np.nan

    # -------------------------------------------------------------------------
    # Fill nodata temporarily for slope calculation.
    # This helps prevent artificial NaN gradients along the clipped raster edge.
    # -------------------------------------------------------------------------

    fill_input = elevation.copy()

    if nodata is not None:
        fill_input[
            ~valid
        ] = nodata

    fill_mask = valid.astype(
        "uint8"
    )

    try:

        elevation_filled = fillnodata(
            fill_input,
            mask=fill_mask,
            max_search_distance=20,
        ).astype(
            "float64"
        )

    except Exception:

        # Fallback if rasterio fill operation is unavailable for some reason.
        elevation_filled = elevation.copy()

        valid_values = elevation[
            valid
        ]

        fallback_value = float(
            np.nanmedian(
                valid_values
            )
        )

        elevation_filled[
            ~valid
        ] = fallback_value

    pixel_width = abs(
        transform.a
    )

    pixel_height = abs(
        transform.e
    )

    dz_dy, dz_dx = np.gradient(
        elevation_filled,
        pixel_height,
        pixel_width,
    )

    slope_deg = np.degrees(
        np.arctan(
            np.sqrt(
                dz_dx ** 2
                +
                dz_dy ** 2
            )
        )
    )

    slope_deg[
        ~valid
    ] = np.nan

    return {
        "elevation": elevation_clean,
        "slope_deg": slope_deg,
        "transform": transform,
        "crs": crs,
        "width": width,
        "height": height,
        "nodata": nodata,
    }


def get_array_values_in_geometry(
    array,
    geometry,
    transform,
):
    """
    Extract raster-cell values whose centres fall inside a polygon.

    A small raster window is used for efficiency.

    If no centre falls inside a very small boundary sliver, all_touched=True
    is used as a fallback.
    """

    minx, miny, maxx, maxy = geometry.bounds

    inverse = ~transform

    col_a, row_a = inverse * (
        minx,
        maxy,
    )

    col_b, row_b = inverse * (
        maxx,
        miny,
    )

    col_start = max(
        0,
        int(
            math.floor(
                min(
                    col_a,
                    col_b,
                )
            )
        ),
    )

    col_end = min(
        array.shape[1],
        int(
            math.ceil(
                max(
                    col_a,
                    col_b,
                )
            )
        )
        +
        1,
    )

    row_start = max(
        0,
        int(
            math.floor(
                min(
                    row_a,
                    row_b,
                )
            )
        ),
    )

    row_end = min(
        array.shape[0],
        int(
            math.ceil(
                max(
                    row_a,
                    row_b,
                )
            )
        )
        +
        1,
    )

    if (
        col_end <= col_start
        or
        row_end <= row_start
    ):
        return np.array(
            [],
            dtype="float64",
        )

    window = Window(
        col_off=col_start,
        row_off=row_start,
        width=col_end - col_start,
        height=row_end - row_start,
    )

    subset = array[
        row_start:row_end,
        col_start:col_end,
    ]

    subset_transform = window_transform(
        window,
        transform,
    )

    inside_mask = geometry_mask(
        [
            mapping(
                geometry
            )
        ],
        out_shape=subset.shape,
        transform=subset_transform,
        invert=True,
        all_touched=False,
    )

    values = subset[
        inside_mask
        &
        np.isfinite(
            subset
        )
    ]

    # -------------------------------------------------------------------------
    # Boundary sliver fallback
    # -------------------------------------------------------------------------

    if values.size == 0:

        inside_mask = geometry_mask(
            [
                mapping(
                    geometry
                )
            ],
            out_shape=subset.shape,
            transform=subset_transform,
            invert=True,
            all_touched=True,
        )

        values = subset[
            inside_mask
            &
            np.isfinite(
                subset
            )
        ]

    return values.astype(
        "float64"
    )


def summarize_array_by_geometry(
    array,
    geometry,
    transform,
):
    """
    Return zonal statistics for one geometry.
    """

    values = get_array_values_in_geometry(
        array=array,
        geometry=geometry,
        transform=transform,
    )

    if values.size == 0:

        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p90": np.nan,
        }

    return {
        "count": int(
            values.size
        ),
        "mean": float(
            np.nanmean(
                values
            )
        ),
        "median": float(
            np.nanmedian(
                values
            )
        ),
        "min": float(
            np.nanmin(
                values
            )
        ),
        "max": float(
            np.nanmax(
                values
            )
        ),
        "p90": float(
            np.nanpercentile(
                values,
                90,
            )
        ),
    }


# =============================================================================
# 6. START
# =============================================================================

start_time = time.time()

print(
    "=" * 110
)

print(
    "STEP 16B - RAW 100 m DEVELOPMENT SUITABILITY CRITERIA GENERATION"
)

print(
    "=" * 110
)


# =============================================================================
# 7. VERIFY INPUT FILES
# =============================================================================

required_files = [
    GRID_FILE,
    KMC_FILE,
    ROADS_FILE,
    BUILDINGS_FILE,
    GEDTM_FILE,
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input not found:\n{path}"
        )


# =============================================================================
# 8. LOAD CORE VECTOR DATA
# =============================================================================

print(
    "\nLoading vector datasets..."
)

grid = gpd.read_file(
    GRID_FILE
)

kmc = gpd.read_file(
    KMC_FILE
)

roads = gpd.read_file(
    ROADS_FILE
)

buildings = gpd.read_file(
    BUILDINGS_FILE
)


# =============================================================================
# 9. CRS VALIDATION
# =============================================================================

datasets_for_crs = {
    "grid": grid,
    "kmc": kmc,
    "roads": roads,
    "buildings": buildings,
}

for name, gdf in datasets_for_crs.items():

    if gdf.crs is None:

        raise RuntimeError(
            f"{name} has no CRS."
        )

    epsg = gdf.crs.to_epsg()

    if epsg != EXPECTED_EPSG:

        raise RuntimeError(
            f"{name} expected EPSG:{EXPECTED_EPSG}, "
            f"found {gdf.crs}"
        )


print(
    f"All core vectors CRS      : EPSG:{EXPECTED_EPSG}"
)


# =============================================================================
# 10. KMC UNION
# =============================================================================

kmc_union = geometry_union(
    kmc.geometry
)

if kmc_union.is_empty:

    raise RuntimeError(
        "KMC boundary union is empty."
    )

kmc_area_m2 = float(
    kmc_union.area
)

print(
    f"KMC area                  : "
    f"{kmc_area_m2 / 1_000_000:.6f} km²"
)


# =============================================================================
# 11. CLIP 100 m GRID TO TRUE KMC GEOMETRY
# =============================================================================

print(
    "\nCreating exact Grid ∩ KMC geometries..."
)

grid_clip = grid.copy()

grid_clip["geometry"] = grid_clip.geometry.intersection(
    kmc_union
)

grid_clip = grid_clip[
    grid_clip.geometry.notna()
    &
    ~grid_clip.geometry.is_empty
].copy()

grid_clip.reset_index(
    drop=True,
    inplace=True,
)

grid_clip[
    "analysis_area_m2"
] = grid_clip.geometry.area

grid_clip[
    "analysis_area_km2"
] = (
    grid_clip[
        "analysis_area_m2"
    ]
    /
    1_000_000
)

grid_clip[
    "stored_vs_recomputed_area_diff_m2"
] = (
    grid_clip[
        "analysis_area_m2"
    ]
    -
    safe_numeric(
        grid_clip[
            "kmc_overlap_area_m2"
        ]
    )
).abs()


print(
    f"Grid cells after clipping : "
    f"{len(grid_clip):,}"
)

print(
    f"Unique grid IDs           : "
    f"{grid_clip['grid_id'].nunique():,}"
)

print(
    f"Maximum area difference   : "
    f"{grid_clip['stored_vs_recomputed_area_diff_m2'].max():.6f} m²"
)


# =============================================================================
# 12. CLIP ROADS AND BUILDINGS TO KMC
# =============================================================================

print(
    "\nClipping OSM roads/buildings to KMC..."
)

roads = clipped_geometry_series(
    roads,
    kmc_union,
)

buildings = clipped_geometry_series(
    buildings,
    kmc_union,
)

roads[
    "highway_clean"
] = roads[
    "highway_clean"
].astype(
    str
).str.strip().str.lower()

buildings[
    "building_clean"
] = buildings[
    "building_clean"
].astype(
    str
).str.strip().str.lower()


roads[
    "clipped_length_m"
] = roads.geometry.length

buildings[
    "clipped_area_m2"
] = buildings.geometry.area


# =============================================================================
# 13. DEFINE C1 AND C2 ROAD NETWORKS
# =============================================================================

major_roads = roads[
    roads[
        "highway_clean"
    ].isin(
        C1_MAJOR_ROAD_CLASSES
    )
].copy()

local_drivable_roads = roads[
    roads[
        "highway_clean"
    ].isin(
        C2_LOCAL_DRIVABLE_CLASSES
    )
].copy()


if major_roads.empty:

    raise RuntimeError(
        "C1 major-road network is empty."
    )

if local_drivable_roads.empty:

    raise RuntimeError(
        "C2 local-drivable road network is empty."
    )


major_road_union = geometry_union(
    major_roads.geometry
)

local_road_sindex = (
    local_drivable_roads.sindex
)

building_sindex = (
    buildings.sindex
)


print(
    "\nROAD NETWORK DEFINITIONS"
)

print(
    f"C1 major-road features   : "
    f"{len(major_roads):,}"
)

print(
    f"C1 major-road length     : "
    f"{major_roads['clipped_length_m'].sum() / 1000:.3f} km"
)

print(
    f"C2 local-road features   : "
    f"{len(local_drivable_roads):,}"
)

print(
    f"C2 local-road length     : "
    f"{local_drivable_roads['clipped_length_m'].sum() / 1000:.3f} km"
)


# =============================================================================
# 14. C1 - HIGHER-ORDER ROAD PROXIMITY
# =============================================================================

print(
    "\nCalculating C1 - higher-order road proximity..."
)

grid_clip[
    "c1_major_road_distance_m"
] = grid_clip.geometry.distance(
    major_road_union
)


# =============================================================================
# 15. REPRESENTATIVE POINTS FOR NEIGHBOURHOOD ANALYSIS
# =============================================================================

representative_points = grid_clip.geometry.representative_point()


# =============================================================================
# 16. C2 - LOCAL ROAD-NETWORK CONTEXT
# =============================================================================

print(
    "\nCalculating C2 - local road-network density..."
)

for radius_m in ROAD_DENSITY_RADII_M:

    print(
        f"  Radius {radius_m} m"
    )

    density_values = []
    length_values = []
    zone_area_values = []

    for point in representative_points:

        zone = point.buffer(
            radius_m
        ).intersection(
            kmc_union
        )

        zone_area_m2 = float(
            zone.area
        )

        road_length_m = zone_line_length_m(
            zone_geometry=zone,
            line_gdf=local_drivable_roads,
            spatial_index=local_road_sindex,
        )

        if zone_area_m2 > 0:

            # km road / km²
            density = (
                road_length_m
                /
                1000
            ) / (
                zone_area_m2
                /
                1_000_000
            )

        else:

            density = np.nan

        length_values.append(
            road_length_m
        )

        zone_area_values.append(
            zone_area_m2
        )

        density_values.append(
            density
        )

    grid_clip[
        f"c2_local_road_length_{radius_m}m_m"
    ] = length_values

    grid_clip[
        f"c2_context_area_{radius_m}m_m2"
    ] = zone_area_values

    grid_clip[
        f"c2_road_density_{radius_m}m_km_per_km2"
    ] = density_values


# =============================================================================
# 17. C3A - BUILDING FOOTPRINT INSIDE EACH KMC-CLIPPED CELL
# =============================================================================

print(
    "\nCalculating C3 - existing building context..."
)

building_area_cell = []
building_count_cell = []

for geom in grid_clip.geometry:

    area_m2 = zone_building_area_m2(
        zone_geometry=geom,
        building_gdf=buildings,
        spatial_index=building_sindex,
    )

    count = zone_building_count(
        zone_geometry=geom,
        building_gdf=buildings,
        spatial_index=building_sindex,
    )

    building_area_cell.append(
        area_m2
    )

    building_count_cell.append(
        count
    )


grid_clip[
    "c3_building_area_cell_m2"
] = building_area_cell

grid_clip[
    "c3_building_count_cell"
] = building_count_cell

grid_clip[
    "c3_building_footprint_fraction_cell"
] = (
    grid_clip[
        "c3_building_area_cell_m2"
    ]
    /
    grid_clip[
        "analysis_area_m2"
    ]
)

grid_clip[
    "c3_building_footprint_pct_cell"
] = (
    grid_clip[
        "c3_building_footprint_fraction_cell"
    ]
    *
    100
)


# =============================================================================
# 18. C3B - SURROUNDING BUILT-CONTEXT ALTERNATIVES
# =============================================================================

for radius_m in BUILDING_CONTEXT_RADII_M:

    print(
        f"  Building context radius {radius_m} m"
    )

    area_values = []
    fraction_values = []
    count_values = []
    zone_area_values = []

    for point in representative_points:

        zone = point.buffer(
            radius_m
        ).intersection(
            kmc_union
        )

        zone_area_m2 = float(
            zone.area
        )

        building_area_m2 = zone_building_area_m2(
            zone_geometry=zone,
            building_gdf=buildings,
            spatial_index=building_sindex,
        )

        building_count = zone_building_count(
            zone_geometry=zone,
            building_gdf=buildings,
            spatial_index=building_sindex,
        )

        if zone_area_m2 > 0:

            fraction = (
                building_area_m2
                /
                zone_area_m2
            )

        else:

            fraction = np.nan

        area_values.append(
            building_area_m2
        )

        fraction_values.append(
            fraction
        )

        count_values.append(
            building_count
        )

        zone_area_values.append(
            zone_area_m2
        )

    grid_clip[
        f"c3_context_building_area_{radius_m}m_m2"
    ] = area_values

    grid_clip[
        f"c3_context_area_{radius_m}m_m2"
    ] = zone_area_values

    grid_clip[
        f"c3_context_building_fraction_{radius_m}m"
    ] = fraction_values

    grid_clip[
        f"c3_context_building_pct_{radius_m}m"
    ] = (
        grid_clip[
            f"c3_context_building_fraction_{radius_m}m"
        ]
        *
        100
    )

    grid_clip[
        f"c3_context_building_count_{radius_m}m"
    ] = count_values


# =============================================================================
# 19. LOAD GEDTM30 AND DERIVE SLOPE
# =============================================================================

print(
    "\nLoading GEDTM30 and deriving slope..."
)

dem = load_dem_and_slope(
    GEDTM_FILE
)

dem_epsg = (
    dem[
        "crs"
    ].to_epsg()
    if dem[
        "crs"
    ] is not None
    else None
)

if dem_epsg != EXPECTED_EPSG:

    raise RuntimeError(
        f"GEDTM expected EPSG:{EXPECTED_EPSG}, "
        f"found {dem['crs']}"
    )


# =============================================================================
# 20. C4 AND C5 - TERRAIN ZONAL STATISTICS
# =============================================================================

print(
    "\nCalculating C4 slope and C5 elevation..."
)

terrain_records = []

total_cells = len(
    grid_clip
)

for i, geom in enumerate(
    grid_clip.geometry,
    start=1,
):

    elevation_stats = summarize_array_by_geometry(
        array=dem[
            "elevation"
        ],
        geometry=geom,
        transform=dem[
            "transform"
        ],
    )

    slope_stats = summarize_array_by_geometry(
        array=dem[
            "slope_deg"
        ],
        geometry=geom,
        transform=dem[
            "transform"
        ],
    )

    terrain_records.append(
        {
            "c4_slope_pixel_count": slope_stats[
                "count"
            ],
            "c4_slope_mean_deg": slope_stats[
                "mean"
            ],
            "c4_slope_median_deg": slope_stats[
                "median"
            ],
            "c4_slope_p90_deg": slope_stats[
                "p90"
            ],
            "c4_slope_max_deg": slope_stats[
                "max"
            ],
            "c5_elevation_pixel_count": elevation_stats[
                "count"
            ],
            "c5_elevation_mean_m": elevation_stats[
                "mean"
            ],
            "c5_elevation_median_m": elevation_stats[
                "median"
            ],
            "c5_elevation_min_m": elevation_stats[
                "min"
            ],
            "c5_elevation_max_m": elevation_stats[
                "max"
            ],
        }
    )

    if (
        i % 1000 == 0
        or
        i == total_cells
    ):

        print(
            f"  Terrain cells processed: "
            f"{i:,}/{total_cells:,}"
        )


terrain_df = pd.DataFrame(
    terrain_records
)

for column in terrain_df.columns:

    grid_clip[
        column
    ] = terrain_df[
        column
    ].values


grid_clip[
    "c5_elevation_range_m"
] = (
    grid_clip[
        "c5_elevation_max_m"
    ]
    -
    grid_clip[
        "c5_elevation_min_m"
    ]
)


# =============================================================================
# 21. ROAD CLASS REPORT
# =============================================================================

all_road_classes = sorted(
    roads[
        "highway_clean"
    ]
    .dropna()
    .unique()
    .tolist()
)

road_class_records = []

for road_class in all_road_classes:

    class_rows = roads[
        roads[
            "highway_clean"
        ]
        ==
        road_class
    ]

    road_class_records.append(
        {
            "highway_clean": road_class,
            "feature_count": int(
                len(
                    class_rows
                )
            ),
            "total_length_km": float(
                class_rows[
                    "clipped_length_m"
                ].sum()
                /
                1000
            ),
            "included_in_c1_major_access": (
                road_class
                in
                C1_MAJOR_ROAD_CLASSES
            ),
            "included_in_c2_local_density": (
                road_class
                in
                C2_LOCAL_DRIVABLE_CLASSES
            ),
        }
    )


road_class_df = pd.DataFrame(
    road_class_records
).sort_values(
    "total_length_km",
    ascending=False,
)


# =============================================================================
# 22. RAW CRITERIA DEFINITION REPORT
# =============================================================================

criteria_definition_df = pd.DataFrame(
    [
        {
            "criterion_id": "C1",
            "criterion_name": "Higher-order road proximity",
            "raw_primary_measure": "c1_major_road_distance_m",
            "source": "OSM roads 2026",
            "raw_direction_note": (
                "Smaller distance generally represents stronger road accessibility; "
                "formal standardization occurs later."
            ),
            "status": "RAW_GENERATED",
            "weight_assigned": False,
        },
        {
            "criterion_id": "C2",
            "criterion_name": "Local drivable road-network context",
            "raw_primary_measure": (
                "Candidate radii: 250 m, 500 m, 1000 m road density"
            ),
            "source": "OSM roads 2026",
            "raw_direction_note": (
                "Higher local road density may represent stronger accessibility. "
                "Only one representative radius will be selected later."
            ),
            "status": "RAW_ALTERNATIVES_GENERATED",
            "weight_assigned": False,
        },
        {
            "criterion_id": "C3",
            "criterion_name": "Existing development context",
            "raw_primary_measure": (
                "Cell building footprint plus candidate 250 m and 500 m "
                "neighbourhood building intensity"
            ),
            "source": "OSM buildings 2026",
            "raw_direction_note": (
                "Neighbourhood development context and current cell occupation "
                "must be interpreted separately before standardization."
            ),
            "status": "RAW_ALTERNATIVES_GENERATED",
            "weight_assigned": False,
        },
        {
            "criterion_id": "C4",
            "criterion_name": "Slope",
            "raw_primary_measure": "c4_slope_mean_deg",
            "source": "GEDTM30",
            "raw_direction_note": (
                "Slope is retained in raw degrees. Suitability standardization "
                "thresholds will be justified later."
            ),
            "status": "RAW_GENERATED",
            "weight_assigned": False,
        },
        {
            "criterion_id": "C5",
            "criterion_name": "Elevation",
            "raw_primary_measure": "c5_elevation_mean_m",
            "source": "GEDTM30",
            "raw_direction_note": (
                "Elevation direction is NOT assumed here. Its inclusion and "
                "standardization require explicit justification."
            ),
            "status": "RAW_GENERATED_PENDING_JUSTIFICATION",
            "weight_assigned": False,
        },
        {
            "criterion_id": "C6",
            "criterion_name": "Land-cover reference/screening",
            "raw_primary_measure": "Not generated in Step 16B",
            "source": "ESA WorldCover 2021",
            "raw_direction_note": (
                "Retained as reference/screening context because it is a 2021 "
                "reference product rather than 2026 ground truth."
            ),
            "status": "REFERENCE_ONLY_AT_THIS_STAGE",
            "weight_assigned": False,
        },
    ]
)


# =============================================================================
# 23. RAW CRITERION COLUMNS
# =============================================================================

raw_criterion_columns = [
    "c1_major_road_distance_m",

    "c2_road_density_250m_km_per_km2",
    "c2_road_density_500m_km_per_km2",
    "c2_road_density_1000m_km_per_km2",

    "c3_building_footprint_fraction_cell",
    "c3_building_footprint_pct_cell",
    "c3_context_building_fraction_250m",
    "c3_context_building_fraction_500m",
    "c3_context_building_pct_250m",
    "c3_context_building_pct_500m",

    "c4_slope_mean_deg",
    "c4_slope_median_deg",
    "c4_slope_p90_deg",
    "c4_slope_max_deg",

    "c5_elevation_mean_m",
    "c5_elevation_median_m",
    "c5_elevation_min_m",
    "c5_elevation_max_m",
    "c5_elevation_range_m",
]


# =============================================================================
# 24. DESCRIPTIVE STATISTICS
# =============================================================================

stats_df = (
    grid_clip[
        raw_criterion_columns
    ]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .T
    .reset_index()
    .rename(
        columns={
            "index": "variable"
        }
    )
)


# =============================================================================
# 25. QA CHECKS
# =============================================================================

grid_count_ok = (
    len(
        grid_clip
    )
    ==
    EXPECTED_GRID_COUNT
)

grid_ids_unique = (
    grid_clip[
        "grid_id"
    ].nunique()
    ==
    len(
        grid_clip
    )
)

all_analysis_areas_positive = bool(
    (
        grid_clip[
            "analysis_area_m2"
        ]
        >
        0
    ).all()
)

max_area_diff = float(
    grid_clip[
        "stored_vs_recomputed_area_diff_m2"
    ].max()
)

area_reconciliation_ok = (
    max_area_diff
    <=
    1.0
)

major_road_network_nonempty = (
    len(
        major_roads
    )
    >
    0
)

local_road_network_nonempty = (
    len(
        local_drivable_roads
    )
    >
    0
)

c1_missing = int(
    grid_clip[
        "c1_major_road_distance_m"
    ].isna().sum()
)

c2_missing = int(
    grid_clip[
        [
            "c2_road_density_250m_km_per_km2",
            "c2_road_density_500m_km_per_km2",
            "c2_road_density_1000m_km_per_km2",
        ]
    ].isna().sum().sum()
)

c3_missing = int(
    grid_clip[
        [
            "c3_building_footprint_fraction_cell",
            "c3_context_building_fraction_250m",
            "c3_context_building_fraction_500m",
        ]
    ].isna().sum().sum()
)

c4_missing = int(
    grid_clip[
        "c4_slope_mean_deg"
    ].isna().sum()
)

c5_missing = int(
    grid_clip[
        "c5_elevation_mean_m"
    ].isna().sum()
)

building_fraction_min = float(
    grid_clip[
        "c3_building_footprint_fraction_cell"
    ].min()
)

building_fraction_max = float(
    grid_clip[
        "c3_building_footprint_fraction_cell"
    ].max()
)

building_fraction_plausible = (
    building_fraction_min
    >=
    -0.001
    and
    building_fraction_max
    <=
    1.01
)

road_density_nonnegative = bool(
    (
        grid_clip[
            [
                "c2_road_density_250m_km_per_km2",
                "c2_road_density_500m_km_per_km2",
                "c2_road_density_1000m_km_per_km2",
            ]
        ]
        >=
        0
    ).all().all()
)

slope_nonnegative = bool(
    (
        grid_clip[
            "c4_slope_mean_deg"
        ]
        >=
        0
    ).all()
)

# Explicit leakage/separation protection.
forbidden_column_tokens = [
    "pressure_score",
    "pressure_category",
    "shap",
    "conservation_priority",
    "ahp_weight",
    "suitability_score",
]

output_column_names_lower = [
    str(
        column
    ).lower()
    for column in grid_clip.columns
]

forbidden_columns_found = []

for token in forbidden_column_tokens:

    matches = [
        column
        for column in output_column_names_lower
        if token in column
    ]

    forbidden_columns_found.extend(
        matches
    )


no_forbidden_outputs = (
    len(
        forbidden_columns_found
    )
    ==
    0
)


overall_pass = all(
    [
        grid_count_ok,
        grid_ids_unique,
        all_analysis_areas_positive,
        area_reconciliation_ok,
        major_road_network_nonempty,
        local_road_network_nonempty,
        c1_missing == 0,
        c2_missing == 0,
        c3_missing == 0,
        c4_missing == 0,
        c5_missing == 0,
        building_fraction_plausible,
        road_density_nonnegative,
        slope_nonnegative,
        no_forbidden_outputs,
    ]
)

overall_status = (
    "PASS"
    if overall_pass
    else
    "REVIEW_REQUIRED"
)


qa_df = pd.DataFrame(
    [
        {
            "grid_cells": len(
                grid_clip
            ),
            "expected_grid_cells": EXPECTED_GRID_COUNT,
            "grid_count_ok": grid_count_ok,
            "unique_grid_ids": grid_ids_unique,
            "positive_analysis_area_all_cells": all_analysis_areas_positive,
            "max_grid_area_reconciliation_error_m2": max_area_diff,
            "grid_area_reconciliation_ok": area_reconciliation_ok,
            "c1_major_road_features": len(
                major_roads
            ),
            "c1_major_road_length_km": float(
                major_roads[
                    "clipped_length_m"
                ].sum()
                /
                1000
            ),
            "c2_local_road_features": len(
                local_drivable_roads
            ),
            "c2_local_road_length_km": float(
                local_drivable_roads[
                    "clipped_length_m"
                ].sum()
                /
                1000
            ),
            "c1_missing_values": c1_missing,
            "c2_missing_values": c2_missing,
            "c3_missing_values": c3_missing,
            "c4_missing_values": c4_missing,
            "c5_missing_values": c5_missing,
            "building_fraction_min": building_fraction_min,
            "building_fraction_max": building_fraction_max,
            "building_fraction_plausible": building_fraction_plausible,
            "road_density_nonnegative": road_density_nonnegative,
            "slope_nonnegative": slope_nonnegative,
            "pressure_shap_conservation_ahp_outputs_absent": no_forbidden_outputs,
            "forbidden_columns_found": ";".join(
                forbidden_columns_found
            ),
            "ahp_weights_assigned": False,
            "suitability_score_calculated": False,
            "standardization_performed": False,
            "overall_step16b_status": overall_status,
        }
    ]
)


# =============================================================================
# 26. SAVE OUTPUTS
# =============================================================================

print(
    "\nSaving Step 16B outputs..."
)

# CSV without geometry.
csv_output = pd.DataFrame(
    grid_clip.drop(
        columns="geometry"
    )
)

csv_output.to_csv(
    OUTPUT_CSV,
    index=False,
)

if OUTPUT_GPKG.exists():
    OUTPUT_GPKG.unlink()

grid_clip.to_file(
    OUTPUT_GPKG,
    layer="raw_suitability_criteria_100m",
    driver="GPKG",
)

road_class_df.to_csv(
    ROAD_CLASS_REPORT,
    index=False,
)

criteria_definition_df.to_csv(
    CRITERIA_DEFINITION_REPORT,
    index=False,
)

stats_df.to_csv(
    RAW_STATS_REPORT,
    index=False,
)

qa_df.to_csv(
    QA_REPORT,
    index=False,
)


# =============================================================================
# 27. SUMMARY JSON
# =============================================================================

elapsed_seconds = float(
    time.time()
    -
    start_time
)

summary = {
    "step": "16B",
    "name": "Raw 100 m Development Suitability Criteria Generation",
    "analysis_grid_cells": int(
        len(
            grid_clip
        )
    ),
    "analysis_crs": f"EPSG:{EXPECTED_EPSG}",
    "kmc_area_km2": float(
        kmc_area_m2
        /
        1_000_000
    ),
    "criteria_generated": {
        "C1": "Higher-order road distance",
        "C2": (
            "Local drivable road density at 250 m, 500 m and 1000 m "
            "candidate radii"
        ),
        "C3": (
            "Cell building footprint plus neighbourhood building context "
            "at 250 m and 500 m"
        ),
        "C4": "GEDTM30 slope",
        "C5": "GEDTM30 elevation",
    },
    "c1_major_road_classes": sorted(
        C1_MAJOR_ROAD_CLASSES
    ),
    "c2_local_drivable_classes": sorted(
        C2_LOCAL_DRIVABLE_CLASSES
    ),
    "excluded_from_core_calculation": [
        "Urbanization Pressure outputs",
        "SHAP/XAI outputs",
        "Conservation Priority outputs",
        "Provisional UDA zoning",
        "OSM motorway mainline from C1/C2 primary access definitions",
        "OSM service roads from C2 primary road-density definition",
    ],
    "ahp_weights_assigned": False,
    "suitability_score_calculated": False,
    "standardization_performed": False,
    "overall_step16b_status": overall_status,
    "elapsed_seconds": elapsed_seconds,
    "outputs": {
        "raw_csv": relative_path(
            OUTPUT_CSV
        ),
        "raw_gpkg": relative_path(
            OUTPUT_GPKG
        ),
        "road_class_report": relative_path(
            ROAD_CLASS_REPORT
        ),
        "criteria_definition_report": relative_path(
            CRITERIA_DEFINITION_REPORT
        ),
        "descriptive_statistics": relative_path(
            RAW_STATS_REPORT
        ),
        "qa_report": relative_path(
            QA_REPORT
        ),
    },
}

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 28. TERMINAL REPORT
# =============================================================================

print(
    "\n"
    +
    "=" * 110
)

print(
    "RAW CRITERIA SUMMARY"
)

print(
    "=" * 110
)

summary_columns = [
    "c1_major_road_distance_m",
    "c2_road_density_250m_km_per_km2",
    "c2_road_density_500m_km_per_km2",
    "c2_road_density_1000m_km_per_km2",
    "c3_building_footprint_pct_cell",
    "c3_context_building_pct_250m",
    "c3_context_building_pct_500m",
    "c4_slope_mean_deg",
    "c5_elevation_mean_m",
]

print(
    grid_clip[
        summary_columns
    ]
    .describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
    .T
    .to_string()
)


print(
    "\n"
    +
    "=" * 110
)

print(
    "STEP 16B QA"
)

print(
    "=" * 110
)

print(
    f"Grid count = 9,240                         : "
    f"{grid_count_ok}"
)

print(
    f"Grid IDs unique                            : "
    f"{grid_ids_unique}"
)

print(
    f"All clipped-cell areas > 0                 : "
    f"{all_analysis_areas_positive}"
)

print(
    f"Grid/KMC area reconciliation <= 1 m²       : "
    f"{area_reconciliation_ok}"
)

print(
    f"Maximum area reconciliation difference     : "
    f"{max_area_diff:.6f} m²"
)

print(
    f"C1 major-road network non-empty            : "
    f"{major_road_network_nonempty}"
)

print(
    f"C2 local-road network non-empty            : "
    f"{local_road_network_nonempty}"
)

print(
    f"C1 missing values                          : "
    f"{c1_missing}"
)

print(
    f"C2 missing values                          : "
    f"{c2_missing}"
)

print(
    f"C3 missing values                          : "
    f"{c3_missing}"
)

print(
    f"C4 slope missing values                    : "
    f"{c4_missing}"
)

print(
    f"C5 elevation missing values                : "
    f"{c5_missing}"
)

print(
    f"Building fraction plausible                : "
    f"{building_fraction_plausible}"
)

print(
    f"Road density non-negative                  : "
    f"{road_density_nonnegative}"
)

print(
    f"Slope non-negative                         : "
    f"{slope_nonnegative}"
)

print(
    f"Pressure/SHAP/conservation/AHP absent      : "
    f"{no_forbidden_outputs}"
)

print(
    f"AHP weights assigned                       : False"
)

print(
    f"Standardization performed                  : False"
)

print(
    f"Suitability score calculated               : False"
)

print(
    f"Overall Step 16B status                    : "
    f"{overall_status}"
)


print(
    "\nGenerated outputs:"
)

for output_path in [
    OUTPUT_CSV,
    OUTPUT_GPKG,
    ROAD_CLASS_REPORT,
    CRITERIA_DEFINITION_REPORT,
    RAW_STATS_REPORT,
    QA_REPORT,
    SUMMARY_JSON,
]:

    print(
        f" - {relative_path(output_path)}"
    )


print(
    "\nIMPORTANT INTERPRETATION"
)

print(
    "1. C1 is raw distance to the selected higher-order road network."
)

print(
    "2. C2 250/500/1000 m road-density variables are alternatives, not three weighted criteria."
)

print(
    "3. C3 cell occupation and neighbourhood built context must not be interpreted as the same concept."
)

print(
    "4. C3 building type categories were not used because most OSM buildings are only labelled 'yes'."
)

print(
    "5. C4 slope remains in raw degrees."
)

print(
    "6. C5 elevation direction is intentionally NOT assigned yet."
)

print(
    "7. ESA WorldCover 2021 remains reference/screening context and is not a core weighted criterion yet."
)

print(
    "8. No AHP weighting, normalization or final suitability scoring was performed."
)

print(
    "9. Step 16C must first examine distributions, redundancy/correlation and criterion direction before standardization."
)

print(
    f"\nElapsed time: "
    f"{elapsed_seconds:.2f} seconds"
)

print(
    "=" * 110
)