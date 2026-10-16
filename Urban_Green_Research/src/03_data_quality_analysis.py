"""
STEP 03 - DATA QUALITY ANALYSIS

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Evaluate Landsat annual raster quality from 2010-2026.
2. Check Landsat band consistency, valid data coverage and grid consistency.
3. Evaluate GEDTM30 and NASADEM data quality and cross-dataset consistency.
4. Validate ESA WorldCover class values.
5. Profile OSM road/building attributes and geometry quality.
6. Review UDA provisional zoning QA information.
7. Produce reproducible data-quality reports.

IMPORTANT:
- This script is READ-ONLY for all source datasets.
- No raw or processed source dataset is modified.
- UDA provisional zoning is treated as a research/planning reference,
  not as authoritative legal GIS.
- NASADEM is used only as a secondary terrain consistency reference,
  not as surveyed ground truth.
"""

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import rasterio

from rasterio.features import geometry_mask
from shapely.geometry import mapping


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
QUALITY_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "data_quality"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
QUALITY_DIR.mkdir(parents=True, exist_ok=True)


KMC_BOUNDARY = (
    RAW_DIR
    / "boundaries"
    / "kmc"
    / "kmc boundry.kml"
)

LANDSAT_DIR = (
    RAW_DIR
    / "landsat"
    / "annual_composites"
)

GEDTM30_PATH = (
    RAW_DIR
    / "dem"
    / "gedtm30"
    / "GEDTM30_Kaduwela_Buffer.tif"
)

NASADEM_PATH = (
    RAW_DIR
    / "dem"
    / "nasadem"
    / "NASADEM_Kaduwela_Buffer.tif"
)

WORLD_COVER_PATH = (
    RAW_DIR
    / "reference_lulc"
    / "esa_worldcover_2021"
    / "Kaduwela_ESA_WorldCover_2021.tif"
)

OSM_PATH = (
    RAW_DIR
    / "osm"
    / "roads_buildings"
    / "Kaduwela_OSM_Roads_Buildings_2026.gpkg"
)

UDA_DIR = (
    RAW_DIR
    / "uda"
    / "zoning_provisional"
)

ANALYSIS_CRS = "EPSG:32644"


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """
    Return a path relative to the project root when possible.
    """

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def safe_percent(
    numerator: float,
    denominator: float
) -> float:
    """
    Safely calculate a percentage.
    """

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator * 100
    )


def numeric_stats(values: np.ndarray) -> dict:
    """
    Calculate basic statistics from finite numeric values.
    """

    values = np.asarray(
        values,
        dtype="float64"
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return {
            "minimum": None,
            "maximum": None,
            "mean": None,
            "std": None,
            "median": None,
        }

    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
    }


def find_column_case_insensitive(
    columns,
    target_name
):
    """
    Find a column name without depending on capitalization.
    """

    target_lower = target_name.lower()

    for column in columns:
        if str(column).lower() == target_lower:
            return column

    return None


def clean_text_series(series: pd.Series) -> pd.Series:
    """
    Convert a field to cleaned nullable text values.
    """

    cleaned = series.astype("string").str.strip()

    cleaned = cleaned.replace(
        {
            "": pd.NA,
            "None": pd.NA,
            "nan": pd.NA,
            "<NA>": pd.NA,
        }
    )

    return cleaned


# =============================================================================
# 3. LOAD KMC STUDY AREA
# =============================================================================

print("=" * 100)
print("STEP 03 - DATA QUALITY ANALYSIS")
print("=" * 100)


if not KMC_BOUNDARY.exists():
    raise FileNotFoundError(
        f"KMC boundary not found:\n{KMC_BOUNDARY}"
    )


kmc_gdf = gpd.read_file(
    KMC_BOUNDARY
)


if kmc_gdf.crs is None:
    raise ValueError(
        "KMC boundary CRS is missing."
    )


print(
    f"\nKMC study boundary loaded: "
    f"{relative_path(KMC_BOUNDARY)}"
)

print(
    f"KMC features: {len(kmc_gdf)}"
)

print(
    f"KMC CRS: {kmc_gdf.crs}"
)


# =============================================================================
# 4. RASTER STUDY-AREA MASK HELPER
# =============================================================================

def create_kmc_mask_for_raster(
    src: rasterio.io.DatasetReader
) -> np.ndarray:
    """
    Create a boolean raster mask where True represents pixels
    whose centres fall inside the KMC boundary.
    """

    if src.crs is None:
        raise ValueError(
            "Raster CRS is missing."
        )

    kmc_in_raster_crs = (
        kmc_gdf.to_crs(src.crs)
    )

    kmc_union = (
        kmc_in_raster_crs
        .geometry
        .union_all()
    )

    mask = geometry_mask(
        [mapping(kmc_union)],
        out_shape=(
            src.height,
            src.width
        ),
        transform=src.transform,
        invert=True,
        all_touched=False,
    )

    return mask


# =============================================================================
# 5. LANDSAT DATA QUALITY
# =============================================================================

print("\n" + "=" * 100)
print("LANDSAT 2010-2026 DATA QUALITY")
print("=" * 100)


landsat_files = sorted(
    LANDSAT_DIR.glob(
        "Kaduwela_Landsat_*.tif"
    )
)


expected_years = list(
    range(2010, 2027)
)

found_years = []

landsat_band_records = []
landsat_grid_records = []


if len(landsat_files) == 0:
    raise FileNotFoundError(
        f"No Landsat files found in:\n{LANDSAT_DIR}"
    )


reference_grid = None


for landsat_file in landsat_files:

    year_match = re.search(
        r"(20\d{2})",
        landsat_file.stem
    )

    if year_match:
        year = int(
            year_match.group(1)
        )
        found_years.append(year)
    else:
        year = None

    with rasterio.open(
        landsat_file
    ) as src:

        if src.crs is None:
            raise ValueError(
                f"CRS missing for {landsat_file.name}"
            )

        kmc_mask = (
            create_kmc_mask_for_raster(
                src
            )
        )

        inside_pixel_count = int(
            np.count_nonzero(
                kmc_mask
            )
        )

        current_grid = {
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "transform": src.transform,
            "resolution_x": src.res[0],
            "resolution_y": src.res[1],
        }

        if reference_grid is None:

            reference_grid = (
                current_grid.copy()
            )

            same_crs = True
            same_dimensions = True
            same_band_count = True
            same_transform = True
            same_resolution = True

        else:

            same_crs = (
                current_grid["crs"]
                == reference_grid["crs"]
            )

            same_dimensions = (
                current_grid["width"]
                == reference_grid["width"]
                and
                current_grid["height"]
                == reference_grid["height"]
            )

            same_band_count = (
                current_grid["count"]
                == reference_grid["count"]
            )

            same_transform = (
                current_grid[
                    "transform"
                ].almost_equals(
                    reference_grid[
                        "transform"
                    ]
                )
            )

            same_resolution = (
                math.isclose(
                    current_grid[
                        "resolution_x"
                    ],
                    reference_grid[
                        "resolution_x"
                    ],
                    rel_tol=0,
                    abs_tol=1e-9,
                )
                and
                math.isclose(
                    current_grid[
                        "resolution_y"
                    ],
                    reference_grid[
                        "resolution_y"
                    ],
                    rel_tol=0,
                    abs_tol=1e-9,
                )
            )

        grid_consistent = all(
            [
                same_crs,
                same_dimensions,
                same_band_count,
                same_transform,
                same_resolution,
            ]
        )

        landsat_grid_records.append(
            {
                "year": year,
                "file": (
                    relative_path(
                        landsat_file
                    )
                ),
                "crs": str(src.crs),
                "bands": src.count,
                "width": src.width,
                "height": src.height,
                "resolution_x": (
                    src.res[0]
                ),
                "resolution_y": (
                    src.res[1]
                ),
                "same_crs_as_reference": (
                    same_crs
                ),
                "same_dimensions_as_reference": (
                    same_dimensions
                ),
                "same_band_count_as_reference": (
                    same_band_count
                ),
                "same_transform_as_reference": (
                    same_transform
                ),
                "same_resolution_as_reference": (
                    same_resolution
                ),
                "grid_consistent": (
                    grid_consistent
                ),
            }
        )

        for band_number in range(
            1,
            src.count + 1
        ):

            band_array = (
                src.read(
                    band_number,
                    masked=True
                )
                .astype("float64")
            )

            band_values = (
                band_array.filled(
                    np.nan
                )
            )

            valid_inside = (
                kmc_mask
                &
                np.isfinite(
                    band_values
                )
            )

            valid_count = int(
                np.count_nonzero(
                    valid_inside
                )
            )

            invalid_count = (
                inside_pixel_count
                - valid_count
            )

            valid_percent = (
                safe_percent(
                    valid_count,
                    inside_pixel_count,
                )
            )

            valid_values = (
                band_values[
                    valid_inside
                ]
            )

            stats = numeric_stats(
                valid_values
            )

            below_zero_count = int(
                np.count_nonzero(
                    valid_values < 0
                )
            )

            above_one_count = int(
                np.count_nonzero(
                    valid_values > 1
                )
            )

            below_zero_percent = (
                safe_percent(
                    below_zero_count,
                    valid_count,
                )
            )

            above_one_percent = (
                safe_percent(
                    above_one_count,
                    valid_count,
                )
            )

            band_description = (
                src.descriptions[
                    band_number - 1
                ]
                if src.descriptions
                else None
            )

            landsat_band_records.append(
                {
                    "year": year,
                    "file": (
                        landsat_file.name
                    ),
                    "band_number": (
                        band_number
                    ),
                    "band_description": (
                        band_description
                    ),
                    "inside_kmc_pixels": (
                        inside_pixel_count
                    ),
                    "valid_inside_kmc_pixels": (
                        valid_count
                    ),
                    "invalid_inside_kmc_pixels": (
                        invalid_count
                    ),
                    "valid_inside_kmc_percent": (
                        valid_percent
                    ),
                    "minimum": (
                        stats["minimum"]
                    ),
                    "maximum": (
                        stats["maximum"]
                    ),
                    "mean": (
                        stats["mean"]
                    ),
                    "std": (
                        stats["std"]
                    ),
                    "median": (
                        stats["median"]
                    ),
                    "percent_below_zero": (
                        below_zero_percent
                    ),
                    "percent_above_one": (
                        above_one_percent
                    ),
                    "dtype": str(
                        src.dtypes[
                            band_number - 1
                        ]
                    ),
                    "nodata": (
                        src.nodata
                    ),
                }
            )


landsat_grid_df = pd.DataFrame(
    landsat_grid_records
)

landsat_band_df = pd.DataFrame(
    landsat_band_records
)


missing_years = sorted(
    set(expected_years)
    -
    set(found_years)
)

unexpected_years = sorted(
    set(found_years)
    -
    set(expected_years)
)


all_grids_consistent = bool(
    landsat_grid_df[
        "grid_consistent"
    ].all()
)


minimum_landsat_valid_percent = float(
    landsat_band_df[
        "valid_inside_kmc_percent"
    ].min()
)


print(
    f"Expected years             : "
    f"2010-2026"
)

print(
    f"Files found                : "
    f"{len(landsat_files)} / 17"
)

print(
    f"Missing years              : "
    f"{missing_years if missing_years else 'None'}"
)

print(
    f"Unexpected years           : "
    f"{unexpected_years if unexpected_years else 'None'}"
)

print(
    f"All annual grids identical : "
    f"{all_grids_consistent}"
)

print(
    f"Minimum band valid coverage: "
    f"{minimum_landsat_valid_percent:.4f} %"
)


# =============================================================================
# 6. OPTIONAL EXISTING GEE LANDSAT QA REPORT
# =============================================================================

print("\n" + "-" * 100)
print("EXISTING LANDSAT QA REPORT CHECK")
print("-" * 100)


landsat_qa_candidates = list(
    (
        RAW_DIR
        / "landsat"
    ).rglob(
        "*Quality*Check*.csv"
    )
)


landsat_existing_qa_status = (
    "NOT_PRESENT"
)


if landsat_qa_candidates:

    landsat_qa_file = (
        landsat_qa_candidates[0]
    )

    try:

        existing_qa_df = pd.read_csv(
            landsat_qa_file
        )

        landsat_existing_qa_status = (
            "AVAILABLE"
        )

        print(
            f"Existing GEE QA report found: "
            f"{relative_path(landsat_qa_file)}"
        )

        print(
            f"Rows: {len(existing_qa_df)}"
        )

        print(
            f"Columns: "
            f"{list(existing_qa_df.columns)}"
        )

    except Exception as error:

        landsat_existing_qa_status = (
            f"ERROR: {error}"
        )

        print(
            f"Existing QA report could "
            f"not be read: {error}"
        )

else:

    print(
        "No existing Landsat GEE QA CSV "
        "was found under data/raw/landsat/."
    )

    print(
        "This does not stop Step 03. "
        "The current script performs its "
        "own raster quality checks."
    )


# =============================================================================
# 7. DEM QUALITY ANALYSIS
# =============================================================================

print("\n" + "=" * 100)
print("DEM DATA QUALITY")
print("=" * 100)


def inspect_dem(
    path: Path,
    dataset_name: str
) -> tuple[dict, np.ndarray]:

    """
    Inspect one DEM and calculate statistics.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} not found:\n{path}"
        )

    with rasterio.open(
        path
    ) as src:

        band = (
            src.read(
                1,
                masked=True
            )
            .astype("float64")
        )

        values = (
            band.filled(
                np.nan
            )
        )

        valid_mask = (
            np.isfinite(values)
        )

        valid_values = (
            values[
                valid_mask
            ]
        )

        stats = numeric_stats(
            valid_values
        )

        total_pixels = int(
            values.size
        )

        valid_pixels = int(
            valid_values.size
        )

        valid_percent = (
            safe_percent(
                valid_pixels,
                total_pixels,
            )
        )

        record = {
            "dataset_name": (
                dataset_name
            ),
            "file": (
                relative_path(path)
            ),
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "resolution_x": (
                src.res[0]
            ),
            "resolution_y": (
                src.res[1]
            ),
            "total_pixels": (
                total_pixels
            ),
            "valid_pixels": (
                valid_pixels
            ),
            "valid_percent": (
                valid_percent
            ),
            "minimum": (
                stats["minimum"]
            ),
            "maximum": (
                stats["maximum"]
            ),
            "mean": (
                stats["mean"]
            ),
            "std": (
                stats["std"]
            ),
            "median": (
                stats["median"]
            ),
            "nodata": (
                src.nodata
            ),
            "dtype": str(
                src.dtypes[0]
            ),
            "transform": str(
                src.transform
            ),
        }

        return (
            record,
            values
        )


gedtm_record, gedtm_values = (
    inspect_dem(
        GEDTM30_PATH,
        "GEDTM30"
    )
)

nasadem_record, nasadem_values = (
    inspect_dem(
        NASADEM_PATH,
        "NASADEM"
    )
)


dem_df = pd.DataFrame(
    [
        gedtm_record,
        nasadem_record,
    ]
)


print(
    dem_df[
        [
            "dataset_name",
            "crs",
            "width",
            "height",
            "resolution_x",
            "resolution_y",
            "valid_percent",
            "minimum",
            "maximum",
            "mean",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# 8. GEDTM30 vs NASADEM CONSISTENCY
# =============================================================================

print("\n" + "-" * 100)
print("GEDTM30 vs NASADEM CONSISTENCY")
print("-" * 100)


with rasterio.open(
    GEDTM30_PATH
) as gedtm_src, rasterio.open(
    NASADEM_PATH
) as nasa_src:

    same_crs = (
        gedtm_src.crs
        == nasa_src.crs
    )

    same_dimensions = (
        gedtm_src.width
        == nasa_src.width
        and
        gedtm_src.height
        == nasa_src.height
    )

    same_transform = (
        gedtm_src.transform
        .almost_equals(
            nasa_src.transform
        )
    )

    same_grid = all(
        [
            same_crs,
            same_dimensions,
            same_transform,
        ]
    )


dem_comparison_record = {
    "same_crs": (
        same_crs
    ),
    "same_dimensions": (
        same_dimensions
    ),
    "same_transform": (
        same_transform
    ),
    "same_grid": (
        same_grid
    ),
    "paired_valid_pixels": None,
    "mean_difference_gedtm_minus_nasa_m": None,
    "mae_m": None,
    "rmse_m": None,
    "correlation": None,
}


if same_grid:

    paired_valid = (
        np.isfinite(
            gedtm_values
        )
        &
        np.isfinite(
            nasadem_values
        )
    )

    gedtm_paired = (
        gedtm_values[
            paired_valid
        ]
    )

    nasa_paired = (
        nasadem_values[
            paired_valid
        ]
    )

    differences = (
        gedtm_paired
        -
        nasa_paired
    )

    paired_count = int(
        differences.size
    )

    mean_difference = float(
        np.mean(
            differences
        )
    )

    mae = float(
        np.mean(
            np.abs(
                differences
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                differences ** 2
            )
        )
    )

    if (
        paired_count > 1
        and
        np.std(
            gedtm_paired
        ) > 0
        and
        np.std(
            nasa_paired
        ) > 0
    ):

        correlation = float(
            np.corrcoef(
                gedtm_paired,
                nasa_paired
            )[0, 1]
        )

    else:
        correlation = None

    dem_comparison_record.update(
        {
            "paired_valid_pixels": (
                paired_count
            ),
            "mean_difference_gedtm_minus_nasa_m": (
                mean_difference
            ),
            "mae_m": (
                mae
            ),
            "rmse_m": (
                rmse
            ),
            "correlation": (
                correlation
            ),
        }
    )

    print(
        f"Same raster grid       : "
        f"{same_grid}"
    )

    print(
        f"Paired valid pixels    : "
        f"{paired_count}"
    )

    print(
        f"Mean difference        : "
        f"{mean_difference:.4f} m"
    )

    print(
        f"MAE                    : "
        f"{mae:.4f} m"
    )

    print(
        f"RMSE                   : "
        f"{rmse:.4f} m"
    )

    print(
        f"Correlation            : "
        f"{correlation:.6f}"
        if correlation is not None
        else
        "Correlation            : N/A"
    )

else:

    print(
        "The two DEM rasters are not on "
        "the same grid."
    )

    print(
        "Direct pixel comparison was skipped."
    )


dem_comparison_df = pd.DataFrame(
    [dem_comparison_record]
)


# =============================================================================
# 9. ESA WORLDCOVER QUALITY
# =============================================================================

print("\n" + "=" * 100)
print("ESA WORLDCOVER 2021 QUALITY")
print("=" * 100)


# Official WorldCover land-cover class codes.
# Value 0 is handled separately as exported background/no-data.
WORLD_COVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


worldcover_records = []


with rasterio.open(
    WORLD_COVER_PATH
) as src:

    worldcover_kmc_mask = (
        create_kmc_mask_for_raster(
            src
        )
    )

    worldcover_band = (
        src.read(
            1,
            masked=True
        )
    )

    worldcover_values = (
        worldcover_band
        .astype("float64")
        .filled(
            np.nan
        )
    )

    valid_inside = (
        worldcover_kmc_mask
        &
        np.isfinite(
            worldcover_values
        )
    )

    all_inside_values = (
        worldcover_values[
            valid_inside
        ].astype(int)
    )

    total_inside_pixels = int(
        all_inside_values.size
    )

    # Value 0 is not treated as a land-cover class.
    # It represents background/no-data in this exported raster.
    worldcover_zero_count = int(
        np.count_nonzero(
            all_inside_values == 0
        )
    )

    worldcover_zero_percent = (
        safe_percent(
            worldcover_zero_count,
            total_inside_pixels,
        )
    )

    # Only actual non-zero class values are assessed
    # against the official WorldCover class codes.
    landcover_values = (
        all_inside_values[
            all_inside_values != 0
        ]
    )

    unique_values, counts = (
        np.unique(
            landcover_values,
            return_counts=True,
        )
    )

    total_valid_worldcover = int(
        counts.sum()
    )

    for class_value, count in zip(
        unique_values,
        counts
    ):

        class_value = int(
            class_value
        )

        worldcover_records.append(
            {
                "class_value": (
                    class_value
                ),
                "class_name": (
                    WORLD_COVER_CLASSES.get(
                        class_value,
                        "UNEXPECTED_CLASS",
                    )
                ),
                "pixel_count": int(
                    count
                ),
                "percent_of_valid_kmc": (
                    safe_percent(
                        int(count),
                        total_valid_worldcover,
                    )
                ),
                "expected_class": (
                    class_value
                    in
                    WORLD_COVER_CLASSES
                ),
            }
        )


worldcover_df = pd.DataFrame(
    worldcover_records
)


unexpected_worldcover_classes = (
    worldcover_df[
        worldcover_df[
            "expected_class"
        ] == False
    ]
)


print(
    worldcover_df.to_string(
        index=False
    )
)


print(
    f"\nBackground / no-data value 0 pixels : "
    f"{worldcover_zero_count}"
)

print(
    f"Background / no-data percentage      : "
    f"{worldcover_zero_percent:.6f} %"
)

print(
    f"Unexpected land-cover class codes     : "
    f"{len(unexpected_worldcover_classes)}"
)


# =============================================================================
# 10. OSM ROADS / BUILDINGS QUALITY
# =============================================================================

print("\n" + "=" * 100)
print("OSM ROADS AND BUILDINGS DATA QUALITY")
print("=" * 100)


if not OSM_PATH.exists():
    raise FileNotFoundError(
        f"OSM GeoPackage not found:\n{OSM_PATH}"
    )


osm_layers = (
    pyogrio.list_layers(
        OSM_PATH
    )
)


print("\nOSM layers:")

for layer_info in osm_layers:
    print(
        f" - {layer_info[0]}"
    )


osm_layer_name = str(
    osm_layers[0][0]
)


osm_gdf = gpd.read_file(
    OSM_PATH,
    layer=osm_layer_name,
)


osm_geometry_types = (
    osm_gdf
    .geometry
    .geom_type
    .value_counts(
        dropna=False
    )
)


osm_null_geometries = int(
    osm_gdf
    .geometry
    .isna()
    .sum()
)

osm_empty_geometries = int(
    osm_gdf
    .geometry
    .is_empty
    .sum()
)

osm_invalid_geometries = int(
    (
        ~osm_gdf
        .geometry
        .is_valid
    )
    .sum()
)

osm_duplicate_geometries = int(
    osm_gdf
    .geometry
    .duplicated()
    .sum()
)


highway_column = (
    find_column_case_insensitive(
        osm_gdf.columns,
        "highway"
    )
)

building_column = (
    find_column_case_insensitive(
        osm_gdf.columns,
        "building"
    )
)


osm_highway_records = []


if highway_column is not None:

    highway_values = clean_text_series(
        osm_gdf[
            highway_column
        ]
    )

    highway_non_null = int(
        highway_values
        .notna()
        .sum()
    )

    highway_counts = (
        highway_values
        .value_counts(
            dropna=True
        )
    )

    for value, count in (
        highway_counts.items()
    ):

        osm_highway_records.append(
            {
                "highway_value": (
                    str(value)
                ),
                "feature_count": int(
                    count
                ),
            }
        )

else:

    highway_non_null = 0


if building_column is not None:

    building_values = clean_text_series(
        osm_gdf[
            building_column
        ]
    )

    building_non_null = int(
        building_values
        .notna()
        .sum()
    )

    building_candidate_count = int(
        (
            building_values
            .notna()
            &
            (
                building_values
                .str.lower()
                != "no"
            )
        )
        .sum()
    )

else:

    building_non_null = 0
    building_candidate_count = 0


osm_quality_record = {
    "layer_name": (
        osm_layer_name
    ),
    "feature_count": (
        len(osm_gdf)
    ),
    "crs": str(
        osm_gdf.crs
    ),
    "field_count": (
        len(osm_gdf.columns) - 1
    ),
    "null_geometries": (
        osm_null_geometries
    ),
    "empty_geometries": (
        osm_empty_geometries
    ),
    "invalid_geometries": (
        osm_invalid_geometries
    ),
    "duplicate_geometries": (
        osm_duplicate_geometries
    ),
    "geometry_types": (
        str(
            osm_geometry_types
            .to_dict()
        )
    ),
    "highway_field_found": (
        highway_column
        is not None
    ),
    "highway_non_null_features": (
        highway_non_null
    ),
    "building_field_found": (
        building_column
        is not None
    ),
    "building_non_null_features": (
        building_non_null
    ),
    "building_candidate_features": (
        building_candidate_count
    ),
}


osm_quality_df = pd.DataFrame(
    [osm_quality_record]
)

osm_highway_df = pd.DataFrame(
    osm_highway_records
)


print(
    f"Feature count          : "
    f"{len(osm_gdf)}"
)

print(
    f"CRS                    : "
    f"{osm_gdf.crs}"
)

print(
    f"Geometry types         : "
    f"{osm_geometry_types.to_dict()}"
)

print(
    f"Null geometries        : "
    f"{osm_null_geometries}"
)

print(
    f"Empty geometries       : "
    f"{osm_empty_geometries}"
)

print(
    f"Invalid geometries     : "
    f"{osm_invalid_geometries}"
)

print(
    f"Duplicate geometries   : "
    f"{osm_duplicate_geometries}"
)

print(
    f"Highway field found    : "
    f"{highway_column is not None}"
)

print(
    f"Highway records        : "
    f"{highway_non_null}"
)

print(
    f"Building field found   : "
    f"{building_column is not None}"
)

print(
    f"Building candidates    : "
    f"{building_candidate_count}"
)


if not osm_highway_df.empty:

    print(
        "\nTop highway values:"
    )

    print(
        osm_highway_df
        .head(25)
        .to_string(
            index=False
        )
    )


# =============================================================================
# 11. UDA PROVISIONAL ZONING QUALITY
# =============================================================================

print("\n" + "=" * 100)
print("UDA PROVISIONAL ZONING QUALITY")
print("=" * 100)


uda_main_files = list(
    UDA_DIR.rglob(
        "Kaduwela_UDA_2024_AUDITED_PROVISIONAL.gpkg"
    )
)


if len(uda_main_files) == 0:
    raise FileNotFoundError(
        "Main UDA provisional GeoPackage "
        "was not found."
    )


uda_main_gpkg = (
    uda_main_files[0]
)


uda_layers = (
    pyogrio.list_layers(
        uda_main_gpkg
    )
)


uda_layer_records = []


for layer_info in uda_layers:

    layer_name = str(
        layer_info[0]
    )

    info = pyogrio.read_info(
        uda_main_gpkg,
        layer=layer_name,
    )

    uda_layer_records.append(
        {
            "layer_name": (
                layer_name
            ),
            "feature_count": (
                info.get(
                    "features"
                )
            ),
            "geometry_type": (
                info.get(
                    "geometry_type"
                )
            ),
            "field_count": len(
                info.get(
                    "fields",
                    []
                )
            ),
            "crs": str(
                info.get(
                    "crs"
                )
            ),
        }
    )


uda_layer_df = pd.DataFrame(
    uda_layer_records
)


print("\nUDA layers:")

print(
    uda_layer_df.to_string(
        index=False
    )
)


UDA_QA_LAYER = (
    "QA_508_Screening_With_Flags"
)


uda_qa_gdf = gpd.read_file(
    uda_main_gpkg,
    layer=UDA_QA_LAYER,
)


uda_invalid_geometries = int(
    (
        ~uda_qa_gdf
        .geometry
        .is_valid
    )
    .sum()
)

uda_null_geometries = int(
    uda_qa_gdf
    .geometry
    .isna()
    .sum()
)

uda_duplicate_geometries = int(
    uda_qa_gdf
    .geometry
    .duplicated()
    .sum()
)


uda_quality_summary_record = {
    "source_file": (
        relative_path(
            uda_main_gpkg
        )
    ),
    "qa_layer": (
        UDA_QA_LAYER
    ),
    "feature_count": (
        len(uda_qa_gdf)
    ),
    "field_count": (
        len(
            uda_qa_gdf.columns
        ) - 1
    ),
    "crs": str(
        uda_qa_gdf.crs
    ),
    "invalid_geometries": (
        uda_invalid_geometries
    ),
    "null_geometries": (
        uda_null_geometries
    ),
    "duplicate_geometries": (
        uda_duplicate_geometries
    ),
}


# =============================================================================
# 12. UDA QA FIELD PROFILING
# =============================================================================

uda_field_profile_records = []


for column in uda_qa_gdf.columns:

    if column == (
        uda_qa_gdf.geometry.name
    ):
        continue

    series = uda_qa_gdf[
        column
    ]

    non_null_count = int(
        series.notna().sum()
    )

    missing_count = int(
        series.isna().sum()
    )

    unique_count = int(
        series.nunique(
            dropna=True
        )
    )

    uda_field_profile_records.append(
        {
            "field": (
                column
            ),
            "dtype": str(
                series.dtype
            ),
            "non_null_count": (
                non_null_count
            ),
            "missing_count": (
                missing_count
            ),
            "unique_count": (
                unique_count
            ),
            "missing_percent": (
                safe_percent(
                    missing_count,
                    len(series),
                )
            ),
        }
    )


uda_field_profile_df = pd.DataFrame(
    uda_field_profile_records
)


# =============================================================================
# 13. UDA LOW-CARDINALITY QA VALUE COUNTS
# =============================================================================

uda_value_count_records = []


for column in uda_qa_gdf.columns:

    if column == (
        uda_qa_gdf.geometry.name
    ):
        continue

    series = uda_qa_gdf[
        column
    ]

    unique_count = int(
        series.nunique(
            dropna=True
        )
    )

    if (
        unique_count > 0
        and
        unique_count <= 25
    ):

        cleaned = clean_text_series(
            series
        )

        counts = (
            cleaned
            .value_counts(
                dropna=False
            )
        )

        for value, count in (
            counts.items()
        ):

            value_text = (
                "<MISSING>"
                if pd.isna(value)
                else str(value)
            )

            uda_value_count_records.append(
                {
                    "field": (
                        column
                    ),
                    "value": (
                        value_text
                    ),
                    "count": int(
                        count
                    ),
                }
            )


uda_value_counts_df = pd.DataFrame(
    uda_value_count_records
)


print(
    f"\nQA layer features       : "
    f"{len(uda_qa_gdf)}"
)

print(
    f"QA layer fields         : "
    f"{len(uda_qa_gdf.columns) - 1}"
)

print(
    f"Invalid geometries      : "
    f"{uda_invalid_geometries}"
)

print(
    f"Duplicate geometries    : "
    f"{uda_duplicate_geometries}"
)

print(
    "\nQA fields:"
)

print(
    uda_field_profile_df
    .to_string(
        index=False
    )
)


# =============================================================================
# 14. OPTIONAL UDA QA_METRICS.JSON
# =============================================================================

print("\n" + "-" * 100)
print("UDA QA_METRICS.JSON CHECK")
print("-" * 100)


uda_json_candidates = list(
    UDA_DIR.rglob(
        "QA_METRICS.json"
    )
)


uda_json_status = (
    "NOT_PRESENT"
)

uda_json_records = []


def flatten_json(
    obj,
    parent_key=""
):
    """
    Flatten nested JSON content into key/value records.
    """

    records = []

    if isinstance(
        obj,
        dict
    ):

        for key, value in obj.items():

            new_key = (
                f"{parent_key}.{key}"
                if parent_key
                else str(key)
            )

            records.extend(
                flatten_json(
                    value,
                    new_key
                )
            )

    elif isinstance(
        obj,
        list
    ):

        for index, value in enumerate(
            obj
        ):

            new_key = (
                f"{parent_key}[{index}]"
            )

            records.extend(
                flatten_json(
                    value,
                    new_key
                )
            )

    else:

        records.append(
            {
                "key": (
                    parent_key
                ),
                "value": (
                    str(obj)
                ),
            }
        )

    return records


if uda_json_candidates:

    uda_json_file = (
        uda_json_candidates[0]
    )

    try:

        with open(
            uda_json_file,
            "r",
            encoding="utf-8"
        ) as file:

            uda_json_data = (
                json.load(
                    file
                )
            )

        uda_json_records = (
            flatten_json(
                uda_json_data
            )
        )

        uda_json_status = (
            "AVAILABLE"
        )

        print(
            f"QA metrics file found: "
            f"{relative_path(uda_json_file)}"
        )

        print(
            f"Flattened QA values: "
            f"{len(uda_json_records)}"
        )

    except Exception as error:

        uda_json_status = (
            f"ERROR: {error}"
        )

        print(
            f"QA_METRICS.json error: "
            f"{error}"
        )

else:

    print(
        "QA_METRICS.json was not found."
    )


uda_json_df = pd.DataFrame(
    uda_json_records
)


# =============================================================================
# 15. SAVE DATA QUALITY REPORTS
# =============================================================================

landsat_grid_report = (
    QUALITY_DIR
    / "03_landsat_grid_consistency.csv"
)

landsat_band_report = (
    QUALITY_DIR
    / "03_landsat_band_quality.csv"
)

dem_quality_report = (
    QUALITY_DIR
    / "03_dem_quality.csv"
)

dem_comparison_report = (
    QUALITY_DIR
    / "03_dem_cross_dataset_comparison.csv"
)

worldcover_report = (
    QUALITY_DIR
    / "03_worldcover_class_quality.csv"
)

osm_quality_report = (
    QUALITY_DIR
    / "03_osm_quality_summary.csv"
)

osm_highway_report = (
    QUALITY_DIR
    / "03_osm_highway_values.csv"
)

uda_layer_report = (
    QUALITY_DIR
    / "03_uda_layer_inventory.csv"
)

uda_summary_report = (
    QUALITY_DIR
    / "03_uda_quality_summary.csv"
)

uda_field_report = (
    QUALITY_DIR
    / "03_uda_field_quality.csv"
)

uda_value_report = (
    QUALITY_DIR
    / "03_uda_low_cardinality_values.csv"
)

uda_json_report = (
    QUALITY_DIR
    / "03_uda_qa_metrics_json.csv"
)


landsat_grid_df.to_csv(
    landsat_grid_report,
    index=False,
)

landsat_band_df.to_csv(
    landsat_band_report,
    index=False,
)

dem_df.to_csv(
    dem_quality_report,
    index=False,
)

dem_comparison_df.to_csv(
    dem_comparison_report,
    index=False,
)

worldcover_df.to_csv(
    worldcover_report,
    index=False,
)

osm_quality_df.to_csv(
    osm_quality_report,
    index=False,
)

osm_highway_df.to_csv(
    osm_highway_report,
    index=False,
)

uda_layer_df.to_csv(
    uda_layer_report,
    index=False,
)

pd.DataFrame(
    [uda_quality_summary_record]
).to_csv(
    uda_summary_report,
    index=False,
)

uda_field_profile_df.to_csv(
    uda_field_report,
    index=False,
)

uda_value_counts_df.to_csv(
    uda_value_report,
    index=False,
)

uda_json_df.to_csv(
    uda_json_report,
    index=False,
)


# =============================================================================
# 16. FINAL QUALITY STATUS
# =============================================================================

landsat_year_status = (
    len(landsat_files) == 17
    and
    len(missing_years) == 0
    and
    len(unexpected_years) == 0
)

landsat_band_count_status = bool(
    (
        landsat_grid_df[
            "bands"
        ] == 6
    ).all()
)

landsat_quality_status = (
    landsat_year_status
    and
    landsat_band_count_status
    and
    all_grids_consistent
)

dem_quality_status = bool(
    (
        dem_df[
            "valid_percent"
        ] > 99
    ).all()
)

worldcover_quality_status = (
    len(
        unexpected_worldcover_classes
    )
    == 0
)

osm_required_fields_status = (
    highway_column is not None
    and
    building_column is not None
)

uda_quality_status = (
    len(uda_qa_gdf) == 508
    and
    uda_invalid_geometries == 0
)


overall_checks = {
    "landsat_structure_quality": (
        landsat_quality_status
    ),
    "dem_basic_quality": (
        dem_quality_status
    ),
    "worldcover_class_quality": (
        worldcover_quality_status
    ),
    "osm_required_fields": (
        osm_required_fields_status
    ),
    "uda_qa_structure": (
        uda_quality_status
    ),
}


overall_pass = all(
    overall_checks.values()
)


summary_records = [
    {
        "check": key,
        "passed": value,
    }
    for key, value
    in overall_checks.items()
]


summary_df = pd.DataFrame(
    summary_records
)


summary_report = (
    REPORT_DIR
    / "03_data_quality_summary.csv"
)


summary_df.to_csv(
    summary_report,
    index=False,
)


# =============================================================================
# 17. FINAL TERMINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("STEP 03 SUMMARY")
print("=" * 100)

print(
    f"Landsat years complete           : "
    f"{landsat_year_status}"
)

print(
    f"Landsat six-band consistency     : "
    f"{landsat_band_count_status}"
)

print(
    f"Landsat grids consistent         : "
    f"{all_grids_consistent}"
)

print(
    f"Minimum Landsat valid coverage   : "
    f"{minimum_landsat_valid_percent:.4f} %"
)

print(
    f"Existing Landsat GEE QA report   : "
    f"{landsat_existing_qa_status}"
)

print(
    f"DEM basic quality                : "
    f"{dem_quality_status}"
)

print(
    f"DEM same-grid comparison         : "
    f"{same_grid}"
)

if (
    dem_comparison_record[
        "correlation"
    ]
    is not None
):

    print(
        f"GEDTM30/NASADEM correlation      : "
        f"{dem_comparison_record['correlation']:.6f}"
    )

    print(
        f"GEDTM30/NASADEM MAE              : "
        f"{dem_comparison_record['mae_m']:.4f} m"
    )

    print(
        f"GEDTM30/NASADEM RMSE             : "
        f"{dem_comparison_record['rmse_m']:.4f} m"
    )

print(
    f"Unexpected WorldCover classes    : "
    f"{len(unexpected_worldcover_classes)}"
)

print(
    f"OSM required fields available    : "
    f"{osm_required_fields_status}"
)

print(
    f"OSM highway records              : "
    f"{highway_non_null}"
)

print(
    f"OSM building candidates          : "
    f"{building_candidate_count}"
)

print(
    f"UDA QA features                  : "
    f"{len(uda_qa_gdf)}"
)

print(
    f"UDA invalid geometries           : "
    f"{uda_invalid_geometries}"
)

print(
    f"UDA QA_METRICS.json              : "
    f"{uda_json_status}"
)

print(
    f"\nOverall structural quality checks: "
    f"{'PASS' if overall_pass else 'REVIEW REQUIRED'}"
)


print("\nReports created under:")

print(
    f" - "
    f"{QUALITY_DIR.relative_to(PROJECT_ROOT)}"
)

print(
    f" - "
    f"{summary_report.relative_to(PROJECT_ROOT)}"
)


print("\nIMPORTANT INTERPRETATION NOTES:")

print(
    "1. Negative or >1 Landsat reflectance values are reported "
    "for diagnostic purposes and are not automatically treated as errors."
)

print(
    "2. NASADEM is only a secondary terrain consistency reference "
    "and is not treated as surveyed ground truth."
)

print(
    "3. ESA WorldCover is an independent model-generated land-cover "
    "reference and is not treated as ground truth."
)

print(
    "4. OSM records are current infrastructure/reference data; "
    "temporal leakage must be considered before using them in historical modelling."
)

print(
    "5. UDA zoning remains a provisional research/planning reference "
    "and must not be represented as authoritative legal zoning GIS."
)

print(
    "6. No source dataset was modified by this step."
)

print("=" * 100)