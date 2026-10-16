"""
STEP 11A - CREATE PERMANENT 100 M URBANIZATION-PRESSURE MODELLING GRID

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Create a reproducible 100 m x 100 m UTM modelling lattice over Kaduwela.
2. Align grid edges to exact 100 m EPSG:32644 coordinates.
3. Retain every full grid cell that intersects the official KMC boundary.
4. Record how much of each full square lies inside KMC.
5. Create stable grid IDs for all later pressure/suitability/conservation analysis.

IMPORTANT:
- Grid geometry remains a full 100 m x 100 m square.
- Boundary cells may contain only part of KMC.
- kmc_overlap_fraction records usable municipal coverage.
- This step creates spatial units only.
- No ML target or predictor is created here.
"""

from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KMC_BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "boundaries"
    / "kmc"
    / "kaduwela_kmc_boundary_utm44n.gpkg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "grid"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. OUTPUTS
# =============================================================================

GRID_GPKG = (
    OUTPUT_DIR
    / "kaduwela_100m_modelling_grid.gpkg"
)

GRID_CSV = (
    OUTPUT_DIR
    / "kaduwela_100m_modelling_grid_attributes.csv"
)

GRID_SUMMARY_CSV = (
    REPORT_DIR
    / "11a_100m_modelling_grid_summary.csv"
)

GRID_COVERAGE_CSV = (
    REPORT_DIR
    / "11a_100m_modelling_grid_coverage_summary.csv"
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

TARGET_CRS = "EPSG:32644"

GRID_SIZE_M = 100.0

FULL_CELL_AREA_M2 = (
    GRID_SIZE_M
    *
    GRID_SIZE_M
)

FULL_CELL_AREA_KM2 = (
    FULL_CELL_AREA_M2
    /
    1_000_000
)

BOUNDARY_CELL_THRESHOLD = 0.999999


# =============================================================================
# 4. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """Return project-relative path where possible."""

    try:

        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:

        return str(path)


def find_kmc_boundary() -> Path:
    """
    Find processed KMC boundary if expected filename differs.

    The project previously established the processed official KMC boundary.
    """

    if KMC_BOUNDARY_FILE.exists():

        return KMC_BOUNDARY_FILE


    boundary_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "boundaries"
    )


    candidate_files = list(
        boundary_dir.rglob(
            "*.gpkg"
        )
    )


    kmc_candidates = [
        path
        for path in candidate_files
        if (
            "kmc"
            in path.name.lower()
            or
            "kaduwela"
            in path.name.lower()
        )
    ]


    if len(
        kmc_candidates
    ) == 1:

        return kmc_candidates[
            0
        ]


    if len(
        kmc_candidates
    ) == 0:

        raise FileNotFoundError(
            "Processed KMC boundary GeoPackage could not be found."
        )


    raise RuntimeError(
        "Multiple possible KMC boundary files found:\n"
        +
        "\n".join(
            str(
                path
            )
            for path in kmc_candidates
        )
        +
        "\nSet KMC_BOUNDARY_FILE explicitly."
    )


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 11A - CREATE PERMANENT 100 M URBANIZATION-PRESSURE MODELLING GRID")
print("=" * 100)


kmc_file = find_kmc_boundary()


print(
    f"\nKMC boundary : "
    f"{relative_path(kmc_file)}"
)


# =============================================================================
# 6. LOAD OFFICIAL KMC BOUNDARY
# =============================================================================

kmc_gdf = gpd.read_file(
    kmc_file
)


if len(
    kmc_gdf
) == 0:

    raise RuntimeError(
        "KMC boundary dataset is empty."
    )


if kmc_gdf.crs is None:

    raise RuntimeError(
        "KMC boundary has no CRS."
    )


if str(
    kmc_gdf.crs
).upper() != TARGET_CRS:

    kmc_gdf = kmc_gdf.to_crs(
        TARGET_CRS
    )


if not kmc_gdf.geometry.is_valid.all():

    raise RuntimeError(
        "Invalid geometry exists in processed KMC boundary."
    )


kmc_geometry = (
    kmc_gdf.geometry
    .union_all()
)


if kmc_geometry.is_empty:

    raise RuntimeError(
        "Unified KMC geometry is empty."
    )


kmc_area_m2 = float(
    kmc_geometry.area
)


kmc_area_km2 = (
    kmc_area_m2
    /
    1_000_000
)


print(
    f"KMC area     : "
    f"{kmc_area_km2:.6f} km²"
)


# =============================================================================
# 7. ALIGN GRID EXTENT TO EXACT 100 M UTM COORDINATES
# =============================================================================

minx, miny, maxx, maxy = (
    kmc_geometry.bounds
)


aligned_minx = (
    math.floor(
        minx
        /
        GRID_SIZE_M
    )
    *
    GRID_SIZE_M
)


aligned_miny = (
    math.floor(
        miny
        /
        GRID_SIZE_M
    )
    *
    GRID_SIZE_M
)


aligned_maxx = (
    math.ceil(
        maxx
        /
        GRID_SIZE_M
    )
    *
    GRID_SIZE_M
)


aligned_maxy = (
    math.ceil(
        maxy
        /
        GRID_SIZE_M
    )
    *
    GRID_SIZE_M
)


n_cols = int(
    round(
        (
            aligned_maxx
            -
            aligned_minx
        )
        /
        GRID_SIZE_M
    )
)


n_rows = int(
    round(
        (
            aligned_maxy
            -
            aligned_miny
        )
        /
        GRID_SIZE_M
    )
)


print("\n" + "=" * 100)
print("ALIGNED GRID EXTENT")
print("=" * 100)


print(
    f"X extent : "
    f"{aligned_minx:.0f} -> {aligned_maxx:.0f}"
)

print(
    f"Y extent : "
    f"{aligned_miny:.0f} -> {aligned_maxy:.0f}"
)

print(
    f"Columns  : "
    f"{n_cols}"
)

print(
    f"Rows     : "
    f"{n_rows}"
)


# =============================================================================
# 8. GENERATE FULL 100 M GRID CELLS
# =============================================================================

records = []


for row_index in range(
    n_rows
):

    y_min = (
        aligned_miny
        +
        row_index
        *
        GRID_SIZE_M
    )

    y_max = (
        y_min
        +
        GRID_SIZE_M
    )


    for col_index in range(
        n_cols
    ):

        x_min = (
            aligned_minx
            +
            col_index
            *
            GRID_SIZE_M
        )

        x_max = (
            x_min
            +
            GRID_SIZE_M
        )


        cell_geometry = box(
            x_min,
            y_min,
            x_max,
            y_max,
        )


        if not cell_geometry.intersects(
            kmc_geometry
        ):

            continue


        intersection = (
            cell_geometry.intersection(
                kmc_geometry
            )
        )


        overlap_area_m2 = float(
            intersection.area
        )


        if overlap_area_m2 <= 0:

            continue


        overlap_fraction = (
            overlap_area_m2
            /
            FULL_CELL_AREA_M2
        )


        centroid = (
            cell_geometry.centroid
        )


        grid_id = (
            f"G100_"
            f"R{row_index:03d}_"
            f"C{col_index:03d}"
        )


        records.append(
            {
                "grid_id": (
                    grid_id
                ),
                "grid_row": (
                    row_index
                ),
                "grid_col": (
                    col_index
                ),
                "centroid_x": (
                    float(
                        centroid.x
                    )
                ),
                "centroid_y": (
                    float(
                        centroid.y
                    )
                ),
                "full_cell_area_m2": (
                    FULL_CELL_AREA_M2
                ),
                "full_cell_area_km2": (
                    FULL_CELL_AREA_KM2
                ),
                "kmc_overlap_area_m2": (
                    overlap_area_m2
                ),
                "kmc_overlap_area_km2": (
                    overlap_area_m2
                    /
                    1_000_000
                ),
                "kmc_overlap_fraction": (
                    overlap_fraction
                ),
                "boundary_cell": (
                    overlap_fraction
                    <
                    BOUNDARY_CELL_THRESHOLD
                ),
                "geometry": (
                    cell_geometry
                ),
            }
        )


grid_gdf = gpd.GeoDataFrame(
    records,
    geometry="geometry",
    crs=TARGET_CRS,
)


if len(
    grid_gdf
) == 0:

    raise RuntimeError(
        "No 100 m cells were generated."
    )


# =============================================================================
# 9. GRID QA
# =============================================================================

duplicate_grid_ids = int(
    grid_gdf[
        "grid_id"
    ]
    .duplicated()
    .sum()
)


invalid_geometries = int(
    (
        ~grid_gdf.geometry.is_valid
    ).sum()
)


incorrect_cell_areas = int(
    (
        np.abs(
            grid_gdf.geometry.area.to_numpy()
            -
            FULL_CELL_AREA_M2
        )
        >
        1e-6
    ).sum()
)


overlap_fraction_invalid = int(
    (
        (
            grid_gdf[
                "kmc_overlap_fraction"
            ]
            <=
            0
        )
        |
        (
            grid_gdf[
                "kmc_overlap_fraction"
            ]
            >
            1.0
            +
            1e-9
        )
    ).sum()
)


full_cells = int(
    (
        ~grid_gdf[
            "boundary_cell"
        ]
    ).sum()
)


boundary_cells = int(
    grid_gdf[
        "boundary_cell"
    ].sum()
)


summed_kmc_overlap_area_m2 = float(
    grid_gdf[
        "kmc_overlap_area_m2"
    ].sum()
)


area_difference_m2 = (
    summed_kmc_overlap_area_m2
    -
    kmc_area_m2
)


absolute_area_difference_m2 = abs(
    area_difference_m2
)


area_match = (
    absolute_area_difference_m2
    <=
    max(
        1.0,
        kmc_area_m2
        *
        1e-8,
    )
)


# =============================================================================
# 10. NEIGHBOR-FRIENDLY INTEGER INDEX CHECK
# =============================================================================

unique_row_col_pairs = (
    grid_gdf[
        [
            "grid_row",
            "grid_col",
        ]
    ]
    .drop_duplicates()
)


row_col_unique = (
    len(
        unique_row_col_pairs
    )
    ==
    len(
        grid_gdf
    )
)


# =============================================================================
# 11. COVERAGE GROUPS
# =============================================================================

def coverage_group(
    fraction,
):

    if fraction >= 0.999999:

        return "A_FULL_100%"

    if fraction >= 0.75:

        return "B_75_TO_<100%"

    if fraction >= 0.50:

        return "C_50_TO_<75%"

    if fraction >= 0.25:

        return "D_25_TO_<50%"

    return "E_<25%"


grid_gdf[
    "coverage_group"
] = [
    coverage_group(
        value
    )
    for value in grid_gdf[
        "kmc_overlap_fraction"
    ]
]


coverage_df = (
    grid_gdf.groupby(
        "coverage_group",
        as_index=False,
    )
    .agg(
        grid_cells=(
            "grid_id",
            "count",
        ),
        kmc_overlap_area_km2=(
            "kmc_overlap_area_km2",
            "sum",
        ),
        mean_overlap_fraction=(
            "kmc_overlap_fraction",
            "mean",
        ),
    )
)


coverage_order = [
    "A_FULL_100%",
    "B_75_TO_<100%",
    "C_50_TO_<75%",
    "D_25_TO_<50%",
    "E_<25%",
]


coverage_df[
    "coverage_group"
] = pd.Categorical(
    coverage_df[
        "coverage_group"
    ],
    categories=coverage_order,
    ordered=True,
)


coverage_df = (
    coverage_df.sort_values(
        "coverage_group"
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 12. SUMMARY
# =============================================================================

summary_df = pd.DataFrame(
    [
        {
            "grid_crs": (
                TARGET_CRS
            ),
            "grid_size_m": (
                GRID_SIZE_M
            ),
            "full_cell_area_m2": (
                FULL_CELL_AREA_M2
            ),
            "kmc_area_km2": (
                kmc_area_km2
            ),
            "retained_grid_cells": (
                len(
                    grid_gdf
                )
            ),
            "full_cells": (
                full_cells
            ),
            "boundary_cells": (
                boundary_cells
            ),
            "summed_kmc_overlap_area_km2": (
                summed_kmc_overlap_area_m2
                /
                1_000_000
            ),
            "kmc_overlap_area_difference_m2": (
                area_difference_m2
            ),
            "duplicate_grid_ids": (
                duplicate_grid_ids
            ),
            "invalid_geometries": (
                invalid_geometries
            ),
            "incorrect_full_cell_areas": (
                incorrect_cell_areas
            ),
            "invalid_overlap_fractions": (
                overlap_fraction_invalid
            ),
            "row_col_pairs_unique": (
                row_col_unique
            ),
            "grid_overlap_reconstructs_kmc_area": (
                area_match
            ),
            "target_created": (
                False
            ),
            "predictors_created": (
                False
            ),
        }
    ]
)


# =============================================================================
# 13. FINAL STATUS
# =============================================================================

overall_status = (
    "PASS"
    if all(
        [
            duplicate_grid_ids
            ==
            0,
            invalid_geometries
            ==
            0,
            incorrect_cell_areas
            ==
            0,
            overlap_fraction_invalid
            ==
            0,
            row_col_unique,
            area_match,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


summary_df[
    "overall_step11a_status"
] = (
    overall_status
)


# =============================================================================
# 14. SAVE OUTPUTS
# =============================================================================

if GRID_GPKG.exists():

    GRID_GPKG.unlink()


grid_gdf.to_file(
    GRID_GPKG,
    layer="grid_100m",
    driver="GPKG",
)


grid_attribute_df = (
    grid_gdf.drop(
        columns="geometry"
    )
)


grid_attribute_df.to_csv(
    GRID_CSV,
    index=False,
)


summary_df.to_csv(
    GRID_SUMMARY_CSV,
    index=False,
)


coverage_df.to_csv(
    GRID_COVERAGE_CSV,
    index=False,
)


# =============================================================================
# 15. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("100 M MODELLING GRID")
print("=" * 100)


print(
    f"Retained cells                : "
    f"{len(grid_gdf)}"
)

print(
    f"Full 100 m cells              : "
    f"{full_cells}"
)

print(
    f"Boundary partial cells        : "
    f"{boundary_cells}"
)

print(
    f"KMC area                      : "
    f"{kmc_area_km2:.6f} km²"
)

print(
    f"Summed grid/KMC overlap       : "
    f"{summed_kmc_overlap_area_m2 / 1_000_000:.6f} km²"
)

print(
    f"Area difference               : "
    f"{area_difference_m2:.6f} m²"
)


print("\n" + "=" * 100)
print("BOUNDARY COVERAGE DISTRIBUTION")
print("=" * 100)


print(
    coverage_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 11A QA")
print("=" * 100)


print(
    f"Duplicate grid IDs            : "
    f"{duplicate_grid_ids}"
)

print(
    f"Invalid geometries            : "
    f"{invalid_geometries}"
)

print(
    f"Incorrect full-cell areas     : "
    f"{incorrect_cell_areas}"
)

print(
    f"Invalid overlap fractions     : "
    f"{overlap_fraction_invalid}"
)

print(
    f"Row/column pairs unique       : "
    f"{row_col_unique}"
)

print(
    f"Grid overlap reconstructs KMC : "
    f"{area_match}"
)

print(
    f"Target created                : "
    f"False"
)

print(
    f"Predictors created            : "
    f"False"
)

print(
    f"Overall Step 11A status       : "
    f"{overall_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(GRID_GPKG)}"
)

print(
    f" - {relative_path(GRID_CSV)}"
)

print(
    f" - {relative_path(GRID_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(GRID_COVERAGE_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. Every modelling unit remains a full 100 m x 100 m square."
)

print(
    "2. Boundary cells retain their KMC overlap fraction rather than being geometrically clipped."
)

print(
    "3. The grid is aligned to exact 100 m EPSG:32644 coordinates."
)

print(
    "4. This stable grid ID will be reused throughout pressure, suitability and conservation analysis."
)

print(
    "5. No urbanization-pressure target has been created yet."
)

print(
    "6. The next step will construct a leakage-safe historical conversion target using time-ordered LULC evidence."
)

print("=" * 100)