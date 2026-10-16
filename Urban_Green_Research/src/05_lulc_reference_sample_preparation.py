"""
STEP 05A - LULC REFERENCE SAMPLE CANDIDATE PREPARATION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Use ESA WorldCover 2021 only as an external reference source.
2. Reclassify WorldCover classes into the four research LULC classes:
      1 - Green / Vegetation
      2 - Built-up
      3 - Open / Bare
      4 - Water
3. Identify homogeneous candidate reference locations.
4. Spatially distribute candidate points across the KMC study area.
5. Create candidate reference samples for later MANUAL verification.
6. Produce reproducible sample-preparation reports.

IMPORTANT:
- ESA WorldCover is NOT treated as ground truth.
- These outputs are CANDIDATE samples only.
- Candidate points must be manually reviewed before model training.
- No Landsat classifier is trained in this step.
- No training/validation split is created in this step.
- Historical temporal stability will be checked later.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from numpy.lib.stride_tricks import sliding_window_view
from rasterio.transform import xy
from shapely.geometry import Point


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

SAMPLE_DIR = (
    PROCESSED_DIR
    / "lulc"
    / "reference_samples"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SAMPLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------

KMC_BOUNDARY = (
    PROCESSED_DIR
    / "boundaries"
    / "kmc"
    / "kaduwela_kmc_boundary_utm44n.gpkg"
)

WORLDCOVER = (
    PROCESSED_DIR
    / "reference_lulc"
    / "esa_worldcover_2021_kmc_10m.tif"
)


# -------------------------------------------------------------------------
# Outputs
# -------------------------------------------------------------------------

CANDIDATE_GPKG = (
    SAMPLE_DIR
    / "lulc_candidate_reference_points_2021.gpkg"
)

CANDIDATE_CSV = (
    SAMPLE_DIR
    / "lulc_candidate_reference_points_2021.csv"
)

CLASS_SUMMARY_REPORT = (
    REPORT_DIR
    / "05a_lulc_candidate_class_summary.csv"
)

PROCESS_SUMMARY_REPORT = (
    REPORT_DIR
    / "05a_lulc_candidate_preparation_summary.csv"
)


# =============================================================================
# 2. CONSTANTS
# =============================================================================

ANALYSIS_CRS = "EPSG:32644"

RANDOM_SEED = 42


# Desired maximum candidate points per class.
# These are NOT final training sample counts.
TARGET_CANDIDATES_PER_CLASS = 200


# Spatial block size used to reduce strong clustering.
# One candidate per class is selected from each occupied block.
SPATIAL_BLOCK_SIZE_M = 150.0


# Homogeneity window.
# 3 means a 3 x 3 WorldCover pixel neighbourhood.
# At 10 m WorldCover resolution this represents approximately 30 m x 30 m.
HOMOGENEITY_WINDOW = 3


# =============================================================================
# 3. RESEARCH LULC CLASSES
# =============================================================================

CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}


# =============================================================================
# 4. WORLDCOVER TO RESEARCH CLASS MAPPING
# =============================================================================

# WorldCover:
# 10  Tree cover
# 20  Shrubland
# 30  Grassland
# 40  Cropland
# 50  Built-up
# 60  Bare / sparse vegetation
# 70  Snow and ice
# 80  Permanent water
# 90  Herbaceous wetland
# 95  Mangroves
# 100 Moss and lichen
#
# For this research:
#
# Vegetated classes are grouped into Green / Vegetation.
# Permanent water is Water.
# Built-up remains Built-up.
# Bare/sparse vegetation becomes Open / Bare.
#
# Snow/ice is excluded because it is not relevant to Kaduwela.

WORLDCOVER_TO_RESEARCH = {
    10: 1,
    20: 1,
    30: 1,
    40: 1,
    50: 2,
    60: 3,
    80: 4,
    90: 1,
    95: 1,
    100: 1,
}


# =============================================================================
# 5. GENERAL HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """
    Return a project-relative path where possible.
    """

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def remove_existing_file(path: Path):
    """
    Remove an existing processed output before rewriting it.
    """

    if path.exists():
        path.unlink()


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
        numerator
        / denominator
        * 100
    )


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 05A - LULC REFERENCE SAMPLE CANDIDATE PREPARATION")
print("=" * 100)

print(
    f"\nWorldCover input : "
    f"{relative_path(WORLDCOVER)}"
)

print(
    f"Target candidates per class : "
    f"{TARGET_CANDIDATES_PER_CLASS}"
)

print(
    f"Spatial block size          : "
    f"{SPATIAL_BLOCK_SIZE_M:.0f} m"
)

print(
    f"Homogeneity window          : "
    f"{HOMOGENEITY_WINDOW} x {HOMOGENEITY_WINDOW}"
)


# =============================================================================
# 7. VALIDATE INPUTS
# =============================================================================

if not KMC_BOUNDARY.exists():

    raise FileNotFoundError(
        f"Processed KMC boundary not found:\n"
        f"{KMC_BOUNDARY}"
    )


if not WORLDCOVER.exists():

    raise FileNotFoundError(
        f"Processed WorldCover raster not found:\n"
        f"{WORLDCOVER}"
    )


kmc_gdf = gpd.read_file(
    KMC_BOUNDARY
)


if str(
    kmc_gdf.crs
) != ANALYSIS_CRS:

    raise ValueError(
        f"Unexpected KMC CRS: "
        f"{kmc_gdf.crs}"
    )


# =============================================================================
# 8. LOAD WORLDCOVER
# =============================================================================

with rasterio.open(
    WORLDCOVER
) as src:

    if str(
        src.crs
    ) != ANALYSIS_CRS:

        raise ValueError(
            f"Unexpected WorldCover CRS: "
            f"{src.crs}"
        )


    if not (
        np.isclose(
            src.res[0],
            10.0
        )
        and
        np.isclose(
            src.res[1],
            10.0
        )
    ):

        raise ValueError(
            f"Unexpected WorldCover resolution: "
            f"{src.res}"
        )


    worldcover_array = (
        src.read(
            1
        )
    )


    worldcover_transform = (
        src.transform
    )


    worldcover_nodata = (
        src.nodata
    )


print(
    f"\nWorldCover dimensions : "
    f"{worldcover_array.shape[1]} x "
    f"{worldcover_array.shape[0]}"
)

print(
    f"WorldCover nodata     : "
    f"{worldcover_nodata}"
)


# =============================================================================
# 9. RECLASSIFY WORLDCOVER TO FOUR RESEARCH CLASSES
# =============================================================================

research_class_array = np.zeros(
    worldcover_array.shape,
    dtype=np.uint8,
)


for worldcover_code, research_class in (
    WORLDCOVER_TO_RESEARCH.items()
):

    research_class_array[
        worldcover_array
        == worldcover_code
    ] = research_class


unique_research_classes = sorted(
    [
        int(value)
        for value
        in np.unique(
            research_class_array
        )
    ]
)


print(
    f"\nMapped research class values : "
    f"{unique_research_classes}"
)


# =============================================================================
# 10. CLASS PIXEL SUMMARY BEFORE HOMOGENEITY FILTERING
# =============================================================================

before_records = []


total_mapped_pixels = int(
    np.count_nonzero(
        research_class_array
    )
)


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    count = int(
        np.count_nonzero(
            research_class_array
            == class_id
        )
    )

    before_records.append(
        {
            "class_id": (
                class_id
            ),
            "class_name": (
                class_name
            ),
            "mapped_pixels": (
                count
            ),
            "mapped_percent": (
                safe_percent(
                    count,
                    total_mapped_pixels
                )
            ),
        }
    )


before_df = pd.DataFrame(
    before_records
)


print(
    "\nMapped WorldCover pixels by research class:"
)

print(
    before_df.to_string(
        index=False
    )
)


# =============================================================================
# 11. HOMOGENEITY FILTER
# =============================================================================

if (
    HOMOGENEITY_WINDOW
    % 2
    == 0
):

    raise ValueError(
        "HOMOGENEITY_WINDOW must be an odd number."
    )


window_radius = (
    HOMOGENEITY_WINDOW
    // 2
)


candidate_mask = np.zeros(
    research_class_array.shape,
    dtype=bool,
)


# Create sliding neighbourhood windows.
windows = sliding_window_view(
    research_class_array,
    (
        HOMOGENEITY_WINDOW,
        HOMOGENEITY_WINDOW,
    )
)


for class_id in (
    CLASS_DEFINITIONS.keys()
):

    centre_values = (
        research_class_array[
            window_radius:
            research_class_array.shape[0]
            - window_radius,
            window_radius:
            research_class_array.shape[1]
            - window_radius,
        ]
    )


    homogeneous = (
        (
            windows
            == class_id
        )
        .all(
            axis=(-2, -1)
        )
        &
        (
            centre_values
            == class_id
        )
    )


    candidate_mask[
        window_radius:
        research_class_array.shape[0]
        - window_radius,
        window_radius:
        research_class_array.shape[1]
        - window_radius,
    ] |= homogeneous


# =============================================================================
# 12. BUILD ALL HOMOGENEOUS CANDIDATE PIXELS
# =============================================================================

candidate_rows = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    rows, cols = np.where(
        candidate_mask
        &
        (
            research_class_array
            == class_id
        )
    )


    print(
        f"\n{class_name}: "
        f"{len(rows)} homogeneous candidate pixels"
    )


    for row, col in zip(
        rows,
        cols
    ):

        x_coord, y_coord = xy(
            worldcover_transform,
            row,
            col,
            offset="center",
        )


        worldcover_code = int(
            worldcover_array[
                row,
                col
            ]
        )


        candidate_rows.append(
            {
                "class_id": (
                    class_id
                ),
                "class_name": (
                    class_name
                ),
                "wc_code": (
                    worldcover_code
                ),
                "row": (
                    int(row)
                ),
                "col": (
                    int(col)
                ),
                "x": (
                    float(x_coord)
                ),
                "y": (
                    float(y_coord)
                ),
            }
        )


all_candidates_df = pd.DataFrame(
    candidate_rows
)


if all_candidates_df.empty:

    raise ValueError(
        "No homogeneous candidate pixels "
        "were generated."
    )


# =============================================================================
# 13. ADD SPATIAL BLOCK IDENTIFIERS
# =============================================================================

kmc_minx, kmc_miny, _, _ = (
    kmc_gdf.total_bounds
)


all_candidates_df[
    "block_x"
] = np.floor(
    (
        all_candidates_df[
            "x"
        ]
        - kmc_minx
    )
    / SPATIAL_BLOCK_SIZE_M
).astype(int)


all_candidates_df[
    "block_y"
] = np.floor(
    (
        all_candidates_df[
            "y"
        ]
        - kmc_miny
    )
    / SPATIAL_BLOCK_SIZE_M
).astype(int)


all_candidates_df[
    "spatial_block"
] = (
    all_candidates_df[
        "block_x"
    ].astype(str)
    +
    "_"
    +
    all_candidates_df[
        "block_y"
    ].astype(str)
)


# =============================================================================
# 14. SPATIALLY DISTRIBUTED SAMPLING
# =============================================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


selected_class_dfs = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    class_df = (
        all_candidates_df[
            all_candidates_df[
                "class_id"
            ]
            == class_id
        ]
        .copy()
    )


    if class_df.empty:

        print(
            f"\nWARNING: "
            f"No candidates available for "
            f"{class_name}."
        )

        continue


    # Randomize rows reproducibly.
    random_order = rng.permutation(
        len(class_df)
    )

    class_df = (
        class_df
        .iloc[
            random_order
        ]
        .copy()
    )


    # One candidate per spatial block.
    block_sample = (
        class_df
        .drop_duplicates(
            subset=[
                "spatial_block"
            ],
            keep="first",
        )
        .copy()
    )


    # Randomize the spatially distributed candidates again.
    if len(
        block_sample
    ) > 0:

        second_order = (
            rng.permutation(
                len(
                    block_sample
                )
            )
        )

        block_sample = (
            block_sample
            .iloc[
                second_order
            ]
            .copy()
        )


    selected_count = min(
        TARGET_CANDIDATES_PER_CLASS,
        len(
            block_sample
        )
    )


    selected = (
        block_sample
        .head(
            selected_count
        )
        .copy()
    )


    selected_class_dfs.append(
        selected
    )


    print(
        f"{class_name}: "
        f"{len(class_df)} homogeneous pixels, "
        f"{len(block_sample)} occupied spatial blocks, "
        f"{selected_count} selected candidates"
    )


if len(
    selected_class_dfs
) == 0:

    raise ValueError(
        "No class candidates were selected."
    )


selected_df = pd.concat(
    selected_class_dfs,
    ignore_index=True,
)


# =============================================================================
# 15. CREATE SAMPLE IDS AND REVIEW FIELDS
# =============================================================================

selected_df = (
    selected_df
    .sort_values(
        [
            "class_id",
            "spatial_block",
        ]
    )
    .reset_index(
        drop=True
    )
)


sample_ids = []


class_counters = {
    class_id: 0
    for class_id
    in CLASS_DEFINITIONS.keys()
}


for class_id in (
    selected_df[
        "class_id"
    ]
):

    class_counters[
        class_id
    ] += 1


    sample_ids.append(
        f"C{class_id}_"
        f"{class_counters[class_id]:03d}"
    )


selected_df[
    "sample_id"
] = sample_ids


selected_df[
    "reference_year"
] = 2021


selected_df[
    "reference_source"
] = (
    "ESA WorldCover 2021 candidate"
)


selected_df[
    "homogeneity_rule"
] = (
    f"{HOMOGENEITY_WINDOW}x"
    f"{HOMOGENEITY_WINDOW}"
    f"_mapped_class"
)


# -------------------------------------------------------------------------
# These fields are intentionally left for manual review.
# -------------------------------------------------------------------------

selected_df[
    "review_status"
] = "PENDING"


selected_df[
    "manual_class_id"
] = pd.NA


selected_df[
    "manual_class_name"
] = pd.NA


selected_df[
    "temporal_stability"
] = "NOT_CHECKED"


selected_df[
    "training_eligible"
] = "NOT_DECIDED"


selected_df[
    "validation_eligible"
] = "NOT_DECIDED"


selected_df[
    "review_notes"
] = ""


# =============================================================================
# 16. CREATE GEODATAFRAME
# =============================================================================

geometry = [
    Point(
        x_value,
        y_value
    )
    for x_value, y_value
    in zip(
        selected_df[
            "x"
        ],
        selected_df[
            "y"
        ],
    )
]


candidate_gdf = (
    gpd.GeoDataFrame(
        selected_df,
        geometry=geometry,
        crs=ANALYSIS_CRS,
    )
)


# =============================================================================
# 17. VERIFY POINTS FALL INSIDE KMC
# =============================================================================

kmc_union = (
    kmc_gdf
    .geometry
    .union_all()
)


inside_kmc = (
    candidate_gdf
    .geometry
    .apply(
        kmc_union.covers
    )
)


outside_count = int(
    (
        ~inside_kmc
    ).sum()
)


if outside_count > 0:

    raise ValueError(
        f"{outside_count} candidate samples "
        f"fall outside the KMC boundary."
    )


# =============================================================================
# 18. FINAL CLASS SUMMARY
# =============================================================================

class_summary_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    mapped_pixels = int(
        np.count_nonzero(
            research_class_array
            == class_id
        )
    )


    homogeneous_pixels = int(
        np.count_nonzero(
            candidate_mask
            &
            (
                research_class_array
                == class_id
            )
        )
    )


    selected_count = int(
        (
            candidate_gdf[
                "class_id"
            ]
            == class_id
        )
        .sum()
    )


    class_summary_records.append(
        {
            "class_id": (
                class_id
            ),
            "class_name": (
                class_name
            ),
            "mapped_worldcover_pixels": (
                mapped_pixels
            ),
            "homogeneous_candidate_pixels": (
                homogeneous_pixels
            ),
            "selected_candidate_points": (
                selected_count
            ),
            "target_candidate_points": (
                TARGET_CANDIDATES_PER_CLASS
            ),
            "target_reached": (
                selected_count
                >=
                TARGET_CANDIDATES_PER_CLASS
            ),
        }
    )


class_summary_df = pd.DataFrame(
    class_summary_records
)


print("\n" + "=" * 100)
print("FINAL CANDIDATE SAMPLE COUNTS")
print("=" * 100)

print(
    class_summary_df[
        [
            "class_id",
            "class_name",
            "homogeneous_candidate_pixels",
            "selected_candidate_points",
            "target_reached",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# 19. SAVE CANDIDATE SAMPLE OUTPUTS
# =============================================================================

remove_existing_file(
    CANDIDATE_GPKG
)


candidate_gdf.to_file(
    CANDIDATE_GPKG,
    layer="candidate_reference_points",
    driver="GPKG",
)


csv_columns = [
    "sample_id",
    "class_id",
    "class_name",
    "wc_code",
    "reference_year",
    "reference_source",
    "homogeneity_rule",
    "spatial_block",
    "x",
    "y",
    "review_status",
    "manual_class_id",
    "manual_class_name",
    "temporal_stability",
    "training_eligible",
    "validation_eligible",
    "review_notes",
]


candidate_gdf[
    csv_columns
].to_csv(
    CANDIDATE_CSV,
    index=False,
)


class_summary_df.to_csv(
    CLASS_SUMMARY_REPORT,
    index=False,
)


# =============================================================================
# 20. SAVE PROCESS SUMMARY
# =============================================================================

process_summary_df = pd.DataFrame(
    [
        {
            "source_dataset": (
                relative_path(
                    WORLDCOVER
                )
            ),
            "analysis_crs": (
                ANALYSIS_CRS
            ),
            "worldcover_resolution_m": (
                10
            ),
            "homogeneity_window": (
                HOMOGENEITY_WINDOW
            ),
            "homogeneity_ground_size_m": (
                HOMOGENEITY_WINDOW
                * 10
            ),
            "spatial_block_size_m": (
                SPATIAL_BLOCK_SIZE_M
            ),
            "random_seed": (
                RANDOM_SEED
            ),
            "target_candidates_per_class": (
                TARGET_CANDIDATES_PER_CLASS
            ),
            "total_candidate_points": (
                len(
                    candidate_gdf
                )
            ),
            "outside_kmc_points": (
                outside_count
            ),
            "candidate_gpkg": (
                relative_path(
                    CANDIDATE_GPKG
                )
            ),
            "candidate_csv": (
                relative_path(
                    CANDIDATE_CSV
                )
            ),
            "status": (
                "CANDIDATES_ONLY_MANUAL_REVIEW_REQUIRED"
            ),
        }
    ]
)


process_summary_df.to_csv(
    PROCESS_SUMMARY_REPORT,
    index=False,
)


# =============================================================================
# 21. FINAL SUMMARY
# =============================================================================

all_classes_present = (
    set(
        candidate_gdf[
            "class_id"
        ].unique()
    )
    ==
    set(
        CLASS_DEFINITIONS.keys()
    )
)


minimum_class_candidates = int(
    class_summary_df[
        "selected_candidate_points"
    ].min()
)


print("\n" + "=" * 100)
print("STEP 05A SUMMARY")
print("=" * 100)

print(
    f"Total candidate points       : "
    f"{len(candidate_gdf)}"
)

print(
    f"All four classes represented : "
    f"{all_classes_present}"
)

print(
    f"Minimum candidates in a class: "
    f"{minimum_class_candidates}"
)

print(
    f"Points outside KMC           : "
    f"{outside_count}"
)


print("\nCandidate outputs:")

print(
    f" - "
    f"{relative_path(CANDIDATE_GPKG)}"
)

print(
    f" - "
    f"{relative_path(CANDIDATE_CSV)}"
)


print("\nReports:")

print(
    f" - "
    f"{relative_path(CLASS_SUMMARY_REPORT)}"
)

print(
    f" - "
    f"{relative_path(PROCESS_SUMMARY_REPORT)}"
)


print("\nIMPORTANT:")

print(
    "1. These are candidate reference points only."
)

print(
    "2. ESA WorldCover was used only to identify likely class locations."
)

print(
    "3. Candidate samples must be manually reviewed before training."
)

print(
    "4. Training and validation samples must remain independent."
)

print(
    "5. Temporal stability must be checked before using samples "
    "for historical 2010-2026 classification."
)

print(
    "6. No Landsat model was trained in this step."
)

print("=" * 100)