"""
STEP 05C - AUTOMATED LULC CANDIDATE PURITY SCREENING

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Screen all 645 WorldCover-derived LULC candidate points automatically.
2. Link every candidate to its actual 2021 Landsat 30 m pixel.
3. Measure WorldCover class composition inside that Landsat pixel footprint.
4. Extract Landsat spectral bands and spectral indices.
5. Assess spectral consistency with the proposed candidate class.
6. Assign HIGH / MEDIUM / LOW screening confidence.
7. Produce a cleaner candidate pool for later manual audit and training preparation.

Research LULC Classes:
1 - Green / Vegetation
2 - Built-up
3 - Open / Bare
4 - Water

IMPORTANT:
- This is an AUTOMATED SCREENING step, not ground-truth validation.
- HIGH confidence does not mean a point is guaranteed correct.
- LOW confidence does not automatically mean the point is wrong.
- ESA WorldCover remains a reference product, not ground truth.
- The manual quality audit remains independent evidence.
- No classifier is trained in this step.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, mapping


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REFERENCE_DIR = (
    PROCESSED_DIR
    / "lulc"
    / "reference_samples"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. INPUT FILES
# =============================================================================

CANDIDATE_GPKG = (
    REFERENCE_DIR
    / "lulc_candidate_reference_points_2021.gpkg"
)

LANDSAT_2021 = (
    PROCESSED_DIR
    / "landsat"
    / "annual_composites_kmc"
    / "Kaduwela_Landsat_2021.tif"
)

WORLDCOVER_2021 = (
    PROCESSED_DIR
    / "reference_lulc"
    / "esa_worldcover_2021_kmc_10m.tif"
)


# =============================================================================
# 3. OUTPUT FILES
# =============================================================================

SCREENED_GPKG = (
    REFERENCE_DIR
    / "lulc_candidate_purity_screened_2021.gpkg"
)

SCREENED_CSV = (
    REFERENCE_DIR
    / "lulc_candidate_purity_screened_2021.csv"
)

CLASS_SUMMARY_REPORT = (
    REPORT_DIR
    / "05c_candidate_purity_class_summary.csv"
)

SPECTRAL_STATS_REPORT = (
    REPORT_DIR
    / "05c_candidate_spectral_reference_stats.csv"
)

PROCESS_SUMMARY_REPORT = (
    REPORT_DIR
    / "05c_candidate_purity_screening_summary.csv"
)


# =============================================================================
# 4. CONSTANTS
# =============================================================================

ANALYSIS_CRS = "EPSG:32644"

EPSILON = 1e-10


CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}


# WorldCover -> research-class mapping
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


# -------------------------------------------------------------------------
# Screening thresholds
# -------------------------------------------------------------------------

# At least this fraction of the valid WorldCover cells whose centres
# fall inside the Landsat 30 m pixel should agree with the proposed class.
HIGH_WC_PURITY = 0.80

MEDIUM_WC_PURITY = 0.67


# Robust spectral outlier thresholds.
# These are screening thresholds, not model thresholds.
HIGH_MAX_ROBUST_Z = 4.0

MEDIUM_MAX_ROBUST_Z = 6.0


# At least this many valid WorldCover 10 m cells should fall within
# the Landsat pixel footprint for a reliable footprint calculation.
MIN_WC_CELLS = 6


# =============================================================================
# 5. GENERAL HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """
    Return a project-relative path when possible.
    """

    try:

        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:

        return str(path)


def remove_existing_file(path: Path):
    """
    Remove an existing generated output.
    """

    if path.exists():
        path.unlink()


def safe_ratio(
    numerator: float,
    denominator: float
) -> float:
    """
    Safely calculate a ratio.
    """

    if denominator == 0:
        return np.nan

    return float(
        numerator / denominator
    )


def normalized_band_name(
    description
):
    """
    Normalize a raster band description.
    """

    if description is None:
        return None

    value = (
        str(description)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    aliases = {
        "blue": "blue",
        "b": "blue",
        "green": "green",
        "g": "green",
        "red": "red",
        "r": "red",
        "nir": "nir",
        "nearinfrared": "nir",
        "swir1": "swir1",
        "swir01": "swir1",
        "shortwaveinfrared1": "swir1",
        "swir2": "swir2",
        "swir02": "swir2",
        "shortwaveinfrared2": "swir2",
    }

    return aliases.get(
        value
    )


def robust_scale_value(
    value: float,
    median: float,
    mad: float,
) -> float:
    """
    Calculate an absolute robust z-score.

    1.4826 * MAD approximates standard deviation for
    normally distributed data.
    """

    if (
        not np.isfinite(value)
        or
        not np.isfinite(median)
    ):
        return np.nan


    robust_scale = (
        1.4826
        * mad
    )


    if (
        not np.isfinite(
            robust_scale
        )
        or
        robust_scale
        < EPSILON
    ):

        return 0.0


    return float(
        abs(
            value
            -
            median
        )
        /
        robust_scale
    )


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 05C - AUTOMATED LULC CANDIDATE PURITY SCREENING")
print("=" * 100)


for required_path in [
    CANDIDATE_GPKG,
    LANDSAT_2021,
    WORLDCOVER_2021,
]:

    if not required_path.exists():

        raise FileNotFoundError(
            f"Required input not found:\n"
            f"{required_path}"
        )


print(
    f"\nCandidate pool : "
    f"{relative_path(CANDIDATE_GPKG)}"
)

print(
    f"Landsat 2021   : "
    f"{relative_path(LANDSAT_2021)}"
)

print(
    f"WorldCover     : "
    f"{relative_path(WORLDCOVER_2021)}"
)


# =============================================================================
# 7. LOAD CANDIDATE POOL
# =============================================================================

candidate_gdf = gpd.read_file(
    CANDIDATE_GPKG
)


if str(
    candidate_gdf.crs
) != ANALYSIS_CRS:

    raise ValueError(
        f"Unexpected candidate CRS: "
        f"{candidate_gdf.crs}"
    )


if len(
    candidate_gdf
) != 645:

    print(
        f"\nWARNING: Expected 645 candidates "
        f"from Step 05A, but found "
        f"{len(candidate_gdf)}."
    )


required_candidate_columns = {
    "sample_id",
    "class_id",
    "class_name",
    "wc_code",
    "x",
    "y",
}


missing_columns = (
    required_candidate_columns
    -
    set(
        candidate_gdf.columns
    )
)


if missing_columns:

    raise ValueError(
        f"Candidate dataset is missing columns: "
        f"{sorted(missing_columns)}"
    )


candidate_gdf[
    "class_id"
] = pd.to_numeric(
    candidate_gdf[
        "class_id"
    ],
    errors="raise",
).astype(int)


if candidate_gdf[
    "sample_id"
].duplicated().any():

    raise ValueError(
        "Duplicate sample IDs exist "
        "in the candidate pool."
    )


print(
    f"Candidate features : "
    f"{len(candidate_gdf)}"
)


# =============================================================================
# 8. OPEN LANDSAT AND DETERMINE BAND ORDER
# =============================================================================

with rasterio.open(
    LANDSAT_2021
) as landsat_src:

    if str(
        landsat_src.crs
    ) != ANALYSIS_CRS:

        raise ValueError(
            f"Unexpected Landsat CRS: "
            f"{landsat_src.crs}"
        )


    if landsat_src.count != 6:

        raise ValueError(
            f"Expected 6 Landsat bands, "
            f"found {landsat_src.count}."
        )


    landsat_transform = (
        landsat_src.transform
    )

    landsat_width = (
        landsat_src.width
    )

    landsat_height = (
        landsat_src.height
    )

    landsat_nodata = (
        landsat_src.nodata
    )


    descriptions = list(
        landsat_src.descriptions
    )


print("\n" + "=" * 100)
print("LANDSAT BAND INTERPRETATION")
print("=" * 100)


print(
    f"Band descriptions : "
    f"{descriptions}"
)


band_lookup = {}


for band_number, description in enumerate(
    descriptions,
    start=1,
):

    normalized = (
        normalized_band_name(
            description
        )
    )

    if normalized is not None:

        band_lookup[
            normalized
        ] = band_number


expected_band_names = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
]


if all(
    band_name
    in
    band_lookup

    for band_name
    in expected_band_names
):

    band_method = (
        "Band descriptions"
    )

else:

    # This follows the already-established Landsat export contract:
    # blue, green, red, nir, swir1, swir2.
    band_lookup = {
        "blue": 1,
        "green": 2,
        "red": 3,
        "nir": 4,
        "swir1": 5,
        "swir2": 6,
    }

    band_method = (
        "Established six-band export order fallback"
    )


print(
    f"Band interpretation method : "
    f"{band_method}"
)

print(
    f"Band mapping               : "
    f"{band_lookup}"
)


# =============================================================================
# 9. LOAD RASTERS INTO MEMORY
# =============================================================================

with rasterio.open(
    LANDSAT_2021
) as landsat_src:

    landsat_data = (
        landsat_src
        .read()
        .astype(
            "float64"
        )
    )


with rasterio.open(
    WORLDCOVER_2021
) as wc_src:

    if str(
        wc_src.crs
    ) != ANALYSIS_CRS:

        raise ValueError(
            f"Unexpected WorldCover CRS: "
            f"{wc_src.crs}"
        )


    wc_array = (
        wc_src.read(
            1
        )
    )

    wc_transform = (
        wc_src.transform
    )

    wc_width = (
        wc_src.width
    )

    wc_height = (
        wc_src.height
    )

    wc_nodata = (
        wc_src.nodata
    )


# =============================================================================
# 10. WORLDCOVER -> RESEARCH CLASS ARRAY
# =============================================================================

wc_research_array = np.zeros(
    wc_array.shape,
    dtype=np.uint8,
)


for wc_code, research_class in (
    WORLDCOVER_TO_RESEARCH.items()
):

    wc_research_array[
        wc_array
        == wc_code
    ] = research_class


# =============================================================================
# 11. EXTRACT ONE LANDSAT PIXEL AND WORLDCOVER FOOTPRINT
# =============================================================================

def extract_candidate_record(
    candidate_row
):
    """
    Extract Landsat and WorldCover purity information
    for one candidate point.
    """

    x_value = float(
        candidate_row.geometry.x
    )

    y_value = float(
        candidate_row.geometry.y
    )

    class_id = int(
        candidate_row[
            "class_id"
        ]
    )


    # -------------------------------------------------------------------------
    # Identify the 30 m Landsat pixel containing the point
    # -------------------------------------------------------------------------

    landsat_row, landsat_col = (
        rasterio.transform.rowcol(
            landsat_transform,
            x_value,
            y_value,
        )
    )


    landsat_row = int(
        landsat_row
    )

    landsat_col = int(
        landsat_col
    )


    if not (
        0
        <= landsat_row
        < landsat_height
        and
        0
        <= landsat_col
        < landsat_width
    ):

        return {
            "landsat_row": (
                landsat_row
            ),
            "landsat_col": (
                landsat_col
            ),
            "landsat_valid": (
                False
            ),
            "wc_valid_cells": (
                0
            ),
            "wc_matching_cells": (
                0
            ),
            "wc_pixel_purity": (
                np.nan
            ),
        }


    # -------------------------------------------------------------------------
    # Extract the six Landsat band values
    # -------------------------------------------------------------------------

    band_values = {}


    landsat_valid = True


    for band_name, band_number in (
        band_lookup.items()
    ):

        value = float(
            landsat_data[
                band_number - 1,
                landsat_row,
                landsat_col,
            ]
        )


        if (
            not np.isfinite(
                value
            )
            or
            (
                landsat_nodata
                is not None
                and
                np.isclose(
                    value,
                    landsat_nodata
                )
            )
        ):

            landsat_valid = False


        band_values[
            band_name
        ] = value


    # -------------------------------------------------------------------------
    # Calculate spectral indices
    # -------------------------------------------------------------------------

    blue = (
        band_values[
            "blue"
        ]
    )

    green = (
        band_values[
            "green"
        ]
    )

    red = (
        band_values[
            "red"
        ]
    )

    nir = (
        band_values[
            "nir"
        ]
    )

    swir1 = (
        band_values[
            "swir1"
        ]
    )

    swir2 = (
        band_values[
            "swir2"
        ]
    )


    def normalized_difference(
        numerator_a,
        numerator_b
    ):

        denominator = (
            numerator_a
            +
            numerator_b
        )


        if (
            not np.isfinite(
                denominator
            )
            or
            abs(
                denominator
            )
            < EPSILON
        ):

            return np.nan


        return float(
            (
                numerator_a
                -
                numerator_b
            )
            /
            denominator
        )


    ndvi = normalized_difference(
        nir,
        red
    )


    ndbi = normalized_difference(
        swir1,
        nir
    )


    mndwi = normalized_difference(
        green,
        swir1
    )


    bsi_numerator = (
        (
            swir1
            +
            red
        )
        -
        (
            nir
            +
            blue
        )
    )


    bsi_denominator = (
        (
            swir1
            +
            red
        )
        +
        (
            nir
            +
            blue
        )
    )


    if (
        not np.isfinite(
            bsi_denominator
        )
        or
        abs(
            bsi_denominator
        )
        < EPSILON
    ):

        bsi = np.nan

    else:

        bsi = float(
            bsi_numerator
            /
            bsi_denominator
        )


    # -------------------------------------------------------------------------
    # Exact Landsat pixel footprint
    # -------------------------------------------------------------------------

    landsat_window = Window(
        landsat_col,
        landsat_row,
        1,
        1,
    )


    with rasterio.open(
        LANDSAT_2021
    ) as landsat_src:

        pixel_bounds = (
            rasterio.windows.bounds(
                landsat_window,
                landsat_src.transform,
            )
        )


    pixel_polygon = box(
        *pixel_bounds
    )


    # -------------------------------------------------------------------------
    # Find WorldCover cells intersecting the Landsat pixel bounding box
    # -------------------------------------------------------------------------

    raw_wc_window = from_bounds(
        *pixel_bounds,
        transform=wc_transform,
    )


    row_start = max(
        0,
        int(
            np.floor(
                raw_wc_window.row_off
            )
        )
    )


    col_start = max(
        0,
        int(
            np.floor(
                raw_wc_window.col_off
            )
        )
    )


    row_stop = min(
        wc_height,
        int(
            np.ceil(
                raw_wc_window.row_off
                +
                raw_wc_window.height
            )
        )
    )


    col_stop = min(
        wc_width,
        int(
            np.ceil(
                raw_wc_window.col_off
                +
                raw_wc_window.width
            )
        )
    )


    if (
        row_stop
        <= row_start
        or
        col_stop
        <= col_start
    ):

        wc_valid_cells = 0
        wc_matching_cells = 0
        wc_pixel_purity = np.nan

    else:

        wc_subset = (
            wc_research_array[
                row_start:row_stop,
                col_start:col_stop,
            ]
        )


        wc_window = Window(
            col_start,
            row_start,
            col_stop
            -
            col_start,
            row_stop
            -
            row_start,
        )


        subset_transform = (
            rasterio.windows.transform(
                wc_window,
                wc_transform,
            )
        )


        centre_inside_pixel = (
            geometry_mask(
                [
                    mapping(
                        pixel_polygon
                    )
                ],
                out_shape=(
                    wc_subset.shape
                ),
                transform=subset_transform,
                invert=True,
                all_touched=False,
            )
        )


        valid_wc_mask = (
            centre_inside_pixel
            &
            (
                wc_subset
                != 0
            )
        )


        wc_valid_cells = int(
            np.count_nonzero(
                valid_wc_mask
            )
        )


        wc_matching_cells = int(
            np.count_nonzero(
                valid_wc_mask
                &
                (
                    wc_subset
                    == class_id
                )
            )
        )


        wc_pixel_purity = (
            safe_ratio(
                wc_matching_cells,
                wc_valid_cells,
            )
        )


    return {
        "landsat_row": (
            landsat_row
        ),
        "landsat_col": (
            landsat_col
        ),
        "landsat_valid": (
            landsat_valid
        ),
        "blue": (
            blue
        ),
        "green": (
            green
        ),
        "red": (
            red
        ),
        "nir": (
            nir
        ),
        "swir1": (
            swir1
        ),
        "swir2": (
            swir2
        ),
        "ndvi": (
            ndvi
        ),
        "ndbi": (
            ndbi
        ),
        "mndwi": (
            mndwi
        ),
        "bsi": (
            bsi
        ),
        "wc_valid_cells": (
            wc_valid_cells
        ),
        "wc_matching_cells": (
            wc_matching_cells
        ),
        "wc_pixel_purity": (
            wc_pixel_purity
        ),
    }


# =============================================================================
# 12. SCREEN ALL 645 CANDIDATES
# =============================================================================

print("\n" + "=" * 100)
print("EXTRACTING LANDSAT / WORLDCOVER SCREENING FEATURES")
print("=" * 100)


screening_records = []


for index, candidate in (
    candidate_gdf.iterrows()
):

    screening_record = (
        extract_candidate_record(
            candidate
        )
    )


    screening_record[
        "sample_id"
    ] = candidate[
        "sample_id"
    ]


    screening_records.append(
        screening_record
    )


    if (
        len(
            screening_records
        )
        % 100
        == 0
    ):

        print(
            f"Processed "
            f"{len(screening_records)} / "
            f"{len(candidate_gdf)}"
        )


screening_df = pd.DataFrame(
    screening_records
)


candidate_screened = (
    candidate_gdf
    .merge(
        screening_df,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
)


print(
    f"\nScreened candidates : "
    f"{len(candidate_screened)}"
)


# =============================================================================
# 13. CLASS-SPECIFIC ROBUST SPECTRAL STATISTICS
# =============================================================================

print("\n" + "=" * 100)
print("CALCULATING CLASS-SPECIFIC ROBUST SPECTRAL STATISTICS")
print("=" * 100)


SPECTRAL_FEATURES = [
    "ndvi",
    "ndbi",
    "mndwi",
    "bsi",
]


spectral_stats_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    class_df = (
        candidate_screened[
            (
                candidate_screened[
                    "class_id"
                ]
                == class_id
            )
            &
            (
                candidate_screened[
                    "landsat_valid"
                ]
                == True
            )
        ]
        .copy()
    )


    for feature in (
        SPECTRAL_FEATURES
    ):

        values = pd.to_numeric(
            class_df[
                feature
            ],
            errors="coerce",
        )


        values = values[
            np.isfinite(
                values
            )
        ]


        if len(values) == 0:

            median = np.nan
            mad = np.nan

        else:

            median = float(
                np.median(
                    values
                )
            )


            mad = float(
                np.median(
                    np.abs(
                        values
                        -
                        median
                    )
                )
            )


        spectral_stats_records.append(
            {
                "class_id": (
                    class_id
                ),
                "class_name": (
                    class_name
                ),
                "feature": (
                    feature
                ),
                "count": (
                    len(values)
                ),
                "median": (
                    median
                ),
                "mad": (
                    mad
                ),
            }
        )


spectral_stats_df = pd.DataFrame(
    spectral_stats_records
)


print(
    spectral_stats_df.to_string(
        index=False
    )
)


# =============================================================================
# 14. APPLY CLASS-SPECIFIC ROBUST Z SCORES
# =============================================================================

for feature in (
    SPECTRAL_FEATURES
):

    candidate_screened[
        f"{feature}_robust_z"
    ] = np.nan


for class_id in (
    CLASS_DEFINITIONS.keys()
):

    class_mask = (
        candidate_screened[
            "class_id"
        ]
        == class_id
    )


    for feature in (
        SPECTRAL_FEATURES
    ):

        stats_row = (
            spectral_stats_df[
                (
                    spectral_stats_df[
                        "class_id"
                    ]
                    == class_id
                )
                &
                (
                    spectral_stats_df[
                        "feature"
                    ]
                    == feature
                )
            ]
            .iloc[0]
        )


        median = float(
            stats_row[
                "median"
            ]
        )


        mad = float(
            stats_row[
                "mad"
            ]
        )


        values = pd.to_numeric(
            candidate_screened.loc[
                class_mask,
                feature,
            ],
            errors="coerce",
        )


        robust_z_values = [
            robust_scale_value(
                float(value),
                median,
                mad,
            )
            if pd.notna(value)
            else np.nan

            for value
            in values
        ]


        candidate_screened.loc[
            class_mask,
            f"{feature}_robust_z",
        ] = robust_z_values


robust_z_columns = [
    f"{feature}_robust_z"
    for feature
    in SPECTRAL_FEATURES
]


candidate_screened[
    "max_spectral_robust_z"
] = (
    candidate_screened[
        robust_z_columns
    ]
    .max(
        axis=1,
        skipna=True,
    )
)


# =============================================================================
# 15. GLOBAL SPECTRAL CLASS-CENTROID CONSISTENCY
# =============================================================================

# Use robust global scaling so NDVI/NDBI/MNDWI/BSI
# contribute on comparable scales.

global_medians = {}

global_scales = {}


for feature in (
    SPECTRAL_FEATURES
):

    values = pd.to_numeric(
        candidate_screened[
            feature
        ],
        errors="coerce",
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    median = float(
        np.median(
            values
        )
    )


    q25 = float(
        np.percentile(
            values,
            25
        )
    )


    q75 = float(
        np.percentile(
            values,
            75
        )
    )


    iqr = (
        q75
        -
        q25
    )


    scale = (
        iqr
        /
        1.349
    )


    if (
        not np.isfinite(
            scale
        )
        or
        scale
        < EPSILON
    ):

        scale = float(
            np.std(
                values
            )
        )


    if (
        not np.isfinite(
            scale
        )
        or
        scale
        < EPSILON
    ):

        scale = 1.0


    global_medians[
        feature
    ] = median


    global_scales[
        feature
    ] = scale


# -------------------------------------------------------------------------
# Standardized feature matrix
# -------------------------------------------------------------------------

standardized_features = np.full(
    (
        len(
            candidate_screened
        ),
        len(
            SPECTRAL_FEATURES
        ),
    ),
    np.nan,
    dtype="float64",
)


for feature_index, feature in enumerate(
    SPECTRAL_FEATURES
):

    values = pd.to_numeric(
        candidate_screened[
            feature
        ],
        errors="coerce",
    ).to_numpy(
        dtype="float64"
    )


    standardized_features[
        :,
        feature_index,
    ] = (
        values
        -
        global_medians[
            feature
        ]
    ) / global_scales[
        feature
    ]


# -------------------------------------------------------------------------
# Class centroids
# -------------------------------------------------------------------------

class_centroids = {}


for class_id in (
    CLASS_DEFINITIONS.keys()
):

    class_mask = (
        candidate_screened[
            "class_id"
        ].to_numpy()
        == class_id
    )


    class_matrix = (
        standardized_features[
            class_mask
        ]
    )


    class_centroids[
        class_id
    ] = np.nanmedian(
        class_matrix,
        axis=0,
    )


# -------------------------------------------------------------------------
# Candidate-to-centroid distances
# -------------------------------------------------------------------------

own_distances = []

nearest_class_ids = []

nearest_distances = []

nearest_other_distances = []

spectral_margins = []


for row_index in range(
    len(
        candidate_screened
    )
):

    vector = (
        standardized_features[
            row_index
        ]
    )


    proposed_class = int(
        candidate_screened.iloc[
            row_index
        ][
            "class_id"
        ]
    )


    if not np.all(
        np.isfinite(
            vector
        )
    ):

        own_distances.append(
            np.nan
        )

        nearest_class_ids.append(
            np.nan
        )

        nearest_distances.append(
            np.nan
        )

        nearest_other_distances.append(
            np.nan
        )

        spectral_margins.append(
            np.nan
        )

        continue


    distances = {}


    for class_id, centroid in (
        class_centroids.items()
    ):

        distances[
            class_id
        ] = float(
            np.sqrt(
                np.sum(
                    (
                        vector
                        -
                        centroid
                    )
                    ** 2
                )
            )
        )


    own_distance = (
        distances[
            proposed_class
        ]
    )


    nearest_class = min(
        distances,
        key=distances.get,
    )


    nearest_distance = (
        distances[
            nearest_class
        ]
    )


    other_distances = [
        distance

        for class_id, distance
        in distances.items()

        if class_id
        != proposed_class
    ]


    nearest_other = min(
        other_distances
    )


    spectral_margin = (
        nearest_other
        -
        own_distance
    )


    own_distances.append(
        own_distance
    )

    nearest_class_ids.append(
        nearest_class
    )

    nearest_distances.append(
        nearest_distance
    )

    nearest_other_distances.append(
        nearest_other
    )

    spectral_margins.append(
        spectral_margin
    )


candidate_screened[
    "spectral_own_distance"
] = own_distances


candidate_screened[
    "spectral_nearest_class"
] = nearest_class_ids


candidate_screened[
    "spectral_nearest_distance"
] = nearest_distances


candidate_screened[
    "spectral_nearest_other_distance"
] = nearest_other_distances


candidate_screened[
    "spectral_margin"
] = spectral_margins


candidate_screened[
    "spectral_class_agreement"
] = (
    candidate_screened[
        "spectral_nearest_class"
    ]
    ==
    candidate_screened[
        "class_id"
    ]
)


# =============================================================================
# 16. ASSIGN AUTOMATED SCREENING CONFIDENCE
# =============================================================================

confidence_values = []

confidence_reasons = []


for _, row in (
    candidate_screened.iterrows()
):

    reasons = []


    landsat_valid = bool(
        row[
            "landsat_valid"
        ]
    )


    wc_cells = int(
        row[
            "wc_valid_cells"
        ]
    )


    wc_purity = (
        row[
            "wc_pixel_purity"
        ]
    )


    max_z = (
        row[
            "max_spectral_robust_z"
        ]
    )


    spectral_agreement = bool(
        row[
            "spectral_class_agreement"
        ]
    )


    # -------------------------------------------------------------------------
    # LOW confidence conditions
    # -------------------------------------------------------------------------

    if not landsat_valid:

        confidence = "LOW"

        reasons.append(
            "invalid Landsat pixel"
        )


    elif wc_cells < MIN_WC_CELLS:

        confidence = "LOW"

        reasons.append(
            "insufficient WorldCover cells in Landsat footprint"
        )


    elif (
        not np.isfinite(
            wc_purity
        )
        or
        wc_purity
        < MEDIUM_WC_PURITY
    ):

        confidence = "LOW"

        reasons.append(
            "low 30m WorldCover class purity"
        )


    elif (
        not np.isfinite(
            max_z
        )
        or
        max_z
        >
        MEDIUM_MAX_ROBUST_Z
    ):

        confidence = "LOW"

        reasons.append(
            "strong spectral outlier"
        )


    # -------------------------------------------------------------------------
    # HIGH confidence conditions
    # -------------------------------------------------------------------------

    elif (
        wc_purity
        >=
        HIGH_WC_PURITY
        and
        max_z
        <=
        HIGH_MAX_ROBUST_Z
        and
        spectral_agreement
    ):

        confidence = "HIGH"

        reasons.append(
            "high 30m WorldCover purity"
        )

        reasons.append(
            "spectrally consistent with proposed class"
        )


    # -------------------------------------------------------------------------
    # Remaining valid candidates are MEDIUM confidence
    # -------------------------------------------------------------------------

    else:

        confidence = "MEDIUM"


        if (
            wc_purity
            <
            HIGH_WC_PURITY
        ):

            reasons.append(
                "moderate 30m WorldCover purity"
            )


        if (
            max_z
            >
            HIGH_MAX_ROBUST_Z
        ):

            reasons.append(
                "moderate spectral outlier"
            )


        if not spectral_agreement:

            reasons.append(
                "nearest spectral centroid is another class"
            )


    confidence_values.append(
        confidence
    )


    confidence_reasons.append(
        "; ".join(
            reasons
        )
        if reasons
        else
        "screening criteria satisfied"
    )


candidate_screened[
    "screen_confidence"
] = confidence_values


candidate_screened[
    "screen_reason"
] = confidence_reasons


candidate_screened[
    "high_confidence_candidate"
] = (
    candidate_screened[
        "screen_confidence"
    ]
    == "HIGH"
)


# =============================================================================
# 17. CLASS SUMMARY
# =============================================================================

summary_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    class_df = (
        candidate_screened[
            candidate_screened[
                "class_id"
            ]
            == class_id
        ]
    )


    total = len(
        class_df
    )


    high_count = int(
        (
            class_df[
                "screen_confidence"
            ]
            == "HIGH"
        ).sum()
    )


    medium_count = int(
        (
            class_df[
                "screen_confidence"
            ]
            == "MEDIUM"
        ).sum()
    )


    low_count = int(
        (
            class_df[
                "screen_confidence"
            ]
            == "LOW"
        ).sum()
    )


    mean_wc_purity = float(
        class_df[
            "wc_pixel_purity"
        ].mean()
    )


    median_wc_purity = float(
        class_df[
            "wc_pixel_purity"
        ].median()
    )


    spectral_agreement_count = int(
        class_df[
            "spectral_class_agreement"
        ].sum()
    )


    summary_records.append(
        {
            "class_id": (
                class_id
            ),
            "class_name": (
                class_name
            ),
            "total_candidates": (
                total
            ),
            "high_confidence": (
                high_count
            ),
            "medium_confidence": (
                medium_count
            ),
            "low_confidence": (
                low_count
            ),
            "high_percent": (
                safe_ratio(
                    high_count,
                    total,
                )
                * 100
            ),
            "mean_wc_pixel_purity": (
                mean_wc_purity
            ),
            "median_wc_pixel_purity": (
                median_wc_purity
            ),
            "spectral_class_agreement_count": (
                spectral_agreement_count
            ),
            "spectral_class_agreement_percent": (
                safe_ratio(
                    spectral_agreement_count,
                    total,
                )
                * 100
            ),
        }
    )


class_summary_df = pd.DataFrame(
    summary_records
)


# =============================================================================
# 18. PRINT THREE KNOWN REVIEW POINTS
# =============================================================================

print("\n" + "=" * 100)
print("KNOWN POINT SCREENING CHECK")
print("=" * 100)


known_ids = [
    "C1_001",
    "C1_003",
    "C1_008",
]


known_columns = [
    "sample_id",
    "class_id",
    "class_name",
    "wc_valid_cells",
    "wc_matching_cells",
    "wc_pixel_purity",
    "ndvi",
    "ndbi",
    "mndwi",
    "bsi",
    "max_spectral_robust_z",
    "spectral_nearest_class",
    "spectral_margin",
    "screen_confidence",
    "screen_reason",
]


known_df = (
    candidate_screened[
        candidate_screened[
            "sample_id"
        ].isin(
            known_ids
        )
    ][
        known_columns
    ]
    .sort_values(
        "sample_id"
    )
)


print(
    known_df.to_string(
        index=False
    )
)


# =============================================================================
# 19. SAVE OUTPUTS
# =============================================================================

remove_existing_file(
    SCREENED_GPKG
)


candidate_screened.to_file(
    SCREENED_GPKG,
    layer="purity_screened_candidates",
    driver="GPKG",
)


candidate_screened.drop(
    columns="geometry"
).to_csv(
    SCREENED_CSV,
    index=False,
)


class_summary_df.to_csv(
    CLASS_SUMMARY_REPORT,
    index=False,
)


spectral_stats_df.to_csv(
    SPECTRAL_STATS_REPORT,
    index=False,
)


# =============================================================================
# 20. PROCESS SUMMARY
# =============================================================================

confidence_counts = (
    candidate_screened[
        "screen_confidence"
    ]
    .value_counts()
)


high_total = int(
    confidence_counts.get(
        "HIGH",
        0
    )
)


medium_total = int(
    confidence_counts.get(
        "MEDIUM",
        0
    )
)


low_total = int(
    confidence_counts.get(
        "LOW",
        0
    )
)


invalid_landsat_total = int(
    (
        candidate_screened[
            "landsat_valid"
        ]
        == False
    ).sum()
)


process_summary_df = pd.DataFrame(
    [
        {
            "candidate_count": (
                len(
                    candidate_screened
                )
            ),
            "high_confidence": (
                high_total
            ),
            "medium_confidence": (
                medium_total
            ),
            "low_confidence": (
                low_total
            ),
            "invalid_landsat_pixels": (
                invalid_landsat_total
            ),
            "high_wc_purity_threshold": (
                HIGH_WC_PURITY
            ),
            "medium_wc_purity_threshold": (
                MEDIUM_WC_PURITY
            ),
            "high_max_robust_z_threshold": (
                HIGH_MAX_ROBUST_Z
            ),
            "medium_max_robust_z_threshold": (
                MEDIUM_MAX_ROBUST_Z
            ),
            "minimum_wc_cells": (
                MIN_WC_CELLS
            ),
            "band_interpretation_method": (
                band_method
            ),
            "status": (
                "AUTOMATED_SCREENING_COMPLETE"
            ),
        }
    ]
)


process_summary_df.to_csv(
    PROCESS_SUMMARY_REPORT,
    index=False,
)


# =============================================================================
# 21. FINAL TERMINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("CLASS CONFIDENCE SUMMARY")
print("=" * 100)


print(
    class_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 05C SUMMARY")
print("=" * 100)


print(
    f"Input candidates            : "
    f"{len(candidate_screened)}"
)

print(
    f"HIGH confidence             : "
    f"{high_total}"
)

print(
    f"MEDIUM confidence           : "
    f"{medium_total}"
)

print(
    f"LOW confidence              : "
    f"{low_total}"
)

print(
    f"Invalid Landsat pixels      : "
    f"{invalid_landsat_total}"
)


print("\nOutputs created:")

print(
    f" - "
    f"{relative_path(SCREENED_GPKG)}"
)

print(
    f" - "
    f"{relative_path(SCREENED_CSV)}"
)

print(
    f" - "
    f"{relative_path(CLASS_SUMMARY_REPORT)}"
)

print(
    f" - "
    f"{relative_path(SPECTRAL_STATS_REPORT)}"
)

print(
    f" - "
    f"{relative_path(PROCESS_SUMMARY_REPORT)}"
)


print("\nIMPORTANT INTERPRETATION:")

print(
    "1. HIGH confidence means the candidate passed the automated "
    "WorldCover-footprint and spectral-consistency screening."
)

print(
    "2. HIGH confidence is not equivalent to independently verified ground truth."
)

print(
    "3. MEDIUM candidates require more caution and may be useful for manual review."
)

print(
    "4. LOW candidates should not automatically enter the final training pool."
)

print(
    "5. The manual audit and final model validation remain separate steps."
)

print(
    "6. No classifier was trained in Step 05C."
)

print("=" * 100)