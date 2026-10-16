"""
STEP 07C - 2021 WHOLE-AREA WORLDCOVER BENCHMARK CONSISTENCY AUDIT

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Generate the 2021 production RF prediction in memory.
2. Aggregate ESA WorldCover 2021 to the Landsat 30 m grid.
3. Convert WorldCover classes to the four research LULC classes.
4. Compare whole-area RF predictions with the WorldCover benchmark.
5. Quantify class composition differences and pixel-level agreement.
6. Examine RF confidence for agreement and disagreement pixels.

IMPORTANT:
- WorldCover is a benchmark/reference product, NOT ground truth.
- These metrics must NOT be reported as independent model accuracy.
- Final annual LULC rasters are NOT exported in this step.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio

from rasterio.enums import Resampling
from rasterio.warp import reproject

from sklearn.metrics import confusion_matrix


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LANDSAT_2021 = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landsat"
    / "annual_composites_kmc"
    / "Kaduwela_Landsat_2021.tif"
)

WORLDCOVER_2021 = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_lulc"
    / "esa_worldcover_2021_kmc_10m.tif"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "lulc"
    / "lulc_random_forest_2021_final_production.joblib"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. OUTPUTS
# =============================================================================

OVERALL_SUMMARY_CSV = (
    REPORT_DIR
    / "07c_worldcover_benchmark_overall_summary.csv"
)

CLASS_COMPARISON_CSV = (
    REPORT_DIR
    / "07c_worldcover_benchmark_class_comparison.csv"
)

CONFUSION_MATRIX_CSV = (
    REPORT_DIR
    / "07c_worldcover_benchmark_confusion_matrix.csv"
)

DISAGREEMENT_SUMMARY_CSV = (
    REPORT_DIR
    / "07c_worldcover_benchmark_disagreement_summary.csv"
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

CLASS_IDS = [
    1,
    2,
    3,
    4,
]

CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}

REQUIRED_BANDS = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
]


# ESA WorldCover -> research class mapping

WORLDCOVER_TO_RESEARCH = {
    10: 1,   # Tree cover
    20: 1,   # Shrubland
    30: 1,   # Grassland
    40: 1,   # Cropland
    50: 2,   # Built-up
    60: 3,   # Bare / sparse vegetation
    80: 4,   # Permanent water
    90: 1,   # Herbaceous wetland
    95: 1,   # Mangroves
    100: 1,  # Moss / lichen
}


# =============================================================================
# 4. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """Return project-relative path when possible."""

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def normalized_band_name(description):

    if description is None:
        return None

    value = (
        str(description)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    aliases = {
        "blue": "blue",
        "green": "green",
        "red": "red",
        "nir": "nir",
        "nearinfrared": "nir",
        "swir1": "swir1",
        "shortwaveinfrared1": "swir1",
        "swir2": "swir2",
        "shortwaveinfrared2": "swir2",
    }

    return aliases.get(
        value
    )


def get_band_lookup(
    descriptions,
    band_count,
):

    lookup = {}

    for band_number, description in enumerate(
        descriptions,
        start=1,
    ):

        name = normalized_band_name(
            description
        )

        if name is not None:

            lookup[
                name
            ] = band_number


    if set(
        REQUIRED_BANDS
    ).issubset(
        lookup.keys()
    ):

        return lookup


    if band_count == 6:

        return {
            "blue": 1,
            "green": 2,
            "red": 3,
            "nir": 4,
            "swir1": 5,
            "swir2": 6,
        }


    raise ValueError(
        "Unable to identify Landsat bands."
    )


def safe_normalized_difference(
    first,
    second,
):

    numerator = (
        first
        -
        second
    )

    denominator = (
        first
        +
        second
    )

    result = np.full(
        first.shape,
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(
            numerator
        )
        &
        np.isfinite(
            denominator
        )
        &
        (
            np.abs(
                denominator
            )
            >
            1e-12
        )
    )

    result[
        valid
    ] = (
        numerator[
            valid
        ]
        /
        denominator[
            valid
        ]
    )

    return result


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 07C - 2021 WHOLE-AREA WORLDCOVER BENCHMARK CONSISTENCY AUDIT")
print("=" * 100)


for required_file in [
    LANDSAT_2021,
    WORLDCOVER_2021,
    MODEL_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 6. LOAD PRODUCTION MODEL
# =============================================================================

model_bundle = joblib.load(
    MODEL_FILE
)

model = (
    model_bundle[
        "model"
    ]
)

MODEL_FEATURES = list(
    model_bundle[
        "features"
    ]
)


print(
    f"\nProduction model : "
    f"{relative_path(MODEL_FILE)}"
)

print(
    f"Model features   : "
    f"{MODEL_FEATURES}"
)


# =============================================================================
# 7. LOAD 2021 LANDSAT AND BUILD FEATURES
# =============================================================================

with rasterio.open(
    LANDSAT_2021
) as src:

    landsat_crs = (
        src.crs
    )

    landsat_transform = (
        src.transform
    )

    landsat_width = (
        src.width
    )

    landsat_height = (
        src.height
    )

    landsat_profile = (
        src.profile.copy()
    )

    landsat_data = src.read(
        masked=True
    )

    band_lookup = get_band_lookup(
        list(
            src.descriptions
        ),
        src.count,
    )


raw_data = np.asarray(
    landsat_data.filled(
        np.nan
    ),
    dtype=float,
)


masks = np.ma.getmaskarray(
    landsat_data
)


landsat_valid = (
    ~np.any(
        masks,
        axis=0,
    )
    &
    np.all(
        np.isfinite(
            raw_data
        ),
        axis=0,
    )
)


band_arrays = {}


for band_name in REQUIRED_BANDS:

    band_arrays[
        band_name
    ] = raw_data[
        band_lookup[
            band_name
        ]
        -
        1
    ]


blue = band_arrays[
    "blue"
]

green = band_arrays[
    "green"
]

red = band_arrays[
    "red"
]

nir = band_arrays[
    "nir"
]

swir1 = band_arrays[
    "swir1"
]

swir2 = band_arrays[
    "swir2"
]


ndvi = safe_normalized_difference(
    nir,
    red,
)

ndbi = safe_normalized_difference(
    swir1,
    nir,
)

mndwi = safe_normalized_difference(
    green,
    swir1,
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


bsi = np.full(
    blue.shape,
    np.nan,
    dtype=float,
)


valid_bsi = (
    np.isfinite(
        bsi_numerator
    )
    &
    np.isfinite(
        bsi_denominator
    )
    &
    (
        np.abs(
            bsi_denominator
        )
        >
        1e-12
    )
)


bsi[
    valid_bsi
] = (
    bsi_numerator[
        valid_bsi
    ]
    /
    bsi_denominator[
        valid_bsi
    ]
)


feature_arrays = {
    "blue": blue,
    "green": green,
    "red": red,
    "nir": nir,
    "swir1": swir1,
    "swir2": swir2,
    "ndvi": ndvi,
    "ndbi": ndbi,
    "mndwi": mndwi,
    "bsi": bsi,
}


model_valid = (
    landsat_valid.copy()
)


for feature_name in MODEL_FEATURES:

    model_valid &= np.isfinite(
        feature_arrays[
            feature_name
        ]
    )


X = pd.DataFrame(
    {
        feature_name:
        feature_arrays[
            feature_name
        ][
            model_valid
        ]

        for feature_name in MODEL_FEATURES
    }
)


# =============================================================================
# 8. PREDICT 2021 RF MAP IN MEMORY
# =============================================================================

rf_predictions = (
    model.predict(
        X
    )
    .astype(
        np.uint8
    )
)


rf_probabilities = (
    model.predict_proba(
        X
    )
)


rf_max_probability = np.max(
    rf_probabilities,
    axis=1,
)


rf_grid = np.zeros(
    (
        landsat_height,
        landsat_width,
    ),
    dtype=np.uint8,
)


rf_confidence_grid = np.full(
    (
        landsat_height,
        landsat_width,
    ),
    np.nan,
    dtype=np.float32,
)


rf_grid[
    model_valid
] = rf_predictions


rf_confidence_grid[
    model_valid
] = rf_max_probability


# =============================================================================
# 9. LOAD + MAP WORLDCOVER CLASSES
# =============================================================================

with rasterio.open(
    WORLDCOVER_2021
) as wc_src:

    worldcover_raw = (
        wc_src.read(
            1
        )
    )

    wc_transform = (
        wc_src.transform
    )

    wc_crs = (
        wc_src.crs
    )


worldcover_mapped = np.zeros(
    worldcover_raw.shape,
    dtype=np.uint8,
)


for worldcover_class, research_class in (
    WORLDCOVER_TO_RESEARCH.items()
):

    worldcover_mapped[
        worldcover_raw
        ==
        worldcover_class
    ] = research_class


# =============================================================================
# 10. AGGREGATE WORLDCOVER TO LANDSAT 30 M GRID
# =============================================================================

worldcover_30m = np.zeros(
    (
        landsat_height,
        landsat_width,
    ),
    dtype=np.uint8,
)


reproject(
    source=worldcover_mapped,
    destination=worldcover_30m,
    src_transform=wc_transform,
    src_crs=wc_crs,
    dst_transform=landsat_transform,
    dst_crs=landsat_crs,
    src_nodata=0,
    dst_nodata=0,
    resampling=Resampling.mode,
)


benchmark_valid = (
    model_valid
    &
    np.isin(
        worldcover_30m,
        CLASS_IDS,
    )
)


rf_compare = (
    rf_grid[
        benchmark_valid
    ]
)


wc_compare = (
    worldcover_30m[
        benchmark_valid
    ]
)


confidence_compare = (
    rf_confidence_grid[
        benchmark_valid
    ]
)


comparison_pixels = int(
    len(
        rf_compare
    )
)


if comparison_pixels == 0:

    raise RuntimeError(
        "No valid overlapping RF/WorldCover pixels."
    )


# =============================================================================
# 11. WHOLE-AREA AGREEMENT
# =============================================================================

agreement_mask = (
    rf_compare
    ==
    wc_compare
)


agreement_pixels = int(
    agreement_mask.sum()
)


disagreement_pixels = int(
    comparison_pixels
    -
    agreement_pixels
)


overall_agreement_percent = float(
    agreement_pixels
    /
    comparison_pixels
    *
    100
)


mean_confidence_agreement = float(
    np.mean(
        confidence_compare[
            agreement_mask
        ]
    )
)


if disagreement_pixels > 0:

    mean_confidence_disagreement = float(
        np.mean(
            confidence_compare[
                ~agreement_mask
            ]
        )
    )

else:

    mean_confidence_disagreement = (
        np.nan
    )


# =============================================================================
# 12. CONFUSION / AGREEMENT MATRIX
# =============================================================================

cm = confusion_matrix(
    wc_compare,
    rf_compare,
    labels=CLASS_IDS,
)


confusion_df = pd.DataFrame(
    cm,
    index=[
        (
            f"WorldCover_{class_id}_"
            f"{CLASS_DEFINITIONS[class_id]}"
        )
        for class_id in CLASS_IDS
    ],
    columns=[
        (
            f"RF_{class_id}_"
            f"{CLASS_DEFINITIONS[class_id]}"
        )
        for class_id in CLASS_IDS
    ],
)


confusion_df.index.name = (
    "benchmark_class"
)


# =============================================================================
# 13. CLASS COMPOSITION COMPARISON
# =============================================================================

pixel_area_km2 = float(
    abs(
        landsat_transform.a
        *
        landsat_transform.e
    )
    /
    1_000_000
)


class_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    rf_count = int(
        np.sum(
            rf_compare
            ==
            class_id
        )
    )

    wc_count = int(
        np.sum(
            wc_compare
            ==
            class_id
        )
    )


    rf_percent = float(
        rf_count
        /
        comparison_pixels
        *
        100
    )

    wc_percent = float(
        wc_count
        /
        comparison_pixels
        *
        100
    )


    class_records.append(
        {
            "class_id": (
                class_id
            ),
            "class_name": (
                class_name
            ),
            "rf_pixels": (
                rf_count
            ),
            "worldcover_pixels": (
                wc_count
            ),
            "rf_area_km2": (
                rf_count
                *
                pixel_area_km2
            ),
            "worldcover_area_km2": (
                wc_count
                *
                pixel_area_km2
            ),
            "rf_percent": (
                rf_percent
            ),
            "worldcover_percent": (
                wc_percent
            ),
            "rf_minus_worldcover_percent_points": (
                rf_percent
                -
                wc_percent
            ),
            "rf_minus_worldcover_area_km2": (
                (
                    rf_count
                    -
                    wc_count
                )
                *
                pixel_area_km2
            ),
        }
    )


class_comparison_df = pd.DataFrame(
    class_records
)


# =============================================================================
# 14. DISAGREEMENT TYPES
# =============================================================================

disagreement_records = []


for wc_class in CLASS_IDS:

    for rf_class in CLASS_IDS:

        if wc_class == rf_class:
            continue


        count = int(
            np.sum(
                (
                    wc_compare
                    ==
                    wc_class
                )
                &
                (
                    rf_compare
                    ==
                    rf_class
                )
            )
        )


        if count == 0:
            continue


        disagreement_records.append(
            {
                "worldcover_class_id": (
                    wc_class
                ),
                "worldcover_class_name": (
                    CLASS_DEFINITIONS[
                        wc_class
                    ]
                ),
                "rf_class_id": (
                    rf_class
                ),
                "rf_class_name": (
                    CLASS_DEFINITIONS[
                        rf_class
                    ]
                ),
                "pixels": (
                    count
                ),
                "area_km2": (
                    count
                    *
                    pixel_area_km2
                ),
                "percent_of_comparison_pixels": (
                    count
                    /
                    comparison_pixels
                    *
                    100
                ),
            }
        )


disagreement_df = pd.DataFrame(
    disagreement_records
)


if len(
    disagreement_df
) > 0:

    disagreement_df = (
        disagreement_df.sort_values(
            "pixels",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# 15. OVERALL SUMMARY
# =============================================================================

overall_summary_df = pd.DataFrame(
    [
        {
            "comparison_pixels": (
                comparison_pixels
            ),
            "comparison_area_km2": (
                comparison_pixels
                *
                pixel_area_km2
            ),
            "agreement_pixels": (
                agreement_pixels
            ),
            "disagreement_pixels": (
                disagreement_pixels
            ),
            "overall_benchmark_agreement_percent": (
                overall_agreement_percent
            ),
            "mean_rf_confidence_agreement_pixels": (
                mean_confidence_agreement
            ),
            "mean_rf_confidence_disagreement_pixels": (
                mean_confidence_disagreement
            ),
            "worldcover_used_as_ground_truth": (
                False
            ),
            "final_annual_maps_exported": (
                False
            ),
            "audit_status": (
                "COMPLETE"
            ),
        }
    ]
)


# =============================================================================
# 16. SAVE OUTPUTS
# =============================================================================

overall_summary_df.to_csv(
    OVERALL_SUMMARY_CSV,
    index=False,
)


class_comparison_df.to_csv(
    CLASS_COMPARISON_CSV,
    index=False,
)


confusion_df.to_csv(
    CONFUSION_MATRIX_CSV,
)


disagreement_df.to_csv(
    DISAGREEMENT_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 17. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("WHOLE-AREA BENCHMARK AGREEMENT")
print("=" * 100)


print(
    f"Comparison pixels             : "
    f"{comparison_pixels}"
)

print(
    f"Comparison area               : "
    f"{comparison_pixels * pixel_area_km2:.3f} km²"
)

print(
    f"Agreement pixels              : "
    f"{agreement_pixels}"
)

print(
    f"Disagreement pixels           : "
    f"{disagreement_pixels}"
)

print(
    f"Overall benchmark agreement   : "
    f"{overall_agreement_percent:.3f}%"
)

print(
    f"Mean RF confidence - agreement: "
    f"{mean_confidence_agreement:.6f}"
)

print(
    f"Mean RF confidence - disagree : "
    f"{mean_confidence_disagreement:.6f}"
)


print("\n" + "=" * 100)
print("CLASS COMPOSITION COMPARISON")
print("=" * 100)


print(
    class_comparison_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("WORLDCOVER VS RF AGREEMENT MATRIX")
print("=" * 100)


print(
    confusion_df.to_string()
)


print("\n" + "=" * 100)
print("LARGEST DISAGREEMENT TYPES")
print("=" * 100)


if len(
    disagreement_df
) == 0:

    print(
        "No disagreements."
    )

else:

    print(
        disagreement_df.head(
            10
        ).to_string(
            index=False
        )
    )


print("\n" + "=" * 100)
print("STEP 07C SUMMARY")
print("=" * 100)


print(
    f"WorldCover treated as ground truth : "
    f"False"
)

print(
    f"Final annual maps exported         : "
    f"False"
)

print(
    f"Audit status                       : "
    f"COMPLETE"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(OVERALL_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(CLASS_COMPARISON_CSV)}"
)

print(
    f" - {relative_path(CONFUSION_MATRIX_CSV)}"
)

print(
    f" - {relative_path(DISAGREEMENT_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. WorldCover is used only as a whole-area benchmark, not independent ground truth."
)

print(
    "2. The comparison is performed on the common Landsat 30 m grid."
)

print(
    "3. Large systematic class differences must be investigated before temporal smoothing."
)

print(
    "4. We will not create final 2010-2026 maps until this benchmark result is reviewed."
)

print("=" * 100)