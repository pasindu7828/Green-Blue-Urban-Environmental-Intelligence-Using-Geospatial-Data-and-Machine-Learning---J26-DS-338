"""
STEP 07D - WORLDCOVER PURITY-STRATIFIED WHOLE-AREA AUDIT
CORRECTED VERSION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Reproduce the 2021 production RF prediction in memory.
2. Calculate the true WorldCover class fractions inside each Landsat 30 m pixel.
3. Calculate WorldCover mapped-data coverage separately.
4. Identify the dominant WorldCover research class and its purity.
5. Compare RF predictions with the benchmark at different purity levels.
6. Determine whether disagreement is concentrated in mixed 30 m pixels.

IMPORTANT:
- WorldCover remains a benchmark, NOT ground truth.
- Zero values in binary class masks are VALID ABSENCE values,
  not NoData. They must therefore be included during averaging.
- No model retraining occurs.
- No final annual LULC maps are exported.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio

from rasterio.enums import Resampling
from rasterio.warp import reproject


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

PURITY_SUMMARY_CSV = (
    REPORT_DIR
    / "07d_worldcover_purity_agreement_summary.csv"
)

CLASS_PURITY_SUMMARY_CSV = (
    REPORT_DIR
    / "07d_worldcover_class_purity_agreement.csv"
)

PURE_PIXEL_COMPARISON_CSV = (
    REPORT_DIR
    / "07d_worldcover_high_purity_class_comparison.csv"
)

OVERALL_SUMMARY_CSV = (
    REPORT_DIR
    / "07d_worldcover_purity_audit_summary.csv"
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

WORLDCOVER_TO_RESEARCH = {
    10: 1,    # Tree cover
    20: 1,    # Shrubland
    30: 1,    # Grassland
    40: 1,    # Cropland
    50: 2,    # Built-up
    60: 3,    # Bare / sparse vegetation
    80: 4,    # Permanent water
    90: 1,    # Herbaceous wetland
    95: 1,    # Mangroves
    100: 1,   # Moss / lichen
}

# Avoid partially covered KMC edge pixels when evaluating purity.
MIN_WORLDCOVER_COVERAGE = 0.95


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


def normalized_band_name(description):
    """Standardize Landsat band names."""

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
    count,
):
    """Determine required Landsat band positions."""

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


    if count == 6:

        return {
            "blue": 1,
            "green": 2,
            "red": 3,
            "nir": 4,
            "swir1": 5,
            "swir2": 6,
        }


    raise ValueError(
        "Unable to identify required Landsat bands."
    )


def safe_normalized_difference(
    first,
    second,
):
    """Vectorized normalized difference."""

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


def purity_group(
    purity,
):
    """Assign benchmark-purity category."""

    if purity >= 0.90:
        return "A_>=90%"

    if purity >= 0.80:
        return "B_80-<90%"

    if purity >= 0.67:
        return "C_67-<80%"

    return "D_<67%_Mixed"


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 07D - WORLDCOVER PURITY-STRATIFIED WHOLE-AREA AUDIT")
print("CORRECTED FRACTION CALCULATION")
print("=" * 100)


for required_file in [
    LANDSAT_2021,
    WORLDCOVER_2021,
    MODEL_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required input not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 6. LOAD PRODUCTION MODEL
# =============================================================================

bundle = joblib.load(
    MODEL_FILE
)

model = bundle[
    "model"
]

MODEL_FEATURES = list(
    bundle[
        "features"
    ]
)


print(
    f"\nProduction model : "
    f"{relative_path(MODEL_FILE)}"
)

print(
    f"Training samples : "
    f"{bundle.get('training_samples', 'UNKNOWN')}"
)


# =============================================================================
# 7. BUILD 2021 LANDSAT MODEL FEATURES
# =============================================================================

with rasterio.open(
    LANDSAT_2021
) as src:

    landsat_data = src.read(
        masked=True
    )

    landsat_crs = src.crs
    landsat_transform = src.transform
    landsat_height = src.height
    landsat_width = src.width

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


bands = {}


for band_name in REQUIRED_BANDS:

    bands[
        band_name
    ] = raw_data[
        band_lookup[
            band_name
        ]
        -
        1
    ]


blue = bands[
    "blue"
]

green = bands[
    "green"
]

red = bands[
    "red"
]

nir = bands[
    "nir"
]

swir1 = bands[
    "swir1"
]

swir2 = bands[
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


for feature in MODEL_FEATURES:

    model_valid &= np.isfinite(
        feature_arrays[
            feature
        ]
    )


X = pd.DataFrame(
    {
        feature:
        feature_arrays[
            feature
        ][
            model_valid
        ]

        for feature in MODEL_FEATURES
    }
)


# =============================================================================
# 8. PREDICT 2021 IN MEMORY
# =============================================================================

rf_predictions = (
    model.predict(
        X
    )
    .astype(
        np.uint8
    )
)


rf_grid = np.zeros(
    (
        landsat_height,
        landsat_width,
    ),
    dtype=np.uint8,
)


rf_grid[
    model_valid
] = rf_predictions


# =============================================================================
# 9. LOAD AND MAP NATIVE WORLDCOVER
# =============================================================================

with rasterio.open(
    WORLDCOVER_2021
) as wc_src:

    wc_raw = (
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


wc_research = np.zeros(
    wc_raw.shape,
    dtype=np.uint8,
)


for source_class, research_class in (
    WORLDCOVER_TO_RESEARCH.items()
):

    wc_research[
        wc_raw
        ==
        source_class
    ] = research_class


native_mapped_mask = np.isin(
    wc_research,
    CLASS_IDS,
)


print(
    f"\nNative WorldCover mapped pixels : "
    f"{int(native_mapped_mask.sum())}"
)


# =============================================================================
# 10. CALCULATE WORLDCOVER MAPPED COVERAGE AT 30 M
# =============================================================================

# IMPORTANT:
# Here 0 means "not a mapped WorldCover research pixel".
# It is a VALID numeric value for averaging.
# Therefore src_nodata=0 MUST NOT be used.

native_coverage_binary = (
    native_mapped_mask.astype(
        np.float32
    )
)


mapped_coverage_30m = np.zeros(
    (
        landsat_height,
        landsat_width,
    ),
    dtype=np.float32,
)


reproject(
    source=native_coverage_binary,
    destination=mapped_coverage_30m,
    src_transform=wc_transform,
    src_crs=wc_crs,
    dst_transform=landsat_transform,
    dst_crs=landsat_crs,
    resampling=Resampling.average,
)


mapped_coverage_30m = np.clip(
    mapped_coverage_30m,
    0.0,
    1.0,
)


# =============================================================================
# 11. CALCULATE TRUE CLASS FRACTIONS AT 30 M
# =============================================================================

class_area_fraction_grids = {}


for class_id in CLASS_IDS:

    # 1 = target class
    # 0 = all other pixels, including other VALID LULC classes.
    #
    # ZERO MUST BE INCLUDED IN THE AVERAGE.

    source_binary = (
        wc_research
        ==
        class_id
    ).astype(
        np.float32
    )


    destination_fraction = np.zeros(
        (
            landsat_height,
            landsat_width,
        ),
        dtype=np.float32,
    )


    reproject(
        source=source_binary,
        destination=destination_fraction,
        src_transform=wc_transform,
        src_crs=wc_crs,
        dst_transform=landsat_transform,
        dst_crs=landsat_crs,
        resampling=Resampling.average,
    )


    destination_fraction = np.clip(
        destination_fraction,
        0.0,
        1.0,
    )


    class_area_fraction_grids[
        class_id
    ] = destination_fraction


fraction_stack = np.stack(
    [
        class_area_fraction_grids[
            class_id
        ]
        for class_id in CLASS_IDS
    ],
    axis=0,
)


summed_class_fraction = np.sum(
    fraction_stack,
    axis=0,
)


# =============================================================================
# 12. FRACTION QA
# =============================================================================

fraction_difference = np.abs(
    summed_class_fraction
    -
    mapped_coverage_30m
)


maximum_fraction_coverage_difference = float(
    np.max(
        fraction_difference
    )
)


print(
    f"Maximum class-fraction vs mapped-coverage difference : "
    f"{maximum_fraction_coverage_difference:.8f}"
)


if maximum_fraction_coverage_difference > 1e-4:

    raise RuntimeError(
        "Class fractions do not sum to mapped WorldCover coverage. "
        "Do not continue."
    )


# =============================================================================
# 13. SELECT WELL-COVERED LANDSAT PIXELS
# =============================================================================

benchmark_valid = (
    model_valid
    &
    (
        mapped_coverage_30m
        >=
        MIN_WORLDCOVER_COVERAGE
    )
)


comparison_pixel_count = int(
    benchmark_valid.sum()
)


if comparison_pixel_count == 0:

    raise RuntimeError(
        "No common high-coverage Landsat/WorldCover pixels found."
    )


# =============================================================================
# 14. NORMALIZE FRACTIONS BY VALID WORLDCOVER COVERAGE
# =============================================================================

normalized_fraction_stack = np.full(
    fraction_stack.shape,
    np.nan,
    dtype=np.float32,
)


for index in range(
    len(
        CLASS_IDS
    )
):

    normalized_fraction_stack[
        index
    ][
        benchmark_valid
    ] = (
        fraction_stack[
            index
        ][
            benchmark_valid
        ]
        /
        mapped_coverage_30m[
            benchmark_valid
        ]
    )


normalized_sum = np.nansum(
    normalized_fraction_stack,
    axis=0,
)


normalization_error = float(
    np.max(
        np.abs(
            normalized_sum[
                benchmark_valid
            ]
            -
            1.0
        )
    )
)


print(
    f"Maximum normalized fraction-sum error              : "
    f"{normalization_error:.8f}"
)


if normalization_error > 1e-4:

    raise RuntimeError(
        "Normalized WorldCover class fractions "
        "do not sum to 1."
    )


# =============================================================================
# 15. DOMINANT CLASS + PURITY
# =============================================================================

dominant_index = np.nanargmax(
    np.where(
        np.isnan(
            normalized_fraction_stack
        ),
        -np.inf,
        normalized_fraction_stack,
    ),
    axis=0,
)


dominant_class = np.zeros(
    (
        landsat_height,
        landsat_width,
    ),
    dtype=np.uint8,
)


for index, class_id in enumerate(
    CLASS_IDS
):

    dominant_class[
        benchmark_valid
        &
        (
            dominant_index
            ==
            index
        )
    ] = class_id


purity_grid = np.full(
    (
        landsat_height,
        landsat_width,
    ),
    np.nan,
    dtype=np.float32,
)


purity_grid[
    benchmark_valid
] = np.max(
    normalized_fraction_stack[
        :,
        benchmark_valid
    ],
    axis=0,
)


# =============================================================================
# 16. BUILD COMPARISON TABLE
# =============================================================================

rf_values = (
    rf_grid[
        benchmark_valid
    ]
)


benchmark_values = (
    dominant_class[
        benchmark_valid
    ]
)


purity_values = (
    purity_grid[
        benchmark_valid
    ]
)


coverage_values = (
    mapped_coverage_30m[
        benchmark_valid
    ]
)


comparison_df = pd.DataFrame(
    {
        "benchmark_class": (
            benchmark_values
        ),
        "rf_class": (
            rf_values
        ),
        "purity": (
            purity_values
        ),
        "worldcover_coverage": (
            coverage_values
        ),
    }
)


comparison_df[
    "agreement"
] = (
    comparison_df[
        "benchmark_class"
    ]
    ==
    comparison_df[
        "rf_class"
    ]
)


comparison_df[
    "purity_group"
] = [
    purity_group(
        value
    )
    for value in comparison_df[
        "purity"
    ]
]


# =============================================================================
# 17. PURITY DISTRIBUTION QA
# =============================================================================

print("\n" + "=" * 100)
print("PURITY VALUE QA")
print("=" * 100)


print(
    comparison_df[
        "purity"
    ]
    .describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
    .to_string()
)


unique_purity_rounded = np.unique(
    np.round(
        comparison_df[
            "purity"
        ].to_numpy(),
        4,
    )
)


print(
    f"\nUnique rounded purity values : "
    f"{len(unique_purity_rounded)}"
)


print(
    "First purity values         : "
    f"{unique_purity_rounded[:20]}"
)


# =============================================================================
# 18. AGREEMENT BY PURITY GROUP
# =============================================================================

purity_order = [
    "A_>=90%",
    "B_80-<90%",
    "C_67-<80%",
    "D_<67%_Mixed",
]


purity_records = []


for group_name in purity_order:

    subset = (
        comparison_df[
            comparison_df[
                "purity_group"
            ]
            ==
            group_name
        ]
    )


    if len(
        subset
    ) == 0:

        continue


    purity_records.append(
        {
            "purity_group": (
                group_name
            ),
            "pixels": (
                len(
                    subset
                )
            ),
            "percent_of_all_comparison_pixels": (
                len(
                    subset
                )
                /
                len(
                    comparison_df
                )
                *
                100
            ),
            "mean_worldcover_purity": (
                float(
                    subset[
                        "purity"
                    ].mean()
                )
            ),
            "agreement_pixels": (
                int(
                    subset[
                        "agreement"
                    ].sum()
                )
            ),
            "benchmark_agreement_percent": (
                float(
                    subset[
                        "agreement"
                    ].mean()
                    *
                    100
                )
            ),
        }
    )


purity_summary_df = pd.DataFrame(
    purity_records
)


# =============================================================================
# 19. CLASS × PURITY AGREEMENT
# =============================================================================

class_purity_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    for group_name in purity_order:

        subset = (
            comparison_df[
                (
                    comparison_df[
                        "benchmark_class"
                    ]
                    ==
                    class_id
                )
                &
                (
                    comparison_df[
                        "purity_group"
                    ]
                    ==
                    group_name
                )
            ]
        )


        if len(
            subset
        ) == 0:

            continue


        class_purity_records.append(
            {
                "benchmark_class_id": (
                    class_id
                ),
                "benchmark_class_name": (
                    class_name
                ),
                "purity_group": (
                    group_name
                ),
                "pixels": (
                    len(
                        subset
                    )
                ),
                "mean_purity": (
                    float(
                        subset[
                            "purity"
                        ].mean()
                    )
                ),
                "agreement_pixels": (
                    int(
                        subset[
                            "agreement"
                    ].sum()
                )
            ),
            "benchmark_agreement_percent": (
                float(
                    subset[
                        "agreement"
                    ].mean()
                    *
                    100
                )
            ),
        }
    )


class_purity_df = pd.DataFrame(
    class_purity_records
)


# =============================================================================
# 20. HIGH-PURITY COMPOSITION COMPARISON
# =============================================================================

high_purity_df = (
    comparison_df[
        comparison_df[
            "purity"
        ]
        >=
        0.90
    ]
    .copy()
)


if len(
    high_purity_df
) == 0:

    raise RuntimeError(
        "No >=90% benchmark-purity pixels found."
    )


high_purity_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    rf_count = int(
        (
            high_purity_df[
                "rf_class"
            ]
            ==
            class_id
        ).sum()
    )


    benchmark_count = int(
        (
            high_purity_df[
                "benchmark_class"
            ]
            ==
            class_id
        ).sum()
    )


    rf_percent = float(
        rf_count
        /
        len(
            high_purity_df
        )
        *
        100
    )


    benchmark_percent = float(
        benchmark_count
        /
        len(
            high_purity_df
        )
        *
        100
    )


    high_purity_records.append(
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
            "benchmark_pixels": (
                benchmark_count
            ),
            "rf_percent": (
                rf_percent
            ),
            "benchmark_percent": (
                benchmark_percent
            ),
            "difference_percent_points": (
                rf_percent
                -
                benchmark_percent
            ),
        }
    )


high_purity_comparison_df = pd.DataFrame(
    high_purity_records
)


# =============================================================================
# 21. OVERALL SUMMARY
# =============================================================================

overall_agreement = float(
    comparison_df[
        "agreement"
    ].mean()
    *
    100
)


high_purity_agreement = float(
    high_purity_df[
        "agreement"
    ].mean()
    *
    100
)


mixed_df = (
    comparison_df[
        comparison_df[
            "purity"
        ]
        <
        0.67
    ]
)


if len(
    mixed_df
) > 0:

    mixed_agreement = float(
        mixed_df[
            "agreement"
        ].mean()
        *
        100
    )

else:

    mixed_agreement = np.nan


overall_summary_df = pd.DataFrame(
    [
        {
            "comparison_pixels": (
                len(
                    comparison_df
                )
            ),
            "minimum_worldcover_coverage_required": (
                MIN_WORLDCOVER_COVERAGE
            ),
            "overall_benchmark_agreement_percent": (
                overall_agreement
            ),
            "high_purity_90_plus_pixels": (
                len(
                    high_purity_df
                )
            ),
            "high_purity_90_plus_percent_of_comparison_pixels": (
                len(
                    high_purity_df
                )
                /
                len(
                    comparison_df
                )
                *
                100
            ),
            "high_purity_90_plus_agreement_percent": (
                high_purity_agreement
            ),
            "mixed_below_67_pixels": (
                len(
                    mixed_df
                )
            ),
            "mixed_below_67_agreement_percent": (
                mixed_agreement
            ),
            "maximum_fraction_coverage_difference": (
                maximum_fraction_coverage_difference
            ),
            "maximum_normalized_fraction_sum_error": (
                normalization_error
            ),
            "worldcover_used_as_ground_truth": (
                False
            ),
            "model_retrained": (
                False
            ),
            "final_maps_exported": (
                False
            ),
            "audit_status": (
                "COMPLETE"
            ),
        }
    ]
)


# =============================================================================
# 22. SAVE OUTPUTS
# =============================================================================

purity_summary_df.to_csv(
    PURITY_SUMMARY_CSV,
    index=False,
)


class_purity_df.to_csv(
    CLASS_PURITY_SUMMARY_CSV,
    index=False,
)


high_purity_comparison_df.to_csv(
    PURE_PIXEL_COMPARISON_CSV,
    index=False,
)


overall_summary_df.to_csv(
    OVERALL_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 23. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("AGREEMENT BY WORLDCOVER 30 M PURITY")
print("=" * 100)


print(
    purity_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("CLASS-SPECIFIC AGREEMENT BY PURITY")
print("=" * 100)


print(
    class_purity_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("HIGH-PURITY (>=90%) CLASS COMPOSITION")
print("=" * 100)


print(
    high_purity_comparison_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 07D SUMMARY")
print("=" * 100)


print(
    f"Comparison pixels                  : "
    f"{len(comparison_df)}"
)

print(
    f"Minimum WorldCover coverage        : "
    f"{MIN_WORLDCOVER_COVERAGE * 100:.1f}%"
)

print(
    f"Overall benchmark agreement        : "
    f"{overall_agreement:.3f}%"
)

print(
    f"High-purity >=90% pixels           : "
    f"{len(high_purity_df)}"
)

print(
    f"High-purity >=90% area share       : "
    f"{len(high_purity_df) / len(comparison_df) * 100:.3f}%"
)

print(
    f"High-purity >=90% agreement        : "
    f"{high_purity_agreement:.3f}%"
)

print(
    f"Mixed <67% pixels                  : "
    f"{len(mixed_df)}"
)


if np.isfinite(
    mixed_agreement
):

    print(
        f"Mixed <67% agreement               : "
        f"{mixed_agreement:.3f}%"
    )

else:

    print(
        "Mixed <67% agreement               : N/A"
    )


print(
    f"WorldCover treated as ground truth : "
    f"False"
)

print(
    f"Model retrained                    : "
    f"False"
)

print(
    f"Final annual maps exported         : "
    f"False"
)

print(
    f"Overall Step 07D status            : "
    f"COMPLETE"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(PURITY_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(CLASS_PURITY_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(PURE_PIXEL_COMPARISON_CSV)}"
)

print(
    f" - {relative_path(OVERALL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. Binary-mask zeros were correctly treated as valid class absence values."
)

print(
    "2. WorldCover mapped coverage was calculated separately from class fractions."
)

print(
    "3. This is a benchmark-consistency audit, not independent accuracy assessment."
)

print(
    "4. No model parameters were changed."
)

print(
    "5. No annual LULC map was exported."
)

print(
    "6. The next methodological decision depends on these corrected purity results."
)

print("=" * 100)