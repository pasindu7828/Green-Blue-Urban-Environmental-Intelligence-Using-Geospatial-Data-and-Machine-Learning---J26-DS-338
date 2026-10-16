"""
STEP 08 - ANNUAL 2010-2026 LULC MAP EXPORT

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Apply the final production Random Forest to all annual Landsat composites.
2. Export transparent RAW annual LULC GeoTIFFs for 2010-2026.
3. Export retrospective temporally stabilized LULC GeoTIFFs.
4. Use the selected MODE_3YR method only for 2011-2024.
5. Keep 2010 and 2025 raw because they are temporal edge years.
6. Keep 2026 raw and explicitly provisional.
7. Verify the exported stabilized statistics against Step 07E.

LULC classes:
1 = Green / Vegetation
2 = Built-up
3 = Open / Bare
4 = Water
0 = NoData

IMPORTANT:
- RAW maps preserve direct annual RF predictions.
- STABILIZED maps are retrospective historical-analysis products.
- 2011-2024 use centered three-year categorical majority.
- 2010 and 2025 remain raw edge-year classifications.
- 2026 remains raw and provisional.
- Centered stabilized maps must not later be treated as leakage-safe
  predictor information for forecasting at the same date.
"""

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import rasterio


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LANDSAT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landsat"
    / "annual_composites_kmc"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "lulc"
    / "lulc_random_forest_2021_final_production.joblib"
)

RAW_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lulc"
    / "annual_maps"
    / "raw"
)

STABILIZED_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lulc"
    / "annual_maps"
    / "stabilized_mode_3yr"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

STEP_07E_ANNUAL_CSV = (
    REPORT_DIR
    / "07e_temporal_stabilization_annual_class_summary.csv"
)


RAW_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STABILIZED_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. OUTPUT REPORTS
# =============================================================================

CLASS_AREA_SUMMARY_CSV = (
    REPORT_DIR
    / "08_annual_lulc_class_area_summary.csv"
)

EXPORT_MANIFEST_CSV = (
    REPORT_DIR
    / "08_annual_lulc_export_manifest.csv"
)

METHOD_DECISION_CSV = (
    REPORT_DIR
    / "08_temporal_stabilization_method_decision.csv"
)

STEP_07E_CONSISTENCY_CSV = (
    REPORT_DIR
    / "08_consistency_check_against_step_07e.csv"
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

EXPECTED_YEARS = list(
    range(
        2010,
        2027,
    )
)

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

SELECTED_STABILIZATION_METHOD = (
    "MODE_3YR"
)

SMOOTH_FIRST_YEAR = 2011
SMOOTH_LAST_YEAR = 2024

NODATA_VALUE = 0

CONSISTENCY_TOLERANCE_PERCENT = 1e-6


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


def extract_year(path: Path) -> int:
    """Extract YYYY from annual Landsat filename."""

    match = re.search(
        r"(20\d{2})",
        path.stem,
    )

    if match is None:

        raise ValueError(
            f"Could not determine year from "
            f"{path.name}"
        )

    return int(
        match.group(1)
    )


def sensor_era(year: int) -> str:
    """Established annual composite sensor grouping."""

    if year <= 2011:

        return "Landsat 5 TM"

    if year == 2012:

        return "Landsat 5 TM + Landsat 7 ETM+"

    if 2013 <= year <= 2020:

        return "Landsat 8 OLI"

    return "Landsat 8 OLI + Landsat 9 OLI-2"


def normalized_band_name(description):
    """Normalize raster band descriptions."""

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
        "Unable to identify required Landsat bands."
    )


def safe_normalized_difference(
    first,
    second,
):
    """Calculate normalized difference safely."""

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


def predict_annual_lulc(
    raster_path: Path,
    model,
    model_features,
):
    """
    Generate one raw annual LULC prediction grid.

    Returns:
    - prediction grid
    - valid mask
    - output raster profile
    - metadata
    """

    with rasterio.open(
        raster_path
    ) as src:

        data = src.read(
            masked=True
        )

        descriptions = list(
            src.descriptions
        )

        band_lookup = get_band_lookup(
            descriptions,
            src.count,
        )

        raw_data = np.asarray(
            data.filled(
                np.nan
            ),
            dtype=float,
        )

        masks = np.ma.getmaskarray(
            data
        )

        base_valid = (
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


        valid_mask = (
            base_valid.copy()
        )


        for feature_name in model_features:

            if feature_name not in feature_arrays:

                raise ValueError(
                    f"Unable to construct model feature: "
                    f"{feature_name}"
                )

            valid_mask &= np.isfinite(
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
                    valid_mask
                ]

                for feature_name in model_features
            }
        )


        predicted_values = (
            model.predict(
                X
            )
            .astype(
                np.uint8
            )
        )


        prediction_grid = np.full(
            (
                src.height,
                src.width,
            ),
            NODATA_VALUE,
            dtype=np.uint8,
        )


        prediction_grid[
            valid_mask
        ] = predicted_values


        output_profile = (
            src.profile.copy()
        )


        output_profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=NODATA_VALUE,
            compress="deflate",
        )


        metadata = {
            "crs": (
                str(
                    src.crs
                )
            ),
            "width": (
                src.width
            ),
            "height": (
                src.height
            ),
            "transform": (
                src.transform
            ),
            "pixel_area_m2": float(
                abs(
                    src.transform.a
                    *
                    src.transform.e
                )
            ),
            "valid_pixels": int(
                valid_mask.sum()
            ),
        }


    return (
        prediction_grid,
        valid_mask,
        output_profile,
        metadata,
    )


def three_year_mode(
    previous_grid,
    current_grid,
    next_grid,
    common_valid,
):
    """
    Centered 3-year categorical majority.

    A class must receive at least 2 of 3 votes.
    If all three years disagree, retain the current raw class.
    """

    output = (
        current_grid.copy()
    )


    stacked = np.stack(
        [
            previous_grid,
            current_grid,
            next_grid,
        ],
        axis=0,
    )


    vote_counts = np.stack(
        [
            np.sum(
                stacked
                ==
                class_id,
                axis=0,
            )
            for class_id in CLASS_IDS
        ],
        axis=0,
    )


    maximum_votes = np.max(
        vote_counts,
        axis=0,
    )


    winning_index = np.argmax(
        vote_counts,
        axis=0,
    )


    winning_class_grid = np.zeros_like(
        current_grid,
        dtype=np.uint8,
    )


    for index, class_id in enumerate(
        CLASS_IDS
    ):

        winning_class_grid[
            winning_index
            ==
            index
        ] = class_id


    majority_mask = (
        common_valid
        &
        (
            maximum_votes
            >=
            2
        )
    )


    output[
        majority_mask
    ] = winning_class_grid[
        majority_mask
    ]


    return output


def write_lulc_raster(
    output_path,
    class_grid,
    profile,
    year,
    product_type,
    stabilization_applied,
    stabilization_window,
    provisional,
):
    """Write one classified LULC GeoTIFF."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    if output_path.exists():

        output_path.unlink()


    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            class_grid.astype(
                np.uint8
            ),
            1,
        )


        dst.set_band_description(
            1,
            "LULC class",
        )


        dst.update_tags(
            research_component=(
                "Urban Green Transition, Urbanization Pressure "
                "and Conservation Intelligence"
            ),
            year=str(
                year
            ),
            product_type=(
                product_type
            ),
            class_0=(
                "NoData"
            ),
            class_1=(
                "Green / Vegetation"
            ),
            class_2=(
                "Built-up"
            ),
            class_3=(
                "Open / Bare"
            ),
            class_4=(
                "Water"
            ),
            stabilization_applied=str(
                stabilization_applied
            ),
            stabilization_method=(
                SELECTED_STABILIZATION_METHOD
                if stabilization_applied
                else
                "NONE"
            ),
            stabilization_window=(
                stabilization_window
            ),
            provisional=str(
                provisional
            ),
        )


def class_summary_records(
    year,
    product_type,
    grid,
    valid_mask,
    pixel_area_m2,
    stabilization_applied,
    provisional,
):
    """Return class-area records for one map."""

    values = (
        grid[
            valid_mask
        ]
    )


    total_valid_pixels = int(
        len(
            values
        )
    )


    records = []


    for class_id, class_name in (
        CLASS_DEFINITIONS.items()
    ):

        class_pixels = int(
            np.sum(
                values
                ==
                class_id
            )
        )


        records.append(
            {
                "year": (
                    year
                ),
                "product_type": (
                    product_type
                ),
                "class_id": (
                    class_id
                ),
                "class_name": (
                    class_name
                ),
                "valid_pixels": (
                    total_valid_pixels
                ),
                "class_pixels": (
                    class_pixels
                ),
                "class_area_km2": (
                    class_pixels
                    *
                    pixel_area_m2
                    /
                    1_000_000
                ),
                "class_percent": (
                    class_pixels
                    /
                    total_valid_pixels
                    *
                    100
                ),
                "stabilization_applied": (
                    stabilization_applied
                ),
                "provisional_year": (
                    provisional
                ),
            }
        )


    return records


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 08 - ANNUAL 2010-2026 LULC MAP EXPORT")
print("=" * 100)


for required_file in [
    MODEL_FILE,
    STEP_07E_ANNUAL_CSV,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required input not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 6. LOAD FINAL PRODUCTION MODEL
# =============================================================================

bundle = joblib.load(
    MODEL_FILE
)


required_bundle_keys = {
    "model",
    "features",
    "training_samples",
}


missing_bundle_keys = (
    required_bundle_keys
    -
    set(
        bundle.keys()
    )
)


if missing_bundle_keys:

    raise ValueError(
        "Production model bundle missing keys: "
        f"{sorted(missing_bundle_keys)}"
    )


model = (
    bundle[
        "model"
    ]
)

MODEL_FEATURES = list(
    bundle[
        "features"
    ]
)


model_classes = [
    int(
        class_id
    )
    for class_id in model.classes_
]


if model_classes != CLASS_IDS:

    raise ValueError(
        f"Unexpected model class order: "
        f"{model_classes}"
    )


print(
    f"\nProduction model : "
    f"{relative_path(MODEL_FILE)}"
)

print(
    f"Training samples : "
    f"{bundle['training_samples']}"
)

print(
    f"Predictors       : "
    f"{len(MODEL_FEATURES)}"
)

print(
    f"Selected temporal method : "
    f"{SELECTED_STABILIZATION_METHOD}"
)


# =============================================================================
# 7. FIND LANDSAT RASTERS
# =============================================================================

raster_files = sorted(
    LANDSAT_DIR.glob(
        "Kaduwela_Landsat_*.tif"
    ),
    key=extract_year,
)


years_found = [
    extract_year(
        raster_path
    )
    for raster_path in raster_files
]


if years_found != EXPECTED_YEARS:

    raise ValueError(
        "Expected exactly one annual Landsat raster "
        "for each year 2010-2026."
    )


print(
    f"\nAnnual rasters : "
    f"{len(raster_files)}"
)


# =============================================================================
# 8. CREATE ALL RAW PREDICTIONS
# =============================================================================

print("\n" + "=" * 100)
print("GENERATING RAW ANNUAL LULC PREDICTIONS")
print("=" * 100)


raw_grids = {}
valid_masks = {}
profiles = {}
metadata_by_year = {}


for raster_path in raster_files:

    year = extract_year(
        raster_path
    )


    print(
        f"Predicting {year}..."
    )


    (
        prediction_grid,
        valid_mask,
        output_profile,
        metadata,
    ) = predict_annual_lulc(
        raster_path,
        model,
        MODEL_FEATURES,
    )


    raw_grids[
        year
    ] = prediction_grid

    valid_masks[
        year
    ] = valid_mask

    profiles[
        year
    ] = output_profile

    metadata_by_year[
        year
    ] = metadata


# =============================================================================
# 9. CREATE STABILIZED PRODUCT
# =============================================================================

print("\n" + "=" * 100)
print("CREATING SELECTED TEMPORALLY STABILIZED PRODUCT")
print("=" * 100)


stabilized_grids = {
    year: raw_grids[
        year
    ].copy()
    for year in EXPECTED_YEARS
}


for year in range(
    SMOOTH_FIRST_YEAR,
    SMOOTH_LAST_YEAR
    +
    1,
):

    previous_year = (
        year
        -
        1
    )

    next_year = (
        year
        +
        1
    )


    common_valid = (
        valid_masks[
            previous_year
        ]
        &
        valid_masks[
            year
        ]
        &
        valid_masks[
            next_year
        ]
    )


    stabilized_grids[
        year
    ] = three_year_mode(
        raw_grids[
            previous_year
        ],
        raw_grids[
            year
        ],
        raw_grids[
            next_year
        ],
        common_valid,
    )


    changed_pixels = int(
        np.sum(
            (
                stabilized_grids[
                    year
                ]
                !=
                raw_grids[
                    year
                ]
            )
            &
            valid_masks[
                year
            ]
        )
    )


    changed_percent = (
        changed_pixels
        /
        int(
            valid_masks[
                year
            ].sum()
        )
        *
        100
    )


    print(
        f"{year}: changed from raw = "
        f"{changed_percent:.3f}%"
    )


# =============================================================================
# 10. EXPORT RAW + STABILIZED MAPS
# =============================================================================

print("\n" + "=" * 100)
print("EXPORTING ANNUAL GEOTIFFS")
print("=" * 100)


manifest_records = []

class_area_records = []


for year in EXPECTED_YEARS:

    provisional = (
        year
        ==
        2026
    )


    # -------------------------------------------------------------------------
    # RAW
    # -------------------------------------------------------------------------

    raw_output = (
        RAW_OUTPUT_DIR
        /
        f"Kaduwela_LULC_RAW_{year}.tif"
    )


    write_lulc_raster(
        output_path=raw_output,
        class_grid=raw_grids[
            year
        ],
        profile=profiles[
            year
        ],
        year=year,
        product_type="RAW_RF",
        stabilization_applied=False,
        stabilization_window="NONE",
        provisional=provisional,
    )


    manifest_records.append(
        {
            "year": (
                year
            ),
            "sensor_era": (
                sensor_era(
                    year
                )
            ),
            "product_type": (
                "RAW_RF"
            ),
            "output_file": (
                relative_path(
                    raw_output
                )
            ),
            "stabilization_applied": (
                False
            ),
            "stabilization_method": (
                "NONE"
            ),
            "stabilization_window": (
                "NONE"
            ),
            "provisional_year": (
                provisional
            ),
            "recommended_role": (
                "Transparent direct RF output / QA"
            ),
        }
    )


    class_area_records.extend(
        class_summary_records(
            year=year,
            product_type="RAW_RF",
            grid=raw_grids[
                year
            ],
            valid_mask=valid_masks[
                year
            ],
            pixel_area_m2=metadata_by_year[
                year
            ][
                "pixel_area_m2"
            ],
            stabilization_applied=False,
            provisional=provisional,
        )
    )


    # -------------------------------------------------------------------------
    # STABILIZED / ANALYTICAL PRODUCT
    # -------------------------------------------------------------------------

    stabilized_output = (
        STABILIZED_OUTPUT_DIR
        /
        f"Kaduwela_LULC_STABILIZED_{year}.tif"
    )


    if (
        SMOOTH_FIRST_YEAR
        <=
        year
        <=
        SMOOTH_LAST_YEAR
    ):

        stabilization_applied = (
            True
        )

        stabilization_window = (
            f"{year - 1}-{year}-{year + 1}"
        )

        recommended_role = (
            "Retrospective historical LULC analysis"
        )

    elif year == 2010:

        stabilization_applied = (
            False
        )

        stabilization_window = (
            "RAW_EDGE_YEAR"
        )

        recommended_role = (
            "Historical edge year; interpret cautiously"
        )

    elif year == 2025:

        stabilization_applied = (
            False
        )

        stabilization_window = (
            "RAW_EDGE_YEAR_NO_2026_SMOOTHING"
        )

        recommended_role = (
            "Historical edge year; provisional 2026 excluded from smoothing"
        )

    else:

        stabilization_applied = (
            False
        )

        stabilization_window = (
            "RAW_PROVISIONAL_YEAR"
        )

        recommended_role = (
            "Provisional partial-year output; not confirmed long-term change"
        )


    write_lulc_raster(
        output_path=stabilized_output,
        class_grid=stabilized_grids[
            year
        ],
        profile=profiles[
            year
        ],
        year=year,
        product_type=(
            "STABILIZED_ANALYTICAL"
        ),
        stabilization_applied=(
            stabilization_applied
        ),
        stabilization_window=(
            stabilization_window
        ),
        provisional=(
            provisional
        ),
    )


    manifest_records.append(
        {
            "year": (
                year
            ),
            "sensor_era": (
                sensor_era(
                    year
                )
            ),
            "product_type": (
                "STABILIZED_ANALYTICAL"
            ),
            "output_file": (
                relative_path(
                    stabilized_output
                )
            ),
            "stabilization_applied": (
                stabilization_applied
            ),
            "stabilization_method": (
                SELECTED_STABILIZATION_METHOD
                if stabilization_applied
                else
                "NONE"
            ),
            "stabilization_window": (
                stabilization_window
            ),
            "provisional_year": (
                provisional
            ),
            "recommended_role": (
                recommended_role
            ),
        }
    )


    class_area_records.extend(
        class_summary_records(
            year=year,
            product_type=(
                "STABILIZED_ANALYTICAL"
            ),
            grid=stabilized_grids[
                year
            ],
            valid_mask=valid_masks[
                year
            ],
            pixel_area_m2=metadata_by_year[
                year
            ][
                "pixel_area_m2"
            ],
            stabilization_applied=(
                stabilization_applied
            ),
            provisional=(
                provisional
            ),
        )
    )


    print(
        f"{year}: RAW + STABILIZED exported"
    )


# =============================================================================
# 11. BUILD REPORT TABLES
# =============================================================================

manifest_df = pd.DataFrame(
    manifest_records
)


class_area_df = pd.DataFrame(
    class_area_records
)


# =============================================================================
# 12. VERIFY EXPORTED STABILIZED STATISTICS AGAINST STEP 07E
# =============================================================================

step_07e_df = pd.read_csv(
    STEP_07E_ANNUAL_CSV
)


expected_07e = (
    step_07e_df[
        step_07e_df[
            "method"
        ]
        ==
        SELECTED_STABILIZATION_METHOD
    ][
        [
            "year",
            "class_id",
            "percent",
            "area_km2",
        ]
    ]
    .copy()
    .rename(
        columns={
            "percent": (
                "step_07e_percent"
            ),
            "area_km2": (
                "step_07e_area_km2"
            ),
        }
    )
)


actual_08 = (
    class_area_df[
        class_area_df[
            "product_type"
        ]
        ==
        "STABILIZED_ANALYTICAL"
    ][
        [
            "year",
            "class_id",
            "class_percent",
            "class_area_km2",
        ]
    ]
    .copy()
    .rename(
        columns={
            "class_percent": (
                "step_08_percent"
            ),
            "class_area_km2": (
                "step_08_area_km2"
            ),
        }
    )
)


consistency_df = (
    expected_07e.merge(
        actual_08,
        on=[
            "year",
            "class_id",
        ],
        how="outer",
        validate="one_to_one",
    )
)


consistency_df[
    "absolute_percent_difference"
] = np.abs(
    consistency_df[
        "step_07e_percent"
    ]
    -
    consistency_df[
        "step_08_percent"
    ]
)


consistency_df[
    "absolute_area_difference_km2"
] = np.abs(
    consistency_df[
        "step_07e_area_km2"
    ]
    -
    consistency_df[
        "step_08_area_km2"
    ]
)


consistency_df[
    "percent_match"
] = (
    consistency_df[
        "absolute_percent_difference"
    ]
    <=
    CONSISTENCY_TOLERANCE_PERCENT
)


all_07e_percentages_match = bool(
    consistency_df[
        "percent_match"
    ].all()
)


if not all_07e_percentages_match:

    raise RuntimeError(
        "Step 08 stabilized statistics do not match "
        "the selected Step 07E MODE_3YR results."
    )


# =============================================================================
# 13. METHOD DECISION RECORD
# =============================================================================

method_decision_df = pd.DataFrame(
    [
        {
            "selected_method": (
                SELECTED_STABILIZATION_METHOD
            ),
            "selection_basis": (
                "Step 07E sensitivity analysis"
            ),
            "smoothing_years": (
                "2011-2024"
            ),
            "year_2010_status": (
                "RAW edge year"
            ),
            "year_2025_status": (
                "RAW edge year; provisional 2026 excluded"
            ),
            "year_2026_status": (
                "RAW provisional partial-year"
            ),
            "raw_maps_preserved": (
                True
            ),
            "stabilized_maps_for_retrospective_analysis": (
                True
            ),
            "stabilized_maps_leakage_safe_for_pressure_ml": (
                False
            ),
            "reason_for_selection": (
                "Large reduction in temporal flicker with negligible "
                "change in 2021 high-purity WorldCover benchmark agreement"
            ),
        }
    ]
)


# =============================================================================
# 14. SAVE REPORTS
# =============================================================================

class_area_df.to_csv(
    CLASS_AREA_SUMMARY_CSV,
    index=False,
)


manifest_df.to_csv(
    EXPORT_MANIFEST_CSV,
    index=False,
)


method_decision_df.to_csv(
    METHOD_DECISION_CSV,
    index=False,
)


consistency_df.to_csv(
    STEP_07E_CONSISTENCY_CSV,
    index=False,
)


# =============================================================================
# 15. FINAL QA
# =============================================================================

expected_map_count = (
    len(
        EXPECTED_YEARS
    )
    *
    2
)


actual_map_count = (
    len(
        list(
            RAW_OUTPUT_DIR.glob(
                "*.tif"
            )
        )
    )
    +
    len(
        list(
            STABILIZED_OUTPUT_DIR.glob(
                "*.tif"
            )
        )
    )
)


raw_count = len(
    list(
        RAW_OUTPUT_DIR.glob(
            "*.tif"
        )
    )
)


stabilized_count = len(
    list(
        STABILIZED_OUTPUT_DIR.glob(
            "*.tif"
        )
    )
)


all_maps_exist = (
    raw_count
    ==
    17
    and
    stabilized_count
    ==
    17
)


overall_status = (
    "PASS"
    if (
        all_maps_exist
        and
        all_07e_percentages_match
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 16. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("EXPORTED MAP COUNTS")
print("=" * 100)


print(
    f"RAW annual maps             : "
    f"{raw_count}"
)

print(
    f"STABILIZED annual maps      : "
    f"{stabilized_count}"
)

print(
    f"Total annual GeoTIFFs       : "
    f"{actual_map_count}"
)

print(
    f"Expected total GeoTIFFs     : "
    f"{expected_map_count}"
)


print("\n" + "=" * 100)
print("STABILIZED ANALYTICAL CLASS TRAJECTORY")
print("=" * 100)


stabilized_wide = (
    class_area_df[
        class_area_df[
            "product_type"
        ]
        ==
        "STABILIZED_ANALYTICAL"
    ]
    .pivot(
        index="year",
        columns="class_name",
        values="class_percent",
    )
    .reset_index()
)


print(
    stabilized_wide.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 07E CONSISTENCY CHECK")
print("=" * 100)


print(
    f"All selected MODE_3YR percentages match Step 07E : "
    f"{all_07e_percentages_match}"
)

print(
    f"Maximum percentage difference                      : "
    f"{consistency_df['absolute_percent_difference'].max():.10f}"
)


print("\n" + "=" * 100)
print("STEP 08 SUMMARY")
print("=" * 100)


print(
    f"Production model              : "
    f"Final 397-sample RF"
)

print(
    f"Years                         : "
    f"2010-2026"
)

print(
    f"Raw maps exported             : "
    f"{raw_count}"
)

print(
    f"Stabilized maps exported      : "
    f"{stabilized_count}"
)

print(
    f"MODE_3YR applied              : "
    f"2011-2024"
)

print(
    f"2010                           : "
    f"RAW edge year"
)

print(
    f"2025                           : "
    f"RAW edge year"
)

print(
    f"2026                           : "
    f"RAW provisional"
)

print(
    f"Centered stabilization leakage-safe for forecasting : "
    f"False"
)

print(
    f"Overall Step 08 status        : "
    f"{overall_status}"
)


print("\nGenerated directories:")

print(
    f" - {relative_path(RAW_OUTPUT_DIR)}"
)

print(
    f" - {relative_path(STABILIZED_OUTPUT_DIR)}"
)


print("\nGenerated reports:")

print(
    f" - {relative_path(CLASS_AREA_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(EXPORT_MANIFEST_CSV)}"
)

print(
    f" - {relative_path(METHOD_DECISION_CSV)}"
)

print(
    f" - {relative_path(STEP_07E_CONSISTENCY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. RAW maps preserve the direct Random Forest output."
)

print(
    "2. Stabilized maps are a separate retrospective analytical product."
)

print(
    "3. Years 2011-2024 use centered 3-year categorical majority."
)

print(
    "4. 2010 and 2025 remain unsmoothed temporal edge years."
)

print(
    "5. 2026 remains provisional and must not be interpreted as confirmed annual change."
)

print(
    "6. Future-informed stabilized maps will NOT be blindly used as leakage-safe pressure-model predictors."
)

print(
    "7. Long-term transition analysis will use persistent/consensus change rules rather than simple endpoint subtraction."
)

print("=" * 100)