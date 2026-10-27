"""
STEP 07B - TEMPORAL LULC PREDICTION STABILITY AUDIT

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Load the frozen final production Random Forest model from Step 06E.
2. Apply it temporarily to every annual Landsat composite from 2010-2026.
3. Do NOT export final annual LULC rasters yet.
4. Measure annual predicted LULC composition.
5. Measure Random Forest prediction-confidence diagnostics.
6. Measure adjacent-year pixel-level classification changes.
7. Examine built-up gains/losses and green-cover gains/losses.
8. Identify whether sensor-era boundaries produce suspicious prediction jumps.
9. Combine these diagnostics with Step 07A feature-space evidence.

IMPORTANT:
- This step is a temporal stability audit, not an accuracy assessment.
- Historical years do not have independent ground-truth labels.
- Annual predictions are temporary and held in memory.
- Final annual LULC maps are NOT exported in this step.
- 2026 remains provisional because the annual Landsat composite is partial-year data.
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

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "lulc"
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
# 2. INPUTS
# =============================================================================

PRODUCTION_MODEL_FILE = (
    MODEL_DIR
    / "lulc_random_forest_2021_final_production.joblib"
)

STEP_07A_YEAR_SUMMARY = (
    REPORT_DIR
    / "07a_temporal_transfer_year_summary.csv"
)


# =============================================================================
# 3. OUTPUTS
# =============================================================================

ANNUAL_CLASS_SUMMARY_CSV = (
    REPORT_DIR
    / "07b_annual_lulc_prediction_summary.csv"
)

ANNUAL_CONFIDENCE_CSV = (
    REPORT_DIR
    / "07b_annual_prediction_confidence_summary.csv"
)

ADJACENT_CHANGE_SUMMARY_CSV = (
    REPORT_DIR
    / "07b_adjacent_year_change_summary.csv"
)

ADJACENT_TRANSITION_MATRIX_CSV = (
    REPORT_DIR
    / "07b_adjacent_year_transition_matrix.csv"
)

SENSOR_TRANSITION_SUMMARY_CSV = (
    REPORT_DIR
    / "07b_sensor_transition_summary.csv"
)

TEMPORAL_AUDIT_SUMMARY_CSV = (
    REPORT_DIR
    / "07b_temporal_prediction_stability_summary.csv"
)


# =============================================================================
# 4. CONSTANTS
# =============================================================================

EXPECTED_YEARS = list(
    range(
        2010,
        2027,
    )
)

CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}

CLASS_IDS = [
    1,
    2,
    3,
    4,
]

REQUIRED_BANDS = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
]

LOW_CONFIDENCE_THRESHOLD_1 = 0.60

LOW_CONFIDENCE_THRESHOLD_2 = 0.70


# =============================================================================
# 5. HELPERS
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
    """Extract YYYY from Landsat filename."""

    match = re.search(
        r"(20\d{2})",
        path.stem,
    )

    if match is None:
        raise ValueError(
            f"Unable to extract year from: "
            f"{path.name}"
        )

    return int(
        match.group(1)
    )


def sensor_era(year: int) -> str:
    """
    Sensor configuration according to the established
    annual composite workflow.
    """

    if year <= 2011:
        return "Landsat 5 TM"

    if year == 2012:
        return "Landsat 5 TM + Landsat 7 ETM+"

    if 2013 <= year <= 2020:
        return "Landsat 8 OLI"

    return "Landsat 8 OLI + Landsat 9 OLI-2"


def normalized_band_name(description):
    """Standardize raster band names."""

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
    """Identify the six required Landsat bands."""

    lookup = {}

    for band_number, description in enumerate(
        descriptions,
        start=1,
    ):

        normalized_name = normalized_band_name(
            description
        )

        if normalized_name is not None:
            lookup[
                normalized_name
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
        "Unable to identify the required Landsat bands."
    )


def safe_normalized_difference(
    first,
    second,
):
    """Vectorized normalized-difference calculation."""

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
            > 1e-12
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


def prepare_raster_features(
    raster_path: Path,
    model_features,
):
    """
    Convert an annual Landsat raster into the same predictor
    matrix used by the production Random Forest.

    Returns:
    - X dataframe
    - valid model pixel mask
    - raster metadata
    """

    with rasterio.open(
        raster_path
    ) as src:

        raster_data = src.read(
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
            raster_data.filled(
                np.nan
            ),
            dtype=float,
        )

        raster_masks = np.ma.getmaskarray(
            raster_data
        )

        base_valid_mask = (
            ~np.any(
                raster_masks,
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

            band_number = (
                band_lookup[
                    band_name
                ]
            )

            band_arrays[
                band_name
            ] = (
                raw_data[
                    band_number - 1
                ]
            )


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
                > 1e-12
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


        missing_features = (
            set(
                model_features
            )
            -
            set(
                feature_arrays.keys()
            )
        )

        if missing_features:

            raise ValueError(
                "Unable to construct required model features: "
                f"{sorted(missing_features)}"
            )


        model_valid_mask = (
            base_valid_mask.copy()
        )


        for feature_name in model_features:

            model_valid_mask &= np.isfinite(
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
                    model_valid_mask
                ]

                for feature_name in model_features
            }
        )


        pixel_area_m2 = float(
            abs(
                src.transform.a
                *
                src.transform.e
            )
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
            "pixel_area_m2": (
                pixel_area_m2
            ),
            "valid_pixels": (
                int(
                    model_valid_mask.sum()
                )
            ),
        }


    return (
        X,
        model_valid_mask,
        metadata,
    )


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 07B - TEMPORAL LULC PREDICTION STABILITY AUDIT")
print("=" * 100)


for required_file in [
    PRODUCTION_MODEL_FILE,
    STEP_07A_YEAR_SUMMARY,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required input not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 7. LOAD PRODUCTION MODEL
# =============================================================================

production_bundle = joblib.load(
    PRODUCTION_MODEL_FILE
)


required_model_keys = {
    "model",
    "features",
    "class_definitions",
    "training_samples",
}


missing_model_keys = (
    required_model_keys
    -
    set(
        production_bundle.keys()
    )
)


if missing_model_keys:

    raise ValueError(
        "Production model bundle is missing keys: "
        f"{sorted(missing_model_keys)}"
    )


model = (
    production_bundle[
        "model"
    ]
)

MODEL_FEATURES = list(
    production_bundle[
        "features"
    ]
)


model_classes = [
    int(
        value
    )
    for value in model.classes_
]


if set(
    model_classes
) != set(
    CLASS_IDS
):

    raise ValueError(
        f"Unexpected model classes: "
        f"{model_classes}"
    )


print(
    f"\nProduction model    : "
    f"{production_bundle.get('model_name', 'RandomForestClassifier')}"
)

print(
    f"Training samples    : "
    f"{production_bundle['training_samples']}"
)

print(
    f"Predictor variables : "
    f"{len(MODEL_FEATURES)}"
)


# =============================================================================
# 8. FIND ANNUAL RASTERS
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


print(
    f"\nAnnual rasters found : "
    f"{len(raster_files)}"
)

print(
    f"Years                : "
    f"{years_found}"
)


if years_found != EXPECTED_YEARS:

    raise ValueError(
        "Expected exactly one raster per year "
        "from 2010 through 2026."
    )


# =============================================================================
# 9. LOAD STEP 07A DIAGNOSTICS
# =============================================================================

step_07a_df = pd.read_csv(
    STEP_07A_YEAR_SUMMARY
)


if sorted(
    step_07a_df[
        "year"
    ].astype(int).tolist()
) != EXPECTED_YEARS:

    raise ValueError(
        "Step 07A summary does not contain all 2010-2026 years."
    )


# =============================================================================
# 10. TEMPORARY ANNUAL PREDICTIONS
# =============================================================================

print("\n" + "=" * 100)
print("TEMPORARY ANNUAL PREDICTIONS")
print("=" * 100)


annual_class_records = []

annual_confidence_records = []

prediction_grids = {}

valid_masks = {}

metadata_by_year = {}


for raster_path in raster_files:

    year = extract_year(
        raster_path
    )


    print(
        f"Predicting {year}..."
    )


    (
        X,
        valid_mask,
        raster_metadata,
    ) = prepare_raster_features(
        raster_path,
        MODEL_FEATURES,
    )


    predictions = (
        model.predict(
            X
        )
        .astype(
            np.int16
        )
    )


    probabilities = (
        model.predict_proba(
            X
        )
    )


    maximum_probability = np.max(
        probabilities,
        axis=1,
    )


    prediction_grid = np.zeros(
        (
            raster_metadata[
                "height"
            ],
            raster_metadata[
                "width"
            ],
        ),
        dtype=np.int16,
    )


    prediction_grid[
        valid_mask
    ] = predictions


    prediction_grids[
        year
    ] = prediction_grid


    valid_masks[
        year
    ] = valid_mask.copy()


    metadata_by_year[
        year
    ] = raster_metadata


    valid_pixel_count = len(
        predictions
    )


    valid_area_km2 = (
        valid_pixel_count
        *
        raster_metadata[
            "pixel_area_m2"
        ]
        /
        1_000_000
    )


    for class_id, class_name in (
        CLASS_DEFINITIONS.items()
    ):

        class_pixel_count = int(
            np.sum(
                predictions
                ==
                class_id
            )
        )


        class_area_km2 = float(
            class_pixel_count
            *
            raster_metadata[
                "pixel_area_m2"
            ]
            /
            1_000_000
        )


        class_percent = float(
            class_pixel_count
            /
            valid_pixel_count
            *
            100
        )


        annual_class_records.append(
            {
                "year": (
                    year
                ),
                "sensor_era": (
                    sensor_era(
                        year
                    )
                ),
                "provisional_year": (
                    year
                    == 2026
                ),
                "class_id": (
                    class_id
                ),
                "class_name": (
                    class_name
                ),
                "valid_pixels": (
                    valid_pixel_count
                ),
                "valid_area_km2": (
                    valid_area_km2
                ),
                "class_pixels": (
                    class_pixel_count
                ),
                "class_area_km2": (
                    class_area_km2
                ),
                "class_percent": (
                    class_percent
                ),
            }
        )


    annual_confidence_records.append(
        {
            "year": (
                year
            ),
            "sensor_era": (
                sensor_era(
                    year
                )
            ),
            "provisional_year": (
                year
                == 2026
            ),
            "valid_pixels": (
                valid_pixel_count
            ),
            "mean_max_probability": (
                float(
                    np.mean(
                        maximum_probability
                    )
                )
            ),
            "median_max_probability": (
                float(
                    np.median(
                        maximum_probability
                    )
                )
            ),
            "p05_max_probability": (
                float(
                    np.percentile(
                        maximum_probability,
                        5,
                    )
                )
            ),
            "p25_max_probability": (
                float(
                    np.percentile(
                        maximum_probability,
                        25,
                    )
                )
            ),
            "percent_probability_below_0_60": (
                float(
                    np.mean(
                        maximum_probability
                        <
                        LOW_CONFIDENCE_THRESHOLD_1
                    )
                    *
                    100
                )
            ),
            "percent_probability_below_0_70": (
                float(
                    np.mean(
                        maximum_probability
                        <
                        LOW_CONFIDENCE_THRESHOLD_2
                    )
                    *
                    100
                )
            ),
        }
    )


annual_class_df = pd.DataFrame(
    annual_class_records
)


annual_confidence_df = pd.DataFrame(
    annual_confidence_records
)


# =============================================================================
# 11. ADJACENT-YEAR TEMPORAL CHANGE
# =============================================================================

print("\n" + "=" * 100)
print("ADJACENT-YEAR PREDICTION CHANGE")
print("=" * 100)


change_summary_records = []

transition_matrix_records = []


for previous_year, current_year in zip(
    EXPECTED_YEARS[:-1],
    EXPECTED_YEARS[1:],
):

    previous_prediction = (
        prediction_grids[
            previous_year
        ]
    )

    current_prediction = (
        prediction_grids[
            current_year
        ]
    )


    common_valid_mask = (
        valid_masks[
            previous_year
        ]
        &
        valid_masks[
            current_year
        ]
    )


    previous_classes = (
        previous_prediction[
            common_valid_mask
        ]
    )


    current_classes = (
        current_prediction[
            common_valid_mask
        ]
    )


    comparison_pixel_count = len(
        previous_classes
    )


    pixel_area_m2 = (
        metadata_by_year[
            current_year
        ][
            "pixel_area_m2"
        ]
    )


    comparison_area_km2 = (
        comparison_pixel_count
        *
        pixel_area_m2
        /
        1_000_000
    )


    changed_mask = (
        previous_classes
        !=
        current_classes
    )


    changed_pixels = int(
        changed_mask.sum()
    )


    unchanged_pixels = int(
        comparison_pixel_count
        -
        changed_pixels
    )


    changed_percent = float(
        changed_pixels
        /
        comparison_pixel_count
        *
        100
    )


    previous_built = (
        previous_classes
        ==
        2
    )

    current_built = (
        current_classes
        ==
        2
    )


    built_gain_pixels = int(
        (
            ~previous_built
            &
            current_built
        ).sum()
    )


    built_loss_pixels = int(
        (
            previous_built
            &
            ~current_built
        ).sum()
    )


    previous_green = (
        previous_classes
        ==
        1
    )

    current_green = (
        current_classes
        ==
        1
    )


    green_gain_pixels = int(
        (
            ~previous_green
            &
            current_green
        ).sum()
    )


    green_loss_pixels = int(
        (
            previous_green
            &
            ~current_green
        ).sum()
    )


    green_to_built_pixels = int(
        (
            (
                previous_classes
                ==
                1
            )
            &
            (
                current_classes
                ==
                2
            )
        ).sum()
    )


    built_to_green_pixels = int(
        (
            (
                previous_classes
                ==
                2
            )
            &
            (
                current_classes
                ==
                1
            )
        ).sum()
    )


    previous_sensor = sensor_era(
        previous_year
    )

    current_sensor = sensor_era(
        current_year
    )


    change_summary_records.append(
        {
            "from_year": (
                previous_year
            ),
            "to_year": (
                current_year
            ),
            "from_sensor_era": (
                previous_sensor
            ),
            "to_sensor_era": (
                current_sensor
            ),
            "sensor_era_changed": (
                previous_sensor
                !=
                current_sensor
            ),
            "comparison_pixels": (
                comparison_pixel_count
            ),
            "comparison_area_km2": (
                comparison_area_km2
            ),
            "unchanged_pixels": (
                unchanged_pixels
            ),
            "changed_pixels": (
                changed_pixels
            ),
            "changed_percent": (
                changed_percent
            ),
            "built_gain_pixels": (
                built_gain_pixels
            ),
            "built_gain_km2": (
                built_gain_pixels
                *
                pixel_area_m2
                /
                1_000_000
            ),
            "built_loss_pixels": (
                built_loss_pixels
            ),
            "built_loss_km2": (
                built_loss_pixels
                *
                pixel_area_m2
                /
                1_000_000
            ),
            "built_net_change_km2": (
                (
                    built_gain_pixels
                    -
                    built_loss_pixels
                )
                *
                pixel_area_m2
                /
                1_000_000
            ),
            "green_gain_pixels": (
                green_gain_pixels
            ),
            "green_gain_km2": (
                green_gain_pixels
                *
                pixel_area_m2
                /
                1_000_000
            ),
            "green_loss_pixels": (
                green_loss_pixels
            ),
            "green_loss_km2": (
                green_loss_pixels
                *
                pixel_area_m2
                /
                1_000_000
            ),
            "green_net_change_km2": (
                (
                    green_gain_pixels
                    -
                    green_loss_pixels
                )
                *
                pixel_area_m2
                /
                1_000_000
            ),
            "green_to_built_pixels": (
                green_to_built_pixels
            ),
            "green_to_built_km2": (
                green_to_built_pixels
                *
                pixel_area_m2
                /
                1_000_000
            ),
            "built_to_green_pixels": (
                built_to_green_pixels
            ),
            "built_to_green_km2": (
                built_to_green_pixels
                *
                pixel_area_m2
                /
                1_000_000
            ),
        }
    )


    for from_class_id in CLASS_IDS:

        for to_class_id in CLASS_IDS:

            transition_pixels = int(
                np.sum(
                    (
                        previous_classes
                        ==
                        from_class_id
                    )
                    &
                    (
                        current_classes
                        ==
                        to_class_id
                    )
                )
            )


            transition_matrix_records.append(
                {
                    "from_year": (
                        previous_year
                    ),
                    "to_year": (
                        current_year
                    ),
                    "sensor_era_changed": (
                        previous_sensor
                        !=
                        current_sensor
                    ),
                    "from_class_id": (
                        from_class_id
                    ),
                    "from_class_name": (
                        CLASS_DEFINITIONS[
                            from_class_id
                        ]
                    ),
                    "to_class_id": (
                        to_class_id
                    ),
                    "to_class_name": (
                        CLASS_DEFINITIONS[
                            to_class_id
                        ]
                    ),
                    "transition_pixels": (
                        transition_pixels
                    ),
                    "transition_area_km2": (
                        transition_pixels
                        *
                        pixel_area_m2
                        /
                        1_000_000
                    ),
                    "percent_of_comparison_area": (
                        transition_pixels
                        /
                        comparison_pixel_count
                        *
                        100
                    ),
                }
            )


change_summary_df = pd.DataFrame(
    change_summary_records
)


transition_matrix_df = pd.DataFrame(
    transition_matrix_records
)


# =============================================================================
# 12. ANNUAL WIDE CLASS TABLE
# =============================================================================

class_percent_wide = (
    annual_class_df.pivot(
        index="year",
        columns="class_id",
        values="class_percent",
    )
    .rename(
        columns={
            1: "green_percent",
            2: "built_percent",
            3: "open_bare_percent",
            4: "water_percent",
        }
    )
    .reset_index()
)


class_area_wide = (
    annual_class_df.pivot(
        index="year",
        columns="class_id",
        values="class_area_km2",
    )
    .rename(
        columns={
            1: "green_area_km2",
            2: "built_area_km2",
            3: "open_bare_area_km2",
            4: "water_area_km2",
        }
    )
    .reset_index()
)


annual_wide_df = (
    class_percent_wide.merge(
        class_area_wide,
        on="year",
        how="left",
    )
)


annual_wide_df[
    "built_area_change_from_previous_year_km2"
] = (
    annual_wide_df[
        "built_area_km2"
    ].diff()
)


annual_wide_df[
    "green_area_change_from_previous_year_km2"
] = (
    annual_wide_df[
        "green_area_km2"
    ].diff()
)


annual_wide_df[
    "built_percent_change_from_previous_year_pp"
] = (
    annual_wide_df[
        "built_percent"
    ].diff()
)


annual_wide_df[
    "green_percent_change_from_previous_year_pp"
] = (
    annual_wide_df[
        "green_percent"
    ].diff()
)


# =============================================================================
# 13. SENSOR-ERA TRANSITION SUMMARY
# =============================================================================

sensor_transition_df = (
    change_summary_df[
        change_summary_df[
            "sensor_era_changed"
        ]
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 14. MERGE STEP 07A + 07B YEAR DIAGNOSTICS
# =============================================================================

combined_year_df = (
    annual_wide_df.merge(
        annual_confidence_df[
            [
                "year",
                "mean_max_probability",
                "median_max_probability",
                "p05_max_probability",
                "percent_probability_below_0_60",
                "percent_probability_below_0_70",
            ]
        ],
        on="year",
        how="left",
    )
)


combined_year_df = (
    combined_year_df.merge(
        step_07a_df[
            [
                "year",
                "sensor_era",
                "features_with_median_shift_flag",
                "features_with_outside_range_increase_flag",
                "mean_absolute_median_shift_in_2021_iqr",
                "max_absolute_median_shift_in_2021_iqr",
            ]
        ],
        on="year",
        how="left",
    )
)


combined_year_df[
    "provisional_year"
] = (
    combined_year_df[
        "year"
    ]
    ==
    2026
)


# =============================================================================
# 15. BASIC AUDIT COMPLETENESS CHECKS
# =============================================================================

all_years_predicted = (
    sorted(
        combined_year_df[
            "year"
        ].astype(int).tolist()
    )
    ==
    EXPECTED_YEARS
)


all_probabilities_valid = bool(
    (
        (
            annual_confidence_df[
                "mean_max_probability"
            ]
            >= 0
        )
        &
        (
            annual_confidence_df[
                "mean_max_probability"
            ]
            <= 1
        )
    ).all()
)


all_class_percent_totals = (
    annual_class_df.groupby(
        "year"
    )[
        "class_percent"
    ]
    .sum()
)


class_percent_totals_valid = bool(
    np.allclose(
        all_class_percent_totals.to_numpy(),
        100.0,
        atol=1e-6,
    )
)


all_07a_shift_flags_zero = bool(
    (
        step_07a_df[
            "features_with_median_shift_flag"
        ]
        ==
        0
    ).all()
    and
    (
        step_07a_df[
            "features_with_outside_range_increase_flag"
        ]
        ==
        0
    ).all()
)


audit_complete = all(
    [
        all_years_predicted,
        all_probabilities_valid,
        class_percent_totals_valid,
    ]
)


# =============================================================================
# 16. OVERALL TEMPORAL AUDIT SUMMARY
# =============================================================================

largest_change_row = (
    change_summary_df.loc[
        change_summary_df[
            "changed_percent"
        ].idxmax()
    ]
)


lowest_confidence_row = (
    annual_confidence_df.loc[
        annual_confidence_df[
            "mean_max_probability"
        ].idxmin()
    ]
)


summary_df = pd.DataFrame(
    [
        {
            "annual_years_predicted": (
                len(
                    combined_year_df
                )
            ),
            "first_year": (
                int(
                    combined_year_df[
                        "year"
                    ].min()
                )
            ),
            "last_year": (
                int(
                    combined_year_df[
                        "year"
                    ].max()
                )
            ),
            "all_years_predicted": (
                all_years_predicted
            ),
            "all_probability_values_valid": (
                all_probabilities_valid
            ),
            "annual_class_percent_totals_valid": (
                class_percent_totals_valid
            ),
            "step_07a_all_shift_flags_zero": (
                all_07a_shift_flags_zero
            ),
            "lowest_mean_prediction_confidence_year": (
                int(
                    lowest_confidence_row[
                        "year"
                    ]
                )
            ),
            "lowest_mean_prediction_confidence": (
                float(
                    lowest_confidence_row[
                        "mean_max_probability"
                    ]
                )
            ),
            "largest_adjacent_change_from_year": (
                int(
                    largest_change_row[
                        "from_year"
                    ]
                )
            ),
            "largest_adjacent_change_to_year": (
                int(
                    largest_change_row[
                        "to_year"
                    ]
                )
            ),
            "largest_adjacent_change_percent": (
                float(
                    largest_change_row[
                        "changed_percent"
                    ]
                )
            ),
            "sensor_transition_pairs": (
                len(
                    sensor_transition_df
                )
            ),
            "historical_ground_truth_accuracy_available": (
                False
            ),
            "final_maps_exported_in_this_step": (
                False
            ),
            "year_2026_provisional": (
                True
            ),
            "audit_status": (
                "AUDIT_COMPLETE"
                if audit_complete
                else
                "REVIEW_REQUIRED"
            ),
        }
    ]
)


# =============================================================================
# 17. SAVE REPORTS
# =============================================================================

annual_class_df.to_csv(
    ANNUAL_CLASS_SUMMARY_CSV,
    index=False,
)


annual_confidence_df.to_csv(
    ANNUAL_CONFIDENCE_CSV,
    index=False,
)


change_summary_df.to_csv(
    ADJACENT_CHANGE_SUMMARY_CSV,
    index=False,
)


transition_matrix_df.to_csv(
    ADJACENT_TRANSITION_MATRIX_CSV,
    index=False,
)


sensor_transition_df.to_csv(
    SENSOR_TRANSITION_SUMMARY_CSV,
    index=False,
)


combined_year_df.to_csv(
    TEMPORAL_AUDIT_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 18. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("ANNUAL PREDICTED LULC COMPOSITION")
print("=" * 100)


print(
    combined_year_df[
        [
            "year",
            "sensor_era",
            "green_percent",
            "built_percent",
            "open_bare_percent",
            "water_percent",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("ANNUAL PREDICTION CONFIDENCE")
print("=" * 100)


print(
    annual_confidence_df[
        [
            "year",
            "mean_max_probability",
            "median_max_probability",
            "p05_max_probability",
            "percent_probability_below_0_60",
            "percent_probability_below_0_70",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("ADJACENT-YEAR CHANGE SUMMARY")
print("=" * 100)


print(
    change_summary_df[
        [
            "from_year",
            "to_year",
            "sensor_era_changed",
            "changed_percent",
            "built_gain_km2",
            "built_loss_km2",
            "built_net_change_km2",
            "green_gain_km2",
            "green_loss_km2",
            "green_net_change_km2",
            "green_to_built_km2",
            "built_to_green_km2",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("SENSOR-ERA TRANSITION PAIRS")
print("=" * 100)


if len(
    sensor_transition_df
) == 0:

    print(
        "No sensor-era transitions found."
    )

else:

    print(
        sensor_transition_df[
            [
                "from_year",
                "to_year",
                "from_sensor_era",
                "to_sensor_era",
                "changed_percent",
                "built_net_change_km2",
                "green_net_change_km2",
            ]
        ].to_string(
            index=False
        )
    )


print("\n" + "=" * 100)
print("STEP 07B SUMMARY")
print("=" * 100)


print(
    f"Years predicted                     : "
    f"{len(combined_year_df)}"
)

print(
    f"All 2010-2026 years present         : "
    f"{all_years_predicted}"
)

print(
    f"Annual class totals equal 100%      : "
    f"{class_percent_totals_valid}"
)

print(
    f"Step 07A shift flags all zero       : "
    f"{all_07a_shift_flags_zero}"
)

print(
    f"Lowest mean confidence year         : "
    f"{int(lowest_confidence_row['year'])}"
)

print(
    f"Lowest mean confidence              : "
    f"{float(lowest_confidence_row['mean_max_probability']):.6f}"
)

print(
    f"Largest adjacent prediction change  : "
    f"{int(largest_change_row['from_year'])}"
    f" -> "
    f"{int(largest_change_row['to_year'])}"
)

print(
    f"Largest changed-pixel percentage    : "
    f"{float(largest_change_row['changed_percent']):.3f}%"
)

print(
    f"Final annual rasters exported       : "
    f"False"
)

print(
    f"2026 provisional                    : "
    f"True"
)

print(
    f"Overall Step 07B status             : "
    f"{'AUDIT_COMPLETE' if audit_complete else 'REVIEW_REQUIRED'}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(ANNUAL_CLASS_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(ANNUAL_CONFIDENCE_CSV)}"
)

print(
    f" - {relative_path(ADJACENT_CHANGE_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(ADJACENT_TRANSITION_MATRIX_CSV)}"
)

print(
    f" - {relative_path(SENSOR_TRANSITION_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(TEMPORAL_AUDIT_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. These are temporary annual model predictions for temporal QA."
)

print(
    "2. No final annual LULC GeoTIFFs were exported."
)

print(
    "3. Historical classification accuracy cannot be claimed without historical reference labels."
)

print(
    "4. Year-to-year changes can include true change, seasonal variation and classification noise."
)

print(
    "5. Sensor-transition pairs require particular inspection before annual maps are finalized."
)

print(
    "6. 2026 must remain labelled as provisional because it is a partial-year composite."
)

print(
    "7. The Step 07A and 07B evidence together will determine whether direct temporal transfer is acceptable."
)

print("=" * 100)