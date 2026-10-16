"""
STEP 07E - TEMPORAL STABILIZATION SENSITIVITY AUDIT

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Reproduce annual RF predictions for 2010-2026.
2. Compare raw annual predictions against several temporal stabilization methods.
3. Quantify how much each method reduces annual classification flicker.
4. Evaluate whether stabilization materially damages 2021 high-purity
   WorldCover benchmark consistency.
5. Select NO method automatically.
6. Export diagnostic reports only.

Methods compared:
- RAW
- MODE_3YR
- PROB_SUM_3YR
- CONFIDENCE_GATED_PROB_3YR

Important methodological rules:
- 2010 remains raw because no preceding year exists.
- 2025 remains raw because using provisional 2026 to smooth 2025
  would contaminate a complete historical year.
- 2026 remains raw and provisional.
- Therefore temporal smoothing is applied only to 2011-2024.
- No final annual LULC rasters are exported.
- WorldCover remains a benchmark, not ground truth.
"""

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import rasterio

from rasterio.enums import Resampling
from rasterio.warp import reproject


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LANDSAT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landsat"
    / "annual_composites_kmc"
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

METHOD_SUMMARY_CSV = (
    REPORT_DIR
    / "07e_temporal_stabilization_method_summary.csv"
)

ANNUAL_CLASS_CSV = (
    REPORT_DIR
    / "07e_temporal_stabilization_annual_class_summary.csv"
)

ADJACENT_CHANGE_CSV = (
    REPORT_DIR
    / "07e_temporal_stabilization_adjacent_change_summary.csv"
)

BENCHMARK_CSV = (
    REPORT_DIR
    / "07e_temporal_stabilization_2021_benchmark_summary.csv"
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

MIN_WORLDCOVER_COVERAGE = 0.95
HIGH_PURITY_THRESHOLD = 0.90

CONFIDENCE_GATE = 0.90

SMOOTH_FIRST_YEAR = 2011
SMOOTH_LAST_YEAR = 2024


# =============================================================================
# 4. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def extract_year(path: Path) -> int:

    match = re.search(
        r"(20\d{2})",
        path.stem,
    )

    if match is None:

        raise ValueError(
            f"Unable to determine year from {path.name}"
        )

    return int(
        match.group(1)
    )


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
    count,
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

    output = np.full(
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

    output[
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

    return output


def annual_prediction(
    raster_path,
    model,
    model_features,
):
    """
    Generate prediction + probability grids for one annual raster.
    """

    with rasterio.open(
        raster_path
    ) as src:

        raster_data = src.read(
            masked=True
        )

        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "pixel_area_m2": float(
                abs(
                    src.transform.a
                    *
                    src.transform.e
                )
            ),
        }

        band_lookup = get_band_lookup(
            list(
                src.descriptions
            ),
            src.count,
        )


    raw_data = np.asarray(
        raster_data.filled(
            np.nan
        ),
        dtype=float,
    )


    masks = np.ma.getmaskarray(
        raster_data
    )


    valid_mask = (
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


    for feature in model_features:

        valid_mask &= np.isfinite(
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
                valid_mask
            ]

            for feature in model_features
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


    predicted_probabilities = (
        model.predict_proba(
            X
        )
        .astype(
            np.float32
        )
    )


    prediction_grid = np.zeros(
        (
            metadata[
                "height"
            ],
            metadata[
                "width"
            ],
        ),
        dtype=np.uint8,
    )


    prediction_grid[
        valid_mask
    ] = predicted_values


    probability_grid = np.full(
        (
            len(
                CLASS_IDS
            ),
            metadata[
                "height"
            ],
            metadata[
                "width"
            ],
        ),
        np.nan,
        dtype=np.float32,
    )


    for probability_index in range(
        len(
            CLASS_IDS
        )
    ):

        layer = probability_grid[
            probability_index
        ]

        layer[
            valid_mask
        ] = predicted_probabilities[
            :,
            probability_index
        ]


    return (
        prediction_grid,
        probability_grid,
        valid_mask,
        metadata,
    )


def three_year_mode(
    previous_grid,
    current_grid,
    next_grid,
    common_valid,
):
    """
    Three-year categorical majority.
    Ties retain the current year's raw class.
    """

    output = current_grid.copy()


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


    winner_grid = np.zeros_like(
        current_grid
    )


    for index, class_id in enumerate(
        CLASS_IDS
    ):

        winner_grid[
            winning_index
            ==
            index
        ] = class_id


    majority_exists = (
        common_valid
        &
        (
            maximum_votes
            >=
            2
        )
    )


    output[
        majority_exists
    ] = winner_grid[
        majority_exists
    ]


    return output


def three_year_probability_sum(
    previous_probability,
    current_probability,
    next_probability,
    current_grid,
    common_valid,
):
    """
    Sum class probabilities over three years and select the
    class with highest cumulative probability.
    """

    output = current_grid.copy()


    summed_probability = (
        previous_probability
        +
        current_probability
        +
        next_probability
    )


    winning_index = np.nanargmax(
        np.where(
            np.isnan(
                summed_probability
            ),
            -np.inf,
            summed_probability,
        ),
        axis=0,
    )


    winner_grid = np.zeros_like(
        current_grid
    )


    for index, class_id in enumerate(
        CLASS_IDS
    ):

        winner_grid[
            winning_index
            ==
            index
        ] = class_id


    output[
        common_valid
    ] = winner_grid[
        common_valid
    ]


    return output


def high_purity_worldcover_benchmark(
    landsat_metadata,
):
    """
    Create >=90% WorldCover-purity benchmark on the Landsat grid.
    """

    with rasterio.open(
        WORLDCOVER_2021
    ) as src:

        wc_raw = src.read(
            1
        )

        wc_transform = src.transform
        wc_crs = src.crs


    wc_research = np.zeros(
        wc_raw.shape,
        dtype=np.uint8,
    )


    for original_class, research_class in (
        WORLDCOVER_TO_RESEARCH.items()
    ):

        wc_research[
            wc_raw
            ==
            original_class
        ] = research_class


    mapped_native = np.isin(
        wc_research,
        CLASS_IDS,
    )


    coverage_30m = np.zeros(
        (
            landsat_metadata[
                "height"
            ],
            landsat_metadata[
                "width"
            ],
        ),
        dtype=np.float32,
    )


    reproject(
        source=mapped_native.astype(
            np.float32
        ),
        destination=coverage_30m,
        src_transform=wc_transform,
        src_crs=wc_crs,
        dst_transform=landsat_metadata[
            "transform"
        ],
        dst_crs=landsat_metadata[
            "crs"
        ],
        resampling=Resampling.average,
    )


    class_fraction_layers = []


    for class_id in CLASS_IDS:

        source_binary = (
            wc_research
            ==
            class_id
        ).astype(
            np.float32
        )


        destination = np.zeros_like(
            coverage_30m
        )


        reproject(
            source=source_binary,
            destination=destination,
            src_transform=wc_transform,
            src_crs=wc_crs,
            dst_transform=landsat_metadata[
                "transform"
            ],
            dst_crs=landsat_metadata[
                "crs"
            ],
            resampling=Resampling.average,
        )


        class_fraction_layers.append(
            destination
        )


    fraction_stack = np.stack(
        class_fraction_layers,
        axis=0,
    )


    coverage_valid = (
        coverage_30m
        >=
        MIN_WORLDCOVER_COVERAGE
    )


    normalized_fractions = np.zeros_like(
        fraction_stack,
        dtype=np.float32,
    )


    for index in range(
        len(
            CLASS_IDS
        )
    ):

        normalized_fractions[
            index
        ][
            coverage_valid
        ] = (
            fraction_stack[
                index
            ][
                coverage_valid
            ]
            /
            coverage_30m[
                coverage_valid
            ]
        )


    dominant_index = np.argmax(
        normalized_fractions,
        axis=0,
    )


    dominant_class = np.zeros(
        coverage_30m.shape,
        dtype=np.uint8,
    )


    for index, class_id in enumerate(
        CLASS_IDS
    ):

        dominant_class[
            coverage_valid
            &
            (
                dominant_index
                ==
                index
            )
        ] = class_id


    purity = np.max(
        normalized_fractions,
        axis=0,
    )


    high_purity_mask = (
        coverage_valid
        &
        (
            purity
            >=
            HIGH_PURITY_THRESHOLD
        )
    )


    return (
        dominant_class,
        high_purity_mask,
    )


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 07E - TEMPORAL STABILIZATION SENSITIVITY AUDIT")
print("=" * 100)


if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Production model not found:\n"
        f"{MODEL_FILE}"
    )


if not WORLDCOVER_2021.exists():

    raise FileNotFoundError(
        f"WorldCover benchmark not found:\n"
        f"{WORLDCOVER_2021}"
    )


# =============================================================================
# 6. LOAD MODEL
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


if [
    int(
        value
    )
    for value in model.classes_
] != CLASS_IDS:

    raise ValueError(
        "Unexpected Random Forest class order."
    )


# =============================================================================
# 7. FIND ANNUAL RASTERS
# =============================================================================

raster_files = sorted(
    LANDSAT_DIR.glob(
        "Kaduwela_Landsat_*.tif"
    ),
    key=extract_year,
)


years_found = [
    extract_year(
        path
    )
    for path in raster_files
]


if years_found != EXPECTED_YEARS:

    raise ValueError(
        "Expected one Landsat raster for each year 2010-2026."
    )


# =============================================================================
# 8. PRODUCE RAW ANNUAL PREDICTIONS
# =============================================================================

print("\nCreating raw annual prediction arrays...")


raw_predictions = {}
probability_grids = {}
valid_masks = {}
metadata_by_year = {}


for raster_path in raster_files:

    year = extract_year(
        raster_path
    )

    print(
        f"  {year}"
    )


    (
        prediction_grid,
        probability_grid,
        valid_mask,
        metadata,
    ) = annual_prediction(
        raster_path,
        model,
        MODEL_FEATURES,
    )


    raw_predictions[
        year
    ] = prediction_grid

    probability_grids[
        year
    ] = probability_grid

    valid_masks[
        year
    ] = valid_mask

    metadata_by_year[
        year
    ] = metadata


# =============================================================================
# 9. CREATE METHODS
# =============================================================================

methods = {
    "RAW": {
        year: raw_predictions[
            year
        ].copy()
        for year in EXPECTED_YEARS
    },
    "MODE_3YR": {
        year: raw_predictions[
            year
        ].copy()
        for year in EXPECTED_YEARS
    },
    "PROB_SUM_3YR": {
        year: raw_predictions[
            year
        ].copy()
        for year in EXPECTED_YEARS
    },
    "CONFIDENCE_GATED_PROB_3YR": {
        year: raw_predictions[
            year
        ].copy()
        for year in EXPECTED_YEARS
    },
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


    methods[
        "MODE_3YR"
    ][
        year
    ] = three_year_mode(
        raw_predictions[
            previous_year
        ],
        raw_predictions[
            year
        ],
        raw_predictions[
            next_year
        ],
        common_valid,
    )


    probability_smoothed = (
        three_year_probability_sum(
            probability_grids[
                previous_year
            ],
            probability_grids[
                year
            ],
            probability_grids[
                next_year
            ],
            raw_predictions[
                year
            ],
            common_valid,
        )
    )


    methods[
        "PROB_SUM_3YR"
    ][
        year
    ] = (
        probability_smoothed
    )


    current_max_probability = np.nanmax(
        probability_grids[
            year
        ],
        axis=0,
    )


    gated_output = (
        probability_smoothed.copy()
    )


    keep_raw = (
        valid_masks[
            year
        ]
        &
        (
            current_max_probability
            >=
            CONFIDENCE_GATE
        )
    )


    gated_output[
        keep_raw
    ] = raw_predictions[
        year
    ][
        keep_raw
    ]


    methods[
        "CONFIDENCE_GATED_PROB_3YR"
    ][
        year
    ] = gated_output


# =============================================================================
# 10. CREATE 2021 HIGH-PURITY BENCHMARK
# =============================================================================

benchmark_class_grid, high_purity_mask = (
    high_purity_worldcover_benchmark(
        metadata_by_year[
            2021
        ]
    )
)


high_purity_mask &= valid_masks[
    2021
]


benchmark_values = (
    benchmark_class_grid[
        high_purity_mask
    ]
)


# =============================================================================
# 11. EVALUATE EACH METHOD
# =============================================================================

annual_records = []
change_records = []
benchmark_records = []
method_records = []


for method_name, annual_grids in (
    methods.items()
):

    print(
        f"\nEvaluating {method_name}..."
    )


    # -------------------------------------------------------------------------
    # Annual class composition
    # -------------------------------------------------------------------------

    for year in EXPECTED_YEARS:

        valid = valid_masks[
            year
        ]

        values = annual_grids[
            year
        ][
            valid
        ]


        pixel_area_m2 = (
            metadata_by_year[
                year
            ][
                "pixel_area_m2"
            ]
        )


        for class_id, class_name in (
            CLASS_DEFINITIONS.items()
        ):

            count = int(
                np.sum(
                    values
                    ==
                    class_id
                )
            )


            annual_records.append(
                {
                    "method": (
                        method_name
                    ),
                    "year": (
                        year
                    ),
                    "class_id": (
                        class_id
                    ),
                    "class_name": (
                        class_name
                    ),
                    "pixels": (
                        count
                    ),
                    "area_km2": (
                        count
                        *
                        pixel_area_m2
                        /
                        1_000_000
                    ),
                    "percent": (
                        count
                        /
                        len(
                            values
                        )
                        *
                        100
                    ),
                }
            )


    # -------------------------------------------------------------------------
    # Adjacent-year changes
    # -------------------------------------------------------------------------

    for from_year, to_year in zip(
        EXPECTED_YEARS[:-1],
        EXPECTED_YEARS[1:],
    ):

        common_valid = (
            valid_masks[
                from_year
            ]
            &
            valid_masks[
                to_year
            ]
        )


        from_values = annual_grids[
            from_year
        ][
            common_valid
        ]

        to_values = annual_grids[
            to_year
        ][
            common_valid
        ]


        changed_percent = float(
            np.mean(
                from_values
                !=
                to_values
            )
            *
            100
        )


        built_from = (
            from_values
            ==
            2
        )

        built_to = (
            to_values
            ==
            2
        )


        green_from = (
            from_values
            ==
            1
        )

        green_to = (
            to_values
            ==
            1
        )


        pixel_area_m2 = (
            metadata_by_year[
                to_year
            ][
                "pixel_area_m2"
            ]
        )


        built_net_pixels = int(
            built_to.sum()
            -
            built_from.sum()
        )


        green_net_pixels = int(
            green_to.sum()
            -
            green_from.sum()
        )


        change_records.append(
            {
                "method": (
                    method_name
                ),
                "from_year": (
                    from_year
                ),
                "to_year": (
                    to_year
                ),
                "provisional_transition": (
                    to_year
                    ==
                    2026
                ),
                "changed_percent": (
                    changed_percent
                ),
                "built_net_change_km2": (
                    built_net_pixels
                    *
                    pixel_area_m2
                    /
                    1_000_000
                ),
                "green_net_change_km2": (
                    green_net_pixels
                    *
                    pixel_area_m2
                    /
                    1_000_000
                ),
            }
        )


    # -------------------------------------------------------------------------
    # 2021 high-purity benchmark
    # -------------------------------------------------------------------------

    method_2021 = annual_grids[
        2021
    ][
        high_purity_mask
    ]


    agreement_percent = float(
        np.mean(
            method_2021
            ==
            benchmark_values
        )
        *
        100
    )


    raw_2021 = raw_predictions[
        2021
    ][
        high_purity_mask
    ]


    changed_from_raw_2021 = float(
        np.mean(
            method_2021
            !=
            raw_2021
        )
        *
        100
    )


    benchmark_records.append(
        {
            "method": (
                method_name
            ),
            "high_purity_benchmark_pixels": (
                len(
                    benchmark_values
                )
            ),
            "high_purity_2021_agreement_percent": (
                agreement_percent
            ),
            "percent_2021_high_purity_pixels_changed_from_raw": (
                changed_from_raw_2021
            ),
        }
    )


# =============================================================================
# 12. DATAFRAMES
# =============================================================================

annual_df = pd.DataFrame(
    annual_records
)

change_df = pd.DataFrame(
    change_records
)

benchmark_df = pd.DataFrame(
    benchmark_records
)


# =============================================================================
# 13. METHOD-LEVEL SUMMARY
# =============================================================================

method_summary_records = []


for method_name in methods:

    method_changes = (
        change_df[
            (
                change_df[
                    "method"
                ]
                ==
                method_name
            )
            &
            (
                change_df[
                    "to_year"
                ]
                <=
                2025
            )
        ]
    )


    method_annual = (
        annual_df[
            annual_df[
                "method"
            ]
            ==
            method_name
        ]
    )


    built_series = (
        method_annual[
            method_annual[
                "class_id"
            ]
            ==
            2
        ]
        .sort_values(
            "year"
        )
    )


    green_series = (
        method_annual[
            method_annual[
                "class_id"
            ]
            ==
            1
        ]
        .sort_values(
            "year"
        )
    )


    benchmark_row = (
        benchmark_df[
            benchmark_df[
                "method"
            ]
            ==
            method_name
        ].iloc[
            0
        ]
    )


    built_abs_change = (
        built_series[
            "percent"
        ]
        .diff()
        .abs()
        .dropna()
    )


    green_abs_change = (
        green_series[
            "percent"
        ]
        .diff()
        .abs()
        .dropna()
    )


    raw_change_counts = []


    if method_name != "RAW":

        for year in range(
            SMOOTH_FIRST_YEAR,
            SMOOTH_LAST_YEAR
            +
            1,
        ):

            valid = valid_masks[
                year
            ]

            raw_change_counts.append(
                float(
                    np.mean(
                        methods[
                            method_name
                        ][
                            year
                        ][
                            valid
                        ]
                        !=
                        raw_predictions[
                            year
                        ][
                            valid
                        ]
                    )
                    *
                    100
                )
            )


        mean_pixels_changed_from_raw = float(
            np.mean(
                raw_change_counts
            )
        )

    else:

        mean_pixels_changed_from_raw = (
            0.0
        )


    method_summary_records.append(
        {
            "method": (
                method_name
            ),
            "mean_adjacent_changed_percent_2010_2025": (
                float(
                    method_changes[
                        "changed_percent"
                    ].mean()
                )
            ),
            "maximum_adjacent_changed_percent_2010_2025": (
                float(
                    method_changes[
                        "changed_percent"
                    ].max()
                )
            ),
            "mean_absolute_built_percent_change_pp": (
                float(
                    built_abs_change.iloc[
                        :-1
                    ].mean()
                    if len(
                        built_abs_change
                    )
                    >
                    1
                    else
                    built_abs_change.mean()
                )
            ),
            "mean_absolute_green_percent_change_pp": (
                float(
                    green_abs_change.iloc[
                        :-1
                    ].mean()
                    if len(
                        green_abs_change
                    )
                    >
                    1
                    else
                    green_abs_change.mean()
                )
            ),
            "mean_percent_pixels_changed_from_raw_2011_2024": (
                mean_pixels_changed_from_raw
            ),
            "high_purity_2021_benchmark_agreement_percent": (
                float(
                    benchmark_row[
                        "high_purity_2021_agreement_percent"
                    ]
                )
            ),
            "high_purity_2021_changed_from_raw_percent": (
                float(
                    benchmark_row[
                        "percent_2021_high_purity_pixels_changed_from_raw"
                    ]
                )
            ),
        }
    )


method_summary_df = pd.DataFrame(
    method_summary_records
)


# =============================================================================
# 14. ADD BENCHMARK CHANGE FROM RAW
# =============================================================================

raw_benchmark_agreement = float(
    method_summary_df.loc[
        method_summary_df[
            "method"
        ]
        ==
        "RAW",
        "high_purity_2021_benchmark_agreement_percent",
    ].iloc[
        0
    ]
)


method_summary_df[
    "benchmark_agreement_change_vs_raw_pp"
] = (
    method_summary_df[
        "high_purity_2021_benchmark_agreement_percent"
    ]
    -
    raw_benchmark_agreement
)


raw_mean_change = float(
    method_summary_df.loc[
        method_summary_df[
            "method"
        ]
        ==
        "RAW",
        "mean_adjacent_changed_percent_2010_2025",
    ].iloc[
        0
    ]
)


method_summary_df[
    "mean_adjacent_change_reduction_vs_raw_pp"
] = (
    raw_mean_change
    -
    method_summary_df[
        "mean_adjacent_changed_percent_2010_2025"
    ]
)


# =============================================================================
# 15. SAVE OUTPUTS
# =============================================================================

method_summary_df.to_csv(
    METHOD_SUMMARY_CSV,
    index=False,
)


annual_df.to_csv(
    ANNUAL_CLASS_CSV,
    index=False,
)


change_df.to_csv(
    ADJACENT_CHANGE_CSV,
    index=False,
)


benchmark_df.to_csv(
    BENCHMARK_CSV,
    index=False,
)


# =============================================================================
# 16. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("TEMPORAL STABILIZATION METHOD COMPARISON")
print("=" * 100)


print(
    method_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("2021 HIGH-PURITY BENCHMARK CHECK")
print("=" * 100)


print(
    benchmark_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 07E SUMMARY")
print("=" * 100)


print(
    f"Methods compared                  : "
    f"{len(methods)}"
)

print(
    f"Temporal smoothing years          : "
    f"{SMOOTH_FIRST_YEAR}-{SMOOTH_LAST_YEAR}"
)

print(
    f"2010 left raw                     : "
    f"True"
)

print(
    f"2025 left raw                     : "
    f"True"
)

print(
    f"2026 left raw and provisional     : "
    f"True"
)

print(
    f"Confidence-gate threshold         : "
    f"{CONFIDENCE_GATE:.2f}"
)

print(
    f"Final stabilization method chosen : "
    f"False"
)

print(
    f"Final rasters exported            : "
    f"False"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(METHOD_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(ANNUAL_CLASS_CSV)}"
)

print(
    f" - {relative_path(ADJACENT_CHANGE_CSV)}"
)

print(
    f" - {relative_path(BENCHMARK_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. This is a sensitivity analysis; no stabilization method is selected automatically."
)

print(
    "2. Lower temporal flicker is desirable, but not if benchmark consistency is seriously degraded."
)

print(
    "3. 2025 is not smoothed using provisional 2026 information."
)

print(
    "4. 2026 remains a separate provisional raw classification."
)

print(
    "5. Final methodology will be selected only after reviewing these comparative results."
)

print("=" * 100)