"""
STEP 07A - TEMPORAL LANDSAT FEATURE-SPACE AUDIT

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Audit all processed annual Landsat composites from 2010-2026.
2. Verify that raster grid, CRS, resolution and band structure remain consistent.
3. Recalculate the same 10 predictors used by the frozen 2021 LULC model.
4. Compare annual feature distributions against 2021.
5. Measure how much annual imagery falls outside the 2021 training feature ranges.
6. Identify possible temporal/sensor feature shifts before historical classification.

IMPORTANT:
- This step DOES NOT classify historical LULC.
- This step DOES NOT retrain the 2021 Random Forest.
- Distribution differences may represent sensor effects, real land-cover change,
  seasonal effects, or combinations of these.
- A shift flag is diagnostic evidence, not automatic proof that a year is unusable.
"""

from pathlib import Path
import re

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

MODEL_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lulc"
    / "model_data"
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

TRAINING_CSV = (
    MODEL_DATA_DIR
    / "lulc_2021_training_features.csv"
)

REFERENCE_RASTER = (
    LANDSAT_DIR
    / "Kaduwela_Landsat_2021.tif"
)


# =============================================================================
# 3. OUTPUTS
# =============================================================================

GRID_AUDIT_CSV = (
    REPORT_DIR
    / "07a_temporal_raster_grid_audit.csv"
)

FEATURE_STATS_CSV = (
    REPORT_DIR
    / "07a_temporal_feature_statistics.csv"
)

FEATURE_SHIFT_CSV = (
    REPORT_DIR
    / "07a_temporal_feature_shift_vs_2021.csv"
)

TRAINING_RANGE_CSV = (
    REPORT_DIR
    / "07a_training_feature_reference_ranges.csv"
)

YEAR_SUMMARY_CSV = (
    REPORT_DIR
    / "07a_temporal_transfer_year_summary.csv"
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


MODEL_FEATURES = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "ndvi",
    "ndbi",
    "mndwi",
    "bsi",
]


REQUIRED_BANDS = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
]


# Diagnostic thresholds only.
# These are NOT automatic acceptance/rejection thresholds.

MEDIAN_SHIFT_FLAG_IQR = 1.0

OUTSIDE_RANGE_INCREASE_FLAG_PERCENT = 10.0


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
            f"Could not determine year from filename: "
            f"{path.name}"
        )

    return int(
        match.group(1)
    )


def sensor_era(year: int) -> str:
    """
    Return the sensor configuration used by the established
    annual-composite workflow.
    """

    if year <= 2011:

        return "Landsat 5 TM"

    if year == 2012:

        return "Landsat 5 TM + Landsat 7 ETM+"

    if 2013 <= year <= 2020:

        return "Landsat 8 OLI"

    return "Landsat 8 OLI + Landsat 9 OLI-2"


def normalized_band_name(description):
    """Standardize band descriptions."""

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
    """Determine six-band order."""

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

        return (
            lookup,
            "Raster band descriptions",
        )


    if band_count == 6:

        return (
            {
                "blue": 1,
                "green": 2,
                "red": 3,
                "nir": 4,
                "swir1": 5,
                "swir2": 6,
            },
            "Established six-band export order fallback",
        )


    raise ValueError(
        "Unable to identify Landsat bands."
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
        numerator.shape,
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


def load_feature_arrays(
    raster_path: Path,
):
    """
    Load six Landsat bands and calculate the same four indices
    used by the frozen 2021 RF model.
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

        band_lookup, mapping_method = (
            get_band_lookup(
                descriptions,
                src.count,
            )
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


        common_valid_mask = (
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


        band_values = {}


        for band_name in REQUIRED_BANDS:

            band_number = (
                band_lookup[
                    band_name
                ]
            )

            band_values[
                band_name
            ] = (
                raw_data[
                    band_number - 1
                ][
                    common_valid_mask
                ]
            )


        blue = band_values[
            "blue"
        ]

        green = band_values[
            "green"
        ]

        red = band_values[
            "red"
        ]

        nir = band_values[
            "nir"
        ]

        swir1 = band_values[
            "swir1"
        ]

        swir2 = band_values[
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
            bsi_numerator.shape,
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


        metadata = {
            "crs": str(
                src.crs
            ),
            "width": (
                src.width
            ),
            "height": (
                src.height
            ),
            "count": (
                src.count
            ),
            "transform": (
                src.transform
            ),
            "descriptions": (
                descriptions
            ),
            "valid_pixels": int(
                common_valid_mask.sum()
            ),
            "band_mapping_method": (
                mapping_method
            ),
        }


    return (
        feature_arrays,
        metadata,
    )


def numeric_statistics(values):
    """Return robust distribution statistics."""

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]


    if len(
        values
    ) == 0:

        raise ValueError(
            "Feature contains no valid values."
        )


    return {
        "count": int(
            len(
                values
            )
        ),
        "minimum": float(
            np.min(
                values
            )
        ),
        "p05": float(
            np.percentile(
                values,
                5,
            )
        ),
        "p25": float(
            np.percentile(
                values,
                25,
            )
        ),
        "median": float(
            np.percentile(
                values,
                50,
            )
        ),
        "p75": float(
            np.percentile(
                values,
                75,
            )
        ),
        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),
        "maximum": float(
            np.max(
                values
            )
        ),
        "mean": float(
            np.mean(
                values
            )
        ),
        "std": float(
            np.std(
                values
            )
        ),
    }


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 07A - TEMPORAL LANDSAT FEATURE-SPACE AUDIT")
print("=" * 100)


if not TRAINING_CSV.exists():

    raise FileNotFoundError(
        f"Training feature dataset not found:\n"
        f"{TRAINING_CSV}"
    )


if not REFERENCE_RASTER.exists():

    raise FileNotFoundError(
        f"2021 reference raster not found:\n"
        f"{REFERENCE_RASTER}"
    )


# =============================================================================
# 7. FIND ALL ANNUAL LANDSAT RASTERS
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
        "Expected exactly annual Landsat rasters "
        "from 2010 through 2026."
    )


# =============================================================================
# 8. LOAD 2021 TRAINING FEATURE RANGES
# =============================================================================

training_df = pd.read_csv(
    TRAINING_CSV
)


missing_training_features = (
    set(
        MODEL_FEATURES
    )
    -
    set(
        training_df.columns
    )
)


if missing_training_features:

    raise ValueError(
        "Training dataset is missing model features: "
        f"{sorted(missing_training_features)}"
    )


training_range_records = []

training_ranges = {}


for feature in MODEL_FEATURES:

    values = pd.to_numeric(
        training_df[
            feature
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    stats = numeric_statistics(
        values
    )


    training_ranges[
        feature
    ] = {
        "minimum": (
            stats[
                "minimum"
            ]
        ),
        "maximum": (
            stats[
                "maximum"
            ]
        ),
    }


    training_range_records.append(
        {
            "feature": (
                feature
            ),
            **stats,
        }
    )


training_range_df = pd.DataFrame(
    training_range_records
)


# =============================================================================
# 9. LOAD 2021 FULL-AREA FEATURE BASELINE
# =============================================================================

reference_features, reference_metadata = (
    load_feature_arrays(
        REFERENCE_RASTER
    )
)


reference_feature_stats = {}


for feature in MODEL_FEATURES:

    reference_feature_stats[
        feature
    ] = numeric_statistics(
        reference_features[
            feature
        ]
    )


reference_transform = (
    reference_metadata[
        "transform"
    ]
)


# =============================================================================
# 10. AUDIT ALL YEARS
# =============================================================================

grid_records = []

feature_records = []


for raster_path in raster_files:

    year = extract_year(
        raster_path
    )


    print(
        f"Processing {year}..."
    )


    feature_arrays, metadata = (
        load_feature_arrays(
            raster_path
        )
    )


    grid_match = all(
        [
            metadata[
                "crs"
            ]
            ==
            reference_metadata[
                "crs"
            ],
            metadata[
                "width"
            ]
            ==
            reference_metadata[
                "width"
            ],
            metadata[
                "height"
            ]
            ==
            reference_metadata[
                "height"
            ],
            metadata[
                "count"
            ]
            ==
            reference_metadata[
                "count"
            ],
            metadata[
                "transform"
            ].almost_equals(
                reference_transform
            ),
        ]
    )


    band_description_match = (
        metadata[
            "descriptions"
        ]
        ==
        reference_metadata[
            "descriptions"
        ]
    )


    valid_pixel_difference = (
        metadata[
            "valid_pixels"
        ]
        -
        reference_metadata[
            "valid_pixels"
        ]
    )


    grid_records.append(
        {
            "year": (
                year
            ),
            "sensor_era": (
                sensor_era(
                    year
                )
            ),
            "raster": (
                relative_path(
                    raster_path
                )
            ),
            "crs": (
                metadata[
                    "crs"
                ]
            ),
            "width": (
                metadata[
                    "width"
                ]
            ),
            "height": (
                metadata[
                    "height"
                ]
            ),
            "band_count": (
                metadata[
                    "count"
                ]
            ),
            "valid_pixels": (
                metadata[
                    "valid_pixels"
                ]
            ),
            "valid_pixel_difference_vs_2021": (
                valid_pixel_difference
            ),
            "band_mapping_method": (
                metadata[
                    "band_mapping_method"
                ]
            ),
            "grid_match_2021": (
                grid_match
            ),
            "band_descriptions_match_2021": (
                band_description_match
            ),
        }
    )


    for feature in MODEL_FEATURES:

        values = np.asarray(
            feature_arrays[
                feature
            ],
            dtype=float,
        )


        values = values[
            np.isfinite(
                values
            )
        ]


        stats = numeric_statistics(
            values
        )


        training_minimum = (
            training_ranges[
                feature
            ][
                "minimum"
            ]
        )

        training_maximum = (
            training_ranges[
                feature
            ][
                "maximum"
            ]
        )


        below_training = int(
            np.sum(
                values
                <
                training_minimum
            )
        )


        above_training = int(
            np.sum(
                values
                >
                training_maximum
            )
        )


        outside_training = (
            below_training
            +
            above_training
        )


        outside_training_percent = float(
            outside_training
            /
            len(
                values
            )
            *
            100
        )


        reference_stats = (
            reference_feature_stats[
                feature
            ]
        )


        reference_iqr = (
            reference_stats[
                "p75"
            ]
            -
            reference_stats[
                "p25"
            ]
        )


        if abs(
            reference_iqr
        ) < 1e-12:

            median_shift_iqr = (
                np.nan
            )

        else:

            median_shift_iqr = float(
                abs(
                    stats[
                        "median"
                    ]
                    -
                    reference_stats[
                        "median"
                    ]
                )
                /
                abs(
                    reference_iqr
                )
            )


        reference_p05_p95_width = (
            reference_stats[
                "p95"
            ]
            -
            reference_stats[
                "p05"
            ]
        )


        overlap_width = max(
            0.0,
            min(
                stats[
                    "p95"
                ],
                reference_stats[
                    "p95"
                ],
            )
            -
            max(
                stats[
                    "p05"
                ],
                reference_stats[
                    "p05"
                ],
            ),
        )


        if (
            reference_p05_p95_width
            <= 1e-12
        ):

            p05_p95_overlap_fraction = (
                np.nan
            )

        else:

            p05_p95_overlap_fraction = float(
                min(
                    1.0,
                    overlap_width
                    /
                    reference_p05_p95_width,
                )
            )


        feature_records.append(
            {
                "year": (
                    year
                ),
                "sensor_era": (
                    sensor_era(
                        year
                    )
                ),
                "feature": (
                    feature
                ),
                **stats,
                "training_minimum": (
                    training_minimum
                ),
                "training_maximum": (
                    training_maximum
                ),
                "below_training_range_pixels": (
                    below_training
                ),
                "above_training_range_pixels": (
                    above_training
                ),
                "outside_training_range_pixels": (
                    outside_training
                ),
                "outside_training_range_percent": (
                    outside_training_percent
                ),
                "reference_2021_median": (
                    reference_stats[
                        "median"
                    ]
                ),
                "reference_2021_iqr": (
                    reference_iqr
                ),
                "absolute_median_shift_in_2021_iqr": (
                    median_shift_iqr
                ),
                "p05_p95_overlap_fraction_vs_2021": (
                    p05_p95_overlap_fraction
                ),
            }
        )


# =============================================================================
# 11. CREATE DATAFRAMES
# =============================================================================

grid_df = pd.DataFrame(
    grid_records
)


feature_df = pd.DataFrame(
    feature_records
)


# =============================================================================
# 12. ADD 2021 BASELINE OUTSIDE-RANGE COMPARISON
# =============================================================================

baseline_outside = (
    feature_df[
        feature_df[
            "year"
        ]
        == 2021
    ][
        [
            "feature",
            "outside_training_range_percent",
        ]
    ]
    .rename(
        columns={
            "outside_training_range_percent":
            "outside_training_range_percent_2021"
        }
    )
)


feature_shift_df = (
    feature_df.merge(
        baseline_outside,
        on="feature",
        how="left",
    )
)


feature_shift_df[
    "outside_training_range_change_vs_2021_percent_points"
] = (
    feature_shift_df[
        "outside_training_range_percent"
    ]
    -
    feature_shift_df[
        "outside_training_range_percent_2021"
    ]
)


feature_shift_df[
    "median_shift_flag"
] = (
    feature_shift_df[
        "absolute_median_shift_in_2021_iqr"
    ]
    >
    MEDIAN_SHIFT_FLAG_IQR
)


feature_shift_df[
    "outside_range_increase_flag"
] = (
    feature_shift_df[
        "outside_training_range_change_vs_2021_percent_points"
    ]
    >
    OUTSIDE_RANGE_INCREASE_FLAG_PERCENT
)


# =============================================================================
# 13. YEAR-LEVEL SUMMARY
# =============================================================================

year_summary_records = []


for year in EXPECTED_YEARS:

    year_features = (
        feature_shift_df[
            feature_shift_df[
                "year"
            ]
            == year
        ]
    )


    year_grid = (
        grid_df[
            grid_df[
                "year"
            ]
            == year
        ].iloc[
            0
        ]
    )


    year_summary_records.append(
        {
            "year": (
                year
            ),
            "sensor_era": (
                sensor_era(
                    year
                )
            ),
            "valid_pixels": (
                int(
                    year_grid[
                        "valid_pixels"
                    ]
                )
            ),
            "grid_match_2021": (
                bool(
                    year_grid[
                        "grid_match_2021"
                    ]
                )
            ),
            "band_descriptions_match_2021": (
                bool(
                    year_grid[
                        "band_descriptions_match_2021"
                    ]
                )
            ),
            "mean_outside_training_range_percent": (
                float(
                    year_features[
                        "outside_training_range_percent"
                    ].mean()
                )
            ),
            "max_outside_training_range_percent": (
                float(
                    year_features[
                        "outside_training_range_percent"
                    ].max()
                )
            ),
            "mean_outside_range_change_vs_2021_percent_points": (
                float(
                    year_features[
                        "outside_training_range_change_vs_2021_percent_points"
                    ].mean()
                )
            ),
            "max_outside_range_change_vs_2021_percent_points": (
                float(
                    year_features[
                        "outside_training_range_change_vs_2021_percent_points"
                    ].max()
                )
            ),
            "mean_absolute_median_shift_in_2021_iqr": (
                float(
                    year_features[
                        "absolute_median_shift_in_2021_iqr"
                    ].mean()
                )
            ),
            "max_absolute_median_shift_in_2021_iqr": (
                float(
                    year_features[
                        "absolute_median_shift_in_2021_iqr"
                    ].max()
                )
            ),
            "features_with_median_shift_flag": (
                int(
                    year_features[
                        "median_shift_flag"
                    ].sum()
                )
            ),
            "features_with_outside_range_increase_flag": (
                int(
                    year_features[
                        "outside_range_increase_flag"
                    ].sum()
                )
            ),
        }
    )


year_summary_df = pd.DataFrame(
    year_summary_records
)


# =============================================================================
# 14. FINAL STRUCTURAL QA
# =============================================================================

all_years_present = (
    sorted(
        grid_df[
            "year"
        ].tolist()
    )
    ==
    EXPECTED_YEARS
)


all_grids_match = bool(
    grid_df[
        "grid_match_2021"
    ].all()
)


all_band_descriptions_match = bool(
    grid_df[
        "band_descriptions_match_2021"
    ].all()
)


overall_structural_status = (
    "PASS"
    if all(
        [
            all_years_present,
            all_grids_match,
            all_band_descriptions_match,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 15. SAVE OUTPUTS
# =============================================================================

grid_df.to_csv(
    GRID_AUDIT_CSV,
    index=False,
)


feature_df.to_csv(
    FEATURE_STATS_CSV,
    index=False,
)


feature_shift_df.to_csv(
    FEATURE_SHIFT_CSV,
    index=False,
)


training_range_df.to_csv(
    TRAINING_RANGE_CSV,
    index=False,
)


year_summary_df.to_csv(
    YEAR_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 16. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("ANNUAL RASTER STRUCTURAL AUDIT")
print("=" * 100)


print(
    grid_df[
        [
            "year",
            "sensor_era",
            "valid_pixels",
            "grid_match_2021",
            "band_descriptions_match_2021",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("TEMPORAL FEATURE-SPACE SUMMARY")
print("=" * 100)


print(
    year_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 07A SUMMARY")
print("=" * 100)


print(
    f"Annual Landsat rasters       : "
    f"{len(raster_files)}"
)

print(
    f"Years 2010-2026 complete     : "
    f"{all_years_present}"
)

print(
    f"All grids match 2021         : "
    f"{all_grids_match}"
)

print(
    f"All band descriptions match  : "
    f"{all_band_descriptions_match}"
)

print(
    f"Structural audit status      : "
    f"{overall_structural_status}"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(GRID_AUDIT_CSV)}"
)

print(
    f" - {relative_path(FEATURE_STATS_CSV)}"
)

print(
    f" - {relative_path(FEATURE_SHIFT_CSV)}"
)

print(
    f" - {relative_path(TRAINING_RANGE_CSV)}"
)

print(
    f" - {relative_path(YEAR_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. No historical LULC classification was performed."
)

print(
    "2. No Random Forest retraining was performed."
)

print(
    "3. Feature shifts may contain both real land-cover change and sensor/season effects."
)

print(
    "4. A shift flag does not automatically mean that a year is unusable."
)

print(
    "5. These diagnostics will determine the temporal-transfer strategy used before 2010-2026 classification."
)

print("=" * 100)