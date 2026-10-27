"""
STEP 06A - 2021 LULC MODEL FEATURE EXTRACTION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Read the final curated 2021 LULC reference pool from Step 05G.
2. Extract the exact processed Landsat 2021 pixel values.
3. Calculate spectral indices.
4. Build a clean machine-learning feature table.
5. Check missing values, duplicate pixels and class distribution.
6. Cross-check newly extracted values against earlier stored values when available.

LULC Classes:
1 - Green / Vegetation
2 - Built-up
3 - Open / Bare
4 - Water

Model Features:
- Blue
- Green
- Red
- NIR
- SWIR1
- SWIR2
- NDVI
- NDBI
- MNDWI
- BSI

IMPORTANT:
- No model is trained in this step.
- Coordinates are retained only for spatial validation.
- Coordinates must NOT be used as LULC predictor variables.
- Class labels come from the curated Step 05G reference workflow.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

# Current file:
# Urban_Green_Research/src/01_lulc_green_transition/06a_...
#
# parents[2] = Urban_Green_Research
PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lulc"
    / "reference_samples"
)

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

MODEL_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. INPUTS
# =============================================================================

CURATED_REFERENCE_GPKG = (
    REFERENCE_DIR
    / "lulc_curated_training_reference_2021.gpkg"
)

LANDSAT_2021 = (
    LANDSAT_DIR
    / "Kaduwela_Landsat_2021.tif"
)


# =============================================================================
# 3. OUTPUTS
# =============================================================================

MODEL_FEATURE_CSV = (
    MODEL_DATA_DIR
    / "lulc_2021_model_features.csv"
)

CLASS_SUMMARY_CSV = (
    REPORT_DIR
    / "06a_2021_lulc_feature_class_summary.csv"
)

FEATURE_STATS_CSV = (
    REPORT_DIR
    / "06a_2021_lulc_feature_statistics.csv"
)

FEATURE_CONSISTENCY_CSV = (
    REPORT_DIR
    / "06a_2021_lulc_feature_consistency.csv"
)

PROCESS_SUMMARY_CSV = (
    REPORT_DIR
    / "06a_2021_lulc_feature_extraction_summary.csv"
)


# =============================================================================
# 4. CONSTANTS
# =============================================================================

ANALYSIS_CRS = "EPSG:32644"

EXPECTED_REFERENCE_COUNT = 397


CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}


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


# =============================================================================
# 5. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """
    Return project-relative path when possible.
    """

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def safe_normalized_difference(
    value_a: float,
    value_b: float,
) -> float:
    """
    Calculate (A - B) / (A + B) safely.
    """

    denominator = (
        value_a
        +
        value_b
    )

    if (
        not np.isfinite(
            denominator
        )
        or
        abs(
            denominator
        )
        < 1e-12
    ):
        return np.nan

    return float(
        (
            value_a
            -
            value_b
        )
        /
        denominator
    )


def normalized_band_name(description):
    """
    Convert raster band descriptions to standard names.
    """

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


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 06A - 2021 LULC MODEL FEATURE EXTRACTION")
print("=" * 100)


for required_file in [
    CURATED_REFERENCE_GPKG,
    LANDSAT_2021,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required input not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 7. LOAD CURATED REFERENCE DATA
# =============================================================================

reference_gdf = gpd.read_file(
    CURATED_REFERENCE_GPKG
)


print(
    f"\nCurated reference : "
    f"{relative_path(CURATED_REFERENCE_GPKG)}"
)

print(
    f"Reference samples : "
    f"{len(reference_gdf)}"
)

print(
    f"CRS               : "
    f"{reference_gdf.crs}"
)


# =============================================================================
# 8. VALIDATE REFERENCE DATA
# =============================================================================

if str(
    reference_gdf.crs
) != ANALYSIS_CRS:

    raise ValueError(
        f"Unexpected reference CRS: "
        f"{reference_gdf.crs}. "
        f"Expected {ANALYSIS_CRS}."
    )


if len(
    reference_gdf
) != EXPECTED_REFERENCE_COUNT:

    raise ValueError(
        f"Expected {EXPECTED_REFERENCE_COUNT} curated samples, "
        f"but found {len(reference_gdf)}."
    )


required_reference_columns = {
    "sample_id",
    "class_id",
    "class_name",
}


missing_columns = (
    required_reference_columns
    -
    set(
        reference_gdf.columns
    )
)


if missing_columns:

    raise ValueError(
        f"Reference dataset is missing required columns: "
        f"{sorted(missing_columns)}"
    )


if reference_gdf[
    "sample_id"
].duplicated().any():

    raise ValueError(
        "Duplicate sample IDs exist "
        "in the curated reference dataset."
    )


if not reference_gdf.geometry.geom_type.eq(
    "Point"
).all():

    raise ValueError(
        "All curated reference geometries "
        "must be Point features."
    )


reference_gdf[
    "class_id"
] = pd.to_numeric(
    reference_gdf[
        "class_id"
    ],
    errors="raise",
).astype(int)


invalid_classes = (
    ~reference_gdf[
        "class_id"
    ].isin(
        CLASS_DEFINITIONS.keys()
    )
)


if invalid_classes.any():

    raise ValueError(
        "Unexpected LULC class IDs exist "
        "in the curated reference dataset."
    )


# =============================================================================
# 9. REFERENCE CLASS COUNTS
# =============================================================================

print("\n" + "=" * 100)
print("CURATED REFERENCE CLASS COUNTS")
print("=" * 100)


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    count = int(
        (
            reference_gdf[
                "class_id"
            ]
            == class_id
        ).sum()
    )

    print(
        f"{class_id} - "
        f"{class_name:<20}: "
        f"{count}"
    )


# =============================================================================
# 10. OPEN LANDSAT RASTER
# =============================================================================

with rasterio.open(
    LANDSAT_2021
) as src:

    landsat_crs = src.crs

    landsat_width = src.width

    landsat_height = src.height

    landsat_transform = src.transform

    landsat_count = src.count

    landsat_descriptions = list(
        src.descriptions
    )

    landsat_data = src.read(
        masked=True
    )


print("\n" + "=" * 100)
print("LANDSAT 2021 RASTER")
print("=" * 100)


print(
    f"Raster            : "
    f"{relative_path(LANDSAT_2021)}"
)

print(
    f"CRS               : "
    f"{landsat_crs}"
)

print(
    f"Raster size       : "
    f"{landsat_width} x {landsat_height}"
)

print(
    f"Band count        : "
    f"{landsat_count}"
)

print(
    f"Band descriptions : "
    f"{landsat_descriptions}"
)

print(
    f"Pixel size        : "
    f"{abs(landsat_transform.a):.3f} m x "
    f"{abs(landsat_transform.e):.3f} m"
)


if str(
    landsat_crs
) != ANALYSIS_CRS:

    raise ValueError(
        f"Unexpected Landsat CRS: "
        f"{landsat_crs}"
    )


if landsat_count != 6:

    raise ValueError(
        f"Expected 6 Landsat bands, "
        f"but found {landsat_count}."
    )


# =============================================================================
# 11. DETERMINE BAND ORDER
# =============================================================================

band_lookup = {}


for band_number, description in enumerate(
    landsat_descriptions,
    start=1,
):

    normalized_name = normalized_band_name(
        description
    )

    if normalized_name is not None:

        band_lookup[
            normalized_name
        ] = band_number


required_band_names = {
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
}


if required_band_names.issubset(
    band_lookup.keys()
):

    band_mapping_method = (
        "Raster band descriptions"
    )

else:

    # Established export order from the Landsat preprocessing workflow.
    band_lookup = {
        "blue": 1,
        "green": 2,
        "red": 3,
        "nir": 4,
        "swir1": 5,
        "swir2": 6,
    }

    band_mapping_method = (
        "Established six-band export order fallback"
    )


print(
    f"\nBand mapping method : "
    f"{band_mapping_method}"
)

print(
    f"Band mapping        : "
    f"{band_lookup}"
)


# =============================================================================
# 12. EXTRACT LANDSAT FEATURES
# =============================================================================

print("\n" + "=" * 100)
print("EXTRACTING FEATURES FOR 397 CURATED REFERENCE PIXELS")
print("=" * 100)


feature_records = []


for record_number, (_, row) in enumerate(
    reference_gdf.iterrows(),
    start=1,
):

    sample_id = str(
        row[
            "sample_id"
        ]
    )


    x = float(
        row.geometry.x
    )

    y = float(
        row.geometry.y
    )


    # -------------------------------------------------------------------------
    # Use stored raster row/column if available.
    # Otherwise calculate from geometry.
    # -------------------------------------------------------------------------

    if (
        "landsat_row"
        in reference_gdf.columns
        and
        "landsat_col"
        in reference_gdf.columns
        and
        pd.notna(
            row.get(
                "landsat_row"
            )
        )
        and
        pd.notna(
            row.get(
                "landsat_col"
            )
        )
    ):

        raster_row = int(
            row[
                "landsat_row"
            ]
        )

        raster_col = int(
            row[
                "landsat_col"
            ]
        )

    else:

        raster_row, raster_col = (
            rasterio.transform.rowcol(
                landsat_transform,
                x,
                y,
            )
        )

        raster_row = int(
            raster_row
        )

        raster_col = int(
            raster_col
        )


    if not (
        0
        <= raster_row
        < landsat_height
        and
        0
        <= raster_col
        < landsat_width
    ):

        raise ValueError(
            f"{sample_id} falls outside "
            f"the Landsat raster."
        )


    # -------------------------------------------------------------------------
    # Extract six bands
    # -------------------------------------------------------------------------

    band_values = {}


    for band_name, band_number in (
        band_lookup.items()
    ):

        raster_value = (
            landsat_data[
                band_number - 1,
                raster_row,
                raster_col,
            ]
        )


        if np.ma.is_masked(
            raster_value
        ):

            value = np.nan

        else:

            value = float(
                raster_value
            )


        band_values[
            band_name
        ] = value


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


    # -------------------------------------------------------------------------
    # Spectral indices
    # -------------------------------------------------------------------------

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


    if (
        not np.isfinite(
            bsi_denominator
        )
        or
        abs(
            bsi_denominator
        )
        < 1e-12
    ):

        bsi = np.nan

    else:

        bsi = float(
            bsi_numerator
            /
            bsi_denominator
        )


    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    manually_audited = (
        bool(
            row[
                "manually_audited"
            ]
        )
        if
        "manually_audited"
        in reference_gdf.columns
        else
        False
    )


    manual_audit_status = (
        str(
            row[
                "manual_audit_status"
            ]
        )
        if
        "manual_audit_status"
        in reference_gdf.columns
        else
        "NOT_AVAILABLE"
    )


    reference_source = (
        str(
            row[
                "reference_source"
            ]
        )
        if
        "reference_source"
        in reference_gdf.columns
        else
        "CURATED_REFERENCE"
    )


    pixel_id = (
        f"{raster_row}_{raster_col}"
    )


    feature_records.append(
        {
            "sample_id": (
                sample_id
            ),
            "class_id": (
                int(
                    row[
                        "class_id"
                    ]
                )
            ),
            "class_name": (
                row[
                    "class_name"
                ]
            ),
            "easting": (
                x
            ),
            "northing": (
                y
            ),
            "landsat_row": (
                raster_row
            ),
            "landsat_col": (
                raster_col
            ),
            "landsat_pixel_id": (
                pixel_id
            ),
            "manually_audited": (
                manually_audited
            ),
            "manual_audit_status": (
                manual_audit_status
            ),
            "reference_source": (
                reference_source
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
        }
    )


    if (
        record_number % 100
        == 0
    ):

        print(
            f"Processed "
            f"{record_number} / "
            f"{len(reference_gdf)}"
        )


feature_df = pd.DataFrame(
    feature_records
)


print(
    f"\nExtracted records : "
    f"{len(feature_df)}"
)


# =============================================================================
# 13. FEATURE QUALITY CHECKS
# =============================================================================

print("\n" + "=" * 100)
print("FEATURE QUALITY CHECK")
print("=" * 100)


missing_feature_counts = (
    feature_df[
        MODEL_FEATURES
    ]
    .isna()
    .sum()
)


infinite_feature_counts = pd.Series(
    {
        feature: int(
            np.isinf(
                feature_df[
                    feature
                ].to_numpy(
                    dtype=float
                )
            ).sum()
        )

        for feature in MODEL_FEATURES
    }
)


duplicate_sample_ids = int(
    feature_df[
        "sample_id"
    ]
    .duplicated()
    .sum()
)


duplicate_pixel_ids = int(
    feature_df[
        "landsat_pixel_id"
    ]
    .duplicated()
    .sum()
)


duplicate_feature_vectors = int(
    feature_df[
        MODEL_FEATURES
    ]
    .duplicated()
    .sum()
)


print(
    "\nMissing values by feature:"
)

print(
    missing_feature_counts.to_string()
)


print(
    "\nInfinite values by feature:"
)

print(
    infinite_feature_counts.to_string()
)


print(
    f"\nDuplicate sample IDs      : "
    f"{duplicate_sample_ids}"
)

print(
    f"Duplicate Landsat pixels  : "
    f"{duplicate_pixel_ids}"
)

print(
    f"Duplicate feature vectors : "
    f"{duplicate_feature_vectors}"
)


total_missing_features = int(
    missing_feature_counts.sum()
)


total_infinite_features = int(
    infinite_feature_counts.sum()
)


# =============================================================================
# 14. CLASS SUMMARY
# =============================================================================

class_summary_records = []


for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    class_df = (
        feature_df[
            feature_df[
                "class_id"
            ]
            == class_id
        ]
    )


    class_summary_records.append(
        {
            "class_id": (
                class_id
            ),
            "class_name": (
                class_name
            ),
            "sample_count": (
                len(
                    class_df
                )
            ),
            "percent_of_dataset": (
                float(
                    len(
                        class_df
                    )
                    /
                    len(
                        feature_df
                    )
                    *
                    100
                )
            ),
            "manually_audited_count": (
                int(
                    class_df[
                        "manually_audited"
                    ].sum()
                )
            ),
        }
    )


class_summary_df = pd.DataFrame(
    class_summary_records
)


# =============================================================================
# 15. FEATURE STATISTICS
# =============================================================================

feature_statistics_records = []


for feature_name in (
    MODEL_FEATURES
):

    values = pd.to_numeric(
        feature_df[
            feature_name
        ],
        errors="coerce",
    )


    feature_statistics_records.append(
        {
            "feature": (
                feature_name
            ),
            "count": (
                int(
                    values.notna().sum()
                )
            ),
            "minimum": (
                float(
                    values.min()
                )
            ),
            "maximum": (
                float(
                    values.max()
                )
            ),
            "mean": (
                float(
                    values.mean()
                )
            ),
            "median": (
                float(
                    values.median()
                )
            ),
            "std": (
                float(
                    values.std()
                )
            ),
        }
    )


feature_stats_df = pd.DataFrame(
    feature_statistics_records
)


# =============================================================================
# 16. CROSS-CHECK AGAINST STORED STEP 05 FEATURES
# =============================================================================

consistency_records = []


for feature_name in (
    MODEL_FEATURES
):

    if feature_name not in (
        reference_gdf.columns
    ):

        consistency_records.append(
            {
                "feature": (
                    feature_name
                ),
                "stored_feature_available": (
                    False
                ),
                "compared_records": (
                    0
                ),
                "maximum_absolute_difference": (
                    np.nan
                ),
                "mean_absolute_difference": (
                    np.nan
                ),
                "all_close": (
                    np.nan
                ),
            }
        )

        continue


    stored_values = pd.to_numeric(
        reference_gdf[
            feature_name
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


    extracted_values = (
        feature_df[
            feature_name
        ]
        .to_numpy(
            dtype=float
        )
    )


    valid_mask = (
        np.isfinite(
            stored_values
        )
        &
        np.isfinite(
            extracted_values
        )
    )


    absolute_difference = np.abs(
        stored_values[
            valid_mask
        ]
        -
        extracted_values[
            valid_mask
        ]
    )


    if len(
        absolute_difference
    ) == 0:

        maximum_difference = (
            np.nan
        )

        mean_difference = (
            np.nan
        )

        all_close = (
            np.nan
        )

    else:

        maximum_difference = float(
            np.max(
                absolute_difference
            )
        )

        mean_difference = float(
            np.mean(
                absolute_difference
            )
        )

        all_close = bool(
            np.allclose(
                stored_values[
                    valid_mask
                ],
                extracted_values[
                    valid_mask
                ],
                rtol=1e-9,
                atol=1e-9,
            )
        )


    consistency_records.append(
        {
            "feature": (
                feature_name
            ),
            "stored_feature_available": (
                True
            ),
            "compared_records": (
                int(
                    valid_mask.sum()
                )
            ),
            "maximum_absolute_difference": (
                maximum_difference
            ),
            "mean_absolute_difference": (
                mean_difference
            ),
            "all_close": (
                all_close
            ),
        }
    )


consistency_df = pd.DataFrame(
    consistency_records
)


# =============================================================================
# 17. FINAL VALIDATION
# =============================================================================

all_classes_present = (
    set(
        feature_df[
            "class_id"
        ].unique()
    )
    ==
    set(
        CLASS_DEFINITIONS.keys()
    )
)


all_features_complete = (
    total_missing_features
    == 0
    and
    total_infinite_features
    == 0
)


unique_samples = (
    duplicate_sample_ids
    == 0
)


unique_pixels = (
    duplicate_pixel_ids
    == 0
)


expected_record_count = (
    len(
        feature_df
    )
    ==
    EXPECTED_REFERENCE_COUNT
)


overall_pass = all(
    [
        expected_record_count,
        all_classes_present,
        all_features_complete,
        unique_samples,
        unique_pixels,
    ]
)


# =============================================================================
# 18. PROCESS SUMMARY
# =============================================================================

process_summary_df = pd.DataFrame(
    [
        {
            "input_curated_reference_samples": (
                len(
                    reference_gdf
                )
            ),
            "output_feature_records": (
                len(
                    feature_df
                )
            ),
            "model_feature_count": (
                len(
                    MODEL_FEATURES
                )
            ),
            "missing_feature_values": (
                total_missing_features
            ),
            "infinite_feature_values": (
                total_infinite_features
            ),
            "duplicate_sample_ids": (
                duplicate_sample_ids
            ),
            "duplicate_landsat_pixels": (
                duplicate_pixel_ids
            ),
            "duplicate_feature_vectors": (
                duplicate_feature_vectors
            ),
            "all_four_classes_present": (
                all_classes_present
            ),
            "coordinates_used_as_model_features": (
                False
            ),
            "band_mapping_method": (
                band_mapping_method
            ),
            "overall_status": (
                "PASS"
                if overall_pass
                else
                "REVIEW_REQUIRED"
            ),
        }
    ]
)


# =============================================================================
# 19. SAVE OUTPUTS
# =============================================================================

feature_df.to_csv(
    MODEL_FEATURE_CSV,
    index=False,
)


class_summary_df.to_csv(
    CLASS_SUMMARY_CSV,
    index=False,
)


feature_stats_df.to_csv(
    FEATURE_STATS_CSV,
    index=False,
)


consistency_df.to_csv(
    FEATURE_CONSISTENCY_CSV,
    index=False,
)


process_summary_df.to_csv(
    PROCESS_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 20. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("2021 LULC MODEL FEATURE TABLE - CLASS DISTRIBUTION")
print("=" * 100)


print(
    class_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("FEATURE STATISTICS")
print("=" * 100)


print(
    feature_stats_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("FEATURE EXTRACTION CONSISTENCY CHECK")
print("=" * 100)


print(
    consistency_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 06A SUMMARY")
print("=" * 100)


print(
    f"Input curated references    : "
    f"{len(reference_gdf)}"
)

print(
    f"Output feature records      : "
    f"{len(feature_df)}"
)

print(
    f"Model features              : "
    f"{len(MODEL_FEATURES)}"
)

print(
    f"Missing feature values      : "
    f"{total_missing_features}"
)

print(
    f"Infinite feature values     : "
    f"{total_infinite_features}"
)

print(
    f"Duplicate sample IDs        : "
    f"{duplicate_sample_ids}"
)

print(
    f"Duplicate Landsat pixels    : "
    f"{duplicate_pixel_ids}"
)

print(
    f"Duplicate feature vectors   : "
    f"{duplicate_feature_vectors}"
)

print(
    f"All four classes present    : "
    f"{all_classes_present}"
)

print(
    f"Coordinates used as ML "
    f"predictors                 : "
    f"False"
)

print(
    f"Overall Step 06A status     : "
    f"{'PASS' if overall_pass else 'REVIEW REQUIRED'}"
)


print("\nModel predictor columns:")

for feature_name in (
    MODEL_FEATURES
):
    print(
        f" - {feature_name}"
    )


print("\nMetadata retained for later spatial validation:")

print(
    " - sample_id"
)

print(
    " - class_id / class_name"
)

print(
    " - easting / northing"
)

print(
    " - landsat_row / landsat_col"
)

print(
    " - landsat_pixel_id"
)


print("\nGenerated outputs:")

print(
    f" - {relative_path(MODEL_FEATURE_CSV)}"
)

print(
    f" - {relative_path(CLASS_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(FEATURE_STATS_CSV)}"
)

print(
    f" - {relative_path(FEATURE_CONSISTENCY_CSV)}"
)

print(
    f" - {relative_path(PROCESS_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. No classifier was trained in Step 06A."
)

print(
    "2. Only spectral Landsat features and indices are model predictors."
)

print(
    "3. Easting and northing are retained only for spatial splitting/validation."
)

print(
    "4. The rare Open / Bare class remains limited and must be handled carefully."
)

print(
    "5. The next step will design the spatially safe 2021 model-validation split."
)

print("=" * 100)