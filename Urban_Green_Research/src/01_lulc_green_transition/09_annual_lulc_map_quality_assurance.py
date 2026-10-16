"""
STEP 09 - ANNUAL LULC MAP QUALITY ASSURANCE

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Validate all exported RAW and STABILIZED annual LULC GeoTIFFs.
2. Confirm exact 2010-2026 temporal coverage.
3. Confirm CRS, grid, transform, resolution, NoData and class values.
4. Confirm class-area statistics agree with Step 08 reports.
5. Confirm only 2011-2024 differ between RAW and STABILIZED products.
6. Confirm 2010, 2025 and 2026 remain identical to RAW.
7. Confirm metadata correctly records stabilization and provisional status.

IMPORTANT:
- This is output quality assurance only.
- No model retraining is performed.
- No classification changes are performed.
- No transition interpretation is performed.
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

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lulc"
    / "annual_maps"
    / "raw"
)

STABILIZED_DIR = (
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

STEP_08_CLASS_SUMMARY = (
    REPORT_DIR
    / "08_annual_lulc_class_area_summary.csv"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. OUTPUT REPORTS
# =============================================================================

RASTER_QA_CSV = (
    REPORT_DIR
    / "09_annual_lulc_raster_qa.csv"
)

CLASS_QA_CSV = (
    REPORT_DIR
    / "09_annual_lulc_class_statistics_qa.csv"
)

RAW_STABILIZED_COMPARISON_CSV = (
    REPORT_DIR
    / "09_raw_vs_stabilized_comparison.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "09_annual_lulc_qa_summary.csv"
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

VALID_CLASS_VALUES = {
    0,
    1,
    2,
    3,
    4,
}

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

EXPECTED_STABILIZED_YEARS = set(
    range(
        2011,
        2025,
    )
)

EXPECTED_IDENTICAL_YEARS = {
    2010,
    2025,
    2026,
}

EXPECTED_CRS = "EPSG:32644"

EXPECTED_PIXEL_SIZE = 30.0

STAT_TOLERANCE = 1e-8


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
    """Extract year from exported LULC filename."""

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


def expected_raw_path(year: int) -> Path:

    return (
        RAW_DIR
        /
        f"Kaduwela_LULC_RAW_{year}.tif"
    )


def expected_stabilized_path(year: int) -> Path:

    return (
        STABILIZED_DIR
        /
        f"Kaduwela_LULC_STABILIZED_{year}.tif"
    )


def inspect_raster(
    raster_path: Path,
    year: int,
    product_type: str,
):
    """
    Inspect one LULC GeoTIFF and return raster + class statistics.
    """

    with rasterio.open(
        raster_path
    ) as src:

        data = src.read(
            1
        )

        tags = src.tags()

        unique_values = set(
            int(
                value
            )
            for value in np.unique(
                data
            )
        )

        invalid_values = (
            unique_values
            -
            VALID_CLASS_VALUES
        )

        valid_mask = np.isin(
            data,
            CLASS_IDS,
        )

        valid_pixels = int(
            valid_mask.sum()
        )

        nodata_pixels = int(
            np.sum(
                data
                ==
                0
            )
        )

        total_pixels = int(
            data.size
        )

        pixel_width = float(
            abs(
                src.transform.a
            )
        )

        pixel_height = float(
            abs(
                src.transform.e
            )
        )

        pixel_area_m2 = float(
            pixel_width
            *
            pixel_height
        )


        raster_record = {
            "year": (
                year
            ),
            "product_type": (
                product_type
            ),
            "file": (
                relative_path(
                    raster_path
                )
            ),
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
            "band_count": (
                src.count
            ),
            "dtype": (
                src.dtypes[
                    0
                ]
            ),
            "nodata": (
                src.nodata
            ),
            "pixel_width_m": (
                pixel_width
            ),
            "pixel_height_m": (
                pixel_height
            ),
            "pixel_area_m2": (
                pixel_area_m2
            ),
            "total_pixels": (
                total_pixels
            ),
            "valid_pixels": (
                valid_pixels
            ),
            "nodata_pixels": (
                nodata_pixels
            ),
            "unique_values": (
                ",".join(
                    str(
                        value
                    )
                    for value in sorted(
                        unique_values
                    )
                )
            ),
            "invalid_class_value_count": (
                len(
                    invalid_values
                )
            ),
            "transform": (
                str(
                    src.transform
                )
            ),
            "tag_year": (
                tags.get(
                    "year"
                )
            ),
            "tag_product_type": (
                tags.get(
                    "product_type"
                )
            ),
            "tag_stabilization_applied": (
                tags.get(
                    "stabilization_applied"
                )
            ),
            "tag_stabilization_method": (
                tags.get(
                    "stabilization_method"
                )
            ),
            "tag_stabilization_window": (
                tags.get(
                    "stabilization_window"
                )
            ),
            "tag_provisional": (
                tags.get(
                    "provisional"
                )
            ),
        }


        class_records = []


        for class_id, class_name in (
            CLASS_DEFINITIONS.items()
        ):

            class_pixels = int(
                np.sum(
                    data
                    ==
                    class_id
                )
            )

            class_percent = (
                class_pixels
                /
                valid_pixels
                *
                100
                if valid_pixels
                >
                0
                else
                np.nan
            )

            class_area_km2 = (
                class_pixels
                *
                pixel_area_m2
                /
                1_000_000
            )


            class_records.append(
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
                    "class_pixels": (
                        class_pixels
                    ),
                    "class_percent": (
                        class_percent
                    ),
                    "class_area_km2": (
                        class_area_km2
                    ),
                }
            )


    return (
        raster_record,
        class_records,
        data,
    )


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 09 - ANNUAL LULC MAP QUALITY ASSURANCE")
print("=" * 100)


if not STEP_08_CLASS_SUMMARY.exists():

    raise FileNotFoundError(
        f"Step 08 class summary not found:\n"
        f"{STEP_08_CLASS_SUMMARY}"
    )


# =============================================================================
# 6. VERIFY EXACT FILE SET
# =============================================================================

raw_files = sorted(
    RAW_DIR.glob(
        "*.tif"
    ),
    key=extract_year,
)


stabilized_files = sorted(
    STABILIZED_DIR.glob(
        "*.tif"
    ),
    key=extract_year,
)


raw_years = [
    extract_year(
        path
    )
    for path in raw_files
]


stabilized_years = [
    extract_year(
        path
    )
    for path in stabilized_files
]


raw_years_complete = (
    raw_years
    ==
    EXPECTED_YEARS
)


stabilized_years_complete = (
    stabilized_years
    ==
    EXPECTED_YEARS
)


if not raw_years_complete:

    raise RuntimeError(
        f"RAW year coverage invalid: "
        f"{raw_years}"
    )


if not stabilized_years_complete:

    raise RuntimeError(
        f"STABILIZED year coverage invalid: "
        f"{stabilized_years}"
    )


print(
    f"\nRAW maps        : "
    f"{len(raw_files)}"
)

print(
    f"STABILIZED maps : "
    f"{len(stabilized_files)}"
)

print(
    f"Years complete  : "
    f"{raw_years_complete and stabilized_years_complete}"
)


# =============================================================================
# 7. INSPECT ALL MAPS
# =============================================================================

raster_records = []

class_records = []

raw_arrays = {}

stabilized_arrays = {}


reference_transform = None
reference_width = None
reference_height = None


for year in EXPECTED_YEARS:

    raw_path = expected_raw_path(
        year
    )

    stabilized_path = (
        expected_stabilized_path(
            year
        )
    )


    for product_type, raster_path in [
        (
            "RAW_RF",
            raw_path,
        ),
        (
            "STABILIZED_ANALYTICAL",
            stabilized_path,
        ),
    ]:

        if not raster_path.exists():

            raise FileNotFoundError(
                f"Expected map not found:\n"
                f"{raster_path}"
            )


        (
            raster_record,
            raster_class_records,
            raster_array,
        ) = inspect_raster(
            raster_path,
            year,
            product_type,
        )


        if reference_transform is None:

            reference_transform = (
                raster_record[
                    "transform"
                ]
            )

            reference_width = (
                raster_record[
                    "width"
                ]
            )

            reference_height = (
                raster_record[
                    "height"
                ]
            )


        raster_record[
            "grid_matches_reference"
        ] = (
            raster_record[
                "transform"
            ]
            ==
            reference_transform
            and
            raster_record[
                "width"
            ]
            ==
            reference_width
            and
            raster_record[
                "height"
            ]
            ==
            reference_height
        )


        raster_record[
            "crs_valid"
        ] = (
            raster_record[
                "crs"
            ]
            ==
            EXPECTED_CRS
        )


        raster_record[
            "resolution_valid"
        ] = (
            abs(
                raster_record[
                    "pixel_width_m"
                ]
                -
                EXPECTED_PIXEL_SIZE
            )
            <
            1e-9
            and
            abs(
                raster_record[
                    "pixel_height_m"
                ]
                -
                EXPECTED_PIXEL_SIZE
            )
            <
            1e-9
        )


        raster_record[
            "nodata_valid"
        ] = (
            raster_record[
                "nodata"
            ]
            ==
            0
        )


        raster_record[
            "class_values_valid"
        ] = (
            raster_record[
                "invalid_class_value_count"
            ]
            ==
            0
        )


        raster_record[
            "band_count_valid"
        ] = (
            raster_record[
                "band_count"
            ]
            ==
            1
        )


        raster_record[
            "dtype_valid"
        ] = (
            raster_record[
                "dtype"
            ]
            ==
            "uint8"
        )


        raster_record[
            "tag_year_valid"
        ] = (
            raster_record[
                "tag_year"
            ]
            ==
            str(
                year
            )
        )


        expected_provisional = (
            year
            ==
            2026
        )


        raster_record[
            "tag_provisional_valid"
        ] = (
            raster_record[
                "tag_provisional"
            ]
            ==
            str(
                expected_provisional
            )
        )


        raster_records.append(
            raster_record
        )


        class_records.extend(
            raster_class_records
        )


        if product_type == "RAW_RF":

            raw_arrays[
                year
            ] = raster_array

        else:

            stabilized_arrays[
                year
            ] = raster_array


# =============================================================================
# 8. CLASS PERCENTAGE TOTAL CHECK
# =============================================================================

raster_qa_df = pd.DataFrame(
    raster_records
)


class_qa_df = pd.DataFrame(
    class_records
)


percentage_totals = (
    class_qa_df.groupby(
        [
            "year",
            "product_type",
        ]
    )[
        "class_percent"
    ]
    .sum()
    .reset_index(
        name="class_percent_total"
    )
)


percentage_totals[
    "class_percent_total_valid"
] = np.isclose(
    percentage_totals[
        "class_percent_total"
    ],
    100.0,
    atol=1e-8,
)


class_qa_df = class_qa_df.merge(
    percentage_totals,
    on=[
        "year",
        "product_type",
    ],
    how="left",
)


# =============================================================================
# 9. RAW VS STABILIZED COMPARISON
# =============================================================================

comparison_records = []


for year in EXPECTED_YEARS:

    raw_array = raw_arrays[
        year
    ]

    stabilized_array = (
        stabilized_arrays[
            year
        ]
    )


    common_valid = (
        np.isin(
            raw_array,
            CLASS_IDS,
        )
        &
        np.isin(
            stabilized_array,
            CLASS_IDS,
        )
    )


    comparison_pixels = int(
        common_valid.sum()
    )


    changed_pixels = int(
        np.sum(
            (
                raw_array
                !=
                stabilized_array
            )
            &
            common_valid
        )
    )


    changed_percent = (
        changed_pixels
        /
        comparison_pixels
        *
        100
        if comparison_pixels
        >
        0
        else
        np.nan
    )


    should_be_identical = (
        year
        in
        EXPECTED_IDENTICAL_YEARS
    )


    should_allow_change = (
        year
        in
        EXPECTED_STABILIZED_YEARS
    )


    if should_be_identical:

        temporal_rule_valid = (
            changed_pixels
            ==
            0
        )

    elif should_allow_change:

        temporal_rule_valid = (
            True
        )

    else:

        temporal_rule_valid = (
            False
        )


    comparison_records.append(
        {
            "year": (
                year
            ),
            "comparison_pixels": (
                comparison_pixels
            ),
            "changed_pixels": (
                changed_pixels
            ),
            "changed_percent": (
                changed_percent
            ),
            "expected_identical": (
                should_be_identical
            ),
            "stabilization_expected": (
                should_allow_change
            ),
            "temporal_rule_valid": (
                temporal_rule_valid
            ),
        }
    )


comparison_df = pd.DataFrame(
    comparison_records
)


# =============================================================================
# 10. VERIFY AGAINST STEP 08 STATISTICS
# =============================================================================

step_08_df = pd.read_csv(
    STEP_08_CLASS_SUMMARY
)


comparison_stats_df = (
    class_qa_df[
        [
            "year",
            "product_type",
            "class_id",
            "class_percent",
            "class_area_km2",
        ]
    ]
    .merge(
        step_08_df[
            [
                "year",
                "product_type",
                "class_id",
                "class_percent",
                "class_area_km2",
            ]
        ],
        on=[
            "year",
            "product_type",
            "class_id",
        ],
        how="outer",
        suffixes=(
            "_step09",
            "_step08",
        ),
        validate="one_to_one",
    )
)


comparison_stats_df[
    "percent_difference"
] = np.abs(
    comparison_stats_df[
        "class_percent_step09"
    ]
    -
    comparison_stats_df[
        "class_percent_step08"
    ]
)


comparison_stats_df[
    "area_difference_km2"
] = np.abs(
    comparison_stats_df[
        "class_area_km2_step09"
    ]
    -
    comparison_stats_df[
        "class_area_km2_step08"
    ]
)


statistics_match_step08 = bool(
    (
        comparison_stats_df[
            "percent_difference"
        ]
        <=
        STAT_TOLERANCE
    ).all()
    and
    (
        comparison_stats_df[
            "area_difference_km2"
        ]
        <=
        STAT_TOLERANCE
    ).all()
)


# =============================================================================
# 11. METADATA RULE CHECKS
# =============================================================================

metadata_rule_records = []


for _, row in raster_qa_df.iterrows():

    year = int(
        row[
            "year"
        ]
    )

    product_type = (
        row[
            "product_type"
        ]
    )


    if product_type == "RAW_RF":

        expected_stabilization = (
            "False"
        )

        expected_method = (
            "NONE"
        )


    else:

        if year in EXPECTED_STABILIZED_YEARS:

            expected_stabilization = (
                "True"
            )

            expected_method = (
                "MODE_3YR"
            )

        else:

            expected_stabilization = (
                "False"
            )

            expected_method = (
                "NONE"
            )


    stabilization_tag_valid = (
        row[
            "tag_stabilization_applied"
        ]
        ==
        expected_stabilization
    )


    stabilization_method_tag_valid = (
        row[
            "tag_stabilization_method"
        ]
        ==
        expected_method
    )


    metadata_rule_records.append(
        {
            "year": (
                year
            ),
            "product_type": (
                product_type
            ),
            "stabilization_tag_valid": (
                stabilization_tag_valid
            ),
            "stabilization_method_tag_valid": (
                stabilization_method_tag_valid
            ),
        }
    )


metadata_rule_df = pd.DataFrame(
    metadata_rule_records
)


metadata_tags_valid = bool(
    metadata_rule_df[
        [
            "stabilization_tag_valid",
            "stabilization_method_tag_valid",
        ]
    ].all().all()
)


# =============================================================================
# 12. FINAL QA CONDITIONS
# =============================================================================

all_grid_match = bool(
    raster_qa_df[
        "grid_matches_reference"
    ].all()
)


all_crs_valid = bool(
    raster_qa_df[
        "crs_valid"
    ].all()
)


all_resolution_valid = bool(
    raster_qa_df[
        "resolution_valid"
    ].all()
)


all_nodata_valid = bool(
    raster_qa_df[
        "nodata_valid"
    ].all()
)


all_classes_valid = bool(
    raster_qa_df[
        "class_values_valid"
    ].all()
)


all_band_counts_valid = bool(
    raster_qa_df[
        "band_count_valid"
    ].all()
)


all_dtypes_valid = bool(
    raster_qa_df[
        "dtype_valid"
    ].all()
)


all_year_tags_valid = bool(
    raster_qa_df[
        "tag_year_valid"
    ].all()
)


all_provisional_tags_valid = bool(
    raster_qa_df[
        "tag_provisional_valid"
    ].all()
)


all_percentage_totals_valid = bool(
    percentage_totals[
        "class_percent_total_valid"
    ].all()
)


temporal_rules_valid = bool(
    comparison_df[
        "temporal_rule_valid"
    ].all()
)


all_checks_pass = all(
    [
        raw_years_complete,
        stabilized_years_complete,
        all_grid_match,
        all_crs_valid,
        all_resolution_valid,
        all_nodata_valid,
        all_classes_valid,
        all_band_counts_valid,
        all_dtypes_valid,
        all_year_tags_valid,
        all_provisional_tags_valid,
        all_percentage_totals_valid,
        temporal_rules_valid,
        statistics_match_step08,
        metadata_tags_valid,
    ]
)


overall_status = (
    "PASS"
    if all_checks_pass
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 13. SUMMARY
# =============================================================================

summary_df = pd.DataFrame(
    [
        {
            "raw_map_count": (
                len(
                    raw_files
                )
            ),
            "stabilized_map_count": (
                len(
                    stabilized_files
                )
            ),
            "raw_years_complete_2010_2026": (
                raw_years_complete
            ),
            "stabilized_years_complete_2010_2026": (
                stabilized_years_complete
            ),
            "all_grids_match": (
                all_grid_match
            ),
            "all_crs_epsg32644": (
                all_crs_valid
            ),
            "all_resolution_30m": (
                all_resolution_valid
            ),
            "all_nodata_zero": (
                all_nodata_valid
            ),
            "all_class_values_valid": (
                all_classes_valid
            ),
            "all_single_band": (
                all_band_counts_valid
            ),
            "all_uint8": (
                all_dtypes_valid
            ),
            "all_year_tags_valid": (
                all_year_tags_valid
            ),
            "all_provisional_tags_valid": (
                all_provisional_tags_valid
            ),
            "all_stabilization_tags_valid": (
                metadata_tags_valid
            ),
            "all_class_percent_totals_100": (
                all_percentage_totals_valid
            ),
            "raw_stabilized_temporal_rules_valid": (
                temporal_rules_valid
            ),
            "statistics_match_step08": (
                statistics_match_step08
            ),
            "overall_step09_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 14. SAVE REPORTS
# =============================================================================

raster_qa_df.to_csv(
    RASTER_QA_CSV,
    index=False,
)


class_qa_df.to_csv(
    CLASS_QA_CSV,
    index=False,
)


comparison_df.to_csv(
    RAW_STABILIZED_COMPARISON_CSV,
    index=False,
)


summary_df.to_csv(
    FINAL_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 15. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("RASTER STRUCTURAL QA")
print("=" * 100)


print(
    f"All grids match             : "
    f"{all_grid_match}"
)

print(
    f"All CRS EPSG:32644          : "
    f"{all_crs_valid}"
)

print(
    f"All resolution 30 m         : "
    f"{all_resolution_valid}"
)

print(
    f"All NoData = 0              : "
    f"{all_nodata_valid}"
)

print(
    f"All class values valid      : "
    f"{all_classes_valid}"
)

print(
    f"All single-band             : "
    f"{all_band_counts_valid}"
)

print(
    f"All uint8                   : "
    f"{all_dtypes_valid}"
)


print("\n" + "=" * 100)
print("RAW VS STABILIZED QA")
print("=" * 100)


print(
    comparison_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("METADATA / REPORT CONSISTENCY")
print("=" * 100)


print(
    f"Year tags valid             : "
    f"{all_year_tags_valid}"
)

print(
    f"Provisional tags valid      : "
    f"{all_provisional_tags_valid}"
)

print(
    f"Stabilization tags valid    : "
    f"{metadata_tags_valid}"
)

print(
    f"Class totals = 100%         : "
    f"{all_percentage_totals_valid}"
)

print(
    f"Statistics match Step 08    : "
    f"{statistics_match_step08}"
)

print(
    f"Temporal rules valid        : "
    f"{temporal_rules_valid}"
)


print("\n" + "=" * 100)
print("STEP 09 SUMMARY")
print("=" * 100)


print(
    f"RAW maps                    : "
    f"{len(raw_files)}"
)

print(
    f"STABILIZED maps             : "
    f"{len(stabilized_files)}"
)

print(
    f"Years                       : "
    f"2010-2026"
)

print(
    f"Overall Step 09 status      : "
    f"{overall_status}"
)


print("\nGenerated reports:")

print(
    f" - {relative_path(RASTER_QA_CSV)}"
)

print(
    f" - {relative_path(CLASS_QA_CSV)}"
)

print(
    f" - {relative_path(RAW_STABILIZED_COMPARISON_CSV)}"
)

print(
    f" - {relative_path(FINAL_SUMMARY_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. This step validates exported files only; it does not estimate classification accuracy."
)

print(
    "2. 2010, 2025 and 2026 must remain identical between RAW and STABILIZED products."
)

print(
    "3. Only 2011-2024 are allowed to contain MODE_3YR changes."
)

print(
    "4. 2026 remains provisional."
)

print(
    "5. After QA passes, persistent green-cover change analysis can begin."
)

print("=" * 100)