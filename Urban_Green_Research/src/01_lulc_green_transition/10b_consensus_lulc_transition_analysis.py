"""
STEP 10B - CONSENSUS-BASED SPATIAL LULC TRANSITION ANALYSIS

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Analyse pixel-level LULC transitions using stabilized annual maps.
2. Avoid unsafe single-year endpoint subtraction.
3. Build an EARLY consensus from 2012-2014.
4. Build a LATE consensus from 2021-2023.
5. Calculate both:
      ROBUST consensus = at least 2 of 3 years agree
      STRICT consensus = all 3 of 3 years agree
6. Quantify Green retention, Green -> Built-up conversion,
   Green -> Open/Bare, Green -> Water, and Green gains.
7. Export transition rasters and summary reports.

IMPORTANT:
- These are model-derived retrospective transition products.
- They are NOT independent ground-truth land-change observations.
- 2010, 2025 and provisional 2026 are excluded.
- No model retraining or classification modification occurs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STABILIZED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lulc"
    / "annual_maps"
    / "stabilized_mode_3yr"
)

TRANSITION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lulc"
    / "transitions"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

TRANSITION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. ANALYSIS WINDOWS
# =============================================================================

EARLY_YEARS = [
    2012,
    2013,
    2014,
]

LATE_YEARS = [
    2021,
    2022,
    2023,
]

ALL_REQUIRED_YEARS = (
    EARLY_YEARS
    +
    LATE_YEARS
)


# =============================================================================
# 3. CLASS DEFINITIONS
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

NODATA_VALUE = 0


# =============================================================================
# 4. GREEN TRANSITION CODES
# =============================================================================

GREEN_TRANSITION_CODES = {
    1: "Green retained",
    2: "Green -> Built-up",
    3: "Green -> Open / Bare",
    4: "Green -> Water",
    5: "Built-up -> Green",
    6: "Open / Bare -> Green",
    7: "Water -> Green",
    8: "Other non-green transition / retention",
}


# =============================================================================
# 5. OUTPUT RASTERS
# =============================================================================

ROBUST_EARLY_CONSENSUS_TIF = (
    TRANSITION_DIR
    / "Kaduwela_LULC_Early_Consensus_2012_2014_ROBUST.tif"
)

ROBUST_LATE_CONSENSUS_TIF = (
    TRANSITION_DIR
    / "Kaduwela_LULC_Late_Consensus_2021_2023_ROBUST.tif"
)

STRICT_EARLY_CONSENSUS_TIF = (
    TRANSITION_DIR
    / "Kaduwela_LULC_Early_Consensus_2012_2014_STRICT.tif"
)

STRICT_LATE_CONSENSUS_TIF = (
    TRANSITION_DIR
    / "Kaduwela_LULC_Late_Consensus_2021_2023_STRICT.tif"
)

ROBUST_TRANSITION_TIF = (
    TRANSITION_DIR
    / "Kaduwela_LULC_Transition_2012_2014_to_2021_2023_ROBUST.tif"
)

STRICT_TRANSITION_TIF = (
    TRANSITION_DIR
    / "Kaduwela_LULC_Transition_2012_2014_to_2021_2023_STRICT.tif"
)

ROBUST_GREEN_TRANSITION_TIF = (
    TRANSITION_DIR
    / "Kaduwela_Green_Transition_2012_2014_to_2021_2023_ROBUST.tif"
)

STRICT_GREEN_TRANSITION_TIF = (
    TRANSITION_DIR
    / "Kaduwela_Green_Transition_2012_2014_to_2021_2023_STRICT.tif"
)


# =============================================================================
# 6. OUTPUT REPORTS
# =============================================================================

CONSENSUS_COVERAGE_CSV = (
    REPORT_DIR
    / "10b_consensus_coverage_summary.csv"
)

TRANSITION_MATRIX_CSV = (
    REPORT_DIR
    / "10b_lulc_transition_matrix.csv"
)

GREEN_TRANSITION_SUMMARY_CSV = (
    REPORT_DIR
    / "10b_green_transition_summary.csv"
)

GREEN_TRANSITION_CLASS_CSV = (
    REPORT_DIR
    / "10b_green_transition_class_breakdown.csv"
)

FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "10b_consensus_transition_analysis_summary.csv"
)


# =============================================================================
# 7. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """Return project-relative path where possible."""

    try:

        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:

        return str(path)


def annual_path(year: int) -> Path:
    """Return stabilized annual map path."""

    return (
        STABILIZED_DIR
        /
        f"Kaduwela_LULC_STABILIZED_{year}.tif"
    )


def consensus_grid(
    annual_stack,
    common_valid_mask,
    minimum_votes,
):
    """
    Calculate categorical consensus.

    minimum_votes:
        2 -> robust 2-of-3 consensus
        3 -> strict 3-of-3 consensus

    Output:
        0 = unresolved / NoData
        1-4 = LULC class
    """

    vote_counts = np.stack(
        [
            np.sum(
                annual_stack
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


    winner_index = np.argmax(
        vote_counts,
        axis=0,
    )


    output = np.zeros(
        common_valid_mask.shape,
        dtype=np.uint8,
    )


    consensus_valid = (
        common_valid_mask
        &
        (
            maximum_votes
            >=
            minimum_votes
        )
    )


    for index, class_id in enumerate(
        CLASS_IDS
    ):

        output[
            consensus_valid
            &
            (
                winner_index
                ==
                index
            )
        ] = class_id


    return (
        output,
        maximum_votes,
    )


def create_transition_code(
    early_grid,
    late_grid,
):
    """
    Encode transition as:
        early class * 10 + late class

    Examples:
        11 = Green -> Green
        12 = Green -> Built-up
        21 = Built-up -> Green
        22 = Built-up -> Built-up

    0 = unresolved / NoData
    """

    output = np.zeros(
        early_grid.shape,
        dtype=np.uint8,
    )


    valid = (
        np.isin(
            early_grid,
            CLASS_IDS,
        )
        &
        np.isin(
            late_grid,
            CLASS_IDS,
        )
    )


    output[
        valid
    ] = (
        early_grid[
            valid
        ]
        *
        10
        +
        late_grid[
            valid
        ]
    )


    return output


def create_green_transition_grid(
    early_grid,
    late_grid,
):
    """
    Green-focused transition categories.

    1 = Green retained
    2 = Green -> Built-up
    3 = Green -> Open / Bare
    4 = Green -> Water
    5 = Built-up -> Green
    6 = Open / Bare -> Green
    7 = Water -> Green
    8 = Other non-green transition / retention
    0 = unresolved / NoData
    """

    output = np.zeros(
        early_grid.shape,
        dtype=np.uint8,
    )


    valid = (
        np.isin(
            early_grid,
            CLASS_IDS,
        )
        &
        np.isin(
            late_grid,
            CLASS_IDS,
        )
    )


    # Other resolved non-green pixels
    output[
        valid
    ] = 8


    output[
        valid
        &
        (
            early_grid
            ==
            1
        )
        &
        (
            late_grid
            ==
            1
        )
    ] = 1


    output[
        valid
        &
        (
            early_grid
            ==
            1
        )
        &
        (
            late_grid
            ==
            2
        )
    ] = 2


    output[
        valid
        &
        (
            early_grid
            ==
            1
        )
        &
        (
            late_grid
            ==
            3
        )
    ] = 3


    output[
        valid
        &
        (
            early_grid
            ==
            1
        )
        &
        (
            late_grid
            ==
            4
        )
    ] = 4


    output[
        valid
        &
        (
            early_grid
            ==
            2
        )
        &
        (
            late_grid
            ==
            1
        )
    ] = 5


    output[
        valid
        &
        (
            early_grid
            ==
            3
        )
        &
        (
            late_grid
            ==
            1
        )
    ] = 6


    output[
        valid
        &
        (
            early_grid
            ==
            4
        )
        &
        (
            late_grid
            ==
            1
        )
    ] = 7


    return output


def write_single_band_raster(
    output_path,
    grid,
    profile,
    description,
    tags,
):
    """Write one uint8 GeoTIFF."""

    output_profile = (
        profile.copy()
    )


    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
    )


    if output_path.exists():

        output_path.unlink()


    with rasterio.open(
        output_path,
        "w",
        **output_profile,
    ) as dst:

        dst.write(
            grid.astype(
                np.uint8
            ),
            1,
        )


        dst.set_band_description(
            1,
            description,
        )


        dst.update_tags(
            **tags
        )


def transition_matrix_records(
    method_name,
    early_grid,
    late_grid,
    pixel_area_km2,
):
    """Create complete 4 x 4 transition matrix records."""

    valid = (
        np.isin(
            early_grid,
            CLASS_IDS,
        )
        &
        np.isin(
            late_grid,
            CLASS_IDS,
        )
    )


    resolved_pixels = int(
        valid.sum()
    )


    records = []


    for from_class in CLASS_IDS:

        from_resolved_pixels = int(
            np.sum(
                valid
                &
                (
                    early_grid
                    ==
                    from_class
                )
            )
        )


        for to_class in CLASS_IDS:

            transition_mask = (
                valid
                &
                (
                    early_grid
                    ==
                    from_class
                )
                &
                (
                    late_grid
                    ==
                    to_class
                )
            )


            transition_pixels = int(
                transition_mask.sum()
            )


            records.append(
                {
                    "consensus_method": (
                        method_name
                    ),
                    "from_class_id": (
                        from_class
                    ),
                    "from_class_name": (
                        CLASS_DEFINITIONS[
                            from_class
                        ]
                    ),
                    "to_class_id": (
                        to_class
                    ),
                    "to_class_name": (
                        CLASS_DEFINITIONS[
                            to_class
                        ]
                    ),
                    "transition_code": (
                        from_class
                        *
                        10
                        +
                        to_class
                    ),
                    "transition_pixels": (
                        transition_pixels
                    ),
                    "transition_area_km2": (
                        transition_pixels
                        *
                        pixel_area_km2
                    ),
                    "percent_of_resolved_area": (
                        transition_pixels
                        /
                        resolved_pixels
                        *
                        100
                        if resolved_pixels
                        >
                        0
                        else
                        np.nan
                    ),
                    "percent_of_resolved_from_class": (
                        transition_pixels
                        /
                        from_resolved_pixels
                        *
                        100
                        if from_resolved_pixels
                        >
                        0
                        else
                        np.nan
                    ),
                }
            )


    return records


def green_summary_record(
    method_name,
    early_grid,
    late_grid,
    common_valid,
    pixel_area_km2,
):
    """Create Green-focused summary for one consensus method."""

    early_valid = np.isin(
        early_grid,
        CLASS_IDS,
    )

    late_valid = np.isin(
        late_grid,
        CLASS_IDS,
    )

    transition_valid = (
        early_valid
        &
        late_valid
    )


    early_green_consensus = (
        common_valid
        &
        (
            early_grid
            ==
            1
        )
    )


    early_green_resolved = (
        transition_valid
        &
        (
            early_grid
            ==
            1
        )
    )


    late_green_resolved = (
        transition_valid
        &
        (
            late_grid
            ==
            1
        )
    )


    green_retained = (
        transition_valid
        &
        (
            early_grid
            ==
            1
        )
        &
        (
            late_grid
            ==
            1
        )
    )


    green_to_built = (
        transition_valid
        &
        (
            early_grid
            ==
            1
        )
        &
        (
            late_grid
            ==
            2
        )
    )


    green_to_open = (
        transition_valid
        &
        (
            early_grid
            ==
            1
        )
        &
        (
            late_grid
            ==
            3
        )
    )


    green_to_water = (
        transition_valid
        &
        (
            early_grid
            ==
            1
        )
        &
        (
            late_grid
            ==
            4
        )
    )


    built_to_green = (
        transition_valid
        &
        (
            early_grid
            ==
            2
        )
        &
        (
            late_grid
            ==
            1
        )
    )


    open_to_green = (
        transition_valid
        &
        (
            early_grid
            ==
            3
        )
        &
        (
            late_grid
            ==
            1
        )
    )


    water_to_green = (
        transition_valid
        &
        (
            early_grid
            ==
            4
        )
        &
        (
            late_grid
            ==
            1
        )
    )


    green_loss = (
        green_to_built
        |
        green_to_open
        |
        green_to_water
    )


    green_gain = (
        built_to_green
        |
        open_to_green
        |
        water_to_green
    )


    early_green_consensus_pixels = int(
        early_green_consensus.sum()
    )


    early_green_resolved_pixels = int(
        early_green_resolved.sum()
    )


    late_green_resolved_pixels = int(
        late_green_resolved.sum()
    )


    green_retained_pixels = int(
        green_retained.sum()
    )


    green_loss_pixels = int(
        green_loss.sum()
    )


    green_gain_pixels = int(
        green_gain.sum()
    )


    unresolved_late_for_early_green = int(
        early_green_consensus_pixels
        -
        early_green_resolved_pixels
    )


    return {
        "consensus_method": (
            method_name
        ),
        "common_source_pixels": (
            int(
                common_valid.sum()
            )
        ),
        "resolved_transition_pixels": (
            int(
                transition_valid.sum()
            )
        ),
        "early_green_consensus_pixels": (
            early_green_consensus_pixels
        ),
        "early_green_resolved_transition_pixels": (
            early_green_resolved_pixels
        ),
        "early_green_without_late_consensus_pixels": (
            unresolved_late_for_early_green
        ),
        "green_retained_pixels": (
            green_retained_pixels
        ),
        "green_retained_area_km2": (
            green_retained_pixels
            *
            pixel_area_km2
        ),
        "green_to_built_pixels": (
            int(
                green_to_built.sum()
            )
        ),
        "green_to_built_area_km2": (
            int(
                green_to_built.sum()
            )
            *
            pixel_area_km2
        ),
        "green_to_open_bare_pixels": (
            int(
                green_to_open.sum()
            )
        ),
        "green_to_open_bare_area_km2": (
            int(
                green_to_open.sum()
            )
            *
            pixel_area_km2
        ),
        "green_to_water_pixels": (
            int(
                green_to_water.sum()
            )
        ),
        "green_to_water_area_km2": (
            int(
                green_to_water.sum()
            )
            *
            pixel_area_km2
        ),
        "total_green_loss_pixels": (
            green_loss_pixels
        ),
        "total_green_loss_area_km2": (
            green_loss_pixels
            *
            pixel_area_km2
        ),
        "built_to_green_pixels": (
            int(
                built_to_green.sum()
            )
        ),
        "built_to_green_area_km2": (
            int(
                built_to_green.sum()
            )
            *
            pixel_area_km2
        ),
        "open_bare_to_green_pixels": (
            int(
                open_to_green.sum()
            )
        ),
        "open_bare_to_green_area_km2": (
            int(
                open_to_green.sum()
            )
            *
            pixel_area_km2
        ),
        "water_to_green_pixels": (
            int(
                water_to_green.sum()
            )
        ),
        "water_to_green_area_km2": (
            int(
                water_to_green.sum()
            )
            *
            pixel_area_km2
        ),
        "total_green_gain_pixels": (
            green_gain_pixels
        ),
        "total_green_gain_area_km2": (
            green_gain_pixels
            *
            pixel_area_km2
        ),
        "net_consensus_green_change_pixels": (
            late_green_resolved_pixels
            -
            early_green_resolved_pixels
        ),
        "net_consensus_green_change_area_km2": (
            (
                late_green_resolved_pixels
                -
                early_green_resolved_pixels
            )
            *
            pixel_area_km2
        ),
        "green_retention_percent_of_resolved_early_green": (
            green_retained_pixels
            /
            early_green_resolved_pixels
            *
            100
            if early_green_resolved_pixels
            >
            0
            else
            np.nan
        ),
        "green_loss_percent_of_resolved_early_green": (
            green_loss_pixels
            /
            early_green_resolved_pixels
            *
            100
            if early_green_resolved_pixels
            >
            0
            else
            np.nan
        ),
        "green_to_built_percent_of_resolved_early_green": (
            int(
                green_to_built.sum()
            )
            /
            early_green_resolved_pixels
            *
            100
            if early_green_resolved_pixels
            >
            0
            else
            np.nan
        ),
    }


# =============================================================================
# 8. START
# =============================================================================

print("=" * 100)
print("STEP 10B - CONSENSUS-BASED SPATIAL LULC TRANSITION ANALYSIS")
print("=" * 100)


print(
    f"\nEarly consensus window : "
    f"{EARLY_YEARS}"
)

print(
    f"Late consensus window  : "
    f"{LATE_YEARS}"
)

print(
    "ROBUST consensus       : >=2 of 3 years"
)

print(
    "STRICT consensus       : 3 of 3 years"
)


# =============================================================================
# 9. LOAD REQUIRED STABILIZED MAPS
# =============================================================================

annual_arrays = {}

valid_masks = {}

reference_profile = None
reference_crs = None
reference_transform = None
reference_width = None
reference_height = None
pixel_area_km2 = None


for year in ALL_REQUIRED_YEARS:

    raster_path = annual_path(
        year
    )


    if not raster_path.exists():

        raise FileNotFoundError(
            f"Required stabilized map not found:\n"
            f"{raster_path}"
        )


    with rasterio.open(
        raster_path
    ) as src:

        data = src.read(
            1
        )


        if reference_profile is None:

            reference_profile = (
                src.profile.copy()
            )

            reference_crs = (
                src.crs
            )

            reference_transform = (
                src.transform
            )

            reference_width = (
                src.width
            )

            reference_height = (
                src.height
            )

            pixel_area_km2 = float(
                abs(
                    src.transform.a
                    *
                    src.transform.e
                )
                /
                1_000_000
            )


        else:

            if (
                src.crs
                !=
                reference_crs
                or
                src.width
                !=
                reference_width
                or
                src.height
                !=
                reference_height
                or
                not src.transform.almost_equals(
                    reference_transform
                )
            ):

                raise RuntimeError(
                    f"Grid mismatch detected for {year}."
                )


        invalid_values = (
            set(
                int(
                    value
                )
                for value in np.unique(
                    data
                )
            )
            -
            {
                0,
                1,
                2,
                3,
                4,
            }
        )


        if invalid_values:

            raise RuntimeError(
                f"Invalid LULC values in {year}: "
                f"{sorted(invalid_values)}"
            )


        annual_arrays[
            year
        ] = data.astype(
            np.uint8
        )


        valid_masks[
            year
        ] = np.isin(
            data,
            CLASS_IDS,
        )


print(
    f"\nRaster grid            : "
    f"{reference_width} x {reference_height}"
)

print(
    f"Pixel area             : "
    f"{pixel_area_km2:.6f} km²"
)


# =============================================================================
# 10. COMMON VALID PIXELS
# =============================================================================

common_valid = np.logical_and.reduce(
    [
        valid_masks[
            year
        ]
        for year in ALL_REQUIRED_YEARS
    ]
)


common_valid_pixels = int(
    common_valid.sum()
)


common_valid_area_km2 = (
    common_valid_pixels
    *
    pixel_area_km2
)


print(
    f"Common valid pixels    : "
    f"{common_valid_pixels}"
)

print(
    f"Common valid area      : "
    f"{common_valid_area_km2:.3f} km²"
)


# =============================================================================
# 11. BUILD EARLY + LATE STACKS
# =============================================================================

early_stack = np.stack(
    [
        annual_arrays[
            year
        ]
        for year in EARLY_YEARS
    ],
    axis=0,
)


late_stack = np.stack(
    [
        annual_arrays[
            year
        ]
        for year in LATE_YEARS
    ],
    axis=0,
)


# =============================================================================
# 12. ROBUST CONSENSUS
# =============================================================================

(
    robust_early,
    robust_early_votes,
) = consensus_grid(
    early_stack,
    common_valid,
    minimum_votes=2,
)


(
    robust_late,
    robust_late_votes,
) = consensus_grid(
    late_stack,
    common_valid,
    minimum_votes=2,
)


robust_transition = (
    create_transition_code(
        robust_early,
        robust_late,
    )
)


robust_green_transition = (
    create_green_transition_grid(
        robust_early,
        robust_late,
    )
)


# =============================================================================
# 13. STRICT CONSENSUS
# =============================================================================

(
    strict_early,
    strict_early_votes,
) = consensus_grid(
    early_stack,
    common_valid,
    minimum_votes=3,
)


(
    strict_late,
    strict_late_votes,
) = consensus_grid(
    late_stack,
    common_valid,
    minimum_votes=3,
)


strict_transition = (
    create_transition_code(
        strict_early,
        strict_late,
    )
)


strict_green_transition = (
    create_green_transition_grid(
        strict_early,
        strict_late,
    )
)


# =============================================================================
# 14. CONSENSUS COVERAGE SUMMARY
# =============================================================================

coverage_records = []


for (
    method_name,
    early_grid,
    late_grid,
) in [
    (
        "ROBUST_2_OF_3",
        robust_early,
        robust_late,
    ),
    (
        "STRICT_3_OF_3",
        strict_early,
        strict_late,
    ),
]:

    early_consensus_pixels = int(
        np.isin(
            early_grid,
            CLASS_IDS,
        ).sum()
    )


    late_consensus_pixels = int(
        np.isin(
            late_grid,
            CLASS_IDS,
        ).sum()
    )


    resolved_transition_pixels = int(
        (
            np.isin(
                early_grid,
                CLASS_IDS,
            )
            &
            np.isin(
                late_grid,
                CLASS_IDS,
            )
        ).sum()
    )


    coverage_records.append(
        {
            "consensus_method": (
                method_name
            ),
            "common_source_pixels": (
                common_valid_pixels
            ),
            "common_source_area_km2": (
                common_valid_area_km2
            ),
            "early_consensus_pixels": (
                early_consensus_pixels
            ),
            "early_consensus_coverage_percent": (
                early_consensus_pixels
                /
                common_valid_pixels
                *
                100
            ),
            "late_consensus_pixels": (
                late_consensus_pixels
            ),
            "late_consensus_coverage_percent": (
                late_consensus_pixels
                /
                common_valid_pixels
                *
                100
            ),
            "resolved_transition_pixels": (
                resolved_transition_pixels
            ),
            "resolved_transition_area_km2": (
                resolved_transition_pixels
                *
                pixel_area_km2
            ),
            "resolved_transition_coverage_percent": (
                resolved_transition_pixels
                /
                common_valid_pixels
                *
                100
            ),
        }
    )


coverage_df = pd.DataFrame(
    coverage_records
)


# =============================================================================
# 15. COMPLETE TRANSITION MATRICES
# =============================================================================

transition_records = []


transition_records.extend(
    transition_matrix_records(
        "ROBUST_2_OF_3",
        robust_early,
        robust_late,
        pixel_area_km2,
    )
)


transition_records.extend(
    transition_matrix_records(
        "STRICT_3_OF_3",
        strict_early,
        strict_late,
        pixel_area_km2,
    )
)


transition_matrix_df = pd.DataFrame(
    transition_records
)


# =============================================================================
# 16. GREEN TRANSITION SUMMARIES
# =============================================================================

green_summary_df = pd.DataFrame(
    [
        green_summary_record(
            "ROBUST_2_OF_3",
            robust_early,
            robust_late,
            common_valid,
            pixel_area_km2,
        ),
        green_summary_record(
            "STRICT_3_OF_3",
            strict_early,
            strict_late,
            common_valid,
            pixel_area_km2,
        ),
    ]
)


green_class_records = []


for (
    method_name,
    green_grid,
) in [
    (
        "ROBUST_2_OF_3",
        robust_green_transition,
    ),
    (
        "STRICT_3_OF_3",
        strict_green_transition,
    ),
]:

    resolved_mask = (
        green_grid
        >
        0
    )


    resolved_pixels = int(
        resolved_mask.sum()
    )


    for code, description in (
        GREEN_TRANSITION_CODES.items()
    ):

        pixels = int(
            np.sum(
                green_grid
                ==
                code
            )
        )


        green_class_records.append(
            {
                "consensus_method": (
                    method_name
                ),
                "green_transition_code": (
                    code
                ),
                "green_transition_name": (
                    description
                ),
                "pixels": (
                    pixels
                ),
                "area_km2": (
                    pixels
                    *
                    pixel_area_km2
                ),
                "percent_of_resolved_transition_area": (
                    pixels
                    /
                    resolved_pixels
                    *
                    100
                    if resolved_pixels
                    >
                    0
                    else
                    np.nan
                ),
            }
        )


green_class_df = pd.DataFrame(
    green_class_records
)


# =============================================================================
# 17. EXPORT RASTERS
# =============================================================================

base_tags = {
    "research_component": (
        "Urban Green Transition, Urbanization Pressure "
        "and Conservation Intelligence"
    ),
    "early_window": (
        "2012,2013,2014"
    ),
    "late_window": (
        "2021,2022,2023"
    ),
    "source_product": (
        "STABILIZED_ANALYTICAL"
    ),
    "worldcover_ground_truth": (
        "False"
    ),
    "transition_type": (
        "Model-derived retrospective consensus transition"
    ),
}


write_single_band_raster(
    ROBUST_EARLY_CONSENSUS_TIF,
    robust_early,
    reference_profile,
    "Early LULC consensus",
    {
        **base_tags,
        "consensus_method": (
            "ROBUST_2_OF_3"
        ),
        "consensus_period": (
            "EARLY_2012_2014"
        ),
    },
)


write_single_band_raster(
    ROBUST_LATE_CONSENSUS_TIF,
    robust_late,
    reference_profile,
    "Late LULC consensus",
    {
        **base_tags,
        "consensus_method": (
            "ROBUST_2_OF_3"
        ),
        "consensus_period": (
            "LATE_2021_2023"
        ),
    },
)


write_single_band_raster(
    STRICT_EARLY_CONSENSUS_TIF,
    strict_early,
    reference_profile,
    "Early LULC consensus",
    {
        **base_tags,
        "consensus_method": (
            "STRICT_3_OF_3"
        ),
        "consensus_period": (
            "EARLY_2012_2014"
        ),
    },
)


write_single_band_raster(
    STRICT_LATE_CONSENSUS_TIF,
    strict_late,
    reference_profile,
    "Late LULC consensus",
    {
        **base_tags,
        "consensus_method": (
            "STRICT_3_OF_3"
        ),
        "consensus_period": (
            "LATE_2021_2023"
        ),
    },
)


write_single_band_raster(
    ROBUST_TRANSITION_TIF,
    robust_transition,
    reference_profile,
    "LULC transition code",
    {
        **base_tags,
        "consensus_method": (
            "ROBUST_2_OF_3"
        ),
        "transition_code_rule": (
            "early_class*10+late_class"
        ),
    },
)


write_single_band_raster(
    STRICT_TRANSITION_TIF,
    strict_transition,
    reference_profile,
    "LULC transition code",
    {
        **base_tags,
        "consensus_method": (
            "STRICT_3_OF_3"
        ),
        "transition_code_rule": (
            "early_class*10+late_class"
        ),
    },
)


write_single_band_raster(
    ROBUST_GREEN_TRANSITION_TIF,
    robust_green_transition,
    reference_profile,
    "Green transition category",
    {
        **base_tags,
        "consensus_method": (
            "ROBUST_2_OF_3"
        ),
        "code_1": (
            "Green retained"
        ),
        "code_2": (
            "Green to Built-up"
        ),
        "code_3": (
            "Green to Open Bare"
        ),
        "code_4": (
            "Green to Water"
        ),
        "code_5": (
            "Built-up to Green"
        ),
        "code_6": (
            "Open Bare to Green"
        ),
        "code_7": (
            "Water to Green"
        ),
        "code_8": (
            "Other non-green"
        ),
    },
)


write_single_band_raster(
    STRICT_GREEN_TRANSITION_TIF,
    strict_green_transition,
    reference_profile,
    "Green transition category",
    {
        **base_tags,
        "consensus_method": (
            "STRICT_3_OF_3"
        ),
        "code_1": (
            "Green retained"
        ),
        "code_2": (
            "Green to Built-up"
        ),
        "code_3": (
            "Green to Open Bare"
        ),
        "code_4": (
            "Green to Water"
        ),
        "code_5": (
            "Built-up to Green"
        ),
        "code_6": (
            "Open Bare to Green"
        ),
        "code_7": (
            "Water to Green"
        ),
        "code_8": (
            "Other non-green"
        ),
    },
)


# =============================================================================
# 18. QA
# =============================================================================

robust_resolved_pixels = int(
    (
        robust_transition
        >
        0
    ).sum()
)


strict_resolved_pixels = int(
    (
        strict_transition
        >
        0
    ).sum()
)


strict_not_greater_than_robust = (
    strict_resolved_pixels
    <=
    robust_resolved_pixels
)


all_transition_codes_valid = bool(
    set(
        int(
            value
        )
        for value in np.unique(
            robust_transition
        )
    ).issubset(
        {
            0,
            11,
            12,
            13,
            14,
            21,
            22,
            23,
            24,
            31,
            32,
            33,
            34,
            41,
            42,
            43,
            44,
        }
    )
    and
    set(
        int(
            value
        )
        for value in np.unique(
            strict_transition
        )
    ).issubset(
        {
            0,
            11,
            12,
            13,
            14,
            21,
            22,
            23,
            24,
            31,
            32,
            33,
            34,
            41,
            42,
            43,
            44,
        }
    )
)


all_green_codes_valid = bool(
    set(
        int(
            value
        )
        for value in np.unique(
            robust_green_transition
        )
    ).issubset(
        set(
            range(
                0,
                9,
            )
        )
    )
    and
    set(
        int(
            value
        )
        for value in np.unique(
            strict_green_transition
        )
    ).issubset(
        set(
            range(
                0,
                9,
            )
        )
    )
)


all_outputs_exist = all(
    path.exists()
    for path in [
        ROBUST_EARLY_CONSENSUS_TIF,
        ROBUST_LATE_CONSENSUS_TIF,
        STRICT_EARLY_CONSENSUS_TIF,
        STRICT_LATE_CONSENSUS_TIF,
        ROBUST_TRANSITION_TIF,
        STRICT_TRANSITION_TIF,
        ROBUST_GREEN_TRANSITION_TIF,
        STRICT_GREEN_TRANSITION_TIF,
    ]
)


overall_status = (
    "PASS"
    if all(
        [
            common_valid_pixels
            >
            0,
            strict_not_greater_than_robust,
            all_transition_codes_valid,
            all_green_codes_valid,
            all_outputs_exist,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 19. FINAL SUMMARY
# =============================================================================

robust_green_row = (
    green_summary_df[
        green_summary_df[
            "consensus_method"
        ]
        ==
        "ROBUST_2_OF_3"
    ]
    .iloc[
        0
    ]
)


strict_green_row = (
    green_summary_df[
        green_summary_df[
            "consensus_method"
        ]
        ==
        "STRICT_3_OF_3"
    ]
    .iloc[
        0
    ]
)


final_summary_df = pd.DataFrame(
    [
        {
            "early_window": (
                "2012-2014"
            ),
            "late_window": (
                "2021-2023"
            ),
            "common_valid_pixels": (
                common_valid_pixels
            ),
            "common_valid_area_km2": (
                common_valid_area_km2
            ),
            "robust_resolved_transition_pixels": (
                robust_resolved_pixels
            ),
            "strict_resolved_transition_pixels": (
                strict_resolved_pixels
            ),
            "robust_green_to_built_area_km2": (
                robust_green_row[
                    "green_to_built_area_km2"
                ]
            ),
            "strict_green_to_built_area_km2": (
                strict_green_row[
                    "green_to_built_area_km2"
                ]
            ),
            "robust_total_green_loss_area_km2": (
                robust_green_row[
                    "total_green_loss_area_km2"
                ]
            ),
            "strict_total_green_loss_area_km2": (
                strict_green_row[
                    "total_green_loss_area_km2"
                ]
            ),
            "robust_total_green_gain_area_km2": (
                robust_green_row[
                    "total_green_gain_area_km2"
                ]
            ),
            "strict_total_green_gain_area_km2": (
                strict_green_row[
                    "total_green_gain_area_km2"
                ]
            ),
            "single_year_endpoint_subtraction_used": (
                False
            ),
            "year_2010_used": (
                False
            ),
            "year_2025_used": (
                False
            ),
            "year_2026_used": (
                False
            ),
            "model_retrained": (
                False
            ),
            "overall_step10b_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 20. SAVE REPORTS
# =============================================================================

coverage_df.to_csv(
    CONSENSUS_COVERAGE_CSV,
    index=False,
)


transition_matrix_df.to_csv(
    TRANSITION_MATRIX_CSV,
    index=False,
)


green_summary_df.to_csv(
    GREEN_TRANSITION_SUMMARY_CSV,
    index=False,
)


green_class_df.to_csv(
    GREEN_TRANSITION_CLASS_CSV,
    index=False,
)


final_summary_df.to_csv(
    FINAL_SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 21. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("CONSENSUS COVERAGE")
print("=" * 100)


print(
    coverage_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("GREEN TRANSITION SUMMARY")
print("=" * 100)


print(
    green_summary_df[
        [
            "consensus_method",
            "early_green_resolved_transition_pixels",
            "green_retained_area_km2",
            "green_to_built_area_km2",
            "green_to_open_bare_area_km2",
            "green_to_water_area_km2",
            "total_green_loss_area_km2",
            "total_green_gain_area_km2",
            "net_consensus_green_change_area_km2",
            "green_retention_percent_of_resolved_early_green",
            "green_loss_percent_of_resolved_early_green",
            "green_to_built_percent_of_resolved_early_green",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("ROBUST 2-OF-3 TRANSITION MATRIX")
print("=" * 100)


robust_matrix = (
    transition_matrix_df[
        transition_matrix_df[
            "consensus_method"
        ]
        ==
        "ROBUST_2_OF_3"
    ]
    .pivot(
        index="from_class_name",
        columns="to_class_name",
        values="transition_area_km2",
    )
)


print(
    robust_matrix.to_string()
)


print("\n" + "=" * 100)
print("STRICT 3-OF-3 TRANSITION MATRIX")
print("=" * 100)


strict_matrix = (
    transition_matrix_df[
        transition_matrix_df[
            "consensus_method"
        ]
        ==
        "STRICT_3_OF_3"
    ]
    .pivot(
        index="from_class_name",
        columns="to_class_name",
        values="transition_area_km2",
    )
)


print(
    strict_matrix.to_string()
)


print("\n" + "=" * 100)
print("STEP 10B QA")
print("=" * 100)


print(
    f"Common valid pixels                 : "
    f"{common_valid_pixels}"
)

print(
    f"Common valid area                   : "
    f"{common_valid_area_km2:.3f} km²"
)

print(
    f"ROBUST resolved transition pixels   : "
    f"{robust_resolved_pixels}"
)

print(
    f"STRICT resolved transition pixels   : "
    f"{strict_resolved_pixels}"
)

print(
    f"STRICT <= ROBUST coverage           : "
    f"{strict_not_greater_than_robust}"
)

print(
    f"Transition codes valid              : "
    f"{all_transition_codes_valid}"
)

print(
    f"Green transition codes valid        : "
    f"{all_green_codes_valid}"
)

print(
    f"2010 / 2025 / 2026 excluded         : "
    f"True"
)

print(
    f"Overall Step 10B status             : "
    f"{overall_status}"
)


print("\nGenerated rasters:")

for output_path in [
    ROBUST_EARLY_CONSENSUS_TIF,
    ROBUST_LATE_CONSENSUS_TIF,
    STRICT_EARLY_CONSENSUS_TIF,
    STRICT_LATE_CONSENSUS_TIF,
    ROBUST_TRANSITION_TIF,
    STRICT_TRANSITION_TIF,
    ROBUST_GREEN_TRANSITION_TIF,
    STRICT_GREEN_TRANSITION_TIF,
]:

    print(
        f" - {relative_path(output_path)}"
    )


print("\nGenerated reports:")

for output_path in [
    CONSENSUS_COVERAGE_CSV,
    TRANSITION_MATRIX_CSV,
    GREEN_TRANSITION_SUMMARY_CSV,
    GREEN_TRANSITION_CLASS_CSV,
    FINAL_SUMMARY_CSV,
]:

    print(
        f" - {relative_path(output_path)}"
    )


print("\nIMPORTANT:")

print(
    "1. These transitions are based on multi-year consensus, not single-year endpoint subtraction."
)

print(
    "2. ROBUST requires at least 2 of 3 annual classifications to agree in each window."
)

print(
    "3. STRICT requires all 3 annual classifications to agree in each window."
)

print(
    "4. Green -> Built-up area can now be distinguished from Green -> other classes."
)

print(
    "5. The results remain model-derived retrospective transition intelligence, not independent ground truth."
)

print(
    "6. We will compare ROBUST and STRICT coverage before choosing the primary reporting rule."
)

print("=" * 100)