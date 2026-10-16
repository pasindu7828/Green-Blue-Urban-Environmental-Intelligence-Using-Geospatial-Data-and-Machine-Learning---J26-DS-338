"""
STEP 10C - TRANSITION CONFIDENCE STRATIFICATION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Compare ROBUST 2-of-3 and STRICT 3-of-3 transition products.
2. Confirm that every STRICT transition is a subset of the ROBUST result.
3. Separate transition evidence into:
      ROBUST_ONLY
      STRICT_CONFIRMED
4. Create confidence layers for:
      - all LULC transitions
      - Green -> Built-up
      - total Green loss
5. Record the final reporting rule for the historical transition stage.

IMPORTANT:
- ROBUST is the primary retrospective transition product.
- STRICT is the conservative/high-confidence sensitivity product.
- The difference between ROBUST and STRICT is NOT a statistical
  confidence interval.
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

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. INPUT RASTERS
# =============================================================================

ROBUST_TRANSITION_TIF = (
    TRANSITION_DIR
    / "Kaduwela_LULC_Transition_2012_2014_to_2021_2023_ROBUST.tif"
)

STRICT_TRANSITION_TIF = (
    TRANSITION_DIR
    / "Kaduwela_LULC_Transition_2012_2014_to_2021_2023_STRICT.tif"
)


# =============================================================================
# 3. OUTPUT RASTERS
# =============================================================================

TRANSITION_CONFIDENCE_TIF = (
    TRANSITION_DIR
    / "Kaduwela_Transition_Confidence_2012_2014_to_2021_2023.tif"
)

GREEN_TO_BUILT_CONFIDENCE_TIF = (
    TRANSITION_DIR
    / "Kaduwela_GreenToBuilt_Confidence_2012_2014_to_2021_2023.tif"
)

GREEN_LOSS_CONFIDENCE_TIF = (
    TRANSITION_DIR
    / "Kaduwela_GreenLoss_Confidence_2012_2014_to_2021_2023.tif"
)


# =============================================================================
# 4. OUTPUT REPORTS
# =============================================================================

TRANSITION_CONFIDENCE_SUMMARY_CSV = (
    REPORT_DIR
    / "10c_transition_confidence_summary.csv"
)

TRANSITION_CODE_CONFIDENCE_CSV = (
    REPORT_DIR
    / "10c_transition_code_confidence_breakdown.csv"
)

GREEN_CONFIDENCE_SUMMARY_CSV = (
    REPORT_DIR
    / "10c_green_transition_confidence_summary.csv"
)

METHOD_DECISION_CSV = (
    REPORT_DIR
    / "10c_transition_reporting_method_decision.csv"
)


# =============================================================================
# 5. CONSTANTS
# =============================================================================

VALID_TRANSITION_CODES = {
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

CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}

# Transition confidence raster:
# 0 = unresolved / NoData
# 1 = ROBUST-only
# 2 = STRICT-confirmed

CONFIDENCE_CODES = {
    0: "Unresolved / NoData",
    1: "ROBUST-only",
    2: "STRICT-confirmed",
}

GREEN_TO_BUILT_CODE = 12

GREEN_LOSS_CODES = {
    12,
    13,
    14,
}


# =============================================================================
# 6. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """Return project-relative path when possible."""

    try:

        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:

        return str(path)


def transition_name(code: int) -> str:
    """Convert transition code to readable class transition."""

    if code == 0:

        return "Unresolved / NoData"


    from_class = (
        code
        //
        10
    )

    to_class = (
        code
        %
        10
    )


    return (
        f"{CLASS_DEFINITIONS[from_class]}"
        f" -> "
        f"{CLASS_DEFINITIONS[to_class]}"
    )


def write_confidence_raster(
    output_path,
    grid,
    profile,
    description,
    product_type,
):
    """Write one confidence GeoTIFF."""

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
            research_component=(
                "Urban Green Transition, Urbanization Pressure "
                "and Conservation Intelligence"
            ),
            product_type=(
                product_type
            ),
            early_window=(
                "2012-2014"
            ),
            late_window=(
                "2021-2023"
            ),
            primary_transition_rule=(
                "ROBUST_2_OF_3"
            ),
            high_confidence_rule=(
                "STRICT_3_OF_3"
            ),
            code_0=(
                "Unresolved or not applicable"
            ),
            code_1=(
                "ROBUST-only evidence"
            ),
            code_2=(
                "STRICT-confirmed evidence"
            ),
            statistical_confidence_interval=(
                "False"
            ),
        )


# =============================================================================
# 7. START
# =============================================================================

print("=" * 100)
print("STEP 10C - TRANSITION CONFIDENCE STRATIFICATION")
print("=" * 100)


for required_file in [
    ROBUST_TRANSITION_TIF,
    STRICT_TRANSITION_TIF,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required transition raster not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 8. LOAD ROBUST TRANSITION
# =============================================================================

with rasterio.open(
    ROBUST_TRANSITION_TIF
) as robust_src:

    robust_grid = (
        robust_src.read(
            1
        )
        .astype(
            np.uint8
        )
    )

    reference_profile = (
        robust_src.profile.copy()
    )

    reference_crs = (
        robust_src.crs
    )

    reference_transform = (
        robust_src.transform
    )

    reference_width = (
        robust_src.width
    )

    reference_height = (
        robust_src.height
    )

    pixel_area_km2 = float(
        abs(
            robust_src.transform.a
            *
            robust_src.transform.e
        )
        /
        1_000_000
    )


# =============================================================================
# 9. LOAD STRICT TRANSITION
# =============================================================================

with rasterio.open(
    STRICT_TRANSITION_TIF
) as strict_src:

    if (
        strict_src.crs
        !=
        reference_crs
        or
        strict_src.width
        !=
        reference_width
        or
        strict_src.height
        !=
        reference_height
        or
        not strict_src.transform.almost_equals(
            reference_transform
        )
    ):

        raise RuntimeError(
            "ROBUST and STRICT transition rasters do not share the same grid."
        )


    strict_grid = (
        strict_src.read(
            1
        )
        .astype(
            np.uint8
        )
    )


print(
    f"\nRaster grid       : "
    f"{reference_width} x {reference_height}"
)

print(
    f"Pixel area        : "
    f"{pixel_area_km2:.6f} km²"
)


# =============================================================================
# 10. VALIDATE TRANSITION CODES
# =============================================================================

robust_unique = set(
    int(
        value
    )
    for value in np.unique(
        robust_grid
    )
)


strict_unique = set(
    int(
        value
    )
    for value in np.unique(
        strict_grid
    )
)


if not robust_unique.issubset(
    VALID_TRANSITION_CODES
):

    raise RuntimeError(
        "Invalid transition code detected in ROBUST raster."
    )


if not strict_unique.issubset(
    VALID_TRANSITION_CODES
):

    raise RuntimeError(
        "Invalid transition code detected in STRICT raster."
    )


# =============================================================================
# 11. ROBUST / STRICT LOGICAL QA
# =============================================================================

robust_resolved = (
    robust_grid
    >
    0
)


strict_resolved = (
    strict_grid
    >
    0
)


strict_without_robust = (
    strict_resolved
    &
    ~robust_resolved
)


strict_transition_disagreement = (
    strict_resolved
    &
    robust_resolved
    &
    (
        strict_grid
        !=
        robust_grid
    )
)


strict_without_robust_pixels = int(
    strict_without_robust.sum()
)


strict_disagreement_pixels = int(
    strict_transition_disagreement.sum()
)


print("\n" + "=" * 100)
print("STRICT SUBSET QA")
print("=" * 100)


print(
    f"STRICT resolved without ROBUST      : "
    f"{strict_without_robust_pixels}"
)

print(
    f"STRICT/ROBUST transition disagreement: "
    f"{strict_disagreement_pixels}"
)


if strict_without_robust_pixels != 0:

    raise RuntimeError(
        "STRICT consensus exists where ROBUST consensus is unresolved."
    )


if strict_disagreement_pixels != 0:

    raise RuntimeError(
        "STRICT transition disagrees with ROBUST transition."
    )


# =============================================================================
# 12. COMPLETE TRANSITION CONFIDENCE GRID
# =============================================================================

transition_confidence_grid = np.zeros(
    robust_grid.shape,
    dtype=np.uint8,
)


# ROBUST transition exists but STRICT does not.
transition_confidence_grid[
    robust_resolved
    &
    ~strict_resolved
] = 1


# STRICT transition exists and therefore exactly confirms ROBUST.
transition_confidence_grid[
    strict_resolved
] = 2


# =============================================================================
# 13. GREEN -> BUILT-UP CONFIDENCE GRID
# =============================================================================

robust_green_to_built = (
    robust_grid
    ==
    GREEN_TO_BUILT_CODE
)


strict_green_to_built = (
    strict_grid
    ==
    GREEN_TO_BUILT_CODE
)


if np.any(
    strict_green_to_built
    &
    ~robust_green_to_built
):

    raise RuntimeError(
        "STRICT Green -> Built-up is not a subset of ROBUST Green -> Built-up."
    )


green_to_built_confidence_grid = np.zeros(
    robust_grid.shape,
    dtype=np.uint8,
)


green_to_built_confidence_grid[
    robust_green_to_built
    &
    ~strict_green_to_built
] = 1


green_to_built_confidence_grid[
    strict_green_to_built
] = 2


# =============================================================================
# 14. TOTAL GREEN-LOSS CONFIDENCE GRID
# =============================================================================

robust_green_loss = np.isin(
    robust_grid,
    list(
        GREEN_LOSS_CODES
    ),
)


strict_green_loss = np.isin(
    strict_grid,
    list(
        GREEN_LOSS_CODES
    ),
)


if np.any(
    strict_green_loss
    &
    ~robust_green_loss
):

    raise RuntimeError(
        "STRICT Green-loss pixels are not a subset of ROBUST Green-loss pixels."
    )


green_loss_confidence_grid = np.zeros(
    robust_grid.shape,
    dtype=np.uint8,
)


green_loss_confidence_grid[
    robust_green_loss
    &
    ~strict_green_loss
] = 1


green_loss_confidence_grid[
    strict_green_loss
] = 2


# =============================================================================
# 15. OVERALL TRANSITION CONFIDENCE SUMMARY
# =============================================================================

robust_resolved_pixels = int(
    robust_resolved.sum()
)


strict_confirmed_pixels = int(
    strict_resolved.sum()
)


robust_only_pixels = int(
    (
        robust_resolved
        &
        ~strict_resolved
    ).sum()
)


transition_confidence_summary_df = pd.DataFrame(
    [
        {
            "primary_consensus_rule": (
                "ROBUST_2_OF_3"
            ),
            "high_confidence_rule": (
                "STRICT_3_OF_3"
            ),
            "robust_resolved_pixels": (
                robust_resolved_pixels
            ),
            "robust_resolved_area_km2": (
                robust_resolved_pixels
                *
                pixel_area_km2
            ),
            "strict_confirmed_pixels": (
                strict_confirmed_pixels
            ),
            "strict_confirmed_area_km2": (
                strict_confirmed_pixels
                *
                pixel_area_km2
            ),
            "robust_only_pixels": (
                robust_only_pixels
            ),
            "robust_only_area_km2": (
                robust_only_pixels
                *
                pixel_area_km2
            ),
            "strict_confirmation_percent_of_robust_area": (
                strict_confirmed_pixels
                /
                robust_resolved_pixels
                *
                100
                if robust_resolved_pixels
                >
                0
                else
                np.nan
            ),
            "strict_without_robust_pixels": (
                strict_without_robust_pixels
            ),
            "strict_transition_disagreement_pixels": (
                strict_disagreement_pixels
            ),
        }
    ]
)


# =============================================================================
# 16. TRANSITION-CODE CONFIDENCE BREAKDOWN
# =============================================================================

transition_code_records = []


for transition_code in sorted(
    VALID_TRANSITION_CODES
):

    if transition_code == 0:

        continue


    robust_pixels = int(
        np.sum(
            robust_grid
            ==
            transition_code
        )
    )


    strict_pixels = int(
        np.sum(
            strict_grid
            ==
            transition_code
        )
    )


    robust_only_code_pixels = (
        robust_pixels
        -
        strict_pixels
    )


    if robust_only_code_pixels < 0:

        raise RuntimeError(
            f"STRICT count exceeds ROBUST count "
            f"for transition {transition_code}."
        )


    transition_code_records.append(
        {
            "transition_code": (
                transition_code
            ),
            "transition_name": (
                transition_name(
                    transition_code
                )
            ),
            "robust_pixels": (
                robust_pixels
            ),
            "robust_area_km2": (
                robust_pixels
                *
                pixel_area_km2
            ),
            "strict_confirmed_pixels": (
                strict_pixels
            ),
            "strict_confirmed_area_km2": (
                strict_pixels
                *
                pixel_area_km2
            ),
            "robust_only_pixels": (
                robust_only_code_pixels
            ),
            "robust_only_area_km2": (
                robust_only_code_pixels
                *
                pixel_area_km2
            ),
            "strict_confirmation_percent_of_robust": (
                strict_pixels
                /
                robust_pixels
                *
                100
                if robust_pixels
                >
                0
                else
                np.nan
            ),
        }
    )


transition_code_df = pd.DataFrame(
    transition_code_records
)


# =============================================================================
# 17. GREEN TRANSITION CONFIDENCE SUMMARY
# =============================================================================

robust_gtb_pixels = int(
    robust_green_to_built.sum()
)


strict_gtb_pixels = int(
    strict_green_to_built.sum()
)


robust_only_gtb_pixels = (
    robust_gtb_pixels
    -
    strict_gtb_pixels
)


robust_green_loss_pixels = int(
    robust_green_loss.sum()
)


strict_green_loss_pixels = int(
    strict_green_loss.sum()
)


robust_only_green_loss_pixels = (
    robust_green_loss_pixels
    -
    strict_green_loss_pixels
)


green_confidence_summary_df = pd.DataFrame(
    [
        {
            "metric": (
                "Green -> Built-up"
            ),
            "robust_primary_pixels": (
                robust_gtb_pixels
            ),
            "robust_primary_area_km2": (
                robust_gtb_pixels
                *
                pixel_area_km2
            ),
            "strict_confirmed_pixels": (
                strict_gtb_pixels
            ),
            "strict_confirmed_area_km2": (
                strict_gtb_pixels
                *
                pixel_area_km2
            ),
            "robust_only_pixels": (
                robust_only_gtb_pixels
            ),
            "robust_only_area_km2": (
                robust_only_gtb_pixels
                *
                pixel_area_km2
            ),
            "strict_confirmation_percent_of_robust": (
                strict_gtb_pixels
                /
                robust_gtb_pixels
                *
                100
                if robust_gtb_pixels
                >
                0
                else
                np.nan
            ),
        },
        {
            "metric": (
                "Total Green loss"
            ),
            "robust_primary_pixels": (
                robust_green_loss_pixels
            ),
            "robust_primary_area_km2": (
                robust_green_loss_pixels
                *
                pixel_area_km2
            ),
            "strict_confirmed_pixels": (
                strict_green_loss_pixels
            ),
            "strict_confirmed_area_km2": (
                strict_green_loss_pixels
                *
                pixel_area_km2
            ),
            "robust_only_pixels": (
                robust_only_green_loss_pixels
            ),
            "robust_only_area_km2": (
                robust_only_green_loss_pixels
                *
                pixel_area_km2
            ),
            "strict_confirmation_percent_of_robust": (
                strict_green_loss_pixels
                /
                robust_green_loss_pixels
                *
                100
                if robust_green_loss_pixels
                >
                0
                else
                np.nan
            ),
        },
    ]
)


# =============================================================================
# 18. FINAL METHOD DECISION
# =============================================================================

method_decision_df = pd.DataFrame(
    [
        {
            "historical_transition_primary_rule": (
                "ROBUST_2_OF_3"
            ),
            "historical_transition_sensitivity_rule": (
                "STRICT_3_OF_3"
            ),
            "primary_rule_reason": (
                "Retains near-complete spatial coverage while requiring "
                "multi-year agreement in both early and late periods."
            ),
            "strict_rule_role": (
                "Conservative high-confidence confirmation subset."
            ),
            "robust_green_to_built_area_km2": (
                robust_gtb_pixels
                *
                pixel_area_km2
            ),
            "strict_green_to_built_area_km2": (
                strict_gtb_pixels
                *
                pixel_area_km2
            ),
            "robust_only_green_to_built_area_km2": (
                robust_only_gtb_pixels
                *
                pixel_area_km2
            ),
            "robust_total_green_loss_area_km2": (
                robust_green_loss_pixels
                *
                pixel_area_km2
            ),
            "strict_total_green_loss_area_km2": (
                strict_green_loss_pixels
                *
                pixel_area_km2
            ),
            "difference_interpreted_as_statistical_confidence_interval": (
                False
            ),
            "recommended_reporting_language": (
                "Report ROBUST as the primary consensus transition estimate "
                "and STRICT as the conservative confirmed subset/sensitivity result."
            ),
            "transition_method_locked": (
                True
            ),
        }
    ]
)


# =============================================================================
# 19. EXPORT CONFIDENCE RASTERS
# =============================================================================

write_confidence_raster(
    TRANSITION_CONFIDENCE_TIF,
    transition_confidence_grid,
    reference_profile,
    "Transition confidence category",
    "ALL_LULC_TRANSITION_CONFIDENCE",
)


write_confidence_raster(
    GREEN_TO_BUILT_CONFIDENCE_TIF,
    green_to_built_confidence_grid,
    reference_profile,
    "Green to Built-up confidence category",
    "GREEN_TO_BUILT_CONFIDENCE",
)


write_confidence_raster(
    GREEN_LOSS_CONFIDENCE_TIF,
    green_loss_confidence_grid,
    reference_profile,
    "Green-loss confidence category",
    "GREEN_LOSS_CONFIDENCE",
)


# =============================================================================
# 20. FINAL QA
# =============================================================================

valid_confidence_codes = {
    0,
    1,
    2,
}


all_confidence_codes_valid = all(
    set(
        int(
            value
        )
        for value in np.unique(
            grid
        )
    ).issubset(
        valid_confidence_codes
    )
    for grid in [
        transition_confidence_grid,
        green_to_built_confidence_grid,
        green_loss_confidence_grid,
    ]
)


all_output_rasters_exist = all(
    path.exists()
    for path in [
        TRANSITION_CONFIDENCE_TIF,
        GREEN_TO_BUILT_CONFIDENCE_TIF,
        GREEN_LOSS_CONFIDENCE_TIF,
    ]
)


strict_subset_valid = (
    strict_without_robust_pixels
    ==
    0
    and
    strict_disagreement_pixels
    ==
    0
)


overall_status = (
    "PASS"
    if all(
        [
            strict_subset_valid,
            all_confidence_codes_valid,
            all_output_rasters_exist,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 21. SAVE REPORTS
# =============================================================================

transition_confidence_summary_df.to_csv(
    TRANSITION_CONFIDENCE_SUMMARY_CSV,
    index=False,
)


transition_code_df.to_csv(
    TRANSITION_CODE_CONFIDENCE_CSV,
    index=False,
)


green_confidence_summary_df.to_csv(
    GREEN_CONFIDENCE_SUMMARY_CSV,
    index=False,
)


method_decision_df.to_csv(
    METHOD_DECISION_CSV,
    index=False,
)


# =============================================================================
# 22. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("OVERALL TRANSITION CONFIDENCE")
print("=" * 100)


print(
    transition_confidence_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("GREEN TRANSITION CONFIDENCE")
print("=" * 100)


print(
    green_confidence_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("GREEN -> BUILT-UP REPORTING RESULT")
print("=" * 100)


print(
    f"ROBUST primary area             : "
    f"{robust_gtb_pixels * pixel_area_km2:.4f} km²"
)

print(
    f"STRICT-confirmed area           : "
    f"{strict_gtb_pixels * pixel_area_km2:.4f} km²"
)

print(
    f"ROBUST-only additional area     : "
    f"{robust_only_gtb_pixels * pixel_area_km2:.4f} km²"
)

print(
    f"STRICT confirmation of ROBUST   : "
    f"{strict_gtb_pixels / robust_gtb_pixels * 100:.3f}%"
)


print("\n" + "=" * 100)
print("TOTAL GREEN-LOSS REPORTING RESULT")
print("=" * 100)


print(
    f"ROBUST primary Green loss       : "
    f"{robust_green_loss_pixels * pixel_area_km2:.4f} km²"
)

print(
    f"STRICT-confirmed Green loss     : "
    f"{strict_green_loss_pixels * pixel_area_km2:.4f} km²"
)

print(
    f"ROBUST-only additional loss     : "
    f"{robust_only_green_loss_pixels * pixel_area_km2:.4f} km²"
)


print("\n" + "=" * 100)
print("STEP 10C QA")
print("=" * 100)


print(
    f"STRICT subset of ROBUST          : "
    f"{strict_subset_valid}"
)

print(
    f"STRICT/ROBUST disagreements      : "
    f"{strict_disagreement_pixels}"
)

print(
    f"Confidence codes valid           : "
    f"{all_confidence_codes_valid}"
)

print(
    f"Method decision locked           : "
    f"True"
)

print(
    f"Overall Step 10C status          : "
    f"{overall_status}"
)


print("\nGenerated rasters:")

print(
    f" - {relative_path(TRANSITION_CONFIDENCE_TIF)}"
)

print(
    f" - {relative_path(GREEN_TO_BUILT_CONFIDENCE_TIF)}"
)

print(
    f" - {relative_path(GREEN_LOSS_CONFIDENCE_TIF)}"
)


print("\nGenerated reports:")

print(
    f" - {relative_path(TRANSITION_CONFIDENCE_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(TRANSITION_CODE_CONFIDENCE_CSV)}"
)

print(
    f" - {relative_path(GREEN_CONFIDENCE_SUMMARY_CSV)}"
)

print(
    f" - {relative_path(METHOD_DECISION_CSV)}"
)


print("\nIMPORTANT:")

print(
    "1. ROBUST 2-of-3 is now the primary historical transition reporting rule."
)

print(
    "2. STRICT 3-of-3 is retained as the conservative high-confidence confirmation subset."
)

print(
    "3. The ROBUST-minus-STRICT difference is not a statistical confidence interval."
)

print(
    "4. Green -> Built-up confidence is now spatially mapped."
)

print(
    "5. No model or annual LULC classification was modified in this step."
)

print(
    "6. After this passes, the historical LULC / Green-transition stage can be frozen."
)

print("=" * 100)