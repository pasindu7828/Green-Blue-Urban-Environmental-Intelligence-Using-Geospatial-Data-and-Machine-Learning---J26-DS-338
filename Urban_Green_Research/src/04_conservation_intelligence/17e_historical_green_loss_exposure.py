
"""
J26-DS-338 | Section 04 - Conservation Intelligence

Stage 4.6: Historical Green-to-Built Loss Exposure

FROZEN HISTORICAL SOURCES:
- ROBUST 2-of-3 transition: primary
- STRICT 3-of-3 transition: sensitivity

Historical comparison:
Early consensus: 2012-2014
Late consensus : 2021-2023

This script:
1. Verifies specialized Green Transition code 2 against
   full LULC transition code 12 at identical pixels.
2. Reuses frozen historical conversion rasters.
3. Calculates nearest conversion-pixel-centre distances.
4. Calculates conversion exposure at 250 m and 500 m.
5. Preserves missing historical observations as NaN.
6. Retains all 9,240 KMC-intersecting 100 m cells.
7. Exports CSV, GeoPackage and one QA JSON.

METHOD:
Neighbourhoods use 30 m transition pixel centres within
250 m or 500 m of each full 100 m grid-cell centre.

Pixels within the focal 100 m grid footprint are excluded
from neighbourhood conversion measurements.

Pixel-count area = converted pixels x 900 square metres.
This is a raster pixel-centre approximation, not exact
circular-buffer or KMC-boundary fractional clipping.

Historical conversion exposure is descriptive, NOT a
future conversion probability.

EXCLUDED:
- Urbanization Pressure
- Development Suitability
- SHAP
- AHP
- Provisional UDA zoning

No Green Importance or Conservation Priority is calculated.
"""

from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import pyogrio

from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]

TRANSITION_DIR = (
    ROOT / "data/processed/lulc/transitions"
)

CURRENT_DIR = (
    ROOT / "data/processed/conservation_intelligence/"
    "current_green"
)

GRID_FILE = (
    ROOT / "data/processed/urbanization_pressure/grid/"
    "kaduwela_100m_modelling_grid.gpkg"
)

CURRENT_CSV = (
    CURRENT_DIR / "17b_current_green_condition_2025_100m.csv"
)

CURRENT_GPKG = (
    CURRENT_DIR / "17b_current_green_condition_2025_100m.gpkg"
)

CONNECTIVITY_QA = (
    ROOT / "outputs/reports/conservation_intelligence/"
    "17d_green_connectivity_qa.json"
)

RASTERS = {
    "ROBUST_GREEN": (
        TRANSITION_DIR /
        "Kaduwela_Green_Transition_"
        "2012_2014_to_2021_2023_ROBUST.tif"
    ),
    "STRICT_GREEN": (
        TRANSITION_DIR /
        "Kaduwela_Green_Transition_"
        "2012_2014_to_2021_2023_STRICT.tif"
    ),
    "ROBUST_FULL": (
        TRANSITION_DIR /
        "Kaduwela_LULC_Transition_"
        "2012_2014_to_2021_2023_ROBUST.tif"
    ),
    "STRICT_FULL": (
        TRANSITION_DIR /
        "Kaduwela_LULC_Transition_"
        "2012_2014_to_2021_2023_STRICT.tif"
    ),
}

OUTPUT_DIR = (
    ROOT / "data/processed/conservation_intelligence/"
    "historical_threat"
)

REPORT_DIR = (
    ROOT / "outputs/reports/conservation_intelligence"
)

CSV_OUT = (
    OUTPUT_DIR /
    "17e_historical_green_loss_exposure_100m.csv"
)

GPKG_OUT = (
    OUTPUT_DIR /
    "17e_historical_green_loss_exposure_100m.gpkg"
)

QA_OUT = (
    REPORT_DIR /
    "17e_historical_green_loss_exposure_qa.json"
)

# These encodings belong to DIFFERENT raster products.
GREEN_TRANSITION_CODE = 2
FULL_LULC_TRANSITION_CODE = 12

RADII = (250, 500)

FROZEN_AREAS_KM2 = {
    "ROBUST": 7.9821,
    "STRICT": 4.8267,
}


def load_raster(path):

    with rasterio.open(path) as src:

        return {
            "data": src.read(1),
            "crs": src.crs,
            "transform": src.transform,
            "shape": (src.height, src.width),
            "nodata": src.nodata,
            "pixel_area_m2": abs(
                src.transform.a * src.transform.e
                - src.transform.b * src.transform.d
            ),
        }


def raster_aligned(a, b):

    return (
        a["crs"] is not None
        and b["crs"] is not None
        and a["crs"].to_epsg() == 32644
        and b["crs"].to_epsg() == 32644
        and a["shape"] == b["shape"]
        and np.allclose(
            tuple(a["transform"]),
            tuple(b["transform"]),
            atol=1e-9,
        )
    )


def distribution(values):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {"count": 0}

    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def main():

    print("=" * 74)
    print(
        "SECTION 04 | STAGE 4.6 "
        "- HISTORICAL GREEN-LOSS EXPOSURE"
    )
    print("=" * 74)

    required = [
        GRID_FILE,
        CURRENT_CSV,
        CURRENT_GPKG,
        CONNECTIVITY_QA,
        *RASTERS.values(),
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    previous = json.loads(
        CONNECTIVITY_QA.read_text(encoding="utf-8")
    )

    if previous.get("status") != (
        "PASS_CONNECTIVITY_CANDIDATES"
    ):
        raise ValueError(
            "Stage 4.5 must pass before Stage 4.6"
        )

    checks = {}

    def check(name, condition):

        checks[name] = bool(condition)

        print(
            f"[{'PASS' if checks[name] else 'FAIL'}] "
            f"{name}"
        )

    # ==================================================
    # 1. VERIFY FROZEN TRANSITION CODEBOOK
    # ==================================================

    print("\n[1] VERIFY TRANSITION RASTER ENCODINGS")

    rasters = {
        key: load_raster(path)
        for key, path in RASTERS.items()
    }

    reference = rasters["ROBUST_GREEN"]

    for name, raster in rasters.items():

        check(
            f"{name.lower()}_spatial_alignment",
            raster_aligned(reference, raster),
        )

    check(
        "pixel_area_900m2",
        all(
            np.isclose(
                raster["pixel_area_m2"],
                900.0,
                atol=1e-8,
            )
            for raster in rasters.values()
        ),
    )

    if not all(checks.values()):
        raise SystemExit(
            "Raster alignment QA failed."
        )

    conversion = {}
    valid_transition = {}

    for version in ("ROBUST", "STRICT"):

        specialized = rasters[
            f"{version}_GREEN"
        ]["data"]

        full = rasters[
            f"{version}_FULL"
        ]["data"]

        # GREEN transition raster code 2:
        # Candidate historical Green-to-Built conversion.
        specialized_mask = (
            specialized == GREEN_TRANSITION_CODE
        )

        # FULL LULC transition encoding:
        # Early Green (1) -> Late Built (2) = 12.
        full_mask = (
            full == FULL_LULC_TRANSITION_CODE
        )

        matching_pixels = np.array_equal(
            specialized_mask, full_mask
        )

        check(
            f"{version.lower()}_green_to_built_codebook_match",
            matching_pixels,
        )

        # Specialized green-transition product has
        # observed classes 1 through 8; zero is nodata.
        valid_codes = np.unique(specialized)

        check(
            f"{version.lower()}_green_transition_codes_valid",
            np.isin(
                valid_codes,
                np.arange(0, 9),
            ).all(),
        )

        valid_transition[version] = (
            (specialized >= 1)
            & (specialized <= 8)
        )

        conversion[version] = specialized_mask

        pixel_count = int(
            specialized_mask.sum()
        )

        area_km2 = (
            pixel_count
            * reference["pixel_area_m2"]
            / 1e6
        )

        print(
            f"{version}: "
            f"{pixel_count:,} Green-to-Built pixels, "
            f"{area_km2:.4f} km2"
        )

        check(
            f"{version.lower()}_frozen_area_matches",
            np.isclose(
                area_km2,
                FROZEN_AREAS_KM2[version],
                atol=1e-6,
            ),
        )

    check(
        "strict_conversion_subset_of_robust",
        bool(
            np.all(
                ~conversion["STRICT"]
                | conversion["ROBUST"]
            )
        ),
    )

    if not all(checks.values()):
        raise SystemExit(
            "Transition encoding validation failed. "
            "Do not calculate historical exposure yet."
        )

    print(
        "Green Transition code 2 spatially matches "
        "Full LULC Transition code 12."
    )

    # ==================================================
    # 2. LOAD THE KMC GRID AND CURRENT GREEN EVIDENCE
    # ==================================================

    print("\n[2] VERIFY 100 M GRID AND CURRENT GREEN")

    grid = gpd.read_file(GRID_FILE)
    current = pd.read_csv(CURRENT_CSV)
    clipped = gpd.read_file(CURRENT_GPKG)

    required_current = [
        "grid_id",
        "kmc_overlap_area_m2",
        "current_green_fraction",
        "current_green_area_m2",
        "valid_lulc_coverage_fraction",
        "green_condition_status",
        "source_product_type",
    ]

    missing = sorted(
        set(required_current) - set(current.columns)
    )

    if missing:
        raise ValueError(
            f"Missing current-green fields: {missing}"
        )

    check(
        "grid_9240_unique_ids",
        len(grid) == 9240
        and grid.grid_id.notna().all()
        and grid.grid_id.is_unique,
    )

    check(
        "current_9240_unique_ids",
        len(current) == 9240
        and current.grid_id.notna().all()
        and current.grid_id.is_unique,
    )

    check(
        "clipped_9240_unique_ids",
        len(clipped) == 9240
        and clipped.grid_id.notna().all()
        and clipped.grid_id.is_unique,
    )

    check(
        "all_grid_ids_match",
        set(grid.grid_id)
        == set(current.grid_id)
        == set(clipped.grid_id),
    )

    check(
        "grid_and_clipped_epsg32644",
        grid.crs is not None
        and clipped.crs is not None
        and grid.crs.to_epsg() == 32644
        and clipped.crs.to_epsg() == 32644,
    )

    check(
        "current_green_is_2025_raw_rf",
        current.source_product_type.eq(
            "RAW_RF"
        ).all(),
    )

    if not all(checks.values()):
        raise SystemExit(
            "Grid/current-green verification failed."
        )

    ids = current.grid_id.tolist()

    grid = (
        grid.set_index("grid_id")
        .loc[ids]
    )

    clipped = (
        clipped.set_index("grid_id")
        .loc[ids]
    )

    centres = grid[
        ["centroid_x", "centroid_y"]
    ].to_numpy(dtype=float)

    bounds = grid.geometry.bounds[
        ["minx", "miny", "maxx", "maxy"]
    ].to_numpy(dtype=float)

    areas = current[
        "kmc_overlap_area_m2"
    ].to_numpy(dtype=float)

    check(
        "centres_and_bounds_finite",
        np.isfinite(centres).all()
        and np.isfinite(bounds).all(),
    )

    check(
        "clipped_areas_match_stored",
        np.isfinite(areas).all()
        and (areas > 0).all()
        and np.max(
            np.abs(
                clipped.geometry.area.to_numpy(dtype=float)
                - areas
            )
        ) <= 1.0,
    )

    if not all(checks.values()):
        raise SystemExit(
            "KMC area verification failed."
        )

    # ==================================================
    # 3. BUILD 30 M RASTER PIXEL-CENTRE SPATIAL INDEX
    # ==================================================

    print("\n[3] INDEX VALID HISTORICAL TRANSITION PIXELS")

    union_valid = (
        valid_transition["ROBUST"]
        | valid_transition["STRICT"]
    )

    pixel_rows, pixel_cols = np.nonzero(
        union_valid
    )

    transform = reference["transform"]

    c = pixel_cols.astype(float) + 0.5
    r = pixel_rows.astype(float) + 0.5

    pixel_x = (
        transform.a * c
        + transform.b * r
        + transform.c
    )

    pixel_y = (
        transform.d * c
        + transform.e * r
        + transform.f
    )

    pixel_centres = np.column_stack(
        [pixel_x, pixel_y]
    )

    if len(pixel_centres) == 0:
        raise ValueError(
            "No valid historical transition pixels"
        )

    spatial_tree = cKDTree(
        pixel_centres
    )

    valid_flags = {
        version: valid_transition[version][union_valid]
        for version in ("ROBUST", "STRICT")
    }

    loss_flags = {
        version: conversion[version][union_valid]
        for version in ("ROBUST", "STRICT")
    }

    nearest_distances = {}

    for version in ("ROBUST", "STRICT"):

        loss_pixel_centres = pixel_centres[
            loss_flags[version]
        ]

        if len(loss_pixel_centres) == 0:
            raise ValueError(
                f"{version}: no Green-to-Built pixels"
            )

        loss_tree = cKDTree(
            loss_pixel_centres
        )

        distances, _ = loss_tree.query(
            centres, k=1
        )

        nearest_distances[version] = distances

    print(
        "Valid union transition pixels:",
        len(pixel_centres)
    )

    print(
        "Nearest distances measured between "
        "100 m grid centres and 30 m pixel centres."
    )

    # ==================================================
    # 4. CALCULATE HISTORICAL-LOSS NEIGHBOURHOODS
    # ==================================================

    print("\n[4] CALCULATE 250 M AND 500 M EXPOSURE")

    n = len(current)

    counts = {}

    for version in ("ROBUST", "STRICT"):

        for radius in RADII:

            counts[(version, radius)] = {
                "valid": np.zeros(n, dtype=int),
                "converted": np.zeros(n, dtype=int),
            }

    focal_pixels_excluded = 0

    for i in range(n):

        nearby = np.asarray(
            spatial_tree.query_ball_point(
                centres[i],
                r=max(RADII),
            ),
            dtype=int,
        )

        if nearby.size == 0:
            continue

        xy = pixel_centres[nearby]

        # Exclude pixels in the focal full 100 m grid
        # footprint. This avoids counting the cell's
        # own historical conversion as surrounding loss.
        xmin, ymin, xmax, ymax = bounds[i]

        within_focal = (
            (xy[:, 0] >= xmin - 1e-9)
            & (xy[:, 0] <= xmax + 1e-9)
            & (xy[:, 1] >= ymin - 1e-9)
            & (xy[:, 1] <= ymax + 1e-9)
        )

        focal_pixels_excluded += int(
            within_focal.sum()
        )

        nearby = nearby[~within_focal]
        xy = xy[~within_focal]

        if nearby.size == 0:
            continue

        distances_sq = np.sum(
            (xy - centres[i]) ** 2,
            axis=1,
        )

        for radius in RADII:

            selected = nearby[
                distances_sq <= radius ** 2 + 1e-8
            ]

            if selected.size == 0:
                continue

            for version in ("ROBUST", "STRICT"):

                counts[(version, radius)][
                    "valid"
                ][i] = int(
                    valid_flags[version][selected].sum()
                )

                counts[(version, radius)][
                    "converted"
                ][i] = int(
                    loss_flags[version][selected].sum()
                )

    print(
        "Focal-cell pixel occurrences excluded:",
        focal_pixels_excluded,
    )

    # ==================================================
    # 5. CREATE GRID-LEVEL HISTORICAL EXPOSURE
    # ==================================================

    print("\n[5] BUILD HISTORICAL EXPOSURE DATASET")

    result = current[
        [
            "grid_id",
            "kmc_overlap_area_m2",
            "kmc_overlap_fraction",
            "current_green_fraction",
            "current_green_area_m2",
            "valid_lulc_coverage_fraction",
            "green_condition_status",
        ]
    ].copy()

    result["historical_loss_early_period"] = (
        "2012-2014"
    )

    result["historical_loss_late_period"] = (
        "2021-2023"
    )

    results_summary = {}

    for version in ("ROBUST", "STRICT"):

        prefix = f"historical_g2b_{version.lower()}"

        distances = nearest_distances[version]

        result[
            f"{prefix}_nearest_pixel_centre_m"
        ] = distances

        results_summary[version] = {
            "nearest_pixel_centre_distance_m": (
                distribution(distances)
            )
        }

        for radius in RADII:

            data = counts[(version, radius)]

            valid_n = data["valid"]
            loss_n = data["converted"]

            share = np.full(
                n, np.nan, dtype=float
            )

            available = valid_n > 0

            share[available] = (
                loss_n[available] / valid_n[available]
            )

            loss_area_km2 = (
                loss_n
                * reference["pixel_area_m2"]
                / 1e6
            )

            result[
                f"{prefix}_valid_pixels_{radius}m"
            ] = valid_n

            result[
                f"{prefix}_converted_pixels_{radius}m"
            ] = loss_n

            result[
                f"{prefix}_converted_area_km2_{radius}m"
            ] = loss_area_km2

            result[
                f"{prefix}_exposure_share_{radius}m"
            ] = share

            result[
                f"{prefix}_context_status_{radius}m"
            ] = np.where(
                available,
                "VALID_HISTORICAL_CONTEXT",
                "NO_VALID_HISTORICAL_CONTEXT",
            )

            results_summary[version][f"{radius}m"] = {
                "exposure_share": distribution(share),
                "converted_area_km2": (
                    distribution(loss_area_km2)
                ),
                "no_valid_context_cells": int(
                    (~available).sum()
                ),
                "cells_with_detected_conversion": int(
                    (loss_n > 0).sum()
                ),
            }

            print(
                f"{version} {radius} m | "
                f"Valid context: {int(available.sum()):,} | "
                f"Conversion detected: "
                f"{int((loss_n > 0).sum()):,} | "
                f"Missing context: {int((~available).sum()):,}"
            )

    # ==================================================
    # 6. VALIDATE HISTORICAL EXPOSURE
    # ==================================================

    print("\n[6] HISTORICAL LOSS QA")

    check(
        "9240_cells_preserved",
        len(result) == 9240
        and result.grid_id.notna().all()
        and result.grid_id.is_unique,
    )

    check(
        "current_green_unknowns_preserved",
        result["current_green_fraction"].isna().sum()
        == 45,
    )

    for version in ("ROBUST", "STRICT"):

        check(
            f"{version.lower()}_nearest_distances_valid",
            np.isfinite(
                nearest_distances[version]
            ).all()
            and (
                nearest_distances[version] >= 0
            ).all(),
        )

        for radius in RADII:

            data = counts[(version, radius)]

            valid_n = data["valid"]
            loss_n = data["converted"]

            prefix = f"historical_g2b_{version.lower()}"

            share = result[
                f"{prefix}_exposure_share_{radius}m"
            ].to_numpy(dtype=float)

            available = valid_n > 0

            check(
                f"{version.lower()}_{radius}m_counts_valid",
                bool(
                    (valid_n >= 0).all()
                    and (loss_n >= 0).all()
                    and (loss_n <= valid_n).all()
                ),
            )

            check(
                f"{version.lower()}_{radius}m_share_valid",
                bool(
                    np.isfinite(
                        share[available]
                    ).all()
                    and (
                        (share[available] >= 0)
                        & (share[available] <= 1)
                    ).all()
                    and np.isnan(
                        share[~available]
                    ).all()
                ),
            )

            check(
                f"{version.lower()}_{radius}m_reproduced",
                bool(
                    np.allclose(
                        share[available],
                        (
                            loss_n[available]
                            / valid_n[available]
                        ),
                        atol=1e-12,
                    )
                ),
            )

    forbidden = [
        "pressure",
        "shap",
        "suitability",
        "ahp",
        "zoning",
        "priority",
    ]

    check(
        "no_forbidden_model_inputs",
        not any(
            term in column.lower()
            for column in result.columns
            for term in forbidden
        ),
    )

    check(
        "all_frozen_transition_areas_preserved",
        all(
            np.isclose(
                conversion[version].sum()
                * reference["pixel_area_m2"] / 1e6,
                FROZEN_AREAS_KM2[version],
                atol=1e-6,
            )
            for version in ("ROBUST", "STRICT")
        ),
    )

    if not all(checks.values()):
        raise SystemExit(
            "Historical-loss QA failed. "
            "No exposure outputs exported."
        )

    # ==================================================
    # 7. EXPORT DATA AND REPORT
    # ==================================================

    print("\n[7] EXPORT HISTORICAL THREAT DATA")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        CSV_OUT, index=False
    )

    geo = clipped[
        ["geometry"]
    ].reset_index().merge(
        result,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    if len(geo) != 9240:
        raise ValueError(
            "GIS join lost grid cells"
        )

    geo.to_file(
        GPKG_OUT,
        layer="historical_green_loss_exposure_100m",
        driver="GPKG",
    )

    saved_csv = pd.read_csv(CSV_OUT)
    saved_gpkg = pyogrio.read_info(GPKG_OUT)

    check(
        "csv_export_9240",
        len(saved_csv) == 9240
        and saved_csv.grid_id.is_unique,
    )

    check(
        "gis_export_9240",
        saved_gpkg["features"] == 9240,
    )

    status = (
        "PASS_HISTORICAL_LOSS_EXPOSURE"
        if all(checks.values())
        else "REVIEW_REQUIRED"
    )

    qa = {
        "section": "04_conservation_intelligence",
        "stage": "4.6",
        "status": status,
        "grid_cells": 9240,
        "historical_early_period": "2012-2014",
        "historical_late_period": "2021-2023",
        "primary_transition": "ROBUST_2_OF_3",
        "sensitivity_transition": "STRICT_3_OF_3",
        "specialized_green_transition_code": 2,
        "full_lulc_transition_code": 12,
        "spatial_codebook_concordance_required": True,
        "frozen_conversion_areas_km2": (
            FROZEN_AREAS_KM2
        ),
        "neighbourhood_radii_m": list(RADII),
        "pixel_area_m2": (
            reference["pixel_area_m2"]
        ),
        "focal_pixel_occurrences_excluded": (
            focal_pixels_excluded
        ),
        "exposure_summary": results_summary,
        "checks": checks,
        "methodology": {
            "nearest_distance": (
                "Distance from grid-cell centre "
                "to nearest historical converted "
                "30m raster pixel centre"
            ),
            "neighbourhood_conversion_area": (
                "Number of historical G2B pixels "
                "within radius, excluding focal full "
                "100m grid cell, multiplied by 900m2"
            ),
            "neighbourhood_exposure_share": (
                "Historical G2B pixel count divided "
                "by valid labelled transition pixel "
                "count in the same neighbourhood"
            ),
            "missing_context": (
                "No valid transition pixels -> NaN, "
                "not zero historical exposure"
            ),
        },
        "limitations": [
            "Conversion reflects two historical "
            "consensus periods, not annual event dates.",
            "Pixel-centre neighbourhood approximation, "
            "not exact circular buffer overlay.",
            "Raster conversion areas use full 30m pixels, "
            "not fractional KMC-boundary pixel areas.",
            "Historical exposure is not future "
            "conversion probability.",
            "Neighbourhood exposure is descriptive.",
            "Current-green candidate threshold "
            "has not been selected.",
            "No Green Importance score calculated.",
            "Historical loss remains outside "
            "intrinsic Green Importance.",
            "No future Urbanization Pressure used.",
            "No Development Suitability used.",
        ],
        "green_importance_calculated": False,
        "future_pressure_used": False,
        "development_suitability_used": False,
    }

    QA_OUT.write_text(
        json.dumps(
            qa, indent=2
        ),
        encoding="utf-8",
    )

    print("CSV:", CSV_OUT)
    print("GPKG:", GPKG_OUT)
    print("QA:", QA_OUT)

    print("\n" + "=" * 74)
    print("STAGE 4.6:", status)
    print("Grid cells:", len(result))
    print("Historical loss source: FROZEN")
    print("Green Importance score: NOT CALCULATED")
    print("Future pressure used: NO")
    print("=" * 74)

    if status != "PASS_HISTORICAL_LOSS_EXPOSURE":
        raise SystemExit(
            "Stage 4.6 export QA failed."
        )


if __name__ == "__main__":
    main()
