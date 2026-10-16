
from pathlib import Path
from datetime import datetime, timezone
import csv
import json

import numpy as np
import rasterio
from scipy import ndimage


# ============================================================
# PART 4 - SEASONAL WATERLOGGING SUSCEPTIBILITY
# SCRIPT 07 - TERRAIN FEATURE GENERATION
#
# Terrain factors:
# 1. Slope in degrees (Horn 3x3)
# 2. Topographic Position Index / relative elevation
#    (elevation minus circular 300m neighbourhood mean)
#
# Primary DEM: GEDTM30
# Sensitivity DEM: NASADEM
#
# Outputs are restricted to the 56-GN study mask.
# Original DEMs and existing Part outputs are unchanged.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/processed/part4_waterlogging"
METRIC_DIR = ROOT / "outputs/metrics/part4_waterlogging"
REPORT_DIR = ROOT / "outputs/reports/part4_waterlogging"

MASK_FILE = DATA_DIR / "06_Kaduwela_Study_Mask_30m.tif"

DEMS = {
    "GEDTM30": DATA_DIR / "06_GEDTM30_UTM44N_30m.tif",
    "NASADEM": DATA_DIR / "06_NASADEM_UTM44N_30m.tif",
}

CELL_SIZE_M = 30.0
TPI_RADIUS_M = 300.0
NODATA = -9999.0
TARGET_CRS = "EPSG:32644"


def write_feature(path, values, reference):
    profile = reference.profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=NODATA,
        compress="deflate",
        predictor=3,
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype(np.float32), 1)


def summarize(values, study_mask):
    selected = values[study_mask]

    if not np.isfinite(selected).all():
        raise ValueError(
            "Feature contains invalid study-area values"
        )

    return {
        "minimum": round(float(selected.min()), 6),
        "maximum": round(float(selected.max()), 6),
        "mean": round(float(selected.mean()), 6),
        "median": round(float(np.median(selected)), 6),
        "p05": round(float(np.percentile(selected, 5)), 6),
        "p95": round(float(np.percentile(selected, 95)), 6),
    }


def main():
    print("=" * 70)
    print("PART 4 - TERRAIN FEATURE GENERATION")
    print("SCRIPT 07")
    print("=" * 70)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load common study mask
    # --------------------------------------------------------

    with rasterio.open(MASK_FILE) as src:
        study_mask = src.read(1) == 1
        expected_transform = src.transform
        expected_crs = src.crs
        expected_shape = src.shape

    if str(expected_crs) != TARGET_CRS:
        raise ValueError("Unexpected study-mask CRS")

    study_pixels = int(study_mask.sum())

    if study_pixels == 0:
        raise ValueError("Empty study-area mask")

    print("\nStudy-area cells:", study_pixels)

    # --------------------------------------------------------
    # 2. Build circular 300m neighbourhood
    # --------------------------------------------------------

    radius_cells = int(
        round(TPI_RADIUS_M / CELL_SIZE_M)
    )

    if not np.isclose(
        radius_cells * CELL_SIZE_M,
        TPI_RADIUS_M,
    ):
        raise ValueError(
            "TPI radius must align with grid resolution"
        )

    yy, xx = np.ogrid[
        -radius_cells:radius_cells + 1,
        -radius_cells:radius_cells + 1,
    ]

    footprint = (
        (xx * xx + yy * yy)
        <= radius_cells * radius_cells
    ).astype(np.float64)

    neighbour_count = int(footprint.sum())

    print("TPI radius:", TPI_RADIUS_M, "m")
    print("Neighbourhood cells:", neighbour_count)

    # --------------------------------------------------------
    # 3. Horn slope kernels
    # --------------------------------------------------------

    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=np.float64) / (8.0 * CELL_SIZE_M)

    kernel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ], dtype=np.float64) / (8.0 * CELL_SIZE_M)

    outputs = {}
    metrics = []

    # --------------------------------------------------------
    # 4. Validate and calculate both DEMs
    # --------------------------------------------------------

    for dem_name, dem_path in DEMS.items():

        print("\n" + "-" * 70)
        print("DEM:", dem_name)

        with rasterio.open(dem_path) as src:

            if (
                src.crs != expected_crs
                or src.transform != expected_transform
                or src.shape != expected_shape
            ):
                raise ValueError(
                    f"{dem_name}: grid alignment mismatch"
                )

            elevation = src.read(1).astype(np.float64)

            valid = (
                np.isfinite(elevation)
                & (elevation != src.nodata)
            )

            if not np.all(valid[study_mask]):
                raise ValueError(
                    f"{dem_name}: missing study-area elevation"
                )

            # Inspect the complete 3x3 slope neighbourhood.
            slope_support = ndimage.minimum_filter(
                valid.astype(np.uint8),
                size=3,
                mode="constant",
                cval=0,
            ) == 1

            # Inspect the circular TPI neighbourhood.
            support_counts = ndimage.convolve(
                valid.astype(np.float64),
                footprint,
                mode="constant",
                cval=0.0,
            )

            slope_valid = int(
                np.count_nonzero(
                    slope_support & study_mask
                )
            )

            tpi_valid = int(
                np.count_nonzero(
                    (support_counts == neighbour_count)
                    & study_mask
                )
            )

            print(
                "Cells with complete slope support:",
                slope_valid,
                "/", study_pixels
            )

            print(
                "Cells with complete TPI support:",
                tpi_valid,
                "/", study_pixels
            )

            if (
                slope_valid != study_pixels
                or tpi_valid != study_pixels
            ):
                raise ValueError(
                    f"{dem_name}: insufficient DEM support "
                    "in the terrain-analysis buffer. "
                    "Review missing edge/halo coverage "
                    "before generating features."
                )

            # Fill source NoData with zero exclusively
            # for numerical convolution.
            # The complete-support checks above prevent
            # this fill value from affecting study cells.
            filled = np.where(
                valid, elevation, 0.0
            )

            # Horn 3x3 slope.
            dzdx = ndimage.correlate(
                filled,
                kernel_x,
                mode="constant",
                cval=0.0,
            )

            dzdy = ndimage.correlate(
                filled,
                kernel_y,
                mode="constant",
                cval=0.0,
            )

            slope = np.degrees(
                np.arctan(
                    np.hypot(dzdx, dzdy)
                )
            )

            # TPI / local relative elevation.
            neighbourhood_sum = ndimage.convolve(
                filled,
                footprint,
                mode="constant",
                cval=0.0,
            )

            neighbourhood_mean = np.divide(
                neighbourhood_sum,
                support_counts,
                out=np.zeros_like(neighbourhood_sum),
                where=support_counts > 0,
            )

            tpi = elevation - neighbourhood_mean

            # Restrict exports to the study area.
            slope_output = np.full(
                expected_shape,
                NODATA,
                dtype=np.float32,
            )

            tpi_output = np.full(
                expected_shape,
                NODATA,
                dtype=np.float32,
            )

            slope_output[study_mask] = (
                slope[study_mask]
            )

            tpi_output[study_mask] = (
                tpi[study_mask]
            )

            slope_path = (
                DATA_DIR
                / f"07_{dem_name}_Slope_Degrees_30m.tif"
            )

            tpi_path = (
                DATA_DIR
                / f"07_{dem_name}_TPI_300m_30m.tif"
            )

            slope_stats = summarize(
                slope, study_mask
            )

            tpi_stats = summarize(
                tpi, study_mask
            )

            if (
                slope_stats["minimum"] < 0
                or slope_stats["maximum"] >= 90
            ):
                raise ValueError(
                    f"{dem_name}: implausible slope range"
                )

            outputs[dem_name] = {
                "slope_path": slope_path,
                "tpi_path": tpi_path,
                "slope_array": slope_output,
                "tpi_array": tpi_output,
                "source_path": dem_path,
                "statistics": {
                    "slope_degrees": slope_stats,
                    "tpi_metres": tpi_stats,
                },
            }

            for feature_name, stats in [
                ("slope_degrees", slope_stats),
                ("tpi_metres", tpi_stats),
            ]:
                metrics.append({
                    "dem": dem_name,
                    "feature": feature_name,
                    "valid_study_pixels": study_pixels,
                    "minimum": stats["minimum"],
                    "maximum": stats["maximum"],
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "p05": stats["p05"],
                    "p95": stats["p95"],
                })

            print(
                "Slope range:",
                slope_stats["minimum"],
                "to",
                slope_stats["maximum"],
                "degrees"
            )

            print(
                "TPI range:",
                tpi_stats["minimum"],
                "to",
                tpi_stats["maximum"],
                "m"
            )

    # --------------------------------------------------------
    # 5. Save features once both DEMs pass validation
    # --------------------------------------------------------

    output_records = []

    for dem_name, result in outputs.items():

        with rasterio.open(
            result["source_path"]
        ) as reference:

            write_feature(
                result["slope_path"],
                result["slope_array"],
                reference,
            )

            write_feature(
                result["tpi_path"],
                result["tpi_array"],
                reference,
            )

        output_records.append({
            "dem": dem_name,
            "slope_file": str(
                result["slope_path"].relative_to(ROOT)
            ),
            "tpi_file": str(
                result["tpi_path"].relative_to(ROOT)
            ),
            "statistics": result["statistics"],
        })

    metrics_path = (
        METRIC_DIR
        / "script07_terrain_feature_statistics.csv"
    )

    with metrics_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(metrics[0]),
        )
        writer.writeheader()
        writer.writerows(metrics)

    report_path = (
        REPORT_DIR
        / "script07_terrain_feature_validation.json"
    )

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "Terrain feature generation",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "analysis_crs": TARGET_CRS,
        "grid_resolution_m": CELL_SIZE_M,
        "study_pixels": study_pixels,
        "slope_method":
            "Horn 3x3 finite-difference gradient",
        "slope_units": "degrees",
        "tpi_method":
            "Cell elevation minus mean elevation "
            "within circular 300m neighbourhood",
        "tpi_radius_m": TPI_RADIUS_M,
        "tpi_neighbourhood_cells": neighbour_count,
        "tpi_units": "metres",
        "tpi_interpretation":
            "Negative values are locally lower "
            "than the surrounding neighbourhood. "
            "TPI does not by itself identify "
            "hydrologically closed depressions.",
        "full_neighbourhood_support":
            "PASS_FOR_ALL_STUDY_PIXELS",
        "outputs": output_records,
        "scientific_limitations": [
            "Slope and TPI depend on DEM quality "
            "and 30m spatial resolution.",
            "The 300m TPI radius is an initial "
            "analysis parameter requiring sensitivity "
            "assessment at alternative scales.",
            "Slope and TPI are not direct observations "
            "of flooding or seasonal waterlogging.",
            "DEM vertical accuracy and datum "
            "compatibility are not independently verified.",
            "The derived terrain factors do not resolve "
            "small drains, kerbs or road-level barriers.",
            "No susceptibility weights or waterlogging "
            "training labels are assigned by this script.",
        ],
        "raw_data_modified": False,
        "status": "TERRAIN_FEATURE_GENERATION_PASS",
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 07 - SUMMARY")
    print("=" * 70)

    print("Terrain factors generated: 4")
    print("DEMs processed: 2")
    print("Study-area cells:", study_pixels)

    for record in output_records:
        print("Saved:", ROOT / record["slope_file"])
        print("Saved:", ROOT / record["tpi_file"])

    print("Saved:", metrics_path)
    print("Saved:", report_path)

    print("\nTERRAIN FEATURES: PASS")
    print("DRAINAGE FEATURES: NOT GENERATED YET")
    print("SUSCEPTIBILITY MODEL: NOT STARTED")
    print("RAW SOURCE DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
