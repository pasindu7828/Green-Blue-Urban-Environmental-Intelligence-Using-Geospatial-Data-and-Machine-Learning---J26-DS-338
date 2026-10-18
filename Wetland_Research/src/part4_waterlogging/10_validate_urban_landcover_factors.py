
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling


# ============================================================
# PART 4 - SCRIPT 10
# URBAN AND LAND-COVER FACTOR VALIDATION
#
# - Align 2020 LULC to Part 4's 30m grid
# - Preserve missing / NoData classifications
# - Compare 2020 built-up with 2026 OSM indicators
# - Check building footprint vs building density
# - Save permanent comparison evidence
#
# No susceptibility weights or predictions.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed/part4_waterlogging"
METRICS = ROOT / "outputs/metrics/part4_waterlogging"
REPORTS = ROOT / "outputs/reports/part4_waterlogging"

LULC_PATH = (
    ROOT
    / "data/processed/part3_future_projection"
    / "LULC_2020_6Class_CommonMask.tif"
)

MASK_PATH = DATA / "06_Kaduwela_Study_Mask_30m.tif"

DENSITY_PATH = (
    DATA / "09_OSM_Building_Density_300m_30m.tif"
)

FOOTPRINT_PATH = (
    DATA / "09_OSM_Building_Footprint_Percent_30m.tif"
)

CLASSES = {
    1: "Wetland_Marsh",
    2: "Open_Water",
    3: "Built_Up",
    4: "Paddy_Agriculture",
    5: "Other_Vegetation",
    6: "Bare_Fill_Disturbed",
}


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None

    return round(float(np.corrcoef(a, b)[0, 1]), 6)


def main():

    print("=" * 70)
    print("PART 4 - URBAN / LAND-COVER FACTOR VALIDATION")
    print("SCRIPT 10")
    print("=" * 70)

    METRICS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    with rasterio.open(MASK_PATH) as mask_src, \
         rasterio.open(DENSITY_PATH) as den_src, \
         rasterio.open(FOOTPRINT_PATH) as foot_src, \
         rasterio.open(LULC_PATH) as lulc_src:

        for src in (den_src, foot_src):
            if (
                src.crs != mask_src.crs
                or src.transform != mask_src.transform
                or src.shape != mask_src.shape
            ):
                raise ValueError("Urban grid alignment mismatch")

        study = mask_src.read(1) == 1
        density = den_src.read(1)
        footprint = foot_src.read(1)

        if int(study.sum()) != 94431:
            raise ValueError("Unexpected study-area cell count")

        urban_valid = (
            study
            & np.isfinite(density)
            & np.isfinite(footprint)
            & (density != den_src.nodata)
            & (footprint != foot_src.nodata)
        )

        if not np.all(urban_valid[study]):
            raise ValueError(
                "Incomplete urban coverage in study area"
            )

        if (
            np.any(density[study] < 0)
            or np.any(density[study] > 100)
            or np.any(footprint[study] < 0)
            or np.any(footprint[study] > 100)
        ):
            raise ValueError("Invalid urban percentage values")

        # Align categorical 2020 land cover without
        # introducing interpolated class values.
        lulc = np.zeros(
            mask_src.shape,
            dtype=np.uint8,
        )

        reproject(
            source=rasterio.band(lulc_src, 1),
            destination=lulc,
            src_transform=lulc_src.transform,
            src_crs=lulc_src.crs,
            src_nodata=lulc_src.nodata,
            dst_transform=mask_src.transform,
            dst_crs=mask_src.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

        # Zero means missing or outside the study area.
        lulc[~study] = 0

        valid = study & np.isin(
            lulc, list(CLASSES.keys())
        )

        unexpected = (
            study
            & (lulc != 0)
            & ~np.isin(lulc, list(CLASSES.keys()))
        )

        if np.any(unexpected):
            raise ValueError("Unexpected 2020 LULC classes")

        valid_count = int(valid.sum())
        missing_count = int(study.sum() - valid_count)

        if valid_count == 0:
            raise ValueError("No valid LULC comparisons")

        builtup = (lulc[valid] == 3).astype(float)

        density_valid = density[valid].astype(float)
        footprint_valid = footprint[valid].astype(float)

        comparisons = {
            "builtup_vs_building_density": pearson(
                builtup, density_valid
            ),
            "builtup_vs_building_footprint": pearson(
                builtup, footprint_valid
            ),
            "building_density_vs_footprint": pearson(
                density[study], footprint[study]
            ),
        }

        rows = []

        for class_id, class_name in CLASSES.items():
            selected = valid & (lulc == class_id)
            count = int(selected.sum())

            if count == 0:
                raise ValueError(
                    f"Missing land-cover class: {class_name}"
                )

            rows.append({
                "class_id": class_id,
                "class_name": class_name,
                "valid_cells": count,
                "percent_valid_lulc":
                    round(100 * count / valid_count, 4),
                "mean_2026_building_density_percent":
                    round(float(density[selected].mean()), 4),
                "mean_2026_footprint_percent":
                    round(float(footprint[selected].mean()), 4),
            })

        # Save the aligned categorical LULC raster.
        aligned_path = (
            DATA / "10_LULC_2020_Aligned_30m.tif"
        )

        profile = mask_src.profile.copy()
        profile.update(
            dtype="uint8",
            nodata=0,
            count=1,
            compress="deflate",
        )

        with rasterio.open(aligned_path, "w", **profile) as dst:
            dst.write(lulc, 1)

    metrics_path = (
        METRICS
        / "script10_lulc_urban_comparison.csv"
    )

    pd.DataFrame(rows).to_csv(
        metrics_path,
        index=False,
    )

    report_path = (
        REPORTS
        / "script10_urban_landcover_validation.json"
    )

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "Urban and land-cover factor validation",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "lulc_reference_year": 2020,
        "osm_reference_year": 2026,
        "study_cells": int(study.sum()),
        "valid_lulc_cells": valid_count,
        "missing_lulc_cells": missing_count,
        "lulc_coverage_percent":
            round(100 * valid_count / int(study.sum()), 4),
        "correlations": comparisons,
        "aligned_lulc_file":
            str(aligned_path.relative_to(ROOT)),
        "comparison_csv":
            str(metrics_path.relative_to(ROOT)),
        "interpretation": [
            "The 2020 built-up class is categorical land cover.",
            "2026 OSM footprint and neighbourhood density "
            "represent mapped building development.",
            "Correlations describe spatial association, "
            "not causation or source accuracy.",
            "Missing 2020 LULC cells remain NoData (0).",
            "Spatial autocorrelation limits interpretation "
            "of cell-level statistical independence.",
            "Urban indicators are candidates only. "
            "No independent susceptibility weights "
            "or risk directions have been assigned.",
        ],
        "raw_data_modified": False,
        "status": "URBAN_LULC_COMPARISON_COMPLETE",
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nValid LULC cells:", valid_count)
    print("Missing LULC cells:", missing_count)
    print(
        "LULC coverage:",
        report["lulc_coverage_percent"], "%"
    )

    print("\nCORRELATIONS")
    for name, value in comparisons.items():
        print(name, ":", value)

    print("\nSaved:", aligned_path)
    print("Saved:", metrics_path)
    print("Saved:", report_path)

    print("\nSCRIPT 10: COMPLETE")
    print("SUSCEPTIBILITY WEIGHTS: NOT ASSIGNED")
    print("RAW DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
