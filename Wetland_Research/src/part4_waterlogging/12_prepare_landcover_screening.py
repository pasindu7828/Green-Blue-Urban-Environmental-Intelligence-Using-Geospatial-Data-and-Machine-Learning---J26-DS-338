
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PART 4 - SCRIPT 12
# LAND-COVER SCREENING FOR SUSCEPTIBILITY ASSESSMENT
#
# This script categorizes 2020 land cover for
# interpretation and reporting.
#
# It does NOT:
# - assign susceptibility or risk scores
# - create waterlogging training labels
# - modify Part 3 data
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed/part4_waterlogging"
METRICS = ROOT / "outputs/metrics/part4_waterlogging"
REPORTS = ROOT / "outputs/reports/part4_waterlogging"

MASK_PATH = DATA / "06_Kaduwela_Study_Mask_30m.tif"
LULC_PATH = DATA / "10_LULC_2020_Aligned_30m.tif"

OUTPUT_PATH = DATA / "12_Landcover_Screening_30m.tif"

GROUPS = {
    1: "General_land_susceptibility",
    2: "Paddy_agricultural_context",
    3: "Wetland_marsh_context",
    4: "Open_water_context",
    5: "Unknown_landcover",
}

# Original 2020 LULC -> interpretation category.
# 1 Wetland, 2 Water, 3 Built-up,
# 4 Paddy, 5 Vegetation, 6 Bare/Disturbed.
CROSSWALK = {
    0: 5,
    1: 3,
    2: 4,
    3: 1,
    4: 2,
    5: 1,
    6: 1,
}


def main():

    print("=" * 70)
    print("PART 4 - LAND-COVER SCREENING")
    print("SCRIPT 12")
    print("=" * 70)

    METRICS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    with rasterio.open(MASK_PATH) as mask_src, \
         rasterio.open(LULC_PATH) as lulc_src:

        if (
            mask_src.crs != lulc_src.crs
            or mask_src.transform != lulc_src.transform
            or mask_src.shape != lulc_src.shape
        ):
            raise ValueError("Raster alignment mismatch")

        study = mask_src.read(1) == 1
        lulc = lulc_src.read(1)

        if int(study.sum()) != 94431:
            raise ValueError("Unexpected study-area cell count")

        if not np.isin(
            lulc[study],
            list(CROSSWALK),
        ).all():
            raise ValueError("Unexpected LULC category")

        # Outside Kaduwela = 0.
        # Within Kaduwela = screening category 1-5.
        screening = np.zeros(
            mask_src.shape,
            dtype=np.uint8,
        )

        for source_class, group_code in CROSSWALK.items():
            screening[
                study & (lulc == source_class)
            ] = group_code

        if np.any(screening[study] == 0):
            raise ValueError("Unclassified study-area cells")

        rows = []

        for code, label in GROUPS.items():

            count = int(
                np.count_nonzero(
                    screening[study] == code
                )
            )

            rows.append({
                "screening_code": code,
                "screening_category": label,
                "study_cells": count,
                "percent_of_study":
                    round(100 * count / study.sum(), 4),
            })

        if sum(r["study_cells"] for r in rows) != study.sum():
            raise ValueError("Screening counts inconsistent")

        profile = mask_src.profile.copy()

        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=0,
            compress="deflate",
        )

        with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
            dst.write(screening, 1)

    metrics_path = (
        METRICS / "script12_landcover_screening_counts.csv"
    )

    pd.DataFrame(rows).to_csv(
        metrics_path,
        index=False,
    )

    report_path = (
        REPORTS / "script12_landcover_screening.json"
    )

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "Land-cover interpretation screening",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "lulc_reference_year": 2020,
        "study_cells": int(study.sum()),
        "categories": rows,
        "crosswalk": CROSSWALK,
        "screening_raster":
            str(OUTPUT_PATH.relative_to(ROOT)),
        "interpretation_policy": [
            "General land is eligible for primary "
            "land-waterlogging susceptibility reporting.",
            "Paddy requires separate interpretation "
            "because standing water may be intentional.",
            "Wetland/marsh remains a distinct ecological "
            "context, not a confirmed waterlogging label.",
            "Open water is not treated as ordinary "
            "land-waterlogging exposure.",
            "Missing LULC remains unknown.",
            "Terrain susceptibility may be calculated "
            "across all valid study-area cells.",
            "Screening codes are not susceptibility "
            "classes or prediction labels.",
            "The 2020 LULC classification is not "
            "assumed to represent exact 2018 conditions.",
        ],
        "raw_data_modified": False,
        "status": "SCREENING_LAYER_PREPARED",
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSCREENING DISTRIBUTION")

    for row in rows:
        print(
            row["screening_category"],
            "| Cells:", row["study_cells"],
            "| Share:", row["percent_of_study"], "%"
        )

    print("\nSaved:", OUTPUT_PATH)
    print("Saved:", metrics_path)
    print("Saved:", report_path)

    print("\nSCREENING: COMPLETE")
    print("SUSCEPTIBILITY MODEL: NOT STARTED")
    print("RAW DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
