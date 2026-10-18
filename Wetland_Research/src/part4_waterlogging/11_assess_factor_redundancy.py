
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PART 4 - SCRIPT 11
# CROSS-FACTOR REDUNDANCY ASSESSMENT
#
# Compare candidate factors on identical study cells.
# Identify strong linear associations.
# No weighting or susceptibility predictions.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed/part4_waterlogging"
METRICS = ROOT / "outputs/metrics/part4_waterlogging"
REPORTS = ROOT / "outputs/reports/part4_waterlogging"

FEATURES = {
    "Elevation": "06_GEDTM30_UTM44N_30m.tif",
    "Slope": "07_GEDTM30_Slope_Degrees_30m.tif",
    "TPI_300m": "07_GEDTM30_TPI_300m_30m.tif",
    "All_Waterways_Distance":
        "08_OSM_All_Waterways_Distance_30m.tif",
    "River_Stream_Canal_Distance":
        "08_OSM_River_Stream_Canal_Distance_30m.tif",
    "Ditch_Drain_Distance":
        "08_OSM_Ditch_Drain_Distance_30m.tif",
    "Building_Density_300m":
        "09_OSM_Building_Density_300m_30m.tif",
    "Building_Footprint":
        "09_OSM_Building_Footprint_Percent_30m.tif",
}

THRESHOLD = 0.70


def main():
    print("=" * 70)
    print("PART 4 - CROSS-FACTOR REDUNDANCY")
    print("SCRIPT 11")
    print("=" * 70)

    METRICS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    mask_path = DATA / "06_Kaduwela_Study_Mask_30m.tif"

    with rasterio.open(mask_path) as ref:
        study = ref.read(1) == 1
        crs = ref.crs
        transform = ref.transform
        shape = ref.shape

    if int(study.sum()) != 94431:
        raise ValueError("Unexpected study-area cell count")

    feature_values = {}

    for name, filename in FEATURES.items():
        path = DATA / filename

        with rasterio.open(path) as src:
            if (
                src.crs != crs
                or src.transform != transform
                or src.shape != shape
            ):
                raise ValueError(
                    f"Grid mismatch: {name}"
                )

            raster = src.read(1)
            values = raster[study].astype(np.float64)

            if (
                not np.isfinite(values).all()
                or (
                    src.nodata is not None
                    and np.any(values == src.nodata)
                )
            ):
                raise ValueError(
                    f"Missing or invalid values: {name}"
                )

            feature_values[name] = values

    frame = pd.DataFrame(feature_values)

    if frame.isna().any().any():
        raise ValueError("Missing feature observations")

    correlations = frame.corr(method="pearson")

    if correlations.isna().any().any():
        raise ValueError(
            "Undefined correlations: check constant features"
        )

    correlation_path = (
        METRICS / "script11_factor_correlation_matrix.csv"
    )

    correlations.to_csv(correlation_path)

    flagged = []

    names = list(FEATURES)

    print("\nSTRONGLY CORRELATED FACTOR PAIRS")
    print(f"Screening threshold: |r| >= {THRESHOLD}")

    for i, first in enumerate(names):
        for second in names[i + 1:]:
            r = float(correlations.loc[first, second])

            if abs(r) >= THRESHOLD:
                record = {
                    "factor_1": first,
                    "factor_2": second,
                    "pearson_r": round(r, 6),
                }

                flagged.append(record)

                print(
                    first, "<->", second,
                    "| r =", round(r, 4)
                )

    if not flagged:
        print("No pairs exceeded the threshold.")

    # The all-waterways distance is the minimum
    # distance to either waterway subgroup.
    all_distances = frame[
        "All_Waterways_Distance"
    ].to_numpy()

    subgroup_min = np.minimum(
        frame["River_Stream_Canal_Distance"].to_numpy(),
        frame["Ditch_Drain_Distance"].to_numpy(),
    )

    max_subgroup_difference = float(
        np.max(np.abs(all_distances - subgroup_min))
    )

    print("\nDRAINAGE FEATURE RELATIONSHIP")
    print(
        "Maximum difference between all-waterways "
        "distance and minimum subgroup distance:",
        round(max_subgroup_difference, 6), "m"
    )

    if max_subgroup_difference > 0.01:
        raise ValueError(
            "Unexpected drainage distance relationship"
        )

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "Cross-factor redundancy assessment",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "study_cells": int(study.sum()),
        "factors_checked": names,
        "method": "Pearson correlation",
        "screening_threshold_absolute_r": THRESHOLD,
        "flagged_factor_pairs": flagged,
        "all_waterways_subgroup_identity_max_error_m":
            round(max_subgroup_difference, 6),
        "interpretation": [
            "This is a screening threshold, not a "
            "statistical significance decision.",
            "Strong factor associations can cause "
            "double-counting in weighted models.",
            "Low correlation does not prove independence.",
            "Raster cells have spatial autocorrelation.",
            "Alternative NASADEM terrain factors are "
            "reserved for sensitivity analysis.",
            "No susceptibility weights are assigned.",
        ],
        "status": "FACTOR_REDUNDANCY_AUDIT_COMPLETE",
        "raw_data_modified": False,
    }

    report_path = (
        REPORTS / "script11_factor_redundancy_audit.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 11 - SUMMARY")
    print("=" * 70)

    print("Cells compared:", len(frame))
    print("Candidate factors checked:", len(FEATURES))
    print("Flagged factor pairs:", len(flagged))

    print("\nSaved:", correlation_path)
    print("Saved:", report_path)

    print("\nFACTOR AUDIT: COMPLETE")
    print("FINAL FACTOR SELECTION: PENDING")
    print("SUSCEPTIBILITY MODEL: NOT STARTED")
    print("RAW DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
