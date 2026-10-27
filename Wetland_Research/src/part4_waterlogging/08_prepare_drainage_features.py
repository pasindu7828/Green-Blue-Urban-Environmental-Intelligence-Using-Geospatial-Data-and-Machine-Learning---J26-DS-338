
from pathlib import Path
from datetime import datetime, timezone
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely

from shapely import STRtree
from shapely.ops import unary_union


# ============================================================
# PART 4 - SEASONAL WATERLOGGING SUSCEPTIBILITY
# SCRIPT 08 - OSM DRAINAGE DISTANCE FEATURES
#
# 1. Validate the 2026 OSM waterway network
# 2. Report feature counts and channel lengths
# 3. Calculate exact vector-line distances at grid centres
# 4. Export three aligned 30m distance rasters
# 5. Save reproducible audit reports
#
# Distances are explanatory spatial features only.
# No waterlogging risk direction or weight is assigned.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

OSM_FILE = (
    ROOT / "data/raw/osm/"
    "Kaduwela_OSM_Water_Drainage_2026.gpkg"
)

BOUNDARY_FILE = (
    ROOT / "data/raw/boundaries/Kaduwela TO Area.kml"
)

DATA_DIR = ROOT / "data/processed/part4_waterlogging"
METRIC_DIR = ROOT / "outputs/metrics/part4_waterlogging"
REPORT_DIR = ROOT / "outputs/reports/part4_waterlogging"

MASK_FILE = DATA_DIR / "06_Kaduwela_Study_Mask_30m.tif"

TARGET_CRS = "EPSG:32644"
NODATA = -9999.0

WATERWAY_TYPES = {
    "river",
    "stream",
    "canal",
    "ditch",
    "drain",
}

GROUPS = {
    "All_Waterways": WATERWAY_TYPES,
    "River_Stream_Canal": {
        "river", "stream", "canal"
    },
    "Ditch_Drain": {
        "ditch", "drain"
    },
}


def save_raster(path, values, reference):
    profile = reference.profile.copy()

    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=3,
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype(np.float32), 1)


def main():

    print("=" * 70)
    print("PART 4 - OSM DRAINAGE DISTANCE FEATURES")
    print("SCRIPT 08")
    print("=" * 70)

    for folder in (DATA_DIR, METRIC_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Read and validate OSM waterways
    # --------------------------------------------------------

    osm = gpd.read_file(OSM_FILE)

    if osm.crs is None:
        raise ValueError("OSM CRS is missing")

    if "waterway" not in osm.columns:
        raise ValueError("Missing waterway classification")

    if osm.geometry.isna().any():
        raise ValueError("Null OSM geometries found")

    if osm.geometry.is_empty.any():
        raise ValueError("Empty OSM geometries found")

    if not osm.geometry.is_valid.all():
        raise ValueError("Invalid OSM geometries found")

    if not osm.geometry.geom_type.isin([
        "LineString", "MultiLineString"
    ]).all():
        raise ValueError("Unexpected OSM geometry type")

    osm["waterway"] = (
        osm["waterway"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    unknown = (
        set(osm["waterway"].dropna())
        - WATERWAY_TYPES
    )

    if unknown or osm["waterway"].isna().any():
        raise ValueError(
            f"Missing or unexpected waterway tags: {unknown}"
        )

    osm = osm.to_crs(TARGET_CRS)

    print("\nOSM features:", len(osm))
    print("CRS:", osm.crs)

    # --------------------------------------------------------
    # 2. Load the 56-GN study boundary
    # --------------------------------------------------------

    gn = gpd.read_file(
        BOUNDARY_FILE,
        layer="GN Boundary Colombo District (#2)",
    )

    if len(gn) != 56:
        raise ValueError("Unexpected GN feature count")

    gn = gn.to_crs(TARGET_CRS)

    if (
        gn.geometry.isna().any()
        or not gn.geometry.is_valid.all()
    ):
        raise ValueError("Unusable GN geometry")

    study_area = unary_union(gn.geometry)

    # --------------------------------------------------------
    # 3. Audit each waterway classification
    # --------------------------------------------------------

    class_rows = []

    for category in sorted(WATERWAY_TYPES):

        subset = osm[
            osm["waterway"] == category
        ]

        total_length_m = float(
            subset.geometry.length.sum()
        )

        inside_lengths = [
            geometry.intersection(study_area).length
            for geometry in subset.geometry
        ]

        intersecting = sum(
            length > 0 for length in inside_lengths
        )

        class_rows.append({
            "waterway": category,
            "feature_count": len(subset),
            "features_intersecting_study_area":
                intersecting,
            "total_mapped_length_km":
                round(total_length_m / 1000, 4),
            "length_inside_study_area_km":
                round(sum(inside_lengths) / 1000, 4),
        })

    print("\nWATERWAY NETWORK INVENTORY")

    for item in class_rows:
        print(
            f"{item['waterway']}: "
            f"{item['feature_count']} features | "
            f"{item['length_inside_study_area_km']} km "
            "inside GNs"
        )

    # --------------------------------------------------------
    # 4. Read the existing common 30m analysis grid
    # --------------------------------------------------------

    with rasterio.open(MASK_FILE) as mask_src:

        if str(mask_src.crs) != TARGET_CRS:
            raise ValueError("Study-mask CRS mismatch")

        transform = mask_src.transform
        shape = mask_src.shape

        if (
            abs(transform.a - 30) > 1e-6
            or abs(transform.e + 30) > 1e-6
            or transform.b != 0
            or transform.d != 0
        ):
            raise ValueError(
                "Unexpected analysis-grid orientation "
                "or resolution"
            )

        mask_values = mask_src.read(1)

        if not np.isin(
            mask_values, [0, 1]
        ).all():
            raise ValueError("Unexpected mask values")

        inside = mask_values == 1

        if int(inside.sum()) != 94431:
            raise ValueError(
                "Study mask differs from Script 06"
            )

        # Centre coordinates for study-area cells.
        rows, cols = np.where(inside)

        xs = (
            transform.c
            + (cols + 0.5) * transform.a
        )

        ys = (
            transform.f
            + (rows + 0.5) * transform.e
        )

        sample_points = shapely.points(xs, ys)

        output_arrays = {}
        group_audits = []

        # ----------------------------------------------------
        # 5. Calculate distances to vector waterways
        # ----------------------------------------------------

        for group_name, categories in GROUPS.items():

            subset = osm[
                osm["waterway"].isin(categories)
            ]

            if subset.empty:
                raise ValueError(
                    f"No waterways in {group_name}"
                )

            geometries = subset.geometry.to_numpy()

            tree = STRtree(geometries)

            nearest_indices = tree.nearest(
                sample_points
            )

            nearest_geometries = geometries[
                nearest_indices
            ]

            distances = shapely.distance(
                sample_points,
                nearest_geometries,
            )

            distances = np.asarray(
                distances, dtype=np.float64
            )

            if (
                not np.isfinite(distances).all()
                or np.any(distances < 0)
            ):
                raise ValueError(
                    f"Invalid distances in {group_name}"
                )

            raster_values = np.full(
                shape,
                NODATA,
                dtype=np.float32,
            )

            raster_values[rows, cols] = (
                distances.astype(np.float32)
            )

            output_arrays[group_name] = raster_values

            record = {
                "group": group_name,
                "feature_count": len(subset),
                "valid_study_pixels": len(distances),
                "minimum_distance_m":
                    round(float(distances.min()), 4),
                "median_distance_m":
                    round(float(np.median(distances)), 4),
                "mean_distance_m":
                    round(float(distances.mean()), 4),
                "p95_distance_m":
                    round(float(np.percentile(
                        distances, 95
                    )), 4),
                "maximum_distance_m":
                    round(float(distances.max()), 4),
                "cells_within_100m_percent":
                    round(float(
                        100 * np.mean(distances <= 100)
                    ), 4),
                "cells_within_300m_percent":
                    round(float(
                        100 * np.mean(distances <= 300)
                    ), 4),
            }

            group_audits.append(record)

            print("\nDISTANCE GROUP:", group_name)
            print("Features:", len(subset))
            print(
                "Median distance:",
                record["median_distance_m"], "m"
            )
            print(
                "Maximum distance:",
                record["maximum_distance_m"], "m"
            )

        # ----------------------------------------------------
        # 6. Save rasters after all groups pass
        # ----------------------------------------------------

        output_files = {}

        for group_name, array in output_arrays.items():

            path = (
                DATA_DIR
                / f"08_OSM_{group_name}_Distance_30m.tif"
            )

            save_raster(path, array, mask_src)

            output_files[group_name] = str(
                path.relative_to(ROOT)
            )

    # --------------------------------------------------------
    # 7. Save permanent audit reports
    # --------------------------------------------------------

    inventory_path = (
        METRIC_DIR
        / "script08_osm_waterway_inventory.csv"
    )

    distance_path = (
        METRIC_DIR
        / "script08_drainage_distance_statistics.csv"
    )

    report_path = (
        REPORT_DIR
        / "script08_drainage_feature_validation.json"
    )

    pd.DataFrame(class_rows).to_csv(
        inventory_path, index=False
    )

    pd.DataFrame(group_audits).to_csv(
        distance_path, index=False
    )

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "OSM drainage distance preparation",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "source": str(OSM_FILE.relative_to(ROOT)),
        "source_mapping_year": 2026,
        "analysis_crs": TARGET_CRS,
        "grid_resolution_m": 30,
        "study_pixels": int(inside.sum()),
        "method":
            "Euclidean distance from each study-area "
            "30m cell centre to nearest OSM vector "
            "waterway line using a spatial index.",
        "waterway_inventory": class_rows,
        "distance_statistics": group_audits,
        "output_rasters": output_files,
        "scientific_limitations": [
            "OSM network completeness and drainage "
            "accuracy are not independently verified.",
            "Mapped OSM waterways may omit small "
            "urban drains and underground connections.",
            "The 2026 OSM network does not establish "
            "the exact drainage network in 2018.",
            "Distance is not drainage capacity, "
            "flow direction or overflow probability.",
            "Proximity to waterways may represent "
            "different physical processes depending "
            "on the waterway type and local conditions.",
            "Distances are measured to mapped vector "
            "lines at 30m cell centres.",
            "No susceptibility weights or risk "
            "directions are assigned.",
        ],
        "raw_data_modified": False,
        "status": "DRAINAGE_FEATURE_PREPARATION_PASS",
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 08 - SUMMARY")
    print("=" * 70)

    print("OSM waterway features:", len(osm))
    print("GN divisions:", len(gn))
    print("Distance rasters generated:", len(output_files))
    print("Study-area pixels:", int(inside.sum()))

    for path in output_files.values():
        print("Saved:", ROOT / path)

    print("Saved:", inventory_path)
    print("Saved:", distance_path)
    print("Saved:", report_path)

    print("\nDRAINAGE FEATURES: PASS")
    print("SUSCEPTIBILITY WEIGHTS: NOT ASSIGNED")
    print("SUSCEPTIBILITY MODEL: NOT STARTED")
    print("RAW SOURCE DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
