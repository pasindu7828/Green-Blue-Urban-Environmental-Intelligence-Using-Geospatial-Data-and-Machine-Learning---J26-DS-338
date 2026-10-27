
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import math

import geopandas as gpd
import numpy as np
import rasterio

from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from shapely.geometry import mapping
from shapely.ops import unary_union


# ============================================================
# PART 4 - SEASONAL WATERLOGGING SUSCEPTIBILITY
# SCRIPT 06 - COMMON TERRAIN ANALYSIS GRID
#
# - Establish a shared 30 m UTM analysis grid
# - Reproject GEDTM30 and NASADEM
# - Create the 56-GN study-area mask
# - Validate aligned DEM coverage
# - Save permanent GIS outputs and quality reports
#
# No slope, susceptibility scores, or labels yet.
# Original datasets are not modified.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DEM_DIR = ROOT / "data/raw/dem"

BOUNDARY = (
    ROOT / "data/raw/boundaries/Kaduwela TO Area.kml"
)

OUT_DIR = (
    ROOT / "data/processed/part4_waterlogging"
)

REPORT_DIR = (
    ROOT / "outputs/reports/part4_waterlogging"
)

METRIC_DIR = (
    ROOT / "outputs/metrics/part4_waterlogging"
)

TARGET_CRS = "EPSG:32644"
RESOLUTION_M = 30
BUFFER_M = 300
NODATA = -9999.0

DEM_FILES = {
    "GEDTM30": "GEDTM30_Kaduwela_Buffer.tif",
    "NASADEM": "NASADEM_Kaduwela_Buffer.tif",
}


def write_raster(path, array, transform, crs, nodata):
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": str(array.dtype),
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def main():

    print("=" * 70)
    print("PART 4 - COMMON TERRAIN ANALYSIS GRID")
    print("SCRIPT 06")
    print("=" * 70)

    for folder in (OUT_DIR, REPORT_DIR, METRIC_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load study area
    # --------------------------------------------------------

    gn = gpd.read_file(
        BOUNDARY,
        layer="GN Boundary Colombo District (#2)",
    )

    if len(gn) != 56:
        raise ValueError(
            f"Expected 56 GN polygons, found {len(gn)}"
        )

    if gn.crs is None:
        raise ValueError("Missing GN boundary CRS")

    gn = gn.to_crs(TARGET_CRS)

    if gn.geometry.isna().any():
        raise ValueError("Missing GN geometry")

    if not gn.geometry.is_valid.all():
        raise ValueError(
            "Invalid GN geometries need review"
        )

    study_area = unary_union(gn.geometry)

    # --------------------------------------------------------
    # 2. Create a snapped 30 m projected grid
    # --------------------------------------------------------

    minx, miny, maxx, maxy = study_area.bounds

    left = (
        math.floor(
            (minx - BUFFER_M) / RESOLUTION_M
        ) * RESOLUTION_M
    )

    bottom = (
        math.floor(
            (miny - BUFFER_M) / RESOLUTION_M
        ) * RESOLUTION_M
    )

    right = (
        math.ceil(
            (maxx + BUFFER_M) / RESOLUTION_M
        ) * RESOLUTION_M
    )

    top = (
        math.ceil(
            (maxy + BUFFER_M) / RESOLUTION_M
        ) * RESOLUTION_M
    )

    width = int(
        round((right - left) / RESOLUTION_M)
    )

    height = int(
        round((top - bottom) / RESOLUTION_M)
    )

    transform = from_origin(
        left,
        top,
        RESOLUTION_M,
        RESOLUTION_M,
    )

    grid_shape = (height, width)

    print("\nANALYSIS GRID")
    print("CRS:", TARGET_CRS)
    print("Resolution:", RESOLUTION_M, "m")
    print("Width:", width)
    print("Height:", height)
    print("Study-area GN divisions:", len(gn))

    # --------------------------------------------------------
    # 3. Rasterize the unified study area
    # --------------------------------------------------------

    inside_mask = geometry_mask(
        [mapping(study_area)],
        out_shape=grid_shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )

    if not inside_mask.any():
        raise ValueError("Empty study-area raster mask")

    mask_path = (
        OUT_DIR / "06_Kaduwela_Study_Mask_30m.tif"
    )

    write_raster(
        mask_path,
        inside_mask.astype(np.uint8),
        transform,
        TARGET_CRS,
        0,
    )

    print("\nStudy-area pixels:", int(inside_mask.sum()))

    # --------------------------------------------------------
    # 4. Align both DEM datasets
    # --------------------------------------------------------

    aligned = {}
    dem_audits = []

    for dem_name, filename in DEM_FILES.items():

        path = DEM_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        destination = np.full(
            grid_shape,
            NODATA,
            dtype=np.float32,
        )

        with rasterio.open(path) as src:

            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                dst_nodata=NODATA,
                resampling=Resampling.bilinear,
            )

            valid = (
                np.isfinite(destination)
                & (destination != NODATA)
            )

            study_valid = valid & inside_mask

            total_study = int(inside_mask.sum())
            valid_study = int(study_valid.sum())

            coverage = (
                100 * valid_study / total_study
            )

            if valid_study == 0:
                raise ValueError(
                    f"{dem_name}: no valid study-area cells"
                )

            output_path = (
                OUT_DIR
                / f"06_{dem_name}_UTM44N_30m.tif"
            )

            write_raster(
                output_path,
                destination,
                transform,
                TARGET_CRS,
                NODATA,
            )

            values = destination[study_valid]

            audit = {
                "dem": dem_name,
                "source_file": str(path.relative_to(ROOT)),
                "aligned_file":
                    str(output_path.relative_to(ROOT)),
                "study_pixels": total_study,
                "valid_study_pixels": valid_study,
                "study_coverage_percent":
                    round(coverage, 5),
                "min_elevation_m":
                    float(values.min()),
                "max_elevation_m":
                    float(values.max()),
                "mean_elevation_m":
                    float(values.mean()),
                "negative_study_pixels":
                    int(np.sum(values < 0)),
                "resampling": "bilinear",
            }

            dem_audits.append(audit)
            aligned[dem_name] = destination

            print("\nDEM:", dem_name)
            print(
                "Study-area coverage:",
                round(coverage, 3), "%"
            )
            print(
                "Elevation range:",
                round(audit["min_elevation_m"], 3),
                "to",
                round(audit["max_elevation_m"], 3),
                "m"
            )

    # --------------------------------------------------------
    # 5. Confirm all 56 GNs have terrain coverage
    # --------------------------------------------------------

    gn_rows = []

    for index, geom in enumerate(
        gn.geometry, start=1
    ):

        gn_mask = geometry_mask(
            [mapping(geom)],
            out_shape=grid_shape,
            transform=transform,
            invert=True,
            all_touched=False,
        )

        gn_mask &= inside_mask

        total = int(gn_mask.sum())

        if total == 0:
            raise ValueError(
                f"GN feature {index} has zero sampled pixels"
            )

        row = {
            "gn_feature_index": index,
            "gn_sampled_pixels": total,
        }

        for dem_name, array in aligned.items():

            valid = (
                gn_mask
                & np.isfinite(array)
                & (array != NODATA)
            )

            count = int(valid.sum())

            coverage = 100 * count / total

            row[f"{dem_name}_valid_pixels"] = count

            row[f"{dem_name}_coverage_percent"] = (
                round(coverage, 5)
            )

            if coverage < 99.9:
                raise ValueError(
                    f"{dem_name}: insufficient coverage "
                    f"for GN feature {index}: "
                    f"{coverage:.2f}%"
                )

        gn_rows.append(row)

    # --------------------------------------------------------
    # 6. Save permanent validation evidence
    # --------------------------------------------------------

    gn_csv = (
        METRIC_DIR / "script06_dem_gn_coverage.csv"
    )

    with gn_csv.open(
        "w", newline="", encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(gn_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(gn_rows)

    report_path = (
        REPORT_DIR
        / "script06_terrain_grid_validation.json"
    )

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "Common projected terrain grid",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "analysis_crs": TARGET_CRS,
        "pixel_resolution_m": RESOLUTION_M,
        "grid_buffer_m": BUFFER_M,
        "grid_width": width,
        "grid_height": height,
        "grid_bounds_m": [
            left, bottom, right, top
        ],
        "study_mask": str(
            mask_path.relative_to(ROOT)
        ),
        "gn_features": len(gn),
        "dem_audits": dem_audits,
        "gn_coverage_check": "PASS",
        "scientific_limitations": [
            "Bilinear interpolation changes source "
            "DEM pixel values during reprojection.",
            "GEDTM30 and NASADEM are not assumed "
            "to have identical vertical references.",
            "Elevation agreement does not establish "
            "absolute vertical accuracy.",
            "The 30m DEM does not resolve narrow "
            "drains or road-level elevation changes.",
            "DEM coverage in the 300m analysis buffer "
            "is not guaranteed by GN coverage checks.",
            "Grid output is terrain preparation only; "
            "no susceptibility prediction is made.",
        ],
        "raw_data_modified": False,
        "status": "PROJECTED_GRID_PREPARATION_PASS",
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 06 - SUMMARY")
    print("=" * 70)

    print("GNs validated:", len(gn_rows))
    print("Aligned DEMs:", len(dem_audits))

    print("\nSaved:", mask_path)

    for audit in dem_audits:
        print(
            "Saved:",
            ROOT / audit["aligned_file"]
        )

    print("Saved:", gn_csv)
    print("Saved:", report_path)

    print("\nTERRAIN GRID: PASS")
    print("SLOPE / RELATIVE ELEVATION: NOT GENERATED YET")
    print("SUSCEPTIBILITY MODEL: NOT STARTED")
    print("RAW DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
