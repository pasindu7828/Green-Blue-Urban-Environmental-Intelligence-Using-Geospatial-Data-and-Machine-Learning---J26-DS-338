
"""
J26-DS-338 - River Corridor Research

STEP 11: PRELIMINARY LANDSAT WATER EXTRACTION

Input:
    18 Landsat index GeoTIFF files
    River corridor and centreline GeoJSON

Method:
    MNDWI threshold classification
    Pilot threshold = 0.0

Outputs:
    18 candidate water-mask GeoTIFF files
    6 visual QA comparison images
    1 threshold sensitivity and QA CSV

Class values:
    0 = Non-water
    1 = Candidate water
    255 = Invalid / no-data

IMPORTANT:
    These are preliminary candidate water masks.
    They are NOT validated riverbank boundaries.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from rasterio.features import geometry_mask


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT / "data" / "processed" / "landsat_indices"
)

CORRIDOR_PATH = (
    ROOT / "data" / "raw" / "boundaries"
    / "kelani_river_corridor.geojson"
)

CENTRELINE_PATH = (
    ROOT / "data" / "raw" / "boundaries"
    / "kelani_river_centreline.geojson"
)

MASK_DIR = (
    ROOT / "data" / "processed" / "water_masks"
)

FIGURE_DIR = (
    ROOT / "outputs" / "figures"
    / "water_extraction"
)

REPORT_DIR = ROOT / "outputs" / "reports"

for folder in (MASK_DIR, FIGURE_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

PILOT_THRESHOLD = 0.0
TEST_THRESHOLDS = [-0.1, 0.0, 0.1, 0.2]

INVALID_CLASS = 255
INDEX_NODATA = -9999.0

QUICKLOOK_YEARS = {2010, 2012, 2013, 2015, 2020, 2025}


# ==================================================
# 2. LOAD VECTOR GEOMETRIES
# ==================================================

print("=" * 75)
print("STEP 11 - PRELIMINARY WATER EXTRACTION")
print("=" * 75)

corridor = gpd.read_file(CORRIDOR_PATH)
centreline = gpd.read_file(CENTRELINE_PATH)

if (
    corridor.empty
    or centreline.empty
    or corridor.crs is None
    or centreline.crs is None
):
    raise ValueError(
        "Corridor or centreline is missing "
        "valid geometry/CRS information."
    )

files = sorted(INPUT_DIR.glob("*_indices.tif"))

print(f"\nIndex files found: {len(files)}")

if len(files) != 18:
    raise ValueError(
        "Expected 18 Landsat index files."
    )

records = []


# ==================================================
# 3. PROCESS EACH LANDSAT IMAGE
# ==================================================

for number, path in enumerate(files, 1):

    print("\n" + "-" * 75)
    print(f"IMAGE {number}/{len(files)}")
    print(f"Source: {path.name}")

    year = int(path.stem.split("_")[2])

    with rasterio.open(path) as src:

        if src.count != 3:
            raise ValueError(
                f"Expected 3 index bands: {path.name}"
            )

        if src.crs is None:
            raise ValueError(
                f"Missing raster CRS: {path.name}"
            )

        if src.crs.to_epsg() != 32644:
            raise ValueError(
                f"Unexpected raster CRS: {src.crs}"
            )

        mndwi = src.read(1)

        # ------------------------------------------
        # A. CREATE CORRIDOR MASK
        # ------------------------------------------

        projected_corridor = corridor.to_crs(
            src.crs
        )

        geometries = [
            geom.__geo_interface__
            for geom in projected_corridor.geometry
            if geom is not None and not geom.is_empty
        ]

        corridor_mask = geometry_mask(
            geometries,
            out_shape=mndwi.shape,
            transform=src.transform,
            invert=True
        )

        corridor_pixels = int(
            corridor_mask.sum()
        )

        if corridor_pixels == 0:
            raise ValueError(
                "Study corridor does not overlap image."
            )

        # ------------------------------------------
        # B. IDENTIFY USABLE MNDWI PIXELS
        # ------------------------------------------

        valid = (
            corridor_mask
            & np.isfinite(mndwi)
            & (mndwi != INDEX_NODATA)
            & (mndwi >= -1.0)
            & (mndwi <= 1.0)
        )

        valid_pixels = int(valid.sum())

        if valid_pixels == 0:
            raise ValueError(
                f"No valid MNDWI pixels: {path.name}"
            )

        valid_coverage = (
            100.0 * valid_pixels / corridor_pixels
        )

        # ------------------------------------------
        # C. PILOT WATER CLASSIFICATION
        # ------------------------------------------

        candidate_water = (
            valid & (mndwi > PILOT_THRESHOLD)
        )

        mask = np.full(
            mndwi.shape,
            INVALID_CLASS,
            dtype=np.uint8
        )

        mask[valid] = 0
        mask[candidate_water] = 1

        water_pixels = int(
            candidate_water.sum()
        )

        nonwater_pixels = (
            valid_pixels - water_pixels
        )

        water_percentage = (
            100.0 * water_pixels / valid_pixels
        )

        pixel_area_km2 = (
            abs(
                src.transform.a * src.transform.e
                - src.transform.b * src.transform.d
            ) / 1_000_000.0
        )

        water_area_km2 = (
            water_pixels * pixel_area_km2
        )

        print(
            f"Valid coverage : {valid_coverage:.2f}%"
        )

        print(
            f"Water pixels   : {water_pixels}"
        )

        print(
            f"Water fraction : {water_percentage:.2f}%"
        )

        print(
            f"Water area     : {water_area_km2:.4f} km2"
        )

        # ------------------------------------------
        # D. THRESHOLD SENSITIVITY CHECK
        # ------------------------------------------

        record = {
            "source_file": path.name,
            "year": year,
            "pilot_threshold": PILOT_THRESHOLD,
            "corridor_pixels": corridor_pixels,
            "valid_pixels": valid_pixels,
            "valid_coverage_pct": round(
                valid_coverage, 3
            ),
            "candidate_water_pixels": water_pixels,
            "candidate_nonwater_pixels": nonwater_pixels,
            "candidate_water_pct_of_valid": round(
                water_percentage, 3
            ),
            "candidate_water_area_km2": round(
                water_area_km2, 5
            ),
        }

        for threshold in TEST_THRESHOLDS:

            count = int(
                np.count_nonzero(
                    valid & (mndwi > threshold)
                )
            )

            label = (
                str(threshold)
                .replace("-", "minus_")
                .replace(".", "_")
            )

            record[
                f"water_area_threshold_{label}_km2"
            ] = round(
                count * pixel_area_km2, 5
            )

        records.append(record)

        # ------------------------------------------
        # E. SAVE CANDIDATE WATER MASK
        # ------------------------------------------

        output_path = (
            MASK_DIR
            / f"{path.stem}_pilot_water_mask.tif"
        )

        profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=INVALID_CLASS,
            compress="deflate"
        )

        profile.pop("predictor", None)

        with rasterio.open(
            output_path, "w", **profile
        ) as dst:

            dst.write(mask, 1)

            dst.set_band_description(
                1, "PILOT_CANDIDATE_WATER"
            )

            dst.update_tags(
                threshold=str(PILOT_THRESHOLD),
                validation_status="UNVALIDATED_PILOT",
                class_0="Non-water",
                class_1="Candidate water",
                class_255="Invalid"
            )

        print(f"Saved mask: {output_path.name}")

        # ------------------------------------------
        # F. GENERATE SELECTED VISUAL QA MAPS
        # ------------------------------------------

        make_quicklook = (
            year in QUICKLOOK_YEARS
            and (
                year not in {2012, 2013}
                or "primary" in path.name
            )
        )

        if make_quicklook:

            fig, axes = plt.subplots(
                1, 2, figsize=(14, 5)
            )

            bounds = src.bounds

            extent = [
                bounds.left,
                bounds.right,
                bounds.bottom,
                bounds.top
            ]

            # Left panel: MNDWI
            mndwi_display = np.ma.masked_where(
                ~valid, mndwi
            )

            index_cmap = plt.get_cmap(
                "BrBG"
            ).copy()

            index_cmap.set_bad("lightgray")

            im = axes[0].imshow(
                mndwi_display,
                cmap=index_cmap,
                vmin=-1,
                vmax=1,
                extent=extent,
                origin="upper"
            )

            axes[0].set_title(
                f"{year} - MNDWI"
            )

            fig.colorbar(
                im,
                ax=axes[0],
                shrink=0.7
            )

            # Right panel: candidate water
            water_display = np.ma.masked_where(
                mask == INVALID_CLASS,
                mask
            )

            water_cmap = ListedColormap(
                ["#eeeeee", "#2471a3"]
            )

            water_cmap.set_bad("#888888")

            axes[1].imshow(
                water_display,
                cmap=water_cmap,
                vmin=0,
                vmax=1,
                extent=extent,
                origin="upper"
            )

            axes[1].set_title(
                f"{year} - Pilot Water Mask"
            )

            projected_line = centreline.to_crs(
                src.crs
            )

            for ax in axes:

                projected_line.plot(
                    ax=ax,
                    color="red",
                    linewidth=0.7
                )

                ax.set_aspect("equal")
                ax.set_xlabel("UTM Easting (m)")
                ax.set_ylabel("UTM Northing (m)")

            fig.suptitle(
                f"Kelani River Corridor - {path.stem}",
                fontsize=11
            )

            fig.tight_layout()

            
            figure_path = FIGURE_DIR / f"{year}_pilot_qa.png"


            fig.savefig(
                figure_path,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close(fig)

            print(
                f"Saved QA map: {figure_path.name}"
            )


# ==================================================
# 4. EXPORT WATER EXTRACTION QA REPORT
# ==================================================

report = pd.DataFrame(records)

report_path = (
    REPORT_DIR
    / "03_pilot_water_extraction_qa.csv"
)

report.to_csv(
    report_path,
    index=False
)


# ==================================================
# 5. PRINT FINAL RESULTS
# ==================================================

print("\n" + "=" * 75)
print("STEP 11 - PILOT WATER EXTRACTION SUMMARY")
print("=" * 75)

print(f"Images processed: {len(report)}")

print(
    "Minimum valid coverage: "
    f"{report['valid_coverage_pct'].min():.2f}%"
)

print(
    "Maximum valid coverage: "
    f"{report['valid_coverage_pct'].max():.2f}%"
)

print(
    "Minimum candidate water area: "
    f"{report['candidate_water_area_km2'].min():.4f} km2"
)

print(
    "Maximum candidate water area: "
    f"{report['candidate_water_area_km2'].max():.4f} km2"
)

print("\nLow-coverage observations (<80%):")

low_coverage = report[
    report["valid_coverage_pct"] < 80
]

if low_coverage.empty:
    print("None")
else:
    print(
        low_coverage[
            ["source_file", "valid_coverage_pct"]
        ].to_string(index=False)
    )

print("\nOutput directories:")
print(MASK_DIR)
print(FIGURE_DIR)

print("\nQA report:")
print(report_path)

print("\nIMPORTANT:")
print("Outputs are preliminary water candidates.")
print("Riverbank delineation is NOT yet validated.")

print("\nSTEP 11 FINISHED")
print("=" * 75)
