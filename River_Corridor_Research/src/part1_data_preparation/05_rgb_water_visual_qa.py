
"""
J26-DS-338 - River Corridor Research

STEP 11C: RGB AND WATER EXTRACTION VISUAL QA

Compare true-colour Landsat imagery, MNDWI,
and candidate water masks.

This is diagnostic work, NOT validation
against independent reference labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt

from rasterio.features import geometry_mask
from matplotlib.colors import ListedColormap


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data/raw/landsat"

INDEX_DIR = (
    ROOT / "data/processed/landsat_indices"
)

CENTRELINE_PATH = (
    ROOT / "data/raw/boundaries"
    / "kelani_river_centreline.geojson"
)

FIGURE_DIR = (
    ROOT / "outputs/figures/water_extraction"
)

REPORT_DIR = ROOT / "outputs/reports"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# Selected observations for diagnosis.
SCENES = {
    "2010": "kelani_landsat_2010_L5_20100126",
    "2012_primary": "kelani_landsat_2012_L7_20120328_primary_gapyear",
    "2012_may": "kelani_landsat_2012_L7_20120531_supplementary_gapyear",
    "2015": "kelani_landsat_2015_L8_20150108",
    "2020": "kelani_landsat_2020_L8_20200106",
    "2025": "kelani_landsat_2025_L8_20250103",
}


# ==========================================
# 2. LOAD RIVER CENTRELINE
# ==========================================

line = gpd.read_file(CENTRELINE_PATH)

if line.empty or line.crs is None:
    raise ValueError("Missing centreline or CRS.")

line_utm = line.to_crs("EPSG:32644")

print("=" * 70)
print("STEP 11C - RGB / MNDWI WATER QA")
print("=" * 70)

records = []


# ==========================================
# 3. PROCESS SELECTED OBSERVATIONS
# ==========================================

for label, stem in SCENES.items():

    raw_file = RAW_DIR / f"{stem}.tif"
    index_file = INDEX_DIR / f"{stem}_indices.tif"

    if not raw_file.exists():
        raise FileNotFoundError(raw_file)

    if not index_file.exists():
        raise FileNotFoundError(index_file)

    with rasterio.open(raw_file) as raw, \
         rasterio.open(index_file) as idx:

        # Check spatial alignment.
        if (
            raw.crs != idx.crs
            or raw.transform != idx.transform
            or (raw.height, raw.width)
            != (idx.height, idx.width)
        ):
            raise ValueError(
                f"Raster grid mismatch: {label}"
            )

        # Read index and RGB bands.
        mndwi = idx.read(1)

        red = raw.read(3).astype("float32")
        green = raw.read(2).astype("float32")
        blue = raw.read(1).astype("float32")

        valid = (
            np.isfinite(mndwi)
            & (mndwi != -9999)
            & (mndwi >= -1)
            & (mndwi <= 1)
        )

        if not valid.any():
            raise ValueError(
                f"No valid pixels: {label}"
            )

        # ==================================
        # 4. PREPARE RGB VISUALIZATION
        # ==================================

        rgb_raw = np.stack(
            [red, green, blue],
            axis=-1
        )

        rgb = np.zeros_like(
            rgb_raw,
            dtype="float32"
        )

        # Contrast stretching for display only.
        for channel in range(3):

            values = rgb_raw[..., channel][valid]

            low, high = np.percentile(
                values,
                [2, 98]
            )

            rgb[..., channel] = np.clip(
                (rgb_raw[..., channel] - low)
                / max(high - low, 1e-6),
                0,
                1
            )

        rgb[~valid] = 0.55

        extent = [
            idx.bounds.left,
            idx.bounds.right,
            idx.bounds.bottom,
            idx.bounds.top
        ]

        # ==================================
        # 5. NEAR-RIVER DIAGNOSTICS
        # ==================================

        buffer_geometries = [
            geom.buffer(90).__geo_interface__
            for geom in line_utm.geometry
            if geom is not None
            and not geom.is_empty
        ]

        near_line = geometry_mask(
            buffer_geometries,
            out_shape=mndwi.shape,
            transform=idx.transform,
            invert=True
        )

        sample = near_line & valid

        total_near_pixels = int(
            near_line.sum()
        )

        valid_near_pixels = int(
            sample.sum()
        )

        water_zero = int(
            np.count_nonzero(
                sample & (mndwi > 0.0)
            )
        )

        water_negative = int(
            np.count_nonzero(
                sample & (mndwi > -0.1)
            )
        )

        valid_pct = (
            100 * valid_near_pixels
            / total_near_pixels
        )

        water_pct = (
            100 * water_zero
            / max(valid_near_pixels, 1)
        )

        water_negative_pct = (
            100 * water_negative
            / max(valid_near_pixels, 1)
        )

        review_flag = (
            "LOW_COVERAGE_OR_LOW_WATER"
            if valid_pct < 80 or water_pct < 5
            else "VISUAL_REVIEW_REQUIRED"
        )

        records.append({
            "scene": label,
            "source": raw_file.name,
            "near_river_valid_pct": round(
                valid_pct, 2
            ),
            "water_pct_threshold_0": round(
                water_pct, 2
            ),
            "water_pct_threshold_minus_0_1": round(
                water_negative_pct, 2
            ),
            "review_flag": review_flag
        })

        # ==================================
        # 6. CREATE VISUAL COMPARISON
        # ==================================

        fig, axes = plt.subplots(
            2, 2,
            figsize=(15, 8),
            constrained_layout=True
        )

        axes[0, 0].imshow(
            rgb,
            extent=extent,
            origin="upper"
        )

        axes[0, 0].set_title(
            "Contrast-stretched RGB"
        )

        mndwi_display = np.ma.masked_where(
            ~valid,
            mndwi
        )

        index_cmap = plt.get_cmap(
            "BrBG"
        ).copy()

        index_cmap.set_bad("#888888")

        axes[0, 1].imshow(
            mndwi_display,
            cmap=index_cmap,
            vmin=-0.5,
            vmax=0.5,
            extent=extent,
            origin="upper"
        )

        axes[0, 1].set_title(
            "MNDWI Distribution"
        )

        for ax, threshold in [
            (axes[1, 0], 0.0),
            (axes[1, 1], -0.1)
        ]:

            classified = np.ma.masked_where(
                ~valid,
                (mndwi > threshold).astype(
                    "uint8"
                )
            )

            water_cmap = ListedColormap(
                ["#eeeeee", "#2471a3"]
            )

            water_cmap.set_bad("#888888")

            ax.imshow(
                classified,
                cmap=water_cmap,
                vmin=0,
                vmax=1,
                extent=extent,
                origin="upper"
            )

            ax.set_title(
                f"Candidate Water: MNDWI > {threshold}"
            )

        # Add reference centreline.
        for ax in axes.flat:

            line_utm.plot(
                ax=ax,
                color="red",
                linewidth=0.6
            )

            ax.set_aspect("equal")
            ax.set_xlabel("UTM Easting (m)")
            ax.set_ylabel("UTM Northing (m)")

        fig.suptitle(
            f"River Water Extraction QA - {label}"
        )

        # Short filenames avoid Windows path errors.
        figure_path = (
            FIGURE_DIR / f"qa_rgb_{label}.png"
        )

        fig.savefig(
            figure_path,
            dpi=130
        )

        plt.close(fig)

        print(
            f"{label}: "
            f"Valid={valid_pct:.2f}% | "
            f"Water={water_pct:.2f}%"
        )

        print(
            f"Saved: {figure_path.name}"
        )


# ==========================================
# 7. EXPORT REPORT
# ==========================================

df = pd.DataFrame(records)

report_path = (
    REPORT_DIR
    / "05_rgb_water_visual_qa.csv"
)

df.to_csv(
    report_path,
    index=False
)

print("\n" + "=" * 70)
print("STEP 11C - FINAL SUMMARY")
print("=" * 70)

print(f"Scenes compared: {len(df)}")
print(df.to_string(index=False))

print("\nCSV Report:")
print(report_path)

print("\nIMPORTANT:")
print("Visual QA is not independent accuracy validation.")

print("\nSTEP 11C FINISHED")
