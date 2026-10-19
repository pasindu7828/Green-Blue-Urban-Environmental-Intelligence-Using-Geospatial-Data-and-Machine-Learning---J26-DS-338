
"""
J26-DS-338 - River Corridor Research

STEP 11B: MNDWI DIAGNOSTICS

Investigate threshold sensitivity and the
2012 Landsat 7 supplementary observation.

This script does NOT select a final threshold
or alter the existing water masks.
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
# 1. PATH CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

INDEX_DIR = (
    ROOT / "data/processed/landsat_indices"
)

RAW_DIR = ROOT / "data/raw/landsat"

CENTRELINE_PATH = (
    ROOT / "data/raw/boundaries"
    / "kelani_river_centreline.geojson"
)

REPORT_DIR = ROOT / "outputs/reports"

FIGURE_DIR = (
    ROOT / "outputs/figures/water_extraction"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BUFFER_DISTANCE_M = 90

THRESHOLDS = [-0.1, 0.0, 0.1]


# ==========================================
# 2. LOAD REFERENCE CENTRELINE
# ==========================================

print("=" * 70)
print("STEP 11B - MNDWI DIAGNOSTIC ANALYSIS")
print("=" * 70)

centreline = gpd.read_file(CENTRELINE_PATH)

if centreline.empty or centreline.crs is None:
    raise ValueError("Invalid river centreline.")

centreline_utm = centreline.to_crs("EPSG:32644")

buffer_geometries = [
    geom.buffer(BUFFER_DISTANCE_M).__geo_interface__
    for geom in centreline_utm.geometry
    if geom is not None and not geom.is_empty
]

if not buffer_geometries:
    raise ValueError("No usable centreline geometry.")

files = sorted(INDEX_DIR.glob("*_indices.tif"))

if len(files) != 18:
    raise ValueError(
        f"Expected 18 index files, found {len(files)}"
    )

records = []


# ==========================================
# 3. ANALYZE ALL OBSERVATIONS
# ==========================================

for file in files:

    with rasterio.open(file) as src:

        mndwi = src.read(1)

        near_line = geometry_mask(
            buffer_geometries,
            out_shape=mndwi.shape,
            transform=src.transform,
            invert=True
        )

        valid = (
            near_line
            & np.isfinite(mndwi)
            & (mndwi != -9999)
            & (mndwi >= -1)
            & (mndwi <= 1)
        )

        values = mndwi[valid]

        if values.size == 0:
            raise ValueError(
                f"No usable near-river pixels: {file.name}"
            )

        # Retrieve original reflectance values.
        raw_name = file.name.replace(
            "_indices.tif", ".tif"
        )

        raw_path = RAW_DIR / raw_name

        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

        with rasterio.open(raw_path) as raw:

            if (
                raw.transform != src.transform
                or raw.crs != src.crs
                or raw.width != src.width
                or raw.height != src.height
            ):
                raise ValueError(
                    f"Raw/index grid mismatch: {file.name}"
                )

            green = raw.read(2)
            swir1 = raw.read(5)

        record = {
            "observation": raw_name,
            "buffer_distance_m": BUFFER_DISTANCE_M,
            "buffer_pixels": int(near_line.sum()),
            "valid_near_river_pixels": int(
                valid.sum()
            ),
            "near_river_valid_pct": round(
                100 * valid.sum() / near_line.sum(), 2
            ),
            "mndwi_p10": float(
                np.percentile(values, 10)
            ),
            "mndwi_median": float(
                np.median(values)
            ),
            "mndwi_p90": float(
                np.percentile(values, 90)
            ),
            "green_median": float(
                np.median(green[valid])
            ),
            "swir1_median": float(
                np.median(swir1[valid])
            ),
            "pixels_mndwi_minus01_to_zero": int(
                ((values > -0.1) & (values <= 0)).sum()
            )
        }

        for threshold in THRESHOLDS:

            key = (
                str(threshold)
                .replace("-", "minus")
                .replace(".", "_")
            )

            record[f"water_pixels_gt_{key}"] = int(
                (values > threshold).sum()
            )

            record[f"water_pct_gt_{key}"] = round(
                100 * (values > threshold).mean(),
                2
            )

        records.append(record)

        print(
            f"{raw_name} | "
            f"valid={record['near_river_valid_pct']}% | "
            f"median MNDWI="
            f"{record['mndwi_median']:.3f}"
        )

        # ==================================
        # 4. SPECIAL 2012 MAY DIAGNOSTIC MAP
        # ==================================

        if "2012_L7_20120531" in file.name:

            fig, axes = plt.subplots(
                1, 3, figsize=(16, 5)
            )

            bounds = src.bounds

            extent = [
                bounds.left,
                bounds.right,
                bounds.bottom,
                bounds.top
            ]

            all_valid = (
                np.isfinite(mndwi)
                & (mndwi != -9999)
                & (mndwi >= -1)
                & (mndwi <= 1)
            )

            display_mndwi = np.ma.masked_where(
                ~all_valid,
                mndwi
            )

            axes[0].imshow(
                display_mndwi,
                cmap="BrBG",
                vmin=-0.3,
                vmax=0.3,
                extent=extent,
                origin="upper"
            )

            axes[0].set_title(
                "2012 May - MNDWI"
            )

            water_cmap = ListedColormap(
                ["#eeeeee", "#2471a3"]
            )

            water_cmap.set_bad("#888888")

            for axis, threshold in [
                (axes[1], 0.0),
                (axes[2], -0.1)
            ]:

                classified = np.ma.masked_where(
                    ~all_valid,
                    (mndwi > threshold).astype(
                        np.uint8
                    )
                )

                axis.imshow(
                    classified,
                    cmap=water_cmap,
                    vmin=0,
                    vmax=1,
                    extent=extent,
                    origin="upper"
                )

                axis.set_title(
                    f"Threshold > {threshold}"
                )

            for ax in axes:

                centreline_utm.plot(
                    ax=ax,
                    color="red",
                    linewidth=0.7
                )

                ax.set_aspect("equal")
                ax.set_xlabel("UTM Easting (m)")

            fig.suptitle(
                "2012 Supplementary - Diagnostic Only"
            )

            fig.tight_layout()

            figure_path = (
                FIGURE_DIR
                / "2012_supp_threshold.png"
            )

            fig.savefig(
                figure_path,
                dpi=130
            )

            plt.close(fig)

            print(
                "Saved special diagnostic map."
            )


# ==========================================
# 5. EXPORT DIAGNOSTIC REPORT
# ==========================================

df = pd.DataFrame(records)

csv_path = (
    REPORT_DIR
    / "04_mndwi_diagnostics.csv"
)

df.to_csv(csv_path, index=False)


# ==========================================
# 6. SUMMARY
# ==========================================

print("\n" + "=" * 70)
print("STEP 11B - DIAGNOSTIC SUMMARY")
print("=" * 70)

print(f"Observations analyzed: {len(df)}")

print("\nLowest near-river valid coverage:")

print(
    df.nsmallest(
        5,
        "near_river_valid_pct"
    )[
        [
            "observation",
            "near_river_valid_pct",
            "mndwi_median"
        ]
    ].to_string(index=False)
)

print("\nCSV REPORT:")
print(csv_path)

print("\nSPECIAL MAP:")
print(FIGURE_DIR / "2012_supp_threshold.png")

print("\nThreshold selection remains pending.")
print("STEP 11B FINISHED")
