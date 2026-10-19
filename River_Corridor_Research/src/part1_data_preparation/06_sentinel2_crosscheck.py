
"""
J26-DS-338 - River Corridor Research

STEP 11D: SENTINEL-2 / LANDSAT CROSS-CHECK

Compare:
- Sentinel-2 NDWI (native 10 m bands)
- Sentinel-2 MNDWI (20 m SWIR1 resampled to 10 m)
- Landsat pilot MNDWI water mask (30 m)

All classifications are preliminary.

This is cross-sensor visual QA, not independent
ground-truth accuracy validation.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from matplotlib.colors import ListedColormap


# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

S2_DIR = ROOT / "data/raw/sentinel2"

LANDSAT_MASK_DIR = (
    ROOT / "data/processed/water_masks"
)

BOUNDARY_DIR = ROOT / "data/raw/boundaries"

FIGURE_DIR = (
    ROOT / "outputs/figures/sentinel2_crosscheck"
)

REPORT_DIR = ROOT / "outputs/reports"

FIGURE_DIR.mkdir(
    parents=True, exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True, exist_ok=True
)

YEARS = [2018, 2020, 2025]

NODATA = -9999.0
MASK_NODATA = 255

# SCL classes retained in the original GEE workflow.
ALLOWED_SCL = [2, 4, 5, 6]


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def reproject_band(
    source, source_raster, target_raster,
    method=Resampling.nearest,
    source_nodata=None,
    destination_nodata=NODATA
):
    destination = np.full(
        (target_raster.height, target_raster.width),
        destination_nodata,
        dtype=np.float32
    )

    reproject(
        source=source,
        destination=destination,
        src_transform=source_raster.transform,
        src_crs=source_raster.crs,
        src_nodata=source_nodata,
        dst_transform=target_raster.transform,
        dst_crs=target_raster.crs,
        dst_nodata=destination_nodata,
        resampling=method
    )

    return destination


def spectral_index(a, b, valid):
    denominator = a + b

    good = (
        valid
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-6)
    )

    result = np.full(
        a.shape, NODATA, dtype=np.float32
    )

    np.divide(
        a - b,
        denominator,
        out=result,
        where=good
    )

    good = (
        good
        & np.isfinite(result)
        & (result >= -1)
        & (result <= 1)
    )

    result[~good] = NODATA

    return result


def make_mask(index, valid):
    mask = np.full(
        index.shape,
        MASK_NODATA,
        dtype=np.uint8
    )

    usable = (
        valid
        & np.isfinite(index)
        & (index != NODATA)
    )

    mask[usable] = 0
    mask[usable & (index > 0)] = 1

    return mask


def plot_mask(axis, mask, extent, title):
    cmap = ListedColormap(
        ["#eeeeee", "#2471a3"]
    )
    cmap.set_bad("#888888")

    display = np.ma.masked_where(
        mask == MASK_NODATA,
        mask
    )

    axis.imshow(
        display,
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=extent,
        origin="upper"
    )

    axis.set_title(title)


# ==================================================
# 3. LOAD GEOMETRIES
# ==================================================

print("=" * 70)
print("STEP 11D - SENTINEL-2 / LANDSAT CROSS-CHECK")
print("=" * 70)

corridor = gpd.read_file(
    BOUNDARY_DIR / "kelani_river_corridor.geojson"
)

centreline = gpd.read_file(
    BOUNDARY_DIR / "kelani_river_centreline.geojson"
)

if (
    corridor.empty
    or centreline.empty
    or corridor.crs is None
    or centreline.crs is None
):
    raise ValueError(
        "Missing corridor/centreline geometry or CRS."
    )

records = []


# ==================================================
# 4. PROCESS SELECTED YEARS
# ==================================================

for year in YEARS:

    print("\n" + "-" * 70)
    print(f"PROCESSING YEAR: {year}")

    matches = sorted(
        S2_DIR.glob(
            f"kelani_sentinel2_{year}_*_10m.tif"
        )
    )

    if len(matches) != 1:
        raise ValueError(
            f"{year}: Expected one Sentinel-2 10m file, "
            f"found {len(matches)}"
        )

    file10 = matches[0]

    file20 = Path(
        str(file10).replace("_10m.tif", "_20m.tif")
    )

    if not file20.exists():
        raise FileNotFoundError(file20)

    landsat_matches = sorted(
        LANDSAT_MASK_DIR.glob(
            f"kelani_landsat_{year}_*_pilot_water_mask.tif"
        )
    )

    if len(landsat_matches) != 1:
        raise ValueError(
            f"{year}: Expected one Landsat pilot mask, "
            f"found {len(landsat_matches)}"
        )

    with rasterio.open(file10) as s10, \
         rasterio.open(file20) as s20, \
         rasterio.open(landsat_matches[0]) as landsat:

        if s10.count != 5 or s20.count != 4:
            raise ValueError(
                f"{year}: Unexpected Sentinel-2 bands."
            )

        if (
            s10.crs is None
            or s20.crs != s10.crs
            or s10.crs.to_epsg() != 32644
        ):
            raise ValueError(
                f"{year}: Unexpected Sentinel-2 CRS."
            )

        # 10 m bands: blue, green, red, NIR, valid.
        blue = s10.read(1).astype(np.float32)
        green = s10.read(2).astype(np.float32)
        red = s10.read(3).astype(np.float32)
        nir = s10.read(4).astype(np.float32)
        valid10_band = s10.read(5)

        # 20 m bands: SWIR1, SWIR2, SCL, valid.
        # SWIR is resampled for grid alignment only.
        # It does not gain true 10 m spatial detail.
        swir1 = reproject_band(
            s20.read(1),
            s20,
            s10,
            method=Resampling.bilinear,
            source_nodata=s20.nodata
        )

        scl = reproject_band(
            s20.read(3),
            s20,
            s10,
            method=Resampling.nearest,
            source_nodata=s20.nodata
        )

        valid20_band = reproject_band(
            s20.read(4),
            s20,
            s10,
            method=Resampling.nearest,
            source_nodata=s20.nodata
        )

        projected_corridor = corridor.to_crs(s10.crs)

        corridor_mask = geometry_mask(
            [
                g.__geo_interface__
                for g in projected_corridor.geometry
                if g is not None and not g.is_empty
            ],
            out_shape=green.shape,
            transform=s10.transform,
            invert=True
        )

        valid = (
            corridor_mask
            & (valid10_band > 0.5)
            & (valid20_band > 0.5)
            & np.isin(scl, ALLOWED_SCL)
        )

        for band in (blue, green, red, nir, swir1):
            valid &= (
                np.isfinite(band)
                & (band != NODATA)
            )

        ndwi = spectral_index(
            green, nir, valid
        )

        mndwi = spectral_index(
            green, swir1, valid
        )

        ndwi_mask = make_mask(ndwi, valid)
        mndwi_mask = make_mask(mndwi, valid)

        # Align Landsat pilot mask to S2 grid
        # using nearest-neighbour classification.
        landsat_aligned = np.full(
            green.shape,
            MASK_NODATA,
            dtype=np.uint8
        )

        reproject(
            source=rasterio.band(landsat, 1),
            destination=landsat_aligned,
            src_transform=landsat.transform,
            src_crs=landsat.crs,
            src_nodata=MASK_NODATA,
            dst_transform=s10.transform,
            dst_crs=s10.crs,
            dst_nodata=MASK_NODATA,
            resampling=Resampling.nearest
        )

        total_corridor = int(corridor_mask.sum())
        common_valid = int(
            np.count_nonzero(
                (ndwi_mask != MASK_NODATA)
                & (mndwi_mask != MASK_NODATA)
            )
        )

        ndwi_water = int(
            np.count_nonzero(ndwi_mask == 1)
        )

        mndwi_water = int(
            np.count_nonzero(mndwi_mask == 1)
        )

        records.append({
            "year": year,
            "sentinel2_date_file": file10.name,
            "landsat_mask_file": landsat_matches[0].name,
            "corridor_pixels_10m": total_corridor,
            "common_valid_pixels": common_valid,
            "valid_coverage_pct": round(
                100 * common_valid / total_corridor, 2
            ),
            "ndwi_candidate_water_pixels": ndwi_water,
            "mndwi_candidate_water_pixels": mndwi_water,
            "ndwi_candidate_area_km2": round(
                ndwi_water * 0.0001, 5
            ),
            "mndwi_candidate_area_km2": round(
                mndwi_water * 0.0001, 5
            ),
            "status": "CROSS_SENSOR_QA_NOT_GROUND_TRUTH"
        })

        # ------------------------------------------
        # CREATE VISUAL COMPARISON
        # ------------------------------------------

        rgb_raw = np.stack(
            [red, green, blue],
            axis=-1
        )

        rgb = np.zeros_like(rgb_raw)

        for c in range(3):

            vals = rgb_raw[..., c][valid]

            if vals.size == 0:
                continue

            low, high = np.percentile(
                vals, [2, 98]
            )

            rgb[..., c] = np.clip(
                (rgb_raw[..., c] - low)
                / max(high - low, 1e-6),
                0, 1
            )

        rgb[~valid] = 0.55

        extent = [
            s10.bounds.left,
            s10.bounds.right,
            s10.bounds.bottom,
            s10.bounds.top
        ]

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
            "Sentinel-2 RGB"
        )

        plot_mask(
            axes[0, 1],
            ndwi_mask,
            extent,
            "Sentinel-2 NDWI > 0 (10m bands)"
        )

        plot_mask(
            axes[1, 0],
            mndwi_mask,
            extent,
            "Sentinel-2 MNDWI > 0 (SWIR 20m)"
        )

        plot_mask(
            axes[1, 1],
            landsat_aligned,
            extent,
            "Landsat MNDWI > 0 (30m original)"
        )

        projected_line = centreline.to_crs(s10.crs)

        for ax in axes.flat:

            projected_line.plot(
                ax=ax,
                color="red",
                linewidth=0.6
            )

            ax.set_aspect("equal")
            ax.set_xlabel("UTM Easting (m)")
            ax.set_ylabel("UTM Northing (m)")

        fig.suptitle(
            f"Kelani River Cross-Sensor QA - {year}"
        )

        figure_path = (
            FIGURE_DIR / f"s2_landsat_{year}_qa.png"
        )

        fig.savefig(
            figure_path,
            dpi=130
        )

        plt.close(fig)

        print(
            f"Valid coverage: "
            f"{100 * common_valid / total_corridor:.2f}%"
        )

        print(
            f"Sentinel-2 NDWI candidate water: "
            f"{ndwi_water} pixels"
        )

        print(
            f"Sentinel-2 MNDWI candidate water: "
            f"{mndwi_water} pixels"
        )

        print(f"Saved: {figure_path.name}")


# ==================================================
# 5. SAVE REPORT
# ==================================================

report = pd.DataFrame(records)

report_path = (
    REPORT_DIR / "06_sentinel2_crosscheck_qa.csv"
)

report.to_csv(
    report_path,
    index=False
)

print("\n" + "=" * 70)
print("STEP 11D SUMMARY")
print("=" * 70)

print(report.to_string(index=False))

print("\nReport:")
print(report_path)

print("\nImportant:")
print("Comparison uses different acquisition dates.")
print("No final riverbank validation performed.")

print("\nSTEP 11D FINISHED")
