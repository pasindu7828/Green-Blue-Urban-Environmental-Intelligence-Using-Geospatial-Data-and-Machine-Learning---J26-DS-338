"""Step 22 — Eight-year Sentinel-2 river-corridor water-footprint screening.

READS (never modifies):
  River_Corridor_Research/data/raw/sentinel2/kelani_sentinel2_YYYY_YYYYMMDD_10m.tif
  River_Corridor_Research/data/raw/sentinel2/kelani_sentinel2_YYYY_YYYYMMDD_20m.tif
  River_Corridor_Research/data/raw/boundaries/kelani_river_corridor.geojson

WRITES only new, independently named River_Corridor research outputs:
  data/processed/sentinel2_change_v2/*.tif
  outputs/reports/22_s2_annual_water_qa.csv
  outputs/reports/22_s2_pairwise_change_qa.csv
  outputs/figures/22_s2_change_2018_2024.png
  outputs/figures/22_s2_change_2018_2025.png

This is a satellite-derived water-FOOTPRINT comparison. It is NOT validated
riverbank displacement, erosion/accretion, flood risk or legal encroachment.
Water stage, season, tide, cloud, classification and pixel size all matter.

Method: use native 10 m NDWI=(G-NIR)/(G+NIR), and MNDWI=(G-SWIR1)/(G+SWIR1),
with native 20 m SWIR1 bilinearly aligned to the 10 m grid (no gained detail).
Only pixel-centre corridor-covered, valid, conservatively SCL-screened pixels
are compared. Primary exploratory water = BOTH NDWI>0 and MNDWI>0. Changing
thresholds or SCL policy requires rerunning and recording sensitivity analysis.
"""
from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[2]
S2_DIR = ROOT / "data" / "raw" / "sentinel2"
CORRIDOR_FILE = ROOT / "data" / "raw" / "boundaries" / "kelani_river_corridor.geojson"
PROCESSED = ROOT / "data" / "processed" / "sentinel2_change_v2"
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"

ALLOWED_SCL = {4, 5, 6}  # Vegetation, bare/non-vegetated, water; excludes shadow/cloud/unclassified.
MIN_PAIR_COMMON_COVERAGE_PCT = 80.0  # QA flag only; does not invent/fill missing pixels.
MASK_NODATA = 255
EPS = 1e-6


def checked_raster(src: rasterio.io.DatasetReader, expected_res: int, bands: tuple[str, ...]):
    if src.crs is None:
        raise ValueError(f"Missing CRS: {src.name}")
    if not np.allclose(src.res, (expected_res, expected_res), atol=1e-5):
        raise ValueError(f"Unexpected pixel resolution for {src.name}: {src.res}")
    if src.count != len(bands) or tuple(src.descriptions) != bands:
        raise ValueError(
            f"Unexpected bands in {src.name}. Expected {bands}, got {src.descriptions}. "
            "Do not guess band order."
        )
    if src.nodata is None:
        raise ValueError(f"No explicit NoData marker in {src.name}; inspect before processing")


def regrid(source: np.ndarray, src20, dst10, method: Resampling) -> np.ndarray:
    out = np.full((dst10.height, dst10.width), -9999.0, dtype=np.float32)
    reproject(
        source=source.astype(np.float32), destination=out,
        src_transform=src20.transform, src_crs=src20.crs, src_nodata=src20.nodata,
        dst_transform=dst10.transform, dst_crs=dst10.crs, dst_nodata=-9999.0,
        resampling=method,
    )
    return out


def normalized_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = a + b
    out = np.full(a.shape, np.nan, dtype=np.float32)
    where = np.isfinite(den) & np.isfinite(a) & np.isfinite(b) & (np.abs(den) > EPS)
    np.divide(a - b, den, out=out, where=where)
    return out


def write_mask(path: Path, arr: np.ndarray, profile: dict, desc: str, categories: dict[int, str]):
    p = profile.copy()
    p.update(driver="GTiff", dtype="uint8", count=1, nodata=MASK_NODATA,
             compress="deflate", predictor=1)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, desc)
        dst.update_tags(**{f"class_{key}": value for key, value in categories.items()})
        dst.update_tags(interpretation="EXPLORATORY_NOT_VALIDATED_EROSION")


def table(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No QA rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coverage_pct(n: int, base: int):
    return round(100.0 * n / base, 2) if base else 0.0


def make_rgb(scene: dict, valid: np.ndarray):
    rgb = np.stack((scene["red"], scene["green"], scene["blue"]), axis=-1)
    values = rgb[valid & np.isfinite(rgb).all(axis=2)]
    if values.size > 100:
        # Same stretch for all three channels retains approximate RGB colour balance.
        lo, hi = np.nanpercentile(values, [2, 98])
        if hi > lo + EPS:
            scaled = np.clip((rgb - lo) / (hi - lo), 0, 1)
        else:
            scaled = np.clip(rgb / max(abs(hi), EPS), 0, 1)
    else:
        scaled = np.zeros_like(rgb, dtype=np.float32)
    scaled[~np.isfinite(scaled)] = 0
    scaled[~scene["corridor"]] = 0.86
    return scaled


def overlay_colormap(scene: dict, display_water: np.ndarray, display_gain=None, display_loss=None):
    h, w = scene["valid"].shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    # Stable or year-specific water: vivid blue.
    rgba[display_water] = [0.03, 0.40, 0.90, 0.78]
    if display_gain is not None:
        rgba[display_gain] = [0.00, 0.80, 0.48, 0.88]
    if display_loss is not None:
        rgba[display_loss] = [1.00, 0.35, 0.08, 0.88]
    return rgba


def plot_pair(old_year: int, new_year: int, scenes: dict, pair_row: dict, output: Path):
    old = scenes[old_year]
    new = scenes[new_year]
    common = old["valid"] & new["valid"]
    stable = common & old["water"] & new["water"]
    gain = common & ~old["water"] & new["water"]
    loss = common & old["water"] & ~new["water"]

    fig = plt.figure(figsize=(18, 7.2), facecolor="white")
    grid = fig.add_gridspec(1, 3, left=0.025, right=0.975, bottom=0.17, top=0.77, wspace=0.03)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    for ax, scene, year in [(axes[0], old, old_year), (axes[1], new, new_year)]:
        ax.imshow(make_rgb(scene, scene["valid"]))
        ax.imshow(overlay_colormap(scene, scene["water"] & common), interpolation="nearest")
        ax.set_title(f"{year} | {scene['date']}\nSatellite RGB + consensus water", fontsize=11)
        ax.axis("off")

    axes[2].imshow(make_rgb(new, new["valid"]))
    axes[2].imshow(overlay_colormap(new, stable, gain, loss), interpolation="nearest")
    axes[2].set_title(f"{old_year} to {new_year} | Change\nCompared on common-valid pixels only", fontsize=11)
    axes[2].axis("off")
    fig.legend(handles=[
        Patch(facecolor="#0666e6", label="Stable/observed water"),
        Patch(facecolor="#00cc7a", label="Water gain"),
        Patch(facecolor="#ff5a14", label="Water loss"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.10), frameon=False, fontsize=11)
    fig.suptitle("KELANI RIVER CORRIDOR | EXPLORATORY SENTINEL-2 WATER CHANGE",
                 y=0.97, fontsize=16, weight="bold")
    fig.text(0.5, 0.84,
             f"Common valid: {pair_row['common_valid_pct']}%   |   "
             f"Gain: {pair_row['gain_ha']} ha   |   "
             f"Loss: {pair_row['loss_ha']} ha   |   QA: {pair_row['qa_status']}",
             ha="center", fontsize=11)
    fig.text(0.5, 0.05, "NOT validated riverbank erosion. Water stage, acquisition date, pixel size and index threshold matter. "
             "MNDWI aligns 20 m SWIR to a 10 m grid.",
             ha="center", fontsize=10, color="#555555")
    fig.savefig(output, dpi=165, facecolor="white")
    plt.close(fig)


def main():
    files = sorted(S2_DIR.glob("kelani_sentinel2_*_*_10m.tif"))
    if len(files) < 2:
        raise FileNotFoundError(f"Expected >=2 year pairs of Sentinel-2 GeoTIFFs in {S2_DIR}")
    if not CORRIDOR_FILE.is_file():
        raise FileNotFoundError(f"Missing corridor geometry: {CORRIDOR_FILE}")
    corridor = gpd.read_file(CORRIDOR_FILE)
    if corridor.empty or corridor.crs is None:
        raise ValueError("Corridor GeoJSON has empty features or unknown CRS; stop rather than guess")
    if corridor.geometry.isna().any() or (~corridor.geometry.is_valid).any():
        raise ValueError("Invalid or missing corridor polygon; inspect/repair separately")

    # Validate all input presence before producing outputs.
    inputs = {}
    for f10 in files:
        m = re.fullmatch(r"kelani_sentinel2_(\d{4})_(\d{8})_10m\.tif", f10.name)
        if not m:
            raise ValueError(f"Unexpected Sentinel filename {f10.name}")
        year, datestr = int(m.group(1)), m.group(2)
        if datetime.strptime(datestr, "%Y%m%d").year != year:
            raise ValueError(f"Filename year/date mismatch {f10.name}")
        f20 = f10.with_name(f10.name.replace("_10m.tif", "_20m.tif"))
        if not f20.is_file():
            raise FileNotFoundError(f"Missing 20 m partner: {f20}")
        if year in inputs:
            raise ValueError(f"More than one selected scene for year {year}; choose deliberately")
        inputs[year] = (f10, f20, datestr)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    annual, pairs, scenes = [], [], {}
    first_grid = None
    corridor_pixels = None
    for year, (f10, f20, datestr) in sorted(inputs.items()):
        with rasterio.open(f10) as s10, rasterio.open(f20) as s20:
            checked_raster(s10, 10, ("blue", "green", "red", "nir", "valid_pixel"))
            checked_raster(s20, 20, ("swir1", "swir2", "scl", "valid_pixel"))
            if s10.crs != s20.crs:
                raise ValueError(f"10/20 m CRS mismatch in {year}")
            shape_grid = (s10.height, s10.width, s10.crs.to_string(), tuple(s10.transform))
            if first_grid is None:
                first_grid = shape_grid
                geo = corridor.to_crs(s10.crs)
                inside = geometry_mask(geo.geometry.to_list(), out_shape=(s10.height, s10.width),
                                       transform=s10.transform, invert=True, all_touched=False)
                corridor_pixels = int(inside.sum())
                if not corridor_pixels:
                    raise ValueError("Corridor does not overlap satellite extent/CRS; stop")
                pixel_area_ha = abs(s10.transform.a * s10.transform.e - s10.transform.b * s10.transform.d) / 10000
            elif shape_grid != first_grid:
                raise ValueError(f"Year {year} uses a different 10 m grid; explicitly align before change detection")
            blue, green, red, nir, valid10 = [s10.read(i).astype(np.float32) for i in range(1, 6)]
            swir = regrid(s20.read(1), s20, s10, Resampling.bilinear)
            scl = regrid(s20.read(3), s20, s10, Resampling.nearest)
            valid20 = regrid(s20.read(4), s20, s10, Resampling.nearest)
            band_ok = np.ones(green.shape, dtype=bool)
            for band in [blue, green, red, nir, swir]:
                band_ok &= np.isfinite(band) & (band != -9999.0)
            # NODATA value in all supplied exports is reported as -9999.
            valid = (inside & band_ok & np.isfinite(valid10) & (valid10 > 0.5)
                     & np.isfinite(valid20) & (valid20 > 0.5)
                     & np.isin(scl, list(ALLOWED_SCL)))
            ndwi = normalized_diff(green, nir)
            mndwi = normalized_diff(green, swir)
            valid &= np.isfinite(ndwi) & np.isfinite(mndwi)
            ndwi_water = valid & (ndwi > 0)
            mndwi_water = valid & (mndwi > 0)
            consensus_water = ndwi_water & mndwi_water
            either = ndwi_water | mndwi_water
            agreement_water_iou = round(float(consensus_water.sum() / either.sum()), 4) if either.any() else None
            class_mask = np.full(green.shape, MASK_NODATA, dtype=np.uint8)
            class_mask[valid] = 0
            class_mask[consensus_water] = 1
            class_mask[ndwi_water & ~mndwi_water] = 2
            class_mask[mndwi_water & ~ndwi_water] = 3
            outfile = PROCESSED / f"22_s2_{year}_water_agreement_10m.tif"
            write_mask(outfile, class_mask, s10.profile, "Water-index agreement classification",
                       {0: "nonwater", 1: "NDWI_and_MNDWI_water", 2: "NDWI_only_water",
                        3: "MNDWI_only_water", 255: "invalid_or_outside_corridor"})
            scl_counts = {f"scl_{k}_pct": coverage_pct(int((inside & (scl == k)).sum()), corridor_pixels)
                          for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
            row = {
                "year": year, "observation_date": datestr, "s2_10m_file": f10.name,
                "s2_20m_file": f20.name, "corridor_pixels": corridor_pixels,
                "valid_pixels": int(valid.sum()), "valid_coverage_pct": coverage_pct(int(valid.sum()), corridor_pixels),
                "ndwi_water_pixels": int(ndwi_water.sum()), "mndwi_water_pixels": int(mndwi_water.sum()),
                "consensus_water_pixels": int(consensus_water.sum()),
                "consensus_water_area_ha_on_valid_pixels": round(float(consensus_water.sum() * pixel_area_ha), 3),
                "ndwi_mndwi_water_iou": agreement_water_iou,
                "ndwi_only_pixels": int((ndwi_water & ~mndwi_water).sum()),
                "mndwi_only_pixels": int((mndwi_water & ~ndwi_water).sum()),
                "mask_file": outfile.name, "status": "PRELIMINARY_NOT_GROUND_TRUTH",
                **scl_counts,
            }
            annual.append(row)
            scenes[year] = {"date": datestr, "blue": blue, "green": green, "red": red,
                            "valid": valid, "water": consensus_water,
                            "corridor": inside, "profile": s10.profile}
            print(f"{year}: valid {row['valid_coverage_pct']:.2f}% | consensus water "
                  f"{row['consensus_water_area_ha_on_valid_pixels']:.3f} ha | water IoU {agreement_water_iou}")

    years = sorted(scenes)
    selected_pairs = [(a, b) for a, b in zip(years[:-1], years[1:])]
    for a, b in [(2018, 2024), (2018, 2025)]:
        if a in scenes and b in scenes and (a, b) not in selected_pairs:
            selected_pairs.append((a, b))
    for year_a, year_b in selected_pairs:
        a, b = scenes[year_a], scenes[year_b]
        common = a["valid"] & b["valid"]
        n_common = int(common.sum())
        old_water = common & a["water"]
        new_water = common & b["water"]
        stable = old_water & new_water
        gain = ~old_water & new_water
        loss = old_water & ~new_water
        gained = int(gain.sum())
        lost = int(loss.sum())
        result = np.full(common.shape, MASK_NODATA, dtype=np.uint8)
        result[common] = 0
        result[stable] = 1
        result[gain] = 2
        result[loss] = 3
        file_out = PROCESSED / f"22_s2_change_{year_a}_{year_b}_10m.tif"
        write_mask(file_out, result, a["profile"], "Change from year A to year B (common-valid pixels only)",
                   {0: "stable_nonwater", 1: "stable_water", 2: "water_gain", 3: "water_loss",
                    255: "invalid_or_outside_common_valid"})
        pct = coverage_pct(n_common, corridor_pixels)
        qa_flags = []
        if pct < MIN_PAIR_COMMON_COVERAGE_PCT:
            qa_flags.append("LOW_COMMON_COVERAGE")
        date_a = datetime.strptime(a["date"], "%Y%m%d")
        date_b = datetime.strptime(b["date"], "%Y%m%d")
        if abs((date_a.month - date_b.month)) > 1:
            qa_flags.append("DIFFERENT_ACQUISITION_MONTH")
        if not qa_flags:
            qa_flags.append("EXPLORATORY_ONLY")
        row = {
            "year_a": year_a, "date_a": a["date"], "year_b": year_b, "date_b": b["date"],
            "common_valid_pixels": n_common, "corridor_pixels": corridor_pixels,
            "common_valid_pct": pct, "water_a_ha_common_valid": round(float(old_water.sum() * pixel_area_ha), 3),
            "water_b_ha_common_valid": round(float(new_water.sum() * pixel_area_ha), 3),
            "stable_water_ha": round(float(stable.sum() * pixel_area_ha), 3),
            "gain_ha": round(float(gained * pixel_area_ha), 3),
            "loss_ha": round(float(lost * pixel_area_ha), 3),
            "net_water_ha": round(float((gained - lost) * pixel_area_ha), 3),
            "qa_status": "+".join(qa_flags), "change_raster": file_out.name,
            "interpretation": "SATELLITE_WATER_FOOTPRINT_NOT_VALIDATED_BANK_EROSION"
        }
        pairs.append(row)
        print(f"{year_a} -> {year_b}: common {pct:.2f}% | gain {row['gain_ha']} ha | "
              f"loss {row['loss_ha']} ha | {row['qa_status']}")
        if (year_a, year_b) in [(2018, 2024), (2018, 2025)]:
            figfile = FIGURES / f"22_s2_change_{year_a}_{year_b}.png"
            plot_pair(year_a, year_b, scenes, row, figfile)
            print("    PNG:", figfile)

    table(REPORTS / "22_s2_annual_water_qa.csv", annual)
    table(REPORTS / "22_s2_pairwise_change_qa.csv", pairs)
    print("\nREPORTS:")
    print(REPORTS / "22_s2_annual_water_qa.csv")
    print(REPORTS / "22_s2_pairwise_change_qa.csv")
    print("\nIMPORTANT: No synthetic observational pixels, no erosion labels, no validated bankline claim.")
    print("STEP 22 COMPLETE — REVIEW QA AND VISUALS BEFORE ANY MORPHOLOGY CLAIM")


if __name__ == "__main__":
    main()
