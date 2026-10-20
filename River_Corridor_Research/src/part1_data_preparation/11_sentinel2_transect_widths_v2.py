"""STEP 27B.3 -- Sentinel-2 apparent active-water-channel width along fixed transects.

Source of truth: Step-22 CONSENSUS water agreement classification raster, code 1.
Other raster categories: 0=nonwater, 2=NDWI only, 3=MNDWI only,
255=invalid OR outside the study corridor. Categories 2/3 are NOT water in the
strict consensus rule. The script never bridges 0/2/3/255 gaps.

Width is computed by exact intersection length of a fixed vector transect with
10 m classified pixel cells, selecting only the contiguous water run containing
the nominal river-centreline position (transect midpoint). It does NOT imply
subpixel positional accuracy or independently verified banklines. It also
cannot establish physical erosion/accretion, water stage, or legal encroachment.

Inputs: data/processed/transects/01_transects_50m.gpkg
        data/processed/sentinel2_change_v2/22_s2_YYYY_water_agreement_10m.tif
Outputs: outputs/reports/27_s2_transect_widths_2018_2025.csv
         outputs/reports/27_s2_transect_width_change_pairs.csv
         outputs/reports/27_s2_transect_width_qa_summary.csv
         outputs/reports/27_s2_width_methodology.txt
         outputs/figures/27_s2_width_change_{2018_2024,2020_2024}.png

No modification of existing data/models/backend/frontend.
"""
from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

YEARS = tuple(range(2018, 2026))
NODATA = 255
MIN_WIDTH_M = 20.0  # screening heuristic, NOT precision/validation assertion
TARGET_PAIRS = ((2018, 2024), (2020, 2024), (2018, 2025), (2024, 2025))


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boundary_ts(start: float, delta: float, grid_origin: float, step: float) -> list[float]:
    """t in (0,1) where segment crosses a 10m pixel grid line."""
    if abs(delta) < 1e-12:
        return []
    endpoint = start + delta
    a, b = (start - grid_origin) / step, (endpoint - grid_origin) / step
    result = []
    for k in range(math.floor(min(a, b)) - 1, math.ceil(max(a, b)) + 2):
        edge = grid_origin + k * step
        t = (edge - start) / delta
        if 1e-11 < t < 1 - 1e-11:
            result.append(t)
    return result


def calculate_width(line, mask: np.ndarray, transform, resolution=10.0) -> dict:
    coords = list(line.coords)
    if len(coords) != 2:
        raise ValueError("Expected straight 2-vertex transect, inspect geometry before use")
    (x0, y0), (x1, y1) = coords
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if not (length > 0):
        raise ValueError("Zero length transect")

    # Whole transect is subdivided exactly at boundaries of the raster grid cells.
    # Pixel colours are NOT interpolated or smoothed.
    cuts = [0.0, 1.0]
    cuts += boundary_ts(x0, dx, transform.c, transform.a)
    cuts += boundary_ts(y0, dy, transform.f, transform.e)
    cuts = sorted(set(round(t, 12) for t in cuts))
    cuts[0], cuts[-1] = 0.0, 1.0

    samples = []  # (t0, t1, class)
    height, width = mask.shape
    for t0, t1 in zip(cuts[:-1], cuts[1:]):
        if t1 - t0 < 1e-11:
            continue
        tm = (t0 + t1) / 2
        x, y = x0 + tm * dx, y0 + tm * dy
        col = math.floor((x - transform.c) / transform.a)
        row = math.floor((y - transform.f) / transform.e)
        label = int(mask[row, col]) if (0 <= row < height and 0 <= col < width) else NODATA
        if label not in (0, 1, 2, 3, 255):
            raise ValueError(f"Unexpected raster category {label}")
        samples.append((t0, t1, label))

    valid_m = sum((b - a) * length for a, b, c in samples if c != NODATA)
    consensus_m = sum((b - a) * length for a, b, c in samples if c == 1)
    disagreement_m = sum((b - a) * length for a, b, c in samples if c in (2, 3))

    # Contiguous runs of water along transect. Pixel boundaries between adjacent
    # consensus-water cells don't split the run.
    runs = []
    start_i = None
    for idx, (_, _, category) in enumerate(samples):
        if category == 1:
            if start_i is None:
                start_i = idx
        elif start_i is not None:
            runs.append((start_i, idx - 1))
            start_i = None
    if start_i is not None:
        runs.append((start_i, len(samples) - 1))

    result = {
        "transect_length_m": round(length, 3),
        "observed_valid_transect_pct": round(valid_m / length * 100, 2),
        "all_consensus_water_intersection_m": round(consensus_m, 3),
        "index_disagreement_intersection_m": round(disagreement_m, 3),
        "water_run_count": len(runs),
        "centre_class": next((c for a, b, c in samples if a <= .5 < b), NODATA),
        "nearest_water_offset_m": None,
        "nearest_water_run_width_m": None,
        "apparent_width_raw_m": None,
        "apparent_width_screened_m": None,
        "start_side_water_edge_offset_m": None,
        "end_side_water_edge_offset_m": None,
        "qa_pass": 0,
        "qa_status": None,
        "qa_flags": None,
    }
    if not runs:
        result.update(qa_status="NO_CONSENSUS_WATER", qa_flags="NO_WATER_RUN")
        return result

    def run_offset(ij):
        i, j = ij
        a, b = samples[i][0], samples[j][1]
        return max(0., a - .5, .5 - b) * length

    i, j = min(runs, key=run_offset)
    a, b = samples[i][0], samples[j][1]
    offset = run_offset((i, j))
    apparent_width = (b - a) * length
    result["nearest_water_offset_m"] = round(offset, 3)
    result["nearest_water_run_width_m"] = round(apparent_width, 3)
    if offset > .001:
        # A nearby pond / side channel cannot be declared the main river merely
        # because it is geographically closest to the digitized centreline.
        status = "CENTRELINE_NOT_IN_CONSENSUS_WATER"
        if result["centre_class"] == NODATA:
            status = "CENTRELINE_INVALID_OR_OUTSIDE"
        result.update(qa_status=status, qa_flags="CANDIDATE_RUN_NOT_USED")
        return result

    result["apparent_width_raw_m"] = round(apparent_width, 3)
    result["start_side_water_edge_offset_m"] = round((a - .5) * length, 3)
    result["end_side_water_edge_offset_m"] = round((b - .5) * length, 3)
    flags = []
    if a <= 1e-10 or b >= 1 - 1e-10:
        flags.append("WIDTH_TRUNCATED_BY_TRANSECT")
    left_label = samples[i - 1][2] if i > 0 else None
    right_label = samples[j + 1][2] if j + 1 < len(samples) else None
    if left_label == NODATA or right_label == NODATA:
        flags.append("WATER_EDGE_ADJACENT_TO_INVALID")
    if left_label in (2, 3) or right_label in (2, 3):
        flags.append("WATER_EDGE_INDEX_DISAGREEMENT")
    if apparent_width < MIN_WIDTH_M:
        flags.append("NARROWER_THAN_20M_SCREEN")
    blocking = {"WIDTH_TRUNCATED_BY_TRANSECT", "WATER_EDGE_ADJACENT_TO_INVALID", "NARROWER_THAN_20M_SCREEN"}
    screened = not any(flag in blocking for flag in flags)
    result["qa_pass"] = int(screened)
    result["apparent_width_screened_m"] = round(apparent_width, 3) if screened else None
    result["qa_status"] = "PASS_EXPLORATORY" if screened else "SCREENED_OUT"
    result["qa_flags"] = ";".join(flags) if flags else "EXPLORATORY_NO_QA_FLAG"
    return result


def plot_pair(a, b, change_rows, transects, centreline, out: Path):
    selected = [r for r in change_rows if r["year_a"] == a and r["year_b"] == b]
    vals = np.array([float(r["apparent_width_delta_m"]) if r["qa_matched"] else np.nan for r in selected])
    x = transects["centre_x"].to_numpy(dtype=float)
    y = transects["centre_y"].to_numpy(dtype=float)
    valid = np.isfinite(vals)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2), gridspec_kw={"width_ratios": [1.6, 1.]})
    fig.patch.set_facecolor("white")
    ax = axes[0]
    if centreline is not None:
        centreline.plot(ax=ax, color="#808b95", linewidth=2, zorder=1)
    ax.scatter(x[~valid], y[~valid], s=10, color="#d0d5da", label="QA unavailable", zorder=2)
    if valid.any():
        lim = max(20, float(np.nanpercentile(np.abs(vals[valid]), 95)))
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        scatter = ax.scatter(x[valid], y[valid], c=np.clip(vals[valid], -lim, lim),
                             s=24, cmap="RdBu_r", norm=norm, edgecolors="none", zorder=3)
        cb = fig.colorbar(scatter, ax=ax, shrink=.75, pad=.02)
        cb.set_label("Apparent width difference (m), positive = wider classified water")
    ax.set_title(f"Matched transect width differences | {a} to {b}")
    ax.set_xlabel("Easting (m), EPSG:32644")
    ax.set_ylabel("Northing (m), EPSG:32644")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=.15)

    ax2 = axes[1]
    group = {}
    for r in selected:
        if r["qa_matched"]:
            group.setdefault(r["reach_id"], []).append(float(r["apparent_width_delta_m"]))
    keys = sorted(set(r["reach_id"] for r in selected))
    medians = [float(np.median(group[k])) if len(group.get(k, [])) >= 2 else np.nan for k in keys]
    yy = np.arange(len(keys))
    ax2.barh(yy[np.isfinite(medians)], np.array(medians)[np.isfinite(medians)], color="#467c9b")
    ax2.set_yticks(yy)
    ax2.set_yticklabels(keys, fontsize=7)
    ax2.invert_yaxis()
    ax2.axvline(0, color="#4c5760", linewidth=.8)
    ax2.set_xlabel("Reach median apparent width difference (m)")
    ax2.set_title("Reaches with >=2 matched transects")
    ax2.grid(axis="x", alpha=.2)
    fig.suptitle(f"KELANI RIVER | SENTINEL-2 EXPLORATORY TRANSECT WIDTH | {a} - {b}",
                 fontsize=14, weight="bold", y=.99)
    n = int(valid.sum())
    fig.text(.5, .028,
             f"Matched: {n}/{len(selected)} | Strict NDWI+MNDWI agreement (10m grid; SWIR native 20m). "
             "Not validated physical riverbank change. Water stage / season / index sensitivity may dominate.",
             ha="center", fontsize=8.2, color="#444444")
    fig.tight_layout(rect=(0, .052, 1, .94), w_pad=3)
    fig.savefig(out, dpi=175, facecolor="white")
    plt.close(fig)


def run(root: Path):
    root = root.resolve()
    tran_path = root / "data/processed/transects/01_transects_50m.gpkg"
    mask_dir = root / "data/processed/sentinel2_change_v2"
    report_dir = root / "outputs/reports"
    figure_dir = root / "outputs/figures"
    if not tran_path.is_file():
        raise FileNotFoundError(tran_path)
    transects = gpd.read_file(tran_path)
    required = {"transect_id", "reach_id", "chainage_m", "centre_x", "centre_y", "geometry"}
    if not required.issubset(transects.columns) or transects.crs is None:
        raise ValueError(f"Transect attributes / CRS missing: {required - set(transects.columns)}")
    if transects.empty or transects.transect_id.duplicated().any():
        raise ValueError("Missing transects or duplicated transect ID")
    if not transects.geometry.geom_type.eq("LineString").all():
        raise ValueError("Unsupported transect geometry type")
    if not transects.geometry.is_valid.all() or transects.geometry.is_empty.any():
        raise ValueError("Invalid / empty transect geometry")
    transects = transects.sort_values("transect_id").reset_index(drop=True)

    # Validate all raster files and grids before any output is written.
    paths = {year: mask_dir / f"22_s2_{year}_water_agreement_10m.tif" for year in YEARS}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    masks = {}
    ref = None
    for year, path in paths.items():
        with rasterio.open(path) as ds:
            if ds.crs is None or ds.crs.to_string() != "EPSG:32644":
                raise ValueError(f"Unexpected CRS in {path}: {ds.crs}")
            if ds.count != 1 or ds.dtypes[0] != "uint8" or ds.nodata != 255:
                raise ValueError(f"Unexpected mask format in {path}")
            if not np.allclose(ds.res, (10., 10.)):
                raise ValueError(f"Unexpected resolution in {path}")
            if abs(ds.transform.b) > 1e-9 or abs(ds.transform.d) > 1e-9:
                raise ValueError("Rotated rasters not supported; must align explicitly")
            grid = (ds.width, ds.height, ds.crs.to_string(), tuple(ds.transform))
            if ref is None:
                ref = grid
            elif grid != ref:
                raise ValueError(f"Raster grid mismatch for year {year}")
            masks[year] = (ds.read(1), ds.transform)
    transects = transects.to_crs("EPSG:32644")
    for transect in transects.itertuples(index=False):
        midpoint = transect.geometry.interpolate(0.5, normalized=True)
        if math.hypot(midpoint.x - float(transect.centre_x), midpoint.y - float(transect.centre_y)) > 0.05:
            raise ValueError(f"Transect midpoint is not at stored centre: {transect.transect_id}")
    if not np.allclose(transects.geometry.length.to_numpy(), 300., atol=.05):
        raise ValueError("Expected original 300m transects; inspect before sampling")

    annual_qa_path = root / "outputs/reports/22_s2_annual_water_qa.csv"
    if not annual_qa_path.is_file():
        raise FileNotFoundError(f"Need Step-22 acquisition dates: {annual_qa_path}")
    with annual_qa_path.open(newline="", encoding="utf-8-sig") as stream:
        date_rows = list(csv.DictReader(stream))
    obs_dates = {int(r["year"]): datetime.strptime(r["observation_date"], "%Y%m%d") for r in date_rows}
    if any(year not in obs_dates for year in YEARS):
        raise ValueError("Missing Sentinel-2 acquisition dates in Step-22 annual QA report")

    centre_path = root / "data/raw/boundaries/kelani_river_centreline.geojson"
    centreline = None
    if centre_path.is_file():
        centreline = gpd.read_file(centre_path).to_crs("EPSG:32644")

    print("STEP 27B.3 | Sentinel-2 apparent width extraction, pixel-cell intersections")
    print(f"Transects: {len(transects)} | Years: {len(YEARS)} | Grid: 10m | strict water=1")
    annual = []
    for year in YEARS:
        mask, transform = masks[year]
        pass_count = 0
        for item in transects.itertuples(index=False):
            measure = calculate_width(item.geometry, mask, transform)
            row = {
                "year": year,
                "transect_id": item.transect_id,
                "reach_id": item.reach_id,
                "chainage_m": round(float(item.chainage_m), 3),
                "centre_x": round(float(item.centre_x), 3),
                "centre_y": round(float(item.centre_y), 3),
                **measure,
                "interpretation": "SATELLITE_WATER_WIDTH_PROXY_NOT_SURVEYED_RIVERBANK",
            }
            annual.append(row)
            pass_count += measure["qa_pass"]
        print(f"  {year}: strict centre-anchored pilot widths {pass_count}/{len(transects)}")

    by_id_year = {(row["transect_id"], row["year"]): row for row in annual}
    pair_list = [(a, b) for a, b in zip(YEARS[:-1], YEARS[1:])]
    pair_list += [pair for pair in TARGET_PAIRS if pair not in pair_list]
    change = []
    summary = []
    for a, b in pair_list:
        good_deltas = []
        for row in transects.itertuples(index=False):
            ra, rb = by_id_year[(row.transect_id, a)], by_id_year[(row.transect_id, b)]
            matched = bool(ra["qa_pass"] and rb["qa_pass"])
            delta = (rb["apparent_width_screened_m"] - ra["apparent_width_screened_m"]) if matched else None
            if matched:
                good_deltas.append(delta)
            change.append({
                "year_a": a, "year_b": b,
                "transect_id": row.transect_id, "reach_id": row.reach_id,
                "chainage_m": round(float(row.chainage_m), 3),
                "width_a_m": ra["apparent_width_screened_m"],
                "width_b_m": rb["apparent_width_screened_m"],
                "apparent_width_delta_m": round(delta, 3) if matched else None,
                "qa_matched": int(matched),
                "qa_a": ra["qa_status"], "qa_b": rb["qa_status"],
                "acquisition_month_difference_caution": int(obs_dates[a].month != obs_dates[b].month),
                "interpretation": "APPARENT_DETECTED_WATER_WIDTH_DIFFERENCE_NOT_VALIDATED_EROSION",
            })
        arr = np.array(good_deltas, dtype=float)
        summary.append({
            "year_a": a, "year_b": b,
            "total_transects": len(transects), "qa_matched_transects": len(arr),
            "matched_fraction_pct": round(len(arr) / len(transects) * 100, 2),
            "median_apparent_width_delta_m": round(float(np.median(arr)), 3) if len(arr) else None,
            "mean_apparent_width_delta_m": round(float(np.mean(arr)), 3) if len(arr) else None,
            "count_apparent_increase_gt20m": int((arr > 20.).sum()),
            "count_apparent_decrease_lt_minus20m": int((arr < -20.).sum()),
            "qa_comment": "PROVISIONAL_WATER_WIDTH_NOT_PHYSICAL_BANK_MOVEMENT",
        })
        print(f"  {a}->{b}: matched {len(arr)}/{len(transects)}, median delta "
              f"{float(np.median(arr)):.2f}m" if len(arr) else f"  {a}->{b}: NO MATCHED OBSERVATIONS")

    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_annual = report_dir / "27_s2_transect_widths_2018_2025.csv"
    csv_change = report_dir / "27_s2_transect_width_change_pairs.csv"
    csv_summary = report_dir / "27_s2_transect_width_qa_summary.csv"
    save_csv(csv_annual, annual)
    save_csv(csv_change, change)
    save_csv(csv_summary, summary)
    for a, b in ((2018, 2024), (2020, 2024)):
        fig = figure_dir / f"27_s2_width_change_{a}_{b}.png"
        plot_pair(a, b, change, transects, centreline, fig)
        print("  Figure:", fig)
    notes = report_dir / "27_s2_width_methodology.txt"
    notes.write_text(
        "STEP 27B.3 - UNVALIDATED SENTINEL-2 WIDTH SCREENING\n"
        "Input water class=1 means BOTH NDWI>0 and MNDWI>0 in valid pixels.\n"
        "Native 20m SWIR1 was resampled onto the 10m analysis grid in Step22.\n"
        "Transect intersections with 10m cells are measured geometrically.\n"
        "The contiguous water run MUST contain the digitized centreline midpoint.\n"
        "Never bridge invalid, disagreement or nonwater pixels to create width.\n"
        "QA requires width >=20m, no transect truncation, both apparent edges observable.\n"
        "Adjacent NDWI-only/MNDWI-only pixels are flagged as threshold uncertainty.\n"
        "The 20m minimum is an exploratory screening heuristic, not validated accuracy.\n"
        "The output can have decimal metres from vector-grid intersections, which DO NOT imply\n"
        "subpixel precision: spectral bands include native 20m SWIR and water stage differences.\n"
        "Missing water, disconnected / offset water, invalid observations are kept as NA.\n"
        "This is apparent WATER width, not surveyed bankline, legal encroachment, or erosion.\n"
        "DO NOT train ML or claim widening until QA and manual visual checks pass.\n",
        encoding="utf-8")
    print("\nOUTPUTS:")
    for p in (csv_annual, csv_change, csv_summary, notes):
        print(" ", p)
    print("STEP 27B.3 COMPLETE - REVIEW QA COVERAGE BEFORE ANY MORPHOLOGY CLAIM")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                   help="River_Corridor_Research root (defaults to path inferred from script location)")
    args = p.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
