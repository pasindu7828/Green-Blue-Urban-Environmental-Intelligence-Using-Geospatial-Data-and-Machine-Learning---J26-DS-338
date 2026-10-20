"""Step 23: Sentinel-2 matched-domain QA and reach-wise water-footprint review.

Run AFTER Step 22 (07_sentinel2_multiyear_change.py). Reads only its saved
classified water masks, QA CSVs, and the original centreline geometry.
NEVER fills invalid pixels or labels observed water as bank erosion.

Outputs (new files, no original input overwrite):
  outputs/reports/23_s2_matched_domain_audit.csv
  outputs/reports/23_s2_reachwise_audit.csv
  outputs/figures/23_s2_2018_2024_spatial_qa.png
  outputs/figures/23_s2_2018_2025_spatial_qa.png

Methods:
  consensus water = class 1 in Step 22 (both NDWI and MNDWI > 0)
  valid = classes 0/1/2/3; nodata/outside = 255
  gains/losses measured ONLY on common-valid pixels
  A-only and B-only observation water explicitly separate and NEVER called change
  500m reaches APPROXIMATED by proximity to 5m-spaced digitized centreline
  QA flags are exploratory heuristics, not calibrated confidence or risk scores.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import rasterio
import geopandas as gpd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
MASKS = ROOT / 'data' / 'processed' / 'sentinel2_change_v2'
CENTRELINE = ROOT / 'data' / 'raw' / 'boundaries' / 'kelani_river_centreline.geojson'
REPORTS = ROOT / 'outputs' / 'reports'
FIGURES = ROOT / 'outputs' / 'figures'
PAIRS = [(2018, 2024), (2018, 2025)]
REACH_LENGTH_M = 500.0  # Must agree with Step 07 reach IDs.
CENTERLINE_SAMPLING_M = 5.0


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Prerequisite missing: {path}')
    with path.open(encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError('Refusing to export empty QA table')
    with path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_water_mask(year: int):
    path = MASKS / f'22_s2_{year}_water_agreement_10m.tif'
    if not path.is_file():
        raise FileNotFoundError(f'Run Step 22 first: missing {path}')
    with rasterio.open(path) as ds:
        if ds.crs is None or ds.nodata != 255 or ds.count != 1:
            raise ValueError(f'Unexpected metadata in {path}')
        arr = ds.read(1)
        if np.any(~np.isin(arr, [0, 1, 2, 3, 255])):
            raise ValueError(f'Unrecognized mask classes for year {year}')
        meta = (ds.crs, ds.transform, ds.height, ds.width)
    return arr, meta


def pair_masks(arr_a, arr_b):
    va, vb = arr_a != 255, arr_b != 255
    wa, wb = arr_a == 1, arr_b == 1
    common = va & vb
    stable = common & wa & wb
    gain = common & ~wa & wb
    loss = common & wa & ~wb
    a_only_water = wa & ~vb
    b_only_water = wb & ~va
    return dict(va=va, vb=vb, wa=wa, wb=wb, common=common,
                stable=stable, gain=gain, loss=loss,
                a_only_water=a_only_water, b_only_water=b_only_water)


def nearest_reach_map(meta, domain):
    crs, transform, h, w = meta
    if not CENTRELINE.is_file():
        raise FileNotFoundError(f'Missing centreline: {CENTRELINE}')
    gdf = gpd.read_file(CENTRELINE)
    if gdf.empty or gdf.crs is None or len(gdf) != 1:
        raise ValueError('Expected a single, projected-capable digitized centreline')
    line = gdf.to_crs(crs).geometry.iloc[0]
    if line.geom_type != 'LineString' or line.length < REACH_LENGTH_M:
        raise ValueError('Unexpected centreline geometry or length')
    sample_dist = np.arange(0.0, line.length + CENTERLINE_SAMPLING_M, CENTERLINE_SAMPLING_M)
    sample_dist = np.minimum(sample_dist, line.length)
    sample_pts = [line.interpolate(float(d)) for d in sample_dist]
    kd = cKDTree(np.array([(p.x, p.y) for p in sample_pts], dtype=float))

    # Allocate reach identifiers only for the union of relevant valid pixels.
    rr, cc = np.nonzero(domain)
    xs, ys = rasterio.transform.xy(transform, rr, cc, offset='center')
    pts = np.column_stack((np.asarray(xs), np.asarray(ys)))
    _, nearest = kd.query(pts, k=1, workers=1)
    n_reaches = int(np.ceil(line.length / REACH_LENGTH_M))
    indices = np.minimum((sample_dist[nearest] // REACH_LENGTH_M).astype(np.int16), n_reaches - 1)
    reach_map = np.full((h, w), -1, dtype=np.int16)
    reach_map[rr, cc] = indices
    return reach_map, n_reaches, float(line.length), line


def metric(mask, region, pixel_ha):
    return round(float(np.count_nonzero(mask & region) * pixel_ha), 3)


def audit_row(a, b, m, area_ha, region, coverage_denominator, row_label, month_mismatch=False):
    mask = region
    common = int(np.count_nonzero(m['common'] & mask))
    a_valid = int(np.count_nonzero(m['va'] & mask))
    b_valid = int(np.count_nonzero(m['vb'] & mask))
    water_a = int(np.count_nonzero(m['wa'] & mask))
    water_b = int(np.count_nonzero(m['wb'] & mask))
    gain = metric(m['gain'], mask, area_ha)
    loss = metric(m['loss'], mask, area_ha)
    aonly = metric(m['a_only_water'], mask, area_ha)
    bonly = metric(m['b_only_water'], mask, area_ha)
    annual_delta = round((water_b - water_a) * area_ha, 3)
    matched_delta = round(gain - loss, 3)
    missing_domain_effect = round(bonly - aonly, 3)
    # Identity checks: annual delta = matched delta + B-only - A-only.
    if abs(annual_delta - matched_delta - missing_domain_effect) > (area_ha * 2 + 1e-4):
        raise AssertionError('Matched-domain decomposition not conserved')
    observed_water_union = np.count_nonzero((m['wa'] | m['wb']) & mask)
    matched_water_union = np.count_nonzero((m['wa'] | m['wb']) & m['common'] & mask)
    water_domain_coverage_pct = (round(100 * matched_water_union / observed_water_union, 2)
                                 if observed_water_union else None)
    common_pct = round(100 * common / coverage_denominator, 2) if coverage_denominator else 0.0
    flags = []
    if common_pct < 80:
        flags.append('LOW_COMMON_COVERAGE')
    # Interpretation safeguard: missing-domain effect materially affects a change comparison.
    if abs(missing_domain_effect) >= 2.0 and abs(missing_domain_effect) >= 0.25 * max(1.0, abs(matched_delta)):
        flags.append('MISSING_DOMAIN_EFFECT')
    if water_domain_coverage_pct is not None and water_domain_coverage_pct < 85:
        flags.append('LOW_WATER_DOMAIN_COVERAGE')
    if month_mismatch:
        flags.append('DIFFERENT_ACQUISITION_MONTH')
    if not flags:
        flags.append('EXPLORATORY_REVIEW')
    return {
        'year_a': a, 'year_b': b, 'region_id': row_label,
        'domain_pixels': coverage_denominator, 'common_valid_pixels': common,
        'common_valid_pct': common_pct,
        'year_a_valid_pixels': a_valid, 'year_b_valid_pixels': b_valid,
        'year_a_annual_water_ha': round(water_a * area_ha, 3),
        'year_b_annual_water_ha': round(water_b * area_ha, 3),
        'annual_total_delta_ha_not_directly_comparable': annual_delta,
        'matched_water_gain_ha': gain, 'matched_water_loss_ha': loss,
        'matched_net_change_ha': matched_delta,
        'year_a_water_where_year_b_invalid_ha': aonly,
        'year_b_water_where_year_a_invalid_ha': bonly,
        'unmatched_domain_effect_ha': missing_domain_effect,
        'observed_water_common_domain_pct': water_domain_coverage_pct,
        'qa_flags': ';'.join(flags),
        'interpretation': 'WATER_FOOTPRINT_EXPLORATORY_NOT_EROSION',
    }


def plot_report(a, b, m, region_ids, reach_rows, global_row, line, meta, outpath):
    # 0: outside all observed domain; 1: stable nonwater; 2 stable water; 3 gain;
    # 4 loss; 5 A-only-water (B invalid); 6 B-only-water (A invalid).
    code = np.zeros(m['va'].shape, dtype=np.uint8)
    code[m['common']] = 1
    code[m['stable']] = 2
    code[m['gain']] = 3
    code[m['loss']] = 4
    code[m['a_only_water']] = 5
    code[m['b_only_water']] = 6
    palette = ['#eeeeee', '#e7e8e0', '#2473b5', '#05b68d', '#f05a28', '#9068a6', '#d9a32c']
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(-0.5, 7, 1), cmap.N)
    fig = plt.figure(figsize=(17.5, 8.4), facecolor='white')
    grid = fig.add_gridspec(2, 1, height_ratios=[2.1, 1], left=0.055, right=0.985,
                            top=0.78, bottom=0.18, hspace=0.12)
    ax = fig.add_subplot(grid[0, 0]); ax.imshow(code, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title('Spatial water-footprint classification (on observed pixels only)', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    # Label selected reaches in projected coordinate space.
    transform = meta[1]
    for index in range(0, len(reach_rows), 4):
        p = line.interpolate(min(index * REACH_LENGTH_M + 250, line.length))
        c, r = (~transform) * (p.x, p.y)
        ax.text(c, r, f'R{index+1:03d}', color='black', fontsize=8,
                bbox={'facecolor':'white','alpha':0.8,'edgecolor':'none','pad':1.5})
    bars = fig.add_subplot(grid[1, 0])
    xx = np.arange(len(reach_rows))
    gains = [x['matched_water_gain_ha'] for x in reach_rows]
    losses = [x['matched_water_loss_ha'] for x in reach_rows]
    invalid_b = [x['year_b_water_where_year_a_invalid_ha'] for x in reach_rows]
    bars.bar(xx-0.27, gains, width=0.27, color=palette[3], label='Gain (matched)')
    bars.bar(xx, losses, width=0.27, color=palette[4], label='Loss (matched)')
    bars.bar(xx+0.27, invalid_b, width=0.27, color=palette[6], label=f'{b} water; {a} invalid')
    bars.set_xticks(xx, [f'R{i+1:02d}' for i in xx], rotation=60, fontsize=7)
    bars.set_ylabel('Observed water area (ha)')
    bars.grid(axis='y', alpha=0.2)
    bars.legend(loc='upper right', fontsize=9, ncol=3, frameon=False)
    fig.suptitle(f'KELANI RIVER CORRIDOR | SPATIAL QA | {a} → {b}', fontsize=18, weight='bold', y=0.98)
    fig.text(0.5, 0.915,
             f'Matched valid: {global_row["common_valid_pct"]}%  •  Matched gain: {global_row["matched_water_gain_ha"]} ha'
             f'  •  Matched loss: {global_row["matched_water_loss_ha"]} ha'
             f'  •  {b} observed water where {a} invalid: {global_row["year_b_water_where_year_a_invalid_ha"]} ha',
             ha='center', fontsize=11)
    handles = [Patch(facecolor=palette[i], label=label) for i, label in [(0,'Unobserved / outside'),(2,'Stable water'),
                (3,'Matched water gain'),(4,'Matched water loss'),(5,f'{a} water, {b} invalid'),(6,f'{b} water, {a} invalid')]]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5,0.085), ncol=3, frameon=False, fontsize=9)
    fig.text(0.5,0.032, 'Exploratory spectral water footprint, NOT measured riverbank erosion. '
             '500 m reaches approximated from candidate centreline; mismatched observation domains explicitly flagged.',
             ha='center', fontsize=9, color='#555555')
    fig.savefig(outpath, dpi=160, facecolor='white')
    plt.close(fig)


def main():
    annual = read_csv(REPORTS / '22_s2_annual_water_qa.csv')
    pairs_qa = read_csv(REPORTS / '22_s2_pairwise_change_qa.csv')
    known_years = {int(row['year']) for row in annual}
    observation_dates = {int(row['year']): row.get('observation_date') for row in annual}
    if not all(a in known_years and b in known_years for a,b in PAIRS):
        raise ValueError('Step 22 annual QA does not cover requested pairs')
    rasters = {}; meta0 = None
    for year in sorted({y for p in PAIRS for y in p}):
        array, meta = read_water_mask(year)
        if meta0 is None:
            meta0 = meta
        elif meta != meta0:
            raise ValueError(f'Raster grids differ for year {year}; cannot compare without alignment')
        rasters[year] = array
    crs, transform, h, w = meta0
    pixel_ha = abs(transform.a * transform.e - transform.b * transform.d) / 10000
    # Step22 mask uses value 255 for outside OR invalid; corridor polygon is the
    # ground coverage denominator. Use the supplied constant corridor_pixels count
    # for the global QA, avoiding incorrect domain inference from valid union.
    domain_pixels = {int(row['year']):int(row['corridor_pixels']) for row in annual}
    if len(set(domain_pixels.values())) != 1:
        raise ValueError('Step 22 annual domains not identical')
    all_domain_count = next(iter(domain_pixels.values()))
    # For reach assignment, union of valid pixels may omit totally invalid patches.
    # Rasterize original corridor separately so denominator is meaningful.
    from rasterio.features import geometry_mask
    corridor = gpd.read_file(ROOT / 'data' / 'raw' / 'boundaries' / 'kelani_river_corridor.geojson')
    if corridor.crs is None or corridor.empty:
        raise ValueError('Missing corridor CRS or geometry')
    inside = geometry_mask(corridor.to_crs(crs).geometry.to_list(), out_shape=(h,w),
                           transform=transform, invert=True, all_touched=False)
    if int(inside.sum()) != all_domain_count:
        raise ValueError('Corridor rasterization disagrees with Step 22, stop rather than guess')
    reach_map, n_reaches, length, line = nearest_reach_map(meta0, inside)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    globals_out = []; reaches_out = []
    print(f'Centreline: {length:.2f} m  |  Approximate reaches: {n_reaches}  |  Pixel: {pixel_ha:.5f} ha')
    if n_reaches != 28:
        print('WARNING: previous transect layout had 28 reaches. Verify orientation and centreline source.')
    for a,b in PAIRS:
        m = pair_masks(rasters[a], rasters[b])
        da, db = observation_dates[a], observation_dates[b]
        if not da or not db or len(da) != 8 or len(db) != 8:
            raise ValueError('Step 22 annual QA missing observation_date YYYYMMDD')
        month_mismatch = da[4:6] != db[4:6]
        global_row = audit_row(a,b,m,pixel_ha,inside,all_domain_count,'ALL',month_mismatch)
        actual = next((r for r in pairs_qa if int(r['year_a'])==a and int(r['year_b'])==b),None)
        if actual is None:
            raise ValueError(f'Missing Step 22 pair row {a}->{b}')
        for col, expected in [('matched_water_gain_ha', 'gain_ha'),('matched_water_loss_ha','loss_ha')]:
            if abs(float(global_row[col]) - float(actual[expected])) > max(0.02,2*pixel_ha):
                raise AssertionError(f'Step 23 {col} differs from Step 22 for {a}->{b}: inspect before proceeding')
        global_row.update(centreline_length_m=round(length,2),
                          reach_length_m=REACH_LENGTH_M,
                          reach_method='5m_nearest_centreline_point_APPROXIMATE')
        globals_out.append(global_row)
        subrows=[]
        for ri in range(n_reaches):
            region = reach_map == ri
            subrow = audit_row(a,b,m,pixel_ha,region,int(region.sum()),f'R{ri+1:03d}',month_mismatch)
            subrow['reach_start_m']=round(ri*REACH_LENGTH_M,1)
            subrow['reach_end_m']=round(min((ri+1)*REACH_LENGTH_M,length),1)
            subrows.append(subrow)
            reaches_out.append(subrow)
        for col in ['matched_water_gain_ha','matched_water_loss_ha','year_b_water_where_year_a_invalid_ha']:
            if abs(round(sum(r[col] for r in subrows),3) - float(global_row[col])) > 0.05:
                raise AssertionError(f'Reach totals do not match global for {a}->{b}, field={col}')
        output = FIGURES / f'23_s2_{a}_{b}_spatial_qa.png'
        plot_report(a,b,m,reach_map,subrows,global_row,line,meta0,output)
        print(f'\n{a} -> {b} | matched gain {global_row["matched_water_gain_ha"]}ha | matched loss '
              f'{global_row["matched_water_loss_ha"]}ha | {b}-water/{a}-invalid '
              f'{global_row["year_b_water_where_year_a_invalid_ha"]}ha')
        print(f'  Annual net {global_row["annual_total_delta_ha_not_directly_comparable"]}ha '
              f'vs matched net {global_row["matched_net_change_ha"]}ha | QA: {global_row["qa_flags"]}')
        print('  Figure:', output)
    write_csv(REPORTS / '23_s2_matched_domain_audit.csv', globals_out)
    write_csv(REPORTS / '23_s2_reachwise_audit.csv', reaches_out)
    print('\nSTEP 23 COMPLETE — REACH-WISE QA ONLY, NOT EROSION OR RISK VALIDATION')

if __name__ == '__main__':
    main()
