"""Step 24: Sentinel-2 river-proximity and water-index robustness screening.

READ ONLY:
  Step22 annual NDWI/MNDWI agreement rasters (classes 0,1,2,3,255)
  Original river centreline and corridor geometries, Step22 pair QA CSV

WRITES (new files only):
  outputs/reports/24_s2_robustness_screen.csv
  outputs/reports/24_s2_robustness_readme.txt
  outputs/figures/24_s2_150m_comparison.png

This is an EXPLORATORY stress-test, not bank erosion validation or prediction.
Changing distance from centreline is not bankline extraction; candidate
river-area windows may exclude actual river water or include unrelated water.
No missing pixels are filled and no training targets are manufactured.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / 'data' / 'processed' / 'sentinel2_change_v2'
BOUNDARIES = ROOT / 'data' / 'raw' / 'boundaries'
REPORTS = ROOT / 'outputs' / 'reports'
FIGURES = ROOT / 'outputs' / 'figures'

PAIRS = [(2018, 2024), (2019, 2024), (2020, 2024), (2018, 2025), (2024, 2025)]
WINDOWS_M = [('corridor_500m', None), ('centreline_150m', 150.0), ('centreline_75m', 75.0)]
DEFINITIONS = [
    ('consensus', (1,)),
    ('ndwi_positive', (1, 2)),
    ('mndwi_positive', (1, 3)),
]


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f'Required Step22 QA file missing: {path}')
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_raster(year):
    path = PROCESSED / f'22_s2_{year}_water_agreement_10m.tif'
    if not path.exists():
        raise FileNotFoundError(f'Run Step22 before Step24; missing {path}')
    with rasterio.open(path) as ds:
        if ds.crs is None or not ds.crs.is_projected or ds.nodata != 255 or ds.count != 1:
            raise ValueError(f'Unexpected CRS, nodata or band metadata in {path}')
        arr = ds.read(1)
        if not np.isin(arr, [0, 1, 2, 3, 255]).all():
            raise ValueError(f'Unknown water class code in {path}')
        meta = (ds.crs.to_string(), tuple(ds.transform), ds.height, ds.width)
        crs, transform = ds.crs, ds.transform
    return arr, meta, crs, transform


def main():
    annual = read_csv(REPORTS / '22_s2_annual_water_qa.csv')
    pairqa = read_csv(REPORTS / '22_s2_pairwise_change_qa.csv')
    observed_dates = {int(x['year']): x['observation_date'] for x in annual}
    available = {int(x['year']) for x in annual}
    selected = {y for pair in PAIRS for y in pair}
    if not selected.issubset(available):
        raise ValueError(f'Missing selected years in annual QA: {sorted(selected-available)}')

    arrays = {}
    reference = None
    for year in sorted(selected):
        arr, meta, crs, transform = load_raster(year)
        if reference is None:
            reference = meta
        elif reference != meta:
            raise ValueError('Step22 rasters have differing grids. Do not calculate pixel-to-pixel changes.')
        arrays[year] = arr
    h, w = next(iter(arrays.values())).shape

    corridor = gpd.read_file(BOUNDARIES / 'kelani_river_corridor.geojson')
    centreline = gpd.read_file(BOUNDARIES / 'kelani_river_centreline.geojson')
    for name, geo in [('corridor', corridor), ('centreline', centreline)]:
        if geo.empty or geo.crs is None or geo.geometry.isna().any() or not geo.geometry.is_valid.all():
            raise ValueError(f'Invalid or missing {name} geometry. Stop; do not guess.')
    projected_line = centreline.to_crs(crs)
    if len(projected_line) != 1:
        raise ValueError('Expected one existing candidate centreline feature.')
    line = projected_line.geometry.iloc[0]
    if line.geom_type not in ('LineString', 'MultiLineString') or line.length < 500:
        raise ValueError('Unexpected centreline geometry.')
    inside = geometry_mask(corridor.to_crs(crs).geometry.to_list(),
                           out_shape=(h, w), transform=transform,
                           invert=True, all_touched=False)
    centre_pixels = rasterize([(line, 1)], out_shape=(h, w), transform=transform,
                              fill=0, all_touched=True, dtype='uint8').astype(bool)
    if not inside.any() or not (centre_pixels & inside).any():
        raise ValueError('Raster grid does not intersect centreline/corridor.')
    if abs(transform.b) > 1e-7 or abs(transform.d) > 1e-7:
        raise ValueError('Rotated grids not supported by distance raster. Inspect alignment.')
    metre_distance = distance_transform_edt(~centre_pixels,
                                            sampling=(abs(transform.e), abs(transform.a)))
    windows = {label: (inside if distance is None else (inside & (metre_distance <= distance)))
               for label, distance in WINDOWS_M}
    if any(not m.any() for m in windows.values()):
        raise ValueError('Some centreline windows contain no pixels; inspect centreline location.')
    px_ha = abs(transform.a*transform.e-transform.b*transform.d)/10000
    rows = []
    for year_a, year_b in PAIRS:
        a, b = arrays[year_a], arrays[year_b]
        date_a = datetime.strptime(observed_dates[year_a], '%Y%m%d')
        date_b = datetime.strptime(observed_dates[year_b], '%Y%m%d')
        va, vb = (a != 255), (b != 255)
        for window_name, domain in windows.items():
            common = va & vb & domain
            n_common, n_domain = int(common.sum()), int(domain.sum())
            if not n_common:
                continue
            for definition, water_classes in DEFINITIONS:
                water_a = np.isin(a, water_classes) & domain
                water_b = np.isin(b, water_classes) & domain
                gain = common & ~water_a & water_b
                loss = common & water_a & ~water_b
                stable = common & water_a & water_b
                missing_a = water_b & ~va
                missing_b = water_a & ~vb
                observed_union = water_a | water_b
                common_water_union = observed_union & common
                water_cov = (100*common_water_union.sum()/observed_union.sum()) if observed_union.any() else np.nan
                gain_ha = int(gain.sum())*px_ha
                loss_ha = int(loss.sum())*px_ha
                missing_a_ha = int(missing_a.sum())*px_ha
                missing_b_ha = int(missing_b.sum())*px_ha
                qa = []
                if date_a.month != date_b.month: qa.append('DIFFERENT_ACQUISITION_MONTH')
                if (n_common/n_domain)*100 < 90: qa.append('COMMON_COVERAGE_LT_90')
                if np.isfinite(water_cov) and water_cov < 85: qa.append('WATER_DOMAIN_COVERAGE_LT_85')
                if missing_a_ha + missing_b_ha >= 0.25 * max(gain_ha+loss_ha, 0.01):
                    qa.append('UNMATCHED_WATER_DOMAIN_MATERIAL')
                if window_name == 'corridor_500m': qa.append('INCLUDES_OFF_CHANNEL_WATER')
                rows.append({
                    'year_a': year_a, 'date_a': observed_dates[year_a],
                    'year_b': year_b, 'date_b': observed_dates[year_b],
                    'spatial_window': window_name, 'water_rule': definition,
                    'domain_pixels': n_domain,
                    'common_valid_pct': round(n_common*100/n_domain, 2),
                    'water_domain_coverage_pct': round(float(water_cov), 2) if np.isfinite(water_cov) else '',
                    'stable_water_ha': round(int(stable.sum())*px_ha, 3),
                    'matched_gain_ha': round(gain_ha, 3),
                    'matched_loss_ha': round(loss_ha, 3),
                    'matched_net_ha': round(gain_ha - loss_ha, 3),
                    'later_water_where_earlier_invalid_ha': round(missing_a_ha, 3),
                    'earlier_water_where_later_invalid_ha': round(missing_b_ha, 3),
                    'unmatched_domain_effect_ha': round(missing_a_ha - missing_b_ha, 3),
                    'qa_flags': ';'.join(qa) if qa else 'EXPLORATORY_ONLY',
                    'meaning': 'WATER_FOOTPRINT_SCREENING_NOT_EROSION',
                })

    if not rows:
        raise RuntimeError('No valid comparisons; no outputs written.')
    # Regression guard: ensure this script reproduces Step22 full-corridor consensus data.
    existing = {(int(r['year_a']), int(r['year_b'])): r for r in pairqa}
    for r in rows:
        if r['spatial_window'] != 'corridor_500m' or r['water_rule'] != 'consensus':
            continue
        prior = existing.get((r['year_a'], r['year_b']))
        if prior is None:
            continue
        for new_key, old_key in [('matched_gain_ha', 'gain_ha'), ('matched_loss_ha', 'loss_ha')]:
            if abs(float(r[new_key]) - float(prior[old_key])) > 0.021:
                raise AssertionError(f'Step24 disagrees with Step22 {r["year_a"]}->{r["year_b"]} '
                                     f'{new_key}: {r[new_key]} vs {prior[old_key]}')
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = REPORTS / '24_s2_robustness_screen.csv'
    write_csv(path, rows)

    # One auditable visual: 150m near-centreline change varies with observation pair and water index.
    subset = [r for r in rows if r['spatial_window']=='centreline_150m']
    compare_labels = [f'{a}→{b}' for a, b in PAIRS]
    x = np.arange(len(compare_labels))
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    palette = ['#1472b8', '#16a276', '#cc7834']
    for i, (water_rule, _) in enumerate(DEFINITIONS):
        values = [next((r['matched_net_ha'] for r in subset if r['year_a']==a and
                         r['year_b']==b and r['water_rule']==water_rule), np.nan)
                  for a, b in PAIRS]
        ax.bar(x + (i-1)*0.23, values, width=0.22, label=water_rule.replace('_',' '), color=palette[i])
    ax.axhline(0, color='black', lw=0.7)
    ax.set_xticks(x, compare_labels)
    ax.set_ylabel('Matched net water-footprint difference (ha)')
    ax.set_xlabel('Selected observation years (different scenes/water stages)')
    ax.set_title('Kelani River: water-index sensitivity within 150m of candidate centreline')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.2)
    fig.text(0.5, 0.012, 'Exploratory observations only; NOT riverbank erosion, model accuracy, or validated change.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0,0.055,1,1))
    png = FIGURES / '24_s2_150m_comparison.png'
    fig.savefig(png, dpi=170, facecolor='white')
    plt.close(fig)

    readme = REPORTS / '24_s2_robustness_readme.txt'
    readme.write_text(
        'Step 24 is a change-screening sensitivity test, NOT independent validation.\n'
        'The 500m study corridor includes ponds, side channels, and wetland water.\n'
        'The 150m and 75m near-centreline windows are diagnostic filters ONLY;\n'
        'they are not surveyed riverbanks and can omit actual river sections.\n'
        'NDWI positive = agreement mask codes 1+2; MNDWI positive = codes 1+3;\n'
        'consensus = code 1. All use the original Step22 threshold zero.\n'
        'Only common-valid pixels are counted as gain/loss. Missing is not filled.\n'
        'Same-month imagery may still differ in river stage and atmospheric conditions.\n'
        'A result is robust only to tested definitions/windows, not necessarily physically true.\n',
        encoding='utf-8'
    )
    print('STEP 24 COMPLETE — EXPLORATORY ROBUSTNESS SCREEN')
    print(f'Comparisons: {len(PAIRS)}, spatial windows: {len(WINDOWS_M)}, water rules: {len(DEFINITIONS)}')
    print(f'Rows: {len(rows)} | Saved: {path}\nFigure: {png}\nNotes: {readme}')
    print('\n150m centreline CONSENSUS water comparisons:')
    for r in subset:
        if r['water_rule']=='consensus':
            print(f"  {r['year_a']}->{r['year_b']}: net {r['matched_net_ha']} ha; "
                  f"common {r['common_valid_pct']}%; water-domain {r['water_domain_coverage_pct']}%; "
                  f"QA {r['qa_flags']}")
    print('\nCaution: centreline distance is NOT a mapped riverbank. Do not call gains erosion.')

if __name__ == '__main__':
    main()
