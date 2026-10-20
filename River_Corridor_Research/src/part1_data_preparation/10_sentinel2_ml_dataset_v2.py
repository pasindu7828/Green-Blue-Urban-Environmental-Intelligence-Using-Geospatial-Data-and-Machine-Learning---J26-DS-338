"""Step 25: Sentinel-2 reach-level ML dataset, WITH provenance and leakage-safe split.

Inputs (read-only)
 - Step22 yearly 10m classified water agreement rasters, 2018..2025
 - Original Sentinel-2 10m & 20m reflectance exports
 - Existing digitized centreline and 500m-corridor geometry

Outputs (NEW only) in outputs/reports:
 - 25_s2_annual_reach_features.csv
 - 25_s2_reach_transition_ml_all.csv
 - 25_s2_ml_dataset_qa.json

Observational targets are changes in S2 water CLASSIFICATION on pixels valid
at BOTH dates. NOT erosion, surveyed riverbank displacement, or flood risk.
Missing measurements are never made up. 150m proximity is an experimental
study window, NOT a confirmed bankline; 20m SWIR retains its native detail.

The 2023->2024 transition is reserved for validation, 2024->2025 for test.
Do NOT train preprocessing, imputation, or feature selection on val/test.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import reproject, Resampling
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw' / 'sentinel2'
MASKDIR = ROOT / 'data' / 'processed' / 'sentinel2_change_v2'
BOUNDARY = ROOT / 'data' / 'raw' / 'boundaries'
REPORTS = ROOT / 'outputs' / 'reports'
YEARS = list(range(2018, 2026))
WINDOW_M = 150.0
REACH_LENGTH_M = 500.0
MIN_COMMON_VALID_PCT = 90.0
MIN_WATER_DOMAIN_PCT = 85.0
SAMPLE_SPACING_M = 5.0
EPS = 1e-6


def csv_write(path, rows):
    if not rows:
        raise ValueError(f'No records for {path}')
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def desc_check(ds, descriptions):
    if tuple(ds.descriptions) != descriptions or ds.nodata != -9999.0:
        raise ValueError(f'Unexpected band descriptions or nodata: {ds.name}: {ds.descriptions}')


def read_year(year, reference):
    matches = sorted(RAW.glob(f'kelani_sentinel2_{year}_????????_10m.tif'))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one 10m Sentinel file for {year}: found {len(matches)}')
    ten_path = matches[0]
    date_str = re.fullmatch(r'kelani_sentinel2_\d{4}_(\d{8})_10m\.tif', ten_path.name).group(1)
    dt = datetime.strptime(date_str, '%Y%m%d')
    if dt.year != year:
        raise ValueError(f'Year/date mismatch for {ten_path}')
    twenty_path = ten_path.with_name(ten_path.name.replace('_10m.tif', '_20m.tif'))
    mask_path = MASKDIR / f'22_s2_{year}_water_agreement_10m.tif'
    if not twenty_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f'Missing paired S2 or Step22 mask for {year}')
    with rasterio.open(ten_path) as ten, rasterio.open(twenty_path) as twenty, rasterio.open(mask_path) as md:
        desc_check(ten, ('blue', 'green', 'red', 'nir', 'valid_pixel'))
        desc_check(twenty, ('swir1', 'swir2', 'scl', 'valid_pixel'))
        if ten.count != 5 or twenty.count != 4 or md.count != 1 or md.nodata != 255:
            raise ValueError(f'Band count/mask nodata unexpected in {year}')
        ref = (ten.crs.to_string() if ten.crs else None, tuple(ten.transform), ten.height, ten.width)
        mask_ref = (md.crs.to_string() if md.crs else None, tuple(md.transform), md.height, md.width)
        if ref != mask_ref or (reference is not None and ref != reference):
            raise ValueError(f'Rasters not aligned in {year}; do not assume pixel equivalence')
        if ten.crs != twenty.crs:
            raise ValueError(f'10/20m CRS mismatch {year}')
        water_class = md.read(1)
        if not np.isin(water_class, [0, 1, 2, 3, 255]).all():
            raise ValueError(f'Unknown water classes {year}')
        red, nir = ten.read(3), ten.read(4)
        swir = np.full(red.shape, -9999.0, dtype=np.float32)
        reproject(source=rasterio.band(twenty, 1), destination=swir,
                  src_transform=twenty.transform, src_crs=twenty.crs,
                  src_nodata=twenty.nodata, dst_transform=ten.transform,
                  dst_crs=ten.crs, dst_nodata=-9999.0,
                  resampling=Resampling.bilinear)
        reflectance_valid = ((water_class != 255) & np.isfinite(red) & np.isfinite(nir)
                             & np.isfinite(swir) & (red != -9999) & (nir != -9999)
                             & (swir != -9999) & (np.abs(nir+red) > EPS)
                             & (np.abs(swir+nir) > EPS))
        ndvi = np.full(red.shape, np.nan, dtype=np.float32)
        ndbi = np.full(red.shape, np.nan, dtype=np.float32)
        np.divide(nir-red, nir+red, out=ndvi, where=reflectance_valid)
        np.divide(swir-nir, swir+nir, out=ndbi, where=reflectance_valid)
        ndvi[~reflectance_valid] = np.nan
        ndbi[~reflectance_valid] = np.nan
        return water_class, ndvi, ndbi, dt, ref, ten.crs, ten.transform


def reach_geometry(reference, crs, transform):
    _, _, h, w = reference
    corridor = gpd.read_file(BOUNDARY / 'kelani_river_corridor.geojson')
    centre = gpd.read_file(BOUNDARY / 'kelani_river_centreline.geojson')
    for label, geo in [('corridor', corridor), ('centreline', centre)]:
        if geo.empty or geo.crs is None or geo.geometry.isna().any() or not geo.geometry.is_valid.all():
            raise ValueError(f'Invalid {label} geometry')
    if len(centre) != 1:
        raise ValueError('Expected exactly one digitized centreline feature')
    line = centre.to_crs(crs).geometry.iloc[0]
    if line.geom_type != 'LineString' or line.length < 500:
        raise ValueError('Expected a single line >500 m')
    inside = geometry_mask(corridor.to_crs(crs).geometry.to_list(),
                           out_shape=(h,w), transform=transform, invert=True, all_touched=False)
    if not inside.any():
        raise ValueError('Corridor does not overlap satellite grid')
    sample_dist = np.arange(0.0, line.length + SAMPLE_SPACING_M, SAMPLE_SPACING_M)
    sample_dist = np.minimum(sample_dist, line.length)
    pts = np.array([(line.interpolate(float(d)).x, line.interpolate(float(d)).y)
                    for d in sample_dist], dtype=np.float64)
    tree = cKDTree(pts)
    if abs(transform.b)>1e-7 or abs(transform.d)>1e-7:
        raise ValueError('Rotated raster grids require separately reviewed distance logic')
    line_pixels = rasterize([(line, 1)], out_shape=(h,w), transform=transform,
                            fill=0, all_touched=True, dtype='uint8').astype(bool)
    distance_to_line = distance_transform_edt(~line_pixels, sampling=(abs(transform.e),abs(transform.a)))
    domain = inside & (distance_to_line <= WINDOW_M)
    rr, cc = np.nonzero(domain)
    xs, ys = rasterio.transform.xy(transform, rr, cc, offset='center')
    points = np.column_stack((np.asarray(xs), np.asarray(ys)))
    _, nearest = tree.query(points, k=1)
    nreach = int(np.ceil(line.length/REACH_LENGTH_M))
    labels = np.full((h,w), -1, dtype=np.int16)
    labels[rr, cc] = np.minimum(
        (sample_dist[nearest] // REACH_LENGTH_M).astype('int16'), nreach-1)
    return labels, nreach, float(line.length)


def median_or_none(values):
    clean = values[np.isfinite(values)]
    return round(float(np.median(clean)), 6) if clean.size else None


def rounded_pct(a,b):
    return round(100*float(a)/float(b),3) if b else None


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    annual = []
    reference = crs = transform = labels = None
    nreach = None
    pixel_ha = None
    for year in YEARS:
        mask, ndvi, ndbi, dt, meta, scene_crs, scene_transform = read_year(year, reference)
        if reference is None:
            reference, crs, transform = meta, scene_crs, scene_transform
            labels, nreach, line_length = reach_geometry(reference, crs, transform)
            pixel_ha = abs(transform.a*transform.e-transform.b*transform.d)/10000
            if pixel_ha <= 0:
                raise ValueError('Invalid pixel area')
        snapshots[year] = {'mask':mask, 'ndvi':ndvi, 'ndbi':ndbi, 'date':dt}
        for idx in range(nreach):
            region = labels == idx
            total = int(region.sum())
            if not total:
                raise ValueError(f'Empty 150m candidate region R{idx+1:03}')
            valid = region & (mask != 255)
            nw = int(np.count_nonzero(region & (mask == 1)))
            nv = int(valid.sum())
            strict_nonwater = region & (mask == 0)
            good_veg = strict_nonwater & np.isfinite(ndvi)
            ndvi_vals = ndvi[good_veg]
            ndbi_vals = ndbi[strict_nonwater & np.isfinite(ndbi)]
            annual.append({
                'year':year, 'date':dt.strftime('%Y-%m-%d'), 'region_id':f'R{idx+1:03}',
                'region_index':idx+1, 'start_chainage_m':round(idx*REACH_LENGTH_M,1),
                'region_domain_pixels':total, 'valid_pixels':nv,
                'valid_pct':rounded_pct(nv,total),
                'observed_consensus_water_ha':round(nw*pixel_ha,4),
                'observed_water_pct_valid':rounded_pct(nw,nv),
                'strict_nonwater_ndvi_median':median_or_none(ndvi_vals),
                'strict_nonwater_ndbi_median':median_or_none(ndbi_vals),
                'vegetated_pct_of_index_valid_nonwater':rounded_pct(int(np.sum(ndvi_vals>0.3)),len(ndvi_vals)),
                'spectral_index_sample_pixels':int(len(ndvi_vals)),
                'measurement_status':'OBSERVED_CLASSIFICATION_NOT_BANKLINE',
            })
        print(f'{year} annual reach indicators processed ({nreach} reaches)')
    annual_by = {(r['year'],r['region_id']):r for r in annual}
    pairs = []
    for y in YEARS[:-1]:
        a, b = snapshots[y], snapshots[y+1]
        date_mismatch = a['date'].month != b['date'].month
        for idx in range(nreach):
            region = labels == idx
            ma, mb = a['mask'], b['mask']
            va, vb = region & (ma!=255), region & (mb!=255)
            common = va & vb
            wa, wb = region & (ma==1), region & (mb==1)
            common_n = int(common.sum())
            gain_n = int(np.count_nonzero(common & ~wa & wb))
            loss_n = int(np.count_nonzero(common & wa & ~wb))
            water_union = wa | wb
            water_union_n = int(water_union.sum())
            water_common_n = int(np.count_nonzero(water_union & common))
            matched_water_cov = rounded_pct(water_common_n,water_union_n)
            common_valid_pct = rounded_pct(common_n,int(region.sum()))
            eligible = (common_valid_pct is not None and common_valid_pct >= MIN_COMMON_VALID_PCT
                        and matched_water_cov is not None and matched_water_cov >= MIN_WATER_DOMAIN_PCT)
            source = annual_by[(y, f'R{idx+1:03}')]
            target_delta = (gain_n-loss_n)*pixel_ha if common_n else None
            split = ('train' if y+1<=2023 else 'validation' if y+1==2024 else 'test')
            qa_flags=[]
            if date_mismatch: qa_flags.append('DIFFERENT_ACQUISITION_MONTH')
            if common_valid_pct is None or common_valid_pct < MIN_COMMON_VALID_PCT:
                qa_flags.append('LOW_COMMON_VALID_COVERAGE')
            if matched_water_cov is None or matched_water_cov < MIN_WATER_DOMAIN_PCT:
                qa_flags.append('LOW_WATER_DOMAIN_COVERAGE')
            if not common_n: qa_flags.append('NO_COMMON_VALID_PIXEL')
            pairs.append({
                'reach_id':source['region_id'],'reach_index':idx+1,
                'source_year':y,'target_year':y+1,
                'source_date':a['date'].strftime('%Y-%m-%d'),
                'target_date':b['date'].strftime('%Y-%m-%d'),
                'split':split,'x_reach_chainage_km':round(idx*REACH_LENGTH_M/1000,3),
                'x_source_year':y, 'x_source_day_of_year':a['date'].timetuple().tm_yday,
                'x_source_water_ha':source['observed_consensus_water_ha'],
                'x_source_water_pct_valid':source['observed_water_pct_valid'],
                'x_source_valid_pct':source['valid_pct'],
                'x_source_nonwater_ndvi_median':source['strict_nonwater_ndvi_median'],
                'x_source_nonwater_ndbi_median':source['strict_nonwater_ndbi_median'],
                'x_source_vegetated_pct_nonwater':source['vegetated_pct_of_index_valid_nonwater'],
                'y_matched_water_delta_ha':round(target_delta,4) if target_delta is not None else None,
                'observed_gain_ha':round(gain_n*pixel_ha,4),
                'observed_loss_ha':round(loss_n*pixel_ha,4),
                'common_valid_pct':common_valid_pct,
                'water_domain_coverage_pct':matched_water_cov,
                'high_coverage_for_evaluation':int(eligible),
                'qa_flags':';'.join(qa_flags) if qa_flags else 'EXPLORATORY_NO_QA_FLAG',
                'target_interpretation':'NEXT_YEAR_SATELLITE_WATER_CLASSIFICATION_CHANGE_NOT_EROSION',
            })
    csv_write(REPORTS/'25_s2_annual_reach_features.csv',annual)
    csv_write(REPORTS/'25_s2_reach_transition_ml_all.csv',pairs)
    split_counts = {split:{'all':sum(r['split']==split for r in pairs),
                           'high_coverage':sum(r['split']==split and r['high_coverage_for_evaluation']==1
                                               for r in pairs)}
                    for split in ('train','validation','test')}
    feature_names = [k for k in pairs[0] if k.startswith('x_')]
    missing = {f:sum(row[f] is None for row in pairs) for f in feature_names}
    qa={
        'status':'EXPLORATORY_MODEL_DATASET_REQUIRES_VALIDATION',
        'dataset_source':'Sentinel-2 GEE export 2018-2025, Step22 masks, original centreline',
        'water_definition':'Class1 NDWI>0 AND MNDWI>0 under Step22 pixel QA',
        'native_swir_resolution_m':20,
        'raster_analysis_grid_m':10,
        'target':'next-year matched-domain change in classified consensus-water hectares',
        'not_targets':['independently surveyed riverbank erosion','flood risk','bank failure'],
        'windows':'150m from candidate centreline, 500m length reaches',
        'reach_count':nreach,'centreline_length_m':round(line_length,2),
        'annual_rows':len(annual),'transition_rows':len(pairs),
        'temporal_split_counts':split_counts,
        'feature_fields':feature_names,
        'missing_feature_value_count':missing,
        'high_coverage_definition':'common_valid_pct >=90 AND water_domain_coverage_pct >=85',
        'split_policy':'train target year <=2023; val 2024; test 2025. Never fit preprocessing on val/test.',
        'known_limitations':['Only 8 annual satellite scenes; 7 transitions per reach',
                             'Near-coincident adjacent reaches are spatially correlated',
                             'Acquisition month and water stage can differ',
                             'Water labels and QA are satellite-derived, not ground truth',
                             'QA based on target-date pixels must only filter scoring and training samples, not be a feature',
                             '20m SWIR resampled to 10m for MNDWI; no native 10m SWIR detail',
                             'Synthetic pixels/labels were not created'],
    }
    path=REPORTS/'25_s2_ml_dataset_qa.json'
    path.write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print('\nSTEP 25 COMPLETE - SATELLITE REACH ML DATASET (NOT MODEL TRAINING)')
    print('Reaches:',nreach,'Annual rows:',len(annual),'Transition rows:',len(pairs))
    print('Temporal splits:',json.dumps(split_counts))
    print('Missing predictors:',json.dumps(missing))
    print('Annual:',REPORTS/'25_s2_annual_reach_features.csv')
    print('Model-ready:',REPORTS/'25_s2_reach_transition_ml_all.csv')
    print('QA:',path)
    print('Caution: NO erosion labels or independently validated predictions.')


if __name__=='__main__':
    main()
