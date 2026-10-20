"""Step 28B | Kelani River Sentinel-2 riparian vegetation screening (2018-2025).

READS (no alterations):
 - data/raw/sentinel2/kelani_sentinel2_YYYY_YYYYMMDD_10m.tif (bands Blue Green Red NIR valid)
 - data/raw/sentinel2/kelani_sentinel2_YYYY_YYYYMMDD_20m.tif (SCL band 3)
 - data/processed/sentinel2_change_v2/22_s2_YYYY_water_agreement_10m.tif
 - data/raw/boundaries/kelani_river_centreline.geojson
 - data/raw/boundaries/kelani_river_corridor.geojson

GENERATES NEW FILES ONLY (prefix 28_s2_) under data/processed,
outputs/reports and outputs/figures. No model training or label fabrication.

Design:
 - Fixed 2020 (January) strictly classified, centreline-associated river-water mask.
 - 0-30, 30-50 and 50-100 m zones OUTSIDE the 2020 reference water footprint,
   within a 250 m centreline screening window. These are experimental buffers,
   not legal reservations, surveyed banklines, or independently verified vegetation.
 - Use SCL classes 4/5, Step22 class=0, valid reflectance, quality observations.
 - Comparisons use the SAME spatial pixels valid as land on both dates.
 - Vegetation = NDVI>=0.30 (sensitivity of threshold remains for follow-up).
 - EVI computed as supplementary indicator with denominators and range screened.
 - Pixels with water, disagreement, or invalid imagery on either date are NOT
   counted as vegetation gain/loss.
 - Approximate reach assignment: nearest sampled 2020-centreline location,
   subdivided into 500m reaches (R001..R028). Ends of meanders may require GIS QA.

Science: These are satellite-derived vegetation index / vegetation-cover PROXIES.
Seasonal observation differences and reflectance/spectral index sensitivity can
mimic land-cover changes. No independent reference-validation or trained forecast.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import Resampling, reproject
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/sentinel2'
WATER = ROOT / 'data/processed/sentinel2_change_v2'
BOUND = ROOT / 'data/raw/boundaries'
DERIVED = ROOT / 'data/processed/riparian_vegetation_v2'
REPORTS = ROOT / 'outputs/reports'
FIGURES = ROOT / 'outputs/figures'
YEARS = tuple(range(2018, 2026))
REFERENCE_YEAR = 2020
VEG_NDVI_THRESHOLD = 0.30
RING_BANDS = ((0, 30), (30, 50), (50, 100), (0, 100))
REACH_LENGTH_M = 500.0
MAX_CENTRELINE_PROXIMITY_M = 250.0
MIN_REFERENCE_COMPONENT_PIXELS = 25
MIN_PAIR_COVERAGE_PCT = 70.0
MIN_PAIR_PIXELS_PER_REACH = 30
EPS = 1.0e-6


def pct(part, total):
    return round(100.0 * float(part) / float(total), 2) if total else None


def median(values):
    return round(float(np.median(values)), 5) if values.size else None


def csv_out(path, rows):
    if not rows:
        raise ValueError('Empty report ' + str(path))
    with path.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_difference(a, b):
    result = np.full(a.shape, np.nan, dtype='float32')
    d = a + b
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(d) > EPS)
    np.divide(a - b, d, where=mask, out=result)
    return result


def index_arrays(blue, red, nir, valid):
    """Valid red/nir/blue reflectance expected in scene export (fractional, not DN)."""
    valid_ndvi = (valid & np.isfinite(red) & np.isfinite(nir)
                  & (red >= 0) & (nir >= 0) & (red < 2) & (nir < 2)
                  & ((red + nir) > EPS))
    ndvi = normalize_difference(nir, red)
    ndvi[~valid_ndvi] = np.nan
    den = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = np.full(nir.shape, np.nan, dtype='float32')
    valid_evi = (valid_ndvi & np.isfinite(blue) & (blue >= 0) & (blue < 2)
                 & np.isfinite(den) & (np.abs(den) > 0.20))
    np.divide(2.5 * (nir - red), den, where=valid_evi, out=evi)
    evi[~valid_evi | ~np.isfinite(evi) | (evi < -1) | (evi > 1)] = np.nan
    return ndvi, evi


def chosen_files(year):
    matches = list(RAW.glob(f'kelani_sentinel2_{year}_????????_10m.tif'))
    if len(matches) != 1:
        raise ValueError(f'{year}: expected one Sentinel-2 10m scene, got {len(matches)}')
    ten = matches[0]
    m = re.fullmatch(r'kelani_sentinel2_(\d{4})_(\d{8})_10m\.tif', ten.name)
    if not m:
        raise ValueError('Unexpected filename ' + ten.name)
    date = datetime.strptime(m.group(2), '%Y%m%d')
    if date.year != year:
        raise ValueError('Scene year/date mismatch ' + ten.name)
    twenty = ten.with_name(ten.name.replace('_10m.tif', '_20m.tif'))
    water = WATER / f'22_s2_{year}_water_agreement_10m.tif'
    if not twenty.exists() or not water.exists():
        raise FileNotFoundError(f'Missing original S2 20m or Step22 raster for {year}')
    return ten, twenty, water, date


def grid_id(src):
    return (src.crs.to_string() if src.crs else None, tuple(src.transform), src.height, src.width)


def read_year(year, common_grid=None):
    ten, twenty, maskpath, date = chosen_files(year)
    with rasterio.open(ten) as s10, rasterio.open(twenty) as s20, rasterio.open(maskpath) as sm:
        if tuple(s10.descriptions) != ('blue', 'green', 'red', 'nir', 'valid_pixel'):
            raise ValueError('Unexpected Sentinel-2 10 m band metadata for '+str(year))
        if tuple(s20.descriptions) != ('swir1', 'swir2', 'scl', 'valid_pixel'):
            raise ValueError('Unexpected Sentinel-2 20 m band metadata for '+str(year))
        if s10.nodata != -9999 or s20.nodata != -9999 or sm.nodata != 255:
            raise ValueError('Unexpected nodata metadata '+str(year))
        if s10.res != (10., 10.) or s20.res != (20.,20.):
            raise ValueError('Expected 10 m and 20 m input rasters')
        if s10.crs != s20.crs or grid_id(s10) != grid_id(sm):
            raise ValueError('Input raster grids do not align for year '+str(year))
        if common_grid is not None and common_grid != grid_id(s10):
            raise ValueError('Scene is not on the fixed eight-year grid: '+str(year))
        blue, red, nir, valid10 = (s10.read(i).astype('float32') for i in (1,3,4,5))
        water_class = sm.read(1)
        if not np.isin(water_class, [0,1,2,3,255]).all():
            raise ValueError('Unknown Step22 water class: '+str(year))
        scl = np.full(water_class.shape, -9999., dtype='float32')
        reproject(source=rasterio.band(s20, 3), destination=scl,
                  src_transform=s20.transform, src_crs=s20.crs, src_nodata=s20.nodata,
                  dst_transform=s10.transform, dst_crs=s10.crs, dst_nodata=-9999,
                  resampling=Resampling.nearest)
        # Land observed as SCL=vegetation(4) OR bare(5) AND agreed nonwater(0).
        stable_observable_land = ((water_class == 0) & (valid10 > 0.5)
                                  & np.isin(scl, [4, 5]))
        ndvi, evi = index_arrays(blue, red, nir, stable_observable_land)
        return dict(date=date, water=water_class, ndvi=ndvi, evi=evi,
                    land=np.isfinite(ndvi), scl=scl, grid=grid_id(s10),
                    profile=s10.profile, transform=s10.transform, crs=s10.crs,
                    width=s10.width, height=s10.height)


def reference_zone(scenes):
    ref = scenes[REFERENCE_YEAR]
    crs = ref['crs']
    h, w = ref['height'], ref['width']
    transform = ref['transform']
    linefile = BOUND / 'kelani_river_centreline.geojson'
    corridorfile = BOUND / 'kelani_river_corridor.geojson'
    centre = gpd.read_file(linefile)
    corridor = gpd.read_file(corridorfile)
    for name, g in [('centreline',centre), ('corridor', corridor)]:
        if g.empty or g.crs is None or not g.geometry.is_valid.all():
            raise ValueError(f'Invalid/missing {name} GIS geometry')
    line = centre.to_crs(crs).geometry.iloc[0]
    if len(centre)!=1 or line.geom_type!='LineString' or line.length < 1000:
        raise ValueError('Expected one continuous digitized centreline')
    corridor_mask = geometry_mask(corridor.to_crs(crs).geometry.to_list(),
                                  out_shape=(h,w), transform=transform,
                                  invert=True, all_touched=False)
    linepix = rasterize([(line,1)], out_shape=(h,w), transform=transform,
                        dtype='uint8', all_touched=True).astype(bool)
    distance_line = distance_transform_edt(~linepix, sampling=(10,10))
    strict_ref = (ref['water']==1) & corridor_mask
    # Exclude isolated pond/wetland components: retain 2020 water components
    # of meaningful size touching a 30m radius from the reference centreline.
    cc, component_count = label(strict_ref, structure=np.ones((3,3),dtype='uint8'))
    sizes = np.bincount(cc.ravel())
    anchor = binary_dilation(linepix, iterations=3)
    ids = np.unique(cc[anchor & strict_ref])
    ids = ids[(ids > 0) & (sizes[ids] >= MIN_REFERENCE_COMPONENT_PIXELS)]
    river_reference = np.isin(cc, ids)
    if river_reference.sum() < 300:
        raise ValueError('Reference river water too fragmented/missing; inspect QGIS before interpreting land buffers')
    dist_ref = distance_transform_edt(~river_reference, sampling=(10,10))
    base_land = corridor_mask & ~river_reference & (distance_line <= MAX_CENTRELINE_PROXIMITY_M)
    rings = {}
    for inner, outer in RING_BANDS:
        rings[f'{inner}-{outer}m'] = base_land & (dist_ref > inner) & (dist_ref <= outer)
    if any(rings[k].sum() < 100 for k in rings):
        raise ValueError('Too few river-edge pixels in one distance ring')
    # Assign near-river pixels approximately by nearest 5m centreline segment samples.
    sample_d = np.arange(0, line.length+5.,5.)
    sample_d = np.minimum(sample_d,line.length)
    pts = np.array([(line.interpolate(float(d)).x,line.interpolate(float(d)).y)
                    for d in sample_d], dtype='float64')
    rr, cc0 = np.nonzero(rings['0-100m'])
    xx, yy = rasterio.transform.xy(transform, rr, cc0, offset='center')
    nearest = cKDTree(pts).query(np.column_stack([xx,yy]))[1]
    nreach = int(np.ceil(line.length/REACH_LENGTH_M))
    reaches = np.full((h,w), -1, dtype='int16')
    reaches[rr,cc0] = np.minimum((sample_d[nearest]/REACH_LENGTH_M).astype('int16'), nreach-1)
    for i in range(nreach):
        if not ((reaches==i)&rings['0-100m']).any():
            raise ValueError('Reach with no reference riparian zone: '+str(i+1))
    meta=dict(reference_year=REFERENCE_YEAR, reference_water_pixels=int(strict_ref.sum()),
              centreline_anchored_reference_water_pixels=int(river_reference.sum()),
              centreline_anchored_components=int(len(ids)),
              total_reference_water_components=int(component_count),
              centreline_length_m=round(float(line.length),2), reach_count=nreach,
              fixed_zone_pixel_counts={k:int(v.sum()) for k,v in rings.items()},
              minimum_component_pixels=MIN_REFERENCE_COMPONENT_PIXELS,
              max_centreline_distance_m=MAX_CENTRELINE_PROXIMITY_M)
    return rings, reaches, river_reference, line, meta


def write_tif(path, array, profile, description, nodata, categories=None):
    p=profile.copy()
    p.update(driver='GTiff', count=1, dtype=array.dtype.name, nodata=nodata,
             compress='deflate', predictor=3 if np.issubdtype(array.dtype,np.floating) else 1)
    with rasterio.open(path,'w',**p) as dst:
        dst.write(array,1)
        dst.set_band_description(1,description)
        dst.update_tags(interpretation='EXPLORATORY_NOT_VERIFIED_VEGETATION_LOSS')
        if categories:
            dst.update_tags(**{f'class_{k}':v for k,v in categories.items()})


def annual_rows(scenes,rings,reaches,pixelha):
    rows=[]
    for year in YEARS:
        s=scenes[year]
        for zone,ring in rings.items():
            for ix in range(int(reaches.max())+1):
                m=ring & (reaches==ix)
                total=int(m.sum())
                good=m & s['land']
                valid=int(good.sum())
                vals=s['ndvi'][good]
                evals=s['evi'][good & np.isfinite(s['evi'])]
                fraction=pct(valid,total)
                veg=int(np.count_nonzero(vals>=VEG_NDVI_THRESHOLD))
                rows.append(dict(year=year, observation_date=s['date'].strftime('%Y-%m-%d'),
                    reach_id=f'R{ix+1:03d}', zone=zone, fixed_zone_pixels=total,
                    eligible_land_pixels=valid, observed_land_coverage_pct=fraction,
                    observed_veg_pixels=veg, observed_veg_area_ha=round(veg*pixelha,3),
                    veg_pct_of_valid_land=pct(veg,valid), median_ndvi=median(vals),
                    median_evi=median(evals), evi_valid_pixels=int(evals.size),
                    qa_flag=('LOW_LAND_OBSERVATION_COVERAGE' if fraction is None or fraction<MIN_PAIR_COVERAGE_PCT
                             else 'EXPLORATORY_NO_COVERAGE_FLAG'),
                    interpretation='SATELLITE_NDVI_VEGETATION_PROXY_NOT_FIELD_VALIDATED'))
    return rows


def compute_change_codes(old,new,zone):
    good=zone & old['land'] & new['land']
    a=old['ndvi']>=VEG_NDVI_THRESHOLD
    b=new['ndvi']>=VEG_NDVI_THRESHOLD
    codes=np.full(old['water'].shape,255,dtype='uint8')
    codes[good & ~a & ~b]=0
    codes[good & a & b]=1
    codes[good & ~a & b]=2
    codes[good & a & ~b]=3
    return codes


def pair_rows(scenes,rings,reaches,pixelha):
    pairs=[(a,b) for a,b in zip(YEARS[:-1],YEARS[1:])]
    for pair in [(2018,2024),(2020,2024),(2018,2025)]:
        if pair not in pairs:
            pairs.append(pair)
    results=[]
    for a,b in pairs:
        old,new=scenes[a],scenes[b]
        code=compute_change_codes(old,new,rings['0-100m'])
        if (a,b) in [(2018,2024),(2020,2024)]:
            write_tif(DERIVED / f'28_s2_vegetation_change_{a}_{b}_10m.tif',code,
                new['profile'],'Fixed 2020 riparian zone NDVI status and changes',255,
                {0:'stable_low_ndvi',1:'stable_vegetated',2:'vegetation_gain_proxy',
                 3:'vegetation_loss_proxy',255:'invalid_unobserved_water_or_outside_zone'})
        for zone,ring in rings.items():
            for ix in range(int(reaches.max())+1):
                z=ring & (reaches==ix)
                total=int(z.sum())
                land_valid=z & old['land'] & new['land']
                n=int(land_valid.sum())
                coverage=pct(n,total)
                y_old=old['ndvi'][land_valid]
                y_new=new['ndvi'][land_valid]
                diff=y_new-y_old
                loss=int(np.count_nonzero((y_old >= VEG_NDVI_THRESHOLD) & (y_new < VEG_NDVI_THRESHOLD)))
                gain=int(np.count_nonzero((y_old < VEG_NDVI_THRESHOLD) & (y_new >= VEG_NDVI_THRESHOLD)))
                stable_high=int(np.count_nonzero((y_old >= VEG_NDVI_THRESHOLD) & (y_new >= VEG_NDVI_THRESHOLD)))
                stable_low=int(np.count_nonzero((y_old < VEG_NDVI_THRESHOLD) & (y_new < VEG_NDVI_THRESHOLD)))
                water_excl=int(np.count_nonzero(z & ((old['water']==1) | (new['water']==1))))
                uncertain_water=int(np.count_nonzero(z & (np.isin(old['water'],[2,3])|np.isin(new['water'],[2,3]))))
                flags=[]
                if old['date'].month != new['date'].month:
                    flags.append('DIFFERENT_ACQUISITION_MONTH')
                if coverage is None or coverage<MIN_PAIR_COVERAGE_PCT:
                    flags.append('LOW_PAIR_LAND_COVERAGE')
                if n<MIN_PAIR_PIXELS_PER_REACH:
                    flags.append('FEW_MATCHED_LAND_PIXELS')
                if water_excl>0:
                    flags.append('WATER_DOMAIN_EXCLUSIONS')
                if uncertain_water>0:
                    flags.append('INDEX_DISAGREEMENT_EXCLUSIONS')
                results.append(dict(year_a=a,year_b=b, date_a=old['date'].strftime('%Y-%m-%d'),
                    date_b=new['date'].strftime('%Y-%m-%d'), reach_id=f'R{ix+1:03d}',zone=zone,
                    fixed_zone_pixels=total, matched_valid_land_pixels=n,
                    matched_land_coverage_pct=coverage,
                    median_ndvi_delta=median(diff),
                    mean_ndvi_delta=round(float(np.mean(diff)),5) if n else None,
                    vegetation_loss_ha=round(loss*pixelha,3),
                    vegetation_gain_ha=round(gain*pixelha,3),
                    net_vegetation_change_ha=round((gain-loss)*pixelha,3),
                    stable_vegetated_ha=round(stable_high*pixelha,3),
                    stable_low_ndvi_ha=round(stable_low*pixelha,3),
                    water_excluded_pixels=water_excl,
                    disagreement_excluded_pixels=uncertain_water,
                    qa_flags=';'.join(flags) if flags else 'EXPLORATORY_NO_QA_FLAG',
                    interpretation='NDVI_THRESHOLD_CHANGE_ON_MATCHED_STABLE_LAND_NOT_CONFIRMED_DEFORESTATION'))
        print(f'{a}->{b}: fixed-zone matched land = '
              f'{sum(r["matched_valid_land_pixels"] for r in results if r["year_a"]==a and r["year_b"]==b and r["zone"]=="0-100m")} px')
    return results


def plot_pair(scenes,rings,reaches,line,rows,a,b,out):
    old,new=scenes[a],scenes[b]
    classes=compute_change_codes(old,new,rings['0-100m'])
    fig=plt.figure(figsize=(16,7),facecolor='white')
    gs=fig.add_gridspec(1,2,width_ratios=[1.55,0.85],left=.05,right=.98,top=.83,bottom=.17,wspace=.20)
    ax=fig.add_subplot(gs[0,0]);bar=fig.add_subplot(gs[0,1])
    colours=np.array([[0.75,0.77,0.80,0.48],[0.15,0.56,0.31,0.72],
                      [0.00,0.75,0.45,0.95],[0.94,0.29,0.17,0.98],[0,0,0,0]],dtype='float32')
    draw=colours[np.where(classes==255,4,classes)]
    extent=(new['transform'].c, new['transform'].c+new['width']*10,
            new['transform'].f-new['height']*10,new['transform'].f)
    ax.imshow(draw,extent=extent,origin='upper',interpolation='nearest')
    x,y=line.xy
    ax.plot(x,y,color='#335b82',linewidth=1.05,alpha=.75)
    ax.set_aspect('equal');ax.set_xlabel('Easting (m), EPSG:32644');ax.set_ylabel('Northing (m), EPSG:32644')
    ax.set_title('Fixed reference river-edge zone (0–100m)\nNDVI>=0.30 on matched non-water land',fontsize=11)
    subset=[r for r in rows if r['year_a']==a and r['year_b']==b and r['zone']=='0-100m']
    top=sorted(subset,key=lambda r:r['vegetation_loss_ha'],reverse=True)[:12]
    bar.barh([r['reach_id'] for r in top[::-1]],[r['vegetation_loss_ha'] for r in top[::-1]],color='#db553a')
    bar.set_xlabel('Apparent vegetation loss (ha; matched land)')
    bar.set_title('Largest apparent NDVI vegetation-loss reaches\n(not independently verified)',fontsize=10)
    bar.grid(axis='x',alpha=.18)
    fig.suptitle(f'KELANI RIVER | EXPLORATORY RIPARIAN VEGETATION | {a}–{b}',fontsize=15,weight='bold',y=.97)
    fig.legend(handles=[Patch(facecolor='#319051',label='Stable NDVI-vegetated'),
                        Patch(facecolor='#00c777',label='NDVI vegetation gain'),
                        Patch(facecolor='#f04a2b',label='NDVI vegetation loss'),
                        Patch(facecolor='#bfc4cb',label='Stable low NDVI')],
               loc='lower center',bbox_to_anchor=(.5,.09),ncol=4,frameon=False,fontsize=9)
    fig.text(.5,.035,'Satellite-derived threshold transitions only. Fixed 2020 water-boundary proxy; '\
                  '2020 river centreline is not surveyed. Unobserved/water pixels excluded. '\
                  'Season and cloud may bias change.',fontsize=8.5,color='#555',ha='center')
    fig.savefig(out,dpi=165,bbox_inches='tight',facecolor='white');plt.close(fig)


def main():
    for folder in [DERIVED,REPORTS,FIGURES]:
        folder.mkdir(parents=True,exist_ok=True)
    scenes={}; grid=None
    for year in YEARS:
        s=read_year(year,grid)
        if grid is None:
            grid=s['grid']
        scenes[year]=s
        p=s['profile']
        ndvi_out=np.where(np.isfinite(s['ndvi']),s['ndvi'],-9999.).astype('float32')
        evi_out=np.where(np.isfinite(s['evi']),s['evi'],-9999.).astype('float32')
        write_tif(DERIVED/f'28_s2_ndvi_{year}_10m.tif',ndvi_out,p,
                  'NDVI of valid nonwater SCL 4-5 pixels',-9999.)
        write_tif(DERIVED/f'28_s2_evi_{year}_10m.tif',evi_out,p,
                  'EVI of valid nonwater SCL 4-5 pixels (screened)',-9999.)
        print(f'{year}: valid nonwater NDVI pixels = {int(s["land"].sum())}')
    rings,reaches,river_mask,line,reference_meta=reference_zone(scenes)
    pixelha=abs(scenes[REFERENCE_YEAR]['transform'].a*scenes[REFERENCE_YEAR]['transform'].e)/10000.
    annual=annual_rows(scenes,rings,reaches,pixelha)
    pairs=pair_rows(scenes,rings,reaches,pixelha)
    csv_out(REPORTS/'28_s2_annual_riparian_vegetation.csv',annual)
    csv_out(REPORTS/'28_s2_riparian_change_by_reach.csv',pairs)
    zones=np.zeros(river_mask.shape,dtype='uint8')
    zones[river_mask]=1
    zones[rings['0-30m']]=2; zones[rings['30-50m']]=3;zones[rings['50-100m']]=4
    write_tif(DERIVED/'28_s2_fixed_riparian_reference_zones_2020_10m.tif',zones,
              scenes[REFERENCE_YEAR]['profile'],'Fixed 2020 river-associated water and edge analysis zones',0,
              {0:'outside_zone',1:'reference_river_water',2:'river_edge_0_30m',
               3:'river_edge_30_50m',4:'river_edge_50_100m'})
    meta=dict(status='EXPLORATORY_VEGETATION_PROXY_NOT_FIELD_VALIDATED',
              years=YEARS, bands='2018-2025 Sentinel-2 10 m Blue/Red/NIR; 20m SCL nearest regridded',
              primary_index='NDVI=(NIR-Red)/(NIR+Red)',
              auxiliary_index='EVI=2.5*(NIR-Red)/(NIR+6Red-7.5Blue+1)',
              threshold=VEG_NDVI_THRESHOLD, fixed_reference=reference_meta,
              observation_rule='Step22 water_class==0, SCL in 4/5, original valid_pixel, reflectance and index QA',
              comparison='Same 10m stable-nonwater eligible pixels in both images inside fixed 2020 river-edge zones',
              annual_rows=len(annual),comparison_rows=len(pairs),
              limitations=['Not surveyed bankline or confirmed tree-cover loss',
                           '2020 water reference sensitive to water stage and segmentation',
                           'Mixed Sentinel-2 acquisition months and limited validated land data',
                           'Index thresholds and nearest-centreline reach allocation are exploratory',
                           'EVI screened at -1..1 and denominator abs >0.20; changes not compared in primary metric',
                           'No synthetic replacement pixels, no field/reference accuracy validation',
                           'Stable-land comparison excludes land-water transitions (handled separately by morphology analysis)'])
    (REPORTS/'28_s2_riparian_vegetation_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    for a,b in [(2018,2024),(2020,2024)]:
        pic=FIGURES/f'28_s2_riparian_change_{a}_{b}.png'
        plot_pair(scenes,rings,reaches,line,pairs,a,b,pic)
        print('Figure:',pic)
    print('\n========== STEP 28B COMPLETE | EXPLORATORY =========')
    print('Reference main-water pixels:',reference_meta['centreline_anchored_reference_water_pixels'])
    print('Fixed river-edge pixel zones:',reference_meta['fixed_zone_pixel_counts'])
    print('Annual rows:',len(annual),'| Comparison rows:',len(pairs))
    print('Annual report:',REPORTS/'28_s2_annual_riparian_vegetation.csv')
    print('Pair report:',REPORTS/'28_s2_riparian_change_by_reach.csv')
    print('QA manifest:',REPORTS/'28_s2_riparian_vegetation_metadata.json')
    print('WARNING: This is satellite-derived NDVI threshold screening, not independently validated riparian deforestation.')

if __name__=='__main__':
    main()
