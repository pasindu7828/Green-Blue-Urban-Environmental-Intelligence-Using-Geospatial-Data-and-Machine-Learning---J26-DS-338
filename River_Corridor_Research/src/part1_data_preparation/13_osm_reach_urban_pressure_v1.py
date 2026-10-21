"""Step 29C | Kelani River static OSM development-pressure measurement.

Uses the EXACT fixed 2020-water-reference river-edge zones from Step 28B
and regenerates the Step 28B nearest-5m centreline reach attribution. Reports
OSM building footprint and road length within the 28 reach's 0-30m, 30-50m,
50-100m and combined 0-100m analysis zones. QGIS-compatible GeoPackages.

Static OSM snapshots are NOT evidence of construction dates, urban growth,
illegal encroachment, surveyed riverbank positions, or legal reservation zones.
Only writes prefix 29_osm_ products. All existing source files are read-only.
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window, transform as window_transform
from scipy.spatial import cKDTree
from shapely.geometry import shape
from shapely import union_all

ROOT = Path(__file__).resolve().parents[2]
RAW_OSM = ROOT / 'data/raw/osm'
BOUNDARIES = ROOT / 'data/raw/boundaries'
REFERENCE = ROOT / 'data/processed/riparian_vegetation_v2/28_s2_fixed_riparian_reference_zones_2020_10m.tif'
VEG_REPORT = ROOT / 'outputs/reports/28_s2_annual_riparian_vegetation.csv'
OUTPUT_GIS = ROOT / 'data/processed/urban_pressure_v1'
REPORTS = ROOT / 'outputs/reports'
FIGURES = ROOT / 'outputs/figures'
CRS = 'EPSG:32644'
REACH_M = 500.0
SAMPLE_M = 5.0
RINGS = {'0-30m': (2,), '30-50m': (3,), '50-100m': (4,), '0-100m': (2,3,4)}
EXPECTED_REACHES = 28


def load_reference():
    if not REFERENCE.exists():
        raise FileNotFoundError('Run Step 28B first: missing ' + str(REFERENCE))
    with rasterio.open(REFERENCE) as ds:
        if ds.crs is None or ds.crs.to_string().upper() != CRS or ds.count != 1:
            raise ValueError('Reference zone CRS/band mismatch; cannot match Step28')
        if not (abs(ds.res[0]-10)<1e-5 and abs(ds.res[1]-10)<1e-5):
            raise ValueError('Expected original 10m Step28 zone grid')
        zones = ds.read(1)
        transform = ds.transform
        raster_crs = ds.crs
    if not np.isin(zones, [0,1,2,3,4]).all():
        raise ValueError('Unexpected reference zone raster class; inspect Step28 metadata')
    print('Step28 reference raster classes:', {str(k): int((zones==k).sum()) for k in [0,1,2,3,4]})
    return zones, transform, raster_crs


def construct_reach_map(zones, transform, raster_crs):
    path = BOUNDARIES / 'kelani_river_centreline.geojson'
    centre = gpd.read_file(path)
    if len(centre) != 1 or centre.crs is None or centre.geometry.isna().any():
        raise ValueError('Expected one valid, georeferenced centreline')
    line = centre.to_crs(raster_crs).geometry.iloc[0]
    if line.geom_type != 'LineString' or not line.is_valid or line.length < 1000:
        raise ValueError('Expected valid single LineString centreline')
    nreach = math.ceil(line.length / REACH_M)
    if nreach != EXPECTED_REACHES:
        raise ValueError(f'Unexpected centreline reach count {nreach}; do not change Step28 reach IDs')

    positions = np.arange(0, line.length + SAMPLE_M, SAMPLE_M)
    positions = np.minimum(positions, line.length)
    points = np.asarray([(line.interpolate(float(p)).x, line.interpolate(float(p)).y)
                         for p in positions], dtype=np.float64)
    rows, cols = np.nonzero((zones >= 2) & (zones <= 4))
    xx, yy = rasterio.transform.xy(transform, rows, cols, offset='center')
    near = cKDTree(points).query(np.column_stack((xx, yy)), k=1)[1]
    r_id = np.minimum((positions[near]/REACH_M).astype(np.int16), nreach-1)
    reach_map = np.full(zones.shape, -1, dtype=np.int16)
    reach_map[rows, cols] = r_id
    if any(not np.any(reach_map == i) for i in range(nreach)):
        raise ValueError('A 500 m reach has no raster zone pixels')
    print('Projected centreline length:', round(line.length, 2), 'metres; reaches:', nreach)
    return reach_map, line, nreach


def verify_against_step28(zones, reach_map, nreach):
    if not VEG_REPORT.exists():
        raise FileNotFoundError('Missing Step28 annual QA report for alignment check')
    df = pd.read_csv(VEG_REPORT)
    required = {'year', 'reach_id', 'zone', 'fixed_zone_pixels'}
    if not required.issubset(df.columns):
        raise ValueError('Step28 report schema mismatch: ' + str(required - set(df.columns)))
    reference = df[df['year'] == 2020]
    if len(reference) != nreach * len(RINGS):
        raise ValueError('Step28 reference-year zone records incomplete')
    for ix in range(nreach):
        for zone, categories in RINGS.items():
            actual = int(np.count_nonzero((reach_map==ix) & np.isin(zones, categories)))
            expected_rows = reference.loc[(reference['reach_id'] == f'R{ix+1:03d}') &
                                          (reference['zone'] == zone), 'fixed_zone_pixels']
            if len(expected_rows) != 1 or int(expected_rows.iloc[0]) != actual:
                raise ValueError(f'Step28/29 reach zone mismatch R{ix+1:03d} {zone}: '
                                 f'{actual} pixels, Step28 {expected_rows.tolist()}')
    print('PASS: all 112 reference reach-zone pixel counts exactly match Step28 outputs')


def load_osm(filename, required_type, category):
    path = RAW_OSM / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    data = gpd.read_file(path)
    if data.crs is None or data.empty:
        raise ValueError('Missing CRS or empty OSM dataset: ' + str(path))
    if category not in data.columns:
        raise ValueError(f'OSM {filename} lacks expected {category} attribute')
    if 'id' not in data.columns:
        raise ValueError(f'OSM {filename} lacks source id, cannot reliably QA features')
    original_count = len(data)
    has_geom = data.geometry.notna() & ~data.geometry.is_empty
    geometries_ok = data.geometry.is_valid
    geom_allowed = data.geom_type.isin(required_type)
    keep = has_geom & geometries_ok & geom_allowed
    discarded = data.loc[~keep, ['id', 'geometry']]
    if discarded.shape[0]:
        print(f'WARNING: {filename} excluded {len(discarded)} invalid/wrong-type features',
              'IDs:', discarded['id'].astype(str).head(5).tolist())
    data = data.loc[keep].copy().to_crs(CRS)
    data = data[['id', category, 'geometry']].reset_index(drop=True)
    data['id'] = data['id'].astype(str)
    duplicated = int(data['id'].duplicated().sum())
    if duplicated:
        warnings.warn(f'{filename}: {duplicated} repeated OSM IDs; counts are per source feature')
    print(f'{filename}: raw={original_count}, usable={len(data)}, excluded={original_count-len(data)}, id_duplicates={duplicated}')
    return data, {'raw':original_count, 'usable':len(data), 'excluded':original_count-len(data),
                  'duplicated_osm_id':duplicated}


def pixel_polygon(mask, transform):
    rr, cc = np.nonzero(mask)
    if len(rr)==0:
        return None
    r0, r1 = int(rr.min()), int(rr.max())+1
    c0, c1 = int(cc.min()), int(cc.max())+1
    subset = mask[r0:r1, c0:c1].astype('uint8')
    t = window_transform(Window(c0, r0, c1-c0, r1-r0), transform)
    polygons = [shape(geom) for geom, code in shapes(subset, mask=subset.astype(bool), transform=t) if code == 1]
    geom = union_all(polygons)
    if geom.is_empty or not geom.is_valid:
        raise ValueError('Invalid pixel-derived polygon geometry')
    actual_area = float(mask.sum())*abs(transform.a*transform.e)
    if abs(geom.area - actual_area) > 0.0001:
        raise AssertionError(f'Polygon/raster area mismatch {geom.area} vs {actual_area}')
    return geom


def clipped_features(gdf, zone_geom, kind):
    if zone_geom is None:
        return [], 0, 0.0
    positions = list(gdf.sindex.query(zone_geom, predicate='intersects'))
    pieces = []
    for i in positions:
        geom = gdf.geometry.iloc[i].intersection(zone_geom)
        if geom.is_empty:
            continue
        measure = geom.area if kind == 'buildings' else geom.length
        if measure <= 1e-8:
            continue
        pieces.append((gdf['id'].iloc[i], geom))
    if kind=='buildings':
        # Dissolve overlaps to avoid percentages exceeding 100% from repeated footprints.
        measure_sum = union_all([x[1] for x in pieces]).area if pieces else 0.0
    else:
        # Roads can overlap if OSM ways are duplicated/colocated; this is cumulative
        # mapped way-segment length, NOT unique pavement length.
        measure_sum = sum(geom.length for _,geom in pieces)
    return pieces, len(pieces), float(measure_sum)


def assemble(zones, transform, reach_map, nreach, buildings, roads):
    records = []
    geometries = []
    for ix in range(nreach):
        print(f'  Urban-pressure spatial overlay R{ix+1:03d}/{nreach} ...', flush=True)
        for zone_name, codes in RINGS.items():
            m = (reach_map==ix) & np.isin(zones, codes)
            region = pixel_polygon(m, transform)
            if region is None:
                raise ValueError(f'Empty zone R{ix+1:03d} {zone_name}')
            b_pieces, b_count, b_area = clipped_features(buildings, region, 'buildings')
            r_pieces, r_count, r_length = clipped_features(roads, region, 'roads')
            area_km2 = region.area / 1_000_000.0
            row = dict(reach_id=f'R{ix+1:03d}', zone=zone_name,
                       reference_year=2020, zone_pixels=int(m.sum()),
                       zone_area_ha=round(region.area/10000.,4),
                       building_features_intersecting=b_count,
                       building_count_per_km2=round(b_count/area_km2,2),
                       building_footprint_ha=round(b_area/10000.,4),
                       building_footprint_pct=round(100*b_area/region.area,3),
                       road_features_intersecting=r_count,
                       road_length_km=round(r_length/1000.,4),
                       road_density_km_per_km2=round((r_length/1000.)/area_km2,3),
                       interpretation='STATIC_OSM_MAPPED_DEVELOPMENT_PROXY_NOT_LEGAL_ENCROACHMENT')
            records.append(row)
            geometries.append(region)
    return records, geometries


def write_outputs(rows, geoms, osm_meta, line):
    OUTPUT_GIS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_REACHES*len(RINGS):
        raise ValueError('Output missing reach/zone rows')
    if df.duplicated(subset=['reach_id','zone']).any():
        raise ValueError('Duplicate reach-zone records')
    combined = df[df.zone=='0-100m'].copy().reset_index(drop=True)
    if len(combined)!=EXPECTED_REACHES:
        raise ValueError('Not exactly 28 combined reach rows')
    df.to_csv(REPORTS/'29_osm_pressure_by_reach_and_zone.csv', index=False)
    combined.to_csv(REPORTS/'29_osm_pressure_by_reach.csv', index=False)
    gs = gpd.GeoDataFrame(df, geometry=geoms, crs=CRS)
    gs.to_file(OUTPUT_GIS/'29_osm_urban_pressure_zones.gpkg', layer='reach_zones', driver='GPKG')
    gs[gs.zone=='0-100m'].to_file(OUTPUT_GIS/'29_osm_urban_pressure_28_reaches.gpkg', layer='reach_pressure', driver='GPKG')
    meta = {'status':'EXPLORATORY_STATIC_OSM_PRESSURE_NOT_ENCROACHMENT',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'source_files':{'buildings':'data/raw/osm/kelani_osm_buildings_raw.geojson',
                            'roads':'data/raw/osm/kelani_osm_roads_raw.geojson',
                            'riparian_reference':'data/processed/riparian_vegetation_v2/28_s2_fixed_riparian_reference_zones_2020_10m.tif',
                            'reach_reference':'data/raw/boundaries/kelani_river_centreline.geojson'},
            'osm_snapshot_date':'UNKNOWN_VERIFY_SEPARATELY',
            'definitions':{'zones':'0-30m, 30-50m, 50-100m and union 0-100m OUTSIDE fixed Jan 2020 water footprint',
                           'geometry':'10m pixel-derived areas, clipped exact-vector intersection',
                           'reach_assignment':'nearest 5m sampled existing centreline; 500m chainage, as Step28',
                           'building_features_intersecting':'unique source rows intersecting each reach-zone with positive area; can count across multiple reach-zones',
                           'building_footprint_pct':'geometric union of building footprints intersected with reach-zone / analysis zone area, not building parcel cover',
                           'road_density_km_per_km2':'sum of clipped OSM way segment lengths in km / zone km2; duplicated overlapping ways not dissolved',
                           'reference_not_legal':'not official riverbank, buffer reservation, or property survey'},
            'quality':osm_meta,
            'limitations':['Static OSM attributes do not establish construction year or urban growth',
                           'Mapped OSM coverage, source completeness and positional accuracy are not independently validated',
                           'Footprint and roads may be absent or incorrectly mapped',
                           '2020 water raster/spectral masks are not surveyed boundaries',
                           'Building/road proximity is not evidence of illegal encroachment',
                           'Adjacent reaches and distance zones may count a given OSM feature more than once',
                           'Reach assignment by nearest sampled centreline is approximate near meanders']}
    (REPORTS/'29_osm_urban_pressure_metadata.json').write_text(json.dumps(meta,indent=2), encoding='utf-8')

    fig, axs = plt.subplots(2,1,figsize=(14,8.5),sharex=True,constrained_layout=True)
    x=np.arange(len(combined))
    axs[0].bar(x,combined['building_footprint_pct'],color='#566bb5')
    axs[0].set_ylabel('OSM building footprint (%)')
    axs[0].grid(axis='y',alpha=.2)
    axs[1].bar(x,combined['road_density_km_per_km2'],color='#319185')
    axs[1].set_ylabel('OSM road length (km/km²)')
    axs[1].grid(axis='y',alpha=.2)
    axs[1].set_xticks(x,combined['reach_id'],rotation=60,fontsize=8)
    fig.suptitle('Kelani River | Static OSM Urban Development Pressure | 0–100m fixed river-edge zones\n'
                 'Not verified illegal encroachment, change over time, or legal setbacks',fontsize=13)
    figure=FIGURES/'29_osm_reach_development_pressure.png'
    fig.savefig(figure,dpi=170,bbox_inches='tight')
    plt.close(fig)
    print('\n========== STEP 29C COMPLETE | EXPLORATORY ==========')
    print('CSV all-zone rows:',len(df),'| Combined-reach rows:',len(combined))
    print('Builds mapped in 0-100m (sum of reach-level intersection counts; potential repeats):',
          int(combined['building_features_intersecting'].sum()))
    print('Road km within 0-100m (sum per reach, possible repeated boundary segments):',
          round(float(combined['road_length_km'].sum()),3))
    print('Top building coverage reaches:')
    print(combined.nlargest(5,'building_footprint_pct')[['reach_id','zone_area_ha','building_features_intersecting','building_footprint_pct','road_density_km_per_km2']].to_string(index=False))
    print('Outputs:',REPORTS/'29_osm_pressure_by_reach.csv',OUTPUT_GIS/'29_osm_urban_pressure_28_reaches.gpkg',figure,sep='\n  ')


def main():
    print('========== STEP 29C | STATIC OSM URBAN DEVELOPMENT PRESSURE ==========')
    zones, transform, raster_crs=load_reference()
    reach_map, line, nreach=construct_reach_map(zones,transform,raster_crs)
    verify_against_step28(zones,reach_map,nreach)
    buildings, bmeta=load_osm('kelani_osm_buildings_raw.geojson',('Polygon','MultiPolygon'),'building')
    roads, rmeta=load_osm('kelani_osm_roads_raw.geojson',('LineString','MultiLineString'),'highway')
    rows,geoms=assemble(zones,transform,reach_map,nreach,buildings,roads)
    write_outputs(rows,geoms,{'buildings':bmeta,'roads':rmeta},line)


if __name__=='__main__':
    main()
