"""Step 29D — No-QGIS visual GIS verification for Kelani River urban pressure.

Run in repository root with the existing river_corridor_venv. Place this script in
River_Corridor_Research/src/part1_data_preparation/.

Reads the Step 29C reach GeoPackage and CSV plus raw OSM/building and river files.
Creates two PNG maps and one human-readable QA report. No existing input is edited.
Source-specific static OSM measurements are NOT evidence of illegal encroachment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shapely.geometry import box

PROJECT_CRS = 'EPSG:32644'
EXPECTED_REACHES = 28


def load_vector(path: Path, *, layer: str | None = None) -> gpd.GeoDataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')
    data = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if data.crs is None:
        raise ValueError(f'Missing CRS: {path}')
    return data.to_crs(PROJECT_CRS)


def prepare_osm(path: Path, valid_types: set[str]) -> tuple[gpd.GeoDataFrame, int]:
    data = load_vector(path)
    good = (data.geometry.notna() & ~data.geometry.is_empty & data.geometry.is_valid &
            data.geom_type.isin(valid_types))
    removed = int((~good).sum())
    return data.loc[good].copy(), removed


def safe_view(ax, geometry, pad_m: float = 140.0):
    x0, y0, x1, y1 = geometry.bounds
    ax.set_xlim(x0 - pad_m, x1 + pad_m)
    ax.set_ylim(y0 - pad_m, y1 + pad_m)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_footprints(ax, footprints):
    if not footprints.empty:
        footprints.plot(ax=ax, color='#667783', edgecolor='#38474f', linewidth=0.17,
                        alpha=0.80, zorder=3)


def plot_roads(ax, roads):
    if not roads.empty:
        roads.plot(ax=ax, color='#e19b34', linewidth=0.75, alpha=0.85, zorder=4)


def save_overview(reaches, centreline, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    descriptors = [
        ('building_footprint_pct', 'OSM building footprint coverage', '%'),
        ('road_density_km_per_km2', 'OSM road density', 'km/km²'),
    ]
    for ax, (field, title, unit) in zip(axes, descriptors):
        v = pd.to_numeric(reaches[field], errors='raise')
        vmax = float(max(v.max(), 0.01))
        reaches.plot(ax=ax, column=field, cmap='YlOrRd', vmin=0, vmax=vmax,
                     edgecolor='#5b676e', linewidth=0.36, zorder=2)
        centreline.plot(ax=ax, color='#1769c2', linewidth=1.4, zorder=3)
        ax.set_title(f'{title} ({unit})', fontsize=12, fontweight='bold')
        safe_view(ax, reaches.geometry.union_all(), pad_m=250)
        bar = fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap='YlOrRd'),
                           ax=ax, shrink=0.61, pad=0.01)
        bar.ax.tick_params(labelsize=8)
        for reach_id in ['R020', 'R021']:
            selected = reaches.loc[reaches.reach_id == reach_id]
            if len(selected):
                p = selected.geometry.iloc[0].representative_point()
                ax.annotate(reach_id, (p.x, p.y), xytext=(4, 5),
                            textcoords='offset points', fontsize=8,
                            fontweight='bold', color='#152530', zorder=6)
    fig.suptitle('Kelani River | 28 reach-level mapped development-pressure indicators\n'
                 'Step 29C fixed 2020 river-edge reference: 0–100 m',
                 fontsize=14, fontweight='bold')
    fig.text(0.5, 0.005,
             'Exploratory OSM snapshot; not surveyed/legal encroachment or historical urban growth.',
             ha='center', fontsize=9, color='#4f5964')
    fig.savefig(out_path, dpi=175, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def save_zoom(reaches, centreline, buildings, roads, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    for ax, reach_id in zip(axes, ['R020', 'R021']):
        chosen = reaches.loc[reaches.reach_id == reach_id]
        if len(chosen) != 1:
            raise ValueError(f'Expected exactly one {reach_id} in reach layer')
        footprint = chosen.geometry.iloc[0]
        x0, y0, x1, y1 = footprint.bounds
        frame = box(x0-150, y0-150, x1+150, y1+150)
        subset_b = buildings.loc[buildings.geometry.intersects(frame)]
        subset_r = roads.loc[roads.geometry.intersects(frame)]
        subset_c = centreline.loc[centreline.geometry.intersects(frame)]
        chosen.plot(ax=ax, color='#f8d87f', edgecolor='#bc6249', linewidth=1,
                    alpha=0.64, zorder=2)
        plot_footprints(ax, subset_b)
        plot_roads(ax, subset_r)
        if not subset_c.empty:
            subset_c.plot(ax=ax, color='#1769c2', linewidth=2.1, zorder=5)
        safe_view(ax, footprint, pad_m=150)
        rec = chosen.iloc[0]
        ax.set_title(f'{reach_id} | mapped buildings {rec.building_footprint_pct:.2f}%'
                     f' | roads {rec.road_density_km_per_km2:.2f} km/km²',
                     fontsize=11, fontweight='bold')
        ax.text(0.01, 0.02,
                f'Nearby OSM features displayed: {len(subset_b)} buildings, {len(subset_r)} road ways',
                fontsize=8, transform=ax.transAxes, va='bottom',
                bbox={'facecolor': 'white', 'edgecolor': '#cccccc', 'alpha': 0.85})
    fig.suptitle('Spatial overlay inspection: R020 and R021\n'
                 'Yellow = fixed riparian analysis zone | dark grey = building | orange = road | blue = centreline',
                 fontsize=12, fontweight='bold')
    fig.text(0.5, 0.005,
             'Check by eye for positional misalignment. This plot is NOT independent survey validation.',
             ha='center', fontsize=9, color='#4f5964')
    fig.savefig(out_path, dpi=175, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description='Step 29D no-QGIS visual QA')
    cli.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2],
                     help='River_Corridor_Research project directory')
    args = cli.parse_args()
    root = args.root.resolve()
    osm_dir = root / 'data/raw/osm'
    reach_file = root / 'data/processed/urban_pressure_v1/29_osm_urban_pressure_28_reaches.gpkg'
    report_file = root / 'outputs/reports/29_osm_pressure_by_reach.csv'
    centre_file = root / 'data/raw/boundaries/kelani_river_centreline.geojson'
    out_fig = root / 'outputs/figures'
    out_report = root / 'outputs/reports'

    print('\n========== STEP 29D | GIS VISUAL QA WITHOUT QGIS ==========', flush=True)
    reaches = load_vector(reach_file, layer='reach_pressure')
    if len(reaches) != EXPECTED_REACHES or 'reach_id' not in reaches.columns or reaches.reach_id.nunique() != EXPECTED_REACHES:
        raise ValueError('Expected exactly 28 unique reaches from Step29C')
    required = {'building_footprint_pct', 'road_density_km_per_km2'}
    if not required.issubset(reaches.columns):
        raise ValueError(f'Missing reach indicators: {required-set(reaches.columns)}')
    if reaches.geometry.isna().any() or (~reaches.geometry.is_valid).any() or (~reaches.geom_type.isin(['Polygon', 'MultiPolygon'])).any():
        raise ValueError('Invalid or unexpected reach polygon geometry')

    if not report_file.is_file():
        raise FileNotFoundError(report_file)
    report = pd.read_csv(report_file)
    joined = reaches[['reach_id', *sorted(required)]].merge(
        report[['reach_id', *sorted(required)]], on='reach_id', how='outer',
        suffixes=('_gis', '_csv'), indicator=True, validate='one_to_one')
    if not (joined._merge == 'both').all():
        raise ValueError('Step29C GIS/CSV reach IDs mismatch')
    for col in required:
        if not np.allclose(joined[col+'_gis'], joined[col+'_csv'], rtol=0, atol=0.0015):
            raise ValueError(f'Step29C GIS/CSV {col} mismatch')
    print('PASS: 28 reach indicators consistent between CSV and GeoPackage')

    centreline = load_vector(centre_file)
    buildings, bad_b = prepare_osm(osm_dir/'kelani_osm_buildings_raw.geojson', {'Polygon','MultiPolygon'})
    roads, bad_r = prepare_osm(osm_dir/'kelani_osm_roads_raw.geojson', {'LineString','MultiLineString'})
    print(f'OSM polygons loaded: {len(buildings)} (excluded {bad_b})')
    print(f'OSM road lines loaded: {len(roads)} (excluded {bad_r})')
    if buildings.empty or roads.empty or centreline.empty:
        raise ValueError('Missing spatial context for visual map verification')

    out_fig.mkdir(parents=True, exist_ok=True)
    out_report.mkdir(parents=True, exist_ok=True)
    overview = out_fig/'29d_osm_development_pressure_overview.png'
    zoom = out_fig/'29d_osm_R020_R021_overlay_inspection.png'
    qa_path = out_report/'29d_visual_gis_verification_report.txt'
    save_overview(reaches, centreline, overview)
    save_zoom(reaches, centreline, buildings, roads, zoom)
    lines = [
        'STEP 29D | AUTOMATED GIS CONSISTENCY PASS; VISUAL REVIEW PENDING',
        f'GIS reaches: {len(reaches)}',
        'CSV vs GPKG main indicators: PASS (tolerance 0.0015)',
        f'Usable OSM buildings: {len(buildings)}; excluded: {bad_b}',
        f'Usable OSM roads: {len(roads)}; excluded: {bad_r}',
        'Inspect screenshots manually for river/zone/building/road misalignment.',
        'OSM source date and positional completeness have not been independently validated.',
        'High mapped building density is not proof of illegal encroachment.',
        'No formal QGIS verification or survey validation has been performed.',
    ]
    qa_path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('PASS: automated consistency and PNG generation; human visual review still needed')
    print('Overview:', overview)
    print('Zoom:', zoom)
    print('QA report:', qa_path)
    print('\nSTEP 29D COMPLETE | AUTOMATED PART ONLY\n')


if __name__ == '__main__':
    main()
