
"""J26-DS-338 | Section 03, Stage 3.3: raw 100 m GIS-MCDA criteria.

Regenerates measurements from source GIS (never from legacy Section 03 outputs).
No normalization, AHP weights, pressure/XAI, or suitability scoring.
Requires GeoPandas, Shapely 2, Rasterio, SciPy, NumPy and Pandas.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds, transform as window_transform
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/processed/development_suitability/raw_criteria"
REPORT = ROOT / "outputs/reports/development_suitability"
GRID = ROOT / "data/processed/urbanization_pressure/grid/kaduwela_100m_modelling_grid.gpkg"
KMC = ROOT / "data/processed/boundaries/kmc/kaduwela_kmc_boundary_utm44n.gpkg"
ROADS = ROOT / "data/processed/osm/kaduwela_roads_2026.gpkg"
BUILDINGS = ROOT / "data/processed/osm/kaduwela_buildings_2026.gpkg"
DEM = ROOT / "data/processed/dem/gedtm30_kmc_30m.tif"

# The agreed access hierarchy excludes the main controlled-access motorway.
MAJOR = {
    "primary", "primary_link", "secondary", "tertiary", "tertiary_link",
    "trunk", "trunk_link", "motorway_link",
}
LOCAL = MAJOR | {"residential", "living_street", "unclassified"}
RADII = (250, 500, 1000)  # alternative scales, not separate AHP criteria
NEAREST_MAX_M = 90.0  # used only if no valid DEM pixel touches a cell


def planar(geometries):
    return shapely.force_2d(np.asarray(geometries, dtype=object))


def clipped_nonempty(geometries, boundary, *, polygon=False):
    clipped = shapely.intersection(planar(geometries), boundary)
    sizes = shapely.area(clipped) if polygon else shapely.length(clipped)
    return clipped[np.asarray(sizes > 0)]


def footprint_area(region, geometries, tree):
    """Union within region avoids counting overlapping OSM footprints twice."""
    idx = tree.query(region, predicate="intersects")
    if len(idx) == 0:
        return 0.0
    pieces = shapely.intersection(geometries[idx], region)
    return float(shapely.area(shapely.union_all(pieces)))


def road_length(region, geometries, tree):
    idx = tree.query(region, predicate="intersects")
    if len(idx) == 0:
        return 0.0
    return float(shapely.length(shapely.intersection(geometries[idx], region)).sum())


class Terrain:
    """DEM pixel-centre zonal mean, then all-touched, then local nearest valid."""

    def __init__(self, path):
        with rasterio.open(path) as src:
            if src.crs.to_epsg() != 32644 or src.count != 1:
                raise ValueError("Primary DEM must be single-band EPSG:32644")
            self.transform = src.transform
            self.height, self.width = src.height, src.width
            elev = src.read(1).astype(np.float64)
            valid = np.isfinite(elev)
            if src.nodata is not None:
                valid &= elev != src.nodata
        if not valid.any():
            raise ValueError("Primary DEM contains no valid pixels")

        self.valid = valid
        self.elevation = elev
        # Fill only for calculating gradients at NoData edges. NoData pixels
        # themselves are NEVER considered valid terrain observations.
        nearest_idx = distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        filled = elev[tuple(nearest_idx)]
        dy, dx = np.gradient(
            filled, abs(self.transform.e), abs(self.transform.a)
        )
        self.slope = np.degrees(np.arctan(np.hypot(dx, dy)))
        rr, cc = np.nonzero(valid)
        xx, yy = self.transform * (cc + 0.5, rr + 0.5)
        self.valid_rows, self.valid_cols = rr, cc
        self.nearest_tree = cKDTree(np.column_stack([xx, yy]))

    def sample(self, geom):
        win = from_bounds(*geom.bounds, transform=self.transform)
        c0 = max(0, int(np.floor(win.col_off)))
        r0 = max(0, int(np.floor(win.row_off)))
        c1 = min(self.width, int(np.ceil(win.col_off + win.width)))
        r1 = min(self.height, int(np.ceil(win.row_off + win.height)))

        if c1 > c0 and r1 > r0:
            window = Window(c0, r0, c1 - c0, r1 - r0)
            shape = (r1 - r0, c1 - c0)
            local_valid = self.valid[r0:r1, c0:c1]
            for all_touched, mode in ((False, "pixel_center"),
                                      (True, "all_touched")):
                inside = geometry_mask(
                    [geom], out_shape=shape,
                    transform=window_transform(window, self.transform),
                    invert=True, all_touched=all_touched,
                )
                selected = inside & local_valid
                if np.any(selected):
                    return (
                        float(np.mean(self.slope[r0:r1, c0:c1][selected])),
                        float(np.mean(self.elevation[r0:r1, c0:c1][selected])),
                        mode,
                        int(selected.sum()),
                        0.0,
                    )

        p = geom.representative_point()
        distance, idx = self.nearest_tree.query([p.x, p.y])
        if float(distance) <= NEAREST_MAX_M:
            r = self.valid_rows[idx]
            c = self.valid_cols[idx]
            return (
                float(self.slope[r, c]), float(self.elevation[r, c]),
                "nearest_valid", 1, float(distance),
            )
        return (np.nan, np.nan, "unresolved", 0, float(distance))


def main():
    print("\n=== STAGE 3.3 | RAW 100 m SUITABILITY CRITERIA ===")
    grid = gpd.read_file(GRID)
    boundary = gpd.read_file(KMC)
    roads = gpd.read_file(ROADS, columns=["highway_clean", "geometry"])
    buildings = gpd.read_file(BUILDINGS, columns=["geometry"])

    for name, layer in (("grid", grid), ("KMC", boundary),
                        ("roads", roads), ("buildings", buildings)):
        if layer.crs is None or layer.crs.to_epsg() != 32644:
            raise ValueError(f"{name}: incorrect CRS")
        if layer.geometry.isna().any() or (~layer.geometry.is_valid).any():
            raise ValueError(f"{name}: null/invalid geometry")

    if len(grid) != 9240 or not grid["grid_id"].is_unique:
        raise ValueError("Unexpected grid count or duplicate IDs")
    if len(boundary) != 1:
        raise ValueError("Expected one official KMC polygon")

    kmc = planar(boundary.geometry.values)[0]
    cell_geoms = shapely.intersection(planar(grid.geometry.values), kmc)
    cell_areas = np.asarray(shapely.area(cell_geoms), dtype=float)
    expected_areas = grid["kmc_overlap_area_m2"].to_numpy(dtype=float)
    area_error = np.abs(cell_areas - expected_areas)
    if (cell_areas <= 0).any() or not np.isfinite(cell_areas).all():
        raise ValueError("One or more grid/KMC intersections have no area")
    if area_error.max() > 1.0:
        raise ValueError(f"Cell area disagreement: {area_error.max():.3f} m2")

    roads["highway_clean"] = roads["highway_clean"].astype(str).str.lower().str.strip()
    major = clipped_nonempty(
        roads.loc[roads.highway_clean.isin(MAJOR), "geometry"].values, kmc
    )
    local = clipped_nonempty(
        roads.loc[roads.highway_clean.isin(LOCAL), "geometry"].values, kmc
    )
    bldgs = clipped_nonempty(buildings.geometry.values, kmc, polygon=True)
    if not len(major) or not len(local) or not len(bldgs):
        raise ValueError("Empty selected road network or building dataset")

    major_tree, local_tree, building_tree = (
        STRtree(major), STRtree(local), STRtree(bldgs)
    )
    terrain = Terrain(DEM)
    print(f"Grid: {len(grid):,}; KMC area: {kmc.area / 1e6:.6f} km2")
    print(f"Clipped features: major roads={len(major):,}, "
          f"local roads={len(local):,}, buildings={len(bldgs):,}")
    print("Radii: roads=250/500/1000 m; buildings=250/500 m")

    results = []
    for i, (geom, cell_area) in enumerate(zip(cell_geoms, cell_areas)):
        # For consistent neighbourhood footprints, use full 100 m grid centre;
        # the measured neighbourhood itself is clipped to official KMC.
        center = shapely.centroid(grid.geometry.iloc[i])
        nearest = major_tree.nearest(geom)
        c1 = float(shapely.distance(geom, major[int(nearest)]))
        cell_built = footprint_area(geom, bldgs, building_tree)
        slope, elevation, terrain_mode, pixel_count, fallback_m = terrain.sample(geom)
        rec = {
            "grid_id": grid["grid_id"].iloc[i],
            "kmc_overlap_area_m2": cell_area,
            "kmc_overlap_fraction": grid["kmc_overlap_fraction"].iloc[i],
            "boundary_cell": grid["boundary_cell"].iloc[i],
            "coverage_group": grid["coverage_group"].iloc[i],
            "c1_major_road_distance_m": c1,
            "c3_building_footprint_pct_cell": 100.0 * cell_built / cell_area,
            "c4_slope_mean_deg": slope,
            "c5_elevation_mean_m": elevation,
            "terrain_sample_method": terrain_mode,
            "terrain_sample_pixel_count": pixel_count,
            "terrain_fallback_distance_m": fallback_m,
        }
        for radius in RADII:
            search_area = shapely.intersection(
                shapely.buffer(center, float(radius)), kmc
            )
            area_km2 = float(shapely.area(search_area)) / 1e6
            rec[f"c2_road_density_{radius}m_km_per_km2"] = (
                road_length(search_area, local, local_tree) / 1000.0 / area_km2
            ) if area_km2 > 0 else np.nan
            if radius in (250, 500):
                rec[f"c3_context_building_pct_{radius}m"] = (
                    100.0 * footprint_area(search_area, bldgs, building_tree)
                    / (area_km2 * 1e6)
                ) if area_km2 > 0 else np.nan
        results.append(rec)
        if (i + 1) % 1000 == 0 or i + 1 == len(grid):
            print(f"Processed {i + 1:,}/{len(grid):,} cells")

    table = pd.DataFrame(results)
    numeric_cols = [
        c for c in table.columns if c.startswith(("c1_", "c2_", "c3_", "c4_", "c5_"))
    ]
    missing_by_col = {c: int(table[c].isna().sum()) for c in numeric_cols}
    methods = dict(Counter(table["terrain_sample_method"]))
    nonfinite = int((~np.isfinite(table[numeric_cols].to_numpy(dtype=float))).sum())
    fractions = [c for c in numeric_cols if c.startswith("c3_")]
    checks = {
        "all_9240_grid_cells": len(table) == 9240,
        "unique_grid_ids": table.grid_id.is_unique,
        "reconciled_cell_areas": float(area_error.max()) <= 1.0,
        "all_raw_criteria_finite": nonfinite == 0,
        "distance_and_density_nonnegative": bool(
            (table[[c for c in numeric_cols if c.startswith(("c1_", "c2_"))]] >= -1e-8).all().all()
        ),
        "building_percentages_between_0_and_100": bool(
            (table[fractions].ge(-1e-6) & table[fractions].le(100.000001)).all().all()
        ),
        "slope_between_0_and_90_degrees": bool(
            table["c4_slope_mean_deg"].between(0, 90).all()
        ),
        "no_unresolved_terrain": methods.get("unresolved", 0) == 0,
    }
    passed = all(checks.values())
    qa = {
        "stage": "3.3",
        "status": "EXPORT_PENDING" if passed else "REVIEW_REQUIRED",
        "source": "original GIS inputs, not legacy 16b results",
        "grid_cells": len(table),
        "kmc_area_km2": round(float(cell_areas.sum()) / 1e6, 6),
        "max_per_cell_area_difference_m2": float(area_error.max()),
        "terrain_methods": methods,
        "terrain_nearest_limit_m": NEAREST_MAX_M,
        "terrain_source": "GEDTM30 only; nearest-DEM-value fill for gradient computation at NoData edges",
        "terrain_sampling_note": "cell-centre pixels first; then all-touched; then nearest valid GEDTM pixel up to 90 m",
        "neighbourhood_note": "buffer around full-grid centre, intersected with official KMC",
        "building_note": "union of intersected building footprints; measured in true KMC-clipped area",
        "criteria_not_finalized": True,
        "standardization_done": False,
        "ahp_done": False,
        "missing_per_criterion": missing_by_col,
        "raw_summary": {
            c: {"min": float(table[c].min()), "median": float(table[c].median()),
                "max": float(table[c].max())} for c in numeric_cols
        },
        "checks": checks,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    qa_path = REPORT / "03b_raw_criteria_qa.json"
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"Terrain sampling methods: {methods}")
    print(f"QA report: {qa_path}")
    if not passed:
        raise SystemExit("STAGE 3.3: REVIEW_REQUIRED. No final CSV/GPKG exported.")

    RAW.mkdir(parents=True, exist_ok=True)
    csv_path = RAW / "03b_raw_criteria_100m.csv"
    gpkg_path = RAW / "03b_raw_criteria_100m.gpkg"
    table.to_csv(csv_path, index=False)
    geo = gpd.GeoDataFrame(table, geometry=list(cell_geoms), crs="EPSG:32644")
    geo.to_file(gpkg_path, layer="raw_criteria_100m", driver="GPKG")
    csv_verified = pd.read_csv(csv_path, usecols=["grid_id"])
    geo_verified = pyogrio.read_info(gpkg_path)
    if (len(csv_verified) != len(table) or not csv_verified.grid_id.is_unique
            or geo_verified["features"] != len(table)):
        raise SystemExit("Export row-count/ID QA failed; review output files")
    qa["status"] = "PASS"
    qa["export_verification"] = {"csv_rows": len(csv_verified),
                                  "gpkg_features": int(geo_verified["features"])}
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(f"CSV: {csv_path}")
    print(f"GPKG: {gpkg_path}")
    print("STAGE 3.3: PASS. Raw criteria only; no normalization or AHP.")


if __name__ == "__main__":
    main()
