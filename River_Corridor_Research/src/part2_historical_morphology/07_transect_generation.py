
"""
J26-DS-338 - River Corridor Research

STEP 12: RIVER TRANSECT GENERATION

Purpose:
    Generate perpendicular measurement transects
    along the manually delineated Kelani River
    centreline.

Input:
    data/raw/boundaries/
    kelani_river_centreline.geojson

Outputs:
    data/processed/transects/
        01_transects_50m.gpkg
        01_transects_50m.csv

    outputs/figures/
        07_transect_layout.png

    outputs/reports/
        07_transect_generation_qa.csv

IMPORTANT:
    Transects are geometrical measurement templates.
    They are NOT measured river widths or banklines.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import LineString


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

CENTRELINE_PATH = (
    ROOT / "data/raw/boundaries"
    / "kelani_river_centreline.geojson"
)

OUTPUT_DIR = (
    ROOT / "data/processed/transects"
)

FIGURE_DIR = ROOT / "outputs/figures"
REPORT_DIR = ROOT / "outputs/reports"

for folder in (OUTPUT_DIR, FIGURE_DIR, REPORT_DIR):
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================
# 2. PARAMETERS
# ==========================================

CRS_UTM = "EPSG:32644"

SPACING_M = 50.0
TRANSECT_LENGTH_M = 300.0
HALF_LENGTH_M = TRANSECT_LENGTH_M / 2

TANGENT_DISTANCE_M = 10.0
REACH_LENGTH_M = 500.0


# ==========================================
# 3. LOAD CENTRELINE
# ==========================================

print("=" * 70)
print("STEP 12 - RIVER TRANSECT GENERATION")
print("=" * 70)

centreline_gdf = gpd.read_file(
    CENTRELINE_PATH
)

if centreline_gdf.empty:
    raise ValueError(
        "River centreline dataset is empty."
    )

if centreline_gdf.crs is None:
    raise ValueError(
        "Centreline CRS is missing."
    )

if len(centreline_gdf) != 1:
    raise ValueError(
        "Expected one river centreline feature."
    )

centreline_utm = centreline_gdf.to_crs(
    CRS_UTM
)

line = centreline_utm.geometry.iloc[0]

if not isinstance(line, LineString):
    raise ValueError(
        "Expected a LineString centreline. "
        "Please inspect the source geometry."
    )

if line.is_empty or not line.is_valid:
    raise ValueError(
        "Invalid river centreline geometry."
    )

length_m = float(line.length)

print(f"Centreline length: {length_m:.2f} m")
print(f"Centreline length: {length_m / 1000:.3f} km")

if not line.is_simple:
    print(
        "WARNING: Centreline contains "
        "self-intersections. Review transects."
    )

if not (10000 <= length_m <= 17000):
    print(
        "WARNING: Unexpected centreline length. "
        "Check the study boundary."
    )


# ==========================================
# 4. GENERATE TRANSECTS
# ==========================================

distances = np.arange(
    SPACING_M / 2,
    length_m,
    SPACING_M
)

records = []

for index, distance in enumerate(distances, 1):

    distance = float(distance)

    centre_point = line.interpolate(
        distance
    )

    before = line.interpolate(
        max(0, distance - TANGENT_DISTANCE_M)
    )

    after = line.interpolate(
        min(length_m, distance + TANGENT_DISTANCE_M)
    )

    dx = after.x - before.x
    dy = after.y - before.y

    tangent_length = np.hypot(dx, dy)

    if tangent_length <= 0:
        raise ValueError(
            f"Cannot calculate direction at {distance} m"
        )

    # Perpendicular unit vector
    nx = -dy / tangent_length
    ny = dx / tangent_length

    start_x = centre_point.x - HALF_LENGTH_M * nx
    start_y = centre_point.y - HALF_LENGTH_M * ny

    end_x = centre_point.x + HALF_LENGTH_M * nx
    end_y = centre_point.y + HALF_LENGTH_M * ny

    transect = LineString([
        (start_x, start_y),
        (end_x, end_y)
    ])

    reach_id = (
        int(distance // REACH_LENGTH_M) + 1
    )

    records.append({
        "transect_id": f"T{index:04d}",
        "reach_id": f"R{reach_id:03d}",
        "chainage_m": round(distance, 2),
        "length_m": round(transect.length, 2),
        "centre_x": round(centre_point.x, 3),
        "centre_y": round(centre_point.y, 3),
        "start_x": round(start_x, 3),
        "start_y": round(start_y, 3),
        "end_x": round(end_x, 3),
        "end_y": round(end_y, 3),
        "geometry": transect
    })


# ==========================================
# 5. CREATE GEOSPATIAL DATASET
# ==========================================

transects_gdf = gpd.GeoDataFrame(
    records,
    geometry="geometry",
    crs=CRS_UTM
)

if transects_gdf.empty:
    raise ValueError(
        "No transects were generated."
    )

if not transects_gdf.geometry.is_valid.all():
    raise ValueError(
        "Invalid transect geometries detected."
    )

intersects_centreline = transects_gdf.geometry.apply(
    lambda geom: geom.intersects(line)
)

if not intersects_centreline.all():
    raise ValueError(
        "Some transects do not intersect the centreline."
    )

print(f"Transects generated: {len(transects_gdf)}")
print(f"Spacing: {SPACING_M} m")
print(f"Transect length: {TRANSECT_LENGTH_M} m")


# ==========================================
# 6. EXPORT GPKG AND CSV
# ==========================================

gpkg_path = (
    OUTPUT_DIR / "01_transects_50m.gpkg"
)

csv_path = (
    OUTPUT_DIR / "01_transects_50m.csv"
)

# This script owns only this generated GPKG.
# Rerunning replaces this output file.
if gpkg_path.exists():
    gpkg_path.unlink()

transects_gdf.to_file(
    gpkg_path,
    layer="river_transects",
    driver="GPKG",
    index=False
)

transects_gdf.drop(
    columns="geometry"
).to_csv(
    csv_path,
    index=False
)

print("GIS and CSV outputs saved.")


# ==========================================
# 7. GENERATE TRANSECT QA MAP
# ==========================================

fig, ax = plt.subplots(
    figsize=(15, 7)
)

transects_gdf.plot(
    ax=ax,
    color="steelblue",
    linewidth=0.6,
    alpha=0.65
)

centreline_utm.plot(
    ax=ax,
    color="red",
    linewidth=1.5
)

ax.set_title(
    "Kelani River - Preliminary Measurement Transects"
)

ax.set_xlabel("UTM Easting (m)")
ax.set_ylabel("UTM Northing (m)")
ax.set_aspect("equal")
ax.grid(alpha=0.2)

fig.tight_layout()

figure_path = (
    FIGURE_DIR / "07_transect_layout.png"
)

fig.savefig(
    figure_path,
    dpi=160
)

plt.close(fig)

print("Transect layout map saved.")


# ==========================================
# 8. SAVE QUALITY REPORT
# ==========================================

qa_records = [{
    "centreline_length_m": round(length_m, 3),
    "spacing_m": SPACING_M,
    "transect_length_m": TRANSECT_LENGTH_M,
    "tangent_distance_m": TANGENT_DISTANCE_M,
    "number_of_transects": len(transects_gdf),
    "number_of_reaches": transects_gdf[
        "reach_id"
    ].nunique(),
    "all_geometry_valid": bool(
        transects_gdf.geometry.is_valid.all()
    ),
    "all_intersect_centreline": bool(
        intersects_centreline.all()
    ),
    "validation_status": (
        "GEOMETRY_QA_ONLY_VISUAL_REVIEW_PENDING"
    )
}]

qa_path = (
    REPORT_DIR / "07_transect_generation_qa.csv"
)

pd.DataFrame(
    qa_records
).to_csv(
    qa_path,
    index=False
)


# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 70)
print("STEP 12 - TRANSECT GENERATION SUMMARY")
print("=" * 70)

print(f"Centreline length : {length_m:.2f} m")
print(f"Transects created : {len(transects_gdf)}")
print(
    f"500 m reaches     : "
    f"{transects_gdf['reach_id'].nunique()}"
)

print("\nOUTPUT FILES:")
print(gpkg_path)
print(csv_path)
print(figure_path)
print(qa_path)

print("\nIMPORTANT:")
print("These are preliminary transect geometries.")
print("No river width has been measured yet.")

print("\nSTEP 12 FINISHED")
print("=" * 70)
