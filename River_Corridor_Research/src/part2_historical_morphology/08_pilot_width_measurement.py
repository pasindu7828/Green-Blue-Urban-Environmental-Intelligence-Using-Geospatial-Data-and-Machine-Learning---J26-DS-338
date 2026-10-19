
"""
J26-DS-338 - River Corridor Research

STEP 13: PILOT RIVER-WIDTH MEASUREMENT

Uses:
    271 transects from Step 12
    Landsat pilot water masks from Step 11

Selected years:
    2010, 2015, 2020, 2025

Measures the classified water segment
that contains the transect centre point.

This is NOT a surveyed river width.
Results are NOT approved for ML training yet.
"""

from pathlib import Path
from collections import Counter

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from rasterio.features import shapes
from shapely.geometry import shape, Point
from shapely.ops import unary_union


# ==================================================
# 1. PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

TRANSECT_FILE = (
    ROOT / "data/processed/transects"
    / "01_transects_50m.gpkg"
)

MASK_DIR = ROOT / "data/processed/water_masks"

CENTRELINE_FILE = (
    ROOT / "data/raw/boundaries"
    / "kelani_river_centreline.geojson"
)

OUTPUT_DIR = (
    ROOT / "data/processed/morphology"
)

REPORT_DIR = ROOT / "outputs/reports"

FIGURE_DIR = ROOT / "outputs/figures"

for folder in (OUTPUT_DIR, REPORT_DIR, FIGURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

YEARS = [2010, 2015, 2020, 2025]

MIN_RELIABLE_PIXELS = 2
LANDSAT_RESOLUTION_M = 30.0


# ==================================================
# 2. HELPERS
# ==================================================

def extract_lines(geometry):
    """Collect line segments from any intersection."""

    if geometry.is_empty:
        return []

    if geometry.geom_type == "LineString":
        return [geometry]

    if geometry.geom_type in (
        "MultiLineString",
        "GeometryCollection"
    ):
        result = []
        for part in geometry.geoms:
            result.extend(extract_lines(part))
        return result

    return []


def centreline_intersection_count(geometry):
    """Count distinct point intersections."""

    if geometry.is_empty:
        return 0

    if geometry.geom_type == "Point":
        return 1

    if geometry.geom_type == "MultiPoint":
        return len(geometry.geoms)

    if geometry.geom_type == "GeometryCollection":
        return sum(
            centreline_intersection_count(g)
            for g in geometry.geoms
        )

    # Overlapping lines are ambiguous.
    return -1


# ==================================================
# 3. LOAD TRANSECTS AND CENTRELINE
# ==================================================

print("=" * 70)
print("STEP 13 - PILOT RIVER WIDTH MEASUREMENT")
print("=" * 70)

transects = gpd.read_file(
    TRANSECT_FILE,
    layer="river_transects"
)

centreline = gpd.read_file(
    CENTRELINE_FILE
)

transects = transects.to_crs("EPSG:32644")
centreline = centreline.to_crs("EPSG:32644")

if transects.empty or len(centreline) != 1:
    raise ValueError("Invalid transects or centreline.")

if len(transects) != 271:
    raise ValueError(
        f"Expected 271 transects, got {len(transects)}"
    )

main_line = centreline.geometry.iloc[0]

records = []
maps = {}


# ==================================================
# 4. PROCESS EACH YEAR
# ==================================================

for year in YEARS:

    matches = sorted(
        MASK_DIR.glob(
            f"kelani_landsat_{year}_"
            f"*_pilot_water_mask.tif"
        )
    )

    if len(matches) != 1:
        raise ValueError(
            f"{year}: Expected one pilot mask, "
            f"found {len(matches)}"
        )

    mask_file = matches[0]

    print("\n" + "-" * 70)
    print(f"PROCESSING: {year}")
    print(f"Mask: {mask_file.name}")

    with rasterio.open(mask_file) as src:

        water_mask = src.read(1)

        if src.crs != transects.crs:
            raise ValueError(
                f"{year}: CRS mismatch."
            )

        if not np.allclose(
            src.res, (30.0, 30.0)
        ):
            raise ValueError(
                f"{year}: Unexpected raster resolution."
            )

        if not set(np.unique(water_mask)).issubset(
            {0, 1, 255}
        ):
            raise ValueError(
                f"{year}: Unexpected mask class."
            )

        # Convert only water-class pixels
        # to their actual 30 m raster footprints.
        water_pixels = water_mask == 1

        polygons = [
            shape(geom)
            for geom, value in shapes(
                water_pixels.astype("uint8"),
                mask=water_pixels,
                transform=src.transform,
                connectivity=8
            )
            if int(value) == 1
        ]

        if polygons:
            water_geometry = unary_union(polygons)
        else:
            water_geometry = None

        counts = Counter()

        # ------------------------------------------
        # MEASURE AT EACH TRANSECT
        # ------------------------------------------

        for _, row in transects.iterrows():

            transect = row.geometry

            centre = Point(
                float(row["centre_x"]),
                float(row["centre_y"])
            )

            width = None
            status = "UNDETERMINED"

            # Check that the transect has only
            # one intersection with the centreline.
            intersection = transect.intersection(
                main_line
            )

            intersection_count = (
                centreline_intersection_count(
                    intersection
                )
            )

            if intersection_count != 1:

                status = "AMBIGUOUS_CENTRELINE"

            elif water_geometry is None:

                status = "NO_WATER_DETECTED"

            elif not water_geometry.covers(centre):

                status = "CENTRE_NOT_WATER"

            else:

                crossed_water = transect.intersection(
                    water_geometry
                )

                segments = extract_lines(
                    crossed_water
                )

                # Select only water segments touching
                # the transect's centre point.
                centre_segments = [
                    segment
                    for segment in segments
                    if segment.distance(centre) < 0.05
                ]

                if len(centre_segments) != 1:

                    status = "AMBIGUOUS_WATER_SEGMENT"

                else:

                    segment = centre_segments[0]

                    start = Point(
                        transect.coords[0]
                    )

                    end = Point(
                        transect.coords[-1]
                    )

                    if (
                        segment.distance(start) < 0.05
                        or segment.distance(end) < 0.05
                    ):

                        status = "TRANSECT_TRUNCATED"

                    else:

                        width = round(
                            float(segment.length), 1
                        )

                        if width < (
                            MIN_RELIABLE_PIXELS
                            * LANDSAT_RESOLUTION_M
                        ):

                            status = "LOW_RESOLUTION"

                        else:

                            status = "PILOT_UNVALIDATED"

            counts[status] += 1

            records.append({
                "year": year,
                "transect_id": row["transect_id"],
                "reach_id": row["reach_id"],
                "chainage_m": row["chainage_m"],
                "pilot_width_m": width,
                "qa_status": status,
                "centreline_intersections":
                    intersection_count,
                "sensor": "Landsat",
                "resolution_m": 30,
                "validated": False,
                "ml_training_approved": False
            })

        print(f"Transects checked: {len(transects)}")

        for status, count in sorted(counts.items()):
            print(f"  {status}: {count}")

        if year == 2025:
            maps["array"] = water_mask.copy()
            maps["bounds"] = src.bounds


# ==================================================
# 5. EXPORT PRELIMINARY DATASET
# ==================================================

df = pd.DataFrame(records)

csv_path = (
    OUTPUT_DIR / "08_pilot_widths.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ==================================================
# 6. SUMMARIZE QA BY YEAR
# ==================================================

summary_rows = []

for year in YEARS:

    subset = df[df["year"] == year]

    measured = subset[
        subset["pilot_width_m"].notna()
    ]

    usable = subset[
        subset["qa_status"] == "PILOT_UNVALIDATED"
    ]

    summary_rows.append({
        "year": year,
        "transects_checked": len(subset),
        "widths_measured": len(measured),
        "widths_at_least_60m": len(usable),
        "missing_or_flagged": (
            len(subset) - len(usable)
        ),
        "median_pilot_width_m": (
            measured["pilot_width_m"].median()
            if not measured.empty else np.nan
        ),
        "status": "NOT_VALIDATED_FOR_ML"
    })

summary = pd.DataFrame(summary_rows)

qa_path = (
    REPORT_DIR / "08_pilot_width_qa.csv"
)

summary.to_csv(
    qa_path,
    index=False
)


# ==================================================
# 7. CREATE 2025 VISUAL QA MAP
# ==================================================

fig, ax = plt.subplots(
    figsize=(14, 7)
)

water = maps["array"]
bounds = maps["bounds"]

extent = [
    bounds.left,
    bounds.right,
    bounds.bottom,
    bounds.top
]

display = np.ma.masked_where(
    water != 1,
    water.astype(float)
)

cmap = plt.get_cmap("Blues").copy()
cmap.set_bad((0, 0, 0, 0))

ax.imshow(
    display,
    cmap=cmap,
    vmin=0,
    vmax=1,
    extent=extent,
    origin="upper"
)

# Display one in every ten transects
# to keep this overview readable.
transects.iloc[::10].plot(
    ax=ax,
    color="orange",
    linewidth=0.8,
    label="Sample transects"
)

centreline.plot(
    ax=ax,
    color="red",
    linewidth=1,
    label="Centreline"
)

ax.set_title(
    "2025 - Pilot Water Pixels and Transects"
)

ax.set_xlabel("UTM Easting (m)")
ax.set_ylabel("UTM Northing (m)")
ax.set_aspect("equal")

fig.tight_layout()

figure_path = (
    FIGURE_DIR / "08_width_pilot_qa.png"
)

fig.savefig(
    figure_path,
    dpi=150
)

plt.close(fig)


# ==================================================
# 8. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 70)
print("STEP 13 - PILOT WIDTH SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

print("\nQA status counts:")
print(
    df["qa_status"].value_counts().to_string()
)

print("\nOutputs:")
print(csv_path)
print(qa_path)
print(figure_path)

print("\nIMPORTANT:")
print("These are 30 m pixel-based pilot measurements.")
print("They are NOT validated riverbank widths.")
print("ML training approval remains FALSE.")

print("\nSTEP 13 FINISHED")
print("=" * 70)
