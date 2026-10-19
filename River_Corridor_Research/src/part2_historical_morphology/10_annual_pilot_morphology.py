
"""
J26-DS-338 - River Corridor Research

STEP 15: ANNUAL PILOT MORPHOLOGY DATASET

- Process 2010, 2011 and 2014-2025
- Exclude gap-affected 2012 and 2013
- Apply Step 13 pilot measurement logic
- Verify results against Step 13 milestones
- Create consecutive-year candidate pairs

Outputs are unvalidated remote-sensing proxies.
They are NOT confirmed erosion measurements
or approved supervised ML labels.
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio

from rasterio.features import shapes
from shapely.geometry import Point, shape
from shapely.ops import unary_union


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

MASK_DIR = ROOT / "data/processed/water_masks"
MORPH_DIR = ROOT / "data/processed/morphology"
REPORT_DIR = ROOT / "outputs/reports"
FIG_DIR = ROOT / "outputs/figures"

TRANSECT_PATH = (
    ROOT / "data/processed/transects"
    / "01_transects_50m.gpkg"
)

CENTRELINE_PATH = (
    ROOT / "data/raw/boundaries"
    / "kelani_river_centreline.geojson"
)

MILESTONE_PATH = MORPH_DIR / "08_pilot_widths.csv"

YEARS = [2010, 2011] + list(range(2014, 2026))
MILESTONES = [2010, 2015, 2020, 2025]

PIXEL_SIZE_M = 30.0
MIN_SCREENED_WIDTH_M = 60.0

for folder in [MORPH_DIR, REPORT_DIR, FIG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. GEOMETRY HELPERS
# ==========================================

def line_parts(geometry):
    if geometry.is_empty:
        return []

    if geometry.geom_type == "LineString":
        return [geometry]

    if geometry.geom_type in (
        "MultiLineString",
        "GeometryCollection"
    ):
        parts = []
        for component in geometry.geoms:
            parts.extend(line_parts(component))
        return parts

    return []


def point_intersections(geometry):
    if geometry.is_empty:
        return 0

    if geometry.geom_type == "Point":
        return 1

    if geometry.geom_type == "MultiPoint":
        return len(geometry.geoms)

    if geometry.geom_type == "GeometryCollection":
        values = [
            point_intersections(g)
            for g in geometry.geoms
        ]
        if -1 in values:
            return -1
        return sum(values)

    return -1


def measure_pilot_width(transect, centre, river, water):
    """
    Match Step 13's preliminary
    water-width extraction logic.
    """

    intersections = point_intersections(
        transect.intersection(river)
    )

    if intersections != 1:
        return np.nan, "AMBIGUOUS_CENTRELINE"

    if water is None:
        return np.nan, "NO_WATER_DETECTED"

    if not water.covers(centre):
        return np.nan, "CENTRE_NOT_WATER"

    segments = line_parts(
        transect.intersection(water)
    )

    touching = [
        segment
        for segment in segments
        if segment.distance(centre) < 0.05
    ]

    if len(touching) != 1:
        return np.nan, "AMBIGUOUS_WATER_SEGMENT"

    segment = touching[0]

    start = Point(transect.coords[0])
    end = Point(transect.coords[-1])

    if (
        segment.distance(start) < 0.05
        or segment.distance(end) < 0.05
    ):
        return np.nan, "TRANSECT_TRUNCATED"

    width = round(float(segment.length), 1)

    if width < MIN_SCREENED_WIDTH_M:
        return width, "LOW_RESOLUTION"

    return width, "PILOT_UNVALIDATED"


# ==========================================
# 3. LOAD FIXED MEASUREMENT LOCATIONS
# ==========================================

print("=" * 75)
print("STEP 15 - ANNUAL PILOT MORPHOLOGY")
print("=" * 75)

transects = gpd.read_file(
    TRANSECT_PATH,
    layer="river_transects"
).to_crs("EPSG:32644")

centreline = gpd.read_file(
    CENTRELINE_PATH
).to_crs("EPSG:32644")

if len(transects) != 271:
    raise ValueError("Unexpected number of transects.")

if len(centreline) != 1:
    raise ValueError("Expected one river centreline.")

river = centreline.geometry.iloc[0]

records = []
qa_rows = []


# ==========================================
# 4. PROCESS ANNUAL WATER MASKS
# ==========================================

for year in YEARS:

    candidates = sorted(
        MASK_DIR.glob(
            f"kelani_landsat_{year}_"
            f"*_pilot_water_mask.tif"
        )
    )

    if len(candidates) != 1:
        raise ValueError(
            f"{year}: Expected one water mask, "
            f"found {len(candidates)}"
        )

    mask_path = candidates[0]

    with rasterio.open(mask_path) as src:

        if src.crs != transects.crs:
            raise ValueError(
                f"{year}: CRS mismatch."
            )

        if not np.allclose(
            src.res,
            (PIXEL_SIZE_M, PIXEL_SIZE_M)
        ):
            raise ValueError(
                f"{year}: Unexpected resolution."
            )

        mask = src.read(1)

        if not set(np.unique(mask)).issubset(
            {0, 1, 255}
        ):
            raise ValueError(
                f"{year}: Invalid mask classes."
            )

        water_pixels = mask == 1

        polygons = [
            shape(geometry)
            for geometry, value in shapes(
                water_pixels.astype("uint8"),
                mask=water_pixels,
                transform=src.transform,
                connectivity=8
            )
            if int(value) == 1
        ]

        water_geometry = (
            unary_union(polygons)
            if polygons else None
        )

    year_counts = Counter()

    for _, row in transects.iterrows():

        centre = Point(
            float(row["centre_x"]),
            float(row["centre_y"])
        )

        width, status = measure_pilot_width(
            row.geometry,
            centre,
            river,
            water_geometry
        )

        year_counts[status] += 1

        records.append({
            "year": year,
            "transect_id": row["transect_id"],
            "reach_id": row["reach_id"],
            "chainage_m": row["chainage_m"],
            "pilot_width_m": width,
            "qa_status": status,
            "pilot_screened": (
                status == "PILOT_UNVALIDATED"
            ),
            "resolution_m": 30,
            "validated": False,
            "ml_training_approved": False
        })

    qa_rows.append({
        "year": year,
        "total_transects": len(transects),
        "pilot_screened": year_counts[
            "PILOT_UNVALIDATED"
        ],
        "low_resolution": year_counts[
            "LOW_RESOLUTION"
        ],
        "centre_not_water": year_counts[
            "CENTRE_NOT_WATER"
        ],
        "other_qa_flags": (
            len(transects)
            - year_counts["PILOT_UNVALIDATED"]
            - year_counts["LOW_RESOLUTION"]
            - year_counts["CENTRE_NOT_WATER"]
        ),
        "screened_pct": round(
            year_counts["PILOT_UNVALIDATED"]
            / len(transects) * 100,
            2
        ),
        "source_mask": mask_path.name,
        "status": "PILOT_ONLY"
    })

    print(
        f"{year}: "
        f"screened={year_counts['PILOT_UNVALIDATED']}, "
        f"low_res={year_counts['LOW_RESOLUTION']}, "
        f"centre_not_water="
        f"{year_counts['CENTRE_NOT_WATER']}"
    )


annual = pd.DataFrame(records)
qa = pd.DataFrame(qa_rows)

expected_rows = len(YEARS) * len(transects)

if len(annual) != expected_rows:
    raise ValueError(
        "Annual dataset has an unexpected row count."
    )

if annual.duplicated(
    ["year", "transect_id"]
).any():
    raise ValueError("Duplicate year-transect IDs.")


# ==========================================
# 5. REPRODUCIBILITY CHECK
# ==========================================

previous = pd.read_csv(MILESTONE_PATH)

current_subset = annual[
    annual["year"].isin(MILESTONES)
]

comparison = current_subset.merge(
    previous[
        [
            "year",
            "transect_id",
            "pilot_width_m",
            "qa_status"
        ]
    ],
    on=["year", "transect_id"],
    how="inner",
    validate="one_to_one",
    suffixes=("_new", "_previous")
)

if len(comparison) != 4 * len(transects):
    raise ValueError(
        "Milestone comparison is incomplete."
    )

same_status = (
    comparison["qa_status_new"]
    == comparison["qa_status_previous"]
)

same_width = np.isclose(
    comparison["pilot_width_m_new"],
    comparison["pilot_width_m_previous"],
    atol=0.11,
    equal_nan=True
)

if not (same_status.all() and same_width.all()):
    raise ValueError(
        "Annual measurements disagree "
        "with Step 13 milestone results."
    )

print("\nMilestone reproducibility: PASS")


# ==========================================
# 6. CREATE CONSECUTIVE-YEAR PAIRS
# ==========================================

pair_tables = []

for year in YEARS:

    if year + 1 not in YEARS:
        continue

    current = annual[
        annual["year"] == year
    ][[
        "transect_id",
        "reach_id",
        "chainage_m",
        "pilot_width_m",
        "pilot_screened"
    ]].rename(columns={
        "pilot_width_m": "current_width_m",
        "pilot_screened": "current_screened"
    })

    following = annual[
        annual["year"] == year + 1
    ][[
        "transect_id",
        "pilot_width_m",
        "pilot_screened"
    ]].rename(columns={
        "pilot_width_m": "next_width_m",
        "pilot_screened": "next_screened"
    })

    paired = current.merge(
        following,
        on="transect_id",
        validate="one_to_one"
    )

    paired = paired[
        paired["current_screened"]
        & paired["next_screened"]
    ].copy()

    paired["base_year"] = year
    paired["target_year"] = year + 1

    paired["pilot_change_m"] = (
        paired["next_width_m"]
        - paired["current_width_m"]
    )

    paired["validated"] = False
    paired["ml_training_approved"] = False

    pair_tables.append(paired)

pairs = pd.concat(
    pair_tables,
    ignore_index=True
)

if pairs.empty:
    raise ValueError(
        "No consecutive-year screened pairs."
    )

if not (
    pairs["target_year"]
    == pairs["base_year"] + 1
).all():
    raise ValueError(
        "Non-consecutive temporal pair detected."
    )


# ==========================================
# 7. EXPORT DATA AND QA
# ==========================================

annual_path = (
    MORPH_DIR / "10_annual_pilot_widths.csv"
)

pairs_path = (
    MORPH_DIR / "10_next_year_pilot_pairs.csv"
)

qa_path = (
    REPORT_DIR / "10_annual_pilot_qa.csv"
)

annual.to_csv(
    annual_path, index=False
)

pairs.to_csv(
    pairs_path, index=False
)

qa.to_csv(
    qa_path, index=False
)


# ==========================================
# 8. QA FIGURE
# ==========================================

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.bar(
    qa["year"].astype(str),
    qa["pilot_screened"]
)

ax.set_title(
    "Annual Transects Passing Pilot Screening"
)

ax.set_ylabel(
    "Number of transects"
)

ax.set_xlabel(
    "Landsat observation year"
)

ax.set_ylim(
    0, len(transects) + 15
)

ax.tick_params(
    axis="x",
    rotation=45
)

fig.tight_layout()

fig_path = (
    FIG_DIR / "10_annual_pilot_qa.png"
)

fig.savefig(fig_path, dpi=150)
plt.close(fig)


# ==========================================
# 9. SUMMARY
# ==========================================

print("\n" + "=" * 75)
print("STEP 15 - FINAL SUMMARY")
print("=" * 75)

print(f"Annual observation years: {len(YEARS)}")
print(f"Total annual records: {len(annual)}")
print(f"Consecutive-year screened pairs: {len(pairs)}")

print("\nAnnual screening:")
print(
    qa[
        [
            "year",
            "pilot_screened",
            "low_resolution",
            "centre_not_water",
            "screened_pct"
        ]
    ].to_string(index=False)
)

print("\nTemporal pair counts:")
print(
    pairs.groupby(
        ["base_year", "target_year"]
    ).size().to_string()
)

print("\nSaved outputs:")
print(annual_path)
print(pairs_path)
print(qa_path)
print(fig_path)

print("\nIMPORTANT:")
print("2012 and 2013 excluded due to gap quality.")
print("These are preliminary 30m water-width proxies.")
print("No validated bank erosion target exists.")
print("ML training approval remains FALSE.")

print("\nSTEP 15 FINISHED")
