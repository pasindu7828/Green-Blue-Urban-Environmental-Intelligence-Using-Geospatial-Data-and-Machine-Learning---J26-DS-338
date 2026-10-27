from pathlib import Path
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import theilslopes


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

EXPERIMENT_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]

INPUT_PATH = (
    EXPERIMENT_ROOT
    / "outputs"
    / "01_validated_history.csv"
)

TRAJECTORY_DIR = (
    EXPERIMENT_ROOT
    / "outputs"
    / "trajectory"
)

REPORT_DIR = (
    EXPERIMENT_ROOT
    / "outputs"
    / "reports"
)

GEOMETRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kaduwela_grid_geometry_wgs84.geojson"
)

REFERENCE_TRAJECTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "23b_grid_heat_trajectory_classes.csv"
)

TRAJECTORY_CSV = (
    TRAJECTORY_DIR
    / "02_dynamic_trajectory_classes.csv"
)

PERCENTILE_CSV = (
    TRAJECTORY_DIR
    / "02_dynamic_relative_heat_percentiles.csv"
)

COUNT_CSV = (
    TRAJECTORY_DIR
    / "02_dynamic_trajectory_class_counts.csv"
)

TRAJECTORY_GEOJSON = (
    TRAJECTORY_DIR
    / "02_dynamic_trajectory_wgs84.geojson"
)

AUDIT_PATH = (
    REPORT_DIR
    / "02_dynamic_trajectory_audit.csv"
)

METHODOLOGY_PATH = (
    REPORT_DIR
    / "02_dynamic_trajectory_methodology.json"
)


# ============================================================
# SETTINGS - SAME AS VALIDATED STEP 23B v2
# ============================================================

EXPECTED_GRIDS = 2194

HOT_THRESHOLD = 0.75
STABLE_SPLIT = 0.50
SLOPE_THRESHOLD = 0.010

WINDOW_SIZE = 5

EXPECTED_CLASSES = {
    "Persistent Cool",
    "Diminishing",
    "Emerging",
    "Intensifying",
    "Persistent Hot",
}


# ============================================================
# HELPERS
# ============================================================

def section(title):

    print("\n" + "=" * 125)
    print(title)
    print("=" * 125)


def classify_trajectory(
    slope,
    baseline_median,
    recent_median,
    long_term_median,
):

    # Emerging
    if (
        slope >= SLOPE_THRESHOLD
        and baseline_median < HOT_THRESHOLD
        and recent_median >= HOT_THRESHOLD
    ):
        return "Emerging"

    # Intensifying
    if slope >= SLOPE_THRESHOLD:
        return "Intensifying"

    # Diminishing
    if slope <= -SLOPE_THRESHOLD:
        return "Diminishing"

    # Stable relative heat position
    if long_term_median >= STABLE_SPLIT:
        return "Persistent Hot"

    return "Persistent Cool"


# ============================================================
# PREPARE
# ============================================================

TRAJECTORY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

section(
    "DYNAMIC 5-YEAR FORECAST - STEP 02 REFRESH HEAT TRAJECTORY"
)

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        f"Step 01 validated history missing:\n{INPUT_PATH}"
    )

df = pd.read_csv(
    INPUT_PATH,
    dtype={
        "grid_id": str,
        "area_name": str,
    },
)

years = sorted(
    df["year"]
    .astype(int)
    .unique()
    .tolist()
)

latest_year = int(
    max(years)
)

baseline_years = (
    years[:WINDOW_SIZE]
)

recent_years = (
    years[-WINDOW_SIZE:]
)

print(
    f"Observed years       : "
    f"{years[0]}-{years[-1]}"
)

print(
    f"Unique grids         : "
    f"{df['grid_id'].nunique():,}"
)

print(
    f"Baseline window      : "
    f"{baseline_years}"
)

print(
    f"Recent window        : "
    f"{recent_years}"
)

print(
    f"Latest observed year : "
    f"{latest_year}"
)


# ============================================================
# 1. WITHIN-YEAR RELATIVE HEAT PERCENTILES
# ============================================================

section(
    "1. CALCULATE WITHIN-YEAR RELATIVE HEAT PERCENTILES"
)

df["relative_heat_percentile"] = (
    df.groupby(
        "year"
    )["LST_C"]
    .rank(
        method="average",
        pct=True,
    )
)


year_check = (
    df.groupby(
        "year"
    )
    .agg(
        grid_count=(
            "grid_id",
            "nunique",
        ),
        min_percentile=(
            "relative_heat_percentile",
            "min",
        ),
        median_percentile=(
            "relative_heat_percentile",
            "median",
        ),
        max_percentile=(
            "relative_heat_percentile",
            "max",
        ),
    )
    .reset_index()
)

print(
    year_check.to_string(
        index=False
    )
)

df[
    [
        "grid_id",
        "year",
        "LST_C",
        "area_name",
        "relative_heat_percentile",
    ]
].to_csv(
    PERCENTILE_CSV,
    index=False,
)


# ============================================================
# 2. ALL-YEAR RELATIVE THEIL-SEN SLOPE
# ============================================================

section(
    "2. CALCULATE ALL-YEAR RELATIVE THEIL-SEN SLOPE"
)

records = []

for grid_id, group in df.groupby(
    "grid_id",
    sort=True,
):

    group = (
        group
        .sort_values(
            "year"
        )
        .copy()
    )

    x = (
        group["year"]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        group[
            "relative_heat_percentile"
        ]
        .to_numpy(
            dtype=float
        )
    )

    slope_result = theilslopes(
        y,
        x,
        alpha=0.95,
    )

    slope = float(
        slope_result.slope
    )

    slope_low = float(
        slope_result.low_slope
    )

    slope_high = float(
        slope_result.high_slope
    )


    baseline_median = float(
        group[
            group["year"].isin(
                baseline_years
            )
        ][
            "relative_heat_percentile"
        ]
        .median()
    )

    recent_median = float(
        group[
            group["year"].isin(
                recent_years
            )
        ][
            "relative_heat_percentile"
        ]
        .median()
    )

    long_term_median = float(
        group[
            "relative_heat_percentile"
        ]
        .median()
    )


    trajectory_class = (
        classify_trajectory(
            slope,
            baseline_median,
            recent_median,
            long_term_median,
        )
    )


    if slope >= SLOPE_THRESHOLD:

        direction = "Increasing"

    elif slope <= -SLOPE_THRESHOLD:

        direction = "Decreasing"

    else:

        direction = "Stable"


    area_name = str(
        group[
            "area_name"
        ]
        .iloc[0]
    )


    records.append(
        {
            "grid_id":
                str(grid_id),

            "area_name":
                area_name,

            "observed_start_year":
                int(years[0]),

            "latest_observed_year":
                latest_year,

            "baseline_window":
                "-".join(
                    map(
                        str,
                        baseline_years,
                    )
                ),

            "recent_window":
                "-".join(
                    map(
                        str,
                        recent_years,
                    )
                ),

            "baseline_relative_heat_median":
                baseline_median,

            "recent_relative_heat_median":
                recent_median,

            "long_term_relative_heat_median":
                long_term_median,

            "relative_sen_slope_percentile_per_year":
                slope,

            "relative_sen_ci95_lower":
                slope_low,

            "relative_sen_ci95_upper":
                slope_high,

            "relative_direction_operational":
                direction,

            "trajectory_class":
                trajectory_class,

            "trajectory_scope":
                (
                    "Relative spatial heat trajectory "
                    "within Kaduwela"
                ),

            "trajectory_is_absolute_warming_claim":
                False,

            "trajectory_is_statistical_significance_claim":
                False,

            "trajectory_rule_version":
                "relative_sen_v2_dynamic",
        }
    )


trajectory = pd.DataFrame(
    records
)


# ============================================================
# 3. VALIDATE CLASSES
# ============================================================

section(
    "3. VALIDATE REFRESHED TRAJECTORY CLASSES"
)

if len(trajectory) != EXPECTED_GRIDS:

    raise RuntimeError(
        f"Expected {EXPECTED_GRIDS} grids, "
        f"found {len(trajectory)}."
    )


if (
    trajectory[
        "trajectory_class"
    ]
    .isna()
    .any()
):

    raise RuntimeError(
        "Null trajectory classes detected."
    )


actual_classes = set(
    trajectory[
        "trajectory_class"
    ]
    .unique()
)

if not actual_classes.issubset(
    EXPECTED_CLASSES
):

    raise RuntimeError(
        f"Unexpected trajectory classes: "
        f"{actual_classes}"
    )


class_counts = (
    trajectory[
        "trajectory_class"
    ]
    .value_counts()
    .rename_axis(
        "trajectory_class"
    )
    .reset_index(
        name="grid_count"
    )
)

class_counts[
    "percent_of_grids"
] = (
    class_counts[
        "grid_count"
    ]
    /
    EXPECTED_GRIDS
    *
    100.0
)

print(
    class_counts.to_string(
        index=False
    )
)


trajectory.to_csv(
    TRAJECTORY_CSV,
    index=False,
)

class_counts.to_csv(
    COUNT_CSV,
    index=False,
)


# ============================================================
# 4. REPRODUCTION CHECK FOR CURRENT 2015-2025 DATA
# ============================================================

section(
    "4. CURRENT-BASELINE REPRODUCTION CHECK"
)

reproduction_status = (
    "NOT_APPLICABLE"
)

agreement_percent = None


if (
    years[0] == 2015
    and latest_year == 2025
    and REFERENCE_TRAJECTORY_PATH.exists()
):

    reference = pd.read_csv(
        REFERENCE_TRAJECTORY_PATH,
        dtype={
            "grid_id": str,
        },
    )

    comparison = (
        trajectory[
            [
                "grid_id",
                "trajectory_class",
            ]
        ]
        .merge(
            reference[
                [
                    "grid_id",
                    "trajectory_class",
                ]
            ],
            on="grid_id",
            suffixes=(
                "_dynamic",
                "_reference",
            ),
            how="inner",
        )
    )


    agreement_percent = float(
        (
            comparison[
                "trajectory_class_dynamic"
            ]
            ==
            comparison[
                "trajectory_class_reference"
            ]
        )
        .mean()
        *
        100.0
    )


    print(
        f"Reference agreement : "
        f"{agreement_percent:.2f}%"
    )


    if agreement_percent < 100.0:

        raise RuntimeError(
            "Dynamic trajectory implementation "
            "does not exactly reproduce the "
            "validated 2015-2025 Step-23B result."
        )


    reproduction_status = (
        "PASS"
    )

else:

    print(
        "Reference reproduction check skipped "
        "because uploaded history extends beyond "
        "the original 2015-2025 research period."
    )


# ============================================================
# 5. EXPORT MAP-READY GEOJSON
# ============================================================

section(
    "5. EXPORT DYNAMIC TRAJECTORY GEOJSON"
)

geometry = gpd.read_file(
    GEOMETRY_PATH
)

geometry[
    "grid_id"
] = (
    geometry[
        "grid_id"
    ]
    .astype(str)
)


map_gdf = (
    geometry
    .merge(
        trajectory,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )
)


if (
    map_gdf[
        "trajectory_class"
    ]
    .isna()
    .any()
):

    raise RuntimeError(
        "Trajectory / geometry alignment failed."
    )


map_gdf.to_file(
    TRAJECTORY_GEOJSON,
    driver="GeoJSON",
)


print(
    f"Map rows : "
    f"{len(map_gdf):,}"
)

print(
    f"Map CRS  : "
    f"{map_gdf.crs}"
)

print(
    f"Saved    : "
    f"{TRAJECTORY_GEOJSON}"
)


# ============================================================
# 6. AUDIT / METHODOLOGY
# ============================================================

section(
    "6. SAVE AUDIT"
)

audit_rows = [
    {
        "check":
            "validated_history_loaded",

        "status":
            "PASS",

        "details":
            str(INPUT_PATH),
    },

    {
        "check":
            "all_observed_years_used_for_relative_direction",

        "status":
            "PASS",

        "details":
            f"{years[0]}-{years[-1]}",
    },

    {
        "check":
            "trajectory_grid_count",

        "status":
            (
                "PASS"
                if len(trajectory)
                ==
                EXPECTED_GRIDS
                else "FAIL"
            ),

        "details":
            (
                f"actual={len(trajectory)}, "
                f"expected={EXPECTED_GRIDS}"
            ),
    },

    {
        "check":
            "exactly_one_class_per_grid",

        "status":
            "PASS",

        "details":
            "Every grid has one trajectory class.",
    },

    {
        "check":
            "reference_thresholds_preserved",

        "status":
            "PASS",

        "details":
            (
                "hot=0.75, stable_split=0.50, "
                "relative_slope=0.010"
            ),
    },

    {
        "check":
            "trajectory_scope",

        "status":
            "PASS",

        "details":
            (
                "Relative spatial heat trajectory; "
                "not absolute warming significance."
            ),
    },

    {
        "check":
            "2015_2025_reproduction",

        "status":
            reproduction_status,

        "details":
            (
                (
                    f"agreement={agreement_percent:.2f}%"
                )
                if agreement_percent is not None
                else
                "Not applicable to updated history."
            ),
    },

    {
        "check":
            "map_alignment",

        "status":
            (
                "PASS"
                if len(map_gdf)
                ==
                EXPECTED_GRIDS
                else "FAIL"
            ),

        "details":
            f"rows={len(map_gdf)}",
    },
]


pd.DataFrame(
    audit_rows
).to_csv(
    AUDIT_PATH,
    index=False,
)


methodology = {
    "analysis":
        "Dynamic refreshed relative spatial heat trajectory",

    "observed_period":
        f"{years[0]}-{years[-1]}",

    "latest_observed_year":
        latest_year,

    "relative_heat_definition":
        (
            "Within-year percentile rank of observed "
            "LST_C among all Kaduwela grids."
        ),

    "direction_definition":
        (
            "Theil-Sen slope of yearly relative "
            "heat percentile across all observed years."
        ),

    "baseline_window":
        baseline_years,

    "recent_window":
        recent_years,

    "status_statistic":
        "median",

    "hot_threshold":
        HOT_THRESHOLD,

    "stable_split":
        STABLE_SPLIT,

    "relative_sen_slope_threshold":
        SLOPE_THRESHOLD,

    "trajectory_classes":
        sorted(
            EXPECTED_CLASSES
        ),

    "forecast_model_used":
        False,

    "trajectory_is_absolute_warming_claim":
        False,

    "trajectory_is_statistical_significance_claim":
        False,
}


with open(
    METHODOLOGY_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        methodology,
        file,
        indent=2,
    )


print(
    f"Trajectory CSV : "
    f"{TRAJECTORY_CSV}"
)

print(
    f"Counts CSV     : "
    f"{COUNT_CSV}"
)

print(
    f"GeoJSON        : "
    f"{TRAJECTORY_GEOJSON}"
)

print(
    f"Audit          : "
    f"{AUDIT_PATH}"
)


section(
    "STEP 02 COMPLETED SUCCESSFULLY"
)

print(
    f"Trajectory refreshed through {latest_year}."
)

print(
    "NEXT: controlled model update + "
    "dynamic next-five-year forecasting."
)