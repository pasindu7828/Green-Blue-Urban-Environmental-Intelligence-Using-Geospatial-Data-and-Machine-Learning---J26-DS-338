from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes, mannwhitneyu


# ============================================================
# SETTINGS
# ============================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET = "LST_C"

EXPECTED_GRIDS = 2194

BASELINE_YEARS = [2015, 2016, 2017, 2018, 2019]

HOT_THRESHOLD = 0.75
STABLE_SPLIT = 0.50
RELATIVE_SLOPE_THRESHOLD = 0.010

CUTOFF_TARGETS = {
    2023: 2024,
    2024: 2025,
}

HOT_SIDE_CLASSES = {
    "Persistent Hot",
    "Intensifying",
    "Emerging",
}

COOL_SIDE_CLASSES = {
    "Persistent Cool",
    "Diminishing",
}

CLASS_ORDER = [
    "Persistent Cool",
    "Diminishing",
    "Emerging",
    "Intensifying",
    "Persistent Hot",
]


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[4]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "kaduwela_preprocessed_base.csv"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "validation"
    / "trend_validation"
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETAIL_PATH = (
    OUTPUT_DIR
    / "04b_historical_trajectory_relevance_detail.csv"
)

CLASS_SUMMARY_PATH = (
    OUTPUT_DIR
    / "04b_historical_trajectory_relevance_by_class.csv"
)

CUTOFF_SUMMARY_PATH = (
    OUTPUT_DIR
    / "04b_historical_trajectory_relevance_by_cutoff.csv"
)

OVERALL_PATH = (
    OUTPUT_DIR
    / "04b_historical_trajectory_relevance_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def add_relative_percentile(df):
    out = df.copy()

    out["relative_heat_percentile"] = (
        out.groupby(YEAR_COL)[TARGET]
        .rank(
            method="average",
            pct=True,
        )
    )

    return out


def classify_trajectory(
    baseline_score,
    recent_score,
    long_score,
    relative_slope,
):

    if (
        relative_slope >= RELATIVE_SLOPE_THRESHOLD
        and baseline_score < HOT_THRESHOLD
        and recent_score >= HOT_THRESHOLD
    ):
        return "Emerging"

    if relative_slope >= RELATIVE_SLOPE_THRESHOLD:
        return "Intensifying"

    if relative_slope <= -RELATIVE_SLOPE_THRESHOLD:
        return "Diminishing"

    if long_score >= STABLE_SPLIT:
        return "Persistent Hot"

    return "Persistent Cool"


def build_history_only_trajectory(
    data,
    cutoff,
):

    history = data[
        data[YEAR_COL] <= cutoff
    ].copy()

    recent_years = list(
        range(
            2021,
            cutoff + 1,
        )
    )

    rows = []

    for grid_id, group in history.groupby(
        GRID_COL,
        sort=True,
    ):

        g = group.sort_values(
            YEAR_COL
        )

        baseline = g[
            g[YEAR_COL].isin(
                BASELINE_YEARS
            )
        ]

        recent = g[
            g[YEAR_COL].isin(
                recent_years
            )
        ]

        if len(baseline) != 5:
            raise RuntimeError(
                f"Incomplete baseline for {grid_id}"
            )

        if recent.empty:
            raise RuntimeError(
                f"No recent history for {grid_id}"
            )

        sen = theilslopes(
            g[
                "relative_heat_percentile"
            ].to_numpy(dtype=float),
            g[
                YEAR_COL
            ].to_numpy(dtype=float),
            alpha=0.95,
        )

        relative_slope = float(
            sen.slope
        )

        baseline_score = float(
            baseline[
                "relative_heat_percentile"
            ].median()
        )

        recent_score = float(
            recent[
                "relative_heat_percentile"
            ].median()
        )

        long_score = float(
            g[
                "relative_heat_percentile"
            ].median()
        )

        trajectory_class = classify_trajectory(
            baseline_score,
            recent_score,
            long_score,
            relative_slope,
        )

        if trajectory_class in HOT_SIDE_CLASSES:
            side = "Hot-side"

        elif trajectory_class in COOL_SIDE_CLASSES:
            side = "Cool-side"

        else:
            raise RuntimeError(
                "Unknown trajectory class"
            )

        rows.append(
            {
                GRID_COL: str(grid_id),

                "cutoff_year":
                    cutoff,

                "history_year_count":
                    len(g),

                "baseline_relative_heat_median":
                    baseline_score,

                "recent_relative_heat_median":
                    recent_score,

                "long_term_relative_heat_median":
                    long_score,

                "historical_relative_sen_slope":
                    relative_slope,

                "historical_trajectory_class":
                    trajectory_class,

                "trajectory_side":
                    side,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 1. LOAD
# ============================================================

section(
    "1. LOAD HISTORICAL DATA"
)

data = pd.read_csv(
    DATA_PATH,
    dtype={GRID_COL: str},
)

data[YEAR_COL] = pd.to_numeric(
    data[YEAR_COL],
    errors="raise",
).astype(int)

data[TARGET] = pd.to_numeric(
    data[TARGET],
    errors="raise",
)

print(
    f"Rows : {len(data):,}"
)

print(
    f"Grids: {data[GRID_COL].nunique():,}"
)


# ============================================================
# 2. STRUCTURAL AUDIT
# ============================================================

section(
    "2. STRUCTURAL AUDIT"
)

if data[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        "Grid count mismatch"
    )

if data.duplicated(
    [GRID_COL, YEAR_COL]
).any():
    raise RuntimeError(
        "Duplicate grid-year records"
    )

print(
    "PASS - dataset structurally valid."
)


# ============================================================
# 3. YEARLY RELATIVE HEAT
# ============================================================

section(
    "3. CALCULATE YEARLY RELATIVE HEAT PERCENTILES"
)

data = add_relative_percentile(
    data
)

print(
    "PASS - yearly relative percentiles calculated."
)


# ============================================================
# 4. RETROSPECTIVE CLASSIFICATION + NEXT-YEAR EVALUATION
# ============================================================

section(
    "4. HISTORY-ONLY TRAJECTORY VS NEXT-YEAR ACTUAL HEAT"
)

detail_frames = []
class_summary_rows = []
cutoff_summary_rows = []

for cutoff, target_year in CUTOFF_TARGETS.items():

    print(
        f"\n--- History <= {cutoff} "
        f"→ evaluate actual {target_year} ---"
    )

    trajectory = (
        build_history_only_trajectory(
            data,
            cutoff,
        )
    )

    actual = data[
        data[YEAR_COL] == target_year
    ][
        [
            GRID_COL,
            TARGET,
            "relative_heat_percentile",
        ]
    ].copy()

    actual = actual.rename(
        columns={
            TARGET:
                "actual_next_year_lst_c",

            "relative_heat_percentile":
                "actual_next_year_relative_heat",
        }
    )

    result = trajectory.merge(
        actual,
        on=GRID_COL,
        how="inner",
        validate="one_to_one",
    )

    if len(result) != EXPECTED_GRIDS:
        raise RuntimeError(
            "Target-year merge grid mismatch"
        )

    result[
        "actual_next_year_top25_heat"
    ] = (
        result[
            "actual_next_year_relative_heat"
        ]
        >= 0.75
    )

    result[
        "actual_next_year_top10_heat"
    ] = (
        result[
            "actual_next_year_relative_heat"
        ]
        >= 0.90
    )

    # --------------------------------------------------------
    # CLASS-LEVEL RESULTS
    # --------------------------------------------------------

    for class_name in CLASS_ORDER:

        subset = result[
            result[
                "historical_trajectory_class"
            ]
            == class_name
        ]

        if subset.empty:
            continue

        class_summary_rows.append(
            {
                "cutoff_year":
                    cutoff,

                "target_year":
                    target_year,

                "trajectory_class":
                    class_name,

                "grid_count":
                    len(subset),

                "mean_next_year_relative_heat":
                    float(
                        subset[
                            "actual_next_year_relative_heat"
                        ].mean()
                    ),

                "median_next_year_relative_heat":
                    float(
                        subset[
                            "actual_next_year_relative_heat"
                        ].median()
                    ),

                "mean_next_year_lst_c":
                    float(
                        subset[
                            "actual_next_year_lst_c"
                        ].mean()
                    ),

                "top25_actual_heat_percent":
                    float(
                        subset[
                            "actual_next_year_top25_heat"
                        ].mean()
                        * 100.0
                    ),

                "top10_actual_heat_percent":
                    float(
                        subset[
                            "actual_next_year_top10_heat"
                        ].mean()
                        * 100.0
                    ),
            }
        )

    # --------------------------------------------------------
    # HOT-SIDE VS COOL-SIDE
    # --------------------------------------------------------

    hot = result[
        result[
            "trajectory_side"
        ]
        == "Hot-side"
    ]

    cool = result[
        result[
            "trajectory_side"
        ]
        == "Cool-side"
    ]

    if hot.empty or cool.empty:
        raise RuntimeError(
            "Hot-side/cool-side group missing"
        )

    u_test = mannwhitneyu(
        hot[
            "actual_next_year_relative_heat"
        ],
        cool[
            "actual_next_year_relative_heat"
        ],
        alternative="greater",
    )

    hot_top25_rate = float(
        hot[
            "actual_next_year_top25_heat"
        ].mean()
    )

    cool_top25_rate = float(
        cool[
            "actual_next_year_top25_heat"
        ].mean()
    )

    hot_top10_rate = float(
        hot[
            "actual_next_year_top10_heat"
        ].mean()
    )

    cool_top10_rate = float(
        cool[
            "actual_next_year_top10_heat"
        ].mean()
    )

    top25_risk_ratio = (
        hot_top25_rate
        / cool_top25_rate
        if cool_top25_rate > 0
        else np.nan
    )

    top10_risk_ratio = (
        hot_top10_rate
        / cool_top10_rate
        if cool_top10_rate > 0
        else np.nan
    )

    row = {
        "cutoff_year":
            cutoff,

        "target_year":
            target_year,

        "hot_side_grid_count":
            len(hot),

        "cool_side_grid_count":
            len(cool),

        "hot_side_mean_next_year_relative_heat":
            float(
                hot[
                    "actual_next_year_relative_heat"
                ].mean()
            ),

        "cool_side_mean_next_year_relative_heat":
            float(
                cool[
                    "actual_next_year_relative_heat"
                ].mean()
            ),

        "hot_minus_cool_mean_relative_heat":
            float(
                hot[
                    "actual_next_year_relative_heat"
                ].mean()
                -
                cool[
                    "actual_next_year_relative_heat"
                ].mean()
            ),

        "hot_side_mean_next_year_lst_c":
            float(
                hot[
                    "actual_next_year_lst_c"
                ].mean()
            ),

        "cool_side_mean_next_year_lst_c":
            float(
                cool[
                    "actual_next_year_lst_c"
                ].mean()
            ),

        "hot_minus_cool_mean_lst_c":
            float(
                hot[
                    "actual_next_year_lst_c"
                ].mean()
                -
                cool[
                    "actual_next_year_lst_c"
                ].mean()
            ),

        "hot_side_top25_actual_heat_percent":
            hot_top25_rate * 100.0,

        "cool_side_top25_actual_heat_percent":
            cool_top25_rate * 100.0,

        "top25_heat_risk_ratio_hot_vs_cool":
            top25_risk_ratio,

        "hot_side_top10_actual_heat_percent":
            hot_top10_rate * 100.0,

        "cool_side_top10_actual_heat_percent":
            cool_top10_rate * 100.0,

        "top10_heat_risk_ratio_hot_vs_cool":
            top10_risk_ratio,

        "mann_whitney_u":
            float(
                u_test.statistic
            ),

        "mann_whitney_one_sided_p":
            float(
                u_test.pvalue
            ),
    }

    cutoff_summary_rows.append(
        row
    )

    detail_frames.append(
        result
    )

    print(
        f"Hot-side mean relative heat : "
        f"{row['hot_side_mean_next_year_relative_heat']:.4f}"
    )

    print(
        f"Cool-side mean relative heat: "
        f"{row['cool_side_mean_next_year_relative_heat']:.4f}"
    )

    print(
        f"Difference                 : "
        f"{row['hot_minus_cool_mean_relative_heat']:+.4f}"
    )

    print(
        f"Hot-side mean LST          : "
        f"{row['hot_side_mean_next_year_lst_c']:.3f} C"
    )

    print(
        f"Cool-side mean LST         : "
        f"{row['cool_side_mean_next_year_lst_c']:.3f} C"
    )

    print(
        f"LST difference             : "
        f"{row['hot_minus_cool_mean_lst_c']:+.3f} C"
    )

    print(
        f"Top-25% heat risk ratio    : "
        f"{row['top25_heat_risk_ratio_hot_vs_cool']:.3f}"
    )

    print(
        f"Mann-Whitney p             : "
        f"{row['mann_whitney_one_sided_p']:.6g}"
    )


# ============================================================
# 5. SAVE
# ============================================================

section(
    "5. SAVE RETROSPECTIVE RELEVANCE RESULTS"
)

detail = pd.concat(
    detail_frames,
    ignore_index=True,
)

class_summary = pd.DataFrame(
    class_summary_rows
)

cutoff_summary = pd.DataFrame(
    cutoff_summary_rows
)

detail.to_csv(
    DETAIL_PATH,
    index=False,
)

class_summary.to_csv(
    CLASS_SUMMARY_PATH,
    index=False,
)

cutoff_summary.to_csv(
    CUTOFF_SUMMARY_PATH,
    index=False,
)

print(
    "\nClass-level summary:"
)

print(
    class_summary.to_string(
        index=False
    )
)

print(
    "\nCutoff-level summary:"
)

print(
    cutoff_summary.to_string(
        index=False
    )
)


# ============================================================
# 6. OVERALL SUMMARY
# ============================================================

section(
    "6. OVERALL HISTORICAL TRAJECTORY RELEVANCE"
)

overall = pd.DataFrame(
    [
        {
            "retrospective_tests":
                len(
                    cutoff_summary
                ),

            "total_grid_tests":
                len(
                    detail
                ),

            "mean_hot_side_next_year_relative_heat":
                float(
                    cutoff_summary[
                        "hot_side_mean_next_year_relative_heat"
                    ].mean()
                ),

            "mean_cool_side_next_year_relative_heat":
                float(
                    cutoff_summary[
                        "cool_side_mean_next_year_relative_heat"
                    ].mean()
                ),

            "mean_hot_minus_cool_relative_heat_difference":
                float(
                    cutoff_summary[
                        "hot_minus_cool_mean_relative_heat"
                    ].mean()
                ),

            "mean_hot_minus_cool_lst_difference_c":
                float(
                    cutoff_summary[
                        "hot_minus_cool_mean_lst_c"
                    ].mean()
                ),

            "mean_top25_heat_risk_ratio_hot_vs_cool":
                float(
                    cutoff_summary[
                        "top25_heat_risk_ratio_hot_vs_cool"
                    ].mean()
                ),

            "mean_top10_heat_risk_ratio_hot_vs_cool":
                float(
                    cutoff_summary[
                        "top10_heat_risk_ratio_hot_vs_cool"
                    ].mean()
                ),

            "maximum_mann_whitney_one_sided_p":
                float(
                    cutoff_summary[
                        "mann_whitney_one_sided_p"
                    ].max()
                ),
        }
    ]
)

overall.to_csv(
    OVERALL_PATH,
    index=False,
)

print(
    overall.to_string(
        index=False
    )
)


# ============================================================
# 7. STATUS
# ============================================================

section(
    "7. STEP 4B VALIDATION STATUS"
)

expected_detail_rows = (
    EXPECTED_GRIDS
    * len(
        CUTOFF_TARGETS
    )
)

if (
    len(detail) == expected_detail_rows
    and
    len(cutoff_summary)
    == len(CUTOFF_TARGETS)
):
    print(
        "PASS - Historical trajectory relevance "
        "analysis completed successfully."
    )

else:
    raise RuntimeError(
        "Historical trajectory relevance analysis incomplete."
    )

print(
    "\nIMPORTANT:"
)

print(
    "- This test evaluates whether history-only trajectory "
    "classes identify subsequent relative heat burden."
)

print(
    "- It does NOT test whether the historical slope continues "
    "in the same future direction."
)

print(
    "- Target-year observations are used only for evaluation."
)

print(
    "- Trajectory remains a relative spatial-behavior "
    "classification, not a future-temperature forecast."
)

print(
    "\nSTEP 4B HISTORICAL TRAJECTORY RELEVANCE COMPLETE."
)