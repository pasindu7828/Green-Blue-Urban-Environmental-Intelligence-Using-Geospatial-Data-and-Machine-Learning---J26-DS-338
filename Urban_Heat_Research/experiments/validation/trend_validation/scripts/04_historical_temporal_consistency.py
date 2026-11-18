from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes, spearmanr


# ============================================================
# SETTINGS
# ============================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET = "LST_C"

EXPECTED_GRIDS = 2194

RELATIVE_SLOPE_THRESHOLD = 0.010

CUTOFFS = [
    2020,
    2021,
    2022,
]

FUTURE_WINDOW = 3
RECENT_WINDOW = 3


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
    / "04_historical_temporal_consistency_detail.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "04_historical_temporal_consistency_by_cutoff.csv"
)

OVERALL_PATH = (
    OUTPUT_DIR
    / "04_historical_temporal_consistency_summary.csv"
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


def relative_direction(slope):
    if slope >= RELATIVE_SLOPE_THRESHOLD:
        return "Increasing"

    if slope <= -RELATIVE_SLOPE_THRESHOLD:
        return "Decreasing"

    return "Stable"


def absolute_direction(slope):
    if slope > 0:
        return "Increasing"

    if slope < 0:
        return "Decreasing"

    return "Flat"


def observed_direction(change):
    if change > 0:
        return "Increasing"

    if change < 0:
        return "Decreasing"

    return "No Change"


# ============================================================
# 1. LOAD DATA
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

print(
    f"Years: {data[YEAR_COL].min()}-"
    f"{data[YEAR_COL].max()}"
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
        "Duplicate grid-year records found"
    )

if data[TARGET].isna().any():
    raise RuntimeError(
        "Missing LST values found"
    )

print(
    "PASS - historical dataset structure is valid."
)


# ============================================================
# 3. CALCULATE YEARLY RELATIVE HEAT
# ============================================================

section(
    "3. CALCULATE YEARLY RELATIVE HEAT PERCENTILES"
)

data = add_relative_percentile(
    data
)

if not data[
    "relative_heat_percentile"
].between(
    0,
    1,
).all():
    raise RuntimeError(
        "Invalid relative percentile"
    )

print(
    "PASS - yearly relative heat percentiles calculated."
)


# ============================================================
# 4. RETROSPECTIVE TEMPORAL TESTS
# ============================================================

section(
    "4. RUN RETROSPECTIVE TEMPORAL CONSISTENCY TESTS"
)

detail_frames = []
summary_rows = []

for cutoff in CUTOFFS:

    future_years = list(
        range(
            cutoff + 1,
            cutoff + FUTURE_WINDOW + 1,
        )
    )

    recent_years = list(
        range(
            cutoff - RECENT_WINDOW + 1,
            cutoff + 1,
        )
    )

    print(
        f"\n--- History <= {cutoff} | "
        f"Future actual {future_years} ---"
    )

    rows = []

    for grid_id, group in data.groupby(
        GRID_COL,
        sort=True,
    ):

        g = group.sort_values(
            YEAR_COL
        )

        history = g[
            g[YEAR_COL] <= cutoff
        ]

        recent = g[
            g[YEAR_COL].isin(
                recent_years
            )
        ]

        future = g[
            g[YEAR_COL].isin(
                future_years
            )
        ]

        if history.empty:
            raise RuntimeError(
                f"No history for {grid_id}"
            )

        if len(recent) != RECENT_WINDOW:
            raise RuntimeError(
                f"Recent window incomplete for {grid_id}"
            )

        if len(future) != FUTURE_WINDOW:
            raise RuntimeError(
                f"Future window incomplete for {grid_id}"
            )

        # --------------------------------------------
        # Historical absolute LST Sen slope
        # --------------------------------------------

        absolute_sen = theilslopes(
            history[TARGET].to_numpy(
                dtype=float
            ),
            history[YEAR_COL].to_numpy(
                dtype=float
            ),
            alpha=0.95,
        )

        absolute_slope = float(
            absolute_sen.slope
        )

        # --------------------------------------------
        # Historical relative heat Sen slope
        # --------------------------------------------

        relative_sen = theilslopes(
            history[
                "relative_heat_percentile"
            ].to_numpy(
                dtype=float
            ),
            history[YEAR_COL].to_numpy(
                dtype=float
            ),
            alpha=0.95,
        )

        relative_slope = float(
            relative_sen.slope
        )

        # --------------------------------------------
        # Actual subsequent change
        # --------------------------------------------

        recent_lst = float(
            recent[TARGET].median()
        )

        future_lst = float(
            future[TARGET].median()
        )

        absolute_future_change = (
            future_lst
            - recent_lst
        )

        recent_relative = float(
            recent[
                "relative_heat_percentile"
            ].median()
        )

        future_relative = float(
            future[
                "relative_heat_percentile"
            ].median()
        )

        relative_future_change = (
            future_relative
            - recent_relative
        )

        predicted_absolute_direction = (
            absolute_direction(
                absolute_slope
            )
        )

        observed_absolute_direction = (
            observed_direction(
                absolute_future_change
            )
        )

        predicted_relative_direction = (
            relative_direction(
                relative_slope
            )
        )

        observed_relative_direction = (
            observed_direction(
                relative_future_change
            )
        )

        rows.append(
            {
                "cutoff_year":
                    cutoff,

                GRID_COL:
                    str(grid_id),

                "history_year_count":
                    len(history),

                "recent_window":
                    "-".join(
                        map(
                            str,
                            recent_years,
                        )
                    ),

                "future_window":
                    "-".join(
                        map(
                            str,
                            future_years,
                        )
                    ),

                "historical_absolute_sen_slope_c_per_year":
                    absolute_slope,

                "historical_relative_sen_slope_per_year":
                    relative_slope,

                "predicted_absolute_direction":
                    predicted_absolute_direction,

                "predicted_relative_direction":
                    predicted_relative_direction,

                "recent_median_lst_c":
                    recent_lst,

                "future_median_lst_c":
                    future_lst,

                "actual_future_lst_change_c":
                    absolute_future_change,

                "recent_relative_heat_median":
                    recent_relative,

                "future_relative_heat_median":
                    future_relative,

                "actual_future_relative_change":
                    relative_future_change,

                "observed_absolute_direction":
                    observed_absolute_direction,

                "observed_relative_direction":
                    observed_relative_direction,
            }
        )

    result = pd.DataFrame(
        rows
    )

    # ========================================================
    # DIRECTION AGREEMENT
    # ========================================================

    absolute_testable = result[
        result[
            "predicted_absolute_direction"
        ].isin(
            [
                "Increasing",
                "Decreasing",
            ]
        )
    ].copy()

    absolute_testable[
        "absolute_direction_agreement"
    ] = (
        absolute_testable[
            "predicted_absolute_direction"
        ]
        ==
        absolute_testable[
            "observed_absolute_direction"
        ]
    )

    relative_testable = result[
        result[
            "predicted_relative_direction"
        ].isin(
            [
                "Increasing",
                "Decreasing",
            ]
        )
    ].copy()

    relative_testable[
        "relative_direction_agreement"
    ] = (
        relative_testable[
            "predicted_relative_direction"
        ]
        ==
        relative_testable[
            "observed_relative_direction"
        ]
    )

    result[
        "absolute_direction_agreement"
    ] = np.nan

    result.loc[
        absolute_testable.index,
        "absolute_direction_agreement",
    ] = absolute_testable[
        "absolute_direction_agreement"
    ].astype(float)

    result[
        "relative_direction_agreement"
    ] = np.nan

    result.loc[
        relative_testable.index,
        "relative_direction_agreement",
    ] = relative_testable[
        "relative_direction_agreement"
    ].astype(float)

    # ========================================================
    # ASSOCIATION BETWEEN HISTORICAL SLOPE AND FUTURE CHANGE
    # ========================================================

    absolute_rho = float(
        spearmanr(
            result[
                "historical_absolute_sen_slope_c_per_year"
            ],
            result[
                "actual_future_lst_change_c"
            ],
        ).statistic
    )

    relative_rho = float(
        spearmanr(
            result[
                "historical_relative_sen_slope_per_year"
            ],
            result[
                "actual_future_relative_change"
            ],
        ).statistic
    )

    # ========================================================
    # GROUP SUMMARIES
    # ========================================================

    relative_group_means = (
        result.groupby(
            "predicted_relative_direction"
        )[
            "actual_future_relative_change"
        ]
        .mean()
        .to_dict()
    )

    absolute_group_means = (
        result.groupby(
            "predicted_absolute_direction"
        )[
            "actual_future_lst_change_c"
        ]
        .mean()
        .to_dict()
    )

    summary_row = {
        "cutoff_year":
            cutoff,

        "history_year_count":
            cutoff - 2015 + 1,

        "future_years":
            "-".join(
                map(
                    str,
                    future_years,
                )
            ),

        "absolute_slope_vs_future_change_spearman":
            absolute_rho,

        "relative_slope_vs_future_change_spearman":
            relative_rho,

        "absolute_direction_testable_grids":
            len(
                absolute_testable
            ),

        "absolute_direction_agreement_percent":
            float(
                absolute_testable[
                    "absolute_direction_agreement"
                ].mean()
                * 100.0
            ),

        "relative_direction_testable_grids":
            len(
                relative_testable
            ),

        "relative_direction_agreement_percent":
            float(
                relative_testable[
                    "relative_direction_agreement"
                ].mean()
                * 100.0
            ),

        "relative_increasing_group_mean_future_change":
            float(
                relative_group_means.get(
                    "Increasing",
                    np.nan,
                )
            ),

        "relative_stable_group_mean_future_change":
            float(
                relative_group_means.get(
                    "Stable",
                    np.nan,
                )
            ),

        "relative_decreasing_group_mean_future_change":
            float(
                relative_group_means.get(
                    "Decreasing",
                    np.nan,
                )
            ),

        "absolute_increasing_group_mean_future_change_c":
            float(
                absolute_group_means.get(
                    "Increasing",
                    np.nan,
                )
            ),

        "absolute_decreasing_group_mean_future_change_c":
            float(
                absolute_group_means.get(
                    "Decreasing",
                    np.nan,
                )
            ),
    }

    summary_rows.append(
        summary_row
    )

    detail_frames.append(
        result
    )

    print(
        f"Absolute slope vs future change rho : "
        f"{absolute_rho:.4f}"
    )

    print(
        f"Absolute direction agreement        : "
        f"{summary_row['absolute_direction_agreement_percent']:.2f}%"
    )

    print(
        f"Relative slope vs future change rho : "
        f"{relative_rho:.4f}"
    )

    print(
        f"Relative direction agreement        : "
        f"{summary_row['relative_direction_agreement_percent']:.2f}%"
    )

    print(
        "Relative future change by historical direction:"
    )

    print(
        f"  Increasing: "
        f"{summary_row['relative_increasing_group_mean_future_change']:+.4f}"
    )

    print(
        f"  Stable    : "
        f"{summary_row['relative_stable_group_mean_future_change']:+.4f}"
    )

    print(
        f"  Decreasing: "
        f"{summary_row['relative_decreasing_group_mean_future_change']:+.4f}"
    )


# ============================================================
# 5. SAVE
# ============================================================

section(
    "5. SAVE TEMPORAL CONSISTENCY RESULTS"
)

detail = pd.concat(
    detail_frames,
    ignore_index=True,
)

summary = pd.DataFrame(
    summary_rows
)

expected_rows = (
    EXPECTED_GRIDS
    * len(CUTOFFS)
)

if len(detail) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} detail rows, "
        f"got {len(detail)}"
    )

detail.to_csv(
    DETAIL_PATH,
    index=False,
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 6. OVERALL SUMMARY
# ============================================================

section(
    "6. OVERALL HISTORICAL TEMPORAL CONSISTENCY"
)

overall = pd.DataFrame(
    [
        {
            "retrospective_cutoffs":
                len(summary),

            "total_grid_cutoff_tests":
                len(detail),

            "mean_absolute_slope_vs_future_change_spearman":
                float(
                    summary[
                        "absolute_slope_vs_future_change_spearman"
                    ].mean()
                ),

            "mean_absolute_direction_agreement_percent":
                float(
                    summary[
                        "absolute_direction_agreement_percent"
                    ].mean()
                ),

            "mean_relative_slope_vs_future_change_spearman":
                float(
                    summary[
                        "relative_slope_vs_future_change_spearman"
                    ].mean()
                ),

            "mean_relative_direction_agreement_percent":
                float(
                    summary[
                        "relative_direction_agreement_percent"
                    ].mean()
                ),

            "minimum_relative_direction_agreement_percent":
                float(
                    summary[
                        "relative_direction_agreement_percent"
                    ].min()
                ),

            "mean_increasing_group_future_relative_change":
                float(
                    summary[
                        "relative_increasing_group_mean_future_change"
                    ].mean()
                ),

            "mean_stable_group_future_relative_change":
                float(
                    summary[
                        "relative_stable_group_mean_future_change"
                    ].mean()
                ),

            "mean_decreasing_group_future_relative_change":
                float(
                    summary[
                        "relative_decreasing_group_mean_future_change"
                    ].mean()
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
    "7. STEP 4 VALIDATION STATUS"
)

if (
    len(summary) == len(CUTOFFS)
    and
    len(detail) == expected_rows
):
    print(
        "PASS - All retrospective historical temporal "
        "consistency tests completed successfully."
    )

else:
    raise RuntimeError(
        "Historical temporal consistency validation incomplete."
    )


print(
    "\nIMPORTANT INTERPRETATION:"
)

print(
    "- This is a retrospective consistency test, "
    "not a forecasting-accuracy test."
)

print(
    "- Historical Sen slope is calculated using only "
    "information available up to each cutoff year."
)

print(
    "- Subsequent 3-year observations are held out "
    "and used only for evaluation."
)

print(
    "- Relative trajectory direction describes change "
    "in spatial heat position within Kaduwela."
)

print(
    "- Absolute and relative trend evidence must remain "
    "separate in interpretation."
)

print(
    "- No arbitrary pass threshold is imposed."
)

print(
    f"\nSaved: {DETAIL_PATH}"
)

print(
    f"Saved: {SUMMARY_PATH}"
)

print(
    f"Saved: {OVERALL_PATH}"
)

print(
    "\nSTEP 4 HISTORICAL TEMPORAL CONSISTENCY COMPLETE."
)