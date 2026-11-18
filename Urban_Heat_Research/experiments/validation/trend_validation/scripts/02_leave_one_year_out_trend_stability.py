from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


# ============================================================
# SETTINGS
# ============================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET = "LST_C"

YEARS = list(range(2015, 2026))

EXPECTED_GRIDS = 2194
EXPECTED_FULL_YEARS = 11
EXPECTED_LOO_YEARS = 10

ALPHA = 0.05
FDR_ALPHA = 0.05


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

PRODUCTION_TREND_PATH = (
    ROOT
    / "outputs"
    / "metrics"
    / "23_grid_mann_kendall_sens_slope.csv"
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
    / "02_leave_one_year_out_trend_stability_detail.csv"
)

YEAR_SUMMARY_PATH = (
    OUTPUT_DIR
    / "02_leave_one_year_out_trend_stability_by_year.csv"
)

OVERALL_SUMMARY_PATH = (
    OUTPUT_DIR
    / "02_leave_one_year_out_trend_stability_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def mann_kendall(values):
    y = np.asarray(values, dtype=float)

    n = len(y)

    s = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            if y[j] > y[i]:
                s += 1
            elif y[j] < y[i]:
                s -= 1

    _, counts = np.unique(
        y,
        return_counts=True,
    )

    tie_term = sum(
        c * (c - 1) * (2 * c + 5)
        for c in counts
        if c > 1
    )

    var_s = (
        n * (n - 1) * (2 * n + 5)
        - tie_term
    ) / 18.0

    if var_s <= 0:
        z = 0.0

    elif s > 0:
        z = (
            s - 1
        ) / math.sqrt(var_s)

    elif s < 0:
        z = (
            s + 1
        ) / math.sqrt(var_s)

    else:
        z = 0.0

    p = float(
        2.0 * norm.sf(abs(z))
    )

    return {
        "mk_s": int(s),
        "mk_p_value": p,
    }


def sen_slope(years, values):
    years = np.asarray(
        years,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    slopes = []

    for i in range(len(years) - 1):
        for j in range(i + 1, len(years)):
            slopes.append(
                (
                    values[j] - values[i]
                )
                /
                (
                    years[j] - years[i]
                )
            )

    return float(
        np.median(slopes)
    )


def benjamini_hochberg(p_values):
    p = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(p)

    order = np.argsort(p)

    ranked = p[order]

    adjusted = (
        ranked
        * m
        / np.arange(
            1,
            m + 1,
            dtype=float,
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    result = np.empty(
        m,
        dtype=float,
    )

    result[order] = adjusted

    return result


def trend_class(
    significance_value,
    slope,
    alpha=0.05,
):
    if significance_value < alpha:

        if slope > 0:
            return "Significant Increasing"

        if slope < 0:
            return "Significant Decreasing"

    return "No Significant Trend"


def slope_direction(slope):
    if slope > 0:
        return "Increasing"

    if slope < 0:
        return "Decreasing"

    return "Flat"


def calculate_trends(data):
    rows = []

    for grid_id, group in data.groupby(
        GRID_COL,
        sort=True,
    ):

        g = group.sort_values(
            YEAR_COL
        )

        years = g[
            YEAR_COL
        ].to_numpy(
            dtype=float
        )

        values = g[
            TARGET
        ].to_numpy(
            dtype=float
        )

        mk = mann_kendall(
            values
        )

        slope = sen_slope(
            years,
            values,
        )

        rows.append(
            {
                GRID_COL: str(grid_id),
                "loo_n_years": int(len(g)),
                "loo_mk_s": mk["mk_s"],
                "loo_mk_p_value": (
                    mk["mk_p_value"]
                ),
                "loo_sen_slope_c_per_year": (
                    slope
                ),
            }
        )

    result = pd.DataFrame(rows)

    result[
        "loo_mk_fdr_q_value"
    ] = benjamini_hochberg(
        result[
            "loo_mk_p_value"
        ].to_numpy(dtype=float)
    )

    result[
        "loo_raw_trend_class"
    ] = [
        trend_class(
            p,
            slope,
            ALPHA,
        )
        for p, slope in zip(
            result[
                "loo_mk_p_value"
            ],
            result[
                "loo_sen_slope_c_per_year"
            ],
        )
    ]

    result[
        "loo_fdr_trend_class"
    ] = [
        trend_class(
            q,
            slope,
            FDR_ALPHA,
        )
        for q, slope in zip(
            result[
                "loo_mk_fdr_q_value"
            ],
            result[
                "loo_sen_slope_c_per_year"
            ],
        )
    ]

    result[
        "loo_slope_direction"
    ] = result[
        "loo_sen_slope_c_per_year"
    ].apply(
        slope_direction
    )

    return result


# ============================================================
# 1. LOAD
# ============================================================

section(
    "1. LOAD HISTORICAL DATA AND FROZEN FULL-PERIOD TREND RESULTS"
)

data = pd.read_csv(
    DATA_PATH,
    dtype={GRID_COL: str},
)

full = pd.read_csv(
    PRODUCTION_TREND_PATH,
    dtype={GRID_COL: str},
)

print(
    f"Historical rows : {len(data):,}"
)

print(
    f"Historical grids: {data[GRID_COL].nunique():,}"
)

print(
    f"Full trend rows : {len(full):,}"
)


# ============================================================
# 2. STRUCTURAL AUDIT
# ============================================================

section(
    "2. STRUCTURAL AUDIT"
)

if data[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        "Historical grid count mismatch"
    )

if full[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        "Production trend grid count mismatch"
    )

if sorted(
    data[YEAR_COL]
    .unique()
    .tolist()
) != YEARS:
    raise RuntimeError(
        "Historical year range mismatch"
    )

if data.duplicated(
    [GRID_COL, YEAR_COL]
).any():
    raise RuntimeError(
        "Duplicate grid-year records found"
    )

year_counts = (
    data.groupby(GRID_COL)[YEAR_COL]
    .nunique()
)

if (
    year_counts.min()
    != EXPECTED_FULL_YEARS
    or
    year_counts.max()
    != EXPECTED_FULL_YEARS
):
    raise RuntimeError(
        "Each grid must have exactly 11 years"
    )

print(
    "PASS - full historical dataset is structurally valid."
)


# ============================================================
# 3. PREPARE FULL REFERENCE
# ============================================================

section(
    "3. PREPARE FROZEN FULL-PERIOD REFERENCE"
)

reference = full[
    [
        GRID_COL,
        "sen_slope_c_per_year",
        "mk_p_value",
        "mk_fdr_q_value",
        "trend_raw_alpha_0_05",
        "trend_fdr_alpha_0_05",
    ]
].copy()

reference = reference.rename(
    columns={
        "sen_slope_c_per_year":
            "full_sen_slope_c_per_year",

        "mk_p_value":
            "full_mk_p_value",

        "mk_fdr_q_value":
            "full_mk_fdr_q_value",

        "trend_raw_alpha_0_05":
            "full_raw_trend_class",

        "trend_fdr_alpha_0_05":
            "full_fdr_trend_class",
    }
)

reference[
    "full_slope_direction"
] = reference[
    "full_sen_slope_c_per_year"
].apply(
    slope_direction
)

reference[
    "full_fdr_significant"
] = (
    reference[
        "full_mk_fdr_q_value"
    ]
    < FDR_ALPHA
)

print(
    "Frozen 2015-2025 trend results prepared."
)


# ============================================================
# 4. LEAVE-ONE-YEAR-OUT RUNS
# ============================================================

section(
    "4. RUN 11 LEAVE-ONE-YEAR-OUT TREND ANALYSES"
)

detail_frames = []
summary_rows = []

for omitted_year in YEARS:

    print(
        f"\n--- Removing {omitted_year} ---"
    )

    loo_data = data[
        data[YEAR_COL] != omitted_year
    ].copy()

    loo_counts = (
        loo_data.groupby(
            GRID_COL
        )[YEAR_COL]
        .nunique()
    )

    if (
        loo_counts.min()
        != EXPECTED_LOO_YEARS
        or
        loo_counts.max()
        != EXPECTED_LOO_YEARS
    ):
        raise RuntimeError(
            f"Invalid LOO year count after removing "
            f"{omitted_year}"
        )

    loo = calculate_trends(
        loo_data
    )

    if len(loo) != EXPECTED_GRIDS:
        raise RuntimeError(
            f"LOO {omitted_year}: "
            f"unexpected grid count"
        )

    compare = reference.merge(
        loo,
        on=GRID_COL,
        how="inner",
        validate="one_to_one",
    )

    compare.insert(
        0,
        "omitted_year",
        omitted_year,
    )

    # --------------------------------------------------------
    # Per-grid stability diagnostics
    # --------------------------------------------------------

    compare[
        "abs_sen_slope_change"
    ] = (
        compare[
            "loo_sen_slope_c_per_year"
        ]
        -
        compare[
            "full_sen_slope_c_per_year"
        ]
    ).abs()

    compare[
        "slope_direction_same"
    ] = (
        compare[
            "loo_slope_direction"
        ]
        ==
        compare[
            "full_slope_direction"
        ]
    )

    compare[
        "raw_trend_class_same"
    ] = (
        compare[
            "loo_raw_trend_class"
        ]
        ==
        compare[
            "full_raw_trend_class"
        ]
    )

    compare[
        "fdr_trend_class_same"
    ] = (
        compare[
            "loo_fdr_trend_class"
        ]
        ==
        compare[
            "full_fdr_trend_class"
        ]
    )

    compare[
        "loo_fdr_significant"
    ] = (
        compare[
            "loo_mk_fdr_q_value"
        ]
        < FDR_ALPHA
    )

    compare[
        "lost_fdr_significance"
    ] = (
        compare[
            "full_fdr_significant"
        ]
        &
        ~compare[
            "loo_fdr_significant"
        ]
    )

    compare[
        "gained_fdr_significance"
    ] = (
        ~compare[
            "full_fdr_significant"
        ]
        &
        compare[
            "loo_fdr_significant"
        ]
    )

    # --------------------------------------------------------
    # Summary for this omitted year
    # --------------------------------------------------------

    rho = float(
        spearmanr(
            compare[
                "full_sen_slope_c_per_year"
            ],
            compare[
                "loo_sen_slope_c_per_year"
            ],
        ).statistic
    )

    row = {
        "omitted_year": omitted_year,

        "grids": len(compare),

        "years_per_grid_after_removal":
            EXPECTED_LOO_YEARS,

        "sen_slope_spearman":
            rho,

        "median_abs_sen_slope_change":
            float(
                compare[
                    "abs_sen_slope_change"
                ].median()
            ),

        "mean_abs_sen_slope_change":
            float(
                compare[
                    "abs_sen_slope_change"
                ].mean()
            ),

        "p95_abs_sen_slope_change":
            float(
                compare[
                    "abs_sen_slope_change"
                ].quantile(0.95)
            ),

        "max_abs_sen_slope_change":
            float(
                compare[
                    "abs_sen_slope_change"
                ].max()
            ),

        "slope_direction_agreement_percent":
            float(
                compare[
                    "slope_direction_same"
                ].mean()
                * 100.0
            ),

        "raw_trend_class_agreement_percent":
            float(
                compare[
                    "raw_trend_class_same"
                ].mean()
                * 100.0
            ),

        "fdr_trend_class_agreement_percent":
            float(
                compare[
                    "fdr_trend_class_same"
                ].mean()
                * 100.0
            ),

        "full_fdr_significant_count":
            int(
                compare[
                    "full_fdr_significant"
                ].sum()
            ),

        "loo_fdr_significant_count":
            int(
                compare[
                    "loo_fdr_significant"
                ].sum()
            ),

        "lost_fdr_significance_count":
            int(
                compare[
                    "lost_fdr_significance"
                ].sum()
            ),

        "gained_fdr_significance_count":
            int(
                compare[
                    "gained_fdr_significance"
                ].sum()
            ),
    }

    summary_rows.append(row)

    detail_frames.append(
        compare
    )

    print(
        f"Sen slope Spearman       : "
        f"{rho:.6f}"
    )

    print(
        f"Slope direction agreement: "
        f"{row['slope_direction_agreement_percent']:.2f}%"
    )

    print(
        f"FDR class agreement      : "
        f"{row['fdr_trend_class_agreement_percent']:.2f}%"
    )

    print(
        f"FDR significant grids    : "
        f"{row['loo_fdr_significant_count']}"
    )


# ============================================================
# 5. SAVE DETAILED RESULTS
# ============================================================

section(
    "5. SAVE DETAILED LEAVE-ONE-YEAR-OUT RESULTS"
)

detail = pd.concat(
    detail_frames,
    ignore_index=True,
)

expected_detail_rows = (
    EXPECTED_GRIDS
    * len(YEARS)
)

if len(detail) != expected_detail_rows:
    raise RuntimeError(
        f"Expected {expected_detail_rows} detail rows, "
        f"got {len(detail)}"
    )

detail.to_csv(
    DETAIL_PATH,
    index=False,
)

print(
    f"Detailed rows: {len(detail):,}"
)

print(
    f"Saved: {DETAIL_PATH}"
)


# ============================================================
# 6. BY-YEAR SUMMARY
# ============================================================

section(
    "6. LEAVE-ONE-YEAR-OUT STABILITY SUMMARY"
)

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    YEAR_SUMMARY_PATH,
    index=False,
)

print(
    summary.to_string(
        index=False
    )
)

print(
    f"\nSaved: {YEAR_SUMMARY_PATH}"
)


# ============================================================
# 7. OVERALL SUMMARY
# ============================================================

section(
    "7. OVERALL TREND STABILITY SUMMARY"
)

worst_fdr_idx = (
    summary[
        "fdr_trend_class_agreement_percent"
    ]
    .idxmin()
)

worst_spearman_idx = (
    summary[
        "sen_slope_spearman"
    ]
    .idxmin()
)

overall = pd.DataFrame(
    [
        {
            "loo_runs":
                len(summary),

            "total_grid_run_comparisons":
                len(detail),

            "mean_sen_slope_spearman":
                float(
                    summary[
                        "sen_slope_spearman"
                    ].mean()
                ),

            "median_sen_slope_spearman":
                float(
                    summary[
                        "sen_slope_spearman"
                    ].median()
                ),

            "minimum_sen_slope_spearman":
                float(
                    summary[
                        "sen_slope_spearman"
                    ].min()
                ),

            "worst_spearman_omitted_year":
                int(
                    summary.loc[
                        worst_spearman_idx,
                        "omitted_year",
                    ]
                ),

            "mean_slope_direction_agreement_percent":
                float(
                    summary[
                        "slope_direction_agreement_percent"
                    ].mean()
                ),

            "minimum_slope_direction_agreement_percent":
                float(
                    summary[
                        "slope_direction_agreement_percent"
                    ].min()
                ),

            "mean_raw_class_agreement_percent":
                float(
                    summary[
                        "raw_trend_class_agreement_percent"
                    ].mean()
                ),

            "minimum_raw_class_agreement_percent":
                float(
                    summary[
                        "raw_trend_class_agreement_percent"
                    ].min()
                ),

            "mean_fdr_class_agreement_percent":
                float(
                    summary[
                        "fdr_trend_class_agreement_percent"
                    ].mean()
                ),

            "minimum_fdr_class_agreement_percent":
                float(
                    summary[
                        "fdr_trend_class_agreement_percent"
                    ].min()
                ),

            "worst_fdr_agreement_omitted_year":
                int(
                    summary.loc[
                        worst_fdr_idx,
                        "omitted_year",
                    ]
                ),

            "median_of_median_abs_slope_changes":
                float(
                    summary[
                        "median_abs_sen_slope_change"
                    ].median()
                ),

            "maximum_p95_abs_slope_change":
                float(
                    summary[
                        "p95_abs_sen_slope_change"
                    ].max()
                ),

            "mean_lost_fdr_significance_count":
                float(
                    summary[
                        "lost_fdr_significance_count"
                    ].mean()
                ),

            "mean_gained_fdr_significance_count":
                float(
                    summary[
                        "gained_fdr_significance_count"
                    ].mean()
                ),
        }
    ]
)

overall.to_csv(
    OVERALL_SUMMARY_PATH,
    index=False,
)

print(
    overall.to_string(
        index=False
    )
)

print(
    f"\nSaved: {OVERALL_SUMMARY_PATH}"
)


# ============================================================
# 8. STRUCTURAL VALIDATION
# ============================================================

section(
    "8. STEP 2 VALIDATION STATUS"
)

if (
    len(summary) == 11
    and
    len(detail)
    == EXPECTED_GRIDS * 11
    and
    np.isfinite(
        summary[
            "sen_slope_spearman"
        ]
    ).all()
):
    print(
        "PASS - All 11 leave-one-year-out "
        "trend stability runs completed successfully."
    )

else:
    raise RuntimeError(
        "Leave-one-year-out validation is incomplete."
    )


print(
    "\nIMPORTANT INTERPRETATION:"
)

print(
    "- Sen slope Spearman measures stability of relative "
    "grid-level trend ordering."
)

print(
    "- Class agreement measures whether statistical "
    "trend labels remain unchanged after one year is removed."
)

print(
    "- Lower FDR class agreement does not automatically "
    "mean the trend method is invalid; significance can change "
    "when sample size drops from 11 to 10 years."
)

print(
    "- No arbitrary robustness threshold is imposed here. "
    "Observed stability will be reported transparently."
)

print(
    "\nSTEP 2 LEAVE-ONE-YEAR-OUT TREND STABILITY COMPLETE."
)