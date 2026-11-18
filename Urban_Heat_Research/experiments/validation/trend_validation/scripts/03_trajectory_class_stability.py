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
AREA_COL = "area_name"

EXPECTED_GRIDS = 2194
YEARS = list(range(2015, 2026))

BASELINE_YEARS = [2015, 2016, 2017, 2018, 2019]
RECENT_YEARS = [2021, 2022, 2023, 2024, 2025]

HOT_THRESHOLD = 0.75
STABLE_SPLIT = 0.50
RELATIVE_SLOPE_THRESHOLD = 0.010

CLASS_ORDER = [
    "Persistent Cool",
    "Diminishing",
    "Emerging",
    "Intensifying",
    "Persistent Hot",
]

TOL = 1e-10


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

PRODUCTION_PATH = (
    ROOT
    / "outputs"
    / "metrics"
    / "23b_grid_heat_trajectory_classes.csv"
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

REPRO_PATH = (
    OUTPUT_DIR
    / "03_trajectory_independent_reproduction.csv"
)

LOO_DETAIL_PATH = (
    OUTPUT_DIR
    / "03_trajectory_leave_one_year_out_detail.csv"
)

LOO_SUMMARY_PATH = (
    OUTPUT_DIR
    / "03_trajectory_leave_one_year_out_by_year.csv"
)

THRESHOLD_PATH = (
    OUTPUT_DIR
    / "03_trajectory_threshold_sensitivity.csv"
)

FINAL_SUMMARY_PATH = (
    OUTPUT_DIR
    / "03_trajectory_validation_summary.csv"
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
    relative_sen_slope,
    hot_threshold=HOT_THRESHOLD,
    stable_split=STABLE_SPLIT,
    slope_threshold=RELATIVE_SLOPE_THRESHOLD,
):
    if (
        relative_sen_slope >= slope_threshold
        and baseline_score < hot_threshold
        and recent_score >= hot_threshold
    ):
        return "Emerging"

    if relative_sen_slope >= slope_threshold:
        return "Intensifying"

    if relative_sen_slope <= -slope_threshold:
        return "Diminishing"

    if long_score >= stable_split:
        return "Persistent Hot"

    return "Persistent Cool"


def slope_direction(
    slope,
    threshold=RELATIVE_SLOPE_THRESHOLD,
):
    if slope >= threshold:
        return "Increasing"

    if slope <= -threshold:
        return "Decreasing"

    return "Stable"


def build_trajectory(
    data,
    baseline_years,
    recent_years,
):
    percentile_data = add_relative_percentile(
        data
    )

    rows = []

    for grid_id, group in percentile_data.groupby(
        GRID_COL,
        sort=True,
    ):
        g = group.sort_values(
            YEAR_COL
        )

        years = g[
            YEAR_COL
        ].to_numpy(dtype=float)

        values = g[
            "relative_heat_percentile"
        ].to_numpy(dtype=float)

        if len(g) < 3:
            raise RuntimeError(
                f"Too few years for grid {grid_id}"
            )

        sen = theilslopes(
            values,
            years,
            alpha=0.95,
        )

        relative_slope = float(
            sen.slope
        )

        baseline = g[
            g[YEAR_COL].isin(
                baseline_years
            )
        ]

        recent = g[
            g[YEAR_COL].isin(
                recent_years
            )
        ]

        if baseline.empty:
            raise RuntimeError(
                f"Missing baseline for {grid_id}"
            )

        if recent.empty:
            raise RuntimeError(
                f"Missing recent period for {grid_id}"
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

        trajectory_class = (
            classify_trajectory(
                baseline_score,
                recent_score,
                long_score,
                relative_slope,
            )
        )

        rows.append(
            {
                GRID_COL: str(grid_id),

                "baseline_relative_heat_median":
                    baseline_score,

                "recent_relative_heat_median":
                    recent_score,

                "long_term_relative_heat_median":
                    long_score,

                "relative_sen_slope_percentile_per_year":
                    relative_slope,

                "relative_direction_operational":
                    slope_direction(
                        relative_slope
                    ),

                "trajectory_class":
                    trajectory_class,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 1. LOAD
# ============================================================

section(
    "1. LOAD HISTORICAL DATA AND FROZEN TRAJECTORY OUTPUT"
)

data = pd.read_csv(
    DATA_PATH,
    dtype={GRID_COL: str},
)

production = pd.read_csv(
    PRODUCTION_PATH,
    dtype={GRID_COL: str},
)

print(
    f"Historical rows : {len(data):,}"
)

print(
    f"Historical grids: {data[GRID_COL].nunique():,}"
)

print(
    f"Trajectory rows : {len(production):,}"
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

if production[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        "Production trajectory grid count mismatch"
    )

if sorted(
    data[YEAR_COL].unique().tolist()
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

print(
    "PASS - structural audit complete."
)


# ============================================================
# 3. INDEPENDENT FULL TRAJECTORY REPRODUCTION
# ============================================================

section(
    "3. INDEPENDENT FULL TRAJECTORY REPRODUCTION"
)

independent = build_trajectory(
    data=data,
    baseline_years=BASELINE_YEARS,
    recent_years=RECENT_YEARS,
)

reference_columns = [
    GRID_COL,
    "baseline_relative_heat_median",
    "recent_relative_heat_median",
    "long_term_relative_heat_median",
    "relative_sen_slope_percentile_per_year",
    "relative_direction_operational",
    "trajectory_class",
]

compare = production[
    reference_columns
].merge(
    independent,
    on=GRID_COL,
    how="inner",
    suffixes=(
        "_production",
        "_independent",
    ),
    validate="one_to_one",
)

for col in [
    "baseline_relative_heat_median",
    "recent_relative_heat_median",
    "long_term_relative_heat_median",
    "relative_sen_slope_percentile_per_year",
]:
    compare[
        f"abs_diff_{col}"
    ] = (
        compare[
            f"{col}_production"
        ]
        -
        compare[
            f"{col}_independent"
        ]
    ).abs()

compare[
    "direction_match"
] = (
    compare[
        "relative_direction_operational_production"
    ]
    ==
    compare[
        "relative_direction_operational_independent"
    ]
)

compare[
    "trajectory_class_match"
] = (
    compare[
        "trajectory_class_production"
    ]
    ==
    compare[
        "trajectory_class_independent"
    ]
)

compare.to_csv(
    REPRO_PATH,
    index=False,
)

class_match_pct = (
    compare[
        "trajectory_class_match"
    ].mean()
    * 100.0
)

direction_match_pct = (
    compare[
        "direction_match"
    ].mean()
    * 100.0
)

max_slope_diff = (
    compare[
        "abs_diff_relative_sen_slope_percentile_per_year"
    ].max()
)

print(
    f"Trajectory class match : "
    f"{class_match_pct:.2f}%"
)

print(
    f"Direction match        : "
    f"{direction_match_pct:.2f}%"
)

print(
    f"Max relative slope diff: "
    f"{max_slope_diff:.12e}"
)


# ============================================================
# 4. LEAVE-ONE-YEAR-OUT TRAJECTORY STABILITY
# ============================================================

section(
    "4. LEAVE-ONE-YEAR-OUT TRAJECTORY STABILITY"
)

reference = production[
    [
        GRID_COL,
        "relative_sen_slope_percentile_per_year",
        "relative_direction_operational",
        "trajectory_class",
    ]
].copy()

reference = reference.rename(
    columns={
        "relative_sen_slope_percentile_per_year":
            "full_relative_sen_slope",

        "relative_direction_operational":
            "full_relative_direction",

        "trajectory_class":
            "full_trajectory_class",
    }
)

detail_frames = []
summary_rows = []

for omitted_year in YEARS:

    loo_data = data[
        data[YEAR_COL] != omitted_year
    ].copy()

    baseline_years = [
        y
        for y in BASELINE_YEARS
        if y != omitted_year
    ]

    recent_years = [
        y
        for y in RECENT_YEARS
        if y != omitted_year
    ]

    loo = build_trajectory(
        data=loo_data,
        baseline_years=baseline_years,
        recent_years=recent_years,
    )

    loo = loo.rename(
        columns={
            "relative_sen_slope_percentile_per_year":
                "loo_relative_sen_slope",

            "relative_direction_operational":
                "loo_relative_direction",

            "trajectory_class":
                "loo_trajectory_class",
        }
    )

    c = reference.merge(
        loo[
            [
                GRID_COL,
                "loo_relative_sen_slope",
                "loo_relative_direction",
                "loo_trajectory_class",
            ]
        ],
        on=GRID_COL,
        how="inner",
        validate="one_to_one",
    )

    c.insert(
        0,
        "omitted_year",
        omitted_year,
    )

    c[
        "abs_relative_slope_change"
    ] = (
        c[
            "loo_relative_sen_slope"
        ]
        -
        c[
            "full_relative_sen_slope"
        ]
    ).abs()

    c[
        "direction_same"
    ] = (
        c[
            "loo_relative_direction"
        ]
        ==
        c[
            "full_relative_direction"
        ]
    )

    c[
        "trajectory_class_same"
    ] = (
        c[
            "loo_trajectory_class"
        ]
        ==
        c[
            "full_trajectory_class"
        ]
    )

    rho = float(
        spearmanr(
            c[
                "full_relative_sen_slope"
            ],
            c[
                "loo_relative_sen_slope"
            ],
        )[0]
    )

    row = {
        "omitted_year":
            omitted_year,

        "grid_count":
            len(c),

        "relative_slope_spearman":
            rho,

        "direction_agreement_percent":
            float(
                c[
                    "direction_same"
                ].mean()
                * 100.0
            ),

        "trajectory_class_agreement_percent":
            float(
                c[
                    "trajectory_class_same"
                ].mean()
                * 100.0
            ),

        "median_abs_relative_slope_change":
            float(
                c[
                    "abs_relative_slope_change"
                ].median()
            ),

        "p95_abs_relative_slope_change":
            float(
                c[
                    "abs_relative_slope_change"
                ].quantile(0.95)
            ),

        "max_abs_relative_slope_change":
            float(
                c[
                    "abs_relative_slope_change"
                ].max()
            ),
    }

    # Class-specific retention
    for class_name in CLASS_ORDER:

        mask = (
            c[
                "full_trajectory_class"
            ]
            == class_name
        )

        n = int(
            mask.sum()
        )

        if n > 0:
            retained = float(
                c.loc[
                    mask,
                    "trajectory_class_same",
                ].mean()
                * 100.0
            )

        else:
            retained = np.nan

        key = (
            class_name.lower()
            .replace(
                " ",
                "_",
            )
        )

        row[
            f"{key}_retention_percent"
        ] = retained

    summary_rows.append(
        row
    )

    detail_frames.append(
        c
    )

    print(
        f"{omitted_year}: "
        f"rho={rho:.4f} | "
        f"direction={row['direction_agreement_percent']:.2f}% | "
        f"class={row['trajectory_class_agreement_percent']:.2f}%"
    )


loo_detail = pd.concat(
    detail_frames,
    ignore_index=True,
)

loo_summary = pd.DataFrame(
    summary_rows
)

loo_detail.to_csv(
    LOO_DETAIL_PATH,
    index=False,
)

loo_summary.to_csv(
    LOO_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 5. THRESHOLD SENSITIVITY
# ============================================================

section(
    "5. FIXED THRESHOLD SENSITIVITY"
)

independent_full = independent.copy()

reference_labels = (
    independent_full[
        "trajectory_class"
    ].to_numpy()
)

hot_variants = [
    0.70,
    0.75,
    0.80,
]

slope_variants = [
    0.008,
    0.010,
    0.012,
]

threshold_rows = []

for hot_threshold in hot_variants:

    for slope_threshold in slope_variants:

        labels = []

        for _, r in independent_full.iterrows():

            label = classify_trajectory(
                baseline_score=r[
                    "baseline_relative_heat_median"
                ],

                recent_score=r[
                    "recent_relative_heat_median"
                ],

                long_score=r[
                    "long_term_relative_heat_median"
                ],

                relative_sen_slope=r[
                    "relative_sen_slope_percentile_per_year"
                ],

                hot_threshold=hot_threshold,

                stable_split=STABLE_SPLIT,

                slope_threshold=slope_threshold,
            )

            labels.append(
                label
            )

        labels = np.asarray(
            labels,
            dtype=object,
        )

        row = {
            "hot_threshold":
                hot_threshold,

            "stable_split":
                STABLE_SPLIT,

            "relative_slope_threshold":
                slope_threshold,

            "is_reference_configuration":
                bool(
                    np.isclose(
                        hot_threshold,
                        HOT_THRESHOLD,
                    )
                    and
                    np.isclose(
                        slope_threshold,
                        RELATIVE_SLOPE_THRESHOLD,
                    )
                ),

            "pct_same_as_reference_classification":
                float(
                    np.mean(
                        labels
                        ==
                        reference_labels
                    )
                    * 100.0
                ),
        }

        for class_name in CLASS_ORDER:

            key = (
                class_name.lower()
                .replace(
                    " ",
                    "_",
                )
            )

            row[
                f"count_{key}"
            ] = int(
                np.sum(
                    labels
                    ==
                    class_name
                )
            )

        threshold_rows.append(
            row
        )


threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_df.to_csv(
    THRESHOLD_PATH,
    index=False,
)

print(
    threshold_df.to_string(
        index=False
    )
)


# ============================================================
# 6. OVERALL SUMMARY
# ============================================================

section(
    "6. OVERALL TRAJECTORY VALIDATION SUMMARY"
)

worst_class_idx = (
    loo_summary[
        "trajectory_class_agreement_percent"
    ]
    .idxmin()
)

worst_rho_idx = (
    loo_summary[
        "relative_slope_spearman"
    ]
    .idxmin()
)

non_reference_thresholds = (
    threshold_df[
        ~threshold_df[
            "is_reference_configuration"
        ]
    ]
)

final_summary = pd.DataFrame(
    [
        {
            "independent_full_class_match_percent":
                class_match_pct,

            "independent_full_direction_match_percent":
                direction_match_pct,

            "max_independent_relative_slope_difference":
                max_slope_diff,

            "loo_runs":
                len(
                    loo_summary
                ),

            "mean_loo_relative_slope_spearman":
                float(
                    loo_summary[
                        "relative_slope_spearman"
                    ].mean()
                ),

            "minimum_loo_relative_slope_spearman":
                float(
                    loo_summary[
                        "relative_slope_spearman"
                    ].min()
                ),

            "worst_spearman_omitted_year":
                int(
                    loo_summary.loc[
                        worst_rho_idx,
                        "omitted_year",
                    ]
                ),

            "mean_loo_direction_agreement_percent":
                float(
                    loo_summary[
                        "direction_agreement_percent"
                    ].mean()
                ),

            "minimum_loo_direction_agreement_percent":
                float(
                    loo_summary[
                        "direction_agreement_percent"
                    ].min()
                ),

            "mean_loo_trajectory_class_agreement_percent":
                float(
                    loo_summary[
                        "trajectory_class_agreement_percent"
                    ].mean()
                ),

            "minimum_loo_trajectory_class_agreement_percent":
                float(
                    loo_summary[
                        "trajectory_class_agreement_percent"
                    ].min()
                ),

            "worst_class_agreement_omitted_year":
                int(
                    loo_summary.loc[
                        worst_class_idx,
                        "omitted_year",
                    ]
                ),

            "minimum_nonreference_threshold_class_agreement_percent":
                float(
                    non_reference_thresholds[
                        "pct_same_as_reference_classification"
                    ].min()
                ),

            "mean_nonreference_threshold_class_agreement_percent":
                float(
                    non_reference_thresholds[
                        "pct_same_as_reference_classification"
                    ].mean()
                ),
        }
    ]
)

final_summary.to_csv(
    FINAL_SUMMARY_PATH,
    index=False,
)

print(
    final_summary.to_string(
        index=False
    )
)


# ============================================================
# 7. STATUS
# ============================================================

section(
    "7. STEP 3 VALIDATION STATUS"
)

reproduction_ok = bool(
    class_match_pct == 100.0
    and
    direction_match_pct == 100.0
    and
    max_slope_diff <= TOL
)

structural_ok = bool(
    len(loo_summary) == 11
    and
    len(loo_detail)
    == EXPECTED_GRIDS * 11
    and
    len(threshold_df) == 9
)

if reproduction_ok and structural_ok:

    print(
        "PASS - Frozen trajectory logic was independently "
        "reproduced and all stability analyses completed."
    )

else:

    print(
        "REVIEW REQUIRED - Independent reproduction or "
        "structural checks did not fully match."
    )


print(
    "\nIMPORTANT INTERPRETATION:"
)

print(
    "- Trajectory classes describe RELATIVE spatial heat "
    "position within Kaduwela."
)

print(
    "- They are not claims of statistically significant "
    "absolute warming or cooling."
)

print(
    "- Leave-one-year-out analysis measures temporal "
    "sensitivity of the operational classification."
)

print(
    "- Threshold sensitivity measures dependence on the "
    "fixed operational cut-offs."
)

print(
    "- No threshold configuration is optimized using these "
    "validation results."
)

print(
    f"\nSaved: {REPRO_PATH}"
)

print(
    f"Saved: {LOO_DETAIL_PATH}"
)

print(
    f"Saved: {LOO_SUMMARY_PATH}"
)

print(
    f"Saved: {THRESHOLD_PATH}"
)

print(
    f"Saved: {FINAL_SUMMARY_PATH}"
)

print(
    "\nSTEP 3 TRAJECTORY CLASS STABILITY COMPLETE."
)