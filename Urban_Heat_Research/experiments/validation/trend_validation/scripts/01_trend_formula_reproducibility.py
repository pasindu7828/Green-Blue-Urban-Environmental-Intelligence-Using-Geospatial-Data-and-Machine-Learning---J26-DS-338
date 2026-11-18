from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================
# SETTINGS
# ============================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET = "LST_C"

EXPECTED_GRIDS = 2194
EXPECTED_ROWS = 24134
EXPECTED_YEARS_PER_GRID = 11

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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_PATH = OUTPUT_DIR / "01_trend_formula_reproducibility.csv"

SUMMARY_PATH = OUTPUT_DIR / "01_trend_formula_reproducibility_summary.csv"


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def independent_mann_kendall(values):
    y = np.asarray(values, dtype=float)
    n = len(y)

    s = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            if y[j] > y[i]:
                s += 1
            elif y[j] < y[i]:
                s -= 1

    _, counts = np.unique(y, return_counts=True)

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
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    p = float(
        2.0 * norm.sf(abs(z))
    )

    denominator = n * (n - 1) / 2.0
    tau = float(s / denominator)

    return {
        "independent_mk_s": int(s),
        "independent_mk_z": float(z),
        "independent_mk_p_value": p,
        "independent_mk_tau": tau,
    }


def independent_sen_slope(years, values):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)

    slopes = []

    for i in range(len(years) - 1):
        for j in range(i + 1, len(years)):
            slope = (
                values[j] - values[i]
            ) / (
                years[j] - years[i]
            )

            slopes.append(slope)

    return float(np.median(slopes))


def independent_bh(p_values):
    p = np.asarray(p_values, dtype=float)

    m = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted = (
        ranked
        * m
        / np.arange(1, m + 1, dtype=float)
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    result = np.empty(m, dtype=float)
    result[order] = adjusted

    return result


# ============================================================
# 1. LOAD DATA
# ============================================================

section("1. LOAD HISTORICAL DATA AND PRODUCTION TREND OUTPUT")

data = pd.read_csv(
    DATA_PATH,
    dtype={GRID_COL: str},
)

production = pd.read_csv(
    PRODUCTION_TREND_PATH,
    dtype={GRID_COL: str},
)

print(f"Historical rows : {len(data):,}")
print(f"Historical grids: {data[GRID_COL].nunique():,}")
print(f"Trend rows      : {len(production):,}")


# ============================================================
# 2. STRUCTURAL CHECK
# ============================================================

section("2. STRUCTURAL AUDIT")

if len(data) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS} rows, got {len(data)}"
    )

if data[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        "Historical grid count mismatch"
    )

if production[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        "Production trend grid count mismatch"
    )

counts = (
    data.groupby(GRID_COL)[YEAR_COL]
    .nunique()
)

if (
    counts.min() != EXPECTED_YEARS_PER_GRID
    or counts.max() != EXPECTED_YEARS_PER_GRID
):
    raise RuntimeError(
        "Each grid must contain exactly 11 historical years"
    )

if data.duplicated([GRID_COL, YEAR_COL]).any():
    raise RuntimeError(
        "Duplicate grid-year records found"
    )

print("PASS - historical dataset structure is valid.")


# ============================================================
# 3. INDEPENDENT MK + SEN CALCULATION
# ============================================================

section("3. INDEPENDENTLY RECOMPUTE MK AND SEN'S SLOPE")

rows = []

for grid_id, group in data.groupby(
    GRID_COL,
    sort=True,
):

    g = group.sort_values(YEAR_COL)

    years = g[YEAR_COL].to_numpy(
        dtype=float
    )

    values = g[TARGET].to_numpy(
        dtype=float
    )

    mk = independent_mann_kendall(
        values
    )

    sen_slope = independent_sen_slope(
        years,
        values,
    )

    rows.append(
        {
            GRID_COL: str(grid_id),
            **mk,
            "independent_sen_slope_c_per_year": (
                sen_slope
            ),
        }
    )

independent = pd.DataFrame(rows)

independent[
    "independent_mk_fdr_q_value"
] = independent_bh(
    independent[
        "independent_mk_p_value"
    ].to_numpy(dtype=float)
)

print(
    f"Independent grids calculated: "
    f"{len(independent):,}"
)


# ============================================================
# 4. COMPARE WITH PRODUCTION STEP 23
# ============================================================

section("4. COMPARE WITH FROZEN STEP-23 OUTPUT")

compare = production[
    [
        GRID_COL,
        "mk_s",
        "mk_z",
        "mk_p_value",
        "mk_tau",
        "sen_slope_c_per_year",
        "mk_fdr_q_value",
        "trend_raw_alpha_0_05",
        "trend_fdr_alpha_0_05",
    ]
].merge(
    independent,
    on=GRID_COL,
    how="inner",
    validate="one_to_one",
)

compare["abs_diff_mk_s"] = (
    compare["mk_s"]
    - compare["independent_mk_s"]
).abs()

compare["abs_diff_mk_z"] = (
    compare["mk_z"]
    - compare["independent_mk_z"]
).abs()

compare["abs_diff_mk_p"] = (
    compare["mk_p_value"]
    - compare["independent_mk_p_value"]
).abs()

compare["abs_diff_mk_tau"] = (
    compare["mk_tau"]
    - compare["independent_mk_tau"]
).abs()

compare["abs_diff_sen_slope"] = (
    compare["sen_slope_c_per_year"]
    - compare["independent_sen_slope_c_per_year"]
).abs()

compare["abs_diff_fdr_q"] = (
    compare["mk_fdr_q_value"]
    - compare["independent_mk_fdr_q_value"]
).abs()


# Independent labels
compare["independent_raw_class"] = np.where(
    (
        compare["independent_mk_p_value"] < 0.05
    )
    & (
        compare[
            "independent_sen_slope_c_per_year"
        ] > 0
    ),
    "Significant Increasing",
    np.where(
        (
            compare["independent_mk_p_value"] < 0.05
        )
        & (
            compare[
                "independent_sen_slope_c_per_year"
            ] < 0
        ),
        "Significant Decreasing",
        "No Significant Trend",
    ),
)

compare["independent_fdr_class"] = np.where(
    (
        compare["independent_mk_fdr_q_value"] < 0.05
    )
    & (
        compare[
            "independent_sen_slope_c_per_year"
        ] > 0
    ),
    "Significant Increasing",
    np.where(
        (
            compare["independent_mk_fdr_q_value"] < 0.05
        )
        & (
            compare[
                "independent_sen_slope_c_per_year"
            ] < 0
        ),
        "Significant Decreasing",
        "No Significant Trend",
    ),
)

compare["raw_class_match"] = (
    compare["trend_raw_alpha_0_05"]
    == compare["independent_raw_class"]
)

compare["fdr_class_match"] = (
    compare["trend_fdr_alpha_0_05"]
    == compare["independent_fdr_class"]
)

compare.to_csv(
    DETAIL_PATH,
    index=False,
)


# ============================================================
# 5. SUMMARY
# ============================================================

section("5. REPRODUCIBILITY SUMMARY")

summary = pd.DataFrame(
    [
        {
            "grid_count": len(compare),

            "max_abs_diff_mk_s": (
                compare["abs_diff_mk_s"].max()
            ),

            "max_abs_diff_mk_z": (
                compare["abs_diff_mk_z"].max()
            ),

            "max_abs_diff_mk_p": (
                compare["abs_diff_mk_p"].max()
            ),

            "max_abs_diff_mk_tau": (
                compare["abs_diff_mk_tau"].max()
            ),

            "max_abs_diff_sen_slope": (
                compare[
                    "abs_diff_sen_slope"
                ].max()
            ),

            "max_abs_diff_fdr_q": (
                compare[
                    "abs_diff_fdr_q"
                ].max()
            ),

            "raw_class_match_percent": (
                compare[
                    "raw_class_match"
                ].mean()
                * 100.0
            ),

            "fdr_class_match_percent": (
                compare[
                    "fdr_class_match"
                ].mean()
                * 100.0
            ),
        }
    ]
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

print(summary.to_string(index=False))


# ============================================================
# 6. STRICT VALIDATION
# ============================================================

section("6. STRICT VALIDATION RESULT")

numeric_checks = [
    compare["abs_diff_mk_s"].max() <= TOL,
    compare["abs_diff_mk_z"].max() <= TOL,
    compare["abs_diff_mk_p"].max() <= TOL,
    compare["abs_diff_mk_tau"].max() <= TOL,
    compare["abs_diff_sen_slope"].max() <= TOL,
    compare["abs_diff_fdr_q"].max() <= TOL,
]

class_checks = [
    compare["raw_class_match"].all(),
    compare["fdr_class_match"].all(),
]

if all(numeric_checks) and all(class_checks):
    print(
        "PASS - Independent MK, Sen slope and "
        "BH-FDR calculations reproduce the frozen "
        "Step-23 results."
    )
else:
    print(
        "REVIEW REQUIRED - At least one independently "
        "calculated trend statistic differs from the "
        "production Step-23 output."
    )

print(f"\nSaved: {DETAIL_PATH}")
print(f"Saved: {SUMMARY_PATH}")

print("\nSTEP 1 TREND FORMULA REPRODUCIBILITY COMPLETE.")