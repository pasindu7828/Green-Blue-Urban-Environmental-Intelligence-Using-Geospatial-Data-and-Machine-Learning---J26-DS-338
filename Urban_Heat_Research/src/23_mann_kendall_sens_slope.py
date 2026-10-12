"""
STEP 23 - GRID-LEVEL MANN-KENDALL TREND TEST + SEN'S SLOPE
Urban Heat Research - Kaduwela, Sri Lanka

PRIMARY PURPOSE
---------------
Quantify historical observed LST trends for every 210 m x 210 m grid cell
using the complete 2015-2025 annual LST_C series.

For each of the 2,194 grids this script calculates:
- Mann-Kendall S statistic
- tie-corrected variance
- Z statistic
- two-sided p-value
- Kendall-style Mann-Kendall tau
- Sen's slope (degrees C / year)
- 95% confidence interval for Sen's slope
- raw p<0.05 trend class
- Benjamini-Hochberg FDR q-value across all grid tests
- FDR-controlled trend class

2017 POLICY
-----------
The primary result keeps 2017 exactly as observed, consistent with the
approved data-quality decision. A second run excluding 2017 is saved only
as a sensitivity analysis; it does not replace the primary result.

IMPORTANT
---------
This step uses OBSERVED historical LST_C only.
It does NOT use forecast-model predictions.
It does NOT modify or retrain the final forecasting model.
"""

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from scipy.stats import norm, theilslopes


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET = "LST_C"
AREA_COL = "area_name"

START_YEAR = 2015
END_YEAR = 2025
EXPECTED_YEARS = list(range(START_YEAR, END_YEAR + 1))

EXPECTED_ROWS = 24134
EXPECTED_GRIDS = 2194
EXPECTED_OBS_PER_GRID = 11

ALPHA = 0.05
FDR_ALPHA = 0.05

# Verified in finalized Step 04.
EXPECTED_INPUT_SHA256 = (
    "0e030b2ec0b5c4080cd4226bec798189a178aa45efd9962d19f38dc9cb170f46"
)


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "kaduwela_preprocessed_base.csv"
)

GRID_UTM_PATH = (
    ROOT
    / "data"
    / "processed"
    / "kaduwela_grid_geometry_utm44n.geojson"
)

GRID_WGS84_PATH = (
    ROOT
    / "data"
    / "processed"
    / "kaduwela_grid_geometry_wgs84.geojson"
)

METRICS_DIR = ROOT / "outputs" / "metrics"
REPORTS_DIR = ROOT / "outputs" / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"
MAPS_DIR = ROOT / "outputs" / "maps"

for folder in [
    METRICS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    MAPS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

GRID_RESULTS_PATH = (
    METRICS_DIR
    / "23_grid_mann_kendall_sens_slope.csv"
)

TREND_COUNTS_PATH = (
    METRICS_DIR
    / "23_fdr_trend_class_counts.csv"
)

AREA_SUMMARY_PATH = (
    METRICS_DIR
    / "23_area_trend_summary.csv"
)

CITY_ANNUAL_PATH = (
    METRICS_DIR
    / "23_citywide_annual_lst_series.csv"
)

CITY_TREND_PATH = (
    METRICS_DIR
    / "23_citywide_mann_kendall_sens_slope.csv"
)

SENSITIVITY_PATH = (
    METRICS_DIR
    / "23_2017_sensitivity_comparison.csv"
)

SENSITIVITY_SUMMARY_PATH = (
    METRICS_DIR
    / "23_2017_sensitivity_summary.csv"
)

MAP_GEOJSON_PATH = (
    MAPS_DIR
    / "23_mann_kendall_sens_slope_wgs84.geojson"
)

SLOPE_MAP_PATH = (
    FIGURES_DIR
    / "23_sens_slope_map.png"
)

TREND_MAP_PATH = (
    FIGURES_DIR
    / "23_fdr_trend_class_map.png"
)

SLOPE_HIST_PATH = (
    FIGURES_DIR
    / "23_sens_slope_distribution.png"
)

TREND_COUNT_FIG_PATH = (
    FIGURES_DIR
    / "23_fdr_trend_class_counts.png"
)

METHOD_PATH = (
    REPORTS_DIR
    / "23_trend_methodology.json"
)

AUDIT_PATH = (
    REPORTS_DIR
    / "23_mann_kendall_sens_slope_audit.csv"
)

REPORT_PATH = (
    REPORTS_DIR
    / "23_mann_kendall_sens_slope_report.txt"
)

HASH_PATH = (
    REPORTS_DIR
    / "23_mann_kendall_sens_slope_hashes.csv"
)


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 128)
    print(title)
    print("=" * 128)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file missing: {path}"
        )


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def mann_kendall_test(values):
    """
    Original Mann-Kendall test for a time-ordered series.

    Uses:
    - S statistic
    - tie-corrected variance
    - continuity-corrected Z
    - two-sided normal-approximation p-value
    - tau = S / [n(n-1)/2]

    Annual observations are already ordered by year before this function.
    """
    y = np.asarray(
        values,
        dtype=float,
    )

    n = len(y)

    if n < 3:
        raise ValueError(
            "Mann-Kendall requires at least 3 observations"
        )

    s = 0

    for i in range(
        n - 1
    ):
        s += int(
            np.sign(
                y[
                    i + 1:
                ]
                - y[
                    i
                ]
            ).sum()
        )

    _, tie_counts = np.unique(
        y,
        return_counts=True,
    )

    tie_counts = tie_counts[
        tie_counts
        > 1
    ]

    tie_correction = float(
        np.sum(
            tie_counts
            * (
                tie_counts
                - 1
            )
            * (
                2
                * tie_counts
                + 5
            )
        )
    )

    var_s = (
        n
        * (
            n
            - 1
        )
        * (
            2
            * n
            + 5
        )
        - tie_correction
    ) / 18.0

    if var_s <= 0:
        z = 0.0
    elif s > 0:
        z = (
            s
            - 1
        ) / math.sqrt(
            var_s
        )
    elif s < 0:
        z = (
            s
            + 1
        ) / math.sqrt(
            var_s
        )
    else:
        z = 0.0

    p_value = float(
        2.0
        * norm.sf(
            abs(
                z
            )
        )
    )

    denominator = (
        n
        * (
            n
            - 1
        )
        / 2.0
    )

    tau = float(
        s
        / denominator
    )

    return {
        "mk_s": int(
            s
        ),
        "mk_var_s": float(
            var_s
        ),
        "mk_z": float(
            z
        ),
        "mk_p_value": p_value,
        "mk_tau": tau,
    }


def calculate_sen_slope(years, values):
    years = np.asarray(
        years,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    result = theilslopes(
        values,
        years,
        alpha=0.95,
        method="separate",
    )

    return {
        "sen_slope_c_per_year": float(
            result.slope
        ),
        "sen_intercept_c": float(
            result.intercept
        ),
        "sen_slope_ci95_low": float(
            result.low_slope
        ),
        "sen_slope_ci95_high": float(
            result.high_slope
        ),
    }


def raw_trend_class(p_value, slope):
    if p_value < ALPHA:
        if slope > 0:
            return "Significant Increasing"
        if slope < 0:
            return "Significant Decreasing"

    return "No Significant Trend"


def fdr_trend_class(q_value, slope):
    if q_value < FDR_ALPHA:
        if slope > 0:
            return "Significant Increasing"
        if slope < 0:
            return "Significant Decreasing"

    return "No Significant Trend"


def benjamini_hochberg(p_values):
    """
    Return Benjamini-Hochberg FDR-adjusted p-values (q-values).
    """
    p = np.asarray(
        p_values,
        dtype=float,
    )

    if np.any(
        ~np.isfinite(
            p
        )
    ):
        raise ValueError(
            "Non-finite p-values found before FDR adjustment"
        )

    m = len(
        p
    )

    order = np.argsort(
        p
    )

    ranked_p = p[
        order
    ]

    adjusted_ranked = (
        ranked_p
        * m
        / np.arange(
            1,
            m + 1,
            dtype=float,
        )
    )

    # Enforce monotonicity.
    adjusted_ranked = np.minimum.accumulate(
        adjusted_ranked[
            ::-1
        ]
    )[
        ::-1
    ]

    adjusted_ranked = np.clip(
        adjusted_ranked,
        0.0,
        1.0,
    )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    adjusted[
        order
    ] = adjusted_ranked

    return adjusted


def compute_grid_trends(data, exclude_2017=False):
    rows = []

    for grid_id, group in data.groupby(
        GRID_COL,
        sort=True,
    ):
        g = group.sort_values(
            YEAR_COL
        ).copy()

        if exclude_2017:
            g = g[
                g[
                    YEAR_COL
                ]
                != 2017
            ].copy()

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

        mk = mann_kendall_test(
            values
        )

        sen = calculate_sen_slope(
            years,
            values,
        )

        area_names = (
            group[
                AREA_COL
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(
            area_names
        ) != 1:
            raise RuntimeError(
                f"Grid {grid_id} has inconsistent area_name values: "
                f"{area_names}"
            )

        first_idx = int(
            np.argmin(
                years
            )
        )

        last_idx = int(
            np.argmax(
                years
            )
        )

        row = {
            GRID_COL: str(
                grid_id
            ),
            AREA_COL: area_names[
                0
            ],
            "analysis_start_year": int(
                np.min(
                    years
                )
            ),
            "analysis_end_year": int(
                np.max(
                    years
                )
            ),
            "n_years": int(
                len(
                    years
                )
            ),
            "mean_lst_c": float(
                np.mean(
                    values
                )
            ),
            "median_lst_c": float(
                np.median(
                    values
                )
            ),
            "std_lst_c": float(
                np.std(
                    values,
                    ddof=1,
                )
            ),
            "first_observed_lst_c": float(
                values[
                    first_idx
                ]
            ),
            "last_observed_lst_c": float(
                values[
                    last_idx
                ]
            ),
            "observed_first_to_last_change_c": float(
                values[
                    last_idx
                ]
                - values[
                    first_idx
                ]
            ),
        }

        row.update(
            mk
        )

        row.update(
            sen
        )

        row[
            "sen_estimated_change_over_analysis_period_c"
        ] = float(
            row[
                "sen_slope_c_per_year"
            ]
            * (
                row[
                    "analysis_end_year"
                ]
                - row[
                    "analysis_start_year"
                ]
            )
        )

        row[
            "trend_raw_alpha_0_05"
        ] = raw_trend_class(
            row[
                "mk_p_value"
            ],
            row[
                "sen_slope_c_per_year"
            ],
        )

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "mk_fdr_q_value"
    ] = benjamini_hochberg(
        result[
            "mk_p_value"
        ].to_numpy(
            dtype=float
        )
    )

    result[
        "trend_fdr_alpha_0_05"
    ] = [
        fdr_trend_class(
            q,
            slope,
        )
        for q, slope in zip(
            result[
                "mk_fdr_q_value"
            ],
            result[
                "sen_slope_c_per_year"
            ],
        )
    ]

    result[
        "raw_significant"
    ] = (
        result[
            "mk_p_value"
        ]
        < ALPHA
    )

    result[
        "fdr_significant"
    ] = (
        result[
            "mk_fdr_q_value"
        ]
        < FDR_ALPHA
    )

    return result


# =============================================================================
# 0. PREREQUISITE AUDIT
# =============================================================================

section(
    "STEP 23 - GRID-LEVEL MANN-KENDALL + SEN'S SLOPE"
)

for path in [
    DATA_PATH,
    GRID_UTM_PATH,
    GRID_WGS84_PATH,
]:
    require_file(
        path
    )

input_hash = sha256_file(
    DATA_PATH
)

if input_hash != EXPECTED_INPUT_SHA256:
    raise RuntimeError(
        "Processed historical input hash changed.\n"
        f"Expected: {EXPECTED_INPUT_SHA256}\n"
        f"Actual  : {input_hash}"
    )

print(
    f"PASS: historical input hash verified: {input_hash}"
)

print(
    "Primary historical trend period : 2015-2025"
)

print(
    "Primary 2017 policy              : PRESERVE"
)

print(
    "Secondary sensitivity            : repeat without 2017"
)

print(
    "Forecast-model predictions       : NOT USED"
)


# =============================================================================
# 1. LOAD + VALIDATE HISTORICAL OBSERVED LST
# =============================================================================

section(
    "1. LOAD + VALIDATE 2015-2025 OBSERVED LST"
)

data = pd.read_csv(
    DATA_PATH,
    dtype={
        GRID_COL: str,
    },
)

required_columns = {
    GRID_COL,
    YEAR_COL,
    TARGET,
    AREA_COL,
}

missing = sorted(
    required_columns
    - set(
        data.columns
    )
)

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

data[
    YEAR_COL
] = pd.to_numeric(
    data[
        YEAR_COL
    ],
    errors="coerce",
)

data[
    TARGET
] = pd.to_numeric(
    data[
        TARGET
    ],
    errors="coerce",
)

if data[
    [
        YEAR_COL,
        TARGET,
    ]
].isna().any().any():
    raise RuntimeError(
        "Missing or non-numeric year/LST values found"
    )

data[
    YEAR_COL
] = data[
    YEAR_COL
].astype(
    int
)

years = sorted(
    data[
        YEAR_COL
    ].unique().tolist()
)

if years != EXPECTED_YEARS:
    raise RuntimeError(
        f"Years={years}, expected={EXPECTED_YEARS}"
    )

if len(
    data
) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Rows={len(data)}, expected={EXPECTED_ROWS}"
    )

if data[
    GRID_COL
].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Unique grids={data[GRID_COL].nunique()}, "
        f"expected={EXPECTED_GRIDS}"
    )

if data.duplicated(
    subset=[
        GRID_COL,
        YEAR_COL,
    ]
).any():
    raise RuntimeError(
        "Duplicate grid-year rows found"
    )

counts = data.groupby(
    GRID_COL
).size()

if not (
    counts
    == EXPECTED_OBS_PER_GRID
).all():
    raise RuntimeError(
        "Not every grid has exactly 11 observations"
    )

if not np.isfinite(
    data[
        TARGET
    ].to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "Non-finite LST values found"
    )

rows_2017 = int(
    (
        data[
            YEAR_COL
        ]
        == 2017
    ).sum()
)

if rows_2017 != EXPECTED_GRIDS:
    raise RuntimeError(
        f"2017 rows={rows_2017}, expected={EXPECTED_GRIDS}"
    )

print(
    f"Rows          : {len(data):,}"
)

print(
    f"Unique grids  : {data[GRID_COL].nunique():,}"
)

print(
    f"Years         : {years}"
)

print(
    f"Rows per grid : {counts.min()} to {counts.max()}"
)

print(
    f"2017 rows kept: {rows_2017:,}"
)

print(
    "PASS: every grid has a complete 2015-2025 observed LST series."
)


# =============================================================================
# 2. PRIMARY GRID-LEVEL MK + SEN
# =============================================================================

section(
    "2. PRIMARY 2015-2025 GRID-LEVEL TREND ANALYSIS"
)

grid_results = compute_grid_trends(
    data,
    exclude_2017=False,
)

grid_results = grid_results.sort_values(
    GRID_COL
).reset_index(
    drop=True
)

if len(
    grid_results
) != EXPECTED_GRIDS:
    raise RuntimeError(
        "Unexpected number of primary grid trend results"
    )

grid_results.to_csv(
    GRID_RESULTS_PATH,
    index=False,
)

raw_sig_count = int(
    grid_results[
        "raw_significant"
    ].sum()
)

fdr_sig_count = int(
    grid_results[
        "fdr_significant"
    ].sum()
)

trend_counts = (
    grid_results[
        "trend_fdr_alpha_0_05"
    ]
    .value_counts()
    .rename_axis(
        "trend_class"
    )
    .reset_index(
        name="grid_count"
    )
)

trend_counts[
    "percent_of_grids"
] = (
    trend_counts[
        "grid_count"
    ]
    / EXPECTED_GRIDS
    * 100.0
)

trend_counts.to_csv(
    TREND_COUNTS_PATH,
    index=False,
)

print(
    f"Raw p<0.05 significant grids : {raw_sig_count:,}"
)

print(
    f"FDR q<0.05 significant grids : {fdr_sig_count:,}"
)

print(
    "\nFDR-controlled trend classes:"
)

print(
    trend_counts.to_string(
        index=False
    )
)

print(
    f"\nSaved: {GRID_RESULTS_PATH}"
)

print(
    f"Saved: {TREND_COUNTS_PATH}"
)


# =============================================================================
# 3. CITYWIDE ANNUAL TREND
# =============================================================================

section(
    "3. CITYWIDE ANNUAL LST TREND"
)

city_annual = (
    data.groupby(
        YEAR_COL,
        as_index=False,
    )
    .agg(
        mean_lst_c=(
            TARGET,
            "mean",
        ),
        median_lst_c=(
            TARGET,
            "median",
        ),
        std_lst_c=(
            TARGET,
            "std",
        ),
        grid_count=(
            TARGET,
            "size",
        ),
    )
)

city_annual.to_csv(
    CITY_ANNUAL_PATH,
    index=False,
)

city_mk = mann_kendall_test(
    city_annual[
        "mean_lst_c"
    ].to_numpy(
        dtype=float
    )
)

city_sen = calculate_sen_slope(
    city_annual[
        YEAR_COL
    ].to_numpy(
        dtype=float
    ),
    city_annual[
        "mean_lst_c"
    ].to_numpy(
        dtype=float
    ),
)

city_result = {
    "analysis": "Kaduwela citywide annual mean LST",
    "start_year": START_YEAR,
    "end_year": END_YEAR,
    "n_years": len(
        city_annual
    ),
}

city_result.update(
    city_mk
)

city_result.update(
    city_sen
)

city_result[
    "trend_raw_alpha_0_05"
] = raw_trend_class(
    city_result[
        "mk_p_value"
    ],
    city_result[
        "sen_slope_c_per_year"
    ],
)

city_df = pd.DataFrame(
    [
        city_result
    ]
)

city_df.to_csv(
    CITY_TREND_PATH,
    index=False,
)

print(
    city_annual.to_string(
        index=False
    )
)

print(
    "\nCitywide result:"
)

print(
    city_df.to_string(
        index=False
    )
)


# =============================================================================
# 4. AREA-LEVEL SUMMARY
# =============================================================================

section(
    "4. AREA-LEVEL TREND SUMMARY"
)

area_summary = (
    grid_results.groupby(
        AREA_COL,
        as_index=False,
    )
    .agg(
        grid_count=(
            GRID_COL,
            "count",
        ),
        mean_historical_lst_c=(
            "mean_lst_c",
            "mean",
        ),
        mean_sen_slope_c_per_year=(
            "sen_slope_c_per_year",
            "mean",
        ),
        median_sen_slope_c_per_year=(
            "sen_slope_c_per_year",
            "median",
        ),
        significant_increasing_grids=(
            "trend_fdr_alpha_0_05",
            lambda x: int(
                (
                    x
                    == "Significant Increasing"
                ).sum()
            ),
        ),
        significant_decreasing_grids=(
            "trend_fdr_alpha_0_05",
            lambda x: int(
                (
                    x
                    == "Significant Decreasing"
                ).sum()
            ),
        ),
        no_significant_trend_grids=(
            "trend_fdr_alpha_0_05",
            lambda x: int(
                (
                    x
                    == "No Significant Trend"
                ).sum()
            ),
        ),
    )
)

area_summary[
    "pct_significant_increasing"
] = (
    area_summary[
        "significant_increasing_grids"
    ]
    / area_summary[
        "grid_count"
    ]
    * 100.0
)

area_summary[
    "pct_significant_decreasing"
] = (
    area_summary[
        "significant_decreasing_grids"
    ]
    / area_summary[
        "grid_count"
    ]
    * 100.0
)

area_summary.to_csv(
    AREA_SUMMARY_PATH,
    index=False,
)

print(
    f"Area count: {len(area_summary)}"
)

print(
    "\nTop 15 areas by mean Sen slope:"
)

print(
    area_summary.sort_values(
        "mean_sen_slope_c_per_year",
        ascending=False,
    )
    .head(
        15
    )
    .to_string(
        index=False
    )
)


# =============================================================================
# 5. 2017 SENSITIVITY - SECONDARY ONLY
# =============================================================================

section(
    "5. 2017 SENSITIVITY ANALYSIS - SECONDARY ONLY"
)

sens = compute_grid_trends(
    data,
    exclude_2017=True,
)

sens = sens[
    [
        GRID_COL,
        "n_years",
        "mk_p_value",
        "mk_fdr_q_value",
        "mk_tau",
        "sen_slope_c_per_year",
        "sen_slope_ci95_low",
        "sen_slope_ci95_high",
        "trend_raw_alpha_0_05",
        "trend_fdr_alpha_0_05",
    ]
].rename(
    columns={
        "n_years": "without_2017_n_years",
        "mk_p_value": "without_2017_mk_p_value",
        "mk_fdr_q_value": "without_2017_mk_fdr_q_value",
        "mk_tau": "without_2017_mk_tau",
        "sen_slope_c_per_year": "without_2017_sen_slope_c_per_year",
        "sen_slope_ci95_low": "without_2017_sen_slope_ci95_low",
        "sen_slope_ci95_high": "without_2017_sen_slope_ci95_high",
        "trend_raw_alpha_0_05": "without_2017_trend_raw_alpha_0_05",
        "trend_fdr_alpha_0_05": "without_2017_trend_fdr_alpha_0_05",
    }
)

comparison = grid_results[
    [
        GRID_COL,
        AREA_COL,
        "mk_p_value",
        "mk_fdr_q_value",
        "mk_tau",
        "sen_slope_c_per_year",
        "trend_raw_alpha_0_05",
        "trend_fdr_alpha_0_05",
    ]
].merge(
    sens,
    on=GRID_COL,
    how="inner",
    validate="one_to_one",
)

comparison[
    "sen_slope_sign_same"
] = (
    np.sign(
        comparison[
            "sen_slope_c_per_year"
        ]
    )
    == np.sign(
        comparison[
            "without_2017_sen_slope_c_per_year"
        ]
    )
)

comparison[
    "fdr_trend_class_same"
] = (
    comparison[
        "trend_fdr_alpha_0_05"
    ]
    == comparison[
        "without_2017_trend_fdr_alpha_0_05"
    ]
)

comparison[
    "absolute_slope_difference_c_per_year"
] = np.abs(
    comparison[
        "sen_slope_c_per_year"
    ]
    - comparison[
        "without_2017_sen_slope_c_per_year"
    ]
)

comparison.to_csv(
    SENSITIVITY_PATH,
    index=False,
)

sensitivity_summary = pd.DataFrame(
    [
        {
            "grid_count": len(
                comparison
            ),
            "primary_includes_2017": True,
            "secondary_excludes_2017": True,
            "pct_same_sen_slope_sign": float(
                comparison[
                    "sen_slope_sign_same"
                ].mean()
                * 100.0
            ),
            "pct_same_fdr_trend_class": float(
                comparison[
                    "fdr_trend_class_same"
                ].mean()
                * 100.0
            ),
            "median_abs_slope_difference_c_per_year": float(
                comparison[
                    "absolute_slope_difference_c_per_year"
                ].median()
            ),
            "mean_abs_slope_difference_c_per_year": float(
                comparison[
                    "absolute_slope_difference_c_per_year"
                ].mean()
            ),
        }
    ]
)

sensitivity_summary.to_csv(
    SENSITIVITY_SUMMARY_PATH,
    index=False,
)

print(
    sensitivity_summary.to_string(
        index=False
    )
)

print(
    "\nPrimary result still includes 2017."
)

print(
    "The without-2017 result is diagnostic sensitivity evidence only."
)


# =============================================================================
# 6. MAP-READY OUTPUT + LOCATION
# =============================================================================

section(
    "6. BUILD MAP-READY TREND OUTPUT"
)

grid_utm = gpd.read_file(
    GRID_UTM_PATH
)

grid_wgs = gpd.read_file(
    GRID_WGS84_PATH
)

for gdf, label in [
    (
        grid_utm,
        "UTM grid",
    ),
    (
        grid_wgs,
        "WGS84 grid",
    ),
]:
    if GRID_COL not in gdf.columns:
        raise RuntimeError(
            f"{label} missing {GRID_COL}"
        )

    gdf[
        GRID_COL
    ] = gdf[
        GRID_COL
    ].astype(
        str
    )

    if len(
        gdf
    ) != EXPECTED_GRIDS:
        raise RuntimeError(
            f"{label} row count invalid"
        )

    if gdf[
        GRID_COL
    ].nunique() != EXPECTED_GRIDS:
        raise RuntimeError(
            f"{label} grid IDs not unique"
        )

# Centroids calculated in projected CRS, then converted to WGS84.
centroid_points = grid_utm[
    [
        GRID_COL,
        "geometry",
    ]
].copy()

centroid_points[
    "geometry"
] = centroid_points.geometry.centroid

centroid_points = centroid_points.to_crs(
    "EPSG:4326"
)

centroid_lookup = pd.DataFrame(
    {
        GRID_COL: centroid_points[
            GRID_COL
        ].astype(
            str
        ),
        "centroid_lon": centroid_points.geometry.x,
        "centroid_lat": centroid_points.geometry.y,
    }
)

map_data = grid_wgs[
    [
        GRID_COL,
        "geometry",
    ]
].merge(
    grid_results,
    on=GRID_COL,
    how="left",
    validate="one_to_one",
)

map_data = map_data.merge(
    centroid_lookup,
    on=GRID_COL,
    how="left",
    validate="one_to_one",
)

map_data = gpd.GeoDataFrame(
    map_data,
    geometry="geometry",
    crs=grid_wgs.crs,
)

if map_data[
    "sen_slope_c_per_year"
].isna().any():
    raise RuntimeError(
        "Trend-to-geometry join created missing results"
    )

map_data.to_file(
    MAP_GEOJSON_PATH,
    driver="GeoJSON",
)

print(
    f"Map rows      : {len(map_data):,}"
)

print(
    f"Map CRS       : {map_data.crs}"
)

print(
    f"Saved GeoJSON : {MAP_GEOJSON_PATH}"
)


# =============================================================================
# 7. FIGURES
# =============================================================================

section(
    "7. SAVE TREND FIGURES"
)

fig, ax = plt.subplots(
    figsize=(
        9,
        8,
    )
)

map_data.plot(
    column="sen_slope_c_per_year",
    legend=True,
    ax=ax,
)

ax.set_title(
    "2015-2025 Sen's Slope of LST by Grid"
)

ax.set_axis_off()

plt.tight_layout()

plt.savefig(
    SLOPE_MAP_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


fig, ax = plt.subplots(
    figsize=(
        9,
        8,
    )
)

map_data.plot(
    column="trend_fdr_alpha_0_05",
    categorical=True,
    legend=True,
    ax=ax,
)

ax.set_title(
    "2015-2025 Mann-Kendall Trend Class by Grid (FDR 0.05)"
)

ax.set_axis_off()

plt.tight_layout()

plt.savefig(
    TREND_MAP_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


fig, ax = plt.subplots(
    figsize=(
        8,
        5,
    )
)

ax.hist(
    grid_results[
        "sen_slope_c_per_year"
    ],
    bins=50,
)

ax.axvline(
    0,
    linewidth=1,
)

ax.set_xlabel(
    "Sen's slope (°C/year)"
)

ax.set_ylabel(
    "Grid count"
)

ax.set_title(
    "Distribution of 2015-2025 Grid-Level Sen's Slopes"
)

ax.grid(
    axis="y",
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    SLOPE_HIST_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


plot_counts = trend_counts.sort_values(
    "trend_class"
)

fig, ax = plt.subplots(
    figsize=(
        8,
        5,
    )
)

ax.bar(
    plot_counts[
        "trend_class"
    ],
    plot_counts[
        "grid_count"
    ],
)

ax.set_ylabel(
    "Grid count"
)

ax.set_title(
    "FDR-Controlled Mann-Kendall Trend Classes"
)

ax.tick_params(
    axis="x",
    rotation=20,
)

ax.grid(
    axis="y",
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    TREND_COUNT_FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

for path in [
    SLOPE_MAP_PATH,
    TREND_MAP_PATH,
    SLOPE_HIST_PATH,
    TREND_COUNT_FIG_PATH,
]:
    print(
        f"Saved: {path}"
    )


# =============================================================================
# 8. METHODOLOGY MANIFEST
# =============================================================================

section(
    "8. SAVE TREND METHODOLOGY"
)

methodology = {
    "step": 23,
    "analysis": (
        "Grid-level historical observed LST trend analysis"
    ),
    "period": "2015-2025",
    "grid_count": EXPECTED_GRIDS,
    "observations_per_grid": EXPECTED_OBS_PER_GRID,
    "target": TARGET,
    "mann_kendall": {
        "type": "original",
        "alternative": "two-sided",
        "variance": "tie-corrected",
        "alpha": ALPHA,
    },
    "sen_slope": {
        "unit": "degrees Celsius per year",
        "confidence_interval": "95%",
    },
    "multiple_testing": {
        "method": "Benjamini-Hochberg FDR",
        "family": (
            "all 2,194 primary grid-level Mann-Kendall tests"
        ),
        "fdr_alpha": FDR_ALPHA,
    },
    "2017_policy": {
        "primary": (
            "2017 retained exactly as observed"
        ),
        "secondary_sensitivity": (
            "repeat grid trend analysis with 2017 omitted"
        ),
        "secondary_role": (
            "diagnostic only; does not replace primary result"
        ),
    },
    "forecast_model_used": False,
    "forecast_predictions_used": False,
    "five_trajectory_classes_assigned_here": False,
    "next_note": (
        "Persistent Hot / Intensifying / Emerging / Diminishing / "
        "Persistent Cool should be defined after reviewing the objective "
        "trend results and using a predeclared heat-status rule."
    ),
}

METHOD_PATH.write_text(
    json.dumps(
        methodology,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        methodology,
        indent=2,
    )
)


# =============================================================================
# 9. STRICT AUDIT
# =============================================================================

section(
    "9. STRICT STEP-23 AUDIT"
)

same_grid_sets = bool(
    set(
        data[
            GRID_COL
        ].astype(
            str
        )
    )
    == set(
        grid_results[
            GRID_COL
        ].astype(
            str
        )
    )
    == set(
        map_data[
            GRID_COL
        ].astype(
            str
        )
    )
)

primary_stats_finite = bool(
    np.isfinite(
        grid_results[
            [
                "mk_s",
                "mk_var_s",
                "mk_z",
                "mk_p_value",
                "mk_tau",
                "sen_slope_c_per_year",
                "sen_intercept_c",
                "sen_slope_ci95_low",
                "sen_slope_ci95_high",
                "mk_fdr_q_value",
            ]
        ].to_numpy(
            dtype=float
        )
    ).all()
)

q_values_valid = bool(
    grid_results[
        "mk_fdr_q_value"
    ].between(
        0.0,
        1.0,
        inclusive="both",
    ).all()
)

audit_rows = [
    {
        "check": "historical_input_hash",
        "status": "PASS",
        "details": input_hash,
    },
    {
        "check": "row_count",
        "status": (
            "PASS"
            if len(
                data
            )
            == EXPECTED_ROWS
            else "FAIL"
        ),
        "details": (
            f"actual={len(data)}, expected={EXPECTED_ROWS}"
        ),
    },
    {
        "check": "unique_grid_count",
        "status": (
            "PASS"
            if data[
                GRID_COL
            ].nunique()
            == EXPECTED_GRIDS
            else "FAIL"
        ),
        "details": (
            f"actual={data[GRID_COL].nunique()}, "
            f"expected={EXPECTED_GRIDS}"
        ),
    },
    {
        "check": "complete_year_range",
        "status": (
            "PASS"
            if years
            == EXPECTED_YEARS
            else "FAIL"
        ),
        "details": str(
            years
        ),
    },
    {
        "check": "complete_11_year_series_per_grid",
        "status": (
            "PASS"
            if (
                counts
                == EXPECTED_OBS_PER_GRID
            ).all()
            else "FAIL"
        ),
        "details": (
            f"min={counts.min()}, max={counts.max()}"
        ),
    },
    {
        "check": "2017_preserved_primary",
        "status": (
            "PASS"
            if rows_2017
            == EXPECTED_GRIDS
            else "FAIL"
        ),
        "details": (
            f"2017 rows={rows_2017}"
        ),
    },
    {
        "check": "primary_grid_result_count",
        "status": (
            "PASS"
            if len(
                grid_results
            )
            == EXPECTED_GRIDS
            else "FAIL"
        ),
        "details": (
            f"actual={len(grid_results)}, expected={EXPECTED_GRIDS}"
        ),
    },
    {
        "check": "primary_statistics_finite",
        "status": (
            "PASS"
            if primary_stats_finite
            else "FAIL"
        ),
        "details": str(
            primary_stats_finite
        ),
    },
    {
        "check": "fdr_q_values_valid",
        "status": (
            "PASS"
            if q_values_valid
            else "FAIL"
        ),
        "details": str(
            q_values_valid
        ),
    },
    {
        "check": "multiple_testing_control",
        "status": "PASS",
        "details": (
            "Benjamini-Hochberg FDR applied across all 2,194 primary grid tests."
        ),
    },
    {
        "check": "2017_sensitivity_secondary_only",
        "status": "PASS",
        "details": (
            "No-2017 analysis saved separately and does not replace primary result."
        ),
    },
    {
        "check": "geometry_alignment",
        "status": (
            "PASS"
            if same_grid_sets
            else "FAIL"
        ),
        "details": (
            "Data, trend table, and map have identical grid-ID sets."
        ),
    },
    {
        "check": "map_rows",
        "status": (
            "PASS"
            if len(
                map_data
            )
            == EXPECTED_GRIDS
            else "FAIL"
        ),
        "details": (
            f"actual={len(map_data)}, expected={EXPECTED_GRIDS}"
        ),
    },
    {
        "check": "forecast_predictions_used",
        "status": "PASS",
        "details": (
            "No. Step 23 uses observed historical LST_C only."
        ),
    },
    {
        "check": "forecast_model_modified",
        "status": "PASS",
        "details": (
            "No. Step 23 is independent of forecasting-model training."
        ),
    },
]

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
)

print(
    audit_df.to_string(
        index=False
    )
)

if (
    audit_df[
        "status"
    ]
    != "PASS"
).any():
    raise RuntimeError(
        "One or more Step-23 audit checks failed"
    )


# =============================================================================
# 10. REPORT
# =============================================================================

section(
    "10. SAVE STEP-23 REPORT"
)

slope_stats = grid_results[
    "sen_slope_c_per_year"
].describe()

report_lines = [
    "STEP 23 - GRID-LEVEL MANN-KENDALL TREND TEST + SEN'S SLOPE",
    "=" * 116,
    "",
    "PRIMARY ANALYSIS",
    "-" * 116,
    "Observed LST period: 2015-2025",
    f"Grid cells: {EXPECTED_GRIDS:,}",
    "Observations per grid: 11 annual values",
    "2017: preserved in primary analysis",
    "Trend test: original two-sided Mann-Kendall with tie correction",
    "Magnitude: Sen's slope (°C/year), 95% CI",
    "Multiple testing: Benjamini-Hochberg FDR at q<0.05",
    "",
    "PRIMARY GRID RESULTS",
    "-" * 116,
    f"Raw p<0.05 significant grids: {raw_sig_count:,}",
    f"FDR q<0.05 significant grids: {fdr_sig_count:,}",
    "",
    trend_counts.to_string(
        index=False
    ),
    "",
    "SEN'S SLOPE DISTRIBUTION",
    "-" * 116,
    slope_stats.to_string(),
    "",
    "CITYWIDE TREND",
    "-" * 116,
    city_df.to_string(
        index=False
    ),
    "",
    "2017 SENSITIVITY",
    "-" * 116,
    sensitivity_summary.to_string(
        index=False
    ),
    "",
    "INTERPRETATION POLICY",
    "-" * 116,
    (
        "FDR-controlled trend class is the primary significance layer. "
        "Sen's slope gives direction and magnitude of historical change."
    ),
    (
        "The no-2017 result is sensitivity evidence only and must not "
        "silently replace the approved primary 2015-2025 analysis."
    ),
    (
        "The five heat-trajectory labels are intentionally not assigned "
        "inside Step 23. They should be defined next using these objective "
        "trend statistics plus a predeclared heat-status rule."
    ),
]

REPORT_PATH.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)

print(
    f"Saved: {REPORT_PATH}"
)

print(
    f"Saved: {AUDIT_PATH}"
)


# =============================================================================
# 11. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "11. REPRODUCIBILITY HASHES"
)

hash_files = [
    DATA_PATH,
    GRID_UTM_PATH,
    GRID_WGS84_PATH,
    GRID_RESULTS_PATH,
    TREND_COUNTS_PATH,
    AREA_SUMMARY_PATH,
    CITY_ANNUAL_PATH,
    CITY_TREND_PATH,
    SENSITIVITY_PATH,
    SENSITIVITY_SUMMARY_PATH,
    MAP_GEOJSON_PATH,
    METHOD_PATH,
    AUDIT_PATH,
    REPORT_PATH,
    SLOPE_MAP_PATH,
    TREND_MAP_PATH,
    SLOPE_HIST_PATH,
    TREND_COUNT_FIG_PATH,
]

hash_rows = []

for path in hash_files:
    digest = sha256_file(
        path
    )

    hash_rows.append(
        {
            "file": str(
                path
            ),
            "sha256": digest,
        }
    )

    print(
        f"{path.name}: {digest}"
    )

pd.DataFrame(
    hash_rows
).to_csv(
    HASH_PATH,
    index=False,
)

print(
    f"Saved: {HASH_PATH}"
)


# =============================================================================
# FINAL STATUS
# =============================================================================

section(
    "STEP 23 COMPLETED SUCCESSFULLY"
)

print(
    "Grid-level Mann-Kendall + Sen's slope analysis completed.\n"
    "Primary analysis used observed LST_C for all years 2015-2025.\n"
    "2017 was preserved exactly as approved.\n"
    "Benjamini-Hochberg FDR was applied across all 2,194 grid tests.\n"
    "A separate no-2017 sensitivity analysis was saved.\n"
    "Map-ready trend output with centroid longitude/latitude was created.\n"
    "No forecasting model or prediction values were used or modified.\n"
    "\nNEXT: review Step-23 results, then define the five heat-trajectory "
    "classes using the trend evidence plus a predefined heat-status rule."
)
