"""
STEP 29A V3S - 2017 ANOMALY SENSITIVITY ANALYSIS
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Step 29A V3 estimated a dedicated historical green-cover response using:

    LST_C_it
      = beta_green * green_mask_it
      + beta_1 * NDBI_it
      + beta_2 * NDWI_it
      + beta_3 * Albedo_it
      + beta_4 * NightLights_it
      + grid fixed effects
      + year fixed effects
      + error_it

with grid-clustered standard errors.

V3 produced a positive green-cover coefficient. Because 2017 is a known
anomalous year in the dataset, V3S asks a strict sensitivity question:

    "Is the V3 green-cover coefficient materially driven by 2017?"

IMPORTANT
---------
- 2017 is NOT deleted from the main research dataset.
- V3 remains the primary all-year historical estimate.
- V3S is a sensitivity analysis only.
- No sign is forced.
- No coefficient is selected because it gives a preferred result.
- Step-28 future forecasts remain unchanged.
- Step-22B LightGBM remains unchanged.

V3S analyses:
1) Reproduce the full 2015-2025 V3 TWFE coefficient.
2) Re-fit the exact same TWFE specification excluding only year 2017.
3) Leave-one-year-out TWFE for every year 2015-2025.
4) First-difference sensitivity using genuine adjacent-year differences:
      FULL
      exclude 2016->2017 transition
      exclude 2017->2018 transition
      exclude BOTH transitions touching 2017
5) Compare future +5/+10/+20 pp response estimates under:
      V3 full-history beta
      V3S no-2017 beta

Interpretation:
This sensitivity evaluates robustness to the anomalous 2017 observation year.
It does NOT convert observational evidence into causal proof.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET = "LST_C"
TREATMENT = "green_mask"

ALL_YEARS = list(range(2015, 2026))
ANOMALY_YEAR = 2017
NO_2017_YEARS = [year for year in ALL_YEARS if year != ANOMALY_YEAR]

EXPECTED_GRIDS = 2194
EXPECTED_FULL_ROWS = EXPECTED_GRIDS * len(ALL_YEARS)
EXPECTED_NO2017_ROWS = EXPECTED_GRIDS * len(NO_2017_YEARS)

CONTROLS = [
    "NDBI",
    "NDWI",
    "Albedo",
    "NightLights",
]

SCENARIO_INCREASE_PP = [5, 10, 20]
GREEN_MIN = 0.0
GREEN_MAX = 1.0

NUMERIC_TOL = 1e-12

EXPECTED_HASHES = {
    "kaduwela_preprocessed_base.csv":
        "0e030b2ec0b5c4080cd4226bec798189a178aa45efd9962d19f38dc9cb170f46",
    "29a_v3_twfe_primary_coefficients.csv":
        "c6813b6c3902f34b679368ee3a1daa15dd4c5270a24811ffc3436daea39eedb3",
    "29a_v3_response_model_audit.csv":
        "5b02db7471f2245f99b6cd90efaaa167e6ed19106b973386297f2b3a822125fd",
    "28_future_model_features_2026_2030.csv":
        "d1e68a2e5d063f231d7a4d379e2534005ad5873f3a6017d3c4f28de9b8acba38",
    "28_future_grid_lst_predictions_2026_2030.csv":
        "55586971042bbce04845a4a33ae724f51b8417fce9bafc2c15bc0a5e75d7ff3e",
}

V3_REPORTED_BETA = 1.971988411609
V3_BETA_REPRO_TOL = 1e-9


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_PATH = (
    ROOT / "data" / "processed"
    / "kaduwela_preprocessed_base.csv"
)

V3_COEFFICIENT_PATH = (
    ROOT / "outputs" / "reports"
    / "29a_v3_twfe_primary_coefficients.csv"
)

V3_AUDIT_PATH = (
    ROOT / "outputs" / "reports"
    / "29a_v3_response_model_audit.csv"
)

FUTURE_FEATURE_PATH = (
    ROOT / "outputs" / "forecasting"
    / "28_future_model_features_2026_2030.csv"
)

FUTURE_PRED_PATH = (
    ROOT / "outputs" / "forecasting"
    / "28_future_grid_lst_predictions_2026_2030.csv"
)

REPORT_DIR = ROOT / "outputs" / "reports"
SCENARIO_DIR = ROOT / "outputs" / "scenarios"
FIGURE_DIR = ROOT / "outputs" / "figures"

for folder in [
    REPORT_DIR,
    SCENARIO_DIR,
    FIGURE_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

TWFE_COMPARISON_PATH = (
    REPORT_DIR
    / "29a_v3s_2017_exclusion_twfe_comparison.csv"
)

LOYO_PATH = (
    REPORT_DIR
    / "29a_v3s_leave_one_year_out_twfe.csv"
)

FD_COMPARISON_PATH = (
    REPORT_DIR
    / "29a_v3s_first_difference_comparison.csv"
)

ANOMALY_CONTEXT_PATH = (
    REPORT_DIR
    / "29a_v3s_2017_anomaly_context.csv"
)

SCENARIO_COMPARISON_PATH = (
    SCENARIO_DIR
    / "29a_v3s_green_response_scenario_comparison_2026_2030.csv"
)

SCENARIO_SUMMARY_PATH = (
    SCENARIO_DIR
    / "29a_v3s_green_response_scenario_comparison_summary.csv"
)

AUDIT_PATH = (
    REPORT_DIR
    / "29a_v3s_2017_sensitivity_audit.csv"
)

METHOD_PATH = (
    REPORT_DIR
    / "29a_v3s_2017_sensitivity_methodology.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "29a_v3s_2017_sensitivity_report.txt"
)

HASH_PATH = (
    REPORT_DIR
    / "29a_v3s_2017_sensitivity_hashes.csv"
)

COEFFICIENT_FIGURE_PATH = (
    FIGURE_DIR
    / "29a_v3s_twfe_green_coefficient_comparison.png"
)

LOYO_FIGURE_PATH = (
    FIGURE_DIR
    / "29a_v3s_leave_one_year_out_green_beta.png"
)

FD_FIGURE_PATH = (
    FIGURE_DIR
    / "29a_v3s_first_difference_green_beta.png"
)


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 132)
    print(title)
    print("=" * 132)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def verify_hash(path, expected, label):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual  : {actual}"
        )

    print(
        f"PASS hash: {label} = {actual}"
    )

    return actual


def require_all_pass(path, label):
    audit = pd.read_csv(path)

    if "status" not in audit.columns:
        raise RuntimeError(
            f"{label} has no 'status' column."
        )

    bad = audit[
        audit["status"].astype(str).str.upper()
        != "PASS"
    ]

    if not bad.empty:
        print(
            bad.to_string(index=False)
        )

        raise RuntimeError(
            f"{label} is not fully PASS."
        )


def check_balanced_panel(df, years):
    expected_rows = (
        EXPECTED_GRIDS
        * len(years)
    )

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Panel rows={len(df)}, expected={expected_rows}"
        )

    if df[
        GRID_COL
    ].nunique() != EXPECTED_GRIDS:
        raise RuntimeError(
            f"Unique grids={df[GRID_COL].nunique()}, "
            f"expected={EXPECTED_GRIDS}"
        )

    counts = (
        df.groupby(
            GRID_COL
        )[
            YEAR_COL
        ]
        .nunique()
    )

    if not (
        counts
        == len(years)
    ).all():
        raise RuntimeError(
            "Panel is not balanced for requested year set."
        )

    expected_years = sorted(
        years
    )

    actual_years = sorted(
        df[
            YEAR_COL
        ]
        .unique()
        .tolist()
    )

    if actual_years != expected_years:
        raise RuntimeError(
            f"Panel years={actual_years}, expected={expected_years}"
        )


def two_way_demean(df, columns):
    """
    Exact two-way demeaning for a balanced entity x time panel.
    """

    transformed = pd.DataFrame(
        index=df.index
    )

    for column in columns:
        values = pd.to_numeric(
            df[
                column
            ],
            errors="raise",
        ).astype(
            float
        )

        entity_mean = (
            values.groupby(
                df[
                    GRID_COL
                ]
            )
            .transform(
                "mean"
            )
        )

        year_mean = (
            values.groupby(
                df[
                    YEAR_COL
                ]
            )
            .transform(
                "mean"
            )
        )

        grand_mean = float(
            values.mean()
        )

        transformed[
            column
        ] = (
            values
            - entity_mean
            - year_mean
            + grand_mean
        )

    return transformed


def fit_twfe(panel, years, model_name):
    sub = (
        panel[
            panel[
                YEAR_COL
            ].isin(
                years
            )
        ]
        .copy()
        .sort_values(
            [
                GRID_COL,
                YEAR_COL,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    check_balanced_panel(
        sub,
        years,
    )

    model_columns = [
        TARGET,
        TREATMENT,
        *CONTROLS,
    ]

    transformed = two_way_demean(
        sub,
        model_columns,
    )

    y = transformed[
        TARGET
    ].to_numpy(
        dtype=float
    )

    X = transformed[
        [
            TREATMENT,
            *CONTROLS,
        ]
    ].copy()

    variances = X.var(
        ddof=0
    )

    bad_variance = variances[
        variances
        <= NUMERIC_TOL
    ]

    if not bad_variance.empty:
        raise RuntimeError(
            f"{model_name}: zero/near-zero within variance:\n"
            + bad_variance.to_string()
        )

    rank = int(
        np.linalg.matrix_rank(
            X.to_numpy(
                dtype=float
            )
        )
    )

    if rank != X.shape[1]:
        raise RuntimeError(
            f"{model_name}: rank={rank}, predictors={X.shape[1]}"
        )

    fit = sm.OLS(
        y,
        X,
        hasconst=False,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                sub[
                    GRID_COL
                ].astype(
                    str
                ).to_numpy(),
            "use_correction":
                True,
        },
    )

    ci = fit.conf_int(
        alpha=0.05
    ).loc[
        TREATMENT
    ]

    beta = float(
        fit.params[
            TREATMENT
        ]
    )

    lower = float(
        ci[
            0
        ]
    )

    upper = float(
        ci[
            1
        ]
    )

    return {
        "model":
            model_name,
        "years":
            years,
        "rows":
            int(
                fit.nobs
            ),
        "grids":
            int(
                sub[
                    GRID_COL
                ].nunique()
            ),
        "beta":
            beta,
        "beta_per_10pp":
            beta
            * 0.10,
        "se":
            float(
                fit.bse[
                    TREATMENT
                ]
            ),
        "p_value":
            float(
                fit.pvalues[
                    TREATMENT
                ]
            ),
        "ci_lower":
            lower,
        "ci_upper":
            upper,
        "ci_lower_per_10pp":
            lower
            * 0.10,
        "ci_upper_per_10pp":
            upper
            * 0.10,
        "within_r2":
            float(
                fit.rsquared
            ),
        "fit":
            fit,
    }


def build_adjacent_differences(panel):
    """
    Build true adjacent-year differences BEFORE any sensitivity rows are removed.

    This prevents a false 2016->2018 "one-year" difference after excluding 2017.
    """

    sub = (
        panel
        .sort_values(
            [
                GRID_COL,
                YEAR_COL,
            ]
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    variables = [
        TARGET,
        TREATMENT,
        *CONTROLS,
    ]

    sub[
        "previous_year"
    ] = (
        sub.groupby(
            GRID_COL
        )[
            YEAR_COL
        ]
        .shift(
            1
        )
    )

    for column in variables:
        sub[
            f"d_{column}"
        ] = (
            sub.groupby(
                GRID_COL
            )[
                column
            ]
            .diff()
        )

    fd = (
        sub[
            sub[
                YEAR_COL
            ].between(
                2016,
                2025,
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if not (
        fd[
            YEAR_COL
        ]
        - fd[
            "previous_year"
        ]
        == 1
    ).all():
        raise RuntimeError(
            "Non-adjacent differences detected."
        )

    expected_rows = (
        EXPECTED_GRIDS
        * 10
    )

    if len(fd) != expected_rows:
        raise RuntimeError(
            f"Adjacent-difference rows={len(fd)}, expected={expected_rows}"
        )

    return fd


def fit_fd(fd, excluded_target_years, model_name):
    """
    First-difference model:
      d_LST ~ d_green + d_controls + target-year FE

    excluded_target_years are transition END years.
    Example:
      target year 2017 = 2016->2017 transition
      target year 2018 = 2017->2018 transition
    """

    use = (
        fd[
            ~fd[
                YEAR_COL
            ].isin(
                excluded_target_years
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    y = use[
        f"d_{TARGET}"
    ].astype(
        float
    )

    X_core = use[
        [
            f"d_{TREATMENT}",
            *[
                f"d_{column}"
                for column in CONTROLS
            ],
        ]
    ].astype(
        float
    )

    year_dummies = pd.get_dummies(
        use[
            YEAR_COL
        ].astype(
            int
        ),
        prefix="target_year",
        drop_first=True,
        dtype=float,
    )

    X = pd.concat(
        [
            X_core.reset_index(
                drop=True
            ),
            year_dummies.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    X = sm.add_constant(
        X,
        has_constant="add",
    )

    fit = sm.OLS(
        y.to_numpy(
            dtype=float
        ),
        X,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                use[
                    GRID_COL
                ].astype(
                    str
                ).to_numpy(),
            "use_correction":
                True,
        },
    )

    term = (
        f"d_{TREATMENT}"
    )

    ci = fit.conf_int(
        alpha=0.05
    ).loc[
        term
    ]

    beta = float(
        fit.params[
            term
        ]
    )

    lower = float(
        ci[
            0
        ]
    )

    upper = float(
        ci[
            1
        ]
    )

    return {
        "model":
            model_name,
        "excluded_target_years":
            excluded_target_years,
        "rows":
            int(
                fit.nobs
            ),
        "beta":
            beta,
        "beta_per_10pp":
            beta
            * 0.10,
        "se":
            float(
                fit.bse[
                    term
                ]
            ),
        "p_value":
            float(
                fit.pvalues[
                    term
                ]
            ),
        "ci_lower":
            lower,
        "ci_upper":
            upper,
        "ci_lower_per_10pp":
            lower
            * 0.10,
        "ci_upper_per_10pp":
            upper
            * 0.10,
        "fit":
            fit,
    }


def sign_label(beta):
    if beta > 0:
        return "POSITIVE"
    if beta < 0:
        return "NEGATIVE"
    return "ZERO"


def ci_zero_label(lower, upper):
    if lower > 0:
        return "ENTIRELY_POSITIVE"
    if upper < 0:
        return "ENTIRELY_NEGATIVE"
    return "INCLUDES_ZERO"


# =============================================================================
# 0. VERIFY INPUTS + V3 OUTPUTS
# =============================================================================

section(
    "STEP 29A V3S - 2017 ANOMALY SENSITIVITY ANALYSIS"
)

verify_hash(
    BASE_PATH,
    EXPECTED_HASHES[
        "kaduwela_preprocessed_base.csv"
    ],
    "Canonical historical base data",
)

verify_hash(
    V3_COEFFICIENT_PATH,
    EXPECTED_HASHES[
        "29a_v3_twfe_primary_coefficients.csv"
    ],
    "Step-29A V3 primary coefficient table",
)

verify_hash(
    V3_AUDIT_PATH,
    EXPECTED_HASHES[
        "29a_v3_response_model_audit.csv"
    ],
    "Step-29A V3 strict audit",
)

verify_hash(
    FUTURE_FEATURE_PATH,
    EXPECTED_HASHES[
        "28_future_model_features_2026_2030.csv"
    ],
    "Step-28 future feature matrix",
)

verify_hash(
    FUTURE_PRED_PATH,
    EXPECTED_HASHES[
        "28_future_grid_lst_predictions_2026_2030.csv"
    ],
    "Step-28 future LST predictions",
)

require_all_pass(
    V3_AUDIT_PATH,
    "Step-29A V3 audit",
)

print(
    "\nPASS: Step-29A V3 audit is fully PASS."
)

print(
    "2017 will be excluded only in sensitivity estimates; "
    "the original data and V3 primary result remain unchanged."
)


# =============================================================================
# 1. LOAD HISTORICAL PANEL
# =============================================================================

section(
    "1. LOAD 2015-2025 HISTORICAL PANEL"
)

base = pd.read_csv(
    BASE_PATH,
    dtype={
        GRID_COL:
            str,
        "area_name":
            str,
    },
)

base[
    YEAR_COL
] = pd.to_numeric(
    base[
        YEAR_COL
    ],
    errors="raise",
).astype(
    int
)

required_columns = [
    GRID_COL,
    YEAR_COL,
    TARGET,
    TREATMENT,
    *CONTROLS,
]

missing = [
    column
    for column in required_columns
    if column not in base.columns
]

if missing:
    raise RuntimeError(
        "Missing V3S columns:\n"
        + str(
            missing
        )
    )

panel = (
    base[
        base[
            YEAR_COL
        ].isin(
            ALL_YEARS
        )
    ][
        required_columns
    ]
    .copy()
    .sort_values(
        [
            GRID_COL,
            YEAR_COL,
        ]
    )
    .reset_index(
        drop=True
    )
)

for column in [
    TARGET,
    TREATMENT,
    *CONTROLS,
]:
    panel[
        column
    ] = pd.to_numeric(
        panel[
            column
        ],
        errors="raise",
    )

if panel[
    [
        TARGET,
        TREATMENT,
        *CONTROLS,
    ]
].isna().any().any():
    raise RuntimeError(
        "Historical panel contains missing model values."
    )

if not np.isfinite(
    panel[
        [
            TARGET,
            TREATMENT,
            *CONTROLS,
        ]
    ].to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "Historical panel contains non-finite values."
    )

check_balanced_panel(
    panel,
    ALL_YEARS,
)

print(
    f"Rows         : {len(panel):,}"
)

print(
    f"Unique grids : {panel[GRID_COL].nunique():,}"
)

print(
    f"Years        : {ALL_YEARS}"
)


# =============================================================================
# 2. DESCRIPTIVE 2017 ANOMALY CONTEXT
# =============================================================================

section(
    "2. DESCRIPTIVE 2017 ANOMALY CONTEXT"
)

annual_context = (
    panel.groupby(
        YEAR_COL,
        as_index=False,
    )
    .agg(
        mean_lst_c=(
            TARGET,
            "mean",
        ),
        mean_green_mask=(
            TREATMENT,
            "mean",
        ),
        mean_ndbi=(
            "NDBI",
            "mean",
        ),
        mean_ndwi=(
            "NDWI",
            "mean",
        ),
        mean_albedo=(
            "Albedo",
            "mean",
        ),
        mean_nightlights=(
            "NightLights",
            "mean",
        ),
    )
)

annual_context[
    "delta_mean_lst_c"
] = annual_context[
    "mean_lst_c"
].diff()

annual_context[
    "delta_mean_green_mask"
] = annual_context[
    "mean_green_mask"
].diff()

annual_context.to_csv(
    ANOMALY_CONTEXT_PATH,
    index=False,
)

print(
    annual_context[
        annual_context[
            YEAR_COL
        ].isin(
            [
                2016,
                2017,
                2018,
            ]
        )
    ].to_string(
        index=False
    )
)


# =============================================================================
# 3. REPRODUCE V3 FULL TWFE + EXCLUDE 2017
# =============================================================================

section(
    "3. FULL V3 TWFE VS 2017-EXCLUDED TWFE"
)

full = fit_twfe(
    panel=panel,
    years=ALL_YEARS,
    model_name=
        "TWFE_FULL_2015_2025",
)

no2017 = fit_twfe(
    panel=panel,
    years=NO_2017_YEARS,
    model_name=
        "TWFE_EXCLUDE_2017",
)

if abs(
    full[
        "beta"
    ]
    - V3_REPORTED_BETA
) > V3_BETA_REPRO_TOL:
    raise RuntimeError(
        "Full-history V3 beta was not reproduced.\n"
        f"Expected≈{V3_REPORTED_BETA:.12f}\n"
        f"Actual  ={full['beta']:.12f}"
    )

absolute_beta_change = (
    no2017[
        "beta"
    ]
    - full[
        "beta"
    ]
)

relative_beta_change_pct = (
    absolute_beta_change
    / abs(
        full[
            "beta"
        ]
    )
    * 100.0
)

sign_changed = (
    np.sign(
        no2017[
            "beta"
        ]
    )
    != np.sign(
        full[
            "beta"
        ]
    )
)

twfe_rows = []

for result in [
    full,
    no2017,
]:
    twfe_rows.append(
        {
            "model":
                result[
                    "model"
                ],
            "years":
                ",".join(
                    map(
                        str,
                        result[
                            "years"
                        ],
                    )
                ),
            "rows":
                result[
                    "rows"
                ],
            "grids":
                result[
                    "grids"
                ],
            "green_beta_c_per_fraction":
                result[
                    "beta"
                ],
            "green_beta_c_per_10pp":
                result[
                    "beta_per_10pp"
                ],
            "clustered_se":
                result[
                    "se"
                ],
            "p_value":
                result[
                    "p_value"
                ],
            "ci95_lower_per_10pp":
                result[
                    "ci_lower_per_10pp"
                ],
            "ci95_upper_per_10pp":
                result[
                    "ci_upper_per_10pp"
                ],
            "coefficient_sign":
                sign_label(
                    result[
                        "beta"
                    ]
                ),
            "ci_position":
                ci_zero_label(
                    result[
                        "ci_lower"
                    ],
                    result[
                        "ci_upper"
                    ],
                ),
            "within_r2":
                result[
                    "within_r2"
                ],
        }
    )

twfe_comparison = pd.DataFrame(
    twfe_rows
)

twfe_comparison.to_csv(
    TWFE_COMPARISON_PATH,
    index=False,
)

print(
    twfe_comparison.to_string(
        index=False
    )
)

print(
    "\n2017-exclusion effect on green coefficient:"
)

print(
    f"  Full beta per +10 pp      : {full['beta_per_10pp']:+.6f} °C"
)

print(
    f"  No-2017 beta per +10 pp   : {no2017['beta_per_10pp']:+.6f} °C"
)

print(
    f"  Absolute beta change       : {absolute_beta_change:+.6f} °C per 1.0 green fraction"
)

print(
    f"  Relative beta change       : {relative_beta_change_pct:+.2f}%"
)

print(
    f"  Sign changed?              : {sign_changed}"
)


# =============================================================================
# 4. LEAVE-ONE-YEAR-OUT TWFE
# =============================================================================

section(
    "4. LEAVE-ONE-YEAR-OUT TWFE INFLUENCE CHECK"
)

loyo_rows = []

for omitted_year in ALL_YEARS:
    years = [
        year
        for year in ALL_YEARS
        if year != omitted_year
    ]

    result = fit_twfe(
        panel=panel,
        years=years,
        model_name=
            f"TWFE_OMIT_{omitted_year}",
    )

    loyo_rows.append(
        {
            "omitted_year":
                omitted_year,
            "rows":
                result[
                    "rows"
                ],
            "green_beta_c_per_fraction":
                result[
                    "beta"
                ],
            "green_beta_c_per_10pp":
                result[
                    "beta_per_10pp"
                ],
            "clustered_se":
                result[
                    "se"
                ],
            "p_value":
                result[
                    "p_value"
                ],
            "ci95_lower_per_10pp":
                result[
                    "ci_lower_per_10pp"
                ],
            "ci95_upper_per_10pp":
                result[
                    "ci_upper_per_10pp"
                ],
            "coefficient_sign":
                sign_label(
                    result[
                        "beta"
                    ]
                ),
            "ci_position":
                ci_zero_label(
                    result[
                        "ci_lower"
                    ],
                    result[
                        "ci_upper"
                    ],
                ),
            "absolute_change_from_full_beta":
                result[
                    "beta"
                ]
                - full[
                    "beta"
                ],
            "absolute_change_from_full_per_10pp":
                (
                    result[
                        "beta"
                    ]
                    - full[
                        "beta"
                    ]
                )
                * 0.10,
            "relative_change_from_full_pct":
                (
                    result[
                        "beta"
                    ]
                    - full[
                        "beta"
                    ]
                )
                / abs(
                    full[
                        "beta"
                    ]
                )
                * 100.0,
        }
    )

loyo_df = pd.DataFrame(
    loyo_rows
)

loyo_df.to_csv(
    LOYO_PATH,
    index=False,
)

print(
    loyo_df.to_string(
        index=False
    )
)

largest_influence_row = (
    loyo_df.iloc[
        loyo_df[
            "absolute_change_from_full_beta"
        ]
        .abs()
        .argmax()
    ]
)

largest_influence_year = int(
    largest_influence_row[
        "omitted_year"
    ]
)

print(
    f"\nLargest absolute leave-one-year-out coefficient shift occurs when omitting: "
    f"{largest_influence_year}"
)

print(
    f"2017 is the single most influential omitted year: "
    f"{largest_influence_year == 2017}"
)


# =============================================================================
# 5. FIRST-DIFFERENCE TRANSITION SENSITIVITY
# =============================================================================

section(
    "5. FIRST-DIFFERENCE TRANSITION SENSITIVITY"
)

fd = build_adjacent_differences(
    panel
)

fd_variants = [
    (
        "FD_FULL",
        [],
    ),
    (
        "FD_EXCLUDE_2016_TO_2017",
        [
            2017,
        ],
    ),
    (
        "FD_EXCLUDE_2017_TO_2018",
        [
            2018,
        ],
    ),
    (
        "FD_EXCLUDE_BOTH_TOUCHING_2017",
        [
            2017,
            2018,
        ],
    ),
]

fd_results = []

for model_name, excluded_target_years in fd_variants:
    fd_results.append(
        fit_fd(
            fd=fd,
            excluded_target_years=
                excluded_target_years,
            model_name=
                model_name,
        )
    )

fd_rows = []

for result in fd_results:
    fd_rows.append(
        {
            "model":
                result[
                    "model"
                ],
            "excluded_transition_end_years":
                ",".join(
                    map(
                        str,
                        result[
                            "excluded_target_years"
                        ],
                    )
                )
                if result[
                    "excluded_target_years"
                ]
                else "NONE",
            "rows":
                result[
                    "rows"
                ],
            "green_beta_c_per_fraction":
                result[
                    "beta"
                ],
            "green_beta_c_per_10pp":
                result[
                    "beta_per_10pp"
                ],
            "clustered_se":
                result[
                    "se"
                ],
            "p_value":
                result[
                    "p_value"
                ],
            "ci95_lower_per_10pp":
                result[
                    "ci_lower_per_10pp"
                ],
            "ci95_upper_per_10pp":
                result[
                    "ci_upper_per_10pp"
                ],
            "coefficient_sign":
                sign_label(
                    result[
                        "beta"
                    ]
                ),
            "ci_position":
                ci_zero_label(
                    result[
                        "ci_lower"
                    ],
                    result[
                        "ci_upper"
                    ],
                ),
        }
    )

fd_comparison = pd.DataFrame(
    fd_rows
)

fd_comparison.to_csv(
    FD_COMPARISON_PATH,
    index=False,
)

print(
    fd_comparison.to_string(
        index=False
    )
)

fd_full = fd_results[
    0
]

fd_exclude_both = fd_results[
    3
]

fd_sign_changed_after_removing_2017_transitions = (
    np.sign(
        fd_full[
            "beta"
        ]
    )
    != np.sign(
        fd_exclude_both[
            "beta"
        ]
    )
)

print(
    "\nFirst-difference robustness:"
)

print(
    f"  Full FD per +10 pp                    : {fd_full['beta_per_10pp']:+.6f} °C"
)

print(
    f"  Excluding both transitions touching 2017: "
    f"{fd_exclude_both['beta_per_10pp']:+.6f} °C"
)

print(
    f"  Sign changed?                         : "
    f"{fd_sign_changed_after_removing_2017_transitions}"
)


# =============================================================================
# 6. FUTURE SCENARIO RESPONSE COMPARISON
# =============================================================================

section(
    "6. FUTURE SCENARIO RESPONSE: FULL V3 VS NO-2017 SENSITIVITY"
)

future = pd.read_csv(
    FUTURE_FEATURE_PATH,
    dtype={
        GRID_COL:
            str,
        "area_name":
            str,
    },
)

pred = pd.read_csv(
    FUTURE_PRED_PATH,
    dtype={
        GRID_COL:
            str,
        "area_name":
            str,
    },
)

for frame in [
    future,
    pred,
]:
    frame[
        YEAR_COL
    ] = pd.to_numeric(
        frame[
            YEAR_COL
        ],
        errors="raise",
    ).astype(
        int
    )

future = (
    future
    .sort_values(
        [
            YEAR_COL,
            GRID_COL,
        ]
    )
    .reset_index(
        drop=True
    )
)

pred = (
    pred
    .sort_values(
        [
            YEAR_COL,
            GRID_COL,
        ]
    )
    .reset_index(
        drop=True
    )
)

if not future[
    [
        GRID_COL,
        YEAR_COL,
    ]
].equals(
    pred[
        [
            GRID_COL,
            YEAR_COL,
        ]
    ]
):
    raise RuntimeError(
        "Step-28 future feature/prediction keys do not align."
    )

baseline_green = pd.to_numeric(
    future[
        TREATMENT
    ],
    errors="raise",
).to_numpy(
    dtype=float
)

baseline_lst = pd.to_numeric(
    pred[
        "predicted_lst_c"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)

scenario_rows = []

for pp in SCENARIO_INCREASE_PP:
    requested_fraction = (
        pp
        / 100.0
    )

    scenario_green = np.clip(
        baseline_green
        + requested_fraction,
        GREEN_MIN,
        GREEN_MAX,
    )

    actual_increase = (
        scenario_green
        - baseline_green
    )

    full_response = (
        full[
            "beta"
        ]
        * actual_increase
    )

    no2017_response = (
        no2017[
            "beta"
        ]
        * actual_increase
    )

    for i in range(
        len(
            future
        )
    ):
        scenario_rows.append(
            {
                GRID_COL:
                    str(
                        future.at[
                            i,
                            GRID_COL,
                        ]
                    ),
                YEAR_COL:
                    int(
                        future.at[
                            i,
                            YEAR_COL,
                        ]
                    ),
                "area_name":
                    str(
                        future.at[
                            i,
                            "area_name",
                        ]
                    ),
                "requested_green_increase_pp":
                    pp,
                "baseline_green_mask":
                    float(
                        baseline_green[
                            i
                        ]
                    ),
                "scenario_green_mask":
                    float(
                        scenario_green[
                            i
                        ]
                    ),
                "actual_green_increase_pp":
                    float(
                        actual_increase[
                            i
                        ]
                        * 100.0
                    ),
                "baseline_predicted_lst_c":
                    float(
                        baseline_lst[
                            i
                        ]
                    ),
                "v3_full_estimated_response_c":
                    float(
                        full_response[
                            i
                        ]
                    ),
                "v3_full_cooling_benefit_c":
                    float(
                        -full_response[
                            i
                        ]
                    ),
                "v3s_no2017_estimated_response_c":
                    float(
                        no2017_response[
                            i
                        ]
                    ),
                "v3s_no2017_cooling_benefit_c":
                    float(
                        -no2017_response[
                            i
                        ]
                    ),
                "response_difference_no2017_minus_full_c":
                    float(
                        no2017_response[
                            i
                        ]
                        - full_response[
                            i
                        ]
                    ),
            }
        )

scenario_comparison = pd.DataFrame(
    scenario_rows
)

scenario_comparison.to_csv(
    SCENARIO_COMPARISON_PATH,
    index=False,
)

scenario_summary = (
    scenario_comparison
    .groupby(
        [
            YEAR_COL,
            "requested_green_increase_pp",
        ],
        as_index=False,
    )
    .agg(
        grid_count=(
            GRID_COL,
            "size",
        ),
        mean_actual_green_increase_pp=(
            "actual_green_increase_pp",
            "mean",
        ),
        mean_v3_full_response_c=(
            "v3_full_estimated_response_c",
            "mean",
        ),
        mean_v3_full_cooling_benefit_c=(
            "v3_full_cooling_benefit_c",
            "mean",
        ),
        mean_v3s_no2017_response_c=(
            "v3s_no2017_estimated_response_c",
            "mean",
        ),
        mean_v3s_no2017_cooling_benefit_c=(
            "v3s_no2017_cooling_benefit_c",
            "mean",
        ),
        mean_response_difference_no2017_minus_full_c=(
            "response_difference_no2017_minus_full_c",
            "mean",
        ),
    )
)

scenario_summary.to_csv(
    SCENARIO_SUMMARY_PATH,
    index=False,
)

print(
    scenario_summary.to_string(
        index=False
    )
)


# =============================================================================
# 7. FIGURES
# =============================================================================

section(
    "7. SAVE SENSITIVITY FIGURES"
)

# Full vs no-2017
comparison_plot = twfe_comparison.copy()

x = np.arange(
    len(
        comparison_plot
    )
)

y = comparison_plot[
    "green_beta_c_per_10pp"
].to_numpy(
    dtype=float
)

lower = (
    y
    - comparison_plot[
        "ci95_lower_per_10pp"
    ].to_numpy(
        dtype=float
    )
)

upper = (
    comparison_plot[
        "ci95_upper_per_10pp"
    ].to_numpy(
        dtype=float
    )
    - y
)

fig, ax = plt.subplots(
    figsize=(
        8,
        6,
    )
)

ax.errorbar(
    x,
    y,
    yerr=np.vstack(
        [
            lower,
            upper,
        ]
    ),
    fmt="o",
    capsize=5,
)

ax.axhline(
    0.0,
    linewidth=1.0,
    color="black",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        "Full 2015-2025",
        "Exclude 2017",
    ]
)

ax.set_ylabel(
    "Green response per +10 pp (°C)"
)

ax.set_title(
    "2017 Exclusion Sensitivity — TWFE Green Coefficient"
)

ax.grid(
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    COEFFICIENT_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

# Leave-one-year-out
fig, ax = plt.subplots(
    figsize=(
        10,
        6,
    )
)

ax.plot(
    loyo_df[
        "omitted_year"
    ],
    loyo_df[
        "green_beta_c_per_10pp"
    ],
    marker="o",
)

ax.axhline(
    full[
        "beta_per_10pp"
    ],
    linewidth=1.0,
    color="black",
)

ax.axhline(
    0.0,
    linewidth=1.0,
    color="black",
)

ax.set_xlabel(
    "Omitted year"
)

ax.set_ylabel(
    "Green response per +10 pp (°C)"
)

ax.set_title(
    "Leave-One-Year-Out TWFE Green-Coefficient Influence"
)

ax.grid(
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    LOYO_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

# First-difference variants
fig, ax = plt.subplots(
    figsize=(
        11,
        6,
    )
)

fd_x = np.arange(
    len(
        fd_comparison
    )
)

fd_y = fd_comparison[
    "green_beta_c_per_10pp"
].to_numpy(
    dtype=float
)

fd_lower = (
    fd_y
    - fd_comparison[
        "ci95_lower_per_10pp"
    ].to_numpy(
        dtype=float
    )
)

fd_upper = (
    fd_comparison[
        "ci95_upper_per_10pp"
    ].to_numpy(
        dtype=float
    )
    - fd_y
)

ax.errorbar(
    fd_x,
    fd_y,
    yerr=np.vstack(
        [
            fd_lower,
            fd_upper,
        ]
    ),
    fmt="o",
    capsize=5,
)

ax.axhline(
    0.0,
    linewidth=1.0,
    color="black",
)

ax.set_xticks(
    fd_x
)

ax.set_xticklabels(
    [
        "Full",
        "Drop 2016→2017",
        "Drop 2017→2018",
        "Drop both",
    ],
    rotation=15,
)

ax.set_ylabel(
    "First-difference green response per +10 pp (°C)"
)

ax.set_title(
    "2017 Transition Sensitivity — First-Difference Model"
)

ax.grid(
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    FD_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    f"Saved: {COEFFICIENT_FIGURE_PATH}"
)

print(
    f"Saved: {LOYO_FIGURE_PATH}"
)

print(
    f"Saved: {FD_FIGURE_PATH}"
)


# =============================================================================
# 8. STRICT AUDIT
# =============================================================================

section(
    "8. STRICT STEP-29A V3S AUDIT"
)

no2017_panel = panel[
    panel[
        YEAR_COL
    ]
    != ANOMALY_YEAR
].copy()

audit_rows = [
    {
        "check":
            "historical_base_hash",
        "status":
            "PASS",
        "details":
            EXPECTED_HASHES[
                "kaduwela_preprocessed_base.csv"
            ],
    },
    {
        "check":
            "v3_primary_coefficient_hash",
        "status":
            "PASS",
        "details":
            EXPECTED_HASHES[
                "29a_v3_twfe_primary_coefficients.csv"
            ],
    },
    {
        "check":
            "v3_audit",
        "status":
            "PASS",
        "details":
            "Step-29A V3 strict audit verified fully PASS.",
    },
    {
        "check":
            "full_panel_rows",
        "status":
            "PASS"
            if len(
                panel
            )
            == EXPECTED_FULL_ROWS
            else "FAIL",
        "details":
            f"actual={len(panel)}, expected={EXPECTED_FULL_ROWS}",
    },
    {
        "check":
            "no2017_panel_rows",
        "status":
            "PASS"
            if len(
                no2017_panel
            )
            == EXPECTED_NO2017_ROWS
            else "FAIL",
        "details":
            f"actual={len(no2017_panel)}, expected={EXPECTED_NO2017_ROWS}",
    },
    {
        "check":
            "2017_excluded_only_in_sensitivity",
        "status":
            "PASS"
            if (
                ANOMALY_YEAR
                in panel[
                    YEAR_COL
                ].unique()
                and ANOMALY_YEAR
                not in no2017_panel[
                    YEAR_COL
                ].unique()
            )
            else "FAIL",
        "details":
            "Primary data retains 2017; sensitivity subset omits only 2017.",
    },
    {
        "check":
            "full_v3_beta_reproduced",
        "status":
            "PASS"
            if abs(
                full[
                    "beta"
                ]
                - V3_REPORTED_BETA
            )
            <= V3_BETA_REPRO_TOL
            else "FAIL",
        "details":
            (
                f"actual={full['beta']:.12f}, "
                f"expected≈{V3_REPORTED_BETA:.12f}"
            ),
    },
    {
        "check":
            "same_twfe_specification_full_vs_no2017",
        "status":
            "PASS",
        "details":
            (
                "Same outcome, treatment, controls, two-way FE transformation, "
                "and grid-clustered SE; only year 2017 omitted."
            ),
    },
    {
        "check":
            "leave_one_year_out_count",
        "status":
            "PASS"
            if len(
                loyo_df
            )
            == len(
                ALL_YEARS
            )
            else "FAIL",
        "details":
            f"actual={len(loyo_df)}, expected={len(ALL_YEARS)}",
    },
    {
        "check":
            "fd_differences_are_true_adjacent_years",
        "status":
            "PASS"
            if (
                fd[
                    YEAR_COL
                ]
                - fd[
                    "previous_year"
                ]
                == 1
            ).all()
            else "FAIL",
        "details":
            "Differences constructed before removing 2017-related transitions.",
    },
    {
        "check":
            "fd_full_rows",
        "status":
            "PASS"
            if fd_results[
                0
            ][
                "rows"
            ]
            == EXPECTED_GRIDS
            * 10
            else "FAIL",
        "details":
            f"actual={fd_results[0]['rows']}, expected={EXPECTED_GRIDS * 10}",
    },
    {
        "check":
            "fd_drop_both_2017_transitions_rows",
        "status":
            "PASS"
            if fd_exclude_both[
                "rows"
            ]
            == EXPECTED_GRIDS
            * 8
            else "FAIL",
        "details":
            (
                f"actual={fd_exclude_both['rows']}, "
                f"expected={EXPECTED_GRIDS * 8}"
            ),
    },
    {
        "check":
            "coefficient_sign_forced",
        "status":
            "PASS",
        "details":
            "No. All estimated signs retained.",
    },
    {
        "check":
            "preferred_model_selected_by_sensitivity_result",
        "status":
            "PASS",
        "details":
            "No. V3S reports robustness only and does not auto-select a replacement coefficient.",
    },
    {
        "check":
            "step28_forecasts_modified",
        "status":
            "PASS",
        "details":
            "No. Scenario comparison outputs are separate.",
    },
    {
        "check":
            "scenario_rows",
        "status":
            "PASS"
            if len(
                scenario_comparison
            )
            == len(
                future
            )
            * len(
                SCENARIO_INCREASE_PP
            )
            else "FAIL",
        "details":
            (
                f"actual={len(scenario_comparison)}, "
                f"expected={len(future) * len(SCENARIO_INCREASE_PP)}"
            ),
    },
    {
        "check":
            "causal_claim",
        "status":
            "PASS",
        "details":
            "No. V3S is observational robustness analysis only.",
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
        "One or more Step-29A V3S audit checks failed."
    )


# =============================================================================
# 9. METHODOLOGY + REPORT
# =============================================================================

section(
    "9. SAVE STEP-29A V3S METHODOLOGY + REPORT"
)

methodology = {
    "step":
        "29A V3S",
    "name":
        "2017 anomaly sensitivity analysis for green-cover response",
    "primary_v3_preserved":
        True,
    "anomaly_year":
        ANOMALY_YEAR,
    "twfe_specification":
        {
            "outcome":
                TARGET,
            "treatment":
                TREATMENT,
            "controls":
                CONTROLS,
            "grid_fixed_effects":
                True,
            "year_fixed_effects":
                True,
            "standard_errors":
                "clustered by grid_id",
        },
    "full_twfe":
        {
            "beta_per_10pp_c":
                full[
                    "beta_per_10pp"
                ],
            "ci95_lower_per_10pp_c":
                full[
                    "ci_lower_per_10pp"
                ],
            "ci95_upper_per_10pp_c":
                full[
                    "ci_upper_per_10pp"
                ],
        },
    "no2017_twfe":
        {
            "beta_per_10pp_c":
                no2017[
                    "beta_per_10pp"
                ],
            "ci95_lower_per_10pp_c":
                no2017[
                    "ci_lower_per_10pp"
                ],
            "ci95_upper_per_10pp_c":
                no2017[
                    "ci_upper_per_10pp"
                ],
        },
    "twfe_sensitivity":
        {
            "absolute_beta_change_per_fraction":
                absolute_beta_change,
            "relative_beta_change_pct":
                relative_beta_change_pct,
            "sign_changed":
                bool(
                    sign_changed
                ),
            "largest_leave_one_year_out_influence_year":
                largest_influence_year,
            "is_2017_largest_leave_one_out_influence":
                bool(
                    largest_influence_year
                    == 2017
                ),
        },
    "first_difference_sensitivity":
        {
            "construction":
                (
                    "True adjacent-year differences are created on the full "
                    "2015-2025 sequence before excluding anomaly transitions."
                ),
            "full_beta_per_10pp_c":
                fd_full[
                    "beta_per_10pp"
                ],
            "exclude_both_transitions_touching_2017_beta_per_10pp_c":
                fd_exclude_both[
                    "beta_per_10pp"
                ],
            "sign_changed_after_excluding_both":
                bool(
                    fd_sign_changed_after_removing_2017_transitions
                ),
        },
    "future_scenario_comparison":
        {
            "scenario_levels_pp":
                SCENARIO_INCREASE_PP,
            "purpose":
                (
                    "Compare response magnitude under full V3 coefficient "
                    "versus no-2017 sensitivity coefficient only."
                ),
            "step28_baseline_modified":
                False,
        },
    "interpretation":
        (
            "V3S tests robustness to the anomalous 2017 year. It does not "
            "delete 2017 from the primary research pipeline and does not "
            "automatically replace the V3 primary coefficient."
        ),
    "causal_limit":
        (
            "All response estimates remain observational associations and "
            "are not guaranteed causal effects."
        ),
}

METHOD_PATH.write_text(
    json.dumps(
        methodology,
        indent=2,
    ),
    encoding="utf-8",
)

report_lines = [
    "STEP 29A V3S - 2017 ANOMALY SENSITIVITY ANALYSIS",
    "=" * 118,
    "",
    "PRIMARY QUESTION",
    "-" * 118,
    (
        "Is the positive V3 green-cover coefficient materially driven by "
        "the anomalous 2017 year?"
    ),
    "",
    "2017 CONTEXT",
    "-" * 118,
    annual_context[
        annual_context[
            YEAR_COL
        ].isin(
            [
                2016,
                2017,
                2018,
            ]
        )
    ].to_string(
        index=False
    ),
    "",
    "FULL VS NO-2017 TWFE",
    "-" * 118,
    twfe_comparison.to_string(
        index=False
    ),
    "",
    (
        f"Absolute beta change (no2017-full): {absolute_beta_change:+.8f}"
    ),
    (
        f"Relative beta change: {relative_beta_change_pct:+.4f}%"
    ),
    (
        f"Sign changed: {sign_changed}"
    ),
    "",
    "LEAVE-ONE-YEAR-OUT",
    "-" * 118,
    loyo_df.to_string(
        index=False
    ),
    "",
    (
        f"Largest absolute coefficient shift from omitting year: "
        f"{largest_influence_year}"
    ),
    (
        f"2017 is the most influential omitted year: "
        f"{largest_influence_year == 2017}"
    ),
    "",
    "FIRST-DIFFERENCE TRANSITION SENSITIVITY",
    "-" * 118,
    fd_comparison.to_string(
        index=False
    ),
    "",
    (
        f"Full FD beta per +10 pp: {fd_full['beta_per_10pp']:+.6f} °C"
    ),
    (
        "FD beta per +10 pp excluding both 2016->2017 and 2017->2018: "
        f"{fd_exclude_both['beta_per_10pp']:+.6f} °C"
    ),
    (
        "FD sign changed after excluding both transitions: "
        f"{fd_sign_changed_after_removing_2017_transitions}"
    ),
    "",
    "FUTURE RESPONSE COMPARISON",
    "-" * 118,
    scenario_summary.to_string(
        index=False
    ),
    "",
    "INTERPRETATION",
    "-" * 118,
    (
        "2017 remains in the primary dataset and V3 primary result. V3S is "
        "only a robustness/sensitivity analysis."
    ),
    (
        "No coefficient is automatically preferred based on whether it "
        "produces cooling or warming."
    ),
    (
        "If the positive coefficient remains stable without 2017 and across "
        "leave-one-year-out / first-difference checks, then 2017 alone does "
        "not explain the positive association."
    ),
    (
        "If the sign or magnitude changes strongly when 2017 is removed, "
        "the final scenario interpretation must explicitly report that "
        "sensitivity."
    ),
    (
        "Observational panel associations are not causal proof."
    ),
]

REPORT_PATH.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)

print(
    f"Saved: {METHOD_PATH}"
)

print(
    f"Saved: {REPORT_PATH}"
)


# =============================================================================
# 10. HASHES
# =============================================================================

section(
    "10. REPRODUCIBILITY HASHES"
)

hash_files = [
    BASE_PATH,
    V3_COEFFICIENT_PATH,
    V3_AUDIT_PATH,
    FUTURE_FEATURE_PATH,
    FUTURE_PRED_PATH,
    TWFE_COMPARISON_PATH,
    LOYO_PATH,
    FD_COMPARISON_PATH,
    ANOMALY_CONTEXT_PATH,
    SCENARIO_COMPARISON_PATH,
    SCENARIO_SUMMARY_PATH,
    AUDIT_PATH,
    METHOD_PATH,
    REPORT_PATH,
    COEFFICIENT_FIGURE_PATH,
    LOYO_FIGURE_PATH,
    FD_FIGURE_PATH,
]

hash_rows = []

for path in hash_files:
    digest = sha256_file(
        path
    )

    hash_rows.append(
        {
            "file":
                str(
                    path
                ),
            "sha256":
                digest,
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
# FINAL
# =============================================================================

section(
    "STEP 29A V3S COMPLETED SUCCESSFULLY"
)

print(
    "2017 anomaly sensitivity analysis completed.\n"
    f"Full V3 TWFE green response per +10 pp: "
    f"{full['beta_per_10pp']:+.6f} °C.\n"
    f"No-2017 TWFE green response per +10 pp: "
    f"{no2017['beta_per_10pp']:+.6f} °C.\n"
    f"Relative coefficient change after excluding 2017: "
    f"{relative_beta_change_pct:+.2f}%.\n"
    f"TWFE sign changed after excluding 2017: {sign_changed}.\n"
    f"Largest leave-one-year-out influence year: "
    f"{largest_influence_year}.\n"
    f"2017 is largest leave-one-year-out influence: "
    f"{largest_influence_year == 2017}.\n"
    f"Full FD response per +10 pp: "
    f"{fd_full['beta_per_10pp']:+.6f} °C.\n"
    f"FD response excluding both transitions touching 2017: "
    f"{fd_exclude_both['beta_per_10pp']:+.6f} °C.\n"
    f"FD sign changed after excluding both transitions: "
    f"{fd_sign_changed_after_removing_2017_transitions}.\n"
    "2017 remains in the primary dataset and V3 primary estimate.\n"
    "No result was sign-forced or selected to obtain a preferred cooling outcome.\n"
    "Step-28 forecasts were not modified.\n"
    "V3S is sensitivity evidence only, not causal proof.\n"
    "\nNEXT: review V3S before deciding whether Step 29A can be frozen "
    "or requires a change in scenario interpretation."
)
