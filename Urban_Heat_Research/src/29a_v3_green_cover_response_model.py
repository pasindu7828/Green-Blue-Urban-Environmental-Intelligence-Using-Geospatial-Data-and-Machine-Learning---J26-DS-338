"""
STEP 29A V3 - DEDICATED GREEN-COVER RESPONSE MODEL
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
V1 and V2 used the frozen LightGBM heat-prediction model as a direct
intervention engine. The code was valid, but the model was trained for
prediction, not for estimating an intervention response.

V3 therefore separates the two tasks:

1) Step 28 remains the frozen future BASELINE heat forecast.
2) A dedicated historical panel-response model estimates the average
   within-grid association between green-cover level and LST.
3) The estimated green response is applied as a local adjustment to each
   Step-28 baseline future grid LST.

PRIMARY RESPONSE MODEL
----------------------
Two-way fixed-effects (TWFE) panel regression:

    LST_C_it
      = beta_green * green_mask_it
      + beta_1 * NDBI_it
      + beta_2 * NDWI_it
      + beta_3 * Albedo_it
      + beta_4 * NightLights_it
      + grid fixed effects
      + year fixed effects
      + error_it

Implemented by exact two-way demeaning because the historical panel is balanced.

Why this is more suitable:
- grid fixed effects remove time-invariant differences between locations;
- year fixed effects remove Kaduwela-wide annual thermal shocks;
- beta_green is estimated from within-grid changes over time;
- the coefficient has a direct level interpretation for a persistent scenario.

Cluster-robust standard errors are calculated by grid_id.

SENSITIVITY MODEL
-----------------
A first-difference regression is also fit:

    delta LST
      = gamma_green * delta green_mask
      + delta controls
      + target-year fixed effects
      + error

with grid-clustered standard errors.

This is NOT used to tune the primary result. It is a pre-defined robustness
check for sign and magnitude.

SCENARIO USE
------------
For 2026-2030:

    scenario green = clip(Step-28 baseline green + requested increase, 0, 1)

    estimated LST response
      = beta_green * actual green increase

    scenario LST
      = Step-28 baseline predicted LST + estimated response

    cooling benefit
      = - estimated response

The response is applied as a persistent level offset; it is NOT accumulated
again every year.

IMPORTANT LIMITATION
--------------------
TWFE improves intervention interpretation relative to direct predictive-model
perturbation, but observational panel data still do NOT prove causality.
Unmeasured time-varying confounding, measurement error, land-use co-change,
and reverse relationships may remain.

Therefore outputs are called:
    "historically estimated within-grid response"
not:
    "guaranteed causal cooling".

No coefficient sign is forced.
No result is corrected to produce cooling.
If the green coefficient is positive or uncertain, the code reports that
honestly and flags the scenario evidence accordingly.

Step 28 baseline forecasts are never modified on disk.
Step-22B LightGBM is not retrained.
Final total forecast uncertainty remains Step 30.
"""

from pathlib import Path
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

try:
    import geopandas as gpd
except ImportError as exc:
    raise ImportError(
        "\ngeopandas is required for Step 29A V3 spatial outputs.\n"
        "Install it with:\n"
        "    pip install geopandas\n"
    ) from exc


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET = "LST_C"
TREATMENT = "green_mask"

HIST_YEARS = list(range(2015, 2026))
FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]

EXPECTED_GRIDS = 2194
EXPECTED_HIST_ROWS = EXPECTED_GRIDS * len(HIST_YEARS)
EXPECTED_FUTURE_ROWS = EXPECTED_GRIDS * len(FUTURE_YEARS)

# Pre-defined primary controls.
# These represent important time-varying non-green environmental/urban context
# while avoiding other vegetation proxies such as NDVI/EVI that could
# over-control the green-cover treatment itself.
PRIMARY_CONTROLS = [
    "NDBI",
    "NDWI",
    "Albedo",
    "NightLights",
]

# Pre-defined coefficient stability windows. These are NOT used for tuning.
STABILITY_WINDOWS = [
    (2015, 2022),
    (2015, 2023),
    (2015, 2024),
    (2015, 2025),
]

SCENARIO_INCREASE_PP = [5, 10, 20]
PRIMARY_SCENARIO_PP = 5

GREEN_MIN = 0.0
GREEN_MAX = 1.0

BALANCE_TOL = 0
NUMERIC_TOL = 1e-12
PREDICTION_REPRO_TOL = 1e-10

EXPECTED_HASHES = {
    "kaduwela_preprocessed_base.csv":
        "0e030b2ec0b5c4080cd4226bec798189a178aa45efd9962d19f38dc9cb170f46",
    "21_final_feature_list.csv":
        "6769a195b5027850ebd458a4bef7c9da0ecc43c4969bcc81f746e46bce3be9c0",
    "22b_production_lightgbm_anomaly_trained_2018_2025.joblib":
        "a3548c58ba0a708c81b4f6471e9183aa70ce9572c716ee55bf2eddafe183e24f",
    "28_future_model_features_2026_2030.csv":
        "d1e68a2e5d063f231d7a4d379e2534005ad5873f3a6017d3c4f28de9b8acba38",
    "28_future_grid_lst_predictions_2026_2030.csv":
        "55586971042bbce04845a4a33ae724f51b8417fce9bafc2c15bc0a5e75d7ff3e",
    "28_future_prediction_audit.csv":
        "368daa91dcbce7ac009c8f11f0ecedde5817d5eb1ed0d2c99888f7be9017f3af",
    "kaduwela_grid_geometry_wgs84.geojson":
        "6b758ddaae19b54219404b86740472038cfd21391cf349ea766124678050f608",
}


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_PATH = (
    ROOT / "data" / "processed"
    / "kaduwela_preprocessed_base.csv"
)

FEATURE_LIST_PATH = (
    ROOT / "outputs" / "reports"
    / "21_final_feature_list.csv"
)

MODEL_PATH = (
    ROOT / "models" / "final"
    / "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
)

FUTURE_FEATURE_PATH = (
    ROOT / "outputs" / "forecasting"
    / "28_future_model_features_2026_2030.csv"
)

FUTURE_PRED_PATH = (
    ROOT / "outputs" / "forecasting"
    / "28_future_grid_lst_predictions_2026_2030.csv"
)

STEP28_AUDIT_PATH = (
    ROOT / "outputs" / "reports"
    / "28_future_prediction_audit.csv"
)

GRID_WGS84_PATH = (
    ROOT / "data" / "processed"
    / "kaduwela_grid_geometry_wgs84.geojson"
)

BOUNDARY_KML_PATH = (
    ROOT / "data" / "raw" / "boundary"
    / "kaduwela_kmc_boundary.kml"
)

SCENARIO_DIR = ROOT / "outputs" / "scenarios"
FIGURE_DIR = ROOT / "outputs" / "figures"
MAP_DIR = ROOT / "outputs" / "maps"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [
    SCENARIO_DIR,
    FIGURE_DIR,
    MAP_DIR,
    REPORT_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


COEFFICIENT_PATH = (
    REPORT_DIR
    / "29a_v3_twfe_primary_coefficients.csv"
)

STABILITY_PATH = (
    REPORT_DIR
    / "29a_v3_green_coefficient_stability.csv"
)

FD_COEFFICIENT_PATH = (
    REPORT_DIR
    / "29a_v3_first_difference_sensitivity_coefficients.csv"
)

MODEL_DIAGNOSTIC_PATH = (
    REPORT_DIR
    / "29a_v3_response_model_diagnostics.csv"
)

SCENARIO_PATH = (
    SCENARIO_DIR
    / "29a_v3_green_response_scenarios_2026_2030.csv"
)

PRIMARY_SCENARIO_PATH = (
    SCENARIO_DIR
    / f"29a_v3_green_response_primary_{PRIMARY_SCENARIO_PP}pp_2026_2030.csv"
)

SUMMARY_PATH = (
    SCENARIO_DIR
    / "29a_v3_green_response_scenario_summary.csv"
)

SUPPORT_PATH = (
    REPORT_DIR
    / "29a_v3_green_scenario_support_summary.csv"
)

AUDIT_PATH = (
    REPORT_DIR
    / "29a_v3_response_model_audit.csv"
)

METHOD_PATH = (
    REPORT_DIR
    / "29a_v3_response_model_methodology.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "29a_v3_response_model_report.txt"
)

HASH_PATH = (
    REPORT_DIR
    / "29a_v3_response_model_hashes.csv"
)

STABILITY_FIGURE_PATH = (
    FIGURE_DIR
    / "29a_v3_green_coefficient_stability.png"
)

SCENARIO_FIGURE_PATH = (
    FIGURE_DIR
    / "29a_v3_green_response_scenario_summary.png"
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
            f"{label} audit has no status column."
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
            f"{label} audit is not fully PASS."
        )


def check_balanced_panel(df, years):
    expected_per_grid = len(years)

    counts = (
        df.groupby(
            GRID_COL
        )[
            YEAR_COL
        ]
        .nunique()
    )

    bad = counts[
        counts
        != expected_per_grid
    ]

    if not bad.empty:
        raise RuntimeError(
            "Panel is not balanced for requested years. "
            f"Bad grids={len(bad)}"
        )

    if df[
        GRID_COL
    ].nunique() != EXPECTED_GRIDS:
        raise RuntimeError(
            f"Unique grids={df[GRID_COL].nunique()}, "
            f"expected={EXPECTED_GRIDS}"
        )


def two_way_demean(df, columns):
    """
    Exact two-way demeaning for the balanced grid x year panel:

        x_it_tilde
          = x_it
          - mean_i(x)
          - mean_t(x)
          + grand_mean(x)

    This removes grid and year fixed effects.
    """

    result = pd.DataFrame(
        index=df.index
    )

    for column in columns:
        values = pd.to_numeric(
            df[column],
            errors="raise",
        ).astype(float)

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

        time_mean = (
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

        result[
            column
        ] = (
            values
            - entity_mean
            - time_mean
            + grand_mean
        )

    return result


def fit_twfe(
    panel,
    start_year,
    end_year,
):
    """
    Fit pre-defined TWFE response model with grid-clustered SE.
    """

    sub = (
        panel[
            panel[
                YEAR_COL
            ].between(
                start_year,
                end_year,
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

    years = list(
        range(
            start_year,
            end_year + 1,
        )
    )

    check_balanced_panel(
        sub,
        years,
    )

    model_columns = [
        TARGET,
        TREATMENT,
        *PRIMARY_CONTROLS,
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
            *PRIMARY_CONTROLS,
        ]
    ].copy()

    # Strict rank/variance check.
    variances = X.var(
        ddof=0
    )

    zero_var = variances[
        variances
        <= NUMERIC_TOL
    ]

    if not zero_var.empty:
        raise RuntimeError(
            "Zero/near-zero within variation in TWFE predictors:\n"
            + zero_var.to_string()
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
            f"TWFE design matrix rank={rank}, "
            f"columns={X.shape[1]}."
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

    # Model diagnostics on within-transformed outcome.
    residual = np.asarray(
        fit.resid,
        dtype=float,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(
                residual
            )
        )
    )

    return {
        "fit":
            fit,
        "sub":
            sub,
        "transformed":
            transformed,
        "within_rmse":
            rmse,
        "within_mae":
            mae,
        "within_r2":
            float(
                fit.rsquared
            ),
        "nobs":
            int(
                fit.nobs
            ),
        "grid_count":
            int(
                sub[
                    GRID_COL
                ].nunique()
            ),
        "year_count":
            len(
                years
            ),
        "rank":
            rank,
    }


def coefficient_table(
    fit,
    model_name,
):
    conf = fit.conf_int(
        alpha=0.05
    )

    rows = []

    for parameter in fit.params.index:
        rows.append(
            {
                "model":
                    model_name,
                "term":
                    parameter,
                "coefficient":
                    float(
                        fit.params[
                            parameter
                        ]
                    ),
                "std_error_clustered":
                    float(
                        fit.bse[
                            parameter
                        ]
                    ),
                "p_value_clustered":
                    float(
                        fit.pvalues[
                            parameter
                        ]
                    ),
                "ci95_lower":
                    float(
                        conf.loc[
                            parameter,
                            0,
                        ]
                    ),
                "ci95_upper":
                    float(
                        conf.loc[
                            parameter,
                            1,
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def fit_first_difference_sensitivity(panel):
    """
    Pre-defined first-difference sensitivity model with target-year fixed effects
    and grid-clustered standard errors.
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
    )

    diff_variables = [
        TARGET,
        TREATMENT,
        *PRIMARY_CONTROLS,
    ]

    for column in diff_variables:
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
            ]
            .between(
                2016,
                2025,
            )
        ]
        .dropna(
            subset=[
                f"d_{column}"
                for column in diff_variables
            ]
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    expected_rows = (
        EXPECTED_GRIDS
        * 10
    )

    if len(
        fd
    ) != expected_rows:
        raise RuntimeError(
            f"First-difference rows={len(fd)}, "
            f"expected={expected_rows}"
        )

    y = fd[
        f"d_{TARGET}"
    ].astype(
        float
    )

    X_core = fd[
        [
            f"d_{TREATMENT}",
            *[
                f"d_{column}"
                for column in PRIMARY_CONTROLS
            ],
        ]
    ].astype(
        float
    )

    year_dummies = pd.get_dummies(
        fd[
            YEAR_COL
        ].astype(
            int
        ),
        prefix="year",
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
                fd[
                    GRID_COL
                ].astype(
                    str
                ).to_numpy(),
            "use_correction":
                True,
        },
    )

    return {
        "fit":
            fit,
        "data":
            fd,
    }


def evidence_status(
    beta,
    ci_lower,
    ci_upper,
):
    if beta < 0 and ci_upper < 0:
        return (
            "NEGATIVE_ASSOCIATION_SUPPORTED"
        )

    if beta < 0 and ci_upper >= 0:
        return (
            "NEGATIVE_BUT_UNCERTAIN"
        )

    if beta >= 0 and ci_lower <= 0:
        return (
            "NONNEGATIVE_AND_UNCERTAIN"
        )

    return (
        "POSITIVE_ASSOCIATION_SUPPORTED"
    )


def load_boundary_wgs84():
    if not BOUNDARY_KML_PATH.exists():
        return None

    try:
        boundary = gpd.read_file(
            BOUNDARY_KML_PATH
        )

        if boundary.empty:
            return None

        if boundary.crs is None:
            boundary = boundary.set_crs(
                "EPSG:4326"
            )
        else:
            boundary = boundary.to_crs(
                "EPSG:4326"
            )

        return boundary

    except Exception as exc:
        print(
            "WARNING: boundary KML could not be loaded."
        )

        print(
            f"  {type(exc).__name__}: {exc}"
        )

        return None


# =============================================================================
# 0. VERIFY FROZEN INPUTS
# =============================================================================

section(
    "STEP 29A V3 - DEDICATED GREEN-COVER RESPONSE MODEL"
)

verify_hash(
    BASE_PATH,
    EXPECTED_HASHES[
        "kaduwela_preprocessed_base.csv"
    ],
    "Canonical historical base data",
)

verify_hash(
    FEATURE_LIST_PATH,
    EXPECTED_HASHES[
        "21_final_feature_list.csv"
    ],
    "Frozen Step-21 feature list",
)

verify_hash(
    MODEL_PATH,
    EXPECTED_HASHES[
        "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
    ],
    "Frozen Step-22B production LightGBM",
)

verify_hash(
    FUTURE_FEATURE_PATH,
    EXPECTED_HASHES[
        "28_future_model_features_2026_2030.csv"
    ],
    "Step-28 future frozen-feature matrix",
)

verify_hash(
    FUTURE_PRED_PATH,
    EXPECTED_HASHES[
        "28_future_grid_lst_predictions_2026_2030.csv"
    ],
    "Step-28 future grid-LST predictions",
)

verify_hash(
    STEP28_AUDIT_PATH,
    EXPECTED_HASHES[
        "28_future_prediction_audit.csv"
    ],
    "Step-28 strict audit",
)

verify_hash(
    GRID_WGS84_PATH,
    EXPECTED_HASHES[
        "kaduwela_grid_geometry_wgs84.geojson"
    ],
    "Corrected WGS84 grid geometry",
)

require_all_pass(
    STEP28_AUDIT_PATH,
    "Step 28",
)

print(
    "\nPASS: Step-28 audit is fully PASS."
)

print(
    "\nV3 does NOT use LightGBM feature perturbation as the intervention estimator."
)

print(
    "LightGBM remains the frozen Step-28 baseline forecast model only."
)


# =============================================================================
# 1. LOAD + VALIDATE HISTORICAL PANEL
# =============================================================================

section(
    "1. LOAD AND VALIDATE 2015-2025 HISTORICAL PANEL"
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
    *PRIMARY_CONTROLS,
]

missing = [
    column
    for column in required_columns
    if column not in base.columns
]

if missing:
    raise RuntimeError(
        "Historical base data is missing V3 columns:\n"
        + str(
            missing
        )
    )

panel = (
    base[
        base[
            YEAR_COL
        ].isin(
            HIST_YEARS
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

if len(
    panel
) != EXPECTED_HIST_ROWS:
    raise RuntimeError(
        f"Historical panel rows={len(panel)}, "
        f"expected={EXPECTED_HIST_ROWS}"
    )

if panel.duplicated(
    [
        GRID_COL,
        YEAR_COL,
    ]
).any():
    raise RuntimeError(
        "Historical panel contains duplicate grid-year rows."
    )

check_balanced_panel(
    panel,
    HIST_YEARS,
)

for column in [
    TARGET,
    TREATMENT,
    *PRIMARY_CONTROLS,
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
        *PRIMARY_CONTROLS,
    ]
].isna().any().any():
    raise RuntimeError(
        "Historical V3 panel contains missing model values."
    )

if not np.isfinite(
    panel[
        [
            TARGET,
            TREATMENT,
            *PRIMARY_CONTROLS,
        ]
    ].to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "Historical V3 panel contains non-finite values."
    )

green_min_hist = float(
    panel[
        TREATMENT
    ].min()
)

green_max_hist = float(
    panel[
        TREATMENT
    ].max()
)

if (
    green_min_hist
    < GREEN_MIN
    - 1e-12
    or green_max_hist
    > GREEN_MAX
    + 1e-12
):
    raise RuntimeError(
        "Historical green_mask outside expected [0,1] range."
    )

print(
    f"Rows          : {len(panel):,}"
)

print(
    f"Unique grids  : {panel[GRID_COL].nunique():,}"
)

print(
    f"Years         : {panel[YEAR_COL].min()}-{panel[YEAR_COL].max()}"
)

print(
    f"Green range   : {green_min_hist:.6f} to {green_max_hist:.6f}"
)

print(
    f"Primary controls: {PRIMARY_CONTROLS}"
)


# =============================================================================
# 2. FIT PRIMARY FULL-HISTORY TWO-WAY FIXED-EFFECTS MODEL
# =============================================================================

section(
    "2. FIT PRIMARY TWO-WAY FIXED-EFFECTS RESPONSE MODEL"
)

primary_result = fit_twfe(
    panel=panel,
    start_year=2015,
    end_year=2025,
)

primary_fit = primary_result[
    "fit"
]

primary_coefficients = coefficient_table(
    primary_fit,
    "TWFE_2015_2025_PRIMARY",
)

primary_coefficients.to_csv(
    COEFFICIENT_PATH,
    index=False,
)

beta_green = float(
    primary_fit.params[
        TREATMENT
    ]
)

se_green = float(
    primary_fit.bse[
        TREATMENT
    ]
)

p_green = float(
    primary_fit.pvalues[
        TREATMENT
    ]
)

green_ci = primary_fit.conf_int(
    alpha=0.05
).loc[
    TREATMENT
]

green_ci_lower = float(
    green_ci[
        0
    ]
)

green_ci_upper = float(
    green_ci[
        1
    ]
)

green_status = evidence_status(
    beta_green,
    green_ci_lower,
    green_ci_upper,
)

print(
    primary_coefficients.to_string(
        index=False
    )
)

print(
    "\nPrimary green-cover response:"
)

print(
    f"  beta per 1.00 green fraction : {beta_green:+.6f} °C"
)

print(
    f"  beta per +10 percentage pts  : {beta_green * 0.10:+.6f} °C"
)

print(
    f"  clustered SE                 : {se_green:.6f}"
)

print(
    f"  95% CI per +10 pp            : "
    f"[{green_ci_lower * 0.10:+.6f}, {green_ci_upper * 0.10:+.6f}] °C"
)

print(
    f"  clustered p-value            : {p_green:.6g}"
)

print(
    f"  evidence status              : {green_status}"
)

print(
    f"  within R²                    : {primary_result['within_r2']:.6f}"
)

print(
    f"  within MAE                   : {primary_result['within_mae']:.6f} °C"
)

print(
    f"  within RMSE                  : {primary_result['within_rmse']:.6f} °C"
)


# =============================================================================
# 3. PRE-DEFINED TIME-WINDOW COEFFICIENT STABILITY
# =============================================================================

section(
    "3. PRE-DEFINED GREEN-COEFFICIENT STABILITY CHECK"
)

stability_rows = []

for start_year, end_year in STABILITY_WINDOWS:
    result = fit_twfe(
        panel=panel,
        start_year=start_year,
        end_year=end_year,
    )

    fit = result[
        "fit"
    ]

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

    lo = float(
        ci[
            0
        ]
    )

    hi = float(
        ci[
            1
        ]
    )

    stability_rows.append(
        {
            "start_year":
                start_year,
            "end_year":
                end_year,
            "rows":
                int(
                    result[
                        "nobs"
                    ]
                ),
            "grids":
                int(
                    result[
                        "grid_count"
                    ]
                ),
            "green_beta_c_per_fraction":
                beta,
            "green_beta_c_per_10pp":
                beta
                * 0.10,
            "clustered_se":
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
            "ci95_lower_per_10pp":
                lo
                * 0.10,
            "ci95_upper_per_10pp":
                hi
                * 0.10,
            "evidence_status":
                evidence_status(
                    beta,
                    lo,
                    hi,
                ),
            "within_r2":
                float(
                    result[
                        "within_r2"
                    ]
                ),
            "within_mae_c":
                float(
                    result[
                        "within_mae"
                    ]
                ),
            "within_rmse_c":
                float(
                    result[
                        "within_rmse"
                    ]
                ),
        }
    )

stability_df = pd.DataFrame(
    stability_rows
)

stability_df.to_csv(
    STABILITY_PATH,
    index=False,
)

print(
    stability_df.to_string(
        index=False
    )
)

stability_signs = np.sign(
    stability_df[
        "green_beta_c_per_fraction"
    ].to_numpy(
        dtype=float
    )
)

same_sign_all_windows = bool(
    np.all(
        stability_signs
        == stability_signs[
            -1
        ]
    )
)

print(
    f"\nGreen coefficient has same sign across all pre-defined windows: "
    f"{same_sign_all_windows}"
)


# =============================================================================
# 4. FIRST-DIFFERENCE SENSITIVITY MODEL
# =============================================================================

section(
    "4. FIRST-DIFFERENCE SENSITIVITY MODEL"
)

fd_result = fit_first_difference_sensitivity(
    panel
)

fd_fit = fd_result[
    "fit"
]

fd_coefficients = coefficient_table(
    fd_fit,
    "FIRST_DIFFERENCE_2016_2025_SENSITIVITY",
)

fd_coefficients.to_csv(
    FD_COEFFICIENT_PATH,
    index=False,
)

fd_green_term = (
    f"d_{TREATMENT}"
)

fd_beta = float(
    fd_fit.params[
        fd_green_term
    ]
)

fd_ci = fd_fit.conf_int(
    alpha=0.05
).loc[
    fd_green_term
]

fd_lo = float(
    fd_ci[
        0
    ]
)

fd_hi = float(
    fd_ci[
        1
    ]
)

fd_status = evidence_status(
    fd_beta,
    fd_lo,
    fd_hi,
)

print(
    f"First-difference green coefficient per +10 pp: "
    f"{fd_beta * 0.10:+.6f} °C"
)

print(
    f"95% CI per +10 pp: "
    f"[{fd_lo * 0.10:+.6f}, {fd_hi * 0.10:+.6f}] °C"
)

print(
    f"Evidence status: {fd_status}"
)

primary_fd_same_sign = bool(
    np.sign(
        beta_green
    )
    == np.sign(
        fd_beta
    )
)

print(
    f"Primary TWFE and first-difference green coefficients have same sign: "
    f"{primary_fd_same_sign}"
)


# =============================================================================
# 5. MODEL / SUPPORT DIAGNOSTICS
# =============================================================================

section(
    "5. RESPONSE-MODEL AND HISTORICAL-SUPPORT DIAGNOSTICS"
)

hist_grid_support = (
    panel.groupby(
        GRID_COL
    )[
        TREATMENT
    ]
    .agg(
        hist_green_min="min",
        hist_green_max="max",
        hist_green_mean="mean",
        hist_green_std="std",
    )
    .reset_index()
)

hist_grid_support[
    "hist_green_range"
] = (
    hist_grid_support[
        "hist_green_max"
    ]
    - hist_grid_support[
        "hist_green_min"
    ]
)

# Historical year-to-year green changes are descriptive support information.
panel_sorted = panel.sort_values(
    [
        GRID_COL,
        YEAR_COL,
    ]
).copy()

panel_sorted[
    "d_green_mask"
] = (
    panel_sorted.groupby(
        GRID_COL
    )[
        TREATMENT
    ]
    .diff()
)

observed_abs_change = (
    panel_sorted[
        "d_green_mask"
    ]
    .dropna()
    .abs()
)

q50_abs_change = float(
    observed_abs_change.quantile(
        0.50
    )
)

q90_abs_change = float(
    observed_abs_change.quantile(
        0.90
    )
)

q95_abs_change = float(
    observed_abs_change.quantile(
        0.95
    )
)

q99_abs_change = float(
    observed_abs_change.quantile(
        0.99
    )
)

diagnostics = pd.DataFrame(
    [
        {
            "metric":
                "primary_twfe_green_beta_per_10pp_c",
            "value":
                beta_green
                * 0.10,
            "note":
                "Within-grid association controlling for NDBI, NDWI, Albedo, NightLights, grid FE and year FE.",
        },
        {
            "metric":
                "primary_twfe_green_ci95_lower_per_10pp_c",
            "value":
                green_ci_lower
                * 0.10,
            "note":
                "Cluster-robust by grid_id.",
        },
        {
            "metric":
                "primary_twfe_green_ci95_upper_per_10pp_c",
            "value":
                green_ci_upper
                * 0.10,
            "note":
                "Cluster-robust by grid_id.",
        },
        {
            "metric":
                "primary_twfe_within_r2",
            "value":
                primary_result[
                    "within_r2"
                ],
            "note":
                "Descriptive within-panel model fit; not a causal validation score.",
        },
        {
            "metric":
                "primary_twfe_within_mae_c",
            "value":
                primary_result[
                    "within_mae"
                ],
            "note":
                "Descriptive residual MAE on two-way demeaned panel.",
        },
        {
            "metric":
                "primary_twfe_within_rmse_c",
            "value":
                primary_result[
                    "within_rmse"
                ],
            "note":
                "Descriptive residual RMSE on two-way demeaned panel.",
        },
        {
            "metric":
                "first_difference_green_beta_per_10pp_c",
            "value":
                fd_beta
                * 0.10,
            "note":
                "Pre-defined robustness sensitivity.",
        },
        {
            "metric":
                "historical_abs_delta_green_q50",
            "value":
                q50_abs_change,
            "note":
                "Median absolute year-to-year green_mask change.",
        },
        {
            "metric":
                "historical_abs_delta_green_q90",
            "value":
                q90_abs_change,
            "note":
                "90th percentile absolute year-to-year green_mask change.",
        },
        {
            "metric":
                "historical_abs_delta_green_q95",
            "value":
                q95_abs_change,
            "note":
                "95th percentile absolute year-to-year green_mask change.",
        },
        {
            "metric":
                "historical_abs_delta_green_q99",
            "value":
                q99_abs_change,
            "note":
                "99th percentile absolute year-to-year green_mask change.",
        },
    ]
)

diagnostics.to_csv(
    MODEL_DIAGNOSTIC_PATH,
    index=False,
)

print(
    diagnostics.to_string(
        index=False
    )
)


# =============================================================================
# 6. LOAD STEP-28 FUTURE BASELINE + STRICTLY REPRODUCE LIGHTGBM
# =============================================================================

section(
    "6. LOAD STEP-28 FUTURE BASELINE AND VERIFY IT REMAINS FROZEN"
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

for frame, label in [
    (
        future,
        "future feature matrix",
    ),
    (
        pred,
        "future prediction file",
    ),
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

    if len(
        frame
    ) != EXPECTED_FUTURE_ROWS:
        raise RuntimeError(
            f"{label}: rows={len(frame)}, "
            f"expected={EXPECTED_FUTURE_ROWS}"
        )

    if sorted(
        frame[
            YEAR_COL
        ]
        .unique()
        .tolist()
    ) != FUTURE_YEARS:
        raise RuntimeError(
            f"{label}: unexpected future years."
        )

    if frame.duplicated(
        [
            GRID_COL,
            YEAR_COL,
        ]
    ).any():
        raise RuntimeError(
            f"{label}: duplicate grid-year rows."
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

if TREATMENT not in future.columns:
    raise RuntimeError(
        "Step-28 future feature matrix does not contain green_mask."
    )

baseline_green_future = pd.to_numeric(
    future[
        TREATMENT
    ],
    errors="raise",
).to_numpy(
    dtype=float
)

baseline_lst_future = pd.to_numeric(
    pred[
        "predicted_lst_c"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)

# Reproduce the frozen Step-28 LightGBM output as an integrity check only.
feature_list_df = pd.read_csv(
    FEATURE_LIST_PATH
)

feature_column = None

for candidate in [
    "feature",
    "feature_name",
    "predictor",
]:
    if (
        candidate
        in feature_list_df.columns
        and feature_list_df[
            candidate
        ].dropna().shape[
            0
        ]
        == 93
    ):
        feature_column = candidate
        break

if feature_column is None:
    for column in feature_list_df.columns:
        values = (
            feature_list_df[
                column
            ]
            .dropna()
            .astype(
                str
            )
            .tolist()
        )

        if (
            len(
                values
            )
            == 93
            and len(
                set(
                    values
                )
            )
            == 93
        ):
            feature_column = column
            break

if feature_column is None:
    raise RuntimeError(
        "Could not identify frozen Step-21 feature column."
    )

frozen_features = (
    feature_list_df[
        feature_column
    ]
    .dropna()
    .astype(
        str
    )
    .tolist()
)

missing_future_features = [
    feature
    for feature in frozen_features
    if feature not in future.columns
]

if missing_future_features:
    raise RuntimeError(
        "Step-28 matrix missing frozen features:\n"
        + str(
            missing_future_features
        )
    )

X_future = future[
    frozen_features
].copy()

for feature in frozen_features:
    X_future[
        feature
    ] = pd.to_numeric(
        X_future[
            feature
        ],
        errors="raise",
    )

model = joblib.load(
    MODEL_PATH
)

raw_reproduced = np.asarray(
    model.predict(
        X_future
    ),
    dtype=float,
)

raw_saved = pd.to_numeric(
    pred[
        "raw_spatial_anomaly_prediction_c"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)

step28_reproduction_error = float(
    np.max(
        np.abs(
            raw_reproduced
            - raw_saved
        )
    )
)

print(
    f"Step-28 raw LightGBM reproduction max error: "
    f"{step28_reproduction_error:.12e} °C"
)

if (
    step28_reproduction_error
    > PREDICTION_REPRO_TOL
):
    raise RuntimeError(
        "Frozen Step-28 baseline LightGBM could not be reproduced."
    )

print(
    "PASS: Step-28 baseline remains exactly reproducible."
)


# =============================================================================
# 7. BUILD V3 PERSISTENT GREEN RESPONSE SCENARIOS
# =============================================================================

section(
    "7. BUILD 2026-2030 V3 GREEN-RESPONSE SCENARIOS"
)

hist_support_lookup = hist_grid_support.set_index(
    GRID_COL
)

scenario_rows = []

for pp in SCENARIO_INCREASE_PP:
    requested_fraction = (
        pp
        / 100.0
    )

    scenario_green = np.clip(
        baseline_green_future
        + requested_fraction,
        GREEN_MIN,
        GREEN_MAX,
    )

    actual_increase = (
        scenario_green
        - baseline_green_future
    )

    # Dedicated response model adjustment.
    estimated_response = (
        beta_green
        * actual_increase
    )

    response_ci_a = (
        green_ci_lower
        * actual_increase
    )

    response_ci_b = (
        green_ci_upper
        * actual_increase
    )

    response_ci_lower = np.minimum(
        response_ci_a,
        response_ci_b,
    )

    response_ci_upper = np.maximum(
        response_ci_a,
        response_ci_b,
    )

    scenario_lst = (
        baseline_lst_future
        + estimated_response
    )

    scenario_lst_response_ci_lower = (
        baseline_lst_future
        + response_ci_lower
    )

    scenario_lst_response_ci_upper = (
        baseline_lst_future
        + response_ci_upper
    )

    cooling_benefit = (
        -estimated_response
    )

    cooling_ci_lower = (
        -response_ci_upper
    )

    cooling_ci_upper = (
        -response_ci_lower
    )

    limited_by_upper_bound = (
        actual_increase
        < requested_fraction
        - 1e-12
    )

    for i in range(
        EXPECTED_FUTURE_ROWS
    ):
        grid_id = str(
            future.at[
                i,
                GRID_COL,
            ]
        )

        hist_min = float(
            hist_support_lookup.at[
                grid_id,
                "hist_green_min",
            ]
        )

        hist_max = float(
            hist_support_lookup.at[
                grid_id,
                "hist_green_max",
            ]
        )

        hist_range = float(
            hist_support_lookup.at[
                grid_id,
                "hist_green_range",
            ]
        )

        scenario_value = float(
            scenario_green[
                i
            ]
        )

        outside_hist_max = bool(
            scenario_value
            > hist_max
            + 1e-12
        )

        increase_fraction_i = float(
            actual_increase[
                i
            ]
        )

        # Scenario size relative to historical annual-change support.
        if increase_fraction_i <= q95_abs_change + 1e-12:
            support_class = (
                "WITHIN_95PCT_HISTORICAL_ANNUAL_CHANGE"
            )
        elif increase_fraction_i <= q99_abs_change + 1e-12:
            support_class = (
                "BETWEEN_95_AND_99PCT_HISTORICAL_ANNUAL_CHANGE"
            )
        else:
            support_class = (
                "ABOVE_99PCT_HISTORICAL_ANNUAL_CHANGE"
            )

        response = float(
            estimated_response[
                i
            ]
        )

        benefit = float(
            cooling_benefit[
                i
            ]
        )

        if benefit > 1e-12:
            direction = (
                "cooling"
            )
        elif benefit < -1e-12:
            direction = (
                "warming"
            )
        else:
            direction = (
                "no_change"
            )

        scenario_rows.append(
            {
                GRID_COL:
                    grid_id,
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
                "scenario_id":
                    f"v3_green_plus_{pp}pp",
                "response_model":
                    "TWFE_2015_2025_PRIMARY",
                "response_evidence_status":
                    green_status,
                "requested_green_increase_pp":
                    pp,
                "baseline_green_mask":
                    float(
                        baseline_green_future[
                            i
                        ]
                    ),
                "scenario_green_mask":
                    scenario_value,
                "actual_green_increase_pp":
                    increase_fraction_i
                    * 100.0,
                "green_upper_bound_limited":
                    bool(
                        limited_by_upper_bound[
                            i
                        ]
                    ),
                "historical_grid_green_min":
                    hist_min,
                "historical_grid_green_max":
                    hist_max,
                "historical_grid_green_range":
                    hist_range,
                "scenario_green_above_historical_grid_max":
                    outside_hist_max,
                "scenario_support_class":
                    support_class,
                "baseline_predicted_lst_c":
                    float(
                        baseline_lst_future[
                            i
                        ]
                    ),
                "estimated_lst_response_c":
                    response,
                "response_ci95_lower_c":
                    float(
                        response_ci_lower[
                            i
                        ]
                    ),
                "response_ci95_upper_c":
                    float(
                        response_ci_upper[
                            i
                        ]
                    ),
                "scenario_predicted_lst_c":
                    float(
                        scenario_lst[
                            i
                        ]
                    ),
                "scenario_lst_response_ci95_lower_c":
                    float(
                        scenario_lst_response_ci_lower[
                            i
                        ]
                    ),
                "scenario_lst_response_ci95_upper_c":
                    float(
                        scenario_lst_response_ci_upper[
                            i
                        ]
                    ),
                "cooling_benefit_c":
                    benefit,
                "cooling_benefit_ci95_lower_c":
                    float(
                        cooling_ci_lower[
                            i
                        ]
                    ),
                "cooling_benefit_ci95_upper_c":
                    float(
                        cooling_ci_upper[
                            i
                        ]
                    ),
                "scenario_effect_direction":
                    direction,
                "response_ci_excludes_zero":
                    bool(
                        (
                            response_ci_lower[
                                i
                            ]
                            > 0
                        )
                        or (
                            response_ci_upper[
                                i
                            ]
                            < 0
                        )
                    ),
            }
        )

scenario_df = pd.DataFrame(
    scenario_rows
)

expected_scenario_rows = (
    EXPECTED_FUTURE_ROWS
    * len(
        SCENARIO_INCREASE_PP
    )
)

if len(
    scenario_df
) != expected_scenario_rows:
    raise RuntimeError(
        f"Scenario rows={len(scenario_df)}, "
        f"expected={expected_scenario_rows}"
    )

scenario_df.to_csv(
    SCENARIO_PATH,
    index=False,
)

scenario_df[
    scenario_df[
        "requested_green_increase_pp"
    ]
    == PRIMARY_SCENARIO_PP
].to_csv(
    PRIMARY_SCENARIO_PATH,
    index=False,
)

for year in FUTURE_YEARS:
    for pp in SCENARIO_INCREASE_PP:
        scenario_df[
            (
                scenario_df[
                    YEAR_COL
                ]
                == year
            )
            & (
                scenario_df[
                    "requested_green_increase_pp"
                ]
                == pp
            )
        ].to_csv(
            SCENARIO_DIR
            / f"29a_v3_green_response_{pp}pp_{year}.csv",
            index=False,
        )

print(
    f"Saved: {SCENARIO_PATH}"
)


# =============================================================================
# 8. V3 SCENARIO SUMMARY + SUPPORT SUMMARY
# =============================================================================

section(
    "8. V3 SCENARIO SUMMARY AND EMPIRICAL-SUPPORT SUMMARY"
)

summary = (
    scenario_df
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
        upper_bound_limited_grid_count=(
            "green_upper_bound_limited",
            "sum",
        ),
        mean_baseline_lst_c=(
            "baseline_predicted_lst_c",
            "mean",
        ),
        mean_scenario_lst_c=(
            "scenario_predicted_lst_c",
            "mean",
        ),
        mean_estimated_lst_response_c=(
            "estimated_lst_response_c",
            "mean",
        ),
        mean_cooling_benefit_c=(
            "cooling_benefit_c",
            "mean",
        ),
        median_cooling_benefit_c=(
            "cooling_benefit_c",
            "median",
        ),
        mean_cooling_ci95_lower_c=(
            "cooling_benefit_ci95_lower_c",
            "mean",
        ),
        mean_cooling_ci95_upper_c=(
            "cooling_benefit_ci95_upper_c",
            "mean",
        ),
        above_historical_grid_max_count=(
            "scenario_green_above_historical_grid_max",
            "sum",
        ),
    )
)

summary[
    "above_historical_grid_max_pct"
] = (
    summary[
        "above_historical_grid_max_count"
    ]
    / summary[
        "grid_count"
    ]
    * 100.0
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

support_summary = (
    scenario_df.groupby(
        [
            "requested_green_increase_pp",
            "scenario_support_class",
        ],
        as_index=False,
    )
    .agg(
        rows=(
            GRID_COL,
            "size",
        )
    )
)

support_total = (
    support_summary.groupby(
        "requested_green_increase_pp"
    )[
        "rows"
    ]
    .transform(
        "sum"
    )
)

support_summary[
    "pct"
] = (
    support_summary[
        "rows"
    ]
    / support_total
    * 100.0
)

support_summary.to_csv(
    SUPPORT_PATH,
    index=False,
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nScenario support distribution:"
)

print(
    support_summary.to_string(
        index=False
    )
)

print(
    "\nNOTE:"
)

print(
    "Response-model CI reflects uncertainty in the historical green-response coefficient only."
)

print(
    "It is NOT the final total future-LST prediction interval; Step 30 handles forecast uncertainty."
)


# =============================================================================
# 9. FIGURES
# =============================================================================

section(
    "9. SAVE RESPONSE-MODEL FIGURES"
)

# 9A. Coefficient stability
fig, ax = plt.subplots(
    figsize=(
        9,
        6,
    )
)

x = np.arange(
    len(
        stability_df
    )
)

y = stability_df[
    "green_beta_c_per_10pp"
].to_numpy(
    dtype=float
)

lower = (
    y
    - stability_df[
        "ci95_lower_per_10pp"
    ].to_numpy(
        dtype=float
    )
)

upper = (
    stability_df[
        "ci95_upper_per_10pp"
    ].to_numpy(
        dtype=float
    )
    - y
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
        f"{row.start_year}-{row.end_year}"
        for row in stability_df.itertuples()
    ]
)

ax.set_ylabel(
    "Estimated LST response per +10 pp green cover (°C)"
)

ax.set_xlabel(
    "Pre-defined historical estimation window"
)

ax.set_title(
    "Green-Cover Response Coefficient Stability"
)

ax.grid(
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    STABILITY_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

# 9B. Scenario summary
scenario_plot = (
    scenario_df.groupby(
        "requested_green_increase_pp",
        as_index=False,
    )
    .agg(
        mean_cooling=(
            "cooling_benefit_c",
            "mean",
        ),
        mean_lower=(
            "cooling_benefit_ci95_lower_c",
            "mean",
        ),
        mean_upper=(
            "cooling_benefit_ci95_upper_c",
            "mean",
        ),
    )
)

fig, ax = plt.subplots(
    figsize=(
        8,
        6,
    )
)

x = scenario_plot[
    "requested_green_increase_pp"
].to_numpy(
    dtype=float
)

y = scenario_plot[
    "mean_cooling"
].to_numpy(
    dtype=float
)

lower = (
    y
    - scenario_plot[
        "mean_lower"
    ].to_numpy(
        dtype=float
    )
)

upper = (
    scenario_plot[
        "mean_upper"
    ].to_numpy(
        dtype=float
    )
    - y
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
    marker="o",
    capsize=5,
)

ax.axhline(
    0.0,
    linewidth=1.0,
    color="black",
)

ax.set_xlabel(
    "Green-cover increase (percentage points)"
)

ax.set_ylabel(
    "Historically estimated cooling benefit (°C)"
)

ax.set_title(
    "V3 Green-Cover Response Scenario"
)

ax.grid(
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    SCENARIO_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    f"Saved: {STABILITY_FIGURE_PATH}"
)

print(
    f"Saved: {SCENARIO_FIGURE_PATH}"
)


# =============================================================================
# 10. FRONTEND-READY GEOJSON + MAPS
# =============================================================================

section(
    "10. SAVE FRONTEND-READY V3 GEOJSON + MAPS"
)

grid = gpd.read_file(
    GRID_WGS84_PATH
)

grid[
    GRID_COL
] = grid[
    GRID_COL
].astype(
    str
)

if grid[
    GRID_COL
].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        "WGS84 grid geometry count mismatch."
    )

grid = grid[
    [
        GRID_COL,
        "geometry",
    ]
].copy()

boundary = load_boundary_wgs84()

geojson_files = []
map_files = []

for year in FUTURE_YEARS:
    year_base = (
        pred[
            pred[
                YEAR_COL
            ]
            == year
        ][
            [
                GRID_COL,
                YEAR_COL,
                "area_name",
                "predicted_lst_c",
            ]
        ]
        .copy()
        .rename(
            columns={
                "predicted_lst_c":
                    "baseline_predicted_lst_c",
            }
        )
    )

    year_merge = year_base.copy()

    for pp in SCENARIO_INCREASE_PP:
        subset = (
            scenario_df[
                (
                    scenario_df[
                        YEAR_COL
                    ]
                    == year
                )
                & (
                    scenario_df[
                        "requested_green_increase_pp"
                    ]
                    == pp
                )
            ][
                [
                    GRID_COL,
                    "scenario_green_mask",
                    "actual_green_increase_pp",
                    "estimated_lst_response_c",
                    "scenario_predicted_lst_c",
                    "cooling_benefit_c",
                    "cooling_benefit_ci95_lower_c",
                    "cooling_benefit_ci95_upper_c",
                    "scenario_green_above_historical_grid_max",
                    "scenario_support_class",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "scenario_green_mask":
                        f"green_mask_{pp}pp",
                    "actual_green_increase_pp":
                        f"actual_green_increase_{pp}pp",
                    "estimated_lst_response_c":
                        f"response_{pp}pp_c",
                    "scenario_predicted_lst_c":
                        f"scenario_lst_{pp}pp_c",
                    "cooling_benefit_c":
                        f"cooling_benefit_{pp}pp_c",
                    "cooling_benefit_ci95_lower_c":
                        f"cooling_ci95_lower_{pp}pp_c",
                    "cooling_benefit_ci95_upper_c":
                        f"cooling_ci95_upper_{pp}pp_c",
                    "scenario_green_above_historical_grid_max":
                        f"above_hist_green_max_{pp}pp",
                    "scenario_support_class":
                        f"support_class_{pp}pp",
                }
            )
        )

        year_merge = year_merge.merge(
            subset,
            on=GRID_COL,
            how="left",
            validate="one_to_one",
        )

    year_gdf = grid.merge(
        year_merge,
        on=GRID_COL,
        how="left",
        validate="one_to_one",
    )

    geo_path = (
        MAP_DIR
        / f"29a_v3_green_response_{year}_wgs84.geojson"
    )

    year_gdf.to_file(
        geo_path,
        driver="GeoJSON",
    )

    geojson_files.append(
        geo_path
    )

    for pp in SCENARIO_INCREASE_PP:
        column = (
            f"cooling_benefit_{pp}pp_c"
        )

        values = year_gdf[
            column
        ].to_numpy(
            dtype=float
        )

        max_abs = float(
            np.nanmax(
                np.abs(
                    values
                )
            )
        )

        if max_abs == 0:
            max_abs = 1e-6

        fig, ax = plt.subplots(
            figsize=(
                10,
                10,
            )
        )

        year_gdf.plot(
            column=column,
            ax=ax,
            legend=True,
            edgecolor="none",
            cmap="RdBu",
            vmin=-max_abs,
            vmax=max_abs,
            legend_kwds={
                "label":
                    "Estimated cooling benefit (°C)\n"
                    "positive=cooling; negative=warming",
                "shrink":
                    0.72,
            },
        )

        if boundary is not None:
            boundary.boundary.plot(
                ax=ax,
                linewidth=1.4,
                color="black",
            )
        else:
            year_gdf.dissolve().boundary.plot(
                ax=ax,
                linewidth=1.4,
                color="black",
            )

        ax.set_title(
            (
                f"Kaduwela V3 Green-Cover Response — {year}\n"
                f"+{pp} percentage points green cover"
            ),
            fontsize=14,
            pad=12,
        )

        ax.set_axis_off()

        plt.tight_layout()

        map_path = (
            FIGURE_DIR
            / f"29a_v3_green_response_{year}_{pp}pp.png"
        )

        plt.savefig(
            map_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        map_files.append(
            map_path
        )

    print(
        f"{year}: saved GeoJSON + {len(SCENARIO_INCREASE_PP)} response maps"
    )


# =============================================================================
# 11. STRICT AUDIT
# =============================================================================

section(
    "11. STRICT STEP-29A V3 AUDIT"
)

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
            "step28_audit",
        "status":
            "PASS",
        "details":
            "Step-28 strict audit verified fully PASS.",
    },
    {
        "check":
            "production_model_hash",
        "status":
            "PASS",
        "details":
            EXPECTED_HASHES[
                "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
            ],
    },
    {
        "check":
            "historical_panel_rows",
        "status":
            "PASS"
            if len(
                panel
            )
            == EXPECTED_HIST_ROWS
            else "FAIL",
        "details":
            f"actual={len(panel)}, expected={EXPECTED_HIST_ROWS}",
    },
    {
        "check":
            "historical_panel_balanced",
        "status":
            "PASS",
        "details":
            f"{EXPECTED_GRIDS} grids x {len(HIST_YEARS)} years",
    },
    {
        "check":
            "primary_model_specification",
        "status":
            "PASS",
        "details":
            (
                "TWFE: LST_C ~ green_mask + NDBI + NDWI + "
                "Albedo + NightLights + grid FE + year FE"
            ),
    },
    {
        "check":
            "clustered_standard_errors",
        "status":
            "PASS",
        "details":
            "Clustered by grid_id.",
    },
    {
        "check":
            "primary_green_coefficient_finite",
        "status":
            "PASS"
            if np.isfinite(
                beta_green
            )
            else "FAIL",
        "details":
            f"beta={beta_green:+.12f}",
    },
    {
        "check":
            "primary_green_ci_finite",
        "status":
            "PASS"
            if (
                np.isfinite(
                    green_ci_lower
                )
                and np.isfinite(
                    green_ci_upper
                )
            )
            else "FAIL",
        "details":
            (
                f"ci95=[{green_ci_lower:+.12f}, "
                f"{green_ci_upper:+.12f}]"
            ),
    },
    {
        "check":
            "coefficient_sign_forced",
        "status":
            "PASS",
        "details":
            "No. Estimated sign is retained exactly.",
    },
    {
        "check":
            "coefficient_stability_windows_tuned",
        "status":
            "PASS",
        "details":
            "No. Windows were pre-defined and only used as robustness evidence.",
    },
    {
        "check":
            "first_difference_used_for_tuning",
        "status":
            "PASS",
        "details":
            "No. It is a pre-defined sensitivity model only.",
    },
    {
        "check":
            "step28_baseline_reproduction",
        "status":
            "PASS"
            if step28_reproduction_error
            <= PREDICTION_REPRO_TOL
            else "FAIL",
        "details":
            f"max_raw_error_c={step28_reproduction_error:.12e}",
    },
    {
        "check":
            "lightgbm_used_as_intervention_engine",
        "status":
            "PASS",
        "details":
            "No. LightGBM is used only to verify/preserve Step-28 baseline forecasts.",
    },
    {
        "check":
            "scenario_rows",
        "status":
            "PASS"
            if len(
                scenario_df
            )
            == expected_scenario_rows
            else "FAIL",
        "details":
            f"actual={len(scenario_df)}, expected={expected_scenario_rows}",
    },
    {
        "check":
            "scenario_green_bounds",
        "status":
            "PASS"
            if (
                scenario_df[
                    "scenario_green_mask"
                ].min()
                >= GREEN_MIN
                - 1e-12
                and scenario_df[
                    "scenario_green_mask"
                ].max()
                <= GREEN_MAX
                + 1e-12
            )
            else "FAIL",
        "details":
            (
                f"min={scenario_df['scenario_green_mask'].min():.6f}, "
                f"max={scenario_df['scenario_green_mask'].max():.6f}"
            ),
    },
    {
        "check":
            "persistent_effect_accumulated_each_year",
        "status":
            "PASS",
        "details":
            "No. The estimated level response is applied once as a persistent offset each year.",
    },
    {
        "check":
            "scenario_empirical_support_flagged",
        "status":
            "PASS",
        "details":
            "Per-grid historical max and annual-change support classes saved.",
    },
    {
        "check":
            "response_ci_labeled_as_total_forecast_uncertainty",
        "status":
            "PASS",
        "details":
            "No. It is explicitly response-coefficient uncertainty only.",
    },
    {
        "check":
            "causal_guarantee_claim",
        "status":
            "PASS",
        "details":
            "No. Observational within-grid response is not claimed as causal proof.",
    },
    {
        "check":
            "step28_files_modified",
        "status":
            "PASS",
        "details":
            "No. All V3 outputs are written separately.",
    },
    {
        "check":
            "step22b_model_retrained",
        "status":
            "PASS",
        "details":
            "No.",
    },
    {
        "check":
            "geojson_count",
        "status":
            "PASS"
            if len(
                geojson_files
            )
            == len(
                FUTURE_YEARS
            )
            else "FAIL",
        "details":
            f"saved={len(geojson_files)}, expected={len(FUTURE_YEARS)}",
    },
    {
        "check":
            "map_count",
        "status":
            "PASS"
            if len(
                map_files
            )
            == len(
                FUTURE_YEARS
            )
            * len(
                SCENARIO_INCREASE_PP
            )
            else "FAIL",
        "details":
            (
                f"saved={len(map_files)}, "
                f"expected={len(FUTURE_YEARS) * len(SCENARIO_INCREASE_PP)}"
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
        "One or more Step-29A V3 audit checks failed."
    )


# =============================================================================
# 12. METHODOLOGY + REPORT
# =============================================================================

section(
    "12. SAVE STEP-29A V3 METHODOLOGY + REPORT"
)

methodology = {
    "step":
        "29A V3",
    "version":
        "dedicated_green_response_twfe_v1",
    "historical_period":
        [
            2015,
            2025,
        ],
    "historical_rows":
        int(
            len(
                panel
            )
        ),
    "grids":
        int(
            panel[
                GRID_COL
            ].nunique()
        ),
    "primary_response_model":
        {
            "type":
                "Two-way fixed-effects linear panel regression",
            "outcome":
                TARGET,
            "treatment":
                TREATMENT,
            "controls":
                PRIMARY_CONTROLS,
            "grid_fixed_effects":
                True,
            "year_fixed_effects":
                True,
            "implementation":
                "Exact two-way demeaning for balanced panel",
            "standard_errors":
                "Cluster-robust by grid_id",
            "green_beta_c_per_fraction":
                beta_green,
            "green_beta_c_per_10pp":
                beta_green
                * 0.10,
            "green_ci95_lower_per_10pp":
                green_ci_lower
                * 0.10,
            "green_ci95_upper_per_10pp":
                green_ci_upper
                * 0.10,
            "green_p_value_clustered":
                p_green,
            "evidence_status":
                green_status,
            "within_r2":
                primary_result[
                    "within_r2"
                ],
        },
    "predefined_stability_windows":
        [
            [
                start,
                end,
            ]
            for start, end in STABILITY_WINDOWS
        ],
    "coefficient_same_sign_all_windows":
        same_sign_all_windows,
    "first_difference_sensitivity":
        {
            "green_beta_c_per_10pp":
                fd_beta
                * 0.10,
            "green_ci95_lower_per_10pp":
                fd_lo
                * 0.10,
            "green_ci95_upper_per_10pp":
                fd_hi
                * 0.10,
            "evidence_status":
                fd_status,
            "same_sign_as_primary":
                primary_fd_same_sign,
        },
    "future_baseline":
        "Frozen Step-28 grid LST predictions",
    "future_scenarios_percentage_points":
        SCENARIO_INCREASE_PP,
    "scenario_formula":
        {
            "scenario_green":
                "clip(baseline future green_mask + requested increase, 0, 1)",
            "estimated_response":
                "TWFE green beta * actual green increase fraction",
            "scenario_lst":
                "Step-28 baseline predicted LST + estimated response",
            "cooling_benefit":
                "-estimated response",
        },
    "persistent_interpretation":
        (
            "Scenario is a maintained level difference from baseline. "
            "The estimated level response is applied once each future year, "
            "not accumulated repeatedly."
        ),
    "support_checks":
        {
            "per_grid_historical_green_max":
                True,
            "historical_annual_green_change_quantiles":
                {
                    "q50":
                        q50_abs_change,
                    "q90":
                        q90_abs_change,
                    "q95":
                        q95_abs_change,
                    "q99":
                        q99_abs_change,
                },
        },
    "uncertainty_note":
        (
            "Scenario response CI represents coefficient estimation uncertainty "
            "only. Total future forecast uncertainty remains Step 30."
        ),
    "causal_limit":
        (
            "Observational TWFE reduces time-invariant grid confounding and "
            "common year shocks but does not prove causal cooling."
        ),
    "v1_v2_status":
        (
            "V1 and V2 retained as diagnostic predictive-model perturbation "
            "experiments; V3 is the dedicated response-model candidate."
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
    "STEP 29A V3 - DEDICATED GREEN-COVER RESPONSE MODEL",
    "=" * 118,
    "",
    "PRIMARY MODEL",
    "-" * 118,
    (
        "Two-way fixed-effects panel regression using 2015-2025 historical "
        "grid-year data, with grid and year fixed effects and grid-clustered SE."
    ),
    (
        "Controls: NDBI, NDWI, Albedo, NightLights."
    ),
    "",
    "GREEN RESPONSE",
    "-" * 118,
    (
        f"Coefficient per +10 percentage points green cover: "
        f"{beta_green * 0.10:+.6f} °C"
    ),
    (
        f"95% cluster-robust CI per +10 pp: "
        f"[{green_ci_lower * 0.10:+.6f}, "
        f"{green_ci_upper * 0.10:+.6f}] °C"
    ),
    (
        f"Clustered p-value: {p_green:.8g}"
    ),
    (
        f"Evidence status: {green_status}"
    ),
    "",
    "TIME-WINDOW STABILITY",
    "-" * 118,
    stability_df.to_string(
        index=False
    ),
    "",
    "FIRST-DIFFERENCE SENSITIVITY",
    "-" * 118,
    (
        f"Green coefficient per +10 pp: {fd_beta * 0.10:+.6f} °C"
    ),
    (
        f"95% CI: [{fd_lo * 0.10:+.6f}, {fd_hi * 0.10:+.6f}] °C"
    ),
    (
        f"Evidence status: {fd_status}"
    ),
    (
        f"Same sign as primary TWFE: {primary_fd_same_sign}"
    ),
    "",
    "SCENARIO SUMMARY",
    "-" * 118,
    summary.to_string(
        index=False
    ),
    "",
    "EMPIRICAL SUPPORT",
    "-" * 118,
    support_summary.to_string(
        index=False
    ),
    "",
    "INTERPRETATION",
    "-" * 118,
    (
        "V3 estimates a historical within-grid response rather than directly "
        "perturbing the frozen LightGBM prediction model."
    ),
    (
        "Grid fixed effects remove time-invariant location differences; year "
        "fixed effects remove Kaduwela-wide annual shocks."
    ),
    (
        "This strengthens intervention interpretation, but observational data "
        "still do not prove causality."
    ),
    (
        "Positive cooling_benefit_c means the estimated historical green "
        "response lowers LST relative to the Step-28 baseline."
    ),
    (
        "Response-model 95% CI is not total forecast uncertainty. Step 30 "
        "will handle future prediction uncertainty."
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
# 13. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "13. REPRODUCIBILITY HASHES"
)

hash_files = [
    BASE_PATH,
    FEATURE_LIST_PATH,
    MODEL_PATH,
    FUTURE_FEATURE_PATH,
    FUTURE_PRED_PATH,
    STEP28_AUDIT_PATH,
    GRID_WGS84_PATH,
    COEFFICIENT_PATH,
    STABILITY_PATH,
    FD_COEFFICIENT_PATH,
    MODEL_DIAGNOSTIC_PATH,
    SCENARIO_PATH,
    PRIMARY_SCENARIO_PATH,
    SUMMARY_PATH,
    SUPPORT_PATH,
    AUDIT_PATH,
    METHOD_PATH,
    REPORT_PATH,
    STABILITY_FIGURE_PATH,
    SCENARIO_FIGURE_PATH,
]

for year in FUTURE_YEARS:
    for pp in SCENARIO_INCREASE_PP:
        hash_files.append(
            SCENARIO_DIR
            / f"29a_v3_green_response_{pp}pp_{year}.csv"
        )

    hash_files.append(
        MAP_DIR
        / f"29a_v3_green_response_{year}_wgs84.geojson"
    )

for path in map_files:
    hash_files.append(
        path
    )

unique_hash_files = []
seen = set()

for path in hash_files:
    path = Path(
        path
    )

    key = str(
        path
    )

    if key not in seen:
        seen.add(
            key
        )

        unique_hash_files.append(
            path
        )

hash_rows = []

for path in unique_hash_files:
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
    "STEP 29A V3 COMPLETED SUCCESSFULLY"
)

print(
    "Dedicated green-cover response model completed.\n"
    f"Historical panel: {len(panel):,} rows, "
    f"{EXPECTED_GRIDS:,} grids, 2015-2025.\n"
    "Primary response model: two-way fixed effects with grid-clustered SE.\n"
    f"Green response per +10 pp: {beta_green * 0.10:+.6f} °C.\n"
    f"95% CI per +10 pp: "
    f"[{green_ci_lower * 0.10:+.6f}, "
    f"{green_ci_upper * 0.10:+.6f}] °C.\n"
    f"Evidence status: {green_status}.\n"
    f"Primary vs first-difference same sign: {primary_fd_same_sign}.\n"
    f"Same coefficient sign across pre-defined windows: {same_sign_all_windows}.\n"
    f"Scenario rows: {len(scenario_df):,}.\n"
    "Step-28 future LST remains the frozen baseline.\n"
    "LightGBM was NOT used as the intervention-response estimator.\n"
    "The response model does not force green cover to produce cooling.\n"
    "Scenario response CI is coefficient uncertainty only, not total forecast uncertainty.\n"
    "Observational panel evidence is not claimed as causal proof.\n"
    "\nNEXT: review V3 coefficient sign, CI, stability, first-difference sensitivity, "
    "and empirical-support flags before freezing Step 29A or starting Step 29B."
)
