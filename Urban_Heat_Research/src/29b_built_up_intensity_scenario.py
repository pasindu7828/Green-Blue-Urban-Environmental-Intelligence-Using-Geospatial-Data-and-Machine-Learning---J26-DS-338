"""
STEP 29B - BUILT-UP INTENSITY WHAT-IF SCENARIO
Urban Heat Research - Kaduwela, Sri Lanka

FINAL STEP-29B DESIGN
---------------------
The current historical dataset contains:

    building_mask = binary 0/1 static GIS indicator

It does NOT contain a continuous, time-varying OSM building-density percentage.
Therefore it is methodologically incorrect to expose building_mask as a
"reduce building density by 5/10/20 percentage points" slider.

Instead, Step 29B uses:

    NDBI = continuous built-up intensity proxy

for the built-up scenario.

This is consistent with the project's methodological use of NDBI for built-up
intensity, while building_mask remains a static contextual feature.

SCENARIO ENGINE
---------------
The frontend scenario uses the already-frozen Step-22B LightGBM.

No new ML model is trained for scenario inference.

For each selected grid:
1) Start from Step-28 future baseline.
2) Reduce NDBI persistently from 2026 onward.
3) Rebuild only the NDBI-dependent frozen features.
4) Run the same frozen LightGBM.
5) Calculate:

       modeled_lst_change
         = scenario_lst - baseline_lst

       cooling_benefit
         = baseline_lst - scenario_lst

No batch re-centering is performed because each row is an independent local
counterfactual.

SCENARIO MAGNITUDES
-------------------
Rather than invent arbitrary NDBI changes, LOW / MODERATE / HIGH magnitudes
are derived from historical absolute within-grid annual NDBI changes:

    LOW       = 50th percentile
    MODERATE  = 75th percentile
    HIGH      = 90th percentile

This is determined BEFORE looking at scenario LST responses.

HISTORICAL RESPONSE QA
----------------------
A separate two-way fixed-effects diagnostic estimates whether higher NDBI has
historically been associated with higher/lower LST within grids after removing
grid fixed effects and year fixed effects.

This QA model is NOT used as the frontend scenario engine.

INTERPRETATION
--------------
Outputs are model-based what-if sensitivity, not guaranteed causal physical
cooling/heating.

Do not call NDBI change "building-density percentage change".
Use wording such as:

    Built-up intensity scenario
    Development-pressure scenario
    Modeled LST change

If a true OSM building-coverage fraction is generated later, a literal
building-density slider can be added as a separate future extension.
"""

from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

try:
    import geopandas as gpd
except ImportError as exc:
    raise ImportError(
        "\ngeopandas is required for Step 29B.\n"
        "Install it with:\n"
        "    pip install geopandas\n"
    ) from exc


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET = "LST_C"
TREATMENT = "NDBI"

HIST_YEARS = list(range(2015, 2026))
HISTORY_SEED_YEARS = [2023, 2024, 2025]
FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]

EXPECTED_GRIDS = 2194
EXPECTED_HIST_ROWS = EXPECTED_GRIDS * len(HIST_YEARS)
EXPECTED_FUTURE_ROWS = EXPECTED_GRIDS * len(FUTURE_YEARS)
EXPECTED_FEATURES = 93

# Scenario magnitudes are derived from data, not manually tuned.
SCENARIO_QUANTILES = {
    "LOW": 0.50,
    "MODERATE": 0.75,
    "HIGH": 0.90,
}
PRIMARY_SCENARIO = "MODERATE"

FORMULA_TOL = 1e-8
PREDICTION_REPRO_TOL = 1e-10
UNCHANGED_FEATURE_TOL = 1e-12

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
MAP_DIR = ROOT / "outputs" / "maps"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [
    SCENARIO_DIR,
    MAP_DIR,
    FIGURE_DIR,
    REPORT_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

ALL_SCENARIOS_PATH = (
    SCENARIO_DIR
    / "29b_built_up_intensity_scenarios_2026_2030.csv"
)

PRIMARY_SCENARIO_PATH = (
    SCENARIO_DIR
    / "29b_built_up_intensity_primary_moderate_2026_2030.csv"
)

FEATURE_EXPORT_PATH = (
    SCENARIO_DIR
    / "29b_built_up_intensity_scenario_features_2026_2030.csv"
)

SUMMARY_PATH = (
    SCENARIO_DIR
    / "29b_built_up_intensity_scenario_summary.csv"
)

SCENARIO_LEVEL_PATH = (
    REPORT_DIR
    / "29b_built_up_intensity_scenario_levels.csv"
)

BASELINE_RECON_PATH = (
    REPORT_DIR
    / "29b_ndbi_feature_reconstruction_audit.csv"
)

REPRESENTATION_AUDIT_PATH = (
    REPORT_DIR
    / "29b_built_up_representation_audit.csv"
)

RESPONSE_QA_PATH = (
    REPORT_DIR
    / "29b_ndbi_historical_response_qa.csv"
)

AUDIT_PATH = (
    REPORT_DIR
    / "29b_built_up_intensity_scenario_audit.csv"
)

METHOD_PATH = (
    REPORT_DIR
    / "29b_built_up_intensity_methodology.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "29b_built_up_intensity_report.txt"
)

HASH_PATH = (
    REPORT_DIR
    / "29b_built_up_intensity_hashes.csv"
)

SUMMARY_FIGURE_PATH = (
    FIGURE_DIR
    / "29b_built_up_intensity_scenario_summary.png"
)

RESPONSE_FIGURE_PATH = (
    FIGURE_DIR
    / "29b_ndbi_historical_response_qa.png"
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
            f"{label} has no status column."
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


def detect_feature_column(feature_df):
    for conventional in [
        "feature",
        "feature_name",
        "predictor",
    ]:
        if conventional in feature_df.columns:
            values = (
                feature_df[conventional]
                .dropna()
                .astype(str)
                .tolist()
            )

            if len(values) == EXPECTED_FEATURES:
                return conventional

    for column in feature_df.columns:
        values = (
            feature_df[column]
            .dropna()
            .astype(str)
            .tolist()
        )

        if (
            len(values) == EXPECTED_FEATURES
            and len(set(values)) == EXPECTED_FEATURES
        ):
            return column

    raise RuntimeError(
        "Could not identify frozen 93-feature column."
    )


def sample_std_three(values):
    return float(
        np.std(
            np.asarray(values, dtype=float),
            ddof=1,
        )
    )


def slope_three(oldest, middle, newest):
    return float(
        (newest - oldest)
        / 2.0
    )


def build_ndbi_temporal_features(
    historical_ndbi,
    future_ndbi,
):
    """
    Rebuild strict-past NDBI temporal features for 2026-2030.
    """

    hist_lookup = {
        (str(row[GRID_COL]), int(row[YEAR_COL])):
            float(row[TREATMENT])
        for _, row in historical_ndbi.iterrows()
    }

    future_lookup = {
        (str(row[GRID_COL]), int(row[YEAR_COL])):
            float(row[TREATMENT])
        for _, row in future_ndbi.iterrows()
    }

    grid_ids = sorted(
        future_ndbi[GRID_COL]
        .astype(str)
        .unique()
        .tolist()
    )

    rows = []

    for grid_id in grid_ids:
        series = {}

        for year in HISTORY_SEED_YEARS:
            key = (
                grid_id,
                year,
            )

            if key not in hist_lookup:
                raise RuntimeError(
                    f"Missing historical NDBI for {key}"
                )

            series[year] = hist_lookup[key]

        for year in FUTURE_YEARS:
            key = (
                grid_id,
                year,
            )

            if key not in future_lookup:
                raise RuntimeError(
                    f"Missing future NDBI for {key}"
                )

            current = future_lookup[key]

            prev1 = series[year - 1]
            prev2 = series[year - 2]
            prev3 = series[year - 3]

            prior = [
                prev3,
                prev2,
                prev1,
            ]

            rows.append(
                {
                    GRID_COL:
                        grid_id,
                    YEAR_COL:
                        year,
                    TREATMENT:
                        current,
                    "NDBI_lag1":
                        prev1,
                    "NDBI_lag2":
                        prev2,
                    "NDBI_hist3_mean":
                        float(
                            np.mean(prior)
                        ),
                    "NDBI_hist3_std":
                        sample_std_three(
                            prior
                        ),
                    "NDBI_hist3_slope":
                        slope_three(
                            prev3,
                            prev2,
                            prev1,
                        ),
                    "NDBI_delta1":
                        float(
                            current - prev1
                        ),
                }
            )

            series[year] = current

    result = (
        pd.DataFrame(rows)
        .sort_values(
            [
                YEAR_COL,
                GRID_COL,
            ]
        )
        .reset_index(drop=True)
    )

    return result


def two_way_demean(df, columns):
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
                df[GRID_COL]
            )
            .transform("mean")
        )

        year_mean = (
            values.groupby(
                df[YEAR_COL]
            )
            .transform("mean")
        )

        result[column] = (
            values
            - entity_mean
            - year_mean
            + float(values.mean())
        )

    return result


def fit_ndbi_twfe(
    historical,
    years,
    label,
):
    """
    Historical response QA only.
    Not used for frontend scenario inference.
    """

    controls = [
        "green_mask",
        "NDWI",
        "Albedo",
        "NightLights",
    ]

    sub = (
        historical[
            historical[YEAR_COL].isin(years)
        ]
        .copy()
        .sort_values(
            [
                GRID_COL,
                YEAR_COL,
            ]
        )
        .reset_index(drop=True)
    )

    expected_rows = (
        EXPECTED_GRIDS
        * len(years)
    )

    if len(sub) != expected_rows:
        raise RuntimeError(
            f"{label}: rows={len(sub)}, expected={expected_rows}"
        )

    cols = [
        TARGET,
        TREATMENT,
        *controls,
    ]

    transformed = two_way_demean(
        sub,
        cols,
    )

    y = transformed[
        TARGET
    ].to_numpy(dtype=float)

    X = transformed[
        [
            TREATMENT,
            *controls,
        ]
    ].copy()

    rank = int(
        np.linalg.matrix_rank(
            X.to_numpy(dtype=float)
        )
    )

    if rank != X.shape[1]:
        raise RuntimeError(
            f"{label}: rank deficient TWFE design."
        )

    fit = sm.OLS(
        y,
        X,
        hasconst=False,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                sub[GRID_COL]
                .astype(str)
                .to_numpy(),
            "use_correction":
                True,
        },
    )

    ci = fit.conf_int(
        alpha=0.05
    ).loc[TREATMENT]

    return {
        "variant":
            label,
        "rows":
            int(fit.nobs),
        "ndbi_beta_c_per_1_0":
            float(
                fit.params[TREATMENT]
            ),
        "response_c_per_plus_0_05_ndbi":
            float(
                fit.params[TREATMENT]
                * 0.05
            ),
        "clustered_se":
            float(
                fit.bse[TREATMENT]
            ),
        "p_value":
            float(
                fit.pvalues[TREATMENT]
            ),
        "ci95_lower_per_plus_0_05":
            float(
                ci[0] * 0.05
            ),
        "ci95_upper_per_plus_0_05":
            float(
                ci[1] * 0.05
            ),
        "within_r2":
            float(
                fit.rsquared
            ),
    }


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
            "WARNING: Kaduwela boundary KML could not be loaded."
        )

        print(
            f"  {type(exc).__name__}: {exc}"
        )

        return None


# =============================================================================
# 0. VERIFY FROZEN INPUTS
# =============================================================================

section(
    "STEP 29B - BUILT-UP INTENSITY WHAT-IF SCENARIO"
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
    "Step-28 future feature matrix",
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


# =============================================================================
# 1. BUILT-UP REPRESENTATION AUDIT
# =============================================================================

section(
    "1. BUILT-UP REPRESENTATION AUDIT"
)

base = pd.read_csv(
    BASE_PATH,
    dtype={
        GRID_COL: str,
        "area_name": str,
    },
)

base[YEAR_COL] = pd.to_numeric(
    base[YEAR_COL],
    errors="raise",
).astype(int)

for column in [
    TARGET,
    TREATMENT,
    "building_mask",
    "green_mask",
    "NDWI",
    "Albedo",
    "NightLights",
]:
    if column not in base.columns:
        raise RuntimeError(
            f"Required column missing: {column}"
        )

    base[column] = pd.to_numeric(
        base[column],
        errors="raise",
    )

historical = (
    base[
        base[YEAR_COL].isin(
            HIST_YEARS
        )
    ]
    .copy()
    .sort_values(
        [
            GRID_COL,
            YEAR_COL,
        ]
    )
    .reset_index(drop=True)
)

if len(historical) != EXPECTED_HIST_ROWS:
    raise RuntimeError(
        f"Historical rows={len(historical)}, "
        f"expected={EXPECTED_HIST_ROWS}"
    )

building_unique = sorted(
    historical[
        "building_mask"
    ].unique().tolist()
)

building_binary = (
    set(building_unique)
    .issubset(
        {
            0.0,
            1.0,
        }
    )
)

building_grid_nunique = (
    historical.groupby(
        GRID_COL
    )[
        "building_mask"
    ]
    .nunique()
)

building_grids_changing = int(
    (
        building_grid_nunique
        > 1
    ).sum()
)

building_static = (
    building_grids_changing
    == 0
)

representation_rows = [
    {
        "variable":
            "building_mask",
        "observed_min":
            float(
                historical[
                    "building_mask"
                ].min()
            ),
        "observed_max":
            float(
                historical[
                    "building_mask"
                ].max()
            ),
        "unique_values":
            len(building_unique),
        "grids_changing_through_time":
            building_grids_changing,
        "binary_like":
            building_binary,
        "scenario_role":
            (
                "STATIC CONTEXT ONLY - not used as a percentage-density slider"
            ),
    },
    {
        "variable":
            "NDBI",
        "observed_min":
            float(
                historical[
                    TREATMENT
                ].min()
            ),
        "observed_max":
            float(
                historical[
                    TREATMENT
                ].max()
            ),
        "unique_values":
            int(
                historical[
                    TREATMENT
                ].nunique()
            ),
        "grids_changing_through_time":
            int(
                (
                    historical.groupby(
                        GRID_COL
                    )[
                        TREATMENT
                    ]
                    .nunique()
                    > 1
                ).sum()
            ),
        "binary_like":
            False,
        "scenario_role":
            "PRIMARY CONTINUOUS BUILT-UP INTENSITY PROXY",
    },
]

representation_df = pd.DataFrame(
    representation_rows
)

representation_df.to_csv(
    REPRESENTATION_AUDIT_PATH,
    index=False,
)

print(
    representation_df.to_string(
        index=False
    )
)

if not building_binary:
    raise RuntimeError(
        "building_mask is not binary as expected. "
        "Review representation before continuing."
    )

if not building_static:
    raise RuntimeError(
        "building_mask unexpectedly varies through time. "
        "Review before continuing."
    )

print(
    "\nDecision:"
)

print(
    "  building_mask = static binary context; NOT a building-density percentage."
)

print(
    "  NDBI          = Step-29B continuous built-up intensity scenario variable."
)


# =============================================================================
# 2. DATA-DERIVED SCENARIO MAGNITUDES
# =============================================================================

section(
    "2. DERIVE LOW / MODERATE / HIGH NDBI REDUCTION MAGNITUDES"
)

hist_sorted = historical[
    [
        GRID_COL,
        YEAR_COL,
        TREATMENT,
    ]
].copy()

hist_sorted[
    "previous_year"
] = (
    hist_sorted.groupby(
        GRID_COL
    )[
        YEAR_COL
    ]
    .shift(1)
)

hist_sorted[
    "d_ndbi"
] = (
    hist_sorted.groupby(
        GRID_COL
    )[
        TREATMENT
    ]
    .diff()
)

transitions = (
    hist_sorted[
        hist_sorted[YEAR_COL]
        .between(
            2016,
            2025,
        )
    ]
    .copy()
)

if not (
    transitions[YEAR_COL]
    - transitions["previous_year"]
    == 1
).all():
    raise RuntimeError(
        "Historical NDBI differences are not true adjacent-year transitions."
    )

abs_changes = (
    transitions[
        "d_ndbi"
    ]
    .abs()
    .dropna()
)

scenario_levels = []

for label, q in SCENARIO_QUANTILES.items():
    magnitude = float(
        abs_changes.quantile(q)
    )

    if magnitude <= 0:
        raise RuntimeError(
            f"Scenario magnitude for {label} is not positive."
        )

    scenario_levels.append(
        {
            "scenario_level":
                label,
            "historical_abs_change_quantile":
                q,
            "ndbi_reduction_magnitude":
                magnitude,
        }
    )

scenario_level_df = pd.DataFrame(
    scenario_levels
)

scenario_level_df.to_csv(
    SCENARIO_LEVEL_PATH,
    index=False,
)

print(
    scenario_level_df.to_string(
        index=False
    )
)

scenario_magnitude_lookup = {
    row[
        "scenario_level"
    ]:
        float(
            row[
                "ndbi_reduction_magnitude"
            ]
        )
    for _, row in scenario_level_df.iterrows()
}


# =============================================================================
# 3. HISTORICAL RESPONSE QA
# =============================================================================

section(
    "3. HISTORICAL NDBI RESPONSE QA - NOT THE FRONTEND ENGINE"
)

response_rows = []

response_rows.append(
    fit_ndbi_twfe(
        historical=historical,
        years=HIST_YEARS,
        label="TWFE_ALL_2015_2025",
    )
)

response_rows.append(
    fit_ndbi_twfe(
        historical=historical,
        years=[
            year
            for year in HIST_YEARS
            if year != 2017
        ],
        label="TWFE_EXCLUDE_2017",
    )
)

response_df = pd.DataFrame(
    response_rows
)

response_df.to_csv(
    RESPONSE_QA_PATH,
    index=False,
)

print(
    response_df.to_string(
        index=False
    )
)

historical_response_same_sign = (
    np.sign(
        response_df.iloc[0][
            "ndbi_beta_c_per_1_0"
        ]
    )
    == np.sign(
        response_df.iloc[1][
            "ndbi_beta_c_per_1_0"
        ]
    )
)

print(
    f"\nHistorical NDBI response sign stable after excluding 2017: "
    f"{historical_response_same_sign}"
)


# =============================================================================
# 4. LOAD FROZEN FUTURE BASELINE
# =============================================================================

section(
    "4. LOAD FROZEN STEP-28 FUTURE BASELINE"
)

feature_df = pd.read_csv(
    FEATURE_LIST_PATH
)

feature_col = detect_feature_column(
    feature_df
)

ordered_features = (
    feature_df[
        feature_col
    ]
    .dropna()
    .astype(str)
    .tolist()
)

if len(ordered_features) != EXPECTED_FEATURES:
    raise RuntimeError(
        f"Feature count={len(ordered_features)}, "
        f"expected={EXPECTED_FEATURES}"
    )

future = pd.read_csv(
    FUTURE_FEATURE_PATH,
    dtype={
        GRID_COL: str,
        "area_name": str,
    },
)

pred = pd.read_csv(
    FUTURE_PRED_PATH,
    dtype={
        GRID_COL: str,
        "area_name": str,
    },
)

for frame, label in [
    (
        future,
        "future feature matrix",
    ),
    (
        pred,
        "future predictions",
    ),
]:
    frame[YEAR_COL] = pd.to_numeric(
        frame[YEAR_COL],
        errors="raise",
    ).astype(int)

    if len(frame) != EXPECTED_FUTURE_ROWS:
        raise RuntimeError(
            f"{label}: rows={len(frame)}, "
            f"expected={EXPECTED_FUTURE_ROWS}"
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
    .reset_index(drop=True)
)

pred = (
    pred
    .sort_values(
        [
            YEAR_COL,
            GRID_COL,
        ]
    )
    .reset_index(drop=True)
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
        "Future feature/prediction keys do not align."
    )

missing_features = [
    feature
    for feature in ordered_features
    if feature not in future.columns
]

if missing_features:
    raise RuntimeError(
        "Step-28 future matrix missing frozen features:\n"
        + str(missing_features)
    )

X_base = future[
    ordered_features
].copy()

for feature in ordered_features:
    X_base[feature] = pd.to_numeric(
        X_base[feature],
        errors="raise",
    )

if X_base.isna().any().any():
    raise RuntimeError(
        "Step-28 frozen matrix contains missing values."
    )

if not np.isfinite(
    X_base.to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Step-28 frozen matrix contains non-finite values."
    )

if TREATMENT not in X_base.columns:
    raise RuntimeError(
        "NDBI is not present in the frozen feature matrix."
    )

print(
    f"Future rows     : {len(future):,}"
)

print(
    f"Unique grids    : {future[GRID_COL].nunique():,}"
)

print(
    f"Frozen features : {len(ordered_features)}"
)


# =============================================================================
# 5. RECONSTRUCT BASELINE NDBI DEPENDENCIES
# =============================================================================

section(
    "5. RECONSTRUCT BASELINE NDBI-DEPENDENT FROZEN FEATURES"
)

historical_seed = (
    historical[
        historical[YEAR_COL]
        .isin(
            HISTORY_SEED_YEARS
        )
    ][
        [
            GRID_COL,
            YEAR_COL,
            TREATMENT,
        ]
    ]
    .copy()
)

baseline_future_ndbi = future[
    [
        GRID_COL,
        YEAR_COL,
    ]
].copy()

baseline_future_ndbi[
    TREATMENT
] = X_base[
    TREATMENT
].to_numpy(dtype=float)

baseline_rebuilt = build_ndbi_temporal_features(
    historical_ndbi=
        historical_seed,
    future_ndbi=
        baseline_future_ndbi,
)

if not baseline_rebuilt[
    [
        GRID_COL,
        YEAR_COL,
    ]
].equals(
    future[
        [
            GRID_COL,
            YEAR_COL,
        ]
    ]
):
    raise RuntimeError(
        "Rebuilt NDBI baseline keys do not align."
    )

candidate_temporal = [
    "NDBI_lag1",
    "NDBI_lag2",
    "NDBI_hist3_mean",
    "NDBI_hist3_std",
    "NDBI_hist3_slope",
    "NDBI_delta1",
]

candidate_current = [
    "ndbi_minus_ndvi",
    "ndbi_x_vegetation_deficit",
    "ndbi_x_building_mask",
    "NDBI_nbr_contrast",
]

retained_temporal = [
    feature
    for feature in candidate_temporal
    if feature in X_base.columns
]

retained_current = [
    feature
    for feature in candidate_current
    if feature in X_base.columns
]

print(
    "Retained NDBI temporal dependencies:"
)

for feature in retained_temporal:
    print(
        f"  {feature}"
    )

print(
    "\nRetained current NDBI dependencies:"
)

for feature in retained_current:
    print(
        f"  {feature}"
    )

recon_rows = []

for feature in retained_temporal:
    max_error = float(
        np.max(
            np.abs(
                X_base[
                    feature
                ].to_numpy(dtype=float)
                - baseline_rebuilt[
                    feature
                ].to_numpy(dtype=float)
            )
        )
    )

    recon_rows.append(
        {
            "feature":
                feature,
            "status":
                (
                    "PASS"
                    if max_error
                    <= FORMULA_TOL
                    else "FAIL"
                ),
            "max_abs_error":
                max_error,
            "tolerance":
                FORMULA_TOL,
        }
    )

baseline_ndbi = X_base[
    TREATMENT
].to_numpy(dtype=float)

if "ndbi_minus_ndvi" in X_base.columns:
    expected = (
        baseline_ndbi
        - X_base[
            "NDVI"
        ].to_numpy(dtype=float)
    )

    max_error = float(
        np.max(
            np.abs(
                X_base[
                    "ndbi_minus_ndvi"
                ].to_numpy(dtype=float)
                - expected
            )
        )
    )

    recon_rows.append(
        {
            "feature":
                "ndbi_minus_ndvi",
            "status":
                (
                    "PASS"
                    if max_error
                    <= FORMULA_TOL
                    else "FAIL"
                ),
            "max_abs_error":
                max_error,
            "tolerance":
                FORMULA_TOL,
        }
    )

if "ndbi_x_vegetation_deficit" in X_base.columns:
    if "green_mask" not in X_base.columns:
        raise RuntimeError(
            "green_mask required to verify ndbi_x_vegetation_deficit."
        )

    expected = (
        baseline_ndbi
        * (
            1.0
            - X_base[
                "green_mask"
            ].to_numpy(dtype=float)
        )
    )

    max_error = float(
        np.max(
            np.abs(
                X_base[
                    "ndbi_x_vegetation_deficit"
                ].to_numpy(dtype=float)
                - expected
            )
        )
    )

    recon_rows.append(
        {
            "feature":
                "ndbi_x_vegetation_deficit",
            "status":
                (
                    "PASS"
                    if max_error
                    <= FORMULA_TOL
                    else "FAIL"
                ),
            "max_abs_error":
                max_error,
            "tolerance":
                FORMULA_TOL,
        }
    )

if "ndbi_x_building_mask" in X_base.columns:
    if "building_mask" not in X_base.columns:
        raise RuntimeError(
            "building_mask required to verify ndbi_x_building_mask."
        )

    expected = (
        baseline_ndbi
        * X_base[
            "building_mask"
        ].to_numpy(dtype=float)
    )

    max_error = float(
        np.max(
            np.abs(
                X_base[
                    "ndbi_x_building_mask"
                ].to_numpy(dtype=float)
                - expected
            )
        )
    )

    recon_rows.append(
        {
            "feature":
                "ndbi_x_building_mask",
            "status":
                (
                    "PASS"
                    if max_error
                    <= FORMULA_TOL
                    else "FAIL"
                ),
            "max_abs_error":
                max_error,
            "tolerance":
                FORMULA_TOL,
        }
    )

if "NDBI_nbr_contrast" in X_base.columns:
    if "NDBI_nbr_mean" in X_base.columns:
        baseline_nbr_mean = X_base[
            "NDBI_nbr_mean"
        ].to_numpy(dtype=float)
    else:
        baseline_nbr_mean = (
            baseline_ndbi
            - X_base[
                "NDBI_nbr_contrast"
            ].to_numpy(dtype=float)
        )

    expected = (
        baseline_ndbi
        - baseline_nbr_mean
    )

    max_error = float(
        np.max(
            np.abs(
                X_base[
                    "NDBI_nbr_contrast"
                ].to_numpy(dtype=float)
                - expected
            )
        )
    )

    recon_rows.append(
        {
            "feature":
                "NDBI_nbr_contrast",
            "status":
                (
                    "PASS"
                    if max_error
                    <= FORMULA_TOL
                    else "FAIL"
                ),
            "max_abs_error":
                max_error,
            "tolerance":
                FORMULA_TOL,
        }
    )

recon_df = pd.DataFrame(
    recon_rows
)

recon_df.to_csv(
    BASELINE_RECON_PATH,
    index=False,
)

print(
    recon_df.to_string(
        index=False
    )
)

if recon_df.empty:
    raise RuntimeError(
        "No NDBI-dependent frozen features were verified."
    )

if (
    recon_df["status"]
    != "PASS"
).any():
    raise RuntimeError(
        "Baseline NDBI feature reconstruction failed."
    )


# =============================================================================
# 6. REPRODUCE FROZEN STEP-28 MODEL
# =============================================================================

section(
    "6. REPRODUCE FROZEN STEP-28 LIGHTGBM"
)

model = joblib.load(
    MODEL_PATH
)

baseline_raw_model = np.asarray(
    model.predict(
        X_base
    ),
    dtype=float,
)

baseline_raw_saved = pd.to_numeric(
    pred[
        "raw_spatial_anomaly_prediction_c"
    ],
    errors="raise",
).to_numpy(dtype=float)

baseline_lst = pd.to_numeric(
    pred[
        "predicted_lst_c"
    ],
    errors="raise",
).to_numpy(dtype=float)

baseline_reproduction_error = float(
    np.max(
        np.abs(
            baseline_raw_model
            - baseline_raw_saved
        )
    )
)

print(
    f"Max reproduction error: "
    f"{baseline_reproduction_error:.12e} °C"
)

if (
    baseline_reproduction_error
    > PREDICTION_REPRO_TOL
):
    raise RuntimeError(
        "Frozen Step-28 model output was not reproduced."
    )

print(
    "PASS: Step-28 frozen baseline reproduced."
)


# =============================================================================
# 7. RUN PERSISTENT LOCAL BUILT-UP INTENSITY REDUCTION SCENARIOS
# =============================================================================

section(
    "7. RUN PERSISTENT LOCAL NDBI REDUCTION SCENARIOS"
)

global_ndbi_min = float(
    historical[
        TREATMENT
    ].min()
)

global_ndbi_max = float(
    historical[
        TREATMENT
    ].max()
)

scenario_rows = []
feature_exports = []
change_audit_rows = []

allowed_changed_features = {
    TREATMENT,
    *retained_temporal,
    *retained_current,
}

# NDBI_nbr_mean stays baseline for an independent local scenario.
allowed_changed_features.discard(
    "NDBI_nbr_mean"
)

for level in [
    "LOW",
    "MODERATE",
    "HIGH",
]:
    magnitude = scenario_magnitude_lookup[
        level
    ]

    print(
        f"\nRunning {level} built-up intensity reduction:"
    )

    print(
        f"  Requested NDBI reduction magnitude = {magnitude:.6f}"
    )

    scenario_future_ndbi = (
        baseline_future_ndbi
        .copy()
    )

    scenario_future_ndbi[
        "baseline_NDBI"
    ] = scenario_future_ndbi[
        TREATMENT
    ]

    scenario_future_ndbi[
        TREATMENT
    ] = np.clip(
        scenario_future_ndbi[
            "baseline_NDBI"
        ].to_numpy(dtype=float)
        - magnitude,
        global_ndbi_min,
        global_ndbi_max,
    )

    scenario_future_ndbi[
        "actual_ndbi_reduction"
    ] = (
        scenario_future_ndbi[
            "baseline_NDBI"
        ]
        - scenario_future_ndbi[
            TREATMENT
        ]
    )

    scenario_future_ndbi[
        "lower_support_bound_limited"
    ] = (
        scenario_future_ndbi[
            "actual_ndbi_reduction"
        ]
        < magnitude
        - 1e-12
    )

    rebuilt = build_ndbi_temporal_features(
        historical_ndbi=
            historical_seed,
        future_ndbi=
            scenario_future_ndbi[
                [
                    GRID_COL,
                    YEAR_COL,
                    TREATMENT,
                ]
            ],
    )

    if not rebuilt[
        [
            GRID_COL,
            YEAR_COL,
        ]
    ].equals(
        future[
            [
                GRID_COL,
                YEAR_COL,
            ]
        ]
    ):
        raise RuntimeError(
            f"{level}: rebuilt scenario keys do not align."
        )

    X_scenario = X_base.copy()

    scenario_ndbi = rebuilt[
        TREATMENT
    ].to_numpy(dtype=float)

    X_scenario[
        TREATMENT
    ] = scenario_ndbi

    for feature in retained_temporal:
        X_scenario[
            feature
        ] = rebuilt[
            feature
        ].to_numpy(dtype=float)

    if "ndbi_minus_ndvi" in X_scenario.columns:
        X_scenario[
            "ndbi_minus_ndvi"
        ] = (
            scenario_ndbi
            - X_scenario[
                "NDVI"
            ].to_numpy(dtype=float)
        )

    if "ndbi_x_vegetation_deficit" in X_scenario.columns:
        X_scenario[
            "ndbi_x_vegetation_deficit"
        ] = (
            scenario_ndbi
            * (
                1.0
                - X_scenario[
                    "green_mask"
                ].to_numpy(dtype=float)
            )
        )

    if "ndbi_x_building_mask" in X_scenario.columns:
        X_scenario[
            "ndbi_x_building_mask"
        ] = (
            scenario_ndbi
            * X_scenario[
                "building_mask"
            ].to_numpy(dtype=float)
        )

    if "NDBI_nbr_contrast" in X_scenario.columns:
        if "NDBI_nbr_mean" in X_base.columns:
            local_nbr_mean = X_base[
                "NDBI_nbr_mean"
            ].to_numpy(dtype=float)
        else:
            local_nbr_mean = (
                baseline_ndbi
                - X_base[
                    "NDBI_nbr_contrast"
                ].to_numpy(dtype=float)
            )

        X_scenario[
            "NDBI_nbr_contrast"
        ] = (
            scenario_ndbi
            - local_nbr_mean
        )

    if X_scenario.isna().any().any():
        raise RuntimeError(
            f"{level}: scenario feature matrix contains missing values."
        )

    if not np.isfinite(
        X_scenario.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            f"{level}: scenario feature matrix contains non-finite values."
        )

    changed_features = []

    for feature in ordered_features:
        max_change = float(
            np.max(
                np.abs(
                    X_scenario[
                        feature
                    ].to_numpy(dtype=float)
                    - X_base[
                        feature
                    ].to_numpy(dtype=float)
                )
            )
        )

        if max_change > UNCHANGED_FEATURE_TOL:
            changed_features.append(
                feature
            )

    unexpected = sorted(
        set(changed_features)
        - allowed_changed_features
    )

    if unexpected:
        raise RuntimeError(
            f"{level}: unexpected frozen features changed: {unexpected}"
        )

    change_audit_rows.append(
        {
            "scenario_level":
                level,
            "changed_feature_count":
                len(changed_features),
            "changed_features":
                "; ".join(
                    sorted(changed_features)
                ),
            "unexpected_feature_count":
                len(unexpected),
            "status":
                "PASS",
        }
    )

    scenario_raw = np.asarray(
        model.predict(
            X_scenario
        ),
        dtype=float,
    )

    local_delta = (
        scenario_raw
        - baseline_raw_model
    )

    scenario_lst = (
        baseline_lst
        + local_delta
    )

    cooling_benefit = (
        baseline_lst
        - scenario_lst
    )

    meta = (
        scenario_future_ndbi
        .sort_values(
            [
                YEAR_COL,
                GRID_COL,
            ]
        )
        .reset_index(drop=True)
    )

    for i in range(
        EXPECTED_FUTURE_ROWS
    ):
        delta = float(
            local_delta[i]
        )

        benefit = float(
            cooling_benefit[i]
        )

        if delta < -1e-12:
            direction = "cooling"
        elif delta > 1e-12:
            direction = "warming"
        else:
            direction = "no_change"

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
                "scenario_id":
                    f"built_up_intensity_reduce_{level.lower()}",
                "scenario_level":
                    level,
                "scenario_variable":
                    "NDBI",
                "scenario_interpretation":
                    "built_up_intensity_reduction",
                "requested_ndbi_reduction":
                    magnitude,
                "baseline_ndbi":
                    float(
                        meta.at[
                            i,
                            "baseline_NDBI",
                        ]
                    ),
                "scenario_ndbi":
                    float(
                        meta.at[
                            i,
                            TREATMENT,
                        ]
                    ),
                "actual_ndbi_reduction":
                    float(
                        meta.at[
                            i,
                            "actual_ndbi_reduction",
                        ]
                    ),
                "support_bound_limited":
                    bool(
                        meta.at[
                            i,
                            "lower_support_bound_limited",
                        ]
                    ),
                "baseline_predicted_lst_c":
                    float(
                        baseline_lst[i]
                    ),
                "scenario_predicted_lst_c":
                    float(
                        scenario_lst[i]
                    ),
                "modeled_lst_change_c":
                    delta,
                "cooling_benefit_c":
                    benefit,
                "scenario_effect_direction":
                    direction,
            }
        )

    feature_export = pd.concat(
        [
            future[
                [
                    GRID_COL,
                    YEAR_COL,
                    "area_name",
                ]
            ].reset_index(drop=True),
            pd.DataFrame(
                {
                    "scenario_level":
                        level,
                    "requested_ndbi_reduction":
                        magnitude,
                },
                index=np.arange(
                    EXPECTED_FUTURE_ROWS
                ),
            ),
            X_scenario.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    feature_exports.append(
        feature_export
    )

    print(
        f"  Mean actual NDBI reduction : "
        f"{meta['actual_ndbi_reduction'].mean():.6f}"
    )

    print(
        f"  Support-bound-limited rows : "
        f"{int(meta['lower_support_bound_limited'].sum()):,}"
    )

    print(
        f"  Mean modeled LST change    : "
        f"{local_delta.mean():+.6f} °C"
    )

    print(
        f"  Mean cooling benefit       : "
        f"{cooling_benefit.mean():+.6f} °C"
    )


scenario_df = pd.DataFrame(
    scenario_rows
)

feature_export_df = pd.concat(
    feature_exports,
    ignore_index=True,
)

expected_scenario_rows = (
    EXPECTED_FUTURE_ROWS
    * len(SCENARIO_QUANTILES)
)

if len(scenario_df) != expected_scenario_rows:
    raise RuntimeError(
        f"Scenario rows={len(scenario_df)}, "
        f"expected={expected_scenario_rows}"
    )

scenario_df.to_csv(
    ALL_SCENARIOS_PATH,
    index=False,
)

feature_export_df.to_csv(
    FEATURE_EXPORT_PATH,
    index=False,
)

scenario_df[
    scenario_df[
        "scenario_level"
    ]
    == PRIMARY_SCENARIO
].to_csv(
    PRIMARY_SCENARIO_PATH,
    index=False,
)

for year in FUTURE_YEARS:
    for level in [
        "LOW",
        "MODERATE",
        "HIGH",
    ]:
        scenario_df[
            (
                scenario_df[
                    YEAR_COL
                ]
                == year
            )
            & (
                scenario_df[
                    "scenario_level"
                ]
                == level
            )
        ].to_csv(
            SCENARIO_DIR
            / (
                f"29b_built_up_intensity_"
                f"{level.lower()}_{year}.csv"
            ),
            index=False,
        )

print(
    f"\nSaved: {ALL_SCENARIOS_PATH}"
)


# =============================================================================
# 8. SCENARIO SUMMARY
# =============================================================================

section(
    "8. BUILT-UP INTENSITY SCENARIO SUMMARY"
)

summary = (
    scenario_df
    .groupby(
        [
            YEAR_COL,
            "scenario_level",
        ],
        as_index=False,
    )
    .agg(
        grid_count=(
            GRID_COL,
            "size",
        ),
        requested_ndbi_reduction=(
            "requested_ndbi_reduction",
            "first",
        ),
        mean_actual_ndbi_reduction=(
            "actual_ndbi_reduction",
            "mean",
        ),
        support_bound_limited_count=(
            "support_bound_limited",
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
        mean_modeled_lst_change_c=(
            "modeled_lst_change_c",
            "mean",
        ),
        median_modeled_lst_change_c=(
            "modeled_lst_change_c",
            "median",
        ),
        min_modeled_lst_change_c=(
            "modeled_lst_change_c",
            "min",
        ),
        max_modeled_lst_change_c=(
            "modeled_lst_change_c",
            "max",
        ),
        mean_cooling_benefit_c=(
            "cooling_benefit_c",
            "mean",
        ),
    )
)

direction_counts = (
    scenario_df
    .pivot_table(
        index=[
            YEAR_COL,
            "scenario_level",
        ],
        columns=
            "scenario_effect_direction",
        values=
            GRID_COL,
        aggfunc=
            "count",
        fill_value=
            0,
    )
    .reset_index()
)

for column in [
    "cooling",
    "warming",
    "no_change",
]:
    if column not in direction_counts.columns:
        direction_counts[column] = 0

direction_counts = direction_counts.rename(
    columns={
        "cooling":
            "cooling_grid_count",
        "warming":
            "warming_grid_count",
        "no_change":
            "no_change_grid_count",
    }
)

summary = summary.merge(
    direction_counts,
    on=[
        YEAR_COL,
        "scenario_level",
    ],
    how="left",
    validate="one_to_one",
)

summary[
    "cooling_grid_pct"
] = (
    summary[
        "cooling_grid_count"
    ]
    / summary[
        "grid_count"
    ]
    * 100.0
)

summary[
    "warming_grid_pct"
] = (
    summary[
        "warming_grid_count"
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

print(
    summary.to_string(
        index=False
    )
)


# =============================================================================
# 9. FRONTEND-READY GEOJSON + MAPS
# =============================================================================

section(
    "9. SAVE FRONTEND-READY GEOJSON + MAPS"
)

grid = gpd.read_file(
    GRID_WGS84_PATH
)

grid[GRID_COL] = grid[
    GRID_COL
].astype(str)

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
            pred[YEAR_COL] == year
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

    merged = year_base.copy()

    for level in [
        "LOW",
        "MODERATE",
        "HIGH",
    ]:
        sub = (
            scenario_df[
                (
                    scenario_df[
                        YEAR_COL
                    ]
                    == year
                )
                & (
                    scenario_df[
                        "scenario_level"
                    ]
                    == level
                )
            ][
                [
                    GRID_COL,
                    "requested_ndbi_reduction",
                    "scenario_ndbi",
                    "scenario_predicted_lst_c",
                    "modeled_lst_change_c",
                    "cooling_benefit_c",
                    "scenario_effect_direction",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "requested_ndbi_reduction":
                        f"requested_ndbi_reduction_{level.lower()}",
                    "scenario_ndbi":
                        f"scenario_ndbi_{level.lower()}",
                    "scenario_predicted_lst_c":
                        f"scenario_lst_{level.lower()}_c",
                    "modeled_lst_change_c":
                        f"modeled_lst_change_{level.lower()}_c",
                    "cooling_benefit_c":
                        f"cooling_benefit_{level.lower()}_c",
                    "scenario_effect_direction":
                        f"effect_{level.lower()}",
                }
            )
        )

        merged = merged.merge(
            sub,
            on=GRID_COL,
            how="left",
            validate="one_to_one",
        )

    gdf = grid.merge(
        merged,
        on=GRID_COL,
        how="left",
        validate="one_to_one",
    )

    geo_path = (
        MAP_DIR
        / f"29b_built_up_intensity_{year}_wgs84.geojson"
    )

    gdf.to_file(
        geo_path,
        driver="GeoJSON",
    )

    geojson_files.append(
        geo_path
    )

    for level in [
        "LOW",
        "MODERATE",
        "HIGH",
    ]:
        column = (
            f"modeled_lst_change_{level.lower()}_c"
        )

        values = gdf[
            column
        ].to_numpy(dtype=float)

        max_abs = float(
            np.nanmax(
                np.abs(values)
            )
        )

        if max_abs == 0:
            max_abs = 1e-6

        fig, ax = plt.subplots(
            figsize=(10, 10)
        )

        gdf.plot(
            column=column,
            ax=ax,
            legend=True,
            edgecolor="none",
            cmap="RdBu_r",
            vmin=-max_abs,
            vmax=max_abs,
            legend_kwds={
                "label":
                    "Modeled LST change (°C)\n"
                    "negative = cooling; positive = warming",
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
            gdf.dissolve().boundary.plot(
                ax=ax,
                linewidth=1.4,
                color="black",
            )

        ax.set_title(
            (
                f"Kaduwela Built-up Intensity Scenario — {year}\n"
                f"{level.title()} persistent NDBI reduction"
            ),
            fontsize=14,
            pad=12,
        )

        ax.set_axis_off()

        plt.tight_layout()

        map_path = (
            FIGURE_DIR
            / (
                f"29b_built_up_intensity_lst_change_"
                f"{year}_{level.lower()}.png"
            )
        )

        plt.savefig(
            map_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        map_files.append(
            map_path
        )

    print(
        f"{year}: saved GeoJSON + 3 scenario maps"
    )


# =============================================================================
# 10. SUMMARY FIGURES
# =============================================================================

section(
    "10. SAVE SUMMARY FIGURES"
)

level_order = [
    "LOW",
    "MODERATE",
    "HIGH",
]

overall_summary = (
    scenario_df
    .groupby(
        "scenario_level",
        as_index=False,
    )
    .agg(
        mean_modeled_lst_change_c=(
            "modeled_lst_change_c",
            "mean",
        )
    )
)

overall_summary[
    "scenario_level"
] = pd.Categorical(
    overall_summary[
        "scenario_level"
    ],
    categories=
        level_order,
    ordered=
        True,
)

overall_summary = (
    overall_summary
    .sort_values(
        "scenario_level"
    )
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.plot(
    overall_summary[
        "scenario_level"
    ].astype(str),
    overall_summary[
        "mean_modeled_lst_change_c"
    ],
    marker="o",
)

ax.axhline(
    0.0,
    linewidth=1.0,
    color="black",
)

ax.set_xlabel(
    "Built-up intensity reduction scenario"
)

ax.set_ylabel(
    "Mean modeled LST change (°C)"
)

ax.set_title(
    "Step 29B Built-up Intensity Scenario Sensitivity"
)

ax.grid(
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    SUMMARY_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


fig, ax = plt.subplots(
    figsize=(8, 6)
)

x = np.arange(
    len(response_df)
)

y = (
    response_df[
        "response_c_per_plus_0_05_ndbi"
    ].to_numpy(dtype=float)
)

lower = (
    y
    - response_df[
        "ci95_lower_per_plus_0_05"
    ].to_numpy(dtype=float)
)

upper = (
    response_df[
        "ci95_upper_per_plus_0_05"
    ].to_numpy(dtype=float)
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
        "2015–2025",
        "Exclude 2017",
    ]
)

ax.set_ylabel(
    "Historical LST response per +0.05 NDBI (°C)"
)

ax.set_title(
    "Historical NDBI Response QA"
)

ax.grid(
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    RESPONSE_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    f"Saved: {SUMMARY_FIGURE_PATH}"
)

print(
    f"Saved: {RESPONSE_FIGURE_PATH}"
)


# =============================================================================
# 11. STRICT AUDIT
# =============================================================================

section(
    "11. STRICT STEP-29B AUDIT"
)

change_audit_df = pd.DataFrame(
    change_audit_rows
)

cooling_identity_error = float(
    np.max(
        np.abs(
            scenario_df[
                "cooling_benefit_c"
            ].to_numpy(dtype=float)
            + scenario_df[
                "modeled_lst_change_c"
            ].to_numpy(dtype=float)
        )
    )
)

audit_rows = [
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
            "building_mask_binary",
        "status":
            "PASS"
            if building_binary
            else "FAIL",
        "details":
            f"unique={building_unique}",
    },
    {
        "check":
            "building_mask_static",
        "status":
            "PASS"
            if building_static
            else "FAIL",
        "details":
            f"grids_changing={building_grids_changing}",
    },
    {
        "check":
            "literal_building_density_slider_used",
        "status":
            "PASS",
        "details":
            "No. building_mask is static binary; NDBI is used as continuous built-up intensity proxy.",
    },
    {
        "check":
            "scenario_levels_data_derived",
        "status":
            "PASS",
        "details":
            "LOW=q50, MODERATE=q75, HIGH=q90 of historical |annual delta NDBI|.",
    },
    {
        "check":
            "baseline_ndbi_feature_reconstruction",
        "status":
            "PASS"
            if (
                recon_df[
                    "status"
                ]
                == "PASS"
            ).all()
            else "FAIL",
        "details":
            f"features_checked={len(recon_df)}",
    },
    {
        "check":
            "baseline_raw_prediction_reproduction",
        "status":
            "PASS"
            if baseline_reproduction_error
            <= PREDICTION_REPRO_TOL
            else "FAIL",
        "details":
            f"max_error_c={baseline_reproduction_error:.12e}",
    },
    {
        "check":
            "only_intended_frozen_features_changed",
        "status":
            "PASS"
            if (
                change_audit_df[
                    "unexpected_feature_count"
                ]
                == 0
            ).all()
            else "FAIL",
        "details":
            "; ".join(
                change_audit_df[
                    "changed_features"
                ].astype(str).tolist()
            ),
    },
    {
        "check":
            "scenario_rows",
        "status":
            "PASS"
            if len(scenario_df)
            == expected_scenario_rows
            else "FAIL",
        "details":
            f"actual={len(scenario_df)}, expected={expected_scenario_rows}",
    },
    {
        "check":
            "scenario_outputs_finite",
        "status":
            "PASS"
            if np.isfinite(
                scenario_df[
                    [
                        "scenario_predicted_lst_c",
                        "modeled_lst_change_c",
                        "cooling_benefit_c",
                    ]
                ].to_numpy(dtype=float)
            ).all()
            else "FAIL",
        "details":
            f"rows={len(scenario_df)}",
    },
    {
        "check":
            "cooling_identity",
        "status":
            "PASS"
            if cooling_identity_error
            <= PREDICTION_REPRO_TOL
            else "FAIL",
        "details":
            f"max_error_c={cooling_identity_error:.12e}",
    },
    {
        "check":
            "local_neighbor_semantics",
        "status":
            "PASS",
        "details":
            "NDBI neighbour mean remains baseline; selected-grid contrast changes if retained.",
    },
    {
        "check":
            "scenario_recentering",
        "status":
            "PASS",
        "details":
            "Not applied; local effect uses same-grid raw anomaly delta.",
    },
    {
        "check":
            "model_retraining",
        "status":
            "PASS",
        "details":
            "None. Frozen Step-22B LightGBM used for scenario inference.",
    },
    {
        "check":
            "historical_response_model_used_as_frontend_engine",
        "status":
            "PASS",
        "details":
            "No. TWFE is QA evidence only.",
    },
    {
        "check":
            "causal_claim",
        "status":
            "PASS",
        "details":
            "No. Output is model-based what-if sensitivity.",
    },
    {
        "check":
            "geojson_count",
        "status":
            "PASS"
            if len(geojson_files)
            == len(FUTURE_YEARS)
            else "FAIL",
        "details":
            f"saved={len(geojson_files)}, expected={len(FUTURE_YEARS)}",
    },
    {
        "check":
            "map_count",
        "status":
            "PASS"
            if len(map_files)
            == (
                len(FUTURE_YEARS)
                * len(SCENARIO_QUANTILES)
            )
            else "FAIL",
        "details":
            (
                f"saved={len(map_files)}, "
                f"expected={len(FUTURE_YEARS) * len(SCENARIO_QUANTILES)}"
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
        "One or more Step-29B audit checks failed."
    )


# =============================================================================
# 12. METHODOLOGY + REPORT
# =============================================================================

section(
    "12. SAVE STEP-29B METHODOLOGY + REPORT"
)

methodology = {
    "step":
        "29B",
    "name":
        "Built-up intensity what-if scenario",
    "frontend_scenario_engine":
        "Frozen Step-22B LightGBM",
    "new_ml_model_for_frontend":
        False,
    "building_mask_status":
        {
            "type":
                "static binary GIS indicator",
            "used_as_density_percentage_slider":
                False,
            "reason":
                (
                    "No within-grid temporal variation and only values 0/1; "
                    "therefore it is not a continuous building-density percentage."
                ),
        },
    "scenario_variable":
        {
            "name":
                "NDBI",
            "meaning":
                "continuous built-up intensity proxy",
            "frontend_wording":
                "Built-up intensity / development pressure",
            "not_worded_as":
                "building-density percentage",
        },
    "scenario_levels":
        scenario_level_df.to_dict(
            orient="records"
        ),
    "scenario_direction":
        "persistent NDBI reduction from 2026 through 2030",
    "historical_response_qa":
        response_df.to_dict(
            orient="records"
        ),
    "response_qa_used_for_scenario_prediction":
        False,
    "model_retraining":
        False,
    "step28_baseline_modified":
        False,
    "local_neighbor_policy":
        (
            "Neighbours remain Step-28 baseline; focal-grid NDBI contrast "
            "changes if the contrast feature is retained."
        ),
    "scenario_effect_formula":
        (
            "scenario LST = Step-28 baseline LST + "
            "(scenario raw anomaly - baseline raw anomaly)"
        ),
    "frontend_output_fields":
        [
            "baseline_ndbi",
            "scenario_ndbi",
            "baseline_predicted_lst_c",
            "scenario_predicted_lst_c",
            "modeled_lst_change_c",
            "cooling_benefit_c",
            "scenario_effect_direction",
        ],
    "interpretation_warning":
        (
            "Model-based sensitivity only. NDBI is a spectral built-up "
            "intensity proxy, not a literal percentage of building density, "
            "and the LST change is not a guaranteed causal outcome."
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
    "STEP 29B - BUILT-UP INTENSITY WHAT-IF SCENARIO",
    "=" * 118,
    "",
    "REPRESENTATION DECISION",
    "-" * 118,
    (
        "building_mask is binary and static through time. It is therefore "
        "retained as contextual information and is NOT used as a continuous "
        "building-density percentage slider."
    ),
    (
        "NDBI is used as the continuous built-up intensity proxy for Step 29B."
    ),
    "",
    "DATA-DERIVED SCENARIO LEVELS",
    "-" * 118,
    scenario_level_df.to_string(
        index=False
    ),
    "",
    "HISTORICAL RESPONSE QA",
    "-" * 118,
    response_df.to_string(
        index=False
    ),
    "",
    "SCENARIO SUMMARY",
    "-" * 118,
    summary.to_string(
        index=False
    ),
    "",
    "FRONTEND INTERPRETATION",
    "-" * 118,
    (
        "The planner selects a grid/year and a LOW, MODERATE or HIGH built-up "
        "intensity reduction scenario."
    ),
    (
        "The system displays baseline NDBI, scenario NDBI, baseline predicted "
        "LST, scenario predicted LST, modeled LST change and direction."
    ),
    (
        "Use 'built-up intensity' or 'development pressure'; do not label "
        "NDBI reduction as a literal percentage change in building density."
    ),
    "",
    "SCIENTIFIC LIMITATION",
    "-" * 118,
    (
        "This is a frozen-model what-if sensitivity analysis, not a guaranteed "
        "causal physical temperature response."
    ),
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(
    f"Saved: {METHOD_PATH}"
)

print(
    f"Saved: {REPORT_PATH}"
)


# =============================================================================
# 13. HASHES
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
    ALL_SCENARIOS_PATH,
    PRIMARY_SCENARIO_PATH,
    FEATURE_EXPORT_PATH,
    SUMMARY_PATH,
    SCENARIO_LEVEL_PATH,
    BASELINE_RECON_PATH,
    REPRESENTATION_AUDIT_PATH,
    RESPONSE_QA_PATH,
    AUDIT_PATH,
    METHOD_PATH,
    REPORT_PATH,
    SUMMARY_FIGURE_PATH,
    RESPONSE_FIGURE_PATH,
]

for year in FUTURE_YEARS:
    hash_files.append(
        MAP_DIR
        / f"29b_built_up_intensity_{year}_wgs84.geojson"
    )

    for level in [
        "LOW",
        "MODERATE",
        "HIGH",
    ]:
        hash_files.append(
            SCENARIO_DIR
            / (
                f"29b_built_up_intensity_"
                f"{level.lower()}_{year}.csv"
            )
        )

for path in map_files:
    hash_files.append(path)

unique_files = []
seen = set()

for path in hash_files:
    path = Path(path)

    key = str(path)

    if key not in seen:
        seen.add(key)
        unique_files.append(path)

hash_rows = []

for path in unique_files:
    digest = sha256_file(path)

    hash_rows.append(
        {
            "file":
                str(path),
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
    "STEP 29B COMPLETED SUCCESSFULLY"
)

print(
    "Built-up intensity scenario simulation completed.\n"
    "building_mask remained a static binary context feature.\n"
    "NDBI was used as the continuous built-up-intensity proxy.\n"
    "LOW/MODERATE/HIGH scenario magnitudes were derived from historical "
    "absolute annual NDBI changes.\n"
    "The intervention is persistent from 2026 through 2030.\n"
    "NDBI temporal/dependent frozen features were rebuilt consistently.\n"
    "The frozen Step-22B LightGBM was used without retraining.\n"
    "Step-28 future baseline predictions were not modified.\n"
    "Historical TWFE response analysis is QA evidence only, not the frontend engine.\n"
    "Frontend can show baseline LST, scenario LST and modeled LST change.\n"
    "Do NOT label NDBI scenario magnitude as building-density percentage.\n"
    "No causal or guaranteed cooling/heating claim is made.\n"
    "\nNEXT: review Step-29B terminal results before freezing 29B "
    "and moving to Step 29C combined scenario."
)
