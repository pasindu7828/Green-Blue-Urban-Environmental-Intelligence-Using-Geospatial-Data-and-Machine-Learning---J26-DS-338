
"""
STEP 29D - BLUE-SPACE / WATER-WETNESS INTENSITY WHAT-IF SCENARIO
Urban Heat Research - Kaduwela, Sri Lanka

WHY THIS DESIGN
---------------
The current dataset contains two different blue-space related variables:

1) NDWI
   - continuous
   - changes through time
   - represents spectral water/wetness intensity
   - suitable for a model-based what-if sensitivity scenario

2) dist_water
   - static structural distance-to-water feature
   - does not change through time in the current dataset
   - cannot honestly be changed by a simple slider unless a NEW water-body
     geometry is explicitly added and distances are spatially recomputed

Therefore Step 29D uses NDWI as the intervention variable and keeps dist_water
unchanged as structural spatial context.

IMPORTANT FRONTEND WORDING
--------------------------
Use:
    Blue-space / water-wetness intensity scenario
    LOW / MODERATE / HIGH restoration intensity
    NDWI increase
    Modeled LST change

Do NOT use:
    "+X% water coverage"
    "water distance reduced by X%"
    "new lake added"

because the present dataset does not contain a continuous water-cover
percentage and this step does not create new water-body geometry.

SCENARIO LEVELS
---------------
LOW / MODERATE / HIGH NDWI-increase magnitudes are derived BEFORE looking at
scenario LST responses, using the historical distribution of POSITIVE adjacent
annual within-grid NDWI changes:

    LOW       = 50th percentile
    MODERATE  = 75th percentile
    HIGH      = 90th percentile

The scenario is persistent from 2026 through 2030. Each future baseline NDWI is
increased by the selected data-derived magnitude and clipped to the historical
NDWI support maximum.

SCENARIO ENGINE
---------------
No new frontend ML model is trained.

For each scenario:
1) Start from the frozen Step-28 future feature matrix.
2) Increase NDWI.
3) Rebuild all retained NDWI temporal-history features.
4) Keep NDWI neighbour mean at the Step-28 baseline because the intervention
   is local to the selected grid.
5) Rebuild NDWI neighbour contrast if retained.
6) Keep dist_water unchanged.
7) Run the SAME frozen Step-22B LightGBM.
8) Compute:

       modeled_lst_change
         = scenario_lst - baseline_lst

       cooling_benefit
         = baseline_lst - scenario_lst

No batch recentering is applied.

HISTORICAL QA
-------------
A separate two-way fixed-effects analysis checks the historical within-grid
association between NDWI and LST, with grid/year fixed effects and grid-clustered
standard errors. This QA is NOT used as the frontend scenario engine.

All scenario results are model-based statistical sensitivity, not guaranteed
causal physical outcomes.
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
        "geopandas is required for Step 29D. Install with: pip install geopandas"
    ) from exc


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET = "LST_C"
TREATMENT = "NDWI"

HIST_YEARS = list(range(2015, 2026))
HISTORY_SEED_YEARS = [2023, 2024, 2025]
FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]

EXPECTED_GRIDS = 2194
EXPECTED_HIST_ROWS = EXPECTED_GRIDS * len(HIST_YEARS)
EXPECTED_FUTURE_ROWS = EXPECTED_GRIDS * len(FUTURE_YEARS)
EXPECTED_FEATURES = 93

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

BASE_PATH = ROOT / "data" / "processed" / "kaduwela_preprocessed_base.csv"
FEATURE_LIST_PATH = ROOT / "outputs" / "reports" / "21_final_feature_list.csv"
MODEL_PATH = ROOT / "models" / "final" / "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"

FUTURE_FEATURE_PATH = ROOT / "outputs" / "forecasting" / "28_future_model_features_2026_2030.csv"
FUTURE_PRED_PATH = ROOT / "outputs" / "forecasting" / "28_future_grid_lst_predictions_2026_2030.csv"
STEP28_AUDIT_PATH = ROOT / "outputs" / "reports" / "28_future_prediction_audit.csv"

GRID_WGS84_PATH = ROOT / "data" / "processed" / "kaduwela_grid_geometry_wgs84.geojson"
BOUNDARY_KML_PATH = ROOT / "data" / "raw" / "boundary" / "kaduwela_kmc_boundary.kml"

SCENARIO_DIR = ROOT / "outputs" / "scenarios"
REPORT_DIR = ROOT / "outputs" / "reports"
MAP_DIR = ROOT / "outputs" / "maps"
FIGURE_DIR = ROOT / "outputs" / "figures"

for folder in [SCENARIO_DIR, REPORT_DIR, MAP_DIR, FIGURE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

ALL_SCENARIOS_PATH = (
    SCENARIO_DIR
    / "29d_blue_space_ndwi_scenarios_2026_2030.csv"
)

PRIMARY_SCENARIO_PATH = (
    SCENARIO_DIR
    / "29d_blue_space_ndwi_primary_moderate_2026_2030.csv"
)

FEATURE_EXPORT_PATH = (
    SCENARIO_DIR
    / "29d_blue_space_ndwi_scenario_features_2026_2030.csv"
)

SUMMARY_PATH = (
    SCENARIO_DIR
    / "29d_blue_space_ndwi_scenario_summary.csv"
)

SCENARIO_LEVEL_PATH = (
    REPORT_DIR
    / "29d_blue_space_ndwi_scenario_levels.csv"
)

REPRESENTATION_AUDIT_PATH = (
    REPORT_DIR
    / "29d_blue_space_representation_audit.csv"
)

BASELINE_RECON_PATH = (
    REPORT_DIR
    / "29d_ndwi_feature_reconstruction_audit.csv"
)

RESPONSE_QA_PATH = (
    REPORT_DIR
    / "29d_ndwi_historical_response_qa.csv"
)

FEATURE_CHANGE_AUDIT_PATH = (
    REPORT_DIR
    / "29d_blue_space_feature_change_audit.csv"
)

AUDIT_PATH = (
    REPORT_DIR
    / "29d_blue_space_scenario_audit.csv"
)

METHOD_PATH = (
    REPORT_DIR
    / "29d_blue_space_scenario_methodology.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "29d_blue_space_scenario_report.txt"
)

HASH_PATH = (
    REPORT_DIR
    / "29d_blue_space_scenario_hashes.csv"
)

SUMMARY_FIGURE_PATH = (
    FIGURE_DIR
    / "29d_blue_space_mean_lst_change_by_scenario.png"
)

RESPONSE_FIGURE_PATH = (
    FIGURE_DIR
    / "29d_ndwi_historical_response_qa.png"
)


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 136)
    print(title)
    print("=" * 136)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def verify_hash(path, expected, label):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual  : {actual}"
        )

    print(f"PASS hash: {label} = {actual}")
    return actual


def require_all_pass(path, label):
    audit = pd.read_csv(path)

    if "status" not in audit.columns:
        raise RuntimeError(f"{label} has no status column.")

    bad = audit[
        audit["status"].astype(str).str.upper() != "PASS"
    ]

    if not bad.empty:
        print(bad.to_string(index=False))
        raise RuntimeError(f"{label} is not fully PASS.")

    print(f"PASS: {label} is fully PASS.")


def detect_feature_column(feature_df):
    for conventional in ["feature", "feature_name", "predictor"]:
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


def build_ndwi_temporal_features(
    historical_ndwi,
    future_ndwi,
):
    """
    Rebuild strict-past NDWI temporal features for 2026-2030.

    For each future year:
    - lag1/lag2 use previous scenario values once the intervention has begun
    - hist3 features use previous scenario history
    - delta1 uses current scenario NDWI minus scenario lag1
    """

    hist_lookup = {
        (str(row[GRID_COL]), int(row[YEAR_COL])):
            float(row[TREATMENT])
        for _, row in historical_ndwi.iterrows()
    }

    future_lookup = {
        (str(row[GRID_COL]), int(row[YEAR_COL])):
            float(row[TREATMENT])
        for _, row in future_ndwi.iterrows()
    }

    grid_ids = sorted(
        future_ndwi[GRID_COL]
        .astype(str)
        .unique()
        .tolist()
    )

    rows = []

    for grid_id in grid_ids:
        series = {}

        for year in HISTORY_SEED_YEARS:
            key = (grid_id, year)

            if key not in hist_lookup:
                raise RuntimeError(
                    f"Missing historical NDWI for {key}"
                )

            series[year] = hist_lookup[key]

        for year in FUTURE_YEARS:
            key = (grid_id, year)

            if key not in future_lookup:
                raise RuntimeError(
                    f"Missing future NDWI for {key}"
                )

            current = future_lookup[key]

            prev1 = series[year - 1]
            prev2 = series[year - 2]
            prev3 = series[year - 3]

            prior = [prev3, prev2, prev1]

            rows.append(
                {
                    GRID_COL:
                        grid_id,
                    YEAR_COL:
                        year,
                    TREATMENT:
                        current,
                    "NDWI_lag1":
                        prev1,
                    "NDWI_lag2":
                        prev2,
                    "NDWI_hist3_mean":
                        float(np.mean(prior)),
                    "NDWI_hist3_std":
                        sample_std_three(prior),
                    "NDWI_hist3_slope":
                        slope_three(prev3, prev2, prev1),
                    "NDWI_delta1":
                        float(current - prev1),
                }
            )

            series[year] = current

    return (
        pd.DataFrame(rows)
        .sort_values([YEAR_COL, GRID_COL])
        .reset_index(drop=True)
    )


def two_way_demean(df, columns):
    result = pd.DataFrame(index=df.index)

    for column in columns:
        values = pd.to_numeric(
            df[column],
            errors="raise",
        ).astype(float)

        grid_mean = (
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
            - grid_mean
            - year_mean
            + float(values.mean())
        )

    return result


def fit_ndwi_twfe(
    historical,
    years,
    label,
):
    """
    Historical response QA only.
    NOT used for frontend scenario inference.
    """

    controls = [
        "green_mask",
        "NDBI",
        "Albedo",
        "NightLights",
    ]

    sub = (
        historical[
            historical[YEAR_COL].isin(years)
        ]
        .copy()
        .sort_values([GRID_COL, YEAR_COL])
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
            f"{label}: rank-deficient TWFE design."
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
        "ndwi_beta_c_per_1_0":
            float(
                fit.params[TREATMENT]
            ),
        "response_c_per_plus_0_05_ndwi":
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
            f"WARNING: Kaduwela boundary could not be loaded: {exc}"
        )
        return None


# =============================================================================
# 0. VERIFY FROZEN INPUTS
# =============================================================================

section(
    "STEP 29D - BLUE-SPACE / WATER-WETNESS INTENSITY WHAT-IF SCENARIO"
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
    "Step-28 future LST predictions",
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
    "Corrected WGS84 geometry",
)

require_all_pass(
    STEP28_AUDIT_PATH,
    "Step-28 audit",
)


# =============================================================================
# 1. BLUE-SPACE REPRESENTATION AUDIT
# =============================================================================

section(
    "1. BLUE-SPACE REPRESENTATION AUDIT"
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

required_columns = [
    TARGET,
    TREATMENT,
    "dist_water",
    "green_mask",
    "NDBI",
    "Albedo",
    "NightLights",
]

for column in required_columns:
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
    .sort_values([GRID_COL, YEAR_COL])
    .reset_index(drop=True)
)

if len(historical) != EXPECTED_HIST_ROWS:
    raise RuntimeError(
        f"Historical rows={len(historical)}, "
        f"expected={EXPECTED_HIST_ROWS}"
    )

dist_nunique_by_grid = (
    historical.groupby(
        GRID_COL
    )[
        "dist_water"
    ]
    .nunique()
)

dist_grids_changing = int(
    (
        dist_nunique_by_grid
        > 1
    ).sum()
)

dist_water_static = (
    dist_grids_changing
    == 0
)

ndwi_nunique_by_grid = (
    historical.groupby(
        GRID_COL
    )[
        TREATMENT
    ]
    .nunique()
)

ndwi_grids_changing = int(
    (
        ndwi_nunique_by_grid
        > 1
    ).sum()
)

representation_df = pd.DataFrame(
    [
        {
            "variable":
                "dist_water",
            "observed_min":
                float(
                    historical[
                        "dist_water"
                    ].min()
                ),
            "observed_max":
                float(
                    historical[
                        "dist_water"
                    ].max()
                ),
            "unique_values":
                int(
                    historical[
                        "dist_water"
                    ].nunique()
                ),
            "grids_changing_through_time":
                dist_grids_changing,
            "scenario_role":
                (
                    "STATIC STRUCTURAL CONTEXT ONLY; "
                    "not changed without new water-body geometry"
                ),
        },
        {
            "variable":
                "NDWI",
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
                ndwi_grids_changing,
            "scenario_role":
                (
                    "PRIMARY CONTINUOUS BLUE-SPACE / WATER-WETNESS INTENSITY PROXY"
                ),
        },
    ]
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

if not dist_water_static:
    raise RuntimeError(
        "dist_water unexpectedly changes through time. "
        "Review blue-space representation before continuing."
    )

if ndwi_grids_changing != EXPECTED_GRIDS:
    raise RuntimeError(
        "NDWI does not vary through time for every grid as expected."
    )

print(
    "\nDecision:"
)

print(
    "  dist_water = frozen structural distance context."
)

print(
    "  NDWI       = Step-29D continuous blue-space/water-wetness scenario variable."
)


# =============================================================================
# 2. DATA-DERIVED POSITIVE NDWI SCENARIO MAGNITUDES
# =============================================================================

section(
    "2. DERIVE LOW / MODERATE / HIGH POSITIVE NDWI-INCREASE MAGNITUDES"
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
    "d_ndwi"
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
        "NDWI differences are not true adjacent-year transitions."
    )

positive_changes = (
    transitions.loc[
        transitions[
            "d_ndwi"
        ]
        > 0,
        "d_ndwi",
    ]
    .dropna()
)

if len(positive_changes) < 100:
    raise RuntimeError(
        "Too few positive historical NDWI changes to derive scenario magnitudes."
    )

scenario_levels = []

for label, q in SCENARIO_QUANTILES.items():
    magnitude = float(
        positive_changes.quantile(q)
    )

    if magnitude <= 0:
        raise RuntimeError(
            f"{label} NDWI scenario magnitude is not positive."
        )

    scenario_levels.append(
        {
            "scenario_level":
                label,
            "historical_positive_change_quantile":
                q,
            "positive_change_sample_count":
                int(
                    len(
                        positive_changes
                    )
                ),
            "ndwi_increase_magnitude":
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
                "ndwi_increase_magnitude"
            ]
        )
    for _, row in scenario_level_df.iterrows()
}


# =============================================================================
# 3. HISTORICAL RESPONSE QA
# =============================================================================

section(
    "3. HISTORICAL NDWI RESPONSE QA - NOT THE FRONTEND ENGINE"
)

response_rows = [
    fit_ndwi_twfe(
        historical=historical,
        years=HIST_YEARS,
        label="TWFE_ALL_2015_2025",
    ),
    fit_ndwi_twfe(
        historical=historical,
        years=[
            year
            for year in HIST_YEARS
            if year != 2017
        ],
        label="TWFE_EXCLUDE_2017",
    ),
]

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
            "ndwi_beta_c_per_1_0"
        ]
    )
    == np.sign(
        response_df.iloc[1][
            "ndwi_beta_c_per_1_0"
        ]
    )
)

print(
    "\nHistorical NDWI response sign stable after excluding 2017: "
    f"{historical_response_same_sign}"
)


# =============================================================================
# 4. LOAD FROZEN STEP-28 FUTURE BASELINE
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
        "future prediction matrix",
    ),
]:
    frame[YEAR_COL] = pd.to_numeric(
        frame[YEAR_COL],
        errors="raise",
    ).astype(int)

    frame[GRID_COL] = (
        frame[GRID_COL]
        .astype(str)
    )

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
        "Step-28 feature/prediction keys do not align."
    )

missing_features = [
    feature
    for feature in ordered_features
    if feature not in future.columns
]

if missing_features:
    raise RuntimeError(
        "Future matrix missing frozen features:\n"
        + str(
            missing_features
        )
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
        "Step-28 frozen feature matrix contains missing values."
    )

if not np.isfinite(
    X_base.to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "Step-28 frozen feature matrix contains non-finite values."
    )

if TREATMENT not in X_base.columns:
    raise RuntimeError(
        "NDWI is not in the frozen 93-feature list."
    )

if "dist_water" not in X_base.columns:
    raise RuntimeError(
        "dist_water is not in the frozen feature list."
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
# 5. RECONSTRUCT BASELINE NDWI DEPENDENCIES
# =============================================================================

section(
    "5. RECONSTRUCT BASELINE NDWI-DEPENDENT FROZEN FEATURES"
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

baseline_future_ndwi = future[
    [
        GRID_COL,
        YEAR_COL,
    ]
].copy()

baseline_future_ndwi[
    TREATMENT
] = X_base[
    TREATMENT
].to_numpy(
    dtype=float
)

baseline_rebuilt = build_ndwi_temporal_features(
    historical_ndwi=
        historical_seed,
    future_ndwi=
        baseline_future_ndwi,
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
        "Rebuilt baseline NDWI keys do not align."
    )

candidate_temporal = [
    "NDWI_lag1",
    "NDWI_lag2",
    "NDWI_hist3_mean",
    "NDWI_hist3_std",
    "NDWI_hist3_slope",
    "NDWI_delta1",
]

candidate_current = [
    "NDWI_nbr_contrast",
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
    "Retained NDWI temporal dependencies:"
)

for feature in retained_temporal:
    print(
        f"  {feature}"
    )

print(
    "\nRetained current NDWI dependencies:"
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
                ].to_numpy(
                    dtype=float
                )
                - baseline_rebuilt[
                    feature
                ].to_numpy(
                    dtype=float
                )
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

baseline_ndwi = X_base[
    TREATMENT
].to_numpy(
    dtype=float
)

if "NDWI_nbr_contrast" in X_base.columns:
    if "NDWI_nbr_mean" in X_base.columns:
        baseline_ndwi_nbr_mean = X_base[
            "NDWI_nbr_mean"
        ].to_numpy(
            dtype=float
        )
    else:
        baseline_ndwi_nbr_mean = (
            baseline_ndwi
            - X_base[
                "NDWI_nbr_contrast"
            ].to_numpy(
                dtype=float
            )
        )

    expected = (
        baseline_ndwi
        - baseline_ndwi_nbr_mean
    )

    max_error = float(
        np.max(
            np.abs(
                X_base[
                    "NDWI_nbr_contrast"
                ].to_numpy(
                    dtype=float
                )
                - expected
            )
        )
    )

    recon_rows.append(
        {
            "feature":
                "NDWI_nbr_contrast",
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
        "No retained NDWI-dependent frozen features were verified."
    )

if (
    recon_df[
        "status"
    ]
    != "PASS"
).any():
    raise RuntimeError(
        "Baseline NDWI feature reconstruction failed."
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

baseline_reproduction_error = float(
    np.max(
        np.abs(
            baseline_raw_model
            - baseline_raw_saved
        )
    )
)

print(
    "Max reproduction error: "
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
    "PASS: frozen Step-28 baseline reproduced."
)


# =============================================================================
# 7. RUN PERSISTENT LOCAL BLUE-SPACE / NDWI SCENARIOS
# =============================================================================

section(
    "7. RUN PERSISTENT LOCAL BLUE-SPACE / NDWI-INCREASE SCENARIOS"
)

historical_ndwi_max = float(
    historical[
        TREATMENT
    ].max()
)

historical_ndwi_min = float(
    historical[
        TREATMENT
    ].min()
)

baseline_dist_water = X_base[
    "dist_water"
].to_numpy(
    dtype=float
)

scenario_rows = []
feature_exports = []
feature_change_rows = []

allowed_changed_features = {
    TREATMENT,
    *retained_temporal,
    *retained_current,
}

# Explicitly frozen structural context.
allowed_changed_features.discard(
    "dist_water"
)

if "log1p_dist_water" in ordered_features:
    allowed_changed_features.discard(
        "log1p_dist_water"
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
        f"\nRunning {level} blue-space intensity scenario:"
    )

    print(
        f"  Requested NDWI increase magnitude = {magnitude:.6f}"
    )

    scenario_future_ndwi = (
        baseline_future_ndwi
        .copy()
    )

    scenario_future_ndwi[
        "baseline_NDWI"
    ] = scenario_future_ndwi[
        TREATMENT
    ]

    scenario_future_ndwi[
        TREATMENT
    ] = np.clip(
        scenario_future_ndwi[
            "baseline_NDWI"
        ].to_numpy(
            dtype=float
        )
        + magnitude,
        historical_ndwi_min,
        historical_ndwi_max,
    )

    scenario_future_ndwi[
        "actual_ndwi_increase"
    ] = (
        scenario_future_ndwi[
            TREATMENT
        ]
        - scenario_future_ndwi[
            "baseline_NDWI"
        ]
    )

    scenario_future_ndwi[
        "upper_support_bound_limited"
    ] = (
        scenario_future_ndwi[
            "actual_ndwi_increase"
        ]
        < magnitude
        - 1e-12
    )

    rebuilt = build_ndwi_temporal_features(
        historical_ndwi=
            historical_seed,
        future_ndwi=
            scenario_future_ndwi[
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

    scenario_ndwi = rebuilt[
        TREATMENT
    ].to_numpy(
        dtype=float
    )

    X_scenario[
        TREATMENT
    ] = scenario_ndwi

    for feature in retained_temporal:
        X_scenario[
            feature
        ] = rebuilt[
            feature
        ].to_numpy(
            dtype=float
        )

    if "NDWI_nbr_contrast" in X_scenario.columns:
        if "NDWI_nbr_mean" in X_base.columns:
            local_nbr_mean = X_base[
                "NDWI_nbr_mean"
            ].to_numpy(
                dtype=float
            )
        else:
            local_nbr_mean = (
                baseline_ndwi
                - X_base[
                    "NDWI_nbr_contrast"
                ].to_numpy(
                    dtype=float
                )
            )

        X_scenario[
            "NDWI_nbr_contrast"
        ] = (
            scenario_ndwi
            - local_nbr_mean
        )

    # dist_water and any distance transform MUST remain frozen.
    dist_water_change = float(
        np.max(
            np.abs(
                X_scenario[
                    "dist_water"
                ].to_numpy(
                    dtype=float
                )
                - baseline_dist_water
            )
        )
    )

    if dist_water_change > UNCHANGED_FEATURE_TOL:
        raise RuntimeError(
            f"{level}: dist_water changed unexpectedly."
        )

    if "log1p_dist_water" in ordered_features:
        log_dist_change = float(
            np.max(
                np.abs(
                    X_scenario[
                        "log1p_dist_water"
                    ].to_numpy(
                        dtype=float
                    )
                    - X_base[
                        "log1p_dist_water"
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
        )

        if log_dist_change > UNCHANGED_FEATURE_TOL:
            raise RuntimeError(
                f"{level}: log1p_dist_water changed unexpectedly."
            )

    if X_scenario.isna().any().any():
        raise RuntimeError(
            f"{level}: scenario feature matrix contains missing values."
        )

    if not np.isfinite(
        X_scenario.to_numpy(
            dtype=float
        )
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
                    ].to_numpy(
                        dtype=float
                    )
                    - X_base[
                        feature
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
        )

        if max_change > UNCHANGED_FEATURE_TOL:
            changed_features.append(
                feature
            )

    unexpected = sorted(
        set(
            changed_features
        )
        - allowed_changed_features
    )

    if unexpected:
        raise RuntimeError(
            f"{level}: unexpected frozen features changed: {unexpected}"
        )

    feature_change_rows.append(
        {
            "scenario_level":
                level,
            "changed_feature_count":
                len(
                    changed_features
                ),
            "changed_features":
                "; ".join(
                    sorted(
                        changed_features
                    )
                ),
            "unexpected_feature_count":
                len(
                    unexpected
                ),
            "dist_water_max_change":
                dist_water_change,
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
        scenario_future_ndwi
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

    for i in range(
        EXPECTED_FUTURE_ROWS
    ):
        delta = float(
            local_delta[i]
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
                    f"blue_space_ndwi_increase_{level.lower()}",
                "scenario_level":
                    level,
                "scenario_variable":
                    "NDWI",
                "scenario_interpretation":
                    "blue_space_water_wetness_intensity_increase",
                "requested_ndwi_increase":
                    magnitude,
                "baseline_ndwi":
                    float(
                        meta.at[
                            i,
                            "baseline_NDWI",
                        ]
                    ),
                "scenario_ndwi":
                    float(
                        meta.at[
                            i,
                            TREATMENT,
                        ]
                    ),
                "actual_ndwi_increase":
                    float(
                        meta.at[
                            i,
                            "actual_ndwi_increase",
                        ]
                    ),
                "support_bound_limited":
                    bool(
                        meta.at[
                            i,
                            "upper_support_bound_limited",
                        ]
                    ),
                "dist_water_m":
                    float(
                        baseline_dist_water[i]
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
                    float(
                        cooling_benefit[i]
                    ),
                "scenario_effect_direction":
                    direction,
            }
        )

    # Unique-column feature export.
    feature_export = X_scenario.reset_index(
        drop=True
    ).copy()

    metadata = pd.DataFrame(
        {
            GRID_COL:
                future[
                    GRID_COL
                ].astype(str).to_numpy(),
            YEAR_COL:
                future[
                    YEAR_COL
                ].astype(int).to_numpy(),
            "area_name":
                future[
                    "area_name"
                ].astype(str).to_numpy(),
            "scenario_level":
                level,
            "requested_ndwi_increase":
                magnitude,
        }
    )

    for column in reversed(
        metadata.columns.tolist()
    ):
        if column in feature_export.columns:
            if column == YEAR_COL:
                existing = pd.to_numeric(
                    feature_export[
                        column
                    ],
                    errors="raise",
                ).astype(int).to_numpy()

                expected = metadata[
                    column
                ].astype(int).to_numpy()

                if not np.array_equal(
                    existing,
                    expected,
                ):
                    raise RuntimeError(
                        f"{level}: feature year does not match metadata year."
                    )

            continue

        feature_export.insert(
            0,
            column,
            metadata[
                column
            ].to_numpy(),
        )

    if feature_export.columns.duplicated().any():
        duplicated = feature_export.columns[
            feature_export.columns.duplicated()
        ].tolist()

        raise RuntimeError(
            f"{level}: duplicate export columns remain: {duplicated}"
        )

    feature_exports.append(
        feature_export
    )

    print(
        "  Mean actual NDWI increase : "
        f"{meta['actual_ndwi_increase'].mean():.6f}"
    )

    print(
        "  Upper-support-limited rows: "
        f"{int(meta['upper_support_bound_limited'].sum()):,}"
    )

    print(
        "  Mean modeled LST change   : "
        f"{local_delta.mean():+.6f} °C"
    )

    print(
        "  Mean cooling benefit      : "
        f"{cooling_benefit.mean():+.6f} °C"
    )


scenario_df = pd.DataFrame(
    scenario_rows
)

feature_export_df = pd.concat(
    feature_exports,
    ignore_index=True,
)

feature_change_df = pd.DataFrame(
    feature_change_rows
)

expected_scenario_rows = (
    EXPECTED_FUTURE_ROWS
    * len(
        SCENARIO_QUANTILES
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
    ALL_SCENARIOS_PATH,
    index=False,
)

feature_export_df.to_csv(
    FEATURE_EXPORT_PATH,
    index=False,
)

feature_change_df.to_csv(
    FEATURE_CHANGE_AUDIT_PATH,
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
                f"29d_blue_space_ndwi_"
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
    "8. BLUE-SPACE / NDWI SCENARIO SUMMARY"
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
        requested_ndwi_increase=(
            "requested_ndwi_increase",
            "first",
        ),
        mean_actual_ndwi_increase=(
            "actual_ndwi_increase",
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
        direction_counts[
            column
        ] = 0

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

grid[
    GRID_COL
] = grid[
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
                    "requested_ndwi_increase",
                    "scenario_ndwi",
                    "dist_water_m",
                    "scenario_predicted_lst_c",
                    "modeled_lst_change_c",
                    "cooling_benefit_c",
                    "scenario_effect_direction",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "requested_ndwi_increase":
                        f"ndwi_increase_{level.lower()}",
                    "scenario_ndwi":
                        f"scenario_ndwi_{level.lower()}",
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
        / f"29d_blue_space_ndwi_{year}_wgs84.geojson"
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

        fig, ax = plt.subplots(
            figsize=(10, 10)
        )

        gdf.plot(
            column=
                column,
            ax=ax,
            legend=True,
            edgecolor="none",
            legend_kwds={
                "label":
                    "Modeled LST change (°C)",
                "shrink":
                    0.72,
            },
        )

        if boundary is not None:
            boundary.boundary.plot(
                ax=ax,
                linewidth=1.4,
            )
        else:
            gdf.dissolve().boundary.plot(
                ax=ax,
                linewidth=1.4,
            )

        ax.set_title(
            (
                f"Kaduwela Blue-Space / NDWI Scenario — {year}\n"
                f"{level.title()} persistent water-wetness intensity increase"
            ),
            fontsize=14,
            pad=12,
        )

        ax.set_axis_off()

        plt.tight_layout()

        map_path = (
            FIGURE_DIR
            / (
                f"29d_blue_space_lst_change_"
                f"{year}_{level.lower()}.png"
            )
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
)

ax.set_xlabel(
    "Blue-space / water-wetness intensity scenario"
)

ax.set_ylabel(
    "Mean modeled LST change (°C)"
)

ax.set_title(
    "Step 29D Blue-Space / NDWI Scenario Sensitivity"
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

plt.close(
    fig
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)

x = np.arange(
    len(
        response_df
    )
)

y = (
    response_df[
        "response_c_per_plus_0_05_ndwi"
    ].to_numpy(
        dtype=float
    )
)

lower = (
    y
    - response_df[
        "ci95_lower_per_plus_0_05"
    ].to_numpy(
        dtype=float
    )
)

upper = (
    response_df[
        "ci95_upper_per_plus_0_05"
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
    "Historical LST response per +0.05 NDWI (°C)"
)

ax.set_title(
    "Historical NDWI Response QA"
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

plt.close(
    fig
)

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
    "11. STRICT STEP-29D AUDIT"
)

cooling_identity_error = float(
    np.max(
        np.abs(
            scenario_df[
                "cooling_benefit_c"
            ].to_numpy(
                dtype=float
            )
            + scenario_df[
                "modeled_lst_change_c"
            ].to_numpy(
                dtype=float
            )
        )
    )
)

dist_water_scenario_change = float(
    np.max(
        np.abs(
            scenario_df[
                "dist_water_m"
            ].to_numpy(
                dtype=float
            )
            - np.tile(
                baseline_dist_water,
                len(
                    SCENARIO_QUANTILES
                ),
            )
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
            "dist_water_static_historically",
        "status":
            (
                "PASS"
                if dist_water_static
                else "FAIL"
            ),
        "details":
            f"grids_changing={dist_grids_changing}",
    },
    {
        "check":
            "ndwi_time_varying",
        "status":
            (
                "PASS"
                if ndwi_grids_changing
                == EXPECTED_GRIDS
                else "FAIL"
            ),
        "details":
            f"grids_changing={ndwi_grids_changing}",
    },
    {
        "check":
            "literal_water_coverage_percent_slider_used",
        "status":
            "PASS",
        "details":
            (
                "No. NDWI is used as a continuous spectral water/wetness "
                "intensity proxy."
            ),
    },
    {
        "check":
            "distance_to_water_slider_used",
        "status":
            "PASS",
        "details":
            (
                "No. dist_water remains frozen because no new water-body "
                "geometry is created."
            ),
    },
    {
        "check":
            "scenario_levels_data_derived",
        "status":
            "PASS",
        "details":
            (
                "LOW=q50, MODERATE=q75, HIGH=q90 of positive historical "
                "adjacent annual within-grid NDWI changes."
            ),
    },
    {
        "check":
            "baseline_ndwi_feature_reconstruction",
        "status":
            (
                "PASS"
                if (
                    recon_df[
                        "status"
                    ]
                    == "PASS"
                ).all()
                else "FAIL"
            ),
        "details":
            f"features_checked={len(recon_df)}",
    },
    {
        "check":
            "baseline_raw_prediction_reproduction",
        "status":
            (
                "PASS"
                if baseline_reproduction_error
                <= PREDICTION_REPRO_TOL
                else "FAIL"
            ),
        "details":
            f"max_error_c={baseline_reproduction_error:.12e}",
    },
    {
        "check":
            "only_intended_frozen_features_changed",
        "status":
            (
                "PASS"
                if (
                    feature_change_df[
                        "unexpected_feature_count"
                    ]
                    == 0
                ).all()
                else "FAIL"
            ),
        "details":
            "; ".join(
                feature_change_df[
                    "changed_features"
                ]
                .astype(str)
                .tolist()
            ),
    },
    {
        "check":
            "dist_water_unchanged_in_scenario",
        "status":
            (
                "PASS"
                if dist_water_scenario_change
                <= UNCHANGED_FEATURE_TOL
                else "FAIL"
            ),
        "details":
            f"max_change={dist_water_scenario_change:.12e}",
    },
    {
        "check":
            "scenario_row_count",
        "status":
            (
                "PASS"
                if len(
                    scenario_df
                )
                == expected_scenario_rows
                else "FAIL"
            ),
        "details":
            (
                f"actual={len(scenario_df)}, "
                f"expected={expected_scenario_rows}"
            ),
    },
    {
        "check":
            "scenario_outputs_finite",
        "status":
            (
                "PASS"
                if np.isfinite(
                    scenario_df[
                        [
                            "scenario_predicted_lst_c",
                            "modeled_lst_change_c",
                            "cooling_benefit_c",
                        ]
                    ].to_numpy(
                        dtype=float
                    )
                ).all()
                else "FAIL"
            ),
        "details":
            f"rows={len(scenario_df)}",
    },
    {
        "check":
            "cooling_identity",
        "status":
            (
                "PASS"
                if cooling_identity_error
                <= PREDICTION_REPRO_TOL
                else "FAIL"
            ),
        "details":
            f"max_error_c={cooling_identity_error:.12e}",
    },
    {
        "check":
            "local_neighbor_semantics",
        "status":
            "PASS",
        "details":
            (
                "NDWI neighbour mean remains Step-28 baseline; selected-grid "
                "contrast changes if retained."
            ),
    },
    {
        "check":
            "scenario_recentering",
        "status":
            "PASS",
        "details":
            "Not applied. Local same-grid raw anomaly delta used.",
    },
    {
        "check":
            "model_retraining",
        "status":
            "PASS",
        "details":
            "None. Frozen Step-22B LightGBM used for inference only.",
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
            "uncertainty_added_here",
        "status":
            "PASS",
        "details":
            "No. Uncertainty/confidence remains Step 30.",
    },
    {
        "check":
            "yearly_geojson_count",
        "status":
            (
                "PASS"
                if len(
                    geojson_files
                )
                == len(
                    FUTURE_YEARS
                )
                else "FAIL"
            ),
        "details":
            (
                f"saved={len(geojson_files)}, "
                f"expected={len(FUTURE_YEARS)}"
            ),
    },
    {
        "check":
            "scenario_map_count",
        "status":
            (
                "PASS"
                if len(
                    map_files
                )
                == (
                    len(
                        FUTURE_YEARS
                    )
                    * len(
                        SCENARIO_QUANTILES
                    )
                )
                else "FAIL"
            ),
        "details":
            (
                f"saved={len(map_files)}, "
                f"expected="
                f"{len(FUTURE_YEARS) * len(SCENARIO_QUANTILES)}"
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
        "One or more Step-29D audit checks failed."
    )


# =============================================================================
# 12. METHODOLOGY + REPORT
# =============================================================================

section(
    "12. SAVE STEP-29D METHODOLOGY + REPORT"
)

methodology = {
    "step":
        "29D",
    "name":
        "Blue-space / water-wetness intensity what-if scenario",
    "frontend_scenario_engine":
        "Frozen Step-22B LightGBM",
    "new_ml_model_for_frontend":
        False,
    "scenario_variable":
        {
            "name":
                "NDWI",
            "meaning":
                "continuous spectral water/wetness intensity proxy",
            "frontend_wording":
                "Blue-space / water-wetness restoration intensity",
            "not_worded_as":
                "water-cover percentage",
        },
    "structural_water_distance":
        {
            "name":
                "dist_water",
            "status":
                "frozen",
            "reason":
                (
                    "Current distance-to-water layer is static. A literal "
                    "distance change requires adding/restoring explicit "
                    "water-body geometry and spatially recomputing distances."
                ),
        },
    "scenario_levels":
        scenario_level_df.to_dict(
            orient="records"
        ),
    "scenario_direction":
        "persistent NDWI increase from 2026 through 2030",
    "support_policy":
        (
            "Scenario NDWI is clipped to historical 2015-2025 observed "
            "NDWI maximum."
        ),
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
            "NDWI neighbour mean remains Step-28 baseline; selected-grid "
            "NDWI neighbour contrast changes if retained."
        ),
    "scenario_effect_formula":
        (
            "scenario LST = Step-28 baseline LST + "
            "(scenario raw anomaly - baseline raw anomaly)"
        ),
    "frontend_outputs":
        [
            "baseline_ndwi",
            "scenario_ndwi",
            "dist_water_m",
            "baseline_predicted_lst_c",
            "scenario_predicted_lst_c",
            "modeled_lst_change_c",
            "cooling_benefit_c",
            "scenario_effect_direction",
        ],
    "interpretation_warning":
        (
            "Model-based sensitivity only. NDWI is not literal water-cover "
            "percentage, dist_water is not changed, and LST change is not a "
            "guaranteed causal physical outcome."
        ),
    "uncertainty":
        "Deferred to Step 30",
}

METHOD_PATH.write_text(
    json.dumps(
        methodology,
        indent=2,
    ),
    encoding="utf-8",
)

report_lines = [
    "STEP 29D - BLUE-SPACE / WATER-WETNESS INTENSITY WHAT-IF SCENARIO",
    "=" * 124,
    "",
    "REPRESENTATION DECISION",
    "-" * 124,
    (
        "NDWI is used as the continuous blue-space / water-wetness intensity "
        "proxy."
    ),
    (
        "dist_water remains frozen structural context because this step does "
        "not create new water-body geometry."
    ),
    "",
    "DATA-DERIVED SCENARIO LEVELS",
    "-" * 124,
    scenario_level_df.to_string(
        index=False
    ),
    "",
    "HISTORICAL RESPONSE QA",
    "-" * 124,
    response_df.to_string(
        index=False
    ),
    "",
    "SCENARIO SUMMARY",
    "-" * 124,
    summary.to_string(
        index=False
    ),
    "",
    "FRONTEND INTERPRETATION",
    "-" * 124,
    (
        "The planner selects a grid/year and LOW, MODERATE or HIGH "
        "blue-space/water-wetness intensity increase."
    ),
    (
        "The system displays baseline NDWI, scenario NDWI, structural "
        "distance to existing water, baseline predicted LST, scenario "
        "predicted LST, modeled LST change and direction."
    ),
    (
        "Do not label NDWI increase as a literal percentage increase in "
        "water coverage."
    ),
    "",
    "SCIENTIFIC LIMITATION",
    "-" * 124,
    (
        "This is a frozen-model what-if sensitivity analysis, not a guaranteed "
        "causal physical cooling/heating response."
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
    ALL_SCENARIOS_PATH,
    PRIMARY_SCENARIO_PATH,
    FEATURE_EXPORT_PATH,
    SUMMARY_PATH,
    SCENARIO_LEVEL_PATH,
    REPRESENTATION_AUDIT_PATH,
    BASELINE_RECON_PATH,
    RESPONSE_QA_PATH,
    FEATURE_CHANGE_AUDIT_PATH,
    AUDIT_PATH,
    METHOD_PATH,
    REPORT_PATH,
    SUMMARY_FIGURE_PATH,
    RESPONSE_FIGURE_PATH,
    *geojson_files,
    *map_files,
]

for year in FUTURE_YEARS:
    for level in [
        "LOW",
        "MODERATE",
        "HIGH",
    ]:
        hash_files.append(
            SCENARIO_DIR
            / (
                f"29d_blue_space_ndwi_"
                f"{level.lower()}_{year}.csv"
            )
        )

unique_files = []
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
        unique_files.append(
            path
        )

hash_rows = []

for path in unique_files:
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
    "STEP 29D COMPLETED SUCCESSFULLY"
)

print(
    "Blue-space / water-wetness intensity scenario completed.\n"
    "NDWI was used as the continuous intervention proxy.\n"
    "dist_water remained frozen structural context.\n"
    "LOW/MODERATE/HIGH NDWI increases were derived from historical positive "
    "adjacent annual within-grid NDWI changes.\n"
    "The intervention is persistent from 2026 through 2030.\n"
    "NDWI temporal/dependent frozen features were rebuilt consistently.\n"
    "The same frozen Step-22B LightGBM was used without retraining.\n"
    "Step-28 baseline predictions were not modified.\n"
    "Historical TWFE response analysis is QA evidence only, not the frontend "
    "scenario engine.\n"
    "Frontend can show baseline NDWI, scenario NDWI, baseline LST, scenario "
    "LST and modeled LST change.\n"
    "Do NOT label NDWI increase as water-cover percentage.\n"
    "No new water-body geometry or distance-to-water change was simulated.\n"
    "No causal or guaranteed cooling/heating claim is made.\n"
    "Uncertainty/confidence remains deferred to Step 30.\n"
    "\nNEXT: review Step-29D terminal output before freezing 29D and moving "
    "to Step 30 scenario/forecast uncertainty."
)
