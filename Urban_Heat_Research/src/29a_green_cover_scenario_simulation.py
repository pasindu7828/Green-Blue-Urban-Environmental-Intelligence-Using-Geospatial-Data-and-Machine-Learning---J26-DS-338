"""
STEP 29A - GREEN-COVER WHAT-IF COOLING SCENARIO SIMULATION
Urban Heat Research - Kaduwela, Sri Lanka

FINAL SCENARIO DESIGN
---------------------
This step evaluates user-selectable, LOCAL grid-cell green-cover interventions
against the frozen Step-28 future heat forecasts.

Primary scenario:
    +5 percentage points green cover

Additional planning-sensitivity scenarios:
    +10 percentage points
    +20 percentage points

Example:
    baseline green_mask = 0.40
    +5 pp scenario       = 0.45

All scenario green-cover values are clipped to [0, 1].

WHY LOCAL / INDEPENDENT SCENARIOS?
----------------------------------
The research proposal defines the scenario as a planner selecting a real grid
cell, perturbing green-space %, and re-querying the trained model. Therefore
each grid-year row below is an INDEPENDENT local counterfactual: that grid is
modified while the rest of Kaduwela remains at its Step-28 baseline forecast.

This is NOT a simultaneous citywide greening scenario.

WHY USE RAW-ANOMALY DELTA?
--------------------------
Step 28 decomposes future LST as:

    final grid LST
      = annual Kaduwela thermal level
      + centered spatial anomaly

For an independent local intervention, the model-based scenario effect is:

    local anomaly delta
      = scenario raw anomaly - baseline raw anomaly

Then:

    scenario LST
      = baseline Step-28 LST + local anomaly delta

and:

    cooling benefit
      = baseline LST - scenario LST
      = - local anomaly delta

We deliberately DO NOT re-center the scenario predictions across all 2,194
rows, because doing so would incorrectly treat the batch as if every grid were
greened simultaneously.

FEATURE-CONSISTENCY RULE
------------------------
Only green cover is directly perturbed. All current-year model features that
mathematically depend on current green_mask are rebuilt consistently:

    green_mask
    vegetation_deficit = 1 - green_mask
    ndbi_x_vegetation_deficit = NDBI * vegetation_deficit
    ndvi_x_green_mask = NDVI * green_mask
    green_mask_delta1 = green_mask - green_mask_lag1
    green_mask_nbr_contrast = green_mask - green_mask_nbr_mean

For a LOCAL current-year intervention:
- green_mask_lag1 / lag2 / hist3_* remain baseline because they describe
  prior years;
- green_mask_nbr_mean remains baseline because neighbouring grids are not
  changed in that local counterfactual;
- green_mask_nbr_contrast changes because the selected grid changes relative
  to its unchanged neighbours;
- NDVI/NDBI/etc. are NOT automatically changed. Doing so would impose an
  unsupported physical relationship. This is a direct feature-perturbation
  sensitivity test.

INTERPRETATION
--------------
This is modeled statistical sensitivity based on the fitted Kaduwela model.
It is NOT a guaranteed physical cooling outcome. Real cooling depends on
species, canopy structure, placement, irrigation, maintenance, land conversion,
microclimate, and other factors not explicitly represented in the model.

No model is retrained. Step-28 forecasts remain unchanged.
Uncertainty is deferred to Step 30.
"""

from pathlib import Path
import hashlib
import json
import re

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except ImportError as exc:
    raise ImportError(
        "\ngeopandas is required for Step 29A spatial outputs.\n"
        "Install it with:\n"
        "    pip install geopandas\n"
    ) from exc


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"

EXPECTED_GRIDS = 2194
FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]
EXPECTED_ROWS = EXPECTED_GRIDS * len(FUTURE_YEARS)
EXPECTED_FEATURES = 93

# Proposal-aligned primary scenario + useful planning sensitivity levels.
SCENARIO_INCREASE_PP = [5, 10, 20]
PRIMARY_SCENARIO_PP = 5

GREEN_MIN = 0.0
GREEN_MAX = 1.0

FORMULA_TOL = 1e-8
PREDICTION_REPRO_TOL = 1e-10
UNCHANGED_FEATURE_TOL = 1e-12

EXPECTED_HASHES = {
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

FEATURE_LIST_PATH = (
    ROOT / "outputs" / "reports" / "21_final_feature_list.csv"
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
    / "29a_green_cover_scenarios_2026_2030.csv"
)

PRIMARY_SCENARIO_PATH = (
    SCENARIO_DIR
    / f"29a_green_cover_primary_{PRIMARY_SCENARIO_PP}pp_2026_2030.csv"
)

SUMMARY_PATH = (
    SCENARIO_DIR
    / "29a_green_cover_scenario_summary.csv"
)

MONOTONIC_PATH = (
    SCENARIO_DIR
    / "29a_green_cover_response_monotonicity.csv"
)

FORMULA_AUDIT_PATH = (
    REPORT_DIR
    / "29a_green_cover_feature_formula_audit.csv"
)

AUDIT_PATH = (
    REPORT_DIR
    / "29a_green_cover_scenario_audit.csv"
)

METHOD_PATH = (
    REPORT_DIR
    / "29a_green_cover_scenario_methodology.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "29a_green_cover_scenario_report.txt"
)

HASH_PATH = (
    REPORT_DIR
    / "29a_green_cover_scenario_hashes.csv"
)

SUMMARY_FIGURE_PATH = (
    FIGURE_DIR
    / "29a_mean_cooling_benefit_by_green_scenario.png"
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
            f"{label} audit has no 'status' column."
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
        "Could not identify the frozen 93-feature column."
    )


def safe_name(value):
    return re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        str(value),
    ).strip("_")


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
            "WARNING: real Kaduwela KML boundary could not be loaded."
        )
        print(
            f"  {type(exc).__name__}: {exc}"
        )

        return None


# =============================================================================
# 0. VERIFY FROZEN INPUTS
# =============================================================================

section(
    "STEP 29A - GREEN-COVER WHAT-IF COOLING SCENARIO SIMULATION"
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
    "\nScenario definition:"
)

for pp in SCENARIO_INCREASE_PP:
    primary = (
        "  [PRIMARY]"
        if pp == PRIMARY_SCENARIO_PP
        else ""
    )

    print(
        f"  +{pp} percentage points green cover{primary}"
    )

print(
    "\nEach row is an independent local-grid counterfactual."
)


# =============================================================================
# 1. LOAD FROZEN FEATURES + STEP-28 BASELINE
# =============================================================================

section(
    "1. LOAD FROZEN FUTURE FEATURES + STEP-28 BASELINE PREDICTIONS"
)

feature_df = pd.read_csv(
    FEATURE_LIST_PATH
)

feature_col = detect_feature_column(
    feature_df
)

ordered_features = (
    feature_df[feature_col]
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
    (future, "future features"),
    (pred, "future predictions"),
]:
    frame[YEAR_COL] = pd.to_numeric(
        frame[YEAR_COL],
        errors="raise",
    ).astype(int)

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(
            f"{label}: rows={len(frame)}, "
            f"expected={EXPECTED_ROWS}"
        )

    if sorted(
        frame[YEAR_COL]
        .unique()
        .tolist()
    ) != FUTURE_YEARS:
        raise RuntimeError(
            f"{label}: unexpected future years."
        )

    if frame.duplicated(
        [GRID_COL, YEAR_COL]
    ).any():
        raise RuntimeError(
            f"{label}: duplicate grid-year rows."
        )

future = (
    future
    .sort_values(
        [YEAR_COL, GRID_COL]
    )
    .reset_index(drop=True)
)

pred = (
    pred
    .sort_values(
        [YEAR_COL, GRID_COL]
    )
    .reset_index(drop=True)
)

if not (
    future[
        [GRID_COL, YEAR_COL]
    ].equals(
        pred[
            [GRID_COL, YEAR_COL]
        ]
    )
):
    raise RuntimeError(
        "Future feature and prediction grid-year keys do not align."
    )

missing_features = [
    feature
    for feature in ordered_features
    if feature not in future.columns
]

if missing_features:
    raise RuntimeError(
        "Future feature matrix is missing frozen features:\n"
        + str(missing_features)
    )

X_base = future[
    ordered_features
].copy()

for column in ordered_features:
    X_base[column] = pd.to_numeric(
        X_base[column],
        errors="raise",
    )

if X_base.isna().any().any():
    raise RuntimeError(
        "Frozen Step-28 feature matrix contains missing values."
    )

if not np.isfinite(
    X_base.to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Frozen Step-28 feature matrix contains non-finite values."
    )

print(
    f"Future baseline rows : {len(future):,}"
)

print(
    f"Unique grids         : {future[GRID_COL].nunique():,}"
)

print(
    f"Future years         : {FUTURE_YEARS}"
)

print(
    f"Frozen features      : {len(ordered_features)}"
)


# =============================================================================
# 2. VERIFY GREEN-FEATURE FORMULAS ON STEP-28 BASELINE
# =============================================================================

section(
    "2. VERIFY GREEN-COVER DEPENDENT FEATURE FORMULAS"
)

# The Step-28 file contains ONLY the frozen 93 production predictors.
# Some Step-07 engineered intermediates (for example vegetation_deficit)
# may have been excluded by final feature selection. That is valid.
#
# We therefore require only the direct intervention feature itself and then
# update/verify each dependent feature ONLY if that feature is actually part
# of the frozen 93-feature production model.

if "green_mask" not in X_base.columns:
    raise RuntimeError(
        "green_mask is not present in the frozen Step-28 feature matrix; "
        "the green-cover scenario cannot be evaluated."
    )

green = X_base[
    "green_mask"
].to_numpy(dtype=float)

if (
    np.nanmin(green) < GREEN_MIN - FORMULA_TOL
    or np.nanmax(green) > GREEN_MAX + FORMULA_TOL
):
    raise RuntimeError(
        "green_mask is outside expected [0,1] range."
    )

# Candidate current-year features that are mathematically affected by changing
# green_mask. Only the subset that survived final feature selection is updated.
candidate_dependent_features = [
    "vegetation_deficit",
    "ndbi_x_vegetation_deficit",
    "ndvi_x_green_mask",
    "green_mask_delta1",
    "green_mask_nbr_contrast",
]

scenario_dependent_features = [
    feature
    for feature in candidate_dependent_features
    if feature in X_base.columns
]

print(
    "Scenario-dependent frozen features present:"
)

for feature in scenario_dependent_features:
    print(
        f"  {feature}"
    )

excluded_intermediates = [
    feature
    for feature in candidate_dependent_features
    if feature not in X_base.columns
]

if excluded_intermediates:
    print(
        "\nValid Step-21 feature-selection exclusions "
        "(not required as model inputs):"
    )

    for feature in excluded_intermediates:
        print(
            f"  {feature}"
        )

# Strictly require source columns only when a retained dependent feature needs
# them. This prevents silently inventing unavailable inputs.
dependency_requirements = {
    "ndbi_x_vegetation_deficit": ["NDBI"],
    "ndvi_x_green_mask": ["NDVI"],
    "green_mask_delta1": ["green_mask_lag1"],
    "green_mask_nbr_contrast": ["green_mask_nbr_mean"],
}

for feature in scenario_dependent_features:
    for dependency in dependency_requirements.get(
        feature,
        [],
    ):
        if dependency not in X_base.columns:
            raise RuntimeError(
                f"Frozen feature '{feature}' requires '{dependency}', "
                "but that dependency is not present in the Step-28 matrix."
            )

# -------------------------------------------------------------------------
# Verify the formulas that CAN be verified from retained frozen predictors.
# vegetation_deficit itself does not need to be retained in order to rebuild
# interactions; it can be computed internally as 1 - green_mask.
# -------------------------------------------------------------------------

formula_checks = {}

if "vegetation_deficit" in X_base.columns:
    formula_checks[
        "vegetation_deficit"
    ] = (
        X_base[
            "vegetation_deficit"
        ].to_numpy(dtype=float)
        - (
            1.0
            - X_base[
                "green_mask"
            ].to_numpy(dtype=float)
        )
    )

if "ndbi_x_vegetation_deficit" in X_base.columns:
    formula_checks[
        "ndbi_x_vegetation_deficit"
    ] = (
        X_base[
            "ndbi_x_vegetation_deficit"
        ].to_numpy(dtype=float)
        - (
            X_base[
                "NDBI"
            ].to_numpy(dtype=float)
            * (
                1.0
                - X_base[
                    "green_mask"
                ].to_numpy(dtype=float)
            )
        )
    )

if "ndvi_x_green_mask" in X_base.columns:
    formula_checks[
        "ndvi_x_green_mask"
    ] = (
        X_base[
            "ndvi_x_green_mask"
        ].to_numpy(dtype=float)
        - (
            X_base[
                "NDVI"
            ].to_numpy(dtype=float)
            * X_base[
                "green_mask"
            ].to_numpy(dtype=float)
        )
    )

if "green_mask_delta1" in X_base.columns:
    formula_checks[
        "green_mask_delta1"
    ] = (
        X_base[
            "green_mask_delta1"
        ].to_numpy(dtype=float)
        - (
            X_base[
                "green_mask"
            ].to_numpy(dtype=float)
            - X_base[
                "green_mask_lag1"
            ].to_numpy(dtype=float)
        )
    )

if "green_mask_nbr_contrast" in X_base.columns:
    formula_checks[
        "green_mask_nbr_contrast"
    ] = (
        X_base[
            "green_mask_nbr_contrast"
        ].to_numpy(dtype=float)
        - (
            X_base[
                "green_mask"
            ].to_numpy(dtype=float)
            - X_base[
                "green_mask_nbr_mean"
            ].to_numpy(dtype=float)
        )
    )

if not formula_checks:
    raise RuntimeError(
        "No green-cover dependent frozen feature formulas were available "
        "for verification. Review the Step-21 feature list before scenario use."
    )

formula_audit_rows = []

for feature, error_array in formula_checks.items():
    max_error = float(
        np.nanmax(
            np.abs(
                error_array
            )
        )
    )

    status = (
        "PASS"
        if max_error <= FORMULA_TOL
        else "FAIL"
    )

    formula_audit_rows.append(
        {
            "feature_formula": feature,
            "status": status,
            "max_abs_error": max_error,
            "tolerance": FORMULA_TOL,
        }
    )

formula_audit = pd.DataFrame(
    formula_audit_rows
)

formula_audit.to_csv(
    FORMULA_AUDIT_PATH,
    index=False,
)

print(
    "\nVerified retained green-cover feature formulas:"
)

print(
    formula_audit.to_string(
        index=False
    )
)

if (
    formula_audit[
        "status"
    ]
    != "PASS"
).any():
    raise RuntimeError(
        "Baseline scenario-feature formulas do not reproduce. "
        "Do not continue until fixed."
    )

print(
    "\nPASS: all retained green-cover dependent feature formulas verified."
)


# =============================================================================
# 3. REPRODUCE STEP-28 BASELINE MODEL OUTPUT
# =============================================================================

section(
    "3. REPRODUCE FROZEN STEP-28 BASELINE LIGHTGBM OUTPUT"
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

baseline_predicted_lst = pd.to_numeric(
    pred[
        "predicted_lst_c"
    ],
    errors="raise",
).to_numpy(dtype=float)

baseline_centered_anomaly = pd.to_numeric(
    pred[
        "centered_spatial_anomaly_c"
    ],
    errors="raise",
).to_numpy(dtype=float)

annual_level = pd.to_numeric(
    pred[
        "forecast_annual_mean_lst_c"
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
    f"Max raw prediction reproduction error: "
    f"{baseline_reproduction_error:.12e} °C"
)

if (
    baseline_reproduction_error
    > PREDICTION_REPRO_TOL
):
    raise RuntimeError(
        "Could not reproduce frozen Step-28 baseline model output."
    )

print(
    "PASS: frozen Step-28 baseline model output reproduced."
)


# =============================================================================
# 4. RUN INDEPENDENT LOCAL GREEN-COVER SCENARIOS
# =============================================================================

section(
    "4. RUN LOCAL GREEN-COVER SCENARIO PERTURBATIONS"
)

scenario_rows = []

scenario_model_outputs = {}
scenario_feature_audits = []

allowed_changed_features = {
    "green_mask",
    *scenario_dependent_features,
}

for pp in SCENARIO_INCREASE_PP:
    print(
        f"\nRunning +{pp} percentage-point green-cover scenario..."
    )

    X_scenario = X_base.copy()

    baseline_green = X_base[
        "green_mask"
    ].to_numpy(dtype=float)

    requested_fraction = (
        pp / 100.0
    )

    scenario_green = np.clip(
        baseline_green
        + requested_fraction,
        GREEN_MIN,
        GREEN_MAX,
    )

    actual_increase_fraction = (
        scenario_green
        - baseline_green
    )

    X_scenario[
        "green_mask"
    ] = scenario_green

    # Compute vegetation deficit internally even when the standalone
    # vegetation_deficit column was excluded from the frozen 93 features.
    scenario_vegetation_deficit = (
        1.0
        - scenario_green
    )

    if "vegetation_deficit" in X_scenario.columns:
        X_scenario[
            "vegetation_deficit"
        ] = scenario_vegetation_deficit

    if "ndbi_x_vegetation_deficit" in X_scenario.columns:
        X_scenario[
            "ndbi_x_vegetation_deficit"
        ] = (
            X_scenario[
                "NDBI"
            ].to_numpy(dtype=float)
            * scenario_vegetation_deficit
        )

    if "ndvi_x_green_mask" in X_scenario.columns:
        X_scenario[
            "ndvi_x_green_mask"
        ] = (
            X_scenario[
                "NDVI"
            ].to_numpy(dtype=float)
            * scenario_green
        )

    # Prior-year history is unchanged in this current-year local scenario.
    if "green_mask_delta1" in X_scenario.columns:
        X_scenario[
            "green_mask_delta1"
        ] = (
            scenario_green
            - X_scenario[
                "green_mask_lag1"
            ].to_numpy(dtype=float)
        )

    # Neighbours stay at baseline; only the selected grid changes.
    if "green_mask_nbr_contrast" in X_scenario.columns:
        X_scenario[
            "green_mask_nbr_contrast"
        ] = (
            scenario_green
            - X_scenario[
                "green_mask_nbr_mean"
            ].to_numpy(dtype=float)
        )

    if X_scenario.isna().any().any():
        raise RuntimeError(
            f"+{pp} pp scenario produced missing feature values."
        )

    if not np.isfinite(
        X_scenario.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            f"+{pp} pp scenario produced non-finite feature values."
        )

    # Strictly confirm that no other predictor was unintentionally changed.
    changed_features = []

    max_changes = {}

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

        max_changes[
            feature
        ] = max_change

        if max_change > UNCHANGED_FEATURE_TOL:
            changed_features.append(
                feature
            )

    unexpected_changed = sorted(
        set(changed_features)
        - allowed_changed_features
    )

    missing_expected_change = []

    # Some expected features can be numerically unchanged in edge cases;
    # therefore we do not require each one to change globally.
    if unexpected_changed:
        raise RuntimeError(
            f"+{pp} pp scenario changed unexpected features: "
            f"{unexpected_changed}"
        )

    scenario_feature_audits.append(
        {
            "scenario_increase_pp": pp,
            "changed_features": "; ".join(
                sorted(
                    changed_features
                )
            ),
            "unexpected_changed_feature_count": len(
                unexpected_changed
            ),
            "status": "PASS",
        }
    )

    scenario_raw = np.asarray(
        model.predict(
            X_scenario
        ),
        dtype=float,
    )

    scenario_model_outputs[
        pp
    ] = scenario_raw

    local_anomaly_delta = (
        scenario_raw
        - baseline_raw_model
    )

    scenario_lst = (
        baseline_predicted_lst
        + local_anomaly_delta
    )

    cooling_benefit = (
        baseline_predicted_lst
        - scenario_lst
    )

    # Equivalent by construction; useful strict check.
    cooling_identity_error = float(
        np.max(
            np.abs(
                cooling_benefit
                + local_anomaly_delta
            )
        )
    )

    if cooling_identity_error > PREDICTION_REPRO_TOL:
        raise RuntimeError(
            f"+{pp} pp cooling identity failed."
        )

    limited_by_upper_bound = (
        actual_increase_fraction
        < requested_fraction
        - 1e-12
    )

    for i in range(
        EXPECTED_ROWS
    ):
        benefit = float(
            cooling_benefit[i]
        )

        if benefit > 1e-12:
            direction = "cooling"
        elif benefit < -1e-12:
            direction = "warming"
        else:
            direction = "no_change"

        scenario_rows.append(
            {
                GRID_COL: str(
                    future.at[
                        i,
                        GRID_COL
                    ]
                ),
                YEAR_COL: int(
                    future.at[
                        i,
                        YEAR_COL
                    ]
                ),
                "area_name": str(
                    future.at[
                        i,
                        "area_name"
                    ]
                ),
                "scenario_id": f"green_plus_{pp}pp",
                "requested_green_increase_pp": pp,
                "baseline_green_mask": float(
                    baseline_green[i]
                ),
                "scenario_green_mask": float(
                    scenario_green[i]
                ),
                "actual_green_increase_pp": float(
                    actual_increase_fraction[i]
                    * 100.0
                ),
                "green_upper_bound_limited": bool(
                    limited_by_upper_bound[i]
                ),
                "baseline_predicted_lst_c": float(
                    baseline_predicted_lst[i]
                ),
                "forecast_annual_mean_lst_c": float(
                    annual_level[i]
                ),
                "baseline_centered_spatial_anomaly_c": float(
                    baseline_centered_anomaly[i]
                ),
                "baseline_raw_spatial_anomaly_c": float(
                    baseline_raw_model[i]
                ),
                "scenario_raw_spatial_anomaly_c": float(
                    scenario_raw[i]
                ),
                "local_anomaly_delta_c": float(
                    local_anomaly_delta[i]
                ),
                "scenario_predicted_lst_c": float(
                    scenario_lst[i]
                ),
                "cooling_benefit_c": benefit,
                "scenario_effect_direction": direction,
                "baseline_NDVI": float(
                    X_base.at[
                        i,
                        "NDVI"
                    ]
                ),
                "baseline_NDBI": float(
                    X_base.at[
                        i,
                        "NDBI"
                    ]
                ),
                "baseline_NDWI": float(
                    X_base.at[
                        i,
                        "NDWI"
                    ]
                )
                if "NDWI" in X_base.columns
                else np.nan,
            }
        )

    print(
        f"  Mean actual green increase : "
        f"{actual_increase_fraction.mean() * 100.0:.4f} pp"
    )

    print(
        f"  Upper-bound limited rows   : "
        f"{int(limited_by_upper_bound.sum()):,}"
    )

    print(
        f"  Mean cooling benefit       : "
        f"{cooling_benefit.mean():.6f} °C"
    )

    print(
        f"  Median cooling benefit     : "
        f"{np.median(cooling_benefit):.6f} °C"
    )

    print(
        f"  Min / Max benefit          : "
        f"{cooling_benefit.min():.6f} / "
        f"{cooling_benefit.max():.6f} °C"
    )


scenario_df = pd.DataFrame(
    scenario_rows
)

expected_scenario_rows = (
    EXPECTED_ROWS
    * len(
        SCENARIO_INCREASE_PP
    )
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

primary_df = (
    scenario_df[
        scenario_df[
            "requested_green_increase_pp"
        ]
        == PRIMARY_SCENARIO_PP
    ]
    .copy()
)

primary_df.to_csv(
    PRIMARY_SCENARIO_PATH,
    index=False,
)

for year in FUTURE_YEARS:
    year_all = (
        scenario_df[
            scenario_df[
                YEAR_COL
            ] == year
        ]
        .copy()
    )

    year_all.to_csv(
        SCENARIO_DIR
        / f"29a_green_cover_scenarios_{year}.csv",
        index=False,
    )

    for pp in SCENARIO_INCREASE_PP:
        (
            year_all[
                year_all[
                    "requested_green_increase_pp"
                ] == pp
            ]
            .to_csv(
                SCENARIO_DIR
                / f"29a_green_cover_{pp}pp_{year}.csv",
                index=False,
            )
        )

print(
    f"\nSaved: {ALL_SCENARIOS_PATH}"
)

print(
    f"Saved primary +{PRIMARY_SCENARIO_PP}pp: "
    f"{PRIMARY_SCENARIO_PATH}"
)


# =============================================================================
# 5. SCENARIO SUMMARY
# =============================================================================

section(
    "5. SCENARIO SUMMARY BY YEAR + GREEN-COVER INCREASE"
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
        mean_independent_scenario_lst_c=(
            "scenario_predicted_lst_c",
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
        min_cooling_benefit_c=(
            "cooling_benefit_c",
            "min",
        ),
        max_cooling_benefit_c=(
            "cooling_benefit_c",
            "max",
        ),
        std_cooling_benefit_c=(
            "cooling_benefit_c",
            "std",
        ),
    )
)

direction_counts = (
    scenario_df
    .pivot_table(
        index=[
            YEAR_COL,
            "requested_green_increase_pp",
        ],
        columns="scenario_effect_direction",
        values=GRID_COL,
        aggfunc="count",
        fill_value=0,
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
        "requested_green_increase_pp",
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

summary[
    "no_change_grid_pct"
] = (
    summary[
        "no_change_grid_count"
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

print(
    "\nIMPORTANT:"
)

print(
    "The mean scenario LST above summarizes independently evaluated local "
    "counterfactuals. It is NOT a simultaneous citywide scenario mean."
)


# =============================================================================
# 6. RESPONSE MONOTONICITY DIAGNOSTIC
# =============================================================================

section(
    "6. GREEN-INCREASE RESPONSE MONOTONICITY DIAGNOSTIC"
)

benefit_wide = (
    scenario_df
    .pivot(
        index=[
            GRID_COL,
            YEAR_COL,
        ],
        columns="requested_green_increase_pp",
        values="cooling_benefit_c",
    )
    .reset_index()
)

for pp in SCENARIO_INCREASE_PP:
    if pp not in benefit_wide.columns:
        raise RuntimeError(
            f"Missing +{pp}pp benefit column."
        )

b5 = benefit_wide[5].to_numpy(dtype=float)
b10 = benefit_wide[10].to_numpy(dtype=float)
b20 = benefit_wide[20].to_numpy(dtype=float)

monotonic_non_decreasing = (
    (b10 >= b5 - 1e-12)
    & (b20 >= b10 - 1e-12)
)

benefit_wide[
    "cooling_benefit_5pp_c"
] = b5

benefit_wide[
    "cooling_benefit_10pp_c"
] = b10

benefit_wide[
    "cooling_benefit_20pp_c"
] = b20

benefit_wide[
    "monotonic_non_decreasing"
] = monotonic_non_decreasing

benefit_wide.to_csv(
    MONOTONIC_PATH,
    index=False,
)

monotonic_pct = float(
    monotonic_non_decreasing.mean()
    * 100.0
)

print(
    f"Grid-year cases with non-decreasing modeled cooling "
    f"from +5 -> +10 -> +20 pp: {monotonic_pct:.2f}%"
)

print(
    "This is a diagnostic only. Tree-model responses are not forced to be "
    "monotonic, so non-monotonic cases are retained rather than corrected."
)


# =============================================================================
# 7. FRONTEND-READY YEARLY GEOJSON
# =============================================================================

section(
    "7. SAVE FRONTEND-READY YEARLY SCENARIO GEOJSON"
)

grid = gpd.read_file(
    GRID_WGS84_PATH
)

grid[GRID_COL] = (
    grid[GRID_COL]
    .astype(str)
)

if grid[GRID_COL].nunique() != EXPECTED_GRIDS:
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
            ] == year
        ][
            [
                GRID_COL,
                YEAR_COL,
                "area_name",
                "predicted_lst_c",
                "forecast_annual_mean_lst_c",
                "centered_spatial_anomaly_c",
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
                    ] == year
                )
                & (
                    scenario_df[
                        "requested_green_increase_pp"
                    ] == pp
                )
            ][
                [
                    GRID_COL,
                    "scenario_green_mask",
                    "actual_green_increase_pp",
                    "scenario_predicted_lst_c",
                    "cooling_benefit_c",
                    "scenario_effect_direction",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "scenario_green_mask":
                        f"green_mask_{pp}pp",
                    "actual_green_increase_pp":
                        f"actual_green_increase_{pp}pp",
                    "scenario_predicted_lst_c":
                        f"scenario_lst_{pp}pp_c",
                    "cooling_benefit_c":
                        f"cooling_benefit_{pp}pp_c",
                    "scenario_effect_direction":
                        f"effect_{pp}pp",
                }
            )
        )

        year_merge = year_merge.merge(
            subset,
            on=GRID_COL,
            how="left",
            validate="one_to_one",
        )

    year_gdf = (
        grid
        .merge(
            year_merge,
            on=GRID_COL,
            how="left",
            validate="one_to_one",
        )
    )

    if year_gdf[
        "baseline_predicted_lst_c"
    ].isna().any():
        raise RuntimeError(
            f"{year}: scenario GeoJSON merge produced missing rows."
        )

    geo_path = (
        MAP_DIR
        / f"29a_green_cover_scenarios_{year}_wgs84.geojson"
    )

    year_gdf.to_file(
        geo_path,
        driver="GeoJSON",
    )

    geojson_files.append(
        geo_path
    )

    # Cooling-benefit maps for every tested scenario.
    for pp in SCENARIO_INCREASE_PP:
        column = (
            f"cooling_benefit_{pp}pp_c"
        )

        fig, ax = plt.subplots(
            figsize=(10, 10)
        )

        year_gdf.plot(
            column=column,
            ax=ax,
            legend=True,
            edgecolor="none",
            legend_kwds={
                "label":
                    "Modeled local cooling benefit (°C)",
                "shrink": 0.72,
            },
        )

        if boundary is not None:
            boundary.boundary.plot(
                ax=ax,
                linewidth=1.4,
            )
        else:
            year_gdf.dissolve().boundary.plot(
                ax=ax,
                linewidth=1.4,
            )

        ax.set_title(
            (
                f"Kaduwela Local Green-Cover Scenario — {year}\n"
                f"+{pp} percentage points green cover"
            ),
            fontsize=14,
            pad=12,
        )

        ax.set_axis_off()

        plt.tight_layout()

        map_path = (
            FIGURE_DIR
            / (
                f"29a_green_cover_cooling_benefit_"
                f"{year}_{pp}pp.png"
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
        f"{year}: saved GeoJSON + "
        f"{len(SCENARIO_INCREASE_PP)} cooling-benefit maps"
    )


# =============================================================================
# 8. SUMMARY FIGURE
# =============================================================================

section(
    "8. SAVE SCENARIO SUMMARY FIGURE"
)

plot_summary = (
    summary.groupby(
        "requested_green_increase_pp",
        as_index=False,
    )[
        "mean_cooling_benefit_c"
    ]
    .mean()
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.plot(
    plot_summary[
        "requested_green_increase_pp"
    ],
    plot_summary[
        "mean_cooling_benefit_c"
    ],
    marker="o",
)

ax.set_xlabel(
    "Requested green-cover increase (percentage points)"
)

ax.set_ylabel(
    "Mean modeled local cooling benefit (°C)"
)

ax.set_title(
    "Green-Cover Scenario Sensitivity — 2026–2030"
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

print(
    f"Saved: {SUMMARY_FIGURE_PATH}"
)


# =============================================================================
# 9. STRICT STEP-29A AUDIT
# =============================================================================

section(
    "9. STRICT STEP-29A AUDIT"
)

scenario_feature_audit_df = pd.DataFrame(
    scenario_feature_audits
)

all_scenario_green = scenario_df[
    "scenario_green_mask"
].to_numpy(dtype=float)

all_actual_increase = scenario_df[
    "actual_green_increase_pp"
].to_numpy(dtype=float)

all_benefit = scenario_df[
    "cooling_benefit_c"
].to_numpy(dtype=float)

all_scenario_lst = scenario_df[
    "scenario_predicted_lst_c"
].to_numpy(dtype=float)

all_local_delta = scenario_df[
    "local_anomaly_delta_c"
].to_numpy(dtype=float)

scenario_identity_error = float(
    np.max(
        np.abs(
            all_benefit
            + all_local_delta
        )
    )
)

audit_rows = [
    {
        "check": "step28_audit",
        "status": "PASS",
        "details":
            "Step-28 strict audit verified fully PASS.",
    },
    {
        "check": "production_model_hash",
        "status": "PASS",
        "details": EXPECTED_HASHES[
            "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
        ],
    },
    {
        "check": "future_feature_hash",
        "status": "PASS",
        "details": EXPECTED_HASHES[
            "28_future_model_features_2026_2030.csv"
        ],
    },
    {
        "check": "future_prediction_hash",
        "status": "PASS",
        "details": EXPECTED_HASHES[
            "28_future_grid_lst_predictions_2026_2030.csv"
        ],
    },
    {
        "check": "baseline_raw_prediction_reproduction",
        "status": (
            "PASS"
            if baseline_reproduction_error
            <= PREDICTION_REPRO_TOL
            else "FAIL"
        ),
        "details":
            f"max_error_c={baseline_reproduction_error:.12e}",
    },
    {
        "check": "baseline_green_feature_formula_reproduction",
        "status": (
            "PASS"
            if (
                formula_audit[
                    "status"
                ]
                == "PASS"
            ).all()
            else "FAIL"
        ),
        "details":
            f"formulas_checked={len(formula_audit)}",
    },
    {
        "check": "scenario_levels",
        "status": (
            "PASS"
            if SCENARIO_INCREASE_PP
            == [5, 10, 20]
            else "FAIL"
        ),
        "details": str(
            SCENARIO_INCREASE_PP
        ),
    },
    {
        "check": "primary_scenario",
        "status": (
            "PASS"
            if PRIMARY_SCENARIO_PP == 5
            else "FAIL"
        ),
        "details":
            f"+{PRIMARY_SCENARIO_PP} percentage points",
    },
    {
        "check": "scenario_row_count",
        "status": (
            "PASS"
            if len(scenario_df)
            == expected_scenario_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(scenario_df)}, "
            f"expected={expected_scenario_rows}"
        ),
    },
    {
        "check": "scenario_rows_per_grid_year",
        "status": (
            "PASS"
            if (
                scenario_df.groupby(
                    [
                        GRID_COL,
                        YEAR_COL,
                    ]
                )
                .size()
                == len(
                    SCENARIO_INCREASE_PP
                )
            ).all()
            else "FAIL"
        ),
        "details":
            f"expected={len(SCENARIO_INCREASE_PP)} scenarios/grid-year",
    },
    {
        "check": "scenario_green_bounds",
        "status": (
            "PASS"
            if (
                np.min(
                    all_scenario_green
                )
                >= GREEN_MIN - 1e-12
                and np.max(
                    all_scenario_green
                )
                <= GREEN_MAX + 1e-12
            )
            else "FAIL"
        ),
        "details": (
            f"min={np.min(all_scenario_green):.6f}, "
            f"max={np.max(all_scenario_green):.6f}"
        ),
    },
    {
        "check": "actual_green_increase_nonnegative",
        "status": (
            "PASS"
            if np.min(
                all_actual_increase
            )
            >= -1e-10
            else "FAIL"
        ),
        "details":
            f"min_actual_increase_pp={np.min(all_actual_increase):.12f}",
    },
    {
        "check": "only_intended_features_changed",
        "status": (
            "PASS"
            if (
                scenario_feature_audit_df[
                    "unexpected_changed_feature_count"
                ]
                == 0
            ).all()
            else "FAIL"
        ),
        "details":
            "; ".join(
                scenario_feature_audit_df[
                    "changed_features"
                ].astype(str).tolist()
            ),
    },
    {
        "check": "scenario_outputs_finite",
        "status": (
            "PASS"
            if (
                np.isfinite(
                    all_scenario_lst
                ).all()
                and np.isfinite(
                    all_benefit
                ).all()
            )
            else "FAIL"
        ),
        "details":
            f"rows={len(scenario_df)}",
    },
    {
        "check": "cooling_benefit_identity",
        "status": (
            "PASS"
            if scenario_identity_error
            <= PREDICTION_REPRO_TOL
            else "FAIL"
        ),
        "details":
            f"max_error_c={scenario_identity_error:.12e}",
    },
    {
        "check": "independent_local_scenario_semantics",
        "status": "PASS",
        "details": (
            "Each grid-year is perturbed independently; "
            "scenario batch is not treated as simultaneous citywide greening."
        ),
    },
    {
        "check": "scenario_recentering",
        "status": "PASS",
        "details": (
            "Not applied. Local scenario effect uses scenario_raw - baseline_raw "
            "for the same grid-year."
        ),
    },
    {
        "check": "annual_thermal_level_changed",
        "status": "PASS",
        "details": (
            "No. Local scenario keeps the frozen Step-28 annual thermal level."
        ),
    },
    {
        "check": "model_retraining",
        "status": "PASS",
        "details":
            "None. Frozen Step-22B model loaded for inference only.",
    },
    {
        "check": "baseline_step28_predictions_modified",
        "status": "PASS",
        "details":
            "No. Scenario outputs are saved separately.",
    },
    {
        "check": "uncertainty_added_here",
        "status": "PASS",
        "details":
            "No. Scenario uncertainty/confidence is deferred to Step 30.",
    },
    {
        "check": "causal_or_guaranteed_cooling_claim",
        "status": "PASS",
        "details": (
            "No. Outputs are documented as model-based statistical sensitivity."
        ),
    },
    {
        "check": "monotonic_response_forced",
        "status": "PASS",
        "details": (
            "No. Non-monotonic tree-model responses are retained and audited."
        ),
    },
    {
        "check": "yearly_geojson_count",
        "status": (
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
            f"saved={len(geojson_files)}, expected={len(FUTURE_YEARS)}",
    },
    {
        "check": "cooling_benefit_map_count",
        "status": (
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
            else "FAIL"
        ),
        "details": (
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
        "One or more Step-29A audit checks failed."
    )


# =============================================================================
# 10. METHODOLOGY + REPORT
# =============================================================================

section(
    "10. SAVE STEP-29 METHODOLOGY + REPORT"
)

methodology = {
    "step": "29A",
    "version":
        "local_green_cover_direct_perturbation_v1",
    "analysis":
        "User-selectable local green-cover what-if cooling scenario simulation",
    "baseline_forecast_source":
        str(
            FUTURE_PRED_PATH
        ),
    "feature_source":
        str(
            FUTURE_FEATURE_PATH
        ),
    "frozen_model":
        str(
            MODEL_PATH
        ),
    "scenario_variable":
        "green_mask",
    "scenario_units":
        "percentage-point increase in grid green-cover fraction",
    "scenario_levels_pp":
        SCENARIO_INCREASE_PP,
    "primary_scenario_pp":
        PRIMARY_SCENARIO_PP,
    "green_bounds":
        [
            GREEN_MIN,
            GREEN_MAX,
        ],
    "scenario_scope": (
        "Independent local grid-cell counterfactual. "
        "One selected grid changes while all other grids stay at "
        "their Step-28 baseline forecast."
    ),
    "feature_selection_handling": (
        "Only green-cover-dependent predictors that survived the frozen "
        "Step-21 93-feature selection are updated. Excluded engineered "
        "intermediates are computed internally only when required to rebuild "
        "a retained predictor."
    ),
    "changed_current_year_frozen_features": sorted(
        allowed_changed_features
    ),
    "unchanged_prior_history_features": [
        "green_mask_lag1",
        "green_mask_lag2",
        "green_mask_hist3_mean",
        "green_mask_hist3_std",
        "green_mask_hist3_slope",
    ],
    "unchanged_neighbour_mean":
        "green_mask_nbr_mean",
    "local_effect_formula": (
        "scenario_raw_spatial_anomaly - baseline_raw_spatial_anomaly"
    ),
    "scenario_lst_formula": (
        "baseline Step-28 predicted LST + local anomaly delta"
    ),
    "cooling_benefit_formula": (
        "baseline Step-28 predicted LST - scenario predicted LST"
    ),
    "scenario_recentering": (
        "Not performed because rows represent independent local "
        "interventions rather than a simultaneous citywide intervention."
    ),
    "annual_level_policy": (
        "Frozen Step-28 annual Kaduwela thermal level is unchanged."
    ),
    "model_retraining":
        False,
    "step28_baseline_modified":
        False,
    "uncertainty":
        "Deferred to Step 30.",
    "interpretation_warning": (
        "Scenario results are modeled statistical sensitivity, not "
        "guaranteed physical cooling. Real-world outcomes depend on "
        "implementation conditions not represented in the model."
    ),
    "monotonicity_policy": (
        "No monotonic constraint is imposed. Non-monotonic LightGBM "
        "responses are retained and reported."
    ),
}

METHOD_PATH.write_text(
    json.dumps(
        methodology,
        indent=2,
    ),
    encoding="utf-8",
)

overall_scenario_summary = (
    scenario_df
    .groupby(
        "requested_green_increase_pp",
        as_index=False,
    )
    .agg(
        rows=(
            GRID_COL,
            "size",
        ),
        mean_actual_green_increase_pp=(
            "actual_green_increase_pp",
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
        max_cooling_benefit_c=(
            "cooling_benefit_c",
            "max",
        ),
        min_cooling_benefit_c=(
            "cooling_benefit_c",
            "min",
        ),
    )
)

report_lines = [
    "STEP 29A - GREEN-COVER WHAT-IF COOLING SCENARIO SIMULATION",
    "=" * 118,
    "",
    "SCENARIO DESIGN",
    "-" * 118,
    (
        "Primary: +5 percentage points green cover. "
        "Sensitivity levels: +10 pp and +20 pp."
    ),
    (
        "Each grid-year is an independent local intervention; "
        "the batch is not a simultaneous citywide greening scenario."
    ),
    "",
    "MODEL POLICY",
    "-" * 118,
    (
        "The frozen Step-22B LightGBM is used without retraining. "
        "Step-28 baseline forecasts are not modified."
    ),
    (
        "Scenario LST = Step-28 baseline LST + "
        "(scenario raw anomaly - baseline raw anomaly)."
    ),
    (
        "Cooling benefit = baseline LST - scenario LST. "
        "Positive values indicate modeled cooling."
    ),
    "",
    "OVERALL SCENARIO SUMMARY",
    "-" * 118,
    overall_scenario_summary.to_string(index=False),
    "",
    "YEAR-BY-YEAR SUMMARY",
    "-" * 118,
    summary.to_string(index=False),
    "",
    "MONOTONICITY DIAGNOSTIC",
    "-" * 118,
    (
        f"{monotonic_pct:.2f}% of grid-year cases show non-decreasing "
        "modeled cooling from +5 to +10 to +20 pp."
    ),
    (
        "Non-monotonic cases are retained; tree-model monotonicity "
        "is not forced."
    ),
    "",
    "INTERPRETATION LIMITATION",
    "-" * 118,
    (
        "These outputs represent modeled statistical sensitivity, "
        "not guaranteed physical cooling outcomes."
    ),
    (
        "Real cooling depends on vegetation species, canopy geometry, "
        "placement, irrigation, maintenance, land conversion, and other "
        "factors not explicitly represented."
    ),
    (
        "Uncertainty/confidence is not added in Step 29; it is handled "
        "in Step 30."
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
# 11. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "11. REPRODUCIBILITY HASHES"
)

hash_files = [
    FEATURE_LIST_PATH,
    MODEL_PATH,
    FUTURE_FEATURE_PATH,
    FUTURE_PRED_PATH,
    STEP28_AUDIT_PATH,
    GRID_WGS84_PATH,
    ALL_SCENARIOS_PATH,
    PRIMARY_SCENARIO_PATH,
    SUMMARY_PATH,
    MONOTONIC_PATH,
    FORMULA_AUDIT_PATH,
    AUDIT_PATH,
    METHOD_PATH,
    REPORT_PATH,
    SUMMARY_FIGURE_PATH,
]

for year in FUTURE_YEARS:
    hash_files.append(
        SCENARIO_DIR
        / f"29a_green_cover_scenarios_{year}.csv"
    )

    for pp in SCENARIO_INCREASE_PP:
        hash_files.append(
            SCENARIO_DIR
            / f"29a_green_cover_{pp}pp_{year}.csv"
        )

    hash_files.append(
        MAP_DIR
        / f"29a_green_cover_scenarios_{year}_wgs84.geojson"
    )

for path in map_files:
    hash_files.append(
        path
    )

# Remove duplicate paths while preserving order.
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
# FINAL
# =============================================================================

section(
    "STEP 29A COMPLETED SUCCESSFULLY"
)

print(
    "Green-cover what-if scenario simulation completed.\n"
    f"Future baseline grid-years: {EXPECTED_ROWS:,}.\n"
    f"Scenario levels: {SCENARIO_INCREASE_PP} percentage points.\n"
    f"Primary scenario: +{PRIMARY_SCENARIO_PP} percentage points.\n"
    f"Total independent scenario rows: {len(scenario_df):,}.\n"
    "Step-28 baseline LightGBM output was reproduced before scenario inference.\n"
    "Only mathematically dependent current-year green-cover features were changed.\n"
    "The frozen Step-22B LightGBM was used for scenario inference only.\n"
    "Step-28 baseline forecasts were not modified.\n"
    "Scenario effects were computed as local raw-anomaly deltas without batch recentering.\n"
    "Positive cooling_benefit_c means the scenario is cooler than the same grid-year baseline.\n"
    "Separate 2026-2030 CSVs and frontend-ready GeoJSON files were saved.\n"
    "Report-quality cooling-benefit maps were saved for +5, +10 and +20 pp scenarios.\n"
    "No monotonic cooling response was forced or fabricated.\n"
    "Outputs are model-based planning sensitivity, not guaranteed physical outcomes.\n"
    "Uncertainty/confidence remains deferred to Step 30. Next scenario stage is Step 29B.\n"
    "\nNEXT: review Step-29 scenario results carefully before Step 29B built-up-density scenario."
)
