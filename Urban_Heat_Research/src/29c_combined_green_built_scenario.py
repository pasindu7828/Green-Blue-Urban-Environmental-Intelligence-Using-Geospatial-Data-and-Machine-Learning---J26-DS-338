
"""
STEP 29C - COMBINED GREEN + BUILT-UP INTENSITY WHAT-IF SCENARIO
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Combine the FINAL 29A green-cover scenario mechanism and FINAL 29B built-up
intensity scenario mechanism using the SAME frozen Step-22B LightGBM.

29A frontend choices:
    +5 / +10 / +20 percentage points green cover

29B frontend choices:
    LOW / MODERATE / HIGH built-up intensity reduction (NDBI)

This step evaluates all 3 x 3 = 9 combinations for every grid-year in
2026-2030.

Important methodology
---------------------
The combined scenario is NOT calculated by simply adding the separate 29A and
29B temperature changes.

Instead:
1. Start from the frozen Step-28 feature vector.
2. Apply the finalized 29A green-dependent feature changes.
3. Apply the finalized 29B NDBI-dependent feature changes.
4. Recompute the feature that depends jointly on BOTH interventions:
       ndbi_x_vegetation_deficit
       = scenario_NDBI * (1 - scenario_green_mask)
5. Run the frozen Step-22B LightGBM once.
6. Compute:
       combined_delta = combined_scenario_LST - baseline_LST

For diagnosis only:
       additive_reference
         = green_only_delta + built_only_delta

       model_non_additivity
         = combined_delta - additive_reference

"Model non-additivity" is NOT causal synergy.

No new ML model is trained.
No Step-28 baseline is modified.
Uncertainty remains Step 30.
"""

from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except ImportError as exc:
    raise ImportError(
        "geopandas is required. Install with: pip install geopandas"
    ) from exc


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"

GREEN_LEVELS = [5, 10, 20]
BUILT_LEVELS = ["LOW", "MODERATE", "HIGH"]
FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]

EXPECTED_GRIDS = 2194
EXPECTED_FUTURE_ROWS = EXPECTED_GRIDS * len(FUTURE_YEARS)
EXPECTED_FEATURES = 93
EXPECTED_INDIVIDUAL_ROWS = EXPECTED_FUTURE_ROWS * 3
EXPECTED_COMBINED_ROWS = EXPECTED_FUTURE_ROWS * 9

PRIMARY_GREEN_PP = 10
PRIMARY_BUILT_LEVEL = "MODERATE"

PRED_TOL = 1e-10
FEATURE_TOL = 1e-12
FORMULA_TOL = 1e-10
ZERO_TOL = 1e-10

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
    "29a_v2_persistent_green_scenario_features_2026_2030.csv":
        "20f29b33d6e1a080fb94f3321ab2e509ac23f456b014b475a5e1f6e5e0a93295",
    "29a_v2_persistent_green_scenario_audit.csv":
        "eaac9e8e0e16066c3a0a24ca4a9328a3c411f04d17e15940175c87c7d8c751d4",
    "29b_built_up_intensity_scenario_features_2026_2030.csv":
        "2798e58b2751ba8a37a2f98fd8e17bd2b64b6174da1994d8e39a33c90801658c",
    "29b_built_up_intensity_scenario_audit.csv":
        "044c1a70f3643888ed279a4d67442b2d2e5bf9d1fdf64a20b77ffb26e1f851dc",
}


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_LIST_PATH = ROOT / "outputs" / "reports" / "21_final_feature_list.csv"
MODEL_PATH = ROOT / "models" / "final" / "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"

FUTURE_FEATURE_PATH = ROOT / "outputs" / "forecasting" / "28_future_model_features_2026_2030.csv"
FUTURE_PRED_PATH = ROOT / "outputs" / "forecasting" / "28_future_grid_lst_predictions_2026_2030.csv"
STEP28_AUDIT_PATH = ROOT / "outputs" / "reports" / "28_future_prediction_audit.csv"

GREEN_FEATURE_PATH = ROOT / "outputs" / "scenarios" / "29a_v2_persistent_green_scenario_features_2026_2030.csv"
GREEN_AUDIT_PATH = ROOT / "outputs" / "reports" / "29a_v2_persistent_green_scenario_audit.csv"

BUILT_FEATURE_PATH = ROOT / "outputs" / "scenarios" / "29b_built_up_intensity_scenario_features_2026_2030.csv"
BUILT_AUDIT_PATH = ROOT / "outputs" / "reports" / "29b_built_up_intensity_scenario_audit.csv"

GRID_WGS84_PATH = ROOT / "data" / "processed" / "kaduwela_grid_geometry_wgs84.geojson"
BOUNDARY_KML_PATH = ROOT / "data" / "raw" / "boundary" / "kaduwela_kmc_boundary.kml"

SCENARIO_DIR = ROOT / "outputs" / "scenarios"
REPORT_DIR = ROOT / "outputs" / "reports"
MAP_DIR = ROOT / "outputs" / "maps"
FIGURE_DIR = ROOT / "outputs" / "figures"

for folder in [SCENARIO_DIR, REPORT_DIR, MAP_DIR, FIGURE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

COMBINED_PATH = SCENARIO_DIR / "29c_combined_green_built_scenarios_2026_2030.csv"
PRIMARY_PATH = SCENARIO_DIR / "29c_combined_primary_green10_builtmoderate_2026_2030.csv"
FEATURE_EXPORT_PATH = SCENARIO_DIR / "29c_combined_scenario_features_2026_2030.csv"
SUMMARY_PATH = SCENARIO_DIR / "29c_combined_scenario_summary.csv"
OVERALL_SUMMARY_PATH = SCENARIO_DIR / "29c_combined_scenario_overall_summary.csv"

FEATURE_AUDIT_PATH = REPORT_DIR / "29c_combined_feature_change_audit.csv"
NONADDITIVITY_PATH = REPORT_DIR / "29c_combined_non_additivity_summary.csv"
AUDIT_PATH = REPORT_DIR / "29c_combined_scenario_audit.csv"
METHOD_PATH = REPORT_DIR / "29c_combined_scenario_methodology.json"
REPORT_PATH = REPORT_DIR / "29c_combined_scenario_report.txt"
HASH_PATH = REPORT_DIR / "29c_combined_scenario_hashes.csv"

SUMMARY_FIGURE_PATH = FIGURE_DIR / "29c_combined_mean_lst_change_by_scenario.png"
NONADD_FIGURE_PATH = FIGURE_DIR / "29c_combined_mean_non_additivity.png"


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
            f"{label} hash mismatch.\nExpected: {expected}\nActual  : {actual}"
        )

    print(f"PASS hash: {label} = {actual}")
    return actual


def require_all_pass(path, label):
    audit = pd.read_csv(path)

    status_col = None
    for candidate in ["status", "Status"]:
        if candidate in audit.columns:
            status_col = candidate
            break

    if status_col is None:
        raise RuntimeError(f"{label}: no status column.")

    bad = audit[
        audit[status_col].astype(str).str.upper() != "PASS"
    ]

    if not bad.empty:
        print(bad.to_string(index=False))
        raise RuntimeError(f"{label} is not fully PASS.")

    print(f"PASS: {label} is fully PASS.")


def detect_feature_column(df):
    for candidate in ["feature", "feature_name", "predictor"]:
        if candidate in df.columns:
            values = df[candidate].dropna().astype(str).tolist()
            if len(values) == EXPECTED_FEATURES:
                return candidate

    for column in df.columns:
        values = df[column].dropna().astype(str).tolist()
        if len(values) == EXPECTED_FEATURES and len(set(values)) == EXPECTED_FEATURES:
            return column

    raise RuntimeError("Could not detect frozen 93-feature column.")


def detect_green_level_column(df):
    preferred = [
        "requested_green_increase_pp",
        "green_increase_pp",
        "scenario_green_increase_pp",
        "scenario_pp",
    ]

    for column in preferred:
        if column in df.columns:
            values = sorted(
                pd.to_numeric(df[column], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            if values == GREEN_LEVELS:
                return column

    # Fallback: find any metadata column whose unique integer values are exactly 5,10,20.
    for column in df.columns:
        if column in [GRID_COL, YEAR_COL]:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        values = sorted(
            numeric.dropna().astype(int).unique().tolist()
        )

        if values == GREEN_LEVELS:
            return column

    raise RuntimeError(
        "Could not detect 29A green scenario level column with values [5,10,20]."
    )


def prepare_feature_slice(df, mask, ordered_features, baseline_keys, label):
    # IMPORTANT:
    # The frozen model feature list may itself contain "year".
    # Therefore [GRID_COL, YEAR_COL, *ordered_features] can request "year"
    # twice and create duplicate column labels inside the temporary frame.
    # Build a unique ordered selection list before sorting.
    select_columns = list(
        dict.fromkeys(
            [
                GRID_COL,
                YEAR_COL,
                *ordered_features,
            ]
        )
    )

    sub = (
        df.loc[
            mask,
            select_columns,
        ]
        .copy()
        .sort_values([YEAR_COL, GRID_COL])
        .reset_index(drop=True)
    )

    if len(sub) != EXPECTED_FUTURE_ROWS:
        raise RuntimeError(
            f"{label}: rows={len(sub)}, expected={EXPECTED_FUTURE_ROWS}"
        )

    if not sub[[GRID_COL, YEAR_COL]].equals(baseline_keys):
        raise RuntimeError(f"{label}: grid-year keys do not align with Step-28.")

    X = sub[ordered_features].copy()

    for feature in ordered_features:
        X[feature] = pd.to_numeric(X[feature], errors="raise")

    if X.isna().any().any():
        raise RuntimeError(f"{label}: missing values in scenario feature matrix.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{label}: non-finite values in scenario feature matrix.")

    return X


def get_changed_features(X_scenario, X_base, ordered_features):
    changed = []

    for feature in ordered_features:
        max_change = float(
            np.max(
                np.abs(
                    X_scenario[feature].to_numpy(dtype=float)
                    - X_base[feature].to_numpy(dtype=float)
                )
            )
        )

        if max_change > FEATURE_TOL:
            changed.append(feature)

    return changed


def effect_direction(values):
    values = np.asarray(values, dtype=float)

    out = np.full(len(values), "no_change", dtype=object)
    out[values < -ZERO_TOL] = "cooling"
    out[values > ZERO_TOL] = "warming"

    return out


def nonadditivity_direction(values):
    values = np.asarray(values, dtype=float)

    out = np.full(len(values), "approximately_additive", dtype=object)

    # Negative: joint model output is cooler / less warm than simple sum.
    out[values < -ZERO_TOL] = "more_cooling_than_additive"

    # Positive: joint model output is warmer / less cool than simple sum.
    out[values > ZERO_TOL] = "more_warming_than_additive"

    return out


def load_boundary():
    if not BOUNDARY_KML_PATH.exists():
        return None

    try:
        boundary = gpd.read_file(BOUNDARY_KML_PATH)

        if boundary.empty:
            return None

        if boundary.crs is None:
            boundary = boundary.set_crs("EPSG:4326")
        else:
            boundary = boundary.to_crs("EPSG:4326")

        return boundary

    except Exception as exc:
        print(f"WARNING: boundary KML could not be loaded: {exc}")
        return None


# =============================================================================
# 0. VERIFY FINALIZED INPUTS
# =============================================================================

section("STEP 29C - COMBINED GREEN + BUILT-UP INTENSITY WHAT-IF SCENARIO")

verify_hash(
    FEATURE_LIST_PATH,
    EXPECTED_HASHES["21_final_feature_list.csv"],
    "Frozen Step-21 feature list",
)

verify_hash(
    MODEL_PATH,
    EXPECTED_HASHES["22b_production_lightgbm_anomaly_trained_2018_2025.joblib"],
    "Frozen Step-22B production LightGBM",
)

verify_hash(
    FUTURE_FEATURE_PATH,
    EXPECTED_HASHES["28_future_model_features_2026_2030.csv"],
    "Step-28 future feature matrix",
)

verify_hash(
    FUTURE_PRED_PATH,
    EXPECTED_HASHES["28_future_grid_lst_predictions_2026_2030.csv"],
    "Step-28 future LST predictions",
)

verify_hash(
    STEP28_AUDIT_PATH,
    EXPECTED_HASHES["28_future_prediction_audit.csv"],
    "Step-28 audit",
)

verify_hash(
    GRID_WGS84_PATH,
    EXPECTED_HASHES["kaduwela_grid_geometry_wgs84.geojson"],
    "Corrected WGS84 grid geometry",
)

verify_hash(
    GREEN_FEATURE_PATH,
    EXPECTED_HASHES["29a_v2_persistent_green_scenario_features_2026_2030.csv"],
    "Final 29A V2 scenario feature matrix",
)

verify_hash(
    GREEN_AUDIT_PATH,
    EXPECTED_HASHES["29a_v2_persistent_green_scenario_audit.csv"],
    "Final 29A V2 audit",
)

verify_hash(
    BUILT_FEATURE_PATH,
    EXPECTED_HASHES["29b_built_up_intensity_scenario_features_2026_2030.csv"],
    "Final 29B scenario feature matrix",
)

verify_hash(
    BUILT_AUDIT_PATH,
    EXPECTED_HASHES["29b_built_up_intensity_scenario_audit.csv"],
    "Final 29B audit",
)

require_all_pass(STEP28_AUDIT_PATH, "Step-28 audit")
require_all_pass(GREEN_AUDIT_PATH, "Step-29A V2 audit")
require_all_pass(BUILT_AUDIT_PATH, "Step-29B audit")


# =============================================================================
# 1. LOAD BASELINE + FROZEN MODEL
# =============================================================================

section("1. LOAD FROZEN STEP-28 BASELINE")

feature_list_df = pd.read_csv(FEATURE_LIST_PATH)
feature_col = detect_feature_column(feature_list_df)

ordered_features = (
    feature_list_df[feature_col]
    .dropna()
    .astype(str)
    .tolist()
)

if len(ordered_features) != EXPECTED_FEATURES:
    raise RuntimeError(
        f"Frozen feature count={len(ordered_features)}, expected={EXPECTED_FEATURES}"
    )

print(
    f"Frozen feature list contains raw 'year': "
    f"{YEAR_COL in ordered_features}"
)

future = pd.read_csv(
    FUTURE_FEATURE_PATH,
    dtype={GRID_COL: str, "area_name": str},
)

pred = pd.read_csv(
    FUTURE_PRED_PATH,
    dtype={GRID_COL: str, "area_name": str},
)

for frame, label in [
    (future, "Step-28 feature matrix"),
    (pred, "Step-28 prediction matrix"),
]:
    frame[YEAR_COL] = pd.to_numeric(frame[YEAR_COL], errors="raise").astype(int)
    frame[GRID_COL] = frame[GRID_COL].astype(str)

    if len(frame) != EXPECTED_FUTURE_ROWS:
        raise RuntimeError(
            f"{label}: rows={len(frame)}, expected={EXPECTED_FUTURE_ROWS}"
        )

future = (
    future.sort_values([YEAR_COL, GRID_COL])
    .reset_index(drop=True)
)

pred = (
    pred.sort_values([YEAR_COL, GRID_COL])
    .reset_index(drop=True)
)

baseline_keys = future[[GRID_COL, YEAR_COL]].copy()

if not baseline_keys.equals(pred[[GRID_COL, YEAR_COL]]):
    raise RuntimeError("Step-28 feature and prediction keys do not align.")

missing_features = [
    feature
    for feature in ordered_features
    if feature not in future.columns
]

if missing_features:
    raise RuntimeError(
        "Step-28 feature matrix missing frozen features:\n"
        + str(missing_features)
    )

X_base = future[ordered_features].copy()

for feature in ordered_features:
    X_base[feature] = pd.to_numeric(X_base[feature], errors="raise")

if X_base.isna().any().any():
    raise RuntimeError("Step-28 baseline contains missing feature values.")

if not np.isfinite(X_base.to_numpy(dtype=float)).all():
    raise RuntimeError("Step-28 baseline contains non-finite feature values.")

model = joblib.load(MODEL_PATH)

baseline_raw = np.asarray(
    model.predict(X_base),
    dtype=float,
)

saved_raw = pd.to_numeric(
    pred["raw_spatial_anomaly_prediction_c"],
    errors="raise",
).to_numpy(dtype=float)

baseline_lst = pd.to_numeric(
    pred["predicted_lst_c"],
    errors="raise",
).to_numpy(dtype=float)

baseline_reproduction_error = float(
    np.max(np.abs(baseline_raw - saved_raw))
)

print(f"Future rows     : {len(future):,}")
print(f"Unique grids    : {future[GRID_COL].nunique():,}")
print(f"Frozen features : {len(ordered_features)}")
print(
    f"Baseline raw-prediction reproduction error: "
    f"{baseline_reproduction_error:.12e} °C"
)

if baseline_reproduction_error > PRED_TOL:
    raise RuntimeError("Frozen Step-28 baseline was not reproduced.")


# =============================================================================
# 2. LOAD 29A V2 + 29B FINAL FEATURE MATRICES
# =============================================================================

section("2. LOAD FINAL 29A V2 + FINAL 29B SCENARIO FEATURES")

green_df = pd.read_csv(
    GREEN_FEATURE_PATH,
    dtype={GRID_COL: str, "area_name": str},
)

built_df = pd.read_csv(
    BUILT_FEATURE_PATH,
    dtype={GRID_COL: str, "area_name": str},
)

for frame, label in [
    (green_df, "29A V2"),
    (built_df, "29B"),
]:
    frame[YEAR_COL] = pd.to_numeric(frame[YEAR_COL], errors="raise").astype(int)
    frame[GRID_COL] = frame[GRID_COL].astype(str)

    if len(frame) != EXPECTED_INDIVIDUAL_ROWS:
        raise RuntimeError(
            f"{label}: rows={len(frame)}, expected={EXPECTED_INDIVIDUAL_ROWS}"
        )

    for feature in ordered_features:
        if feature not in frame.columns:
            raise RuntimeError(
                f"{label}: missing frozen feature {feature}"
            )

green_level_col = detect_green_level_column(green_df)

green_level_values = pd.to_numeric(
    green_df[green_level_col],
    errors="raise",
).astype(int)

if "scenario_level" not in built_df.columns:
    raise RuntimeError(
        "29B scenario feature file has no scenario_level column."
    )

built_df["scenario_level"] = (
    built_df["scenario_level"]
    .astype(str)
    .str.upper()
)

found_green_levels = sorted(
    green_level_values.unique().tolist()
)

found_built_levels = sorted(
    built_df["scenario_level"].unique().tolist()
)

print(f"Green-level metadata column : {green_level_col}")
print(f"Green levels                : {found_green_levels}")
print(f"Built-up levels             : {found_built_levels}")

if found_green_levels != GREEN_LEVELS:
    raise RuntimeError(
        f"Unexpected 29A levels: {found_green_levels}"
    )

if set(found_built_levels) != set(BUILT_LEVELS):
    raise RuntimeError(
        f"Unexpected 29B levels: {found_built_levels}"
    )


# =============================================================================
# 3. REPRODUCE INDIVIDUAL SCENARIO FEATURE CHANGES
# =============================================================================

section("3. REPRODUCE FINAL 29A AND 29B FEATURE CHANGES")

green_X = {}
green_changed = {}
green_delta = {}

for green_pp in GREEN_LEVELS:
    Xg = prepare_feature_slice(
        green_df,
        green_level_values == green_pp,
        ordered_features,
        baseline_keys,
        f"29A V2 +{green_pp} pp",
    )

    changed = get_changed_features(
        Xg,
        X_base,
        ordered_features,
    )

    raw = np.asarray(
        model.predict(Xg),
        dtype=float,
    )

    green_X[green_pp] = Xg
    green_changed[green_pp] = set(changed)
    green_delta[green_pp] = raw - baseline_raw

    print(
        f"+{green_pp} pp green: {len(changed)} changed frozen features"
    )
    print("  " + "; ".join(changed))


built_X = {}
built_changed = {}
built_delta = {}

for built_level in BUILT_LEVELS:
    Xb = prepare_feature_slice(
        built_df,
        built_df["scenario_level"] == built_level,
        ordered_features,
        baseline_keys,
        f"29B {built_level}",
    )

    changed = get_changed_features(
        Xb,
        X_base,
        ordered_features,
    )

    raw = np.asarray(
        model.predict(Xb),
        dtype=float,
    )

    built_X[built_level] = Xb
    built_changed[built_level] = set(changed)
    built_delta[built_level] = raw - baseline_raw

    print(
        f"{built_level} built-up: {len(changed)} changed frozen features"
    )
    print("  " + "; ".join(changed))


green_union = set().union(*green_changed.values())
built_union = set().union(*built_changed.values())

shared_changed = green_union & built_union

allowed_shared = set()

if "ndbi_x_vegetation_deficit" in ordered_features:
    allowed_shared.add("ndbi_x_vegetation_deficit")

unexpected_shared = shared_changed - allowed_shared

print("\nShared features changed by both finalized scenarios:")
print(
    "  "
    + (
        "; ".join(sorted(shared_changed))
        if shared_changed
        else "NONE"
    )
)

if unexpected_shared:
    raise RuntimeError(
        "Unexpected overlap between 29A and 29B feature mechanisms:\n"
        + str(sorted(unexpected_shared))
    )


# =============================================================================
# 4. BUILD TRUE JOINT FEATURE VECTORS + PREDICT
# =============================================================================

section("4. RUN ALL 9 TRUE JOINT SCENARIO COMBINATIONS")

result_frames = []
feature_frames = []
feature_audit_rows = []

for green_pp in GREEN_LEVELS:
    for built_level in BUILT_LEVELS:
        scenario_id = f"G{green_pp}pp_B{built_level}"

        Xg = green_X[green_pp]
        Xb = built_X[built_level]

        Xc = X_base.copy()

        # Apply green-dependent changes, excluding features that need a
        # special true-joint recomputation.
        for feature in sorted(
            green_changed[green_pp] - allowed_shared
        ):
            Xc[feature] = Xg[feature].to_numpy(dtype=float)

        # Apply built-up-dependent changes, excluding true-joint features.
        for feature in sorted(
            built_changed[built_level] - allowed_shared
        ):
            Xc[feature] = Xb[feature].to_numpy(dtype=float)

        # Recompute shared green x built-up feature using BOTH scenario inputs.
        joint_formula_error = 0.0

        if "ndbi_x_vegetation_deficit" in allowed_shared:
            if (
                "NDBI" not in Xc.columns
                or "green_mask" not in Xc.columns
            ):
                raise RuntimeError(
                    "Joint interaction requires NDBI and green_mask."
                )

            expected_joint = (
                Xc["NDBI"].to_numpy(dtype=float)
                * (
                    1.0
                    - Xc["green_mask"].to_numpy(dtype=float)
                )
            )

            Xc["ndbi_x_vegetation_deficit"] = expected_joint

            joint_formula_error = float(
                np.max(
                    np.abs(
                        Xc["ndbi_x_vegetation_deficit"]
                        .to_numpy(dtype=float)
                        - expected_joint
                    )
                )
            )

        if Xc.isna().any().any():
            raise RuntimeError(
                f"{scenario_id}: combined matrix contains missing values."
            )

        if not np.isfinite(Xc.to_numpy(dtype=float)).all():
            raise RuntimeError(
                f"{scenario_id}: combined matrix contains non-finite values."
            )

        changed_combined = set(
            get_changed_features(
                Xc,
                X_base,
                ordered_features,
            )
        )

        intended_union = (
            green_changed[green_pp]
            | built_changed[built_level]
        )

        unexpected = changed_combined - intended_union

        if unexpected:
            raise RuntimeError(
                f"{scenario_id}: unexpected combined features changed: "
                f"{sorted(unexpected)}"
            )

        required_core = set()

        if "green_mask" in ordered_features:
            required_core.add("green_mask")

        if "NDBI" in ordered_features:
            required_core.add("NDBI")

        missing_core = required_core - changed_combined

        if missing_core:
            raise RuntimeError(
                f"{scenario_id}: core scenario feature(s) failed to change: "
                f"{sorted(missing_core)}"
            )

        combined_raw = np.asarray(
            model.predict(Xc),
            dtype=float,
        )

        combined_delta = combined_raw - baseline_raw
        combined_lst = baseline_lst + combined_delta

        green_only_delta = green_delta[green_pp]
        built_only_delta = built_delta[built_level]

        additive_reference = (
            green_only_delta
            + built_only_delta
        )

        nonadditivity = (
            combined_delta
            - additive_reference
        )

        actual_green_increase_pp = (
            (
                Xg["green_mask"].to_numpy(dtype=float)
                - X_base["green_mask"].to_numpy(dtype=float)
            )
            * 100.0
        )

        actual_ndbi_reduction = (
            X_base["NDBI"].to_numpy(dtype=float)
            - Xb["NDBI"].to_numpy(dtype=float)
        )

        output = pd.DataFrame(
            {
                GRID_COL:
                    future[GRID_COL].astype(str),
                YEAR_COL:
                    future[YEAR_COL].astype(int),
                "area_name":
                    future["area_name"].astype(str),
                "scenario_id":
                    scenario_id,
                "green_increase_pp_requested":
                    green_pp,
                "green_increase_pp_actual":
                    actual_green_increase_pp,
                "built_up_level":
                    built_level,
                "actual_ndbi_reduction":
                    actual_ndbi_reduction,
                "baseline_green_mask":
                    X_base["green_mask"].to_numpy(dtype=float),
                "scenario_green_mask":
                    Xg["green_mask"].to_numpy(dtype=float),
                "baseline_ndbi":
                    X_base["NDBI"].to_numpy(dtype=float),
                "scenario_ndbi":
                    Xb["NDBI"].to_numpy(dtype=float),
                "baseline_predicted_lst_c":
                    baseline_lst,
                "combined_scenario_predicted_lst_c":
                    combined_lst,
                "combined_modeled_lst_change_c":
                    combined_delta,
                "combined_cooling_benefit_c":
                    -combined_delta,
                "combined_effect_direction":
                    effect_direction(combined_delta),
                "green_only_modeled_lst_change_c":
                    green_only_delta,
                "built_only_modeled_lst_change_c":
                    built_only_delta,
                "additive_expected_lst_change_c":
                    additive_reference,
                "model_non_additivity_c":
                    nonadditivity,
                "non_additivity_direction":
                    nonadditivity_direction(nonadditivity),
            }
        )

        result_frames.append(output)

        # Build a feature export with UNIQUE column names.
        # If "year" is already one of the frozen model features, do not add
        # another metadata "year" column.
        feature_export = Xc.reset_index(drop=True).copy()

        metadata_columns = [
            GRID_COL,
            YEAR_COL,
            "area_name",
            "scenario_id",
            "green_increase_pp_requested",
            "built_up_level",
        ]

        metadata_source = output[
            metadata_columns
        ].reset_index(drop=True)

        for column in reversed(metadata_columns):
            if column in feature_export.columns:
                # Verify the existing model-feature column agrees with metadata.
                if column == YEAR_COL:
                    existing = pd.to_numeric(
                        feature_export[column],
                        errors="raise",
                    ).astype(int).to_numpy()

                    expected = pd.to_numeric(
                        metadata_source[column],
                        errors="raise",
                    ).astype(int).to_numpy()

                    if not np.array_equal(existing, expected):
                        raise RuntimeError(
                            f"{scenario_id}: model year feature does not match metadata year."
                        )

                # Keep the existing unique column; do not duplicate it.
                continue

            feature_export.insert(
                0,
                column,
                metadata_source[column].to_numpy(),
            )

        if feature_export.columns.duplicated().any():
            duplicated = feature_export.columns[
                feature_export.columns.duplicated()
            ].tolist()

            raise RuntimeError(
                f"{scenario_id}: duplicate feature-export columns remain: {duplicated}"
            )

        feature_frames.append(feature_export)

        feature_audit_rows.append(
            {
                "scenario_id":
                    scenario_id,
                "green_pp":
                    green_pp,
                "built_level":
                    built_level,
                "changed_feature_count":
                    len(changed_combined),
                "intended_union_count":
                    len(intended_union),
                "changed_features":
                    "; ".join(sorted(changed_combined)),
                "shared_joint_features":
                    "; ".join(sorted(shared_changed)),
                "unexpected_feature_count":
                    len(unexpected),
                "joint_formula_max_error":
                    joint_formula_error,
                "status":
                    "PASS",
            }
        )

        print(
            f"{scenario_id}: "
            f"mean ΔLST={combined_delta.mean():+.6f} °C | "
            f"mean non-additivity={nonadditivity.mean():+.6f} °C"
        )


combined_df = pd.concat(
    result_frames,
    ignore_index=True,
)

combined_feature_df = pd.concat(
    feature_frames,
    ignore_index=True,
)

feature_audit_df = pd.DataFrame(
    feature_audit_rows
)

if len(combined_df) != EXPECTED_COMBINED_ROWS:
    raise RuntimeError(
        f"Combined rows={len(combined_df)}, expected={EXPECTED_COMBINED_ROWS}"
    )

combined_df.to_csv(
    COMBINED_PATH,
    index=False,
)

combined_feature_df.to_csv(
    FEATURE_EXPORT_PATH,
    index=False,
)

feature_audit_df.to_csv(
    FEATURE_AUDIT_PATH,
    index=False,
)

combined_df[
    (
        combined_df["green_increase_pp_requested"]
        == PRIMARY_GREEN_PP
    )
    & (
        combined_df["built_up_level"]
        == PRIMARY_BUILT_LEVEL
    )
].to_csv(
    PRIMARY_PATH,
    index=False,
)

# Save 45 small scenario files: 5 years x 9 combinations.
for year in FUTURE_YEARS:
    for green_pp in GREEN_LEVELS:
        for built_level in BUILT_LEVELS:
            combined_df[
                (
                    combined_df[YEAR_COL] == year
                )
                & (
                    combined_df["green_increase_pp_requested"] == green_pp
                )
                & (
                    combined_df["built_up_level"] == built_level
                )
            ].to_csv(
                SCENARIO_DIR
                / (
                    f"29c_green{green_pp}pp_"
                    f"built{built_level.lower()}_{year}.csv"
                ),
                index=False,
            )

print(f"\nSaved: {COMBINED_PATH}")


# =============================================================================
# 5. SUMMARY + NON-ADDITIVITY
# =============================================================================

section("5. COMBINED SCENARIO SUMMARY")

summary = (
    combined_df
    .groupby(
        [
            YEAR_COL,
            "green_increase_pp_requested",
            "built_up_level",
        ],
        as_index=False,
    )
    .agg(
        grid_count=(GRID_COL, "size"),
        mean_actual_green_increase_pp=(
            "green_increase_pp_actual",
            "mean",
        ),
        mean_actual_ndbi_reduction=(
            "actual_ndbi_reduction",
            "mean",
        ),
        mean_baseline_lst_c=(
            "baseline_predicted_lst_c",
            "mean",
        ),
        mean_combined_scenario_lst_c=(
            "combined_scenario_predicted_lst_c",
            "mean",
        ),
        mean_combined_lst_change_c=(
            "combined_modeled_lst_change_c",
            "mean",
        ),
        median_combined_lst_change_c=(
            "combined_modeled_lst_change_c",
            "median",
        ),
        min_combined_lst_change_c=(
            "combined_modeled_lst_change_c",
            "min",
        ),
        max_combined_lst_change_c=(
            "combined_modeled_lst_change_c",
            "max",
        ),
        mean_green_only_delta_c=(
            "green_only_modeled_lst_change_c",
            "mean",
        ),
        mean_built_only_delta_c=(
            "built_only_modeled_lst_change_c",
            "mean",
        ),
        mean_additive_expected_delta_c=(
            "additive_expected_lst_change_c",
            "mean",
        ),
        mean_model_non_additivity_c=(
            "model_non_additivity_c",
            "mean",
        ),
    )
)

direction_counts = (
    combined_df
    .pivot_table(
        index=[
            YEAR_COL,
            "green_increase_pp_requested",
            "built_up_level",
        ],
        columns="combined_effect_direction",
        values=GRID_COL,
        aggfunc="count",
        fill_value=0,
    )
    .reset_index()
)

for col in ["cooling", "warming", "no_change"]:
    if col not in direction_counts.columns:
        direction_counts[col] = 0

direction_counts = direction_counts.rename(
    columns={
        "cooling": "cooling_grid_count",
        "warming": "warming_grid_count",
        "no_change": "no_change_grid_count",
    }
)

summary = summary.merge(
    direction_counts,
    on=[
        YEAR_COL,
        "green_increase_pp_requested",
        "built_up_level",
    ],
    how="left",
    validate="one_to_one",
)

summary["cooling_grid_pct"] = (
    summary["cooling_grid_count"]
    / summary["grid_count"]
    * 100.0
)

summary["warming_grid_pct"] = (
    summary["warming_grid_count"]
    / summary["grid_count"]
    * 100.0
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

overall = (
    combined_df
    .groupby(
        [
            "green_increase_pp_requested",
            "built_up_level",
        ],
        as_index=False,
    )
    .agg(
        rows=(GRID_COL, "size"),
        mean_actual_green_increase_pp=(
            "green_increase_pp_actual",
            "mean",
        ),
        mean_actual_ndbi_reduction=(
            "actual_ndbi_reduction",
            "mean",
        ),
        mean_combined_lst_change_c=(
            "combined_modeled_lst_change_c",
            "mean",
        ),
        mean_green_only_delta_c=(
            "green_only_modeled_lst_change_c",
            "mean",
        ),
        mean_built_only_delta_c=(
            "built_only_modeled_lst_change_c",
            "mean",
        ),
        mean_additive_expected_delta_c=(
            "additive_expected_lst_change_c",
            "mean",
        ),
        mean_model_non_additivity_c=(
            "model_non_additivity_c",
            "mean",
        ),
    )
)

overall.to_csv(
    OVERALL_SUMMARY_PATH,
    index=False,
)

nonadd_counts = (
    combined_df
    .pivot_table(
        index=[
            "green_increase_pp_requested",
            "built_up_level",
        ],
        columns="non_additivity_direction",
        values=GRID_COL,
        aggfunc="count",
        fill_value=0,
    )
    .reset_index()
)

for col in [
    "more_cooling_than_additive",
    "more_warming_than_additive",
    "approximately_additive",
]:
    if col not in nonadd_counts.columns:
        nonadd_counts[col] = 0

nonadd_counts["total"] = (
    nonadd_counts[
        [
            "more_cooling_than_additive",
            "more_warming_than_additive",
            "approximately_additive",
        ]
    ].sum(axis=1)
)

for col in [
    "more_cooling_than_additive",
    "more_warming_than_additive",
    "approximately_additive",
]:
    nonadd_counts[f"{col}_pct"] = (
        nonadd_counts[col]
        / nonadd_counts["total"]
        * 100.0
    )

nonadd_counts.to_csv(
    NONADDITIVITY_PATH,
    index=False,
)

print(summary.to_string(index=False))

print("\nOVERALL 2026-2030:")
print(overall.to_string(index=False))

print("\nMODEL NON-ADDITIVITY:")
print(nonadd_counts.to_string(index=False))


# =============================================================================
# 6. FRONTEND GEOJSON - ALL 9 COMBINATIONS
# =============================================================================

section("6. SAVE FRONTEND-READY YEARLY GEOJSON")

grid = gpd.read_file(GRID_WGS84_PATH)
grid[GRID_COL] = grid[GRID_COL].astype(str)

if grid[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Geometry unique grids={grid[GRID_COL].nunique()}, "
        f"expected={EXPECTED_GRIDS}"
    )

grid = grid[[GRID_COL, "geometry"]].copy()

geojson_files = []

for year in FUTURE_YEARS:
    base_year = (
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
                    "baseline_predicted_lst_c"
            }
        )
    )

    wide = base_year.copy()

    for green_pp in GREEN_LEVELS:
        for built_level in BUILT_LEVELS:
            tag = f"g{green_pp}_b{built_level.lower()}"

            sub = (
                combined_df[
                    (
                        combined_df[YEAR_COL] == year
                    )
                    & (
                        combined_df[
                            "green_increase_pp_requested"
                        ] == green_pp
                    )
                    & (
                        combined_df[
                            "built_up_level"
                        ] == built_level
                    )
                ][
                    [
                        GRID_COL,
                        "scenario_green_mask",
                        "scenario_ndbi",
                        "combined_scenario_predicted_lst_c",
                        "combined_modeled_lst_change_c",
                        "combined_effect_direction",
                        "model_non_additivity_c",
                    ]
                ]
                .copy()
                .rename(
                    columns={
                        "scenario_green_mask":
                            f"green_{tag}",
                        "scenario_ndbi":
                            f"ndbi_{tag}",
                        "combined_scenario_predicted_lst_c":
                            f"lst_{tag}",
                        "combined_modeled_lst_change_c":
                            f"delta_lst_{tag}",
                        "combined_effect_direction":
                            f"effect_{tag}",
                        "model_non_additivity_c":
                            f"nonadd_{tag}",
                    }
                )
            )

            wide = wide.merge(
                sub,
                on=GRID_COL,
                how="left",
                validate="one_to_one",
            )

    gdf = grid.merge(
        wide,
        on=GRID_COL,
        how="left",
        validate="one_to_one",
    )

    geo_path = (
        MAP_DIR
        / f"29c_combined_scenarios_{year}_wgs84.geojson"
    )

    gdf.to_file(
        geo_path,
        driver="GeoJSON",
    )

    geojson_files.append(geo_path)

    print(
        f"{year}: saved frontend GeoJSON containing all 9 combinations"
    )


# =============================================================================
# 7. REPORT MAPS - REPRESENTATIVE COMBINATIONS
# =============================================================================

section("7. SAVE 15 REPRESENTATIVE REPORT MAPS")

boundary = load_boundary()

representative = [
    (5, "LOW"),
    (10, "MODERATE"),
    (20, "HIGH"),
]

map_files = []

for year in FUTURE_YEARS:
    for green_pp, built_level in representative:
        sub = combined_df[
            (
                combined_df[YEAR_COL] == year
            )
            & (
                combined_df[
                    "green_increase_pp_requested"
                ] == green_pp
            )
            & (
                combined_df[
                    "built_up_level"
                ] == built_level
            )
        ][
            [
                GRID_COL,
                "combined_modeled_lst_change_c",
            ]
        ].copy()

        gdf = grid.merge(
            sub,
            on=GRID_COL,
            how="left",
            validate="one_to_one",
        )

        values = gdf[
            "combined_modeled_lst_change_c"
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
            column="combined_modeled_lst_change_c",
            ax=ax,
            legend=True,
            edgecolor="none",
            cmap="RdBu_r",
            vmin=-max_abs,
            vmax=max_abs,
            legend_kwds={
                "label":
                    "Modeled combined LST change (°C)\n"
                    "negative = cooling; positive = warming",
                "shrink": 0.72,
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
                f"Kaduwela Combined Scenario — {year}\n"
                f"Green +{green_pp} pp | "
                f"Built-up {built_level.title()} reduction"
            ),
            fontsize=14,
            pad=12,
        )

        ax.set_axis_off()

        plt.tight_layout()

        map_path = (
            FIGURE_DIR
            / (
                f"29c_combined_lst_change_{year}_"
                f"green{green_pp}pp_"
                f"built{built_level.lower()}.png"
            )
        )

        plt.savefig(
            map_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        map_files.append(map_path)

    print(f"{year}: saved 3 representative combined maps")


# =============================================================================
# 8. SUMMARY FIGURES
# =============================================================================

section("8. SAVE SUMMARY FIGURES")

plot_df = overall.copy()

plot_df["combination"] = (
    "G+"
    + plot_df[
        "green_increase_pp_requested"
    ].astype(str)
    + " / B-"
    + plot_df[
        "built_up_level"
    ].astype(str)
)

fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.bar(
    plot_df["combination"],
    plot_df["mean_combined_lst_change_c"],
)

ax.axhline(
    0.0,
    linewidth=1.0,
    color="black",
)

ax.set_xlabel("Combined scenario")
ax.set_ylabel("Mean modeled LST change (°C)")
ax.set_title("Step 29C Combined Green + Built-up Scenario")
ax.tick_params(axis="x", rotation=35)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    SUMMARY_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.bar(
    plot_df["combination"],
    plot_df["mean_model_non_additivity_c"],
)

ax.axhline(
    0.0,
    linewidth=1.0,
    color="black",
)

ax.set_xlabel("Combined scenario")
ax.set_ylabel("Mean model non-additivity (°C)")
ax.set_title(
    "Combined Response Minus Sum of Individual Model Responses"
)
ax.tick_params(axis="x", rotation=35)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    NONADD_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved: {SUMMARY_FIGURE_PATH}")
print(f"Saved: {NONADD_FIGURE_PATH}")


# =============================================================================
# 9. STRICT AUDIT
# =============================================================================

section("9. STRICT STEP-29C AUDIT")

cooling_identity_error = float(
    np.max(
        np.abs(
            combined_df[
                "combined_cooling_benefit_c"
            ].to_numpy(dtype=float)
            + combined_df[
                "combined_modeled_lst_change_c"
            ].to_numpy(dtype=float)
        )
    )
)

nonadd_identity_error = float(
    np.max(
        np.abs(
            combined_df[
                "model_non_additivity_c"
            ].to_numpy(dtype=float)
            - (
                combined_df[
                    "combined_modeled_lst_change_c"
                ].to_numpy(dtype=float)
                - combined_df[
                    "additive_expected_lst_change_c"
                ].to_numpy(dtype=float)
            )
        )
    )
)

per_grid_year = (
    combined_df
    .groupby(
        [GRID_COL, YEAR_COL]
    )
    .size()
)

audit_rows = [
    {
        "check": "step28_audit",
        "status": "PASS",
        "details": "Verified fully PASS.",
    },
    {
        "check": "final_29a_v2_audit",
        "status": "PASS",
        "details": "Verified fully PASS.",
    },
    {
        "check": "final_29b_audit",
        "status": "PASS",
        "details": "Verified fully PASS.",
    },
    {
        "check": "baseline_raw_prediction_reproduction",
        "status":
            "PASS"
            if baseline_reproduction_error <= PRED_TOL
            else "FAIL",
        "details":
            f"max_error_c={baseline_reproduction_error:.12e}",
    },
    {
        "check": "green_levels",
        "status":
            "PASS"
            if found_green_levels == GREEN_LEVELS
            else "FAIL",
        "details":
            str(found_green_levels),
    },
    {
        "check": "built_levels",
        "status":
            "PASS"
            if set(found_built_levels) == set(BUILT_LEVELS)
            else "FAIL",
        "details":
            str(found_built_levels),
    },
    {
        "check": "shared_feature_semantics",
        "status":
            "PASS"
            if not unexpected_shared
            else "FAIL",
        "details":
            (
                "shared="
                + (
                    "; ".join(sorted(shared_changed))
                    if shared_changed
                    else "NONE"
                )
                + "; shared features are recomputed jointly"
            ),
    },
    {
        "check": "combined_row_count",
        "status":
            "PASS"
            if len(combined_df) == EXPECTED_COMBINED_ROWS
            else "FAIL",
        "details":
            f"actual={len(combined_df)}, expected={EXPECTED_COMBINED_ROWS}",
    },
    {
        "check": "nine_scenarios_per_grid_year",
        "status":
            "PASS"
            if (per_grid_year == 9).all()
            else "FAIL",
        "details": "expected=9",
    },
    {
        "check": "combined_outputs_finite",
        "status":
            "PASS"
            if np.isfinite(
                combined_df[
                    [
                        "combined_scenario_predicted_lst_c",
                        "combined_modeled_lst_change_c",
                        "green_only_modeled_lst_change_c",
                        "built_only_modeled_lst_change_c",
                        "additive_expected_lst_change_c",
                        "model_non_additivity_c",
                    ]
                ].to_numpy(dtype=float)
            ).all()
            else "FAIL",
        "details":
            f"rows={len(combined_df)}",
    },
    {
        "check": "only_intended_features_changed",
        "status":
            "PASS"
            if (
                feature_audit_df[
                    "unexpected_feature_count"
                ] == 0
            ).all()
            else "FAIL",
        "details":
            "Combined changes are contained in union of finalized 29A + 29B changes.",
    },
    {
        "check": "joint_interaction_formula",
        "status":
            "PASS"
            if (
                feature_audit_df[
                    "joint_formula_max_error"
                ] <= FORMULA_TOL
            ).all()
            else "FAIL",
        "details":
            "ndbi_x_vegetation_deficit = scenario NDBI * (1 - scenario green_mask)",
    },
    {
        "check": "cooling_identity",
        "status":
            "PASS"
            if cooling_identity_error <= PRED_TOL
            else "FAIL",
        "details":
            f"max_error_c={cooling_identity_error:.12e}",
    },
    {
        "check": "non_additivity_identity",
        "status":
            "PASS"
            if nonadd_identity_error <= PRED_TOL
            else "FAIL",
        "details":
            f"max_error_c={nonadd_identity_error:.12e}",
    },
    {
        "check": "combined_prediction_not_additive_sum",
        "status": "PASS",
        "details":
            (
                "True joint feature vector is passed through frozen LightGBM; "
                "individual-sum is diagnostic only."
            ),
    },
    {
        "check": "scenario_recentering",
        "status": "PASS",
        "details":
            "Not applied. Local same-grid raw anomaly delta used.",
    },
    {
        "check": "model_retraining",
        "status": "PASS",
        "details":
            "None. Frozen Step-22B LightGBM used.",
    },
    {
        "check": "step28_baseline_modified",
        "status": "PASS",
        "details":
            "No. All 29C outputs saved separately.",
    },
    {
        "check": "causal_synergy_claim",
        "status": "PASS",
        "details":
            "No. Difference from additive reference is model non-additivity only.",
    },
    {
        "check": "uncertainty_added_here",
        "status": "PASS",
        "details":
            "No. Uncertainty remains Step 30.",
    },
    {
        "check": "yearly_geojson_count",
        "status":
            "PASS"
            if len(geojson_files) == 5
            else "FAIL",
        "details":
            f"saved={len(geojson_files)}, expected=5",
    },
    {
        "check": "representative_map_count",
        "status":
            "PASS"
            if len(map_files) == 15
            else "FAIL",
        "details":
            f"saved={len(map_files)}, expected=15",
    },
]

audit_df = pd.DataFrame(audit_rows)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
)

print(audit_df.to_string(index=False))

if (audit_df["status"] != "PASS").any():
    raise RuntimeError(
        "One or more Step-29C audit checks failed."
    )


# =============================================================================
# 10. METHODOLOGY + REPORT
# =============================================================================

section("10. SAVE STEP-29C METHODOLOGY + REPORT")

methodology = {
    "step": "29C",
    "name": "Combined green-cover and built-up-intensity what-if scenario",
    "new_ml_model": False,
    "scenario_engine": "Frozen Step-22B LightGBM",
    "green_source": "Final Step-29A V2 persistent green scenario",
    "green_choices_percentage_points": GREEN_LEVELS,
    "built_source": "Final Step-29B NDBI built-up-intensity scenario",
    "built_choices": BUILT_LEVELS,
    "combinations_per_grid_year": 9,
    "combined_prediction_method":
        (
            "Construct true joint feature vector and run frozen LightGBM once. "
            "Do not use arithmetic sum as combined prediction."
        ),
    "shared_joint_features":
        sorted(shared_changed),
    "joint_formula":
        (
            "ndbi_x_vegetation_deficit = "
            "scenario_NDBI * (1 - scenario_green_mask)"
        ),
    "additive_reference":
        "green_only_delta + built_only_delta; diagnostic only",
    "model_non_additivity":
        (
            "combined_delta - additive_reference; "
            "model interaction/non-additivity, not causal synergy"
        ),
    "frontend_outputs": [
        "baseline_green_mask",
        "scenario_green_mask",
        "baseline_ndbi",
        "scenario_ndbi",
        "baseline_predicted_lst_c",
        "combined_scenario_predicted_lst_c",
        "combined_modeled_lst_change_c",
        "combined_effect_direction",
    ],
    "primary_report_combination": {
        "green_pp": PRIMARY_GREEN_PP,
        "built_level": PRIMARY_BUILT_LEVEL,
    },
    "uncertainty": "Deferred to Step 30",
    "causal_claim": False,
}

METHOD_PATH.write_text(
    json.dumps(
        methodology,
        indent=2,
    ),
    encoding="utf-8",
)

report_lines = [
    "STEP 29C - COMBINED GREEN + BUILT-UP INTENSITY WHAT-IF SCENARIO",
    "=" * 120,
    "",
    "FRONTEND CHOICES",
    "-" * 120,
    "Green cover: +5 pp / +10 pp / +20 pp",
    "Built-up intensity reduction: LOW / MODERATE / HIGH",
    "",
    "METHOD",
    "-" * 120,
    (
        "Each combined scenario is generated from a true joint feature vector "
        "and one frozen Step-22B LightGBM inference."
    ),
    (
        "The separate green-only and built-only LST effects are NOT simply "
        "added to produce the final combined LST."
    ),
    "",
    "SHARED JOINT FEATURE(S)",
    "-" * 120,
    (
        "; ".join(sorted(shared_changed))
        if shared_changed
        else "NONE"
    ),
    "",
    "OVERALL 2026-2030 SUMMARY",
    "-" * 120,
    overall.to_string(index=False),
    "",
    "MODEL NON-ADDITIVITY",
    "-" * 120,
    nonadd_counts.to_string(index=False),
    "",
    "INTERPRETATION",
    "-" * 120,
    (
        "Negative model_non_additivity_c means the joint model response is "
        "cooler / less warm than the simple additive reference."
    ),
    (
        "Positive model_non_additivity_c means the joint model response is "
        "warmer / less cool than the simple additive reference."
    ),
    (
        "This is model interaction/non-additivity, not causal synergy."
    ),
    (
        "All scenario LST changes are model-based what-if sensitivities, "
        "not guaranteed physical outcomes."
    ),
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {METHOD_PATH}")
print(f"Saved: {REPORT_PATH}")


# =============================================================================
# 11. HASHES
# =============================================================================

section("11. REPRODUCIBILITY HASHES")

hash_files = [
    FEATURE_LIST_PATH,
    MODEL_PATH,
    FUTURE_FEATURE_PATH,
    FUTURE_PRED_PATH,
    STEP28_AUDIT_PATH,
    GRID_WGS84_PATH,
    GREEN_FEATURE_PATH,
    GREEN_AUDIT_PATH,
    BUILT_FEATURE_PATH,
    BUILT_AUDIT_PATH,
    COMBINED_PATH,
    PRIMARY_PATH,
    FEATURE_EXPORT_PATH,
    SUMMARY_PATH,
    OVERALL_SUMMARY_PATH,
    FEATURE_AUDIT_PATH,
    NONADDITIVITY_PATH,
    AUDIT_PATH,
    METHOD_PATH,
    REPORT_PATH,
    SUMMARY_FIGURE_PATH,
    NONADD_FIGURE_PATH,
    *geojson_files,
    *map_files,
]

for year in FUTURE_YEARS:
    for green_pp in GREEN_LEVELS:
        for built_level in BUILT_LEVELS:
            hash_files.append(
                SCENARIO_DIR
                / (
                    f"29c_green{green_pp}pp_"
                    f"built{built_level.lower()}_{year}.csv"
                )
            )

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
            "file": str(path),
            "sha256": digest,
        }
    )

    print(f"{path.name}: {digest}")

pd.DataFrame(hash_rows).to_csv(
    HASH_PATH,
    index=False,
)

print(f"Saved: {HASH_PATH}")


# =============================================================================
# FINAL
# =============================================================================

section("STEP 29C COMPLETED SUCCESSFULLY")

print(
    "Combined green + built-up scenario completed.\n"
    "Green choices: +5 / +10 / +20 percentage points.\n"
    "Built-up choices: LOW / MODERATE / HIGH NDBI reduction.\n"
    "All 9 combinations were evaluated for every 2026-2030 grid-year.\n"
    "Combined LST was generated from a true joint feature vector.\n"
    "The same frozen Step-22B LightGBM was used without retraining.\n"
    "The shared NDBI x vegetation-deficit interaction was recomputed using "
    "both scenario values.\n"
    "Green-only + built-only arithmetic sum is diagnostic only.\n"
    "Combined-minus-additive is reported as MODEL NON-ADDITIVITY, "
    "not causal synergy.\n"
    "Frontend-ready yearly GeoJSON files contain all 9 combinations.\n"
    "Step-28 baseline predictions were not modified.\n"
    "No causal or guaranteed cooling claim is made.\n"
    "Scenario uncertainty remains deferred to Step 30.\n"
    "\nNEXT: review Step-29C terminal output before freezing 29C "
    "and moving to Step 29D blue-space / water-access scenario."
)
