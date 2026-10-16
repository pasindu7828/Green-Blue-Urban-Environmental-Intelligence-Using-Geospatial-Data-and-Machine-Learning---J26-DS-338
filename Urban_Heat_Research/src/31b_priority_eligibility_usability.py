
"""
STEP 31B - PRIORITY INDICATOR ELIGIBILITY + USABILITY SCREENING
===============================================================

Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Screen the Step-31A candidate inventory BEFORE any redundancy analysis,
final indicator selection, CRITIC weighting or TOPSIS ranking.

This step checks:
1. Step-31A prerequisite integrity
2. Source availability
3. Leakage eligibility
4. Exact candidate materialization from completed outputs
5. Grid coverage
6. Missing / finite-value usability
7. Variation / constant-indicator problems
8. Planning relevance status
9. Direction-definition status
10. Static / structural context flags

IMPORTANT
---------
- NO final indicator selection occurs here.
- NO Spearman redundancy removal occurs here.
- NO temporal representation choice is finalized here.
- NO CRITIC weight is calculated here.
- NO TOPSIS score/rank is calculated here.
- NO model is retrained.
- NO Step 23/28/29/30 output is modified.

Expected project structure:
<project root>/
    data/
    models/
    outputs/
    src/
        31a_priority_candidate_inventory.py
        31b_priority_eligibility_usability.py
"""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


# =============================================================================
# 0. SETTINGS / PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

OUT_DIR = ROOT / "outputs"
METRICS_DIR = OUT_DIR / "metrics"
FORECAST_DIR = OUT_DIR / "forecasting"
SCENARIO_DIR = OUT_DIR / "scenarios"
PRIORITY_DIR = OUT_DIR / "priority"
REPORT_DIR = OUT_DIR / "reports"

PRIORITY_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_GRIDS = 2194
EXPECTED_FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]
EXPECTED_FUTURE_ROWS_PER_SCENARIO = EXPECTED_GRIDS * len(EXPECTED_FUTURE_YEARS)

STEP31A_INVENTORY = (
    PRIORITY_DIR
    / "31a_candidate_priority_indicator_inventory.csv"
)

STEP31A_AUDIT = (
    REPORT_DIR
    / "31a_priority_candidate_inventory_audit.csv"
)

# Upstream completed outputs.
HISTORICAL_BASE = (
    PROCESSED_DIR
    / "kaduwela_preprocessed_base.csv"
)

STEP23_TREND = (
    METRICS_DIR
    / "23_grid_mann_kendall_sens_slope.csv"
)

STEP23B_TRAJECTORY = (
    METRICS_DIR
    / "23b_grid_heat_trajectory_classes.csv"
)

STEP28_FUTURE_BASE = (
    FORECAST_DIR
    / "28_future_base_predictors_2026_2030.csv"
)

STEP28_FUTURE_LST = (
    FORECAST_DIR
    / "28_future_grid_lst_predictions_2026_2030.csv"
)

STEP29A_GREEN = (
    SCENARIO_DIR
    / "29a_v2_persistent_green_scenarios_2026_2030.csv"
)

STEP29B_BUILT = (
    SCENARIO_DIR
    / "29b_built_up_intensity_scenarios_2026_2030.csv"
)

STEP29C_COMBINED = (
    SCENARIO_DIR
    / "29c_combined_green_built_scenarios_2026_2030.csv"
)

STEP29D_BLUE = (
    SCENARIO_DIR
    / "29d_blue_space_ndwi_scenarios_2026_2030.csv"
)

# Outputs.
SCREENING_PATH = (
    PRIORITY_DIR
    / "31b_priority_candidate_screening.csv"
)

ELIGIBLE_PATH = (
    PRIORITY_DIR
    / "31b_candidates_passed_to_31c.csv"
)

EVIDENCE_PATH = (
    PRIORITY_DIR
    / "31b_candidate_usability_evidence.csv"
)

AUDIT_PATH = (
    REPORT_DIR
    / "31b_priority_eligibility_usability_audit.csv"
)

METHOD_PATH = (
    REPORT_DIR
    / "31b_priority_eligibility_usability_methodology.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "31b_priority_eligibility_usability_report.txt"
)

HASH_PATH = (
    REPORT_DIR
    / "31b_priority_eligibility_usability_hashes.csv"
)


# =============================================================================
# 1. HELPERS
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


def require_file(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def require_audit_all_pass(path, label):
    require_file(path, label)

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    status_col = next(
        (
            c for c in df.columns
            if str(c).strip().lower() == "status"
        ),
        None,
    )

    if status_col is None:
        raise RuntimeError(
            f"{label} has no status column."
        )

    status = (
        df[status_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if not status.eq("PASS").all():
        raise RuntimeError(
            f"{label} is not fully PASS."
        )


def normalize_grid_year(df, label):
    if "grid_id" not in df.columns:
        raise RuntimeError(
            f"{label}: grid_id column missing."
        )

    out = df.copy()
    out["grid_id"] = out["grid_id"].astype(str)

    if "year" in out.columns:
        out["year"] = pd.to_numeric(
            out["year"],
            errors="raise",
        ).astype(int)

    return out


def require_columns(df, columns, label):
    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label}: missing required columns: {missing}"
        )


def numeric_usability(values):
    s = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    )

    row_count = int(len(s))
    missing_count = int(s.isna().sum())

    finite_mask = (
        s.notna()
        & np.isfinite(
            s.fillna(0).to_numpy(dtype=float)
        )
    )

    finite_count = int(finite_mask.sum())

    valid = s[
        s.notna()
        & np.isfinite(
            s.fillna(0).to_numpy(dtype=float)
        )
    ]

    if len(valid) == 0:
        return {
            "row_count": row_count,
            "missing_count": missing_count,
            "missing_pct": (
                100.0 * missing_count / row_count
                if row_count
                else np.nan
            ),
            "finite_count": finite_count,
            "unique_count": 0,
            "min_value": np.nan,
            "max_value": np.nan,
            "mean_value": np.nan,
            "std_value": np.nan,
            "has_variation": False,
        }

    unique_count = int(valid.nunique(dropna=True))
    std_value = float(valid.std(ddof=0))

    return {
        "row_count": row_count,
        "missing_count": missing_count,
        "missing_pct": (
            100.0 * missing_count / row_count
            if row_count
            else np.nan
        ),
        "finite_count": finite_count,
        "unique_count": unique_count,
        "min_value": float(valid.min()),
        "max_value": float(valid.max()),
        "mean_value": float(valid.mean()),
        "std_value": std_value,
        "has_variation": bool(unique_count > 1),
    }


def categorical_usability(values):
    s = pd.Series(values)
    row_count = int(len(s))
    missing_count = int(s.isna().sum())
    valid = (
        s.dropna()
        .astype(str)
        .str.strip()
    )

    valid = valid[
        valid != ""
    ]

    return {
        "row_count": row_count,
        "missing_count": missing_count,
        "missing_pct": (
            100.0 * missing_count / row_count
            if row_count
            else np.nan
        ),
        "finite_count": np.nan,
        "unique_count": int(valid.nunique()),
        "min_value": np.nan,
        "max_value": np.nan,
        "mean_value": np.nan,
        "std_value": np.nan,
        "has_variation": bool(
            valid.nunique() > 1
        ),
    }


def check_one_value_per_grid(df, value_col, label):
    counts = (
        df.groupby("grid_id")[value_col]
        .nunique(dropna=False)
    )

    max_unique = int(counts.max())

    return (
        max_unique == 1,
        max_unique,
        label,
    )


def future_summary(
    df,
    value_col,
    mode,
):
    require_columns(
        df,
        ["grid_id", "year", value_col],
        f"future summary {value_col}",
    )

    d = df[
        df["year"].isin(
            EXPECTED_FUTURE_YEARS
        )
    ].copy()

    if mode == "2030":
        out = (
            d[d["year"] == 2030][
                ["grid_id", value_col]
            ]
            .copy()
        )

    elif mode == "MEAN":
        out = (
            d.groupby(
                "grid_id",
                as_index=False,
            )[value_col]
            .mean()
        )

    elif mode == "MAX":
        out = (
            d.groupby(
                "grid_id",
                as_index=False,
            )[value_col]
            .max()
        )

    else:
        raise ValueError(
            f"Unknown future summary mode: {mode}"
        )

    if out["grid_id"].nunique() != EXPECTED_GRIDS:
        raise RuntimeError(
            f"{value_col}/{mode}: "
            f"grid_count={out['grid_id'].nunique()}, "
            f"expected={EXPECTED_GRIDS}"
        )

    return out[
        ["grid_id", value_col]
    ].copy()


def historical_static_values(
    df,
    value_col,
):
    require_columns(
        df,
        ["grid_id", value_col],
        f"historical static {value_col}",
    )

    # First, check whether each grid has one fixed value
    # through the historical period.
    nunique = (
        df.groupby("grid_id")[value_col]
        .nunique(dropna=False)
    )

    static_within_grid = bool(
        (nunique <= 1).all()
    )

    out = (
        df.sort_values(
            ["grid_id", "year"]
            if "year" in df.columns
            else ["grid_id"]
        )
        .groupby(
            "grid_id",
            as_index=False,
        )
        .first()[
            ["grid_id", value_col]
        ]
    )

    return out, static_within_grid


def scenario_values(
    df,
    benefit_col,
    filter_mask,
    label,
):
    require_columns(
        df,
        [
            "grid_id",
            "year",
            benefit_col,
        ],
        label,
    )

    part = df.loc[
        filter_mask
    ].copy()

    expected_rows = (
        EXPECTED_FUTURE_ROWS_PER_SCENARIO
    )

    if len(part) != expected_rows:
        raise RuntimeError(
            f"{label}: rows={len(part)}, "
            f"expected={expected_rows}"
        )

    if (
        part["grid_id"].nunique()
        != EXPECTED_GRIDS
    ):
        raise RuntimeError(
            f"{label}: unique grids="
            f"{part['grid_id'].nunique()}, "
            f"expected={EXPECTED_GRIDS}"
        )

    years = sorted(
        part["year"].unique().tolist()
    )

    if years != EXPECTED_FUTURE_YEARS:
        raise RuntimeError(
            f"{label}: years={years}, "
            f"expected={EXPECTED_FUTURE_YEARS}"
        )

    # Step 31D will decide the exact scenario temporal representation.
    # For 31B usability, we test per-grid mean across 2026-2030.
    out = (
        part.groupby(
            "grid_id",
            as_index=False,
        )[benefit_col]
        .mean()
    )

    return out


# =============================================================================
# 2. LOAD / VERIFY STEP 31A + UPSTREAM DATA
# =============================================================================

section(
    "STEP 31B - PRIORITY INDICATOR ELIGIBILITY + USABILITY SCREENING"
)

require_audit_all_pass(
    STEP31A_AUDIT,
    "Step 31A audit",
)

require_file(
    STEP31A_INVENTORY,
    "Step 31A candidate inventory",
)

inventory = pd.read_csv(
    STEP31A_INVENTORY,
    low_memory=False,
)

required_inventory_cols = [
    "candidate_id",
    "decision_dimension",
    "indicator_name",
    "source_key",
    "candidate_representation",
    "raw_priority_direction",
    "31a_status",
]

require_columns(
    inventory,
    required_inventory_cols,
    "Step 31A inventory",
)

if not inventory[
    "candidate_id"
].is_unique:
    raise RuntimeError(
        "Step 31A candidate_id values are not unique."
    )

print(
    f"Step 31A inventory rows: {len(inventory)}"
)

# Load completed source files.
for path, label in [
    (
        HISTORICAL_BASE,
        "historical processed base",
    ),
    (
        STEP23_TREND,
        "Step 23 trend output",
    ),
    (
        STEP23B_TRAJECTORY,
        "Step 23B trajectory output",
    ),
    (
        STEP28_FUTURE_BASE,
        "Step 28 future base predictors",
    ),
    (
        STEP28_FUTURE_LST,
        "Step 28 future LST predictions",
    ),
    (
        STEP29A_GREEN,
        "Step 29A green scenario",
    ),
    (
        STEP29B_BUILT,
        "Step 29B built scenario",
    ),
    (
        STEP29C_COMBINED,
        "Step 29C combined scenario",
    ),
    (
        STEP29D_BLUE,
        "Step 29D blue scenario",
    ),
]:
    require_file(
        path,
        label,
    )

historical = normalize_grid_year(
    pd.read_csv(
        HISTORICAL_BASE,
        low_memory=False,
    ),
    "historical base",
)

trend = normalize_grid_year(
    pd.read_csv(
        STEP23_TREND,
        low_memory=False,
    ),
    "Step 23 trend",
)

trajectory = normalize_grid_year(
    pd.read_csv(
        STEP23B_TRAJECTORY,
        low_memory=False,
    ),
    "Step 23B trajectory",
)

future_base = normalize_grid_year(
    pd.read_csv(
        STEP28_FUTURE_BASE,
        low_memory=False,
    ),
    "Step 28 future base",
)

future_lst = normalize_grid_year(
    pd.read_csv(
        STEP28_FUTURE_LST,
        low_memory=False,
    ),
    "Step 28 future LST",
)

green = normalize_grid_year(
    pd.read_csv(
        STEP29A_GREEN,
        low_memory=False,
    ),
    "Step 29A",
)

built = normalize_grid_year(
    pd.read_csv(
        STEP29B_BUILT,
        low_memory=False,
    ),
    "Step 29B",
)

combined = normalize_grid_year(
    pd.read_csv(
        STEP29C_COMBINED,
        low_memory=False,
    ),
    "Step 29C",
)

blue = normalize_grid_year(
    pd.read_csv(
        STEP29D_BLUE,
        low_memory=False,
    ),
    "Step 29D",
)

# Core schema checks.
require_columns(
    future_lst,
    [
        "grid_id",
        "year",
        "predicted_lst_c",
    ],
    "Step 28 future LST",
)

require_columns(
    trajectory,
    [
        "grid_id",
        "trajectory_class",
        "relative_sen_slope_percentile_per_year",
    ],
    "Step 23B trajectory",
)

require_columns(
    trend,
    [
        "grid_id",
        "sen_slope_c_per_year",
    ],
    "Step 23 trend",
)

require_columns(
    green,
    [
        "requested_green_increase_pp",
        "cooling_benefit_c",
    ],
    "Step 29A",
)

require_columns(
    built,
    [
        "scenario_level",
        "cooling_benefit_c",
    ],
    "Step 29B",
)

require_columns(
    combined,
    [
        "green_increase_pp_requested",
        "built_up_level",
        "combined_cooling_benefit_c",
    ],
    "Step 29C",
)

require_columns(
    blue,
    [
        "scenario_level",
        "cooling_benefit_c",
    ],
    "Step 29D",
)

built["scenario_level"] = (
    built["scenario_level"]
    .astype(str)
    .str.strip()
    .str.upper()
)

blue["scenario_level"] = (
    blue["scenario_level"]
    .astype(str)
    .str.strip()
    .str.upper()
)

combined["built_up_level"] = (
    combined["built_up_level"]
    .astype(str)
    .str.strip()
    .str.upper()
)

green[
    "requested_green_increase_pp"
] = pd.to_numeric(
    green[
        "requested_green_increase_pp"
    ],
    errors="raise",
).astype(int)

combined[
    "green_increase_pp_requested"
] = pd.to_numeric(
    combined[
        "green_increase_pp_requested"
    ],
    errors="raise",
).astype(int)

print("PASS: all Step-31B source files loaded and required schemas verified.")


# =============================================================================
# 3. MATERIALIZE EACH DIRECT CANDIDATE
# =============================================================================

section(
    "3. MATERIALIZE + TEST DIRECT CANDIDATES"
)

evidence_rows = []
screen_rows = []

support_only_ids = {
    "SUPPORT_SHAP",
    "SUPPORT_UNCERTAINTY",
}

known_safe_source_keys = {
    "step28_future_lst",
    "step23_trend",
    "step23b_trajectory",
    "step28_future_base",
    "historical_base",
    "step29a_green",
    "step29b_built",
    "step29c_combined",
    "step29d_blue",
}

context_review_ids = {
    "BUILT_CONTEXT_building_mask",
    "BUILT_CONTEXT_road_mask",
    "BUILT_CONTEXT_dist_road",
    "BUILT_CONTEXT_dist_main_road",
    "BUILT_CONTEXT_LCZ",
}

categorical_ids = {
    "TREND_CLASS",
}


def record_candidate(
    row,
    values_df=None,
    value_col=None,
    is_categorical=False,
    static_within_grid=None,
    leakage_status="PASS",
    planning_relevance="PASS",
    direction_status="PASS",
    special_note="",
):
    cid = row["candidate_id"]

    if values_df is None:
        stats = {
            "row_count": np.nan,
            "missing_count": np.nan,
            "missing_pct": np.nan,
            "finite_count": np.nan,
            "unique_count": np.nan,
            "min_value": np.nan,
            "max_value": np.nan,
            "mean_value": np.nan,
            "std_value": np.nan,
            "has_variation": np.nan,
        }
        grid_count = np.nan
        coverage_status = "NOT_APPLICABLE"
        usability_status = "SUPPORT_ONLY"

    else:
        grid_count = int(
            values_df["grid_id"].nunique()
        )

        coverage_status = (
            "PASS"
            if grid_count == EXPECTED_GRIDS
            else "FAIL"
        )

        if is_categorical:
            stats = categorical_usability(
                values_df[value_col]
            )
        else:
            stats = numeric_usability(
                values_df[value_col]
            )

        usability_status = "PASS"

        if stats["missing_count"] != 0:
            usability_status = "FAIL"

        if not is_categorical:
            if (
                stats["finite_count"]
                != stats["row_count"]
            ):
                usability_status = "FAIL"

        if not bool(
            stats["has_variation"]
        ):
            usability_status = "FAIL"

        if coverage_status != "PASS":
            usability_status = "FAIL"

    # Step 31B does not perform final indicator selection.
    if cid in support_only_ids:
        decision = (
            "SUPPORT_ONLY_NOT_DIRECT_CRITERION"
        )

    elif (
        leakage_status == "PASS"
        and usability_status == "PASS"
        and direction_status in [
            "PASS",
            "DEFER_CATEGORICAL_ENCODING",
            "REVIEW_CONTEXT_DIRECTION",
        ]
        and planning_relevance in [
            "PASS",
            "CONTEXT_REVIEW",
        ]
    ):
        decision = "PASS_TO_31C"

    else:
        decision = "FAIL_STEP31B_REVIEW_REQUIRED"

    evidence_rows.append(
        {
            "candidate_id": cid,
            "grid_count": grid_count,
            "coverage_status": coverage_status,
            **stats,
            "static_within_grid": static_within_grid,
        }
    )

    screen_rows.append(
        {
            "candidate_id": cid,
            "decision_dimension": row["decision_dimension"],
            "indicator_name": row["indicator_name"],
            "source_key": row["source_key"],
            "candidate_representation": row["candidate_representation"],
            "raw_priority_direction": row["raw_priority_direction"],
            "leakage_status": leakage_status,
            "planning_relevance_status": planning_relevance,
            "direction_status": direction_status,
            "grid_coverage_status": coverage_status,
            "usability_status": usability_status,
            "static_within_grid": static_within_grid,
            "step31b_decision": decision,
            "step31b_notes": special_note,
        }
    )


for _, row in inventory.iterrows():

    cid = str(
        row["candidate_id"]
    )

    # ----------------------------------------------------------
    # SUPPORT-ONLY ITEMS
    # ----------------------------------------------------------
    if cid in support_only_ids:
        record_candidate(
            row=row,
            values_df=None,
            value_col=None,
            leakage_status="NOT_APPLICABLE_SUPPORT_ONLY",
            planning_relevance="SUPPORT_ONLY",
            direction_status="NOT_APPLICABLE",
            special_note=(
                "Retained as supporting evidence only; "
                "never enters CRITIC/TOPSIS directly."
            ),
        )
        continue

    source_key = str(
        row["source_key"]
    )

    if source_key not in known_safe_source_keys:
        raise RuntimeError(
            f"{cid}: unknown direct source_key={source_key}"
        )

    leakage_status = "PASS"
    planning_relevance = "PASS"
    direction_status = "PASS"
    static_flag = None
    special_note = ""

    # ----------------------------------------------------------
    # FUTURE HEAT
    # ----------------------------------------------------------
    if cid == "HEAT_2030":
        values = future_summary(
            future_lst,
            "predicted_lst_c",
            "2030",
        )
        value_col = "predicted_lst_c"

    elif cid == "HEAT_MEAN":
        values = future_summary(
            future_lst,
            "predicted_lst_c",
            "MEAN",
        )
        value_col = "predicted_lst_c"

    elif cid == "HEAT_MAX":
        values = future_summary(
            future_lst,
            "predicted_lst_c",
            "MAX",
        )
        value_col = "predicted_lst_c"

    # ----------------------------------------------------------
    # TREND
    # ----------------------------------------------------------
    elif cid == "TREND_ABS_SEN":
        values = trend[
            [
                "grid_id",
                "sen_slope_c_per_year",
            ]
        ].copy()
        value_col = "sen_slope_c_per_year"

    elif cid == "TREND_REL_SEN":
        values = trajectory[
            [
                "grid_id",
                "relative_sen_slope_percentile_per_year",
            ]
        ].copy()
        value_col = (
            "relative_sen_slope_percentile_per_year"
        )

    elif cid == "TREND_CLASS":
        values = trajectory[
            [
                "grid_id",
                "trajectory_class",
            ]
        ].copy()
        value_col = "trajectory_class"
        direction_status = (
            "DEFER_CATEGORICAL_ENCODING"
        )
        special_note = (
            "Five classes are usable, but Step 31B does "
            "not invent an ordinal numeric score."
        )

    # ----------------------------------------------------------
    # VEGETATION / GREEN DEFICIT
    # ----------------------------------------------------------
    elif cid.startswith("VEG_"):
        if cid.startswith("VEG_DEFICIT_"):
            base_col = "green_mask"
            mode = (
                "2030"
                if cid.endswith("_2030")
                else "MEAN"
            )

            tmp = future_summary(
                future_base,
                base_col,
                mode,
            )
            tmp["vegetation_deficit"] = (
                1.0
                - pd.to_numeric(
                    tmp[base_col],
                    errors="raise",
                )
            )

            values = tmp[
                [
                    "grid_id",
                    "vegetation_deficit",
                ]
            ].copy()
            value_col = "vegetation_deficit"

        else:
            # Example: VEG_NDVI_2030 / VEG_EVI_MEAN / VEG_green_mask_2030
            variable = None
            for candidate_var in [
                "NDVI",
                "EVI",
                "green_mask",
            ]:
                token = (
                    f"VEG_{candidate_var}_"
                )
                if cid.startswith(token):
                    variable = candidate_var
                    break

            if variable is None:
                raise RuntimeError(
                    f"Unknown vegetation candidate: {cid}"
                )

            mode = (
                "2030"
                if cid.endswith("_2030")
                else "MEAN"
            )

            values = future_summary(
                future_base,
                variable,
                mode,
            )
            value_col = variable

    # ----------------------------------------------------------
    # BUILT-UP / URBAN PRESSURE
    # ----------------------------------------------------------
    elif cid.startswith("BUILT_") and not cid.startswith("BUILT_CONTEXT_"):
        variable = None

        for candidate_var in [
            "NDBI",
            "NDBI_ADJ",
            "NightLights",
        ]:
            token = (
                f"BUILT_{candidate_var}_"
            )
            if cid.startswith(token):
                variable = candidate_var
                break

        if variable is None:
            raise RuntimeError(
                f"Unknown built candidate: {cid}"
            )

        mode = (
            "2030"
            if cid.endswith("_2030")
            else "MEAN"
        )

        values = future_summary(
            future_base,
            variable,
            mode,
        )
        value_col = variable

    elif cid.startswith("BUILT_CONTEXT_"):
        variable = cid.replace(
            "BUILT_CONTEXT_",
            "",
            1,
        )

        require_columns(
            historical,
            [variable],
            cid,
        )

        (
            values,
            static_flag,
        ) = historical_static_values(
            historical,
            variable,
        )

        value_col = variable
        planning_relevance = "CONTEXT_REVIEW"
        direction_status = (
            "REVIEW_CONTEXT_DIRECTION"
        )
        special_note = (
            "Structurally usable across grids, but Step 31C "
            "must decide whether this contextual indicator "
            "adds planning value beyond direct intervention dimensions."
        )

    # ----------------------------------------------------------
    # BLUE-SPACE
    # ----------------------------------------------------------
    elif cid == "BLUE_NDWI_2030":
        values = future_summary(
            future_base,
            "NDWI",
            "2030",
        )
        value_col = "NDWI"

    elif cid == "BLUE_NDWI_MEAN":
        values = future_summary(
            future_base,
            "NDWI",
            "MEAN",
        )
        value_col = "NDWI"

    elif cid == "BLUE_DIST_WATER":
        (
            values,
            static_flag,
        ) = historical_static_values(
            historical,
            "dist_water",
        )
        value_col = "dist_water"
        planning_relevance = "PASS"
        direction_status = "PASS"
        special_note = (
            "Static spatial context is acceptable because "
            "distance-to-water varies among grids."
        )

    # ----------------------------------------------------------
    # GREEN SCENARIOS
    # ----------------------------------------------------------
    elif cid.startswith("SCEN_GREEN_"):
        pp = int(
            cid.split("_")[-1]
            .replace("PP", "")
        )

        values = scenario_values(
            green,
            "cooling_benefit_c",
            (
                green[
                    "requested_green_increase_pp"
                ]
                == pp
            ),
            f"{cid} Step29A",
        )
        value_col = "cooling_benefit_c"
        special_note = (
            "Usability tested using per-grid mean cooling benefit "
            "across 2026-2030 only for screening. "
            "Step 31D still chooses the final temporal representation."
        )

    # ----------------------------------------------------------
    # BUILT SCENARIOS
    # ----------------------------------------------------------
    elif cid.startswith("SCEN_BUILT_"):
        level = cid.replace(
            "SCEN_BUILT_",
            "",
            1,
        )

        values = scenario_values(
            built,
            "cooling_benefit_c",
            (
                built[
                    "scenario_level"
                ]
                == level
            ),
            f"{cid} Step29B",
        )
        value_col = "cooling_benefit_c"
        special_note = (
            "Usability tested using per-grid mean 2026-2030 benefit. "
            "Step 31D decides final temporal representation."
        )

    # ----------------------------------------------------------
    # BLUE SCENARIOS
    # ----------------------------------------------------------
    elif cid.startswith("SCEN_BLUE_"):
        level = cid.replace(
            "SCEN_BLUE_",
            "",
            1,
        )

        values = scenario_values(
            blue,
            "cooling_benefit_c",
            (
                blue[
                    "scenario_level"
                ]
                == level
            ),
            f"{cid} Step29D",
        )
        value_col = "cooling_benefit_c"
        special_note = (
            "Usability tested using per-grid mean 2026-2030 benefit. "
            "Step 29D support/extrapolation remains separate evidence."
        )

    # ----------------------------------------------------------
    # COMBINED SCENARIOS
    # ----------------------------------------------------------
    elif cid.startswith("SCEN_COMBINED_G"):
        tail = cid.replace(
            "SCEN_COMBINED_G",
            "",
            1,
        )

        pp_text, level = (
            tail.split("_", 1)
        )

        pp = int(
            pp_text
        )

        values = scenario_values(
            combined,
            "combined_cooling_benefit_c",
            (
                combined[
                    "green_increase_pp_requested"
                ]
                == pp
            )
            & (
                combined[
                    "built_up_level"
                ]
                == level
            ),
            f"{cid} Step29C",
        )

        value_col = (
            "combined_cooling_benefit_c"
        )

        special_note = (
            "True joint Step-29C response used. "
            "No arithmetic sum of green-only + built-only effects."
        )

    else:
        raise RuntimeError(
            f"Step 31B has no materialization rule for candidate: {cid}"
        )

    # Final value table check.
    if (
        values["grid_id"].nunique()
        != EXPECTED_GRIDS
    ):
        raise RuntimeError(
            f"{cid}: final materialized unique grids="
            f"{values['grid_id'].nunique()}, "
            f"expected={EXPECTED_GRIDS}"
        )

    record_candidate(
        row=row,
        values_df=values,
        value_col=value_col,
        is_categorical=(
            cid in categorical_ids
        ),
        static_within_grid=static_flag,
        leakage_status=leakage_status,
        planning_relevance=planning_relevance,
        direction_status=direction_status,
        special_note=special_note,
    )


# =============================================================================
# 4. SAVE SCREENING TABLES
# =============================================================================

screening = pd.DataFrame(
    screen_rows
)

evidence = pd.DataFrame(
    evidence_rows
)

screening = screening.merge(
    evidence[
        [
            "candidate_id",
            "grid_count",
            "missing_count",
            "missing_pct",
            "finite_count",
            "unique_count",
            "min_value",
            "max_value",
            "mean_value",
            "std_value",
            "has_variation",
        ]
    ],
    on="candidate_id",
    how="left",
    validate="one_to_one",
)

screening.to_csv(
    SCREENING_PATH,
    index=False,
)

evidence.to_csv(
    EVIDENCE_PATH,
    index=False,
)

passed_to_31c = screening[
    screening[
        "step31b_decision"
    ]
    == "PASS_TO_31C"
].copy()

passed_to_31c.to_csv(
    ELIGIBLE_PATH,
    index=False,
)

print(
    f"Direct candidates passed to 31C: "
    f"{len(passed_to_31c)}"
)

support_count = int(
    (
        screening[
            "step31b_decision"
        ]
        == "SUPPORT_ONLY_NOT_DIRECT_CRITERION"
    ).sum()
)

print(
    f"Support-only evidence rows        : "
    f"{support_count}"
)


# =============================================================================
# 5. SCREENING SUMMARY
# =============================================================================

section(
    "5. STEP 31B SCREENING SUMMARY"
)

decision_summary = (
    screening.groupby(
        [
            "decision_dimension",
            "step31b_decision",
        ],
        as_index=False,
    )
    .size()
    .rename(
        columns={
            "size":
                "candidate_count",
        }
    )
)

print(
    decision_summary.to_string(
        index=False
    )
)

fail_rows = screening[
    screening[
        "step31b_decision"
    ]
    == "FAIL_STEP31B_REVIEW_REQUIRED"
]

if len(fail_rows):
    print(
        "\nCandidates requiring review:"
    )
    print(
        fail_rows[
            [
                "candidate_id",
                "decision_dimension",
                "leakage_status",
                "planning_relevance_status",
                "direction_status",
                "grid_coverage_status",
                "usability_status",
            ]
        ].to_string(
            index=False
        )
    )


# =============================================================================
# 6. STRICT STEP 31B AUDIT
# =============================================================================

audit_rows = []


def add_audit(
    check,
    passed,
    details,
):
    audit_rows.append(
        {
            "check": check,
            "status": (
                "PASS"
                if passed
                else "FAIL"
            ),
            "details": details,
        }
    )


direct_input_count = int(
    (
        inventory[
            "31a_status"
        ]
        == "CANDIDATE_NOT_YET_SELECTED"
    ).sum()
)

support_input_count = int(
    (
        inventory[
            "31a_status"
        ]
        == "SUPPORT_ONLY"
    ).sum()
)

add_audit(
    "step31a_audit",
    True,
    "Step 31A audit verified fully PASS.",
)

add_audit(
    "screening_row_count",
    len(screening)
    == len(inventory),
    (
        f"screening={len(screening)}, "
        f"inventory={len(inventory)}"
    ),
)

add_audit(
    "candidate_ids_preserved",
    set(
        screening[
            "candidate_id"
        ]
    )
    == set(
        inventory[
            "candidate_id"
        ]
    ),
    (
        f"screening_unique="
        f"{screening['candidate_id'].nunique()}, "
        f"inventory_unique="
        f"{inventory['candidate_id'].nunique()}"
    ),
)

add_audit(
    "all_direct_candidates_materialized",
    (
        screening[
            ~screening[
                "candidate_id"
            ].isin(
                support_only_ids
            )
        ][
            "grid_count"
        ]
        == EXPECTED_GRIDS
    ).all(),
    (
        f"expected grids per direct candidate="
        f"{EXPECTED_GRIDS}"
    ),
)

add_audit(
    "direct_candidate_missingness",
    (
        screening[
            ~screening[
                "candidate_id"
            ].isin(
                support_only_ids
            )
        ][
            "missing_count"
        ]
        == 0
    ).all(),
    "All direct candidates must have zero missing materialized values.",
)

add_audit(
    "direct_candidate_variation",
    (
        screening[
            ~screening[
                "candidate_id"
            ].isin(
                support_only_ids
            )
        ][
            "has_variation"
        ]
        == True
    ).all(),
    "No direct candidate may be constant across all 2,194 grids.",
)

add_audit(
    "leakage_screen",
    (
        screening[
            ~screening[
                "candidate_id"
            ].isin(
                support_only_ids
            )
        ][
            "leakage_status"
        ]
        == "PASS"
    ).all(),
    (
        "Future heat uses Step-28 forecasts; historical/trend sources "
        "use observed 2015-2025 history; scenario candidates use frozen "
        "Step-29 model-based outputs. No future observed LST target used."
    ),
)

add_audit(
    "support_only_count_preserved",
    support_count
    == support_input_count,
    (
        f"Step31A support={support_input_count}, "
        f"Step31B support={support_count}"
    ),
)

add_audit(
    "no_final_indicator_selection",
    True,
    (
        "31B only screens eligibility/usability. "
        "Final non-redundant selection remains Step31C."
    ),
)

add_audit(
    "no_redundancy_removal",
    True,
    "Spearman redundancy analysis not performed here.",
)

add_audit(
    "no_critic_weighting",
    True,
    "No CRITIC weights calculated.",
)

add_audit(
    "no_topsis_ranking",
    True,
    "No TOPSIS score/rank calculated.",
)

add_audit(
    "no_model_retraining",
    True,
    "No model fitting/retraining performed.",
)

add_audit(
    "upstream_outputs_modified",
    True,
    "No. Only new Step-31B files were written.",
)

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
)

section(
    "6. STRICT STEP 31B AUDIT"
)

print(
    audit_df.to_string(
        index=False
    )
)

if not audit_df[
    "status"
].eq("PASS").all():
    raise RuntimeError(
        "Step 31B audit FAILED. "
        "Do not proceed to Step 31C."
    )

if len(fail_rows):
    raise RuntimeError(
        "One or more candidates failed Step 31B screening. "
        "Review them before Step 31C."
    )


# =============================================================================
# 7. METHODOLOGY / REPORT
# =============================================================================

methodology = {
    "step": "31B",
    "title": (
        "Priority Indicator Eligibility + "
        "Usability Screening"
    ),
    "purpose": (
        "Verify that each Step-31A direct candidate can be safely "
        "materialized at grid level without leakage, missingness, "
        "coverage failure, or zero spatial variation."
    ),
    "expected_grids": EXPECTED_GRIDS,
    "future_years": EXPECTED_FUTURE_YEARS,
    "screening_rules": {
        "leakage": (
            "No future observed LST target. Step-28 forecast outputs "
            "and frozen Step-29 scenario outputs are eligible."
        ),
        "coverage": (
            "Every direct candidate must represent all 2,194 grids."
        ),
        "missingness": (
            "Materialized direct candidate values must have zero missing values."
        ),
        "variation": (
            "A direct candidate must vary across grids; a citywide constant "
            "cannot support grid ranking."
        ),
        "categorical_trajectory": (
            "Trajectory class is checked for completeness/variation but is not "
            "numerically ordered in Step 31B."
        ),
        "static_context": (
            "Static structural variables are not rejected merely because "
            "they are time-invariant. They are flagged for planning-relevance "
            "review in Step 31C."
        ),
        "scenario_temporal_screen": (
            "Scenario candidates are tested using per-grid 2026-2030 mean "
            "benefit only as a usability probe. Step 31D/31C selection logic "
            "still decides the final representation."
        ),
        "support_layers": (
            "SHAP and Step-30 uncertainty remain support-only and never enter "
            "CRITIC/TOPSIS directly."
        ),
    },
    "not_done_here": [
        "final indicator selection",
        "Spearman redundancy removal",
        "final temporal representation freeze",
        "CRITIC weighting",
        "TOPSIS ranking",
        "model retraining",
    ],
    "input_candidate_rows": int(
        len(inventory)
    ),
    "direct_candidates_from_31a": direct_input_count,
    "support_only_from_31a": support_input_count,
    "passed_to_31c": int(
        len(passed_to_31c)
    ),
}

with METHOD_PATH.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        methodology,
        f,
        indent=2,
        ensure_ascii=False,
    )

report_lines = [
    "STEP 31B - PRIORITY INDICATOR ELIGIBILITY + USABILITY SCREENING",
    "=" * 78,
    "",
    "STATUS: COMPLETED",
    "",
    f"Step31A total rows          : {len(inventory)}",
    f"Direct candidates input     : {direct_input_count}",
    f"Support-only rows input     : {support_input_count}",
    f"Direct candidates -> 31C    : {len(passed_to_31c)}",
    "",
    "31B checks:",
    "- leakage eligibility",
    "- exact source/schema usability",
    "- 2,194-grid coverage",
    "- missing / finite values",
    "- cross-grid variation",
    "- planning relevance status",
    "- direction-definition status",
    "- static/context flags",
    "",
    "Important:",
    "- No final indicator has been selected yet.",
    "- No redundancy analysis has been performed yet.",
    "- No CRITIC or TOPSIS calculation has been performed.",
    "- SHAP and uncertainty remain support-only.",
    "",
    "NEXT:",
    "Step 31C -> temporal representation + redundancy + stability/evidence + final indicator freeze.",
]

REPORT_PATH.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


# =============================================================================
# 8. REPRODUCIBILITY HASHES
# =============================================================================

hash_targets = [
    SCREENING_PATH,
    ELIGIBLE_PATH,
    EVIDENCE_PATH,
    AUDIT_PATH,
    METHOD_PATH,
    REPORT_PATH,
]

hash_df = pd.DataFrame(
    [
        {
            "file": path.name,
            "sha256": sha256_file(
                path
            ),
        }
        for path in hash_targets
    ]
)

hash_df.to_csv(
    HASH_PATH,
    index=False,
)


# =============================================================================
# 9. FINAL
# =============================================================================

section(
    "STEP 31B COMPLETED SUCCESSFULLY"
)

print(
    f"Direct candidates passed to Step 31C: "
    f"{len(passed_to_31c)} / {direct_input_count}"
)

print(
    f"Support-only evidence rows preserved: "
    f"{support_count}"
)

print("\nSaved:")
for path in [
    SCREENING_PATH,
    ELIGIBLE_PATH,
    EVIDENCE_PATH,
    AUDIT_PATH,
    METHOD_PATH,
    REPORT_PATH,
    HASH_PATH,
]:
    print(
        f"- {path}"
    )

print(
    "\nNo final indicator selection occurred."
)
print(
    "No Spearman redundancy removal occurred."
)
print(
    "No CRITIC/TOPSIS calculation occurred."
)
print(
    "No model retraining occurred."
)
print(
    "NEXT: review Step 31B output before Step 31C."
)
