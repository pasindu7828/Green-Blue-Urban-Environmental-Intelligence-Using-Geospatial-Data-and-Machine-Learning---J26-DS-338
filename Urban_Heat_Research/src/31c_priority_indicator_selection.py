
"""
STEP 31C - PRIORITY INDICATOR SELECTION + REDUNDANCY / STABILITY REVIEW
======================================================================

Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Take the technically usable Step-31B candidates and freeze ONE defensible
indicator for each of the six planning decision dimensions.

This step performs:
1. Step-31B integrity checks
2. Re-materialization of numeric candidate summaries at grid level
3. Within-dimension Spearman redundancy diagnostics
4. Temporal rank-stability diagnostics for selected future indicators
5. Future SHAP supporting-evidence review (support only)
6. Verification of the frozen Step-29C PRIMARY combined scenario
7. Final six-indicator freeze for Step 31D
8. Cross-dimension redundancy diagnostics

IMPORTANT
---------
Selection is DECISION-DIMENSION-DRIVEN, not "pick whatever predicts LST best".

The final six indicators are intentionally one-per-dimension:

1. Future Heat Severity
   -> HEAT_MEAN
      2026-2030 mean predicted LST

2. Trend Urgency
   -> TREND_REL_SEN
      2015-2025 relative heat-percentile Sen slope

3. Vegetation / Green Deficit
   -> VEG_DEFICIT_MEAN
      2026-2030 mean (1 - green_mask)

4. Built-up / Urban Pressure
   -> BUILT_NDBI_MEAN
      2026-2030 mean projected NDBI

5. Blue-space / Water Deficit
   -> BLUE_DIST_WATER
      static distance-to-water context

6. Modeled Intervention Benefit
   -> SCEN_COMBINED_G10_MODERATE
      2026-2030 mean modeled cooling benefit from the already-frozen
      Step-29C PRIMARY combined scenario (+10pp green, MODERATE NDBI reduction)

Why this policy?
----------------
- Mean future conditions represent sustained 2026-2030 planning exposure and
  avoid over-weighting one endpoint year or one maximum year.
- Relative trend is used for within-Kaduwela spatial urgency; Step-23B
  trajectory class remains descriptive and is NOT assigned an arbitrary
  ordinal numeric score.
- green_mask deficit is directly interpretable as green-cover deficit and is
  aligned with the green intervention dimension.
- NDBI is the direct built-up intensity index used by Step 29B.
- dist_water is the direct blue-access/proximity deficit; NDWI is retained as
  supporting wetness information and is not treated as literal water coverage.
- Step 29C itself froze +10pp green + MODERATE built reduction as its PRIMARY
  combined scenario, so Step 31C does not invent a new intervention combination.

This step DOES NOT:
- calculate CRITIC weights
- calculate TOPSIS scores/ranks
- normalize the final decision matrix
- retrain any model
- change Step 23/28/29/30 outputs
- convert uncertainty into a priority criterion
- convert SHAP into a priority criterion

NEXT
----
Step 31D -> build the 2,194-grid final decision matrix, apply benefit/cost
direction logic, and normalize for CRITIC.
"""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


# =============================================================================
# 0. PATHS / SETTINGS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

OUT_DIR = ROOT / "outputs"
METRICS_DIR = OUT_DIR / "metrics"
FORECAST_DIR = OUT_DIR / "forecasting"
SCENARIO_DIR = OUT_DIR / "scenarios"
EXPLAIN_DIR = OUT_DIR / "explainability"
PRIORITY_DIR = OUT_DIR / "priority"
REPORT_DIR = OUT_DIR / "reports"

PRIORITY_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_GRIDS = 2194
FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]
EXPECTED_FUTURE_ROWS = EXPECTED_GRIDS * len(FUTURE_YEARS)

REDUNDANCY_WARNING_ABS_RHO = 0.90
NEAR_DUPLICATE_ABS_RHO = 0.999

# -------------------------------------------------------------------------
# Step 31B
# -------------------------------------------------------------------------

STEP31B_SCREENING = (
    PRIORITY_DIR
    / "31b_priority_candidate_screening.csv"
)

STEP31B_PASSED = (
    PRIORITY_DIR
    / "31b_candidates_passed_to_31c.csv"
)

STEP31B_AUDIT = (
    REPORT_DIR
    / "31b_priority_eligibility_usability_audit.csv"
)

# -------------------------------------------------------------------------
# Upstream sources
# -------------------------------------------------------------------------

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

STEP29C_PRIMARY = (
    SCENARIO_DIR
    / "29c_combined_primary_green10_builtmoderate_2026_2030.csv"
)

STEP29D_BLUE = (
    SCENARIO_DIR
    / "29d_blue_space_ndwi_scenarios_2026_2030.csv"
)

STEP28B_SHAP = (
    EXPLAIN_DIR
    / "28b_future_global_shap_importance_2026_2030.csv"
)

STEP28B_AUDIT = (
    REPORT_DIR
    / "28b_future_shap_audit.csv"
)

# -------------------------------------------------------------------------
# Outputs
# -------------------------------------------------------------------------

NUMERIC_VALUES_PATH = (
    PRIORITY_DIR
    / "31c_numeric_candidate_values.csv"
)

WITHIN_REDUNDANCY_PATH = (
    PRIORITY_DIR
    / "31c_within_dimension_spearman_redundancy.csv"
)

TEMPORAL_STABILITY_PATH = (
    PRIORITY_DIR
    / "31c_selected_temporal_rank_stability.csv"
)

SHAP_SUPPORT_PATH = (
    PRIORITY_DIR
    / "31c_future_shap_supporting_evidence.csv"
)

FINAL_SELECTION_PATH = (
    PRIORITY_DIR
    / "31c_final_priority_indicator_selection.csv"
)

SELECTED_RAW_DIAGNOSTIC_PATH = (
    PRIORITY_DIR
    / "31c_selected_indicator_raw_diagnostic.csv"
)

FINAL_REDUNDANCY_PATH = (
    PRIORITY_DIR
    / "31c_final_indicator_cross_dimension_spearman.csv"
)

AUDIT_PATH = (
    REPORT_DIR
    / "31c_priority_indicator_selection_audit.csv"
)

METHOD_PATH = (
    REPORT_DIR
    / "31c_priority_indicator_selection_methodology.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "31c_priority_indicator_selection_report.txt"
)

HASH_PATH = (
    REPORT_DIR
    / "31c_priority_indicator_selection_hashes.csv"
)


# =============================================================================
# 1. FINAL SELECTION POLICY - FROZEN IN THIS STEP
# =============================================================================

FINAL_POLICY = {
    "Future Heat Severity": {
        "candidate_id": "HEAT_MEAN",
        "final_indicator_name": "Future Heat Severity",
        "final_field_name": "future_heat_mean_lst_c",
        "priority_direction": "HIGHER=HIGHER_PRIORITY",
        "selection_rule": (
            "Use the sustained 2026-2030 mean predicted LST rather than "
            "a single endpoint or maximum year."
        ),
    },
    "Trend Urgency": {
        "candidate_id": "TREND_REL_SEN",
        "final_indicator_name": "Relative Trend Urgency",
        "final_field_name": "relative_heat_sen_slope",
        "priority_direction": "MORE_POSITIVE=HIGHER_PRIORITY",
        "selection_rule": (
            "Use continuous relative heat-percentile Sen slope for spatial "
            "urgency within Kaduwela. Do not assign an arbitrary ordinal "
            "score to the descriptive trajectory classes."
        ),
    },
    "Vegetation / Green Deficit": {
        "candidate_id": "VEG_DEFICIT_MEAN",
        "final_indicator_name": "Future Green-cover Deficit",
        "final_field_name": "future_green_deficit_mean",
        "priority_direction": "HIGHER=HIGHER_PRIORITY",
        "selection_rule": (
            "Use mean 2026-2030 (1 - green_mask): directly interpretable "
            "as persistent green-cover deficit and aligned with Step 29A."
        ),
    },
    "Built-up / Urban Pressure": {
        "candidate_id": "BUILT_NDBI_MEAN",
        "final_indicator_name": "Future Built-up Intensity",
        "final_field_name": "future_ndbi_mean",
        "priority_direction": "HIGHER=HIGHER_PRIORITY",
        "selection_rule": (
            "Use mean 2026-2030 NDBI: direct built-up intensity measure and "
            "the same intervention dimension manipulated in Step 29B."
        ),
    },
    "Blue-space / Water Deficit": {
        "candidate_id": "BLUE_DIST_WATER",
        "final_indicator_name": "Water-access Deficit",
        "final_field_name": "distance_to_water",
        "priority_direction": "HIGHER=HIGHER_PRIORITY",
        "selection_rule": (
            "Use distance-to-water as direct spatial accessibility/proximity "
            "deficit. NDWI remains wetness evidence and is not interpreted as "
            "literal water-cover percentage."
        ),
    },
    "Modeled Intervention Benefit": {
        "candidate_id": "SCEN_COMBINED_G10_MODERATE",
        "final_indicator_name": "Modeled Combined Cooling Benefit",
        "final_field_name": "combined_g10_moderate_mean_cooling_benefit_c",
        "priority_direction": "HIGHER_BENEFIT=HIGHER_PRIORITY",
        "selection_rule": (
            "Use mean 2026-2030 cooling benefit from Step 29C's already-frozen "
            "PRIMARY scenario: +10pp green with MODERATE built-up reduction."
        ),
    },
}

FINAL_IDS = [
    spec["candidate_id"]
    for spec in FINAL_POLICY.values()
]


# =============================================================================
# 2. HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 136)
    print(title)
    print("=" * 136)


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
            f"{label}: no status column."
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


def require_columns(df, columns, label):
    missing = [
        c
        for c in columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label}: missing required columns: {missing}"
        )


def normalize_grid_year(df, label):
    require_columns(
        df,
        ["grid_id"],
        label,
    )

    out = df.copy()
    out["grid_id"] = (
        out["grid_id"]
        .astype(str)
    )

    if "year" in out.columns:
        out["year"] = pd.to_numeric(
            out["year"],
            errors="raise",
        ).astype(int)

    return out


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def future_summary(
    df,
    value_col,
    mode,
):
    require_columns(
        df,
        [
            "grid_id",
            "year",
            value_col,
        ],
        f"future summary {value_col}",
    )

    d = df[
        df["year"].isin(
            FUTURE_YEARS
        )
    ].copy()

    if mode == "2030":
        out = (
            d[
                d["year"] == 2030
            ][
                [
                    "grid_id",
                    value_col,
                ]
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
            f"Unknown future-summary mode: {mode}"
        )

    if (
        out["grid_id"].nunique()
        != EXPECTED_GRIDS
    ):
        raise RuntimeError(
            f"{value_col}/{mode}: "
            f"unique grids="
            f"{out['grid_id'].nunique()}, "
            f"expected={EXPECTED_GRIDS}"
        )

    return out


def historical_static_values(
    df,
    value_col,
):
    require_columns(
        df,
        [
            "grid_id",
            value_col,
        ],
        f"historical static {value_col}",
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
            [
                "grid_id",
                value_col,
            ]
        ]
    )

    if (
        out["grid_id"].nunique()
        != EXPECTED_GRIDS
    ):
        raise RuntimeError(
            f"{value_col}: static grid count "
            f"{out['grid_id'].nunique()} != "
            f"{EXPECTED_GRIDS}"
        )

    return out


def scenario_mean(
    df,
    benefit_col,
    mask,
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
        mask
    ].copy()

    if len(part) != EXPECTED_FUTURE_ROWS:
        raise RuntimeError(
            f"{label}: rows={len(part)}, "
            f"expected={EXPECTED_FUTURE_ROWS}"
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

    if years != FUTURE_YEARS:
        raise RuntimeError(
            f"{label}: years={years}, "
            f"expected={FUTURE_YEARS}"
        )

    out = (
        part.groupby(
            "grid_id",
            as_index=False,
        )[benefit_col]
        .mean()
    )

    return out


def one_series(
    df,
    value_col,
    candidate_id,
):
    require_columns(
        df,
        [
            "grid_id",
            value_col,
        ],
        candidate_id,
    )

    if (
        df["grid_id"].nunique()
        != EXPECTED_GRIDS
    ):
        raise RuntimeError(
            f"{candidate_id}: grid count "
            f"{df['grid_id'].nunique()} != "
            f"{EXPECTED_GRIDS}"
        )

    if df["grid_id"].duplicated().any():
        raise RuntimeError(
            f"{candidate_id}: duplicate grid rows."
        )

    values = pd.to_numeric(
        df[value_col],
        errors="coerce",
    )

    if values.isna().any():
        raise RuntimeError(
            f"{candidate_id}: missing/non-numeric values."
        )

    arr = values.to_numpy(
        dtype=float
    )

    if not np.isfinite(arr).all():
        raise RuntimeError(
            f"{candidate_id}: non-finite values."
        )

    if pd.Series(arr).nunique() <= 1:
        raise RuntimeError(
            f"{candidate_id}: no cross-grid variation."
        )

    out = (
        df[
            [
                "grid_id",
                value_col,
            ]
        ]
        .copy()
        .rename(
            columns={
                value_col:
                    candidate_id,
            }
        )
    )

    return out


def merge_series(
    base,
    add,
):
    if base is None:
        return add.copy()

    return base.merge(
        add,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )


def pairwise_spearman_long(
    matrix,
    metadata,
):
    numeric_cols = [
        c
        for c in matrix.columns
        if c != "grid_id"
    ]

    corr = (
        matrix[numeric_cols]
        .corr(
            method="spearman"
        )
    )

    rows = []

    for i, a in enumerate(
        numeric_cols
    ):
        for b in numeric_cols[
            i + 1:
        ]:
            dim_a = metadata.loc[
                a,
                "decision_dimension",
            ]

            dim_b = metadata.loc[
                b,
                "decision_dimension",
            ]

            rows.append(
                {
                    "candidate_a": a,
                    "candidate_b": b,
                    "dimension_a": dim_a,
                    "dimension_b": dim_b,
                    "same_dimension": (
                        dim_a == dim_b
                    ),
                    "spearman_rho": float(
                        corr.loc[
                            a,
                            b,
                        ]
                    ),
                    "abs_spearman_rho": float(
                        abs(
                            corr.loc[
                                a,
                                b,
                            ]
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def temporal_stability_rows(
    selected_id,
    selected_values,
    yearly_df,
    yearly_value_col,
    transform=None,
):
    rows = []

    selected = (
        selected_values[
            [
                "grid_id",
                selected_id,
            ]
        ]
        .copy()
    )

    for year in FUTURE_YEARS:
        y = yearly_df[
            yearly_df["year"] == year
        ][
            [
                "grid_id",
                yearly_value_col,
            ]
        ].copy()

        if transform is not None:
            y["year_value"] = transform(
                pd.to_numeric(
                    y[yearly_value_col],
                    errors="raise",
                )
            )
        else:
            y["year_value"] = pd.to_numeric(
                y[yearly_value_col],
                errors="raise",
            )

        joined = selected.merge(
            y[
                [
                    "grid_id",
                    "year_value",
                ]
            ],
            on="grid_id",
            how="inner",
            validate="one_to_one",
        )

        rho = joined[
            [
                selected_id,
                "year_value",
            ]
        ].corr(
            method="spearman"
        ).iloc[
            0,
            1,
        ]

        rows.append(
            {
                "candidate_id": selected_id,
                "year": int(year),
                "spearman_selected_summary_vs_year": float(
                    rho
                ),
            }
        )

    return rows


# =============================================================================
# 3. LOAD / VERIFY INPUTS
# =============================================================================

section(
    "STEP 31C - PRIORITY INDICATOR SELECTION + REDUNDANCY / STABILITY REVIEW"
)

require_audit_all_pass(
    STEP31B_AUDIT,
    "Step 31B audit",
)

require_file(
    STEP31B_SCREENING,
    "Step 31B screening",
)

require_file(
    STEP31B_PASSED,
    "Step 31B passed-candidate table",
)

screening = pd.read_csv(
    STEP31B_SCREENING,
    low_memory=False,
)

passed = pd.read_csv(
    STEP31B_PASSED,
    low_memory=False,
)

require_columns(
    passed,
    [
        "candidate_id",
        "decision_dimension",
        "step31b_decision",
    ],
    "Step 31B passed candidates",
)

if len(passed) != 46:
    raise RuntimeError(
        f"Expected 46 direct Step-31B candidates, "
        f"found {len(passed)}."
    )

if not passed[
    "step31b_decision"
].eq(
    "PASS_TO_31C"
).all():
    raise RuntimeError(
        "Not every Step-31B direct candidate is PASS_TO_31C."
    )

missing_final_ids = [
    cid
    for cid in FINAL_IDS
    if cid not in set(
        passed["candidate_id"]
    )
]

if missing_final_ids:
    raise RuntimeError(
        "Final-policy candidates missing from Step31B: "
        f"{missing_final_ids}"
    )

print(
    "PASS: Step 31B is fully PASS and all six "
    "policy candidates are available."
)

# Required source files.
for path, label in [
    (
        HISTORICAL_BASE,
        "Historical processed base",
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
        STEP29C_PRIMARY,
        "Step 29C PRIMARY combined scenario",
    ),
    (
        STEP29D_BLUE,
        "Step 29D blue scenario",
    ),
    (
        STEP28B_SHAP,
        "Step 28B future global SHAP importance",
    ),
]:
    require_file(
        path,
        label,
    )

require_audit_all_pass(
    STEP28B_AUDIT,
    "Step 28B SHAP audit",
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
    "Step 29A green",
)

built = normalize_grid_year(
    pd.read_csv(
        STEP29B_BUILT,
        low_memory=False,
    ),
    "Step 29B built",
)

combined = normalize_grid_year(
    pd.read_csv(
        STEP29C_COMBINED,
        low_memory=False,
    ),
    "Step 29C combined",
)

primary_combined = normalize_grid_year(
    pd.read_csv(
        STEP29C_PRIMARY,
        low_memory=False,
    ),
    "Step 29C primary combined",
)

blue = normalize_grid_year(
    pd.read_csv(
        STEP29D_BLUE,
        low_memory=False,
    ),
    "Step 29D blue",
)

shap_global = pd.read_csv(
    STEP28B_SHAP,
    low_memory=False,
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
    future_base,
    [
        "grid_id",
        "year",
        "NDVI",
        "EVI",
        "green_mask",
        "NDBI",
        "NDBI_ADJ",
        "NightLights",
        "NDWI",
    ],
    "Step 28 future base",
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
    trajectory,
    [
        "grid_id",
        "relative_sen_slope_percentile_per_year",
        "trajectory_class",
    ],
    "Step 23B trajectory",
)

require_columns(
    historical,
    [
        "grid_id",
        "year",
        "building_mask",
        "road_mask",
        "dist_road",
        "dist_main_road",
        "dist_water",
        "LCZ",
    ],
    "historical base",
)

require_columns(
    green,
    [
        "requested_green_increase_pp",
        "cooling_benefit_c",
    ],
    "Step 29A green",
)

require_columns(
    built,
    [
        "scenario_level",
        "cooling_benefit_c",
    ],
    "Step 29B built",
)

require_columns(
    combined,
    [
        "green_increase_pp_requested",
        "built_up_level",
        "combined_cooling_benefit_c",
    ],
    "Step 29C combined",
)

require_columns(
    primary_combined,
    [
        "grid_id",
        "year",
        "green_increase_pp_requested",
        "built_up_level",
        "combined_cooling_benefit_c",
    ],
    "Step 29C primary",
)

require_columns(
    blue,
    [
        "scenario_level",
        "cooling_benefit_c",
    ],
    "Step 29D blue",
)

require_columns(
    shap_global,
    [
        "rank",
        "feature",
        "mean_abs_centered_shap_c",
    ],
    "Step 28B future SHAP",
)

# Normalize scenario labels.
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

primary_combined["built_up_level"] = (
    primary_combined["built_up_level"]
    .astype(str)
    .str.strip()
    .str.upper()
)

green["requested_green_increase_pp"] = pd.to_numeric(
    green["requested_green_increase_pp"],
    errors="raise",
).astype(int)

combined["green_increase_pp_requested"] = pd.to_numeric(
    combined["green_increase_pp_requested"],
    errors="raise",
).astype(int)

primary_combined["green_increase_pp_requested"] = pd.to_numeric(
    primary_combined["green_increase_pp_requested"],
    errors="raise",
).astype(int)

print(
    "PASS: all required Step 31C source files and schemas verified."
)


# =============================================================================
# 4. VERIFY STEP 29C PRIMARY SCENARIO
# =============================================================================

section(
    "4. VERIFY FROZEN STEP-29C PRIMARY COMBINED SCENARIO"
)

primary_subset = combined[
    (
        combined[
            "green_increase_pp_requested"
        ]
        == 10
    )
    &
    (
        combined[
            "built_up_level"
        ]
        == "MODERATE"
    )
].copy()

if len(primary_subset) != EXPECTED_FUTURE_ROWS:
    raise RuntimeError(
        f"Step29C G10/MODERATE rows={len(primary_subset)}, "
        f"expected={EXPECTED_FUTURE_ROWS}"
    )

if len(primary_combined) != EXPECTED_FUTURE_ROWS:
    raise RuntimeError(
        f"Step29C PRIMARY file rows={len(primary_combined)}, "
        f"expected={EXPECTED_FUTURE_ROWS}"
    )

if not (
    primary_combined[
        "green_increase_pp_requested"
    ].eq(10).all()
    and
    primary_combined[
        "built_up_level"
    ].eq("MODERATE").all()
):
    raise RuntimeError(
        "Step29C PRIMARY file is not +10pp / MODERATE throughout."
    )

compare_cols = [
    "grid_id",
    "year",
    "combined_cooling_benefit_c",
]

a = (
    primary_subset[
        compare_cols
    ]
    .sort_values(
        [
            "grid_id",
            "year",
        ]
    )
    .reset_index(drop=True)
)

b = (
    primary_combined[
        compare_cols
    ]
    .sort_values(
        [
            "grid_id",
            "year",
        ]
    )
    .reset_index(drop=True)
)

keys_match = (
    a[
        [
            "grid_id",
            "year",
        ]
    ]
    .equals(
        b[
            [
                "grid_id",
                "year",
            ]
        ]
    )
)

benefits_match = np.allclose(
    pd.to_numeric(
        a[
            "combined_cooling_benefit_c"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    ),
    pd.to_numeric(
        b[
            "combined_cooling_benefit_c"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    ),
    rtol=0.0,
    atol=1e-12,
)

if not (
    keys_match
    and
    benefits_match
):
    raise RuntimeError(
        "Step29C PRIMARY file does not exactly match "
        "the G10/MODERATE subset of the full combined output."
    )

print(
    "PASS: Step29C PRIMARY = +10pp green + MODERATE built reduction."
)
print(
    "PASS: PRIMARY file exactly matches the corresponding "
    "full Step29C scenario subset."
)


# =============================================================================
# 5. MATERIALIZE ALL NUMERIC STEP-31B CANDIDATES
# =============================================================================

section(
    "5. MATERIALIZE NUMERIC CANDIDATE SUMMARIES"
)

candidate_matrix = None

metadata = (
    passed.set_index(
        "candidate_id"
    )[
        [
            "decision_dimension",
            "indicator_name",
            "candidate_representation",
            "raw_priority_direction",
        ]
    ]
    .copy()
)

# Future heat candidates.
heat_specs = {
    "HEAT_2030": "2030",
    "HEAT_MEAN": "MEAN",
    "HEAT_MAX": "MAX",
}

for cid, mode in heat_specs.items():
    tmp = future_summary(
        future_lst,
        "predicted_lst_c",
        mode,
    )

    s = one_series(
        tmp,
        "predicted_lst_c",
        cid,
    )

    candidate_matrix = merge_series(
        candidate_matrix,
        s,
    )

# Trend candidates - continuous only.
tmp = trend[
    [
        "grid_id",
        "sen_slope_c_per_year",
    ]
].copy()

candidate_matrix = merge_series(
    candidate_matrix,
    one_series(
        tmp,
        "sen_slope_c_per_year",
        "TREND_ABS_SEN",
    ),
)

tmp = trajectory[
    [
        "grid_id",
        "relative_sen_slope_percentile_per_year",
    ]
].copy()

candidate_matrix = merge_series(
    candidate_matrix,
    one_series(
        tmp,
        "relative_sen_slope_percentile_per_year",
        "TREND_REL_SEN",
    ),
)

# TREND_CLASS deliberately excluded from numeric correlation.
# No arbitrary ordinal mapping is invented.

# Vegetation candidates.
for variable in [
    "NDVI",
    "EVI",
    "green_mask",
]:
    for mode, suffix in [
        (
            "2030",
            "2030",
        ),
        (
            "MEAN",
            "MEAN",
        ),
    ]:
        cid = (
            f"VEG_{variable}_{suffix}"
        )

        tmp = future_summary(
            future_base,
            variable,
            mode,
        )

        candidate_matrix = merge_series(
            candidate_matrix,
            one_series(
                tmp,
                variable,
                cid,
            ),
        )

for mode, suffix in [
    (
        "2030",
        "2030",
    ),
    (
        "MEAN",
        "MEAN",
    ),
]:
    tmp = future_summary(
        future_base,
        "green_mask",
        mode,
    )

    tmp[
        "vegetation_deficit"
    ] = (
        1.0
        - pd.to_numeric(
            tmp["green_mask"],
            errors="raise",
        )
    )

    cid = (
        f"VEG_DEFICIT_{suffix}"
    )

    candidate_matrix = merge_series(
        candidate_matrix,
        one_series(
            tmp[
                [
                    "grid_id",
                    "vegetation_deficit",
                ]
            ],
            "vegetation_deficit",
            cid,
        ),
    )

# Built-up future candidates.
for variable in [
    "NDBI",
    "NDBI_ADJ",
    "NightLights",
]:
    for mode, suffix in [
        (
            "2030",
            "2030",
        ),
        (
            "MEAN",
            "MEAN",
        ),
    ]:
        cid = (
            f"BUILT_{variable}_{suffix}"
        )

        tmp = future_summary(
            future_base,
            variable,
            mode,
        )

        candidate_matrix = merge_series(
            candidate_matrix,
            one_series(
                tmp,
                variable,
                cid,
            ),
        )

# Built-up contextual candidates.
for variable in [
    "building_mask",
    "road_mask",
    "dist_road",
    "dist_main_road",
    "LCZ",
]:
    cid = (
        f"BUILT_CONTEXT_{variable}"
    )

    tmp = historical_static_values(
        historical,
        variable,
    )

    candidate_matrix = merge_series(
        candidate_matrix,
        one_series(
            tmp,
            variable,
            cid,
        ),
    )

# Blue candidates.
for mode, suffix in [
    (
        "2030",
        "2030",
    ),
    (
        "MEAN",
        "MEAN",
    ),
]:
    cid = (
        f"BLUE_NDWI_{suffix}"
    )

    tmp = future_summary(
        future_base,
        "NDWI",
        mode,
    )

    candidate_matrix = merge_series(
        candidate_matrix,
        one_series(
            tmp,
            "NDWI",
            cid,
        ),
    )

tmp = historical_static_values(
    historical,
    "dist_water",
)

candidate_matrix = merge_series(
    candidate_matrix,
    one_series(
        tmp,
        "dist_water",
        "BLUE_DIST_WATER",
    ),
)

# Green scenarios.
for pp in [
    5,
    10,
    20,
]:
    cid = (
        f"SCEN_GREEN_{pp}PP"
    )

    tmp = scenario_mean(
        green,
        "cooling_benefit_c",
        (
            green[
                "requested_green_increase_pp"
            ]
            == pp
        ),
        cid,
    )

    candidate_matrix = merge_series(
        candidate_matrix,
        one_series(
            tmp,
            "cooling_benefit_c",
            cid,
        ),
    )

# Built scenarios.
for level in [
    "LOW",
    "MODERATE",
    "HIGH",
]:
    cid = (
        f"SCEN_BUILT_{level}"
    )

    tmp = scenario_mean(
        built,
        "cooling_benefit_c",
        (
            built[
                "scenario_level"
            ]
            == level
        ),
        cid,
    )

    candidate_matrix = merge_series(
        candidate_matrix,
        one_series(
            tmp,
            "cooling_benefit_c",
            cid,
        ),
    )

# Blue scenarios.
for level in [
    "LOW",
    "MODERATE",
    "HIGH",
]:
    cid = (
        f"SCEN_BLUE_{level}"
    )

    tmp = scenario_mean(
        blue,
        "cooling_benefit_c",
        (
            blue[
                "scenario_level"
            ]
            == level
        ),
        cid,
    )

    candidate_matrix = merge_series(
        candidate_matrix,
        one_series(
            tmp,
            "cooling_benefit_c",
            cid,
        ),
    )

# Combined scenarios.
for pp in [
    5,
    10,
    20,
]:
    for level in [
        "LOW",
        "MODERATE",
        "HIGH",
    ]:
        cid = (
            f"SCEN_COMBINED_G{pp}_{level}"
        )

        tmp = scenario_mean(
            combined,
            "combined_cooling_benefit_c",
            (
                combined[
                    "green_increase_pp_requested"
                ]
                == pp
            )
            &
            (
                combined[
                    "built_up_level"
                ]
                == level
            ),
            cid,
        )

        candidate_matrix = merge_series(
            candidate_matrix,
            one_series(
                tmp,
                "combined_cooling_benefit_c",
                cid,
            ),
        )

if (
    candidate_matrix[
        "grid_id"
    ].nunique()
    != EXPECTED_GRIDS
):
    raise RuntimeError(
        "Final numeric candidate matrix does not contain 2,194 grids."
    )

numeric_candidate_ids = [
    c
    for c in candidate_matrix.columns
    if c != "grid_id"
]

expected_numeric_count = 45
# 46 direct candidates minus categorical TREND_CLASS = 45 numeric candidates.
if len(
    numeric_candidate_ids
) != expected_numeric_count:
    raise RuntimeError(
        f"Numeric candidate count={len(numeric_candidate_ids)}, "
        f"expected={expected_numeric_count}."
    )

candidate_matrix.to_csv(
    NUMERIC_VALUES_PATH,
    index=False,
)

print(
    f"Numeric candidate matrix: "
    f"{len(candidate_matrix):,} grids x "
    f"{len(numeric_candidate_ids)} numeric candidates"
)
print(
    "TREND_CLASS intentionally excluded from numeric correlation "
    "because no arbitrary class score is assigned."
)


# =============================================================================
# 6. WITHIN-DIMENSION SPEARMAN REDUNDANCY
# =============================================================================

section(
    "6. WITHIN-DIMENSION SPEARMAN REDUNDANCY"
)

pairwise = pairwise_spearman_long(
    candidate_matrix,
    metadata,
)

within = (
    pairwise[
        pairwise[
            "same_dimension"
        ]
    ]
    .copy()
    .sort_values(
        "abs_spearman_rho",
        ascending=False,
    )
)

within.to_csv(
    WITHIN_REDUNDANCY_PATH,
    index=False,
)

print(
    "Top within-dimension absolute Spearman correlations:"
)

print(
    within[
        [
            "candidate_a",
            "candidate_b",
            "dimension_a",
            "spearman_rho",
            "abs_spearman_rho",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)

print(
    "\nInterpretation: high within-dimension correlation supports using "
    "ONE representative instead of double-counting closely related alternatives."
)


# =============================================================================
# 7. TEMPORAL RANK STABILITY OF SELECTED FUTURE INDICATORS
# =============================================================================

section(
    "7. TEMPORAL RANK STABILITY OF SELECTED FUTURE INDICATORS"
)

stability_rows = []

selected_lookup = (
    candidate_matrix.set_index(
        "grid_id"
    )
)

# HEAT_MEAN
selected_heat = (
    candidate_matrix[
        [
            "grid_id",
            "HEAT_MEAN",
        ]
    ]
)

stability_rows += temporal_stability_rows(
    "HEAT_MEAN",
    selected_heat,
    future_lst,
    "predicted_lst_c",
)

# VEG_DEFICIT_MEAN
selected_green_def = (
    candidate_matrix[
        [
            "grid_id",
            "VEG_DEFICIT_MEAN",
        ]
    ]
)

stability_rows += temporal_stability_rows(
    "VEG_DEFICIT_MEAN",
    selected_green_def,
    future_base,
    "green_mask",
    transform=lambda x: 1.0 - x,
)

# BUILT_NDBI_MEAN
selected_ndbi = (
    candidate_matrix[
        [
            "grid_id",
            "BUILT_NDBI_MEAN",
        ]
    ]
)

stability_rows += temporal_stability_rows(
    "BUILT_NDBI_MEAN",
    selected_ndbi,
    future_base,
    "NDBI",
)

# SCEN_COMBINED_G10_MODERATE
selected_combined = (
    candidate_matrix[
        [
            "grid_id",
            "SCEN_COMBINED_G10_MODERATE",
        ]
    ]
)

stability_rows += temporal_stability_rows(
    "SCEN_COMBINED_G10_MODERATE",
    selected_combined,
    primary_combined,
    "combined_cooling_benefit_c",
)

stability = pd.DataFrame(
    stability_rows
)

stability_summary = (
    stability.groupby(
        "candidate_id",
        as_index=False,
    )
    .agg(
        mean_yearly_spearman=(
            "spearman_selected_summary_vs_year",
            "mean",
        ),
        min_yearly_spearman=(
            "spearman_selected_summary_vs_year",
            "min",
        ),
        max_yearly_spearman=(
            "spearman_selected_summary_vs_year",
            "max",
        ),
    )
)

stability = stability.merge(
    stability_summary,
    on="candidate_id",
    how="left",
    validate="many_to_one",
)

stability.to_csv(
    TEMPORAL_STABILITY_PATH,
    index=False,
)

print(
    stability_summary.to_string(
        index=False
    )
)

print(
    "\nThese are diagnostics only. Genuine temporal variation is retained; "
    "no candidate is manipulated to increase stability."
)


# =============================================================================
# 8. FUTURE SHAP - SUPPORTING EVIDENCE ONLY
# =============================================================================

section(
    "8. FUTURE SHAP SUPPORTING EVIDENCE - NOT A SELECTION SCORE"
)

shap = shap_global.copy()
shap["feature"] = (
    shap["feature"]
    .astype(str)
)

shap["rank"] = pd.to_numeric(
    shap["rank"],
    errors="raise",
).astype(int)

shap[
    "mean_abs_centered_shap_c"
] = pd.to_numeric(
    shap[
        "mean_abs_centered_shap_c"
    ],
    errors="raise",
)

support_families = {
    "Vegetation / Green Deficit": [
        "NDVI",
        "EVI",
        "GREEN_MASK",
        "VEGETATION",
    ],
    "Built-up / Urban Pressure": [
        "NDBI",
        "NIGHTLIGHT",
        "BUILDING",
    ],
    "Blue-space / Water Deficit": [
        "NDWI",
        "DIST_WATER",
        "WATER",
    ],
}

support_rows = []

for dimension, tokens in support_families.items():
    feature_upper = (
        shap["feature"]
        .str.upper()
    )

    mask = pd.Series(
        False,
        index=shap.index,
    )

    for token in tokens:
        mask = (
            mask
            |
            feature_upper.str.contains(
                token,
                regex=False,
                na=False,
            )
        )

    matched = (
        shap[
            mask
        ]
        .sort_values(
            "rank"
        )
        .copy()
    )

    if len(matched):
        best = matched.iloc[0]

        support_rows.append(
            {
                "decision_dimension": dimension,
                "matched_feature_count": int(
                    len(matched)
                ),
                "best_future_shap_rank": int(
                    best["rank"]
                ),
                "best_matching_feature": str(
                    best["feature"]
                ),
                "best_mean_abs_centered_shap_c": float(
                    best[
                        "mean_abs_centered_shap_c"
                    ]
                ),
                "role": (
                    "SUPPORT_ONLY_NOT_SELECTION_SCORE"
                ),
            }
        )

    else:
        support_rows.append(
            {
                "decision_dimension": dimension,
                "matched_feature_count": 0,
                "best_future_shap_rank": np.nan,
                "best_matching_feature": "",
                "best_mean_abs_centered_shap_c": np.nan,
                "role": (
                    "SUPPORT_ONLY_NOT_SELECTION_SCORE"
                ),
            }
        )

shap_support = pd.DataFrame(
    support_rows
)

shap_support.to_csv(
    SHAP_SUPPORT_PATH,
    index=False,
)

print(
    shap_support.to_string(
        index=False
    )
)

print(
    "\nSHAP is association/explanation evidence only. "
    "It is not converted into a CRITIC/TOPSIS criterion."
)


# =============================================================================
# 9. FREEZE FINAL SIX INDICATORS
# =============================================================================

section(
    "9. FINAL SIX-INDICATOR FREEZE"
)

selection_rows = []

for order, (
    dimension,
    spec,
) in enumerate(
    FINAL_POLICY.items(),
    start=1,
):
    cid = spec[
        "candidate_id"
    ]

    source_row = passed[
        passed[
            "candidate_id"
        ]
        == cid
    ].iloc[0]

    alternatives = (
        passed[
            passed[
                "decision_dimension"
            ]
            == dimension
        ][
            "candidate_id"
        ]
        .tolist()
    )

    alternatives_not_selected = [
        x
        for x in alternatives
        if x != cid
    ]

    selection_rows.append(
        {
            "criterion_order": order,
            "decision_dimension": dimension,
            "selected_candidate_id": cid,
            "final_indicator_name": spec[
                "final_indicator_name"
            ],
            "final_field_name": spec[
                "final_field_name"
            ],
            "source_key": source_row[
                "source_key"
            ],
            "candidate_representation": source_row[
                "candidate_representation"
            ],
            "priority_direction": spec[
                "priority_direction"
            ],
            "selection_rule": spec[
                "selection_rule"
            ],
            "alternatives_not_selected": ";".join(
                alternatives_not_selected
            ),
            "selection_status": "FINAL_SELECTED_STEP31C",
        }
    )

final_selection = pd.DataFrame(
    selection_rows
)

final_selection.to_csv(
    FINAL_SELECTION_PATH,
    index=False,
)

print(
    final_selection[
        [
            "criterion_order",
            "decision_dimension",
            "selected_candidate_id",
            "final_indicator_name",
            "priority_direction",
        ]
    ].to_string(
        index=False
    )
)

# Raw diagnostic table only.
selected_raw = (
    candidate_matrix[
        [
            "grid_id",
            *FINAL_IDS,
        ]
    ]
    .copy()
)

rename_map = {
    spec["candidate_id"]:
        spec["final_field_name"]
    for spec in FINAL_POLICY.values()
}

selected_raw = selected_raw.rename(
    columns=rename_map
)

selected_raw.to_csv(
    SELECTED_RAW_DIAGNOSTIC_PATH,
    index=False,
)

print(
    "\nSaved a raw selected-indicator diagnostic table only."
)
print(
    "Normalization and the final CRITIC-ready decision matrix remain Step 31D."
)


# =============================================================================
# 10. CROSS-DIMENSION REDUNDANCY OF FINAL SIX
# =============================================================================

section(
    "10. FINAL-SIX CROSS-DIMENSION SPEARMAN DIAGNOSTIC"
)

selected_for_corr = (
    candidate_matrix[
        [
            "grid_id",
            *FINAL_IDS,
        ]
    ]
    .copy()
)

final_meta = metadata.loc[
    FINAL_IDS
]

final_pairwise = pairwise_spearman_long(
    selected_for_corr,
    final_meta,
)

final_pairwise = (
    final_pairwise[
        final_pairwise[
            "same_dimension"
        ]
        == False
    ]
    .copy()
    .sort_values(
        "abs_spearman_rho",
        ascending=False,
    )
)

final_pairwise[
    "redundancy_flag"
] = np.where(
    final_pairwise[
        "abs_spearman_rho"
    ]
    >= NEAR_DUPLICATE_ABS_RHO,
    "NEAR_DUPLICATE_REVIEW_REQUIRED",
    np.where(
        final_pairwise[
            "abs_spearman_rho"
        ]
        >= REDUNDANCY_WARNING_ABS_RHO,
        "HIGH_CORRELATION_RETAIN_DISTINCT_DIMENSIONS",
        "OK",
    ),
)

final_pairwise.to_csv(
    FINAL_REDUNDANCY_PATH,
    index=False,
)

print(
    final_pairwise[
        [
            "candidate_a",
            "candidate_b",
            "spearman_rho",
            "abs_spearman_rho",
            "redundancy_flag",
        ]
    ].to_string(
        index=False
    )
)

max_final_abs_rho = float(
    final_pairwise[
        "abs_spearman_rho"
    ].max()
)

near_duplicate_count = int(
    (
        final_pairwise[
            "abs_spearman_rho"
        ]
        >= NEAR_DUPLICATE_ABS_RHO
    ).sum()
)

high_corr_count = int(
    (
        final_pairwise[
            "abs_spearman_rho"
        ]
        >= REDUNDANCY_WARNING_ABS_RHO
    ).sum()
)

print(
    f"\nMaximum final cross-dimension |rho|: "
    f"{max_final_abs_rho:.6f}"
)
print(
    f"Pairs >= {REDUNDANCY_WARNING_ABS_RHO:.2f}: "
    f"{high_corr_count}"
)
print(
    f"Near-duplicate pairs >= {NEAR_DUPLICATE_ABS_RHO:.3f}: "
    f"{near_duplicate_count}"
)

print(
    "\nHigh correlation across DIFFERENT planning dimensions is reported "
    "rather than silently removed. Step 33 sensitivity will test ranking robustness."
)


# =============================================================================
# 11. STRICT STEP 31C AUDIT
# =============================================================================

section(
    "11. STRICT STEP 31C AUDIT"
)

audit_rows = []


def add_audit(
    check,
    passed_check,
    details,
):
    audit_rows.append(
        {
            "check": check,
            "status": (
                "PASS"
                if passed_check
                else "FAIL"
            ),
            "details": details,
        }
    )


add_audit(
    "step31b_audit",
    True,
    "Step 31B audit verified fully PASS.",
)

add_audit(
    "step31b_direct_candidates",
    (
        len(passed)
        == 46
    ),
    (
        f"actual={len(passed)}, expected=46"
    ),
)

add_audit(
    "numeric_candidates_materialized",
    (
        len(
            numeric_candidate_ids
        )
        == 45
    ),
    (
        "45 numeric candidates expected; "
        "TREND_CLASS remains categorical/descriptive."
    ),
)

add_audit(
    "numeric_grid_count",
    (
        candidate_matrix[
            "grid_id"
        ].nunique()
        == EXPECTED_GRIDS
    ),
    (
        f"actual="
        f"{candidate_matrix['grid_id'].nunique()}, "
        f"expected={EXPECTED_GRIDS}"
    ),
)

add_audit(
    "step29c_primary_verified",
    (
        keys_match
        and
        benefits_match
    ),
    (
        "PRIMARY file is exactly the +10pp green / "
        "MODERATE built subset."
    ),
)

add_audit(
    "final_dimension_count",
    (
        final_selection[
            "decision_dimension"
        ].nunique()
        == 6
    ),
    (
        f"actual="
        f"{final_selection['decision_dimension'].nunique()}, "
        f"expected=6"
    ),
)

add_audit(
    "one_indicator_per_dimension",
    (
        final_selection.groupby(
            "decision_dimension"
        ).size().eq(1).all()
    ),
    "Exactly one final criterion per planning decision dimension.",
)

add_audit(
    "final_candidate_ids_unique",
    (
        final_selection[
            "selected_candidate_id"
        ].is_unique
    ),
    (
        f"selected={len(final_selection)}, "
        f"unique="
        f"{final_selection['selected_candidate_id'].nunique()}"
    ),
)

add_audit(
    "trajectory_class_not_arbitrarily_encoded",
    (
        "TREND_CLASS"
        not in FINAL_IDS
    ),
    (
        "TREND_CLASS remains descriptive; "
        "continuous TREND_REL_SEN selected."
    ),
)

add_audit(
    "future_heat_not_single_year_only",
    (
        "HEAT_MEAN"
        in FINAL_IDS
    ),
    "Sustained 2026-2030 mean heat selected.",
)

add_audit(
    "vegetation_deficit_direction_explicit",
    (
        "VEG_DEFICIT_MEAN"
        in FINAL_IDS
    ),
    (
        "1-green_mask selected, so larger value "
        "directly means larger green deficit."
    ),
)

add_audit(
    "built_intervention_alignment",
    (
        "BUILT_NDBI_MEAN"
        in FINAL_IDS
    ),
    "NDBI selected, aligned with Step 29B built-up intervention.",
)

add_audit(
    "blue_accessibility_not_watercover_claim",
    (
        "BLUE_DIST_WATER"
        in FINAL_IDS
    ),
    (
        "dist_water selected as accessibility/proximity deficit; "
        "NDWI not relabeled as water-cover percentage."
    ),
)

add_audit(
    "combined_primary_benefit_selected",
    (
        "SCEN_COMBINED_G10_MODERATE"
        in FINAL_IDS
    ),
    (
        "Uses Step29C frozen PRIMARY combined scenario."
    ),
)

add_audit(
    "negative_scenario_benefits_preserved",
    True,
    (
        "No clipping/sign-forcing applied. Higher modeled cooling benefit "
        "will be treated as higher priority in Step31D; negative values remain negative."
    ),
)

add_audit(
    "shap_support_only",
    True,
    (
        "Step28B SHAP used only as supporting evidence, "
        "not as a direct criterion or automatic selector."
    ),
)

add_audit(
    "uncertainty_not_direct_criterion",
    True,
    (
        "Step30 uncertainty remains separate confidence information."
    ),
)

add_audit(
    "no_near_duplicate_final_pair",
    (
        near_duplicate_count
        == 0
    ),
    (
        f"near-duplicate threshold="
        f"{NEAR_DUPLICATE_ABS_RHO}; "
        f"count={near_duplicate_count}"
    ),
)

add_audit(
    "no_critic_weighting",
    True,
    "No CRITIC weights calculated in Step31C.",
)

add_audit(
    "no_topsis_ranking",
    True,
    "No TOPSIS scores/ranks calculated in Step31C.",
)

add_audit(
    "no_model_retraining",
    True,
    "No model fitting/retraining performed.",
)

add_audit(
    "upstream_outputs_modified",
    True,
    "No. Only new Step31C outputs were written.",
)

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

if not audit_df[
    "status"
].eq("PASS").all():
    raise RuntimeError(
        "Step 31C strict audit FAILED. "
        "Do not proceed to Step 31D."
    )


# =============================================================================
# 12. METHODOLOGY / REPORT
# =============================================================================

methodology = {
    "step": "31C",
    "title": (
        "Priority Indicator Selection + "
        "Redundancy / Stability Review"
    ),
    "selection_philosophy": (
        "Decision-dimension-driven. The purpose is to answer which grids "
        "planners should prioritize, not to reuse all predictive ML features."
    ),
    "expected_grids": EXPECTED_GRIDS,
    "future_period": "2026-2030",
    "historical_trend_period": "2015-2025",
    "final_indicator_count": 6,
    "final_policy": FINAL_POLICY,
    "redundancy": {
        "method": "Spearman rank correlation",
        "within_dimension_use": (
            "Diagnostic support for choosing one representative and "
            "avoiding multiple same-concept criteria."
        ),
        "cross_dimension_policy": (
            "Report high correlation but do not automatically delete a "
            "conceptually distinct planning dimension."
        ),
        "warning_abs_rho": REDUNDANCY_WARNING_ABS_RHO,
        "near_duplicate_abs_rho": NEAR_DUPLICATE_ABS_RHO,
        "max_final_abs_rho": max_final_abs_rho,
        "near_duplicate_pair_count": near_duplicate_count,
    },
    "temporal_stability": {
        "method": (
            "Spearman correlation between selected 2026-2030 mean "
            "summary and each individual future year."
        ),
        "role": "diagnostic only",
        "no_manipulation": True,
    },
    "shap_policy": (
        "Future SHAP is supporting model-association evidence only; "
        "not a CRITIC/TOPSIS criterion and not an automatic selection score."
    ),
    "scenario_policy": (
        "Intervention benefit uses the already-frozen Step29C PRIMARY "
        "combined scenario (+10pp green, MODERATE built reduction). "
        "Values are model-based sensitivity, not causal guarantees."
    ),
    "uncertainty_policy": (
        "Step30 uncertainty remains a separate confidence/reference layer "
        "and is not included as a priority criterion."
    ),
    "not_done_here": [
        "normalization",
        "CRITIC weighting",
        "TOPSIS ranking",
        "priority-class assignment",
        "model retraining",
    ],
    "next": (
        "31D final grid decision matrix + direction conversion + normalization"
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
    "STEP 31C - PRIORITY INDICATOR SELECTION + REDUNDANCY / STABILITY REVIEW",
    "=" * 86,
    "",
    "STATUS: COMPLETED",
    "",
    "FINAL SIX INDICATORS:",
]

for _, r in final_selection.iterrows():
    report_lines.append(
        (
            f"{int(r['criterion_order'])}. "
            f"{r['decision_dimension']} -> "
            f"{r['selected_candidate_id']} "
            f"({r['final_indicator_name']})"
        )
    )

report_lines += [
    "",
    f"Maximum final cross-dimension |Spearman rho|: {max_final_abs_rho:.6f}",
    (
        f"Pairs >= {REDUNDANCY_WARNING_ABS_RHO:.2f}: "
        f"{high_corr_count}"
    ),
    (
        f"Near-duplicate pairs >= {NEAR_DUPLICATE_ABS_RHO:.3f}: "
        f"{near_duplicate_count}"
    ),
    "",
    "METHOD NOTES:",
    "- one criterion frozen per planning decision dimension",
    "- future-state criteria use sustained 2026-2030 representations where appropriate",
    "- trajectory class is not arbitrarily converted to an ordinal score",
    "- Step29C PRIMARY +10pp/MODERATE combined scenario verified exactly",
    "- SHAP remains support-only",
    "- uncertainty remains a separate confidence layer",
    "- no scenario response was clipped or forced to be beneficial",
    "- no CRITIC/TOPSIS calculation was performed",
    "",
    "NEXT:",
    "Step 31D -> build final 2,194-grid decision matrix, apply direction logic, normalize.",
]

REPORT_PATH.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


# =============================================================================
# 13. REPRODUCIBILITY HASHES
# =============================================================================

hash_targets = [
    NUMERIC_VALUES_PATH,
    WITHIN_REDUNDANCY_PATH,
    TEMPORAL_STABILITY_PATH,
    SHAP_SUPPORT_PATH,
    FINAL_SELECTION_PATH,
    SELECTED_RAW_DIAGNOSTIC_PATH,
    FINAL_REDUNDANCY_PATH,
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
# 14. FINAL
# =============================================================================

section(
    "STEP 31C COMPLETED SUCCESSFULLY"
)

print(
    "Final priority indicators frozen: 6"
)

for _, r in final_selection.iterrows():
    print(
        f"  {int(r['criterion_order'])}. "
        f"{r['selected_candidate_id']} -> "
        f"{r['final_field_name']}"
    )

print(
    "\nNo CRITIC weighting performed."
)
print(
    "No TOPSIS ranking performed."
)
print(
    "No model retraining performed."
)
print(
    "No upstream outputs modified."
)
print(
    "\nNEXT: review Step 31C outputs before Step 31D."
)
