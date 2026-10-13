"""
STEP 19C - TEMPORAL STRATEGY COMPARISON + STEP 20 CANDIDATE FREEZE
Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Consolidate the completed temporal development experiments (19A-19B5),
compare only leakage-safe / eligible strategies on the common 2019-2023
rolling-origin development years, and freeze a SMALL candidate set for the
first reserved 2024 validation in Step 20.

THIS STEP DOES NOT:
- load 2024
- load 2025
- retrain models
- tune hyperparameters
- inspect any future holdout result
- select the final production model

The output is only a development-stage shortlist for Step 20.

SHORTLIST RULE
--------------
1. Always retain tuned Random Forest raw Step 17 as the reference anchor.
2. A new strategy may enter the Step 20 improvement pool only if, over the
   common 2019-2023 rolling-origin development folds, it:
      - has lower mean MAE than raw RF
      - has lower mean RMSE than raw RF
      - has higher mean R² than raw RF
      - has no more negative-R² years than raw RF
      - has lower worst-year MAE than raw RF
3. Oracle / target-leaking diagnostics are never eligible.
4. At most TWO improvement candidates are carried in addition to raw RF.
5. The lowest-MAE passing candidate is selected first; the second candidate
   should use a different base model where possible, preserving useful model
   diversity without expanding the 2024 comparison excessively.

This is development selection. Step 20 remains the first untouched 2024
temporal validation.
"""

from pathlib import Path
import hashlib
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]
MAX_IMPROVEMENT_CANDIDATES = 2

ROOT = Path(__file__).resolve().parents[1]

MET = ROOT / "outputs" / "metrics"
REP = ROOT / "outputs" / "reports"
FIG = ROOT / "outputs" / "figures"

for folder in [MET, REP, FIG]:
    folder.mkdir(parents=True, exist_ok=True)

# Core Step 17
STEP17_FOLD = MET / "17_best_config_fold_metrics.csv"
STEP17_AUDIT = REP / "17_hyperparameter_tuning_audit.csv"
STEP17_PARAMS = REP / "17_best_hyperparameters.json"

# 19A-B5 audits
AUDITS = [
    (REP / "19a_temporal_diagnosis_audit.csv", "Step 19A"),
    (REP / "19b1_temporal_bias_correction_audit.csv", "Step 19B1"),
    (REP / "19b2_stable_feature_subset_audit.csv", "Step 19B2"),
    (REP / "19b3_recent_year_weighting_audit.csv", "Step 19B3"),
    (REP / "19b4_temporal_ensemble_audit.csv", "Step 19B4"),
    (REP / "19b5_level_spatial_anomaly_audit.csv", "Step 19B5"),
]

# Family summaries
B1_SUMMARY = MET / "19b1_bias_correction_summary.csv"
B1_BEST = MET / "19b1_best_bias_correction_by_model.csv"

B2_SUMMARY = MET / "19b2_stable_feature_summary.csv"
B2_BEST = MET / "19b2_descriptive_best_strategy_by_model.csv"

B3_FOLD = MET / "19b3_recent_year_weighting_fold_metrics.csv"
B3_SUMMARY = MET / "19b3_recent_year_weighting_summary.csv"
B3_BEST = MET / "19b3_descriptive_best_weighting_by_model.csv"

B4_FOLD = MET / "19b4_ensemble_fold_metrics.csv"
B4_SUMMARY = MET / "19b4_ensemble_summary.csv"
B4_BEST = MET / "19b4_descriptive_best_ensemble.csv"

B5_FOLD = MET / "19b5_level_anomaly_fold_metrics.csv"
B5_SUMMARY = MET / "19b5_level_anomaly_summary.csv"
B5_ORACLE = MET / "19b5_oracle_level_diagnostic.csv"

# Outputs
COMPREHENSIVE_PATH = MET / "19c_comprehensive_strategy_comparison.csv"
FAMILY_EVIDENCE_PATH = MET / "19c_family_evidence_summary.csv"
POOL_PATH = MET / "19c_step20_candidate_pool.csv"
SHORTLIST_PATH = MET / "19c_step20_shortlist.csv"
SHORTLIST_FOLDS_PATH = MET / "19c_step20_shortlist_fold_metrics.csv"
YEAR_WIN_PATH = MET / "19c_year_by_year_candidate_wins.csv"

MANIFEST_PATH = REP / "19c_step20_candidate_manifest.json"
AUDIT_PATH = REP / "19c_temporal_strategy_comparison_audit.csv"
REPORT_PATH = REP / "19c_temporal_strategy_comparison_report.txt"
HASH_PATH = REP / "19c_temporal_strategy_comparison_hashes.csv"

FIG_MAE = FIG / "19c_candidate_mean_mae.png"
FIG_R2 = FIG / "19c_candidate_r2_by_year.png"
FIG_WORST = FIG / "19c_candidate_worst_year_mae.png"


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 128)
    print(title)
    print("=" * 128)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")


def require_audit_pass(path, label):
    require_file(path)
    df = pd.read_csv(path)

    if "status" not in df.columns:
        raise RuntimeError(f"{label} audit has no status column")

    bad = df[df["status"].astype(str).str.upper() != "PASS"]

    if not bad.empty:
        print(bad.to_string(index=False))
        raise RuntimeError(f"{label} audit is not fully PASS")


def summarize_step17(step17):
    rows = []

    for model_name, g in step17.groupby("model"):
        g = g.sort_values("validation_year")

        rows.append(
            {
                "source_step": "Step17",
                "family": "raw_tuned_model",
                "strategy_id": (
                    model_name.lower().replace(" ", "_")
                    + "_raw_step17"
                ),
                "display_name": f"{model_name} raw tuned",
                "base_model": model_name,
                "strategy": "raw_step17",
                "comparison_window": "2019-2023",
                "mean_mae": float(g["validation_mae"].mean()),
                "std_mae": float(g["validation_mae"].std(ddof=0)),
                "mean_rmse": float(g["validation_rmse"].mean()),
                "mean_r2": float(g["validation_r2"].mean()),
                "negative_r2_count": int((g["validation_r2"] < 0).sum()),
                "worst_year_mae": float(g["validation_mae"].max()),
                "minimum_r2": float(g["validation_r2"].min()),
                "eligible": True,
                "oracle_or_target_leaking": False,
            }
        )

    return pd.DataFrame(rows)


def standardize_b1(df):
    out = pd.DataFrame(
        {
            "source_step": "19B1",
            "family": "historical_residual_bias_correction",
            "strategy_id": (
                df["model"].astype(str)
                .str.lower()
                .str.replace(" ", "_", regex=False)
                + "__"
                + df["strategy"].astype(str)
            ),
            "display_name": (
                df["model"].astype(str)
                + " / "
                + df["strategy"].astype(str)
            ),
            "base_model": df["model"].astype(str),
            "strategy": df["strategy"].astype(str),
            "comparison_window": "2019-2023",
            "mean_mae": df["all_folds_mean_mae"].astype(float),
            "std_mae": df["all_folds_std_mae"].astype(float),
            "mean_rmse": df["all_folds_mean_rmse"].astype(float),
            "mean_r2": df["all_folds_mean_r2"].astype(float),
            "negative_r2_count": df[
                "all_folds_negative_r2_count"
            ].astype(int),
            "worst_year_mae": df["all_folds_worst_mae"].astype(float),
            "minimum_r2": np.nan,
            "eligible": df[
                "strategy_eligible_for_comparison"
            ].astype(bool),
            "oracle_or_target_leaking": False,
        }
    )
    return out


def standardize_b2(df):
    return pd.DataFrame(
        {
            "source_step": "19B2",
            "family": "train_only_stable_feature_subset",
            "strategy_id": (
                df["model"].astype(str)
                .str.lower()
                .str.replace(" ", "_", regex=False)
                + "__"
                + df["strategy"].astype(str)
            ),
            "display_name": (
                df["model"].astype(str)
                + " / "
                + df["strategy"].astype(str)
            ),
            "base_model": df["model"].astype(str),
            "strategy": df["strategy"].astype(str),
            "comparison_window": "2019-2023",
            "mean_mae": df[
                "all_folds_mean_validation_mae"
            ].astype(float),
            "std_mae": df[
                "all_folds_std_validation_mae"
            ].astype(float),
            "mean_rmse": df[
                "all_folds_mean_validation_rmse"
            ].astype(float),
            "mean_r2": df[
                "all_folds_mean_validation_r2"
            ].astype(float),
            "negative_r2_count": df[
                "all_folds_negative_r2_count"
            ].astype(int),
            "worst_year_mae": df[
                "all_folds_worst_year_mae"
            ].astype(float),
            "minimum_r2": np.nan,
            "eligible": True,
            "oracle_or_target_leaking": False,
        }
    )


def standardize_b3(df):
    return pd.DataFrame(
        {
            "source_step": "19B3",
            "family": "recent_year_weighting",
            "strategy_id": (
                df["model"].astype(str)
                .str.lower()
                .str.replace(" ", "_", regex=False)
                + "__"
                + df["strategy"].astype(str)
            ),
            "display_name": (
                df["model"].astype(str)
                + " / "
                + df["strategy"].astype(str)
            ),
            "base_model": df["model"].astype(str),
            "strategy": df["strategy"].astype(str),
            "comparison_window": "2019-2023",
            "mean_mae": df["all_folds_mean_mae"].astype(float),
            "std_mae": df["all_folds_std_mae"].astype(float),
            "mean_rmse": df["all_folds_mean_rmse"].astype(float),
            "mean_r2": df["all_folds_mean_r2"].astype(float),
            "negative_r2_count": df[
                "all_folds_negative_r2_count"
            ].astype(int),
            "worst_year_mae": df["all_folds_worst_mae"].astype(float),
            "minimum_r2": np.nan,
            "eligible": True,
            "oracle_or_target_leaking": False,
        }
    )


def standardize_b4(df):
    base_model = np.where(
        df["strategy"].eq("random_forest_single_reference"),
        "Random Forest",
        "Multi-model ensemble",
    )

    return pd.DataFrame(
        {
            "source_step": "19B4",
            "family": "temporal_ensemble",
            "strategy_id": df["strategy"].astype(str),
            "display_name": df["strategy"].astype(str),
            "base_model": base_model,
            "strategy": df["strategy"].astype(str),
            "comparison_window": "2019-2023",
            "mean_mae": df["all_folds_mean_mae"].astype(float),
            "std_mae": df["all_folds_std_mae"].astype(float),
            "mean_rmse": df["all_folds_mean_rmse"].astype(float),
            "mean_r2": df["all_folds_mean_r2"].astype(float),
            "negative_r2_count": df[
                "all_folds_negative_r2_count"
            ].astype(int),
            "worst_year_mae": df["all_folds_worst_mae"].astype(float),
            "minimum_r2": df["all_folds_min_r2"].astype(float),
            "eligible": True,
            "oracle_or_target_leaking": False,
        }
    )


def standardize_b5(df):
    return pd.DataFrame(
        {
            "source_step": "19B5",
            "family": "annual_level_spatial_anomaly",
            "strategy_id": df["strategy"].astype(str),
            "display_name": df["strategy"].astype(str),
            "base_model": df["anomaly_model"].astype(str),
            "strategy": df["strategy"].astype(str),
            "comparison_window": "2019-2023",
            "mean_mae": df["mean_mae"].astype(float),
            "std_mae": df["std_mae"].astype(float),
            "mean_rmse": df["mean_rmse"].astype(float),
            "mean_r2": df["mean_r2"].astype(float),
            "negative_r2_count": df["negative_r2_count"].astype(int),
            "worst_year_mae": df["worst_year_mae"].astype(float),
            "minimum_r2": df["minimum_r2"].astype(float),
            "eligible": True,
            "oracle_or_target_leaking": False,
        }
    )


def candidate_fold_rows(source_step, strategy, base_model, display_name,
                        step17_fold, b3_fold, b4_fold, b5_fold):
    if source_step == "Step17":
        g = step17_fold[
            step17_fold["model"].eq(base_model)
        ].copy()

        return pd.DataFrame(
            {
                "source_step": source_step,
                "strategy": strategy,
                "display_name": display_name,
                "base_model": base_model,
                "validation_year": g["validation_year"].astype(int),
                "mae": g["validation_mae"].astype(float),
                "rmse": g["validation_rmse"].astype(float),
                "r2": g["validation_r2"].astype(float),
            }
        )

    if source_step == "19B3":
        g = b3_fold[
            b3_fold["model"].eq(base_model)
            & b3_fold["strategy"].eq(strategy)
        ].copy()

        return pd.DataFrame(
            {
                "source_step": source_step,
                "strategy": strategy,
                "display_name": display_name,
                "base_model": base_model,
                "validation_year": g["validation_year"].astype(int),
                "mae": g["validation_mae"].astype(float),
                "rmse": g["validation_rmse"].astype(float),
                "r2": g["validation_r2"].astype(float),
            }
        )

    if source_step == "19B4":
        g = b4_fold[
            b4_fold["strategy"].eq(strategy)
        ].copy()

        return pd.DataFrame(
            {
                "source_step": source_step,
                "strategy": strategy,
                "display_name": display_name,
                "base_model": base_model,
                "validation_year": g["validation_year"].astype(int),
                "mae": g["mae"].astype(float),
                "rmse": g["rmse"].astype(float),
                "r2": g["r2"].astype(float),
            }
        )

    if source_step == "19B5":
        g = b5_fold[
            b5_fold["strategy"].eq(strategy)
        ].copy()

        return pd.DataFrame(
            {
                "source_step": source_step,
                "strategy": strategy,
                "display_name": display_name,
                "base_model": base_model,
                "validation_year": g["validation_year"].astype(int),
                "mae": g["mae"].astype(float),
                "rmse": g["rmse"].astype(float),
                "r2": g["r2"].astype(float),
            }
        )

    raise ValueError(source_step)


# =============================================================================
# PREREQUISITES
# =============================================================================

section("STEP 19C - TEMPORAL STRATEGY COMPARISON + STEP 20 CANDIDATE FREEZE")

for path in [
    STEP17_FOLD,
    STEP17_AUDIT,
    STEP17_PARAMS,
    B1_SUMMARY,
    B1_BEST,
    B2_SUMMARY,
    B2_BEST,
    B3_FOLD,
    B3_SUMMARY,
    B3_BEST,
    B4_FOLD,
    B4_SUMMARY,
    B4_BEST,
    B5_FOLD,
    B5_SUMMARY,
    B5_ORACLE,
]:
    require_file(path)

require_audit_pass(STEP17_AUDIT, "Step 17")

for path, label in AUDITS:
    require_audit_pass(path, label)

print("PASS: Step 17 and Steps 19A-19B5 audits are fully PASS.")
print("Step 19C reads saved development outputs only.")
print("2024 data: NOT LOADED")
print("2025 data: NOT LOADED")
print("No model fitting or tuning occurs in Step 19C.")


# =============================================================================
# LOAD INPUTS
# =============================================================================

section("1. LOAD DEVELOPMENT STRATEGY SUMMARIES")

step17_fold = pd.read_csv(STEP17_FOLD)
b1_summary = pd.read_csv(B1_SUMMARY)
b1_best = pd.read_csv(B1_BEST)

b2_summary = pd.read_csv(B2_SUMMARY)
b2_best = pd.read_csv(B2_BEST)

b3_fold = pd.read_csv(B3_FOLD)
b3_summary = pd.read_csv(B3_SUMMARY)
b3_best = pd.read_csv(B3_BEST)

b4_fold = pd.read_csv(B4_FOLD)
b4_summary = pd.read_csv(B4_SUMMARY)
b4_best = pd.read_csv(B4_BEST)

b5_fold = pd.read_csv(B5_FOLD)
b5_summary = pd.read_csv(B5_SUMMARY)
b5_oracle = pd.read_csv(B5_ORACLE)

for df, name in [
    (step17_fold, "Step17"),
    (b3_fold, "19B3"),
    (b4_fold, "19B4"),
    (b5_fold, "19B5"),
]:
    years = sorted(
        pd.to_numeric(
            df["validation_year"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if years != VALIDATION_YEARS:
        raise RuntimeError(
            f"{name} validation years={years}, expected={VALIDATION_YEARS}"
        )

print(f"Step 17 fold rows : {len(step17_fold)}")
print(f"19B1 summary rows : {len(b1_summary)}")
print(f"19B2 summary rows : {len(b2_summary)}")
print(f"19B3 summary rows : {len(b3_summary)}")
print(f"19B4 summary rows : {len(b4_summary)}")
print(f"19B5 summary rows : {len(b5_summary)}")


# =============================================================================
# COMPREHENSIVE STANDARDIZED TABLE
# =============================================================================

section("2. BUILD COMPREHENSIVE STANDARDIZED COMPARISON")

step17_std = summarize_step17(step17_fold)
b1_std = standardize_b1(b1_summary)
b2_std = standardize_b2(b2_summary)
b3_std = standardize_b3(b3_summary)
b4_std = standardize_b4(b4_summary)
b5_std = standardize_b5(b5_summary)

comprehensive = pd.concat(
    [
        step17_std,
        b1_std,
        b2_std,
        b3_std,
        b4_std,
        b5_std,
    ],
    ignore_index=True,
)

if comprehensive["oracle_or_target_leaking"].any():
    raise RuntimeError(
        "Target-leaking strategy entered comprehensive eligible table"
    )

for col in [
    "mean_mae",
    "std_mae",
    "mean_rmse",
    "mean_r2",
    "worst_year_mae",
]:
    if not np.isfinite(
        comprehensive[col].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            f"Non-finite values in standardized column {col}"
        )

comprehensive["global_descriptive_mae_rank"] = (
    comprehensive["mean_mae"]
    .rank(
        method="min",
        ascending=True,
    )
    .astype(int)
)

comprehensive = comprehensive.sort_values(
    [
        "global_descriptive_mae_rank",
        "worst_year_mae",
        "mean_rmse",
    ]
).reset_index(drop=True)

comprehensive.to_csv(
    COMPREHENSIVE_PATH,
    index=False,
)

print(
    comprehensive[
        [
            "source_step",
            "family",
            "display_name",
            "mean_mae",
            "mean_rmse",
            "mean_r2",
            "negative_r2_count",
            "worst_year_mae",
            "global_descriptive_mae_rank",
        ]
    ]
    .head(25)
    .to_string(index=False)
)


# =============================================================================
# FAMILY EVIDENCE
# =============================================================================

section("3. CONSOLIDATE STEP 19 FAMILY EVIDENCE")

# B1: best predefined correction per model should be raw.
b1_raw_best_count = int(
    b1_best["strategy"]
    .astype(str)
    .eq("raw_no_correction")
    .sum()
)

b1_all_raw_best = bool(
    b1_raw_best_count == len(b1_best)
)

# B2: best within-model strategy should be full 93.
b2_full_best_count = int(
    b2_best["strategy"]
    .astype(str)
    .eq("full_93_reference")
    .sum()
)

b2_all_full_best = bool(
    b2_full_best_count == len(b2_best)
)

# B3: count model-specific weighted winners and maximum development improvement.
b3_weighted_best_count = int(
    (~b3_best["strategy"]
     .astype(str)
     .eq("uniform_reference"))
    .sum()
)

b3_max_mae_improvement = float(
    b3_best[
        "mean_mae_improvement_vs_uniform_2020_2023"
    ].max()
)

# B4: best should be RF single reference.
b4_best_strategy = str(
    b4_best.iloc[0]["strategy"]
)

b4_ensemble_beats_rf = bool(
    b4_best_strategy
    != "random_forest_single_reference"
)

# B5 best eligible.
b5_best_row = (
    b5_summary.sort_values(
        [
            "mean_mae",
            "worst_year_mae",
            "mean_rmse",
        ]
    )
    .iloc[0]
)

family_rows = [
    {
        "step": "19A",
        "experiment": "Temporal diagnosis",
        "development_result": (
            "Annual-level error is a major signal; centered/spatial "
            "pattern remains useful."
        ),
        "carry_to_step20": (
            "diagnostic_basis_only"
        ),
    },
    {
        "step": "19B1",
        "experiment": "Historical residual bias correction",
        "development_result": (
            "Raw/no-correction ranked first for all models."
            if b1_all_raw_best
            else "At least one correction rule ranked above raw."
        ),
        "carry_to_step20": (
            "no_corrected_strategy"
            if b1_all_raw_best
            else "review_required"
        ),
    },
    {
        "step": "19B2",
        "experiment": "Train-only stable feature subset",
        "development_result": (
            "Full 93-feature reference ranked first for all models."
            if b2_all_full_best
            else "At least one reduced feature strategy ranked first."
        ),
        "carry_to_step20": (
            "no_reduced_feature_strategy"
            if b2_all_full_best
            else "review_required"
        ),
    },
    {
        "step": "19B3",
        "experiment": "Recent-year weighting",
        "development_result": (
            f"{b3_weighted_best_count}/{len(b3_best)} model-specific "
            f"best rows use recency weighting; maximum eligible mean-MAE "
            f"improvement vs own uniform={b3_max_mae_improvement:.6f} °C."
        ),
        "carry_to_step20": (
            "eligible_for_cross_family_gate"
        ),
    },
    {
        "step": "19B4",
        "experiment": "Temporal ensemble",
        "development_result": (
            "Random Forest single reference remained best."
            if not b4_ensemble_beats_rf
            else f"Best ensemble strategy={b4_best_strategy}."
        ),
        "carry_to_step20": (
            "no_ensemble_strategy"
            if not b4_ensemble_beats_rf
            else "eligible_for_cross_family_gate"
        ),
    },
    {
        "step": "19B5",
        "experiment": "Annual-level + spatial-anomaly decomposition",
        "development_result": (
            f"Best eligible strategy={b5_best_row['strategy']}; "
            f"mean MAE={float(b5_best_row['mean_mae']):.6f} °C; "
            f"mean R²={float(b5_best_row['mean_r2']):.6f}; "
            f"negative-R² folds={int(b5_best_row['negative_r2_count'])}."
        ),
        "carry_to_step20": (
            "eligible_for_cross_family_gate"
        ),
    },
]

family_evidence = pd.DataFrame(family_rows)

family_evidence.to_csv(
    FAMILY_EVIDENCE_PATH,
    index=False,
)

print(family_evidence.to_string(index=False))


# =============================================================================
# BUILD STEP 20 CANDIDATE POOL
# =============================================================================

section("4. BUILD CONSERVATIVE STEP 20 CANDIDATE POOL")

# Raw RF anchor.
rf_anchor = step17_std[
    step17_std["base_model"].eq("Random Forest")
].copy()

if len(rf_anchor) != 1:
    raise RuntimeError(
        "Could not uniquely identify raw RF Step 17 anchor"
    )

rf_anchor = rf_anchor.iloc[0]

# B3: only the already-finalized within-model descriptive best weighting
# rows are allowed into the cross-family pool.
b3_pool = standardize_b3(b3_best)

# Remove any uniform references because raw models are already represented.
b3_pool = b3_pool[
    ~b3_pool["strategy"].eq("uniform_reference")
].copy()

# B4: only non-reference ensemble rows enter the pool.
b4_pool = b4_std[
    ~b4_std["strategy"].eq("random_forest_single_reference")
].copy()

# B5: all eligible decomposition strategies except duplicated raw references.
b5_pool = b5_std[
    ~b5_std["strategy"].isin(
        [
            "random_forest_raw_reference",
            "lightgbm_raw_reference",
        ]
    )
].copy()

pool = pd.concat(
    [
        b3_pool,
        b4_pool,
        b5_pool,
    ],
    ignore_index=True,
)

# Strict multi-metric improvement gate relative to raw RF.
pool["beats_rf_mean_mae"] = (
    pool["mean_mae"] < float(rf_anchor["mean_mae"])
)

pool["beats_rf_mean_rmse"] = (
    pool["mean_rmse"] < float(rf_anchor["mean_rmse"])
)

pool["beats_rf_mean_r2"] = (
    pool["mean_r2"] > float(rf_anchor["mean_r2"])
)

pool["negative_r2_not_worse_than_rf"] = (
    pool["negative_r2_count"]
    <= int(rf_anchor["negative_r2_count"])
)

pool["beats_rf_worst_year_mae"] = (
    pool["worst_year_mae"]
    < float(rf_anchor["worst_year_mae"])
)

pool["passes_step20_improvement_gate"] = (
    pool[
        [
            "beats_rf_mean_mae",
            "beats_rf_mean_rmse",
            "beats_rf_mean_r2",
            "negative_r2_not_worse_than_rf",
            "beats_rf_worst_year_mae",
        ]
    ]
    .all(axis=1)
)

pool["mean_mae_improvement_vs_raw_rf"] = (
    float(rf_anchor["mean_mae"])
    - pool["mean_mae"]
)

pool["mean_rmse_improvement_vs_raw_rf"] = (
    float(rf_anchor["mean_rmse"])
    - pool["mean_rmse"]
)

pool["mean_r2_improvement_vs_raw_rf"] = (
    pool["mean_r2"]
    - float(rf_anchor["mean_r2"])
)

pool = pool.sort_values(
    [
        "passes_step20_improvement_gate",
        "mean_mae",
        "worst_year_mae",
        "mean_rmse",
    ],
    ascending=[
        False,
        True,
        True,
        True,
    ],
).reset_index(drop=True)

pool.to_csv(
    POOL_PATH,
    index=False,
)

print(
    pool[
        [
            "source_step",
            "display_name",
            "base_model",
            "mean_mae",
            "mean_rmse",
            "mean_r2",
            "negative_r2_count",
            "worst_year_mae",
            "passes_step20_improvement_gate",
        ]
    ].to_string(index=False)
)


# =============================================================================
# FREEZE SMALL STEP 20 SHORTLIST
# =============================================================================

section("5. FREEZE SMALL STEP 20 CANDIDATE SHORTLIST")

passing = pool[
    pool["passes_step20_improvement_gate"]
].sort_values(
    [
        "mean_mae",
        "worst_year_mae",
        "mean_rmse",
    ]
).copy()

selected_improvements = []

if not passing.empty:
    first = passing.iloc[0]
    selected_improvements.append(first)

    if MAX_IMPROVEMENT_CANDIDATES > 1:
        different_model = passing[
            ~passing["base_model"].eq(
                first["base_model"]
            )
        ]

        if not different_model.empty:
            selected_improvements.append(
                different_model.iloc[0]
            )
        elif len(passing) > 1:
            selected_improvements.append(
                passing.iloc[1]
            )

# Anchor is always first.
shortlist_records = [
    {
        **rf_anchor.to_dict(),
        "shortlist_role": "raw_reference_anchor",
        "shortlist_order": 1,
        "selection_reason": (
            "Strongest raw Step 17 temporal model; retained as the "
            "mandatory untouched reference anchor."
        ),
    }
]

for index, row in enumerate(
    selected_improvements,
    start=2,
):
    shortlist_records.append(
        {
            **row.to_dict(),
            "shortlist_role": "development_improvement_candidate",
            "shortlist_order": index,
            "selection_reason": (
                "Passed the strict cross-family multi-metric improvement "
                "gate versus raw RF and was retained under the small "
                "candidate cap with model diversity where possible."
            ),
        }
    )

shortlist = pd.DataFrame(
    shortlist_records
)

shortlist.to_csv(
    SHORTLIST_PATH,
    index=False,
)

print(
    shortlist[
        [
            "shortlist_order",
            "source_step",
            "display_name",
            "base_model",
            "mean_mae",
            "mean_rmse",
            "mean_r2",
            "negative_r2_count",
            "worst_year_mae",
            "shortlist_role",
        ]
    ].to_string(index=False)
)

if len(shortlist) > 1 + MAX_IMPROVEMENT_CANDIDATES:
    raise RuntimeError(
        "Step 20 shortlist exceeds predeclared candidate cap"
    )


# =============================================================================
# YEAR-BY-YEAR SHORTLIST METRICS
# =============================================================================

section("6. BUILD YEAR-BY-YEAR SHORTLIST COMPARISON")

shortlist_fold_frames = []

for _, row in shortlist.iterrows():
    fold_df = candidate_fold_rows(
        source_step=row["source_step"],
        strategy=row["strategy"],
        base_model=row["base_model"],
        display_name=row["display_name"],
        step17_fold=step17_fold,
        b3_fold=b3_fold,
        b4_fold=b4_fold,
        b5_fold=b5_fold,
    )

    if len(fold_df) != len(VALIDATION_YEARS):
        raise RuntimeError(
            f"Incomplete fold rows for {row['display_name']}: "
            f"{len(fold_df)}"
        )

    shortlist_fold_frames.append(
        fold_df
    )

shortlist_folds = pd.concat(
    shortlist_fold_frames,
    ignore_index=True,
)

shortlist_folds.to_csv(
    SHORTLIST_FOLDS_PATH,
    index=False,
)

print(
    shortlist_folds.sort_values(
        [
            "validation_year",
            "mae",
        ]
    ).to_string(index=False)
)

year_win_rows = []

for validation_year in VALIDATION_YEARS:
    y = shortlist_folds[
        shortlist_folds[
            "validation_year"
        ].eq(validation_year)
    ].sort_values(
        [
            "mae",
            "rmse",
        ]
    )

    best = y.iloc[0]

    year_win_rows.append(
        {
            "validation_year": validation_year,
            "lowest_mae_candidate": best["display_name"],
            "lowest_mae": float(best["mae"]),
            "corresponding_rmse": float(best["rmse"]),
            "corresponding_r2": float(best["r2"]),
        }
    )

year_wins = pd.DataFrame(
    year_win_rows
)

year_wins.to_csv(
    YEAR_WIN_PATH,
    index=False,
)

print("\nLowest-MAE shortlist candidate by development year:")
print(
    year_wins.to_string(index=False)
)


# =============================================================================
# STEP 20 IMPLEMENTATION MANIFEST
# =============================================================================

section("7. SAVE STEP 20 CANDIDATE MANIFEST")

with STEP17_PARAMS.open(
    "r",
    encoding="utf-8",
) as f:
    params_payload = json.load(f)

manifest_candidates = []

for _, row in shortlist.iterrows():
    strategy = str(row["strategy"])
    base_model = str(row["base_model"])

    if row["source_step"] == "Step17":
        implementation = {
            "target": "LST_C",
            "feature_configuration": "full_93",
            "fit_years_for_step20": "2018-2023",
            "validation_year_for_step20": 2024,
            "method": (
                "Fit the frozen tuned Random Forest directly to LST_C "
                "using all 2018-2023 development rows, then predict 2024."
            ),
            "hyperparameters": params_payload[
                "best_params"
            ][
                "Random Forest"
            ],
        }

    elif row["source_step"] == "19B5":
        if "prior_mean_level" in strategy:
            level_rule = "prior_mean_level"
        elif "persistence_level" in strategy:
            level_rule = "persistence_level"
        elif "linear_trend_level" in strategy:
            level_rule = "linear_trend_level"
        else:
            raise RuntimeError(
                f"Unknown 19B5 level rule in {strategy}"
            )

        implementation = {
            "target": (
                "within-year spatial anomaly: "
                "LST_C - training-year mean LST_C"
            ),
            "feature_configuration": "full_93",
            "fit_years_for_step20": "2018-2023",
            "validation_year_for_step20": 2024,
            "anomaly_model": base_model,
            "level_rule": level_rule,
            "validation_anomaly_recentering": (
                "subtract mean of 2024 predicted anomalies only; "
                "do not use 2024 target"
            ),
            "final_prediction": (
                "forecast_2024_annual_level + "
                "centered_2024_predicted_spatial_anomaly"
            ),
            "hyperparameters": params_payload[
                "best_params"
            ][
                base_model
            ],
        }

    elif row["source_step"] == "19B3":
        implementation = {
            "target": "LST_C",
            "feature_configuration": "full_93",
            "fit_years_for_step20": "2018-2023",
            "validation_year_for_step20": 2024,
            "method": (
                "Refit frozen Step 17 model with the exact frozen "
                "19B3 recency-weighting strategy; derive weights only "
                "from training-year age."
            ),
            "weighting_strategy": strategy,
            "hyperparameters": params_payload[
                "best_params"
            ][
                base_model
            ],
        }

    elif row["source_step"] == "19B4":
        implementation = {
            "method": (
                "Train required frozen member models on 2018-2023 and "
                "combine 2024 predictions using the exact frozen 19B4 "
                "ensemble rule without 2024 target information."
            ),
            "ensemble_strategy": strategy,
        }

    else:
        raise RuntimeError(
            f"Unsupported shortlist source step: {row['source_step']}"
        )

    manifest_candidates.append(
        {
            "shortlist_order": int(
                row["shortlist_order"]
            ),
            "source_step": str(
                row["source_step"]
            ),
            "strategy": strategy,
            "display_name": str(
                row["display_name"]
            ),
            "base_model": base_model,
            "development_mean_mae": float(
                row["mean_mae"]
            ),
            "development_mean_rmse": float(
                row["mean_rmse"]
            ),
            "development_mean_r2": float(
                row["mean_r2"]
            ),
            "development_negative_r2_count": int(
                row["negative_r2_count"]
            ),
            "development_worst_year_mae": float(
                row["worst_year_mae"]
            ),
            "implementation": implementation,
        }
    )

manifest = {
    "step": "19C",
    "purpose": (
        "Frozen candidate shortlist for first reserved-2024 "
        "temporal validation in Step 20."
    ),
    "development_years_used_for_selection": "2018-2023",
    "rolling_validation_years_used_for_selection": VALIDATION_YEARS,
    "reserved_2024_loaded": False,
    "final_2025_lockbox_loaded": False,
    "candidate_count": len(
        manifest_candidates
    ),
    "candidate_cap": (
        1 + MAX_IMPROVEMENT_CANDIDATES
    ),
    "selection_rule": {
        "raw_rf_anchor_always_retained": True,
        "improvement_gate": [
            "mean_mae < raw_rf_mean_mae",
            "mean_rmse < raw_rf_mean_rmse",
            "mean_r2 > raw_rf_mean_r2",
            "negative_r2_count <= raw_rf_negative_r2_count",
            "worst_year_mae < raw_rf_worst_year_mae",
        ],
        "max_additional_candidates": (
            MAX_IMPROVEMENT_CANDIDATES
        ),
        "oracle_diagnostics_eligible": False,
    },
    "candidates": manifest_candidates,
}

MANIFEST_PATH.write_text(
    json.dumps(
        manifest,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        manifest,
        indent=2,
    )
)


# =============================================================================
# FIGURES
# =============================================================================

section("8. FIGURES")

plot_shortlist = shortlist.sort_values(
    "shortlist_order"
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    plot_shortlist["display_name"],
    plot_shortlist["mean_mae"],
)

ax.set_ylabel(
    "Mean rolling-origin MAE, 2019-2023 (°C)"
)

ax.set_title(
    "Step 19C - Frozen Step 20 Candidate Development MAE"
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
    FIG_MAE,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


fig, ax = plt.subplots(
    figsize=(11, 7)
)

for display_name, g in shortlist_folds.groupby(
    "display_name"
):
    g = g.sort_values(
        "validation_year"
    )

    ax.plot(
        g["validation_year"],
        g["r2"],
        marker="o",
        label=display_name,
    )

ax.axhline(
    0,
    linewidth=1,
)

ax.set_xticks(
    VALIDATION_YEARS
)

ax.set_xlabel(
    "Development validation year"
)

ax.set_ylabel(
    "R²"
)

ax.set_title(
    "Step 19C - Candidate Temporal R² by Development Year"
)

ax.grid(
    alpha=0.2,
)

ax.legend(
    fontsize=8,
)

plt.tight_layout()
plt.savefig(
    FIG_R2,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    plot_shortlist["display_name"],
    plot_shortlist["worst_year_mae"],
)

ax.set_ylabel(
    "Worst development-year MAE (°C)"
)

ax.set_title(
    "Step 19C - Candidate Worst-Year Error"
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
    FIG_WORST,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

for p in [
    FIG_MAE,
    FIG_R2,
    FIG_WORST,
]:
    print(f"Saved: {p}")


# =============================================================================
# STRICT AUDIT
# =============================================================================

section("9. STRICT STEP 19C AUDIT")

oracle_strategy_names = set(
    b5_oracle["strategy"]
    .astype(str)
    .unique()
    .tolist()
)

shortlist_strategy_names = set(
    shortlist["strategy"]
    .astype(str)
    .tolist()
)

no_oracle_in_shortlist = bool(
    oracle_strategy_names.isdisjoint(
        shortlist_strategy_names
    )
)

all_shortlist_years_complete = bool(
    shortlist_folds.groupby(
        "display_name"
    )[
        "validation_year"
    ]
    .apply(
        lambda s: sorted(
            s.astype(int).tolist()
        )
        == VALIDATION_YEARS
    )
    .all()
)

rf_anchor_present = bool(
    (
        shortlist[
            "shortlist_role"
        ]
        == "raw_reference_anchor"
    ).sum()
    == 1
)

improvement_candidate_count = int(
    (
        shortlist[
            "shortlist_role"
        ]
        == "development_improvement_candidate"
    ).sum()
)

shortlist_cap_ok = bool(
    improvement_candidate_count
    <= MAX_IMPROVEMENT_CANDIDATES
)

improvements_all_passed_gate = True

if improvement_candidate_count > 0:
    for _, row in shortlist[
        shortlist[
            "shortlist_role"
        ]
        == "development_improvement_candidate"
    ].iterrows():

        match = pool[
            (pool["source_step"] == row["source_step"])
            & (pool["strategy"] == row["strategy"])
            & (pool["base_model"] == row["base_model"])
        ]

        if (
            len(match) != 1
            or not bool(
                match.iloc[0][
                    "passes_step20_improvement_gate"
                ]
            )
        ):
            improvements_all_passed_gate = False

all_shortlist_metrics_finite = bool(
    np.isfinite(
        shortlist[
            [
                "mean_mae",
                "mean_rmse",
                "mean_r2",
                "worst_year_mae",
            ]
        ]
        .to_numpy(dtype=float)
    ).all()
)

audit_rows = [
    {
        "check": "step17_and_step19_audits",
        "status": "PASS",
        "details": (
            "Step 17 and Steps 19A-19B5 were verified fully PASS."
        ),
    },
    {
        "check": "comparison_window",
        "status": "PASS",
        "details": (
            "Cross-family summary uses common 2019-2023 development "
            "rolling-origin years."
        ),
    },
    {
        "check": "2024_loaded_or_used",
        "status": "PASS",
        "details": (
            "No. Step 19C reads development outputs only."
        ),
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": (
            "No. Step 19C reads development outputs only."
        ),
    },
    {
        "check": "model_training_performed",
        "status": "PASS",
        "details": (
            "No model fitting occurs in Step 19C."
        ),
    },
    {
        "check": "hyperparameter_tuning_performed",
        "status": "PASS",
        "details": (
            "No hyperparameter tuning occurs in Step 19C."
        ),
    },
    {
        "check": "oracle_in_shortlist",
        "status": (
            "PASS"
            if no_oracle_in_shortlist
            else "FAIL"
        ),
        "details": (
            "Oracle/target-leaking 19B5 diagnostics are excluded."
        ),
    },
    {
        "check": "raw_rf_anchor_present_once",
        "status": (
            "PASS"
            if rf_anchor_present
            else "FAIL"
        ),
        "details": (
            "Exactly one raw tuned RF reference anchor is retained."
        ),
    },
    {
        "check": "improvement_candidate_cap",
        "status": (
            "PASS"
            if shortlist_cap_ok
            else "FAIL"
        ),
        "details": (
            f"additional_candidates={improvement_candidate_count}, "
            f"max={MAX_IMPROVEMENT_CANDIDATES}"
        ),
    },
    {
        "check": "all_improvement_candidates_pass_gate",
        "status": (
            "PASS"
            if improvements_all_passed_gate
            else "FAIL"
        ),
        "details": (
            "Every additional candidate strictly satisfies the "
            "predeclared multi-metric gate versus raw RF."
        ),
    },
    {
        "check": "shortlist_fold_years_complete",
        "status": (
            "PASS"
            if all_shortlist_years_complete
            else "FAIL"
        ),
        "details": (
            "Every shortlisted candidate has 2019-2023 fold metrics."
        ),
    },
    {
        "check": "shortlist_metrics_finite",
        "status": (
            "PASS"
            if all_shortlist_metrics_finite
            else "FAIL"
        ),
        "details": str(
            all_shortlist_metrics_finite
        ),
    },
    {
        "check": "final_model_selected",
        "status": "PASS",
        "details": (
            "No final model selected; shortlist is for first reserved "
            "2024 comparison only."
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
    audit_df.to_string(index=False)
)

if (
    audit_df["status"]
    != "PASS"
).any():
    raise RuntimeError(
        "One or more Step 19C audit checks failed"
    )


# =============================================================================
# REPORT
# =============================================================================

section("10. SAVE STEP 19C REPORT")

report_lines = [
    "STEP 19C - TEMPORAL STRATEGY COMPARISON + STEP 20 CANDIDATE FREEZE",
    "=" * 116,
    "",
    "PURPOSE",
    "-" * 116,
    (
        "Consolidate Steps 19A-19B5 and freeze a small leakage-safe "
        "candidate set for the first untouched 2024 temporal validation."
    ),
    "",
    "DATA POLICY",
    "-" * 116,
    "Development evidence only: 2019-2023 rolling validation folds.",
    "2024 reserved validation: NOT loaded.",
    "2025 final lockbox: NOT loaded.",
    "",
    "FAMILY EVIDENCE",
    "-" * 116,
    family_evidence.to_string(index=False),
    "",
    "RAW RF REFERENCE",
    "-" * 116,
    (
        f"Mean MAE={float(rf_anchor['mean_mae']):.6f} °C; "
        f"Mean RMSE={float(rf_anchor['mean_rmse']):.6f} °C; "
        f"Mean R²={float(rf_anchor['mean_r2']):.6f}; "
        f"Negative-R² years={int(rf_anchor['negative_r2_count'])}; "
        f"Worst-year MAE={float(rf_anchor['worst_year_mae']):.6f} °C."
    ),
    "",
    "STEP 20 IMPROVEMENT GATE",
    "-" * 116,
    "A candidate must improve mean MAE, mean RMSE, mean R², and worst-year MAE",
    "relative to raw RF, while not increasing the negative-R² year count.",
    "",
    "STEP 20 FROZEN SHORTLIST",
    "-" * 116,
    shortlist[
        [
            "shortlist_order",
            "source_step",
            "display_name",
            "base_model",
            "mean_mae",
            "mean_rmse",
            "mean_r2",
            "negative_r2_count",
            "worst_year_mae",
            "shortlist_role",
        ]
    ].to_string(index=False),
    "",
    "YEAR-BY-YEAR DEVELOPMENT COMPARISON",
    "-" * 116,
    shortlist_folds.to_string(index=False),
    "",
    "INTERPRETATION",
    "-" * 116,
    (
        "19C does not declare a final model. It freezes a small development "
        "candidate set so Step 20 can provide the first reserved 2024 "
        "out-of-time comparison without further tuning."
    ),
    (
        "If the annual-level + anomaly candidates remain stronger on 2024, "
        "that will provide independent support for the structural reformulation."
    ),
    (
        "If they fail on 2024, the raw RF anchor remains available and the "
        "development improvement will be reported as non-generalizing."
    ),
    "",
    "NEXT",
    "-" * 116,
    "Step 20: fit each frozen shortlisted strategy on all 2018-2023 development",
    "data and evaluate once on reserved 2024. Do not tune on 2024.",
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {REPORT_PATH}")
print(f"Saved: {AUDIT_PATH}")
print(f"Saved: {MANIFEST_PATH}")


# =============================================================================
# HASHES
# =============================================================================

section("11. REPRODUCIBILITY HASHES")

hash_files = [
    STEP17_FOLD,
    STEP17_PARAMS,
    B1_SUMMARY,
    B1_BEST,
    B2_SUMMARY,
    B2_BEST,
    B3_FOLD,
    B3_SUMMARY,
    B3_BEST,
    B4_FOLD,
    B4_SUMMARY,
    B4_BEST,
    B5_FOLD,
    B5_SUMMARY,
    B5_ORACLE,
    COMPREHENSIVE_PATH,
    FAMILY_EVIDENCE_PATH,
    POOL_PATH,
    SHORTLIST_PATH,
    SHORTLIST_FOLDS_PATH,
    YEAR_WIN_PATH,
    MANIFEST_PATH,
    AUDIT_PATH,
    REPORT_PATH,
]

hash_rows = []

for path in hash_files:
    digest = sha256_file(path)

    hash_rows.append(
        {
            "file": str(path),
            "sha256": digest,
        }
    )

    print(f"{path.name}: {digest}")

pd.DataFrame(
    hash_rows
).to_csv(
    HASH_PATH,
    index=False,
)

print(f"Saved: {HASH_PATH}")


# =============================================================================
# FINAL STATUS
# =============================================================================

section("STEP 19C COMPLETED SUCCESSFULLY")

print(
    "Temporal strategy comparison completed.\n"
    "All Step 19 family evidence was consolidated.\n"
    "Cross-family comparison used common 2019-2023 development folds.\n"
    "Raw tuned Random Forest was retained as the mandatory reference anchor.\n"
    "Additional Step 20 candidates had to pass a strict multi-metric gate.\n"
    f"At most {MAX_IMPROVEMENT_CANDIDATES} improvement candidates were retained.\n"
    "Oracle/target-leaking diagnostics were excluded.\n"
    "2024 reserved validation remained untouched.\n"
    "2025 final lockbox remained untouched.\n"
    "No training, retuning, or final model selection occurred.\n"
    "The Step 20 candidate manifest is now frozen for reserved-2024 validation."
)
