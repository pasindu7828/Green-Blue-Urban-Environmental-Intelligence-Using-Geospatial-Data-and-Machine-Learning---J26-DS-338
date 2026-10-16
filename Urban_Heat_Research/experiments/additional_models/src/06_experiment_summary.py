"""
STEP 06 - SUPPLEMENTARY EXPERIMENT SUMMARY AND DECISION GATE
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Summarize the completed additional-model experiment and record a conservative
production decision without retraining or modifying the finalized pipeline.

This step DOES NOT:
- load 2024 validation data
- load 2025 test/lockbox data
- retrain any model
- overwrite production models
- overwrite production outputs
- modify backend files

Decision principle
------------------
A production model should not be replaced merely because another model has a
very small average development advantage. The evidence should also be stable
across rolling-origin folds and meaningfully stronger overall.

The decision recorded here is a governance/research decision based on the
already-completed Step 05 comparison; it is not a new statistical test.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]

METRIC_DIR = EXPERIMENT_ROOT / "outputs" / "metrics"
REPORT_DIR = EXPERIMENT_ROOT / "outputs" / "reports"

STEP05_SUMMARY = (
    METRIC_DIR
    / "05_model_comparison_summary.csv"
)

STEP05_VS_LIGHTGBM = (
    METRIC_DIR
    / "05_candidate_vs_lightgbm.csv"
)

STEP05_FOLD_MAE = (
    METRIC_DIR
    / "05_fold_by_fold_mae_comparison.csv"
)

STEP05_AUDIT = (
    REPORT_DIR
    / "05_model_comparison_audit.csv"
)

DECISION_CSV = (
    METRIC_DIR
    / "06_experiment_decision.csv"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "06_experiment_summary.json"
)

AUDIT_PATH = (
    REPORT_DIR
    / "06_experiment_summary_audit.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "06_experiment_summary_report.txt"
)

HASH_PATH = (
    REPORT_DIR
    / "06_experiment_summary_hashes.csv"
)


# =============================================================================
# 2. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 124)
    print(title)
    print("=" * 124)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


# =============================================================================
# 3. START
# =============================================================================

section("STEP 06 - SUPPLEMENTARY EXPERIMENT SUMMARY AND DECISION GATE")

print(f"Project root       : {PROJECT_ROOT}")
print(f"Experiment root    : {EXPERIMENT_ROOT}")
print("2024 data          : NOT LOADED BY THIS STEP")
print("2025 data          : NOT LOADED BY THIS STEP")
print("Production model   : NOT MODIFIED")
print("Backend            : NOT MODIFIED")


# =============================================================================
# 4. VERIFY STEP 05
# =============================================================================

section("1. VERIFY STEP 05")

for path in [
    STEP05_SUMMARY,
    STEP05_VS_LIGHTGBM,
    STEP05_FOLD_MAE,
    STEP05_AUDIT,
]:
    require_file(path)

audit05 = pd.read_csv(STEP05_AUDIT)

if not (
    audit05["status"].astype(str) == "PASS"
).all():
    raise RuntimeError(
        "Step 05 audit is not fully PASS."
    )

print("Step 05 audit: PASS")


# =============================================================================
# 5. LOAD RESULTS
# =============================================================================

section("2. LOAD DEVELOPMENT COMPARISON RESULTS")

summary = pd.read_csv(
    STEP05_SUMMARY
)

vs_lgbm = pd.read_csv(
    STEP05_VS_LIGHTGBM
)

fold_mae = pd.read_csv(
    STEP05_FOLD_MAE
)

required_models = {
    "LightGBM",
    "CatBoost",
    "Extra Trees",
    "HistGradientBoosting",
}

actual_models = set(
    summary["model"].astype(str)
)

if actual_models != required_models:
    raise RuntimeError(
        "Unexpected Step 05 model set.\n"
        f"Expected: {sorted(required_models)}\n"
        f"Actual  : {sorted(actual_models)}"
    )

summary = summary.sort_values(
    "development_rank"
).reset_index(drop=True)

winner_row = summary.iloc[0]
winner = str(
    winner_row["model"]
)

winner_mae = float(
    winner_row["mean_mae"]
)

lightgbm_row = summary[
    summary["model"] == "LightGBM"
].iloc[0]

lightgbm_mae = float(
    lightgbm_row["mean_mae"]
)

lightgbm_rmse = float(
    lightgbm_row["mean_rmse"]
)

lightgbm_spatial_r2 = float(
    lightgbm_row[
        "mean_centered_spatial_r2"
    ]
)

winner_rmse = float(
    winner_row["mean_rmse"]
)

winner_spatial_r2 = float(
    winner_row[
        "mean_centered_spatial_r2"
    ]
)

mae_delta = float(
    winner_mae - lightgbm_mae
)

mae_improvement_percent = float(
    (
        lightgbm_mae - winner_mae
    )
    / lightgbm_mae
    * 100.0
)

winner_vs_row = vs_lgbm[
    vs_lgbm["candidate"].astype(str)
    == winner
]

if winner == "LightGBM":
    candidate_fold_wins = 0
    lightgbm_fold_wins = 0
    mae_ties = 5
else:
    if len(winner_vs_row) != 1:
        raise RuntimeError(
            "Could not find winner-vs-LightGBM comparison row."
        )

    winner_vs_row = winner_vs_row.iloc[0]

    candidate_fold_wins = int(
        winner_vs_row[
            "candidate_mae_fold_wins"
        ]
    )

    lightgbm_fold_wins = int(
        winner_vs_row[
            "lightgbm_mae_fold_wins"
        ]
    )

    mae_ties = int(
        winner_vs_row["mae_ties"]
    )


print("\nDevelopment ranking:")
print(
    summary[
        [
            "development_rank",
            "model",
            "mean_mae",
            "mean_rmse",
            "mean_r2",
            "mean_centered_spatial_r2",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print(
    f"\nDevelopment winner       : {winner}"
)
print(
    f"Winner mean MAE          : {winner_mae:.6f} °C"
)
print(
    f"LightGBM mean MAE        : {lightgbm_mae:.6f} °C"
)
print(
    f"MAE improvement          : {mae_improvement_percent:.4f}%"
)
print(
    f"Winner fold wins vs LGBM : {candidate_fold_wins}/5"
)
print(
    f"LightGBM fold wins       : {lightgbm_fold_wins}/5"
)


# =============================================================================
# 6. EVIDENCE CHARACTERIZATION
# =============================================================================

section("3. EVIDENCE CHARACTERIZATION")

mean_mae_better = bool(
    winner_mae < lightgbm_mae
)

mean_rmse_better = bool(
    winner_rmse < lightgbm_rmse
)

spatial_r2_better = bool(
    winner_spatial_r2 > lightgbm_spatial_r2
)

majority_fold_wins = bool(
    candidate_fold_wins >= 3
)

small_average_gain = bool(
    mae_improvement_percent < 1.0
)

print(f"Lower mean MAE than LightGBM       : {mean_mae_better}")
print(f"Lower mean RMSE than LightGBM      : {mean_rmse_better}")
print(f"Higher mean spatial R²             : {spatial_r2_better}")
print(f"Majority of fold MAE wins (>=3/5)  : {majority_fold_wins}")
print(f"Average MAE gain below 1%           : {small_average_gain}")

# NOTE:
# The 1% flag is descriptive only. It is NOT treated as a universal
# statistical threshold and is NOT used alone to select/reject a model.


# =============================================================================
# 7. DECISION GATE
# =============================================================================

section("4. PRODUCTION DECISION GATE")

if winner == "LightGBM":
    decision = "KEEP_CURRENT_PRODUCTION_LIGHTGBM"
    decision_reason = (
        "The existing LightGBM strategy remains the best development model."
    )

elif (
    mean_mae_better
    and majority_fold_wins
    and mean_rmse_better
    and spatial_r2_better
):
    decision = "REVIEW_CANDIDATE_FOR_FORMAL_PRODUCTION_REOPENING"
    decision_reason = (
        f"{winner} shows a broader and more consistent development advantage "
        "across mean MAE, fold wins, RMSE, and spatial R2. "
        "This still does not automatically replace the production model."
    )

else:
    decision = "KEEP_CURRENT_PRODUCTION_LIGHTGBM"
    decision_reason = (
        f"{winner} has the lowest average development MAE, but its advantage "
        f"over LightGBM is only {mae_improvement_percent:.4f}% and it wins "
        f"{candidate_fold_wins} of 5 matched MAE folds versus "
        f"{lightgbm_fold_wins} wins for LightGBM. "
        "The evidence is therefore not sufficiently consistent to justify "
        "reopening the finalized production pipeline."
    )

print(f"Decision: {decision}")
print()
print(decision_reason)

print()
print(
    "No model is retrained or replaced in Step 06."
)


# =============================================================================
# 8. SAVE DECISION
# =============================================================================

section("5. SAVE EXPERIMENT DECISION")

decision_df = pd.DataFrame(
    [
        {
            "development_winner": winner,
            "winner_mean_mae": winner_mae,
            "lightgbm_mean_mae": lightgbm_mae,
            "mae_delta_winner_minus_lightgbm": mae_delta,
            "mae_improvement_percent_vs_lightgbm": (
                mae_improvement_percent
            ),
            "winner_mean_rmse": winner_rmse,
            "lightgbm_mean_rmse": lightgbm_rmse,
            "winner_mean_centered_spatial_r2": (
                winner_spatial_r2
            ),
            "lightgbm_mean_centered_spatial_r2": (
                lightgbm_spatial_r2
            ),
            "candidate_mae_fold_wins": (
                candidate_fold_wins
            ),
            "lightgbm_mae_fold_wins": (
                lightgbm_fold_wins
            ),
            "mae_ties": mae_ties,
            "majority_fold_wins": majority_fold_wins,
            "decision": decision,
            "production_model_after_experiment": (
                "LightGBM"
                if decision
                == "KEEP_CURRENT_PRODUCTION_LIGHTGBM"
                else "UNCHANGED_PENDING_FORMAL_REVIEW"
            ),
        }
    ]
)

decision_df.to_csv(
    DECISION_CSV,
    index=False,
    encoding="utf-8",
)

summary_payload = {
    "experiment": "additional_model_comparison",
    "models_compared": [
        "LightGBM",
        "CatBoost",
        "Extra Trees",
        "HistGradientBoosting",
    ],
    "development_validation_years": [
        2019,
        2020,
        2021,
        2022,
        2023,
    ],
    "primary_metric": "mean rolling-origin MAE",
    "development_winner": winner,
    "winner_mean_mae_c": winner_mae,
    "lightgbm_mean_mae_c": lightgbm_mae,
    "mae_improvement_percent_vs_lightgbm": (
        mae_improvement_percent
    ),
    "winner_mae_fold_wins_vs_lightgbm": (
        candidate_fold_wins
    ),
    "lightgbm_mae_fold_wins_vs_winner": (
        lightgbm_fold_wins
    ),
    "decision": decision,
    "decision_reason": decision_reason,
    "production_pipeline_modified": False,
    "production_model_replaced": False,
    "2024_loaded_by_step06": False,
    "2025_loaded_by_step06": False,
    "note": (
        "This supplementary experiment does not erase the fact that "
        "2024/2025 were used elsewhere in the finalized historical pipeline; "
        "it only records that they were not loaded by this experiment step."
    ),
}

SUMMARY_JSON.write_text(
    json.dumps(
        summary_payload,
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Saved: {DECISION_CSV}")
print(f"Saved: {SUMMARY_JSON}")


# =============================================================================
# 9. AUDIT
# =============================================================================

section("6. STRICT STEP 06 AUDIT")

audit_rows = [
    {
        "check": "step05_audit",
        "status": "PASS",
        "details": "Step 05 audit loaded and fully PASS.",
    },
    {
        "check": "model_count",
        "status": (
            "PASS"
            if len(summary) == 4
            else "FAIL"
        ),
        "details": (
            f"actual={len(summary)}, expected=4"
        ),
    },
    {
        "check": "development_winner_identified",
        "status": "PASS",
        "details": winner,
    },
    {
        "check": "decision_recorded",
        "status": (
            "PASS"
            if decision in {
                "KEEP_CURRENT_PRODUCTION_LIGHTGBM",
                "REVIEW_CANDIDATE_FOR_FORMAL_PRODUCTION_REOPENING",
            }
            else "FAIL"
        ),
        "details": decision,
    },
    {
        "check": "2024_loaded_or_used_by_step06",
        "status": "PASS",
        "details": "No. Step 06 reads Step 05 summaries only.",
    },
    {
        "check": "2025_loaded_or_used_by_step06",
        "status": "PASS",
        "details": "No. Step 06 reads Step 05 summaries only.",
    },
    {
        "check": "model_retraining",
        "status": "PASS",
        "details": "No model fitting occurs in Step 06.",
    },
    {
        "check": "production_pipeline_modified",
        "status": "PASS",
        "details": (
            "Only experiments/additional_models outputs are written."
        ),
    },
    {
        "check": "production_model_replaced",
        "status": "PASS",
        "details": "No production model is replaced.",
    },
]

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
    encoding="utf-8",
)

print(
    audit_df.to_string(
        index=False
    )
)

if not (
    audit_df["status"] == "PASS"
).all():
    raise RuntimeError(
        "Step 06 audit contains one or more FAIL rows."
    )


# =============================================================================
# 10. HASHES
# =============================================================================

section("7. REPRODUCIBILITY HASHES")

hash_paths = [
    STEP05_SUMMARY,
    STEP05_VS_LIGHTGBM,
    STEP05_FOLD_MAE,
    STEP05_AUDIT,
    DECISION_CSV,
    SUMMARY_JSON,
    AUDIT_PATH,
]

hash_rows = []

for path in hash_paths:
    hash_rows.append(
        {
            "file": str(
                path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": sha256_file(path),
        }
    )

hash_df = pd.DataFrame(
    hash_rows
)

hash_df.to_csv(
    HASH_PATH,
    index=False,
    encoding="utf-8",
)

print(
    hash_df.to_string(
        index=False
    )
)

print(f"\nSaved: {HASH_PATH}")


# =============================================================================
# 11. REPORT
# =============================================================================

section("8. SAVE FINAL EXPERIMENT REPORT")

ranking_text = summary[
    [
        "development_rank",
        "model",
        "mean_mae",
        "mean_rmse",
        "mean_r2",
        "mean_centered_spatial_r2",
    ]
].to_string(
    index=False,
    float_format=lambda x: f"{x:.6f}",
)

report_lines = [
    "STEP 06 - SUPPLEMENTARY EXPERIMENT SUMMARY AND DECISION GATE",
    "",
    "DEVELOPMENT MODEL RANKING",
    ranking_text,
    "",
    f"Development winner: {winner}",
    f"Winner mean MAE: {winner_mae:.6f} C",
    f"LightGBM mean MAE: {lightgbm_mae:.6f} C",
    (
        "MAE improvement vs LightGBM: "
        f"{mae_improvement_percent:.4f}%"
    ),
    (
        "Winner MAE fold wins vs LightGBM: "
        f"{candidate_fold_wins}/5"
    ),
    (
        "LightGBM MAE fold wins vs winner: "
        f"{lightgbm_fold_wins}/5"
    ),
    "",
    f"FINAL EXPERIMENT DECISION: {decision}",
    decision_reason,
    "",
    "Production pipeline modified: NO",
    "Production model replaced: NO",
    "Backend modified: NO",
    "2024 loaded by this step: NO",
    "2025 loaded by this step: NO",
    "",
    (
        "Note: this statement refers only to the supplementary experiment. "
        "It does not claim that 2024/2025 were never used elsewhere in the "
        "previously finalized production pipeline."
    ),
    "",
    "STEP 06 STATUS: PASS",
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {REPORT_PATH}")


section("STEP 06 COMPLETED SUCCESSFULLY")

print(f"Development winner : {winner}")
print(f"Decision           : {decision}")
print("Production model   : LightGBM remains unchanged unless a separate")
print("                     formal reopening is explicitly approved.")
print("Supplementary additional-model experiment is complete.")
