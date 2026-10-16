"""
STEP 05 - DEVELOPMENT MODEL COMPARISON
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Compare the finalized development LightGBM strategy against the three new
supplementary candidates on the SAME 2019-2023 expanding rolling-origin folds:

    1. LightGBM production-reference development strategy
    2. CatBoost
    3. Extra Trees
    4. HistGradientBoosting

IMPORTANT
---------
- This is DEVELOPMENT-ONLY comparison evidence.
- 2024 reserved validation is NOT loaded.
- 2025 final lockbox is NOT loaded.
- No production model is replaced.
- No production file is modified.
- Primary ranking metric: mean rolling-origin MAE.
- Tie-breakers: mean RMSE, then mean centered spatial R².
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. SETTINGS
# =============================================================================

EXPECTED_VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]
EXPECTED_FOLDS = 5

LIGHTGBM_MODEL_LABEL = "LightGBM"
LIGHTGBM_STRATEGY = "lgbm_anomaly_plus_prior_mean_level"

MODEL_ORDER = [
    "LightGBM",
    "CatBoost",
    "Extra Trees",
    "HistGradientBoosting",
]


# =============================================================================
# 2. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]

# Existing finalized development evidence.
PRODUCTION_LEVEL_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "19b5_level_anomaly_fold_metrics.csv"
)

PRODUCTION_SPATIAL_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "19b5_spatial_anomaly_fold_metrics.csv"
)

# Supplementary experiment evidence.
CATBOOST_METRICS = (
    EXPERIMENT_ROOT
    / "outputs"
    / "metrics"
    / "02_catboost_best_fold_metrics.csv"
)

EXTRA_TREES_METRICS = (
    EXPERIMENT_ROOT
    / "outputs"
    / "metrics"
    / "03_extra_trees_best_fold_metrics.csv"
)

HISTGB_METRICS = (
    EXPERIMENT_ROOT
    / "outputs"
    / "metrics"
    / "04_hist_gradient_boosting_best_fold_metrics.csv"
)

CATBOOST_AUDIT = (
    EXPERIMENT_ROOT
    / "outputs"
    / "reports"
    / "02_catboost_experiment_audit.csv"
)

EXTRA_TREES_AUDIT = (
    EXPERIMENT_ROOT
    / "outputs"
    / "reports"
    / "03_extra_trees_experiment_audit.csv"
)

HISTGB_AUDIT = (
    EXPERIMENT_ROOT
    / "outputs"
    / "reports"
    / "04_hist_gradient_boosting_experiment_audit.csv"
)

METRIC_DIR = EXPERIMENT_ROOT / "outputs" / "metrics"
FIGURE_DIR = EXPERIMENT_ROOT / "outputs" / "figures"
REPORT_DIR = EXPERIMENT_ROOT / "outputs" / "reports"

for directory in [
    METRIC_DIR,
    FIGURE_DIR,
    REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

COMBINED_FOLD_METRICS_PATH = (
    METRIC_DIR
    / "05_all_model_fold_metrics.csv"
)

SUMMARY_PATH = (
    METRIC_DIR
    / "05_model_comparison_summary.csv"
)

VS_LIGHTGBM_PATH = (
    METRIC_DIR
    / "05_candidate_vs_lightgbm.csv"
)

FOLD_MAE_PATH = (
    METRIC_DIR
    / "05_fold_by_fold_mae_comparison.csv"
)

FIGURE_PATH = (
    FIGURE_DIR
    / "05_model_comparison_mean_mae.png"
)

AUDIT_PATH = (
    REPORT_DIR
    / "05_model_comparison_audit.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "05_model_comparison_report.txt"
)

HASH_PATH = (
    REPORT_DIR
    / "05_model_comparison_hashes.csv"
)


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 124)
    print(title)
    print("=" * 124)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def require_full_pass(audit_path: Path, label: str) -> None:
    require_file(audit_path)

    audit = pd.read_csv(audit_path)

    if "status" not in audit.columns:
        raise RuntimeError(
            f"{label} audit has no 'status' column."
        )

    if not (audit["status"].astype(str) == "PASS").all():
        raise RuntimeError(
            f"{label} audit is not fully PASS."
        )


def detect_model_column(df: pd.DataFrame) -> str | None:
    for candidate in [
        "model",
        "anomaly_model",
        "model_name",
    ]:
        if candidate in df.columns:
            return candidate
    return None


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.replace("-", "", regex=False)
    )


def load_lightgbm_reference() -> pd.DataFrame:
    """
    Load the FINALIZED DEVELOPMENT LightGBM reference from Step 19B5.

    The final-LST metrics come from:
        19b5_level_anomaly_fold_metrics.csv

    The centered spatial/anomaly R² comes from:
        19b5_spatial_anomaly_fold_metrics.csv

    Only the eligible strategy:
        lgbm_anomaly_plus_prior_mean_level
    is retained.
    """

    require_file(PRODUCTION_LEVEL_METRICS)
    require_file(PRODUCTION_SPATIAL_METRICS)

    level_df = pd.read_csv(
        PRODUCTION_LEVEL_METRICS
    )

    spatial_df = pd.read_csv(
        PRODUCTION_SPATIAL_METRICS
    )

    if "strategy" not in level_df.columns:
        raise RuntimeError(
            "Production level metrics file does not contain 'strategy'."
        )

    required_level_columns = [
        "validation_year",
        "mae",
        "rmse",
        "r2",
    ]

    missing = [
        c
        for c in required_level_columns
        if c not in level_df.columns
    ]

    if missing:
        raise RuntimeError(
            "Production level metrics missing columns: "
            + ", ".join(missing)
        )

    level_ref = level_df[
        level_df["strategy"].astype(str)
        == LIGHTGBM_STRATEGY
    ].copy()

    if len(level_ref) != EXPECTED_FOLDS:
        raise RuntimeError(
            "Expected exactly 5 LightGBM prior-mean development folds, "
            f"found {len(level_ref)}."
        )

    # Find the LightGBM rows in the spatial anomaly file.
    spatial_model_col = detect_model_column(
        spatial_df
    )

    if spatial_model_col is None:
        raise RuntimeError(
            "Could not identify model column in "
            "19b5_spatial_anomaly_fold_metrics.csv."
        )

    if "validation_year" not in spatial_df.columns:
        raise RuntimeError(
            "Spatial metrics file has no validation_year column."
        )

    # Step 19B5 spatial-metric column names can differ from the
    # generic experiment files. Detect the spatial R2 column safely.
    spatial_r2_candidates = [
        "r2",
        "anomaly_r2",
        "spatial_anomaly_r2",
        "centered_spatial_r2",
        "spatial_r2",
    ]

    spatial_r2_col = next(
        (
            c
            for c in spatial_r2_candidates
            if c in spatial_df.columns
        ),
        None,
    )

    if spatial_r2_col is None:
        raise RuntimeError(
            "Could not identify the spatial R2 column in "
            "19b5_spatial_anomaly_fold_metrics.csv. "
            f"Available columns: {spatial_df.columns.tolist()}"
        )

    print(
        f"Detected production spatial R2 column: {spatial_r2_col}"
    )

    spatial_norm = normalize_text(
        spatial_df[spatial_model_col]
    )

    spatial_ref = spatial_df[
        spatial_norm == "lightgbm"
    ].copy()

    if len(spatial_ref) != EXPECTED_FOLDS:
        raise RuntimeError(
            "Expected exactly 5 LightGBM spatial-anomaly folds, "
            f"found {len(spatial_ref)}."
        )

    spatial_ref = spatial_ref[
        [
            "validation_year",
            spatial_r2_col,
        ]
    ].rename(
        columns={
            spatial_r2_col: "centered_spatial_r2",
        }
    )

    ref = level_ref.merge(
        spatial_ref,
        on="validation_year",
        how="left",
        validate="one_to_one",
    )

    if ref["centered_spatial_r2"].isna().any():
        raise RuntimeError(
            "Failed to merge LightGBM centered spatial R²."
        )

    ref["model"] = LIGHTGBM_MODEL_LABEL
    ref["strategy"] = LIGHTGBM_STRATEGY

    # Optional columns may not exist in older production files.
    if "mape_percent" not in ref.columns:
        ref["mape_percent"] = np.nan

    if (
        "mean_prediction_bias_pred_minus_actual_c"
        not in ref.columns
    ):
        ref[
            "mean_prediction_bias_pred_minus_actual_c"
        ] = np.nan

    return ref[
        [
            "model",
            "strategy",
            "validation_year",
            "mae",
            "rmse",
            "r2",
            "mape_percent",
            "centered_spatial_r2",
            "mean_prediction_bias_pred_minus_actual_c",
        ]
    ].copy()


def load_experiment_model(
    path: Path,
    display_name: str,
) -> pd.DataFrame:

    require_file(path)

    df = pd.read_csv(path)

    required = [
        "validation_year",
        "mae",
        "rmse",
        "r2",
        "centered_spatial_r2",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{display_name} metrics missing columns: "
            + ", ".join(missing)
        )

    if len(df) != EXPECTED_FOLDS:
        raise RuntimeError(
            f"{display_name}: expected {EXPECTED_FOLDS} folds, "
            f"found {len(df)}."
        )

    out = df.copy()
    out["model"] = display_name

    if "strategy" not in out.columns:
        out["strategy"] = display_name

    if "mape_percent" not in out.columns:
        out["mape_percent"] = np.nan

    if (
        "mean_prediction_bias_pred_minus_actual_c"
        not in out.columns
    ):
        out[
            "mean_prediction_bias_pred_minus_actual_c"
        ] = np.nan

    return out[
        [
            "model",
            "strategy",
            "validation_year",
            "mae",
            "rmse",
            "r2",
            "mape_percent",
            "centered_spatial_r2",
            "mean_prediction_bias_pred_minus_actual_c",
        ]
    ].copy()


def validate_years(
    df: pd.DataFrame,
    model_name: str,
) -> None:

    years = sorted(
        pd.to_numeric(
            df["validation_year"],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    if years != EXPECTED_VALIDATION_YEARS:
        raise RuntimeError(
            f"{model_name}: validation years mismatch. "
            f"Expected {EXPECTED_VALIDATION_YEARS}, got {years}."
        )


# =============================================================================
# 4. START
# =============================================================================

section("STEP 05 - DEVELOPMENT MODEL COMPARISON")

print(f"Project root       : {PROJECT_ROOT}")
print(f"Experiment root    : {EXPERIMENT_ROOT}")
print("Comparison scope   : 2019-2023 development folds only")
print("2024 validation    : NOT LOADED")
print("2025 final lockbox : NOT LOADED")
print("Production model   : UNCHANGED")
print("Primary metric     : mean rolling-origin MAE")


# =============================================================================
# 5. VERIFY EXPERIMENT AUDITS
# =============================================================================

section("1. VERIFY PREVIOUS EXPERIMENT AUDITS")

require_full_pass(
    CATBOOST_AUDIT,
    "CatBoost",
)

require_full_pass(
    EXTRA_TREES_AUDIT,
    "Extra Trees",
)

require_full_pass(
    HISTGB_AUDIT,
    "HistGradientBoosting",
)

print("CatBoost audit            : PASS")
print("Extra Trees audit         : PASS")
print("HistGradientBoosting audit: PASS")


# =============================================================================
# 6. LOAD ALL FOUR MODELS
# =============================================================================

section("2. LOAD MATCHED DEVELOPMENT FOLD METRICS")

lightgbm_df = load_lightgbm_reference()

catboost_df = load_experiment_model(
    CATBOOST_METRICS,
    "CatBoost",
)

extra_trees_df = load_experiment_model(
    EXTRA_TREES_METRICS,
    "Extra Trees",
)

histgb_df = load_experiment_model(
    HISTGB_METRICS,
    "HistGradientBoosting",
)

for name, frame in [
    ("LightGBM", lightgbm_df),
    ("CatBoost", catboost_df),
    ("Extra Trees", extra_trees_df),
    ("HistGradientBoosting", histgb_df),
]:
    validate_years(
        frame,
        name,
    )

    print(
        f"{name:<22} "
        f"rows={len(frame)} | "
        f"years={sorted(frame['validation_year'].astype(int).tolist())}"
    )

combined = pd.concat(
    [
        lightgbm_df,
        catboost_df,
        extra_trees_df,
        histgb_df,
    ],
    ignore_index=True,
)

combined["validation_year"] = (
    combined["validation_year"]
    .astype(int)
)

numeric_cols = [
    "mae",
    "rmse",
    "r2",
    "centered_spatial_r2",
]

if not np.isfinite(
    combined[numeric_cols].to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Non-finite primary comparison metric detected."
    )

combined.to_csv(
    COMBINED_FOLD_METRICS_PATH,
    index=False,
    encoding="utf-8",
)

print(
    f"\nSaved: {COMBINED_FOLD_METRICS_PATH}"
)


# =============================================================================
# 7. SUMMARY
# =============================================================================

section("3. DEVELOPMENT MODEL RANKING")

summary = (
    combined.groupby(
        "model",
        as_index=False,
    )
    .agg(
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"),
        mean_r2=("r2", "mean"),
        mean_centered_spatial_r2=(
            "centered_spatial_r2",
            "mean",
        ),
        negative_r2_count=(
            "r2",
            lambda s: int((s < 0).sum()),
        ),
        worst_year_mae=("mae", "max"),
        minimum_r2=("r2", "min"),
    )
)

summary = (
    summary.sort_values(
        [
            "mean_mae",
            "mean_rmse",
            "mean_centered_spatial_r2",
        ],
        ascending=[
            True,
            True,
            False,
        ],
        kind="mergesort",
    )
    .reset_index(drop=True)
)

summary["development_rank"] = (
    np.arange(len(summary)) + 1
)

summary = summary[
    [
        "development_rank",
        "model",
        "mean_mae",
        "std_mae",
        "mean_rmse",
        "mean_r2",
        "mean_centered_spatial_r2",
        "negative_r2_count",
        "worst_year_mae",
        "minimum_r2",
    ]
]

summary.to_csv(
    SUMMARY_PATH,
    index=False,
    encoding="utf-8",
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

winner = str(
    summary.iloc[0]["model"]
)

winner_mae = float(
    summary.iloc[0]["mean_mae"]
)

lightgbm_summary = summary[
    summary["model"] == "LightGBM"
].iloc[0]

lightgbm_mae = float(
    lightgbm_summary["mean_mae"]
)

print(
    f"\nDevelopment winner by mean MAE: {winner}"
)


# =============================================================================
# 8. CANDIDATES VS LIGHTGBM
# =============================================================================

section("4. SUPPLEMENTARY CANDIDATES VS LIGHTGBM")

comparison_rows = []

lgbm_fold = (
    lightgbm_df[
        [
            "validation_year",
            "mae",
            "rmse",
            "r2",
            "centered_spatial_r2",
        ]
    ]
    .rename(
        columns={
            "mae": "lightgbm_mae",
            "rmse": "lightgbm_rmse",
            "r2": "lightgbm_r2",
            "centered_spatial_r2": (
                "lightgbm_centered_spatial_r2"
            ),
        }
    )
)

for candidate_name, candidate_df in [
    ("CatBoost", catboost_df),
    ("Extra Trees", extra_trees_df),
    ("HistGradientBoosting", histgb_df),
]:
    candidate_fold = candidate_df[
        [
            "validation_year",
            "mae",
            "rmse",
            "r2",
            "centered_spatial_r2",
        ]
    ].copy()

    paired = candidate_fold.merge(
        lgbm_fold,
        on="validation_year",
        how="inner",
        validate="one_to_one",
    )

    mae_delta = (
        paired["mae"]
        - paired["lightgbm_mae"]
    )

    rmse_delta = (
        paired["rmse"]
        - paired["lightgbm_rmse"]
    )

    spatial_r2_delta = (
        paired["centered_spatial_r2"]
        - paired[
            "lightgbm_centered_spatial_r2"
        ]
    )

    mean_candidate_mae = float(
        paired["mae"].mean()
    )

    mean_lgbm_mae = float(
        paired["lightgbm_mae"].mean()
    )

    improvement_percent = float(
        (
            mean_lgbm_mae
            - mean_candidate_mae
        )
        / mean_lgbm_mae
        * 100.0
    )

    comparison_rows.append(
        {
            "candidate": candidate_name,
            "candidate_mean_mae": mean_candidate_mae,
            "lightgbm_mean_mae": mean_lgbm_mae,
            "mae_delta_candidate_minus_lightgbm": float(
                mae_delta.mean()
            ),
            "mae_improvement_percent_vs_lightgbm": (
                improvement_percent
            ),
            "candidate_mae_fold_wins": int(
                (mae_delta < 0).sum()
            ),
            "lightgbm_mae_fold_wins": int(
                (mae_delta > 0).sum()
            ),
            "mae_ties": int(
                np.isclose(
                    mae_delta,
                    0.0,
                    atol=1e-12,
                ).sum()
            ),
            "mean_rmse_delta_candidate_minus_lightgbm": float(
                rmse_delta.mean()
            ),
            "mean_centered_spatial_r2_delta_candidate_minus_lightgbm": float(
                spatial_r2_delta.mean()
            ),
        }
    )

vs_lgbm = pd.DataFrame(
    comparison_rows
).sort_values(
    "candidate_mean_mae",
    ascending=True,
    kind="mergesort",
).reset_index(drop=True)

vs_lgbm.to_csv(
    VS_LIGHTGBM_PATH,
    index=False,
    encoding="utf-8",
)

print(
    vs_lgbm.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# =============================================================================
# 9. FOLD-BY-FOLD MAE TABLE
# =============================================================================

section("5. FOLD-BY-FOLD MAE COMPARISON")

fold_mae = combined.pivot(
    index="validation_year",
    columns="model",
    values="mae",
).reset_index()

ordered_columns = [
    "validation_year",
] + [
    model
    for model in MODEL_ORDER
    if model in fold_mae.columns
]

fold_mae = fold_mae[
    ordered_columns
]

fold_mae["best_model"] = (
    fold_mae[
        [
            c
            for c in ordered_columns
            if c != "validation_year"
        ]
    ]
    .idxmin(axis=1)
)

fold_mae["best_mae"] = (
    fold_mae[
        [
            c
            for c in ordered_columns
            if c != "validation_year"
        ]
    ]
    .min(axis=1)
)

fold_mae.to_csv(
    FOLD_MAE_PATH,
    index=False,
    encoding="utf-8",
)

print(
    fold_mae.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\nFold-win counts:")
print(
    fold_mae["best_model"]
    .value_counts()
    .to_string()
)


# =============================================================================
# 10. FIGURE
# =============================================================================

section("6. FIGURE")

plot_df = summary.sort_values(
    "mean_mae",
    ascending=True,
)

fig, ax = plt.subplots(
    figsize=(8.5, 5.2)
)

ax.bar(
    plot_df["model"],
    plot_df["mean_mae"],
)

ax.set_ylabel(
    "Mean rolling-origin MAE (°C)"
)

ax.set_xlabel(
    "Model"
)

ax.set_title(
    "Development Model Comparison (2019-2023)"
)

ax.tick_params(
    axis="x",
    rotation=20,
)

ax.grid(
    axis="y",
    alpha=0.25,
)

for index, value in enumerate(
    plot_df["mean_mae"].to_numpy(dtype=float)
):
    ax.text(
        index,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom",
    )

fig.tight_layout()

fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved: {FIGURE_PATH}")


# =============================================================================
# 11. INTERPRETATION
# =============================================================================

section("7. DEVELOPMENT INTERPRETATION")

winner_vs_lgbm_improvement_percent = float(
    (
        lightgbm_mae
        - winner_mae
    )
    / lightgbm_mae
    * 100.0
)

winner_vs_lgbm_delta = float(
    winner_mae
    - lightgbm_mae
)

if winner == "LightGBM":
    development_message = (
        "LightGBM remains the strongest model on the matched "
        "2019-2023 development folds."
    )
else:
    development_message = (
        f"{winner} has the lowest development mean MAE, but the "
        "production model is NOT changed by this comparison."
    )

print(development_message)

print(
    f"Winner mean MAE              : {winner_mae:.6f} °C"
)

print(
    f"LightGBM mean MAE            : {lightgbm_mae:.6f} °C"
)

print(
    f"Winner - LightGBM MAE delta  : "
    f"{winner_vs_lgbm_delta:+.6f} °C"
)

print(
    f"MAE improvement vs LightGBM  : "
    f"{winner_vs_lgbm_improvement_percent:.4f}%"
)

print()
print(
    "IMPORTANT: this is development evidence only."
)

print(
    "2024 reserved validation has NOT been opened."
)

print(
    "2025 final lockbox has NOT been opened."
)

print(
    "No production model replacement decision is made in Step 05."
)


# =============================================================================
# 12. AUDIT
# =============================================================================

section("8. STRICT STEP 05 AUDIT")

expected_combined_rows = (
    len(MODEL_ORDER)
    * EXPECTED_FOLDS
)

audit_rows = [
    {
        "check": "model_count",
        "status": (
            "PASS"
            if combined["model"].nunique() == 4
            else "FAIL"
        ),
        "details": (
            f"actual={combined['model'].nunique()}, expected=4"
        ),
    },
    {
        "check": "combined_fold_rows",
        "status": (
            "PASS"
            if len(combined) == expected_combined_rows
            else "FAIL"
        ),
        "details": (
            f"actual={len(combined)}, "
            f"expected={expected_combined_rows}"
        ),
    },
    {
        "check": "validation_years",
        "status": (
            "PASS"
            if sorted(
                combined["validation_year"]
                .unique()
                .tolist()
            )
            == EXPECTED_VALIDATION_YEARS
            else "FAIL"
        ),
        "details": str(
            sorted(
                combined["validation_year"]
                .unique()
                .tolist()
            )
        ),
    },
    {
        "check": "each_model_has_five_folds",
        "status": (
            "PASS"
            if (
                combined.groupby("model")
                .size()
                .eq(EXPECTED_FOLDS)
                .all()
            )
            else "FAIL"
        ),
        "details": str(
            combined.groupby("model")
            .size()
            .to_dict()
        ),
    },
    {
        "check": "primary_metrics_finite",
        "status": (
            "PASS"
            if np.isfinite(
                combined[
                    [
                        "mae",
                        "rmse",
                        "r2",
                        "centered_spatial_r2",
                    ]
                ].to_numpy(dtype=float)
            ).all()
            else "FAIL"
        ),
        "details": "MAE/RMSE/R2/spatial R2 are finite.",
    },
    {
        "check": "catboost_audit",
        "status": "PASS",
        "details": "Step 02 audit fully PASS.",
    },
    {
        "check": "extra_trees_audit",
        "status": "PASS",
        "details": "Step 03 audit fully PASS.",
    },
    {
        "check": "histgb_audit",
        "status": "PASS",
        "details": "Step 04 audit fully PASS.",
    },
    {
        "check": "2024_loaded_or_used",
        "status": "PASS",
        "details": (
            "No 2024 data is loaded by Step 05."
        ),
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": (
            "No 2025 data is loaded by Step 05."
        ),
    },
    {
        "check": "production_pipeline_modified",
        "status": "PASS",
        "details": (
            "Step 05 writes only under experiments/additional_models."
        ),
    },
    {
        "check": "production_model_replaced",
        "status": "PASS",
        "details": (
            "No production model replacement occurs in Step 05."
        ),
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
        "Step 05 audit contains one or more FAIL rows."
    )


# =============================================================================
# 13. HASHES
# =============================================================================

section("9. REPRODUCIBILITY HASHES")

hash_paths = [
    PRODUCTION_LEVEL_METRICS,
    PRODUCTION_SPATIAL_METRICS,
    CATBOOST_METRICS,
    EXTRA_TREES_METRICS,
    HISTGB_METRICS,
    COMBINED_FOLD_METRICS_PATH,
    SUMMARY_PATH,
    VS_LIGHTGBM_PATH,
    FOLD_MAE_PATH,
    FIGURE_PATH,
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
# 14. REPORT
# =============================================================================

section("10. SAVE STEP 05 REPORT")

summary_text = summary.to_string(
    index=False,
    float_format=lambda x: f"{x:.6f}",
)

vs_text = vs_lgbm.to_string(
    index=False,
    float_format=lambda x: f"{x:.6f}",
)

report_lines = [
    "STEP 05 - DEVELOPMENT MODEL COMPARISON",
    "",
    "Scope: matched 2019-2023 expanding rolling-origin folds",
    "2024 reserved validation: NOT LOADED",
    "2025 final lockbox: NOT LOADED",
    "Production model: UNCHANGED",
    "",
    "Primary ranking metric: mean rolling-origin MAE",
    "Tie-breakers: mean RMSE, then mean centered spatial R2",
    "",
    "MODEL RANKING",
    summary_text,
    "",
    "CANDIDATES VS LIGHTGBM",
    vs_text,
    "",
    f"Development winner: {winner}",
    f"Winner mean MAE: {winner_mae:.6f} C",
    f"LightGBM mean MAE: {lightgbm_mae:.6f} C",
    (
        "Winner - LightGBM MAE delta: "
        f"{winner_vs_lgbm_delta:+.6f} C"
    ),
    (
        "MAE improvement vs LightGBM: "
        f"{winner_vs_lgbm_improvement_percent:.4f}%"
    ),
    "",
    development_message,
    "",
    (
        "IMPORTANT: This is supplementary development evidence only. "
        "No production-model replacement decision is made here."
    ),
    "",
    "STEP 05 STATUS: PASS",
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {REPORT_PATH}")


section("STEP 05 COMPLETED SUCCESSFULLY")

print(f"Development winner: {winner}")
print("2024 reserved validation remains untouched.")
print("2025 final lockbox remains untouched.")
print("Production model remains unchanged.")
print("Next: Step 06 - experiment summary / decision gate.")
