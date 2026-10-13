
"""
STEP 16 - UNTUNED CANDIDATE MODEL COMPARISON
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Formally compare the COMPLETED UNTUNED candidate models using the same
2018-2023 rolling-origin development evidence.

Models compared
---------------
1. Random Forest
2. XGBoost
3. LightGBM
4. NGBoost

Reference baselines from Step 11 are also summarized.

STRICT PROTECTION
-----------------
- This script does NOT load 2024 validation data.
- This script does NOT load 2025 final lockbox data.
- This script does NOT train or tune any model.
- This script does NOT select the final model.
- Step 17 remains the hyperparameter-tuning stage.

Primary development comparison metric
-------------------------------------
MAE is treated as the primary comparison metric because it is directly
interpretable in degrees Celsius.

Supporting metrics
------------------
RMSE, R², MAPE, train-vs-validation generalization gap, negative-R² fold count,
year-wise MAE rank, and temporal stability.

Important
---------
Any ordering/ranking generated here is DEVELOPMENT-STAGE evidence only.
It is NOT the final model choice.
"""

from pathlib import Path
import hashlib
import itertools
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. SETTINGS
# =============================================================================

EXPECTED_YEARS = [2019, 2020, 2021, 2022, 2023]

CANDIDATE_FILES = {
    "Random Forest": "12_random_forest_rolling_origin_metrics.csv",
    "XGBoost": "13_xgboost_rolling_origin_metrics.csv",
    "LightGBM": "14_lightgbm_rolling_origin_metrics.csv",
    "NGBoost": "15_ngboost_rolling_origin_metrics.csv",
}

BASELINE_MODEL_NAMES = [
    "Global Historical Mean",
    "Grid Historical Mean",
    "Last-Year Persistence",
    "3-Year Grid Moving Average",
    "Linear Regression",
]


# =============================================================================
# 2. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

for d in [METRICS_DIR, REPORT_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BASELINE_PATH = METRICS_DIR / "11_baseline_rolling_origin_metrics.csv"
NGB_PROB_PATH = METRICS_DIR / "15_ngboost_probabilistic_metrics.csv"

CANDIDATE_PATHS = {
    name: METRICS_DIR / filename
    for name, filename in CANDIDATE_FILES.items()
}

BY_YEAR_PATH = METRICS_DIR / "16_candidate_comparison_by_year.csv"
SUMMARY_PATH = METRICS_DIR / "16_candidate_model_summary.csv"
METRIC_RANKS_PATH = METRICS_DIR / "16_candidate_metric_ranks.csv"
PAIRWISE_PATH = METRICS_DIR / "16_candidate_pairwise_mae_comparison.csv"
BASELINE_SUMMARY_PATH = METRICS_DIR / "16_baseline_summary.csv"
CANDIDATE_VS_BASELINES_PATH = METRICS_DIR / "16_candidate_vs_fixed_baselines.csv"
YEAR_DIFFICULTY_PATH = METRICS_DIR / "16_year_difficulty_summary.csv"
NGB_UNCERTAINTY_PATH = METRICS_DIR / "16_ngboost_uncertainty_reference.csv"

AUDIT_PATH = REPORT_DIR / "16_candidate_comparison_audit.csv"
HASH_PATH = REPORT_DIR / "16_candidate_comparison_file_hashes.csv"
REPORT_PATH = REPORT_DIR / "16_candidate_model_comparison_report.txt"

MAE_BY_YEAR_FIG = FIGURE_DIR / "16_candidate_mae_by_year.png"
R2_BY_YEAR_FIG = FIGURE_DIR / "16_candidate_r2_by_year.png"
MEAN_MAE_FIG = FIGURE_DIR / "16_mean_mae_candidates_and_baselines.png"
GAP_FIG = FIGURE_DIR / "16_candidate_generalization_gap.png"
NGB_COVERAGE_FIG = FIGURE_DIR / "16_ngboost_nominal_vs_observed_coverage.png"


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


def require_columns(df: pd.DataFrame, columns, label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        print(f"\nERROR: {label} missing required columns: {missing}")
        sys.exit(1)


# =============================================================================
# 4. LOAD STEP 11-15 METRIC OUTPUTS ONLY
# =============================================================================

section("STEP 16 - UNTUNED CANDIDATE MODEL COMPARISON")

print(f"Project root : {PROJECT_ROOT}")
print("2024 data    : NOT LOADED")
print("2025 data    : NOT LOADED")
print("Training     : NONE")
print("Tuning       : NONE")
print("Final model  : NOT selected")

required_paths = [
    BASELINE_PATH,
    NGB_PROB_PATH,
    *CANDIDATE_PATHS.values(),
]

missing_paths = [
    str(path)
    for path in required_paths
    if not path.exists()
]

if missing_paths:
    print("\nERROR: required prior-step outputs are missing:")
    for item in missing_paths:
        print(f" - {item}")
    sys.exit(1)

baseline_df = pd.read_csv(BASELINE_PATH)
ngb_prob_df = pd.read_csv(NGB_PROB_PATH)

candidate_frames = []

candidate_required = [
    "validation_year",
    "train_mae",
    "validation_mae",
    "mae_generalization_gap",
    "train_rmse",
    "validation_rmse",
    "rmse_generalization_gap",
    "train_r2",
    "validation_r2",
    "r2_generalization_gap",
    "train_mape_percent",
    "validation_mape_percent",
]

for model_name, path in CANDIDATE_PATHS.items():
    model_df = pd.read_csv(path)

    require_columns(model_df, candidate_required, model_name)

    years = sorted(
        model_df["validation_year"].astype(int).unique().tolist()
    )

    if years != EXPECTED_YEARS:
        print(
            f"\nERROR: {model_name} validation years {years} "
            f"do not match expected {EXPECTED_YEARS}."
        )
        sys.exit(1)

    if len(model_df) != len(EXPECTED_YEARS):
        print(
            f"\nERROR: {model_name} has {len(model_df)} rows; "
            f"expected {len(EXPECTED_YEARS)}."
        )
        sys.exit(1)

    if model_df["validation_year"].duplicated().any():
        print(f"\nERROR: duplicate validation years in {model_name} metrics.")
        sys.exit(1)

    model_df = model_df.copy()
    model_df["model"] = model_name
    candidate_frames.append(model_df)

candidates = pd.concat(candidate_frames, ignore_index=True)

print(f"Candidate models : {len(CANDIDATE_FILES)}")
print(f"Candidate rows   : {len(candidates)}")
print(f"Validation years : {EXPECTED_YEARS}")


# =============================================================================
# 5. BASELINE AUDIT
# =============================================================================

section("1. STEP 11 BASELINE AUDIT")

baseline_required = [
    "validation_year",
    "model",
    "mae",
    "rmse",
    "r2",
    "mape_percent",
]

require_columns(baseline_df, baseline_required, "Step 11 baseline metrics")

baseline_years = sorted(
    baseline_df["validation_year"].astype(int).unique().tolist()
)

if baseline_years != EXPECTED_YEARS:
    print(
        f"\nERROR: Step 11 baseline years {baseline_years} "
        f"do not match expected {EXPECTED_YEARS}."
    )
    sys.exit(1)

actual_baselines = sorted(
    baseline_df["model"].astype(str).unique().tolist()
)

missing_baselines = sorted(
    set(BASELINE_MODEL_NAMES) - set(actual_baselines)
)

if missing_baselines:
    print(f"\nERROR: expected baseline model(s) missing: {missing_baselines}")
    sys.exit(1)

for model_name in BASELINE_MODEL_NAMES:
    count = int((baseline_df["model"] == model_name).sum())
    if count != len(EXPECTED_YEARS):
        print(
            f"\nERROR: baseline '{model_name}' has {count} rows; "
            f"expected {len(EXPECTED_YEARS)}."
        )
        sys.exit(1)

print(f"Baseline rows   : {len(baseline_df)}")
print(f"Baseline models : {len(actual_baselines)}")
print(f"Baseline years  : {baseline_years}")
print("PASS: Step 11 baselines align with candidate rolling years.")


# =============================================================================
# 6. YEAR-BY-YEAR CANDIDATE COMPARISON
# =============================================================================

section("2. YEAR-BY-YEAR UNTUNED CANDIDATE COMPARISON")

by_year = candidates[
    [
        "model",
        "validation_year",
        "train_mae",
        "validation_mae",
        "mae_generalization_gap",
        "validation_rmse",
        "validation_r2",
        "validation_mape_percent",
    ]
].copy()

by_year["mae_rank_within_year"] = (
    by_year.groupby("validation_year")["validation_mae"]
    .rank(method="min", ascending=True)
)

by_year["rmse_rank_within_year"] = (
    by_year.groupby("validation_year")["validation_rmse"]
    .rank(method="min", ascending=True)
)

by_year["r2_rank_within_year"] = (
    by_year.groupby("validation_year")["validation_r2"]
    .rank(method="min", ascending=False)
)

by_year = by_year.sort_values(
    ["validation_year", "mae_rank_within_year", "model"]
).reset_index(drop=True)

by_year.to_csv(BY_YEAR_PATH, index=False)

print(
    by_year[
        [
            "validation_year",
            "model",
            "validation_mae",
            "validation_rmse",
            "validation_r2",
            "mae_generalization_gap",
            "mae_rank_within_year",
        ]
    ].to_string(index=False)
)


# =============================================================================
# 7. CANDIDATE SUMMARY
# =============================================================================

section("3. CANDIDATE DEVELOPMENT SUMMARY")

summary = (
    candidates.groupby("model")
    .agg(
        folds=("validation_year", "count"),
        mean_train_mae=("train_mae", "mean"),
        mean_validation_mae=("validation_mae", "mean"),
        median_validation_mae=("validation_mae", "median"),
        std_validation_mae=("validation_mae", "std"),
        best_year_mae=("validation_mae", "min"),
        worst_year_mae=("validation_mae", "max"),
        mean_validation_rmse=("validation_rmse", "mean"),
        median_validation_rmse=("validation_rmse", "median"),
        mean_validation_r2=("validation_r2", "mean"),
        median_validation_r2=("validation_r2", "median"),
        minimum_validation_r2=("validation_r2", "min"),
        mean_validation_mape_percent=("validation_mape_percent", "mean"),
        mean_mae_generalization_gap=("mae_generalization_gap", "mean"),
        mean_rmse_generalization_gap=("rmse_generalization_gap", "mean"),
        mean_r2_generalization_gap=("r2_generalization_gap", "mean"),
    )
    .reset_index()
)

negative_counts = (
    candidates.assign(
        negative_r2=(candidates["validation_r2"] < 0).astype(int)
    )
    .groupby("model")["negative_r2"]
    .sum()
    .rename("negative_r2_folds")
    .reset_index()
)

year_wins = (
    by_year[by_year["mae_rank_within_year"] == 1]
    .groupby("model")
    .size()
    .rename("mae_year_wins")
    .reset_index()
)

mean_rank = (
    by_year.groupby("model")["mae_rank_within_year"]
    .mean()
    .rename("mean_mae_rank")
    .reset_index()
)

summary = (
    summary
    .merge(negative_counts, on="model", how="left")
    .merge(year_wins, on="model", how="left")
    .merge(mean_rank, on="model", how="left")
)

summary["mae_year_wins"] = summary["mae_year_wins"].fillna(0).astype(int)

summary["development_mean_mae_rank"] = (
    summary["mean_validation_mae"]
    .rank(method="min", ascending=True)
    .astype(int)
)

summary["development_mean_rmse_rank"] = (
    summary["mean_validation_rmse"]
    .rank(method="min", ascending=True)
    .astype(int)
)

summary["development_mean_r2_rank"] = (
    summary["mean_validation_r2"]
    .rank(method="min", ascending=False)
    .astype(int)
)

summary = summary.sort_values(
    ["development_mean_mae_rank", "model"]
).reset_index(drop=True)

summary.to_csv(SUMMARY_PATH, index=False)

print(
    summary[
        [
            "model",
            "mean_validation_mae",
            "std_validation_mae",
            "worst_year_mae",
            "mean_validation_rmse",
            "mean_validation_r2",
            "negative_r2_folds",
            "mean_mae_generalization_gap",
            "mae_year_wins",
            "mean_mae_rank",
            "development_mean_mae_rank",
        ]
    ].to_string(index=False)
)

print(
    "\nIMPORTANT: this is an UNTUNED development comparison only. "
    "No final model is selected in Step 16."
)


# =============================================================================
# 8. SEPARATE METRIC RANKS - NO COMPOSITE SCORE
# =============================================================================

section("4. SEPARATE DEVELOPMENT METRIC RANKS")

metric_ranks = summary[
    [
        "model",
        "development_mean_mae_rank",
        "development_mean_rmse_rank",
        "development_mean_r2_rank",
        "mean_mae_rank",
        "mae_year_wins",
        "negative_r2_folds",
    ]
].copy()

metric_ranks.to_csv(METRIC_RANKS_PATH, index=False)

print(metric_ranks.to_string(index=False))

print(
    "\nNo artificial composite score is created. "
    "Metrics are kept separate so later tuning and validation can change "
    "the evidence fairly."
)


# =============================================================================
# 9. PAIRWISE CANDIDATE MAE COMPARISON
# =============================================================================

section("5. PAIRED FOLD-LEVEL CANDIDATE MAE COMPARISON")

pairwise_rows = []
candidate_names = list(CANDIDATE_FILES.keys())

mae_pivot = candidates.pivot(
    index="validation_year",
    columns="model",
    values="validation_mae",
).sort_index()

for model_a, model_b in itertools.combinations(candidate_names, 2):
    diff = mae_pivot[model_a] - mae_pivot[model_b]

    pairwise_rows.append(
        {
            "model_a": model_a,
            "model_b": model_b,
            "mean_mae_a_minus_b": float(diff.mean()),
            "median_mae_a_minus_b": float(diff.median()),
            "folds_a_lower_mae": int((diff < 0).sum()),
            "folds_b_lower_mae": int((diff > 0).sum()),
            "ties": int((diff == 0).sum()),
            "n_temporal_folds": int(len(diff)),
        }
    )

pairwise_df = pd.DataFrame(pairwise_rows)
pairwise_df.to_csv(PAIRWISE_PATH, index=False)

print(pairwise_df.to_string(index=False))

print(
    "\nNo significance claim is made from only five temporal folds. "
    "These paired differences are descriptive development evidence."
)


# =============================================================================
# 10. BASELINE SUMMARY
# =============================================================================

section("6. FIXED BASELINE SUMMARY")

baseline_filtered = baseline_df[
    baseline_df["model"].isin(BASELINE_MODEL_NAMES)
].copy()

baseline_summary = (
    baseline_filtered.groupby("model")
    .agg(
        folds=("validation_year", "count"),
        mean_mae=("mae", "mean"),
        median_mae=("mae", "median"),
        std_mae=("mae", "std"),
        worst_year_mae=("mae", "max"),
        mean_rmse=("rmse", "mean"),
        mean_r2=("r2", "mean"),
        mean_mape_percent=("mape_percent", "mean"),
    )
    .reset_index()
)

baseline_negative = (
    baseline_filtered.assign(
        negative_r2=(baseline_filtered["r2"] < 0).astype(int)
    )
    .groupby("model")["negative_r2"]
    .sum()
    .rename("negative_r2_folds")
    .reset_index()
)

baseline_summary = (
    baseline_summary
    .merge(baseline_negative, on="model", how="left")
    .sort_values(["mean_mae", "model"])
    .reset_index(drop=True)
)

baseline_summary["development_mean_mae_rank"] = np.arange(
    1,
    len(baseline_summary) + 1,
)

baseline_summary.to_csv(BASELINE_SUMMARY_PATH, index=False)

print(baseline_summary.to_string(index=False))


# =============================================================================
# 11. CANDIDATES VS FIXED BASELINES
# =============================================================================

section("7. CANDIDATE VS FIXED BASELINES")

candidate_vs_baseline_rows = []

for _, candidate_row in summary.iterrows():
    candidate_name = candidate_row["model"]
    candidate_mean_mae = float(candidate_row["mean_validation_mae"])

    for _, baseline_row in baseline_summary.iterrows():
        baseline_name = baseline_row["model"]
        baseline_mean_mae = float(baseline_row["mean_mae"])

        candidate_vs_baseline_rows.append(
            {
                "candidate_model": candidate_name,
                "baseline_model": baseline_name,
                "candidate_mean_mae": candidate_mean_mae,
                "baseline_mean_mae": baseline_mean_mae,
                "candidate_minus_baseline_mean_mae": (
                    candidate_mean_mae - baseline_mean_mae
                ),
                "candidate_lower_mean_mae": (
                    candidate_mean_mae < baseline_mean_mae
                ),
            }
        )

candidate_vs_baselines = pd.DataFrame(candidate_vs_baseline_rows)
candidate_vs_baselines.to_csv(CANDIDATE_VS_BASELINES_PATH, index=False)

print(candidate_vs_baselines.to_string(index=False))


# =============================================================================
# 12. TEMPORAL DIFFICULTY SUMMARY
# =============================================================================

section("8. COMMON TEMPORAL DIFFICULTY ACROSS CANDIDATES")

year_difficulty = (
    candidates.groupby("validation_year")
    .agg(
        mean_candidate_mae=("validation_mae", "mean"),
        median_candidate_mae=("validation_mae", "median"),
        std_candidate_mae=("validation_mae", "std"),
        best_candidate_mae=("validation_mae", "min"),
        worst_candidate_mae=("validation_mae", "max"),
        mean_candidate_rmse=("validation_rmse", "mean"),
        mean_candidate_r2=("validation_r2", "mean"),
    )
    .reset_index()
)

year_difficulty["difficulty_rank_by_mean_mae"] = (
    year_difficulty["mean_candidate_mae"]
    .rank(method="min", ascending=False)
    .astype(int)
)

year_difficulty = year_difficulty.sort_values(
    ["difficulty_rank_by_mean_mae", "validation_year"]
).reset_index(drop=True)

year_difficulty.to_csv(YEAR_DIFFICULTY_PATH, index=False)

print(year_difficulty.to_string(index=False))

hardest_year = int(year_difficulty.iloc[0]["validation_year"])

print(
    f"\nHighest mean candidate MAE occurs in {hardest_year}. "
    "This is a flag for later temporal-validation/shift analysis, "
    "not a coding error."
)


# =============================================================================
# 13. NGBOOST UNCERTAINTY REFERENCE
# =============================================================================

section("9. NGBOOST RAW UNCERTAINTY REFERENCE")

ngb_prob_required = [
    "validation_year",
    "validation_nll",
    "validation_mean_predicted_std",
    "validation_coverage_80",
    "validation_coverage_90",
    "validation_coverage_95",
    "validation_mean_width_80",
    "validation_mean_width_90",
    "validation_mean_width_95",
]

require_columns(
    ngb_prob_df,
    ngb_prob_required,
    "NGBoost probabilistic metrics",
)

ngb_prob_years = sorted(
    ngb_prob_df["validation_year"].astype(int).unique().tolist()
)

if ngb_prob_years != EXPECTED_YEARS:
    print(
        f"\nERROR: NGBoost probabilistic years {ngb_prob_years} "
        f"do not match expected {EXPECTED_YEARS}."
    )
    sys.exit(1)

coverage_rows = []

for pct in [80, 90, 95]:
    nominal = pct / 100.0
    observed_mean = float(
        ngb_prob_df[f"validation_coverage_{pct}"].mean()
    )

    coverage_rows.append(
        {
            "nominal_interval_percent": pct,
            "nominal_coverage": nominal,
            "mean_observed_coverage": observed_mean,
            "coverage_gap_observed_minus_nominal": (
                observed_mean - nominal
            ),
            "mean_interval_width_C": float(
                ngb_prob_df[f"validation_mean_width_{pct}"].mean()
            ),
        }
    )

ngb_uncertainty = pd.DataFrame(coverage_rows)

ngb_uncertainty["mean_validation_nll"] = float(
    ngb_prob_df["validation_nll"].mean()
)

ngb_uncertainty["mean_validation_predicted_std_C"] = float(
    ngb_prob_df["validation_mean_predicted_std"].mean()
)

ngb_uncertainty.to_csv(NGB_UNCERTAINTY_PATH, index=False)

print(ngb_uncertainty.to_string(index=False))

print(
    "\nThis raw NGBoost uncertainty is diagnostic only. "
    "It is not automatically the final dashboard uncertainty method."
)


# =============================================================================
# 14. FIGURES
# =============================================================================

section("10. FIGURES")

fig, ax = plt.subplots(figsize=(10, 6))

for model_name in candidate_names:
    model_data = (
        candidates[candidates["model"] == model_name]
        .sort_values("validation_year")
    )

    ax.plot(
        model_data["validation_year"].astype(str),
        model_data["validation_mae"],
        marker="o",
        label=model_name,
    )

grid_baseline = (
    baseline_df[
        baseline_df["model"] == "Grid Historical Mean"
    ]
    .sort_values("validation_year")
)

ax.plot(
    grid_baseline["validation_year"].astype(str),
    grid_baseline["mae"],
    marker="o",
    linestyle="--",
    label="Grid Historical Mean",
)

ax.set_title("Untuned Candidate MAE by Rolling Validation Year")
ax.set_xlabel("Validation Year")
ax.set_ylabel("MAE (°C)")
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()
plt.savefig(MAE_BY_YEAR_FIG, dpi=300, bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 6))

for model_name in candidate_names:
    model_data = (
        candidates[candidates["model"] == model_name]
        .sort_values("validation_year")
    )

    ax.plot(
        model_data["validation_year"].astype(str),
        model_data["validation_r2"],
        marker="o",
        label=model_name,
    )

ax.axhline(0, linewidth=1)

ax.set_title("Untuned Candidate R² by Rolling Validation Year")
ax.set_xlabel("Validation Year")
ax.set_ylabel("R²")
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()
plt.savefig(R2_BY_YEAR_FIG, dpi=300, bbox_inches="tight")
plt.close(fig)


all_mean_mae = pd.concat(
    [
        summary[
            ["model", "mean_validation_mae"]
        ].rename(
            columns={"mean_validation_mae": "mean_mae"}
        ),
        baseline_summary[["model", "mean_mae"]],
    ],
    ignore_index=True,
)

all_mean_mae = all_mean_mae.sort_values(
    ["mean_mae", "model"],
    ascending=[True, True],
)

fig, ax = plt.subplots(figsize=(11, 7))

ax.barh(
    all_mean_mae["model"],
    all_mean_mae["mean_mae"],
)

ax.invert_yaxis()
ax.set_title("Development Mean MAE - Candidates and Fixed Baselines")
ax.set_xlabel("Mean Rolling Validation MAE (°C)")
ax.set_ylabel("Model")
ax.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig(MEAN_MAE_FIG, dpi=300, bbox_inches="tight")
plt.close(fig)


gap_plot = summary.sort_values(
    "mean_mae_generalization_gap",
    ascending=True,
)

fig, ax = plt.subplots(figsize=(9, 6))

ax.barh(
    gap_plot["model"],
    gap_plot["mean_mae_generalization_gap"],
)

ax.set_title("Untuned Candidate Mean MAE Generalization Gap")
ax.set_xlabel("Validation MAE - Train MAE (°C)")
ax.set_ylabel("Candidate Model")
ax.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig(GAP_FIG, dpi=300, bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(ngb_uncertainty))
width = 0.35

ax.bar(
    x - width / 2,
    ngb_uncertainty["nominal_coverage"] * 100.0,
    width,
    label="Nominal",
)

ax.bar(
    x + width / 2,
    ngb_uncertainty["mean_observed_coverage"] * 100.0,
    width,
    label="Observed",
)

ax.set_xticks(
    x,
    [
        f"{pct}%"
        for pct in ngb_uncertainty["nominal_interval_percent"]
    ],
)

ax.set_ylim(0, 100)
ax.set_title("NGBoost Raw Prediction Interval Coverage")
ax.set_xlabel("Nominal Interval")
ax.set_ylabel("Coverage (%)")
ax.grid(axis="y", alpha=0.2)
ax.legend()

plt.tight_layout()
plt.savefig(NGB_COVERAGE_FIG, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {MAE_BY_YEAR_FIG}")
print(f"Saved: {R2_BY_YEAR_FIG}")
print(f"Saved: {MEAN_MAE_FIG}")
print(f"Saved: {GAP_FIG}")
print(f"Saved: {NGB_COVERAGE_FIG}")


# =============================================================================
# 15. STRICT AUDIT
# =============================================================================

section("11. STRICT STEP 16 AUDIT")

candidate_years_all = sorted(
    candidates["validation_year"].astype(int).unique().tolist()
)

candidate_models_all = sorted(
    candidates["model"].unique().tolist()
)

all_candidate_metrics_finite = bool(
    np.isfinite(
        candidates[
            [
                "train_mae",
                "validation_mae",
                "validation_rmse",
                "validation_r2",
                "validation_mape_percent",
                "mae_generalization_gap",
            ]
        ].to_numpy(dtype=float)
    ).all()
)

audit_rows = [
    {
        "check": "candidate_models_complete",
        "status": (
            "PASS"
            if candidate_models_all == sorted(CANDIDATE_FILES.keys())
            else "FAIL"
        ),
        "details": str(candidate_models_all),
    },
    {
        "check": "candidate_validation_years",
        "status": (
            "PASS"
            if candidate_years_all == EXPECTED_YEARS
            else "FAIL"
        ),
        "details": str(candidate_years_all),
    },
    {
        "check": "candidate_row_count",
        "status": (
            "PASS"
            if len(candidates)
            == len(CANDIDATE_FILES) * len(EXPECTED_YEARS)
            else "FAIL"
        ),
        "details": (
            f"actual={len(candidates)}, "
            f"expected={len(CANDIDATE_FILES) * len(EXPECTED_YEARS)}"
        ),
    },
    {
        "check": "baseline_validation_years",
        "status": (
            "PASS"
            if baseline_years == EXPECTED_YEARS
            else "FAIL"
        ),
        "details": str(baseline_years),
    },
    {
        "check": "2024_loaded_or_used",
        "status": "PASS",
        "details": "No. Step 16 loads only Step 11-15 development metric files.",
    },
    {
        "check": "2025_loaded_or_used",
        "status": "PASS",
        "details": "No. Step 16 loads only Step 11-15 development metric files.",
    },
    {
        "check": "model_training_performed",
        "status": "PASS",
        "details": "No model training occurs in Step 16.",
    },
    {
        "check": "hyperparameter_tuning_performed",
        "status": "PASS",
        "details": "No tuning occurs in Step 16.",
    },
    {
        "check": "all_candidate_metrics_finite",
        "status": (
            "PASS"
            if all_candidate_metrics_finite
            else "FAIL"
        ),
        "details": str(all_candidate_metrics_finite),
    },
    {
        "check": "summary_candidate_count",
        "status": (
            "PASS"
            if len(summary) == len(CANDIDATE_FILES)
            else "FAIL"
        ),
        "details": f"rows={len(summary)}",
    },
    {
        "check": "pairwise_comparisons_complete",
        "status": (
            "PASS"
            if len(pairwise_df) == 6
            else "FAIL"
        ),
        "details": f"rows={len(pairwise_df)}, expected=6",
    },
    {
        "check": "ngboost_uncertainty_reference_complete",
        "status": (
            "PASS"
            if len(ngb_uncertainty) == 3
            else "FAIL"
        ),
        "details": f"rows={len(ngb_uncertainty)}",
    },
    {
        "check": "composite_final_score_created",
        "status": "PASS",
        "details": "No. Metrics are kept separate.",
    },
    {
        "check": "final_model_selected",
        "status": "PASS",
        "details": "No. Step 16 provides development comparison only.",
    },
]

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(AUDIT_PATH, index=False)

print(audit_df.to_string(index=False))

if (audit_df["status"] != "PASS").any():
    print("\nERROR: one or more Step 16 audit checks failed.")
    sys.exit(1)


# =============================================================================
# 16. REPRODUCIBILITY HASHES
# =============================================================================

section("12. REPRODUCIBILITY HASHES")

hash_files = [
    BASELINE_PATH,
    *CANDIDATE_PATHS.values(),
    NGB_PROB_PATH,
    BY_YEAR_PATH,
    SUMMARY_PATH,
    METRIC_RANKS_PATH,
    PAIRWISE_PATH,
    BASELINE_SUMMARY_PATH,
    CANDIDATE_VS_BASELINES_PATH,
    YEAR_DIFFICULTY_PATH,
    NGB_UNCERTAINTY_PATH,
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

pd.DataFrame(hash_rows).to_csv(HASH_PATH, index=False)


# =============================================================================
# 17. REPORT
# =============================================================================

section("13. SAVE STEP 16 REPORT")

summary_for_report = summary.sort_values(
    "mean_validation_mae",
    ascending=True,
)

report_lines = [
    "STEP 16 - UNTUNED CANDIDATE MODEL COMPARISON",
    "=" * 108,
    "",
    "SCOPE",
    "-" * 108,
    "Compared Random Forest, XGBoost, LightGBM and NGBoost using the same",
    "2019-2023 rolling-origin development validation years.",
    "2024 was NOT loaded.",
    "2025 was NOT loaded.",
    "No model training or tuning occurred in this step.",
    "No final model was selected.",
    "",
    "PRIMARY METRIC",
    "-" * 108,
    "MAE is the primary development comparison metric because it is directly",
    "interpretable as average LST error in degrees Celsius.",
    "",
    "UNTUNED DEVELOPMENT MEAN-MAE ORDERING",
    "-" * 108,
]

for _, row in summary_for_report.iterrows():
    report_lines.append(
        f"{int(row['development_mean_mae_rank'])}. "
        f"{row['model']}: "
        f"mean MAE={row['mean_validation_mae']:.6f} °C, "
        f"mean RMSE={row['mean_validation_rmse']:.6f} °C, "
        f"mean R²={row['mean_validation_r2']:.6f}, "
        f"negative-R² folds={int(row['negative_r2_folds'])}, "
        f"mean MAE gap={row['mean_mae_generalization_gap']:.6f} °C."
    )

report_lines.extend(
    [
        "",
        "IMPORTANT INTERPRETATION",
        "-" * 108,
        "The ordering above is NOT the final model selection.",
        "All four core candidates proceed to the controlled tuning stage.",
        "Step 17 will tune models using 2018-2023 development data only.",
        "Later spatial, temporal and reserved-2024 validation may change the evidence.",
        "",
        "TEMPORAL STABILITY",
        "-" * 108,
        (
            f"The highest average candidate MAE occurred in {hardest_year}. "
            "Difficult years are retained as evidence of temporal instability/"
            "distribution shift rather than treated as coding failures."
        ),
        "",
        "BASELINE CONTEXT",
        "-" * 108,
        "Fixed baseline models are summarized separately. Candidate models are",
        "compared against each fixed baseline using mean rolling-origin MAE.",
        "A different 'best baseline' chosen retrospectively for each year is not",
        "treated as a deployable single baseline strategy.",
        "",
        "NGBOOST UNCERTAINTY",
        "-" * 108,
    ]
)

for _, row in ngb_uncertainty.iterrows():
    report_lines.append(
        f"Nominal {int(row['nominal_interval_percent'])}% interval: "
        f"mean observed coverage="
        f"{row['mean_observed_coverage'] * 100:.2f}%, "
        f"mean width={row['mean_interval_width_C']:.6f} °C."
    )

report_lines.extend(
    [
        "",
        "The raw NGBoost uncertainty is diagnostic only and is not accepted",
        "as the final forecasting uncertainty method at this stage.",
        "",
        "STATISTICAL CAUTION",
        "-" * 108,
        "Only five temporal validation folds are available here. Pairwise fold",
        "differences are reported descriptively; no strong significance claim",
        "is made from these five folds.",
        "",
        "NEXT",
        "-" * 108,
        "Step 17 - Hyperparameter Tuning of RF, XGBoost, LightGBM and NGBoost",
        "using leakage-safe rolling-origin development validation only.",
    ]
)

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {REPORT_PATH}")
print(f"Saved: {AUDIT_PATH}")
print(f"Saved: {HASH_PATH}")


# =============================================================================
# 18. FINAL STATUS
# =============================================================================

section("STEP 16 COMPLETED SUCCESSFULLY")

print(
    "Untuned candidate comparison completed.\n"
    "Random Forest, XGBoost, LightGBM and NGBoost were compared using the same "
    "five rolling-origin development folds.\n"
    "Step 11 baselines were retained as reference models.\n"
    "Temporal difficulty and NGBoost uncertainty diagnostics were summarized.\n"
    "No 2024 or 2025 data was loaded.\n"
    "No model was trained or tuned.\n"
    "No final model was selected.\n"
    "Review the complete Step 16 output before Step 17 hyperparameter tuning."
)
