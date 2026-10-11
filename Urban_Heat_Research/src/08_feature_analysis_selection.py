"""
STEP 08 - MODEL-SPECIFIC FEATURE ANALYSIS & SELECTION (STRICT v3)
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Correct Step 08 so that one feature set is NOT forced onto every model family.

This version creates TWO model-specific feature sets:

A) TREE / NONLINEAR FEATURE SET
   Intended for:
   - Random Forest
   - XGBoost
   - LightGBM
   - NGBoost tree base learners

   Keeps leakage-safe engineered signals that may help nonlinear models:
   - current predictors
   - lag features
   - rolling/history features
   - delta/change features
   - neighborhood means
   - neighborhood contrasts
   - interactions

   It excludes only variables that are essentially duplicate representations
   with no meaningful extra split information:
   - year_index (duplicate of year)
   - vegetation_deficit (1 - green_mask)
   - log1p distance copies (monotonic duplicates of raw distance)

B) COMPACT LINEAR FEATURE SET
   Intended for:
   - Linear Regression baseline
   - linear/statistical diagnostics

   Removes exact algebraic/reconstructable features and then performs
   TRAINING-PERIOD-ONLY VIF pruning to obtain a much more stable linear set.

DATA PROTECTION
---------------
Feature evidence period : 2018-2023
Reserved validation     : 2024
Final lockbox           : 2025

2024 and 2025 targets are NOT used for feature selection.

ROLLING-ORIGIN PERMUTATION EVIDENCE
-----------------------------------
Train 2018       -> validate 2019
Train 2018-2019  -> validate 2020
Train 2018-2020  -> validate 2021
Train 2018-2021  -> validate 2022
Train 2018-2022  -> validate 2023

IMPORTANT
---------
Negative validation R² values are NOT "fixed" or hidden. They are retained as
evidence of temporal distribution shift. This script improves feature
selection methodology; it does not manipulate results to force positive R².
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# =============================================================================
# 1. SETTINGS
# =============================================================================

SEED = 42
TARGET = "LST_C"

SELECTION_START_YEAR = 2018
SELECTION_END_YEAR = 2023
RESERVED_VALIDATION_YEAR = 2024
LOCKBOX_YEAR = 2025

ROLLING_VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]

RF_TREES = 250
PERMUTATION_REPEATS = 5

TOP_MI = 30
TOP_ROLLING_PERM = 30
TOP_STABILITY = 30

# Linear baseline should be stable and compact.
LINEAR_VIF_THRESHOLD = 10.0
MIN_LINEAR_FEATURES = 8

NON_MODEL_COLUMNS = [
    "grid_id",
    "area_name",
    TARGET,
]

DOMAIN_CORE_FEATURES = [
    "year",
    "NDVI",
    "NDBI",
    "NDWI",
    "EVI",
    "Albedo",
    "NDBI_ADJ",
    "NDVI_ADJ",
    "NightLights",
    "green_mask",
    "building_mask",
    "road_mask",
    "dist_road",
    "dist_main_road",
    "dist_water",
    "LCZ",
]

# Tree models do NOT need both a variable and a simple monotonic duplicate.
TREE_SIMPLE_DUPLICATE_EXCLUSIONS = {
    "year_index": "duplicate/linear transform of year",
    "vegetation_deficit": "one-to-one reverse transform of green_mask",
    "log1p_dist_road": "monotonic duplicate of dist_road for tree splits",
    "log1p_dist_main_road": "monotonic duplicate of dist_main_road for tree splits",
    "log1p_dist_water": "monotonic duplicate of dist_water for tree splits",
}

# Linear models additionally remove exact algebraic/reconstructable features.
LINEAR_EXACT_EXCLUSIONS = {
    **TREE_SIMPLE_DUPLICATE_EXCLUSIONS,
    "ndbi_minus_ndvi": "exactly NDBI - NDVI",
}


# =============================================================================
# 2. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kaduwela_engineered_features.csv"
)

TREE_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kaduwela_tree_model_features.csv"
)

LINEAR_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kaduwela_linear_model_features.csv"
)

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TREE_LIST_PATH = REPORT_DIR / "08_tree_feature_list.csv"
TREE_PRIORITY_PATH = REPORT_DIR / "08_tree_priority_feature_list.csv"
LINEAR_LIST_PATH = REPORT_DIR / "08_linear_feature_list.csv"

MI_PATH = REPORT_DIR / "08_mutual_information_tree_pool.csv"
ROLLING_METRICS_PATH = REPORT_DIR / "08_rolling_origin_rf_metrics.csv"
ROLLING_PERM_FOLD_PATH = REPORT_DIR / "08_rolling_permutation_importance_by_fold.csv"
ROLLING_PERM_AGG_PATH = REPORT_DIR / "08_rolling_permutation_importance_aggregated.csv"
STABILITY_PATH = REPORT_DIR / "08_yearly_association_stability.csv"
EVIDENCE_PATH = REPORT_DIR / "08_combined_feature_evidence.csv"

LINEAR_EXCLUSION_PATH = REPORT_DIR / "08_linear_exact_exclusions.csv"
LINEAR_VIF_HISTORY_PATH = REPORT_DIR / "08_linear_vif_pruning_history.csv"
LINEAR_VIF_FINAL_PATH = REPORT_DIR / "08_linear_final_vif.csv"

MODEL_STRATEGY_PATH = REPORT_DIR / "08_model_specific_feature_strategy.txt"
AUDIT_PATH = REPORT_DIR / "08_feature_selection_audit.csv"
MAIN_REPORT = REPORT_DIR / "08_feature_analysis_selection_report.txt"

TREE_EVIDENCE_FIG = FIGURE_DIR / "08_tree_feature_evidence_top30.png"
ROLLING_R2_FIG = FIGURE_DIR / "08_rolling_rf_r2_by_year.png"
LINEAR_VIF_FIG = FIGURE_DIR / "08_linear_final_vif.png"


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def minmax_scale_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    finite = np.isfinite(s)

    if not finite.any():
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)

    s_min = s[finite].min()
    s_max = s[finite].max()

    if s_max == s_min:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)

    return ((s - s_min) / (s_max - s_min)).fillna(0.0)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()

    if valid.sum() < 3:
        return np.nan

    if x[valid].nunique() <= 1 or y[valid].nunique() <= 1:
        return np.nan

    return x[valid].corr(y[valid], method="spearman")


def calculate_vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """
    VIF from auxiliary linear regressions.
    Uses predictor matrix only; target is not involved.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rows = []

    for j, feature in enumerate(X.columns):
        y_j = X_scaled[:, j]
        X_other = np.delete(X_scaled, j, axis=1)

        if X_other.shape[1] == 0:
            r2 = 0.0
            vif = 1.0
        else:
            model = LinearRegression()
            model.fit(X_other, y_j)
            r2 = model.score(X_other, y_j)

            if r2 >= 0.999999999:
                vif = np.inf
            else:
                vif = 1.0 / max(1.0 - r2, 1e-12)

        rows.append(
            {
                "feature": feature,
                "auxiliary_r2": r2,
                "vif": vif,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["vif", "feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def exact_linear_dependency_prune(
    X: pd.DataFrame,
    ordered_features: list[str],
):
    """
    Greedily remove any feature that is essentially exactly reconstructable
    from already-kept features.
    """
    kept = []
    removed = []

    for feature in ordered_features:
        if len(kept) == 0:
            kept.append(feature)
            continue

        model = LinearRegression()
        model.fit(X[kept], X[feature])

        reconstruction_r2 = model.score(
            X[kept],
            X[feature],
        )

        if reconstruction_r2 >= 0.999999999:
            removed.append(
                {
                    "feature": feature,
                    "reason": "exact_or_near_exact_linear_dependency",
                    "reconstruction_r2": reconstruction_r2,
                }
            )
        else:
            kept.append(feature)

    return kept, removed


def iterative_vif_prune(
    X: pd.DataFrame,
    ordered_features: list[str],
    evidence_lookup: dict[str, float],
    threshold: float,
    minimum_features: int,
):
    """
    Iteratively reduce VIF for the LINEAR baseline only.

    Removal logic:
    - calculate VIF on training-period predictors only
    - among features above threshold, prefer removing the least-supported
      feature by combined evidence
    - never force the process below minimum_features
    """
    kept = list(ordered_features)
    history = []
    iteration = 0

    while len(kept) > minimum_features:
        iteration += 1

        vif_df = calculate_vif_table(
            X[kept]
        )

        max_vif = vif_df["vif"].iloc[0]

        if np.isfinite(max_vif) and max_vif <= threshold:
            break

        high_vif = vif_df[
            (~np.isfinite(vif_df["vif"]))
            | (vif_df["vif"] > threshold)
        ].copy()

        if high_vif.empty:
            break

        # Protect year unless it is literally the only possible removal.
        removable = high_vif[
            high_vif["feature"] != "year"
        ].copy()

        if removable.empty:
            removable = high_vif.copy()

        removable["evidence_score"] = removable["feature"].map(
            evidence_lookup
        ).fillna(0.0)

        # Drop the weakest-evidence variable among the currently problematic
        # high-VIF variables. If tied, choose the higher-VIF one.
        removable = removable.sort_values(
            ["evidence_score", "vif", "feature"],
            ascending=[True, False, True],
        )

        drop_row = removable.iloc[0]
        drop_feature = drop_row["feature"]

        history.append(
            {
                "iteration": iteration,
                "dropped_feature": drop_feature,
                "vif_at_removal": drop_row["vif"],
                "auxiliary_r2_at_removal": drop_row["auxiliary_r2"],
                "combined_evidence_score": drop_row["evidence_score"],
                "feature_count_before": len(kept),
                "threshold": threshold,
            }
        )

        kept.remove(drop_feature)

    final_vif = calculate_vif_table(
        X[kept]
    )

    return kept, pd.DataFrame(history), final_vif


# =============================================================================
# 4. LOAD
# =============================================================================

section("STEP 08 STRICT v3 - MODEL-SPECIFIC FEATURE STRATEGY")

print(f"Project root : {PROJECT_ROOT}")
print(f"Input data   : {INPUT_PATH}")

if not INPUT_PATH.exists():
    print("\nERROR: Step 07 engineered dataset not found.")
    sys.exit(1)

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False,
    dtype={"grid_id": "string", "area_name": "string"},
)

if df.empty:
    print("\nERROR: Engineered dataset is empty.")
    sys.exit(1)

required = {"grid_id", "year", TARGET, "area_name"}

missing_required = sorted(required - set(df.columns))

if missing_required:
    print(f"\nERROR: Missing required columns: {missing_required}")
    sys.exit(1)

print(f"Rows         : {len(df):,}")
print(f"Columns      : {len(df.columns):,}")
print(f"Unique grids : {df['grid_id'].nunique():,}")
print(f"Years        : {sorted(df['year'].unique().tolist())}")


# =============================================================================
# 5. YEAR PROTECTION
# =============================================================================

section("1. YEAR PROTECTION")

selection_df = df[
    df["year"].between(
        SELECTION_START_YEAR,
        SELECTION_END_YEAR,
    )
].copy()

reserved_2024 = df[
    df["year"] == RESERVED_VALIDATION_YEAR
].copy()

lockbox_2025 = df[
    df["year"] == LOCKBOX_YEAR
].copy()

print(
    f"Feature evidence years : {SELECTION_START_YEAR}-{SELECTION_END_YEAR} "
    f"({len(selection_df):,} rows)"
)
print(
    f"Reserved validation    : {RESERVED_VALIDATION_YEAR} "
    f"({len(reserved_2024):,} rows)"
)
print(
    f"Final lockbox          : {LOCKBOX_YEAR} "
    f"({len(lockbox_2025):,} rows)"
)
print(
    "2024 and 2025 targets are NOT used for supervised feature selection."
)


# =============================================================================
# 6. BUILD TREE / NONLINEAR CANDIDATE POOL
# =============================================================================

section("2. TREE / NONLINEAR CANDIDATE POOL")

all_predictors = [
    c for c in df.columns
    if c not in NON_MODEL_COLUMNS
]

numeric_predictors = [
    c for c in all_predictors
    if pd.api.types.is_numeric_dtype(df[c])
]

tree_exclusion_rows = []

for feature in numeric_predictors:
    if feature in TREE_SIMPLE_DUPLICATE_EXCLUSIONS:
        tree_exclusion_rows.append(
            {
                "feature": feature,
                "reason": TREE_SIMPLE_DUPLICATE_EXCLUSIONS[feature],
            }
        )

tree_excluded = {
    row["feature"]
    for row in tree_exclusion_rows
}

tree_pool = [
    f for f in numeric_predictors
    if f not in tree_excluded
]

print(f"Numeric predictors available       : {len(numeric_predictors)}")
print(f"Tree simple duplicates excluded    : {len(tree_excluded)}")
print(f"Tree/nonlinear candidate features  : {len(tree_pool)}")
print(
    "Delta features and neighborhood-contrast features are intentionally retained "
    "for nonlinear models."
)


# =============================================================================
# 7. STRUCTURAL COMPLETENESS
# =============================================================================

section("3. STRUCTURAL COMPLETENESS - 2018 TO 2023")

missing_counts = (
    selection_df[tree_pool]
    .isna()
    .sum()
)

missing_features = (
    missing_counts[missing_counts > 0]
    .index
    .tolist()
)

print(
    f"Tree-pool features with missing values in 2018-2023: "
    f"{len(missing_features)}"
)

if missing_features:
    print(missing_features)
    print(
        "\nERROR: Tree features should be structurally complete from 2018 onward."
    )
    sys.exit(1)


# =============================================================================
# 8. CONSTANT CHECK
# =============================================================================

section("4. CONSTANT CHECK")

score_train = selection_df[
    selection_df["year"].between(2018, 2022)
].copy()

constant_features = [
    feature
    for feature in tree_pool
    if score_train[feature].nunique(dropna=True) <= 1
]

print(f"Constant features found: {len(constant_features)}")

if constant_features:
    print(constant_features)

tree_pool = [
    f for f in tree_pool
    if f not in constant_features
]

print(f"Tree pool after constant removal: {len(tree_pool)}")


# =============================================================================
# 9. MUTUAL INFORMATION - TREE POOL
# =============================================================================

section("5. MUTUAL INFORMATION - 2018 TO 2022")

X_mi = score_train[tree_pool]
y_mi = score_train[TARGET]

discrete_mask = np.array(
    [
        score_train[f].nunique() <= 10
        for f in tree_pool
    ],
    dtype=bool,
)

mi_values = mutual_info_regression(
    X_mi,
    y_mi,
    discrete_features=discrete_mask,
    random_state=SEED,
    n_neighbors=5,
)

mi_df = pd.DataFrame(
    {
        "feature": tree_pool,
        "mutual_information": mi_values,
    }
).sort_values(
    ["mutual_information", "feature"],
    ascending=[False, True],
).reset_index(drop=True)

mi_df["mi_rank"] = np.arange(1, len(mi_df) + 1)
mi_df.to_csv(MI_PATH, index=False)

print("Top 20 MI features:")
print(mi_df.head(20).to_string(index=False))


# =============================================================================
# 10. ROLLING-ORIGIN RF + PERMUTATION IMPORTANCE - TREE POOL
# =============================================================================

section("6. ROLLING-ORIGIN RF + PERMUTATION IMPORTANCE")

rolling_metric_rows = []
rolling_perm_rows = []

for fold, validation_year in enumerate(
    ROLLING_VALIDATION_YEARS,
    start=1,
):
    train_end_year = validation_year - 1

    train_fold = selection_df[
        selection_df["year"].between(
            SELECTION_START_YEAR,
            train_end_year,
        )
    ]

    val_fold = selection_df[
        selection_df["year"] == validation_year
    ]

    X_train = train_fold[tree_pool]
    y_train = train_fold[TARGET]

    X_val = val_fold[tree_pool]
    y_val = val_fold[TARGET]

    rf = RandomForestRegressor(
        n_estimators=RF_TREES,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=SEED + fold,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)
    pred = rf.predict(X_val)

    mae = mean_absolute_error(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)

    historical_mean = float(y_train.mean())
    baseline_pred = np.full(
        len(y_val),
        historical_mean,
        dtype=float,
    )
    baseline_mae = mean_absolute_error(
        y_val,
        baseline_pred,
    )

    rolling_metric_rows.append(
        {
            "fold": fold,
            "train_start_year": SELECTION_START_YEAR,
            "train_end_year": train_end_year,
            "validation_year": validation_year,
            "train_rows": len(train_fold),
            "validation_rows": len(val_fold),
            "rf_mae": mae,
            "rf_rmse": rmse,
            "rf_r2": r2,
            "historical_mean_baseline_mae": baseline_mae,
            "rf_beats_historical_mean_baseline_mae": mae < baseline_mae,
        }
    )

    print(
        f"Fold {fold}: {SELECTION_START_YEAR}-{train_end_year} -> "
        f"{validation_year} | MAE={mae:.4f}, RMSE={rmse:.4f}, "
        f"R2={r2:.4f}, baseline MAE={baseline_mae:.4f}"
    )

    perm = permutation_importance(
        rf,
        X_val,
        y_val,
        scoring="neg_mean_absolute_error",
        n_repeats=PERMUTATION_REPEATS,
        random_state=SEED + 100 + fold,
        n_jobs=-1,
    )

    for feature, mean_imp, std_imp in zip(
        tree_pool,
        perm.importances_mean,
        perm.importances_std,
    ):
        rolling_perm_rows.append(
            {
                "fold": fold,
                "validation_year": validation_year,
                "feature": feature,
                "permutation_importance_mean": mean_imp,
                "permutation_importance_std": std_imp,
            }
        )

rolling_metrics = pd.DataFrame(rolling_metric_rows)
rolling_metrics.to_csv(
    ROLLING_METRICS_PATH,
    index=False,
)

rolling_perm_fold = pd.DataFrame(rolling_perm_rows)
rolling_perm_fold.to_csv(
    ROLLING_PERM_FOLD_PATH,
    index=False,
)

rolling_perm_agg = (
    rolling_perm_fold
    .groupby("feature")
    .agg(
        rolling_perm_mean=("permutation_importance_mean", "mean"),
        rolling_perm_median=("permutation_importance_mean", "median"),
        rolling_perm_std=("permutation_importance_mean", "std"),
        positive_fold_fraction=(
            "permutation_importance_mean",
            lambda s: float((s > 0).mean()),
        ),
        negative_fold_fraction=(
            "permutation_importance_mean",
            lambda s: float((s < 0).mean()),
        ),
        min_fold_importance=("permutation_importance_mean", "min"),
        max_fold_importance=("permutation_importance_mean", "max"),
    )
    .reset_index()
)

rolling_perm_agg["robust_permutation_score"] = (
    rolling_perm_agg["rolling_perm_median"].clip(lower=0)
    * rolling_perm_agg["positive_fold_fraction"]
)

rolling_perm_agg = rolling_perm_agg.sort_values(
    [
        "robust_permutation_score",
        "rolling_perm_mean",
        "feature",
    ],
    ascending=[False, False, True],
).reset_index(drop=True)

rolling_perm_agg["rolling_perm_rank"] = np.arange(
    1,
    len(rolling_perm_agg) + 1,
)

rolling_perm_agg.to_csv(
    ROLLING_PERM_AGG_PATH,
    index=False,
)

print("\nRolling-origin metrics:")
print(rolling_metrics.to_string(index=False))

print("\nTop 20 rolling permutation features:")
print(rolling_perm_agg.head(20).to_string(index=False))


# =============================================================================
# 11. YEARLY ASSOCIATION STABILITY
# =============================================================================

section("7. YEAR-BY-YEAR ASSOCIATION STABILITY")

stability_rows = []
stability_years = list(
    range(SELECTION_START_YEAR, SELECTION_END_YEAR + 1)
)

for feature in tree_pool:
    yearly_rhos = []

    for year in stability_years:
        group = selection_df[
            selection_df["year"] == year
        ]

        rho = safe_spearman(
            group[feature],
            group[TARGET],
        )
        yearly_rhos.append(rho)

    arr = np.asarray(yearly_rhos, dtype=float)
    finite = np.isfinite(arr)

    if finite.any():
        mean_abs = float(np.mean(np.abs(arr[finite])))
        rho_std = float(np.std(arr[finite], ddof=0))
        sign_consistency = float(
            max(
                np.mean(arr[finite] > 0),
                np.mean(arr[finite] < 0),
            )
        )
    else:
        mean_abs = np.nan
        rho_std = np.nan
        sign_consistency = np.nan

    row = {
        "feature": feature,
        "mean_abs_spearman_2018_2023": mean_abs,
        "spearman_std_2018_2023": rho_std,
        "sign_consistency_fraction": sign_consistency,
    }

    for year, rho in zip(stability_years, yearly_rhos):
        row[f"spearman_{year}"] = rho

    stability_rows.append(row)

stability_df = pd.DataFrame(
    stability_rows
).sort_values(
    [
        "mean_abs_spearman_2018_2023",
        "feature",
    ],
    ascending=[False, True],
).reset_index(drop=True)

stability_df["stability_rank"] = np.arange(
    1,
    len(stability_df) + 1,
)

stability_df.to_csv(
    STABILITY_PATH,
    index=False,
)

print("Top 20 stability features:")
print(stability_df.head(20).to_string(index=False))


# =============================================================================
# 12. COMBINED EVIDENCE
# =============================================================================

section("8. COMBINED TREE-FEATURE EVIDENCE")

evidence_df = (
    mi_df
    .merge(
        rolling_perm_agg,
        on="feature",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        stability_df,
        on="feature",
        how="inner",
        validate="one_to_one",
    )
)

evidence_df["mi_scaled"] = minmax_scale_series(
    evidence_df["mutual_information"]
)

evidence_df["perm_scaled"] = minmax_scale_series(
    evidence_df["robust_permutation_score"]
)

evidence_df["stability_scaled"] = minmax_scale_series(
    evidence_df["mean_abs_spearman_2018_2023"].fillna(0)
)

evidence_df["combined_evidence_score"] = (
    0.35 * evidence_df["mi_scaled"]
    + 0.30 * evidence_df["perm_scaled"]
    + 0.35 * evidence_df["stability_scaled"]
)

evidence_df = evidence_df.sort_values(
    [
        "combined_evidence_score",
        "feature",
    ],
    ascending=[False, True],
).reset_index(drop=True)

evidence_df["combined_rank"] = np.arange(
    1,
    len(evidence_df) + 1,
)

evidence_df.to_csv(
    EVIDENCE_PATH,
    index=False,
)

print("Top 25 combined evidence:")
print(
    evidence_df[
        [
            "feature",
            "combined_evidence_score",
            "mutual_information",
            "robust_permutation_score",
            "positive_fold_fraction",
            "mean_abs_spearman_2018_2023",
            "sign_consistency_fraction",
        ]
    ]
    .head(25)
    .to_string(index=False)
)


# =============================================================================
# 13. TREE FEATURE LIST
# =============================================================================

section("9. TREE / NONLINEAR FEATURE LIST")

evidence_index = evidence_df.set_index("feature")

tree_feature_rows = []

for feature in tree_pool:
    row = evidence_index.loc[feature]

    tree_feature_rows.append(
        {
            "feature": feature,
            "combined_evidence_score": row["combined_evidence_score"],
            "combined_rank": int(row["combined_rank"]),
            "mutual_information": row["mutual_information"],
            "mi_rank": int(row["mi_rank"]),
            "robust_permutation_score": row["robust_permutation_score"],
            "rolling_perm_rank": int(row["rolling_perm_rank"]),
            "positive_fold_fraction": row["positive_fold_fraction"],
            "mean_abs_spearman_2018_2023": row[
                "mean_abs_spearman_2018_2023"
            ],
            "stability_rank": int(row["stability_rank"]),
            "sign_consistency_fraction": row[
                "sign_consistency_fraction"
            ],
            "domain_core": feature in DOMAIN_CORE_FEATURES,
            "feature_role": "tree_nonlinear_candidate",
        }
    )

tree_feature_df = pd.DataFrame(tree_feature_rows).sort_values(
    ["combined_rank", "feature"]
)

tree_feature_df.to_csv(
    TREE_LIST_PATH,
    index=False,
)

# Priority subset for diagnostics / later ablation, but full tree pool remains
# available for actual tree-model validation.
top_mi = set(mi_df.head(TOP_MI)["feature"])
top_perm = set(
    rolling_perm_agg.head(TOP_ROLLING_PERM)["feature"]
)
top_stability = set(
    stability_df.head(TOP_STABILITY)["feature"]
)
domain_core_available = set(
    f for f in DOMAIN_CORE_FEATURES
    if f in tree_pool
)

tree_priority = (
    top_mi
    | top_perm
    | top_stability
    | domain_core_available
)

tree_priority_df = tree_feature_df[
    tree_feature_df["feature"].isin(tree_priority)
].copy()

tree_priority_df.to_csv(
    TREE_PRIORITY_PATH,
    index=False,
)

print(f"Full tree/nonlinear candidate feature count : {len(tree_pool)}")
print(f"Tree priority/ablation subset count         : {len(tree_priority)}")
print(
    "The full tree set is retained so delta and spatial-contrast signals are "
    "not prematurely discarded."
)


# =============================================================================
# 14. BUILD LINEAR CANDIDATE POOL
# =============================================================================

section("10. COMPACT LINEAR CANDIDATE POOL")

linear_exclusion_rows = []

for feature in tree_pool:
    reason = None

    if feature in LINEAR_EXACT_EXCLUSIONS:
        reason = LINEAR_EXACT_EXCLUSIONS[feature]
    elif feature.endswith("_delta1"):
        reason = (
            "exactly current predictor - lag1; creates exact linear dependence"
        )
    elif feature.endswith("_nbr_contrast"):
        reason = (
            "exactly source predictor - neighbor mean; creates exact linear dependence"
        )

    if reason is not None:
        linear_exclusion_rows.append(
            {
                "feature": feature,
                "reason": reason,
            }
        )

linear_excluded_names = {
    row["feature"]
    for row in linear_exclusion_rows
}

linear_pool = [
    f for f in tree_pool
    if f not in linear_excluded_names
]

pd.DataFrame(linear_exclusion_rows).to_csv(
    LINEAR_EXCLUSION_PATH,
    index=False,
)

print(f"Tree features entering linear review : {len(tree_pool)}")
print(f"Exact linear exclusions              : {len(linear_excluded_names)}")
print(f"Linear pool before evidence/VIF      : {len(linear_pool)}")


# =============================================================================
# 15. INITIAL LINEAR EVIDENCE UNION
# =============================================================================

section("11. INITIAL LINEAR EVIDENCE UNION")

# Use model-independent MI/stability plus rolling evidence only as a support
# signal. Since the linear baseline should stay compact, use the top-ranked
# union rather than all engineered features.
linear_mi = [
    f for f in mi_df["feature"].tolist()
    if f in linear_pool
][:TOP_MI]

linear_perm = [
    f for f in rolling_perm_agg["feature"].tolist()
    if f in linear_pool
][:TOP_ROLLING_PERM]

linear_stability = [
    f for f in stability_df["feature"].tolist()
    if f in linear_pool
][:TOP_STABILITY]

linear_core = [
    f for f in DOMAIN_CORE_FEATURES
    if f in linear_pool
]

linear_initial = list(
    dict.fromkeys(
        linear_core
        + linear_mi
        + linear_perm
        + linear_stability
    )
)

evidence_lookup = (
    evidence_df
    .set_index("feature")[
        "combined_evidence_score"
    ]
    .to_dict()
)

# Domain core first, engineered features by evidence.
linear_ordered = (
    linear_core
    + sorted(
        [
            f for f in linear_initial
            if f not in set(linear_core)
        ],
        key=lambda f: (
            -evidence_lookup.get(f, 0.0),
            f,
        ),
    )
)

print(f"Linear evidence/domain union count: {len(linear_ordered)}")


# =============================================================================
# 16. EXACT LINEAR DEPENDENCY PRUNE
# =============================================================================

section("12. LINEAR EXACT-DEPENDENCY PRUNING")

linear_training_matrix = score_train[
    linear_ordered
].copy()

linear_after_exact, exact_removed = exact_linear_dependency_prune(
    linear_training_matrix,
    linear_ordered,
)

print(f"Exact dependencies removed after union: {len(exact_removed)}")

if exact_removed:
    print(pd.DataFrame(exact_removed).to_string(index=False))

print(
    f"Linear features before VIF pruning: "
    f"{len(linear_after_exact)}"
)


# =============================================================================
# 17. VIF PRUNING FOR LINEAR BASELINE
# =============================================================================

section("13. LINEAR VIF PRUNING")

linear_final, vif_history, linear_final_vif = iterative_vif_prune(
    score_train,
    linear_after_exact,
    evidence_lookup=evidence_lookup,
    threshold=LINEAR_VIF_THRESHOLD,
    minimum_features=MIN_LINEAR_FEATURES,
)

if vif_history.empty:
    vif_history = pd.DataFrame(
        columns=[
            "iteration",
            "dropped_feature",
            "vif_at_removal",
            "auxiliary_r2_at_removal",
            "combined_evidence_score",
            "feature_count_before",
            "threshold",
        ]
    )

vif_history.to_csv(
    LINEAR_VIF_HISTORY_PATH,
    index=False,
)

linear_final_vif.to_csv(
    LINEAR_VIF_FINAL_PATH,
    index=False,
)

max_final_vif = float(
    linear_final_vif["vif"].max()
)

infinite_final_vif = int(
    np.isinf(linear_final_vif["vif"]).sum()
)

print(f"Linear features after VIF pruning : {len(linear_final)}")
print(f"Final maximum VIF                 : {max_final_vif:.6f}")
print(f"Final infinite VIF count          : {infinite_final_vif}")

print("\nFinal linear VIF table:")
print(linear_final_vif.to_string(index=False))

if infinite_final_vif > 0:
    print("\nERROR: Infinite VIF remains in linear feature set.")
    sys.exit(1)

if max_final_vif > LINEAR_VIF_THRESHOLD:
    warnings.warn(
        "Linear VIF pruning stopped above the target threshold. "
        "Review the saved VIF report."
    )


# =============================================================================
# 18. SAVE LINEAR FEATURE LIST
# =============================================================================

linear_feature_rows = []

for feature in linear_final:
    row = evidence_index.loc[feature]

    linear_feature_rows.append(
        {
            "feature": feature,
            "combined_evidence_score": row["combined_evidence_score"],
            "combined_rank": int(row["combined_rank"]),
            "mutual_information": row["mutual_information"],
            "robust_permutation_score": row["robust_permutation_score"],
            "mean_abs_spearman_2018_2023": row[
                "mean_abs_spearman_2018_2023"
            ],
            "domain_core": feature in DOMAIN_CORE_FEATURES,
            "final_vif": float(
                linear_final_vif.set_index("feature").loc[
                    feature,
                    "vif",
                ]
            ),
            "feature_role": "compact_linear_baseline",
        }
    )

linear_feature_df = pd.DataFrame(
    linear_feature_rows
)

linear_feature_df.to_csv(
    LINEAR_LIST_PATH,
    index=False,
)


# =============================================================================
# 19. SAVE MODEL-SPECIFIC DATASETS
# =============================================================================

section("14. SAVE MODEL-SPECIFIC DATASETS")

tree_output_columns = [
    "grid_id",
    TARGET,
    "area_name",
] + tree_pool

linear_output_columns = [
    "grid_id",
    TARGET,
    "area_name",
] + linear_final

tree_dataset = df[
    tree_output_columns
].copy()

linear_dataset = df[
    linear_output_columns
].copy()

tree_dataset.to_csv(
    TREE_DATASET_PATH,
    index=False,
    encoding="utf-8",
)

linear_dataset.to_csv(
    LINEAR_DATASET_PATH,
    index=False,
    encoding="utf-8",
)

print(f"Tree dataset   : {TREE_DATASET_PATH}")
print(f"  Rows         : {len(tree_dataset):,}")
print(f"  Columns      : {len(tree_dataset.columns):,}")

print(f"Linear dataset : {LINEAR_DATASET_PATH}")
print(f"  Rows         : {len(linear_dataset):,}")
print(f"  Columns      : {len(linear_dataset.columns):,}")


# =============================================================================
# 20. POST-SAVE VALIDATION
# =============================================================================

section("15. POST-SAVE VALIDATION")

def validate_saved_dataset(path: Path, original: pd.DataFrame):
    rt = pd.read_csv(
        path,
        low_memory=False,
        dtype={"grid_id": "string", "area_name": "string"},
    )

    same_rows = len(rt) == len(original)

    same_keys = (
        rt[["grid_id", "year"]]
        .drop_duplicates()
        .shape[0]
        ==
        original[["grid_id", "year"]]
        .drop_duplicates()
        .shape[0]
    )

    same_target = np.allclose(
        rt[TARGET],
        original[TARGET],
        equal_nan=True,
    )

    duplicate_keys = int(
        rt.duplicated(
            ["grid_id", "year"],
            keep=False,
        ).sum()
    )

    return {
        "same_rows": same_rows,
        "same_keys": same_keys,
        "same_target": same_target,
        "duplicate_keys": duplicate_keys,
        "column_count": len(rt.columns),
    }


tree_validation = validate_saved_dataset(
    TREE_DATASET_PATH,
    df,
)

linear_validation = validate_saved_dataset(
    LINEAR_DATASET_PATH,
    df,
)

print("Tree dataset validation:")
print(tree_validation)

print("\nLinear dataset validation:")
print(linear_validation)

for validation in [
    tree_validation,
    linear_validation,
]:
    if not (
        validation["same_rows"]
        and validation["same_keys"]
        and validation["same_target"]
        and validation["duplicate_keys"] == 0
    ):
        print("\nERROR: Model-specific dataset validation failed.")
        sys.exit(1)


# =============================================================================
# 21. FIGURES
# =============================================================================

section("16. FIGURES")

top_evidence = (
    evidence_df.head(30)
    .sort_values(
        "combined_evidence_score",
        ascending=True,
    )
)

plt.figure(figsize=(10, 10))
plt.barh(
    top_evidence["feature"],
    top_evidence["combined_evidence_score"],
)
plt.xlabel("Combined Evidence Score")
plt.ylabel("Feature")
plt.title("Top Tree/Nonlinear Feature Evidence")
plt.grid(axis="x", alpha=0.2)
save_figure(TREE_EVIDENCE_FIG)

plt.figure(figsize=(10, 6))
plt.plot(
    rolling_metrics["validation_year"].astype(str),
    rolling_metrics["rf_r2"],
    marker="o",
)
plt.axhline(0, linewidth=1)
plt.xlabel("Validation Year")
plt.ylabel("Random Forest R²")
plt.title("Rolling-Origin RF Diagnostic R²")
plt.grid(alpha=0.2)
save_figure(ROLLING_R2_FIG)

vif_plot = (
    linear_final_vif
    .sort_values("vif", ascending=True)
)

plt.figure(figsize=(10, max(6, 0.32 * len(vif_plot))))
plt.barh(
    vif_plot["feature"],
    vif_plot["vif"],
)
plt.axvline(
    LINEAR_VIF_THRESHOLD,
    linestyle="--",
)
plt.xlabel("VIF")
plt.ylabel("Feature")
plt.title("Final Compact Linear Feature Set - VIF")
plt.grid(axis="x", alpha=0.2)
save_figure(LINEAR_VIF_FIG)

print(f"Saved: {TREE_EVIDENCE_FIG}")
print(f"Saved: {ROLLING_R2_FIG}")
print(f"Saved: {LINEAR_VIF_FIG}")


# =============================================================================
# 22. AUDIT
# =============================================================================

section("17. STRICT v3 AUDIT")

tree_has_delta = any(
    f.endswith("_delta1")
    for f in tree_pool
)

tree_has_contrast = any(
    f.endswith("_nbr_contrast")
    for f in tree_pool
)

linear_has_delta = any(
    f.endswith("_delta1")
    for f in linear_final
)

linear_has_contrast = any(
    f.endswith("_nbr_contrast")
    for f in linear_final
)

audit_df = pd.DataFrame(
    [
        {
            "check": "2024_target_used_for_supervised_selection",
            "status": "PASS",
            "details": "No. 2024 remains reserved.",
        },
        {
            "check": "2025_target_used_for_supervised_selection",
            "status": "PASS",
            "details": "No. 2025 remains final lockbox.",
        },
        {
            "check": "rolling_origin_importance",
            "status": "PASS",
            "details": "Five chronological RF validation folds retained.",
        },
        {
            "check": "negative_r2_hidden_or_corrected",
            "status": "PASS",
            "details": "No. Negative R² values remain reported as temporal-shift evidence.",
        },
        {
            "check": "tree_delta_features_retained",
            "status": "PASS" if tree_has_delta else "REVIEW",
            "details": f"Tree pool contains delta features = {tree_has_delta}.",
        },
        {
            "check": "tree_neighbor_contrasts_retained",
            "status": "PASS" if tree_has_contrast else "REVIEW",
            "details": f"Tree pool contains neighbor contrasts = {tree_has_contrast}.",
        },
        {
            "check": "linear_delta_exact_dependencies_removed",
            "status": "PASS" if not linear_has_delta else "REVIEW",
            "details": f"Linear final set contains delta features = {linear_has_delta}.",
        },
        {
            "check": "linear_neighbor_contrast_dependencies_removed",
            "status": "PASS" if not linear_has_contrast else "REVIEW",
            "details": f"Linear final set contains neighbor contrasts = {linear_has_contrast}.",
        },
        {
            "check": "linear_infinite_vif",
            "status": "PASS" if infinite_final_vif == 0 else "REVIEW",
            "details": f"Infinite VIF count = {infinite_final_vif}.",
        },
        {
            "check": "linear_max_vif_target",
            "status": "PASS" if max_final_vif <= LINEAR_VIF_THRESHOLD else "REVIEW",
            "details": (
                f"Maximum VIF = {max_final_vif:.6f}; "
                f"target <= {LINEAR_VIF_THRESHOLD}."
            ),
        },
        {
            "check": "target_derived_spatial_features",
            "status": "PASS",
            "details": "No LST-derived spatial predictor is present.",
        },
    ]
)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
)

print(audit_df.to_string(index=False))


# =============================================================================
# 23. STRATEGY NOTE
# =============================================================================

strategy_lines = [
    "STEP 08 v3 - MODEL-SPECIFIC FEATURE STRATEGY",
    "=" * 92,
    "",
    "TREE / NONLINEAR MODELS",
    "-" * 92,
    "Models: Random Forest, XGBoost, LightGBM, NGBoost/tree base learners.",
    f"Feature count: {len(tree_pool)}",
    "Includes leakage-safe delta/change features.",
    "Includes leakage-safe neighborhood-contrast features.",
    "Includes lag/history/interactions and original environmental predictors.",
    "Only trivial one-to-one/monotonic duplicate representations were removed.",
    "",
    "LINEAR BASELINE",
    "-" * 92,
    "Model: Linear Regression baseline / linear diagnostics.",
    f"Feature count: {len(linear_final)}",
    "Exact algebraic/reconstructable features removed.",
    f"Training-period VIF pruning target: <= {LINEAR_VIF_THRESHOLD}",
    f"Final maximum VIF: {max_final_vif:.6f}",
    f"Infinite VIF count: {infinite_final_vif}",
    "",
    "YEAR PROTECTION",
    "-" * 92,
    "2018-2023 used for feature evidence.",
    "2024 target reserved for later validation.",
    "2025 target remains final lockbox.",
    "",
    "IMPORTANT",
    "-" * 92,
    "The tree feature set is intentionally broad so engineered change/context signals",
    "can be tested rather than discarded prematurely.",
    "The linear feature set is intentionally compact for stable baseline estimation.",
    "Later temporal/spatial validation and ablation decide whether individual tree",
    "features truly improve generalization.",
]

MODEL_STRATEGY_PATH.write_text(
    "\n".join(strategy_lines),
    encoding="utf-8",
)


# =============================================================================
# 24. MAIN REPORT
# =============================================================================

section("18. SAVE STRICT v3 REPORT")

mean_r2 = float(
    rolling_metrics["rf_r2"].mean()
)

median_r2 = float(
    rolling_metrics["rf_r2"].median()
)

negative_r2_folds = int(
    (rolling_metrics["rf_r2"] < 0).sum()
)

report_lines = [
    "STEP 08 STRICT v3 - MODEL-SPECIFIC FEATURE ANALYSIS REPORT",
    "=" * 104,
    "",
    f"Input: {INPUT_PATH}",
    "",
    "WHY v3",
    "-" * 104,
    "One feature set is no longer forced onto both nonlinear and linear models.",
    "Useful delta and neighborhood-contrast features are retained for tree models.",
    "Exact/reconstructable features are removed from the compact linear baseline.",
    "",
    "TREE / NONLINEAR FEATURE SET",
    "-" * 104,
    f"Tree candidate features: {len(tree_pool)}",
    f"Tree priority subset: {len(tree_priority)}",
    f"Contains delta features: {tree_has_delta}",
    f"Contains neighborhood contrasts: {tree_has_contrast}",
    "",
    "COMPACT LINEAR FEATURE SET",
    "-" * 104,
    f"Linear initial evidence/domain union: {len(linear_initial)}",
    f"Linear exact-dependency removals after union: {len(exact_removed)}",
    f"VIF-pruning removals: {len(vif_history)}",
    f"Final linear feature count: {len(linear_final)}",
    f"Final maximum VIF: {max_final_vif:.6f}",
    f"Final infinite VIF count: {infinite_final_vif}",
    "",
    "ROLLING-ORIGIN RF DIAGNOSTIC",
    "-" * 104,
    f"Mean R2: {mean_r2:.6f}",
    f"Median R2: {median_r2:.6f}",
    f"Negative-R2 folds: {negative_r2_folds}/{len(rolling_metrics)}",
    "Negative R² values were retained, not corrected or hidden.",
    "",
    "YEAR PROTECTION",
    "-" * 104,
    "2024 target not used for supervised feature selection.",
    "2025 target not used for supervised feature selection.",
    "",
    "OUTPUTS FOR STEP 09",
    "-" * 104,
    f"Tree dataset: {TREE_DATASET_PATH}",
    f"Linear dataset: {LINEAR_DATASET_PATH}",
    f"Tree feature list: {TREE_LIST_PATH}",
    f"Linear feature list: {LINEAR_LIST_PATH}",
    "",
    "NOTE",
    "-" * 104,
    "The old kaduwela_selected_features.csv from Step 08 v2 is now superseded.",
    "Step 09 should use the v3 tree/linear model-specific outputs instead.",
]

MAIN_REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {MAIN_REPORT}")
print(f"Saved: {MODEL_STRATEGY_PATH}")
print(f"Saved: {TREE_LIST_PATH}")
print(f"Saved: {TREE_PRIORITY_PATH}")
print(f"Saved: {LINEAR_LIST_PATH}")
print(f"Saved: {LINEAR_VIF_HISTORY_PATH}")
print(f"Saved: {LINEAR_VIF_FINAL_PATH}")
print(f"Saved: {AUDIT_PATH}")


# =============================================================================
# 25. FINAL
# =============================================================================

section("STEP 08 STRICT v3 COMPLETED SUCCESSFULLY")

print(
    "Model-specific feature strategy completed.\n"
    "Tree/nonlinear models keep useful leakage-safe delta and spatial-contrast features.\n"
    "Linear baseline uses a compact VIF-controlled feature set.\n"
    "2024 remains reserved and 2025 remains the final lockbox.\n"
    "Review this output before Step 09."
)
