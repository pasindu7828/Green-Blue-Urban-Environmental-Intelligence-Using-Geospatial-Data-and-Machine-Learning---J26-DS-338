"""
STEP 02 - TARGET VALIDATION
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Deeply validate the regression target variable (LST_C) before cleaning,
feature engineering, splitting, or model training.

IMPORTANT:
- This script DOES NOT delete, replace, cap, winsorize, transform, or impute LST_C.
- Outliers are treated as investigation candidates, not automatic errors.
- 2025 is still part of raw-data validation here. It is NOT used for model tuning/training.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wasserstein_distance


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "kaduwela_master_raw.csv"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_REPORT_PATH = REPORT_DIR / "02_target_validation_report.txt"
OVERALL_STATS_PATH = REPORT_DIR / "02_target_overall_statistics.csv"
YEAR_STATS_PATH = REPORT_DIR / "02_target_year_statistics.csv"
AREA_STATS_PATH = REPORT_DIR / "02_target_area_statistics.csv"
OUTLIER_PATH = REPORT_DIR / "02_target_outlier_candidates.csv"
EXTREMES_PATH = REPORT_DIR / "02_target_extreme_values.csv"
YEAR_DRIFT_PATH = REPORT_DIR / "02_target_year_drift.csv"
GRID_CHANGE_PATH = REPORT_DIR / "02_target_grid_year_changes.csv"
TOP_GRID_CHANGE_PATH = REPORT_DIR / "02_target_largest_grid_year_changes.csv"

HIST_PATH = FIGURE_DIR / "02_lst_distribution_histogram.png"
BOX_PATH = FIGURE_DIR / "02_lst_overall_boxplot.png"
YEAR_BOX_PATH = FIGURE_DIR / "02_lst_yearly_boxplot.png"
YEAR_TREND_PATH = FIGURE_DIR / "02_lst_yearly_mean_median.png"
QQ_PATH = FIGURE_DIR / "02_lst_qq_plot.png"


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def modified_z_scores(values: pd.Series) -> pd.Series:
    """
    Robust modified z-score based on median absolute deviation (MAD).

    Formula:
        0.6745 * (x - median) / MAD

    If MAD = 0, return NaN because the score is not defined.
    """
    median = values.median()
    mad = np.median(np.abs(values - median))

    if mad == 0 or np.isnan(mad):
        return pd.Series(np.nan, index=values.index)

    return 0.6745 * (values - median) / mad


# =============================================================================
# 3. LOAD RAW DATA
# =============================================================================

section("STEP 02 - TARGET VALIDATION")

print(f"Project root : {PROJECT_ROOT}")
print(f"Input file   : {RAW_DATA_PATH}")

if not RAW_DATA_PATH.exists():
    print("\nERROR: Raw master CSV was not found.")
    print(f"Expected: {RAW_DATA_PATH}")
    sys.exit(1)

try:
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
except Exception as exc:
    print(f"\nERROR: Could not read the raw CSV.\n{exc}")
    sys.exit(1)

if df.empty:
    print("\nERROR: Dataset is empty.")
    sys.exit(1)

required_columns = {"grid_id", "year", "LST_C"}
missing_required = required_columns - set(df.columns)

if missing_required:
    print("\nERROR: Required columns are missing:")
    print(sorted(missing_required))
    sys.exit(1)

print("Dataset loaded successfully.")
print(f"Rows: {len(df):,}")


# =============================================================================
# 4. TARGET TYPE / PARSING / FINITE-VALUE VALIDATION
# =============================================================================

section("1. TARGET TYPE AND FINITE-VALUE VALIDATION")

raw_target = df["LST_C"].copy()
target = pd.to_numeric(raw_target, errors="coerce")

parse_failures = int((raw_target.notna() & target.isna()).sum())
missing_count = int(target.isna().sum())
positive_inf_count = int(np.isposinf(target).sum())
negative_inf_count = int(np.isneginf(target).sum())
finite_count = int(np.isfinite(target).sum())

print(f"Original dtype              : {df['LST_C'].dtype}")
print(f"Numeric parse failures      : {parse_failures:,}")
print(f"Missing target values       : {missing_count:,}")
print(f"Positive infinity values    : {positive_inf_count:,}")
print(f"Negative infinity values    : {negative_inf_count:,}")
print(f"Finite target values        : {finite_count:,}")

if parse_failures > 0:
    warnings.warn(
        "Some LST_C values could not be converted to numeric. "
        "They must be investigated before modeling."
    )

if missing_count > 0:
    warnings.warn(
        "Missing LST_C values were detected. "
        "Do not impute/delete them automatically."
    )

if positive_inf_count > 0 or negative_inf_count > 0:
    warnings.warn(
        "Infinite LST_C values were detected and require investigation."
    )

df = df.copy()
df["LST_C_numeric"] = target


# =============================================================================
# 5. OVERALL TARGET DISTRIBUTION
# =============================================================================

section("2. OVERALL TARGET DISTRIBUTION")

valid_target = target.replace([np.inf, -np.inf], np.nan).dropna()

quantiles = valid_target.quantile(
    [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
)

overall_stats = {
    "count": int(valid_target.count()),
    "missing_count": missing_count,
    "mean": valid_target.mean(),
    "median": valid_target.median(),
    "std": valid_target.std(),
    "variance": valid_target.var(),
    "min": valid_target.min(),
    "q01": quantiles.loc[0.01],
    "q05": quantiles.loc[0.05],
    "q10": quantiles.loc[0.10],
    "q25": quantiles.loc[0.25],
    "q50": quantiles.loc[0.50],
    "q75": quantiles.loc[0.75],
    "q90": quantiles.loc[0.90],
    "q95": quantiles.loc[0.95],
    "q99": quantiles.loc[0.99],
    "max": valid_target.max(),
    "range": valid_target.max() - valid_target.min(),
    "iqr": quantiles.loc[0.75] - quantiles.loc[0.25],
    "skewness": valid_target.skew(),
    "excess_kurtosis": valid_target.kurt(),
    "unique_values": int(valid_target.nunique()),
}

overall_stats_df = pd.DataFrame([overall_stats])
overall_stats_df.to_csv(OVERALL_STATS_PATH, index=False)

for key, value in overall_stats.items():
    if isinstance(value, (float, np.floating)):
        print(f"{key:>20}: {value:.6f}")
    else:
        print(f"{key:>20}: {value}")


# =============================================================================
# 6. DISTRIBUTION DIAGNOSTIC TEST
# =============================================================================

section("3. DISTRIBUTION SHAPE DIAGNOSTIC")

# D'Agostino-Pearson normality test is used only as a diagnostic.
# Tree-based models do NOT require a normally distributed target.
if len(valid_target) >= 20:
    normaltest_stat, normaltest_p = stats.normaltest(valid_target)
    print(f"D'Agostino K² statistic : {normaltest_stat:.6f}")
    print(f"D'Agostino p-value      : {normaltest_p:.6g}")
    print(
        "NOTE: With a large dataset, even small departures from normality can "
        "produce very small p-values. This is NOT a reason to transform or remove data."
    )
else:
    normaltest_stat = np.nan
    normaltest_p = np.nan
    print("Not enough observations for D'Agostino-Pearson normality test.")


# =============================================================================
# 7. GLOBAL OUTLIER DIAGNOSTICS - IQR + MAD
# =============================================================================

section("4. GLOBAL OUTLIER DIAGNOSTICS")

q1 = valid_target.quantile(0.25)
q3 = valid_target.quantile(0.75)
iqr = q3 - q1

iqr_lower_15 = q1 - 1.5 * iqr
iqr_upper_15 = q3 + 1.5 * iqr

iqr_lower_30 = q1 - 3.0 * iqr
iqr_upper_30 = q3 + 3.0 * iqr

global_mod_z = modified_z_scores(target)

df["global_iqr_1_5_flag"] = (
    (target < iqr_lower_15) | (target > iqr_upper_15)
)

df["global_iqr_3_0_flag"] = (
    (target < iqr_lower_30) | (target > iqr_upper_30)
)

df["global_mad_flag"] = global_mod_z.abs() > 3.5
df["global_modified_z"] = global_mod_z

print(f"Q1                    : {q1:.6f}")
print(f"Q3                    : {q3:.6f}")
print(f"IQR                   : {iqr:.6f}")
print(f"1.5×IQR lower fence   : {iqr_lower_15:.6f}")
print(f"1.5×IQR upper fence   : {iqr_upper_15:.6f}")
print(f"3.0×IQR lower fence   : {iqr_lower_30:.6f}")
print(f"3.0×IQR upper fence   : {iqr_upper_30:.6f}")
print(f"1.5×IQR candidates    : {int(df['global_iqr_1_5_flag'].sum()):,}")
print(f"3.0×IQR candidates    : {int(df['global_iqr_3_0_flag'].sum()):,}")
print(f"MAD |modified z| >3.5 : {int(df['global_mad_flag'].sum()):,}")

print(
    "\nIMPORTANT: These are statistical outlier CANDIDATES only. "
    "They are not automatically errors and are not removed."
)


# =============================================================================
# 8. YEAR-SPECIFIC TARGET VALIDATION
# =============================================================================

section("5. YEAR-SPECIFIC TARGET STATISTICS")

year_rows = []
year_flag_frames = []

for year, group in df.groupby("year", sort=True):
    y = pd.to_numeric(group["LST_C"], errors="coerce")
    y_valid = y.replace([np.inf, -np.inf], np.nan).dropna()

    y_q1 = y_valid.quantile(0.25)
    y_q3 = y_valid.quantile(0.75)
    y_iqr = y_q3 - y_q1
    y_lower = y_q1 - 1.5 * y_iqr
    y_upper = y_q3 + 1.5 * y_iqr

    y_mod_z = modified_z_scores(y)

    year_rows.append(
        {
            "year": year,
            "count": int(y_valid.count()),
            "mean": y_valid.mean(),
            "median": y_valid.median(),
            "std": y_valid.std(),
            "min": y_valid.min(),
            "q01": y_valid.quantile(0.01),
            "q05": y_valid.quantile(0.05),
            "q25": y_q1,
            "q75": y_q3,
            "q95": y_valid.quantile(0.95),
            "q99": y_valid.quantile(0.99),
            "max": y_valid.max(),
            "iqr": y_iqr,
            "skewness": y_valid.skew(),
            "lower_iqr_fence": y_lower,
            "upper_iqr_fence": y_upper,
            "iqr_candidate_count": int(((y < y_lower) | (y > y_upper)).sum()),
            "mad_candidate_count": int((y_mod_z.abs() > 3.5).sum()),
        }
    )

    temp = group[["grid_id", "year", "LST_C"]].copy()
    if "area_name" in group.columns:
        temp["area_name"] = group["area_name"].values

    temp["year_iqr_lower_fence"] = y_lower
    temp["year_iqr_upper_fence"] = y_upper
    temp["year_iqr_flag"] = ((y < y_lower) | (y > y_upper)).values
    temp["year_modified_z"] = y_mod_z.values
    temp["year_mad_flag"] = (y_mod_z.abs() > 3.5).values

    year_flag_frames.append(temp)

year_stats = pd.DataFrame(year_rows)
year_stats.to_csv(YEAR_STATS_PATH, index=False)

print(year_stats.to_string(index=False))


# =============================================================================
# 9. AREA-LEVEL TARGET SUMMARY
# =============================================================================

section("6. AREA-LEVEL TARGET SUMMARY")

if "area_name" in df.columns:
    area_stats = (
        df.groupby("area_name", dropna=False)
        .agg(
            row_count=("LST_C", "size"),
            unique_grids=("grid_id", "nunique"),
            mean_lst=("LST_C", "mean"),
            median_lst=("LST_C", "median"),
            std_lst=("LST_C", "std"),
            min_lst=("LST_C", "min"),
            max_lst=("LST_C", "max"),
        )
        .reset_index()
        .sort_values("mean_lst", ascending=False)
    )

    area_stats.to_csv(AREA_STATS_PATH, index=False)

    print("Top 10 areas by mean LST:")
    print(area_stats.head(10).to_string(index=False))

    print("\nBottom 10 areas by mean LST:")
    print(area_stats.tail(10).sort_values("mean_lst").to_string(index=False))
else:
    area_stats = pd.DataFrame()
    print("area_name is not available. Area-level summary skipped.")


# =============================================================================
# 10. BUILD OUTLIER-CANDIDATE TABLE
# =============================================================================

section("7. OUTLIER-CANDIDATE TABLE")

year_flags = pd.concat(year_flag_frames, ignore_index=True)

key_cols = ["grid_id", "year", "LST_C"]
if "area_name" in df.columns:
    key_cols.append("area_name")

outlier_table = df[key_cols].copy()
outlier_table["global_iqr_1_5_flag"] = df["global_iqr_1_5_flag"].values
outlier_table["global_iqr_3_0_flag"] = df["global_iqr_3_0_flag"].values
outlier_table["global_modified_z"] = df["global_modified_z"].values
outlier_table["global_mad_flag"] = df["global_mad_flag"].values

merge_cols = ["grid_id", "year", "year_iqr_lower_fence", "year_iqr_upper_fence",
              "year_iqr_flag", "year_modified_z", "year_mad_flag"]

outlier_table = outlier_table.merge(
    year_flags[merge_cols],
    on=["grid_id", "year"],
    how="left",
    validate="one_to_one"
)

outlier_table["any_outlier_candidate"] = (
    outlier_table[
        [
            "global_iqr_1_5_flag",
            "global_iqr_3_0_flag",
            "global_mad_flag",
            "year_iqr_flag",
            "year_mad_flag",
        ]
    ]
    .fillna(False)
    .any(axis=1)
)

outlier_candidates = (
    outlier_table[outlier_table["any_outlier_candidate"]]
    .copy()
    .sort_values(["year", "LST_C"])
)

outlier_candidates.to_csv(OUTLIER_PATH, index=False)

print(f"Rows flagged by at least one diagnostic: {len(outlier_candidates):,}")
print(f"Saved candidate table: {OUTLIER_PATH}")


# =============================================================================
# 11. EXTREME VALUES FOR MANUAL INVESTIGATION
# =============================================================================

section("8. EXTREME TARGET VALUES")

extreme_cols = ["grid_id", "year", "LST_C"]
if "area_name" in df.columns:
    extreme_cols.append("area_name")

lowest_25 = df.nsmallest(25, "LST_C")[extreme_cols].copy()
lowest_25["extreme_type"] = "lowest_25"

highest_25 = df.nlargest(25, "LST_C")[extreme_cols].copy()
highest_25["extreme_type"] = "highest_25"

extremes = pd.concat([lowest_25, highest_25], ignore_index=True)
extremes.to_csv(EXTREMES_PATH, index=False)

print("\nLowest 10 LST observations:")
print(lowest_25.head(10).to_string(index=False))

print("\nHighest 10 LST observations:")
print(highest_25.head(10).to_string(index=False))


# =============================================================================
# 12. YEAR-TO-YEAR TARGET DISTRIBUTION DRIFT
# =============================================================================

section("9. YEAR-TO-YEAR TARGET DISTRIBUTION DRIFT")

years = sorted(df["year"].dropna().unique().tolist())
drift_rows = []

for previous_year, current_year in zip(years[:-1], years[1:]):
    previous = (
        pd.to_numeric(
            df.loc[df["year"] == previous_year, "LST_C"],
            errors="coerce"
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    current = (
        pd.to_numeric(
            df.loc[df["year"] == current_year, "LST_C"],
            errors="coerce"
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    ks_stat, ks_p = stats.ks_2samp(previous, current)

    drift_rows.append(
        {
            "from_year": previous_year,
            "to_year": current_year,
            "mean_change": current.mean() - previous.mean(),
            "median_change": current.median() - previous.median(),
            "std_change": current.std() - previous.std(),
            "wasserstein_distance": wasserstein_distance(previous, current),
            "ks_statistic": ks_stat,
            "ks_p_value": ks_p,
        }
    )

year_drift = pd.DataFrame(drift_rows)
year_drift.to_csv(YEAR_DRIFT_PATH, index=False)

print(year_drift.to_string(index=False))

print(
    "\nNOTE: KS p-values are diagnostic only. With 2,194 observations per year, "
    "small distribution differences may be statistically significant."
)


# =============================================================================
# 13. GRID-LEVEL YEAR-TO-YEAR LST CHANGES
# =============================================================================

section("10. GRID-LEVEL YEAR-TO-YEAR LST CHANGES")

grid_change_cols = ["grid_id", "year", "LST_C"]
if "area_name" in df.columns:
    grid_change_cols.append("area_name")

grid_changes = (
    df[grid_change_cols]
    .sort_values(["grid_id", "year"])
    .copy()
)

grid_changes["previous_year"] = grid_changes.groupby("grid_id")["year"].shift(1)
grid_changes["previous_lst"] = grid_changes.groupby("grid_id")["LST_C"].shift(1)
grid_changes["lst_change"] = grid_changes["LST_C"] - grid_changes["previous_lst"]
grid_changes["abs_lst_change"] = grid_changes["lst_change"].abs()

grid_changes.to_csv(GRID_CHANGE_PATH, index=False)

valid_changes = grid_changes["lst_change"].dropna()

print(f"Year-to-year grid comparisons: {valid_changes.count():,}")
print(f"Mean LST change              : {valid_changes.mean():.6f}")
print(f"Median LST change            : {valid_changes.median():.6f}")
print(f"Mean absolute change         : {valid_changes.abs().mean():.6f}")
print(f"95th pct absolute change     : {valid_changes.abs().quantile(0.95):.6f}")
print(f"99th pct absolute change     : {valid_changes.abs().quantile(0.99):.6f}")
print(f"Maximum absolute change      : {valid_changes.abs().max():.6f}")

largest_changes = (
    grid_changes.dropna(subset=["abs_lst_change"])
    .nlargest(100, "abs_lst_change")
    .copy()
)

largest_changes.to_csv(TOP_GRID_CHANGE_PATH, index=False)

print("\nTop 10 largest absolute grid-level year-to-year changes:")
print(largest_changes.head(10).to_string(index=False))


# =============================================================================
# 14. FIGURES
# =============================================================================

section("11. GENERATING TARGET VALIDATION FIGURES")

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(valid_target, bins=50)
plt.xlabel("Land Surface Temperature (°C)")
plt.ylabel("Frequency")
plt.title("Distribution of LST_C - 2015 to 2025")
plt.grid(alpha=0.2)
save_figure(HIST_PATH)

# Overall boxplot
plt.figure(figsize=(8, 5))
plt.boxplot(valid_target, vert=False)
plt.xlabel("Land Surface Temperature (°C)")
plt.title("Overall LST_C Boxplot")
plt.grid(axis="x", alpha=0.2)
save_figure(BOX_PATH)

# Yearly boxplot
year_data = [
    df.loc[df["year"] == year, "LST_C"].dropna().values
    for year in years
]

plt.figure(figsize=(12, 6))
plt.boxplot(year_data, tick_labels=[str(year) for year in years])
plt.xlabel("Year")
plt.ylabel("Land Surface Temperature (°C)")
plt.title("Year-wise Distribution of LST_C")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.2)
save_figure(YEAR_BOX_PATH)

# Mean and median by year
plt.figure(figsize=(10, 6))
plt.plot(year_stats["year"], year_stats["mean"], marker="o", label="Mean")
plt.plot(year_stats["year"], year_stats["median"], marker="o", label="Median")
plt.xlabel("Year")
plt.ylabel("Land Surface Temperature (°C)")
plt.title("Yearly Mean and Median LST_C")
plt.legend()
plt.grid(alpha=0.2)
save_figure(YEAR_TREND_PATH)

# Q-Q plot
plt.figure(figsize=(7, 7))
stats.probplot(valid_target, dist="norm", plot=plt)
plt.title("Q-Q Plot of LST_C")
plt.grid(alpha=0.2)
save_figure(QQ_PATH)

print(f"Saved: {HIST_PATH}")
print(f"Saved: {BOX_PATH}")
print(f"Saved: {YEAR_BOX_PATH}")
print(f"Saved: {YEAR_TREND_PATH}")
print(f"Saved: {QQ_PATH}")


# =============================================================================
# 15. BUILD TEXT REPORT
# =============================================================================

section("12. SAVING TARGET VALIDATION REPORT")

report_lines = [
    "STEP 02 - TARGET VALIDATION REPORT",
    "=" * 80,
    "",
    f"Input file: {RAW_DATA_PATH}",
    "Target variable: LST_C",
    "",
    "TARGET INTEGRITY",
    "-" * 80,
    f"Rows: {len(df):,}",
    f"Numeric parse failures: {parse_failures:,}",
    f"Missing target values: {missing_count:,}",
    f"Positive infinity count: {positive_inf_count:,}",
    f"Negative infinity count: {negative_inf_count:,}",
    f"Finite target values: {finite_count:,}",
    "",
    "OVERALL TARGET STATISTICS",
    "-" * 80,
    f"Mean: {valid_target.mean():.6f}",
    f"Median: {valid_target.median():.6f}",
    f"Std: {valid_target.std():.6f}",
    f"Min: {valid_target.min():.6f}",
    f"Q01: {valid_target.quantile(0.01):.6f}",
    f"Q05: {valid_target.quantile(0.05):.6f}",
    f"Q25: {q1:.6f}",
    f"Q75: {q3:.6f}",
    f"Q95: {valid_target.quantile(0.95):.6f}",
    f"Q99: {valid_target.quantile(0.99):.6f}",
    f"Max: {valid_target.max():.6f}",
    f"IQR: {iqr:.6f}",
    f"Skewness: {valid_target.skew():.6f}",
    f"Excess kurtosis: {valid_target.kurt():.6f}",
    "",
    "GLOBAL OUTLIER DIAGNOSTICS",
    "-" * 80,
    f"1.5x IQR lower fence: {iqr_lower_15:.6f}",
    f"1.5x IQR upper fence: {iqr_upper_15:.6f}",
    f"1.5x IQR candidates: {int(df['global_iqr_1_5_flag'].sum()):,}",
    f"3.0x IQR candidates: {int(df['global_iqr_3_0_flag'].sum()):,}",
    f"MAD candidates (|modified z| > 3.5): {int(df['global_mad_flag'].sum()):,}",
    f"Rows flagged by at least one diagnostic: {len(outlier_candidates):,}",
    "",
    "NORMALITY DIAGNOSTIC",
    "-" * 80,
    f"D'Agostino K2 statistic: {normaltest_stat}",
    f"D'Agostino p-value: {normaltest_p}",
    "Normality is NOT required for the planned tree-based regression models.",
    "",
    "IMPORTANT INTERPRETATION RULE",
    "-" * 80,
    "Statistical outliers are investigation candidates, not automatic errors.",
    "No LST_C value was deleted, capped, winsorized, transformed, or imputed in Step 02.",
    "Extreme observations should be checked using year, grid, area, and spatial context.",
    "",
    "TEMPORAL VALIDATION",
    "-" * 80,
    f"Years assessed: {years}",
    f"Year-to-year grid comparisons: {valid_changes.count():,}",
    f"Mean absolute grid-level yearly change: {valid_changes.abs().mean():.6f}",
    f"95th percentile absolute yearly change: {valid_changes.abs().quantile(0.95):.6f}",
    f"99th percentile absolute yearly change: {valid_changes.abs().quantile(0.99):.6f}",
    f"Maximum absolute yearly change: {valid_changes.abs().max():.6f}",
    "",
    "MODEL-DEVELOPMENT SAFEGUARD",
    "-" * 80,
    "This step validates the raw target only.",
    "No train/validation/test split, tuning, feature selection, or model fitting was performed.",
    "The final 2025 lockbox rule still applies during model development.",
]

TARGET_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

print(f"Saved: {TARGET_REPORT_PATH}")
print(f"Saved: {OVERALL_STATS_PATH}")
print(f"Saved: {YEAR_STATS_PATH}")
if not area_stats.empty:
    print(f"Saved: {AREA_STATS_PATH}")
print(f"Saved: {OUTLIER_PATH}")
print(f"Saved: {EXTREMES_PATH}")
print(f"Saved: {YEAR_DRIFT_PATH}")
print(f"Saved: {GRID_CHANGE_PATH}")
print(f"Saved: {TOP_GRID_CHANGE_PATH}")


# =============================================================================
# 16. FINAL STATUS
# =============================================================================

section("STEP 02 COMPLETED SUCCESSFULLY")

print(
    "Target validation finished without modifying the raw dataset.\n"
    "Review the terminal output, outlier candidates, extreme values, yearly statistics,\n"
    "target drift, grid-level year-to-year changes, and figures before Step 03."
)
