from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[4]

BASE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "kaduwela_preprocessed_base.csv"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "validation"
    / "scenario_validation"
    / "outputs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = ROOT / "outputs" / "reports"


# ============================================================
# SETTINGS
# ============================================================

GRID_COL = "grid_id"
YEAR_COL = "year"

EXPECTED_GRIDS = 2194
EXPECTED_YEARS = list(range(2015, 2026))
EXPECTED_ROWS = EXPECTED_GRIDS * len(EXPECTED_YEARS)
EXPECTED_TRANSITIONS = EXPECTED_GRIDS * 10

REQUIRED_COLUMNS = [
    GRID_COL,
    YEAR_COL,
    "LST_C",
    "green_mask",
    "NDBI",
    "NDWI",
]


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

print("\n============================================================")
print("SCENARIO VALIDATION 01")
print("HISTORICAL SCENARIO SUPPORT VALIDATION")
print("============================================================\n")

df = pd.read_csv(BASE_PATH, low_memory=False)

missing = [
    col
    for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

df = df[REQUIRED_COLUMNS].copy()

df[GRID_COL] = df[GRID_COL].astype(str).str.strip()
df[YEAR_COL] = pd.to_numeric(
    df[YEAR_COL],
    errors="raise",
).astype(int)

df = df[
    df[YEAR_COL].between(2015, 2025)
].copy()

print(f"Historical rows : {len(df):,}")
print(f"Unique grids    : {df[GRID_COL].nunique():,}")
print(f"Years           : {sorted(df[YEAR_COL].unique())}")

if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} historical rows, "
        f"found {len(df):,}."
    )

if df[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Expected {EXPECTED_GRIDS} grids."
    )


# ============================================================
# BUILD TRUE ADJACENT-YEAR TRANSITIONS
# ============================================================

df = df.sort_values(
    [GRID_COL, YEAR_COL]
).reset_index(drop=True)

df["previous_year"] = (
    df.groupby(GRID_COL)[YEAR_COL].shift(1)
)

df["d_green"] = (
    df.groupby(GRID_COL)["green_mask"].diff()
)

df["d_ndbi"] = (
    df.groupby(GRID_COL)["NDBI"].diff()
)

df["d_ndwi"] = (
    df.groupby(GRID_COL)["NDWI"].diff()
)

df["d_lst_c"] = (
    df.groupby(GRID_COL)["LST_C"].diff()
)

transitions = df[
    df[YEAR_COL].between(2016, 2025)
].copy()

transitions["year_gap"] = (
    transitions[YEAR_COL]
    - transitions["previous_year"]
)

if not (transitions["year_gap"] == 1).all():
    raise RuntimeError(
        "Non-adjacent historical transitions detected."
    )

if len(transitions) != EXPECTED_TRANSITIONS:
    raise RuntimeError(
        f"Expected {EXPECTED_TRANSITIONS:,} transitions, "
        f"found {len(transitions):,}."
    )

print(
    f"Adjacent annual transitions : "
    f"{len(transitions):,}"
)


# ============================================================
# HISTORICAL INTERVENTION MAGNITUDE DISTRIBUTIONS
# ============================================================

# Green scenario is an increase in green cover.
green_positive_pp = (
    transitions.loc[
        transitions["d_green"] > 0,
        "d_green",
    ]
    * 100.0
)

# Step 29B derives NDBI scenario levels from
# absolute annual within-grid NDBI changes.
ndbi_abs_change = (
    transitions["d_ndbi"]
    .abs()
    .dropna()
)

# Step 29D derives NDWI scenario levels from
# positive annual within-grid NDWI changes only.
ndwi_positive_change = (
    transitions.loc[
        transitions["d_ndwi"] > 0,
        "d_ndwi",
    ]
    .dropna()
)


def distribution_summary(name, values):
    values = pd.Series(values).dropna()

    return {
        "variable": name,
        "sample_count": len(values),
        "mean": values.mean(),
        "median_q50": values.quantile(0.50),
        "q75": values.quantile(0.75),
        "q90": values.quantile(0.90),
        "q95": values.quantile(0.95),
        "maximum": values.max(),
    }


distribution_df = pd.DataFrame(
    [
        distribution_summary(
            "positive_green_change_pp",
            green_positive_pp,
        ),
        distribution_summary(
            "absolute_ndbi_change",
            ndbi_abs_change,
        ),
        distribution_summary(
            "positive_ndwi_change",
            ndwi_positive_change,
        ),
    ]
)

distribution_path = (
    OUTPUT_DIR
    / "01_historical_change_distribution_summary.csv"
)

distribution_df.to_csv(
    distribution_path,
    index=False,
)


# ============================================================
# GREEN SCENARIO SUPPORT
# ============================================================

green_rows = []

for level in [5, 10, 20]:

    count_ge = int(
        (green_positive_pp >= level).sum()
    )

    support_pct = (
        count_ge
        / len(green_positive_pp)
        * 100.0
    )

    percentile_position = (
        (green_positive_pp <= level).mean()
        * 100.0
    )

    green_rows.append(
        {
            "scenario_family": "GREEN",
            "scenario_level": f"+{level}pp",
            "requested_magnitude": level,
            "historical_positive_change_count":
                len(green_positive_pp),
            "historical_transitions_at_or_above_level":
                count_ge,
            "historical_support_pct":
                support_pct,
            "historical_percentile_position":
                percentile_position,
            "within_observed_global_max":
                bool(
                    level
                    <= green_positive_pp.max()
                ),
        }
    )

green_support_df = pd.DataFrame(green_rows)


# ============================================================
# NDBI LEVEL REPRODUCTION
# ============================================================

quantiles = {
    "LOW": 0.50,
    "MODERATE": 0.75,
    "HIGH": 0.90,
}

ndbi_rows = []

for level, q in quantiles.items():

    magnitude = float(
        ndbi_abs_change.quantile(q)
    )

    ndbi_rows.append(
        {
            "scenario_family": "BUILT_NDBI",
            "scenario_level": level,
            "source_quantile": q,
            "validated_magnitude": magnitude,
            "historical_sample_count":
                len(ndbi_abs_change),
        }
    )

ndbi_validation_df = pd.DataFrame(
    ndbi_rows
)


# ============================================================
# NDWI LEVEL REPRODUCTION
# ============================================================

ndwi_rows = []

for level, q in quantiles.items():

    magnitude = float(
        ndwi_positive_change.quantile(q)
    )

    ndwi_rows.append(
        {
            "scenario_family": "BLUE_NDWI",
            "scenario_level": level,
            "source_quantile": q,
            "validated_magnitude": magnitude,
            "historical_sample_count":
                len(ndwi_positive_change),
        }
    )

ndwi_validation_df = pd.DataFrame(
    ndwi_rows
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

green_path = (
    OUTPUT_DIR
    / "01_green_scenario_historical_support.csv"
)

ndbi_path = (
    OUTPUT_DIR
    / "01_ndbi_scenario_level_validation.csv"
)

ndwi_path = (
    OUTPUT_DIR
    / "01_ndwi_scenario_level_validation.csv"
)

transition_path = (
    OUTPUT_DIR
    / "01_historical_adjacent_transitions.csv"
)

green_support_df.to_csv(
    green_path,
    index=False,
)

ndbi_validation_df.to_csv(
    ndbi_path,
    index=False,
)

ndwi_validation_df.to_csv(
    ndwi_path,
    index=False,
)

transitions[
    [
        GRID_COL,
        YEAR_COL,
        "previous_year",
        "d_green",
        "d_ndbi",
        "d_ndwi",
        "d_lst_c",
    ]
].to_csv(
    transition_path,
    index=False,
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("\n============================================================")
print("GREEN SCENARIO HISTORICAL SUPPORT")
print("============================================================")
print(
    green_support_df.to_string(index=False)
)

print("\n============================================================")
print("NDBI SCENARIO LEVEL VALIDATION")
print("============================================================")
print(
    ndbi_validation_df.to_string(index=False)
)

print("\n============================================================")
print("NDWI SCENARIO LEVEL VALIDATION")
print("============================================================")
print(
    ndwi_validation_df.to_string(index=False)
)

print("\n============================================================")
print("VALIDATION STATUS")
print("============================================================")

print(
    f"PASS - balanced historical panel: "
    f"{EXPECTED_GRIDS} grids x 11 years"
)

print(
    f"PASS - true adjacent transitions: "
    f"{len(transitions):,}"
)

print(
    "PASS - NDBI levels independently "
    "re-derived from historical data"
)

print(
    "PASS - NDWI levels independently "
    "re-derived from historical data"
)

print(
    "PASS - green planning increments "
    "compared against historical support"
)

print("\nSaved outputs:")
print(distribution_path)
print(green_path)
print(ndbi_path)
print(ndwi_path)
print(transition_path)

print("\nSCENARIO SUPPORT VALIDATION COMPLETE.")