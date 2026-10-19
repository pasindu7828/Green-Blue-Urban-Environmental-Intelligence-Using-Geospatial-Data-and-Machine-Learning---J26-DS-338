
"""
J26-DS-338 - River Corridor Research

STEP 14: MATCHED-TRANSECT HISTORICAL CHANGE QA

Input:
    data/processed/morphology/08_pilot_widths.csv

Purpose:
    Compare pilot widths only at matching
    transect locations with acceptable
    preliminary screening status.

Important:
    These measurements are NOT validated
    riverbank changes or ML training labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data/processed/morphology"
    / "08_pilot_widths.csv"
)

OUTPUT_DIR = (
    ROOT / "data/processed/morphology"
)

REPORT_DIR = ROOT / "outputs/reports"
FIGURE_DIR = ROOT / "outputs/figures"

for folder in (OUTPUT_DIR, REPORT_DIR, FIGURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

YEARS = [2010, 2015, 2020, 2025]

PAIRS = [
    (2010, 2015),
    (2015, 2020),
    (2020, 2025),
    (2010, 2025)
]

SCREENING_STATUS = "PILOT_UNVALIDATED"


# ==========================================
# 2. READ INPUT DATA
# ==========================================

print("=" * 75)
print("STEP 14 - MATCHED TRANSECT CHANGE QA")
print("=" * 75)

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "year",
    "transect_id",
    "reach_id",
    "chainage_m",
    "pilot_width_m",
    "qa_status",
    "validated",
    "ml_training_approved"
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing input columns: {missing}"
    )

print(f"Input records: {len(df)}")

if df.duplicated(
    ["year", "transect_id"]
).any():
    raise ValueError(
        "Duplicate year-transect records detected."
    )

if set(df["year"]) != set(YEARS):
    raise ValueError(
        "Unexpected or missing observation years."
    )

# Refuse to reinterpret unvalidated pilot
# measurements as confirmed ML labels.
for column in [
    "validated",
    "ml_training_approved"
]:
    flag = (
        df[column]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    if flag.any():
        raise ValueError(
            f"Unexpected TRUE flag in {column}."
        )

# Every selected year should contain
# the same set of transect IDs.
reference_ids = set(
    df.loc[
        df["year"] == YEARS[0],
        "transect_id"
    ]
)

for year in YEARS:

    ids = set(
        df.loc[
            df["year"] == year,
            "transect_id"
        ]
    )

    if ids != reference_ids:
        raise ValueError(
            f"Transect IDs differ in {year}."
        )

# Check that the location metadata
# remains unchanged between years.
for column in ["reach_id", "chainage_m"]:

    inconsistent = (
        df.groupby("transect_id")[column]
        .nunique()
        .gt(1)
    )

    if inconsistent.any():
        raise ValueError(
            f"Inconsistent {column} values."
        )

print(
    f"Unique transects: {len(reference_ids)}"
)


# ==========================================
# 3. CREATE MATCHED YEAR TABLES
# ==========================================

width = df.pivot(
    index="transect_id",
    columns="year",
    values="pilot_width_m"
)

status = df.pivot(
    index="transect_id",
    columns="year",
    values="qa_status"
)

metadata = (
    df[df["year"] == YEARS[0]]
    .set_index("transect_id")[
        ["reach_id", "chainage_m"]
    ]
)

result = metadata.copy()

for year in YEARS:

    result[f"width_{year}_m"] = width[year]

    result[f"qa_{year}"] = status[year]

    result[f"screened_{year}"] = (
        (status[year] == SCREENING_STATUS)
        & width[year].notna()
        & (width[year] >= 60.0)
    )


# ==========================================
# 4. FIND FOUR-YEAR MATCHED TRANSECTS
# ==========================================

screen_columns = [
    f"screened_{year}"
    for year in YEARS
]

result["all_four_years_screened"] = (
    result[screen_columns].all(axis=1)
)

all_four_count = int(
    result["all_four_years_screened"].sum()
)

print(
    "\nTransects passing pilot screening "
    f"in all four years: {all_four_count}"
)


# ==========================================
# 5. PAIRED HISTORICAL COMPARISONS
# ==========================================

summary_records = []

for start, end in PAIRS:

    print("\n" + "-" * 75)
    print(f"COMPARISON: {start} -> {end}")

    paired_valid = (
        result[f"screened_{start}"]
        & result[f"screened_{end}"]
    )

    difference = (
        width[end] - width[start]
    )

    difference = difference.where(
        paired_valid
    )

    column = f"pilot_change_{start}_{end}_m"

    result[column] = difference

    valid_changes = difference.dropna()

    matched_count = len(valid_changes)

    zero_count = int(
        np.isclose(
            valid_changes.to_numpy(),
            0.0,
            atol=0.05
        ).sum()
    )

    if matched_count:

        median_change = float(
            valid_changes.median()
        )

        mean_change = float(
            valid_changes.mean()
        )

        p25 = float(
            valid_changes.quantile(0.25)
        )

        p75 = float(
            valid_changes.quantile(0.75)
        )

        unchanged_pct = (
            100.0 * zero_count / matched_count
        )

    else:

        median_change = np.nan
        mean_change = np.nan
        p25 = np.nan
        p75 = np.nan
        unchanged_pct = np.nan

    summary_records.append({
        "comparison": f"{start}_{end}",
        "start_year": start,
        "end_year": end,
        "total_transects": len(result),
        "matched_screened_transects": matched_count,
        "excluded_or_flagged": (
            len(result) - matched_count
        ),
        "median_pilot_change_m": median_change,
        "mean_pilot_change_m": mean_change,
        "p25_pilot_change_m": p25,
        "p75_pilot_change_m": p75,
        "near_zero_change_pct": unchanged_pct,
        "resolution_m": 30,
        "interpretation": (
            "PIXEL_BASED_PILOT_NOT_VALIDATED"
        ),
        "ml_training_approved": False
    })

    print(
        f"Matched screened transects: "
        f"{matched_count}"
    )

    print(
        f"Excluded/flagged: "
        f"{len(result) - matched_count}"
    )

    print(
        f"Median pilot change: "
        f"{median_change:.2f} m"
    )


# ==========================================
# 6. SAVE MATCHED-TRANSECT DATASET
# ==========================================

result = result.reset_index()

matched_path = (
    OUTPUT_DIR
    / "09_matched_transect_widths.csv"
)

result.to_csv(
    matched_path,
    index=False
)

summary = pd.DataFrame(
    summary_records
)

qa_path = (
    REPORT_DIR
    / "09_matched_change_qa.csv"
)

summary.to_csv(
    qa_path,
    index=False
)


# ==========================================
# 7. GENERATE QA FIGURE
# ==========================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.bar(
    summary["comparison"],
    summary["matched_screened_transects"]
)

ax.set_title(
    "Matched Transects Passing Pilot Screening"
)

ax.set_ylabel(
    "Number of matched transects"
)

ax.set_xlabel(
    "Observation year comparison"
)

ax.set_ylim(
    0,
    len(result) + 15
)

for index, count in enumerate(
    summary["matched_screened_transects"]
):

    ax.text(
        index,
        count + 4,
        str(count),
        ha="center"
    )

fig.tight_layout()

figure_path = (
    FIGURE_DIR / "09_matched_change_qa.png"
)

fig.savefig(
    figure_path,
    dpi=150
)

plt.close(fig)


# ==========================================
# 8. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 75)
print("STEP 14 - FINAL SUMMARY")
print("=" * 75)

print(
    f"Total transects: {len(result)}"
)

print(
    "All-four-year screened transects: "
    f"{all_four_count}"
)

print("\nPaired comparisons:")

print(
    summary[
        [
            "comparison",
            "matched_screened_transects",
            "median_pilot_change_m",
            "near_zero_change_pct"
        ]
    ].to_string(index=False)
)

print("\nSaved outputs:")
print(matched_path)
print(qa_path)
print(figure_path)

print("\nIMPORTANT:")
print("Width differences are preliminary.")
print("They are below or near Landsat resolution.")
print("No physical erosion claim is supported.")
print("ML training is NOT approved.")

print("\nSTEP 14 FINISHED")
print("=" * 75)
