
"""
STEP 29D-QA - BLUE-SPACE / NDWI SUPPORT AUDIT
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Step 29D itself completed successfully, but its LOW/MODERATE/HIGH NDWI
increments were derived from the pooled city-wide distribution of positive
annual within-grid NDWI changes.

This QA checks whether those increments are also supported at the INDIVIDUAL
GRID level.

It does NOT change any Step-29D predictions, does NOT retrain any model and
does NOT tune scenario magnitudes after looking at LST results.

Checks
------
1. Verify frozen historical data and finalized 29D outputs/audit.
2. Rebuild historical adjacent annual NDWI changes for every grid.
3. Compute per-grid positive-change support:
      - positive transition count
      - max positive annual NDWI increase
      - q90 positive annual NDWI increase (where estimable)
4. Compare every 29D scenario increment against that grid's own historical
   positive-change support.
5. Compare baseline/scenario NDWI against each grid's historical NDWI range.
6. Separate:
      - baseline already outside historical range
      - scenario newly pushes a previously in-support baseline outside range
7. Save year/level and overall support summaries.
8. Do not alter 29D outputs.

Interpretation
--------------
A scenario can still be shown in the frontend if it is outside a grid's own
historical support, but it should be labelled lower-confidence / extrapolative
for that selected grid rather than silently treated as equally supported.
"""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

GRID_COL = "grid_id"
YEAR_COL = "year"
TREATMENT = "NDWI"

EXPECTED_GRIDS = 2194
EXPECTED_HIST_YEARS = list(range(2015, 2026))
EXPECTED_FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]
EXPECTED_SCENARIO_LEVELS = ["LOW", "MODERATE", "HIGH"]
EXPECTED_SCENARIO_ROWS = 2194 * 5 * 3

EXPECTED_HASHES = {
    "kaduwela_preprocessed_base.csv":
        "0e030b2ec0b5c4080cd4226bec798189a178aa45efd9962d19f38dc9cb170f46",
    "29d_blue_space_ndwi_scenarios_2026_2030.csv":
        "328421e5b4d3c36c0aef7747009573a5bb39f03bdb83c9c5a88d923e10a27521",
    "29d_blue_space_ndwi_scenario_levels.csv":
        "3bf7a80bb9c110efa7ec9ef54f197ad80d6d44d42cff7ed817df639614a9e94a",
    "29d_blue_space_scenario_audit.csv":
        "97c3f0a25ace0ecb18803653ec5b9c2c2f1f84b49cc29331b91e3b0d78944891",
}

ROOT = Path(__file__).resolve().parents[1]

BASE_PATH = ROOT / "data" / "processed" / "kaduwela_preprocessed_base.csv"
SCENARIO_PATH = ROOT / "outputs" / "scenarios" / "29d_blue_space_ndwi_scenarios_2026_2030.csv"
LEVEL_PATH = ROOT / "outputs" / "reports" / "29d_blue_space_ndwi_scenario_levels.csv"
STEP29D_AUDIT_PATH = ROOT / "outputs" / "reports" / "29d_blue_space_scenario_audit.csv"

REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

GRID_SUPPORT_PATH = REPORT_DIR / "29d_qa_grid_ndwi_support.csv"
ROW_AUDIT_PATH = REPORT_DIR / "29d_qa_scenario_row_support.csv"
YEAR_LEVEL_SUMMARY_PATH = REPORT_DIR / "29d_qa_year_level_support_summary.csv"
OVERALL_SUMMARY_PATH = REPORT_DIR / "29d_qa_overall_support_summary.csv"
QA_AUDIT_PATH = REPORT_DIR / "29d_qa_support_audit.csv"
METHOD_PATH = REPORT_DIR / "29d_qa_support_methodology.json"
REPORT_PATH = REPORT_DIR / "29d_qa_support_report.txt"
HASH_PATH = REPORT_DIR / "29d_qa_support_hashes.csv"


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


def verify_hash(path, expected, label):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual  : {actual}"
        )

    print(f"PASS hash: {label} = {actual}")
    return actual


section("STEP 29D-QA - BLUE-SPACE / NDWI GRID-LEVEL SUPPORT AUDIT")

verify_hash(
    BASE_PATH,
    EXPECTED_HASHES["kaduwela_preprocessed_base.csv"],
    "Canonical historical base",
)

verify_hash(
    SCENARIO_PATH,
    EXPECTED_HASHES["29d_blue_space_ndwi_scenarios_2026_2030.csv"],
    "Final Step-29D scenario output",
)

verify_hash(
    LEVEL_PATH,
    EXPECTED_HASHES["29d_blue_space_ndwi_scenario_levels.csv"],
    "Final Step-29D scenario levels",
)

verify_hash(
    STEP29D_AUDIT_PATH,
    EXPECTED_HASHES["29d_blue_space_scenario_audit.csv"],
    "Final Step-29D strict audit",
)

audit29d = pd.read_csv(STEP29D_AUDIT_PATH)

if "status" not in audit29d.columns:
    raise RuntimeError("29D audit has no status column.")

if (
    audit29d["status"].astype(str).str.upper()
    != "PASS"
).any():
    raise RuntimeError("29D audit is not fully PASS.")

print("PASS: final Step-29D strict audit is fully PASS.")


# =============================================================================
# 1. LOAD + VERIFY HISTORICAL NDWI
# =============================================================================

section("1. HISTORICAL NDWI SUPPORT DATA")

base = pd.read_csv(
    BASE_PATH,
    dtype={GRID_COL: str},
)

base[YEAR_COL] = pd.to_numeric(
    base[YEAR_COL],
    errors="raise",
).astype(int)

base[TREATMENT] = pd.to_numeric(
    base[TREATMENT],
    errors="raise",
)

hist = (
    base[
        base[YEAR_COL].isin(
            EXPECTED_HIST_YEARS
        )
    ][
        [
            GRID_COL,
            YEAR_COL,
            TREATMENT,
        ]
    ]
    .copy()
    .sort_values(
        [
            GRID_COL,
            YEAR_COL,
        ]
    )
    .reset_index(drop=True)
)

if hist[GRID_COL].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Historical grids={hist[GRID_COL].nunique()}, expected={EXPECTED_GRIDS}"
    )

if len(hist) != EXPECTED_GRIDS * len(EXPECTED_HIST_YEARS):
    raise RuntimeError("Historical NDWI row count mismatch.")

years_per_grid = hist.groupby(GRID_COL)[YEAR_COL].nunique()

if not (
    years_per_grid
    == len(EXPECTED_HIST_YEARS)
).all():
    raise RuntimeError("Historical NDWI panel is not balanced 2015-2025.")

hist["previous_year"] = (
    hist.groupby(GRID_COL)[YEAR_COL]
    .shift(1)
)

hist["d_ndwi"] = (
    hist.groupby(GRID_COL)[TREATMENT]
    .diff()
)

transitions = hist[
    hist[YEAR_COL].between(2016, 2025)
].copy()

if not (
    transitions[YEAR_COL]
    - transitions["previous_year"]
    == 1
).all():
    raise RuntimeError("Historical differences are not adjacent-year changes.")

pooled_positive = transitions.loc[
    transitions["d_ndwi"] > 0,
    "d_ndwi",
].dropna()

pooled_support = {
    "positive_transition_count":
        int(len(pooled_positive)),
    "q50":
        float(pooled_positive.quantile(0.50)),
    "q75":
        float(pooled_positive.quantile(0.75)),
    "q90":
        float(pooled_positive.quantile(0.90)),
    "q95":
        float(pooled_positive.quantile(0.95)),
    "q99":
        float(pooled_positive.quantile(0.99)),
    "max":
        float(pooled_positive.max()),
}

print("Pooled positive annual NDWI-change support:")
for key, value in pooled_support.items():
    print(f"  {key}: {value}")


# =============================================================================
# 2. PER-GRID SUPPORT
# =============================================================================

section("2. PER-GRID HISTORICAL SUPPORT")

hist_range = (
    hist.groupby(GRID_COL)[TREATMENT]
    .agg(
        hist_ndwi_min="min",
        hist_ndwi_max="max",
        hist_ndwi_mean="mean",
        hist_ndwi_std="std",
    )
    .reset_index()
)

positive = transitions[
    transitions["d_ndwi"] > 0
].copy()

support_rows = []

for grid_id, group in transitions.groupby(GRID_COL, sort=False):
    pos = group.loc[
        group["d_ndwi"] > 0,
        "d_ndwi",
    ].dropna()

    support_rows.append(
        {
            GRID_COL:
                str(grid_id),
            "annual_transition_count":
                int(group["d_ndwi"].notna().sum()),
            "positive_transition_count":
                int(len(pos)),
            "max_positive_annual_ndwi_change":
                float(pos.max())
                if len(pos) > 0
                else np.nan,
            "q90_positive_annual_ndwi_change":
                float(pos.quantile(0.90))
                if len(pos) > 0
                else np.nan,
            "median_positive_annual_ndwi_change":
                float(pos.median())
                if len(pos) > 0
                else np.nan,
        }
    )

support_df = pd.DataFrame(support_rows)

grid_support = hist_range.merge(
    support_df,
    on=GRID_COL,
    how="left",
    validate="one_to_one",
)

grid_support["has_positive_historical_ndwi_change"] = (
    grid_support["positive_transition_count"] > 0
)

grid_support.to_csv(
    GRID_SUPPORT_PATH,
    index=False,
)

print(
    "Grids with >=1 positive historical annual NDWI change: "
    f"{int(grid_support['has_positive_historical_ndwi_change'].sum()):,} / "
    f"{EXPECTED_GRIDS:,}"
)

print(
    "Grids with no positive historical annual NDWI change: "
    f"{int((~grid_support['has_positive_historical_ndwi_change']).sum()):,}"
)


# =============================================================================
# 3. SCENARIO-ROW SUPPORT AUDIT
# =============================================================================

section("3. AUDIT EVERY STEP-29D SCENARIO ROW")

scenario = pd.read_csv(
    SCENARIO_PATH,
    dtype={GRID_COL: str},
)

scenario[YEAR_COL] = pd.to_numeric(
    scenario[YEAR_COL],
    errors="raise",
).astype(int)

for col in [
    "requested_ndwi_increase",
    "baseline_ndwi",
    "scenario_ndwi",
    "actual_ndwi_increase",
]:
    scenario[col] = pd.to_numeric(
        scenario[col],
        errors="raise",
    )

scenario["scenario_level"] = (
    scenario["scenario_level"]
    .astype(str)
    .str.upper()
)

if len(scenario) != EXPECTED_SCENARIO_ROWS:
    raise RuntimeError(
        f"Scenario rows={len(scenario)}, expected={EXPECTED_SCENARIO_ROWS}"
    )

if set(scenario[YEAR_COL].unique()) != set(EXPECTED_FUTURE_YEARS):
    raise RuntimeError("Unexpected future years in 29D scenario output.")

if set(scenario["scenario_level"].unique()) != set(EXPECTED_SCENARIO_LEVELS):
    raise RuntimeError("Unexpected scenario levels in 29D output.")

row_audit = scenario.merge(
    grid_support,
    on=GRID_COL,
    how="left",
    validate="many_to_one",
)

if row_audit[
    "hist_ndwi_min"
].isna().any():
    raise RuntimeError("Some scenario grids did not match historical support table.")

row_audit["increase_within_grid_max_positive_change"] = (
    row_audit["has_positive_historical_ndwi_change"]
    & (
        row_audit["actual_ndwi_increase"]
        <= row_audit["max_positive_annual_ndwi_change"] + 1e-12
    )
)

row_audit["increase_within_grid_q90_positive_change"] = (
    row_audit["has_positive_historical_ndwi_change"]
    & (
        row_audit["actual_ndwi_increase"]
        <= row_audit["q90_positive_annual_ndwi_change"] + 1e-12
    )
)

row_audit["baseline_within_grid_historical_ndwi_range"] = (
    (row_audit["baseline_ndwi"] >= row_audit["hist_ndwi_min"] - 1e-12)
    & (row_audit["baseline_ndwi"] <= row_audit["hist_ndwi_max"] + 1e-12)
)

row_audit["scenario_within_grid_historical_ndwi_range"] = (
    (row_audit["scenario_ndwi"] >= row_audit["hist_ndwi_min"] - 1e-12)
    & (row_audit["scenario_ndwi"] <= row_audit["hist_ndwi_max"] + 1e-12)
)

row_audit["scenario_newly_exits_grid_historical_range"] = (
    row_audit["baseline_within_grid_historical_ndwi_range"]
    & (~row_audit["scenario_within_grid_historical_ndwi_range"])
)

row_audit["baseline_already_outside_grid_historical_range"] = (
    ~row_audit["baseline_within_grid_historical_ndwi_range"]
)

row_audit["increase_to_grid_max_support_ratio"] = np.where(
    row_audit["max_positive_annual_ndwi_change"].notna()
    & (row_audit["max_positive_annual_ndwi_change"] > 0),
    row_audit["actual_ndwi_increase"]
    / row_audit["max_positive_annual_ndwi_change"],
    np.nan,
)

row_audit["grid_level_support_class"] = np.select(
    [
        (
            row_audit["increase_within_grid_q90_positive_change"]
            & row_audit["scenario_within_grid_historical_ndwi_range"]
        ),
        (
            row_audit["increase_within_grid_max_positive_change"]
            & row_audit["scenario_within_grid_historical_ndwi_range"]
        ),
    ],
    [
        "strong_within_grid_support",
        "within_grid_max_support",
    ],
    default="extrapolative_for_selected_grid",
)

row_audit.to_csv(
    ROW_AUDIT_PATH,
    index=False,
)


# =============================================================================
# 4. YEAR / LEVEL SUMMARY
# =============================================================================

section("4. YEAR-LEVEL SUPPORT SUMMARY")

year_level = (
    row_audit
    .groupby(
        [
            YEAR_COL,
            "scenario_level",
        ],
        as_index=False,
    )
    .agg(
        rows=(GRID_COL, "size"),
        requested_ndwi_increase=(
            "requested_ndwi_increase",
            "first",
        ),
        mean_actual_ndwi_increase=(
            "actual_ndwi_increase",
            "mean",
        ),
        grids_with_positive_hist_change=(
            "has_positive_historical_ndwi_change",
            "sum",
        ),
        within_grid_q90_count=(
            "increase_within_grid_q90_positive_change",
            "sum",
        ),
        within_grid_max_count=(
            "increase_within_grid_max_positive_change",
            "sum",
        ),
        baseline_in_hist_range_count=(
            "baseline_within_grid_historical_ndwi_range",
            "sum",
        ),
        scenario_in_hist_range_count=(
            "scenario_within_grid_historical_ndwi_range",
            "sum",
        ),
        newly_exits_hist_range_count=(
            "scenario_newly_exits_grid_historical_range",
            "sum",
        ),
        baseline_already_outside_hist_range_count=(
            "baseline_already_outside_grid_historical_range",
            "sum",
        ),
    )
)

for count_col in [
    "within_grid_q90_count",
    "within_grid_max_count",
    "baseline_in_hist_range_count",
    "scenario_in_hist_range_count",
    "newly_exits_hist_range_count",
    "baseline_already_outside_hist_range_count",
]:
    year_level[
        count_col.replace("_count", "_pct")
    ] = (
        year_level[count_col]
        / year_level["rows"]
        * 100.0
    )

year_level.to_csv(
    YEAR_LEVEL_SUMMARY_PATH,
    index=False,
)

print(
    year_level.to_string(
        index=False
    )
)


# =============================================================================
# 5. OVERALL SUPPORT SUMMARY
# =============================================================================

section("5. OVERALL SUPPORT SUMMARY")

overall = (
    row_audit
    .groupby(
        "scenario_level",
        as_index=False,
    )
    .agg(
        rows=(GRID_COL, "size"),
        requested_ndwi_increase=(
            "requested_ndwi_increase",
            "first",
        ),
        mean_actual_ndwi_increase=(
            "actual_ndwi_increase",
            "mean",
        ),
        within_grid_q90_count=(
            "increase_within_grid_q90_positive_change",
            "sum",
        ),
        within_grid_max_count=(
            "increase_within_grid_max_positive_change",
            "sum",
        ),
        scenario_in_hist_range_count=(
            "scenario_within_grid_historical_ndwi_range",
            "sum",
        ),
        newly_exits_hist_range_count=(
            "scenario_newly_exits_grid_historical_range",
            "sum",
        ),
        baseline_already_outside_hist_range_count=(
            "baseline_already_outside_grid_historical_range",
            "sum",
        ),
    )
)

for count_col in [
    "within_grid_q90_count",
    "within_grid_max_count",
    "scenario_in_hist_range_count",
    "newly_exits_hist_range_count",
    "baseline_already_outside_hist_range_count",
]:
    overall[
        count_col.replace("_count", "_pct")
    ] = (
        overall[count_col]
        / overall["rows"]
        * 100.0
    )

support_class_counts = (
    row_audit
    .pivot_table(
        index="scenario_level",
        columns="grid_level_support_class",
        values=GRID_COL,
        aggfunc="count",
        fill_value=0,
    )
    .reset_index()
)

for col in [
    "strong_within_grid_support",
    "within_grid_max_support",
    "extrapolative_for_selected_grid",
]:
    if col not in support_class_counts.columns:
        support_class_counts[col] = 0

overall = overall.merge(
    support_class_counts,
    on="scenario_level",
    how="left",
    validate="one_to_one",
)

overall["strong_within_grid_support_pct"] = (
    overall["strong_within_grid_support"]
    / overall["rows"]
    * 100.0
)

overall["within_grid_max_support_pct"] = (
    overall["within_grid_max_support"]
    / overall["rows"]
    * 100.0
)

overall["extrapolative_for_selected_grid_pct"] = (
    overall["extrapolative_for_selected_grid"]
    / overall["rows"]
    * 100.0
)

overall.to_csv(
    OVERALL_SUMMARY_PATH,
    index=False,
)

print(
    overall.to_string(
        index=False
    )
)


# =============================================================================
# 6. QA AUDIT
# =============================================================================

section("6. STRICT 29D-QA AUDIT")

scenario_level_table = pd.read_csv(LEVEL_PATH)

scenario_level_table["scenario_level"] = (
    scenario_level_table["scenario_level"]
    .astype(str)
    .str.upper()
)

level_lookup = {
    str(row["scenario_level"]):
        float(row["ndwi_increase_magnitude"])
    for _, row in scenario_level_table.iterrows()
}

pooled_level_errors = []

for level, q in [
    ("LOW", 0.50),
    ("MODERATE", 0.75),
    ("HIGH", 0.90),
]:
    expected = float(
        pooled_positive.quantile(q)
    )

    actual = level_lookup[level]

    pooled_level_errors.append(
        abs(actual - expected)
    )

scenario_level_reproduction_error = max(
    pooled_level_errors
)

audit_rows = [
    {
        "check":
            "final_29d_audit",
        "status":
            "PASS",
        "details":
            "Final Step-29D strict audit verified fully PASS.",
    },
    {
        "check":
            "balanced_historical_panel",
        "status":
            "PASS",
        "details":
            f"grids={EXPECTED_GRIDS}, years=2015-2025",
    },
    {
        "check":
            "adjacent_year_differences_only",
        "status":
            "PASS",
        "details":
            "2016-2025 transitions are all true one-year differences.",
    },
    {
        "check":
            "pooled_scenario_level_reproduction",
        "status":
            (
                "PASS"
                if scenario_level_reproduction_error <= 1e-12
                else "FAIL"
            ),
        "details":
            f"max_error={scenario_level_reproduction_error:.12e}",
    },
    {
        "check":
            "scenario_row_count",
        "status":
            (
                "PASS"
                if len(row_audit) == EXPECTED_SCENARIO_ROWS
                else "FAIL"
            ),
        "details":
            f"actual={len(row_audit)}, expected={EXPECTED_SCENARIO_ROWS}",
    },
    {
        "check":
            "all_scenario_grids_have_historical_support_record",
        "status":
            (
                "PASS"
                if row_audit["hist_ndwi_min"].notna().all()
                else "FAIL"
            ),
        "details":
            f"matched_rows={int(row_audit['hist_ndwi_min'].notna().sum())}",
    },
    {
        "check":
            "support_flags_computed_without_lsts",
        "status":
            "PASS",
        "details":
            (
                "Support classification uses only historical NDWI and "
                "scenario NDWI magnitude/range; no scenario LST response "
                "is used to choose or modify support thresholds."
            ),
    },
    {
        "check":
            "scenario_outputs_modified",
        "status":
            "PASS",
        "details":
            "No. Step-29D outputs are read-only inputs.",
    },
    {
        "check":
            "model_retraining",
        "status":
            "PASS",
        "details":
            "None.",
    },
    {
        "check":
            "causal_claim",
        "status":
            "PASS",
        "details":
            "No. This is support/extrapolation QA only.",
    },
]

qa_audit_df = pd.DataFrame(
    audit_rows
)

qa_audit_df.to_csv(
    QA_AUDIT_PATH,
    index=False,
)

print(
    qa_audit_df.to_string(
        index=False
    )
)

if (
    qa_audit_df["status"]
    != "PASS"
).any():
    raise RuntimeError(
        "One or more 29D-QA audit checks failed."
    )


# =============================================================================
# 7. REPORT + METHODOLOGY
# =============================================================================

section("7. SAVE QA METHODOLOGY + REPORT")

methodology = {
    "step":
        "29D-QA",
    "purpose":
        "Grid-level historical-support audit for finalized Step-29D NDWI scenarios",
    "scenario_outputs_modified":
        False,
    "model_retraining":
        False,
    "support_basis":
        {
            "historical_period":
                "2015-2025",
            "adjacent_annual_changes":
                "2016-2025",
            "positive_changes_only":
                True,
            "grid_level_checks":
                [
                    "scenario increment <= grid historical q90 positive annual change",
                    "scenario increment <= grid historical max positive annual change",
                    "scenario NDWI within grid historical NDWI range",
                    "scenario newly exits grid historical range",
                ],
        },
    "support_classes":
        {
            "strong_within_grid_support":
                (
                    "increment within grid q90 positive-change support AND "
                    "scenario NDWI within grid historical range"
                ),
            "within_grid_max_support":
                (
                    "increment exceeds grid q90 but remains within grid max "
                    "positive-change support AND scenario NDWI within historical range"
                ),
            "extrapolative_for_selected_grid":
                (
                    "increment exceeds grid max positive annual change, grid "
                    "has no positive historical change, or scenario NDWI lies "
                    "outside that grid's historical NDWI range"
                ),
        },
    "interpretation":
        (
            "Extrapolative scenarios are not deleted. They should be shown "
            "with lower-confidence / extrapolation wording for that selected grid."
        ),
}

METHOD_PATH.write_text(
    json.dumps(
        methodology,
        indent=2,
    ),
    encoding="utf-8",
)

report_lines = [
    "STEP 29D-QA - BLUE-SPACE / NDWI GRID-LEVEL SUPPORT AUDIT",
    "=" * 118,
    "",
    "POOLED POSITIVE ANNUAL NDWI-CHANGE SUPPORT",
    "-" * 118,
    json.dumps(
        pooled_support,
        indent=2,
    ),
    "",
    "OVERALL SCENARIO SUPPORT",
    "-" * 118,
    overall.to_string(index=False),
    "",
    "YEAR / LEVEL SUPPORT",
    "-" * 118,
    year_level.to_string(index=False),
    "",
    "INTERPRETATION",
    "-" * 118,
    (
        "This QA does not alter scenario predictions. It identifies where a "
        "selected grid's scenario is supported by that grid's own historical "
        "NDWI-change behavior versus where it is extrapolative."
    ),
    (
        "Frontend can retain all scenarios but should expose lower confidence "
        "or extrapolation status for grid-scenario combinations outside local support."
    ),
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {METHOD_PATH}")
print(f"Saved: {REPORT_PATH}")


# =============================================================================
# 8. HASHES
# =============================================================================

section("8. REPRODUCIBILITY HASHES")

hash_files = [
    BASE_PATH,
    SCENARIO_PATH,
    LEVEL_PATH,
    STEP29D_AUDIT_PATH,
    GRID_SUPPORT_PATH,
    ROW_AUDIT_PATH,
    YEAR_LEVEL_SUMMARY_PATH,
    OVERALL_SUMMARY_PATH,
    QA_AUDIT_PATH,
    METHOD_PATH,
    REPORT_PATH,
]

hash_rows = []

for path in hash_files:
    digest = sha256_file(path)

    hash_rows.append(
        {
            "file":
                str(path),
            "sha256":
                digest,
        }
    )

    print(
        f"{path.name}: {digest}"
    )

pd.DataFrame(
    hash_rows
).to_csv(
    HASH_PATH,
    index=False,
)

print(
    f"Saved: {HASH_PATH}"
)


# =============================================================================
# FINAL
# =============================================================================

section("STEP 29D-QA COMPLETED SUCCESSFULLY")

print(
    "Grid-level historical support audit completed.\n"
    "No Step-29D predictions were changed.\n"
    "No model was retrained.\n"
    "Scenario magnitudes were not tuned using LST responses.\n"
    "Each scenario row now has a grid-level support/extrapolation classification.\n"
    "\nNEXT: review this QA output. If the support pattern is acceptable and "
    "properly labelled, freeze Step 29D and move to Step 30 uncertainty."
)
