
"""
STEP 16C-C - RADIUS SENSITIVITY AUDIT

Component:
Urban Green Transition, Urbanization Pressure and
Conservation Intelligence

Purpose:
Evaluate spatial-scale sensitivity of C2 and C3
candidate neighbourhood measurements.

Methods:
- Pearson and Spearman agreement
- Absolute percentile-rank differences
- Quintile disagreement
- Upper-decile overlap and Jaccard similarity
- Identification of cells with different upper-decile status

No criterion selection, standardization or AHP weighting.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data/processed/development_suitability"
    / "raw_criteria"
    / "16b_raw_suitability_criteria_100m.csv"
)

REPORTS = ROOT / "outputs/reports"

PREVIOUS = REPORTS / "16c_b_correlation_summary.json"

PREVIOUS_PEARSON = (
    REPORTS / "16c_b_pearson_correlation_matrix.csv"
)

PREVIOUS_SPEARMAN = (
    REPORTS / "16c_b_spearman_correlation_matrix.csv"
)

RESULT_CSV = (
    REPORTS / "16c_c_radius_sensitivity_comparison.csv"
)

CELLS_CSV = (
    REPORTS / "16c_c_radius_sensitive_cells.csv"
)

QA_CSV = (
    REPORTS / "16c_c_radius_sensitivity_qa.csv"
)

SUMMARY_JSON = (
    REPORTS / "16c_c_radius_sensitivity_summary.json"
)


# ============================================================
# 2. ALTERNATIVE PAIRS
# ============================================================

C2_250 = "c2_road_density_250m_km_per_km2"
C2_500 = "c2_road_density_500m_km_per_km2"
C2_1000 = "c2_road_density_1000m_km_per_km2"

C3_250 = "c3_context_building_pct_250m"
C3_500 = "c3_context_building_pct_500m"

PAIRS = [
    ("C2", "250m_vs_500m", C2_250, C2_500),
    ("C2", "250m_vs_1000m", C2_250, C2_1000),
    ("C2", "500m_vs_1000m", C2_500, C2_1000),
    ("C3", "250m_vs_500m", C3_250, C3_500),
]


# ============================================================
# 3. LOAD AND VALIDATE
# ============================================================

print("=" * 95)
print("STEP 16C-C - RADIUS SENSITIVITY AUDIT")
print("=" * 95)

required_files = [
    INPUT,
    PREVIOUS,
    PREVIOUS_PEARSON,
    PREVIOUS_SPEARMAN,
]

for path in required_files:
    if not path.is_file():
        raise FileNotFoundError(path)

df = pd.read_csv(INPUT)

with open(PREVIOUS, encoding="utf-8") as f:
    previous = json.load(f)

input_hash = hashlib.sha256(
    INPUT.read_bytes()
).hexdigest()

pearson_previous = pd.read_csv(
    PREVIOUS_PEARSON, index_col=0
)

spearman_previous = pd.read_csv(
    PREVIOUS_SPEARMAN, index_col=0
)

variables = sorted(
    set(
        variable
        for _, _, a, b in PAIRS
        for variable in (a, b)
    )
)

missing = [
    column for column in ["grid_id"] + variables
    if column not in df.columns
]

if missing:
    raise RuntimeError(
        "Missing columns: " + ", ".join(missing)
    )

for column in variables:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

qa_records = []


def qa(name, passed, details):
    qa_records.append({
        "check": name,
        "passed": bool(passed),
        "details": str(details),
    })


qa(
    "9240 unique grid cells",
    len(df) == 9240
    and df["grid_id"].is_unique
    and df["grid_id"].notna().all(),
    f"Actual rows: {len(df)}",
)

qa(
    "Previous Step 16C-B PASS",
    previous.get("status") == "PASS",
    "Correlation audit accepted",
)

qa(
    "Frozen input SHA256 unchanged",
    previous.get("input_sha256") == input_hash,
    "Input consistency with Step 16C-B",
)

qa(
    "Candidate measurements finite",
    bool(
        np.isfinite(
            df[variables].to_numpy(dtype=float)
        ).all()
    ),
    "No missing or infinite measurements",
)

qa(
    "Candidate measurements nonconstant",
    bool(
        (df[variables].nunique() > 1).all()
    ),
    "All candidate radii must vary",
)

REPORTS.mkdir(parents=True, exist_ok=True)

if not all(record["passed"] for record in qa_records):
    pd.DataFrame(qa_records).to_csv(
        QA_CSV, index=False
    )
    raise RuntimeError(
        "Input QA failed. Review the QA report."
    )


# ============================================================
# 4. SCALE SENSITIVITY COMPARISON
# ============================================================

results = []
sensitive_cells = []
correlation_reproduced = True

for criterion, label, column_a, column_b in PAIRS:

    a = df[column_a]
    b = df[column_b]

    pearson_r = float(a.corr(b, method="pearson"))
    spearman_rho = float(a.corr(b, method="spearman"))

    previous_r = float(
        pearson_previous.loc[column_a, column_b]
    )
    previous_rho = float(
        spearman_previous.loc[column_a, column_b]
    )

    correlation_reproduced &= bool(
        np.isclose(
            pearson_r, previous_r, atol=1e-10
        )
        and np.isclose(
            spearman_rho, previous_rho, atol=1e-10
        )
    )

    # Percentile ranks use average ranks for tied values.
    rank_a = a.rank(method="average", pct=True)
    rank_b = b.rank(method="average", pct=True)

    rank_difference = (rank_a - rank_b).abs()

    # Five equally spaced rank classes.
    # Ties may cause unequal class sizes.
    quintile_a = np.ceil(rank_a * 5).clip(1, 5)
    quintile_b = np.ceil(rank_b * 5).clip(1, 5)

    quintile_changed = quintile_a != quintile_b

    # Upper decile is defined using the observed
    # 90th percentile; ties are included.
    threshold_a = float(a.quantile(0.90))
    threshold_b = float(b.quantile(0.90))

    top_a = a >= threshold_a
    top_b = b >= threshold_b

    intersection = int((top_a & top_b).sum())
    union = int((top_a | top_b).sum())

    jaccard = (
        intersection / union
        if union > 0
        else np.nan
    )

    disagreement = top_a != top_b

    pair_name = f"{criterion}_{label}"

    results.append({
        "criterion": criterion,
        "pair": pair_name,
        "variable_a": column_a,
        "variable_b": column_b,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "mean_abs_percentile_rank_difference":
            float(rank_difference.mean()),
        "p95_abs_percentile_rank_difference":
            float(rank_difference.quantile(0.95)),
        "quintile_changed_cells":
            int(quintile_changed.sum()),
        "quintile_changed_pct":
            float(quintile_changed.mean() * 100),
        "top10_threshold_a": threshold_a,
        "top10_threshold_b": threshold_b,
        "top10_count_a": int(top_a.sum()),
        "top10_count_b": int(top_b.sum()),
        "top10_intersection": intersection,
        "top10_union": union,
        "top10_jaccard": float(jaccard),
        "top10_disagreement_cells":
            int(disagreement.sum()),
        "top10_disagreement_pct":
            float(disagreement.mean() * 100),
        "decision": "REVIEW_ONLY",
    })

    for index in np.flatnonzero(
        disagreement.to_numpy()
    ):

        sensitive_cells.append({
            "grid_id": df.iloc[index]["grid_id"],
            "criterion": criterion,
            "pair": pair_name,
            "variable_a": column_a,
            "variable_b": column_b,
            "value_a": float(a.iloc[index]),
            "value_b": float(b.iloc[index]),
            "percentile_rank_a":
                float(rank_a.iloc[index]),
            "percentile_rank_b":
                float(rank_b.iloc[index]),
            "top_decile_a": bool(
                top_a.iloc[index]
            ),
            "top_decile_b": bool(
                top_b.iloc[index]
            ),
        })

results_df = pd.DataFrame(results)

sensitive_df = pd.DataFrame(sensitive_cells)


# ============================================================
# 5. FINAL QA
# ============================================================

qa(
    "Four alternative-radius comparisons",
    len(results_df) == 4,
    f"Actual: {len(results_df)}",
)

qa(
    "Pearson and Spearman results reproduced",
    correlation_reproduced,
    "Must agree with frozen Step 16C-B matrices",
)

qa(
    "All sensitivity metrics finite",
    bool(
        np.isfinite(
            results_df[
                [
                    "pearson_r",
                    "spearman_rho",
                    "mean_abs_percentile_rank_difference",
                    "p95_abs_percentile_rank_difference",
                    "quintile_changed_pct",
                    "top10_jaccard",
                    "top10_disagreement_pct",
                ]
            ].to_numpy(dtype=float)
        ).all()
    ),
    "No undefined comparison results",
)

qa(
    "Jaccard similarity within 0-1",
    bool(
        results_df["top10_jaccard"]
        .between(0, 1).all()
    ),
    "Valid intersection-over-union values",
)

qa(
    "Sensitive-cell totals consistent",
    len(sensitive_df)
    == int(
        results_df[
            "top10_disagreement_cells"
        ].sum()
    ),
    "Detailed records match pairwise counts",
)

qa(
    "No criterion automatically selected",
    bool(
        results_df["decision"].eq(
            "REVIEW_ONLY"
        ).all()
    ),
    "Selection requires methodological justification",
)

qa_df = pd.DataFrame(qa_records)

overall_pass = bool(qa_df["passed"].all())
status = "PASS" if overall_pass else "REVIEW_REQUIRED"

qa_df.to_csv(QA_CSV, index=False)

if not overall_pass:
    raise RuntimeError(
        "Step 16C-C QA failed. Inspect the QA report."
    )


# ============================================================
# 6. SAVE OUTPUTS
# ============================================================

results_df.to_csv(
    RESULT_CSV, index=False
)

sensitive_df.to_csv(
    CELLS_CSV, index=False
)

summary = {
    "step": "16C-C",
    "name": "Alternative Radius Sensitivity Audit",
    "status": status,
    "timestamp_utc":
        datetime.now(timezone.utc).isoformat(),
    "input_sha256": input_hash,
    "grid_cells": int(len(df)),
    "comparisons": len(results_df),
    "qa_checks": len(qa_df),
    "qa_passed": int(qa_df["passed"].sum()),
    "methods": [
        "Pearson correlation",
        "Spearman correlation",
        "Average-tie percentile ranks",
        "Rank-based quintile disagreement",
        "90th-percentile upper-tail overlap",
        "Jaccard similarity",
    ],
    "interpretation_notes": [
        "Upper deciles include all values tied at the threshold.",
        "Ties can make quintile class sizes unequal.",
        "High correlation does not guarantee identical high-value cells.",
        "Spatial autocorrelation limits independent-sample interpretation.",
        "No criterion selected, removed or standardized.",
        "No AHP weights or suitability scores calculated.",
    ],
    "outputs": {
        "comparison": RESULT_CSV.name,
        "sensitive_cells": CELLS_CSV.name,
        "qa": QA_CSV.name,
        "summary": SUMMARY_JSON.name,
    },
}

with open(
    SUMMARY_JSON, "w", encoding="utf-8"
) as file:
    json.dump(
        summary, file, indent=2, ensure_ascii=False
    )


# ============================================================
# 7. TERMINAL REPORT
# ============================================================

print("\n=== RADIUS SENSITIVITY RESULTS ===")

display_columns = [
    "pair",
    "pearson_r",
    "spearman_rho",
    "mean_abs_percentile_rank_difference",
    "quintile_changed_pct",
    "top10_jaccard",
    "top10_disagreement_cells",
    "top10_disagreement_pct",
]

print(
    results_df[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\n=== STEP 16C-C QA ===")

for _, row in qa_df.iterrows():
    outcome = "PASS" if row["passed"] else "FAIL"
    print(f"{row['check']}: {outcome}")

print("\n=== GENERATED REPORTS ===")

for path in [
    RESULT_CSV,
    CELLS_CSV,
    QA_CSV,
    SUMMARY_JSON,
]:
    print(path.relative_to(ROOT))

print("\nOverall Step 16C-C status:", status)
print("No criterion selection or weighting performed.")
