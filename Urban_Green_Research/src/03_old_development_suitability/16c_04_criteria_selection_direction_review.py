
"""
STEP 16C-D - CRITERIA SELECTION AND DIRECTION REVIEW

Component:
Urban Green Transition, Urbanization Pressure and
Conservation Intelligence

Purpose:
- Record proposed C1-C5 measurements.
- Review C2 and C3 radius alternatives.
- Document criterion direction and limitations.
- Identify decisions requiring further justification.
- Save a reproducible registry and QA reports.

This is a PROVISIONAL methodology review.

No standardization, AHP weighting, criterion removal,
or final suitability scoring is performed.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data/processed/development_suitability"
    / "raw_criteria"
    / "16b_raw_suitability_criteria_100m.csv"
)

REPORTS = ROOT / "outputs/reports"

STEP16B_QA = (
    REPORTS / "16b_raw_suitability_criteria_qa.csv"
)

PREVIOUS_A = (
    REPORTS / "16c_a_raw_distribution_summary.json"
)

PREVIOUS_B = (
    REPORTS / "16c_b_correlation_summary.json"
)

PREVIOUS_C = (
    REPORTS / "16c_c_radius_sensitivity_summary.json"
)

SENSITIVITY_CSV = (
    REPORTS / "16c_c_radius_sensitivity_comparison.csv"
)

REGISTRY_CSV = (
    REPORTS / "16c_d_provisional_criteria_registry.csv"
)

QA_CSV = (
    REPORTS / "16c_d_criteria_selection_qa.csv"
)

SUMMARY_JSON = (
    REPORTS / "16c_d_criteria_selection_summary.json"
)


# ============================================================
# 2. LOAD PREVIOUS RESEARCH RESULTS
# ============================================================

print("=" * 90)
print("STEP 16C-D - CRITERIA SELECTION AND DIRECTION REVIEW")
print("=" * 90)

required_files = [
    INPUT,
    STEP16B_QA,
    PREVIOUS_A,
    PREVIOUS_B,
    PREVIOUS_C,
    SENSITIVITY_CSV,
]

for path in required_files:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


df = pd.read_csv(INPUT)

qa16b = pd.read_csv(STEP16B_QA)

summary_a = load_json(PREVIOUS_A)
summary_b = load_json(PREVIOUS_B)
summary_c = load_json(PREVIOUS_C)

sensitivity = pd.read_csv(SENSITIVITY_CSV)

input_hash = hashlib.sha256(
    INPUT.read_bytes()
).hexdigest()

print(f"\nGrid cells: {len(df):,}")
print(f"Input file: {INPUT.name}")


# ============================================================
# 3. PROVISIONAL CRITERION REGISTRY
# ============================================================

# IMPORTANT:
# These are working methodological decisions.
# Final suitability directions and thresholds must
# be supported by planning evidence and later review.

records = [
    {
        "criterion": "C1",
        "variable": "c1_major_road_distance_m",
        "role": "PROVISIONAL_CORE",
        "direction": "LOWER_GENERALLY_FAVORED",
        "reason": (
            "Smaller distance can indicate better "
            "higher-order road accessibility."
        ),
        "limitation": (
            "Nearness to a road does not guarantee "
            "direct road access or planning permission."
        ),
        "pending_work": (
            "Justify an accessibility scoring function "
            "and suitable distance thresholds."
        ),
    },

    {
        "criterion": "C2",
        "variable":
            "c2_road_density_250m_km_per_km2",
        "role": "SENSITIVITY_ALTERNATIVE",
        "direction": "NOT_STANDARDIZED",
        "reason": "Measures very local road context.",
        "limitation": (
            "Can be sensitive to local road patterns."
        ),
        "pending_work": (
            "Compare with the proposed 500m measurement."
        ),
    },

    {
        "criterion": "C2",
        "variable":
            "c2_road_density_500m_km_per_km2",
        "role": "PROVISIONAL_CORE",
        "direction": "HIGHER_CAN_INDICATE_ACCESS",
        "reason": (
            "Proposed balance between local road "
            "accessibility and broader network context."
        ),
        "limitation": (
            "Road density is not direct parcel access. "
            "Higher values do not guarantee suitability."
        ),
        "pending_work": (
            "Justify 500m planning scale and "
            "the standardization function."
        ),
    },

    {
        "criterion": "C2",
        "variable":
            "c2_road_density_1000m_km_per_km2",
        "role": "SENSITIVITY_ALTERNATIVE",
        "direction": "NOT_STANDARDIZED",
        "reason": (
            "Measures broader road-network context."
        ),
        "limitation": (
            "May smooth important local differences."
        ),
        "pending_work": (
            "Compare with the proposed 500m measurement."
        ),
    },

    {
        "criterion": "C3",
        "variable":
            "c3_building_footprint_pct_cell",
        "role": "SCREENING_ONLY",
        "direction": "NOT_WEIGHTED",
        "reason": (
            "Describes existing building occupation "
            "inside a grid cell."
        ),
        "limitation": (
            "An unbuilt cell is not necessarily "
            "available or legally suitable for development."
        ),
        "pending_work": (
            "Consider existing occupation during later "
            "planning and screening checks."
        ),
    },

    {
        "criterion": "C3",
        "variable":
            "c3_context_building_pct_250m",
        "role": "SENSITIVITY_ALTERNATIVE",
        "direction": "NOT_STANDARDIZED",
        "reason": (
            "Measures nearby existing development."
        ),
        "limitation": (
            "Captures a more local built context."
        ),
        "pending_work": (
            "Compare with proposed 500m context."
        ),
    },

    {
        "criterion": "C3",
        "variable":
            "c3_context_building_pct_500m",
        "role": "PROVISIONAL_CORE",
        "direction": "UNDECIDED",
        "reason": (
            "Proposed neighbourhood development-context "
            "measurement at a broader local scale."
        ),
        "limitation": (
            "Higher building density can mean more "
            "infrastructure but also development saturation."
        ),
        "pending_work": (
            "Justify whether a non-monotonic suitability "
            "function or contextual interpretation is needed."
        ),
    },

    {
        "criterion": "C4",
        "variable": "c4_slope_mean_deg",
        "role": "PROVISIONAL_CORE",
        "direction": "LOWER_GENERALLY_FAVORED",
        "reason": (
            "Gentler terrain is generally easier "
            "for construction."
        ),
        "limitation": (
            "Planning or engineering slope thresholds "
            "have not yet been selected."
        ),
        "pending_work": (
            "Review defensible slope suitability thresholds "
            "and sensitivity to the 39 terrain proxy cells."
        ),
    },

    {
        "criterion": "C5",
        "variable": "c5_elevation_mean_m",
        "role": "PENDING_JUSTIFICATION",
        "direction": "UNDECIDED",
        "reason": (
            "Elevation describes terrain height."
        ),
        "limitation": (
            "Higher or lower elevation is not automatically "
            "better. Elevation alone is not flood risk."
        ),
        "pending_work": (
            "Determine whether elevation provides an "
            "independent, planning-relevant suitability role."
        ),
    },
]

registry = pd.DataFrame(records)


# ============================================================
# 4. RADIUS SENSITIVITY EVIDENCE
# ============================================================

expected_pairs = {
    "C2_250m_vs_500m",
    "C2_250m_vs_1000m",
    "C2_500m_vs_1000m",
    "C3_250m_vs_500m",
}

comparison = sensitivity[
    [
        "pair",
        "spearman_rho",
        "top10_jaccard",
        "top10_disagreement_cells",
        "top10_disagreement_pct",
    ]
].copy()

comparison = comparison.sort_values("pair")

comparison_evidence = []

for _, row in comparison.iterrows():
    comparison_evidence.append({
        "pair": str(row["pair"]),
        "spearman_rho": float(row["spearman_rho"]),
        "top10_jaccard": float(row["top10_jaccard"]),
        "top10_disagreement_cells": int(
            row["top10_disagreement_cells"]
        ),
        "top10_disagreement_pct": float(
            row["top10_disagreement_pct"]
        ),
    })


# ============================================================
# 5. QUALITY ASSURANCE
# ============================================================

checks = []


def check(name, passed, details):
    checks.append({
        "check": name,
        "passed": bool(passed),
        "details": str(details),
    })


check(
    "Step 16B input previously passed QA",
    not qa16b.empty
    and qa16b.iloc[0]["overall_step16b_status"]
    == "PASS",
    "Step 16B must remain frozen",
)

for stage, previous in [
    ("16C-A", summary_a),
    ("16C-B", summary_b),
    ("16C-C", summary_c),
]:
    check(
        f"Previous {stage} PASS",
        previous.get("status") == "PASS",
        "Previous stage must be completed",
    )

    check(
        f"{stage} input checksum unchanged",
        previous.get("input_sha256") == input_hash,
        "Same frozen Step 16B dataset",
    )

check(
    "9240 unique grid cells",
    len(df) == 9240
    and df["grid_id"].is_unique
    and df["grid_id"].notna().all(),
    f"Actual rows: {len(df)}",
)

check(
    "All nine measurements registered",
    len(registry) == 9
    and registry["variable"].is_unique
    and set(registry["variable"]).issubset(df.columns),
    "One entry for each analyzed raw measurement",
)

check(
    "Registered variables contain valid values",
    bool(
        np.isfinite(
            df[registry["variable"].tolist()]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        ).all()
    ),
    "No invalid numeric measurements",
)

core = registry[
    registry["role"] == "PROVISIONAL_CORE"
]

check(
    "Exactly four provisional core measurements",
    len(core) == 4
    and set(core["criterion"]) == {
        "C1", "C2", "C3", "C4"
    },
    "C5 remains pending",
)

check(
    "One proposed C2 and C3 scale",
    int((core["criterion"] == "C2").sum()) == 1
    and int((core["criterion"] == "C3").sum()) == 1,
    "No double counting of alternative radii",
)

check(
    "C3 direction remains undecided",
    registry.loc[
        registry["variable"]
        == "c3_context_building_pct_500m",
        "direction"
    ].iloc[0] == "UNDECIDED",
    "Prevent untested positive building-density assumption",
)

check(
    "C5 not prematurely weighted",
    registry.loc[
        registry["criterion"] == "C5",
        "role"
    ].iloc[0] == "PENDING_JUSTIFICATION",
    "Elevation needs independent planning justification",
)

check(
    "All four sensitivity comparisons available",
    len(sensitivity) == 4
    and set(sensitivity["pair"]) == expected_pairs,
    "Evidence from Step 16C-C",
)

check(
    "Sensitivity evidence valid",
    bool(
        np.isfinite(
            comparison[
                [
                    "spearman_rho",
                    "top10_jaccard",
                    "top10_disagreement_pct",
                ]
            ].to_numpy(dtype=float)
        ).all()
        and comparison["top10_jaccard"]
        .between(0, 1).all()
        and comparison["top10_disagreement_pct"]
        .between(0, 100).all()
    ),
    "Correlation and spatial-overlap evidence checked",
)

check(
    "No AHP weights or suitability scores",
    not any(
        column in registry.columns
        for column in [
            "ahp_weight",
            "suitability_score",
            "standardized_score",
        ]
    ),
    "This is only a provisional selection review",
)

qa = pd.DataFrame(checks)

qa_pass = bool(qa["passed"].all())

qa_status = "PASS" if qa_pass else "REVIEW_REQUIRED"


# ============================================================
# 6. SAVE REPRODUCIBLE REPORTS
# ============================================================

REPORTS.mkdir(parents=True, exist_ok=True)

qa.to_csv(QA_CSV, index=False)

if not qa_pass:
    print(qa.to_string(index=False))
    raise RuntimeError(
        "Step 16C-D QA failed. Review the QA CSV."
    )

registry.to_csv(
    REGISTRY_CSV,
    index=False,
)

summary = {
    "step": "16C-D",
    "name": "Criteria Selection and Direction Review",
    "qa_status": qa_status,
    "methodology_status": "PROVISIONAL",
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "input_sha256": input_hash,
    "grid_cells": int(len(df)),
    "registered_measurements": len(registry),
    "provisional_core_measurements": (
        core["variable"].tolist()
    ),
    "pending_criterion": [
        "C5 elevation"
    ],
    "sensitivity_evidence": comparison_evidence,
    "important_notes": [
        "500m road density is a proposed, not final, radius.",
        "500m building context is a proposed, not final, radius.",
        "C3 suitability direction is still undecided.",
        "C5 elevation requires further justification.",
        "Existing cell building occupation is screening context.",
        "Alternative radii remain available for sensitivity tests.",
        "ESA WorldCover and planning constraints will be reviewed separately.",
        "No Urbanization Pressure or SHAP outputs used.",
        "No AHP weighting or standardization performed.",
    ],
    "qa_checks": len(qa),
    "qa_passed": int(qa["passed"].sum()),
    "outputs": {
        "registry": REGISTRY_CSV.name,
        "qa": QA_CSV.name,
        "summary": SUMMARY_JSON.name,
    },
}

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 7. TERMINAL REPORT
# ============================================================

print("\n=== PROVISIONAL CRITERIA REGISTRY ===")

print(
    registry[
        [
            "criterion",
            "variable",
            "role",
            "direction",
        ]
    ].to_string(index=False)
)

print("\n=== RADIUS SENSITIVITY EVIDENCE ===")

print(
    comparison.round(4)
    .to_string(index=False)
)

print("\n=== STEP 16C-D QA ===")

for _, row in qa.iterrows():
    outcome = (
        "PASS" if row["passed"] else "FAIL"
    )
    print(f"{row['check']}: {outcome}")

print("\n=== GENERATED REPORTS ===")

for path in [
    REGISTRY_CSV,
    QA_CSV,
    SUMMARY_JSON,
]:
    print(path.relative_to(ROOT))

print("\nOverall QA:", qa_status)
print("Methodology status: PROVISIONAL")
print("No final criterion direction or AHP weights assigned.")
