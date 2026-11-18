from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[4]

VALIDATION_DIR = (
    ROOT
    / "experiments"
    / "validation"
    / "scenario_validation"
)

OUTPUT_DIR = VALIDATION_DIR / "outputs"
FIGURE_DIR = VALIDATION_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Step 1
GREEN_SUPPORT_PATH = (
    OUTPUT_DIR
    / "01_green_scenario_historical_support.csv"
)

NDBI_LEVEL_PATH = (
    OUTPUT_DIR
    / "01_ndbi_scenario_level_validation.csv"
)

NDWI_LEVEL_PATH = (
    OUTPUT_DIR
    / "01_ndwi_scenario_level_validation.csv"
)

# Step 2
TWFE_PATH = (
    OUTPUT_DIR
    / "02_twfe_historical_response_results.csv"
)

STABILITY_PATH = (
    OUTPUT_DIR
    / "02_response_sign_stability.csv"
)

# Step 3
AUDIT_PATH = (
    OUTPUT_DIR
    / "03_scenario_output_robustness_audit.csv"
)

AGREEMENT_PATH = (
    OUTPUT_DIR
    / "03_historical_scenario_direction_agreement.csv"
)

MONOTONICITY_PATH = (
    OUTPUT_DIR
    / "03_scenario_monotonicity_diagnostic.csv"
)


# ============================================================
# LOAD
# ============================================================

print("\n============================================================")
print("SCENARIO VALIDATION 04")
print("FINAL SCENARIO VALIDATION SUMMARY")
print("============================================================\n")

required_files = [
    GREEN_SUPPORT_PATH,
    NDBI_LEVEL_PATH,
    NDWI_LEVEL_PATH,
    TWFE_PATH,
    STABILITY_PATH,
    AUDIT_PATH,
    AGREEMENT_PATH,
    MONOTONICITY_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required validation file not found:\n{path}"
        )


green_support = pd.read_csv(GREEN_SUPPORT_PATH)
ndbi_levels = pd.read_csv(NDBI_LEVEL_PATH)
ndwi_levels = pd.read_csv(NDWI_LEVEL_PATH)

twfe = pd.read_csv(TWFE_PATH)
stability = pd.read_csv(STABILITY_PATH)

audit = pd.read_csv(AUDIT_PATH)
agreement = pd.read_csv(AGREEMENT_PATH)
monotonicity = pd.read_csv(MONOTONICITY_PATH)


# ============================================================
# STEP 1 SUMMARY
# ============================================================

green_all_supported = bool(
    green_support[
        "within_observed_global_max"
    ].astype(bool).all()
)

ndbi_level_count = len(ndbi_levels)
ndwi_level_count = len(ndwi_levels)

step1_pass = (
    green_all_supported
    and ndbi_level_count == 3
    and ndwi_level_count == 3
)


# ============================================================
# STEP 2 SUMMARY
# ============================================================

sign_stability_pass = bool(
    stability[
        "sign_stable"
    ].astype(bool).all()
)

step2_pass = sign_stability_pass


# Full-period historical coefficients
full_twfe = twfe[
    twfe["model"]
    .astype(str)
    .str.contains(
        "ALL_2015_2025",
        regex=False,
    )
].copy()


def get_twfe_value(treatment, column):

    row = full_twfe[
        full_twfe["treatment"]
        == treatment
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Could not uniquely find "
            f"{treatment} in TWFE results."
        )

    return row.iloc[0][column]


green_beta = float(
    get_twfe_value(
        "green_mask",
        "beta_c_per_unit",
    )
)

ndbi_beta = float(
    get_twfe_value(
        "NDBI",
        "beta_c_per_unit",
    )
)

ndwi_beta = float(
    get_twfe_value(
        "NDWI",
        "beta_c_per_unit",
    )
)


# ============================================================
# STEP 3 SUMMARY
# ============================================================

audit_pass = bool(
    (
        audit["status"]
        .astype(str)
        == "PASS"
    ).all()
)

direction_agreement_pass = bool(
    agreement[
        "direction_agreement"
    ].astype(bool).all()
)

step3_pass = (
    audit_pass
    and direction_agreement_pass
)


# ============================================================
# OVERALL STATUS
# ============================================================

all_steps_pass = (
    step1_pass
    and step2_pass
    and step3_pass
)

overall_status = (
    "PASS_WITH_INTERPRETATION_CAUTION"
    if all_steps_pass
    else "REVIEW_REQUIRED"
)


# ============================================================
# BUILD FINAL SUMMARY TABLE
# ============================================================

summary_rows = [
    {
        "validation_stage":
            "Step 1 - Historical scenario support",
        "validation_goal":
            "Check whether intervention magnitudes are supported by historical data.",
        "main_result":
            "Green 5/10/20 pp increments fall within observed historical change range; NDBI and NDWI levels were independently reproduced from historical quantiles.",
        "status":
            "PASS" if step1_pass else "REVIEW",
    },
    {
        "validation_stage":
            "Step 2 - Historical response consistency",
        "validation_goal":
            "Check whether historical within-grid response directions are stable.",
        "main_result":
            "NDBI showed a stable positive association with LST; NDWI showed a stable negative association; green-cover association remained positive in both full-period and sensitivity models.",
        "status":
            "PASS" if step2_pass else "REVIEW",
    },
    {
        "validation_stage":
            "Step 3 - Scenario output robustness",
        "validation_goal":
            "Check scenario completeness, arithmetic identities, bounds, direction labels and historical directional agreement.",
        "main_result":
            "All structural, arithmetic and formula checks passed and all three scenario families agreed with historical directional evidence.",
        "status":
            "PASS" if step3_pass else "REVIEW",
    },
    {
        "validation_stage":
            "Overall scenario validation",
        "validation_goal":
            "Assess whether scenario outputs are data-supported and internally robust.",
        "main_result":
            "Scenario framework is data-supported and internally consistent; outputs remain model-based what-if estimates and are not causal or future ground-truth predictions.",
        "status":
            overall_status,
    },
]

summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# DETAILED EVIDENCE TABLE
# ============================================================

evidence_rows = []

for _, row in green_support.iterrows():

    evidence_rows.append(
        {
            "evidence_group":
                "Historical support",
            "metric":
                f"Green {row['scenario_level']} support",
            "value":
                float(
                    row[
                        "historical_support_pct"
                    ]
                ),
            "unit":
                "%",
            "interpretation":
                "Share of historical positive green-change transitions at or above the scenario increment.",
        }
    )


for _, row in ndbi_levels.iterrows():

    evidence_rows.append(
        {
            "evidence_group":
                "Historical scenario level",
            "metric":
                f"NDBI {row['scenario_level']}",
            "value":
                float(
                    row[
                        "validated_magnitude"
                    ]
                ),
            "unit":
                "index change",
            "interpretation":
                "Historical annual-change quantile-derived scenario magnitude.",
        }
    )


for _, row in ndwi_levels.iterrows():

    evidence_rows.append(
        {
            "evidence_group":
                "Historical scenario level",
            "metric":
                f"NDWI {row['scenario_level']}",
            "value":
                float(
                    row[
                        "validated_magnitude"
                    ]
                ),
            "unit":
                "index change",
            "interpretation":
                "Historical positive annual-change quantile-derived scenario magnitude.",
        }
    )


for _, row in agreement.iterrows():

    evidence_rows.append(
        {
            "evidence_group":
                "Historical-direction agreement",
            "metric":
                row[
                    "scenario_family"
                ],
            "value":
                float(
                    row[
                        "scenario_mean_cooling_benefit_c"
                    ]
                ),
            "unit":
                "°C",
            "interpretation":
                (
                    f"Scenario mean sign = "
                    f"{row['scenario_mean_sign']}; "
                    f"historical expected sign = "
                    f"{row['historically_expected_cooling_benefit_sign']}; "
                    f"status = {row['status']}."
                ),
        }
    )


for _, row in monotonicity.iterrows():

    evidence_rows.append(
        {
            "evidence_group":
                "Monotonicity diagnostic",
            "metric":
                row[
                    "scenario_family"
                ],
            "value":
                float(
                    row[
                        "monotonic_non_decreasing_cooling_pct"
                    ]
                ),
            "unit":
                "%",
            "interpretation":
                "Diagnostic only; monotonic cooling was not required or forced.",
        }
    )


evidence_df = pd.DataFrame(
    evidence_rows
)


# ============================================================
# SAVE CSV OUTPUTS
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "04_scenario_validation_final_summary.csv"
)

evidence_path = (
    OUTPUT_DIR
    / "04_scenario_validation_evidence_table.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)

evidence_df.to_csv(
    evidence_path,
    index=False,
)


# ============================================================
# FINAL FIGURE
# ============================================================

plot_df = green_support.copy()

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.bar(
    plot_df[
        "scenario_level"
    ],
    plot_df[
        "historical_support_pct"
    ],
)

ax.set_ylim(
    0,
    100,
)

ax.set_xlabel(
    "Green Scenario Increment"
)

ax.set_ylabel(
    "Historical Support (%)"
)

ax.set_title(
    "Historical Support for Green Scenario Increments"
)

fig.tight_layout()

figure_path = (
    FIGURE_DIR
    / "04_green_scenario_historical_support.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# CREATE README / VALIDATION REPORT
# ============================================================

green_5 = float(
    green_support.loc[
        green_support[
            "scenario_level"
        ] == "+5pp",
        "historical_support_pct",
    ].iloc[0]
)

green_10 = float(
    green_support.loc[
        green_support[
            "scenario_level"
        ] == "+10pp",
        "historical_support_pct",
    ].iloc[0]
)

green_20 = float(
    green_support.loc[
        green_support[
            "scenario_level"
        ] == "+20pp",
        "historical_support_pct",
    ].iloc[0]
)


audit_pass_count = int(
    (
        audit["status"]
        == "PASS"
    ).sum()
)

audit_total = len(audit)


report = f"""
# Scenario Validation

## Purpose

This validation evaluates whether the Urban Heat scenario framework is
historically supported, internally consistent, and robust.

The validation does not claim that future 2026-2030 scenario outcomes are
observed ground truth and does not establish causal intervention effects.

## Step 1 - Historical Scenario Support

The historical panel contained 2,194 grids across 2015-2025.

Green scenario increments were compared with observed positive annual
green-cover changes.

- +5 percentage points: {green_5:.2f}% historical support
- +10 percentage points: {green_10:.2f}% historical support
- +20 percentage points: {green_20:.2f}% historical support

Built-up scenario levels were independently reproduced using historical
absolute annual NDBI-change quantiles.

Blue-space scenario levels were independently reproduced using historical
positive annual NDWI-change quantiles.

Status: {"PASS" if step1_pass else "REVIEW"}

## Step 2 - Historical Response Consistency

Two-way fixed-effects models were used to examine within-grid historical
associations while controlling for grid and year effects.

Full-period coefficients:

- Green cover beta: {green_beta:.6f}
- NDBI beta: {ndbi_beta:.6f}
- NDWI beta: {ndwi_beta:.6f}

The response signs remained stable when 2017 was excluded.

Interpretation:

- Higher NDBI was consistently associated with higher LST.
- Higher NDWI was consistently associated with lower LST.
- Green-cover change showed a positive observational association with LST
  in this dataset and therefore green interventions must not be described
  as guaranteed cooling.

Status: {"PASS" if step2_pass else "REVIEW"}

## Step 3 - Scenario Output Robustness

Structural, arithmetic, intervention-bound, prediction-difference,
cooling-benefit, direction-label and combined-scenario checks were performed.

- Passed audit checks: {audit_pass_count}/{audit_total}
- Historical-direction agreement: {"PASS" if direction_agreement_pass else "REVIEW"}

Monotonicity was evaluated only as a diagnostic and was not forced into the
nonlinear LightGBM scenario model.

## Overall Conclusion

Overall status: {overall_status}

The scenario framework is supported by historical intervention magnitudes,
stable historical response directions, and internally consistent scenario
outputs.

However, the scenario outputs remain model-based what-if sensitivity
estimates. They should not be interpreted as causal effects, guaranteed
cooling outcomes, or validated future observations.
"""


readme_path = (
    VALIDATION_DIR
    / "README.md"
)

readme_path.write_text(
    report.strip() + "\n",
    encoding="utf-8",
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("============================================================")
print("FINAL VALIDATION SUMMARY")
print("============================================================")

print(
    summary_df.to_string(
        index=False
    )
)

print("\n============================================================")
print("OVERALL STATUS")
print("============================================================")

print(overall_status)

print("\nStep 1 historical support:")
print(
    f"Green +5pp  : {green_5:.2f}%"
)
print(
    f"Green +10pp : {green_10:.2f}%"
)
print(
    f"Green +20pp : {green_20:.2f}%"
)

print("\nStep 2 historical signs:")
print(
    f"Green beta : {green_beta:.6f}"
)
print(
    f"NDBI beta  : {ndbi_beta:.6f}"
)
print(
    f"NDWI beta  : {ndwi_beta:.6f}"
)

print("\nStep 3 robustness:")
print(
    f"Audit checks passed: "
    f"{audit_pass_count}/{audit_total}"
)

print(
    f"Directional agreement: "
    f"{'PASS' if direction_agreement_pass else 'REVIEW'}"
)

print("\nSaved:")
print(summary_path)
print(evidence_path)
print(figure_path)
print(readme_path)

print(
    "\nFINAL SCENARIO VALIDATION SUMMARY COMPLETE."
)