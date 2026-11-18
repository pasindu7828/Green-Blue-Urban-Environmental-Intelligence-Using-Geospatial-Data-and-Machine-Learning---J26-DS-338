from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[4]

PRIORITY_DIR = ROOT / "outputs" / "priority"
REPORT_DIR = ROOT / "outputs" / "reports"

VALIDATION_DIR = (
    ROOT
    / "experiments"
    / "validation"
    / "priority_validation"
)

OUTPUT_DIR = VALIDATION_DIR / "outputs"
FIGURE_DIR = VALIDATION_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Existing production sensitivity validation
STEP33_SUMMARY_PATH = (
    PRIORITY_DIR
    / "33_priority_sensitivity_summary.csv"
)

STEP33_FAMILY_PATH = (
    PRIORITY_DIR
    / "33_sensitivity_family_summary.csv"
)

STEP33_METRICS_PATH = (
    PRIORITY_DIR
    / "33_sensitivity_scenario_metrics.csv"
)

STEP33_AUDIT_PATH = (
    REPORT_DIR
    / "33_priority_sensitivity_validation_audit.csv"
)


# Priority Validation Step 1
STEP1_PATH = (
    OUTPUT_DIR
    / "01_critic_global_stability_summary.csv"
)

STEP1_CRITERIA_PATH = (
    OUTPUT_DIR
    / "01_critic_weight_stability_summary.csv"
)


# Priority Validation Step 2
STEP2_PATH = (
    OUTPUT_DIR
    / "02_topsis_rank_stability_summary.csv"
)


# Priority Validation Step 3
STEP3_PATH = (
    OUTPUT_DIR
    / "03_criterion_removal_summary.csv"
)

STEP3_REMOVAL_PATH = (
    OUTPUT_DIR
    / "03_leave_one_criterion_out_metrics.csv"
)

STEP3_C6_PATH = (
    OUTPUT_DIR
    / "03_c6_intervention_benefit_sensitivity.csv"
)


# Original CRITIC weights
CRITIC_WEIGHTS_PATH = (
    PRIORITY_DIR
    / "31e_critic_weights.csv"
)


# Final outputs
FINAL_SUMMARY_PATH = (
    OUTPUT_DIR
    / "04_priority_validation_final_summary.csv"
)

EVIDENCE_PATH = (
    OUTPUT_DIR
    / "04_priority_validation_evidence_table.csv"
)

CRITERION_DEPENDENCE_PATH = (
    OUTPUT_DIR
    / "04_priority_criterion_dependence_summary.csv"
)

PAPER_TEXT_PATH = (
    OUTPUT_DIR
    / "04_priority_validation_paper_text.txt"
)

README_PATH = (
    VALIDATION_DIR
    / "README.md"
)

FIGURE_PATH = (
    FIGURE_DIR
    / "04_priority_validation_rank_robustness.png"
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path, label):

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def require_columns(
    df,
    columns,
    label,
):

    missing = [
        col
        for col in columns
        if col not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{label} missing columns: "
            f"{missing}"
        )


def require_audit_pass(
    path,
    label,
):

    require_file(
        path,
        label,
    )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    status_col = next(
        (
            c
            for c in df.columns
            if str(c).strip().lower()
            == "status"
        ),
        None,
    )

    if status_col is None:

        raise RuntimeError(
            f"{label} has no status column."
        )

    status = (
        df[status_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if not status.eq(
        "PASS"
    ).all():

        raise RuntimeError(
            f"{label} is not fully PASS."
        )


def pct(value):

    return (
        100.0
        * float(value)
    )


def fmt4(value):

    return f"{float(value):.4f}"


def fmt2(value):

    return f"{float(value):.2f}"


# ============================================================
# LOAD + VERIFY
# ============================================================

print(
    "\n============================================================"
)

print(
    "PRIORITY VALIDATION 04"
)

print(
    "FINAL ROBUSTNESS + SENSITIVITY CONSOLIDATION"
)

print(
    "============================================================\n"
)


required_files = [
    (
        STEP33_SUMMARY_PATH,
        "Step 33 summary",
    ),
    (
        STEP33_FAMILY_PATH,
        "Step 33 family summary",
    ),
    (
        STEP33_METRICS_PATH,
        "Step 33 scenario metrics",
    ),
    (
        STEP1_PATH,
        "Priority validation Step 1",
    ),
    (
        STEP1_CRITERIA_PATH,
        "Step 1 criterion summary",
    ),
    (
        STEP2_PATH,
        "Priority validation Step 2",
    ),
    (
        STEP3_PATH,
        "Priority validation Step 3",
    ),
    (
        STEP3_REMOVAL_PATH,
        "Step 3 leave-one-out metrics",
    ),
    (
        STEP3_C6_PATH,
        "Step 3 C6 sensitivity",
    ),
    (
        CRITIC_WEIGHTS_PATH,
        "CRITIC weights",
    ),
]


for path, label in required_files:

    require_file(
        path,
        label,
    )


require_audit_pass(
    STEP33_AUDIT_PATH,
    "Step 33 audit",
)


print(
    "PASS - Existing Step 33 sensitivity "
    "audit is fully PASS."
)


step1 = pd.read_csv(
    STEP1_PATH
).iloc[0]

step1_criteria = pd.read_csv(
    STEP1_CRITERIA_PATH
)

step2 = pd.read_csv(
    STEP2_PATH
).iloc[0]

step3 = pd.read_csv(
    STEP3_PATH
).iloc[0]

removal = pd.read_csv(
    STEP3_REMOVAL_PATH
)

c6_sensitivity = pd.read_csv(
    STEP3_C6_PATH
)

step33 = pd.read_csv(
    STEP33_SUMMARY_PATH
).iloc[0]

step33_family = pd.read_csv(
    STEP33_FAMILY_PATH
)

step33_metrics = pd.read_csv(
    STEP33_METRICS_PATH
)

critic_weights = pd.read_csv(
    CRITIC_WEIGHTS_PATH
)


# ============================================================
# VERIFY KEY COLUMNS
# ============================================================

require_columns(
    removal,
    [
        "removed_criterion",
        "removed_dimension",
        "weight_method",
        "spearman_rho",
        "kendall_tau_b",
        "top100_overlap_pct",
        "mean_abs_rank_shift",
    ],
    "Criterion-removal results",
)


require_columns(
    critic_weights,
    [
        "criterion_code",
        "decision_dimension",
        "critic_weight",
    ],
    "CRITIC weights",
)


# ============================================================
# STEP 1 EVIDENCE
# ============================================================

print(
    "\n============================================================"
)

print(
    "1. CRITIC RESAMPLING EVIDENCE"
)

print(
    "============================================================"
)


print(
    f"Bootstrap runs: "
    f"{int(step1['bootstrap_runs'])}"
)

print(
    f"Mean Spearman weight-rank correlation: "
    f"{step1['mean_spearman_weight_rank']:.6f}"
)

print(
    f"Minimum Spearman weight-rank correlation: "
    f"{step1['minimum_spearman_weight_rank']:.6f}"
)

print(
    f"Top-weight criterion retained: "
    f"{step1['top_weight_criterion_retained_pct']:.2f}%"
)


# ============================================================
# STEP 2 EVIDENCE
# ============================================================

print(
    "\n============================================================"
)

print(
    "2. TOPSIS UNDER RESAMPLED CRITIC WEIGHTS"
)

print(
    "============================================================"
)


print(
    f"Spearman minimum: "
    f"{step2['spearman_min']:.6f}"
)

print(
    f"Spearman median: "
    f"{step2['spearman_median']:.6f}"
)

print(
    f"Top-100 overlap minimum: "
    f"{100 * step2['top100_overlap_min']:.2f}%"
)

print(
    f"Top-100 overlap median: "
    f"{100 * step2['top100_overlap_median']:.2f}%"
)


# ============================================================
# STEP 33 EXISTING SENSITIVITY
# ============================================================

print(
    "\n============================================================"
)

print(
    "3. EXISTING STEP-33 WEIGHT SENSITIVITY"
)

print(
    "============================================================"
)


print(
    f"Controlled OAT scenarios: "
    f"{int(step33['controlled_oat_scenarios'])}"
)

print(
    f"Monte Carlo scenarios: "
    f"{int(step33['monte_carlo_scenarios'])}"
)

print(
    f"Controlled minimum Spearman: "
    f"{step33['controlled_spearman_min']:.6f}"
)

print(
    f"Controlled minimum Top-100 overlap: "
    f"{100 * step33['controlled_top100_overlap_min']:.2f}%"
)

print(
    f"Monte Carlo minimum Spearman: "
    f"{step33['mc_spearman_min']:.6f}"
)

print(
    f"Monte Carlo median Spearman: "
    f"{step33['mc_spearman_median']:.6f}"
)

print(
    f"Monte Carlo minimum Top-100 overlap: "
    f"{100 * step33['mc_top100_overlap_min']:.2f}%"
)

print(
    f"Monte Carlo median Top-100 overlap: "
    f"{100 * step33['mc_top100_overlap_median']:.2f}%"
)

print(
    f"Equal-weight benchmark Spearman: "
    f"{step33['equal_weight_spearman']:.6f}"
)

print(
    f"Equal-weight benchmark Top-100 overlap: "
    f"{100 * step33['equal_weight_top100_overlap']:.2f}%"
)


# ============================================================
# STEP 3 C6 EVIDENCE
# ============================================================

print(
    "\n============================================================"
)

print(
    "4. C6 INTERVENTION-BENEFIT DEPENDENCE"
)

print(
    "============================================================"
)


c6_fixed = (
    removal[
        (
            removal[
                "removed_criterion"
            ] == "C6"
        )
        &
        (
            removal[
                "weight_method"
            ]
            == "BASELINE_WEIGHTS_RENORMALIZED"
        )
    ]
    .iloc[0]
)


c6_recomputed = (
    removal[
        (
            removal[
                "removed_criterion"
            ] == "C6"
        )
        &
        (
            removal[
                "weight_method"
            ]
            == "CRITIC_RECOMPUTED_ON_REMAINING_5"
        )
    ]
    .iloc[0]
)


print(
    "C6 removed + original weights renormalized:"
)

print(
    f"  Spearman = "
    f"{c6_fixed['spearman_rho']:.6f}"
)

print(
    f"  Top-100 overlap = "
    f"{c6_fixed['top100_overlap_pct']:.2f}%"
)

print(
    f"  Mean rank shift = "
    f"{c6_fixed['mean_abs_rank_shift']:.2f}"
)


print(
    "\nC6 removed + CRITIC recomputed:"
)

print(
    f"  Spearman = "
    f"{c6_recomputed['spearman_rho']:.6f}"
)

print(
    f"  Top-100 overlap = "
    f"{c6_recomputed['top100_overlap_pct']:.2f}%"
)

print(
    f"  Mean rank shift = "
    f"{c6_recomputed['mean_abs_rank_shift']:.2f}"
)


# ============================================================
# CRITERION DEPENDENCE SUMMARY
# ============================================================

criterion_rows = []


for code in sorted(
    removal[
        "removed_criterion"
    ].unique()
):

    subset = removal[
        removal[
            "removed_criterion"
        ] == code
    ]


    fixed = subset[
        subset[
            "weight_method"
        ]
        == "BASELINE_WEIGHTS_RENORMALIZED"
    ].iloc[0]


    recomputed = subset[
        subset[
            "weight_method"
        ]
        == "CRITIC_RECOMPUTED_ON_REMAINING_5"
    ].iloc[0]


    weight_row = (
        critic_weights[
            critic_weights[
                "criterion_code"
            ].astype(str)
            == str(code)
        ]
        .iloc[0]
    )


    criterion_rows.append(
        {
            "criterion_code":
                code,

            "decision_dimension":
                fixed[
                    "removed_dimension"
                ],

            "baseline_critic_weight":
                float(
                    weight_row[
                        "critic_weight"
                    ]
                ),

            "renormalized_spearman":
                float(
                    fixed[
                        "spearman_rho"
                    ]
                ),

            "renormalized_top100_overlap_pct":
                float(
                    fixed[
                        "top100_overlap_pct"
                    ]
                ),

            "renormalized_mean_abs_rank_shift":
                float(
                    fixed[
                        "mean_abs_rank_shift"
                    ]
                ),

            "recomputed_critic_spearman":
                float(
                    recomputed[
                        "spearman_rho"
                    ]
                ),

            "recomputed_critic_top100_overlap_pct":
                float(
                    recomputed[
                        "top100_overlap_pct"
                    ]
                ),

            "recomputed_critic_mean_abs_rank_shift":
                float(
                    recomputed[
                        "mean_abs_rank_shift"
                    ]
                ),
        }
    )


criterion_dependence = pd.DataFrame(
    criterion_rows
)


criterion_dependence.to_csv(
    CRITERION_DEPENDENCE_PATH,
    index=False,
)


# ============================================================
# EVIDENCE TABLE
# ============================================================

evidence_rows = [
    {
        "validation_stage":
            "CRITIC resampling stability",

        "design":
            (
                f"{int(step1['bootstrap_runs'])} bootstrap "
                "resamples of the 2,194-grid decision matrix"
            ),

        "key_result":
            (
                f"Minimum Spearman weight-rank correlation = "
                f"{step1['minimum_spearman_weight_rank']:.4f}; "
                f"top-weight criterion retained in "
                f"{step1['top_weight_criterion_retained_pct']:.1f}% "
                "of runs"
            ),

        "interpretation":
            (
                "CRITIC weighting structure is highly stable "
                "under grid resampling."
            ),
    },

    {
        "validation_stage":
            "TOPSIS under resampled CRITIC weights",

        "design":
            (
                f"{int(step2['bootstrap_topsis_runs'])} TOPSIS "
                "runs using the bootstrap CRITIC weight sets"
            ),

        "key_result":
            (
                f"Spearman min={step2['spearman_min']:.4f}, "
                f"median={step2['spearman_median']:.4f}; "
                f"Top-100 overlap min="
                f"{100 * step2['top100_overlap_min']:.1f}%"
            ),

        "interpretation":
            (
                "Grid priority ranking is highly stable to "
                "sampling-driven CRITIC weight variation."
            ),
    },

    {
        "validation_stage":
            "Controlled Step-33 weight sensitivity",

        "design":
            (
                f"{int(step33['controlled_oat_scenarios'])} "
                "one-at-a-time weight perturbations "
                "(-20%, -10%, +10%, +20%)"
            ),

        "key_result":
            (
                f"Minimum Spearman="
                f"{step33['controlled_spearman_min']:.4f}; "
                f"minimum Top-100 overlap="
                f"{100 * step33['controlled_top100_overlap_min']:.1f}%"
            ),

        "interpretation":
            (
                "Ranking response to reasonable controlled "
                "weight changes is quantified without changing "
                "the frozen CRITIC-TOPSIS solution."
            ),
    },

    {
        "validation_stage":
            "Step-33 simultaneous weight stress test",

        "design":
            (
                f"{int(step33['monte_carlo_scenarios'])} "
                "simultaneous random weight perturbations, "
                "each baseline weight multiplied within 0.80-1.20"
            ),

        "key_result":
            (
                f"Spearman min={step33['mc_spearman_min']:.4f}, "
                f"median={step33['mc_spearman_median']:.4f}; "
                f"Top-100 overlap min="
                f"{100 * step33['mc_top100_overlap_min']:.1f}%"
            ),

        "interpretation":
            (
                "Provides a broader multi-weight robustness "
                "stress test."
            ),
    },

    {
        "validation_stage":
            "Equal-weight benchmark",

        "design":
            "All six criteria assigned weight 1/6",

        "key_result":
            (
                f"Spearman="
                f"{step33['equal_weight_spearman']:.4f}; "
                f"Top-100 overlap="
                f"{100 * step33['equal_weight_top100_overlap']:.1f}%"
            ),

        "interpretation":
            (
                "Tests how different the objective CRITIC "
                "solution is from a neutral equal-weight benchmark."
            ),
    },

    {
        "validation_stage":
            "C6 intervention-benefit removal",

        "design":
            (
                "Remove modeled intervention benefit entirely; "
                "test both weight renormalization and CRITIC "
                "recalculation on remaining five criteria"
            ),

        "key_result":
            (
                f"Spearman={c6_fixed['spearman_rho']:.4f} "
                f"(renormalized) and "
                f"{c6_recomputed['spearman_rho']:.4f} "
                f"(CRITIC recomputed); "
                f"Top-100 overlap="
                f"{c6_fixed['top100_overlap_pct']:.1f}-"
                f"{c6_recomputed['top100_overlap_pct']:.1f}%"
            ),

        "interpretation":
            (
                "Priority ranking is not materially dependent "
                "on the scenario-derived intervention-benefit "
                "criterion."
            ),
    },
]


evidence_df = pd.DataFrame(
    evidence_rows
)


evidence_df.to_csv(
    EVIDENCE_PATH,
    index=False,
)


# ============================================================
# IDENTIFY MOST INFLUENTIAL REMOVAL
# ============================================================

renormalized_only = (
    criterion_dependence
    .sort_values(
        "renormalized_spearman",
        ascending=True,
    )
    .reset_index(drop=True)
)


most_influential = (
    renormalized_only
    .iloc[0]
)


# ============================================================
# FINAL SUMMARY
# ============================================================

final_summary = pd.DataFrame(
    [
        {
            "validation_scope":
                (
                    "MCDA robustness and rank stability; "
                    "not predictive accuracy"
                ),

            "grid_count":
                int(
                    step2[
                        "grid_count"
                    ]
                ),

            "criteria_count":
                int(
                    step2[
                        "criteria_count"
                    ]
                ),

            "critic_bootstrap_runs":
                int(
                    step1[
                        "bootstrap_runs"
                    ]
                ),

            "topsis_bootstrap_runs":
                int(
                    step2[
                        "bootstrap_topsis_runs"
                    ]
                ),

            "step33_controlled_scenarios":
                int(
                    step33[
                        "controlled_oat_scenarios"
                    ]
                ),

            "step33_monte_carlo_scenarios":
                int(
                    step33[
                        "monte_carlo_scenarios"
                    ]
                ),

            "bootstrap_topsis_spearman_min":
                float(
                    step2[
                        "spearman_min"
                    ]
                ),

            "bootstrap_topsis_spearman_median":
                float(
                    step2[
                        "spearman_median"
                    ]
                ),

            "bootstrap_top100_overlap_min_pct":
                pct(
                    step2[
                        "top100_overlap_min"
                    ]
                ),

            "c6_removed_spearman":
                float(
                    c6_fixed[
                        "spearman_rho"
                    ]
                ),

            "c6_removed_top100_overlap_pct":
                float(
                    c6_fixed[
                        "top100_overlap_pct"
                    ]
                ),

            "most_influential_removed_criterion":
                most_influential[
                    "criterion_code"
                ],

            "most_influential_removed_dimension":
                most_influential[
                    "decision_dimension"
                ],

            "most_influential_removal_spearman":
                float(
                    most_influential[
                        "renormalized_spearman"
                    ]
                ),

            "most_influential_removal_top100_overlap_pct":
                float(
                    most_influential[
                        "renormalized_top100_overlap_pct"
                    ]
                ),

            "overall_interpretation":
                (
                    "The CRITIC-TOPSIS priority ranking is "
                    "highly stable to resampling and reasonable "
                    "weight uncertainty, and is not materially "
                    "dependent on C6 modeled intervention benefit. "
                    "Leave-one-criterion-out analysis shows that "
                    "some decision dimensions, particularly the "
                    "highest-information criterion, materially "
                    "shape the ranking and should be retained and "
                    "reported transparently."
                ),
        }
    ]
)


final_summary.to_csv(
    FINAL_SUMMARY_PATH,
    index=False,
)


# ============================================================
# FIGURE
# ============================================================

plot_labels = [
    "Bootstrap\nTOPSIS worst",
    "Step33 OAT\nworst",
    "Step33 MC\nworst",
    "Equal\nweights",
    "Remove C6",
    f"Remove {most_influential['criterion_code']}",
]


plot_values = [
    float(
        step2[
            "spearman_min"
        ]
    ),

    float(
        step33[
            "controlled_spearman_min"
        ]
    ),

    float(
        step33[
            "mc_spearman_min"
        ]
    ),

    float(
        step33[
            "equal_weight_spearman"
        ]
    ),

    float(
        c6_fixed[
            "spearman_rho"
        ]
    ),

    float(
        most_influential[
            "renormalized_spearman"
        ]
    ),
]


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    plot_labels,
    plot_values,
)

ax.set_ylim(
    0,
    1.05,
)

ax.set_ylabel(
    "Spearman Rank Correlation vs Baseline"
)

ax.set_title(
    "Cooling-Priority Ranking Robustness Across Validation Tests"
)

ax.axhline(
    1.0,
    linewidth=1,
)

fig.tight_layout()


fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# PAPER-READY TEXT
# ============================================================

paper_text = f"""
PRIORITY VALIDATION - PAPER-READY INTERPRETATION

The cooling-priority ranking was evaluated using multiple complementary
robustness and sensitivity analyses rather than predictive accuracy, because
the CRITIC-TOPSIS stage produces a decision ranking and no external
ground-truth priority labels are available.

First, CRITIC weight stability was assessed using
{int(step1['bootstrap_runs'])} bootstrap resamples of the 2,194-grid decision
matrix. The criterion ordering remained stable, with a minimum Spearman
weight-rank correlation of
{step1['minimum_spearman_weight_rank']:.4f} and the highest-weight criterion
retained in {step1['top_weight_criterion_retained_pct']:.1f}% of runs.

Second, TOPSIS was recomputed using all
{int(step2['bootstrap_topsis_runs'])} resampled CRITIC weight sets. Rank
agreement with the baseline was very high
(median Spearman rho = {step2['spearman_median']:.4f};
minimum rho = {step2['spearman_min']:.4f}), while the minimum Top-100 overlap
was {100 * step2['top100_overlap_min']:.1f}%.

The original Step-33 sensitivity analysis provided additional controlled and
simultaneous weight-stress tests. Across
{int(step33['controlled_oat_scenarios'])} controlled one-at-a-time weight
perturbations and {int(step33['monte_carlo_scenarios'])} simultaneous random
weight scenarios, ranking stability was quantified using Spearman and Kendall
correlations, top-k overlap, and absolute rank shifts.

Because the modeled intervention-benefit criterion was derived from what-if
scenario outputs, an additional conservative test removed this criterion
entirely. The resulting ranking remained highly similar to the baseline
(Spearman rho = {c6_fixed['spearman_rho']:.4f}; Top-100 overlap =
{c6_fixed['top100_overlap_pct']:.1f}%). Recomputing CRITIC after removing the
criterion produced similarly high agreement
(rho = {c6_recomputed['spearman_rho']:.4f}; Top-100 overlap =
{c6_recomputed['top100_overlap_pct']:.1f}%).

Leave-one-criterion-out analysis also showed that
{most_influential['criterion_code']}
({most_influential['decision_dimension']}) had the greatest structural
influence on the ranking. Removing it reduced Spearman agreement to
{most_influential['renormalized_spearman']:.4f} and Top-100 overlap to
{most_influential['renormalized_top100_overlap_pct']:.1f}%. This is reported
as criterion dependence rather than an accuracy failure, because MCDA rankings
are expected to respond when an influential decision dimension is completely
removed.

Overall, the evidence indicates that the CRITIC-TOPSIS cooling-priority
ranking is highly robust to sampling variation and reasonable weight
uncertainty, while remaining meaningfully responsive to the decision
dimensions that define the prioritization problem. Priority outputs are
therefore interpreted as robust decision-support rankings rather than
ground-truth or causal labels.
""".strip()


PAPER_TEXT_PATH.write_text(
    paper_text,
    encoding="utf-8",
)


# ============================================================
# README
# ============================================================

readme = f"""# Priority Validation

## Purpose

This folder validates the robustness of the Urban Heat cooling-priority
ranking. The priority stage is a CRITIC-TOPSIS decision-ranking system, so
validation focuses on weight stability, rank stability and sensitivity rather
than predictive accuracy.

## Baseline

- Study area: Kaduwela, Sri Lanka
- Grid cells: 2,194
- Criteria: 6
- Weighting: CRITIC
- Ranking: TOPSIS

## Validation Stages

### Step 1 - CRITIC Weight Resampling Stability

The 2,194-grid decision matrix was bootstrap-resampled
{int(step1['bootstrap_runs'])} times.

- Minimum weight-rank Spearman:
  {step1['minimum_spearman_weight_rank']:.4f}
- Top-weight criterion retained:
  {step1['top_weight_criterion_retained_pct']:.1f}%

Interpretation:
CRITIC weight ordering was highly stable under grid resampling.

### Step 2 - TOPSIS Rank Stability Under Resampled CRITIC Weights

TOPSIS was recomputed
{int(step2['bootstrap_topsis_runs'])} times.

- Minimum Spearman:
  {step2['spearman_min']:.4f}
- Median Spearman:
  {step2['spearman_median']:.4f}
- Minimum Top-100 overlap:
  {100 * step2['top100_overlap_min']:.1f}%
- Median Top-100 overlap:
  {100 * step2['top100_overlap_median']:.1f}%

Interpretation:
The priority ranking remained highly stable to sampling-driven CRITIC
weight variation.

### Existing Step 33 - Weight Sensitivity

Existing production sensitivity analysis includes:

- {int(step33['controlled_oat_scenarios'])} controlled OAT scenarios
- {int(step33['monte_carlo_scenarios'])} simultaneous random-weight scenarios
- equal-weight benchmark

Controlled OAT minimum Spearman:
{step33['controlled_spearman_min']:.4f}

Monte Carlo minimum Spearman:
{step33['mc_spearman_min']:.4f}

Monte Carlo median Spearman:
{step33['mc_spearman_median']:.4f}

### Step 3 - Criterion Removal

C6 Modeled Intervention Benefit was tested conservatively because scenario
validation requires cautious interpretation.

Removing C6:

- Spearman:
  {c6_fixed['spearman_rho']:.4f}
- Top-100 overlap:
  {c6_fixed['top100_overlap_pct']:.1f}%
- Mean absolute rank shift:
  {c6_fixed['mean_abs_rank_shift']:.2f}

Recomputing CRITIC without C6:

- Spearman:
  {c6_recomputed['spearman_rho']:.4f}
- Top-100 overlap:
  {c6_recomputed['top100_overlap_pct']:.1f}%

Therefore, the final priority ranking is not materially dependent on the
scenario-derived intervention-benefit criterion.

The strongest criterion-removal effect occurred for:

- Criterion:
  {most_influential['criterion_code']}
- Dimension:
  {most_influential['decision_dimension']}
- Spearman after removal:
  {most_influential['renormalized_spearman']:.4f}
- Top-100 overlap:
  {most_influential['renormalized_top100_overlap_pct']:.1f}%

This is treated as criterion dependence / decision importance, not as model
accuracy failure.

## Final Interpretation

The cooling-priority ranking is robust to resampling and reasonable weight
uncertainty. It also remains highly stable when the scenario-derived
intervention-benefit criterion is removed. However, removal of certain major
decision dimensions can materially change the ranking, showing that these
criteria genuinely contribute to the prioritization structure.

The priority output should therefore be described as a robust decision-support
ranking, not as ground-truth priority or predictive accuracy.
"""


README_PATH.write_text(
    readme,
    encoding="utf-8",
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print(
    "\n============================================================"
)

print(
    "FINAL PRIORITY VALIDATION SUMMARY"
)

print(
    "============================================================"
)


print(
    final_summary.T.to_string(
        header=False
    )
)


print(
    "\n============================================================"
)

print(
    "CRITERION DEPENDENCE"
)

print(
    "============================================================"
)


print(
    criterion_dependence[
        [
            "criterion_code",
            "decision_dimension",
            "baseline_critic_weight",
            "renormalized_spearman",
            "renormalized_top100_overlap_pct",
            "recomputed_critic_spearman",
            "recomputed_critic_top100_overlap_pct",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nFINAL INTERPRETATION:"
)

print(
    "The CRITIC-TOPSIS priority ranking is "
    "highly stable under resampling and "
    "reasonable weight uncertainty."
)

print(
    "C6 intervention-benefit removal has "
    "little effect on the overall ranking."
)

print(
    f"The strongest leave-one-out dependence "
    f"is associated with "
    f"{most_influential['criterion_code']} - "
    f"{most_influential['decision_dimension']}."
)

print(
    "This should be reported transparently as "
    "criterion influence, not hidden or treated "
    "as predictive-accuracy failure."
)


print(
    "\nSaved:"
)

print(FINAL_SUMMARY_PATH)
print(EVIDENCE_PATH)
print(CRITERION_DEPENDENCE_PATH)
print(PAPER_TEXT_PATH)
print(README_PATH)
print(FIGURE_PATH)


print(
    "\nPRIORITY VALIDATION CONSOLIDATION COMPLETE."
)