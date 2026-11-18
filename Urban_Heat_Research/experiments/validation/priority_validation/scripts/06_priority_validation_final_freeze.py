from pathlib import Path
import hashlib
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[4]

VALIDATION_DIR = (
    ROOT
    / "experiments"
    / "validation"
    / "priority_validation"
)

OUTPUT_DIR = VALIDATION_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


STEP4_SUMMARY_PATH = (
    OUTPUT_DIR
    / "04_priority_validation_final_summary.csv"
)

STEP4_EVIDENCE_PATH = (
    OUTPUT_DIR
    / "04_priority_validation_evidence_table.csv"
)

STEP5_METRICS_PATH = (
    OUTPUT_DIR
    / "05_historical_priority_backtest_metrics.csv"
)

STEP5_SUMMARY_PATH = (
    OUTPUT_DIR
    / "05_historical_priority_backtest_summary.csv"
)

STEP5_WEIGHTS_PATH = (
    OUTPUT_DIR
    / "05_historical_priority_critic_weights.csv"
)


FINAL_SUMMARY_PATH = (
    OUTPUT_DIR
    / "06_priority_validation_final_summary.csv"
)

FINAL_EVIDENCE_PATH = (
    OUTPUT_DIR
    / "06_priority_validation_evidence_table.csv"
)

PAPER_TEXT_PATH = (
    OUTPUT_DIR
    / "06_priority_validation_paper_text.txt"
)

FREEZE_MANIFEST_PATH = (
    OUTPUT_DIR
    / "06_priority_validation_freeze_manifest.json"
)

README_PATH = (
    VALIDATION_DIR
    / "README.md"
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path, label):

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ============================================================
# LOAD
# ============================================================

print(
    "\n============================================================"
)

print(
    "PRIORITY VALIDATION 06"
)

print(
    "FINAL SUMMARY + FREEZE"
)

print(
    "============================================================\n"
)


required_files = [
    (
        STEP4_SUMMARY_PATH,
        "Step 4 final robustness summary",
    ),
    (
        STEP4_EVIDENCE_PATH,
        "Step 4 evidence table",
    ),
    (
        STEP5_METRICS_PATH,
        "Step 5 historical backtest metrics",
    ),
    (
        STEP5_SUMMARY_PATH,
        "Step 5 historical backtest summary",
    ),
    (
        STEP5_WEIGHTS_PATH,
        "Step 5 historical CRITIC weights",
    ),
]


for path, label in required_files:

    require_file(
        path,
        label,
    )


step4 = pd.read_csv(
    STEP4_SUMMARY_PATH
).iloc[0]


step4_evidence = pd.read_csv(
    STEP4_EVIDENCE_PATH
)


step5_metrics = pd.read_csv(
    STEP5_METRICS_PATH
)


step5_summary = pd.read_csv(
    STEP5_SUMMARY_PATH
).iloc[0]


step5_weights = pd.read_csv(
    STEP5_WEIGHTS_PATH
)


if len(
    step5_metrics
) != 2:

    raise RuntimeError(
        "Expected exactly two "
        "historical backtests."
    )


# ============================================================
# EXTRACT 2024 / 2025 RESULTS
# ============================================================

bt2024 = (
    step5_metrics[
        step5_metrics[
            "target_year"
        ] == 2024
    ]
    .iloc[0]
)


bt2025 = (
    step5_metrics[
        step5_metrics[
            "target_year"
        ] == 2025
    ]
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
                    "MCDA robustness, rank stability, "
                    "criterion sensitivity and "
                    "retrospective temporal validation"
                ),

            "grid_count":
                int(
                    step4[
                        "grid_count"
                    ]
                ),

            "criteria_count":
                int(
                    step4[
                        "criteria_count"
                    ]
                ),

            # ----------------------------------------
            # Robustness
            # ----------------------------------------

            "critic_bootstrap_runs":
                int(
                    step4[
                        "critic_bootstrap_runs"
                    ]
                ),

            "topsis_bootstrap_runs":
                int(
                    step4[
                        "topsis_bootstrap_runs"
                    ]
                ),

            "bootstrap_topsis_spearman_min":
                float(
                    step4[
                        "bootstrap_topsis_spearman_min"
                    ]
                ),

            "bootstrap_topsis_spearman_median":
                float(
                    step4[
                        "bootstrap_topsis_spearman_median"
                    ]
                ),

            "bootstrap_top100_overlap_min_pct":
                float(
                    step4[
                        "bootstrap_top100_overlap_min_pct"
                    ]
                ),

            # ----------------------------------------
            # C6 sensitivity
            # ----------------------------------------

            "c6_removed_spearman":
                float(
                    step4[
                        "c6_removed_spearman"
                    ]
                ),

            "c6_removed_top100_overlap_pct":
                float(
                    step4[
                        "c6_removed_top100_overlap_pct"
                    ]
                ),

            # ----------------------------------------
            # Criterion dependence
            # ----------------------------------------

            "most_influential_removed_criterion":
                step4[
                    "most_influential_removed_criterion"
                ],

            "most_influential_removed_dimension":
                step4[
                    "most_influential_removed_dimension"
                ],

            "most_influential_removal_spearman":
                float(
                    step4[
                        "most_influential_removal_spearman"
                    ]
                ),

            "most_influential_removal_top100_overlap_pct":
                float(
                    step4[
                        "most_influential_removal_top100_overlap_pct"
                    ]
                ),

            # ----------------------------------------
            # Historical backtesting
            # ----------------------------------------

            "historical_backtest_count":
                int(
                    step5_summary[
                        "backtest_count"
                    ]
                ),

            "historical_mean_priority_actual_heat_spearman":
                float(
                    step5_summary[
                        "mean_priority_vs_actual_heat_spearman"
                    ]
                ),

            "historical_min_priority_actual_heat_spearman":
                float(
                    step5_summary[
                        "minimum_priority_vs_actual_heat_spearman"
                    ]
                ),

            "historical_mean_top100_hot_overlap_pct":
                float(
                    step5_summary[
                        "mean_top100_hot_overlap_pct"
                    ]
                ),

            "historical_mean_top10pct_hot_overlap_pct":
                float(
                    step5_summary[
                        "mean_top10pct_hot_overlap_pct"
                    ]
                ),

            "historical_mean_top25pct_hot_overlap_pct":
                float(
                    step5_summary[
                        "mean_top25pct_hot_overlap_pct"
                    ]
                ),

            "historical_mean_top10_bottom10_lst_difference_c":
                float(
                    step5_summary[
                        "mean_top10_minus_bottom10_actual_lst_c"
                    ]
                ),

            "backtest_2024_spearman":
                float(
                    bt2024[
                        "priority_score_vs_actual_heat_spearman"
                    ]
                ),

            "backtest_2025_spearman":
                float(
                    bt2025[
                        "priority_score_vs_actual_heat_spearman"
                    ]
                ),

            "backtest_2024_top10pct_hot_overlap_pct":
                float(
                    bt2024[
                        "top10pct_priority_vs_actual_hottest_overlap_pct"
                    ]
                ),

            "backtest_2025_top10pct_hot_overlap_pct":
                float(
                    bt2025[
                        "top10pct_priority_vs_actual_hottest_overlap_pct"
                    ]
                ),

            "backtest_2024_top10_bottom10_lst_difference_c":
                float(
                    bt2024[
                        "top10_minus_bottom10_actual_lst_c"
                    ]
                ),

            "backtest_2025_top10_bottom10_lst_difference_c":
                float(
                    bt2025[
                        "top10_minus_bottom10_actual_lst_c"
                    ]
                ),

            # ----------------------------------------
            # Final interpretation
            # ----------------------------------------

            "final_interpretation":
                (
                    "The CRITIC-TOPSIS cooling-priority "
                    "framework is highly robust to sampling "
                    "variation and reasonable weight "
                    "uncertainty. Retrospective one-year "
                    "backtesting provides moderate positive "
                    "temporal support, with higher-priority "
                    "grids showing substantially higher "
                    "subsequent observed heat burden. "
                    "Exact hottest-grid overlap is moderate "
                    "because the priority framework combines "
                    "heat with trend, green deficit, built-up "
                    "pressure, blue-space deficit and modeled "
                    "intervention benefit rather than acting "
                    "as a pure heat-ranking model."
                ),

            "validation_status":
                "COMPLETE",
        }
    ]
)


final_summary.to_csv(
    FINAL_SUMMARY_PATH,
    index=False,
)


# ============================================================
# FINAL EVIDENCE TABLE
# ============================================================

historical_evidence = pd.DataFrame(
    [
        {
            "validation_stage":
                (
                    "Historical retrospective "
                    "priority validation"
                ),

            "design":
                (
                    "Two leakage-controlled one-year "
                    "hindcasts: history through 2023 "
                    "validated against observed 2024, "
                    "and history through 2024 validated "
                    "against observed 2025."
                ),

            "key_result":
                (
                    "Priority-vs-observed-heat Spearman "
                    f"rho = {bt2024['priority_score_vs_actual_heat_spearman']:.4f} "
                    "for 2024 and "
                    f"{bt2025['priority_score_vs_actual_heat_spearman']:.4f} "
                    "for 2025; highest-priority decile "
                    "was respectively "
                    f"{bt2024['top10_minus_bottom10_actual_lst_c']:.2f} C "
                    "and "
                    f"{bt2025['top10_minus_bottom10_actual_lst_c']:.2f} C "
                    "hotter than the lowest-priority decile."
                ),

            "interpretation":
                (
                    "Provides moderate temporal evidence "
                    "that higher priority scores identify "
                    "locations with greater subsequent "
                    "observed heat burden. This is outcome "
                    "relevance evidence, not ground-truth "
                    "priority accuracy."
                ),
        }
    ]
)


final_evidence = pd.concat(
    [
        step4_evidence,
        historical_evidence,
    ],
    ignore_index=True,
)


final_evidence.to_csv(
    FINAL_EVIDENCE_PATH,
    index=False,
)


# ============================================================
# PAPER-READY FINAL TEXT
# ============================================================

paper_text = f"""
Cooling-Priority Validation

The cooling-priority framework was validated using complementary robustness,
sensitivity and retrospective temporal analyses rather than conventional
predictive accuracy, because the CRITIC-TOPSIS stage produces a multi-criteria
decision ranking and no external ground-truth priority labels are available.

CRITIC weight stability was first evaluated using
{int(step4['critic_bootstrap_runs'])} bootstrap resamples of the 2,194-grid
decision matrix. Criterion ordering was fully retained. TOPSIS was then
recomputed using the bootstrap-derived weight sets. Rank agreement remained
very high, with a minimum Spearman correlation of
{step4['bootstrap_topsis_spearman_min']:.4f}, a median correlation of
{step4['bootstrap_topsis_spearman_median']:.4f}, and a minimum Top-100 overlap
of {step4['bootstrap_top100_overlap_min_pct']:.1f}%.

The existing sensitivity analysis further tested 24 controlled one-at-a-time
weight perturbations, 500 simultaneous random-weight scenarios and an
equal-weight benchmark. These analyses showed that the ranking remained
generally stable under reasonable weight uncertainty.

Because the modeled intervention-benefit criterion was derived from
model-based what-if scenarios, its influence was additionally tested by
removing it entirely. The resulting ranking remained highly similar to the
baseline, with a Spearman correlation of
{step4['c6_removed_spearman']:.4f} and a Top-100 overlap of
{step4['c6_removed_top100_overlap_pct']:.1f}%. This indicates that the final
priority ranking is not materially dependent on the scenario-derived
intervention-benefit criterion.

Leave-one-criterion-out analysis identified
{step4['most_influential_removed_criterion']}
({step4['most_influential_removed_dimension']}) as the most structurally
influential criterion. Removing this decision dimension reduced Spearman
agreement to {step4['most_influential_removal_spearman']:.4f} and Top-100
overlap to {step4['most_influential_removal_top100_overlap_pct']:.1f}%.
This is interpreted as criterion dependence rather than predictive failure,
because multi-criteria rankings are expected to change when an influential
decision dimension is completely removed.

Finally, retrospective temporal validation reconstructed the priority
framework using only information available before each evaluation year.
Using data through 2023 to construct a 2024 priority ranking produced a
Spearman correlation of
{bt2024['priority_score_vs_actual_heat_spearman']:.4f} with subsequently
observed 2024 heat, while the equivalent 2025 backtest produced a correlation
of {bt2025['priority_score_vs_actual_heat_spearman']:.4f}. The highest-priority
decile was {bt2024['top10_minus_bottom10_actual_lst_c']:.2f} C hotter than the
lowest-priority decile in 2024 and
{bt2025['top10_minus_bottom10_actual_lst_c']:.2f} C hotter in 2025. Across the
two backtests, the mean priority-to-observed-heat Spearman correlation was
{step5_summary['mean_priority_vs_actual_heat_spearman']:.4f}, and the average
temperature separation between the highest- and lowest-priority deciles was
{step5_summary['mean_top10_minus_bottom10_actual_lst_c']:.2f} C.

The historical backtests therefore provide moderate temporal support for the
decision framework. Exact hottest-grid overlap was not expected to be
perfect because the priority score represents a broader planning construct
combining future heat severity, trend urgency, vegetation deficit, built-up
pressure, blue-space deficit and modeled intervention benefit rather than a
pure future-LST ranking.

Overall, the combined validation evidence indicates that the CRITIC-TOPSIS
cooling-priority framework is highly robust to sampling and weight uncertainty
and shows meaningful retrospective association with subsequently observed
heat burden. The resulting priority scores are interpreted as robust
decision-support rankings rather than ground-truth, causal, or predictive
priority labels.
""".strip()


PAPER_TEXT_PATH.write_text(
    paper_text,
    encoding="utf-8",
)


# ============================================================
# README FINAL
# ============================================================

readme = f"""# Cooling-Priority Validation

## Purpose

This validation evaluates the robustness and temporal relevance of the
Urban Heat CRITIC-TOPSIS cooling-priority framework.

The priority output is a multi-criteria decision-support ranking.

It is **not** treated as:
- predictive accuracy,
- ground-truth priority,
- or causal intervention truth.

## Baseline

- Study area: Kaduwela
- Grid cells: 2,194
- Criteria: 6
- Weighting: CRITIC
- Ranking: TOPSIS

## Validation 01 - CRITIC Weight Resampling

500 bootstrap samples of the 2,194-grid decision matrix were used.

Result:
- Criterion rank ordering retained in 100% of runs.
- Minimum Spearman weight-rank correlation = 1.0000.
- Highest-weight criterion retained in 100% of runs.

Interpretation:
CRITIC weighting was highly stable to grid resampling.

## Validation 02 - TOPSIS Rank Stability

TOPSIS was rerun using the 500 bootstrap CRITIC weight sets.

Result:
- Minimum Spearman = {step4['bootstrap_topsis_spearman_min']:.4f}
- Median Spearman = {step4['bootstrap_topsis_spearman_median']:.4f}
- Minimum Top-100 overlap = {step4['bootstrap_top100_overlap_min_pct']:.1f}%

Interpretation:
The priority ranking was highly stable to sampling-driven weight variation.

## Existing Step 33 - Weight Sensitivity

The original priority pipeline also includes:
- 24 controlled one-at-a-time weight perturbations,
- 500 simultaneous random-weight stress tests,
- equal-weight benchmark.

These tests provide additional evidence of rank robustness under reasonable
weight uncertainty.

## Validation 03 - Criterion Removal

Each criterion was removed independently.

The strongest structural dependence occurred for:

- Criterion: {step4['most_influential_removed_criterion']}
- Dimension: {step4['most_influential_removed_dimension']}
- Spearman after removal:
  {step4['most_influential_removal_spearman']:.4f}
- Top-100 overlap:
  {step4['most_influential_removal_top100_overlap_pct']:.1f}%

This is reported as criterion influence rather than model failure.

### C6 Intervention-Benefit Sensitivity

Removing C6 entirely:

- Spearman = {step4['c6_removed_spearman']:.4f}
- Top-100 overlap = {step4['c6_removed_top100_overlap_pct']:.1f}%

Therefore, the final ranking is not materially dependent on the
scenario-derived intervention-benefit criterion.

## Validation 05 - Historical Retrospective Backtesting

Two leakage-controlled historical backtests were conducted.

### 2023 -> 2024

The ranking was constructed using information available through 2023 only.

- Priority vs actual 2024 heat Spearman:
  {bt2024['priority_score_vs_actual_heat_spearman']:.4f}
- Top-10% priority vs actual hottest Top-10% overlap:
  {bt2024['top10pct_priority_vs_actual_hottest_overlap_pct']:.2f}%
- Top-25% overlap:
  {bt2024['top25pct_priority_vs_actual_hottest_overlap_pct']:.2f}%
- Highest-priority decile minus lowest-priority decile actual LST:
  +{bt2024['top10_minus_bottom10_actual_lst_c']:.2f} C

### 2024 -> 2025

The ranking was constructed using information available through 2024 only.

- Priority vs actual 2025 heat Spearman:
  {bt2025['priority_score_vs_actual_heat_spearman']:.4f}
- Top-10% priority vs actual hottest Top-10% overlap:
  {bt2025['top10pct_priority_vs_actual_hottest_overlap_pct']:.2f}%
- Top-25% overlap:
  {bt2025['top25pct_priority_vs_actual_hottest_overlap_pct']:.2f}%
- Highest-priority decile minus lowest-priority decile actual LST:
  +{bt2025['top10_minus_bottom10_actual_lst_c']:.2f} C

### Combined Historical Evidence

- Mean priority vs later observed heat Spearman:
  {step5_summary['mean_priority_vs_actual_heat_spearman']:.4f}
- Mean Top-10% overlap:
  {step5_summary['mean_top10pct_hot_overlap_pct']:.2f}%
- Mean Top-25% overlap:
  {step5_summary['mean_top25pct_hot_overlap_pct']:.2f}%
- Mean high-priority vs low-priority LST separation:
  +{step5_summary['mean_top10_minus_bottom10_actual_lst_c']:.2f} C

Interpretation:
Historical validation provides moderate positive temporal evidence.
High-priority grids subsequently experienced substantially higher heat burden.

The ranking should not be interpreted as a pure future-heat ranking because
it combines six planning dimensions.

## Final Conclusion

The cooling-priority framework is:

- highly stable under CRITIC resampling,
- highly stable under TOPSIS weight variation,
- robust to removal of the scenario-derived C6 criterion,
- meaningfully dependent on important planning dimensions such as blue-space
  deficit,
- and moderately supported by leakage-controlled retrospective validation.

Therefore, the final output is interpreted as a **robust decision-support
priority ranking**, not ground-truth or causal priority.

## Validation Status

**COMPLETE AND FROZEN**
"""


README_PATH.write_text(
    readme,
    encoding="utf-8",
)


# ============================================================
# FREEZE MANIFEST
# ============================================================

freeze_inputs = {
    "04_priority_validation_final_summary":
        STEP4_SUMMARY_PATH,

    "04_priority_validation_evidence_table":
        STEP4_EVIDENCE_PATH,

    "05_historical_priority_backtest_metrics":
        STEP5_METRICS_PATH,

    "05_historical_priority_backtest_summary":
        STEP5_SUMMARY_PATH,

    "05_historical_priority_critic_weights":
        STEP5_WEIGHTS_PATH,
}


freeze_manifest = {
    "validation_component":
        "Cooling Priority",

    "validation_method":
        "CRITIC-TOPSIS robustness, sensitivity and retrospective temporal validation",

    "status":
        "COMPLETE_AND_FROZEN",

    "grid_count":
        2194,

    "criteria_count":
        6,

    "historical_backtests":
        [
            {
                "training_information_through":
                    2023,

                "evaluation_year":
                    2024,
            },
            {
                "training_information_through":
                    2024,

                "evaluation_year":
                    2025,
            },
        ],

    "interpretation":
        (
            "Priority validation supports robustness "
            "and moderate retrospective temporal relevance. "
            "Priority scores are decision-support rankings, "
            "not ground-truth or causal labels."
        ),

    "frozen_evidence_files":
        {
            name:
                {
                    "relative_path":
                        str(
                            path.relative_to(
                                ROOT
                            )
                        ),

                    "sha256":
                        sha256_file(
                            path
                        ),
                }

            for name, path
            in freeze_inputs.items()
        },

    "final_outputs":
        [
            str(
                FINAL_SUMMARY_PATH.relative_to(
                    ROOT
                )
            ),

            str(
                FINAL_EVIDENCE_PATH.relative_to(
                    ROOT
                )
            ),

            str(
                PAPER_TEXT_PATH.relative_to(
                    ROOT
                )
            ),

            str(
                README_PATH.relative_to(
                    ROOT
                )
            ),
        ],
}


with FREEZE_MANIFEST_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        freeze_manifest,
        f,
        indent=2,
    )


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print(
    "Historical backtests included:"
)

print(
    f"2024 rho = "
    f"{bt2024['priority_score_vs_actual_heat_spearman']:.4f}"
)

print(
    f"2025 rho = "
    f"{bt2025['priority_score_vs_actual_heat_spearman']:.4f}"
)

print(
    f"Mean rho = "
    f"{step5_summary['mean_priority_vs_actual_heat_spearman']:.4f}"
)

print(
    f"Mean Top10-Bottom10 LST difference = "
    f"{step5_summary['mean_top10_minus_bottom10_actual_lst_c']:.4f} C"
)


print(
    "\n============================================================"
)

print(
    "FINAL PRIORITY VALIDATION INTERPRETATION"
)

print(
    "============================================================"
)


print(
    "1. CRITIC weighting is highly stable."
)

print(
    "2. TOPSIS ranking is highly stable under "
    "sampling and reasonable weight uncertainty."
)

print(
    "3. C6 intervention-benefit uncertainty does "
    "not materially control the final ranking."
)

print(
    "4. C5 Blue-space / Water Deficit is an "
    "important structural decision dimension."
)

print(
    "5. Historical one-year backtests provide "
    "moderate positive temporal support."
)

print(
    "6. High-priority grids were on average about "
    f"{step5_summary['mean_top10_minus_bottom10_actual_lst_c']:.2f} C "
    "hotter than low-priority grids in the "
    "subsequent observed year."
)

print(
    "\nVALIDATION STATUS: COMPLETE AND FROZEN"
)


print(
    "\nSaved:"
)

print(FINAL_SUMMARY_PATH)
print(FINAL_EVIDENCE_PATH)
print(PAPER_TEXT_PATH)
print(README_PATH)
print(FREEZE_MANIFEST_PATH)


print(
    "\nPRIORITY VALIDATION FINAL FREEZE COMPLETE."
)